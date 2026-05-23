#!/usr/bin/env python3
"""
9pfuse — 9P2000 FUSE client with Tauth support.

Mounts a remote 9P2000 server as a local filesystem via FUSE.
Supports token-based authentication via the Tauth/Rauth mechanism.

Usage:
    python -m ninepfuse localhost:5642 /mnt/point
    python -m ninepfuse localhost:5642 /mnt/point --auth-token mytoken
    python -m ninepfuse localhost:5642 /mnt/point --auth-token mytoken --user myname

Address formats:
    host:port               TCP (recommended)
    tcp!host!port           TCP (Plan 9 style, quote in bash: 'tcp!host!port')
    unix:/path/to/sock      Unix socket
    unix!/path/to/sock      Unix socket (Plan 9 style)

Examples:
    # No auth (backward compatible)
    python -m ninepfuse localhost:5642 /n/mux

    # With auth token
    python -m ninepfuse localhost:5642 /n/mux --auth-token secret123

    # With auth token from environment
    NINEPFUSE_TOKEN=secret123 python -m ninepfuse localhost:5642 /n/mux

    # Debug mode
    python -m ninepfuse localhost:5642 /n/mux -d
"""

import argparse
import asyncio
import errno
import logging
import os
import signal
import stat
import struct
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyfuse3
import trio

logger = logging.getLogger("ninepfuse")


# ═══════════════════════════════════════════════════════════════════
#  9P2000 Wire Protocol
# ═══════════════════════════════════════════════════════════════════
#
# Minimal 9P2000 wire implementation — just enough for a FUSE client.
# Mirrors the subset used by riomux/wire.py but is self-contained
# so the client can be deployed independently.

# Message types
TVERSION = 100; RVERSION = 101
TAUTH    = 102; RAUTH    = 103
TATTACH  = 104; RATTACH  = 105
RERROR   = 107
TFLUSH   = 108; RFLUSH   = 109
TWALK    = 110; RWALK    = 111
TOPEN    = 112; ROPEN    = 113
TCREATE  = 114; RCREATE  = 115
TREAD    = 116; RREAD    = 117
TWRITE   = 118; RWRITE   = 119
TCLUNK   = 120; RCLUNK   = 121
TREMOVE  = 122; RREMOVE  = 123
TSTAT    = 124; RSTAT    = 125
TWSTAT   = 126; RWSTAT   = 127

NOTAG = 0xFFFF
NOFID = 0xFFFFFFFF

# Qid types
QTDIR  = 0x80
QTAUTH = 0x08
QTFILE = 0x00

# Open modes
OREAD  = 0
OWRITE = 1
ORDWR  = 2
OTRUNC = 0x10

# Dir mode bit
DMDIR = 0x80000000

MSG_NAMES = {
    100: "Tversion", 101: "Rversion", 102: "Tauth", 103: "Rauth",
    104: "Tattach", 105: "Rattach", 107: "Rerror",
    108: "Tflush", 109: "Rflush", 110: "Twalk", 111: "Rwalk",
    112: "Topen", 113: "Ropen", 114: "Tcreate", 115: "Rcreate",
    116: "Tread", 117: "Rread", 118: "Twrite", 119: "Rwrite",
    120: "Tclunk", 121: "Rclunk", 122: "Tremove", 123: "Rremove",
    124: "Tstat", 125: "Rstat", 126: "Twstat", 127: "Rwstat",
}


def _msg_name(mtype: int) -> str:
    return MSG_NAMES.get(mtype, f"?{mtype}")


def _pack_string(s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<H', len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    slen = struct.unpack_from('<H', data, offset)[0]
    s = data[offset + 2:offset + 2 + slen].decode('utf-8')
    return s, offset + 2 + slen


# ═══════════════════════════════════════════════════════════════════
#  9P Stat parsing
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Stat9P:
    """Parsed 9P stat entry."""
    type: int = 0
    dev: int = 0
    qid_type: int = 0
    qid_vers: int = 0
    qid_path: int = 0
    mode: int = 0
    atime: int = 0
    mtime: int = 0
    length: int = 0
    name: str = ""
    uid: str = ""
    gid: str = ""
    muid: str = ""

    @property
    def is_dir(self) -> bool:
        return bool(self.qid_type & QTDIR)


def parse_stat(data: bytes, offset: int) -> Tuple[Stat9P, int]:
    """
    Parse one stat entry from data at offset.
    Returns (Stat9P, new_offset).
    """
    if offset + 2 > len(data):
        raise ValueError("Not enough data for stat size")

    stat_size = struct.unpack_from('<H', data, offset)[0]
    start = offset + 2
    end = start + stat_size

    if end > len(data):
        raise ValueError(f"Stat entry truncated: need {end}, have {len(data)}")

    s = Stat9P()
    p = start

    s.type, s.dev = struct.unpack_from('<HI', data, p); p += 6
    s.qid_type = data[p]; p += 1
    s.qid_vers = struct.unpack_from('<I', data, p)[0]; p += 4
    s.qid_path = struct.unpack_from('<Q', data, p)[0]; p += 8
    s.mode = struct.unpack_from('<I', data, p)[0]; p += 4
    s.atime = struct.unpack_from('<I', data, p)[0]; p += 4
    s.mtime = struct.unpack_from('<I', data, p)[0]; p += 4
    s.length = struct.unpack_from('<Q', data, p)[0]; p += 8
    s.name, p = _unpack_string(data, p)
    s.uid, p = _unpack_string(data, p)
    s.gid, p = _unpack_string(data, p)
    s.muid, p = _unpack_string(data, p)

    return s, end


def parse_dir_data(data: bytes) -> List[Stat9P]:
    """Parse directory read data into a list of stat entries."""
    entries = []
    offset = 0
    while offset < len(data):
        try:
            s, offset = parse_stat(data, offset)
            entries.append(s)
        except ValueError:
            break
    return entries


# ═══════════════════════════════════════════════════════════════════
#  9P Connection
# ═══════════════════════════════════════════════════════════════════

class NineP:
    """
    Async 9P2000 client connection.

    Handles version negotiation, optional Tauth, attach, and all
    file operations. Multiplexes concurrent requests via tags.
    """

    def __init__(self, host: str, port: int, unix_path: str = None):
        self.host = host
        self.port = port
        self.unix_path = unix_path

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

        self._msize = 65536
        self._next_tag = 1
        self._next_fid = 10
        self._root_fid: Optional[int] = None

        # tag → Future for pending requests
        self._pending: Dict[int, asyncio.Future] = {}

        # reader task
        self._reader_task: Optional[asyncio.Task] = None

        # write lock
        self._write_lock = asyncio.Lock()
        self._closed = False

    @property
    def msize(self) -> int:
        return self._msize

    @property
    def root_fid(self) -> int:
        return self._root_fid

    def _alloc_tag(self) -> int:
        tag = self._next_tag
        self._next_tag += 1
        if self._next_tag >= 0xFFFE:
            self._next_tag = 1
        return tag

    def _alloc_fid(self) -> int:
        fid = self._next_fid
        self._next_fid += 1
        return fid

    # ── Connect / disconnect ────────────────────────────────

    async def connect(self):
        """Establish TCP or Unix connection."""
        if self.unix_path:
            self._reader, self._writer = await asyncio.open_unix_connection(
                self.unix_path
            )
        else:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=10.0
            )
        self._connected = True
        self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self):
        """Close the connection."""
        self._closed = True
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False

    # ── Low-level I/O ───────────────────────────────────────

    async def _send(self, data: bytes):
        async with self._write_lock:
            self._writer.write(data)
            await self._writer.drain()

    async def _transact(self, data: bytes, tag: int, timeout: float = 30.0) -> bytes:
        """Send a message and wait for the response with matching tag."""
        future = asyncio.get_event_loop().create_future()
        self._pending[tag] = future
        await self._send(data)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(tag, None)
            raise TimeoutError(f"9P request timed out (tag={tag})")

    async def _read_loop(self):
        """Read responses and dispatch to pending futures."""
        buf = b''
        while not self._closed and self._reader:
            try:
                chunk = await self._reader.read(65536)
                if not chunk:
                    break
                buf += chunk
            except (ConnectionError, asyncio.CancelledError):
                break

            while len(buf) >= 4:
                size = struct.unpack_from('<I', buf, 0)[0]
                if len(buf) < size:
                    break
                msg = buf[:size]
                buf = buf[size:]

                _, mtype, tag = struct.unpack_from('<IBH', msg, 0)
                logger.debug(f"← {_msg_name(mtype)} tag={tag} size={size}")

                future = self._pending.pop(tag, None)
                if future and not future.done():
                    future.set_result(msg)
                elif future is None:
                    logger.warning(f"No pending request for tag {tag}")

        # Fail any remaining futures
        for tag, future in self._pending.items():
            if not future.done():
                future.set_exception(ConnectionError("Connection closed"))
        self._pending.clear()
        self._connected = False

    def _check_rerror(self, resp: bytes) -> Optional[str]:
        """If resp is Rerror, return the error string. Otherwise None."""
        mtype = resp[4]
        if mtype == RERROR:
            elen = struct.unpack_from('<H', resp, 7)[0]
            return resp[9:9 + elen].decode('utf-8')
        return None

    def _raise_if_error(self, resp: bytes, context: str = ""):
        """Raise an OSError if resp is Rerror."""
        err = self._check_rerror(resp)
        if err:
            # Map common 9P errors to errno
            eno = errno.EIO
            lower = err.lower()
            if "not found" in lower or "does not exist" in lower or "no such" in lower:
                eno = errno.ENOENT
            elif "permission" in lower or "denied" in lower:
                eno = errno.EACCES
            elif "not a directory" in lower:
                eno = errno.ENOTDIR
            elif "is a directory" in lower:
                eno = errno.EISDIR
            elif "exists" in lower:
                eno = errno.EEXIST
            elif "not empty" in lower:
                eno = errno.ENOTEMPTY
            elif "authentication" in lower:
                eno = errno.EACCES
            raise OSError(eno, f"{context}: {err}" if context else err)

    # ── 9P Operations ───────────────────────────────────────

    async def version(self) -> int:
        """Negotiate version. Returns msize."""
        tag = NOTAG
        body = bytearray()
        body += struct.pack('<IBH', 0, TVERSION, tag)
        body += struct.pack('<I', self._msize)
        body += _pack_string("9P2000")
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, "version")

        self._msize = struct.unpack_from('<I', resp, 7)[0]
        return self._msize

    async def auth(self, uname: str, aname: str, token: str) -> int:
        """
        Perform Tauth + write token + read result.

        Returns the afid that can be passed to attach().
        Raises OSError if auth fails or isn't required.
        """
        afid = self._alloc_fid()
        tag = self._alloc_tag()

        # Tauth: afid[4] uname[s] aname[s]
        body = bytearray()
        body += struct.pack('<IBH', 0, TAUTH, tag)
        body += struct.pack('<I', afid)
        body += _pack_string(uname)
        body += _pack_string(aname)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)

        # If server says "not required", that's fine — return NOFID
        err = self._check_rerror(resp)
        if err:
            if "not required" in err.lower():
                logger.info("Server says auth not required")
                return NOFID
            raise OSError(errno.EACCES, f"Tauth failed: {err}")

        logger.info(f"Auth fid {afid} created, writing token...")

        # Open the auth fid for read+write
        # (9P spec says auth fids are implicitly open, but our mux
        # handles Topen on auth fids explicitly for compatibility)
        await self._open_fid(afid, ORDWR)

        # Write token
        token_bytes = token.encode('utf-8')
        await self._write_fid(afid, 0, token_bytes)

        # Read result
        result = await self._read_fid(afid, 0, 256)
        result_str = result.decode('utf-8').strip()

        if result_str == "ok":
            logger.info("Authentication succeeded")
            return afid
        else:
            raise OSError(errno.EACCES, f"Authentication failed: {result_str}")

    async def attach(self, uname: str, aname: str = "",
                     afid: int = NOFID) -> int:
        """Attach and return the root fid."""
        self._root_fid = self._alloc_fid()
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TATTACH, tag)
        body += struct.pack('<I', self._root_fid)
        body += struct.pack('<I', afid)
        body += _pack_string(uname)
        body += _pack_string(aname)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, "attach")

        logger.info(f"Attached, root_fid={self._root_fid}")
        return self._root_fid

    async def walk(self, fid: int, newfid: int,
                   names: List[str]) -> List[Tuple[int, int, int]]:
        """
        Walk from fid to newfid along names.
        Returns list of (qid_type, qid_vers, qid_path) tuples.
        """
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TWALK, tag)
        body += struct.pack('<I', fid)
        body += struct.pack('<I', newfid)
        body += struct.pack('<H', len(names))
        for n in names:
            body += _pack_string(n)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, f"walk {'/'.join(names)}")

        nwqid = struct.unpack_from('<H', resp, 7)[0]
        qids = []
        p = 9
        for _ in range(nwqid):
            qt = resp[p]; p += 1
            qv = struct.unpack_from('<I', resp, p)[0]; p += 4
            qp = struct.unpack_from('<Q', resp, p)[0]; p += 8
            qids.append((qt, qv, qp))

        return qids

    async def walk_path(self, path: str) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        Walk a full path from root. Returns (fid, qids).
        Caller must clunk the fid when done.
        """
        names = [n for n in path.split('/') if n]
        newfid = self._alloc_fid()
        qids = await self.walk(self._root_fid, newfid, names)
        if len(qids) != len(names):
            await self.clunk(newfid)
            raise OSError(errno.ENOENT, f"Partial walk: {path}")
        return newfid, qids

    async def clone_fid(self, fid: int) -> int:
        """Clone a fid (zero-length walk)."""
        newfid = self._alloc_fid()
        await self.walk(fid, newfid, [])
        return newfid

    async def open(self, fid: int, mode: int) -> Tuple[int, int, int, int]:
        """
        Open a fid. Returns (qid_type, qid_vers, qid_path, iounit).
        """
        return await self._open_fid(fid, mode)

    async def _open_fid(self, fid: int, mode: int) -> Tuple[int, int, int, int]:
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TOPEN, tag)
        body += struct.pack('<I', fid)
        body += struct.pack('<B', mode)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, "open")

        qt = resp[7]
        qv = struct.unpack_from('<I', resp, 8)[0]
        qp = struct.unpack_from('<Q', resp, 12)[0]
        iounit = struct.unpack_from('<I', resp, 20)[0]

        return qt, qv, qp, iounit

    async def read(self, fid: int, offset: int, count: int) -> bytes:
        """Read from an open fid."""
        return await self._read_fid(fid, offset, count)

    async def _read_fid(self, fid: int, offset: int, count: int) -> bytes:
        # Clamp to msize - header overhead
        max_data = self._msize - 24
        count = min(count, max_data)

        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TREAD, tag)
        body += struct.pack('<I', fid)
        body += struct.pack('<Q', offset)
        body += struct.pack('<I', count)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag, timeout=120.0)
        self._raise_if_error(resp, "read")

        data_count = struct.unpack_from('<I', resp, 7)[0]
        return resp[11:11 + data_count]

    async def write(self, fid: int, offset: int, data: bytes) -> int:
        """Write to an open fid. Returns bytes written."""
        return await self._write_fid(fid, offset, data)

    async def _write_fid(self, fid: int, offset: int, data: bytes) -> int:
        max_data = self._msize - 23
        data = data[:max_data]

        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TWRITE, tag)
        body += struct.pack('<I', fid)
        body += struct.pack('<Q', offset)
        body += struct.pack('<I', len(data))
        body += data
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, "write")

        return struct.unpack_from('<I', resp, 7)[0]

    async def clunk(self, fid: int):
        """Clunk (close) a fid."""
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TCLUNK, tag)
        body += struct.pack('<I', fid)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        # Don't raise on clunk error — best effort
        err = self._check_rerror(resp)
        if err:
            logger.debug(f"Clunk fid={fid} error: {err}")

    async def stat(self, fid: int) -> Stat9P:
        """Stat a fid. Returns Stat9P."""
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TSTAT, tag)
        body += struct.pack('<I', fid)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, "stat")

        # Rstat: count[2] stat[n]
        # The stat data starts after the 7-byte header + 2-byte count
        stat_obj, _ = parse_stat(resp, 9)
        return stat_obj

    async def create(self, fid: int, name: str, perm: int,
                     mode: int) -> Tuple[int, int, int, int]:
        """
        Create a file/dir. fid must point to the parent directory.
        After create, fid points to the new file (opened with mode).
        Returns (qid_type, qid_vers, qid_path, iounit).
        """
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TCREATE, tag)
        body += struct.pack('<I', fid)
        body += _pack_string(name)
        body += struct.pack('<I', perm)
        body += struct.pack('<B', mode)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, f"create {name}")

        qt = resp[7]
        qv = struct.unpack_from('<I', resp, 8)[0]
        qp = struct.unpack_from('<Q', resp, 12)[0]
        iounit = struct.unpack_from('<I', resp, 20)[0]

        return qt, qv, qp, iounit

    async def remove(self, fid: int):
        """Remove a fid (also clunks it)."""
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TREMOVE, tag)
        body += struct.pack('<I', fid)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag)
        self._raise_if_error(resp, "remove")

    async def flush(self, oldtag: int):
        """Flush a pending request."""
        tag = self._alloc_tag()

        body = bytearray()
        body += struct.pack('<IBH', 0, TFLUSH, tag)
        body += struct.pack('<H', oldtag)
        struct.pack_into('<I', body, 0, len(body))

        resp = await self._transact(bytes(body), tag, timeout=5.0)
        # Rflush doesn't have error content

    async def read_dir(self, fid: int) -> List[Stat9P]:
        """Read an entire directory. fid must be open for reading."""
        entries = []
        offset = 0
        while True:
            data = await self._read_fid(fid, offset, self._msize - 24)
            if not data:
                break
            entries.extend(parse_dir_data(data))
            offset += len(data)
        return entries


# ═══════════════════════════════════════════════════════════════════
#  FUSE ↔ 9P Mapping
# ═══════════════════════════════════════════════════════════════════
#
# FUSE works with inodes. 9P works with fids and qid.path.
# We use qid.path as the inode number (they're unique per file
# on a given 9P server).
#
# The tricky part: FUSE expects to look up by (parent_inode, name)
# while 9P walks from a fid. So we maintain:
#
#   - inode → path mapping (for stat / walk purposes)
#   - fh (file handle) → open fid mapping
#   - inode → cached Stat9P (with TTL)
#

@dataclass
class InodeInfo:
    """Tracks what we know about an inode."""
    path: str                   # Full path from root
    qid_type: int = 0
    qid_vers: int = 0
    qid_path: int = 0          # This IS the inode number
    stat: Optional[Stat9P] = None
    stat_time: float = 0.0     # When stat was fetched
    lookup_count: int = 0

    @property
    def is_dir(self) -> bool:
        return bool(self.qid_type & QTDIR)


@dataclass
class OpenFile:
    """Tracks an open file handle."""
    fid: int          # 9P fid, open on the server
    inode: int        # Which inode this belongs to
    mode: int         # Open mode (OREAD, OWRITE, etc.)
    iounit: int = 0   # Server's preferred I/O unit


class AsyncioBridge:
    """
    Runs an asyncio event loop in a background thread.
    
    Allows trio code (pyfuse3 callbacks) to call asyncio coroutines
    on the NineP connection via run_coro(). The call blocks the
    calling trio task until the asyncio coroutine completes.
    """
    
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="asyncio-bridge",
            daemon=True,
        )
        self._thread.start()
    
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def run_coro(self, coro):
        """
        Submit an asyncio coroutine and block until it completes.
        Safe to call from trio context (or any thread).
        Returns the coroutine's result or raises its exception.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120.0)
    
    async def run_coro_async(self, coro):
        """
        Submit an asyncio coroutine from a trio task.
        Uses trio's thread blocking to avoid starving the trio scheduler.
        """
        return await trio.to_thread.run_sync(
            lambda: self.run_coro(coro)
        )
    
    def stop(self):
        """Stop the asyncio event loop."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)


class NinePFuse(pyfuse3.Operations):
    """
    FUSE operations backed by a 9P2000 connection.
    
    Maps FUSE inode operations to 9P fid operations, maintaining
    a cache of inode→path mappings and open file handles.
    """

    STAT_TTL = 1.0  # Seconds to cache stat results

    def __init__(self, conn: NineP, bridge: AsyncioBridge):
        super().__init__()
        self._conn = conn
        self._bridge = bridge

        # inode (qid_path) → InodeInfo
        self._inodes: Dict[int, InodeInfo] = {}

        # file handle counter → OpenFile
        self._next_fh = 1
        self._open_files: Dict[int, OpenFile] = {}

        # dir handle → (fid, cached entries)
        self._open_dirs: Dict[int, Tuple[int, List[Stat9P]]] = {}

        # Parent inode + name → child inode (cache for lookup)
        self._lookup_cache: Dict[Tuple[int, str], int] = {}

        # The root inode is always 1 in FUSE (pyfuse3.ROOT_INODE)
        # We'll map it once we know the root qid_path.
        self._root_qid_path: Optional[int] = None

    async def _aio(self, coro):
        """Bridge an asyncio coroutine into trio context."""
        return await self._bridge.run_coro_async(coro)

    def set_root(self, qid_type: int, qid_vers: int, qid_path: int):
        """Set the root inode from the attach qid."""
        self._root_qid_path = qid_path
        info = InodeInfo(
            path="/",
            qid_type=qid_type,
            qid_vers=qid_vers,
            qid_path=qid_path,
            lookup_count=1,
        )
        # Map both the real qid_path and FUSE's ROOT_INODE (1)
        self._inodes[pyfuse3.ROOT_INODE] = info
        if qid_path != pyfuse3.ROOT_INODE:
            self._inodes[qid_path] = info

    def _inode_to_info(self, inode: int) -> InodeInfo:
        info = self._inodes.get(inode)
        if info is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        return info

    def _alloc_fh(self) -> int:
        fh = self._next_fh
        self._next_fh += 1
        return fh

    def _resolve_inode(self, qid_path: int) -> int:
        """Map a qid_path to the FUSE inode. Root is special."""
        if qid_path == self._root_qid_path:
            return pyfuse3.ROOT_INODE
        return qid_path

    async def _walk_path(self, path: str) -> int:
        """Walk to a path and return the fid. Caller must clunk."""
        names = [n for n in path.split('/') if n]
        fid = self._conn._alloc_fid()
        await self._aio(self._conn.walk(self._conn.root_fid, fid, names))
        return fid

    async def _stat_inode(self, info: InodeInfo) -> Stat9P:
        """Get stat for an inode, using cache if fresh."""
        now = time.time()
        if info.stat and (now - info.stat_time) < self.STAT_TTL:
            return info.stat

        fid = await self._walk_path(info.path)
        try:
            st = await self._aio(self._conn.stat(fid))
            info.stat = st
            info.stat_time = now
            return st
        finally:
            await self._aio(self._conn.clunk(fid))

    def _stat_to_entry(self, st: Stat9P, inode: int) -> pyfuse3.EntryAttributes:
        """Convert a 9P Stat to a FUSE EntryAttributes."""
        entry = pyfuse3.EntryAttributes()
        entry.st_ino = inode
        entry.generation = 0
        entry.entry_timeout = 1
        entry.attr_timeout = 1

        if st.is_dir:
            entry.st_mode = stat.S_IFDIR | (st.mode & 0o777)
            entry.st_nlink = 2
            entry.st_size = 0
        else:
            entry.st_mode = stat.S_IFREG | (st.mode & 0o777)
            entry.st_nlink = 1
            entry.st_size = st.length

        entry.st_uid = os.getuid()
        entry.st_gid = os.getgid()
        entry.st_blksize = 4096
        entry.st_blocks = (entry.st_size + 511) // 512

        entry.st_atime_ns = st.atime * 1_000_000_000
        entry.st_mtime_ns = st.mtime * 1_000_000_000
        entry.st_ctime_ns = st.mtime * 1_000_000_000

        return entry

    # ── FUSE Operations ─────────────────────────────────────

    async def getattr(self, inode, ctx=None):
        info = self._inode_to_info(inode)
        st = await self._stat_inode(info)
        return self._stat_to_entry(st, inode)

    async def lookup(self, parent_inode, name, ctx=None):
        name_str = name.decode('utf-8')
        parent = self._inode_to_info(parent_inode)

        # Check cache
        cached_ino = self._lookup_cache.get((parent_inode, name_str))
        if cached_ino and cached_ino in self._inodes:
            info = self._inodes[cached_ino]
            if info.stat and (time.time() - info.stat_time) < self.STAT_TTL:
                info.lookup_count += 1
                return self._stat_to_entry(info.stat, cached_ino)

        # Walk to the child
        child_path = parent.path.rstrip('/') + '/' + name_str
        fid = self._conn._alloc_fid()
        names = [n for n in child_path.split('/') if n]

        try:
            qids = await self._aio(
                self._conn.walk(self._conn.root_fid, fid, names)
            )
        except OSError:
            raise pyfuse3.FUSEError(errno.ENOENT)

        if len(qids) != len(names):
            await self._aio(self._conn.clunk(fid))
            raise pyfuse3.FUSEError(errno.ENOENT)

        qt, qv, qp = qids[-1]

        try:
            st = await self._aio(self._conn.stat(fid))
        finally:
            await self._aio(self._conn.clunk(fid))

        ino = self._resolve_inode(qp)

        info = InodeInfo(
            path=child_path,
            qid_type=qt, qid_vers=qv, qid_path=qp,
            stat=st, stat_time=time.time(),
            lookup_count=1,
        )
        self._inodes[ino] = info
        self._lookup_cache[(parent_inode, name_str)] = ino

        return self._stat_to_entry(st, ino)

    async def forget(self, inode_list):
        for inode, nlookup in inode_list:
            info = self._inodes.get(inode)
            if info:
                info.lookup_count -= nlookup
                if info.lookup_count <= 0 and inode != pyfuse3.ROOT_INODE:
                    self._inodes.pop(inode, None)

    async def opendir(self, inode, ctx):
        info = self._inode_to_info(inode)
        if not info.is_dir:
            raise pyfuse3.FUSEError(errno.ENOTDIR)

        fid = await self._walk_path(info.path)
        await self._aio(self._conn.open(fid, OREAD))
        entries = await self._aio(self._conn.read_dir(fid))
        await self._aio(self._conn.clunk(fid))

        # Cache child inodes from directory listing
        for st in entries:
            child_path = info.path.rstrip('/') + '/' + st.name
            child_ino = self._resolve_inode(st.qid_path)
            child_info = InodeInfo(
                path=child_path,
                qid_type=st.qid_type, qid_vers=st.qid_vers,
                qid_path=st.qid_path,
                stat=st, stat_time=time.time(),
                lookup_count=1,
            )
            self._inodes[child_ino] = child_info
            self._lookup_cache[(inode, st.name)] = child_ino

        fh = self._alloc_fh()
        self._open_dirs[fh] = (0, entries)
        return fh

    async def readdir(self, fh, start_id, token):
        _, entries = self._open_dirs.get(fh, (0, []))

        for i in range(start_id, len(entries)):
            st = entries[i]
            ino = self._resolve_inode(st.qid_path)

            entry = self._stat_to_entry(st, ino)

            if not pyfuse3.readdir_reply(
                token, st.name.encode('utf-8'), entry, i + 1
            ):
                break

    async def releasedir(self, fh):
        self._open_dirs.pop(fh, None)

    async def open(self, inode, flags, ctx):
        info = self._inode_to_info(inode)

        # Map POSIX flags to 9P open mode
        accmode = flags & os.O_ACCMODE
        if accmode == os.O_RDONLY:
            mode = OREAD
        elif accmode == os.O_WRONLY:
            mode = OWRITE
        else:
            mode = ORDWR

        if flags & os.O_TRUNC:
            mode |= OTRUNC

        fid = await self._walk_path(info.path)
        try:
            qt, qv, qp, iounit = await self._aio(self._conn.open(fid, mode))
        except OSError:
            await self._aio(self._conn.clunk(fid))
            raise pyfuse3.FUSEError(errno.EACCES)

        fh = self._alloc_fh()
        self._open_files[fh] = OpenFile(
            fid=fid, inode=inode, mode=mode, iounit=iounit
        )

        # Invalidate stat cache (file may change)
        info.stat = None

        fi = pyfuse3.FileInfo(fh=fh)
        # Disable kernel caching — 9P files can be dynamic
        fi.direct_io = True
        fi.keep_cache = False
        return fi

    async def read(self, fh, offset, size):
        of = self._open_files.get(fh)
        if of is None:
            raise pyfuse3.FUSEError(errno.EBADF)

        try:
            return await self._aio(self._conn.read(of.fid, offset, size))
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno or errno.EIO)

    async def write(self, fh, offset, buf):
        of = self._open_files.get(fh)
        if of is None:
            raise pyfuse3.FUSEError(errno.EBADF)

        try:
            return await self._aio(self._conn.write(of.fid, offset, buf))
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno or errno.EIO)

    async def release(self, fh):
        of = self._open_files.pop(fh, None)
        if of:
            await self._aio(self._conn.clunk(of.fid))
            # Invalidate stat cache
            info = self._inodes.get(of.inode)
            if info:
                info.stat = None

    async def create(self, parent_inode, name, mode, flags, ctx):
        name_str = name.decode('utf-8')
        parent = self._inode_to_info(parent_inode)

        # Walk to parent dir, clone it for create
        fid = await self._walk_path(parent.path)

        perm = mode & 0o777

        # Map flags to 9P mode
        accmode = flags & os.O_ACCMODE
        if accmode == os.O_RDONLY:
            omode = OREAD
        elif accmode == os.O_WRONLY:
            omode = OWRITE
        else:
            omode = ORDWR

        try:
            qt, qv, qp, iounit = await self._aio(
                self._conn.create(fid, name_str, perm, omode)
            )
        except OSError:
            await self._aio(self._conn.clunk(fid))
            raise pyfuse3.FUSEError(errno.EIO)

        ino = self._resolve_inode(qp)

        child_path = parent.path.rstrip('/') + '/' + name_str
        now = time.time()
        child_stat = Stat9P(
            qid_type=qt, qid_vers=qv, qid_path=qp,
            mode=perm, length=0, name=name_str,
            atime=int(now), mtime=int(now),
        )
        info = InodeInfo(
            path=child_path,
            qid_type=qt, qid_vers=qv, qid_path=qp,
            stat=child_stat, stat_time=now,
            lookup_count=1,
        )
        self._inodes[ino] = info
        self._lookup_cache[(parent_inode, name_str)] = ino

        fh = self._alloc_fh()
        self._open_files[fh] = OpenFile(
            fid=fid, inode=ino, mode=omode, iounit=iounit
        )

        fi = pyfuse3.FileInfo(fh=fh)
        fi.direct_io = True
        fi.keep_cache = False

        entry = self._stat_to_entry(child_stat, ino)
        return fi, entry

    async def mkdir(self, parent_inode, name, mode, ctx):
        name_str = name.decode('utf-8')
        parent = self._inode_to_info(parent_inode)

        fid = await self._walk_path(parent.path)

        perm = DMDIR | (mode & 0o777)

        try:
            qt, qv, qp, iounit = await self._aio(
                self._conn.create(fid, name_str, perm, OREAD)
            )
        except OSError:
            await self._aio(self._conn.clunk(fid))
            raise pyfuse3.FUSEError(errno.EIO)

        await self._aio(self._conn.clunk(fid))

        ino = self._resolve_inode(qp)
        child_path = parent.path.rstrip('/') + '/' + name_str
        now = time.time()
        child_stat = Stat9P(
            qid_type=QTDIR, qid_vers=qv, qid_path=qp,
            mode=mode & 0o777, length=0, name=name_str,
            atime=int(now), mtime=int(now),
        )
        info = InodeInfo(
            path=child_path,
            qid_type=QTDIR, qid_vers=qv, qid_path=qp,
            stat=child_stat, stat_time=now,
            lookup_count=1,
        )
        self._inodes[ino] = info
        self._lookup_cache[(parent_inode, name_str)] = ino

        return self._stat_to_entry(child_stat, ino)

    async def unlink(self, parent_inode, name, ctx):
        name_str = name.decode('utf-8')
        parent = self._inode_to_info(parent_inode)

        child_path = parent.path.rstrip('/') + '/' + name_str
        fid = await self._walk_path(child_path)

        try:
            await self._aio(self._conn.remove(fid))
        except OSError as e:
            raise pyfuse3.FUSEError(e.errno or errno.EIO)

        # Clean caches
        self._lookup_cache.pop((parent_inode, name_str), None)

    async def rmdir(self, parent_inode, name, ctx):
        # 9P remove works for both files and dirs
        await self.unlink(parent_inode, name, ctx)

    async def setattr(self, inode, attr, fields, fh, ctx):
        # 9P wstat is complex — for now, just return current attrs
        # This allows truncate etc. to "succeed" from the client's view.
        # A full implementation would send Twstat.
        info = self._inode_to_info(inode)
        st = await self._stat_inode(info)
        return self._stat_to_entry(st, inode)

    async def statfs(self, ctx):
        s = pyfuse3.StatvfsData()
        s.f_bsize = 4096
        s.f_frsize = 4096
        s.f_blocks = 0
        s.f_bfree = 0
        s.f_bavail = 0
        s.f_files = len(self._inodes)
        s.f_ffree = 0
        s.f_favail = 0
        s.f_namemax = 255
        return s


# ═══════════════════════════════════════════════════════════════════
#  CLI + main
# ═══════════════════════════════════════════════════════════════════

def parse_address(addr: str) -> Tuple[str, str, int, Optional[str]]:
    """
    Parse server address. Supports multiple formats:
        host:port               TCP (convenient, no bash escaping)
        tcp!host!port           TCP (Plan 9 style)
        unix!/path/to/sock      Unix socket (Plan 9 style)
        unix:/path/to/sock      Unix socket (colon variant)
    Returns (proto, host, port, unix_path).
    """
    # Plan 9 style: tcp!host!port or unix!/path
    parts = addr.split('!')
    if len(parts) == 3 and parts[0].lower() == 'tcp':
        return 'tcp', parts[1], int(parts[2]), None
    elif len(parts) >= 2 and parts[0].lower() == 'unix':
        path = '!'.join(parts[1:])
        return 'unix', '', 0, path

    # Unix socket with colon: unix:/path
    if addr.lower().startswith('unix:'):
        return 'unix', '', 0, addr[5:]

    # host:port (most common on Linux)
    if ':' in addr:
        host, port_str = addr.rsplit(':', 1)
        try:
            port = int(port_str)
            return 'tcp', host, port, None
        except ValueError:
            pass

    raise ValueError(
        f"Invalid address: {addr}\n"
        f"Expected: host:port, tcp!host!port, or unix!/path"
    )


async def setup_connection(args, bridge: AsyncioBridge) -> Tuple[NineP, NinePFuse]:
    """
    Connect, authenticate, attach — runs on the bridge's asyncio loop.
    Returns (conn, ops) ready for the FUSE loop.
    """
    proto, host, port, unix_path = parse_address(args.address)

    # Connect
    conn = NineP(host=host, port=port, unix_path=unix_path)
    await conn.connect()
    logger.info(f"Connected to {args.address}")

    # Version
    msize = await conn.version()
    logger.info(f"Negotiated msize={msize}")

    # Auth (if token provided)
    uname = args.user or os.environ.get('USER', 'none')
    token = args.auth_token or os.environ.get('NINEPFUSE_TOKEN', '')
    aname = args.aname or ""

    afid = NOFID
    if token:
        try:
            afid = await conn.auth(uname, aname, token)
        except OSError as e:
            if "not required" in str(e).lower():
                afid = NOFID
            else:
                await conn.close()
                raise

    # Attach
    root_fid = await conn.attach(uname, aname, afid)

    # If we used an afid and it's not NOFID, clunk it
    if afid != NOFID:
        await conn.clunk(afid)

    # Get root qid via stat
    root_stat = await conn.stat(root_fid)

    # Create FUSE operations
    ops = NinePFuse(conn, bridge)
    ops.set_root(root_stat.qid_type, root_stat.qid_vers, root_stat.qid_path)

    print(f"Mounted {args.address} on {args.mountpoint}")
    if token:
        print(f"  Authenticated as: {uname}")
    print(f"  msize: {msize}")

    return conn, ops


async def trio_main(ops, bridge, conn, args):
    """Run the FUSE main loop under trio."""
    try:
        await pyfuse3.main()
    except Exception as e:
        logger.error(f"FUSE main loop error: {e}")
    finally:
        pyfuse3.close()
        bridge.run_coro(conn.close())
        bridge.stop()
        print(f"Unmounted {args.mountpoint}")


def main():
    parser = argparse.ArgumentParser(
        description="9pfuse — Mount a 9P2000 server via FUSE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Address format:
    tcp!host!port       TCP connection
    unix!/path/to/sock  Unix socket

Examples:
    %(prog)s tcp!localhost!5642 /n/mux
    %(prog)s tcp!localhost!5642 /n/mux --auth-token secret123
    %(prog)s tcp!localhost!5642 /n/mux -u myuser -t secret123
"""
    )
    parser.add_argument(
        'address',
        help='Server address (tcp!host!port or unix!/path)'
    )
    parser.add_argument(
        'mountpoint',
        help='Local mount point'
    )
    parser.add_argument(
        '-t', '--auth-token',
        help='Auth token (or set NINEPFUSE_TOKEN env var)'
    )
    parser.add_argument(
        '-u', '--user',
        help='Username for attach (default: $USER)'
    )
    parser.add_argument(
        '-a', '--aname',
        default='',
        help='Attach name (aname, default: empty)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '-o', '--allow-other',
        action='store_true',
        help='Allow other users to access the mount (requires /etc/fuse.conf)'
    )

    args = parser.parse_args()

    level = logging.WARNING
    if args.verbose:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )

    # Start the asyncio bridge — all 9P I/O runs on this loop
    bridge = AsyncioBridge()
    bridge.start()

    # Phase 1: Connect, auth, attach on the bridge's asyncio loop
    try:
        conn, ops = bridge.run_coro(setup_connection(args, bridge))
    except KeyboardInterrupt:
        bridge.stop()
        return
    except OSError as e:
        bridge.stop()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Init FUSE
    fuse_options = set(pyfuse3.default_options)
    fuse_options.add('fsname=9pfuse')
    fuse_options.discard('default_permissions')

    if args.allow_other:
        fuse_options.add('allow_other')

    if args.debug:
        fuse_options.add('debug')

    os.makedirs(args.mountpoint, exist_ok=True)

    try:
        pyfuse3.init(ops, args.mountpoint, fuse_options)
    except RuntimeError as e:
        bridge.run_coro(conn.close())
        bridge.stop()
        print(f"Mount failed: {e}", file=sys.stderr)
        if 'allow_other' in str(e):
            print("Hint: set 'user_allow_other' in /etc/fuse.conf "
                  "or remove --allow-other", file=sys.stderr)
        sys.exit(1)

    # Phase 3: Run FUSE main loop on trio
    try:
        trio.run(trio_main, ops, bridge, conn, args)
    except KeyboardInterrupt:
        pyfuse3.close()
        bridge.run_coro(conn.close())
        bridge.stop()


if __name__ == '__main__':
    main()
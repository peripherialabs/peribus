"""
9P2000 File Server

This module implements a 9P file server that serves synthetic filesystems.
It handles multiple concurrent connections and maps 9P operations to
the SyntheticFile interface.

Fixed to properly handle:
- Tag echoing to prevent protocol violations
- Async stat() and lookup() methods for remote proxies
"""

import asyncio
import struct
import logging
import inspect
from typing import Dict, Optional, Union
from dataclasses import dataclass

from .protocol import (
    Message, NOTAG, NOFID,
    Tversion, Rversion,
    Tauth, Rauth,
    Tattach, Rattach,
    Rerror,
    Tflush, Rflush,
    Twalk, Rwalk,
    Topen, Ropen,
    Tcreate, Rcreate,
    Tread, Rread,
    Twrite, Rwrite,
    Tclunk, Rclunk,
    Tremove, Rremove,
    Tstat, Rstat,
    Twstat, Rwstat,
)
from .codec import Codec
from core.files import SyntheticFile, SyntheticDir
from core.types import FidState

logger = logging.getLogger(__name__)


class Server9P:
    """
    9P file server.
    
    Serves a synthetic filesystem tree over TCP or Unix sockets.
    Handles multiple concurrent client connections.
    """
    
    def __init__(self, root: SyntheticDir, msize: int = 65536):
        """
        Initialize server.
        
        Args:
            root: Root directory of the filesystem to serve
            msize: Maximum message size
        """
        self.root = root
        self.msize = msize
        self.codec = Codec(msize)
        self.connections: Dict[int, 'Connection9P'] = {}
        self._conn_id = 0
        self._server = None
    
    async def serve_tcp(self, host: str = '0.0.0.0', port: int = 5640):
        """
        Start TCP server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self._server = await asyncio.start_server(
            self._handle_connection,
            host, port
        )
        
        addr = self._server.sockets[0].getsockname()
        logger.info(f"9P server listening on {addr[0]}:{addr[1]}")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def serve_unix(self, path: str):
        """
        Start Unix domain socket server.
        
        Args:
            path: Path to Unix socket
        """
        import os
        
        # Remove existing socket
        try:
            os.unlink(path)
        except OSError:
            pass
        
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path
        )
        
        logger.info(f"9P server listening on {path}")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self):
        """Stop the server"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """Handle a new client connection"""
        self._conn_id += 1
        conn_id = self._conn_id
        
        peer = writer.get_extra_info('peername')
        logger.info(f"New connection {conn_id} from {peer}")
        
        conn = Connection9P(conn_id, self.root, self.codec, reader, writer)
        self.connections[conn_id] = conn
        
        try:
            await conn.serve()
        except Exception as e:
            logger.error(f"Connection {conn_id} error: {e}", exc_info=True)
        finally:
            logger.info(f"Connection {conn_id} closed")
            del self.connections[conn_id]
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


class Connection9P:
    """
    Handles a single 9P client connection.
    
    Manages the fid namespace and handles message dispatch.
    """
    
    def __init__(
        self,
        conn_id: int,
        root: SyntheticDir,
        codec: Codec,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        self.conn_id = conn_id
        self.root = root
        self.codec = codec
        self.reader = reader
        self.writer = writer
        self.msize = 8192  # Until negotiated
        self.fids: Dict[int, FidState] = {}
        self.pending: Dict[int, asyncio.Task] = {}  # tag -> task
        self._closed = False
        self._write_lock = asyncio.Lock()  # Serialize wire writes
    
    async def serve(self):
        """
        Main serve loop — read messages and dispatch concurrently.
        
        Each message is handled in its own asyncio task so that a
        blocking read() on one file doesn't prevent writes to another.
        This is critical for patterns like:
        
            Terminal A: cat $agent/output     (blocks waiting for data)
            Terminal B: echo 'hi' > $agent/input  (must proceed!)
        
        Version negotiation is handled inline (must complete before
        anything else). All other messages run as concurrent tasks.
        """
        buffer = b''
        
        while not self._closed:
            # Read data
            try:
                chunk = await self.reader.read(4096)
                if not chunk:
                    break  # Connection closed
                buffer += chunk
            except (ConnectionError, asyncio.CancelledError):
                break
            
            # Process complete messages
            while len(buffer) >= 4:
                # Check if we have complete message
                size = struct.unpack('<I', buffer[:4])[0]
                if size > self.msize:
                    logger.error(f"Message too large: {size} > {self.msize}")
                    return
                
                if len(buffer) < size:
                    break  # Need more data
                
                msg_data = buffer[:size]
                buffer = buffer[size:]
                
                # Decode
                try:
                    msg, _ = self.codec.decode(msg_data)
                    logger.debug(f"← Received {type(msg).__name__} tag={msg.tag}")
                except Exception as e:
                    logger.exception(f"Decode error: {e}")
                    continue
                
                # Version must be handled inline (sets msize for framing)
                if isinstance(msg, Tversion):
                    try:
                        response = await self._handle_message(msg)
                        if response:
                            await self._send(response)
                    except Exception as e:
                        logger.exception(f"Version handling error: {e}")
                    continue
                
                # Everything else: dispatch as concurrent task
                task = asyncio.create_task(
                    self._dispatch(msg)
                )
                self.pending[msg.tag] = task
                # Auto-cleanup when done
                task.add_done_callback(
                    lambda t, tag=msg.tag: self.pending.pop(tag, None)
                )
        
        # Cancel any pending tasks on disconnect
        for tag, task in list(self.pending.items()):
            task.cancel()
        self.pending.clear()
    
    async def _dispatch(self, msg: Message):
        """Handle a single message and send the response."""
        try:
            response = await self._handle_message(msg)
            if response:
                logger.debug(f"→ Sending {type(response).__name__} tag={response.tag}")
                await self._send(response)
        except Exception as e:
            logger.exception(f"Error handling {type(msg).__name__}: {e}")
            try:
                await self._send(Rerror(tag=msg.tag, ename=str(e)))
            except Exception:
                pass
    
    async def _send(self, msg: Message):
        """Send a message to the client (serialized with write lock)"""
        async with self._write_lock:
            try:
                msg_type_name = type(msg).__name__
                if msg.tag == NOTAG and msg_type_name not in ('Tversion', 'Rversion'):
                    logger.error(
                        f"PROTOCOL VIOLATION: {msg_type_name} using NOTAG! "
                        f"This will cause mount errors on Plan 9."
                    )
                
                data = self.codec.encode(msg)
                self.writer.write(data)
                await self.writer.drain()
            except Exception as e:
                logger.error(f"Send error: {e}")
                self._closed = True
    
    async def _handle_message(self, msg: Message) -> Optional[Message]:
        """Dispatch message to appropriate handler"""
        logger.info(f"Handling: {type(msg).__name__} tag={msg.tag}")
        handlers = {
            Tversion: self._handle_version,
            Tauth: self._handle_auth,
            Tattach: self._handle_attach,
            Twalk: self._handle_walk,
            Topen: self._handle_open,
            Tread: self._handle_read,
            Twrite: self._handle_write,
            Tclunk: self._handle_clunk,
            Tstat: self._handle_stat,
            Tflush: self._handle_flush,
            Tcreate: self._handle_create,
            Tremove: self._handle_remove,
            Twstat: self._handle_wstat,
        }
        
        handler = handlers.get(type(msg))
        if handler:
            try:
                response = await handler(msg)
                
                # Verify response tag matches request tag
                if response and response.tag != msg.tag:
                    logger.error(
                        f"BUG: Response tag {response.tag} != request tag {msg.tag}. "
                        f"Request: {type(msg).__name__}, Response: {type(response).__name__}"
                    )
                
                return response
            except Exception as e:
                logger.exception(f"Handler error for {type(msg).__name__}: {e}")
                return Rerror(tag=msg.tag, ename=str(e))
        else:
            logger.warning(f"Unknown message type: {type(msg).__name__}")
            return Rerror(tag=msg.tag, ename=f"Unknown message type: {type(msg).__name__}")
    
    # ========================================================================
    # Helper methods for async-aware operations
    # ========================================================================
    
    async def _get_stat(self, file: SyntheticFile):
        """
        Get stat from a file, handling both sync and async stat() methods.
        """
        stat_method = getattr(file, 'stat', None)
        if stat_method is None:
            raise AttributeError(f"File {file} has no stat method")
        
        result = stat_method()
        
        # If it's a coroutine, await it
        if inspect.iscoroutine(result):
            result = await result
        
        return result
    
    async def _lookup_child(self, directory: SyntheticDir, name: str) -> Optional[SyntheticFile]:
        """
        Look up a child in a directory, handling both sync and async lookup.
        
        Tries in order:
        1. async lookup() method (for RemoteProxyDir)
        2. sync get() method (for regular SyntheticDir)
        3. direct children access (fallback)
        """
        # Try async lookup first (for remote proxy dirs)
        if hasattr(directory, 'lookup'):
            lookup_method = directory.lookup
            logger.info(f"    -> Using lookup() on {type(directory).__name__}")
            result = lookup_method(name)
            if inspect.iscoroutine(result):
                logger.info(f"    -> Awaiting coroutine...")
                result = await result
            logger.info(f"    -> lookup() returned: {type(result).__name__ if result else None}")
            return result
        
        # Try sync get method
        if hasattr(directory, 'get'):
            logger.info(f"    -> Using get() on {type(directory).__name__}")
            return directory.get(name)
        
        # Direct children access as fallback
        if hasattr(directory, 'children'):
            logger.info(f"    -> Using children dict on {type(directory).__name__}")
            return directory.children.get(name)
        
        return None
    
    # ========================================================================
    # Message Handlers
    # ========================================================================
    
    async def _handle_version(self, msg: Tversion) -> Rversion:
        """Handle version negotiation"""
        # Negotiate msize
        self.msize = min(msg.msize, self.codec.msize)
        
        # We only speak 9P2000
        if "9P2000" in msg.version:
            version = "9P2000"
        else:
            version = "unknown"
        
        # Clear any existing state on version
        self.fids.clear()
        
        logger.debug(f"Version negotiated: msize={self.msize}, version={version}")
        
        return Rversion(tag=msg.tag, msize=self.msize, version=version)
    
    async def _handle_auth(self, msg: Tauth) -> Rerror:
        """Handle authentication - we don't support it"""
        return Rerror(tag=msg.tag, ename="Authentication not required")
    
    async def _handle_attach(self, msg: Tattach) -> Union[Rattach, Rerror]:
        """Handle attach (mount root)"""
        if msg.fid in self.fids:
            return Rerror(tag=msg.tag, ename="Fid already in use")
        
        qid = self.root.qid
        
        self.fids[msg.fid] = FidState(
            fid=msg.fid,
            path="/",
            qid=qid,
            file=self.root
        )
        
        logger.debug(f"Attach: fid={msg.fid}, uname={msg.uname}, aname={msg.aname}")
        
        return Rattach(tag=msg.tag, qid=qid)
    
    async def _handle_walk(self, msg: Twalk) -> Union[Rwalk, Rerror]:
        """Handle path traversal - now with async lookup support"""
        if msg.fid not in self.fids:
            logger.error(f"Walk FAILED: unknown fid={msg.fid}")
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        source = self.fids[msg.fid]
        # Always log walk attempts at INFO level for debugging
        logger.info(f"Walk: fid={msg.fid} from='{source.path}' wnames={msg.wnames} newfid={msg.newfid}")
        current = source.file
        current_path = source.path
        qids = []
        
        # Empty walk = clone fid
        if not msg.wnames:
            if msg.newfid != msg.fid:
                self.fids[msg.newfid] = FidState(
                    fid=msg.newfid,
                    path=current_path,
                    qid=current.qid,
                    file=current
                )
            logger.info(f"Walk: clone fid {msg.fid} -> {msg.newfid}")
            return Rwalk(tag=msg.tag, qids=[])
        
        # Walk each path component
        for i, name in enumerate(msg.wnames):
            logger.info(f"  Walk[{i}]: '{name}' in '{current_path}' (type={type(current).__name__})")
            
            if not isinstance(current, SyntheticDir):
                # Can't walk into non-directory
                logger.warning(f"  Walk[{i}]: FAIL - not a directory")
                if i == 0:
                    return Rerror(tag=msg.tag, ename="Not a directory")
                break  # Partial walk is OK
            
            if name == "..":
                current = current.parent or current
                parts = current_path.rstrip("/").split("/")
                current_path = "/".join(parts[:-1]) or "/"
                logger.info(f"  Walk[{i}]: '..' -> '{current_path}'")
            elif name == ".":
                logger.info(f"  Walk[{i}]: '.' -> stay")
                pass  # Stay in place
            else:
                # USE ASYNC LOOKUP instead of sync get()
                child = await self._lookup_child(current, name)
                
                if child is None:
                    logger.warning(f"  Walk[{i}]: FAIL - '{name}' not found")
                    if i == 0:
                        return Rerror(tag=msg.tag, ename=f"File not found: {name}")
                    break  # Partial walk is OK
                
                logger.info(f"  Walk[{i}]: OK -> {type(child).__name__}")
                current = child
                if current_path == "/":
                    current_path = f"/{name}"
                else:
                    current_path = f"{current_path}/{name}"
            
            qids.append(current.qid)
        
        # Success - create newfid
        if qids:
            self.fids[msg.newfid] = FidState(
                fid=msg.newfid,
                path=current_path,
                qid=current.qid,
                file=current
            )
        
        logger.info(f"Walk DONE: '{source.path}' -> '{current_path}' ({len(qids)}/{len(msg.wnames)} qids)")
        
        return Rwalk(tag=msg.tag, qids=qids)
    
    async def _handle_open(self, msg: Topen) -> Union[Ropen, Rerror]:
        """Handle file open"""
        if msg.fid not in self.fids:
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        fid_state = self.fids[msg.fid]
        
        if fid_state.opened:
            return Rerror(tag=msg.tag, ename="Fid already open")
        
        # Linux 9p client checks permissions - validate write access
        from .protocol import OWRITE, ORDWR, OTRUNC
        if msg.mode & (OWRITE | ORDWR):
            # Use async stat
            stat = await self._get_stat(fid_state.file)
            # Check if file has write permissions (owner write bit)
            if not (stat.mode & 0o200):
                logger.warning(f"Open for write denied: {fid_state.path} has mode {stat.mode:#o}")
                return Rerror(tag=msg.tag, ename="Permission denied")
        
        # Call file's open method
        try:
            open_result = fid_state.file.open(fid_state, msg.mode)
            # Handle async open if needed
            if inspect.iscoroutine(open_result):
                await open_result
            
            fid_state.mode = msg.mode
            fid_state.opened = True
            fid_state.offset = 0
        except Exception as e:
            return Rerror(tag=msg.tag, ename=str(e))
        
        # iounit of 0 means use msize - header overhead
        iounit = self.msize - 24
        
        logger.debug(f"Open: {fid_state.path}, mode={msg.mode:#x}")
        
        return Ropen(tag=msg.tag, qid=fid_state.qid, iounit=iounit)
    
    async def _handle_read(self, msg: Tread) -> Union[Rread, Rerror]:
        """Handle file read — with proper directory stat packing."""
        if msg.fid not in self.fids:
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        fid_state = self.fids[msg.fid]
        
        if not fid_state.opened:
            return Rerror(tag=msg.tag, ename="Fid not open")
        
        try:
            # Limit count to iounit
            count = min(msg.count, self.msize - 24)
            
            # Directory reads need special handling to pack child stat entries
            if isinstance(fid_state.file, SyntheticDir):
                data = await self._read_directory(fid_state, msg.offset, count)
            else:
                # Regular file read
                data = await fid_state.file.read(fid_state, msg.offset, count)
            
            logger.debug(
                f"Read: {fid_state.path}, offset={msg.offset}, "
                f"requested={count}, got={len(data)} bytes"
            )
            
            return Rread(tag=msg.tag, data=data)
        except Exception as e:
            logger.exception(f"Read error on {fid_state.path}: {e}")
            return Rerror(tag=msg.tag, ename=str(e))

    async def _read_directory(self, fid_state: FidState, offset: int, count: int) -> bytes:
        """
        Read directory entries as packed 9P2000 stat records.
        
        Per the 9P spec, directory reads must:
        1. Build a blob of stat entries for all children
        2. Cache the blob on the fid so offsets stay consistent
        3. Never split a stat entry across two reads
        4. Rebuild the cache when offset == 0 (fresh readdir)
        
        Each child stat is wrapped in asyncio.wait_for to prevent
        a single blocking stat() from stalling the entire readdir.
        """
        directory = fid_state.file
        
        # Build/cache the directory stat blob.
        # Rebuild on offset==0 (fresh readdir from kernel).
        if offset == 0 or not hasattr(fid_state, '_dir_cache'):
            blob = bytearray()
            
            # Get children - handle both dict and ordered approaches
            children = []
            if hasattr(directory, 'children') and isinstance(directory.children, dict):
                children = list(directory.children.values())
            elif hasattr(directory, '_children'):
                if isinstance(directory._children, dict):
                    children = list(directory._children.values())
                else:
                    children = list(directory._children)
            
            for child in children:
                try:
                    stat = await asyncio.wait_for(
                        self._get_stat(child), timeout=1.0
                    )
                    stat_bytes = self._pack_stat(stat)
                    blob.extend(stat_bytes)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout getting stat for child in {fid_state.path}, skipping"
                    )
                    continue
                except Exception as e:
                    logger.warning(f"Skipping child in dir read ({fid_state.path}): {e}")
                    continue
            
            fid_state._dir_cache = bytes(blob)
        
        cached = fid_state._dir_cache
        
        # Offset past the end → EOF
        if offset >= len(cached):
            return b""
        
        # Slice from offset, but don't split stat entries.
        # Walk stat boundaries up to count bytes.
        pos = offset
        end = min(offset + count, len(cached))
        
        while pos < end:
            # Each stat entry starts with a 2-byte size field.
            # The total entry length is size + 2 (the size field itself).
            if pos + 2 > len(cached):
                break
            entry_size = struct.unpack_from('<H', cached, pos)[0]
            entry_total = entry_size + 2
            
            if pos + entry_total > end:
                # This entry doesn't fit in the remaining count — stop here.
                # (But include it if we haven't returned anything yet and it
                #  fits in count, to avoid infinite loops.)
                if pos == offset and entry_total <= count:
                    pos += entry_total
                break
            pos += entry_total
        
        return cached[offset:pos]

    @staticmethod
    def _pack_stat(stat) -> bytes:
        """
        Pack a Stat object into the 9P2000 wire format for directory reads.
        
        Wire format (all little-endian):
            size[2]    — byte count of everything AFTER this field
            type[2]    — kernel use
            dev[4]     — kernel use
            qid[13]    — type[1] vers[4] path[8]
            mode[4]    — permissions + DMDIR etc
            atime[4]   — last access time
            mtime[4]   — last modification time
            length[8]  — file length (0 for dirs)
            name[s]    — file name (2-byte len + UTF-8)
            uid[s]     — owner
            gid[s]     — group
            muid[s]    — last modifier
        """
        name_bytes = (getattr(stat, 'name', '') or '').encode('utf-8')
        uid_bytes  = (getattr(stat, 'uid', 'none') or 'none').encode('utf-8')
        gid_bytes  = (getattr(stat, 'gid', 'none') or 'none').encode('utf-8')
        muid_bytes = (getattr(stat, 'muid', '') or '').encode('utf-8')
        
        qid = stat.qid
        qid_bytes = struct.pack('<BIQ',
            getattr(qid, 'type', 0),
            getattr(qid, 'version', 0),
            getattr(qid, 'path', 0),
        )
        
        # Body = everything after the 2-byte size field
        body = struct.pack('<HI', getattr(stat, 'type', 0), getattr(stat, 'dev', 0))
        body += qid_bytes                                          # 13 bytes
        body += struct.pack('<I', getattr(stat, 'mode', 0))
        body += struct.pack('<I', getattr(stat, 'atime', 0))
        body += struct.pack('<I', getattr(stat, 'mtime', 0))
        body += struct.pack('<Q', getattr(stat, 'length', 0))
        body += struct.pack('<H', len(name_bytes)) + name_bytes
        body += struct.pack('<H', len(uid_bytes))  + uid_bytes
        body += struct.pack('<H', len(gid_bytes))  + gid_bytes
        body += struct.pack('<H', len(muid_bytes)) + muid_bytes
        
        # Prepend 2-byte size (byte count of body)
        return struct.pack('<H', len(body)) + body
    
    async def _handle_write(self, msg: Twrite) -> Union[Rwrite, Rerror]:
        """Handle file write"""
        logger.info(f"Write request: fid={msg.fid}, offset={msg.offset}, len={len(msg.data)}")
        
        if msg.fid not in self.fids:
            logger.error(f"Write failed: Unknown fid {msg.fid}")
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        fid_state = self.fids[msg.fid]
        logger.info(f"Write to: {fid_state.path}, opened={fid_state.opened}, mode={fid_state.mode if hasattr(fid_state, 'mode') else 'N/A'}")
        
        if not fid_state.opened:
            logger.error(f"Write failed: Fid not open")
            return Rerror(tag=msg.tag, ename="Fid not open")
        
        try:
            count = await fid_state.file.write(fid_state, msg.offset, msg.data)
            
            logger.info(
                f"Write SUCCESS: {fid_state.path}, offset={msg.offset}, "
                f"requested={len(msg.data)}, wrote={count} bytes"
            )
            
            return Rwrite(tag=msg.tag, count=count)
        except Exception as e:
            logger.exception(f"Write error on {fid_state.path}: {e}")
            return Rerror(tag=msg.tag, ename=str(e))
            
    async def _handle_clunk(self, msg: Tclunk) -> Union[Rclunk, Rerror]:
        """Handle fid close"""
        if msg.fid not in self.fids:
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        fid_state = self.fids[msg.fid]
        
        # Call file's clunk method for cleanup
        try:
            clunk_result = fid_state.file.clunk(fid_state)
            # Handle async clunk if needed
            if inspect.iscoroutine(clunk_result):
                await clunk_result
        except Exception as e:
            logger.warning(f"Clunk cleanup error on {fid_state.path}: {e}")
        
        logger.debug(f"Clunk: fid={msg.fid} ({fid_state.path})")
        
        del self.fids[msg.fid]
        
        return Rclunk(tag=msg.tag)
    
    async def _handle_stat(self, msg: Tstat) -> Union[Rstat, Rerror]:
        """Handle stat request - now with async support"""
        if msg.fid not in self.fids:
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        fid_state = self.fids[msg.fid]
        
        try:
            # USE ASYNC STAT
            stat = await self._get_stat(fid_state.file)
            logger.info(f"Stat: {fid_state.path} - mode={stat.mode:#o}, uid={stat.uid}, gid={stat.gid}")
            return Rstat(tag=msg.tag, stat=stat)
        except Exception as e:
            logger.exception(f"Stat error on {fid_state.path}: {e}")
            return Rerror(tag=msg.tag, ename=str(e))
    
    async def _handle_flush(self, msg: Tflush) -> Rflush:
        """Handle flush (cancel pending request)"""
        if msg.oldtag in self.pending:
            task = self.pending[msg.oldtag]
            task.cancel()
            del self.pending[msg.oldtag]
            logger.debug(f"Flush: cancelled tag={msg.oldtag}")
        else:
            logger.debug(f"Flush: tag={msg.oldtag} not found (already completed?)")
        
        return Rflush(tag=msg.tag)
    
    async def _handle_create(self, msg: Tcreate) -> Union[Rcreate, Rerror]:
        """Handle file creation (mkdir and touch)"""
        if msg.fid not in self.fids:
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        fid_state = self.fids[msg.fid]
        
        if not isinstance(fid_state.file, SyntheticDir):
            return Rerror(tag=msg.tag, ename="Not a directory")
        
        try:
            # Call the create method on the directory
            # This will trigger our Agent creation logic in AgentsDir
            new_fid_state = await fid_state.file.create(
                fid_state, 
                msg.name, 
                msg.perm, 
                msg.mode
            )
            
            # After a successful 9P create, the original FID now 
            # points to the newly created file/directory.
            self.fids[msg.fid] = new_fid_state
            
            logger.info(f"Create SUCCESS: {new_fid_state.path}")
            
            return Rcreate(
                tag=msg.tag, 
                qid=new_fid_state.qid, 
                iounit=self.msize - 24
            )
            
        except Exception as e:
            logger.exception(f"Create error in {fid_state.path}: {e}")
            return Rerror(tag=msg.tag, ename=str(e))
    
    async def _handle_remove(self, msg: Tremove) -> Union[Rremove, Rerror]:
        """Handle file removal"""
        if msg.fid not in self.fids:
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        # For now, don't support removal
        return Rerror(tag=msg.tag, ename="Remove not supported")
    
    async def _handle_wstat(self, msg: Twstat) -> Union[Rwstat, Rerror]:
        """Handle stat modification"""
        if msg.fid not in self.fids:
            return Rerror(tag=msg.tag, ename="Unknown fid")
        
        fid_state = self.fids[msg.fid]
        
        # Log what Linux is trying to change
        logger.info(f"Wstat request on {fid_state.path}: length={msg.stat.length}, mode={msg.stat.mode:#o}")
        
        # Linux often does wstat with length=0 to truncate on O_TRUNC open
        # We can safely ignore this for most synthetic files
        # Just succeed silently
        
        # If length is being set and it's a DataFile, we could truncate
        if msg.stat.length is not None and hasattr(fid_state.file, '_data'):
            if msg.stat.length == 0:
                fid_state.file._data = bytearray()
                fid_state.file.touch()
                logger.info(f"Truncated {fid_state.path} to zero length")
        
        return Rwstat(tag=msg.tag)  # Success!
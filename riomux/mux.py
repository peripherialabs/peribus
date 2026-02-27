"""
riomux.mux — Per-client multiplexer connection.

Each client connection gets a MuxConnection that:

1. Serves a virtual root directory listing the backend names
2. Routes walks into backend names to actual backend 9P connections
3. Proxies all subsequent operations (read, write, clunk, etc.)
   through to the backend with fid/tag rewriting
4. Handles flush by forwarding to the correct backend
5. Exposes a `ctl` file for dynamic backend add/remove

Virtual filesystem:
    /           → virtual root (lists backends + ctl)
    /ctl        → write "add name host:port" or "remove name"
    /rio/       → proxied to rio backend
    /llm/       → proxied to llm backend

Dynamic backend management:
    echo 'add rio2 127.0.0.1:5643' > /n/ctl    # add a backend
    echo 'remove rio2' > /n/ctl                  # remove it
    cat /n/ctl                                    # list backends

Fid mapping:
    client_fid → either:
      - MUX_ROOT: points to virtual root
      - MUX_CTL: points to the ctl file
      - (backend_name, backend_fid): proxied to a backend

When the client walks from root into "rio", we:
  1. Connect to the rio backend (if not already)
  2. Clone the backend root fid
  3. Walk the remaining path components on the backend
  4. Map client_fid → ("rio", backend_fid)

All subsequent operations on that fid go directly to the backend.
"""

import asyncio
import struct
import logging
import time
from typing import Dict, Optional, Tuple, List

from . import wire
from .backend import BackendConnection

logger = logging.getLogger("riomux.mux")

# Sentinel for fids pointing to the virtual mux root or backend dirs
MUX_ROOT = "__mux_root__"
MUX_BACKEND_DIR = "__mux_backend_dir__"  # fid pointing to a backend's root (virtual)
MUX_CTL = "__mux_ctl__"  # fid pointing to the ctl file


class FidInfo:
    """Tracks what a client fid points to."""
    __slots__ = ('kind', 'backend', 'backend_fid', 'path', 'ctl_write_buf')
    
    def __init__(self, kind: str, backend: str = "", backend_fid: int = 0,
                 path: str = "/"):
        self.kind = kind          # MUX_ROOT, MUX_CTL, "proxied"
        self.backend = backend    # backend name (if proxied)
        self.backend_fid = backend_fid  # fid on the backend
        self.path = path
        self.ctl_write_buf = b""  # accumulates writes when kind == MUX_CTL
    
    def __repr__(self):
        if self.kind == MUX_ROOT:
            return f"FidInfo(MUX_ROOT)"
        if self.kind == MUX_CTL:
            return f"FidInfo(MUX_CTL)"
        return f"FidInfo(proxied:{self.backend} fid={self.backend_fid})"


class MuxConnection:
    """
    Handles one client connection to the multiplexer.
    
    Manages:
    - Virtual root directory (listing backends + ctl)
    - ctl file for dynamic backend add/remove
    - Per-backend connections with fid/tag mapping
    - Message proxying with full fidelity
    """
    
    def __init__(
        self,
        conn_id: int,
        backend_configs: Dict[str, Tuple[str, int]],  # name → (host, port)
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self.conn_id = conn_id
        self._backend_configs = dict(backend_configs)  # mutable copy
        self._reader = reader
        self._writer = writer
        self._closed = False
        
        # Negotiated msize
        self._msize = 8192
        
        # Client fid → FidInfo
        self._fids: Dict[int, FidInfo] = {}
        
        # Client tag → backend name (for flush routing)
        self._tag_routes: Dict[int, str] = {}
        
        # Backend connections (lazy, one per backend per client)
        self._backends: Dict[str, BackendConnection] = {}
        
        # Write lock for client-facing writes
        self._write_lock = asyncio.Lock()
        
        # Virtual root qid path (unique per connection, small enough for ino_t)
        # Use conn_id shifted to avoid collisions with backend qid paths.
        # Keep values in 32-bit range — 9pfuse maps qid.path to ino_t and
        # some builds use 32-bit inodes. Large values cause ERANGE.
        _base = (conn_id * 1000) & 0x7FFFFFFF
        self._root_qid_path = _base
        
        # Ctl file qid path
        self._ctl_qid_path = _base + 1
        
        # Per-backend virtual dir qid paths (counter starts at 2)
        self._next_backend_qid = _base + 2
        self._backend_qid_paths: Dict[str, int] = {}
        self._rebuild_backend_qid_paths()
        
        # Callback for notifying server of add/remove (set by server)
        self._on_backend_add = None
        self._on_backend_remove = None
    
    def _rebuild_backend_qid_paths(self):
        """Rebuild qid path map when backends change."""
        for name in sorted(self._backend_configs.keys()):
            if name not in self._backend_qid_paths:
                self._backend_qid_paths[name] = self._next_backend_qid
                self._next_backend_qid += 1
    
    # ── Dynamic backend management (called from server) ─────────
    
    def add_backend(self, name: str, host: str, port: int):
        """
        Add a backend to this connection's config.
        New walks will find it; existing fids are unaffected.
        """
        self._backend_configs[name] = (host, port)
        self._backend_qid_paths[name] = self._next_backend_qid
        self._next_backend_qid += 1
        logger.info(f"[{self.conn_id}] Backend '{name}' added ({host}:{port})")
    
    async def remove_backend(self, name: str):
        """
        Remove a backend from this connection's config.
        
        Existing fids pointing to this backend remain valid until
        clunked — the backend connection stays alive for draining.
        New walks will not find the backend.
        """
        self._backend_configs.pop(name, None)
        self._backend_qid_paths.pop(name, None)
        # Don't close the backend connection — existing fids may still use it.
        # It will be cleaned up when the last fid using it is clunked,
        # or when the client disconnects.
        logger.info(f"[{self.conn_id}] Backend '{name}' removed from config "
                     f"(existing fids still valid)")
    
    async def _handle_ctl_command(self, command: str) -> str:
        """
        Process a ctl command. Returns a status message.
        
        Commands:
            add <name> <host>:<port>    — add a backend
            remove <name>               — remove a backend
            (empty / whitespace)        — ignored
        """
        command = command.strip()
        if not command:
            return ""
        
        parts = command.split()
        verb = parts[0].lower()
        
        if verb == "add" and len(parts) == 3:
            name = parts[1]
            addr = parts[2]
            
            if ':' not in addr:
                return f"error: invalid address '{addr}', expected host:port\n"
            
            host, port_str = addr.rsplit(':', 1)
            try:
                port = int(port_str)
            except ValueError:
                return f"error: invalid port '{port_str}'\n"
            
            if name in self._backend_configs:
                return f"error: backend '{name}' already exists\n"
            
            # Add locally
            self.add_backend(name, host, port)
            
            # Notify server to propagate to other connections
            if self._on_backend_add:
                await self._on_backend_add(name, host, port, exclude_conn=self.conn_id)
            
            logger.info(f"[{self.conn_id}] ctl: added backend '{name}' → {host}:{port}")
            return f"added {name} {host}:{port}\n"
        
        elif verb == "remove" and len(parts) == 2:
            name = parts[1]
            
            if name not in self._backend_configs:
                return f"error: backend '{name}' not found\n"
            
            await self.remove_backend(name)
            
            # Notify server to propagate
            if self._on_backend_remove:
                await self._on_backend_remove(name, exclude_conn=self.conn_id)
            
            logger.info(f"[{self.conn_id}] ctl: removed backend '{name}'")
            return f"removed {name}\n"
        
        else:
            return (f"error: unknown command '{command}'\n"
                    f"usage: add <name> <host>:<port> | remove <name>\n")
    
    async def serve(self):
        """Main client read loop."""
        buffer = b''
        
        while not self._closed:
            try:
                chunk = await self._reader.read(65536)
                if not chunk:
                    break
                buffer += chunk
            except (ConnectionError, asyncio.CancelledError):
                break
            
            while len(buffer) >= 4:
                size = struct.unpack_from('<I', buffer, 0)[0]
                if size > self._msize + 256:
                    logger.error(f"[{self.conn_id}] Message too large: {size}")
                    self._closed = True
                    break
                
                if len(buffer) < size:
                    break
                
                msg_data = buffer[:size]
                buffer = buffer[size:]
                
                _, mtype, tag = wire.parse_header(msg_data)
                
                # Version must be handled inline (sets msize for framing)
                if mtype == wire.TVERSION:
                    await self._handle_version(msg_data, tag)
                    continue
                
                # Everything else: dispatch concurrently
                asyncio.create_task(
                    self._dispatch(msg_data, mtype, tag),
                    name=f"mux-dispatch-{self.conn_id}-{tag}"
                )
        
        await self._cleanup()
    
    async def _dispatch(self, data: bytes, mtype: int, tag: int):
        """Route a single client message."""
        try:
            if mtype == wire.TAUTH:
                await self._handle_auth(data, tag)
            elif mtype == wire.TATTACH:
                await self._handle_attach(data, tag)
            elif mtype == wire.TWALK:
                await self._handle_walk(data, tag)
            elif mtype == wire.TFLUSH:
                await self._handle_flush(data, tag)
            elif mtype == wire.TCLUNK:
                await self._handle_clunk(data, tag)
            elif mtype in (wire.TOPEN, wire.TREAD, wire.TWRITE,
                           wire.TSTAT, wire.TWSTAT, wire.TCREATE,
                           wire.TREMOVE):
                await self._handle_proxied(data, mtype, tag)
            else:
                await self._send_client(
                    wire.build_rerror(tag, f"Unknown message type {mtype}")
                )
        except Exception as e:
            logger.error(f"[{self.conn_id}] Dispatch error for {wire.msg_name(mtype)}: {e}")
            try:
                await self._send_client(wire.build_rerror(tag, str(e)))
            except Exception:
                pass
    
    # ── Version ──────────────────────────────────────────────────
    
    async def _handle_version(self, data: bytes, tag: int):
        """Handle Tversion — negotiate msize."""
        client_msize = struct.unpack_from('<I', data, 7)[0]
        self._msize = min(client_msize, 65536)
        await self._send_client(wire.build_rversion(tag, self._msize, "9P2000"))
    
    # ── Auth ─────────────────────────────────────────────────────
    
    async def _handle_auth(self, data: bytes, tag: int):
        """Auth not required."""
        await self._send_client(
            wire.build_rerror(tag, "Authentication not required")
        )
    
    # ── Attach ───────────────────────────────────────────────────
    
    async def _handle_attach(self, data: bytes, tag: int):
        """
        Attach — the client's fid becomes the virtual mux root.
        
        If aname is a backend name, attach directly to that backend
        (for clients that want to skip the virtual root).
        """
        fid, afid, uname, aname = wire.parse_tattach(data)
        
        if aname and aname in self._backend_configs:
            # Direct attach to a specific backend
            backend = await self._get_backend(aname)
            if backend is None:
                await self._send_client(
                    wire.build_rerror(tag, f"Backend '{aname}' unreachable")
                )
                return
            
            # Clone backend root and assign to client fid
            clone_fid = backend.alloc_fid()
            clone_ok = await self._backend_walk(
                backend, backend.root_fid, clone_fid, [], tag
            )
            if clone_ok is not None:
                self._fids[fid] = FidInfo(
                    kind="proxied", backend=aname,
                    backend_fid=clone_fid, path=f"/{aname}"
                )
                # Return the walk's qid (from Rwalk)
                # For simplicity, use a synthetic qid
                await self._send_client(
                    wire.build_rattach(tag, wire.QTDIR, 0,
                                       self._backend_qid_paths.get(aname, 0))
                )
            return
        
        # Default: attach to virtual root
        self._fids[fid] = FidInfo(kind=MUX_ROOT, path="/")
        await self._send_client(
            wire.build_rattach(tag, wire.QTDIR, 0, self._root_qid_path)
        )
    
    # ── Walk ─────────────────────────────────────────────────────
    
    async def _handle_walk(self, data: bytes, tag: int):
        """
        Walk — the heart of the multiplexer.
        
        Cases:
        1. Walk from mux root with empty names → clone root fid
        2. Walk from mux root into "ctl" → return ctl fid
        3. Walk from mux root into a backend name → connect + walk on backend
        4. Walk on a proxied fid → forward to backend
        """
        fid, newfid, names = wire.parse_twalk(data)
        
        if fid not in self._fids:
            await self._send_client(wire.build_rerror(tag, "Unknown fid"))
            return
        
        source = self._fids[fid]
        
        # ── Case: source is the virtual mux root ────────────
        if source.kind == MUX_ROOT:
            
            # Empty walk = clone fid
            if not names:
                self._fids[newfid] = FidInfo(kind=MUX_ROOT, path="/")
                await self._send_client(wire.build_rwalk(tag, []))
                return
            
            first = names[0]
            rest = names[1:]
            
            # ── Walk to ctl file ──────────────────────────────
            if first == "ctl":
                if rest:
                    # ctl is a file, can't walk into it
                    await self._send_client(
                        wire.build_rerror(tag, "Not a directory")
                    )
                    return
                
                self._fids[newfid] = FidInfo(kind=MUX_CTL, path="/ctl")
                qids = [(wire.QTFILE, 0, self._ctl_qid_path)]
                await self._send_client(wire.build_rwalk(tag, qids))
                return
            
            # ── Walk to a backend ─────────────────────────────
            # Check if first component is a backend name
            if first not in self._backend_configs:
                await self._send_client(
                    wire.build_rerror(tag, f"File not found: {first}")
                )
                return
            
            backend = await self._get_backend(first)
            if backend is None:
                await self._send_client(
                    wire.build_rerror(tag, f"Backend '{first}' unreachable")
                )
                return
            
            # Clone backend root, then walk the rest
            clone_fid = backend.alloc_fid()
            
            # First: clone root
            clone_result = await self._backend_walk_raw(
                backend, backend.root_fid, clone_fid, [], tag
            )
            if clone_result is None:
                return  # Error already sent
            
            # Build the qids list: first qid is for the backend dir
            qids = [(wire.QTDIR, 0, self._backend_qid_paths[first])]
            
            if rest:
                # Walk remaining components on the backend
                walk_fid = backend.alloc_fid()
                walk_result = await self._backend_walk_raw(
                    backend, clone_fid, walk_fid, rest, tag
                )
                if walk_result is None:
                    # Partial walk — the first step succeeded
                    # Return just the backend dir qid
                    await self._send_client(wire.build_rwalk(tag, qids))
                    self._fids[newfid] = FidInfo(
                        kind="proxied", backend=first,
                        backend_fid=clone_fid, path=f"/{first}"
                    )
                    return
                
                # Parse the Rwalk response to get qids
                _, resp_type, _ = wire.parse_header(walk_result)
                if resp_type == wire.RWALK:
                    nwqid = struct.unpack_from('<H', walk_result, 7)[0]
                    offset = 9
                    for i in range(nwqid):
                        qt = walk_result[offset]
                        qv = struct.unpack_from('<I', walk_result, offset + 1)[0]
                        qp = struct.unpack_from('<Q', walk_result, offset + 5)[0]
                        qids.append((qt, qv, qp))
                        offset += 13
                    
                    if nwqid == len(rest):
                        # Full walk succeeded
                        # Clunk the intermediate clone since walk_fid is the result
                        await self._backend_clunk(backend, clone_fid)
                        self._fids[newfid] = FidInfo(
                            kind="proxied", backend=first,
                            backend_fid=walk_fid,
                            path=f"/{first}/{'/'.join(rest)}"
                        )
                    else:
                        # Partial walk
                        await self._backend_clunk(backend, walk_fid)
                        self._fids[newfid] = FidInfo(
                            kind="proxied", backend=first,
                            backend_fid=clone_fid, path=f"/{first}"
                        )
                elif resp_type == wire.RERROR:
                    # Walk failed entirely on backend — just return the backend dir
                    await self._backend_clunk(backend, walk_fid)
                    await self._send_client(wire.build_rwalk(tag, qids))
                    self._fids[newfid] = FidInfo(
                        kind="proxied", backend=first,
                        backend_fid=clone_fid, path=f"/{first}"
                    )
                    return
            else:
                # Just walked to the backend root
                self._fids[newfid] = FidInfo(
                    kind="proxied", backend=first,
                    backend_fid=clone_fid, path=f"/{first}"
                )
            
            await self._send_client(wire.build_rwalk(tag, qids))
            return
        
        # ── Case: source is the ctl file ────────────────────
        if source.kind == MUX_CTL:
            if not names:
                # Clone ctl fid
                self._fids[newfid] = FidInfo(kind=MUX_CTL, path="/ctl")
                await self._send_client(wire.build_rwalk(tag, []))
            else:
                await self._send_client(
                    wire.build_rerror(tag, "Not a directory")
                )
            return
        
        # ── Case: source is a proxied fid ────────────────────
        if source.kind == "proxied":
            backend = self._backends.get(source.backend)
            if backend is None:
                await self._send_client(
                    wire.build_rerror(tag, f"Backend '{source.backend}' disconnected")
                )
                return
            
            if not names:
                # Clone fid
                clone_fid = backend.alloc_fid()
                result = await self._backend_walk_raw(
                    backend, source.backend_fid, clone_fid, [], tag
                )
                if result is not None:
                    _, resp_type, _ = wire.parse_header(result)
                    if resp_type == wire.RWALK:
                        self._fids[newfid] = FidInfo(
                            kind="proxied", backend=source.backend,
                            backend_fid=clone_fid, path=source.path
                        )
                        await self._send_client(wire.build_rwalk(tag, []))
                    else:
                        # Forward error
                        response = bytearray(result)
                        wire.set_tag(response, tag)
                        await self._send_client(bytes(response))
                return
            
            # Walk on the backend
            walk_fid = backend.alloc_fid()
            result = await self._backend_walk_raw(
                backend, source.backend_fid, walk_fid, names, tag
            )
            
            if result is None:
                return  # Error already sent
            
            _, resp_type, _ = wire.parse_header(result)
            
            if resp_type == wire.RWALK:
                # Parse qids from response
                nwqid = struct.unpack_from('<H', result, 7)[0]
                qids = []
                offset = 9
                for i in range(nwqid):
                    qt = result[offset]
                    qv = struct.unpack_from('<I', result, offset + 1)[0]
                    qp = struct.unpack_from('<Q', result, offset + 5)[0]
                    qids.append((qt, qv, qp))
                    offset += 13
                
                if nwqid > 0:
                    walked_names = names[:nwqid]
                    new_path = source.path.rstrip('/') + '/' + '/'.join(walked_names)
                    self._fids[newfid] = FidInfo(
                        kind="proxied", backend=source.backend,
                        backend_fid=walk_fid, path=new_path
                    )
                else:
                    # Walk returned 0 qids (shouldn't happen for non-empty names)
                    await self._backend_clunk(backend, walk_fid)
                
                await self._send_client(wire.build_rwalk(tag, qids))
            elif resp_type == wire.RERROR:
                # Forward error to client with correct tag
                response = bytearray(result)
                wire.set_tag(response, tag)
                await self._send_client(bytes(response))
            return
        
        await self._send_client(wire.build_rerror(tag, "Invalid fid state"))
    
    # ── Clunk ────────────────────────────────────────────────────
    
    async def _handle_clunk(self, data: bytes, tag: int):
        """
        Clunk — critical for correct behavior.
        
        For proxied fids, we forward the clunk to the backend so that
        the backend's clunk handlers fire (e.g., ParseFile executes
        accumulated code on close).
        
        For ctl fids, process any accumulated write buffer as a command.
        """
        fid = wire.get_fid(data, wire.TCLUNK)
        
        if fid not in self._fids:
            await self._send_client(wire.build_rerror(tag, "Unknown fid"))
            return
        
        info = self._fids.pop(fid)
        
        if info.kind == MUX_ROOT:
            await self._send_client(wire.build_rclunk(tag))
            return
        
        if info.kind == MUX_CTL:
            # Process accumulated writes on clunk (Plan 9 idiom)
            if info.ctl_write_buf:
                cmd = info.ctl_write_buf.decode('utf-8', errors='replace')
                for line in cmd.strip().splitlines():
                    await self._handle_ctl_command(line)
            await self._send_client(wire.build_rclunk(tag))
            return
        
        if info.kind == "proxied":
            backend = self._backends.get(info.backend)
            if backend:
                # Forward clunk to backend
                await self._proxy_to_backend(
                    backend, data, wire.TCLUNK, tag, info.backend_fid
                )
            else:
                await self._send_client(wire.build_rclunk(tag))
            return
        
        await self._send_client(wire.build_rclunk(tag))
    
    # ── Flush ────────────────────────────────────────────────────
    
    async def _handle_flush(self, data: bytes, tag: int):
        """
        Flush — forward to the correct backend.
        """
        oldtag = wire.get_flush_oldtag(data)
        
        # Find which backend the oldtag was sent to
        backend_name = self._tag_routes.pop(oldtag, None)
        
        if backend_name:
            backend = self._backends.get(backend_name)
            if backend:
                result = await backend.send_flush(tag, oldtag)
                if result is not None:
                    # The Rflush will come back through the backend callback
                    return
        
        # If we can't find the backend or tag, just respond
        await self._send_client(wire.build_rflush(tag))
    
    # ── Open/Read/Write/Stat/etc — proxied operations ────────────
    
    async def _handle_proxied(self, data: bytes, mtype: int, tag: int):
        """
        Handle operations on proxied fids by forwarding to the backend.
        
        For mux-root and ctl fids, handle locally.
        """
        fid = wire.get_fid(data, mtype)
        
        if fid not in self._fids:
            await self._send_client(wire.build_rerror(tag, "Unknown fid"))
            return
        
        info = self._fids[fid]
        
        # ── Handle ctl file operations locally ──────────────
        if info.kind == MUX_CTL:
            if mtype == wire.TOPEN:
                # Reset write buffer on open (like OTRUNC)
                info.ctl_write_buf = b""
                await self._send_client(
                    wire.build_ropen(tag, wire.QTFILE, 0,
                                     self._ctl_qid_path,
                                     self._msize - 24)
                )
            elif mtype == wire.TREAD:
                await self._handle_ctl_read(data, tag)
            elif mtype == wire.TWRITE:
                await self._handle_ctl_write(data, tag, info)
            elif mtype == wire.TSTAT:
                # ctl files traditionally report length 0 in Plan 9
                stat_data = wire.pack_stat(
                    "ctl", self._ctl_qid_path,
                    is_dir=False, length=0
                )
                await self._send_client(wire.build_rstat(tag, stat_data))
            elif mtype == wire.TCREATE:
                # ctl already exists — return error per 9P spec
                await self._send_client(
                    wire.build_rerror(tag, "file already exists")
                )
            elif mtype == wire.TWSTAT:
                # Accept wstat silently (truncate, etc.)
                body = bytearray()
                body += struct.pack('<IBH', 0, wire.RWSTAT, tag)
                struct.pack_into('<I', body, 0, len(body))
                await self._send_client(bytes(body))
            else:
                await self._send_client(
                    wire.build_rerror(tag, "Operation not supported on ctl")
                )
            return
        
        # Handle mux root operations locally
        if info.kind == MUX_ROOT:
            if mtype == wire.TOPEN:
                await self._send_client(
                    wire.build_ropen(tag, wire.QTDIR, 0, self._root_qid_path,
                                     self._msize - 24)
                )
            elif mtype == wire.TREAD:
                await self._handle_root_read(data, tag)
            elif mtype == wire.TSTAT:
                stat_data = wire.pack_stat("", self._root_qid_path, is_dir=True)
                await self._send_client(wire.build_rstat(tag, stat_data))
            elif mtype == wire.TWSTAT:
                # Accept wstat silently
                body = bytearray()
                body += struct.pack('<IBH', 0, wire.RWSTAT, tag)
                struct.pack_into('<I', body, 0, len(body))
                await self._send_client(bytes(body))
            else:
                await self._send_client(
                    wire.build_rerror(tag, "Operation not supported on mux root")
                )
            return
        
        # Proxied fid: forward to backend
        if info.kind == "proxied":
            backend = self._backends.get(info.backend)
            if backend is None:
                await self._send_client(
                    wire.build_rerror(tag, f"Backend '{info.backend}' disconnected")
                )
                return
            
            await self._proxy_to_backend(
                backend, data, mtype, tag, info.backend_fid
            )
            return
        
        await self._send_client(wire.build_rerror(tag, "Invalid fid state"))
    
    # ── Ctl file read/write ─────────────────────────────────────
    
    def _format_ctl_listing(self) -> bytes:
        """Format the ctl file content: list of current backends."""
        lines = []
        for name in sorted(self._backend_configs.keys()):
            host, port = self._backend_configs[name]
            lines.append(f"{name} {host}:{port}")
        return ('\n'.join(lines) + '\n').encode('utf-8') if lines else b''
    
    async def _handle_ctl_read(self, data: bytes, tag: int):
        """
        Read the ctl file — returns the current backend listing.
        
        Format (one per line):
            rio 127.0.0.1:5641
            llm 127.0.0.1:5640
        """
        offset = struct.unpack_from('<Q', data, 11)[0]
        count = struct.unpack_from('<I', data, 19)[0]
        
        content = self._format_ctl_listing()
        chunk = content[offset:offset + count]
        
        await self._send_client(wire.build_rread_dir(tag, chunk))
    
    async def _handle_ctl_write(self, data: bytes, tag: int, info: FidInfo):
        """
        Write to the ctl file — processes commands immediately.
        
        Commands:
            add <name> <host>:<port>
            remove <name>
        
        Each write can contain one command. The command is also
        executed immediately (not deferred to clunk).
        """
        # Parse Twrite: fid[4] offset[8] count[4] data[count]
        write_offset = struct.unpack_from('<Q', data, 11)[0]
        write_count = struct.unpack_from('<I', data, 19)[0]
        write_data = data[23:23 + write_count]
        
        # Accumulate in buffer (for multi-write commands)
        info.ctl_write_buf += write_data
        
        # Also process immediately line by line
        cmd = write_data.decode('utf-8', errors='replace').strip()
        if cmd:
            result = await self._handle_ctl_command(cmd)
            if result:
                logger.info(f"[{self.conn_id}] ctl result: {result.strip()}")
        
        # Respond with Rwrite (count of bytes written)
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.RWRITE, tag)
        body += struct.pack('<I', write_count)
        struct.pack_into('<I', body, 0, len(body))
        await self._send_client(bytes(body))
    
    async def _handle_root_read(self, data: bytes, tag: int):
        """
        Read the virtual mux root directory.
        
        Returns stat entries for each backend name as a subdirectory,
        plus a "ctl" file entry.
        """
        offset = struct.unpack_from('<Q', data, 11)[0]
        count = struct.unpack_from('<I', data, 19)[0]
        
        # Build directory listing
        listing = b''
        
        # ctl file entry (length=0, Plan 9 convention for ctl files)
        listing += wire.pack_stat("ctl", self._ctl_qid_path,
                                   is_dir=False, length=0)
        
        # Backend directory entries
        for name in sorted(self._backend_configs.keys()):
            qpath = self._backend_qid_paths[name]
            listing += wire.pack_stat(name, qpath, is_dir=True)
        
        # Slice for the requested range
        chunk = listing[offset:offset + count]
        
        await self._send_client(wire.build_rread_dir(tag, chunk))
    
    # ── Backend proxy helpers ────────────────────────────────────
    
    async def _proxy_to_backend(
        self,
        backend: BackendConnection,
        data: bytes,
        mtype: int,
        client_tag: int,
        backend_fid: int,
    ):
        """
        Rewrite fid(s) in the message and send to backend.
        
        The response comes back through the backend's reader loop
        and is tag-rewritten back to client space automatically.
        """
        msg = bytearray(data)
        
        # Rewrite primary fid
        wire.set_fid(msg, 7, backend_fid)
        
        # For Twalk, also rewrite newfid
        if mtype == wire.TWALK:
            # newfid at offset 11 — but we shouldn't get here for walks
            # (walks are handled in _handle_walk). Just in case:
            pass
        
        # Track which backend this tag goes to (for flush)
        self._tag_routes[client_tag] = backend.name
        
        # Send to backend (tag rewriting happens inside send())
        await backend.send(bytes(msg), client_tag)
    
    async def _backend_walk_raw(
        self,
        backend: BackendConnection,
        fid: int,
        newfid: int,
        names: list,
        client_tag: int,
    ) -> Optional[bytes]:
        """
        Send a Twalk to the backend and wait for the response.
        
        Returns the raw Rwalk/Rerror response bytes, or None on error.
        Uses a temporary internal tag so it doesn't conflict with the
        client's tag space.
        """
        # Build the Twalk
        # We need to use a backend-internal tag and wait for the response
        internal_tag = backend._next_tag
        backend._next_tag += 1
        if backend._next_tag >= 0xFFFE:
            backend._next_tag = 1
        
        msg = wire.build_twalk(internal_tag, fid, newfid, names)
        
        future = asyncio.get_event_loop().create_future()
        backend._pending[internal_tag] = future
        
        await backend._send_raw(msg)
        
        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        except asyncio.TimeoutError:
            backend._pending.pop(internal_tag, None)
            await self._send_client(
                wire.build_rerror(client_tag, "Backend walk timeout")
            )
            return None
        except Exception as e:
            backend._pending.pop(internal_tag, None)
            await self._send_client(
                wire.build_rerror(client_tag, f"Backend walk error: {e}")
            )
            return None
    
    async def _backend_walk(
        self,
        backend: BackendConnection,
        fid: int,
        newfid: int,
        names: list,
        client_tag: int,
    ) -> Optional[list]:
        """Walk on backend and return list of qid tuples, or None on error."""
        result = await self._backend_walk_raw(backend, fid, newfid, names, client_tag)
        if result is None:
            return None
        
        _, resp_type, _ = wire.parse_header(result)
        if resp_type == wire.RWALK:
            nwqid = struct.unpack_from('<H', result, 7)[0]
            qids = []
            offset = 9
            for i in range(nwqid):
                qt = result[offset]
                qv = struct.unpack_from('<I', result, offset + 1)[0]
                qp = struct.unpack_from('<Q', result, offset + 5)[0]
                qids.append((qt, qv, qp))
                offset += 13
            return qids
        
        return None
    
    async def _backend_clunk(self, backend: BackendConnection, fid: int):
        """Send a Tclunk to the backend (fire-and-forget internal)."""
        internal_tag = backend._next_tag
        backend._next_tag += 1
        if backend._next_tag >= 0xFFFE:
            backend._next_tag = 1
        
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.TCLUNK, internal_tag)
        body += struct.pack('<I', fid)
        struct.pack_into('<I', body, 0, len(body))
        
        future = asyncio.get_event_loop().create_future()
        backend._pending[internal_tag] = future
        
        try:
            await backend._send_raw(bytes(body))
            await asyncio.wait_for(future, timeout=2.0)
        except Exception:
            backend._pending.pop(internal_tag, None)
    
    # ── Backend management ───────────────────────────────────────
    
    async def _get_backend(self, name: str) -> Optional[BackendConnection]:
        """Get or create a backend connection."""
        if name in self._backends:
            if self._backends[name]._connected:
                return self._backends[name]
            # Reconnect
            await self._backends[name].close()
        
        if name not in self._backend_configs:
            return None
        
        host, port = self._backend_configs[name]
        
        backend = BackendConnection(
            name=name,
            host=host,
            port=port,
            response_callback=self._send_client,
        )
        
        if await backend.connect():
            self._backends[name] = backend
            return backend
        
        return None
    
    # ── Client I/O ───────────────────────────────────────────────
    
    async def _send_client(self, data: bytes):
        """Send raw bytes to the client."""
        if self._closed:
            return
        async with self._write_lock:
            try:
                self._writer.write(data)
                await self._writer.drain()
            except Exception as e:
                logger.error(f"[{self.conn_id}] Client send error: {e}")
                self._closed = True
    
    # ── Cleanup ──────────────────────────────────────────────────
    
    async def _cleanup(self):
        """Clean up all backend connections."""
        self._closed = True
        
        for name, backend in self._backends.items():
            try:
                await backend.close()
            except Exception as e:
                logger.warning(f"[{self.conn_id}] Backend '{name}' close error: {e}")
        
        self._backends.clear()
        self._fids.clear()
        self._tag_routes.clear()
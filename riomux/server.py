"""
riomux.server — TCP server that accepts 9P clients and multiplexes them.

Now supports dynamic backend addition/removal at runtime.

Usage:
    server = MuxServer(backends={"rio": ("127.0.0.1", 5641),
                                  "llm": ("127.0.0.1", 5640)})
    await server.serve(host="0.0.0.0", port=5642)

    # Later, dynamically:
    await server.add_backend("rio2", "127.0.0.1", 5643)
    await server.remove_backend("rio2")
"""

import asyncio
import logging
from typing import Dict, Tuple

from .mux import MuxConnection

logger = logging.getLogger("riomux.server")


class MuxServer:
    """
    9P multiplexer TCP server.
    
    Accepts client connections and creates a MuxConnection for each.
    Each client gets independent backend connections (9P requires
    per-connection fid space).
    
    Supports dynamic backend add/remove — changes propagate to all
    existing client connections (new walks will see the new backend;
    existing fids to removed backends remain valid until clunked).
    """
    
    def __init__(self, backends: Dict[str, Tuple[str, int]]):
        """
        Args:
            backends: Mapping of backend name → (host, port).
                      e.g. {"rio": ("127.0.0.1", 5641),
                            "llm": ("127.0.0.1", 5640)}
        """
        self.backends = dict(backends)  # mutable copy
        self._server: asyncio.Server = None
        self._conn_id = 0
        self._connections: Dict[int, MuxConnection] = {}
    
    async def serve(self, host: str = '0.0.0.0', port: int = 5642):
        """Start the multiplexer server."""
        self._server = await asyncio.start_server(
            self._handle_client,
            host, port,
        )
        
        addr = self._server.sockets[0].getsockname()
        logger.info(f"riomux listening on {addr[0]}:{addr[1]}")
        logger.info(f"Backends: {', '.join(f'{n}={h}:{p}' for n, (h, p) in self.backends.items())}")
        
        # Notify LLM backends about all initial machines
        await self._notify_llm_backends_machines_initial()
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self):
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        # Close all connections
        for conn in self._connections.values():
            await conn._cleanup()
        self._connections.clear()
    
    # ── Dynamic backend management ──────────────────────────────
    
    async def add_backend(self, name: str, host: str, port: int) -> bool:
        """
        Add a backend at runtime. Immediately visible to all clients
        on their next directory read or walk.
        
        Also notifies all LLM backends about the new machine via
        their ctl file: echo 'machine add <name>' > $llm/ctl
        
        Returns True if added, False if name already exists.
        """
        if name in self.backends:
            logger.warning(f"Backend '{name}' already exists ({self.backends[name]})")
            return False
        
        self.backends[name] = (host, port)
        
        # Propagate to all active connections
        for conn in self._connections.values():
            conn.add_backend(name, host, port)
        
        logger.info(f"Backend '{name}' added: {host}:{port} "
                     f"(propagated to {len(self._connections)} connections)")
        
        # Notify LLM backends about all machines
        await self._notify_llm_backends_machine_add(name)
        
        return True
    
    async def remove_backend(self, name: str) -> bool:
        """
        Remove a backend at runtime.
        
        Existing fids pointing to this backend remain valid until
        clunked (graceful drain). New walks won't find the backend.
        
        Also notifies LLM backends to unregister the machine.
        
        Returns True if removed, False if not found.
        """
        if name not in self.backends:
            logger.warning(f"Backend '{name}' not found")
            return False
        
        del self.backends[name]
        
        # Notify LLM backends before removing
        await self._notify_llm_backends_machine_remove(name)
        
        # Propagate to all active connections
        for conn in self._connections.values():
            await conn.remove_backend(name)
        
        logger.info(f"Backend '{name}' removed "
                     f"(propagated to {len(self._connections)} connections)")
        return True
    
    def list_backends(self) -> Dict[str, Tuple[str, int]]:
        """Return current backend configuration."""
        return dict(self.backends)
    
    async def _propagate_add(self, name: str, host: str, port: int,
                              exclude_conn: int = -1):
        """Propagate a backend addition from one connection's ctl to all others."""
        self.backends[name] = (host, port)
        for cid, conn in self._connections.items():
            if cid != exclude_conn:
                conn.add_backend(name, host, port)
        logger.info(f"Propagated add '{name}' → {host}:{port} "
                     f"to {len(self._connections) - 1} other connections")
        
        # Notify LLM backends about the new machine
        await self._notify_llm_backends_machine_add(name)
    
    async def _propagate_remove(self, name: str, exclude_conn: int = -1):
        """Propagate a backend removal from one connection's ctl to all others."""
        self.backends.pop(name, None)
        
        # Notify LLM backends before removing
        await self._notify_llm_backends_machine_remove(name)
        
        for cid, conn in self._connections.items():
            if cid != exclude_conn:
                await conn.remove_backend(name)
        logger.info(f"Propagated remove '{name}' "
                     f"to {len(self._connections) - 1} other connections")
    
    # ── LLM backend machine notifications ───────────────────────
    
    async def _notify_llm_backends_machine_add(self, machine_name: str):
        """
        Notify all LLM backends about a new machine.
        
        Writes 'machine add <name>' to each LLM backend's /ctl file.
        An "LLM backend" is detected by name convention — backends
        whose name contains 'llm' are treated as LLM filesystems.
        """
        for backend_name in list(self.backends.keys()):
            if 'llm' in backend_name.lower():
                await self._write_to_backend_ctl(
                    backend_name, f"machine add {machine_name}"
                )
    
    async def _notify_llm_backends_machine_remove(self, machine_name: str):
        """Notify all LLM backends that a machine was removed."""
        for backend_name in list(self.backends.keys()):
            if 'llm' in backend_name.lower():
                await self._write_to_backend_ctl(
                    backend_name, f"machine remove {machine_name}"
                )
    
    async def _notify_llm_backends_machines_initial(self):
        """
        On startup, notify LLM backends about all currently configured
        backends (machines). Called after all backends are known.
        """
        for backend_name in list(self.backends.keys()):
            if 'llm' not in backend_name.lower():
                continue
            # Tell each LLM backend about every other backend
            for machine_name in self.backends:
                if machine_name != backend_name:
                    await self._write_to_backend_ctl(
                        backend_name, f"machine add {machine_name}"
                    )
    
    async def _write_to_backend_ctl(self, backend_name: str, command: str):
        """
        Write a command to a backend's /ctl file via 9P.
        
        This opens a fresh connection, walks to /ctl, writes the
        command, and clunks. It's a fire-and-forget best-effort
        notification.
        """
        if backend_name not in self.backends:
            return
        
        host, port = self.backends[backend_name]
        
        try:
            from .backend import BackendConnection
            
            async def _noop(data: bytes):
                pass
            
            conn = BackendConnection(
                name=f"notify-{backend_name}",
                host=host,
                port=port,
                response_callback=_noop,
            )
            
            if not await conn.connect():
                logger.warning(f"Could not connect to {backend_name} for ctl notification")
                return
            
            # Walk to /ctl
            ctl_fid = conn.alloc_fid()
            qids = await self._backend_walk_for_notify(conn, conn.root_fid, ctl_fid, ["ctl"])
            if qids is None:
                await conn.close()
                return
            
            # Open for write
            await self._backend_open_for_notify(conn, ctl_fid)
            
            # Write command
            await self._backend_write_for_notify(conn, ctl_fid, command.encode('utf-8'))
            
            # Clunk (triggers processing)
            await self._backend_clunk_for_notify(conn, ctl_fid)
            
            await conn.close()
            logger.debug(f"Notified {backend_name}: {command}")
            
        except Exception as e:
            logger.warning(f"Failed to notify {backend_name} ctl: {e}")
    
    async def _backend_walk_for_notify(self, conn, fid, newfid, names):
        """Walk helper for ctl notification."""
        import struct
        from . import wire
        
        tag = conn._next_tag
        conn._next_tag += 1
        
        msg = wire.build_twalk(tag, fid, newfid, names)
        future = asyncio.get_event_loop().create_future()
        conn._pending[tag] = future
        await conn._send_raw(msg)
        
        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            _, mtype, _ = wire.parse_header(result)
            if mtype == wire.RWALK:
                return True
            return None
        except Exception:
            conn._pending.pop(tag, None)
            return None
    
    async def _backend_open_for_notify(self, conn, fid):
        """Open a fid for writing (OWRITE=1)."""
        import struct
        from . import wire
        
        tag = conn._next_tag
        conn._next_tag += 1
        
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.TOPEN, tag)
        body += struct.pack('<I', fid)
        body += struct.pack('<B', 1)  # OWRITE
        struct.pack_into('<I', body, 0, len(body))
        
        future = asyncio.get_event_loop().create_future()
        conn._pending[tag] = future
        await conn._send_raw(bytes(body))
        
        try:
            await asyncio.wait_for(future, timeout=5.0)
        except Exception:
            conn._pending.pop(tag, None)
    
    async def _backend_write_for_notify(self, conn, fid, data: bytes):
        """Write data to a fid."""
        import struct
        from . import wire
        
        tag = conn._next_tag
        conn._next_tag += 1
        
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.TWRITE, tag)
        body += struct.pack('<I', fid)
        body += struct.pack('<Q', 0)  # offset
        body += struct.pack('<I', len(data))
        body += data
        struct.pack_into('<I', body, 0, len(body))
        
        future = asyncio.get_event_loop().create_future()
        conn._pending[tag] = future
        await conn._send_raw(bytes(body))
        
        try:
            await asyncio.wait_for(future, timeout=5.0)
        except Exception:
            conn._pending.pop(tag, None)
    
    async def _backend_clunk_for_notify(self, conn, fid):
        """Clunk a fid."""
        import struct
        from . import wire
        
        tag = conn._next_tag
        conn._next_tag += 1
        
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.TCLUNK, tag)
        body += struct.pack('<I', fid)
        struct.pack_into('<I', body, 0, len(body))
        
        future = asyncio.get_event_loop().create_future()
        conn._pending[tag] = future
        await conn._send_raw(bytes(body))
        
        try:
            await asyncio.wait_for(future, timeout=2.0)
        except Exception:
            conn._pending.pop(tag, None)
    
    # ── Client handling ─────────────────────────────────────────
    
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle a new client connection."""
        self._conn_id += 1
        conn_id = self._conn_id
        
        peer = writer.get_extra_info('peername')
        logger.info(f"Client {conn_id} connected from {peer}")
        
        conn = MuxConnection(
            conn_id=conn_id,
            backend_configs=self.backends,
            reader=reader,
            writer=writer,
        )
        
        # Wire ctl callbacks so writes to /n/ctl propagate to all connections
        conn._on_backend_add = self._propagate_add
        conn._on_backend_remove = self._propagate_remove
        
        self._connections[conn_id] = conn
        
        try:
            await conn.serve()
        except Exception as e:
            logger.error(f"Client {conn_id} error: {e}", exc_info=True)
        finally:
            logger.info(f"Client {conn_id} disconnected")
            self._connections.pop(conn_id, None)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
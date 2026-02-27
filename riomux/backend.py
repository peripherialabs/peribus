"""
riomux.backend — Backend 9P connection pool.

Each client connection gets its own dedicated TCP connection to each
backend it touches. This is essential because 9P fid space is
per-connection — two clients can't share fids on the same backend conn.

The backend connection:
  1. Performs Tversion negotiation on first use
  2. Performs Tattach to get a root fid
  3. Proxies all subsequent messages with tag rewriting
  4. Handles response demultiplexing (multiple concurrent requests)

Tag rewriting:
  The client uses its own tag space. The backend uses its own.
  We maintain a mapping: backend_tag → client_tag so we can
  rewrite response tags before sending back to the client.
"""

import asyncio
import struct
import logging
from typing import Dict, Optional, Tuple, Callable, Awaitable

from . import wire

logger = logging.getLogger("riomux.backend")


class BackendConnection:
    """
    A single TCP connection to a backend 9P server, owned by one client.
    
    Handles:
    - Version negotiation
    - Tag allocation and mapping (backend_tag ↔ client_tag)
    - Fid allocation for backend-side fids
    - Response routing back to the owning client
    - Concurrent request handling via a reader task
    
    The response_callback is called with raw 9P response bytes
    (already tag-rewritten to client tag space).
    """
    
    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        response_callback: Callable[[bytes], Awaitable[None]],
    ):
        self.name = name
        self.host = host
        self.port = port
        self._response_callback = response_callback
        
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._negotiated = False
        self._msize = 65536
        
        # Tag mapping: backend_tag → client_tag
        self._tag_map: Dict[int, int] = {}
        self._next_tag = 1
        
        # Fid allocation for backend side
        self._next_fid = 100  # Start high to avoid confusion
        
        # Root fid on the backend (from Tattach)
        self.root_fid: Optional[int] = None
        
        # Reader task
        self._reader_task: Optional[asyncio.Task] = None
        
        # Pending requests: backend_tag → asyncio.Future
        # Used only for synchronous internal requests (version, attach)
        self._pending: Dict[int, asyncio.Future] = {}
        
        # Write lock to serialize sends to backend
        self._write_lock = asyncio.Lock()
        
        self._closed = False
    
    @property
    def msize(self) -> int:
        return self._msize
    
    def alloc_fid(self) -> int:
        """Allocate a new backend-side fid."""
        fid = self._next_fid
        self._next_fid += 1
        return fid
    
    def alloc_tag(self, client_tag: int) -> int:
        """Allocate a backend tag and map it to the client tag."""
        tag = self._next_tag
        self._next_tag += 1
        if self._next_tag >= 0xFFFE:
            self._next_tag = 1
        self._tag_map[tag] = client_tag
        return tag
    
    def _find_backend_tag(self, client_tag: int) -> Optional[int]:
        """Find the backend tag for a client tag (for flush)."""
        for bt, ct in self._tag_map.items():
            if ct == client_tag:
                return bt
        return None
    
    async def connect(self) -> bool:
        """Establish TCP connection and negotiate version."""
        if self._connected:
            return True
        
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=5.0
            )
            self._connected = True
            
            # Start reader task
            self._reader_task = asyncio.create_task(
                self._read_loop(),
                name=f"backend-reader-{self.name}"
            )
            
            # Negotiate version
            await self._negotiate_version()
            
            # Attach
            await self._attach()
            
            logger.info(
                f"Backend '{self.name}' connected to {self.host}:{self.port} "
                f"(msize={self._msize}, root_fid={self.root_fid})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Backend '{self.name}' connect failed: {e}")
            self._connected = False
            return False
    
    async def _negotiate_version(self):
        """Send Tversion, wait for Rversion."""
        tag = wire.NOTAG
        msg = wire.build_rversion(tag, self._msize, "9P2000")
        # Tversion has the same format as Rversion for the body,
        # but different type byte. Build manually:
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.TVERSION, tag)
        body += struct.pack('<I', self._msize)
        ver = b"9P2000"
        body += struct.pack('<H', len(ver)) + ver
        struct.pack_into('<I', body, 0, len(body))
        
        future = asyncio.get_event_loop().create_future()
        self._pending[tag] = future
        
        await self._send_raw(bytes(body))
        
        response = await asyncio.wait_for(future, timeout=5.0)
        
        # Parse Rversion
        _, mtype, _ = wire.parse_header(response)
        if mtype == wire.RVERSION:
            backend_msize = struct.unpack_from('<I', response, 7)[0]
            self._msize = min(self._msize, backend_msize)
            self._negotiated = True
        elif mtype == wire.RERROR:
            ename_len = struct.unpack_from('<H', response, 7)[0]
            ename = response[9:9 + ename_len].decode('utf-8')
            raise ConnectionError(f"Backend version failed: {ename}")
    
    async def _attach(self):
        """Send Tattach to get root fid."""
        self.root_fid = self.alloc_fid()
        tag = self._next_tag
        self._next_tag += 1
        
        uname = b"mux"
        aname = b""
        
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.TATTACH, tag)
        body += struct.pack('<I', self.root_fid)
        body += struct.pack('<I', wire.NOFID)  # afid
        body += struct.pack('<H', len(uname)) + uname
        body += struct.pack('<H', len(aname)) + aname
        struct.pack_into('<I', body, 0, len(body))
        
        future = asyncio.get_event_loop().create_future()
        self._pending[tag] = future
        
        await self._send_raw(bytes(body))
        
        response = await asyncio.wait_for(future, timeout=5.0)
        
        _, mtype, _ = wire.parse_header(response)
        if mtype == wire.RATTACH:
            logger.debug(f"Backend '{self.name}' attached, root_fid={self.root_fid}")
        elif mtype == wire.RERROR:
            ename_len = struct.unpack_from('<H', response, 7)[0]
            ename = response[9:9 + ename_len].decode('utf-8')
            raise ConnectionError(f"Backend attach failed: {ename}")
    
    async def send(self, data: bytes, client_tag: int) -> int:
        """
        Send a message to the backend with tag rewriting.
        
        Rewrites the client's tag to a backend tag.
        Returns the backend tag used.
        """
        msg = bytearray(data)
        
        mtype = wire.get_type(msg)
        
        # Allocate backend tag and rewrite
        backend_tag = self.alloc_tag(client_tag)
        wire.set_tag(msg, backend_tag)
        
        await self._send_raw(bytes(msg))
        return backend_tag
    
    async def send_flush(self, client_tag: int, client_oldtag: int) -> Optional[int]:
        """
        Send a Tflush to the backend.
        
        Finds the backend tag for client_oldtag and sends a flush
        with a new backend tag. Returns the backend tag used, or
        None if the oldtag wasn't found.
        """
        backend_oldtag = self._find_backend_tag(client_oldtag)
        if backend_oldtag is None:
            return None
        
        backend_tag = self.alloc_tag(client_tag)
        
        body = bytearray()
        body += struct.pack('<IBH', 0, wire.TFLUSH, backend_tag)
        body += struct.pack('<H', backend_oldtag)
        struct.pack_into('<I', body, 0, len(body))
        
        await self._send_raw(bytes(body))
        return backend_tag
    
    async def _send_raw(self, data: bytes):
        """Send raw bytes to the backend."""
        if not self._connected or self._writer is None:
            raise ConnectionError(f"Backend '{self.name}' not connected")
        
        async with self._write_lock:
            self._writer.write(data)
            await self._writer.drain()
    
    async def _read_loop(self):
        """
        Read responses from the backend and route them.
        
        Internal requests (version, attach) are routed to pending futures.
        Proxied requests are tag-rewritten and sent back via the callback.
        """
        buffer = b''
        
        while not self._closed and self._reader:
            try:
                chunk = await self._reader.read(65536)
                if not chunk:
                    logger.info(f"Backend '{self.name}' disconnected")
                    break
                buffer += chunk
            except (ConnectionError, asyncio.CancelledError):
                break
            
            # Process complete messages
            while len(buffer) >= 4:
                size = struct.unpack_from('<I', buffer, 0)[0]
                if size > self._msize + 256:  # Some slack
                    logger.error(f"Backend '{self.name}': message too large ({size})")
                    buffer = b''
                    break
                
                if len(buffer) < size:
                    break
                
                msg_data = buffer[:size]
                buffer = buffer[size:]
                
                _, mtype, tag = wire.parse_header(msg_data)
                
                # Check if this is a response to an internal request
                if tag in self._pending:
                    future = self._pending.pop(tag)
                    if not future.done():
                        future.set_result(msg_data)
                    continue
                
                # Rewrite tag back to client space
                client_tag = self._tag_map.pop(tag, None)
                if client_tag is None:
                    logger.warning(
                        f"Backend '{self.name}': response for unknown tag {tag} "
                        f"({wire.msg_name(mtype)}), dropping"
                    )
                    continue
                
                # Rewrite the tag in the response
                response = bytearray(msg_data)
                wire.set_tag(response, client_tag)
                
                # Send to client
                try:
                    await self._response_callback(bytes(response))
                except Exception as e:
                    logger.error(f"Backend '{self.name}': callback error: {e}")
        
        self._connected = False
    
    async def close(self):
        """Close the backend connection."""
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
        
        # Fail any pending futures
        for tag, future in self._pending.items():
            if not future.done():
                future.set_exception(ConnectionError("Backend closed"))
        self._pending.clear()
        self._tag_map.clear()
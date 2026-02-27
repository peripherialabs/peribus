"""
ninep.client - Async 9P2000 Client Library for LLMFS

This provides a high-level Python client for LLMFS that completely bypasses
the need for mounting via `mount -t 9p` or `9pfuse`. All operations happen
over a direct TCP connection using the 9P protocol.

Advantages over filesystem mounting:
  - No sudo required
  - No mount/umount commands
  - Better streaming behavior (no VFS buffering)
  - Direct control over connections and lifecycle
  - Easier to manage multiple agents programmatically
  - Clean Python API with async/await
  - Proper error handling and reconnection logic

Usage:
    async with LLMFSClient() as client:
        # Create agent
        agent = await client.create_agent('claude')
        
        # Send prompt and stream response
        async for chunk in agent.prompt("Hello!"):
            print(chunk, end='', flush=True)
        
        # Set up routing
        await agent.route_output('/n/rioa/scene/parse')
"""

import asyncio
import struct
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, AsyncIterator
from enum import IntEnum


# =============================================================================
# 9P2000 Protocol Constants
# =============================================================================

class MessageType(IntEnum):
    """9P message types"""
    Tversion = 100
    Rversion = 101
    Tauth = 102
    Rauth = 103
    Tattach = 104
    Rattach = 105
    Terror = 106  # illegal
    Rerror = 107
    Tflush = 108
    Rflush = 109
    Twalk = 110
    Rwalk = 111
    Topen = 112
    Ropen = 113
    Tcreate = 114
    Rcreate = 115
    Tread = 116
    Rread = 117
    Twrite = 118
    Rwrite = 119
    Tclunk = 120
    Rclunk = 121
    Tremove = 122
    Rremove = 123
    Tstat = 124
    Rstat = 125
    Twstat = 126
    Rwstat = 127


class OpenMode(IntEnum):
    """9P open modes"""
    OREAD = 0
    OWRITE = 1
    ORDWR = 2
    OEXEC = 3
    OTRUNC = 0x10


NOTAG = 0xFFFF
NOFID = 0xFFFFFFFF


# =============================================================================
# Exceptions
# =============================================================================

class P9Error(Exception):
    """Error from 9P server (Rerror message)"""
    pass


class ConnectionError(Exception):
    """Connection to 9P server lost"""
    pass


# =============================================================================
# Low-Level 9P Client
# =============================================================================

@dataclass
class Fid:
    """File identifier"""
    fid: int
    qid: Optional[bytes] = None


class P9Client:
    """
    Low-level async 9P2000 client.
    
    Handles the 9P wire protocol over TCP. This is the foundation
    for the higher-level LLMFS client.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5640):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.msize = 8192
        self._tag = 0
        self._next_fid_num = 1
        self._root_fid = 0
        self._fid_cache: Dict[str, Fid] = {}
        self._cache_lock = asyncio.Lock()   # Protects fid cache & alloc
        self._write_lock = asyncio.Lock()   # Serializes sends on the socket
        self._pending: Dict[int, asyncio.Future] = {}  # tag -> Future for response
        self._reader_task: Optional[asyncio.Task] = None
    
    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    
    async def connect(self):
        """Connect to 9P server and perform handshake"""
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port
        )
        
        # Disable Nagle for low latency
        sock = self.writer.get_extra_info('socket')
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        # Protocol version negotiation (before reader task starts)
        await self._version()
        
        # Start background reader that demuxes responses by tag
        self._reader_task = asyncio.ensure_future(self._reader_loop())
        
        # Attach to filesystem root
        await self._attach()
    
    async def disconnect(self):
        """Close connection and clunk all fids"""
        if self.writer is None:
            return
        
        # Stop reader task
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        
        # Cancel any pending RPCs
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        
        # Clunk all open fids
        # (skip since reader is stopped — just close socket)
        self._fid_cache.clear()
        
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass
        
        self.reader = None
        self.writer = None
    
    @property
    def connected(self) -> bool:
        """Check if connected"""
        return self.writer is not None and not self.writer.is_closing()
    
    # -------------------------------------------------------------------------
    # High-Level Operations
    # -------------------------------------------------------------------------
    
    async def walk_open(self, path: str, mode: int = OpenMode.OREAD) -> Fid:
        """
        Walk to path and open it.
        
        Caches fids so repeated calls for the same path reuse the fid.
        
        Args:
            path: Path relative to root (e.g., "agents/claude/input")
            mode: Open mode (OREAD, OWRITE, ORDWR)
        
        Returns:
            Fid object
        """
        async with self._cache_lock:
            cache_key = f"{path}:{mode}"
            
            if cache_key in self._fid_cache:
                return self._fid_cache[cache_key]
            
            # Allocate new fid
            fid = self._alloc_fid()
            
            # Parse path into components
            elements = [e for e in path.split("/") if e]
            
            # Twalk from root
            qids = await self._walk(self._root_fid, fid.fid, elements)
            if qids:
                fid.qid = qids[-1]
            
            # Topen
            await self._open(fid.fid, mode)
            
            # Cache the fid
            self._fid_cache[cache_key] = fid
            
            return fid
    
    async def read(self, fid: Fid, offset: int, count: int = 0) -> bytes:
        """
        Read from an open file.
        
        Args:
            fid: File identifier
            offset: Byte offset to read from
            count: Maximum bytes to read (0 = use default)
        
        Returns:
            Data read (may be shorter than count, empty on EOF)
        """
        if count <= 0:
            count = self.msize - 24  # Leave room for header
        
        payload = struct.pack("<IQI", fid.fid, offset, count)
        response = await self._rpc(MessageType.Tread, payload)
        
        rtype = response[0]
        if rtype == MessageType.Rerror:
            self._parse_error(response)
        
        # Rread: type[1] tag[2] count[4] data[count]
        data_count = struct.unpack_from("<I", response, 3)[0]
        return response[7 : 7 + data_count]
    
    async def write(self, fid: Fid, offset: int, data: bytes) -> int:
        """
        Write to an open file.
        
        Args:
            fid: File identifier
            offset: Byte offset to write to
            data: Data to write (must fit in a single 9P message)
        
        Returns:
            Number of bytes written
        """
        payload = struct.pack("<IQI", fid.fid, offset, len(data))
        payload += data
        
        response = await self._rpc(MessageType.Twrite, payload)
        
        rtype = response[0]
        if rtype == MessageType.Rerror:
            self._parse_error(response)
        
        # Rwrite: type[1] tag[2] count[4]
        return struct.unpack_from("<I", response, 3)[0]
    
    async def write_all(self, fid: Fid, offset: int, data: bytes) -> int:
        """
        Write all data, chunking if necessary to fit within msize.
        
        Returns:
            Total number of bytes written
        """
        # Twrite header overhead: size[4] type[1] tag[2] fid[4] offset[8] count[4] = 23
        max_chunk = self.msize - 23 - 4  # extra safety margin
        total = 0
        while data:
            chunk = data[:max_chunk]
            data = data[max_chunk:]
            written = await self.write(fid, offset + total, chunk)
            total += written
        return total
    
    async def clunk_path(self, path: str, mode: int = OpenMode.OREAD):
        """Close a previously opened path"""
        async with self._cache_lock:
            cache_key = f"{path}:{mode}"
            fid = self._fid_cache.pop(cache_key, None)
            
            if fid:
                try:
                    await self._clunk(fid.fid)
                except Exception:
                    pass
    
    # -------------------------------------------------------------------------
    # 9P Protocol Primitives
    # -------------------------------------------------------------------------
    
    async def _version(self):
        """Negotiate protocol version (called before reader loop starts)"""
        version = b"9P2000"
        payload = struct.pack("<I", self.msize)
        payload += struct.pack("<H", len(version)) + version
        
        response = await self._rpc_inline(MessageType.Tversion, payload, tag=NOTAG)
        
        # Parse Rversion
        server_msize = struct.unpack_from("<I", response, 3)[0]
        self.msize = min(self.msize, server_msize)
    
    async def _attach(self):
        """Attach to filesystem root"""
        uname = b"rio"
        aname = b""
        
        payload = struct.pack("<II", self._root_fid, NOFID)
        payload += struct.pack("<H", len(uname)) + uname
        payload += struct.pack("<H", len(aname)) + aname
        
        response = await self._rpc(MessageType.Tattach, payload)
        
        rtype = response[0]
        if rtype == MessageType.Rerror:
            self._parse_error(response)
    
    async def _walk(self, fid: int, newfid: int, wnames: List[str]) -> List[bytes]:
        """Walk from fid to newfid following wnames"""
        payload = struct.pack("<II", fid, newfid)
        payload += struct.pack("<H", len(wnames))
        
        for name in wnames:
            name_bytes = name.encode("utf-8")
            payload += struct.pack("<H", len(name_bytes)) + name_bytes
        
        response = await self._rpc(MessageType.Twalk, payload)
        
        rtype = response[0]
        if rtype == MessageType.Rerror:
            self._parse_error(response)
        
        # Parse Rwalk: type[1] tag[2] nwqid[2] qids...
        nwqid = struct.unpack_from("<H", response, 3)[0]
        
        qids = []
        offset = 5
        for _ in range(nwqid):
            qid = response[offset : offset + 13]
            qids.append(qid)
            offset += 13
        
        return qids
    
    async def _open(self, fid: int, mode: int):
        """Open a file"""
        payload = struct.pack("<IB", fid, mode)
        response = await self._rpc(MessageType.Topen, payload)
        
        rtype = response[0]
        if rtype == MessageType.Rerror:
            self._parse_error(response)
    
    async def _clunk(self, fid: int):
        """Close a fid"""
        payload = struct.pack("<I", fid)
        await self._rpc(MessageType.Tclunk, payload)
    
    # -------------------------------------------------------------------------
    # Wire Protocol
    # -------------------------------------------------------------------------
    
    def _alloc_fid(self) -> Fid:
        """Allocate a new fid"""
        fid_num = self._next_fid_num
        self._next_fid_num += 1
        return Fid(fid_num)
    
    def _next_tag(self) -> int:
        """Get next tag"""
        self._tag = (self._tag + 1) & 0x7FFF
        return self._tag
    
    async def _rpc(self, msg_type: int, payload: bytes, tag: int = None) -> bytes:
        """Send T-message and receive R-message.
        
        Uses tag-based demultiplexing so multiple RPCs can be in-flight
        concurrently (e.g. a blocking Tread on output + a Twrite to input).
        
        The _write_lock serializes sends; a background _reader_loop
        dispatches each response to the Future registered for its tag.
        """
        if not self.connected:
            raise ConnectionError("Not connected to 9P server")
        
        if tag is None:
            tag = self._next_tag()
        
        # Register a future for this tag
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[tag] = fut
        
        try:
            # Build and send message (serialized)
            size = 4 + 1 + 2 + len(payload)
            header = struct.pack("<IBH", size, msg_type, tag)
            
            async with self._write_lock:
                self.writer.write(header + payload)
                await self.writer.drain()
            
            # Wait for the reader loop to deliver our response
            body = await fut
            return body
        except asyncio.CancelledError:
            self._pending.pop(tag, None)
            raise
        finally:
            self._pending.pop(tag, None)
    
    async def _rpc_inline(self, msg_type: int, payload: bytes, tag: int) -> bytes:
        """Send T-message and read R-message inline (before reader loop starts).
        
        Used only during version negotiation when no reader task is running.
        """
        size = 4 + 1 + 2 + len(payload)
        header = struct.pack("<IBH", size, msg_type, tag)
        
        self.writer.write(header + payload)
        await self.writer.drain()
        
        size_bytes = await self._read_exact(4)
        resp_size = struct.unpack("<I", size_bytes)[0]
        body = await self._read_exact(resp_size - 4)
        return body
    
    async def _reader_loop(self):
        """Background task: read responses and dispatch by tag."""
        try:
            while self.connected:
                # Read message size
                size_bytes = await self._read_exact(4)
                size = struct.unpack("<I", size_bytes)[0]
                
                # Read message body
                body = await self._read_exact(size - 4)
                
                # Extract tag: body[0]=type, body[1:3]=tag
                if len(body) < 3:
                    continue
                resp_tag = struct.unpack_from("<H", body, 1)[0]
                
                # Deliver to waiting future
                fut = self._pending.get(resp_tag)
                if fut and not fut.done():
                    fut.set_result(body)
                else:
                    # Unsolicited or late response — ignore
                    pass
        except (asyncio.CancelledError, asyncio.IncompleteReadError,
                ConnectionError, OSError):
            pass
        except Exception as e:
            # Cancel all pending futures on unexpected error
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(e)
    
    async def _read_exact(self, n: int) -> bytes:
        """Read exactly n bytes"""
        data = await self.reader.readexactly(n)
        return data
    
    def _parse_error(self, response: bytes):
        """Parse Rerror and raise exception"""
        # Rerror: type[1] tag[2] ename[s]
        ename_len = struct.unpack_from("<H", response, 3)[0]
        ename = response[5 : 5 + ename_len].decode("utf-8", errors="replace")
        raise P9Error(ename)


# =============================================================================
# High-Level LLMFS Client
# =============================================================================

class Agent:
    """
    High-level agent interface.
    
    Represents a single LLMFS agent with convenient methods for:
    - Sending prompts
    - Streaming responses
    - Modifying configuration
    - Setting up routes
    """
    
    def __init__(self, client: 'LLMFSClient', name: str):
        self.client = client
        self.name = name
        self._base_path = f"agents/{name}"
    
    async def prompt(self, text: str, stream: bool = True) -> AsyncIterator[str]:
        """
        Send a prompt and stream the response.
        
        Args:
            text: Prompt text
            stream: If True, yield chunks as they arrive; if False, return complete response
        
        Yields:
            Response chunks as they arrive
        """
        # Write to input
        input_path = f"{self._base_path}/input"
        fid = await self.client.p9.walk_open(input_path, OpenMode.OWRITE)
        await self.client.p9.write(fid, 0, text.encode("utf-8"))
        await self.client.p9.clunk_path(input_path, OpenMode.OWRITE)
        
        # Read from output
        output_path = f"{self._base_path}/output"
        output_fid = await self.client.p9.walk_open(output_path, OpenMode.OREAD)
        
        offset = 0
        full_response = []
        
        while True:
            chunk = await self.client.p9.read(output_fid, offset, 4096)
            
            if not chunk:
                # EOF - generation complete
                break
            
            text_chunk = chunk.decode("utf-8", errors="replace")
            full_response.append(text_chunk)
            
            if stream:
                yield text_chunk
            
            offset += len(chunk)
        
        await self.client.p9.clunk_path(output_path, OpenMode.OREAD)
        
        if not stream:
            yield "".join(full_response)
    
    async def set_system(self, prompt: str):
        """Set system prompt (handles large prompts via chunked write)"""
        path = f"{self._base_path}/system"
        fid = await self.client.p9.walk_open(path, OpenMode.OWRITE)
        await self.client.p9.write_all(fid, 0, prompt.encode("utf-8"))
        await self.client.p9.clunk_path(path, OpenMode.OWRITE)
    
    async def set_model(self, model: str):
        """Change model"""
        await self._ctl(f"model {model}")
    
    async def set_temperature(self, temp: float):
        """Set temperature"""
        await self._ctl(f"temperature {temp}")
    
    async def clear_history(self):
        """Clear conversation history"""
        await self._ctl("clear")
    
    async def cancel(self):
        """Cancel current generation"""
        await self._ctl("cancel")
    
    async def get_config(self) -> str:
        """Get agent configuration"""
        return await self._read_file("config")
    
    async def get_history(self) -> str:
        """Get conversation history"""
        return await self._read_file("history")
    
    async def get_errors(self) -> str:
        """Get error log"""
        return await self._read_file("errors")
    
    async def route_to(self, destination: str):
        """
        Route agent output to a destination.
        
        This uses the LLMFS routes system to automatically pipe
        all agent output to the specified destination.
        
        Args:
            destination: Destination path (e.g., '/n/rioa/scene/parse')
        """
        # Add route via routes/ctl
        route_ctl = "routes/ctl"
        fid = await self.client.p9.walk_open(route_ctl, OpenMode.OWRITE)
        cmd = f"add {self._base_path}/output {destination}\n"
        await self.client.p9.write(fid, 0, cmd.encode("utf-8"))
        await self.client.p9.clunk_path(route_ctl, OpenMode.OWRITE)
    
    async def unroute(self):
        """Remove all routes for this agent"""
        route_ctl = "routes/ctl"
        fid = await self.client.p9.walk_open(route_ctl, OpenMode.OWRITE)
        cmd = f"remove {self._base_path}/output\n"
        await self.client.p9.write(fid, 0, cmd.encode("utf-8"))
        await self.client.p9.clunk_path(route_ctl, OpenMode.OWRITE)
    
    async def _ctl(self, command: str):
        """Send command to agent ctl file"""
        path = f"{self._base_path}/ctl"
        fid = await self.client.p9.walk_open(path, OpenMode.OWRITE)
        await self.client.p9.write(fid, 0, (command + "\n").encode("utf-8"))
        await self.client.p9.clunk_path(path, OpenMode.OWRITE)
    
    async def _read_file(self, filename: str) -> str:
        """Read a file from agent directory"""
        path = f"{self._base_path}/{filename}"
        fid = await self.client.p9.walk_open(path, OpenMode.OREAD)
        
        data = []
        offset = 0
        
        while True:
            chunk = await self.client.p9.read(fid, offset)
            if not chunk:
                break
            data.append(chunk)
            offset += len(chunk)
        
        await self.client.p9.clunk_path(path, OpenMode.OREAD)
        
        return b"".join(data).decode("utf-8", errors="replace")


class LLMFSClient:
    """
    High-level client for LLMFS.
    
    This is the main entry point for interacting with LLMFS without
    requiring any filesystem mounts.
    
    Usage:
        async with LLMFSClient() as client:
            agent = await client.create_agent('claude')
            async for chunk in agent.prompt("Hello!"):
                print(chunk, end='')
    """
    
    def __init__(self, host: str = "localhost", port: int = 5640):
        self.host = host
        self.port = port
        self.p9 = P9Client(host, port)
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, *args):
        await self.disconnect()
    
    async def connect(self):
        """Connect to LLMFS server"""
        await self.p9.connect()
    
    async def disconnect(self):
        """Disconnect from LLMFS server"""
        await self.p9.disconnect()
    
    async def create_agent(
        self,
        name: str,
        provider: str = None,
        model: str = None,
        system: str = None
    ) -> Agent:
        """
        Create a new agent.
        
        Args:
            name: Agent name
            provider: LLM provider (claude, openai, gemini, etc.)
            model: Model name
            system: System prompt
        
        Returns:
            Agent object
        """
        # Build command
        cmd = f"new {name}"
        if provider:
            cmd += f" {provider}"
        if model:
            cmd += f" {model}"
        
        # Write to ctl
        fid = await self.p9.walk_open("ctl", OpenMode.OWRITE)
        await self.p9.write(fid, 0, (cmd + "\n").encode("utf-8"))
        await self.p9.clunk_path("ctl", OpenMode.OWRITE)
        
        # Create agent object
        agent = Agent(self, name)
        
        # Set system prompt if provided
        if system:
            await agent.set_system(system)
        
        return agent
    
    async def get_agent(self, name: str) -> Agent:
        """Get an existing agent"""
        return Agent(self, name)
    
    async def list_agents(self) -> List[str]:
        """List all agents"""
        # Read from ctl to get agent list
        fid = await self.p9.walk_open("ctl", OpenMode.OREAD)
        data = await self.p9.read(fid, 0)
        await self.p9.clunk_path("ctl", OpenMode.OREAD)
        
        # Parse agent list
        text = data.decode("utf-8", errors="replace")
        agents = []
        
        for line in text.split("\n"):
            if line.startswith("agent "):
                parts = line.split()
                if len(parts) >= 2:
                    agents.append(parts[1])
        
        return agents
    
    async def delete_agent(self, name: str):
        """Delete an agent"""
        cmd = f"delete {name}\n"
        fid = await self.p9.walk_open("ctl", OpenMode.OWRITE)
        await self.p9.write(fid, 0, cmd.encode("utf-8"))
        await self.p9.clunk_path("ctl", OpenMode.OWRITE)


# =============================================================================
# Streaming Helper
# =============================================================================

class StreamReader:
    """
    Helper for continuously reading from an agent output stream.
    
    This maintains a persistent connection and automatically handles
    the generation lifecycle (reset, stream, EOF, repeat).
    
    Usage:
        reader = StreamReader(client, 'agents/claude/output')
        async for chunk in reader.stream():
            print(chunk, end='')
    """
    
    def __init__(self, client: LLMFSClient, path: str):
        self.client = client
        self.path = path
        self._running = False
    
    async def stream(self) -> AsyncIterator[str]:
        """
        Continuously read from the stream.
        
        Yields chunks as they arrive. Automatically handles generation
        lifecycle: waits for new generation, streams it, then waits again.
        """
        self._running = True
        
        while self._running:
            # Open the output file
            fid = await self.client.p9.walk_open(self.path, OpenMode.OREAD)
            offset = 0
            
            # Stream until EOF
            while self._running:
                chunk = await self.client.p9.read(fid, offset, 4096)
                
                if not chunk:
                    # EOF - generation complete
                    await self.client.p9.clunk_path(self.path, OpenMode.OREAD)
                    break
                
                text = chunk.decode("utf-8", errors="replace")
                yield text
                offset += len(chunk)
            
            # Next iteration will block on walk_open until next generation
    
    def stop(self):
        """Stop streaming"""
        self._running = False


# =============================================================================
# Example Usage
# =============================================================================

async def example_basic():
    """Basic example: create agent, send prompt, get response"""
    async with LLMFSClient() as client:
        # Create agent
        agent = await client.create_agent('claude', system='You are helpful')
        
        # Send prompt and stream response
        print("Prompt: What is 2+2?")
        print("Response: ", end='')
        
        async for chunk in agent.prompt("What is 2+2?"):
            print(chunk, end='', flush=True)
        
        print("\n")


async def example_routes():
    """Example: set up automatic routing"""
    async with LLMFSClient() as client:
        agent = await client.create_agent('coder')
        
        # Route all output to a destination
        await agent.route_to('/n/rioa/scene/parse')
        
        # Now all agent output automatically goes to that destination
        async for chunk in agent.prompt("Write some code"):
            print(chunk, end='', flush=True)


async def example_multiple_agents():
    """Example: manage multiple agents"""
    async with LLMFSClient() as client:
        # Create multiple agents
        claude = await client.create_agent('claude', provider='claude')
        gpt = await client.create_agent('gpt', provider='openai')
        
        # Send same prompt to both
        prompt = "Explain quantum computing in one sentence"
        
        print("Claude says: ", end='')
        async for chunk in claude.prompt(prompt):
            print(chunk, end='', flush=True)
        print("\n")
        
        print("GPT says: ", end='')
        async for chunk in gpt.prompt(prompt):
            print(chunk, end='', flush=True)
        print("\n")


if __name__ == "__main__":
    # Run basic example
    asyncio.run(example_basic())
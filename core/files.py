"""
Synthetic file implementations for Plan 9-style filesystems.

These classes provide the building blocks for creating file servers
that expose arbitrary functionality through the file interface.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from .types import Qid, Stat, FidState, QTDIR, QTFILE, DMDIR


class SyntheticFile(ABC):
    """
    Base class for all synthetic files.
    
    A synthetic file is a file-like interface to some underlying
    functionality. It supports read/write operations and can be
    statted like a regular file.
    """
    
    def __init__(self, name: str, parent: 'SyntheticDir' = None):
        self.name = name
        self.parent = parent
        self._qid = Qid(type=QTFILE, version=0, path=id(self))
        self._mode = 0o666
        self._mtime = int(time.time())
        self._atime = int(time.time())
    
    @property
    def qid(self) -> Qid:
        return self._qid
    
    @property
    def path(self) -> str:
        """Full path to this file"""
        if self.parent is None:
            return "/" if self.name == "" else f"/{self.name}"
        parent_path = self.parent.path
        if parent_path == "/":
            return f"/{self.name}"
        return f"{parent_path}/{self.name}"
    
    def stat(self) -> Stat:
        """Generate stat for this file"""
        return Stat(
            type=0,
            dev=0,
            qid=self._qid,
            mode=self._mode,
            atime=self._atime,
            mtime=self._mtime,
            length=self._get_length(),
            name=self.name,
            uid="llm",
            gid="llm",
            muid="llm"
        )
    
    def _get_length(self) -> int:
        """Override to report file size"""
        return 0
    
    def touch(self):
        """Update modification time"""
        self._mtime = int(time.time())
        self._qid.version += 1
    
    def open(self, fid: FidState, mode: int):
        """Called when file is opened. Override for setup."""
        pass
    
    def clunk(self, fid: FidState):
        """Called when file handle is closed. Override for cleanup."""
        pass
    
    @abstractmethod
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read data from file"""
        pass
    
    @abstractmethod
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Write data to file, return bytes written"""
        pass


class SyntheticDir(SyntheticFile):
    """
    Directory containing other files.
    
    Directories can contain both files and subdirectories.
    The read operation returns directory listings in 9P stat format.
    """
    
    def __init__(self, name: str, parent: 'SyntheticDir' = None):
        super().__init__(name, parent)
        self._qid = Qid(type=QTDIR, version=0, path=id(self))
        self._mode = DMDIR | 0o777  # Change from 0o755 to 0o777 for testing
        self.children: Dict[str, SyntheticFile] = {}
    
    def add(self, child: SyntheticFile):
        """Add a child file or directory"""
        child.parent = self
        self.children[child.name] = child
        self.touch()
    
    def remove(self, name: str) -> Optional[SyntheticFile]:
        """Remove a child by name"""
        if name in self.children:
            child = self.children.pop(name)
            child.parent = None
            self.touch()
            return child
        return None
    
    def get(self, name: str) -> Optional[SyntheticFile]:
        """Get child by name"""
        return self.children.get(name)
    
    def walk(self, path: List[str]) -> Optional[SyntheticFile]:
        """Walk a path from this directory"""
        if not path:
            return self
        
        name = path[0]
        rest = path[1:]
        
        if name == "..":
            target = self.parent or self
        elif name == ".":
            target = self
        else:
            target = self.children.get(name)
        
        if target is None:
            return None
        
        if rest:
            if isinstance(target, SyntheticDir):
                return target.walk(rest)
            return None
        
        return target
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read directory listing in 9P stat format"""
        # Generate stat entries for all children
        data = b""
        for child in self.children.values():
            stat = child.stat()
            data += stat.pack()
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Directories cannot be written to directly"""
        raise PermissionError("Cannot write to directory")
    
    def _get_length(self) -> int:
        """Directory length is 0 in 9P"""
        return 0


class DataFile(SyntheticFile):
    """
    Simple file that stores data in memory.
    
    Useful for configuration files, status files, etc.
    """
    
    def __init__(self, name: str, initial_content: bytes = b"", writable: bool = True):
        super().__init__(name)
        self._data = bytearray(initial_content)
        self._mode = 0o666
        if writable:
            self._mode = 0o666  # Make it clearly writable
        else:
            self._mode = 0o444  # Read-only
    
    @property
    def data(self) -> bytes:
        return bytes(self._data)
    
    @data.setter
    def data(self, value: bytes):
        self._data = bytearray(value)
        self.touch()
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        self._atime = int(time.time())
        return bytes(self._data[offset:offset + count])
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        # Extend if necessary
        end = offset + len(data)
        if end > len(self._data):
            self._data.extend(b'\x00' * (end - len(self._data)))
        
        self._data[offset:end] = data
        self.touch()
        return len(data)
    
    def _get_length(self) -> int:
        return len(self._data)


class StreamFile(SyntheticFile):
    """
    File with streaming read semantics and state-aware blocking.
    
    Readers block until data is available. Multiple readers each get
    their own cursor into a shared buffer. When the stream is finished,
    readers get EOF.
    
    STATE-AWARE LIFECYCLE (enables `while true; do cat $agent/output; done`):
    
    1. IDLE state: read() blocks on generation gate until reset() is called
    2. STREAMING state: read() blocks until data arrives, returns chunks
    3. EOF: read() returns b"" (empty), signalling cat to exit
    4. Next read at offset 0 blocks again at step 1
    
    The 9P server dispatches each message as a concurrent asyncio task,
    so a blocked read() never prevents writes to other files.
    """
    
    def __init__(self, name: str, buffer_size: int = 1_000_000):
        super().__init__(name)
        self._buffer = bytearray()
        self._buffer_size = buffer_size
        self._fid_cursors: Dict[int, int] = {}
        self._waiters: List[asyncio.Event] = []
        self._eof = False
        self._lock = asyncio.Lock()
        self._trim_offset = 0
        
        # Generation gate: readers wait here when idle.
        # Set by reset(), cleared by finish()/init.
        self._generation_started = asyncio.Event()
    
    @property
    def is_streaming(self) -> bool:
        return not self._eof
    
    @property
    def buffer_content(self) -> bytes:
        return bytes(self._buffer)
    
    async def append(self, data: bytes):
        """Append data to stream (producer side)."""
        async with self._lock:
            self._buffer.extend(data)
            self.touch()
            
            if len(self._buffer) > self._buffer_size:
                trim = len(self._buffer) - self._buffer_size
                self._buffer = self._buffer[trim:]
                self._trim_offset += trim
                for fid in list(self._fid_cursors.keys()):
                    self._fid_cursors[fid] = max(0, self._fid_cursors[fid] - trim)
            
            # Wake all blocked readers
            for event in self._waiters:
                event.set()
            self._waiters.clear()
    
    async def finish(self):
        """Mark stream as complete (EOF)."""
        async with self._lock:
            self._eof = True
            for event in self._waiters:
                event.set()
            self._waiters.clear()
        # Close generation gate for next cycle
        self._generation_started.clear()
    
    async def reset(self):
        """Reset stream for new generation — unblocks waiting readers."""
        async with self._lock:
            self._buffer.clear()
            self._fid_cursors.clear()
            self._eof = False
            self._trim_offset = 0
            self.touch()
        # Open the generation gate
        self._generation_started.set()
    
    def open(self, fid: FidState, mode: int):
        """New reader starts at beginning of current buffer."""
        self._fid_cursors[fid.fid] = 0
    
    def clunk(self, fid: FidState):
        """Clean up cursor when fid is closed."""
        self._fid_cursors.pop(fid.fid, None)
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """
        Read from stream with state-aware blocking.
        
        - cursor==0 and no generation active: block on generation gate
        - during streaming: block until data arrives
        - EOF: return b""
        """
        if fid.fid not in self._fid_cursors:
            self._fid_cursors[fid.fid] = 0
        
        cursor = self._fid_cursors[fid.fid]
        
        # ── GENERATION GATE ──
        # Block until a generation starts. Safe because the 9P server
        # dispatches each message as its own asyncio task — the Twrite
        # to input runs concurrently even while this Tread is here.
        if cursor == 0:
            await self._generation_started.wait()
        
        # ── STREAMING LOOP ──
        while True:
            async with self._lock:
                available = len(self._buffer) - cursor
                
                if available > 0:
                    data = bytes(self._buffer[cursor:cursor + count])
                    self._fid_cursors[fid.fid] = cursor + len(data)
                    return data
                
                if self._eof:
                    return b""
                
                event = asyncio.Event()
                self._waiters.append(event)
            
            await event.wait()
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Use append() to write to stream")
    
    def _get_length(self) -> int:
        return len(self._buffer) + self._trim_offset

class QueueFile(SyntheticFile):
    """
    FIFO queue as a file - read consumes items.
    
    This is ideal for event streams and error queues.
    Each read returns and removes one item from the queue.
    """
    
    def __init__(self, name: str, max_items: int = 1000):
        super().__init__(name)
        self._queue: asyncio.Queue = None
        self._max_items = max_items
        self._pending_data = b""  # Partial read buffer
    
    def _ensure_queue(self):
        """Lazily create queue (must be in async context)"""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_items)
    
    async def post(self, item: bytes):
        """Add item to queue (producer side)"""
        self._ensure_queue()
        try:
            self._queue.put_nowait(item)
            self.touch()
        except asyncio.QueueFull:
            # Drop oldest if full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except asyncio.QueueEmpty:
                pass
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read and consume from queue"""
        self._ensure_queue()
        
        # Return any pending data first
        if self._pending_data:
            data = self._pending_data[:count]
            self._pending_data = self._pending_data[count:]
            return data
        
        try:
            # Brief timeout for non-blocking behavior
            item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            
            if len(item) <= count:
                return item
            else:
                # Item larger than requested, save remainder
                self._pending_data = item[count:]
                return item[:count]
                
        except asyncio.TimeoutError:
            return b""
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Write adds to queue"""
        await self.post(data)
        return len(data)
    
    def _get_length(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.qsize()


class CtlHandler(ABC):
    """
    Handler for control file commands.
    
    Implement this to define the commands your control file accepts.
    """
    
    @abstractmethod
    async def execute(self, command: str) -> Optional[str]:
        """
        Execute a command, return optional response.
        
        Raise ValueError for unknown commands.
        """
        pass
    
    @abstractmethod
    async def get_status(self) -> bytes:
        """Return current status as bytes"""
        pass


class CtlFile(SyntheticFile):
    """
    Control file - write commands, read status.
    
    This is the Plan 9 pattern for controlling services.
    Write a command to execute it, read to get current status.
    """
    
    def __init__(self, name: str, handler: CtlHandler):
        super().__init__(name)
        self._handler = handler
        self._mode = 0o666
        self._last_response = b""
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read current status"""
        status = await self._handler.get_status()
        return status[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Execute command"""
        command = data.decode('utf-8').strip()
        
        # Handle multiple commands separated by newlines
        for line in command.split('\n'):
            line = line.strip()
            if line:
                result = await self._handler.execute(line)
                if result:
                    self._last_response = result.encode('utf-8')
        
        self.touch()
        return len(data)


class CallbackFile(SyntheticFile):
    """
    File that calls callbacks on read/write.
    
    Useful for simple dynamic files without needing a full class.
    """
    
    def __init__(
        self,
        name: str,
        read_callback: Callable[[], bytes] = None,
        write_callback: Callable[[bytes], int] = None
    ):
        super().__init__(name)
        self._read_cb = read_callback
        self._write_cb = write_callback
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        if self._read_cb is None:
            return b""
        
        data = self._read_cb()
        if asyncio.iscoroutine(data):
            data = await data
        
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        if self._write_cb is None:
            raise PermissionError("File is read-only")
        
        result = self._write_cb(data)
        if asyncio.iscoroutine(result):
            result = await result
        
        self.touch()
        return result if result is not None else len(data)
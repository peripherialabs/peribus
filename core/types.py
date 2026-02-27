"""
Core types for 9P protocol and synthetic filesystems.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import struct
import time

# Qid types
QTDIR = 0x80      # Directory
QTAPPEND = 0x40   # Append-only
QTEXCL = 0x20     # Exclusive use
QTAUTH = 0x08     # Authentication file
QTFILE = 0x00     # Regular file

# Permission/mode bits
DMDIR = 0x80000000     # Directory
DMAPPEND = 0x40000000  # Append-only
DMEXCL = 0x20000000    # Exclusive use
DMAUTH = 0x08000000    # Authentication file

# Open modes
OREAD = 0
OWRITE = 1  
ORDWR = 2
OEXEC = 3
OTRUNC = 0x10


@dataclass
class Qid:
    """
    Unique file identifier in 9P.
    
    The qid is a unique identifier for a file, consisting of:
    - type: file type (directory, regular, etc.)
    - version: increments on modification
    - path: unique path number
    """
    type: int = QTFILE
    version: int = 0
    path: int = 0
    
    def pack(self) -> bytes:
        """Pack qid to wire format (13 bytes)"""
        return struct.pack('<BIQ', self.type, self.version, self.path)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'Qid':
        """Unpack qid from wire format"""
        type_, version, path = struct.unpack('<BIQ', data[:13])
        return cls(type_, version, path)
    
    @classmethod
    def size(cls) -> int:
        return 13
    
    def __hash__(self):
        return hash((self.type, self.version, self.path))


@dataclass
class Stat:
    """
    File metadata (stat structure) in 9P.
    """
    type: int = 0           # Server type
    dev: int = 0            # Server subtype
    qid: Qid = None         # Unique id
    mode: int = 0o644       # Permissions and flags
    atime: int = 0          # Last access time
    mtime: int = 0          # Last modification time
    length: int = 0         # File length
    name: str = ""          # File name
    uid: str = "none"       # Owner
    gid: str = "none"       # Group
    muid: str = "none"      # Last modifier
    
    def __post_init__(self):
        if self.qid is None:
            self.qid = Qid()
        if self.atime == 0:
            self.atime = int(time.time())
        if self.mtime == 0:
            self.mtime = int(time.time())
    
    def pack(self) -> bytes:
        """Pack stat to wire format"""
        def pack_str(s: str) -> bytes:
            b = s.encode('utf-8')
            return struct.pack('<H', len(b)) + b
        
        # Build body
        body = struct.pack('<HI', self.type, self.dev)
        body += self.qid.pack()
        body += struct.pack('<IIIQ', self.mode, self.atime, self.mtime, self.length)
        body += pack_str(self.name)
        body += pack_str(self.uid)
        body += pack_str(self.gid)
        body += pack_str(self.muid)
        
        # Prepend with 2-byte size
        return struct.pack('<H', len(body)) + body
    
    @classmethod
    def unpack(cls, data: bytes) -> tuple['Stat', int]:
        """Unpack stat from wire format, returns (stat, bytes_consumed)"""
        size = struct.unpack('<H', data[:2])[0]
        pos = 2
        
        type_, dev = struct.unpack('<HI', data[pos:pos+6])
        pos += 6
        
        qid = Qid.unpack(data[pos:pos+13])
        pos += 13
        
        mode, atime, mtime, length = struct.unpack('<IIIQ', data[pos:pos+20])
        pos += 20
        
        def unpack_str(d: bytes, p: int) -> tuple[str, int]:
            slen = struct.unpack('<H', d[p:p+2])[0]
            s = d[p+2:p+2+slen].decode('utf-8')
            return s, p + 2 + slen
        
        name, pos = unpack_str(data, pos)
        uid, pos = unpack_str(data, pos)
        gid, pos = unpack_str(data, pos)
        muid, pos = unpack_str(data, pos)
        
        return cls(type_, dev, qid, mode, atime, mtime, length, name, uid, gid, muid), size + 2


@dataclass
class FidState:
    """
    State associated with an open file descriptor (fid).
    
    Each client connection maintains its own fid namespace.
    """
    fid: int
    path: str
    qid: Qid
    file: Any  # The SyntheticFile
    mode: int = 0
    opened: bool = False
    offset: int = 0
    aux: Any = None  # File-type specific state
    
    def __hash__(self):
        return hash(self.fid)

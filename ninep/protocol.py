"""
9P2000 Protocol Message Types

This module defines all the message types used in the 9P2000 protocol.
Each T-message (transmit) has a corresponding R-message (response).

Fixed to properly handle tag inheritance and ensure correct tag echoing.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from core.types import Qid, Stat

# Open modes
OREAD = 0       # Open for read
OWRITE = 1      # Open for write
ORDWR = 2       # Open for read/write
OEXEC = 3       # Open for execute
OTRUNC = 0x10   # Truncate file first

# Special constants
NOFID = 0xFFFFFFFF  # No fid
NOTAG = 0xFFFF      # No tag (used only for Tversion/Rversion)


class MsgType(IntEnum):
    """9P message types"""
    Tversion = 100
    Rversion = 101
    Tauth = 102
    Rauth = 103
    Tattach = 104
    Rattach = 105
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


@dataclass
class Message:
    """
    Base 9P message.
    
    CRITICAL: tag has no default value to ensure it's always
    explicitly provided. This prevents tag assignment bugs.
    """
    tag: int
    
    @classmethod
    def msg_type(cls) -> MsgType:
        """Return the message type for this class"""
        raise NotImplementedError


# ============================================================================
# Version Negotiation
# ============================================================================

@dataclass
class Tversion(Message):
    """Version negotiation request"""
    msize: int = 8192      # Maximum message size
    version: str = "9P2000"
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tversion


@dataclass
class Rversion(Message):
    """Version negotiation response"""
    msize: int = 8192
    version: str = "9P2000"
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rversion


# ============================================================================
# Authentication (optional - we don't implement it)
# ============================================================================

@dataclass
class Tauth(Message):
    """Authentication request"""
    afid: int = 0
    uname: str = ""
    aname: str = ""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tauth


@dataclass
class Rauth(Message):
    """Authentication response"""
    aqid: Qid = None
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rauth


# ============================================================================
# Attach (mount root)
# ============================================================================

@dataclass
class Tattach(Message):
    """Attach to filesystem root"""
    fid: int = 0           # Fid to assign to root
    afid: int = NOFID      # Auth fid (NOFID if no auth)
    uname: str = ""        # User name
    aname: str = ""        # Attach name (tree to mount)
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tattach


@dataclass
class Rattach(Message):
    """Attach response"""
    qid: Qid = None
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rattach


# ============================================================================
# Error
# ============================================================================

@dataclass
class Rerror(Message):
    """Error response (no corresponding T-message)"""
    ename: str = ""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rerror


# ============================================================================
# Flush (cancel pending request)
# ============================================================================

@dataclass
class Tflush(Message):
    """Cancel pending request"""
    oldtag: int = 0
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tflush


@dataclass
class Rflush(Message):
    """Flush response"""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rflush


# ============================================================================
# Walk (path traversal)
# ============================================================================

@dataclass
class Twalk(Message):
    """Walk path components"""
    fid: int = 0           # Starting fid
    newfid: int = 0        # Fid to assign to result
    wnames: List[str] = field(default_factory=list)  # Path components
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Twalk


@dataclass
class Rwalk(Message):
    """Walk response"""
    qids: List[Qid] = field(default_factory=list)  # Qid for each component walked
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rwalk


# ============================================================================
# Open
# ============================================================================

@dataclass
class Topen(Message):
    """Open file"""
    fid: int = 0
    mode: int = OREAD
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Topen


@dataclass
class Ropen(Message):
    """Open response"""
    qid: Qid = None
    iounit: int = 0        # Maximum read/write size (0 = use msize)
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Ropen


# ============================================================================
# Create
# ============================================================================

@dataclass
class Tcreate(Message):
    """Create file"""
    fid: int = 0           # Directory fid, becomes new file fid
    name: str = ""         # File name to create
    perm: int = 0          # Permissions
    mode: int = OREAD      # Open mode
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tcreate


@dataclass
class Rcreate(Message):
    """Create response"""
    qid: Qid = None
    iounit: int = 0
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rcreate


# ============================================================================
# Read
# ============================================================================

@dataclass
class Tread(Message):
    """Read from file"""
    fid: int = 0
    offset: int = 0        # 64-bit offset
    count: int = 0         # Number of bytes to read
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tread


@dataclass
class Rread(Message):
    """Read response"""
    data: bytes = b""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rread


# ============================================================================
# Write
# ============================================================================

@dataclass
class Twrite(Message):
    """Write to file"""
    fid: int = 0
    offset: int = 0        # 64-bit offset
    data: bytes = b""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Twrite


@dataclass
class Rwrite(Message):
    """Write response"""
    count: int = 0         # Bytes actually written
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rwrite


# ============================================================================
# Clunk (close fid)
# ============================================================================

@dataclass
class Tclunk(Message):
    """Close fid"""
    fid: int = 0
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tclunk


@dataclass
class Rclunk(Message):
    """Clunk response"""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rclunk


# ============================================================================
# Remove
# ============================================================================

@dataclass
class Tremove(Message):
    """Remove file"""
    fid: int = 0
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tremove


@dataclass
class Rremove(Message):
    """Remove response"""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rremove


# ============================================================================
# Stat
# ============================================================================

@dataclass
class Tstat(Message):
    """Get file status"""
    fid: int = 0
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Tstat


@dataclass
class Rstat(Message):
    """Stat response"""
    stat: Stat = None
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rstat


# ============================================================================
# Wstat (write stat)
# ============================================================================

@dataclass
class Twstat(Message):
    """Modify file status"""
    fid: int = 0
    stat: Stat = None
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Twstat


@dataclass
class Rwstat(Message):
    """Wstat response"""
    
    @classmethod
    def msg_type(cls) -> MsgType:
        return MsgType.Rwstat
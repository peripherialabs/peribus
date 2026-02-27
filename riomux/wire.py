"""
riomux.wire — Low-level 9P2000 wire format reader/writer.

Operates on raw bytes. Understands just enough of the 9P2000 framing
to extract and rewrite fid, tag, and message-type fields without
needing the full protocol/codec stack.

This is critical: we do NOT decode message payloads (stat entries,
directory data, file content, etc.). We pass them through byte-for-byte.
This guarantees that blocking reads, streaming output, clunk semantics,
and every other behavior is preserved exactly.

9P2000 message format:
    size[4] type[1] tag[2] ... (type-specific fields)

Message types (T=request from client, R=response from server):
    100 Tversion  101 Rversion
    102 Tauth     103 Rauth
    104 Tattach   105 Rattach
    106 Rerror          (no Terror)
    108 Tflush    109 Rflush
    110 Twalk     111 Rwalk
    112 Topen     113 Ropen
    114 Tcreate   115 Rcreate
    116 Tread     117 Rread
    118 Twrite    119 Rwrite
    120 Tclunk    121 Rclunk
    122 Tremove   123 Rremove
    124 Tstat     125 Rstat
    126 Twstat    127 Rwstat
"""

import struct
from typing import Optional, Tuple

# 9P2000 message types
TVERSION = 100; RVERSION = 101
TAUTH    = 102; RAUTH    = 103
TATTACH  = 104; RATTACH  = 105
RERROR   = 107  # Note: no TERROR (106)
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

# Message type name lookup for debugging
MSG_NAMES = {
    100: "Tversion", 101: "Rversion",
    102: "Tauth",    103: "Rauth",
    104: "Tattach",  105: "Rattach",
    107: "Rerror",
    108: "Tflush",   109: "Rflush",
    110: "Twalk",    111: "Rwalk",
    112: "Topen",    113: "Ropen",
    114: "Tcreate",  115: "Rcreate",
    116: "Tread",    117: "Rread",
    118: "Twrite",   119: "Rwrite",
    120: "Tclunk",   121: "Rclunk",
    122: "Tremove",  123: "Rremove",
    124: "Tstat",    125: "Rstat",
    126: "Twstat",   127: "Rwstat",
}


def msg_name(mtype: int) -> str:
    return MSG_NAMES.get(mtype, f"Unknown({mtype})")


# ── Header parsing ──────────────────────────────────────────────

def parse_header(data: bytes) -> Tuple[int, int, int]:
    """
    Parse the 7-byte 9P header.
    Returns (size, type, tag).
    """
    size, mtype, tag = struct.unpack_from('<IBH', data, 0)
    return size, mtype, tag


def get_size(data: bytes) -> Optional[int]:
    """Read message size from first 4 bytes, or None if not enough data."""
    if len(data) < 4:
        return None
    return struct.unpack_from('<I', data, 0)[0]


# ── Fid extraction/rewriting ────────────────────────────────────
#
# Different message types have their fid(s) at different offsets.
# We need to know where to read and rewrite them.
#
# All offsets are from the start of the message (including the 4-byte size).
# Header: size[4] type[1] tag[2] = 7 bytes
# Fid is always a uint32 starting at offset 7 for most T-messages.

# For T-messages that contain fid fields:
#   Tauth:    fid @ 7
#   Tattach:  fid @ 7, afid @ 11
#   Twalk:    fid @ 7, newfid @ 11
#   Topen:    fid @ 7
#   Tcreate:  fid @ 7
#   Tread:    fid @ 7
#   Twrite:   fid @ 7
#   Tclunk:   fid @ 7
#   Tremove:  fid @ 7
#   Tstat:    fid @ 7
#   Twstat:   fid @ 7
#   Tflush:   oldtag @ 7 (uint16, not a fid)

def get_fid(data: bytes, mtype: int) -> Optional[int]:
    """Extract the primary fid from a T-message, or None if N/A."""
    if mtype in (TAUTH, TATTACH, TWALK, TOPEN, TCREATE, TREAD,
                 TWRITE, TCLUNK, TREMOVE, TSTAT, TWSTAT):
        if len(data) >= 11:
            return struct.unpack_from('<I', data, 7)[0]
    return None


def get_newfid(data: bytes, mtype: int) -> Optional[int]:
    """Extract newfid from Twalk, afid from Tattach, or None."""
    if mtype == TWALK:
        if len(data) >= 15:
            return struct.unpack_from('<I', data, 11)[0]
    elif mtype == TATTACH:
        # afid @ offset 11
        if len(data) >= 15:
            return struct.unpack_from('<I', data, 11)[0]
    return None


def set_fid(data: bytearray, offset: int, fid: int):
    """Write a fid (uint32) at the given offset."""
    struct.pack_into('<I', data, offset, fid)


def set_tag(data: bytearray, tag: int):
    """Write tag (uint16) at offset 5."""
    struct.pack_into('<H', data, 5, tag)


def get_tag(data: bytes) -> int:
    """Read tag from offset 5."""
    return struct.unpack_from('<H', data, 5)[0]


def get_type(data: bytes) -> int:
    """Read message type from offset 4."""
    return data[4]


def get_flush_oldtag(data: bytes) -> int:
    """Get oldtag from Tflush message (uint16 at offset 7)."""
    return struct.unpack_from('<H', data, 7)[0]


def set_flush_oldtag(data: bytearray, oldtag: int):
    """Set oldtag in Tflush message."""
    struct.pack_into('<H', data, 7, oldtag)


# ── Twalk parsing (needed for routing) ──────────────────────────

def parse_twalk(data: bytes) -> Tuple[int, int, list]:
    """
    Parse Twalk: fid[4] newfid[4] nwname[2] nwname*(name[s])
    Returns (fid, newfid, [name, ...])
    """
    fid = struct.unpack_from('<I', data, 7)[0]
    newfid = struct.unpack_from('<I', data, 11)[0]
    nwname = struct.unpack_from('<H', data, 15)[0]
    
    names = []
    offset = 17
    for _ in range(nwname):
        if offset + 2 > len(data):
            break
        slen = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        if offset + slen > len(data):
            break
        names.append(data[offset:offset + slen].decode('utf-8'))
        offset += slen
    
    return fid, newfid, names


def build_twalk(tag: int, fid: int, newfid: int, names: list) -> bytes:
    """Build a Twalk message from components."""
    body = struct.pack('<IBH', 0, TWALK, tag)  # size placeholder, type, tag
    body += struct.pack('<I', fid)
    body += struct.pack('<I', newfid)
    body += struct.pack('<H', len(names))
    for name in names:
        encoded = name.encode('utf-8')
        body += struct.pack('<H', len(encoded)) + encoded
    # Fix size
    size = len(body)
    return struct.pack('<I', size) + body[4:]


# ── Tattach parsing ─────────────────────────────────────────────

def parse_tattach(data: bytes) -> Tuple[int, int, str, str]:
    """
    Parse Tattach: fid[4] afid[4] uname[s] aname[s]
    Returns (fid, afid, uname, aname)
    """
    fid = struct.unpack_from('<I', data, 7)[0]
    afid = struct.unpack_from('<I', data, 11)[0]
    
    offset = 15
    uname_len = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    uname = data[offset:offset + uname_len].decode('utf-8')
    offset += uname_len
    
    aname_len = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    aname = data[offset:offset + aname_len].decode('utf-8')
    
    return fid, afid, uname, aname


def build_rattach(tag: int, qid_type: int, qid_vers: int, qid_path: int) -> bytes:
    """Build an Rattach response."""
    body = bytearray()
    body += struct.pack('<IBH', 0, RATTACH, tag)  # size placeholder
    body += struct.pack('<BIQ', qid_type, qid_vers, qid_path)  # qid
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_rerror(tag: int, ename: str) -> bytes:
    """Build an Rerror response."""
    ename_bytes = ename.encode('utf-8')
    body = bytearray()
    body += struct.pack('<IBH', 0, RERROR, tag)
    body += struct.pack('<H', len(ename_bytes)) + ename_bytes
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_rversion(tag: int, msize: int, version: str) -> bytes:
    """Build an Rversion response."""
    ver_bytes = version.encode('utf-8')
    body = bytearray()
    body += struct.pack('<IBH', 0, RVERSION, tag)
    body += struct.pack('<I', msize)
    body += struct.pack('<H', len(ver_bytes)) + ver_bytes
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_rwalk(tag: int, qids: list) -> bytes:
    """
    Build an Rwalk response.
    qids is a list of (type, vers, path) tuples.
    """
    body = bytearray()
    body += struct.pack('<IBH', 0, RWALK, tag)
    body += struct.pack('<H', len(qids))
    for qtype, qvers, qpath in qids:
        body += struct.pack('<BIQ', qtype, qvers, qpath)
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_rread_dir(tag: int, stat_entries: bytes) -> bytes:
    """Build an Rread containing directory stat entries."""
    body = bytearray()
    body += struct.pack('<IBH', 0, RREAD, tag)
    body += struct.pack('<I', len(stat_entries))
    body += stat_entries
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_ropen(tag: int, qid_type: int, qid_vers: int, qid_path: int,
                iounit: int) -> bytes:
    """Build an Ropen response."""
    body = bytearray()
    body += struct.pack('<IBH', 0, ROPEN, tag)
    body += struct.pack('<BIQ', qid_type, qid_vers, qid_path)
    body += struct.pack('<I', iounit)
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_rstat(tag: int, stat_data: bytes) -> bytes:
    """
    Build an Rstat response.
    stat_data is the already-packed stat (with leading 2-byte size).
    Rstat wraps it in another 2-byte count prefix.
    """
    body = bytearray()
    body += struct.pack('<IBH', 0, RSTAT, tag)
    body += struct.pack('<H', len(stat_data))
    body += stat_data
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_rclunk(tag: int) -> bytes:
    """Build an Rclunk response."""
    body = bytearray()
    body += struct.pack('<IBH', 0, RCLUNK, tag)
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


def build_rflush(tag: int) -> bytes:
    """Build an Rflush response."""
    body = bytearray()
    body += struct.pack('<IBH', 0, RFLUSH, tag)
    size = len(body)
    struct.pack_into('<I', body, 0, size)
    return bytes(body)


# ── Synthetic stat packing ──────────────────────────────────────

QTDIR = 0x80
QTFILE = 0x00
DMDIR = 0x80000000


def pack_stat(name: str, qid_path: int, is_dir: bool = False,
              length: int = 0) -> bytes:
    """
    Pack a synthetic 9P stat entry for virtual mux directories/files.
    """
    name_bytes = name.encode('utf-8')
    uid_bytes = b"mux"
    gid_bytes = b"mux"
    muid_bytes = b""
    
    qtype = QTDIR if is_dir else QTFILE
    mode = (DMDIR | 0o777) if is_dir else 0o666
    
    import time
    now = int(time.time())
    
    qid_bytes = struct.pack('<BIQ', qtype, 0, qid_path)
    
    body = struct.pack('<HI', 0, 0)  # type, dev
    body += qid_bytes
    body += struct.pack('<I', mode)
    body += struct.pack('<I', now)   # atime
    body += struct.pack('<I', now)   # mtime
    body += struct.pack('<Q', length)
    body += struct.pack('<H', len(name_bytes)) + name_bytes
    body += struct.pack('<H', len(uid_bytes)) + uid_bytes
    body += struct.pack('<H', len(gid_bytes)) + gid_bytes
    body += struct.pack('<H', len(muid_bytes)) + muid_bytes
    
    return struct.pack('<H', len(body)) + body
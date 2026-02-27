"""
9P2000 Wire Format Codec

Handles encoding and decoding of 9P messages to/from their binary wire format.

Wire format for all messages:
    size[4] type[1] tag[2] ... message-specific data ...

Where size includes itself (minimum 7 bytes for header).
"""

import struct
from typing import Tuple, Union

from .protocol import *
from core.types import Qid, Stat


class Codec:
    """
    9P message encoder/decoder.
    
    Handles the binary serialization of all 9P2000 message types.
    """
    
    def __init__(self, msize: int = 65536):
        self.msize = msize
    
    def encode(self, msg: Message) -> bytes:
        """Encode message to wire format"""
        body = self._encode_body(msg)
        msg_type = msg.msg_type().value
        
        # Header: size[4] type[1] tag[2]
        size = 4 + 1 + 2 + len(body)
        header = struct.pack('<IBH', size, msg_type, msg.tag)
        
        return header + body
    
    def decode(self, data: bytes) -> Tuple[Message, int]:
        """
        Decode message from wire format.
        
        Returns (message, bytes_consumed).
        Raises ValueError if data is incomplete.
        """
        if len(data) < 7:
            raise ValueError("Incomplete header")
        
        size, msg_type, tag = struct.unpack('<IBH', data[:7])
        
        if len(data) < size:
            raise ValueError(f"Incomplete message: have {len(data)}, need {size}")
        
        body = data[7:size]
        msg = self._decode_body(MsgType(msg_type), tag, body)
        
        return msg, size
    
    def _encode_body(self, msg: Message) -> bytes:
        """Encode message body (without header)"""
        
        if isinstance(msg, Tversion):
            return struct.pack('<I', msg.msize) + self._pack_str(msg.version)
        
        elif isinstance(msg, Rversion):
            return struct.pack('<I', msg.msize) + self._pack_str(msg.version)
        
        elif isinstance(msg, Tattach):
            return (struct.pack('<II', msg.fid, msg.afid) +
                    self._pack_str(msg.uname) +
                    self._pack_str(msg.aname))
        
        elif isinstance(msg, Rattach):
            return msg.qid.pack()
        
        elif isinstance(msg, Rerror):
            return self._pack_str(msg.ename)
        
        elif isinstance(msg, Tflush):
            return struct.pack('<H', msg.oldtag)
        
        elif isinstance(msg, Rflush):
            return b''
        
        elif isinstance(msg, Twalk):
            body = struct.pack('<IIH', msg.fid, msg.newfid, len(msg.wnames))
            for name in msg.wnames:
                body += self._pack_str(name)
            return body
        
        elif isinstance(msg, Rwalk):
            body = struct.pack('<H', len(msg.qids))
            for qid in msg.qids:
                body += qid.pack()
            return body
        
        elif isinstance(msg, Topen):
            return struct.pack('<IB', msg.fid, msg.mode)
        
        elif isinstance(msg, Ropen):
            return msg.qid.pack() + struct.pack('<I', msg.iounit)
        
        elif isinstance(msg, Tcreate):
            return (struct.pack('<I', msg.fid) +
                    self._pack_str(msg.name) +
                    struct.pack('<IB', msg.perm, msg.mode))
        
        elif isinstance(msg, Rcreate):
            return msg.qid.pack() + struct.pack('<I', msg.iounit)
        
        elif isinstance(msg, Tread):
            return struct.pack('<IQI', msg.fid, msg.offset, msg.count)
        
        elif isinstance(msg, Rread):
            return struct.pack('<I', len(msg.data)) + msg.data
        
        elif isinstance(msg, Twrite):
            return struct.pack('<IQI', msg.fid, msg.offset, len(msg.data)) + msg.data
        
        elif isinstance(msg, Rwrite):
            return struct.pack('<I', msg.count)
        
        elif isinstance(msg, Tclunk):
            return struct.pack('<I', msg.fid)
        
        elif isinstance(msg, Rclunk):
            return b''
        
        elif isinstance(msg, Tremove):
            return struct.pack('<I', msg.fid)
        
        elif isinstance(msg, Rremove):
            return b''
        
        elif isinstance(msg, Tstat):
            return struct.pack('<I', msg.fid)
        
        elif isinstance(msg, Rstat):
            stat_data = msg.stat.pack()
            # Rstat has additional 2-byte size prefix before the stat
            return struct.pack('<H', len(stat_data)) + stat_data
        
        elif isinstance(msg, Twstat):
            stat_data = msg.stat.pack()
            return struct.pack('<I', msg.fid) + struct.pack('<H', len(stat_data)) + stat_data
        
        elif isinstance(msg, Rwstat):
            return b''
        
        else:
            raise ValueError(f"Cannot encode message type: {type(msg)}")
    
    def _decode_body(self, msg_type: MsgType, tag: int, body: bytes) -> Message:
        """Decode message body"""
        
        if msg_type == MsgType.Tversion:
            msize = struct.unpack('<I', body[:4])[0]
            version, _ = self._unpack_str(body, 4)
            return Tversion(tag, msize, version)
        
        elif msg_type == MsgType.Rversion:
            msize = struct.unpack('<I', body[:4])[0]
            version, _ = self._unpack_str(body, 4)
            return Rversion(tag, msize, version)
        
        elif msg_type == MsgType.Tattach:
            fid, afid = struct.unpack('<II', body[:8])
            uname, pos = self._unpack_str(body, 8)
            aname, _ = self._unpack_str(body, pos)
            return Tattach(tag, fid, afid, uname, aname)
        
        elif msg_type == MsgType.Rattach:
            qid = Qid.unpack(body[:13])
            return Rattach(tag, qid)
        
        elif msg_type == MsgType.Rerror:
            ename, _ = self._unpack_str(body, 0)
            return Rerror(tag, ename)
        
        elif msg_type == MsgType.Tflush:
            oldtag = struct.unpack('<H', body[:2])[0]
            return Tflush(tag, oldtag)
        
        elif msg_type == MsgType.Rflush:
            return Rflush(tag)
        
        elif msg_type == MsgType.Twalk:
            fid, newfid, nwname = struct.unpack('<IIH', body[:10])
            pos = 10
            wnames = []
            for _ in range(nwname):
                name, pos = self._unpack_str(body, pos)
                wnames.append(name)
            return Twalk(tag, fid, newfid, wnames)
        
        elif msg_type == MsgType.Rwalk:
            nqid = struct.unpack('<H', body[:2])[0]
            pos = 2
            qids = []
            for _ in range(nqid):
                qid = Qid.unpack(body[pos:pos+13])
                qids.append(qid)
                pos += 13
            return Rwalk(tag, qids)
        
        elif msg_type == MsgType.Topen:
            fid, mode = struct.unpack('<IB', body[:5])
            return Topen(tag, fid, mode)
        
        elif msg_type == MsgType.Ropen:
            qid = Qid.unpack(body[:13])
            iounit = struct.unpack('<I', body[13:17])[0]
            return Ropen(tag, qid, iounit)
        
        elif msg_type == MsgType.Tcreate:
            fid = struct.unpack('<I', body[:4])[0]
            name, pos = self._unpack_str(body, 4)
            perm, mode = struct.unpack('<IB', body[pos:pos+5])
            return Tcreate(tag, fid, name, perm, mode)
        
        elif msg_type == MsgType.Rcreate:
            qid = Qid.unpack(body[:13])
            iounit = struct.unpack('<I', body[13:17])[0]
            return Rcreate(tag, qid, iounit)
        
        elif msg_type == MsgType.Tread:
            fid, offset, count = struct.unpack('<IQI', body[:16])
            return Tread(tag, fid, offset, count)
        
        elif msg_type == MsgType.Rread:
            count = struct.unpack('<I', body[:4])[0]
            data = body[4:4+count]
            return Rread(tag, data)
        
        elif msg_type == MsgType.Twrite:
            fid, offset, count = struct.unpack('<IQI', body[:16])
            data = body[16:16+count]
            return Twrite(tag, fid, offset, data)
        
        elif msg_type == MsgType.Rwrite:
            count = struct.unpack('<I', body[:4])[0]
            return Rwrite(tag, count)
        
        elif msg_type == MsgType.Tclunk:
            fid = struct.unpack('<I', body[:4])[0]
            return Tclunk(tag, fid)
        
        elif msg_type == MsgType.Rclunk:
            return Rclunk(tag)
        
        elif msg_type == MsgType.Tremove:
            fid = struct.unpack('<I', body[:4])[0]
            return Tremove(tag, fid)
        
        elif msg_type == MsgType.Rremove:
            return Rremove(tag)
        
        elif msg_type == MsgType.Tstat:
            fid = struct.unpack('<I', body[:4])[0]
            return Tstat(tag, fid)
        
        elif msg_type == MsgType.Rstat:
            # Skip the 2-byte size prefix
            stat, _ = Stat.unpack(body[2:])
            return Rstat(tag, stat)
        
        elif msg_type == MsgType.Twstat:
            fid = struct.unpack('<I', body[:4])[0]
            # Skip 2-byte size prefix
            stat, _ = Stat.unpack(body[6:])
            return Twstat(tag, fid, stat)
        
        elif msg_type == MsgType.Tauth:
            afid = struct.unpack('<I', body[:4])[0]
            uname, pos = self._unpack_str(body, 4)
            aname, _ = self._unpack_str(body, pos)
            return Tauth(tag, afid, uname, aname)
        
        elif msg_type == MsgType.Rwstat:
            return Rwstat(tag)
        
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
    
    @staticmethod
    def _pack_str(s: str) -> bytes:
        """Pack string with 2-byte length prefix"""
        b = s.encode('utf-8')
        return struct.pack('<H', len(b)) + b
    
    @staticmethod
    def _unpack_str(data: bytes, pos: int) -> Tuple[str, int]:
        """Unpack string, returns (string, new_position)"""
        slen = struct.unpack('<H', data[pos:pos+2])[0]
        s = data[pos+2:pos+2+slen].decode('utf-8')
        return s, pos + 2 + slen

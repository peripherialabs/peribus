"""
Terminal Widget for Rio Display Server - LLMFS Integration

Design Principles:
  - ALL interaction with agents goes through the filesystem (/n/llm)
  - Enter submits, Shift+Enter inserts newline
  - /new name creates+connects an agent in one step
  - Plain text sends prompt to connected agent's $agent/input
  - >>> for Python, $ for shell, / for macros
  - Agent output is streamed Plan9-style: continuous non-blocking read
    from $agent/OUTPUT, always printing as data arrives.

Plan 9 Blocking Semantics:
  - cat $claude/RIOA BLOCKS until content is ready (just like cat $claude/OUTPUT)
  - /attach uses blocking I/O, NO POLLING
  - echo 'prompt' > $claude/input && cat $claude/RIOA > /n/rioa/scene/parse works!

Signal contract:
  command_submitted(str) is emitted ONLY for code that should reach the
  Rio executor (Python via >>>).  Macro commands, shell commands, and
  agent prompts are handled entirely inside the widget and do NOT emit
  the signal.

Command Reference:
  /new claude [system]   Create agent "claude", set system prompt, auto-connect
  /new claude groq kimi  Create agent with specific provider+model
  /connect <n>           Connect to existing agent
  /disconnect            Disconnect from current agent
  /provider <n> [model]  Switch provider on connected agent
  /use <prov> <hint>     Fuzzy-match provider+model (e.g. /use groq kimi)
  /use <alias>           Quick alias (kimi, zai, sonnet, opus, haiku, flash, ...)
  /master [prov] [model] Spawn master agent (auto-exec bash, coordinates)
  /coder [prov] [model]  Spawn coder agent (workspace-aware)
  /tcoder [prov] [model] Same as /coder, but routes output to a specific
                         terminal's inline stream: $coder/<MACHINE> ->
                         /n/<machine>/terms/<term_id>/inline.
                         For our local machine, <term_id> is this terminal.
                         For other machines, the first terminal found is used.
  /av [voice]            Start Grok voice agent with function tools
  /av_gemini [voice]     Start Gemini voice agent with function tools
  /attach <src> <dst>    Auto-route source to destination (blocking, no polling)
  /detach <src>          Stop auto-routing from source
  /attachments           List active attachments
  /context <n>           Route workspace CONTEXT to agent's history
  /system <text>         Set system prompt on connected agent
  /model <model>         Set model on connected agent
  /temperature <val>     Set temperature
  /clear                 Clear agent history
  /cancel                Cancel current generation
  /retry                 Retry last message
  /history               Show conversation history
  /config                Show agent config
  /errors                Show agent errors
  /list (/ls)            List agents
  /status                Show connection status
  /delete <n>            Delete agent
  /help                  Show help
  /cls                   Clear terminal output
  /color (/colors)       Open color scheme picker
  /dark (/darkmode)      Toggle dark/light mode
  /versions (/ver)       Toggle version panel
  /acme                  Open ACME editor
  /operator              Open Operator graph panel
  /scene                 Toggle this terminal's live UI panel
                         (writes to /n/rioa/terms/<term_id>/parse draw on it)
  /pop                   Detach terminal to floating window
  /dock                  Re-dock terminal into scene
  /restart               Restart shell (fresh env, re-seed vars)
  /setup                 Unmount & remount 9pfuse (LLMFS + Rio)
  /mount <IP!Port> <n>   Mount 9P service at /n/name via 9pfuse
  /signal on|off         Full-mesh subscribe/unsubscribe across every
                         machine in /n/ctl (writes to each one's
                         /scene/signals/ctl)
  /peribus               Connect to peribus mycelium (feed + inbox → $inline)
  /peribus post <text>   Publish a short post to your peribus feed
  /peribus stop|status   Stop tailer / show state
  /share                 Open composer dialog (text + optional file picker)
  /share <text>          Publish a text post (rich envelope)
  /share <path>          Publish a file as a media post (kind auto-detected)
  /share <path> <text>   File + caption (also: drag & drop a file in)
  /share scene [text]    Publish the live CONTEXT (compacted scene code) as a .py post
  >>> <code>             Execute Python code
  $ <command>            Execute shell command
  $                      Toggle persistent shell mode
  <text>                 Send prompt to connected agent
"""

from PySide6.QtWidgets import (
    QWidget, QTextEdit, QVBoxLayout, QHBoxLayout, QFrame,
    QSizePolicy, QApplication, QScrollArea, QGraphicsDropShadowEffect, QSplitter,
    QGraphicsView, QGraphicsScene, QPushButton, QLabel, QMainWindow,
    QLineEdit, QStackedLayout
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QPointF, QRectF, QThread, QObject, QEvent, Slot, QVariantAnimation, QEasingCurve, Q_ARG, QMetaObject, QProcess
from PySide6.QtGui import QColor, QPalette, QFont, QTextCursor, QKeyEvent, QTextCharFormat, QPixmap, QPainter, QBrush, QPen
import asyncio
import collections
import errno
import json
import os
import signal
import socket
import struct
import sys
import fcntl
import subprocess
import tempfile
import time
import re
from typing import Dict, Optional

import pty
import selectors
import termios
import tty
import uuid
import threading

from rio.acme.acme_core import Acme
from .operator_panel import OperatorPanel
from .version_panel import VersionPanel
from .shell_sandbox import check_command as _sandbox_check
from .theme import get_theme, DEFAULT_THEME_NAME, Theme as _Theme


# ---------------------------------------------------------------------------
# Focus-tint overlay — avoids setStyleSheet cascade on terminal_frame
# ---------------------------------------------------------------------------

class FocusTintOverlay(QWidget):
    """Transparent widget that paints a rounded-rect focus tint.

    Parented to terminal_frame and kept full-size via resizeEvent.
    On each animation tick we update ``_alpha`` and call ``update()``
    which repaints *only this widget* — zero stylesheet invalidation
    on the hundreds of inline children underneath.

    The overlay is always stacked on top (raised) and is transparent
    to mouse/keyboard events (WA_TransparentForMouseEvents).
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")
        # Tint colour (RGB) and animated alpha — set by the terminal.
        self._r = 0
        self._g = 0
        self._b = 0
        self._alpha = 0
        self._radius = 8  # matched to terminal_frame border-radius
        self.raise_()
        self.show()

    # --- geometry sync ---------------------------------------------------

    def sync_geometry(self):
        """Match parent size.  Called on parent resizeEvent."""
        p = self.parent()
        if p is not None:
            self.setGeometry(0, 0, p.width(), p.height())

    # --- painting --------------------------------------------------------

    def paintEvent(self, event):
        if self._alpha <= 0:
            return  # fully transparent — nothing to paint
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(self._r, self._g, self._b, int(self._alpha)))
        painter.drawRoundedRect(self.rect(), self._radius, self._radius)
        painter.end()

    def set_tint(self, r: int, g: int, b: int, alpha: float, radius: int = None):
        """Update tint parameters and schedule a repaint."""
        self._r = r
        self._g = g
        self._b = b
        self._alpha = alpha
        if radius is not None:
            self._radius = radius
        self.update()  # schedules paintEvent — O(1), no children walked


# ---------------------------------------------------------------------------
# Plan 9 Style Attachment - Blocking I/O (No Polling!)
# ---------------------------------------------------------------------------

class Plan9Attachment:
    """
    Manages a single source->destination attachment using blocking I/O.
    
    Spawns a subprocess that runs:
        while true; do cat $source > $destination; done
    
    The cat BLOCKS on the server side until content is ready:
    - StreamFile: blocks on generation gate until reset(), then streams
    - SupplementaryOutputFile: blocks on _content_ready until mark_ready()
    - TerminalStdoutFile: blocks on _output_ready until mark_ready()
    
    After content is delivered, cat gets EOF and exits. The while loop
    re-runs cat, which blocks again. Zero polling, zero CPU in steady state.
    """
    
    def __init__(self, source: str, destination: str):
        self.source = source
        self.destination = destination
        self.process = None
    
    def start(self):
        """Start the attachment process"""
        import tempfile
        
        fd, script_path = tempfile.mkstemp(suffix='.sh', prefix='llmfs_attach_')
        
        script_content = f"""#!/bin/bash
SOURCE="{self.source}"
DEST="{self.destination}"

mkdir -p "$(dirname "$DEST")" 2>/dev/null || true

# cat blocks server-side until content is ready.
# On EOF, loop restarts and cat blocks again. No polling.
while true; do
    cat "$SOURCE" > "$DEST" 2>/dev/null
done
"""
        
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.script_path = script_path
        
        self.process = subprocess.Popen(
            ['nohup', 'bash', script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )
    
    def stop(self):
        """Stop the attachment process and all its children (cat, etc.)"""
        if self.process:
            try:
                # Kill the entire process group (bash + cat children)
                # since start_new_session=True gives it its own pgid
                import signal as _signal
                os.killpg(self.process.pid, _signal.SIGTERM)
                self.process.wait(timeout=2)
            except Exception:
                try:
                    os.killpg(self.process.pid, _signal.SIGKILL)
                    self.process.wait(timeout=1)
                except Exception:
                    try:
                        self.process.kill()
                        self.process.wait(timeout=1)
                    except Exception:
                        pass
            self.process = None
        
        if hasattr(self, 'script_path') and os.path.exists(self.script_path):
            try:
                os.unlink(self.script_path)
            except Exception:
                pass
    
    @property
    def is_running(self) -> bool:  # DEAD CODE — unused, kept for external callers
        return self.process is not None and self.process.poll() is None


# ---------------------------------------------------------------------------
# Minimal 9P2000 client for Plan9-style streaming reads
# ---------------------------------------------------------------------------

# 9P2000 message types
_Tversion = 100; _Rversion = 101
_Tauth    = 102; _Rauth    = 103
_Tattach  = 104; _Rattach  = 105
_Rerror   = 107
_Twalk    = 110; _Rwalk    = 111
_Topen    = 112; _Ropen    = 113
_Tread    = 116; _Rread    = 117
_Twrite   = 118; _Rwrite   = 119
_Tclunk   = 120; _Rclunk   = 121

_NOTAG = 0xFFFF
_NOFID = 0xFFFFFFFF


class P9Error(Exception):
    """Error returned by the 9P server (Rerror)."""
    pass


class P9Client:
    """
    Minimal 9P2000 client that speaks the wire protocol directly over TCP.

    This bypasses the Linux kernel's VFS / page cache / read-ahead entirely.
    Each Tread returns immediately with whatever data the server has — exactly
    the same behaviour as Plan 9's cat.
    """

    def __init__(self, host: str = "localhost", port: int = 5640,
                 auth_token: str = None):
        self.host = host
        self.port = port
        # Token used for the raw-token 9P auth handshake. When None,
        # we still attempt Tattach with NOFID (works against unauthed
        # servers and rejects with "authentication required" against
        # authed ones — which is exactly the error message that surfaces
        # in the UI as "Stream error: 9P connect failed: ...").
        self.auth_token = auth_token
        self.sock: socket.socket = None
        self.msize = 8192
        self._tag = 0
        self._fids = {}       # path_key -> fid
        self._next_fid = 1
        self._root_fid = 0

    # ---- connection lifecycle -----------------------------------------

    def connect(self):
        """Connect and perform Tversion, optional Tauth, then Tattach."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.host, self.port))
        # No Nagle — we want low latency
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Tversion
        self._version()

        # Optional raw-token auth. If self.auth_token is set we do the
        # Tauth → Twrite(token) → Tread("ok\n") dance and Tattach with
        # the resulting afid. If unset, _authenticate returns NOFID and
        # we Tattach unauthenticated (which the server allows only when
        # auth is disabled). Mirrors riomux's BackendConnection._authenticate.
        afid = self._authenticate()

        # Tattach with root fid 0
        self._attach(afid=afid)

    def close(self):
        """Clunk all open fids and close socket."""
        if self.sock is None:
            return
        for fid in list(self._fids.values()):
            try:
                self._clunk(fid)
            except Exception:
                pass
        self._fids.clear()
        try:
            self.sock.close()
        except Exception:
            pass
        self.sock = None

    @property
    def connected(self) -> bool:
        return self.sock is not None

    # ---- public API ---------------------------------------------------

    def walk_open(self, path: str, mode: int = 0) -> int:
        """
        Walk to *path* (relative to root) and open.
        Returns the fid.  Caches so repeated calls reuse the fid.
        
        mode: 9P open mode. Default 0 = OREAD.
        """
        if path in self._fids:
            return self._fids[path]

        fid = self._alloc_fid()
        elements = [e for e in path.split("/") if e]

        # Twalk from root
        payload = struct.pack("<II", self._root_fid, fid)
        payload += struct.pack("<H", len(elements))
        for e in elements:
            eb = e.encode("utf-8")
            payload += struct.pack("<H", len(eb)) + eb

        resp = self._rpc(_Twalk, payload)
        rtype = resp[0]
        if rtype == _Rerror:
            self._parse_error(resp)

        # Topen
        payload = struct.pack("<IB", fid, mode)
        resp = self._rpc(_Topen, payload)
        rtype = resp[0]
        if rtype == _Rerror:
            self._parse_error(resp)

        self._fids[path] = fid
        return fid

    def read(self, fid: int, offset: int, count: int = 0) -> bytes:
        """
        Issue a single Tread and return whatever the server sends back.
        A short read (including 0 bytes) is perfectly normal for a stream.
        """
        if count <= 0:
            count = self.msize - 24   # leave room for 9P header
        payload = struct.pack("<IQI", fid, offset, count)
        resp = self._rpc(_Tread, payload)
        rtype = resp[0]
        if rtype == _Rerror:
            self._parse_error(resp)

        # Rread: type[1] tag[2] count[4] data[count]
        data_count = struct.unpack_from("<I", resp, 3)[0]
        return resp[7 : 7 + data_count]

    def close_fid(self, path: str):
        """Clunk a previously opened fid."""
        fid = self._fids.pop(path, None)
        if fid is not None:
            try:
                self._clunk(fid)
            except Exception:
                pass

    # ---- 9P2000 primitives --------------------------------------------

    def _version(self):
        ver = b"9P2000"
        payload = struct.pack("<I", self.msize)
        payload += struct.pack("<H", len(ver)) + ver
        resp = self._rpc(_Tversion, payload, tag=_NOTAG)
        # Rversion: type[1] tag[2] msize[4] version[s]
        server_msize = struct.unpack_from("<I", resp, 3)[0]
        self.msize = min(self.msize, server_msize)

    def _authenticate(self) -> int:
        """
        Raw-token 9P auth handshake. Returns the afid to quote in
        Tattach, or _NOFID when no token is configured.
        
        Protocol (matches ninep.auth.AuthFid's raw-token mode):
          1. Tauth(afid, uname, aname)             → server creates AuthFid
          2. Twrite(afid, token_bytes)             → server validates token
          3. Tread(afid)                           → server returns "ok\\n"
                                                     or "err: ...\\n"
        
        Plays nicely against an unauthed server too: if there's no
        token configured, this returns NOFID and the caller's Tattach
        proceeds as before.
        """
        if not self.auth_token:
            return _NOFID
        
        afid = self._alloc_fid()
        uname = b"rio"
        aname = b""
        
        # ── Tauth ─────────────────────────────────────────────────
        payload = struct.pack("<I", afid)
        payload += struct.pack("<H", len(uname)) + uname
        payload += struct.pack("<H", len(aname)) + aname
        resp = self._rpc(_Tauth, payload)
        rtype = resp[0]
        if rtype == _Rerror:
            self._parse_error(resp)   # raises P9Error
        if rtype != _Rauth:
            raise P9Error(f"Tauth: unexpected reply type {rtype}")
        
        # ── Twrite the token ──────────────────────────────────────
        token_bytes = self.auth_token.encode("utf-8")
        payload = struct.pack("<IQI", afid, 0, len(token_bytes)) + token_bytes
        resp = self._rpc(_Twrite, payload)
        rtype = resp[0]
        if rtype == _Rerror:
            self._parse_error(resp)
        if rtype != _Rwrite:
            raise P9Error(f"auth Twrite: unexpected reply type {rtype}")
        
        # ── Tread to confirm status ───────────────────────────────
        payload = struct.pack("<IQI", afid, 0, 256)
        resp = self._rpc(_Tread, payload)
        rtype = resp[0]
        if rtype == _Rerror:
            self._parse_error(resp)
        if rtype != _Rread:
            raise P9Error(f"auth Tread: unexpected reply type {rtype}")
        data_count = struct.unpack_from("<I", resp, 3)[0]
        status = resp[7:7 + data_count].rstrip(b"\0").decode("utf-8", "replace").strip()
        if not status.startswith("ok"):
            raise P9Error(f"auth rejected: {status or 'no status'}")
        
        return afid
    
    def _attach(self, afid: int = _NOFID):
        uname = b"rio"
        aname = b""
        payload = struct.pack("<II", self._root_fid, afid)
        payload += struct.pack("<H", len(uname)) + uname
        payload += struct.pack("<H", len(aname)) + aname
        resp = self._rpc(_Tattach, payload)
        rtype = resp[0]
        if rtype == _Rerror:
            self._parse_error(resp)

    def _clunk(self, fid: int):
        payload = struct.pack("<I", fid)
        self._rpc(_Tclunk, payload)

    # ---- wire format --------------------------------------------------

    def _alloc_fid(self) -> int:
        fid = self._next_fid
        self._next_fid += 1
        return fid

    def _next_tag(self) -> int:
        self._tag = (self._tag + 1) & 0x7FFF
        return self._tag

    def _rpc(self, msg_type: int, payload: bytes, tag: int = None) -> bytes:
        """Send a T-message, receive and return the R-message body."""
        if tag is None:
            tag = self._next_tag()

        # Build message: size[4] type[1] tag[2] payload...
        header = struct.pack("<IBH", 4 + 1 + 2 + len(payload), msg_type, tag)
        self.sock.sendall(header + payload)

        # Read response: size[4] then rest
        size_buf = self._recv_exact(4)
        size = struct.unpack("<I", size_buf)[0]
        body = self._recv_exact(size - 4)
        return body   # body[0]=type, body[1:3]=tag, body[3:]=data

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly n bytes from socket."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("9P server closed connection")
            buf.extend(chunk)
        return bytes(buf)

    def _parse_error(self, resp: bytes):
        """Parse an Rerror response and raise P9Error."""
        # Rerror: type[1] tag[2] ename[s]
        ename_len = struct.unpack_from("<H", resp, 3)[0]
        ename = resp[5 : 5 + ename_len].decode("utf-8", errors="replace")
        raise P9Error(ename)


# ---------------------------------------------------------------------------
# Plan9-style output stream reader using raw 9P
# ---------------------------------------------------------------------------

class OutputStreamReader(QThread):
    """
    Plan 9 state-aware output reader.
    
    Acts like `while true; do cat $agent/OUTPUT; done` using raw 9P.
    
    Because StreamFile is now state-aware:
    - open()+read() blocks until a generation starts (generation gate)
    - read() streams data as it arrives
    - read() returns b"" on EOF (generation done)
    - Re-open blocks again until the next generation
    
    NO POLLING. Zero CPU in steady state. Pure blocking I/O.
    """

    new_data = Signal(str)
    stream_reset = Signal()
    stream_done = Signal()
    error_occurred = Signal(str)

    def __init__(self, agent_path: str, host: str = "localhost", port: int = 5640,
                 auth_token: str = None):
        super().__init__()
        self.agent_path = agent_path
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._running = True

    def run(self):
        client = P9Client(self.host, self.port, auth_token=self.auth_token)
        try:
            client.connect()
        except Exception as e:
            self.error_occurred.emit(f"9P connect failed: {e}")
            return

        # Remove socket timeout — blocking reads wait for LLM tokens
        # and the generation gate can block indefinitely between generations
        client.sock.settimeout(None)

        output_path = f"{self.agent_path}/OUTPUT"

        while self._running:
            try:
                # Walk+Open the output file. On the server side,
                # StreamFile.read() blocks on the generation gate if idle,
                # so this reader naturally sleeps until a generation starts.
                output_fid = client.walk_open(output_path)
                position = 0

                while self._running:
                    # This call blocks on the 9P server:
                    # - If waiting for generation: blocks on generation gate
                    # - If streaming: blocks until next chunk arrives
                    # - Returns b"" on EOF (generation complete)
                    data = client.read(output_fid, position, count=4096)
                    
                    if data:
                        text = data.decode("utf-8", errors="replace")
                        self.new_data.emit(text)
                        position += len(data)
                    else:
                        # EOF reached (generation finished)
                        self.stream_done.emit()
                        # Close FID — next iteration will re-open and block
                        # on the generation gate until the next generation starts
                        client.close_fid(output_path)
                        break 
                
                # NO SLEEP NEEDED — the next walk_open+read will block
                # on the server-side generation gate automatically

            except (ConnectionError, BrokenPipeError, OSError) as e:
                self.error_occurred.emit(f"9P connection lost: {e}")
                break
            except Exception as e:
                self.error_occurred.emit(f"Stream error: {e}")
                time.sleep(1.0)

        client.close()

    def stop(self):
        self._running = False

# ---------------------------------------------------------------------------
# Master Agent - Bash Router (reads $master/BASH, executes in terminal)
# ---------------------------------------------------------------------------

class MasterBashReader(QThread):
    """
    Reads from the master agent's 'BASH' supplementary output file
    using raw 9P and emits each command for the terminal to execute.

    Because SupplementaryOutputFile is now state-aware:
    - read() blocks until plumbing extracts a ```bash block and marks ready
    - Returns content, then returns b"" (EOF)
    - Re-open and read again: blocks until the next generation
    
    This is `while true; do cat $master/BASH; done` over raw 9P.
    NO POLLING. Zero CPU in steady state.
    """

    command_ready = Signal(str)   # shell command to execute
    error_occurred = Signal(str)
    finished_signal = Signal()

    def __init__(self, agent_path: str = "master",
                 host: str = "localhost", port: int = 5640,
                 auth_token: str = None,
                 **_kwargs):
        super().__init__()
        self.agent_path = agent_path
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._running = True

    def run(self):
        client = P9Client(self.host, self.port, auth_token=self.auth_token)
        try:
            client.connect()
        except Exception as e:
            self.error_occurred.emit(f"MasterBashReader: 9P connect failed: {e}")
            return

        # Remove socket timeout — reads on SupplementaryOutputFile block
        # until the LLM generation completes, which can take 30+ seconds.
        client.sock.settimeout(None)

        bash_path = f"{self.agent_path}/BASH"

        while self._running:
            try:
                # Open the supplementary output file — server blocks
                # on _content_ready until plumbing extracts content
                fid = client.walk_open(bash_path)
                position = 0
                accumulated = ""

                while self._running:
                    # Blocking 9P read — suspends on server side until
                    # plumbing extracts content and mark_ready() fires.
                    # On first read of a new generation, this blocks on the
                    # state gate until content is available.
                    data = client.read(fid, position, count=4096)

                    if data:
                        text = data.decode("utf-8", errors="replace")
                        accumulated += text
                        position += len(data)
                    else:
                        # EOF — generation done, process what we got
                        if accumulated.strip():
                            for line in accumulated.strip().split('\n'):
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    self.command_ready.emit(line)
                        accumulated = ""
                        client.close_fid(bash_path)
                        break

                # NO SLEEP NEEDED — the next walk_open+read will block
                # on the server-side state gate automatically

            except (ConnectionError, BrokenPipeError, OSError) as e:
                if not self._running:
                    break
                self.error_occurred.emit(f"MasterBashReader: connection lost: {e}")
                break
            except P9Error as e:
                # File might not exist yet (rule not yet added), retry
                if not self._running:
                    break
                time.sleep(1.0)
            except Exception as e:
                if not self._running:
                    break
                self.error_occurred.emit(f"MasterBashReader: {e}")
                time.sleep(1.0)

        try:
            client.close()
        except Exception:
            pass
        self.finished_signal.emit()

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Plan 9 Mouse Menu - press to open, release to select
# ---------------------------------------------------------------------------

class Plan9MenuFilter(QObject):
    """
    Event filter implementing Plan 9-style right-click menus.

    Behaviour:
      - Right mouse button PRESS  → menu appears under the cursor
      - Mouse MOVE (button held)  → items highlight as the pointer passes
      - Right mouse button RELEASE on an item → that action fires
      - Release outside the menu  → menu closes, nothing happens

    This is how acme / rio / sam menus work: one fluid press-drag-release
    gesture, much faster than the conventional click-to-open, click-to-select.
    """

    def __init__(self, terminal):
        super().__init__(terminal)
        self.terminal = terminal
        self._menu = None
        self._actions = {}

    def eventFilter(self, obj, event):
        from PySide6.QtGui import QMouseEvent
        if event.type() == QMouseEvent.Type.MouseButtonPress and event.button() == Qt.RightButton:
            # Find the QTextEdit that owns this viewport
            text_edit = obj.parent()
            if not isinstance(text_edit, QTextEdit):
                return False

            self._source_edit = text_edit
            self._build_and_show_menu(event.globalPosition().toPoint())
            return True  # swallow the press

        if event.type() == QMouseEvent.Type.MouseButtonRelease and event.button() == Qt.RightButton:
            if self._menu and self._menu.isVisible():
                # Find which action is under the cursor
                action = self._menu.actionAt(self._menu.mapFromGlobal(event.globalPosition().toPoint()))
                self._menu.hide()
                if action and not action.isSeparator():
                    action.trigger()
                self._menu = None
                return True  # swallow the release

        return False

    def _build_and_show_menu(self, global_pos):
        from PySide6.QtWidgets import QMenu

        _CSS_NORMAL = (
            "QMenu { background-color: rgba(255,255,255,200); border: 1px solid #000000;"
            " padding: 2px 0px; font-family: 'Consolas','Monaco',monospace; font-size: 12px; }"
            " QMenu::item { color: #000000; padding: 4px 20px 4px 10px; }"
            " QMenu::item:selected { background-color: rgba(0,0,0,242); color: #ffffff; }"
            " QMenu::separator { height: 1px; background: #000000; margin: 2px 4px; }"
        )
        _CSS_FLASH = (
            "QMenu { background-color: rgba(0,0,0,242); border: 1px solid #000000;"
            " padding: 2px 0px; font-family: 'Consolas','Monaco',monospace; font-size: 12px; }"
            " QMenu::item { color: #ffffff; padding: 4px 20px 4px 10px; }"
            " QMenu::item:selected { background-color: rgba(255,255,255,242); color: #000000; }"
            " QMenu::separator { height: 1px; background: #ffffff; margin: 2px 4px; }"
        )

        # Custom menu with blink-on-select (matches Rio main window)
        class _BlinkMenu(QMenu):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._blink_active = False
            def mouseReleaseEvent(self, event):
                action = self.actionAt(event.pos())
                if action and action.isEnabled() and not action.isSeparator():
                    self._blink_active = True
                    self.triggered.emit(action)
                    event.accept()
                    return  # don't call super — prevents auto-close
                super().mouseReleaseEvent(event)

        menu = _BlinkMenu()
        menu.setStyleSheet(_CSS_NORMAL)

        te = self._source_edit
        has_selection = te.textCursor().hasSelection()
        selected_text = te.textCursor().selectedText().strip() if has_selection else ""

        _action_map = {}

        def _add(label, callback, enabled=True):
            action = menu.addAction(label)
            action.setEnabled(enabled)
            _action_map[action] = callback
            return action

        def _on_triggered(action):
            cb = _action_map.get(action)
            if cb is None or not menu._blink_active:
                return
            menu._blink_active = False
            # Single blink: invert, hold, revert, close
            _step = [0]
            def _tick():
                _step[0] += 1
                if _step[0] == 1:
                    menu.setStyleSheet(_CSS_FLASH)
                elif _step[0] == 2:
                    menu.setStyleSheet(_CSS_NORMAL)
                else:
                    _timer.stop()
                    _timer.deleteLater()
                    menu.close()
                    QTimer.singleShot(0, cb)
                    return
            _timer = QTimer(menu)
            _timer.timeout.connect(_tick)
            _timer.start(80)

        menu.triggered.connect(_on_triggered)

        # --- Menu actions ---
        _add("Send", lambda: self._do_send(selected_text), enabled=bool(selected_text))

        menu.addSeparator()

        _add("Cut", te.cut, enabled=has_selection)
        _add("Snarf", te.copy, enabled=has_selection)
        _add("Paste", te.paste)

        menu.addSeparator()

        _add("→ Input", lambda: self._do_to_input(selected_text), enabled=bool(selected_text))
        _add("Plumb", lambda: self._do_plumb(selected_text), enabled=bool(selected_text))

        self._menu = menu
        menu.popup(global_pos)

    def _do_send(self, text):
        """Send selected text as shell command(s)."""
        for line in text.replace('\u2029', '\n').split('\n'):
            line = line.strip()
            if line:
                self.terminal._execute_shell(line, echo=True)

    def _do_to_input(self, text):
        """Copy selected text into the command input field."""
        self.terminal.command_input.setPlainText(text.replace('\u2029', '\n'))
        cursor = self.terminal.command_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.terminal.command_input.setTextCursor(cursor)
        self.terminal.command_input.setFocus()

    def _do_plumb(self, text):
        """
        Extract fenced code blocks of the form ```machine_name\\ncode\\n```
        and write the code to /n/machine_name/scene/parse.

        The write targets a 9P filesystem which may block, so all I/O is
        done on a background thread.  GUI feedback is marshalled back to
        the main thread via QTimer.singleShot(0, ...).
        """
        text = text.replace('\u2029', '\n')
        # Match ```machine_name\ncode\n``` blocks
        pattern = re.compile(r'```(\S+)\s*\n(.*?)```', re.DOTALL)
        matches = pattern.findall(text)
        if not matches:
            self.terminal._append_output("[plumb] no ```machine_name code``` block found in selection\n")
            return

        terminal = self.terminal  # prevent closure over self

        def _write_all():
            for machine_name, code in matches:
                target = f"/n/{machine_name}/scene/parse"
                try:
                    with open(target, 'w') as f:
                        f.write(code)
                    msg = f"[plumb] wrote to {target}\n"
                except Exception as e:
                    msg = f"[plumb] error writing to {target}: {e}\n"
                # Marshal GUI update back to the main thread
                QTimer.singleShot(0, lambda m=msg: terminal._append_output(m))

        threading.Thread(target=_write_all, daemon=True).start()


# ---------------------------------------------------------------------------
# Inline output widgets
#
# These are embedded in terminal_content_layout alongside the regular
# QTextEdit displays. The fence parser detects ```machine ... ``` in
# streaming agent text and morphs them into live InlineCodeBlockWidgets.
# Media widgets can also be spawned via the $term/inline filesystem file.
# ---------------------------------------------------------------------------


class InlineCodeBlockWidget(QFrame):
    """Collapsible, editable, runnable code block embedded inline in the terminal.
    
    Each widget targets a "machine" — written as ```machine_name in the agent's
    output stream. Clicking Run writes the (possibly edited) contents to
    /n/<machine>/scene/parse, which routes through the 9P filesystem to the
    machine's parse handler.
    
    The widget is dark/light-mode aware and matches the host terminal's theme.
    """
    
    def __init__(self, machine_name: str = "", llmfs_mount: str = "/n", 
                 dark_mode: bool = False, max_width: int = 0,
                 host_terminal=None, parent=None):
        super().__init__(parent)
        self.machine_name = machine_name
        self.llmfs_mount = llmfs_mount  # base mount, e.g. "/n" — target = $mount/<machine>/scene/parse
        self.code_text = ""
        self.is_streaming = True
        self.is_expanded = True
        self.is_edited = False
        self.original_code = ""
        self._dark_mode = dark_mode
        # Optional weakref back to the host TerminalWidget. Used by
        # variable expansion in the editable target-path dialog so that
        # `$term`, `$inline`, `$LLMFS`, `$RIO`, `$peribus` etc. resolve
        # to whatever the host terminal seeded into its shell. We use
        # weakref to avoid creating a parent/child reference cycle that
        # outlives the terminal.
        import weakref as _weakref
        self._host_terminal_ref = _weakref.ref(host_terminal) if host_terminal is not None else None
        # User-supplied custom Run target. None == legacy behaviour
        # (writes to $mount/<machine>/scene/parse). When non-None, this
        # is the *raw* string the user typed — variable expansion happens
        # at run time so changes to the host terminal's environment
        # (e.g. /restart re-seeds vars with a new term_id) are picked up
        # automatically.
        self.custom_target_raw: Optional[str] = None
        # Coalesced height-update timer: streaming chunks would otherwise
        # call setFixedHeight() on every char (a layout invalidation each
        # time). Batch via a single-shot timer that fires ~30 ms after
        # the last char arrives.
        self._size_update_timer = QTimer(self)
        self._size_update_timer.setSingleShot(True)
        self._size_update_timer.setInterval(30)
        self._size_update_timer.timeout.connect(self._update_size)
        self._setup_ui()
        if max_width > 0:
            self.set_inline_max_width(max_width)
    
    def set_inline_max_width(self, max_w: int):
        """Clamp this widget's width to fit the terminal's scroll-area
        viewport. Called at construction and on terminal resize."""
        if max_w <= 0:
            return
        self.setMaximumWidth(max_w)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    
    def _setup_ui(self):
        self.setFrameStyle(QFrame.NoFrame)
        self._apply_frame_theme()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        
        # Clickable header strip with toggle (stretching) + close (✕).
        # Same pattern as InlineMediaWidget — two distinct click targets
        # in one row instead of a single full-width button.
        header_strip = QWidget()
        header_strip.setStyleSheet("background: transparent;")
        header_layout = QHBoxLayout(header_strip)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        # The "title" slot is a stack: in display mode it shows the
        # header button (▼ → /n/.../parse · 12 lines), in edit mode it
        # shows a QLineEdit pre-filled with the current target so the
        # user can retype the path inline. Right-click flips the stack
        # to edit mode; Enter commits, Escape (or focus-loss) cancels.
        # Using a stack keeps the layout geometry stable — no jitter
        # when switching modes.
        self._title_stack_widget = QWidget()
        title_stack = QStackedLayout(self._title_stack_widget)
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setStackingMode(QStackedLayout.StackOne)
        self._title_stack = title_stack

        self.header_btn = QPushButton()
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.clicked.connect(self.toggle_code)
        # Right-press on the button switches to inline edit. Filtered
        # via eventFilter because QPushButton's customContextMenu
        # signal is unreliable on X11 (the press handler swallows it).
        self.header_btn.installEventFilter(self)
        title_stack.addWidget(self.header_btn)

        self.title_edit = QLineEdit()
        self.title_edit.setFrame(False)
        self.title_edit.setClearButtonEnabled(False)
        # Enter commits, Escape cancels — both handled in eventFilter
        # so we get one place that owns the keystroke logic.
        self.title_edit.installEventFilter(self)
        self.title_edit.editingFinished.connect(self._commit_title_edit)
        title_stack.addWidget(self.title_edit)
        title_stack.setCurrentWidget(self.header_btn)

        # Right-press on the surrounding strip also activates edit
        # mode, so the user has a generous click target.
        header_strip.installEventFilter(self)
        self._header_strip = header_strip

        header_layout.addWidget(self._title_stack_widget, stretch=1)

        self.close_btn = QPushButton("✕")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setToolTip("Remove this code block")
        self.close_btn.clicked.connect(self._close_widget)
        header_layout.addWidget(self.close_btn)
        
        self._apply_header_theme()
        self._update_header_text()
        
        # Code edit area
        self.code_frame = QFrame()
        self.code_frame.setFrameStyle(QFrame.NoFrame)
        code_layout = QVBoxLayout(self.code_frame)
        code_layout.setContentsMargins(4, 2, 4, 4)
        code_layout.setSpacing(3)
        
        self.code_edit = QTextEdit()
        self.code_edit.setReadOnly(False)
        self.code_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.code_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.code_edit.textChanged.connect(self._on_code_edited)
        self._apply_code_edit_theme()
        
        # Button row
        button_row = QHBoxLayout()
        button_row.setSpacing(4)
        
        self.run_btn = self._make_button("▶ Run", primary=True)
        self.run_btn.clicked.connect(self.run_code)
        
        self.copy_btn = self._make_button("Copy")
        self.copy_btn.clicked.connect(self.copy_code)
        
        self.reset_btn = self._make_button("↺ Reset")
        self.reset_btn.clicked.connect(self.reset_code)
        self.reset_btn.hide()
        
        self.streaming_label = QLabel("● streaming")
        self.streaming_label.setStyleSheet(
            "color: #ffaa00; font-size: 10px; font-weight: bold; padding: 0 6px;"
        )
        
        self.edited_label = QLabel("✎ edited")
        self.edited_label.setStyleSheet(
            "color: #5588ff; font-size: 10px; font-weight: bold; padding: 0 6px;"
        )
        self.edited_label.hide()
        
        button_row.addWidget(self.run_btn)
        button_row.addWidget(self.copy_btn)
        button_row.addWidget(self.reset_btn)
        button_row.addWidget(self.streaming_label)
        button_row.addWidget(self.edited_label)
        button_row.addStretch()
        
        code_layout.addWidget(self.code_edit)
        code_layout.addLayout(button_row)
        
        layout.addWidget(header_strip)
        layout.addWidget(self.code_frame)
        
        # Start expanded but compact
        self._update_size()
    
    def _make_button(self, label: str, primary: bool = False) -> QPushButton:
        btn = QPushButton(label)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(22)
        if primary:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(80, 170, 90, 220);
                    border: none;
                    border-radius: 3px;
                    padding: 2px 12px;
                    color: white;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: rgba(100, 190, 110, 240);
                }
                QPushButton:pressed {
                    background-color: rgba(70, 150, 80, 255);
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(180, 180, 180, 60);
                    border: 1px solid rgba(160, 160, 160, 80);
                    border-radius: 3px;
                    padding: 2px 10px;
                    color: inherit;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: rgba(180, 180, 180, 110);
                }
            """)
        return btn
    
    # ------------------------------------------------------------------
    # Theming (dark/light)
    # ------------------------------------------------------------------
    
    def _apply_frame_theme(self):
        if self._dark_mode:
            self.setStyleSheet("""
                InlineCodeBlockWidget {
                    background-color: rgba(40, 42, 52, 180);
                    border: 1px solid rgba(90, 100, 120, 140);
                    border-radius: 5px;
                    margin: 4px 0px;
                }
            """)
        else:
            self.setStyleSheet("""
                InlineCodeBlockWidget {
                    background-color: rgba(245, 247, 250, 180);
                    border: 1px solid rgba(180, 190, 210, 140);
                    border-radius: 5px;
                    margin: 4px 0px;
                }
            """)
    
    def _apply_header_theme(self):
        if self._dark_mode:
            txt_color = "rgba(220, 225, 235, 255)"
            hover_bg = "rgba(60, 65, 80, 200)"
            close_color = "rgba(220, 225, 235, 200)"
        else:
            txt_color = "rgba(40, 50, 70, 255)"
            hover_bg = "rgba(220, 225, 235, 200)"
            close_color = "rgba(0, 0, 0, 200)"
        self.header_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 3px;
                padding: 6px 8px;
                color: {txt_color};
                font-weight: bold;
                text-align: left;
                font-size: 11px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
        """)
        # Close (✕) button: paired with the toggle. Subtle by default,
        # red-tinted on hover so it reads as destructive.
        if hasattr(self, 'close_btn') and self.close_btn is not None:
            self.close_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                    border-radius: 3px;
                    color: {close_color};
                    font-size: 13px;
                    font-weight: bold;
                    padding: 0px;
                }}
                QPushButton:hover {{
                    background-color: rgba(220, 60, 60, 180);
                    color: rgba(255, 255, 255, 255);
                }}
                QPushButton:pressed {{
                    background-color: rgba(180, 40, 40, 220);
                }}
            """)
        # Title edit field — same font/color as the header button so
        # entering edit mode reads as "the title became typeable",
        # not "a different control appeared". The 1-px accent border
        # is the only visual cue that we're in edit mode.
        if hasattr(self, 'title_edit') and self.title_edit is not None:
            if self._dark_mode:
                edit_bg = "rgba(50, 55, 70, 220)"
                edit_border = "rgba(85, 136, 255, 200)"
                sel_bg = "rgba(80, 130, 200, 120)"
            else:
                edit_bg = "rgba(255, 255, 255, 220)"
                edit_border = "rgba(85, 136, 255, 200)"
                sel_bg = "rgba(80, 130, 200, 80)"
            self.title_edit.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {edit_bg};
                    border: 1px solid {edit_border};
                    border-radius: 3px;
                    padding: 5px 8px;
                    color: {txt_color};
                    font-weight: bold;
                    font-size: 11px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    selection-background-color: {sel_bg};
                }}
            """)
    
    def _apply_code_edit_theme(self):
        # The text-area background is fully transparent on purpose. The
        # parent terminal applies focus-in/focus-out opacity to the whole
        # frame, and we want the embedded code to follow that — not paint
        # a second opaque rectangle on top. Foreground (text), border
        # tint and selection stay theme-aware.
        if self._dark_mode:
            fg = "rgba(220, 225, 235, 255)"
            border = "rgba(80, 90, 110, 120)"
            sel = "rgba(80, 130, 200, 100)"
        else:
            fg = "rgba(20, 25, 35, 255)"
            border = "rgba(170, 180, 200, 120)"
            sel = "rgba(80, 130, 200, 80)"
        self.code_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba(0, 0, 0, 0);
                color: {fg};
                border: 1px solid {border};
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 4px;
                selection-background-color: {sel};
            }}
            QTextEdit:focus {{
                border: 1px solid rgba(85, 136, 255, 180);
            }}
        """)
        # Without this attribute, QTextEdit's viewport still paints an
        # opaque base color underneath the stylesheet's transparent
        # background. Mark it translucent so transparency reaches through.
        try:
            from PySide6.QtCore import Qt as _Qt
            self.code_edit.setAttribute(_Qt.WA_TranslucentBackground, True)
            vp = self.code_edit.viewport()
            if vp is not None:
                vp.setAutoFillBackground(False)
        except Exception:
            pass
    
    def set_dark_mode(self, dark: bool):
        """Switch theme; called by the host terminal on dark-mode toggle."""
        self._dark_mode = dark
        self._apply_frame_theme()
        self._apply_header_theme()
        self._apply_code_edit_theme()
    
    # ------------------------------------------------------------------
    # Streaming + state
    # ------------------------------------------------------------------
    
    def append_code(self, text_chunk: str):
        """Append a chunk of streaming code text.
        
        Performance: header-text and size updates are coalesced via a
        ~30ms single-shot timer instead of running on every char,
        because streaming agents may emit one char per network frame
        and per-char setFixedHeight() invalidates the entire layout.
        """
        if not text_chunk:
            return
        self.code_text += text_chunk
        cursor = self.code_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text_chunk)
        # Coalesce header + size update; this also fires _update_size,
        # which itself updates the header text now (cheap).
        self._size_update_timer.start()
    
    def finalize_streaming(self):
        """Mark streaming complete; record original code for reset/diff."""
        if not self.is_streaming:
            return
        self.is_streaming = False
        self.streaming_label.hide()
        self.original_code = self.code_edit.toPlainText()
        # Cancel any pending coalesced update and run final layout once.
        self._size_update_timer.stop()
        self._update_size()
    
    def _on_code_edited(self):
        if self.is_streaming:
            return  # streaming changes don't count as user edits
        current = self.code_edit.toPlainText()
        if current != self.original_code:
            if not self.is_edited:
                self.is_edited = True
                self.edited_label.show()
                self.reset_btn.show()
        else:
            if self.is_edited:
                self.is_edited = False
                self.edited_label.hide()
                self.reset_btn.hide()
    
    def reset_code(self):
        """Restore the original streamed code."""
        self.code_edit.blockSignals(True)
        self.code_edit.setPlainText(self.original_code)
        self.code_edit.blockSignals(False)
        self.is_edited = False
        self.edited_label.hide()
        self.reset_btn.hide()
    
    def copy_code(self):
        QApplication.clipboard().setText(self.code_edit.toPlainText())
    
    def toggle_code(self):
        self.is_expanded = not self.is_expanded
        self.code_frame.setVisible(self.is_expanded)
        self._update_header_text()
        self._update_size()
    
    def _close_widget(self):
        """User clicked the ✕ — remove this code block from the terminal.
        
        We cancel any in-flight Run subprocess so it doesn't end up
        writing to a path the user clearly no longer cares about, then
        detach from the parent layout and schedule deletion.
        """
        # Cancel any running async Run cleanly so its callbacks don't
        # try to flash status onto a deleted widget.
        try:
            for attr in ('_run_open_worker',):
                w = getattr(self, attr, None)
                if w is not None:
                    try:
                        w.quit()
                        w.wait(500)
                    except Exception:
                        pass
        except Exception:
            pass
        # Stop any pending coalesced layout timer so it doesn't fire
        # against a deleted widget after this returns.
        try:
            if hasattr(self, '_size_update_timer'):
                self._size_update_timer.stop()
        except Exception:
            pass
        # Detach from the parent layout so the surrounding widgets reflow
        # immediately. deleteLater alone leaves a layout-claimed gap
        # until the next event-loop pass.
        parent = self.parent()
        if parent is not None and parent.layout() is not None:
            parent.layout().removeWidget(self)
        self.setParent(None)
        self.deleteLater()
    
    def _update_header_text(self):
        arrow = "▼" if self.is_expanded else "▶"
        # Prefer the raw user-typed custom target (so the user sees
        # `$inline` or `$term/foo` literally rather than the resolved
        # absolute path — easier to read and to spot mistakes). Fall
        # back to the auto-derived $mount/<machine>/scene/parse.
        if self.custom_target_raw:
            target = f" → {self.custom_target_raw}"
        elif self.machine_name:
            target = f" → {self.target_path}"
        else:
            target = " [no target]"
        lines = self.code_text.count('\n') + (1 if self.code_text else 0)
        live = " ●" if self.is_streaming else ""
        self.header_btn.setText(f"{arrow}{target} · {lines} lines{live}")
    
    def _update_size(self):
        """Resize the code editor based on current content.
        
        Also refreshes the header text so the line count and streaming
        indicator stay in sync — both are coalesced behind one timer.
        """
        self._update_header_text()
        if not self.is_expanded:
            return
        lines = max(self.code_text.count('\n') + 1, 1)
        # Roughly match font metrics; clamp to a reasonable range.
        height = min(max(lines * 16 + 12, 60), 400)
        # Only call setFixedHeight if it actually changed — even a no-op
        # call triggers a layout invalidation in some Qt versions.
        if self.code_edit.height() != height:
            self.code_edit.setFixedHeight(height)
    
    # ------------------------------------------------------------------
    # Run target
    # ------------------------------------------------------------------
    
    @property
    def target_path(self) -> str:
        """Where Run writes the code.
        
        Resolution order:
          1. If the user has set a custom target (right-click → Set
             Target Path…), expand $variables against the host
             terminal's environment and return the result.
          2. Otherwise, fall back to legacy
             $mount/<machine>/scene/parse for the agent's machine.
        
        Variable expansion is deferred until property-read time so a
        path like `$inline` (= $term/inline) automatically picks up
        the new term_id if the host terminal is /restart-ed.
        """
        if self.custom_target_raw:
            return self._resolve_target(self.custom_target_raw)
        if not self.machine_name:
            return ""
        return os.path.join(self.llmfs_mount, self.machine_name, "scene", "parse")

    def _resolve_target(self, raw: str) -> str:
        """Expand $VAR / ${VAR} in `raw` against the host terminal env.
        
        Variable sources, checked in order:
          1. Variables seeded by the host terminal — `LLMFS`, `RIO`,
             `term`, `inline`, `peribus`, plus anything else exposed
             by `host_terminal._terminal_variables()` (subclasses may
             override that hook to expose more).
          2. Process environment via os.environ — this catches generic
             vars like `$HOME` or `$USER` so users aren't surprised.
        
        Unknown variables are left literal (e.g. `$nope` stays
        `$nope`) rather than expanding to empty — empty expansion
        silently breaks paths and is harder to debug than a literal.
        Tilde (`~`) is also expanded for ergonomic typing.
        
        Trailing whitespace is trimmed because copy-pasted paths
        often carry a stray newline.
        """
        s = (raw or "").strip()
        if not s:
            return ""
        env = self._terminal_env()

        # Walk $VAR / ${VAR} occurrences. We do our own pass instead of
        # os.path.expandvars so unknown vars stay literal rather than
        # silently disappearing.
        def _sub(match):
            name = match.group(1) or match.group(2)
            if name in env:
                return env[name]
            return match.group(0)  # leave literal if unknown

        s = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)', _sub, s)
        s = os.path.expanduser(s)
        return s

    def _terminal_env(self) -> Dict[str, str]:
        """Build the variable map used by _resolve_target.
        
        Pulls from the host terminal first (so the user sees the same
        $term, $inline, $RIO etc. their shell sees), then overlays
        os.environ as a fallback for generic vars. Order matters:
        host vars win over process env if both define the same name.
        """
        env: Dict[str, str] = {}
        # Process env first — gets overlaid by host vars below.
        try:
            env.update(os.environ)
        except Exception:
            pass
        host = self._host_terminal_ref() if self._host_terminal_ref else None
        if host is not None:
            try:
                # Prefer the terminal's own helper if it exposes one.
                tv = getattr(host, '_terminal_variables', None)
                if callable(tv):
                    env.update(tv())
                else:
                    # Fall back to reconstructing the seeded vars from
                    # the attributes we know the terminal sets. Mirrors
                    # _seed_environment_variables() in TerminalWidget.
                    rio_mount = getattr(host, 'rio_mount', None)
                    llmfs_mount = getattr(host, 'llmfs_mount', None)
                    term_id = getattr(host, 'term_id', None)
                    peribus_root = getattr(host, '_peribus_root', None)
                    if llmfs_mount:
                        env['LLMFS'] = llmfs_mount
                    if rio_mount:
                        env['RIO'] = rio_mount
                    if rio_mount and term_id:
                        term_dir = f"{rio_mount}/terms/{term_id}"
                        env['term'] = term_dir
                        env['inline'] = f"{term_dir}/inline"
                    if peribus_root:
                        env['peribus'] = peribus_root
            except Exception:
                pass
        return env

    def _target_is_inline(self, path: str) -> bool:
        """True when `path` is an /inline filesystem endpoint.

        Used by run_code to decide whether to JSON-wrap the payload.
        Matches both:
          - /n/<machine>/terms/<id>/inline   — per-terminal inline scope
          - /n/<machine>/scene/inline        — full-scene inline (rare
            but defined symmetrically with /scene/parse)

        We strip the trailing slash and lowercase before comparing so
        `$inline/` and `$INLINE` both work. We also accept anything
        whose final path component is exactly "inline" — that's the
        canonical filesystem leaf and trying to be smarter (parsing
        /terms/<id>/ specifically) would just reject legitimate
        variants.
        """
        if not path:
            return False
        norm = path.rstrip('/').lower()
        if not norm:
            return False
        return os.path.basename(norm) == "inline"

    # ------------------------------------------------------------------
    # Right-click on header → set/reset/copy Run target
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        """Wire right-click → inline edit, plus Escape to cancel.

        Two distinct flows live here so the keystroke / mouse logic
        stays in one place:

        1. Right-press on the header button or the surrounding strip:
           swap the title slot from button → QLineEdit and focus it.
           We swallow the event so QPushButton doesn't see a press it
           would partially process (its press handler eats the event
           before customContextMenuRequested fires on X11; that's why
           a signal-based approach didn't work earlier).

        2. Escape pressed inside the title QLineEdit: cancel without
           committing — restore the previous value and switch back to
           display mode. Enter is handled by the editingFinished
           signal which fires _commit_title_edit(); we don't intercept
           it here.
        """
        try:
            etype = event.type()
        except Exception:
            return super().eventFilter(obj, event)

        # Right-press → enter inline edit
        if etype == QEvent.MouseButtonPress:
            try:
                if event.button() == Qt.RightButton and obj in (
                    getattr(self, '_header_strip', None),
                    getattr(self, 'header_btn', None),
                ):
                    self._enter_title_edit()
                    return True  # eaten
            except Exception:
                pass

        # Escape inside the line edit → cancel
        if etype == QEvent.KeyPress and obj is getattr(self, 'title_edit', None):
            try:
                if event.key() == Qt.Key_Escape:
                    self._cancel_title_edit()
                    return True
            except Exception:
                pass

        return super().eventFilter(obj, event)

    def _enter_title_edit(self):
        """Swap the title display button for an editable line edit.

        Pre-fills with the user's current custom path if set, else the
        auto-derived $mount/<machine>/scene/parse — so a quick tweak
        ("change /parse to /inline") is just a few keystrokes. We
        select-all so typing replaces immediately, matching the
        convention of address-bar style editors.
        """
        if not hasattr(self, 'title_edit') or self.title_edit is None:
            return
        seed = self.custom_target_raw or self.target_path or ""
        # Remember the pre-edit value so Escape can restore it. (This
        # only matters when the user had a custom path already; for
        # the default fallback case _cancel_title_edit just clears
        # the line and switches back, which has the same effect.)
        self._title_edit_pre = self.custom_target_raw
        self.title_edit.blockSignals(True)
        self.title_edit.setText(seed)
        self.title_edit.blockSignals(False)
        self.title_edit.selectAll()
        self._title_stack.setCurrentWidget(self.title_edit)
        self.title_edit.setFocus(Qt.MouseFocusReason)

    def _commit_title_edit(self):
        """Apply the user-typed path. Empty input resets to default.

        editingFinished fires on Enter AND on focus-loss, so clicking
        outside the field also commits. Re-entrancy guard: this can
        be called twice if the user presses Enter then the focus
        change also fires the signal — the guard makes the second
        call a no-op rather than letting it bounce against an already-
        switched stack.
        """
        if not hasattr(self, 'title_edit') or self.title_edit is None:
            return
        # Re-entrancy / spurious-fire guard: editingFinished fires on
        # Enter AND on focus loss. If we've already swapped back to
        # display mode, ignore subsequent fires.
        if self._title_stack.currentWidget() is not self.title_edit:
            return
        text = (self.title_edit.text() or "").strip()
        if text:
            self.custom_target_raw = text
        else:
            self.custom_target_raw = None  # empty == reset
        self._title_stack.setCurrentWidget(self.header_btn)
        self._update_header_text()

    def _cancel_title_edit(self):
        """Discard the in-progress edit, restore the previous value."""
        if not hasattr(self, 'title_edit') or self.title_edit is None:
            return
        # Restore the value we captured at edit-start, so a Cancel
        # after typing doesn't mutate state.
        self.custom_target_raw = getattr(self, '_title_edit_pre', None)
        self._title_stack.setCurrentWidget(self.header_btn)
        self._update_header_text()

    def _reset_target_path(self):
        """Programmatic reset: clear any custom target and refresh
        the header display. Kept as a separate method so external
        callers (or future menu items) can reset without going through
        the inline-edit machinery."""
        self.custom_target_raw = None
        self._update_header_text()

    def run_code(self):
        """Write current contents to the target's scene/parse file.
        
        Implementation notes (this took several iterations to get right):
        
        1) Synthetic 9pfuse files are picky about how the kernel opens
           them (O_WRONLY|O_TRUNC vs O_WRONLY only; stdin-stream writes
           vs argv-passed writes), so a single open() call may not
           always succeed. We try a small ladder of subprocess strategies.
        
        2) **Crucially, we must not block the Qt main thread** while the
           subprocess runs. The 9P server's write/clunk handlers may
           need to dispatch work back to Qt main (e.g. the executor
           creates Qt graphics objects, or the parse handler invokes
           a slot on Qt main). If we use blocking subprocess.run() from
           Qt main, that bounce-back can deadlock and we time out at
           5 seconds — which is exactly what was happening when users
           clicked Run on a real 9pfuse mount.
        
           So we use QProcess (Qt-native, async, integrated with the
           event loop). Qt main keeps spinning, the 9P → Qt-main bounce
           can complete normally, and the user sees a responsive UI
           even on slow writes.
        
        3) Content is fed via stdin to avoid argv-size limits and
           shell-quoting hazards. `cat > "$1"` is the canonical pattern
           and matches what `echo "..." > path` does at the kernel level
           when stdin is closed cleanly (cat doesn't append anything).
        """
        if not self.machine_name and not self.custom_target_raw:
            self._flash_error("No target machine specified")
            return
        if getattr(self, '_run_in_flight', False):
            # Don't queue duplicate writes — visually the user already
            # sees a status flash from the in-flight one.
            return
        path = self.target_path
        if not path:
            self._flash_error("Target path is empty after expansion")
            return
        content = self.code_edit.toPlainText()

        # Auto-wrap for /inline targets.
        #
        # /n/<machine>/scene/parse and /n/<machine>/terms/<id>/parse
        # accept raw Python and execute it (full-scene vs per-terminal
        # respectively). The receiving filesystem expects bytes and
        # runs them as a parse script — no envelope.
        #
        # /n/<machine>/terms/<id>/inline is the inline-rendering peer
        # of /parse: same Python, but rendered into the terminal's
        # inline scope instead of the full scene. The filesystem on
        # *that* path expects a JSON envelope describing what kind of
        # widget to render — schema lives in InlineMediaWidget. Raw
        # text written to /inline gets misclassified by the upstream
        # wrapper (it falls through to {"type":"html",...}), which is
        # why the same code shows up as an "html · NNNN chars" card
        # instead of running.
        #
        # Conceptually the user wrote Python and wants it rendered as
        # Python — same intent as /parse, just inline. So we wrap on
        # the way out: when the resolved target ends in /inline, send
        # {"type":"python","code":...} instead of the raw bytes.
        # /parse and any other path are left untouched.
        send_content = content
        if self._target_is_inline(path):
            try:
                send_content = json.dumps({
                    "type": "python",
                    "code": content,
                })
            except Exception:
                # If wrapping somehow fails (it shouldn't — content is
                # always a str), fall back to the raw write rather
                # than blocking the user.
                send_content = content
        
        # Strategy ladder. Each entry is (label, argv, feed_stdin?).
        # `cat > path` matches `echo > path` byte-perfectly when stdin
        # is closed cleanly (cat appends nothing). We try cat first,
        # tee as a fallback (different FUSE codepath), and a worker-
        # thread Python open() as the absolute last resort.
        self._run_strategies = [
            ("cat>", ['bash', '-c', 'cat > "$1"', '_', path], True),
            ("tee",  ['tee', path],                            True),
        ]
        self._run_path = path
        self._run_content = send_content
        self._run_errors = []
        self._run_in_flight = True
        self._run_started_at = time.monotonic()
        self.run_btn.setEnabled(False)
        self.streaming_label.setText("● writing…")
        self.streaming_label.setStyleSheet(
            "color: #4488dd; font-size: 10px; font-weight: bold; padding: 0 6px;"
        )
        self.streaming_label.show()
        
        self._run_try_next()
    
    def _run_try_next(self):
        """Pull the next strategy off the queue and start it via QProcess."""
        if not self._run_strategies:
            # All subprocess strategies exhausted; final fallback is a
            # Python open() in a worker thread so Qt main isn't blocked.
            self._run_try_open_in_thread()
            return
        
        label, argv, feed_stdin = self._run_strategies.pop(0)
        
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc._run_label = label
        proc.finished.connect(
            lambda code, status, p=proc: self._run_on_finished(p, code, status)
        )
        proc.errorOccurred.connect(
            lambda err, p=proc: self._run_on_error(p, err)
        )
        # Watchdog: 30 s is plenty for a parse.py to compile + execute,
        # but small enough to surface a real hang. UI stays responsive
        # the whole time because everything's async.
        watchdog = QTimer(self)
        watchdog.setSingleShot(True)
        watchdog.setInterval(30_000)
        watchdog.timeout.connect(lambda p=proc, l=label: self._run_watchdog(p, l))
        proc._run_watchdog = watchdog
        
        try:
            proc.start(argv[0], argv[1:])
            if not proc.waitForStarted(3000):
                self._run_errors.append(f"{label}: failed to start")
                proc.deleteLater()
                self._run_try_next()
                return
        except Exception as e:
            self._run_errors.append(f"{label}: {e}")
            proc.deleteLater()
            self._run_try_next()
            return
        
        if feed_stdin:
            try:
                data = self._run_content.encode('utf-8')
            except Exception:
                data = self._run_content.encode('utf-8', errors='replace')
            proc.write(data)
            proc.closeWriteChannel()
        
        watchdog.start()
    
    def _run_on_finished(self, proc, exit_code: int, exit_status):
        """QProcess finished — check the result, fall through on failure."""
        wd = getattr(proc, '_run_watchdog', None)
        if wd is not None:
            wd.stop()
            wd.deleteLater()
        label = getattr(proc, '_run_label', '?')
        if exit_status == QProcess.NormalExit and exit_code == 0:
            self._run_succeeded()
            proc.deleteLater()
            return
        try:
            output = bytes(proc.readAll()).decode('utf-8', errors='replace').strip()
        except Exception:
            output = ""
        msg = output or f"exit {exit_code}"
        if len(msg) > 120:
            msg = msg[:120] + "…"
        self._run_errors.append(f"{label}: {msg}")
        proc.deleteLater()
        self._run_try_next()
    
    def _run_on_error(self, proc, err):
        """QProcess.errorOccurred — usually a spawn or pipe failure.
        
        Don't advance the queue here — finished() also fires on spawn
        failure and that's where we want the single advancement to
        happen (otherwise we double-advance).
        """
        wd = getattr(proc, '_run_watchdog', None)
        if wd is not None:
            wd.stop()
        label = getattr(proc, '_run_label', '?')
        self._run_errors.append(f"{label}: {proc.errorString() or err}")
    
    def _run_watchdog(self, proc, label: str):
        """30-second watchdog — kill the QProcess so finished() fires
        and the queue advances. UI stays responsive."""
        try:
            proc.kill()
            # Don't waitForFinished here — finished() will fire async.
        except Exception:
            pass
    
    def _run_try_open_in_thread(self):
        """Final fallback: Python open(path,'w') on a worker thread so
        Qt main stays free even if the FUSE close() takes a while."""
        path = self._run_path
        content = self._run_content
        
        class _OpenWriteWorker(QThread):
            done = Signal(bool, str)  # ok, error_message
            def run(self):
                try:
                    with open(path, 'w') as f:
                        f.write(content)
                    self.done.emit(True, "")
                except Exception as e:
                    self.done.emit(False, str(e))
        
        worker = _OpenWriteWorker(self)
        self._run_open_worker = worker  # keepalive
        worker.done.connect(self._run_open_done)
        worker.start()
    
    def _run_open_done(self, ok: bool, err: str):
        worker = getattr(self, '_run_open_worker', None)
        if worker is not None:
            worker.deleteLater()
            self._run_open_worker = None
        if ok:
            self._run_succeeded()
            return
        self._run_errors.append(f"open: {err}")
        self._run_failed()
    
    def _run_succeeded(self):
        elapsed = time.monotonic() - getattr(self, '_run_started_at', time.monotonic())
        self._run_in_flight = False
        self.run_btn.setEnabled(True)
        # Show timing only when the write took noticeable time, so the
        # common case stays uncluttered.
        if elapsed > 0.5:
            self._flash_success(f"→ {self._run_path}  ({elapsed:.1f}s)")
        else:
            self._flash_success(f"→ {self._run_path}")
    
    def _run_failed(self):
        self._run_in_flight = False
        self.run_btn.setEnabled(True)
        self._flash_error("Run failed: " + " | ".join(self._run_errors))
    
    def _flash_success(self, msg: str):
        """Briefly turn the Run button green-bright with a status message."""
        self.streaming_label.setText(f"✓ {msg}")
        self.streaming_label.setStyleSheet(
            "color: #44aa55; font-size: 10px; font-weight: bold; padding: 0 6px;"
        )
        self.streaming_label.show()
        QTimer.singleShot(1800, self._reset_status_label)
    
    def _flash_error(self, msg: str):
        self.streaming_label.setText(f"✗ {msg}")
        self.streaming_label.setStyleSheet(
            "color: #cc4444; font-size: 10px; font-weight: bold; padding: 0 6px;"
        )
        self.streaming_label.show()
        QTimer.singleShot(2500, self._reset_status_label)
    
    def _reset_status_label(self):
        if self.is_streaming:
            self.streaming_label.setText("● streaming")
            self.streaming_label.setStyleSheet(
                "color: #ffaa00; font-size: 10px; font-weight: bold; padding: 0 6px;"
            )
            self.streaming_label.show()
        else:
            self.streaming_label.hide()


class _InlineSceneManagerStub:
    """No-op SceneManager-shaped stand-in used by inline-Python payloads
    when no real SceneManager is reachable from the inline widget.
    
    Real /scene/parse code commonly calls things like:
        scene_manager.register_parsed_item(proxy, {"quick": True, ...})
        scene_manager.width / scene_manager.height
        scene_manager.take_snapshot(...)
    
    For inline rendering we don't need versioning or the 9P-side
    snapshot machinery — the widget appears in the conversation and
    that's the entire lifecycle. So we expose just enough of the
    SceneManager API surface to swallow the calls without raising.
    Anything that returns a value returns a sensible default.
    """
    
    __slots__ = ('_scene', 'width', 'height', '_items', 'versions')
    
    def __init__(self, scene):
        self._scene = scene
        # Match SceneManager's default canvas size so any code that
        # divides by scene_manager.width gets the same number it would
        # in /scene/parse.
        rect = scene.sceneRect()
        self.width = int(rect.width()) or 1920
        self.height = int(rect.height()) or 1080
        # Keep refs to registered items so they don't get GC'd if the
        # only reference was from user-code locals.
        self._items = []
        # `scene_manager.versions.current_version` is a common read; give
        # them a stub object with current_version = 0.
        self.versions = type('_VersionsStub', (), {'current_version': 0})()
    
    def register_parsed_item(self, qt_item, metadata=None):
        """Mimic SceneManager.register_parsed_item: keep a reference and
        hand back a synthetic item id. We don't snapshot or version."""
        self._items.append((qt_item, metadata or {}))
        return len(self._items)
    
    def register_widget(self, name, widget, x=0, y=0):
        """Some convenience APIs use this — embed via the scene if we can."""
        try:
            proxy = self._scene.addWidget(widget)
            proxy.setPos(x, y)
            self._items.append((proxy, {'name': name, 'x': x, 'y': y}))
            return proxy
        except Exception:
            return None
    
    def take_snapshot(self, label="", code="", namespace=None):
        """No-op — no version system inline."""
        self.versions.current_version += 1
        return self.versions.current_version
    
    def attach_qt(self, *args, **kwargs):
        """Already attached — do nothing."""
        return None
    
    def __getattr__(self, name):
        """Catch-all: any other method call becomes a no-op returning None.
        This prevents AttributeError from breaking otherwise-valid code
        that uses parts of the API we haven't explicitly stubbed."""
        return lambda *a, **kw: None


class _FittingGraphicsView(QGraphicsView):
    """
    A QGraphicsView that always shows the entire scene contents, scaled
    to fit the viewport with aspect ratio preserved.

    Why this exists. Inline-rendered Python widgets get a 1920×1080
    QGraphicsScene to match the SceneManager default (so absolute-
    coordinate code from /scene/parse runs unchanged), but the inline
    frame is much smaller — typically ~700×480. A plain QGraphicsView
    shows the scene at 1:1 zoom, so anything past the viewport edge
    gets cropped behind scrollbars. Users see a partial dashboard with
    the right side missing, like the dashboard screenshot did before
    this change.

    Strategy. On every resize event (and once after construction),
    call fitInView with the bounding rect of the scene's actual
    items, not the full sceneRect. Using itemsBoundingRect matters:
    a dashboard that places widgets only in the top-left 1000×600
    region should fit those widgets to the viewport, not get shrunk
    to fit a mostly-empty 1920×1080.

    Caveat: scaling raster content (text in proxied widgets) loses
    crispness. KeepAspectRatio with the natural width usually keeps
    text legible because the inline frame's width is close to the
    content's natural width — the scale factor is small (typically
    0.5–0.8×), well within Qt's smooth-transform sweet spot.
    """

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        # Disable the dragging-to-pan behavior — these views are
        # passive renders, not interactive scene navigators. The user
        # interacts with embedded widgets through proxies, not by
        # panning the scene.
        self.setDragMode(QGraphicsView.NoDrag)
        # No scrollbars — fitInView removes the need.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Fit once on a deferred tick so any items added during widget
        # construction (or right after, by user-code-driven signals)
        # are included in the bounding rect. Without the defer, the
        # initial fit happens on an empty scene and the next resize
        # would be the first actual fit.
        QTimer.singleShot(0, self.fit_to_contents)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Re-fit on every resize so the rendering tracks the host
        # frame's size. Cheap — fitInView is just a transform update.
        self.fit_to_contents()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # First show is when the viewport finally has a real size. An
        # earlier fit (during __init__ when the widget hasn't been laid
        # out yet) computes against a 0×0 viewport and does nothing
        # useful. Re-fit here to catch that case.
        self.fit_to_contents()

    def fit_to_contents(self) -> None:
        """Scale the view so the scene's items fit the viewport."""
        scene = self.scene()
        if scene is None:
            return
        # Prefer the items' bounding rect (so an empty top-left only
        # dashboard doesn't get shrunk to fit the empty 1920×1080
        # scene). Fall back to sceneRect if the scene has no items
        # yet — that happens on the deferred initial fit when user
        # code hasn't populated the scene.
        target = scene.itemsBoundingRect()
        if target.isEmpty():
            target = scene.sceneRect()
        if target.isEmpty():
            return
        # Pad the target by a small margin so embedded widget borders
        # don't touch the viewport edge — looks tighter and more
        # intentional than a perfectly snug fit.
        margin = 8.0
        target = target.adjusted(-margin, -margin, margin, margin)
        self.fitInView(target, Qt.KeepAspectRatio)


class _DraggableHeaderFilter(QObject):
    """
    Event filter installed on InlineMediaWidget's header button.

    Watches mouse events on the header to decide between three outcomes:

      - Plain click (press → release without movement, under the
        long-press threshold): pass through to the button's normal
        clicked signal so toggle() runs.

      - Long press (press held for _LONG_PRESS_MS without moving) OR
        click-and-drag past _DRAG_THRESHOLD_PX: switch to drag mode,
        show a translucent ghost following the cursor, and on release
        drop the card onto the QGraphicsScene under the cursor (if
        any).

      - Drag-then-release-off-target: snap the ghost away with no
        side effects.

    State lives on the InlineMediaWidget itself (not this filter) so
    multiple cards can be dragged independently and the filter is
    just a thin event-routing layer.
    """

    # How long a press has to be held before we arm drag mode without
    # any movement. Picked to be longer than a casual double-click but
    # short enough that a deliberate hold feels responsive.
    _LONG_PRESS_MS = 350

    # How far the cursor has to travel from the initial press to switch
    # straight into drag mode (no need to wait for the long-press
    # timer). 6 px is far enough to filter accidental jitter on touch
    # devices and trackpads but tight enough that a quick pull works.
    _DRAG_THRESHOLD_PX = 6

    def __init__(self, card):
        super().__init__(card)
        # Hold a back-reference to the card we belong to. The filter's
        # parent (super().__init__(card)) keeps the QObject alive for
        # the card's lifetime; this attribute is just for readability.
        self._card = card

    def eventFilter(self, obj, event) -> bool:
        # We only care about the header button.
        if obj is not self._card._header_btn:
            return False

        et = event.type()
        if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            return self._on_press(event)
        if et == QEvent.MouseMove and self._card._drag_state != "idle":
            return self._on_move(event)
        if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            return self._on_release(event)
        return False

    def _on_press(self, event) -> bool:
        c = self._card
        c._drag_state = "armed"
        c._drag_press_pos = event.globalPosition().toPoint()
        c._drag_press_time = time.time()

        # Arm a one-shot long-press timer. If the user holds without
        # moving, this fires and we go straight to drag mode.
        if c._drag_arm_timer is None:
            c._drag_arm_timer = QTimer(c)
            c._drag_arm_timer.setSingleShot(True)
            c._drag_arm_timer.timeout.connect(c._drag_long_press_expired)
        c._drag_arm_timer.start(self._LONG_PRESS_MS)

        # Don't consume — we want the button's pressed visual state to
        # show during the hold. We'll consume the release instead if
        # we end up in drag mode.
        return False

    def _on_move(self, event) -> bool:
        c = self._card
        if c._drag_press_pos is None:
            return False

        gp = event.globalPosition().toPoint()
        dx = gp.x() - c._drag_press_pos.x()
        dy = gp.y() - c._drag_press_pos.y()
        dist_sq = dx * dx + dy * dy

        if c._drag_state == "armed":
            # Start the drag once we've moved past threshold.
            if dist_sq >= self._DRAG_THRESHOLD_PX * self._DRAG_THRESHOLD_PX:
                c._begin_drag(gp)
        elif c._drag_state == "dragging":
            c._update_drag(gp)

        return False

    def _on_release(self, event) -> bool:
        c = self._card
        # Stop the long-press timer regardless — the press is over.
        if c._drag_arm_timer is not None:
            c._drag_arm_timer.stop()

        if c._drag_state == "dragging":
            # End the drag and consume the release so the button doesn't
            # also emit clicked.
            c._end_drag(event.globalPosition().toPoint())
            c._drag_state = "idle"
            c._drag_press_pos = None
            return True   # consumed: no click-through

        # Otherwise (armed but never moved past threshold, or already idle):
        # let the click signal fire normally to drive toggle().
        c._drag_state = "idle"
        c._drag_press_pos = None
        return False


class InlineMediaWidget(QFrame):
    """Lightweight inline widget for media: image, gif, audio, video, pdf,
    3D models, html, info banners, link, and inline-rendered Python code.
    
    Used by the $term/inline filesystem file. Each widget renders one
    payload — a JSON envelope describing what to display:
    
        {"type": "image", "path": "/path/to/img.png", "caption": "..."}
        {"type": "image", "data": "<base64>", "format": "png"}
        {"type": "gif",   "path": "/path/to/anim.gif"}
        {"type": "audio", "path": "/path/to/song.mp3"}
        {"type": "video", "path": "/path/to/clip.mp4"}
        {"type": "pdf",   "path": "/path/to/doc.pdf"}
        {"type": "model3d", "path": "/path/to/mesh.obj"}
        {"type": "python", "code": "_widget = QPushButton('hi')"}
        {"type": "html",  "content": "<p>Hello</p>"}
        {"type": "info",  "text": "...", "level": "info"|"warn"|"error"}
        {"type": "link",  "url": "...", "label": "..."}
    
    Layout:
      - Header bar: arrow + type + summary, click toggles expand/collapse.
      - Body: built lazily for heavy types (audio/video/pdf/model3d/python),
        eagerly for cheap types (image/gif/html/link/info).
    
    Performance:
      - Heavy renderers don't construct until the user expands them.
      - On collapse, audio/video pause to free decoder resources; on
        re-expand they resume from where they left off.
      - All embedded content respects the terminal's available width
        (set_inline_max_width) so nothing extends under the scrollbar
        or forces horizontal scrolling.
    """
    
    # Heavy types we defer construction for until the user expands.
    # Cheap types (image, gif, html, link, info, python) build immediately;
    # they're either static or the explicit purpose of writing to $inline
    # is to render the result — no point hiding it behind a click.
    # Heavy decoder-backed types (audio/video/pdf/3d) and url (Chromium
    # subprocess) stay lazy so a long history of conversations doesn't
    # keep N media pipelines / browser instances warm.
    _LAZY_TYPES = {"audio", "video", "pdf", "model3d", "model", "mesh", "url"}
    
    # Conservative max-height for any single inline widget so a video
    # or PDF doesn't push the rest of the terminal off-screen. Width
    # is determined dynamically via set_inline_max_width.
    _MEDIA_MAX_H = 480
    
    # Default fallback width if the terminal hasn't told us one yet.
    _DEFAULT_MAX_W = 700
    
    def __init__(self, payload: dict, dark_mode: bool = False,
                 max_width: int = 0, parent=None,
                 host_terminal=None):
        super().__init__(parent)
        self._dark_mode = dark_mode
        self._payload = payload
        self._kind = (payload.get("type") or "info").lower()
        # Direct reference to the host terminal, if known at construction
        # time. _find_host_terminal prefers this over the parent-chain
        # walk because top-level widgets aren't parented yet when their
        # __init__ runs (insert_media constructs without parent then
        # parents via addWidget afterward). Without this hint, the
        # parent walk inside _render_post returns None, and every
        # post-card action button (deepen / reply / copy path) silently
        # gets skipped — a bug that's been visible as buttonless cards
        # since the post renderer was first added.
        self._host_terminal_hint = host_terminal
        # References we must keep alive so Qt doesn't GC them while the
        # widget is visible. Things like QMediaPlayer, QMovie, etc.
        # don't have a parent in the standard widget tree.
        self._keepalive: list = []
        # Set by lazy builders: the actual content widget(s) live in
        # self._body_layout, populated either eagerly in __init__ (cheap
        # types) or on first expand (heavy types).
        self._is_built = False
        self._is_expanded = True
        # For heavy media: when collapsed we pause; on re-expand we resume.
        # The keepalive entries are not torn down — keeping them lets us
        # come back without re-decoding from scratch.
        self._max_w = max_width if max_width > 0 else self._DEFAULT_MAX_W
        
        self._apply_frame_theme()
        self.setMaximumWidth(self._max_w)
        
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        
        # Header (always present): a thin strip with a clickable toggle
        # area on the left (stretches) and a small close ✕ button on the
        # right. We use a QWidget container instead of a single QPushButton
        # because we need two distinct click targets — the original
        # header-as-toggle plus a delete control — without one swallowing
        # the other's events.
        header_strip = QWidget()
        header_strip.setStyleSheet("background: transparent;")
        header_layout = QHBoxLayout(header_strip)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        
        self._header_btn = QPushButton()
        self._header_btn.setCursor(Qt.PointingHandCursor)
        self._header_btn.clicked.connect(self.toggle)
        # Stretch this so the close button stays pinned right.
        header_layout.addWidget(self._header_btn, stretch=1)
        
        # Close (delete) button. Black ✕ glyph, transparent background,
        # subtle red hover so it's discoverable but doesn't compete with
        # the toggle for attention. Fixed small size.
        self._close_btn = QPushButton("✕")
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.setFixedSize(24, 24)
        self._close_btn.setToolTip("Remove this widget")
        self._close_btn.clicked.connect(self._close_widget)
        header_layout.addWidget(self._close_btn)
        
        self._apply_header_theme()
        outer.addWidget(header_strip)
        
        # Body container.
        self._body = QFrame()
        self._body.setFrameStyle(QFrame.NoFrame)
        self._body.setStyleSheet("background: transparent; border: none;")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 8, 6)
        self._body_layout.setSpacing(4)
        # Minimum body height varies by payload kind. Banners and link
        # widgets stay compact (50 px); HTML and audio sit in the middle;
        # substantial canvas types (python/pdf/3d/video/url) get a 300 px
        # floor so they always render with enough vertical room to be
        # useful, regardless of how tightly QVBoxLayout's AlignTop tries
        # to compress them when sharing space with text displays.
        # 
        # The body's minimum doesn't constrain growth — content taller
        # than this still expands up to _MEDIA_MAX_H — and doesn't affect
        # the collapsed state because setVisible(False) makes the body
        # claim no layout space at all.
        self._body.setMinimumHeight(self._minimum_body_height_for_kind(self._kind))
        outer.addWidget(self._body)
        
        # Cheap types (image, gif, html, link, info, error) build now.
        # They're effectively zero-cost so collapse-to-skip-construction
        # would just make the first expand jankier.
        if self._kind not in self._LAZY_TYPES:
            self._build_body()
        else:
            # Heavy: show a placeholder summary so the header is informative
            # before first expand. The actual renderer runs on toggle.
            placeholder = QLabel(self._lazy_placeholder_text())
            placeholder.setStyleSheet(
                "color: rgba(120, 130, 150, 200); font-size: 11px; "
                "font-style: italic; padding: 4px 0;"
            )
            self._body_layout.addWidget(placeholder)
            self._lazy_placeholder = placeholder
        
        self._update_header_text()

        # Install the drag-to-scene filter on the header button so a
        # press-and-hold (or click-and-drag past a small threshold) on
        # the header initiates a drag that can drop the whole card
        # onto a TerminalScenePanel's QGraphicsScene. A normal click
        # without movement still calls toggle() — see
        # _DraggableHeaderFilter for the gating logic.
        self._drag_state = "idle"   # "idle" | "armed" | "dragging"
        self._drag_press_pos = None
        self._drag_press_time = 0.0
        self._drag_ghost = None
        self._drag_arm_timer = None
        self._drag_filter = _DraggableHeaderFilter(self)
        self._header_btn.installEventFilter(self._drag_filter)
    
    @classmethod
    def _minimum_body_height_for_kind(cls, kind: str) -> int:
        """Per-payload-type minimum body height, in pixels.
        
        Why per-kind: a one-line `info` banner shouldn't claim 300 px
        of vertical space, and a Python widget building a dashboard
        shouldn't be compressed to 60 px when QVBoxLayout's AlignTop
        squeezes children to share remaining space.
        
        These are floors, not ceilings — content larger than the floor
        grows naturally up to _MEDIA_MAX_H.
        """
        if kind in ("info", "warn", "warning", "error", "ok", "success", "link"):
            return 50   # short banners — content is one line of text
        if kind in ("html",):
            return 80   # rich-text content varies; mid-range floor
        if kind in ("audio",):
            return 100  # control row + title is naturally ~80
        if kind in ("image", "gif"):
            return 180  # at least the rough area of a small thumbnail
        # Substantial canvas types: python (graphics_scene canvas),
        # video (player frame), pdf (scrolling pages), model3d (OpenGL
        # viewport), url (web page). All deserve real vertical room.
        if kind in ("python", "pyside", "pyside6",
                    "pdf", "model3d", "model", "mesh",
                    "video", "url"):
            return 300
        # Unknown kind — use the substantial default; better too tall
        # than too thin, since the user can always collapse.
        return 200
    
    # ------------------------------------------------------------------
    # Width clamping (called by router on construction and on terminal
    # resize via _propagate_inline_width).
    # ------------------------------------------------------------------
    
    def set_inline_max_width(self, max_w: int):
        if max_w <= 0 or max_w == self._max_w:
            return
        self._max_w = max_w
        self.setMaximumWidth(max_w)
        # Inner content may have hard-set sizes (image pixmaps, video
        # min-size) — re-clamp them.
        self._reclamp_children()
    
    def _reclamp_children(self):
        """Walk the body and shrink any child wider than max_w.

        Caching note: we attach the *original* pixmap to the QLabel via
        a dynamic attribute (_orig_pixmap). On subsequent reclamps we
        rescale from the original rather than from the previously-
        rescaled (and lossy) pixmap. Without this, repeated narrow
        resizes would slowly degrade image quality, and a grow back to
        a wider terminal couldn't recover the lost resolution.

        Also: we only rescale when the target width actually changes,
        skipping the SmoothTransformation cost when a resize event
        produced the same effective width.
        """
        if not self._is_built:
            return
        # Account for our 8px L + 8px R inner margins.
        inner = max(40, self._max_w - 16)
        for i in range(self._body_layout.count()):
            w = self._body_layout.itemAt(i).widget()
            if w is None:
                continue
            w.setMaximumWidth(inner)
            # If a QLabel holds a pixmap, rescale it to fit.
            if isinstance(w, QLabel):
                pm = w.pixmap()
                if pm is not None and not pm.isNull():
                    # Stash the original on first encounter so further
                    # reclamps don't compound scaling loss.
                    orig = getattr(w, '_orig_pixmap', None)
                    if orig is None:
                        orig = pm
                        w._orig_pixmap = orig
                    # Decide the target. If the original already fits,
                    # restore it; if not, scale it to the new inner
                    # width. Compare against the last applied width
                    # so identical reclamps short-circuit.
                    last_w = getattr(w, '_last_clamp_w', None)
                    if last_w == inner:
                        continue
                    w._last_clamp_w = inner
                    if orig.width() > inner:
                        scaled = orig.scaled(
                            inner, self._MEDIA_MAX_H,
                            Qt.KeepAspectRatio, Qt.SmoothTransformation,
                        )
                        w.setPixmap(scaled)
                    else:
                        # Restore original if we shrank previously
                        w.setPixmap(orig)
        # Post-kind cards embed nested InlineMediaWidget children for each
        # piece of attached media. They sit inside a wrapper QWidget (for
        # left-indent) so the loop above doesn't see them. Reach them via
        # findChildren and reclamp directly. Recursive=True is safe — even
        # for non-post kinds, the only InlineMediaWidget descendants would
        # be nested cards we deliberately added, and they all need the
        # same width treatment.
        if self._kind == "post":
            for child in self.findChildren(InlineMediaWidget):
                # Subtract a little extra for the indent wrapper so a
                # nested card doesn't clip against our right edge.
                child.set_inline_max_width(max(inner - 12, 200))
    
    # ------------------------------------------------------------------
    # Drag-to-scene — header press-and-hold detaches the card and drops
    # it onto a TerminalScenePanel's QGraphicsScene as a proxy widget.
    # ------------------------------------------------------------------
    #
    # State machine, mirrored on _drag_state:
    #
    #   "idle"     — no press in progress. Filter on header is dormant.
    #   "armed"    — press received; waiting for either a move past
    #                _DRAG_THRESHOLD_PX or for the long-press timer to
    #                fire. A release in this state goes to toggle()
    #                via the button's normal clicked signal.
    #   "dragging" — ghost is following the cursor. A release here
    #                drops the card on whatever scene is under the
    #                cursor, or restores it if the cursor is over
    #                nothing droppable.
    #
    # The ghost is a frameless top-level QLabel showing a translucent
    # screenshot of the card. We follow the cursor in screen
    # coordinates so it works across the terminal's nested scroll
    # area, the scene panel, and any sibling layout — wherever Qt
    # can hit-test, we can drop.

    def _drag_long_press_expired(self) -> None:
        """Long-press timer fired — promote 'armed' to 'dragging'."""
        if self._drag_state != "armed":
            return
        # Cursor hasn't moved past threshold but the user has been
        # holding the button. Begin drag at the current cursor
        # position, not the press position, so the ghost lines up
        # with where the user is looking.
        from PySide6.QtGui import QCursor
        gp = QCursor.pos()
        self._begin_drag(gp)

    def _begin_drag(self, global_pos) -> None:
        """Switch into drag mode and spawn the ghost preview."""
        self._drag_state = "dragging"
        if self._drag_arm_timer is not None:
            self._drag_arm_timer.stop()

        # Reset the header button's visual press state. Without this it
        # stays visually depressed while the drag is in progress
        # because we suppress the eventual release. setDown(False)
        # repaints it as un-pressed.
        try:
            self._header_btn.setDown(False)
        except Exception:
            pass

        # Build the ghost: a frameless tooltip-style window showing a
        # screenshot of the card at ~70% opacity. Doesn't intercept
        # mouse events itself (Qt.WA_TransparentForMouseEvents) so
        # hit-testing under the cursor returns the actual drop target,
        # not the ghost.
        from PySide6.QtWidgets import QLabel as _QLabel
        pixmap = self.grab()
        # Scale ghost down so it doesn't dominate the screen for
        # cards as tall as a python widget. Half-size feels right.
        scaled = pixmap.scaled(
            pixmap.width() // 2, pixmap.height() // 2,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        ghost = _QLabel()
        ghost.setPixmap(scaled)
        ghost.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        ghost.setAttribute(Qt.WA_TranslucentBackground, True)
        ghost.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        ghost.setAttribute(Qt.WA_ShowWithoutActivating, True)
        ghost.setWindowOpacity(0.7)
        ghost.resize(scaled.size())
        self._drag_ghost = ghost
        self._update_drag(global_pos)
        ghost.show()

        # Capture the mouse on the header button so subsequent moves
        # arrive at the filter even if the cursor wanders off the
        # button geometry. Without grab, fast diagonal drags lose
        # mouseMove events the moment the cursor leaves the button's
        # bounding rect.
        try:
            self._header_btn.grabMouse()
        except Exception:
            # grabMouse can fail if another grab is active; we'll
            # fall back to whatever events make it through.
            pass

    def _update_drag(self, global_pos) -> None:
        """Move the ghost to follow the cursor."""
        if self._drag_ghost is None:
            return
        # Offset the ghost slightly so it sits to the lower-right of
        # the cursor — keeps the cursor's hotspot visible for hit
        # detection on the drop target.
        self._drag_ghost.move(global_pos.x() + 8, global_pos.y() + 8)

    def _end_drag(self, global_pos) -> None:
        """Drop the card on the parent scene the terminal is embedded on."""
        # Always destroy the ghost first.
        if self._drag_ghost is not None:
            self._drag_ghost.hide()
            self._drag_ghost.deleteLater()
            self._drag_ghost = None
        try:
            self._header_btn.releaseMouse()
        except Exception:
            pass

        # The drop target is the QGraphicsScene that the host
        # TerminalWidget itself is embedded on as a proxy widget. This
        # is the rio app's main scene — the canvas behind the terminal —
        # NOT the per-terminal /scene panel. When the user drags a card
        # off the terminal they're moving it onto the same canvas where
        # the terminal lives, so it sits next to / behind the terminal
        # rather than into a sub-panel.
        scene, view = self._find_parent_scene_and_view()
        if scene is None or view is None:
            # No host scene found — terminal isn't embedded on one. Snap
            # back: nothing to do, the card is still where it was.
            return

        self._drop_on_scene(scene, view, global_pos)

    def _find_parent_scene_and_view(self):
        """
        Walk up the Qt parent chain from this card looking for an
        ancestor that has a graphicsProxyWidget — i.e. an ancestor
        which is itself embedded on a QGraphicsScene. Returns
        (scene, view) or (None, None) if no such ancestor exists.

        Why this works. The card lives inside the TerminalWidget's
        content layout. The TerminalWidget is added to the rio app's
        main scene via `scene.addWidget(terminal)`, which wraps it in
        a QGraphicsProxyWidget. From any descendant we can walk up
        Qt parents until we find a widget whose graphicsProxyWidget()
        is non-None — that's our embedded ancestor. Its proxy's
        scene() is the rio main scene; the scene's first view is
        what we use to map screen coords.

        Limitation: Qt parent walks stop at the embedded widget
        boundary by default. The proxy itself isn't a widget parent;
        it's a graphics item. graphicsProxyWidget() is the bridge —
        it tells us "this widget is embedded, here's how to get back
        to the scene side."
        """
        node = self
        for _ in range(32):
            if node is None:
                return None, None
            try:
                proxy = node.graphicsProxyWidget()
            except Exception:
                proxy = None
            if proxy is not None:
                scene = proxy.scene()
                if scene is None:
                    return None, None
                views = scene.views()
                if not views:
                    return None, None
                return scene, views[0]
            node = node.parent() if hasattr(node, "parent") else None
        return None, None

    def _drop_on_scene(self, scene, view, global_pos) -> None:
        """
        Reparent this card onto `scene` at the given global cursor
        position. The card keeps all its state — replies, byline tick,
        composer — because Qt re-routes events through the
        QGraphicsProxyWidget the scene wraps it in.

        Detach order matters here. Qt enforces that
        QGraphicsProxyWidget::setWidget refuses any widget that is
        "not a toplevel widget, and is not a child of an embedded
        widget" — meaning we cannot embed a widget that is already
        inside an embedded subtree (the terminal is itself embedded
        on this scene, so the card lives in an embedded subtree).
        Calling scene.addWidget(self) directly hits that error and
        the card vanishes without ever appearing on the scene.

        The fix is to make the card top-level first by clearing its
        Qt parent (setParent(None)), THEN ask the scene to embed it.
        Layout removal alone isn't enough — removeWidget unhooks the
        layout slot but leaves the widget parented to the layout's
        owner.
        """
        # Convert the cursor's screen coords to scene coords. The
        # viewport's mapFromGlobal is the right anchor (not view's
        # own mapFromGlobal) because the view itself can have its
        # own borders / chrome that throw off the conversion.
        viewport = view.viewport()
        local = viewport.mapFromGlobal(global_pos)
        scene_pos = view.mapToScene(local)

        # Detach from layout AND from Qt parent so the card becomes a
        # top-level widget. This is the step that lets scene.addWidget
        # accept it without the "not a toplevel widget" error.
        parent = self.parent()
        if parent is not None and parent.layout() is not None:
            parent.layout().removeWidget(self)
        # Hide before reparenting to avoid a one-frame flash of the
        # card appearing as its own top-level window — without this,
        # setParent(None) on a visible widget pops up a brief frameless
        # window before scene.addWidget swallows it back into the scene.
        was_visible = self.isVisible()
        self.hide()
        self.setParent(None)

        try:
            proxy = scene.addWidget(self)
        except Exception:
            # If embedding still fails, restore the card to its
            # original parent so the user doesn't lose it. Best-effort:
            # we can't perfectly restore layout position, but the card
            # at least re-attaches as a hidden child the layout can
            # accept on the next widget cycle.
            if parent is not None:
                self.setParent(parent)
            if was_visible:
                self.show()
            return

        # Place the card so the cursor lands roughly on the header
        # (where the user was holding). Without this, the card would
        # snap to (0, 0) and the user has to pan to find it.
        offset_x = self._header_btn.width() / 2 if self._header_btn else 0
        offset_y = self._header_btn.height() / 2 if self._header_btn else 0
        proxy.setPos(scene_pos.x() - offset_x, scene_pos.y() - offset_y)
        # Lift dropped cards above any existing scene items so they're
        # immediately interactive without z-order surprises.
        proxy.setZValue(100.0)
        # The proxy's setWidget call should re-show the embedded widget,
        # but be explicit so a hide() that survived the embedding
        # doesn't leave the card invisible inside its proxy.
        if was_visible:
            self.show()

    # ------------------------------------------------------------------
    # Expand / collapse
    # ------------------------------------------------------------------
    
    def toggle(self):
        self._is_expanded = not self._is_expanded
        if self._is_expanded:
            # Lazy build on first expand.
            if not self._is_built:
                # Drop placeholder if present.
                if hasattr(self, '_lazy_placeholder') and self._lazy_placeholder is not None:
                    self._lazy_placeholder.deleteLater()
                    self._lazy_placeholder = None
                self._build_body()
            else:
                # Resume any paused media.
                self._resume_media()
            self._body.setVisible(True)
        else:
            # Pause heavy media but keep the widgets alive so re-expand
            # doesn't have to rebuild the whole pipeline.
            self._pause_media()
            self._body.setVisible(False)
        self._update_header_text()
    
    def _close_widget(self):
        """User clicked the ✕ — remove this inline widget from the
        terminal entirely. We pause/stop any media first so QMediaPlayer
        decoder threads, QWebEngine subprocesses, etc. don't linger
        until garbage collection.
        
        Removal is via deleteLater() so it happens on the next event-
        loop tick, after the click signal has fully drained — calling
        delete() inline from a clicked-handler can crash Qt.
        """
        # Tear down media pipelines first so external resources release
        # promptly. _pause_media handles QMediaPlayer/QMovie/QThread items.
        try:
            self._pause_media()
        except Exception:
            pass
        # Hard-stop anything that has a `stop()` so video/web subprocesses
        # exit immediately rather than continuing to render in the
        # background while waiting for delete.
        for item in self._keepalive:
            try:
                if hasattr(item, 'stop') and callable(item.stop):
                    item.stop()
            except Exception:
                pass
        # Detach from parent layout so the surrounding text displays
        # reflow immediately; deleteLater alone leaves a layout-claimed
        # gap until the next event-loop pass.
        parent = self.parent()
        if parent is not None and parent.layout() is not None:
            parent.layout().removeWidget(self)
        self.setParent(None)
        self.deleteLater()
    
    def _pause_media(self):
        """Stop autoplay-style players when collapsed."""
        for item in self._keepalive:
            try:
                if hasattr(item, 'pause') and callable(item.pause):
                    item.pause()
                elif hasattr(item, 'stop') and callable(item.stop):
                    # QMovie has stop() but no pause() — use setPaused if
                    # available, else stop().
                    if hasattr(item, 'setPaused') and callable(item.setPaused):
                        item.setPaused(True)
                    else:
                        item.stop()
            except Exception:
                pass
    
    def _resume_media(self):
        """Resume what we paused on collapse."""
        for item in self._keepalive:
            try:
                if hasattr(item, 'setPaused') and callable(item.setPaused):
                    item.setPaused(False)
                elif hasattr(item, 'play') and callable(item.play):
                    item.play()
                elif hasattr(item, 'start') and callable(item.start):
                    # QMovie
                    item.start()
            except Exception:
                pass
    
    def _build_body(self):
        """Construct the actual renderer for the payload's type."""
        kind = self._kind
        try:
            if kind == "image":
                self._render_image(self._body_layout, self._payload)
            elif kind == "gif":
                self._render_gif(self._body_layout, self._payload)
            elif kind == "audio":
                self._render_audio(self._body_layout, self._payload)
            elif kind == "video":
                self._render_video(self._body_layout, self._payload)
            elif kind == "pdf":
                self._render_pdf(self._body_layout, self._payload)
            elif kind in ("model3d", "model", "mesh"):
                self._render_model3d(self._body_layout, self._payload)
            elif kind in ("python", "pyside", "pyside6"):
                self._render_python(self._body_layout, self._payload)
            elif kind == "url":
                self._render_url(self._body_layout, self._payload)
            elif kind == "html":
                self._render_html(self._body_layout, self._payload)
            elif kind == "link":
                self._render_link(self._body_layout, self._payload)
            elif kind == "post":
                # Peribus feed post — comes in via $term/inline either from
                # the terminal-side tailer (see PeribusFeedTailer) or from a
                # user/agent that writes the JSON envelope directly.
                self._render_post(self._body_layout, self._payload)
            else:  # info / warn / error / fallback
                self._render_info(self._body_layout, self._payload, kind)
        except Exception as e:
            self._render_info(
                self._body_layout,
                {"text": f"inline render error ({kind}): {e}"},
                "error",
            )
        self._is_built = True
        self._reclamp_children()
    
    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    
    def _apply_header_theme(self):
        if self._dark_mode:
            txt_color = "rgba(220, 225, 235, 255)"
            hover_bg = "rgba(60, 65, 80, 200)"
            # In dark mode the ✕ stays light so it remains visible on
            # the dark frame background.
            close_color = "rgba(220, 225, 235, 200)"
        else:
            txt_color = "rgba(40, 50, 70, 255)"
            hover_bg = "rgba(220, 225, 235, 200)"
            # In light mode the ✕ is the requested black (slightly soft
            # alpha so it doesn't punch through the translucent frame).
            close_color = "rgba(0, 0, 0, 200)"
        self._header_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 3px;
                padding: 6px 8px;
                color: {txt_color};
                font-weight: bold;
                text-align: left;
                font-size: 11px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
        """)
        # Close button: subtle by default, red-tinted on hover so it's
        # clearly destructive, and never blends into the toggle area.
        if hasattr(self, '_close_btn') and self._close_btn is not None:
            self._close_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                    border-radius: 3px;
                    color: {close_color};
                    font-size: 13px;
                    font-weight: bold;
                    padding: 0px;
                }}
                QPushButton:hover {{
                    background-color: rgba(220, 60, 60, 180);
                    color: rgba(255, 255, 255, 255);
                }}
                QPushButton:pressed {{
                    background-color: rgba(180, 40, 40, 220);
                }}
            """)
    
    def _update_header_text(self):
        arrow = "▼" if self._is_expanded else "▶"
        self._header_btn.setText(f"{arrow}  {self._header_summary()}")
    
    def _header_summary(self) -> str:
        """Short label describing the payload — shown in the header."""
        kind = self._kind
        p = self._payload
        path = p.get('path')
        if path:
            base = os.path.basename(path)
            return f"{kind} · {base}"
        if kind in ("info", "warn", "warning", "error", "ok", "success"):
            t = (p.get('text') or p.get('message') or "")[:80]
            return f"{kind}: {t}" if t else kind
        if kind == "link":
            label = p.get('label') or p.get('url') or ""
            return f"link · {label[:80]}"
        if kind == "url":
            url = p.get('url') or ""
            return f"url · {url[:80]}"
        if kind == "html":
            n = len(p.get('content') or "")
            return f"html · {n} chars"
        if kind in ("python", "pyside", "pyside6"):
            n = len((p.get('code') or '').splitlines())
            return f"python · {n} lines"
        if kind == "image" and p.get('data'):
            return f"image · {p.get('format', '?')} (inline data)"
        if kind == "post":
            author = (p.get('author') or '?')
            # Short suffix of the nodeid for compactness.
            if len(author) > 12:
                author = author[:6] + '…' + author[-4:]
            title = (p.get('title') or '').strip()
            body = (p.get('body') or '').strip().replace('\n', ' ')
            preview = title or body
            if len(preview) > 80:
                preview = preview[:80] + '…'
            # Inbox / DM card: prefix with "DM" instead of "post" so the
            # collapsed header makes the message kind obvious.
            label = "DM" if p.get("_card_kind") == "inbox" else "post"
            return f"{label} · {author} · {preview}" if preview else f"{label} · {author}"
        return kind
    
    def _lazy_placeholder_text(self) -> str:
        """Shown in the body before first expand for heavy types."""
        return "(click to load)"
    
    def _apply_frame_theme(self):
        # Inbox / DM cards get a pink tint so the user can pick mail
        # out of the feed at a glance. The kind flag arrives on the
        # payload as `_card_kind: "inbox"` (set by the inbox tailer in
        # _on_peribus_inbox_line). Falls back to the default neutral
        # frame for everything else.
        is_inbox = (
            isinstance(self._payload, dict)
            and self._payload.get("_card_kind") == "inbox"
        )
        if is_inbox:
            if self._dark_mode:
                # Muted plum on dark mode — readable, distinct from the
                # default neutral frame, but not aggressive.
                self.setStyleSheet("""
                    InlineMediaWidget {
                        background-color: rgba(70, 38, 58, 160);
                        border: 1px solid rgba(180, 100, 140, 130);
                        border-radius: 5px;
                        margin: 4px 0px;
                    }
                """)
            else:
                # Soft rose on light mode.
                self.setStyleSheet("""
                    InlineMediaWidget {
                        background-color: rgba(252, 232, 240, 220);
                        border: 1px solid rgba(220, 150, 180, 180);
                        border-radius: 5px;
                        margin: 4px 0px;
                    }
                """)
            return
        if self._dark_mode:
            self.setStyleSheet("""
                InlineMediaWidget {
                    background-color: rgba(40, 42, 52, 140);
                    border: 1px solid rgba(90, 100, 120, 100);
                    border-radius: 5px;
                    margin: 4px 0px;
                }
            """)
        else:
            self.setStyleSheet("""
                InlineMediaWidget {
                    background-color: rgba(248, 250, 253, 180);
                    border: 1px solid rgba(190, 200, 215, 140);
                    border-radius: 5px;
                    margin: 4px 0px;
                }
            """)
    
    # ------------------------------------------------------------------
    # Static images (PNG/JPEG/etc.)
    # ------------------------------------------------------------------

    def _materialize_inline_path(self, payload, default_suffix: str) -> Optional[str]:
        """
        Resolve a payload to a local file path.

        If `payload['path']` exists on this filesystem, return it directly.
        Otherwise, if `payload['data']` is a base64 blob, decode it into a
        temp file and pin the filename in self._keepalive so it survives
        until this widget closes (the closeEvent unlinks tempfiles tagged
        as such). The suffix is used so that QMediaPlayer / QPdfDocument /
        trimesh can route by extension; callers should pass something
        sensible (".mp4", ".pdf", ".obj", …).

        Returns None if neither a usable path nor decodable data was
        provided. Callers should render an error in that case.

        Why this lives on the widget rather than as free-standing code:
        the per-widget tempfile teardown (via _keepalive) is the only
        mechanism we have for cleaning up inline-data tempfiles when the
        post card is closed, and reusing that machinery means we don't
        have to invent a parallel cleanup path.
        """
        path = payload.get("path")
        if path and os.path.isfile(path):
            return path
        data_b64 = payload.get("data")
        if not data_b64:
            return None
        # Prefer the format/extension hint from the payload over the
        # caller's default; that lets callers pass a generic suffix and
        # still get the right extension when the producer specified one.
        fmt = payload.get("format") or ""
        if fmt and not fmt.startswith("."):
            suffix = "." + fmt.lstrip(".")
        else:
            suffix = default_suffix or ""
        import base64, tempfile
        try:
            raw = base64.b64decode(data_b64)
        except Exception:
            return None
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=suffix, prefix="inline_",
            )
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
        except OSError:
            return None
        # Tag for teardown: closeEvent unlinks anything tagged 'tempfile'.
        self._keepalive.append(("tempfile", tmp_path))
        return tmp_path

    def _render_image(self, layout, payload):
        """Render an image payload.

        Performance: when the source is a path on disk, we use
        QImageReader with a target size. The reader can short-circuit
        large images — JPEG and PNG decoders both support scaled reads
        — so a 4K screenshot doesn't get fully decoded into RAM just
        to be downscaled. For base64-embedded data we fall through to
        the load-then-scale path because QImageReader's reading from
        QByteArray doesn't always preserve the scaled-read fast path
        depending on backend.

        We use SmoothTransformation on the scale fallback (visual
        quality matters for static images) but the scaled-read path
        does its own quality-appropriate decoding.
        """
        from PySide6.QtGui import QImageReader
        from PySide6.QtCore import QSize

        pix = QPixmap()
        loaded = False
        path = payload.get("path")
        avail_w = max(40, self._max_w - 16)

        if path and os.path.isfile(path):
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            src_size = reader.size()
            if src_size.isValid() and (
                src_size.width() > avail_w or src_size.height() > self._MEDIA_MAX_H
            ):
                # Scaled read: ask the decoder to deliver a smaller image.
                # Preserve aspect ratio manually since QImageReader's
                # setScaledSize doesn't have a KeepAspectRatio mode.
                ratio = min(
                    avail_w / max(src_size.width(), 1),
                    self._MEDIA_MAX_H / max(src_size.height(), 1),
                )
                target = QSize(
                    max(1, int(src_size.width() * ratio)),
                    max(1, int(src_size.height() * ratio)),
                )
                reader.setScaledSize(target)
            img = reader.read()
            if not img.isNull():
                pix = QPixmap.fromImage(img)
                loaded = True

        if not loaded:
            data_b64 = payload.get("data")
            if data_b64:
                import base64
                try:
                    raw = base64.b64decode(data_b64)
                    pix.loadFromData(raw)
                    loaded = True
                except Exception:
                    pass

        if loaded and not pix.isNull():
            # If we took the base64 path, we still need to downscale.
            # The disk-path scaled-read above already produced a
            # correctly-sized pixmap and skips this.
            if pix.width() > avail_w or pix.height() > self._MEDIA_MAX_H:
                pix = pix.scaled(
                    avail_w, self._MEDIA_MAX_H,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            label = QLabel()
            label.setPixmap(pix)
            label.setAlignment(Qt.AlignCenter)
            label.setMaximumWidth(avail_w)
            layout.addWidget(label)
        else:
            err = QLabel(f"⚠ image unavailable: {payload.get('path') or '<inline data>'}")
            err.setStyleSheet("color: #cc6644; font-size: 11px;")
            layout.addWidget(err)

        caption = payload.get("caption")
        if caption:
            cap = QLabel(str(caption))
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("color: rgba(120, 130, 150, 220); font-size: 10px; font-style: italic;")
            cap.setWordWrap(True)
            layout.addWidget(cap)
    
    # ------------------------------------------------------------------
    # GIF — auto-loop via QMovie
    # ------------------------------------------------------------------
    
    def _render_gif(self, layout, payload):
        from PySide6.QtGui import QMovie
        path = payload.get("path")
        movie = None
        if path and os.path.isfile(path):
            movie = QMovie(path)
        else:
            data_b64 = payload.get("data")
            if data_b64:
                import base64, tempfile
                # QMovie wants a file path or QIODevice; the simplest
                # cross-version path is a temp file we keep on disk for
                # the lifetime of the widget.
                try:
                    raw = base64.b64decode(data_b64)
                    fd, tmp_path = tempfile.mkstemp(suffix='.gif', prefix='inline_')
                    with os.fdopen(fd, 'wb') as f:
                        f.write(raw)
                    movie = QMovie(tmp_path)
                    self._keepalive.append(('tempfile', tmp_path))
                except Exception:
                    pass
        
        if movie is None or not movie.isValid():
            err = QLabel("⚠ gif unavailable")
            err.setStyleSheet("color: #cc6644; font-size: 11px;")
            layout.addWidget(err)
            return
        
        # Cache mode: CacheNone decodes each frame on demand. CacheAll
        # was the original setting — it caches every decoded frame for
        # smoother playback, but for inline GIFs in a long-running
        # terminal that's unbounded RAM growth. Multiple long GIFs
        # would accumulate tens-to-hundreds of MB of frame cache that
        # never gets released (the movie is also pinned in
        # self._keepalive forever). Per-frame decode is cheap on
        # modern CPUs; the memory savings are large.
        movie.setCacheMode(QMovie.CacheNone)
        try:
            # QMovie has setLoopCount via setSpeed; loopCount itself is
            # read-only. The default for animated GIFs is to honour the
            # GIF's stored loop count. To force infinite loop we
            # reconnect frameChanged to restart at frame 0 if needed.
            movie.finished.connect(lambda m=movie: m.start())
        except Exception:
            pass
        
        # Cap displayed size to fit the terminal's available width.
        movie.jumpToFrame(0)
        first_size = movie.currentImage().size()
        avail_w = max(40, self._max_w - 16)
        if first_size.isValid() and (
            first_size.width() > avail_w
            or first_size.height() > self._MEDIA_MAX_H
        ):
            ratio = min(
                avail_w / max(first_size.width(), 1),
                self._MEDIA_MAX_H / max(first_size.height(), 1),
            )
            from PySide6.QtCore import QSize
            movie.setScaledSize(QSize(
                int(first_size.width() * ratio),
                int(first_size.height() * ratio),
            ))
        
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMovie(movie)
        label.setMaximumWidth(avail_w)
        movie.start()
        layout.addWidget(label)
        self._keepalive.append(movie)
    
    # ------------------------------------------------------------------
    # Audio — minimal player (play/pause/stop + volume)
    # ------------------------------------------------------------------
    
    def _render_audio(self, layout, payload):
        path = self._materialize_inline_path(payload, ".mp3")
        if not path or not os.path.isfile(path):
            self._render_info(layout, {"text": f"audio not found: {payload.get('path') or '<inline data>'}"}, "error")
            return
        try:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PySide6.QtCore import QUrl
        except ImportError as e:
            self._render_info(layout, {"text": f"audio backend missing: {e}"}, "error")
            return
        from PySide6.QtWidgets import QWidget, QHBoxLayout, QSlider
        
        player = QMediaPlayer()
        audio_out = QAudioOutput()
        player.setAudioOutput(audio_out)
        player.setSource(QUrl.fromLocalFile(path))
        audio_out.setVolume(0.8)
        
        # Title
        title = QLabel(os.path.basename(path))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 12px; font-weight: bold; padding: 2px;")
        layout.addWidget(title)
        
        # Controls
        ctl = QWidget()
        ctl.setStyleSheet("background: transparent;")
        ctl_layout = QHBoxLayout(ctl)
        ctl_layout.setContentsMargins(0, 0, 0, 0)
        ctl_layout.setSpacing(6)
        
        def _make_btn(label, fn):
            b = QPushButton(label)
            b.setFixedHeight(24)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton {
                    background-color: rgba(180, 180, 180, 60);
                    border: 1px solid rgba(160, 160, 160, 80);
                    border-radius: 3px;
                    padding: 2px 10px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: rgba(180, 180, 180, 110);
                }
            """)
            b.clicked.connect(fn)
            return b
        
        def _toggle():
            from PySide6.QtMultimedia import QMediaPlayer as _MP
            if player.playbackState() == _MP.PlayingState:
                player.pause()
            else:
                player.play()
        
        ctl_layout.addWidget(_make_btn("▶ / ⏸", _toggle))
        ctl_layout.addWidget(_make_btn("⏹", player.stop))
        ctl_layout.addWidget(_make_btn("⏮", lambda: (player.setPosition(0), player.play())))
        
        vol = QSlider(Qt.Horizontal)
        vol.setRange(0, 100)
        vol.setValue(80)
        vol.setFixedWidth(120)
        vol.valueChanged.connect(lambda v: audio_out.setVolume(v / 100.0))
        ctl_layout.addWidget(vol)
        ctl_layout.addStretch()
        
        layout.addWidget(ctl)
        # Keep player + audio alive for the widget's lifetime.
        self._keepalive.extend([player, audio_out])
    
    # ------------------------------------------------------------------
    # Video — proxy-safe via QVideoSink (NOT QVideoWidget, which uses a
    # native window and shows white inside addWidget proxies).
    # Adapted from quick_generators.generate_quick_video_player.
    # ------------------------------------------------------------------
    
    def _render_video(self, layout, payload):
        path = self._materialize_inline_path(payload, ".mp4")
        if not path or not os.path.isfile(path):
            self._render_info(layout, {"text": f"video not found: {payload.get('path') or '<inline data>'}"}, "error")
            return
        try:
            from PySide6.QtMultimedia import (
                QMediaPlayer, QAudioOutput, QVideoSink, QVideoFrame,
            )
            from PySide6.QtCore import QUrl, QSize
            from PySide6.QtGui import QImage, QPainter
        except ImportError as e:
            self._render_info(layout, {"text": f"video backend missing: {e}"}, "error")
            return
        from PySide6.QtCore import Slot as _Slot
        
        class _InlineVideoView(QWidget):
            """Compact video viewer safe inside QGraphicsScene proxies.

            Performance notes (each one materially matters at 30-60 fps):

              - When the widget is invisible (collapsed, scrolled off,
                in a hidden tab) we DROP frames at the sink boundary.
                The QMediaPlayer keeps running so audio/sync stay
                correct, but we avoid the toImage/scale/blit pipeline
                for frames the user can't see.
              - We do NOT call convertToFormat() on every frame. The
                old version converted to RGB32 unconditionally — a full
                buffer copy per frame — even when the source was already
                in a paint-friendly format. We only convert if needed.
              - paintEvent scales lazily: if widget size hasn't changed
                AND the source frame hasn't changed, we redraw the
                cached pixmap. Was already mostly true, but we tighten
                it so frame=None paints don't re-scale.
              - We use FastTransformation (bilinear) rather than
                Smooth (Lanczos-ish). For real-time video at typical
                inline sizes the difference is imperceptible; the cost
                difference is ~5x.
              - update() rate is capped: a 60 fps source with a 60 Hz
                display doesn't benefit from more than 60 paints/sec,
                and Qt will gladly fire faster than that under load.
            """
            def __init__(self, source_path, max_w):
                super().__init__()
                self._raw_image = None
                self._cached_pix = None
                self._cached_size = None
                self._cached_for_image_id = None  # id() of the last image we scaled
                # Keep minimum modest so very narrow terminals still fit
                # the video without forcing horizontal scroll.
                self.setMinimumSize(200, 140)
                self.setMaximumWidth(max_w)
                self.setMaximumHeight(InlineMediaWidget._MEDIA_MAX_H)
                self.setStyleSheet("background-color: black; border-radius: 3px;")
                self.setAttribute(Qt.WA_OpaquePaintEvent, True)

                # Paint-rate gate. Frames arriving faster than ~60 Hz
                # are accepted into the buffer but don't trigger extra
                # repaints; the next paint picks up the latest buffered
                # frame.
                self._last_update_t = 0.0
                self._min_update_interval = 1.0 / 60.0

                self.sink = QVideoSink()
                self.sink.videoFrameChanged.connect(self._process_frame)
                self.media = QMediaPlayer()
                self.audio = QAudioOutput()
                self.audio.setVolume(0.8)
                self.media.setAudioOutput(self.audio)
                self.media.setVideoOutput(self.sink)
                self.media.setSource(QUrl.fromLocalFile(source_path))
                self.media.setLoops(-1)
                self.media.play()

            @_Slot(QVideoFrame)
            def _process_frame(self, frame):
                if not frame.isValid():
                    return
                # Visibility gate: if the widget can't be seen, don't
                # do the toImage/scale work. isVisible() walks up to
                # check ancestors, which is what we want — a collapsed
                # InlineMediaWidget hides us via the body layout, so
                # this returns False and we skip the pipeline. Audio
                # and demuxing in QMediaPlayer continue regardless.
                if not self.isVisible():
                    return

                img = frame.toImage()
                if img.isNull():
                    return

                # Avoid the full-buffer convertToFormat copy when the
                # source is already in a Qt-friendly format. ARGB32
                # and RGB32 are both fine for QPainter; everything else
                # we coerce.
                fmt = img.format()
                if fmt != QImage.Format_RGB32 and fmt != QImage.Format_ARGB32:
                    img = img.convertToFormat(QImage.Format_RGB32)

                self._raw_image = img
                # Mark the cached pixmap stale by clearing the image id
                # rather than the pixmap — the size check in paintEvent
                # will pick up the change without us nulling the cache
                # mid-paint.
                self._cached_for_image_id = None

                # Rate-limit repaints. Frames keep arriving into
                # self._raw_image; paintEvent uses whichever was most
                # recent when it fires.
                now = time.monotonic()
                if now - self._last_update_t >= self._min_update_interval:
                    self._last_update_t = now
                    self.update()

            def paintEvent(self, ev):
                p = QPainter(self)
                p.fillRect(self.rect(), QColor(0, 0, 0))
                img = self._raw_image
                if img is None or img.isNull():
                    return

                cur_size = self.size()
                img_id = id(img)
                if (self._cached_pix is None
                        or self._cached_size != cur_size
                        or self._cached_for_image_id != img_id):
                    scaled = img.scaled(
                        cur_size, Qt.KeepAspectRatio, Qt.FastTransformation,
                    )
                    self._cached_pix = QPixmap.fromImage(scaled)
                    self._cached_size = cur_size
                    self._cached_for_image_id = img_id

                x = (cur_size.width() - self._cached_pix.width()) // 2
                y = (cur_size.height() - self._cached_pix.height()) // 2
                p.drawPixmap(x, y, self._cached_pix)

            def showEvent(self, ev):
                # Coming back into view after being hidden: request a
                # fresh frame paint so we don't show whatever was last
                # in the buffer indefinitely. The next videoFrameChanged
                # will arrive within ~16 ms anyway, but this avoids the
                # gap.
                super().showEvent(ev)
                if self._raw_image is not None:
                    self.update()

            def mousePressEvent(self, ev):
                if ev.button() == Qt.LeftButton:
                    from PySide6.QtMultimedia import QMediaPlayer as _MP
                    if self.media.playbackState() == _MP.PlayingState:
                        self.media.pause()
                    else:
                        self.media.play()
            
            def resizeEvent(self, ev):
                super().resizeEvent(ev)
                self._cached_pix = None
                self.update()
        
        view = _InlineVideoView(path, max(40, self._max_w - 16))
        layout.addWidget(view)
        self._keepalive.append(view)
        # Also keep the underlying media player handles for pause/resume
        # on collapse/expand.
        self._keepalive.append(view.media)
    
    # ------------------------------------------------------------------
    # PDF — PyMuPDF rasterization (lazy import).
    # Adapted from quick_generators.generate_quick_pdf_viewer.
    # ------------------------------------------------------------------
    
    def _render_pdf(self, layout, payload):
        path = self._materialize_inline_path(payload, ".pdf")
        if not path or not os.path.isfile(path):
            self._render_info(layout, {"text": f"pdf not found: {payload.get('path') or '<inline data>'}"}, "error")
            return
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            self._render_info(layout, {"text": f"pdf backend (PyMuPDF) missing: {e}"}, "error")
            return
        from PySide6.QtWidgets import QScrollArea, QWidget
        from PySide6.QtGui import QImage
        
        pages_w = QWidget()
        pages_w.setStyleSheet("background: transparent;")
        pages_layout = QVBoxLayout(pages_w)
        pages_layout.setSpacing(8)
        pages_layout.setContentsMargins(8, 8, 8, 8)
        
        try:
            doc = fitz.open(path)
            # Cap rendered pages so large PDFs don't lock up the UI.
            max_pages = 25
            for pn in range(min(len(doc), max_pages)):
                pg = doc[pn]
                pix = pg.get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
                img = QImage(
                    pix.samples, pix.width, pix.height, pix.stride,
                    QImage.Format_RGB888,
                )
                lbl = QLabel()
                lbl.setPixmap(QPixmap.fromImage(img))
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("background: transparent; border: none;")
                pages_layout.addWidget(lbl)
            if len(doc) > max_pages:
                more = QLabel(f"… {len(doc) - max_pages} more page(s) not rendered")
                more.setAlignment(Qt.AlignCenter)
                more.setStyleSheet(
                    "color: rgba(120,130,150,220); font-size: 10px; "
                    "font-style: italic; padding: 4px;"
                )
                pages_layout.addWidget(more)
            doc.close()
        except Exception as e:
            err = QLabel(f"⚠ PDF render error: {e}")
            err.setStyleSheet("color: #cc6644; font-size: 11px;")
            pages_layout.addWidget(err)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(self._MEDIA_MAX_H)
        scroll.setMaximumWidth(max(40, self._max_w - 16))
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        scroll.setWidget(pages_w)
        layout.addWidget(scroll)
    
    # ------------------------------------------------------------------
    # 3D mesh — OpenGL wireframe. Adapted from quick_generators.
    # ------------------------------------------------------------------
    
    def _render_model3d(self, layout, payload):
        path = self._materialize_inline_path(payload, ".obj")
        if not path or not os.path.isfile(path):
            self._render_info(layout, {"text": f"3D model not found: {payload.get('path') or '<inline data>'}"}, "error")
            return
        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget
            from OpenGL.GL import (
                glEnable, glBlendFunc, glClearColor, glClear, glMatrixMode,
                glLoadIdentity, glRotatef, glColor3f, glLineWidth,
                glViewport, glFlush,
                glEnableClientState, glDisableClientState, glVertexPointer,
                glDrawArrays,
                GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_PROJECTION,
                GL_MODELVIEW, GL_LINES, GL_FLOAT, GL_VERTEX_ARRAY,
            )
            from OpenGL.GLU import gluPerspective, gluLookAt
            import trimesh
            import numpy as np
        except ImportError as e:
            self._render_info(layout, {"text": f"3D backend missing: {e}"}, "error")
            return

        class _InlineMesh3DWidget(QOpenGLWidget):
            """Wireframe mesh viewer.

            Performance: the original implementation used
            glBegin(GL_LINES) ... glVertex3fv() ... glEnd() inside a
            Python loop. Each glVertex3fv crosses the Python/C boundary
            via PyOpenGL, marshals a 3-float array, and queues a GL
            command. For a 2000-face mesh (~6000 edges) that's ~12000
            crossings per paint, which dominates even at 30 FPS — the
            GPU is idle, waiting for Python.

            The fixed version uses a flat NumPy vertex array (one entry
            per *line endpoint*, so 2 * num_edges rows) and a single
            glDrawArrays call. All vertices upload in one C-level
            buffer transfer; one GL command issues all the line draws.
            On a typical mid-range GPU this turns multi-millisecond
            per-frame loops into sub-millisecond draws.

            We deliberately stay on the fixed-function pipeline (client
            vertex array) rather than shifting to VBOs + shaders: the
            wireframe is static once loaded, the dataset is small, and
            keeping the fixed-function gluPerspective/gluLookAt path
            avoids dragging in a shader pipeline just for a debug
            preview. The win from one glDrawArrays vs thousands of
            glVertex3fv is what matters.
            """
            def __init__(self, mp):
                super().__init__()
                self.setUpdateBehavior(QOpenGLWidget.NoPartialUpdate)
                self._line_verts = None  # flat (2*E, 3) float32 array
                self._line_count = 0     # number of *vertices* (= 2 * edges)
                self._rx = 20.0
                self._ry = 0.0
                self._zoom = 3.0
                self._last = None
                self.setMinimumSize(240, 200)
                self.setMaximumHeight(InlineMediaWidget._MEDIA_MAX_H)
                self._load(mp)

            def _load(self, mp):
                try:
                    m = trimesh.load(mp, force="mesh")
                    if hasattr(m, 'faces') and len(m.faces) > 2000:
                        try:
                            m = m.simplify_quadric_decimation(2000 / len(m.faces))
                        except Exception:
                            pass
                    edges = m.edges_unique
                    v = m.vertices.astype(np.float32)
                    v -= v.mean(axis=0)
                    d = float(np.max(np.linalg.norm(v, axis=1)))
                    if d > 0:
                        v /= d
                    # Build flat (2*E, 3) array: each pair of consecutive
                    # rows is one line. Vectorized; no Python loop.
                    starts = v[edges[:, 0]]
                    ends = v[edges[:, 1]]
                    flat = np.empty((edges.shape[0] * 2, 3), dtype=np.float32)
                    flat[0::2] = starts
                    flat[1::2] = ends
                    # Force contiguous so glVertexPointer can read it
                    # without an extra copy.
                    self._line_verts = np.ascontiguousarray(flat)
                    self._line_count = self._line_verts.shape[0]
                except Exception as e:
                    print(f"[inline 3D] load error: {e}")

            def initializeGL(self):
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glClearColor(0.96, 0.96, 0.96, 1.0)

            def paintGL(self):
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                if self._line_verts is None or self._line_count == 0:
                    return
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45.0, self.width() / max(self.height(), 1), 0.1, 100.0)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                gluLookAt(0, 0, self._zoom, 0, 0, 0, 0, 1, 0)
                glRotatef(self._rx, 1, 0, 0)
                glRotatef(self._ry, 0, 1, 0)
                glColor3f(0.05, 0.05, 0.05)
                glLineWidth(1.0)

                # One vertex-array upload + one draw call replaces the
                # entire glBegin/glVertex.../glEnd loop.
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, self._line_verts)
                glDrawArrays(GL_LINES, 0, self._line_count)
                glDisableClientState(GL_VERTEX_ARRAY)
                glFlush()

            def resizeGL(self, w, h):
                glViewport(0, 0, w, h)
                self.update()

            def mousePressEvent(self, ev):
                if ev.button() == Qt.LeftButton:
                    self._last = ev.pos()
                    self.setCursor(Qt.ClosedHandCursor)

            def mouseMoveEvent(self, ev):
                if self._last is not None:
                    d = ev.pos() - self._last
                    self._last = ev.pos()
                    self._ry += d.x() * 0.5
                    self._rx += d.y() * 0.5
                    self.update()

            def mouseReleaseEvent(self, ev):
                if ev.button() == Qt.LeftButton:
                    self._last = None
                    self.setCursor(Qt.ArrowCursor)

            def wheelEvent(self, ev):
                f = 0.9 if ev.angleDelta().y() > 0 else 1.1
                self._zoom = max(0.5, min(20.0, self._zoom * f))
                self.update()

        widget = _InlineMesh3DWidget(path)
        widget.setMaximumWidth(max(40, self._max_w - 16))
        layout.addWidget(widget)
        self._keepalive.append(widget)
    
    # ------------------------------------------------------------------
    # Python — exec PySide6 source and embed the resulting widget.
    #
    # Same conventions as $term/parse:
    #   - Exec namespace exposes a few common imports and `parent_widget`.
    #   - User code is expected to assign `_widget` (a QWidget).
    #   - We embed it in our QFrame, so the user gets an inline rendering
    #     equivalent of $term/parse but without spawning a side panel.
    # ------------------------------------------------------------------
    
    def _render_python(self, layout, payload):
        """Render a Python payload.

        Two modes, selected by payload['_unsafe_origin']:

        1. **Trusted (no flag).** Payload came in via a direct write to
           $term/inline — i.e. the user or a local script did something
           like `>>> ...` or `echo apps/foo.py > $inline`. The sender
           had a local handle on the terminal's filesystem, so the
           usual UNIX permission story applies. We auto-exec, same as
           before, via _exec_python_payload.

        2. **Peribus (flag = "peribus").** Payload was resolved from an
           attachment on a post card — bytes that came over the wire
           from some peer's social/ directory. We have no idea who
           wrote it or what it does. Until peribus has a real sandbox,
           the human in front of the terminal is the sandbox: we show
           the source read-only with a ▶ Run code button and let the
           user decide.

        The flag is set by _render_attachments (the only peribus-side
        path that builds python payloads). Anything else stays
        auto-exec, matching the pre-existing trust model.
        """
        code = payload.get("code") or payload.get("content") or ""
        if not code.strip():
            self._render_info(layout, {"text": "python: empty code"}, "warn")
            return

        # Trusted origins skip the gate — same behavior the terminal
        # has always had for direct writes to $inline.
        if payload.get("_unsafe_origin") != "peribus":
            self._exec_python_payload(layout, payload)
            return

        # Container that holds the source preview now and gets replaced
        # by the executed widget tree on Run.
        gate_frame = QFrame()
        gate_layout = QVBoxLayout(gate_frame)
        gate_layout.setContentsMargins(0, 0, 0, 0)
        gate_layout.setSpacing(4)

        # Warning banner — small, dim, but unmistakable.
        warn = QLabel("⚠ Python attached to a peribus post — review before running")
        if self._dark_mode:
            warn.setStyleSheet(
                "color: rgba(230, 180, 90, 220); font-size: 10px; "
                "font-weight: bold; padding: 2px 4px;"
            )
        else:
            warn.setStyleSheet(
                "color: rgba(160, 100, 20, 220); font-size: 10px; "
                "font-weight: bold; padding: 2px 4px;"
            )
        gate_layout.addWidget(warn)

        # Read-only source view. Editable=False is deliberate: this is
        # what arrived over the wire, edits would just confuse the
        # threat model. If the user wants to tweak, they can copy the
        # code into a >>> block or /parse and run from there.
        src_view = QTextEdit()
        src_view.setReadOnly(True)
        src_view.setPlainText(code)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        src_view.setFont(mono)
        src_view.setLineWrapMode(QTextEdit.NoWrap)
        if self._dark_mode:
            src_view.setStyleSheet("""
                QTextEdit {
                    background-color: rgba(255, 255, 255, 12);
                    color: rgba(220, 220, 220, 240);
                    border: 1px solid rgba(255, 255, 255, 30);
                    border-radius: 3px;
                    padding: 4px;
                }
            """)
        else:
            src_view.setStyleSheet("""
                QTextEdit {
                    background-color: rgba(0, 0, 0, 10);
                    color: rgba(40, 40, 40, 240);
                    border: 1px solid rgba(0, 0, 0, 40);
                    border-radius: 3px;
                    padding: 4px;
                }
            """)
        # Roughly cap the preview height so a 2000-line payload doesn't
        # blow out the terminal. Scrollbar handles the rest.
        src_view.setMaximumHeight(min(self._MEDIA_MAX_H, 320))
        gate_layout.addWidget(src_view)

        # Button row: Run + Copy. We reuse InlineCodeBlockWidget's button
        # style verbatim so this looks like the >>> blocks the user
        # already trusts.
        button_row = QHBoxLayout()
        button_row.setSpacing(4)

        run_btn = QPushButton("▶ Run code")
        run_btn.setCursor(Qt.PointingHandCursor)
        run_btn.setFixedHeight(22)
        run_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(80, 170, 90, 220);
                border: none;
                border-radius: 3px;
                padding: 2px 12px;
                color: white;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: rgba(100, 190, 110, 240);
            }
            QPushButton:pressed {
                background-color: rgba(70, 150, 80, 255);
            }
        """)

        copy_btn = QPushButton("Copy")
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.setFixedHeight(22)
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(180, 180, 180, 60);
                border: 1px solid rgba(160, 160, 160, 80);
                border-radius: 3px;
                padding: 2px 10px;
                color: inherit;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: rgba(180, 180, 180, 110);
            }
        """)

        def _on_copy():
            QApplication.clipboard().setText(code)
            copy_btn.setText("Copied ✓")
            QTimer.singleShot(900, lambda: copy_btn.setText("Copy"))
        copy_btn.clicked.connect(_on_copy)

        def _on_run():
            # Swap the gate for the executed widget tree. We build an
            # inner layout inside gate_frame and hand THAT to the exec
            # path, so error messages, the QMainWindow, etc. all land
            # in the same slot in the parent layout.
            # Clear the gate UI (warning, source view, button row).
            warn.hide()
            src_view.hide()
            run_btn.hide()
            copy_btn.hide()
            # New container for the exec output, appended below the
            # (now-hidden) gate widgets. We can't replace gate_frame
            # in `layout` cleanly without holding its index, so we
            # just append; visually the result is identical.
            exec_holder = QVBoxLayout()
            exec_holder.setContentsMargins(0, 0, 0, 0)
            exec_holder.setSpacing(3)
            gate_layout.addLayout(exec_holder)
            self._exec_python_payload(exec_holder, payload)
        run_btn.clicked.connect(_on_run)

        button_row.addWidget(run_btn)
        button_row.addWidget(copy_btn)
        button_row.addStretch()
        gate_layout.addLayout(button_row)

        layout.addWidget(gate_frame)
        # Keep the gate frame alive (the layout owns it via parenting,
        # but keepalive is the convention used elsewhere in this class).
        self._keepalive.append(gate_frame)

    def _exec_python_payload(self, layout, payload):
        """Execute PySide6 code inline. The exec namespace mirrors the
        one used by /n/<machine>/scene/parse and /n/<machine>/terms/<id>/parse,
        so the same code is valid across all three entry points.
        
        Topology constructed for the user code:
        
            QMainWindow (main_window)
              └── QGraphicsView (graphics_view) — set as central widget
                    └── QGraphicsScene (graphics_scene)
                          └── (whatever user code adds via .addWidget())
        
        This makes the inline namespace a true peer of /scene/parse:
        
          - `graphics_scene.addWidget(x)`         renders inline ✓
          - `graphics_scene.sceneRect()`,
            `.views()`, `.setBackgroundBrush()`,
            `.update()`                            all work ✓
          - `main_window.centralWidget()`         returns the QGraphicsView,
            so dashboard.py-style code that does
            `view = main_window.centralWidget(); view.setRenderHint(...)`
            works unchanged                       ✓
          - `main_window.findChild(QGraphicsView)` returns the inline view ✓
          - `main_window.setCentralWidget(w)`     replaces the view (user
            opt-in: they lose inline scene rendering for that widget but
            the rest of the QMainWindow API still works)
          - `_widget = X` legacy convention:      added via
            `graphics_scene.addWidget(X)` so it renders inline as well
        
        Bindings provided (matching parser.py's Parser._build_namespace):
          - main_window     — real QMainWindow embedded in the inline body
          - graphics_scene  — real QGraphicsScene (1920×1080 default rect)
          - graphics_view   — real QGraphicsView (set as main_window's central widget)
          - scene_manager   — host terminal's panel scene_manager if available,
                              else a no-op stub so `.register_parsed_item(...)`
                              calls don't crash. Calls are silently swallowed.
          - All public classes from QtWidgets, common QtCore types
            (Qt, QTimer, QRect, QPoint, …), common QtGui types
            (QColor, QPen, QBrush, QFont, QPixmap, QImage, QPainter).
          - Helpers: math (sin/cos/tan/sqrt/pi/e), asyncio, os, sys, json,
            and numpy/pandas if installed.
        
        Aliased for backwards compat:
          - parent_widget, inline_widget — point at the inline QFrame itself.
        """
        code = payload.get("code") or payload.get("content") or ""
        if not code.strip():
            self._render_info(layout, {"text": "python: empty code"}, "warn")
            return
        
        # ---- Build the QMainWindow + graphics_scene + graphics_view trio ----
        # Done BEFORE exec so all three already exist in the widget tree
        # when user code runs (matters for paint events / signal connections).
        host = QMainWindow()
        host.setAttribute(Qt.WA_TranslucentBackground, True)
        host.setStyleSheet("QMainWindow { background: transparent; border: none; }")
        host.setMaximumHeight(self._MEDIA_MAX_H)
        
        # Match SceneManager's default scene size so code that uses
        # absolute coordinates (proxy.setPos(x, y) with hundreds-pixel
        # offsets, as dashboard.py does) lays out the same way it would
        # in /scene/parse.
        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, 1920, 1080)
        
        # Use a fit-to-content view so scenes designed for 1920×1080
        # (the SceneManager default) fit cleanly into our ~700×480
        # inline frame regardless of what the user code laid out.
        # Without this, content placed at native scene coordinates
        # gets cropped by the smaller viewport and the user has to
        # scroll horizontally and vertically to see it. See
        # _FittingGraphicsView below for the rationale on why we use
        # itemsBoundingRect() instead of sceneRect().
        view = _FittingGraphicsView(scene)
        view.setRenderHint(QPainter.Antialiasing, True)
        view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        # Scrollbars off — fitInView removes the need to scroll, and
        # leaving them on causes a brief flash of scrollbars during the
        # initial layout pass before the first fit.
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setStyleSheet(
            "QGraphicsView { background: transparent; border: none; }"
        )
        host.setCentralWidget(view)
        
        layout.addWidget(host)
        
        ns = self._build_inline_python_namespace(
            host_window=host, scene=scene, scene_view=view,
        )
        
        try:
            exec(code, ns)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            err_label = QLabel("⚠ inline python error")
            err_label.setStyleSheet(
                "color: #cc4444; font-size: 11px; font-weight: bold;"
            )
            layout.addWidget(err_label)
            tb_view = QTextEdit()
            tb_view.setReadOnly(True)
            tb_view.setPlainText(tb)
            tb_view.setStyleSheet("""
                QTextEdit {
                    background-color: rgba(0, 0, 0, 0);
                    color: rgba(200, 80, 80, 240);
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 10px;
                    border: 1px solid rgba(180, 80, 80, 80);
                    border-radius: 3px;
                    padding: 4px;
                }
            """)
            tb_view.setMaximumHeight(140)
            layout.addWidget(tb_view)
            return

        # Hand the namespace to the signal bus so any anonymous
        # RemoteSignal(...) instances get their variable / attribute
        # name bound on the wire. Without this, `stroke = RemoteSignal(...)`
        # at module level would emit locally but peers couldn't route
        # incoming signals to it (the bus wouldn't know it's called
        # "stroke"). Cheap (no-op if the bus isn't running or there
        # are no anonymous instances). Mirrors parser.py's
        # `_bus.adopt_namespace(namespace)` post-exec hook so the
        # inline and /scene/parse paths behave identically.
        try:
            from rio.signals import _global_bus
            _bus = _global_bus()
            if _bus is not None:
                _bus.adopt_namespace(ns)
        except Exception:
            import traceback as _tb_mod
            _tb_mod.print_exc()
        
        # ---- Handle the legacy _widget = X convention ----
        # If the user assigned _widget directly (instead of using
        # graphics_scene.addWidget), embed it via the scene so it
        # still renders inline. Skip if the widget was already added
        # to the scene by user code (e.g. they did `proxy = scene.addWidget(w)`
        # AND happened to assign `_widget = w` — we'd otherwise add twice).
        produced = ns.get('_widget') or ns.get('widget')
        if produced is not None:
            if not isinstance(produced, QWidget):
                self._render_info(
                    layout,
                    {"text": f"python: `_widget` is not a QWidget (got {type(produced).__name__})"},
                    "warn",
                )
            else:
                # graphicsProxyWidget() returns the QGraphicsProxyWidget
                # for an embedded widget, or None if not embedded.
                already_in_scene = produced.graphicsProxyWidget() is not None
                if not already_in_scene:
                    # If the central widget is still our QGraphicsView,
                    # add via the scene. Otherwise the user replaced central
                    # widget with something else — try its layout, falling
                    # back to making _widget the new central widget.
                    cw = host.centralWidget()
                    if cw is view:
                        scene.addWidget(produced)
                    elif cw is not None and cw.layout() is not None:
                        cw.layout().addWidget(produced)
                    else:
                        host.setCentralWidget(produced)
        
        # Keep the namespace alive — user code may have stored event
        # handlers, timers, etc. as locals, and we don't want them GC'd
        # while the widget is still visible.
        self._keepalive.append(ns)
        # Hold explicit references to scene/view/host so they survive
        # even if user code reassigned the namespace bindings.
        self._keepalive.append(host)
        self._keepalive.append(view)
        self._keepalive.append(scene)
        # Now that user code has populated the scene, kick the view to
        # re-fit. The view's __init__ already scheduled a deferred fit
        # via QTimer.singleShot, but doing it here too means the *first*
        # paint is already fitted instead of showing a 1:1 frame for one
        # tick. Safe to call before show() — fit_to_contents bails on
        # empty viewport size, and showEvent re-fits when the viewport
        # is real.
        if isinstance(view, _FittingGraphicsView):
            view.fit_to_contents()
    
    def _build_inline_python_namespace(self, host_window=None,
                                        scene=None, scene_view=None) -> dict:
        """Build the exec namespace for inline-Python payloads.
        
        Source of truth: rio.scene's Parser._build_namespace(). We mirror
        its bindings here so that code written for /scene/parse or
        /term/<id>/parse runs unchanged inline.
        
        Args:
          host_window: QMainWindow used as `main_window`. When provided,
            user code can call .centralWidget(), .setCentralWidget(),
            .menuBar(), .statusBar(), etc. — same API surface as
            /scene/parse's main_window.
          scene: QGraphicsScene used as `graphics_scene`. When provided,
            user code can call .addWidget(), .sceneRect(), .views(),
            .setBackgroundBrush(), .update() etc. — same as /scene/parse.
          scene_view: QGraphicsView used as `graphics_view`. When provided,
            user code can call .mapToScene(), .viewport(), .setRenderHint()
            etc. directly (also reachable via scene.views()[0]).
        
        Falls back to terminal scene_panel bindings if none of the above
        are passed (legacy callers); falls back to None / self if neither
        is available.
        
        Anything added to parser.py's namespace template should also be
        added here.
        """
        import builtins
        ns: dict = {}
        ns['__builtins__'] = builtins.__dict__.copy()
        ns['__name__'] = '__inline_python__'
        
        # --- Resolve scene / view / scene_manager ---
        # 1. Explicit args win — that's what _render_python uses to
        #    inject a per-render scene/view, which is what makes
        #    `graphics_scene.addWidget(x)` actually render inline.
        # 2. Otherwise, fall back to the host terminal's panel scene
        #    (legacy behaviour for callers that don't pass scene/view).
        # 3. Otherwise leave them None and hope user code checks.
        graphics_scene = scene
        graphics_view = scene_view
        scene_manager = None
        
        terminal = self._find_host_terminal()
        if terminal is not None:
            panel = getattr(terminal, 'scene_panel', None)
            if panel is not None:
                if graphics_scene is None:
                    graphics_scene = getattr(panel, '_scene', None)
                if graphics_view is None:
                    graphics_view = getattr(panel, '_view', None)
                scene_manager = getattr(panel, 'scene_manager', None)
        
        # If we built our own inline scene but didn't find a real
        # SceneManager, give the user a no-op stub so calls like
        # `scene_manager.register_parsed_item(proxy, {...})` don't crash.
        # This matches the API surface that /scene/parse code expects
        # without doing anything that would conflict with a real scene
        # manager elsewhere.
        if scene_manager is None and scene is not None:
            scene_manager = _InlineSceneManagerStub(scene)
        
        # main_window: real QMainWindow when one was passed in, so
        # `main_window.centralWidget()`, `setCentralWidget()`, `menuBar()`,
        # `statusBar()`, `addToolBar()` etc. all work — same API surface
        # the global /scene/parse namespace provides.
        ns['main_window'] = host_window if host_window is not None else self
        # Backwards-compat aliases for code written against the older
        # inline-only contract — these stay pointed at the inline widget
        # itself, since some agents may use them for "the place this
        # inline render lives" semantics rather than QMainWindow semantics.
        ns['parent_widget'] = self
        ns['inline_widget'] = self
        ns['graphics_scene'] = graphics_scene
        ns['graphics_view'] = graphics_view
        ns['scene_manager'] = scene_manager
        
        # --- Qt bindings: dump every public class from QtWidgets, plus
        #     the common helper types from QtCore and QtGui, exactly
        #     as parser.py does.
        try:
            from PySide6 import QtWidgets, QtCore, QtGui
            ns['QtWidgets'] = QtWidgets
            ns['QtCore'] = QtCore
            ns['QtGui'] = QtGui
            for name in dir(QtWidgets):
                if not name.startswith('_'):
                    obj = getattr(QtWidgets, name)
                    if isinstance(obj, type):
                        ns[name] = obj
            # Same dump for QtGui — previously this was a cherry-picked
            # list (QColor / QBrush / QPen / QFont / QPixmap / QImage /
            # QPainter) which silently dropped QLinearGradient,
            # QRadialGradient, QPainterPath, QPolygon, QTransform,
            # QFontMetrics, QCursor, QIcon, etc. That meant code which
            # ran fine in /scene/parse (whose namespace is built by
            # rio.scene.Parser._build_namespace and exposes QtGui in
            # full) blew up inside /share scene postcards with
            # `name 'QLinearGradient' is not defined` from paintEvent.
            # Mirror the QtWidgets pattern: dump every public class.
            for name in dir(QtGui):
                if not name.startswith('_'):
                    obj = getattr(QtGui, name)
                    if isinstance(obj, type):
                        ns[name] = obj
            ns['Qt'] = QtCore.Qt
            ns['QTimer'] = QtCore.QTimer
            ns['QRect'] = QtCore.QRect
            ns['QRectF'] = QtCore.QRectF
            ns['QPoint'] = QtCore.QPoint
            ns['QPointF'] = QtCore.QPointF
            ns['QSize'] = QtCore.QSize
            ns['QSizeF'] = QtCore.QSizeF
            ns['Signal'] = QtCore.Signal
            ns['Slot'] = QtCore.Slot
            # Cross-machine signal layer — mirrors parser.py exactly so
            # the same code (e.g. `stroke = RemoteSignal(float, float, ...)`
            # and `subscribe("cirno")`) runs unchanged inline, in
            # /scene/parse, and in /terms/<id>/parse.
            #
            # `RemoteSignal` is a drop-in for PySide6.Signal in shape
            # (connect/emit/disconnect) but also crosses to subscribed
            # peers via UDP. It works both as a class attribute on a
            # QObject AND as a free-standing module-level binding —
            # `bus.adopt_namespace(ns)` below names anonymous instances
            # after each exec so the wire knows what `stroke` is.
            try:
                from rio.signals import (
                    Signal as _RioSignal,
                    subscribe as _rio_subscribe,
                    unsubscribe as _rio_unsubscribe,
                )
                ns['RemoteSignal'] = _RioSignal
                ns['subscribe'] = _rio_subscribe
                ns['unsubscribe'] = _rio_unsubscribe
            except ImportError:
                # rio.signals not available — stub out so user code
                # mentioning these names doesn't crash with NameError.
                # `RemoteSignal` falls back to PySide6.Signal so existing
                # class-attribute uses keep working in a degraded
                # "local only" mode. Free-standing `RemoteSignal(...)`
                # assignment at module scope WILL fail in this fallback
                # (PySide6.Signal isn't instantiable that way), but the
                # name resolves and the error is clearly about Signal
                # rather than NameError.
                ns['RemoteSignal'] = QtCore.Signal
                ns['subscribe'] = lambda *a, **kw: False
                ns['unsubscribe'] = lambda *a, **kw: False
            # The QtGui dump above already binds these, but keep the
            # explicit names too — both for documentation value and so
            # any caller introspecting the namespace doesn't suddenly
            # see them disappear. (They resolve to the same classes.)
            ns['QColor'] = QtGui.QColor
            ns['QBrush'] = QtGui.QBrush
            ns['QPen'] = QtGui.QPen
            ns['QFont'] = QtGui.QFont
            ns['QPixmap'] = QtGui.QPixmap
            ns['QImage'] = QtGui.QImage
            ns['QPainter'] = QtGui.QPainter
        except ImportError:
            pass
        
        # Optional QtWebEngine — same as parser.py.
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            ns['QWebEngineView'] = QWebEngineView
        except ImportError:
            pass
        
        # --- Math / data / system shorthands.
        import math
        import random
        ns.update({
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        })
        import asyncio as _asyncio
        ns['asyncio'] = _asyncio
        try:
            import numpy as np
            ns['np'] = np
            ns['numpy'] = np
        except ImportError:
            pass
        try:
            import pandas as pd
            ns['pd'] = pd
            ns['pandas'] = pd
        except ImportError:
            pass
        import json as _json
        import os as _os
        import sys as _sys
        ns['json'] = _json
        ns['os'] = _os
        ns['sys'] = _sys
        return ns
    
    def _find_host_terminal(self):
        """Find the enclosing TerminalWidget that owns this inline widget.

        Two paths:

          1. Explicit hint via the host_terminal kwarg. Set by callers
             that know the terminal at construction time (the typical
             top-level case via TerminalStreamRouter.insert_media).
             This is the only reliable path for top-level widgets,
             because they're constructed with parent=None and aren't
             added to a layout until *after* __init__ returns. Asking
             for self.parent() during __init__ would return None.

          2. Parent-chain walk. Used by nested widgets created from
             inside another inline widget (e.g. _render_post_media's
             quote-posts, or _render_attachments's resolved media).
             Those have self.parent() set by Qt as soon as they're
             added to the parent's layout, which happens before any
             rendering inside them.

        The hint can also propagate to nested children — see
        _render_post_media and _render_attachments which forward it on
        when they construct a child.

        Returns None if neither path finds a TerminalWidget. Callers
        treat None as "no terminal context"; some features (action
        buttons, byline tick) gracefully degrade in that case.
        """
        # 1) Explicit hint, if any.
        hint = getattr(self, "_host_terminal_hint", None)
        if hint is not None and self._looks_like_terminal(hint):
            return hint

        # 2) Walk the parent chain. Bound the walk so a malformed
        # hierarchy can't loop forever.
        p = self.parent()
        for _ in range(16):
            if p is None:
                return None
            if self._looks_like_terminal(p):
                return p
            p = p.parent() if hasattr(p, 'parent') else None
        return None

    @staticmethod
    def _looks_like_terminal(obj) -> bool:
        """Duck-typed terminal recognition.

        Match by attribute presence rather than class name to avoid
        forward references and circular import dances. Picks the
        smallest signature that's actually unique to TerminalWidget.
        """
        return (hasattr(obj, 'scene_panel') and hasattr(obj, 'term_id')
                and hasattr(obj, 'connected_agent'))
    
    # ------------------------------------------------------------------
    # HTML / link / info (unchanged)
    # ------------------------------------------------------------------
    
    def _render_html(self, layout, payload):
        # We deliberately use QLabel + Qt rich text (limited HTML) instead of
        # QWebEngineView to keep this lightweight and avoid pulling in the
        # web stack. For complex docs, the agent should use the scene panel.
        content = payload.get("content") or ""
        label = QLabel(content)
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        if self._dark_mode:
            label.setStyleSheet("color: rgba(220, 225, 235, 255); font-size: 12px;")
        else:
            label.setStyleSheet("color: rgba(20, 25, 35, 255); font-size: 12px;")
        layout.addWidget(label)
    
    # ------------------------------------------------------------------
    # URL — full QWebEngineView with minimal navigation chrome.
    # 
    # Heavy by design: QWebEngine spins up a Chromium subprocess for
    # rendering. Stays in _LAZY_TYPES so the cost is only paid when the
    # widget is actually expanded.
    # ------------------------------------------------------------------
    
    def _render_url(self, layout, payload):
        url_str = payload.get("url") or ""
        if not url_str:
            self._render_info(layout, {"text": "url: empty"}, "warn")
            return
        # Auto-prepend scheme so bare `www.foo.com` or `foo.com` work.
        normalized = self._normalize_url(url_str)
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtCore import QUrl
        except ImportError as e:
            self._render_info(
                layout,
                {"text": f"url backend (QtWebEngine) missing: {e}"},
                "error",
            )
            return
        from PySide6.QtWidgets import QLineEdit
        
        # Mini nav strip: URL bar + reload + open-externally button.
        # Kept compact so it doesn't compete with the page for vertical
        # space. No back/forward — most inline use cases are point-views,
        # not browsing sessions; for a fuller browser the agent can write
        # explicit Python via $inline.
        nav = QWidget()
        nav.setStyleSheet("background: transparent;")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        
        url_bar = QLineEdit()
        url_bar.setText(normalized)
        url_bar.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 30);
                color: rgba(180, 200, 230, 255);
                border: 1px solid rgba(120, 140, 170, 80);
                border-radius: 3px;
                padding: 2px 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
            QLineEdit:focus { border-color: rgba(120, 160, 220, 200); }
        """)
        url_bar.setFixedHeight(22)
        nav_layout.addWidget(url_bar, stretch=1)
        
        def _btn(label, tooltip):
            b = QPushButton(label)
            b.setFixedSize(24, 22)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tooltip)
            b.setStyleSheet("""
                QPushButton {
                    background-color: rgba(180, 180, 180, 60);
                    border: 1px solid rgba(160, 160, 160, 80);
                    border-radius: 3px;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: rgba(180, 180, 180, 110); }
            """)
            return b
        
        reload_btn = _btn("⟳", "Reload")
        nav_layout.addWidget(reload_btn)
        layout.addWidget(nav)
        
        web_view = QWebEngineView()
        web_view.setMaximumHeight(self._MEDIA_MAX_H - 30)  # leave room for nav strip
        web_view.setUrl(QUrl(normalized))
        layout.addWidget(web_view)
        
        # Wire up navigation
        def _navigate():
            new = self._normalize_url(url_bar.text().strip())
            if new:
                url_bar.setText(new)
                web_view.setUrl(QUrl(new))
        
        url_bar.returnPressed.connect(_navigate)
        reload_btn.clicked.connect(web_view.reload)
        # Keep URL bar in sync with actual page URL (handles redirects /
        # in-page link clicks).
        web_view.urlChanged.connect(lambda qurl: url_bar.setText(qurl.toString()))
        
        # Hold web_view in keepalive so it survives even if Qt's parent-
        # tracking gets confused by the Chromium subprocess.
        self._keepalive.append(web_view)
    
    @staticmethod
    def _normalize_url(text: str) -> str:
        """Make a user-typed URL string into something QUrl can load.
        
        Rules:
          - Already has scheme (http://, https://, file://, ftp://) → as-is.
          - Starts with `www.` → prepend `https://`.
          - Anything else with at least one dot → prepend `https://`.
          - Empty → empty (caller's problem).
        """
        text = (text or '').strip()
        if not text:
            return ''
        lowered = text.lower()
        if (lowered.startswith('http://') or lowered.startswith('https://')
                or lowered.startswith('file://') or lowered.startswith('ftp://')):
            return text
        return 'https://' + text
    
    def _render_link(self, layout, payload):
        url = payload.get("url") or ""
        label_text = payload.get("label") or url
        html = f'<a href="{url}" style="text-decoration: underline;">{label_text}</a>'
        label = QLabel(html)
        label.setTextFormat(Qt.RichText)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.LinksAccessibleByMouse | Qt.TextSelectableByMouse)
        layout.addWidget(label)
    
    def _render_post(self, layout, payload):
        """Render a peribus feed post as an inline card.

        Schema (matches peribus.feed_bridge.Card.from_feed_line plus a
        type tag, so the same JSON line that flows on /n/peribus/feed/new
        becomes a valid $term/inline payload by adding "type":"post"):

            {"type": "post",
             "id":         "<post hash>",
             "author":     "<NodeID>",
             "title":      "...",            (optional)
             "body":       "...",
             "resonance":  0.0–1.0,           (optional)
             "ts":         1690000000.0,      (optional)
             "attachments":["<hash>", ...],   (optional, content-addressed
                                              refs into the rhizome)
             "media":      [<inline-payload>, …]}  (optional, embedded
                                              media: each entry is an
                                              ordinary $term/inline
                                              envelope — image, video,
                                              python, html, etc. — that
                                              renders nested within this
                                              post card)

        ``attachments`` and ``media`` are deliberately separate concerns:
        the former is an *identifier list* (hash refs the daemon can fetch
        out of band), the latter is *embedded payload* (already-rendered-
        in-place content). A post can carry one, both, or neither.
        """
        author = (payload.get('author') or '?').strip()
        title = (payload.get('title') or '').strip()
        body = (payload.get('body') or '').strip()
        resonance = payload.get('resonance')
        ts = payload.get('ts')
        attachments = payload.get('attachments') or []
        media = payload.get('media') or []
        post_id = (payload.get('id') or '').strip()
        is_inbox = payload.get('_card_kind') == "inbox"

        # ---- color palette (mirror the rest of InlineMediaWidget) ----
        # Inbox cards use a pink-leaning palette so they read as "this
        # is mail" the moment they appear, distinct from the cool-blue
        # palette of public posts. Same overall lightness as the post
        # palette so dark/light mode contrast and legibility hold.
        if is_inbox:
            if self._dark_mode:
                byline_color = "rgba(220, 160, 190, 235)"
                title_color = "rgba(245, 220, 232, 255)"
                body_color = "rgba(235, 215, 225, 255)"
                meta_color = "rgba(190, 150, 170, 210)"
                chip_bg = "rgba(120, 60, 90, 170)"
                chip_fg = "rgba(245, 200, 220, 235)"
                badge_bg = "rgba(180, 90, 130, 200)"
                badge_fg = "rgba(255, 240, 248, 245)"
            else:
                byline_color = "rgba(150, 60, 100, 235)"
                title_color = "rgba(80, 25, 55, 255)"
                body_color = "rgba(80, 35, 60, 255)"
                meta_color = "rgba(170, 110, 140, 210)"
                chip_bg = "rgba(248, 215, 230, 230)"
                chip_fg = "rgba(140, 60, 100, 235)"
                badge_bg = "rgba(220, 130, 170, 220)"
                badge_fg = "rgba(255, 250, 252, 250)"
        elif self._dark_mode:
            byline_color = "rgba(150, 165, 200, 230)"
            title_color = "rgba(225, 230, 240, 255)"
            body_color = "rgba(210, 215, 225, 255)"
            meta_color = "rgba(140, 150, 170, 200)"
            chip_bg = "rgba(60, 75, 105, 160)"
            chip_fg = "rgba(190, 210, 235, 230)"
            badge_bg = ""  # unused for non-inbox
            badge_fg = ""
        else:
            byline_color = "rgba(70, 90, 130, 230)"
            title_color = "rgba(20, 30, 55, 255)"
            body_color = "rgba(40, 50, 70, 255)"
            meta_color = "rgba(120, 130, 150, 200)"
            chip_bg = "rgba(225, 232, 245, 200)"
            chip_fg = "rgba(60, 80, 120, 230)"
            badge_bg = ""
            badge_fg = ""

        # ---- DM badge row (inbox cards only) ----
        # A small "✉ DM" pill at the top of the card body — overkill on
        # the byline alone; better as a deliberate marker that's hard to
        # miss when scrolling a busy feed. Sits above the byline so the
        # reader's eye lands on it first.
        if is_inbox:
            badge_row = QWidget()
            badge_row_layout = QHBoxLayout(badge_row)
            badge_row_layout.setContentsMargins(2, 2, 0, 0)
            badge_row_layout.setSpacing(0)
            badge_lbl = QLabel("✉ DM")
            badge_lbl.setStyleSheet(
                f"background: {badge_bg}; color: {badge_fg}; "
                f"font-size: 10px; font-weight: 700; "
                f"padding: 2px 8px; border-radius: 8px;"
            )
            badge_row_layout.addWidget(badge_lbl)
            badge_row_layout.addStretch(1)
            layout.addWidget(badge_row)

        # ---- byline (author + relative time) ----
        # Stored as separate static + dynamic parts so the host terminal's
        # tick can refresh the relative-time portion without us having to
        # rebuild the whole label every 30 seconds.
        author_short = author
        if len(author_short) > 16:
            author_short = author_short[:8] + '…' + author_short[-6:]
        # Locally-rendered posts (from /share before the wire round-trip)
        # carry the sentinel "(local)" so the byline reads naturally
        # rather than as "@(local)".
        if author == "(local)":
            byline_static_bits = ["you (local)"]
        else:
            byline_static_bits = [f"@{author_short}"]
        # Resonance is a feed-ranking score and only meaningful for
        # public posts. DMs don't have one.
        if not is_inbox and isinstance(resonance, (int, float)):
            byline_static_bits.append(f"resonance {float(resonance):.2f}")
        byline_lbl = QLabel()
        byline_lbl.setStyleSheet(
            f"color: {byline_color}; font-size: 11px; "
            f"font-weight: 600; padding: 0px 2px;"
        )
        byline_lbl.setWordWrap(True)
        layout.addWidget(byline_lbl)

        # Stash on self so the host terminal's tick can find us. Using
        # plain attributes (not a registry of weakrefs) because Qt already
        # gives us a deletion signal we can hook into below.
        self._post_byline_label = byline_lbl
        self._post_byline_static_bits = byline_static_bits
        self._post_byline_byline_color = byline_color  # for colorScheme refresh
        try:
            self._post_ts = float(ts) if ts is not None else None
        except (TypeError, ValueError):
            self._post_ts = None
        # Render the byline once, immediately.
        self._refresh_post_byline()

        # ---- orphan-reply context ----
        # If this post is tagged as a reply (reply_to set in the
        # envelope) and we're still rendering at the top level, the
        # parent card isn't mounted on our terminal — either it
        # scrolled off, we joined the network late, or the parent post
        # never reached us. Show a small "↳ reply to <hash>" badge so
        # the user knows this isn't a freestanding post. Without this,
        # the body alone reads as a context-free statement.
        reply_to = (payload.get("reply_to") or "").strip()
        if reply_to and not is_inbox:
            short_parent = (
                reply_to if len(reply_to) <= 14
                else reply_to[:6] + '…' + reply_to[-4:]
            )
            reply_to_author = (
                payload.get("reply_to_author") or ""
            ).strip()
            if reply_to_author:
                short_author = self._short_nodeid(reply_to_author)
                ctx_text = f"↳ reply to @{short_author} · {short_parent}"
            else:
                ctx_text = f"↳ reply to {short_parent}"
            ctx_lbl = QLabel(ctx_text)
            ctx_lbl.setWordWrap(True)
            ctx_lbl.setStyleSheet(
                f"color: {meta_color}; font-size: 10px; "
                f"font-style: italic; padding: 1px 4px 2px 2px;"
            )
            layout.addWidget(ctx_lbl)

        # ---- title (optional) ----
        if title:
            title_lbl = QLabel(title)
            title_lbl.setWordWrap(True)
            title_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            title_lbl.setStyleSheet(
                f"color: {title_color}; font-size: 14px; "
                f"font-weight: 700; padding: 2px 2px 0px 2px;"
            )
            layout.addWidget(title_lbl)

        # ---- body ----
        if body:
            body_lbl = QLabel(body)
            body_lbl.setWordWrap(True)
            body_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            body_lbl.setStyleSheet(
                f"color: {body_color}; font-size: 12px; "
                f"padding: 2px 2px 4px 2px;"
            )
            layout.addWidget(body_lbl)

        # ---- embedded media (image / video / python / html / …) ----
        # Each media entry is a normal inline-widget payload — same schema
        # InlineMediaWidget already accepts. We instantiate a child
        # InlineMediaWidget for each one and stack them in. This way every
        # renderer (image scaling, video lazy-load, python sandbox, …) is
        # reused unchanged; we don't re-implement any of them.
        #
        # When the publisher promoted oversized media to attachments, the
        # wire envelope carries BOTH a `_stripped` media stub (metadata
        # only) AND the blob hash in `attachments`. We render the
        # attachment with the real content and skip the stub — otherwise
        # peers see a redundant "info: shared: foo.py (5,970 bytes)"
        # chip stacked above the actual rendered code. The stub still
        # serves a purpose: if attachments fails to resolve (peer offline,
        # blob expired from cache), the stub is retained as a fallback
        # chip so the user at least sees the file metadata.
        #
        # Infinite recursion is structurally impossible: InlineMediaWidget
        # only nests another InlineMediaWidget when the child payload is
        # itself a "post" type, and a post nested inside a post is a sane
        # operation (a quote-post). We cap nesting depth at 3 just so a
        # malicious or buggy producer can't blow the stack.
        if media:
            if attachments:
                # Suppress stripped stubs that the attachment will replace.
                visible_media = [
                    m for m in media
                    if not (isinstance(m, dict) and m.get("_stripped"))
                ]
            else:
                visible_media = media
            if visible_media:
                self._render_post_media(layout, visible_media)

        # ---- attachments (content-addressed refs from peribus) ----
        # When a post lands with `attachments`, these are content
        # hashes the daemon assigned at publish time. The bytes live
        # somewhere under /n/peribus — we try to fetch and render them
        # inline. Any we can't resolve fall back to a chip.
        #
        # We pass `media` as a kind-hint source: when the publisher
        # promoted oversized media to attachments, the wire envelope
        # carries a stripped media stub (with the publisher's declared
        # `type` and `filename`) alongside the attachment hash. Using
        # the publisher's hint is more reliable than re-sniffing the
        # blob on the receiver, which has no extension to go on (the
        # blob is named by hash, not filename).
        if attachments:
            self._render_attachments(layout, attachments, author,
                                     chip_bg, chip_fg, media_hints=media)

        # ---- action row (deepen / reply / open in browser-of-the-network) ----
        # All actions are filesystem-mediated, matching the rest of peribus.
        # We capture the values needed by each click into local closures.
        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 2, 0, 0)
        action_layout.setSpacing(8)

        terminal = self._find_host_terminal()

        def _styled_btn(label: str) -> QPushButton:
            btn = QPushButton(label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton { background: transparent; border: none; "
                f"color: {byline_color}; font-size: 11px; "
                "font-weight: 600; padding: 2px 6px; } "
                "QPushButton:hover { text-decoration: underline; }"
            )
            return btn

        if terminal is not None:
            # Register with the terminal's tick so the relative time stays
            # current. Unregister when this widget gets destroyed (Qt's
            # `destroyed` signal fires even when the widget is removed via
            # deleteLater(), which is how InlineMediaWidget tears down).
            try:
                terminal._register_post_card(self)
                self.destroyed.connect(
                    lambda _=None, t=terminal, c=self:
                    t._unregister_post_card(c)
                )
            except Exception:
                # If registration fails for any reason, the card still
                # renders — it just won't auto-tick. Better than crashing.
                pass

            # Deepen — write `attract <text>` to /n/peribus/ctl, biasing the
            # identity vector toward this post's body. The cheapest possible
            # "I want more of this" signal. Skipped for DMs: the inbox is
            # private mail, and biasing the public feed off DM content would
            # leak preferences in a confusing way.
            attract_text = (title + " " + body).strip()[:512]
            if attract_text and not is_inbox:
                deepen_btn = _styled_btn("↳ deepen")
                deepen_btn.clicked.connect(
                    lambda _=False, t=attract_text:
                    terminal._peribus_ctl(f"attract {t}")
                )
                action_layout.addWidget(deepen_btn)

            # Reply — opens an inline composer at the bottom of this
            # card, expanding the card into a thread surface. The
            # composer's Send button dispatches by card kind:
            #   - DM card  → write to /n/peribus/inbox/send (private mail)
            #   - post     → write a reply-tagged envelope to share/
            #                (a global comment that threads under the
            #                parent post on every peer's screen)
            # Replacing the previous popup-dialog approach so the user
            # can keep a conversation flowing without losing context.
            if author and author != '?':
                reply_btn = _styled_btn("↩ reply")
                reply_btn.clicked.connect(
                    lambda _=False, c=self: c._toggle_reply_composer()
                )
                action_layout.addWidget(reply_btn)
                # Stash on self so other code (e.g. focus handlers) can
                # find the button later. Cheap and explicit.
                self._reply_btn = reply_btn

            # Open the canonical post path under /n/peribus/nodes/<author>/social/
            # Inbox cards have synthetic ids ("dm:<sender>:<ts>") that don't
            # correspond to a content-addressed file, so the "copy path"
            # button is meaningless for them — suppress.
            if author and author != '?' and post_id and not is_inbox:
                open_btn = _styled_btn("⎘ copy path")
                post_path = f"/n/peribus/nodes/{author}/social/{post_id}"
                open_btn.clicked.connect(
                    lambda _=False, p=post_path, t=terminal:
                    t._copy_to_clipboard(p)
                )
                action_layout.addWidget(open_btn)

        action_layout.addStretch(1)

        # Tiny meta line on the right with the post id, if present.
        # Suppressed for inbox cards because their id is a synthetic
        # "dm:<sender>:<ts>" — protocol bookkeeping, not something the
        # user would want to copy or reference.
        if post_id and not is_inbox:
            short = post_id if len(post_id) <= 14 else post_id[:6] + '…' + post_id[-4:]
            meta = QLabel(short)
            meta.setStyleSheet(
                f"color: {meta_color}; font-size: 10px; font-family: monospace;"
            )
            meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
            action_layout.addWidget(meta)

        layout.addWidget(action_row)

        # ---- thread surface (replies / DM history) ----
        # A vertical layout where new messages can land after the card
        # is rendered. Two writers populate this:
        #
        #   - The host terminal, when a peer's reply-post or DM
        #     arrives. It calls self.append_reply(payload) and a new
        #     row appears here.
        #   - The composer below (when the user sends), which
        #     optimistically appends locally so the thread feels
        #     instant. The receiver-side gossip echo is deduped on
        #     arrival so we don't double-render.
        #
        # We stash the layout (not just the container) so append_reply
        # can insert at index -1 above the composer. The container
        # widget is invisible until at least one reply is appended;
        # otherwise an empty pad would leave the card looking like it's
        # waiting for something.
        self._thread_container = QWidget()
        self._thread_container.setStyleSheet("background: transparent;")
        self._thread_layout = QVBoxLayout(self._thread_container)
        self._thread_layout.setContentsMargins(8, 0, 4, 0)
        self._thread_layout.setSpacing(2)
        self._thread_container.setVisible(False)
        layout.addWidget(self._thread_container)

        # Stash card-identity fields so the host terminal can route
        # incoming messages to the right card without having to dig
        # them back out of self._payload.
        self._post_id = post_id
        self._post_author = author
        self._is_inbox_card = is_inbox
        # Palette echoes for the composer + appended replies so styling
        # stays consistent with the parent card.
        self._reply_palette = {
            "byline_color": byline_color,
            "body_color": body_color,
            "meta_color": meta_color,
            "chip_bg": chip_bg,
            "chip_fg": chip_fg,
        }
        self._compose_widget = None  # built lazily on first toggle

        # Register with the host terminal's card index so peers' replies
        # / DMs can find this card by post_id (for replies) or by
        # author NodeID (for DMs). Idempotent — calling multiple times
        # for the same card just overwrites the same key.
        if terminal is not None:
            try:
                terminal._register_thread_card(self)
                self.destroyed.connect(
                    lambda _=None, t=terminal, c=self:
                    t._unregister_thread_card(c)
                )
            except Exception:
                # Same forgiving stance as the byline-tick registration.
                pass

    # ------------------------------------------------------------------
    # Inline reply composer + thread management
    # ------------------------------------------------------------------
    #
    # When the user clicks ↩ reply we open an inline composer at the
    # bottom of the card (rather than a modal dialog). Sending appends
    # the message both to the wire (DM or reply-post) and to the local
    # thread surface so the conversation reads as a single growing
    # column inside the card.
    #
    # The composer is built lazily because most cards never get a
    # reply — keeps the per-card cost down for busy feeds.

    def _toggle_reply_composer(self) -> None:
        """Show or hide the inline composer for this card."""
        if self._compose_widget is None:
            self._compose_widget = self._build_compose_widget()
            # Insert just below the action row (which we already added
            # to layout). The card's outer layout is self._body_layout
            # for the body and the post-render adds widgets to a
            # caller-supplied layout — we can find that via the
            # compose widget's parent layout after addWidget. Simplest:
            # add to the same layout we used for the thread container.
            #
            # We don't track which layout that is at this point; the
            # _render_post caller passed it as `layout`. We don't keep
            # it. Workaround: walk up from _thread_container to find
            # its parent layout, since _thread_container was added to
            # the same layout right before us.
            parent_layout = (
                self._thread_container.parent().layout()
                if self._thread_container.parent() is not None
                else None
            )
            if parent_layout is not None:
                parent_layout.addWidget(self._compose_widget)
            else:
                # Fall back to dropping into thread_layout — keeps the
                # composer attached to the card tree even if our parent-
                # layout walk fails for any reason.
                self._thread_layout.addWidget(self._compose_widget)
            self._compose_widget.setVisible(True)
            self._focus_compose_input()
            return

        # Toggle visibility on subsequent clicks.
        new_visible = not self._compose_widget.isVisible()
        self._compose_widget.setVisible(new_visible)
        if new_visible:
            self._focus_compose_input()

    def _build_compose_widget(self) -> QWidget:
        """Construct the inline composer (text input + send button)."""
        from PySide6.QtWidgets import QLineEdit
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        wrap_layout = QHBoxLayout(wrap)
        wrap_layout.setContentsMargins(8, 4, 4, 4)
        wrap_layout.setSpacing(6)

        # Single-line input is the right default — replies are short
        # and the Enter key dispatches naturally. For longer replies
        # the user can /share or use the composer dialog from /share.
        line = QLineEdit()
        if self._is_inbox_card:
            placeholder = (
                f"Reply to @{self._short_nodeid(self._post_author)}…"
                if self._post_author else "Reply…"
            )
        else:
            placeholder = "Public reply (everyone will see it)…"
        line.setPlaceholderText(placeholder)
        # Match the card's palette so the composer doesn't read as
        # "stamped on top" of the card. Pink for DMs, the standard
        # neutral for posts.
        if self._is_inbox_card:
            line.setStyleSheet(
                "QLineEdit { background: rgba(255, 240, 247, 200); "
                "color: rgba(80, 35, 60, 255); "
                "border: 1px solid rgba(220, 150, 180, 200); "
                "border-radius: 4px; padding: 5px 8px; font-size: 12px; }"
                "QLineEdit:focus { border-color: rgba(190, 80, 130, 230); }"
            )
        else:
            line.setStyleSheet(
                "QLineEdit { background: rgba(245, 247, 252, 200); "
                "color: rgba(20, 30, 55, 255); "
                "border: 1px solid rgba(190, 200, 215, 180); "
                "border-radius: 4px; padding: 5px 8px; font-size: 12px; }"
                "QLineEdit:focus { border-color: rgba(80, 110, 170, 220); }"
            )
        line.returnPressed.connect(self._submit_reply)
        wrap_layout.addWidget(line, stretch=1)

        send_btn = QPushButton("Send")
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            f"color: {self._reply_palette['byline_color']}; "
            "font-size: 11px; font-weight: 700; padding: 4px 10px; }"
            "QPushButton:hover { text-decoration: underline; }"
        )
        send_btn.clicked.connect(self._submit_reply)
        wrap_layout.addWidget(send_btn)

        # Stash the input so _submit_reply can read it without walking
        # the layout.
        wrap._compose_input = line
        wrap._compose_send = send_btn
        return wrap

    def _focus_compose_input(self) -> None:
        if self._compose_widget is None:
            return
        line = getattr(self._compose_widget, "_compose_input", None)
        if line is not None:
            line.setFocus()

    @staticmethod
    def _short_nodeid(nodeid: str) -> str:
        if not nodeid or len(nodeid) <= 16:
            return nodeid or "?"
        return nodeid[:8] + '…' + nodeid[-6:]

    def _submit_reply(self) -> None:
        """
        Send the composed text. Routing depends on card kind:
          - DM card  → write `<peer> <body>` to /n/peribus/inbox/send.
                       Echo locally because outbound DMs don't come back
                       on inbox/new (they're sent, not received).
          - Post card → write a reply-tagged envelope to share/. The
                       envelope's `reply_to` field carries the parent
                       post id so receivers can thread it. Wait for the
                       gossip echo to render (the existing /share dedupe
                       prevents double-rendering of our own publish).
        """
        if self._compose_widget is None:
            return
        line = getattr(self._compose_widget, "_compose_input", None)
        if line is None:
            return
        body = line.text().strip()
        if not body:
            return
        terminal = self._find_host_terminal()
        if terminal is None:
            return

        line.clear()

        if self._is_inbox_card:
            sent_ok = terminal._peribus_send_dm(
                self._post_author, body,
            )
            if sent_ok:
                # Optimistic local echo. The daemon doesn't replay our
                # outbound DMs back to us through inbox/new, so we have
                # to render our own message here or the user would see
                # an empty card after sending and only see the peer's
                # next reply.
                self.append_reply({
                    "author": "(you)",
                    "ts": time.time(),
                    "body": body,
                })
        else:
            sent_ok = terminal._peribus_send_reply_post(
                parent_id=self._post_id,
                parent_author=self._post_author,
                body=body,
            )
            if sent_ok:
                # Optimistic local echo, threaded directly under this
                # card. This is the most reliable threading path — it
                # bypasses any envelope-shape / index-lookup / wire-
                # round-trip plumbing entirely. We render the reply
                # inside the parent's thread surface the instant the
                # publish acks. The /share dedupe register
                # (_share_remember_publish on the post hash) will then
                # suppress the gossip echo when it bounces back through
                # feed/new, so we render exactly once.
                self.append_reply({
                    "author": "(you)",
                    "ts": time.time(),
                    "body": body,
                })

    def append_reply(self, payload: dict) -> None:
        """
        Add a reply (or DM) entry to this card's thread surface.

        Called from two places:
          - The terminal's feed/inbox tailer routing, when an incoming
            line is recognized as a reply to this card or a DM from
            this card's peer.
          - This card's own _submit_reply, for optimistic local echo
            of outbound DMs and reply-posts.

        Payload shape: {author, ts, body} (subset of the post envelope).
        Anything else in the dict is ignored — replies are deliberately
        plain-text for now.

        Dedup: when WE publish a reply, both the optimistic local echo
        (via _submit_reply) AND the gossip echo (via _on_peribus_feed_line
        → _route_to_thread_card if own-echo suppression misses) can land
        here. We keep a small set of recent reply signatures to drop
        the redundant second arrival. The signature is intentionally
        coarse — body text alone — because the local echo uses
        author='(you)' while the gossip echo uses our real NodeID, and
        their ts values differ by a fraction of a second.
        """
        author = (payload.get("author") or "?").strip()
        body = (payload.get("body") or "").strip()
        ts = payload.get("ts")
        if not body:
            return

        # Card-level dedup. Body-only signature handles both kinds of
        # double-arrival (local echo + gossip echo, or tailer-reopen
        # replay). Bounded to a handful of recent replies; the
        # author=(you) vs real-nodeid mismatch makes a stricter
        # signature unhelpful here.
        if not hasattr(self, "_seen_reply_bodies"):
            self._seen_reply_bodies = collections.deque(maxlen=32)
            self._seen_reply_set = set()
        if body in self._seen_reply_set:
            return
        if len(self._seen_reply_bodies) == self._seen_reply_bodies.maxlen:
            evicted = self._seen_reply_bodies[0]
            self._seen_reply_set.discard(evicted)
        self._seen_reply_bodies.append(body)
        self._seen_reply_set.add(body)

        # Build a compact reply row: byline (author + relative time)
        # above body, with a small left-border accent matching the
        # palette so threading is visually obvious.
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(6, 2, 0, 2)
        row_layout.setSpacing(0)

        # Border-left accent — pink on DM, neutral on post threads.
        if self._is_inbox_card:
            row.setStyleSheet(
                "QWidget { background: transparent; "
                "border-left: 2px solid rgba(220, 150, 180, 200); }"
            )
        else:
            row.setStyleSheet(
                "QWidget { background: transparent; "
                "border-left: 2px solid rgba(180, 195, 220, 200); }"
            )

        # Byline. "(you)" for our optimistic echo; otherwise short nodeid.
        if author == "(you)":
            byline_text = "you · just now"
        else:
            short = self._short_nodeid(author)
            when = self._format_relative_ts(ts) if ts else "just now"
            byline_text = f"@{short} · {when}"
        byline = QLabel(byline_text)
        byline.setStyleSheet(
            f"color: {self._reply_palette['byline_color']}; "
            f"font-size: 10px; font-weight: 600; padding: 0 8px;"
        )
        row_layout.addWidget(byline)

        body_lbl = QLabel(body)
        body_lbl.setWordWrap(True)
        body_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body_lbl.setStyleSheet(
            f"color: {self._reply_palette['body_color']}; "
            f"font-size: 12px; padding: 1px 8px 2px 8px;"
        )
        row_layout.addWidget(body_lbl)

        self._thread_layout.addWidget(row)
        self._thread_container.setVisible(True)

    @staticmethod
    def _format_relative_ts(ts) -> str:
        """Cheap "5m ago" formatter for thread bylines."""
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            return "just now"
        delta = max(0, time.time() - ts)
        if delta < 45:
            return "just now"
        if delta < 90:
            return "1m ago"
        if delta < 3600:
            return f"{int(delta / 60)}m ago"
        if delta < 86400:
            return f"{int(delta / 3600)}h ago"
        return f"{int(delta / 86400)}d ago"

    # Maximum nesting depth for embedded media. Mostly defensive — keeps a
    # bug or malicious producer from infinite-nesting posts-in-posts.
    _POST_MEDIA_MAX_DEPTH = 3

    def _render_post_media(self, layout, media_list):
        """Embed inline-widget payloads inside a post card."""
        # Track depth via a payload-side counter so nested quote-posts can
        # carry their own media without escaping the cap.
        depth = int(self._payload.get("_nest_depth", 0))
        if depth >= self._POST_MEDIA_MAX_DEPTH:
            warn = QLabel("⚠ media nesting too deep — truncated")
            warn.setStyleSheet("color: #cc8844; font-size: 10px;")
            layout.addWidget(warn)
            return

        for entry in media_list[:8]:  # cap fan-out per post
            if not isinstance(entry, dict):
                continue
            # Stamp depth on the child so its own nested media respect
            # the cap. We mutate a shallow copy to avoid touching the
            # caller's dict.
            child_payload = dict(entry)
            child_payload["_nest_depth"] = depth + 1
            try:
                child = InlineMediaWidget(
                    child_payload,
                    dark_mode=self._dark_mode,
                    max_width=max(self._max_w - 16, 240),
                    host_terminal=self._host_terminal_hint,
                )
            except Exception as e:
                err = QLabel(f"⚠ media render failed: {e}")
                err.setWordWrap(True)
                err.setStyleSheet("color: #cc4444; font-size: 11px;")
                layout.addWidget(err)
                continue
            # Add a small left indent so embedded media reads as nested
            # within the post card rather than as a sibling.
            wrap = QWidget()
            wrap_layout = QHBoxLayout(wrap)
            wrap_layout.setContentsMargins(8, 2, 0, 2)
            wrap_layout.setSpacing(0)
            wrap_layout.addWidget(child)
            layout.addWidget(wrap)
            # The nested widget is owned by `wrap` -> body layout, so Qt
            # parent ownership keeps it alive; no need to pin in
            # _keepalive (that's reserved for non-widget resources like
            # media decoders that have no Qt parent).

    # Where to look for attachment bytes on the local mount. The peribus
    # filesystem keeps content-addressed objects under each peer's
    # social/ directory (the `nodes/<author>/social/<hash>` shape, see
    # filesystem.py). We try a few candidate paths in order so this
    # works whether the daemon stores blobs under social/, attachments/,
    # or some other directory; the first one that yields bytes wins.
    #
    # Tuples are (template, needs_author). The {root} placeholder holds
    # the peribus mount (default /n/peribus); {hash} holds the raw
    # attachment id (which may itself include the "b3:" prefix); when
    # needs_author is true the path also includes {author}.
    _ATTACHMENT_PATH_TEMPLATES = (
        ("{root}/nodes/{author}/social/{hash}", True),
        ("{root}/attachments/{hash}",           False),
        ("{root}/cas/{hash}",                   False),
        ("{root}/blobs/{hash}",                 False),
    )

    # Magic-bytes → renderer kind. Used when an attachment has no
    # extension on disk (content-addressed names usually don't).
    _ATTACHMENT_MAGIC = (
        (b"\x89PNG\r\n\x1a\n",       "image", "png"),
        (b"GIF87a",                   "gif",   "gif"),
        (b"GIF89a",                   "gif",   "gif"),
        (b"\xff\xd8\xff",             "image", "jpg"),
        (b"RIFF",                     "_riff", ""),     # disambiguated below
        (b"%PDF-",                    "pdf",   "pdf"),
        (b"\x00\x00\x00\x18ftyp",     "video", "mp4"),
        (b"\x00\x00\x00\x20ftyp",     "video", "mp4"),
        (b"\x1aE\xdf\xa3",            "video", "webm"),
        (b"ID3",                      "audio", "mp3"),
        (b"\xff\xfb",                 "audio", "mp3"),
        (b"OggS",                     "audio", "ogg"),
        (b"fLaC",                     "audio", "flac"),
        (b"<svg",                     "html",  "svg"),
        (b"<!DOCTYPE html",           "html",  "html"),
        (b"<html",                    "html",  "html"),
    )

    def _render_attachments(self, layout, attachments, author,
                            chip_bg, chip_fg,
                            media_hints=None) -> None:
        """
        Render each attachment inline if we can resolve its bytes on
        the local mount; otherwise fall back to a small chip with the
        truncated hash.

        Resolution strategy:
          1. Walk a list of candidate paths under /n/peribus, take the
             first that exists (the daemon does the MSG_FETCH round-
             trip for content not already cached locally).
          2. Determine kind+format. If `media_hints` was provided,
             prefer the publisher's declared kind for the matching
             stub (typically positionally aligned with attachments[i],
             which is how /share emits them). Otherwise sniff from
             magic bytes / extension. The hint matters most for text
             kinds (python, html): blobs are named by hash on the
             receiver, so there's no extension to go on, and Python
             source has no magic prefix — the hint is the only way to
             know it's intended as Python rather than a generic info
             card.
          3. Build an InlineMediaWidget payload pointing at the
             resolved path so the existing renderers (image, gif,
             video, audio, pdf, model3d) handle decoding. We
             deliberately pass `path` rather than `data` for binary
             kinds so we don't load the whole file into memory — let
             Qt's lazy media decoders do their thing.

        Anything we can't resolve gets the chip-fallback treatment so
        the user sees that something is referenced even if we can't
        fetch it.
        """
        # Build a kind+filename hint table from media_hints. Pair by
        # index where possible (publisher's /share emits one stripped
        # stub per attachment, in order). Keyed by the index into
        # `attachments` for now; if the schema later supports explicit
        # hash↔stub binding we can switch on that.
        hints_by_index = {}
        if isinstance(media_hints, list):
            stripped_hints = [
                m for m in media_hints
                if isinstance(m, dict) and m.get("_stripped")
            ]
            for i, hint in enumerate(stripped_hints):
                hints_by_index[i] = hint

        # Cap fan-out, same shape as _render_post_media.
        rendered_any = False
        unresolved = []
        for i, att in enumerate(attachments[:8]):
            resolved = self._resolve_attachment_path(att, author)
            if resolved is None:
                unresolved.append(att)
                continue
            # Skip attachments whose bytes are the post envelope itself.
            # The peribus daemon content-addresses everything written to
            # share/<name>, including our own JSON envelope — which means
            # post.attachments often points at the envelope rather than at
            # any standalone file. Without this filter every share would
            # render as a duplicate "info" card showing its own JSON.
            #
            # The check is cheap: only the first 64 bytes need to look
            # like a post envelope start. False positives would require a
            # real file whose first 16 chars are exactly `{"type":"post"`
            # — unlikely enough to ignore.
            if self._attachment_looks_like_envelope(resolved):
                # Drop silently — no chip either. The envelope is the
                # post we're already rendering; flagging it would just
                # add noise. Track it so we know none of the resolved
                # attachments turned out to be media.
                continue

            # Use publisher hint when present, then fall back to sniff.
            # We trust the hint for `kind` (publisher knew what they
            # shared) but always re-derive `format` from the file
            # because the hint's format may have been the source ext
            # while the bytes are something else after any transformation.
            #
            # Caveat: a hint of `type=info` is treated as soft — if the
            # `format` field points at a real renderer, we promote the
            # kind. Older daemons emitted "info" for every attachment
            # regardless of file type (a since-fixed bug); without this
            # promotion, a GIF or .py from one of those daemons would
            # render as a plain-text "info" card. Even on current
            # daemons, "info" is the explicit fallback for unrecognized
            # extensions, and if we *can* recognize the format, that's
            # better than the fallback.
            hint = hints_by_index.get(i)
            kind = None
            fmt = None
            hint_filename = None
            if hint is not None:
                hint_kind = hint.get("type")
                hint_fmt = hint.get("format")
                if hint_kind in (
                    "image", "gif", "audio", "video", "pdf", "model3d",
                    "python", "html",
                ):
                    # Strong hint — publisher told us exactly what kind.
                    kind = hint_kind
                    fmt = hint_fmt or hint_kind
                elif hint_kind == "info" and hint_fmt:
                    # Soft hint — try to promote based on format. Mirrors
                    # the daemon's _MEDIA_EXT_KIND table.
                    promoted = self._kind_from_format_hint(hint_fmt)
                    if promoted is not None:
                        kind = promoted
                        fmt = hint_fmt
                    else:
                        # Format isn't recognized either — keep info.
                        kind = "info"
                        fmt = hint_fmt
                hint_filename = hint.get("filename")
            if kind is None:
                kind, fmt = self._sniff_attachment_kind(resolved)
            if kind is None:
                unresolved.append(att)
                continue
            child_payload = {
                "type": kind,
                "format": fmt,
                "filename": hint_filename or os.path.basename(resolved) or att,
            }
            # Renderer contracts differ for binary vs. text kinds. Image,
            # gif, audio, video, pdf, model3d all accept a `path` and do
            # their own lazy loading. python/html/info, in contrast,
            # expect inline strings (code / content / text) — passing a
            # path to them silently does nothing useful. So we slurp text
            # kinds into memory here. They're capped by sniff anyway.
            if kind in ("python", "html", "info"):
                try:
                    with open(resolved, "rb") as f:
                        raw = f.read()
                    text = raw.decode("utf-8", errors="replace")
                except OSError:
                    unresolved.append(att)
                    continue
                if kind == "python":
                    child_payload["code"] = text
                    # Python payloads that arrive as post attachments came
                    # in over peribus — content-addressed from some peer's
                    # social/ directory. We don't trust them enough to
                    # auto-exec. The flag is what _render_python checks
                    # to decide between auto-render and the manual Run
                    # gate; direct writes to $term/inline (which never
                    # pass through here) stay unflagged and run on receipt
                    # like they always have.
                    child_payload["_unsafe_origin"] = "peribus"
                elif kind == "html":
                    child_payload["content"] = text
                else:  # info / plain text
                    # Truncate so a giant log doesn't blow up the card.
                    if len(text) > 4000:
                        text = text[:4000] + "\n…(truncated)"
                    child_payload["text"] = (
                        f"{child_payload['filename']}\n\n{text}"
                    )
            else:
                child_payload["path"] = resolved
            try:
                child = InlineMediaWidget(
                    child_payload,
                    dark_mode=self._dark_mode,
                    max_width=max(self._max_w - 16, 240),
                    host_terminal=self._host_terminal_hint,
                )
            except Exception:
                unresolved.append(att)
                continue
            wrap = QWidget()
            wrap_layout = QHBoxLayout(wrap)
            wrap_layout.setContentsMargins(8, 2, 0, 2)
            wrap_layout.setSpacing(0)
            wrap_layout.addWidget(child)
            layout.addWidget(wrap)
            rendered_any = True

        # Anything we couldn't resolve gets the old chip treatment, so
        # the user at least sees that something is referenced even if we
        # can't fetch it yet.
        if unresolved:
            chip_row = QWidget()
            chip_layout = QHBoxLayout(chip_row)
            chip_layout.setContentsMargins(0, 2, 0, 2)
            chip_layout.setSpacing(4)
            for att in unresolved[:6]:
                short = att if len(att) <= 12 else att[:6] + '…' + att[-4:]
                chip = QLabel(f"📎 {short}")
                chip.setToolTip(
                    "attachment not yet available locally — "
                    "the daemon may still be fetching it"
                )
                chip.setStyleSheet(
                    f"background: {chip_bg}; color: {chip_fg}; "
                    f"font-size: 10px; padding: 2px 6px; border-radius: 3px;"
                )
                chip_layout.addWidget(chip)
            chip_layout.addStretch(1)
            layout.addWidget(chip_row)

    def _resolve_attachment_path(self, attachment: str,
                                 author: str) -> Optional[str]:
        """Walk candidate paths and return the first that exists."""
        terminal = self._find_host_terminal()
        root = "/n/peribus"
        if terminal is not None:
            root = getattr(terminal, "_peribus_root", root) or root
        for template, needs_author in self._ATTACHMENT_PATH_TEMPLATES:
            if needs_author and (not author or author in ("?", "(local)")):
                continue
            candidate = template.format(
                root=root, author=author, hash=attachment,
            )
            try:
                if os.path.isfile(candidate):
                    return candidate
            except OSError:
                # Some FUSE quirks raise on stat for non-existent paths
                # rather than returning False. Treat as miss and continue.
                continue
        return None

    def _attachment_looks_like_envelope(self, path: str) -> bool:
        """
        Heuristic: do the first ~64 bytes of this attachment look like
        a post envelope (i.e. our own JSON wrapper, not real media)?

        The peribus daemon hashes whatever bytes get written to share/,
        which means post.attachments often references the JSON envelope
        we wrote — not any file we intended to attach. Rendering that
        would surface the envelope's JSON to the user as an inline
        "info" card, duplicating the card we're already in.

        We check for the literal `{"type":"post"` opening with optional
        leading whitespace. Cheap, narrow, and almost impossible to
        false-positive on real media.
        """
        try:
            with open(path, "rb") as f:
                head = f.read(64)
        except OSError:
            return False
        try:
            text_head = head.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
        stripped = text_head.lstrip()
        # Match both '{"type":"post"' and '{ "type": "post"' (with spaces).
        if not stripped.startswith("{"):
            return False
        # Drop the opening brace and any whitespace, then check for
        # "type":"post". Keeps the check loose enough for variant
        # encoders without matching arbitrary JSON.
        rest = stripped[1:].lstrip()
        if rest.startswith('"type"'):
            tail = rest[6:].lstrip()
            if tail.startswith(":"):
                tail = tail[1:].lstrip()
                if tail.startswith('"post"'):
                    return True
        return False

    def _kind_from_format_hint(self, fmt: str):
        """
        Promote a publisher's `format` field to a renderer kind.

        Used as a fallback when the publisher's `type` hint is the
        generic "info" but the format is specific enough to dispatch.
        Mirrors _SHARE_EXT_KIND on the publish side and the ext_map
        in _sniff_attachment_kind. Returns None for unrecognized
        formats so callers can fall through to magic-byte sniffing.
        """
        if not fmt or not isinstance(fmt, str):
            return None
        f = fmt.lower().lstrip(".")
        # Same shape as _sniff_attachment_kind's ext_map but only
        # returns the kind, not the (kind, format) pair. Kept inline
        # rather than importing/sharing because the two tables exist
        # in different scopes and the cost of duplication is low.
        promotion = {
            "png": "image", "jpg": "image", "jpeg": "image",
            "webp": "image", "bmp": "image",
            "gif": "gif",
            "mp3": "audio", "wav": "audio", "ogg": "audio",
            "flac": "audio", "m4a": "audio",
            "mp4": "video", "mkv": "video", "webm": "video",
            "mov": "video", "avi": "video",
            "pdf": "pdf",
            "obj": "model3d", "stl": "model3d", "glb": "model3d",
            "gltf": "model3d", "ply": "model3d",
            "py": "python",
            "html": "html", "htm": "html", "svg": "html",
        }
        return promotion.get(f)

    def _sniff_attachment_kind(self, path: str):
        """
        Identify (kind, format) from magic bytes, falling back to
        extension. Returns (None, None) if we can't classify it.
        """
        # Try extension first — cheap and usually right.
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        ext_map = {
            "png": ("image", "png"), "jpg": ("image", "jpg"),
            "jpeg": ("image", "jpg"), "webp": ("image", "webp"),
            "gif": ("gif", "gif"),
            "mp3": ("audio", "mp3"), "wav": ("audio", "wav"),
            "ogg": ("audio", "ogg"), "flac": ("audio", "flac"),
            "m4a": ("audio", "m4a"),
            "mp4": ("video", "mp4"), "mkv": ("video", "mkv"),
            "webm": ("video", "webm"), "mov": ("video", "mov"),
            "pdf": ("pdf", "pdf"),
            "obj": ("model3d", "obj"), "stl": ("model3d", "stl"),
            "glb": ("model3d", "glb"), "gltf": ("model3d", "gltf"),
            "py": ("python", "py"),
            "html": ("html", "html"), "svg": ("html", "svg"),
        }
        if ext in ext_map:
            return ext_map[ext]
        # No usable extension — peek magic bytes.
        try:
            with open(path, "rb") as f:
                head = f.read(64)
        except OSError:
            return (None, None)
        for prefix, kind, fmt in self._ATTACHMENT_MAGIC:
            if head.startswith(prefix):
                if kind == "_riff":
                    # RIFF can be WAV (audio) or AVI (video) or WEBP
                    # (image). Disambiguate from the form type at offset 8.
                    form = head[8:12]
                    if form == b"WAVE":
                        return ("audio", "wav")
                    if form == b"AVI ":
                        return ("video", "avi")
                    if form == b"WEBP":
                        return ("image", "webp")
                    return (None, None)
                return (kind, fmt)
        # Looks textual? Try decoding head as utf-8 — if it works, render
        # it as info so at least the user sees the contents.
        try:
            head.decode("utf-8")
            return ("info", "txt")
        except UnicodeDecodeError:
            return (None, None)

    def _refresh_post_byline(self) -> None:
        """Recompute the byline text from current state. Called both at
        construction time and by the host terminal's periodic tick."""
        if not hasattr(self, "_post_byline_label"):
            return
        lbl = self._post_byline_label
        try:
            still_alive = lbl is not None
            # Cheap liveness probe — Qt deletes the underlying C++ object
            # before the Python wrapper, so accessing a property of a dead
            # QLabel raises RuntimeError.
            _ = lbl.text() if still_alive else None
        except RuntimeError:
            return
        bits = list(self._post_byline_static_bits)
        if self._post_ts is not None:
            rel = self._format_relative_ts(self._post_ts)
            if rel:
                # Insert the time right after the @author so it reads
                # naturally: @alice · 3m ago · resonance 0.42
                bits.insert(1, rel)
        try:
            lbl.setText("  ·  ".join(bits))
        except RuntimeError:
            pass

    def _format_relative_ts(self, ts: float) -> str:
        """Render a unix timestamp as 'just now' / '3m' / '1h' / '2d'."""
        try:
            delta = max(0.0, time.time() - ts)
        except Exception:
            return ""
        if delta < 30:
            return "just now"
        if delta < 3600:
            return f"{int(delta // 60)}m ago"
        if delta < 86400:
            return f"{int(delta // 3600)}h ago"
        if delta < 86400 * 7:
            return f"{int(delta // 86400)}d ago"
        # Fall back to a date for older posts.
        try:
            return time.strftime("%Y-%m-%d", time.localtime(ts))
        except Exception:
            return ""

    def _render_info(self, layout, payload, level: str):
        text = payload.get("text") or payload.get("message") or ""
        colors = {
            "info":  ("#5588cc", "ℹ"),
            "warn":  ("#cc8844", "⚠"),
            "warning": ("#cc8844", "⚠"),
            "error": ("#cc4444", "✗"),
            "ok":    ("#44aa55", "✓"),
            "success": ("#44aa55", "✓"),
        }
        color, icon = colors.get(level, ("#888888", "•"))
        label = QLabel(f"{icon}  {text}")
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")
        layout.addWidget(label)
    
    def closeEvent(self, event):
        """Tear down media players cleanly so NVDEC etc. release."""
        for item in self._keepalive:
            try:
                if isinstance(item, tuple) and item[0] == 'tempfile':
                    try:
                        os.unlink(item[1])
                    except Exception:
                        pass
                elif hasattr(item, 'stop'):
                    item.stop()
            except Exception:
                pass
        self._keepalive.clear()
        super().closeEvent(event)


class StreamingFenceParser:
    """Pure-state-machine parser that splits a chunked text stream into:
    
        - plain text (outside any fence)
        - fence open events (with detected machine/language tag)
        - fence body chunks (text inside an open fence)
        - fence close events
    
    The point of isolating this from the rendering layer is concurrency:
    each producer (one connected agent, one FS writer fid, etc.) gets
    its own parser instance, so they cannot corrupt each other's state
    even if both produce fenced output simultaneously.
    
    Fence syntax recognized: ```name\\n ... \\n``` — the name is captured
    as the "machine" identifier. Fences without a name are still emitted
    (machine="") so they stay visible as inline widgets without a Run target.
    
    The parser tolerates fences split across chunks (a ```` arriving in
    pieces, language tag continuing in the next chunk, etc.). It emits
    events through a callback object with on_text / on_fence_open /
    on_fence_chunk / on_fence_close hooks.
    """
    
    # Maximum length of language identifier before we give up and treat
    # the buffered chars as code content (model didn't put a newline).
    _LANG_LIMIT = 32
    
    def __init__(self, sink):
        """sink: object with on_text(str), on_fence_open(name),
                 on_fence_chunk(str), on_fence_close() methods."""
        self.sink = sink
        # State
        self._in_fence = False
        self._waiting_for_lang = False
        self._lang_buf = ""
        # Pending characters for fence-marker detection across chunk
        # boundaries (we may have seen 1 or 2 backticks at the end of
        # a chunk; they're held until we know whether ``` follows).
        self._pending = ""  # never longer than 2 chars
    
    def feed(self, chunk: str):
        """Process a streaming chunk and emit events to the sink."""
        if not chunk:
            return

        # Fast path: not in a fence, no pending backticks, no backtick
        # anywhere in the new chunk. This is the dominant case during
        # ordinary LLM token streaming (chunks of natural-language text
        # arriving rapidly). The per-char state machine below is correct
        # but its Python loop dominates parser CPU for long streams.
        # The check itself is O(n) but in C and extremely cache-friendly.
        if (not self._in_fence
                and not self._waiting_for_lang
                and not self._pending
                and '`' not in chunk):
            self.sink.on_text(chunk)
            return

        # Fast path: already inside a fence, past the language line,
        # no pending and no backtick anywhere. Common during code-block
        # streaming where the model emits ~1 KB of code character by
        # character. Emit the whole chunk as a single fence_chunk event.
        if (self._in_fence
                and not self._waiting_for_lang
                and not self._pending
                and '`' not in chunk):
            self.sink.on_fence_chunk(chunk)
            return

        # Combine any pending chars with the new chunk; we'll re-emit
        # whatever we don't consume.
        s = self._pending + chunk
        self._pending = ""
        i = 0
        n = len(s)
        # Buffer of plain (non-fence) text we'll emit in one batch.
        plain_buf = []
        # Buffer of in-fence text we'll emit in one batch.
        fence_buf = []
        
        def flush_plain():
            if plain_buf:
                self.sink.on_text("".join(plain_buf))
                plain_buf.clear()
        
        def flush_fence():
            if fence_buf:
                self.sink.on_fence_chunk("".join(fence_buf))
                fence_buf.clear()
        
        while i < n:
            # --- Inside a fence, waiting for the language line --------------
            if self._in_fence and self._waiting_for_lang:
                ch = s[i]
                if ch == '\n':
                    # Language line complete.
                    name = self._lang_buf.strip()
                    self._lang_buf = ""
                    self._waiting_for_lang = False
                    self.sink.on_fence_open(name)
                    i += 1
                    continue
                # Buffer until we hit a newline or overflow.
                self._lang_buf += ch
                if len(self._lang_buf) > self._LANG_LIMIT:
                    # Overflow: treat what we have as actual code content,
                    # with no machine tag. Open the fence with empty name.
                    overflow = self._lang_buf
                    self._lang_buf = ""
                    self._waiting_for_lang = False
                    self.sink.on_fence_open("")
                    fence_buf.append(overflow)
                i += 1
                continue
            
            # --- Look for ``` at current position --------------------------
            if s[i] == '`':
                # We need at least 3 chars to confirm a fence marker.
                if i + 3 <= n:
                    if s[i:i+3] == '```':
                        # Confirmed fence marker.
                        if self._in_fence:
                            # Closing fence.
                            flush_fence()
                            self._in_fence = False
                            self.sink.on_fence_close()
                        else:
                            # Opening fence — flush any plain text we have.
                            flush_plain()
                            self._in_fence = True
                            self._waiting_for_lang = True
                            self._lang_buf = ""
                        i += 3
                        # Skip a single trailing newline after a closing
                        # fence to keep output tidy.
                        if not self._in_fence and i < n and s[i] == '\n':
                            i += 1
                        continue
                    else:
                        # Definitely not a fence — emit this backtick.
                        if self._in_fence:
                            fence_buf.append(s[i])
                        else:
                            plain_buf.append(s[i])
                        i += 1
                        continue
                else:
                    # Not enough chars to decide; stash the trailing 1 or 2
                    # backticks for the next feed() call.
                    self._pending = s[i:]
                    break
            
            # --- Ordinary character ----------------------------------------
            if self._in_fence:
                fence_buf.append(s[i])
            else:
                plain_buf.append(s[i])
            i += 1
        
        flush_plain()
        flush_fence()
    
    def flush(self):
        """Force-emit any pending characters as literal text. Called when
        the source ends mid-stream so we don't lose dangling backticks."""
        if self._pending:
            if self._in_fence:
                self.sink.on_fence_chunk(self._pending)
            else:
                self.sink.on_text(self._pending)
            self._pending = ""
        if self._in_fence:
            # Source ended while a fence was still open. Close it so the
            # widget gets finalized. The buffered language (if we never
            # saw \n) becomes the fence name.
            if self._waiting_for_lang:
                # Promote the buffered identifier to the fence name even
                # though we never saw a newline.
                name = self._lang_buf.strip()
                self._lang_buf = ""
                self._waiting_for_lang = False
                self.sink.on_fence_open(name)
            self._in_fence = False
            self.sink.on_fence_close()


class _SourceState:
    """Per-source rendering state inside the TerminalStreamRouter.
    
    Each source (one connected agent / one FS-writer fid / etc.) owns:
      - its own StreamingFenceParser
      - its own "current text display" QTextEdit, where plain text from
        this source is appended
      - its own current InlineCodeBlockWidget (if a fence is currently open)
      - a default text color
    
    When a fence opens for source A, source A finalizes its text display
    (so newer text from source A creates a new one *after* the widget),
    but source B's text display is untouched. Their layout positions stay
    in chronological insert order. This is the property that makes
    concurrent streams safe: fence state is per-source, layout writes
    are serialized by Qt's main thread, and each source threads its own
    text into its own most-recent text segment.
    """
    
    def __init__(self, color: str):
        self.color = color
        self.parser: StreamingFenceParser = None  # set by router
        self.text_display: QTextEdit = None       # current QTextEdit owned by this source
        self.code_widget: InlineCodeBlockWidget = None  # current open fence widget, if any


class TerminalStreamRouter(QObject):
    """Routes streaming text from multiple sources into the terminal,
    detecting ``` machine ``` fences and morphing them into inline widgets.
    
    The router is the single owner of the terminal_content_layout for
    "stream-driven" widgets. It does not interfere with widgets inserted
    by other paths (Acme, scene panel, /clear, etc.) — those still use
    terminal_content_layout directly.
    
    Concurrency:
        - All public methods must be called on the Qt main thread.
        - Each source has independent fence state, so two concurrent
          producers cannot corrupt each other's parsing.
    """
    
    def __init__(self, terminal: 'TerminalWidget'):
        super().__init__(terminal)
        self.terminal = terminal
        self._sources: Dict[str, _SourceState] = {}
        # Scroll coalescing is owned by the terminal — see
        # TerminalWidget._request_scroll_coalesced. The router used to
        # own its own _scroll_timer; centralising it on the terminal
        # means direct append_text / PTY output paths share state with
        # the router, so a burst that hits both paths produces one
        # scroll, not two.

    def _request_scroll(self):
        self.terminal._request_scroll_coalesced()
    
    # ------------------------------------------------------------------
    # Source lifecycle
    # ------------------------------------------------------------------
    
    def _ensure_source(self, source_key: str, color: str = None) -> _SourceState:
        st = self._sources.get(source_key)
        if st is not None:
            if color is not None:
                st.color = color
            return st
        st = _SourceState(color or self.terminal.C_AGENT)
        # The first text display this source claims is the terminal's
        # current_text_display — meaning early plain text continues to
        # land where it would have without the router. Once a fence
        # widget gets inserted, the source's text display advances.
        st.text_display = self.terminal.current_text_display
        st.parser = StreamingFenceParser(_RouterSink(self, source_key))
        self._sources[source_key] = st
        return st
    
    def reset_source(self, source_key: str):
        """Drop a source's parser state. Called e.g. on agent disconnect.
        
        Any open fence is force-closed first so we don't lose the widget.
        """
        st = self._sources.pop(source_key, None)
        if st is None:
            return
        try:
            st.parser.flush()
        except Exception:
            pass
        # Finalize a still-streaming widget so the user can still Run it.
        if st.code_widget is not None and st.code_widget.is_streaming:
            st.code_widget.finalize_streaming()
            st.code_widget = None
    
    def reset_all(self):
        """Drop all source state. Called by clear_output()."""
        for key in list(self._sources.keys()):
            self.reset_source(key)
        self._sources.clear()
    
    # ------------------------------------------------------------------
    # Main feed entry point
    # ------------------------------------------------------------------
    
    def feed(self, source_key: str, text: str, color: str = None):
        """Feed a chunk of text from one source through the parser.
        
        Must be called on the Qt main thread.
        """
        if not text:
            return
        st = self._ensure_source(source_key, color)
        st.parser.feed(text)
        # Coalesced — see __init__.
        self._request_scroll()
    
    def end_of_stream(self, source_key: str):
        """Source has finished emitting. Finalize any open widget."""
        st = self._sources.get(source_key)
        if st is None:
            return
        st.parser.flush()
        if st.code_widget is not None and st.code_widget.is_streaming:
            st.code_widget.finalize_streaming()
            st.code_widget = None
    
    # ------------------------------------------------------------------
    # Internal: applied by _RouterSink
    # ------------------------------------------------------------------
    
    def _emit_text(self, source_key: str, text: str):
        """Append plain text from a source to its current text display."""
        st = self._sources[source_key]
        # If this source's text display has been "released" (because a
        # widget was inserted after it for this source), grow a fresh one.
        if st.text_display is None:
            st.text_display = self._make_text_display_for(source_key)
        self._append_to(st.text_display, text, st.color)
    
    def _emit_fence_open(self, source_key: str, machine: str):
        """Insert an InlineCodeBlockWidget after this source's text display."""
        st = self._sources[source_key]
        # Freeze current text display so it doesn't grow vertically and
        # leave a gap below the new widget. (Display follows its document
        # via _adjust_height; this just locks in the current size.)
        if st.text_display is not None:
            self.terminal._adjust_height(st.text_display)
        # Also freeze the terminal-wide default display, in case it
        # diverged from this source's display (e.g. a different source
        # advanced it earlier).
        ctd = self.terminal.current_text_display
        if ctd is not None and ctd is not st.text_display:
            self.terminal._adjust_height(ctd)
        
        widget = InlineCodeBlockWidget(
            machine_name=machine,
            llmfs_mount=self.terminal._inline_widget_mount(),
            dark_mode=getattr(self.terminal, '_is_dark_mode', False),
            max_width=self.terminal._inline_max_width(),
            host_terminal=self.terminal,
        )
        st.code_widget = widget
        self._append_widget(widget)
        # The source's "current text display" is conceptually now after
        # the new widget; we'll create a fresh one lazily on the next
        # plain text emission.
        st.text_display = None
        # CRITICAL: also advance the terminal-wide default display so
        # that any direct append_text() call (errors, shell, info,
        # disconnect messages...) lands BELOW the inline widget rather
        # than into the now-frozen display sitting above it. Without
        # this, racing non-router writes appear "before" the widget
        # even though the agent stream is correctly routed.
        self.terminal._advance_default_text_display()
    
    def _emit_fence_chunk(self, source_key: str, text: str):
        """Stream text into the source's currently-open code widget."""
        st = self._sources[source_key]
        if st.code_widget is None:
            # Defensive: parser said we're in a fence but we have no widget.
            # Fall back to plain text.
            self._emit_text(source_key, text)
            return
        st.code_widget.append_code(text)
    
    def _emit_fence_close(self, source_key: str):
        """Finalize the source's open code widget.
        
        The source's text_display is left as None so the next plain text
        emission will lazily create a fresh display below the widget.
        The terminal-wide default display was already advanced at fence
        open time, so non-router writes are also pointed correctly.
        """
        st = self._sources[source_key]
        if st.code_widget is not None:
            st.code_widget.finalize_streaming()
            st.code_widget = None
    
    # ------------------------------------------------------------------
    # Inline media (out-of-band, no fence parser involvement)
    # ------------------------------------------------------------------
    
    def insert_media(self, payload: dict):
        """Insert a media widget into the terminal. Used by $term/inline.
        
        This does NOT touch any source's parser state, so it's safe to
        invoke at any time, even while multiple agents are mid-stream.
        """
        # Freeze all live text displays so the new widget sits below them.
        for st in self._sources.values():
            if st.text_display is not None:
                self.terminal._adjust_height(st.text_display)
        # Also freeze the terminal's "default" current_text_display — the
        # one used by direct append_text() calls (errors, shell, etc.).
        ctd = self.terminal.current_text_display
        if ctd is not None:
            self.terminal._adjust_height(ctd)
        
        widget = InlineMediaWidget(
            payload,
            dark_mode=getattr(self.terminal, '_is_dark_mode', False),
            max_width=self.terminal._inline_max_width(),
            host_terminal=self.terminal,
        )
        self._append_widget(widget)
        # Force any subsequent plain text from any source to start a new
        # text display below the media widget.
        for st in self._sources.values():
            st.text_display = None
        # The terminal's own default current_text_display is replaced too.
        self.terminal._advance_default_text_display()
    
    # ------------------------------------------------------------------
    # Helpers — direct manipulation of terminal_content_layout
    # ------------------------------------------------------------------
    
    def _append_widget(self, widget: QWidget):
        """Append a widget to the terminal's content layout."""
        self.terminal.terminal_content_layout.addWidget(widget)
    
    def _make_text_display_for(self, source_key: str) -> QTextEdit:
        """Create a fresh text display for a source and append it."""
        te = self.terminal._create_text_display()
        self.terminal.terminal_content_layout.addWidget(te)
        self.terminal.text_displays.append(te)
        # The terminal's own default current_text_display follows along
        # so subsequent plain append_text() calls land below the widget too.
        self.terminal.current_text_display = te
        return te
    
    def _append_to(self, text_display: QTextEdit, text: str, color: str):
        """Append text into a specific QTextEdit with the given color."""
        # Mirror append_text's behavior but targeted at a specific display.
        adjusted = self.terminal._dm_adjust_color(color)
        cursor = text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(self.terminal._parse_rgba(adjusted))
        cursor.insertText(text, fmt)
        # Don't move the visible cursor in *other* displays — this avoids
        # focus-stealing when concurrent sources are active.


class _RouterSink:
    """Tiny adapter: StreamingFenceParser callbacks → TerminalStreamRouter
    methods bound to a particular source_key.
    
    Plain object (not QObject) — no signals, just direct method calls
    on the Qt main thread."""
    
    __slots__ = ("router", "source_key")
    
    def __init__(self, router: TerminalStreamRouter, source_key: str):
        self.router = router
        self.source_key = source_key
    
    def on_text(self, text: str):
        self.router._emit_text(self.source_key, text)
    
    def on_fence_open(self, name: str):
        self.router._emit_fence_open(self.source_key, name)
    
    def on_fence_chunk(self, text: str):
        self.router._emit_fence_chunk(self.source_key, text)
    
    def on_fence_close(self):
        self.router._emit_fence_close(self.source_key)


# ---------------------------------------------------------------------------
# Terminal Scene Panel - per-terminal live UI surface
# ---------------------------------------------------------------------------

class TerminalScenePanel(QWidget):
    """
    A panel that hosts an isolated QGraphicsScene + QGraphicsView, plus
    its own SceneManager / Executor pair.

    Code written to /n/rioa/terms/<term_id>/parse runs in this panel's
    execution context, so ``graphics_scene``, ``graphics_view`` and
    ``main_window`` inside that code refer to the panel — not to the
    global app scene.

    The scene rect is kept in sync with the viewport on every resize, so
    the scene is always exactly the size of the panel — no scrollbars,
    no virtual canvas. Code that places items in scene coordinates will
    therefore use the same coordinate system as pixels in the panel.

    Versioning, snapshots and undo/redo are also per-panel: they operate
    on the panel's own SceneManager and don't touch the global one.
    """

    # Initial size used before the first resize event lands. Will be
    # overwritten as soon as the view gets a real geometry.
    INITIAL_WIDTH = 600
    INITIAL_HEIGHT = 600

    def __init__(self, term_id: str, parent=None):
        super().__init__(parent)
        self.term_id = term_id

        # Lazily imported to avoid a circular import with rio.scene at
        # module load time (terminal_widget is imported very early).
        from rio.scene import SceneManager
        from rio.parser import Executor, ExecutionContext

        # --- Qt scene + view ----------------------------------------------
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, self.INITIAL_WIDTH, self.INITIAL_HEIGHT)
        self._scene.setBackgroundBrush(QColor("#FAFAFA"))

        self._view = QGraphicsView(self._scene, self)
        self._view.setStyleSheet("""
            QGraphicsView {
                background-color: #FAFAFA;
                border: none;
            }
        """)
        # No scrollbars — the scene is always exactly the viewport size.
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Anchor everything to the top-left so items stay where they were
        # placed when the panel resizes (we don't want Qt to recenter).
        self._view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # Disable any auto-fit transforms; we manage the scene rect ourselves.
        self._view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self._view.setResizeAnchor(QGraphicsView.NoAnchor)

        # Watch for view resizes and reflow the scene rect to match.
        self._view.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view)

        # --- Per-panel scene manager + executor ---------------------------
        self.scene_manager = SceneManager(
            width=self.INITIAL_WIDTH,
            height=self.INITIAL_HEIGHT,
        )
        self.scene_manager.attach_qt(self._scene)

        self._context = ExecutionContext(
            scene_manager=self.scene_manager,
            main_window=self,            # the panel acts as "main window" here
            graphics_scene=self._scene,
            graphics_view=self._view,
        )
        self.executor = Executor(self._context)

        # Take an initial empty snapshot so version 0 exists.
        try:
            self.scene_manager.take_snapshot(label="initial", code="")
        except Exception as e:
            print(f"TerminalScenePanel: initial snapshot failed: {e}")

    # ------------------------------------------------------------------
    # Resize → keep scene exactly viewport-sized
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self._view and event.type() == QEvent.Resize:
            self._sync_scene_to_viewport()
        return super().eventFilter(obj, event)

    def _sync_scene_to_viewport(self):
        """Resize the QGraphicsScene to exactly match the viewport."""
        vp = self._view.viewport()
        if vp is None:
            return
        w = max(1, vp.width())
        h = max(1, vp.height())

        cur = self._scene.sceneRect()
        if int(cur.width()) == w and int(cur.height()) == h:
            return

        self._scene.setSceneRect(0, 0, w, h)
        # Keep the SceneManager's recorded canvas size in sync too — some
        # code reads scene_manager.width / .height (e.g. snapshot bookkeeping).
        self.scene_manager.width = w
        self.scene_manager.height = h

    # ------------------------------------------------------------------
    # Public surface used by the filesystem ParseFile
    # ------------------------------------------------------------------

    def get_scene(self) -> QGraphicsScene:
        return self._scene

    def get_view(self) -> QGraphicsView:
        return self._view


# ---------------------------------------------------------------------------
# Peribus feed tailer — bridges /n/peribus/feed/new into a terminal's $inline
# ---------------------------------------------------------------------------


class PeribusFeedTailer(QObject):
    """
    Tails ``/n/peribus/feed/new`` and emits one Qt signal per JSON line.

    The peribus feed file is a *blocking* 9P stream: a read returns whatever
    posts are buffered and then blocks until a new post arrives. That maps
    naturally to a dedicated thread doing line-buffered raw reads and posting
    each completed line back to the GUI thread via a queued signal.

    Why a thread instead of asyncio.to_thread (the path feed_bridge.py uses):
    the terminal does not have a long-lived asyncio loop owning its lifetime
    — most of its async work is fire-and-forget. A QThread tied to the
    object's lifetime gives us a clean stop story: signal the stop event,
    close the fd to break the blocking read, join.

    Same buffering caveat as feed_bridge.FeedTailer: open with buffering=0
    so Python's BufferedReader doesn't strand short reads.
    """

    line_received = Signal(str)   # JSON line, newline-stripped
    error = Signal(str)           # human-readable error message
    started = Signal()
    stopped = Signal()
    reconnected = Signal()        # fired after a recovery reopen

    # How long to wait between reopen attempts after a transient error.
    # Picked to be short enough that a working daemon feels responsive
    # but long enough that a thrashing failure doesn't pin a CPU core.
    _RECONNECT_DELAY = 1.0
    _RECONNECT_DELAY_MAX = 10.0

    def __init__(self, feed_path: str, parent=None):
        super().__init__(parent)
        self._feed_path = feed_path
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="peribus.feed.tailer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 1.0) -> None:
        self._stop_event.set()
        # Closing the fd unblocks any in-flight read.
        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        t = self._thread
        self._thread = None
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """
        Long-lived read loop. Survives transient I/O errors by closing the
        fd, waiting briefly, and reopening — the typical failure mode for
        an idle 9P-over-FUSE blocking read is EIO from FUSE timing out the
        in-flight request, not the daemon actually going away. The mount
        is still good; only the one fd is poisoned.

        We only exit the loop for:
          * stop() being called (clean shutdown)
          * the mountpoint disappearing (daemon really did go down — there's
            nothing useful we can do until the user reattaches)
        """
        if not os.path.exists(self._feed_path):
            self.error.emit(f"feed not mounted at {self._feed_path}")
            return

        # Initial open. If this fails, give up — the user just asked us to
        # start; emitting an error is better than spinning silently.
        try:
            self._fd = os.open(self._feed_path, os.O_RDONLY)
        except OSError as e:
            self.error.emit(f"cannot open {self._feed_path}: {e}")
            return

        self.started.emit()
        partial = b""
        backoff = self._RECONNECT_DELAY
        had_first_open = True

        try:
            while not self._stop_event.is_set():
                # Lazy reopen if a previous iteration tore the fd down.
                if self._fd is None:
                    if not os.path.exists(self._feed_path):
                        self.error.emit(
                            f"feed disappeared at {self._feed_path} "
                            "(daemon down? mount lost?) — stopping"
                        )
                        return
                    try:
                        self._fd = os.open(self._feed_path, os.O_RDONLY)
                    except OSError as e:
                        # Still mounted but won't open — back off and retry.
                        if self._stop_event.wait(backoff):
                            return
                        backoff = min(backoff * 2, self._RECONNECT_DELAY_MAX)
                        continue
                    # Successful reopen — reset backoff and tell the UI.
                    backoff = self._RECONNECT_DELAY
                    partial = b""
                    self.reconnected.emit()

                try:
                    chunk = os.read(self._fd, 8192)
                except OSError as e:
                    # Stop requested while we were blocked: clean exit. The
                    # close from stop() will surface here as EBADF.
                    if self._stop_event.is_set():
                        return
                    # Anything else (EIO from FUSE timing out the read,
                    # ECONNRESET if the daemon restarted, etc.) is treated
                    # as a transient. Drop the fd and let the loop reopen.
                    self.error.emit(f"feed read: {e}; reconnecting…")
                    self._safe_close_fd()
                    if self._stop_event.wait(backoff):
                        return
                    backoff = min(backoff * 2, self._RECONNECT_DELAY_MAX)
                    continue

                if not chunk:
                    # The daemon's FeedNewFile.read() returns b"" as an
                    # *application-level* keepalive every
                    # _BLOCKING_READ_KEEPALIVE_S seconds — see
                    # peribus._daemon.FeedNewFile.read. It's not EOF, it
                    # means "still alive, no posts right now." Reopening
                    # the fd here would be a bug: it allocates a fresh
                    # cursor on the daemon side, which starts at the
                    # beginning of the feed ring → every buffered post
                    # gets replayed → we render duplicate cards.
                    #
                    # So: empty read = no-op. Keep the same fd, loop
                    # back to read again. The fd only gets torn down on
                    # a real OSError above, where reopening makes sense.
                    continue

                # Successful read — reset the backoff window.
                backoff = self._RECONNECT_DELAY
                partial += chunk
                while b"\n" in partial:
                    line, partial = partial.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    self.line_received.emit(text)
        finally:
            self._safe_close_fd()
            self.stopped.emit()

    def _safe_close_fd(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Terminal Widget
# ---------------------------------------------------------------------------

class TerminalWidget(QWidget):
    """
    Enhanced terminal widget with full LLMFS filesystem integration.

    All agent interaction is mediated through the mounted 9P filesystem at
    ``llmfs_mount`` (default /n/llm).  The terminal never speaks to agent
    objects directly - it reads and writes ordinary files.

    Plan 9 Blocking Attachments:
      /attach now uses blocking I/O - spawns a background process that runs
      'cat $source > $destination' in a loop. The cat blocks until content
      is ready (thanks to asyncio.Event in SupplementaryOutputFile), then
      routes it. No polling!

    Signal contract:
      ``command_submitted`` is emitted ONLY for Python code (>>> prefix).
      The prefix is stripped before emission so the Rio executor receives
      clean Python.  All other command types (macros, shell, agent prompts)
      are handled internally and never reach the executor.
    """

    command_submitted = Signal(str)

    # Colour palette (defaults — overridden at runtime by active color scheme)
    C_DEFAULT  = "rgba(0, 0, 0, 255)"
    C_AGENT    = "rgba(0, 120, 60, 255)"      # agent output
    C_USER     = "rgba(0, 0, 0, 255)"         # user prompt echo
    C_INFO     = "rgba(100, 100, 100, 255)"    # informational
    C_SUCCESS  = "rgba(0, 0, 0, 255)" #"rgba(60, 140, 60, 255)"      # success messages
    C_ERROR    = "rgba(200, 50, 50, 255)"      # errors
    C_MACRO    = "rgba(100, 120, 200, 255)"    # macro commands echo
    C_PYTHON   = "rgba(80, 80, 200, 255)"      # python echo
    C_SHELL    = "rgba(200, 100, 50, 255)"     # shell echo
    C_SYSTEM   = "rgba(160, 130, 60, 255)"     # system/separator

    # ---- Color scheme presets ----
    COLOR_SCHEMES = {
        "Default": {
            "shell_echo":   "rgba(200, 100, 50, 255)",
            "shell_output": "rgba(0, 0, 0, 255)",
            "success":      "rgba(60, 140, 60, 255)",
            "error":        "rgba(200, 50, 50, 255)",
            "info":         "rgba(100, 100, 100, 255)",
            "agent":        "rgba(0, 120, 60, 255)",
            "shadow":       "rgba(0, 0, 0, 120)",
            "ansi_map": {
                '30': '#000000', '31': '#CD0000', '32': '#00CD00', '33': '#CDCD00',
                '34': '#0000EE', '35': '#CD00CD', '36': '#00CDCD', '37': '#E5E5E5',
                '90': '#7F7F7F', '91': '#FF0000', '92': '#00FF00', '93': '#FFFF00',
                '94': '#5C5CFF', '95': '#FF00FF', '96': '#00FFFF', '97': '#FFFFFF',
            },
        },
        "UV Blue": {
            "shell_echo":   "rgba(0, 0, 0, 255)",
            "shell_output": "rgba(0, 0, 0, 255)",
            "success":      "rgba(100, 120, 255, 255)",
            "error":        "rgba(200, 80, 180, 255)",
            "info":         "rgba(120, 100, 180, 255)",
            "agent":        "rgba(80, 100, 220, 255)",
            "shadow":       "rgba(100, 80, 255, 180)",
            "ansi_map": {
                '30': '#1A1030', '31': '#B040E0', '32': '#7C6CFF', '33': '#A88CFF',
                '34': '#5040FF', '35': '#C060FF', '36': '#6C8CFF', '37': '#D0C8FF',
                '90': '#6850B0', '91': '#D060FF', '92': '#8C7CFF', '93': '#C0A8FF',
                '94': '#6450FF', '95': '#E080FF', '96': '#80A0FF', '97': '#E8E0FF',
            },
        },
        "Amber": {
            "shell_echo":   "rgba(220, 160, 40, 255)",
            "shell_output": "rgba(200, 140, 30, 255)",
            "success":      "rgba(180, 200, 60, 255)",
            "error":        "rgba(220, 80, 40, 255)",
            "info":         "rgba(160, 140, 80, 255)",
            "agent":        "rgba(200, 170, 50, 255)",
            "shadow":       "rgba(200, 150, 30, 140)",
            "ansi_map": {
                '30': '#1A1400', '31': '#CC4400', '32': '#88AA00', '33': '#DDAA00',
                '34': '#AA7700', '35': '#CC6600', '36': '#BBAA44', '37': '#EEDDAA',
                '90': '#887744', '91': '#EE6600', '92': '#AACC22', '93': '#FFCC00',
                '94': '#CC9933', '95': '#EE8833', '96': '#DDCC66', '97': '#FFF0CC',
            },
        },
        "Green Terminal": {
            "shell_echo":   "rgba(80, 220, 100, 255)",
            "shell_output": "rgba(60, 200, 80, 255)",
            "success":      "rgba(100, 255, 120, 255)",
            "error":        "rgba(255, 100, 80, 255)",
            "info":         "rgba(80, 160, 80, 255)",
            "agent":        "rgba(60, 200, 100, 255)",
            "shadow":       "rgba(40, 200, 80, 140)",
            "ansi_map": {
                '30': '#0A1A0A', '31': '#CC3030', '32': '#30DD30', '33': '#80CC30',
                '34': '#30AA60', '35': '#60CC80', '36': '#40CCAA', '37': '#C0E8C0',
                '90': '#508050', '91': '#EE5050', '92': '#50FF50', '93': '#A0EE50',
                '94': '#50CC80', '95': '#80DDAA', '96': '#60DDCC', '97': '#E0FFE0',
            },
        },
        "Rose": {
            "shell_echo":   "rgba(220, 80, 120, 255)",
            "shell_output": "rgba(180, 60, 100, 255)",
            "success":      "rgba(220, 120, 160, 255)",
            "error":        "rgba(220, 50, 50, 255)",
            "info":         "rgba(160, 100, 120, 255)",
            "agent":        "rgba(200, 90, 130, 255)",
            "shadow":       "rgba(220, 60, 120, 150)",
            "ansi_map": {
                '30': '#1A0A10', '31': '#DD3060', '32': '#CC6090', '33': '#DD90AA',
                '34': '#AA4080', '35': '#DD50AA', '36': '#CC80AA', '37': '#F0D0E0',
                '90': '#905070', '91': '#FF4070', '92': '#DD80AA', '93': '#FFAACC',
                '94': '#CC60AA', '95': '#FF70CC', '96': '#DDA0CC', '97': '#FFE0F0',
            },
        },
    }

    # ------------------------------------------------------------------
    # Mount point auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_mount(subdir, marker_file, exclude=None):
        """
        Auto-detect a 9P mount point by probing common locations.

        Searches for ``marker_file`` inside candidate paths derived from
        the conventional Plan 9 namespace roots (/n/mux, /n).

        Args:
            subdir:       Expected subdirectory name (e.g. "llm").
                          If None, probes top-level children of each base.
            marker_file:  A file that must exist inside the candidate
                          (e.g. "ctl" for llmfs, "scene" for rio).
            exclude:      Optional directory name to skip when scanning
                          children (used to avoid matching llm as rio).

        Returns:
            The first matching path, or a sensible fallback.
        """
        bases = ["/n/mux", "/n"]

        if subdir:
            # Direct probe: /n/mux/<subdir>, /n/<subdir>
            for base in bases:
                candidate = os.path.join(base, subdir)
                if os.path.isfile(os.path.join(candidate, marker_file)):
                    return candidate
        else:
            # Scan children of each base for the marker file
            for base in bases:
                if not os.path.isdir(base):
                    continue
                try:
                    for name in sorted(os.listdir(base)):
                        if exclude and name == exclude:
                            continue
                        candidate = os.path.join(base, name)
                        if os.path.isdir(candidate) and os.path.exists(
                            os.path.join(candidate, marker_file)
                        ):
                            return candidate
                except OSError:
                    continue

        # Fallback to the most common convention
        if subdir:
            return os.path.join("/n/mux", subdir)
        return "/n/mux/default"

    @staticmethod
    def _resolve_p9_token(explicit_token=None):
        """
        Find a 9P auth token from any of the four supported sources.
        See the docstring at the call site in __init__ for the priority
        order and rationale. Returns the token string, or None when
        none is configured anywhere.
        
        This is intentionally a static method so it's testable in
        isolation and so the discovery logic has zero dependency on
        widget state.
        """
        # 1. Explicit kwarg always wins (must be a non-empty, non-whitespace string).
        if explicit_token and explicit_token.strip():
            return explicit_token.strip()
        
        # 2 & 3. Env vars — try LLMFS first then RIO. Comma-separated;
        #        take the first non-empty entry. Empty/whitespace tokens
        #        are silently skipped so an accidental trailing comma
        #        doesn't poison the result.
        for env_name in ("LLMFS_AUTH_TOKENS", "RIO_AUTH_TOKENS"):
            raw = os.environ.get(env_name, "").strip()
            if not raw:
                continue
            for piece in raw.split(","):
                tok = piece.strip()
                if tok:
                    return tok
        
        # 4. Sniff this process's own argv for --auth-token. When rio
        #    was launched as `python -m rio.main --port ... --auth-token <tok>`,
        #    sys.argv carries that token verbatim. We accept both:
        #         --auth-token <value>
        #         --auth-token=<value>
        #    Multiple --auth-token flags are valid; we use the first one
        #    (matches AuthManager's "first non-empty token wins" behavior
        #    for the env-var case above).
        argv = sys.argv or []
        for i, arg in enumerate(argv):
            if arg == "--auth-token" and i + 1 < len(argv):
                tok = argv[i + 1].strip()
                if tok:
                    return tok
            elif arg.startswith("--auth-token="):
                tok = arg[len("--auth-token="):].strip()
                if tok:
                    return tok
        
        # No token found anywhere — connect unauthenticated.
        return None

    def __init__(self, parent=None, llmfs_mount=None,
                 rio_mount=None,
                 p9_host="localhost", p9_port=5640,
                 p9_token=None):
        super().__init__(parent)

        # Auto-detect mount points if not explicitly provided.
        # Probe common locations for the llmfs ctl file.
        if llmfs_mount is None:
            llmfs_mount = self._detect_mount("llm", "ctl")
        if rio_mount is None:
            rio_mount = self._detect_mount(None, "scene", exclude="llm")

        self.llmfs_mount = llmfs_mount
        self.rio_mount = rio_mount
        self.p9_host = p9_host
        self.p9_port = p9_port
        
        # 9P auth token for OutputStreamReader / MasterBashReader.
        #
        # Resolution order (first hit wins):
        #   1. explicit p9_token kwarg from the caller
        #   2. LLMFS_AUTH_TOKENS env var (first comma-separated value)
        #   3. RIO_AUTH_TOKENS env var (first comma-separated value)
        #   4. Sniff sys.argv for --auth-token / --auth-token=<v>
        #
        # Why all four:
        #
        # main.py constructs us with TerminalWidget(llmfs_mount=..., rio_mount=...)
        # and doesn't (yet) pass a token. Rio's own main.py was launched with
        # --auth-token <tok>, so that token sits in this process's sys.argv —
        # sniffing it lets us auto-pick up rio's token without main.py
        # changes. The env-var fallbacks cover the case where the launcher
        # exports a shared token instead. If none of the four sources have
        # a token, we connect unauthenticated, which:
        #   - works against an unauthed server (legacy / dev),
        #   - fails with "9P connect failed: authentication required"
        #     against an authed one (the exact error you saw in the UI).
        self.p9_token = self._resolve_p9_token(p9_token)
        self.command_history = []
        self.history_index = -1
        self.text_displays = []
        self.current_text_display = None
        self.terminal_mode = False
        self._password_mode = False  # Flag for password prompts

        # Active color scheme (applied globally, not just in terminal mode)
        self._active_scheme_name = "UV Blue"
        self._active_scheme = dict(self.COLOR_SCHEMES["UV Blue"])

        # Connected agent state
        self.connected_agent = None          # str name
        self._response_pending = False       # True while streaming a response

        # Known agents (populated during setup, safe to read from any thread)
        self.known_agents: set = set()
        
        # Known supplementary output files per agent (safe to read from any thread)
        # Maps agent_name -> set of supplementary file names
        self.known_supplementary: Dict[str, set] = {}


        # Plan 9 style attachments — delegated to RoutesManager
        # (shared across all terminals, lives at /n/rioa/routes)
        # Set via set_routes_manager() after filesystem init.
        self._routes_manager = None
        # Raw 9P output stream reader for connected agent
        # (Plan9Attachment cat loops don't stream properly through FUSE —
        #  the kernel re-reads from offset 0 on each cat invocation,
        #  producing superimposed/duplicated output on 2nd+ generations)
        self._output_reader: OutputStreamReader = None

        # Master agent state
        self._master_bash_reader: MasterBashReader = None
        self._master_active = False
        self.term_id = f"term_{uuid.uuid4().hex[:8]}"
        self._term_dir = None  # Set when registered in Rio filesystem

        # Peribus mycelium layer state. Lazily activated via /peribus.
        # _peribus_root is the FUSE mountpoint of /n/peribus on this host;
        # _peribus_tailer streams feed/new into our $term/inline pipeline.
        # _peribus_inbox_tailer does the same for inbox/new — incoming
        # DMs from peers, rendered as pink-tinted post cards so the user
        # can tell mail apart from the public feed at a glance.
        self._peribus_root = "/n/peribus"
        self._peribus_tailer: Optional[PeribusFeedTailer] = None
        self._peribus_inbox_tailer: Optional[PeribusFeedTailer] = None
        # Dedup sets for peribus post/DM ids. The tailer's fd may be
        # reopened on transient FUSE EIO, which makes the daemon hand
        # out a fresh cursor and replay buffered posts — without dedup
        # we'd render the same card twice. We keep these bounded to
        # avoid unbounded growth over long sessions; eviction is FIFO
        # via the deque, which is fine because replays only happen
        # close in time to the original delivery.
        self._peribus_seen_feed_ids: "collections.deque[str]" = collections.deque(maxlen=2048)
        self._peribus_seen_feed_set: set = set()
        self._peribus_seen_inbox_ids: "collections.deque[str]" = collections.deque(maxlen=2048)
        self._peribus_seen_inbox_set: set = set()
        self._suppress_echo_line = None  # Command text to suppress from PTY echo
        self._suppress_echo_buf = ""     # Accumulator for multi-chunk echo suppression
        self._suppress_shell_output = False  # Suppress ALL PTY output (during seeding)
        self.acme_panel = None
        self.operator_panel = None
        self.version_panel = None
        self.scene_panel = None  # Per-terminal live UI surface (lazy)
        self._active_panel = None  # Currently visible side panel in the splitter
        self._proxy = None  # Set by main.py when added to QGraphicsScene
        self._font_size = 12  # Default font size (px)
        self._is_dark_mode = False  # Dark mode state

        # Active visual theme.  Mirrors the RioWindow's theme — when
        # RioWindow.set_theme() is called, it pushes apply_theme() to
        # every terminal which updates this attribute.  Standalone
        # (no parent RioWindow) terminals just use the default theme.
        self._theme_name: str = DEFAULT_THEME_NAME
        self._paper_bg_rgb = None  # Per-terminal paper pastel (set by apply_theme)

        # Pop-out window state (for /pop and /dock)
        self._pop_window = None         # The frameless external QWidget wrapper
        self._pop_scene = None          # The QGraphicsScene we were in
        self._pop_proxy = None          # The QGraphicsProxyWidget we were in
        self._pop_scene_pos = None      # Position in scene before pop
        self._pop_size = None           # Size before pop

        # Tab completion state
        self._tab_state_text = None    # input text at first Tab press
        self._tab_candidates = []      # current candidate list
        self._tab_index = 0            # index into candidates for cycling
        self._tab_prefix = ""          # text before the token being completed

        # Plan 9-style right-click menu filter
        self._plan9_menu_filter = Plan9MenuFilter(self)

        self._init_ui()
        # Stream router: detects ```machine ... ``` fences in agent/FS
        # streams and morphs them into inline widgets. Created after
        # _init_ui because it captures the initial current_text_display.
        # See class docstring for the per-source concurrency model.
        self.stream_router = TerminalStreamRouter(self)
        self._setup_shell_process()
        self.installEventFilter(self)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowFlags(Qt.Widget)
        self.setFocusPolicy(Qt.StrongFocus)
        # Accept file drops — handled in dragEnterEvent / dropEvent for
        # /share. Without this, Qt swallows drag events before they reach
        # us. We accept on the terminal as a whole; the inner QTextEdits
        # would otherwise gobble the drop with their default text-insert
        # behavior, which we don't want for paths that are real files.
        self.setAcceptDrops(True)
        # Ensure fully transparent — critical when embedded in a
        # QGraphicsProxyWidget which otherwise paints an opaque bg
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.setup_terminal_frame()
        main_layout.addWidget(self.terminal_frame)
        #self.setMinimumSize(10, 10)

    # ------------------------------------------------------------------
    # Theme support
    # ------------------------------------------------------------------

    def _find_main_window(self):
        """Walk up to the RioWindow if we're attached to a scene.

        Returns the QMainWindow if found, else None.  Used by the
        ``_theme`` accessor so a newly-created terminal — whose
        ``_theme_name`` is the constructor default — still picks up
        the parent window's active theme on its very first paint.
        Mirrors the walk in ``_toggle_dark_mode_from_terminal``.
        """
        # Proxy → scene → first view → window
        if self._proxy is not None and self._proxy.scene() is not None:
            views = self._proxy.scene().views()
            if views:
                w = views[0].window()
                if hasattr(w, 'set_theme'):
                    return w
        # Fallback: parent chain (covers popped-out / standalone)
        p = self.parent()
        while p is not None:
            if hasattr(p, 'set_theme'):
                return p
            p = p.parent() if hasattr(p, 'parent') else None
        return None

    @property
    def _theme(self) -> _Theme:
        """The currently active Theme.

        Resolution order:
          1. The parent RioWindow's ``current_theme``, if reachable —
             this is the live source of truth and ensures new terminals
             pick up Paper/Glass/etc. without an explicit handoff.
          2. ``self._theme_name`` (last value set by ``apply_theme``).
          3. The default theme.

        Doing the walk on every read sounds expensive but it's just a
        few attribute hops; the QSS calls themselves dwarf it.
        """
        mw = self._find_main_window()
        if mw is not None:
            try:
                self._theme_name = mw.current_theme.name
                return mw.current_theme
            except AttributeError:
                pass
        return get_theme(self._theme_name)

    def setup_terminal_frame(self):
        self.terminal_frame = QFrame()
        self.terminal_frame.setFrameStyle(QFrame.StyledPanel)
        # Initial frame style derived from the active theme (idle, no focus).
        self.terminal_frame.setStyleSheet(
            self._theme.frame_stylesheet(self._is_dark_mode, focus_alpha=0)
        )
        # Focus-tint overlay: paints the animated focus highlight on top
        # of terminal_frame without setStyleSheet, so Qt never re-styles
        # the hundreds of inline widgets inside the content layout.
        self._focus_overlay = FocusTintOverlay(self.terminal_frame)
        self._focus_overlay.sync_geometry()
        # Use the theme's inner padding instead of a hardcoded 10.
        _pad = self._theme.frame.inner_padding

        terminal_layout = QVBoxLayout(self.terminal_frame)
        terminal_layout.setContentsMargins(_pad, _pad, _pad, _pad)
        terminal_layout.setSpacing(5)

        # Scrollable output area
        self.terminal_scroll = QScrollArea()
        self.terminal_scroll.setWidgetResizable(True)
        self.terminal_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }

            /* ── Vertical scrollbar ── */
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 4px 2px 4px 0px;
                border: none;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(160, 160, 160, 0.15);
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(160, 160, 160, 0.15);
            }
            QScrollBar::handle:vertical:pressed {
                background: rgba(160, 160, 160, 0.15);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }

            /* ── Horizontal scrollbar ── */
            QScrollBar:horizontal {
                background: transparent;
                height: 8px;
                margin: 0px 4px 2px 4px;
                border: none;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(255, 255, 255, 0.15);
                min-width: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(255, 255, 255, 0.30);
            }
            QScrollBar::handle:horizontal:pressed {
                background: rgba(255, 255, 255, 0.45);
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                background: transparent;
                border: none;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
            }

            /* Hide the corner widget where scrollbars meet */
            QScrollArea QWidget#qt_scrollarea_corner {
                background: transparent;
            }
        """)
        self.terminal_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.terminal_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.terminal_content = QWidget()
        self.terminal_content.setStyleSheet("QWidget { background-color: transparent; border: none; }")
        self.terminal_content_layout = QVBoxLayout(self.terminal_content)
        # No right margin here — inline widgets and code blocks are
        # individually clamped to _inline_max_width() which already
        # accounts for the scroll bar. Adding margin here would just
        # double-count it for text displays.
        self.terminal_content_layout.setContentsMargins(0, 0, 0, 0)
        self.terminal_content_layout.setSpacing(0)
        self.terminal_content_layout.setAlignment(Qt.AlignTop)

        self.text_display = self._create_text_display()
        self.terminal_content_layout.addWidget(self.text_display)
        self.text_displays = [self.text_display]
        self.current_text_display = self.text_display

        self.terminal_scroll.setWidget(self.terminal_content)

        # Auto-scroll: whenever content grows, scroll to bottom
        self._auto_scroll = True
        vsb = self.terminal_scroll.verticalScrollBar()
        vsb.rangeChanged.connect(self._on_scroll_range_changed)
        vsb.valueChanged.connect(self._on_scroll_value_changed)

        # Command input
        self._setup_command_input()

        terminal_layout.addWidget(self.terminal_scroll)
        terminal_layout.addLayout(self.input_container)

        # Hidden until show_content()
        self.terminal_scroll.hide()
        self.command_input.hide()

    def _create_text_display(self):
        te = QTextEdit()
        # Cap document history at 10,000 blocks. Earlier the QTextDocument
        # grew unboundedly: every append, every cursor positioning, and
        # especially every dark-mode toggle (which walks all fragments
        # to recolor "default-coloured" text) scaled with session age.
        # Long-running terminals visibly stuttered on dark-mode toggle
        # because the per-tick fragment iteration was O(document size).
        # 10k blocks is hours of normal terminal output; older lines
        # silently drop off the top.
        te.document().setMaximumBlockCount(10000)
        size = getattr(self, '_font_size', 12)
        dark = getattr(self, '_is_dark_mode', False)
        if dark:
            text_color = "rgba(230, 230, 230, 255)"
            sel_bg = "rgba(100, 100, 255, 120)"
        else:
            text_color = "rgba(0, 0, 0, 255)"
            sel_bg = "rgba(100, 100, 255, 100)"
        # Lock the terminal mono font to the Glass stack regardless of
        # active theme — paper mode's font is intentionally not used in
        # the terminal.
        mono = "'Consolas', 'Monaco', monospace"
        te.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent; border: none;
                color: {text_color};
                selection-background-color: {sel_bg};
                font-family: {mono};
                font-size: {size}px;
            }}
        """)
        te.setReadOnly(False)
        te.setCursorWidth(2)
        te.setContextMenuPolicy(Qt.CustomContextMenu)
        te.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        te.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        te.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        te.setMinimumHeight(20)
        # Coalesced height adjustment. contentsChanged fires per
        # character insertion — under fast streaming that's hundreds
        # of times per second, each one calling document().size() and
        # then setMin/MaxHeight, which triggers a Qt layout pass up the
        # widget tree. We collapse a burst into one adjustment per
        # frame (~16 ms) using a per-display single-shot timer.
        adjust_timer = QTimer(te)
        adjust_timer.setSingleShot(True)
        adjust_timer.setInterval(16)
        adjust_timer.timeout.connect(lambda _te=te: self._adjust_height(_te))
        te._adjust_height_timer = adjust_timer

        def _on_contents_changed(_te=te, _t=adjust_timer):
            if not _t.isActive():
                _t.start()

        te.document().contentsChanged.connect(_on_contents_changed)
        # Install Plan 9 mouse menu handler
        te.installEventFilter(self)
        te.setFocusPolicy(Qt.StrongFocus)
        te.viewport().installEventFilter(self._plan9_menu_filter)

        # Forward wheel events from text display to the outer scroll area
        te.wheelEvent = lambda event, _te=te: self._forward_wheel_event(event)
        return te

    def _adjust_height(self, te):
        h = int(te.document().size().height() + 10)
        te.setMaximumHeight(h)
        te.setMinimumHeight(h)

    def _forward_wheel_event(self, event):
        """Forward wheel events from text displays to the outer terminal scroll area."""
        if hasattr(self, 'terminal_scroll') and self.terminal_scroll is not None:
            self.terminal_scroll.verticalScrollBar().setValue(
                self.terminal_scroll.verticalScrollBar().value() - event.angleDelta().y()
            )
            event.accept()

    @Slot(int)
    def set_font_size(self, size: int):
        """
        Change the font size of all terminal text displays and command input.

        Callable from:
          - echo 'font 14' > /n/rioa/terms/<term_id>/ctl
          - self.set_font_size(14) from Python
        """
        size = max(6, min(size, 72))  # clamp to sane range
        self._font_size = size

        # Build a QFont using the Glass mono stack. Locked here
        # regardless of active theme (see _create_text_display for
        # rationale). We extract family *names* from the CSS-style
        # stack and feed them to QFont.setFamilies(); Qt picks the
        # first one available.
        mono_css = "'Consolas', 'Monaco', monospace"
        family_list = [s.strip().strip("'").strip('"') for s in mono_css.split(",")]
        # Drop the trailing 'monospace' generic — QFont doesn't know it.
        family_list = [f for f in family_list if f.lower() != "monospace"]
        font = QFont()
        font.setFamilies(family_list or ["Consolas"])
        font.setPointSize(size)

        dark = getattr(self, '_is_dark_mode', False)
        if dark:
            text_color = "rgba(230, 230, 230, 255)"
            sel_bg = "rgba(100, 100, 255, 120)"
        else:
            text_color = "rgba(0, 0, 0, 255)"
            sel_bg = "rgba(100, 100, 255, 100)"

        # Update all existing text displays
        for te in self.text_displays:
            te.setStyleSheet(f"""
                QTextEdit {{
                    background-color: transparent; border: none;
                    color: {text_color};
                    selection-background-color: {sel_bg};
                    font-family: {mono_css};
                    font-size: {size}px;
                }}
            """)
            te.setFont(font)
            # Re-adjust height for new font
            self._adjust_height(te)

        # Update command input
        self.command_input.setFont(font)
        if dark:
            self._set_input_bg_target(40, 40, 50, 180)
        else:
            self._set_input_bg_target(255, 255, 255, 150)

        self.append_text(f"font size: {size}px\n", self.C_INFO)

    def _setup_command_input(self):
        self.input_container = QHBoxLayout()
        self.input_container.setSpacing(5)

        self.command_input = QTextEdit()
        # Initial font: locked to the Glass mono stack regardless of
        # active theme (see _create_text_display for rationale).
        # QFont.setFamilies wants a list while themes hold a CSS-style
        # string, so we feed it the families directly.
        _f = QFont()
        _f.setFamilies(["Consolas", "Monaco"])
        _f.setPointSize(10)
        self.command_input.setFont(_f)
        self.command_input.setMaximumHeight(60)
        self.command_input.setCursorWidth(2)

        # Focus animation state — tracks current bg rgba + target alpha.
        # RGB and target alpha are sourced from the active theme so
        # paper/glass each get appropriate focus tints.
        _spec = self._theme.input
        _bg = _spec.bg_rgb_dark if self._is_dark_mode else _spec.bg_rgb
        self._input_bg_r = _bg[0]
        self._input_bg_g = _bg[1]
        self._input_bg_b = _bg[2]
        self._input_bg_alpha = 0          # current animated alpha
        self._input_bg_target_alpha = (
            _spec.focus_alpha_dark if self._is_dark_mode else _spec.focus_alpha
        )
        self._input_focus_anim = None      # QTimer for animation

        # Frame focus animation state — mirrors input, animates terminal_frame bg alpha
        self._frame_focus_alpha = 0       # current animated alpha
        self._frame_focus_anim = None     # QTimer for animation

        self._apply_input_style()
        self.command_input.setPlaceholderText("Enter command or prompt...")
        self.command_input.installEventFilter(self)
        self.input_container.addWidget(self.command_input, stretch=1)

    def _apply_input_style(self):
        """Apply command input stylesheet using current _input_bg_* state.

        RGB and text colour come from the active theme; alpha is the
        animated focus value (kept at 0 here historically — the actual
        focus tinting is on the frame, not the input — but theme can
        still override font/border).
        """
        size = getattr(self, '_font_size', 12)
        dark = getattr(self, '_is_dark_mode', False)
        spec = self._theme.input
        if dark:
            tr, tg, tb, ta_ = spec.text_rgba_dark
        else:
            tr, tg, tb, ta_ = spec.text_rgba
        text_color = f"rgba({tr}, {tg}, {tb}, {ta_})"
        r, g, b = self._input_bg_r, self._input_bg_g, self._input_bg_b
        a = 0  # input bg stays invisible; focus tint is carried by the frame
        # Mono font locked to Glass stack regardless of active theme.
        mono = "'Consolas', 'Monaco', monospace"
        self.command_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba({r}, {g}, {b}, {a});
                color: {text_color};
                border: none;
                border-radius: {spec.border_radius}px; padding: 5px;
                font-family: {mono};
                font-size: {size}px;
            }}
        """)

    def _animate_input_focus(self, focus_in: bool):
        """Animate command input background alpha on focus in/out.

        IMPORTANT: _apply_input_style() hardcodes the displayed alpha to
        0 (see the `a = 0` line and its comment — focus tinting is
        carried by the frame, not the input). So updating
        _input_bg_alpha and re-applying the stylesheet 12 times in a
        row produces 12 *identical* stylesheets — visually a no-op,
        but each setStyleSheet call re-runs the QSS parser and triggers
        a restyle of the QTextEdit and its descendants.

        Earlier this thrashed the stylesheet system on every focus
        change. Now we just update the bookkeeping value (in case other
        code reads it) without the restyle. The frame-focus animation
        (_animate_frame_focus) carries the actual visible focus fade.
        """
        if self._input_focus_anim is not None:
            self._input_focus_anim.stop()
            self._input_focus_anim.deleteLater()
            self._input_focus_anim = None

        target = self._input_bg_target_alpha if focus_in else 0
        start = self._input_bg_alpha
        if start == target:
            return

        # Snap the bookkeeping value. No QTimer, no setStyleSheet calls.
        # If a future theme actually wants the input bg alpha to
        # animate visibly, _apply_input_style needs to read
        # self._input_bg_alpha instead of hardcoding 0 — and *then*
        # re-introducing the timer here would make sense.
        self._input_bg_alpha = target

    def _set_input_bg_target(self, r, g, b, target_alpha):
        """Update the input background color targets (called by mode/theme changes)."""
        self._input_bg_r = r
        self._input_bg_g = g
        self._input_bg_b = b
        self._input_bg_target_alpha = target_alpha
        # If not focused, keep alpha at 0; if focused, snap to new target.
        # Apply the same logic to the frame so they stay in sync.
        if self.command_input.hasFocus():
            self._input_bg_alpha = target_alpha
            self._frame_focus_alpha = target_alpha
        else:
            self._input_bg_alpha = 0
            self._frame_focus_alpha = 0
        self._apply_input_style()
        self._apply_frame_focus_style()

    # ------------------------------------------------------------------
    # Whole-terminal frame focus animation (same bg color/alpha as input)
    # ------------------------------------------------------------------

    def _apply_frame_focus_style(self):
        """Snap the focus tint overlay to the current _frame_focus_alpha.

        Used by non-animated paths (theme changes, mode switches) that
        need the overlay to reflect the current focus state without
        running the full animation.  Does NOT touch terminal_frame's
        stylesheet — that's set once at init and on theme/dark-mode
        transitions only.
        """
        a = self._frame_focus_alpha
        if hasattr(self, '_focus_overlay'):
            f = self._theme.frame
            if self._is_dark_mode:
                fill = f.fill_rgba_idle_dark if hasattr(f, 'fill_rgba_idle_dark') else (30, 30, 30, 0)
            else:
                fill = f.fill_rgba_idle if hasattr(f, 'fill_rgba_idle') else (255, 255, 255, 0)
            radius = f.radius if hasattr(f, 'radius') else 8
            self._focus_overlay.set_tint(fill[0], fill[1], fill[2], a, radius)
        else:
            # Fallback before overlay is created (shouldn't happen in
            # normal flow, but defensive for subclasses / tests).
            self.terminal_frame.setStyleSheet(
                self._theme.frame_stylesheet(self._is_dark_mode, focus_alpha=a)
            )

    def _animate_frame_focus(self, focus_in: bool):
        """Animate focus tint via FocusTintOverlay — zero stylesheet cascade.

        Previous implementation called setStyleSheet on terminal_frame ~12
        times per animation (200 ms at ~60 Hz). Each call forced Qt to
        re-resolve CSS rules for every descendant widget (QTextEdits,
        inline code/media widgets, buttons, labels…). With many inline
        widgets the restyle walk dominated focus-switch latency.

        Now the animation updates a lightweight overlay that paints a
        single rounded rect and calls update() — repainting only itself.
        terminal_frame's stylesheet is never touched during focus, so Qt
        never walks its children.

        Opaque-theme short-circuit is preserved: if the theme's focus
        tint is invariant (paper-style fills already at full alpha),
        we snap and skip the animation entirely.
        """
        # 1. Stop any existing animation safely
        if hasattr(self, '_frame_focus_anim') and self._frame_focus_anim:
            self._frame_focus_anim.stop()
            self._frame_focus_anim.deleteLater()
            self._frame_focus_anim = None

        target_alpha = 220 if focus_in else 0
        start_alpha = self._frame_focus_alpha

        # Opaque-theme short-circuit: if the stylesheet would be
        # identical at both endpoints, the overlay tint is invariant
        # too — snap and skip.
        css_start = self._theme.frame_stylesheet(
            self._is_dark_mode, focus_alpha=start_alpha
        )
        css_end = self._theme.frame_stylesheet(
            self._is_dark_mode, focus_alpha=target_alpha
        )
        if css_start == css_end:
            self._frame_focus_alpha = target_alpha
            return

        # 2. Resolve the tint RGB from the theme's frame fill.
        #    The frame_stylesheet encodes focus_alpha into the
        #    background-color rgba — we extract the RGB so the overlay
        #    paints the same colour the stylesheet would have used.
        f = self._theme.frame
        if self._is_dark_mode:
            fill = f.fill_rgba_idle_dark if hasattr(f, 'fill_rgba_idle_dark') else (30, 30, 30, 0)
        else:
            fill = f.fill_rgba_idle if hasattr(f, 'fill_rgba_idle') else (255, 255, 255, 0)
        tint_r, tint_g, tint_b = fill[0], fill[1], fill[2]
        radius = f.radius if hasattr(f, 'radius') else 8

        # Ensure overlay geometry is current
        overlay = self._focus_overlay
        overlay.sync_geometry()

        # 3. Create the animation
        self._frame_focus_anim = QVariantAnimation(self)
        self._frame_focus_anim.setDuration(200)
        self._frame_focus_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._frame_focus_anim.setStartValue(float(start_alpha))
        self._frame_focus_anim.setEndValue(float(target_alpha))

        # 4. Per-tick: update overlay tint (triggers a single-widget repaint).
        def update_alpha(value):
            self._frame_focus_alpha = int(value)
            overlay.set_tint(tint_r, tint_g, tint_b, value, radius)

        self._frame_focus_anim.valueChanged.connect(update_alpha)

        # 5. Start WITHOUT DeleteWhenStopped to avoid the race condition crash
        self._frame_focus_anim.start()
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def show_content(self):
        """Reveal terminal content (called after creation animation)."""
        self.terminal_scroll.show()
        self.command_input.show()
        self.animate_shadow_to_position()
        
        # Initialize resize/drag state
        self.RESIZE_MARGIN = 10  # Pixel margin for resize detection
        self._resizing = False
        self._resize_corner = None  # 'tl', 'tr', 'bl', 'br' for corners
        self._resize_start_pos = None
        self._resize_start_geometry = None
        self._dragging = False
        self._drag_offset = QPointF(0, 0)
        
        # Enable mouse tracking for hover detection
        self.setMouseTracking(True)

        self._stream_text(self.term_id, self.C_INFO, interval_ms=32, callback=lambda: self.append_text("\n", self.C_INFO))

    # ------------------------------------------------------------------
    # Routes manager integration
    # ------------------------------------------------------------------
    
    def set_routes_manager(self, manager):
        """
        Bind this terminal to a shared RoutesManager.
        Called after filesystem init provides the manager instance.
        """
        self._routes_manager = manager
    
    @property
    def attachments(self) -> Dict[str, 'Plan9Attachment']:
        """
        Proxy to the shared RoutesManager's attachments dict.
        
        Keeps backward compatibility for code that reads self.attachments.
        Returns an empty dict if no routes manager is set yet.
        """
        if self._routes_manager:
            return self._routes_manager.attachments
        return {}

    def closeEvent(self, event):
        self._stop_master()
        self._teardown_shell()
        
        # Stop raw 9P output reader
        if self._output_reader:
            self._output_reader.stop()
            self._output_reader.wait(2000)
            self._output_reader = None

        # Stop the peribus feed and inbox tailers (the daemon itself
        # outlives us — these are read-only consumers).
        if self._peribus_tailer is not None:
            self._peribus_tailer.stop()
            self._peribus_tailer = None
        if self._peribus_inbox_tailer is not None:
            self._peribus_inbox_tailer.stop()
            self._peribus_inbox_tailer = None

        # Stop the post-card tick timer if any cards ever registered.
        if (hasattr(self, "_post_tick_timer")
                and self._post_tick_timer is not None):
            try:
                self._post_tick_timer.stop()
            except RuntimeError:
                pass

        # Stop all attachments owned by this terminal
        # (routes are shared via RoutesManager, but we stop them on last terminal close)
        if self._routes_manager:
            self._routes_manager.stop_all()
        
        # Close pop-out window if active
        if self._pop_window is not None:
            self._cleanup_overlap_monitor()
            self._pop_window.close()
            self._pop_window = None

        if self.acme_panel is not None:
            self.acme_panel.close()
        if self.operator_panel is not None:
            self.operator_panel.close()
        if self.version_panel is not None:
            self.version_panel.close()
        if self.scene_panel is not None:
            self.scene_panel.close()

        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Key handling  (Enter = submit, Shift+Enter = newline)
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):

        if not hasattr(self, 'command_input') or self.command_input is None:
            return super().eventFilter(obj, event)
        if event.type() == event.Type.FocusIn:
            self._animate_frame_focus(True)
        elif event.type() == event.Type.FocusOut:
            self._animate_frame_focus(False)

        if obj is self.command_input:
            if event.type() == QKeyEvent.Type.KeyPress:
                # --- Global shortcuts (Ctrl+key) ---
                if event.modifiers() == Qt.ControlModifier:
                    if event.key() == Qt.Key_E:
                        self._toggle_acme_panel()
                        return True
                    if event.key() == Qt.Key_O:
                        self._toggle_operator_panel()
                        return True
                    if event.key() == Qt.Key_T:
                        # Toggle terminal mode (same as typing "$")
                        self._route("$")
                        return True
                    if event.key() == Qt.Key_P:
                        self._toggle_version_panel()
                        return True
                if event.key() == Qt.Key_Return and event.modifiers() == Qt.NoModifier:
                    self._reset_tab_state()
                    self._submit_command()
                    return True
                if event.key() == Qt.Key_Return and event.modifiers() == Qt.ShiftModifier:
                    self._reset_tab_state()
                    return False  # default: insert newline
                if event.key() == Qt.Key_Tab and event.modifiers() == Qt.NoModifier:
                    self._tab_complete()
                    return True
                if event.key() == Qt.Key_Up and event.modifiers() == Qt.ControlModifier:
                    self._reset_tab_state()
                    self._history_prev()
                    return True
                if event.key() == Qt.Key_Down and event.modifiers() == Qt.ControlModifier:
                    self._reset_tab_state()
                    self._history_next()
                    return True
                if event.key() == Qt.Key_Delete:
                    self._reset_tab_state()
                    self._interrupt_shell()
                    return True
                # Any other key resets tab cycling state
                if event.key() != Qt.Key_Tab:
                    self._reset_tab_state()
        return super().eventFilter(obj, event)

    def _reset_tab_state(self):
        """Clear tab completion cycling state."""
        self._tab_state_text = None
        self._tab_candidates = []
        self._tab_index = 0

    # ------------------------------------------------------------------
    # Tab completion
    # ------------------------------------------------------------------

    # All known macro commands and their argument hint types:
    #   None       = no argument
    #   'agent'    = complete agent names
    #   'path'     = complete filesystem paths
    #   'free'     = free-form text (no completion)
    _MACRO_COMMANDS = {
        'help':        None,
        'cls':         None,
        'clear':       None,
        'cancel':      None,
        'retry':       None,
        'disconnect':  None,
        'status':      None,
        'list':        None,
        'ls':          None,
        'attachments': None,
        'restart':     None,
        'setup':       None,
        'acme':        None,
        'operator':    None,
        'connect':     'agent',
        'delete':      'agent',
        'master':      'free',
        'coder':       'free',
        'tcoder':      'free',
        'av':          'free',
        'attach':      'path',
        'detach':      'path',
        'mount':       'free',
        'system':      'free',
        'model':       'free',
        'temperature': 'free',
        'history':     None,
        'config':      None,
        'errors':      None,
        'color':       None,
        'colors':      None,
    }

    # Track consecutive tab presses for cycling / showing options

    def _tab_complete(self):
        """
        Handle Tab key press for auto-completion.
        
        Strategy: always complete the LAST token (whitespace-delimited word)
        in the text up to the cursor. This handles:
          - "cd /n/r"  →  complete "/n/r" → "/n/rioa/"
          - "/con"     →  complete macro name
          - "/connect c" → complete agent name
          - "ls /n/llm/" → complete path
        
        Path completion runs in a background thread with a timeout to
        avoid freezing the UI on slow filesystems (9P/FUSE mounts).
        """
        full_text = self.command_input.toPlainText()
        cursor_pos = self.command_input.textCursor().position()
        text_to_cursor = full_text[:cursor_pos]
        text_after_cursor = full_text[cursor_pos:]

        # Detect if this is a continuation of the same tab session
        if text_to_cursor != self._tab_state_text:
            self._tab_state_text = text_to_cursor
            self._tab_candidates = []
            self._tab_index = 0

        # If we already have candidates, cycle through them (instant, no I/O)
        if self._tab_candidates:
            self._tab_index = (self._tab_index + 1) % len(self._tab_candidates)
            self._apply_token_completion(
                self._tab_prefix, self._tab_candidates[self._tab_index], text_after_cursor
            )
            return

        # ---- Split into prefix (before last token) + token to complete ----
        prefix, token = self._split_last_token(text_to_cursor)

        # ---- Determine completion context ----
        candidates = []
        needs_async_path = False   # True when we need _complete_path (I/O)

        if not text_to_cursor.strip():
            # Empty input: no completion
            return

        # Macro command name: only when "/" is the very first char and we're
        # still on the first token (no space yet in the content after "/")
        if text_to_cursor.startswith('/') and ' ' not in text_to_cursor.strip():
            candidates = self._complete_macro_name(token)
        elif text_to_cursor.startswith('/') and ' ' in text_to_cursor:
            # Macro argument completion
            cmd = text_to_cursor[1:].split()[0].lower()
            arg_type = self._MACRO_COMMANDS.get(cmd)
            if arg_type == 'agent':
                candidates = self._complete_agent_name(token)
            elif arg_type == 'path':
                needs_async_path = True
            else:
                # Unknown command or 'free' — try path completion as fallback
                needs_async_path = True
        elif self.terminal_mode or text_to_cursor.startswith('$ ') or \
             text_to_cursor.startswith('$ '):
            # Shell mode: complete paths on the last token
            needs_async_path = True
        else:
            # Agent prompt mode — no completion
            return

        if needs_async_path:
            # Run path completion in a background thread with timeout.
            # This prevents the Qt main thread from freezing when scandir
            # goes through FUSE → 9P and the server is slow or blocking.
            self._complete_path_async(token, prefix, text_after_cursor)
            return

        if not candidates:
            return

        self._apply_tab_candidates(prefix, token, candidates, text_after_cursor)

    def _complete_path_async(self, token: str, prefix: str, text_after_cursor: str):
        """
        Run _complete_path in a background thread with a timeout.
        
        Prevents GUI freeze when scandir hits a slow or blocking
        9P/FUSE mount (e.g. /n/rioa/scene/ with blocking files).
        """
        import concurrent.futures

        if not hasattr(self, '_tab_executor'):
            self._tab_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="tab-complete"
            )

        future = self._tab_executor.submit(self._complete_path, token)

        # Use a QTimer to poll for the result without blocking the GUI.
        # Total timeout: ~1.5s (check every 50ms, up to 30 checks).
        state = {'checks': 0}

        def _poll_result():
            state['checks'] += 1
            if future.done():
                timer.stop()
                try:
                    candidates = future.result(timeout=0)
                except Exception:
                    candidates = []
                if candidates:
                    self._apply_tab_candidates(
                        prefix, token, candidates, text_after_cursor
                    )
            elif state['checks'] >= 30:
                # Timeout — cancel and give up
                timer.stop()
                future.cancel()

        timer = QTimer(self)
        timer.setInterval(50)
        timer.timeout.connect(_poll_result)
        timer.start()

    def _apply_tab_candidates(self, prefix, token, candidates, text_after_cursor):
        """Apply completion candidates to the input field."""
        # Store for cycling
        self._tab_prefix = prefix

        if len(candidates) == 1:
            self._apply_token_completion(prefix, candidates[0], text_after_cursor)
        else:
            # Try inserting common prefix first
            common = os.path.commonprefix(candidates)
            if common and common != token:
                self._apply_token_completion(prefix, common, text_after_cursor)
            else:
                # Multiple ambiguous matches: show them, cycle on next Tab
                self._tab_candidates = candidates
                self._tab_index = 0
                self._show_completion_options(candidates)
                self._apply_token_completion(prefix, candidates[0], text_after_cursor)

    def _split_last_token(self, text: str):
        """
        Split text into (prefix, last_token).
        prefix is everything before the last token including trailing space.
        last_token is the word being completed.
        
        Examples:
          "cd /n/r"       → ("cd ", "/n/r")
          "/con"          → ("", "/con")
          "/connect cla"  → ("/connect ", "cla")
          "ls  "          → ("ls  ", "")
          ""              → ("", "")
        """
        if not text:
            return ("", "")
        
        # If text ends with a space, there's no partial token yet
        if text.endswith(' '):
            return (text, "")
        
        # Find the last whitespace boundary
        last_space = text.rfind(' ')
        if last_space == -1:
            return ("", text)
        
        return (text[:last_space + 1], text[last_space + 1:])

    def _complete_macro_name(self, token: str) -> list:
        """Complete a /command name token."""
        # Token includes the leading /
        prefix = token[1:].lower() if token.startswith('/') else token.lower()
        matches = []
        for cmd in self._MACRO_COMMANDS:
            if cmd.startswith(prefix):
                matches.append(f"/{cmd}")
        # Also match known agent names as shortcuts
        for agent in sorted(self.known_agents):
            candidate = f"/{agent}"
            if agent.startswith(prefix) and candidate not in matches:
                matches.append(candidate)
        return sorted(matches)

    def _complete_agent_name(self, token: str) -> list:
        """Complete an agent name token."""
        return sorted(a for a in self.known_agents if a.startswith(token))

    def _complete_path(self, partial: str) -> list:
        """
        Complete a filesystem path token.
        
        Uses os.scandir() instead of os.listdir()+os.path.isdir() to
        minimise syscalls — critical on 9P/FUSE mounts where each stat
        is a network round-trip.  scandir returns d_type from the single
        readdir call so no extra stat per entry.
        """
        if not partial:
            partial = './'

        expanded = os.path.expanduser(partial)

        if partial.endswith('/'):
            directory = expanded
            name_prefix = ""
        else:
            directory = os.path.dirname(expanded) or '.'
            name_prefix = os.path.basename(expanded)

        matches = []
        try:
            # scandir is one readdir syscall — no per-entry stat
            with os.scandir(directory) as it:
                for entry in sorted(it, key=lambda e: e.name):
                    if entry.name.startswith('.') and not name_prefix.startswith('.'):
                        continue
                    if not entry.name.startswith(name_prefix):
                        continue

                    # Reconstruct path preserving user's directory prefix
                    if partial.endswith('/'):
                        candidate = partial + entry.name
                    else:
                        dir_part = partial[:len(partial) - len(name_prefix)] if name_prefix else partial
                        candidate = dir_part + entry.name

                    # entry.is_dir() uses cached d_type — no extra syscall
                    # on most filesystems.  Wrap in try for broken mounts.
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            candidate += '/'
                    except OSError:
                        pass

                    matches.append(candidate)
        except OSError:
            return []

        return matches

    def _apply_token_completion(self, prefix: str, completed_token: str, suffix: str):
        """Replace input with prefix + completed_token + suffix, cursor after token."""
        new_text = prefix + completed_token + suffix
        self.command_input.setPlainText(new_text)
        # Place cursor right after the completed token
        cursor = self.command_input.textCursor()
        cursor.setPosition(len(prefix) + len(completed_token))
        self.command_input.setTextCursor(cursor)
        # Update tab state for cycling detection
        self._tab_state_text = new_text[:len(prefix) + len(completed_token)]

    def _show_completion_options(self, candidates: list):
        """Display completion candidates in the terminal output."""
        display_items = []
        for c in candidates[:20]:
            # Show just the basename / last segment for readability
            if '/' in c:
                display_items.append(c.split('/')[-1] or c.split('/')[-2] + '/')
            else:
                display_items.append(c)

        if not display_items:
            return

        max_len = max(len(d) for d in display_items)
        col_width = max_len + 2
        cols = max(1, 60 // col_width)
        lines = []
        for i in range(0, len(display_items), cols):
            row = display_items[i:i + cols]
            lines.append("  ".join(item.ljust(col_width) for item in row))

        self.append_text("\n".join(lines) + "\n", self.C_INFO)

    # ------------------------------------------------------------------
    # Command submission & routing
    # ------------------------------------------------------------------

    def _submit_command(self):
        text = self.command_input.toPlainText().strip()
        if not text:
            return

        self.command_history.append(text)
        self.history_index = len(self.command_history)
        self.command_input.clear()

        self._route(text)

    def _route(self, text: str):
        # 1. Check for the toggle command
        if text.strip() == "$":
            self.terminal_mode = not self.terminal_mode
            status = "ENABLED" if self.terminal_mode else "DISABLED"
            color = self._active_shell_echo_color if self.terminal_mode else self.C_ERROR
            
            #self.append_text(f"\n*** Terminal Mode {status} ***\n", color)
            
            # Update placeholder for visual feedback
            if self.terminal_mode:
                self.command_input.setPlaceholderText("SHELL MODE active - Type $ to exit")
            else:
                placeholder = f"[{self.connected_agent}] " if self.connected_agent else "Enter command..."
                self.command_input.setPlaceholderText(placeholder)
                
            #self._update_input_style()
            self.animate_shadow_color(self.terminal_mode)
            return

        # 2. If in Terminal Mode, forward everything to shell
        if self.terminal_mode:
            if self._password_mode:
                self._password_mode = False
                self._execute_shell(text)  # no echo for passwords
            else:
                self._execute_shell(text, echo=True)
            return

        # 3. Standard routing
        if text.startswith('/'):
            self._echo(text, self.C_MACRO)
            self._handle_macro(text)
        elif text.startswith('>>>'):
            code = text[3:].strip()
            self._echo(f">>> {code}", self.C_PYTHON)
            if code:
                self.command_submitted.emit(code)
        elif text.startswith('$'):
            # One-off shell command
            self._execute_shell(text[1:].strip(), echo=True)
        else:
            # Prompt to agent
            self._echo(f">> {text}", self.C_USER)
            self._send_to_agent(text)

    # ------------------------------------------------------------------
    # Macro commands  (all filesystem-mediated)
    # ------------------------------------------------------------------

    def _handle_macro(self, text: str):
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        # ---- built-in macros (no arguments) ----
        dispatch = {
            'help':         lambda: self._show_help(),
            'cls':          lambda: self.clear_output(),
            'clear':        lambda: self._agent_ctl("clear"),
            'cancel':       lambda: self._agent_ctl("cancel"),
            'retry':        lambda: self._agent_ctl("retry"),
            'disconnect':   lambda: self._disconnect_agent(),
            'status':       lambda: self._show_status(),
            'list':         lambda: self._list_agents(),
            'ls':           lambda: self._list_agents(),
            'attachments':  lambda: self._show_attachments(),
            'restart':      lambda: self._restart_shell(),
            'setup':        lambda: self._setup_mounts(),
            'color':        lambda: self._open_color_picker(),
            'colors':       lambda: self._open_color_picker(),
            'pop':          lambda: self._pop_to_window(),
            'dock':         lambda: self._dock_to_scene(),
            'dark':         lambda: self._toggle_dark_mode_from_terminal(),
            'darkmode':     lambda: self._toggle_dark_mode_from_terminal(),
            'versions':     lambda: self._toggle_version_panel(),
            'version':      lambda: self._toggle_version_panel(),
            'ver':          lambda: self._toggle_version_panel(),
        }

        if cmd in dispatch:
            dispatch[cmd]()
            return

        # ---- /master [provider] [model] ----
        if cmd == 'master':
            self._setup_master(arg)
            return
        
        # ---- /coder [provider] [model] ----
        if cmd == 'coder':
            self._setup_coder(arg)
            return

        # ---- /tcoder [provider] [model] ----
        # Same as /coder, but routes the coder's per-machine output to a
        # specific terminal (terms/<id>/parse) rather than the workspace
        # scene/parse.  See _setup_tcoder for the routing rule.
        if cmd == 'tcoder':
            self._setup_tcoder(arg)
            return

        # ---- /av [voice] ----
        if cmd == 'av':
            self._setup_av(arg)
            return

        # ---- /av_gemini [voice] ----
        if cmd == 'av_gemini':
            self._setup_av_gemini(arg)
            return

        # ---- macros with arguments ----
        if cmd == 'connect':
            if not arg:
                self.append_text("Usage: /connect <agent>\n", self.C_ERROR)
            else:
                self._connect_agent(arg)
            return

        if cmd == 'delete':
            if not arg:
                self.append_text("Usage: /delete <agent>\n", self.C_ERROR)
            else:
                self._delete_agent(arg)
            return

        if cmd == 'acme':
            self._toggle_acme_panel()
            return

        if cmd == 'operator':
            self._toggle_operator_panel()
            return

        if cmd == 'scene':
            self._toggle_scene_panel()
            return

        if cmd == 'attach':
            # Parse: /attach <source> <destination>
            parts = arg.split(maxsplit=1)
            if len(parts) != 2:
                self.append_text("Usage: /attach <source> <destination>\n", self.C_ERROR)
                self.append_text("Example: /attach /n/mux/llm/claude/RIOA /n/mux/ws/scene/parse\n", self.C_INFO)
            else:
                self._add_attachment(parts[0], parts[1])
            return

        if cmd == 'mount':
            # /mount IP!Port name  →  9pfuse 'tcp!IP!Port' /n/name
            parts = arg.split(maxsplit=1)
            if len(parts) != 2:
                self.append_text("Usage: /mount <IP!Port> <name>\n", self.C_ERROR)
                self.append_text("Example: /mount 192.168.1.5!5640 llm2\n", self.C_INFO)
            else:
                self._mount_9p(parts[0], parts[1])
            return

        # ---- /signal on|off — bidirectional SignalBus subscribe/unsubscribe ----
        if cmd == 'signal':
            mode = arg.strip().lower()
            if mode not in ('on', 'off'):
                self.append_text("Usage: /signal on|off\n", self.C_ERROR)
                self.append_text("  on  — wire a full mesh of subscriptions between every machine in /n/ctl\n",
                                 self.C_INFO)
                self.append_text("  off — tear the mesh down\n",
                                 self.C_INFO)
            else:
                self._signal_toggle(mode == 'on')
            return

        if cmd == 'detach':
            if not arg:
                self.append_text("Usage: /detach <source>\n", self.C_ERROR)
            else:
                self._remove_attachment(arg)
            return

        if cmd == 'context':
            if not arg:
                self.append_text("Usage: /context <agent_name>\n", self.C_ERROR)
                self.append_text("Routes $RIO/CONTEXT -> $LLMFS/<agent>/history\n", self.C_INFO)
            else:
                self._add_context_route(arg.strip())
            return

        if cmd == 'system':
            if not arg:
                self._read_agent_file("system")
            else:
                self._write_agent_file("system", arg)
            return

        if cmd == 'provider':
            if not arg:
                self._agent_ctl("provider")
            else:
                self._agent_ctl(f"provider {arg}")
            return

        if cmd == 'use':
            # Quick provider+model switch: /use groq kimi-k2 or /use cerebras zai
            # Fuzzy-matches model names against the provider's model list
            self._use_provider_model(arg)
            return

        if cmd == 'model':
            if not arg:
                self._agent_ctl("model")
            else:
                self._agent_ctl(f"model {arg}")
            return

        if cmd == 'temperature':
            if not arg:
                self._agent_ctl("temperature")
            else:
                self._agent_ctl(f"temperature {arg}")
            return

        if cmd == 'history':
            self._show_agent_history()
            return

        if cmd == 'config':
            self._read_agent_file("config")
            return

        if cmd == 'errors':
            self._read_agent_file("errors")
            return

        # ---- /new <name> [provider] [model] [system] -> create + connect ----
        if cmd == 'new':
            if not arg:
                self.append_text("Usage: /new <agent> [provider] [model]\n", self.C_ERROR)
            else:
                parts = arg.split(None, 1)
                agent_name = parts[0]
                rest = parts[1] if len(parts) > 1 else None
                self._ensure_agent(agent_name, rest)
            return

        # ---- /peribus — connect / disconnect / inspect the mycelium layer ----
        if cmd == 'peribus':
            self._handle_peribus_command(arg)
            return

        # ---- /share — high-level publishing (text / file / dialog) ----
        if cmd == 'share':
            self._handle_share_command(arg)
            return

        # ---- unknown command ----
        self.append_text(f"Unknown command: /{cmd}\n", self.C_ERROR)
        self.append_text("Type /help for available commands\n", self.C_INFO)

    # ------------------------------------------------------------------
    # Attachment management (Plan 9 blocking style - no polling!)
    # ------------------------------------------------------------------

    def _add_attachment(self, source: str, destination: str, quiet: bool = False):
        """
        Central method for creating Plan 9 attachment routes.
        
        ALL route creation goes through here — /attach macro, /master,
        /coder, /av, operator panel — all funnel into this single method.
        
        Delegates to the shared RoutesManager ({rio_mount}/routes).
        Routes are accessible even without a terminal:
          cat {rio_mount}/routes
        
        Args:
            source:      Absolute or relative path to read from
            destination: Absolute or relative path to write to
            quiet:       If True, suppress terminal output (used by filesystem writes)
        """
        # Validate paths
        if not source or not destination:
            if not quiet:
                self.append_text("Both source and destination must be specified\n", self.C_ERROR)
            return

        # Expand relative paths
        if not source.startswith('/'):
            source = os.path.join(self.llmfs_mount, source)
        if not destination.startswith('/'):
            destination = os.path.join(self.llmfs_mount, destination)
        
        if self._routes_manager:
            self._routes_manager.add_route(source, destination)
        else:
            # Fallback: create attachment directly (no manager yet)
            if source in self.attachments:
                self.attachments[source].stop()
            attachment = Plan9Attachment(source, destination)
            attachment.start()
            # Can't store without manager — warn
            if not quiet:
                self.append_text("WARNING: No routes manager — route not persisted\n", self.C_ERROR)
        
        if not quiet:
            self.append_text(f"Attached: {source} -> {destination}\n", self.C_SUCCESS)

    def _remove_attachment(self, source: str):
        """Remove an automatic routing."""
        # Expand relative path
        if not source.startswith('/'):
            source = os.path.join(self.llmfs_mount, source)

        if self._routes_manager:
            if self._routes_manager.remove_route(source):
                self.append_text(f"Detached: {source}\n", self.C_SUCCESS)
            else:
                self.append_text(f"No attachment found for: {source}\n", self.C_ERROR)
        elif source in self.attachments:
            self.attachments[source].stop()
            del self.attachments[source]
            self.append_text(f"Detached: {source}\n", self.C_SUCCESS)
        else:
            self.append_text(f"No attachment found for: {source}\n", self.C_ERROR)

    def _show_attachments(self):
        """Display all active attachments."""
        routes = self._routes_manager.list_routes() if self._routes_manager else []
        
        if not routes:
            self.append_text("No active attachments\n", self.C_INFO)
            return

        self.append_text(f"Active attachments ({len(routes)}):\n", self.C_INFO)
        for source, destination, running in routes:
            status = "running" if running else "stopped"
            self.append_text(f"  {source}\n", self.C_DEFAULT)
            self.append_text(f"    -> {destination} [{status}]\n", self.C_SUCCESS)

    def _add_context_route(self, agent_name: str):
        """
        Route the workspace CONTEXT file to an agent's history.

        Creates a Plan 9 attachment:
            {rio_mount}/CONTEXT  ->  {llmfs_mount}/{agent_name}/history

        This feeds the workspace context (scene state, selections, etc.)
        into the agent's conversation history so it can reason about
        the current environment.
        """
        agent_dir = os.path.join(self.llmfs_mount, agent_name)
        if not os.path.isdir(agent_dir):
            self.append_text(f"Agent '{agent_name}' not found\n", self.C_ERROR)
            self.append_text("Create it first: /{}\n".format(agent_name), self.C_INFO)
            return

        source = f"{self.rio_mount}/CONTEXT"
        destination = os.path.join(agent_dir, "history")

        self._add_attachment(source, destination)
        self.append_text(f"  Context route: $RIO/CONTEXT -> ${agent_name}/history\n", self.C_SUCCESS)

    # ------------------------------------------------------------------
    # Master Agent Setup
    # ------------------------------------------------------------------

    # Colour for master-specific output
    C_MASTER = "rgba(180, 100, 255, 255)"

    @property
    def MASTER_SYSTEM_PROMPT(self):
        return f"""You are MASTER, an autonomous coordinating AI agent operating inside a Plan 9-inspired filesystem environment.

## YOUR ENVIRONMENT

You exist as an agent in LLMFS. Everything is a file:
- Your output streams to the user's terminal
- You can execute shell commands by writing ```bash blocks in your responses
- You can read files, run programs, inspect results — all through bash
- The Rio display server scene is at $RIO/scene/parse (write Python code there to render)
- Other agents live under $LLMFS/ — you can create them, write to their input, read their OUTPUT

## SHELL VARIABLES (pre-seeded, shared across all commands)

Your bash blocks run in a persistent shell that shares state. These variables are already set:

    $LLMFS        → {self.llmfs_mount}                         (LLMFS mount root)
    $RIO          → {self.rio_mount}                            (Rio display server mount)
    $master       → {self.llmfs_mount}/master                    (your own agent dir)

When you create a new agent, a variable is automatically seeded:
    echo 'new coder' > $LLMFS/ctl
    # Now $coder is set to {self.llmfs_mount}/coder

So you can write:
    echo 'prompt' > $coder/input
    cat $coder/OUTPUT
    cat $coder/ctl

Variables persist across all your bash blocks within this session. You can also set your own:
    RESULT=$(cat $coder/OUTPUT)

## EXECUTING COMMANDS

To run a shell command, emit a fenced bash block. It will be extracted and executed automatically:

```bash
ls $LLMFS/
```

The command output (stdout AND stderr) will appear in the terminal AND is captured in $term/stdout.

To read back the output of the last command you ran:
```bash
cat $term/stdout
```

This blocks until output settles, returns it, then EOF. Use this to inspect results programmatically.

IMPORTANT: After running a command, ALWAYS check the result. Do not assume success. Verify.

## COORDINATING OTHER AGENTS

You can spawn specialist agents and delegate work:

```bash
echo 'new coder' > $LLMFS/ctl
echo 'You are a Python coding expert. Write clean, production code.' > $coder/system
echo 'Write a function to sort a list of dicts by key' > $coder/input
```

Then wait and check their output:
```bash
cat $coder/OUTPUT
```

Then check their work. Read the output, evaluate it, ask for corrections if needed. You are responsible for quality.

## WRITING TO THE SCENE

To render visual content on the Rio display, write Python code to the scene parse file:
```bash
cat > $RIO/scene/parse << 'PYEOF'
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QFont
label = QLabel("Hello from Master")
label.setFont(QFont("Arial", 24))
label.setStyleSheet("color: white; background: rgba(0,0,0,150); padding: 20px; border-radius: 10px;")
label.move(100, 100)
scene_manager.register_widget("master_label", label, x=100, y=100)
PYEOF
```

## YOUR WORKFLOW

1. Receive user request
2. Break it down into steps
3. Execute each step via bash blocks
4. CHECK THE RESULT of each step — read output, verify files exist, test code
5. If something fails, diagnose and fix it
6. Iterate until the result is correct
7. Report back with a summary

## RULES

- ALWAYS verify your work. After every action, check the result.
- Be methodical. Show your plan before executing.
- Use bash blocks for ALL filesystem and shell operations.
- Use the pre-seeded shell variables ($LLMFS, $RIO, $master, $coder, etc.) — never hardcode paths.
- When delegating to other agents, always read back and validate their OUTPUT.
- If an agent produces bad output, give it corrective feedback and retry.
- Keep the user informed of progress.
- You have access to the full Unix toolset: grep, sed, awk, find, curl, python3, etc.
- You are autonomous. Do not ask the user for permission to proceed unless genuinely ambiguous.

## SELF-ROUTING (feedback loop)

You can route data back to yourself for a follow-up turn by piping to $master/input.
This lets you chain: execute a command, capture its output, and send it back as your
next prompt so you can react to it.

```bash
echo "Here are the results: $(cat $term/stdout)" > $master/input
```

CRITICAL RULE: You may self-route AT MOST ONCE per exchange. Do NOT create infinite
loops. After one self-route, you MUST stop and wait for the result or report to the user.
Pattern: act → observe → self-route once → act on feedback → report.
"""

    def _ensure_splitter(self):
        """
        Ensure the shared QSplitter exists with terminal_frame inside it.
        Called once on first panel open; subsequent calls are a no-op.
        Returns the splitter.
        """
        if hasattr(self, '_splitter') and self._splitter is not None:
            return self._splitter

        main_layout = self.layout()
        main_layout.removeWidget(self.terminal_frame)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #666666;
                width: 3px;
            }
        """)
        self._splitter.addWidget(self.terminal_frame)
        main_layout.addWidget(self._splitter)
        # Track which panel is currently in the splitter
        self._active_panel = None
        return self._splitter

    def _show_panel_in_splitter(self, panel, sizes):
        """
        Show a panel in the splitter, removing any other active panel first.
        This avoids QSplitter setSizes index confusion with hidden widgets.
        """
        splitter = self._ensure_splitter()

        # Remove the currently active panel from the splitter (if different)
        if self._active_panel is not None and self._active_panel is not panel:
            self._active_panel.setParent(None)
            self._active_panel.hide()

        # Add the new panel if it's not already in the splitter
        if panel.parent() is not splitter:
            splitter.addWidget(panel)

        panel.show()
        splitter.setSizes(sizes)
        self._active_panel = panel

    def _hide_active_panel(self):
        """Remove the active panel from the splitter."""
        if self._active_panel is not None:
            self._active_panel.setParent(None)
            self._active_panel.hide()
            self._active_panel = None

    def _toggle_acme_panel(self):
        """
        /acme - Toggle ACME editor panel as a splitter pane inside the terminal.
        
        First call: creates Acme and shows it in the splitter.
        Subsequent calls: toggle Acme visibility.
        """
        if self.acme_panel is None:
            # Create ACME instance - registers itself at /n/rio/acme/
            self.acme = Acme(
                llmfs_mount=self.llmfs_mount,
                rio_mount=self.rio_mount,
                p9_host=self.p9_host,
                p9_port=self.p9_port,
            )
            self.acme_panel = self.acme  # Reference for toggle/cleanup
            self._show_panel_in_splitter(self.acme_panel, [400, 600])
            self.append_text("✓ ACME panel opened (windows at /n/rio/acme/)\n", self.C_SUCCESS)
        else:
            # Toggle: if it's the active panel, hide it; otherwise show it
            if self._active_panel is self.acme_panel:
                self._hide_active_panel()
                self.append_text("ACME panel hidden\n", self.C_INFO)
            else:
                self._show_panel_in_splitter(self.acme_panel, [400, 600])
                self.append_text("ACME panel shown\n", self.C_SUCCESS)

    def _toggle_version_panel(self):
        """
        /versions - Toggle Version Manager panel as a splitter pane.

        First call: creates VersionPanel and shows it in the splitter.
        Subsequent calls: toggle visibility.
        """
        if self.version_panel is None:
            rio_mount = self.rio_mount
            self.version_panel = VersionPanel(rio_mount=rio_mount)
            self._show_panel_in_splitter(self.version_panel, [650, 350])
            self.append_text("✓ Version panel opened\n", self.C_SUCCESS)
        else:
            if self._active_panel is self.version_panel:
                self._hide_active_panel()
                self.append_text("Version panel hidden\n", self.C_INFO)
            else:
                self._show_panel_in_splitter(self.version_panel, [650, 350])
                self.version_panel.refresh()
                self.append_text("Version panel shown\n", self.C_SUCCESS)

    def _toggle_operator_panel(self):
        """
        /operator - Toggle Operator panel as a splitter pane inside the terminal.
        
        First call: creates OperatorPanel and shows it in the splitter.
        Subsequent calls: toggle visibility.
        """
        if self.operator_panel is None:
            self.operator_panel = OperatorPanel(
                llmfs_mount=self.llmfs_mount,
                rio_mount=self.rio_mount,
                terminal_widget=self
            )
            self._show_panel_in_splitter(self.operator_panel, [400, 600])
            self.append_text("✓ Operator panel opened\n", self.C_SUCCESS)
        else:
            if self._active_panel is self.operator_panel:
                self._hide_active_panel()
                self.append_text("Operator panel hidden\n", self.C_INFO)
            else:
                self._show_panel_in_splitter(self.operator_panel, [400, 600])
                self.append_text("Operator panel shown\n", self.C_SUCCESS)

    def get_or_create_scene_panel(self) -> 'TerminalScenePanel':
        """
        Return this terminal's TerminalScenePanel, creating it on first call.

        Used both by the slash command (/scene) and by the filesystem
        ParseFile attached to /n/rioa/terms/<term_id>/parse, which needs
        the panel's executor to run user code.

        Creating the panel does NOT automatically show it in the splitter
        — call _toggle_scene_panel() (or write to /parse) to make it visible.
        """
        if self.scene_panel is None:
            self.scene_panel = TerminalScenePanel(self.term_id, parent=self)
        return self.scene_panel

    def _toggle_scene_panel(self):
        """
        /scene - Toggle the per-terminal live UI panel.

        First call: creates the TerminalScenePanel and shows it.
        Subsequent calls: toggle visibility. The panel's QGraphicsScene
        is the target of code written to /n/rioa/terms/<term_id>/parse.
        """
        first_time = self.scene_panel is None
        panel = self.get_or_create_scene_panel()

        if first_time:
            self._show_panel_in_splitter(panel, [400, 600])
            self.append_text(
                f"✓ Scene panel opened — write to "
                f"/n/rioa/terms/{self.term_id}/parse to draw on it\n",
                self.C_SUCCESS,
            )
        else:
            if self._active_panel is panel:
                self._hide_active_panel()
                self.append_text("Scene panel hidden\n", self.C_INFO)
            else:
                self._show_panel_in_splitter(panel, [400, 600])
                self.append_text("Scene panel shown\n", self.C_SUCCESS)

    @Slot()
    def ensure_scene_panel_visible(self):
        """
        Ensure the scene panel exists, and show it if this is the first
        time it's being created. Called by the filesystem when code is
        first executed against /n/rioa/terms/<term_id>/parse, so the user
        sees the result the first time without running /scene.

        If the user has explicitly hidden the panel (via /scene toggle),
        subsequent writes do NOT pop it back open — they still update the
        scene, the user just has to /scene to view it again.

        Decorated with @Slot so it can be invoked from the asyncio
        thread via QMetaObject.invokeMethod(..., BlockingQueuedConnection).
        """
        first_time = self.scene_panel is None
        panel = self.get_or_create_scene_panel()
        if first_time:
            self._show_panel_in_splitter(panel, [400, 600])

    def _setup_master(self, arg: str = ""):
        """
        /master [provider] [model]

        Creates the master autonomous agent with:
        1. A master agent with the master system prompt
        2. A plumbing rule: ```bash blocks → 'BASH' supplementary output file
        3. A MasterBashReader that reads $master/BASH and executes in terminal
        4. Output streaming connected to the terminal
        """
        if self._master_active:
            self.append_text("Master agent already active. Use /disconnect then /master to restart.\n", self.C_ERROR)
            return

        parts = arg.split() if arg else []
        provider = parts[0] if len(parts) > 0 else None
        model = parts[1] if len(parts) > 1 else None

        agent_name = "master"
        ctl_path = os.path.join(self.llmfs_mount, "ctl")
        agent_dir = os.path.join(self.llmfs_mount, agent_name)

        self.append_text("\n", self.C_MASTER)
        self.append_text("╔══════════════════════════════════════════╗\n", self.C_MASTER)
        self.append_text("║     MASTER AGENT — Initializing...       ║\n", self.C_MASTER)
        self.append_text("╚══════════════════════════════════════════╝\n", self.C_MASTER)

        # Step 1: Create the agent
        if not os.path.isdir(agent_dir):
            try:
                create_cmd = f"new {agent_name}"
                if provider:
                    create_cmd += f" {provider}"
                if model:
                    create_cmd += f" {model}"
                with open(ctl_path, 'w') as f:
                    f.write(create_cmd + "\n")
                self.append_text(f"  ✓ Agent '{agent_name}' created\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ✗ Failed to create agent: {e}\n", self.C_ERROR)
                return
        else:
            self.append_text(f"  • Agent '{agent_name}' already exists\n", self.C_INFO)

        # Step 2: Write system prompt
        try:
            system_path = os.path.join(agent_dir, "system")
            # Load system prompt from file
            prompt_file = "./systems/master.md"
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    system_prompt = f.read()
            else:
                # Fallback to embedded prompt
                system_prompt = self.MASTER_SYSTEM_PROMPT
            
            with open(system_path, 'w') as f:
                f.write(system_prompt)
            self.append_text("  ✓ System prompt configured\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ✗ Failed to set system prompt: {e}\n", self.C_ERROR)
            return

        # Step 3: Set model if specified
        if model:
            try:
                ctl_agent = os.path.join(agent_dir, "ctl")
                with open(ctl_agent, 'w') as f:
                    f.write(f"model {model}\n")
                self.append_text(f"  ✓ Model set to {model}\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ⚠ Could not set model: {e}\n", self.C_ERROR)

        # Step 4: Add plumbing rule for bash extraction
        # Pattern: ```bash\n<code>\n``` → extracts code into 'bash' supplementary output
        try:
            rules_path = os.path.join(agent_dir, "rules")
            bash_rule = r"```(?P<bash>\S*)\n(?P<code>.*?)```" + " -> {bash}"
            with open(rules_path, 'w') as f:
                f.write(bash_rule + "\n")
            self.append_text("  ✓ Plumbing rule: ```bash → $master/BASH\n", self.C_SUCCESS)
            # Track supplementary output file
            self.known_supplementary.setdefault(agent_name, set()).add("BASH")
        except Exception as e:
            self.append_text(f"  ✗ Failed to set plumbing rule: {e}\n", self.C_ERROR)
            return

        # Step 5: Connect terminal output stream (also seeds $master shell var)
        self._connect_agent(agent_name)

        # Step 5b: Seed shell variables for all existing agents
        agents_dir = self.llmfs_mount
        if os.path.isdir(agents_dir):
            for name in os.listdir(agents_dir):
                if os.path.isdir(os.path.join(agents_dir, name)):
                    self._seed_agent_variable(name)
                    self.known_agents.add(name)

        # Step 5c: Seed $term variable so agent can reference this terminal's fs
        self._execute_shell_raw(
            f'export term="{self.rio_mount}/terms/{self.term_id}"'
        )

        # Step 6: Route $master/BASH → $term/stdin via unified attachment
        # This replaces the old MasterBashReader thread — same semantics
        # (while true; cat $master/BASH > $term/stdin; done) but now visible
        # in {rio_mount}/routes and the operator panel.
        master_bash = os.path.join(self.llmfs_mount, agent_name, "BASH")
        term_stdin = f"{self.rio_mount}/terms/{self.term_id}/stdin"
        self._add_attachment(master_bash, term_stdin)
        self.append_text(f"  ✓ Route: $master/BASH → $term/stdin\n", self.C_SUCCESS)

        self._master_active = True

        self.append_text("\n", self.C_MASTER)
        self.append_text("  Master agent ready. Type your request.\n", self.C_SUCCESS)
        self.append_text(f"  $term = {self.rio_mount}/terms/{self.term_id}\n", self.C_INFO)
        self.append_text("  Bash blocks auto-execute. /cancel to stop, /disconnect to detach.\n\n", self.C_INFO)

    def _start_master_bash_reader(self, agent_name: str):  # DEAD CODE — superseded, kept for reference
        """Start the background thread that reads $master/BASH and executes commands."""
        if self._master_bash_reader:
            self._master_bash_reader.stop()
            self._master_bash_reader.wait(2000)

        self._master_bash_reader = MasterBashReader(
            agent_path=f"{agent_name}",
            host=self.p9_host,
            port=self.p9_port,
            auth_token=self.p9_token,
        )
        self._master_bash_reader.command_ready.connect(self._on_master_bash_command)
        self._master_bash_reader.error_occurred.connect(self._on_master_bash_error)
        self._master_bash_reader.start()

        self.append_text("  ✓ Bash router active (raw 9P blocking read)\n", self.C_SUCCESS)

    def _on_master_bash_command(self, command: str):
        """
        Execute a bash command from the master agent.

        Routes through the terminal's stdout capture so the full
        pipeline works:
          $master/BASH → MasterBashReader → _execute_shell()
            → PTY exec → _on_shell_output → $term/stdout.capture()
            → debounce → mark_ready → cat $term/stdout returns output

        If the terminal filesystem isn't registered yet, falls back
        to direct _execute_shell.

        SANDBOX: LLM commands are validated before execution.
          - Read anywhere is allowed
          - Writes only under /n/
          - Destructive ops (rm, dd, etc.) always blocked
        """
        # ── Sandbox gate ────────────────────────────────────────────
        ok, reason = _sandbox_check(command)
        if not ok:
            self.append_text(f"⛔ blocked: {command}\n", self.C_ERROR)
            self.append_text(f"   reason: {reason}\n", self.C_ERROR)
            # Feed rejection back to term/stdout so the LLM sees it
            if self._term_dir is not None:
                self._term_dir.stdout_file.feed_error(
                    f"SANDBOX BLOCKED: {reason}\n"
                )
            return

        # ── Execute (passed sandbox) ────────────────────────────────
        self.append_text(f"⚡ {command}\n", self.C_MASTER)

        if self._term_dir is not None:
            # Arm term/stdout for capture, then execute
            self._term_dir.stdout_file.start_capture()
            self._execute_shell(command)
        else:
            # Fallback: direct execution (no output capture)
            self._execute_shell(command)

        # Detect agent creation: echo 'new <n>' > .../ctl
        import re as _re
        m = _re.search(r"echo\s+['\"\"]?new\s+(\w+)", command)
        if m:
            new_agent = m.group(1)
            QTimer.singleShot(500, lambda: self._seed_agent_variable(new_agent))

    def _on_master_bash_error(self, msg: str):
        """Handle errors from the master bash reader."""
        self.append_text(f"[master/BASH] {msg}\n", self.C_ERROR)

    def _stop_master(self):
        """Stop the master agent's bash route and reader."""
        # Stop the route attachment ($master/BASH → $term/stdin)
        master_bash = os.path.join(self.llmfs_mount, "master", "BASH")
        if self._routes_manager:
            self._routes_manager.remove_route(master_bash)
        
        # Also stop the legacy MasterBashReader if still present
        if self._master_bash_reader:
            self._master_bash_reader.stop()
            self._master_bash_reader.wait(2000)
            self._master_bash_reader = None
        self._master_active = False
    
    def _setup_coder(self, arg: str = ""):
        """
        /coder [provider] [model]

        Creates the coder specialist agent with:
        1. A coder agent with register on + history off
        2. System prompt from ./systems/coder.md
        3. Auto-registration creates rules for every mounted machine
        4. Per-machine context routing:
           - $workspace/CONTEXT -> $coder/<MACHINE> (context injection)
           - $coder/<MACHINE>   -> $workspace/scene/parse (code output)
        5. The "llm" machine is always excluded from registration
        """
        parts = arg.split() if arg else []
        provider = parts[0] if len(parts) > 0 else None
        model = parts[1] if len(parts) > 1 else None

        agent_name = "coder"
        ctl_path = os.path.join(self.llmfs_mount, "ctl")
        agent_dir = os.path.join(self.llmfs_mount, agent_name)

        self.append_text("\n", self.C_INFO)
        self.append_text("╔══════════════════════════════════════════╗\n", self.C_INFO)
        self.append_text("║     CODER AGENT — Initializing...        ║\n", self.C_INFO)
        self.append_text("╚══════════════════════════════════════════╝\n", self.C_INFO)

        # Step 1: Create the agent
        if not os.path.isdir(agent_dir):
            try:
                create_cmd = f"new {agent_name}"
                if provider:
                    create_cmd += f" {provider}"
                if model:
                    create_cmd += f" {model}"
                with open(ctl_path, 'w') as f:
                    f.write(create_cmd + "\n")
                self.append_text(f"  ✓ Agent '{agent_name}' created\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ✗ Failed to create agent: {e}\n", self.C_ERROR)
                return
        else:
            self.append_text(f"  • Agent '{agent_name}' already exists\n", self.C_INFO)

        # Step 2: Write system prompt from file
        try:
            system_path = os.path.join(agent_dir, "system")
            prompt_file = "./systems/coder.md"

            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    system_prompt = f.read()
            else:
                self.append_text(f"  ⚠ Warning: {prompt_file} not found, using default\n", self.C_ERROR)
                system_prompt = "You are a coding specialist. Write clean Python code for the Rio display server."

            with open(system_path, 'w') as f:
                f.write(system_prompt)
            self.append_text("  ✓ System prompt configured\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ✗ Failed to set system prompt: {e}\n", self.C_ERROR)
            return

        # Step 3: Set model if specified
        if model:
            try:
                ctl_agent = os.path.join(agent_dir, "ctl")
                with open(ctl_agent, 'w') as f:
                    f.write(f"model {model}\n")
                self.append_text(f"  ✓ Model set to {model}\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ⚠ Could not set model: {e}\n", self.C_ERROR)

        # Step 4: Enable machine registration + disable history
        # register on: auto-creates plumbing rules for every mounted machine
        # history off: only the latest message + system context is sent
        # or
        # max_history = 2
        ctl_agent = os.path.join(agent_dir, "ctl")
        try:
            with open(ctl_agent, 'w') as f:
                f.write("register on\n")
            self.append_text("  ✓ Machine registration enabled\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ⚠ Could not enable registration: {e}\n", self.C_ERROR)

        try:
            with open(ctl_agent, 'w') as f:
                #f.write("history off\n")
                f.write("max_history 5\n")
            self.append_text("  ✓ History disabled (stateless mode)\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ⚠ Could not disable history: {e}\n", self.C_ERROR)

        # Step 5: Connect terminal output stream and seed $coder variable
        self._connect_agent(agent_name)
        self._seed_agent_variable(agent_name)

        # Step 6: Discover mounted machines and set up bidirectional routes
        self._setup_coder_workspace_routes(agent_name, agent_dir)

        self.append_text("\n", self.C_INFO)
        self.append_text("  Coder agent ready. Type your coding request.\n", self.C_SUCCESS)
        self.append_text("  Machine context auto-injected into system prompt.\n", self.C_INFO)
        self.append_text("  Code blocks tagged with machine names auto-route.\n", self.C_INFO)
        self.append_text("  /cancel to stop, /disconnect to detach.\n\n", self.C_INFO)

    def _setup_coder_workspace_routes(self, agent_name: str, agent_dir: str):
        """
        Set up bidirectional routes between coder and all registered machines.

        For each machine that isn't "llm":
          1. Route its CONTEXT to the coder's supplementary file:
             $workspace/CONTEXT -> $coder/<MACHINE>
             (this writes context into the agent's system prompt)
          2. Route the coder's supplementary output back:
             $coder/<MACHINE> -> $workspace/scene/parse
             (extracted code blocks auto-execute in the workspace)

        Machine discovery: reads from the LLM's own ctl file (the
        'machines' line), which is local and never blocks.  We NEVER
        stat/listdir the mux root or walk into other backends — those
        can hit blocking files or unreachable servers and freeze the UI.
        """
        # Read machine list from the LLM's own ctl (local, non-blocking)
        # The ctl status includes a line like: "machines david alice"
        machines = []
        llm_ctl = os.path.join(self.llmfs_mount, "ctl")
        try:
            with open(llm_ctl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("machines "):
                        rest = line[len("machines "):].strip()
                        if rest and rest != "(none)":
                            machines = rest.split()
                        break
        except Exception:
            pass

        # The mux root is the parent of our llmfs mount
        # e.g. if llmfs_mount is /n/mux/llm, mux_root is /n/mux
        mux_root = os.path.dirname(self.llmfs_mount)

        if not machines:
            self.append_text("  • No machines registered (routes will be set up when machines connect)\n", self.C_INFO)
            return

        for machine in machines:
            machine_upper = machine.upper()
            workspace_dir = os.path.join(mux_root, machine)

            # Track supplementary output
            self.known_supplementary.setdefault(agent_name, set()).add(machine_upper)

            # Route 1: workspace CONTEXT -> coder's supplementary file
            # We don't check if the path exists — the attachment subprocess
            # will retry via its while-true loop until the file appears.
            context_source = os.path.join(workspace_dir, "CONTEXT")
            context_dest = os.path.join(agent_dir, machine_upper)

            try:
                self._add_attachment(context_source, context_dest, quiet=True)
                self.append_text(
                    f"  ✓ Context: ${machine}/CONTEXT → $coder/{machine_upper}\n",
                    self.C_SUCCESS
                )
            except Exception as e:
                self.append_text(
                    f"  ⚠ Context route for {machine} failed: {e}\n",
                    self.C_ERROR
                )

            # Route 2: coder's supplementary output -> workspace scene/parse
            code_source = os.path.join(agent_dir, machine_upper)
            code_dest = os.path.join(workspace_dir, "scene", "parse")

            try:
                self._add_attachment(code_source, code_dest, quiet=True)
                self.append_text(
                    f"  ✓ Output: $coder/{machine_upper} → ${machine}/scene/parse\n",
                    self.C_SUCCESS
                )
            except Exception as e:
                self.append_text(
                    f"  ⚠ Output route for {machine} failed: {e}\n",
                    self.C_ERROR
                )

    # ------------------------------------------------------------------
    # /tcoder — coder agent with per-terminal output routing
    # ------------------------------------------------------------------

    def _setup_tcoder(self, arg: str = ""):
        """
        /tcoder [provider] [model]

        Identical to /coder in every respect — same agent name ("coder"),
        same system prompt, same register/history flags, same context
        injection (Route 1: $workspace/CONTEXT -> $coder/<MACHINE>) —
        except that Route 2 sends the coder's per-machine output to a
        specific terminal's inline stream rather than the machine-wide
        scene/parse:

            $coder/<MACHINE>  ->  /n/<machine>/terms/<term_id>/inline

        Where <term_id> is chosen per-machine:
          - If <machine> is OUR machine (basename(self.rio_mount)),
            <term_id> is THIS terminal's own self.term_id.
          - If <machine> is a remote machine, <term_id> is the first
            terminal found under /n/<machine>/terms/ (lexicographic).
            If none is found, the route is skipped (logged as a warning).

        Use this when you want code blocks tagged with a machine name to
        land in a single terminal's parse stream rather than the
        workspace-wide one — useful for keeping output scoped to one
        scene panel per machine.
        """
        parts = arg.split() if arg else []
        provider = parts[0] if len(parts) > 0 else None
        model = parts[1] if len(parts) > 1 else None

        # Re-use the existing coder agent name so $coder is unchanged
        # and the per-machine supplementary files line up.
        agent_name = "coder"
        ctl_path = os.path.join(self.llmfs_mount, "ctl")
        agent_dir = os.path.join(self.llmfs_mount, agent_name)

        self.append_text("\n", self.C_INFO)
        self.append_text("╔══════════════════════════════════════════╗\n", self.C_INFO)
        self.append_text("║  TCODER (per-terminal) — Initializing... ║\n", self.C_INFO)
        self.append_text("╚══════════════════════════════════════════╝\n", self.C_INFO)

        # Step 1: Create the agent (idempotent — reuses if it exists)
        if not os.path.isdir(agent_dir):
            try:
                create_cmd = f"new {agent_name}"
                if provider:
                    create_cmd += f" {provider}"
                if model:
                    create_cmd += f" {model}"
                with open(ctl_path, 'w') as f:
                    f.write(create_cmd + "\n")
                self.append_text(f"  ✓ Agent '{agent_name}' created\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ✗ Failed to create agent: {e}\n", self.C_ERROR)
                return
        else:
            self.append_text(f"  • Agent '{agent_name}' already exists\n", self.C_INFO)

        # Step 2: Write system prompt from file
        try:
            system_path = os.path.join(agent_dir, "system")
            prompt_file = "./systems/coder.md"

            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    system_prompt = f.read()
            else:
                self.append_text(f"  ⚠ Warning: {prompt_file} not found, using default\n", self.C_ERROR)
                system_prompt = "You are a coding specialist. Write clean Python code for the Rio display server."

            with open(system_path, 'w') as f:
                f.write(system_prompt)
            self.append_text("  ✓ System prompt configured\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ✗ Failed to set system prompt: {e}\n", self.C_ERROR)
            return

        # Step 3: Set model if specified
        if model:
            try:
                ctl_agent = os.path.join(agent_dir, "ctl")
                with open(ctl_agent, 'w') as f:
                    f.write(f"model {model}\n")
                self.append_text(f"  ✓ Model set to {model}\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ⚠ Could not set model: {e}\n", self.C_ERROR)

        # Step 4: Enable machine registration + cap history (same as /coder)
        ctl_agent = os.path.join(agent_dir, "ctl")
        try:
            with open(ctl_agent, 'w') as f:
                f.write("register on\n")
            self.append_text("  ✓ Machine registration enabled\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ⚠ Could not enable registration: {e}\n", self.C_ERROR)

        try:
            with open(ctl_agent, 'w') as f:
                f.write("max_history 5\n")
            self.append_text("  ✓ History capped at 5\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ⚠ Could not cap history: {e}\n", self.C_ERROR)

        # Step 5: Connect terminal output stream and seed $coder variable
        self._connect_agent(agent_name)
        self._seed_agent_variable(agent_name)

        # Step 6: Set up bidirectional routes — but Route 2 differs.
        self._setup_tcoder_workspace_routes(agent_name, agent_dir)

        self.append_text("\n", self.C_INFO)
        self.append_text("  Tcoder agent ready. Type your coding request.\n", self.C_SUCCESS)
        self.append_text("  Output routed to per-terminal parse streams.\n", self.C_INFO)
        self.append_text("  /cancel to stop, /disconnect to detach.\n\n", self.C_INFO)

    def _setup_tcoder_workspace_routes(self, agent_name: str, agent_dir: str):
        """
        Like _setup_coder_workspace_routes, but Route 2 targets a single
        terminal's inline stream instead of the workspace-wide scene/parse.

        Per-machine destination selection:
          - Local machine (basename(self.rio_mount)) → /n/<machine>/terms/<self.term_id>/inline
          - Remote machine                           → /n/<machine>/terms/<first_term>/inline
            where <first_term> is the lexicographically first entry in
            /n/<machine>/terms/.  If the directory is empty/unreadable,
            the route is skipped with a warning.

        We never block the UI by waiting for remote terms/ to populate —
        a single readdir is best-effort and skipped on failure.
        """
        # ---- machine discovery (identical to /coder) ----
        machines = []
        llm_ctl = os.path.join(self.llmfs_mount, "ctl")
        try:
            with open(llm_ctl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("machines "):
                        rest = line[len("machines "):].strip()
                        if rest and rest != "(none)":
                            machines = rest.split()
                        break
        except Exception:
            pass

        mux_root = os.path.dirname(self.llmfs_mount)

        # Identify our local machine by the rio mount's basename.
        # rio_mount is something like /n/rioa or /n/mux/rioa → "rioa".
        local_machine = os.path.basename(self.rio_mount.rstrip("/")) if self.rio_mount else None

        if not machines:
            self.append_text("  • No machines registered (routes will be set up when machines connect)\n", self.C_INFO)
            return

        for machine in machines:
            machine_upper = machine.upper()
            workspace_dir = os.path.join(mux_root, machine)

            # Track supplementary output (same as /coder)
            self.known_supplementary.setdefault(agent_name, set()).add(machine_upper)

            # Route 1: workspace CONTEXT -> coder's supplementary file
            #   (unchanged from /coder)
            context_source = os.path.join(workspace_dir, "CONTEXT")
            context_dest = os.path.join(agent_dir, machine_upper)

            try:
                self._add_attachment(context_source, context_dest, quiet=True)
                self.append_text(
                    f"  ✓ Context: ${machine}/CONTEXT → $coder/{machine_upper}\n",
                    self.C_SUCCESS
                )
            except Exception as e:
                self.append_text(
                    f"  ⚠ Context route for {machine} failed: {e}\n",
                    self.C_ERROR
                )

            # ---- Route 2 (THE DIFFERENCE): per-terminal parse ----
            # Pick a term_id for this machine.
            term_id = None
            if local_machine and machine == local_machine:
                # Our machine: use this terminal's own id.
                term_id = self.term_id
                term_source = "this terminal"
            else:
                # Remote machine: pick the first term found under
                # /n/<machine>/terms/ (lexicographic).  Best-effort —
                # if the readdir fails or the directory is empty, skip.
                terms_dir = os.path.join(workspace_dir, "terms")
                try:
                    entries = sorted(
                        e for e in os.listdir(terms_dir)
                        if os.path.isdir(os.path.join(terms_dir, e))
                    )
                except OSError as e:
                    self.append_text(
                        f"  ⚠ Output route for {machine}: cannot read {terms_dir} ({e})\n",
                        self.C_ERROR
                    )
                    continue
                if not entries:
                    self.append_text(
                        f"  ⚠ Output route for {machine}: no terminals under {terms_dir}\n",
                        self.C_ERROR
                    )
                    continue
                term_id = entries[0]
                term_source = f"first term in {terms_dir}"

            code_source = os.path.join(agent_dir, machine_upper)
            code_dest = os.path.join(workspace_dir, "terms", term_id, "inline")

            try:
                self._add_attachment(code_source, code_dest, quiet=True)
                self.append_text(
                    f"  ✓ Output: $coder/{machine_upper} → "
                    f"${machine}/terms/{term_id}/inline  [{term_source}]\n",
                    self.C_SUCCESS
                )
            except Exception as e:
                self.append_text(
                    f"  ⚠ Output route for {machine} failed: {e}\n",
                    self.C_ERROR
                )

    # ------------------------------------------------------------------
    # Grok AV Agent Setup
    # ------------------------------------------------------------------

    C_AV = "rgba(200, 130, 50, 255)"  # Warm orange for AV agent

    def _setup_av(self, arg: str = ""):
        """
        /av [voice]

        Sets up the Grok AudioVisual voice agent with:
        1. System prompt from ./systems/audiovisual.md
        2. Function tool config (handle_simple_programming)
        3. Shell variable $av pointing to agent dir
        4. Auto-attachment: $av/CODE → /n/rioa/scene/parse (blocking)
        5. Starts the voice session

        The agent directory (grok_av) is created by LLMFS at boot
        when the GrokAVAgent is registered — not via 'new' in ctl.
        """
        parts = arg.split() if arg else []
        voice = parts[0] if len(parts) > 0 else "Ara"

        agent_name = "av"
        agent_dir = os.path.join(self.llmfs_mount, agent_name)

        self.append_text("\n", self.C_AV)
        self.append_text("╔══════════════════════════════════════════╗\n", self.C_AV)
        self.append_text("║     GROK AV AGENT — Initializing...      ║\n", self.C_AV)
        self.append_text("╚══════════════════════════════════════════╝\n", self.C_AV)

        # Step 0: Create the agent via ctl
        ctl_path = os.path.join(self.llmfs_mount, "ctl")
        if not os.path.isdir(agent_dir):
            try:
                with open(ctl_path, 'w') as f:
                    f.write("grok av\n")
                self.append_text(f"  ✓ Agent '{agent_name}' created\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ✗ Failed to create agent: {e}\n", self.C_ERROR)
                return
        else:
            self.append_text(f"  • Agent '{agent_name}' already exists\n", self.C_INFO)

        # Step 1: Write config with function tool + voice
        try:
            config_path = os.path.join(agent_dir, "config")
            config = {
                "voice": voice,
                "functions": [
                    {
                        "name": "handle_simple_programming",
                        "description": "Execute ANY code or programming task. Always call this for: buttons, scripts, UI, calculations, or any coding request.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Raw Python code to execute"
                                }
                            },
                            "required": ["code"]
                        }
                    }
                ],
                "tool_choice": "required",
                "temperature": 0.8,
            }
            with open(config_path, 'w') as f:
                f.write(json.dumps(config))
            self.append_text(f"  ✓ Config: voice={voice}, tool_choice=required\n", self.C_SUCCESS)
            self.append_text("  ✓ Function tool: handle_simple_programming\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ✗ Failed to write config: {e}\n", self.C_ERROR)
            return

        # Step 2: Write system prompt from file
        try:
            system_path = os.path.join(agent_dir, "system")
            prompt_file = "./systems/audiovisual.md"

            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    system_prompt = f.read()
                with open(system_path, 'w') as f:
                    f.write(system_prompt)
                self.append_text("  ✓ System prompt configured (audiovisual.md)\n", self.C_SUCCESS)
            else:
                self.append_text(f"  ⚠ {prompt_file} not found, skipping system prompt\n", self.C_ERROR)
        except Exception as e:
            self.append_text(f"  ✗ Failed to set system prompt: {e}\n", self.C_ERROR)
            return

        # Step 3: Seed $av shell variable
        self._seed_agent_variable(agent_name)
        # Also seed a short alias
        self._suppress_shell_output = True
        self._execute_shell_raw(
            f'export av="{agent_dir}"'
        )
        QTimer.singleShot(300, self._unsuppress_shell_output)

        self.append_text(f"  ✓ Shell: $av = {agent_dir}\n", self.C_SUCCESS)
        self.append_text(f"  ✓ Shell: $grok_av = {agent_dir}\n", self.C_SUCCESS)

        # Step 4: Auto-attach $av/CODE → {rio_mount}/scene/parse (blocking)
        code_source = os.path.join(agent_dir, "CODE")
        scene_dest = f"{self.rio_mount}/scene/parse"

        try:
            self._add_attachment(code_source, scene_dest)
            self.append_text(f"  ✓ Auto-routing: $av/CODE → {scene_dest}\n", self.C_SUCCESS)
            # Track supplementary output file
            self.known_supplementary.setdefault(agent_name, set()).add("CODE")
        except Exception as e:
            self.append_text(f"  ⚠ Could not set up auto-routing: {e}\n", self.C_ERROR)

        # Step 5: Start the voice session
        try:
            ctl_path = os.path.join(agent_dir, "ctl")
            with open(ctl_path, 'w') as f:
                f.write("start\n")
            self.append_text("  ✓ Voice session started\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ⚠ Could not start session: {e}\n", self.C_ERROR)

        # Step 6: Connect terminal output
        self._connect_agent(agent_name)

        self.append_text("\n", self.C_AV)
        self.append_text("  Grok AV agent ready. Speak or type.\n", self.C_SUCCESS)
        self.append_text(f"  $av/CODE blocks until function tool produces code.\n", self.C_INFO)
        self.append_text(f"  Code auto-routes to {scene_dest}\n", self.C_INFO)
        self.append_text("  echo 'stop' > $av/ctl to disconnect voice.\n\n", self.C_INFO)

    def _setup_av_gemini(self, arg: str = ""):
        """
        /av_gemini [voice]

        Sets up the Gemini AudioVisual voice agent with:
        1. System prompt from ./systems/audiovisual.md
        2. Function tool config (handle_simple_programming)
        3. Shell variable $av_gemini pointing to agent dir
        4. Auto-attachment: $av_gemini/CODE → /n/rioa/scene/parse (blocking)
        5. Starts the voice session

        The agent directory (av_gemini) is created by LLMFS at boot
        when the AVAgent is registered — or via 'gemini av_gemini' in ctl.
        """
        parts = arg.split() if arg else []
        voice = parts[0] if len(parts) > 0 else "Aoede"

        agent_name = "av_gemini"
        agent_dir = os.path.join(self.llmfs_mount, agent_name)

        self.append_text("\n", self.C_AV)
        self.append_text("╔══════════════════════════════════════════╗\n", self.C_AV)
        self.append_text("║   GEMINI AV AGENT — Initializing...      ║\n", self.C_AV)
        self.append_text("╚══════════════════════════════════════════╝\n", self.C_AV)

        # Step 0: Create the agent via ctl
        ctl_path = os.path.join(self.llmfs_mount, "ctl")
        if not os.path.isdir(agent_dir):
            try:
                with open(ctl_path, 'w') as f:
                    f.write("av av_gemini\n")
                self.append_text(f"  ✓ Agent '{agent_name}' created\n", self.C_SUCCESS)
            except Exception as e:
                self.append_text(f"  ✗ Failed to create agent: {e}\n", self.C_ERROR)
                return
        else:
            self.append_text(f"  • Agent '{agent_name}' already exists\n", self.C_INFO)

        # Step 1: Write config with function tool + voice
        # Gemini tools use function_declarations format (not OpenAI format)
        try:
            config_path = os.path.join(agent_dir, "config")
            config = {
                "voice": voice,
                "functions": [
                    {
                        "name": "handle_simple_programming",
                        "description": "Execute ANY code or programming task. Always call this for: buttons, scripts, UI, calculations, or any coding request.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Raw Python code to execute"
                                }
                            },
                            "required": ["code"]
                        }
                    }
                ],
                "google_search": True,
            }
            with open(config_path, 'w') as f:
                f.write(json.dumps(config))
            self.append_text(f"  ✓ Config: voice={voice}\n", self.C_SUCCESS)
            self.append_text("  ✓ Function tool: handle_simple_programming\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ✗ Failed to write config: {e}\n", self.C_ERROR)
            return

        # Step 2: Write system prompt from file
        try:
            system_path = os.path.join(agent_dir, "system")
            prompt_file = "./systems/audiovisual.md"

            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    system_prompt = f.read()
                with open(system_path, 'w') as f:
                    f.write(system_prompt)
                self.append_text("  ✓ System prompt configured (audiovisual.md)\n", self.C_SUCCESS)
            else:
                self.append_text(f"  ⚠ {prompt_file} not found, skipping system prompt\n", self.C_ERROR)
        except Exception as e:
            self.append_text(f"  ✗ Failed to set system prompt: {e}\n", self.C_ERROR)
            return

        # Step 3: Seed $av_gemini shell variable
        self._seed_agent_variable(agent_name)
        self._suppress_shell_output = True
        self._execute_shell_raw(
            f'export av_gemini="{agent_dir}"'
        )
        QTimer.singleShot(300, self._unsuppress_shell_output)

        self.append_text(f"  ✓ Shell: $av_gemini = {agent_dir}\n", self.C_SUCCESS)

        # Step 4: Auto-attach $av_gemini/CODE → {rio_mount}/scene/parse (blocking)
        code_source = os.path.join(agent_dir, "CODE")
        scene_dest = f"{self.rio_mount}/scene/parse"

        try:
            self._add_attachment(code_source, scene_dest)
            self.append_text(f"  ✓ Auto-routing: $av_gemini/CODE → {scene_dest}\n", self.C_SUCCESS)
            # Track supplementary output file
            self.known_supplementary.setdefault(agent_name, set()).add("CODE")
        except Exception as e:
            self.append_text(f"  ⚠ Could not set up auto-routing: {e}\n", self.C_ERROR)

        # Step 5: Start the voice session
        try:
            ctl_path = os.path.join(agent_dir, "ctl")
            with open(ctl_path, 'w') as f:
                f.write("start\n")
            self.append_text("  ✓ Voice session started\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"  ⚠ Could not start session: {e}\n", self.C_ERROR)

        # Step 6: Connect terminal output
        self._connect_agent(agent_name)

        self.append_text("\n", self.C_AV)
        self.append_text("  Gemini AV agent ready. Speak or type.\n", self.C_SUCCESS)
        self.append_text(f"  $av_gemini/CODE blocks until function tool produces code.\n", self.C_INFO)
        self.append_text(f"  Code auto-routes to {scene_dest}\n", self.C_INFO)
        self.append_text("  echo 'stop' > $av_gemini/ctl to disconnect voice.\n\n", self.C_INFO)

    # ------------------------------------------------------------------
    # Agent lifecycle  (all via filesystem)
    # ------------------------------------------------------------------

    def _agent_dir(self, name: str = None) -> str:
        """Return path to agent directory."""
        name = name or self.connected_agent
        return os.path.join(self.llmfs_mount, name) if name else ""

    def _ensure_agent(self, name: str, system: str = None):
        """
        Create an agent if it doesn't exist, then connect.

        Supports provider/model in the argument:
          /new myagent                        → default provider
          /new myagent groq kimi-k2           → specific provider + model  
          /new myagent "You are a coder."     → system prompt (quoted or no space in provider name)

        Filesystem operations:
          1. Write 'new <n> [provider] [model]' to /n/llm/ctl
          2. Write system prompt to <n>/system (optional)
          3. Connect terminal I/O
        """
        ctl_path = os.path.join(self.llmfs_mount, "ctl")

        # Parse system arg — detect if it's "provider [model]" vs system prompt
        provider = None
        model = None
        if system:
            # Known provider names from the registry
            known_providers = {"claude", "gemini", "openai", "groq", "openrouter", "cerebras", "moonshot"}
            first_word = system.split()[0].lower() if system.split() else ""
            if first_word in known_providers:
                parts = system.split(None, 2)
                provider = parts[0]
                model = parts[1] if len(parts) > 1 else None
                system = parts[2] if len(parts) > 2 else None

        agent_dir = self._agent_dir(name)

        # Create if needed
        if not os.path.isdir(agent_dir):
            try:
                create_cmd = f"new {name}"
                if provider:
                    create_cmd += f" {provider}"
                if model:
                    create_cmd += f" {model}"
                with open(ctl_path, 'w') as f:
                    f.write(create_cmd + "\n")
                msg = f"Agent '{name}' created"
                if provider:
                    msg += f" ({provider}"
                    if model:
                        msg += f"/{model}"
                    msg += ")"
                self.append_text(msg + "\n", self.C_SUCCESS)
            except FileNotFoundError:
                try:
                    os.makedirs(agent_dir, exist_ok=True)
                    self.append_text(f"Agent '{name}' created (mkdir)\n", self.C_SUCCESS)
                except Exception as e:
                    self.append_text(f"Cannot create agent: {e}\n", self.C_ERROR)
                    self.append_text(f"Is LLMFS mounted at {self.llmfs_mount}?\n", self.C_INFO)
                    return
        else:
            # Agent exists — switch provider if requested
            if provider:
                ctl_agent = os.path.join(agent_dir, "ctl")
                try:
                    cmd = f"provider {provider}"
                    if model:
                        cmd += f" {model}"
                    with open(ctl_agent, 'w') as f:
                        f.write(cmd + "\n")
                    with open(ctl_agent, 'r') as f:
                        result = f.read().strip()
                    if result:
                        self.append_text(f"{result}\n", self.C_INFO)
                except Exception as e:
                    self.append_text(f"Failed to switch provider: {e}\n", self.C_ERROR)

        # Seed $name shell variable so $ commands can use it
        self._seed_agent_variable(name)

        # Set system prompt if provided
        if system:
            self._write_agent_file("system", system, agent_name=name)

        # Connect
        self._connect_agent(name)

    def _connect_agent(self, name: str):
        """
        Connect terminal to an agent's I/O.

        Uses OutputStreamReader (raw 9P) to read $agent/OUTPUT and display
        it directly in the terminal widget.

        Why raw 9P and not Plan9Attachment (while true; do cat src > dst; done)?
        
        The FUSE cat loop doesn't stream properly: the Linux kernel's VFS
        issues a read, gets the current data, then cat sees EOF and exits.
        The while loop restarts cat, which re-walks the entire 9P path,
        re-opens the file, and reads from offset 0 — getting the entire
        growing response buffer each time.  This produces superimposed
        output on the 2nd+ generation (first works because cat was started
        before the generation gate opened and properly blocked).

        OutputStreamReader speaks 9P directly with a persistent connection,
        holds the FID open, reads sequentially with advancing offsets, and
        properly blocks on the server-side generation gate between generations.

        Plan9Attachment (FUSE cat) remains correct for SupplementaryOutputFile
        routes ($coder/<MACHINE> → $workspace/scene/parse) because those files
        deliver complete content atomically after mark_ready(), not streaming
        chunks — so cat's single-read-then-EOF semantics are fine there.
        """
        agent_dir = self._agent_dir(name)

        if not os.path.isdir(agent_dir):
            self.append_text(f"Agent '{name}' not found at {agent_dir}\n", self.C_ERROR)
            return

        # Already connected to this agent — don't recreate the reader
        if self.connected_agent == name:
            if self._output_reader and self._output_reader.isRunning():
                self.append_text(f"Already connected -> {name}\n", self.C_INFO)
                return

        # Switch output stream from previous agent (but keep its routes alive)
        if self.connected_agent:
            self._disconnect_output_route()
            self._response_pending = False

        self.connected_agent = name
        self.known_agents.add(name)

        # Ensure shell variable exists for this agent
        self._seed_agent_variable(name)

        # Stream $agent/OUTPUT via raw 9P (not FUSE cat)
        self._output_reader = OutputStreamReader(
            agent_path=f"{name}",
            host=self.p9_host,
            port=self.p9_port,
            auth_token=self.p9_token,
        )
        # Route agent text through the stream_router under a per-agent
        # source key so its fence parser is independent from FS writers.
        self._output_reader.new_data.connect(self._on_agent_stream)
        self._output_reader.stream_done.connect(self._on_output_stream_done)
        self._output_reader.error_occurred.connect(
            lambda e: self.append_text(f"Stream error: {e}\n", self.C_ERROR)
        )
        self._output_reader.start()

        self.append_text(f"Connected -> {name}\n", self.C_SUCCESS)
        self.command_input.setPlaceholderText(f"[{name}] ")

    def _disconnect_output_route(self):
        """Stop the OutputStreamReader for the current agent."""
        if self._output_reader:
            self._output_reader.stop()
            self._output_reader.wait(2000)
            self._output_reader = None

    def _disconnect_agent(self, quiet=False):
        self._disconnect_output_route()
        old = self.connected_agent
        # Only tear down master routes when disconnecting from master itself
        if old == "master":
            self._stop_master()
        # Drop the router's parser state for this agent. Any fence still
        # open is force-closed so the user can still interact with the
        # widget (Run/edit/copy) — we just stop streaming into it.
        if old and hasattr(self, 'stream_router') and self.stream_router is not None:
            self.stream_router.reset_source(f"agent:{old}")
        self.connected_agent = None
        self._response_pending = False
        self.command_input.setPlaceholderText("Enter command or prompt...")
        if not quiet and old:
            self.append_text(f"Disconnected from {old}\n", self.C_INFO)

    def _delete_agent(self, name: str):
        """Delete agent via /n/llm/ctl."""
        if name == self.connected_agent:
            self._disconnect_agent(quiet=True)
        ctl_path = os.path.join(self.llmfs_mount, "ctl")
        try:
            with open(ctl_path, 'w') as f:
                f.write(f"delete {name}\n")
            self.append_text(f"Agent '{name}' deleted\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"Error deleting agent: {e}\n", self.C_ERROR)

    # ------------------------------------------------------------------
    # Sending prompts (write to $agent/input)
    # ------------------------------------------------------------------

    def _send_to_agent(self, prompt: str):
        if not self.connected_agent:
            self.append_text("No agent connected. Use /claude or /connect <name>\n", self.C_ERROR)
            return

        input_path = os.path.join(self._agent_dir(), "input")
        try:
            with open(input_path, 'w') as f:
                f.write(prompt)
            self._response_pending = True
        except Exception as e:
            self.append_text(f"Error writing to agent input: {e}\n", self.C_ERROR)

    # ------------------------------------------------------------------
    # Receiving output (via $term/output filesystem writes and via
    # OutputStreamReader for the connected agent)
    #
    # All streamed text passes through the TerminalStreamRouter so that
    # ```machine ... ``` fences are detected and morphed into inline
    # widgets. Each logical source has its own parser instance keyed by
    # source_key, which is what makes concurrent producers safe.
    # ------------------------------------------------------------------

    def _on_fs_output(self, text: str, source_key: str = None):
        """
        Called by TerminalOutputFile.write() when data arrives at $term/output.
        
        Runs on whatever thread the 9P server dispatches from, so we
        bounce onto the Qt main thread before touching widgets.
        
        source_key is optional; if a writer wants its own parser state
        (recommended when multiple agents share $term/output), it can
        pass e.g. f"fs:{fid.fid}" to keep its fences from interleaving
        with other writers'. Default "fs:default" treats all unkeyed
        writes as a single logical stream.
        """
        if source_key is None:
            source_key = "fs:default"
        QMetaObject.invokeMethod(
            self, "_route_stream",
            Qt.QueuedConnection,
            Q_ARG(str, source_key),
            Q_ARG(str, text),
            Q_ARG(str, self.C_AGENT),
        )

    @Slot(str, str, str)
    def _route_stream(self, source_key: str, text: str, color: str):
        """Qt-main-thread slot: feed a chunk into the router for the
        given source. All multi-thread producers funnel through here."""
        if not text:
            return
        self.stream_router.feed(source_key, text, color)

    @Slot(str)
    def _route_stream_eof(self, source_key: str):
        """Qt-main-thread slot: end-of-stream for an FS source.
        
        Called when a writer closes its file handle on $term/output.
        Forces any open inline fence widget to finalize and drops the
        per-fid parser state so we don't keep dead source entries.
        """
        if not hasattr(self, 'stream_router') or self.stream_router is None:
            return
        self.stream_router.end_of_stream(source_key)
        self.stream_router.reset_source(source_key)

    @Slot(str)
    def _on_agent_stream(self, text: str):
        """Slot for OutputStreamReader.new_data — connected agent's stream.
        
        Keyed on the connected agent's name so its fence parser is
        independent from any other producer writing to $term/output.
        """
        if not text:
            return
        key = f"agent:{self.connected_agent or '_unknown_'}"
        self.stream_router.feed(key, text, self.C_AGENT)

    @Slot(str)
    def _display_agent_text(self, text: str):
        """Backwards-compatible slot — routes through the stream router
        under the generic "fs:default" key. Kept so external callers and
        QMetaObject.invokeMethod targets continue to work."""
        if not text:
            return
        self.stream_router.feed("fs:default", text, self.C_AGENT)

    @Slot(str)
    def _on_inline_media(self, payload_json: str):
        """Slot for $term/inline writes — render a media widget.
        
        payload_json is a JSON string describing the widget to display.
        See InlineMediaWidget for the schema.
        """
        try:
            payload = json.loads(payload_json)
        except Exception as e:
            self.append_text(f"inline: invalid JSON: {e}\n", self.C_ERROR)
            return
        if not isinstance(payload, dict):
            self.append_text(f"inline: payload must be a JSON object\n", self.C_ERROR)
            return
        try:
            self.stream_router.insert_media(payload)
        except Exception as e:
            self.append_text(f"inline: render error: {e}\n", self.C_ERROR)

    @Slot()
    def _on_output_stream_done(self):
        """Called when the OutputStreamReader sees EOF (generation complete)."""
        self._response_pending = False
        # Finalize any open inline widget for the current agent's source.
        if self.connected_agent:
            self.stream_router.end_of_stream(f"agent:{self.connected_agent}")

    # ------------------------------------------------------------------
    # Filesystem helpers (ctl, read, write)
    # ------------------------------------------------------------------

    def _agent_ctl(self, command: str):
        """Write a command to the connected agent's ctl file."""
        if not self.connected_agent:
            self.append_text("No agent connected\n", self.C_ERROR)
            return
        ctl_path = os.path.join(self._agent_dir(), "ctl")
        try:
            with open(ctl_path, 'w') as f:
                f.write(command + "\n")
            # Read back result
            with open(ctl_path, 'r') as f:
                result = f.read().strip()
            if result:
                self.append_text(f"{result}\n", self.C_INFO)
        except Exception as e:
            self.append_text(f"ctl error: {e}\n", self.C_ERROR)

    # ---- Provider shortcut aliases ----
    # Maps short names to (provider, model_substring) pairs.
    # /use <alias> expands to: echo 'provider <provider> <matched_model>' > $agent/ctl
    PROVIDER_ALIASES = {
        "kimi":     ("groq", "kimi"),
        "zai":      ("cerebras", "zai"),
        "llama70":  ("cerebras", "70b"),
        "llama8":   ("cerebras", "8b"),
        "qwen":     ("cerebras", "qwen-3-32b"),
        "gptoss":   ("cerebras", "gpt-oss"),
        "sonnet":   ("claude", "sonnet"),
        "opus":     ("claude", "opus"),
        "haiku":    ("claude", "haiku"),
        "gpt4o":    ("openai", "gpt-4o"),
        "flash":    ("gemini", "flash"),
        "pro":      ("gemini", "pro"),
    }

    def _use_provider_model(self, arg: str):
        """
        Quick provider+model switch with fuzzy matching.
        
        Usage:
            /use groq kimi          → switch to groq, fuzzy-match 'kimi' model
            /use cerebras zai       → switch to cerebras, fuzzy-match 'zai' model
            /use kimi               → alias lookup, expands to groq + kimi model
            /use zai                → alias lookup, expands to cerebras + zai model
            /use                    → show current provider + available aliases
        """
        if not self.connected_agent:
            self.append_text("No agent connected\n", self.C_ERROR)
            return
        
        if not arg:
            # Show current state + aliases
            self._agent_ctl("provider")
            self._agent_ctl("model")
            self.append_text("\nAliases:\n", self.C_INFO)
            for alias, (prov, hint) in sorted(self.PROVIDER_ALIASES.items()):
                self.append_text(f"  {alias:12s} → {prov} ({hint})\n", self.C_DEFAULT)
            return
        
        parts = arg.split(None, 1)
        
        # Check if first word is a known alias
        if len(parts) == 1 and parts[0].lower() in self.PROVIDER_ALIASES:
            provider_name, model_hint = self.PROVIDER_ALIASES[parts[0].lower()]
        elif len(parts) >= 2:
            provider_name = parts[0]
            model_hint = parts[1]
        elif len(parts) == 1:
            # Single word, not an alias — try as provider name with default model
            provider_name = parts[0]
            model_hint = None
        else:
            self.append_text("Usage: /use <provider> [model] or /use <alias>\n", self.C_ERROR)
            return
        
        # Resolve model via fuzzy match against provider's model list
        if model_hint:
            try:
                providers_path = os.path.join(self.llmfs_mount, "providers")
                available = []
                with open(providers_path, 'r') as f:
                    in_provider = False
                    for line in f:
                        line = line.rstrip()
                        if line.startswith(f"{provider_name}:"):
                            in_provider = True
                            continue
                        elif in_provider and line.startswith("  "):
                            available.append(line.strip())
                        elif in_provider:
                            break  # next provider section
            except Exception:
                # Fallback: just pass the hint as-is and let the provider handle it
                available = []
            
            if available:
                # Fuzzy match: find first model containing the hint (case-insensitive)
                hint_lower = model_hint.lower()
                matched = [m for m in available if hint_lower in m.lower()]
                if matched:
                    model = matched[0]
                else:
                    self.append_text(f"No model matching '{model_hint}' in {provider_name}. Available:\n", self.C_ERROR)
                    for m in available:
                        self.append_text(f"  {m}\n", self.C_DEFAULT)
                    return
            else:
                model = model_hint
        else:
            model = None
        
        # Execute the switch
        if model:
            self._agent_ctl(f"provider {provider_name} {model}")
        else:
            self._agent_ctl(f"provider {provider_name}")

    def _read_agent_file(self, filename: str, agent_name: str = None):
        """Read and display an agent file."""
        name = agent_name or self.connected_agent
        if not name:
            self.append_text("No agent connected\n", self.C_ERROR)
            return
        path = os.path.join(self._agent_dir(name), filename)
        try:
            with open(path, 'r') as f:
                content = f.read()
            if content.strip():
                self.append_text(f"-- {filename} --\n", self.C_INFO)
                self.append_text(content, self.C_DEFAULT)
                if not content.endswith('\n'):
                    self.append_text("\n", self.C_DEFAULT)
                self.append_text(f"-- end --\n", self.C_INFO)
            else:
                self.append_text(f"{filename}: (empty)\n", self.C_INFO)
        except Exception as e:
            self.append_text(f"Error reading {filename}: {e}\n", self.C_ERROR)

    def _write_agent_file(self, filename: str, content: str, agent_name: str = None):
        """Write content to an agent file."""
        name = agent_name or self.connected_agent
        if not name:
            self.append_text("No agent connected\n", self.C_ERROR)
            return
        path = os.path.join(self._agent_dir(name), filename)
        try:
            with open(path, 'w') as f:
                f.write(content)
            self.append_text(f"{filename} updated\n", self.C_SUCCESS)
        except Exception as e:
            self.append_text(f"Error writing {filename}: {e}\n", self.C_ERROR)

    def _show_agent_history(self):
        """Read and display agent conversation history from $agent/history."""
        if not self.connected_agent:
            self.append_text("No agent connected\n", self.C_ERROR)
            return
        path = os.path.join(self._agent_dir(), "history")
        try:
            with open(path, 'r') as f:
                raw = f.read()
            if not raw.strip():
                self.append_text("(no history)\n", self.C_INFO)
                return
            history = json.loads(raw)
            self.append_text(f"-- history ({len(history)} messages) --\n", self.C_INFO)
            for msg in history:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                color = self.C_USER if role == "user" else self.C_AGENT
                prefix = ">> " if role == "user" else "<< "
                display = content if len(content) < 300 else content[:300] + "..."
                self.append_text(f"{prefix}{display}\n", color)
            self.append_text(f"-- end --\n", self.C_INFO)
        except Exception as e:
            self.append_text(f"Error reading history: {e}\n", self.C_ERROR)

    def _list_agents(self):
        """List agents by reading the LLMFS root directory."""
        agents_dir = self.llmfs_mount
        if not os.path.isdir(agents_dir):
            self.append_text(f"Not found: {agents_dir}\n", self.C_ERROR)
            return
        try:
            entries = sorted(os.listdir(agents_dir))
            dirs = [e for e in entries if os.path.isdir(os.path.join(agents_dir, e))]
            if not dirs:
                self.append_text("No agents\n", self.C_INFO)
                return
            self.append_text("Agents:\n", self.C_INFO)
            for d in dirs:
                marker = "* " if d == self.connected_agent else "  "
                self.append_text(f"  {marker}{d}\n", self.C_DEFAULT)
        except Exception as e:
            self.append_text(f"Error listing agents: {e}\n", self.C_ERROR)

    def _show_status(self):
        status_lines = [
            f"LLM Mount:   {self.llmfs_mount}",
            f"Rio Mount:   {self.rio_mount}",
            f"Agent:       {self.connected_agent or '(none)'}",
            f"Streaming:   {'yes' if self.connected_agent else 'no'}",
            f"Attachments: {len(self._routes_manager.attachments) if self._routes_manager else 0}",
            f"History:     {len(self.command_history)} commands",
        ]
        ctl_path = os.path.join(self.llmfs_mount, "ctl")
        if os.path.exists(ctl_path):
            try:
                with open(ctl_path, 'r') as f:
                    status_lines.append(f"Server:      {f.read().strip()}")
            except Exception:
                pass
        self.append_text("\n".join(status_lines) + "\n", self.C_INFO)

    # ------------------------------------------------------------------
    # Shell execution
    # ------------------------------------------------------------------

    def _update_input_style(self):
        dark = getattr(self, '_is_dark_mode', False)
        if self.terminal_mode:
            if dark:
                self._set_input_bg_target(30, 35, 45, 180)
            else:
                self._set_input_bg_target(240, 245, 250, 150)
        else:
            if dark:
                self._set_input_bg_target(40, 40, 50, 180)
            else:
                self._set_input_bg_target(255, 255, 255, 150)

    # ANSI color map — fallback used when no active scheme is set
    _ANSI_COLOR_MAP = {
        '30': '#000000', '31': '#CD0000', '32': '#00CD00', '33': '#CDCD00',
        '34': '#0000EE', '35': '#CD00CD', '36': '#00CDCD', '37': '#E5E5E5',
        '90': '#7F7F7F', '91': '#FF0000', '92': '#00FF00', '93': '#FFFF00',
        '94': '#5C5CFF', '95': '#FF00FF', '96': '#00FFFF', '97': '#FFFFFF',
    }

    @property
    def _active_ansi_map(self):
        """Return the ANSI color map from the active scheme."""
        return self._active_scheme.get("ansi_map", self._ANSI_COLOR_MAP)

    @property
    def _active_shell_echo_color(self):
        """Shell echo ($ command) color — always black or white for readability."""
        if getattr(self, '_is_dark_mode', False):
            return "rgba(230, 230, 230, 255)"
        return "rgba(0, 0, 0, 255)"

    @property
    def _active_shell_output_color(self):
        """Shell output color — always black or white for readability."""
        if getattr(self, '_is_dark_mode', False):
            return "rgba(230, 230, 230, 230)"
        return "rgba(0, 0, 0, 230)"

    @property
    def _active_shadow_color(self):
        return self._active_scheme.get("shadow", "rgba(0, 0, 0, 120)")

    # Module-level cache for _parse_rgba — color strings are highly
    # repetitive (a fixed scheme + a handful of ANSI variants), but the
    # parsing itself does string slicing, splitting, and int conversion
    # which adds up under fast streaming. We cache the parsed (r,g,b,a)
    # tuple rather than the QColor instance, then construct a fresh
    # QColor on each call — this keeps callers safe from accidentally
    # sharing a mutable QColor. The cache is bounded by the small
    # universe of distinct color strings the app emits.
    _RGBA_CACHE = {}

    @staticmethod
    def _parse_rgba(color_str):
        """Parse rgba(...)/rgb(...) or hex color strings into QColor."""
        cache = TerminalWidget._RGBA_CACHE
        cached = cache.get(color_str)
        if cached is not None:
            r, g, b, a = cached
            c = QColor()
            c.setRgb(r, g, b, a)
            return c

        c = QColor()
        if color_str.startswith('rgba('):
            inner = color_str[5:].rstrip(')')
            parts = [int(x.strip()) for x in inner.split(',')]
            if len(parts) >= 4:
                c.setRgb(parts[0], parts[1], parts[2], parts[3])
            elif len(parts) == 3:
                c.setRgb(parts[0], parts[1], parts[2])
        elif color_str.startswith('rgb('):
            inner = color_str[4:].rstrip(')')
            parts = [int(x.strip()) for x in inner.split(',')]
            c.setRgb(parts[0], parts[1], parts[2])
        else:
            c.setNamedColor(color_str)

        cache[color_str] = (c.red(), c.green(), c.blue(), c.alpha())
        return c

    # Regex that matches all ANSI escape sequences we care about,
    # splitting text into (plain_text, escape_sequence) pairs.
    _ANSI_RE = re.compile(
        r'('
        r'\x1b\].*?(?:\x07|\x1b\\)'   # OSC (window title etc.)
        r'|\x1b\[[\d;]*m'              # SGR (colors, bold, reset)
        r'|\x1b\[[\x20-\x3F]*[\x40-\x7E]'  # other CSI
        r'|\x1b[\x20-\x7E]'            # two-byte escapes
        r'|\x1b'                        # stray ESC
        r')'
    )

    # Pre-compiled SGR matcher used per segment. fullmatch is slow at scale;
    # a compiled regex .match against a slice is faster than re.fullmatch
    # on the unprefixed module function.
    _SGR_RE = re.compile(r'\x1b\[([\d;]*)m')

    def _insert_ansi_text(self, cursor: QTextCursor, text: str):
        """
        Parse ANSI escape sequences and insert colored plain text
        directly via QTextCursor.insertText + QTextCharFormat.

        This completely avoids insertHtml, so shell metacharacters
        like <, >, &, quotes etc. are never misinterpreted as HTML.

        Hot-path notes:
          - default_color and its parsed brush are computed ONCE per
            call rather than re-derived on every SGR-0 (reset) segment.
          - Constructing QTextCharFormat is non-trivial; we keep one
            format object and mutate its foreground/font as needed
            instead of creating a fresh copy on every reset.
          - Fast path for chunks with no escape sequences (the common
            case during agent token streaming): skip the regex split
            entirely and insert with the prevailing format.
        """
        # Strip \r (PTY sends \r\n, Qt only needs \n)
        text = text.replace('\r', '')

        # Fast path: no escape bytes at all → one direct insert.
        # ANSI-free chunks are the common case during LLM token streaming
        # and during ordinary command output.
        if '\x1b' not in text:
            # We still need to apply the default color so the text picks
            # up dark/light-mode adjustment. Use a memoized format.
            fmt = self._default_ansi_format()
            cursor.insertText(text, fmt)
            return

        color_map = self._active_ansi_map
        default_color = self._dm_adjust_color(self._active_shell_output_color)
        default_brush_color = self._parse_rgba(default_color)

        # Start from the document's current char format so we inherit
        # the font family / size set via the QTextEdit stylesheet.
        base_fmt = cursor.charFormat()

        # One reusable format we mutate. Avoids QTextCharFormat copy
        # allocations on every reset (which used to happen on every
        # SGR-0 and at function entry).
        fmt = QTextCharFormat(base_fmt)
        fmt.setForeground(default_brush_color)
        # Capture base font once; reused on bold toggle.
        base_font = base_fmt.font()

        for segment in self._ANSI_RE.split(text):
            if not segment:
                continue

            if segment[0] == '\x1b':
                # SGR?
                m = self._SGR_RE.fullmatch(segment)
                if m:
                    code_field = m.group(1)
                    codes = code_field.split(';') if code_field else ['0']
                    for code in codes:
                        code = code.lstrip('0') or '0'  # '00'→'0', '01'→'1'
                        if code == '0':
                            # Reset — restore base format + default color,
                            # mutating the existing object instead of
                            # allocating a new one.
                            fmt.setForeground(default_brush_color)
                            f = QFont(base_font)
                            f.setBold(False)
                            fmt.setFont(f)
                        elif code == '1':
                            f = fmt.font()
                            f.setBold(True)
                            fmt.setFont(f)
                        elif code in color_map:
                            fg_color = self._dm_adjust_color(color_map[code])
                            fmt.setForeground(self._parse_rgba(fg_color))
                # All other escape sequences (OSC, CSI, etc.) are silently dropped
                continue

            # Plain text — insert with current format
            cursor.insertText(segment, fmt)

    def _default_ansi_format(self):
        """Cached QTextCharFormat for ANSI-free chunks.

        Rebuilt only when the active scheme or dark mode changes.
        Keyed on (color_str, is_dark) so a theme/mode flip causes a
        fresh format object the next call.
        """
        key = (self._active_shell_output_color,
               bool(getattr(self, '_is_dark_mode', False)))
        cache = getattr(self, '_default_ansi_fmt_cache', None)
        if cache and cache[0] == key:
            return cache[1]
        adjusted = self._dm_adjust_color(self._active_shell_output_color)
        fmt = QTextCharFormat()
        fmt.setForeground(self._parse_rgba(adjusted))
        self._default_ansi_fmt_cache = (key, fmt)
        return fmt

    def ansi_to_html(self, text):  # DEAD CODE — no remaining callers, kept for reference
        """Legacy — kept for any remaining callers.  Prefer _insert_ansi_text."""
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r'\x1b\].*?(?:\x07|\x1b\\)', '', text)
        text = re.sub(r'\x1b\[[\d;]*m', '', text)
        text = re.sub(r'\x1b\[[\x20-\x3F]*[\x40-\x7E]', '', text)
        text = re.sub(r'\x1b[\x20-\x7E]', '', text)
        text = text.replace('\x1b', '')
        text = text.replace('\r', '')
        return text.replace('\n', '<br>')

    def _setup_shell_process(self):
        """
        Initialize a persistent background bash process using a PTY.

        The same PTY is shared by:
          - $ one-off commands
          - $ persistent shell mode
          - master agent bash blocks (via _on_master_bash_command)
          - TerminalStdinFile writes (via /n/rioa/terms/<id>/stdin)

        All of them share one process, one environment, one set of
        variables.  Convenience shell variables are seeded so the
        agent (and the user) can write ``echo 'hi' > $claude/input``
        instead of spelling out the full 9P mount path.
        """
        master_fd, slave_fd = pty.openpty()
        self.shell_fd = master_fd

        # Set a wide terminal size so readline never wraps/redraws
        # long commands (which produces \r + partial redraws that
        # garble the display).  struct winsize: rows, cols, xpix, ypix
        winsize = struct.pack('HHHH', 50, 300, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        env = os.environ.copy()
        # Seed convenience paths so all commands share them
        env['LLMFS'] = self.llmfs_mount
        env['RIO'] = self.rio_mount

        self.shell_process = subprocess.Popen(
            ["/bin/bash", "-i"],  # Interactive bash
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,      # stderr goes to same PTY → shows in terminal
            preexec_fn=os.setsid,
            env=env
        )

        # Close slave in parent
        os.close(slave_fd)

        # Start a thread to listen to shell output (stdout + stderr)
        self._start_shell_reader()

        # Seed shell variables inside the running bash
        # These survive for the lifetime of the shell process.
        self._seed_shell_variables()

    def _start_shell_reader(self):
        """Start (or restart) the background reader thread for the PTY."""
        self.shell_reader_thread = QThread()
        self.shell_reader_worker = ShellReaderWorker(self.shell_fd)
        self.shell_reader_worker.moveToThread(self.shell_reader_thread)
        self.shell_reader_worker.output_ready.connect(self._on_shell_output)
        self.shell_reader_thread.started.connect(self.shell_reader_worker.run)
        self.shell_reader_thread.start()

    def _seed_shell_variables(self):
        """
        Inject convenience shell variables into the running bash.

        Uses _suppress_shell_output to hide the PTY echo of these
        internal commands from the terminal display.
        """
        self._suppress_shell_output = True
        seeds = [
            f'export LLMFS="{self.llmfs_mount}"',
            f'export RIO="{self.rio_mount}"',
            f'export term="{self.rio_mount}/terms/{self.term_id}"',
            # $inline is a shorthand for $term/inline so users can
            # do `echo /tmp/x.png > $inline` instead of the full path.
            f'export inline="{self.rio_mount}/terms/{self.term_id}/inline"',
            # $peribus points at the mycelium-layer mount; once /peribus
            # has run, `cat $peribus/feed/recent` and friends Just Work.
            f'export peribus="{self._peribus_root}"',
            'agent() { echo "${LLMFS}/$1"; }',
            'export PAGER=cat',
            'export GIT_PAGER=cat',
            'export TERM=dumb',
            "bind 'set enable-bracketed-paste off' 2>/dev/null",
        ]
        for line in seeds:
            self._execute_shell_raw(line)
        # Delay unsuppression so the PTY echo of the last command
        # has time to arrive and be swallowed
        QTimer.singleShot(500, self._unsuppress_shell_output)

    def _seed_agent_variable(self, agent_name: str):
        """Create a convenience shell variable for a specific agent."""
        safe = agent_name.replace('-', '_').replace('.', '_')
        self._suppress_shell_output = True
        self._execute_shell_raw(
            f'export {safe}="{self.llmfs_mount}/{agent_name}"'
        )
        QTimer.singleShot(300, self._unsuppress_shell_output)

    def _unsuppress_shell_output(self):
        self._suppress_shell_output = False

    def _interrupt_shell(self):
        """
        Send SIGINT to the shell's process group — interrupts any
        running command.  Triggered by the Delete key or by writing
        to $term/interrupt.
        """
        # Clear any pending echo suppression to avoid swallowing output
        self._suppress_echo_line = None
        self._suppress_echo_buf = ""

        if self.shell_process and self.shell_process.poll() is None:
            try:
                pgid = os.getpgid(self.shell_process.pid)
                os.killpg(pgid, signal.SIGINT)
                self.append_text("^C\n", self.C_ERROR)
            except (OSError, ProcessLookupError):
                pass

    def _execute_shell_raw(self, command: str):
        """
        Low-level: send bytes to the PTY fd.

        Does NOT echo to the terminal display (that happens when the
        shell writes back through the PTY reader).  Captures write
        errors and surfaces them in the widget.

        Multi-line commands are written to a temp file and sourced
        via ``source /tmp/xxx.sh`` so that bash does not echo every
        line back through the PTY (which garbles the display with
        PS2 prompts, HTML-hostile characters, and truncated lines).
        """
        try:
            if '\n' in command.strip():
                # Multi-line: write to temp file, source it
                fd, path = tempfile.mkstemp(suffix='.sh', prefix='llmfs_cmd_')
                with os.fdopen(fd, 'w') as f:
                    f.write(command)
                # source executes in the current shell env, then we
                # remove the temp file.  The whole thing is one PTY line.
                oneliner = f'source {path}; rm -f {path}\n'
                os.write(self.shell_fd, oneliner.encode('utf-8'))
            else:
                if not command.endswith('\n'):
                    command += '\n'
                os.write(self.shell_fd, command.encode('utf-8'))
        except OSError as e:
            self.append_text(f"[shell write error] {e}\n", self.C_ERROR)

    def _execute_shell(self, command: str, echo: bool = False):
        """
        Send a command to the persistent shell.

        All shell execution paths converge here:
          - User types ``$ ls``             → echo=True
          - User is in shell mode           → echo=True
          - Master agent bash block         → echo=False (PTY echoes)
          - External write to term/stdin    → echo=False

        When echo=True, we print the command cleanly in the widget
        before sending it.  The PTY will also echo the command back
        through _on_shell_output, so we suppress that duplicate by
        setting _suppress_next_echo.

        When echo=False (programmatic), we let the PTY echo handle
        the display naturally.
        """
        if self.shell_process is None or self.shell_process.poll() is not None:
            self.append_text(
                "[shell dead — use /restart to create a new one]\n",
                self.C_ERROR
            )
            return

        if echo:
            self.append_text(f"$ {command}\n", self._active_shell_echo_color)
            # Suppress the PTY echo of this command to avoid double-print.
            # The PTY will echo the command text back; we mark it to skip.
            self._suppress_echo_line = command.strip()

        self._execute_shell_raw(command)

        # Schedule mark_ready on term/stdout after output settles.
        if self._term_dir is not None:
            self._bash_mark_ready_debounce()

    def _bash_mark_ready_debounce(self):
        """
        Debounced mark_ready for term/stdout.

        Each call resets a 600ms timer.  When the timer finally fires
        (no new shell output for 600ms), we mark the stdout file's
        captured output as ready for reading.  This lets the
        ``cat $term/stdout`` unblock with the full output.
        """
        if not hasattr(self, '_bash_debounce_timer') or self._bash_debounce_timer is None:
            self._bash_debounce_timer = QTimer(self)
            self._bash_debounce_timer.setSingleShot(True)
            self._bash_debounce_timer.timeout.connect(self._bash_mark_ready_fire)
        # (Re)start the timer — resets if already running
        self._bash_debounce_timer.start(600)

    def _bash_mark_ready_fire(self):
        """Timer fired — mark term/stdout output as ready."""
        if self._term_dir is not None:
            self._term_dir.stdout_file.mark_ready()

    def _on_shell_output(self, text):
        """
        Handle raw output from the PTY.

        Everything the shell writes -- stdout, stderr, prompts, command
        echo -- arrives here because the PTY merges them all.  We:

        1. Render it as HTML in the terminal widget (always)
        2. Feed it into term/output  (so external readers can monitor)
        3. Feed it into term/stdout  (so the master agent can read back
                                      the result of commands it ran)

        When a user types a command with echo=True, we already printed
        it cleanly.  The PTY echoes the same text back; we detect and
        suppress that duplicate.
        """
        # Suppress ALL output during seed commands (export vars etc.)
        if self._suppress_shell_output:
            return

        # Suppress PTY echo of a command we already displayed cleanly.
        # When echo=True, we already printed "$ <command>" in the widget.
        # The PTY echoes back the same command as its first line of output.
        # Strategy: accumulate until we see a \n, drop that first line
        # (the echo), and pass any remainder through normally.
        if self._suppress_echo_line is not None:
            self._suppress_echo_buf += text

            # Feed all raw data into filesystem regardless of suppression
            if self._term_dir is not None:
                try:
                    self._term_dir.stdout_file.capture_output(text)
                except Exception:
                    pass
                if self._term_dir.stdout_file._capturing:
                    self._bash_mark_ready_debounce()

            # Look for the end of the echo line (\n) in the raw buffer
            nl_pos = self._suppress_echo_buf.find('\n')
            if nl_pos >= 0:
                # Found end of echo line — suppress it, pass remainder
                self._suppress_echo_line = None
                remainder = self._suppress_echo_buf[nl_pos + 1:]
                self._suppress_echo_buf = ""
                if remainder:
                    self._on_shell_output(remainder)
            # else: still accumulating, wait for more chunks
            return

        # Check if this is a password prompt
        password_indicators = ['password:', 'Password:', 'password for', 'Password for']
        is_password_prompt = any(indicator in text for indicator in password_indicators)

        if is_password_prompt:
            self._password_mode = True

        # 1. Render in the terminal widget (plain text, no HTML parsing)
        cursor = self.current_text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._insert_ansi_text(cursor, text)
        self.current_text_display.setTextCursor(cursor)
        # Coalesced — under fast output the PTY emits dozens of chunks
        # per second; previously each one queued its own singleShot(0,
        # _scroll_to_bottom). Now a burst collapses to one scroll per
        # ~16 ms via the shared terminal-level timer.
        self._request_scroll_coalesced()

        # 2. Feed into filesystem files (if registered)
        if self._term_dir is not None:
            import asyncio as _aio

            # term/output -- monitoring tap (QueueFile)
            try:
                _aio.ensure_future(
                    self._term_dir.output_file.post(text.encode('utf-8', errors='replace'))
                )
            except Exception:
                pass

            # term/stdout -- capture for read-back (blocking stdout file)
            try:
                self._term_dir.stdout_file.capture_output(text)
            except Exception:
                pass

            # Start/reset the mark_ready debounce whenever the stdout file
            # is actively capturing.  This covers both paths:
            #   - _execute_shell started the debounce (user/master commands)
            #   - TerminalStdinFile.write started capturing (external 9P writes)
            if self._term_dir.stdout_file._capturing:
                self._bash_mark_ready_debounce()

    def _teardown_shell(self):
        """
        Kill the current shell process and reader thread cleanly.
        """
        # Stop the reader worker first
        if hasattr(self, 'shell_reader_worker') and self.shell_reader_worker:
            self.shell_reader_worker._running = False

        if hasattr(self, 'shell_reader_thread') and self.shell_reader_thread:
            self.shell_reader_thread.quit()
            self.shell_reader_thread.wait(2000)
            self.shell_reader_thread = None
            self.shell_reader_worker = None

        # Kill the shell process
        if hasattr(self, 'shell_process') and self.shell_process:
            try:
                os.killpg(os.getpgid(self.shell_process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                self.shell_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.shell_process.pid), signal.SIGKILL)
                    self.shell_process.wait(timeout=1)
                except Exception:
                    pass
            self.shell_process = None

        # Close the PTY master fd
        if hasattr(self, 'shell_fd') and self.shell_fd is not None:
            try:
                os.close(self.shell_fd)
            except OSError:
                pass
            self.shell_fd = None

    def _restart_shell(self):
        """
        /restart — tear down the current shell and spin up a fresh one.

        Preserves:
          - Connected agent
          - Master agent state & bash reader
          - All attachments
          - Terminal output history

        Resets:
          - Shell process (new PID, fresh env)
          - All shell variables (re-seeded)
        """
        self.append_text("\n⟳ Restarting shell...\n", self.C_SYSTEM)

        self._teardown_shell()
        self._setup_shell_process()

        # Re-seed the agent variable for the currently connected agent
        if self.connected_agent:
            self._seed_agent_variable(self.connected_agent)

        # Re-seed variables for all known agents
        agents_dir = self.llmfs_mount
        if os.path.isdir(agents_dir):
            try:
                for name in os.listdir(agents_dir):
                    if os.path.isdir(os.path.join(agents_dir, name)):
                        self._seed_agent_variable(name)
            except OSError:
                pass

        self.append_text("✓ Shell restarted (new PID, variables re-seeded)\n", self.C_SUCCESS)

    def _mount_9p(self, addr: str, name: str):
        """
        Mount a 9P service via 9pfuse.
        
        Usage: /mount IP!Port name
        Mounts tcp!IP!Port at /n/name using 9pfuse.
        Retries up to 5 times with 1s delay.
        """
        mount_point = f"/n/{name}"
        # addr is expected as IP!Port, convert to 9P dial string
        dial = f"tcp!{addr}"
        
        self.append_text(f"\n⟳ Mounting {dial} at {mount_point}...\n", self.C_SYSTEM)
        
        script = f"""set +e
mkdir -p "{mount_point}"
MOUNTED=0
for i in 1 2 3 4 5; do
  if 9pfuse '{dial}' "{mount_point}" 2>/dev/null; then
    echo "✓ {mount_point} mounted ({dial})"
    MOUNTED=1
    break
  fi
  echo "  retry $i for {mount_point}..."
  sleep 1
done
if [ "$MOUNTED" = "0" ]; then
  echo "✗ Failed to mount {mount_point}"
  exit 1
fi
"""
        self._execute_shell(script)

    # ------------------------------------------------------------------
    # /signal — bidirectional SignalBus subscription via /scene/signals/ctl
    # ------------------------------------------------------------------

    # Signal-bus UDP port. The 9P rio port is `self.p9_port + 1` (default
    # 5641); the SignalBus binds separately (default 5741). Kept here so
    # a future user can tweak both in one place.
    _SIGNAL_BUS_PORT = 5741
    # Port on which peer rio servers expose their 9P scene tree. Lines
    # in /n/ctl on this port are machines; anything else (e.g. the LLMFS
    # port 5640) is filtered out.
    _RIO_9P_PORT_DEFAULT = 5641

    # Loopback addresses we treat as "this host" when rewriting subscribe
    # lines for remote targets. /n/ctl is relative to the local mux —
    # any machine here means "on the same host as the mux" — so when we
    # tell a *remote* machine to subscribe to a loopback peer, we
    # substitute the loopback with our LAN IP.
    _LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

    def _detect_local_ip(self) -> str:
        """
        Best-effort discovery of our LAN-routable IP.

        We open a UDP socket to a public address — no packets are sent —
        and read back which local interface the kernel picked. This is
        the standard way to get "the IP a peer would see" without
        parsing `ip addr` or guessing interface names. Falls back to
        127.0.0.1 if there's no network at all (in which case any
        remote-rewrite that needed a real IP just becomes a no-op,
        which is the safest failure mode).
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
            finally:
                s.close()
        except Exception:
            return "127.0.0.1"

    # Where the mux's combined ctl lives. `self.llmfs_mount` is the
    # *individual machine* mount (e.g. /n/mux/llm) — that's a different
    # file. We want the mux-root ctl. Probed in order.
    _MUX_BASES = ("/n/mux", "/n")

    def _parse_n_ctl(self):
        """
        Read the mux's `ctl` file and return (mux_base, machines).

          mux_base : str            — the mux root dir we found, e.g.
                                      "/n/mux" or "/n". Caller uses
                                      this to build per-machine paths
                                      <mux_base>/<name>/scene/signals/ctl.
          machines : list[(name, host)] — every entry on the rio 9P
                                      port (5641). LLM-port entries
                                      (5640) and malformed lines are
                                      discarded.

        We probe /n/mux/ctl first, then /n/ctl. `self.llmfs_mount`
        is *not* used here — that points at one individual LLMFS
        mount (e.g. /n/mux/llm), whose ctl is the per-machine LLMFS
        ctl, not the mux's combined view.

        Format of /n/ctl as documented by the user:
            cirno   192.168.1.162:5641
            ekanza  127.0.0.1:5641
            llm     127.0.0.1:5640     # not a machine — filtered out
        """
        ctl_path = None
        mux_base = None
        last_error = None
        for base in self._MUX_BASES:
            candidate = os.path.join(base, "ctl")
            try:
                with open(candidate, "r") as f:
                    raw = f.read()
                ctl_path = candidate
                mux_base = base
                break
            except OSError as e:
                last_error = e
                continue

        if ctl_path is None:
            tried = ", ".join(os.path.join(b, "ctl") for b in self._MUX_BASES)
            self.append_text(
                f"[signal] cannot read mux ctl (tried {tried}): {last_error}\n",
                self.C_ERROR,
            )
            return None, []

        rio_port = self._RIO_9P_PORT_DEFAULT
        machines: list[tuple[str, str]] = []  # (name, host)

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2 or ":" not in parts[1]:
                continue
            name = parts[0]
            host, _, port_s = parts[1].rpartition(":")
            try:
                port = int(port_s)
            except ValueError:
                continue
            if port != rio_port:
                # Not a machine — likely the LLMFS line (5640). Skip.
                continue
            machines.append((name, host))

        return mux_base, machines

    def _signal_toggle(self, enable: bool):
        """
        /signal on  — full-mesh wire-up. For every ordered pair (M, N)
        of machines in /n/ctl on port 5641 with M ≠ N:

            echo 'subscribe N <N_addr>:5741' > /n/M/scene/signals/ctl

        where <N_addr> is N's host as reported in /n/ctl, except:
        when N's host is loopback (127.0.0.1 / localhost / ::1) **and**
        M is remote (host is *not* loopback), we substitute <N_addr>
        with our LAN IP. /n/ctl is mux-relative — a loopback entry
        means "on the same host as this mux", which from a remote
        machine's perspective is *us*, so the loopback is wrong to
        ship across the wire.

        /signal off — same mesh, `unsubscribe` instead (no addr).

        All writes go through the persistent shell so the user sees
        the actual `echo` lines scroll by. `set +e` so one unreachable
        machine doesn't stop the rest.
        """
        mux_base, machines = self._parse_n_ctl()
        if mux_base is None:
            return  # _parse_n_ctl already printed the error
        if len(machines) < 2:
            self.append_text(
                f"[signal] need ≥2 machines on port 5641 in /n/ctl, found {len(machines)}\n",
                self.C_INFO,
            )
            return

        verb = "subscribe" if enable else "unsubscribe"
        bus_port = self._SIGNAL_BUS_PORT
        mount = mux_base  # /n/mux or /n — whichever had the ctl
        loopback = self._LOOPBACK_HOSTS

        # Only pay the socket cost if there's any loopback-hosted machine
        # that might need rewriting. If everything is already on real
        # IPs, _detect_local_ip is wasted work.
        any_loopback = any(h in loopback for _, h in machines)
        local_ip = self._detect_local_ip() if any_loopback else None

        names = [m for m, _ in machines]
        suffix = f" (loopback→{local_ip} for remote targets)" if any_loopback else ""
        self.append_text(
            f"\n⟳ /signal {'on' if enable else 'off'} — full mesh across {names} "
            f"(bus port {bus_port}){suffix}\n",
            self.C_SYSTEM,
        )

        lines = ["set +e"]
        for target_name, target_host in machines:
            target_ctl = f"{mount}/{target_name}/scene/signals/ctl"
            target_is_remote = target_host not in loopback
            for other_name, other_host in machines:
                if other_name == target_name:
                    continue
                if enable:
                    # Rewrite loopback → LAN IP only when shipping the
                    # addr to a remote target; loopback-to-loopback is
                    # fine (same host on both ends).
                    addr_host = (
                        local_ip
                        if other_host in loopback and target_is_remote
                        else other_host
                    )
                    lines.append(
                        f"echo '{verb} {other_name} {addr_host}:{bus_port}' > {target_ctl}"
                    )
                else:
                    lines.append(f"echo {verb} {other_name} > {target_ctl}")

        self._execute_shell("\n".join(lines))

    def _setup_mounts(self):
        """
        Setup — clean unmount and remount 9pfuse for LLMFS and Rio.
        Uses the mux mount point if riomux is in use, or individual
        mounts for standalone mode.
        Retries mount up to 5 times with 1s delay (server may not be
        ready yet).
        """
        self.append_text("\n⟳ Setting up 9P mounts...\n", self.C_SYSTEM)
        
        # Determine ports from our config
        llm_port = self.p9_port     # default 5640
        rio_port = llm_port + 1     # default 5641
        mounts = [
            (self.rio_mount, rio_port),
            (self.llmfs_mount, llm_port),
        ]
        # Kill stale attachment scripts from previous runs
        subprocess.run(['pkill', '-f', 'llmfs_attach'], capture_output=True)
        subprocess.run(['pkill', '-f', 'acme_attach'], capture_output=True)
        # Build and execute the setup script
        script_lines = [
            'set +e',  # Don't exit on errors - we handle them ourselves
            '',
            '# Unmount existing mounts if present',
            f'pkexec sh -c "umount -f {self.llmfs_mount} 2>/dev/null || true; umount -f {self.rio_mount} 2>/dev/null || true"',
            'sleep 0.5',
            '',
        ]
        
        # Add mount logic for each mount point
        for mount_point, port in mounts:
            script_lines += [
                f'# --- {mount_point} (port {port}) ---',
                f'mkdir -p "{mount_point}"',
                f'MOUNTED=0',
                f'for i in 1 2 3 4 5; do',
                f'  if 9pfuse \'tcp!127.0.0.1!{port}\' "{mount_point}" 2>/dev/null; then',
                f'    echo "✓ {mount_point} mounted (port {port})"',
                f'    MOUNTED=1',
                f'    break',
                f'  fi',
                f'  echo "  retry $i for {mount_point}..."',
                f'  sleep 1',
                f'done',
                f'if [ "$MOUNTED" = "0" ]; then',
                f'  echo "✗ Failed to mount {mount_point}"',
                f'  exit 1',
                f'fi',
                '',
            ]
        
        script = '\n'.join(script_lines)
        self._execute_shell(script)
    
    def _scroll_to_bottom(self):
        """Helper to scroll terminal to bottom."""
        self._auto_scroll = True
        sb = self.terminal_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _request_scroll_coalesced(self):
        """Coalesced version of _scroll_to_bottom.

        Multiple producers (PTY output, agent stream, append_text from
        commands, etc.) used to schedule a fresh `QTimer.singleShot(0,
        _scroll_to_bottom)` per chunk. A burst of N chunks then queued
        N redundant scroll callbacks, each one a setValue() and a
        scrollbar signal emission. With fast output (e.g. `find /`,
        build logs, LLM token streaming) this dominated the GUI thread.

        This single-shot timer collapses any burst into one scroll per
        ~16 ms (one frame at 60 fps). It mirrors the StreamRouter's
        own _scroll_timer; in fact StreamRouter._request_scroll now
        delegates here, so all output paths share a single coalescer.
        """
        if not hasattr(self, '_coalesced_scroll_timer') or self._coalesced_scroll_timer is None:
            self._coalesced_scroll_timer = QTimer(self)
            self._coalesced_scroll_timer.setSingleShot(True)
            self._coalesced_scroll_timer.setInterval(16)
            self._coalesced_scroll_timer.timeout.connect(self._scroll_to_bottom)
        if not self._coalesced_scroll_timer.isActive():
            self._coalesced_scroll_timer.start()

    def _on_scroll_range_changed(self, _min, _max):
        """Scroll to bottom when content grows, if auto-scroll is active."""
        if self._auto_scroll:
            self.terminal_scroll.verticalScrollBar().setValue(_max)

    def _on_scroll_value_changed(self, value):
        """Track whether the user has scrolled away from the bottom."""
        sb = self.terminal_scroll.verticalScrollBar()
        # Consider "at bottom" if within 20px of maximum
        self._auto_scroll = value >= sb.maximum() - 20

    # ------------------------------------------------------------------
    # Peribus — the mycelium layer (/peribus command + feed -> $inline)
    # ------------------------------------------------------------------
    #
    # The peribus daemon (`python -m peribus --mount`) exposes the social
    # network at /n/peribus. Once mounted, this terminal:
    #   * Tails /n/peribus/feed/new in a worker thread.
    #   * Wraps each JSON post line as {"type":"post", ...} and routes it
    #     through the same $term/inline pipeline used by image/video/etc.,
    #     so feed posts render as cards in this terminal's output.
    #   * Adds a $peribus environment variable so shell snippets can
    #     `echo ... > $peribus/share/whatever` to publish.
    #
    # Posting model (matches the daemon):
    #   * Direct post:  write the body to /n/peribus/share/<name>; the
    #     SharedItemFile publishes on clunk.
    #   * Reply / DM:   write "<nodeid> <body>" to /n/peribus/inbox/send.
    #   * Bias feed:    write "attract <text>" to /n/peribus/ctl.
    #
    # We deliberately do NOT touch peribus.feed_bridge from here — that
    # module owns the canvas (QGraphicsScene) renderer; this is the
    # terminal-side renderer. Two views, one feed.

    def _handle_peribus_command(self, arg: str) -> None:
        """Dispatch /peribus subcommands."""
        parts = arg.split(maxsplit=1) if arg else []
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("", "start", "connect", "up"):
            self._peribus_start()
            return
        if sub in ("stop", "disconnect", "down"):
            self._peribus_stop()
            return
        if sub in ("status", "info"):
            self._peribus_status()
            return
        if sub == "post":
            if not rest:
                self.append_text("Usage: /peribus post <text>\n", self.C_ERROR)
                return
            self._peribus_post(rest)
            return
        if sub in ("attract", "follow"):
            if not rest:
                self.append_text(f"Usage: /peribus {sub} <text>\n", self.C_ERROR)
                return
            self._peribus_ctl(f"{sub} {rest}")
            return
        if sub == "ctl":
            if not rest:
                self.append_text("Usage: /peribus ctl <command>\n", self.C_ERROR)
                return
            self._peribus_ctl(rest)
            return
        if sub == "debug":
            # Toggle (or set explicitly with `on`/`off`) verbose logging
            # of every feed line and unwrap decision. Diagnostic — off by
            # default. With debug on, every incoming line dumps as:
            #   [peribus.dbg <verdict>] <first 240 chars of line>
            # where <verdict> is one of:
            #   U  unwrap matched (rich envelope detected & merged)
            #   =  passed through (plain text post or non-post JSON)
            #   X  outer JSON parse failed
            current = getattr(self, "_peribus_debug", False)
            if rest == "on":
                self._peribus_debug = True
            elif rest == "off":
                self._peribus_debug = False
            else:
                self._peribus_debug = not current
            state = "ON" if getattr(self, "_peribus_debug", False) else "OFF"
            self.append_text(
                f"peribus debug: {state}\n", self.C_INFO,
            )
            return
        if sub in ("help", "?", "h"):
            self._peribus_show_help()
            return

        self.append_text(
            f"Unknown /peribus subcommand: {sub}\n", self.C_ERROR
        )
        self.append_text(
            "Try: /peribus | /peribus stop | /peribus status | "
            "/peribus post <text> | /peribus help\n",
            self.C_INFO,
        )

    def _peribus_show_help(self) -> None:
        h = (
            "/peribus               Start daemon + mount, tail feed + inbox → $inline\n"
            "                        (public posts render normally; DMs render in pink)\n"
            "/peribus stop          Stop tailing feed and inbox (daemon keeps running)\n"
            "/peribus status        Show mount + tailer state\n"
            "/peribus post <text>   Publish a short post to your feed\n"
            "/peribus attract <t>   Bias your identity vector toward <t>\n"
            "/peribus follow <id>   Bias toward NodeID <id>\n"
            "/peribus ctl <line>    Raw write to /n/peribus/ctl\n"
            "/peribus debug [on|off]  Toggle verbose feed-line logging\n"
        )
        self.append_text(h, self.C_INFO)

    def _peribus_feed_path(self) -> str:
        return os.path.join(self._peribus_root, "feed", "new")

    def _peribus_is_mounted(self) -> bool:
        return os.path.ismount(self._peribus_root) or os.path.exists(
            self._peribus_feed_path()
        )

    def _peribus_start(self) -> None:
        """
        Bring up peribus end-to-end:
          1) launch `python -m peribus --mount` in the background if the
             feed file isn't reachable yet
          2) wait briefly for the mountpoint to appear
          3) start the in-process tailer that pipes posts into $inline
        """
        if self._peribus_tailer is not None and self._peribus_tailer.is_running():
            self.append_text(
                "peribus: already running — tailing "
                f"{self._peribus_feed_path()}\n",
                self.C_INFO,
            )
            return

        feed_path = self._peribus_feed_path()
        already_mounted = self._peribus_is_mounted()

        if not already_mounted:
            self.append_text(
                "\n⟳ Starting peribusd + mounting /n/peribus...\n",
                self.C_SYSTEM,
            )
            # `setsid` detaches from the controlling terminal so the daemon
            # outlives the shell session. `nohup` keeps it alive across
            # SIGHUP. Logs go to ~/.peribus/peribusd.log so we can debug
            # post-mortem without owning the process's stdout.
            #
            # We deliberately do NOT pkexec — peribus mounts via 9pfuse,
            # which the daemon handles itself (see __main__._ensure_mountpoint).
            # If 9pfuse needs root for a particular system, the user can run
            # the daemon out-of-band and just /peribus afterwards.
            script = (
                'set +e\n'
                'mkdir -p ~/.peribus\n'
                'if pgrep -f "python.*-m peribus" >/dev/null 2>&1; then\n'
                '  echo "peribusd already running"\n'
                'else\n'
                '  echo "launching peribusd..."\n'
                '  nohup setsid python -m peribus --mount '
                '>> ~/.peribus/peribusd.log 2>&1 &\n'
                '  disown 2>/dev/null || true\n'
                '  echo "peribusd PID=$! (logs: ~/.peribus/peribusd.log)"\n'
                'fi\n'
            )
            self._execute_shell(script)
            # Give the daemon + 9pfuse a moment to come up before we try
            # to attach the tailer. _execute_shell is async (writes to a
            # PTY), so use a QTimer rather than a sleep here.
            QTimer.singleShot(2000, self._peribus_start_tailer)
        else:
            self.append_text(
                "peribus: /n/peribus already mounted; attaching tailer\n",
                self.C_INFO,
            )
            self._peribus_start_tailer()

        # Make $peribus discoverable in shell scripts. We push it into the
        # already-running shell so users don't need to /restart.
        try:
            self._execute_shell_raw(
                f'export peribus="{self._peribus_root}"'
            )
        except Exception:
            # If the shell is dead, the user will see the standard
            # "[shell dead]" message via _execute_shell next time.
            pass

    def _peribus_start_tailer(self) -> None:
        """Attach the feed and inbox tailers once the mount is ready.

        Two tailers run in parallel:
          - feed/new  → public posts (rendered with the default theme)
          - inbox/new → direct messages (rendered with a pink theme so
                        they're visually distinct from feed posts —
                        opening a /peribus session means "check both
                        the public timeline and my mail" in one step)

        Both fail soft-independently: if the daemon doesn't yet expose
        inbox/ (older daemon, or transient mount race), the feed tailer
        still starts. We log a quiet info line for the missing inbox
        rather than treating it as an error.
        """
        feed_path = self._peribus_feed_path()
        if not os.path.exists(feed_path):
            self.append_text(
                f"peribus: feed not yet at {feed_path} — "
                "is the daemon up? (check ~/.peribus/peribusd.log)\n",
                self.C_ERROR,
            )
            return

        if self._peribus_tailer is not None:
            # Idempotent — replace any previous tailer cleanly.
            self._peribus_tailer.stop()
            self._peribus_tailer = None

        tailer = PeribusFeedTailer(feed_path, parent=self)
        # Qt.QueuedConnection is implicit when crossing threads, but make
        # it explicit so future code-readers don't have to think about it.
        tailer.line_received.connect(
            self._on_peribus_feed_line, type=Qt.QueuedConnection
        )
        tailer.error.connect(
            self._on_peribus_tailer_error, type=Qt.QueuedConnection
        )
        tailer.started.connect(
            lambda: self.append_text(
                f"✓ peribus tailer attached to {feed_path}\n",
                self.C_SUCCESS,
            ),
            type=Qt.QueuedConnection,
        )
        tailer.reconnected.connect(
            lambda: self.append_text(
                "↻ peribus tailer reconnected\n",
                self.C_INFO,
            ),
            type=Qt.QueuedConnection,
        )
        tailer.start()
        self._peribus_tailer = tailer

        # Now the inbox tailer — same shape, different file and signal.
        self._peribus_start_inbox_tailer()

    def _peribus_inbox_path(self) -> str:
        return os.path.join(self._peribus_root, "inbox", "new")

    def _peribus_start_inbox_tailer(self) -> None:
        """
        Attach the inbox tailer (DMs → pink post cards inline).

        Runs alongside the feed tailer. Same JSON-line-blocking-stream
        contract as feed/new — see filesystem.py:InboxFile.

        Lines look like: {"from": "<nodeid>", "ts": <unix>, "body": "..."}
        We reshape each into a post-style payload with `_card_kind: "inbox"`
        so _render_post can apply pink theming and a DM badge while
        re-using the rest of the post-rendering pipeline (byline, body,
        relative-time tick, reply action).
        """
        inbox_path = self._peribus_inbox_path()
        if not os.path.exists(inbox_path):
            # Soft fallback — the feed already started. Older daemons
            # may not expose inbox/, and mount races can briefly leave
            # the inbox tree absent. Log and continue.
            self.append_text(
                f"peribus: inbox not yet at {inbox_path} — "
                "DMs won't render until it appears\n",
                self.C_INFO,
            )
            return

        if self._peribus_inbox_tailer is not None:
            self._peribus_inbox_tailer.stop()
            self._peribus_inbox_tailer = None

        # Re-use PeribusFeedTailer — it's a generic blocking-line tailer,
        # not feed-specific. The signal wiring below makes it inbox-aware.
        tailer = PeribusFeedTailer(inbox_path, parent=self)
        tailer.line_received.connect(
            self._on_peribus_inbox_line, type=Qt.QueuedConnection
        )
        tailer.error.connect(
            self._on_peribus_tailer_error, type=Qt.QueuedConnection
        )
        tailer.started.connect(
            lambda: self.append_text(
                f"✓ peribus inbox attached to {inbox_path}\n",
                self.C_SUCCESS,
            ),
            type=Qt.QueuedConnection,
        )
        tailer.reconnected.connect(
            lambda: self.append_text(
                "↻ peribus inbox reconnected\n",
                self.C_INFO,
            ),
            type=Qt.QueuedConnection,
        )
        tailer.start()
        self._peribus_inbox_tailer = tailer

    def _peribus_stop(self) -> None:
        """Stop tailing the feed and inbox. The daemon itself keeps running."""
        had_any = (self._peribus_tailer is not None
                   or self._peribus_inbox_tailer is not None)
        if self._peribus_tailer is not None:
            self._peribus_tailer.stop()
            self._peribus_tailer = None
        if self._peribus_inbox_tailer is not None:
            self._peribus_inbox_tailer.stop()
            self._peribus_inbox_tailer = None
        if not had_any:
            self.append_text("peribus: tailer not running\n", self.C_INFO)
            return
        self.append_text(
            "peribus: tailer stopped (daemon untouched — "
            "kill it via the shell if you also want it down)\n",
            self.C_INFO,
        )

    def _peribus_status(self) -> None:
        mounted = self._peribus_is_mounted()
        running = (
            self._peribus_tailer is not None
            and self._peribus_tailer.is_running()
        )
        inbox_running = (
            self._peribus_inbox_tailer is not None
            and self._peribus_inbox_tailer.is_running()
        )
        lines = [
            f"peribus root:    {self._peribus_root}",
            f"  mounted:       {'yes' if mounted else 'no'}",
            f"  feed path:     {self._peribus_feed_path()}",
            f"  tailer:        {'running' if running else 'stopped'}",
            f"  inbox path:    {self._peribus_inbox_path()}",
            f"  inbox tailer:  {'running' if inbox_running else 'stopped'}",
        ]
        self.append_text("\n".join(lines) + "\n", self.C_INFO)

    def _peribus_post(self, text: str) -> None:
        """
        Publish a post by writing into /n/peribus/share/<auto-name>.

        The peribus filesystem treats any new file under share/ as a fresh
        post; the SharedItemFile.clunk() handler does the actual publish
        once the fd closes. We use a unique filename so concurrent /peribus
        post calls don't collide.
        """
        if not self._peribus_is_mounted():
            self.append_text(
                "peribus: not mounted — run /peribus first\n", self.C_ERROR
            )
            return
        # Sanity: the daemon enforces a 4 KiB cap on posts (see
        # widget_runtime.PeribusAPI.post). Match that so we fail fast in
        # the terminal rather than after the round-trip to the daemon.
        if len(text.encode("utf-8")) > 4096:
            self.append_text(
                "peribus: post too large (max 4 KiB)\n", self.C_ERROR
            )
            return
        # Slash-safe, sortable, distinct: ts + short uuid.
        name = f"post-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        path = os.path.join(self._peribus_root, "share", name)
        try:
            with open(path, "w") as f:
                f.write(text)
            self.append_text(
                f"✓ peribus: published as {name}\n", self.C_SUCCESS
            )
        except OSError as e:
            self.append_text(f"peribus post: {e}\n", self.C_ERROR)

    def _peribus_ctl(self, line: str) -> None:
        """Write a raw command to /n/peribus/ctl."""
        if not self._peribus_is_mounted():
            self.append_text(
                "peribus: not mounted — run /peribus first\n", self.C_ERROR
            )
            return
        ctl_path = os.path.join(self._peribus_root, "ctl")
        try:
            with open(ctl_path, "w") as f:
                f.write(line + "\n")
            self.append_text(f"peribus ctl: {line}\n", self.C_INFO)
        except OSError as e:
            self.append_text(f"peribus ctl: {e}\n", self.C_ERROR)

    def _peribus_reply_prompt(self, author: str) -> None:
        """
        Pop a small input dialog asking for the reply body, then send it
        as a DM via /n/peribus/inbox/send. Triggered from the post card's
        ↩ reply action.
        """
        if not self._peribus_is_mounted():
            self.append_text(
                "peribus: not mounted — run /peribus first\n", self.C_ERROR
            )
            return
        from PySide6.QtWidgets import QInputDialog
        body, ok = QInputDialog.getMultiLineText(
            self,
            "Reply",
            f"Direct message to @{author[:24]}…" if len(author) > 24
            else f"Direct message to @{author}",
            "",
        )
        if not ok:
            return
        body = body.strip()
        if not body:
            return
        send_path = os.path.join(self._peribus_root, "inbox", "send")
        # The daemon expects "<nodeid> <body>" on a single write — body may
        # contain newlines, that's fine, the parser treats everything
        # after the first whitespace as the body.
        try:
            with open(send_path, "w") as f:
                f.write(f"{author} {body}")
            self.append_text(
                f"✓ peribus: DM sent to @{author[:12]}…\n", self.C_SUCCESS
            )
        except OSError as e:
            self.append_text(f"peribus reply: {e}\n", self.C_ERROR)

    @Slot(str)
    def _on_peribus_feed_line(self, line: str) -> None:
        """
        A new JSON line arrived on /n/peribus/feed/new.

        Two layers of envelope are possible here, and we have to flatten
        them. The outer one is what the daemon ships on the feed:

            {"id": "...", "author": "...", "title": "post-1234-abcd",
             "body": "<file contents from share/>", "ts": ..., ...}

        That's authoritative for *identity* fields (id, author, ts,
        resonance, attachments) — the network agreed on them.

        But the body field is opaque to the daemon — it's just the bytes
        the publisher wrote into share/. When the publisher used /share,
        those bytes are themselves a JSON envelope:

            {"type": "post", "body": "look at this", "media": [...]}

        Without unwrapping, we'd render that inner JSON string as the
        post body — which is exactly the "JSON shows up as text in the
        card" symptom we're fixing.

        Strategy: if the outer body parses as JSON and looks like a rich
        envelope (dict with type=='post'), merge the two:
          - identity fields come from outer (daemon-authoritative)
          - content fields (body, title, media, attachments) come from
            inner where present, falling back to outer otherwise
        Plain-text posts (`echo hi > share/foo`) hit the JSONDecodeError
        path and stay unmodified.

        Keeping the routing through stream_router.insert_media(...) means
        a third party (an agent, a shell snippet) can still produce a
        post card by writing the rich envelope directly to $term/inline.
        That path doesn't pass through this method, so it's unaffected
        either way.
        """
        debug = getattr(self, "_peribus_debug", False)

        try:
            outer = json.loads(line)
        except json.JSONDecodeError as e:
            if debug:
                self.append_text(
                    f"[peribus.dbg X] parse error: {e} | "
                    f"first 240: {line[:240]!r}\n",
                    self.C_INFO,
                )
            return
        if not isinstance(outer, dict):
            if debug:
                self.append_text(
                    f"[peribus.dbg X] not a dict: {type(outer).__name__}\n",
                    self.C_INFO,
                )
            return

        # Dedup by stable post id. The daemon's feed/new replays the
        # buffer from the top whenever a client opens a fresh fid (the
        # cursor is per-fid and feed_cursor(from_start=True) starts at
        # -1). Tailer reopens — even rare ones from genuine FUSE EIO —
        # therefore deliver every buffered post again. Without this
        # guard each replay turns into a fresh card in the terminal,
        # which is the "messages reappear after reconnect" symptom.
        #
        # We dedup on the raw outer id rather than the unwrapped
        # merged.id because outer is daemon-authoritative for identity
        # fields (see the docstring above), so it's the most stable
        # value we have at this point.
        post_id = outer.get("id")
        if isinstance(post_id, str) and post_id:
            if post_id in self._peribus_seen_feed_set:
                if debug:
                    self.append_text(
                        f"[peribus.dbg D] dedup id={post_id[:16]}…\n",
                        self.C_INFO,
                    )
                return
            self._peribus_seen_feed_set.add(post_id)
            # deque(maxlen=N) evicts the oldest on overflow; mirror the
            # eviction into the lookup set so the two stay in sync.
            if len(self._peribus_seen_feed_ids) == self._peribus_seen_feed_ids.maxlen:
                evicted = self._peribus_seen_feed_ids[0]
                self._peribus_seen_feed_set.discard(evicted)
            self._peribus_seen_feed_ids.append(post_id)

        # Threading first, suppression second. We used to suppress own
        # echoes (via _share_is_recent_publish) before even considering
        # thread routing — but that leaves a failure mode where any
        # hiccup in the dedup register (terminal restart between publish
        # and echo, hash-mismatch between what the daemon returned and
        # what feed/new emits, GC of the entry, …) leaves the OP's
        # reply rendered as a freestanding card instead of under the
        # parent. We invert the order: route to thread first, suppress
        # second. The card itself dedupes by body (see
        # InlineMediaWidget.append_reply), so a stray double — local
        # echo plus gossip echo — collapses cleanly into one rendered
        # reply.
        #
        # If thread routing succeeds (return True), we don't need to
        # check suppression at all: the reply is already in the right
        # place, and the card's dedup made it idempotent.
        merged = self._unwrap_peribus_post(outer, _diag=(diag := [] if debug else None))
        merged.setdefault("type", "post")

        if debug:
            verdict = "U" if (diag and diag[0] == "unwrap-ok") else "="
            keys = sorted(outer.keys())
            body_preview = outer.get("body")
            if isinstance(body_preview, str):
                body_preview = body_preview[:120]
            reason = f" reason={diag[0]}" if diag else ""
            self.append_text(
                f"[peribus.dbg {verdict}]{reason} "
                f"keys={keys} body[:120]={body_preview!r} "
                f"merged_has_media={'media' in merged}\n",
                self.C_INFO,
            )

        if self._route_to_thread_card(merged):
            return

        # Now check own-echo suppression for top-level posts only. If
        # routing failed (no parent card mounted, e.g. parent scrolled
        # past), suppression still protects against double-rendering
        # our own freshly-published top-level posts.
        if self._share_is_recent_publish(outer):
            if debug:
                self.append_text(
                    "[peribus.dbg .] suppressed own echo (top-level)\n",
                    self.C_INFO,
                )
            return

        try:
            self.stream_router.insert_media(merged)
        except Exception as e:
            # Render errors should not silently kill the tailer — surface
            # them so the user knows something is up, then keep going.
            self.append_text(
                f"peribus: render error: {e}\n", self.C_ERROR
            )

    @staticmethod
    def _unwrap_peribus_post(outer: dict, _diag: Optional[list] = None) -> dict:
        """
        Merge a daemon-shipped post envelope with a rich inner envelope,
        if the body field happens to be one. See _on_peribus_feed_line
        for the design rationale.

        Identity comes from the outer envelope (the network agreed on it);
        content (body, title, media, attachments) comes from the inner
        when present, with outer as fallback.

        If `_diag` is a list, this method appends a single short string
        explaining which gate decided to (not) unwrap. Used only by
        debug logging — production callers pass None and pay nothing.
        """
        body = outer.get("body")
        if not isinstance(body, str):
            if _diag is not None:
                _diag.append(f"body-not-str={type(body).__name__}")
            return dict(outer)
        stripped = body.lstrip()
        if not stripped.startswith("{"):
            if _diag is not None:
                _diag.append(f"body-no-brace head={stripped[:20]!r}")
            return dict(outer)

        try:
            inner = json.loads(stripped)
        except (json.JSONDecodeError, ValueError) as e:
            if _diag is not None:
                # Show the parse error AND the tail of the body, since
                # truncation-during-publish is the most likely cause of
                # parse failure on otherwise-valid envelopes.
                tail = stripped[-60:] if len(stripped) > 60 else stripped
                _diag.append(
                    f"json-parse-fail err={e} "
                    f"body_len={len(stripped)} "
                    f"tail={tail!r}"
                )
            return dict(outer)
        if not isinstance(inner, dict):
            if _diag is not None:
                _diag.append(f"inner-not-dict={type(inner).__name__}")
            return dict(outer)
        # Only unwrap if the inner envelope is explicitly a post. A user
        # who happens to write `{"some": "json"}` as a post body should
        # see that text rendered, not have it interpreted as an envelope.
        if inner.get("type") != "post":
            if _diag is not None:
                _diag.append(f"inner-type={inner.get('type')!r}")
            return dict(outer)

        if _diag is not None:
            _diag.append("unwrap-ok")

        merged = dict(outer)
        # Inner wins for these — they are the publisher's intent. Note
        # that we deliberately *replace* attachments rather than concat,
        # because a publisher producing both lists likely intends the
        # inner one to be canonical (the outer comes from share/'s file
        # bytes which the publisher already controls).
        #
        # `reply_to` / `reply_to_author` are the threading hints set by
        # _peribus_send_reply_post on the publisher. They MUST survive
        # the unwrap or _route_to_thread_card on the receiver can't
        # find the parent card and the reply renders as a freestanding
        # post with no relationship to its parent.
        for key in ("body", "media", "attachments",
                    "reply_to", "reply_to_author"):
            if key in inner:
                merged[key] = inner[key]

        # Title is special. The daemon auto-generates one from the
        # share/ filename (e.g. "post-1777826323-51572a"), which is
        # noise once we have the post id displayed elsewhere. If the
        # publisher used /share, they didn't set a title; respect that
        # by dropping the outer auto-title rather than displaying it.
        # If the inner envelope DOES provide a real title, use it.
        if "title" in inner:
            merged["title"] = inner["title"]
        else:
            merged.pop("title", None)

        # The inner envelope's "ts" might be the publisher's local clock
        # at /share time, which is more meaningful than the daemon's
        # gossip-arrival timestamp. Prefer it when present and plausible.
        inner_ts = inner.get("ts")
        if isinstance(inner_ts, (int, float)) and inner_ts > 0:
            merged["ts"] = inner_ts

        return merged

    @Slot(str)
    def _on_peribus_tailer_error(self, msg: str) -> None:
        self.append_text(f"peribus tailer: {msg}\n", self.C_ERROR)

    @Slot(str)
    def _on_peribus_inbox_line(self, line: str) -> None:
        """
        A new JSON line arrived on /n/peribus/inbox/new.

        Inbox messages are simpler than feed posts — no double envelope,
        no media, no attachments. Just {from, ts, body}. We reshape into
        a post-style payload with `_card_kind: "inbox"` and route through
        stream_router.insert_media so the existing post renderer handles
        layout, byline-tick, and reply.

        The `_card_kind` flag tells _render_post to use a pink-tinted
        palette and a "DM" badge so DMs are obvious vs. public posts.
        Stays compatible with everything else: a third-party producing
        a DM-shaped envelope can render via $term/inline by writing the
        same payload directly.
        """
        debug = getattr(self, "_peribus_debug", False)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            if debug:
                self.append_text(
                    f"[peribus.inbox.dbg X] parse error: {e} | "
                    f"first 240: {line[:240]!r}\n",
                    self.C_INFO,
                )
            return
        if not isinstance(msg, dict):
            return

        sender = msg.get("from") or "?"
        body = msg.get("body") or ""
        ts = msg.get("ts")

        # Build a post-shaped payload. Author = sender so the byline,
        # reply target, and self-DM detection all work the same way as
        # for a feed post. The "_card_kind" sentinel triggers pink theming
        # in _render_post. We also synthesize a stable id from sender+ts
        # so the byline tick (which keys on payload id when present)
        # has something to hash against.
        if isinstance(ts, (int, float)) and ts > 0:
            synth_id = f"dm:{sender}:{int(ts * 1000)}"
        else:
            # No usable ts: fall back to a content-derived id so the
            # dedup downstream still works across tailer reopens.
            # Using time.time() here (as the original did) would mint
            # a fresh id on every replay, which is exactly what dedup
            # needs to prevent. A short blake2b over (sender + body)
            # is collision-resistant enough for inbox dedup and stable
            # across reopens.
            import hashlib
            digest = hashlib.blake2b(
                f"{sender}\x00{body}".encode("utf-8"),
                digest_size=12,
            ).hexdigest()
            synth_id = f"dm:{sender}:{digest}"

        # Dedup by id. Same rationale as the feed path: tailer reopens
        # cause the daemon to replay the inbox buffer, and we don't
        # want to render the same DM twice.
        if synth_id in self._peribus_seen_inbox_set:
            if debug:
                self.append_text(
                    f"[peribus.inbox.dbg D] dedup id={synth_id[:24]}…\n",
                    self.C_INFO,
                )
            return
        self._peribus_seen_inbox_set.add(synth_id)
        if len(self._peribus_seen_inbox_ids) == self._peribus_seen_inbox_ids.maxlen:
            evicted = self._peribus_seen_inbox_ids[0]
            self._peribus_seen_inbox_set.discard(evicted)
        self._peribus_seen_inbox_ids.append(synth_id)

        payload = {
            "type": "post",
            "_card_kind": "inbox",
            "id": synth_id,
            "author": sender,
            "ts": ts if isinstance(ts, (int, float)) else time.time(),
            "body": body,
        }

        if debug:
            sender_short = sender if len(sender) <= 24 else sender[:12] + "…"
            preview = body[:80].replace("\n", "\\n")
            self.append_text(
                f"[peribus.inbox.dbg ✉] from={sender_short} "
                f"body[:80]={preview!r}\n",
                self.C_INFO,
            )

        # If we already have a DM card for this peer, append the new
        # message there so the conversation reads as a single growing
        # thread. Otherwise fall through to creating a fresh top-level
        # card via insert_media.
        if self._route_to_thread_card(payload):
            return

        try:
            self.stream_router.insert_media(payload)
        except Exception as e:
            self.append_text(
                f"peribus inbox: render error: {e}\n", self.C_ERROR,
            )

    # ---- Live byline tick for post cards ---------------------------------
    #
    # Post cards show relative times ("3m ago"). Without help they'd freeze
    # at whatever delta was current when they were rendered. Rather than
    # spawn a per-card timer, the terminal owns one shared QTimer and asks
    # every registered card to refresh its byline. We register on first use
    # and tear down to nothing when the last card is gone, so idle
    # terminals (no posts ever displayed) pay zero cost.

    # 30 s is the right granularity: under "Xm ago" each minute matters,
    # but a tick faster than that would just produce no visible change for
    # most cards. (For posts older than an hour the label only changes
    # once per hour, so a slower tick is fine — we deliberately don't
    # adapt the period; uniform behavior is easier to reason about.)
    _POST_TICK_INTERVAL_MS = 30_000

    def _register_post_card(self, card) -> None:
        """Hook a fresh InlineMediaWidget post card into the live tick."""
        if not hasattr(self, "_post_cards"):
            self._post_cards = set()
        self._post_cards.add(card)
        # Lazy-create the timer the first time we have something to tick.
        if not hasattr(self, "_post_tick_timer") or self._post_tick_timer is None:
            self._post_tick_timer = QTimer(self)
            self._post_tick_timer.setInterval(self._POST_TICK_INTERVAL_MS)
            self._post_tick_timer.timeout.connect(self._tick_post_cards)
        if not self._post_tick_timer.isActive():
            self._post_tick_timer.start()

    def _unregister_post_card(self, card) -> None:
        """Drop a card from the tick set; stop the timer when set is empty."""
        if not hasattr(self, "_post_cards"):
            return
        self._post_cards.discard(card)
        if (not self._post_cards
                and hasattr(self, "_post_tick_timer")
                and self._post_tick_timer is not None
                and self._post_tick_timer.isActive()):
            self._post_tick_timer.stop()

    # ---- thread-card index --------------------------------------------
    #
    # When a post card or DM card is rendered in the terminal we keep
    # an index so subsequent replies / DMs from the same thread can
    # find the original card and append into it instead of producing a
    # second top-level card.
    #
    # Two separate maps:
    #
    #   _post_card_index : post_id (str) -> InlineMediaWidget
    #     Public posts. When a reply-post arrives (envelope contains
    #     reply_to=<parent_post_id>) we look up the parent and append.
    #
    #   _dm_card_index : peer NodeID (str) -> InlineMediaWidget
    #     Inbox cards. When a DM arrives from a peer we already have a
    #     thread for, route the new message to that card. When we
    #     initiate a DM via the post card's reply button, the
    #     optimistic local echo also goes through this index — we
    #     create a card if one didn't exist yet so subsequent peer
    #     replies thread under it.
    #
    # Index entries are wiped from _unregister_thread_card, which is
    # connected to each card's `destroyed` signal in _render_post.

    def _register_thread_card(self, card) -> None:
        """Index a card so replies / DMs can find it by id or peer."""
        if not hasattr(self, "_post_card_index"):
            self._post_card_index = {}
        if not hasattr(self, "_dm_card_index"):
            self._dm_card_index = {}
        if getattr(card, "_is_inbox_card", False):
            peer = getattr(card, "_post_author", None)
            if peer:
                self._dm_card_index[peer] = card
        else:
            post_id = getattr(card, "_post_id", None)
            if post_id:
                self._post_card_index[post_id] = card
            else:
                # A post card with no id can't be threaded under: peer
                # replies tagged `reply_to=<hash>` will miss the index
                # and render as freestanding top-level posts. This used
                # to happen silently when the local-echo path rendered
                # the parent before the daemon assigned a hash. Now
                # `_share_publish_text` / `_share_publish_file` commit
                # first and stamp the hash into the envelope, so this
                # branch should be unreachable in normal flow. Surface
                # it (debug only) so any future regression is visible
                # instead of silently breaking replies.
                if getattr(self, "_peribus_debug", False):
                    self.append_text(
                        "[peribus.dbg !] post card rendered without id — "
                        "peer replies to this card will not thread\n",
                        self.C_INFO,
                    )

    def _unregister_thread_card(self, card) -> None:
        """Drop indices for a card that's going away."""
        if hasattr(self, "_post_card_index"):
            post_id = getattr(card, "_post_id", None)
            if post_id and self._post_card_index.get(post_id) is card:
                self._post_card_index.pop(post_id, None)
        if hasattr(self, "_dm_card_index"):
            peer = getattr(card, "_post_author", None)
            if peer and self._dm_card_index.get(peer) is card:
                self._dm_card_index.pop(peer, None)

    def _route_to_thread_card(self, payload: dict) -> bool:
        """
        If `payload` belongs to an existing thread card, append to it
        and return True. Otherwise return False so the caller can
        render it as a top-level card.

        Routing rules:
          - DM payload (`_card_kind: "inbox"`) → look up by author in
            the DM index; if a card exists, append.
          - Reply post (envelope has `reply_to`) → look up parent by
            id in the post-card index; if a card exists, append.
        """
        # DM routing.
        if payload.get("_card_kind") == "inbox":
            peer = (payload.get("author") or "").strip()
            if not peer or not hasattr(self, "_dm_card_index"):
                return False
            card = self._dm_card_index.get(peer)
            if card is None:
                return False
            try:
                card.append_reply(payload)
                return True
            except RuntimeError:
                # Card's QWidget was deleted; clean the index.
                self._dm_card_index.pop(peer, None)
                return False

        # Reply-post routing.
        reply_to = (payload.get("reply_to") or "").strip()
        if reply_to and hasattr(self, "_post_card_index"):
            card = self._post_card_index.get(reply_to)
            if card is None:
                return False
            try:
                card.append_reply(payload)
                return True
            except RuntimeError:
                self._post_card_index.pop(reply_to, None)
                return False

        return False

    # ---- DM + reply-post send paths -----------------------------------

    def _peribus_send_dm(self, peer: str, body: str) -> bool:
        """
        Send a DM to a peer by writing "<peer> <body>" to inbox/send.

        Returns True on a successful write. Used by the inline reply
        composer on DM cards. Bypasses the popup dialog that
        _peribus_reply_prompt uses (kept around for the old code path
        until everything moves over).
        """
        if not self._peribus_is_mounted():
            self.append_text(
                "peribus: not mounted — run /peribus first\n", self.C_ERROR,
            )
            return False
        if len(body.encode("utf-8")) > 4096:
            self.append_text(
                "peribus DM: too large (max 4 KiB)\n", self.C_ERROR,
            )
            return False
        send_path = os.path.join(self._peribus_root, "inbox", "send")
        try:
            with open(send_path, "w") as f:
                f.write(f"{peer} {body}")
            return True
        except OSError as e:
            self.append_text(f"peribus DM: {e}\n", self.C_ERROR)
            return False

    def _peribus_send_reply_post(self, parent_id: str,
                                 parent_author: str,
                                 body: str) -> bool:
        """
        Publish a reply-tagged post via the clone draft interface.

        Allocates a fresh draft, writes the body and reply_to attribute
        files, and commits. The daemon's publish_draft auto-promotes
        kind='post' to 'reply' when reply_to is set, so we don't have
        to write the kind file explicitly.

        Wire shape (built by the daemon, no longer hand-constructed
        here): a normal post envelope with a `reply_to` field. Receivers
        running this client recognize it and route into the parent
        card's thread surface; older receivers see a top-level post
        with a "↳ reply to <hash>" tag — graceful degradation.
        """
        if not self._peribus_is_mounted():
            self.append_text(
                "peribus: not mounted — run /peribus first\n", self.C_ERROR,
            )
            return False
        if not body.strip():
            return False
        if len(body.encode("utf-8")) > 4096:
            self.append_text(
                "peribus reply: too large (max 4 KiB)\n", self.C_ERROR,
            )
            return False

        # The local echo happens inside the parent card itself —
        # _submit_reply calls self.append_reply(...) directly on the
        # InlineMediaWidget after this returns successfully. We do NOT
        # call _share_render_locally here: that path renders via the
        # stream router (top-level cards) and only learned about
        # thread-routing later. Going direct through the card avoids
        # the entire thread-index lookup, the envelope-shape dance,
        # and any timing window where the parent might not yet be
        # registered. The /share dedupe register set below still
        # suppresses the gossip-feed echo of our own publish.

        draft_path = self._share_open_draft()
        if draft_path is None:
            return False

        if not self._share_write_attr(draft_path, "body", body):
            self._share_discard_draft(draft_path)
            return False
        if not self._share_write_attr(draft_path, "reply_to", parent_id):
            self._share_discard_draft(draft_path)
            return False
        # The daemon auto-promotes kind to "reply" when reply_to is set,
        # so we don't write `kind` here. We could write it explicitly
        # for clarity, but leaving the default in place keeps the
        # publisher's wire footprint smaller.

        post_hash = self._share_commit_draft(draft_path)
        if post_hash is None:
            return False
        self._share_remember_publish(post_hash)
        return True

    def _tick_post_cards(self) -> None:
        """Refresh every live post card's relative timestamp."""
        if not hasattr(self, "_post_cards") or not self._post_cards:
            return
        # Snapshot to allow a card to unregister itself mid-iteration if
        # its underlying QObject was deleted between ticks (the destroyed
        # signal handles that, but races are possible across threads).
        for card in list(self._post_cards):
            try:
                card._refresh_post_byline()
            except RuntimeError:
                # Underlying QWidget went away without us hearing destroyed
                # (rare but possible on app shutdown). Drop it.
                self._post_cards.discard(card)
            except Exception:
                # Don't let one bad card kill the tick for everybody else.
                pass

    def _copy_to_clipboard(self, text: str) -> None:
        """Generic clipboard helper, used by post-card '⎘ copy path' button."""
        try:
            QApplication.clipboard().setText(text)
            preview = text if len(text) <= 80 else text[:77] + "…"
            self.append_text(
                f"📋 copied: {preview}\n", self.C_INFO
            )
        except Exception as e:
            self.append_text(f"clipboard: {e}\n", self.C_ERROR)

    # ------------------------------------------------------------------
    # /share — high-level publishing for posts, code, and media
    # ------------------------------------------------------------------
    #
    # The raw filesystem (`echo … > /n/peribus/share/foo`, `cp f.png
    # /n/peribus/share`) works but always publishes opaque bytes — the
    # receiver gets a post body of "PNG\x89..." and renders it as broken
    # text. /share builds the *rich* envelope that InlineMediaWidget on
    # the receiving side already knows how to render, and writes that
    # envelope (one JSON line) into share/<auto-name>.
    #
    # Argument shapes:
    #   /share                      → open the composer dialog
    #   /share path/to/file         → publish file as media (auto-detect kind)
    #   /share path/to/file caption → file as media + caption as post body
    #   /share some text here       → publish as plain-text post
    #
    # Drag-and-drop a file onto the terminal does the same thing as
    # passing the file path. A `share` shell function is exported in the
    # preamble so the same UX works from $-mode too.

    # Soft caps on inlining file bytes into the wire envelope. Above
    # ------------------------------------------------------------------
    # Publishing via /n/peribus/share/clone — the new draft interface.
    # ------------------------------------------------------------------
    #
    # The widget no longer constructs JSON envelopes or hand-uploads
    # carrier blobs to /n/peribus/share/. Both jobs live in the daemon
    # now: the widget describes a post by writing to attribute files
    # under share/<n>/, then commits with `echo publish > share/<n>/ctl`.
    #
    # Each publish goes through these steps:
    #
    #   1. cat share/clone        → "<n>"            allocate draft
    #   2. echo … > share/<n>/body                   write text
    #   3. echo path > share/<n>/attach              queue attachments
    #   4. echo b3:hash > share/<n>/reply_to         (if reply)
    #   5. echo publish > share/<n>/ctl              commits — blocks
    #   6. cat share/<n>/result                      → post hash
    #
    # That's it. No wire envelopes, no carrier-blob fiddling, no
    # 280-byte caption clipping, no echo-dedupe prefixes — the daemon
    # handles all of it now and the round-trip echo arrives with the
    # author set to our own NodeID, which is sufficient signal for
    # local-publish dedup (see _share_is_own_post).
    #
    # The local-render side (showing the post in our terminal before
    # the wire round-trip) is unchanged: we still call
    # stream_router.insert_media with the rich local envelope.

    def _share_open_draft(self) -> Optional[str]:
        """
        Allocate a fresh draft directory by reading share/clone.

        Returns the draft path (e.g. "/n/peribus/share/3") or None on
        failure. The draft is owned by the daemon for cleanup; the widget
        just writes attribute files into it.
        """
        if not self._peribus_is_mounted():
            self.append_text(
                "share: /n/peribus not mounted — run /peribus first\n",
                self.C_ERROR,
            )
            return None
        clone_path = os.path.join(self._peribus_root, "share", "clone")
        try:
            with open(clone_path, "r") as f:
                n = f.read().strip()
        except OSError as e:
            self.append_text(f"share: clone read failed: {e}\n", self.C_ERROR)
            return None
        if not n.isdigit():
            self.append_text(
                f"share: clone returned unexpected value {n!r}\n",
                self.C_ERROR,
            )
            return None
        return os.path.join(self._peribus_root, "share", n)

    def _share_write_attr(self, draft_path: str, attr: str, value: str,
                          *, append: bool = False) -> bool:
        """Write a single attribute file under the draft directory."""
        path = os.path.join(draft_path, attr)
        mode = "a" if append else "w"
        try:
            with open(path, mode) as f:
                f.write(value)
            return True
        except OSError as e:
            self.append_text(
                f"share: {attr} write failed: {e}\n", self.C_ERROR,
            )
            return False

    def _share_commit_draft(self, draft_path: str) -> Optional[str]:
        """
        Send `publish` to the draft's ctl file, then read result.

        Returns the post hash on success, None on failure. Blocks for
        the whole publish (the daemon's ctl write doesn't return until
        the post is signed and broadcast); the result read is
        instantaneous after that since the future is already set.
        """
        ctl_path = os.path.join(draft_path, "ctl")
        result_path = os.path.join(draft_path, "result")
        try:
            with open(ctl_path, "w") as f:
                f.write("publish")
        except OSError as e:
            self.append_text(f"share: publish failed: {e}\n", self.C_ERROR)
            return None
        try:
            with open(result_path, "r") as f:
                outcome = f.read().strip()
        except OSError as e:
            self.append_text(f"share: result read failed: {e}\n", self.C_ERROR)
            return None
        if outcome.startswith("b3:"):
            return outcome
        # Anything else (error: …, discarded) is a failure path.
        self.append_text(f"share: {outcome}\n", self.C_ERROR)
        return None

    def _share_discard_draft(self, draft_path: str) -> None:
        """Best-effort cleanup of an aborted draft."""
        ctl_path = os.path.join(draft_path, "ctl")
        try:
            with open(ctl_path, "w") as f:
                f.write("discard")
        except OSError:
            pass  # Daemon's reaper will sweep it eventually.

    def _share_render_locally(self, envelope: dict) -> None:
        """
        Show the rich (uncapped) envelope in our own terminal immediately,
        without waiting for the wire round-trip. Used by all publish
        helpers so the publisher sees their post the moment they hit
        send.

        Reply handling: if the envelope is tagged as a reply
        (``reply_to`` field set), try to thread it under the parent card
        the same way peer replies are threaded — via
        ``_route_to_thread_card``. Without this, an OP replying to their
        own post would always land in a fresh top-level card, while
        every *peer's* reply correctly threads under the parent. The
        asymmetry was a bug, not a design choice: the receive side
        (``_on_peribus_feed_line`` / ``_on_peribus_inbox_line``)
        consults the thread index before falling through to
        ``insert_media``; the local-echo path didn't. We mirror that
        check here so the publisher's view matches what their peers see.
        """
        local = dict(envelope)
        local.setdefault("type", "post")
        local.setdefault("ts", time.time())
        local.setdefault("author", "(local)")

        # If the envelope is a reply (post or DM), try threading first.
        # _route_to_thread_card returns False when no parent card is
        # mounted (e.g. publisher just /peribus'd in and the parent
        # scrolled past before the index was built); in that case we
        # fall through to a normal top-level render.
        try:
            if self._route_to_thread_card(local):
                return
        except Exception as e:
            # Defensive: a broken thread index shouldn't suppress the
            # local echo entirely. Log and fall through.
            self.append_text(
                f"share: thread routing error: {e}\n", self.C_ERROR,
            )

        try:
            self.stream_router.insert_media(local)
        except Exception as e:
            self.append_text(f"share: local render error: {e}\n", self.C_ERROR)

    # Mapping from filename extension to InlineMediaWidget `type`. Used
    # by _share_kind_for_path to drive the LOCAL render — the daemon
    # decides what to ship over the wire (envelope-with-attachments via
    # publish_draft), so the wire side no longer needs to know about
    # this map. Keep in sync with InlineMediaWidget._build_body's
    # dispatch table.
    _SHARE_EXT_KIND = {
        # raster images
        ".png": "image", ".jpg": "image", ".jpeg": "image",
        ".webp": "image", ".bmp": "image",
        # animated
        ".gif": "gif",
        # audio
        ".mp3": "audio", ".wav": "audio", ".ogg": "audio",
        ".flac": "audio", ".m4a": "audio",
        # video
        ".mp4": "video", ".mkv": "video", ".webm": "video",
        ".mov": "video", ".avi": "video",
        # documents
        ".pdf": "pdf",
        # 3D
        ".obj": "model3d", ".stl": "model3d", ".glb": "model3d",
        ".gltf": "model3d", ".ply": "model3d",
        # code (rendered by the python sandbox, after security review)
        ".py": "python",
        # markup
        ".html": "html", ".htm": "html", ".svg": "html",
    }

    def _handle_share_command(self, arg: str) -> None:
        """Top-level dispatch for /share."""
        arg = arg.strip()
        if not arg:
            self._share_open_composer()
            return

        # ---- /share scene [caption]  /share context [caption] ----
        # Snapshot the live CONTEXT (compacted scene code) and publish
        # it as a .py post. The renderer treats .py specially via
        # _SHARE_EXT_KIND, so peers see syntax-highlighted code, not an
        # opaque blob. The remainder of the arg becomes the caption.
        first, _, rest = arg.partition(" ")
        if first.lower() in ("scene", "context"):
            self._share_publish_context(caption=rest.strip())
            return

        # Try to interpret the first token as a file path. Quoted paths
        # ("my doc.png" or 'foo bar.gif') let users include spaces.
        path, rest = self._share_split_path_and_caption(arg)
        if path is not None and os.path.isfile(path):
            self._share_publish_file(path, caption=rest)
            return

        # Not a file — treat the whole arg as post body text.
        self._share_publish_text(arg)

    def _resolve_context_file(self):
        """
        Find the in-process SmartContextFile object.

        CONTEXT is a *blocking* 9P stream — reading it from the FS mount
        deadlocks the Qt event loop until the next code execution wakes
        it. So instead of `open(<rio_mount>/CONTEXT).read()`, we reach
        RioWindow (which holds .rio_server.filesystem.context_file) and
        call get_all_code() directly. Same data the AI flicker bridge
        uses in main.py — pure synchronous Python, no 9P round-trip.

        The trick: in-scene terminals are embedded via
        QGraphicsProxyWidget (see RioWindow._handle_terminal_mouse_press —
        the terminal is created with NO parent and then wrapped by
        graphics_scene.addWidget). When a QWidget is proxied, its
        parent() is None — Qt doesn't expose the proxy through the
        QWidget parent chain. So self.window() returns the terminal
        itself, not RioWindow, and the obvious walk fails.

        We try, in order:
          1) self.window().rio_server  — works for popped/floating
             terminals that ARE real children of a QMainWindow.
          2) graphicsProxyWidget().scene().views()[0].window().rio_server
             — works for scene-embedded terminals. The proxy knows its
             scene, the scene knows its views, and the view's window
             is RioWindow.
          3) Same dance via window().rio_window.rio_server, in case
             the popped-out main window stashes a back-reference.

        Returns the context_file object or None if every probe fails.
        """
        def _from(rio_server):
            if rio_server is None:
                return None
            fs = getattr(rio_server, "filesystem", None)
            if fs is None:
                return None
            return getattr(fs, "context_file", None)

        # --- Path 1: real Qt parent chain (popped / floating terminals)
        win = self.window()
        if win is not None:
            cf = _from(getattr(win, "rio_server", None))
            if cf is not None:
                return cf
            # Pop-out windows may stash a back-ref to the original window.
            cf = _from(getattr(getattr(win, "rio_window", None),
                                "rio_server", None))
            if cf is not None:
                return cf

        # --- Path 2: through the QGraphicsProxyWidget (scene-embedded)
        # graphicsProxyWidget() returns the proxy that's hosting this
        # widget in a QGraphicsScene, or None if not proxied.
        proxy = self.graphicsProxyWidget() if hasattr(
            self, "graphicsProxyWidget") else None
        if proxy is not None:
            scene = proxy.scene()
            if scene is not None:
                for view in scene.views():
                    vw = view.window() if view is not None else None
                    if vw is None:
                        continue
                    cf = _from(getattr(vw, "rio_server", None))
                    if cf is not None:
                        return cf
                    cf = _from(getattr(getattr(vw, "rio_window", None),
                                        "rio_server", None))
                    if cf is not None:
                        return cf

        # --- Path 3: last-resort QApplication scan. RioWindow is a
        # QMainWindow subclass; in a Rio session there's exactly one of
        # them holding the rio_server. Cheap, deterministic, and only
        # runs when paths 1 & 2 didn't find anything (e.g. exotic
        # parenting we haven't anticipated).
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                for tlw in app.topLevelWidgets():
                    cf = _from(getattr(tlw, "rio_server", None))
                    if cf is not None:
                        return cf
        except Exception:
            pass

        return None

    def _share_publish_context(self, caption: str = "") -> None:
        """
        Publish the current CONTEXT as a Python-code post.

        Pulls the live compacted snapshot directly from the in-process
        SmartContextFile (NOT from the 9P mount — that path's read is
        blocking and would freeze the event loop). Stages the snapshot
        in a temp `.py` file so the existing _SHARE_EXT_KIND mapping
        picks ``python`` as the post kind, then hands off to
        _share_publish_file. The temp file is cleaned up after the
        publish + local-render round-trip completes.
        """
        context_file = self._resolve_context_file()
        if context_file is None:
            self.append_text(
                "share scene: cannot reach the CONTEXT object — "
                "rio_server.filesystem.context_file not found via parent chain\n",
                self.C_ERROR,
            )
            return

        # SmartContextFile.get_all_code() is sync and idempotent — it
        # just runs the compactor over the accumulated blocks and
        # returns the cached string. No I/O, no event-loop interaction.
        try:
            code = context_file.get_all_code()
        except Exception as e:
            self.append_text(
                f"share scene: get_all_code() failed ({e})\n",
                self.C_ERROR,
            )
            return

        code = (code or "").strip()
        if not code:
            self.append_text(
                "share scene: CONTEXT is empty — nothing to publish yet "
                "(execute some code first)\n",
                self.C_ERROR,
            )
            return

        # Stage the snapshot in a temp .py file so _share_publish_file
        # picks the python kind and the daemon stores plain bytes that
        # peers can re-render with the python sandbox.
        import tempfile
        ts = time.strftime("%Y%m%d-%H%M%S")
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"scene-{ts}-",
                suffix=".py",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(code)
                    if not code.endswith("\n"):
                        f.write("\n")
            except Exception:
                # fdopen took ownership; if write failed, best-effort cleanup
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            self.append_text(
                f"share scene: cannot stage snapshot ({e})\n",
                self.C_ERROR,
            )
            return

        try:
            self._share_publish_file(tmp_path, caption=caption)
        finally:
            # The publish path has both rendered locally (which copied or
            # referenced the bytes) and committed via the clone draft
            # (which the daemon has already content-addressed). The temp
            # file is no longer needed. If the local renderer is still
            # holding the path lazily, it'll have buffered what it needs
            # by now — but if you ever see broken local thumbnails for
            # /share scene, this is the first place to look.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _share_split_path_and_caption(self, arg: str):
        """
        Try to parse the argument as `<path>` or `<path> <caption…>`.

        Returns (path, caption_or_empty) if the leading token resolves to
        an existing file, or (None, "") if not. Handles three cases:
          - quoted path:    "my pic.png" some caption   →  ("my pic.png", "some caption")
          - bare path:      ./demo.gif look at this     →  ("./demo.gif",  "look at this")
          - whole-arg path: ~/some long name with spaces.png
                            →  ("/home/u/some long name with spaces.png", "")

        We deliberately try the whole arg as a path first (after expansion)
        so files with embedded spaces work without the user having to quote.
        """
        # Expand ~ and env vars so $HOME etc. work.
        whole = os.path.expanduser(os.path.expandvars(arg))

        # 1) The entire argument might *be* a path with spaces. Try it
        #    first — if it resolves, we're done.
        if os.path.isfile(whole):
            return whole, ""

        # 2) Quoted path?
        for q in ('"', "'"):
            if arg.startswith(q):
                end = arg.find(q, 1)
                if end > 0:
                    inner = arg[1:end]
                    rest = arg[end + 1:].strip()
                    inner = os.path.expanduser(os.path.expandvars(inner))
                    if os.path.isfile(inner):
                        return inner, rest
                break  # only try the matching quote style

        # 3) Bare leading token + trailing caption.
        first, _, rest = arg.partition(" ")
        first = os.path.expanduser(os.path.expandvars(first))
        if os.path.isfile(first):
            return first, rest.strip()

        return None, ""

    def _share_kind_for_path(self, path: str) -> str:
        """Pick an InlineMediaWidget `type` for a given file path."""
        ext = os.path.splitext(path)[1].lower()
        if ext in self._SHARE_EXT_KIND:
            return self._SHARE_EXT_KIND[ext]
        # Fall back to MIME sniffing for extensions we didn't list.
        import mimetypes
        mt, _ = mimetypes.guess_type(path)
        if mt:
            if mt.startswith("image/gif"):
                return "gif"
            if mt.startswith("image/"):
                return "image"
            if mt.startswith("audio/"):
                return "audio"
            if mt.startswith("video/"):
                return "video"
            if mt == "application/pdf":
                return "pdf"
            if mt.startswith("text/html"):
                return "html"
            if mt.startswith("text/"):
                # Plain text we send as a small "info" payload so it shows
                # cleanly. (The post body field would also work, but using
                # the media slot keeps the post-card layout uniform.)
                return "info"
        # Unknown — render as a download-style info widget that just shows
        # the filename. Better than crashing or guessing wrong.
        return "info"

    def _share_build_local_payload(self, path: str, kind: str) -> Optional[dict]:
        """
        Build the InlineMediaWidget payload used for the *local* render.

        For binary kinds (image/gif/audio/video/pdf/3d) we just point at
        the file path — the renderers handle decoding lazily. No size
        cap, no in-memory inflation: a 24 MiB audio file stays on disk
        and Qt streams from there.

        For text kinds (python/html/info) we still inline the contents,
        because the renderers need a string. Capped at a few MiB to keep
        the local card responsive — text files that big should be info-
        rendered with a tail-truncated body anyway.
        """
        try:
            size = os.path.getsize(path)
        except OSError as e:
            self.append_text(f"share: {path}: {e}\n", self.C_ERROR)
            return None

        ext = os.path.splitext(path)[1].lower().lstrip(".")
        basename = os.path.basename(path)

        if kind in ("python", "html"):
            # Hard cap text inlining at 1 MiB so a giant code file
            # doesn't freeze the card.
            if size > 1024 * 1024:
                return {"type": "info",
                        "text": f"{basename} ({size:,} bytes — too large to render)"}
            try:
                with open(path, "rb") as f:
                    blob = f.read()
                text = blob.decode("utf-8")
            except (OSError, UnicodeDecodeError) as e:
                self.append_text(
                    f"share: {basename}: {e}\n", self.C_ERROR,
                )
                return None
            if kind == "python":
                return {"type": "python", "code": text,
                        "filename": basename}
            return {"type": "html", "content": text,
                    "filename": basename}

        if kind == "info":
            # Generic / text-ish: show contents up to a small limit.
            try:
                with open(path, "rb") as f:
                    blob = f.read(8192)
                text = blob.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                return {"type": "info",
                        "text": f"{basename} ({size:,} bytes, binary)"}
            if len(text) > 4000:
                text = text[:4000] + "\n…(truncated)"
            return {"type": "info",
                    "text": f"{basename}\n\n{text}"}

        # Binary media — point at the path. Qt will load on demand.
        return {
            "type": kind,
            "path": path,
            "format": ext or kind,
            "filename": basename,
            "bytes": size,
        }

    # ---- Echo dedupe -----------------------------------------------
    #
    # The daemon broadcasts every published post back to all subscribers
    # — including us. To avoid double-rendering our own posts (once via
    # the local-render call in _share_render_locally, once via the feed
    # tailer), we register the hash of every post we publish in a
    # short-lived set and skip feed lines whose id matches.
    #
    # This is dramatically simpler than the previous prefix-match scheme:
    # because publish_draft hands us back the post's content hash, and
    # the feed line carries the same hash in `id`, we just compare
    # strings. No JSON parsing, no truncation worries, no carrier-blob
    # exclusions. (Carrier blobs no longer exist — the daemon's
    # publish_draft handles attachments by hash directly without
    # broadcasting a separate post for each one.)
    _SHARE_DEDUPE_WINDOW_S = 60.0

    def _share_remember_publish(self, post_hash: str) -> None:
        """Register a freshly-published post hash for echo suppression."""
        if not post_hash:
            return
        if not hasattr(self, "_share_recent"):
            self._share_recent = {}
        now = time.time()
        cutoff = now - self._SHARE_DEDUPE_WINDOW_S
        # Lazy GC of expired entries.
        for k in list(self._share_recent.keys()):
            if self._share_recent[k] < cutoff:
                del self._share_recent[k]
        self._share_recent[post_hash] = now

    def _share_is_recent_publish(self, outer: dict) -> bool:
        """Should this feed-arrived envelope be suppressed before rendering?"""
        if not hasattr(self, "_share_recent") or not self._share_recent:
            return False
        post_id = outer.get("id")
        if not isinstance(post_id, str):
            return False
        cutoff = time.time() - self._SHARE_DEDUPE_WINDOW_S
        ts = self._share_recent.get(post_id)
        return ts is not None and ts >= cutoff

    def _share_publish_text(self, text: str) -> None:
        """
        Publish a plain-text post.

        Allocates a draft, writes the body, commits, THEN renders locally
        with the daemon-returned hash stamped into the envelope. We
        deliberately commit before rendering: without the hash, the local
        card's `_post_id` is empty, `_register_thread_card` silently
        skips indexing (the `if post_id:` guard treats "" as falsy), and
        peer replies tagged `reply_to=<canonical_hash>` can never find
        the parent card in `_post_card_index` — they fall through to
        `insert_media` and render as freestanding top-level posts.

        The cost of waiting for commit is one local IPC round-trip (the
        draft ctl write blocks until the post is signed and broadcast),
        which is sub-millisecond. The visible-latency saving from
        rendering early was never worth losing reply threading.
        """
        if not text.strip():
            self.append_text("share: empty post\n", self.C_ERROR)
            return
        if len(text.encode("utf-8")) > 4096:
            self.append_text(
                "share: post too large (max 4 KiB of text — "
                "use a file for longer content)\n",
                self.C_ERROR,
            )
            return

        draft_path = self._share_open_draft()
        if draft_path is None:
            return

        if not self._share_write_attr(draft_path, "body", text):
            self._share_discard_draft(draft_path)
            return

        post_hash = self._share_commit_draft(draft_path)
        if post_hash is None:
            return
        self._share_remember_publish(post_hash)

        # Render locally with the canonical id so the card registers in
        # _post_card_index and subsequent peer replies thread under it.
        self._share_render_locally({
            "type": "post",
            "body": text,
            "id": post_hash,
        })

        preview = text if len(text) <= 60 else text[:57] + "…"
        self.append_text(f"✓ shared: {preview}\n", self.C_SUCCESS)

    def _share_publish_file(self, path: str, caption: str = "") -> None:
        """
        Publish a file as a post.

        The clone-style draft makes this trivial: queue the file path
        on `attach`, optionally set a caption on `body`, commit. The
        daemon reads the bytes, content-addresses them in the gossip
        store, and broadcasts a small envelope post that references
        the attachment by hash. Peers fetch the bytes on demand via
        nodes/<author>/social/<hash>.

        We commit BEFORE the local render so the daemon-returned hash
        can be stamped into the envelope. See _share_publish_text for
        the rationale: without the id, the local card isn't registered
        in _post_card_index and peer replies can never thread under it.
        The local-render envelope still carries the absolute file path
        so the publisher sees full media without waiting for the
        broadcast echo to round-trip.
        """
        kind = self._share_kind_for_path(path)
        local_payload = self._share_build_local_payload(path, kind)
        if local_payload is None:
            return  # error already surfaced

        # ---- Wire publish via clone draft ----
        draft_path = self._share_open_draft()
        if draft_path is None:
            return

        if caption and not self._share_write_attr(draft_path, "body", caption):
            self._share_discard_draft(draft_path)
            return

        if not self._share_write_attr(draft_path, "attach", path + "\n"):
            self._share_discard_draft(draft_path)
            return

        post_hash = self._share_commit_draft(draft_path)
        if post_hash is None:
            return
        self._share_remember_publish(post_hash)

        # ---- Local render (with canonical id) ----
        local_envelope = {
            "type": "post",
            "body": caption or "",
            "ts": time.time(),
            "media": [local_payload],
            "id": post_hash,
        }
        self._share_render_locally(local_envelope)

        basename = os.path.basename(path)
        extra = f" — {caption}" if caption else ""
        self.append_text(
            f"✓ shared: {basename} ({kind}){extra}\n",
            self.C_SUCCESS,
        )

    def _share_open_composer(self) -> None:
        """Pop a small composer dialog: multiline text + optional file pick."""
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout,
            QHBoxLayout, QFileDialog, QPushButton, QLabel,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Share to peribus")
        dlg.resize(480, 320)
        v = QVBoxLayout(dlg)

        v.addWidget(QLabel("What do you want to share?"))
        text_edit = QPlainTextEdit()
        text_edit.setPlaceholderText("Post body (optional if you attach a file)")
        v.addWidget(text_edit)

        # Attached-file row
        file_row = QHBoxLayout()
        file_label = QLabel("(no file)")
        file_label.setStyleSheet("color: gray; font-style: italic;")
        attach_btn = QPushButton("Attach file…")
        clear_btn = QPushButton("✕")
        clear_btn.setVisible(False)
        clear_btn.setFixedWidth(28)
        file_row.addWidget(attach_btn)
        file_row.addWidget(file_label, 1)
        file_row.addWidget(clear_btn)
        v.addLayout(file_row)

        # Mutable holder so the inner closures can write to a single slot.
        chosen = {"path": None}

        def _pick():
            path, _ = QFileDialog.getOpenFileName(
                dlg, "Attach file", os.path.expanduser("~"),
                "All files (*)",
            )
            if path:
                chosen["path"] = path
                file_label.setText(os.path.basename(path))
                file_label.setStyleSheet("")
                clear_btn.setVisible(True)

        def _clear():
            chosen["path"] = None
            file_label.setText("(no file)")
            file_label.setStyleSheet("color: gray; font-style: italic;")
            clear_btn.setVisible(False)

        attach_btn.clicked.connect(_pick)
        clear_btn.clicked.connect(_clear)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Ok).setText("Share")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        body = text_edit.toPlainText().strip()
        path = chosen["path"]
        if path:
            self._share_publish_file(path, caption=body)
        elif body:
            self._share_publish_text(body)
        else:
            self.append_text("share: nothing to share\n", self.C_INFO)

    # ---- Drag-and-drop wiring -------------------------------------------
    #
    # Drop a file (or several) onto the terminal and we publish each as a
    # /share. We accept both file URLs and plain text drops (some apps
    # only offer the latter). Multi-file drops generate one post per file
    # rather than a compound post — keeps the UX predictable.

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md is not None and (md.hasUrls() or md.hasText()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        # Same logic as dragEnter — needed for some platforms to keep
        # the cursor styled as a valid drop target throughout the drag.
        md = event.mimeData()
        if md is not None and (md.hasUrls() or md.hasText()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()
        if md is None:
            return super().dropEvent(event)
        paths = []
        if md.hasUrls():
            for url in md.urls():
                if url.isLocalFile():
                    paths.append(url.toLocalFile())
        elif md.hasText():
            # Some sources offer only a text path — try it.
            t = md.text().strip()
            if t and os.path.isfile(t):
                paths.append(t)
        if not paths:
            return super().dropEvent(event)
        event.acceptProposedAction()
        for p in paths:
            self._share_publish_file(p)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _show_help(self):
        h = """\
+----------------------------------------------+
|                  Terminal                    |
+----------------------------------------------+

Agent creation:
  /new <n> [system]            Create agent & connect  (e.g. /new claude)
  /new <n> <provider> [model]  Create with provider    (e.g. /new fast groq kimi-k2)
  /connect <n>                 Connect to existing agent
  /disconnect                  Disconnect

Provider switching (on connected agent):
  /provider <n> [model]      Switch provider+model
  /use <provider> <hint>     Fuzzy-match model       (e.g. /use groq kimi)
  /use <alias>               Quick alias             (e.g. /use kimi, /use zai)
  /use                       Show aliases

Composite agents:
  /master [prov] [model]  Spawn master agent (auto-exec bash, coordinates)
  /coder [prov] [model]   Spawn coder agent (workspace-aware)
  /tcoder [prov] [model]  Like /coder, but routes $coder/<MACHINE> output
                          to /n/<machine>/terms/<term_id>/inline — this
                          terminal for our machine, first term found
                          for any other machine.

Grok AV (voice agent):
  /av [voice]            Start Grok voice agent with function tools
    Code from voice → $av/CODE → $RIO/scene/parse
  /av_gemini [voice]     Start Gemini voice agent with function tools
    Code from voice → $av_gemini/CODE → $RIO/scene/parse

Agent control:
  /system /model /temperature /clear /cancel /retry

Agent info:
  /history /config /errors

Routing:
  /attach <src> <dst>   Route source -> destination
  /detach <src>         Stop routing
  /attachments          List active attachments
  /context <n>          Route $RIO/CONTEXT -> $agent/history

Global:
  /list (/ls) /delete <n> /status /cls /help
  /color (/colors)       Open color scheme picker
  /dark (/darkmode)      Toggle dark/light mode
  /versions (/ver)       Toggle version panel
  /acme                  Open ACME editor
  /operator              Open Operator graph panel
  /scene                 Toggle terminal scene panel (write to term/parse)
  /pop                   Detach terminal to floating window
  /dock                  Re-dock terminal into scene
  /restart               Restart shell (fresh env, re-seed vars)
  /setup                 Unmount & remount 9pfuse (LLMFS + Rio)
  /mount <IP!P> <n>      Mount 9P at /n/name via 9pfuse
  /signal on|off         Full-mesh (un)subscribe across every
                         machine in /n/ctl via /scene/signals/ctl
  /peribus               Connect to peribus mycelium (feed → $inline)
  /peribus post <text>   Publish a short post to your feed
  /peribus stop|status   Stop tailer / show state
  /share                 Open composer dialog (text + optional file)
  /share <text>          Publish a text post
  /share <path>          Publish a file as a media post (also: drag & drop)
  /share <path> <text>   File + caption

Prefixes:
  >>> <code>    Python     $ <cmd>    Shell
  $             Toggle shell mode
  <text>        Prompt to connected agent

Keys:
  Delete        Interrupt running shell command (SIGINT)
"""
        self.append_text(h, self.C_MACRO)

    # ------------------------------------------------------------------
    # Color scheme management
    # ------------------------------------------------------------------

    def _apply_color_scheme(self, scheme_name: str, animate_shadow: bool = True):
        """Apply a named color scheme globally."""
        if scheme_name not in self.COLOR_SCHEMES:
            self.append_text(f"Unknown color scheme: {scheme_name}\n", self.C_ERROR)
            return

        self._active_scheme_name = scheme_name
        self._active_scheme = dict(self.COLOR_SCHEMES[scheme_name])

        # Update class-level convenience colors so append_text callers
        # that pass e.g. self.C_SHELL directly also pick up the new scheme.
        # shell_echo/output are always black/white — use mode-aware property.
        self.C_SHELL   = self._active_shell_echo_color
        self.C_AGENT   = self._dm_adjust_color(self._active_scheme["agent"])
        self.C_SUCCESS = self._dm_adjust_color(self._active_scheme["success"])
        self.C_ERROR   = self._dm_adjust_color(self._active_scheme["error"])
        self.C_INFO    = self._dm_adjust_color(self._active_scheme["info"])

        # Animate shadow to the new scheme's shadow color
        if animate_shadow:
            self._set_shadow_to_scheme()

        self.append_text(f"Color scheme: {scheme_name}\n", self._active_shell_echo_color)

    def _set_shadow_to_scheme(self):
        """
        Immediately set shadow to match the active color scheme.
        Uses QVariantAnimation for a smooth, hardware-independent transition.
        """
        shadow_target = self._proxy if self._proxy is not None else self
        current_effect = shadow_target.graphicsEffect()

        # If there's no shadow effect to animate, just exit
        if not isinstance(current_effect, QGraphicsDropShadowEffect):
            return

        shadow = current_effect
        target_color = self._parse_rgba(self._active_shadow_color)
        start_color = shadow.color()

        # 1. Clean up any existing shadow color animation to prevent race conditions
        if hasattr(self, '_shadow_scheme_anim') and self._shadow_scheme_anim:
            self._shadow_scheme_anim.stop()
            self._shadow_scheme_anim.deleteLater()
            self._shadow_scheme_anim = None

        # 2. Initialize the QVariantAnimation
        # 30 steps * 16ms is roughly 480ms
        anim = QVariantAnimation(self)
        anim.setDuration(480) 
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutQuad)

        def update_shadow_color(t):
            # Manually interpolate the RGBA channels
            r = int(start_color.red() + (target_color.red() - start_color.red()) * t)
            g = int(start_color.green() + (target_color.green() - start_color.green()) * t)
            b = int(start_color.blue() + (target_color.blue() - start_color.blue()) * t)
            a = int(start_color.alpha() + (target_color.alpha() - start_color.alpha()) * t)
            
            shadow.setColor(QColor(r, g, b, a))

        anim.valueChanged.connect(update_shadow_color)

        # 3. Store reference and start
        # Storing the reference is key to preventing the "Use-After-Free" crash
        self._shadow_scheme_anim = anim
        anim.start()

    def _open_color_picker(self):
        """Open the color scheme picker as an inline terminal widget.

        Renders a compact frame in the scrollback (not a modal dialog):
          • a strip of preset chips, with the active one ringed
          • an "Advanced" disclosure that expands per-channel swatches
            and the two ANSI rows
          • a close (×) corner button

        Adopts dark/light mode and uses the active scheme's accent color
        for highlights so it visually belongs to the running terminal.
        """
        from PySide6.QtWidgets import (
            QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QGridLayout, QColorDialog, QWidget, QSizePolicy
        )
        from PySide6.QtGui import QPainter, QBrush, QPen
        from PySide6.QtCore import QSize

        terminal = self
        dark = bool(getattr(self, '_is_dark_mode', False))

        # ---- palette derived from dark mode ----
        if dark:
            bg_rgba       = "rgba(40, 42, 52, 160)"
            border_rgba   = "rgba(90, 100, 120, 110)"
            text_rgba     = "rgba(220, 224, 232, 255)"
            sub_text_rgba = "rgba(150, 156, 168, 255)"
            chip_bg       = "rgba(55, 58, 70, 200)"
            chip_border   = "rgba(110, 118, 134, 140)"
            chip_hover    = "rgba(70, 74, 88, 220)"
            close_hover   = "rgba(200, 80, 80, 200)"
        else:
            bg_rgba       = "rgba(248, 250, 253, 200)"
            border_rgba   = "rgba(190, 200, 215, 150)"
            text_rgba     = "rgba(40, 46, 60, 255)"
            sub_text_rgba = "rgba(120, 128, 140, 255)"
            chip_bg       = "rgba(255, 255, 255, 220)"
            chip_border   = "rgba(195, 202, 215, 180)"
            chip_hover    = "rgba(238, 242, 248, 240)"
            close_hover   = "rgba(220, 90, 90, 220)"

        class ColorSwatch(QWidget):
            """Clickable color swatch (compact)."""

            def __init__(self, color_str, label, on_changed=None,
                         size=(40, 22), parent=None):
                super().__init__(parent)
                self._color = terminal._parse_rgba(color_str)
                self._label = label
                self._on_changed = on_changed
                self.setFixedSize(size[0], size[1])
                self.setCursor(Qt.PointingHandCursor)
                self.setToolTip(f"{label} — click to edit")

            def paintEvent(self, event):
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                # If color has alpha < 255, hint at it with a tiny diagonal
                # band rather than a noisy checkerboard.
                if self._color.alpha() < 255:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(QColor(180, 180, 180, 90)))
                    p.drawRoundedRect(1, 1, self.width() - 2,
                                      self.height() - 2, 4, 4)
                p.setBrush(QBrush(self._color))
                pen_col = QColor(0, 0, 0, 80) if not dark else QColor(255, 255, 255, 70)
                p.setPen(QPen(pen_col, 1))
                p.drawRoundedRect(1, 1, self.width() - 2,
                                  self.height() - 2, 4, 4)
                p.end()

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    new_color = QColorDialog.getColor(
                        self._color, self,
                        f"Pick {self._label} color",
                        QColorDialog.ShowAlphaChannel
                    )
                    if new_color.isValid():
                        self._color = new_color
                        self.update()
                        if self._on_changed:
                            self._on_changed()

            def color_rgba(self):
                c = self._color
                return f"rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})"

            def color_hex(self):
                return self._color.name()

            def set_color(self, color_str):
                self._color = terminal._parse_rgba(color_str)
                self.update()

        class PresetChip(QPushButton):
            """A preset button that shows a small dot of the scheme accent
            and ring-highlights itself when active."""

            def __init__(self, name, scheme, parent=None):
                super().__init__(name, parent)
                self._name = name
                self._scheme = scheme
                self._active = False
                self.setCursor(Qt.PointingHandCursor)
                self.setFlat(True)
                self.setMinimumHeight(26)
                self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                self._apply_style()

            def set_active(self, active: bool):
                if self._active != active:
                    self._active = active
                    self._apply_style()

            def _apply_style(self):
                accent = terminal._parse_rgba(self._scheme["shell_echo"])
                # Dim accents that are nearly the same as text get bumped
                # toward the scheme's "agent" color so the chip stays
                # readable on both modes.
                if accent.alpha() < 30 or (
                    dark and accent.lightness() < 50
                ) or (not dark and accent.lightness() > 220):
                    accent = terminal._parse_rgba(self._scheme["agent"])
                a_r, a_g, a_b = accent.red(), accent.green(), accent.blue()

                if self._active:
                    border = f"2px solid rgba({a_r}, {a_g}, {a_b}, 220)"
                    bg = chip_hover
                else:
                    border = f"1px solid {chip_border}"
                    bg = chip_bg

                self.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {bg};
                        border: {border};
                        border-radius: 13px;
                        padding: 3px 12px 3px 22px;
                        font-family: Consolas, 'DejaVu Sans Mono', monospace;
                        font-size: 11px;
                        font-weight: 500;
                        color: {text_rgba};
                        text-align: left;
                    }}
                    QPushButton:hover {{
                        background-color: {chip_hover};
                    }}
                """)

            def paintEvent(self, event):
                # Draw the base button, then overlay the accent dot.
                super().paintEvent(event)
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                accent = terminal._parse_rgba(self._scheme["shell_echo"])
                if accent.alpha() < 30 or (
                    dark and accent.lightness() < 50
                ) or (not dark and accent.lightness() > 220):
                    accent = terminal._parse_rgba(self._scheme["agent"])
                p.setBrush(QBrush(accent))
                p.setPen(QPen(QColor(0, 0, 0, 60), 0.5))
                # 10px dot, vertically centered, padded from the left edge
                dot_d = 10
                x = 8
                y = (self.height() - dot_d) // 2
                p.drawEllipse(x, y, dot_d, dot_d)
                p.end()

        class InlineColorPicker(QFrame):
            """The whole inline color-picker widget."""

            def __init__(self, parent_terminal):
                super().__init__()
                self.terminal = parent_terminal
                self._advanced_open = False
                self._setup()

            def _setup(self):
                self.setStyleSheet(f"""
                    InlineColorPicker {{
                        background-color: {bg_rgba};
                        border: 1px solid {border_rgba};
                        border-radius: 6px;
                        margin: 4px 0px;
                    }}
                    QLabel {{
                        color: {text_rgba};
                        font-family: Consolas, 'DejaVu Sans Mono', monospace;
                    }}
                """)
                # Match the inline width budget used by other terminal widgets.
                try:
                    self.setMaximumWidth(self.terminal._inline_max_width())
                except Exception:
                    pass

                root = QVBoxLayout(self)
                root.setContentsMargins(12, 8, 8, 10)
                root.setSpacing(8)

                # ---- top row: title + preset chips + close ----
                top = QHBoxLayout()
                top.setSpacing(8)

                title = QLabel("colors")
                title.setStyleSheet(
                    f"color: {sub_text_rgba}; font-size: 11px;"
                    f"font-family: Consolas, monospace;"
                )
                top.addWidget(title)

                self._chips = {}
                schemes = list(self.terminal.COLOR_SCHEMES.keys())
                for name in schemes:
                    scheme = self.terminal.COLOR_SCHEMES[name]
                    chip = PresetChip(name, scheme)
                    chip.clicked.connect(
                        lambda checked=False, n=name: self._select_preset(n)
                    )
                    chip.set_active(
                        name == self.terminal._active_scheme_name
                    )
                    self._chips[name] = chip
                    top.addWidget(chip)

                top.addStretch()

                # Close button (small, subtle until hover)
                close_btn = QPushButton("×")
                close_btn.setCursor(Qt.PointingHandCursor)
                close_btn.setFixedSize(22, 22)
                close_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent;
                        border: none;
                        color: {sub_text_rgba};
                        font-size: 16px;
                        font-weight: bold;
                        border-radius: 11px;
                        padding-bottom: 2px;
                    }}
                    QPushButton:hover {{
                        background-color: {close_hover};
                        color: white;
                    }}
                """)
                close_btn.clicked.connect(self._close)
                top.addWidget(close_btn)

                root.addLayout(top)

                # ---- advanced toggle row ----
                toggle_row = QHBoxLayout()
                toggle_row.setSpacing(6)
                self._toggle_btn = QPushButton("▸ advanced")
                self._toggle_btn.setCursor(Qt.PointingHandCursor)
                self._toggle_btn.setFlat(True)
                self._toggle_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent;
                        border: none;
                        color: {sub_text_rgba};
                        font-family: Consolas, monospace;
                        font-size: 11px;
                        padding: 2px 4px;
                        text-align: left;
                    }}
                    QPushButton:hover {{
                        color: {text_rgba};
                    }}
                """)
                self._toggle_btn.clicked.connect(self._toggle_advanced)
                toggle_row.addWidget(self._toggle_btn)
                toggle_row.addStretch()
                root.addLayout(toggle_row)

                # ---- advanced panel (hidden by default) ----
                self._advanced = QWidget()
                adv_layout = QVBoxLayout(self._advanced)
                adv_layout.setContentsMargins(2, 2, 2, 2)
                adv_layout.setSpacing(8)

                # Per-channel swatches in a compact grid (2 columns of pairs)
                self._swatches = {}
                swatch_defs = [
                    ("shell_echo",   "shell echo"),
                    ("shell_output", "shell output"),
                    ("agent",        "agent output"),
                    ("success",      "success"),
                    ("error",        "error"),
                    ("info",         "info"),
                    ("shadow",       "shadow"),
                ]
                grid = QGridLayout()
                grid.setHorizontalSpacing(14)
                grid.setVerticalSpacing(4)
                active = self.terminal._active_scheme
                for i, (key, label) in enumerate(swatch_defs):
                    row, col = divmod(i, 2)
                    cell = QHBoxLayout()
                    cell.setSpacing(8)
                    lbl = QLabel(label)
                    lbl.setStyleSheet(
                        f"color: {text_rgba}; font-size: 11px;"
                        f"font-family: Consolas, monospace;"
                    )
                    lbl.setMinimumWidth(90)
                    sw = ColorSwatch(
                        active.get(key, "rgba(0,0,0,255)"), label,
                        on_changed=lambda k=key: self._on_swatch_changed(k)
                    )
                    self._swatches[key] = sw
                    cell.addWidget(lbl)
                    cell.addWidget(sw)
                    cell.addStretch()
                    holder = QWidget()
                    holder.setLayout(cell)
                    grid.addWidget(holder, row, col)
                adv_layout.addLayout(grid)

                # ANSI swatches: 16 small chips on two rows, no labels
                # (tooltips carry the name; labels were the cramped part
                # of the old design).
                ansi_label = QLabel("ansi")
                ansi_label.setStyleSheet(
                    f"color: {sub_text_rgba}; font-size: 10px;"
                    f"font-family: Consolas, monospace; margin-top: 4px;"
                )
                adv_layout.addWidget(ansi_label)

                ansi_map = active.get("ansi_map", self.terminal._ANSI_COLOR_MAP)
                ansi_labels = {
                    '30': 'black',   '31': 'red',     '32': 'green',   '33': 'yellow',
                    '34': 'blue',    '35': 'magenta', '36': 'cyan',    '37': 'white',
                    '90': 'black+',  '91': 'red+',    '92': 'green+',  '93': 'yellow+',
                    '94': 'blue+',   '95': 'magenta+','96': 'cyan+',   '97': 'white+',
                }
                self._ansi_swatches = {}
                for row_codes in (
                    ['30', '31', '32', '33', '34', '35', '36', '37'],
                    ['90', '91', '92', '93', '94', '95', '96', '97'],
                ):
                    row_layout = QHBoxLayout()
                    row_layout.setSpacing(3)
                    for code in row_codes:
                        sw = ColorSwatch(
                            ansi_map.get(code, '#000000'),
                            ansi_labels[code],
                            on_changed=lambda c=code: self._on_ansi_changed(c),
                            size=(28, 18),
                        )
                        self._ansi_swatches[code] = sw
                        row_layout.addWidget(sw)
                    row_layout.addStretch()
                    adv_layout.addLayout(row_layout)

                self._advanced.setVisible(False)
                root.addWidget(self._advanced)

            # -- behaviour --

            def _toggle_advanced(self):
                self._advanced_open = not self._advanced_open
                self._advanced.setVisible(self._advanced_open)
                arrow = "▾" if self._advanced_open else "▸"
                self._toggle_btn.setText(f"{arrow} advanced")
                self.adjustSize()

            def _select_preset(self, name):
                # Update chip ring states first (visual feedback is instant).
                for n, chip in self._chips.items():
                    chip.set_active(n == name)
                # Sync advanced swatches if the panel is built.
                scheme = self.terminal.COLOR_SCHEMES[name]
                for key, swatch in self._swatches.items():
                    swatch.set_color(scheme.get(key, "rgba(0,0,0,255)"))
                ansi_map = scheme.get("ansi_map", {})
                for code, swatch in self._ansi_swatches.items():
                    swatch.set_color(ansi_map.get(code, '#000000'))
                # Apply.
                self.terminal._apply_color_scheme(name)

            def _on_swatch_changed(self, key):
                swatch = self._swatches[key]
                self.terminal._active_scheme[key] = swatch.color_rgba()
                self.terminal._active_scheme_name = "Custom"
                # Customizing breaks the preset ring.
                for chip in self._chips.values():
                    chip.set_active(False)
                t = self.terminal
                t.C_SHELL   = t._active_shell_echo_color
                t.C_AGENT   = t._dm_adjust_color(t._active_scheme["agent"])
                t.C_SUCCESS = t._dm_adjust_color(t._active_scheme["success"])
                t.C_ERROR   = t._dm_adjust_color(t._active_scheme["error"])
                t.C_INFO    = t._dm_adjust_color(t._active_scheme["info"])
                if key == "shadow":
                    t._set_shadow_to_scheme()

            def _on_ansi_changed(self, code):
                swatch = self._ansi_swatches[code]
                if "ansi_map" not in self.terminal._active_scheme:
                    self.terminal._active_scheme["ansi_map"] = dict(
                        self.terminal._ANSI_COLOR_MAP
                    )
                self.terminal._active_scheme["ansi_map"][code] = swatch.color_hex()
                self.terminal._active_scheme_name = "Custom"
                for chip in self._chips.values():
                    chip.set_active(False)

            def _close(self):
                # Make sure shadow is in sync with whatever's currently selected.
                self.terminal._set_shadow_to_scheme()
                self.deleteLater()

        # Freeze the current default text display so the picker lands
        # below all existing scrollback, then append it and advance the
        # default display so subsequent output goes underneath.
        ctd = self.current_text_display
        if ctd is not None:
            self._adjust_height(ctd)
        picker = InlineColorPicker(self)
        self.terminal_content_layout.addWidget(picker)
        self._advance_default_text_display()

    # ------------------------------------------------------------------
    # Command history navigation
    # ------------------------------------------------------------------

    def _history_prev(self):
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.command_input.setPlainText(self.command_history[self.history_index])
            c = self.command_input.textCursor()
            c.movePosition(QTextCursor.End)
            self.command_input.setTextCursor(c)

    def _history_next(self):
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.command_input.setPlainText(self.command_history[self.history_index])
            c = self.command_input.textCursor()
            c.movePosition(QTextCursor.End)
            self.command_input.setTextCursor(c)
        elif self.history_index == len(self.command_history) - 1:
            self.history_index = len(self.command_history)
            self.command_input.clear()

    # ------------------------------------------------------------------
    # Text output helpers
    # ------------------------------------------------------------------

    def _dm_adjust_color(self, color_str: str) -> str:
        """Adjust a color string for dark/light mode visibility.

        In dark mode, any color that would be too dark (luminance < threshold)
        gets lightened. In light mode, any color that would be too bright gets
        darkened. Colors with good contrast are left untouched.

        Memoized. This is hot: append_text calls it on every chunk, and
        _insert_ansi_text calls it on every SGR code, so an LLM streaming
        2000 tokens through this path can drive the function tens of
        thousands of times — but with only a handful of distinct color
        strings (the scheme + a few ANSI-derived variants). Caching by
        (color_str, is_dark_mode) collapses all that work to one lookup
        per call after the first encounter. Cache is dropped on theme
        toggle (see toggle_dark_mode wiring elsewhere) so stale entries
        can't survive a mode switch.
        """
        is_dark = bool(getattr(self, '_is_dark_mode', False))
        if not hasattr(self, '_dm_color_cache'):
            self._dm_color_cache = {}
        cache_key = (color_str, is_dark)
        cached = self._dm_color_cache.get(cache_key)
        if cached is not None:
            return cached

        c = self._parse_rgba(color_str)
        r, g, b, a = c.red(), c.green(), c.blue(), c.alpha()
        lum = r * 0.299 + g * 0.587 + b * 0.114

        if is_dark:
            if lum < 120:
                # Too dark for dark background — lighten
                factor = max(0.0, min(1.0, (120 - lum) / 120.0))
                boost = factor * 0.7
                nr = min(255, int(r + (255 - r) * boost))
                ng = min(255, int(g + (255 - g) * boost))
                nb = min(255, int(b + (255 - b) * boost))
                result = f"rgba({nr}, {ng}, {nb}, {a})"
                self._dm_color_cache[cache_key] = result
                return result
        else:
            if lum > 200:
                # Too bright for light background — darken
                factor = max(0.0, min(1.0, (lum - 200) / 55.0))
                dampen = factor * 0.6
                nr = max(0, int(r - r * dampen))
                ng = max(0, int(g - g * dampen))
                nb = max(0, int(b - b * dampen))
                result = f"rgba({nr}, {ng}, {nb}, {a})"
                self._dm_color_cache[cache_key] = result
                return result

        self._dm_color_cache[cache_key] = color_str
        return color_str

    def _echo(self, text: str, color: str):
        """Echo a command the user typed."""
        self.append_text(text + "\n", color)

    def _stream_text(self, text: str, color: str = None, interval_ms: int = 32, callback=None):
        """Stream text character-by-character with a typewriter effect.
        
        Args:
            text: The full string to stream.
            color: Color for the text (uses C_DEFAULT if None).
            interval_ms: Milliseconds between each character.
            callback: Optional callable invoked after the last character.
        """
        color = color or self.C_DEFAULT
        idx = 0

        def _tick():
            nonlocal idx
            if idx < len(text):
                self.append_text(text[idx], color)
                idx += 1
            else:
                timer.stop()
                timer.deleteLater()
                if callback:
                    callback()

        timer = QTimer(self)
        timer.timeout.connect(_tick)
        timer.start(interval_ms)

    def append_text(self, text: str, color: str = None):
        color = color or self.C_DEFAULT
        color = self._dm_adjust_color(color)
        cursor = self.current_text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(self._parse_rgba(color))
        cursor.insertText(text, fmt)
        self.current_text_display.setTextCursor(cursor)

        # Coalesced — append_text has 291 callsites in this file, and a
        # routine like _stream_text typewriter or a fast-emitting agent
        # drives it many times per second. Per-call singleShot(0,
        # _scroll_to_bottom) flooded the event loop. Now bursts collapse
        # via the shared terminal-level timer.
        self._request_scroll_coalesced()

    def append_output(self, text: str, color: str = None):
        """Alias for compatibility with LLMFSExtension and rio_main."""
        self.append_text(text, color or self.C_DEFAULT)

    def append_error(self, text: str):  # DEAD CODE — unused, kept for external callers
        self.append_text(text, self.C_ERROR)

    def clear_output(self):
        while self.terminal_content_layout.count():
            child = self.terminal_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.text_display = self._create_text_display()
        self.terminal_content_layout.addWidget(self.text_display)
        self.text_displays = [self.text_display]
        self.current_text_display = self.text_display
        # Reset stream router state so the next agent/FS chunk doesn't
        # try to extend a deleted text display or open code widget.
        if hasattr(self, 'stream_router') and self.stream_router is not None:
            self.stream_router.reset_all()

    # ------------------------------------------------------------------
    # Helpers used by the stream router / inline widgets
    # ------------------------------------------------------------------
    
    def _inline_widget_mount(self) -> str:
        """Base mount path used by inline code widgets to compute their
        Run target (target = $mount/<machine>/scene/parse).
        
        We use the parent of llmfs_mount when possible — typically /n —
        because individual machines are mounted as siblings of the LLMFS
        (e.g. /n/llm, /n/david, /n/rioa). Falls back to /n.
        """
        try:
            parent = os.path.dirname(self.llmfs_mount.rstrip('/'))
            if parent and os.path.isdir(parent):
                return parent
        except Exception:
            pass
        return "/n"
    
    def _advance_default_text_display(self):
        """Create a fresh default current_text_display and append it to
        the layout. Used by the stream router after inserting a media
        widget, so subsequent direct append_text() calls don't try to
        write into a frozen display above the new widget.
        """
        te = self._create_text_display()
        self.terminal_content_layout.addWidget(te)
        self.text_displays.append(te)
        self.current_text_display = te
        return te

    def _shell_cwd(self) -> Optional[str]:
        """Return the current working directory of the terminal's bash
        process, or None if it can't be determined.
        
        Reads /proc/<pid>/cwd, which is a kernel symlink that always
        points at the live cwd of the process — automatically updated
        when the user runs `cd`. This lets the inline filesystem
        resolve relative paths the way the user typed them in the shell.
        
        Thread-safe (just an os.readlink); cheap (single syscall).
        Returns None on platforms without /proc, or if the shell isn't
        running, or on permission errors.
        """
        sp = getattr(self, 'shell_process', None)
        if sp is None or sp.poll() is not None:
            return None
        try:
            pid = sp.pid
            return os.readlink(f"/proc/{pid}/cwd")
        except (OSError, FileNotFoundError, PermissionError):
            return None

    def _inline_max_width(self) -> int:
        """Maximum width an inline widget should occupy, in pixels.
        
        Computed from the terminal's scroll-area viewport so that
        widgets never paint under the vertical scroll bar and never
        force horizontal scrolling. Falls back to a conservative
        default if the scroll area hasn't been laid out yet.
        """
        try:
            vp = self.terminal_scroll.viewport()
            if vp is not None:
                w = vp.width()
                if w > 100:  # protect against pre-show 0/tiny values
                    # Reserve a couple of pixels for breathing room.
                    return max(200, w - 4)
        except Exception:
            pass
        return 600

    def _propagate_inline_width(self):
        """Push the current inline-max-width to every live inline widget.
        
        Called on terminal resize so widgets reflow within the new bounds.
        Iterates terminal_content_layout once and asks each inline widget
        to re-clamp its size. Cheap: O(N children) with a couple of
        attribute reads.
        """
        if not hasattr(self, 'terminal_content_layout'):
            return
        max_w = self._inline_max_width()
        for i in range(self.terminal_content_layout.count()):
            w = self.terminal_content_layout.itemAt(i).widget()
            if isinstance(w, (InlineCodeBlockWidget, InlineMediaWidget)):
                if hasattr(w, 'set_inline_max_width'):
                    w.set_inline_max_width(max_w)

    # ------------------------------------------------------------------
    # Shadow animation
    # ------------------------------------------------------------------

    def animate_shadow_to_position(self):
        """
        Animate shadow offset from (0,0) to its target on widget appearance.
        Uses QVariantAnimation with OutCubic easing for a smooth 'pop' effect.

        When the active theme has no shadow (``theme.shadow is None``),
        this is a no-op — paper/flat themes remove the entire shadow
        affordance, including this entrance animation.
        """
        spec = self._theme.shadow
        if spec is None:
            # Active theme is flat — make sure no shadow lingers.
            target = self._proxy if self._proxy is not None else self
            if target.graphicsEffect() is not None:
                target.setGraphicsEffect(None)
            return

        # 1. Determine target (Proxy for embedded, self for standalone)
        shadow_target = self._proxy if self._proxy is not None else self

        # 2. Determine shadow color from theme
        rgba = spec.color_dark if getattr(self, '_is_dark_mode', False) else spec.color
        shadow_color = QColor(*rgba)

        # 3. Create/Configure the shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(spec.blur_radius)
        shadow.setColor(shadow_color)
        shadow.setOffset(0, 0)
        shadow_target.setGraphicsEffect(shadow)

        # Register in the RioWindow's shadowed-items cache so that
        # theme transitions (glass→paper) can find this proxy and
        # remove the effect.  Without this, _animate_theme_transition
        # only sees items in _shadowed_items and silently skips this
        # proxy — leaving a stale shadow on paper/flat themes.
        mw = self._find_main_window()
        if mw is not None and hasattr(mw, 'register_shadowed'):
            mw.register_shadowed(shadow_target)

        # 4. Cleanup existing animation
        if hasattr(self, '_shadow_pos_anim') and self._shadow_pos_anim:
            self._shadow_pos_anim.stop()
            self._shadow_pos_anim.deleteLater()
            self._shadow_pos_anim = None

        # 5. Initialize Animation
        # We animate a QPointF directly from (0,0) to the theme's offset
        anim = QVariantAnimation(self)
        anim.setDuration(480) # ~30 steps * 16ms
        anim.setStartValue(QPointF(0, 0))
        anim.setEndValue(QPointF(spec.offset_x, spec.offset_y))
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def update_shadow_pos(pos):
            shadow.setOffset(pos)
            # Trigger updates to ensure the scene repaints the shadow area
            self.update()
            if self.parent():
                self.parent().update()

        anim.valueChanged.connect(update_shadow_pos)

        # 6. Store reference and start
        self._shadow_pos_anim = anim
        anim.start()


    def animate_shadow_color(self, entering_terminal: bool):
            """
            Animate shadow color and blur between base and active scheme shadow color.
            Uses QVariantAnimation to ensure smooth timing regardless of frame rate.

            No-op on themes without shadows (paper/flat) — terminal-mode
            entry/exit on flat themes is signalled only via the focus
            tint on the frame, not via shadow tint changes.
            """
            spec = self._theme.shadow
            if spec is None:
                return

            shadow_target = self._proxy if self._proxy is not None else self

            # 1. Grab or create the shadow effect
            shadow = shadow_target.graphicsEffect()
            if not isinstance(shadow, QGraphicsDropShadowEffect):
                shadow = QGraphicsDropShadowEffect(self)
                shadow.setBlurRadius(spec.blur_radius)
                shadow.setOffset(QPointF(spec.offset_x, spec.offset_y))
                shadow_target.setGraphicsEffect(shadow)

            # 2. Setup color targets — base from theme, scheme from active palette
            base_rgba = spec.color_dark if getattr(self, '_is_dark_mode', False) else spec.color
            base_color = QColor(*base_rgba)

            scheme_color = self._parse_rgba(self._active_shadow_color)

            start_color = base_color if entering_terminal else scheme_color
            end_color = scheme_color if entering_terminal else base_color

            # 3. Setup blur targets — entry blooms from theme baseline to baseline+20
            base_blur = float(spec.blur_radius)
            bloom_blur = base_blur + 20.0
            start_blur = base_blur if entering_terminal else bloom_blur
            end_blur = bloom_blur if entering_terminal else base_blur

            # 4. Clean up any existing shadow animation
            if hasattr(self, '_shadow_color_anim') and self._shadow_color_anim:
                self._shadow_color_anim.stop()
                self._shadow_color_anim.deleteLater()

            # 5. Initialize the animation (0.0 to 1.0 progress)
            anim = QVariantAnimation(self)
            anim.setDuration(350)  # Roughly equivalent to 35 steps @ 10ms
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.InOutCubic)

            def update_shadow(t):
                # Manually interpolate color channels
                r = int(start_color.red() + (end_color.red() - start_color.red()) * t)
                g = int(start_color.green() + (end_color.green() - start_color.green()) * t)
                b = int(start_color.blue() + (end_color.blue() - start_color.blue()) * t)
                a = int(start_color.alpha() + (end_color.alpha() - start_color.alpha()) * t)
                shadow.setColor(QColor(r, g, b, a))

                # Interpolate blur
                blur = start_blur + (end_blur - start_blur) * t
                shadow.setBlurRadius(blur)

            anim.valueChanged.connect(update_shadow)

            # Store reference and start
            self._shadow_color_anim = anim
            anim.start()


    # ------------------------------------------------------------------
    # Dark mode support (called from RioWindow.toggle_dark_mode)
    # ------------------------------------------------------------------

    def _toggle_dark_mode_from_terminal(self):
        """Called from /dark command — walk up to the RioWindow and toggle."""
        # Find main window through proxy → scene → views chain
        main_window = None
        if self._proxy and self._proxy.scene():
            views = self._proxy.scene().views()
            if views:
                w = views[0].window()
                if hasattr(w, 'toggle_dark_mode'):
                    main_window = w
        # Fallback: walk parent chain
        if main_window is None:
            p = self.parent()
            while p is not None:
                if hasattr(p, 'toggle_dark_mode'):
                    main_window = p
                    break
                p = p.parent() if hasattr(p, 'parent') else None

        if main_window:
            main_window.toggle_dark_mode()
        else:
            self.append_text("Cannot find main window for dark mode toggle.\n", self.C_ERROR)

    # ------------------------------------------------------------------
    # Theme switching (animated)
    # ------------------------------------------------------------------

    def apply_theme(self, new_theme: _Theme, dark: bool, duration_ms: int,
                    *, paper_bg_rgb: tuple = None):
        """Animate this terminal's frame, text, input, and shadow toward
        the look defined by ``new_theme``.

        Called by ``RioWindow.set_theme``.  ``duration_ms`` of 0 means
        snap immediately (no animation) — used for headless tests and
        when the caller passes ``animate=False``.

        ``paper_bg_rgb`` — optional (r, g, b) tuple for the terminal's
        unique paper-pastel fill. When switching to a paper theme, each
        terminal receives a distinct colour from the palette generated
        by ``RioWindow._generate_paper_palette``.  When switching to
        glass, this is None and the frame reverts to the theme's
        translucent fill.

        We reuse the existing ``_animate_*_dark_mode`` helpers because
        the underlying mechanism — interpolating an RGBA target over a
        QVariantAnimation — is the same.  The only difference is the
        *target*: we pull from ``new_theme`` instead of the active theme,
        so we have to swap ``self._theme_name`` first and let the
        helpers read the new values via ``self._theme``.
        """
        # Idempotent for animated transitions, but always apply when
        # duration_ms=0 — a snap call is the caller saying "force-sync
        # this terminal to the given theme right now, regardless of
        # what we think we already have".  Used by RioWindow when
        # creating a new terminal, so the frame QSS / inner padding /
        # input targets all match the active theme on first paint.
        if new_theme.name == self._theme_name and duration_ms > 0:
            return

        # Stash the paper colour so it can be re-applied on dark-mode
        # toggles and focus animations without a fresh palette.
        self._paper_bg_rgb = paper_bg_rgb

        # Swap the active theme *before* the animations start so any
        # template strings the helpers compose (border-width, radius,
        # input border-radius, frame-stylesheet shape) use the new
        # theme's values mid-flight.
        self._theme_name = new_theme.name
        self._is_dark_mode = dark
        if hasattr(self, '_dm_color_cache'):
            self._dm_color_cache.clear()

        # Steps: 16 ms/tick is the project's convention.
        steps = max(1, duration_ms // 16) if duration_ms > 0 else 0

        # ---- Update default char colors so NEW text uses the right color ----
        if dark:
            self.C_DEFAULT = "rgba(230, 230, 230, 240)"
            self.C_USER    = "rgba(230, 230, 230, 240)"
        else:
            self.C_DEFAULT = "rgba(0, 0, 0, 230)"
            self.C_USER    = "rgba(0, 0, 0, 230)"

        f, txt, inp = new_theme.frame, new_theme.text, new_theme.input

        def _rgba(t):
            return f"rgba({t[0]}, {t[1]}, {t[2]}, {t[3]})"

        if dark:
            border_color = _rgba(f.border_rgba_dark)
            text_rgba    = _rgba(txt.default_rgba_dark)
            ib = (inp.bg_rgb_dark[0], inp.bg_rgb_dark[1], inp.bg_rgb_dark[2],
                  inp.focus_alpha_dark)
            input_bg     = _rgba(ib)
            input_text   = _rgba(inp.text_rgba_dark)
            selection_bg = _rgba(txt.selection_rgba_dark)
        else:
            border_color = _rgba(f.border_rgba)
            text_rgba    = _rgba(txt.default_rgba)
            ib = (inp.bg_rgb[0], inp.bg_rgb[1], inp.bg_rgb[2], inp.focus_alpha)
            input_bg     = _rgba(ib)
            input_text   = _rgba(inp.text_rgba)
            selection_bg = _rgba(txt.selection_rgba)
        input_focus_border = (
            "rgba(160, 160, 255, 200)" if dark else "rgba(100, 100, 255, 200)"
        )

        # ---- Snap path (no animation requested) ----
        if steps == 0:
            self.terminal_frame.setStyleSheet(
                new_theme.frame_stylesheet(dark, focus_alpha=self._frame_focus_alpha)
            )
            self._set_input_bg_target(
                ib[0], ib[1], ib[2],
                inp.focus_alpha_dark if dark else inp.focus_alpha,
            )
            self._apply_input_style()
            self._reapply_shadow_for_theme(dark)
            self._propagate_theme_to_inline_widgets(dark)
            self._apply_frame_padding(f.inner_padding)
            return

        # ---- Animated path ----
        self._animate_frame_dark_mode(border_color, steps,
                                      paper_bg_rgb=paper_bg_rgb)
        self._animate_text_dark_mode(text_rgba, selection_bg, steps)
        self._animate_input_dark_mode(input_bg, input_text, input_focus_border, steps)

        self._set_input_bg_target(
            ib[0], ib[1], ib[2],
            inp.focus_alpha_dark if dark else inp.focus_alpha,
        )

        self._apply_frame_padding(f.inner_padding)

        # Shadow: handled by RioWindow's _animate_theme_transition for
        # the proxy-level shadow.  But we may still need to re-skin the
        # standalone (popped-out) case, where the shadow lives on `self`
        # rather than the proxy.
        if self._proxy is None:
            self._reapply_shadow_for_theme(dark)

        # Inline widgets snap to the new mode at end of animation
        self._propagate_theme_to_inline_widgets(dark)

    def _apply_frame_padding(self, pad: int):
        """Update the inner ContentsMargins of the terminal frame layout."""
        layout = self.terminal_frame.layout()
        if layout is not None:
            layout.setContentsMargins(pad, pad, pad, pad)

    def _reapply_shadow_for_theme(self, dark: bool):
        """Add/remove the drop shadow on this terminal to match the active theme.

        Used in the standalone (popped-out) case where shadows live on
        the widget itself.  In the embedded case (proxy), RioWindow's
        ``_animate_theme_transition`` handles shadow add/remove globally.
        """
        target = self._proxy if self._proxy is not None else self
        spec = self._theme.shadow
        if spec is None:
            if target.graphicsEffect() is not None:
                target.setGraphicsEffect(None)
            return
        # Theme expects shadows — install fresh effect if missing
        eff = target.graphicsEffect()
        if not isinstance(eff, QGraphicsDropShadowEffect):
            eff = QGraphicsDropShadowEffect(self)
            target.setGraphicsEffect(eff)
        eff.setBlurRadius(spec.blur_radius)
        eff.setOffset(QPointF(spec.offset_x, spec.offset_y))
        rgba = spec.color_dark if dark else spec.color
        eff.setColor(QColor(*rgba))

    def _propagate_theme_to_inline_widgets(self, dark: bool):
        """Re-skin inline code/media widgets after a theme switch."""
        if not hasattr(self, 'terminal_content_layout'):
            return
        for i in range(self.terminal_content_layout.count()):
            w = self.terminal_content_layout.itemAt(i).widget()
            if isinstance(w, (InlineCodeBlockWidget, InlineMediaWidget)):
                if hasattr(w, 'set_dark_mode'):
                    w.set_dark_mode(dark)
                else:
                    w._dark_mode = dark
                    if hasattr(w, '_apply_frame_theme'):
                        w._apply_frame_theme()

    # ------------------------------------------------------------------
    # Dark mode (animated)
    # ------------------------------------------------------------------

    def set_dark_mode(self, enabled: bool, duration_steps: int = 50):
        """Animate this terminal between light and dark mode.

        Targets (border, text, input bg, selection) come from the active
        theme so a paper-themed terminal fades to its hairline ink-on-paper
        colour scheme, while a glass-themed one fades to its translucent
        white-on-charcoal scheme.  Glass values are tuned to be identical
        to the historical hardcoded constants for backward compatibility.

        Transitions:
          - Frame border:  light -> dark stroke (per theme)
          - Text color in text_displays: per theme.text
          - Command input styling: per theme.input
          - Existing output text: recolor inline char formats
          - Shadow tint is handled globally by RioWindow._start_dark_mode_animation
        """
        self._is_dark_mode = enabled
        # See note in the theme-transition path: clear the memo so the
        # cache doesn't grow stale entries across many mode flips.
        if hasattr(self, '_dm_color_cache'):
            self._dm_color_cache.clear()

        # ---- Update default text colors so NEW text uses the right color ----
        if enabled:
            self.C_DEFAULT = "rgba(230, 230, 230, 240)"
            self.C_USER    = "rgba(230, 230, 230, 240)"
        else:
            self.C_DEFAULT = "rgba(0, 0, 0, 230)"
            self.C_USER    = "rgba(0, 0, 0, 230)"

        # ---- Re-derive scheme-mapped colors for the new mode ----
        # Shell echo/output are always black/white via the property.
        self.C_SHELL   = self._active_shell_echo_color
        self.C_AGENT   = self._dm_adjust_color(self._active_scheme["agent"])
        self.C_SUCCESS = self._dm_adjust_color(self._active_scheme["success"])
        self.C_ERROR   = self._dm_adjust_color(self._active_scheme["error"])
        self.C_INFO    = self._dm_adjust_color(self._active_scheme["info"])

        # ---- Target colors from active theme ----
        theme = self._theme
        f, txt, inp = theme.frame, theme.text, theme.input

        def _rgba(t):  # quick formatter
            return f"rgba({t[0]}, {t[1]}, {t[2]}, {t[3]})"

        if enabled:
            border_color = _rgba(f.border_rgba_dark)
            text_rgba    = _rgba(txt.default_rgba_dark)
            # For input bg in dark mode, use focused fill so the animation
            # has a visible target; the focus-fade handles idle separately.
            ib = (inp.bg_rgb_dark[0], inp.bg_rgb_dark[1], inp.bg_rgb_dark[2],
                  inp.focus_alpha_dark)
            input_bg     = _rgba(ib)
            input_text   = _rgba(inp.text_rgba_dark)
            selection_bg = _rgba(txt.selection_rgba_dark)
        else:
            border_color = _rgba(f.border_rgba)
            text_rgba    = _rgba(txt.default_rgba)
            ib = (inp.bg_rgb[0], inp.bg_rgb[1], inp.bg_rgb[2], inp.focus_alpha)
            input_bg     = _rgba(ib)
            input_text   = _rgba(inp.text_rgba)
            selection_bg = _rgba(txt.selection_rgba)

        # Input focus border colour isn't part of the theme dataclass yet
        # (it's a shared ~purple highlight on both themes); keep the legacy
        # values so the focus underline doesn't shift hue between themes.
        input_focus_border = (
            "rgba(160, 160, 255, 200)" if enabled
            else "rgba(100, 100, 255, 200)"
        )

        # ---- Animate frame border ----
        self._animate_frame_dark_mode(border_color, duration_steps)

        # ---- Animate text displays (stylesheet for future + recolor existing) ----
        self._animate_text_dark_mode(
            text_rgba, selection_bg, duration_steps
        )

        # ---- Animate command input ----
        self._animate_input_dark_mode(
            input_bg, input_text, input_focus_border, duration_steps
        )

        # ---- Update focus targets so subsequent focus-in/out uses new theme ----
        # (RGB and alpha — the focus animation reads these via _input_bg_*.)
        ib2 = inp.bg_rgb_dark if enabled else inp.bg_rgb
        self._set_input_bg_target(
            ib2[0], ib2[1], ib2[2],
            inp.focus_alpha_dark if enabled else inp.focus_alpha,
        )

        # ---- Propagate to live inline widgets ----
        # Inline code/media widgets are styled at construction time with
        # explicit dark/light QSS, so we have to re-theme them here.
        # We walk terminal_content_layout and update any of our widgets;
        # other widgets (Acme, scene panel, etc.) are left untouched.
        if hasattr(self, 'terminal_content_layout'):
            for i in range(self.terminal_content_layout.count()):
                w = self.terminal_content_layout.itemAt(i).widget()
                if isinstance(w, (InlineCodeBlockWidget, InlineMediaWidget)):
                    if hasattr(w, 'set_dark_mode'):
                        w.set_dark_mode(enabled)
                    else:
                        # InlineMediaWidget doesn't expose set_dark_mode
                        # because its content is rendered once at init;
                        # just re-apply the frame theme.
                        w._dark_mode = enabled
                        if hasattr(w, '_apply_frame_theme'):
                            w._apply_frame_theme()

    def _animate_frame_dark_mode(self, target_border: str, steps: int,
                                 *, paper_bg_rgb: tuple = None):
        """Animate terminal_frame border + fill for theme/dark-mode transitions.

        Border width, fill, and radius are sourced from the active theme so
        a paper-themed scene keeps its 1-px hairline (not 2-px) during the
        dark-mode fade.

        When ``paper_bg_rgb`` is provided (switching to paper), the fill
        animates from the current background to the given (r, g, b) at
        full opacity — each terminal gets its own distinct pastel.
        When ``paper_bg_rgb`` is None (switching to glass or dark-mode
        toggle), the fill animates toward the theme's idle fill.
        """
        import re as _re

        # 1. Clean up existing animation to prevent collisions/crashes
        if hasattr(self, '_dm_frame_anim') and self._dm_frame_anim:
            self._dm_frame_anim.stop()
            self._dm_frame_anim.deleteLater()
            self._dm_frame_anim = None

        # 2. Parse current state from stylesheet
        current_style = self.terminal_frame.styleSheet()

        # Theme-defined geometry
        f = self._theme.frame
        bw = f.border_width
        radius = f.radius

        # Current Border (best-effort regex — falls back to theme defaults)
        m = _re.search(
            r'border:\s*\d+px\s+solid\s+rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            current_style
        )
        if m:
            sr, sg, sb, sa = (int(m.group(i)) for i in range(1, 5))
        else:
            sr, sg, sb, sa = (
                f.border_rgba_dark if self._is_dark_mode else f.border_rgba
            )

        # Current Background
        m2 = _re.search(
            r'background-color:\s*rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            current_style
        )
        if m2:
            bg_sr, bg_sg, bg_sb, bg_sa = (int(m2.group(i)) for i in range(1, 5))
        else:
            bg = (
                f.fill_rgba_idle_dark if self._is_dark_mode else f.fill_rgba_idle
            )
            bg_sr, bg_sg, bg_sb, bg_sa = bg

        # Target fill: paper overrides with opaque pastel; otherwise
        # use theme idle fill.
        if paper_bg_rgb is not None:
            bg_tr, bg_tg, bg_tb, bg_ta = (*paper_bg_rgb, 255)
        else:
            bg_target = (
                f.fill_rgba_idle_dark if self._is_dark_mode else f.fill_rgba_idle
            )
            bg_tr, bg_tg, bg_tb, bg_ta = bg_target

        # 3. Parse Target border
        tc = self._parse_rgba(target_border)
        tr_, tg_, tb_, ta_ = tc.red(), tc.green(), tc.blue(), tc.alpha()

        # 4. Initialize Animation
        anim = QVariantAnimation(self)
        anim.setDuration(steps * 16)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutQuad)

        # Pre-compute the border CSS template once — paper themes may
        # have border_width=0, in which case we render "border: none;"
        if bw <= 0:
            def _border_css(r, g, b, a): return "border: none;"
        else:
            def _border_css(r, g, b, a):
                return f"border: {bw}px solid rgba({r}, {g}, {b}, {a});"

        def update_frame(t):
            # Interpolate border
            r = int(sr + (tr_ - sr) * t)
            g = int(sg + (tg_ - sg) * t)
            b = int(sb + (tb_ - sb) * t)
            a = int(sa + (ta_ - sa) * t)
            # Interpolate fill
            fr = int(bg_sr + (bg_tr - bg_sr) * t)
            fg = int(bg_sg + (bg_tg - bg_sg) * t)
            fb = int(bg_sb + (bg_tb - bg_sb) * t)
            fa = int(bg_sa + (bg_ta - bg_sa) * t)

            self.terminal_frame.setStyleSheet(
                "QFrame {\n"
                f"    background-color: rgba({fr}, {fg}, {fb}, {fa});\n"
                f"    {_border_css(r, g, b, a)}\n"
                f"    border-radius: {radius}px;\n"
                "}\n"
            )

        anim.valueChanged.connect(update_frame)

        # 5. Store reference and start
        self._dm_frame_anim = anim
        anim.start()

    def _animate_text_dark_mode(self, target_rgba: str, selection_bg: str, steps: int):
        """Animate all text display colors for dark/light mode using QVariantAnimation.

        Earlier this function did two things per animation tick:
          1. Rebuild and re-apply a full stylesheet to every text display.
          2. Walk every previously-collected "default-coloured" fragment
             range and call setPosition / mergeCharFormat to interpolate
             that fragment's colour.

        Step 2 is what made dark-mode toggles get heavier the longer a
        terminal had been running — fragment count scales with session
        history, and each toggle did O(fragments × ticks) document
        mutations, each invalidating layout. On a long session it
        visibly stuttered for the full ~800 ms.

        New strategy:
          - Per tick: only update the stylesheet colour. That animates
            *new* text and the selection tint, which is what the user
            primarily sees mid-fade. Stylesheet swap is O(1) per text
            display per tick.
          - Once on `finished`: snap the existing fragment colours to
            their final values in a single pass. The cost is paid once
            instead of `steps` times, and the visible result is
            essentially identical because already-emitted text doesn't
            crossfade — it just shows the right colour at the end.
        """
        import re as _re

        # 1. Cleanup existing animation
        if hasattr(self, '_dm_text_anim') and self._dm_text_anim:
            self._dm_text_anim.stop()
            self._dm_text_anim.deleteLater()
            self._dm_text_anim = None

        # 2. Parse current text color from stylesheet
        if self.text_displays:
            style = self.text_displays[0].styleSheet()
            m = _re.search(r'color:\s*rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', style)
            sr, sg, sb, sa = (int(m.group(i)) for i in range(1, 5)) if m else (0, 0, 0, 230)
        else:
            sr, sg, sb, sa = 0, 0, 0, 230

        tc = self._parse_rgba(target_rgba)
        tr_, tg_, tb_, ta_ = tc.red(), tc.green(), tc.blue(), tc.alpha()
        size = getattr(self, '_font_size', 12)
        # Mono font locked to the Glass stack regardless of active
        # theme. Earlier this read self._theme.font.mono_family so paper
        # mode could ship IBM Plex Mono; user prefers the Glass mono
        # everywhere.
        mono = "'Consolas', 'Monaco', monospace"
        entering_dark = self._is_dark_mode

        # 3. Pre-collect ranges of "default-coloured" text once. We still
        # need this — but only to snap once at the end of the animation,
        # not on every tick.
        recolor_ranges = []
        for te in self.text_displays:
            doc = te.document()
            ranges = []
            block = doc.begin()
            while block.isValid():
                it = block.begin()
                while not it.atEnd():
                    frag = it.fragment()
                    if frag.isValid():
                        fg = frag.charFormat().foreground().color()
                        lum = fg.red() * 0.299 + fg.green() * 0.587 + fg.blue() * 0.114
                        if (entering_dark and lum < 80) or (not entering_dark and lum > 180):
                            ranges.append({
                                'start': frag.position(),
                                'end': frag.position() + frag.length(),
                            })
                    it += 1
                block = block.next()
            if ranges:
                recolor_ranges.append((te, ranges))

        # 4. Setup the Animation
        anim = QVariantAnimation(self)
        anim.setDuration(steps * 16)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutQuad)

        def update_text_transition(t):
            # Interpolate current color values
            curr_r = int(sr + (tr_ - sr) * t)
            curr_g = int(sg + (tg_ - sg) * t)
            curr_b = int(sb + (tb_ - sb) * t)
            curr_a = int(sa + (ta_ - sa) * t)

            # Update stylesheet (affects new text and selection). One
            # write per display per tick, no document mutations.
            css = (
                "QTextEdit {\n"
                "  background-color: transparent; border: none;\n"
                f"  color: rgba({curr_r}, {curr_g}, {curr_b}, {curr_a});\n"
                f"  selection-background-color: {selection_bg};\n"
                f"  font-family: {mono};\n"
                f"  font-size: {size}px;\n"
                "}\n"
            )
            for te, _ranges in recolor_ranges:
                te.setStyleSheet(css)
            # Also style displays that had no recolor ranges, so their
            # selection-bg / new-text colour stay consistent with the
            # rest of the scene.
            for te in self.text_displays:
                if not any(te is x for x, _ in recolor_ranges):
                    te.setStyleSheet(css)

        def snap_fragment_colors():
            """Snap existing fragment colours to their final values once,
            after the stylesheet animation completes."""
            from PySide6.QtGui import QColor as _QColor
            target = _QColor(tr_, tg_, tb_, ta_)
            fmt = QTextCharFormat()
            fmt.setForeground(target)
            for te, ranges in recolor_ranges:
                cursor = te.textCursor()
                for r_data in ranges:
                    cursor.setPosition(r_data['start'])
                    cursor.setPosition(r_data['end'], QTextCursor.KeepAnchor)
                    cursor.mergeCharFormat(fmt)

        anim.valueChanged.connect(update_text_transition)
        anim.finished.connect(snap_fragment_colors)

        # 5. Start and store
        self._dm_text_anim = anim
        anim.start()

    def _animate_input_dark_mode(self, target_bg: str, target_text: str,
                                  target_focus: str, steps: int):
        """Animate command input bg color for dark/light mode transition."""
        if hasattr(self, '_dm_input_timer'):
            self._dm_input_timer.stop()
            self._dm_input_timer.deleteLater()

        tb = self._parse_rgba(target_bg)

        # Starting values from current state
        sbr, sbg, sbb = self._input_bg_r, self._input_bg_g, self._input_bg_b
        s_target_alpha = self._input_bg_target_alpha

        # End values
        ebr, ebg, ebb, eba = tb.red(), tb.green(), tb.blue(), tb.alpha()

        step = [0]

        def lerp(a, b, t):
            return int(a + (b - a) * t)

        def tick():
            if step[0] <= steps:
                t = step[0] / steps
                t = t * t * (3.0 - 2.0 * t)

                self._input_bg_r = lerp(sbr, ebr, t)
                self._input_bg_g = lerp(sbg, ebg, t)
                self._input_bg_b = lerp(sbb, ebb, t)
                self._input_bg_target_alpha = lerp(s_target_alpha, eba, t)

                # Keep current alpha in sync: focused = target, unfocused = 0
                if self.command_input.hasFocus():
                    self._input_bg_alpha = self._input_bg_target_alpha
                else:
                    self._input_bg_alpha = 0

                self._apply_input_style()
                step[0] += 1
            else:
                self._input_bg_r = ebr
                self._input_bg_g = ebg
                self._input_bg_b = ebb
                self._input_bg_target_alpha = eba
                if self.command_input.hasFocus():
                    self._input_bg_alpha = eba
                else:
                    self._input_bg_alpha = 0
                self._apply_input_style()
                self._dm_input_timer.stop()
                self._dm_input_timer.deleteLater()
                delattr(self, '_dm_input_timer')

        self._dm_input_timer = QTimer(self)
        self._dm_input_timer.timeout.connect(tick)
        # 16 ms ≈ 60 fps. Same start(0) → start(16) fix as in
        # _animate_input_focus — see that method for the rationale.
        self._dm_input_timer.start(16)

    # ------------------------------------------------------------------
    # Pop-out / Dock  (/pop extracts to external window, /dock returns)
    # ------------------------------------------------------------------

    def _pop_to_window(self):
        """
        Extract the terminal from the QGraphicsScene and place it in a
        frameless external window with shadow effects.
        """
        if self._pop_window is not None:
            self.append_text("Already popped out. Use /dock to return.\n", self.C_INFO)
            return

        if self._proxy is None:
            self.append_text("Not embedded in a scene — nothing to pop.\n", self.C_ERROR)
            return

        scene = self._proxy.scene()
        if scene is None:
            self.append_text("Proxy has no scene.\n", self.C_ERROR)
            return

        # ---- Save state for docking back ----
        self._pop_scene = scene
        self._pop_proxy = self._proxy
        self._pop_scene_pos = self._proxy.pos()
        self._pop_size = self.size()

        # ---- Compute screen position from scene position ----
        views = scene.views()
        if views:
            view = views[0]
            view_pos = view.mapFromScene(self._pop_scene_pos)
            screen_pos = view.mapToGlobal(view_pos)
        else:
            screen_pos = QPoint(200, 200)

        # ---- Remove from scene ----
        # Clear the graphics effect BEFORE removing from proxy to avoid
        # the effect being destroyed with the proxy
        self._proxy.setGraphicsEffect(None)
        self._proxy.setWidget(None)
        scene.removeItem(self._proxy)
        self._proxy = None

        # ---- Create frameless external window ----
        window = QWidget(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        window.setAttribute(Qt.WA_TranslucentBackground, True)
        window.setAttribute(Qt.WA_NoSystemBackground, True)

        # Padding around the terminal for shadow to render into
        shadow_pad = 50
        layout = QVBoxLayout(window)
        layout.setContentsMargins(shadow_pad, shadow_pad, shadow_pad, shadow_pad)
        layout.setSpacing(0)

        # Reparent terminal into the window
        self.setParent(window)
        layout.addWidget(self)
        self.show()

        # Size the window: terminal size + shadow padding on all sides
        w = self._pop_size.width()
        h = self._pop_size.height()
        window.resize(w + shadow_pad * 2, h + shadow_pad * 2)
        window.move(screen_pos - QPoint(shadow_pad, shadow_pad))

        # ---- Apply shadow directly on the terminal widget ----
        # Only if the active theme uses shadows (paper/flat → no shadow).
        spec = self._theme.shadow
        if spec is not None:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(spec.blur_radius)
            shadow.setOffset(QPointF(spec.offset_x, spec.offset_y))
            shadow_color = self._parse_rgba(self._active_shadow_color)
            shadow.setColor(shadow_color)
            self.setGraphicsEffect(shadow)

        # ---- Enable dragging via title-bar-less window ----
        window._drag_pos = None
        window._terminal = self

        original_mouse_press = window.mousePressEvent
        original_mouse_move = window.mouseMoveEvent
        original_mouse_release = window.mouseReleaseEvent
        original_move_event = window.moveEvent

        def win_press(event):
            if event.button() == Qt.LeftButton:
                window._drag_pos = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
                event.accept()
            else:
                original_mouse_press(event)

        def win_move(event):
            if event.buttons() & Qt.LeftButton and window._drag_pos is not None:
                window.move(event.globalPosition().toPoint() - window._drag_pos)
                event.accept()
            else:
                original_mouse_move(event)

        def win_release(event):
            if event.button() == Qt.LeftButton:
                window._drag_pos = None
                event.accept()
            else:
                original_mouse_release(event)

        def win_moved(event):
            """Fires on ANY window move — our drag, WM drag, anything."""
            original_move_event(event)
            self._check_overlap()

        window.mousePressEvent = win_press
        window.mouseMoveEvent = win_move
        window.mouseReleaseEvent = win_release
        window.moveEvent = win_moved

        self._pop_window = window
        self._overlap_state = None
        self._pop_scene_view = None

        window.show()
        window.raise_()
        self.command_input.setFocus()

        # ---- Initialize overlap state ----
        self._init_overlap_monitor()

        self.append_text("Popped out to external window. /dock to return.\n", self.C_SUCCESS)

    def _dock_to_scene(self):
        """
        Return the terminal from the external window back into the
        QGraphicsScene at its original position.
        """
        if self._pop_window is None:
            self.append_text("Not popped out. Use /pop first.\n", self.C_INFO)
            return

        scene = self._pop_scene
        if scene is None:
            self.append_text("Original scene no longer exists.\n", self.C_ERROR)
            self._pop_window = None
            return

        # ---- Stop overlap monitor and reset background ----
        self._cleanup_overlap_monitor()

        # ---- Remove shadow from widget (will reapply on proxy) ----
        self.setGraphicsEffect(None)

        # ---- Remove from external window ----
        self.setParent(None)

        # ---- Restore size ----
        self.resize(self._pop_size)

        # ---- Re-embed in scene via a new QGraphicsProxyWidget ----
        from PySide6.QtWidgets import QGraphicsProxyWidget
        proxy = scene.addWidget(self)
        proxy.setPos(self._pop_scene_pos)
        self._proxy = proxy

        # ---- Reapply shadow on the proxy (only if theme uses shadows) ----
        spec = self._theme.shadow
        if spec is not None:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(spec.blur_radius)
            shadow.setOffset(QPointF(spec.offset_x, spec.offset_y))
            shadow_color = self._parse_rgba(self._active_shadow_color)
            shadow.setColor(shadow_color)
            proxy.setGraphicsEffect(shadow)
            # Register so theme transitions can find and manage this effect.
            mw = self._find_main_window()
            if mw is not None and hasattr(mw, 'register_shadowed'):
                mw.register_shadowed(proxy)

        # ---- Tear down external window ----
        self._pop_window.close()
        self._pop_window.deleteLater()
        self._pop_window = None
        self._pop_scene = None
        self._pop_proxy = None
        self._pop_scene_pos = None
        self._pop_size = None

        self.show()
        self.command_input.setFocus()

        # Reset frame to match the active theme (scene-embedded state)
        self.terminal_frame.setStyleSheet(
            self._theme.frame_stylesheet(self._is_dark_mode,
                                         focus_alpha=self._frame_focus_alpha)
        )

        self.append_text("Docked back into scene.\n", self.C_SUCCESS)

    # ------------------------------------------------------------------
    # Overlap monitor — event-driven, fires only on window move
    # ------------------------------------------------------------------

    def _init_overlap_monitor(self):
        """
        Initialize overlap tracking state and run one initial check.

        The actual checking is driven by moveEvent on the pop window
        (hooked in _pop_to_window) — no polling timer, zero cost at rest.
        """
        self._overlap_state = None  # None = first check, True = over scene, False = outside

        # Cache the scene view
        if self._pop_scene and self._pop_scene.views():
            self._pop_scene_view = self._pop_scene.views()[0]
        else:
            self._pop_scene_view = None

        # Set initial state
        self._check_overlap()

    def _cleanup_overlap_monitor(self):
        """Clear overlap tracking state."""
        # Kill any in-flight opacity animation
        if hasattr(self, '_frame_opacity_timer') and self._frame_opacity_timer is not None:
            self._frame_opacity_timer.stop()
            self._frame_opacity_timer.deleteLater()
            delattr(self, '_frame_opacity_timer')
        self._overlap_state = None
        self._pop_scene_view = None

    def _check_overlap(self):
        """
        Compare pop-out window against the scene view's screen rect.
        Trigger opacity animation on state change.

        Called from the pop window's moveEvent — only runs when the
        window actually moves.  Cost: two mapToGlobal, one rect
        intersection, one float divide.
        """
        if self._pop_window is None or self._pop_scene_view is None:
            return

        # Scene view's global screen rectangle
        view = self._pop_scene_view
        try:
            view_global = view.mapToGlobal(QPoint(0, 0))
            view_rect = QRectF(
                view_global.x(), view_global.y(),
                view.viewport().width(), view.viewport().height(),
            )
        except RuntimeError:
            # View was deleted
            self._cleanup_overlap_monitor()
            return

        # Pop window's terminal area (excluding shadow padding)
        win_geo = self._pop_window.frameGeometry()
        shadow_pad = 50
        terminal_rect = QRectF(
            win_geo.x() + shadow_pad,
            win_geo.y() + shadow_pad,
            win_geo.width() - shadow_pad * 2,
            win_geo.height() - shadow_pad * 2,
        )

        # Overlap ratio: how much of the terminal is over the scene view
        intersection = view_rect.intersected(terminal_rect)
        if terminal_rect.width() > 0 and terminal_rect.height() > 0:
            overlap_area = intersection.width() * intersection.height()
            terminal_area = terminal_rect.width() * terminal_rect.height()
            ratio = overlap_area / terminal_area
        else:
            ratio = 0.0

        # Threshold: >40% overlap = "over scene" → transparent
        over_scene = ratio > 0.4

        if over_scene != self._overlap_state:
            self._overlap_state = over_scene
            if over_scene:
                self._animate_frame_opacity(target_alpha=0)
            else:
                self._animate_frame_opacity(target_alpha=230)

    def _animate_frame_opacity(self, target_alpha: int, duration_steps: int = 25):
        """
        Animate terminal_frame background-color alpha from current to target.
        Uses QVariantAnimation for time-respecting, lag-resistant transitions.
        """
        import re as _re

        # 1. Kill any running frame opacity animation safely
        if hasattr(self, '_frame_opacity_anim') and self._frame_opacity_anim:
            self._frame_opacity_anim.stop()
            self._frame_opacity_anim.deleteLater()
            self._frame_opacity_anim = None

        # 2. Determine correct background RGB + border based on current mode
        dark = getattr(self, '_is_dark_mode', False)
        if dark:
            r, g, b = 30, 30, 35
            border_css = "border: 2px solid rgba(200, 200, 200, 220);"
        else:
            r, g, b = 255, 255, 255
            border_css = "border: 2px solid rgba(150, 150, 150, 200);"

        # 3. Parse current alpha from the stylesheet to ensure a smooth start
        current_style = self.terminal_frame.styleSheet()
        m = _re.search(r'background-color:\s*rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)', current_style)
        start_alpha = int(m.group(1)) if m else 0

        # 4. Initialize the QVariantAnimation
        anim = QVariantAnimation(self)
        # Map original duration_steps to milliseconds (~16ms per step)
        anim.setDuration(duration_steps * 16)
        anim.setStartValue(start_alpha)
        anim.setEndValue(target_alpha)
        anim.setEasingCurve(QEasingCurve.InOutQuad)  # Standard smoothstep equivalent

        def update_opacity(alpha_value):
            # Update the entire stylesheet with the new interpolated alpha
            self.terminal_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba({r}, {g}, {b}, {alpha_value});
                    {border_css}
                    border-radius: 5px;
                }}
            """)

        anim.valueChanged.connect(update_opacity)

        # 5. Store reference and start
        self._frame_opacity_anim = anim
        anim.start()

    # ------------------------------------------------------------------
    # Resize handling
    # ------------------------------------------------------------------
    
    def _get_resize_corner(self, pos):
        """Detect which corner (if any) the mouse is near.
        Returns: 'tl', 'tr', 'bl', 'br', or None
        """
        rect = self.rect()
        margin = self.RESIZE_MARGIN
        
        # Check corners (priority over edges)
        if pos.x() <= margin and pos.y() <= margin:
            return 'tl'  # Top-left
        elif pos.x() >= rect.width() - margin and pos.y() <= margin:
            return 'tr'  # Top-right
        elif pos.x() <= margin and pos.y() >= rect.height() - margin:
            return 'bl'  # Bottom-left
        elif pos.x() >= rect.width() - margin and pos.y() >= rect.height() - margin:
            return 'br'  # Bottom-right
        
        return None
    
    def _update_cursor_for_resize(self, corner):
        """Update cursor shape based on resize corner."""
        if corner == 'tl' or corner == 'br':
            self.setCursor(Qt.SizeFDiagCursor)
        elif corner == 'tr' or corner == 'bl':
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    # Misc overrides
    # ------------------------------------------------------------------


    def mousePressEvent(self, event):
        """Handle clicking to drag the widget or resize from corners."""
        if event.button() == Qt.LeftButton:
            # Check if clicking on a resize corner
            corner = self._get_resize_corner(event.position().toPoint())
            
            if corner:
                # Start resizing
                self._resizing = True
                self._resize_corner = corner
                self._resize_start_pos = event.globalPosition().toPoint()
                if self._proxy is not None:
                    proxy_pos = self._proxy.pos()
                    self._resize_start_geometry = QRectF(
                        proxy_pos.x(), proxy_pos.y(),
                        self.width(), self.height()
                    ).toRect()
                else:
                    self._resize_start_geometry = self.geometry()
                event.accept()
                return
            elif event.modifiers() & Qt.ControlModifier:
                # Ctrl+click to start dragging
                self._dragging = True
                if self._proxy is not None:
                    # Record the mouse position in scene coords relative to proxy origin
                    view = self._proxy.scene().views()[0]
                    scene_pos = view.mapToScene(view.mapFromGlobal(event.globalPosition().toPoint()))
                    self._drag_offset = scene_pos - self._proxy.pos()
                else:
                    self._drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        
        super().mousePressEvent(event)  

    def mouseMoveEvent(self, event):
        """Handle dragging the widget or resizing from corners."""
        # Update cursor if hovering over corners
        if not self._resizing and event.buttons() == Qt.NoButton:
            corner = self._get_resize_corner(event.position().toPoint())
            self._update_cursor_for_resize(corner)
        
        if event.buttons() & Qt.LeftButton:
            if self._resizing and self._resize_corner:
                # Handle resizing
                delta = event.globalPosition().toPoint() - self._resize_start_pos
                geo = self._resize_start_geometry
                
                # Minimum size constraints
                min_width = 200
                min_height = 150
                
                if self._resize_corner == 'tl':
                    new_x = geo.x() + delta.x()
                    new_y = geo.y() + delta.y()
                    new_width = geo.width() - delta.x()
                    new_height = geo.height() - delta.y()
                    
                    if new_width >= min_width and new_height >= min_height:
                        self._set_geometry_proxy_aware(new_x, new_y, new_width, new_height)
                
                elif self._resize_corner == 'tr':
                    new_y = geo.y() + delta.y()
                    new_width = geo.width() + delta.x()
                    new_height = geo.height() - delta.y()
                    
                    if new_width >= min_width and new_height >= min_height:
                        self._set_geometry_proxy_aware(geo.x(), new_y, new_width, new_height)
                
                elif self._resize_corner == 'bl':
                    new_x = geo.x() + delta.x()
                    new_width = geo.width() - delta.x()
                    new_height = geo.height() + delta.y()
                    
                    if new_width >= min_width and new_height >= min_height:
                        self._set_geometry_proxy_aware(new_x, geo.y(), new_width, new_height)
                
                elif self._resize_corner == 'br':
                    new_width = geo.width() + delta.x()
                    new_height = geo.height() + delta.y()
                    
                    if new_width >= min_width and new_height >= min_height:
                        if self._proxy is not None:
                            self.setFixedSize(int(new_width), int(new_height))
                        else:
                            self.resize(int(new_width), int(new_height))
                
                event.accept()
            elif getattr(self, '_dragging', False):
                # Handle Ctrl+drag movement
                if self._proxy is not None:
                    view = self._proxy.scene().views()[0]
                    scene_pos = view.mapToScene(view.mapFromGlobal(event.globalPosition().toPoint()))
                    self._proxy.setPos(scene_pos - self._drag_offset)
                else:
                    new_pos = event.globalPosition().toPoint() - self._drag_start_pos
                    self.move(new_pos)
                event.accept()
            else:
                super().mouseMoveEvent(event)
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to end resizing or dragging."""
        if event.button() == Qt.LeftButton:
            if self._resizing:
                self._resizing = False
                self._resize_corner = None
                self._resize_start_pos = None
                self._resize_start_geometry = None
                event.accept()
                return
            if getattr(self, '_dragging', False):
                self._dragging = False
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def _set_geometry_proxy_aware(self, x, y, w, h):
        """Set position and size, routing through the proxy when embedded in a scene."""
        if self._proxy is not None:
            self._proxy.setPos(x, y)
            self.setFixedSize(int(w), int(h))
        else:
            self.setGeometry(int(x), int(y), int(w), int(h))

    def make_always_on_top(self):  # DEAD CODE — unused, kept for external callers
        self.raise_()

    def show(self):
        super().show()
        self.raise_()

    def paintEvent(self, event):
        super().paintEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()
        # Keep the focus-tint overlay sized to terminal_frame.
        if hasattr(self, '_focus_overlay'):
            self._focus_overlay.sync_geometry()
        # NOTE: a previous version called QApplication.processEvents()
        # here. That was a re-entrancy trap: during drag-resize this
        # fires per pixel, and processEvents pumps the *entire* event
        # queue (paints, mouse moves, timers) before returning. It also
        # defeats the inline-reflow debounce on the next line — that
        # debounce only works if we let resizeEvent return.
        # Debounce inline-widget reflow to once per ~100 ms of quiet.
        # During an active drag-resize we'd otherwise re-clamp every
        # widget on every pixel of motion, which jitters layout work.
        if not hasattr(self, '_inline_reflow_timer'):
            self._inline_reflow_timer = QTimer(self)
            self._inline_reflow_timer.setSingleShot(True)
            self._inline_reflow_timer.timeout.connect(self._propagate_inline_width)
        self._inline_reflow_timer.start(100)


class ShellReaderWorker(QObject):
    output_ready = Signal(str)

    def __init__(self, fd):
        super().__init__()
        self.fd = fd
        self._running = True

    def run(self):
        while self._running:
            try:
                # Read from PTY
                data = os.read(self.fd, 1024)
                if not data:
                    break
                self.output_ready.emit(data.decode('utf-8', errors='replace'))
            except Exception:
                break
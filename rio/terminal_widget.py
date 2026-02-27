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
  /pop                   Detach terminal to floating window
  /dock                  Re-dock terminal into scene
  /restart               Restart shell (fresh env, re-seed vars)
  /setup                 Unmount & remount 9pfuse (LLMFS + Rio)
  /mount <IP!Port> <n>   Mount 9P service at /n/name via 9pfuse
  >>> <code>             Execute Python code
  $ <command>            Execute shell command
  $                      Toggle persistent shell mode
  <text>                 Send prompt to connected agent
"""

from PySide6.QtWidgets import (
    QWidget, QTextEdit, QVBoxLayout, QHBoxLayout, QFrame,
    QSizePolicy, QApplication, QScrollArea, QGraphicsDropShadowEffect, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QPointF, QRectF, QThread, QObject, Slot
from PySide6.QtGui import QColor, QPalette, QFont, QTextCursor, QKeyEvent, QTextCharFormat
import asyncio
import errno
import json
import os
import signal
import socket
import struct
import fcntl
import subprocess
import tempfile
import time
import re
from typing import Dict

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
_Tattach  = 104; _Rattach  = 105
_Rerror   = 107
_Twalk    = 110; _Rwalk    = 111
_Topen    = 112; _Ropen    = 113
_Tread    = 116; _Rread    = 117
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

    def __init__(self, host: str = "localhost", port: int = 5640):
        self.host = host
        self.port = port
        self.sock: socket.socket = None
        self.msize = 8192
        self._tag = 0
        self._fids = {}       # path_key -> fid
        self._next_fid = 1
        self._root_fid = 0

    # ---- connection lifecycle -----------------------------------------

    def connect(self):
        """Connect and perform Tversion + Tattach."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.host, self.port))
        # No Nagle — we want low latency
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Tversion
        self._version()

        # Tattach with root fid 0
        self._attach()

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

    def _attach(self):
        uname = b"rio"
        aname = b""
        payload = struct.pack("<II", self._root_fid, _NOFID)
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

    def __init__(self, agent_path: str, host: str = "localhost", port: int = 5640):
        super().__init__()
        self.agent_path = agent_path
        self.host = host
        self.port = port
        self._running = True

    def run(self):
        client = P9Client(self.host, self.port)
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
                 **_kwargs):
        super().__init__()
        self.agent_path = agent_path
        self.host = host
        self.port = port
        self._running = True

    def run(self):
        client = P9Client(self.host, self.port)
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

    def __init__(self, parent=None, llmfs_mount=None,
                 rio_mount=None,
                 p9_host="localhost", p9_port=5640):
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
        self._suppress_echo_line = None  # Command text to suppress from PTY echo
        self._suppress_echo_buf = ""     # Accumulator for multi-chunk echo suppression
        self._suppress_shell_output = False  # Suppress ALL PTY output (during seeding)
        self.acme_panel = None
        self.operator_panel = None
        self.version_panel = None
        self._active_panel = None  # Currently visible side panel in the splitter
        self._proxy = None  # Set by main.py when added to QGraphicsScene
        self._font_size = 12  # Default font size (px)
        self._is_dark_mode = False  # Dark mode state

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
        self._setup_shell_process()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowFlags(Qt.Widget)
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
        self.setMinimumSize(200, 150)

    def setup_terminal_frame(self):
        self.terminal_frame = QFrame()
        self.terminal_frame.setFrameStyle(QFrame.StyledPanel)
        self.terminal_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0);
                border: 2px solid rgba(150, 150, 150, 200);
                border-radius: 5px;
            }
        """)

        terminal_layout = QVBoxLayout(self.terminal_frame)
        terminal_layout.setContentsMargins(10, 10, 10, 10)
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
        size = getattr(self, '_font_size', 12)
        dark = getattr(self, '_is_dark_mode', False)
        if dark:
            text_color = "rgba(230, 230, 230, 255)"
            sel_bg = "rgba(100, 100, 255, 120)"
        else:
            text_color = "rgba(0, 0, 0, 255)"
            sel_bg = "rgba(100, 100, 255, 100)"
        te.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent; border: none;
                color: {text_color};
                selection-background-color: {sel_bg};
                font-family: 'Consolas', 'Monaco', monospace;
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
        te.document().contentsChanged.connect(lambda: self._adjust_height(te))
        # Install Plan 9 mouse menu handler
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
        font = QFont("Consolas", size)

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
                    font-family: 'Consolas', 'Monaco', monospace;
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
        self.command_input.setFont(QFont("Consolas", 10))
        self.command_input.setMaximumHeight(60)
        self.command_input.setCursorWidth(2)

        # Focus animation state — tracks current bg rgba + target alpha
        self._input_bg_r = 255
        self._input_bg_g = 255
        self._input_bg_b = 255
        self._input_bg_alpha = 0          # current animated alpha
        self._input_bg_target_alpha = 150  # alpha when focused
        self._input_focus_anim = None      # QTimer for animation

        self._apply_input_style()
        self.command_input.setPlaceholderText("Enter command or prompt...")
        self.command_input.installEventFilter(self)
        self.input_container.addWidget(self.command_input, stretch=1)

    def _apply_input_style(self):
        """Apply command input stylesheet using current _input_bg_* state."""
        size = getattr(self, '_font_size', 12)
        dark = getattr(self, '_is_dark_mode', False)
        text_color = "rgba(230, 230, 230, 255)" if dark else "rgba(0, 0, 0, 255)"
        r, g, b = self._input_bg_r, self._input_bg_g, self._input_bg_b
        a = self._input_bg_alpha
        self.command_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba({r}, {g}, {b}, {a});
                color: {text_color};
                border: none;
                border-radius: 3px; padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {size}px;
            }}
        """)

    def _animate_input_focus(self, focus_in: bool):
        """Animate command input background alpha on focus in/out."""
        if self._input_focus_anim is not None:
            self._input_focus_anim.stop()
            self._input_focus_anim.deleteLater()
            self._input_focus_anim = None

        target = self._input_bg_target_alpha if focus_in else 0
        start = self._input_bg_alpha
        if start == target:
            return

        steps = 12  # ~192ms at 16ms interval
        step = [0]

        def tick():
            step[0] += 1
            t = min(step[0] / steps, 1.0)
            t = t * t * (3.0 - 2.0 * t)  # smoothstep
            self._input_bg_alpha = int(start + (target - start) * t)
            self._apply_input_style()
            if step[0] >= steps:
                self._input_bg_alpha = target
                self._apply_input_style()
                if self._input_focus_anim is not None:
                    self._input_focus_anim.stop()
                    self._input_focus_anim.deleteLater()
                    self._input_focus_anim = None

        self._input_focus_anim = QTimer(self)
        self._input_focus_anim.timeout.connect(tick)
        self._input_focus_anim.start(16)

    def _set_input_bg_target(self, r, g, b, target_alpha):
        """Update the input background color targets (called by mode/theme changes)."""
        self._input_bg_r = r
        self._input_bg_g = g
        self._input_bg_b = b
        self._input_bg_target_alpha = target_alpha
        # If not focused, keep alpha at 0; if focused, snap to new target
        if self.command_input.hasFocus():
            self._input_bg_alpha = target_alpha
        else:
            self._input_bg_alpha = 0
        self._apply_input_style()

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

        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Key handling  (Enter = submit, Shift+Enter = newline)
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.command_input:
            if event.type() == event.Type.FocusIn:
                self._animate_input_focus(True)
            elif event.type() == event.Type.FocusOut:
                self._animate_input_focus(False)
            elif event.type() == QKeyEvent.Type.KeyPress:
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
            
            self.append_text(f"\n*** Terminal Mode {status} ***\n", color)
            
            # Update placeholder for visual feedback
            if self.terminal_mode:
                self.command_input.setPlaceholderText("SHELL MODE active - Type $ to exit")
            else:
                placeholder = f"[{self.connected_agent}] " if self.connected_agent else "Enter command..."
                self.command_input.setPlaceholderText(placeholder)
                
            self._update_input_style()
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
        self.append_text("║     MASTER AGENT — Initializing...      ║\n", self.C_MASTER)
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
        self.append_text("║     CODER AGENT — Initializing...       ║\n", self.C_INFO)
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
        self.append_text("║     GROK AV AGENT — Initializing...     ║\n", self.C_AV)
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
        self.append_text("║   GEMINI AV AGENT — Initializing...     ║\n", self.C_AV)
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
        )
        self._output_reader.new_data.connect(self._display_agent_text)
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
    # Receiving output (via $term/output filesystem writes)
    # ------------------------------------------------------------------

    def _on_fs_output(self, text: str):  # DEAD CODE — route-based output replaced this, kept for reference
        """
        Called by TerminalOutputFile.write() when data arrives via
        the $agent/OUTPUT → $term/output route.
        
        This runs on whatever thread the 9P server dispatches from,
        so we use QTimer.singleShot to bounce onto the Qt main thread.
        """
        from PySide6.QtCore import QMetaObject, Qt as _Qt, Q_ARG
        QMetaObject.invokeMethod(
            self, "_display_agent_text",
            _Qt.QueuedConnection,
            Q_ARG(str, text),
        )

    @Slot(str)
    def _display_agent_text(self, text: str):
        """Qt-thread-safe slot that actually appends agent text."""
        self.append_text(text, self.C_AGENT)

    @Slot()
    def _on_output_stream_done(self):
        """Called when the OutputStreamReader sees EOF (generation complete)."""
        self._response_pending = False

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

    @staticmethod
    def _parse_rgba(color_str):
        """Parse rgba(...)/rgb(...) or hex color strings into QColor."""
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

    def _insert_ansi_text(self, cursor: QTextCursor, text: str):
        """
        Parse ANSI escape sequences and insert colored plain text
        directly via QTextCursor.insertText + QTextCharFormat.

        This completely avoids insertHtml, so shell metacharacters
        like <, >, &, quotes etc. are never misinterpreted as HTML.
        """
        # Strip \r (PTY sends \r\n, Qt only needs \n)
        text = text.replace('\r', '')

        # Use active color scheme
        color_map = self._active_ansi_map
        default_color = self._dm_adjust_color(self._active_shell_output_color)

        # Start from the document's current char format so we inherit
        # the font family / size set via the QTextEdit stylesheet.
        base_fmt = cursor.charFormat()

        # Current format state
        fmt = QTextCharFormat(base_fmt)
        fmt.setForeground(self._parse_rgba(default_color))
        bold = False
        fg_color = None

        for segment in self._ANSI_RE.split(text):
            if not segment:
                continue

            # Is this an escape sequence?
            if segment.startswith('\x1b'):
                # SGR (Select Graphic Rendition)?
                m = re.fullmatch(r'\x1b\[([\d;]*)m', segment)
                if m:
                    codes = m.group(1).split(';') if m.group(1) else ['0']
                    for code in codes:
                        code = code.lstrip('0') or '0'  # '00' → '0', '01' → '1'
                        if code == '0':
                            # Reset
                            bold = False
                            fg_color = None
                            fmt = QTextCharFormat(base_fmt)
                            fmt.setForeground(self._parse_rgba(default_color))
                        elif code == '1':
                            bold = True
                            font = fmt.font()
                            font.setBold(True)
                            fmt.setFont(font)
                        elif code in color_map:
                            fg_color = self._dm_adjust_color(color_map[code])
                            fmt.setForeground(self._parse_rgba(fg_color))
                # All other escape sequences (OSC, CSI, etc.) are silently dropped
                continue

            # Plain text — insert with current format
            cursor.insertText(segment, fmt)

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
        QTimer.singleShot(0, self._scroll_to_bottom)

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
  /pop                   Detach terminal to floating window
  /dock                  Re-dock terminal into scene
  /restart               Restart shell (fresh env, re-seed vars)
  /setup                 Unmount & remount 9pfuse (LLMFS + Rio)
  /mount <IP!P> <n>      Mount 9P at /n/name via 9pfuse

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
        """Immediately set shadow to match the active color scheme."""
        shadow_target = self._proxy if self._proxy is not None else self
        current_effect = shadow_target.graphicsEffect()

        if not isinstance(current_effect, QGraphicsDropShadowEffect):
            return

        shadow = current_effect
        target_color = self._parse_rgba(self._active_shadow_color)
        start_color = shadow.color()

        steps = 30
        step = [0]

        if hasattr(self, '_shadow_color_timer'):
            self._shadow_color_timer.stop()
            self._shadow_color_timer.deleteLater()

        def lerp(a, b, t):
            return int(a + (b - a) * t)

        def tick():
            if step[0] <= steps:
                t = step[0] / steps
                t = t * t * (3.0 - 2.0 * t)
                r = lerp(start_color.red(), target_color.red(), t)
                g = lerp(start_color.green(), target_color.green(), t)
                b = lerp(start_color.blue(), target_color.blue(), t)
                a = lerp(start_color.alpha(), target_color.alpha(), t)
                shadow.setColor(QColor(r, g, b, a))
                step[0] += 1
            else:
                shadow.setColor(target_color)
                if hasattr(self, '_shadow_color_timer'):
                    self._shadow_color_timer.stop()
                    self._shadow_color_timer.deleteLater()
                    delattr(self, '_shadow_color_timer')

        self._shadow_color_timer = QTimer(self)
        self._shadow_color_timer.timeout.connect(tick)
        self._shadow_color_timer.start(16)

    def _open_color_picker(self):
        """Open the color scheme picker dialog."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QGridLayout, QColorDialog, QGroupBox, QScrollArea
        )
        from PySide6.QtGui import QPainter, QBrush, QPen
        from PySide6.QtCore import QSize

        terminal = self

        class ColorSwatch(QWidget):
            """Clickable color swatch that opens a QColorDialog."""

            def __init__(self, color_str, label, on_changed=None, parent=None):
                super().__init__(parent)
                self._color = terminal._parse_rgba(color_str)
                self._label = label
                self._on_changed = on_changed
                self.setFixedSize(48, 32)
                self.setCursor(Qt.PointingHandCursor)
                self.setToolTip(f"{label}\nClick to customize")

            def paintEvent(self, event):
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                # Checkerboard background to show alpha
                for row in range(0, self.height(), 6):
                    for col in range(0, self.width(), 6):
                        shade = QColor(200, 200, 200) if (row // 6 + col // 6) % 2 == 0 else QColor(255, 255, 255)
                        p.fillRect(col, row, 6, 6, shade)
                # The actual color on top
                p.setBrush(QBrush(self._color))
                p.setPen(QPen(QColor(100, 100, 100), 1.5))
                p.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 5, 5)
                p.end()

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    new_color = QColorDialog.getColor(
                        self._color, self, f"Pick {self._label} color",
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

        class ColorSchemeDialog(QDialog):
            def __init__(self, parent_terminal):
                # Find the top-level window for proper parenting
                parent_widget = parent_terminal
                while parent_widget.parent():
                    parent_widget = parent_widget.parent()
                super().__init__(parent_widget)
                self.terminal = parent_terminal
                self.setWindowTitle("Terminal Color Scheme")
                self.setMinimumWidth(420)
                self.setMinimumHeight(480)
                self._build_ui()

            def _build_ui(self):
                layout = QVBoxLayout(self)
                layout.setSpacing(12)

                # ---- Preset buttons ----
                presets_group = QGroupBox("Presets")
                presets_layout = QGridLayout()
                presets_layout.setSpacing(6)

                schemes = list(self.terminal.COLOR_SCHEMES.keys())
                for i, name in enumerate(schemes):
                    btn = QPushButton(name)
                    scheme = self.terminal.COLOR_SCHEMES[name]
                    accent = QColor(scheme["shell_echo"])
                    shadow_c = QColor(scheme["shadow"])

                    # Style button with scheme accent
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: rgba({accent.red()}, {accent.green()}, {accent.blue()}, 40);
                            border: 2px solid rgba({accent.red()}, {accent.green()}, {accent.blue()}, 120);
                            border-radius: 8px;
                            padding: 10px 16px;
                            font-family: Consolas, monospace;
                            font-weight: bold;
                            font-size: 13px;
                            color: rgba({accent.red()}, {accent.green()}, {accent.blue()}, 255);
                        }}
                        QPushButton:hover {{
                            background-color: rgba({accent.red()}, {accent.green()}, {accent.blue()}, 80);
                            border-color: rgba({accent.red()}, {accent.green()}, {accent.blue()}, 200);
                        }}
                        QPushButton:pressed {{
                            background-color: rgba({accent.red()}, {accent.green()}, {accent.blue()}, 120);
                        }}
                    """)
                    btn.setCursor(Qt.PointingHandCursor)
                    btn.clicked.connect(lambda checked=False, n=name: self._select_preset(n))
                    row, col = divmod(i, 3)
                    presets_layout.addWidget(btn, row, col)

                presets_group.setLayout(presets_layout)
                layout.addWidget(presets_group)

                # ---- Individual color swatches ----
                custom_group = QGroupBox("Customize Active Scheme")
                custom_layout = QGridLayout()
                custom_layout.setSpacing(8)

                active = self.terminal._active_scheme
                self._swatches = {}

                swatch_defs = [
                    ("shell_echo",   "Shell Echo ($)"),
                    ("shell_output", "Shell Output"),
                    ("success",      "Success"),
                    ("error",        "Error"),
                    ("info",         "Info"),
                    ("agent",        "Agent Output"),
                    ("shadow",       "Shadow"),
                ]

                for i, (key, label) in enumerate(swatch_defs):
                    lbl = QLabel(label)
                    lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
                    swatch = ColorSwatch(
                        active.get(key, "rgba(0,0,0,255)"), label,
                        on_changed=lambda k=key: self._on_swatch_changed(k)
                    )
                    self._swatches[key] = swatch
                    custom_layout.addWidget(lbl, i, 0)
                    custom_layout.addWidget(swatch, i, 1)

                # ANSI color row
                ansi_label = QLabel("ANSI Colors")
                ansi_label.setStyleSheet("font-family: Consolas, monospace; font-size: 12px; font-weight: bold;")
                custom_layout.addWidget(ansi_label, len(swatch_defs), 0, 1, 2)

                ansi_map = active.get("ansi_map", self.terminal._ANSI_COLOR_MAP)
                self._ansi_swatches = {}
                ansi_labels = {
                    '30': 'Blk', '31': 'Red', '32': 'Grn', '33': 'Yel',
                    '34': 'Blu', '35': 'Mag', '36': 'Cyn', '37': 'Wht',
                    '90': 'Blk+', '91': 'Red+', '92': 'Grn+', '93': 'Yel+',
                    '94': 'Blu+', '95': 'Mag+', '96': 'Cyn+', '97': 'Wht+',
                }

                ansi_row = QHBoxLayout()
                ansi_row.setSpacing(4)
                for code in ['30', '31', '32', '33', '34', '35', '36', '37']:
                    col_layout = QVBoxLayout()
                    col_layout.setSpacing(1)
                    sw = ColorSwatch(
                        ansi_map.get(code, '#000000'), ansi_labels[code],
                        on_changed=lambda c=code: self._on_ansi_changed(c)
                    )
                    sw.setFixedSize(36, 24)
                    self._ansi_swatches[code] = sw
                    lbl = QLabel(ansi_labels[code])
                    lbl.setStyleSheet("font-size: 9px; color: #888; font-family: Consolas, monospace;")
                    lbl.setAlignment(Qt.AlignCenter)
                    col_layout.addWidget(sw)
                    col_layout.addWidget(lbl)
                    ansi_row.addLayout(col_layout)
                ansi_row.addStretch()

                ansi_row2 = QHBoxLayout()
                ansi_row2.setSpacing(4)
                for code in ['90', '91', '92', '93', '94', '95', '96', '97']:
                    col_layout = QVBoxLayout()
                    col_layout.setSpacing(1)
                    sw = ColorSwatch(
                        ansi_map.get(code, '#000000'), ansi_labels[code],
                        on_changed=lambda c=code: self._on_ansi_changed(c)
                    )
                    sw.setFixedSize(36, 24)
                    self._ansi_swatches[code] = sw
                    lbl = QLabel(ansi_labels[code])
                    lbl.setStyleSheet("font-size: 9px; color: #888; font-family: Consolas, monospace;")
                    lbl.setAlignment(Qt.AlignCenter)
                    col_layout.addWidget(sw)
                    col_layout.addWidget(lbl)
                    ansi_row2.addLayout(col_layout)
                ansi_row2.addStretch()

                row_offset = len(swatch_defs) + 1
                ansi_widget1 = QWidget()
                ansi_widget1.setLayout(ansi_row)
                custom_layout.addWidget(ansi_widget1, row_offset, 0, 1, 2)
                ansi_widget2 = QWidget()
                ansi_widget2.setLayout(ansi_row2)
                custom_layout.addWidget(ansi_widget2, row_offset + 1, 0, 1, 2)

                custom_group.setLayout(custom_layout)
                layout.addWidget(custom_group)

                # ---- Preview + Close ----
                btn_layout = QHBoxLayout()
                self._preview_btn = QPushButton("Preview")
                self._preview_btn.clicked.connect(self._preview)
                self._preview_btn.setCursor(Qt.PointingHandCursor)
                self._apply_btn = QPushButton("Apply && Close")
                self._apply_btn.clicked.connect(self._apply_and_close)
                self._apply_btn.setCursor(Qt.PointingHandCursor)

                for btn in (self._preview_btn, self._apply_btn):
                    btn.setStyleSheet("""
                        QPushButton {
                            padding: 8px 20px; border-radius: 6px;
                            font-family: Consolas, monospace; font-size: 12px;
                            background-color: rgba(60, 60, 60, 200); color: white;
                            border: 1px solid rgba(120, 120, 120, 150);
                        }
                        QPushButton:hover { background-color: rgba(80, 80, 80, 220); }
                    """)

                btn_layout.addStretch()
                btn_layout.addWidget(self._preview_btn)
                btn_layout.addWidget(self._apply_btn)
                layout.addLayout(btn_layout)

                # Active scheme label
                self._scheme_label = QLabel(f"Active: {self.terminal._active_scheme_name}")
                self._scheme_label.setStyleSheet(
                    "font-family: Consolas, monospace; font-size: 11px; color: rgba(120,120,120,255);"
                )
                layout.addWidget(self._scheme_label)

            def _select_preset(self, name):
                """Load a preset into the swatches."""
                scheme = self.terminal.COLOR_SCHEMES[name]
                for key, swatch in self._swatches.items():
                    swatch.set_color(scheme.get(key, "rgba(0,0,0,255)"))
                ansi_map = scheme.get("ansi_map", {})
                for code, swatch in self._ansi_swatches.items():
                    swatch.set_color(ansi_map.get(code, '#000000'))
                self._scheme_label.setText(f"Active: {name}")
                # Immediately apply preset
                self.terminal._apply_color_scheme(name)

            def _on_swatch_changed(self, key):
                """A color swatch was changed — update active scheme."""
                swatch = self._swatches[key]
                self.terminal._active_scheme[key] = swatch.color_rgba()
                self.terminal._active_scheme_name = "Custom"
                self._scheme_label.setText("Active: Custom")
                # Update convenience colors (mode-aware)
                self.terminal.C_SHELL   = self.terminal._active_shell_echo_color
                self.terminal.C_AGENT   = self.terminal._dm_adjust_color(self.terminal._active_scheme["agent"])
                self.terminal.C_SUCCESS = self.terminal._dm_adjust_color(self.terminal._active_scheme["success"])
                self.terminal.C_ERROR   = self.terminal._dm_adjust_color(self.terminal._active_scheme["error"])
                self.terminal.C_INFO    = self.terminal._dm_adjust_color(self.terminal._active_scheme["info"])
                if key == "shadow":
                    self.terminal._set_shadow_to_scheme()

            def _on_ansi_changed(self, code):
                """An ANSI color swatch was changed."""
                swatch = self._ansi_swatches[code]
                if "ansi_map" not in self.terminal._active_scheme:
                    self.terminal._active_scheme["ansi_map"] = dict(self.terminal._ANSI_COLOR_MAP)
                self.terminal._active_scheme["ansi_map"][code] = swatch.color_hex()
                self.terminal._active_scheme_name = "Custom"
                self._scheme_label.setText("Active: Custom")

            def _preview(self):
                """Print a sample line using current scheme colors."""
                t = self.terminal
                t.append_text("$ ls -la /home\n", t._active_shell_echo_color)
                t.append_text("drwxr-xr-x  5 user user 4096 Jan  1 ", t._active_shell_output_color)
                t.append_text("documents\n", t._active_scheme.get("ansi_map", t._ANSI_COLOR_MAP).get('34', '#0000EE'))
                t.append_text("✓ Success message\n", t._active_scheme.get("success", t.C_SUCCESS))
                t.append_text("✗ Error message\n", t._active_scheme.get("error", t.C_ERROR))

            def _apply_and_close(self):
                """Apply current swatch state and close."""
                self.terminal._set_shadow_to_scheme()
                self.terminal.append_text(
                    f"Color scheme applied: {self.terminal._active_scheme_name}\n",
                    self.terminal._active_shell_echo_color
                )
                self.accept()

        dialog = ColorSchemeDialog(self)
        dialog.setStyleSheet("""
            QDialog {
                background-color: rgba(245, 245, 250, 240);
                border-radius: 10px;
            }
            QGroupBox {
                font-family: Consolas, monospace;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid rgba(180, 180, 180, 150);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)
        dialog.exec()

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
        """
        c = self._parse_rgba(color_str)
        r, g, b, a = c.red(), c.green(), c.blue(), c.alpha()
        lum = r * 0.299 + g * 0.587 + b * 0.114

        if getattr(self, '_is_dark_mode', False):
            if lum < 120:
                # Too dark for dark background — lighten
                factor = max(0.0, min(1.0, (120 - lum) / 120.0))
                boost = factor * 0.7
                nr = min(255, int(r + (255 - r) * boost))
                ng = min(255, int(g + (255 - g) * boost))
                nb = min(255, int(b + (255 - b) * boost))
                return f"rgba({nr}, {ng}, {nb}, {a})"
        else:
            if lum > 200:
                # Too bright for light background — darken
                factor = max(0.0, min(1.0, (lum - 200) / 55.0))
                dampen = factor * 0.6
                nr = max(0, int(r - r * dampen))
                ng = max(0, int(g - g * dampen))
                nb = max(0, int(b - b * dampen))
                return f"rgba({nr}, {ng}, {nb}, {a})"

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

        # Defer scroll to next event loop iteration
        QTimer.singleShot(0, self._scroll_to_bottom)

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

    # ------------------------------------------------------------------
    # Shadow animation
    # ------------------------------------------------------------------

    def animate_shadow_to_position(self):
        # When embedded in a QGraphicsProxyWidget, the shadow MUST be
        # applied on the proxy — QGraphicsEffect on an embedded widget
        # causes it to vanish.  When standalone (no proxy), apply on self.
        shadow_target = self._proxy if self._proxy is not None else self

        # Shadow color depends on dark mode
        if getattr(self, '_is_dark_mode', False):
            shadow_color = QColor(255, 255, 255, 160)
        else:
            shadow_color = QColor(0, 0, 0, 120)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(shadow_color)
        shadow.setOffset(0, 0)
        shadow_target.setGraphicsEffect(shadow)
        self.update()
        self.repaint()
        #QApplication.processEvents()

        steps = 30
        step = [0]
        target = 30

        def tick():
            if step[0] <= steps:
                p = 1 - pow(1 - step[0] / steps, 3)
                shadow.setOffset(QPointF(target * p, target * p))
                self.update()
                self.repaint()
                if self.parent():
                    self.parent().update()
                #QApplication.processEvents()
                step[0] += 1
            else:
                self.update()
                self.repaint()
                if self.parent():
                    self.parent().update()
                #QApplication.processEvents()
                if hasattr(self, '_shadow_timer'):
                    self._shadow_timer.stop()
                    self._shadow_timer.deleteLater()
                    delattr(self, '_shadow_timer')

        self._shadow_timer = QTimer(self)
        self._shadow_timer.timeout.connect(tick)
        self._shadow_timer.start(16)

    def animate_shadow_color(self, entering_terminal: bool):
        """
        Animate shadow color between base and active scheme shadow color.
        
        entering_terminal=True:  base → scheme shadow color
        entering_terminal=False: scheme shadow color → base
        
        The base color respects dark mode: white in dark, black in light.
        """
        shadow_target = self._proxy if self._proxy is not None else self

        # Grab existing shadow or create one
        current_effect = shadow_target.graphicsEffect()
        if not isinstance(current_effect, QGraphicsDropShadowEffect):
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(25)
            shadow.setOffset(QPointF(30, 30))
            shadow_target.setGraphicsEffect(shadow)
        else:
            shadow = current_effect

        # Base color depends on dark mode
        if getattr(self, '_is_dark_mode', False):
            base_color = QColor(255, 255, 255, 160)
        else:
            base_color = QColor(0, 0, 0, 120)

        scheme_color = self._parse_rgba(self._active_shadow_color)

        start_color = base_color if entering_terminal else scheme_color
        end_color = scheme_color if entering_terminal else base_color

        # Also animate blur radius for extra punch
        start_blur = 25.0 if entering_terminal else 45.0
        end_blur = 45.0 if entering_terminal else 25.0

        steps = 35
        step = [0]

        # Kill any existing color animation timer
        if hasattr(self, '_shadow_color_timer'):
            self._shadow_color_timer.stop()
            self._shadow_color_timer.deleteLater()

        def lerp(a, b, t):
            return int(a + (b - a) * t)

        def tick():
            if step[0] <= steps:
                # Ease-in-out cubic
                t = step[0] / steps
                t = t * t * (3.0 - 2.0 * t)

                r = lerp(start_color.red(), end_color.red(), t)
                g = lerp(start_color.green(), end_color.green(), t)
                b = lerp(start_color.blue(), end_color.blue(), t)
                a = lerp(start_color.alpha(), end_color.alpha(), t)
                shadow.setColor(QColor(r, g, b, a))

                blur = start_blur + (end_blur - start_blur) * t
                shadow.setBlurRadius(blur)

                step[0] += 1
            else:
                shadow.setColor(end_color)
                shadow.setBlurRadius(end_blur)
                if hasattr(self, '_shadow_color_timer'):
                    self._shadow_color_timer.stop()
                    self._shadow_color_timer.deleteLater()
                    delattr(self, '_shadow_color_timer')

        self._shadow_color_timer = QTimer(self)
        self._shadow_color_timer.timeout.connect(tick)
        self._shadow_color_timer.start(16)

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

    def set_dark_mode(self, enabled: bool, duration_steps: int = 50):  # DEAD CODE — unused, kept for external callers
        """Animate this terminal between light and dark mode.

        Transitions:
          - Frame border:  grey → white  (dark) or white → grey (light)
          - Text color in text_displays: black → white  or reverse
          - Command input styling: light bg → dark bg, text swap
          - Existing output text: recolor dark ↔ light inline char formats
          - Shadow color is handled globally by RioWindow._animate_all_shadows
        """
        self._is_dark_mode = enabled

        # ---- Update default text colors so NEW text uses the right color ----
        if enabled:
            self.C_DEFAULT = "rgba(230, 230, 230, 240)"
            self.C_USER    = "rgba(230, 230, 230, 240)"
        else:
            self.C_DEFAULT = "rgba(0, 0, 0, 230)"
            self.C_USER    = "rgba(0, 0, 0, 230)"

        # ---- Re-derive theme colors for the new mode ----
        # Shell echo/output are always black/white via the property.
        self.C_SHELL   = self._active_shell_echo_color
        self.C_AGENT   = self._dm_adjust_color(self._active_scheme["agent"])
        self.C_SUCCESS = self._dm_adjust_color(self._active_scheme["success"])
        self.C_ERROR   = self._dm_adjust_color(self._active_scheme["error"])
        self.C_INFO    = self._dm_adjust_color(self._active_scheme["info"])

        # ---- Target colors ----
        if enabled:
            # Dark mode targets
            border_color = "rgba(200, 200, 200, 220)"
            text_rgba = "rgba(230, 230, 230, 255)"
            input_bg = "rgba(40, 40, 50, 180)"
            input_text = "rgba(230, 230, 230, 255)"
            input_focus_border = "rgba(160, 160, 255, 200)"
            selection_bg = "rgba(100, 100, 255, 120)"
        else:
            # Light mode targets (original)
            border_color = "rgba(150, 150, 150, 200)"
            text_rgba = "rgba(0, 0, 0, 255)"
            input_bg = "rgba(255, 255, 255, 150)"
            input_text = "rgba(0, 0, 0, 255)"
            input_focus_border = "rgba(100, 100, 255, 200)"
            selection_bg = "rgba(100, 100, 255, 100)"

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

    def _animate_frame_dark_mode(self, target_border: str, steps: int):
        """Animate terminal_frame border color for dark/light mode."""
        import re as _re

        if hasattr(self, '_dm_frame_timer'):
            self._dm_frame_timer.stop()
            self._dm_frame_timer.deleteLater()

        # Parse current border from stylesheet
        current_style = self.terminal_frame.styleSheet()
        m = _re.search(
            r'border:\s*\d+px\s+solid\s+rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            current_style
        )
        if m:
            sr, sg, sb, sa = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        else:
            sr, sg, sb, sa = 150, 150, 150, 200

        # Parse current bg alpha
        m2 = _re.search(
            r'background-color:\s*rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            current_style
        )
        if m2:
            bg_r, bg_g, bg_b, bg_a = int(m2.group(1)), int(m2.group(2)), int(m2.group(3)), int(m2.group(4))
        else:
            bg_r, bg_g, bg_b, bg_a = 255, 255, 255, 0

        # Parse target border
        tc = self._parse_rgba(target_border)
        tr_, tg_, tb_, ta_ = tc.red(), tc.green(), tc.blue(), tc.alpha()

        step = [0]

        def tick():
            if step[0] <= steps:
                t = step[0] / steps
                t = t * t * (3.0 - 2.0 * t)
                r = int(sr + (tr_ - sr) * t)
                g = int(sg + (tg_ - sg) * t)
                b = int(sb + (tb_ - sb) * t)
                a = int(sa + (ta_ - sa) * t)
                self.terminal_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba({bg_r}, {bg_g}, {bg_b}, {bg_a});
                        border: 2px solid rgba({r}, {g}, {b}, {a});
                        border-radius: 5px;
                    }}
                """)
                step[0] += 1
            else:
                self.terminal_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba({bg_r}, {bg_g}, {bg_b}, {bg_a});
                        border: 2px solid rgba({tr_}, {tg_}, {tb_}, {ta_});
                        border-radius: 5px;
                    }}
                """)
                self._dm_frame_timer.stop()
                self._dm_frame_timer.deleteLater()
                delattr(self, '_dm_frame_timer')

        self._dm_frame_timer = QTimer(self)
        self._dm_frame_timer.timeout.connect(tick)
        self._dm_frame_timer.start(16)

    def _animate_text_dark_mode(self, target_rgba: str, selection_bg: str, steps: int):
        """Animate all text display colors for dark/light mode.

        Updates both:
          - The stylesheet (affects new text default color + selection)
          - Existing inline character formats: any text with near-black
            foreground gets animated to near-white (dark mode) and vice versa.
        """
        if hasattr(self, '_dm_text_timer'):
            self._dm_text_timer.stop()
            self._dm_text_timer.deleteLater()

        import re as _re

        # Parse current text color from first text display stylesheet
        if self.text_displays:
            style = self.text_displays[0].styleSheet()
            m = _re.search(r'color:\s*rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', style)
            if m:
                sr, sg, sb, sa = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            else:
                sr, sg, sb, sa = 0, 0, 0, 230
        else:
            sr, sg, sb, sa = 0, 0, 0, 230

        tc = self._parse_rgba(target_rgba)
        tr_, tg_, tb_, ta_ = tc.red(), tc.green(), tc.blue(), tc.alpha()

        size = getattr(self, '_font_size', 12)

        # ---- Collect ranges of "default-colored" text to recolor ----
        # Default text is near-black or near-white (low or high luminance).
        # We skip colored text (agent output, errors, etc.) that has
        # distinctive hue/saturation.
        entering_dark = self._is_dark_mode

        # Build per-text-display list of (cursor_start, cursor_end, start_color) ranges
        recolor_ranges = []  # list of (QTextEdit, [(start, end, QColor), ...])
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
                        # Determine if this is "default" colored text:
                        # near-black (entering dark) or near-white (leaving dark)
                        lum = fg.red() * 0.299 + fg.green() * 0.587 + fg.blue() * 0.114
                        is_default = False
                        if entering_dark and lum < 80:
                            is_default = True
                        elif not entering_dark and lum > 180:
                            is_default = True
                        if is_default:
                            ranges.append((
                                frag.position(),
                                frag.position() + frag.length(),
                                QColor(fg)
                            ))
                    it += 1
                block = block.next()
            if ranges:
                recolor_ranges.append((te, ranges))

        step = [0]

        def lerp(a, b, t):
            return int(a + (b - a) * t)

        def tick():
            if step[0] <= steps:
                t = step[0] / steps
                t = t * t * (3.0 - 2.0 * t)
                r = int(sr + (tr_ - sr) * t)
                g = int(sg + (tg_ - sg) * t)
                b = int(sb + (tb_ - sb) * t)
                a = int(sa + (ta_ - sa) * t)

                # Update stylesheet for all text displays
                css = f"""
                    QTextEdit {{
                        background-color: transparent; border: none;
                        color: rgba({r}, {g}, {b}, {a});
                        selection-background-color: {selection_bg};
                        font-family: 'Consolas', 'Monaco', monospace;
                        font-size: {size}px;
                    }}
                """
                for te in self.text_displays:
                    te.setStyleSheet(css)

                # Recolor existing inline text
                for te, ranges in recolor_ranges:
                    cursor = te.textCursor()
                    for start, end, orig_color in ranges:
                        cr = lerp(orig_color.red(), tr_, t)
                        cg = lerp(orig_color.green(), tg_, t)
                        cb = lerp(orig_color.blue(), tb_, t)
                        ca = lerp(orig_color.alpha(), ta_, t)
                        cursor.setPosition(start)
                        cursor.setPosition(end, QTextCursor.KeepAnchor)
                        fmt = QTextCharFormat()
                        fmt.setForeground(QColor(cr, cg, cb, ca))
                        cursor.mergeCharFormat(fmt)

                step[0] += 1
            else:
                css = f"""
                    QTextEdit {{
                        background-color: transparent; border: none;
                        color: rgba({tr_}, {tg_}, {tb_}, {ta_});
                        selection-background-color: {selection_bg};
                        font-family: 'Consolas', 'Monaco', monospace;
                        font-size: {size}px;
                    }}
                """
                for te in self.text_displays:
                    te.setStyleSheet(css)

                # Final recolor pass
                for te, ranges in recolor_ranges:
                    cursor = te.textCursor()
                    for start, end, _ in ranges:
                        cursor.setPosition(start)
                        cursor.setPosition(end, QTextCursor.KeepAnchor)
                        fmt = QTextCharFormat()
                        fmt.setForeground(QColor(tr_, tg_, tb_, ta_))
                        cursor.mergeCharFormat(fmt)

                self._dm_text_timer.stop()
                self._dm_text_timer.deleteLater()
                delattr(self, '_dm_text_timer')

        self._dm_text_timer = QTimer(self)
        self._dm_text_timer.timeout.connect(tick)
        self._dm_text_timer.start(16)

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
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setOffset(QPointF(30, 30))
        # Use scheme shadow color
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

        # ---- Reapply shadow on the proxy ----
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setOffset(QPointF(30, 30))
        shadow_color = self._parse_rgba(self._active_shadow_color)
        shadow.setColor(shadow_color)
        proxy.setGraphicsEffect(shadow)

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

        # Reset frame to fully transparent (scene-embedded state)
        self.terminal_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0);
                border: 2px solid rgba(150, 150, 150, 200);
                border-radius: 5px;
            }
        """)

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

        The border and border-radius are preserved; only the fill opacity
        changes — transparent (0) when embedded in the scene, opaque (~230)
        when popped out to an external window.
        """
        # Kill any running frame opacity animation
        if hasattr(self, '_frame_opacity_timer'):
            self._frame_opacity_timer.stop()
            self._frame_opacity_timer.deleteLater()

        # Determine correct background RGB + border for dark/light mode
        dark = getattr(self, '_is_dark_mode', False)
        if dark:
            r, g, b = 30, 30, 35
            border_css = "border: 2px solid rgba(200, 200, 200, 220);"
        else:
            r, g, b = 255, 255, 255
            border_css = "border: 2px solid rgba(150, 150, 150, 200);"

        # Parse current alpha from the stylesheet
        current_style = self.terminal_frame.styleSheet()
        import re as _re
        m = _re.search(r'background-color:\s*rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)', current_style)
        start_alpha = int(m.group(1)) if m else 0

        step = [0]

        def tick():
            if step[0] <= duration_steps:
                t = step[0] / duration_steps
                t = t * t * (3.0 - 2.0 * t)   # ease-in-out
                alpha = int(start_alpha + (target_alpha - start_alpha) * t)
                self.terminal_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba({r}, {g}, {b}, {alpha});
                        {border_css}
                        border-radius: 5px;
                    }}
                """)
                step[0] += 1
            else:
                self.terminal_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: rgba({r}, {g}, {b}, {target_alpha});
                        {border_css}
                        border-radius: 5px;
                    }}
                """)
                self._frame_opacity_timer.stop()
                self._frame_opacity_timer.deleteLater()
                delattr(self, '_frame_opacity_timer')

        self._frame_opacity_timer = QTimer(self)
        self._frame_opacity_timer.timeout.connect(tick)
        self._frame_opacity_timer.start(16)

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
        QApplication.processEvents()


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
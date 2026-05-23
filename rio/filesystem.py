"""
Rio Filesystem - With Output Capture and Version/Undo/Redo

Exposes the Rio display server as a 9P filesystem.

Enhancements:
- stdout: Queue file for print() output
- STDERR: Blocking errors file (uppercase = blocking read)
- CONTEXT: All successfully executed code (blocking read, at rio root)
- screen: Screenshot image (PNG)
- version: Unified version file with undo/redo (replaces versions/ dir)
- vars: Namespace inspection
"""

import asyncio
import json
import os
import re
import signal
import threading
import time
import sys
import io
import weakref
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr

from core.files import (
    SyntheticDir, SyntheticFile, StreamFile,
    CtlFile, CtlHandler
)

# Shell sandbox for LLM command validation
try:
    from .shell_sandbox import check_command as _sandbox_check
except ImportError:
    try:
        from shell_sandbox import check_command as _sandbox_check
    except ImportError:
        # Fallback: no sandbox (allow everything)
        def _sandbox_check(cmd):
            return True, None
from core.types import FidState

from .scene import SceneManager, SceneItem
from .parser import Executor, ExecutionContext, StreamingParser
from .context_file import create_smart_context_file_class
from .quick_file import QuickFile

# Cross-machine signals — exposed under /scene/signals/. The Signal
# factory and subscribe/unsubscribe entry points live in rio.signals;
# the parser injects them into the user namespace, so this import is
# only needed for the 9P dir + bus_getter.
try:
    from .signals_fs import SignalsDir
    from .signals import _global_bus as _rio_signals_bus
    _HAS_SIGNALS = True
except ImportError:
    _HAS_SIGNALS = False


class RioCtlHandler(CtlHandler):
    """Control handler for Rio root"""
    
    def __init__(self, rio: 'RioRoot'):
        self.rio = rio
    
    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "refresh":
            self.rio.scene_manager.refresh()
            return "refreshed"
        
        elif cmd == "clear":
            await self.rio.scene_manager.clear()
            return "cleared"
        
        elif cmd == "export":
            return self.rio.scene_manager.to_json()
        
        elif cmd == "import":
            if arg:
                self.rio.scene_manager.from_json(arg)
                return "imported"
            raise ValueError("Usage: import <json>")
        
        elif cmd == "size":
            if arg:
                parts = arg.split()
                if len(parts) >= 2:
                    self.rio.scene_manager.width = int(parts[0])
                    self.rio.scene_manager.height = int(parts[1])
                    return f"size set to {parts[0]}x{parts[1]}"
            return f"{self.rio.scene_manager.width} {self.rio.scene_manager.height}"
        
        elif cmd == "background":
            if arg:
                self.rio.scene_manager.background_color = arg
            return self.rio.scene_manager.refresh_background_color()
        
        elif cmd == "save":
            # NEW: Save state for crash recovery
            filepath = arg if arg else None
            if self.rio.scene_manager.save_state(filepath):
                return f"state saved to {filepath or '~/.rio_state.pkl'}"
            return "save failed"
        
        elif cmd == "load":
            # NEW: Load saved state
            filepath = arg if arg else None
            if self.rio.scene_manager.load_state(filepath):
                return f"state loaded from {filepath or '~/.rio_state.pkl'}"
            return "load failed (file not found or corrupt)"
        
        else:
            raise ValueError(f"Unknown command: {cmd}")
    
    async def get_status(self) -> bytes:
        sm = self.rio.scene_manager
        lines = [
            f"items {len(sm.list_parsed_items())}",
            f"size {sm.width} {sm.height}",
            f"background {sm.background_color}",
            f"qt_attached {sm._qt_scene is not None}",
            f"version {sm.versions.current_version}",
        ]
        return ("\n".join(lines) + "\n").encode()


class ScreenFile(SyntheticFile):
    """
    Screenshot file — returns a PNG image of the current scene.
    
    cat /n/rioa/screen > screenshot.png
    """
    
    def __init__(self, scene_manager: SceneManager, qt_objects: dict = None):
        super().__init__("screen")
        self.scene_manager = scene_manager
        self.qt_objects = qt_objects or {}
        self._read_cache = {}  # fid -> bytes
    
    def _capture_screenshot(self) -> bytes:
        """Render the current scene to a PNG image in memory."""
        sm = self.scene_manager
        qt_scene = sm._qt_scene
        
        if qt_scene is None:
            # Headless: return a minimal 1x1 PNG with scene info embedded
            # (or a proper render if we have offscreen support)
            try:
                from PySide6.QtGui import QImage, QPainter, QColor
                from PySide6.QtCore import QRectF, QBuffer, QIODevice
                
                img = QImage(sm.width, sm.height, QImage.Format_ARGB32)
                img.fill(QColor(sm.background_color))
                
                buf = QBuffer()
                buf.open(QIODevice.WriteOnly)
                img.save(buf, "PNG")
                return bytes(buf.data())
            except ImportError:
                return b""
        
        try:
            from PySide6.QtGui import QImage, QPainter, QColor
            from PySide6.QtCore import QRectF, QBuffer, QIODevice
            
            # Render scene to image
            img = QImage(sm.width, sm.height, QImage.Format_ARGB32)
            img.fill(QColor(sm.background_color))
            
            painter = QPainter(img)
            painter.setRenderHint(QPainter.Antialiasing)
            qt_scene.render(painter, QRectF(0, 0, sm.width, sm.height),
                           QRectF(0, 0, sm.width, sm.height))
            painter.end()
            
            # Encode to PNG in memory
            buf = QBuffer()
            buf.open(QIODevice.WriteOnly)
            img.save(buf, "PNG")
            return bytes(buf.data())
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            return b""
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        # Cache the screenshot for this fid on first read
        if fid.fid not in self._read_cache:
            self._read_cache[fid.fid] = self._capture_screenshot()
        
        data = self._read_cache[fid.fid]
        return data[offset:offset + count]
    
    def clunk(self, fid: FidState):
        """Free cached screenshot on file close."""
        self._read_cache.pop(fid.fid, None)
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Screen file is read-only")


class SceneCtlHandler(CtlHandler):
    """Control handler for scene"""
    
    def __init__(self, scene_manager: SceneManager):
        self.scene_manager = scene_manager
    
    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        
        if cmd == "clear":
            await self.scene_manager.clear()
            return "cleared"
        
        elif cmd == "refresh":
            self.scene_manager.refresh()
            return "refreshed"
        
        elif cmd == "export":
            return self.scene_manager.to_json()
        
        elif cmd == "undo":
            snap = self.scene_manager.undo()
            if snap:
                return f"restored to version {snap.version}"
            return "nothing to undo"
        
        elif cmd == "redo":
            snap = self.scene_manager.redo()
            if snap:
                return f"restored to version {snap.version}"
            return "nothing to redo"
        
        elif cmd == "goto":
            arg = parts[1] if len(parts) > 1 else ""
            if arg:
                version = int(arg)
                snap = self.scene_manager.goto_version(version)
                if snap:
                    return f"restored to version {snap.version}"
                return f"version {version} not found"
            raise ValueError("Usage: goto <version_number>")
        
        elif cmd == "snapshot":
            arg = parts[1] if len(parts) > 1 else ""
            snap = self.scene_manager.take_snapshot(label=arg)
            return f"snapshot saved as version {snap.version}"
        
        else:
            raise ValueError(f"Unknown command: {cmd}")
    
    async def get_status(self) -> bytes:
        sm = self.scene_manager
        vm = sm.versions
        lines = [
            f"items {len(sm.list_parsed_items())}",
            f"version {vm.current_version}",
            f"versions {len(vm.snapshots)}",
            f"can_undo {vm.can_undo()}",
            f"can_redo {vm.can_redo()}",
        ]
        return ("\n".join(lines) + "\n").encode()


# ============================================================
# Output Capture Files
# ============================================================

class StdoutFile(SyntheticFile):
    """
    Blocking stdout file - captures print() output from code execution.
    
    STATE-AWARE BLOCKING (enables `while true; do cat $scene/stdout; done`):
    
    1. IDLE: read() returns b"" immediately (non-blocking)
    2. ACTIVE: post() has been called, chunks are accumulating
    3. READY: mark_ready() fires, read() returns content
    4. CONSUMED: read() returns b"" (EOF — cat exits)
    5. Next read at offset 0: rearms, returns to IDLE
    
    The IDLE → b"" return is critical: it prevents ls, tab-completion,
    and stat from hanging when no execution is in progress.
    
    The 9P server dispatches each message as a concurrent asyncio task,
    so a blocked read() never prevents writes to other files.
    """
    def __init__(self):
        super().__init__("stdout")
        self._chunks: List[str] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._active = False      # True once post() is called (execution in flight)
        self._lock = asyncio.Lock()
    
    async def post(self, data: bytes):
        """Add data to the output buffer"""
        async with self._lock:
            self._active = True
            self._chunks.append(data.decode('utf-8'))
    
    def mark_ready(self):
        """Mark content as ready for reading (called when execution completes)"""
        self._active = True
        self._content_ready.set()
    
    def clear(self):
        """Clear all content and reset state for next execution"""
        self._chunks.clear()
        self._content_ready.clear()
        self._content_consumed = False
        self._active = False
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """
        State-aware read — non-blocking when idle.
        
        Returns b"" immediately if no execution is in flight,
        preventing ls / tab-completion / stat from hanging.
        Only blocks when content is actively being produced.
        """
        # If consumed and back at offset 0 → rearm for next execution
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()
                    self._chunks.clear()
                    self._active = False
        
        # IDLE: nothing pending → return empty immediately (non-blocking)
        if not self._active and not self._content_ready.is_set():
            return b""
        
        # ACTIVE/READY: block until execution completes
        await self._content_ready.wait()
        
        async with self._lock:
            content = "".join(self._chunks)
            data = content.encode()
            
            chunk = data[offset:offset + count]
            
            if offset + len(chunk) >= len(data):
                self._content_consumed = True
            
            return chunk
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Cannot write to stdout. Content is auto-generated from code execution.")


class StderrFile(SyntheticFile):
    """
    Blocking STDERR file - captures errors from code execution.
    
    Uppercase name indicates blocking read semantics.
    
    TRUE BLOCKING (enables `while true; do cat $scene/STDERR; done`):
    
    1. WAITING: read() blocks until mark_ready() fires
    2. READY: read() returns content
    3. CONSUMED: read() returns b"" (EOF — cat exits)
    4. Next read at offset 0: rearms, blocks again at step 1
    
    Unlike the IDLE-aware pattern, this always blocks when there is no
    content — making it safe for infinite routing loops. The 9P server
    dispatches each message as a concurrent asyncio task, so a blocked
    read() never prevents writes to other files.
    """
    def __init__(self):
        super().__init__("STDERR")
        self._chunks: List[str] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()
    
    async def post(self, data: bytes):
        """Add data to the output buffer"""
        async with self._lock:
            self._chunks.append(data.decode('utf-8'))
    
    def mark_ready(self):
        """Mark content as ready for reading (called when execution completes)"""
        self._content_ready.set()
    
    def clear(self):
        """Clear all content and reset state for next execution"""
        self._chunks.clear()
        self._content_ready.clear()
        self._content_consumed = False
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """
        True blocking read — blocks until content is ready.
        
        Always blocks when no content is available, enabling infinite
        routing with `while true; do cat $scene/stderr; done`.
        """
        # If consumed and back at offset 0 → rearm for next execution
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()
                    self._chunks.clear()
        
        # Block until content is ready
        await self._content_ready.wait()
        
        async with self._lock:
            content = "".join(self._chunks)
            data = content.encode()
            
            chunk = data[offset:offset + count]
            
            if offset + len(chunk) >= len(data):
                self._content_consumed = True
            
            return chunk
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Cannot write to stderr. Content is auto-generated from code execution.")


class VarsFile(SyntheticFile):
    """Namespace variables file. Shows all variables as JSON."""
    def __init__(self, executor):
        super().__init__("vars")
        self.executor = executor
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        namespace = self.executor.context.get_namespace()
        vars_dict = {}
        for key, value in namespace.items():
            if key.startswith('_'):
                continue
            if key in ('__builtins__', '__name__'):
                continue
            if hasattr(value, '__file__'):
                continue
            try:
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    vars_dict[key] = value
                else:
                    vars_dict[key] = f"<{type(value).__name__} object>"
            except:
                vars_dict[key] = "<unknown>"
        text = json.dumps(vars_dict, indent=2)
        return (text + "\n").encode()[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Vars file is read-only")


# ============================================================
# Enhanced Parse File with Output Capture
# ============================================================

class ParseFile(SyntheticFile):
    """
    Enhanced parse file with stdout/STDERR capture.
    Executes code and captures output. Triggers a version snapshot.
    On success, appends the code to the CONTEXT file.
    """
    
    def __init__(self, scene_manager, executor, stdout_file, stderr_file, context_file=None):
        super().__init__("parse")
        self.scene_manager = scene_manager
        self.executor = executor
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file
        self.context_file = context_file
        self._fid_parsers = {}
        self._fid_decoders = {}  # incremental UTF-8 decoders per fid
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        # If this fid has un-flushed bytes mid-write, surface that —
        # returning CONTEXT here would be misleading since it doesn't
        # yet reflect the in-progress block.
        parser = self._fid_parsers.get(fid.fid)
        if parser and parser.has_content():
            data = b"buffering...\n"
            return data[offset:offset + count]

        # One-shot, non-blocking view of the compacted CONTEXT.
        # Same content as `cat /n/<machine>/CONTEXT`, but reads here
        # never block waiting for new code to land.  Empty CONTEXT
        # → empty read (immediate EOF).
        if self.context_file is None:
            return b""
        content = self.context_file.get_all_code()
        if not content:
            return b""
        data = content.encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            if fid.fid not in self._fid_parsers:
                self._fid_parsers[fid.fid] = StreamingParser()
            if fid.fid not in self._fid_decoders:
                import codecs
                self._fid_decoders[fid.fid] = codecs.getincrementaldecoder('utf-8')('strict')
            
            parser = self._fid_parsers[fid.fid]
            decoder = self._fid_decoders[fid.fid]
            text = decoder.decode(data, False)  # False = more data may follow
            if text:
                parser.feed(text)
            
            buffer = parser.get_buffer()
            print(f"BUFFER [fid={fid.fid}] ({len(buffer)} chars): {buffer[:50]}...")
            
            return len(data)
        except UnicodeDecodeError:
            await self.stderr_file.post(b"Error: Invalid UTF-8\n")
            raise ValueError("Invalid UTF-8")
    
    def clunk(self, fid: FidState):
        """Execute accumulated code when file is closed"""
        # Flush any remaining bytes from the incremental decoder
        decoder = self._fid_decoders.pop(fid.fid, None)
        if decoder:
            try:
                tail = decoder.decode(b'', True)  # True = final, flush remaining
                if tail:
                    parser = self._fid_parsers.get(fid.fid)
                    if parser:
                        parser.feed(tail)
            except UnicodeDecodeError:
                pass  # truncated trailing bytes — nothing we can do

        parser = self._fid_parsers.get(fid.fid)
        if not parser:
            print(f"WARNING: No parser found for fid={fid.fid}")
            return
        
        code = parser.flush()
        del self._fid_parsers[fid.fid]
        
        if code:
            print(f"\n{'='*60}")
            print(f"EXECUTING on file close [fid={fid.fid}]:")
            print(f"{'='*60}")
            print(code)
            print(f"{'='*60}\n")
            
            asyncio.create_task(self._execute_code(code))

    def dispatch(self, code: str):
        """
        Internal shortcut — execute code directly without going through
        the 9P write/clunk cycle. Used by QuickFile and other internal
        callers that generate code programmatically.
        """
        if code and code.strip():
            asyncio.create_task(self._execute_code(code))

    async def _execute_code(self, code: str):
        """Execute code with output capture"""
        # Clear previous execution state
        self.stdout_file.clear()
        self.stderr_file.clear()
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = await self.executor.execute(code)
        
        stdout_text = stdout_capture.getvalue()
        if stdout_text:
            await self.stdout_file.post(stdout_text.encode())
        
        stderr_text = stderr_capture.getvalue()
        if stderr_text:
            await self.stderr_file.post(stderr_text.encode())
        
        if result.success:
            print(f"✓ Execution successful")
            
            if result.result is not None:
                result_msg = f"→ {repr(result.result)}\n"
                await self.stdout_file.post(result_msg.encode())
            
            if result.widgets_created:
                msg = f"✓ Created {len(result.widgets_created)} widget(s)\n"
                print(msg)
                await self.stdout_file.post(msg.encode())
            
            if result.items_registered:
                msg = f"✓ Registered {len(result.items_registered)} scene item(s)\n"
                print(msg)
                await self.stdout_file.post(msg.encode())
            
            # Report version
            vm = self.scene_manager.versions
            msg = f"✓ Version {vm.current_version}\n"
            await self.stdout_file.post(msg.encode())
            
            # Append successfully executed code to CONTEXT
            if self.context_file:
                self.context_file.append_code(code)
        else:
            print(f"✗ Execution failed:")
            print(result.error)
            await self.stderr_file.post(result.error.encode())
        
        # Mark outputs as ready (unblocks waiting reads)
        self.stdout_file.mark_ready()
        self.stderr_file.mark_ready()


# ============================================================
# Version File (replaces versions/ directory)
# ============================================================

class VersionFile(SyntheticFile):
    """
    Unified version file — replaces the versions/ directory.
    
    Read (cat):
        Returns version list with current version indicated.
    
    Write:
        echo undo > version    → undo one step
        echo redo > version    → redo one step
    
    Usage:
        cat /n/rio/scene/version
        echo undo > /n/rio/scene/version
        echo redo > /n/rio/scene/version
    """
    
    def __init__(self, scene_manager: SceneManager):
        super().__init__("version")
        self.scene_manager = scene_manager
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        vm = self.scene_manager.versions
        current = vm.current_version
        
        lines = []
        for snap_dict in vm.list_versions():
            ver = snap_dict.get("version", 0)
            label = snap_dict.get("label", "")
            item_count = snap_dict.get("item_count", 0)
            marker = " *" if ver == current else ""
            # Truncate label for readability
            if len(label) > 60:
                label = label[:57] + "..."
            lines.append(f"{ver}\t{item_count} items\t{label}{marker}")
        
        if not lines:
            lines.append("(no versions)")
        
        lines.append(f"\ncurrent {current}")
        lines.append(f"can_undo {vm.can_undo()}")
        lines.append(f"can_redo {vm.can_redo()}")
        
        text = "\n".join(lines) + "\n"
        return text.encode()[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        cmd = data.decode('utf-8').strip().lower()
        
        if cmd == "undo":
            snap = self.scene_manager.undo()
            if snap:
                print(f"✓ Undo -> version {snap.version}")
            else:
                print("✗ Nothing to undo")
        elif cmd == "redo":
            snap = self.scene_manager.redo()
            if snap:
                print(f"✓ Redo -> version {snap.version}")
            else:
                print("✗ Nothing to redo")
        else:
            # Try as version number (goto)
            try:
                version = int(cmd)
                snap = self.scene_manager.goto_version(version)
                if snap:
                    print(f"✓ Goto -> version {snap.version}")
                else:
                    print(f"✗ Version {version} not found")
            except ValueError:
                raise ValueError(f"Unknown version command: {cmd}. Use: undo, redo, or a version number")
        
        return len(data)


# ============================================================
# CONTEXT File (blocking, all successful code — smart compaction)
# ============================================================

# SmartContextFile inherits from SyntheticFile and provides the same
# interface as the old ContextFile (append_code, get_all_code, read,
# write) but returns compacted code on read — eliminating superseded
# variable definitions, redundant imports, and stale side effects.
ContextFile = create_smart_context_file_class(SyntheticFile)


# ============================================================
# Terminal Filesystem Exposure
# ============================================================

class TerminalCtlHandler(CtlHandler):
    """
    Control handler for a terminal exposed in the filesystem.
    
    Writing a command to /n/rioa/terms/<term_id>/ctl executes it
    in the terminal's persistent shell (like typing $ <cmd>).
    """
    
    def __init__(self, terminal_ref):
        # terminal_ref is a weakref or callback to the TerminalWidget
        self._terminal_ref = terminal_ref
    
    async def execute(self, command: str) -> Optional[str]:
        terminal = self._terminal_ref()
        if terminal is None:
            raise IOError("Terminal no longer exists")
        
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "font":
            if not arg:
                size = getattr(terminal, '_font_size', 12)
                return f"font {size}"
            try:
                size = int(arg)
            except ValueError:
                raise ValueError(f"Usage: font <size>  (got: {arg})")
            # Schedule on Qt main thread
            from PySide6.QtCore import QMetaObject, Qt as _Qt, Q_ARG
            QMetaObject.invokeMethod(
                terminal, "set_font_size",
                _Qt.QueuedConnection,
                Q_ARG(int, size),
            )
            return f"font {size}"
        
        # Default: execute as shell command
        try:
            cmd_str = command if command.endswith('\n') else command + '\n'
            os.write(terminal.shell_fd, cmd_str.encode('utf-8'))
        except OSError as e:
            raise IOError(f"Shell write failed: {e}")
        return f"executed: {command[:60]}"
    
    async def get_status(self) -> bytes:
        terminal = self._terminal_ref()
        if terminal is None:
            return b"status disconnected\n"
        lines = [
            f"term_id {terminal.term_id}",
            f"agent {terminal.connected_agent or 'none'}",
            f"master {terminal._master_active}",
            f"font {getattr(terminal, '_font_size', 12)}",
        ]
        return ("\n".join(lines) + "\n").encode()


class TerminalStdoutFile(SyntheticFile):
    """
    Blocking stdout file for terminal shell output.
    
    /n/rioa/terms/<term_id>/stdout
    
    Read-only.  Captures PTY output and delivers it to readers with
    blocking semantics (enables `while true; do cat $term/stdout; done`).
    
    STATE-AWARE BLOCKING:
    
    1. WAITING: read() blocks until mark_ready() fires
    2. READY: read() returns content
    3. CONSUMED: read() returns b"" (EOF — cat exits)
    4. Next read at offset 0: rearms, blocks again at step 1
    
    Uses threading primitives (not asyncio) because callers span
    the 9P asyncio server, the Qt main thread, and the shell reader.
    """
    
    # Strip ALL ANSI escape sequences from captured output so that
    # content fed back to agents / read via `cat $term/stdout` is
    # clean plain text (no color codes, OSC window titles, etc.).
    _ANSI_STRIP_RE = re.compile(
        r'\x1b\].*?(?:\x07|\x1b\\)'       # OSC (window title etc.)
        r'|\x1b\[[\d;]*[A-Za-z]'          # CSI (colors, cursor, etc.)
        r'|\x1b[\x20-\x7E]'               # two-byte escapes
        r'|\x1b'                           # stray ESC
        r'|\r'                             # carriage returns (\r\n → \n)
    )
    
    def __init__(self):
        super().__init__("stdout")
        self._output_chunks: list = []
        self._output_ready = threading.Event()
        self._lock = threading.Lock()
        self._capturing = False
        self._content_consumed = False
    
    def capture_output(self, text: str):
        """Called from Qt thread (_on_shell_output) to feed PTY data.
        
        Strips ANSI escape sequences and carriage returns so that
        readers (including the master agent) get clean plain text.
        """
        if self._capturing:
            clean = self._ANSI_STRIP_RE.sub('', text)
            if clean:
                with self._lock:
                    self._output_chunks.append(clean)
    
    def mark_ready(self):
        """Debounce timer fired: output settled, unblock readers.
        
        Only signals if there are actually captured chunks — prevents
        spurious wakeups from shell startup or prompt-only output
        that would cause the first read to return empty.
        """
        self._capturing = False
        with self._lock:
            if self._output_chunks:
                self._output_ready.set()
    
    def start_capture(self):
        """Begin capturing output for the next command."""
        with self._lock:
            self._output_chunks.clear()
            self._output_ready.clear()
            self._content_consumed = False
            self._capturing = True
    
    def feed_error(self, message: str):
        """Feed an error message (e.g. sandbox rejection) as readable output."""
        with self._lock:
            self._output_chunks.clear()
            self._output_chunks.append(message)
            self._content_consumed = False
            self._capturing = False
        self._output_ready.set()
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """
        Blocking read — always blocks until content is ready,
        including the very first read before any command is run.
        
        1. WAITING: block until mark_ready() fires
        2. READY: return content
        3. CONSUMED: return b"" (EOF — cat exits)
        4. Next read at offset 0: rearm, block again at step 1
        """
        # If consumed and back at offset 0 → rearm for next command
        if offset == 0 and self._content_consumed:
            with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._output_chunks.clear()
                    self._output_ready.clear()
        
        # Block until output is ready (in executor so asyncio isn't frozen)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._output_ready.wait)
        
        with self._lock:
            content = "".join(self._output_chunks)
            data = content.encode('utf-8')
            chunk = data[offset:offset + count]
            
            # If this read reaches end of content, mark consumed.
            if offset + len(chunk) >= len(data):
                self._content_consumed = True
            
            return chunk
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Cannot write to stdout. Use stdin to send commands.")


class TerminalStdinFile(SyntheticFile):
    """
    Write-only stdin file for sending commands to the terminal shell.
    
    /n/rioa/terms/<term_id>/stdin
    
    Write: executes the command in the terminal's persistent shell
           and arms the sibling stdout file for output capture.
    Read:  returns usage hint.
    
    SANDBOX: All commands are validated before execution.
      - Read anywhere is allowed
      - Writes only under /n/
      - Destructive ops (rm, dd, etc.) always blocked
    """
    
    def __init__(self, terminal_ref, stdout_file: TerminalStdoutFile):
        super().__init__("stdin")
        self._terminal_ref = terminal_ref
        self._stdout_file = stdout_file
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return b"Write shell commands here. Output appears in stdout.\n"
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        terminal = self._terminal_ref()
        if terminal is None:
            raise IOError("Terminal no longer exists")
        
        command = data.decode('utf-8').strip()
        if command:
            # ── Sandbox gate ────────────────────────────────────────
            ok, reason = _sandbox_check(command)
            if not ok:
                # Show blocked command in terminal
                try:
                    from PySide6.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(
                        terminal, "append_text",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"⛔ blocked: {command}\n"),
                        Q_ARG(str, "rgba(255, 80, 80, 255)")
                    )
                    QMetaObject.invokeMethod(
                        terminal, "append_text",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"   reason: {reason}\n"),
                        Q_ARG(str, "rgba(255, 80, 80, 255)")
                    )
                except Exception:
                    pass
                
                # Feed rejection to stdout so the LLM sees it
                self._stdout_file.feed_error(f"SANDBOX BLOCKED: {reason}\n")
                return len(data)
            # ── End sandbox gate ────────────────────────────────────

            # Show command with visual indicator in terminal
            try:
                from PySide6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    terminal, "append_text",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"⚡ {command}\n"),
                    Q_ARG(str, "rgba(200, 170, 80, 255)")
                )
            except Exception:
                pass  # Non-critical: visual feedback only
            
            # Arm stdout for capture
            self._stdout_file.start_capture()
            
            try:
                cmd = command if command.endswith('\n') else command + '\n'
                os.write(terminal.shell_fd, cmd.encode('utf-8'))
            except OSError as e:
                self._stdout_file._capturing = False
                raise IOError(f"Shell write failed: {e}")
            
            # Detect agent creation: echo 'new <n>' > .../ctl
            import re as _re
            m = _re.search(r"echo\s+['\"]?new\s+(\w+)", command)
            if m and hasattr(terminal, 'known_agents'):
                new_agent = m.group(1)
                terminal.known_agents.add(new_agent)
        
        return len(data)


class TerminalInputFile(SyntheticFile):
    """
    Write text to send as a prompt to the terminal's connected agent.
    
    /n/rioa/terms/<term_id>/input
    """
    
    def __init__(self, terminal_ref):
        super().__init__("input")
        self._terminal_ref = terminal_ref
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return b"Write prompts here to send to the terminal's connected agent\n"
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        terminal = self._terminal_ref()
        if terminal is None:
            raise IOError("Terminal no longer exists")
        
        prompt = data.decode('utf-8').strip()
        if prompt and terminal.connected_agent:
            # Write directly to the agent's input file (thread-safe file I/O)
            import os as _os
            input_path = _os.path.join(
                terminal.llmfs_mount, "agents",
                terminal.connected_agent, "input"
            )
            try:
                with open(input_path, 'w') as f:
                    f.write(prompt)
            except Exception as e:
                raise IOError(f"Agent input write failed: {e}")
        return len(data)


class TerminalOutputFile(SyntheticFile):
    """
    Writable output file — writing to $term/output prints text
    in the terminal widget's output area.
    
    /n/rioa/terms/<term_id>/output
    
    Write path (agent → terminal display):
        Anything written here is forwarded to the terminal widget's
        append_text via the _on_fs_output callback.  This is the
        standard destination for agent output routes:
            $agent/output → $term/output
    
    Read path (blocking monitoring tap):
        STATE-AWARE BLOCKING (enables `while true; do cat $term/output; done`):
        
        1. WAITING: read() blocks until mark_ready() fires
        2. READY: read() returns content
        3. CONSUMED: read() returns b"" (EOF — cat exits)
        4. Next read at offset 0: rearms, blocks again at step 1
        
        Always blocks — including the very first read before any
        output has arrived.
        
        The 9P server dispatches each message as a concurrent asyncio task,
        so a blocked read() never prevents writes to other files.
    """
    
    def __init__(self, terminal_ref):
        super().__init__("output")
        self._terminal_ref = terminal_ref
        # Blocking read buffer
        self._chunks: list = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Write text to the terminal's display area."""
        terminal = self._terminal_ref()
        if terminal is None:
            raise IOError("Terminal no longer exists")
        
        text = data.decode('utf-8', errors='replace')
        if text:
            # Display in the terminal widget (must run on Qt thread).
            # The fid-derived source key gives each open file handle its
            # own fence parser, so two concurrent producers writing fenced
            # blocks (e.g. two agents both routed to $term/output) cannot
            # corrupt each other's state. Older terminals that don't
            # accept the source_key kwarg fall back to the unkeyed call.
            try:
                terminal._on_fs_output(text, source_key=f"fs:{fid.fid}")
            except TypeError:
                terminal._on_fs_output(text)
            # Also feed the blocking monitoring tap
            await self.post(data)
        
        return len(data)
    
    async def post(self, data: bytes):
        """Accumulate data into the monitoring buffer.
        
        Does NOT signal readiness — call mark_ready() when a logical
        batch is complete (e.g. after agent generation finishes or
        a shell command's output settles).
        """
        async with self._lock:
            self._chunks.append(data.decode('utf-8', errors='replace'))
    
    def mark_ready(self):
        """Signal that accumulated content is ready for reading.
        
        Only signals if there are actually chunks — prevents spurious
        wakeups that would cause reads to return empty.
        """
        # No lock needed: _chunks is only mutated under _lock in post(),
        # and a momentary stale read is harmless (worst case: we skip
        # signalling and the next mark_ready() picks it up).
        if self._chunks:
            self._content_ready.set()
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """
        Blocking read — always blocks until content is ready,
        including the very first read before any output arrives.
        """
        # If consumed and back at offset 0 → rearm for next batch
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()
                    self._chunks.clear()
        
        # Block until content is ready
        await self._content_ready.wait()
        
        async with self._lock:
            content = "".join(self._chunks)
            data = content.encode('utf-8')
            
            chunk = data[offset:offset + count]
            
            if offset + len(chunk) >= len(data):
                self._content_consumed = True
            
            return chunk
    
    def clunk(self, fid: FidState):
        """Finalize the per-fid stream when this file handle closes.
        
        If a writer disconnected mid-fence, this lets the terminal widget
        force-close the inline widget so it stops showing as 'streaming'.
        Best-effort: if the terminal is gone or doesn't expose the hook,
        we silently ignore.
        """
        terminal = self._terminal_ref()
        if terminal is None:
            return
        # Bounce to Qt main thread; the router itself isn't thread-safe.
        try:
            from PySide6.QtCore import QMetaObject, Qt as _Qt, Q_ARG
            QMetaObject.invokeMethod(
                terminal, "_route_stream_eof",
                _Qt.QueuedConnection,
                Q_ARG(str, f"fs:{fid.fid}"),
            )
        except Exception:
            pass


class TerminalInterruptFile(SyntheticFile):
    """
    Write anything to this file to send SIGINT to the terminal's
    shell process group — interrupts any running command.
    
    /n/rioa/terms/<term_id>/interrupt
    
    Also triggered by the Delete key in the terminal widget.
    """
    
    def __init__(self, terminal_ref):
        super().__init__("interrupt")
        self._terminal_ref = terminal_ref
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return b"Write anything to send SIGINT to the shell process group\n"
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        terminal = self._terminal_ref()
        if terminal is None:
            raise IOError("Terminal no longer exists")
        
        # Send SIGINT to the shell's process group
        try:
            if terminal.shell_process and terminal.shell_process.poll() is None:
                pgid = os.getpgid(terminal.shell_process.pid)
                os.killpg(pgid, signal.SIGINT)
        except (OSError, ProcessLookupError):
            pass
        return len(data)


class TerminalInlineFile(SyntheticFile):
    """
    Write here to render a media widget inline in the terminal output.
    
    /n/rioa/terms/<term_id>/inline
    
    Three input formats are auto-detected:
    
    1. **Path** (one line, valid existing file path):
    
           echo /tmp/result.png  > $inline
           echo /tmp/anim.gif    > $inline   # auto-loops
           echo /tmp/song.mp3    > $inline   # mini player
           echo /tmp/clip.mp4    > $inline   # video player
           echo /tmp/doc.pdf     > $inline   # PDF scroll view
           echo /tmp/mesh.obj    > $inline   # OpenGL 3D viewer
           echo /tmp/widget.py   > $inline   # exec PySide6 source
           echo /tmp/report.html > $inline   # rich-text HTML
    
       The file is read by this filesystem and its type is inferred from
       extension and magic bytes, then dispatched to the appropriate
       inline widget.
    
    2. **Raw content** (binary or multi-line):
    
           cat /tmp/result.png > $inline   # PNG/JPEG/GIF/WEBP/PDF sniffed
           cat /tmp/report.html > $inline  # HTML/SVG sniffed
           cat /tmp/widget.py   > $inline  # Python source detected
    
       The bytes are sniffed for magic numbers (PNG/JPEG/GIF/WEBP/PDF/
       SVG/HTML), Python source heuristics, and rendered directly.
    
    3. **JSON envelope** (advanced control):
    
           echo '{"type":"info","text":"build done","level":"ok"}' > $inline
           echo '{"type":"link","url":"...","label":"..."}' > $inline
           echo '{"type":"python","code":"_widget = QPushButton(\\"Hi\\")"}' > $inline
    
       Detected by a leading `{`. Same schema as before, plus the new
       `gif`, `audio`, `video`, `pdf`, `model3d`, and `python` types.
    
    `$inline` is a convenience shell variable that points at this file —
    seeded by the terminal so the agent never has to spell out the
    full /n/rioa/terms/<term_id>/inline path.
    
    Concurrency:
        Each fid (open file handle) buffers independently, so multiple
        producers can write inline payloads at the same time. This file
        does NOT participate in fence parsing on $term/output, so writes
        here stay safe even while agents are mid-stream.
    """
    
    # Magic byte signatures for binary-content sniffing.
    _MAGIC_PNG  = b'\x89PNG\r\n\x1a\n'
    _MAGIC_JPEG = b'\xff\xd8\xff'
    _MAGIC_GIF  = (b'GIF87a', b'GIF89a')
    _MAGIC_WEBP = b'WEBP'  # at offset 8 inside RIFF
    
    # Cap on how big a payload we'll buffer in memory. 32 MB is plenty
    # for any reasonable inline media; rejects accidental cat-of-huge-file.
    _MAX_BYTES = 32 * 1024 * 1024
    
    def __init__(self, terminal_ref):
        super().__init__("inline")
        self._terminal_ref = terminal_ref
        self._fid_buffers: Dict[int, bytearray] = {}
        self._fid_truncated: Dict[int, bool] = {}
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return (
            b"Write to spawn an inline widget in the terminal:\n"
            b"  echo /path/to/file > $inline    (path: image/gif/audio/video/pdf/3d/.py/.html)\n"
            b"  echo www.google.com > $inline   (URL: opens an inline browser via QWebEngineView)\n"
            b"  cat  /path/to/file > $inline    (raw bytes, magic-sniffed)\n"
            b'  echo \'{"type":"...","..."}\' > $inline   (JSON envelope)\n'
            b"types: image gif audio video pdf model3d python url html link info warn error\n"
        )
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        buf = self._fid_buffers.get(fid.fid)
        if buf is None:
            buf = bytearray()
            self._fid_buffers[fid.fid] = buf
            self._fid_truncated[fid.fid] = False
        # Cap memory usage.
        room = self._MAX_BYTES - len(buf)
        if room <= 0:
            self._fid_truncated[fid.fid] = True
            return len(data)  # ack, but drop
        if len(data) > room:
            buf.extend(data[:room])
            self._fid_truncated[fid.fid] = True
        else:
            buf.extend(data)
        return len(data)
    
    def clunk(self, fid: FidState):
        """Dispatch the accumulated payload when the file handle closes."""
        buf = self._fid_buffers.pop(fid.fid, None)
        truncated = self._fid_truncated.pop(fid.fid, False)
        if buf is None or not buf:
            return
        terminal = self._terminal_ref()
        if terminal is None:
            return
        
        # Capture the terminal's bash cwd so relative paths
        # (e.g. `echo README.md > $inline` from /home/user/proj)
        # resolve the way the user expects.
        cwd = None
        if hasattr(terminal, '_shell_cwd'):
            try:
                cwd = terminal._shell_cwd()
            except Exception:
                cwd = None
        
        # Auto-detect what we got: JSON envelope, file path, or raw bytes.
        payload = self._dispatch(bytes(buf), truncated=truncated, cwd=cwd)
        if payload is None:
            return
        
        # Bounce to Qt main thread.
        try:
            import json as _json
            from PySide6.QtCore import QMetaObject, Qt as _Qt, Q_ARG
            QMetaObject.invokeMethod(
                terminal, "_on_inline_media",
                _Qt.QueuedConnection,
                Q_ARG(str, _json.dumps(payload)),
            )
        except Exception:
            pass
    
    # ------------------------------------------------------------------
    # Auto-detection logic
    # ------------------------------------------------------------------
    
    def _dispatch(self, data: bytes, truncated: bool = False,
                  cwd: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Examine the buffer and return a payload dict for the widget.
        
        Returns None if we can't make sense of the input.
        Resolution order:
          1. Looks like JSON (first non-whitespace byte is '{') → parse it.
          2. Single short text line that resolves to an existing file path
             → read & sniff. Both absolute paths and paths relative to the
             terminal's shell cwd are accepted.
          3. Looks like Python source (PySide6 inline scene) → python payload.
          4. Binary or multi-line content → sniff magic bytes.
          5. Plain text fallback → info widget.
        """
        # --- 1. JSON envelope ------------------------------------------
        # We test on the raw bytes to avoid wasting a UTF-8 decode on
        # potentially-binary data.
        stripped = data.lstrip()
        if stripped[:1] == b'{':
            try:
                import json as _json
                obj = _json.loads(stripped.decode('utf-8'))
                if isinstance(obj, dict) and obj.get('type'):
                    return obj
            except Exception:
                # Falls through to other detectors — maybe it just
                # happens to start with '{' but is actually content.
                pass
        
        # --- 2. URL (single short text line that looks like a URL) ----
        # Recognized forms:
        #   https://example.com/foo
        #   http://example.com
        #   www.example.com
        #   example.com / sub.example.org/path
        # We test before path resolution because (a) URLs are
        # unambiguous when they have a scheme, and (b) bare-domain
        # heuristics shouldn't be defeated by an unrelated file with
        # the same name happening to exist on disk.
        if len(data) <= 4096 and b'\x00' not in data:
            try:
                text_for_url = data.decode('utf-8').strip()
            except UnicodeDecodeError:
                text_for_url = None
            if text_for_url and '\n' not in text_for_url and self._looks_like_url(text_for_url):
                return {"type": "url", "url": text_for_url}
        
        # --- 3. Path (single short text line, exists on disk) ----------
        # Heuristic: <= 4096 bytes, no NULs, decodable as UTF-8, exactly
        # one logical line. We accept absolute paths AND paths relative
        # to the terminal's bash cwd, so `echo README.md > $inline` from
        # /home/user/proj resolves to /home/user/proj/README.md.
        #
        # We refuse to treat tokens that look obviously like Python
        # source as paths even if a file by that exact name happens to
        # exist (e.g. the buffer literally is `import os` and there's
        # an `import os` file somewhere). Path-likeness here means
        # the line has no spaces, contains a slash or a recognizable
        # extension, and isn't a Python keyword sequence.
        if len(data) <= 4096 and b'\x00' not in data:
            try:
                text = data.decode('utf-8').strip()
            except UnicodeDecodeError:
                text = None
            if text and '\n' not in text and self._looks_like_path(text):
                # Try the path the user gave us, then a cwd-relative one.
                resolved = self._resolve_path(text, cwd)
                if resolved is not None:
                    return self._dispatch_from_path(resolved)
                # Path-shaped but not found — surface as error so the
                # user sees the typo instead of a confusing info banner.
                # We attach the cwd we tried so the message is debuggable.
                if cwd and not text.startswith('/'):
                    return {
                        "type": "error",
                        "text": f"inline: file not found: {text}  (cwd: {cwd})",
                    }
                return {
                    "type": "error",
                    "text": f"inline: file not found: {text}",
                }
        
        # --- 3. Python source ------------------------------------------
        # Heuristic: looks like Python and isn't already a recognized
        # binary format. We accept either an explicit `#!python` shebang
        # or any code that opens with familiar Python markers near the
        # top. The magic-bytes sniff in step 4 takes precedence for
        # binary formats, so we test it first to avoid false positives
        # on e.g. an HTML file that happens to contain `import` somewhere.
        if self._looks_like_python(data):
            try:
                code = data.decode('utf-8', errors='replace')
                # Strip optional `#!python` shebang.
                first_line, _, rest = code.partition('\n')
                if first_line.strip().lower() in ('#!python', '#!pyside', '#!pyside6'):
                    code = rest
                return {"type": "python", "code": code}
            except Exception:
                pass
        
        # --- 4. Raw bytes — sniff magic ---------------------------------
        kind = self._sniff_magic(data)
        if kind == 'png':
            return self._image_payload(data, 'png', truncated)
        if kind == 'jpeg':
            return self._image_payload(data, 'jpg', truncated)
        if kind == 'gif':
            # GIFs route to the animated widget, not the static image
            # widget — animated GIFs should auto-loop.
            return self._gif_payload(data, truncated)
        if kind == 'webp':
            return self._image_payload(data, 'webp', truncated)
        if kind == 'pdf':
            # PDF needs a path (PyMuPDF reads from disk). Materialize
            # the bytes to a temp file.
            return self._pdf_payload_from_bytes(data, truncated)
        if kind == 'svg':
            try:
                text = data.decode('utf-8')
                if len(text) <= 64 * 1024:
                    return {"type": "html", "content": text}
            except Exception:
                pass
            return {"type": "info", "text": "(SVG content received)"}
        if kind == 'html':
            try:
                text = data.decode('utf-8', errors='replace')
                if len(text) > 256 * 1024:
                    text = text[:256 * 1024] + '\n<!-- ... truncated ... -->'
                return {"type": "html", "content": text}
            except Exception:
                pass
        
        # --- 5. Plain text fallback ------------------------------------
        try:
            text = data.decode('utf-8', errors='replace').rstrip('\n')
        except Exception:
            return None
        if not text:
            return None
        if len(text) <= 200 and '\n' not in text:
            return {"type": "info", "text": text}
        escaped = (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))
        if len(escaped) > 256 * 1024:
            escaped = escaped[:256 * 1024] + '\n... truncated ...'
        return {"type": "html", "content": f"<pre>{escaped}</pre>"}
    
    def _dispatch_from_path(self, path: str) -> Dict[str, Any]:
        """Build a widget payload from an on-disk file path.
        
        Routes by extension. The widget loads from `path` directly when
        possible (images, gifs, audio, video, pdf, 3d) so we don't have
        to round-trip through base64 for files that already exist.
        """
        ext = os.path.splitext(path)[1].lower()
        
        # GIFs first — they need the animated widget, NOT the static one.
        if ext == '.gif':
            return {"type": "gif", "path": path}
        # Other still images.
        if ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.svg'):
            return {"type": "image", "path": path}
        # Audio.
        if ext in ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma'):
            return {"type": "audio", "path": path}
        # Video.
        if ext in ('.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.flv', '.wmv'):
            return {"type": "video", "path": path}
        # PDF.
        if ext == '.pdf':
            return {"type": "pdf", "path": path}
        # 3D models.
        if ext in ('.obj', '.stl', '.ply', '.glb', '.gltf', '.3ds', '.dae', '.fbx'):
            return {"type": "model3d", "path": path}
        # Python source — exec inline.
        if ext == '.py':
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    code = f.read(1024 * 1024)  # 1 MB cap on script size
                return {"type": "python", "code": code}
            except Exception as e:
                return {"type": "error", "text": f"inline: read failed: {e}"}
        # HTML — read and embed.
        if ext in ('.html', '.htm'):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(256 * 1024)
                return {"type": "html", "content": content}
            except Exception as e:
                return {"type": "error", "text": f"inline: read failed: {e}"}
        # Plain text-ish.
        if ext in ('.txt', '.log', '.md'):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(256 * 1024)
                escaped = (content
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;'))
                return {"type": "html", "content": f"<pre>{escaped}</pre>"}
            except Exception as e:
                return {"type": "error", "text": f"inline: read failed: {e}"}
        # Anything else: info banner with the path.
        return {"type": "info", "text": f"file: {path}"}
    
    @staticmethod
    def _looks_like_path(text: str) -> bool:
        """Cheap heuristic: does this string look like a file path?
        
        Path-shaped means:
          - no spaces (paths with spaces would need to be quoted by the
            shell anyway, and `echo` would normalize them)
          - no obvious Python/shell syntax (parens, brackets, '=', etc.)
          - either contains a slash, or has a known media-file extension,
            or both
        
        We deliberately reject things like `import os` or `_widget = QPushButton()`
        even if a file by that exact name happens to exist somewhere — those
        are Python source and we want them to fall through to the Python
        detector.
        """
        if not text or len(text) > 4096:
            return False
        # Must be a single token on a single line.
        if any(c in text for c in ' \t\n\r'):
            return False
        # Reject obvious non-paths.
        if any(c in text for c in '()[]{}=;<>|&'):
            return False
        # Accept: contains a slash (any kind of path).
        if '/' in text:
            return True
        # Accept: bare filename with a recognized extension. We restrict
        # to the set of extensions our dispatch actually handles, so
        # arbitrary tokens like `import` (no dot) or `__init__.py` don't
        # both register — only real media/code files.
        _PATH_EXTS = {
            # images
            '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg',
            # audio
            '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma',
            # video
            '.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.flv', '.wmv',
            # documents
            '.pdf', '.html', '.htm', '.txt', '.log', '.md',
            # 3D
            '.obj', '.stl', '.ply', '.glb', '.gltf', '.3ds', '.dae', '.fbx',
            # code
            '.py',
        }
        ext = os.path.splitext(text)[1].lower()
        return ext in _PATH_EXTS
    
    # Common TLDs we accept for bare-domain detection. Not exhaustive
    # — agents can always be explicit and use https:// for anything
    # else. Keep this list narrow enough that random English words
    # like `back.up` don't get mistaken for URLs.
    _COMMON_TLDS = frozenset({
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'io', 'co', 'ai', 'app', 'dev', 'me', 'tv', 'tech',
        'info', 'biz', 'name', 'mobi', 'pro', 'xyz', 'site',
        'online', 'shop', 'cloud', 'page', 'blog', 'news',
        # ccTLDs we expect agents to encounter most often
        'us', 'uk', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'kr',
        'in', 'br', 'mx', 'es', 'it', 'nl', 'se', 'no', 'fi',
        'dk', 'pl', 'ru', 'ch', 'be', 'at', 'eu',
    })
    
    @classmethod
    def _looks_like_url(cls, text: str) -> bool:
        """Cheap heuristic: does this string look like a URL?
        
        Accepts:
          - Anything with an explicit scheme: http://, https://, ftp://,
            file://. The `://` is unambiguous.
          - Bare domains starting with `www.` (case-insensitive).
          - `<host>.<tld>(/...)?` where <tld> is a recognized TLD. This
            keeps us from false-positiving on filenames like `notes.txt`
            (txt isn't a TLD).
        
        Rejects:
          - Anything with whitespace (URLs don't have spaces).
          - File paths starting with `/` or `~` (they're not URLs).
          - Single-segment tokens with no dots.
        """
        if not text:
            return False
        # No whitespace anywhere.
        if any(c in text for c in ' \t\n\r'):
            return False
        # Filesystem-anchored paths are never URLs.
        if text.startswith('/') or text.startswith('~'):
            return False
        # Reject obvious non-URL syntax.
        if any(c in text for c in '()[]{}<>|&;'):
            return False
        lowered = text.lower()
        # Explicit scheme.
        for scheme in ('http://', 'https://', 'ftp://', 'file://'):
            if lowered.startswith(scheme):
                return True
        # `www.` prefix is a strong signal.
        if lowered.startswith('www.'):
            return True
        # Bare domain: must have at least one dot, and the segment after
        # the final dot (before any path/query) must be a known TLD.
        # We split on '/' first to peel off any path component.
        host_part = lowered.split('/', 1)[0].split('?', 1)[0].split('#', 1)[0]
        if '.' not in host_part:
            return False
        # No leading or trailing dot.
        if host_part.startswith('.') or host_part.endswith('.'):
            return False
        last_segment = host_part.rsplit('.', 1)[-1]
        # Bare ccTLDs / gTLDs we recognize.
        if last_segment in cls._COMMON_TLDS:
            return True
        return False
    
    @staticmethod
    def _resolve_path(text: str, cwd: Optional[str]) -> Optional[str]:
        """Resolve a (possibly relative) path against the terminal's
        bash cwd. Returns the absolute path if it exists and is a
        regular file, else None.
        
        Tries in order:
          1. The path as given (handles already-absolute paths).
          2. Joined with the terminal's bash cwd (relative paths).
          3. Expanded ~ (handles `echo ~/foo.png > $inline`).
        """
        # Try as-is first — covers absolute paths.
        if os.path.isabs(text):
            if os.path.isfile(text):
                return text
            return None
        # Tilde expansion.
        if text.startswith('~'):
            expanded = os.path.expanduser(text)
            if os.path.isfile(expanded):
                return expanded
        # Relative to bash cwd.
        if cwd:
            candidate = os.path.normpath(os.path.join(cwd, text))
            if os.path.isfile(candidate):
                return candidate
        return None
    
    @classmethod
    def _sniff_magic(cls, data: bytes) -> Optional[str]:
        """Return a short type name based on magic bytes, or None."""
        if data.startswith(cls._MAGIC_PNG):
            return 'png'
        if data.startswith(cls._MAGIC_JPEG):
            return 'jpeg'
        if any(data.startswith(m) for m in cls._MAGIC_GIF):
            return 'gif'
        # WEBP: RIFF....WEBP
        if data[:4] == b'RIFF' and len(data) >= 12 and data[8:12] == b'WEBP':
            return 'webp'
        # PDF: starts with %PDF-
        if data.startswith(b'%PDF-'):
            return 'pdf'
        # SVG: starts with <?xml or <svg
        head = data[:512].lstrip()
        lower = head.lower()
        if lower.startswith(b'<?xml') and b'<svg' in lower[:512]:
            return 'svg'
        if lower.startswith(b'<svg'):
            return 'svg'
        # HTML: <!doctype html, <html, or starts with <p>/<div>/etc.
        if lower.startswith(b'<!doctype html') or lower.startswith(b'<html'):
            return 'html'
        # Crude check: starts with '<' and has a closing '>' nearby and
        # contains nothing obviously binary in the first 512 bytes.
        if (lower.startswith(b'<') and b'>' in lower[:64]
                and b'\x00' not in head):
            return 'html'
        return None
    
    @classmethod
    def _looks_like_python(cls, data: bytes) -> bool:
        """Heuristic: does this look like Python source?
        
        We require either an explicit shebang or that the first ~512
        bytes contain familiar Python markers AND are pure UTF-8 text
        with no NUL bytes (rules out binary formats).
        """
        if b'\x00' in data[:512]:
            return False
        try:
            head = data[:512].decode('utf-8')
        except UnicodeDecodeError:
            return False
        head_stripped = head.lstrip()
        # Explicit opt-in.
        first_line = head_stripped.split('\n', 1)[0].strip().lower()
        if first_line in ('#!python', '#!pyside', '#!pyside6'):
            return True
        # Heuristic markers: any of these strongly indicate Python.
        markers = (
            'from PySide6',
            'from PyQt5',
            'from PyQt6',
            'import PySide6',
            'import PyQt',
            '_widget = ',
            '_widget=',
        )
        if any(m in head for m in markers):
            return True
        # Weaker: the buffer starts with `import ` or `from ` AND the
        # full data parses as Python. We only run compile() if cheap
        # markers were seen.
        if head_stripped.startswith(('import ', 'from ', 'def ', 'class ')):
            try:
                # Cap at 64 KB for the syntax check — anything larger is
                # an unusual case and we'll let the full exec catch it.
                sample = data[:64 * 1024].decode('utf-8', errors='replace')
                compile(sample, '<inline-python-probe>', 'exec')
                return True
            except SyntaxError:
                return False
            except Exception:
                return False
        return False
    
    def _image_payload(self, data: bytes, fmt: str, truncated: bool) -> Dict[str, Any]:
        """Wrap raw image bytes as a base64 image payload."""
        import base64
        if truncated:
            # Image was cut off mid-stream — don't try to render bad bytes.
            return {
                "type": "error",
                "text": f"inline: image exceeded {self._MAX_BYTES // (1024*1024)}MB cap",
            }
        return {
            "type": "image",
            "data": base64.b64encode(data).decode('ascii'),
            "format": fmt,
        }
    
    def _gif_payload(self, data: bytes, truncated: bool) -> Dict[str, Any]:
        """Materialize raw GIF bytes to disk so QMovie can stream-decode it.
        
        QMovie wants a file path or QIODevice to animate; passing raw
        bytes via QPixmap.loadFromData would render only the first frame.
        We write to a tempfile and rely on the inline widget's keepalive
        list to clean up.
        """
        if truncated:
            return {
                "type": "error",
                "text": f"inline: gif exceeded {self._MAX_BYTES // (1024*1024)}MB cap",
            }
        import base64
        return {
            "type": "gif",
            "data": base64.b64encode(data).decode('ascii'),
        }
    
    def _pdf_payload_from_bytes(self, data: bytes, truncated: bool) -> Dict[str, Any]:
        """Write PDF bytes to a temp file and return a path payload.
        
        PyMuPDF wants an on-disk path (or a stream open kwarg); the
        path approach is simpler and matches how files arriving via
        `echo /path > $inline` are already routed.
        """
        if truncated:
            return {
                "type": "error",
                "text": f"inline: pdf exceeded {self._MAX_BYTES // (1024*1024)}MB cap",
            }
        import tempfile
        try:
            fd, tmp_path = tempfile.mkstemp(suffix='.pdf', prefix='inline_')
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            return {"type": "pdf", "path": tmp_path}
        except Exception as e:
            return {"type": "error", "text": f"inline: pdf write failed: {e}"}


class TerminalParseFile(SyntheticFile):
    """
    /n/rioa/terms/<term_id>/parse

    Per-terminal parse file. Writing Python code here executes it in the
    terminal's TerminalScenePanel — so ``graphics_scene``, ``graphics_view``
    and ``main_window`` inside the code refer to the panel's QGraphicsScene,
    QGraphicsView and the panel widget itself.

    The panel is created lazily on first write, and made visible in the
    terminal splitter so the user immediately sees the result.

    Mirrors the structural contract of the global ParseFile: streaming
    writes are buffered per-fid and dispatched on clunk.
    """

    def __init__(self, terminal_ref):
        super().__init__("parse")
        self._terminal_ref = terminal_ref
        self._fid_parsers: Dict[int, StreamingParser] = {}
        self._fid_decoders: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_panel(self):
        """
        Resolve the terminal and (lazily) its TerminalScenePanel.
        Returns None if the terminal has been garbage-collected.

        We schedule both panel creation and visibility on the Qt main
        thread because TerminalScenePanel constructs QGraphicsScene /
        QGraphicsView and reparents the panel into the terminal splitter.
        """
        terminal = self._terminal_ref() if callable(self._terminal_ref) else self._terminal_ref
        if terminal is None:
            return None

        # Fast path: panel already exists.
        if getattr(terminal, "scene_panel", None) is not None:
            return terminal.scene_panel

        # Slow path: build it on the Qt main thread, then wait for it.
        from PySide6.QtCore import QMetaObject, Qt as _Qt
        try:
            QMetaObject.invokeMethod(
                terminal, "ensure_scene_panel_visible",
                _Qt.BlockingQueuedConnection,
            )
        except Exception:
            # BlockingQueuedConnection cannot be used on the same thread as
            # the receiver; fall back to a direct call (we're on Qt main).
            terminal.ensure_scene_panel_visible()

        return getattr(terminal, "scene_panel", None)

    # ------------------------------------------------------------------
    # 9P file protocol
    # ------------------------------------------------------------------

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        parser = self._fid_parsers.get(fid.fid)
        if parser and parser.has_content():
            return b"buffering...\n"
        return b"ready\n"

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            if fid.fid not in self._fid_parsers:
                self._fid_parsers[fid.fid] = StreamingParser()
            if fid.fid not in self._fid_decoders:
                import codecs
                self._fid_decoders[fid.fid] = codecs.getincrementaldecoder('utf-8')('strict')

            parser = self._fid_parsers[fid.fid]
            decoder = self._fid_decoders[fid.fid]
            text = decoder.decode(data, False)
            if text:
                parser.feed(text)
            return len(data)
        except UnicodeDecodeError:
            raise ValueError("Invalid UTF-8")

    def clunk(self, fid: FidState):
        """Execute accumulated code when the file is closed."""
        decoder = self._fid_decoders.pop(fid.fid, None)
        if decoder:
            try:
                tail = decoder.decode(b'', True)
                if tail:
                    parser = self._fid_parsers.get(fid.fid)
                    if parser:
                        parser.feed(tail)
            except UnicodeDecodeError:
                pass

        parser = self._fid_parsers.get(fid.fid)
        if not parser:
            return
        code = parser.flush()
        del self._fid_parsers[fid.fid]

        if code:
            asyncio.create_task(self._execute_code(code))

    async def _execute_code(self, code: str):
        """Execute code inside the terminal's TerminalScenePanel."""
        panel = self._get_panel()
        if panel is None:
            print(f"TerminalParseFile: terminal no longer exists, skipping execute")
            return

        executor = panel.executor

        # Capture stdout/stderr so prints inside user code surface in the
        # server log (the terminal itself doesn't currently expose a
        # dedicated output sink for /parse, mirroring Acme/operator panels).
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                result = await executor.execute(code)
            except Exception as e:
                print(f"TerminalParseFile [{panel.term_id}] execution error: {e}")
                return

        out = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        if out:
            print(f"[term {panel.term_id} /parse stdout]\n{out}", end="")
        if err:
            print(f"[term {panel.term_id} /parse stderr]\n{err}", end="")

        if result.success:
            print(
                f"✓ term {panel.term_id} /parse: "
                f"{len(result.items_registered)} item(s), "
                f"{len(result.widgets_created)} widget(s), "
                f"version {panel.scene_manager.versions.current_version}"
            )
        else:
            print(f"✗ term {panel.term_id} /parse failed:\n{result.error}")


class RoutesManager:
    """
    Centralized route (Plan9Attachment) manager — independent of any terminal.
    
    All route creation/removal goes through here. The terminal widget,
    operator panel, /attach macro, and $rioa/routes filesystem writes
    all funnel into this single manager.
    
    Lives at the RioRoot level so routes are accessible even without
    a terminal: /n/rioa/routes
    """
    
    def __init__(self, llmfs_mount: str = "/n/llm"):
        self.attachments: Dict[str, 'Plan9Attachment'] = {}
        self.llmfs_mount = llmfs_mount
        self._listeners = []   # callbacks: fn(event, source, destination)
    
    def add_listener(self, fn):
        """Register a callback for route changes: fn('add'|'remove', src, dst)"""
        self._listeners.append(fn)
    
    def remove_listener(self, fn):
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass
    
    def _notify(self, event: str, source: str, destination: str = ""):
        for fn in self._listeners:
            try:
                fn(event, source, destination)
            except Exception:
                pass
    
    def _expand(self, path: str) -> str:
        if not path.startswith('/'):
            return os.path.join(self.llmfs_mount, path)
        return path
    
    def add_route(self, source: str, destination: str) -> bool:
        """
        Create a Plan9Attachment route.  Returns True on success.
        
        Imports Plan9Attachment lazily to avoid circular imports.
        """
        source = self._expand(source)
        destination = self._expand(destination)
        
        if not source or not destination:
            return False
        
        # Stop existing attachment for this source
        if source in self.attachments:
            self.attachments[source].stop()
        
        # Lazy import — Plan9Attachment lives in terminal_widget.py
        from .terminal_widget import Plan9Attachment
        
        attachment = Plan9Attachment(source, destination)
        attachment.start()
        self.attachments[source] = attachment
        self._notify('add', source, destination)
        return True
    
    def remove_route(self, source: str) -> bool:
        """Remove a route by source path.  Returns True if found."""
        source = self._expand(source)
        if source in self.attachments:
            self.attachments[source].stop()
            dst = self.attachments[source].destination
            del self.attachments[source]
            self._notify('remove', source, dst)
            return True
        return False
    
    def list_routes(self) -> list:
        """Return list of (source, destination, is_running) tuples."""
        return [
            (src, att.destination, att.is_running)
            for src, att in self.attachments.items()
        ]
    
    def stop_all(self):
        """Stop and remove all routes."""
        for att in self.attachments.values():
            att.stop()
        self.attachments.clear()


class RoutesFile(SyntheticFile):
    """
    Routes file for managing Plan9-style attachments via the filesystem.
    
    /n/rioa/routes
    
    Read:  returns all active attachments, one per line. SECOND open
           (after the first read fully consumed the previous content)
           BLOCKS until a route is added/removed/changed — this is the
           push channel for clients like the operator UI. Shell users
           who `cat routes` once still get an immediate return + EOF
           because each cat does one open → read-to-EOF → close.
    
           Format per line:
           /path/to/source -> /path/to/destination [running|stopped]
    
    Write: /path/to/a -> /path/to/b
           Creates a new Plan9Attachment (while-loop cat) route.
    
    Delete (write "-source"):
           echo '-/n/llm/agents/claude/bash' > /n/rioa/routes
    """
    
    def __init__(self, routes_manager: RoutesManager):
        super().__init__("routes")
        self._manager = routes_manager
        # Push notification: every route change `set()`s this event.
        # Reads at offset 0 wait on it after the previous content was
        # consumed. First read after construction returns immediately
        # because the event starts set (initial state is "ready").
        self._changed = asyncio.Event()
        self._changed.set()
        self._content_consumed = False
        self._lock = asyncio.Lock()
        # Hook into the manager so add/remove fires the event.
        self._manager.add_listener(self._on_route_change)
    
    def _on_route_change(self, event: str, source: str, destination: str = ""):
        """Called by RoutesManager._notify on every add/remove. Wakes
        any blocked readers."""
        self._changed.set()
    
    async def _render(self) -> bytes:
        """Serialize current routes to bytes."""
        routes = self._manager.list_routes()
        if not routes:
            return b"(no routes)\n"
        lines = []
        for source, destination, running in routes:
            status = "running" if running else "stopped"
            lines.append(f"{source} -> {destination} [{status}]")
        return ("\n".join(lines) + "\n").encode()
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        # If a previous read fully consumed the content AND we're back
        # at offset 0, we're starting a new read cycle. Block until
        # something changes.
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._changed.clear()
        # Wait for content to be ready (initially set, then set again
        # on every route change).
        await self._changed.wait()
        
        async with self._lock:
            data = await self._render()
            chunk = data[offset:offset + count]
            if offset + len(chunk) >= len(data):
                # Reader hit EOF — mark consumed so the next offset-0
                # read will rearm.
                self._content_consumed = True
            return chunk
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        text = data.decode('utf-8').strip()
        if not text:
            return len(data)
        
        # Delete route: -/path/to/source
        if text.startswith('-'):
            source = text[1:].strip()
            self._manager.remove_route(source)
            return len(data)
        
        # Create route: /path/to/a -> /path/to/b
        if ' -> ' not in text:
            raise ValueError("Format: /path/to/source -> /path/to/destination")
        
        parts = text.split(' -> ', 1)
        source = parts[0].strip()
        destination = parts[1].strip()
        
        if not source or not destination:
            raise ValueError("Both source and destination must be specified")
        
        self._manager.add_route(source, destination)
        
        return len(data)


class TerminalDir(SyntheticDir):
    """
    A single terminal exposed as a filesystem directory.
    
    /n/rioa/terms/<term_id>/
    ├── ctl        # Write commands to execute in shell
    ├── input      # Write prompts to send to connected agent
    ├── output     # Write to display text, read to monitor (bidirectional)
    ├── inline     # Write JSON envelopes to render inline media widgets
    ├── parse      # Write Python code to draw on this terminal's scene panel
    ├── stdin      # Write commands to execute in shell (route target)
    ├── stdout     # Read shell output (blocking, enables while-cat loops)
    └── interrupt  # Write anything to send SIGINT to shell
    
    NOTE: Routes have moved to /n/rioa/routes (shared, terminal-independent).
    """
    
    def __init__(self, term_id: str, terminal_ref):
        super().__init__(term_id)
        self.term_id = term_id
        self._terminal_ref = terminal_ref
        
        self.output_file = TerminalOutputFile(terminal_ref)
        self.stdout_file = TerminalStdoutFile()
        self.stdin_file = TerminalStdinFile(terminal_ref, self.stdout_file)
        self.parse_file = TerminalParseFile(terminal_ref)
        self.inline_file = TerminalInlineFile(terminal_ref)
        
        self.add(CtlFile("ctl", TerminalCtlHandler(terminal_ref)))
        self.add(TerminalInputFile(terminal_ref))
        self.add(self.output_file)
        self.add(self.stdout_file)
        self.add(self.stdin_file)
        self.add(self.parse_file)
        self.add(self.inline_file)
        self.add(TerminalInterruptFile(terminal_ref))


class TerminalsDir(SyntheticDir):
    """
    Directory containing all active terminals.
    
    /n/rioa/terms/
    ├── term_a1b2c3d4/
    │   ├── ctl
    │   ├── input
    │   ├── output       # text (fence parser → inline code widgets)
    │   ├── inline       # JSON envelopes → inline media widgets
    │   └── ...
    └── term_e5f6g7h8/
        └── ...
    """
    
    def __init__(self, routes_manager: RoutesManager = None):
        super().__init__("terms")
        self._terminals: Dict[str, TerminalDir] = {}
        self._routes_manager = routes_manager
    
    def register_terminal(self, term_id: str, terminal_ref) -> TerminalDir:
        """Register a terminal and create its filesystem directory.
        
        Also wires the shared RoutesManager into the terminal widget
        so that /attach, /coder, /master etc. all store routes in
        the centralized /n/rioa/routes file.
        """
        term_dir = TerminalDir(term_id, terminal_ref)
        self._terminals[term_id] = term_dir
        self.add(term_dir)
        
        # Wire the shared routes manager into the terminal widget
        if self._routes_manager is not None:
            terminal = terminal_ref() if callable(terminal_ref) else terminal_ref
            if terminal is not None and hasattr(terminal, 'set_routes_manager'):
                terminal.set_routes_manager(self._routes_manager)
        
        return term_dir
    
    def unregister_terminal(self, term_id: str):
        """Remove a terminal from the filesystem."""
        if term_id in self._terminals:
            del self._terminals[term_id]
            self.remove(term_id)


# ============================================================
# Session State File (cp-friendly save/restore)
# ============================================================

class StateFile(SyntheticFile):
    """
    Complete session state as a single cp-friendly JSON file.
    
    Save:    cp /n/rioa/scene/state /tmp/my_session.json
    Restore: cp /tmp/my_session.json /n/rioa/scene/state
    
    The JSON envelope contains:
    
    {
      "rio_state": 1,                    # format version
      "timestamp": 1234567890.0,
      "scene": { ... },                  # scene items (parsed, not double-encoded)
      "settings": {
        "width": 1920, "height": 1080,
        "background": "#000000"
      },
      "versions": [ ... ],               # full version history
      "vars": { ... },                   # serializable namespace variables
      "code_history": [                  # all code from every version
        {"version": 1, "code": "..."},
        ...
      ]
    }
    
    On write (restore), the scene is cleared and rebuilt:
      1. Scene items are imported via scene_manager.from_json()
      2. Settings (size, background) are restored
      3. Code from the latest version is re-executed to rebuild
         the Python namespace (widgets, variables, etc.)
      4. A fresh snapshot is taken
    
    Read behaviour:
      The state is serialized once on the first read after open and
      cached per-fid.  Subsequent reads at increasing offsets return
      slices of that cached blob.  When offset reaches the end, b""
      is returned (EOF) and cat terminates.  The cache is freed on
      clunk (file close).
    """
    
    def __init__(self, scene_manager, executor):
        super().__init__("state")
        self.scene_manager = scene_manager
        self.executor = executor
        self._read_cache = {}      # fid -> bytes (serialized state)
        self._write_buf = {}       # fid -> bytearray (incoming data)
    
    def _serialize_vars(self) -> dict:
        """Serialize the namespace — same logic as VarsFile."""
        namespace = self.executor.context.get_namespace()
        vars_dict = {}
        for key, value in namespace.items():
            if key.startswith('_'):
                continue
            if key in ('__builtins__', '__name__'):
                continue
            if hasattr(value, '__file__'):
                continue
            try:
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    vars_dict[key] = value
            except Exception:
                pass
        return vars_dict
    
    def _build_state(self) -> bytes:
        """Build the complete state envelope and return serialized bytes."""
        sm = self.scene_manager
        vm = sm.versions
        
        # scene_manager.to_json() may return a JSON string or a dict/list.
        # Ensure we store it as a parsed object so json.dumps doesn't
        # double-encode it.
        scene_raw = sm.to_json()
        if isinstance(scene_raw, str):
            try:
                scene_obj = json.loads(scene_raw)
            except (json.JSONDecodeError, ValueError):
                scene_obj = scene_raw   # fallback: store as string
        else:
            scene_obj = scene_raw
        
        # Collect code from every version
        code_history = []
        for snap_dict in vm.list_versions():
            ver = snap_dict.get("version", 0)
            code = snap_dict.get("label", "")
            # Try to get the full code from the snapshot object.
            # vm.snapshots may be a list or a dict depending on implementation.
            if hasattr(vm, 'snapshots'):
                snaps = vm.snapshots
                snap_obj = None
                if isinstance(snaps, dict):
                    snap_obj = snaps.get(ver)
                elif isinstance(snaps, list):
                    for s in snaps:
                        if hasattr(s, 'version') and s.version == ver:
                            snap_obj = s
                            break
                if snap_obj and hasattr(snap_obj, 'code') and snap_obj.code:
                    code = snap_obj.code
            if code:
                code_history.append({
                    "version": ver,
                    "code": code,
                })
        
        state = {
            "rio_state": 1,
            "timestamp": time.time(),
            "scene": scene_obj,
            "settings": {
                "width": sm.width,
                "height": sm.height,
                "background": getattr(sm, 'background_color', '#000000'),
            },
            "versions": vm.list_versions(),
            "vars": self._serialize_vars(),
            "code_history": code_history,
        }
        
        return json.dumps(state, indent=2).encode()
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """
        Read the serialized session state.
        
        The state is built once on the first read (offset 0) and cached
        for subsequent reads at increasing offsets. cat / cp will read
        in msize-sized chunks until we return b"" (EOF).
        """
        # Build and cache on first read
        if fid.fid not in self._read_cache:
            try:
                self._read_cache[fid.fid] = self._build_state()
            except Exception as e:
                error_msg = json.dumps({"error": str(e)}).encode()
                self._read_cache[fid.fid] = error_msg
        
        data = self._read_cache[fid.fid]
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """
        Accumulate incoming data (cp may write in chunks).
        Actual restore happens on clunk (file close).
        """
        if fid.fid not in self._write_buf:
            self._write_buf[fid.fid] = bytearray()
        self._write_buf[fid.fid].extend(data)
        return len(data)
    
    def clunk(self, fid: FidState):
        """File closed — free read cache; if we have write data, restore."""
        self._read_cache.pop(fid.fid, None)
        buf = self._write_buf.pop(fid.fid, None)
        if buf:
            asyncio.create_task(self._restore_state(bytes(buf)))
    
    async def _restore_state(self, raw: bytes):
        """Parse the JSON state and restore the scene."""
        try:
            state = json.loads(raw.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"✗ State restore failed: invalid JSON — {e}")
            return
        
        if state.get("rio_state") != 1:
            print(f"✗ State restore failed: unknown format version")
            return
        
        sm = self.scene_manager
        
        print(f"\n{'='*60}")
        print(f"RESTORING SESSION STATE")
        print(f"{'='*60}")
        
        # 1. Clear current scene
        await sm.clear()
        print("  ✓ Scene cleared")
        
        # 2. Restore settings
        settings = state.get("settings", {})
        if "width" in settings:
            sm.width = settings["width"]
        if "height" in settings:
            sm.height = settings["height"]
        if "background" in settings:
            sm.background_color = settings["background"]
            if hasattr(sm, 'refresh_background_color'):
                sm.refresh_background_color()
        print(f"  ✓ Settings restored ({sm.width}x{sm.height})")
        
        # 3. Replay ALL code versions in order to rebuild scene + namespace
        #
        #    Each version's code builds on the previous one's namespace
        #    (e.g., v2 creates `proxy`, v3 references `proxy`).
        #    Replaying just the last version fails because earlier
        #    variables don't exist yet.
        #
        #    We skip from_json() entirely — the code replay IS the
        #    source of truth for both scene items and namespace state.
        #    from_json() would only create visual proxies without the
        #    live Python objects behind them.
        
        code_history = state.get("code_history", [])
        last_code = ""
        replayed = 0
        
        for entry in code_history:
            code = entry.get("code", "")
            ver = entry.get("version", "?")
            
            # Skip empty code and bare labels (e.g., "initial")
            if not code or code == "initial" or '\n' not in code:
                continue
            
            last_code = code
            print(f"  → Replaying v{ver} ({len(code)} chars)...")
            try:
                result = await self.executor.execute(code)
                if result.success:
                    replayed += 1
                else:
                    print(f"    ⚠ v{ver} errors: {result.error[:120]}")
            except Exception as e:
                print(f"    ⚠ v{ver} failed: {e}")
        
        if replayed:
            print(f"  ✓ Replayed {replayed} version(s)")
        else:
            # No code to replay — fall back to importing scene items
            # so at least the visual state is restored
            scene_data = state.get("scene")
            if scene_data:
                try:
                    if isinstance(scene_data, (dict, list)):
                        sm.from_json(json.dumps(scene_data))
                    else:
                        sm.from_json(scene_data)
                    print(f"  ✓ Scene items imported (no code to replay)")
                except Exception as e:
                    print(f"  ✗ Scene import failed: {e}")
        
        # 4. Restore serializable variables that code didn't recreate
        saved_vars = state.get("vars", {})
        if saved_vars:
            namespace = self.executor.context.get_namespace()
            restored = 0
            for key, value in saved_vars.items():
                if key not in namespace:
                    namespace[key] = value
                    restored += 1
            if restored:
                print(f"  ✓ Restored {restored} variable(s)")
        
        # 5. Take a fresh snapshot
        sm.take_snapshot(label="restored session", code=last_code)
        
        # 6. Refresh the display
        sm.refresh()
        
        ts = state.get("timestamp", 0)
        if ts:
            import datetime
            dt = datetime.datetime.fromtimestamp(ts)
            print(f"  ✓ Session from {dt.strftime('%Y-%m-%d %H:%M:%S')} restored")
        
        print(f"{'='*60}\n")


# ============================================================
# Scene and Root Directories
# ============================================================

class SceneDir(SyntheticDir):
    """
    Enhanced Scene directory.
    
    /scene/
    ├── ctl          # Scene control (with undo/redo/goto/snapshot commands)
    ├── parse        # Code execution
    ├── stdout       # Print output
    ├── STDERR       # Errors (blocking read)
    ├── vars         # Namespace vars
    ├── state        # cp-friendly session save/restore
    └── version      # cat = list + current; echo undo/redo > version
    """
    
    def __init__(self, scene_manager: SceneManager, qt_objects: dict = None, context_file=None):
        super().__init__("scene")
        self.scene_manager = scene_manager
        self.qt_objects = qt_objects or {}
        
        print("  SceneDir: creating executor...")
        # Create executor
        context = ExecutionContext(
            scene_manager,
            main_window=qt_objects.get('main_window'),
            graphics_scene=qt_objects.get('graphics_scene'),
            graphics_view=qt_objects.get('graphics_view')
        )
        self.executor = Executor(context)
        
        print("  SceneDir: creating output files...")
        # Output files
        self.stdout = StdoutFile()
        self.stderr = StderrFile()
        
        self.add(self.stdout)
        self.add(self.stderr)
        
        print("  SceneDir: creating parse file...")
        # Parse file (connected to context_file for successful code tracking)
        self.parse_file = ParseFile(
            scene_manager, self.executor,
            self.stdout, self.stderr, context_file
        )
        self.add(self.parse_file)
        
        print("  SceneDir: creating vars...")
        # Vars
        self.add(VarsFile(self.executor))
        
        print("  SceneDir: creating state file...")
        # State file (cp-friendly session save/restore)
        self.state_file = StateFile(scene_manager, self.executor)
        self.add(self.state_file)
        
        print("  SceneDir: creating version file...")
        # Version file (replaces versions/ directory)
        self.add(VersionFile(scene_manager))
        
        print("  SceneDir: creating ctl...")
        # Control (with undo/redo commands)
        self.add(CtlFile("ctl", SceneCtlHandler(scene_manager)))

        print("  SceneDir: creating quick file...")
        self.quick_file = QuickFile(
            scene_manager, self.executor,
            stdout_file=self.stdout,
            stderr_file=self.stderr,
            parse_file=self.parse_file,
        )
        self.add(self.quick_file)

        # Signals directory — UDP-backed cross-machine PySide6 Signals.
        # The bus itself is started by RioServer (main.py) so this dir
        # just exposes its state and ctl. bus_getter is a callable
        # rather than the bus directly because the dir is built before
        # the server actually starts the bus.
        if _HAS_SIGNALS:
            try:
                print("  SceneDir: creating signals dir...")
                self.signals_dir = SignalsDir(bus_getter=_rio_signals_bus)
                self.add(self.signals_dir)
            except Exception as e:
                print(f"WARNING: Could not create SignalsDir: {e}")
                import traceback
                traceback.print_exc()

        print("  SceneDir: taking initial snapshot...")
        # Take initial snapshot (version 0 = empty scene)
        scene_manager.take_snapshot(label="initial", code="")
        
        print(f"  SceneDir: DONE ({len(self.children)} children)")


class RioRoot(SyntheticDir):
    """
    Root of the Rio filesystem.
    
    /n/rioa/
    ├── ctl
    ├── screen            # Screenshot image (PNG)
    ├── CONTEXT           # All successfully executed code (blocking read)
    ├── routes            # Plan9-style attachment routes (shared, terminal-independent)
    ├── terms/            # Terminal filesystem exposure
    │   └── <term_id>/
    │       ├── ctl       # Execute shell commands
    │       ├── input     # Send prompts to connected agent
    │       ├── output    # Monitor terminal output
    │       ├── parse     # Python code → draws on this terminal's scene panel
    │       ├── stdin     # Write commands to shell (route target)
    │       ├── stdout    # Read shell output (blocking)
    │       └── interrupt # Send SIGINT
    ├── nodes/            # User-created operator nodes
    │   ├── ctl           # echo "new text foo" / "delete text_foo" > ctl
    │   └── <kind>_<id>/  # text/debug/media/bash/python — see nodes.py
    └── scene/
        ├── ctl
        ├── parse
        ├── stdout
        ├── STDERR        # Blocking read (uppercase = blocking)
        ├── vars
        ├── state         # cp-friendly session save/restore
        ├── version       # cat = list + current; echo undo/redo > version
        └── quick
    """
    
    def __init__(self, scene_manager: SceneManager = None, qt_objects: dict = None):
        super().__init__("")
        
        if scene_manager is None:
            scene_manager = SceneManager()
        
        self.scene_manager = scene_manager
        self.qt_objects = qt_objects or {}
        
        self.add(CtlFile("ctl", RioCtlHandler(self)))
        self.add(ScreenFile(scene_manager, qt_objects))
        
        # CONTEXT: all successfully executed code (blocking read)
        self.context_file = ContextFile()
        self.add(self.context_file)
        
        # Routes manager + file (shared, terminal-independent)
        self.routes_manager = RoutesManager()
        self.routes_file = RoutesFile(self.routes_manager)
        self.add(self.routes_file)
        
        # Terminals directory (receives routes_manager for auto-wiring)
        self.terms_dir = TerminalsDir(self.routes_manager)
        self.add(self.terms_dir)

        # User-created operator nodes (text/debug/media/bash/python).
        # See nodes.py — the operator's UI writes "new <kind> <id>" to
        # nodes/ctl and the matching <kind>_<id>/ directory appears.
        # Routes manager is passed in so deleting a node also clears
        # any /n/<m>/routes entries whose endpoints lived inside that
        # node — otherwise Plan9Attachment keeps trying to cat the now-
        # vanished port files.
        try:
            from .nodes import NodesDir
        except ImportError:
            # Standalone (no package) layout — nodes.py sits next to
            # this file on the path.
            from nodes import NodesDir
        try:
            self.nodes_dir = NodesDir(routes_manager=self.routes_manager)
            self.add(self.nodes_dir)
        except Exception as e:
            print(f"WARNING: Could not create NodesDir: {e}")
            import traceback
            traceback.print_exc()

        try:
            from .acme.acme_fs import get_acme_dir
            self.acme_dir = get_acme_dir()
            self.add(self.acme_dir)
        except Exception as e:
            print(f"WARNING: Could not load acme filesystem: {e}")
            import traceback
            traceback.print_exc()

        try:
            self.scene_dir = SceneDir(scene_manager, qt_objects, self.context_file)
            self.add(self.scene_dir)
        except Exception as e:
            print(f"FATAL: Could not create SceneDir: {e}")
            import traceback
            traceback.print_exc()
            # Create a minimal stub so /scene at least exists
            self.scene_dir = SyntheticDir("scene")
            self.add(self.scene_dir)
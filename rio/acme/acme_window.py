"""
Acme Window - dual-pane window with Plan 9 mouse semantics

Plan 9 Acme Mouse Semantics:
  SELECTION COLORS:
    Left  = Blue #99CCFF  (normal selection)
    Mid   = Red  #FF8888  (select-to-execute)
    Right = Green #88EE88 (select-to-plumb/search)

  ON RELEASE:
    Mid release   → execute the red-selected text as ctl command
    Right release → plumb (open) or search for the green-selected text

  CHORDING:
    Left held + Mid press   = CUT
    Left held + Right press = PASTE

  CURSOR WARPING:
    Right-click search warps mouse to found text
    Plumb open warps mouse to first char of new window
"""

import os
import subprocess
import threading
from pathlib import Path
import json

from PySide6.QtWidgets import (QFrame, QVBoxLayout, QTextEdit, QLabel,
                               QWidget, QApplication, QHBoxLayout,
                               QStackedWidget)
from PySide6.QtCore import (Qt, QEvent, Signal, QPoint, QTimer)
from PySide6.QtGui import (QPalette, QColor, QTextCursor, QCursor)

from .content_detector import detect_content_type, is_executable_code
from .program_generators import *
from .acme_fs import get_acme_dir

import subprocess as _subprocess
import tempfile as _tempfile

ACME_FONT_SIZE = 13
ACME_TAG_BG = "#EAEACC"
ACME_BODY_BG = "#FFFFEA"

# Plan 9 selection colors
SEL_BLUE  = QColor(0x99, 0xCC, 0xFF)
SEL_RED   = QColor(0xFF, 0x88, 0x88)
SEL_GREEN = QColor(0x88, 0xEE, 0x88)
SEL_TEXT  = QColor(0, 0, 0)


def _is_9p_path(path):
    """Check if path is a 9P filesystem path.
    
    Matches '/n', '/n/', and '/n/anything/...'
    The naive check path.startswith('/n/') misses '/n' itself
    (after rstrip('/') normalization), causing deadlocks when
    os.path.isdir('/n') goes through FUSE on the main thread.
    """
    return path == '/n' or path.startswith('/n/')


class Plan9Attachment:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.process = None
        self.script_path = None

    def start(self):
        fd, self.script_path = _tempfile.mkstemp(suffix='.sh', prefix='acme_attach_')
        script = f"""#!/bin/bash
SOURCE="{self.source}"
DEST="{self.destination}"
mkdir -p "$(dirname "$DEST")" 2>/dev/null || true
while true; do
    CONTENT=$(cat "$SOURCE" 2>/dev/null)
    RC=$?
    [ $RC -ne 0 ] && break
    [ -n "$CONTENT" ] && echo "$CONTENT" > "$DEST" 2>/dev/null
done
"""
        with os.fdopen(fd, 'w') as f:
            f.write(script)
        os.chmod(self.script_path, 0o755)
        self.process = _subprocess.Popen(
            ['nohup', 'bash', self.script_path],
            stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
            stdin=_subprocess.DEVNULL, start_new_session=True)

    def stop(self):
        if self.process:
            try:
                import signal as _signal
                os.killpg(self.process.pid, _signal.SIGTERM)
                self.process.wait(timeout=2)
            except Exception:
                try: self.process.kill(); self.process.wait(timeout=1)
                except Exception: pass
            self.process = None
        if self.script_path and os.path.exists(self.script_path):
            try: os.unlink(self.script_path)
            except Exception: pass

    @property
    def is_running(self):
        return self.process is not None and self.process.poll() is None


class AcmeWindow(QFrame):
    close_requested = Signal(object)
    _stream_append = Signal(str)
    _stream_exec = Signal(str)
    _term_output = Signal(str)
    _content_loaded = Signal(str, str)  # (content_type, code_or_text)

    def __init__(self, path="", parent=None, llmfs_mount="/n/mux/llm",
                 rio_mount="/n/mux/default",
                 p9_host="localhost", p9_port=5640):
        super().__init__(parent)
        self.path = os.path.abspath(path) if (path and not os.path.isabs(path)) else (path or "")

        from .acme_clean import get_next_window_id
        self.window_id = get_next_window_id()

        self.accumulated_code = ""
        self.last_error = ""

        self.llmfs_mount = llmfs_mount
        self.rio_mount = rio_mount
        self.p9_host = p9_host
        self.p9_port = p9_port

        self.agent_name = f"acme_{self.window_id}"
        self.agent_path = os.path.join(self.llmfs_mount, self.agent_name)

        self._routes_manager = None
        self.attachment = None
        self._attachment_source = None

        acme_dir = get_acme_dir()
        self.fs_dir = acme_dir.register_window(self.window_id, self)

        # Drag state
        self.drag_active = False
        self.drag_start_global_pos = QPoint(0, 0)
        self.drag_start_window_y = 0

        # Plan 9 mouse state
        self._active_button = None
        self._sel_start_pos = None
        self._left_held = False

        # Terminal mode (Plan 9 acme 'win' style)
        self.is_terminal = False
        self.working_dir = ""
        self._term_executing = False

        # UI
        self.drag_handle = None
        self.tag_line = None
        self.text_pane = None
        self.graphical_pane = None
        self.pane_stack = None
        self.media_player = None
        self.audio_output = None
        self.execution_namespace = None
        self._stream_thread = None

        self.setup_ui()
        self.load_content()

        self._stream_append.connect(self._on_stream_append)
        self._stream_exec.connect(self._on_stream_exec)
        self._term_output.connect(self._on_term_output)
        self._content_loaded.connect(self._on_content_loaded)
        QTimer.singleShot(100, self._setup_agent)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def set_routes_manager(self, manager):
        self._routes_manager = manager
        if self.attachment and self._attachment_source:
            src, dst = self._attachment_source, self.attachment.destination
            self.attachment.stop(); self.attachment = None
            manager.add_route(src, dst)

    def _add_route(self, source, destination):
        self._attachment_source = source
        if self._routes_manager:
            self._routes_manager.add_route(source, destination)
        else:
            if self.attachment: self.attachment.stop()
            self.attachment = Plan9Attachment(source, destination)
            self.attachment.start()

    def _remove_route(self):
        if self._attachment_source and self._routes_manager:
            self._routes_manager.remove_route(self._attachment_source)
        if self.attachment: self.attachment.stop(); self.attachment = None
        self._attachment_source = None

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def setup_ui(self):
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("background-color: transparent; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: transparent;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.setSpacing(0)

        self.drag_handle = QLabel()
        self.drag_handle.setFixedSize(12, 12)
        self.drag_handle.setStyleSheet("QLabel { background-color: #8888AA; border: none; margin: 2px; }")
        self.drag_handle.setCursor(Qt.SizeAllCursor)

        self.tag_line = QTextEdit()
        self.tag_line.setStyleSheet(f"""
            QTextEdit {{
                background-color: {ACME_TAG_BG};
                border: none;
                border-bottom: 1px solid #888888;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {ACME_FONT_SIZE}px;
                color: black;
                padding: 0px 2px;
            }}
        """)
        self.tag_line.document().setDocumentMargin(1)
        self.tag_line.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tag_line.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tag_line.setReadOnly(False)
        self.tag_line.setCursorWidth(2)
        self.tag_line.setLineWrapMode(QTextEdit.WidgetWidth)

        palette = self.tag_line.palette()
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Highlight, SEL_BLUE)
        palette.setColor(QPalette.HighlightedText, SEL_TEXT)
        self.tag_line.setPalette(palette)

        self._update_tag_line()
        # Fit height AFTER setting text
        self._fit_tag_height()
        self.tag_line.textChanged.connect(self._fit_tag_height)
        self.tag_line.viewport().setContextMenuPolicy(Qt.NoContextMenu)

        top_bar_layout.addWidget(self.drag_handle, 0, Qt.AlignTop)
        top_bar_layout.addWidget(self.tag_line, 1)

        # Panes
        self.pane_stack = QStackedWidget()

        self.text_pane = QTextEdit()
        self.text_pane.setStyleSheet(f"""
            QTextEdit {{
                background-color: {ACME_BODY_BG};
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {ACME_FONT_SIZE}px;
                border: none;
                color: black;
            }}
        """)
        self.text_pane.document().setDocumentMargin(2)
        self.text_pane.setReadOnly(False)
        self.text_pane.setContextMenuPolicy(Qt.NoContextMenu)
        p2 = self.text_pane.palette()
        p2.setColor(QPalette.Highlight, SEL_BLUE)
        p2.setColor(QPalette.HighlightedText, SEL_TEXT)
        self.text_pane.setPalette(p2)

        self.graphical_pane = QWidget()
        self.graphical_pane_layout = QVBoxLayout(self.graphical_pane)
        self.graphical_pane_layout.setContentsMargins(0, 0, 0, 0)

        self.pane_stack.addWidget(self.text_pane)
        self.pane_stack.addWidget(self.graphical_pane)

        layout.addWidget(top_bar)
        layout.addWidget(self.pane_stack, 1)

        self.drag_handle.installEventFilter(self)
        self.tag_line.viewport().installEventFilter(self)
        self.text_pane.viewport().installEventFilter(self)

    def _update_tag_line(self):
        if self.path:
            # For 9P paths (e.g. /n/...) keep as-is; for local paths, normalize
            if os.path.isabs(self.path):
                display_path = self.path
            else:
                display_path = os.path.abspath(self.path)
        else:
            display_path = f"Window {self.window_id}"
        
        # For /n/ paths, avoid ANY filesystem probe on the main thread.
        # FUSE calls back to the 9P server which may need the Qt event loop,
        # causing a deadlock.  Infer directory-ness from trailing slash or
        # absence of file extension.
        if self.path and _is_9p_path(self.path):
            # Heuristic: treat as dir if it ends with '/' or has no extension
            looks_like_dir = self.path.endswith('/') or not os.path.splitext(self.path)[1]
            ct = None  # skip detect_content_type for 9P
            is_dir = looks_like_dir
        else:
            ct = detect_content_type(self.path) if self.path else None
            is_dir = self._path_isdir(self.path) if self.path else False
        
        if ct in ['video', 'audio']:
            cmds = "Del Get Play Pause Stop Code Main Clear"
        elif is_dir:
            cmds = "Del Get Code Main Clear"
        else:
            cmds = "Del Get Put Code Main Clear"
        self.tag_line.setPlainText(f"{display_path} {cmds}")

    def _fit_tag_height(self):
        doc_h = int(self.tag_line.document().size().height()) + 2
        h = max(20, min(doc_h, 120))
        if self.tag_line.maximumHeight() != h:
            self.tag_line.setFixedHeight(h)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _set_sel_color(self, widget, color):
        p = widget.palette()
        p.setColor(QPalette.Highlight, color)
        p.setColor(QPalette.HighlightedText, SEL_TEXT)
        widget.setPalette(p)

    def _warp_cursor(self, widget, char_pos):
        """Warp mouse to screen position of char_pos in widget.
        Does NOT alter the text cursor or selection — only moves the mouse."""
        widget.ensureCursorVisible()
        # Build a temporary cursor just to compute screen position
        tmp = QTextCursor(widget.document())
        tmp.setPosition(char_pos)
        rect = widget.cursorRect(tmp)
        gpos = widget.viewport().mapToGlobal(rect.center())
        QCursor.setPos(gpos)

    # ------------------------------------------------------------------
    # Robust path helpers (work with 9P mounts at /n/)
    # ------------------------------------------------------------------

    # Timeout (seconds) for filesystem probes on 9P paths.
    # Keeps the UI responsive even if a file blocks.
    _9P_PROBE_TIMEOUT = 0.4

    @staticmethod
    def _safe_probe(func, path, timeout=None):
        """Run a filesystem probe in a thread with a timeout.
        
        9P synthetic files (StreamFile, SupplementaryOutputFile, etc.)
        block indefinitely on open/read/stat when no data is available.
        We can't maintain a static list of blocking filenames because
        supplementary outputs have arbitrary user-defined names.
        
        Instead, every probe on a /n/ path runs in a worker thread with
        a short timeout.  If the probe doesn't return in time we assume
        the target is a blocking file and return the fallback value.
        
        Returns the probe result, or None on timeout/error.
        """
        if timeout is None:
            timeout = AcmeWindow._9P_PROBE_TIMEOUT
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(func, path)
                return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            return None

    @staticmethod
    def _path_exists(path):
        """Check if path exists, with fallback for 9P mounts where
        os.path.exists / os.stat may fail even though the path is valid.
        Never blocks on 9P blocking files."""
        if not path:
            return False
        # Fast path for non-9P
        if not _is_9p_path(path):
            try:
                if os.path.exists(path):
                    return True
            except (OSError, PermissionError):
                pass
            try:
                os.listdir(path)
                return True
            except NotADirectoryError:
                return True
            except (OSError, PermissionError):
                pass
            try:
                with open(path, 'rb') as f:
                    f.read(1)
                return True
            except (OSError, PermissionError, IsADirectoryError):
                return False
        
        # 9P path — ALL probes must be timeout-protected.
        # Bare os.path.exists() does a stat that can block forever
        # on synthetic files.
        def _probe(p):
            try:
                os.listdir(p)
                return True
            except NotADirectoryError:
                return True
            except (OSError, PermissionError):
                pass
            try:
                with open(p, 'rb') as f:
                    f.read(1)
                return True
            except IsADirectoryError:
                return True
            except (OSError, PermissionError):
                return False
        
        result = AcmeWindow._safe_probe(_probe, path)
        return bool(result)

    @staticmethod
    def _path_isdir(path):
        """Check if path is a directory, with fallback for 9P mounts.
        Never blocks on 9P blocking files."""
        if not path:
            return False
        # Fast path for non-9P
        if not _is_9p_path(path):
            try:
                if os.path.isdir(path):
                    return True
            except (OSError, PermissionError):
                pass
            try:
                os.listdir(path)
                return True
            except (NotADirectoryError, FileNotFoundError, PermissionError, OSError):
                return False
        
        # 9P path — ALL probes must be timeout-protected.
        # Bare os.path.isdir() does a stat that can block forever
        # on synthetic files.
        def _probe(p):
            try:
                os.listdir(p)
                return True
            except (NotADirectoryError, FileNotFoundError, PermissionError, OSError):
                return False

        result = AcmeWindow._safe_probe(_probe, path)
        return bool(result)

    @staticmethod
    def _path_isfile(path):
        """Check if path is a file, with fallback for 9P mounts.
        Never blocks on 9P blocking files."""
        if not path:
            return False
        # Fast path for non-9P
        if not _is_9p_path(path):
            try:
                if os.path.isfile(path):
                    return True
            except (OSError, PermissionError):
                pass
            try:
                with open(path, 'rb') as f:
                    f.read(1)
                return True
            except (IsADirectoryError, FileNotFoundError, PermissionError, OSError):
                return False
        
        # 9P path — timeout-protected
        try:
            if os.path.isfile(path):
                return True
        except (OSError, PermissionError):
            pass

        def _probe(p):
            try:
                with open(p, 'rb') as f:
                    f.read(1)
                return True
            except (IsADirectoryError, FileNotFoundError, PermissionError, OSError):
                return False

        result = AcmeWindow._safe_probe(_probe, path)
        return bool(result)

    def _is_plumbable_path(self, text):
        """Determine if text looks like a filesystem path that could be opened.
        Returns the resolved absolute path if plumbable, else None.
        
        For /n/ paths, assumes existence without probing to avoid deadlocking
        the Qt main thread on FUSE → 9P calls.
        """
        if not text:
            return None
        # Already absolute
        if os.path.isabs(text):
            # For /n/ paths, assume they exist — load_content will handle
            # errors asynchronously in a background thread
            if _is_9p_path(text):
                return text
            if self._path_exists(text):
                return text
            return None
        # Relative — resolve against current window's directory
        if self.path:
            base = self.path if (not _is_9p_path(self.path) and self._path_isdir(self.path)) else os.path.dirname(self.path)
            full = os.path.join(base, text)
            if _is_9p_path(full):
                return full
            if self._path_exists(full):
                return full
        # Try cwd
        full = os.path.abspath(text)
        if self._path_exists(full):
            return full
        return None

    # ------------------------------------------------------------------
    # Content access
    # ------------------------------------------------------------------

    def get_text_content(self):
        return self.text_pane.toPlainText()

    def set_text_content_and_raise(self, text):
        self.text_pane.setPlainText(text)
        self.pane_stack.setCurrentIndex(0)

    def set_path(self, path):
        self.path = path
        self._update_tag_line()

    # ------------------------------------------------------------------
    # ctl dispatch
    # ------------------------------------------------------------------

    def execute_ctl_command(self, command):
        print(f"[Acme {self.window_id}] ctl: {command}")
        cmd = command.strip()
        cl = cmd.lower()
        if cl.startswith("ai "):
            prompt = cmd[3:].strip()
            if prompt: self.call_ai(prompt)
            return
        if cl.startswith("show "):
            what = cmd[5:].strip().lower()
            if what == "text": self.pane_stack.setCurrentIndex(0)
            elif what == "code":
                self.text_pane.setPlainText(self.accumulated_code)
                self.pane_stack.setCurrentIndex(0)
            elif what in ("err", "error"):
                self.text_pane.setPlainText(self.last_error or "(no error)")
                self.pane_stack.setCurrentIndex(0)
            return
        if cmd == "Del": self.close_requested.emit(self)
        elif cmd == "Get": self._cmd_get()
        elif cmd == "Put": self._cmd_put()
        elif cmd == "Code":
            self.text_pane.setPlainText(self.accumulated_code)
            self.pane_stack.setCurrentIndex(0)
        elif cmd == "Main": self.pane_stack.setCurrentIndex(1)
        elif cmd == "Clear":
            self.accumulated_code = ""; self.execution_namespace = None
        elif cmd == "Play" and self.media_player: self.media_player.play()
        elif cmd == "Pause" and self.media_player: self.media_player.pause()
        elif cmd == "Stop" and self.media_player: self.media_player.stop()
        elif cmd == "ClearHistory": self._clear_conversation_history()
        elif cmd in ("Term", "term"):
            acme = self.get_acme_parent()
            if acme:
                col = acme.preferred_column or self.get_parent_column()
                if col:
                    # Avoid os.path.isdir on /n/ paths (deadlock risk)
                    if self.path and _is_9p_path(self.path):
                        wd = os.getcwd()
                    else:
                        wd = self.path if self.path and os.path.isdir(self.path) else os.getcwd()
                    col.add_terminal(wd)

    def _cmd_get(self):
        words = self.tag_line.toPlainText().strip().split()
        if words:
            p = words[0]
            if not os.path.isabs(p):
                if self.path and os.path.dirname(self.path):
                    p = os.path.join(os.path.dirname(self.path), p)
                else: p = os.path.abspath(p)
            self.path = p; self._update_tag_line(); self.load_content()

    def _cmd_put(self):
        if not self.path:
            return
        # For /n/ paths, skip isdir check (deadlock risk); writes to dirs
        # will fail naturally
        if not _is_9p_path(self.path) and os.path.isdir(self.path):
            return
        try:
            with open(self.path, 'w') as f: f.write(self.text_pane.toPlainText())
        except Exception as e: self.last_error = str(e)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute_code_from_fs(self, code):
        try:
            self._exec_code_in_graphical_pane(code)
            if self.accumulated_code: self.accumulated_code += "\n\n"
            self.accumulated_code += code
            self.pane_stack.setCurrentIndex(1)
        except Exception as e:
            self.last_error = f"Execution Error: {e}"
            error_msg = f"# {self.last_error}\n\n{code}"
            try:
                error_code = generate_message_display(error_msg, "Execution Error")
                self._exec_code_in_graphical_pane(error_code)
                self.pane_stack.setCurrentIndex(1)
            except Exception:
                # Ultimate fallback — use text pane
                self.text_pane.setPlainText(error_msg)
                self.pane_stack.setCurrentIndex(0)

    def _exec_code_in_graphical_pane(self, code):
        while self.graphical_pane_layout.count():
            item = self.graphical_pane_layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()

        # Free canvas: no layout on the execution container.
        # The AI decides widget placement via setGeometry / move / resize.
        # A convenience 'canvas_size' tuple is provided so the AI knows the
        # available space.  For built-in generators that want full-fill
        # behaviour, a _FillParent helper auto-resizes on parent resize.
        ec = QWidget()
        # No layout set on ec — this IS the free canvas.

        from PySide6.QtWidgets import (QPushButton, QLabel, QLineEdit,
            QTextEdit, QVBoxLayout, QHBoxLayout, QMainWindow, QSlider,
            QSpinBox, QCheckBox, QRadioButton, QComboBox, QListWidget,
            QTableWidget, QGridLayout, QGroupBox, QScrollArea, QFrame,
            QProgressBar, QTabWidget)

        if self.execution_namespace is None:
            self.execution_namespace = {
                '__builtins__': __builtins__, 'QApplication': QApplication,
                'QWidget': QWidget, 'QPushButton': QPushButton, 'QLabel': QLabel,
                'QLineEdit': QLineEdit, 'QTextEdit': QTextEdit,
                'QVBoxLayout': QVBoxLayout, 'QHBoxLayout': QHBoxLayout,
                'QGridLayout': QGridLayout, 'QGroupBox': QGroupBox,
                'QScrollArea': QScrollArea, 'QFrame': QFrame,
                'QMainWindow': QMainWindow, 'QSlider': QSlider, 'QSpinBox': QSpinBox,
                'QCheckBox': QCheckBox, 'QRadioButton': QRadioButton,
                'QComboBox': QComboBox, 'QListWidget': QListWidget,
                'QTableWidget': QTableWidget, 'QProgressBar': QProgressBar,
                'QTabWidget': QTabWidget, 'Qt': Qt,
            }
        self.execution_namespace['container'] = ec
        # canvas_size gives the AI the current available dimensions
        gp = self.graphical_pane
        self.execution_namespace['canvas_size'] = (gp.width() or 600, gp.height() or 400)
        exec(code, self.execution_namespace)

        if hasattr(ec, 'media_player'): self.media_player = ec.media_player
        if hasattr(ec, 'audio_output'): self.audio_output = ec.audio_output
        if hasattr(ec, 'text_edit'):
            ec.text_edit.viewport().installEventFilter(self)
            ec.text_edit.setContextMenuPolicy(Qt.NoContextMenu)
            ec.text_edit.viewport().setContextMenuPolicy(Qt.NoContextMenu)
            ec.text_edit.acme_window = self
        if hasattr(ec, 'terminal'):
            ec.terminal.viewport().installEventFilter(self)
            ec.terminal.setContextMenuPolicy(Qt.NoContextMenu)
            ec.terminal.viewport().setContextMenuPolicy(Qt.NoContextMenu)

        self.graphical_pane_layout.addWidget(ec)
        ec.show()

        # Drive repaints for animated/OpenGL content.
        # QOpenGLWidget.update() only schedules a repaint — if the Qt event
        # loop is idle (no mouse/keyboard/resize), the paint event never fires.
        # This timer ensures the graphical pane repaints at ~60fps when it
        # contains animated widgets (3D viewers, QTimers driving rotation, etc.).
        #
        # The timer is lightweight: it calls update() on the container which
        # propagates to children.  If no child has pending updates, it's a no-op.
        self._start_repaint_driver(ec)

    # ------------------------------------------------------------------
    # Repaint driver for animated graphical pane content
    # ------------------------------------------------------------------

    def _start_repaint_driver(self, container):
        """Start a timer that drives repaints on the graphical pane container.

        Needed because QOpenGLWidget.update() only marks the widget dirty —
        it relies on the event loop to actually process the paint event.
        When the app is idle (no user input), Qt may never get around to it.

        Strategy: tick at 60fps, calling update() on the container.  Qt
        coalesces redundant updates so this is cheap when nothing changed.
        If the graphical pane is hidden or the container has no OpenGL/animated
        children, we stop automatically.
        """
        self._stop_repaint_driver()
        self._repaint_container = container

        self._repaint_timer = QTimer(self)
        self._repaint_timer.setTimerType(Qt.PreciseTimer)
        self._repaint_timer.timeout.connect(self._repaint_tick)
        self._repaint_timer.start(16)  # ~60 fps

    def _stop_repaint_driver(self):
        """Stop the repaint driver if running."""
        if hasattr(self, '_repaint_timer') and self._repaint_timer is not None:
            self._repaint_timer.stop()
            self._repaint_timer.deleteLater()
            self._repaint_timer = None
        self._repaint_container = None

    def _repaint_tick(self):
        """Called at ~60fps — nudge the graphical pane to repaint."""
        # Stop if we switched away from the graphical pane
        if self.pane_stack.currentIndex() != 1:
            return
        ec = getattr(self, '_repaint_container', None)
        if ec is not None:
            ec.update()

    # ------------------------------------------------------------------
    # Content loading
    # ------------------------------------------------------------------

    def load_content(self):
        """Load and display content for the current path.
        
        For /n/ (9P) paths, ALL filesystem I/O runs in a background thread
        to prevent deadlocks.  The 9P FUSE mount sends requests back to the
        9P server in this process, which may need the Qt event loop — so any
        blocking I/O on the main thread would deadlock.
        """
        if not self.path:
            return

        if _is_9p_path(self.path):
            # Show loading indicator immediately in the graphical pane
            try:
                loading_code = generate_message_display(
                    f"Loading {self.path} ...", "Loading")
                self._exec_code_in_graphical_pane(loading_code)
                self.pane_stack.setCurrentIndex(1)
            except Exception:
                self.text_pane.setPlainText(f"Loading {self.path} ...")
                self.pane_stack.setCurrentIndex(0)
            # Offload all filesystem probing to a background thread
            thread = threading.Thread(
                target=self._load_content_bg, args=(self.path,), daemon=True)
            thread.start()
            return

        # Non-9P path: safe to probe on main thread
        self._load_content_sync(self.path)

    def _load_content_bg(self, path):
        """Background thread: probe filesystem and generate content code.
        
        For 9P directory listings, the actual os.listdir() and isdir probes
        run HERE (in the background) so they can safely talk to the FUSE mount
        without deadlocking the Qt main thread.  The result is sent as
        pre-built text, not as executable code that would do I/O on exec().
        
        Posts result back to Qt main thread via signal.
        """
        try:
            ct = detect_content_type(path)
            if ct is None:
                # Fallback probing — safe in background thread
                if self._path_isdir(path):
                    ct = 'directory'
                elif self._path_isfile(path):
                    ct = 'text'

            code = None
            if ct == 'directory':
                # For 9P dirs, do the I/O HERE and build a non-I/O display widget
                code = self._generate_dir_listing_precomputed(path)
            elif ct == 'image':
                code = generate_image_viewer(path)
            elif ct == 'video':
                code = generate_video_player(path)
            elif ct == 'audio':
                code = generate_audio_player(path)
            elif ct == '3d':
                code = generate_3d_viewer(path)
            elif ct == 'pdf':
                code = generate_pdf_viewer(path)
            elif ct == 'text' or self._path_isfile(path):
                # For 9P files, read content HERE in background
                code = self._generate_file_content_precomputed(path)

            if code:
                self._content_loaded.emit(ct or 'text', code)
            else:
                self._content_loaded.emit('empty', f"Cannot determine content type for: {path}")
        except Exception as e:
            self._content_loaded.emit('error', f"Error loading {path}: {e}")

    def _generate_dir_listing_precomputed(self, path):
        """Read directory in background thread and return code that just
        displays the pre-computed listing (no I/O at exec time)."""
        try:
            # For 9P paths, even os.listdir can block if there are
            # stale/dead mounts — timeout-protect it
            if _is_9p_path(path):
                result = AcmeWindow._safe_probe(
                    lambda p: sorted(os.listdir(p)), path, timeout=2.0)
                if result is None:
                    return generate_message_display(
                        f"Timeout reading {path} (stale mount?)", "Timeout")
                entries = result
            else:
                entries = sorted(os.listdir(path))
        except Exception as e:
            return generate_message_display(
                f"Error reading {path}: {e}", "Error")

        all_entries = []
        parent = os.path.dirname(path.rstrip('/'))
        if path.rstrip('/') != parent and parent:
            all_entries.append("../")

        for entry in entries:
            full_path = os.path.join(path, entry)
            # For 9P paths, isdir/listdir can block forever on synthetic
            # files — use timeout-protected probes
            is_dir = False
            if _is_9p_path(path):
                result = AcmeWindow._safe_probe(
                    lambda p: os.path.isdir(p), full_path, timeout=0.4)
                if result:
                    is_dir = True
                elif result is None:
                    # Timed out — try listdir with timeout as fallback
                    result2 = AcmeWindow._safe_probe(
                        lambda p: (os.listdir(p), True)[1], full_path, timeout=0.4)
                    if result2:
                        is_dir = True
            else:
                try:
                    if os.path.isdir(full_path):
                        is_dir = True
                except (OSError, PermissionError):
                    pass
                if not is_dir:
                    try:
                        os.listdir(full_path)
                        is_dir = True
                    except NotADirectoryError:
                        pass
                    except (OSError, PermissionError, FileNotFoundError):
                        pass

            if is_dir:
                all_entries.append(entry + "/")
            else:
                all_entries.append(entry)

        # Format columns
        if all_entries:
            max_len = max(len(e) for e in all_entries)
            col_w = max_len + 2
            num_cols = max(1, 80 // col_w)
            num_rows = (len(all_entries) + num_cols - 1) // num_cols
            lines = []
            for row in range(num_rows):
                parts = []
                for col in range(num_cols):
                    idx = row + col * num_rows
                    if idx < len(all_entries):
                        e = all_entries[idx]
                        parts.append(e.ljust(col_w) if col < num_cols - 1 else e)
                lines.append("".join(parts).rstrip())
            content = "\n".join(lines)
        else:
            content = "(empty directory)"

        # Escape for Python string embedding
        content_escaped = content.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')

        return f'''# Directory listing: {path}
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt

text_edit = QTextEdit()
text_edit.setReadOnly(True)
text_edit.setStyleSheet("""
    QTextEdit {{
        background-color: rgba(255, 255, 255, 0);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        border: none;
        color: black;
    }}
""")
text_edit.setPlainText('{content_escaped}')
text_edit.setParent(container)
text_edit.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=text_edit, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize
container.text_edit = text_edit
'''

    def _generate_file_content_precomputed(self, path):
        """Read file content in background thread and return code that
        displays the pre-read content (no I/O at exec time)."""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading file: {e}"

        # Escape for Python string embedding
        content_escaped = content.replace('\\', '\\\\').replace("'''", "\\'\\'\\'" ).replace('\r', '\\r')

        path_escaped = path.replace('\\', '\\\\').replace("'", "\\'")

        return f"""# File: {path}
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt

text_edit = QTextEdit()
text_edit.setReadOnly(False)
text_edit.setStyleSheet(\"\"\"
    QTextEdit {{{{
        background-color: rgba(255, 255, 255, 80);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        border: none;
        color: black;
    }}}}
\"\"\")
text_edit.setPlainText('''{content_escaped}''')
text_edit.setParent(container)
text_edit.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=text_edit, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize
container.text_edit = text_edit
container.file_path = '{path_escaped}'
"""

    def _on_content_loaded(self, content_type, code):
        """Qt main thread: receive content from background thread and display it."""
        if content_type in ('empty', 'error'):
            try:
                err_code = generate_message_display(code, content_type.capitalize())
                self._exec_code_in_graphical_pane(err_code)
                self.pane_stack.setCurrentIndex(1)
            except Exception:
                self.text_pane.setPlainText(code)
                self.pane_stack.setCurrentIndex(0)
            return
        if is_executable_code(code):
            self.execute_code_from_fs(code)
        else:
            self.text_pane.setPlainText(code)
            self.pane_stack.setCurrentIndex(0)

    def _load_content_sync(self, path):
        """Synchronous content loading for non-9P paths (runs on main thread)."""
        ct = detect_content_type(path)
        if ct is None:
            if self._path_isdir(path):
                ct = 'directory'
            elif self._path_isfile(path):
                ct = 'text'
        code = None
        if ct == 'directory': code = generate_directory_listing(path)
        elif ct == 'image': code = generate_image_viewer(path)
        elif ct == 'video': code = generate_video_player(path)
        elif ct == 'audio': code = generate_audio_player(path)
        elif ct == '3d': code = generate_3d_viewer(path)
        elif ct == 'pdf': code = generate_pdf_viewer(path)
        elif ct == 'text' or (path and self._path_isfile(path)):
            code = generate_file_content(path)
        if code:
            if is_executable_code(code): self.execute_code_from_fs(code)
            else:
                self.text_pane.setPlainText(code)
                self.pane_stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Agent / AI
    # ------------------------------------------------------------------

    def _setup_agent(self):
        """Set up the LLM agent for this window.
        Runs in a background thread to avoid deadlocking on /n/ FUSE calls."""
        thread = threading.Thread(target=self._setup_agent_bg, daemon=True)
        thread.start()

    def _setup_agent_bg(self):
        ctl = os.path.join(self.llmfs_mount, "ctl")
        if not os.path.isdir(self.agent_path):
            try:
                with open(ctl, 'w') as f: f.write(f"new {self.agent_name}\n")
            except Exception: return
        try:
            sp = os.path.join(self.agent_path, "system")
            pf = "./systems/acme.md"
            txt = open(pf).read() if os.path.exists(pf) else (
                "You are an Acme window agent. Generate PySide6/Qt widget code "
                "in ```acme fences. Use `container` (a QWidget with NO layout — free canvas). "
                "Place widgets with setParent(container) and setGeometry(x, y, w, h). "
                "Use `canvas_size` tuple for available (width, height).")
            with open(sp, 'w') as f: f.write(txt)
        except Exception: pass
        try:
            rp = os.path.join(self.agent_path, "rules")
            with open(rp, 'w') as f:
                f.write(r"```(?P<acme>acme)\n(?P<code>.*?)```" + " -> {acme}\n")
        except Exception: pass
        src = os.path.join(self.agent_path, "acme")
        dst = f"{self.rio_mount}/acme/{self.window_id}/exec"
        try: self._add_route(src, dst)
        except Exception: pass
        if self.accumulated_code: self._feed_history(self.accumulated_code)

    def _feed_history(self, code):
        """Feed code history to agent. Safe to call from any thread."""
        if not os.path.isdir(self.agent_path): return
        try:
            hp = os.path.join(self.agent_path, "history")
            with open(hp, 'w') as f:
                f.write(json.dumps({"role": "assistant",
                    "content": f"Currently running code:\n```acme\n{code}\n```"}))
        except Exception: pass

    def call_ai(self, prompt):
        """Send prompt to AI agent. Runs I/O in background thread."""
        def _call_ai_bg():
            if not os.path.isdir(self.agent_path):
                self._stream_append.emit(f"# Agent not found at {self.agent_path}")
                return
            if self.accumulated_code: self._feed_history(self.accumulated_code)
            try:
                with open(os.path.join(self.agent_path, "input"), 'w') as f: f.write(prompt)
            except Exception as e:
                self._stream_append.emit(f"# Error: {e}")
                return
            self._stream_agent_output(os.path.join(self.agent_path, "OUTPUT"))
        # Create a transparent streaming widget in the graphical pane
        self._setup_stream_widget()
        self._stream_thread = threading.Thread(target=_call_ai_bg, daemon=True)
        self._stream_thread.start()

    def _setup_stream_widget(self):
        """Create a transparent QTextEdit in the graphical pane for AI streaming."""
        code = '''
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt

text_edit = QTextEdit()
text_edit.setReadOnly(True)
text_edit.setStyleSheet("""
    QTextEdit {
        background-color: rgba(255, 255, 255, 0);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        border: none;
        color: black;
    }
""")
text_edit.setParent(container)
text_edit.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=text_edit, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize
container.text_edit = text_edit
'''
        try:
            self._exec_code_in_graphical_pane(code)
            self.pane_stack.setCurrentIndex(1)
            # Keep a reference to the stream widget for appending
            ec = self.graphical_pane_layout.itemAt(
                self.graphical_pane_layout.count() - 1).widget()
            self._stream_widget = ec.text_edit if hasattr(ec, 'text_edit') else None
        except Exception:
            # Fallback to text pane
            self.text_pane.setPlainText("")
            self.pane_stack.setCurrentIndex(0)
            self._stream_widget = None

    def _stream_agent_output(self, path):
        import re; full = ""
        try:
            with open(path, 'r') as f:
                while True:
                    chunk = f.read(256)
                    if not chunk: break
                    full += chunk; self._stream_append.emit(chunk)
        except Exception as e:
            self._stream_append.emit(f"\n# Stream error: {e}\n"); return
        for m in re.finditer(r'```acme\s*\n(.*?)```', full, re.DOTALL):
            c = m.group(1).strip()
            if c: self._stream_exec.emit(c)

    def _on_stream_append(self, text):
        w = getattr(self, '_stream_widget', None) or self.text_pane
        c = w.textCursor(); c.movePosition(QTextCursor.End)
        c.insertText(text); w.setTextCursor(c)
        w.ensureCursorVisible()

    def _on_stream_exec(self, code):
        self.execute_code_from_fs(code)

    def _clear_conversation_history(self):
        def _clear():
            try:
                with open(os.path.join(self.agent_path, "ctl"), 'w') as f: f.write("clear")
            except Exception: pass
        threading.Thread(target=_clear, daemon=True).start()

    def cleanup(self):
        self._stop_repaint_driver()
        self._remove_route()

    # ------------------------------------------------------------------
    # Terminal mode (Plan 9 acme 'win' style)
    # ------------------------------------------------------------------

    def init_terminal(self, working_dir):
        """Turn this window into a Plan 9 acme-style terminal ('win').
        Uses the text pane only — all text is editable, commands run
        from the last line on Enter, mid-click exec copies text to the
        last line and runs it."""
        self.is_terminal = True
        self.working_dir = os.path.abspath(working_dir)
        self._term_executing = False

        # Tag line
        self.tag_line.setPlainText(f"{self.working_dir} Del Term")
        self._fit_tag_height()

        # Stay on text pane
        self.pane_stack.setCurrentIndex(0)

        # Show initial prompt
        self._term_prompt()

        # Install key filter on text pane for Enter handling
        self.text_pane.installEventFilter(self)

    def _term_prompt(self):
        """Append the cwd% prompt at the end of the text pane."""
        c = self.text_pane.textCursor()
        c.movePosition(QTextCursor.End)
        text = self.text_pane.toPlainText()
        if text and not text.endswith('\n'):
            c.insertText('\n')
        c.insertText(f"{self.working_dir}% ")
        self.text_pane.setTextCursor(c)
        self.text_pane.ensureCursorVisible()

    def _term_get_last_line_command(self):
        """Extract command from the last line after the % prompt."""
        text = self.text_pane.toPlainText()
        lines = text.split('\n')
        if not lines:
            return ""
        last = lines[-1]
        if '% ' in last:
            return last.split('% ', 1)[1]
        return last.strip()

    def _term_run(self, command):
        """Run a command in the terminal's working directory."""
        command = command.strip()
        if not command:
            return
        if self._term_executing:
            self._term_append("\n[busy]\n")
            return

        # Handle cd specially
        if command.startswith('cd '):
            path = command[3:].strip()
            if path.startswith('~'):
                path = os.path.expanduser(path)
            elif not os.path.isabs(path):
                path = os.path.join(self.working_dir, path)
            path = os.path.abspath(path)
            if os.path.isdir(path):
                self.working_dir = path
                self.tag_line.setPlainText(f"{self.working_dir} Del Term")
                self._fit_tag_height()
            else:
                self._term_append(f"\ncd: {path}: no such directory")
            self._term_prompt()
            return

        if command == 'cd':
            self.working_dir = os.path.expanduser('~')
            self.tag_line.setPlainText(f"{self.working_dir} Del Term")
            self._fit_tag_height()
            self._term_prompt()
            return

        self._term_executing = True

        def run():
            try:
                result = subprocess.run(
                    command, shell=True, cwd=self.working_dir,
                    capture_output=True, text=True, timeout=30)
                out = result.stdout
                if result.stderr:
                    out += result.stderr
                if not out:
                    if result.returncode != 0:
                        out = f"[exit {result.returncode}]"
                self._term_output.emit(out)
            except subprocess.TimeoutExpired:
                self._term_output.emit("\n[timeout after 30s]\n")
            except Exception as e:
                self._term_output.emit(f"\n[error: {e}]\n")

        threading.Thread(target=run, daemon=True).start()

    def _on_term_output(self, text):
        """Receive command output (main thread) and append + prompt."""
        if text:
            self._term_append(text)
        self._term_executing = False
        self._term_prompt()

    def _term_append(self, text):
        """Append text at the end of the text pane."""
        c = self.text_pane.textCursor()
        c.movePosition(QTextCursor.End)
        if text and not text.startswith('\n'):
            c.insertText('\n')
        c.insertText(text)
        self.text_pane.setTextCursor(c)
        self.text_pane.ensureCursorVisible()

    def _term_exec_selection(self, text):
        """Mid-click exec in terminal: append text to last line and run.
        Like Plan 9 acme — copies selection, appends after prompt, executes."""
        cmd = text.strip()
        if not cmd:
            return
        # Move to end, ensure we're on a prompt line
        c = self.text_pane.textCursor()
        c.movePosition(QTextCursor.End)
        full = self.text_pane.toPlainText()
        # If last line doesn't have a prompt, add one
        lines = full.split('\n')
        last = lines[-1] if lines else ""
        if '% ' not in last:
            self._term_prompt()
        # Append command text after prompt
        c = self.text_pane.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertText(cmd)
        self.text_pane.setTextCursor(c)
        self.text_pane.ensureCursorVisible()
        # Run it
        self._term_append("")
        self._term_run(cmd)

    def get_parent_column(self):
        p = self.parent()
        while p:
            from .acme_column import AcmeColumn
            if isinstance(p, AcmeColumn): return p
            p = p.parent() if hasattr(p, 'parent') else None
        return None

    def get_acme_parent(self):
        p = self.parent()
        while p:
            from .acme_core import Acme
            if isinstance(p, Acme): return p
            p = p.parent() if hasattr(p, 'parent') else None
        return None

    def find_column_at_pos(self, gpos):
        acme = self.get_acme_parent()
        if not acme: return None
        for col in acme.columns:
            r = col.container.rect()
            r.moveTo(col.container.mapToGlobal(QPoint(0, 0)))
            if r.contains(gpos): return col
        return None

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def plumb(self, text):
        full = text.rstrip('/')  # normalize trailing slash but keep path
        if not os.path.isabs(full):
            if self.path:
                # For /n/ paths, skip _path_isdir check to avoid deadlock
                if _is_9p_path(self.path):
                    base = os.path.dirname(self.path) if os.path.splitext(self.path)[1] else self.path
                else:
                    base = self.path if self._path_isdir(self.path) else os.path.dirname(self.path)
                full = os.path.join(base, full)
            else:
                full = os.path.abspath(full)
        # For /n/ paths, always open — load_content handles errors async
        if _is_9p_path(full) or self._path_exists(full):
            col = self.get_parent_column()
            if col:
                acme = self.get_acme_parent()
                tc = acme.preferred_column if acme and acme.preferred_column else col
                nw = tc.add_window(full)
                if nw and nw.text_pane:
                    QTimer.singleShot(50, lambda w=nw: self._warp_to_start(w))

    def _warp_to_start(self, w):
        tp = w.text_pane
        if tp and tp.toPlainText(): self._warp_cursor(tp, 0)
        elif tp: self._warp_cursor(w.tag_line, 0)

    # ------------------------------------------------------------------
    # Search (with cursor warp + selection of found word)
    # ------------------------------------------------------------------

    def search_forward(self, text):
        self._search_in(self.text_pane, text)

    def search_forward_in_widget(self, tw, text):
        self._search_in(tw, text)

    def _search_in(self, w, text):
        if not text: return
        c = w.textCursor()
        pos = c.selectionEnd() if c.hasSelection() else c.position()
        full = w.toPlainText()
        idx = full.find(text, pos)
        if idx == -1: idx = full.find(text, 0)
        if idx != -1:
            c.setPosition(idx)
            c.setPosition(idx + len(text), QTextCursor.KeepAnchor)
            w.setTextCursor(c)
            w.setFocus()
            self._warp_cursor(w, idx)

    # ------------------------------------------------------------------
    # Path / word detection
    # ------------------------------------------------------------------

    def detect_path_at_cursor_widget(self, tw, cpos):
        cur = tw.cursorForPosition(cpos)
        text = tw.toPlainText(); pos = cur.position()
        if pos < 0 or pos >= len(text): return None
        s = pos
        while s > 0 and (text[s-1].isalnum() or text[s-1] in '/-_.~:+'): s -= 1
        e = pos
        while e < len(text) and (text[e].isalnum() or text[e] in '/-_.~:+'): e += 1
        if s >= e: return None
        pt = text[s:e]
        while pt and pt[-1] in ',:;!?)': pt = pt[:-1]; e -= 1
        if not pt: return None
        if '/' in pt or pt.startswith('./') or pt.startswith('../'):
            return (pt, s, e)
        # For /n/ paths, skip _path_isdir/_path_exists probes (deadlock risk)
        if self.path and _is_9p_path(self.path):
            # Assume current path is a directory if it has no extension
            if not os.path.splitext(self.path)[1]:
                return (pt, s, e)
        elif self.path and self._path_isdir(self.path):
            full = os.path.join(self.path, pt.rstrip('/'))
            if self._path_exists(full):
                return (pt, s, e)
        return None

    def extract_word_at_cursor_widget(self, tw, cpos):
        cur = tw.cursorForPosition(cpos)
        text = tw.toPlainText(); pos = cur.position()
        if pos < 0 or pos >= len(text): return None
        s = pos
        while s > 0 and text[s-1].isalnum(): s -= 1
        e = pos
        while e < len(text) and text[e].isalnum(): e += 1
        if s >= e: return None
        return (text[s:e], s, e)

    def _mark_column_active(self):
        """Mark this window's column as the last-used (preferred) column.
        Called on any mouse interaction so that right-click plumb opens
        new windows in the column the user last touched."""
        col = self.get_parent_column()
        if col:
            acme = self.get_acme_parent()
            if acme:
                acme.set_preferred_column(col)

    def _resolve_text_widget(self, obj):
        if obj == self.text_pane.viewport() or obj == self.text_pane:
            return self.text_pane
        for i in range(self.graphical_pane_layout.count()):
            w = self.graphical_pane_layout.itemAt(i).widget()
            if w and hasattr(w, 'text_edit'):
                if obj in (w.text_edit.viewport(), w.text_edit): return w.text_edit
            if w and hasattr(w, 'terminal'):
                if obj in (w.terminal.viewport(), w.terminal): return w.terminal
        return None

    # ==================================================================
    # EVENT FILTER
    # ==================================================================

    def eventFilter(self, obj, event):
        if not hasattr(self, 'text_pane') or self.text_pane is None: return False
        if not hasattr(self, 'tag_line') or self.tag_line is None: return False
        if not hasattr(self, 'drag_handle') or self.drag_handle is None: return False

        # Any mouse press in this window → mark its column as last-used
        if event.type() == QEvent.MouseButtonPress:
            self._mark_column_active()

        # Terminal mode: intercept Enter key on text_pane
        if self.is_terminal and obj == self.text_pane and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                cmd = self._term_get_last_line_command()
                # Move cursor to end, add newline
                c = self.text_pane.textCursor()
                c.movePosition(QTextCursor.End)
                self.text_pane.setTextCursor(c)
                if cmd:
                    self._term_append("")
                    self._term_run(cmd)
                else:
                    self._term_prompt()
                return True

        if obj == self.drag_handle:
            return self._ev_drag(event)

        if obj == self.tag_line.viewport():
            return self._ev_tag(event)

        tw = self._resolve_text_widget(obj)
        if tw:
            return self._ev_text(tw, event)

        if event.type() == QEvent.ContextMenu:
            return True
        return super().eventFilter(obj, event)

    # --- Drag ---

    def _ev_drag(self, ev):
        if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            self.drag_active = True
            self.drag_start_global_pos = ev.globalPosition().toPoint()
            self.drag_start_window_y = self.y()
            col = self.get_parent_column()
            if col:
                acme = self.get_acme_parent()
                if acme: acme.set_preferred_column(col)
            return True
        elif ev.type() == QEvent.MouseMove and self.drag_active:
            dy = ev.globalPosition().toPoint().y() - self.drag_start_global_pos.y()
            col = self.get_parent_column()
            if col: col.reposition_window_during_drag(self, self.drag_start_window_y + dy)
            return True
        elif ev.type() == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
            if self.drag_active:
                self.drag_active = False
                gp = ev.globalPosition().toPoint()
                tc = self.find_column_at_pos(gp)
                cc = self.get_parent_column()
                if tc and tc != cc:
                    if cc: cc.remove_window_without_delete(self)
                    tc.insert_window_at_position(self, tc.container.mapFromGlobal(gp).y())
                    # Window landed on new column — mark it as last-used
                    self._mark_column_active()
                else:
                    if cc: cc.finalize_window_position(self)
            return True
        return False

    # --- Tag line (mid=red exec, right=green search) ---

    def _ev_tag(self, ev):
        et = ev.type()

        if et == QEvent.MouseButtonPress and ev.button() == Qt.MiddleButton:
            self._set_sel_color(self.tag_line, SEL_RED)
            c = self.tag_line.cursorForPosition(ev.pos())
            if not self.tag_line.textCursor().hasSelection():
                c.select(QTextCursor.WordUnderCursor)
            self.tag_line.setTextCursor(c)
            self._sel_start_pos = c.anchor()
            return True

        if et == QEvent.MouseMove and self._sel_start_pos is not None:
            # Extend mid/right selection in tag line
            btns = ev.buttons()
            if btns & Qt.MiddleButton or btns & Qt.RightButton:
                cur = self.tag_line.cursorForPosition(ev.pos())
                c = self.tag_line.textCursor()
                c.setPosition(self._sel_start_pos)
                c.setPosition(cur.position(), QTextCursor.KeepAnchor)
                self.tag_line.setTextCursor(c)
                return True

        if et == QEvent.MouseButtonRelease and ev.button() == Qt.MiddleButton:
            cmd = self.tag_line.textCursor().selectedText().strip()
            self._set_sel_color(self.tag_line, SEL_BLUE)
            self._sel_start_pos = None

            # "ai ..." grabs rest of line
            if cmd.lower() == "ai":
                c2 = self.tag_line.cursorForPosition(ev.pos())
                c2.select(QTextCursor.LineUnderCursor)
                line = c2.selectedText()
                if line.lower().startswith("ai "): cmd = line.strip()

            if cmd: self.execute_ctl_command(cmd)
            return True

        if et == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
            self._set_sel_color(self.tag_line, SEL_GREEN)
            c = self.tag_line.cursorForPosition(ev.pos())
            if not self.tag_line.textCursor().hasSelection():
                # Try to detect a path at cursor position first
                dp = self.detect_path_at_cursor_widget(self.tag_line, ev.pos())
                if dp:
                    pt, s, e = dp
                    c.setPosition(s)
                    c.setPosition(e, QTextCursor.KeepAnchor)
                else:
                    c.select(QTextCursor.WordUnderCursor)
            self.tag_line.setTextCursor(c)
            self._sel_start_pos = c.anchor()
            return True

        if et == QEvent.MouseButtonRelease and ev.button() == Qt.RightButton:
            word = self.tag_line.textCursor().selectedText().strip()
            self._set_sel_color(self.tag_line, SEL_BLUE)
            self._sel_start_pos = None
            if word:
                # Check if the selected text is a plumbable path
                resolved = self._is_plumbable_path(word)
                if resolved:
                    self.plumb(resolved)
                else:
                    self.search_forward(word)
            return True

        if et == QEvent.MouseButtonPress:
            self.tag_line.setFocus()
            self.tag_line.setTextCursor(self.tag_line.cursorForPosition(ev.pos()))
            return False

        if et == QEvent.ContextMenu:
            return True
        return False

    # --- Text widget (full Plan 9 three-button) ---

    def _ev_text(self, tw, ev):
        et = ev.type()

        # ---- PRESS ----
        if et == QEvent.MouseButtonPress:
            btn = ev.button()

            # CHORDING: second button while left held
            if self._left_held and self._active_button == Qt.LeftButton:
                if btn == Qt.MiddleButton:
                    if tw.textCursor().hasSelection(): tw.cut()
                    self._active_button = None
                    return True
                elif btn == Qt.RightButton:
                    tw.paste()
                    self._active_button = None; self._left_held = False
                    return True

            self._active_button = btn
            self._sel_start_pos = tw.cursorForPosition(ev.pos()).position()

            if btn == Qt.LeftButton:
                self._left_held = True
                self._set_sel_color(tw, SEL_BLUE)
                return False  # let Qt handle native selection

            elif btn == Qt.MiddleButton:
                self._set_sel_color(tw, SEL_RED)
                c = tw.cursorForPosition(ev.pos())
                if not tw.textCursor().hasSelection():
                    c.select(QTextCursor.WordUnderCursor)
                tw.setTextCursor(c)
                return True

            elif btn == Qt.RightButton:
                self._set_sel_color(tw, SEL_GREEN)
                tw.setTextCursor(tw.cursorForPosition(ev.pos()))
                return True

        # ---- DRAG ----
        elif et == QEvent.MouseMove:
            if self._active_button in (Qt.MiddleButton, Qt.RightButton) and self._sel_start_pos is not None:
                cur = tw.cursorForPosition(ev.pos())
                c = tw.textCursor()
                c.setPosition(self._sel_start_pos)
                c.setPosition(cur.position(), QTextCursor.KeepAnchor)
                tw.setTextCursor(c)
                return True
            return False

        # ---- RELEASE ----
        elif et == QEvent.MouseButtonRelease:
            btn = ev.button()

            if btn == Qt.LeftButton:
                self._left_held = False
                if self._active_button == Qt.LeftButton:
                    self._active_button = None
                    self._set_sel_color(tw, SEL_BLUE)
                return False

            elif btn == Qt.MiddleButton and self._active_button == Qt.MiddleButton:
                self._active_button = None
                self._set_sel_color(tw, SEL_BLUE)

                text = tw.textCursor().selectedText().strip()

                # Terminal mode: copy selection to prompt, run it
                if self.is_terminal and tw == self.text_pane:
                    if text:
                        self._term_exec_selection(text)
                    return True

                # "ai ..." grabs rest of line
                if text.lower() == "ai":
                    c2 = tw.cursorForPosition(ev.pos())
                    c2.select(QTextCursor.LineUnderCursor)
                    line = c2.selectedText()
                    if line.lower().startswith('ai '):
                        self.execute_ctl_command(line.strip()); return True
                if text: self.execute_ctl_command(text)
                return True

            elif btn == Qt.RightButton and self._active_button == Qt.RightButton:
                self._active_button = None
                self._set_sel_color(tw, SEL_BLUE)

                sel = tw.textCursor().selectedText().strip()
                if sel:
                    self.plumb(sel)
                else:
                    dp = self.detect_path_at_cursor_widget(tw, ev.pos())
                    if dp:
                        pt, s, e = dp
                        c = tw.textCursor(); c.setPosition(s)
                        c.setPosition(e, QTextCursor.KeepAnchor)
                        tw.setTextCursor(c)
                        self.plumb(pt)
                    else:
                        dw = self.extract_word_at_cursor_widget(tw, ev.pos())
                        if dw:
                            wt, s, e = dw
                            c = tw.textCursor(); c.setPosition(s)
                            c.setPosition(e, QTextCursor.KeepAnchor)
                            tw.setTextCursor(c)
                            self.search_forward_in_widget(tw, wt)
                return True

            return False

        if et == QEvent.ContextMenu:
            return True
        return False
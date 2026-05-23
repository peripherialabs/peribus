#!/usr/bin/env python3
"""
Rio Display Server

This version provides the core Rio display server functionality
with context-menu based LLMFS connectivity.
"""

import asyncio
import argparse
import colorsys
import logging
import random
import signal
import socket as _sock
import sys
import os
import glob
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ninep.server import Server9P
from rio.filesystem import RioRoot
from rio.scene import SceneManager
from rio.terminal_widget import TerminalWidget
from rio.parser import Executor, ExecutionContext
from rio.ai_voice_control import AIVoiceControlWidget
from rio.immersive_mode import install_immersive_mode
from rio.theme import get_theme, THEMES, DEFAULT_THEME_NAME, Theme

# Cross-machine signals. Optional — if the module is missing rio still
# runs, just without `subscribe()` / cross-machine `Signal`.
try:
    from rio.signals import (
        init_global_bus as _init_signal_bus,
        shutdown_global_bus as _shutdown_signal_bus,
        DEFAULT_SIGNAL_PORT_OFFSET as _SIGNAL_PORT_OFFSET,
    )
    _HAS_SIGNALS = True
except ImportError:
    _HAS_SIGNALS = False
    _SIGNAL_PORT_OFFSET = 100

logger = logging.getLogger(__name__)


class RioServer:
    """Rio display server"""
    
    def __init__(
        self,
        headless: bool = False,
        width: int = 3840,  # 4K width
        height: int = 2160,  # 4K height
        workspace: str = None,
        mux_mount: str = "/n/mux",
        fullscreen: bool = False,
        auth_manager=None,
    ):
        self.headless = headless
        self.fullscreen = fullscreen
        
        # Optional 9P token auth. If None or has no secrets, auth is off.
        # We hold onto the AuthManager so it can be wired into Server9P
        # at _initialize_filesystem() time (the server is built lazily).
        self.auth_manager = auth_manager
        
        # Mux-aware mount paths
        # When workspace is set (via --workspace), all paths go through the mux:
        #   llmfs_mount = /n/mux/llm
        #   rio_mount   = /n/mux/<workspace>
        # When workspace is None (standalone), use legacy paths:
        #   llmfs_mount = /n/llm
        #   rio_mount   = /n/rioa
        self.workspace = workspace
        self.mux_mount = mux_mount
        if workspace:
            self.llmfs_mount = f"{mux_mount}/llm"
            self.rio_mount = f"{mux_mount}/{workspace}"
        else:
            self.llmfs_mount = "/n/llm"
            self.rio_mount = "/n/rioa"
        
        # Create scene manager
        self.scene_manager = SceneManager()
        self.scene_manager.width = width
        self.scene_manager.height = height
        
        # Qt components (if not headless)
        self.app = None
        self.window = None
        self._running = False
                
        # Create filesystem (will be updated with Qt objects later)
        self.filesystem = None
        
        # Create 9P server (will be set later)
        self.server = None

        # Cross-machine signal bus. Lazily started in start_tcp /
        # start_unix once we know the bind port. `machine_name` is
        # what peers use in `subscribe("<machine_name>")` — workspace
        # name if we're in mux mode, hostname otherwise.
        self.machine_name = workspace or _sock.gethostname().split(".")[0]
        self.signal_port = None
        self.signal_bus = None
    
    def _initialize_filesystem(self):
        """Initialize filesystem with Qt objects"""
        qt_objects = {}
        
        if self.window:
            qt_objects['main_window'] = self.window
            qt_objects['graphics_scene'] = self.window.graphics_scene
            qt_objects['graphics_view'] = self.window.graphics_view
        
        # Create filesystem with Qt objects
        self.filesystem = RioRoot(self.scene_manager, qt_objects)
        
        # Create 9P server (with auth if configured)
        self.server = Server9P(self.filesystem, auth_manager=self.auth_manager)
    
    async def start_tcp(self, host: str = '0.0.0.0', port: int = 5641):
        """Start TCP server"""
        self._running = True
        
        print(f"Rio display server starting...")
        print(f"  Scene size: {self.scene_manager.width}x{self.scene_manager.height}")
        print(f"  Headless: {self.headless}")
        print(f"  Listening on: {host}:{port}")
        print()
        print(f"Mount with: mount -t 9p -o trans=tcp,port={port} localhost /n/rio")
        if self.workspace:
            print(f"  Mux workspace: {self.workspace}")
            print(f"  LLM mount:     {self.llmfs_mount}")
            print(f"  Rio mount:     {self.rio_mount}")
        print()
        
        if not self.headless:
            await self._start_qt()
        
        # Start the cross-machine signal bus before the filesystem is
        # built — the /scene/signals/ dir resolves the bus lazily, but
        # peers may sub the moment we accept 9P connections, so we
        # want the UDP socket bound by then. Convention:
        # signal_port = 9p_port + DEFAULT_SIGNAL_PORT_OFFSET (100).
        self._start_signal_bus(host=host, base_port=port)

        # Initialize filesystem after Qt is ready
        self._initialize_filesystem()
        
        # Start 9P server
        server_task = asyncio.create_task(
            self.server.serve_tcp(host, port)
        )
        
        if not self.headless:
            # Run Qt event loop
            await self._run_qt_loop()
        else:
            await server_task
    
    async def start_unix(self, path: str):
        """Start Unix socket server"""
        self._running = True
        
        print(f"Rio display server starting...")
        print(f"  Socket: {path}")
        print()
        
        if not self.headless:
            await self._start_qt()

        # Signal bus is transport-agnostic — works fine even when 9P
        # is on a Unix socket. We can't derive a TCP port here, so
        # fall back to OS-assigned (peers can read it from
        # /scene/signals/port). Bind on all interfaces so LAN peers
        # can still reach us.
        self._start_signal_bus(host="0.0.0.0", base_port=None)

        # Initialize filesystem after Qt is ready
        self._initialize_filesystem()
        
        server_task = asyncio.create_task(
            self.server.serve_unix(path)
        )
        
        if not self.headless:
            await self._run_qt_loop()
        else:
            await server_task
    
    def _start_signal_bus(self, host: str, base_port):
        """
        Initialize the rio.signals bus once we know the bind address.

        `base_port` is the 9P TCP port for the convention
        `signal_port = base_port + DEFAULT_SIGNAL_PORT_OFFSET` (100).
        When base_port is None (Unix-socket mode) the OS picks the
        UDP port — peers can still find it via /scene/signals/port.

        No-op if rio.signals failed to import.
        """
        if not _HAS_SIGNALS:
            return
        try:
            if base_port is not None:
                self.signal_port = base_port + _SIGNAL_PORT_OFFSET
            else:
                self.signal_port = None  # let the OS pick
            self.signal_bus = _init_signal_bus(
                machine_name=self.machine_name,
                bind_host=host,
                bind_port=self.signal_port,
                # In mux mode the mux's ctl listing lives at
                # /n/mux/ctl (the mux is mounted under mux_mount).
                # Otherwise the registry is the top-level /n/ctl from
                # whatever riomux mounted there.
                mux_ctl_path=f"{self.mux_mount}/ctl" if self.workspace else "/n/ctl",
            )
            # If we asked for OS-assigned, surface whatever we got so
            # the print line below is accurate.
            if self.signal_port is None:
                self.signal_port = self.signal_bus.bind_port
            print(f"  Signal bus: UDP {host}:{self.signal_port}  "
                  f"(machine={self.machine_name})")
        except Exception as e:
            logger.warning(f"Could not start signal bus: {e}")
            self.signal_bus = None

    async def stop(self):
        """Stop the server.

        Made idempotent because we can be called from multiple paths:
        the signal handler in main(), the Qt closeEvent, and the post-
        loop shutdown. Calling server.stop() twice would raise on the
        second wait_closed(); calling app.quit() twice is harmless but
        noisy.

        We schedule app.quit() via QTimer.singleShot rather than calling
        it directly: QApplication is owned by the Qt thread and any
        cross-thread call has to be marshalled. Since asyncio and Qt
        share a thread here it usually works, but the singleShot makes
        it correct regardless.
        """
        if not self._running and self.server is None and self.app is None:
            return  # Already stopped or never started

        self._running = False

        if self.server is not None:
            srv, self.server = self.server, None
            try:
                await srv.stop()
            except Exception as e:
                logger.warning(f"Error stopping 9P server: {e}")

        # Tear down the signal bus AFTER the 9P server — we want
        # peers to see 9P go quiet first, then the UDP `bye` packets
        # arrive as confirmation. `shutdown_global_bus` is idempotent
        # (no-op if no bus was ever started), so this is safe even
        # in standalone / unix-socket modes that skipped it.
        if _HAS_SIGNALS:
            try:
                _shutdown_signal_bus()
            except Exception as e:
                logger.warning(f"shutdown_global_bus failed: {e}")
            self.signal_bus = None

        if self.app is not None:
            try:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self.app.quit)
            except Exception:
                # Fallback if Qt is already torn down
                try:
                    self.app.quit()
                except Exception:
                    pass
    
    async def _start_qt(self):
        """Initialize Qt"""
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            logger.warning("PySide6 not available, running headless")
            self.headless = True
            return
        
        #self.app = QApplication(sys.argv)
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = RioWindow(self.scene_manager, self)
        if self.fullscreen:
            self.window.showFullScreen()
        else:
            self.window.show()
        
        print("✓ Qt window created")
        print(f"  • main_window available in code")
        print(f"  • graphics_scene available in code")
        print(f"  • graphics_view available in code")
        print(f"  • Right-click for context menu")
    
    def _connect_events(self):
        """Connect Qt events to filesystem"""
        if not self.window or not self.filesystem:
            return
        
        print("✓ Events connected")
    
    async def _run_qt_loop(self):
        """Run Qt event loop alongside asyncio.

        Preferred path: if qasync is installed, we don't poll at all —
        the Qt event loop IS the asyncio loop (installed in main() via
        qasync.QEventLoop), so this coroutine just awaits self._running
        going False. Zero idle CPU; Qt and asyncio share wakeups
        natively. Input latency drops because input events no longer
        wait up to 8 ms for the next poll.

        Fallback path (no qasync): poll processEvents at ~125 Hz. This
        is the historical behaviour, kept so the app still runs in
        environments without the optional dependency. The polling
        comment below explains why 8 ms specifically.

        Earlier this used asyncio.sleep(0.001), which woke ~1000×/sec
        forever and showed up as a permanent few-percent CPU floor.
        8 ms gives ~120 Hz responsiveness (well above human perception
        for input → repaint) at 1/8 the wakeup rate. qasync removes
        even that floor.
        """
        # Connect events after filesystem is ready
        self._connect_events()

        # Detect qasync mode: main() installs a qasync.QEventLoop as
        # the running asyncio loop, which means asyncio events ARE Qt
        # events. In that mode processEvents is unnecessary and harmful
        # (it can cause re-entrant event dispatch).
        loop = asyncio.get_running_loop()
        qasync_mode = loop.__class__.__module__.startswith('qasync')

        if qasync_mode:
            # Just wait until shutdown — Qt drives both halves.
            while self._running:
                await asyncio.sleep(0.25)
            return

        # Fallback: legacy poll loop
        while self._running:
            if self.app:
                self.app.processEvents()
            await asyncio.sleep(0.008)


# ============================================================================
# Qt Window
# ============================================================================

from PySide6.QtWidgets import (
    QMainWindow, QGraphicsView, QGraphicsScene,
    QWidget, QVBoxLayout, QMenu, QGraphicsProxyWidget,
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem,
    QApplication, QLabel, QTextEdit, QLineEdit
)
from PySide6.QtCore import (
    Qt, QRectF, QPoint, QPointF, QObject, QTimer, QEvent,
    QPropertyAnimation, QEasingCurve, QVariantAnimation
)
from PySide6.QtGui import (
    QColor, QBrush, QAction, QPen,
    QTransform, QCursor, QPainter, QWheelEvent
)

from PySide6.QtOpenGLWidgets import QOpenGLWidget


# ═══════════════════════════════════════════════════════════════════════════
# Debug Overlay Widget - top-right HUD on main window
# ═══════════════════════════════════════════════════════════════════════════

class _ClickableLabel(QLabel):
    """QLabel that emits clicked() on mouse press."""
    from PySide6.QtCore import Signal as _Signal
    clicked = _Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class DebugOverlayWidget(QWidget):
    """
    Semi-transparent debug output overlay pinned to the top-right corner of
    the main RioWindow.  NOT part of the graphics scene — stays fixed on
    screen during pan/zoom.

    Features:
      - Collapsible: click the title bar to expand / collapse the body.
      - push_message(tag, content): append a tagged message.
      - clear_messages(): wipe all messages.
      - Auto-scrolls to the newest message.
      - Hidden by default; shown when a DebugNode has connections.
      - Each message is tagged with its source input port.
    """

    MAX_MESSAGES = 200
    EXPANDED_WIDTH = 440
    EXPANDED_HEIGHT = 360
    TITLE_HEIGHT = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setVisible(False)
        self._collapsed = False
        # Messages are stored as document blocks in self._text now —
        # see push_message and the maximumBlockCount setup below.

        # --- Outer layout (no margins, stacks title + body) ---
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Title bar (clickable) ---
        self._title = _ClickableLabel("  🐛 Debug  ▾")
        self._title.setFixedHeight(self.TITLE_HEIGHT)
        self._title.setCursor(Qt.PointingHandCursor)
        self._title.setStyleSheet("""
            QLabel {
                background-color: rgba(180, 58, 58, 230);
                color: #ffffff;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 12px;
                font-weight: 600;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding-left: 8px;
            }
            QLabel:hover {
                background-color: rgba(200, 70, 70, 240);
            }
        """)
        self._title.clicked.connect(self._toggle_collapsed)
        outer.addWidget(self._title)

        # --- Body: read-only QTextEdit (handles scroll + word-wrap natively) ---
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setAcceptRichText(True)
        self._text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Cap the document at MAX_MESSAGES blocks — Qt will silently drop
        # the oldest block when a new one arrives past the cap. This
        # replaces the old "rebuild full HTML on every push" pattern,
        # which was O(N) per append and O(N²) cumulative — a 200-message
        # buffer with 2 KB/message rebuilds a 400 KB blob each call.
        self._text.document().setMaximumBlockCount(self.MAX_MESSAGES)
        self._text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(24, 24, 30, 225);
                color: #d4d4d4;
                font-family: 'Consolas', 'Source Code Pro', 'Courier New', monospace;
                font-size: 11px;
                border: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                padding: 6px 8px;
                selection-background-color: rgba(68, 130, 255, 120);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 7px;
                margin: 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 50);
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 90);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        outer.addWidget(self._text)

        # Apply expanded size
        self._apply_size()

    # ── Collapse / Expand ────────────────────────────────────────────────

    def _toggle_collapsed(self):
        self._collapsed = not self._collapsed
        self._text.setVisible(not self._collapsed)
        self._title.setText(
            "  🐛 Debug  ▸" if self._collapsed else "  🐛 Debug  ▾"
        )
        # Update rounded corners on title when body is hidden
        if self._collapsed:
            self._title.setStyleSheet("""
                QLabel {
                    background-color: rgba(180, 58, 58, 230);
                    color: #ffffff;
                    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                    font-size: 12px;
                    font-weight: 600;
                    border-radius: 8px;
                    padding-left: 8px;
                }
                QLabel:hover {
                    background-color: rgba(200, 70, 70, 240);
                }
            """)
        else:
            self._title.setStyleSheet("""
                QLabel {
                    background-color: rgba(180, 58, 58, 230);
                    color: #ffffff;
                    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                    font-size: 12px;
                    font-weight: 600;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    padding-left: 8px;
                }
                QLabel:hover {
                    background-color: rgba(200, 70, 70, 240);
                }
            """)
        self._apply_size()

    def _apply_size(self):
        if self._collapsed:
            self.setFixedSize(self.EXPANDED_WIDTH, self.TITLE_HEIGHT)
        else:
            self.setFixedSize(self.EXPANDED_WIDTH, self.EXPANDED_HEIGHT)
        # Re-anchor to top-right after size change
        parent = self.parentWidget()
        if parent:
            self.reposition(parent.width())

    # ── Message API ──────────────────────────────────────────────────────

    def push_message(self, tag: str, content: str):
        """Add a tagged debug message.

        Appends one block (paragraph) to the QTextEdit's document via a
        cursor. The document's maximumBlockCount handles the rolling
        window — Qt drops the oldest block automatically when the cap
        is reached, so this is O(1) per push regardless of buffer size.
        """
        import html as _html
        from PySide6.QtGui import QTextCursor

        if len(content) > 2000:
            content = content[:2000] + "…"

        safe_tag = _html.escape(tag)
        safe_content = _html.escape(content).replace('\n', '<br>')

        entry = (
            f'<span style="color:#e06c75; font-weight:600;">[{safe_tag}]</span> '
            f'<span style="color:#c8ccd4;">{safe_content}</span>'
        )

        # Append at the end. insertHtml on a fresh block keeps each
        # message as its own block so maximumBlockCount can roll them
        # off cleanly.
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not self._text.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(entry)

        # Auto-scroll to bottom
        QTimer.singleShot(20, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_messages(self):
        self._text.clear()

    def reposition(self, parent_width: int):
        """Anchor to the top-right corner of the parent widget."""
        margin = 14
        self.move(parent_width - self.width() - margin, margin)



# ═══════════════════════════════════════════════════════════════════════════
# App Launcher Widget — drag-and-drop app icons onto scene
# ═══════════════════════════════════════════════════════════════════════════

class _AppIconLabel(QLabel):
    """A single draggable app icon in the launcher."""

    def __init__(self, app_name: str, app_path: str, launcher: 'AppLauncherWidget', parent=None):
        super().__init__(parent)
        self.app_name = app_name
        self.app_path = app_path
        self.launcher = launcher
        self._drag_start = None
        self._drag_executed = False

        display = app_name.replace('_', ' ').replace('-', ' ')
        icon = self._pick_icon(app_name)
        self.setText(f"{icon}\n{display}")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setFixedSize(80, 72)
        self.setCursor(Qt.OpenHandCursor)
        self.setStyleSheet("""
            QLabel {
                color: #1a1a1a;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                border-radius: 6px;
                padding: 4px 2px;
                background: transparent;
            }
            QLabel:hover {
                background: rgba(0, 0, 0, 25);
            }
        """)

    @staticmethod
    def _pick_icon(name: str) -> str:
        n = name.lower()
        mapping = [
            (['terminal', 'shell', 'console', 'term'], '🖥'),
            (['chat', 'message', 'talk'], '💬'),
            (['image', 'photo', 'picture', 'camera', 'img'], '🖼'),
            (['music', 'audio', 'sound', 'synth'], '🎵'),
            (['video', 'movie', 'film'], '🎬'),
            (['game', 'play'], '🎮'),
            (['chart', 'graph', 'plot', 'data', 'dashboard'], '📊'),
            (['note', 'text', 'edit', 'write', 'doc'], '📝'),
            (['web', 'browser', 'http', 'url'], '🌐'),
            (['map', 'geo', 'location'], '🗺'),
            (['clock', 'timer', 'time'], '⏱'),
            (['calc', 'math'], '🧮'),
            (['file', 'folder', 'dir'], '📁'),
            (['paint', 'draw', 'canvas', 'art'], '🎨'),
            (['cube', '3d', 'gl', 'render'], '🧊'),
            (['debug', 'log', 'monitor'], '🐛'),
            (['ai', 'llm', 'model', 'neural'], '🧠'),
            (['search', 'find', 'query'], '🔍'),
            (['settings', 'config', 'pref'], '⚙'),
        ]
        for keywords, icon in mapping:
            if any(k in n for k in keywords):
                return icon
        return '📦'

    # Use Qt's own drag threshold (platform-correct: 4px Win/Linux, 10px macOS)
    # rather than a hardcoded 8 — too small triggers spurious drags from jitter.
    _DRAG_SLOP = None

    def _drag_threshold(self):
        if _AppIconLabel._DRAG_SLOP is None:
            try:
                _AppIconLabel._DRAG_SLOP = QApplication.startDragDistance()
            except Exception:
                _AppIconLabel._DRAG_SLOP = 10
        return _AppIconLabel._DRAG_SLOP

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_executed = False  # set True iff QDrag.exec actually ran
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._drag_start is not None
                and not self._drag_executed
                and (event.buttons() & Qt.LeftButton)
                and (event.pos() - self._drag_start).manhattanLength() >= self._drag_threshold()):
            from PySide6.QtGui import QDrag
            from PySide6.QtCore import QMimeData
            self._drag_executed = True
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.app_path)
            mime.setData("application/x-rio-app", self.app_path.encode('utf-8'))
            drag.setMimeData(mime)
            # exec is modal; when it returns the drop has been handled
            # (or cancelled). Either way we should NOT also fire a click.
            drag.exec(Qt.CopyAction)
            self.setCursor(Qt.OpenHandCursor)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            start = self._drag_start
            drag_executed = self._drag_executed
            self._drag_start = None
            self._drag_executed = False
            self.setCursor(Qt.OpenHandCursor)
            # Click only if no drag was started. We don't need a distance
            # check here — if motion exceeded the threshold, mouseMoveEvent
            # would have set _drag_executed.
            if not drag_executed and start is not None:
                self.launcher.launch_app(self.app_path, self.app_name)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AppLauncherWidget(QWidget):
    """
    Floating icon grid showing available apps from ./apps/*.py.

    Launches apps by writing code to the filesystem's parse file
    (equivalent to: cat ./apps/app.py > /n/{workspace}/scene/parse),
    so the code goes through the filesystem executor and appears in CONTEXT.

    Animation:
      Open  → black border draws clockwise from top-left,
              then content fades in + shadow grows (0,0)→(45,45).
      Close → content fades out + shadow shrinks (45,45)→(0,0),
              then border erases counter-clockwise back to top-left.
    """

    # Animation phases
    _PHASE_IDLE = 0
    _PHASE_BORDER_IN = 1
    _PHASE_CONTENT_IN = 2
    _PHASE_VISIBLE = 3
    _PHASE_CONTENT_OUT = 4
    _PHASE_BORDER_OUT = 5
    _PHASE_DONE = 6

    def __init__(self, rio_window: 'RioWindow', parent=None):
        super().__init__(parent)
        self.rio_window = rio_window
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        # Animation state
        self._anim_phase = self._PHASE_IDLE
        self._anim_t = 0.0          # 0→1 progress within current phase
        self._anim_timer = None
        self._border_progress = 0.0  # 0→1 how much border is drawn
        self._content_opacity = 0.0  # 0→1
        self._shadow_effect = None
        self._proxy = None
        self._on_close_done = None   # callback when close animation finishes
        self._border_radius = 6

        # Cached perimeter path + total length, rebuilt lazily on size
        # change. Previously paintEvent allocated a fresh QPainterPath,
        # called length() (curve walk), and looped pointAtPercent up to
        # 400 times PER PAINT. With ~60 paints/sec that adds up.
        self._cached_path = None
        self._cached_path_size = None
        self._cached_path_len = 0.0

        self._build_ui()

        # Hide content children until border is drawn
        self._content_frame.setVisible(False)

    def _get_apps_dir(self) -> str:
        rio_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(rio_dir)
        return os.path.join(project_root, "apps")

    def _discover_apps(self) -> list:
        apps_dir = self._get_apps_dir()
        if not os.path.isdir(apps_dir):
            return []
        apps = []
        for filepath in sorted(glob.glob(os.path.join(apps_dir, "*.py"))):
            basename = os.path.basename(filepath)
            if basename.startswith('_'):
                continue
            name = basename[:-3]
            apps.append((name, filepath))
        return apps

    def _build_ui(self):
        from PySide6.QtWidgets import QGridLayout, QScrollArea

        apps = self._discover_apps()
        # Stash the full list — _populate_grid filters this on every keystroke.
        self._all_apps = apps

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # All visible content goes in this frame
        self._content_frame = QWidget()
        self._content_frame.setAttribute(Qt.WA_TranslucentBackground, True)
        frame_layout = QVBoxLayout(self._content_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        title = QLabel("  📦 Apps")
        title.setFixedHeight(28)
        title.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 160);
                color: #000000;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                font-weight: 600;
                border: none;
                padding-left: 8px;
            }
        """)
        frame_layout.addWidget(title)

        # Search bar — filters the icon grid live as the user types.
        # Sits between the title strip and the body, matching the body's
        # translucent fill so it reads as part of the same surface.
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search apps…")
        self._search_edit.setFixedHeight(26)
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 140);
                color: #1a1a1a;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                border: none;
                border-bottom: 1px solid rgba(0, 0, 0, 40);
                padding: 2px 8px;
                selection-background-color: rgba(0, 0, 0, 60);
            }
            QLineEdit:focus {
                background-color: rgba(255, 255, 255, 180);
                border-bottom: 1px solid rgba(0, 0, 0, 90);
            }
        """)
        self._search_edit.textChanged.connect(self._populate_grid)
        frame_layout.addWidget(self._search_edit)

        body = QWidget()
        body.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 120);
                border: none;
            }
        """)
        grid = QGridLayout(body)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)
        # Hold references for _populate_grid to mutate later.
        self._grid_body = body
        self._grid_layout = grid

        # Initial population (no filter).
        self._populate_grid("")

        # Scroll area lets the icon grid grow past the launcher's max height
        # without expanding the launcher itself. Vertical scroll only — the
        # column count is fixed, so horizontal scroll would be a bug surface.
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setAttribute(Qt.WA_TranslucentBackground, True)
        scroll.viewport().setAutoFillBackground(False)
        scroll.setStyleSheet("""
            QScrollArea, QScrollArea > QWidget > QWidget {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 2px 2px 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 80);
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 0, 0, 130);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
                background: none;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        # The scroll viewport itself needs the body's translucent fill —
        # otherwise the area behind a partially-filled grid shows the
        # window's default opaque colour.
        scroll_holder = QWidget()
        scroll_holder.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 120);
                border: none;
            }
        """)
        holder_layout = QVBoxLayout(scroll_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(0)
        holder_layout.addWidget(scroll)

        frame_layout.addWidget(scroll_holder)
        outer.addWidget(self._content_frame)

        # Sizing: width is determined by the column count (unchanged).
        # Height: prefer to show all rows up to MAX_HEIGHT; beyond that the
        # scrollbar takes over. Size is based on the FULL app list so the
        # launcher doesn't resize as the user types in the search bar.
        # The +26 below is the search bar height added in this commit.
        MAX_HEIGHT = 420
        n = len(self._all_apps) if self._all_apps else 1
        cols = max(1, min(4, n))
        rows = (n + cols - 1) // cols
        w = cols * 88 + 16
        natural_h = 28 + 26 + rows * 80 + 16
        self.setFixedSize(max(w, 160), min(natural_h, MAX_HEIGHT))

    def _populate_grid(self, filter_text: str = ""):
        """
        Rebuild the icon grid, showing only apps whose name matches
        `filter_text` (case-insensitive substring match on both the raw
        filename stem and the display form with separators replaced).

        Called from _build_ui for the initial fill and from the search
        bar's textChanged signal on every keystroke. Cheap because the
        app set is small (tens of items at most) — full teardown +
        rebuild keeps the layout code in one place rather than juggling
        per-widget show/hide and remembering grid positions.
        """
        grid = self._grid_layout

        # Clear existing items. takeAt(0) repeatedly is the idiomatic
        # way to drain a QLayout without leaking widgets — each removed
        # item's widget is reparented out and scheduled for deletion.
        while grid.count():
            item = grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        q = (filter_text or "").strip().lower()

        def _matches(name: str) -> bool:
            if not q:
                return True
            n = name.lower()
            # Match against both the raw name ("voice_chat") and the
            # display form ("voice chat") so users can type either.
            display = n.replace('_', ' ').replace('-', ' ')
            return q in n or q in display

        filtered = [(name, path) for (name, path) in self._all_apps if _matches(name)]

        if not self._all_apps:
            empty = QLabel("No apps found\nin ./apps/")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: rgba(0,0,0,100); font-size: 11px;")
            grid.addWidget(empty, 0, 0)
            return

        if not filtered:
            empty = QLabel(f"No matches for\n“{filter_text}”")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: rgba(0,0,0,100); font-size: 11px;")
            grid.addWidget(empty, 0, 0)
            return

        # Use the same column count as the unfiltered layout so icons
        # don't reflow into different column positions while the user is
        # typing — the visual jitter would be distracting.
        cols = max(1, min(4, len(self._all_apps)))
        for idx, (name, path) in enumerate(filtered):
            icon = _AppIconLabel(name, path, self)
            grid.addWidget(icon, idx // cols, idx % cols)

    # ── Paint: animated border + translucent fill ────────────────────

    def paintEvent(self, event):
        """Draw the border as a partial rounded-rect path based on _border_progress."""
        if self._border_progress <= 0.0:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r = self._border_radius
        inset = 0.5  # half-pixel inset so the 1px stroke lands inside

        # Fill — white translucent, clipped to the same rounded rect
        if self._content_opacity > 0.01:
            fill_alpha = int(140 * self._content_opacity)
            p.setBrush(QBrush(QColor(255, 255, 255, fill_alpha)))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(inset, inset, w - 2 * inset, h - 2 * inset), r, r)

        # Border stroke
        pen = QPen(QColor(0, 0, 0, 220), 1.0)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Build (or reuse) the full rounded-rect perimeter as a QPainterPath.
        # Cached across paints; rebuilt only when the widget is resized.
        size_key = (w, h)
        if self._cached_path is None or self._cached_path_size != size_key:
            from PySide6.QtGui import QPainterPath
            full = QPainterPath()
            rect = QRectF(inset, inset, w - 2 * inset, h - 2 * inset)
            full.addRoundedRect(rect, r, r)
            self._cached_path = full
            self._cached_path_size = size_key
            self._cached_path_len = full.length()

        full = self._cached_path
        total_len = self._cached_path_len
        draw_len = total_len * min(self._border_progress, 1.0)

        if draw_len >= total_len - 0.5:
            # Full border
            p.drawPath(full)
        else:
            # Partial: walk the path and extract a trimmed sub-path.
            # Previously sampled up to 400 points; ~80 is more than
            # enough for a 1px stroke at any reasonable widget size,
            # and pointAtPercent isn't free.
            from PySide6.QtGui import QPainterPath
            partial = QPainterPath()
            samples = 80
            end_pct = min(draw_len / total_len, 1.0)
            partial.moveTo(full.pointAtPercent(0.0))
            for i in range(1, samples + 1):
                pct = end_pct * (i / samples)
                partial.lineTo(full.pointAtPercent(pct))
            p.drawPath(partial)

        p.end()

    def resizeEvent(self, event):
        # Invalidate the cached perimeter path so paintEvent rebuilds at
        # the new size on the next paint.
        self._cached_path = None
        super().resizeEvent(event)

    # ── Open animation ───────────────────────────────────────────────

    def animate_open(self, proxy):
        """Start the open animation sequence."""
        from PySide6.QtWidgets import QGraphicsDropShadowEffect

        self._proxy = proxy
        self._border_progress = 0.0
        self._content_opacity = 0.0
        self._content_frame.setVisible(False)

        # Pre-create shadow at (0,0) — but only if the active theme
        # uses shadows.  Walk up to the RioWindow to query its theme;
        # if we can't find one (test/standalone), fall back to the
        # default theme.
        self._shadow_effect = None
        active_theme = None
        if proxy.scene() is not None:
            views = proxy.scene().views()
            if views:
                w = views[0].window()
                if hasattr(w, 'current_theme'):
                    active_theme = w.current_theme
        if active_theme is None:
            active_theme = get_theme(DEFAULT_THEME_NAME)

        if active_theme.shadow is not None:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(0)
            shadow.setOffset(QPointF(0, 0))
            shadow.setColor(QColor(0, 0, 0, 0))
            proxy.setGraphicsEffect(shadow)
            self._shadow_effect = shadow

        self._anim_phase = self._PHASE_BORDER_IN
        self._anim_t = 0.0
        self._start_timer()

    def _start_timer(self):
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer.deleteLater()
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)
        # 16 ms ≈ 60 fps. Earlier code used start(0) here, which fires as
        # fast as the event loop can dispatch — _anim_tick advances by
        # `speed` per *event* rather than per frame, so the animation
        # blew through its phases in milliseconds while the CPU pegged on
        # repaints. The comment in _anim_tick already says "per tick
        # (~16ms → full phase in ~400ms)"; this just restores that.
        self._anim_timer.start(16)

    def _stop_timer(self):
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer.deleteLater()
            self._anim_timer = None

    @staticmethod
    def _ease(t):
        """Smoothstep ease-in-out."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _anim_tick(self):
        speed = 0.04  # per tick (~16ms → full phase in ~400ms)

        if self._anim_phase == self._PHASE_BORDER_IN:
            self._anim_t += speed
            self._border_progress = self._ease(self._anim_t)
            self.update()
            if self._anim_t >= 1.0:
                self._border_progress = 1.0
                self._anim_phase = self._PHASE_CONTENT_IN
                self._anim_t = 0.0
                self._content_frame.setVisible(True)

        elif self._anim_phase == self._PHASE_CONTENT_IN:
            self._anim_t += speed
            t = self._ease(self._anim_t)
            self._content_opacity = t
            self._content_frame.setWindowOpacity(t) if hasattr(self._content_frame, 'setWindowOpacity') else None
            # Style children opacity via stylesheet trick — we paint fill in paintEvent
            self._content_frame.setStyleSheet(
                f"QWidget {{ opacity: {t}; }}"
                if False else ""  # Qt stylesheets don't support opacity; we use paintEvent fill
            )
            # Shadow grows
            if self._shadow_effect:
                ox = 45.0 * t
                blur = 10.0 + 35.0 * t
                alpha = int(180 * t)
                self._shadow_effect.setOffset(QPointF(ox, ox))
                self._shadow_effect.setBlurRadius(blur)
                self._shadow_effect.setColor(QColor(0, 0, 0, alpha))
            self.update()
            if self._anim_t >= 1.0:
                self._content_opacity = 1.0
                self._anim_phase = self._PHASE_VISIBLE
                self._stop_timer()

        elif self._anim_phase == self._PHASE_CONTENT_OUT:
            self._anim_t += speed
            t = 1.0 - self._ease(self._anim_t)
            self._content_opacity = max(t, 0.0)
            # Shadow shrinks
            if self._shadow_effect:
                ox = 45.0 * t
                blur = 10.0 + 35.0 * t
                alpha = int(180 * t)
                self._shadow_effect.setOffset(QPointF(ox, ox))
                self._shadow_effect.setBlurRadius(blur)
                self._shadow_effect.setColor(QColor(0, 0, 0, alpha))
            self.update()
            if self._anim_t >= 1.0:
                self._content_opacity = 0.0
                self._content_frame.setVisible(False)
                self._anim_phase = self._PHASE_BORDER_OUT
                self._anim_t = 0.0

        elif self._anim_phase == self._PHASE_BORDER_OUT:
            self._anim_t += speed
            self._border_progress = 1.0 - self._ease(self._anim_t)
            self.update()
            if self._anim_t >= 1.0:
                self._border_progress = 0.0
                self._anim_phase = self._PHASE_DONE
                self._stop_timer()
                if self._on_close_done:
                    # Fire callback on next event loop tick
                    QTimer.singleShot(0, self._on_close_done)

    # ── Close animation ──────────────────────────────────────────────

    def animate_close(self, on_done: callable = None):
        """Start the close animation (reverse of open). Calls on_done when finished."""
        self._on_close_done = on_done
        if self._anim_phase == self._PHASE_VISIBLE:
            self._anim_phase = self._PHASE_CONTENT_OUT
            self._anim_t = 0.0
            self._start_timer()
        elif self._anim_phase in (self._PHASE_BORDER_IN, self._PHASE_CONTENT_IN):
            # Mid-open: jump straight to closing from current state
            self._anim_phase = self._PHASE_CONTENT_OUT
            self._anim_t = 0.0
            self._start_timer()
        else:
            # Already closing or idle — just fire done
            if on_done:
                QTimer.singleShot(0, on_done)

    def _get_parse_file(self):
        """Get the filesystem's ParseFile to inject code through the proper pipeline."""
        fs = self.rio_window.rio_server.filesystem
        if fs and hasattr(fs, 'scene_dir') and hasattr(fs.scene_dir, 'parse_file'):
            return fs.scene_dir.parse_file
        return None

    def _inject_via_parse(self, code: str, label: str):
        """
        Inject code through the filesystem's parse file.

        This is the in-process equivalent of:
            cat app.py > /n/{workspace}/scene/parse

        The ParseFile._execute_code() path runs the filesystem's executor
        and calls context_file.append_code() on success, so the code
        appears in CONTEXT for the LLM.
        """
        parse_file = self._get_parse_file()
        if parse_file:
            print(f"[Apps] Injecting {label} through filesystem parse file")
            asyncio.create_task(parse_file._execute_code(code))
        else:
            print(f"[Apps] WARNING: No parse file available, filesystem not ready?")

    def launch_app(self, app_path: str, app_name: str):
        """Load and execute an app through the filesystem parse pipeline."""
        if not os.path.exists(app_path):
            print(f"[Apps] File not found: {app_path}")
            return
        try:
            with open(app_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"[Apps] Failed to read {app_path}: {e}")
            return

        print(f"[Apps] Launching {app_name} from {app_path}")
        self._inject_via_parse(code, app_name)

    def launch_app_at(self, app_path: str, app_name: str, scene_x: float, scene_y: float):
        """Load and execute an app, prepending drop coordinates."""
        if not os.path.exists(app_path):
            print(f"[Apps] File not found: {app_path}")
            return
        try:
            with open(app_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"[Apps] Failed to read {app_path}: {e}")
            return

        # Prepend drop position so the app code can reference _drop_x, _drop_y
        preamble = f"_drop_x = {scene_x!r}\n_drop_y = {scene_y!r}\n"
        code = preamble + code

        print(f"[Apps] Launching {app_name} at ({scene_x:.0f}, {scene_y:.0f})")
        self._inject_via_parse(code, app_name)


class _AltMaskItem(QGraphicsRectItem):
    """QGraphicsRectItem subclass for the Alt magic-pointer highlight
    mask. Adds a `border_progress` value (0.0 – 1.0) controlling how
    much of the rectangle's perimeter is stroked: 0 = no border, 1 =
    full border, intermediate values stroke a fraction starting at
    the top-left corner going clockwise.

    Used by the hover-in / hover-out animations to "draw" the border
    clockwise on appearance and "un-draw" it on disappearance (the
    drawn portion shrinks back toward the starting corner as progress
    decreases, which reads as the trace retreating anticlockwise).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._border_progress = 1.0  # default: fully drawn
        # We do the border drawing ourselves in paint(), so disable
        # the base class's pen-stroking of the rect outline by storing
        # the pen we'd like to use and forcing the rect item's own
        # pen to NoPen at paint time.
        self._stroke_pen = QPen(QColor(255, 255, 255, 230), 3)
        self._stroke_pen.setCosmetic(True)

    def set_border_progress(self, p):
        """Update the drawn fraction of the perimeter (0.0 – 1.0).
        Triggers a repaint."""
        # Clamp to keep paint() math sane.
        p = max(0.0, min(1.0, float(p)))
        if p == self._border_progress:
            return
        self._border_progress = p
        self.update()

    def border_progress(self):
        return self._border_progress

    def set_stroke_pen(self, pen):
        """Set the pen used to stroke the (partial) border. Triggers
        a repaint so the new pen is visible immediately."""
        self._stroke_pen = QPen(pen)
        self.update()

    def stroke_pen(self):
        return QPen(self._stroke_pen)

    def paint(self, painter, option, widget=None):
        """Custom paint: fill the rect via the brush, then stroke a
        fraction of the perimeter manually.

        The base QGraphicsRectItem.paint would stroke the full rect
        using its own pen — we suppress that by temporarily swapping
        the base pen to NoPen, calling super().paint(), then restoring,
        and finally drawing our own partial stroke on top.
        """
        # Step 1: let the base class draw the fill (no border).
        saved_pen = self.pen()
        try:
            self.setPen(Qt.NoPen)
            super().paint(painter, option, widget)
        finally:
            self.setPen(saved_pen)

        progress = self._border_progress
        if progress <= 0.0:
            return

        # Step 2: draw a fraction of the perimeter. The perimeter is
        # composed of four straight segments. We walk them in order
        # (top → right → bottom → left, i.e. clockwise) and emit line
        # segments until we've covered `progress * perimeter` length.
        r = self.rect()
        w = r.width()
        h = r.height()
        if w <= 0 or h <= 0:
            return
        perimeter = 2.0 * (w + h)
        target_len = progress * perimeter

        # Corner coordinates (TL, TR, BR, BL).
        tl = r.topLeft()
        tr = r.topRight()
        br = r.bottomRight()
        bl = r.bottomLeft()

        # Each segment as (start, end, length).
        segments = (
            (tl, tr, w),  # top, left → right
            (tr, br, h),  # right, top → bottom
            (br, bl, w),  # bottom, right → left
            (bl, tl, h),  # left, bottom → top
        )

        painter.save()
        try:
            painter.setPen(self._stroke_pen)
            painter.setBrush(Qt.NoBrush)

            remaining = target_len
            for start, end, length in segments:
                if remaining <= 0:
                    break
                if remaining >= length:
                    painter.drawLine(start, end)
                    remaining -= length
                else:
                    # Partial segment: interpolate end point.
                    t = remaining / length
                    px = start.x() + (end.x() - start.x()) * t
                    py = start.y() + (end.y() - start.y()) * t
                    painter.drawLine(
                        start,
                        QPointF(px, py),
                    )
                    remaining = 0.0
                    break
        finally:
            painter.restore()


class RioWindow(QMainWindow):
    """Main Rio window with graphics scene"""
    
    def __init__(
        self, 
        scene_manager: SceneManager, 
        rio_server: RioServer,
    ):
        super().__init__()
        self.scene_manager = scene_manager
        self.rio_server = rio_server
        
        self.mouse_callback = None
        self.key_callback = None
        
        # Terminal management
        self.terminals = []
        
        # Creation state for Plan 9-style terminal creation
        self.new_terminal_mode = False
        self.delete_mode = False
        self.pop_mode = False
        
        # Popped-out widget tracking: {top_item: pop_info_dict}
        self._popped_widgets = {}
        self.is_creating_terminal = False
        self.current_terminal = None
        self.current_proxy = None
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.selection_rect = None
        
        # Dark mode state
        self._dark_mode = False
        self._dark_mode_bg_step = [0]
        self._dark_mode_bg_timer = None

        # Active visual theme (see rio.theme).  "glass" is the original
        # translucent-with-shadows look; "paper" is a flat editorial style.
        # Themes can be swapped at runtime via set_theme().
        self._active_theme_name = DEFAULT_THEME_NAME
        self._theme_anim = None  # in-flight theme transition animation
        
        # Ctrl+Mouse pan state
        self._ctrl_panning = False
        self._ctrl_pan_last_pos = None
        self._ctrl_pan_pre_transform = None
        self._zoom_back_transform = None
        
        # Ctrl+Mouse object drag state
        self._ctrl_dragging_item = None
        self._ctrl_drag_offset = QPointF()
        
        # Ctrl+RightMouse orbit state
        self._ctrl_orbit_active = False
        self._ctrl_orbit_anchor = None        # viewport pixel where press started
        self._ctrl_orbit_pre_transform = None # transform before orbit began
        self._ctrl_orbit_used = False         # set True if orbit was used, suppresses context menu

        # Ctrl+1 temporary mouse-zoom toggle state
        self._temp_zoom_active = False
        self._temp_zoom_pre_transform = None
        self._temp_zoom_pre_center = None

        # Alt magic-pointer state (inspect/highlight scene widgets)
        self._alt_magic_active = False
        self._alt_hover_proxy = None       # QGraphicsProxyWidget currently hovered
        self._alt_hover_widget = None      # inner QWidget under cursor (may be child-of-child)
        self._alt_hover_item = None        # non-proxy QGraphicsItem currently hovered (fallback)
        self._alt_overlay_item = None      # QGraphicsRectItem used as the highlight mask
        self._alt_preview_text = None      # QGraphicsTextItem drawn INSIDE the mask
        self._alt_prev_cursor = None       # cursor to restore when Alt is released
        # Multi-select pool. Click adds the hovered widget's snapshotted
        # payload here; Ctrl+S sends the whole batch and exits. Keyed by
        # id(target) for O(1) dedup, ordered so the agent receives picks
        # in click order. Values are the rendered payload strings.
        self._alt_selection = {}

        # Fade animation state — runs the overlay's opacity from 0→1 on
        # hover-in and 1→0 on hover-out (which then disposes of the item).
        self._alt_fade_anim = None
        self._alt_fading_out_item = None   # overlay currently animating out

        # White-noise cursor animation — a QTimer rebuilds the cursor
        # pixmap on a slow tick so it visibly shimmers.
        self._alt_noise_timer = None
        # Shared noise tile that feeds both the cursor fill and the
        # mask border pen each tick. Holding one pixmap and refreshing
        # it in place keeps the two visuals in sync.
        self._alt_noise_tile_pm = None

        # Streaming-reveal animation state for the preview text
        self._alt_preview_target_text = ""     # full payload being revealed
        self._alt_preview_visible_chars = 0    # chars currently shown
        self._alt_preview_anim_timer = None    # QTimer driving the typewriter
        self._alt_preview_widget_key = None    # used to detect "same widget" re-hover
        
        # Cache of items that have a QGraphicsDropShadowEffect attached.
        # Maintained explicitly so theme/dark animations don't have to
        # sweep self.graphics_scene.items() — which is O(N) even with
        # BSP indexing. With ~50 proxies on screen this cut per-frame
        # work in _start_dark_mode_animation from one full scene scan
        # to a direct list iteration. Keys are the QGraphicsItem; we use
        # a list-of-weakrefs pattern via a set of strong refs and prune
        # on access.
        self._shadowed_items = set()

        # Initialize execution context
        self.execution_context = None
        self.executor = None
        
        # Initialize UI
        self._init_ui()
        
        # Attach scene manager to Qt (IMPORTANT: pass main_window too!)
        scene_manager.attach_qt(self.graphics_scene, main_window=self)
        
        # Initialize executor after UI is ready
        self._init_executor()
        
        # Initialize AI Voice Control widget (hidden by default)
        self._init_voice_control()
        
        # Initialize Debug Overlay (hidden by default, shown by DebugNode)
        self._init_debug_overlay()
        
        # Initialize Immersive Mode (Ctrl+I)
        self._immersive_mode = install_immersive_mode(self)
        
        # App Launcher (hidden by default)
        self._app_launcher_proxy = None

        # Operator (node graph) overlay — toggled via the right-click
        # menu. None when not running; an Operator instance while shown.
        # This replaces the old external apps/operator_fs.py that was
        # injected through the parser; construction and teardown now
        # happen in-process via _toggle_operator().
        self._operator = None
    
    def _init_executor(self):
        """Initialize the code execution system"""
        self.execution_context = ExecutionContext(
            self.scene_manager,
            main_window=self,
            graphics_scene=self.graphics_scene,
            graphics_view=self.graphics_view
        )
        self.executor = Executor(
            self.execution_context,
            error_callback=self._handle_execution_error
        )

    # ------------------------------------------------------------------
    # Performance helpers — proxy caching & shadow tracking
    # ------------------------------------------------------------------

    def _apply_proxy_cache(self, proxy):
        """Switch a proxy to bitmap caching.

        DeviceCoordinateCache renders the proxy's contents (and any
        graphics effect) to an offscreen pixmap, then blits that pixmap
        during view repaints. The pixmap is only re-rendered when the
        proxy itself signals an update (content change, geometry
        change, effect change). This is a large win for things like
        terminals, web views, and panel widgets that are visually stable
        between user actions but whose layouts are expensive to walk
        every frame the view repaints.

        Notable: panning, zooming, and orbiting the view do NOT
        invalidate the cache — they just retransform the cached bitmap,
        which is exactly what we want. Caches DO invalidate on resize,
        which is why we don't apply this to widgets that resize every
        frame (none exist currently but if added, opt them out).
        """
        if proxy is None:
            return
        try:
            proxy.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        except Exception as e:
            logger.debug(f"setCacheMode failed on proxy: {e}")

    def register_shadowed(self, item):
        """Track an item that has (or just got) a QGraphicsDropShadowEffect.

        Called by code that attaches shadows so the theme/dark
        animations don't need to discover them via a scene scan.
        Safe to call multiple times.
        """
        if item is not None:
            self._shadowed_items.add(item)

    def unregister_shadowed(self, item):
        """Drop an item from the shadowed cache."""
        self._shadowed_items.discard(item)

    def _iter_live_shadowed(self):
        """Yield (item, effect) for cached shadowed items that are still alive.

        Prunes dead Qt items as it goes — RuntimeError on attribute
        access is the canonical signal that a QGraphicsItem's C++ side
        has been deleted out from under the Python wrapper.
        """
        from PySide6.QtWidgets import QGraphicsDropShadowEffect as _DSE
        dead = []
        for item in self._shadowed_items:
            try:
                effect = item.graphicsEffect()
            except RuntimeError:
                dead.append(item)
                continue
            if isinstance(effect, _DSE):
                yield item, effect
            else:
                # Effect was swapped/cleared elsewhere; drop from cache
                dead.append(item)
        for d in dead:
            self._shadowed_items.discard(d)

    def _set_shadows_enabled_during_animation(self, enabled: bool):
        """Toggle drop-shadow effects on all tracked items.

        QGraphicsDropShadowEffect re-rasterizes its host to an offscreen
        buffer and Gaussian-blurs it on the CPU on every paint. With N
        shadowed proxies and Qt repainting them all during a theme
        animation, this is the single largest paint cost in the scene.
        Disabling for the duration of the animation is visually
        acceptable (the colours are interpolating anyway) and roughly
        doubles animation FPS in scenes with >20 proxies.
        """
        for _item, effect in self._iter_live_shadowed():
            try:
                effect.setEnabled(enabled)
            except RuntimeError:
                pass

    def _handle_execution_error(self, error: str):
        """Handle execution errors by displaying in the most recent terminal"""
        if self.terminals:
            self.terminals[-1].append_output(f"\nError:\n{error}\n", color="#f48771")
    
    def _init_voice_control(self):
        """Create the AI Voice Control eye widget and add it to the scene."""
        self.voice_control = AIVoiceControlWidget(
            scale_factor=0.32,
            llmfs_mount=self.rio_server.llmfs_mount,
            rio_mount=self.rio_server.rio_mount,
        )
        self.voice_control_proxy = self.graphics_scene.addWidget(self.voice_control)
        self.voice_control_proxy.setZValue(1000)
        self.voice_control_proxy.setPos(200, -350)
        self.voice_control_proxy.setVisible(False)
        # Bitmap-cache this proxy: the eye animation is internal to the
        # widget (it issues its own update()s when its frame changes),
        # so caching at the proxy level is safe and skips Qt re-walking
        # the widget tree during view repaints.
        self._apply_proxy_cache(self.voice_control_proxy)
        # Shadow + perspective tilt on proxy — managed by the widget.
        # Only attach the shadow if the active theme uses shadows; on
        # flat themes (Paper) we want zero drop-shadows anywhere.
        if self.current_theme.shadow is not None:
            self.voice_control.attach_proxy_shadow(self.voice_control_proxy)
            self.register_shadowed(self.voice_control_proxy)
        self.voice_control.flicker_triggered.connect(self._send_flicker_context_to_ai)

    def _send_flicker_context_to_ai(self):
        """Retrieves live compacted code and sends it to the active AI agent."""
        rio_fs = self.rio_server.filesystem
        if not rio_fs or not hasattr(rio_fs, 'context_file'):
            return

        # Get the smart-compacted code from the context file
        compacted_context = rio_fs.context_file.get_all_code()
        
        # Build the payload
        # We wrap it in a system header so the AI knows why it's receiving this
        payload = f"[SYSTEM: User triggered flicker. Current Scene Context:]\n{compacted_context}"
        
        # Send to the active agent input
        agent_input = os.path.join(self.voice_control.llmfs_mount, "av", "input")
        
        def _write():
            try:
                with open(agent_input, 'w') as f:
                    f.write(payload)
                print(f"[AIVoice] Sent {len(compacted_context)} chars of CONTEXT to AI.")
            except Exception as e:
                print(f"Error sending context: {e}")

        threading.Thread(target=_write, daemon=True).start()

    # ------------------------------------------------------------------
    # Alt Magic-Pointer (inspect / highlight scene widgets)
    # ------------------------------------------------------------------

    # Tunable colours for the highlight mask + cursor tint.
    # Bright-white at higher alpha so the mask actually obscures the
    # underlying widget text (the previous lower alpha let punctuation
    # bleed through and made the preview hard to read). The border is
    # painted via a tile-textured pen (see _alt_noise_tile) so its
    # colour here is irrelevant when noise is applied.
    _ALT_MASK_FILL = QColor(255, 255, 255, 175)   # bright white, ~69% alpha
    _ALT_MASK_BORDER = QColor(255, 255, 255, 230) # solid fallback if noise tile fails
    _ALT_CURSOR_COLOR = QColor(80, 170, 255)      # baseline cursor accent;
                                                  # the live "white noise"
                                                  # animation overrides this
                                                  # while magic mode is active
    # Preview text inside the mask — black for legibility against the
    # bright-white interior. Bumped up a few px so it's actually
    # readable at typical widget sizes.
    _ALT_PREVIEW_TEXT_COLOR = QColor(0, 0, 0, 240)
    _ALT_PREVIEW_FONT_PX = 13
    _ALT_PREVIEW_PAD = 5    # inner padding from the mask border
    # Streaming-reveal tuning. Each tick reveals CHARS_PER_TICK characters
    # every TICK_MS milliseconds. 4 chars per 25 ms ≈ 160 cps — fast enough
    # to feel snappy on short payloads, slow enough that the streaming
    # effect is visible to a human eye.
    _ALT_PREVIEW_TICK_MS = 25
    _ALT_PREVIEW_CHARS_PER_TICK = 4
    # Fade-in / fade-out animation timing for the mask itself.
    _ALT_FADE_IN_MS = 140
    _ALT_FADE_OUT_MS = 180
    # White-noise cursor — regenerate the cursor + mask-border noise
    # texture on a slow timer so both visibly "shimmer" in sync.
    # 80 ms ≈ 12 fps, fast enough to read as animation, cheap to render.
    _ALT_NOISE_TICK_MS = 80
    # Arrow cursor dimensions (kept proportional to a normal system
    # arrow). Hotspot is at the tip (top-left of the bounding box).
    _ALT_CURSOR_W = 22
    _ALT_CURSOR_H = 28
    # Noise tile size for the mask border. Smaller = the pattern repeats
    # more often around the border, which looks more "static-like".
    _ALT_NOISE_TILE_SIZE = 12
    # Mask border width in pixels (cosmetic, i.e. constant on screen).
    _ALT_MASK_BORDER_WIDTH = 3

    def _activate_alt_magic_pointer(self):
        """Enter magic-pointer mode: animated white-noise cursor + hover
        mask on scene items. Called from keyPressEvent on Alt-down."""
        if self._alt_magic_active:
            return
        self._alt_magic_active = True

        viewport = self.graphics_view.viewport()
        self._alt_prev_cursor = viewport.cursor()

        # First frame of the noise cursor + start the shimmer timer.
        self._alt_apply_noise_cursor()
        self._alt_start_noise_cursor_animation()

        # Sync the highlight to wherever the mouse currently is.
        try:
            global_pos = QCursor.pos()
            vp_pos = viewport.mapFromGlobal(global_pos)
            if viewport.rect().contains(vp_pos):
                self._alt_update_hover_from_viewport_pos(vp_pos)
        except Exception:
            pass

    def _alt_start_noise_cursor_animation(self):
        """Tick the shared noise tile so both the cursor and the mask
        border visibly shimmer in sync."""
        self._alt_stop_noise_cursor_animation()
        # Build an initial tile + apply once so the visuals don't wait
        # 80 ms for their first noise frame.
        self._alt_noise_tile_pm = self._alt_build_noise_tile()
        self._alt_apply_noise_cursor()
        self._alt_apply_noise_border()
        timer = QTimer(self)
        timer.setInterval(self._ALT_NOISE_TICK_MS)
        timer.timeout.connect(self._alt_noise_tick)
        timer.start()
        self._alt_noise_timer = timer

    def _alt_noise_tick(self):
        """One frame of the shared noise animation: regenerate the tile,
        then push it to the cursor and the mask border. Both visuals
        end up displaying the same noise field on every tick."""
        if not self._alt_magic_active:
            return
        try:
            self._alt_noise_tile_pm = self._alt_build_noise_tile()
        except Exception:
            # If tile generation fails (e.g. mid-shutdown), leave the
            # previous tile in place — better than blank cursor.
            pass
        self._alt_apply_noise_cursor()
        self._alt_apply_noise_border()

    def _alt_stop_noise_cursor_animation(self):
        t = self._alt_noise_timer
        if t is not None:
            try:
                t.stop()
                t.deleteLater()
            except Exception:
                pass
            self._alt_noise_timer = None
        # Drop the shared tile so a fresh activation builds a new one.
        self._alt_noise_tile_pm = None

    def _alt_build_noise_tile(self, size=None):
        """Build a small square QPixmap of greyscale white-noise.

        Used both as the cursor's fill texture (clipped to an arrow path)
        and as the mask's border-pen texture, so the two shimmer in
        visual sync. Building one shared tile per timer tick is
        cheaper than two independent noise fields and keeps the
        aesthetics consistent.
        """
        import random
        from PySide6.QtGui import QImage, QPixmap

        if size is None:
            size = self._ALT_NOISE_TILE_SIZE
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        # Build the noise directly. Bias toward the brighter end so the
        # texture reads as "mostly white with dark speckles", matching
        # the bright-white mask aesthetic.
        for y in range(size):
            for x in range(size):
                v = random.randint(80, 255)
                img.setPixel(x, y, 0xFF000000 | (v << 16) | (v << 8) | v)
        return QPixmap.fromImage(img)

    def _alt_arrow_path(self, w, h):
        """Return a QPainterPath shaped like a classic arrow cursor,
        bounded by an w×h rect with the tip at (0,0). Reused for the
        cursor fill, outline, and clipping."""
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        # Coordinates chosen to look like a standard left-tip arrow:
        # tip at top-left, body slanting down-right, a notch on the
        # right side of the bottom, and a small tail. All scaled by w/h
        # against a nominal 22×28 reference so changing the size keeps
        # the proportions.
        sx = w / 22.0
        sy = h / 28.0
        pts = [
            (1, 1),     # tip
            (1, 20),    # left edge bottom
            (6, 16),    # inner notch top
            (10, 24),   # tail bottom-left
            (13, 23),   # tail bottom-right
            (9, 15),    # back up to notch
            (16, 14),   # right shoulder
        ]
        path.moveTo(pts[0][0] * sx, pts[0][1] * sy)
        for x, y in pts[1:]:
            path.lineTo(x * sx, y * sy)
        path.closeSubpath()
        return path

    def _alt_apply_noise_cursor(self):
        """Build a fresh arrow-shaped cursor whose interior is filled
        with greyscale white-noise, and apply it to the viewport.

        Shape: a classic left-tip arrow (hotspot at the tip).
        Fill: tiled noise texture (same tile feeds the mask border on
              this tick, so both shimmer in sync).
        Outline: thin dark stroke for legibility against bright UI.
        """
        if not self._alt_magic_active:
            # Race: timer fired after deactivation. Bail.
            return
        try:
            from PySide6.QtGui import QPixmap, QPainter as _QPainter, QPen as _QPen, QBrush as _QBrush

            w = self._ALT_CURSOR_W
            h = self._ALT_CURSOR_H
            pm = QPixmap(w, h)
            pm.fill(Qt.transparent)

            arrow = self._alt_arrow_path(w, h)
            # Reuse the per-tick noise tile so the cursor and the mask
            # border are drawn from the same noise field. Build a fresh
            # one if we somehow lost it.
            tile = self._alt_noise_tile_pm
            if tile is None or tile.isNull():
                tile = self._alt_build_noise_tile()
                self._alt_noise_tile_pm = tile

            p = _QPainter(pm)
            p.setRenderHint(_QPainter.Antialiasing, True)
            # Fill: clip to the arrow shape, then paint the noise tile
            # by drawing it as a tiled brush across the full pixmap.
            p.setClipPath(arrow)
            p.fillRect(0, 0, w, h, _QBrush(tile))
            p.setClipping(False)
            # Outline: thin dark stroke around the arrow path. Without
            # this the cursor disappears over bright/white widgets.
            outline = _QPen(QColor(20, 20, 20, 220))
            outline.setWidthF(1.2)
            p.setPen(outline)
            p.setBrush(Qt.NoBrush)
            p.drawPath(arrow)
            p.end()

            # Hotspot at the arrow tip (top-left of the bounding box).
            cursor = QCursor(pm, 1, 1)
        except Exception:
            # If anything goes wrong building the noise cursor, fall
            # back to a benign system cursor so the user still has a
            # visible pointer.
            cursor = QCursor(Qt.PointingHandCursor)

        try:
            self.graphics_view.viewport().setCursor(cursor)
        except Exception:
            pass

    def _alt_apply_noise_border(self):
        """Push the current shared noise tile onto the mask's border
        pen so the border shimmers in sync with the cursor.

        Uses set_stroke_pen rather than setPen because _AltMaskItem
        suppresses the base pen during paint and uses its own stroke
        pen instead (so we can stroke a fraction of the perimeter
        instead of the whole thing).
        """
        item = self._alt_overlay_item
        if item is None:
            return
        tile = self._alt_noise_tile_pm
        try:
            from PySide6.QtGui import QPen as _QPen, QBrush as _QBrush
            if tile is not None and not tile.isNull():
                pen = _QPen(_QBrush(tile), self._ALT_MASK_BORDER_WIDTH)
            else:
                pen = _QPen(self._ALT_MASK_BORDER, self._ALT_MASK_BORDER_WIDTH)
            pen.setCosmetic(True)
            if isinstance(item, _AltMaskItem):
                item.set_stroke_pen(pen)
            else:
                item.setPen(pen)
        except Exception:
            pass

    def _deactivate_alt_magic_pointer(self):
        """Leave magic-pointer mode: stop the noise cursor + fade out
        the mask."""
        if not self._alt_magic_active:
            return
        self._alt_magic_active = False

        self._alt_stop_noise_cursor_animation()

        viewport = self.graphics_view.viewport()
        if self._alt_prev_cursor is not None:
            viewport.setCursor(self._alt_prev_cursor)
        else:
            viewport.unsetCursor()
        self._alt_prev_cursor = None

        # Fade the current mask out rather than yanking it. This keeps
        # the visual transition smooth when the user lifts the Alt key.
        self._alt_fade_out_overlay()
        self._alt_hover_proxy = None
        self._alt_hover_widget = None
        self._alt_hover_item = None
        # Clear the per-id cache — id()s are reused as Python frees
        # objects, so a stale entry would silently misattribute a name
        # to whatever widget happens to land on that id next session.
        if hasattr(self, "_alt_var_name_cache"):
            self._alt_var_name_cache.clear()
        # Drop any unsent multi-selection — leaving entries here would
        # leak picks from a previous session into the next batch.
        self._alt_selection.clear()

    def _alt_ensure_overlay(self):
        """Lazily create the QGraphicsRectItem used as the highlight mask
        plus the QGraphicsTextItem that streams the AV payload live
        INSIDE the mask. The text item is a child of the mask so it
        follows position/visibility automatically.

        New items always start at opacity 0 and are faded in by
        _alt_fade_in_overlay so the appearance is smooth rather than a
        hard pop.
        """
        if self._alt_overlay_item is not None:
            return self._alt_overlay_item
        rect_item = _AltMaskItem()
        rect_item.setBrush(QBrush(self._ALT_MASK_FILL))
        # Border pen — uses the shared noise tile so the border
        # shimmers in sync with the cursor. Falls back to a solid
        # white pen if the tile isn't ready yet (e.g. magic mode
        # activated less than one tick ago and the timer hasn't built
        # the first tile).
        tile = self._alt_noise_tile_pm
        if tile is not None and not tile.isNull():
            pen = QPen(QBrush(tile), self._ALT_MASK_BORDER_WIDTH)
        else:
            pen = QPen(self._ALT_MASK_BORDER, self._ALT_MASK_BORDER_WIDTH)
        pen.setCosmetic(True)  # constant pixel width regardless of zoom
        rect_item.set_stroke_pen(pen)
        # Start with no border drawn — the trace-in animation will
        # animate this from 0 → 1 as the fade-in plays.
        rect_item.set_border_progress(0.0)
        # Sit above everything else (voice_control_proxy is z=1000).
        rect_item.setZValue(10000)
        # Don't intercept events — purely visual.
        rect_item.setAcceptedMouseButtons(Qt.NoButton)
        rect_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        rect_item.setFlag(QGraphicsItem.ItemIsMovable, False)
        # Clip children (the preview text) to the mask rect so the
        # streamed content never spills outside the highlighted region.
        rect_item.setFlag(QGraphicsItem.ItemClipsChildrenToShape, True)
        # Start invisible — the fade-in animation will ramp it up.
        rect_item.setOpacity(0.0)
        self.graphics_scene.addItem(rect_item)
        self._alt_overlay_item = rect_item

        # Preview text as a child of the mask — it inherits position,
        # visibility, and clipping. No separate backdrop: the mask's
        # own white fill is what makes the black text legible.
        text = QGraphicsTextItem(rect_item)
        from PySide6.QtGui import QFont
        # Qt's QFont(name) treats `name` as a single literal family —
        # a comma-separated CSS-style string won't fall through. To
        # actually walk a fallback chain we have to use setFamilies()
        # with a real list, OR (cheaper / older Qt) set StyleHint to
        # Monospace as a fallback when the primary family is missing.
        font = QFont("Consolas")
        try:
            # Available since Qt 5.13; preferred path on PySide6.
            font.setFamilies(["Consolas", "Menlo", "DejaVu Sans Mono",
                              "Liberation Mono", "Courier New", "monospace"])
        except Exception:
            pass
        # Belt-and-suspenders: if none of the listed families exist,
        # Qt will substitute a font matching the style hint instead
        # of the default sans-serif.
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPixelSize(self._ALT_PREVIEW_FONT_PX)
        text.setFont(font)
        text.setDefaultTextColor(self._ALT_PREVIEW_TEXT_COLOR)
        text.setZValue(1)
        text.setAcceptedMouseButtons(Qt.NoButton)
        text.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._alt_preview_text = text

        # Kick off the fade-in. This is animated on the rect_item's
        # opacity, which Qt cascades to the child text item.
        self._alt_fade_in_overlay()

        return rect_item

    def _alt_fade_in_overlay(self):
        """Animate the current overlay's opacity from its current value
        up to 1.0 AND trace the border clockwise from 0 → 1 over the
        same duration. Cancels any in-flight fade so the latest call
        wins. Idempotent — if the overlay is already at 1.0, the
        opacity anim finishes harmlessly while the border anim either
        completes the trace or no-ops if already at 1.0.
        """
        if self._alt_overlay_item is None:
            return

        # If a fade-IN is in flight on the current overlay, cancel it
        # so the latest call wins. (Fade-OUTs on orphan items are
        # untouched — those live on a separate code path.)
        if self._alt_fade_anim is not None and \
                self._alt_fade_anim.property("_alt_target_kind") == "in":
            try:
                self._alt_fade_anim.stop()
            except Exception:
                pass
            self._alt_fade_anim = None

        item = self._alt_overlay_item

        # ---- Opacity animation (fill ramps up) ----
        start = item.opacity()
        anim = QVariantAnimation(self)
        anim.setStartValue(start)
        anim.setEndValue(1.0)
        anim.setDuration(self._ALT_FADE_IN_MS)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setProperty("_alt_target_kind", "in")

        def _on_value(v, _item=item):
            try:
                _item.setOpacity(float(v))
            except Exception:
                pass
        anim.valueChanged.connect(_on_value)
        anim.start()
        self._alt_fade_anim = anim

        # ---- Border-trace animation (perimeter draws clockwise) ----
        # Runs in parallel with the opacity ramp so by the time the
        # fill is fully opaque the border has finished tracing.
        self._alt_animate_border_progress(item, 1.0, self._ALT_FADE_IN_MS,
                                          QEasingCurve.OutCubic)

    def _alt_animate_border_progress(self, item, end_progress, duration_ms,
                                     easing=QEasingCurve.Linear):
        """Animate the `border_progress` of an _AltMaskItem from its
        current value to `end_progress` over `duration_ms`. The
        animation reference is stored on the item itself so its
        lifetime tracks the item.
        """
        if not isinstance(item, _AltMaskItem):
            return
        # Stop any previous trace anim on this item.
        prev = getattr(item, "_alt_border_anim", None)
        if prev is not None:
            try:
                prev.stop()
            except Exception:
                pass

        start_progress = item.border_progress()
        if abs(start_progress - end_progress) < 1e-4:
            # Already at target; nothing to animate.
            item._alt_border_anim = None
            return

        anim = QVariantAnimation(self)
        anim.setStartValue(float(start_progress))
        anim.setEndValue(float(end_progress))
        anim.setDuration(duration_ms)
        anim.setEasingCurve(easing)

        def _on_value(v, _item=item):
            try:
                _item.set_border_progress(float(v))
            except Exception:
                pass
        anim.valueChanged.connect(_on_value)
        anim.start()
        item._alt_border_anim = anim

    def _alt_fade_out_overlay(self):
        """Detach the current overlay and fade it out, removing it from
        the scene when the animation finishes. The "current" overlay
        slot is cleared immediately so a new hover can build a fresh
        mask without waiting for the fade-out to complete.
        """
        if self._alt_overlay_item is None:
            # Nothing to fade; still cancel any stale streaming/anim.
            self._alt_stop_preview_stream()
            self._alt_preview_target_text = ""
            self._alt_preview_visible_chars = 0
            self._alt_preview_widget_key = None
            return

        # If there's already an orphan fading out, drop it abruptly
        # to avoid a pile-up of ghost overlays during fast hover
        # transitions.
        if self._alt_fading_out_item is not None:
            try:
                self.graphics_scene.removeItem(self._alt_fading_out_item)
            except Exception:
                pass
            self._alt_fading_out_item = None

        # Detach: current slot becomes free, old item becomes the orphan.
        orphan = self._alt_overlay_item
        self._alt_overlay_item = None
        self._alt_preview_text = None
        self._alt_fading_out_item = orphan

        # Stop the streaming reveal — the old text shouldn't keep
        # growing inside a fading-out mask.
        self._alt_stop_preview_stream()
        self._alt_preview_target_text = ""
        self._alt_preview_visible_chars = 0
        self._alt_preview_widget_key = None

        start = orphan.opacity()
        anim = QVariantAnimation(self)
        anim.setStartValue(start)
        anim.setEndValue(0.0)
        anim.setDuration(self._ALT_FADE_OUT_MS)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.setProperty("_alt_target_kind", "out")

        def _on_value(v, _item=orphan):
            try:
                _item.setOpacity(float(v))
            except Exception:
                pass
        anim.valueChanged.connect(_on_value)

        def _on_finished(_item=orphan):
            try:
                self.graphics_scene.removeItem(_item)
            except Exception:
                pass
            if self._alt_fading_out_item is _item:
                self._alt_fading_out_item = None
        anim.finished.connect(_on_finished)

        anim.start()
        # We don't store this in self._alt_fade_anim; that slot is for
        # the active overlay's fade-in. Orphan fade-outs are kept
        # alive via the QObject parent (self), so they don't get GC'd
        # mid-flight.

        # Untrace the border in parallel: progress runs from its
        # current value back to 0.0 over the same duration. As the
        # drawn fraction shrinks, the trace appears to retreat
        # toward the top-left corner — the visual inverse of the
        # clockwise draw on hover-in.
        self._alt_animate_border_progress(orphan, 0.0, self._ALT_FADE_OUT_MS,
                                          QEasingCurve.InCubic)

    def _alt_clear_overlay(self):
        """Hover-out clear: fade the mask out and dispose when done.
        Stops the streaming animation immediately so the preview text
        doesn't keep growing inside a fading mask.

        This is the user-facing path — the smooth disappearance. For a
        guaranteed-synchronous removal, use _alt_clear_overlay_immediate.
        """
        # _alt_fade_out_overlay handles the stream-stop + state reset
        # internally; just delegate.
        self._alt_fade_out_overlay()

    def _alt_clear_overlay_immediate(self):
        """Synchronous removal of the mask + any fading-out orphan.
        Used during full deactivation paths where we don't want a
        post-cursor-restore ghost overlay lingering on screen.
        """
        self._alt_stop_preview_stream()
        self._alt_preview_target_text = ""
        self._alt_preview_visible_chars = 0
        self._alt_preview_widget_key = None

        for attr in ("_alt_overlay_item", "_alt_fading_out_item"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    self.graphics_scene.removeItem(obj)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._alt_preview_text = None

        # Cancel any in-flight fade animation tied to the current slot.
        if self._alt_fade_anim is not None:
            try:
                self._alt_fade_anim.stop()
            except Exception:
                pass
            self._alt_fade_anim = None

    def _alt_update_preview_text(self, mask_rect):
        """Stream the AV payload into the preview text item, INSIDE the
        mask. Called from _alt_update_hover_from_viewport_pos after the
        mask has been sized for the current hover.

        Behaviour:
          * If the hovered widget changed (payload differs from last
            time), restart the typewriter reveal from char 0.
          * If the hovered widget is unchanged (small mouse jitter on
            the same target), keep the current reveal position so the
            text doesn't flicker by restarting.
          * The text item is a child of the mask with
            ItemClipsChildrenToShape set — anything past the mask edge
            is clipped, so a tiny widget shows just the first line or
            two while a large widget shows the whole payload.
        """
        if self._alt_preview_text is None:
            return

        payload = self._alt_build_payload() or ""

        # "Identity" of this hover for restart-detection: prefer the
        # actual hovered widget/item identity rather than the payload
        # string, so two visually-identical widgets still get their
        # own animations.
        widget_key = (
            id(self._alt_hover_widget) if self._alt_hover_widget is not None
            else id(self._alt_hover_item) if self._alt_hover_item is not None
            else None
        )

        text = self._alt_preview_text
        pad = self._ALT_PREVIEW_PAD
        text.setPos(mask_rect.x() + pad, mask_rect.y() + pad)
        inner_w = max(0.0, mask_rect.width() - pad * 2)
        text.setTextWidth(inner_w if inner_w > 0 else -1)

        if widget_key == self._alt_preview_widget_key and \
                payload == self._alt_preview_target_text:
            # Same widget, same payload — just keep streaming what
            # we already started. No restart, no jitter.
            return

        # New target. Reset and kick off the typewriter.
        self._alt_preview_widget_key = widget_key
        self._alt_preview_target_text = payload
        self._alt_preview_visible_chars = 0
        text.setPlainText("")
        self._alt_start_preview_stream()

    def _alt_start_preview_stream(self):
        """(Re)start the streaming reveal timer."""
        # Stop any in-flight animation before starting a new one.
        self._alt_stop_preview_stream()

        if not self._alt_preview_target_text:
            return

        timer = QTimer(self)
        timer.setInterval(self._ALT_PREVIEW_TICK_MS)
        timer.timeout.connect(self._alt_preview_tick)
        timer.start()
        self._alt_preview_anim_timer = timer

    def _alt_stop_preview_stream(self):
        """Tear down the animation timer (if any)."""
        t = self._alt_preview_anim_timer
        if t is not None:
            try:
                t.stop()
                t.deleteLater()
            except Exception:
                pass
            self._alt_preview_anim_timer = None

    def _alt_preview_tick(self):
        """One step of the typewriter reveal — advance the visible char
        count and update the text item. Stops the timer once the full
        payload is on screen."""
        if self._alt_preview_text is None:
            self._alt_stop_preview_stream()
            return

        target = self._alt_preview_target_text
        if not target:
            self._alt_stop_preview_stream()
            return

        self._alt_preview_visible_chars = min(
            len(target),
            self._alt_preview_visible_chars + self._ALT_PREVIEW_CHARS_PER_TICK,
        )
        self._alt_preview_text.setPlainText(
            target[: self._alt_preview_visible_chars]
        )

        if self._alt_preview_visible_chars >= len(target):
            self._alt_stop_preview_stream()

    def _alt_update_hover_from_viewport_pos(self, viewport_pos):
        """Recompute the hovered widget under `viewport_pos` and resize
        the highlight mask to match.

        Hover model — supports "child of child":
          * If the cursor is over a QGraphicsProxyWidget, ask the inner
            QWidget for its deepest child at that point (childAt). That
            gives natural drill-down: hovering a button inside a frame
            highlights the button; sliding off the button onto the
            frame's body highlights the frame.
          * If the cursor is over a non-proxy scene item, highlight that
            item's sceneBoundingRect directly.
          * If the cursor is over nothing, clear the mask.

        After the mask is positioned, also refresh the preview-text
        block so the user sees, live, the exact payload that would be
        sent to the AV agent if they clicked.
        """
        if not self._alt_magic_active:
            return

        scene_pos = self.graphics_view.mapToScene(viewport_pos)

        # The highlight overlay lives in the scene at a very high Z, so
        # a naive itemAt() would always return the overlay (or its child
        # text item, or a fading-out orphan from the previous widget)
        # once placed. Hide all of them for the duration of the hit
        # test. Hiding the parent hides its child text automatically.
        hidden_items = []
        for obj in (self._alt_overlay_item, self._alt_fading_out_item):
            if obj is not None and obj.isVisible():
                obj.setVisible(False)
                hidden_items.append(obj)

        try:
            item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
        finally:
            # Restore visibility. The placement branches below set
            # visible=True on success; the no-hit branch calls
            # _alt_clear_overlay (i.e. fade-out) which manages
            # its own visibility.
            for obj in hidden_items:
                try:
                    obj.setVisible(True)
                except Exception:
                    pass

        if item is None:
            self._alt_hover_proxy = None
            self._alt_hover_widget = None
            self._alt_hover_item = None
            self._alt_clear_overlay()
            return

        # Defensive: if for any reason one of our own items still came
        # back as the hit, don't get stuck on ourselves.
        if item is self._alt_overlay_item or \
           item is self._alt_preview_text or \
           item is self._alt_fading_out_item:
            self._alt_hover_proxy = None
            self._alt_hover_widget = None
            self._alt_hover_item = None
            self._alt_clear_overlay()
            return

        # Walk up to find the owning QGraphicsProxyWidget (if any).
        proxy = item
        while proxy is not None and not isinstance(proxy, QGraphicsProxyWidget):
            proxy = proxy.parentItem()

        if proxy is not None and proxy.widget() is not None:
            embedded = proxy.widget()
            widget_pos = proxy.mapFromScene(scene_pos)
            target = embedded.childAt(int(widget_pos.x()), int(widget_pos.y()))
            if target is None:
                target = embedded

            # Composite widgets like QCalendarWidget, QSpinBox, QComboBox,
            # QDateEdit etc. are made of internal sub-widgets — typically
            # unnamed QWidgets or private Qt classes. childAt() returns
            # one of those internals, which is useless to a user or to
            # the AV agent ("you clicked a QWidget"). If the click
            # landed inside a known composite, promote the target up
            # to that composite. This preserves drill-down for genuinely
            # user-meaningful nesting (a button inside a frame) while
            # avoiding the unhelpful drill-down into Qt's internals.
            target = self._alt_promote_to_meaningful_widget(target, embedded)

            self._alt_hover_proxy = proxy
            self._alt_hover_widget = target
            self._alt_hover_item = None

            # Build the scene-space rect for `target`. subWidgetRect
            # returns the rect in the proxy's local coordinate system;
            # mapping to scene then handles any proxy transform/pos.
            try:
                local_rect = proxy.subWidgetRect(target)
            except Exception:
                # Fallback: use target's geometry mapped through parents.
                tl = target.mapTo(embedded, QPoint(0, 0))
                local_rect = QRectF(tl.x(), tl.y(), target.width(), target.height())
            scene_rect = proxy.mapRectToScene(local_rect)

            overlay = self._alt_ensure_overlay()
            overlay.setPos(0, 0)
            overlay.setRect(scene_rect)
            overlay.setVisible(True)
            self._alt_update_preview_text(scene_rect)
            return

        # Non-proxy scene item (e.g. a custom QGraphicsItem). Walk to
        # the top-level so the mask covers the whole logical object
        # rather than a single internal sub-item.
        top_item = item
        while top_item.parentItem() is not None:
            top_item = top_item.parentItem()

        self._alt_hover_proxy = None
        self._alt_hover_widget = None
        self._alt_hover_item = top_item

        scene_rect = top_item.sceneBoundingRect()
        overlay = self._alt_ensure_overlay()
        overlay.setPos(0, 0)
        overlay.setRect(scene_rect)
        overlay.setVisible(True)
        self._alt_update_preview_text(scene_rect)

    def _alt_collect_hover_info(self):
        """Build a dict describing the currently hovered object.

        The returned dict is consumed by _alt_send_hovered_to_av_agent
        which converts it to plain text. The dict deliberately leads
        with a one-line 'summary' aimed at an LLM reader, followed by
        structured fields for precise reference."""
        # This wrapper just forwards to _alt_collect_hover_info_impl,
        # declared below alongside its helpers. The split lets the
        # module-level constants (_ALT_ROLE_MAP) and helper methods
        # appear before the function that uses them — easier to read
        # top-to-bottom without forward references.
        return self._alt_collect_hover_info_impl()

    # Map common QWidget subclasses to a plain-English "role" so the AV
    # agent gets "Button" instead of having to recognise "QPushButton".
    # Order matters — earlier entries take priority when subclass chains
    # match multiple rows. The first column is the class name (string),
    # the second is the role.
    _ALT_ROLE_MAP = [
        ("QPushButton",      "Button"),
        ("QToolButton",      "Tool Button"),
        ("QCheckBox",        "Checkbox"),
        ("QRadioButton",     "Radio Button"),
        ("QComboBox",        "Dropdown"),
        ("QLineEdit",        "Text Field"),
        ("QPlainTextEdit",   "Text Area"),
        ("QTextEdit",        "Text Area"),
        ("QSpinBox",         "Number Input"),
        ("QDoubleSpinBox",   "Number Input"),
        ("QSlider",          "Slider"),
        ("QDial",            "Dial"),
        ("QProgressBar",     "Progress Bar"),
        ("QCalendarWidget",  "Calendar"),
        ("QDateEdit",        "Date Picker"),
        ("QTimeEdit",        "Time Picker"),
        ("QDateTimeEdit",    "Date/Time Picker"),
        ("QTabBar",          "Tab Bar"),
        ("QTabWidget",       "Tab Container"),
        ("QListView",        "List"),
        ("QListWidget",      "List"),
        ("QTreeView",        "Tree"),
        ("QTreeWidget",      "Tree"),
        ("QTableView",       "Table"),
        ("QTableWidget",     "Table"),
        ("QMenu",            "Menu"),
        ("QMenuBar",         "Menu Bar"),
        ("QToolBar",         "Toolbar"),
        ("QScrollBar",       "Scrollbar"),
        ("QGroupBox",        "Group Box"),
        ("QFrame",           "Frame"),
        ("QSplitter",        "Splitter"),
        ("QStatusBar",       "Status Bar"),
        ("QDockWidget",      "Dock"),
        ("QLabel",           "Label"),
    ]

    def _alt_describe_widget_role(self, widget):
        """Return a human role string like 'Button' for a QWidget, walking
        the MRO so subclasses still match (e.g. a custom MyButton(QPushButton)
        is still reported as 'Button')."""
        try:
            mro_names = [cls.__name__ for cls in type(widget).__mro__]
        except Exception:
            return None
        for cls_name, role in self._ALT_ROLE_MAP:
            if cls_name in mro_names:
                return role
        return None

    # Composite widgets whose internal sub-widgets should NEVER be
    # reported as the click target. If childAt() lands inside one of
    # these, the magic-pointer reports the composite itself instead.
    # Keyed by class-name strings so the check works without importing
    # every Qt class up front.
    _ALT_COMPOSITE_CLASSES = frozenset({
        "QCalendarWidget",
        "QComboBox",
        "QSpinBox", "QDoubleSpinBox",
        "QDateEdit", "QTimeEdit", "QDateTimeEdit",
        "QTabBar", "QTabWidget",
        "QToolBar",
        "QMenu", "QMenuBar",
        "QProgressBar",
        "QSlider", "QDial",
        "QScrollBar",
        "QStatusBar",
        "QDockWidget",
        "QSplitter",
        "QHeaderView",
    })

    def _alt_promote_to_meaningful_widget(self, target, root):
        """Walk up from `target` (which came from QWidget.childAt) and
        decide whether to promote it to a more meaningful ancestor.

        Rule of thumb: composite widgets (QCalendarWidget, QComboBox,
        QSpinBox, ...) own internal sub-widgets that are not part of
        the public API — `qt_calendar_navigationbar`, `qt_scrollarea_viewport`,
        and so on. If `target` lives inside one of those composites,
        we always report the composite, never the internal piece.

        For everything else, drill-down is welcome: a QPushButton inside
        a QFrame stays as the button.
        """
        if target is None or target is root:
            return target

        # First pass: look upward for a composite ancestor. Composites
        # always win, even if the immediate target also has a recognised
        # role (e.g. QToolButton#qt_calendar_prevmonth — that's a button,
        # but it's a Qt-internal button inside a calendar; report the
        # calendar instead).
        walker = target
        while walker is not None and walker is not root.parent():
            try:
                cls_chain = {cls.__name__ for cls in type(walker).__mro__}
            except Exception:
                cls_chain = set()
            if walker is not target and (cls_chain & self._ALT_COMPOSITE_CLASSES):
                return walker
            if walker is root:
                break
            walker = walker.parent()

        # No composite ancestor: keep the immediate target if it carries
        # a useful identity. "Useful" = a role we recognise OR a
        # user-set objectName (objectNames starting with "qt_" are Qt
        # internals and don't count).
        if self._alt_describe_widget_role(target) is not None:
            return target
        try:
            obj_name = target.objectName() or ""
            if obj_name and not obj_name.startswith("qt_"):
                return target
        except Exception:
            pass

        # Otherwise walk up to the first ancestor with a role, stopping
        # at the proxy root.
        walker = target.parent()
        while walker is not None and walker is not root.parent():
            if self._alt_describe_widget_role(walker) is not None:
                return walker
            if walker is root:
                return root
            walker = walker.parent()
        return target

    def _alt_extract_widget_label(self, widget):
        """Pull the most informative human-readable text out of a widget.
        Tries text(), title(), currentText(), value() in priority order,
        plus a small set of fallbacks. Returns (label, source_attr) or
        (None, None)."""
        candidates = [
            "text",            # QPushButton, QLabel, QCheckBox, QRadioButton, QLineEdit
            "currentText",     # QComboBox, QTabBar
            "title",           # QGroupBox, QDockWidget
            "windowTitle",     # top-level widgets
            "placeholderText", # QLineEdit when empty
        ]
        for attr in candidates:
            getter = getattr(widget, attr, None)
            if not callable(getter):
                continue
            try:
                value = getter()
            except Exception:
                continue
            if value is None:
                continue
            s = str(value).strip()
            if s:
                return s[:200], attr
        # Numeric widgets — give the current value
        for attr in ("value", "currentIndex"):
            getter = getattr(widget, attr, None)
            if not callable(getter):
                continue
            try:
                v = getter()
                if v is not None:
                    return str(v), attr
            except Exception:
                continue
        return None, None

    def _alt_deep_search_namespace(self, target, max_depth=3):
        """Breadth-first search for `target` by object identity through
        the application's namespaces. Returns the shortest dotted path
        ("dial_idx", "panel.dial_idx", "app.controls.dial_idx") or
        None.

        Search roots:
          * The execution context's instance attributes
            (vars(self.execution_context)).
          * Every dict-typed attribute hanging off the execution
            context — this is where the executor stores its globals
            (the namespace `exec()` writes into). Different parsers
            name it differently (globals, namespace, env, …) so we
            scan all dict attrs rather than hard-coding a name.

        Walking strategy: BFS, so the first hit is the shortest-path
        hit. Each node yields child names from either dict items (for
        dicts) or `vars(node)` (for objects). We skip dunders,
        callables, modules, classes, and large standard containers
        (sets/tuples) — those rarely carry user names. Lists are
        scanned by index to handle e.g. `panels[0].dial_idx`.
        """
        if target is None:
            return None

        ctx = getattr(self, "execution_context", None)
        if ctx is None:
            return None

        # ---- Collect root namespaces to start the BFS from. ----
        # Each root is a (label_prefix, container) pair. label_prefix
        # is empty for the executor's globals (so a top-level user
        # name shows as just "dial_idx"), and the attribute name
        # otherwise.
        roots = []
        try:
            ctx_vars = vars(ctx)
        except Exception:
            ctx_vars = {}

        # Every dict-typed attribute is a candidate globals namespace.
        # The single most common case is `exec_globals` / `globals`
        # being a plain dict.
        for attr_name, attr_value in ctx_vars.items():
            if attr_name.startswith("__"):
                continue
            if isinstance(attr_value, dict):
                # User-name dicts get no prefix — top-level user vars
                # should read as just their name.
                roots.append(("", attr_value))

        # Also treat the context object itself as a root (covers
        # `self.dial_idx = ...` set directly on the context).
        roots.append(("", ctx_vars))

        # ---- BFS ----
        # Queue entries are (dotted_path, container). Container is
        # either a dict (label is a key) or a non-dict object (label
        # is an attribute name in vars(container)).
        from collections import deque
        queue = deque()
        for prefix, container in roots:
            queue.append((prefix, container, 0))

        # Visited set keyed by id(container) — avoids re-walking
        # cycles and the same dict reached via multiple roots.
        visited = set()

        # Cheap "should I recurse into this child?" test.
        def _is_walkable(v):
            # Don't recurse into primitives, large standard containers
            # that don't typically hold user names, or imported things.
            if v is None:
                return False
            if isinstance(v, (int, float, str, bytes, bool, complex)):
                return False
            if isinstance(v, (set, frozenset, tuple)):
                return False
            if callable(v) and not hasattr(v, "__dict__"):
                # bare functions / built-ins
                return False
            # Modules / classes: skip; they almost never hold the
            # user's live widget references.
            import types
            if isinstance(v, (types.ModuleType, type)):
                return False
            return True

        while queue:
            prefix, container, depth = queue.popleft()
            cid = id(container)
            if cid in visited:
                continue
            visited.add(cid)

            # Enumerate children of this container.
            try:
                if isinstance(container, dict):
                    iterator = container.items()
                elif isinstance(container, list):
                    iterator = ((str(i), v) for i, v in enumerate(container))
                else:
                    iterator = vars(container).items()
            except Exception:
                continue

            # First pass: direct hit at this level (so we report the
            # shallowest match even if a deeper one exists).
            children_to_recurse = []
            for name, value in iterator:
                # Skip dunders & private internals — almost never user
                # names, and some of them recurse infinitely (e.g.
                # __builtins__).
                if isinstance(name, str) and name.startswith("_"):
                    continue
                # Identity hit?
                if value is target:
                    # Build the dotted path.
                    if isinstance(container, list):
                        # e.g. "panels[0].dial_idx" — bracket notation
                        return f"{prefix}[{name}]" if prefix else f"[{name}]"
                    if prefix:
                        return f"{prefix}.{name}"
                    return str(name)
                # Queue children for the next BFS level.
                if depth + 1 < max_depth and _is_walkable(value):
                    if isinstance(container, list):
                        child_prefix = f"{prefix}[{name}]" if prefix else f"[{name}]"
                    elif prefix:
                        child_prefix = f"{prefix}.{name}"
                    else:
                        child_prefix = str(name)
                    children_to_recurse.append((child_prefix, value))

            # Queue all walkable children after we've checked every
            # child at this level — guarantees BFS ordering and so
            # guarantees the shortest path on the first identity hit.
            for child_prefix, child_value in children_to_recurse:
                if id(child_value) not in visited:
                    queue.append((child_prefix, child_value, depth + 1))

        return None

    def _alt_find_var_via_gc(self, target):
        """Last-resort variable-name lookup using gc.get_referrers.

        The execution pipeline (parse_file._execute_code) runs user
        code in an executor whose globals dict isn't directly reachable
        from RioWindow — it lives inside the filesystem's parse-file
        machinery and may be wrapped behind coroutines. Rather than
        guess where that dict is, we ask Python's garbage collector
        which objects currently hold references to the target widget,
        then look for one that's a dict with the widget as a value.
        The corresponding key is the variable name the user typed.

        Returns the variable name (a string) or None. Always returns a
        string without a dotted prefix — the deep-namespace caller is
        responsible for combining it with a container variable when
        appropriate.
        """
        if target is None:
            return None
        try:
            import gc
        except Exception:
            return None

        try:
            referrers = gc.get_referrers(target)
        except Exception:
            return None

        # `gc.get_referrers` returns the local refs of this frame too,
        # which would pollute the search if we're not careful. Skip the
        # caller frames by remembering our own locals/globals.
        try:
            import sys as _sys
            frame = _sys._getframe(0)
            blocked_ids = set()
            # Walk a few frames up to capture this method's locals
            # plus the caller frames' locals/globals so we don't
            # match against our own references.
            f = frame
            for _ in range(8):
                if f is None:
                    break
                try:
                    blocked_ids.add(id(f.f_locals))
                    blocked_ids.add(id(f.f_globals))
                except Exception:
                    pass
                f = f.f_back
        except Exception:
            blocked_ids = set()

        # Score each candidate dict so we pick the most plausible
        # variable name. Heuristics:
        #   * Skip frames' f_locals / f_globals we identified above.
        #   * Skip dicts that look like Qt internal bookkeeping
        #     (object-name-keyed children, etc.).
        #   * Prefer dicts with short identifier-like string keys
        #     (the typical user-globals shape).
        #   * Prefer dicts that contain other widget-like values too
        #     (suggests it's a UI-construction namespace).
        best_name = None
        best_score = -1

        try:
            from PySide6.QtWidgets import QWidget as _QW
        except Exception:
            _QW = None

        for ref in referrers:
            if not isinstance(ref, dict):
                continue
            if id(ref) in blocked_ids:
                continue

            # Find the key whose value IS the target.
            name_for_target = None
            try:
                for k, v in ref.items():
                    if v is target and isinstance(k, str):
                        # Must look like a Python identifier — Qt internals
                        # sometimes use weird keys.
                        if k.isidentifier() and not k.startswith("_"):
                            name_for_target = k
                            break
            except Exception:
                continue

            if name_for_target is None:
                continue

            # Score this candidate dict.
            score = 0
            # Bigger user namespaces (those containing multiple widgets)
            # are likely the executor globals — prefer them.
            try:
                widget_neighbours = 0
                for v in ref.values():
                    if _QW is not None and isinstance(v, _QW) and v is not target:
                        widget_neighbours += 1
                        if widget_neighbours >= 5:
                            break
                score += widget_neighbours * 2
            except Exception:
                pass
            # Penalise dicts that look like Qt-internal bookkeeping
            # (most values are not widgets, lots of dunder keys).
            try:
                dunder_keys = sum(
                    1 for k in ref.keys()
                    if isinstance(k, str) and k.startswith("__")
                )
                if dunder_keys > 3:
                    # Probably a module / __dict__ with imports; still
                    # acceptable but de-prioritise.
                    score -= 1
            except Exception:
                pass

            if score > best_score:
                best_score = score
                best_name = name_for_target

        return best_name

    def _alt_collect_hover_info_impl(self):
        """Implementation for _alt_collect_hover_info — see that method's
        docstring."""
        info = {}
        widget = self._alt_hover_widget
        proxy = self._alt_hover_proxy
        item = self._alt_hover_item

        # The variable-name lookup can be expensive (gc.get_referrers
        # walks the heap on the fallback path), and _alt_collect_hover_info
        # is called on every mouse move while Alt is held. Cache per
        # widget id so repeated hovers on the same widget pay the cost
        # only once. The cache lives on the magic-pointer session: it
        # gets seeded lazily here and cleared when magic mode ends.
        if not hasattr(self, "_alt_var_name_cache"):
            self._alt_var_name_cache = {}

        def _find_var_name(target):
            """Best-effort lookup of an attribute name whose value IS
            `target` (object identity, not equality).

            Search roots, in priority order:
              1. RioWindow itself (`self`).
              2. The QObject parent chain — useful when the widget is a
                 direct attribute of its containing user-subclassed
                 widget (e.g. `class Panel(QFrame): self.dial = QDial()`).
              3. The execution context's instance dict, then any
                 dict-typed attributes hanging off it. This is the path
                 to find names set by user code injected via
                 _inject_via_parse — `dial_idx = QDial(...)` lands in
                 the executor's globals dict, NOT on any widget.

            Returns a dotted-path string ("self.foo", "Panel.dial",
            "panel.dial_idx") or None.
            """
            if target is None:
                return None

            # Cache hit?
            cache_key = id(target)
            cached = self._alt_var_name_cache.get(cache_key)
            if cached is not None:
                # Cached value can be the sentinel "" meaning "we tried
                # before and found nothing" — return None for that.
                return cached or None

            # Don't report ourselves — the overlay item, our hover-state
            # bookkeeping, etc.
            _BLOCKED = {
                "_alt_overlay_item", "_alt_hover_item",
                "_alt_hover_proxy", "_alt_hover_widget",
                "_alt_preview_text", "_alt_fading_out_item",
            }

            def _remember(value):
                """Store the resolved name (or sentinel) in the cache
                and return it."""
                self._alt_var_name_cache[cache_key] = value or ""
                return value

            # 1. Direct attributes on RioWindow.
            try:
                for name, value in vars(self).items():
                    if name in _BLOCKED:
                        continue
                    if value is target:
                        return _remember(f"self.{name}")
            except Exception:
                pass

            # 2. Walk the QObject parent chain — finds e.g.
            #    Panel(QFrame) instances where the user wrote
            #    self.dial_idx = QDial(self).
            try:
                from PySide6.QtWidgets import QWidget as _QW
                if isinstance(target, _QW):
                    parent = target.parent()
                    depth = 0
                    while parent is not None and depth < 6:
                        try:
                            for name, value in vars(parent).items():
                                if value is target:
                                    return _remember(
                                        f"{type(parent).__name__}.{name}"
                                    )
                        except Exception:
                            pass
                        parent = parent.parent()
                        depth += 1
            except Exception:
                pass

            # 3. Deep scan of the execution context (user-injected code
            #    namespace). Returns the shortest dotted path.
            try:
                found = self._alt_deep_search_namespace(target)
                if found is not None:
                    return _remember(found)
            except Exception:
                pass

            # 4. Final fallback: ask Python's garbage collector who
            #    currently holds a reference to this widget. The user's
            #    executor globals dict — wherever it lives in the
            #    filesystem-pipeline plumbing — is reachable this way
            #    even when it isn't an attribute of anything we can
            #    enumerate via vars(). Costs more than the cheap paths
            #    above, so it runs only when those failed.
            try:
                gc_name = self._alt_find_var_via_gc(target)
                if gc_name:
                    return _remember(gc_name)
            except Exception:
                pass

            # Nothing found — cache the negative result so we don't
            # retry on every mouse move.
            return _remember(None)

        if widget is not None:
            try:
                cls_name = type(widget).__name__
                role = self._alt_describe_widget_role(widget)
                label, label_attr = self._alt_extract_widget_label(widget)
                obj_name = widget.objectName() or None
                var_name = _find_var_name(widget)

                # Build the lead summary — what the agent will read first.
                # Pattern: "<Role> labeled '<Label>' (<Class>)"
                #      or: "<Role> (<Class>)"
                #      or: "<Class> instance"
                pieces = []
                if role:
                    pieces.append(role)
                else:
                    pieces.append(cls_name)
                if label:
                    pieces.append(f"labeled \"{label}\"")
                if role:
                    pieces.append(f"({cls_name})")
                summary = " ".join(pieces)
                info["summary"] = summary

                info["class"] = cls_name
                if role:
                    info["role"] = role
                if label:
                    info["label"] = label
                if obj_name:
                    info["object_name"] = obj_name
                if var_name:
                    info["variable_name"] = var_name

                # Tooltip is often the most useful disambiguator and is
                # rarely the same as the label.
                tt = None
                tt_getter = getattr(widget, "toolTip", None)
                if callable(tt_getter):
                    try:
                        tt = tt_getter()
                    except Exception:
                        tt = None
                if tt:
                    info["tooltip"] = str(tt)[:200]

                # For QLineEdit etc., also surface a separate placeholder
                # when we ended up using 'text' as the label (which may
                # have been empty if the field is unfilled).
                if label_attr != "placeholderText":
                    ph_getter = getattr(widget, "placeholderText", None)
                    if callable(ph_getter):
                        try:
                            ph = ph_getter()
                            if ph:
                                info["placeholder"] = str(ph)[:200]
                        except Exception:
                            pass

                # State info for stateful widgets.
                state_bits = []
                for attr, formatter in (
                    ("isChecked",  lambda v: f"checked={v}"),
                    ("isEnabled",  lambda v: None if v else "disabled"),
                    ("isVisible",  lambda v: None if v else "hidden"),
                    ("isReadOnly", lambda v: "readonly" if v else None),
                ):
                    getter = getattr(widget, attr, None)
                    if not callable(getter):
                        continue
                    try:
                        v = getter()
                        formatted = formatter(v)
                        if formatted:
                            state_bits.append(formatted)
                    except Exception:
                        pass
                if state_bits:
                    info["state"] = ", ".join(state_bits)

                # Containing proxy info — what window/panel this widget
                # lives in. Useful for the agent to anchor where in the
                # app the action happened.
                if proxy is not None and proxy.widget() is not None:
                    container = proxy.widget()
                    info["container_class"] = type(container).__name__
                    container_var = _find_var_name(container)
                    if container_var:
                        info["container_variable"] = container_var
                    # If the click target IS the proxy's root widget,
                    # the "container" repeats the click target — flag
                    # that so the summary line doesn't sound redundant.
                    if container is widget:
                        info["is_container_root"] = True

                # Concise widget-tree path: container > intermediate > clicked
                # Limited to 4 hops for readability.
                chain = []
                p = widget.parent()
                while p is not None and len(chain) < 4:
                    chain.append(type(p).__name__)
                    p = p.parent()
                if chain:
                    # Most-distant ancestor first, then the clicked widget
                    chain_str = " > ".join(reversed(chain)) + " > " + cls_name
                    info["widget_path"] = chain_str

                # Scene rect — only useful when the agent needs to do
                # spatial reasoning. Keep it but pushed to the end.
                if proxy is not None:
                    try:
                        local_rect = proxy.subWidgetRect(widget)
                        sr = proxy.mapRectToScene(local_rect)
                        info["scene_rect"] = {
                            "x": sr.x(), "y": sr.y(),
                            "width": sr.width(), "height": sr.height(),
                        }
                    except Exception:
                        pass
            except Exception as e:
                info["error"] = f"Failed to introspect widget: {e}"
        elif item is not None:
            cls_name = type(item).__name__
            info["summary"] = f"{cls_name} (scene item)"
            info["class"] = cls_name
            var_name = _find_var_name(item)
            if var_name:
                info["variable_name"] = var_name
            sr = item.sceneBoundingRect()
            info["scene_rect"] = {
                "x": sr.x(), "y": sr.y(),
                "width": sr.width(), "height": sr.height(),
            }
        else:
            return None
        return info

    def _alt_build_payload(self):
        """Build the plain-text payload that would be sent to the AV agent
        for the currently hovered widget. Returns the string, or None if
        nothing is hovered.

        Shared by _alt_send_hovered_to_av_agent (on click) and the live
        preview text inside the highlight mask (on hover), so the user
        sees exactly what will be sent before they click."""
        info = self._alt_collect_hover_info()
        if not info:
            return None

        lines = ["[SYSTEM: User Alt-clicked a UI element. Details:]"]

        if "summary" in info:
            lines.append(f"clicked: {info['summary']}")

        # Variable info up front — this is what the AV agent needs first
        # to actually act on the widget (read its value, change its
        # attributes, hook a signal, etc).
        var_name = info.get("variable_name")
        container_var = info.get("container_variable")
        if var_name:
            lines.append(f"variable: {var_name}")
            # If we also resolved the containing widget to a variable
            # AND the click target is NOT the container root itself,
            # surface a parent.child convenience form so the agent
            # doesn't have to assemble it.
            if (container_var
                    and not info.get("is_container_root")
                    and container_var != var_name
                    and "." not in var_name
                    and "[" not in var_name):
                lines.append(f"qualified: {container_var}.{var_name}")
        if container_var and container_var != var_name:
            lines.append(f"container_variable: {container_var}")

        for key, prefix in (
            ("label",                "label"),
            ("tooltip",              "tooltip"),
            ("placeholder",          "placeholder"),
            ("state",                "state"),
            ("object_name",          "object_name"),
            ("container_class",      "lives_in"),
            ("widget_path",          "widget_path"),
        ):
            if key in info and info[key] is not None:
                if key == "container_class" and info.get("is_container_root"):
                    continue
                lines.append(f"{prefix}: {info[key]}")

        if "scene_rect" in info:
            r = info["scene_rect"]
            lines.append(
                f"scene_rect: x={r['x']:.0f} y={r['y']:.0f} "
                f"w={r['width']:.0f} h={r['height']:.0f}"
            )
        if "error" in info:
            lines.append(f"error: {info['error']}")

        return "\n".join(lines)

    def _alt_send_hovered_to_av_agent(self):
        """Ship the hovered widget's identity to the AV agent input file
        (same channel _send_flicker_context_to_ai uses)."""
        payload = self._alt_build_payload()
        if not payload:
            return

        try:
            agent_input = os.path.join(
                self.voice_control.llmfs_mount, "av", "input"
            )
        except Exception as e:
            print(f"[AltMagicPointer] Could not resolve agent input path: {e}")
            return

        def _write():
            try:
                with open(agent_input, 'w') as f:
                    f.write(payload)
                print(f"[AltMagicPointer] Sent inspector info ({len(payload)} chars) to AI.")
            except Exception as e:
                print(f"[AltMagicPointer] Error sending inspector info: {e}")

        threading.Thread(target=_write, daemon=True).start()

    def _alt_add_hovered_to_selection(self):
        """Snapshot the currently hovered widget into the multi-select
        pool. Dedup is by id(target) — clicking the same widget twice is
        a no-op (no toast, no error; matches "I already picked that").
        Payload is built and stored *now* so later scene mutations can't
        change what eventually gets sent."""
        # Pick the most specific handle Qt gave us for this hover.
        target = (self._alt_hover_widget
                  or self._alt_hover_proxy
                  or self._alt_hover_item)
        if target is None:
            return
        key = id(target)
        if key in self._alt_selection:
            print(f"[AltMagicPointer] Already in selection ({len(self._alt_selection)} total).")
            return
        payload = self._alt_build_payload()
        if not payload:
            return
        self._alt_selection[key] = payload
        print(f"[AltMagicPointer] Added to selection ({len(self._alt_selection)} total).")

    def _alt_flush_selection_to_av_agent(self):
        """Send the accumulated multi-selection as a single AV-agent
        message and clear the pool. Returns True if anything was sent."""
        if not self._alt_selection:
            return False

        picks = list(self._alt_selection.values())
        header = (
            f"[SYSTEM: User Ctrl+S-picked {len(picks)} UI element"
            f"{'s' if len(picks) != 1 else ''} for inspection.]"
        )
        # Each per-widget payload already starts with its own
        # "[SYSTEM: ...]" line from _alt_build_payload; separate with a
        # blank line + rule so the agent can tell items apart.
        body = "\n\n--- next element ---\n\n".join(picks)
        combined = f"{header}\n\n{body}"

        try:
            agent_input = os.path.join(
                self.voice_control.llmfs_mount, "av", "input"
            )
        except Exception as e:
            print(f"[AltMagicPointer] Could not resolve agent input path: {e}")
            self._alt_selection.clear()
            return False

        count = len(picks)
        # Clear before the background write so a re-activation can't see
        # stale state if the user is fast.
        self._alt_selection.clear()

        def _write():
            try:
                with open(agent_input, 'w') as f:
                    f.write(combined)
                print(f"[AltMagicPointer] Sent {count} selection(s), "
                      f"{len(combined)} chars, to AI.")
            except Exception as e:
                print(f"[AltMagicPointer] Error sending selection: {e}")

        threading.Thread(target=_write, daemon=True).start()
        return True

    def _init_debug_overlay(self):
        """Create the debug overlay widget on the main window (not the scene).
        It lives as a direct child of the main window and floats in the top-right."""
        self.debug_overlay = DebugOverlayWidget(self)
        self.debug_overlay.setVisible(False)
        self.debug_overlay.reposition(self.width())
        
        # Register with DebugNode so any DebugNode instance can push messages
        try:
            from rio.operator_panel import DebugNode
        except ImportError:
            try:
                from operator_panel import DebugNode
            except ImportError:
                DebugNode = None
        if DebugNode is not None:
            DebugNode._overlay_ref = self.debug_overlay
    
    def _toggle_voice_control(self):
        """Toggle AI Voice Control widget visibility.
        On show: position at mouse cursor in scene coords and play draw-on animation."""
        vis = self.voice_control_proxy.isVisible()
        if vis:
            self.voice_control_proxy.setVisible(False)
        else:
            # Position centered on mouse cursor in scene coordinates
            global_pos = QCursor.pos()
            viewport_pos = self.graphics_view.mapFromGlobal(global_pos)
            scene_pos = self.graphics_view.mapToScene(viewport_pos)
            # Offset so the eyes are centered on the cursor
            w = self.voice_control.width()
            h = self.voice_control.height()
            self.voice_control_proxy.setPos(
                scene_pos.x() - w / 2,
                scene_pos.y() - h / 2,
            )
            self.voice_control_proxy.setVisible(True)
            self.voice_control.start_intro_animation()

    # ------------------------------------------------------------------
    # Dark Mode
    # ------------------------------------------------------------------

    @property
    def dark_mode(self) -> bool:
        return self._dark_mode

    def toggle_dark_mode(self):
        """Toggle between light and dark mode with animated transitions.

        Animates:
          1. Scene background  (light ↔ dark)
          2. Every QGraphicsDropShadowEffect on every proxy in the scene
             (dark shadow ↔ white shadow)
          3. Every TerminalWidget: frame border, text color, input styling

        Performance: uses a SINGLE QTimer to batch-update the background
        and ALL shadow effects together, instead of spawning one timer per
        shadow.  This reduces timer overhead from O(N) to O(1) and batches
        all repaint-triggering calls into one event-loop tick so Qt can
        coalesce the scene updates.
        """
        self._dark_mode = not self._dark_mode
        entering_dark = self._dark_mode

        duration_steps = 50  # ~800 ms at 16 ms/tick

        # --- Unified dark-mode animation (background + all shadows) ---
        self._start_dark_mode_animation(entering_dark, duration_steps)

        # --- Terminal widgets (these manage their own internal styling) ---
        for terminal in self.terminals:
            terminal.set_dark_mode(entering_dark, duration_steps)
            if hasattr(terminal, 'operator_panel') and terminal.operator_panel is not None:
                terminal.operator_panel.set_dark_mode(entering_dark, duration_steps)
            if hasattr(terminal, 'version_panel') and terminal.version_panel is not None:
                terminal.version_panel.set_dark_mode(entering_dark, duration_steps)

    # ---- unified dark-mode animation (single timer) ----

    def _start_dark_mode_animation(self, to_dark: bool, steps: int):
        """
        Run a single batch-update for the scene background and all shadows.
        Uses QVariantAnimation to prevent lag when many widgets are present.

        Targets (background colour, shadow tint) are sourced from the
        active theme so a "paper" scene fades to its warm-cream
        canvas while a "glass" scene fades to off-white.  When the
        active theme has no shadows (``theme.shadow is None``), the
        shadow loop is skipped — the only work done is the bg fade.

        Performance: uses self._shadowed_items rather than
        self.graphics_scene.items() so we don't pay an O(N_total) scan
        of every item including non-shadowed ones. We also flip every
        shadow's setEnabled(False) for the duration of the animation —
        the Gaussian-blur re-rasterization of each shadowed proxy is by
        far the heaviest paint cost in a Glass-theme scene, and the
        colour interpolation isn't visually meaningful when the host
        widget is itself fading. Effects are re-enabled (with final
        colours) on completion.
        """
        # 1. Kill any in-flight animation to prevent "Use-After-Free" crashes
        if hasattr(self, '_dark_mode_bg_anim') and self._dark_mode_bg_anim:
            self._dark_mode_bg_anim.stop()
            self._dark_mode_bg_anim.deleteLater()
            self._dark_mode_bg_anim = None

        theme = self.current_theme

        # 2. Snapshot Background State
        brush = self.graphics_scene.backgroundBrush()
        start_bg = brush.color()
        bg_sr, bg_sg, bg_sb = start_bg.red(), start_bg.green(), start_bg.blue()

        # Target Background Colors come from the active theme
        bg_tr, bg_tg, bg_tb = (
            theme.scene_bg_rgb_dark if to_dark else theme.scene_bg_rgb
        )

        # 3. Snapshot Shadow States — only meaningful if theme uses shadows.
        # Pull directly from our maintained cache; no scene scan.
        shadow_targets = []
        ts_r = ts_g = ts_b = ts_a = 0
        if theme.shadow is not None:
            target_rgba = (
                theme.shadow.color_dark if to_dark else theme.shadow.color
            )
            ts_r, ts_g, ts_b, ts_a = target_rgba

            for _item, effect in self._iter_live_shadowed():
                shadow_targets.append({
                    'effect': effect,
                    'sr': effect.color().red(),
                    'sg': effect.color().green(),
                    'sb': effect.color().blue(),
                    'sa': effect.color().alpha(),
                })

        # 3b. Disable shadow effects for the duration of the animation.
        # Cheaper than blurring N proxies per frame; we restore final
        # colours and re-enable on completion.
        self._set_shadows_enabled_during_animation(False)

        # 4. Initialize the Batch Animation
        anim = QVariantAnimation(self)
        anim.setDuration(steps * 16) # Maintain the intended timing
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutQuad) # Smoothstep equivalent

        # Reusable scratch objects so the per-tick path doesn't allocate
        # a fresh QBrush/QColor on every frame. With ~50 shadowed proxies
        # over ~50 ticks this skips ~2500 short-lived QObjects per
        # dark-mode toggle.
        bg_color = QColor()
        bg_brush = QBrush(bg_color)
        shadow_color = QColor()

        def update_scene_batch(t):
            # -- Interpolate and set Background --
            r = int(bg_sr + (bg_tr - bg_sr) * t)
            g = int(bg_sg + (bg_tg - bg_sg) * t)
            b = int(bg_sb + (bg_tb - bg_sb) * t)
            bg_color.setRgb(r, g, b)
            bg_brush.setColor(bg_color)
            self.graphics_scene.setBackgroundBrush(bg_brush)

            # Update colour on the (currently disabled) shadow effects.
            # We still want them to hold the right end-state colour so
            # that when we re-enable on finish there's no visible jump.
            for data in shadow_targets:
                sr = int(data['sr'] + (ts_r - data['sr']) * t)
                sg = int(data['sg'] + (ts_g - data['sg']) * t)
                sb = int(data['sb'] + (ts_b - data['sb']) * t)
                sa = int(data['sa'] + (ts_a - data['sa']) * t)
                shadow_color.setRgb(sr, sg, sb, sa)
                data['effect'].setColor(shadow_color)

        def on_finished():
            # Restore shadow rendering with the final interpolated colour
            self._set_shadows_enabled_during_animation(True)

        anim.valueChanged.connect(update_scene_batch)
        anim.finished.connect(on_finished)

        # 5. Store and start
        self._dark_mode_bg_anim = anim
        anim.start()

    # ------------------------------------------------------------------
    # Theme switching
    # ------------------------------------------------------------------

    @property
    def current_theme(self) -> Theme:
        """The currently active Theme object."""
        return get_theme(self._active_theme_name)

    @property
    def theme_name(self) -> str:
        return self._active_theme_name

    def set_theme(self, name: str, animate: bool = True, duration_ms: int = 800):
        """Switch the active theme, optionally with an animated transition.

        Animates:
          - Scene background colour
          - Every existing drop shadow (interpolating colour, blur, offset
            — offset animates to (0,0) for paper, from (0,0) for glass).
            If the new theme has no shadows, effects are removed at the end.
            If the old theme had no shadows but the new one does, fresh
            effects are installed and animated in.
          - Each terminal/operator/version panel re-applies its frame & input
            QSS (snapshot → animate via QVariantAnimation, identical mechanism
            to the dark-mode transitions).
          - On paper: each terminal gets a distinct random paper-pastel
            fill so the scene and terminals form a cohesive palette.

        Idempotent: switching to the already-active theme is a no-op.
        Concurrent switches are coalesced — the in-flight animation is
        stopped and replaced.
        """
        if name not in THEMES:
            logger.warning(f"[theme] unknown theme '{name}', ignoring")
            return
        if name == self._active_theme_name:
            return

        old = self.current_theme
        self._active_theme_name = name
        new = self.current_theme
        dark = self._dark_mode

        # Cancel any in-flight theme animation
        if self._theme_anim is not None:
            self._theme_anim.stop()
            self._theme_anim.deleteLater()
            self._theme_anim = None

        # Generate a paper palette when switching TO paper so every
        # terminal gets a distinct pastel fill and the scene bg is
        # a complementary colour.  The palette is ignored for glass
        # (terminals revert to the theme's translucent fill).
        paper_scene_rgb = None
        paper_term_rgbs = []
        if new.shadow is None and self.terminals:
            paper_scene_rgb, paper_term_rgbs = self._generate_paper_palette(
                len(self.terminals)
            )

        # Tell each terminal / panel to re-skin itself.  The widgets read
        # the current theme via self._theme accessor we add to them.
        for idx, terminal in enumerate(self.terminals):
            if hasattr(terminal, 'apply_theme'):
                pbg = paper_term_rgbs[idx] if idx < len(paper_term_rgbs) else None
                terminal.apply_theme(
                    new, dark, duration_ms if animate else 0,
                    paper_bg_rgb=pbg,
                )
            for sub in ('operator_panel', 'version_panel'):
                panel = getattr(terminal, sub, None)
                if panel is not None and hasattr(panel, 'apply_theme'):
                    panel.apply_theme(new, dark, duration_ms if animate else 0)

        # Animate the scene background + existing shadows.
        self._animate_theme_transition(
            old, new, dark, duration_ms if animate else 0,
            paper_scene_rgb=paper_scene_rgb,
        )

    def _animate_theme_transition(self, old: Theme, new: Theme, dark: bool,
                                   duration_ms: int, *,
                                   paper_scene_rgb: tuple = None):
        """Animate scene background + existing shadow effects between themes.

        Mirrors ``_start_dark_mode_animation`` but interpolates between
        the *theme-defined* targets, not light/dark targets.

        Shadow offset animation:
          - glass → paper: offset smoothly retracts to (0, 0), blur and
            alpha fade to 0, then effects are removed on_finished.
            Shadows stay **enabled** so the retraction is visible.
          - paper → glass: fresh effects start at offset (0, 0) /
            blur 0 / alpha 0 and grow to the theme target.
            Shadows stay enabled for the same reason.

        When ``paper_scene_rgb`` is provided (switching to paper with
        a generated palette), the scene background animates toward
        that colour instead of the theme's default.
        """
        from PySide6.QtWidgets import QGraphicsDropShadowEffect as _DSE

        # Start values (snapshot current scene)
        brush = self.graphics_scene.backgroundBrush()
        sb = brush.color()
        bg_sr, bg_sg, bg_sb = sb.red(), sb.green(), sb.blue()

        # Target bg — paper palette overrides theme default
        if paper_scene_rgb is not None:
            bg_tr, bg_tg, bg_tb = paper_scene_rgb
        else:
            bg_tr, bg_tg, bg_tb = (
                new.scene_bg_rgb_dark if dark else new.scene_bg_rgb
            )

        # Persist paper scene colour so it survives dark-mode toggles
        if paper_scene_rgb is not None:
            hex_color = "#{:02x}{:02x}{:02x}".format(*paper_scene_rgb)
            try:
                self.scene_manager.background_color = hex_color
            except Exception:
                pass

        # Snapshot shadow effects.  Each gets a per-effect target:
        #   - new theme has shadow:  fade colour/blur/offset to its values
        #   - new theme has no shadow: fade colour alpha to 0, blur to 0,
        #     offset to (0,0); we delete the effect at the very end.
        # Cache-first iteration: only walks our tracked items.
        shadow_records = []
        for item, effect in self._iter_live_shadowed():
            c = effect.color()
            shadow_records.append({
                'item': item,
                'effect': effect,
                'sr': c.red(), 'sg': c.green(), 'sb': c.blue(), 'sa': c.alpha(),
                's_blur': effect.blurRadius(),
                's_off_x': effect.offset().x(),
                's_off_y': effect.offset().y(),
            })

        if new.shadow is not None:
            tr_, tg_, tb_, ta_ = (
                new.shadow.color_dark if dark else new.shadow.color
            )
            t_blur = new.shadow.blur_radius
            t_off_x = new.shadow.offset_x
            t_off_y = new.shadow.offset_y
            removing_shadows = False

            # Install fresh shadows on any proxy that doesn't yet have one.
            # Start at offset (0,0), blur 0, alpha 0 so the animation
            # smoothly grows the shadow outward from nothing.
            from PySide6.QtWidgets import QGraphicsProxyWidget
            for item in self.graphics_scene.items():
                if not isinstance(item, QGraphicsProxyWidget):
                    continue
                if item.graphicsEffect() is not None:
                    continue
                eff = _DSE()
                eff.setColor(QColor(tr_, tg_, tb_, 0))
                eff.setBlurRadius(0.0)
                eff.setOffset(QPointF(0.0, 0.0))
                item.setGraphicsEffect(eff)
                self.register_shadowed(item)
                shadow_records.append({
                    'item': item, 'effect': eff,
                    'sr': tr_, 'sg': tg_, 'sb': tb_, 'sa': 0,
                    's_blur': 0.0, 's_off_x': 0.0, 's_off_y': 0.0,
                })
        else:
            tr_, tg_, tb_, ta_ = 0, 0, 0, 0
            t_blur = 0.0
            t_off_x = 0.0
            t_off_y = 0.0
            removing_shadows = True

            # Scene scan: catch any proxies that have a shadow effect but
            # were never registered in _shadowed_items.
            from PySide6.QtWidgets import QGraphicsProxyWidget
            seen = {rec['item'] for rec in shadow_records}
            for item in self.graphics_scene.items():
                if not isinstance(item, QGraphicsProxyWidget):
                    continue
                if item in seen:
                    continue
                eff = item.graphicsEffect()
                if not isinstance(eff, _DSE):
                    continue
                c = eff.color()
                shadow_records.append({
                    'item': item,
                    'effect': eff,
                    'sr': c.red(), 'sg': c.green(), 'sb': c.blue(), 'sa': c.alpha(),
                    's_blur': eff.blurRadius(),
                    's_off_x': eff.offset().x(),
                    's_off_y': eff.offset().y(),
                })

        # Instant path
        if duration_ms <= 0:
            self.graphics_scene.setBackgroundBrush(QBrush(QColor(bg_tr, bg_tg, bg_tb)))
            for rec in shadow_records:
                if removing_shadows:
                    try:
                        rec['item'].setGraphicsEffect(None)
                        self.unregister_shadowed(rec['item'])
                    except RuntimeError:
                        pass
                else:
                    rec['effect'].setColor(QColor(tr_, tg_, tb_, ta_))
                    rec['effect'].setBlurRadius(t_blur)
                    rec['effect'].setOffset(QPointF(t_off_x, t_off_y))
            return

        # Shadows stay ENABLED during the animation so the user sees
        # the offset retract to (0,0) on glass→paper, or grow from
        # (0,0) on paper→glass.  The earlier approach of disabling
        # them (for perf) made the shadow vanish instantly — acceptable
        # for colour-only tweens but jarring when the offset itself is
        # the visual payoff of the transition.
        #
        # Performance note: this is more expensive than the disabled
        # path (~N Gaussian blurs per frame), but theme switches are
        # infrequent and the visual payoff justifies it.

        anim = QVariantAnimation(self)
        anim.setDuration(duration_ms)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutQuad)

        # Reusable scratch objects
        bg_color = QColor()
        bg_brush = QBrush(bg_color)
        shadow_color = QColor()
        shadow_offset = QPointF()

        def tick(t):
            r = int(bg_sr + (bg_tr - bg_sr) * t)
            g = int(bg_sg + (bg_tg - bg_sg) * t)
            b = int(bg_sb + (bg_tb - bg_sb) * t)
            bg_color.setRgb(r, g, b)
            bg_brush.setColor(bg_color)
            self.graphics_scene.setBackgroundBrush(bg_brush)

            for rec in shadow_records:
                sr = int(rec['sr'] + (tr_ - rec['sr']) * t)
                sg = int(rec['sg'] + (tg_ - rec['sg']) * t)
                sb_ = int(rec['sb'] + (tb_ - rec['sb']) * t)
                sa = int(rec['sa'] + (ta_ - rec['sa']) * t)
                shadow_color.setRgb(sr, sg, sb_, sa)
                rec['effect'].setColor(shadow_color)
                rec['effect'].setBlurRadius(
                    rec['s_blur'] + (t_blur - rec['s_blur']) * t
                )
                shadow_offset.setX(rec['s_off_x'] + (t_off_x - rec['s_off_x']) * t)
                shadow_offset.setY(rec['s_off_y'] + (t_off_y - rec['s_off_y']) * t)
                rec['effect'].setOffset(shadow_offset)

        def on_finished():
            if removing_shadows:
                for rec in shadow_records:
                    try:
                        rec['item'].setGraphicsEffect(None)
                        self.unregister_shadowed(rec['item'])
                    except RuntimeError:
                        pass
            self._theme_anim = None

        anim.valueChanged.connect(tick)
        anim.finished.connect(on_finished)
        self._theme_anim = anim
        anim.start()

    def _launch_onboarding(self):
        """Launch the onboarding tutorial by executing onboarding.py
        directly through the local Executor.

        We do NOT write through /n/rioa/scene/parse (9P) because:
        - 9P chunks large writes at msize (~4-8 KB boundaries)
        - If a chunk boundary splits a multi-byte UTF-8 character,
          the server rejects it with 'Invalid UTF-8'
        - Even pure-ASCII files can arrive as multiple Twrite ops,
          and the StreamingParser may execute partial code

        Instead we read the file locally and hand it to self.executor
        which runs in-process with no serialization boundary.
        """
        onboarding_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "onboarding.py"
        )
        
        if not os.path.exists(onboarding_script_path):
            print(f"[Onboarding] Script not found at {onboarding_script_path}")
            return
        
        try:
            with open(onboarding_script_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"[Onboarding] Failed to read script: {e}")
            return
        
        if not self.executor:
            print("[Onboarding] No executor available")
            return

        # Execute directly through the local executor (no 9P involved)
        async def _run():
            result = await self.executor.execute(code)
            if result.success:
                print("[Onboarding] Script executed successfully")
            else:
                print(f"[Onboarding] Script error: {result.error}")

        asyncio.create_task(_run())
    
    def _init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle(self.rio_server.workspace or "Rio")
        self.setGeometry(100, 100, 
                        self.scene_manager.width,
                        self.scene_manager.height)
        
        # Create graphics scene — large canvas with (0,0) at center.
        self.graphics_scene = QGraphicsScene()
        # BspTreeIndex (the Qt default): proxies are mostly stationary
        # between user actions, so the BSP rebuild cost is amortised, and
        # itemAt() / hit-testing becomes O(log N) instead of the O(N)
        # linear scan NoIndex performs. NoIndex is faster only if items
        # move every frame, which isn't this app's pattern.
        self.graphics_scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        scene_half = 10000  # total scene: 20000 x 20000
        self.graphics_scene.setSceneRect(
            -scene_half, -scene_half,
            scene_half * 2, scene_half * 2
        )
        
        # Set background — prefer the active theme's bg, but honour
        # any explicit override the SceneManager may already hold.
        # Glass theme's default (250,250,250) matches the legacy bg
        # so this is backward-compatible.
        theme_bg = self.current_theme.scene_bg(self._dark_mode)
        sm_bg = getattr(self.scene_manager, 'background_color', None)
        if sm_bg and sm_bg not in ('#fafafa', '#FAFAFA', '#fffeff'):
            # SceneManager has a non-default override — respect it
            bg_color = QColor(sm_bg)
        else:
            bg_color = theme_bg
        self.graphics_scene.setBackgroundBrush(QBrush(bg_color))
        
        # Create view — scrollbars hidden, panning is Ctrl+Mouse only
        self.graphics_view = QGraphicsView(self.graphics_scene)
        #self.graphics_view.setRenderHint(QPainter.Antialiasing)

        # OpenGL?
        #gl_viewport = QOpenGLWidget()
        #self.graphics_view.setViewport(gl_viewport)
        #
        
        # BoundingRectViewportUpdate: cheaper than Full because Qt only
        # repaints the union of changed-item bounding rects each frame,
        # not the whole 4K viewport. The historical reason for
        # FullViewportUpdate was drop-shadow blur extending past item
        # bounds and proxies under-reporting dirty regions; we now
        # compensate by (a) caching proxies as device-coordinate bitmaps
        # so their dirty rect reflects their visible extent, and
        # (b) disabling drop-shadow effects during theme/dark animations
        # (the only time many items repaint simultaneously). If you see
        # smearing of shadows, the right fix is to grow the affected
        # item's boundingRect() to include its shadow's blur radius —
        # NOT to revert this flag, which costs roughly 2-5x in paint CPU
        # during animation.
        #self.graphics_view.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # Painter-state and AA adjustments are skipped per-item. Safe
        # because we render most items via proxies (which manage their
        # own painter state) and disable global antialiasing on the view.
        self.graphics_view.setOptimizationFlag(
            QGraphicsView.DontSavePainterState, True
        )
        self.graphics_view.setOptimizationFlag(
            QGraphicsView.DontAdjustForAntialiasing, True
        )

        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Center alignment
        self.graphics_view.setAlignment(Qt.AlignCenter)
        
        # Enable context menu
        self.graphics_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.graphics_view.customContextMenuRequested.connect(self._show_context_menu)
        
        self.setCentralWidget(self.graphics_view)
        
        # Enable mouse tracking
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().installEventFilter(self)
        
        # Enable drag-and-drop for app launcher
        self.graphics_view.setAcceptDrops(True)
        self.graphics_view.dragEnterEvent = self._view_drag_enter
        self.graphics_view.dragMoveEvent = self._view_drag_move
        self.graphics_view.dropEvent = self._view_drop
    
    def _show_context_menu(self, pos: QPoint):
        """Show context menu on right-click — only on empty scene area"""
        # Suppress context menu if Ctrl+Right orbit was just used
        if self._ctrl_orbit_used:
            self._ctrl_orbit_used = False
            return
        
        # Check if the click hit a scene item (e.g. terminal proxy)
        scene_pos = self.graphics_view.mapToScene(pos)
        item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
        if item is not None:
            # Click landed on a scene item — don't show the window menu
            return
        
        # Create menu — Plan 9 style: clean, compact, square
        # Custom menu that doesn't auto-close on click (blink first)
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
        
        menu = _BlinkMenu(self)
        
        # Menu styling pulled from the active theme.  The "flash" colour
        # is the opposite-mode CSS — same trick the legacy hardcoded
        # version used (white-on-black for light mode, vice-versa).
        _theme = self.current_theme
        if self._dark_mode:
            _CSS_NORMAL = _theme.menu_css_dark
            _CSS_FLASH  = _theme.menu_css_light
        else:
            _CSS_NORMAL = _theme.menu_css_light
            _CSS_FLASH  = _theme.menu_css_dark
        menu.setStyleSheet(_CSS_NORMAL)
        
        _action_map = {}
        
        def _add(label, callback):
            action = menu.addAction(label)
            _action_map[action] = callback
            return action
        
        def _on_triggered(action):
            cb = _action_map.get(action)
            if cb is None:
                return
            # Find the originating menu via the action's parent — this
            # lets the same handler service both the top-level menu and
            # any submenu, flashing whichever one the user clicked in.
            src_menu = action.parent() if action is not None else None
            if not isinstance(src_menu, QMenu):
                src_menu = menu
            if not getattr(src_menu, '_blink_active', False):
                return
            src_menu._blink_active = False
            # Single blink: invert, hold, revert, close
            _step = [0]
            def _tick():
                _step[0] += 1
                if _step[0] == 1:
                    src_menu.setStyleSheet(_CSS_FLASH)
                elif _step[0] == 2:
                    src_menu.setStyleSheet(_CSS_NORMAL)
                else:
                    _timer.stop()
                    _timer.deleteLater()
                    # Always close the top-level menu — that recursively
                    # tears down any open submenu chain.
                    menu.close()
                    QTimer.singleShot(0, cb)
                    return
            _timer = QTimer(src_menu)
            _timer.timeout.connect(_tick)
            _timer.start(80)

        menu.triggered.connect(_on_triggered)
        
        _add("New Terminal", self._enter_new_terminal_mode)
        _add("Apps", self._toggle_app_launcher)
        _operator_label = "Hide Operator" if self._operator is not None else "Operator"
        _add(_operator_label, self._toggle_operator)
        menu.addSeparator()
        is_visible = self.voice_control_proxy.isVisible()
        _add("Hide AI Voice" if is_visible else "Show AI Voice", self._toggle_voice_control)
        _add("Onboarding", self._launch_onboarding)
        menu.addSeparator()
        _add("Light Mode" if self._dark_mode else "Dark Mode", self.toggle_dark_mode)
        # Theme submenu — switch between Glass / Paper, plus pick a
        # random paper-like (pastel) scene background. The submenu shares
        # _action_map and _on_triggered so the blink-on-select effect
        # works exactly like the top-level menu.
        theme_submenu = _BlinkMenu(menu)
        theme_submenu.setTitle("Theme")
        theme_submenu.setStyleSheet(_CSS_NORMAL)
        theme_submenu.triggered.connect(_on_triggered)

        def _add_sub(label, callback):
            action = theme_submenu.addAction(label)
            _action_map[action] = callback
            return action

        # Mark the active theme with a leading bullet so the user sees
        # what they're already on. Both rows are still clickable —
        # clicking the active one is a harmless re-apply.
        _glass_label = ("• " if self._active_theme_name == "glass" else "  ") + "Glass"
        _paper_label = ("• " if self._active_theme_name == "paper" else "  ") + "Paper"
        _add_sub(_glass_label, lambda: self.set_theme("glass"))
        _add_sub(_paper_label, lambda: self.set_theme("paper"))
        theme_submenu.addSeparator()
        _add_sub("Random Paper Color", self._apply_random_paper_color)

        menu.addMenu(theme_submenu)
        menu.addSeparator()
        _immersive_label = "Exit Immersive" if (hasattr(self, '_immersive_mode') and self._immersive_mode.is_active) else "Immersive Mode (Ctrl+I)"
        _add(_immersive_label, lambda: self._immersive_mode.toggle() if hasattr(self, '_immersive_mode') else None)
        menu.addSeparator()
        _add("Clear Scene", self._clear_scene)
        _add("Refresh", self.scene_manager.refresh)
        _add("Delete Widget", self._enter_delete_mode)
        if self._popped_widgets:
            _add("Dock All", self._dock_all_widgets)
        _add("Pop Widget", self._enter_pop_mode)
        menu.addSeparator()
        _fullscreen_label = "Exit Fullscreen" if self.isFullScreen() else "Fullscreen"
        _add(_fullscreen_label, self._toggle_fullscreen)
        menu.addSeparator()
        _add("Exit", self._exit_app)
        
        menu.popup(self.graphics_view.mapToGlobal(pos))
    
    def _enter_new_terminal_mode(self):
        """Enter mode to create a new terminal"""
        self.new_terminal_mode = True
        self.delete_mode = False
        self.graphics_view.setCursor(Qt.CrossCursor)

    def _enter_delete_mode(self):
        """Enter mode to delete a widget by clicking on it"""
        self.delete_mode = True
        self.new_terminal_mode = False
        self.pop_mode = False
        self.graphics_view.setCursor(Qt.ForbiddenCursor)

    def _enter_pop_mode(self):
        """Enter mode to pop a widget out or dock it back by clicking on it"""
        self.pop_mode = True
        self.new_terminal_mode = False
        self.delete_mode = False
        self.graphics_view.setCursor(Qt.PointingHandCursor)
    
    def _exit_special_modes(self):
        """Exit all special modes and reset cursor"""
        self.new_terminal_mode = False
        self.delete_mode = False
        self.pop_mode = False
        self.is_creating_terminal = False
        self.graphics_view.setCursor(Qt.ArrowCursor)
    
    # ------------------------------------------------------------------
    # App Launcher
    # ------------------------------------------------------------------

    def _toggle_app_launcher(self):
        """Toggle the App Launcher widget on the scene."""
        if self._app_launcher_proxy is not None:
            # Close with animation, then remove
            widget = self._app_launcher_proxy.widget()
            if widget and hasattr(widget, 'animate_close'):
                proxy_ref = self._app_launcher_proxy
                self._app_launcher_proxy = None  # prevent double-toggle

                def _on_done():
                    w = proxy_ref.widget()
                    proxy_ref.setWidget(None)
                    self.graphics_scene.removeItem(proxy_ref)
                    if w:
                        w.deleteLater()

                widget.animate_close(on_done=_on_done)
            else:
                self._app_launcher_proxy.setWidget(None)
                self.graphics_scene.removeItem(self._app_launcher_proxy)
                self._app_launcher_proxy = None
                if widget:
                    widget.deleteLater()
            return

        launcher = AppLauncherWidget(self)
        self._app_launcher_proxy = self.graphics_scene.addWidget(launcher)
        self._app_launcher_proxy.setZValue(2000)

        global_pos = QCursor.pos()
        viewport_pos = self.graphics_view.mapFromGlobal(global_pos)
        scene_pos = self.graphics_view.mapToScene(viewport_pos)
        self._app_launcher_proxy.setPos(scene_pos.x(), scene_pos.y())

        # Animate: border draw → content fade in → shadow grow
        launcher.animate_open(self._app_launcher_proxy)

        self.scene_manager.register_infrastructure(
            self._app_launcher_proxy, label="app_launcher"
        )

    # ------------------------------------------------------------------
    # Operator (node graph overlay)
    # ------------------------------------------------------------------
    #
    # The Operator used to live in apps/operator_fs.py and was injected
    # into the scene by writing Python to the parser, which exec()'d it
    # with `graphics_scene` / `graphics_view` in scope and kept a module
    # `op` name alive across runs to implement the on/off toggle. That
    # round-trip through the parser is gone: we construct and tear down
    # the Operator directly here, holding the instance in self._operator.

    def _toggle_operator(self):
        """Toggle the Operator node-graph overlay on the scene.

        First call constructs an Operator bound to the live scene; the
        next call tears down everything it added (nodes, connections,
        toolbar, event filter, temp connection) and clears the handle so
        a subsequent call starts fresh.
        """
        if self._operator is not None:
            self._teardown_operator()
            return

        # ── Construct ──────────────────────────────────────────────────
        # Import lazily: op_core pulls in the whole node-view stack, and
        # we only want to pay that cost (and risk an import error) when
        # the user actually opens the operator.
        try:
            from rio.op_core import Operator
        except ImportError:
            from op_core import Operator

        # Probe the visible viewport so the toolbar and initial layout
        # land inside what the user can actually see. Fall back to a
        # sane default region if the mapping comes back empty.
        region = QRectF(0, 0, 1400.0, 900.0)
        try:
            vr = self.graphics_view.mapToScene(
                self.graphics_view.viewport().rect()
            ).boundingRect()
            if vr.width() > 0 and vr.height() > 0:
                region = QRectF(vr)
        except Exception as e:
            logger.warning(f"operator: viewport probe failed: {e}")

        # Mount paths come straight from the server config now, instead
        # of being re-probed via `ls /n` the way operator_fs.py did.
        try:
            self._operator = Operator(
                self.graphics_scene,
                llm_mount=self.rio_server.llmfs_mount,
                rio_mount=self.rio_server.rio_mount,
                region=region,
                dark=self._dark_mode,
            )
        except Exception as e:
            logger.exception(f"operator: construction failed: {e}")
            self._operator = None

    def _teardown_operator(self):
        """Remove everything the Operator added to the scene and drop the
        instance. Mirrors the inline teardown the old operator_fs.py did,
        because Operator.cleanup() only walks node views — it does not
        remove items from the scene, touch the toolbar, or uninstall the
        event filter."""
        op = self._operator
        self._operator = None
        if op is None:
            return

        scene = self.graphics_scene

        # Best-effort cleanup first (stops the routes subscription, etc.).
        # Don't trust it to remove anything visible.
        try:
            op.cleanup()
        except Exception as e:
            logger.warning(f"operator: cleanup() raised: {e}")

        # Uninstall the scene event filter so its handlers can't fire on
        # a half-dead Operator.
        try:
            ef = getattr(op, "_event_filter", None)
            if ef is not None:
                scene.removeEventFilter(ef)
        except Exception as e:
            logger.warning(f"operator: removeEventFilter raised: {e}")

        # Remove every connection item from the scene.
        try:
            conn_views = getattr(op, "_conn_views", {}) or {}
            for c in list(conn_views.values()):
                try:
                    scene.removeItem(c)
                except Exception:
                    pass
            conn_views.clear()
        except Exception as e:
            logger.warning(f"operator: conn teardown raised: {e}")

        # Remove every node view from the scene.
        try:
            node_views = getattr(op, "_node_views", {}) or {}
            for v in list(node_views.values()):
                try:
                    scene.removeItem(v)
                except Exception:
                    pass
            node_views.clear()
        except Exception as e:
            logger.warning(f"operator: node teardown raised: {e}")

        # Remove the toolbar.
        try:
            tb = getattr(op, "_toolbar", None)
            if tb is not None:
                scene.removeItem(tb)
        except Exception as e:
            logger.warning(f"operator: toolbar removal raised: {e}")

        # Remove any in-flight temp connection (mid-drag preview).
        try:
            tc = getattr(op, "_temp_conn", None)
            if tc is not None:
                scene.removeItem(tc)
        except Exception as e:
            logger.warning(f"operator: temp conn removal raised: {e}")

        # The 1s-delayed initial scan (QTimer.singleShot in
        # Operator.__init__) still holds a bound-method reference to the
        # operator and will fire if we toggle off within ~1s of toggling
        # on. We can't cancel a singleShot from outside, so defang it by
        # swapping the operator's _scene for a stub that drops
        # addItem/removeItem silently — a late scan then adds nodes to
        # nowhere instead of polluting the live scene.
        try:
            class _DeadScene:
                def addItem(self, *_a, **_kw): pass
                def removeItem(self, *_a, **_kw): pass
                def installEventFilter(self, *_a, **_kw): pass
                def removeEventFilter(self, *_a, **_kw): pass
                def items(self, *_a, **_kw): return []
                def selectedItems(self, *_a, **_kw): return []
            op._scene = _DeadScene()
        except Exception as e:
            logger.warning(f"operator: scene stub install raised: {e}")

    def _view_drag_enter(self, event):
        if event.mimeData().hasFormat("application/x-rio-app"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _view_drag_move(self, event):
        if event.mimeData().hasFormat("application/x-rio-app"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _view_drop(self, event):
        mime = event.mimeData()
        if not mime.hasFormat("application/x-rio-app"):
            event.ignore()
            return

        app_path = bytes(mime.data("application/x-rio-app")).decode('utf-8')
        app_name = os.path.splitext(os.path.basename(app_path))[0]
        # event.position() is in viewport coords; QGraphicsView.mapToScene
        # accepts viewport coords for the QPoint overload, so this is correct.
        drop_pos = self.graphics_view.mapToScene(event.position().toPoint())

        # Grab the live launcher widget BEFORE we tear it down — it's the
        # one that's wired to the filesystem / parse pipeline. Constructing
        # a fresh AppLauncherWidget here and immediately deleteLater()'ing
        # it was racy: the asyncio task spawned by launch_app_at captures
        # the parse_file ref, so it survived, but any state on the launcher
        # (e.g. parent window lookups) could be invalidated mid-flight.
        launcher = None
        if self._app_launcher_proxy is not None:
            launcher = self._app_launcher_proxy.widget()

        # Close the launcher with animation after drop
        if self._app_launcher_proxy is not None:
            proxy_ref = self._app_launcher_proxy
            self._app_launcher_proxy = None

            def _on_done():
                w = proxy_ref.widget()
                proxy_ref.setWidget(None)
                self.graphics_scene.removeItem(proxy_ref)
                if w:
                    w.deleteLater()

            if launcher is not None and hasattr(launcher, 'animate_close'):
                launcher.animate_close(on_done=_on_done)
            else:
                _on_done()

        # Launch through filesystem parse file. Fall back to a fresh
        # launcher only if we somehow had no live one (shouldn't happen
        # — drops only come from the launcher's own icons).
        if launcher is None:
            launcher = AppLauncherWidget(self)
            launcher.launch_app_at(app_path, app_name, drop_pos.x(), drop_pos.y())
            launcher.deleteLater()
        else:
            launcher.launch_app_at(app_path, app_name, drop_pos.x(), drop_pos.y())

        event.acceptProposedAction()
    
    def _clear_scene(self):
        """Clear the scene and wipe the accumulated code context.

        Two pieces of state need to go:
          1. The parsed Qt items on the scene (scene_manager.clear()).
          2. The compacted code history in the CONTEXT file, which is the
             same content surfaced by `cat /n/<machine>/scene/parse` and
             `cat /n/<machine>/CONTEXT`. Without this, the LLM would still
             see code referencing widgets that no longer exist.
        """
        asyncio.create_task(self.scene_manager.clear())

        # Wipe the CONTEXT file. ParseFile reads pull from the same
        # context_file instance, so this clears /scene/parse reads too.
        try:
            fs = self.rio_server.filesystem
            if fs and getattr(fs, 'context_file', None) is not None:
                fs.context_file.clear()
        except Exception as e:
            print(f"Warning: Could not clear context file: {e}")

    def _delete_item_at(self, scene_pos):
        """Delete the top-level item (and all its children) at scene_pos.
        
        Walks from the clicked item up to the root-level scene item,
        then recursively removes it and all descendants from the scene.
        Also cleans up terminal references if the deleted item is a terminal proxy.
        """
        item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
        if item is None:
            return

        # Walk up to the top-level item (direct child of the scene)
        top_item = item
        while top_item.parentItem() is not None:
            top_item = top_item.parentItem()

        # Clean up terminal references if this is a terminal proxy
        if isinstance(top_item, QGraphicsProxyWidget) and top_item.widget() is not None:
            widget = top_item.widget()
            # Remove from self.terminals list
            if widget in self.terminals:
                self.terminals.remove(widget)
            # Unregister from filesystem
            if hasattr(widget, 'term_id') and self.rio_server.filesystem and \
               hasattr(self.rio_server.filesystem, 'terms_dir'):
                try:
                    self.rio_server.filesystem.terms_dir.unregister_terminal(widget.term_id)
                except Exception:
                    pass

        # Remove the top-level item (Qt automatically removes all children)
        self.graphics_scene.removeItem(top_item)

    def _pop_widget_at(self, scene_pos):
        """Pop a widget out of the scene into a frameless external window.

        Works for any QGraphicsProxyWidget — terminals, voice control, etc.
        Saves enough state to dock it back later.
        """
        from PySide6.QtWidgets import QGraphicsDropShadowEffect

        item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
        if item is None:
            return

        # Walk up to top-level proxy
        top_item = item
        while top_item.parentItem() is not None:
            top_item = top_item.parentItem()

        if not isinstance(top_item, QGraphicsProxyWidget):
            return
        
        # Already popped?
        if id(top_item) in self._popped_widgets:
            return

        widget = top_item.widget()
        if widget is None:
            return

        proxy = top_item
        proxy_pos = proxy.pos()
        widget_size = widget.size()

        # Compute screen position from scene position
        view_pos = self.graphics_view.mapFromScene(proxy_pos)
        screen_pos = self.graphics_view.mapToGlobal(view_pos)

        # Remove from scene
        proxy.setGraphicsEffect(None)
        proxy.setWidget(None)
        self.graphics_scene.removeItem(proxy)

        # Create frameless external window
        window = QWidget(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        window.setAttribute(Qt.WA_TranslucentBackground, True)
        window.setAttribute(Qt.WA_NoSystemBackground, True)

        shadow_pad = 50
        layout = QVBoxLayout(window)
        layout.setContentsMargins(shadow_pad, shadow_pad, shadow_pad, shadow_pad)
        layout.setSpacing(0)

        # Reparent widget into the window
        widget.setParent(window)
        layout.addWidget(widget)
        widget.show()

        # Size the window: widget size + shadow padding
        w, h = widget_size.width(), widget_size.height()
        window.resize(w + shadow_pad * 2, h + shadow_pad * 2)
        window.move(screen_pos - QPoint(shadow_pad, shadow_pad))

        # Apply shadow on the widget — only if the active theme uses shadows.
        spec = self.current_theme.shadow_for(self._dark_mode)
        if spec is not None:
            blur, ox, oy, sc = spec
            shadow = QGraphicsDropShadowEffect(widget)
            shadow.setBlurRadius(blur)
            shadow.setOffset(QPointF(ox, oy))
            shadow.setColor(sc)
            widget.setGraphicsEffect(shadow)

        # Enable dragging the frameless window
        window._drag_pos = None

        def win_press(event):
            if event.button() == Qt.LeftButton:
                window._drag_pos = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
                event.accept()
            else:
                type(window).mousePressEvent(window, event)

        def win_move(event):
            if event.buttons() & Qt.LeftButton and window._drag_pos is not None:
                window.move(event.globalPosition().toPoint() - window._drag_pos)
                event.accept()
            else:
                type(window).mouseMoveEvent(window, event)

        def win_release(event):
            if event.button() == Qt.LeftButton:
                window._drag_pos = None
                event.accept()
            else:
                type(window).mouseReleaseEvent(window, event)

        window.mousePressEvent = win_press
        window.mouseMoveEvent = win_move
        window.mouseReleaseEvent = win_release

        # Store state for docking back
        pop_id = id(proxy)
        self._popped_widgets[pop_id] = {
            'window': window,
            'widget': widget,
            'scene_pos': proxy_pos,
            'widget_size': widget_size,
        }

        window.show()
        window.raise_()

    def _dock_widget(self, pop_id):
        """Dock a single popped-out widget back into the scene."""
        from PySide6.QtWidgets import QGraphicsDropShadowEffect

        info = self._popped_widgets.pop(pop_id, None)
        if info is None:
            return

        window = info['window']
        widget = info['widget']
        scene_pos = info['scene_pos']
        widget_size = info['widget_size']

        # Remove shadow from widget
        widget.setGraphicsEffect(None)

        # Remove from external window
        widget.setParent(None)

        # Restore size
        widget.resize(widget_size)

        # Re-embed in scene
        proxy = self.graphics_scene.addWidget(widget)
        proxy.setPos(scene_pos)

        # Reapply shadow on proxy — only if the active theme uses shadows.
        spec = self.current_theme.shadow_for(self._dark_mode)
        if spec is not None:
            blur, ox, oy, sc = spec
            shadow = QGraphicsDropShadowEffect(widget)
            shadow.setBlurRadius(blur)
            shadow.setOffset(QPointF(ox, oy))
            shadow.setColor(sc)
            proxy.setGraphicsEffect(shadow)

        # If it's a terminal, update its _proxy reference
        if hasattr(widget, '_proxy'):
            widget._proxy = proxy

        # Tear down external window
        window.close()
        window.deleteLater()

        widget.show()

    def _dock_all_widgets(self):
        """Dock all popped-out widgets back into the scene."""
        for pop_id in list(self._popped_widgets.keys()):
            self._dock_widget(pop_id)
    
    def _connect_llmfs(self):
        """Connect to LLMFS.
        If a terminal exists, delegates to its /setup handler.
        Otherwise, runs the mount script directly via subprocess.
        
        When using riomux (workspace is set), this is a no-op since
        the mux handles all mounts. For standalone mode, mounts
        LLMFS and Rio individually.
        """
        if self.terminals:
            self.terminals[-1]._setup_mounts()
            return
        
        # If using riomux, mounts are handled by start_mux.py — nothing to do
        if self.rio_server.workspace:
            logger.info("Using riomux — mounts handled externally")
            return
        
        # No terminal, standalone mode — run the mount setup directly
        import subprocess
        
        llm_port = 5640
        rio_port = 5641
        llmfs_mount = self.rio_server.llmfs_mount
        rio_mount = self.rio_server.rio_mount
        mounts = [
            (rio_mount, rio_port),
            (llmfs_mount, llm_port),
        ]
        
        # Kill stale attachment scripts
        subprocess.run(['pkill', '-f', 'llmfs_attach'], capture_output=True)
        subprocess.run(['pkill', '-f', 'acme_attach'], capture_output=True)
        
        script_lines = [
            '#!/bin/bash',
            'set +e',
            f'pkexec sh -c "umount -f {llmfs_mount} 2>/dev/null || true; umount -f {rio_mount} 2>/dev/null || true"',
            'sleep 0.5',
        ]
        
        for mount_point, port in mounts:
            script_lines += [
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
                f'fi',
            ]
        
        script = '\n'.join(script_lines)
        
        try:
            result = subprocess.Popen(
                ['bash', '-c', script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logger.info("Mount script started (no terminal, running in background)")
        except Exception as e:
            logger.error(f"Failed to start mount: {e}")
    
    def _forward_wheel_to_widget(self, event, item, scene_pos):
        """Forward a wheel event directly to the embedded widget inside a
        QGraphicsProxyWidget, bypassing the proxy's own event propagation
        so that the view never receives the event back when the inner
        widget has reached its scroll limit.

        Special care for QScrollArea: childAt() returns the deepest child
        under the mouse (e.g. an icon label or a grid body widget), which
        typically doesn't handle wheel events itself.  When the target
        sits inside a QScrollArea we forward to that scroll area's
        viewport instead, so the scroll area's built-in wheel handling
        kicks in and the list scrolls naturally.
        """
        from PySide6.QtWidgets import QScrollArea

        # Walk up to the proxy widget
        proxy = item
        while proxy is not None and not isinstance(proxy, QGraphicsProxyWidget):
            proxy = proxy.parentItem()
        
        if proxy is None or proxy.widget() is None:
            return
        
        embedded = proxy.widget()
        
        # Map scene position into the embedded widget's coordinate space
        widget_pos = proxy.mapFromScene(scene_pos)
        
        # Find the actual child widget at that position (e.g. a QTextEdit viewport)
        target = embedded.childAt(int(widget_pos.x()), int(widget_pos.y()))
        if target is None:
            target = embedded

        # Walk up from target to find a QScrollArea ancestor.  If found,
        # redirect the event to its viewport — that's the widget the
        # scroll area listens to for wheel events.  Without this, wheel
        # events land on labels / grid bodies that ignore them, and the
        # scroll area never sees them.
        scroll_area = None
        walker = target
        while walker is not None and walker is not embedded:
            if isinstance(walker, QScrollArea):
                scroll_area = walker
                break
            walker = walker.parent()
        # Also check if target itself is a viewport of a QScrollArea
        if scroll_area is None and target.parent() and isinstance(target.parent(), QScrollArea):
            scroll_area = target.parent()

        if scroll_area is not None:
            target = scroll_area.viewport()

        # Map into the target widget's local coordinates
        local_pos = target.mapFrom(embedded, QPoint(int(widget_pos.x()), int(widget_pos.y())))
        
        # Build a new wheel event in the target's local coordinate space
        new_event = QWheelEvent(
            QPointF(local_pos),
            event.globalPosition(),
            event.pixelDelta(),
            event.angleDelta(),
            event.buttons(),
            event.modifiers(),
            event.phase(),
            event.inverted(),
        )
        
        QApplication.sendEvent(target, new_event)
    
    def eventFilter(self, obj, event):
        """Filter events from the view.
        
        Interaction model:
        - Ctrl+Scroll: zoom in/out centered on mouse
        - Plain scroll: blocked (no scroll-to-pan)
        - Ctrl+LeftClick on empty scene: pan view + slight zoom-out animation
        - Ctrl+LeftClick on scene object: drag/move the object
        - Ctrl+RightClick+Drag: orbit/tilt camera around scene center
        - Right-click on empty scene: context menu
        """
        if obj == self.graphics_view.viewport():
            # ---- Alt magic-pointer mode (highest priority for inspection) ----
            # When Alt is held we override mouse moves/clicks to highlight
            # the scene widget under the cursor and (on click) ship its
            # identity to the AV agent. Wheel and key events still pass
            # through to the normal handlers below.
            if self._alt_magic_active:
                et = event.type()
                if et == QEvent.Type.MouseMove:
                    self._alt_update_hover_from_viewport_pos(event.position().toPoint())
                    return True
                if et == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                    # Make sure the hover state reflects the click position
                    self._alt_update_hover_from_viewport_pos(event.position().toPoint())
                    # Multi-select: stash this widget into the pool. The
                    # actual send happens on Ctrl+S, which also exits
                    # the mode. The matching MouseButtonRelease is still
                    # swallowed by the branch below.
                    self._alt_add_hovered_to_selection()
                    return True
                if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                    return True

            # ---- Wheel events ----
            # ALWAYS consume wheel events so the view never scrolls.
            # If a scene item (proxy widget) is under the mouse we forward
            # the event directly to the embedded QWidget, which avoids the
            # problem of the proxy propagating unhandled scroll back to the
            # view when the inner widget has reached its scroll limit.
            if event.type() == QEvent.Type.Wheel:
                if event.modifiers() & Qt.ControlModifier:
                    self._handle_zoom(event)
                    return True
                
                pos = event.position().toPoint()
                scene_pos = self.graphics_view.mapToScene(pos)
                item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
                
                if item is not None:
                    self._forward_wheel_to_widget(event, item, scene_pos)
                
                # Always consume — the view itself must never scroll
                return True
            
            # ---- Terminal creation mode (highest priority) ----
            if self.new_terminal_mode or self.is_creating_terminal:
                if event.type() == QEvent.Type.MouseButtonPress:
                    return self._handle_terminal_mouse_press(event)
                elif event.type() == QEvent.Type.MouseMove:
                    return self._handle_terminal_mouse_move(event)
                elif event.type() == QEvent.Type.MouseButtonRelease:
                    return self._handle_terminal_mouse_release(event)

            # ---- Delete mode: click to delete a widget ----
            if self.delete_mode:
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                    pos = event.position().toPoint()
                    scene_pos = self.graphics_view.mapToScene(pos)
                    item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
                    if item is not None:
                        self._delete_item_at(scene_pos)
                    # Exit delete mode after click (hit or miss)
                    self._exit_special_modes()
                    return True
                # Right-click or Escape cancels delete mode
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.RightButton:
                    self._exit_special_modes()
                    return True
                if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key_Escape:
                    self._exit_special_modes()
                    return True

            # ---- Pop mode: click to pop a widget out of the scene ----
            if self.pop_mode:
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                    pos = event.position().toPoint()
                    scene_pos = self.graphics_view.mapToScene(pos)
                    item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
                    if item is not None:
                        self._pop_widget_at(scene_pos)
                    elif self._popped_widgets:
                        # Clicked empty space — dock all popped widgets back
                        self._dock_all_widgets()
                    self._exit_special_modes()
                    return True
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.RightButton:
                    self._exit_special_modes()
                    return True
                if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key_Escape:
                    self._exit_special_modes()
                    return True
            
            # ---- Ctrl+RightClick: orbit/tilt around scene center ----
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.RightButton:
                if event.modifiers() & Qt.ControlModifier:
                    self._ctrl_orbit_active = True
                    self._ctrl_orbit_used = True
                    self._ctrl_orbit_anchor = event.position().toPoint()
                    self._ctrl_orbit_pre_transform = QTransform(self.graphics_view.transform())
                    self._ctrl_orbit_pre_center = self.graphics_view.mapToScene(
                        self.graphics_view.viewport().rect().center()
                    )
                    # Stop any running animation so it doesn't fight
                    if hasattr(self, '_view_transform_animation') and \
                       self._view_transform_animation.state() == QPropertyAnimation.Running:
                        self._view_transform_animation.stop()
                    self.graphics_view.setCursor(Qt.SizeAllCursor)
                    return True
            
            if event.type() == QEvent.Type.MouseMove and self._ctrl_orbit_active:
                # Throttle to ~60 Hz. Mouse-moves arrive at 60–250 Hz on
                # modern peripherals; each one would otherwise rebuild
                # a QTransform and repaint the viewport. Storing the
                # latest position and firing _update_orbit_transform on
                # a single-shot timer collapses any burst into one
                # update per ~16 ms with no visible lag.
                self._orbit_pending_pos = event.position().toPoint()
                if not getattr(self, '_orbit_throttle_armed', False):
                    self._orbit_throttle_armed = True
                    QTimer.singleShot(16, self._fire_orbit_update)
                return True
            
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.RightButton:
                if self._ctrl_orbit_active:
                    self._ctrl_orbit_active = False
                    self._ctrl_orbit_anchor = None
                    self._ctrl_orbit_pre_transform = None
                    self.graphics_view.setCursor(Qt.ArrowCursor)
                    return True
            
            # ---- Ctrl+LeftClick: pan scene or drag objects ----
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                if event.modifiers() & Qt.ControlModifier:
                    pos = event.position().toPoint()
                    scene_pos = self.graphics_view.mapToScene(pos)
                    item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
                    
                    if item is not None:
                        # Ctrl+Click on a scene object → start dragging it
                        # Walk up to the top-level item (e.g. proxy widget)
                        top_item = item
                        while top_item.parentItem() is not None:
                            top_item = top_item.parentItem()
                        self._ctrl_dragging_item = top_item
                        self._ctrl_drag_offset = scene_pos - top_item.pos()
                        self.graphics_view.setCursor(Qt.ClosedHandCursor)
                        return True
                    else:
                        # Ctrl+Click on empty scene → start panning
                        self._ctrl_panning = True
                        self._ctrl_pan_last_pos = pos
                        self.graphics_view.setCursor(Qt.ClosedHandCursor)
                        # Animate slight zoom-out effect
                        self._animate_zoom_out()
                        return True
            
            # ---- Mouse move: pan or drag ----
            if event.type() == QEvent.Type.MouseMove:
                if self._ctrl_panning and self._ctrl_pan_last_pos is not None:
                    # Cancel any in-flight zoom-out animation so it doesn't
                    # fight with the manual pan position
                    if hasattr(self, '_view_transform_animation') and \
                       self._view_transform_animation.state() == QPropertyAnimation.Running:
                        self._view_transform_animation.stop()
                    pos = event.position().toPoint()
                    delta = pos - self._ctrl_pan_last_pos
                    self._ctrl_pan_last_pos = pos
                    # Pan by adjusting scrollbars
                    h = self.graphics_view.horizontalScrollBar()
                    v = self.graphics_view.verticalScrollBar()
                    h.setValue(h.value() - delta.x())
                    v.setValue(v.value() - delta.y())
                    return True
                
                if self._ctrl_dragging_item is not None:
                    pos = event.position().toPoint()
                    scene_pos = self.graphics_view.mapToScene(pos)
                    self._ctrl_dragging_item.setPos(scene_pos - self._ctrl_drag_offset)
                    return True
            
            # ---- Mouse release: end pan or drag ----
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._ctrl_panning:
                    self._ctrl_panning = False
                    self._ctrl_pan_last_pos = None
                    self.graphics_view.setCursor(Qt.ArrowCursor)
                    # Animate zoom back to previous scale
                    self._animate_zoom_back()
                    return True
                
                if self._ctrl_dragging_item is not None:
                    self._ctrl_dragging_item = None
                    self._ctrl_drag_offset = QPointF()
                    self.graphics_view.setCursor(Qt.ArrowCursor)
                    return True
            
            # ---- Regular mouse events for filesystem (empty scene area) ----
            #
            # Old code unconditionally ran QGraphicsScene.itemAt() on every
            # MouseMove just to early-return False if the cursor was over
            # a scene item. With dozens of items that's an O(N) hit-test
            # 60–250 times per second during any mouse activity. Worse,
            # it ran even when mouse_callback was None (no consumer for
            # the empty-scene events anyway).
            #
            # New behaviour: only test for an underlying scene item when
            # there's actually a mouse_callback to fire — and only for the
            # event we're about to dispatch. If no callback, skip the
            # test entirely and just fall through (Qt's normal event
            # routing reaches the proxy on its own).
            if self.mouse_callback is None:
                return super().eventFilter(obj, event)

            ev_type = event.type()
            if ev_type not in (
                QEvent.Type.MouseMove,
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonRelease,
            ):
                return super().eventFilter(obj, event)

            pos = event.position()
            # Only now do the (expensive-ish) hit-test, and only for the
            # one event we'd otherwise mis-deliver to the callback.
            scene_pos = self.graphics_view.mapToScene(pos.toPoint())
            if self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform()) is not None:
                return False

            if ev_type == QEvent.Type.MouseMove:
                self.mouse_callback("m", int(pos.x()), int(pos.y()))
            elif ev_type == QEvent.Type.MouseButtonPress:
                self.mouse_callback("b", int(pos.x()), int(pos.y()), event.button().value)
            else:  # MouseButtonRelease
                self.mouse_callback("r", int(pos.x()), int(pos.y()), event.button().value)
        
        return super().eventFilter(obj, event)
    
    # ---- Ctrl+Pan zoom animations ----
    
    def _animate_zoom_out(self):
        """Animate a slight zoom-out when Ctrl+press starts panning.
        Saves the current transform so we can animate back on release."""
        current = self.graphics_view.transform()
        self._zoom_back_transform = QTransform(current)
        
        # Build a slightly zoomed-out version (90% of current scale)
        zoom_factor = 0.90
        target = QTransform(
            current.m11() * zoom_factor, current.m12(), current.m13(),
            current.m21(), current.m22() * zoom_factor, current.m23(),
            current.m31(), current.m32(), current.m33()
        )
        self._animate_view_transform(target, duration=300)
    
    def _animate_zoom_back(self):
        """Animate zoom back to pre-pan transform on Ctrl+release."""
        if self._zoom_back_transform is not None:
            self._animate_view_transform(self._zoom_back_transform, duration=400)
            self._zoom_back_transform = None
    
    # ---- Terminal creation handlers ----
    
    def _handle_terminal_mouse_press(self, event):
        """Handle mouse press for terminal creation"""
        if event.button() == Qt.LeftButton and self.new_terminal_mode:
            # Start creating terminal at mouse press position
            self.start_point = event.pos()
            self.end_point = self.start_point
            
            # Map to scene coordinates
            scene_start = self.graphics_view.mapToScene(self.start_point)
            
            # Create terminal (no parent — it will live on the scene via proxy)
            self.current_terminal = TerminalWidget(
                llmfs_mount=self.rio_server.llmfs_mount,
                rio_mount=self.rio_server.rio_mount,
            )
            self.current_terminal.resize(10, 10)
            self.current_terminal.setAttribute(Qt.WA_TranslucentBackground, True)
            self.current_terminal.setAutoFillBackground(False)
            
            # Add to scene via proxy widget
            self.current_proxy = self.graphics_scene.addWidget(
                self.current_terminal, Qt.Widget
            )
            self.current_proxy.setAutoFillBackground(False)
            self.current_proxy.setPos(scene_start.x(), scene_start.y())
            
            # Disable proxy from receiving events during creation
            self.current_proxy.setAcceptedMouseButtons(Qt.NoButton)
            
            self.current_terminal.show()
            self.terminals.append(self.current_terminal)
            
            # Connect command submission
            self.current_terminal.command_submitted.connect(self._execute_command)

            self.is_creating_terminal = True
            return True
            
        return False
    
    def _handle_terminal_mouse_move(self, event):
        """Handle mouse move for terminal creation"""
        if self.is_creating_terminal:
            self.end_point = event.pos()
            
            scene_start = self.graphics_view.mapToScene(self.start_point)
            scene_end = self.graphics_view.mapToScene(self.end_point)
            
            frame_rect = QRectF(scene_start, scene_end).normalized()
            
            #if frame_rect.width() < 100:
            #    frame_rect.setWidth(100)
            #if frame_rect.height() < 150:
            #    frame_rect.setHeight(150)
            
            self.current_proxy.setPos(frame_rect.x(), frame_rect.y())
            self.current_terminal.resize(
                int(frame_rect.width()),
                int(frame_rect.height())
            )
            
            return True
            
        return False
    
    def _handle_terminal_mouse_release(self, event):
        """Handle mouse release for terminal creation"""
        if event.button() == Qt.LeftButton and self.is_creating_terminal:
            self.end_point = event.pos()
            
            scene_start = self.graphics_view.mapToScene(self.start_point)
            scene_end = self.graphics_view.mapToScene(self.end_point)
            
            frame_rect = QRectF(scene_start, scene_end).normalized()
            
            if frame_rect.width() < 100:
                frame_rect.setWidth(100)
            if frame_rect.height() < 150:
                frame_rect.setHeight(150)
            
            self.current_proxy.setPos(frame_rect.x(), frame_rect.y())
            self.current_terminal.resize(
                int(frame_rect.width()),
                int(frame_rect.height())
            )
            
            # Store proxy reference on the terminal BEFORE show_content
            self.current_terminal._proxy = self.current_proxy

            # Sync the new terminal to the active theme BEFORE show_content
            # runs.  We can't call apply_theme(snap) here because that would
            # install a static shadow which show_content would immediately
            # re-install + animate, causing a visible jump (Glass) or
            # redundant work (both).  Instead we just point the terminal
            # at the right theme name; the _theme accessor walks up to us
            # for the live value, and show_content's animate_shadow_to_position
            # consults theme.shadow to decide whether to install one.
            # Frame QSS and input style still need an explicit sync though,
            # because setup_terminal_frame already painted them with Glass
            # defaults during __init__.
            self.current_terminal._theme_name = self._active_theme_name
            self.current_terminal._is_dark_mode = self._dark_mode
            tf = self.current_terminal
            theme = self.current_theme
            f, inp = theme.frame, theme.input
            tf.terminal_frame.setStyleSheet(
                theme.frame_stylesheet(self._dark_mode, focus_alpha=tf._frame_focus_alpha)
            )
            tf._apply_frame_padding(f.inner_padding)
            ib = inp.bg_rgb_dark if self._dark_mode else inp.bg_rgb
            tf._set_input_bg_target(
                ib[0], ib[1], ib[2],
                inp.focus_alpha_dark if self._dark_mode else inp.focus_alpha,
            )

            # NOW show the content (output and input)
            self.current_terminal.show_content()
            
            # If dark mode is active, apply it to the new terminal immediately
            if self._dark_mode:
                self.current_terminal.set_dark_mode(True, duration_steps=1)
            
            # Lock the size so layouts inside don't collapse it
            self.current_terminal.setFixedSize(
                int(frame_rect.width()),
                int(frame_rect.height())
            )
            
            # Re-enable proxy mouse events
            self.current_proxy.setAcceptedMouseButtons(
                Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
            )
            self.current_proxy.setFlag(QGraphicsItem.ItemIsSelectable, True)
            
            # Make terminal stay on top via proxy z-value
            self.current_proxy.setZValue(1000)
            
            # Register terminal in the Rio filesystem (terms/ directory)
            if self.rio_server.filesystem and hasattr(self.rio_server.filesystem, 'terms_dir'):
                import weakref
                term_ref = weakref.ref(self.current_terminal)
                term_dir = self.rio_server.filesystem.terms_dir.register_terminal(
                    self.current_terminal.term_id, term_ref
                )
                self.current_terminal._term_dir = term_dir
                self.current_terminal.append_output(
                    f"Terminal ID: \n",
                    color="rgba(0, 0, 0, 255)"
                )
            
            
            # Clean up creation state
            self.is_creating_terminal = False
            self.current_terminal = None
            self.current_proxy = None
            
            self._exit_special_modes()
            return True
            
        return False
    
    def _execute_command(self, command: str):
        """Execute a command from a terminal"""
        if not self.executor:
            return
        asyncio.create_task(self._run_command(command))
    
    async def _run_command(self, command: str):
        """Async wrapper for command execution."""
        result = await self.executor.execute(command)
        
        sender = self.sender()
        if sender and sender in self.terminals:
            terminal = sender
        else:
            terminal = self.terminals[-1] if self.terminals else None
        
        if not terminal:
            return
        
        if result.success:
            if result.result is not None:
                terminal.append_output(f"{result.result}\n", color="#ce9178")
            if result.items_registered:
                terminal.append_output(
                    f"✓ Registered {len(result.items_registered)} scene item(s)\n",
                    color="#4ec9b0"
                )
        else:
            terminal.append_output(f"Error: {result.error}\n", color="#f48771")
    
    def keyPressEvent_old(self, event):
        """Handle key press"""
        try:
            if self.key_callback:
                key = event.key()
                mods = event.modifiers()
                text = event.text()
                self.key_callback(str(key), mods, text)
        except Exception as e:
            logger.exception(f"KeyPress error: {e}")
        
        super().keyPressEvent(event)

    def keyPressEvent(self, event):
        """Handle key press, including View Tilt controls"""
        modifiers = event.modifiers()
        key = event.key()

        # Handle Ctrl + Number view controls
        if modifiers & Qt.ControlModifier:
            # Ctrl+I: Toggle Immersive Mode
            if key == Qt.Key_I:
                if hasattr(self, '_immersive_mode'):
                    self._immersive_mode.toggle()
                return

            # ---- Ctrl+S: enter inspection mode, or send picks + exit ----
            # While inactive: enter multi-select inspection.
            # While active:   flush all clicked widgets to the AV agent
            #                 as a single combined message, then exit.
            #                 If nothing was picked, just exit silently.
            if key == Qt.Key_S and not event.isAutoRepeat():
                if self._alt_magic_active:
                    self._alt_flush_selection_to_av_agent()
                    self._deactivate_alt_magic_pointer()
                    self._alt_clear_overlay_immediate()
                else:
                    self._activate_alt_magic_pointer()
                return
            
            view_mapping = {
                Qt.Key_1: self.view_controller_tilt_left,
                Qt.Key_2: self.view_controller_tilt_down,
                Qt.Key_3: self.view_controller_tilt_right,
                Qt.Key_4: self.view_controller_pan_left,
                Qt.Key_5: self.view_controller_center,
                Qt.Key_6: self.view_controller_pan_right,
                Qt.Key_7: self.view_controller_corner_top_left,
                Qt.Key_8: self.view_controller_tilt_up,
                Qt.Key_9: self.view_controller_corner_top_right,
                Qt.Key_0: self.view_controller_reset,
            }
            if key in view_mapping:
                view_mapping[key]()
                return

        try:
            if self.key_callback:
                self.key_callback(str(key), modifiers, event.text())
        except Exception as e:
            logger.exception(f"KeyPress error: {e}")
        
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # Magic-pointer mode used to track Alt key-up here; it's now a
        # Ctrl+S toggle (see keyPressEvent), so this override is just a
        # passthrough.
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        # Magic-pointer mode used to be tied to the Alt key, so we had
        # to force-deactivate here to recover from Alt+Tab swallowing
        # the key-release. Now that it's a Ctrl+S toggle, the mode is
        # sticky across app switches by design — the user explicitly
        # turns it off.
        super().focusOutEvent(event)

    def focusInEvent(self, event):
        # See focusOutEvent: no Alt-tracking cleanup needed any more.
        super().focusInEvent(event)

    def _alt_force_cleanup_if_unfocused(self):
        """Retained as a no-op for any stale QTimer.singleShot callbacks
        that might still be pending from before the Ctrl+S toggle change.
        Safe to remove entirely once the process has been restarted."""
        return

    # --- View Tilt Controller Methods ---

    def _centered_tilt_transform(self, sx=1.0, sy=1.0, shx=0.0, shy=0.0):
        """Build a QTransform with scale/shear that pivots around the
        viewport centre rather than the top-left corner.

        QGraphicsView.setTransform() treats the viewport top-left as the
        matrix origin.  To make the effect visually centred we sandwich the
        operation:  T(cx,cy) · Scale · Shear · T(-cx,-cy)
        """
        vp = self.graphics_view.viewport()
        cx = vp.width() / 2.0
        cy = vp.height() / 2.0

        t = QTransform()
        t.translate(cx, cy)
        t.scale(sx, sy)
        t.shear(shx, shy)
        t.translate(-cx, -cy)
        return t

    def view_controller_center(self):
        """Ctrl+5: Center view on current mouse position"""
        global_pos = QCursor.pos()
        viewport_pos = self.graphics_view.mapFromGlobal(global_pos)
        scene_pos = self.graphics_view.mapToScene(viewport_pos)
        self._animate_view_transform(QTransform(), scene_pos)

    def _current_view_center(self):
        """Return the scene point currently at the center of the viewport."""
        return self.graphics_view.mapToScene(
            self.graphics_view.viewport().rect().center()
        )

    def view_controller_pan_left(self):
        """Ctrl+4: Pan Left — shift view left by 30% of viewport width"""
        center = self._current_view_center()
        # Compute scene-space offset: viewport pixels / current scale
        scale = self.graphics_view.transform().m11() or 1.0
        offset = self.graphics_view.viewport().width() * 0.3 / abs(scale)
        self._animate_view_transform(QTransform(), QPointF(center.x() - offset, center.y()))

    def view_controller_pan_right(self):
        """Ctrl+6: Pan Right — shift view right by 30% of viewport width"""
        center = self._current_view_center()
        scale = self.graphics_view.transform().m11() or 1.0
        offset = self.graphics_view.viewport().width() * 0.3 / abs(scale)
        self._animate_view_transform(QTransform(), QPointF(center.x() + offset, center.y()))

    def view_controller_tilt_left(self):
        """Ctrl+1: Focus-on-widget zoom (tight fit, widget fills ~55%
        of the smaller viewport dimension). See `_focus_on_widget_under_cursor`
        for the full behaviour spec — Ctrl+1 and Ctrl+2 share that path
        and differ only in how aggressively they zoom in.
        """
        self._focus_on_widget_under_cursor(padding_ratio=0.55)

    def _focus_on_widget_under_cursor(self, padding_ratio: float):
        """Shared focus-zoom path used by Ctrl+1 and Ctrl+2.

        Behaviour depends on what's under the mouse cursor when pressed:

        * **Over a widget** (a QGraphicsProxyWidget's inner content or
          a top-level scene item) — fit-and-center on that widget.
          The zoom level is computed from the widget's scene-rect size
          so it fills ~`padding_ratio` of the smaller viewport
          dimension; the view centers on the widget's middle. A
          snapshot of the pre-focus view is taken on the *first* focus
          from an unfocused state so we can return there later.

        * **Over empty scene** — revert to the snapshotted view from
          before the first focus. If no snapshot exists (we were never
          focused) the press is a no-op.

        * **Already focused, cursor over another widget** — fly
          directly to the new widget without going home first. The
          original snapshot is preserved so a later "press over empty
          scene" still returns to the true starting point.

        Ctrl+1 and Ctrl+2 share `_temp_zoom_*` snapshot state, so
        pressing Ctrl+2 to focus and then Ctrl+1 (or vice versa) on
        empty scene still reverts to the original pre-focus view.
        """
        # Step 1: figure out what (if anything) is under the cursor.
        widget_rect = self._scene_rect_of_item_under_cursor()

        # Step 2: branch on whether we found something.
        if widget_rect is None:
            # Empty scene — revert if we have a snapshot.
            if not getattr(self, '_temp_zoom_active', False):
                return  # nothing to revert to
            target_transform = self._temp_zoom_pre_transform
            target_center = self._temp_zoom_pre_center
            self._temp_zoom_active = False
            self._temp_zoom_pre_transform = None
            self._temp_zoom_pre_center = None
            if target_transform is not None and target_center is not None:
                duration = self._compute_focus_animation_duration(
                    target_transform, target_center
                )
                self._animate_view_transform(target_transform, target_center, duration=duration)
            return

        # Step 3: a widget is under the cursor. Snapshot if this is
        # the first focus from an unfocused state.
        if not getattr(self, '_temp_zoom_active', False):
            self._temp_zoom_pre_transform = QTransform(self.graphics_view.transform())
            self._temp_zoom_pre_center = self._current_view_center()
            self._temp_zoom_active = True
        # Otherwise: keep the existing snapshot — pressing on widget B
        # while focused on widget A should still revert to the original
        # pre-A state when the user later presses on empty scene.

        # Step 4: compute the fit-to-widget transform and animate.
        target_transform, target_center = self._build_focus_transform(
            widget_rect, padding_ratio=padding_ratio
        )
        if target_transform is None:
            return
        duration = self._compute_focus_animation_duration(target_transform, target_center)
        self._animate_view_transform(target_transform, target_center, duration=duration)

    def _compute_focus_animation_duration(self, target_transform, target_center):
        """Pick an animation duration that scales with how zoomed-in
        the view will end up (or be coming from, for reverts).

        Smaller widgets get a larger target scale under fit-to-widget,
        and a higher zoom level needs more time to feel comfortable —
        the eye has to absorb a more detailed, more magnified target.
        A pan-only motion at the same scale doesn't need much time;
        zooming into a tiny dial does, even if the pan distance is
        small.

        We take the LARGER of current_scale and target_scale (so revert
        animations from a tight zoom are also slow) and map it
        logarithmically to a duration:

            scale 1×  → MIN_MS
            scale 2×  → MIN_MS + 1 octave-worth
            scale 4×  → MIN_MS + 2 octaves-worth
            ...
        """
        import math
        MIN_MS = 1200          # baseline duration for ~1× zoom (no magnification)
        MS_PER_OCTAVE = 700    # each zoom-doubling adds this much time
        MAX_MS = 3000          # hard ceiling

        try:
            current_scale = abs(self.graphics_view.transform().m11())
            target_scale = abs(target_transform.m11())
            # The "perceptually loaded" scale is whichever side is more
            # zoomed-in. For focus-in this is target_scale; for revert
            # from a tight focus this is current_scale.
            effective_scale = max(current_scale, target_scale)
            if effective_scale <= 0:
                return MIN_MS

            # Octaves above 1×. A scale below 1× contributes 0 (zoom-outs
            # don't feel rushed because the target is bigger / easier to
            # track).
            octaves_above_1x = max(0.0, math.log2(effective_scale))
            duration = MIN_MS + octaves_above_1x * MS_PER_OCTAVE
            return int(min(MAX_MS, duration))
        except Exception:
            return MIN_MS

    def _scene_rect_of_item_under_cursor(self):
        """Return the scene-space QRectF of the widget/item currently
        under the mouse cursor, or None if the cursor is over empty
        scene (or outside the viewport).

        Uses the same drill-down logic as the Alt magic-pointer so the
        focus target matches what the user just saw highlighted: a
        button inside a frame returns the button's rect, sliding off
        the button onto the frame's body returns the frame's rect.
        """
        viewport = self.graphics_view.viewport()
        global_pos = QCursor.pos()
        vp_pos = viewport.mapFromGlobal(global_pos)
        if not viewport.rect().contains(vp_pos):
            return None

        scene_pos = self.graphics_view.mapToScene(vp_pos)
        item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
        if item is None:
            return None

        # Defensive: ignore any magic-pointer overlay items so Ctrl+1
        # pressed while Alt is also held doesn't think the highlight
        # mask is the target.
        if item is getattr(self, '_alt_overlay_item', None) or \
           item is getattr(self, '_alt_preview_text', None) or \
           item is getattr(self, '_alt_fading_out_item', None):
            return None

        # Find the owning proxy widget (if any) and drill into it.
        proxy = item
        while proxy is not None and not isinstance(proxy, QGraphicsProxyWidget):
            proxy = proxy.parentItem()

        if proxy is not None and proxy.widget() is not None:
            embedded = proxy.widget()
            widget_pos = proxy.mapFromScene(scene_pos)
            target = embedded.childAt(int(widget_pos.x()), int(widget_pos.y()))
            if target is None:
                target = embedded
            # Same composite-collapsing rules as the magic pointer:
            # Qt internals like qt_calendar_navigationbar get promoted
            # back to the public composite (QCalendarWidget).
            try:
                target = self._alt_promote_to_meaningful_widget(target, embedded)
            except Exception:
                pass
            try:
                local_rect = proxy.subWidgetRect(target)
                return proxy.mapRectToScene(local_rect)
            except Exception:
                # Fallback to the proxy's whole scene-bounding-rect.
                return proxy.sceneBoundingRect()

        # Non-proxy scene item: walk to the top-level to get the whole
        # logical object's rect.
        top = item
        while top.parentItem() is not None:
            top = top.parentItem()
        return top.sceneBoundingRect()

    def _build_focus_transform(self, widget_rect, padding_ratio: float = 0.55):
        """Given a widget's scene-space rect, return (target_transform,
        target_center) that fits the widget into the viewport with
        modest padding. Returns (None, None) if the inputs don't make
        sense (zero-area rect, invalid viewport, etc.).

        `padding_ratio` controls how much of the smaller viewport
        dimension the widget should occupy at the end. 0.55 is the
        default (Ctrl+1's tight focus); Ctrl+2 passes a lower value for
        an attenuated zoom that leaves more surrounding context visible.

        The scale is computed by composing onto the *current* transform
        so any active tilt/shear is preserved — only the scale changes.
        """
        if widget_rect is None or widget_rect.isEmpty():
            return None, None

        viewport_size = self.graphics_view.viewport().size()
        vw = viewport_size.width()
        vh = viewport_size.height()
        ww = widget_rect.width()
        wh = widget_rect.height()
        if vw <= 0 or vh <= 0 or ww <= 0 or wh <= 0:
            return None, None

        # Lower ratio = more breathing room around the focused widget,
        # so the zoom feels less aggressive even for tiny targets.
        target_scale = padding_ratio * min(vw / ww, vh / wh)
        # Clamp to a sane range. The upper cap is intentionally lower
        # than _handle_zoom's 10.0 ceiling: focusing a 20px-wide button
        # at 10x is disorienting. 4.0x keeps tiny widgets readable
        # without flying past comfortable reading distance.
        target_scale = max(0.1, min(4.0, target_scale))

        # Compose the new scale onto the existing transform so any
        # rotation/shear from view_controller_tilt_* survives the focus.
        current = self.graphics_view.transform()
        current_scale = abs(current.m11())
        if current_scale <= 1e-6:
            return None, None
        zoom_factor = target_scale / current_scale
        target_transform = QTransform(current).scale(zoom_factor, zoom_factor)

        # Center on the widget's middle.
        target_center = QPointF(widget_rect.center())
        return target_transform, target_center

    def view_controller_tilt_right(self):
        """Ctrl+3: Tilt Right & Zoom Out"""
        target = self._centered_tilt_transform(sx=0.7, sy=0.7, shx=0.2)
        self._animate_view_transform(target, self._current_view_center())

    def view_controller_tilt_down(self):
        """Ctrl+2: Attenuated focus-on-widget zoom.

        Same focus/revert behaviour as Ctrl+1, but with a gentler zoom
        — the focused widget fills only ~35% of the smaller viewport
        dimension (vs Ctrl+1's 55%), leaving more surrounding context
        visible. Useful when you want to lean in on a widget without
        committing to the tight Ctrl+1 framing. Shares the same
        snapshot state, so Ctrl+1 and Ctrl+2 can be mixed freely.
        """
        self._focus_on_widget_under_cursor(padding_ratio=0.35)

    def view_controller_tilt_up(self):
        """Ctrl+8: Tilt Up"""
        target = self._centered_tilt_transform(sx=0.7, sy=0.56, shy=-0.15)
        self._animate_view_transform(target, self._current_view_center())

    def view_controller_corner_top_left(self):
        """Ctrl+7: Top Left perspective"""
        target = self._centered_tilt_transform(sx=0.95, sy=0.76, shy=-0.09)
        self._animate_view_transform(target, self._current_view_center())

    def view_controller_corner_top_right(self):
        """Ctrl+9: Top Right perspective"""
        target = self._centered_tilt_transform(sx=0.5, sy=0.4, shy=-0.15)
        self._animate_view_transform(target, self._current_view_center())

    # --- Ctrl+RightMouse Orbit ---

    def _fire_orbit_update(self):
        """Drain the latest orbit position onto the actual transform.

        Called by the singleShot installed in eventFilter when an orbit
        mouse-move arrives. Reads whatever position is currently
        pending — any moves that happened during the 16 ms wait are
        coalesced into this single update, so the orbit feels smooth
        regardless of mouse-input rate.
        """
        self._orbit_throttle_armed = False
        if not self._ctrl_orbit_active:
            return
        pos = getattr(self, '_orbit_pending_pos', None)
        if pos is not None:
            self._update_orbit_transform(pos)

    def _update_orbit_transform(self, current_pos):
        """Compute a live tilt/perspective transform based on how far the
        mouse has moved from the Ctrl+RightClick anchor point.

        The effect is like a virtual trackball / camera orbit in video
        editing software:
          - Horizontal displacement → horizontal shear (pan-tilt left/right)
          - Vertical displacement   → vertical shear   (tilt up/down)
          - Distance from anchor    → convex zoom-out  (further = more zoom out)

        At zero displacement (mouse hasn't moved) the transform is exactly
        the pre-orbit transform, so there is no jump on the first frame.
        The view stays centred on whatever scene point was visible before.
        """
        if self._ctrl_orbit_anchor is None or self._ctrl_orbit_pre_transform is None:
            return

        vp = self.graphics_view.viewport()
        half_w = vp.width() / 2.0
        half_h = vp.height() / 2.0

        # Normalised displacement from anchor  (-1..+1 range approx)
        dx = (current_pos.x() - self._ctrl_orbit_anchor.x()) / half_w
        dy = (current_pos.y() - self._ctrl_orbit_anchor.y()) / half_h

        # Distance from anchor (0..~1.4 for corner)
        dist = (dx * dx + dy * dy) ** 0.5

        # Convex zoom-out: further = more zoom out, diminishing returns
        # At dist=0 this is 1.0 (no change), so first frame is identity
        zoom = 1.0 / (1.0 + 0.6 * dist * dist)
        zoom = max(zoom, 0.25)

        # Shear amounts — proportional to displacement, clamped
        max_shear = 0.35
        shx = max(-max_shear, min(max_shear, dx * 0.25))
        shy = max(-max_shear, min(max_shear, dy * 0.20))

        # Slight vertical scale compression when looking from above/below
        sy_compression = 1.0 - abs(dy) * 0.25
        sy_compression = max(sy_compression, 0.55)

        # Extract the base scale from the pre-orbit transform so the orbit
        # is relative to whatever zoom level was active before.
        pre = self._ctrl_orbit_pre_transform
        base_sx = (pre.m11()**2 + pre.m21()**2) ** 0.5
        base_sy = (pre.m12()**2 + pre.m22()**2) ** 0.5
        if base_sx < 0.001:
            base_sx = 1.0
        if base_sy < 0.001:
            base_sy = 1.0

        # Build the final transform: base scale * orbit adjustments,
        # centred on the viewport middle so the effect is symmetric.
        cx = half_w
        cy = half_h

        t = QTransform()
        t.translate(cx, cy)
        t.scale(base_sx * zoom, base_sy * zoom * sy_compression)
        t.shear(shx, shy)
        t.translate(-cx, -cy)

        self.graphics_view.setTransform(t)
        self.graphics_view.centerOn(self._ctrl_orbit_pre_center)

    def _handle_zoom(self, event):
        """Ctrl+Scroll zoom centered on mouse position."""
        view_pos = event.position().toPoint()
        scene_pos = self.graphics_view.mapToScene(view_pos)

        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.15
        elif delta < 0:
            factor = 1.0 / 1.15
        else:
            return

        current_scale = self.graphics_view.transform().m11()
        new_scale = current_scale * factor
        if new_scale < 0.05 or new_scale > 10.0:
            return

        self.graphics_view.scale(factor, factor)

        new_scene_pos = self.graphics_view.mapToScene(view_pos)
        delta_scene = new_scene_pos - scene_pos
        h = self.graphics_view.horizontalScrollBar()
        v = self.graphics_view.verticalScrollBar()
        h.setValue(int(h.value() - delta_scene.x()))
        v.setValue(int(v.value() - delta_scene.y()))

    def view_controller_reset(self):
        """Ctrl+0: Reset View — center on origin (0,0)"""
        self._animate_view_transform(QTransform(), QPointF(0, 0))

    # --- Animation Core ---

    def _animate_view_transform(self, target_transform, pan_offset=None, duration=1200):
        """Animate the view transform. If pan_offset is None, keep current center."""
        if pan_offset is None:
            pan_offset = self.graphics_view.mapToScene(
                self.graphics_view.viewport().rect().center()
            )
        
        if hasattr(self, '_view_transform_animation') and self._view_transform_animation.state() == QPropertyAnimation.Running:
            self._view_transform_animation.stop()
        
        self._start_transform = self.graphics_view.transform()
        self._target_transform = target_transform
        self._start_center = self.graphics_view.mapToScene(self.graphics_view.viewport().rect().center())
        self._target_center = pan_offset

        if not hasattr(self, '_animation_helper'):
            self._animation_helper = QObject()
        
        self._view_transform_animation = QPropertyAnimation(self._animation_helper, b"progress")
        self._view_transform_animation.setDuration(duration)
        self._view_transform_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._view_transform_animation.setStartValue(0.0)
        self._view_transform_animation.setEndValue(1.0)
        self._view_transform_animation.valueChanged.connect(self._update_transform_and_pan_progress)
        self._view_transform_animation.start()

    def _update_transform_and_pan_progress(self, progress):
        s, t = self._start_transform, self._target_transform
        
        m11 = s.m11() + (t.m11() - s.m11()) * progress
        m12 = s.m12() + (t.m12() - s.m12()) * progress
        m13 = s.m13() + (t.m13() - s.m13()) * progress
        m21 = s.m21() + (t.m21() - s.m21()) * progress
        m22 = s.m22() + (t.m22() - s.m22()) * progress
        m23 = s.m23() + (t.m23() - s.m23()) * progress
        m31 = s.m31() + (t.m31() - s.m31()) * progress
        m32 = s.m32() + (t.m32() - s.m32()) * progress
        m33 = s.m33() + (t.m33() - s.m33()) * progress
        
        interpolated_transform = QTransform(m11, m12, m13, m21, m22, m23, m31, m32, m33)
        interpolated_center = QPointF(
            self._start_center.x() + (self._target_center.x() - self._start_center.x()) * progress,
            self._start_center.y() + (self._target_center.y() - self._start_center.y()) * progress
        )
        
        self.graphics_view.setTransform(interpolated_transform)
        self.graphics_view.centerOn(interpolated_center)

    def showEvent(self, event):
        """Center view on origin (0,0) once the viewport has a valid size."""
        super().showEvent(event)
        if not hasattr(self, '_initial_center_done'):
            self._initial_center_done = True
            QTimer.singleShot(50, self._center_on_origin)
        if hasattr(self, 'debug_overlay'):
            self.debug_overlay.reposition(self.width())

    def resizeEvent(self, event):
        """Reposition the debug overlay on window resize."""
        super().resizeEvent(event)
        if hasattr(self, 'debug_overlay'):
            self.debug_overlay.reposition(self.width())

    def _center_on_origin(self):
        """Set scrollbars so that scene point (0,0) is at the viewport center."""
        h = self.graphics_view.horizontalScrollBar()
        v = self.graphics_view.verticalScrollBar()
        h.setValue((h.minimum() + h.maximum()) // 2)
        v.setValue((v.minimum() + v.maximum()) // 2)

    def _exit_app(self):
        """Exit the application cleanly — stop the server and close the window."""
        self.close()

    def _toggle_fullscreen(self):
        """Toggle between fullscreen and normal window mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ------------------------------------------------------------------
    # Paper colour palette
    # ------------------------------------------------------------------

    @staticmethod
    def _random_paper_rgb(hue_hint: float = None) -> tuple:
        """Generate one paper-like pastel RGB tuple.

        High lightness + low saturation → soft coloured-paper look.
        ``hue_hint`` pins the hue (0-1); None picks randomly.
        Returns (r, g, b) in 0-255.
        """
        h = hue_hint if hue_hint is not None else random.random()
        s = random.uniform(0.15, 0.35)
        l = random.uniform(0.80, 0.92)
        r_f, g_f, b_f = colorsys.hls_to_rgb(h, l, s)
        return int(r_f * 255), int(g_f * 255), int(b_f * 255)

    def _generate_paper_palette(self, n_terminals: int) -> tuple:
        """Pick distinct paper colours for the scene and each terminal.

        Returns ``(scene_rgb, [term_rgb, ...])``.  Hues are spread
        evenly around the colour wheel with a random offset so no two
        adjacent terminals share a hue, and the scene hue is offset
        by half a step from the nearest terminal.
        """
        total = 1 + max(n_terminals, 1)  # scene + terminals
        base_hue = random.random()
        hues = [(base_hue + i / total) % 1.0 for i in range(total)]
        random.shuffle(hues)

        scene_rgb = self._random_paper_rgb(hues[0])
        term_rgbs = [self._random_paper_rgb(hues[i + 1])
                     for i in range(n_terminals)]
        return scene_rgb, term_rgbs

    def _apply_random_paper_color(self):
        """Set the scene background to a random paper-like pastel.

        High lightness + low saturation gives the soft, washed-out look
        of coloured paper or pastel art stock. The exact ranges (s∈
        [0.15, 0.35], l∈[0.80, 0.95]) are deliberately narrow — too low
        on l and the colour reads as muddy; too high on s and it stops
        looking paper-y. Hue is unconstrained so the user gets variety.

        We update both the live QGraphicsScene brush AND the
        SceneManager's persisted background_color, so the chosen
        colour survives a theme switch (set_theme reads
        scene_manager.background_color when deciding the new bg).
        """
        r, g, b = self._random_paper_rgb()
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        # Live update of the visible scene
        self.graphics_scene.setBackgroundBrush(QBrush(QColor(r, g, b)))

        # Persist on the SceneManager so set_theme() preserves it
        # (see _setup_scene: sm_bg overrides theme_bg when present and
        # non-default).
        try:
            self.scene_manager.background_color = hex_color
        except Exception:
            pass

    def closeEvent(self, event):
        """Handle window close — perform full shutdown of server and Qt.

        Two important things this version does that the old one didn't:

        1. Sets self.rio_server._running = False SYNCHRONOUSLY, before
           returning. The Qt loop checks this every iteration, so the
           moment we set it the loop will exit on its next tick. The
           previous version relied on an async stop() task running first
           — but that task might not be scheduled before Qt actually
           closes the window, leaving the asyncio loop stuck on
           processEvents() of a dead app.

        2. Uses run_coroutine_threadsafe with a fallback. If we're
           called from the Qt thread (which we are) and the asyncio
           loop is running there too, ensure_future works. But if the
           loop has already started shutting down, ensure_future raises;
           we then just return — _running is already False, so the loop
           exits cleanly without the server.stop() call. The signal-
           handler path in main() will catch that and run stop() itself.
        """
        print("\nRio shutting down...")
        try:
            self.scene_manager.detach_qt()
        except Exception as e:
            logger.warning(f"detach_qt failed: {e}")

        # Synchronous flip — guarantees the polling Qt loop exits.
        self.rio_server._running = False

        # Best-effort async cleanup. If the loop is healthy this runs;
        # if not, the main() shutdown path picks up the slack.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.rio_server.stop())
        except RuntimeError:
            pass

        super().closeEvent(event)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Rio Display Server - Graphics scene as a filesystem"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5641,
        help="TCP port (default: 5641)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="TCP host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--unix", "-u",
        metavar="PATH",
        help="Unix socket path (instead of TCP)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3840,
        help="Scene width (default: 3840)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2160,
        help="Scene height (default: 2160)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--workspace", "-w",
        metavar="NAME",
        default=None,
        help="Workspace name for riomux (e.g. 'ekanza'). "
             "When set, paths become /n/mux/llm and /n/mux/<workspace>. "
             "When unset, uses legacy /n/llm and /n/rioa."
    )
    parser.add_argument(
        "--mux-mount",
        default="/n/mux",
        help="Mux mount point (default: /n/mux). Only used with --workspace."
    )
    parser.add_argument(
        "--fullscreen", "-f",
        action="store_true",
        help="Start in fullscreen mode"
    )
    parser.add_argument(
        "--auth-token",
        action="append",
        metavar="TOKEN",
        help="Auth token (repeatable). Enables 9P token auth. "
             "Also reads RIO_AUTH_TOKENS (comma-separated)."
    )
    parser.add_argument(
        "--auth-file",
        metavar="PATH",
        help="Path to a file with auth tokens (one per line, # comments). "
             "Also reads RIO_AUTH_FILE."
    )
    parser.add_argument(
        "--auth-timeout",
        type=float,
        default=60.0,
        help="Auth fid timeout in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # Build auth manager. CLI args take precedence; env vars
    # (RIO_AUTH_TOKENS, RIO_AUTH_FILE, RIO_AUTH_TIMEOUT) are also
    # consulted by AuthManager itself. If neither CLI nor env supplies
    # any token, the manager has zero secrets and Server9P treats auth
    # as disabled — fully backward compatible.
    from ninep.auth import AuthManager
    auth_file = args.auth_file or os.environ.get(AuthManager.ENV_FILE)
    auth_manager = AuthManager(
        secrets=args.auth_token,
        secrets_file=auth_file,
        auth_timeout=args.auth_timeout,
    )
    if auth_manager.enabled:
        logger.info(f"Rio 9P auth: ENABLED ({auth_manager.secret_count} token(s))")
    else:
        logger.info("Rio 9P auth: disabled (no tokens configured)")
    
    # Create server
    server = RioServer(
        headless=args.headless,
        width=args.width,
        height=args.height,
        workspace=args.workspace,
        mux_mount=args.mux_mount,
        fullscreen=args.fullscreen,
        auth_manager=auth_manager,
    )

    # Signal handling.
    #
    # Same fix as llmfs/main.py and riomux/__main__.py: the previous
    # `asyncio.create_task(server.stop())` from a signal handler races
    # the main coroutine — by the time the task runs, the main coroutine
    # may already have raised CancelledError, leaving server.stop()
    # half-done. Worse here, because rio drives a Qt event loop on the
    # asyncio thread; if stop() doesn't fully run, _running stays True
    # and the Qt loop never exits.
    #
    # Solution: race the start_* task against a stop_event set by signals,
    # then call stop() explicitly on a still-live loop, and await the
    # serve task to completion.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    # Start server
    serve_task = None
    try:
        if args.unix:
            serve_task = asyncio.create_task(server.start_unix(args.unix))
        else:
            serve_task = asyncio.create_task(server.start_tcp(args.host, args.port))

        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait(
            {serve_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Either signal arrived or serve died on its own.
        print("\nShutting down rio...")
        await server.stop()  # idempotent; flips _running, closes 9P, quits Qt

        # Now await the serve task. start_tcp() exits cleanly once the
        # Qt loop sees _running == False and processEvents() returns
        # with the app quit.
        if not serve_task.done():
            try:
                await asyncio.wait_for(serve_task, timeout=2.0)
            except asyncio.TimeoutError:
                serve_task.cancel()
                try:
                    await serve_task
                except (asyncio.CancelledError, Exception):
                    pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Serve task exited with error: {e}")

        if not stop_task.done():
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logging.exception(f"Server error: {e}")
        # Best-effort cleanup before exit
        try:
            await server.stop()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    # Prefer qasync: it installs a Qt-driven asyncio event loop, which
    # removes the processEvents polling loop in _run_qt_loop entirely
    # and eliminates the ~125 Hz wakeup floor on idle. With qasync,
    # asyncio events and Qt events share a single dispatch path —
    # there's no "asyncio thread vs Qt thread" decision because Qt's
    # loop IS the asyncio loop. Input latency improves correspondingly.
    #
    # The fallback (plain asyncio.run) is the historical behaviour and
    # remains correct; _run_qt_loop auto-detects which mode it's in.
    try:
        import qasync  # type: ignore
        from PySide6.QtWidgets import QApplication

        # qasync needs a QApplication BEFORE the loop is constructed.
        # main() also creates one inside _start_qt(); QApplication is a
        # singleton, so re-using the instance is fine.
        _qapp = QApplication.instance() or QApplication(sys.argv)
        _loop = qasync.QEventLoop(_qapp)
        asyncio.set_event_loop(_loop)
        with _loop:
            _loop.run_until_complete(main())
    except ImportError:
        # qasync not installed — fall back to polling loop.
        asyncio.run(main())
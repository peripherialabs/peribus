#!/usr/bin/env python3
"""
Rio Display Server

This version provides the core Rio display server functionality
with context-menu based LLMFS connectivity.
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ninep.server import Server9P
from rio.filesystem import RioRoot
from rio.scene import SceneManager
from rio.terminal_widget import TerminalWidget
from rio.parser import Executor, ExecutionContext
from rio.ai_voice_control import AIVoiceControlWidget
from rio.immersive_mode import install_immersive_mode

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
    ):
        self.headless = headless
        
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
    
    def _initialize_filesystem(self):
        """Initialize filesystem with Qt objects"""
        qt_objects = {}
        
        if self.window:
            qt_objects['main_window'] = self.window
            qt_objects['graphics_scene'] = self.window.graphics_scene
            qt_objects['graphics_view'] = self.window.graphics_view
        
        # Create filesystem with Qt objects
        self.filesystem = RioRoot(self.scene_manager, qt_objects)
        
        # Create 9P server
        self.server = Server9P(self.filesystem)
    
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
        
        # Initialize filesystem after Qt is ready
        self._initialize_filesystem()
        
        server_task = asyncio.create_task(
            self.server.serve_unix(path)
        )
        
        if not self.headless:
            await self._run_qt_loop()
        else:
            await server_task
    
    async def stop(self):
        """Stop the server"""
        self._running = False
        if self.server:
            await self.server.stop()
        
        if self.app:
            self.app.quit()
    
    async def _start_qt(self):
        """Initialize Qt"""
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            logger.warning("PySide6 not available, running headless")
            self.headless = True
            return
        
        self.app = QApplication(sys.argv)
        self.window = RioWindow(self.scene_manager, self)
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
        """Run Qt event loop alongside asyncio"""
        # Connect events after filesystem is ready
        self._connect_events()
        
        while self._running:
            if self.app:
                self.app.processEvents()
            await asyncio.sleep(0.001)


# ============================================================================
# Qt Window
# ============================================================================

from PySide6.QtWidgets import (
    QMainWindow, QGraphicsView, QGraphicsScene,
    QWidget, QVBoxLayout, QMenu, QGraphicsProxyWidget,
    QGraphicsItem, QApplication, QLabel, QTextEdit
)
from PySide6.QtCore import (
    Qt, QRectF, QPoint, QPointF, QObject, QTimer, QEvent,
    QPropertyAnimation, QEasingCurve
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
        self._messages: list = []

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
        """Add a tagged debug message."""
        import html as _html

        if len(content) > 2000:
            content = content[:2000] + "…"

        safe_tag = _html.escape(tag)
        safe_content = _html.escape(content).replace('\n', '<br>')

        entry = (
            f'<span style="color:#e06c75; font-weight:600;">[{safe_tag}]</span> '
            f'<span style="color:#c8ccd4;">{safe_content}</span>'
        )
        self._messages.append(entry)

        if len(self._messages) > self.MAX_MESSAGES:
            self._messages = self._messages[-self.MAX_MESSAGES:]

        self._text.setHtml(
            '<div style="line-height:1.45;">'
            + "<br>".join(self._messages)
            + '</div>'
        )

        # Auto-scroll to bottom
        QTimer.singleShot(20, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_messages(self):
        self._messages.clear()
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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and (event.pos() - self._drag_start).manhattanLength() > 8:
            from PySide6.QtGui import QDrag
            from PySide6.QtCore import QMimeData
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.app_path)
            mime.setData("application/x-rio-app", self.app_path.encode('utf-8'))
            drag.setMimeData(mime)
            drag.exec(Qt.CopyAction)
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            start = self._drag_start
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)
            if start is not None and (event.pos() - start).manhattanLength() < 8:
                self.launcher.launch_app(self.app_path, self.app_name)
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
        from PySide6.QtWidgets import QGridLayout

        apps = self._discover_apps()

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

        if not apps:
            empty = QLabel("No apps found\nin ./apps/")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: rgba(0,0,0,100); font-size: 11px;")
            grid.addWidget(empty, 0, 0)
        else:
            cols = max(1, min(4, len(apps)))
            for idx, (name, path) in enumerate(apps):
                icon = _AppIconLabel(name, path, self)
                grid.addWidget(icon, idx // cols, idx % cols)

        frame_layout.addWidget(body)
        outer.addWidget(self._content_frame)

        n = len(apps) if apps else 1
        cols = max(1, min(4, n))
        rows = (n + cols - 1) // cols
        w = cols * 88 + 16
        h = 28 + rows * 80 + 16
        self.setFixedSize(max(w, 160), min(h, 420))

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

        # Build the full rounded-rect perimeter as a QPainterPath
        from PySide6.QtGui import QPainterPath
        full = QPainterPath()
        rect = QRectF(inset, inset, w - 2 * inset, h - 2 * inset)
        full.addRoundedRect(rect, r, r)

        # Compute total perimeter length
        total_len = full.length()
        draw_len = total_len * min(self._border_progress, 1.0)

        if draw_len >= total_len - 0.5:
            # Full border
            p.drawPath(full)
        else:
            # Partial: walk the path and extract a trimmed sub-path
            partial = QPainterPath()
            # Sample the path in small increments
            step = max(0.5, draw_len / 400.0)
            dist = 0.0
            started = False
            while dist <= draw_len:
                pt = full.pointAtPercent(min(dist / total_len, 1.0))
                if not started:
                    partial.moveTo(pt)
                    started = True
                else:
                    partial.lineTo(pt)
                dist += step
            # Final point
            pt = full.pointAtPercent(min(draw_len / total_len, 1.0))
            partial.lineTo(pt)
            p.drawPath(partial)

        p.end()

    # ── Open animation ───────────────────────────────────────────────

    def animate_open(self, proxy):
        """Start the open animation sequence."""
        from PySide6.QtWidgets import QGraphicsDropShadowEffect

        self._proxy = proxy
        self._border_progress = 0.0
        self._content_opacity = 0.0
        self._content_frame.setVisible(False)

        # Pre-create shadow at (0,0)
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
        # Shadow + perspective tilt on proxy — managed by the widget
        self.voice_control.attach_proxy_shadow(self.voice_control_proxy)
    
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
        """Run a single QTimer that batch-updates the scene background and
        every QGraphicsDropShadowEffect each tick.

        Previous implementation created one QTimer *per* shadow effect which
        caused severe lag when many widgets were on screen.  This version
        collects all targets up-front and drives them from one timer.
        """
        from PySide6.QtWidgets import QGraphicsDropShadowEffect as _DSE

        # Kill any in-flight dark-mode animation
        if self._dark_mode_bg_timer is not None:
            self._dark_mode_bg_timer.stop()
            self._dark_mode_bg_timer.deleteLater()
            self._dark_mode_bg_timer = None

        # --- Snapshot current state ---

        # Background
        brush = self.graphics_scene.backgroundBrush()
        start_bg = brush.color()
        bg_sr, bg_sg, bg_sb = start_bg.red(), start_bg.green(), start_bg.blue()
        if to_dark:
            bg_tr, bg_tg, bg_tb = 18, 18, 25
        else:
            bg_tr, bg_tg, bg_tb = 250, 250, 250

        # Shadows — collect (effect, start_color) tuples once
        target_shadow_color = QColor(255, 255, 255, 160) if to_dark else QColor(0, 0, 0, 120)
        shadow_targets = []
        for item in self.graphics_scene.items():
            effect = item.graphicsEffect()
            if isinstance(effect, _DSE):
                shadow_targets.append((effect, QColor(effect.color())))

        # Pre-extract target RGBA once
        ts_r, ts_g, ts_b, ts_a = (
            target_shadow_color.red(),
            target_shadow_color.green(),
            target_shadow_color.blue(),
            target_shadow_color.alpha(),
        )

        step = [0]

        # Suppress per-item repaints while we batch-update; the single
        # setBackgroundBrush at the end of each tick triggers one
        # scene-wide repaint anyway (FullViewportUpdate mode).
        def tick():
            if step[0] <= steps:
                t = step[0] / steps
                t = t * t * (3.0 - 2.0 * t)  # smoothstep ease-in-out

                # -- Background --
                r = int(bg_sr + (bg_tr - bg_sr) * t)
                g = int(bg_sg + (bg_tg - bg_sg) * t)
                b = int(bg_sb + (bg_tb - bg_sb) * t)
                self.graphics_scene.setBackgroundBrush(QBrush(QColor(r, g, b)))

                # -- All shadows in one pass --
                for effect, sc in shadow_targets:
                    sr = int(sc.red()   + (ts_r - sc.red())   * t)
                    sg = int(sc.green() + (ts_g - sc.green()) * t)
                    sb = int(sc.blue()  + (ts_b - sc.blue())  * t)
                    sa = int(sc.alpha() + (ts_a - sc.alpha()) * t)
                    effect.setColor(QColor(sr, sg, sb, sa))

                step[0] += 1
            else:
                # Final values
                self.graphics_scene.setBackgroundBrush(
                    QBrush(QColor(bg_tr, bg_tg, bg_tb))
                )
                for effect, _ in shadow_targets:
                    effect.setColor(target_shadow_color)

                self._dark_mode_bg_timer.stop()
                self._dark_mode_bg_timer.deleteLater()
                self._dark_mode_bg_timer = None

        self._dark_mode_bg_timer = QTimer(self)
        self._dark_mode_bg_timer.timeout.connect(tick)
        self._dark_mode_bg_timer.start(16)

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
        scene_half = 10000  # total scene: 20000 x 20000
        self.graphics_scene.setSceneRect(
            -scene_half, -scene_half,
            scene_half * 2, scene_half * 2
        )
        
        # Set background
        bg_color = QColor(self.scene_manager.background_color)
        self.graphics_scene.setBackgroundBrush(QBrush(bg_color))
        
        # Create view — scrollbars hidden, panning is Ctrl+Mouse only
        self.graphics_view = QGraphicsView(self.graphics_scene)
        #self.graphics_view.setRenderHint(QPainter.Antialiasing)

        # OpenGL?
        #gl_viewport = QOpenGLWidget()
        #self.graphics_view.setViewport(gl_viewport)
        #
        
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        #self.graphics_view.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
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
        menu.setStyleSheet(_CSS_NORMAL)
        
        _action_map = {}
        
        def _add(label, callback):
            action = menu.addAction(label)
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
        
        _add("New Terminal", self._enter_new_terminal_mode)
        _add("Apps", self._toggle_app_launcher)
        menu.addSeparator()
        is_visible = self.voice_control_proxy.isVisible()
        _add("Hide AI Voice" if is_visible else "Show AI Voice", self._toggle_voice_control)
        _add("Onboarding", self._launch_onboarding)
        menu.addSeparator()
        _add("Light Mode" if self._dark_mode else "Dark Mode", self.toggle_dark_mode)
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
        drop_pos = self.graphics_view.mapToScene(event.position().toPoint())

        # Close the launcher with animation after drop
        if self._app_launcher_proxy is not None:
            widget = self._app_launcher_proxy.widget()
            proxy_ref = self._app_launcher_proxy
            self._app_launcher_proxy = None

            def _on_done():
                w = proxy_ref.widget()
                proxy_ref.setWidget(None)
                self.graphics_scene.removeItem(proxy_ref)
                if w:
                    w.deleteLater()

            if widget and hasattr(widget, 'animate_close'):
                widget.animate_close(on_done=_on_done)
            else:
                _on_done()

        # Launch through filesystem parse file
        launcher = AppLauncherWidget(self)
        launcher.launch_app_at(app_path, app_name, drop_pos.x(), drop_pos.y())
        launcher.deleteLater()

        event.acceptProposedAction()
    
    def _clear_scene(self):
        """Clear the scene"""
        asyncio.create_task(self.scene_manager.clear())

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

        # Apply shadow on the widget
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(25)
        shadow.setOffset(QPointF(30, 30))
        shadow.setColor(QColor(0, 0, 0, 120))
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

        # Reapply shadow on proxy
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(25)
        shadow.setOffset(QPointF(30, 30))
        shadow.setColor(QColor(0, 0, 0, 120))
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
        widget has reached its scroll limit."""
        
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
                self._update_orbit_transform(event.position().toPoint())
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
            
            # ---- Pass-through to scene items ----
            is_mouse = event.type() in (
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.MouseMove,
            )
            if is_mouse:
                pos = event.position().toPoint()
                scene_pos = self.graphics_view.mapToScene(pos)
                item = self.graphics_scene.itemAt(scene_pos, self.graphics_view.transform())
                if item is not None:
                    return False
            
            # ---- Regular mouse events for filesystem (empty scene area) ----
            if event.type() == QEvent.Type.MouseMove:
                pos = event.position()
                if self.mouse_callback:
                    self.mouse_callback("m", int(pos.x()), int(pos.y()))
            
            elif event.type() == QEvent.Type.MouseButtonPress:
                pos = event.position()
                btn = event.button().value
                if self.mouse_callback:
                    self.mouse_callback("b", int(pos.x()), int(pos.y()), btn)
            
            elif event.type() == QEvent.Type.MouseButtonRelease:
                pos = event.position()
                btn = event.button().value
                if self.mouse_callback:
                    self.mouse_callback("r", int(pos.x()), int(pos.y()), btn)
        
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
            self.current_terminal.resize(100, 150)
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
            
            if frame_rect.width() < 100:
                frame_rect.setWidth(100)
            if frame_rect.height() < 150:
                frame_rect.setHeight(150)
            
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
        """Ctrl+1: Tilt Left & Zoom Out"""
        target = self._centered_tilt_transform(sx=0.7, sy=0.7, shx=-0.2)
        self._animate_view_transform(target, self._current_view_center())

    def view_controller_tilt_right(self):
        """Ctrl+3: Tilt Right & Zoom Out"""
        target = self._centered_tilt_transform(sx=0.7, sy=0.7, shx=0.2)
        self._animate_view_transform(target, self._current_view_center())

    def view_controller_tilt_down(self):
        """Ctrl+2: Tilt Down"""
        target = self._centered_tilt_transform(sx=0.7, sy=0.56)
        self._animate_view_transform(target, self._current_view_center())

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

    def closeEvent(self, event):
        """Handle window close — perform full shutdown of server and Qt."""
        print("\nRio shutting down...")
        self.scene_manager.detach_qt()
        # Stop the asyncio server (sets _running = False, quits Qt app)
        asyncio.ensure_future(self.rio_server.stop())
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
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # Create server
    server = RioServer(
        headless=args.headless,
        width=args.width,
        height=args.height,
        workspace=args.workspace,
        mux_mount=args.mux_mount,
    )
    
    # Handle signals
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        print("\nShutting down...")
        asyncio.create_task(server.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    # Start server
    try:
        if args.unix:
            await server.start_unix(args.unix)
        else:
            await server.start_tcp(args.host, args.port)
    except Exception as e:
        logging.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
"""
Version Manager Panel for Rio Display Server

A docked sidebar panel that provides graphical control over the scene
version history. All interaction goes through the Plan 9 filesystem
at /n/mux/<workspace>/scene/.

Design: Compact timeline with inline code preview.
Supports dark and light modes via set_dark_mode(), matching
the operator panel and main window theme switching.
Each version is a clickable strip. The current version pulses subtly.
Undo/redo have keyboard-style buttons at the top.
Save/load buttons persist entire sessions via $scene/state.

Integration:
  - Reads version info from:     /n/mux/<ws>/scene/version   (cat)
  - Writes undo to:              /n/mux/<ws>/scene/version   (echo undo)
  - Writes redo to:              /n/mux/<ws>/scene/version   (echo redo)
  - Writes goto to:              /n/mux/<ws>/scene/version   (echo <N>)
  - Writes snapshot to:          /n/mux/<ws>/scene/ctl       (echo snapshot)
  - Reads state from:            /n/mux/<ws>/scene/state     (save)
  - Writes state to:             /n/mux/<ws>/scene/state     (load/restore)

  All I/O is plain file reads/writes through the FUSE mount — no
  direct scene_manager calls.

  All filesystem I/O runs in a dedicated background thread to prevent
  the Qt event loop from blocking on slow 9P/FUSE operations.

  Sessions are saved as JSON to ~/.peribus/sessions/<workspace>/.

Usage:
  From terminal:  /versions
  Programmatic:   VersionPanel(rio_mount="/n/mux/default")
"""

import concurrent.futures
import json
import os
import time
from typing import Optional, List, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QGraphicsOpacityEffect,
    QApplication, QFileDialog
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QSize, Property
)
from PySide6.QtGui import (
    QColor, QPalette, QFont, QFontDatabase, QPainter, QPen,
    QBrush, QLinearGradient, QPainterPath
)


# ─── Theme Classes ──────────────────────────────────────────────────
# Dark and Light palettes — switchable at runtime.

class _DarkPalette:
    """Muted dark scheme — sits comfortably beside the terminal."""
    BG          = "#1E1E24"
    BG_HEADER   = "#26262E"
    BG_STRIP    = "#2A2A34"
    BG_ACTIVE   = "#35354A"
    BG_HOVER    = "#30303E"
    BORDER      = "#3A3A48"
    TEXT         = "#C8C8D0"
    TEXT_DIM     = "#78788A"
    TEXT_CODE    = "#A0A0C0"
    ACCENT       = "#6C8AFF"
    ACCENT_DIM   = "#4A5A90"
    UNDO_COLOR   = "#E8A040"
    REDO_COLOR   = "#50C878"
    DANGER       = "#D05050"
    SUCCESS      = "#50B070"
    SAVE_COLOR   = "#B08CDC"
    LOAD_COLOR   = "#60B8D0"

    # Scrollbar
    SCROLLBAR_BG       = "#1E1E24"
    SCROLLBAR_HANDLE   = "#3A3A48"
    SCROLLBAR_HANDLE_H = "#50506A"


class _LightPalette:
    """Light scheme — clean, professional."""
    BG          = "#F6F6F8"
    BG_HEADER   = "#EDEDF2"
    BG_STRIP    = "#E6E6EC"
    BG_ACTIVE   = "#D8D8E4"
    BG_HOVER    = "#DDDDE6"
    BORDER      = "#CDCDD8"
    TEXT         = "#2A2A34"
    TEXT_DIM     = "#78788A"
    TEXT_CODE    = "#4A4A70"
    ACCENT       = "#4A6AE0"
    ACCENT_DIM   = "#8090C8"
    UNDO_COLOR   = "#C08020"
    REDO_COLOR   = "#38A060"
    DANGER       = "#C04040"
    SUCCESS      = "#38905A"
    SAVE_COLOR   = "#8060B0"
    LOAD_COLOR   = "#4098B0"

    # Scrollbar
    SCROLLBAR_BG       = "#F6F6F8"
    SCROLLBAR_HANDLE   = "#CDCDD8"
    SCROLLBAR_HANDLE_H = "#B0B0C0"


class _VersionThemeProxy:
    """Proxy that delegates attribute lookups to the active palette.

    Switching is instant — every stylesheet rebuild reads T.XXX so the
    panel updates on next restyle call.
    """

    def __init__(self):
        self._active = _LightPalette

    def set_mode(self, dark: bool):
        self._active = _DarkPalette if dark else _LightPalette

    @property
    def is_dark(self) -> bool:
        return self._active is _DarkPalette

    def __getattr__(self, name):
        return getattr(self._active, name)


T = _VersionThemeProxy()





# ─── Version Strip ──────────────────────────────────────────────────

class VersionStrip(QFrame):
    """
    A single version entry in the timeline.

    Shows: version number, timestamp, label (first line of code),
    and item count. The active version has a left accent bar.
    Clicking triggers a goto.
    """

    clicked = Signal(int)  # Emits version number

    def __init__(self, version_data: Dict[str, Any], is_current: bool = False,
                 parent=None):
        super().__init__(parent)
        self.version = version_data.get("version", 0)
        self._is_current = is_current
        self._hovered = False

        self.setFixedHeight(62)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

        self._build_ui(version_data)
        self._apply_style()

    def _build_ui(self, data: Dict[str, Any]):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(0)

        # ── Left accent bar ──
        self._accent_bar = QFrame()
        self._accent_bar.setFixedWidth(3)
        self._accent_bar.setStyleSheet(
            f"background-color: {T.ACCENT if self._is_current else 'transparent'};"
            f"border: none; border-radius: 1px;"
        )
        layout.addWidget(self._accent_bar)

        # ── Content area ──
        content = QVBoxLayout()
        content.setContentsMargins(10, 6, 0, 6)
        content.setSpacing(2)

        # Top row: version number + timestamp
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        ver_label = QLabel(f"v{self.version}")
        ver_label.setFont(self._mono_font(11, bold=True))
        ver_label.setStyleSheet(
            f"color: {T.ACCENT if self._is_current else T.TEXT}; background: transparent;"
        )
        top_row.addWidget(ver_label)

        ts = data.get("timestamp", 0)
        time_str = self._format_time(ts) if ts else ""
        time_label = QLabel(time_str)
        time_label.setFont(self._mono_font(9))
        time_label.setStyleSheet(f"color: {T.TEXT_DIM}; background: transparent;")
        top_row.addWidget(time_label)

        top_row.addStretch()

        # Item count badge
        item_count = data.get("item_count", 0)
        if item_count > 0:
            badge = QLabel(f"{item_count}")
            badge.setFont(self._mono_font(8, bold=True))
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(20, 16)
            badge.setStyleSheet(f"""
                color: {T.TEXT};
                background-color: {T.ACCENT_DIM};
                border-radius: 3px;
            """)
            top_row.addWidget(badge)

        content.addLayout(top_row)

        # Code preview line
        label = data.get("label", "")
        code_line = label.strip().split('\n')[0][:60] if label else "(empty)"
        code_label = QLabel(code_line)
        code_label.setFont(self._mono_font(9))
        code_label.setStyleSheet(f"color: {T.TEXT_CODE}; background: transparent;")
        code_label.setWordWrap(False)
        content.addWidget(code_label)

        layout.addLayout(content)

    def _apply_style(self):
        if self._is_current:
            bg = T.BG_ACTIVE
        else:
            bg = T.BG_STRIP
        self.setStyleSheet(f"""
            VersionStrip {{
                background-color: {bg};
                border: none;
                border-bottom: 1px solid {T.BORDER};
            }}
        """)

    def _mono_font(self, size: int, bold: bool = False) -> QFont:
        font = QFont("Menlo, Monaco, Consolas, monospace", size)
        font.setStyleHint(QFont.Monospace)
        if bold:
            font.setBold(True)
        return font

    def _format_time(self, ts: float) -> str:
        now = time.time()
        delta = now - ts
        if delta < 60:
            return "just now"
        elif delta < 3600:
            m = int(delta / 60)
            return f"{m}m ago"
        elif delta < 86400:
            h = int(delta / 3600)
            return f"{h}h ago"
        else:
            import datetime
            dt = datetime.datetime.fromtimestamp(ts)
            return dt.strftime("%b %d %H:%M")

    # ── Mouse interaction ──

    def enterEvent(self, event):
        self._hovered = True
        if not self._is_current:
            self.setStyleSheet(f"""
                VersionStrip {{
                    background-color: {T.BG_HOVER};
                    border: none;
                    border-bottom: 1px solid {T.BORDER};
                }}
            """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.version)
        super().mousePressEvent(event)


# ─── Main Panel ─────────────────────────────────────────────────────

class VersionPanel(QWidget):
    """
    Docked sidebar for scene version management.

    All filesystem I/O runs in a background thread pool to avoid
    blocking the Qt event loop on slow 9P/FUSE mounts.
    A QTimer polls for changes every ~1.2s.
    """

    # Internal signal: emitted by the bg thread to deliver results
    # to the Qt main thread for UI updates.
    _refresh_ready = Signal(list, dict, str)   # versions, current, code
    _poll_ready    = Signal(dict)              # current
    _write_done    = Signal(str, bool)         # message, is_error
    _sessions_ready = Signal(list)             # list of (name, path, timestamp) tuples

    def __init__(self, rio_mount: str = "/n/mux/default", parent=None):
        super().__init__(parent)
        self._rio_mount = rio_mount
        self._version_path = os.path.join(rio_mount, "scene", "version")
        self._state_path = os.path.join(rio_mount, "scene", "state")
        self._last_version = -1
        self._last_version_count = -1
        self._strips: List[VersionStrip] = []
        self._session_strips: List[QFrame] = []

        # Extract workspace name from mount path
        # Expected form: /n/mux/<workspace_name>  (e.g. /n/mux/default)
        self._workspace_name = os.path.basename(rio_mount.rstrip('/'))

        # Per-workspace sessions directory
        self._SESSIONS_DIR = os.path.expanduser(
            f"~/.peribus/sessions/{self._workspace_name}"
        )

        # Ensure sessions directory exists
        os.makedirs(self._SESSIONS_DIR, exist_ok=True)

        # Single background thread for all filesystem I/O.
        self._io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="verpanel-io"
        )

        self.setMinimumWidth(240)
        self.setMaximumWidth(360)

        self._build_ui()

        # Wire internal signals → main-thread slots
        self._refresh_ready.connect(self._on_refresh_data)
        self._poll_ready.connect(self._on_poll_data)
        self._write_done.connect(self._on_write_done)
        self._sessions_ready.connect(self._on_sessions_data)

        self._start_polling()

        # Initial load (async)
        QTimer.singleShot(100, self.refresh)
        QTimer.singleShot(200, self._refresh_sessions)

    # ════════════════════════════════════════════════════════════════
    # UI Construction
    # ════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.setStyleSheet(f"background-color: {T.BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──
        header = QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG_HEADER};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel(f"VERSIONS · {self._workspace_name.upper()}")
        title.setFont(self._ui_font(10, bold=True))
        title.setStyleSheet(f"color: {T.TEXT_DIM}; background: transparent; letter-spacing: 2px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self._version_badge = QLabel("v0")
        self._version_badge.setFont(self._mono_font(10, bold=True))
        self._version_badge.setStyleSheet(f"""
            color: {T.ACCENT};
            background-color: {T.BG_STRIP};
            border-radius: 4px;
            padding: 2px 6px;
        """)
        header_layout.addWidget(self._version_badge)

        root.addWidget(header)

        # Store refs for theme switching
        self._header_frame = header
        self._title_label = title

        # ── Control bar (undo / redo / snapshot) ──
        controls = QFrame()
        controls.setFixedHeight(46)
        controls.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 6, 8, 6)
        ctrl_layout.setSpacing(6)

        self._undo_btn = self._make_button("↩ Undo", T.UNDO_COLOR)
        self._undo_btn.clicked.connect(self._do_undo)
        ctrl_layout.addWidget(self._undo_btn)

        self._redo_btn = self._make_button("Redo ↪", T.REDO_COLOR)
        self._redo_btn.clicked.connect(self._do_redo)
        ctrl_layout.addWidget(self._redo_btn)

        ctrl_layout.addStretch()

        self._snap_btn = self._make_button("📷", T.ACCENT, fixed_width=34)
        self._snap_btn.setToolTip("Take snapshot")
        self._snap_btn.clicked.connect(self._do_snapshot)
        ctrl_layout.addWidget(self._snap_btn)

        root.addWidget(controls)
        self._controls_frame = controls

        # ── Save / Load bar ──
        save_bar = QFrame()
        save_bar.setFixedHeight(46)
        save_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)
        save_layout = QHBoxLayout(save_bar)
        save_layout.setContentsMargins(8, 6, 8, 6)
        save_layout.setSpacing(6)

        self._save_btn = self._make_button("💾 Save", T.SAVE_COLOR)
        self._save_btn.setToolTip(f"Save session to ~/.peribus/sessions/{self._workspace_name}/")
        self._save_btn.clicked.connect(self._do_save)
        save_layout.addWidget(self._save_btn)

        self._load_btn = self._make_button("📂 Load", T.LOAD_COLOR)
        self._load_btn.setToolTip("Load a saved session")
        self._load_btn.clicked.connect(self._do_load)
        save_layout.addWidget(self._load_btn)

        save_layout.addStretch()

        self._sessions_toggle = self._make_button("▾", T.TEXT_DIM, fixed_width=28)
        self._sessions_toggle.setToolTip("Show saved sessions")
        self._sessions_toggle.clicked.connect(self._toggle_sessions)
        save_layout.addWidget(self._sessions_toggle)

        root.addWidget(save_bar)
        self._save_bar_frame = save_bar

        # ── Saved sessions collapsible area ──
        self._sessions_frame = QFrame()
        self._sessions_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG_HEADER};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)
        self._sessions_frame.setVisible(False)

        sessions_inner = QVBoxLayout(self._sessions_frame)
        sessions_inner.setContentsMargins(0, 0, 0, 0)
        sessions_inner.setSpacing(0)

        sessions_header = QLabel("  SAVED SESSIONS")
        sessions_header.setFont(self._ui_font(8, bold=True))
        sessions_header.setStyleSheet(
            f"color: {T.TEXT_DIM}; background: transparent;"
            f"padding: 6px 8px 2px 8px; letter-spacing: 1px;"
        )
        sessions_inner.addWidget(sessions_header)
        self._sessions_header_label = sessions_header

        self._sessions_scroll = QScrollArea()
        self._sessions_scroll.setWidgetResizable(True)
        self._sessions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._sessions_scroll.setMaximumHeight(180)
        self._sessions_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {T.BG_HEADER};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {T.BG_HEADER};
                width: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {T.BORDER};
                border-radius: 2px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        self._sessions_container = QWidget()
        self._sessions_container.setStyleSheet(f"background: {T.BG_HEADER};")
        self._sessions_list_layout = QVBoxLayout(self._sessions_container)
        self._sessions_list_layout.setContentsMargins(0, 0, 0, 0)
        self._sessions_list_layout.setSpacing(0)
        self._sessions_list_layout.addStretch()

        self._sessions_scroll.setWidget(self._sessions_container)
        sessions_inner.addWidget(self._sessions_scroll)

        root.addWidget(self._sessions_frame)

        # ── Status bar ──
        self._status_bar = QLabel("")
        self._status_bar.setFont(self._mono_font(8))
        self._status_bar.setStyleSheet(f"""
            color: {T.TEXT_DIM};
            background-color: {T.BG_HEADER};
            padding: 4px 12px;
            border-bottom: 1px solid {T.BORDER};
        """)
        self._status_bar.setFixedHeight(22)
        root.addWidget(self._status_bar)

        # ── Scrollable timeline ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {T.BG};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {T.BG};
                width: 6px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {T.BORDER};
                min-height: 20px;
                border-radius: 3px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

        self._timeline_container = QWidget()
        self._timeline_container.setStyleSheet(f"background-color: {T.BG};")
        self._timeline_layout = QVBoxLayout(self._timeline_container)
        self._timeline_layout.setContentsMargins(0, 0, 0, 0)
        self._timeline_layout.setSpacing(0)
        self._timeline_layout.addStretch()

        scroll.setWidget(self._timeline_container)
        root.addWidget(scroll)
        self._timeline_scroll = scroll

        # ── Code preview footer ──
        self._code_preview = QLabel("(no code)")
        self._code_preview.setFont(self._mono_font(9))
        self._code_preview.setWordWrap(True)
        self._code_preview.setMaximumHeight(80)
        self._code_preview.setStyleSheet(f"""
            color: {T.TEXT_CODE};
            background-color: {T.BG_HEADER};
            border-top: 1px solid {T.BORDER};
            padding: 8px 12px;
        """)
        root.addWidget(self._code_preview)

    # ════════════════════════════════════════════════════════════════
    # Dark / Light Mode
    # ════════════════════════════════════════════════════════════════

    def set_dark_mode(self, enabled: bool, duration_steps: int = 50):
        """Switch the version panel between dark and light mode.

        Called by RioWindow.toggle_dark_mode() (via TerminalWidget)
        to keep the panel in sync with the rest of the application.

        Sets the theme proxy, then rebuilds all stylesheets.
        """
        T.set_mode(enabled)
        self._restyle_all()

    def _restyle_all(self):
        """Rebuild every stylesheet in the panel from the current theme."""

        # ── Root background ──
        self.setStyleSheet(f"background-color: {T.BG};")

        # ── Header frame ──
        self._header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG_HEADER};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)
        self._title_label.setStyleSheet(
            f"color: {T.TEXT_DIM}; background: transparent; letter-spacing: 2px;"
        )
        self._version_badge.setStyleSheet(f"""
            color: {T.ACCENT};
            background-color: {T.BG_STRIP};
            border-radius: 4px;
            padding: 2px 6px;
        """)

        # ── Controls frame ──
        self._controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)

        # ── Save / Load bar ──
        self._save_bar_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)

        # ── Buttons ──
        self._restyle_button(self._undo_btn, T.UNDO_COLOR)
        self._restyle_button(self._redo_btn, T.REDO_COLOR)
        self._restyle_button(self._snap_btn, T.ACCENT)
        self._restyle_button(self._save_btn, T.SAVE_COLOR)
        self._restyle_button(self._load_btn, T.LOAD_COLOR)
        self._restyle_button(self._sessions_toggle, T.TEXT_DIM)

        # ── Sessions frame ──
        self._sessions_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG_HEADER};
                border-bottom: 1px solid {T.BORDER};
            }}
        """)
        self._sessions_header_label.setStyleSheet(
            f"color: {T.TEXT_DIM}; background: transparent;"
            f"padding: 6px 8px 2px 8px; letter-spacing: 1px;"
        )
        self._sessions_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {T.BG_HEADER};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {T.BG_HEADER};
                width: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {T.BORDER};
                border-radius: 2px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        self._sessions_container.setStyleSheet(f"background: {T.BG_HEADER};")

        # ── Status bar ──
        self._status_bar.setStyleSheet(f"""
            color: {T.TEXT_DIM};
            background-color: {T.BG_HEADER};
            padding: 4px 12px;
            border-bottom: 1px solid {T.BORDER};
        """)

        # ── Timeline scroll ──
        self._timeline_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {T.BG};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {T.BG};
                width: 6px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {T.BORDER};
                min-height: 20px;
                border-radius: 3px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        self._timeline_container.setStyleSheet(f"background-color: {T.BG};")

        # ── Code preview footer ──
        self._code_preview.setStyleSheet(f"""
            color: {T.TEXT_CODE};
            background-color: {T.BG_HEADER};
            border-top: 1px solid {T.BORDER};
            padding: 8px 12px;
        """)

        # ── Rebuild existing version strips (re-read theme colors) ──
        for strip in self._strips:
            strip._apply_style()

        # ── Rebuild session strips ──
        # Session strips are dynamically created; the simplest approach
        # is to trigger a re-list which will rebuild them with current colors.
        if self._sessions_frame.isVisible():
            self._refresh_sessions()

    def _restyle_button(self, btn: QPushButton, color: str):
        """Restyle a single button for the current theme."""
        btn.setStyleSheet(f"""
            QPushButton {{
                color: {color};
                background-color: {T.BG_STRIP};
                border: 1px solid {T.BORDER};
                border-radius: 4px;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background-color: {T.BG_HOVER};
                border-color: {color};
            }}
            QPushButton:pressed {{
                background-color: {T.BG_ACTIVE};
            }}
            QPushButton:disabled {{
                color: {T.TEXT_DIM};
                border-color: {T.BORDER};
            }}
        """)

    # ════════════════════════════════════════════════════════════════
    # Widget Helpers
    # ════════════════════════════════════════════════════════════════

    def _make_button(self, text: str, color: str,
                     fixed_width: Optional[int] = None) -> QPushButton:
        btn = QPushButton(text)
        btn.setFont(self._ui_font(9, bold=True))
        btn.setCursor(Qt.PointingHandCursor)

        if fixed_width:
            btn.setFixedSize(fixed_width, 30)
        else:
            btn.setFixedHeight(30)

        btn.setStyleSheet(f"""
            QPushButton {{
                color: {color};
                background-color: {T.BG_STRIP};
                border: 1px solid {T.BORDER};
                border-radius: 4px;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background-color: {T.BG_HOVER};
                border-color: {color};
            }}
            QPushButton:pressed {{
                background-color: {T.BG_ACTIVE};
            }}
            QPushButton:disabled {{
                color: {T.TEXT_DIM};
                border-color: {T.BORDER};
            }}
        """)
        return btn

    def _mono_font(self, size: int, bold: bool = False) -> QFont:
        font = QFont("Menlo", size)
        font.setStyleHint(QFont.Monospace)
        if bold:
            font.setBold(True)
        return font

    def _ui_font(self, size: int, bold: bool = False) -> QFont:
        font = QFont("Helvetica Neue", size)
        if bold:
            font.setBold(True)
        return font

    # ════════════════════════════════════════════════════════════════
    # Filesystem I/O — runs ONLY in the background thread
    # ════════════════════════════════════════════════════════════════

    def _fs_read_version(self) -> Optional[str]:
        """Read the version file.  Thread-safe (pure I/O)."""
        try:
            with open(self._version_path, 'r') as f:
                return f.read()
        except Exception:
            return None

    def _fs_write_version(self, content: str):
        """Write to the version file (undo/redo/goto).  Thread-safe (pure I/O)."""
        try:
            with open(self._version_path, 'w') as f:
                f.write(content)
        except Exception as e:
            self._write_done.emit(f"write error: {e}", True)

    def _parse_version_output(self, raw: str) -> tuple:
        """
        Parse the output of `cat scene/version` into structured data.

        The version file returns lines like:
            1	0 items	initial *
            2	3 items	proxy = graphics_scene.addWidget(QTex...

            current 2
            can_undo True
            can_redo False

        Returns: (versions_list, current_version, can_undo, can_redo)
        """
        versions = []
        current_ver = 0
        can_undo = False
        can_redo = False

        if not raw:
            return versions, current_ver, can_undo, can_redo

        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # Parse footer metadata
            if line.startswith("current "):
                try:
                    current_ver = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
                continue
            if line.startswith("can_undo "):
                can_undo = line.split()[1].lower() == "true"
                continue
            if line.startswith("can_redo "):
                can_redo = line.split()[1].lower() == "true"
                continue
            if line.startswith("(no versions"):
                continue

            # Parse version lines: "N\tM items\tlabel[ *]"
            parts = line.split('\t')
            if len(parts) >= 1:
                try:
                    ver_str = parts[0].strip()
                    ver_num = int(ver_str)
                except ValueError:
                    continue

                is_current = line.rstrip().endswith(' *')

                item_count = 0
                if len(parts) >= 2:
                    count_part = parts[1].strip()
                    try:
                        item_count = int(count_part.split()[0])
                    except (IndexError, ValueError):
                        pass

                label = ""
                if len(parts) >= 3:
                    label = parts[2].strip()
                    if label.endswith(' *'):
                        label = label[:-2].strip()

                versions.append({
                    "version": ver_num,
                    "item_count": item_count,
                    "label": label,
                    "timestamp": time.time(),  # approximate
                })

        return versions, current_ver, can_undo, can_redo

    def _read_state(self) -> Optional[str]:
        """Read the full scene state from $scene/state. Thread-safe."""
        try:
            with open(self._state_path, 'r') as f:
                return f.read()
        except Exception:
            return None

    def _write_state(self, content: str):
        """Write state to $scene/state (restore). Thread-safe."""
        try:
            with open(self._state_path, 'w') as f:
                f.write(content)
        except Exception as e:
            self._write_done.emit(f"restore failed: {e}", True)

    def _list_sessions(self) -> List[Dict[str, Any]]:
        """List saved sessions in the workspace sessions dir. Thread-safe."""
        sessions = []
        try:
            for entry in os.scandir(self._SESSIONS_DIR):
                if entry.name.endswith('.json') and entry.is_file():
                    try:
                        stat = entry.stat()
                        # Peek at the JSON to get metadata
                        meta = {}
                        with open(entry.path, 'r') as f:
                            raw = f.read(512)  # just the header
                        try:
                            # Try to parse version count from partial JSON
                            partial = json.loads(raw) if raw.endswith('}') else {}
                            meta = partial
                        except (json.JSONDecodeError, ValueError):
                            pass

                        sessions.append({
                            "name": entry.name.replace('.json', ''),
                            "path": entry.path,
                            "timestamp": stat.st_mtime,
                            "size": stat.st_size,
                            "version": meta.get("versions", [{}])[-1].get("version", "?")
                                       if meta.get("versions") else "?",
                        })
                    except Exception:
                        pass
        except OSError:
            pass
        # Newest first
        sessions.sort(key=lambda s: s["timestamp"], reverse=True)
        return sessions

    # ════════════════════════════════════════════════════════════════
    # Background I/O dispatch + main-thread callbacks
    # ════════════════════════════════════════════════════════════════

    def _bg_refresh(self):
        """Runs in thread pool: read version file, parse, then signal Qt."""
        raw = self._fs_read_version()
        versions, current_ver, can_undo, can_redo = self._parse_version_output(raw)
        current_data = {
            "version": current_ver,
            "can_undo": can_undo,
            "can_redo": can_redo,
            "item_count": 0,
        }
        # Find item count for current version
        for v in versions:
            if v.get("version") == current_ver:
                current_data["item_count"] = v.get("item_count", 0)
                break
        self._refresh_ready.emit(versions, current_data, "")

    def _bg_poll(self):
        """Runs in thread pool: lightweight poll, then signal Qt."""
        raw = self._fs_read_version()
        _versions, current_ver, _can_undo, _can_redo = self._parse_version_output(raw)
        item_count = 0
        for v in _versions:
            if v.get("version") == current_ver:
                item_count = v.get("item_count", 0)
                break
        self._poll_ready.emit({"version": current_ver, "item_count": item_count})

    def _bg_write_version_and_refresh(self, content: str):
        """Runs in thread pool: write to version file then full refresh."""
        self._fs_write_version(content)
        self._bg_refresh()

    def _bg_snapshot_and_refresh(self):
        """Runs in thread pool: write snapshot to scene ctl, then refresh."""
        ctl_path = os.path.join(self._rio_mount, "scene", "ctl")
        try:
            with open(ctl_path, 'w') as f:
                f.write("snapshot manual\n")
            self._write_done.emit("snapshot saved", False)
        except Exception as e:
            self._write_done.emit(f"snapshot failed: {e}", True)
        self._bg_refresh()

    def _bg_save_session(self, filepath: str):
        """Runs in thread pool: read $scene/state → write to filepath."""
        state_json = self._read_state()
        if not state_json:
            self._write_done.emit("save failed: could not read state", True)
            return
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(state_json)
            name = os.path.basename(filepath).replace('.json', '')
            self._write_done.emit(f"saved: {name}", False)
        except Exception as e:
            self._write_done.emit(f"save failed: {e}", True)
        # Refresh sessions list
        self._bg_list_sessions()

    def _bg_load_session(self, filepath: str):
        """Runs in thread pool: read file → write to $scene/state."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            # Validate it's a proper state file
            state = json.loads(content)
            if state.get("rio_state") != 1:
                self._write_done.emit("load failed: not a rio state file", True)
                return
        except (json.JSONDecodeError, ValueError):
            self._write_done.emit("load failed: invalid JSON", True)
            return
        except Exception as e:
            self._write_done.emit(f"load failed: {e}", True)
            return

        self._write_state(content)
        name = os.path.basename(filepath).replace('.json', '')
        self._write_done.emit(f"loaded: {name}", False)
        # Give the server a moment to process the restore, then refresh
        import time as _time
        _time.sleep(0.3)
        self._bg_refresh()

    def _bg_delete_session(self, filepath: str):
        """Runs in thread pool: delete a saved session file."""
        try:
            os.unlink(filepath)
            name = os.path.basename(filepath).replace('.json', '')
            self._write_done.emit(f"deleted: {name}", False)
        except Exception as e:
            self._write_done.emit(f"delete failed: {e}", True)
        self._bg_list_sessions()

    def _bg_list_sessions(self):
        """Runs in thread pool: scan sessions dir, emit results."""
        sessions = self._list_sessions()
        self._sessions_ready.emit(sessions)

    # ════════════════════════════════════════════════════════════════
    # Actions (submit to background thread)
    # ════════════════════════════════════════════════════════════════

    def _do_undo(self):
        self._io_pool.submit(self._bg_write_version_and_refresh, "undo\n")

    def _do_redo(self):
        self._io_pool.submit(self._bg_write_version_and_refresh, "redo\n")

    def _do_snapshot(self):
        self._io_pool.submit(self._bg_snapshot_and_refresh)

    def _do_goto(self, version: int):
        self._io_pool.submit(self._bg_write_version_and_refresh, f"{version}\n")

    def _do_save(self):
        """Save current session — auto-named by timestamp."""
        import datetime
        name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self._SESSIONS_DIR, f"{name}.json")
        self._io_pool.submit(self._bg_save_session, filepath)

    def _do_save_as(self):
        """Save current session with a user-chosen name."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session",
            self._SESSIONS_DIR,
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            if not path.endswith('.json'):
                path += '.json'
            self._io_pool.submit(self._bg_save_session, path)

    def _do_load(self):
        """Load session from file picker."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session",
            self._SESSIONS_DIR,
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._io_pool.submit(self._bg_load_session, path)

    def _do_load_session(self, filepath: str):
        """Load a specific session file (from the sessions list)."""
        self._io_pool.submit(self._bg_load_session, filepath)

    def _do_delete_session(self, filepath: str):
        """Delete a saved session file."""
        self._io_pool.submit(self._bg_delete_session, filepath)

    def _toggle_sessions(self):
        """Toggle the saved sessions panel."""
        visible = not self._sessions_frame.isVisible()
        self._sessions_frame.setVisible(visible)
        self._sessions_toggle.setText("▴" if visible else "▾")
        if visible:
            self._refresh_sessions()

    def _refresh_sessions(self):
        """Refresh the saved sessions list (async)."""
        self._io_pool.submit(self._bg_list_sessions)

    # ════════════════════════════════════════════════════════════════
    # Refresh / Polling (non-blocking)
    # ════════════════════════════════════════════════════════════════

    def _start_polling(self):
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(1200)

    def _poll(self):
        """Lightweight poll — dispatch to bg thread."""
        if not self.isVisible():
            return
        self._io_pool.submit(self._bg_poll)

    def refresh(self):
        """Full refresh — dispatch to bg thread."""
        self._io_pool.submit(self._bg_refresh)

    # ════════════════════════════════════════════════════════════════
    # Main-thread slots (receive data from background thread)
    # ════════════════════════════════════════════════════════════════

    def _on_poll_data(self, current: Dict[str, Any]):
        """Received lightweight poll result — trigger full refresh if changed."""
        ver   = current.get("version", 0)
        count = current.get("item_count", 0)
        if ver != self._last_version or count != self._last_version_count:
            self.refresh()

    def _on_refresh_data(self, versions: List[Dict[str, Any]],
                         current_data: Dict[str, Any], code: str):
        """Received full version data from bg thread — update UI."""
        current_ver = current_data.get("version", 0)
        self._last_version = current_ver
        self._last_version_count = current_data.get("item_count", 0)

        # Update header badge
        self._version_badge.setText(f"v{current_ver}")

        # Update status
        n = len(versions)
        self._status_bar.setText(
            f"{n} version{'s' if n != 1 else ''}  ·  "
            f"{current_data.get('item_count', 0)} items"
        )

        # Update undo/redo enabled state from parsed metadata
        can_undo = current_data.get("can_undo", False)
        can_redo = current_data.get("can_redo", False)
        # Fallback: infer from position in list
        if not can_undo and not can_redo:
            current_idx = -1
            for i, v in enumerate(versions):
                if v.get("version") == current_ver:
                    current_idx = i
                    break
            can_undo = current_idx > 0
            can_redo = current_idx < len(versions) - 1

        self._undo_btn.setEnabled(can_undo)
        self._redo_btn.setEnabled(can_redo)

        # Rebuild timeline strips
        self._rebuild_timeline(versions, current_ver)

        # Update code preview — use label from current version
        current_label = ""
        for v in versions:
            if v.get("version") == current_ver:
                current_label = v.get("label", "")
                break
        preview_lines = current_label.strip().split('\n')
        if len(preview_lines) > 4:
            preview = '\n'.join(preview_lines[:4]) + '\n…'
        else:
            preview = current_label.strip() or "(no code)"
        self._code_preview.setText(preview)

    def _on_write_done(self, message: str, is_error: bool):
        """Received write completion from bg thread — flash status."""
        self._flash_status(message, error=is_error)

    def _on_sessions_data(self, sessions: List[Dict[str, Any]]):
        """Received sessions listing from bg thread — rebuild UI."""
        self._rebuild_sessions(sessions)

    def _rebuild_sessions(self, sessions: List[Dict[str, Any]]):
        """Clear and rebuild the saved sessions list."""
        for strip in self._session_strips:
            self._sessions_list_layout.removeWidget(strip)
            strip.deleteLater()
        self._session_strips.clear()

        if not sessions:
            empty = QLabel("  no saved sessions")
            empty.setFont(self._mono_font(9))
            empty.setStyleSheet(
                f"color: {T.TEXT_DIM}; background: transparent; padding: 8px;"
            )
            self._sessions_list_layout.insertWidget(0, empty)
            self._session_strips.append(empty)
            return

        for sdata in sessions:
            strip = self._make_session_strip(sdata)
            self._sessions_list_layout.insertWidget(
                self._sessions_list_layout.count() - 1, strip
            )
            self._session_strips.append(strip)

    def _make_session_strip(self, sdata: Dict[str, Any]) -> QFrame:
        """Build a single saved session entry."""
        frame = QFrame()
        frame.setFixedHeight(42)
        frame.setCursor(Qt.PointingHandCursor)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {T.BG_HEADER};
                border-bottom: 1px solid {T.BORDER};
            }}
            QFrame:hover {{
                background-color: {T.BG_HOVER};
            }}
        """)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 4, 8, 4)
        layout.setSpacing(6)

        # Name + time info
        info_col = QVBoxLayout()
        info_col.setSpacing(0)

        name_label = QLabel(sdata["name"])
        name_label.setFont(self._mono_font(9, bold=True))
        name_label.setStyleSheet(f"color: {T.TEXT}; background: transparent;")
        info_col.addWidget(name_label)

        ts = sdata.get("timestamp", 0)
        size_kb = sdata.get("size", 0) / 1024
        ver = sdata.get("version", "?")
        meta_text = f"v{ver}  ·  {size_kb:.0f}KB"
        if ts:
            now = time.time()
            delta = now - ts
            if delta < 3600:
                meta_text += f"  ·  {int(delta/60)}m ago"
            elif delta < 86400:
                meta_text += f"  ·  {int(delta/3600)}h ago"
            else:
                import datetime
                dt = datetime.datetime.fromtimestamp(ts)
                meta_text += f"  ·  {dt.strftime('%b %d')}"

        meta_label = QLabel(meta_text)
        meta_label.setFont(self._mono_font(8))
        meta_label.setStyleSheet(f"color: {T.TEXT_DIM}; background: transparent;")
        info_col.addWidget(meta_label)

        layout.addLayout(info_col)
        layout.addStretch()

        # Load button
        load_btn = QPushButton("↥")
        load_btn.setFixedSize(24, 24)
        load_btn.setCursor(Qt.PointingHandCursor)
        load_btn.setToolTip("Load this session")
        load_btn.setFont(self._mono_font(11, bold=True))
        load_btn.setStyleSheet(f"""
            QPushButton {{
                color: {T.LOAD_COLOR};
                background: {T.BG_STRIP};
                border: 1px solid {T.BORDER};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background: {T.BG_HOVER};
                border-color: {T.LOAD_COLOR};
            }}
        """)
        filepath = sdata["path"]
        load_btn.clicked.connect(lambda checked, p=filepath: self._do_load_session(p))
        layout.addWidget(load_btn)

        # Delete button
        del_btn = QPushButton("×")
        del_btn.setFixedSize(24, 24)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setToolTip("Delete this session")
        del_btn.setFont(self._mono_font(11, bold=True))
        del_btn.setStyleSheet(f"""
            QPushButton {{
                color: {T.TEXT_DIM};
                background: {T.BG_STRIP};
                border: 1px solid {T.BORDER};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                color: {T.DANGER};
                background: {T.BG_HOVER};
                border-color: {T.DANGER};
            }}
        """)
        del_btn.clicked.connect(lambda checked, p=filepath: self._do_delete_session(p))
        layout.addWidget(del_btn)

        return frame

    def _rebuild_timeline(self, versions: List[Dict], current_ver: int):
        """Clear and rebuild the version strip list."""
        # Remove old strips
        for strip in self._strips:
            self._timeline_layout.removeWidget(strip)
            strip.deleteLater()
        self._strips.clear()

        # Versions in reverse chronological order (newest at top)
        for vdata in reversed(versions):
            ver = vdata.get("version", 0)
            is_current = (ver == current_ver)
            strip = VersionStrip(vdata, is_current=is_current)
            strip.clicked.connect(self._do_goto)

            # Insert before the stretch
            self._timeline_layout.insertWidget(
                self._timeline_layout.count() - 1, strip
            )
            self._strips.append(strip)

    # ════════════════════════════════════════════════════════════════
    # Status flash
    # ════════════════════════════════════════════════════════════════

    def _flash_status(self, msg: str, error: bool = False):
        color = T.DANGER if error else T.SUCCESS
        self._status_bar.setStyleSheet(f"""
            color: {color};
            background-color: {T.BG_HEADER};
            padding: 4px 12px;
            border-bottom: 1px solid {T.BORDER};
        """)
        self._status_bar.setText(msg)

        def _restore():
            self._status_bar.setStyleSheet(f"""
                color: {T.TEXT_DIM};
                background-color: {T.BG_HEADER};
                padding: 4px 12px;
                border-bottom: 1px solid {T.BORDER};
            """)

        QTimer.singleShot(2000, _restore)
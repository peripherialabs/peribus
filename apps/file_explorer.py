from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView,
    QListView, QFileSystemModel, QSplitter, QLabel,
    QToolBar, QLineEdit, QFileDialog, QGraphicsDropShadowEffect,
    QTextEdit, QScrollArea, QFrame, QSizePolicy, QToolButton,
    QStyledItemDelegate, QStyle, QAbstractItemView, QStackedWidget,
)
from PySide6.QtCore import (
    QDir, Qt, Signal, QSize, QFileInfo, QPoint, QUrl, QMimeDatabase,
    QPropertyAnimation, QEasingCurve, QRect, QTimer, QRectF,
)
from PySide6.QtGui import (
    QIcon, QColor, QMouseEvent, QPixmap, QPainter, QPainterPath,
    QFont, QFontMetrics, QLinearGradient, QBrush, QPen, QPalette,
    QSyntaxHighlighter, QTextCharFormat,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWebEngineWidgets import QWebEngineView
import os
import re

# ── Cleanup ──────────────────────────────────────────────────────────────────
try:
    file_explorer.deleteLater()
except Exception:
    pass


# ── Syntax Highlighter ───────────────────────────────────────────────────────
class SimpleHighlighter(QSyntaxHighlighter):
    KEYWORDS = (
        r'\b(def|class|import|from|return|if|elif|else|for|while|try|except|'
        r'finally|with|as|yield|lambda|pass|break|continue|raise|in|not|and|'
        r'or|is|True|False|None|self|async|await|function|const|let|var|'
        r'export|default|new|this|throw|catch|typeof|instanceof)\b'
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []

        kw = QTextCharFormat()
        kw.setForeground(QColor(80, 100, 200))
        kw.setFontWeight(QFont.Bold)
        self.rules.append((re.compile(self.KEYWORDS), kw))

        s = QTextCharFormat()
        s.setForeground(QColor(40, 130, 80))
        self.rules.append((re.compile(r'(".*?"|\'.*?\')'), s))

        n = QTextCharFormat()
        n.setForeground(QColor(180, 120, 40))
        self.rules.append((re.compile(r'\b\d+(\.\d+)?\b'), n))

        c = QTextCharFormat()
        c.setForeground(QColor(140, 140, 160))
        c.setFontItalic(True)
        self.rules.append((re.compile(r'#[^\n]*'), c))
        self.rules.append((re.compile(r'//[^\n]*'), c))

        f = QTextCharFormat()
        f.setForeground(QColor(50, 120, 170))
        self.rules.append((re.compile(r'\b([A-Za-z_]\w*)\s*\('), f))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for m in pattern.finditer(text):
                start = m.start(1) if m.lastindex else m.start()
                length = m.end(1) - m.start(1) if m.lastindex else m.end() - m.start()
                self.setFormat(start, length, fmt)


# ── Main Widget ──────────────────────────────────────────────────────────────
class DraggableFileExplorer(QWidget):

    RESIZE_MARGIN = 12

    def __init__(self):
        super().__init__()
        self.setParent(main_window)
        self.setGeometry(80, 80, 1320, 700)
        self.setWindowFlags(Qt.Widget)

        # ── Fully transparent background (glass pane) ────────────────────
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")
        self.setObjectName("FileExplorerRoot")

        # Drag / resize state
        self.drag_position = QPoint()
        self.is_dragging = False
        self._resizing = False
        self._resize_corner = None
        self._resize_start_pos = None
        self._resize_start_geometry = None

        # Media player (lazy)
        self._player = None
        self._audio_output = None
        self._video_widget = None
        self._web_view = None
        self._proxy = None  # for scene embedding

        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Outer frame — the "glass pane" ───────────────────────────────
        self.glass_frame = QFrame()
        self.glass_frame.setFrameStyle(QFrame.StyledPanel)
        self.glass_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0);
                border: 2px solid rgba(150, 150, 150, 200);
                border-radius: 5px;
            }
        """)
        frame_layout = QVBoxLayout(self.glass_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        # ── Title bar ────────────────────────────────────────────────────
        self.title_bar = QFrame()
        self.title_bar.setFixedHeight(36)
        self.title_bar.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
                border-bottom: 1px solid rgba(150, 150, 150, 100);
            }
        """)
        tb = QHBoxLayout(self.title_bar)
        tb.setContentsMargins(12, 0, 8, 0)
        tb.setSpacing(7)

        # Traffic-light dots
        for colour in ["rgba(255, 95, 87, 200)", "rgba(255, 189, 46, 200)", "rgba(39, 201, 63, 200)"]:
            dot = QLabel()
            dot.setFixedSize(11, 11)
            dot.setStyleSheet(f"""
                background-color: {colour};
                border-radius: 5px;
                border: none;
            """)
            tb.addWidget(dot)

        tb.addSpacing(8)
        title = QLabel("File Explorer")
        title.setStyleSheet("""
            color: rgba(0, 0, 0, 140);
            font-size: 12px;
            font-weight: 600;
            border: none;
            background: transparent;
        """)
        tb.addWidget(title)
        tb.addStretch()

        close_btn = QToolButton()
        close_btn.setText("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QToolButton {
                background: transparent;
                color: rgba(0, 0, 0, 120);
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: rgba(255, 70, 70, 60);
                color: rgba(200, 40, 40, 220);
            }
        """)
        close_btn.clicked.connect(self.close_explorer)
        tb.addWidget(close_btn)
        frame_layout.addWidget(self.title_bar)

        # ── Navigation bar ───────────────────────────────────────────────
        nav = QFrame()
        nav.setFixedHeight(38)
        nav.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
                border-bottom: 1px solid rgba(150, 150, 150, 60);
            }
        """)
        nav_l = QHBoxLayout(nav)
        nav_l.setContentsMargins(8, 3, 8, 3)
        nav_l.setSpacing(2)

        _BTN_CSS = """
            QToolButton {
                background: transparent;
                color: rgba(0, 0, 0, 150);
                font-size: 15px;
                border: none;
                border-radius: 4px;
                padding: 3px 7px;
            }
            QToolButton:hover {
                background: rgba(0, 0, 0, 30);
            }
            QToolButton:pressed {
                background: rgba(0, 0, 0, 50);
            }
        """

        def _btn(text, tip, slot):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setFixedSize(28, 28)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_BTN_CSS)
            b.clicked.connect(slot)
            nav_l.addWidget(b)
            return b

        _btn("◀", "Back",    self.go_back)
        _btn("▶", "Forward", self.go_forward)
        _btn("▲", "Up",      self.go_up)
        _btn("⌂", "Home",    self.go_home)
        nav_l.addSpacing(4)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Enter path…")
        self.path_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 0);
                color: rgba(0, 0, 0, 200);
                border: 1px solid rgba(150, 150, 150, 120);
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                selection-background-color: rgba(100, 100, 255, 80);
            }
            QLineEdit:focus {
                border: 1px solid rgba(100, 100, 100, 200);
            }
        """)
        self.path_edit.returnPressed.connect(self.on_path_entered)
        nav_l.addWidget(self.path_edit)
        _btn("📂", "Browse…", self.browse_directory)
        frame_layout.addWidget(nav)

        # ── Body ─────────────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet("background: transparent; border: none;")
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)

        # ── Tree sidebar ─────────────────────────────────────────────────
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setIndentation(16)
        self.tree_view.setMinimumWidth(170)
        self.tree_view.setMaximumWidth(320)
        self.tree_view.setStyleSheet("""
            QTreeView {
                background-color: transparent;
                color: rgba(0, 0, 0, 200);
                border: none;
                border-right: 1px solid rgba(150, 150, 150, 60);
                padding: 4px 2px;
                font-size: 12px;
            }
            QTreeView::item {
                padding: 3px 5px;
                border-radius: 4px;
                margin: 1px 3px;
            }
            QTreeView::item:hover {
                background-color: rgba(0, 0, 0, 20);
            }
            QTreeView::item:selected {
                background-color: rgba(0, 0, 0, 40);
                color: rgba(0, 0, 0, 220);
            }
            QTreeView::branch {
                background: transparent;
            }
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {
                image: none; border-image: none;
            }
            QTreeView::branch:open:has-children:!has-siblings,
            QTreeView::branch:open:has-children:has-siblings {
                image: none; border-image: none;
            }
        """)

        # ── File list ────────────────────────────────────────────────────
        self.list_view = QListView()
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setGridSize(QSize(100, 90))
        self.list_view.setIconSize(QSize(42, 42))
        self.list_view.setSpacing(6)
        self.list_view.setWordWrap(True)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_view.setStyleSheet("""
            QListView {
                background-color: transparent;
                color: rgba(0, 0, 0, 200);
                border: none;
                padding: 6px;
                font-size: 12px;
            }
            QListView::item {
                padding: 5px;
                border-radius: 6px;
                margin: 2px;
            }
            QListView::item:hover {
                background-color: rgba(0, 0, 0, 20);
            }
            QListView::item:selected {
                background-color: rgba(0, 0, 0, 40);
                color: rgba(0, 0, 0, 220);
            }
        """)

        # ── Viewer panel ─────────────────────────────────────────────────
        self.viewer_panel = QFrame()
        self.viewer_panel.setMinimumWidth(320)
        self.viewer_panel.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
                border-left: 1px solid rgba(150, 150, 150, 60);
            }
        """)
        vp_l = QVBoxLayout(self.viewer_panel)
        vp_l.setContentsMargins(0, 0, 0, 0)
        vp_l.setSpacing(0)

        # File info header
        self.file_info_label = QLabel("Select a file to preview")
        self.file_info_label.setAlignment(Qt.AlignCenter)
        self.file_info_label.setFixedHeight(30)
        self.file_info_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: rgba(0, 0, 0, 120);
                border: none;
                border-bottom: 1px solid rgba(150, 150, 150, 40);
                padding: 0 10px;
                font-size: 11px;
                font-weight: 600;
            }
        """)
        vp_l.addWidget(self.file_info_label)

        # Viewer stack
        self.viewer_stack = QStackedWidget()
        self.viewer_stack.setStyleSheet("background: transparent; border: none;")

        # 0 — placeholder
        self.placeholder = QLabel("📄\n\nNo file selected")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: rgba(0, 0, 0, 80); font-size: 13px; border: none; background: transparent;")
        self.viewer_stack.addWidget(self.placeholder)

        # 1 — text viewer
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.text_viewer.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                color: rgba(0, 0, 0, 210);
                border: none;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                selection-background-color: rgba(100, 100, 255, 80);
            }
        """)
        self.highlighter = SimpleHighlighter(self.text_viewer.document())
        self.viewer_stack.addWidget(self.text_viewer)

        # 2 — image viewer
        self.image_scroll = QScrollArea()
        self.image_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: transparent; border: none;")
        self.image_scroll.setWidget(self.image_label)
        self.viewer_stack.addWidget(self.image_scroll)

        # 3 — video/audio
        self._media_container = QFrame()
        self._media_container.setStyleSheet("background: transparent; border: none;")
        self._media_layout = QVBoxLayout(self._media_container)
        self._media_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_stack.addWidget(self._media_container)

        # 4 — PDF (webview)
        self._web_container = QFrame()
        self._web_container.setStyleSheet("background: transparent; border: none;")
        self._web_layout = QVBoxLayout(self._web_container)
        self._web_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_stack.addWidget(self._web_container)

        # 5 — unsupported
        self.unsupported_label = QLabel()
        self.unsupported_label.setAlignment(Qt.AlignCenter)
        self.unsupported_label.setWordWrap(True)
        self.unsupported_label.setStyleSheet("color: rgba(0, 0, 0, 100); font-size: 12px; border: none; background: transparent;")
        self.viewer_stack.addWidget(self.unsupported_label)

        self.viewer_stack.setCurrentIndex(0)
        vp_l.addWidget(self.viewer_stack)

        # ── Media controls ───────────────────────────────────────────────
        self.media_controls = QFrame()
        self.media_controls.setFixedHeight(38)
        self.media_controls.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
                border-top: 1px solid rgba(150, 150, 150, 40);
            }
        """)
        mc = QHBoxLayout(self.media_controls)
        mc.setContentsMargins(10, 3, 10, 3)

        _CTRL_CSS = """
            QToolButton {
                background: transparent;
                color: rgba(0, 0, 0, 160);
                font-size: 17px;
                border: none;
                border-radius: 4px;
                padding: 3px 8px;
            }
            QToolButton:hover {
                background: rgba(0, 0, 0, 25);
            }
        """

        self._play_btn = QToolButton()
        self._play_btn.setText("▶")
        self._play_btn.setStyleSheet(_CTRL_CSS)
        self._play_btn.setCursor(Qt.PointingHandCursor)
        self._play_btn.clicked.connect(self._toggle_playback)
        mc.addWidget(self._play_btn)

        self._stop_btn = QToolButton()
        self._stop_btn.setText("⏹")
        self._stop_btn.setStyleSheet(_CTRL_CSS)
        self._stop_btn.setCursor(Qt.PointingHandCursor)
        self._stop_btn.clicked.connect(self._stop_playback)
        mc.addWidget(self._stop_btn)

        self._media_time = QLabel("0:00")
        self._media_time.setStyleSheet("color: rgba(0, 0, 0, 100); font-size: 11px; border: none; background: transparent; font-family: monospace;")
        mc.addStretch()
        mc.addWidget(self._media_time)
        self.media_controls.hide()
        vp_l.addWidget(self.media_controls)

        # ── Splitter ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter {
                background: transparent;
                border: none;
            }
            QSplitter::handle {
                background: rgba(150, 150, 150, 50);
                width: 1px;
            }
        """)
        splitter.setHandleWidth(1)
        splitter.addWidget(self.tree_view)
        splitter.addWidget(self.list_view)
        splitter.addWidget(self.viewer_panel)
        splitter.setSizes([200, 480, 420])

        body_l.addWidget(splitter)
        frame_layout.addWidget(body, 1)

        # ── Status bar ───────────────────────────────────────────────────
        self.status_bar = QLabel()
        self.status_bar.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_bar.setFixedHeight(24)
        self.status_bar.setStyleSheet("""
            QLabel {
                background: transparent;
                color: rgba(0, 0, 0, 90);
                border: none;
                border-top: 1px solid rgba(150, 150, 150, 40);
                padding: 0 12px;
                font-size: 11px;
            }
        """)
        frame_layout.addWidget(self.status_bar)

        # ── Scrollbar styling (matches terminal_widget) ──────────────────
        self.glass_frame.setStyleSheet(self.glass_frame.styleSheet() + """
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
                background: rgba(160, 160, 160, 0.25);
            }
            QScrollBar::handle:vertical:pressed {
                background: rgba(160, 160, 160, 0.35);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; background: transparent; border: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
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
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px; background: transparent; border: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: transparent;
            }
            QToolTip {
                background-color: rgba(255, 255, 255, 210);
                color: rgba(0, 0, 0, 200);
                border: 1px solid rgba(150, 150, 150, 180);
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
        """)

        root.addWidget(self.glass_frame)

        # ── Models ───────────────────────────────────────────────────────
        self.dir_model = QFileSystemModel()
        self.dir_model.setRootPath("")
        self.dir_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        self.file_model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)

        self.tree_view.setModel(self.dir_model)
        self.list_view.setModel(self.file_model)
        for col in range(1, self.dir_model.columnCount()):
            self.tree_view.hideColumn(col)

        # ── Signals ──────────────────────────────────────────────────────
        self.tree_view.clicked.connect(self.on_tree_clicked)
        self.list_view.doubleClicked.connect(self.on_list_double_clicked)
        self.list_view.clicked.connect(self.on_list_single_clicked)

        # ── State ────────────────────────────────────────────────────────
        self.history = []
        self.history_index = -1

        self.text_extensions = {
            '.txt', '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
            '.json', '.xml', '.md', '.cpp', '.c', '.h', '.hpp', '.java',
            '.php', '.rb', '.go', '.rs', '.sql', '.yml', '.yaml', '.ini',
            '.cfg', '.conf', '.log', '.sh', '.bash', '.zsh', '.toml',
            '.env', '.gitignore', '.dockerfile', '.cmake', '.makefile',
            '.r', '.lua', '.pl', '.swift', '.kt', '.cs', '.vb', '.bat',
        }
        self.image_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
            '.webp', '.svg', '.ico',
        }
        self.video_extensions = {
            '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv',
            '.m4v', '.3gp',
        }
        self.audio_extensions = {
            '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.opus',
        }
        self.pdf_extensions = {'.pdf'}

        self.go_home()

    # ══════════════════════════════════════════════════════════════════════
    #  DRAG & RESIZE  (matches terminal_widget contract)
    # ══════════════════════════════════════════════════════════════════════

    def _get_resize_corner(self, pos):
        rect = self.rect()
        m = self.RESIZE_MARGIN
        if pos.x() <= m and pos.y() <= m:
            return 'tl'
        if pos.x() >= rect.width() - m and pos.y() <= m:
            return 'tr'
        if pos.x() <= m and pos.y() >= rect.height() - m:
            return 'bl'
        if pos.x() >= rect.width() - m and pos.y() >= rect.height() - m:
            return 'br'
        return None

    def _update_cursor_for_resize(self, corner):
        if corner in ('tl', 'br'):
            self.setCursor(Qt.SizeFDiagCursor)
        elif corner in ('tr', 'bl'):
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _set_geometry_proxy_aware(self, x, y, w, h):
        if self._proxy is not None:
            self._proxy.setPos(x, y)
            self.setFixedSize(int(w), int(h))
        else:
            self.setGeometry(int(x), int(y), int(w), int(h))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            corner = self._get_resize_corner(event.position().toPoint())
            if corner:
                self._resizing = True
                self._resize_corner = corner
                self._resize_start_pos = event.globalPosition().toPoint()
                if self._proxy is not None:
                    pp = self._proxy.pos()
                    self._resize_start_geometry = QRectF(pp.x(), pp.y(), self.width(), self.height()).toRect()
                else:
                    self._resize_start_geometry = self.geometry()
                event.accept()
                return

            # Drag via title bar OR Ctrl+click anywhere
            bar = self.title_bar.geometry()
            ctrl = event.modifiers() & Qt.ControlModifier
            if bar.contains(event.position().toPoint()) or ctrl:
                self.is_dragging = True
                if self._proxy is not None:
                    view = self._proxy.scene().views()[0]
                    scene_pos = view.mapToScene(view.mapFromGlobal(event.globalPosition().toPoint()))
                    self._drag_offset = scene_pos - self._proxy.pos()
                else:
                    self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._resizing and event.buttons() == Qt.NoButton:
            self._update_cursor_for_resize(self._get_resize_corner(event.position().toPoint()))

        if event.buttons() & Qt.LeftButton:
            if self._resizing and self._resize_corner:
                delta = event.globalPosition().toPoint() - self._resize_start_pos
                geo = self._resize_start_geometry
                mw, mh = 400, 250
                c = self._resize_corner
                if c == 'tl':
                    nw, nh = geo.width() - delta.x(), geo.height() - delta.y()
                    if nw >= mw and nh >= mh:
                        self._set_geometry_proxy_aware(geo.x() + delta.x(), geo.y() + delta.y(), nw, nh)
                elif c == 'tr':
                    nw, nh = geo.width() + delta.x(), geo.height() - delta.y()
                    if nw >= mw and nh >= mh:
                        self._set_geometry_proxy_aware(geo.x(), geo.y() + delta.y(), nw, nh)
                elif c == 'bl':
                    nw, nh = geo.width() - delta.x(), geo.height() + delta.y()
                    if nw >= mw and nh >= mh:
                        self._set_geometry_proxy_aware(geo.x() + delta.x(), geo.y(), nw, nh)
                elif c == 'br':
                    nw, nh = geo.width() + delta.x(), geo.height() + delta.y()
                    if nw >= mw and nh >= mh:
                        if self._proxy is not None:
                            self.setFixedSize(int(nw), int(nh))
                        else:
                            self.resize(int(nw), int(nh))
                event.accept()
            elif self.is_dragging:
                if self._proxy is not None:
                    view = self._proxy.scene().views()[0]
                    scene_pos = view.mapToScene(view.mapFromGlobal(event.globalPosition().toPoint()))
                    self._proxy.setPos(scene_pos - self._drag_offset)
                else:
                    new_pos = event.globalPosition().toPoint() - self.drag_position
                    if hasattr(main_window, 'geometry'):
                        r = main_window.geometry()
                        g = self.geometry()
                        new_pos.setX(max(0, min(new_pos.x(), r.width() - g.width())))
                        new_pos.setY(max(0, min(new_pos.y(), r.height() - g.height())))
                    self.move(new_pos)
                event.accept()
            else:
                super().mouseMoveEvent(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._resizing:
                self._resizing = False
                self._resize_corner = None
                event.accept()
                return
            if self.is_dragging:
                self.is_dragging = False
                event.accept()
                return
        super().mouseReleaseEvent(event)

    # ══════════════════════════════════════════════════════════════════════
    #  NAVIGATION
    # ══════════════════════════════════════════════════════════════════════

    def on_tree_clicked(self, index):
        self.set_current_path(self.dir_model.filePath(index))

    def on_list_double_clicked(self, index):
        path = self.file_model.filePath(index)
        if QFileInfo(path).isDir():
            self.set_current_path(path)
        else:
            self.view_file(path)

    def on_list_single_clicked(self, index):
        path = self.file_model.filePath(index)
        if not QFileInfo(path).isDir():
            self.view_file(path)

    def on_path_entered(self):
        p = self.path_edit.text()
        if QDir(p).exists():
            self.set_current_path(p)
        else:
            self.status_bar.setText("⚠ Invalid path")

    def set_current_path(self, path):
        self.path_edit.setText(path)
        self.list_view.setRootIndex(self.file_model.setRootPath(path))
        idx = self.dir_model.index(path)
        if idx.isValid():
            self.tree_view.setCurrentIndex(idx)
            self.tree_view.expand(idx)

        self.file_info_label.setText("Select a file to preview")
        self.viewer_stack.setCurrentIndex(0)
        self._stop_media()
        self.media_controls.hide()

        try:
            n = len(os.listdir(path))
            self.status_bar.setText(f"{n} item{'s' if n != 1 else ''}")
        except Exception:
            self.status_bar.setText("Access denied")

        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        self.history.append(path)
        self.history_index = len(self.history) - 1

    def go_back(self):
        if self.history_index > 0:
            self.history_index -= 1
            p = self.history[self.history_index]
            self.path_edit.setText(p)
            self.list_view.setRootIndex(self.file_model.setRootPath(p))
            idx = self.dir_model.index(p)
            if idx.isValid():
                self.tree_view.setCurrentIndex(idx)

    def go_forward(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            p = self.history[self.history_index]
            self.path_edit.setText(p)
            self.list_view.setRootIndex(self.file_model.setRootPath(p))

    def go_up(self):
        d = QDir(self.path_edit.text())
        if d.cdUp():
            self.set_current_path(d.path())

    def go_home(self):
        self.set_current_path(QDir.homePath())

    def browse_directory(self):
        p = QFileDialog.getExistingDirectory(self, "Select Directory", self.path_edit.text())
        if p:
            self.set_current_path(p)

    # ══════════════════════════════════════════════════════════════════════
    #  FILE VIEWING
    # ══════════════════════════════════════════════════════════════════════

    def view_file(self, file_path):
        info = QFileInfo(file_path)
        ext = f".{info.suffix().lower()}"
        size = info.size()
        name = info.fileName()

        self.file_info_label.setText(f"{name}  ·  {self._fmt_size(size)}")
        self._stop_media()
        self.media_controls.hide()

        if ext in self.text_extensions:
            self._view_text(file_path)
        elif ext in self.image_extensions:
            self._view_image(file_path)
        elif ext in self.video_extensions:
            self._view_video(file_path)
        elif ext in self.audio_extensions:
            self._view_audio(file_path)
        elif ext in self.pdf_extensions:
            self._view_pdf(file_path)
        elif size < 512 * 1024:
            try:
                self._view_text(file_path)
            except Exception:
                self._show_unsupported(name, ext)
        else:
            self._show_unsupported(name, ext)

    def _view_text(self, path):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            self.text_viewer.setPlainText(f.read())
        self.viewer_stack.setCurrentIndex(1)

    def _view_image(self, path):
        pix = QPixmap(path)
        if not pix.isNull():
            w = max(self.viewer_panel.width() - 20, 200)
            h = max(self.viewer_panel.height() - 60, 200)
            self.image_label.setPixmap(
                pix.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.image_label.setText("Failed to load image")
        self.viewer_stack.setCurrentIndex(2)

    def _ensure_player(self):
        if self._player is None:
            self._player = QMediaPlayer()
            self._audio_output = QAudioOutput()
            self._player.setAudioOutput(self._audio_output)
            self._video_widget = QVideoWidget()
            self._video_widget.setStyleSheet("background: transparent; border: none;")
            self._media_layout.addWidget(self._video_widget)
            self._player.setVideoOutput(self._video_widget)
            self._player.positionChanged.connect(self._on_pos_changed)
            self._player.playbackStateChanged.connect(self._on_state_changed)

    def _view_video(self, path):
        self._ensure_player()
        self._player.setSource(QUrl.fromLocalFile(path))
        self._video_widget.show()
        self.viewer_stack.setCurrentIndex(3)
        self.media_controls.show()
        self._player.play()
        self._play_btn.setText("⏸")

    def _view_audio(self, path):
        self._ensure_player()
        self._player.setSource(QUrl.fromLocalFile(path))
        self._video_widget.hide()
        self.viewer_stack.setCurrentIndex(3)
        self.media_controls.show()
        self._player.play()
        self._play_btn.setText("⏸")

    def _view_pdf(self, path):
        if self._web_view is None:
            self._web_view = QWebEngineView()
            self._web_view.setStyleSheet("border: none;")
            self._web_layout.addWidget(self._web_view)
        self._web_view.setUrl(QUrl.fromLocalFile(path))
        self.viewer_stack.setCurrentIndex(4)

    def _show_unsupported(self, name, ext):
        self.unsupported_label.setText(
            f"🔒\n\nCannot preview\n{name}\n\n"
            f"File type: {ext if ext else 'unknown'}\n\n"
            "Double-click to open with\nyour system's default app."
        )
        self.viewer_stack.setCurrentIndex(5)

    # ── Media controls ───────────────────────────────────────────────────
    def _toggle_playback(self):
        if self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _stop_playback(self):
        self._stop_media()

    def _stop_media(self):
        if self._player and self._player.playbackState() != QMediaPlayer.StoppedState:
            self._player.stop()

    def _on_pos_changed(self, pos):
        s = pos // 1000
        m, s = divmod(s, 60)
        self._media_time.setText(f"{m}:{s:02d}")

    def _on_state_changed(self, state):
        self._play_btn.setText("⏸" if state == QMediaPlayer.PlayingState else "▶")

    # ── Utilities ────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_size(b):
        for u in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {u}"
            b /= 1024.0
        return f"{b:.1f} TB"

    def close_explorer(self):
        self._stop_media()
        self.hide()
        self.deleteLater()

    def show(self):
        super().show()
        self.raise_()

    def paintEvent(self, event):
        super().paintEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


# ── Launch ───────────────────────────────────────────────────────────────────
file_explorer = DraggableFileExplorer()

shadow = QGraphicsDropShadowEffect()
shadow.setBlurRadius(20)
shadow.setColor(QColor(0, 0, 0, 120))
shadow.setOffset(0, 4)
file_explorer.setGraphicsEffect(shadow)

file_explorer.show()
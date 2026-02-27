"""
Acme Column - manages multiple windows in a vertical layout

Plan 9 Positioning:
  Dragging a window handle sets its top edge at the drop position.
  The window above shrinks to make room, the dragged window keeps
  its height. No equal redistribution — you get manual control
  just like in Plan 9 acme.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QTextEdit, QWidget
from PySide6.QtCore import Qt, QEvent, Signal, QPoint
from PySide6.QtGui import QPalette, QColor, QTextCursor

from .acme_window import AcmeWindow

ACME_FONT_SIZE = 13
ACME_COL_TAG_BG = "#AACCDD"


class AcmeColumn(QFrame):
    """Column containing stacked windows"""

    close_requested = Signal(object)

    def __init__(self, parent=None, llmfs_mount="/n/mux/llm",
                 rio_mount="/n/mux/default",
                 p9_host="localhost", p9_port=5640):
        super().__init__(parent)
        self.llmfs_mount = llmfs_mount
        self.rio_mount = rio_mount
        self.p9_host = p9_host
        self.p9_port = p9_port
        self.windows = []
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 80);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tag_line = QTextEdit()
        self.tag_line.setStyleSheet(f"""
            QTextEdit {{
                background-color: {ACME_COL_TAG_BG};
                border: none;
                border-bottom: 1px solid #888888;
                color: black;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {ACME_FONT_SIZE}px;
                font-weight: bold;
                padding: 0px 2px;
            }}
        """)
        self.tag_line.document().setDocumentMargin(1)
        self.tag_line.setPlainText("New Delcol")
        self._fit_tag_height()
        self.tag_line.textChanged.connect(self._fit_tag_height)
        self.tag_line.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tag_line.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tag_line.setReadOnly(False)
        self.tag_line.setContextMenuPolicy(Qt.NoContextMenu)
        self.tag_line.setCursorWidth(2)
        self.tag_line.setLineWrapMode(QTextEdit.WidgetWidth)

        palette = self.tag_line.palette()
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        self.tag_line.setPalette(palette)

        self.container = QWidget()
        self.container.setStyleSheet("background-color: transparent;")

        layout.addWidget(self.tag_line)
        layout.addWidget(self.container, 1)

        self.tag_line.viewport().installEventFilter(self)

    def _fit_tag_height(self):
        doc_h = int(self.tag_line.document().size().height()) + 2
        h = max(20, min(doc_h, 120))
        if self.tag_line.maximumHeight() != h:
            self.tag_line.setFixedHeight(h)

    def get_acme_parent(self):
        parent = self.parent()
        while parent:
            from .acme_core import Acme
            if isinstance(parent, Acme):
                return parent
            parent = parent.parent() if hasattr(parent, 'parent') else None
        return None

    # ------------------------------------------------------------------
    # Add window — new windows get half the column (Plan 9 style)
    # ------------------------------------------------------------------

    def add_window(self, path=""):
        window = AcmeWindow(path, parent=self.container,
                            llmfs_mount=self.llmfs_mount,
                            rio_mount=self.rio_mount,
                            p9_host=self.p9_host,
                            p9_port=self.p9_port)
        window.close_requested.connect(self.remove_window)

        acme = self.get_acme_parent()
        if acme:
            acme.set_preferred_column(self)
            if acme._routes_manager:
                window.set_routes_manager(acme._routes_manager)

        cw = self.container.width()
        ch = self.container.height()

        if not self.windows:
            # First window — takes full column
            window.setGeometry(0, 0, cw, ch)
            self.windows.append((window, 0, ch))
        else:
            # New window takes the bottom half of the last window
            last_w, last_y, last_h = self.windows[-1]
            split = last_h // 2
            top_h = last_h - split
            last_w.setGeometry(0, last_y, cw, top_h)
            self.windows[-1] = (last_w, last_y, top_h)

            new_y = last_y + top_h
            new_h = ch - new_y
            window.setGeometry(0, new_y, cw, new_h)
            self.windows.append((window, new_y, new_h))

        window.show()
        return window

    def add_terminal(self, working_dir):
        """Add a terminal window (Plan 9 acme 'win' style)."""
        window = AcmeWindow("", parent=self.container,
                            llmfs_mount=self.llmfs_mount,
                            rio_mount=self.rio_mount,
                            p9_host=self.p9_host,
                            p9_port=self.p9_port)
        window.close_requested.connect(self.remove_window)

        acme = self.get_acme_parent()
        if acme:
            acme.set_preferred_column(self)
            if acme._routes_manager:
                window.set_routes_manager(acme._routes_manager)

        # Initialize terminal mode
        window.init_terminal(working_dir)

        cw = self.container.width()
        ch = self.container.height()

        if not self.windows:
            window.setGeometry(0, 0, cw, ch)
            self.windows.append((window, 0, ch))
        else:
            last_w, last_y, last_h = self.windows[-1]
            split = last_h // 2
            top_h = last_h - split
            last_w.setGeometry(0, last_y, cw, top_h)
            self.windows[-1] = (last_w, last_y, top_h)
            new_y = last_y + top_h
            new_h = ch - new_y
            window.setGeometry(0, new_y, cw, new_h)
            self.windows.append((window, new_y, new_h))

        window.show()
        return window

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove_window(self, window):
        for i, (w, _, _) in enumerate(self.windows):
            if w == window:
                self.windows.pop(i)
                if hasattr(window, 'cleanup'):
                    window.cleanup()
                from .acme_fs import get_acme_dir
                get_acme_dir().unregister_window(window.window_id)
                window.deleteLater()
                # After removal, let remaining windows fill the gap
                self._reflow_windows()
                break

    def remove_window_without_delete(self, window):
        for i, (w, _, _) in enumerate(self.windows):
            if w == window:
                self.windows.pop(i)
                self._reflow_windows()
                break

    def insert_window_at_position(self, window, y_position):
        """Insert a window at a specific y position (cross-column drag)."""
        window.setParent(self.container)

        if hasattr(window, 'close_requested'):
            try:
                window.close_requested.disconnect()
            except:
                pass
            window.close_requested.connect(self.remove_window)

        insert_index = len(self.windows)
        for i, (w, y, h) in enumerate(self.windows):
            if y_position < y + h // 2:
                insert_index = i
                break

        self.windows.insert(insert_index, (window, 0, 0))
        self._reflow_windows()
        window.show()

    # ------------------------------------------------------------------
    # Plan 9 window positioning
    # ------------------------------------------------------------------

    def _reflow_windows(self):
        """
        After removal or cross-column insert: redistribute evenly.
        (Only used for structural changes, NOT for drag finalization.)
        """
        if not self.windows:
            return
        cw = self.container.width()
        ch = self.container.height()
        n = len(self.windows)
        h = ch // n
        for i, (w, _, _) in enumerate(self.windows):
            y = i * h
            height = h if i < n - 1 else ch - y
            w.setGeometry(0, y, cw, height)
            self.windows[i] = (w, y, height)

    def reposition_window_during_drag(self, window, new_y):
        """
        Plan 9 drag: moving the handle sets the window's top edge.
        The window's bottom edge stays anchored (next window below
        keeps its position, or bottom of container). The window
        above shrinks/grows so its bottom meets the new top edge.
        The dragged window itself resizes to fill from new_y to
        the start of the next window (or container bottom).
        """
        idx = None
        for i, (w, _, _) in enumerate(self.windows):
            if w == window:
                idx = i
                break
        if idx is None:
            return

        cw = self.container.width()
        ch = self.container.height()

        # Compute the fixed bottom edge: where the next window starts,
        # or the container bottom if this is the last window
        if idx + 1 < len(self.windows):
            _, next_y, _ = self.windows[idx + 1]
            bottom_edge = next_y
        else:
            bottom_edge = ch

        # Clamp new_y
        min_y = 20 * idx if idx > 0 else 0
        max_y = bottom_edge - 20  # leave at least 20px for the dragged window
        new_y = max(min_y, min(new_y, max_y))

        # Dragged window: top at new_y, bottom at bottom_edge
        drag_h = bottom_edge - new_y
        window.setGeometry(0, new_y, cw, drag_h)
        self.windows[idx] = (window, new_y, drag_h)

        # Windows above: distribute [0, new_y] proportionally
        above = self.windows[:idx]
        if above:
            h_each = max(20, new_y // len(above))
            y = 0
            for i, (w, _, _) in enumerate(above):
                h = h_each if i < len(above) - 1 else (new_y - y)
                h = max(20, h)
                w.setGeometry(0, y, cw, h)
                self.windows[i] = (w, y, h)
                y += h

    def finalize_window_position(self, window):
        """
        After drag release: keep the current positions as-is.
        (Plan 9 style — no equal redistribution.)
        """
        pass  # Positions are already set during drag

    def resize_window(self, window, new_y, new_height):
        for i, (w, y, h) in enumerate(self.windows):
            if w == window:
                window.setGeometry(0, new_y, self.container.width(), new_height)
                self.windows[i] = (window, new_y, new_height)
                break

    def resizeEvent(self, event):
        """On column resize, scale all windows proportionally."""
        super().resizeEvent(event)
        if not self.windows:
            return
        old_total = sum(h for (_, _, h) in self.windows)
        if old_total <= 0:
            self._reflow_windows()
            return
        cw = self.container.width()
        ch = self.container.height()
        y = 0
        for i, (w, _, old_h) in enumerate(self.windows):
            if i < len(self.windows) - 1:
                h = max(20, int(old_h * ch / old_total))
            else:
                h = ch - y
            w.setGeometry(0, y, cw, max(20, h))
            self.windows[i] = (w, y, max(20, h))
            y += max(20, h)

    # ------------------------------------------------------------------
    # Event filter — mid/right colored selection on column tag line
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj == self.tag_line.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
                # Red selection — execute on release
                palette = self.tag_line.palette()
                palette.setColor(QPalette.Highlight, QColor(0xFF, 0x88, 0x88))
                palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
                self.tag_line.setPalette(palette)

                cursor = self.tag_line.cursorForPosition(event.pos())
                if not self.tag_line.textCursor().hasSelection():
                    cursor.select(QTextCursor.WordUnderCursor)
                self.tag_line.setTextCursor(cursor)
                return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.MiddleButton:
                command = self.tag_line.textCursor().selectedText().strip()
                # Reset color
                palette = self.tag_line.palette()
                palette.setColor(QPalette.Highlight, QColor(0x99, 0xCC, 0xFF))
                palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
                self.tag_line.setPalette(palette)

                if command == "New":
                    self.add_window()
                elif command == "Delcol":
                    self.close_requested.emit(self)
                return True

            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                # Green selection — search on release
                palette = self.tag_line.palette()
                palette.setColor(QPalette.Highlight, QColor(0x88, 0xEE, 0x88))
                palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
                self.tag_line.setPalette(palette)

                cursor = self.tag_line.cursorForPosition(event.pos())
                if not self.tag_line.textCursor().hasSelection():
                    cursor.select(QTextCursor.WordUnderCursor)
                self.tag_line.setTextCursor(cursor)
                return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                word = self.tag_line.textCursor().selectedText().strip()
                palette = self.tag_line.palette()
                palette.setColor(QPalette.Highlight, QColor(0x99, 0xCC, 0xFF))
                palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
                self.tag_line.setPalette(palette)
                # TODO: search in focused window
                return True

            elif event.type() == QEvent.MouseButtonPress:
                self.tag_line.setFocus()
                cursor = self.tag_line.cursorForPosition(event.pos())
                self.tag_line.setTextCursor(cursor)
                return False

        if event.type() == QEvent.ContextMenu:
            return True

        return super().eventFilter(obj, event)
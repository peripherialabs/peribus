"""
Core Acme interface - main window with columns
"""

from PySide6.QtWidgets import (QFrame, QVBoxLayout, QTextEdit, QSplitter,
                               QApplication)
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QPalette, QColor, QTextCursor

from .acme_column import AcmeColumn
from .acme_fs import get_acme_dir
from pathlib import Path

ACME_FONT_SIZE = 13
ACME_TAG_BG = "#EAEACC"       # Plan 9 yellow-ish tag background
ACME_BODY_BG = "#FFFFEA"      # Plan 9 body background
ACME_COL_TAG_BG = "#AACCDD"   # Column tag background
ACME_MAIN_TAG_BG = "#333333"  # Main tag background


class Acme(QFrame):
    """Main Acme interface with columns"""

    def __init__(self, parent=None, llmfs_mount="/n/mux/llm",
                 rio_mount="/n/mux/default",
                 p9_host="localhost", p9_port=5640):
        super().__init__(parent)

        self.llmfs_mount = llmfs_mount
        self.rio_mount = rio_mount
        self.p9_host = p9_host
        self.p9_port = p9_port
        self._routes_manager = None

        self.acme_dir = get_acme_dir()
        self.acme_dir.set_acme(self)

        self.columns = []
        self.preferred_column = None

        self.setup_ui()
        self.add_column()
        self.add_column(initial_path=".")

    def set_routes_manager(self, manager):
        self._routes_manager = manager
        for column in self.columns:
            for (window, _, _) in column.windows:
                window.set_routes_manager(manager)

    def setup_ui(self):
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("background-color: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.main_tag = QTextEdit()
        self.main_tag.setStyleSheet(f"""
            QTextEdit {{
                background-color: {ACME_MAIN_TAG_BG};
                border: none;
                border-bottom: 1px solid #000000;
                color: white;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {ACME_FONT_SIZE}px;
                font-weight: bold;
                padding: 0px 2px;
            }}
        """)
        self.main_tag.document().setDocumentMargin(1)
        self.main_tag.setPlainText("Newcol Exit Term")
        # Set initial height AFTER content so doc height is computed
        self._fit_tag_height(self.main_tag)
        self.main_tag.textChanged.connect(lambda: self._fit_tag_height(self.main_tag))
        self.main_tag.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.main_tag.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.main_tag.setReadOnly(False)
        self.main_tag.setContextMenuPolicy(Qt.NoContextMenu)
        self.main_tag.setCursorWidth(2)
        self.main_tag.setLineWrapMode(QTextEdit.WidgetWidth)

        palette = self.main_tag.palette()
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        self.main_tag.setPalette(palette)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #000000;
                width: 1px;
            }
        """)
        self.splitter.setHandleWidth(1)

        layout.addWidget(self.main_tag)
        layout.addWidget(self.splitter, 1)

        self.main_tag.viewport().installEventFilter(self)

    @staticmethod
    def _fit_tag_height(tag_edit):
        """Fit a tag QTextEdit height to its document content."""
        doc_h = int(tag_edit.document().size().height()) + 2
        h = max(20, min(doc_h, 120))
        if tag_edit.maximumHeight() != h or tag_edit.minimumHeight() != h:
            tag_edit.setFixedHeight(h)

    def add_column(self, initial_path=""):
        column = AcmeColumn(parent=self,
                            llmfs_mount=self.llmfs_mount,
                            rio_mount=self.rio_mount,
                            p9_host=self.p9_host,
                            p9_port=self.p9_port)
        column.close_requested.connect(self.remove_column)

        self.splitter.addWidget(column)
        self.columns.append(column)
        self.preferred_column = column

        if initial_path:
            column.add_window(initial_path)

        if len(self.columns) > 1:
            sizes = [1000 // len(self.columns)] * len(self.columns)
            self.splitter.setSizes(sizes)

    def remove_column(self, column):
        if len(self.columns) > 1 and column in self.columns:
            self.columns.remove(column)
            column.deleteLater()
            if self.preferred_column == column:
                self.preferred_column = self.columns[0] if self.columns else None
        else:
            while column.windows:
                column.remove_window(column.windows[0][0])

    def set_preferred_column(self, column):
        if column in self.columns:
            self.preferred_column = column

    def eventFilter(self, obj, event):
        if not hasattr(self, 'main_tag') or self.main_tag is None:
            return super().eventFilter(obj, event)

        if obj == self.main_tag.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
                cursor = self.main_tag.cursorForPosition(event.pos())
                if not self.main_tag.textCursor().hasSelection():
                    cursor.select(QTextCursor.WordUnderCursor)
                    self.main_tag.setTextCursor(cursor)

                command = self.main_tag.textCursor().selectedText().strip()

                if command == "Newcol":
                    self.add_column()
                elif command == "Exit":
                    QApplication.quit()
                elif command == "Term":
                    if self.preferred_column:
                        import os
                        working_dir = os.getcwd().replace('\\', '/')
                        self.preferred_column.add_terminal(working_dir)

                return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.MiddleButton:
                return True

            elif event.type() == QEvent.MouseButtonPress:
                self.main_tag.setFocus()
                cursor = self.main_tag.cursorForPosition(event.pos())
                self.main_tag.setTextCursor(cursor)
                return False

        if event.type() == QEvent.ContextMenu:
            return True

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        for column in self.columns:
            for (window, _, _) in column.windows:
                window.cleanup()
                self.acme_dir.unregister_window(window.window_id)
        super().closeEvent(event)
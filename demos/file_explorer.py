from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeView, 
                              QListView, QFileSystemModel, QSplitter, QLabel,
                              QToolBar, QLineEdit, QFileDialog, QGraphicsDropShadowEffect,
                              QTextEdit, QScrollArea)
from PySide6.QtCore import QDir, Qt, Signal, QSize, QFileInfo, QPoint
from PySide6.QtGui import QIcon, QColor, QMouseEvent, QPixmap
import os

# Delete existing file explorer if it exists
try:
    file_explorer.deleteLater()
except:
    pass

class DraggableFileExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.setParent(main_window)
        self.setGeometry(141, 182, 1200, 600)  # Made wider to accommodate the viewer panel
        
        # Mouse dragging variables
        self.drag_position = QPoint()
        self.is_dragging = False
    
        
        # Apply styling
        self.setStyleSheet("""
            background-color: rgba(255, 255, 255, 90);
            color: black;
            border: 1px solid black;
            border-radius: 15px;
        """)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create toolbar
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: rgba(240, 240, 240, 80);
                border: none;
                border-radius: 10px;
                padding: 5px;
            }
            QToolButton {
                background-color: rgba(220, 220, 220, 80);
                border: 1px solid rgba(180, 180, 180, 200);
                border-radius: 5px;
                padding: 3px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: rgba(200, 200, 200, 200);
            }
        """)
        
        
        # Navigation buttons
        self.back_btn = self.toolbar.addAction(QIcon.fromTheme("go-previous"), "Back")
        self.forward_btn = self.toolbar.addAction(QIcon.fromTheme("go-next"), "Forward")
        self.up_btn = self.toolbar.addAction(QIcon.fromTheme("go-up"), "Up")
        self.home_btn = self.toolbar.addAction(QIcon.fromTheme("go-home"), "Home")
        
        # Path display
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Enter path or click to browse...")
        self.path_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 0);
                border: 1px solid rgba(150, 150, 150, 150);
                border-radius: 8px;
                padding: 5px;
                margin: 2px;
            }
        """)
        
        
        self.toolbar.addWidget(self.path_edit)
        
        # Browse button
        self.browse_btn = self.toolbar.addAction(QIcon.fromTheme("folder-open"), "Browse")
        
        # Add separator for visual separation
        self.toolbar.addSeparator()
        
        # Close button
        self.close_btn = self.toolbar.addAction(QIcon.fromTheme("window-close"), "Close")
        # Style the close button with a distinctive color
        close_action = self.toolbar.actions()[-1]  # Get the last action (close button)
        close_widget = self.toolbar.widgetForAction(close_action)
        if close_widget:
            close_widget.setStyleSheet("""
                QToolButton {
                    background-color: rgba(255, 100, 100, 100);
                    border: 1px solid rgba(200, 50, 50, 200);
                    border-radius: 5px;
                    padding: 3px;
                    margin: 2px;
                }
                QToolButton:hover {
                    background-color: rgba(255, 150, 150, 200);
                }
            """)
        
        layout.addWidget(self.toolbar)
        
        # Create main horizontal splitter for file browser and viewer
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setStyleSheet("""
            QSplitter {
                background-color: transparent;
            }
            QSplitter::handle {
                background-color: rgba(180, 180, 180, 150);
                border-radius: 3px;
            }
        """)
        
        # Create file browser splitter (tree and list views)
        self.file_splitter = QSplitter(Qt.Horizontal)
        self.file_splitter.setStyleSheet("""
            QSplitter {
                background-color: transparent;
            }
            QSplitter::handle {
                background-color: rgba(180, 180, 180, 150);
                border-radius: 3px;
            }
        """)
        
        # Tree view (directory structure)
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setStyleSheet("""
            QTreeView {
                background-color: rgba(255, 255, 255, 0);
                border: 1px solid rgba(200, 200, 200, 150);
                border-radius: 10px;
                padding: 5px;
            }
            QTreeView::item {
                padding: 3px;
                border-radius: 5px;
            }
            QTreeView::item:hover {
                background-color: rgba(220, 220, 220, 150);
            }
            QTreeView::item:selected {
                background-color: rgba(100, 150, 200, 150);
            }
        """)
        
        
        # List view (file contents)
        self.list_view = QListView()
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setGridSize(QSize(100, 100))
        self.list_view.setStyleSheet("""
            QListView {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(200, 200, 200, 150);
                border-radius: 10px;
                padding: 5px;
            }
            QListView::item {
                padding: 5px;
                border-radius: 8px;
                margin: 2px;
            }
            QListView::item:hover {
                background-color: rgba(220, 220, 220, 150);
            }
            QListView::item:selected {
                background-color: rgba(100, 150, 200, 150);
            }
        """)
        
        
        self.file_splitter.addWidget(self.tree_view)
        self.file_splitter.addWidget(self.list_view)
        self.file_splitter.setSizes([200, 400])
        
        # Create file viewer panel
        self.viewer_panel = QWidget()
        self.viewer_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0);
                border: 1px solid rgba(200, 200, 200, 150);
                border-radius: 10px;
                padding: 5px;
            }
        """)
        

        
        # Viewer layout
        viewer_layout = QVBoxLayout(self.viewer_panel)
        viewer_layout.setContentsMargins(10, 10, 10, 10)
        
        # File info label
        self.file_info_label = QLabel("Select a file to view")
        self.file_info_label.setAlignment(Qt.AlignCenter)
        self.file_info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(240, 240, 240, 0);
                border: 1px solid rgba(200, 200, 200, 100);
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        viewer_layout.addWidget(self.file_info_label)
        
        # Text editor for text files
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.text_viewer.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255, 255, 255, 0);
                border: 1px solid rgba(180, 180, 180, 150);
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.text_viewer.hide()
        viewer_layout.addWidget(self.text_viewer)
        
        # Image viewer with scroll area
        self.image_scroll = QScrollArea()
        self.image_scroll.setStyleSheet("""
            QScrollArea {
                background-color: rgba(255, 255, 255, 0);
                border: 1px solid rgba(180, 180, 180, 150);
                border-radius: 8px;
            }
        """)
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
                padding: 10px;
            }
        """)
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.hide()
        viewer_layout.addWidget(self.image_scroll)
        
        # Add to main splitter
        self.main_splitter.addWidget(self.file_splitter)
        self.main_splitter.addWidget(self.viewer_panel)
        self.main_splitter.setSizes([700, 400])
        
        layout.addWidget(self.main_splitter)
        
        # Status bar - reduced size
        self.status_bar = QLabel()
        self.status_bar.setAlignment(Qt.AlignRight)
        self.status_bar.setMaximumHeight(25)
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: rgba(240, 240, 240, 150);
                border: 1px solid rgba(200, 200, 200, 100);
                border-radius: 8px;
                padding: 3px 8px;
                font-size: 12px;
            }
        """)
        
        
        layout.addWidget(self.status_bar)
        
        # File system models
        self.dir_model = QFileSystemModel()
        self.dir_model.setRootPath("")
        self.dir_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        self.file_model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        
        # Set models to views
        self.tree_view.setModel(self.dir_model)
        self.list_view.setModel(self.file_model)
        
        # Connect signals
        self.tree_view.clicked.connect(self.on_tree_clicked)
        self.list_view.doubleClicked.connect(self.on_list_double_clicked)
        self.path_edit.returnPressed.connect(self.on_path_entered)
        self.back_btn.triggered.connect(self.go_back)
        self.forward_btn.triggered.connect(self.go_forward)
        self.up_btn.triggered.connect(self.go_up)
        self.home_btn.triggered.connect(self.go_home)
        self.browse_btn.triggered.connect(self.browse_directory)
        self.close_btn.triggered.connect(self.close_explorer)  # Connect close button
        
        # Navigation history
        self.history = []
        self.history_index = -1
        
        # Supported file types
        self.text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.md', '.cpp', '.c', '.h', '.java', '.php', '.rb', '.go', '.rs', '.sql', '.yml', '.yaml', '.ini', '.cfg', '.conf', '.log'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg', '.ico'}
        
        # Set initial directory to home
        self.go_home()
    
    def close_explorer(self):
        """Close the file explorer"""
        print("Closing file explorer...")
        self.hide()
        self.deleteLater()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Check if click is on the toolbar area for dragging
            toolbar_rect = self.toolbar.geometry()
            if toolbar_rect.contains(event.position().toPoint()):
                self.is_dragging = True
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton and self.is_dragging:
            new_pos = event.globalPosition().toPoint() - self.drag_position
            
            # Keep the widget within the main window bounds
            if hasattr(main_window, 'geometry'):
                main_rect = main_window.geometry()
                widget_rect = self.geometry()
                
                # Constrain X position
                if new_pos.x() < 0:
                    new_pos.setX(0)
                elif new_pos.x() + widget_rect.width() > main_rect.width():
                    new_pos.setX(main_rect.width() - widget_rect.width())
                
                # Constrain Y position
                if new_pos.y() < 0:
                    new_pos.setY(0)
                elif new_pos.y() + widget_rect.height() > main_rect.height():
                    new_pos.setY(main_rect.height() - widget_rect.height())
            
            self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        
    def on_tree_clicked(self, index):
        path = self.dir_model.filePath(index)
        self.set_current_path(path)
        
    def on_list_double_clicked(self, index):
        path = self.file_model.filePath(index)
        file_info = QFileInfo(path)
        
        if file_info.isDir():
            self.set_current_path(path)
        else:
            # Handle file viewing
            self.view_file(path)
    
    def view_file(self, file_path):
        """View file content based on file type"""
        file_info = QFileInfo(file_path)
        file_name = file_info.fileName()
        file_size = file_info.size()
        extension = file_info.suffix().lower()
        
        # Update file info
        self.file_info_label.setText(f"File: {file_name} | Size: {self.format_file_size(file_size)}")
        
        # Hide both viewers initially
        self.text_viewer.hide()
        self.image_scroll.hide()
        
        if f'.{extension}' in self.text_extensions:
            self.view_text_file(file_path)
        elif f'.{extension}' in self.image_extensions:
            self.view_image_file(file_path)
        else:
            # For unsupported files, try to read as text (limited to reasonable size)
            if file_size < 1024 * 1024:  # Less than 1MB
                try:
                    self.view_text_file(file_path)
                except:
                    self.show_unsupported_file(file_name, extension)
            else:
                self.show_unsupported_file(file_name, extension)
    
    def view_text_file(self, file_path):
        """Display text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                self.text_viewer.setPlainText(content)
                self.text_viewer.show()
                print(f"Text file content:\n{content}")
        except Exception as e:
            self.text_viewer.setPlainText(f"Error reading file: {str(e)}")
            self.text_viewer.show()
            print(f"Error reading file {file_path}: {str(e)}")
    
    def view_image_file(self, file_path):
        """Display image file"""
        try:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale image to fit the viewer while maintaining aspect ratio
                max_size = QSize(400, 400)
                scaled_pixmap = pixmap.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.image_scroll.show()
                print(f"Displaying image: {file_path}")
            else:
                self.image_label.setText("Failed to load image")
                self.image_scroll.show()
                print(f"Failed to load image: {file_path}")
        except Exception as e:
            self.image_label.setText(f"Error loading image: {str(e)}")
            self.image_scroll.show()
            print(f"Error loading image {file_path}: {str(e)}")
    
    def show_unsupported_file(self, file_name, extension):
        """Show message for unsupported file types"""
        message = f"Unsupported file type: .{extension}\nFile: {file_name}"
        self.text_viewer.setPlainText(message)
        self.text_viewer.show()
        print(f"Unsupported file type: {file_name}")
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"
        
    def on_path_entered(self):
        path = self.path_edit.text()
        if QDir(path).exists():
            self.set_current_path(path)
        else:
            self.status_bar.setText("Invalid path")
        
    def set_current_path(self, path):
        # Update path edit
        self.path_edit.setText(path)
        
        # Update list view
        self.list_view.setRootIndex(self.file_model.setRootPath(path))
        
        # Update tree view if needed
        tree_index = self.dir_model.index(path)
        if tree_index.isValid():
            self.tree_view.setCurrentIndex(tree_index)
            self.tree_view.expand(tree_index)
        
        # Clear viewer when changing directories
        self.file_info_label.setText("Select a file to view")
        self.text_viewer.hide()
        self.image_scroll.hide()
        
        # Update status
        try:
            item_count = len(os.listdir(path))
            self.status_bar.setText(f"{item_count} items")
        except:
            self.status_bar.setText("Access denied")
        
        # Add to history
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(path)
        self.history_index = len(self.history) - 1
        
    def go_back(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.set_current_path(self.history[self.history_index])
        
    def go_forward(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.set_current_path(self.history[self.history_index])
        
    def go_up(self):
        current = QDir(self.path_edit.text())
        if current.cdUp():
            self.set_current_path(current.path())
        
    def go_home(self):
        self.set_current_path(QDir.homePath())
        
    def browse_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory", self.path_edit.text())
        if path:
            self.set_current_path(path)

# Create and show the file explorer
file_explorer = DraggableFileExplorer()
shadow_effect = QGraphicsDropShadowEffect()
shadow_effect.setBlurRadius(20)
shadow_effect.setColor(QColor(0, 0, 0, 160))
shadow_effect.setOffset(45, 45)
file_explorer.setGraphicsEffect(shadow_effect)
file_explorer.show()
#file_explorer.update()
#file_explorer.repaint()
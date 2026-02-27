"""
Program generators - create executable code for different content types

Generated code runs on a free canvas (container QWidget with no layout).
Widgets are placed with setParent(container) + setGeometry().
"""

import os
from pathlib import Path


def generate_message_display(message, title=""):
    """Generate a program that displays a message (error, info, etc.)
    styled like the directory listing / file content viewers —
    transparent background QTextEdit in the graphical pane."""
    # Escape for Python string embedding
    message_escaped = message.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    title_escaped = (title or "").replace('\\', '\\\\').replace("'", "\\'")

    return f'''# {title_escaped or "Message"}
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
text_edit.setPlainText('{message_escaped}')
text_edit.setParent(container)
text_edit.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=text_edit, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize
container.text_edit = text_edit
'''


def generate_directory_listing(path):
    """Generate a program that displays a directory listing"""
    # Keep absolute paths as-is (important for 9P mounts at /n/)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    # Escape backslashes for Windows paths
    path_escaped = path.replace('\\', '\\\\')
    
    # Return as executable code that reads and displays the listing
    return f'''# Directory listing: {path}
import os
import concurrent.futures
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt

def format_directory_columns(entries, width=80):
    """Format directory entries in columns"""
    if not entries:
        return ""
    
    max_entry_len = max(len(entry) for entry in entries)
    col_width = max_entry_len + 2
    num_cols = max(1, width // col_width)
    num_rows = (len(entries) + num_cols - 1) // num_cols
    
    lines = []
    for row in range(num_rows):
        line_parts = []
        for col in range(num_cols):
            idx = row + col * num_rows
            if idx < len(entries):
                entry = entries[idx]
                if col < num_cols - 1:
                    line_parts.append(entry.ljust(col_width))
                else:
                    line_parts.append(entry)
        lines.append("".join(line_parts).rstrip())
    
    return "\\n".join(lines)

def _is_dir_robust(p, is_9p=False):
    """Check if path is a directory, with fallback for 9P mounts.
    For 9P paths, uses timeout protection to avoid freezing on
    blocking synthetic files (StreamFile, SupplementaryOutputFile, etc.)."""
    try:
        if os.path.isdir(p):
            return True
    except (OSError, PermissionError):
        pass
    
    # For 9P paths, run listdir in a thread with timeout.
    # Any file under /n/ could be a blocking synthetic file —
    # we can't maintain a static blocklist because supplementary
    # outputs have arbitrary user-defined names.
    if is_9p:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(os.listdir, p)
                future.result(timeout=0.3)
                return True
        except concurrent.futures.TimeoutError:
            return False
        except (NotADirectoryError, FileNotFoundError, PermissionError, OSError):
            return False
    
    try:
        os.listdir(p)
        return True
    except (NotADirectoryError, FileNotFoundError, PermissionError, OSError):
        return False

# Read directory contents
try:
    dir_path = r"{path_escaped}"
    is_9p = dir_path.startswith('/n/')
    entries = sorted(os.listdir(dir_path))
    all_entries = []
    
    # Add parent directory if not at root
    parent = os.path.dirname(dir_path)
    if dir_path != parent and parent:
        all_entries.append("../")
    
    for entry in entries:
        full_path = os.path.join(dir_path, entry)
        if _is_dir_robust(full_path, is_9p):
            all_entries.append(entry + "/")
        else:
            all_entries.append(entry)
    
    content = format_directory_columns(all_entries)
except Exception as e:
    content = f"Error reading directory: {{e}}"

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

text_edit.setPlainText(content)

# Free canvas: fill the container
text_edit.setParent(container)
text_edit.setGeometry(0, 0, container.width() or 600, container.height() or 400)

# Auto-resize with parent
def _resize(ev, w=text_edit, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize

# Store reference for event handling
container.text_edit = text_edit
'''


def format_directory_columns(entries, width=80):
    """Format directory entries in columns"""
    if not entries:
        return ""
    
    # Find the longest entry
    max_entry_len = max(len(entry) for entry in entries)
    
    # Calculate column width (add 2 for spacing)
    col_width = max_entry_len + 2
    
    # Calculate number of columns that fit
    num_cols = max(1, width // col_width)
    
    # Calculate number of rows needed
    num_rows = (len(entries) + num_cols - 1) // num_cols
    
    # Build the output line by line
    lines = []
    for row in range(num_rows):
        line_parts = []
        for col in range(num_cols):
            idx = row + col * num_rows
            if idx < len(entries):
                entry = entries[idx]
                # Pad to column width (except last column)
                if col < num_cols - 1:
                    line_parts.append(entry.ljust(col_width))
                else:
                    line_parts.append(entry)
        lines.append("".join(line_parts).rstrip())
    
    return "\n".join(lines)


def generate_file_content(path):
    """Generate a program that displays file content"""
    # Keep absolute paths as-is (important for 9P mounts at /n/)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')
    
    return f'''# File: {path}
import os
import concurrent.futures
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt

# Read file content
file_path = r"{path_escaped}"
try:
    def _read_file(p):
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    # For 9P paths, use timeout to avoid blocking on synthetic files
    # (StreamFile, SupplementaryOutputFile, etc. block indefinitely)
    if file_path.startswith('/n/'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_read_file, file_path)
            content = future.result(timeout=2.0)
    else:
        content = _read_file(file_path)
except concurrent.futures.TimeoutError:
    content = "(file read timed out — likely a blocking synthetic file)"
except Exception as e:
    content = f"Error reading file: {{e}}"

text_edit = QTextEdit()
text_edit.setReadOnly(False)  # Allow editing
text_edit.setStyleSheet("""
    QTextEdit {{
        background-color: rgba(255, 255, 255, 80);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        border: none;
        color: black;
    }}
""")

text_edit.setPlainText(content)

# Free canvas: fill the container
text_edit.setParent(container)
text_edit.setGeometry(0, 0, container.width() or 600, container.height() or 400)

# Auto-resize with parent
def _resize(ev, w=text_edit, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize

# Store reference for saving
container.text_edit = text_edit
container.file_path = r"{path_escaped}"
'''


def generate_image_viewer(path):
    """Generate a program that displays an image"""
    # Keep absolute paths as-is (important for 9P mounts at /n/)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')
    
    return f'''# Image Viewer
# Path: {path}

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap, QPainter, QColor, QCursor

class ImageViewWidget(QWidget):
    """Widget for viewing images with zoom and pan"""
    
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.pixmap = QPixmap(self.image_path)
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.last_mouse_pos = None
        self.setMinimumSize(200, 200)
        
        if self.pixmap.isNull():
            self.pixmap = None
    
    def paintEvent(self, event):
        if not self.pixmap:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Failed to load image")
            return
        
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        
        scaled_size = self.pixmap.size() * self.scale
        x = (self.width() - scaled_size.width()) / 2 + self.offset.x()
        y = (self.height() - scaled_size.height()) / 2 + self.offset.y()
        
        from PySide6.QtCore import QRectF
        target_rect = QRectF(x, y, scaled_size.width(), scaled_size.height())
        painter.drawPixmap(target_rect, self.pixmap, QRectF(self.pixmap.rect()))
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.scale *= zoom_factor
        self.scale = max(0.1, min(10.0, self.scale))
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap and self.scale == 1.0:
            scale_w = self.width() / self.pixmap.width()
            scale_h = self.height() / self.pixmap.height()
            self.scale = min(scale_w, scale_h, 1.0)
        self.update()

# Create and add the image viewer
viewer = ImageViewWidget(r"{path_escaped}")
viewer.setParent(container)
viewer.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=viewer, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize
'''


def generate_video_player(path):
    """Generate a program that plays video"""
    # Keep absolute paths as-is (important for 9P mounts at /n/)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')
    
    return f'''# Video Player
# Path: {path}

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl

video_widget = QVideoWidget()
video_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

media_player = QMediaPlayer()
audio_output = QAudioOutput()
media_player.setAudioOutput(audio_output)
media_player.setVideoOutput(video_widget)
media_player.setSource(QUrl.fromLocalFile(r"{path_escaped}"))

video_widget.setParent(container)
video_widget.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=video_widget, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize

# Store references to prevent garbage collection
container.media_player = media_player
container.audio_output = audio_output
'''


def generate_audio_player(path):
    """Generate a program that plays audio"""
    # Keep absolute paths as-is (important for 9P mounts at /n/)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')
    basename = os.path.basename(path)
    
    return f'''# Audio Player
# Path: {path}

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import QUrl, Qt

media_player = QMediaPlayer()
audio_output = QAudioOutput()
media_player.setAudioOutput(audio_output)
media_player.setSource(QUrl.fromLocalFile(r"{path_escaped}"))

info_widget = QLabel("Audio: {basename}\\n\\nUse Play/Pause/Stop commands")
info_widget.setAlignment(Qt.AlignCenter)
info_widget.setStyleSheet("""
    QLabel {{
        background-color: rgba(240, 240, 240, 0);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        padding: 20px;
    }}
""")

info_widget.setParent(container)
info_widget.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=info_widget, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize

# Store references to prevent garbage collection
container.media_player = media_player
container.audio_output = audio_output
'''


def generate_3d_viewer(path):
    """Generate a program that displays 3D models"""
    # Keep absolute paths as-is (important for 9P mounts at /n/)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')
    
    return f'''# 3D Model Viewer
# Path: {path}

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh
import numpy as np

try:
    from OpenGL.arrays import vbo
except:
    vbo = None

class Mesh3DWidget(QOpenGLWidget):
    """Widget for viewing 3D meshes with mouse rotation"""
    
    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.mesh = None
        self.vertices = None
        self.edges = None
        self.vbo_vertices = None
        self.vbo_edges = None
        
        self.rot_x = 20
        self.rot_y = 0
        self.last_mouse_pos = None
        self.zoom = 3.0
        
        # Set widget attributes to help with painting issues
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_NoSystemBackground, False)
        self.setAutoFillBackground(False)
        
        # Force updates to propagate
        self.setUpdateBehavior(QOpenGLWidget.PartialUpdate)
        
        self.load_mesh()
    
    def load_mesh(self):
        try:
            mesh = trimesh.load(self.model_path, force='mesh')
            
            target_faces = 2000
            current_faces = len(mesh.faces)
            
            if current_faces > target_faces:
                ratio = target_faces / current_faces
                try:
                    self.mesh = mesh.simplify_quadric_decimation(ratio)
                except:
                    self.mesh = mesh
            else:
                self.mesh = mesh
            
            self.edges = self.mesh.edges_unique
            self.vertices = self.mesh.vertices.astype(np.float32)
            
            center = self.vertices.mean(axis=0)
            self.vertices -= center
            max_dist = np.max(np.linalg.norm(self.vertices, axis=1))
            if max_dist > 0:
                self.vertices /= max_dist
        except Exception as e:
            print(f"Error loading 3D mesh: {{e}}")
            self.mesh = None
    
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        
        if self.vertices is not None and vbo:
            try:
                self.vbo_vertices = vbo.VBO(self.vertices)
                self.vbo_edges = vbo.VBO(self.edges.astype(np.uint32), target=GL_ELEMENT_ARRAY_BUFFER)
            except:
                pass
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self.mesh is None or self.vertices is None:
            return
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width() / max(self.height(), 1)
        gluPerspective(45, aspect, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, self.zoom, 0, 0, 0, 0, 1, 0)
        
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        
        if self.vbo_vertices and self.vbo_edges:
            self.vbo_vertices.bind()
            self.vbo_edges.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glDrawElements(GL_LINES, len(self.edges) * 2, GL_UNSIGNED_INT, None)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            glBegin(GL_LINES)
            for edge in self.edges:
                v1 = self.vertices[edge[0]]
                v2 = self.vertices[edge[1]]
                glVertex3fv(v1)
                glVertex3fv(v2)
            glEnd()
        
        # Force flush to ensure rendering completes
        glFlush()
    
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        # Force repaint after resize
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            
            self.rot_y += delta.x() * 0.5
            self.rot_x += delta.y() * 0.5
            
            # Force immediate repaint
            self.update()
            # Also notify parent to update
            if self.parent():
                self.parent().update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.zoom *= zoom_factor
        self.zoom = max(0.5, min(20.0, self.zoom))
        
        # Force immediate repaint
        self.update()
        # Also notify parent to update
        if self.parent():
            self.parent().update()

# Create and add the 3D viewer
viewer = Mesh3DWidget(r"{path_escaped}")
viewer.setParent(container)
viewer.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=viewer, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize
'''


def generate_pdf_viewer(path):
    """Generate a program that displays PDFs"""
    # Keep absolute paths as-is (important for 9P mounts at /n/)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')
    
    return f'''# PDF Viewer
# Path: {path}
import fitz  # PyMuPDF
from PySide6.QtWidgets import QScrollArea, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# Create scroll area
scroll = QScrollArea()
scroll.setWidgetResizable(True)

# Inner container widget for pages (not the outer canvas container)
pages_container = QWidget()
pages_layout = QVBoxLayout(pages_container)
pages_layout.setSpacing(10)

# Open and render PDF
try:
    doc = fitz.open(r"{path_escaped}")
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 2x resolution for better quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # Convert to QImage
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        
        # Display in label
        label = QLabel()
        label.setPixmap(QPixmap.fromImage(img))
        label.setAlignment(Qt.AlignCenter)
        pages_layout.addWidget(label)
    
    doc.close()
except Exception as e:
    error_label = QLabel(f"Error loading PDF: {{str(e)}}")
    error_label.setAlignment(Qt.AlignCenter)
    pages_layout.addWidget(error_label)

scroll.setWidget(pages_container)

# Free canvas: fill parent
scroll.setParent(container)
scroll.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=scroll, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize
'''


def generate_terminal(working_dir=None):
    """Generate a Plan9 Acme-style terminal program"""
    if working_dir is None:
        working_dir = os.getcwd()
    
    # Normalize path for cross-platform
    working_dir = os.path.abspath(working_dir)
    
    # Use regular string concatenation to avoid escaping issues
    code = """# Terminal
# Working directory: """ + working_dir + """

import os
import subprocess
import threading
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QTextCursor

class TerminalSignals(QObject):
    \"\"\"Signals for thread-safe communication\"\"\"
    output_ready = Signal(str)

class AcmeTerminal(QTextEdit):
    \"\"\"Plan9 Acme-style terminal\"\"\"
    
    def __init__(self, working_dir=r\"""" + working_dir + """\"):
        super().__init__()
        self.working_dir = os.path.abspath(working_dir)
        self.setReadOnly(False)
        
        # Signals for thread communication
        self.signals = TerminalSignals()
        self.signals.output_ready.connect(self.command_finished)
        
        # Style
        self.setStyleSheet(\"\"\"
            QTextEdit {
                background-color: rgba(255, 255, 255, 255);
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                border: none;
                color: black;
            }
        \"\"\")
        
        # Initial prompt
        self.append_prompt()
        
        # Track command execution
        self.executing = False
    
    def append_prompt(self):
        \"\"\"Append the %term prompt\"\"\"
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"{self.working_dir}%term ")
        self.setTextCursor(cursor)
    
    def keyPressEvent(self, event):
        \"\"\"Handle key presses\"\"\"
        from PySide6.QtCore import Qt
        
        # Handle Return/Enter to execute command
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Get current line
            cursor = self.textCursor()
            cursor.select(QTextCursor.LineUnderCursor)
            line = cursor.selectedText()
            
            print(f"[DEBUG] Enter pressed on line: '{line}'")
            
            # Check if line has %term prompt
            if '%term' in line:
                parts = line.split('%term', 1)
                if len(parts) > 1:
                    command = parts[1].strip()
                    print(f"[DEBUG] Extracted command from Enter: '{command}'")
                    
                    if command:
                        # Move cursor to end before adding newline
                        cursor = self.textCursor()
                        cursor.movePosition(QTextCursor.End)
                        self.setTextCursor(cursor)
                        
                        # Add newline
                        super().keyPressEvent(event)
                        
                        # Execute the command
                        print(f"[DEBUG] Executing command from Enter")
                        self.execute_command(command)
                        return
            
            # No command found, just add newline and prompt
            super().keyPressEvent(event)
            if not self.executing:
                self.append_prompt()
            return
        
        # Normal text editing
        super().keyPressEvent(event)
    
    def execute_command(self, command):
        \"\"\"Execute a shell command and append output\"\"\"
        print(f"[DEBUG] execute_command called with: '{command}'")
        
        if self.executing:
            print(f"[DEBUG] Already executing, returning busy message")
            self.append_output("\\n[Busy - command already running]\\n")
            return
        
        self.executing = True
        command = command.strip()
        
        print(f"[DEBUG] Stripped command: '{command}'")
        
        if not command:
            print(f"[DEBUG] Empty command, returning")
            self.executing = False
            return
        
        # Handle cd command specially
        if command.startswith('cd '):
            print(f"[DEBUG] Handling cd command")
            path = command[3:].strip()
            try:
                # Expand ~ and relative paths
                if path.startswith('~'):
                    path = os.path.expanduser(path)
                else:
                    path = os.path.join(self.working_dir, path)
                
                path = os.path.abspath(path)
                
                if os.path.isdir(path):
                    self.working_dir = path
                    self.append_output(f"\\nChanged directory to: {self.working_dir}\\n")
                else:
                    self.append_output(f"\\ncd: no such directory: {path}\\n")
            except Exception as e:
                self.append_output(f"\\ncd error: {e}\\n")
            
            self.executing = False
            self.append_prompt()
            return
        
        # Execute command in thread to not block UI
        print(f"[DEBUG] Starting thread to execute: '{command}'")
        
        def run_command():
            try:
                print(f"[DEBUG] In thread, running command...")
                # Run command with shell to support pipes, redirects, etc.
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                print(f"[DEBUG] Command completed with returncode: {result.returncode}")
                print(f"[DEBUG] stdout length: {len(result.stdout)}")
                print(f"[DEBUG] stderr length: {len(result.stderr)}")
                
                output = result.stdout
                if result.stderr:
                    output += result.stderr
                
                if not output:
                    output = f"[Command exited with code {result.returncode}]"
                
                print(f"[DEBUG] Emitting signal with output length: {len(output)}")
                print(f"[DEBUG] Output preview: {repr(output[:100])}")
                
                # Emit signal to main thread
                self.signals.output_ready.emit("\\n" + output)
                print(f"[DEBUG] Signal emitted")
                
            except subprocess.TimeoutExpired:
                print(f"[DEBUG] Command timed out")
                self.signals.output_ready.emit("\\n[Command timed out after 30s]\\n")
            except Exception as e:
                print(f"[DEBUG] Command error: {e}")
                self.signals.output_ready.emit(f"\\n[Error: {e}]\\n")
        
        thread = threading.Thread(target=run_command, daemon=True)
        thread.start()
        print(f"[DEBUG] Thread started")
    
    def command_finished(self, output):
        \"\"\"Called when command finishes (in main thread via signal)\"\"\"
        print(f"[DEBUG] command_finished called with output length: {len(output)}")
        print(f"[DEBUG] Output preview: {repr(output[:100])}")
        self.append_output(output)
        self.executing = False
        self.append_prompt()
        print(f"[DEBUG] Prompt appended, command finished")
    
    def append_output(self, text):
        \"\"\"Append text to the terminal\"\"\"
        print(f"[DEBUG] append_output called with text length: {len(text)}")
        print(f"[DEBUG] Text preview: {repr(text[:100])}")
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        print(f"[DEBUG] Text inserted into terminal")

# Create terminal
terminal = AcmeTerminal(r\"""" + working_dir + """\")
terminal.setParent(container)
terminal.setGeometry(0, 0, container.width() or 600, container.height() or 400)

def _resize(ev, w=terminal, c=container):
    w.setGeometry(0, 0, c.width(), c.height())
    type(c).resizeEvent(c, ev)
container.resizeEvent = _resize

# Store reference for event handling
container.terminal = terminal
"""
    return code
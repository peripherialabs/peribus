"""
quick_generators.py — scene-native code generators for /scene/quick

Every generator:
  1. Creates a viewer widget named  _widget
  2. Stores the source path in      _quick_path  (str, "" for messages)
  3. Falls through to _PROXY_FOOTER  which embeds in QGraphicsScene and
     registers with scene_manager
  4. Falls through to _ANIM_FOOTER  which runs the entrance animation:
       Phase 1 — QVariantAnimation   opacity  0 → 1  (350 ms, ease-out)
       Phase 2 — QVariantAnimation   shadow offset (0,0) → (8,8)
                                     + blur  0  → 25
                                     + alpha 0  → 100  (400 ms, ease-in-out)
                                     starts only after Phase 1 finishes

The execution namespace provides:
    graphics_scene   — QGraphicsScene
    scene_manager    — SceneManager
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Shared footers appended to every generator
# ──────────────────────────────────────────────────────────────────────────────

_PROXY_FOOTER = '''
# ── View-Aware Placement & Resizing ──────────────────────────────────────────
from PySide6.QtWidgets import QFrame, QVBoxLayout
from PySide6.QtCore import Qt, QPointF, QRectF

# Get the current visible area in scene coordinates
# 'graphics_view' is provided by the execution context
_view_rect = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
_v_w = _view_rect.width()
_v_h = _view_rect.height()

# Automatically resize: target 30% of the visible width/height
# but stay within reasonable bounds (min 300px, max 800px)
_win_w = max(300, min(800, int(_v_w * 0.30)))
_win_h = max(250, min(600, int(_v_h * 0.35)))

_frame = QFrame()
_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
_frame.setStyleSheet("""
    QFrame {
        background-color: rgba(248, 248, 248, 235);
        border: 1px solid rgba(180, 180, 180, 180);
        border-radius: 6px;
    }
""")
_frame_layout = QVBoxLayout(_frame)
_frame_layout.setContentsMargins(3, 3, 3, 3)
_frame_layout.addWidget(_widget)
_frame.resize(_win_w, _win_h)

_proxy = graphics_scene.addWidget(_frame)

# ── Collision-Free Search within Current View ────────────────────────────────
def find_free_spot_in_view(w, h, view_rect):
    margin = 20
    # Search start: top-left of the current visible view
    curr_x = view_rect.x() + margin
    curr_y = view_rect.y() + margin
    
    # Simple grid search within the visible bounds
    step_x = 100
    step_y = 100
    
    limit_x = view_rect.right() - w
    limit_y = view_rect.bottom() - h

    search_y = curr_y
    while search_y < limit_y:
        search_x = curr_x
        while search_x < limit_x:
            test_rect = QRectF(search_x, search_y, w + margin, h + margin)
            # Use graphics_scene.items to check for collisions
            if not graphics_scene.items(test_rect):
                return search_x, search_y
            search_x += step_x
        search_y += step_y
        
    # Fallback: center of view if no perfect spot is found
    return view_rect.center().x() - (w/2), view_rect.center().y() - (h/2)

_px, _py = find_free_spot_in_view(_win_w, _win_h, _view_rect)
_proxy.setPos(_px, _py)
_proxy.setZValue(100) # Ensure it appears on top of background elements
_proxy.setOpacity(0.0)

scene_manager.register_parsed_item(_proxy, {"quick": True, "path": _quick_path})
'''

_ANIM_FOOTER = '''
# ── Entrance animation ────────────────────────────────────────────────────────
from PySide6.QtCore import QVariantAnimation, QEasingCurve, QPointF
from PySide6.QtWidgets import QGraphicsDropShadowEffect
from PySide6.QtGui import QColor

# Pre-attach shadow — fully invisible at start
_shadow = QGraphicsDropShadowEffect()
_shadow.setBlurRadius(0)
_shadow.setOffset(QPointF(0.0, 0.0))
_shadow.setColor(QColor(0, 0, 0, 0))
_proxy.setGraphicsEffect(_shadow)

# ── Phase 1: opacity 0 → 1 (350 ms) ─────────────────────────────────────────
_anim_opacity = QVariantAnimation()
_anim_opacity.setStartValue(0.0)
_anim_opacity.setEndValue(1.0)
_anim_opacity.setDuration(350)
_anim_opacity.setEasingCurve(QEasingCurve.OutCubic)

def _on_opacity(value, p=_proxy):
    p.setOpacity(value)

_anim_opacity.valueChanged.connect(_on_opacity)

# ── Phase 2: shadow grows in (400 ms) — starts only after opacity finishes ───
_anim_shadow = QVariantAnimation()
_anim_shadow.setStartValue(0.0)
_anim_shadow.setEndValue(1.0)
_anim_shadow.setDuration(400)
_anim_shadow.setEasingCurve(QEasingCurve.InOutCubic)

def _on_shadow(t, sh=_shadow):
    sh.setBlurRadius(25.0 * t)
    sh.setOffset(QPointF(45.0 * t, 45.0 * t))
    sh.setColor(QColor(0, 0, 0, int(100 * t)))

_anim_shadow.valueChanged.connect(_on_shadow)

def _start_shadow():
    _anim_shadow.start()

_anim_opacity.finished.connect(_start_shadow)
_anim_opacity.start()

# Keep references alive so GC does not delete running animations
_proxy._quick_anim_opacity = _anim_opacity
_proxy._quick_anim_shadow  = _anim_shadow
_proxy._quick_shadow       = _shadow
_proxy._quick_frame        = _frame
'''


# ──────────────────────────────────────────────────────────────────────────────
# Text / source / data files  →  QTextEdit
# ──────────────────────────────────────────────────────────────────────────────

def generate_quick_file_content(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')

    return f'''# quick: text viewer — {path}
import os, concurrent.futures
from PySide6.QtWidgets import QTextEdit

_quick_path = r"{path_escaped}"

def _read(p):
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

try:
    if _quick_path.startswith("/n/"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            _content = _pool.submit(_read, _quick_path).result(timeout=2.0)
    else:
        _content = _read(_quick_path)
except concurrent.futures.TimeoutError:
    _content = "(file read timed out — likely a blocking synthetic file)"
except Exception as _e:
    _content = f"Error reading file: {{_e}}"

_widget = QTextEdit()
_widget.setReadOnly(False)
_widget.setPlainText(_content)
_widget.setStyleSheet("""
    QTextEdit {{
        background-color: rgba(255, 255, 255, 0);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        border: none;
        color: #1a1a1a;
        padding: 8px;
    }}
""")
''' + _PROXY_FOOTER + _ANIM_FOOTER


# ──────────────────────────────────────────────────────────────────────────────
# Directory listing  →  QTextEdit (column-formatted)
# ──────────────────────────────────────────────────────────────────────────────

def generate_quick_directory_listing(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')

    return f'''# quick: directory listing — {path}
import os, concurrent.futures
from PySide6.QtWidgets import QTextEdit

_quick_path = r"{path_escaped}"

def _fmt_cols(entries, width=80):
    if not entries:
        return ""
    col_w = max(len(e) for e in entries) + 2
    ncols = max(1, width // col_w)
    nrows = (len(entries) + ncols - 1) // ncols
    lines = []
    for row in range(nrows):
        parts = []
        for col in range(ncols):
            idx = row + col * nrows
            if idx < len(entries):
                e = entries[idx]
                parts.append(e.ljust(col_w) if col < ncols - 1 else e)
        lines.append("".join(parts).rstrip())
    return "\\n".join(lines)

def _is_dir(p, is9p=False):
    try:
        if os.path.isdir(p): return True
    except OSError:
        pass
    if is9p:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pl:
                _pl.submit(os.listdir, p).result(timeout=0.3)
            return True
        except Exception:
            return False
    try:
        os.listdir(p); return True
    except Exception:
        return False

try:
    _is9p    = _quick_path.startswith("/n/")
    _raw     = sorted(os.listdir(_quick_path))
    _entries = []
    _parent  = os.path.dirname(_quick_path)
    if _quick_path != _parent:
        _entries.append("../")
    for _e in _raw:
        _fp = os.path.join(_quick_path, _e)
        _entries.append(_e + ("/" if _is_dir(_fp, _is9p) else ""))
    _content = _fmt_cols(_entries)
except Exception as _ex:
    _content = f"Error reading directory: {{_ex}}"

_widget = QTextEdit()
_widget.setReadOnly(True)
_widget.setPlainText(_content)
_widget.setStyleSheet("""
    QTextEdit {{
        background-color: rgba(255, 255, 255, 0);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        border: none;
        color: #1a1a1a;
        padding: 8px;
    }}
""")
''' + _PROXY_FOOTER + _ANIM_FOOTER


# ──────────────────────────────────────────────────────────────────────────────
# Images  →  zoom/pan ImageViewWidget
# ──────────────────────────────────────────────────────────────────────────────

def generate_quick_image_viewer(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')

    return f'''# quick: image viewer — {path}
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QPoint, Qt, QRectF
from PySide6.QtGui import QPixmap, QPainter, QColor

_quick_path = r"{path_escaped}"

class _ImageView(QWidget):
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self._px     = QPixmap(img_path)
        self._scale  = 1.0
        self._offset = QPoint(0, 0)
        self._last   = None
        self._fit_done = False
        if self._px.isNull():
            self._px = None

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(245, 245, 245, 0))
        if not self._px:
            p.drawText(self.rect(), Qt.AlignCenter, "Failed to load image")
            return
        ss = self._px.size() * self._scale
        x  = (self.width()  - ss.width())  / 2 + self._offset.x()
        y  = (self.height() - ss.height()) / 2 + self._offset.y()
        p.drawPixmap(QRectF(x, y, ss.width(), ss.height()),
                     self._px, QRectF(self._px.rect()))

    def wheelEvent(self, ev):
        f = 1.1 if ev.angleDelta().y() > 0 else 0.9
        self._scale = max(0.05, min(20.0, self._scale * f))
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._last = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._last:
            self._offset += ev.pos() - self._last
            self._last = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._last = None
            self.setCursor(Qt.ArrowCursor)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._px and not self._fit_done:
            self._scale = min(
                self.width()  / max(self._px.width(),  1),
                self.height() / max(self._px.height(), 1),
                1.0,
            )
            self._fit_done = True
        self.update()

_widget = _ImageView(r"{path_escaped}")
''' + _PROXY_FOOTER + _ANIM_FOOTER


# ──────────────────────────────────────────────────────────────────────────────
# Video  →  QVideoSink → paintEvent  (QGraphicsScene-compatible)
#
# QVideoWidget uses a native window handle and cannot render inside an
# addWidget() proxy — it shows white.  QVideoSink is the correct approach
# for embedded scenes.  The previous implementation was slow because it called
# QPixmap.scaled(..., SmoothTransformation) on every single frame.
#
# Fix: scale once on resize into a fixed-size intermediate QPixmap (_scaled_pix)
# and in _process_frame only do a fast integer-only scale if the video
# resolution changed.  paintEvent just blits _scaled_pix — zero scaling work
# at 30/60 fps.
# ──────────────────────────────────────────────────────────────────────────────

def generate_quick_video_player(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')

    return f'''# quick: video player — {path}
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink, QVideoFrame
from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtGui import QPixmap, QPainter, QColor, QImage
from PySide6.QtCore import QUrl, Qt, QSize, QRect, Slot

_quick_path = r"{path_escaped}"

class _VideoInstance(QWidget):
    """
    QVideoSink-based player that is safe inside QGraphicsScene proxies.

    Performance design:
    - _process_frame converts QVideoFrame → QImage (unavoidable, decoder output)
      and stores it as _raw_image.  No scaling here.
    - paintEvent scales _raw_image to fit the widget using Qt.KeepAspectRatio
      and FastTransformation.  Qt defers paintEvent coalescing so even at 60 fps
      we never do more scaling work than the screen actually needs.
    - resizeEvent just calls update() — no pixmap work.
    - _scaled_size tracks the last drawn size so paintEvent can skip
      re-scaling when the widget size hasn't changed between frames.

    stop() tears down the full pipeline in the correct order so NVDEC surfaces
    are released before scene_manager.clear() runs.
    """
    def __init__(self, source_path):
        super().__init__()
        self.source_path = source_path
        self._is_stopped = False
        self._raw_image   = None   # latest decoded QImage, unscaled
        self._cached_pix  = None   # scaled QPixmap cache
        self._cached_size = QSize() # widget size when cache was built

        self.setStyleSheet("background-color: black; border-radius: 4px;")
        self.setMinimumSize(400, 300)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)  # skip background clear

        self._init_pipeline()

    # ── media pipeline ────────────────────────────────────────────────────────

    def _init_pipeline(self):
        self.sink  = QVideoSink()
        self.sink.videoFrameChanged.connect(self._process_frame)
        self.media = QMediaPlayer()
        self.audio = QAudioOutput()
        self.media.setAudioOutput(self.audio)
        self.media.setVideoOutput(self.sink)
        self.media.setSource(QUrl.fromLocalFile(self.source_path))
        self.media.setLoops(-1)
        self.media.play()
        self._is_stopped = False

    def stop(self):
        """
        Full synchronous teardown — call before scene_manager.clear().

        Uses sip.delete() rather than deleteLater() so the C++ QMediaPlayer
        and QAudioOutput objects (and their NVDEC decoder contexts) are
        destroyed immediately, not deferred to the next event-loop tick.
        The asyncio.sleep(0) in quick_file._display then lets Qt flush any
        remaining internal cleanup before the new player is created.
        """
        if self._is_stopped:
            return
        self._is_stopped = True
        if hasattr(self, "sink"):
            try:
                self.sink.videoFrameChanged.disconnect(self._process_frame)
            except RuntimeError:
                pass
        if hasattr(self, "media"):
            self.media.stop()
            self.media.setVideoOutput(None)
            self.media.setAudioOutput(None)
            self.media.setSource(QUrl())
            try:
                import sip
                sip.delete(self.media)
            except Exception:
                self.media.deleteLater()
            self.media = None
        if hasattr(self, "audio") and self.audio is not None:
            try:
                import sip
                sip.delete(self.audio)
            except Exception:
                self.audio.deleteLater()
            self.audio = None

    # ── frame ingestion: store raw, invalidate cache, schedule one repaint ────

    @Slot(QVideoFrame)
    def _process_frame(self, frame):
        if not frame.isValid():
            return
        img = frame.toImage()
        if img.isNull():
            return
        # Convert to Format_RGB32 once here — painter handles it fastest
        self._raw_image  = img.convertToFormat(QImage.Format_RGB32)
        self._cached_pix = None   # invalidate cache; paintEvent will rescale
        self.update()             # coalesced by Qt — only one paint per vsync

    # ── paint: scale only when widget size or image changed ──────────────────

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if self._raw_image is None or self._raw_image.isNull():
            if self._is_stopped:
                p.setPen(QColor(136, 136, 136))
                p.drawText(self.rect(), Qt.AlignCenter, "\\u25b6  Click to replay")
            return

        cur_size = self.size()
        if self._cached_pix is None or self._cached_size != cur_size:
            # Scale only on actual size change — FastTransformation, no subpixel work
            scaled = self._raw_image.scaled(
                cur_size,
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
            self._cached_pix  = QPixmap.fromImage(scaled)
            self._cached_size = cur_size

        # Centre the cached pixmap
        x = (cur_size.width()  - self._cached_pix.width())  // 2
        y = (cur_size.height() - self._cached_pix.height()) // 2
        p.drawPixmap(x, y, self._cached_pix)

    # ── click to stop / replay ────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            if not self._is_stopped:
                self.stop()
                self._raw_image  = None
                self._cached_pix = None
                self.update()
            else:
                self._init_pipeline()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._cached_pix = None   # force rescale on next paintEvent
        self.update()

_widget = _VideoInstance(_quick_path)

''' + _PROXY_FOOTER + _ANIM_FOOTER + '''
_proxy._instance_ref = _widget
# Register via builtins — reachable from any exec() namespace without an import
import builtins as _builtins; _builtins._quick_video_instance = _widget
'''


# ──────────────────────────────────────────────────────────────────────────────
# Audio  →  minimal player UI
# ──────────────────────────────────────────────────────────────────────────────

def generate_quick_audio_player(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')
    basename = os.path.basename(path)

    return f'''# quick: audio player — {path}
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PySide6.QtCore import QUrl, Qt

_quick_path = r"{path_escaped}"

_media = QMediaPlayer()
_audio = QAudioOutput()
_media.setAudioOutput(_audio)
_media.setSource(QUrl.fromLocalFile(_quick_path))

class _AudioWidget(QWidget):
    def __init__(self, player, audio_out, filename, parent=None):
        super().__init__(parent)
        self._p = player
        self._a = audio_out
        self.setStyleSheet("""
            QWidget {{
                background-color: transparent;
            }}
            QPushButton {{
                background-color: rgba(230, 230, 230, 220);
                border: 1px solid rgba(180, 180, 180, 160);
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: rgba(215, 215, 215, 240);
            }}
            QPushButton:pressed {{
                background-color: rgba(195, 195, 195, 240);
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setAlignment(Qt.AlignCenter)

        lbl = QLabel(filename)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "font-size: 15px; font-family: 'Segoe UI', sans-serif; "
            "padding: 10px; color: #333;"
        )
        layout.addWidget(lbl)
        layout.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        for text, fn in [
            ("\\u23ee", self._rewind),
            ("\\u25b6 / \\u23f8", self._playpause),
            ("\\u23f9", self._stop),
        ]:
            b = QPushButton(text)
            b.setFixedWidth(90)
            b.setFixedHeight(36)
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)
        layout.addSpacing(12)

        vol_lbl = QLabel("Volume")
        vol_lbl.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(vol_lbl)

        vol = QSlider(Qt.Horizontal)
        vol.setRange(0, 100)
        vol.setValue(80)
        vol.valueChanged.connect(lambda v: self._a.setVolume(v / 100.0))
        self._a.setVolume(0.8)
        layout.addWidget(vol)

    def _playpause(self):
        from PySide6.QtMultimedia import QMediaPlayer as _MP
        if self._p.playbackState() == _MP.PlayingState:
            self._p.pause()
        else:
            self._p.play()

    def _stop(self):   self._p.stop()
    def _rewind(self): self._p.setPosition(0); self._p.play()

_widget = _AudioWidget(_media, _audio, r"{basename}")
''' + _PROXY_FOOTER + _ANIM_FOOTER + '''
# Prevent GC
_proxy._quick_media = _media
_proxy._quick_audio = _audio
'''


# ──────────────────────────────────────────────────────────────────────────────
# PDF  →  PyMuPDF scroll view
# ──────────────────────────────────────────────────────────────────────────────

def generate_quick_pdf_viewer(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')

    return f'''# quick: PDF viewer — {path}
import fitz
from PySide6.QtWidgets import QScrollArea, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

_quick_path = r"{path_escaped}"

_pages_w = QWidget()
_pages_w.setStyleSheet("background: transparent;")
_pages_l = QVBoxLayout(_pages_w)
_pages_l.setSpacing(10)
_pages_l.setContentsMargins(12, 12, 12, 12)

try:
    _doc = fitz.open(_quick_path)
    for _pn in range(len(_doc)):
        _pg  = _doc[_pn]
        _pix = _pg.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        _img = QImage(_pix.samples, _pix.width, _pix.height,
                      _pix.stride, QImage.Format_RGB888)
        _lbl = QLabel()
        _lbl.setPixmap(QPixmap.fromImage(_img))
        _lbl.setAlignment(Qt.AlignCenter)
        _lbl.setStyleSheet("background: transparent; border: none;")
        _pages_l.addWidget(_lbl)
    _doc.close()
except Exception as _ex:
    _err = QLabel(f"Error loading PDF: {{_ex}}")
    _err.setAlignment(Qt.AlignCenter)
    _err.setStyleSheet("color: #cc0000; font-size: 13px; padding: 20px;")
    _pages_l.addWidget(_err)

_scroll = QScrollArea()
_scroll.setWidgetResizable(True)
_scroll.setStyleSheet("""
    QScrollArea {{
        background: transparent;
        border: none;
    }}
""")
_scroll.setWidget(_pages_w)
_widget = _scroll
''' + _PROXY_FOOTER + _ANIM_FOOTER


# ──────────────────────────────────────────────────────────────────────────────
# 3-D models  →  OpenGL wire-frame viewer  (no wrapper frame)
# ──────────────────────────────────────────────────────────────────────────────

_PROXY_FOOTER_3D = '''
# ── Embed 3-D widget directly in QGraphicsScene (no frame wrapper) ────────────
from PySide6.QtCore import QPointF as _QPointF

_scene_w = scene_manager.width  or 1920
_scene_h = scene_manager.height or 1080

_win_w = max(360, min(820, int(_scene_w * 0.42)))
_win_h = max(260, min(600, int(_scene_h * 0.48)))

_widget.resize(_win_w, _win_h)

_proxy = graphics_scene.addWidget(_widget)

_px = (_scene_w - _win_w) / 2
_py = (_scene_h - _win_h) / 2
_proxy.setPos(_px, _py)

_proxy.setZValue(0)
_proxy.setOpacity(0.0)

# _frame alias so _ANIM_FOOTER's GC-pin line doesn't crash
_frame = _widget

scene_manager.register_parsed_item(_proxy, {"quick": True, "path": _quick_path})
'''

def generate_quick_3d_viewer(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path_escaped = path.replace('\\', '\\\\')

    return f'''# quick: 3-D viewer — {path}
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh, numpy as np

_quick_path = r"{path_escaped}"

try:
    from OpenGL.arrays import vbo as _vbo_mod
except Exception:
    _vbo_mod = None

class _Mesh3DWidget(QOpenGLWidget):
    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.setUpdateBehavior(QOpenGLWidget.NoPartialUpdate)

        self._path   = model_path
        self._verts  = None
        self._edges  = None
        self._vbo_v  = None
        self._vbo_e  = None
        self._rx     = 20.0
        self._ry     = 0.0
        self._last   = None
        self._zoom   = 3.0

        self._proxy_ref = None
        self._scene_ref = None

        self._load()

    # ── mesh loading ──────────────────────────────────────────────────────────
    def _load(self):
        try:
            m = trimesh.load(self._path, force="mesh")
            if len(m.faces) > 2000:
                try:
                    m = m.simplify_quadric_decimation(2000 / len(m.faces))
                except Exception:
                    pass
            self._edges = m.edges_unique
            v = m.vertices.astype(np.float32)
            v -= v.mean(axis=0)
            d = np.max(np.linalg.norm(v, axis=1))
            if d > 0:
                v /= d
            self._verts = v
        except Exception as _e:
            print(f"3-D load error: {{_e}}")

    # ── OpenGL ────────────────────────────────────────────────────────────────
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.96, 0.96, 0.96, 1.0)
        if self._verts is not None and _vbo_mod is not None:
            try:
                self._vbo_v = _vbo_mod.VBO(self._verts)
                self._vbo_e = _vbo_mod.VBO(
                    self._edges.astype(np.uint32),
                    target=GL_ELEMENT_ARRAY_BUFFER,
                )
            except Exception:
                pass

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._verts is None:
            return
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.width() / max(self.height(), 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, self._zoom, 0, 0, 0, 0, 1, 0)
        glRotatef(self._rx, 1, 0, 0)
        glRotatef(self._ry, 0, 1, 0)
        glColor3f(0.05, 0.05, 0.05)
        glLineWidth(1.0)
        if self._vbo_v is not None and self._vbo_e is not None:
            self._vbo_v.bind()
            self._vbo_e.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glDrawElements(GL_LINES, len(self._edges) * 2, GL_UNSIGNED_INT, None)
            glDisableClientState(GL_VERTEX_ARRAY)
            self._vbo_v.unbind()
            self._vbo_e.unbind()
        else:
            glBegin(GL_LINES)
            for _e2 in self._edges:
                glVertex3fv(self._verts[_e2[0]])
                glVertex3fv(self._verts[_e2[1]])
            glEnd()
        glFlush()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self._repaint_all()

    def _repaint_all(self):
        self.update()
        if self._proxy_ref is not None:
            self._proxy_ref.update()
        if self._scene_ref is not None:
            self._scene_ref.update()

    # ── interaction ───────────────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._last = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._last is not None:
            d = ev.pos() - self._last
            self._last = ev.pos()
            self._ry += d.x() * 0.5
            self._rx += d.y() * 0.5
            self._repaint_all()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._last = None
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, ev):
        f = 0.9 if ev.angleDelta().y() > 0 else 1.1
        self._zoom = max(0.5, min(20.0, self._zoom * f))
        self._repaint_all()

_widget = _Mesh3DWidget(r"{path_escaped}")
''' + _PROXY_FOOTER_3D + '''
# Back-references so _repaint_all() can poke the scene
_widget._proxy_ref = _proxy
_widget._scene_ref = graphics_scene
''' + _ANIM_FOOTER


# ──────────────────────────────────────────────────────────────────────────────
# Message / error display
# ──────────────────────────────────────────────────────────────────────────────

def generate_quick_message(message: str, title: str = "") -> str:
    msg_esc   = message.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    title_esc = (title or "Info").replace('\\', '\\\\').replace("'", "\\'")

    return f'''# quick: message — {title_esc}
from PySide6.QtWidgets import QTextEdit

_quick_path = ""
_widget = QTextEdit()
_widget.setReadOnly(True)
_widget.setPlainText('{msg_esc}')
_widget.setStyleSheet("""
    QTextEdit {{
        background-color: rgba(255, 255, 255, 0);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        border: none;
        color: #1a1a1a;
        padding: 12px;
    }}
""")
''' + _PROXY_FOOTER + _ANIM_FOOTER
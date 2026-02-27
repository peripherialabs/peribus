"""
Rio Onboarding Animation
=========================
Executed via main_window._launch_onboarding() → self.executor.execute(code)

Available globals: main_window, graphics_scene, graphics_view

Timing (from music start):
  00:00   Fullscreen + Start button
  00:00   Music + welcome screen (on Start click)
  ~10.2s  Welcome done → tutorial begins
  ~15.2s  Terminal ready, cursor moves away, streaming starts
  ~41.5s  LLM code finishes → button appears immediately
  48.0s   Button morph done + "human or machine?" text
  ~49.0s  Voice menu opens → eyes widget
  53.5s   SVG beside eyes + view tilt begins + serious work
  ~54.3s  Eyes wink once (reacting to SVG appearance)
          Widgets spawn rapidly: calendar, color wheel, dashboard,
          sliders, notes, analog clock, sparkline, file tree,
          VU meters, connection matrix — all transparent+border+shadow
  ~57s    SVG draw done → shadow offset animation
  1m20s   View tilt complete → immersive mode begins
  ~1m20.5 Background fade starts (white→dark, 2s ease-in-out)
  ~1m21s  Widget colors invert + SVG turns neon pink
  ~1m23.5 Shadow color party (random vivid glow per widget, 2.5s)
          Eyes roam between widgets watching the transformation
  ~1m24.5 Camera + MediaPipe starts (face mesh off-screen)
  ~1m26   Face mesh slides in from top-right
          Head gaze controls cursor, hand pinch grabs widgets
  1m35    All widgets animate outward to periphery (4s spread)
          Eyes scan wildly during spread
  ~1m39   3D car game spawns in scene center
          Face mesh retreats slowly to top-right (6s)
          View tilt reverses back to normal (25s ease-in-out)
  2m04    View at rest → WelcomeScreen reappears with logo
          Behind the logo: all scene widgets fade out one by one
          Camera stops, overlays removed, scene cleaned
          Welcome screen finishes → onboarding complete
"""

import os
import time
import inspect
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMenu,
    QGraphicsDropShadowEffect, QApplication, QGraphicsProxyWidget
)
from PySide6.QtCore import (
    Qt, QTimer, QPointF, QObject, QUrl, Signal, QPoint, QElapsedTimer,
    QByteArray, QPropertyAnimation, QEasingCurve
)
from PySide6.QtGui import (
    QColor, QBrush, QPen, QMovie, QAction, QCursor, QFont, QPainterPath
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtSvg import QSvgRenderer

try:
    import numpy as np
    import cv2
    import mediapipe as mp_lib
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    print("[Onboarding] cv2/mediapipe/numpy not installed — camera features disabled")


# ============================================================================
# MediaPipe Camera + Face Mesh + Hands (from immersive_mode.py)
# ============================================================================

class CameraManager(QObject):
    """Camera capture with MediaPipe Face Mesh + Hands."""
    frame_ready = Signal(object, object, object)  # frame, face_landmarks, hands_data

    def __init__(self):
        super().__init__()
        self.capture = None
        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._capture_frame)
        self.mp_face_mesh = None
        self.face_mesh_model = None
        self.mp_hands = None
        self.hands_model = None

    def initialize_mediapipe(self) -> bool:
        if not _HAS_MEDIAPIPE:
            return False
        try:
            self.mp_face_mesh = mp_lib.solutions.face_mesh
            self.face_mesh_model = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
            self.mp_hands = mp_lib.solutions.hands
            self.hands_model = self.mp_hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.7, min_tracking_confidence=0.5,
            )
            return True
        except Exception as e:
            print(f"[Onboarding] MediaPipe init error: {e}")
            return False

    def start(self) -> bool:
        if not _HAS_MEDIAPIPE:
            return False
        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened():
            self.is_running = True
            self.timer.start(33)
            return True
        return False

    def _capture_frame(self):
        if not self.capture or not self.is_running:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_landmarks = None
        face_results = self.face_mesh_model.process(rgb)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

        hand_results = self.hands_model.process(rgb)
        hands_data = []
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hl, hc in zip(hand_results.multi_hand_landmarks,
                              hand_results.multi_handedness):
                label = hc.classification[0].label
                hands_data.append((hl, label))

        self.frame_ready.emit(frame, face_landmarks, hands_data)

    def stop(self):
        self.is_running = False
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None


class OnboardingFaceMeshWidget(QWidget):
    """Draws the MediaPipe face mesh with isometric 3D rotation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas_w = 800
        self._canvas_h = 800
        self.setFixedSize(self._canvas_w, self._canvas_h)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.current_frame = None
        self.face_landmarks = None
        self.smoothed_landmarks = None
        self.smoothing_factor = 0.45
        self.rotation_y = 0.4
        self.rotation_x = 0.3
        self.scale_x = 1.0
        self.scale_y = 0.8
        self.face_scale = 2.2

    def _smooth(self, new_landmarks):
        if self.smoothed_landmarks is None:
            self.smoothed_landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in new_landmarks.landmark
            ]
            return self.smoothed_landmarks
        alpha = 1.0 - self.smoothing_factor
        for i, lm in enumerate(new_landmarks.landmark):
            if i < len(self.smoothed_landmarks):
                s = self.smoothed_landmarks[i]
                s["x"] = alpha * lm.x + (1 - alpha) * s["x"]
                s["y"] = alpha * lm.y + (1 - alpha) * s["y"]
                s["z"] = alpha * lm.z + (1 - alpha) * s["z"]
        return self.smoothed_landmarks

    def set_face_data(self, frame, landmarks):
        self.current_frame = frame
        self.face_landmarks = landmarks
        if landmarks:
            self._smooth(landmarks)
        self.update()

    def _project(self, x, y, z=0):
        import math
        cx, cy = 0.5, 0.5
        xc, yc, zc = x - cx, y - cy, z
        cos_y, sin_y = math.cos(self.rotation_y), math.sin(self.rotation_y)
        xr = xc * cos_y + zc * sin_y
        zr = -xc * sin_y + zc * cos_y
        cos_x, sin_x = math.cos(self.rotation_x), math.sin(self.rotation_x)
        yr = yc * cos_x - zr * sin_x
        ix = xr * self.scale_x * self.face_scale + cx
        iy = yr * self.scale_y * self.face_scale + cy
        return ix, iy

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.smoothed_landmarks or self.current_frame is None:
            painter.end(); return
        w, h = self.width(), self.height()

        if _HAS_MEDIAPIPE:
            connections = mp_lib.solutions.face_mesh.FACEMESH_TESSELATION
        else:
            connections = []

        painter.setPen(QPen(QColor(200, 210, 230, 100), 1))
        for c in connections:
            s, e = c
            if s >= len(self.smoothed_landmarks) or e >= len(self.smoothed_landmarks):
                continue
            sl, el = self.smoothed_landmarks[s], self.smoothed_landmarks[e]
            sx, sy = self._project(1 - sl["x"], sl["y"], sl["z"])
            ex, ey = self._project(1 - el["x"], el["y"], el["z"])
            painter.drawLine(int(sx * w), int(sy * h), int(ex * w), int(ey * h))

        painter.setPen(QPen(QColor(240, 245, 255), 2))
        for lm in self.smoothed_landmarks:
            px, py = self._project(1 - lm["x"], lm["y"], lm["z"])
            painter.drawPoint(int(px * w), int(py * h))
        painter.end()


class GazeCursorWidget(QWidget):
    """Subtle reticle that follows head gaze direction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 52)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self._pinching = False

    def set_pinching(self, val: bool):
        if val != self._pinching:
            self._pinching = val
            self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = self.rect().center()
        if self._pinching:
            p.setBrush(QBrush(QColor(255, 160, 80, 200)))
            p.setPen(QPen(QColor(255, 120, 40), 2.5))
            p.drawEllipse(c, 9, 9)
        else:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(140, 210, 255, 160), 1.8))
            p.drawEllipse(c, 14, 14)
            p.setBrush(QBrush(QColor(140, 210, 255, 120)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(c, 2, 2)
        p.end()


class HeadGazeTracker:
    """Converts face landmark data into a stable screen-space gaze point."""

    NOSE_TIP = 1; LEFT_EAR = 234; RIGHT_EAR = 454; FOREHEAD = 10; CHIN = 152

    def __init__(self, sensitivity_x=2.8, sensitivity_y=3.0, dead_zone=0.025,
                 ema_alpha_1=0.40, ema_alpha_2=0.50):
        self.sensitivity_x = sensitivity_x
        self.sensitivity_y = sensitivity_y
        self.dead_zone = dead_zone
        self._a1 = ema_alpha_1
        self._a2 = ema_alpha_2
        self._s1_x = self._s1_y = None
        self._s2_x = self._s2_y = None

    def compute_gaze(self, face_landmarks, win_w, win_h):
        if face_landmarks is None:
            return self._last_pos(win_w, win_h)
        lm = face_landmarks.landmark
        if len(lm) < 468:
            return self._last_pos(win_w, win_h)

        nose = lm[self.NOSE_TIP]
        l_ear, r_ear = lm[self.LEFT_EAR], lm[self.RIGHT_EAR]
        forehead, chin = lm[self.FOREHEAD], lm[self.CHIN]

        face_cx = (l_ear.x + r_ear.x) / 2.0
        face_cy = (forehead.y + chin.y) / 2.0
        face_w = abs(r_ear.x - l_ear.x)
        face_h = abs(chin.y - forehead.y)
        if face_w < 0.01 or face_h < 0.01:
            return self._last_pos(win_w, win_h)

        dx = (nose.x - face_cx) / face_w
        dy = (nose.y - face_cy) / face_h

        if abs(dx) < self.dead_zone:
            dx = 0.0
        else:
            dx = (abs(dx) - self.dead_zone) * (1.0 if dx > 0 else -1.0)
        if abs(dy) < self.dead_zone:
            dy = 0.0
        else:
            dy = (abs(dy) - self.dead_zone) * (1.0 if dy > 0 else -1.0)

        raw_x = max(0.0, min(1.0, 0.5 - dx * self.sensitivity_x))
        raw_y = max(0.0, min(1.0, 0.5 + dy * self.sensitivity_y))

        if self._s1_x is None:
            self._s1_x, self._s1_y = raw_x, raw_y
            self._s2_x, self._s2_y = raw_x, raw_y
        else:
            self._s1_x += self._a1 * (raw_x - self._s1_x)
            self._s1_y += self._a1 * (raw_y - self._s1_y)
            self._s2_x += self._a2 * (self._s1_x - self._s2_x)
            self._s2_y += self._a2 * (self._s1_y - self._s2_y)

        return QPointF(self._s2_x * win_w, self._s2_y * win_h)

    def _last_pos(self, win_w, win_h):
        if self._s2_x is not None:
            return QPointF(self._s2_x * win_w, self._s2_y * win_h)
        return None

    def reset(self):
        self._s1_x = self._s1_y = None
        self._s2_x = self._s2_y = None


class PinchDetector:
    """Detects pinch on the right hand (thumb+middle or thumb+ring)."""
    THUMB_TIP = 4; MIDDLE_TIP = 12; RING_TIP = 16

    def __init__(self, threshold=0.055):
        self.threshold = threshold
        self.is_pinching = False

    def update(self, hands_data) -> bool:
        import math
        right_hand = None
        for hl, label in (hands_data or []):
            if label == "Left":  # user's right hand in mirror
                right_hand = hl; break
        if right_hand is None and hands_data:
            right_hand = hands_data[0][0]

        was = self.is_pinching
        if right_hand is None:
            self.is_pinching = False
            return was != self.is_pinching

        lm = right_hand.landmark
        thumb, middle, ring = lm[self.THUMB_TIP], lm[self.MIDDLE_TIP], lm[self.RING_TIP]
        dist_m = math.sqrt((thumb.x-middle.x)**2 + (thumb.y-middle.y)**2 + (thumb.z-middle.z)**2)
        dist_r = math.sqrt((thumb.x-ring.x)**2 + (thumb.y-ring.y)**2 + (thumb.z-ring.z)**2)
        self.is_pinching = min(dist_m, dist_r) < self.threshold
        return was != self.is_pinching


# ============================================================================
# Precision Clock — monotonic time.perf_counter() based scheduling
# ============================================================================
# QTimer.singleShot / interval timers accumulate drift under CPU load because
# they rely on the Qt event-loop granularity (~15 ms on Windows, variable
# elsewhere).  For a music-synced animation this is unacceptable.
#
# Strategy:
#   • Every tick-based animation uses time.perf_counter() to compute actual
#     elapsed time and derives its progress from that, not from a step counter.
#   • The QTimer is still used to *wake us up* (it's the only option inside
#     the Qt event loop), but its interval is only advisory.  The real time
#     source is perf_counter.
#   • One-shot delays (QTimer.singleShot replacements) are wrapped in a
#     helper that compensates for event-loop jitter by adjusting the
#     remaining delay on each wake-up.
# ============================================================================

def _precise_singleshot(delay_ms, callback, parent=None):
    """
    Schedule *callback* to fire after *delay_ms* milliseconds using
    monotonic perf_counter, not QTimer's drifting timeout.
    
    A QTimer wakes us up slightly early and we re-schedule the remainder
    until the wall-clock target is reached.  Typical accuracy: < 2 ms.
    """
    target = time.perf_counter() + delay_ms / 1000.0

    def _check():
        remaining = target - time.perf_counter()
        if remaining <= 0.001:          # close enough (1 ms tolerance)
            callback()
        else:
            # re-arm: ask for 80% of remaining to avoid overshooting
            ms = max(1, int(remaining * 800))
            QTimer.singleShot(ms, _check)

    # First arm — ask to wake up ~2 ms early
    first = max(1, delay_ms - 2)
    QTimer.singleShot(first, _check)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_rio_dir = os.path.dirname(os.path.abspath(inspect.getfile(main_window.__class__)))
_onboarding_dir = os.path.join(_rio_dir, "onboarding")
_logo_path = os.path.join(_onboarding_dir, "logo.gif")
_music_path = os.path.join(_onboarding_dir, "onboarding.wav")
_svg_path = os.path.join(_onboarding_dir, "both.svg")

print(f"[Onboarding] Rio dir:      {_rio_dir}")
print(f"[Onboarding] Logo exists:  {os.path.exists(_logo_path)}")
print(f"[Onboarding] Music exists: {os.path.exists(_music_path)}")
print(f"[Onboarding] SVG exists:   {os.path.exists(_svg_path)}")


# ============================================================================
# Debug Timer Overlay
# ============================================================================

class DebugTimerOverlay(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(0, 0, 0, 180);"
            "  color: #00ff88;"
            "  font-family: 'Consolas', 'SF Mono', monospace;"
            "  font-size: 14px; font-weight: bold;"
            "  padding: 4px 10px;"
            "  border-bottom-right-radius: 6px;"
            "}"
        )
        self.setFixedHeight(28)
        self.setMinimumWidth(100)
        self.move(0, 0)
        self._elapsed = QElapsedTimer()
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._update)
        self._running = False
        self.setText("00:00.0")
        self.show(); self.raise_()

    def start_timer(self):
        self._elapsed.start()
        self._running = True
        self._tick_timer.start(100)

    def _update(self):
        if not self._running:
            return
        ms = self._elapsed.elapsed()
        s = ms // 1000; t = (ms % 1000) // 100
        self.setText(f"{s//60:02d}:{s%60:02d}.{t}")

    def stop_timer(self):
        self._running = False
        self._tick_timer.stop()


# ============================================================================
# SVG Hand-Draw Widget
# ============================================================================

class HandDrawSvgWidget(QWidget):
    """
    Displays an SVG with an animated hand-drawing effect.
    
    Reads the SVG, modifies each path to have stroke-dasharray/dashoffset
    and fill-opacity=0, then animates them sequentially by injecting
    updated SVG content into a QSvgWidget at each tick.
    """

    def __init__(self, svg_path, parent=None):
        super().__init__(parent)
        self._svg_path = svg_path
        self._raw_svg = ""
        self._paths = []  # list of path 'd' strings
        self._total_progress = 0.0  # 0..1
        self._anim_timer = None
        self._color = "black"  # current stroke/fill color

        # Read SVG
        with open(svg_path, 'r') as f:
            self._raw_svg = f.read()

        # Extract paths
        import re
        self._paths = re.findall(r'<path\s+d="([^"]+)"', self._raw_svg)
        self._num_paths = len(self._paths)

        # Parse viewBox for sizing
        vb_match = re.search(r'viewBox="([\d.\s]+)"', self._raw_svg)
        if vb_match:
            parts = vb_match.group(1).split()
            self._vb_w = float(parts[2])
            self._vb_h = float(parts[3])
        else:
            self._vb_w = 852
            self._vb_h = 113

        # Widget size — scale to reasonable display size
        scale = 0.8
        self._display_w = int(self._vb_w * scale)
        self._display_h = int(self._vb_h * scale)
        self.setFixedSize(self._display_w, self._display_h)

        # SVG widget inside
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._svg_widget = QSvgWidget()
        self._svg_widget.setFixedSize(self._display_w, self._display_h)
        layout.addWidget(self._svg_widget)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        # Start with empty
        self._render_progress(0.0)

    def _build_svg_at_progress(self, progress):
        """
        Build SVG string where paths up to `progress` are progressively revealed.
        Each path gets an equal slice of the total progress.
        """
        import re
        svg = self._raw_svg
        color = self._color

        if self._num_paths == 0:
            return svg

        per_path = 1.0 / self._num_paths

        def replace_path(match):
            idx = replace_path._idx
            replace_path._idx += 1

            path_start = idx * per_path
            path_end = (idx + 1) * per_path
            d_attr = match.group(1)

            if progress <= path_start:
                # Not yet visible
                return f'<path d="{d_attr}" style="stroke: none; fill: none;"/>'
            elif progress >= path_end:
                # Fully drawn
                return f'<path d="{d_attr}" style="stroke: {color}; fill: {color}; stroke-width: 1;"/>'
            else:
                # Partially drawing this path
                local_p = (progress - path_start) / per_path
                # Use stroke-dasharray trick
                total_len = 2000  # generous estimate for dash length
                dash_offset = total_len * (1.0 - local_p)
                fill_opacity = max(0, (local_p - 0.5) * 2)  # fill fades in during second half
                return (
                    f'<path d="{d_attr}" style="'
                    f'stroke: {color}; stroke-width: 1.5; fill: {color}; '
                    f'fill-opacity: {fill_opacity:.2f}; '
                    f'stroke-dasharray: {total_len}; '
                    f'stroke-dashoffset: {dash_offset:.1f};'
                    f'"/>'
                )

        replace_path._idx = 0
        result = re.sub(r'<path\s+d="([^"]+)"\s+style="[^"]*"\s*/>', replace_path, svg)
        return result

    def _render_progress(self, progress):
        svg_str = self._build_svg_at_progress(progress)
        self._svg_widget.load(QByteArray(svg_str.encode('utf-8')))

    def set_color(self, color_str):
        """Change the stroke/fill color and re-render at current progress."""
        self._color = color_str
        self._render_progress(self._total_progress)

    def start_animation(self, duration_ms=3000, callback=None):
        """Animate drawing from 0% to 100% using wall-clock time."""
        self._total_progress = 0.0
        duration_s = duration_ms / 1000.0
        t0 = time.perf_counter()

        def tick():
            elapsed = time.perf_counter() - t0
            p = min(1.0, elapsed / duration_s)
            # Ease out quad
            p_eased = 1 - (1 - p) * (1 - p)
            self._total_progress = p_eased
            self._render_progress(p_eased)
            if p >= 1.0:
                self._anim_timer.stop()
                if callback:
                    callback()

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(tick)
        self._anim_timer.start(30)  # advisory interval; real progress from perf_counter


# ============================================================================
# WelcomeScreen
# ============================================================================

class WelcomeScreen(QWidget):
    finished = Signal()

    def __init__(self, logo_path):
        super().__init__()
        self._logo_path = logo_path
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label)
        self._has_played_once = False

        if os.path.exists(self._logo_path):
            self._movie = QMovie(self._logo_path)
            self.logo_label.setMovie(self._movie)
            self._movie.frameChanged.connect(self._on_frame_changed)
        else:
            self.logo_label.setText("Peribus")
            self.logo_label.setStyleSheet("QLabel{font-size:64px;font-weight:bold;color:#333;}")

        self._gif_timer = QTimer(self)
        self._gif_timer.setSingleShot(True)
        self._gif_timer.timeout.connect(self._on_gif_finished)

        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._fade_in = QPropertyAnimation(self, b"windowOpacity")
        self._fade_in.setDuration(800); self._fade_in.setStartValue(0.0); self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_in.finished.connect(self._start_logo_reveal)
        self._logo_effect = QGraphicsOpacityEffect()
        self.logo_label.setGraphicsEffect(self._logo_effect)
        self._logo_anim = QPropertyAnimation(self._logo_effect, b"opacity")
        self._logo_anim.setDuration(600); self._logo_anim.setStartValue(0.0); self._logo_anim.setEndValue(1.0)
        self._logo_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_out = QPropertyAnimation(self, b"windowOpacity")
        self._fade_out.setDuration(600); self._fade_out.setStartValue(1.0); self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.InCubic)
        self._fade_out.finished.connect(self._on_fade_out_done)

    def show_welcome(self):
        self.showFullScreen(); self.raise_(); self.activateWindow()
        self._logo_effect.setOpacity(0.0); self.setWindowOpacity(0.0)
        self._fade_in.start()

    def _start_logo_reveal(self):
        self._logo_anim.start()
        if hasattr(self, '_movie'):
            self._movie.start(); self._gif_timer.start(9000)
        else:
            self._gif_timer.start(2000)

    def _on_frame_changed(self, n):
        if n == 0 and self._has_played_once: self._on_gif_finished()
        elif n > 0: self._has_played_once = True

    def _on_gif_finished(self):
        if self._gif_timer.isActive(): self._gif_timer.stop()
        if hasattr(self, '_movie'): self._movie.stop()
        _precise_singleshot(300, lambda: (self.raise_(), self._fade_out.start()))

    def _on_fade_out_done(self):
        self.hide(); self.finished.emit()


# ============================================================================
# Onboarding Controller
# ============================================================================

class OnboardingController(QObject):

    _CSS_NORMAL = (
        "QMenu{background-color:rgba(255,255,255,200);border:1px solid #000000;"
        "padding:2px 0px;font-family:'Consolas','Monaco',monospace;font-size:12px;}"
        "QMenu::item{color:#000000;padding:4px 20px 4px 10px;}"
        "QMenu::item:selected{background-color:rgba(0,0,0,242);color:#ffffff;}"
        "QMenu::separator{height:1px;background:#000000;margin:2px 4px;}"
    )
    _CSS_FLASH = (
        "QMenu{background-color:rgba(0,0,0,242);border:1px solid #000000;"
        "padding:2px 0px;font-family:'Consolas','Monaco',monospace;font-size:12px;}"
        "QMenu::item{color:#ffffff;padding:4px 20px 4px 10px;}"
        "QMenu::item:selected{background-color:rgba(255,255,255,242);color:#000000;}"
        "QMenu::separator{height:1px;background:#ffffff;margin:2px 4px;}"
    )

    def __init__(self, mw, scene, view, logo_path, music_path, svg_path):
        super().__init__()
        self.mw = mw
        self.scene = scene
        self.view = view
        self.logo_path = logo_path
        self.music_path = music_path
        self.svg_path = svg_path
        self.terminal = None
        self.terminal_proxy = None
        self.welcome_screen = None
        self.music_player = None
        self.audio_output = None
        self.cursor_item = None
        self._fake_menu = None
        self._nta = None
        self._debug_timer = None
        self._btn_proxy = None  # the Hello World button proxy
        self._btn_widget = None
        self._fake_menu2 = None  # second menu for "Show AI Voice"
        self._voice_action = None
        self._eyes_widget = None
        self._eyes_proxy = None
        self._keep = []
        self._spawned_widgets = []  # (widget, proxy) pairs for dark mode switching

    # --- helpers ---

    def _c(self):
        return self.view.mapToScene(self.view.viewport().rect().center())

    def _cleanup_stopped_timers(self):
        """Remove stopped QTimer objects from _keep to free memory."""
        self._keep = [t for t in self._keep if hasattr(t, 'isActive') and t.isActive()]

    def _fin(self, proxy, ms, cb=None):
        proxy.setOpacity(0.0)
        dur = ms / 1000.0; t0 = time.perf_counter()
        def t():
            p = min(1.0, (time.perf_counter() - t0) / dur)
            p = 1 - pow(1 - p, 3)  # ease-out cubic
            proxy.setOpacity(p)
            if p >= 1.0:
                ti.stop()
                if cb: cb()
        ti = QTimer(self); ti.timeout.connect(t); ti.start(16); self._keep.append(ti)

    def _fout(self, proxy, ms, cb=None):
        o = proxy.opacity(); dur = ms / 1000.0; t0 = time.perf_counter()
        def t():
            p = min(1.0, (time.perf_counter() - t0) / dur)
            p = 1 - pow(1 - p, 3)
            proxy.setOpacity(o * (1 - p))
            if p >= 1.0:
                ti.stop()
                if cb: cb()
        ti = QTimer(self); ti.timeout.connect(t); ti.start(16); self._keep.append(ti)

    def _stream(self, text, color, ms=25, cb=None):
        """Stream text one character at a time, paced by wall-clock time.
        
        *ms* is the target interval per character.  Instead of relying on
        QTimer firing exactly every *ms* milliseconds, we derive the current
        character index from perf_counter so that the total duration is
        always len(text)*ms regardless of event-loop jitter.
        """
        color = color or self.terminal.C_DEFAULT
        t0 = time.perf_counter()
        interval_s = ms / 1000.0
        emitted = [0]  # characters emitted so far

        def t():
            elapsed = time.perf_counter() - t0
            target_idx = min(len(text), int(elapsed / interval_s) + 1)
            # Emit all characters we should have shown by now
            while emitted[0] < target_idx:
                self.terminal.append_text(text[emitted[0]], color)
                emitted[0] += 1
            if emitted[0] >= len(text):
                ti.stop()
                if cb: cb()

        # Wake up at roughly half the per-character interval for smoothness
        wake = max(1, ms // 2)
        ti = QTimer(self); ti.timeout.connect(t); ti.start(wake); self._keep.append(ti)

    def _type(self, text, color, ms=35, cb=None):
        self._stream(text, color, ms, cb)

    def _at_music_time(self, target_s, callback):
        """Schedule *callback* at exactly *target_s* seconds from music start.
        
        If we're already past that time, fire immediately.
        """
        elapsed = time.perf_counter() - self._music_t0
        remaining_ms = max(0, (target_s - elapsed) * 1000)
        if remaining_ms <= 1:
            callback()
        else:
            _precise_singleshot(remaining_ms, callback)

    def _lines(self, lines, delay=150, cb=None):
        i = [0]
        def nx():
            if i[0] < len(lines):
                t, c = lines[i[0]]; self.terminal.append_output(t, color=c); i[0] += 1
                _precise_singleshot(delay, nx, self)
            elif cb: cb()
        nx()

    # --- entry ---

    def start(self):
        # Enter fullscreen before placing any widgets
        if not self.mw.isFullScreen():
            self.mw.showFullScreen()
        # Wait for the view to settle after fullscreen resize,
        # then re-query the center and place the button
        _precise_singleshot(300, self._place_start_button)

    def _place_start_button(self):
        """Create the Start button at the actual view center (after fullscreen settles)."""
        c = self._c()
        btn = QPushButton("Start"); btn.setFixedSize(180, 56); btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("QPushButton{background:#000;color:#fff;font-size:20px;font-weight:600;"
            "font-family:'Consolas','Monaco',monospace;border:none;border-radius:10px;}"
            "QPushButton:hover{background:#222;}QPushButton:pressed{background:#444;}")
        btn.clicked.connect(self._on_start)
        self._sp = self.scene.addWidget(btn); self._sp.setZValue(2000)
        self._sp.setPos(c.x() - 90, c.y() - 28)
        sh = QGraphicsDropShadowEffect(); sh.setBlurRadius(30); sh.setColor(QColor(0, 0, 0, 60)); sh.setOffset(0, 8)
        self._sp.setGraphicsEffect(sh); self._fin(self._sp, 600)

    def _on_start(self):
        self._music_t0 = time.perf_counter()  # master clock reference
        self._debug_timer=None  # DebugTimerOverlay disabled
        self._fout(self._sp,300,cb=lambda:(self.scene.removeItem(self._sp) if self._sp.scene() else None))
        self._start_music()
        self.welcome_screen=WelcomeScreen(self.logo_path)
        self.welcome_screen.finished.connect(self._on_welcome_done)
        _precise_singleshot(200, self.welcome_screen.show_welcome)

    def _start_music(self):
        if not os.path.exists(self.music_path): return
        try:
            self.audio_output=QAudioOutput(); self.audio_output.setVolume(0.7)
            self.music_player=QMediaPlayer(); self.music_player.setAudioOutput(self.audio_output)
            self.music_player.setSource(QUrl.fromLocalFile(self.music_path)); self.music_player.play()
        except Exception as e: print(f"[Onboarding] Music: {e}")

    def _on_welcome_done(self):
        if self.welcome_screen: self.welcome_screen.deleteLater(); self.welcome_screen=None
        _precise_singleshot(500, self._s_cursor)

    # ==================================================================
    # CHAIN
    # ==================================================================

    def _s_cursor(self):
        # Pre-calculate terminal start position so cursor matches exactly
        c=self._c()
        self._tw=680; self._th=460
        self._ts=QPointF(c.x()-self._tw/2-150, c.y()-self._th/2)
        # Cursor appears at the terminal's top-left origin (matches mouse-drawn behavior)
        self.cpos=QPointF(self._ts.x(), self._ts.y())
        self.cursor_item=self.scene.addEllipse(-6,-6,12,12,QPen(QColor(0,0,0,200),2),QBrush(QColor(0,0,0,120)))
        self.cursor_item.setZValue(3000); self.cursor_item.setPos(self.cpos)
        _precise_singleshot(400, self._s_menu)

    def _s_menu(self):
        self._fake_menu=QMenu(self.mw); self._fake_menu.setStyleSheet(self._CSS_NORMAL)
        self._nta=self._fake_menu.addAction("New Terminal")
        self._fake_menu.addSeparator()
        self._fake_menu.addAction("Show AI Voice"); self._fake_menu.addAction("Onboarding")
        self._fake_menu.addSeparator(); self._fake_menu.addAction("Dark Mode")
        self._fake_menu.addSeparator(); self._fake_menu.addAction("Immersive Mode (Ctrl+I)")
        self._fake_menu.addSeparator()
        self._fake_menu.addAction("Clear Scene"); self._fake_menu.addAction("Refresh")
        self._fake_menu.addAction("Delete Widget"); self._fake_menu.addAction("Pop Widget")
        self._fake_menu.addSeparator(); self._fake_menu.addAction("Fullscreen")
        self._fake_menu.addSeparator(); self._fake_menu.addAction("Exit")
        vp=self.view.mapFromScene(self.cpos); gp=self.view.mapToGlobal(vp)
        self._fake_menu.popup(gp)
        _precise_singleshot(800, self._s_hl)

    def _s_hl(self):
        if self._fake_menu and self._nta:
            r=self._fake_menu.actionGeometry(self._nta)
            QCursor.setPos(self._fake_menu.mapToGlobal(r.center()))
        _precise_singleshot(500, self._s_blink)

    def _s_blink(self):
        if not self._fake_menu: _precise_singleshot(10,self._s_xhair); return
        m=self._fake_menu
        blink_t0 = time.perf_counter()
        phase = [0]  # track which phase we've executed
        def t():
            elapsed_ms = (time.perf_counter() - blink_t0) * 1000
            if phase[0] == 0 and elapsed_ms >= 80:
                m.setStyleSheet(self._CSS_FLASH); phase[0] = 1
            elif phase[0] == 1 and elapsed_ms >= 160:
                m.setStyleSheet(self._CSS_NORMAL); phase[0] = 2
            elif phase[0] == 2 and elapsed_ms >= 240:
                ti.stop(); _precise_singleshot(150, self._s_close_menu)
        ti=QTimer(self); ti.timeout.connect(t); ti.start(16); self._keep.append(ti)

    def _s_close_menu(self):
        if self._fake_menu: self._fake_menu.close(); self._fake_menu.deleteLater(); self._fake_menu=None
        self._nta=None; _precise_singleshot(200,self._s_xhair)

    def _s_xhair(self):
        if self.cursor_item and self.cursor_item.scene(): self.scene.removeItem(self.cursor_item)
        pen=QPen(QColor(0,0,0,200),1.5); cx,cy=self.cpos.x(),self.cpos.y()
        self._xh=self.scene.addLine(cx-12,cy,cx+12,cy,pen)
        self._xv=self.scene.addLine(cx,cy-12,cx,cy+12,pen)
        self._xh.setZValue(3000); self._xv.setZValue(3000)
        _precise_singleshot(500,self._s_draw)

    # --- terminal ---

    def _s_draw(self):
        from rio.terminal_widget import TerminalWidget
        # _tw, _th, _ts already set in _s_cursor
        self.terminal=TerminalWidget(llmfs_mount=self.mw.rio_server.llmfs_mount, rio_mount=self.mw.rio_server.rio_mount)
        self.terminal.resize(100,150)
        self.terminal.setAttribute(Qt.WA_TranslucentBackground,True); self.terminal.setAutoFillBackground(False)
        self.terminal_proxy=self.scene.addWidget(self.terminal,Qt.Widget)
        self.terminal_proxy.setAutoFillBackground(False)
        self.terminal_proxy.setPos(self._ts.x(),self._ts.y())
        self.terminal_proxy.setAcceptedMouseButtons(Qt.NoButton); self.terminal_proxy.setZValue(1000)
        self.terminal.show()
        self.mw.terminals.append(self.terminal)
        self.terminal.command_submitted.connect(self.mw._execute_command)
        _draw_dur = 40 * 16 / 1000.0  # ~640 ms total (original: 40 steps × 16 ms)
        _draw_t0 = time.perf_counter()
        def tick():
            elapsed = time.perf_counter() - _draw_t0
            p = min(1.0, elapsed / _draw_dur)
            p = 1 - pow(1 - p, 3)  # ease-out cubic
            w = max(100, int(self._tw * p)); h = max(150, int(self._th * p))
            self.terminal.resize(w, h)
            ex = self._ts.x() + w; ey = self._ts.y() + h
            if hasattr(self, '_xh') and self._xh.scene():
                self._xh.setLine(ex-12, ey, ex+12, ey); self._xv.setLine(ex, ey-12, ex, ey+12)
            if p >= 1.0: ti.stop(); _precise_singleshot(200, self._s_fin_term)
        ti = QTimer(self); ti.timeout.connect(tick); ti.start(16); self._keep.append(ti)

    def _s_fin_term(self):
        for it in [getattr(self,'_xh',None),getattr(self,'_xv',None)]:
            if it and it.scene(): self.scene.removeItem(it)

        # Match main.py _handle_terminal_mouse_release exactly:
        # 1. Set proxy ref BEFORE show_content
        self.terminal._proxy = self.terminal_proxy

        # 2. Final resize + position (mirror the frame_rect.normalized() path)
        self.terminal.resize(self._tw, self._th)
        self.terminal_proxy.setPos(self._ts.x(), self._ts.y())

        # 3. show_content() — this shows scroll/input, animates shadow, sets resize state
        self.terminal.show_content()

        # 4. Dark mode if active
        if self.mw._dark_mode:
            self.terminal.set_dark_mode(True, duration_steps=1)

        # 5. Lock size so layouts don't collapse
        self.terminal.setFixedSize(self._tw, self._th)

        # 6. Re-enable proxy mouse events
        self.terminal_proxy.setAcceptedMouseButtons(
            Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
        )

        # 7. ItemIsSelectable flag (matches main.py)
        from PySide6.QtWidgets import QGraphicsItem
        self.terminal_proxy.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # 8. Z-value
        self.terminal_proxy.setZValue(1000)

        # 9. Register in filesystem
        if self.mw.rio_server.filesystem and hasattr(self.mw.rio_server.filesystem, 'terms_dir'):
            import weakref
            try:
                td = self.mw.rio_server.filesystem.terms_dir.register_terminal(
                    self.terminal.term_id, weakref.ref(self.terminal)
                )
                self.terminal._term_dir = td
            except Exception:
                pass

        _precise_singleshot(500, self._z00_move_cursor_away)

    def _z00_move_cursor_away(self):
        """Move the mouse cursor to the bottom-right of the viewport so it's out of the way."""
        vp = self.view.viewport().rect()
        bottom_right = self.view.mapToGlobal(QPoint(vp.width() - 80, vp.height() - 80))
        QCursor.setPos(bottom_right)
        _precise_singleshot(300, self._z01)

    # ==================================================================
    # STREAMING — tuned for button morph done at 48s
    # ==================================================================

    def _z01(self):  # welcome: 206×18ms + 300ms = 4.0s
        self._stream(
            "\n Welcome to Peribus\n ─────────────────────────────\n\n"
            " Peribus is a Plan 9-inspired graphical workspace\n"
            " where everything is a file. Terminals, AI agents,\n"
            " widgets — all accessible through a 9P filesystem.\n\n",
            self.terminal.C_INFO, 18,
            cb=lambda: _precise_singleshot(300, self._z02))

    def _z02(self):  # features: 110×22ms + 300ms = 2.7s
        self._stream(
            " This terminal is your control surface.\n"
            " Type commands, talk to AI agents, or\n"
            " execute Python code directly.\n\n",
            self.terminal.C_DEFAULT, 22,
            cb=lambda: _precise_singleshot(300, self._z03))

    def _z03(self):  # modes: 6×180ms + 300ms = 1.4s
        S=self.terminal.C_SYSTEM
        self._lines([
            (" ╭── Modes ──────────────────────────────────╮\n",S),
            (" │  /      — Macro commands                  │\n",S),
            (" │  $      — Shell mode (toggle)             │\n",S),
            (" │  >>>    — Python execution                │\n",S),
            (" │  text   — AI prompt (when connected)      │\n",S),
            (" ╰───────────────────────────────────────────╯\n",S),
        ], 180, cb=lambda: _precise_singleshot(300, self._z04))

    def _z04(self):  # resize: 122×18ms + 300ms = 2.5s
        self._stream(
            " Drag corners to resize any terminal.\n"
            " Ctrl+Click to move widgets on the canvas.\n"
            " Ctrl+Scroll to zoom. Ctrl+Drag to pan.\n\n",
            self.terminal.C_INFO, 18,
            cb=lambda: _precise_singleshot(300, self._z05))

    def _z05(self):  # panels: 113×18ms + 400ms = 2.4s
        self._stream(
            " Terminals have side panels for nodes,\n"
            " version control, and operator graphs.\n\n"
            " Now let's create an AI agent...\n\n",
            self.terminal.C_INFO, 18,
            cb=lambda: _precise_singleshot(400, self._z06))

    def _z06(self):  # /coder: 6×70ms + 250ms = 0.7s
        self.terminal.append_output("\n")
        self._type("/coder", self.terminal.C_MACRO, 70,
            cb=lambda: _precise_singleshot(250, self._z07))

    def _z07(self):  # agent output: 8×120ms + 400ms = 1.4s
        I=self.terminal.C_INFO; S=self.terminal.C_SUCCESS; D=self.terminal.C_DEFAULT
        self._lines([
            ("\n",D), ("✓ Agent 'coder' created\n",S),
            ("  provider: anthropic\n",I), ("  model:    claude-sonnet-4-20250514\n",I),
            ("  system:   You are a helpful coding assistant.\n",I),
            ("\n",D), ("Connected to coder\n",S), ("\n",D),
        ], 120, cb=lambda: _precise_singleshot(400, self._z08))

    def _z08(self):  # prompt: 42×25ms + 400ms = 1.5s
        self._type("Create a Hello World button on the canvas",
            self.terminal.C_USER, 25,
            cb=lambda: (self.terminal.append_output("\n\n"), _precise_singleshot(400, self._z09)))

    def _z09(self):  # LLM response: 578×16ms = 9.2s + 200ms → button at ~41.5s
        resp = (
            "I'll create a button widget on your canvas. Here's the code:\n\n"
            "```rioa\n"
            "from PySide6.QtWidgets import QPushButton\n"
            "from PySide6.QtCore import Qt\n\n"
            "btn = QPushButton('Hello World!')\n"
            "btn.setFixedSize(200, 60)\n"
            "btn.setStyleSheet('''\n"
            "    QPushButton {\n"
            "        background: #000;\n"
            "        color: #fff;\n"
            "        font-size: 16px;\n"
            "        font-weight: bold;\n"
            "        border-radius: 10px;\n"
            "        font-family: monospace;\n"
            "    }\n"
            "    QPushButton:hover {\n"
            "        background: #333;\n"
            "    }\n"
            "''')\n"
            "proxy = graphics_scene.addWidget(btn)\n"
            "proxy.setPos(400, 200)\n"
            "```\n\n"
            "The button has been placed on your canvas!"
        )
        self._stream(resp, self.terminal.C_AGENT, 16,
            cb=lambda: _precise_singleshot(200, self._z10_button))

    # ==================================================================
    # BUTTON APPEARS @ ~41.5s
    # ==================================================================

    def _z10_button(self):
        print("[Onboarding] → button @ ~41.5s")
        self._btn_widget = QPushButton("Hello World!")
        self._btn_widget.setFixedSize(220, 64)
        self._btn_widget.setCursor(Qt.PointingHandCursor)
        self._btn_widget.setStyleSheet(
            "QPushButton{background:#000;color:#fff;font-size:18px;font-weight:bold;"
            "border-radius:12px;font-family:'Consolas','Monaco',monospace;}"
            "QPushButton:hover{background:#333;}QPushButton:pressed{background:#555;}")
        term=self.terminal
        self._btn_widget.clicked.connect(lambda: term.append_output(
            "\n✨ Hello World!\n",color=term.C_SUCCESS) if term else None)
        self._btn_proxy=self.scene.addWidget(self._btn_widget); self._btn_proxy.setZValue(1000)
        if self.terminal_proxy:
            tp=self.terminal_proxy.pos()
            self._btn_proxy.setPos(tp.x()+self._tw+60, tp.y()+self._th/2-32)
        sh=QGraphicsDropShadowEffect(); sh.setBlurRadius(25); sh.setColor(QColor(0,0,0,100)); sh.setOffset(8,8)
        self._btn_proxy.setGraphicsEffect(sh)
        self._fin(self._btn_proxy, 500, cb=lambda: _precise_singleshot(300, self._z11_explain))

    # ==================================================================
    # EXPLAIN + MORPH BUTTON → morph done at 48s
    # ==================================================================

    def _z11_explain(self):  # 77×18ms + 200ms = 1.6s
        self.terminal.append_output("\n\n✓ Widget placed on canvas.\n\n", color=self.terminal.C_SUCCESS)
        self._stream(
            " You can even use and move the button\n"
            " with your mouse, while we modify it.\n\n",
            self.terminal.C_INFO, 18,
            cb=lambda: _precise_singleshot(200, self._z12_ask_coder))

    def _z12_ask_coder(self):  # 38×20ms + 150ms = 0.9s
        self._stream(
            " Here, let's ask coder to improve it:\n",
            self.terminal.C_INFO, 20,
            cb=lambda: _precise_singleshot(150, self._z13_morph_prompt))

    def _z13_morph_prompt(self):  # 47×28ms + 300ms pause + 150ms = 1.8s
        self._type(
            "Make the button transparent with shadow effects",
            self.terminal.C_USER, 28,
            cb=lambda: (self.terminal.append_output("\n\n"),
                        _precise_singleshot(300, self._z14_morph_response)))

    def _z14_morph_response(self):  # 57×14ms + 200ms = 1.0s
        self._stream(
            "Sure! Updating the button with transparency and shadow...\n",
            self.terminal.C_AGENT, 14,
            cb=lambda: _precise_singleshot(200, self._z15_morph_button))

    def _z15_morph_button(self):  # 600ms animation
        """Animate button style change: transparent + border + shadow."""
        print("[Onboarding] → morph button")
        if not self._btn_widget:
            _precise_singleshot(100, self._z16_pre_svg)
            return

        # Target style from user spec:
        # background: transparent, border: 2px solid black, color: black
        # shadow: offset(45,45), blur 20, color rgba(0,0,0,180)
        morph_dur = 30 * 20 / 1000.0  # ~600 ms (original: 30 steps × 20 ms)
        morph_t0 = time.perf_counter()
        def tick():
            p = min(1.0, (time.perf_counter() - morph_t0) / morph_dur)
            # Interpolate: bg alpha 255→0, text #fff→#000, border 0→2px
            bg_a = int(255 * (1.0 - p))
            # Text color: white (255)→black (0)
            tc = int(255 * (1.0 - p))
            # Border: fade in from 0 to 2px solid black
            bw = p * 2.0
            # Border radius: 12→4
            br = int(12 - 8 * p)

            self._btn_widget.setStyleSheet(
                f"QPushButton{{background-color:rgba(0,0,0,{bg_a});"
                f"border:{bw:.1f}px solid rgba(0,0,0,{int(255*p)});"
                f"color:rgb({tc},{tc},{tc});"
                f"font-size:16px;font-weight:bold;"
                f"border-radius:{br}px;}}"
                f"QPushButton:hover{{background-color:rgba(0,0,0,{int(20*p)});}}"
                f"QPushButton:pressed{{background-color:rgba(0,0,0,{int(40*p)});}}")

            if self._btn_proxy:
                sh=QGraphicsDropShadowEffect()
                # Shadow: interpolate offset from (8,8)→(45,45), blur 25→20
                sh.setOffset(8 + 37 * p, 8 + 37 * p)
                sh.setBlurRadius(int(25 - 5 * p))
                sh.setColor(QColor(0, 0, 0, int(100 + 80 * p)))
                self._btn_proxy.setGraphicsEffect(sh)

            if p >= 1.0:
                ti.stop()
                self.terminal.append_output(
                    "\n✓ Button updated with new style.\n",
                    color=self.terminal.C_SUCCESS)
                _precise_singleshot(100, self._z16_pre_svg)
        ti=QTimer(self); ti.timeout.connect(tick); ti.start(20); self._keep.append(ti)

    # ==================================================================
    # SVG HAND-DRAW @ ~49s
    # ==================================================================

    def _z16_pre_svg(self):
        """Poetic agent lines before SVG appears — the 'Both' reveal."""
        print("[Onboarding] → pre-SVG poetic lines")
        self.terminal.append_output("\n\n")
        self._stream(
            "You just created a live element on screen.\n",
            self.terminal.C_AGENT, 22,
            cb=lambda: _precise_singleshot(400, self._z16b_line2))

    def _z16b_line2(self):
        self._stream(
            "Code became something you can touch, move, feel.\n",
            self.terminal.C_AGENT, 22,
            cb=lambda: _precise_singleshot(400, self._z16c_line3))

    def _z16c_line3(self):
        self._stream(
            "Is this interface made for a human, or a machine?\n",
            self.terminal.C_AGENT, 22,
            cb=lambda: _precise_singleshot(600, self._z18_voice_menu))

    # ==================================================================
    # SVG BESIDE EYES + VIEW TILT @ 53.5s
    # ==================================================================

    def _z24_svg_beside_eyes(self):
        """Place SVG next to the eyes widget at 53.5s, start hand-draw,
        and begin the slow view tilt/zoom-out to 1m20s."""
        print("[Onboarding] → SVG beside eyes + view tilt @ 53.5s")

        if not os.path.exists(self.svg_path):
            print(f"[Onboarding] SVG not found: {self.svg_path}")
            return

        self._svg_hand = HandDrawSvgWidget(self.svg_path)

        self._svg_proxy = self.scene.addWidget(self._svg_hand)
        self._svg_proxy.setZValue(1500)

        # Position SVG to the right of the eyes widget
        if hasattr(self, '_eyes_proxy') and self._eyes_proxy:
            ep = self._eyes_proxy.pos()
            ew = self._eyes_widget.width() if self._eyes_widget else 280
            eh = self._eyes_widget.height() if self._eyes_widget else 140
            svg_x = ep.x() + ew + 30  # 30 px gap to the right of eyes
            svg_y = ep.y() + eh / 2 - self._svg_hand._display_h / 2  # vertically centred
            self._svg_proxy.setPos(svg_x, svg_y)
        else:
            # Fallback: next to terminal top
            c = self._c()
            self._svg_proxy.setPos(c.x() + 200, c.y() - 200)

        # Fade in + hand-draw simultaneously
        self._fin(self._svg_proxy, 400)
        self._svg_hand.start_animation(
            duration_ms=3500,
            callback=self._z25_svg_shadow
        )

        # Eyes wink once as SVG becomes visible (~800ms after appearance)
        _precise_singleshot(800, self._trigger_eyes_wink)

        self.terminal.append_output(
            "\n ✎ Drawing...\n",
            color=self.terminal.C_INFO
        )

        # Start the slow view tilt simultaneously (53.5s → 1m20s = 26.5s)
        self._start_view_tilt(duration_s=26.5)

        # Start the "serious work" agent sequence after a short beat
        _precise_singleshot(1500, self._w01_serious_intro)

        print("[Onboarding] ✓ SVG + tilt + work sequence started")

    def _start_view_tilt(self, duration_s=29.0):
        """Slowly animate the view from current transform to a Ctrl+9-style
        top-right perspective tilt with zoom-out.
        
        Target values modelled on _centered_tilt_transform(sx=0.5, sy=0.4, shy=-0.15)
        and the orbit controller logic from main.py.
        """
        from PySide6.QtGui import QTransform

        view = self.view
        vp = view.viewport()
        half_w = vp.width() / 2.0
        half_h = vp.height() / 2.0

        # Capture starting transform
        start_t = view.transform()
        # Extract base scale from start
        base_sx = (start_t.m11()**2 + start_t.m21()**2) ** 0.5
        base_sy = (start_t.m12()**2 + start_t.m22()**2) ** 0.5
        if base_sx < 0.001: base_sx = 1.0
        if base_sy < 0.001: base_sy = 1.0

        # Capture the scene centre we want to keep steady
        center_scene = view.mapToScene(view.viewport().rect().center())

        # Target parameters (Ctrl+9 style top-right tilt)
        target_sx = 0.5
        target_sy = 0.4
        target_shy = -0.15
        target_shx = 0.0

        # Start values (identity-like: no shear)
        start_sx = base_sx
        start_sy = base_sy
        start_shx = 0.0
        start_shy = 0.0

        t0 = time.perf_counter()

        def tick():
            elapsed = time.perf_counter() - t0
            p = min(1.0, elapsed / duration_s)
            # Smooth ease-in-out (cubic)
            if p < 0.5:
                ep = 4 * p * p * p
            else:
                ep = 1 - pow(-2 * p + 2, 3) / 2

            # Interpolate all parameters
            sx = start_sx + (target_sx - start_sx) * ep
            sy = start_sy + (target_sy - start_sy) * ep
            shx = start_shx + (target_shx - start_shx) * ep
            shy = start_shy + (target_shy - start_shy) * ep

            # Build transform centred on viewport middle
            cx = half_w
            cy = half_h
            t = QTransform()
            t.translate(cx, cy)
            t.scale(sx, sy)
            t.shear(shx, shy)
            t.translate(-cx, -cy)

            view.setTransform(t)
            view.centerOn(center_scene)

            if p >= 1.0:
                ti.stop()
                print("[Onboarding] ✓ View tilt complete @ 1m20s")
                _precise_singleshot(500, self._m01_dark_mode)

        ti = QTimer(self); ti.timeout.connect(tick); ti.start(25); self._keep.append(ti)

    def _z25_svg_shadow(self):
        """SVG finished drawing — add a drop shadow and animate offset 0,0 → 45,45."""
        print("[Onboarding] → SVG shadow animation")

        if not hasattr(self, '_svg_proxy') or not self._svg_proxy:
            return

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 0)  # start at no offset
        self._svg_proxy.setGraphicsEffect(shadow)
        self._svg_shadow = shadow

        # Animate offset from (0,0) to (45,45) over 1.5s
        dur_s = 1.5
        t0 = time.perf_counter()

        def tick():
            elapsed = time.perf_counter() - t0
            p = min(1.0, elapsed / dur_s)
            # Ease-out cubic
            ep = 1 - pow(1 - p, 3)
            offset = 45.0 * ep
            self._svg_shadow.setOffset(offset, offset)
            if p >= 1.0:
                ti.stop()
                print("[Onboarding] ✓ SVG shadow animation complete")

        ti = QTimer(self); ti.timeout.connect(tick); ti.start(25); self._keep.append(ti)

    # ==================================================================
    # "SERIOUS WORK" — agent code + widget spawning during view tilt
    # ==================================================================
    # Style: transparent bg, 2px solid black border, shadow 45,45 blur 20
    # Streaming: fast (5-7ms/char for code, 15ms for prompts)
    # Eyes: animate position near some spawning widgets

    _WIDGET_CSS = (
        "background: transparent;"
        "border: 2px solid #000;"
        "border-radius: 6px;"
        "font-family: 'Consolas','Monaco',monospace;"
        "color: #000;"
    )

    @staticmethod
    def _wfg(widget):
        """Return foreground QColor for a custom-painted widget (dark_mode aware)."""
        if widget.property("dark_mode"):
            return QColor(230, 230, 230)
        return QColor(0, 0, 0)

    @staticmethod
    def _wfga(widget, alpha):
        """Return foreground QColor with alpha for a custom-painted widget."""
        if widget.property("dark_mode"):
            return QColor(230, 230, 230, alpha)
        return QColor(0, 0, 0, alpha)

    def _spawn_widget(self, widget, x, y, fade_ms=400):
        """Add widget to scene with standard transparent+border+shadow style."""
        proxy = self.scene.addWidget(widget)
        proxy.setZValue(900)
        proxy.setPos(x, y)
        sh = QGraphicsDropShadowEffect()
        sh.setBlurRadius(20)
        sh.setColor(QColor(0, 0, 0, 180))
        sh.setOffset(45, 45)
        proxy.setGraphicsEffect(sh)
        self._fin(proxy, fade_ms)
        self._keep.append(proxy)
        self._spawned_widgets.append((widget, proxy))
        return proxy

    def _animate_eyes_to(self, target_x, target_y, duration_ms=800):
        """Smoothly move the eyes widget to a position near a spawning widget,
        and also direct the pupils (gaze) toward the target location."""
        if not hasattr(self, '_eyes_proxy') or not self._eyes_proxy:
            return
        start_x = self._eyes_proxy.x()
        start_y = self._eyes_proxy.y()
        dur = duration_ms / 1000.0
        t0 = time.perf_counter()
        def tick():
            p = min(1.0, (time.perf_counter() - t0) / dur)
            ep = 1 - pow(1 - p, 3)
            self._eyes_proxy.setPos(
                start_x + (target_x - start_x) * ep,
                start_y + (target_y - start_y) * ep
            )
            if p >= 1.0: ti.stop()
        ti = QTimer(self); ti.timeout.connect(tick); ti.start(16); self._keep.append(ti)

        # Also direct the pupils toward the target scene position
        self._direct_pupils_toward(target_x, target_y)

    def _direct_pupils_toward(self, scene_x, scene_y):
        """Gently direct the pupils toward a scene coordinate without eyebrow theatrics."""
        if not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return
        if not hasattr(self, '_eyes_proxy') or not self._eyes_proxy:
            return
        import math
        ew = self._eyes_widget
        ep = self._eyes_proxy.pos()
        eye_cx = ep.x() + ew.width() / 2
        eye_cy = ep.y() + ew.height() / 2
        dx = scene_x - eye_cx
        dy = scene_y - eye_cy
        dist = math.sqrt(dx * dx + dy * dy) if (dx or dy) else 1.0
        max_gaze = 14.0
        closeness = min(dist / 500, 1.0)
        fx = (dx / dist) * max_gaze * closeness if dist > 0 else 0.0
        fy = (dy / dist) * max_gaze * 0.6 * closeness if dist > 0 else 0.0
        try:
            ew._gaze_target_x = fx
            ew._gaze_target_y = fy
            ew._gaze_speed = 0.10
        except (RuntimeError, AttributeError):
            pass

    def _tp(self):
        """Terminal position helper."""
        if self.terminal_proxy:
            return self.terminal_proxy.pos()
        c = self._c()
        return QPointF(c.x() - self._tw / 2, c.y() - self._th / 2)

    def _w01_serious_intro(self):
        print("[Onboarding] → serious work sequence")
        self.terminal.append_output("\n\n")
        self._stream(
            "Let's start serious work.\n\n",
            self.terminal.C_AGENT, 18,
            cb=lambda: _precise_singleshot(300, self._w02_calendar))

    # ── 1. Calendar ──────────────────────────────────────────────────

    def _w02_calendar(self):
        self._type("Create a calendar", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(200, self._w02b_calendar_code)))

    def _w02b_calendar_code(self):
        code = (
            "```python\n"
            "cal = QCalendarWidget()\n"
            "cal.setGridVisible(True)\n"
            "cal.setSelectedDate(QDate.currentDate())\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w02c_calendar_spawn))

    def _w02c_calendar_spawn(self):
        from PySide6.QtWidgets import QCalendarWidget
        from PySide6.QtCore import QDate

        cal = QCalendarWidget()
        cal.setFixedSize(340, 250)
        cal.setGridVisible(True)
        cal.setSelectedDate(QDate.currentDate())
        cal.setStyleSheet(
            "QCalendarWidget{background:transparent;border:2px solid #000;border-radius:6px;"
            "font-family:monospace;font-size:11px;color:#000;}"
            "QCalendarWidget QToolButton{color:#000;font-weight:bold;background:transparent;border:none;}"
            "QCalendarWidget QWidget#qt_calendar_navigationbar{background:transparent;border-bottom:1px solid #000;}"
            "QCalendarWidget QAbstractItemView{selection-background-color:#000;selection-color:#fff;}"
        )

        tp = self._tp()
        self._spawn_widget(cal, tp.x() + self._tw + 200, tp.y() + self._th - 100)

        self.terminal.append_output("✓ Calendar\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w03_color_wheel)

    # ── 2. Color Wheel ───────────────────────────────────────────────

    def _w03_color_wheel(self):
        self._type("Color picker", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w03b_color_code)))

    def _w03b_color_code(self):
        code = (
            "```python\n"
            "wheel = ColorWheelWidget()\n"
            "wheel.setFixedSize(220, 280)\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w03c_color_spawn))

    def _w03c_color_spawn(self):
        """A color wheel with HSV ring drawn via QPainter and a preview swatch."""
        import math

        class ColorWheelWidget(QWidget):
            def __init__(self):
                super().__init__()
                self.setFixedSize(220, 280)
                self._hue = 0.0
                self._sat = 1.0
                self._val = 1.0
                self.setAttribute(Qt.WA_TranslucentBackground, True)

            def paintEvent(self, event):
                from PySide6.QtGui import QPainter, QConicalGradient, QRadialGradient
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                dark = self.property("dark_mode")
                fg = QColor(230, 230, 230) if dark else QColor(0, 0, 0)
                border = QColor(255, 255, 255, 150) if dark else QColor(0, 0, 0)

                # Border
                p.setPen(QPen(border, 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(1, 1, self.width()-2, self.height()-2, 6, 6)

                # Color wheel
                cx, cy, r = 110, 110, 80
                cg = QConicalGradient(cx, cy, 0)
                for i in range(13):
                    c = QColor.fromHsvF(i / 12.0, 1.0, 1.0)
                    cg.setColorAt(i / 12.0, c)

                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(cg))
                p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

                # White→transparent radial overlay for saturation
                rg = QRadialGradient(cx, cy, r)
                rg.setColorAt(0, QColor(255, 255, 255, 255))
                rg.setColorAt(1, QColor(255, 255, 255, 0))
                p.setBrush(QBrush(rg))
                p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

                # Center dot
                p.setBrush(QBrush(fg))
                p.drawEllipse(cx - 3, cy - 3, 6, 6)

                # Current color preview swatch
                p.setPen(QPen(border, 2))
                preview = QColor.fromHsvF(self._hue, self._sat, self._val)
                p.setBrush(QBrush(preview))
                p.drawRoundedRect(20, 210, 180, 50, 4, 4)

                # Label
                p.setPen(fg)
                p.setFont(QFont("Consolas", 9))
                p.drawText(20, 278, f"HSV: {int(self._hue*360)}° {int(self._sat*100)}% {int(self._val*100)}%")
                p.end()

        wheel = ColorWheelWidget()
        tp = self._tp()
        self._spawn_widget(wheel, tp.x() - 480, tp.y() - 120)

        # Eyes glance toward color wheel — keep well above the widget
        self._animate_eyes_to(tp.x() - 300, tp.y() - 320, 600)

        self.terminal.append_output("✓ Color picker\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(500, self._w04_dashboard)

    # ── 3. System Dashboard ──────────────────────────────────────────

    def _w04_dashboard(self):
        self._type("System dashboard", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w04b_dash_code)))

    def _w04b_dash_code(self):
        code = (
            "```python\n"
            "panel = SystemDashboard()\n"
            "panel.setFixedSize(280, 200)\n"
            "# CPU, Memory, Disk, Network bars\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w04c_dash_spawn))

    def _w04c_dash_spawn(self):
        from PySide6.QtWidgets import QProgressBar, QHBoxLayout

        panel = QWidget()
        panel.setFixedSize(280, 220)
        panel.setStyleSheet(f"QWidget{{{self._WIDGET_CSS}}}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 10, 14, 10)

        title = QLabel("System Monitor")
        title.setStyleSheet("font-size:13px;font-weight:bold;border:none;color:#000;background:transparent;")
        layout.addWidget(title)

        for name, val, color in [
            ("CPU", 73, "#000"), ("Memory", 58, "#333"),
            ("Disk I/O", 34, "#555"), ("Network", 91, "#000")
        ]:
            row = QWidget(); row.setStyleSheet("border:none;background:transparent;")
            rl = QHBoxLayout(row); rl.setContentsMargins(0, 1, 0, 1)
            lbl = QLabel(name)
            lbl.setStyleSheet("font-size:10px;color:#000;border:none;background:transparent;min-width:55px;")
            lbl.setFixedWidth(55)
            bar = QProgressBar()
            bar.setValue(val); bar.setFixedHeight(12); bar.setFormat(f"{val}%")
            bar.setStyleSheet(
                f"QProgressBar{{border:1px solid #000;border-radius:3px;background:transparent;text-align:center;font-size:8px;color:#000;}}"
                f"QProgressBar::chunk{{background:{color};border-radius:2px;}}"
            )
            rl.addWidget(lbl); rl.addWidget(bar)
            layout.addWidget(row)

        tp = self._tp()
        self._spawn_widget(panel, tp.x() - 520, tp.y() + 380)

        self.terminal.append_output("✓ System monitor\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w05_sliders)

    # ── 4. Control Sliders ───────────────────────────────────────────

    def _w05_sliders(self):
        self._type("Control sliders", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w05b_slider_code)))

    def _w05b_slider_code(self):
        code = (
            "```python\n"
            "for p in ['Opacity','Scale','Blur']:\n"
            "    layout.addWidget(QSlider(Qt.Horizontal))\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w05c_slider_spawn))

    def _w05c_slider_spawn(self):
        from PySide6.QtWidgets import QSlider

        panel = QWidget()
        panel.setFixedSize(240, 200)
        panel.setStyleSheet(f"QWidget{{{self._WIDGET_CSS}}}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 10, 14, 10)

        header = QLabel("Transform Controls")
        header.setStyleSheet("font-size:12px;font-weight:bold;border:none;color:#000;background:transparent;")
        layout.addWidget(header)

        for name, val in [("Opacity", 85), ("Scale", 50), ("Blur Radius", 30), ("Rotation", 0)]:
            lbl = QLabel(f"{name}: {val}")
            lbl.setStyleSheet("font-size:10px;color:#000;border:none;background:transparent;margin-top:2px;")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100); slider.setValue(val)
            slider.setStyleSheet(
                "QSlider{background:transparent;border:none;}"
                "QSlider::groove:horizontal{height:4px;background:rgba(0,0,0,0.15);border-radius:2px;}"
                "QSlider::handle:horizontal{width:12px;height:12px;margin:-4px 0;"
                "background:#000;border-radius:6px;}"
                "QSlider::sub-page:horizontal{background:#000;border-radius:2px;}"
            )
            layout.addWidget(lbl)
            layout.addWidget(slider)

        tp = self._tp()
        self._spawn_widget(panel, tp.x() + self._tw + 200, tp.y() - 280)

        self.terminal.append_output("✓ Controls\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w06_notes)

    # ── 5. Notes Pad ─────────────────────────────────────────────────

    def _w06_notes(self):
        self._type("Notes pad", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w06b_notes_code)))

    def _w06b_notes_code(self):
        code = (
            "```python\n"
            "notes = QTextEdit()\n"
            "notes.setMarkdown('# Sprint 12\\n- Ship onboarding')\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w06c_notes_spawn))

    def _w06c_notes_spawn(self):
        from PySide6.QtWidgets import QTextEdit

        notes = QTextEdit()
        notes.setFixedSize(260, 200)
        notes.setStyleSheet(
            "QTextEdit{background:transparent;border:2px solid #000;border-radius:6px;"
            "font-family:monospace;font-size:11px;padding:8px;color:#000;}"
        )
        notes.setMarkdown(
            "## Sprint 12\n\n"
            "- [x] Terminal widget system\n"
            "- [x] AI agent integration\n"
            "- [ ] Onboarding flow\n"
            "- [ ] Voice control polish\n\n"
            "**Next:** Ship v1.0"
        )

        tp = self._tp()
        self._spawn_widget(notes, tp.x() - 100, tp.y() + self._th + 200)

        # Eyes glance down — well above the notes widget
        self._animate_eyes_to(tp.x() + 100, tp.y() + self._th + 40, 700)

        self.terminal.append_output("✓ Notes\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w07_clock)

    # ── 6. Analog Clock ──────────────────────────────────────────────

    def _w07_clock(self):
        self._type("Analog clock", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w07b_clock_code)))

    def _w07b_clock_code(self):
        code = (
            "```python\n"
            "clock = AnalogClock()\n"
            "clock.setFixedSize(180, 180)\n"
            "# QTimer(33ms) → live hands\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w07c_clock_spawn))

    def _w07c_clock_spawn(self):
        """A live analog clock with hour/minute/second hands."""
        import math
        from PySide6.QtCore import QTime

        class AnalogClock(QWidget):
            def __init__(self):
                super().__init__()
                self.setFixedSize(180, 180)
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self._timer = QTimer(self)
                self._timer.timeout.connect(self.update)
                self._timer.start(1000)

            def paintEvent(self, event):
                from PySide6.QtGui import QPainter
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                dark = self.property("dark_mode")
                fg = QColor(230, 230, 230) if dark else QColor(0, 0, 0)
                fga = QColor(230, 230, 230, 150) if dark else QColor(0, 0, 0, 150)
                border = QColor(255, 255, 255, 150) if dark else QColor(0, 0, 0)
                cx, cy, r = 90, 90, 78

                p.setPen(QPen(border, 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(1, 1, 178, 178, 6, 6)
                p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

                for i in range(12):
                    angle = math.radians(i * 30 - 90)
                    x1 = cx + (r - 8) * math.cos(angle)
                    y1 = cy + (r - 8) * math.sin(angle)
                    x2 = cx + (r - 2) * math.cos(angle)
                    y2 = cy + (r - 2) * math.sin(angle)
                    p.setPen(QPen(fg, 2 if i % 3 == 0 else 1))
                    p.drawLine(int(x1), int(y1), int(x2), int(y2))

                now = QTime.currentTime()
                ha = math.radians((now.hour() % 12 + now.minute() / 60.0) * 30 - 90)
                p.setPen(QPen(fg, 3))
                p.drawLine(cx, cy, int(cx + 40 * math.cos(ha)), int(cy + 40 * math.sin(ha)))
                ma = math.radians(now.minute() * 6 - 90)
                p.setPen(QPen(fg, 2))
                p.drawLine(cx, cy, int(cx + 58 * math.cos(ma)), int(cy + 58 * math.sin(ma)))
                sa = math.radians(now.second() * 6 - 90)
                p.setPen(QPen(fga, 1))
                p.drawLine(cx, cy, int(cx + 65 * math.cos(sa)), int(cy + 65 * math.sin(sa)))

                p.setPen(Qt.NoPen); p.setBrush(QBrush(fg))
                p.drawEllipse(cx - 3, cy - 3, 6, 6)
                p.setPen(fg); p.setFont(QFont("Consolas", 8))
                p.drawText(cx - 20, cy + 30, now.toString("hh:mm:ss"))
                p.end()

        clock = AnalogClock()
        tp = self._tp()
        self._spawn_widget(clock, tp.x() + self._tw + 650, tp.y() - 180)

        self.terminal.append_output("✓ Clock\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w08_graph)

    # ── 7. Mini Sparkline Graph ──────────────────────────────────────

    def _w08_graph(self):
        self._type("Activity graph", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w08b_graph_code)))

    def _w08b_graph_code(self):
        code = (
            "```python\n"
            "graph = SparklineWidget(data=random_walk(60))\n"
            "graph.setFixedSize(300, 140)\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w08c_graph_spawn))

    def _w08c_graph_spawn(self):
        """A sparkline chart drawn with QPainter."""
        import random

        class SparklineWidget(QWidget):
            def __init__(self, data):
                super().__init__()
                self.setFixedSize(300, 140)
                self._data = data
                self.setAttribute(Qt.WA_TranslucentBackground, True)

            def paintEvent(self, event):
                from PySide6.QtGui import QPainter
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                dark = self.property("dark_mode")
                fg = QColor(230, 230, 230) if dark else QColor(0, 0, 0)
                fga_low = QColor(230, 230, 230, 25) if dark else QColor(0, 0, 0, 25)
                fga_grid = QColor(230, 230, 230, 40) if dark else QColor(0, 0, 0, 40)
                border = QColor(255, 255, 255, 150) if dark else QColor(0, 0, 0)

                p.setPen(QPen(border, 2)); p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(1, 1, self.width()-2, self.height()-2, 6, 6)

                if not self._data:
                    p.end(); return

                mx, my, mw, mh = 20, 25, 260, 90
                mn = min(self._data); mx_v = max(self._data)
                rng = mx_v - mn if mx_v != mn else 1.0

                p.setPen(fg); p.setFont(QFont("Consolas", 10, QFont.Bold))
                p.drawText(20, 18, "Activity")
                p.setFont(QFont("Consolas", 8))
                p.drawText(220, 18, f"peak: {int(mx_v)}")

                p.setPen(QPen(fga_grid, 1))
                for i in range(5):
                    y = my + mh * i / 4
                    p.drawLine(mx, int(y), mx + mw, int(y))

                path = QPainterPath()
                step = mw / (len(self._data) - 1)
                points = []
                for i, v in enumerate(self._data):
                    x = mx + i * step
                    y = my + mh - (v - mn) / rng * mh
                    points.append((x, y))

                path.moveTo(points[0][0], my + mh)
                for x, y in points: path.lineTo(x, y)
                path.lineTo(points[-1][0], my + mh); path.closeSubpath()
                p.setPen(Qt.NoPen); p.setBrush(QBrush(fga_low)); p.drawPath(path)

                p.setPen(QPen(fg, 2))
                for i in range(len(points) - 1):
                    p.drawLine(int(points[i][0]), int(points[i][1]),
                              int(points[i+1][0]), int(points[i+1][1]))

                p.setBrush(QBrush(fg)); p.setPen(Qt.NoPen)
                for i in [0, len(points)//4, len(points)//2, 3*len(points)//4, len(points)-1]:
                    p.drawEllipse(int(points[i][0])-3, int(points[i][1])-3, 6, 6)
                p.end()

        # Generate random walk data
        data = [50.0]
        for _ in range(59):
            data.append(max(0, min(100, data[-1] + random.gauss(0, 8))))

        graph = SparklineWidget(data)
        tp = self._tp()
        self._spawn_widget(graph, tp.x() + self._tw + 600, tp.y() + 340)

        # Eyes glance right — well to the left of the graph
        self._animate_eyes_to(tp.x() + self._tw + 300, tp.y() + 100, 500)

        self.terminal.append_output("✓ Activity graph\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w09_tree)

    # ── 8. File Tree ─────────────────────────────────────────────────

    def _w09_tree(self):
        self._type("File tree", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w09b_tree_code)))

    def _w09b_tree_code(self):
        code = (
            "```python\n"
            "tree = QTreeWidget()\n"
            "tree.setHeaderLabel('workspace/')\n"
            "# populate with project files\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w09c_tree_spawn))

    def _w09c_tree_spawn(self):
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

        tree = QTreeWidget()
        tree.setFixedSize(240, 260)
        tree.setHeaderLabel("workspace/")
        tree.setStyleSheet(
            "QTreeWidget{background:transparent;border:2px solid #000;border-radius:6px;"
            "font-family:monospace;font-size:11px;color:#000;}"
            "QTreeWidget::item{padding:2px 0;border:none;}"
            "QTreeWidget::item:selected{background:#000;color:#fff;}"
            "QHeaderView::section{background:transparent;border-bottom:1px solid #000;"
            "font-weight:bold;font-size:11px;padding:4px;}"
        )

        folders = {
            "src/": ["main.py", "terminal.py", "ai_voice.py", "onboarding.py"],
            "assets/": ["logo.gif", "both.svg", "onboarding.wav"],
            "config/": ["settings.json", "agents.yaml"],
        }
        for folder, files in folders.items():
            parent = QTreeWidgetItem(tree, [folder])
            parent.setExpanded(True)
            for f in files:
                QTreeWidgetItem(parent, [f])

        tp = self._tp()
        self._spawn_widget(tree, tp.x() - 480, tp.y() + self._th + 100)

        self.terminal.append_output("✓ File tree\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w10_meters)

    # ── 9. VU Meter / Level Indicators ───────────────────────────────

    def _w10_meters(self):
        self._type("Level meters", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w10b_meter_code)))

    def _w10b_meter_code(self):
        code = (
            "```python\n"
            "meters = VUMeterWidget(channels=6)\n"
            "meters.setFixedSize(200, 160)\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w10c_meter_spawn))

    def _w10c_meter_spawn(self):
        """Animated vertical level meters."""
        import random

        class VUMeterWidget(QWidget):
            def __init__(self):
                super().__init__()
                self.setFixedSize(200, 160)
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self._levels = [random.uniform(0.3, 0.9) for _ in range(6)]
                self._targets = list(self._levels)
                self._timer = QTimer(self)
                self._timer.timeout.connect(self._animate)
                self._timer.start(80)

            def _animate(self):
                import random
                for i in range(6):
                    self._targets[i] = max(0.05, min(1.0, self._targets[i] + random.gauss(0, 0.15)))
                    self._levels[i] += (self._targets[i] - self._levels[i]) * 0.3
                self.update()

            def paintEvent(self, event):
                from PySide6.QtGui import QPainter
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                dark = self.property("dark_mode")
                fg = QColor(230, 230, 230) if dark else QColor(0, 0, 0)
                border = QColor(255, 255, 255, 150) if dark else QColor(0, 0, 0)

                p.setPen(QPen(border, 2)); p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(1, 1, self.width()-2, self.height()-2, 6, 6)

                p.setPen(fg); p.setFont(QFont("Consolas", 9, QFont.Bold))
                p.drawText(14, 20, "Levels")

                bar_w = 20; gap = 6; start_x = 16; base_y = 140; max_h = 100
                for i, level in enumerate(self._levels):
                    x = start_x + i * (bar_w + gap)
                    h = int(max_h * level)
                    bg_a = 30 if dark else 20
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(QColor(fg.red(), fg.green(), fg.blue(), bg_a)))
                    p.drawRoundedRect(x, base_y - max_h, bar_w, max_h, 2, 2)
                    bar_a = 40 + int(180 * level)
                    p.setBrush(QBrush(QColor(fg.red(), fg.green(), fg.blue(), bar_a)))
                    p.drawRoundedRect(x, base_y - h, bar_w, h, 2, 2)
                    p.setBrush(QBrush(fg))
                    p.drawEllipse(x + bar_w//2 - 2, base_y - h - 4, 4, 4)

                p.setFont(QFont("Consolas", 7)); p.setPen(fg)
                labels = ["L", "R", "C", "LF", "Ls", "Rs"]
                for i, lbl in enumerate(labels):
                    x = start_x + i * (bar_w + gap)
                    p.drawText(x + 4, 155, lbl)
                p.end()

        meters = VUMeterWidget()
        tp = self._tp()
        self._spawn_widget(meters, tp.x() + self._tw + 700, tp.y() + self._th + 100)

        # Eyes peek right — well away from the meters
        self._animate_eyes_to(tp.x() + self._tw + 380, tp.y() + self._th - 100, 600)

        self.terminal.append_output("✓ Level meters\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(400, self._w11_matrix)

    # ── 10. Connection Matrix ────────────────────────────────────────

    def _w11_matrix(self):
        self._type("Connection matrix", self.terminal.C_USER, 20,
            cb=lambda: (self.terminal.append_output("\n"),
                        _precise_singleshot(150, self._w11b_matrix_code)))

    def _w11b_matrix_code(self):
        code = (
            "```python\n"
            "matrix = NodeMatrix(nodes=5)\n"
            "# adjacency grid with live connections\n"
            "```\n"
        )
        self._stream(code, self.terminal.C_AGENT, 5,
            cb=lambda: _precise_singleshot(100, self._w11c_matrix_spawn))

    def _w11c_matrix_spawn(self):
        """A node connection matrix — adjacency grid."""
        import random

        class NodeMatrixWidget(QWidget):
            def __init__(self):
                super().__init__()
                self.setFixedSize(200, 200)
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self._nodes = ["Term", "Agent", "Voice", "FS", "View"]
                n = len(self._nodes)
                self._adj = [[random.random() > 0.5 if i != j else False for j in range(n)] for i in range(n)]

            def paintEvent(self, event):
                from PySide6.QtGui import QPainter
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing)
                dark = self.property("dark_mode")
                fg = QColor(230, 230, 230) if dark else QColor(0, 0, 0)
                border = QColor(255, 255, 255, 150) if dark else QColor(0, 0, 0)
                fga_diag = QColor(230, 230, 230, 15) if dark else QColor(0, 0, 0, 15)
                fga_grid = QColor(230, 230, 230, 60) if dark else QColor(0, 0, 0, 60)

                p.setPen(QPen(border, 2)); p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(1, 1, self.width()-2, self.height()-2, 6, 6)

                n = len(self._nodes)
                cell = 28; ox = 50; oy = 30

                p.setFont(QFont("Consolas", 7)); p.setPen(fg)
                for i, name in enumerate(self._nodes):
                    p.drawText(6, oy + i * cell + cell // 2 + 3, name)
                    p.drawText(ox + i * cell + 2, oy - 4, name[:2])

                for i in range(n):
                    for j in range(n):
                        x = ox + j * cell
                        y = oy + i * cell
                        if i == j:
                            p.setPen(Qt.NoPen); p.setBrush(QBrush(fga_diag))
                            p.drawRect(x, y, cell, cell)
                        elif self._adj[i][j]:
                            p.setPen(Qt.NoPen); p.setBrush(QBrush(fg))
                            p.drawRoundedRect(x + 6, y + 6, cell - 12, cell - 12, 3, 3)
                        p.setPen(QPen(fga_grid, 1)); p.setBrush(Qt.NoBrush)
                        p.drawRect(x, y, cell, cell)
                p.end()

        matrix = NodeMatrixWidget()
        tp = self._tp()
        self._spawn_widget(matrix, tp.x() + self._tw // 2 - 100, tp.y() + self._th + 500)

        self.terminal.append_output("✓ Connection matrix\n", color=self.terminal.C_SUCCESS)
        _precise_singleshot(500, self._w12_wrap_up)

    # ── Wrap Up ──────────────────────────────────────────────────────

    def _w12_wrap_up(self):
        """Final agent message — return eyes to home position."""
        # Move eyes back above terminal — well clear of all widgets
        tp = self._tp()
        self._animate_eyes_to(
            tp.x() + self._tw / 2 - 140,
            tp.y() - 250,
            1000
        )
        self._stream(
            "\nWorkspace is alive.\n"
            "Every widget, every connection — real.\n",
            self.terminal.C_AGENT, 18,
            cb=lambda: self.terminal.append_output(
                "\n ✓ Onboarding complete.\n", color=self.terminal.C_SUCCESS))

    # ==================================================================
    # IMMERSIVE MODE — dark mode + face mesh + hand control @ 1m20s
    # ==================================================================

    def _m01_dark_mode(self):
        """Activate dark mode with progressive background fade,
        widget color inversion, SVG neon pink, then colorful shadow party."""
        print("[Onboarding] → dark mode + progressive bg fade")

        # Clean up stopped timers from the animation phase
        self._cleanup_stopped_timers()

        # Force dark mode on main window (terminals, etc.)
        if not self.mw._dark_mode:
            self.mw.toggle_dark_mode()

        # Progressive background fade: white → dark over 2 seconds
        self._animate_bg_to_dark(duration_s=2.0)

        # After 600ms (background already noticeably dimming), switch widget styles
        _precise_singleshot(600, self._m01b_switch_widgets)

    def _animate_bg_to_dark(self, duration_s=2.0):
        """Smoothly interpolate scene/view background from current to dark."""
        # Capture current background color
        current_brush = self.scene.backgroundBrush()
        if current_brush and current_brush.color().isValid():
            start_r, start_g, start_b = current_brush.color().red(), current_brush.color().green(), current_brush.color().blue()
        else:
            start_r, start_g, start_b = 255, 255, 255  # assume white
        end_r, end_g, end_b = 30, 30, 30  # target dark

        t0 = time.perf_counter()
        def tick():
            p = min(1.0, (time.perf_counter() - t0) / duration_s)
            # Ease-in-out cubic for smooth feel
            ep = p * p * (3 - 2 * p)
            r = int(start_r + (end_r - start_r) * ep)
            g = int(start_g + (end_g - start_g) * ep)
            b = int(start_b + (end_b - start_b) * ep)
            c = QColor(r, g, b)
            self.scene.setBackgroundBrush(QBrush(c))
            self.view.setBackgroundBrush(QBrush(c))
            if p >= 1.0:
                ti.stop()
                print("[Onboarding] ✓ Background fade complete")
        ti = QTimer(self); ti.timeout.connect(tick); ti.start(30); self._keep.append(ti)

    def _m01b_switch_widgets(self):
        """Switch widget colors and SVG to dark/neon pink, then start shadow party."""
        # Switch all spawned widgets to dark color scheme
        self._switch_widgets_dark()

        # SVG: black → neon pink + pink shadow
        self._switch_svg_neon_pink()

        self.terminal.append_output(
            "\n ◐ Dark mode active.\n",
            color=self.terminal.C_INFO
        )

        # Start colorful shadow party 2.5s after widget switch
        _precise_singleshot(2500, self._m01c_shadow_party)

        # Start camera after total ~4s from dark mode start
        _precise_singleshot(3500, self._m02_start_camera)

    # ------------------------------------------------------------------
    # Colorful shadow party: animate widget shadows to random vivid colors
    # ------------------------------------------------------------------

    _PARTY_COLORS = [
        QColor(255, 16, 240),    # neon pink
        QColor(0, 255, 128),     # neon green
        QColor(80, 200, 255),    # electric blue
        QColor(255, 200, 0),     # golden yellow
        QColor(255, 80, 80),     # coral red
        QColor(180, 100, 255),   # purple
        QColor(0, 255, 255),     # cyan
        QColor(255, 128, 0),     # orange
    ]

    def _m01c_shadow_party(self):
        """Animate each widget's shadow to a random vivid color over ~3s,
        while eyes roam between widgets watching the transformation."""
        import random as _rng
        print("[Onboarding] → shadow color party + roaming eyes")

        if not self._spawned_widgets:
            return

        # Assign a random target color per widget
        targets = []
        for widget, proxy in self._spawned_widgets:
            color = _rng.choice(self._PARTY_COLORS)
            targets.append((proxy, color))

        # Also colorize the button shadow
        if self._btn_proxy:
            targets.append((self._btn_proxy, _rng.choice(self._PARTY_COLORS)))

        # Animate all shadows simultaneously: white glow → vivid color over 2.5s
        dur_s = 2.5
        t0 = time.perf_counter()

        # Snapshot starting shadow colors (all white from dark mode switch)
        start_color = QColor(255, 255, 255, 100)

        def tick():
            p = min(1.0, (time.perf_counter() - t0) / dur_s)
            ep = 1 - pow(1 - p, 3)  # ease-out cubic
            for proxy, target in targets:
                try:
                    r = int(start_color.red() + (target.red() - start_color.red()) * ep)
                    g = int(start_color.green() + (target.green() - start_color.green()) * ep)
                    b = int(start_color.blue() + (target.blue() - start_color.blue()) * ep)
                    a = int(100 + (160 - 100) * ep)  # also increase alpha to 160
                    sh = QGraphicsDropShadowEffect()
                    sh.setBlurRadius(int(20 + 10 * ep))  # bloom slightly
                    sh.setColor(QColor(r, g, b, a))
                    sh.setOffset(45, 45)
                    proxy.setGraphicsEffect(sh)
                except RuntimeError:
                    pass
            if p >= 1.0:
                ti.stop()
                print("[Onboarding] ✓ Shadow party complete")

        ti = QTimer(self); ti.timeout.connect(tick); ti.start(40); self._keep.append(ti)

        # Simultaneously: eyes roam between widgets, looking at each one
        self._start_roaming_eyes(dur_s)

    def _start_roaming_eyes(self, total_dur_s):
        """Direct the eye pupils toward random widgets during shadow party.
        Extreme gaze, dramatic eyebrows, lid squints, double wink."""
        import random as _rng
        if not self._spawned_widgets or not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return

        ew = self._eyes_widget
        ew._gaze_mouse_active = False

        # Visit ALL widgets if possible — eyes scan everything
        visit_count = min(8, len(self._spawned_widgets))
        chosen = _rng.sample(self._spawned_widgets, visit_count)

        dwell_ms = int(total_dur_s * 1000 / visit_count)

        for i, (widget, proxy) in enumerate(chosen):
            delay = i * dwell_ms
            try:
                wcx = proxy.x() + widget.width() / 2
                wcy = proxy.y() + widget.height() / 2
            except RuntimeError:
                continue

            _precise_singleshot(delay, lambda cx=wcx, cy=wcy, idx=i: (
                self._direct_eyes_at(cx, cy, idx)
            ))

        # Two winks — one early, one late
        _precise_singleshot(int(total_dur_s * 250), self._trigger_eyes_wink)
        _precise_singleshot(int(total_dur_s * 750), self._trigger_eyes_wink)

        # At the end, relax
        _precise_singleshot(int(total_dur_s * 1000), self._relax_eyes)

    def _direct_eyes_at(self, scene_x, scene_y, visit_idx):
        """Point eye pupils HARD toward a scene coordinate — extreme expressiveness."""
        import random as _rng
        import math
        if not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return
        if not hasattr(self, '_eyes_proxy') or not self._eyes_proxy:
            return

        ew = self._eyes_widget

        ep = self._eyes_proxy.pos()
        eye_cx = ep.x() + ew.width() / 2
        eye_cy = ep.y() + ew.height() / 2
        dx = scene_x - eye_cx
        dy = scene_y - eye_cy

        dist = math.sqrt(dx * dx + dy * dy) if (dx or dy) else 1.0
        max_gaze = 18.0  # FULL max — eyes slam to the edges
        fx = (dx / dist) * max_gaze if dist > 0 else 0.0
        fy = (dy / dist) * max_gaze * 0.8 if dist > 0 else 0.0

        # No distance dampening — always look hard
        ew._gaze_target_x = fx
        ew._gaze_target_y = fy
        ew._gaze_speed = _rng.uniform(0.14, 0.25)  # fast saccade

        # Dramatic eyebrow expressions — cycle through a rich repertoire
        expressions = [
            (-12.0, 1.0),    # very surprised (eyebrows way up, eyes wide)
            (6.0, 0.75),     # skeptical squint (brows down, lids narrowed)
            (-8.0, 0.95),    # curious
            (0.0, 0.85),     # neutral squint
            (-15.0, 1.0),    # shocked
            (3.0, 0.80),     # suspicious
            (-6.0, 1.0),     # interested
            (8.0, 0.70),     # intense focus (deep squint)
        ]
        brow_offset, lid_level = expressions[visit_idx % len(expressions)]
        ew.eyebrow_offset = brow_offset
        ew.eye_open_level = lid_level

        ew.update()

    def _relax_eyes(self):
        """Return eyes to neutral: center gaze, relaxed brows, full open."""
        if not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return
        ew = self._eyes_widget
        ew._gaze_target_x = 0.0
        ew._gaze_target_y = 0.0
        ew._gaze_speed = 0.05  # slow drift back to center
        ew.eyebrow_offset = 0.0
        ew.eye_open_level = 1.0
        ew.update()

    # ------------------------------------------------------------------
    # Eyes wink (proper wink using the widget's trigger_wink method)
    # ------------------------------------------------------------------

    def _trigger_eyes_wink(self):
        """Make the eyes do a single wink using the widget's built-in wink animation."""
        if not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return
        ew = self._eyes_widget
        try:
            if hasattr(ew, 'trigger_wink'):
                ew.trigger_wink()
            else:
                # Fallback: manual blink cycle
                ew.blink_direction = 1
                ew.is_animating = True
                _precise_singleshot(200, lambda: setattr(ew, 'blink_direction', -1))
        except (RuntimeError, AttributeError) as e:
            print(f"[Onboarding] wink failed: {e}")

    _DARK_WIDGET_CSS = (
        "background: transparent;"
        "border: 2px solid rgba(255,255,255,0.6);"
        "border-radius: 6px;"
        "font-family: 'Consolas','Monaco',monospace;"
        "color: #eee;"
    )

    def _switch_widgets_dark(self):
        """Invert all spawned widget colors: black→white borders, white→light text,
        dark shadows with white glow. Uses batched updates for performance."""
        # Suspend view repaints during bulk style changes
        self.view.setUpdatesEnabled(False)
        try:
            self._apply_dark_styles()
        finally:
            self.view.setUpdatesEnabled(True)
            self.view.viewport().update()

    def _apply_dark_styles(self):
        for widget, proxy in self._spawned_widgets:
            try:
                # Update shadow to white glow for dark mode
                sh = QGraphicsDropShadowEffect()
                sh.setBlurRadius(20)
                sh.setColor(QColor(255, 255, 255, 100))
                sh.setOffset(45, 45)
                proxy.setGraphicsEffect(sh)
            except RuntimeError:
                continue

            cls_name = widget.__class__.__name__

            try:
                if cls_name == "QCalendarWidget":
                    widget.setStyleSheet(
                        "QCalendarWidget{background:transparent;border:2px solid rgba(255,255,255,0.6);"
                        "border-radius:6px;font-family:monospace;font-size:11px;color:#eee;}"
                        "QCalendarWidget QToolButton{color:#eee;font-weight:bold;background:transparent;border:none;}"
                        "QCalendarWidget QWidget#qt_calendar_navigationbar{background:transparent;"
                        "border-bottom:1px solid rgba(255,255,255,0.4);}"
                        "QCalendarWidget QAbstractItemView{selection-background-color:#fff;"
                        "selection-color:#000;color:#ddd;background:transparent;}"
                    )
                elif cls_name == "QTreeWidget":
                    widget.setStyleSheet(
                        "QTreeWidget{background:transparent;border:2px solid rgba(255,255,255,0.6);"
                        "border-radius:6px;font-family:monospace;font-size:11px;color:#eee;}"
                        "QTreeWidget::item{padding:2px 0;border:none;color:#ddd;}"
                        "QTreeWidget::item:selected{background:#fff;color:#000;}"
                        "QHeaderView::section{background:transparent;"
                        "border-bottom:1px solid rgba(255,255,255,0.4);"
                        "font-weight:bold;font-size:11px;padding:4px;color:#eee;}"
                    )
                elif cls_name == "QTextEdit":
                    widget.setStyleSheet(
                        "QTextEdit{background:transparent;border:2px solid rgba(255,255,255,0.6);"
                        "border-radius:6px;font-family:monospace;font-size:11px;"
                        "padding:8px;color:#eee;}"
                    )
                elif hasattr(widget, 'paintEvent') and cls_name not in ("QWidget",):
                    # Custom painted widgets (clock, color wheel, sparkline, VU meters, matrix)
                    # These draw with QPainter — we set a flag they can check
                    widget.setProperty("dark_mode", True)
                    widget.update()
                else:
                    # Generic QWidget containers (dashboard, sliders)
                    widget.setStyleSheet(f"QWidget{{{self._DARK_WIDGET_CSS}}}")
                    # Also restyle child labels, progress bars, sliders
                    for child in widget.findChildren(QLabel):
                        old = child.styleSheet()
                        child.setStyleSheet(
                            old.replace("color:#000", "color:#eee")
                                .replace("color:#555", "color:#bbb")
                        )
                    from PySide6.QtWidgets import QProgressBar, QSlider
                    for bar in widget.findChildren(QProgressBar):
                        bar.setStyleSheet(
                            "QProgressBar{border:1px solid rgba(255,255,255,0.5);border-radius:3px;"
                            "background:transparent;text-align:center;font-size:8px;color:#eee;}"
                            "QProgressBar::chunk{background:#fff;border-radius:2px;}"
                        )
                    for slider in widget.findChildren(QSlider):
                        slider.setStyleSheet(
                            "QSlider{background:transparent;border:none;}"
                            "QSlider::groove:horizontal{height:4px;"
                            "background:rgba(255,255,255,0.15);border-radius:2px;}"
                            "QSlider::handle:horizontal{width:12px;height:12px;margin:-4px 0;"
                            "background:#fff;border-radius:6px;}"
                            "QSlider::sub-page:horizontal{background:#fff;border-radius:2px;}"
                        )
            except RuntimeError:
                continue

        # Also switch the Hello World button if it exists
        if self._btn_widget:
            try:
                self._btn_widget.setStyleSheet(
                    "QPushButton{background-color:transparent;"
                    "border:2px solid rgba(255,255,255,0.6);"
                    "color:#eee;font-size:16px;font-weight:bold;"
                    "border-radius:4px;}"
                    "QPushButton:hover{background-color:rgba(255,255,255,0.1);}"
                )
            except RuntimeError:
                pass
        if self._btn_proxy:
            try:
                sh = QGraphicsDropShadowEffect()
                sh.setBlurRadius(20)
                sh.setColor(QColor(255, 255, 255, 100))
                sh.setOffset(45, 45)
                self._btn_proxy.setGraphicsEffect(sh)
            except RuntimeError:
                pass

    # Neon pink: #FF10F0
    _NEON_PINK = "#FF10F0"

    def _switch_svg_neon_pink(self):
        """Change SVG handwriting from black to neon pink with matching glow shadow."""
        if not hasattr(self, '_svg_hand') or not self._svg_hand:
            return
        try:
            self._svg_hand.set_color(self._NEON_PINK)
        except RuntimeError:
            return

        # Switch shadow to neon pink glow
        if hasattr(self, '_svg_proxy') and self._svg_proxy:
            try:
                sh = QGraphicsDropShadowEffect()
                sh.setBlurRadius(30)
                sh.setColor(QColor(255, 16, 240, 160))  # neon pink glow
                sh.setOffset(0, 0)  # centered glow, no directional offset
                self._svg_proxy.setGraphicsEffect(sh)
                self._svg_shadow = sh
            except RuntimeError:
                pass

    def _m02_start_camera(self):
        """Initialize MediaPipe and start the camera. Create face widget off-screen."""
        print("[Onboarding] → starting camera + mediapipe")

        if not _HAS_MEDIAPIPE:
            print("[Onboarding] MediaPipe not available — skipping face mesh")
            self.terminal.append_output(
                " ⚠ MediaPipe not installed — camera features skipped.\n",
                color=self.terminal.C_INFO
            )
            return

        # Create sub-components
        self._ob_camera = CameraManager()
        self._ob_gaze_tracker = HeadGazeTracker()
        self._ob_pinch_detector = PinchDetector()
        self._ob_face_widget = OnboardingFaceMeshWidget(self.mw)
        self._ob_gaze_cursor = GazeCursorWidget(self.mw)
        self._ob_gaze_cursor.hide()

        # Glide state for post-gravity smoothing
        self._ob_glide_x = None
        self._ob_glide_y = None
        self._ob_glide_alpha = 0.20
        self._ob_gaze_pos = None
        self._ob_is_pinching = False
        self._ob_pinch_item = None
        self._ob_pinch_offset = QPointF()
        self._ob_selected_proxy = None

        # Park face widget off-screen (top-right, outside view)
        win_rect = self.mw.rect()
        start_x = win_rect.width() + 80
        start_y = -self._ob_face_widget.height() - 80
        self._ob_face_widget.move(start_x, start_y)
        self._ob_face_widget.show()
        self._ob_face_widget.raise_()

        # Initialize + start camera
        self._ob_mediapipe_ready = self._ob_camera.initialize_mediapipe()
        if not self._ob_mediapipe_ready:
            print("[Onboarding] MediaPipe init failed")
            self.terminal.append_output(
                " ⚠ Camera initialization failed.\n",
                color=self.terminal.C_INFO
            )
            return

        self._ob_camera.frame_ready.connect(self._m_on_frame)
        self._ob_camera.start()

        self.terminal.append_output(
            " ◉ Camera active — face mesh processing...\n",
            color=self.terminal.C_SUCCESS
        )

        # Wait for a few frames to warm up, then slide in
        _precise_singleshot(1600, self._m03_slide_in_face)

    def _m03_slide_in_face(self):
        """Animate the face mesh widget in from top-right."""
        print("[Onboarding] → slide in face mesh")

        if not hasattr(self, '_ob_face_widget') or not self._ob_face_widget:
            return

        win_rect = self.mw.rect()
        fw = self._ob_face_widget.width()

        final_x = win_rect.width() - fw - 20
        final_y = 20

        start_pos = self._ob_face_widget.pos()

        self._face_slide_anim = QPropertyAnimation(self._ob_face_widget, b"pos")
        self._face_slide_anim.setDuration(1800)
        self._face_slide_anim.setStartValue(start_pos)
        self._face_slide_anim.setEndValue(QPoint(final_x, final_y))
        self._face_slide_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._face_slide_anim.start()

        # Show gaze cursor
        if self._ob_gaze_cursor:
            self._ob_gaze_cursor.show()
            self._ob_gaze_cursor.raise_()

        self.terminal.append_output(
            " 👤 Face mesh active — head gaze controls the cursor.\n",
            color=self.terminal.C_SUCCESS
        )
        self._stream(
            " Pinch with your right hand to grab and move widgets.\n",
            self.terminal.C_INFO, 15,
            cb=None
        )

        # Schedule the finale sequence at 1m35 from music start
        self._at_music_time(95.0, self._f01_spread_widgets)

    # ==================================================================
    # FINALE — widget spread, 3D game, view restore, welcome reset
    # ==================================================================
    # 1m35  _f01_spread_widgets   — animate all widgets to periphery (4s)
    # 1m39  _f02_spawn_game       — 3D car game appears in center
    #       _f03_retreat_face     — face mesh slowly slides top-right
    #       _f04_reverse_tilt     — view tilt reverses back to normal
    # 2m04  _f05_view_rest        — view at rest, trigger welcome reset
    #       _f06_welcome_reset    — show WelcomeScreen, cleanup scene behind it
    # ==================================================================

    def _f01_spread_widgets(self):
        """Animate all spawned widgets from center outward to far periphery over 4s."""
        import math
        print("[Onboarding] → spreading widgets to periphery")

        if not self._spawned_widgets:
            _precise_singleshot(100, self._f02_spawn_game)
            return

        # Calculate the actual centroid of ALL scene proxies (widgets + terminal + button + SVG + eyes)
        all_proxies = []
        for widget, proxy in self._spawned_widgets:
            try:
                all_proxies.append((proxy.x() + widget.width() / 2,
                                    proxy.y() + widget.height() / 2))
            except RuntimeError:
                pass
        for p in [self.terminal_proxy, self._btn_proxy]:
            if p:
                try:
                    all_proxies.append((p.x() + 100, p.y() + 100))
                except RuntimeError:
                    pass
        if hasattr(self, '_svg_proxy') and self._svg_proxy:
            try:
                all_proxies.append((self._svg_proxy.x() + 100, self._svg_proxy.y() + 50))
            except RuntimeError:
                pass
        if hasattr(self, '_eyes_proxy') and self._eyes_proxy:
            try:
                all_proxies.append((self._eyes_proxy.x() + 140, self._eyes_proxy.y() + 70))
            except RuntimeError:
                pass

        if all_proxies:
            center_x = sum(p[0] for p in all_proxies) / len(all_proxies)
            center_y = sum(p[1] for p in all_proxies) / len(all_proxies)
        else:
            tp = self._tp()
            center_x = tp.x() + self._tw / 2
            center_y = tp.y() + self._th / 2

        # Store for game placement and view centering
        self._spread_center = QPointF(center_x, center_y)
        print(f"[Onboarding] Spread/game center: ({center_x:.0f}, {center_y:.0f})")

        dur_s = 4.0
        t0 = time.perf_counter()

        # Snapshot starting positions and calculate peripheral targets
        widget_anims = []
        spread_radius = 1200  # how far out from center
        for i, (widget, proxy) in enumerate(self._spawned_widgets):
            try:
                sx, sy = proxy.x(), proxy.y()
                # Direction from center to current position
                dx = sx - center_x
                dy = sy - center_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 10:
                    # Widgets near center: push outward at evenly spaced angles
                    angle = (i / len(self._spawned_widgets)) * 2 * math.pi
                    dx, dy = math.cos(angle), math.sin(angle)
                    dist = 1.0
                # Target: push along the same direction but much further
                nx, ny = dx / dist, dy / dist
                tx = center_x + nx * spread_radius
                ty = center_y + ny * spread_radius
                widget_anims.append((proxy, sx, sy, tx, ty))
            except RuntimeError:
                continue

        # Also spread the terminal, button, SVG, eyes
        extra_proxies = []
        if self.terminal_proxy:
            try:
                sx, sy = self.terminal_proxy.x(), self.terminal_proxy.y()
                extra_proxies.append((self.terminal_proxy, sx, sy,
                                      center_x - 1400, center_y - 800))
            except RuntimeError:
                pass
        if self._btn_proxy:
            try:
                sx, sy = self._btn_proxy.x(), self._btn_proxy.y()
                extra_proxies.append((self._btn_proxy, sx, sy,
                                      center_x + 1400, center_y - 600))
            except RuntimeError:
                pass
        if hasattr(self, '_svg_proxy') and self._svg_proxy:
            try:
                sx, sy = self._svg_proxy.x(), self._svg_proxy.y()
                extra_proxies.append((self._svg_proxy, sx, sy,
                                      center_x - 1200, center_y + 900))
            except RuntimeError:
                pass
        if hasattr(self, '_eyes_proxy') and self._eyes_proxy:
            try:
                sx, sy = self._eyes_proxy.x(), self._eyes_proxy.y()
                extra_proxies.append((self._eyes_proxy, sx, sy,
                                      center_x, center_y - 1100))
            except RuntimeError:
                pass

        all_anims = widget_anims + extra_proxies

        def tick():
            p = min(1.0, (time.perf_counter() - t0) / dur_s)
            # Ease-in-out cubic for elegant spread
            ep = p * p * (3 - 2 * p)
            for proxy, sx, sy, tx, ty in all_anims:
                try:
                    proxy.setPos(
                        sx + (tx - sx) * ep,
                        sy + (ty - sy) * ep
                    )
                except RuntimeError:
                    pass
            if p >= 1.0:
                ti.stop()
                print("[Onboarding] ✓ Widgets spread complete")
                # Chain: spawn game + retreat face + reverse tilt
                self._f02_spawn_game()
        ti = QTimer(self); ti.timeout.connect(tick); ti.start(25); self._keep.append(ti)

        # Eyes look around wildly during the spread
        self._eyes_wild_scan(dur_s)

    def _eyes_wild_scan(self, dur_s):
        """Make eyes scan wildly in all directions during widget spread."""
        if not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return
        ew = self._eyes_widget
        import random as _rng

        # Schedule rapid gaze shifts every ~300ms
        n_shifts = int(dur_s * 1000 / 300)
        for i in range(n_shifts):
            delay = i * 300
            def shift(idx=i):
                try:
                    angle = _rng.uniform(0, 6.28)
                    import math
                    ew._gaze_target_x = 18.0 * math.cos(angle)
                    ew._gaze_target_y = 14.0 * math.sin(angle)
                    ew._gaze_speed = _rng.uniform(0.20, 0.35)
                    # Eyebrow chaos
                    ew.eyebrow_offset = _rng.uniform(-15, 8)
                    ew.eye_open_level = _rng.uniform(0.65, 1.0)
                except (RuntimeError, AttributeError):
                    pass
            _precise_singleshot(delay, shift)

    def _f02_spawn_game(self):
        """Spawn the 3D city driving game widget in the scene center."""
        print("[Onboarding] → spawning 3D city driving game")

        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWebEngineCore import QWebEngineSettings
            has_web = True
        except ImportError:
            has_web = False

        if has_web:
            game_container = QWidget()
            game_container.setFixedSize(1200, 800)
            game_container.setStyleSheet("background: transparent; border-radius: 15px;")

            from PySide6.QtWidgets import QVBoxLayout as _VBL
            _lay = _VBL(game_container)
            _lay.setContentsMargins(0, 0, 0, 0)

            game_view = QWebEngineView()
            game_view.setStyleSheet("background: transparent; border-radius: 15px;")
            game_view.settings().setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            game_view.settings().setAttribute(
                QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)

            game_html = self._build_car_game_html()
            game_view.setHtml(game_html)
            _lay.addWidget(game_view)
            game = game_container
        else:
            game = QWidget()
            game.setFixedSize(1200, 800)
            game.setStyleSheet(
                "background: qlineargradient(y1:0, y2:1, stop:0 #1a0a2e, stop:1 #16213e);"
                "border: 2px solid rgba(255,150,80,0.3); border-radius: 15px;"
            )
            lbl = QLabel("🏎️ 3D City Drive\n(WebEngine required)", game)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #ffc080; font-size: 28px; font-family: monospace;")
            lbl.setGeometry(0, 0, 1200, 800)

        # Place at the spread/repulsion center
        sc = getattr(self, '_spread_center', None)
        if sc is None:
            tp = self._tp()
            sc = QPointF(tp.x() + self._tw / 2, tp.y() + self._th / 2)
        self._game_center = sc

        self._game_proxy = self.scene.addWidget(game)
        self._game_proxy.setZValue(2000)
        self._game_proxy.setPos(sc.x() - 600, sc.y() - 400)

        # Warm sunset glow shadow
        sh = QGraphicsDropShadowEffect()
        sh.setBlurRadius(50)
        sh.setColor(QColor(255, 150, 80, 180))
        sh.setOffset(0, 0)
        self._game_proxy.setGraphicsEffect(sh)
        self._fin(self._game_proxy, 800)

        self._game_widget = game

        # Simultaneously: retreat face mesh + reverse view tilt
        _precise_singleshot(500, self._f03_retreat_face)
        _precise_singleshot(200, self._f04_reverse_tilt)

    def _build_car_game_html(self):
        """Return the inline HTML for the 3D city driving game."""
        return r'''<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { overflow:hidden; background:#1a0a2e; border-radius:15px; font-family:Arial,sans-serif; }
        canvas { display:block; border-radius:15px; }
        #hud { position:absolute; top:0;left:0;right:0;bottom:0; pointer-events:none; z-index:100; }
        #speed-box {
            position:absolute; bottom:24px; right:24px;
            background:rgba(12,4,20,0.85); padding:18px 28px; border-radius:14px;
            border:1px solid rgba(255,150,80,0.2); text-align:center; min-width:120px;
        }
        #speed { color:#ffcc66; font-family:monospace; font-size:40px; font-weight:900;
            text-shadow:0 0 20px rgba(255,180,80,0.5); line-height:1; }
        #speed-unit { color:rgba(255,180,120,0.4); font-family:monospace; font-size:10px; letter-spacing:3px; margin-top:4px; }
        #gear { color:#ff8844; font-family:monospace; font-size:11px; margin-top:5px; letter-spacing:2px; }
        #controls {
            position:absolute; bottom:24px; left:24px; color:#ffd4a0; font-size:12px;
            background:rgba(12,4,20,0.75); padding:10px 16px; border-radius:10px;
            border:1px solid rgba(255,150,80,0.15); line-height:1.7;
        }
        #controls kbd {
            background:rgba(255,150,80,0.18); border:1px solid rgba(255,150,80,0.3);
            border-radius:3px; padding:1px 5px; font-family:monospace; font-size:10px; color:#ffcc88;
        }
        .pill {
            position:absolute; top:18px; left:18px;
            background:rgba(12,4,20,0.75); padding:8px 16px; border-radius:20px;
            border:1px solid rgba(255,150,80,0.15); color:#ffc080; font-size:13px;
            display:flex; align-items:center; gap:6px;
        }
        .pill .dot { width:5px;height:5px;border-radius:50%;background:#ff8844;box-shadow:0 0 5px #ff8844; }
        #snow-pill {
            position:absolute; top:18px; left:160px;
            background:rgba(12,4,20,0.75); padding:8px 16px; border-radius:20px;
            border:1px solid rgba(255,150,80,0.15); color:#ffc080; font-size:13px;
            display:none; align-items:center; gap:6px;
        }
        #collision-flash {
            position:absolute; top:0;left:0;right:0;bottom:0;
            background:radial-gradient(circle,rgba(255,60,20,0.4),transparent 70%);
            opacity:0; transition:opacity 0.1s; pointer-events:none;
        }
    </style>
</head>
<body>
    <div id="hud">
        <div class="pill"><div class="dot"></div><span id="npc-count">0</span></div>
        <div id="snow-pill"><div class="dot" style="background:#adf;box-shadow:0 0 5px #adf"></div>Snow</div>
        <div id="speed-box"><div id="speed">0</div><div id="speed-unit">KM/H</div><div id="gear">P</div></div>
        <div id="controls"><kbd>W</kbd> Gas <kbd>S</kbd> Brake <kbd>A</kbd><kbd>D</kbd> Steer <kbd>SPACE</kbd> Handbrake</div>
        <div id="collision-flash"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function(){
        var scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a0a2e);
        scene.fog = new THREE.Fog(0x2a1535, 40, 320);
        var camera = new THREE.PerspectiveCamera(68, window.innerWidth/window.innerHeight, 0.1, 500);
        var renderer = new THREE.WebGLRenderer({ antialias:true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.body.appendChild(renderer.domElement);
        var skyGeo = new THREE.SphereGeometry(380,24,16);
        var sp = skyGeo.attributes.position.array;
        var sc = new Float32Array(sp.length);
        for(var i=0;i<sp.length;i+=3){
            var h=(sp[i+1]/380+1)*0.5;
            if(h<0.35){sc[i]=1.0;sc[i+1]=0.35;sc[i+2]=0.1;}
            else if(h<0.5){var t=(h-0.35)/0.15;sc[i]=1.0-t*0.6;sc[i+1]=0.35-t*0.12;sc[i+2]=0.1+t*0.16;}
            else if(h<0.7){var t=(h-0.5)/0.2;sc[i]=0.4-t*0.2;sc[i+1]=0.23-t*0.08;sc[i+2]=0.26+t*0.06;}
            else{var t=(h-0.7)/0.3;sc[i]=0.2-t*0.14;sc[i+1]=0.15-t*0.1;sc[i+2]=0.32-t*0.15;}
        }
        skyGeo.setAttribute('color',new THREE.Float32BufferAttribute(sc,3));
        var skyMesh=new THREE.Mesh(skyGeo,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.BackSide,depthWrite:false,fog:false}));
        scene.add(skyMesh);
        var sunG1=new THREE.Mesh(new THREE.SphereGeometry(40,12,12),new THREE.MeshBasicMaterial({color:0xffdd44,transparent:true,opacity:0.45,fog:false}));
        sunG1.position.set(0,22,370);scene.add(sunG1);
        var sunG2=new THREE.Mesh(new THREE.SphereGeometry(18,12,12),new THREE.MeshBasicMaterial({color:0xffffaa,transparent:true,opacity:0.8,fog:false}));
        sunG2.position.set(0,22,370);scene.add(sunG2);
        var hazeGeo=new THREE.PlaneGeometry(800,60);
        var hazeMat=new THREE.MeshBasicMaterial({color:0xff8833,transparent:true,opacity:0.15,fog:false,side:THREE.DoubleSide});
        var haze=new THREE.Mesh(hazeGeo,hazeMat);haze.position.set(0,8,370);scene.add(haze);
        scene.add(new THREE.AmbientLight(0x665555,0.5));
        var sunLight=new THREE.DirectionalLight(0xff9955,0.9);
        sunLight.position.set(20,40,150);sunLight.castShadow=true;
        sunLight.shadow.mapSize.width=1024;sunLight.shadow.mapSize.height=1024;
        sunLight.shadow.camera.near=1;sunLight.shadow.camera.far=200;
        sunLight.shadow.camera.left=-60;sunLight.shadow.camera.right=60;
        sunLight.shadow.camera.top=60;sunLight.shadow.camera.bottom=-60;
        sunLight.shadow.bias=-0.003;
        scene.add(sunLight);scene.add(sunLight.target);
        scene.add(new THREE.HemisphereLight(0xff6633,0x222244,0.25));
        var headlight=new THREE.PointLight(0xffeedd,0.7,30,2);scene.add(headlight);
        var RW=18,SW=4,laneW=RW/4,colliders=[],npcCars=[];
        var gnd=new THREE.Mesh(new THREE.PlaneGeometry(600,2000),new THREE.MeshLambertMaterial({color:0x151f15}));
        gnd.rotation.x=-Math.PI/2;gnd.receiveShadow=true;scene.add(gnd);
        var roadLen=2000;
        var road=new THREE.Mesh(new THREE.PlaneGeometry(RW,roadLen),new THREE.MeshLambertMaterial({color:0x2a2a2a}));
        road.rotation.x=-Math.PI/2;road.position.y=0.01;road.receiveShadow=true;scene.add(road);
        var dashMat=new THREE.MeshBasicMaterial({color:0xccaa44});
        for(var d=-roadLen/2;d<roadLen/2;d+=6){var dm=new THREE.Mesh(new THREE.PlaneGeometry(0.22,3),dashMat);dm.rotation.x=-Math.PI/2;dm.position.set(0,0.025,d);scene.add(dm);}
        var laneMat=new THREE.MeshBasicMaterial({color:0x444444});
        [-1,1].forEach(function(s){var ln=new THREE.Mesh(new THREE.PlaneGeometry(0.1,roadLen),laneMat);ln.rotation.x=-Math.PI/2;ln.position.set(s*laneW,0.025,0);scene.add(ln);});
        var edgeMat=new THREE.MeshBasicMaterial({color:0x555555});
        [-1,1].forEach(function(s){var el=new THREE.Mesh(new THREE.PlaneGeometry(0.15,roadLen),edgeMat);el.rotation.x=-Math.PI/2;el.position.set(s*(RW/2-0.2),0.025,0);scene.add(el);});
        var swMat=new THREE.MeshLambertMaterial({color:0x3d3030}),curbMat=new THREE.MeshLambertMaterial({color:0x6b6058});
        [-1,1].forEach(function(s){
            var sw=new THREE.Mesh(new THREE.PlaneGeometry(SW,roadLen),swMat);sw.rotation.x=-Math.PI/2;sw.position.set(s*(RW/2+SW/2),0.05,0);sw.receiveShadow=true;scene.add(sw);
            var curb=new THREE.Mesh(new THREE.BoxGeometry(0.3,0.15,roadLen),curbMat);curb.position.set(s*(RW/2),0.075,0);scene.add(curb);
            var curb2=new THREE.Mesh(new THREE.BoxGeometry(0.3,0.15,roadLen),curbMat);curb2.position.set(s*(RW/2+SW),0.075,0);scene.add(curb2);
        });
        var bPal=[0x3a2828,0x2d2838,0x383028,0x282d38,0x443333,0x334444,0x383838,0x4a3030,0x30304a,0x3a3a2a,0x332233,0x2a3333];
        function mkBldg(x,z,w,d,h,col){
            var g=new THREE.Group();
            var box=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshLambertMaterial({color:col}));
            box.position.y=h/2;box.castShadow=true;box.receiveShadow=true;g.add(box);
            if(h>20&&Math.random()>0.3){var rh=1.5+Math.random()*3;var rb=new THREE.Mesh(new THREE.BoxGeometry(w*0.35,rh,d*0.35),new THREE.MeshLambertMaterial({color:0x222222}));rb.position.y=h+rh/2;g.add(rb);}
            if(Math.random()>0.4){for(var a=0;a<1+Math.floor(Math.random()*3);a++){var ac=new THREE.Mesh(new THREE.BoxGeometry(1.2,0.8,1.2),new THREE.MeshLambertMaterial({color:0x555555}));ac.position.set((Math.random()-0.5)*w*0.5,h+0.4,(Math.random()-0.5)*d*0.5);g.add(ac);}}
            var wGeo=new THREE.PlaneGeometry(1.4,2),wSp=4.2;
            for(var fl=3.5;fl<h-2;fl+=wSp){
                for(var wx=-w/2+2;wx<w/2-1;wx+=wSp){
                    var lit=Math.random()>0.15,cool=Math.random()>0.7;
                    var wc=lit?(cool?0x88bbff:0xffdd88):0x181210;
                    var wm=new THREE.MeshBasicMaterial({color:wc});
                    var facing=(x>0)?-1:1;
                    var f=new THREE.Mesh(wGeo,wm);f.position.set(wx,fl,facing*d/2+facing*0.05);if(facing<0)f.rotation.y=Math.PI;g.add(f);
                    var b=new THREE.Mesh(wGeo,wm);b.position.set(wx,fl,-facing*d/2-facing*0.05);if(facing>0)b.rotation.y=Math.PI;g.add(b);
                }
                for(var wz=-d/2+2;wz<d/2-1;wz+=wSp){
                    var lit=Math.random()>0.15;var wc=lit?(Math.random()>0.7?0x88bbff:0xffdd88):0x181210;
                    var wm=new THREE.MeshBasicMaterial({color:wc});
                    var r=new THREE.Mesh(wGeo,wm);r.position.set(w/2+0.05,fl,wz);r.rotation.y=Math.PI/2;g.add(r);
                    var l=new THREE.Mesh(wGeo,wm);l.position.set(-w/2-0.05,fl,wz);l.rotation.y=-Math.PI/2;g.add(l);
                }
            }
            if(h>10){var shopMat=new THREE.MeshBasicMaterial({color:0xffcc77,transparent:true,opacity:0.5});var facing=(x>0)?-1:1;var sg=new THREE.PlaneGeometry(w*0.7,2.5);var sf=new THREE.Mesh(sg,shopMat);sf.position.set(0,1.8,facing*d/2+facing*0.06);if(facing<0)sf.rotation.y=Math.PI;g.add(sf);}
            g.position.set(x,0,z);scene.add(g);
            colliders.push({minX:x-w/2-0.3,maxX:x+w/2+0.3,minZ:z-d/2-0.3,maxZ:z+d/2+0.3});
        }
        var buildingEdge=RW/2+SW+1;
        for(var side=-1;side<=1;side+=2){var bz=-800;while(bz<800){var bw=12+Math.random()*22;var bd=14+Math.random()*25;var bh=18+Math.random()*65;var gap=1+Math.random()*4;var bx=side*(buildingEdge+bw/2+Math.random()*5);mkBldg(bx,bz+bd/2,bw,bd,bh,bPal[Math.floor(Math.random()*bPal.length)]);bz+=bd+gap;}}
        function mkLamp(x,z,rotY){
            var g=new THREE.Group();var pm=new THREE.MeshLambertMaterial({color:0x444444});
            var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.13,5.5,5),pm);pole.position.y=2.75;g.add(pole);
            var arm=new THREE.Mesh(new THREE.BoxGeometry(2.5,0.08,0.08),pm);arm.position.set(1.25,5.5,0);g.add(arm);
            var hous=new THREE.Mesh(new THREE.BoxGeometry(1,0.25,0.4),new THREE.MeshLambertMaterial({color:0x333333}));hous.position.set(2.2,5.38,0);g.add(hous);
            var gp=new THREE.Mesh(new THREE.PlaneGeometry(0.8,0.3),new THREE.MeshBasicMaterial({color:0xffdd88,transparent:true,opacity:0.85,side:THREE.DoubleSide}));gp.position.set(2.2,5.22,0);gp.rotation.x=Math.PI/2;g.add(gp);
            var bulb=new THREE.Mesh(new THREE.SphereGeometry(0.2,6,6),new THREE.MeshBasicMaterial({color:0xffeeaa}));bulb.position.set(2.2,5.15,0);g.add(bulb);
            var pool=new THREE.Mesh(new THREE.CircleGeometry(3.5,8),new THREE.MeshBasicMaterial({color:0xffdd88,transparent:true,opacity:0.06,side:THREE.DoubleSide}));pool.rotation.x=-Math.PI/2;pool.position.set(2.2,0.03,0);g.add(pool);
            var cone=new THREE.Mesh(new THREE.ConeGeometry(2.5,5,8,1,true),new THREE.MeshBasicMaterial({color:0xffdd66,transparent:true,opacity:0.03,side:THREE.DoubleSide}));cone.position.set(2.2,2.7,0);g.add(cone);
            g.position.set(x,0,z);g.rotation.y=rotY||0;scene.add(g);
            colliders.push({minX:x-0.25,maxX:x+0.25,minZ:z-0.25,maxZ:z+0.25});
        }
        for(var lz=-800;lz<800;lz+=30){mkLamp(RW/2+1,lz,0);mkLamp(-RW/2-1,lz+15,Math.PI);}
        function mkTree(x,z){
            var g=new THREE.Group();
            var trunk=new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.2,2.2,5),new THREE.MeshLambertMaterial({color:0x3d2817}));trunk.position.y=1.1;g.add(trunk);
            var lc=[0x1a3a1a,0x1f4420,0x163316][Math.floor(Math.random()*3)];var lm=new THREE.MeshLambertMaterial({color:lc});
            [{r:1.8,h:2.5,y:3.2},{r:1.3,h:2,y:4.8},{r:0.8,h:1.6,y:6}].forEach(function(s){var c=new THREE.Mesh(new THREE.ConeGeometry(s.r,s.h,6),lm);c.position.y=s.y;c.castShadow=true;g.add(c);});
            g.position.set(x,0,z);scene.add(g);colliders.push({minX:x-0.3,maxX:x+0.3,minZ:z-0.3,maxZ:z+0.3});
        }
        for(var tz=-780;tz<780;tz+=14+Math.random()*10){if(Math.random()>0.35){mkTree(RW/2+SW-0.5,tz);mkTree(-RW/2-SW+0.5,tz+7);}}
        function mkCar(bodyCol,cabCol){
            var car=new THREE.Group();
            var body=new THREE.Mesh(new THREE.BoxGeometry(2.2,0.7,4.6),new THREE.MeshLambertMaterial({color:bodyCol}));body.position.set(0,0.55,0);body.castShadow=true;car.add(body);
            var under=new THREE.Mesh(new THREE.BoxGeometry(2.3,0.18,4.7),new THREE.MeshLambertMaterial({color:0x111111}));under.position.set(0,0.24,0);car.add(under);
            var cab=new THREE.Mesh(new THREE.BoxGeometry(1.85,0.6,2.2),new THREE.MeshLambertMaterial({color:cabCol}));cab.position.set(0,1.05,-0.3);cab.castShadow=true;car.add(cab);
            var wsMat=new THREE.MeshBasicMaterial({color:0x88bbdd,transparent:true,opacity:0.55,side:THREE.DoubleSide});
            var ws=new THREE.Mesh(new THREE.PlaneGeometry(1.65,0.5),wsMat);ws.position.set(0,1.1,0.85);ws.rotation.x=Math.PI/6;car.add(ws);
            var rw=new THREE.Mesh(new THREE.PlaneGeometry(1.65,0.5),wsMat);rw.position.set(0,1.1,-1.45);rw.rotation.x=-Math.PI/6;car.add(rw);
            var hlM=new THREE.MeshBasicMaterial({color:0xffffee});
            car.add(new THREE.Mesh(new THREE.CircleGeometry(0.16,8),hlM).clone().translateX(-0.65).translateY(0.5).translateZ(2.31));
            car.add(new THREE.Mesh(new THREE.CircleGeometry(0.16,8),hlM).clone().translateX(0.65).translateY(0.5).translateZ(2.31));
            var beamMat=new THREE.MeshBasicMaterial({color:0xffeecc,transparent:true,opacity:0.1,side:THREE.DoubleSide});
            var beam=new THREE.Mesh(new THREE.PlaneGeometry(1.8,8),beamMat);beam.position.set(0,0.15,6.5);beam.rotation.x=-Math.PI/2;car.add(beam);
            var tlM=new THREE.MeshBasicMaterial({color:0xff2200});
            var tl1=new THREE.Mesh(new THREE.BoxGeometry(0.28,0.12,0.04),tlM);tl1.position.set(-0.72,0.5,-2.31);car.add(tl1);
            var tl2=new THREE.Mesh(new THREE.BoxGeometry(0.28,0.12,0.04),tlM);tl2.position.set(0.72,0.5,-2.31);car.add(tl2);
            var tgM=new THREE.MeshBasicMaterial({color:0xff3300,transparent:true,opacity:0.06,side:THREE.DoubleSide});
            var tgP=new THREE.Mesh(new THREE.PlaneGeometry(1.5,4),tgM);tgP.position.set(0,0.15,-4.5);tgP.rotation.x=-Math.PI/2;car.add(tgP);
            var wg=new THREE.CylinderGeometry(0.35,0.35,0.26,8),wm=new THREE.MeshLambertMaterial({color:0x1a1a1a});
            car.wheels=[];
            [[-1.1,0.35,1.3],[1.1,0.35,1.3],[-1.1,0.35,-1.3],[1.1,0.35,-1.3]].forEach(function(p){
                var w=new THREE.Mesh(wg,wm);w.rotation.z=Math.PI/2;w.position.set(p[0],p[1],p[2]);car.add(w);car.wheels.push(w);
                var hub=new THREE.Mesh(new THREE.CylinderGeometry(0.18,0.18,0.27,5),new THREE.MeshLambertMaterial({color:0x888888}));hub.rotation.z=Math.PI/2;hub.position.set(p[0]+(p[0]>0?0.14:-0.14),p[1],p[2]);car.add(hub);
            });
            return car;
        }
        var playerCar=mkCar(0xcc2222,0xaa1818);playerCar.position.set(-laneW,0,-600);scene.add(playerCar);
        var npcCol=[[0x2255aa,0x1a4488],[0xdddd33,0xbbbb22],[0x22aa44,0x188833],[0xeeeeee,0xcccccc],[0x222222,0x111111],[0xcc6600,0xaa5500],[0x8833aa,0x6622aa],[0x33aaaa,0x228888],[0xaa3355,0x882244],[0x4488cc,0x336699]];
        function spawnNPC(){
            var c=npcCol[Math.floor(Math.random()*npcCol.length)];var npc=mkCar(c[0],c[1]);
            var sameDir=Math.random()>0.35;
            if(sameDir){var lane=-laneW-(Math.random()>0.5?laneW:0);npc.position.set(lane,0,playerCar.position.z-100-Math.random()*600);npc.userData={speed:14+Math.random()*12,dir:1,axis:'z'};npc.rotation.y=0;}
            else{var lane=laneW+(Math.random()>0.5?laneW:0);npc.position.set(lane,0,playerCar.position.z+50+Math.random()*600);npc.userData={speed:14+Math.random()*16,dir:-1,axis:'z'};npc.rotation.y=Math.PI;}
            scene.add(npc);npcCars.push(npc);
        }
        for(var i=0;i<30;i++)spawnNPC();
        var snowOn=false,snowTimer=0,snowDur=0,nextSnow=15+Math.random()*30;
        var SN=2000;var sGeo=new THREE.BufferGeometry();var sPos=new Float32Array(SN*3);var sVel=[];
        for(var i=0;i<SN;i++){sPos[i*3]=(Math.random()-0.5)*80;sPos[i*3+1]=Math.random()*40;sPos[i*3+2]=(Math.random()-0.5)*80;sVel.push({x:(Math.random()-0.5)*1.5,y:-(1+Math.random()*2.5),z:(Math.random()-0.5)*1.5});}
        sGeo.setAttribute('position',new THREE.BufferAttribute(sPos,3));
        var sPts=new THREE.Points(sGeo,new THREE.PointsMaterial({color:0xffffff,size:0.2,transparent:true,opacity:0.7,depthWrite:false}));sPts.visible=false;scene.add(sPts);
        var cs={speed:0,maxSpd:90,accel:30,brake:45,fric:6,turnSpd:1.8,rot:0,wRot:0,colCD:0};
        var ks={f:false,b:false,l:false,r:false,brk:false};
        document.addEventListener('keydown',function(e){switch(e.key.toLowerCase()){case'w':case'arrowup':ks.f=true;break;case's':case'arrowdown':ks.b=true;break;case'a':case'arrowleft':ks.l=true;break;case'd':case'arrowright':ks.r=true;break;case' ':ks.brk=true;e.preventDefault();break;}});
        document.addEventListener('keyup',function(e){switch(e.key.toLowerCase()){case'w':case'arrowup':ks.f=false;break;case's':case'arrowdown':ks.b=false;break;case'a':case'arrowleft':ks.l=false;break;case'd':case'arrowright':ks.r=false;break;case' ':ks.brk=false;break;}});
        var flashEl=document.getElementById('collision-flash');
        function checkCol(nx,nz){var hw=1.2,hd=2.5,sR=Math.sin(cs.rot),cR=Math.cos(cs.rot);var x0=nx-hw-Math.abs(sR)*hd,x1=nx+hw+Math.abs(sR)*hd;var z0=nz-hd-Math.abs(cR)*hw,z1=nz+hd+Math.abs(cR)*hw;for(var i=0;i<colliders.length;i++){var c=colliders[i];if(x1>c.minX&&x0<c.maxX&&z1>c.minZ&&z0<c.maxZ)return true;}for(var i=0;i<npcCars.length;i++){var dx=nx-npcCars[i].position.x,dz=nz-npcCars[i].position.z;if(dx*dx+dz*dz<12)return true;}return false;}
        function doCol(){flashEl.style.opacity=String(Math.min(1,Math.abs(cs.speed)*0.018));setTimeout(function(){flashEl.style.opacity='0';},150);cs.speed*=-0.3;cs.colCD=0.3;}
        var clock=new THREE.Clock();
        var spdEl=document.getElementById('speed'),gearEl=document.getElementById('gear');
        var npcEl=document.getElementById('npc-count'),snowPill=document.getElementById('snow-pill');
        var camOff=new THREE.Vector3(0,3.5,-9);var fc=0;
        camera.position.set(-laneW,3.5,-609);camera.lookAt(-laneW,1.5,-600);
        function animate(){
            requestAnimationFrame(animate);var dt=Math.min(clock.getDelta(),0.05);fc++;
            if(cs.colCD>0)cs.colCD-=dt;
            if(ks.f)cs.speed+=cs.accel*dt;else if(ks.b)cs.speed-=cs.accel*0.7*dt;
            else{if(cs.speed>0)cs.speed=Math.max(0,cs.speed-cs.fric*dt);else if(cs.speed<0)cs.speed=Math.min(0,cs.speed+cs.fric*dt);}
            if(ks.brk){if(cs.speed>0)cs.speed=Math.max(0,cs.speed-cs.brake*2*dt);else cs.speed=Math.min(0,cs.speed+cs.brake*2*dt);}
            cs.speed=Math.max(-cs.maxSpd/3,Math.min(cs.maxSpd,cs.speed));
            if(Math.abs(cs.speed)>0.5){var tf=Math.min(1,Math.abs(cs.speed)/25);if(ks.l)cs.rot+=cs.turnSpd*tf*dt*Math.sign(cs.speed);if(ks.r)cs.rot-=cs.turnSpd*tf*dt*Math.sign(cs.speed);}
            cs.rot*=0.97;cs.rot=Math.max(-0.4,Math.min(0.4,cs.rot));
            var mx=Math.sin(cs.rot)*cs.speed*dt,mz=Math.cos(cs.rot)*cs.speed*dt;
            var nx=playerCar.position.x+mx,nz=playerCar.position.z+mz;
            nx=Math.max(-RW/2+1.5,Math.min(RW/2-1.5,nx));
            if(cs.colCD<=0){if(checkCol(nx,nz))doCol();else{playerCar.position.x=nx;playerCar.position.z=nz;}}
            else{playerCar.position.x=Math.max(-RW/2+1.5,Math.min(RW/2-1.5,playerCar.position.x+mx));playerCar.position.z+=mz;}
            playerCar.rotation.y=cs.rot;cs.wRot+=cs.speed*dt*0.5;playerCar.wheels.forEach(function(w){w.rotation.x=cs.wRot;});
            var hlOff=new THREE.Vector3(0,1.5,5);hlOff.applyAxisAngle(new THREE.Vector3(0,1,0),cs.rot);
            headlight.position.set(playerCar.position.x+hlOff.x,playerCar.position.y+hlOff.y,playerCar.position.z+hlOff.z);
            npcCars.forEach(function(n){var d=n.userData;n.position.z+=d.dir*d.speed*dt;n.wheels.forEach(function(w){w.rotation.x+=d.speed*dt*0.5;});
                if(n.position.z<playerCar.position.z-200){n.position.z=playerCar.position.z+150+Math.random()*400;if(d.dir<0){n.position.x=laneW+(Math.random()>0.5?laneW:0);}else{n.position.x=-laneW-(Math.random()>0.5?laneW:0);}}
                if(d.dir<0&&n.position.z>playerCar.position.z+600){n.position.z=playerCar.position.z+150+Math.random()*300;n.position.x=laneW+(Math.random()>0.5?laneW:0);}
            });
            snowTimer+=dt;
            if(!snowOn&&snowTimer>=nextSnow){snowOn=true;snowTimer=0;snowDur=12+Math.random()*22;sPts.visible=true;snowPill.style.display='flex';}
            if(snowOn){if(snowTimer>=snowDur){snowOn=false;snowTimer=0;nextSnow=18+Math.random()*35;sPts.visible=false;snowPill.style.display='none';}
                var pa=sGeo.getAttribute('position');for(var i=0;i<SN;i++){pa.array[i*3]+=sVel[i].x*dt;pa.array[i*3+1]+=sVel[i].y*dt;pa.array[i*3+2]+=sVel[i].z*dt;
                if(pa.array[i*3+1]<0){pa.array[i*3]=playerCar.position.x+(Math.random()-0.5)*60;pa.array[i*3+1]=20+Math.random()*20;pa.array[i*3+2]=playerCar.position.z+(Math.random()-0.5)*60;}}pa.needsUpdate=true;}
            var dSpd=Math.abs(Math.round(cs.speed*3.6));spdEl.textContent=dSpd;
            if(cs.speed<-0.5)gearEl.textContent='R';else if(Math.abs(cs.speed)<0.5)gearEl.textContent='P';
            else if(dSpd<40)gearEl.textContent='1';else if(dSpd<80)gearEl.textContent='2';else if(dSpd<150)gearEl.textContent='3';else gearEl.textContent='4';
            if(fc%20===0)npcEl.textContent=npcCars.length+' cars';
            var ideal=camOff.clone().applyAxisAngle(new THREE.Vector3(0,1,0),cs.rot);ideal.add(playerCar.position);
            if(fc<10){camera.position.copy(ideal);}else{camera.position.lerp(ideal,1-Math.pow(0.02,dt));}
            var look=playerCar.position.clone();look.y+=1.2;look.z+=4;camera.lookAt(look);
            sunLight.position.set(playerCar.position.x+20,40,playerCar.position.z+150);sunLight.target.position.set(playerCar.position.x,0,playerCar.position.z);
            skyMesh.position.set(playerCar.position.x,0,playerCar.position.z);sunG1.position.set(playerCar.position.x,22,playerCar.position.z+370);
            sunG2.position.set(playerCar.position.x,22,playerCar.position.z+370);haze.position.set(playerCar.position.x,8,playerCar.position.z+370);
            gnd.position.set(playerCar.position.x,0,playerCar.position.z);
            renderer.render(scene,camera);
        }
        animate();
        window.addEventListener('resize',function(){camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);});
        document.body.tabIndex=0;document.body.focus();
    })();
    </script>
</body>
</html>'''

    def _f03_retreat_face(self):
        """Slowly slide face mesh to the top-right corner (very slow, ~6s)."""
        print("[Onboarding] → retreating face mesh")
        if not hasattr(self, '_ob_face_widget') or not self._ob_face_widget:
            return

        win_rect = self.mw.rect()
        fw = self._ob_face_widget.width()
        fh = self._ob_face_widget.height()

        # Far top-right corner, partially off-screen
        final_x = win_rect.width() - fw // 2
        final_y = -fh // 4

        start_pos = self._ob_face_widget.pos()

        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        self._face_retreat_anim = QPropertyAnimation(self._ob_face_widget, b"pos")
        self._face_retreat_anim.setDuration(6000)  # very slow 6s
        self._face_retreat_anim.setStartValue(start_pos)
        self._face_retreat_anim.setEndValue(QPoint(final_x, final_y))
        self._face_retreat_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._face_retreat_anim.start()

        # Also fade out gaze cursor
        if self._ob_gaze_cursor:
            self._ob_gaze_cursor.hide()

    def _f04_reverse_tilt(self):
        """Reverse the view tilt back to normal over ~25s (rests at 2m04).
        Undoes the zoom-out and perspective tilt to focus on center game."""
        from PySide6.QtGui import QTransform
        print("[Onboarding] → reversing view tilt back to normal")

        view = self.view
        # Capture current transform
        ct = view.transform()
        start_m11 = ct.m11(); start_m12 = ct.m12()
        start_m21 = ct.m21(); start_m22 = ct.m22()
        start_dx = ct.dx(); start_dy = ct.dy()

        # Target: identity transform (normal view)
        dur_s = 25.0  # 1m39 → 2m04
        t0 = time.perf_counter()

        # Center on the game (at the spread centroid)
        game_center = getattr(self, '_game_center', None)
        if game_center is None:
            game_center = getattr(self, '_spread_center', None)
        if game_center is None:
            tp = self._tp()
            game_center = QPointF(tp.x() + self._tw / 2, tp.y() + self._th / 2)

        def tick():
            p = min(1.0, (time.perf_counter() - t0) / dur_s)
            # Ease-in-out for smooth transition
            ep = p * p * (3 - 2 * p)

            # Interpolate transform components toward identity
            m11 = start_m11 + (1.0 - start_m11) * ep
            m12 = start_m12 + (0.0 - start_m12) * ep
            m21 = start_m21 + (0.0 - start_m21) * ep
            m22 = start_m22 + (1.0 - start_m22) * ep

            t = QTransform(m11, m12, m21, m22, 0, 0)
            view.setTransform(t)
            view.centerOn(game_center)

            if p >= 1.0:
                ti.stop()
                # Reset to clean identity
                view.setTransform(QTransform())
                view.centerOn(game_center)
                print("[Onboarding] ✓ View restored @ 2m04")
                _precise_singleshot(500, self._f05_welcome_reset)

        ti = QTimer(self); ti.timeout.connect(tick); ti.start(30); self._keep.append(ti)

    def _f05_welcome_reset(self):
        """Re-show the WelcomeScreen logo. While it displays, clean up all scene widgets."""
        print("[Onboarding] → welcome screen reset")

        # Create a fresh WelcomeScreen
        self._final_welcome = WelcomeScreen(self.logo_path)
        self._final_welcome.finished.connect(self._f07_finale_done)

        # Override the fade-out to give us time to clean up
        self._final_welcome.show_welcome()

        # Start cleaning up scene widgets behind the welcome screen
        _precise_singleshot(800, self._f06_cleanup_scene)

    def _f06_cleanup_scene(self):
        """Remove all onboarding widgets from the scene one by one with fade-outs."""
        print("[Onboarding] → cleaning up scene widgets")

        # Collect all proxies to remove
        proxies_to_remove = []

        # Spawned widgets
        for widget, proxy in self._spawned_widgets:
            proxies_to_remove.append(proxy)

        # Terminal
        if self.terminal_proxy:
            proxies_to_remove.append(self.terminal_proxy)

        # Button
        if self._btn_proxy:
            proxies_to_remove.append(self._btn_proxy)

        # SVG
        if hasattr(self, '_svg_proxy') and self._svg_proxy:
            proxies_to_remove.append(self._svg_proxy)

        # Eyes
        if hasattr(self, '_eyes_proxy') and self._eyes_proxy:
            proxies_to_remove.append(self._eyes_proxy)

        # Game
        if hasattr(self, '_game_proxy') and self._game_proxy:
            proxies_to_remove.append(self._game_proxy)

        # Remove one by one with 200ms stagger
        for i, proxy in enumerate(proxies_to_remove):
            delay = i * 200
            _precise_singleshot(delay, lambda p=proxy: self._fade_and_remove(p))

        # After all removed, clean up face mesh overlay
        total_cleanup = len(proxies_to_remove) * 200 + 500
        _precise_singleshot(total_cleanup, self._f06b_cleanup_overlays)

    def _fade_and_remove(self, proxy):
        """Fade out a proxy then remove from scene."""
        try:
            if not proxy.scene():
                return
            self._fout(proxy, 300, cb=lambda: self._safe_remove(proxy))
        except RuntimeError:
            pass

    def _safe_remove(self, proxy):
        """Safely remove a proxy from the scene."""
        try:
            if proxy.scene():
                self.scene.removeItem(proxy)
        except RuntimeError:
            pass

    def _f06b_cleanup_overlays(self):
        """Remove face mesh overlay, gaze cursor, stop camera."""
        print("[Onboarding] → cleaning up overlays")

        # Stop camera
        if hasattr(self, '_ob_camera') and self._ob_camera:
            try:
                self._ob_camera.stop()
            except Exception:
                pass

        # Remove face mesh widget
        if hasattr(self, '_ob_face_widget') and self._ob_face_widget:
            try:
                self._ob_face_widget.hide()
                self._ob_face_widget.deleteLater()
            except RuntimeError:
                pass

        # Remove gaze cursor
        if hasattr(self, '_ob_gaze_cursor') and self._ob_gaze_cursor:
            try:
                self._ob_gaze_cursor.hide()
                self._ob_gaze_cursor.deleteLater()
            except RuntimeError:
                pass

        # Stop music if still playing
        if self.music_player:
            try:
                self.music_player.stop()
            except Exception:
                pass

        # Clean up timer references
        self._cleanup_stopped_timers()

        # Background will be reset to (250,250,250) in _f07_finale_done

        # Clear tracking lists
        self._spawned_widgets.clear()
        self._keep.clear()

        print("[Onboarding] ✓ Scene cleanup complete")

    def _f07_finale_done(self):
        """Called when the final welcome screen finishes — onboarding is complete."""
        print("[Onboarding] ✓ Onboarding complete!")

        # Reset background to light default
        light_bg = QColor(250, 250, 250)
        self.scene.setBackgroundBrush(QBrush(light_bg))
        self.view.setBackgroundBrush(QBrush(light_bg))

        # Toggle dark mode off if it was on
        if self.mw._dark_mode:
            self.mw.toggle_dark_mode()

        # Clean up references
        if hasattr(self, '_final_welcome'):
            self._final_welcome = None

    # ── Frame processing (head gaze + hand pinch) ────────────────────

    def _m_on_frame(self, frame, face_landmarks, hands_data):
        """Called ~30 fps. Face → pointer. Hands → pinch."""
        import math

        # Face mesh visualisation
        if hasattr(self, '_ob_face_widget') and self._ob_face_widget:
            self._ob_face_widget.set_face_data(frame, face_landmarks)

        # Head gaze → cursor
        win_w = self.mw.width()
        win_h = self.mw.height()
        gaze_pos = self._ob_gaze_tracker.compute_gaze(face_landmarks, win_w, win_h)

        if gaze_pos is not None:
            # Widget gravity — attract toward nearby proxies
            gaze_pos = self._m_apply_gravity(gaze_pos)

            # Post-gravity glide
            if self._ob_glide_x is None:
                self._ob_glide_x = gaze_pos.x()
                self._ob_glide_y = gaze_pos.y()
            else:
                self._ob_glide_x += self._ob_glide_alpha * (gaze_pos.x() - self._ob_glide_x)
                self._ob_glide_y += self._ob_glide_alpha * (gaze_pos.y() - self._ob_glide_y)
            gaze_pos = QPointF(self._ob_glide_x, self._ob_glide_y)

            self._ob_gaze_pos = gaze_pos
            if self._ob_gaze_cursor:
                self._ob_gaze_cursor.move(
                    int(gaze_pos.x() - self._ob_gaze_cursor.width() / 2),
                    int(gaze_pos.y() - self._ob_gaze_cursor.height() / 2),
                )
                if not self._ob_gaze_cursor.isVisible():
                    self._ob_gaze_cursor.show()
                    self._ob_gaze_cursor.raise_()
        else:
            if self._ob_gaze_cursor and self._ob_gaze_cursor.isVisible():
                self._ob_gaze_cursor.hide()

        # Pinch detection
        changed = self._ob_pinch_detector.update(hands_data)
        now_pinching = self._ob_pinch_detector.is_pinching

        if self._ob_gaze_cursor:
            self._ob_gaze_cursor.set_pinching(now_pinching)

        if changed:
            if now_pinching and not self._ob_is_pinching:
                if self._ob_gaze_pos is not None:
                    self._m_try_select(self._ob_gaze_pos.x(), self._ob_gaze_pos.y())
            elif not now_pinching and self._ob_is_pinching:
                self._m_release_pinch()

        self._ob_is_pinching = now_pinching

        # Pinch hold → drag with gaze
        if self._ob_is_pinching and self._ob_pinch_item is not None and self._ob_gaze_pos is not None:
            view_pos = QPoint(int(self._ob_gaze_pos.x()), int(self._ob_gaze_pos.y()))
            view_local = self.view.mapFromParent(view_pos)
            scene_pos = self.view.mapToScene(view_local)
            try:
                self._ob_pinch_item.setPos(scene_pos - self._ob_pinch_offset)
            except RuntimeError:
                self._ob_pinch_item = None

    _GRAVITY_RADIUS = 1000
    _GRAVITY_STRENGTH = 1.0
    _SNAP_RADIUS = 200

    def _m_apply_gravity(self, gaze_pos):
        """Bend gaze toward nearest visible proxy widget."""
        import math
        best_centre = None
        best_dist = float("inf")

        for item in self.scene.items():
            if not isinstance(item, QGraphicsProxyWidget):
                continue
            if not item.isVisible():
                continue
            scene_centre = item.mapToScene(item.boundingRect().center())
            view_local = self.view.mapFromScene(scene_centre)
            win_pt = self.view.mapToParent(view_local)
            dx = gaze_pos.x() - win_pt.x()
            dy = gaze_pos.y() - win_pt.y()
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_centre = QPointF(win_pt.x(), win_pt.y())

        if best_centre is None or best_dist > self._GRAVITY_RADIUS:
            return gaze_pos
        if best_dist < self._SNAP_RADIUS:
            return best_centre

        t = 1.0 - (best_dist - self._SNAP_RADIUS) / (self._GRAVITY_RADIUS - self._SNAP_RADIUS)
        t = t * t
        pull = t * self._GRAVITY_STRENGTH
        return QPointF(
            gaze_pos.x() + (best_centre.x() - gaze_pos.x()) * pull,
            gaze_pos.y() + (best_centre.y() - gaze_pos.y()) * pull,
        )

    def _m_try_select(self, win_x, win_y):
        """On pinch start, find topmost proxy under gaze and select it."""
        view_local = self.view.mapFromParent(QPoint(int(win_x), int(win_y)))
        scene_pos = self.view.mapToScene(view_local)

        items = self.scene.items(scene_pos)
        proxy = None
        for item in items:
            p = item
            while p is not None and not isinstance(p, QGraphicsProxyWidget):
                p = p.parentItem()
            if p is not None:
                proxy = p; break

        if proxy is None:
            return

        self._ob_pinch_item = proxy
        self._ob_pinch_offset = scene_pos - proxy.pos()
        self._ob_selected_proxy = proxy

        # Glow effect
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(50)
        glow.setColor(QColor(100, 180, 255, 220))
        glow.setOffset(45, 45)
        proxy.setGraphicsEffect(glow)

    def _m_release_pinch(self):
        """Deselect proxy and restore shadow."""
        proxy = self._ob_selected_proxy
        if proxy is None:
            return
        try:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            dark = self.mw._dark_mode
            shadow.setColor(QColor(255, 255, 255, 160) if dark else QColor(0, 0, 0, 180))
            shadow.setOffset(45, 45)
            proxy.setGraphicsEffect(shadow)
        except RuntimeError:
            pass
        self._ob_selected_proxy = None
        self._ob_pinch_item = None

    # ==================================================================
    # SHOW AI VOICE (during SVG drawing)
    # ==================================================================

    def _z18_voice_menu(self):
        """Open context menu again, this time highlighting 'Show AI Voice'."""
        print("[Onboarding] → voice menu")
        self._fake_menu2 = QMenu(self.mw)
        self._fake_menu2.setStyleSheet(self._CSS_NORMAL)
        self._fake_menu2.addAction("New Terminal")
        self._fake_menu2.addSeparator()
        self._voice_action = self._fake_menu2.addAction("Show AI Voice")
        self._fake_menu2.addAction("Onboarding")
        self._fake_menu2.addSeparator(); self._fake_menu2.addAction("Dark Mode")
        self._fake_menu2.addSeparator(); self._fake_menu2.addAction("Immersive Mode (Ctrl+I)")
        self._fake_menu2.addSeparator()
        self._fake_menu2.addAction("Clear Scene"); self._fake_menu2.addAction("Refresh")
        self._fake_menu2.addAction("Delete Widget"); self._fake_menu2.addAction("Pop Widget")
        self._fake_menu2.addSeparator(); self._fake_menu2.addAction("Fullscreen")
        self._fake_menu2.addSeparator(); self._fake_menu2.addAction("Exit")
        # Pop up near the top of the terminal area (above content)
        vp = self.view.mapFromScene(QPointF(self._ts.x() + self._tw/2, self._ts.y() - 10))
        gp = self.view.mapToGlobal(vp)
        self._fake_menu2.popup(gp)
        _precise_singleshot(600, self._z19_voice_highlight)

    def _z19_voice_highlight(self):
        """Highlight 'Show AI Voice' in the menu."""
        print("[Onboarding] → highlight voice")
        if self._fake_menu2 and self._voice_action:
            r = self._fake_menu2.actionGeometry(self._voice_action)
            QCursor.setPos(self._fake_menu2.mapToGlobal(r.center()))
        _precise_singleshot(400, self._z20_voice_blink)

    def _z20_voice_blink(self):
        """Blink menu, then close and spawn eyes."""
        if not self._fake_menu2:
            _precise_singleshot(10, self._z21_voice_close)
            return
        m = self._fake_menu2
        blink_t0 = time.perf_counter()
        phase = [0]
        def t():
            elapsed_ms = (time.perf_counter() - blink_t0) * 1000
            if phase[0] == 0 and elapsed_ms >= 80:
                m.setStyleSheet(self._CSS_FLASH); phase[0] = 1
            elif phase[0] == 1 and elapsed_ms >= 160:
                m.setStyleSheet(self._CSS_NORMAL); phase[0] = 2
            elif phase[0] == 2 and elapsed_ms >= 240:
                ti.stop(); _precise_singleshot(150, self._z21_voice_close)
        ti = QTimer(self); ti.timeout.connect(t); ti.start(16); self._keep.append(ti)

    def _z21_voice_close(self):
        """Close menu, then create the eyes widget."""
        if self._fake_menu2:
            self._fake_menu2.close()
            self._fake_menu2.deleteLater()
            self._fake_menu2 = None
        self._voice_action = None
        _precise_singleshot(300, self._z22_create_eyes)

    def _z22_create_eyes(self):
        """Create AIVoiceControlWidget (visual only, no LLM) and open the eyes."""
        print("[Onboarding] → create fake eyes")
        try:
            from rio.ai_voice_control import AIVoiceControlWidget
        except ImportError:
            print("[Onboarding] ai_voice_control not available, skipping eyes")
            return

        # Create eyes widget — visual only, no backend connection
        self._eyes_widget = AIVoiceControlWidget(
            scale_factor=0.40,
            llmfs_mount=self.mw.rio_server.llmfs_mount,
            rio_mount=self.mw.rio_server.rio_mount,
        )

        # Add to scene, position above the terminal (top-left area)
        self._eyes_proxy = self.scene.addWidget(self._eyes_widget)
        self._eyes_proxy.setZValue(1500)

        # Position: above terminal, centered horizontally
        if self.terminal_proxy:
            tp = self.terminal_proxy.pos()
            self._eyes_proxy.setPos(
                tp.x() + self._tw / 2 - self._eyes_widget.width() / 2,
                tp.y() - self._eyes_widget.height() - 30
            )
        else:
            c = self._c()
            self._eyes_proxy.setPos(c.x() - 140, c.y() - 350)

        # Fade in
        self._fin(self._eyes_proxy, 500)

        # Visually open the eyes (toggle animation without sending ctl commands)
        # Override _send_ctl to no-op so no LLM calls happen
        self._eyes_widget._send_ctl = lambda cmd: None
        self._eyes_widget._ensure_code_route = lambda: None
        self._eyes_widget._teardown_code_route = lambda: None

        _precise_singleshot(400, self._z23_open_eyes)

    def _z23_open_eyes(self):
        """Trigger the eye-opening animation."""
        if not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return
        print("[Onboarding] → opening eyes")
        # Simulate eye opening: set state and start toggle animation
        ew = self._eyes_widget
        ew.is_animating = True
        ew.blink_direction = -1
        ew.eyes_are_closed = False
        ew.is_recording = True
        # Start the mouse poll timer for eye tracking
        try:
            ew._mouse_poll_timer.start(33)
        except Exception:
            pass
        # Start toggle animation
        ew.blink_timer.stop()
        try:
            ew.blink_timer.timeout.disconnect()
        except RuntimeError:
            pass
        ew.blink_timer.timeout.connect(ew._update_toggle_animation)
        ew.blink_timer.start(16)

        # Start autonomous animations (random gaze, blink, tilt) once eyes
        # are fully open — short delay to let the open animation settle
        _precise_singleshot(600, self._z23b_start_eye_animations)

        self.terminal.append_output(
            "\n 👁 AI Voice agent activated.\n",
            color=self.terminal.C_SUCCESS
        )

    def _z23b_start_eye_animations(self):
        """Kick off random autonomous eye animations (gaze, blink, tilt)."""
        if not hasattr(self, '_eyes_widget') or not self._eyes_widget:
            return
        ew = self._eyes_widget
        if ew.eyes_are_closed:
            return
        print("[Onboarding] → starting autonomous eye animations")
        try:
            ew._start_autonomous_animations()
        except Exception as e:
            print(f"[Onboarding] autonomous animation error: {e}")

        # Schedule SVG + view tilt at exactly 53.5s from music start
        self._at_music_time(53.5, self._z24_svg_beside_eyes)


# ============================================================================
# Launch
# ============================================================================

main_window._onboarding_controller = OnboardingController(
    main_window, graphics_scene, graphics_view,
    _logo_path, _music_path, _svg_path
)
main_window._onboarding_controller.start()
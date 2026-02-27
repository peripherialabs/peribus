"""
Immersive Mode for Rio Display Server

Toggle with Ctrl+I. Orchestrates:
  1. Add drop shadows to all proxies (animate offset 0,0 → 45,45)
  2. Animate view tilt (slower, cinematic)
  3. Switch to dark mode
  4. Slide in the face mesh widget from top-right
  5. Head-based gaze pointer (nose tip direction controls cursor)
  6. Right-hand thumb+middle pinch to select/grab widgets

Deactivation reverses everything.

Head pointer rationale:
  Hand tracking is inherently jittery — finger tip landmarks wobble
  as the hand deforms during gestures, and smoothing adds latency
  that makes it feel unresponsive.  The head, by contrast, is a rigid
  body: nose-tip position is extremely stable frame-to-frame.  We use
  the nose tip (landmark 1) relative to the face bounding-box center
  to derive a gaze vector that maps onto the screen.  This gives a
  natural, steady pointer — you look at what you want to interact
  with, then pinch to grab it.
"""

import time
import math
import sys

from PySide6.QtWidgets import (
    QWidget, QGraphicsDropShadowEffect, QGraphicsProxyWidget,
    QGraphicsItem,
)
from PySide6.QtCore import (
    QObject, Signal, QTimer, QPoint, QPointF, Qt,
    Property, QPropertyAnimation, QEasingCurve, QRectF,
)
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QTransform,
)

try:
    import numpy as np
    import cv2
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    print("[ImmersiveMode] cv2/mediapipe/numpy not installed — camera features disabled")


# ═══════════════════════════════════════════════════════════════════════
# Camera Manager
# ═══════════════════════════════════════════════════════════════════════

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
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh_model = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_hands = mp.solutions.hands
            self.hands_model = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            return True
        except Exception as e:
            print(f"[ImmersiveMode] MediaPipe init error: {e}")
            return False

    def start(self) -> bool:
        if not _HAS_MEDIAPIPE:
            return False
        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened():
            self.is_running = True
            self.timer.start(33)  # ~30 fps
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
                label = hc.classification[0].label  # "Left" or "Right"
                hands_data.append((hl, label))

        self.frame_ready.emit(frame, face_landmarks, hands_data)

    def stop(self):
        self.is_running = False
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None


# ═══════════════════════════════════════════════════════════════════════
# Face Mesh Widget  (lives as a direct child of main_window, NOT scene)
# ═══════════════════════════════════════════════════════════════════════

class FaceMeshWidget(QWidget):
    """Draws the MediaPipe face mesh with isometric 3D rotation.

    Sized generously so the mesh never clips.
    """

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

        # Smoothing for the face mesh rendering (visual only, NOT the pointer)
        self.smoothing_factor = 0.45

        # 3D rotation
        self.rotation_y = 0.4
        self.rotation_x = 0.3

        # Isometric projection
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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self.smoothed_landmarks or self.current_frame is None:
            painter.end()
            return

        w, h = self.width(), self.height()

        if _HAS_MEDIAPIPE:
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
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


# ═══════════════════════════════════════════════════════════════════════
# Gaze Cursor Widget
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Head Gaze Tracker
# ═══════════════════════════════════════════════════════════════════════

class HeadGazeTracker:
    """Converts face landmark data into a *stable* screen-space gaze point.

    Design goals:
      - Feel like a slow, deliberate laser-pointer — not a twitchy cursor.
      - Micro-movements (breathing, talking) are fully absorbed by a dead
        zone + heavy smoothing.
      - Intentional head turns travel the full screen range.

    Pipeline:
      1. Nose-tip displacement from face-box centre (normalised, mirrored).
      2. Dead zone: displacements below a threshold are snapped to zero.
      3. Sensitivity scaling (kept low: 2.0 / 1.6).
      4. Double-pass exponential moving average (EMA):
         - First pass:  alpha = 0.08  (very heavy, kills all jitter)
         - Second pass:  alpha = 0.15  (smooths out the first pass's steps)
         This creates a cursor that glides rather than jumps.
      5. Clamp to [0, 1] and map to window coordinates.
    """

    NOSE_TIP   = 1
    LEFT_EAR   = 234
    RIGHT_EAR  = 454
    FOREHEAD   = 10
    CHIN       = 152

    def __init__(self,
                 sensitivity_x: float = 2.8,
                 sensitivity_y: float = 3.0,
                 dead_zone: float = 0.025,
                 ema_alpha_1: float = 0.40,
                 ema_alpha_2: float = 0.50):
        self.sensitivity_x = sensitivity_x
        self.sensitivity_y = sensitivity_y
        self.dead_zone = dead_zone

        # Double EMA state
        self._a1 = ema_alpha_1   # first pass — very heavy
        self._a2 = ema_alpha_2   # second pass — glide
        self._s1_x = None  # first EMA output
        self._s1_y = None
        self._s2_x = None  # second EMA output (what we return)
        self._s2_y = None

    def compute_gaze(self, face_landmarks, win_w: int, win_h: int):
        """Return a window-coordinate QPointF or None."""
        if face_landmarks is None:
            return self._last_pos(win_w, win_h)

        lm = face_landmarks.landmark
        if len(lm) < 468:
            return self._last_pos(win_w, win_h)

        nose     = lm[self.NOSE_TIP]
        l_ear    = lm[self.LEFT_EAR]
        r_ear    = lm[self.RIGHT_EAR]
        forehead = lm[self.FOREHEAD]
        chin     = lm[self.CHIN]

        # Face bounding box centre
        face_cx = (l_ear.x + r_ear.x) / 2.0
        face_cy = (forehead.y + chin.y) / 2.0

        face_w = abs(r_ear.x - l_ear.x)
        face_h = abs(chin.y - forehead.y)
        if face_w < 0.01 or face_h < 0.01:
            return self._last_pos(win_w, win_h)

        # Normalised displacement of nose from face centre
        dx = (nose.x - face_cx) / face_w
        dy = (nose.y - face_cy) / face_h

        # Dead zone: ignore tiny displacements (breathing, micro-sway)
        if abs(dx) < self.dead_zone:
            dx = 0.0
        else:
            # Subtract dead zone so movement starts from zero at the edge
            dx = (abs(dx) - self.dead_zone) * (1.0 if dx > 0 else -1.0)

        if abs(dy) < self.dead_zone:
            dy = 0.0
        else:
            dy = (abs(dy) - self.dead_zone) * (1.0 if dy > 0 else -1.0)

        # Map to screen (mirrored X for natural feel)
        raw_x = 0.5 - dx * self.sensitivity_x
        raw_y = 0.5 + dy * self.sensitivity_y

        raw_x = max(0.0, min(1.0, raw_x))
        raw_y = max(0.0, min(1.0, raw_y))

        # ── Double EMA ──
        if self._s1_x is None:
            # First frame: seed both passes
            self._s1_x, self._s1_y = raw_x, raw_y
            self._s2_x, self._s2_y = raw_x, raw_y
        else:
            # Pass 1: heavy smooth
            self._s1_x += self._a1 * (raw_x - self._s1_x)
            self._s1_y += self._a1 * (raw_y - self._s1_y)
            # Pass 2: glide smooth on top of pass 1
            self._s2_x += self._a2 * (self._s1_x - self._s2_x)
            self._s2_y += self._a2 * (self._s1_y - self._s2_y)

        return QPointF(self._s2_x * win_w, self._s2_y * win_h)

    def _last_pos(self, win_w, win_h):
        """Return the last known position if we have one, otherwise None.
        This keeps the cursor visible during brief face-detection dropouts."""
        if self._s2_x is not None:
            return QPointF(self._s2_x * win_w, self._s2_y * win_h)
        return None

    def reset(self):
        self._s1_x = self._s1_y = None
        self._s2_x = self._s2_y = None


# ═══════════════════════════════════════════════════════════════════════
# Pinch Detector (boolean only — no positional data from hands)
# ═══════════════════════════════════════════════════════════════════════

class PinchDetector:
    """Detects pinch on the right hand.

    Triggers on either:
      - Thumb tip + Middle finger tip
      - Thumb tip + Ring finger tip
    Whichever pair is closer. Only returns a boolean.
    """
    THUMB_TIP  = 4
    MIDDLE_TIP = 12
    RING_TIP   = 16

    def __init__(self, threshold: float = 0.055):
        self.threshold = threshold
        self.is_pinching = False

    def update(self, hands_data) -> bool:
        """Update pinch state.  Returns True if state changed."""
        right_hand = None
        for hl, label in (hands_data or []):
            if label == "Left":  # user's right hand in mirror
                right_hand = hl
                break
        if right_hand is None and hands_data:
            right_hand = hands_data[0][0]

        was = self.is_pinching

        if right_hand is None:
            self.is_pinching = False
            return was != self.is_pinching

        lm = right_hand.landmark
        thumb  = lm[self.THUMB_TIP]
        middle = lm[self.MIDDLE_TIP]
        ring   = lm[self.RING_TIP]

        dist_middle = math.sqrt(
            (thumb.x - middle.x) ** 2 +
            (thumb.y - middle.y) ** 2 +
            (thumb.z - middle.z) ** 2
        )
        dist_ring = math.sqrt(
            (thumb.x - ring.x) ** 2 +
            (thumb.y - ring.y) ** 2 +
            (thumb.z - ring.z) ** 2
        )

        self.is_pinching = min(dist_middle, dist_ring) < self.threshold
        return was != self.is_pinching


# ═══════════════════════════════════════════════════════════════════════
# Safe C++ object guard
# ═══════════════════════════════════════════════════════════════════════

def _shadow_alive(shadow) -> bool:
    """Check if a QGraphicsDropShadowEffect C++ backend is still alive."""
    try:
        shadow.blurRadius()
        return True
    except (RuntimeError, AttributeError):
        return False


# ═══════════════════════════════════════════════════════════════════════
# Immersive Mode Controller
# ═══════════════════════════════════════════════════════════════════════

class ImmersiveMode(QObject):
    """Main orchestrator.  Attached to RioWindow.  Toggle via Ctrl+I."""

    _progress_val = 0.0

    def _get_progress(self):
        return self._progress_val

    def _set_progress(self, v):
        self._progress_val = v

    progress = Property(float, _get_progress, _set_progress)

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.graphics_scene = main_window.graphics_scene
        self.graphics_view = main_window.graphics_view

        self.is_active = False
        self._activating = False

        # Sub-components
        self.camera = CameraManager()
        self.face_widget = None
        self.gaze_cursor = None

        self._mediapipe_ready = False

        # Trackers
        self.gaze_tracker = HeadGazeTracker()
        self.pinch_detector = PinchDetector()

        # Post-gravity smooth glide — makes transitions between widgets
        # feel like a deliberate linear slide rather than a jump
        self._glide_x = None
        self._glide_y = None
        self._glide_alpha = 0.20  # faster glide between widgets

        # State
        self._is_pinching = False
        self._pinch_item = None
        self._pinch_offset = QPointF()
        self._gaze_pos = None

        # Shadow bookkeeping
        self._shadow_timers = []
        self._added_shadows = {}  # id(item) → shadow

        # Selection
        self._selected_proxy = None
        self._pre_select_pos = None

    # ════════════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════════════

    def toggle(self):
        if self._activating:
            return
        if self.is_active:
            self._deactivate()
        else:
            self._activate()

    # ════════════════════════════════════════════════════════════════
    # Activation sequence
    # ════════════════════════════════════════════════════════════════

    def _activate(self):
        if self.is_active or self._activating:
            return
        self._activating = True
        self.is_active = True
        print("[ImmersiveMode] Activating …")

        # Step 1: shadows 0→45
        self._add_and_animate_shadows(target_offset=45, duration_ms=2200)

        # Step 2: cinematic view tilt
        QTimer.singleShot(600, self._animate_view_tilt)

        # Step 3: dark mode
        QTimer.singleShot(1400, self._ensure_dark_mode)

        # Step 4: start camera early (off-screen) so mesh is ready before slide-in
        QTimer.singleShot(2400, self._start_tracking_early)

        # Step 5: slide in the face mesh widget (mesh already rendering off-screen)
        QTimer.singleShot(4000, self._introduce_face_mesh)

        QTimer.singleShot(5200, self._activation_done)

    def _activation_done(self):
        self._activating = False
        print("[ImmersiveMode] Activation complete")

    # ── Step helpers ──

    def _add_and_animate_shadows(self, target_offset, duration_ms):
        for item in list(self.graphics_scene.items()):
            if not isinstance(item, QGraphicsProxyWidget):
                continue
            effect = item.graphicsEffect()
            if effect is None:
                shadow = QGraphicsDropShadowEffect()
                shadow.setBlurRadius(38)
                shadow.setColor(QColor(0, 0, 0, 120))
                shadow.setOffset(0, 0)
                item.setGraphicsEffect(shadow)
                self._added_shadows[id(item)] = shadow
                effect = shadow

            if isinstance(effect, QGraphicsDropShadowEffect) and _shadow_alive(effect):
                self._animate_shadow_offset(
                    effect,
                    start_x=effect.xOffset(), start_y=effect.yOffset(),
                    end_x=target_offset, end_y=target_offset,
                    duration_ms=duration_ms,
                    easing=QEasingCurve.OutInCirc,
                )

    def _animate_shadow_offset(self, shadow, start_x, start_y,
                                end_x, end_y, duration_ms, easing):
        """Timer-driven shadow offset animation. Guarded against C++ deletion."""
        steps = max(1, duration_ms // 16)
        step = [0]
        curve = QEasingCurve(easing)

        def tick():
            if not _shadow_alive(shadow):
                timer.stop()
                if timer in self._shadow_timers:
                    self._shadow_timers.remove(timer)
                timer.deleteLater()
                return

            if step[0] > steps:
                try:
                    shadow.setOffset(end_x, end_y)
                except RuntimeError:
                    pass
                timer.stop()
                if timer in self._shadow_timers:
                    self._shadow_timers.remove(timer)
                timer.deleteLater()
                return

            t = curve.valueForProgress(step[0] / steps)
            ox = start_x + (end_x - start_x) * t
            oy = start_y + (end_y - start_y) * t
            try:
                shadow.setOffset(ox, oy)
            except RuntimeError:
                timer.stop()
                if timer in self._shadow_timers:
                    self._shadow_timers.remove(timer)
                timer.deleteLater()
                return
            step[0] += 1

        timer = QTimer(self)
        self._shadow_timers.append(timer)
        timer.timeout.connect(tick)
        timer.start(16)

    def _animate_view_tilt(self):
        vp = self.graphics_view.viewport()
        cx, cy = vp.width() / 2.0, vp.height() / 2.0

        t = QTransform()
        t.translate(cx, cy)
        t.scale(0.52, 0.42)
        t.shear(0.0, -0.13)
        t.translate(-cx, -cy)

        self.main_window._animate_view_transform(t, QPointF(0, 0), duration=2800)

    def _ensure_dark_mode(self):
        if not self.main_window._dark_mode:
            self.main_window.toggle_dark_mode()

    def _start_tracking_early(self):
        """Create face widget off-screen and start camera so the mesh
        is already rendering before the slide-in animation."""
        # Create widgets if needed
        if self.face_widget is None:
            self.face_widget = FaceMeshWidget(self.main_window)
        if self.gaze_cursor is None:
            self.gaze_cursor = GazeCursorWidget(self.main_window)
            self.gaze_cursor.hide()

        # Park off-screen (top-right corner, outside view)
        win_rect = self.main_window.rect()
        start_x = win_rect.width() + 80
        start_y = -self.face_widget.height() - 80
        self.face_widget.move(start_x, start_y)
        self.face_widget.show()
        self.face_widget.raise_()

        # Start camera + mediapipe so frames start flowing
        if not _HAS_MEDIAPIPE:
            print("[ImmersiveMode] No mediapipe — skipping camera")
            return
        if not self._mediapipe_ready:
            self._mediapipe_ready = self.camera.initialize_mediapipe()
        if not self._mediapipe_ready:
            return

        self.camera.frame_ready.connect(self._on_frame)
        self.camera.start()

    def _introduce_face_mesh(self):
        """Animate the already-rendering face widget into view."""
        if self.face_widget is None:
            return

        win_rect = self.main_window.rect()
        fw, fh = self.face_widget.width(), self.face_widget.height()

        final_x = win_rect.width() - fw - 20
        final_y = 20

        # Current position (off-screen where _start_tracking_early parked it)
        start_pos = self.face_widget.pos()

        self._face_anim = QPropertyAnimation(self.face_widget, b"pos")
        self._face_anim.setDuration(1800)
        self._face_anim.setStartValue(start_pos)
        self._face_anim.setEndValue(QPoint(final_x, final_y))
        self._face_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._face_anim.start()

        # Show gaze cursor once face is sliding in
        if self.gaze_cursor:
            self.gaze_cursor.show()
            self.gaze_cursor.raise_()

    # ════════════════════════════════════════════════════════════════
    # Deactivation
    # ════════════════════════════════════════════════════════════════

    def _deactivate(self):
        if not self.is_active:
            return
        self._activating = True
        print("[ImmersiveMode] Deactivating …")

        self.camera.stop()
        try:
            self.camera.frame_ready.disconnect(self._on_frame)
        except RuntimeError:
            pass

        self._release_pinch()
        self.gaze_tracker.reset()
        self._gaze_pos = None
        self._glide_x = None
        self._glide_y = None

        if self.gaze_cursor:
            self.gaze_cursor.hide()

        if self.face_widget and self.face_widget.isVisible():
            win_rect = self.main_window.rect()
            cur_pos = self.face_widget.pos()
            out_x = win_rect.width() + 80
            out_y = -self.face_widget.height() - 80

            self._face_out_anim = QPropertyAnimation(self.face_widget, b"pos")
            self._face_out_anim.setDuration(1200)
            self._face_out_anim.setStartValue(cur_pos)
            self._face_out_anim.setEndValue(QPoint(out_x, out_y))
            self._face_out_anim.setEasingCurve(QEasingCurve.InCubic)
            self._face_out_anim.finished.connect(self.face_widget.hide)
            self._face_out_anim.start()

        self._retract_shadows(duration_ms=1800)
        QTimer.singleShot(400, self._reset_view)
        QTimer.singleShot(800, self._ensure_light_mode)
        QTimer.singleShot(2200, self._deactivation_done)

    def _retract_shadows(self, duration_ms):
        for item in list(self.graphics_scene.items()):
            if not isinstance(item, QGraphicsProxyWidget):
                continue
            effect = item.graphicsEffect()
            if isinstance(effect, QGraphicsDropShadowEffect) and _shadow_alive(effect):
                self._animate_shadow_offset(
                    effect,
                    start_x=effect.xOffset(), start_y=effect.yOffset(),
                    end_x=0, end_y=0,
                    duration_ms=duration_ms,
                    easing=QEasingCurve.InOutCubic,
                )

        def _cleanup():
            for item in list(self.graphics_scene.items()):
                if id(item) in self._added_shadows:
                    try:
                        item.setGraphicsEffect(None)
                    except RuntimeError:
                        pass
            self._added_shadows.clear()

        QTimer.singleShot(duration_ms + 200, _cleanup)

    def _reset_view(self):
        self.main_window._animate_view_transform(
            QTransform(), QPointF(0, 0), duration=2000
        )

    def _ensure_light_mode(self):
        if self.main_window._dark_mode:
            self.main_window.toggle_dark_mode()

    def _deactivation_done(self):
        self.is_active = False
        self._activating = False
        print("[ImmersiveMode] Deactivated")

    # ════════════════════════════════════════════════════════════════
    # Widget gravity — cursor attraction toward scene proxies
    # ════════════════════════════════════════════════════════════════

    # Tuning constants
    _GRAVITY_RADIUS = 1000   # px — always attracted
    _GRAVITY_STRENGTH = 1.0  # full pull
    _SNAP_RADIUS = 200       # px — lock on early

    def _apply_widget_gravity(self, gaze_pos: QPointF) -> QPointF:
        """Bend gaze_pos toward the nearest visible proxy widget centre.

        Works in window (pixel) coordinates.  For each proxy we:
          1. Map its scene-space bounding-rect centre to window coords.
          2. Compute distance to gaze_pos.
          3. If within _GRAVITY_RADIUS, pull gaze toward it with a
             smooth inverse-square falloff.
          4. If within _SNAP_RADIUS, snap fully to the widget centre.

        Only the single nearest widget attracts — no tug-of-war.
        """
        best_centre = None
        best_dist = float("inf")

        for item in self.graphics_scene.items():
            if not isinstance(item, QGraphicsProxyWidget):
                continue
            if not item.isVisible():
                continue

            # Scene-space centre of the proxy's bounding rect
            scene_centre = item.mapToScene(item.boundingRect().center())

            # → viewport local → window coords
            view_local = self.graphics_view.mapFromScene(scene_centre)
            win_pt = self.graphics_view.mapToParent(view_local)

            dx = gaze_pos.x() - win_pt.x()
            dy = gaze_pos.y() - win_pt.y()
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < best_dist:
                best_dist = dist
                best_centre = QPointF(win_pt.x(), win_pt.y())

        if best_centre is None or best_dist > self._GRAVITY_RADIUS:
            return gaze_pos  # nothing nearby — no attraction

        if best_dist < self._SNAP_RADIUS:
            # Close enough — snap fully
            return best_centre

        # Smooth attraction: stronger as you get closer
        # t goes from 0 (at GRAVITY_RADIUS) to 1 (at SNAP_RADIUS)
        t = 1.0 - (best_dist - self._SNAP_RADIUS) / (self._GRAVITY_RADIUS - self._SNAP_RADIUS)
        t = t * t  # quadratic ease — gentle at the edge, firm close up
        pull = t * self._GRAVITY_STRENGTH

        attracted_x = gaze_pos.x() + (best_centre.x() - gaze_pos.x()) * pull
        attracted_y = gaze_pos.y() + (best_centre.y() - gaze_pos.y()) * pull

        return QPointF(attracted_x, attracted_y)

    # ════════════════════════════════════════════════════════════════
    # Frame processing — head gaze + hand pinch
    # ════════════════════════════════════════════════════════════════

    def _on_frame(self, frame, face_landmarks, hands_data):
        """Called ~30 fps.  Face → pointer.  Hands → pinch boolean only."""

        # ── Face mesh visualisation ──
        if self.face_widget:
            self.face_widget.set_face_data(frame, face_landmarks)

        # ── Head gaze → cursor ──
        win_w = self.main_window.width()
        win_h = self.main_window.height()

        gaze_pos = self.gaze_tracker.compute_gaze(face_landmarks, win_w, win_h)

        if gaze_pos is not None:
            # Apply widget gravity — bend toward nearby proxies
            gaze_pos = self._apply_widget_gravity(gaze_pos)

            # Post-gravity glide — smooth linear-feeling transition between widgets
            if self._glide_x is None:
                self._glide_x = gaze_pos.x()
                self._glide_y = gaze_pos.y()
            else:
                self._glide_x += self._glide_alpha * (gaze_pos.x() - self._glide_x)
                self._glide_y += self._glide_alpha * (gaze_pos.y() - self._glide_y)
            gaze_pos = QPointF(self._glide_x, self._glide_y)

            self._gaze_pos = gaze_pos
            if self.gaze_cursor:
                self.gaze_cursor.move(
                    int(gaze_pos.x() - self.gaze_cursor.width() / 2),
                    int(gaze_pos.y() - self.gaze_cursor.height() / 2),
                )
                if not self.gaze_cursor.isVisible():
                    self.gaze_cursor.show()
                    self.gaze_cursor.raise_()
        else:
            if self.gaze_cursor and self.gaze_cursor.isVisible():
                self.gaze_cursor.hide()

        # ── Pinch detection (right hand, thumb+middle) — boolean only ──
        changed = self.pinch_detector.update(hands_data)
        now_pinching = self.pinch_detector.is_pinching

        if self.gaze_cursor:
            self.gaze_cursor.set_pinching(now_pinching)

        if changed:
            if now_pinching and not self._is_pinching:
                if self._gaze_pos is not None:
                    self._try_select_at(self._gaze_pos.x(), self._gaze_pos.y())
            elif not now_pinching and self._is_pinching:
                self._release_pinch()

        self._is_pinching = now_pinching

        # ── Pinch hold → drag with gaze ──
        if self._is_pinching and self._pinch_item is not None and self._gaze_pos is not None:
            view_pos = QPoint(int(self._gaze_pos.x()), int(self._gaze_pos.y()))
            view_local = self.graphics_view.mapFromParent(view_pos)
            scene_pos = self.graphics_view.mapToScene(view_local)
            try:
                self._pinch_item.setPos(scene_pos - self._pinch_offset)
            except RuntimeError:
                self._pinch_item = None

    # ════════════════════════════════════════════════════════════════
    # Selection / Pinch
    # ════════════════════════════════════════════════════════════════

    def _try_select_at(self, win_x, win_y):
        """On pinch start, find the topmost proxy under the gaze
        and apply the selected effect."""
        view_local = self.graphics_view.mapFromParent(
            QPoint(int(win_x), int(win_y))
        )
        scene_pos = self.graphics_view.mapToScene(view_local)

        items = self.graphics_scene.items(scene_pos)
        proxy = None
        for item in items:
            p = item
            while p is not None and not isinstance(p, QGraphicsProxyWidget):
                p = p.parentItem()
            if p is not None:
                proxy = p
                break

        if proxy is None:
            return

        self._pinch_item = proxy
        self._pinch_offset = scene_pos - proxy.pos()
        self._selected_proxy = proxy
        self._pre_select_pos = QPointF(proxy.pos())

        # ── Glow effect ──
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(50)
        glow.setColor(QColor(100, 180, 255, 220))
        glow.setOffset(45, 45)
        proxy.setGraphicsEffect(glow)

        # ── Animate: move slightly up-left ──
        cur_pos = proxy.pos()
        target_pos = QPointF(cur_pos.x() - 18, cur_pos.y() - 18)

        self._select_pos_anim = QPropertyAnimation(proxy, b"pos")
        self._select_pos_anim.setDuration(400)
        self._select_pos_anim.setStartValue(cur_pos)
        self._select_pos_anim.setEndValue(target_pos)
        self._select_pos_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._select_pos_anim.start()

        # ── Shadow offset 45→69 ──
        self._animate_shadow_offset(
            glow,
            start_x=45, start_y=45,
            end_x=69, end_y=69,
            duration_ms=500,
            easing=QEasingCurve.OutCubic,
        )

    def _release_pinch(self):
        """Deselect and restore the previously selected proxy."""
        proxy = self._selected_proxy
        if proxy is None:
            return

        try:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(38)
            dark = self.main_window._dark_mode
            shadow.setColor(QColor(255, 255, 255, 160) if dark else QColor(0, 0, 0, 120))
            shadow.setOffset(45, 45)
            proxy.setGraphicsEffect(shadow)
        except RuntimeError:
            pass

        self._selected_proxy = None
        self._pinch_item = None
        self._pre_select_pos = None


# ═══════════════════════════════════════════════════════════════════════
# Integration helper
# ═══════════════════════════════════════════════════════════════════════

def install_immersive_mode(main_window):
    """Attach ImmersiveMode to a RioWindow and wire up Ctrl+I."""
    mode = ImmersiveMode(main_window)
    main_window._immersive_mode = mode
    return mode
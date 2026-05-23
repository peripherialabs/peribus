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
import threading
import queue

from PySide6.QtWidgets import (
    QWidget, QGraphicsDropShadowEffect, QGraphicsProxyWidget,
    QGraphicsItem,
)
from PySide6.QtCore import (
    QObject, Signal, QTimer, QPoint, QPointF, Qt,
    Property, QPropertyAnimation, QEasingCurve, QRectF,
    QVariantAnimation, QThread,
)
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QTransform, QPolygonF,
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

class _MediaPipeWorker(QThread):
    """Background thread that runs MediaPipe inference.

    Why a separate thread:
      face_mesh.process() and hands.process() are CPU-bound and each
      takes 15-40 ms per call. Running them back-to-back at 30 Hz on
      the Qt thread blocks every animation, every paint, every input
      event by up to 80 ms per camera tick — which is exactly when
      the user is moving their head and expecting smooth feedback.

    Design:
      - A bounded queue (size 1) of raw frames. If inference is slow
        we DROP frames rather than buffer them — landmark data from
        300 ms ago is worse than no data, because the cursor would
        lag behind reality.
      - One thread runs both face + hands sequentially (they share
        the same RGB conversion). MediaPipe instances are NOT
        thread-safe so they're created on this thread.
      - Results emitted via a signal, which Qt marshals back to the
        main thread automatically — main thread gets clean callbacks
        with zero locking.
    """
    result_ready = Signal(object, object, object)  # frame, face_landmarks, hands_data

    def __init__(self):
        super().__init__()
        self._frames = queue.Queue(maxsize=1)
        self._running = False
        self._face_model = None
        self._hands_model = None

    def submit_frame(self, frame):
        """Called from camera thread. Drops the oldest pending frame if any."""
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            # Drop the older frame in favour of the newer one
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        self._running = False
        # Unblock the queue
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass

    def run(self):
        if not _HAS_MEDIAPIPE:
            return
        try:
            self._face_model = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands_model = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"[ImmersiveMode] MediaPipe worker init error: {e}")
            return

        self._running = True
        while self._running:
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_landmarks = None
            face_results = self._face_model.process(rgb)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

            hand_results = self._hands_model.process(rgb)
            hands_data = []
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hl, hc in zip(hand_results.multi_hand_landmarks,
                                  hand_results.multi_handedness):
                    hands_data.append((hl, hc.classification[0].label))

            # Emit. Qt auto-marshals to the receiver's thread.
            self.result_ready.emit(frame, face_landmarks, hands_data)

        try:
            self._face_model.close()
            self._hands_model.close()
        except Exception:
            pass


class CameraManager(QObject):
    """Camera capture with MediaPipe inference on a worker thread."""
    frame_ready = Signal(object, object, object)  # frame, face_landmarks, hands_data

    def __init__(self):
        super().__init__()
        self.capture = None
        self.is_running = False
        # Camera read still happens on the Qt thread via a timer —
        # cv2.VideoCapture.read() is fast (a few ms) and keeping it on
        # the main thread avoids cross-thread V4L2/AVFoundation issues.
        # The expensive part (MediaPipe inference) is in _worker.
        self.timer = QTimer()
        self.timer.timeout.connect(self._capture_frame)

        self._worker = _MediaPipeWorker()
        self._worker.result_ready.connect(self._on_worker_result)

    def initialize_mediapipe(self) -> bool:
        # Kept for API compatibility; the worker initializes models
        # lazily on its own thread when start() is called.
        return _HAS_MEDIAPIPE

    def start(self) -> bool:
        if not _HAS_MEDIAPIPE:
            return False
        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened():
            self.is_running = True
            self._worker.start()
            self.timer.start(33)  # ~30 fps camera read; worker drops if behind
            return True
        return False

    def _capture_frame(self):
        if not self.capture or not self.is_running:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        # Hand off to worker. Frame is a numpy array; OpenCV's read()
        # returns a fresh buffer each call so the worker can hold it
        # without race conditions.
        self._worker.submit_frame(frame)

    def _on_worker_result(self, frame, face_landmarks, hands_data):
        # Re-emit on the main thread (Qt has already marshalled here).
        # Existing consumers don't need to change.
        self.frame_ready.emit(frame, face_landmarks, hands_data)

    def stop(self):
        self.is_running = False
        self.timer.stop()
        self._worker.stop()
        if self._worker.isRunning():
            self._worker.wait(2000)
        if self.capture:
            self.capture.release()
            self.capture = None


# ═══════════════════════════════════════════════════════════════════════
# Face Mesh Widget  (lives as a direct child of main_window, NOT scene)
# ═══════════════════════════════════════════════════════════════════════

class FaceMeshWidget(QWidget):
    """Draws the MediaPipe face mesh with isometric 3D rotation.

    Sized generously so the mesh never clips.

    Performance:
      - All 478 landmark projections are done in one NumPy operation
        instead of a Python loop calling math.cos/sin 4 times each.
        Saves ~3000 trig calls per frame.
      - Tesselation index array is built once from MediaPipe's
        FACEMESH_TESSELATION constant and reused.
      - Line drawing is batched via QPainter.drawLines(QLineF[...]),
        which is one C++ call instead of ~2700 (one per edge).
      - Repaint is throttled: the face mesh visual doesn't need 30 Hz
        to look smooth; 20 Hz is fine and frees 33% of the budget.
    """

    # Class-level cache of (start, end) index pairs as a NumPy array.
    # Built lazily once when mediapipe is available.
    _conn_idx = None

    @classmethod
    def _ensure_conn_idx(cls):
        if cls._conn_idx is not None or not _HAS_MEDIAPIPE:
            return
        conns = mp.solutions.face_mesh.FACEMESH_TESSELATION
        cls._conn_idx = np.array(list(conns), dtype=np.int32)  # shape (E, 2)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas_w = 800
        self._canvas_h = 800
        self.setFixedSize(self._canvas_w, self._canvas_h)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        self.current_frame = None
        self.face_landmarks = None
        # Smoothed landmarks stored as a NumPy array (N, 3) — much
        # cheaper to update and reuse than a list of dicts.
        self._smoothed_arr = None  # np.ndarray shape (N, 3) or None

        # Smoothing for the face mesh rendering (visual only, NOT the pointer)
        self.smoothing_factor = 0.45

        # 3D rotation
        self.rotation_y = 0.4
        self.rotation_x = 0.3

        # Isometric projection
        self.scale_x = 1.0
        self.scale_y = 0.8
        self.face_scale = 2.2

        # Repaint throttle: don't repaint more than ~20 Hz even if
        # set_face_data is called at 30 Hz. The face mesh is decorative;
        # smoother is not noticeably better.
        self._last_repaint_t = 0.0
        self._min_repaint_interval = 1.0 / 22.0  # ~22 Hz cap

        self._ensure_conn_idx()

    def _ingest_landmarks(self, new_landmarks):
        """Pull landmark x/y/z into a (N, 3) array, applying EMA smoothing."""
        n = len(new_landmarks.landmark)
        # Build the raw frame array. One Python loop over landmarks is
        # unavoidable since mediapipe returns a protobuf, not a buffer.
        raw = np.empty((n, 3), dtype=np.float32)
        for i, lm in enumerate(new_landmarks.landmark):
            raw[i, 0] = lm.x
            raw[i, 1] = lm.y
            raw[i, 2] = lm.z

        if self._smoothed_arr is None or self._smoothed_arr.shape[0] != n:
            self._smoothed_arr = raw.copy()
        else:
            alpha = 1.0 - self.smoothing_factor
            # In-place vectorized EMA
            self._smoothed_arr *= (1.0 - alpha)
            self._smoothed_arr += alpha * raw

    def set_face_data(self, frame, landmarks):
        self.current_frame = frame
        self.face_landmarks = landmarks
        if landmarks:
            self._ingest_landmarks(landmarks)
        now = time.monotonic()
        if now - self._last_repaint_t >= self._min_repaint_interval:
            self._last_repaint_t = now
            self.update()

    def _project_all(self):
        """Vectorized 3D-to-2D projection of every landmark at once.

        Returns an (N, 2) float32 array of pixel coordinates, already
        scaled by widget width/height.
        """
        a = self._smoothed_arr
        # Mirror X to feel natural
        x = 1.0 - a[:, 0] - 0.5
        y = a[:, 1] - 0.5
        z = a[:, 2]

        cos_y, sin_y = math.cos(self.rotation_y), math.sin(self.rotation_y)
        xr = x * cos_y + z * sin_y
        zr = -x * sin_y + z * cos_y

        cos_x, sin_x = math.cos(self.rotation_x), math.sin(self.rotation_x)
        yr = y * cos_x - zr * sin_x

        ix = (xr * self.scale_x * self.face_scale + 0.5) * self.width()
        iy = (yr * self.scale_y * self.face_scale + 0.5) * self.height()

        return np.column_stack((ix, iy)).astype(np.float32)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._smoothed_arr is None or self.current_frame is None:
            painter.end()
            return

        pts = self._project_all()
        n = pts.shape[0]

        if _HAS_MEDIAPIPE and self._conn_idx is not None:
            # Filter out-of-range edges (defensive — refined mesh has 478)
            cm = self._conn_idx
            valid = (cm[:, 0] < n) & (cm[:, 1] < n)
            edges = cm[valid]

            # Build flat float array of x1,y1,x2,y2,x1,y1,... for drawLines
            starts = pts[edges[:, 0]]
            ends = pts[edges[:, 1]]
            # Interleave into one big array
            lines = np.empty((edges.shape[0], 4), dtype=np.float32)
            lines[:, 0:2] = starts
            lines[:, 2:4] = ends

            from PySide6.QtCore import QLineF
            qlines = [
                QLineF(float(l[0]), float(l[1]), float(l[2]), float(l[3]))
                for l in lines
            ]
            painter.setPen(QPen(QColor(200, 210, 230, 100), 1))
            painter.drawLines(qlines)

        # Draw landmark points as one batched call
        painter.setPen(QPen(QColor(240, 245, 255), 2))
        from PySide6.QtCore import QPointF as _QPF
        qpoints = [_QPF(float(p[0]), float(p[1])) for p in pts]
        painter.drawPoints(qpoints)

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
        # Coalesced QVariantAnimation that drives ALL shadow offsets at
        # once. One timer instead of N — replaces the old per-shadow
        # QTimer pattern that ticked 60×/s × N proxies.
        self._batch_shadow_anim = None

        # Proxy cache for gravity calculation. Invalidated by the
        # explicit method on scene changes; otherwise re-collected
        # every 0.25 s. See _get_visible_proxies for rationale.
        self._proxy_cache = None
        self._proxy_cache_t = 0.0

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
        """Install shadows on all proxies and animate offsets in one batch.

        The earlier implementation created one QTimer per shadow at 16 ms
        cadence. With 20 proxies that's 20 timers firing 60 times/sec,
        each waking Python and invalidating its shadow's offscreen
        buffer separately. The replacement uses one QVariantAnimation
        whose tick callback updates every shadow in a single sweep —
        Qt batches the resulting dirty regions far better.
        """
        # Register shadows with the main window's tracking set so the
        # patched theme/dark animations can manage them too. The window
        # provides _shadowed_items / register_shadowed (see main.py
        # performance patches). Falls back gracefully if absent.
        register = getattr(self.main_window, 'register_shadowed', None)

        records = []
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
                records.append({
                    'effect': effect,
                    'sx': float(effect.xOffset()),
                    'sy': float(effect.yOffset()),
                })
                if register:
                    register(item)

        if not records:
            return

        self._run_batch_shadow_animation(
            records, target_offset, target_offset,
            duration_ms, QEasingCurve.OutInCirc,
        )

    def _run_batch_shadow_animation(self, records, end_x, end_y,
                                     duration_ms, easing):
        """Drive N shadow offsets to the same (end_x, end_y) in one anim."""
        # Stop any in-flight batch first
        if self._batch_shadow_anim is not None:
            try:
                self._batch_shadow_anim.stop()
                self._batch_shadow_anim.deleteLater()
            except RuntimeError:
                pass
            self._batch_shadow_anim = None

        anim = QVariantAnimation(self)
        anim.setDuration(duration_ms)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(easing)

        def tick(t):
            for rec in records:
                effect = rec['effect']
                if not _shadow_alive(effect):
                    continue
                ox = rec['sx'] + (end_x - rec['sx']) * t
                oy = rec['sy'] + (end_y - rec['sy']) * t
                try:
                    effect.setOffset(ox, oy)
                except RuntimeError:
                    pass

        def on_finished():
            # Snap to exact final values to eliminate any rounding drift
            for rec in records:
                effect = rec['effect']
                if _shadow_alive(effect):
                    try:
                        effect.setOffset(end_x, end_y)
                    except RuntimeError:
                        pass
            self._batch_shadow_anim = None

        anim.valueChanged.connect(tick)
        anim.finished.connect(on_finished)
        self._batch_shadow_anim = anim
        anim.start()

    def _animate_shadow_offset(self, shadow, start_x, start_y,
                                end_x, end_y, duration_ms, easing):
        """Single-shadow animation, used by selection/release.

        Kept as its own method (rather than routed through the batched
        path) because pinch select/release operates on exactly one
        shadow at a time, and the batched path's bookkeeping would be
        wasted overhead. Still uses QVariantAnimation rather than a
        raw QTimer so Qt can manage the lifecycle.
        """
        anim = QVariantAnimation(self)
        anim.setDuration(duration_ms)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(easing)

        def tick(t):
            if not _shadow_alive(shadow):
                anim.stop()
                return
            try:
                shadow.setOffset(
                    start_x + (end_x - start_x) * t,
                    start_y + (end_y - start_y) * t,
                )
            except RuntimeError:
                anim.stop()

        def on_finished():
            if _shadow_alive(shadow):
                try:
                    shadow.setOffset(end_x, end_y)
                except RuntimeError:
                    pass

        anim.valueChanged.connect(tick)
        anim.finished.connect(on_finished)
        # Track on self so it isn't GC'd mid-flight. Old timers list
        # is reused for this purpose; entries are pruned on finish.
        self._shadow_timers.append(anim)
        anim.finished.connect(lambda: self._shadow_timers.remove(anim)
                              if anim in self._shadow_timers else None)
        anim.start()

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
        """Animate all immersive shadows back to (0, 0) in one batch."""
        unregister = getattr(self.main_window, 'unregister_shadowed', None)

        records = []
        for item in list(self.graphics_scene.items()):
            if not isinstance(item, QGraphicsProxyWidget):
                continue
            effect = item.graphicsEffect()
            if isinstance(effect, QGraphicsDropShadowEffect) and _shadow_alive(effect):
                records.append({
                    'effect': effect,
                    'sx': float(effect.xOffset()),
                    'sy': float(effect.yOffset()),
                })

        if records:
            self._run_batch_shadow_animation(
                records, 0, 0, duration_ms, QEasingCurve.InOutCubic,
            )

        def _cleanup():
            for item in list(self.graphics_scene.items()):
                if id(item) in self._added_shadows:
                    try:
                        item.setGraphicsEffect(None)
                        if unregister:
                            unregister(item)
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

    def _invalidate_proxy_cache(self):
        """Force the next gravity call to re-scan the scene."""
        self._proxy_cache_t = 0.0
        self._proxy_cache = None

    def _get_visible_proxies(self):
        """Return (proxy, scene_centre_QPointF) pairs, cached.

        The scene scan is the hot path here — it happens at the
        MediaPipe frame rate (30 Hz) and walks every item in the
        scene. Caching for 250 ms removes that scan from 24 of every
        30 frames while still picking up newly-added widgets within
        a noticeable but acceptable delay.

        Cache is invalidated explicitly when we know the scene
        changed (e.g. on pinch release of a moved item).
        """
        now = time.monotonic()
        if (self._proxy_cache is not None and
                now - self._proxy_cache_t < 0.25):
            return self._proxy_cache

        result = []
        for item in self.graphics_scene.items():
            if not isinstance(item, QGraphicsProxyWidget):
                continue
            if not item.isVisible():
                continue
            scene_centre = item.mapToScene(item.boundingRect().center())
            result.append((item, scene_centre))

        self._proxy_cache = result
        self._proxy_cache_t = now
        return result

    def _apply_widget_gravity(self, gaze_pos: QPointF) -> QPointF:
        """Bend gaze_pos toward the nearest visible proxy widget centre.

        Works in window (pixel) coordinates.  For each proxy we:
          1. Map its scene-space bounding-rect centre to window coords.
          2. Compute distance to gaze_pos.
          3. If within _GRAVITY_RADIUS, pull gaze toward it with a
             smooth inverse-square falloff.
          4. If within _SNAP_RADIUS, snap fully to the widget centre.

        Only the single nearest widget attracts — no tug-of-war.

        The scene→window mapping IS done per-frame even with caching,
        because the view transform can change continuously (orbit, pan,
        zoom). Only the proxy list itself is cached.
        """
        best_centre = None
        best_dist_sq = float("inf")
        gx, gy = gaze_pos.x(), gaze_pos.y()

        for item, scene_centre in self._get_visible_proxies():
            # Map scene→viewport→window. We do this in two C++ calls
            # rather than the previous three-step chain.
            view_local = self.graphics_view.mapFromScene(scene_centre)
            win_pt = self.graphics_view.mapToParent(view_local)
            dx = gx - win_pt.x()
            dy = gy - win_pt.y()
            dist_sq = dx * dx + dy * dy  # skip sqrt until we need it

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_centre = QPointF(win_pt.x(), win_pt.y())

        if best_centre is None:
            return gaze_pos

        best_dist = math.sqrt(best_dist_sq)
        if best_dist > self._GRAVITY_RADIUS:
            return gaze_pos  # nothing nearby — no attraction

        if best_dist < self._SNAP_RADIUS:
            # Close enough — snap fully
            return best_centre

        # Smooth attraction: stronger as you get closer
        # t goes from 0 (at GRAVITY_RADIUS) to 1 (at SNAP_RADIUS)
        t = 1.0 - (best_dist - self._SNAP_RADIUS) / (self._GRAVITY_RADIUS - self._SNAP_RADIUS)
        t = t * t  # quadratic ease — gentle at the edge, firm close up
        pull = t * self._GRAVITY_STRENGTH

        attracted_x = gx + (best_centre.x() - gx) * pull
        attracted_y = gy + (best_centre.y() - gy) * pull

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
            # Re-register so it tracks with main window's shadow set
            register = getattr(self.main_window, 'register_shadowed', None)
            if register:
                register(proxy)
        except RuntimeError:
            pass

        # The item likely moved during drag — invalidate gravity cache
        # so the next frame re-collects fresh scene-space centres.
        self._invalidate_proxy_cache()

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
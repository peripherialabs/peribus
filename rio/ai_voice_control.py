"""
AI Voice Control Widget  —  rio/ai_voice_control.py
====================================================
Compact eye-pair widget for controlling AI voice agents.

All agents share a single 'av' directory under the llmfs mount.
Switching backends destroys the current agent and re-creates it
under the same path, so other instances targeting $av always work.

Integration:
  Imported by rio/main.py → RioWindow._init_voice_control()
  Right-click menu → "Show AI Voice" / "Hide AI Voice"

Interaction:
  - Scroll up/down to switch between agents
  - Click to open eyes  → echo start > /n/llm/av/ctl
  - Click to close eyes → echo stop  > /n/llm/av/ctl
  - Long-press (≥600ms) to toggle Master mode:
      • Uses ./systems/audiovisual_master.md system prompt
      • Sets male voices: Puck (gemini), Kai (grok), ash (gpt)
      • Label shows $ prefix when active (e.g. $X, $Y, $Z)
  - Agent name fades in below the eyes on switch

Routes:
  Code→scene routing uses the universal mux pattern:
    echo '/n/mux/llm/av/CODE -> /n/mux/ws/scene/parse' > /n/mux/ws/routes
  The mux handles cross-backend bridging transparently.

Visual differences per agent:
  gemini (Y)  — blue irises, normal proportions, neutral brows
  grok   (X)  — amber irises, curved-up cat-eye shape, raised arched brows
  gpt    (Z)  — green irises, narrow almond/hooded shape, flat sleek brows
"""

import math
import random
import json
import os
import subprocess
import threading

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QPainterPath, QFont, QColor,
    QRadialGradient, QCursor, QTransform,
)
from PySide6.QtCore import (
    Qt, QPointF, QTimer, Signal, QRectF, QElapsedTimer,
)

# Mount paths are instance attributes on AIVoiceControlWidget,
# set via __init__(llmfs_mount=..., rio_mount=...) to support riomux.
# Auto-detects mount point if not provided: probes /n/mux/llm then /n/llm.
# Routes are written to the mux: echo 'src -> dst' > /n/mux/workspace_name/routes

# ── Agent personality profiles ───────────────────────────────────────────────

AGENT_PROFILES = {
    "gemini": {
        "display_name": "Y",
        "agent_name": "av",                              # shared dirname under llmfs root
        "init_cmd": "av av",                              # written to /n/llm/ctl
        "default_voice": "Aoede",
        "config_extra": {"google_search": True},        # extra config keys
        "iris_color": QColor(80, 140, 220),
        "iris_rim_color": QColor(40, 80, 160),
        "eye_width_scale": 1.0,
        "eye_height_scale": 1.0,
        "outer_corner_lift": 0,
        "inner_corner_drop": 0,
        "upper_lid_curve_boost": 0,
        "eyebrow_y_offset": 0,
        "eyebrow_arch": 0,
        "eyebrow_thickness": 2.5,
        "lid_thickness": 3.0,
        "lower_lid_thickness": 0.8,
        "eyelash_density": 80,
        "eyelash_length": 1.0,
        "right_lash_angle_offset": 0,
        "label_color": QColor(100, 160, 255),
    },
    "grok": {
        "display_name": "X",
        "agent_name": "av",                              # shared dirname under llmfs root
        "init_cmd": "grok av",                            # written to /n/llm/ctl
        "default_voice": "Ara",
        "config_extra": {"tool_choice": "required", "temperature": 0.8},
        "iris_color": QColor(160, 115, 55),
        "iris_rim_color": QColor(100, 65, 25),
        "eye_width_scale": 1.06,
        "eye_height_scale": 0.95,
        "outer_corner_lift": 14,        # positive = lift outer corners up
        "inner_corner_drop": -4,        # negative = drop inner corners down
        "upper_lid_curve_boost": 8,     # extra upward pull on upper lid apex
        "eyebrow_y_offset": -6,
        "eyebrow_arch": 5,
        "eyebrow_thickness": 3.2,
        "lid_thickness": 3.5,
        "lower_lid_thickness": 1.0,
        "eyelash_density": 60,
        "eyelash_length": 0.8,
        "right_lash_angle_offset": -10,
        "label_color": QColor(255, 160, 80),
    },
    "gpt": {
        "display_name": "Z",
        "agent_name": "av",                              # shared dirname under llmfs root
        "init_cmd": "openai av",                          # written to /n/llm/ctl
        "default_voice": "marin",
        "config_extra": {"tool_choice": "auto"},
        "iris_color": QColor(60, 180, 90),
        "iris_rim_color": QColor(25, 110, 50),
        "eye_width_scale": 1.10,
        "eye_height_scale": 0.95,
        "outer_corner_lift": 8,         # lifted outer corners — almond/angular
        "inner_corner_drop": -6,        # dropped inner — narrow angular look
        "upper_lid_curve_boost": -2,    # flatter upper lid — hooded appearance
        "eyebrow_y_offset": -2,
        "eyebrow_arch": 0,              # flat, minimal brow arch — sleek
        "eyebrow_thickness": 3.0,
        "lid_thickness": 3.2,
        "lower_lid_thickness": 1.2,
        "eyelash_density": 55,
        "eyelash_length": 0.7,
        "right_lash_angle_offset": -10,
        "label_color": QColor(80, 210, 120),
    },
}

AGENT_ORDER = ["grok", "gemini", "gpt"]

# ── Long-press configuration ─────────────────────────────────────────────────
LONG_PRESS_THRESHOLD_MS = 600   # hold ≥600ms → long press

# Male voice overrides + alternate system prompt for long-press activation
MASTER_PROFILES = {
    "gemini": {
        "voice": "Puck",                       # male Gemini voice
        "system_prompt": "./systems/audiovisual_master.md",
    },
    "grok": {
        "voice": "Kai",                        # male Grok voice
        "system_prompt": "./systems/audiovisual_master.md",
    },
    "gpt": {
        "voice": "ash",                        # male OpenAI voice
        "system_prompt": "./systems/audiovisual_master.md",
    },
}


# ── Main widget ──────────────────────────────────────────────────────────────

class AIVoiceControlWidget(QWidget):
    """Compact eye-pair widget that controls AI voice agents."""

    recording_state_changed = Signal(bool)
    agent_changed = Signal(str)

    @staticmethod
    def _detect_mount(subdir, marker_file, exclude=None):
        """Auto-detect a 9P mount point by probing /n/mux then /n."""
        bases = ["/n/mux", "/n"]
        if subdir:
            for base in bases:
                candidate = os.path.join(base, subdir)
                if os.path.isfile(os.path.join(candidate, marker_file)):
                    return candidate
        else:
            for base in bases:
                if not os.path.isdir(base):
                    continue
                try:
                    for name in sorted(os.listdir(base)):
                        if exclude and name == exclude:
                            continue
                        candidate = os.path.join(base, name)
                        if os.path.isdir(candidate) and os.path.exists(
                            os.path.join(candidate, marker_file)
                        ):
                            return candidate
                except OSError:
                    continue
        if subdir:
            return os.path.join("/n/mux", subdir)
        return "/n/mux/default"

    def __init__(self, parent=None, scale_factor=0.40,
                 llmfs_mount=None, rio_mount=None):
        super().__init__(parent)
        self.scale_factor = scale_factor

        # Auto-detect mount points if not explicitly provided
        if llmfs_mount is None:
            llmfs_mount = self._detect_mount("llm", "ctl")
        if rio_mount is None:
            rio_mount = self._detect_mount(None, "scene", exclude="llm")

        # Mux-aware mount paths
        self.llmfs_mount = llmfs_mount
        self.rio_mount = rio_mount
        self.scene_parse = os.path.join(rio_mount, "scene", "parse")

        self._base_w, self._base_h = 700, 480
        sw = int(self._base_w * scale_factor)
        sh = int(self._base_h * scale_factor)
        self.setFixedSize(sw, sh)

        self.setStyleSheet("background-color: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.Widget)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)

        # ── Agent state ──────────────────────────────────────────────────
        self._agent_index = 0
        self._agent_key = AGENT_ORDER[0]
        self._profile = AGENT_PROFILES[self._agent_key]
        self._agents_created = set()  # tracks which agents have been init'd via /n/llm/ctl
        self._active_backend = None   # which agent_key currently owns the 'av' dir
        self._active_route = None     # (source, destination) tuple for current code→scene route
        self._master_mode = False     # True when long-press activated (male voice / master prompt)

        # ── Label fade animation ─────────────────────────────────────────
        self._label_opacity = 0.0
        self._label_target_opacity = 0.0
        self._label_timer = QTimer(self)
        self._label_timer.timeout.connect(self._animate_label)
        self._label_fade_out_timer = QTimer(self)
        self._label_fade_out_timer.setSingleShot(True)
        self._label_fade_out_timer.timeout.connect(self._start_label_fade_out)

        # ── Eye state ────────────────────────────────────────────────────
        self.eyes_are_closed = True
        self.is_recording = False
        self.is_animating = False
        self.blink_progress = 1.0
        self.blink_direction = 1
        self.eye_open_level = 1.0

        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._update_toggle_animation)

        # ── Flicker ──────────────────────────────────────────────────────
        self.is_flickering = False
        self._flicker_queued = False
        self._flicker_timer = QTimer(self)
        self._flicker_timer.timeout.connect(self._update_flicker)
        self._flicker_dur_timer = QTimer(self)
        self._flicker_dur_timer.setSingleShot(True)
        self._flicker_dur_timer.timeout.connect(self._stop_flicker)

        # ── Wink ─────────────────────────────────────────────────────────
        self.is_winking = False
        self.wink_progress = 0.0
        self.wink_direction = 1
        self.wink_eye = "left"
        self._wink_timer = QTimer(self)
        self._wink_timer.timeout.connect(self._update_wink)
        self._wink_hold = QTimer(self)
        self._wink_hold.setSingleShot(True)
        self._wink_hold.timeout.connect(self._wink_return)

        # ── Mouse tracking ───────────────────────────────────────────────
        self.mouse_tracking_enabled = True
        self.mouse_pos = QPointF(350, 225)
        self.left_eye_center = QPointF(200, 225)
        self.right_eye_center = QPointF(500, 225)
        self.eyebrow_offset = 0.0
        self._last_iris_x = 0.0
        self._last_iris_y = 0.0

        # ── Mouse poll timer (replaces app-wide event filter) ────────
        self._mouse_poll_timer = QTimer(self)
        self._mouse_poll_timer.timeout.connect(self._poll_mouse)

        # ── Long-press detection ─────────────────────────────────────
        self._press_elapsed = QElapsedTimer()
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.timeout.connect(self._on_long_press_detected)
        self._long_press_fired = False

        # ── Intro draw-on animation ──────────────────────────────────
        self._intro_playing = False
        self._intro_progress = 1.0       # 1.0 = fully drawn, 0.0 = invisible
        self._intro_timer = QTimer(self)
        self._intro_timer.timeout.connect(self._update_intro)
        self._INTRO_DURATION_MS = 900    # total animation time

        # ── Random eye motion (saccades + drift) ─────────────────────
        self._auto_gaze_enabled = True
        self._gaze_offset_x = 0.0       # current autonomous gaze offset
        self._gaze_offset_y = 0.0
        self._gaze_target_x = 0.0       # where gaze is drifting toward
        self._gaze_target_y = 0.0
        self._gaze_speed = 0.04          # interpolation speed per tick
        self._gaze_timer = QTimer(self)
        self._gaze_timer.timeout.connect(self._update_gaze)
        self._gaze_saccade_timer = QTimer(self)
        self._gaze_saccade_timer.setSingleShot(True)
        self._gaze_saccade_timer.timeout.connect(self._new_gaze_target)
        self._gaze_mouse_active = False  # True when mouse is actively nearby

        # ── Random blink ──────────────────────────────────────────────
        self._auto_blink_timer = QTimer(self)
        self._auto_blink_timer.setSingleShot(True)
        self._auto_blink_timer.timeout.connect(self._trigger_auto_blink)
        self._auto_blink_anim_timer = QTimer(self)
        self._auto_blink_anim_timer.timeout.connect(self._update_auto_blink)
        self._auto_blink_progress = 0.0
        self._auto_blink_phase = 0  # 0=idle, 1=closing, 2=opening
        self._auto_blink_speed = 0.18

        # ── Tilt/rotate + shadow (proxy circular motion) ──────────────
        self._tilt_enabled = True
        self._tilt_angle = 0.0           # current rotation angle (degrees)
        self._tilt_offset_x = 0.0        # current translation offset
        self._tilt_offset_y = 0.0
        self._tilt_phase = 0.0           # circular motion phase (radians)
        self._tilt_phase_speed = 0.008   # phase advance per tick
        self._tilt_radius = 0.0          # current orbit distance (0-45 px)
        self._tilt_target_radius = 12.0  # target orbit distance
        self._tilt_radius_phase = 0.0    # secondary phase for varying radius
        self._tilt_radius_speed = 0.003  # how fast radius varies
        self._tilt_max_angle = 3.5       # max rotation degrees
        self._tilt_timer = QTimer(self)
        self._tilt_timer.timeout.connect(self._update_tilt)
        self._tilt_elapsed = QElapsedTimer()

        # ── Drastic perspective tilt (QTransform on proxy) ────────────
        #    Affine only (scale + shear around the widget center via
        #    transformOriginPoint).  No projective terms — those make
        #    the rasteriser clip or lose the widget.
        self._proxy = None                # set by attach_proxy_shadow()
        self._drastic_active = False
        self._drastic_progress = 0.0
        self._drastic_phase = 0           # 0=idle 1=attack 2=hold 3=release
        # We animate 4 independent parameters and build QTransform each tick
        self._drastic_sx = 1.0            # current scale-x
        self._drastic_sy = 1.0            # current scale-y
        self._drastic_shx = 0.0           # current shear-x
        self._drastic_shy = 0.0           # current shear-y
        self._drastic_target_sx = 1.0
        self._drastic_target_sy = 1.0
        self._drastic_target_shx = 0.0
        self._drastic_target_shy = 0.0
        self._drastic_start_sx = 1.0
        self._drastic_start_sy = 1.0
        self._drastic_start_shx = 0.0
        self._drastic_start_shy = 0.0
        self._drastic_hold_timer = QTimer(self)
        self._drastic_hold_timer.setSingleShot(True)
        self._drastic_hold_timer.timeout.connect(self._drastic_begin_release)
        self._drastic_trigger_timer = QTimer(self)
        self._drastic_trigger_timer.setSingleShot(True)
        self._drastic_trigger_timer.timeout.connect(self._drastic_trigger)

        # ── Pre-generated data ───────────────────────────────────────────
        self._gen_eyelash_data()
        self._gen_iris_data()

    # ── Geometry lock ────────────────────────────────────────────────────
    def setGeometry(self, *a):
        if hasattr(self, '_allow_geometry_change'):
            super().setGeometry(*a)

    def move(self, *a):
        if hasattr(self, '_allow_move'):
            super().move(*a)

    # ── Agent switching ──────────────────────────────────────────────────
    def _switch_agent(self, direction):
        was_active = self.is_recording
        if was_active:
            self._send_ctl("stop")
            self._teardown_code_route()
            self._mouse_poll_timer.stop()
            self._stop_autonomous_animations()
            self.eyes_are_closed = True
            self.is_recording = False
            self.blink_progress = 1.0
            self.recording_state_changed.emit(False)

        old = self._agent_index
        self._agent_index = (self._agent_index + direction) % len(AGENT_ORDER)
        if self._agent_index == old:
            return
        old_key = self._agent_key
        self._agent_key = AGENT_ORDER[self._agent_index]
        self._profile = AGENT_PROFILES[self._agent_key]

        # Destroy the old agent to free the shared 'av' slot for the new backend.
        # All agents share the same dir so the old one must be fully torn down.
        if self._active_backend and self._active_backend != self._agent_key:
            self._destroy_agent()

        self._gen_eyelash_data()

        self._label_opacity = 0.0
        self._label_target_opacity = 1.0
        self._label_timer.start(16)
        self._label_fade_out_timer.stop()
        self._label_fade_out_timer.start(1800)

        self.agent_changed.emit(self._agent_key)
        self.update()

    def _start_label_fade_out(self):
        self._label_target_opacity = 0.0
        self._label_timer.start(16)

    def _animate_label(self):
        speed = 0.06
        if self._label_target_opacity > self._label_opacity:
            self._label_opacity = min(self._label_target_opacity,
                                      self._label_opacity + speed)
        else:
            self._label_opacity = max(self._label_target_opacity,
                                      self._label_opacity - speed * 0.6)
        if abs(self._label_opacity - self._label_target_opacity) < 0.01:
            self._label_opacity = self._label_target_opacity
            self._label_timer.stop()
        self.update()

    # ── CTL communication ────────────────────────────────────────────────

    def _agent_dir(self):
        """Return full path to the current agent's directory."""
        return os.path.join(self.llmfs_mount,
                            self._profile["agent_name"])

    def _agent_ctl(self):
        """Return full path to the current agent's ctl file."""
        return os.path.join(self._agent_dir(), "ctl")

    def _ensure_agent_created(self):
        """Full agent setup — creates the shared 'av' agent for the current backend.

        All agents share the same 'av' directory. If a different backend
        currently owns it, we destroy the old one first.

        0. Destroy existing agent if owned by a different backend
        1. Create agent via /n/llm/ctl  (if dir doesn't exist)
        2. Write config  (voice, functions, tool_choice, etc.)
        3. Write system prompt from ./systems/audiovisual.md
        """
        key = self._agent_key
        if key in self._agents_created:
            return True

        p = self._profile
        agent_name = p["agent_name"]   # always "av"
        agent_dir = self._agent_dir()
        llmfs_ctl = os.path.join(self.llmfs_mount, "ctl")

        # Step 0: If a different backend owns the av slot, tear it down first
        if self._active_backend is not None and self._active_backend != key:
            self._destroy_agent()

        # Step 1: Create the agent via ctl (if not already present)
        if not os.path.isdir(agent_dir):
            try:
                with open(llmfs_ctl, 'w') as f:
                    f.write(p["init_cmd"] + "\n")
                print(f"[AIVoice] Agent '{agent_name}' created "
                      f"via: {p['init_cmd']} (backend={key})")
            except Exception as e:
                print(f"[AIVoice] Failed to create agent: {e}")
                return False
        else:
            print(f"[AIVoice] Agent '{agent_name}' already exists")

        # Step 1: Write config (voice + function tool)
        try:
            # Use master-mode voice override if active
            voice = p["default_voice"]
            if self._master_mode and key in MASTER_PROFILES:
                voice = MASTER_PROFILES[key]["voice"]

            config = {
                "voice": voice,
                "functions": [
                    {
                        "name": "handle_simple_programming",
                        "description": (
                            "Execute ANY code or programming task. "
                            "Always call this for: buttons, scripts, "
                            "UI, calculations, or any coding request."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description":
                                        "Raw Python code to execute"
                                }
                            },
                            "required": ["code"]
                        }
                    }
                ],
            }
            if self._master_mode:
                config = {
                    "voice": voice,
                    "functions": [
                        {
                            "name": "handle_simple_programming",
                            "description": (
                                "Execute bash code or programming task. "
                                "Always use this function to control the shell."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "code": {
                                        "type": "string",
                                        "description":
                                            "Raw bash code to execute"
                                    }
                                },
                                "required": ["code"]
                            }
                        }
                    ],
                }

            # Merge profile-specific extra config keys
            config.update(p.get("config_extra", {}))

            config_path = os.path.join(agent_dir, "config")
            with open(config_path, 'w') as f:
                f.write(json.dumps(config))
            print(f"[AIVoice] Config written: voice={voice}"
                  f"{' (master)' if self._master_mode else ''}")
        except Exception as e:
            print(f"[AIVoice] Failed to write config: {e}")
            return False

        # Step 2: Write system prompt
        try:
            system_path = os.path.join(agent_dir, "system")
            # Use master system prompt if in master mode
            if self._master_mode and key in MASTER_PROFILES:
                prompt_file = MASTER_PROFILES[key]["system_prompt"]
            else:
                prompt_file = "./systems/audiovisual.md"
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as src:
                    prompt = src.read()
                with open(system_path, 'w') as dst:
                    dst.write(prompt)
                print(f"[AIVoice] System prompt configured from {prompt_file}")
            else:
                print(f"[AIVoice] {prompt_file} not found, skipping")
        except Exception as e:
            print(f"[AIVoice] Failed to set system prompt: {e}")

        self._agents_created.add(key)
        self._active_backend = key
        return True

    def _send_ctl(self, command):
        """Write a command (start/stop) to the agent's ctl file."""
        if not self._ensure_agent_created():
            return
        ctl_path = self._agent_ctl()
        try:
            with open(ctl_path, 'w') as f:
                f.write(command + "\n")
            print(f"[AIVoice] {command} → {ctl_path}")
        except Exception as e:
            print(f"[AIVoice] ctl write failed: {e}")

    def _destroy_agent(self):
        """Fully tear down the current agent occupying the shared 'av' slot.

        Sequence:
          1. echo stop > /n/llm/av/ctl        (graceful shutdown)
          2. echo rm av > /n/llm/ctl           (delete the directory)
          3. Clear _agents_created + _active_backend so next open re-creates

        This frees the 'av' slot for a different backend (gemini/grok/openai).
        Other instances targeting /n/llm/av/ will see the agent disappear and
        reappear under the new backend when the eyes next open.
        """
        agent_dir = os.path.join(self.llmfs_mount, "av")
        llmfs_ctl = os.path.join(self.llmfs_mount, "ctl")

        # Step 1: Stop the running agent (best-effort)
        ctl_path = os.path.join(agent_dir, "ctl")
        if os.path.exists(ctl_path):
            try:
                with open(ctl_path, 'w') as f:
                    f.write("stop\n")
                print(f"[AIVoice] stop → {ctl_path}")
            except Exception as e:
                print(f"[AIVoice] stop failed (non-fatal): {e}")

        # Step 2: Delete the agent directory via the llmfs master ctl
        try:
            with open(llmfs_ctl, 'w') as f:
                f.write("rm av\n")
            print(f"[AIVoice] Destroyed agent 'av' via: rm av")
        except Exception as e:
            print(f"[AIVoice] Failed to destroy agent: {e}")

        # Step 3: Reset tracking state
        self._agents_created.clear()
        self._active_backend = None

    # ── Route management (via /n/mux/workspace_name/routes) ─────────────
    #
    # Universal mux route pattern:
    #   echo '/path/to/output -> /path/to/input' > /n/mux/workspace_name/routes
    #
    # The mux handles cross-backend bridging (llmfs ↔ rio) transparently.
    # No need for custom bridge subprocesses or in-process RoutesManagers.

    def _write_mux_route(self, source: str, destination: str):
        """
        Declare a route by writing to the mux workspace routes file.

        Writes '{source} -> {destination}' to /n/mux/{workspace_name}/routes,
        which is the universal way to create routes across mux backends.

        Runs in a background thread because the mux backend may not be
        connected yet when the eyes first open — the echo will block on
        FUSE until the backend is ready, which is fine as long as we
        don't freeze the UI.
        """
        route_file = os.path.join(self.rio_mount, "routes")
        route_decl = f"{source} -> {destination}"
        self._active_route = (source, destination)

        def _write():
            try:
                subprocess.run(
                    ["bash", "-c", f"echo '{route_decl}' > {route_file}"],
                    check=True, timeout=30,
                )
                print(f"[AIVoice] Route declared: {route_decl} > {route_file}")
            except Exception as e:
                print(f"[AIVoice] Failed to write route to {route_file}: {e}")

        threading.Thread(target=_write, daemon=True).start()

    def _remove_mux_route(self, source: str):
        """
        Remove a route by writing the removal to the mux workspace routes file.

        Writes '{source} ->' (empty destination) to signal teardown.
        """
        if not self._active_route or self._active_route[0] != source:
            return

        route_file = os.path.join(self.rio_mount, "routes")
        route_decl = f"{source} ->"
        self._active_route = None

        def _write():
            try:
                subprocess.run(
                    ["bash", "-c", f"echo '{route_decl}' > {route_file}"],
                    check=True, timeout=30,
                )
                print(f"[AIVoice] Route removed: {source}")
            except Exception as e:
                print(f"[AIVoice] Failed to remove route from {route_file}: {e}")

        threading.Thread(target=_write, daemon=True).start()

    def _ensure_code_route(self):
        """Set up $av/CODE → {rio_mount}/scene/parse route for the current agent."""
        agent_name = self._profile["agent_name"]
        code_source = os.path.join(self.llmfs_mount, agent_name, "CODE")
        if not self._master_mode:
            self._write_mux_route(code_source, self.scene_parse)

    def _teardown_code_route(self):
        """Remove the CODE→scene route for the current agent."""
        if self._active_route:
            self._remove_mux_route(self._active_route[0])

    # ── Mouse / scroll events ────────────────────────────────────────────
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._switch_agent(-1)
        elif delta < 0:
            self._switch_agent(1)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.is_animating \
                and not self.is_flickering:
            self._press_elapsed.start()
            self._long_press_fired = False
            self._long_press_timer.start(LONG_PRESS_THRESHOLD_MS)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._long_press_timer.stop()
            if self._long_press_fired:
                # Long press already handled — don't do a normal toggle
                return
            if not self.is_animating and not self.is_flickering:
                self._toggle_eyes()

    def _on_long_press_detected(self):
        """Fired when the user holds the click for ≥LONG_PRESS_THRESHOLD_MS."""
        self._long_press_fired = True
        # Toggle master mode (male voice + audiovisual_master.md)
        self._master_mode = not self._master_mode

        # If eyes are currently open, stop the active agent first
        if self.is_recording:
            self._send_ctl("stop")
            self._teardown_code_route()
            self._mouse_poll_timer.stop()
            self._stop_autonomous_animations()
            self.eyes_are_closed = True
            self.is_recording = False
            self.blink_progress = 1.0
            self.recording_state_changed.emit(False)

        # Destroy and recreate: master mode changes config (voice, system prompt)
        # so the agent must be torn down and rebuilt
        self._destroy_agent()

        mode_label = "MASTER" if self._master_mode else "DEFAULT"
        print(f"[AIVoice] Long-press → {mode_label} mode "
              f"(male voices + audiovisual_master.md)"
              if self._master_mode else
              f"[AIVoice] Long-press → {mode_label} mode "
              f"(default voices + audiovisual.md)")

        # Show label feedback
        self._label_opacity = 0.0
        self._label_target_opacity = 1.0
        self._label_timer.start(16)
        self._label_fade_out_timer.stop()
        self._label_fade_out_timer.start(2200)

        self.update()

    # ── Eye toggle ───────────────────────────────────────────────────────
    def _toggle_eyes(self):
        self.is_animating = True
        if self.eyes_are_closed:
            self.blink_direction = -1
            self.eyes_are_closed = False
            self.is_recording = True
            self.recording_state_changed.emit(True)
            self._ensure_code_route()
            self._send_ctl("start")
            self._mouse_poll_timer.start(33)  # ~30 fps cursor polling
            self._start_autonomous_animations()
        else:
            self.blink_direction = 1
            self.eyes_are_closed = True
            self.is_recording = False
            self.recording_state_changed.emit(False)
            self._send_ctl("stop")
            self._teardown_code_route()
            self._mouse_poll_timer.stop()
            self._stop_autonomous_animations()
        self.blink_timer.stop()
        try:
            self.blink_timer.timeout.disconnect()
        except RuntimeError:
            pass
        self.blink_timer.timeout.connect(self._update_toggle_animation)
        self.blink_timer.start(16)

    def _update_toggle_animation(self):
        speed = 0.08
        self.blink_progress += speed * self.blink_direction
        if self.blink_direction == 1 and self.blink_progress >= 1.0:
            self.blink_progress = 1.0
            self.blink_timer.stop()
            self.is_animating = False
        elif self.blink_direction == -1 and self.blink_progress <= 0.0:
            self.blink_progress = 0.0
            self.blink_timer.stop()
            self.is_animating = False
        self.update()

    # ── Flicker ──────────────────────────────────────────────────────────
    def trigger_flicker(self):
        if self.is_flickering or self._flicker_queued or self.eyes_are_closed:
            return
        self._flicker_queued = True
        self._start_flicker_blink()

    def _start_flicker_blink(self):
        if self.is_flickering:
            return
        self.blink_direction = 1
        self.blink_timer.stop()
        try:
            self.blink_timer.timeout.disconnect()
        except RuntimeError:
            pass
        self.blink_timer.timeout.connect(self._flicker_blink_step)
        self.blink_timer.start(16)

    def _flicker_blink_step(self):
        self.blink_progress += 0.15 * self.blink_direction
        if self.blink_progress >= 1.0:
            self.blink_progress = 1.0
            if self._flicker_queued:
                self._flicker_queued = False
                self.blink_timer.stop()
                try:
                    self.blink_timer.timeout.disconnect()
                except RuntimeError:
                    pass
                self._commence_flicker()
                return
            self.blink_direction = -1
        elif self.blink_progress <= 0.0:
            self.blink_progress = 0.0
            self.blink_timer.stop()
            try:
                self.blink_timer.timeout.disconnect()
            except RuntimeError:
                pass
            self.blink_timer.timeout.connect(self._update_toggle_animation)
        self.update()

    def _commence_flicker(self):
        self.is_flickering = True
        self.blink_direction = -1
        self._flicker_dur_timer.start(2000)
        self._flicker_timer.start(35)

    def _update_flicker(self):
        self.blink_progress += 1.0 * self.blink_direction
        if self.blink_progress >= 1.0:
            self.blink_progress = 1.0
            self.blink_direction = -1
        elif self.blink_progress <= 0.6:
            self.blink_progress = 0.6
            self.blink_direction = 1
        self.update()

    def _stop_flicker(self):
        self._flicker_timer.stop()
        self.is_flickering = False
        self.blink_direction = -1
        self.blink_timer.stop()
        try:
            self.blink_timer.timeout.disconnect()
        except RuntimeError:
            pass
        self.blink_timer.timeout.connect(self._flicker_blink_step)
        self.blink_timer.start(16)

    # ── Wink ─────────────────────────────────────────────────────────────
    def trigger_wink(self):
        if self.is_winking:
            return
        self.is_winking = True
        self.wink_progress = 0.0
        self.wink_direction = 1
        self.wink_eye = random.choice(["left", "right"])
        self._wink_timer.start(16)

    def _update_wink(self):
        self.wink_progress += 0.12 * self.wink_direction
        if self.wink_direction == 1 and self.wink_progress >= 1.0:
            self.wink_progress = 1.0
            self._wink_timer.stop()
            self._wink_hold.start(100)
        elif self.wink_direction == -1 and self.wink_progress <= 0.0:
            self.wink_progress = 0.0
            self._wink_timer.stop()
            self.is_winking = False
        self.update()

    def _wink_return(self):
        self.wink_direction = -1
        self._wink_timer.start(16)

    # ── Mouse tracking ───────────────────────────────────────────────────
    def _poll_mouse(self):
        """
        Poll QCursor.pos() at ~30fps and update iris position.

        Only runs while eyes are open. Skips repaint if the iris
        hasn't moved enough to be visually noticeable (dead-zone).
        """
        if not self.mouse_tracking_enabled or self.isHidden():
            return

        gpos = QCursor.pos()
        local = self.mapFromGlobal(gpos)
        sf = self.scale_factor
        lx = local.x() / sf
        ly = local.y() / sf

        # Compute what the iris offset would be
        cx, cy = 350, 225
        dx = lx - cx
        dy = ly - cy
        dist = math.sqrt(dx * dx + dy * dy)
        mx = 18
        if dist > 0:
            f = min(dist / 150, 1.0)
            ix = (dx / dist) * mx * f
            iy = (dy / dist) * mx * f
        else:
            ix, iy = 0.0, 0.0

        # Dead-zone: skip repaint if iris moved less than ~0.8px
        if (abs(ix - self._last_iris_x) < 0.8
                and abs(iy - self._last_iris_y) < 0.8):
            return

        self._last_iris_x = ix
        self._last_iris_y = iy
        self.mouse_pos = QPointF(lx, ly)

        # Detect if mouse is actively near the widget (suppress auto-gaze)
        widget_cx, widget_cy = 350, 225
        mouse_dist = math.sqrt((lx - widget_cx) ** 2 + (ly - widget_cy) ** 2)
        self._gaze_mouse_active = mouse_dist < 300

        # Eye-open level and eyebrow offset
        y = ly
        eye_level = 225
        if y < eye_level - 50:
            self.eye_open_level = 1.0
        elif abs(y - eye_level) <= 50:
            self.eye_open_level = 0.75
        else:
            self.eye_open_level = 0.5
        vd = y - eye_level
        brow_max = 8
        if abs(vd) > 80:
            self.eyebrow_offset = (-brow_max if vd < 0 else brow_max) \
                * min(abs(vd) / 150, 1.0)
        else:
            self.eyebrow_offset *= 0.9
            if abs(self.eyebrow_offset) < 0.1:
                self.eyebrow_offset = 0.0

        self.update()

    def closeEvent(self, event):
        self._mouse_poll_timer.stop()
        self._stop_autonomous_animations()
        self._kill_bridge()
        super().closeEvent(event)

    # ── Intro draw-on animation ─────────────────────────────────────────

    def start_intro_animation(self):
        """Begin the line-drawing intro animation."""
        self._intro_playing = True
        self._intro_progress = 0.0
        self._intro_timer.start(16)      # ~60 fps
        self.update()

    def _update_intro(self):
        speed = 16.0 / self._INTRO_DURATION_MS
        self._intro_progress = min(1.0, self._intro_progress + speed)
        if self._intro_progress >= 1.0:
            self._intro_playing = False
            self._intro_timer.stop()
        self.update()

    def _intro_pen(self, base_pen, path, stage_start, stage_end):
        """Return a QPen with a dash-offset that reveals the path progressively.
        stage_start/stage_end map into the 0-1 intro progress range.
        Returns (pen, should_draw)."""
        if not self._intro_playing and self._intro_progress >= 1.0:
            return base_pen, True          # fully drawn, normal pen

        # Map global progress to this stage's local 0-1
        if self._intro_progress < stage_start:
            return base_pen, False         # not started yet
        local = min(1.0, (self._intro_progress - stage_start)
                    / max(0.001, stage_end - stage_start))
        # Ease-out for smooth deceleration
        local = 1.0 - (1.0 - local) ** 2

        path_len = path.length()
        if path_len < 1:
            return base_pen, True

        pen = QPen(base_pen)
        pen.setDashPattern([path_len, path_len])
        pen.setDashOffset(path_len * (1.0 - local))
        return pen, True

    def _intro_opacity(self, stage_start, stage_end):
        """Return an opacity float 0-1 for fade-in elements during intro."""
        if not self._intro_playing and self._intro_progress >= 1.0:
            return 1.0
        if self._intro_progress < stage_start:
            return 0.0
        local = min(1.0, (self._intro_progress - stage_start)
                    / max(0.001, stage_end - stage_start))
        return local

    # ── Random eye motion (saccades + micro-drift) ─────────────────

    def _start_autonomous_animations(self):
        """Start all autonomous animations (called when eyes open)."""
        # Gaze
        self._gaze_offset_x = 0.0
        self._gaze_offset_y = 0.0
        self._gaze_timer.start(33)   # ~30 fps
        self._schedule_next_saccade()
        # Auto-blink
        self._schedule_next_blink()
        # Tilt/rotate
        self._tilt_elapsed.start()
        self._tilt_timer.start(16)   # ~60 fps
        # Drastic perspective tilt
        self._drastic_schedule()

    def _stop_autonomous_animations(self):
        """Stop all autonomous animations (called when eyes close)."""
        self._gaze_timer.stop()
        self._gaze_saccade_timer.stop()
        self._auto_blink_timer.stop()
        self._auto_blink_anim_timer.stop()
        self._auto_blink_progress = 0.0
        self._auto_blink_phase = 0
        self._tilt_timer.stop()
        # Drastic tilt — stop and reset proxy to identity
        self._drastic_trigger_timer.stop()
        self._drastic_hold_timer.stop()
        self._drastic_active = False
        self._drastic_phase = 0
        self._drastic_sx = self._drastic_sy = 1.0
        self._drastic_shx = self._drastic_shy = 0.0
        if self._proxy is not None:
            self._proxy.setTransform(QTransform())

    def _schedule_next_saccade(self):
        """Schedule the next random gaze shift after a random interval."""
        delay = random.randint(400, 2800)
        self._gaze_saccade_timer.start(delay)

    def _new_gaze_target(self):
        """Pick a new random gaze target (saccade)."""
        if self.eyes_are_closed:
            return
        # Small micro-saccade (70%) vs larger gaze shift (30%)
        if random.random() < 0.7:
            r = random.uniform(2, 7)
        else:
            r = random.uniform(7, 16)
        angle = random.uniform(0, 2 * math.pi)
        self._gaze_target_x = r * math.cos(angle)
        self._gaze_target_y = r * math.sin(angle) * 0.6  # less vertical
        # Vary speed: fast saccade then slow settle
        self._gaze_speed = random.uniform(0.08, 0.18)
        self._schedule_next_saccade()

    def _update_gaze(self):
        """Smoothly interpolate gaze offset toward target."""
        if self.eyes_are_closed:
            return
        # If mouse is close, suppress autonomous gaze
        if self._gaze_mouse_active:
            self._gaze_offset_x *= 0.9
            self._gaze_offset_y *= 0.9
            return
        spd = self._gaze_speed
        self._gaze_offset_x += (self._gaze_target_x - self._gaze_offset_x) * spd
        self._gaze_offset_y += (self._gaze_target_y - self._gaze_offset_y) * spd
        # Add micro-jitter (tremor)
        self._gaze_offset_x += random.uniform(-0.15, 0.15)
        self._gaze_offset_y += random.uniform(-0.10, 0.10)

    # ── Random blink ─────────────────────────────────────────────────

    def _schedule_next_blink(self):
        """Schedule next auto-blink with natural random interval."""
        if self.eyes_are_closed:
            return
        # Humans blink every 2-8 seconds on average
        delay = random.randint(2200, 6500)
        # Occasional double-blink
        if random.random() < 0.12:
            delay = random.randint(150, 350)
        self._auto_blink_timer.start(delay)

    def _trigger_auto_blink(self):
        """Initiate an auto-blink (quick close-open cycle)."""
        if self.eyes_are_closed or self.is_animating or self.is_flickering:
            self._schedule_next_blink()
            return
        self._auto_blink_phase = 1  # closing
        self._auto_blink_progress = 0.0
        self._auto_blink_speed = random.uniform(0.14, 0.22)
        self._auto_blink_anim_timer.start(16)

    def _update_auto_blink(self):
        """Animate the auto-blink (fast close, slightly slower open)."""
        if self._auto_blink_phase == 1:  # closing
            self._auto_blink_progress += self._auto_blink_speed
            if self._auto_blink_progress >= 1.0:
                self._auto_blink_progress = 1.0
                self._auto_blink_phase = 2  # opening
        elif self._auto_blink_phase == 2:  # opening
            self._auto_blink_progress -= self._auto_blink_speed * 0.75
            if self._auto_blink_progress <= 0.0:
                self._auto_blink_progress = 0.0
                self._auto_blink_phase = 0
                self._auto_blink_anim_timer.stop()
                self._schedule_next_blink()
        self.update()

    # ── Tilt / rotate / shadow (circular proxy motion) ───────────────

    def attach_proxy_shadow(self, proxy):
        """Store the proxy reference for QTransform perspective tilts.
        Called from main.py after addWidget()."""
        self._proxy = proxy
        # Set the transform origin to the widget center so scale/shear
        # pivot around the middle, not the top-left corner.
        cx = self._base_w * self.scale_factor / 2.0
        cy = self._base_h * self.scale_factor / 2.0
        proxy.setTransformOriginPoint(cx, cy)

    def _update_tilt(self):
        """
        Animate tilt: circular orbit motion with varying radius (0-18px),
        gentle rotation, and drop shadow that responds to offset.
        """
        dt = 1.0  # normalized per tick
        # Advance circular orbit phase
        self._tilt_phase += self._tilt_phase_speed * dt
        if self._tilt_phase > 2 * math.pi:
            self._tilt_phase -= 2 * math.pi

        # Vary the orbit radius using a separate slow sinusoid
        self._tilt_radius_phase += self._tilt_radius_speed * dt
        if self._tilt_radius_phase > 2 * math.pi:
            self._tilt_radius_phase -= 2 * math.pi
        # Radius oscillates between ~2 and 18 pixels
        base = 2.0
        vary = 16.0  # 2 + 16 = 18 max
        self._tilt_radius = base + vary * (0.5 + 0.5 * math.sin(self._tilt_radius_phase))

        # Compute XY offset from circular motion
        self._tilt_offset_x = self._tilt_radius * math.cos(self._tilt_phase)
        self._tilt_offset_y = self._tilt_radius * math.sin(self._tilt_phase) * 0.5  # squash vertical

        # Rotation angle follows the offset direction with dampening
        target_angle = math.degrees(math.atan2(self._tilt_offset_y, self._tilt_offset_x)) * 0.04
        target_angle = max(-self._tilt_max_angle, min(self._tilt_max_angle, target_angle))
        self._tilt_angle += (target_angle - self._tilt_angle) * 0.06

        # Advance drastic perspective tilt
        self._advance_drastic()
        self._apply_proxy_transform()

        self.update()

    # ── Drastic perspective tilt (QTransform on proxy) ───────────────
    #
    # Pure affine: scale(sx, sy) + shear(shx, shy).  The proxy's
    # transformOriginPoint is already set to the widget centre so the
    # tilt pivots naturally.  We interpolate 4 scalar parameters
    # (sx, sy, shx, shy) and build a fresh QTransform each tick.
    # No projective (m13/m23) terms — those cause the rasteriser to
    # clip or blank the proxy on many backends.

    def _drastic_schedule(self):
        """Schedule next drastic tilt 5-6 seconds from now."""
        delay = random.randint(5000, 6000)
        self._drastic_trigger_timer.start(delay)

    def _drastic_trigger(self):
        """Fire a drastic perspective tilt with random parameters."""
        if self.eyes_are_closed or self._proxy is None:
            return
        self._drastic_active = True
        self._drastic_phase = 1   # attack
        self._drastic_progress = 0.0

        # Snapshot current values as animation start
        self._drastic_start_sx = self._drastic_sx
        self._drastic_start_sy = self._drastic_sy
        self._drastic_start_shx = self._drastic_shx
        self._drastic_start_shy = self._drastic_shy

        # Pick a random tilt axis by choosing which shear axis to use
        # and how much vertical foreshortening to apply.
        axis = random.random()
        if axis < 0.35:
            # Horizontal axis tilt (like looking up/down at it)
            self._drastic_target_shx = 0.0
            self._drastic_target_shy = random.choice([-1, 1]) * random.uniform(0.15, 0.28)
            self._drastic_target_sx = random.uniform(0.88, 0.94)
            self._drastic_target_sy = random.uniform(0.70, 0.80)
        elif axis < 0.70:
            # Vertical axis tilt (like turning it left/right)
            self._drastic_target_shx = random.choice([-1, 1]) * random.uniform(0.15, 0.30)
            self._drastic_target_shy = 0.0
            self._drastic_target_sx = random.uniform(0.85, 0.92)
            self._drastic_target_sy = random.uniform(0.82, 0.90)
        else:
            # Diagonal / compound tilt (both axes)
            self._drastic_target_shx = random.choice([-1, 1]) * random.uniform(0.10, 0.22)
            self._drastic_target_shy = random.choice([-1, 1]) * random.uniform(0.08, 0.16)
            self._drastic_target_sx = random.uniform(0.82, 0.90)
            self._drastic_target_sy = random.uniform(0.72, 0.82)

    def _advance_drastic(self):
        """Step the drastic tilt animation forward (called every tilt tick)."""
        if not self._drastic_active:
            return

        if self._drastic_phase == 1:          # ── attack: smooth ease-out
            self._drastic_progress = min(1.0, self._drastic_progress + 0.025)
            t = 1.0 - (1.0 - self._drastic_progress) ** 3
            self._drastic_sx  = self._drastic_start_sx  + (self._drastic_target_sx  - self._drastic_start_sx)  * t
            self._drastic_sy  = self._drastic_start_sy  + (self._drastic_target_sy  - self._drastic_start_sy)  * t
            self._drastic_shx = self._drastic_start_shx + (self._drastic_target_shx - self._drastic_start_shx) * t
            self._drastic_shy = self._drastic_start_shy + (self._drastic_target_shy - self._drastic_start_shy) * t
            if self._drastic_progress >= 1.0:
                self._drastic_phase = 2       # hold
                self._drastic_hold_timer.start(random.randint(300, 650))

        elif self._drastic_phase == 3:        # ── release: slow ease-in-out
            self._drastic_progress = min(1.0, self._drastic_progress + 0.010)
            # smooth ease-in-out
            if self._drastic_progress < 0.5:
                t = 2 * self._drastic_progress ** 2
            else:
                t = 1 - (-2 * self._drastic_progress + 2) ** 2 / 2
            self._drastic_sx  = self._drastic_start_sx  + (1.0 - self._drastic_start_sx)  * t
            self._drastic_sy  = self._drastic_start_sy  + (1.0 - self._drastic_start_sy)  * t
            self._drastic_shx = self._drastic_start_shx + (0.0 - self._drastic_start_shx) * t
            self._drastic_shy = self._drastic_start_shy + (0.0 - self._drastic_start_shy) * t
            if self._drastic_progress >= 1.0:
                self._drastic_active = False
                self._drastic_phase = 0
                self._drastic_sx = self._drastic_sy = 1.0
                self._drastic_shx = self._drastic_shy = 0.0
                self._drastic_schedule()

    def _drastic_begin_release(self):
        """Called after hold timer — begin easing back to identity."""
        self._drastic_start_sx = self._drastic_sx
        self._drastic_start_sy = self._drastic_sy
        self._drastic_start_shx = self._drastic_shx
        self._drastic_start_shy = self._drastic_shy
        self._drastic_phase = 3
        self._drastic_progress = 0.0

    def _apply_proxy_transform(self):
        """Build and apply a QTransform from the current drastic parameters.
        transformOriginPoint is already set to widget centre so we just
        need to supply a plain scale+shear matrix."""
        if self._proxy is None:
            return
        t = QTransform()
        t.scale(self._drastic_sx, self._drastic_sy)
        t.shear(self._drastic_shx, self._drastic_shy)
        self._proxy.setTransform(t)

    # ── Data generation ──────────────────────────────────────────────────
    def _gen_eyelash_data(self):
        p = self._profile
        n_upper = p["eyelash_density"]
        length_mult = p["eyelash_length"]

        self._left_upper_lashes = []
        self._right_upper_lashes = []
        self._left_lower_lashes = []
        self._right_lower_lashes = []

        for i in range(n_upper):
            pos = i / max(n_upper - 1, 1)
            base_x = pos * 140 - 70
            sf = 1 + 0.3 * math.sin(pos * math.pi)
            ax = base_x * sf
            bl = (8 + 12 * math.sin(pos * math.pi)) * length_mult
            lv = random.uniform(0.7, 1.3)
            length = bl * lv + 3
            ba = -25 - 15 * math.sin(pos * math.pi)
            av = random.uniform(-8, 8)
            angle = ba + av
            thickness = max(0.5, 2.2 - pos * 1.2)
            self._left_upper_lashes.append((ax, length, angle, thickness))
            rx = ((1 - pos) * 140 - 70) * -1 * sf
            ra = ba * -1 + av + p.get("right_lash_angle_offset", 0)
            self._right_upper_lashes.append((rx, length, ra, thickness))

        for i in range(15):
            pos = i / 14.0
            bx = pos * 100 - 50
            ln = 3 + 5 * (1 - abs(pos - 0.5) * 2)
            a = 15 + 10 * math.sin(pos * math.pi)
            th = max(0.3, 1.0 - abs(pos - 0.5) * 0.8)
            self._left_lower_lashes.append((bx, ln, a, th))
            rx = ((1 - pos) * 100 - 50) * -1
            ra = -(15 + 10 * math.sin(pos * math.pi)) + p.get("right_lash_angle_offset", 0)
            self._right_lower_lashes.append((rx, ln, ra, th))

    def _gen_iris_data(self):
        self._left_iris = self._make_iris()
        self._right_iris = self._make_iris()

    def _make_iris(self):
        d = {}
        d['fibers'] = [
            (a + random.uniform(-3, 3),
             8 + random.uniform(-2, 2),
             25 - random.uniform(2, 6),
             random.uniform(-3, 3))
            for a in range(0, 360, 12)]
        d['fine'] = [
            (a + random.uniform(-2, 2),
             10 + random.uniform(-3, 3),
             25 - random.uniform(1, 8))
            for a in range(0, 360, 6)]
        d['crypts'] = [
            (random.uniform(-12, 12),
             random.uniform(-12, 12),
             random.uniform(1.5, 3.5))
            for _ in range(8)]
        d['pupil_var'] = [
            random.uniform(-0.5, 0.5)
            for _ in range(0, 360, 20)]
        return d

    # ── Iris position calc ───────────────────────────────────────────────
    def _iris_pos(self, eye_center):
        cx, cy = 350, 225
        dx = self.mouse_pos.x() - cx
        dy = self.mouse_pos.y() - cy
        dist = math.sqrt(dx * dx + dy * dy)
        mx = 18
        if dist > 0:
            f = min(dist / 150, 1.0)
            mouse_ix = (dx / dist) * mx * f
            mouse_iy = (dy / dist) * mx * f
        else:
            mouse_ix, mouse_iy = 0.0, 0.0

        # Blend autonomous gaze with mouse-driven gaze
        if self._gaze_mouse_active:
            blend = 0.1  # mostly mouse
        else:
            blend = 0.7  # mostly autonomous
        ix = mouse_ix * (1 - blend) + self._gaze_offset_x * blend
        iy = mouse_iy * (1 - blend) + self._gaze_offset_y * blend

        # Clamp to max travel
        id_ = math.sqrt(ix * ix + iy * iy)
        if id_ > mx:
            ix = ix / id_ * mx
            iy = iy / id_ * mx
        return ix, iy

    @staticmethod
    def _ease(t):
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2

    # ── Paint ────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(self.scale_factor, self.scale_factor)

        # ── Tilt / rotate transform on proxy ─────────────────────────
        painter.save()
        if self._tilt_enabled and not self.eyes_are_closed:
            painter.translate(350, 225)
            painter.rotate(self._tilt_angle)
            painter.translate(-350 + self._tilt_offset_x,
                              -225 + self._tilt_offset_y)

        self._draw_eye(painter, QPointF(200, 225), False)
        self._draw_eye(painter, QPointF(500, 225), True)
        painter.restore()

        if self._label_opacity > 0.01:
            self._draw_label(painter)

    def _draw_label(self, painter):
        painter.save()
        c = QColor(self._profile["label_color"])
        c.setAlphaF(self._label_opacity)
        painter.setPen(QPen(c, 1))
        f = QFont("Helvetica Neue", 28, QFont.Weight.Light)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 8)
        painter.setFont(f)
        rect = QRectF(0, 370, self._base_w, 60)
        label = self._profile["display_name"]
        if self._master_mode:
            label = "$" + label
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, label)
        painter.restore()

    def _draw_eye(self, painter, center, mirror=False):
        p = self._profile
        x, y = center.x(), center.y()
        mf = -1 if mirror else 1
        ws = p["eye_width_scale"]
        hs = p["eye_height_scale"]

        # Corner shaping — lift outer corners, drop inner for curved-up look
        ocl = p.get("outer_corner_lift", 0)   # outer corner lift (pos = up)
        icd = p.get("inner_corner_drop", 0)   # inner corner drop (neg = down)
        ulcb = p.get("upper_lid_curve_boost", 0)  # extra upper lid curve

        # Confirmed via debug dots: -75*mf = OUTER corner, +75*mf = INNER corner
        neg75_adj = -ocl    # outer corner at -75*mf: lift up
        pos75_adj = -icd    # inner corner at +75*mf: drop down

        bp = self.blink_progress
        if self.is_winking:
            side = "right" if mirror else "left"
            if side == self.wink_eye:
                bp = max(bp, self.wink_progress)
        # Overlay auto-blink on top (combines with existing blink state)
        if self._auto_blink_phase > 0:
            bp = max(bp, self._auto_blink_progress)

        eff = self.eye_open_level * (1 - bp)
        rest = (1.0 - eff) * 32 * hs
        be = self._ease(bp)
        is_closed = bp >= 0.99

        nuy = -25 * hs
        nly = 22 * hs
        ulo = nuy + rest + ((55 * hs - rest) * be)
        lr = rest * 0.25
        lld = 12 * hs
        llo = nly - lr - ((lld - lr) * be)

        # ── Eyebrow — single clean curved line ──────────────────────────
        ebo = self.eyebrow_offset + p["eyebrow_y_offset"]
        arch = p["eyebrow_arch"]
        brow_pen = QPen(Qt.black, p["eyebrow_thickness"],
                        Qt.SolidLine, Qt.RoundCap)
        brow = QPainterPath()
        brow.moveTo(x - 78 * mf * ws, y - 58 + ebo)
        brow.cubicTo(x - 20 * mf * ws, y - 82 + ebo - arch,
                     x + 30 * mf * ws, y - 84 + ebo - arch,
                     x + 78 * mf * ws, y - 52 + ebo)
        pen, draw = self._intro_pen(brow_pen, brow, 0.0, 0.35)
        if draw:
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(brow)

        # ── Iris / pupil ─────────────────────────────────────────────────
        iris_alpha = self._intro_opacity(0.30, 0.55)
        if not is_closed and not self.is_flickering and iris_alpha > 0.01:
            ix, iy = self._iris_pos(center)
            icx = x + ix
            icy = y - 10 + iy

            clip = QPainterPath()
            clip.moveTo(x - 75 * mf * ws, y + 5 + neg75_adj)
            clip.quadTo(x - 40 * mf * ws, y + ulo - ulcb,
                        x - 5 * mf * ws, y + ulo - ulcb)
            clip.quadTo(x + 30 * mf * ws, y + ulo - ulcb,
                        x + 75 * mf * ws, y + 5 + pos75_adj)
            clip.quadTo(x + 35 * mf * ws, y + llo,
                        x + 5 * mf * ws, y + llo)
            clip.quadTo(x - 25 * mf * ws, y + llo,
                        x - 75 * mf * ws, y + 5 + neg75_adj)
            clip.closeSubpath()

            painter.save()
            painter.setClipPath(clip)
            painter.setOpacity(painter.opacity() * iris_alpha)

            # Sclera
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 252)))
            painter.drawPath(clip)

            iris_data = self._left_iris if not mirror else self._right_iris

            # Iris
            ir = 26
            grad = QRadialGradient(icx, icy, ir)
            grad.setColorAt(0.0, p["iris_color"].lighter(130))
            grad.setColorAt(0.7, p["iris_color"])
            grad.setColorAt(1.0, p["iris_rim_color"])
            painter.setBrush(QBrush(grad))
            painter.setPen(QPen(p["iris_rim_color"].darker(130), 1.2))
            painter.drawEllipse(QPointF(icx, icy), ir, ir)

            # Fiber lines
            painter.setPen(QPen(p["iris_rim_color"].darker(110), 0.3))
            for ang, si, eo, co in iris_data['fibers']:
                r = math.radians(ang)
                painter.drawLine(
                    QPointF(icx + si * math.cos(r),
                            icy + si * math.sin(r)),
                    QPointF(icx + eo * math.cos(r + co * 0.02),
                            icy + eo * math.sin(r + co * 0.02)))

            # Crypts
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(p["iris_rim_color"].darker(140)))
            for cx_, cy_, sz in iris_data['crypts']:
                painter.drawEllipse(
                    QPointF(icx + cx_, icy + cy_), sz, sz)

            # Pupil
            pr = 8
            pvars = iris_data['pupil_var']
            pupil_pts = []
            for i, ang in enumerate(range(0, 360, 20)):
                v = pvars[i] if i < len(pvars) else 0
                r = math.radians(ang)
                pupil_pts.append(
                    QPointF(icx + (pr + v) * math.cos(r),
                            icy + (pr + v) * math.sin(r)))
            pp = QPainterPath()
            if pupil_pts:
                pp.moveTo(pupil_pts[0])
                for pt in pupil_pts[1:]:
                    pp.lineTo(pt)
                pp.closeSubpath()
            painter.setPen(QPen(Qt.black, 1.5))
            painter.setBrush(QBrush(Qt.black))
            painter.drawPath(pp)

            # Catchlights
            painter.setBrush(QBrush(Qt.white))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(icx - 6, icy - 5), 4, 5)
            painter.drawEllipse(QPointF(icx + 4, icy - 7), 2, 2.5)
            painter.drawEllipse(QPointF(icx - 2, icy + 6), 1, 1)

            painter.restore()

        # ── Upper eyelid ─────────────────────────────────────────────────
        lid_pen = QPen(Qt.black, p["lid_thickness"])
        ul = QPainterPath()
        ul.moveTo(x - 75 * mf * ws, y + 5 + neg75_adj)
        ul.quadTo(x - 40 * mf * ws, y + ulo - ulcb,
                  x - 5 * mf * ws, y + ulo - ulcb)
        ul.quadTo(x + 30 * mf * ws, y + ulo - ulcb,
                  x + 75 * mf * ws, y + 5 + pos75_adj)
        pen, draw = self._intro_pen(lid_pen, ul, 0.05, 0.45)
        if draw:
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(ul)

        # ── Upper eyelashes ──────────────────────────────────────────────
        lash_alpha = self._intro_opacity(0.20, 0.60)
        lashes = (self._right_upper_lashes if mirror
                  else self._left_upper_lashes)
        y_flip = -1 if (is_closed or self.is_flickering) else 1

        if lash_alpha > 0.01:
            painter.save()
            painter.setOpacity(painter.opacity() * lash_alpha)

            # Build the same upper lid path used for drawing, so we can
            # sample exact points with pointAtPercent().
            _lid = QPainterPath()
            _lid.moveTo(x - 75 * mf * ws, y + 5 + neg75_adj)
            _lid.quadTo(x - 40 * mf * ws, y + ulo - ulcb,
                        x - 5 * mf * ws, y + ulo - ulcb)
            _lid.quadTo(x + 30 * mf * ws, y + ulo - ulcb,
                        x + 75 * mf * ws, y + 5 + pos75_adj)
            _lid_len = _lid.length()

            for lx, ll, la, lt in lashes:
                painter.setPen(
                    QPen(Qt.black, lt, Qt.SolidLine, Qt.RoundCap))
                # Map lash x-position to a percent along the lid path.
                # lx is in "left-eye space" (-70..+70 roughly).
                # The lid path starts at x - 75*mf*ws and ends at
                # x + 75*mf*ws.  For the left eye (mf=+1) the path
                # goes left-to-right; for the right eye (mf=-1) it
                # goes right-to-left.  We need the x-coordinate of
                # the lash in world space: x + lx*ws, then find which
                # percent along the lid has that x.
                lash_world_x = x + lx * ws
                lid_start_x = x - 75 * mf * ws
                lid_end_x = x + 75 * mf * ws
                span_x = lid_end_x - lid_start_x
                if abs(span_x) > 0.1:
                    pct = max(0.0, min(1.0,
                              (lash_world_x - lid_start_x) / span_x))
                else:
                    pct = 0.5
                sp = _lid.pointAtPercent(pct)
                rad = math.radians(la)
                ep = QPointF(sp.x() + ll * math.sin(rad),
                             sp.y() - ll * math.cos(rad) * y_flip)
                cp = QPointF(
                    sp.x() + ll * 0.5 * math.sin(rad) * y_flip,
                    sp.y() - ll * 0.5 * math.cos(
                        rad - 0.8 * mf) * y_flip)
                lp = QPainterPath(sp)
                lp.quadTo(cp, ep)
                painter.drawPath(lp)
            painter.restore()

        # ── Lower lid + lashes (only when open) ─────────────────────────
        lower_alpha = self._intro_opacity(0.35, 0.65)
        if not is_closed and not self.is_flickering and lower_alpha > 0.01:
            painter.save()
            painter.setOpacity(painter.opacity() * lower_alpha)

            lo_lashes = (self._right_lower_lashes if mirror
                         else self._left_lower_lashes)
            for lx, ll, la, lt in lo_lashes:
                painter.setPen(
                    QPen(Qt.black, lt, Qt.SolidLine, Qt.RoundCap))
                nx = lx / (75 * abs(mf) * ws)
                sy = (y + llo) + (y + 5 - (y + llo)) * nx * nx
                sp = QPointF(x + lx * ws, sy)
                rad = math.radians(la)
                ep = QPointF(sp.x() - ll * math.sin(rad),
                             sp.y() + ll * math.cos(rad))
                cp = QPointF(
                    sp.x() + ll * 0.5 * math.sin(rad - 1 * mf),
                    sp.y() + ll * 0.5 * math.cos(
                        rad - 0.1 * mf))
                lp = QPainterPath(sp)
                lp.quadTo(cp, ep)
                painter.drawPath(lp)

            # Lower lid line
            lower_lid_pen = QPen(Qt.black, p["lower_lid_thickness"])
            ll_path = QPainterPath()
            ll_path.moveTo(x - 75 * mf * ws, y + 5 + neg75_adj)
            ll_path.quadTo(x - 25 * mf * ws, y + llo,
                           x + 5 * mf * ws, y + llo)
            ll_path.quadTo(x + 35 * mf * ws, y + llo,
                           x + 75 * mf * ws, y + 5 + pos75_adj)
            pen, draw = self._intro_pen(lower_lid_pen, ll_path, 0.35, 0.65)
            if draw:
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(ll_path)

            # Tear duct
            td = QPainterPath()
            if not mirror:
                td.moveTo(x - 75 * ws, y + 5 + neg75_adj)
                td.quadTo(x - 85 * ws, y - 2 + neg75_adj,
                          x - 75 * ws, y - 8 + neg75_adj)
            else:
                td.moveTo(x + 75 * ws, y + 5 + neg75_adj)
                td.quadTo(x + 85 * ws, y - 2 + neg75_adj,
                          x + 75 * ws, y - 8 + neg75_adj)
            td_pen = QPen(Qt.black, p["lower_lid_thickness"])
            pen, draw = self._intro_pen(td_pen, td, 0.45, 0.70)
            if draw:
                painter.setPen(pen)
                painter.drawPath(td)

            painter.restore()

    # ── Key events ───────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_F and not self.eyes_are_closed:
            self.trigger_flicker()
        elif k == Qt.Key.Key_W and not self.eyes_are_closed:
            self.trigger_wink()
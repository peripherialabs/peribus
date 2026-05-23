"""
op_core.py — Operator core: theme, model, scanner, canvas chrome, app.
======================================================================

Consolidated from what used to be:
    op/theme.py         → "─── THEME ─── " section
    op/graph.py         → "─── GRAPH (MODEL) ───" section
    op/scanner.py       → "─── SCANNER ───" section
    op/canvas.py        → "─── CANVAS CHROME ───" section
    op/operator_app.py  → "─── OPERATOR APP ───" section

The view classes (NodeView and subclasses, PortItem, ConnectionItem,
TempConnectionItem) live in op_nodes.py. They're imported here lazily
(inside Operator._on_node_added) to avoid a hard import cycle, since
op_nodes.py imports the model/theme/types defined below.

Package name is `op` (not `operator`) to avoid shadowing the stdlib.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from PySide6.QtCore import (
    Qt, QObject, QEasingCurve, QEvent, QPointF, QProcess, QRectF, QTimer,
    QVariantAnimation, Signal,
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QGraphicsProxyWidget, QGraphicsScene, QHBoxLayout, QInputDialog,
    QMessageBox, QPushButton, QWidget,
)

from .pipe import FSWorker, Pipe, SubscribeMode

if TYPE_CHECKING:
    from .op_nodes import (
        NodeView, PortItem, ConnectionItem, TempConnectionItem,
    )


# ═══════════════════════════════════════════════════════════════════════
# ─── THEME ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Colors, fonts, and visual constants. Dark and light variants accessed
# through a single `Theme` proxy. Switch via `Theme.set_mode(dark=...)`.


class _Dark:
    # Canvas
    CANVAS_BG = QColor(22, 22, 26)
    GRID_LINE = QColor(38, 38, 44)

    # Node body
    NODE_BG = QColor(32, 32, 38, 210)
    NODE_BG_TRANSLUCENT = QColor(32, 32, 38, 180)
    NODE_BORDER = QColor(62, 62, 72, 140)
    NODE_BORDER_HOVER = QColor(120, 120, 135, 180)
    NODE_BORDER_SELECTED = QColor(180, 180, 195, 220)
    SEPARATOR = QColor(56, 56, 66, 120)

    # Header colors keyed by NodeKind
    HEADER_AGENT = QColor(52, 58, 48, 240)
    HEADER_TERMINAL = QColor(46, 52, 62, 240)
    HEADER_SCENE = QColor(46, 52, 62, 240)
    HEADER_TEXT = QColor(42, 52, 58, 240)
    HEADER_DEBUG = QColor(58, 48, 52, 240)
    HEADER_MEDIA = QColor(54, 50, 58, 240)
    HEADER_BASH = QColor(50, 50, 50, 240)
    HEADER_PYTHON = QColor(48, 52, 62, 240)
    HEADER_GENERIC = QColor(44, 48, 56, 240)

    # Ports
    PORT_INPUT = QColor(110, 130, 165)
    PORT_OUTPUT = QColor(110, 145, 120)
    PORT_BLOCKING_BADGE = QColor(200, 160, 100)   # tiny dot on streaming ports
    PORT_BORDER = QColor(32, 32, 38, 180)

    # Connections
    CONN_DEFAULT = QColor(90, 90, 105, 140)
    CONN_RUNNING = QColor(160, 160, 178, 200)
    CONN_STOPPED = QColor(120, 80, 80, 160)
    CONN_HOVER = QColor(175, 175, 190, 220)
    CONN_DRAGGING = QColor(150, 150, 170, 120)        # right-drag: route
    CONN_DRAGGING_ONESHOT = QColor(220, 175, 110, 220)  # left-drag: one-shot

    # Text
    TEXT_PRIMARY = QColor(210, 210, 218)
    TEXT_SECONDARY = QColor(120, 120, 135)
    TEXT_ON_HEADER = QColor(195, 195, 205)
    TEXT_PORT = QColor(140, 140, 155)

    # Embedded widgets (text edits, line edits inside nodes)
    EDIT_BG = QColor(24, 24, 30, 220)
    EDIT_BG_FOCUS = QColor(28, 28, 36, 235)
    EDIT_BORDER = QColor(56, 56, 66, 180)
    EDIT_BORDER_FOCUS = QColor(120, 140, 175, 220)
    EDIT_SELECTION = QColor(70, 90, 130, 180)
    EDIT_PLACEHOLDER = QColor(95, 95, 110)
    CONTAINER_BG = QColor(0, 0, 0, 0)   # transparent — show node body

    # Buttons inside nodes — neutral default
    BUTTON_BG = QColor(48, 50, 58, 220)
    BUTTON_BG_HOVER = QColor(64, 68, 78, 235)
    BUTTON_BG_PRESSED = QColor(38, 40, 48, 240)
    BUTTON_BORDER = QColor(72, 76, 86, 220)
    BUTTON_BORDER_HOVER = QColor(110, 120, 140, 230)
    BUTTON_TEXT = QColor(220, 220, 228)
    BUTTON_TEXT_DISABLED = QColor(105, 105, 118)

    # Accent (primary action) button — Run / Write
    BUTTON_ACCENT_BG = QColor(58, 80, 110, 230)
    BUTTON_ACCENT_HOVER = QColor(74, 100, 138, 240)
    BUTTON_ACCENT_PRESSED = QColor(46, 64, 90, 240)
    BUTTON_ACCENT_BORDER = QColor(96, 130, 175, 230)
    BUTTON_ACCENT_TEXT = QColor(232, 238, 248)

    # Success-tinted button — Read
    BUTTON_READ_BG = QColor(52, 76, 64, 220)
    BUTTON_READ_HOVER = QColor(70, 100, 84, 235)
    BUTTON_READ_PRESSED = QColor(40, 60, 50, 240)
    BUTTON_READ_BORDER = QColor(90, 125, 105, 220)
    BUTTON_READ_TEXT = QColor(220, 232, 224)

    # Checkbox
    CHECK_BG = QColor(28, 28, 34, 220)
    CHECK_BORDER = QColor(80, 84, 96, 220)
    CHECK_CHECKED_BG = QColor(96, 130, 175, 230)
    CHECK_CHECKED_BORDER = QColor(120, 155, 195, 240)

    # Status text colors
    STATUS_OK = QColor(120, 175, 140)
    STATUS_ERROR = QColor(200, 110, 110)
    STATUS_BUSY = QColor(200, 175, 110)


class _Light:
    CANVAS_BG = QColor(248, 248, 250)
    GRID_LINE = QColor(220, 220, 226)

    NODE_BG = QColor(252, 252, 254, 235)
    NODE_BG_TRANSLUCENT = QColor(252, 252, 254, 200)
    NODE_BORDER = QColor(200, 200, 210, 200)
    NODE_BORDER_HOVER = QColor(140, 140, 155, 220)
    NODE_BORDER_SELECTED = QColor(80, 80, 100, 240)
    SEPARATOR = QColor(220, 220, 228, 140)

    HEADER_AGENT = QColor(220, 232, 200, 240)
    HEADER_TERMINAL = QColor(210, 220, 235, 240)
    HEADER_SCENE = QColor(210, 220, 235, 240)
    HEADER_TEXT = QColor(208, 222, 230, 240)
    HEADER_DEBUG = QColor(232, 212, 218, 240)
    HEADER_MEDIA = QColor(226, 218, 232, 240)
    HEADER_BASH = QColor(225, 225, 225, 240)
    HEADER_PYTHON = QColor(214, 222, 236, 240)
    HEADER_GENERIC = QColor(220, 224, 232, 240)

    PORT_INPUT = QColor(80, 105, 145)
    PORT_OUTPUT = QColor(80, 125, 95)
    PORT_BLOCKING_BADGE = QColor(180, 130, 60)
    PORT_BORDER = QColor(252, 252, 254, 220)

    CONN_DEFAULT = QColor(140, 140, 155, 160)
    CONN_RUNNING = QColor(60, 60, 80, 220)
    CONN_STOPPED = QColor(160, 100, 100, 180)
    CONN_HOVER = QColor(40, 40, 60, 240)
    CONN_DRAGGING = QColor(100, 100, 120, 140)
    CONN_DRAGGING_ONESHOT = QColor(195, 130, 50, 230)

    TEXT_PRIMARY = QColor(30, 30, 40)
    TEXT_SECONDARY = QColor(110, 110, 125)
    TEXT_ON_HEADER = QColor(40, 40, 55)
    TEXT_PORT = QColor(90, 90, 105)

    EDIT_BG = QColor(255, 255, 255, 230)
    EDIT_BG_FOCUS = QColor(255, 255, 255, 245)
    EDIT_BORDER = QColor(200, 200, 210, 200)
    EDIT_BORDER_FOCUS = QColor(95, 130, 185, 230)
    EDIT_SELECTION = QColor(180, 210, 245, 220)
    EDIT_PLACEHOLDER = QColor(165, 165, 175)
    CONTAINER_BG = QColor(0, 0, 0, 0)

    # Buttons inside nodes — neutral default
    BUTTON_BG = QColor(248, 248, 252, 240)
    BUTTON_BG_HOVER = QColor(238, 238, 245, 250)
    BUTTON_BG_PRESSED = QColor(225, 225, 232, 250)
    BUTTON_BORDER = QColor(200, 200, 212, 220)
    BUTTON_BORDER_HOVER = QColor(150, 150, 168, 230)
    BUTTON_TEXT = QColor(45, 45, 60)
    BUTTON_TEXT_DISABLED = QColor(170, 170, 180)

    # Accent (primary action) button
    BUTTON_ACCENT_BG = QColor(82, 122, 175, 235)
    BUTTON_ACCENT_HOVER = QColor(70, 108, 158, 240)
    BUTTON_ACCENT_PRESSED = QColor(56, 92, 138, 240)
    BUTTON_ACCENT_BORDER = QColor(64, 100, 150, 240)
    BUTTON_ACCENT_TEXT = QColor(255, 255, 255)

    # Success-tinted button — Read
    BUTTON_READ_BG = QColor(96, 150, 118, 230)
    BUTTON_READ_HOVER = QColor(82, 134, 102, 240)
    BUTTON_READ_PRESSED = QColor(68, 116, 88, 240)
    BUTTON_READ_BORDER = QColor(72, 124, 92, 240)
    BUTTON_READ_TEXT = QColor(255, 255, 255)

    # Checkbox
    CHECK_BG = QColor(255, 255, 255, 240)
    CHECK_BORDER = QColor(180, 180, 190, 220)
    CHECK_CHECKED_BG = QColor(82, 122, 175, 235)
    CHECK_CHECKED_BORDER = QColor(64, 100, 150, 240)

    # Status text colors
    STATUS_OK = QColor(60, 130, 90)
    STATUS_ERROR = QColor(180, 70, 70)
    STATUS_BUSY = QColor(180, 130, 50)


class _Paper:
    """Editorial 'paper' palette — matches start_gui's PAPER theme and
    the terminal's paper mode.

    Visual rules taken from start_gui.py:
      - Opaque off-white card (250, 247, 240). No translucency anywhere.
      - Single hairline 1px near-black border (42, 42, 42). No shadows.
      - Editorial small radii — handled in _Structure (2px corners).
      - Near-black ink for text, muted brown-gray for labels/captions.
      - Sage selection tint.
      - Amber for warnings, red ink for danger, green for ok.
      - IBM Plex Mono / IBM Plex Sans — set in _Structure.

    Node-kind accent colors are picked from the start_gui palette and
    its close cousins (amber, sage green, muted blues, ink). They're
    intentionally desaturated — paper aesthetic, not crayon — and
    used at 255 alpha for the small accent dot next to the title.
    """

    # Canvas
    CANVAS_BG = QColor(240, 237, 230)      # slightly darker than card
    GRID_LINE = QColor(220, 215, 205)

    # Node body — opaque paper, no translucency
    NODE_BG = QColor(250, 247, 240)
    NODE_BG_TRANSLUCENT = QColor(250, 247, 240)   # same — paper is opaque
    NODE_BORDER = QColor(42, 42, 42, 110)         # hairline soft
    NODE_BORDER_HOVER = QColor(42, 42, 42, 180)
    NODE_BORDER_SELECTED = QColor(26, 26, 26)
    SEPARATOR = QColor(42, 42, 42, 60)

    # Node-kind accent dots. Editorial palette — muted, in-keyed.
    HEADER_AGENT = QColor(120, 160, 110)     # sage green — "alive"
    HEADER_TERMINAL = QColor(95, 120, 155)   # muted blue
    HEADER_SCENE = QColor(95, 120, 155)      # same family as terminal
    HEADER_TEXT = QColor(120, 115, 105)      # ink muted — neutral
    HEADER_DEBUG = QColor(170, 50, 50)       # red ink — diagnostic
    HEADER_MEDIA = QColor(160, 110, 150)     # dusty rose
    HEADER_BASH = QColor(26, 26, 26)         # near-black — system
    HEADER_PYTHON = QColor(212, 142, 60)     # amber — energetic
    HEADER_GENERIC = QColor(140, 135, 125)   # muted

    # Ports — keep direction encoding (input/output) but in paper hues
    PORT_INPUT = QColor(95, 120, 155)        # muted blue
    PORT_OUTPUT = QColor(120, 160, 110)      # sage green
    PORT_BLOCKING_BADGE = QColor(212, 142, 60)   # amber dot
    PORT_BORDER = QColor(26, 26, 26, 160)

    # Connections — ink lines, not glowing wires
    CONN_DEFAULT = QColor(42, 42, 42, 120)
    CONN_RUNNING = QColor(26, 26, 26, 220)        # dark, confident
    CONN_STOPPED = QColor(170, 50, 50, 150)       # red ink, faded
    CONN_HOVER = QColor(26, 26, 26)
    CONN_DRAGGING = QColor(42, 42, 42, 140)
    CONN_DRAGGING_ONESHOT = QColor(212, 142, 60, 220)  # amber dashed

    # Text
    TEXT_PRIMARY = QColor(26, 26, 26)        # near-black ink
    TEXT_SECONDARY = QColor(120, 115, 105)   # ink muted
    TEXT_ON_HEADER = QColor(26, 26, 26)      # unused now, kept for API
    TEXT_PORT = QColor(120, 115, 105)

    # Embedded widgets — paper inputs are underlined, not boxed
    EDIT_BG = QColor(250, 247, 240)
    EDIT_BG_FOCUS = QColor(252, 250, 244)    # log bg from start_gui
    EDIT_BORDER = QColor(42, 42, 42, 80)     # hairline soft
    EDIT_BORDER_FOCUS = QColor(26, 26, 26)   # hairline solid on focus
    EDIT_SELECTION = QColor(180, 200, 180, 140)  # sage selection tint
    EDIT_PLACEHOLDER = QColor(120, 115, 105, 140)
    CONTAINER_BG = QColor(0, 0, 0, 0)

    # Buttons — pill chips from start_gui's [pill="true"] style
    BUTTON_BG = QColor(250, 247, 240)
    BUTTON_BG_HOVER = QColor(244, 240, 230)
    BUTTON_BG_PRESSED = QColor(232, 228, 218)
    BUTTON_BORDER = QColor(42, 42, 42, 80)
    BUTTON_BORDER_HOVER = QColor(26, 26, 26)
    BUTTON_TEXT = QColor(26, 26, 26)
    BUTTON_TEXT_DISABLED = QColor(26, 26, 26, 80)

    # Primary action — flat black, paper's primary
    BUTTON_ACCENT_BG = QColor(26, 26, 26)
    BUTTON_ACCENT_HOVER = QColor(50, 50, 50)
    BUTTON_ACCENT_PRESSED = QColor(10, 10, 10)
    BUTTON_ACCENT_BORDER = QColor(26, 26, 26)
    BUTTON_ACCENT_TEXT = QColor(250, 247, 240)

    # Read action — sage green ink, paper-style
    BUTTON_READ_BG = QColor(250, 247, 240)
    BUTTON_READ_HOVER = QColor(80, 140, 80)
    BUTTON_READ_PRESSED = QColor(60, 110, 60)
    BUTTON_READ_BORDER = QColor(80, 140, 80)
    BUTTON_READ_TEXT = QColor(80, 140, 80)   # green ink; hover inverts

    # Checkbox — square ink mark from start_gui
    CHECK_BG = QColor(250, 247, 240)
    CHECK_BORDER = QColor(42, 42, 42, 80)
    CHECK_CHECKED_BG = QColor(26, 26, 26)
    CHECK_CHECKED_BORDER = QColor(26, 26, 26)

    # Status text colors — straight from start_gui
    STATUS_OK = QColor(80, 140, 80)          # GREEN_OK
    STATUS_ERROR = QColor(170, 50, 50)       # RED_INK
    STATUS_BUSY = QColor(212, 142, 60)       # AMBER


class _Structure:
    """Structural constants — same across palettes.

    Editorial defaults from start_gui.py's PAPER theme:
      - 2px corner radius (mostly flat). 6px on the old translucent
        nodes felt 'rounded card'; the paper aesthetic is 'editorial
        clipping' — hairline borders and very small radii.
      - IBM Plex font stack (Mono for code/labels, Sans for UI).
      - Node header height stays at 28px — that's still room for the
        accent dot + monospace title without crowding.
    """
    NODE_HEADER_HEIGHT = 22.0
    NODE_CORNER_RADIUS = 2.0
    NODE_MIN_WIDTH = 200.0
    # PORT_RADIUS: 3.5 read nicely as a connection nub but was too
    # small to reliably hit — the bounding hit rect is 2r × 2r, so 3.5
    # gave a ~7px target. Bumped to 6.0 for a ~12px target while still
    # reading as a nub rather than a full UI element. Combined with
    # the Z-order flip in PortItem (port painted *below* the body),
    # each port appears half-clipped by the node frame, so only the
    # outer hemisphere shows — the visible footprint is half this.
    PORT_RADIUS = 6.0
    # PORT_SPACING down from 22 → 18 so the input column doesn't waste
    # vertical real estate inside the body. PORT_MARGIN_TOP down from
    # 14 → 8 — with a 22px header the body starts higher and we no
    # longer need a tall gap before the first port.
    PORT_SPACING = 18.0
    PORT_MARGIN_TOP = 8.0

    # IBM Plex stack matches start_gui.py exactly so cross-window text
    # rendering looks identical. Iosevka / JetBrains kept as fallbacks
    # for users without Plex installed.
    FONT_FAMILY = (
        "'IBM Plex Mono', 'JetBrains Mono', 'Iosevka', "
        "'Consolas', 'Menlo', monospace"
    )
    FONT_FAMILY_MONO = FONT_FAMILY
    FONT_FAMILY_UI = (
        "'IBM Plex Sans', 'Inter', 'Segoe UI', "
        "'Helvetica Neue', Arial, sans-serif"
    )

    @property
    def FONT_NODE_TITLE(self) -> QFont:
        # Monospace, regular weight — matches the section labels in
        # start_gui ('output', 'workspace', etc.) and the picker.
        f = QFont(self.FONT_FAMILY_MONO, 9)
        f.setWeight(QFont.Medium)
        return f

    @property
    def FONT_PORT_LABEL(self) -> QFont:
        # Port labels go mono too — sans-mixed text is what makes node
        # editors look like Blender. Paper keeps it all in one family.
        return QFont(self.FONT_FAMILY_MONO, 8)

    @property
    def FONT_CODE(self) -> QFont:
        return QFont(self.FONT_FAMILY_MONO, 10)

    @property
    def FONT_CODE_SMALL(self) -> QFont:
        return QFont(self.FONT_FAMILY_MONO, 9)


class _ThemeProxy:
    """Routes attribute access to the active palette. Structural constants
    live in _Structure and don't change with mode.

    Three palettes:
      - 'paper' (default) — editorial off-white card, hairline border,
        no shadows. Matches start_gui.py.
      - 'dark'            — original dark mode.
      - 'light'           — original light mode.

    Use set_mode_named() for explicit selection. set_mode(dark=bool)
    is kept for back-compat — it flips between dark and light (paper
    is unreachable from the bool API on purpose: callers that want
    paper should ask for it by name).
    """

    def __init__(self):
        self._palette = _Paper()
        self._structure = _Structure()
        self._mode = "paper"

    def set_mode_named(self, name: str) -> None:
        name = name.lower()
        if name == "paper":
            self._palette = _Paper()
        elif name == "dark":
            self._palette = _Dark()
        elif name == "light":
            self._palette = _Light()
        else:
            raise ValueError(f"unknown theme mode: {name!r}")
        self._mode = name

    def set_mode(self, dark: bool) -> None:
        # Back-compat shim. Code that only thinks in dark/light bool
        # still works; paper is opt-in via set_mode_named.
        self.set_mode_named("dark" if dark else "light")

    @property
    def is_dark(self) -> bool:
        return self._mode == "dark"

    @property
    def mode(self) -> str:
        return self._mode

    def __getattr__(self, name: str):
        # __getattr__ is only called when normal lookup fails, so this
        # won't recurse on _palette/_structure access.
        if hasattr(self._palette, name):
            return getattr(self._palette, name)
        if hasattr(self._structure, name):
            return getattr(self._structure, name)
        raise AttributeError(name)


Theme = _ThemeProxy()


# ═══════════════════════════════════════════════════════════════════════
# ─── GRAPH (MODEL) ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Pure-model view of the node graph: who exists, what connects to what,
# which paths each port maps to. No Qt painting code — that's op_nodes.py.
# No filesystem syscalls — those go through pipe.py.
#
# This separation matters because the on-disk filesystem and the on-screen
# canvas are two views of the same graph. The model is the third view —
# the one that lets either side reflect the other without painting code
# having to understand routes-file syntax, or filesystem code having to
# understand QGraphicsItem.
#
# Three core types:
#   - `Port`       One file in a node directory.
#   - `Node`       One directory under /n/<m>/nodes/, /n/llm/agents/, etc.
#   - `Connection` A directed edge from one Port to another, backed by
#                  one line in /n/<m>/routes.
#
# Plus `Routes` — a subscription against the routes file turned into
# add/remove events the model dispatches to listeners.
#
# Path conventions (the operator never invents these):
#     /n/<machine>/nodes/<node_id>/<port>    operator-created nodes
#     /n/llm/agents/<name>/<port>            pre-existing agents
#     /n/<machine>/terms/<term_id>/<port>    pre-existing terminals
#     /n/<machine>/scene/<port>              pre-existing scene controller


class PortDirection(Enum):
    INPUT = "input"     # left side of node; consumes data
    OUTPUT = "output"   # right side of node; produces data

    @property
    def is_input(self) -> bool:
        return self is PortDirection.INPUT


class NodeKind(Enum):
    """What kind of widget to render for this node. The filesystem doesn't
    know about kinds — they're a hint to the view layer based on which
    directory the node lives in and what files are in it."""
    AGENT = "agent"          # /n/llm/agents/<name>
    TERMINAL = "terminal"    # /n/<m>/terms/<id>
    SCENE = "scene"          # /n/<m>/scene
    TEXT = "text"            # /n/<m>/nodes/<id> with text-area widget
    DEBUG = "debug"          # /n/<m>/nodes/<id> with N inputs, log view
    MEDIA = "media"          # /n/<m>/nodes/<id> with image/video preview
    BASH = "bash"            # /n/<m>/nodes/<id> running shell commands
    PYTHON = "python"        # /n/<m>/nodes/<id> running python expressions
    GENERIC = "generic"      # anything else with files we don't recognize


@dataclass
class Port:
    """One file inside a node directory. Carries the path and the list of
    Connections it participates in.

    Equality is identity-based because two ports against the same path
    on different node objects are distinct in the graph.
    """
    name: str
    direction: PortDirection
    path: str
    description: str = ""
    connections: List["Connection"] = field(default_factory=list)
    # Backref to the owning node, set by Node.add_port
    node: Optional["Node"] = field(default=None, repr=False)

    @property
    def is_blocking(self) -> bool:
        """Convenience: forward to Pipe's basename rule. Useful for UI
        hints (a small badge on streaming ports)."""
        from .pipe import _basename_is_uppercase
        return _basename_is_uppercase(self.path)

    @property
    def qualified_name(self) -> str:
        """For status messages and tooltips: 'node_id/port_name'."""
        if self.node is None:
            return self.name
        return f"{self.node.node_id}/{self.name}"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass
class Node:
    """One directory on the filesystem, plus its ports.

    `node_id` is the directory basename — unique within its parent
    (nodes/, agents/, terms/). `dir_path` is the absolute path. `kind`
    is a hint for the view layer.

    The model doesn't enforce a specific port layout; it accepts whatever
    `add_port` calls produce. Builders in each NodeView subclass populate
    the ports based on what files they expect.
    """
    node_id: str
    dir_path: str
    kind: NodeKind
    description: str = ""
    ports: List[Port] = field(default_factory=list)

    def add_port(self, port: Port) -> Port:
        port.node = self
        self.ports.append(port)
        return port

    def get_port(self, name: str) -> Optional[Port]:
        for p in self.ports:
            if p.name == name:
                return p
        return None

    @property
    def inputs(self) -> List[Port]:
        return [p for p in self.ports if p.direction.is_input]

    @property
    def outputs(self) -> List[Port]:
        return [p for p in self.ports if not p.direction.is_input]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass
class Connection:
    """A directed edge from one Port to another.

    Two flavors:

    1. Route-backed (`is_static=False`, the default). Mirrors one line in
       /n/<m>/routes:
           /n/llm/agents/master/OUTPUT -> /n/<m>/nodes/debug_0/in_0
       The server pumps the source's data into the target continuously.
       Created/removed in lock-step with the routes file: the model
       observes /n/<m>/routes and reconciles Connection objects to match.

    2. Static (`is_static=True`). Operator-side only — no entry in the
       routes file. Drawn as a visual link, and tells UI handlers (Read /
       Write / Auto buttons on a TextNode) which port they should resolve
       to when the user clicks them. Lives entirely in the in-memory
       graph; the routes-file reconciler MUST skip these (see
       Graph.apply_routes).

       Use a static link instead of a route when the destination is a
       command sink like `ctl` — routing into `ctl` makes the server
       receive a continuous stream of writes (the value, then the value
       again, then the value again…) which command parsers reject. A
       static link records the intent visually and defers actual I/O
       to user-driven button clicks.

    `running` is a hint from the server (routes file emits [running] or
    [stopped] tags); kept for tooltips and styling. Static links are
    always considered "running" because there's no upstream pump.
    """
    source: Port
    target: Port
    running: bool = True
    is_static: bool = False

    def __post_init__(self):
        # Wire the connection into both endpoints. Idempotent.
        if self not in self.source.connections:
            self.source.connections.append(self)
        if self not in self.target.connections:
            self.target.connections.append(self)

    def detach(self):
        """Remove from both endpoints' connection lists. Called by Graph
        when this connection is being removed."""
        if self in self.source.connections:
            self.source.connections.remove(self)
        if self in self.target.connections:
            self.target.connections.remove(self)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


# ─── Routes: canonical pipe registry ───────────────────────────────────


@dataclass
class _RouteLine:
    """One parsed line from the routes file."""
    source: str
    target: str
    running: bool


def _parse_routes(data: bytes) -> List[_RouteLine]:
    """Parse /n/<m>/routes content into RouteLine objects.

    Format (matches RoutesFile.read in filesystem.py):
        /path/to/source -> /path/to/destination [running]
        /path/to/source -> /path/to/destination [stopped]

    Empty file or '(no routes)' returns [].
    """
    text = data.decode("utf-8", errors="replace")
    lines: List[_RouteLine] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("(no routes"):
            continue

        # Status tag is optional but expected: "[running]" / "[stopped]"
        running = True
        if line.endswith("]"):
            lbracket = line.rfind("[")
            if lbracket > 0:
                tag = line[lbracket + 1: -1].strip().lower()
                running = (tag != "stopped")
                line = line[:lbracket].strip()

        if "->" not in line:
            continue
        src, _, dst = line.partition("->")
        src = src.strip()
        dst = dst.strip()
        if src and dst:
            lines.append(_RouteLine(source=src, target=dst, running=running))
    return lines


class Routes(QObject):
    """Bidirectional view over /n/<m>/routes.

    Uses **subprocess** (`cat` / `tee`) for all I/O, not Python's
    open()/os.read()/os.write(). The operator runs inside the parser's
    Python process, which owns a 9p client connection. Opening another
    file descriptor on the same mount from the same Python process
    causes 9p FID churn that crashes the riomux backend (symptom:
    `Backend 'ekanza' connect failed` firing the moment the operator
    is injected). Subprocess I/O uses the kernel mount via separate
    child processes, sidestepping the contention entirely.

    Push, not poll. After the server-side RoutesFile became
    blocking-on-rearm, `cat /n/<m>/routes` returns the current state
    once and then blocks on the *next* open until something changes.
    We spawn one `cat`, await its exit, parse the output, and
    immediately respawn — the new `cat`'s open blocks server-side
    until a route is added or removed.
    """

    routes_changed = Signal(list)   # list[(src, dst, running)]

    def __init__(self, mount: str, worker: FSWorker, **_kwargs):
        super().__init__()
        self._mount = mount
        self._worker = worker  # kept for API compat; not used for I/O
        self._routes_path = os.path.join(mount, "routes")
        self._current: List[Tuple[str, str, bool]] = []
        self._last_raw: Optional[bytes] = None
        self._reader_proc: Optional[QProcess] = None
        self._stopped = False

        # Kick off the first read on the next event-loop turn so
        # construction finishes without blocking.
        QTimer.singleShot(0, self._start_read)

    def current(self) -> List[Tuple[str, str, bool]]:
        return list(self._current)

    def add(self, source: str, target: str,
            on_done: Optional[Callable[[object], None]] = None) -> None:
        line = f"{source} -> {target}\n"
        self._write_via_subprocess(line, on_done)

    def remove(self, source: str,
               on_done: Optional[Callable[[object], None]] = None) -> None:
        line = f"-{source}\n"
        self._write_via_subprocess(line, on_done)

    def stop(self):
        self._stopped = True
        if self._reader_proc is not None:
            try:
                self._reader_proc.kill()
            except Exception:
                pass
            self._reader_proc = None

    # ── internal: blocking read via QProcess ─────────────────────────
    #
    # The server's RoutesFile blocks reads at offset 0 once previous
    # content was consumed. `cat <path>` does one open + read-to-EOF +
    # close, so each cat returns once: current state when there are
    # changes pending, or after blocking on the server until there are.

    def _start_read(self) -> None:
        if self._stopped:
            return
        if self._reader_proc is not None and \
                self._reader_proc.state() != QProcess.NotRunning:
            return
        proc = QProcess(self)
        self._reader_proc = proc
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.finished.connect(
            lambda _code, _status, p=proc: self._on_read_done(p))
        proc.start("cat", [self._routes_path])

    def _on_read_done(self, proc: "QProcess") -> None:
        if self._stopped:
            return
        if proc is not self._reader_proc:
            return
        data = bytes(proc.readAllStandardOutput())
        self._reader_proc = None
        # Schedule the next read immediately. It'll block server-side
        # until the next route change.
        QTimer.singleShot(0, self._start_read)
        if data == self._last_raw:
            return
        self._last_raw = data
        parsed = _parse_routes(data)
        new = [(r.source, r.target, r.running) for r in parsed]
        if new == self._current:
            return
        self._current = new
        self.routes_changed.emit(list(new))

    # ── internal: write via QProcess ─────────────────────────────────

    def _write_via_subprocess(self, line: str,
                              on_done: Optional[Callable[[object], None]]):
        proc = QProcess(self)
        # Keep a reference so it doesn't get GC'd mid-write.
        # Connect finished BEFORE start so we don't miss it.
        def _done(_code, _status, p=proc):
            try:
                if on_done is not None:
                    on_done(_code)
            except Exception as e:
                print(f"Routes: write callback raised {e}")
            p.deleteLater()
        proc.finished.connect(_done)
        # `tee -a <path>` appends stdin to the file.
        proc.start("tee", ["-a", self._routes_path])
        proc.write(line.encode("utf-8"))
        proc.closeWriteChannel()


class Graph(QObject):
    """The full model graph: a collection of Nodes and Connections.

    The graph is *populated* by:
      1. Filesystem scan: walk /n/llm/agents/, /n/<m>/terms/, etc. —
         one Node per directory.
      2. Routes diff: for each (src, dst) in the routes file, find or
         create a Connection between the matching ports.

    Both happen incrementally. Adding a Node emits `node_added`; the
    view layer listens and spawns the right NodeView. Same for
    connections and removals.

    The graph never makes up paths or invents ports — every Port has a
    real path on disk. View code that needs to write to a port goes
    through `Pipe(port.path, worker)` like anyone else.
    """

    node_added = Signal(object)         # Node
    node_removed = Signal(object)       # Node
    connection_added = Signal(object)   # Connection
    connection_removed = Signal(object)  # Connection

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._nodes_by_id: Dict[str, Node] = {}
        self._connections: List[Connection] = []

    # ── Nodes ────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> Node:
        if node.node_id in self._nodes_by_id:
            return self._nodes_by_id[node.node_id]
        self._nodes_by_id[node.node_id] = node
        self.node_added.emit(node)
        return node

    def remove_node(self, node_id: str) -> Optional[Node]:
        node = self._nodes_by_id.pop(node_id, None)
        if node is None:
            return None
        # Drop any connections touching this node.
        for conn in list(self._connections):
            if conn.source.node is node or conn.target.node is node:
                self._remove_connection(conn)
        self.node_removed.emit(node)
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes_by_id.get(node_id)

    def nodes(self) -> List[Node]:
        return list(self._nodes_by_id.values())

    def find_port_by_path(self, path: str) -> Optional[Port]:
        for node in self._nodes_by_id.values():
            for port in node.ports:
                if port.path == path:
                    return port
        return None

    # ── Connections ──────────────────────────────────────────────────

    def add_connection(self, source: Port, target: Port,
                       running: bool = True,
                       is_static: bool = False) -> Optional[Connection]:
        """Create a Connection between two existing ports. Idempotent —
        returns the existing connection if one's already there.

        `is_static=True` marks the link as operator-side only (no entry
        in /n/<m>/routes); see the Connection docstring for the model.
        """
        for conn in self._connections:
            if conn.source is source and conn.target is target:
                # Update running state if it changed. We deliberately do
                # NOT flip is_static on an existing connection — a route
                # is a route, a static link is a static link, and the
                # caller should remove and re-add to change kind.
                if conn.running != running:
                    conn.running = running
                return conn
        conn = Connection(source=source, target=target, running=running,
                          is_static=is_static)
        self._connections.append(conn)
        self.connection_added.emit(conn)
        return conn

    def add_static_connection(self, source: Port,
                              target: Port) -> Optional[Connection]:
        """Convenience: add a static (visual-only) connection."""
        return self.add_connection(source, target, running=True,
                                   is_static=True)

    def remove_connection_by_paths(self, src_path: str,
                                   tgt_path: str) -> bool:
        for conn in list(self._connections):
            if (conn.source.path == src_path and
                    conn.target.path == tgt_path):
                self._remove_connection(conn)
                return True
        return False

    def _remove_connection(self, conn: Connection) -> None:
        conn.detach()
        if conn in self._connections:
            self._connections.remove(conn)
        self.connection_removed.emit(conn)

    def connections(self) -> List[Connection]:
        return list(self._connections)

    # ── Routes integration ───────────────────────────────────────────

    def apply_routes(self, routes: List[Tuple[str, str, bool]]) -> None:
        """Diff the routes list against current Connections; add/remove
        as needed. Called by the Operator in response to
        Routes.routes_changed.

        Routes pointing at ports we don't have (because the corresponding
        node directory hasn't been scanned yet) are silently skipped —
        they'll appear on the next scan-then-apply cycle.

        Static connections (is_static=True) are NOT touched by this
        reconciliation: they live entirely in the operator's in-memory
        graph and have no entry in /n/<m>/routes to compare against.
        Treating them like routes would delete them on every routes
        update.
        """
        want: List[Tuple[Port, Port, bool]] = []
        for src_path, tgt_path, running in routes:
            src = self.find_port_by_path(src_path)
            tgt = self.find_port_by_path(tgt_path)
            if src is None or tgt is None:
                continue
            want.append((src, tgt, running))

        want_set = {(s, t) for (s, t, _) in want}

        # Route-backed connections only — static links are not in routes.
        route_conns = [c for c in self._connections if not c.is_static]
        have_set = {(c.source, c.target) for c in route_conns}

        # Remove route-backed connections that aren't in routes anymore.
        for conn in list(route_conns):
            if (conn.source, conn.target) not in want_set:
                self._remove_connection(conn)

        # Add connections that are in routes but not yet in the graph.
        for src, tgt, running in want:
            if (src, tgt) not in have_set:
                self.add_connection(src, tgt, running=running)
            else:
                # Update running state in place. Only touches route-backed
                # connections (static ones aren't in have_set).
                for conn in route_conns:
                    if conn.source is src and conn.target is tgt:
                        conn.running = running
                        break


# ═══════════════════════════════════════════════════════════════════════
# ─── SCANNER ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Filesystem walks via subprocess `ls -R`.
#
# Why subprocess and not Python's os.listdir: the operator is exec'd
# inside the parser's Python process, which already owns a 9p client
# connection (the one reading `/n/<m>/scene/parse`). Opening additional
# file descriptors against the same mount from the same process trips
# the rio backend — the symptom was `Backend 'ekanza' connect failed`
# firing before any scan even completes. Running `ls -R` in a separate
# process means the discovery uses its own 9p client connection (via
# the kernel's mount), which doesn't interfere.
#
# Output format from `ls -R -A -1 --indicator-style=slash <path>`:
#
#     /path/to/dir:
#     file1
#     subdir1/
#     file2
#
#     /path/to/dir/subdir1:
#     nested_file
#
# Empty line between sections; trailing slash on directories.
#
# Layout we walk:
#     /n/<workspace>/             workspace root → one node with ports
#       ctl, routes, ...
#       scene/                    → NodeKind.SCENE
#       terms/<term_id>/          → NodeKind.TERMINAL
#       nodes/<id>/               → NodeKind.{TEXT,DEBUG,...}
#
#     /n/llm/                     llmfs root; agents at root
#       ctl, providers            reserved
#       <agent_name>/             → NodeKind.AGENT
#
# Debug log: set OPERATOR_DEBUG=1 to enable checkpoint output to
# /tmp/operator.log and stdout. Used by both the scanner and the
# Operator app below.


_T0 = time.monotonic()
_LOG_PATH = "/tmp/operator.log"
_DEBUG = bool(os.environ.get("OPERATOR_DEBUG", ""))


def _ck_scanner(label: str) -> None:
    if not _DEBUG:
        return
    elapsed = time.monotonic() - _T0
    wall = time.strftime("%H:%M:%S")
    line = f"[scanner {wall} +{elapsed:6.3f}s] {label}\n"
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        pass
    sys.stdout.write(line)
    sys.stdout.flush()


# ─── Port direction classification ─────────────────────────────────────

INPUT_PORT_NAMES: Set[str] = {
    "ctl", "input", "stdin", "cmd", "code", "system", "rules", "config",
    "in", "parse",
    "mask", "model", "backend", "meta", "help",   # llmfs image/video
    "interrupt", "inline",                         # rio terminal
}

OUTPUT_PORT_NAMES: Set[str] = {
    "output", "stdout", "stderr", "out", "err",
    "history", "vars", "version", "status", "screen",
    "state", "context",
    "errors", "image",                             # llmfs
    "routes",                                      # rio workspace root
}


def _classify_port(filename: str) -> PortDirection:
    base = filename.lower().lstrip(".")
    if base in INPUT_PORT_NAMES:
        return PortDirection.INPUT
    if base in OUTPUT_PORT_NAMES:
        return PortDirection.OUTPUT
    letters = [c for c in filename if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return PortDirection.OUTPUT
    return PortDirection.OUTPUT


# ─── LLMFS agent identification ────────────────────────────────────────

_LLMFS_ROOT_RESERVED = {"ctl", "providers"}
AGENT_REQUIRED_FILES = {"ctl", "input", "OUTPUT"}


def _looks_like_agent(files: Set[str]) -> bool:
    return AGENT_REQUIRED_FILES.issubset(files)


# ─── User-node kind inference ──────────────────────────────────────────

_KIND_BY_PREFIX = {
    "text": NodeKind.TEXT,
    "debug": NodeKind.DEBUG,
    "media": NodeKind.MEDIA,
    "bash": NodeKind.BASH,
    "python": NodeKind.PYTHON,
}


def _infer_node_kind(dir_name: str) -> NodeKind:
    prefix = dir_name.split("_", 1)[0].lower()
    return _KIND_BY_PREFIX.get(prefix, NodeKind.GENERIC)


# ─── ls -R subprocess + parser ─────────────────────────────────────────


def _run_ls_recursive(path: str, timeout: float = 5.0) -> str:
    """Run `ls -R -A -1 --indicator-style=slash <path>` and return its
    stdout. Empty string on failure."""
    _start = time.monotonic()
    _ck_scanner(f"_run_ls_recursive({path!r}) START")
    try:
        result = subprocess.run(
            ["ls", "-R", "-A", "-1", "--indicator-style=slash", path],
            capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        _dt = time.monotonic() - _start
        _ck_scanner(
            f"_run_ls_recursive({path!r}) END rc={result.returncode} "
            f"bytes={len(result.stdout)} after {_dt:.3f}s")
        if result.stderr:
            _ck_scanner(f"  stderr: {result.stderr[:200]!r}")
        if result.returncode != 0:
            return ""
        return result.stdout
    except subprocess.TimeoutExpired:
        _dt = time.monotonic() - _start
        _ck_scanner(f"_run_ls_recursive({path!r}) TIMEOUT after {_dt:.3f}s")
        return ""
    except (FileNotFoundError, OSError) as e:
        _dt = time.monotonic() - _start
        _ck_scanner(f"_run_ls_recursive({path!r}) OSError {e} after {_dt:.3f}s")
        return ""


def _parse_ls_output(output: str,
                     root: str) -> Dict[str, Tuple[List[str], List[str]]]:
    """Parse `ls -R` output into {dir_abspath: (files, subdirs)}.

    `ls -R` prints sections like:
        <path>:
        entry1
        entry2/

    The first section's header is `<root>:`. Subsequent sections show
    the path relative to it (or absolute, depending on ls version — we
    handle both).

    Returns paths as absolute, normalized.
    """
    result: Dict[str, Tuple[List[str], List[str]]] = {}
    if not output.strip():
        return result

    current_dir: Optional[str] = None
    current_files: List[str] = []
    current_subdirs: List[str] = []

    def flush():
        nonlocal current_dir, current_files, current_subdirs
        if current_dir is not None:
            result[current_dir] = (sorted(current_files),
                                   sorted(current_subdirs))
        current_files = []
        current_subdirs = []

    for raw in output.splitlines():
        line = raw.rstrip()
        if not line:
            # Blank line: separator between sections. Section headers
            # come on the *next* non-blank line.
            continue
        if line.endswith(":"):
            # New section header
            flush()
            current_dir = os.path.normpath(line[:-1])
            continue
        if current_dir is None:
            # Defensive: output without a header. Skip.
            continue
        if line.endswith("/"):
            current_subdirs.append(line[:-1])
        else:
            current_files.append(line)
    flush()
    return result


def _make_node(dir_path: str, kind: NodeKind, files: List[str], *,
               node_id: Optional[str] = None,
               description: str = "") -> Node:
    nid = node_id or os.path.basename(dir_path.rstrip("/")) or "root"
    node = Node(node_id=nid, dir_path=dir_path, kind=kind,
                description=description)
    for fname in files:
        node.add_port(Port(
            name=fname,
            direction=_classify_port(fname),
            path=os.path.join(dir_path, fname),
            description=f"{kind.value}/{fname}",
        ))
    return node


def _scan_subdir_individually(
        path: str) -> Optional[Tuple[List[str], List[str]]]:
    """Fallback for when `ls -R` skipped a subdirectory. Runs a plain
    `ls -A -1 --indicator-style=slash` on the path and parses files
    vs subdirs. Returns (files, subdirs) or None on error."""
    _t0 = time.monotonic()
    _ck_scanner(f"_scan_subdir_individually({path!r}) START")
    try:
        result = subprocess.run(
            ["ls", "-A", "-1", "--indicator-style=slash", path],
            capture_output=True, text=True, timeout=5.0, check=False,
        )
        _dt = time.monotonic() - _t0
        _ck_scanner(
            f"_scan_subdir_individually({path!r}) END "
            f"rc={result.returncode} in {_dt:.3f}s, "
            f"{len(result.stdout)} bytes")
        if result.returncode != 0:
            if result.stderr:
                _ck_scanner(f"  stderr: {result.stderr[:200]!r}")
            return None
        files: List[str] = []
        subdirs: List[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.endswith("/"):
                subdirs.append(line.rstrip("/"))
            else:
                files.append(line)
        return (sorted(files), sorted(subdirs))
    except subprocess.TimeoutExpired:
        _ck_scanner(f"_scan_subdir_individually({path!r}) TIMEOUT")
        return None
    except OSError as e:
        _ck_scanner(f"_scan_subdir_individually({path!r}) OSError {e}")
        return None


# ─── Universal node detection ──────────────────────────────────────────
#
# The filesystem doesn't tell us what's a node. We infer it from shape.
# A directory becomes a Node if it has files (= ports). A directory with
# only subdirs is a container — we descend into it but don't make a node
# for the container itself.
#
# Kind inference is a layered set of rules, most specific first. None of
# them are name-based against a specific filesystem (llm/peribus/etc) —
# they look at structural cues (which files are present, what the parent
# directory is called).


def _infer_kind_universal(dir_path: str, files: Set[str],
                          parent_basename: str) -> NodeKind:
    """Pick a NodeKind for a directory based on its shape and context.

    Order of rules matters: most specific wins.
    """
    basename = os.path.basename(dir_path.rstrip("/"))

    # Looks like an llmfs-style agent: ctl + input + OUTPUT all present.
    if _looks_like_agent(files):
        return NodeKind.AGENT

    # Parent is `terms/` → this is a terminal session.
    if parent_basename == "terms":
        return NodeKind.TERMINAL

    # Basename `scene` (anywhere in the tree) → scene parser.
    if basename == "scene":
        return NodeKind.SCENE

    # Operator-created nodes use a `<kind>_<id>` naming convention.
    prefix_kind = _infer_node_kind(basename)
    if prefix_kind is not NodeKind.GENERIC:
        return prefix_kind

    return NodeKind.GENERIC


# Files that exist at filesystem roots purely as control surfaces. We
# DON'T want to make them ports on a top-level "root" node — they're
# administrative, not data. (They still get scanned through if they're
# subdirs; this only suppresses port-creation at the very top of a tree.)
_ROOT_LEVEL_CTL_ONLY = {"ctl"}


def scan_subtree(root: str, *,
                 max_depth: int = 3) -> List[Node]:
    """Walk a single subtree under /n and produce Nodes from every
    directory that has files.

    One `ls -R` subprocess per subtree. Per-dir fallback if a section is
    missing from the recursive output.

    `max_depth` caps recursion. 3 is enough for everything observed:
      0: /n/<workspace>                           (root)
      1: /n/<workspace>/nodes, /n/<workspace>/terms, ...
      2: /n/<workspace>/nodes/<id>, .../terms/<id>, ...
      3: room for one more level if a filesystem nests deeper.

    Containers (dirs with subdirs but no files) don't get a node of
    their own — they're just routed through. Leaves with files do.
    """
    output = _run_ls_recursive(root)
    listing = _parse_ls_output(output, root) if output else {}
    _ck_scanner(f"  ls -R {root}: parsed {len(listing)} sections")

    root_norm = os.path.normpath(root)
    nodes: List[Node] = []

    def get_or_fetch(path: str) -> Optional[Tuple[List[str], List[str]]]:
        entry = listing.get(path)
        if entry is not None:
            return entry
        _ck_scanner(f"  section missing for {path}, fetching directly")
        return _scan_subdir_individually(path)

    def visit(path: str, depth: int, parent_basename: str) -> None:
        if depth > max_depth:
            return
        entry = get_or_fetch(path)
        if entry is None:
            return
        files, subdirs = entry

        # Make a node if this directory has files. The root of a
        # subtree is allowed to be a "control-only" directory with
        # just `ctl` — in that case we skip node-creation here and
        # let the children speak. Below the root, every dir with
        # files is fair game.
        is_root = (depth == 0)
        meaningful_files = (
            [f for f in files if f not in _ROOT_LEVEL_CTL_ONLY]
            if is_root else files
        )
        if meaningful_files:
            kind = _infer_kind_universal(
                path, set(files), parent_basename)
            node_basename = os.path.basename(path.rstrip("/"))
            node_id = node_basename or "root"
            # Top-level roots get a more descriptive id so two
            # filesystems with the same basename layout don't collide.
            if is_root:
                node_id = node_basename or "root"
                description = f"{node_id} root"
            else:
                description = f"{kind.value} {node_basename}"
            nodes.append(_make_node(
                path, kind, files,
                node_id=node_id, description=description))

        # Recurse into subdirs regardless of whether this dir was a
        # node — terms/ and nodes/ are containers we still need to
        # walk into.
        my_basename = os.path.basename(path.rstrip("/"))
        for sub in subdirs:
            if sub.startswith("."):
                continue
            visit(os.path.join(path, sub), depth + 1, my_basename)

    parent_of_root = os.path.basename(os.path.dirname(root_norm))
    visit(root_norm, 0, parent_of_root)
    return nodes


def scan_n(base: str = "/n",
           extra_roots: Optional[List[str]] = None) -> List[Node]:
    """Walk every top-level directory under `base` and produce nodes
    for everything we find. This is the universal scan: no filesystem
    is special, no name is excluded.

    `extra_roots` lets the caller force specific roots in even if they
    don't appear under `base` (defensive — for unusual mount layouts).
    """
    _ck_scanner(f"scan_n(base={base!r}) START")
    nodes: List[Node] = []
    seen: Set[str] = set()

    # Enumerate top-level subdirs of /n. We DON'T recurse here —
    # that's scan_subtree's job. We just collect the roots.
    try:
        result = subprocess.run(
            ["ls", "-A", "-1", "--indicator-style=slash", base],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
        if result.returncode != 0:
            _ck_scanner(f"  ls {base} rc={result.returncode}")
    except (subprocess.TimeoutExpired, OSError) as e:
        _ck_scanner(f"  ls {base} failed: {e}")
        result = None

    roots: List[str] = []
    if result is not None and result.returncode == 0:
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.endswith("/"):
                continue
            name = line.rstrip("/")
            if name.startswith("."):
                continue
            roots.append(os.path.join(base, name))

    for r in (extra_roots or []):
        if r not in roots:
            roots.append(r)

    _ck_scanner(f"  roots = {roots}")

    for root in roots:
        norm = os.path.normpath(root)
        if norm in seen:
            continue
        seen.add(norm)
        try:
            nodes.extend(scan_subtree(norm))
        except Exception as exc:
            # Don't let one bad mount kill the whole scan.
            _ck_scanner(f"  scan_subtree({norm}) FAILED: {exc}")
            continue

    _ck_scanner(f"scan_n END: {len(nodes)} nodes total")
    return nodes


# ─── Back-compat shims ─────────────────────────────────────────────────
#
# Older callers used scan_all(rio_mount, llm_mount). The new universal
# scan walks all of /n, so the mount arguments become hints rather than
# limits. Kept for source compatibility; the workspace mount is still
# needed elsewhere (route management, nodes/ctl writes).


def scan_rio_workspace(rio_mount: str) -> List[Node]:
    """Back-compat: scan just one workspace subtree."""
    return scan_subtree(rio_mount)


def scan_llmfs(llm_mount: str) -> List[Node]:
    """Back-compat: scan just the llmfs subtree."""
    return scan_subtree(llm_mount)


def scan_all(rio_mount: str, llm_mount: str) -> List[Node]:
    """Universal scan of /n. The mount arguments are kept for API
    compatibility — they're still used as hints to make sure both
    known mounts show up even if `ls /n` raced or filtered them — but
    every other subdir of /n is also walked and surfaced as nodes."""
    base = "/n"
    # Try to infer the base from one of the supplied mounts (handles
    # unusual layouts where /n isn't the parent).
    for hint in (rio_mount, llm_mount):
        if hint:
            parent = os.path.dirname(os.path.normpath(hint))
            if parent and parent != os.path.normpath(hint):
                base = parent
                break
    extras = [m for m in (rio_mount, llm_mount) if m]
    return scan_n(base=base, extra_roots=extras)


def discover_rio_workspace(base: str = "/n") -> Optional[str]:
    """Find the rio workspace directory under `base`."""
    _t0 = time.monotonic()
    _ck_scanner(f"discover_rio_workspace({base!r}) START")
    try:
        result = subprocess.run(
            ["ls", "-A", "-1", "--indicator-style=slash", base],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
        _dt = time.monotonic() - _t0
        _ck_scanner(f"  ls({base}) rc={result.returncode} in {_dt:.3f}s, "
                    f"{len(result.stdout)} bytes")
        _ck_scanner(f"  ls stdout = {result.stdout!r}")
        if result.returncode != 0:
            return None
    except subprocess.TimeoutExpired:
        _ck_scanner(f"ls({base}) TIMEOUT")
        return None
    except (FileNotFoundError, OSError) as e:
        _ck_scanner(f"ls({base}) OSError {e}")
        return None

    candidates: List[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line.endswith("/"):
            continue
        name = line.rstrip("/")
        if name.startswith("."):
            continue
        candidates.append(name)
    _ck_scanner(f"  candidates = {candidates}")

    for name in candidates:
        path = os.path.join(base, name)
        _t1 = time.monotonic()
        try:
            sub = subprocess.run(
                ["ls", "-A", "-1", path],
                capture_output=True, text=True, timeout=3.0, check=False,
            )
            _dt = time.monotonic() - _t1
            _ck_scanner(f"  ls({path}) rc={sub.returncode} in {_dt:.3f}s")
            if sub.returncode != 0:
                continue
            entries = set(sub.stdout.split())
            if "routes" in entries:
                _ck_scanner(f"  FOUND routes in {path!r}")
                return path
        except subprocess.TimeoutExpired:
            _dt = time.monotonic() - _t1
            _ck_scanner(f"  ls({path}) TIMEOUT after {_dt:.3f}s")
            continue
        except OSError as e:
            _ck_scanner(f"  ls({path}) OSError {e}")
            continue
    _ck_scanner("discover_rio_workspace: no match")
    return None


# ═══════════════════════════════════════════════════════════════════════
# ─── CANVAS CHROME ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Scene-level interaction:
#   - `SceneEventFilter`: Qt event filter on the scene; routes mouse
#     moves and releases (temp connection during a port drag) to the
#     Operator. There is no context menu — all node creation goes
#     through the toolbar; routing is by mouse-button (see below).
#   - `Toolbar`: floating toolbar at the top of the canvas with
#     create-node and refresh buttons.
#   - `auto_layout`: deterministic placement (group by kind, columns).
#
# The drag/drop state machine itself lives on the Operator class
# (start_port_drag / finish_port_drag). Two modes:
#   - Right-button drag: persistent route (written to /n/<m>/routes).
#   - Left-button drag: one-shot transfer between ports, or — if
#     released on empty canvas — spawn a TextNode wired bidirectionally
#     to the source port (Read/Write buttons act on those routes).
# Port classification (INPUT/OUTPUT) drives layout and label casing
# but does NOT gate link creation. Any pair of ports can be linked;
# the filesystem decides whether the route is meaningful.


class Toolbar(QGraphicsProxyWidget):
    """Floating toolbar at the top of the canvas. Holds buttons for
    creating user nodes and refreshing the scan.

    A QGraphicsProxyWidget so it sits inside the QGraphicsScene and
    moves with the canvas's coordinate system. Position is set by the
    Operator after construction (top-left of the visible region).
    """

    def __init__(self, operator: "Operator"):
        super().__init__()
        self.operator = operator

        container = QWidget()
        # Paper-styled floating toolbar: paper card with hairline border,
        # 2px editorial corners, monospace labels — matches the node
        # cards on the canvas so the toolbar reads as another paper
        # element rather than a chrome panel.
        container.setStyleSheet(f"""
            QWidget {{
                background-color: {Theme.NODE_BG.name()};
                border: 1px solid {Theme.NODE_BORDER.name(QColor.HexArgb)};
                border-radius: 2px;
            }}
            QPushButton {{
                background-color: {Theme.BUTTON_BG.name(QColor.HexArgb)};
                color: {Theme.BUTTON_TEXT.name()};
                border: 1px solid {Theme.BUTTON_BORDER.name(QColor.HexArgb)};
                border-radius: 2px;
                padding: 4px 10px;
                font-family: {Theme.FONT_FAMILY_MONO};
                font-size: 11px;
                min-width: 40px;
            }}
            QPushButton:hover {{
                background-color: {Theme.BUTTON_BG_HOVER.name(QColor.HexArgb)};
                border: 1px solid {Theme.BUTTON_BORDER_HOVER.name(QColor.HexArgb)};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BUTTON_BG_PRESSED.name(QColor.HexArgb)};
            }}
        """)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        def add_btn(label, tooltip, slot):
            b = QPushButton(label)
            b.setToolTip(tooltip)
            b.clicked.connect(slot)
            layout.addWidget(b)
            return b

        # Captured scene position used as a default for new-node placement.
        # If the user is just clicking the toolbar, drop new nodes near
        # the toolbar itself rather than at (0,0).
        self._default_drop = QPointF(80, 80)

        add_btn("+ Text",   "Create a text node",
                lambda: self._create(NodeKind.TEXT))
        add_btn("+ Debug",  "Create a debug log node",
                lambda: self._create(NodeKind.DEBUG))
        add_btn("+ Media",  "Create a media preview node",
                lambda: self._create(NodeKind.MEDIA))
        add_btn("+ Bash",   "Create a bash command node",
                lambda: self._create(NodeKind.BASH))
        add_btn("+ Python", "Create a python expression node",
                lambda: self._create(NodeKind.PYTHON))
        add_btn("⟳ Refresh", "Re-scan the filesystem", operator.refresh)

        self.setWidget(container)
        # High z so it stays above nodes and connections.
        self.setZValue(1000)

    def _create(self, kind: NodeKind) -> None:
        # Drop position: middle of the visible region, with a small
        # offset for each successive create so they don't stack.
        region = self.operator.region()
        center = QPointF(
            region.left() + region.width() * 0.4,
            region.top() + region.height() * 0.3,
        )
        self.operator.create_user_node(kind, center)


class SceneEventFilter(QObject):
    """Installed on the QGraphicsScene by the Operator. Forwards relevant
    events to the operator's handlers."""

    def __init__(self, operator: "Operator"):
        super().__init__()
        self.operator = operator

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        et = event.type()
        if et == QEvent.GraphicsSceneMouseMove:
            if self.operator.is_dragging_connection:
                self.operator.update_drag(event.scenePos())
                return False
            return False
        if et == QEvent.GraphicsSceneMouseRelease:
            if self.operator.is_dragging_connection:
                # Mouse release outside any port cancels the drag. The
                # release handler resolves the drop (port hit, empty
                # canvas, or same-node) based on the recorded drag mode
                # (left = one-shot/spawn, right = route).
                self.operator.finish_port_drag(event.scenePos())
                return False
            return False
        if et == QEvent.KeyPress:
            # Delete/Backspace removes selected connections (existing
            # behavior). DEL (only) additionally removes selected nodes
            # by writing "delete <id>" to <workspace>/nodes/ctl — the
            # next scan diff reconciles the model and view. Backspace
            # does NOT delete nodes: Backspace is the universal "I'm
            # editing text" key, and we don't want stray presses to
            # nuke a node when the user thought they were typing.
            # Nodes outside <rio_mount>/nodes/ (agents, terms, peribus,
            # etc.) aren't server-deletable through nodes/ctl and are
            # filtered out by delete_selected_nodes with a status log.
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                handled = self.operator.delete_selected_connections()
                if event.key() == Qt.Key_Delete:
                    if self.operator.delete_selected_nodes():
                        handled = True
                if handled:
                    event.accept()
                    return True
        return False


def auto_layout(nodes: List[Node], region: QRectF,
                node_size_hint: float = 280.0) -> dict:
    """Return a dict[node_id -> QPointF] placing each node.

    Layout: group by kind, lay each group out as a column, columns left
    to right in this order: agents, terminals, user nodes (text/debug/
    media/bash/python), scene, generic.
    """
    column_order = [
        [NodeKind.AGENT],
        [NodeKind.TERMINAL],
        [NodeKind.TEXT, NodeKind.DEBUG, NodeKind.MEDIA,
         NodeKind.BASH, NodeKind.PYTHON],
        [NodeKind.SCENE],
        [NodeKind.GENERIC],
    ]
    by_kind: dict = {k: [] for col in column_order for k in col}
    for n in nodes:
        if n.kind in by_kind:
            by_kind[n.kind].append(n)

    positions: dict = {}
    col_x = region.left() + 40.0
    col_gap = node_size_hint + 60.0
    row_gap = 200.0
    # Reserve top 70px for the toolbar.
    top_y = region.top() + 80.0

    for col in column_order:
        col_nodes: List[Node] = []
        for kind in col:
            col_nodes.extend(by_kind[kind])
        if not col_nodes:
            continue
        y = top_y
        for n in col_nodes:
            positions[n.node_id] = QPointF(col_x, y)
            y += row_gap
        col_x += col_gap
    return positions


# ═══════════════════════════════════════════════════════════════════════
# ─── OPERATOR APP ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Top-level glue. Holds:
#   - `FSWorker` (shared by every Pipe in the operator)
#   - `Graph` (the model)
#   - `Routes` (subscribed view of /n/<m>/routes; attached only after
#     a successful scan)
#   - The QGraphicsScene we're drawing into (injected by the parser)
#   - A `SceneEventFilter` for mouse handling
#   - Maps from model objects to view items so model events update the
#     scene
#
# Filesystem discovery uses `subprocess` (see scanner section above).
# This is deliberate: when the operator is exec'd inside the parser,
# the host Python already owns a 9p client connection (the one reading
# /n/<m>/scene/parse). Opening additional file descriptors against the
# same mount from the same process tripped the rio backend — symptom
# was `Backend 'ekanza' connect failed` firing before any scan
# completed. Running `ls` in a subprocess uses the kernel's mount via
# a separate 9p client, sidestepping the issue.
#
# Lifecycle:
#   1. Construct: takes the scene, optional mount overrides, region.
#      No I/O against the mounts yet.
#   2. `_initial_scan` (deferred 1s): runs `ls -R` per mount on the
#      FSWorker. Builds nodes and adds them to the graph.
#   3. `_attach_routes` (deferred 50ms after scan): subscribes to the
#      routes file. Routes-driven Connections appear/disappear from now on.
#   4. User interaction: clicking a port starts a drag (handled by
#      PortItem → start_port_drag). Releasing on another port writes a
#      route line, which the subscription picks up and turns into a
#      visible Connection.
#   5. Cleanup: stop subscriptions, let the parser tear down items.


def CK(label: str) -> None:
    """Log a debug checkpoint. No-op unless OPERATOR_DEBUG is set."""
    if not _DEBUG:
        return
    elapsed = time.monotonic() - _T0
    wall = time.strftime("%H:%M:%S")
    line = f"[op {wall} +{elapsed:6.3f}s] {label}\n"
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        pass
    sys.stdout.write(line)
    sys.stdout.flush()


# Default candidates for the LLMFS mount, in order of preference.
LLMFS_MOUNT_CANDIDATES = ["/n/mux/llm", "/n/llm"]


def _check_dir_via_ls(path: str) -> bool:
    """Return True if `path` is a readable directory. Uses subprocess
    `ls` to avoid creating a 9p file descriptor inside the parser's
    host Python."""
    CK(f"  _check_dir_via_ls({path!r})  →  spawning `ls -d {path}`")
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["ls", "-d", path],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
        dt = time.monotonic() - t0
        ok = result.returncode == 0
        CK(f"  _check_dir_via_ls({path!r})  ←  ok={ok} in {dt:.3f}s")
        return ok
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        CK(f"  _check_dir_via_ls({path!r})  ←  TIMEOUT after {dt:.3f}s")
        return False
    except OSError as e:
        dt = time.monotonic() - t0
        CK(f"  _check_dir_via_ls({path!r})  ←  OSError {e} after {dt:.3f}s")
        return False


def _resolve_llm_mount(override: Optional[str]) -> Tuple[str, bool]:
    """Resolve the LLMFS mount WITHOUT probing.

    Synchronous `ls` on 9p mounts during the parser's exec window hangs
    the rio backend. We use the first candidate optimistically and let
    the actual scan fail-gracefully if it's wrong.
    """
    CK(f"_resolve_llm_mount(override={override!r})")
    if override:
        return override, True
    # /n/mux/llm is the conventional path when riomux is in play;
    # /n/llm is the standalone case. For auto we use /n/llm as default —
    # matches start_peribus.py's standalone mode AND its mux mode (the
    # mux mounts llm at /n/llm too — the riomux backend just serves
    # multiple workspaces from the same /n root).
    return "/n/llm", True


def _resolve_rio_mount(override: Optional[str]) -> Tuple[str, bool]:
    """Resolve the rio workspace mount — the one subdir of /n that
    holds `routes` and `nodes/ctl`. The universal scanner picks up
    every other subdir of /n on its own; this function exists only
    because route management and operator-created-node dispatch still
    target a specific workspace path.

    Strategy: no hardcoded name list. Prefer the user's login if
    present under /n, otherwise the first non-hidden subdir. We do
    NOT probe candidates with a second `ls` — that risked hanging the
    rio backend in earlier tests, and the universal scanner will
    catch mistakes (any subdir that *isn't* the workspace just gets
    scanned by scan_subtree instead).
    """
    CK(f"_resolve_rio_mount(override={override!r})")
    if override:
        return override, True

    CK("  enumerating /n via `ls -A -1 --indicator-style=slash /n`")
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["ls", "-A", "-1", "--indicator-style=slash", "/n"],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
        dt = time.monotonic() - t0
        CK(f"  ls /n rc={result.returncode} in {dt:.3f}s, "
           f"stdout={result.stdout!r}")
    except subprocess.TimeoutExpired:
        CK("  ls /n TIMEOUT")
        return _username_guess(), True
    except OSError as e:
        CK(f"  ls /n OSError {e}")
        return _username_guess(), True

    if result.returncode != 0:
        return _username_guess(), True

    candidates: List[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.endswith("/"):
            continue
        name = line.rstrip("/")
        if name.startswith("."):
            continue
        candidates.append(name)
    CK(f"  candidates = {candidates}")

    if not candidates:
        CK("  no subdirs under /n, falling back to username guess")
        return _username_guess(), True

    # Prefer login-name match (the most common case where the
    # workspace is per-user). Otherwise take the first subdir
    # alphabetically — deterministic and cheap.
    user = getpass.getuser()
    if user in candidates:
        chosen = user
    else:
        chosen = sorted(candidates)[0]
    CK(f"  chose workspace: {chosen}")
    return f"/n/{chosen}", True


def _username_guess() -> str:
    return f"/n/{getpass.getuser()}"


class Operator(QObject):
    """Top-level operator. Owns the model graph, the FSWorker, and the
    view-item lookup tables."""

    def __init__(self, scene: QGraphicsScene,
                 *, llm_mount: Optional[str] = None,
                 rio_mount: Optional[str] = None,
                 region: Optional[QRectF] = None,
                 dark: bool = True,
                 parent: QObject = None):
        CK("Operator.__init__ ENTER")
        super().__init__(parent)
        CK("  QObject super init done")
        Theme.set_mode(dark=dark)
        CK("  Theme set")

        self._scene = scene
        CK("  resolving llm mount")
        self.llm_mount, llm_ok = _resolve_llm_mount(llm_mount)
        CK(f"  llm_mount = {self.llm_mount} (ok={llm_ok})")
        CK("  resolving rio mount")
        self.rio_mount, rio_ok = _resolve_rio_mount(rio_mount)
        CK(f"  rio_mount = {self.rio_mount} (ok={rio_ok})")
        print(f"operator: llm_mount = {self.llm_mount} "
              f"({'ok' if llm_ok else 'MISSING'})")
        print(f"operator: rio_mount = {self.rio_mount} "
              f"({'ok' if rio_ok else 'MISSING'})")
        self._region = region or QRectF(0, 0, 1400, 900)

        # ── Core machinery ──────────────────────────────────────────
        CK("  constructing FSWorker")
        self.worker = FSWorker(self)
        CK("  constructing Graph")
        self.graph = Graph(self)
        # Routes attaches lazily after the first successful scan.
        self.routes: Optional[Routes] = None

        # ── View-item maps ──────────────────────────────────────────
        self._node_views: Dict[int, "NodeView"] = {}
        self._conn_views: Dict[int, "ConnectionItem"] = {}

        # ── Drag state ──────────────────────────────────────────────
        self._drag_source: Optional["PortItem"] = None
        self._temp_conn: Optional["TempConnectionItem"] = None
        # True for right-button drag (creates a route); False for
        # left-button drag (one-shot transfer or spawn-on-empty).
        self._drag_is_route: bool = False

        # ── Wire up model → view ────────────────────────────────────
        CK("  wiring graph signals")
        self.graph.node_added.connect(self._on_node_added)
        self.graph.node_removed.connect(self._on_node_removed)
        self.graph.connection_added.connect(self._on_connection_added)
        self.graph.connection_removed.connect(self._on_connection_removed)

        # ── Scene event filter ──────────────────────────────────────
        CK("  installing scene event filter")
        self._event_filter = SceneEventFilter(self)
        self._scene.installEventFilter(self._event_filter)

        # ── Toolbar ────────────────────────────────────────────────
        CK("  building toolbar")
        self._toolbar = Toolbar(self)
        # Place at top-left of visible region with a small inset.
        self._toolbar.setPos(self._region.left() + 16,
                             self._region.top() + 16)
        self._scene.addItem(self._toolbar)

        # ── Initial scan ────────────────────────────────────────────
        # 1-second delay before kicking off the scan: gives the rio
        # backend time to warm up. Empirically, the first `ls` of the
        # workspace mount after a fresh parser injection hangs for 5+
        # seconds (and times out the rio backend) if we hit it
        # immediately. Letting the parser session settle first reliably
        # avoids the death-spiral.
        CK("  scheduling _initial_scan via QTimer.singleShot(1000)")
        QTimer.singleShot(1000, self._initial_scan)
        CK("Operator.__init__ EXIT")

    # ── Public API ─────────────────────────────────────────────────────

    def region(self) -> QRectF:
        return self._region

    def refresh(self) -> None:
        """User-triggered rescan (right-click → Refresh). Runs in a
        worker thread because we're past the initial setup race."""
        self.worker.run_async(
            scan_all, self.rio_mount, self.llm_mount,
            on_done=self._on_scan_done,
        )

    def _initial_scan(self) -> None:
        """First scan, run ON THE FSWORKER (background thread). We can
        NOT run this synchronously on the Qt thread — `ls -R /n/<m>`
        has been observed to hang 5+ seconds on first call, during
        which the parser thread is blocked AND the rio backend times
        out. Running on a background thread keeps the Qt event loop
        free, so the parser's 9p session can finish/idle properly."""
        CK("_initial_scan ENTER — dispatching scan_all to FSWorker")
        print("operator: initial scan starting (background)…")
        self.worker.run_async(
            scan_all, self.rio_mount, self.llm_mount,
            on_done=self._on_scan_done_initial,
        )
        CK("_initial_scan EXIT — scan running in background")

    def _on_scan_done_initial(self, result) -> None:
        """Background-thread scan completed. Hand to the normal scan
        handler, then attach Routes."""
        CK(f"_on_scan_done_initial: result type = {type(result).__name__}")
        self._on_scan_done(result)
        CK("_on_scan_done_initial: scheduling _attach_routes in 50ms")
        QTimer.singleShot(50, self._attach_routes)

    def _attach_routes(self) -> None:
        CK("_attach_routes ENTER")
        if self.routes is not None:
            CK("_attach_routes: already attached, skipping")
            return
        routes_path = os.path.join(self.rio_mount, "routes")
        CK(f"_attach_routes: constructing Routes({self.rio_mount})")
        print(f"operator: attaching Routes subscription to {routes_path}")
        self.routes = Routes(self.rio_mount, self.worker)
        self.routes.routes_changed.connect(self.graph.apply_routes)
        CK("_attach_routes EXIT")

    def _on_scan_done(self, result) -> None:
        if isinstance(result, Exception):
            print(f"operator: scan failed: {result}")
            return
        nodes: List[Node] = result

        kinds_summary = {}
        for n in nodes:
            kinds_summary[n.kind.value] = kinds_summary.get(n.kind.value, 0) + 1
        summary = (", ".join(f"{v} {k}" for k, v in kinds_summary.items())
                   if kinds_summary else "0 nodes")
        print(f"operator: scan complete — {summary} "
              f"(rio={self.rio_mount}, llm={self.llm_mount})")
        for n in nodes:
            print(f"operator:   • {n.kind.value:9s} {n.node_id}  "
                  f"({len(n.inputs)}in / {len(n.outputs)}out)")

        # Diff
        existing_ids = set(self.graph._nodes_by_id.keys())
        new_ids = {n.node_id for n in nodes}

        for vanished in existing_ids - new_ids:
            self.graph.remove_node(vanished)

        positions = auto_layout(nodes, self._region)
        for node in nodes:
            if node.node_id in existing_ids:
                continue
            self.graph.add_node(node)
            view = self._node_views.get(id(node))
            if view is not None:
                pos = positions.get(node.node_id)
                if pos is not None:
                    view.setPos(pos)

        # Re-apply routes if subscription is up.
        if self.routes is not None:
            self.graph.apply_routes(self.routes.current())

    # ── Model → view dispatch ──────────────────────────────────────────

    def _on_node_added(self, node: Node) -> None:
        # Lazy import: op_nodes imports from us (Graph types, Theme),
        # so we can't import it at module load without a cycle.
        from .op_nodes import view_class_for
        cls = view_class_for(node.kind)
        view = cls(node, self)
        view.build_ports()
        view.build_body()
        view.layout()
        self._node_views[id(node)] = view
        self._scene.addItem(view)

    def _on_node_removed(self, node: Node) -> None:
        view = self._node_views.pop(id(node), None)
        if view is None:
            return
        view.cleanup()
        self._scene.removeItem(view)

    def _on_connection_added(self, conn: Connection) -> None:
        from .op_nodes import ConnectionItem
        if self.port_item_for(conn.source) is None:
            return
        if self.port_item_for(conn.target) is None:
            return
        item = ConnectionItem(conn, self)
        self._conn_views[id(conn)] = item
        self._scene.addItem(item)

    def _on_connection_removed(self, conn: Connection) -> None:
        item = self._conn_views.pop(id(conn), None)
        if item is None:
            return
        self._scene.removeItem(item)

    def notify_node_moved(self, node: Node) -> None:
        for port in node.ports:
            for conn in port.connections:
                item = self._conn_views.get(id(conn))
                if item is not None:
                    item.update_path()

    def port_item_for(self, port: Port) -> Optional["PortItem"]:
        if port.node is None:
            return None
        view = self._node_views.get(id(port.node))
        if view is None:
            return None
        return view.port_item(port)

    # ── Port-drag handling ─────────────────────────────────────────────

    @property
    def is_dragging_connection(self) -> bool:
        return self._drag_source is not None

    def start_port_drag(self, port_item: "PortItem",
                        is_route: bool = True) -> None:
        """Begin a drag from `port_item`.

        is_route=True  → right-button drag → completing creates a route
                         (persistent pipe written into /n/<m>/routes).
        is_route=False → left-button drag → completing does a one-shot
                         read+write transfer, or spawns a quick text node
                         if released on empty canvas.
        """
        from .op_nodes import TempConnectionItem
        self._drag_source = port_item
        self._drag_is_route = is_route
        self._temp_conn = TempConnectionItem(port_item, is_route=is_route)
        self._temp_conn.set_end(port_item.scene_center())
        self._scene.addItem(self._temp_conn)

    def update_drag(self, scene_pos: QPointF) -> None:
        if self._temp_conn is not None:
            self._temp_conn.set_end(scene_pos)

    def finish_port_drag(self, scene_pos: QPointF) -> None:
        """Resolve the drag started by `start_port_drag`.

        Universal: no input/input or output/output rejection. The user's
        drag direction is the orientation (src → tgt as drawn). Port
        classification (INPUT/OUTPUT) still drives layout and label case,
        but no longer gates link creation. The filesystem decides whether
        the route is meaningful.

        Mode (recorded at drag start):
          right-button → persistent route written to /n/<m>/routes.
          left-button  → static link: visual only, no routes file entry.
                         On empty canvas, spawns a TextNode and wires two
                         static links to it (in + OUT).
        """
        src_item = self._drag_source
        temp = self._temp_conn
        is_route = self._drag_is_route
        self._drag_source = None
        self._temp_conn = None
        self._drag_is_route = False
        if temp is not None:
            self._scene.removeItem(temp)
        if src_item is None:
            return

        tgt_item = self._find_port_at(scene_pos)

        # Empty canvas: only the left-button (static) path spawns a
        # text node. Right-button drag to empty space is a no-op — you
        # explicitly asked for a route somewhere, and there's nowhere.
        if tgt_item is None:
            if not is_route:
                self._spawn_text_node_from_port(src_item.port, scene_pos)
            return

        # Same port or same-node drop: ignore.
        if tgt_item is src_item:
            return
        if tgt_item.port.node is src_item.port.node:
            return

        src_port = src_item.port
        tgt_port = tgt_item.port

        if is_route:
            # Right-drag → persistent route. Orient as the user drew it.
            if self.routes is None:
                print("operator: cannot add route — Routes not attached")
                return
            self.routes.add(src_port.path, tgt_port.path)
        else:
            # Left-drag → static link. Visual only, no routes file entry.
            self.graph.add_static_connection(src_port, tgt_port)

    def _spawn_text_node_from_port(self, source_port: Port,
                                   scene_pos: QPointF) -> None:
        """Create a TextNode at `scene_pos` and wire it bidirectionally
        to `source_port`. Fires when the user left-drags from a port and
        releases on empty canvas.

        Two routes are written (source → new.in and new.OUT → source),
        so the text node mirrors the source's current value via its `in`
        leg and can push edits back via its `OUT` leg. The node's
        existing Read / Write / Auto buttons then act as labeled:
          Read   → re-pull the source through the routed `in` leg.
          Write  → push the text area to the source through `OUT`.
          Auto   → live subscription to `in` (same as for any text node).

        The node is created via the standard async path (writing
        `new text <id>` to /n/<m>/nodes/ctl, then rescanning). We fade
        the new view in via opacity animation to mask the latency, and
        auto-trigger Read once after the routes have flushed so the
        text area isn't blank on arrival.
        """
        # Pick a unique text_N name.
        existing_ids = set(self.graph._nodes_by_id.keys())
        idx = 0
        while f"text_{idx}" in existing_ids:
            idx += 1
        node_id = f"text_{idx}"

        # Anchor offset: nudge the node away from the cursor so its
        # relevant port sits near the drop point and the node body
        # doesn't land directly under the mouse.
        if source_port.direction.is_input:
            # Drag came from an input — put the new node's OUT side
            # near the drop point (node to the left).
            from .op_nodes import TextNodeView
            spawn_pos = QPointF(
                scene_pos.x() - TextNodeView.DEFAULT_WIDTH - 20,
                scene_pos.y() - 40)
        else:
            # Drag came from an output — put the new node's IN side
            # near the drop point (node to the right).
            spawn_pos = QPointF(scene_pos.x() + 20, scene_pos.y() - 40)

        # Dispatch the creation command, then settle the new view.
        nodes_ctl = os.path.join(self.rio_mount, "nodes", "ctl")
        ctl_pipe = Pipe(nodes_ctl, self.worker)
        cmd = f"new text {node_id}\n".encode("utf-8")

        def _after_write(result):
            if isinstance(result, Exception):
                print(f"operator: spawn-on-release failed: {result}")
                return
            # Give the server ~200ms to materialize the node, then rescan.
            QTimer.singleShot(
                200,
                lambda: self._finalize_quick_text_node(
                    node_id, spawn_pos, source_port))

        ctl_pipe.write_async(cmd, on_done=_after_write)

    def _finalize_quick_text_node(self, node_id: str, spawn_pos: QPointF,
                                  source_port: Port) -> None:
        """Rescan, place the new node at spawn_pos, fade it in, wire it
        to the source port with TWO STATIC LINKS, and trigger an initial
        Read so the text area shows the source value immediately.

        Two static (visual-only) connections are added:
          source.path        ─ new_node.in.path     (Read reads source)
          new_node.OUT.path  ─ source.path          (Write pushes source)

        Static, not routes: writing a route into a `ctl`-shaped port
        sends the data into the command parser, which (e.g. for image
        agents) rejects it as an unknown command. The static link is
        purely visual + intent-recording; the TextNodeView's Read /
        Write / Auto buttons follow the link at click time to find the
        port to actually read from or write to.
        """
        def after_scan(result):
            self._on_scan_done(result)
            node = self.graph.get_node(node_id)
            if node is None:
                print(f"operator: spawn-on-release: "
                      f"node {node_id} did not appear after rescan")
                return
            view = self._node_views.get(id(node))
            if view is None:
                return
            view.setPos(spawn_pos)
            self._fade_in(view)

            in_port = node.get_port("in")
            out_port = node.get_port("OUT")
            if in_port is None or out_port is None or not source_port.path:
                return

            # Two static links. Order: read leg first (source → in) so
            # that when we trigger Read below, the TextNodeView's
            # link-resolution finds source.path on its `in` port.
            self.graph.add_static_connection(source_port, in_port)
            self.graph.add_static_connection(out_port, source_port)

            # Trigger an initial Read so the text area is populated
            # immediately. No need to wait for a route to flush — the
            # static link resolution happens in-memory.
            self._initial_read_on_view(view)

        self.worker.run_async(
            scan_all, self.rio_mount, self.llm_mount,
            on_done=after_scan,
        )

    def _initial_read_on_view(self, view: "NodeView") -> None:
        """Programmatically click Read on a freshly-spawned TextNodeView.
        Safe no-op if the view doesn't have the handler (e.g. a future
        non-text node kind ends up here)."""
        handler = getattr(view, "_on_read_clicked", None)
        if callable(handler):
            try:
                handler()
            except Exception as e:
                print(f"operator: initial read failed: {e}")

    def _fade_in(self, view: "NodeView", duration_ms: int = 280) -> None:
        """Animate `view`'s opacity from 0 to 1. The animation is stashed
        on the view so it survives until completion.

        QGraphicsRectItem isn't a QObject, so we can't use
        QPropertyAnimation directly; QVariantAnimation drives setOpacity
        through a lambda.
        """
        view.setOpacity(0.0)
        anim = QVariantAnimation()
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.valueChanged.connect(
            lambda v, n=view: n.setOpacity(float(v)))
        anim.finished.connect(
            lambda n=view: (n.setOpacity(1.0),
                            setattr(n, "_spawn_fade_anim", None)))
        view._spawn_fade_anim = anim  # prevent GC
        anim.start()

    def _find_port_at(self, scene_pos: QPointF) -> Optional["PortItem"]:
        from .op_nodes import PortItem
        items = self._scene.items(scene_pos)
        for it in items:
            if isinstance(it, PortItem):
                return it
        return None

    def remove_connection(self, conn: Connection) -> None:
        """Remove a connection. For route-backed connections, writes the
        removal to /n/<m>/routes (which then propagates back through the
        routes subscription and removes the model object). For static
        connections, removes the model object directly."""
        if conn.is_static:
            self.graph._remove_connection(conn)
            return
        if self.routes is None:
            print("operator: cannot remove route — Routes not attached")
            return
        self.routes.remove(conn.source.path)

    def delete_selected_connections(self) -> bool:
        """Delete every selected ConnectionItem currently on the scene.
        Triggered by Delete/Backspace from the scene event filter.

        Static connections are removed directly from the graph (no
        routes-file write). Route-backed connections are removed by
        writing to /n/<m>/routes; the subsequent routes update reconciles
        the model. Returns True if at least one was deleted.
        """
        from .op_nodes import ConnectionItem
        selected = [it for it in self._scene.selectedItems()
                    if isinstance(it, ConnectionItem)]
        if not selected:
            return False
        for item in selected:
            conn = item.connection
            if conn.is_static:
                self.graph._remove_connection(conn)
            else:
                if self.routes is None:
                    continue
                self.routes.remove(conn.source.path)
        return True

    def delete_selected_nodes(self) -> bool:
        """Delete every selected NodeView currently on the scene.

        Mirrors create_user_node: writes `delete <id>\\n` to
        `<workspace>/nodes/ctl` for each selected node, then refreshes.
        The next scan diff in `_on_scan_done` will reconcile the model
        and view (the node disappears from the filesystem listing →
        graph.remove_node fires → view is dropped).

        Only nodes under `<rio_mount>/nodes/` are server-deletable —
        agents (llm), terminals, scene, and anything from other
        filesystems (peribus, etc.) are not. Those are filtered out and
        reported once via the status print. Returns True if at least
        one delete command was actually dispatched (so the key event
        gets accepted), False otherwise.
        """
        from .op_nodes import NodeView
        selected = [it for it in self._scene.selectedItems()
                    if isinstance(it, NodeView)]
        if not selected:
            return False

        nodes_root = os.path.normpath(
            os.path.join(self.rio_mount, "nodes")) + os.sep
        deletable: List[NodeView] = []
        non_deletable: List[NodeView] = []
        for item in selected:
            node = item.node
            norm = os.path.normpath(node.dir_path)
            if norm.startswith(nodes_root):
                deletable.append(item)
            else:
                non_deletable.append(item)

        if non_deletable:
            names = ", ".join(it.node.node_id for it in non_deletable)
            print(f"operator: cannot delete {len(non_deletable)} node(s) "
                  f"({names}) — only nodes under "
                  f"{self.rio_mount}/nodes/ are server-deletable")

        if not deletable:
            return False

        # Single aggregated confirmation for the whole selection.
        names = ", ".join(it.node.node_id for it in deletable)
        plural = "s" if len(deletable) > 1 else ""
        reply = QMessageBox.question(
            None, f"Delete node{plural}",
            f"Delete {len(deletable)} node{plural}?\n\n{names}\n\n"
            f"This writes 'delete <id>' to {self.rio_mount}/nodes/ctl "
            f"and cannot be undone from the operator.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False

        nodes_ctl = os.path.join(self.rio_mount, "nodes", "ctl")
        ctl_pipe = Pipe(nodes_ctl, self.worker)

        # ── Sweep routes that reference these nodes ─────────────────────
        # Without this, deleting a wired node leaves orphan routes in
        # /n/<m>/routes — the underlying Plan9Attachment keeps trying to
        # cat a now-vanished file, and the operator's graph carries
        # stale Connection objects pointing at port-paths that no longer
        # resolve. The cleanup is the operator's responsibility: the
        # server's `delete <id>` just removes the node directory, it
        # doesn't know about routes (they live under a different
        # subtree, /n/<m>/routes, owned by RoutesManager).
        #
        # We collect every route whose source OR destination path sits
        # under any of the to-be-deleted node directories and issue
        # `-<source>` writes for them. The route subscription then
        # propagates removals back, the graph drops the matching
        # Connection objects, and the ConnectionItems disappear from
        # the scene. By the time the `delete <id>` writes land on
        # nodes/ctl, the route layer is already clean.
        if self.routes is not None:
            # Build path prefixes once. os.sep on the end is critical —
            # without it "/n/m/nodes/bash_0" would also match
            # "/n/m/nodes/bash_01/...", which is a real foot-gun for
            # ids like bash_0 vs bash_01.
            del_prefixes = [
                os.path.normpath(it.node.dir_path) + os.sep
                for it in deletable
            ]

            def _under_any_deleted(path: str) -> bool:
                if not path:
                    return False
                norm = os.path.normpath(path)
                # Include exact match in case a port file's `path` lacks
                # the trailing separator on some platforms.
                return any(norm.startswith(p) or norm == p.rstrip(os.sep)
                           for p in del_prefixes)

            stale_sources: List[str] = []
            for src, dst, _running in self.routes.current():
                if _under_any_deleted(src) or _under_any_deleted(dst):
                    stale_sources.append(src)

            for src in stale_sources:
                # Fire-and-forget: the subscription will reconcile the
                # graph; we don't need the per-write callback. Failures
                # are logged by Routes.remove via its own on_done path.
                self.routes.remove(src)

            if stale_sources:
                print(f"operator: removing {len(stale_sources)} route(s) "
                      f"touching deleted node(s) before delete")

        # Track outstanding writes so we refresh exactly once at the end,
        # not once per node (which would N-amplify the scan load).
        outstanding = {"count": len(deletable), "failed": 0}

        def _after_write(node_id: str):
            def cb(result):
                if isinstance(result, Exception):
                    outstanding["failed"] += 1
                    print(f"operator: delete {node_id!r} failed: {result}")
                outstanding["count"] -= 1
                if outstanding["count"] == 0:
                    if outstanding["failed"] == len(deletable):
                        QMessageBox.warning(
                            None, "Delete failed",
                            f"No nodes were deleted. Is "
                            f"{nodes_ctl} writable and does the server "
                            f"accept 'delete <id>'?")
                    # Refresh once, after all writes have landed (or
                    # failed). The diff handles the rest.
                    QTimer.singleShot(200, self.refresh)
            return cb

        for item in deletable:
            node_id = item.node.node_id
            cmd = f"delete {node_id}\n".encode("utf-8")
            ctl_pipe.write_async(cmd, on_done=_after_write(node_id))

        return True

    # ── Node creation ──────────────────────────────────────────────────

    def create_user_node(self, kind: NodeKind, scene_pos: QPointF) -> None:
        """Write a node-creation command to /n/<m>/nodes/ctl. Requires
        a server-side handler that recognizes 'new <kind> <id>'."""
        kind_name = kind.value
        prefix = kind_name
        default = f"{prefix}_{self._next_node_index(prefix)}"
        node_id, ok = QInputDialog.getText(
            None, f"New {kind_name} node",
            f"Name for new {kind_name} node:",
            text=default,
        )
        if not ok or not node_id.strip():
            return
        node_id = node_id.strip()

        nodes_ctl = os.path.join(self.rio_mount, "nodes", "ctl")
        ctl_pipe = Pipe(nodes_ctl, self.worker)
        cmd = f"new {kind_name} {node_id}\n".encode("utf-8")

        def _after_write(result):
            if isinstance(result, Exception):
                QMessageBox.warning(
                    None, "Create failed",
                    f"Could not create node:\n{result}\n\n"
                    f"Is /n/<m>/nodes/ctl available on this filesystem?")
                return
            QTimer.singleShot(200, lambda: self._refresh_and_place(
                node_id, scene_pos))

        ctl_pipe.write_async(cmd, on_done=_after_write)

    def _refresh_and_place(self, node_id: str, pos: QPointF) -> None:
        def after_scan(result):
            self._on_scan_done(result)
            node = self.graph.get_node(node_id)
            if node is not None:
                view = self._node_views.get(id(node))
                if view is not None:
                    view.setPos(pos)
        self.worker.run_async(
            scan_all, self.rio_mount, self.llm_mount,
            on_done=after_scan,
        )

    def _next_node_index(self, prefix: str) -> int:
        existing = set(self.graph._nodes_by_id.keys())
        i = 0
        while f"{prefix}_{i}" in existing:
            i += 1
        return i

    # ── Teardown ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        for view in self._node_views.values():
            view.cleanup()
        if self.routes is not None:
            self.routes.stop()
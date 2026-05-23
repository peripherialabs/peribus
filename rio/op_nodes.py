"""
op_nodes.py — view-layer classes for the operator canvas.
==========================================================

Consolidated from what used to be:
    op/nodes/base.py    → "─── BASE: PortItem / ConnectionItem / NodeView ───"
    op/nodes/text.py    → "─── TextNodeView ───"
    op/nodes/debug.py   → "─── DebugNodeView ───"
    op/nodes/media.py   → "─── MediaNodeView ───"
    op/nodes/agent.py   → "─── AgentNodeView / GenericNodeView ───"
    op/nodes/exec.py    → "─── BashNodeView / PythonNodeView ───"
    op/nodes/__init__.py → "─── REGISTRY ───"

Why the model/view split: the old operator wove painting code, file
paths, and connection logic into single 6000-line classes. Splitting
them means a NodeView subclass can be 50 lines (paint + custom widget)
because every "what's this node connected to" question goes to the
model, and every "what bytes are at this path" question goes to a Pipe.

NodeView subclasses are responsible for:
  - Constructing their embedded widget (QTextEdit, QLabel, ...) inside
    a QGraphicsProxyWidget.
  - Setting up subscriptions to their own ports' Pipes (most nodes
    subscribe to their input ports on construction).
  - Routing user actions (button clicks) to Pipe writes or ctl commands.

NodeView never reaches across the graph to peer at upstream data — it
reads from its own ports. Data movement between ports is the routes
file's job (server-side `Plan9Attachment` runs the while-cat loop).

To add a new node type:
  1. Add a NodeKind enum value in op_core.py.
  2. Write a NodeView subclass below.
  3. Register it in VIEW_CLASS_BY_KIND at the bottom.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF, QSize, QUrl, Slot
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen,
    QPixmap, QTextCursor,
)
from PySide6.QtWidgets import (
    QCheckBox, QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
    QGraphicsProxyWidget, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
    _MULTIMEDIA_OK = True
except ImportError:
    QMediaPlayer = QAudioOutput = QVideoSink = None
    _MULTIMEDIA_OK = False

from .op_core import Node, NodeKind, Port, Connection, PortDirection, Theme
from .pipe import Pipe, ReadError, Subscription, SubscribeMode

if TYPE_CHECKING:
    from .op_core import Operator


# ═══════════════════════════════════════════════════════════════════════
# ─── BASE: PortItem / ConnectionItem / NodeView ────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# The Qt rendering of a model Node/Port/Connection. The view classes
# hold a reference back to the model object — that's where path,
# direction, and connection-list state lives. The view's job is
# painting + interaction (drag, hover, click).


_HEADER_COLOR_BY_KIND = {
    NodeKind.AGENT: "HEADER_AGENT",
    NodeKind.TERMINAL: "HEADER_TERMINAL",
    NodeKind.SCENE: "HEADER_SCENE",
    NodeKind.TEXT: "HEADER_TEXT",
    NodeKind.DEBUG: "HEADER_DEBUG",
    NodeKind.MEDIA: "HEADER_MEDIA",
    NodeKind.BASH: "HEADER_BASH",
    NodeKind.PYTHON: "HEADER_PYTHON",
    NodeKind.GENERIC: "HEADER_GENERIC",
}


def header_color_for(kind: NodeKind) -> QColor:
    return getattr(Theme, _HEADER_COLOR_BY_KIND.get(kind, "HEADER_GENERIC"))


# ═══════════════════════════════════════════════════════════════════════
# ─── STYLESHEET HELPERS ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Centralised QSS so every embedded widget — buttons, text edits,
# checkboxes — gets the same look. Each helper reads from Theme so
# light/dark mode flips correctly. Keep these short; widgets that need
# bespoke tweaks can append rules to the result.


def _qss_text_edit() -> str:
    """QTextEdit / QLineEdit / QPlainTextEdit shared style.

    Paper aesthetic from start_gui.py:
      - Opaque paper card background, no translucency.
      - 2px radius (editorial / mostly-flat).
      - Hairline soft border that snaps to solid near-black on focus,
        exactly like start_gui's QPlainTextEdit:focus rule.
      - Sage selection tint.
      - Scrollbars are ink-muted hairlines, transparent track.

    QLineEdit in start_gui is underlined (border-bottom only). Operator
    nodes don't use bare QLineEdits in tight rows, so we give all three
    widget types the framed-textarea look — the Acme / 9-paper feel
    consistent with the rest of the node.
    """
    return f"""
        QTextEdit, QLineEdit, QPlainTextEdit {{
            background-color: {Theme.EDIT_BG.name(QColor.HexArgb)};
            color: {Theme.TEXT_PRIMARY.name()};
            border: 1px solid {Theme.EDIT_BORDER.name(QColor.HexArgb)};
            border-radius: 2px;
            padding: 5px 6px;
            selection-background-color: {Theme.EDIT_SELECTION.name(QColor.HexArgb)};
            selection-color: {Theme.TEXT_PRIMARY.name()};
        }}
        QTextEdit:focus, QLineEdit:focus, QPlainTextEdit:focus {{
            background-color: {Theme.EDIT_BG_FOCUS.name(QColor.HexArgb)};
            border: 1px solid {Theme.EDIT_BORDER_FOCUS.name(QColor.HexArgb)};
        }}
        QTextEdit[readOnly="true"], QPlainTextEdit[readOnly="true"] {{
            background-color: {Theme.EDIT_BG.name(QColor.HexArgb)};
            color: {Theme.TEXT_PRIMARY.name()};
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 2px 1px 2px 1px;
        }}
        QScrollBar::handle:vertical {{
            background: {Theme.EDIT_BORDER.name(QColor.HexArgb)};
            border-radius: 2px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {Theme.NODE_BORDER_HOVER.name(QColor.HexArgb)};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 8px;
            margin: 1px 2px 1px 2px;
        }}
        QScrollBar::handle:horizontal {{
            background: {Theme.EDIT_BORDER.name(QColor.HexArgb)};
            border-radius: 2px;
            min-width: 20px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {Theme.NODE_BORDER_HOVER.name(QColor.HexArgb)};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
    """


def _qss_button(variant: str = "default") -> str:
    """QPushButton style. Variants:
        default — paper card, hairline border, snaps to ink on hover.
                  Matches start_gui's QPushButton#subtle.
        accent  — flat near-black ink with paper text. Primary action.
                  Matches start_gui's QPushButton#primary.
        read    — green-ink outlined button that inverts on hover.
                  An "ok"-flavoured action. Same role as #danger in
                  start_gui (outlined-coloured-button) but in green.

    All variants use 2px corner radius — start_gui's editorial radii.
    No pills here: pills (12px radius) in start_gui are reserved for
    *category* selection (the create/connect/standalone mode picker),
    not for general actions. Operator nodes have Run/Read/Write/Clear,
    which are actions, so they get the flat editorial look.
    """
    if variant == "accent":
        # #primary — flat ink button with paper text.
        bg = Theme.BUTTON_ACCENT_BG
        hov = Theme.BUTTON_ACCENT_HOVER
        prs = Theme.BUTTON_ACCENT_PRESSED
        bd = Theme.BUTTON_ACCENT_BORDER
        fg = Theme.BUTTON_ACCENT_TEXT
        # The accent variant uses a same-color border so it reads as
        # a solid filled block, like start_gui's #primary (border: none).
        return f"""
            QPushButton {{
                background-color: {bg.name(QColor.HexArgb)};
                color: {fg.name()};
                border: 1px solid {bd.name(QColor.HexArgb)};
                border-radius: 2px;
                padding: 5px 14px;
                font-family: {Theme.FONT_FAMILY_UI};
                font-size: 11px;
                font-weight: 600;
                min-height: 18px;
            }}
            QPushButton:hover {{
                background-color: {hov.name(QColor.HexArgb)};
                border: 1px solid {hov.name(QColor.HexArgb)};
            }}
            QPushButton:pressed {{
                background-color: {prs.name(QColor.HexArgb)};
                border: 1px solid {prs.name(QColor.HexArgb)};
            }}
            QPushButton:disabled {{
                background-color: {Theme.BUTTON_TEXT_DISABLED.name(QColor.HexArgb)};
                color: {Theme.BUTTON_ACCENT_TEXT.name()};
                border: 1px solid {Theme.BUTTON_TEXT_DISABLED.name(QColor.HexArgb)};
            }}
        """
    if variant == "read":
        # Outlined green button — paper bg + green ink, hover inverts.
        # Same pattern as start_gui's #danger but in GREEN_OK rather
        # than RED_INK.
        return f"""
            QPushButton {{
                background-color: {Theme.BUTTON_READ_BG.name(QColor.HexArgb)};
                color: {Theme.BUTTON_READ_TEXT.name()};
                border: 1px solid {Theme.BUTTON_READ_BORDER.name(QColor.HexArgb)};
                border-radius: 2px;
                padding: 4px 12px;
                font-family: {Theme.FONT_FAMILY_UI};
                font-size: 11px;
                min-height: 18px;
            }}
            QPushButton:hover {{
                background-color: {Theme.BUTTON_READ_HOVER.name(QColor.HexArgb)};
                color: {Theme.NODE_BG.name()};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BUTTON_READ_PRESSED.name(QColor.HexArgb)};
                color: {Theme.NODE_BG.name()};
            }}
            QPushButton:disabled {{
                color: {Theme.BUTTON_TEXT_DISABLED.name(QColor.HexArgb)};
                border: 1px solid {Theme.BUTTON_TEXT_DISABLED.name(QColor.HexArgb)};
            }}
        """
    # default — #subtle. Paper bg, hairline-soft border → solid ink on hover.
    bg = Theme.BUTTON_BG
    hov = Theme.BUTTON_BG_HOVER
    prs = Theme.BUTTON_BG_PRESSED
    bd = Theme.BUTTON_BORDER
    fg = Theme.BUTTON_TEXT
    return f"""
        QPushButton {{
            background-color: {bg.name(QColor.HexArgb)};
            color: {fg.name()};
            border: 1px solid {bd.name(QColor.HexArgb)};
            border-radius: 2px;
            padding: 4px 12px;
            font-family: {Theme.FONT_FAMILY_MONO};
            font-size: 11px;
            min-height: 18px;
        }}
        QPushButton:hover {{
            background-color: {hov.name(QColor.HexArgb)};
            border: 1px solid {Theme.BUTTON_BORDER_HOVER.name(QColor.HexArgb)};
        }}
        QPushButton:pressed {{
            background-color: {prs.name(QColor.HexArgb)};
        }}
        QPushButton:disabled {{
            color: {Theme.BUTTON_TEXT_DISABLED.name()};
            background-color: {Theme.BUTTON_BG.name(QColor.HexArgb)};
            border: 1px solid {Theme.BUTTON_BORDER.name(QColor.HexArgb)};
        }}
    """


def _qss_icon_button(variant: str = "default") -> str:
    """Compact, square-ish icon button — just a glyph, tight padding,
    small fixed footprint. Used for Read/Write/Run/Clear so the control
    row stops eating body space. Color roles match _qss_button:
        accent — filled ink (primary: Write / Run)
        read   — green-ink outline that inverts on hover (Read)
        default— paper card, hairline border (Clear / neutral)
    Pair with btn.setFixedSize(...) at the call site.
    """
    if variant == "accent":
        bg = Theme.BUTTON_ACCENT_BG
        hov = Theme.BUTTON_ACCENT_HOVER
        prs = Theme.BUTTON_ACCENT_PRESSED
        bd = Theme.BUTTON_ACCENT_BORDER
        fg = Theme.BUTTON_ACCENT_TEXT
        disabled_bg = Theme.BUTTON_TEXT_DISABLED
        return f"""
            QPushButton {{
                background-color: {bg.name(QColor.HexArgb)};
                color: {fg.name()};
                border: 1px solid {bd.name(QColor.HexArgb)};
                border-radius: 2px;
                padding: 0px;
                font-family: {Theme.FONT_FAMILY_UI};
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {hov.name(QColor.HexArgb)};
                border: 1px solid {hov.name(QColor.HexArgb)};
            }}
            QPushButton:pressed {{
                background-color: {prs.name(QColor.HexArgb)};
                border: 1px solid {prs.name(QColor.HexArgb)};
            }}
            QPushButton:disabled {{
                background-color: {disabled_bg.name(QColor.HexArgb)};
                color: {fg.name()};
                border: 1px solid {disabled_bg.name(QColor.HexArgb)};
            }}
        """
    if variant == "read":
        return f"""
            QPushButton {{
                background-color: {Theme.BUTTON_READ_BG.name(QColor.HexArgb)};
                color: {Theme.BUTTON_READ_TEXT.name()};
                border: 1px solid {Theme.BUTTON_READ_BORDER.name(QColor.HexArgb)};
                border-radius: 2px;
                padding: 0px;
                font-family: {Theme.FONT_FAMILY_UI};
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {Theme.BUTTON_READ_HOVER.name(QColor.HexArgb)};
                color: {Theme.NODE_BG.name()};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BUTTON_READ_PRESSED.name(QColor.HexArgb)};
                color: {Theme.NODE_BG.name()};
            }}
            QPushButton:disabled {{
                color: {Theme.BUTTON_TEXT_DISABLED.name(QColor.HexArgb)};
                border: 1px solid {Theme.BUTTON_TEXT_DISABLED.name(QColor.HexArgb)};
            }}
        """
    # default — paper card, hairline border.
    return f"""
        QPushButton {{
            background-color: {Theme.BUTTON_BG.name(QColor.HexArgb)};
            color: {Theme.BUTTON_TEXT.name()};
            border: 1px solid {Theme.BUTTON_BORDER.name(QColor.HexArgb)};
            border-radius: 2px;
            padding: 0px;
            font-family: {Theme.FONT_FAMILY_MONO};
            font-size: 11px;
        }}
        QPushButton:hover {{
            background-color: {Theme.BUTTON_BG_HOVER.name(QColor.HexArgb)};
            border: 1px solid {Theme.BUTTON_BORDER_HOVER.name(QColor.HexArgb)};
        }}
        QPushButton:pressed {{
            background-color: {Theme.BUTTON_BG_PRESSED.name(QColor.HexArgb)};
        }}
        QPushButton:disabled {{
            color: {Theme.BUTTON_TEXT_DISABLED.name()};
            background-color: {Theme.BUTTON_BG.name(QColor.HexArgb)};
            border: 1px solid {Theme.BUTTON_BORDER.name(QColor.HexArgb)};
        }}
    """


def _qss_checkbox() -> str:
    """Checkbox style — paper editorial 12px square that fills with
    ink when checked. Matches start_gui's QCheckBox rules verbatim
    so multi-window consistency holds.
    """
    return f"""
        QCheckBox {{
            color: {Theme.TEXT_PRIMARY.name()};
            font-family: {Theme.FONT_FAMILY_MONO};
            font-size: 11px;
            background: transparent;
            spacing: 6px;
            padding: 2px 0;
        }}
        QCheckBox::indicator {{
            width: 12px;
            height: 12px;
        }}
        QCheckBox::indicator:unchecked {{
            background-color: {Theme.CHECK_BG.name(QColor.HexArgb)};
            border: 1px solid {Theme.CHECK_BORDER.name(QColor.HexArgb)};
            border-radius: 2px;
        }}
        QCheckBox::indicator:unchecked:hover {{
            border: 1px solid {Theme.NODE_BORDER_HOVER.name(QColor.HexArgb)};
        }}
        QCheckBox::indicator:checked {{
            background-color: {Theme.CHECK_CHECKED_BG.name(QColor.HexArgb)};
            border: 1px solid {Theme.CHECK_CHECKED_BORDER.name(QColor.HexArgb)};
            border-radius: 2px;
        }}
        QCheckBox:disabled {{
            color: {Theme.BUTTON_TEXT_DISABLED.name(QColor.HexArgb)};
        }}
    """


def _qss_status_label(kind: str = "default") -> str:
    """One-line status pill. kind: default | ok | error | busy."""
    if kind == "ok":
        c = Theme.STATUS_OK
    elif kind == "error":
        c = Theme.STATUS_ERROR
    elif kind == "busy":
        c = Theme.STATUS_BUSY
    else:
        c = Theme.TEXT_SECONDARY
    return (f"color: {c.name()}; "
            f"font-family: {Theme.FONT_FAMILY_MONO}; "
            f"font-size: 9px; "
            f"padding: 1px 0px;")


class _CodeTextEdit(QTextEdit):
    """A QTextEdit that emits `runRequested` on Ctrl+Enter (or Cmd+Enter
    on macOS), so users can run a command without reaching for the mouse.
    Plain Enter still inserts a newline — these editors are multi-line
    by design (shell pipelines, multi-statement Python blocks)."""

    def __init__(self, on_run: Callable[[], None], parent=None):
        super().__init__(parent)
        self._on_run = on_run

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        is_enter = key in (Qt.Key_Return, Qt.Key_Enter)
        has_mod = bool(mods & (Qt.ControlModifier | Qt.MetaModifier))
        if is_enter and has_mod:
            try:
                self._on_run()
            except Exception:
                pass
            event.accept()
            return
        super().keyPressEvent(event)



class PortItem(QGraphicsEllipseItem):
    """The clickable circle for a port. Holds a back-reference to the
    model Port and the parent NodeView for hit-testing.

    Press starts a connection drag; release on another port creates a
    Connection (which writes a line to the routes file, which propagates
    back to the graph, which adds the visual ConnectionItem). No direct
    creation of ConnectionItem from here — the model is the source of
    truth.
    """

    PORT_RADIUS = Theme.PORT_RADIUS

    def __init__(self, port: Port, parent_node: "NodeView"):
        r = self.PORT_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, parent_node)
        self.port = port
        self.parent_node = parent_node

        color = (Theme.PORT_INPUT if port.direction.is_input
                 else Theme.PORT_OUTPUT)
        self.setBrush(QBrush(color))
        self.setPen(QPen(Theme.PORT_BORDER, 0.75))

        self.setAcceptHoverEvents(True)
        # Stack BEHIND the parent NodeView so the node body clips us:
        # when the port is centered on the body edge (x=0 or x=width),
        # only the outer hemisphere shows — the inner half is hidden
        # under the body. ItemStacksBehindParent flips the default
        # parent-then-child paint order, which is the only way a
        # child can be drawn before its parent in QGraphicsScene
        # without reparenting to the scene root. Z-value alone won't
        # do this — parent always paints before children regardless
        # of Z, that's the whole point of the parent/child hierarchy.
        self.setFlag(QGraphicsItem.ItemStacksBehindParent, True)
        self.setZValue(-1)

        # Label text drawn next to the port. Label is NOT a child of
        # the port (it's a child of the NodeView) so it doesn't inherit
        # the stacks-behind flag — labels need to read on top of the
        # body, the port nub doesn't.
        self._label = QGraphicsSimpleTextItem(port.name, parent_node)
        self._label.setFont(Theme.FONT_PORT_LABEL)
        self._label.setBrush(QBrush(Theme.TEXT_PORT))
        self._label.setZValue(3)

        # Small badge for blocking (streaming) ports — a dot inset on the
        # port circle. Cheap visual cue that "this port produces/consumes
        # streamed data."  The badge IS a child of the port, so it
        # inherits ItemStacksBehindParent transitively — meaning the
        # badge is also clipped by the node body. That's the desired
        # behaviour: the badge only shows on the visible outer half.
        self._badge: Optional[QGraphicsEllipseItem] = None
        if port.is_blocking:
            br = r * 0.4
            self._badge = QGraphicsEllipseItem(
                -br, -br, 2 * br, 2 * br, self,
            )
            self._badge.setBrush(QBrush(Theme.PORT_BLOCKING_BADGE))
            self._badge.setPen(Qt.NoPen)
            self._badge.setZValue(3)

    def update_label_position(self, node_width: float) -> None:
        """Position the label text relative to the port.

        With the port now half-clipped by the body (ItemStacksBehindParent),
        only the outer hemisphere shows. Labels read most naturally
        when they sit *inside* the body, a few pixels in from the edge —
        the visible port nub then points outward toward its connection,
        and the label is anchored to the content the port belongs to.
        """
        tr = self._label.boundingRect()
        # 6px inset from the body edge — same gap the editor uses from
        # its own padding, so the port label aligns visually with the
        # textarea / button rows below.
        inset = 6.0
        if self.port.direction.is_input:
            # Input on left edge (x=0): label sits to the right, inside.
            self._label.setPos(
                self.x() + inset,
                self.y() - tr.height() / 2,
            )
        else:
            # Output on right edge (x=width): label sits to the left,
            # right-aligned just inside the body.
            self._label.setPos(
                self.x() - inset - tr.width(),
                self.y() - tr.height() / 2,
            )

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))

    # ── interaction ──────────────────────────────────────────────────

    def hoverEnterEvent(self, event):
        # Slightly enlarge on hover. Cheap visual feedback.
        r = self.PORT_RADIUS * 1.4
        self.setRect(-r, -r, 2 * r, 2 * r)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        r = self.PORT_RADIUS
        self.setRect(-r, -r, 2 * r, 2 * r)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        # Left button  = one-shot transfer (or spawn-text-on-empty).
        # Right button = persistent route (written to /n/<m>/routes).
        btn = event.button()
        if btn == Qt.LeftButton:
            self.parent_node.operator.start_port_drag(self, is_route=False)
            event.accept()
            return
        if btn == Qt.RightButton:
            self.parent_node.operator.start_port_drag(self, is_route=True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        btn = event.button()
        if btn in (Qt.LeftButton, Qt.RightButton):
            self.parent_node.operator.finish_port_drag(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ConnectionItem(QGraphicsPathItem):
    """Bezier curve from source port to target port. Backed by a model
    Connection — the curve is just rendering.

    Selectable via left-click; press Delete/Backspace to remove the
    selected connection (the Operator listens for the key on the view).
    Removal writes to the routes file via Operator.remove_connection,
    which propagates back through the subscription and tears down this
    item.
    """

    def __init__(self, connection: Connection, operator: "Operator"):
        super().__init__()
        self.connection = connection
        self.operator = operator

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(-50)
        self._hovered = False

        # Hit-detection: use a fatter stroker around the curve so the
        # user can click near the line, not pixel-perfect on it.
        # QGraphicsPathItem uses self.path() for shape() by default,
        # which is a zero-width curve — practically un-clickable. We
        # override shape() below to widen it.
        self._hit_width = 10.0

        self.update_path()

    def shape(self):
        """Widen the click target around the curve."""
        from PySide6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(self._hit_width)
        return stroker.createStroke(self.path())

    # Orthogonal routing tuning.
    #   STUB_LENGTH: minimum horizontal "peel-out" before any vertical
    #     run. A short stub anchors the line visually to the port; with
    #     stub=0 the line leaves the port at 90° immediately, which
    #     reads ambiguously when ports are vertically stacked. 8px is
    #     just enough to disambiguate the exit direction without
    #     pushing the elbow far from the port (was 18 — too greedy
    #     with horizontal space on tight layouts).
    #   CORNER_RADIUS: how strongly the elbows are rounded. 0 = sharp
    #     90° corners; a small positive value gives the same family
    #     of soft corners as the node body. Keep ≤ stub/2 so the
    #     radius never eats the whole stub — at stub=8 that caps us
    #     around 4.
    STUB_LENGTH = 8.0
    CORNER_RADIUS = 4.0

    def update_path(self) -> None:
        """Recompute the path. Called when either endpoint moves.

        Routes as an orthogonal (right-angle) "Z" with two elbows:

            p0 ──┐
                 │
                 └── p1

        Each port "peels out" by STUB_LENGTH in the direction its side
        of the node faces (output ports push right, input ports push
        left). Between the two stubs we run vertically. If the two
        stubs would overlap horizontally — e.g. an output on the right
        of node A feeding an input on the left of node B that's *to
        the left* of A — the routing folds back over the top/bottom
        of the nodes via a wider U-shape.

        With CORNER_RADIUS > 0 the elbows are softened by a quadratic
        arc whose control point is the elbow itself; this matches the
        rounded-rect feel of the body without going back to a full bezier.
        """
        src_view = self.operator.port_item_for(self.connection.source)
        tgt_view = self.operator.port_item_for(self.connection.target)
        if src_view is None or tgt_view is None:
            return

        p0 = src_view.scene_center()
        p1 = tgt_view.scene_center()

        # Sign convention: +1 → port faces right (output port on a node's
        # right edge), −1 → faces left (input port on a node's left edge).
        src_dir = -1 if src_view.port.direction.is_input else 1
        tgt_dir = -1 if tgt_view.port.direction.is_input else 1

        stub = self.STUB_LENGTH

        # Anchor points: where the line *enters* the orthogonal grid
        # after peeling away from each port.
        a0 = QPointF(p0.x() + src_dir * stub, p0.y())
        a1 = QPointF(p1.x() + tgt_dir * stub, p1.y())

        # Pick the x for the vertical run.
        #
        # Normal case: src faces right (+1) and tgt faces left (−1),
        # and src is to the left of tgt — the vertical run sits at the
        # midpoint between the two stub ends. That gives the classic
        # ┐─┘ step.
        #
        # Folded case: stubs would overlap or run backwards. We push
        # the vertical run *past* the further stub so the path forms a
        # U around whichever side is needed. Without this, two nodes
        # in a feedback loop produce a Z that runs back through their
        # interiors.
        normal_forward = (src_dir == 1 and tgt_dir == -1
                          and a1.x() >= a0.x())
        normal_backward = (src_dir == -1 and tgt_dir == 1
                           and a1.x() <= a0.x())
        if normal_forward or normal_backward:
            mid_x = (a0.x() + a1.x()) / 2.0
        else:
            # Stubs face the same way, or face inward — the path has
            # to U-turn. Pick mid_x outside the further stub by another
            # STUB_LENGTH so the corners don't crowd the ports.
            if src_dir == 1:
                mid_x = max(a0.x(), a1.x()) + stub
            else:
                mid_x = min(a0.x(), a1.x()) - stub

        # Elbows. e0 is on the source stub's x-line, e1 on the target's.
        e0 = QPointF(mid_x, a0.y())
        e1 = QPointF(mid_x, a1.y())

        # Build the path: p0 → a0 → e0 → e1 → a1 → p1, with rounded
        # corners at e0 and e1 if CORNER_RADIUS > 0.
        path = QPainterPath(p0)
        path.lineTo(a0)

        r = self.CORNER_RADIUS
        if r <= 0 or abs(e1.y() - e0.y()) < 1.0:
            # Sharp corners, or so close to colinear that there's no
            # vertical run to round into.
            path.lineTo(e0)
            path.lineTo(e1)
            path.lineTo(a1)
        else:
            # Clamp the corner radius so it can't exceed either the
            # horizontal stub-to-elbow distance or half the vertical
            # run — otherwise the arcs would back-overlap.
            r = min(r,
                    abs(e0.x() - a0.x()),
                    abs(e1.x() - a1.x()),
                    abs(e1.y() - e0.y()) / 2.0)
            v_dir = 1 if e1.y() > e0.y() else -1
            h_in = 1 if e0.x() > a0.x() else -1
            h_out = 1 if a1.x() > e1.x() else -1
            # First elbow: arrive horizontally at e0 - h_in*r, curve
            # through e0 to e0 + (0, v_dir*r).
            path.lineTo(QPointF(e0.x() - h_in * r, e0.y()))
            path.quadTo(e0, QPointF(e0.x(), e0.y() + v_dir * r))
            # Vertical segment.
            path.lineTo(QPointF(e1.x(), e1.y() - v_dir * r))
            # Second elbow: curve through e1 to e1 + (h_out*r, 0).
            path.quadTo(e1, QPointF(e1.x() + h_out * r, e1.y()))
            path.lineTo(a1)

        path.lineTo(p1)

        self.setPath(path)
        # prepareGeometryChange isn't strictly needed for QGraphicsPathItem
        # (setPath does it), but we call self.update() so the new hit
        # area (from shape() restroking) refreshes.
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        if self.isSelected():
            color = Theme.NODE_BORDER_SELECTED
            width = 2.5
        elif self._hovered:
            color = Theme.CONN_HOVER
            width = 2.0
        elif getattr(self.connection, "is_static", False):
            # Static (visual-only) link. Distinct color + heavier weight
            # so the user can see at a glance that this is NOT a route
            # entry in /n/<m>/routes — the I/O happens only when the
            # user clicks Read/Write on a wired text node.
            color = Theme.CONN_DRAGGING_ONESHOT
            width = 2.6
        elif self.connection.running:
            color = Theme.CONN_RUNNING
            width = 1.5
        else:
            color = Theme.CONN_STOPPED
            width = 1.5
        painter.setPen(QPen(color, width))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        # Left-click: select. Take focus so the operator's keyboard
        # handler knows we're the active deletion target. Right-click
        # has no special meaning on connections — use the Delete key
        # to remove a selected connection.
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
        super().mousePressEvent(event)


class _ResizeGrip(QGraphicsRectItem):
    """A small invisible square at one corner of a NodeView that drives
    a resize drag. It exists as a high-Z child so it catches the press
    even where the embedded proxy widget (a QTextEdit, say) would
    otherwise eat the mouse event. The grip is transparent — the visible
    affordance is just the diagonal cursor on hover.
    """

    def __init__(self, corner: str, node_view: "NodeView"):
        size = node_view.RESIZE_GRIP
        super().__init__(-size / 2, -size / 2, size, size, node_view)
        self._corner = corner
        self._nv = node_view
        self.setBrush(Qt.NoBrush)
        self.setPen(Qt.NoPen)
        self.setAcceptHoverEvents(True)
        self.setCursor(NodeView._cursor_for_corner(corner))
        # Sit above the proxy (z=4), ports (z=3), title (z=5).
        self.setZValue(6)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._nv.begin_corner_resize(self._corner, event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._nv._resize_corner is not None:
            self._nv._perform_resize(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._nv._resize_corner is not None:
            self._nv.end_corner_resize()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class NodeView(QGraphicsRectItem):
    """Base class for visual nodes. Subclasses provide a custom body
    widget and any node-specific behavior, but the header, ports,
    layout, and Qt plumbing live here.

    Construction:
        view = SomeNodeView(model_node, operator)
        view.build_ports()
        view.build_body()
        view.layout()

    The Operator owns the lifecycle; views call back to it via
    `self.operator` for connection drags, deletion, etc.
    """

    DEFAULT_WIDTH = 240.0
    MIN_BODY_HEIGHT = 60.0

    # When True, the embedded body widget starts directly under the
    # header and the port rows sit *over* its top edge instead of
    # reserving a full port-section band above it. This reclaims the
    # vertical gap that used to open up between the header and the
    # editor on nodes whose ports (ctl, in, …) stack two-or-more deep.
    # Header+ports-only nodes (agent, generic) leave this False so their
    # ports still get a dedicated band.
    BODY_OVERLAPS_PORTS = False

    # Vertical clearance reserved above an overlapping body for the port
    # nubs/labels that ride its top edge. With PORT_SPACING=18 this gives
    # a 1-input node ~7px of band and a 2-input node ~25px, versus the
    # ~46px the old full band cost — that delta is the reclaimed space.
    PORT_OVERLAP_CLEARANCE = 7.0

    def __init__(self, node: Node, operator: "Operator"):
        super().__init__()
        self.node = node
        self.operator = operator

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(0)

        self.header_color = header_color_for(node.kind)

        # Port items (created by build_ports). Indexed by model Port id.
        self._port_items: dict[int, PortItem] = {}

        # Title text
        #
        # Matches the /color picker's section labels: monospace,
        # lower-weight, lower-contrast. Without a filled header bar
        # behind it, TEXT_ON_HEADER (tuned for the colored band) is
        # the wrong color — we use TEXT_SECONDARY instead so the title
        # sits on the body the same way "colors" / "ansi" sit in the
        # inline color picker.
        self._title = QGraphicsSimpleTextItem(node.node_id, self)
        self._title.setFont(Theme.FONT_NODE_TITLE)
        self._title.setBrush(QBrush(Theme.TEXT_SECONDARY))
        self._title.setZValue(5)

        self._hovered = False

        # ── Resize state ────────────────────────────────────────────
        # When the user drags a corner, we pin width/height to explicit
        # per-instance values. None means "use DEFAULT_WIDTH and the
        # content-derived height" (the original auto-sizing behaviour).
        # Once set, layout() respects these instead.
        self._user_width: Optional[float] = None
        self._user_height: Optional[float] = None
        # Active-drag bookkeeping. _resize_corner is one of
        # "tl"/"tr"/"bl"/"br" while a corner drag is in flight, else None.
        self._resize_corner: Optional[str] = None
        self._resize_start_scene = QPointF()
        self._resize_start_rect = QRectF()
        self._resize_start_pos = QPointF()
        # Corner grip item — created now, positioned in layout(). It sits
        # above the embedded proxy so a drag on the corner resizes the
        # node even when the corner overlaps an editor/output widget.
        # Bottom-right only: the standard, unambiguous resize handle.
        self._grips = {"br": _ResizeGrip("br", self)}

    # ── construction protocol ─────────────────────────────────────────

    def build_ports(self) -> None:
        """Create PortItem instances for each Port in the model node.
        Subclasses can override to filter or reorder; default builds all."""
        for port in self.node.ports:
            self._port_items[id(port)] = PortItem(port, self)

    def build_body(self) -> None:
        """Subclasses override to add their custom widget(s)."""
        pass

    def body_height_hint(self) -> float:
        """Subclasses override if they have an embedded widget. The base
        version just allocates space for the ports."""
        port_rows = max(len(self.node.inputs), len(self.node.outputs), 1)
        # +2 (was +8) — port section is just ports now, no need for a
        # tall gap before whatever comes below.
        return Theme.PORT_MARGIN_TOP + port_rows * Theme.PORT_SPACING + 2

    def min_width(self) -> float:
        """Smallest width this node may be resized to. Wide enough for
        the title plus a little body; subclasses with editors can leave
        the default."""
        return 160.0

    def min_total_height(self) -> float:
        """Smallest overall (header + body) height this node may be
        resized to."""
        return Theme.NODE_HEADER_HEIGHT + self.MIN_BODY_HEIGHT

    def layout(self) -> None:
        """Recompute size and lay out ports. Call after build_ports and
        build_body, and again after any internal change (e.g. DebugNode
        adding more inputs).

        Width and height come from the user-resized values when set
        (self._user_width / self._user_height); otherwise width falls
        back to DEFAULT_WIDTH and height is derived from content."""
        if self._user_width is not None:
            width = max(self._user_width, self.min_width())
        else:
            width = self.DEFAULT_WIDTH
        # Compute port-section height first so we can place the proxy
        # below it (instead of underneath the port labels, which is what
        # the old code did and which produced the overlap visible in the
        # screenshots).
        port_rows = max(len(self.node.inputs), len(self.node.outputs), 1)
        # Tightened padding (was +8 → now +2). The port labels read
        # plenty visible at 2px below the last port row, and shaving
        # those 6px off compounds across every node on screen.
        port_section_h = Theme.PORT_MARGIN_TOP + port_rows * Theme.PORT_SPACING + 2
        proxy_h = self.proxy_height_hint()

        # ── where does the embedded body start, and where do ports go? ──
        # Overlap mode (text/media/bash/python/debug): the body widget
        # butts right up under the header and the ports ride its top
        # edge, so we don't pay for a full port band. Non-overlap mode
        # (agent/generic and the base default): keep the dedicated band.
        #
        # Guard: only overlap when the deepest port column is shallow
        # (≤2 rows). A node with many stacked inputs would otherwise run
        # its port labels right down over the editor text; those nodes
        # keep the dedicated band. The nodes we actually want this for
        # (ctl + in + OUT) are all 1–2 rows deep.
        overlap = (self.BODY_OVERLAPS_PORTS and proxy_h > 0
                   and port_rows <= 2)
        if overlap:
            # Reserve only the tight height the port rows actually need —
            # no PORT_MARGIN_TOP band above and no +2 slack below. The
            # editor starts immediately under the last port row. This is
            # where the reclaimed vertical space comes from: a 2-input
            # node drops from PORT_MARGIN_TOP(8) + 2*18 + 2 = 46px of
            # dead band to 2*PORT_SPACING_TIGHT.
            tight = self.PORT_OVERLAP_CLEARANCE + (port_rows - 1) * Theme.PORT_SPACING
            proxy_top = Theme.NODE_HEADER_HEIGHT + tight
            port_y_start = Theme.NODE_HEADER_HEIGHT + self.PORT_OVERLAP_CLEARANCE * 0.5
        else:
            proxy_top = Theme.NODE_HEADER_HEIGHT + port_section_h
            port_y_start = Theme.NODE_HEADER_HEIGHT + Theme.PORT_MARGIN_TOP

        body_h = max(self.body_height_hint(), self.MIN_BODY_HEIGHT)
        # Body bottom padding also tightened (+8 → +2). The embedded
        # widget already has its own setContentsMargins, so the node
        # rect shouldn't add another fat gap on top of that.
        content_body_h = max(body_h, (proxy_top - Theme.NODE_HEADER_HEIGHT) + proxy_h + 2)

        if self._user_height is not None:
            # Respect the user's dragged height, clamped to a sane floor
            # so the editor and ports always remain usable.
            total = max(self._user_height, self.min_total_height())
            body_h = total - Theme.NODE_HEADER_HEIGHT
        else:
            body_h = content_body_h
            total = Theme.NODE_HEADER_HEIGHT + body_h
        self.setRect(0, 0, width, total)

        # Position title in the header — right of the accent dot.
        # Layout mirrors PresetChip in the /color picker: dot at x=8,
        # text after a small gap.
        title_text = self.node_id_for_title()
        if self._title.text() != title_text:
            self._title.setText(title_text)
        tr = self._title.boundingRect()
        title_x = (self.ACCENT_DOT_LEFT_MARGIN
                   + self.ACCENT_DOT_DIAMETER + 8.0)
        max_w = width - title_x - 14  # leave a bit of right padding
        if tr.width() > max_w:
            fm = QFontMetrics(Theme.FONT_NODE_TITLE)
            self._title.setText(
                fm.elidedText(title_text, Qt.ElideMiddle, int(max_w)))
            tr = self._title.boundingRect()
        self._title.setPos(title_x,
                           (Theme.NODE_HEADER_HEIGHT - tr.height()) / 2)

        # Position ports
        in_idx = 0
        out_idx = 0
        for port in self.node.ports:
            item = self._port_items.get(id(port))
            if item is None:
                continue
            if port.direction.is_input:
                item.setPos(0, port_y_start + in_idx * Theme.PORT_SPACING)
                in_idx += 1
            else:
                item.setPos(width, port_y_start + out_idx * Theme.PORT_SPACING)
                out_idx += 1
            item.update_label_position(width)

        # Reposition embedded proxy widget. In non-overlap mode this is
        # below the port section; in overlap mode it's just under the
        # header with the ports riding its top edge.
        # Gutter shrunk from 8px → 4px on each side, matching the new
        # internal layout margins of every build_body() method. That
        # reclaims 8px of horizontal width for the editor / output
        # viewer, which is what the user sees as "more textarea".
        proxy = getattr(self, "_proxy", None)
        if proxy is not None:
            proxy.setPos(4, proxy_top)
            inner_w = max(int(width - 8), 40)
            w = proxy.widget()
            if w is not None:
                if w.width() != inner_w:
                    w.setFixedWidth(inner_w)
                # Stretch the embedded widget to fill the body below the
                # proxy_top down to a small bottom inset. Without this the
                # container keeps its sizeHint height and a taller node
                # just shows empty paper under the editor. We only pin a
                # fixed height when the node has been user-resized taller
                # than its natural content; otherwise leave the widget at
                # its natural height so auto-sized nodes look unchanged.
                avail_h = int(total - proxy_top - 4)
                if avail_h > 0 and (self._user_height is not None
                                    or self._user_width is not None):
                    if w.height() != avail_h:
                        w.setFixedHeight(avail_h)

        # Position the bottom-right corner grip.
        r = self.rect()
        grips = getattr(self, "_grips", None)
        if grips and "br" in grips:
            grips["br"].setPos(r.right(), r.bottom())

    def node_id_for_title(self) -> str:
        """Hook so subclasses can override the displayed title. Defaults
        to the model node_id."""
        return self.node.node_id

    def proxy_height_hint(self) -> float:
        """Return the preferred height of the embedded widget (proxy).
        Returns 0 if there's no proxy. Used by layout() to size the body
        so the proxy sits cleanly below the port section."""
        proxy = getattr(self, "_proxy", None)
        if proxy is None or proxy.widget() is None:
            return 0.0
        return float(proxy.widget().sizeHint().height())

    # ── painting ──────────────────────────────────────────────────────
    #
    # Paper-style node (modeled on start_gui.py's PAPER theme + the
    # terminal's /color inline widget):
    #   - Opaque off-white card, no translucency, no shadows.
    #   - Single hairline near-black border. Border strength ramps with
    #     state: idle = soft (80α), hover = firm (180α), selected =
    #     solid ink. Width stays at 1px throughout — paper doesn't
    #     thicken lines, it darkens them.
    #   - Node kind = small accent dot next to the title (8px), same
    #     idea as a PresetChip in the /color picker.
    #   - A hairline under the title row separates header from body.
    #     start_gui uses whitespace alone, but operator nodes carry
    #     ports + editors + output viewers, so a faint divider helps
    #     anchor the title row.

    ACCENT_DOT_DIAMETER: float = 8.0
    ACCENT_DOT_LEFT_MARGIN: float = 12.0

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        radius = Theme.NODE_CORNER_RADIUS

        # Border ramps by state. Width stays at 1.0 — the paper theme
        # gets *darker* on focus, not *thicker*. NODE_BORDER variants
        # in _Paper carry that ramp via their alpha (80 / 180 / 255).
        if self.isSelected():
            border = Theme.NODE_BORDER_SELECTED
        elif self._hovered:
            border = Theme.NODE_BORDER_HOVER
        else:
            border = Theme.NODE_BORDER
        bw = 1.0

        # Body — opaque paper card. Cosmetic pen keeps the border at
        # 1px regardless of view zoom, which matches the editorial
        # hairline feel (a hairline that scales with zoom turns into
        # a fat slab on zoom-in).
        body_path = QPainterPath()
        body_path.addRoundedRect(rect, radius, radius)
        pen = QPen(border, bw)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(Theme.NODE_BG))
        painter.drawPath(body_path)

        # Accent dot — 8px disk next to the title. Uses the per-kind
        # accent so kind is still legible at a glance (sage = agent,
        # amber = python, etc.). The dot is filled at full alpha with
        # no border on paper — the surrounding card is opaque ink-on-
        # paper, so a black outline would compete with the kind color.
        dot_d = self.ACCENT_DOT_DIAMETER
        dot_x = rect.x() + self.ACCENT_DOT_LEFT_MARGIN
        dot_y = rect.y() + (Theme.NODE_HEADER_HEIGHT - dot_d) / 2.0
        accent = QColor(self.header_color)
        accent.setAlpha(255)
        painter.setBrush(QBrush(accent))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(dot_x + dot_d / 2.0,
                                    dot_y + dot_d / 2.0),
                            dot_d / 2.0, dot_d / 2.0)

        # Hairline under the title row.
        sep_y = rect.y() + Theme.NODE_HEADER_HEIGHT
        sep_pen = QPen(Theme.SEPARATOR, 0.5)
        sep_pen.setCosmetic(True)
        painter.setPen(sep_pen)
        painter.drawLine(
            QPointF(rect.x() + 8, sep_y),
            QPointF(rect.right() - 8, sep_y),
        )

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    # ── geometry-change hook ──────────────────────────────────────────

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # Notify operator that our connections need redrawing.
            self.operator.notify_node_moved(self.node)
        return super().itemChange(change, value)

    # ── corner resize ──────────────────────────────────────────────────
    #
    # Drag any of the four corners to resize. The grip is a small square
    # region at each corner of the node rect; the rest of the body still
    # moves/selects the node as before. Top/left corners also shift the
    # node's scene position so the *opposite* corner stays anchored,
    # which is what makes a top-left drag feel like a normal resize
    # rather than a move.

    RESIZE_GRIP = 12.0  # px hit zone at each corner (local coords)

    def _corner_at(self, pos: QPointF) -> Optional[str]:
        """Return 'br' if the local point is within the grip zone of the
        bottom-right corner, else None. Bottom-right is the only
        resize handle."""
        r = self.rect()
        g = self.RESIZE_GRIP
        if abs(pos.x() - r.right()) <= g and abs(pos.y() - r.bottom()) <= g:
            return "br"
        return None

    @staticmethod
    def _cursor_for_corner(corner: Optional[str]):
        if corner == "br":
            return Qt.SizeFDiagCursor   # ↘↖
        return Qt.ArrowCursor

    def begin_corner_resize(self, corner: str, scene_pos: QPointF) -> None:
        """Start a resize drag from the named corner. Called by the grip
        item (or our own fallback press handler). Records the starting
        geometry and suspends movement so the drag resizes, not moves."""
        self._resize_corner = corner
        self._resize_start_scene = QPointF(scene_pos)
        self._resize_start_rect = QRectF(self.rect())
        self._resize_start_pos = QPointF(self.pos())
        self.setFlag(QGraphicsItem.ItemIsMovable, False)

    def end_corner_resize(self) -> None:
        """Finish a resize drag and re-enable movement."""
        self._resize_corner = None
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def mousePressEvent(self, event):
        # Fallback: a left-press on a corner zone that isn't covered by a
        # grip child (e.g. the bare top corners) still starts a resize.
        # Where a grip child exists over the cursor it gets the press
        # first and this never runs — both paths funnel through
        # begin_corner_resize so behaviour is identical.
        if event.button() == Qt.LeftButton:
            corner = self._corner_at(event.pos())
            if corner is not None:
                self.begin_corner_resize(corner, event.scenePos())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_corner is not None:
            self._perform_resize(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resize_corner is not None:
            self.end_corner_resize()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _perform_resize(self, scene_pos: QPointF) -> None:
        """Translate the bottom-right drag delta into a new width/height
        and relayout. The node origin stays fixed — only the right and
        bottom edges move."""
        delta = scene_pos - self._resize_start_scene
        r0 = self._resize_start_rect
        self._user_width = max(r0.width() + delta.x(), self.min_width())
        self._user_height = max(r0.height() + delta.y(), self.min_total_height())

        self.prepareGeometryChange()
        self.layout()
        # Ports moved (width/height changed) → redraw our connections.
        self.operator.notify_node_moved(self.node)
        self.update()

    # ── teardown ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Stop any active subscriptions, timers, etc. Subclasses override
        to clean up Pipe subscriptions; the base does nothing."""
        pass

    # ── lookup helper ─────────────────────────────────────────────────

    def port_item(self, port: Port) -> Optional[PortItem]:
        return self._port_items.get(id(port))


class TempConnectionItem(QGraphicsPathItem):
    """The curve drawn while the user is dragging from a port to nowhere
    yet. Operator owns one of these; it's created on `start_port_drag`
    and removed on `finish_port_drag`.

    Visual mode:
      - is_route=True  (right-button drag): thick, dashed, accent color.
        A persistent route is the heavier commitment, so the preview
        is visually loud — it's the gesture the user will want to be
        sure they meant.
      - is_route=False (left-button drag): thin, solid, neutral text
        color. A one-shot transfer is lightweight (no entry in
        /n/<m>/routes), so the preview is understated to match.
    """

    def __init__(self, start: PortItem, is_route: bool = True):
        super().__init__()
        self.start_port = start
        self.is_route = is_route
        self.setZValue(100)
        if is_route:
            # Right-drag: persistent route. Loud — thick, dashed, orange.
            pen = QPen(Theme.CONN_DRAGGING_ONESHOT, 2.6)
            pen.setStyle(Qt.DashLine)
        else:
            # Left-drag: one-shot. Quiet — thin, solid, neutral text color.
            # TEXT_PRIMARY adapts to dark/light theme so this stays
            # readable in both modes (was hard-coded gray before).
            pen = QPen(Theme.TEXT_PRIMARY, 1.0)
            pen.setStyle(Qt.SolidLine)
        self.setPen(pen)
        self.setBrush(Qt.NoBrush)

    # Match the routing geometry of ConnectionItem so the preview
    # foreshadows what the committed line will look like.
    STUB_LENGTH = 8.0
    CORNER_RADIUS = 4.0

    def set_end(self, scene_pos: QPointF) -> None:
        """Update the dragging preview to terminate at scene_pos.

        Mirrors ConnectionItem.update_path but the cursor end has no
        port, so we treat it as a free endpoint: the path peels off
        the source port for STUB_LENGTH and then steps orthogonally
        to the cursor. The cursor's "stub" is zero — the user is the
        anchor, not a port.
        """
        p0 = self.start_port.scene_center()
        p1 = scene_pos

        src_dir = -1 if self.start_port.port.direction.is_input else 1
        stub = self.STUB_LENGTH

        a0 = QPointF(p0.x() + src_dir * stub, p0.y())

        # Decide whether the cursor is "ahead" of the source stub
        # (in the direction the port faces). If so, a clean Z works:
        # mid_x sits between a0 and p1. If the cursor is *behind* the
        # port (folded), route around by an extra stub so the elbow
        # doesn't cross back through the source node.
        ahead = (src_dir == 1 and p1.x() >= a0.x()) or \
                (src_dir == -1 and p1.x() <= a0.x())
        if ahead:
            mid_x = (a0.x() + p1.x()) / 2.0
        else:
            mid_x = a0.x() + src_dir * stub

        e0 = QPointF(mid_x, a0.y())
        e1 = QPointF(mid_x, p1.y())

        path = QPainterPath(p0)
        path.lineTo(a0)

        r = self.CORNER_RADIUS
        if r <= 0 or abs(e1.y() - e0.y()) < 1.0:
            path.lineTo(e0)
            path.lineTo(e1)
            path.lineTo(p1)
        else:
            # Same clamp as ConnectionItem — keep arcs inside the
            # available straight runs.
            r = min(r,
                    abs(e0.x() - a0.x()),
                    abs(p1.x() - e1.x()) if abs(p1.x() - e1.x()) > 0 else r,
                    abs(e1.y() - e0.y()) / 2.0)
            v_dir = 1 if e1.y() > e0.y() else -1
            h_in = 1 if e0.x() > a0.x() else -1
            # h_out: if cursor and elbow are colinear (mid_x == p1.x()),
            # the second arc would degenerate; skip rounding it then.
            same_col = abs(p1.x() - e1.x()) < 1.0
            path.lineTo(QPointF(e0.x() - h_in * r, e0.y()))
            path.quadTo(e0, QPointF(e0.x(), e0.y() + v_dir * r))
            if same_col:
                path.lineTo(e1)
                path.lineTo(p1)
            else:
                h_out = 1 if p1.x() > e1.x() else -1
                path.lineTo(QPointF(e1.x(), e1.y() - v_dir * r))
                path.quadTo(e1, QPointF(e1.x() + h_out * r, e1.y()))
                path.lineTo(p1)

        self.setPath(path)


# ═══════════════════════════════════════════════════════════════════════
# ─── TextNodeView ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Embedded text area + two ports (`in`, `OUT`) + three controls:
#   - Read button:  one-shot read of `in` into the text area.
#   - Write button: write the text area's contents to `OUT`.
#   - Auto-read:    subscribe to `in`; every chunk replaces the text.
#
# Notable: this node reads from its OWN input port (<node_dir>/in), not
# from upstream. When wired, the routes file pumps the upstream OUTPUT
# into our `in`, and we just subscribe to `in`. The old code reached
# across the graph to read directly from the source port's path; the
# new model puts the abstraction at the route level.


class TextNodeView(NodeView):
    """Text node — embedded QTextEdit + Read/Write/Auto controls."""

    DEFAULT_WIDTH = 320.0
    MIN_BODY_HEIGHT = 220.0
    BODY_OVERLAPS_PORTS = True

    def __init__(self, node: Node, operator: "Operator"):
        super().__init__(node, operator)
        self._sub: Optional[Subscription] = None
        self._in_pipe: Optional[Pipe] = None
        self._out_pipe: Optional[Pipe] = None
        self._proxy: Optional[QGraphicsProxyWidget] = None
        self._text_edit: Optional[QTextEdit] = None
        self._status_label: Optional[QLabel] = None
        self._auto_check: Optional[QCheckBox] = None

    def build_body(self) -> None:
        # Resolve pipes by name, not by position. Port lists are
        # sorted alphabetically by the scanner; the "first input" is
        # frequently `ctl` (not `in`), which would make Auto subscribe
        # to the wrong file.
        in_port = self.node.get_port("in")
        out_port = self.node.get_port("OUT")
        if in_port is not None:
            self._in_pipe = Pipe(in_port.path, self.operator.worker)
        if out_port is not None:
            self._out_pipe = Pipe(out_port.path, self.operator.worker)

        # ── widget container ────────────────────────────────────────
        container = QWidget()
        container.setAttribute(Qt.WA_TranslucentBackground)
        container.setStyleSheet("background: transparent;")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # Text area — comes first so the controls sit beneath it (matches
        # the stable layout where Read/Write are visually anchored to the
        # textarea they act on).
        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText("Text content…")
        self._text_edit.setFont(Theme.FONT_CODE_SMALL)
        self._text_edit.setMinimumHeight(120)
        self._text_edit.setStyleSheet(_qss_text_edit())
        self._text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        v.addWidget(self._text_edit)

        # Buttons row — compact icon buttons: Read (▼, green), Write
        # (▲, ink), plus the Auto checkbox. Icon-only with tooltips so
        # the row stays short and stops eating horizontal/vertical space.
        # Stretch at the end keeps them grouped at the left.
        btns = QHBoxLayout()
        btns.setSpacing(5)
        btns.setContentsMargins(0, 0, 0, 0)

        _ICON_BTN = QSize(26, 22)

        self._read_btn = QPushButton("▼")
        self._read_btn.setToolTip("Read input → text area")
        self._read_btn.setFixedSize(_ICON_BTN)
        self._read_btn.setCursor(Qt.PointingHandCursor)
        self._read_btn.setStyleSheet(_qss_icon_button("read"))
        self._read_btn.clicked.connect(self._on_read_clicked)
        self._read_btn.setEnabled(self._in_pipe is not None)
        btns.addWidget(self._read_btn)

        self._write_btn = QPushButton("▲")
        self._write_btn.setToolTip("Write text area → OUT")
        self._write_btn.setFixedSize(_ICON_BTN)
        self._write_btn.setCursor(Qt.PointingHandCursor)
        self._write_btn.setStyleSheet(_qss_icon_button("accent"))
        self._write_btn.clicked.connect(self._on_write_clicked)
        self._write_btn.setEnabled(self._out_pipe is not None)
        btns.addWidget(self._write_btn)

        self._auto_check = QCheckBox("Auto")
        self._auto_check.setCursor(Qt.PointingHandCursor)
        self._auto_check.setStyleSheet(_qss_checkbox())
        self._auto_check.setEnabled(self._in_pipe is not None)
        self._auto_check.toggled.connect(self._on_auto_toggled)
        btns.addWidget(self._auto_check)
        btns.addStretch()

        # Status sits on the SAME row as the controls now, right-aligned,
        # instead of claiming its own line below — another row of body
        # space reclaimed. It elides rather than wraps.
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_qss_status_label())
        self._status_label.setWordWrap(False)
        self._status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        btns.addWidget(self._status_label)
        v.addLayout(btns)

        container.setFixedWidth(int(self.DEFAULT_WIDTH - 8))
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(container)
        self._proxy.setZValue(4)
        # NodeView.layout() will reposition this below the port section.
        self._proxy.setPos(8, Theme.NODE_HEADER_HEIGHT + 8)

    def body_height_hint(self) -> float:
        # Just return the proxy's preferred height; NodeView.layout combines
        # this with the port section height.
        if self._proxy and self._proxy.widget():
            return float(self._proxy.widget().sizeHint().height())
        return self.MIN_BODY_HEIGHT

    # ── actions ───────────────────────────────────────────────────────

    def _resolve_static_target_path(self, port_name: str) -> Optional[str]:
        """If the node's named port has a static link attached, return
        the path of the *other* endpoint — that's where Read/Write/Auto
        should actually do their I/O. Returns None if no static link.

        Direction-tolerant: we don't care whether the user drew the link
        as `source → my.port` or `my.port → source`, because for a
        visual link "this port is linked to that one" is symmetric. The
        caller decides what read vs write means.
        """
        port = self.node.get_port(port_name)
        if port is None:
            return None
        for conn in port.connections:
            if not getattr(conn, "is_static", False):
                continue
            other = conn.target if conn.source is port else conn.source
            if other.path:
                return other.path
        return None

    def _effective_read_pipe(self) -> Optional[Pipe]:
        """Pipe to actually read from when Read or Auto is triggered.
        Prefers a static link on this node's `in` (the linked port);
        falls back to this node's own `in` for the unwired case."""
        linked = self._resolve_static_target_path("in")
        if linked is not None:
            return Pipe(linked, self.operator.worker)
        return self._in_pipe

    def _effective_write_pipe(self) -> Optional[Pipe]:
        """Pipe to actually write to when Write is triggered. Prefers
        a static link on this node's `OUT`; falls back to own `OUT`."""
        linked = self._resolve_static_target_path("OUT")
        if linked is not None:
            return Pipe(linked, self.operator.worker)
        return self._out_pipe

    def _on_read_clicked(self) -> None:
        pipe = self._effective_read_pipe()
        if pipe is None:
            return
        self._status("reading…")
        pipe.read_async(self._on_read_done)

    def _on_read_done(self, result) -> None:
        if isinstance(result, ReadError):
            self._status(f"read error: {result.cause}", error=True)
            return
        data: bytes = result or b""
        self._text_edit.setPlainText(data.decode("utf-8", errors="replace"))
        self._status(f"read {len(data)} bytes")

    def _on_write_clicked(self) -> None:
        pipe = self._effective_write_pipe()
        if pipe is None:
            return
        data = self._text_edit.toPlainText().encode("utf-8")
        self._status("writing…")
        pipe.write_async(data, on_done=self._on_write_done)

    def _on_write_done(self, result) -> None:
        if isinstance(result, Exception):
            self._status(f"write error: {result}", error=True)
            return
        n = int(result) if isinstance(result, int) else 0
        self._status(f"wrote {n} bytes")

    def _on_auto_toggled(self, enabled: bool) -> None:
        if enabled:
            pipe = self._effective_read_pipe()
            if pipe is None:
                self._auto_check.setChecked(False)
                return
            # Pipe.subscribe picks STREAM (uppercase) or POLL (lowercase)
            # automatically based on the linked path. The static link
            # points us at the source port (e.g. `ctl` is lowercase →
            # POLL); the subscription path inherits that.
            self._sub = pipe.subscribe(
                self._on_chunk,
                on_error=lambda e: self._status(str(e), error=True),
            )
            self._status("subscribed")
        else:
            if self._sub is not None:
                self._sub.stop()
                self._sub = None
            self._status("")

    def _on_chunk(self, data: bytes) -> None:
        # Replace contents on each chunk. For STREAM sources, each chunk
        # is one complete generation. For POLL sources, dedupe means we
        # only get called on change.
        self._text_edit.setPlainText(data.decode("utf-8", errors="replace"))

    def _status(self, msg: str, *, error: bool = False) -> None:
        if self._status_label is None:
            return
        kind = "error" if error else "default"
        self._status_label.setStyleSheet(_qss_status_label(kind))
        self._status_label.setText(msg)

    def cleanup(self) -> None:
        if self._sub is not None:
            self._sub.stop()
            self._sub = None


# ═══════════════════════════════════════════════════════════════════════
# ─── DebugNodeView ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# A node with N input ports and an embedded log view. Each input port
# subscribes to its own file (<node_dir>/in_<i>); when data arrives,
# it's appended to the log view with a header showing which port
# produced it.


class DebugNodeView(NodeView):
    """Debug sink — N inputs, log view."""

    DEFAULT_WIDTH = 380.0
    MIN_BODY_HEIGHT = 260.0
    MAX_LOG_BLOCKS = 500
    PER_MSG_CAP = 4096
    # Overlap only kicks in when the node has ≤2 inputs (the layout()
    # guard); a debug sink wired to many outputs keeps the dedicated
    # port band so its stacked input labels stay legible.
    BODY_OVERLAPS_PORTS = True

    def __init__(self, node: Node, operator: "Operator"):
        super().__init__(node, operator)
        self._subs: List[Subscription] = []
        self._proxy: Optional[QGraphicsProxyWidget] = None
        self._log_view: Optional[QTextEdit] = None
        self._status_label: Optional[QLabel] = None
        self._msg_count = 0

    # ── build ─────────────────────────────────────────────────────────

    def build_body(self) -> None:
        container = QWidget()
        container.setAttribute(Qt.WA_TranslucentBackground)
        container.setStyleSheet("background: transparent;")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # Top row: input count + status
        row = QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"{len(self.node.inputs)} inputs")
        lbl.setStyleSheet(_qss_status_label())
        row.addWidget(lbl)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_qss_status_label())
        row.addWidget(self._status_label)
        row.addStretch()
        v.addLayout(row)

        # Log
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(Theme.FONT_CODE_SMALL)
        self._log_view.setMinimumHeight(180)
        self._log_view.setStyleSheet(_qss_text_edit())
        self._log_view.setPlaceholderText(
            "Waiting for input… wire any output port into this node.")
        v.addWidget(self._log_view)

        container.setFixedWidth(int(self.DEFAULT_WIDTH - 8))
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(container)
        self._proxy.setZValue(4)
        self._proxy.setPos(8, Theme.NODE_HEADER_HEIGHT + 8)

        # Subscribe to input ports — but skip `ctl`, which is for
        # control commands, not data flow. The server creates debug
        # nodes with ports `ctl`, `in_0`, `in_1`, ... `in_<N-1>`; we
        # want the in_* ones.
        for port in self.node.inputs:
            if port.name == "ctl":
                continue
            self._subscribe_port(port)

    def body_height_hint(self) -> float:
        if self._proxy and self._proxy.widget():
            return float(self._proxy.widget().sizeHint().height())
        return self.MIN_BODY_HEIGHT

    # ── subscriptions ─────────────────────────────────────────────────

    def _subscribe_port(self, port: Port) -> None:
        pipe = Pipe(port.path, self.operator.worker)
        sub = pipe.subscribe(
            lambda data, name=port.name: self._on_input(name, data),
            on_error=lambda e: self._on_error(port.name, e),
        )
        self._subs.append(sub)

    def _on_input(self, port_name: str, data: bytes) -> None:
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        if len(text) > self.PER_MSG_CAP:
            text = text[: self.PER_MSG_CAP] + " …[truncated]"
        self._append_log(port_name, text)

    def _on_error(self, port_name: str, err: ReadError) -> None:
        if self._status_label:
            self._status_label.setText(f"{port_name}: {err.cause}")

    def _append_log(self, tag: str, content: str) -> None:
        if self._log_view is None:
            return
        self._msg_count += 1
        separator = "── " + tag + " " + "─" * max(1, 60 - len(tag) - 4)
        block = f"{separator}\n{content}\n"
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(block)
        # Trim from the top if we're past MAX_LOG_BLOCKS.
        doc = self._log_view.document()
        while doc.blockCount() > self.MAX_LOG_BLOCKS:
            trim = self._log_view.textCursor()
            trim.movePosition(QTextCursor.Start)
            trim.select(QTextCursor.LineUnderCursor)
            trim.removeSelectedText()
            trim.deleteChar()
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
        if self._status_label:
            self._status_label.setText(f"{self._msg_count} msg")

    def cleanup(self) -> None:
        for sub in self._subs:
            sub.stop()
        self._subs.clear()


# ═══════════════════════════════════════════════════════════════════════
# ─── MediaNodeView ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# A node with one input port. Subscribes to it (POLL+dedupe with a
# cheap hash key — see `dedupe_key` below), detects the media format
# from the bytes, and renders it in an embedded preview.
#
# The `dedupe_key` exploits Pipe's per-port tuning: for a 4MB image we
# hash a 130-byte tuple instead of the whole payload.


MAX_PREVIEW_SIDE = 320


def detect_media_format(data: bytes) -> str:
    if len(data) < 12:
        return "unknown"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    if data[4:8] == b"ftyp":
        return "mp4"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    return "unknown"


class MediaNodeView(NodeView):
    """Media node — input port, embedded preview."""

    DEFAULT_WIDTH = 340.0
    MIN_BODY_HEIGHT = 200.0
    BODY_OVERLAPS_PORTS = True

    def __init__(self, node: Node, operator: "Operator"):
        super().__init__(node, operator)
        self._sub: Optional[Subscription] = None
        self._proxy: Optional[QGraphicsProxyWidget] = None
        self._display: Optional[QLabel] = None
        self._status_label: Optional[QLabel] = None
        self._video_tmpfile: Optional[str] = None
        self._media_player = None
        self._video_sink = None

    def build_body(self) -> None:
        container = QWidget()
        container.setAttribute(Qt.WA_TranslucentBackground)
        container.setStyleSheet("background: transparent;")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        self._display = QLabel("⌧  no input")
        self._display.setAlignment(Qt.AlignCenter)
        self._display.setMinimumHeight(160)
        self._display.setStyleSheet(f"""
            QLabel {{
                background-color: {Theme.EDIT_BG.name(QColor.HexArgb)};
                color: {Theme.TEXT_SECONDARY.name()};
                border: 1px solid {Theme.EDIT_BORDER.name(QColor.HexArgb)};
                border-radius: 4px;
                padding: 8px;
                font-family: {Theme.FONT_FAMILY_UI};
                font-size: 10px;
            }}
        """)
        v.addWidget(self._display)

        self._status_label = QLabel("ready")
        self._status_label.setStyleSheet(_qss_status_label())
        v.addWidget(self._status_label)

        container.setFixedWidth(int(self.DEFAULT_WIDTH - 8))
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(container)
        self._proxy.setZValue(4)
        self._proxy.setPos(8, Theme.NODE_HEADER_HEIGHT + 8)

        # Subscribe to the `in` port specifically. We can't use
        # `inputs[0]` here — port order depends on filesystem listing
        # (ctl, errors, help, etc. all classify as INPUT), and the
        # first one is usually `ctl`. Subscribing to ctl makes for
        # very noisy I/O against a port that doesn't carry image data.
        in_port = self.node.get_port("in")
        if in_port is not None:
            in_pipe = Pipe(in_port.path, self.operator.worker)
            # Cheap dedupe: length + 64-byte prefix + 64-byte suffix.
            # For a 4MB image that's 130 bytes hashed per tick instead
            # of the full 4MB. Collisions are astronomically rare for
            # bona-fide content changes (different image = different
            # length or different pixel data near the edges).
            self._sub = in_pipe.subscribe(
                self._on_input,
                dedupe_key=lambda d: (len(d), d[:64], d[-64:]),
                on_error=lambda e: self._set_status(
                    f"error: {e.cause}", error=True),
            )

        # Output port (passthrough). We don't subscribe to it; we write
        # to it from _on_input.
        out_port = self.node.get_port("OUT")
        self._out_pipe = (Pipe(out_port.path, self.operator.worker)
                         if out_port is not None else None)

    def body_height_hint(self) -> float:
        if self._proxy and self._proxy.widget():
            return float(self._proxy.widget().sizeHint().height())
        return self.MIN_BODY_HEIGHT

    def _on_input(self, data: bytes) -> None:
        if not data:
            return
        fmt = detect_media_format(data)
        # Diagnostic — visible at every chunk so we can see what's
        # actually flowing through the route. If you're seeing 39 bytes
        # of non-image data, this print will show you the actual content.
        preview = data[:80]
        print(f"MediaNode[{self.node.node_id}]: {len(data)}B fmt={fmt} "
              f"head={preview!r}")
        if fmt in ("jpeg", "png", "gif", "webp", "bmp"):
            self._render_image(data, fmt)
        elif fmt in ("mp4", "webm"):
            self._render_video_poster(data, fmt)
        else:
            self._render_unknown(data, fmt)

        # Passthrough write to OUT.
        if self._out_pipe is not None:
            self._out_pipe.write_async(data)

    def _render_image(self, data: bytes, fmt: str) -> None:
        pix = QPixmap()
        if not pix.loadFromData(data):
            self._render_unknown(data, "load-failed")
            return
        scaled = pix.scaled(
            MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._display.setPixmap(scaled)
        self._display.setText("")
        self._set_status(f"{fmt} • {pix.width()}×{pix.height()} • "
                         f"{len(data):,}B")

    def _render_video_poster(self, data: bytes, fmt: str) -> None:
        if not _MULTIMEDIA_OK:
            self._display.setPixmap(QPixmap())
            self._display.setText(
                f"🎬 {fmt}\n{len(data):,} bytes\n\n"
                "(install PySide6 multimedia for preview)")
            return
        if self._media_player is None:
            self._video_sink = QVideoSink(self)
            self._video_sink.videoFrameChanged.connect(self._on_video_frame)
            self._media_player = QMediaPlayer(self)
            self._media_player.setVideoSink(self._video_sink)
        if self._video_tmpfile is None:
            fd, path = tempfile.mkstemp(
                prefix=f"op_{self.node.node_id}_", suffix=f".{fmt}")
            os.close(fd)
            self._video_tmpfile = path
        try:
            with open(self._video_tmpfile, "wb") as f:
                f.write(data)
            self._media_player.setSource(
                QUrl.fromLocalFile(self._video_tmpfile))
            self._media_player.play()
            self._media_player.pause()  # grab first frame and hold
        except Exception as e:
            self._display.setText(f"🎬 {fmt}\nplayback error:\n{e}")
        self._set_status(f"{fmt} • {len(data):,}B")

    @Slot(object)
    def _on_video_frame(self, frame) -> None:
        try:
            img = frame.toImage()
            if img.isNull():
                return
            pix = QPixmap.fromImage(img).scaled(
                MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._display.setPixmap(pix)
            self._display.setText("")
        except Exception:
            pass

    def _render_unknown(self, data: bytes, fmt: str) -> None:
        self._display.setPixmap(QPixmap())
        preview = data[:80].decode("utf-8", errors="replace")
        self._display.setText(
            f"⌧ {fmt}\n{len(data):,} bytes\n\n{preview[:120]}…")
        self._set_status(f"{fmt} • {len(data):,}B")

    def _set_status(self, text: str, *, error: bool = False) -> None:
        if self._status_label is None:
            return
        kind = "error" if error else "default"
        self._status_label.setStyleSheet(_qss_status_label(kind))
        self._status_label.setText(text)

    def cleanup(self) -> None:
        if self._sub is not None:
            self._sub.stop()
            self._sub = None
        if self._media_player is not None:
            try:
                self._media_player.stop()
            except Exception:
                pass
        if self._video_tmpfile:
            try:
                os.unlink(self._video_tmpfile)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════
# ─── AgentNodeView / GenericNodeView ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# `AgentNodeView`: visualizes an agent under /n/llm/agents/<name>.
# Subscribes to the agent's `ctl` file (POLL+dedupe) to track its
# status in a small label, and displays one port per file.
#
# `GenericNodeView`: fallback for any other directory the operator
# scans (terminals, scene, anything under /n/<m>/nodes/ that doesn't
# match a more specific kind). Just headers + ports, no embedded widget.


class AgentNodeView(NodeView):
    """View for /n/llm/agents/<name>. Shows ctl status under the header."""

    DEFAULT_WIDTH = 260.0
    MIN_BODY_HEIGHT = 80.0

    def __init__(self, node: Node, operator: "Operator"):
        super().__init__(node, operator)
        self._ctl_sub: Optional[Subscription] = None
        self._proxy: Optional[QGraphicsProxyWidget] = None
        self._status_label: Optional[QLabel] = None

    def build_body(self) -> None:
        # Small status label, fed by a subscription to the ctl file.
        container = QWidget()
        container.setAttribute(Qt.WA_TranslucentBackground)
        container.setStyleSheet("background: transparent;")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 3, 4, 3)
        v.setSpacing(2)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_qss_status_label())
        self._status_label.setWordWrap(True)
        v.addWidget(self._status_label)

        container.setFixedWidth(int(self.DEFAULT_WIDTH - 8))
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(container)
        self._proxy.setZValue(4)
        self._proxy.setPos(8, Theme.NODE_HEADER_HEIGHT + 4)

        # NOTE: previously this auto-subscribed to the ctl port to show
        # live agent status. Disabled because each subscription opens a
        # Python-side fd against the 9p mount, and multiple Python fds
        # against the parser's mount session can crash the rio backend.
        # If/when we want live status back, do it via subprocess (cat in
        # a QProcess), the way Routes does. For now the agent renders
        # with header+ports only — same as the original v1.
        # ctl_port = self.node.get_port("ctl")
        # if ctl_port is not None:
        #     ctl_pipe = Pipe(ctl_port.path, self.operator.worker)
        #     self._ctl_sub = ctl_pipe.subscribe(
        #         self._on_ctl_change, mode=SubscribeMode.POLL,
        #     )

    def body_height_hint(self) -> float:
        # Two lines of status max + port rows.
        port_rows = max(len(self.node.inputs), len(self.node.outputs), 1)
        port_h = Theme.PORT_MARGIN_TOP + port_rows * Theme.PORT_SPACING + 2
        return max(self.MIN_BODY_HEIGHT, port_h + 32)

    def _on_ctl_change(self, data: bytes) -> None:
        if self._status_label is None:
            return
        text = data.decode("utf-8", errors="replace").strip()
        # Show just the first line, truncated.
        first = text.split("\n", 1)[0] if text else ""
        if len(first) > 60:
            first = first[:57] + "…"
        self._status_label.setText(first)

    def cleanup(self) -> None:
        if self._ctl_sub is not None:
            self._ctl_sub.stop()
            self._ctl_sub = None


class GenericNodeView(NodeView):
    """Fallback view for any node we don't have a specific renderer for —
    terminals, scene, unknown directories under /n/<m>/nodes/.

    Just header + ports. The operator still wires up routes through them
    like any other node.
    """

    DEFAULT_WIDTH = 220.0
    MIN_BODY_HEIGHT = 50.0

    def body_height_hint(self) -> float:
        port_rows = max(len(self.node.inputs), len(self.node.outputs), 1)
        return Theme.PORT_MARGIN_TOP + port_rows * Theme.PORT_SPACING + 2


# ═══════════════════════════════════════════════════════════════════════
# ─── BashNodeView / PythonNodeView ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Two nodes that take input bytes, run a command/expression, and emit
# the result. The actual execution happens server-side: the operator
# writes the command to <node_dir>/cmd (or /code) and reads from
# <node_dir>/OUT.
#
# This puts both bash and python on the same I/O footing as every other
# node — write a string somewhere, subscribe to a result file. No
# threading, no subprocess.run, no piping. The filesystem's job to
# actually run the command; the operator's job to display the I/O.


class _ExecNodeBase(NodeView):
    """Shared body for Bash and Python nodes.

    Layout:
      ┌─ header ──────────────────────────────────────┐
      │○ multi-line code editor  (ports ride this edge)│
      │  ┌─────────────────────────────────────────┐  │
      │  │  $ command or  py expression            │  │
      │  └─────────────────────────────────────────┘  │
      │  [▶] [⌫]   ← compact icon buttons             │
      │  output / stderr viewer                       │
      │  ┌─────────────────────────────────────────┐  │
      │  │  result lines …                         │  │
      │  └─────────────────────────────────────────┘  │
      │  status: ok • 32B out                         │
      └───────────────────────────────────────────────┘

    All execution happens server-side: the operator writes the source to
    <node_dir>/cmd (or /code) and reads the result from <node_dir>/OUT.
    The view's job is editor + display + plumbing.

    What this picks up from operator_stable's BashNode/PythonNode:
      - Multi-line code editor (was a single-line QLineEdit, which is
        useless for anything beyond `ls`)
      - A real output viewer with monospace font and scrollbars
      - A dedicated Run button styled as a primary action
      - Live subscription to the `cmd` port so an upstream agent wiring
        into `cmd` updates the editor (with markdown-fence stripping)
      - A status pill that switches color for ok / busy / error
    """

    DEFAULT_WIDTH = 380.0
    MIN_BODY_HEIGHT = 240.0
    BODY_OVERLAPS_PORTS = True

    COMMAND_PORT_NAME: str = "cmd"
    OUTPUT_PORT_NAME: str = "OUT"
    ERROR_PORT_NAME: str = "ERR"
    COMMAND_PLACEHOLDER: str = "command"
    PROMPT_GLYPH: str = "$"
    DEFAULT_COMMAND: str = ""
    EDITOR_MIN_HEIGHT: int = 70
    EDITOR_MAX_HEIGHT: int = 140
    OUTPUT_MIN_HEIGHT: int = 90
    # Execution timeout, in seconds. Subclasses can override.
    TIMEOUT_SEC: float = 30.0
    # Subclasses set this to the executor function. Signature:
    #   fn(command: str, stdin: bytes, timeout: float) -> dict
    #     where dict has keys returncode, stdout, stderr, timed_out.
    # The default is a stub that errors out; BashNodeView and
    # PythonNodeView override with the real implementations.
    EXECUTOR: Optional[Callable[..., Dict[str, Any]]] = None

    def __init__(self, node: Node, operator: "Operator"):
        super().__init__(node, operator)
        self._out_sub: Optional[Subscription] = None
        self._err_sub: Optional[Subscription] = None
        self._cmd_sub: Optional[Subscription] = None  # incoming cmd updates
        self._in_sub: Optional[Subscription] = None   # incoming data updates
        self._proxy: Optional[QGraphicsProxyWidget] = None
        self._cmd_edit: Optional[QTextEdit] = None
        self._output_view: Optional[QTextEdit] = None
        self._status_label: Optional[QLabel] = None
        self._run_btn: Optional[QPushButton] = None
        self._clear_btn: Optional[QPushButton] = None
        # Pipes we OWN (for writing results back). Distinct from the
        # *subscriptions* above, which are for read-back via the FS.
        self._cmd_pipe: Optional[Pipe] = None
        self._out_pipe: Optional[Pipe] = None
        self._err_pipe: Optional[Pipe] = None
        self._in_pipe: Optional[Pipe] = None
        self._has_stderr: bool = False
        self._suppress_cmd_subscribe: bool = False
        # Gate: drop OUT/ERR subscription chunks that arrive while we
        # have an exec in flight, because we update the UI directly from
        # the exec callback. Without this gate the FS subscription would
        # double-paint the same data after we wrote it.
        self._exec_in_flight: bool = False
        # Source-vs-transformer mode. The `in` port being connected to an
        # upstream output flips us into "transformer" mode: input bytes
        # are auto-piped into the executor on every change, the Run
        # button hides. With `in` unconnected we stay in "source" mode
        # where Run executes with empty stdin. Mirrors operator_stable's
        # BashNode/PythonNode (`_run_btn.show()` when no `in` connection).
        self._in_connected: bool = False
        # Cache of the last input bytes we ran with — lets a `cmd` port
        # update force a re-run on the existing input (operator_stable
        # does the same via `_last_input_hash = None` + re-poll).
        self._last_input_bytes: bytes = b""

    # ── build ─────────────────────────────────────────────────────────

    def build_body(self) -> None:
        container = QWidget()
        container.setAttribute(Qt.WA_TranslucentBackground)
        container.setStyleSheet("background: transparent;")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # ── prompt label (sits above the editor) ────────────────────
        prompt = QLabel(f"{self.PROMPT_GLYPH}  {self.COMMAND_PLACEHOLDER}")
        prompt.setStyleSheet(_qss_status_label())
        v.addWidget(prompt)

        # ── multi-line code editor (Ctrl/Cmd+Enter = run) ───────────
        self._cmd_edit = _CodeTextEdit(self._on_run)
        self._cmd_edit.setPlaceholderText(self.COMMAND_PLACEHOLDER)
        self._cmd_edit.setFont(Theme.FONT_CODE)
        self._cmd_edit.setMinimumHeight(self.EDITOR_MIN_HEIGHT)
        self._cmd_edit.setMaximumHeight(self.EDITOR_MAX_HEIGHT)
        self._cmd_edit.setStyleSheet(_qss_text_edit())
        self._cmd_edit.setLineWrapMode(QTextEdit.NoWrap)
        self._cmd_edit.setTabChangesFocus(False)
        if self.DEFAULT_COMMAND:
            self._cmd_edit.setPlainText(self.DEFAULT_COMMAND)
        v.addWidget(self._cmd_edit)

        # ── button row ──────────────────────────────────────────────
        # Compact icon buttons: Run (▶, ink) and Clear (⌫). The status
        # pill rides the same row, right-aligned, instead of taking its
        # own line below the output viewer.
        btns = QHBoxLayout()
        btns.setSpacing(5)
        btns.setContentsMargins(0, 0, 0, 0)

        _ICON_BTN = QSize(26, 22)

        self._run_btn = QPushButton("▶")
        self._run_btn.setToolTip("Run  (Ctrl/Cmd+Enter)")
        self._run_btn.setFixedSize(_ICON_BTN)
        self._run_btn.setCursor(Qt.PointingHandCursor)
        self._run_btn.setStyleSheet(_qss_icon_button("accent"))
        self._run_btn.clicked.connect(self._on_run)
        btns.addWidget(self._run_btn)

        self._clear_btn = QPushButton("⌫")
        self._clear_btn.setToolTip("Clear output")
        self._clear_btn.setFixedSize(_ICON_BTN)
        self._clear_btn.setCursor(Qt.PointingHandCursor)
        self._clear_btn.setStyleSheet(_qss_icon_button())
        self._clear_btn.clicked.connect(self._on_clear)
        btns.addWidget(self._clear_btn)

        btns.addStretch()

        self._status_label = QLabel("ready")
        self._status_label.setStyleSheet(_qss_status_label())
        self._status_label.setWordWrap(False)
        self._status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        btns.addWidget(self._status_label)
        v.addLayout(btns)

        # ── output viewer ──────────────────────────────────────────
        self._output_view = QTextEdit()
        self._output_view.setReadOnly(True)
        self._output_view.setFont(Theme.FONT_CODE_SMALL)
        self._output_view.setMinimumHeight(self.OUTPUT_MIN_HEIGHT)
        self._output_view.setPlaceholderText(
            "no output yet — Run the command above")
        self._output_view.setStyleSheet(_qss_text_edit())
        self._output_view.setLineWrapMode(QTextEdit.WidgetWidth)
        v.addWidget(self._output_view)

        container.setFixedWidth(int(self.DEFAULT_WIDTH - 8))
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(container)
        self._proxy.setZValue(4)
        # NodeView.layout() will reposition this below the port section.
        self._proxy.setPos(8, Theme.NODE_HEADER_HEIGHT + 8)

        # ── resolve pipes ──────────────────────────────────────────
        cmd_port = self.node.get_port(self.COMMAND_PORT_NAME)
        if cmd_port is not None:
            self._cmd_pipe = Pipe(cmd_port.path, self.operator.worker)
            # Subscribe to cmd updates from upstream so wired agents can
            # drive the editor live. Drop our own writes (Ctrl+Enter or
            # Run button) via _suppress_cmd_subscribe.
            try:
                self._cmd_sub = self._cmd_pipe.subscribe(
                    self._on_cmd_pipe_chunk,
                    on_error=lambda e: None,  # ignore — port may be POLL/STREAM
                )
            except Exception:
                # If the pipe doesn't exist yet (rare), subscriptions can
                # raise. Continue without — Run still works via write_async.
                self._cmd_sub = None

        out_port = self.node.get_port(self.OUTPUT_PORT_NAME)
        if out_port is not None:
            self._out_pipe = Pipe(out_port.path, self.operator.worker)
            # OUT is uppercase → STREAM by default. Each EOF marks one
            # completed command.
            self._out_sub = self._out_pipe.subscribe(self._on_output)

        err_port = self.node.get_port(self.ERROR_PORT_NAME)
        if err_port is not None:
            self._err_pipe = Pipe(err_port.path, self.operator.worker)
            self._err_sub = self._err_pipe.subscribe(self._on_error)

        # ── data input port (`in`) ─────────────────────────────────
        # Holds the Pipe so we can subscribe/unsubscribe as the port's
        # connection state flips. We do NOT subscribe yet — that only
        # happens when an upstream wire actually attaches to `in`
        # (see _refresh_input_mode). With no wire, the node is a
        # source: Run button visible, executor stdin = b"".
        in_port = self.node.get_port("in")
        if in_port is not None:
            self._in_pipe = Pipe(in_port.path, self.operator.worker)

        # Hook the graph's connection signals so we can flip mode the
        # instant a wire is dropped onto / pulled off the `in` port.
        # The connections are filtered inside the handlers (only ones
        # involving our `in` port matter).
        try:
            self.operator.graph.connection_added.connect(
                self._on_graph_connection_added)
            self.operator.graph.connection_removed.connect(
                self._on_graph_connection_removed)
        except Exception:
            # Operator/graph may be missing in unit tests — degrade
            # gracefully to source mode (Run always available).
            pass

        # Initial mode setup. If the model already shows a connection
        # on `in` (e.g. we're being re-built after a refresh), this
        # subscribes immediately.
        self._refresh_input_mode()

    def body_height_hint(self) -> float:
        if self._proxy and self._proxy.widget():
            return float(self._proxy.widget().sizeHint().height())
        return self.MIN_BODY_HEIGHT

    # ── actions ───────────────────────────────────────────────────────

    def _current_command(self) -> str:
        """Pull the command from the editor, trimmed of trailing whitespace
        but preserving internal newlines (so multi-line scripts work)."""
        if self._cmd_edit is None:
            return ""
        return self._cmd_edit.toPlainText().rstrip()

    def _on_run(self) -> None:
        """Run button click / Ctrl+Enter. In source mode (no `in` wire)
        runs with empty stdin. In transformer mode this is a manual
        re-run on the most recently received input bytes — useful when
        you've edited the command and want to re-process buffered data
        without waiting for the upstream to re-emit. Mirrors the
        "Run now" context-menu entry on operator_stable's BashNode.
        """
        stdin = self._last_input_bytes if self._in_connected else b""
        self._dispatch_executor(stdin)

    def _dispatch_executor(self, stdin_bytes: bytes) -> None:
        """Read the editor, fire the executor on FSWorker, route the
        result through _on_exec_done. Single dispatch point so the Run
        button, Ctrl+Enter, the `in`-port chunk handler, and the `cmd`
        re-run all funnel through the same code path."""
        cmd = self._current_command()
        if not cmd:
            self._set_status("empty command", error=True)
            return

        executor = self.EXECUTOR
        if executor is None:
            self._set_status("no executor configured (subclass bug)",
                             error=True)
            return

        # IMPORTANT: do NOT write to the `cmd` pipe here. The 9p server
        # has a stub handler that responds to cmd-writes by emitting
        # "(no executor yet) cmd=<cmd>" on OUT. That stub races with our
        # own subprocess.Popen write to OUT and the visible output
        # alternates between the real result and the stub's placeholder.
        # We bypass the cmd port entirely — the operator runs the command
        # locally and publishes the result directly to OUT and ERR.

        self._has_stderr = False
        self._exec_in_flight = True
        first_line = cmd.splitlines()[0] if cmd else ""
        preview = first_line[:48] + "…" if len(first_line) > 48 else first_line
        size_tag = f" ◂ {len(stdin_bytes):,}B" if stdin_bytes else ""
        self._set_status(f"running… {preview}{size_tag}", busy=True)

        # Dispatch to the FSWorker thread so a long command doesn't
        # freeze the Qt event loop. The callback runs back on the Qt
        # thread (FSWorker contract — see operator_stable's _on_*
        # callbacks, which manipulate Qt widgets directly).
        try:
            self.operator.worker.run_async(
                executor, cmd, stdin_bytes, float(self.TIMEOUT_SEC),
                on_done=self._on_exec_done,
            )
        except Exception as e:
            # Fallback: run synchronously. Shouldn't happen — operator
            # always provides a worker.
            self._set_status(f"dispatch failed: {e}", error=True)
            result = executor(cmd, stdin_bytes, float(self.TIMEOUT_SEC))
            self._on_exec_done(result)

    def _on_exec_done(self, result: Any) -> None:
        """Completion callback for the executor. Writes stdout to the OUT
        pipe (so downstream nodes wired to OUT pick it up) and stderr to
        ERR, and paints the local output viewer."""
        self._exec_in_flight = False

        # If run_async raised, the worker may pass the exception as the
        # "result". Surface it.
        if isinstance(result, BaseException):
            self._set_status(f"crash: {result}", error=True)
            if self._output_view is not None:
                self._output_view.setPlainText(str(result))
            return
        if not isinstance(result, dict):
            self._set_status(f"bad result type: {type(result).__name__}",
                             error=True)
            return

        stdout: bytes = result.get("stdout", b"") or b""
        stderr: bytes = result.get("stderr", b"") or b""
        rc: int = int(result.get("returncode", -1))
        timed_out: bool = bool(result.get("timed_out", False))

        # 1) Publish to OUT and ERR pipes so downstream wires see it.
        #    Always publish stdout (even on rc != 0) — downstream may
        #    still want partial output.
        if self._out_pipe is not None:
            try:
                self._out_pipe.write_async(stdout)
            except Exception:
                pass
        if self._err_pipe is not None and stderr:
            try:
                self._err_pipe.write_async(stderr)
            except Exception:
                pass

        # 2) Paint the output viewer directly. Don't wait for the OUT
        #    subscription to round-trip our own write — that path adds
        #    latency and (in some 9p configs) may never fire because
        #    the writer doesn't notify itself.
        out_text = stdout.decode("utf-8", errors="replace")
        err_text = stderr.decode("utf-8", errors="replace")
        if self._output_view is not None:
            if err_text.strip():
                if out_text:
                    self._output_view.setPlainText(
                        out_text + "\n--- stderr ---\n" + err_text)
                else:
                    self._output_view.setPlainText(
                        "--- stderr ---\n" + err_text)
                self._has_stderr = True
            else:
                self._output_view.setPlainText(out_text)
                self._has_stderr = False

        # 3) Status pill.
        if timed_out:
            self._set_status(f"timeout after {self.TIMEOUT_SEC}s",
                             error=True)
        elif rc == 0:
            self._set_status(f"ok • {len(stdout):,}B out", ok=True)
        else:
            err_preview = err_text.strip().splitlines()[0] if err_text.strip() else ""
            err_preview = err_preview[:60] + "…" if len(err_preview) > 60 else err_preview
            self._set_status(f"rc={rc} • {err_preview}", error=True)

    def _on_clear(self) -> None:
        if self._output_view is not None:
            self._output_view.setPlainText("")
        self._has_stderr = False
        self._set_status("cleared")

    # ── pipe handlers ─────────────────────────────────────────────────

    def _on_cmd_pipe_chunk(self, data: bytes) -> None:
        """Called when an upstream wire writes to our `cmd` port. Replace
        the editor contents with the (fence-stripped) text. Skips the
        echo of our own Run-button write.

        If we're in transformer mode (an upstream wire is feeding `in`),
        also re-execute against the cached input bytes — matches
        operator_stable's behavior of clearing `_last_input_hash` after a
        cmd-port update so the next poll re-runs on the existing data.
        """
        if self._suppress_cmd_subscribe:
            self._suppress_cmd_subscribe = False
            return
        if self._cmd_edit is None:
            return
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return
        stripped = _strip_code_fence(text).strip()
        if not stripped:
            return
        # Avoid an infinite loop if the editor's text already matches.
        if stripped == self._cmd_edit.toPlainText().strip():
            return
        self._cmd_edit.setPlainText(stripped)
        first = stripped.splitlines()[0] if stripped else ""
        preview = first[:40] + "…" if len(first) > 40 else first
        self._set_status(f"cmd ← port • {preview}")
        # Re-run on the cached input if we're a live transformer. In
        # source mode the user still has to click Run — that's a deliberate
        # UX choice (don't fire side-effecting bash commands just because
        # an LLM streamed an updated prompt).
        if self._in_connected:
            self._dispatch_executor(self._last_input_bytes)

    def _on_output(self, data: bytes) -> None:
        """Called when the OUT pipe receives a chunk from someone else
        (e.g. an external writer to <node_dir>/OUT). When WE produced the
        chunk via _on_exec_done, the FS round-trip will also fire this
        callback — but _on_exec_done already painted the view with the
        same bytes, so it's redundant. The _exec_in_flight gate is
        already cleared by then; the FS write echo is harmless. We keep
        this path so external writers (a streaming agent, say) still
        update the view."""
        if self._output_view is None:
            return
        text = data.decode("utf-8", errors="replace")
        # Defensive filter: there is a 9p server-side stub somewhere in
        # the backend that emits "(no executor yet) cmd=…" on OUT when
        # the cmd pipe is touched. We don't write to cmd ourselves, but
        # if anything else does (e.g. an upstream wire), the stub will
        # publish its placeholder string. Drop those — they're noise,
        # the real executor is local.
        if "(no executor yet)" in text:
            self._set_status(
                "ignored backend stub on OUT (no executor yet)", busy=True)
            return
        if self._has_stderr and self._output_view.toPlainText():
            # Fresh stdout for a new run — start clean.
            self._has_stderr = False
            self._output_view.setPlainText(text)
        else:
            self._output_view.setPlainText(text)
        # Don't overwrite a more informative status (e.g. "rc=1 • …")
        # produced by _on_exec_done.
        if "rc=" not in (self._status_label.text() if self._status_label else ""):
            self._set_status(f"ok • {len(data):,}B out", ok=True)

    def _on_error(self, data: bytes) -> None:
        if self._output_view is None:
            return
        text = data.decode("utf-8", errors="replace")
        if not text.strip():
            return
        self._has_stderr = True
        current = self._output_view.toPlainText()
        sep = "\n--- stderr ---\n"
        if current and sep.strip() not in current:
            self._output_view.setPlainText(current + sep + text)
        elif not current:
            self._output_view.setPlainText(sep.lstrip("\n") + text)
        # else: current already shows this stderr from _on_exec_done — skip
        first = text.strip().splitlines()[0] if text.strip() else ""
        preview = first[:60] + "…" if len(first) > 60 else first
        self._set_status(f"stderr: {preview}", error=True)

    # ── input-port (`in`) wiring ──────────────────────────────────────
    #
    # Source vs transformer is a runtime distinction driven by the
    # connection state of the `in` port. operator_stable's BashNode and
    # PythonNode encode the same idea via a 250ms poll timer that flips
    # `_run_btn.show()/hide()` and short-circuits the poll when the port
    # is unconnected. Here we use the graph's connection_added/removed
    # signals so the toggle is event-driven (no polling).

    def _is_in_port(self, port: Optional[Port]) -> bool:
        """True iff `port` is *our* `in` port."""
        if port is None:
            return False
        if port.node is not self.node:
            return False
        return port.name == "in"

    def _on_graph_connection_added(self, conn: "Connection") -> None:
        if self._is_in_port(conn.source) or self._is_in_port(conn.target):
            self._refresh_input_mode()

    def _on_graph_connection_removed(self, conn: "Connection") -> None:
        if self._is_in_port(conn.source) or self._is_in_port(conn.target):
            self._refresh_input_mode()

    def _in_port_has_connections(self) -> bool:
        in_port = self.node.get_port("in")
        if in_port is None:
            return False
        return bool(in_port.connections)

    def _refresh_input_mode(self) -> None:
        """Sync subscription + Run-button visibility to the current
        connection state of the `in` port. Idempotent — safe to call
        from build_body, the connection signals, or anywhere else
        where the wiring might have changed."""
        connected = self._in_port_has_connections()
        if connected == self._in_connected and self._in_sub is not None:
            return  # nothing to do — already in the right state

        if connected and self._in_pipe is not None:
            # Transformer mode: hide Run, subscribe to `in`. Each fresh
            # chunk drives an executor run. Pipe.subscribe already
            # dedupes (default dedupe=True), so identical bytes re-emitted
            # by the upstream don't cause spurious re-runs — matches
            # operator_stable's `_last_input_hash` check.
            self._in_connected = True
            if self._run_btn is not None:
                self._run_btn.hide()
            if self._in_sub is None:
                try:
                    self._in_sub = self._in_pipe.subscribe(
                        self._on_input_chunk,
                        on_error=lambda e: None,
                    )
                except Exception as e:
                    # Couldn't subscribe — degrade to source mode so the
                    # user still has *some* way to run the node.
                    self._in_sub = None
                    self._in_connected = False
                    if self._run_btn is not None:
                        self._run_btn.show()
                    self._set_status(
                        f"in subscribe failed: {e}", error=True)
                    return
            self._set_status("transformer mode • waiting for input")
        else:
            # Source mode: stop the subscription, show Run.
            self._in_connected = False
            if self._in_sub is not None:
                try:
                    self._in_sub.stop()
                except Exception:
                    pass
                self._in_sub = None
            self._last_input_bytes = b""
            if self._run_btn is not None:
                self._run_btn.show()
            self._set_status("source mode • Run to execute")

    def _on_input_chunk(self, data: bytes) -> None:
        """Upstream wrote bytes to our `in` port — pipe them into the
        executor as stdin/`input`. The subscription is configured with
        dedupe=True (the Pipe default), so a chunk that matches the
        previous one is suppressed at the subscription layer; this
        callback only fires for genuinely-new data."""
        if self._cmd_edit is None:
            return
        # Cache for re-runs (cmd-port updates, manual Run-now via menu).
        self._last_input_bytes = bytes(data)
        self._dispatch_executor(self._last_input_bytes)

    # ── status helper ─────────────────────────────────────────────────

    def _set_status(self, msg: str, *, ok: bool = False,
                    error: bool = False, busy: bool = False) -> None:
        if self._status_label is None:
            return
        if error:
            kind = "error"
        elif ok:
            kind = "ok"
        elif busy:
            kind = "busy"
        else:
            kind = "default"
        self._status_label.setStyleSheet(_qss_status_label(kind))
        self._status_label.setText(msg)

    # ── teardown ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        for sub in (self._out_sub, self._err_sub, self._cmd_sub,
                    self._in_sub):
            if sub is not None:
                try:
                    sub.stop()
                except Exception:
                    pass
        self._out_sub = self._err_sub = self._cmd_sub = None
        self._in_sub = None
        # Detach from graph signals — otherwise a stale view keeps
        # receiving connection_added/removed and crashes trying to use
        # already-freed Qt widgets.
        try:
            self.operator.graph.connection_added.disconnect(
                self._on_graph_connection_added)
        except Exception:
            pass
        try:
            self.operator.graph.connection_removed.disconnect(
                self._on_graph_connection_removed)
        except Exception:
            pass


def _strip_code_fence(text: str) -> str:
    """Strip a single ```lang ... ``` fence from LLM output if present.

    LLMs often wrap commands in markdown fences (```bash, ```python, etc.)
    even when asked for bare output. This unwraps them so the result is
    directly executable. If no fence is found, returns text unchanged.
    Copied from operator_stable's transformer-node logic.
    """
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.splitlines()
    if len(lines) < 2:
        return text
    # First line is ``` or ```lang — drop it.
    body = lines[1:]
    # Drop closing fence if present.
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body)


# ── client-side executors ──────────────────────────────────────────────
#
# The exec nodes run their commands HERE, in the operator process, rather
# than handing off to some server-side handler. The previous design wrote
# the command to <node_dir>/cmd and waited for a server to read, execute,
# and write back to OUT — but the server never did, so the OUT file just
# got a placeholder string. Running locally cuts out that round-trip:
#
#     editor → _run_bash_pipe → subprocess → write to OUT pipe
#                                          → write to ERR pipe
#
# Both helpers run on the FSWorker thread (via worker.run_async), so a
# long-running command can't freeze the Qt event loop. Timeout defends
# against `sleep 9999` and similar pathologies.
#
# Lifted almost verbatim from operator_stable.py's _run_bash_pipe and
# _eval_python_pipe — they were proven, this is just transplanting them
# into the new node-view layer.


BASH_DEFAULT_TIMEOUT_SEC = 30.0


def _run_bash_pipe(command: str, stdin_data: bytes,
                   timeout_sec: float) -> Dict[str, Any]:
    """Run a shell pipeline with bytes on stdin; collect stdout/stderr.

    Returns a dict with keys: returncode, stdout, stderr, timed_out.
    Designed to run on the FSWorker background thread.
    """
    try:
        proc = subprocess.Popen(
            command, shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(input=stdin_data,
                                              timeout=timeout_sec)
            return {
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except Exception:
                stdout, stderr = b"", b""
            return {
                "returncode": -1,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
            }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": b"",
            "stderr": str(e).encode("utf-8"),
            "timed_out": False,
        }


def _eval_python_pipe(expr: str,
                      input_bytes: bytes,
                      timeout_sec: float = 30.0) -> Dict[str, Any]:
    """Run a small Python program with `input` bound to `input_bytes`.

    Returns a dict shaped the same as _run_bash_pipe so the caller can use
    a single completion path: {returncode, stdout, stderr, timed_out}.

    `timeout_sec` is accepted but NOT enforced — this is an in-process
    exec/eval, not a subprocess. The argument is here purely so the
    Python executor has the same arity as the Bash one, letting
    _ExecNodeBase dispatch them through a single call site.

    Output rules (same as operator_stable's PythonNode):
      - If the source assigns `output`, that's the result.
      - Else if the source is a single expression, the expression's value.
      - Else the value of the last line if it's eval'able.
      - bytes → emitted raw; str → utf-8; other → repr().

    Pre-imported in the eval namespace: json, re, base64, os, hashlib,
    struct, math, sys.
    """
    import json as _json
    import re as _re
    import base64 as _base64
    import hashlib as _hashlib
    import struct as _struct
    import math as _math
    import sys as _sys

    ns = {
        "input": input_bytes,
        "json": _json, "re": _re, "base64": _base64,
        "hashlib": _hashlib, "struct": _struct,
        "math": _math, "os": os, "sys": _sys,
        "__builtins__": __builtins__,
    }

    try:
        try:
            # Single-expression fast path.
            result = eval(expr, ns, ns)
        except SyntaxError:
            # Multi-line: exec, then look for `output` or last line.
            tree = compile(expr, "<python_node>", "exec")
            exec(tree, ns, ns)
            result = ns.get("output", None)
            if result is None:
                last = expr.strip().splitlines()[-1].strip()
                try:
                    result = eval(last, ns, ns)
                except Exception:
                    result = b""
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": b"",
            "stderr": f"{type(e).__name__}: {e}".encode("utf-8"),
            "timed_out": False,
        }

    # Normalize result to bytes.
    if isinstance(result, (bytes, bytearray)):
        out = bytes(result)
    elif isinstance(result, str):
        out = result.encode("utf-8")
    elif result is None:
        out = b""
    else:
        out = repr(result).encode("utf-8")

    return {
        "returncode": 0,
        "stdout": out,
        "stderr": b"",
        "timed_out": False,
    }


class BashNodeView(_ExecNodeBase):
    """Bash node — multi-line shell command, executed locally by the
    operator and the result emitted on OUT (stdout) and ERR (stderr).

    Typical commands:
        echo hello
        ls /tmp
        curl -sS https://example.com | head
        for f in *.txt; do wc -l "$f"; done

    Downstream wiring: connect this node's OUT port to anything that
    reads bytes (a text node's `in`, another bash's `cmd`, an LLM
    agent's input). The command runs on click of Run (or Ctrl/Cmd+Enter
    in the editor), and stdout is published to OUT immediately on
    completion.
    """
    COMMAND_PLACEHOLDER = "bash command — e.g. `ls /tmp`"
    PROMPT_GLYPH = "$"
    DEFAULT_COMMAND = "echo hello"
    EXECUTOR = staticmethod(_run_bash_pipe)
    TIMEOUT_SEC = BASH_DEFAULT_TIMEOUT_SEC


class PythonNodeView(_ExecNodeBase):
    """Python node — multi-line expression or script body, run in-process
    via exec/eval. Result is written to OUT.

    The executor binds the bytes received on stdin to `input` in the eval
    namespace. With no input wired, `input` is b"". The result is:
      - the value of `output` if you assigned it
      - the value of a single-expression source
      - the value of the last line if it eval's

    bytes → emitted raw; str → utf-8; other → repr().
    """
    COMMAND_PORT_NAME = "code"
    COMMAND_PLACEHOLDER = "python expression or script — e.g. `1 + 1`"
    PROMPT_GLYPH = "py"
    DEFAULT_COMMAND = "1 + 1"
    EXECUTOR = staticmethod(_eval_python_pipe)
    TIMEOUT_SEC = 30.0  # not enforced (in-process eval), but keep shape


# ═══════════════════════════════════════════════════════════════════════
# ─── REGISTRY ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
#
# Maps NodeKind → NodeView subclass. The Operator uses this when a
# model `Node` is added: pick the right view class, construct it,
# add to scene.


VIEW_CLASS_BY_KIND = {
    NodeKind.AGENT: AgentNodeView,
    NodeKind.TERMINAL: GenericNodeView,
    NodeKind.SCENE: GenericNodeView,
    NodeKind.TEXT: TextNodeView,
    NodeKind.DEBUG: DebugNodeView,
    NodeKind.MEDIA: MediaNodeView,
    NodeKind.BASH: BashNodeView,
    NodeKind.PYTHON: PythonNodeView,
    NodeKind.GENERIC: GenericNodeView,
}


def view_class_for(kind: NodeKind):
    return VIEW_CLASS_BY_KIND.get(kind, GenericNodeView)


__all__ = [
    # base
    "NodeView", "PortItem", "ConnectionItem", "TempConnectionItem",
    "header_color_for",
    # subclasses
    "TextNodeView", "DebugNodeView", "MediaNodeView",
    "AgentNodeView", "GenericNodeView",
    "BashNodeView", "PythonNodeView",
    # registry
    "VIEW_CLASS_BY_KIND", "view_class_for",
    # helpers
    "detect_media_format",
]
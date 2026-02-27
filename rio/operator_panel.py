"""
Operator Panel - Classic Node Graph for LLMFS Architecture

A clean, professional graph-based interface for visualizing and manipulating
the LLMFS filesystem. Inspired by n8n/Node-RED style workflow editors.

Features:
  - Create/delete agents via GUI
  - Each agent node exposes ports for its files (input, output, system, rules, etc.)
  - Drag-to-connect between ports creates filesystem routes
  - Select .md files from disk to assign as system prompts
  - Full graphical control over the LLMFS

Design: Classic, clean, professional - white canvas with subtle grid,
        rounded rectangular nodes, labeled ports, smooth bezier connections.
"""

from PySide6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem,
    QGraphicsRectItem, QGraphicsPathItem, QGraphicsProxyWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QMenu, QInputDialog, QMessageBox, QFileDialog, QToolBar,
    QGraphicsDropShadowEffect, QSizePolicy, QComboBox,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QApplication,
    QGraphicsSimpleTextItem, QCheckBox
)
from PySide6.QtCore import (
    Qt, Signal, QTimer, QPointF, QRectF, QLineF, QSizeF,
    QPropertyAnimation, QEasingCurve, Property, QObject, QMarginsF, Slot
)
from PySide6.QtGui import (
    QColor, QPen, QBrush, QFont, QPainterPath, QPainter,
    QLinearGradient, QRadialGradient, QCursor, QAction,
    QFontMetrics, QIcon, QPixmap, QPolygonF, QTransform
)
import os
import json
import math
import threading
import traceback
from concurrent.futures import Future
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# Theme - Clean Classic
# ═══════════════════════════════════════════════════════════════════════════

class DarkTheme:
    """Minimalist monotone theme — dark, translucent, refined."""

    # ── Canvas ────────────────────────────────────────────────────────────
    CANVAS_BG = QColor(22, 22, 26)
    GRID_LINE = QColor(38, 38, 44)
    GRID_LINE_MAJOR = QColor(48, 48, 56)
    GRID_DOT = QColor(42, 42, 50)

    # ── Node body ─────────────────────────────────────────────────────────
    NODE_BG = QColor(32, 32, 38, 210)
    NODE_BORDER = QColor(62, 62, 72, 140)
    NODE_BORDER_HOVER = QColor(120, 120, 135, 180)
    NODE_BORDER_SELECTED = QColor(180, 180, 195, 220)
    NODE_SHADOW = QColor(0, 0, 0, 60)

    NODE_HEADER_AGENT = QColor(52, 58, 48, 240)
    NODE_HEADER_AGENT_ALT = QColor(46, 52, 62, 240)
    NODE_HEADER_FILE = QColor(58, 54, 44, 240)
    NODE_HEADER_SPECIAL = QColor(54, 46, 58, 240)

    # ── Port colors ───────────────────────────────────────────────────────
    PORT_INPUT = QColor(110, 130, 165)
    PORT_OUTPUT = QColor(110, 145, 120)
    PORT_SYSTEM = QColor(155, 140, 110)
    PORT_RULES = QColor(135, 115, 150)
    PORT_CONFIG = QColor(120, 120, 130)
    PORT_HISTORY = QColor(105, 140, 150)
    PORT_CTL = QColor(155, 105, 105)
    PORT_ERRORS = QColor(160, 110, 100)
    PORT_SUPPLEMENTARY = QColor(125, 120, 155)
    PORT_DEFAULT = QColor(115, 115, 125)
    PORT_BORDER = QColor(32, 32, 38, 180)

    # ── Connections ───────────────────────────────────────────────────────
    CONN_DEFAULT = QColor(90, 90, 105, 140)
    CONN_ACTIVE = QColor(160, 160, 178, 200)
    CONN_HOVER = QColor(175, 175, 190, 220)
    CONN_DRAGGING = QColor(150, 150, 170, 120)

    # ── Text ──────────────────────────────────────────────────────────────
    TEXT_PRIMARY = QColor(210, 210, 218)
    TEXT_SECONDARY = QColor(120, 120, 135)
    TEXT_ON_HEADER = QColor(195, 195, 205)
    TEXT_PORT = QColor(140, 140, 155)
    TEXT_BADGE = QColor(195, 195, 205)

    # ── UI Chrome ─────────────────────────────────────────────────────────
    TOOLBAR_BG = QColor(26, 26, 30)
    TOOLBAR_BORDER = QColor(42, 42, 50)
    STATUS_BG = QColor(24, 24, 28)
    STATUS_BORDER = QColor(42, 42, 50)

    BUTTON_PRIMARY = QColor(62, 62, 74)
    BUTTON_PRIMARY_HOVER = QColor(78, 78, 92)
    BUTTON_DANGER = QColor(120, 55, 55)
    BUTTON_DANGER_HOVER = QColor(140, 65, 65)
    BUTTON_DEFAULT = QColor(38, 38, 46)
    BUTTON_DEFAULT_HOVER = QColor(50, 50, 60)
    BUTTON_DEFAULT_BORDER = QColor(56, 56, 66)

    # ── Standalone node header colors ────────────────────────────────────
    NODE_HEADER_9P = QColor(44, 48, 56, 240)
    NODE_HEADER_TEXT = QColor(42, 52, 58, 240)
    NODE_HEADER_DEBUG = QColor(58, 42, 42, 240)

    # ── Dark-mode specific paint helpers ─────────────────────────────────
    # Used directly in paint() methods for mode-dependent rendering
    SEPARATOR_COLOR = QColor(255, 255, 255, 12)
    COLLAPSE_INDICATOR_COLOR = QColor(255, 255, 255, 60)
    NODE_BODY_TRANSLUCENT = QColor(28, 28, 34, 160)  # TextNode/DebugNode body
    EXISTING_CONN_COLOR = QColor(110, 145, 120, 160)

    # ── Embedded widget styles ───────────────────────────────────────────
    TEXT_EDIT_BG = "rgba(20, 20, 24, 200)"
    TEXT_EDIT_BORDER = "rgba(62, 62, 72, 140)"
    TEXT_EDIT_SELECTION = "rgba(120, 120, 140, 120)"
    SPINBOX_BG = "rgba(20, 20, 24, 200)"
    SPINBOX_BORDER = "rgba(62, 62, 72, 140)"
    CHECKBOX_BG = "rgba(20, 20, 24, 180)"
    CHECKBOX_BORDER = "rgba(62, 62, 72, 160)"
    CHECKBOX_CHECKED_BG = "rgba(110, 145, 120, 160)"
    CHECKBOX_CHECKED_BORDER = "rgba(110, 145, 120, 200)"
    CONTAINER_BG = "rgba(24,24,28,150)"
    BTN_READ_BG = "rgba(110, 130, 165, 60)"
    BTN_READ_BORDER = "rgba(110, 130, 165, 80)"
    BTN_READ_HOVER = "rgba(110, 130, 165, 100)"
    BTN_READ_PRESS = "rgba(110, 130, 165, 140)"
    BTN_WRITE_BG = "rgba(110, 145, 120, 60)"
    BTN_WRITE_BORDER = "rgba(110, 145, 120, 80)"
    BTN_WRITE_HOVER = "rgba(110, 145, 120, 100)"
    BTN_WRITE_PRESS = "rgba(110, 145, 120, 140)"

    # ── Toolbar/Status chrome ────────────────────────────────────────────
    TOOLBAR_BG_RGBA = "rgba(26, 26, 30, 200)"
    STATUS_BG_RGBA = "rgba(24, 24, 28, 200)"
    TITLE_LETTER_SPACING = "2px"
    SEP_COLOR_RGBA = "rgba(120, 120, 135, 80)"

    # ── Toolbar button styles ────────────────────────────────────────────
    BTN_PRIMARY_BG = "rgba(180, 180, 195, 18)"
    BTN_PRIMARY_BORDER = "rgba(180, 180, 195, 40)"
    BTN_PRIMARY_HOVER_BG = "rgba(180, 180, 195, 35)"
    BTN_PRIMARY_HOVER_BORDER = "rgba(180, 180, 195, 60)"
    BTN_DEFAULT_BG_RGBA = "transparent"
    BTN_DEFAULT_BORDER_RGBA = "rgba(62, 62, 72, 80)"
    BTN_DEFAULT_HOVER_BG_RGBA = "rgba(120, 120, 135, 20)"
    BTN_DEFAULT_HOVER_BORDER_RGBA = "rgba(120, 120, 135, 50)"

    # ── Menu/Dialog ──────────────────────────────────────────────────────
    MENU_BG = "rgba(32, 32, 38, 245)"
    MENU_BORDER = "rgba(62, 62, 72, 160)"
    MENU_ITEM_HOVER = "rgba(120, 120, 140, 50)"
    MENU_SEP = "rgba(62, 62, 72, 120)"
    DIALOG_BG = "rgb(28, 28, 32)"
    DIALOG_INPUT_BG = "rgba(20, 20, 24, 220)"
    DIALOG_INPUT_BORDER = "rgba(62, 62, 72, 160)"
    DIALOG_INPUT_FOCUS_BORDER = "rgba(150, 150, 170, 120)"
    DIALOG_BTN_BG = "rgba(50, 50, 60, 200)"
    DIALOG_BTN_BORDER = "rgba(62, 62, 72, 160)"
    DIALOG_BTN_HOVER = "rgba(65, 65, 78, 220)"
    DIALOG_BTN_HOVER_BORDER = "rgba(100, 100, 115, 140)"
    DIALOG_OK_BG = "rgba(150, 150, 170, 40)"
    DIALOG_OK_BORDER = "rgba(150, 150, 170, 60)"
    DIALOG_OK_HOVER = "rgba(150, 150, 170, 70)"


class LightTheme:
    """Minimalist monotone theme — light, translucent, clean."""

    # ── Canvas ────────────────────────────────────────────────────────────
    CANVAS_BG = QColor(248, 248, 250)
    GRID_LINE = QColor(228, 228, 234)
    GRID_LINE_MAJOR = QColor(210, 212, 218)
    GRID_DOT = QColor(200, 202, 210)

    # ── Node body ─────────────────────────────────────────────────────────
    NODE_BG = QColor(255, 255, 255, 220)
    NODE_BORDER = QColor(195, 195, 205, 160)
    NODE_BORDER_HOVER = QColor(140, 140, 155, 200)
    NODE_BORDER_SELECTED = QColor(80, 80, 100, 220)
    NODE_SHADOW = QColor(0, 0, 0, 25)

    NODE_HEADER_AGENT = QColor(218, 225, 212, 245)      # faint sage
    NODE_HEADER_AGENT_ALT = QColor(210, 218, 228, 245)  # faint sky
    NODE_HEADER_FILE = QColor(228, 222, 210, 245)        # faint sand
    NODE_HEADER_SPECIAL = QColor(222, 214, 228, 245)     # faint lilac

    # ── Port colors ───────────────────────────────────────────────────────
    PORT_INPUT = QColor(90, 110, 150)
    PORT_OUTPUT = QColor(90, 128, 100)
    PORT_SYSTEM = QColor(145, 125, 85)
    PORT_RULES = QColor(120, 100, 140)
    PORT_CONFIG = QColor(110, 110, 120)
    PORT_HISTORY = QColor(85, 125, 138)
    PORT_CTL = QColor(145, 90, 90)
    PORT_ERRORS = QColor(150, 95, 85)
    PORT_SUPPLEMENTARY = QColor(108, 102, 142)
    PORT_DEFAULT = QColor(105, 105, 118)
    PORT_BORDER = QColor(255, 255, 255, 200)

    # ── Connections ───────────────────────────────────────────────────────
    CONN_DEFAULT = QColor(165, 165, 178, 150)
    CONN_ACTIVE = QColor(90, 90, 110, 200)
    CONN_HOVER = QColor(80, 80, 100, 220)
    CONN_DRAGGING = QColor(100, 100, 125, 120)

    # ── Text ──────────────────────────────────────────────────────────────
    TEXT_PRIMARY = QColor(42, 42, 50)
    TEXT_SECONDARY = QColor(120, 120, 135)
    TEXT_ON_HEADER = QColor(55, 55, 65)
    TEXT_PORT = QColor(90, 90, 108)
    TEXT_BADGE = QColor(60, 60, 72)

    # ── UI Chrome ─────────────────────────────────────────────────────────
    TOOLBAR_BG = QColor(250, 250, 252)
    TOOLBAR_BORDER = QColor(225, 225, 232)
    STATUS_BG = QColor(248, 248, 250)
    STATUS_BORDER = QColor(225, 225, 232)

    BUTTON_PRIMARY = QColor(230, 230, 238)
    BUTTON_PRIMARY_HOVER = QColor(218, 218, 228)
    BUTTON_DANGER = QColor(195, 105, 105)
    BUTTON_DANGER_HOVER = QColor(180, 90, 90)
    BUTTON_DEFAULT = QColor(242, 242, 246)
    BUTTON_DEFAULT_HOVER = QColor(232, 232, 238)
    BUTTON_DEFAULT_BORDER = QColor(210, 210, 220)

    # ── Standalone node header colors ────────────────────────────────────
    NODE_HEADER_9P = QColor(208, 212, 222, 245)
    NODE_HEADER_TEXT = QColor(206, 218, 225, 245)
    NODE_HEADER_DEBUG = QColor(228, 210, 210, 245)

    # ── Light-mode specific paint helpers ────────────────────────────────
    SEPARATOR_COLOR = QColor(0, 0, 0, 18)
    COLLAPSE_INDICATOR_COLOR = QColor(0, 0, 0, 40)
    NODE_BODY_TRANSLUCENT = QColor(252, 252, 254, 180)
    EXISTING_CONN_COLOR = QColor(90, 128, 100, 160)

    # ── Embedded widget styles ───────────────────────────────────────────
    TEXT_EDIT_BG = "rgba(252, 252, 254, 220)"
    TEXT_EDIT_BORDER = "rgba(195, 195, 205, 160)"
    TEXT_EDIT_SELECTION = "rgba(80, 80, 120, 80)"
    SPINBOX_BG = "rgba(252, 252, 254, 220)"
    SPINBOX_BORDER = "rgba(195, 195, 205, 160)"
    CHECKBOX_BG = "rgba(252, 252, 254, 200)"
    CHECKBOX_BORDER = "rgba(195, 195, 205, 180)"
    CHECKBOX_CHECKED_BG = "rgba(90, 128, 100, 160)"
    CHECKBOX_CHECKED_BORDER = "rgba(90, 128, 100, 200)"
    CONTAINER_BG = "rgba(248,248,250,160)"
    BTN_READ_BG = "rgba(90, 110, 150, 50)"
    BTN_READ_BORDER = "rgba(90, 110, 150, 70)"
    BTN_READ_HOVER = "rgba(90, 110, 150, 90)"
    BTN_READ_PRESS = "rgba(90, 110, 150, 130)"
    BTN_WRITE_BG = "rgba(90, 128, 100, 50)"
    BTN_WRITE_BORDER = "rgba(90, 128, 100, 70)"
    BTN_WRITE_HOVER = "rgba(90, 128, 100, 90)"
    BTN_WRITE_PRESS = "rgba(90, 128, 100, 130)"

    # ── Toolbar/Status chrome ────────────────────────────────────────────
    TOOLBAR_BG_RGBA = "rgba(250, 250, 252, 215)"
    STATUS_BG_RGBA = "rgba(248, 248, 250, 215)"
    TITLE_LETTER_SPACING = "2px"
    SEP_COLOR_RGBA = "rgba(120, 120, 135, 50)"

    # ── Toolbar button styles ────────────────────────────────────────────
    BTN_PRIMARY_BG = "rgba(60, 60, 75, 12)"
    BTN_PRIMARY_BORDER = "rgba(60, 60, 75, 30)"
    BTN_PRIMARY_HOVER_BG = "rgba(60, 60, 75, 25)"
    BTN_PRIMARY_HOVER_BORDER = "rgba(60, 60, 75, 50)"
    BTN_DEFAULT_BG_RGBA = "transparent"
    BTN_DEFAULT_BORDER_RGBA = "rgba(195, 195, 205, 100)"
    BTN_DEFAULT_HOVER_BG_RGBA = "rgba(120, 120, 140, 15)"
    BTN_DEFAULT_HOVER_BORDER_RGBA = "rgba(120, 120, 140, 60)"

    # ── Menu/Dialog ──────────────────────────────────────────────────────
    MENU_BG = "rgba(252, 252, 254, 248)"
    MENU_BORDER = "rgba(195, 195, 205, 180)"
    MENU_ITEM_HOVER = "rgba(60, 60, 80, 30)"
    MENU_SEP = "rgba(195, 195, 205, 140)"
    DIALOG_BG = "rgb(250, 250, 252)"
    DIALOG_INPUT_BG = "rgba(252, 252, 254, 240)"
    DIALOG_INPUT_BORDER = "rgba(195, 195, 205, 180)"
    DIALOG_INPUT_FOCUS_BORDER = "rgba(100, 100, 125, 140)"
    DIALOG_BTN_BG = "rgba(240, 240, 244, 220)"
    DIALOG_BTN_BORDER = "rgba(195, 195, 205, 180)"
    DIALOG_BTN_HOVER = "rgba(228, 228, 235, 240)"
    DIALOG_BTN_HOVER_BORDER = "rgba(160, 160, 175, 160)"
    DIALOG_OK_BG = "rgba(70, 70, 90, 30)"
    DIALOG_OK_BORDER = "rgba(70, 70, 90, 50)"
    DIALOG_OK_HOVER = "rgba(70, 70, 90, 55)"


# ── Fonts & Dimensions (shared between modes) ────────────────────────────

_FONT_FAMILY = "Segoe UI"
_FONT_FAMILY_MONO = "Consolas"
_FONT_NODE_TITLE = QFont("Segoe UI", 10, QFont.Normal)
_FONT_PORT_LABEL = QFont("Segoe UI", 7)
_FONT_STATUS = QFont("Segoe UI", 8)
_FONT_BADGE = QFont("Segoe UI", 7, QFont.DemiBold)
_FONT_TOOLBAR = QFont("Segoe UI", 9)

_NODE_WIDTH = 210
_NODE_CORNER_RADIUS = 6
_NODE_HEADER_HEIGHT = 32
_PORT_RADIUS = 4
_PORT_HIT_RADIUS = 12
_PORT_SPACING = 24
_PORT_MARGIN_TOP = 10
_CONN_LINE_WIDTH = 1.5
_GRID_SIZE = 24
_GRID_MAJOR_EVERY = 5


class _ThemeProxy:
    """Proxy that delegates attribute lookups to the active theme class.

    Shared constants (fonts, dimensions) are returned directly.
    Color/style attributes are forwarded to DarkTheme or LightTheme
    depending on which is active.  Switching is instant — every paint()
    call re-reads Theme.XXX so the scene updates on next repaint.
    """

    def __init__(self):
        self._active = DarkTheme

    def set_mode(self, dark: bool):
        self._active = DarkTheme if dark else LightTheme

    @property
    def is_dark(self) -> bool:
        return self._active is DarkTheme

    def __getattr__(self, name):
        # Shared fonts
        _shared = {
            'FONT_FAMILY': _FONT_FAMILY,
            'FONT_FAMILY_MONO': _FONT_FAMILY_MONO,
            'FONT_NODE_TITLE': _FONT_NODE_TITLE,
            'FONT_PORT_LABEL': _FONT_PORT_LABEL,
            'FONT_STATUS': _FONT_STATUS,
            'FONT_BADGE': _FONT_BADGE,
            'FONT_TOOLBAR': _FONT_TOOLBAR,
            # Shared dimensions
            'NODE_WIDTH': _NODE_WIDTH,
            'NODE_CORNER_RADIUS': _NODE_CORNER_RADIUS,
            'NODE_HEADER_HEIGHT': _NODE_HEADER_HEIGHT,
            'PORT_RADIUS': _PORT_RADIUS,
            'PORT_HIT_RADIUS': _PORT_HIT_RADIUS,
            'PORT_SPACING': _PORT_SPACING,
            'PORT_MARGIN_TOP': _PORT_MARGIN_TOP,
            'CONN_LINE_WIDTH': _CONN_LINE_WIDTH,
            'GRID_SIZE': _GRID_SIZE,
            'GRID_MAJOR_EVERY': _GRID_MAJOR_EVERY,
        }
        if name in _shared:
            return _shared[name]
        # Delegate to active theme
        return getattr(self._active, name)


Theme = _ThemeProxy()


# ═══════════════════════════════════════════════════════════════════════════
# Filesystem Worker - non-blocking I/O for 9P mounts
# ═══════════════════════════════════════════════════════════════════════════

class FSWorker(QObject):
    """
    Runs filesystem operations (listdir, read, stat) on a background thread
    so that blocking 9P reads never freeze the Qt event loop.

    Usage:
        worker = FSWorker()
        worker.run_async(os.listdir, "/n/llm/agents", callback=on_result)

    The callback is invoked on the Qt main thread via signal.
    """

    result_ready = Signal(object, object)   # (tag, result_or_exception)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = []  # reusable threads
        self.result_ready.connect(self._dispatch)
        self._pending: Dict[int, Callable] = {}
        self._tag_counter = 0

    def run_async(self, fn: Callable, *args,
                  callback: Callable = None,
                  timeout: float = 2.0) -> None:
        """
        Execute *fn(*args)* on a daemon thread.
        On completion, *callback(result)* is called on the Qt thread.
        If the call takes longer than *timeout* seconds the thread is
        abandoned (daemon) and callback receives a TimeoutError.
        """
        tag = self._tag_counter
        self._tag_counter += 1
        if callback:
            self._pending[tag] = callback

        t = threading.Thread(target=self._run, args=(tag, fn, args, timeout),
                             daemon=True)
        t.start()

    def _run(self, tag, fn, args, timeout):
        try:
            result = fn(*args)
            self.result_ready.emit(tag, result)
        except Exception as exc:
            self.result_ready.emit(tag, exc)

    @Slot(object, object)
    def _dispatch(self, tag, result):
        cb = self._pending.pop(tag, None)
        if cb is None:
            return
        if isinstance(result, Exception):
            # Silently ignore – caller can check
            try:
                cb(result)
            except Exception:
                pass
        else:
            try:
                cb(result)
            except Exception:
                traceback.print_exc()


def _fs_listdir(path: str) -> List[str]:
    """Safe listdir – returns [] on error."""
    try:
        return sorted(os.listdir(path))
    except Exception:
        return []


def _fs_read(path: str, max_bytes: int = 8192) -> str:
    """Safe read – returns '' on error. Uses O_NONBLOCK where possible."""
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        data = os.read(fd, max_bytes)
        return data.decode('utf-8', errors='replace')
    except (BlockingIOError, OSError):
        # file would block or doesn't exist
        return ''
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _fs_isdir(path: str) -> bool:
    """Safe isdir."""
    try:
        return os.path.isdir(path)
    except Exception:
        return False


def _fs_scan_agent(agent_path: str) -> Dict[str, Any]:
    """
    Scan a single agent directory and return its metadata.
    Runs on background thread.
    Returns: {'name': str, 'files': [str], 'ctl': str, 'config': str}
    """
    result = {
        'name': os.path.basename(agent_path),
        'files': [],
        'ctl': '',
        'config': '',
    }
    try:
        entries = os.listdir(agent_path)
        result['files'] = sorted(entries)
    except Exception:
        pass

    # Try non-blocking reads of ctl and config
    result['ctl'] = _fs_read(os.path.join(agent_path, 'ctl'))
    result['config'] = _fs_read(os.path.join(agent_path, 'config'))
    return result


def _fs_scan_agents_dir(agents_dir: str) -> List[Dict[str, Any]]:
    """
    Scan the entire agents/ directory.  Returns list of agent metadata dicts.
    Runs on background thread.
    """
    agents = []
    try:
        entries = os.listdir(agents_dir)
    except Exception:
        return agents

    for name in sorted(entries):
        path = os.path.join(agents_dir, name)
        if _fs_isdir(path):
            agents.append(_fs_scan_agent(path))
    return agents


def _fs_scan_9p_root(root: str = "/n") -> List[Dict[str, Any]]:
    """
    Recursively scan a filesystem tree from root.
    Returns a flat list of node descriptors, one per directory:
      {'name': str, 'path': str, 'files': [str], 'subdirs': [str], 'depth': int}
    
    The scan walks depth-first: for each directory, it lists entries,
    separates files from subdirectories, and recurses into subdirs.
    Stops at max_depth=6 to avoid runaway traversal.
    """
    results = []
    _fs_scan_tree_recursive(root, results, depth=0, max_depth=6)
    return results


def _fs_scan_tree_recursive(path: str, results: list, depth: int, max_depth: int):
    """Recursive helper for _fs_scan_9p_root."""
    if depth > max_depth:
        return

    try:
        entries = sorted(os.listdir(path))
    except Exception:
        return

    files = []
    subdirs = []

    for entry in entries:
        full = os.path.join(path, entry)
        if _fs_isdir(full):
            subdirs.append(entry)
        else:
            files.append(entry)

    results.append({
        'name': os.path.basename(path) or path,
        'path': path,
        'files': files,
        'subdirs': subdirs,
        'depth': depth,
    })

    # Recurse into subdirectories
    for subdir in subdirs:
        _fs_scan_tree_recursive(os.path.join(path, subdir), results, depth + 1, max_depth)


def _fs_scan_terminals(svc_path: str) -> List[str]:
    """
    Scan /n/<svc>/terms/ for terminal IDs.
    """
    terms_path = os.path.join(svc_path, 'terms')
    try:
        return sorted(os.listdir(terms_path))
    except Exception:
        return []


def _fs_read_routes(mount_path: str) -> List[Tuple[str, str]]:
    """
    Read routes from the shared routes file at /n/<svc>/routes.
    Returns list of (source_path, dest_path) tuples.

    The expected format is one route per line:
        /n/llm/agents/master/bash -> /n/rioa/terms/term_abc/stdin [running]
        /n/llm/agents/coder/rioa -> /n/rioa/scene/parse [running]

    Status tags like [running] or [stopped] are stripped.
    """
    routes: List[Tuple[str, str]] = []
    routes_path = os.path.join(mount_path, 'routes')
    content = _fs_read(routes_path, 32768)
    if not content:
        return routes

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('(no routes'):
            continue
        # Strip trailing status like [running] [stopped]
        line = line.split('[')[0].strip()

        if '->' in line:
            parts = line.split('->', 1)
            src = parts[0].strip()
            dst = parts[1].strip()
            if src and dst:
                routes.append((src, dst))

    return routes


# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════

class PortDirection(Enum):
    LEFT = "left"
    RIGHT = "right"

@dataclass
class PortDef:
    """Definition of a port on a node."""
    name: str
    direction: PortDirection
    color: QColor
    file_path: str = ""       # Filesystem path this port represents
    description: str = ""


# Mapping agent file names to port definitions
AGENT_LEFT_PORTS = [
    PortDef("input",   PortDirection.LEFT, Theme.PORT_INPUT,   description="Write prompts here"),
    PortDef("system",  PortDirection.LEFT, Theme.PORT_SYSTEM,  description="System prompt"),
    PortDef("config",  PortDirection.LEFT, Theme.PORT_CONFIG,  description="JSON configuration"),
    PortDef("rules",   PortDirection.LEFT, Theme.PORT_RULES,   description="Plumbing rules"),
    PortDef("history", PortDirection.LEFT, Theme.PORT_HISTORY, description="Conversation history"),
    PortDef("ctl",     PortDirection.LEFT, Theme.PORT_CTL,     description="Control commands"),
]

AGENT_RIGHT_PORTS = [
    PortDef("output",  PortDirection.RIGHT, Theme.PORT_OUTPUT, description="Stream response"),
    PortDef("errors",  PortDirection.RIGHT, Theme.PORT_ERRORS, description="Error messages"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Port - connection endpoint on a node
# ═══════════════════════════════════════════════════════════════════════════

class Port(QGraphicsEllipseItem):
    """
    A small circular port on the edge of a node.
    Represents one of the agent's files (input, output, system, etc.)
    Can be dragged to create connections.
    """

    def __init__(self, port_def: PortDef, parent_node: 'BaseNode'):
        r = Theme.PORT_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, parent_node)

        self.port_def = port_def
        self.parent_node = parent_node
        self.connections: List['Connection'] = []

        # Hit area larger than visual (must be set before setCursor
        # because setCursor triggers boundingRect())
        self._hit_rect = QRectF(
            -Theme.PORT_HIT_RADIUS, -Theme.PORT_HIT_RADIUS,
            2 * Theme.PORT_HIT_RADIUS, 2 * Theme.PORT_HIT_RADIUS
        )

        # Visual
        self.setBrush(QBrush(port_def.color))
        self.setPen(QPen(Theme.PORT_BORDER, 1.5))
        self.setZValue(3)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)

        # Label
        self._label = QGraphicsSimpleTextItem(port_def.name, parent_node)
        self._label.setFont(Theme.FONT_PORT_LABEL)
        self._label.setBrush(QBrush(Theme.TEXT_PORT))

    def shape(self):
        """Enlarge the clickable area."""
        path = QPainterPath()
        path.addEllipse(self._hit_rect)
        return path

    def boundingRect(self):
        return self._hit_rect

    def update_label_pos(self):
        """Position the label relative to port after layout."""
        label_rect = self._label.boundingRect()
        px, py = self.x(), self.y()
        if self.port_def.direction == PortDirection.LEFT:
            self._label.setPos(px + Theme.PORT_RADIUS + 6, py - label_rect.height() / 2)
        else:
            self._label.setPos(px - label_rect.width() - Theme.PORT_RADIUS - 6,
                               py - label_rect.height() / 2)

    def scene_center(self) -> QPointF:
        """Get the center of this port in scene coordinates."""
        return self.parentItem().mapToScene(self.pos())

    def hoverEnterEvent(self, event):
        lighter = self.port_def.color.lighter(140)
        self.setBrush(QBrush(lighter))
        self.setPen(QPen(lighter, 1.5))
        self.setScale(1.25)
        # Show tooltip
        tooltip = f"{self.port_def.name}"
        if self.port_def.description:
            tooltip += f"\n{self.port_def.description}"
        if self.port_def.file_path:
            tooltip += f"\n{self.port_def.file_path}"
        self.setToolTip(tooltip)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(self.port_def.color))
        self.setPen(QPen(Theme.PORT_BORDER, 1.5))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Start connection drag
            scene = self.scene()
            if isinstance(scene, OperatorGraphScene):
                scene.start_port_connection(self)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene = self.scene()
            if isinstance(scene, OperatorGraphScene):
                scene.finish_port_connection(event.scenePos())
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
# Connection - bezier curve between two ports
# ═══════════════════════════════════════════════════════════════════════════

class Connection(QGraphicsPathItem):
    """
    Smooth bezier connection between two ports.
    Represents a filesystem route (pipe between agent files).
    """

    def __init__(self, source_port: Port, target_port: Port,
                 route_cmd: str = "", is_existing: bool = False):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        self.route_cmd = route_cmd
        self.is_existing = is_existing  # True if imported from terminal attachments

        self.setZValue(0)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self._hovered = False

        # Register with ports
        source_port.connections.append(self)
        target_port.connections.append(self)

        self.update_path()

    def update_path(self):
        """Recalculate bezier path between ports."""
        if not self.source_port or not self.target_port:
            return

        start = self.source_port.scene_center()
        end = self.target_port.scene_center()

        path = QPainterPath()
        path.moveTo(start)

        # Horizontal distance determines curvature
        dx = abs(end.x() - start.x())
        curvature = max(dx * 0.5, 60)

        # Source port goes right, target port goes left
        if self.source_port.port_def.direction == PortDirection.RIGHT:
            ctrl1 = QPointF(start.x() + curvature, start.y())
        else:
            ctrl1 = QPointF(start.x() - curvature, start.y())

        if self.target_port.port_def.direction == PortDirection.LEFT:
            ctrl2 = QPointF(end.x() - curvature, end.y())
        else:
            ctrl2 = QPointF(end.x() + curvature, end.y())

        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        if self.isSelected():
            color = Theme.NODE_BORDER_SELECTED
            width = Theme.CONN_LINE_WIDTH + 1.0
        elif self._hovered:
            color = Theme.CONN_HOVER
            width = Theme.CONN_LINE_WIDTH + 0.5
        elif self.is_existing:
            color = Theme.EXISTING_CONN_COLOR
            width = Theme.CONN_LINE_WIDTH
        else:
            color = Theme.CONN_DEFAULT
            width = Theme.CONN_LINE_WIDTH

        painter.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap))
        painter.drawPath(self.path())

        # Draw small arrow at midpoint
        path = self.path()
        mid_t = 0.5
        mid_pt = path.pointAtPercent(mid_t)
        tangent_angle = path.angleAtPercent(mid_t)

        painter.save()
        painter.translate(mid_pt)
        painter.rotate(-tangent_angle)

        arrow = QPolygonF([
            QPointF(4, 0),
            QPointF(-2.5, -2.5),
            QPointF(-2.5, 2.5),
        ])
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(arrow)
        painter.restore()

    def mousePressEvent(self, event):
        """Handle click to select/deselect this connection."""
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self.setSelected(not self.isSelected())
            else:
                scene = self.scene()
                if scene:
                    scene.clearSelection()
                self.setSelected(True)
            event.accept()
        else:
            super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.setCursor(Qt.PointingHandCursor)
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.setCursor(Qt.ArrowCursor)
        self.update()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(_menu_stylesheet())

        info = menu.addAction(f"⛓  {self.source_port.parent_node.node_name}/"
                              f"{self.source_port.port_def.name}  →  "
                              f"{self.target_port.parent_node.node_name}/"
                              f"{self.target_port.port_def.name}")
        info.setEnabled(False)
        menu.addSeparator()

        delete_action = menu.addAction("✕  Delete Route")

        action = menu.exec(event.screenPos())
        if action == delete_action:
            scene = self.scene()
            if isinstance(scene, OperatorGraphScene):
                scene.remove_connection(self)

    def shape(self):
        """Widen hit area for easier selection."""
        stroker = QPainterPath()
        ps = self.path()
        # Use QPainterPathStroker for wider hit area
        from PySide6.QtGui import QPainterPathStroker
        s = QPainterPathStroker()
        s.setWidth(14)
        return s.createStroke(ps)


# ═══════════════════════════════════════════════════════════════════════════
# Temp Connection Line (while dragging)
# ═══════════════════════════════════════════════════════════════════════════

class TempConnection(QGraphicsPathItem):
    """Temporary bezier shown while user drags from a port."""

    def __init__(self, start_port: Port):
        super().__init__()
        self.start_port = start_port
        self.setZValue(10)
        self.end_pos = start_port.scene_center()
        self._update()

    def set_end(self, pos: QPointF):
        self.end_pos = pos
        self._update()

    def _update(self):
        start = self.start_port.scene_center()
        end = self.end_pos

        path = QPainterPath()
        path.moveTo(start)

        dx = abs(end.x() - start.x())
        curvature = max(dx * 0.5, 60)

        if self.start_port.port_def.direction == PortDirection.RIGHT:
            ctrl1 = QPointF(start.x() + curvature, start.y())
            ctrl2 = QPointF(end.x() - curvature, end.y())
        else:
            ctrl1 = QPointF(start.x() - curvature, start.y())
            ctrl2 = QPointF(end.x() + curvature, end.y())

        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        color = Theme.CONN_DRAGGING
        painter.setPen(QPen(color, 1.5, Qt.DashLine, Qt.RoundCap))
        painter.drawPath(self.path())


# ═══════════════════════════════════════════════════════════════════════════
# Agent Node - rectangular node representing an agent
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 9P Architecture Definitions
# ═══════════════════════════════════════════════════════════════════════════

# Standard 9P services and their ports.
# Only include services that are actually mounted.
# The llm service is not listed because its agents are shown as individual
# nodes. Additional services can be registered at runtime via
# OperatorPanel.register_service().
NINE_P_SERVICES = {}


def _rio_port_defs(mount: str):
    """Return the standard left/right port definitions for a Rio display
    server instance (rioa, riob, rioc, …).  The *mount* path is baked
    into each PortDef.file_path by the NinePNode constructor, so we
    only need the abstract names here."""
    return {
        "left_ports": [
            PortDef("scene/parse", PortDirection.LEFT, Theme.PORT_SYSTEM,
                     description="Write Python code to render"),
            PortDef("scene/ctl", PortDirection.LEFT, Theme.PORT_CTL,
                     description="Scene control commands"),
            PortDef("ctl", PortDirection.LEFT, Theme.PORT_CTL,
                     description="Root control"),
        ],
        "right_ports": [
            PortDef("scene/stdout", PortDirection.RIGHT, Theme.PORT_OUTPUT,
                     description="Execution stdout"),
            PortDef("scene/STDERR", PortDirection.RIGHT, Theme.PORT_ERRORS,
                     description="Execution errors (blocking)"),
            PortDef("CONTEXT", PortDirection.RIGHT, Theme.PORT_OUTPUT,
                     description="All successful code (blocking)"),
        ],
    }


import re as _re
_RIO_NAME_RE = _re.compile(r'^rio[a-z]+$')


def _lookup_service(name: str, mount: str):
    """Return a service definition dict if *name* is a known service type.

    Matches:
      - Any name registered in NINE_P_SERVICES at runtime.
      - Any name matching the pattern ``rio<letter(s)>`` (rioa, riob, …)
        which gets the standard Rio display server port template.

    Returns None for unknown services.
    """
    if name in NINE_P_SERVICES:
        return NINE_P_SERVICES[name]

    if _RIO_NAME_RE.match(name):
        defs = _rio_port_defs(mount)
        return {
            "description": "Rio Display Server",
            "mount": mount,
            **defs,
        }

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Base Node - shared paint/layout/interaction logic
# ═══════════════════════════════════════════════════════════════════════════

class BaseNode(QGraphicsRectItem):
    """
    Base class for all graph nodes (agents, 9P services).
    Handles common painting, ports, layout, and interaction.
    Supports collapse/expand: double-click to toggle, but only if
    no connections are attached to any port (stays expanded if wired).
    """

    def __init__(self, node_name: str, header_color: QColor,
                 scene_ref: 'OperatorGraphScene'):
        super().__init__()

        self.node_name = node_name
        self.header_color = header_color
        self.scene_ref = scene_ref

        self.left_ports: List[Port] = []
        self.right_ports: List[Port] = []

        # Collapse state
        self._collapsed = False

        # Flags
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        self._hovered = False

    def _finish_init(self):
        """Call after ports are built to do layout and add title.
        Safe to call multiple times (e.g. after adding terminal ports)."""
        self._layout()

        # Drop shadow (only once)
        if not self.graphicsEffect():
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(24)
            shadow.setColor(Theme.NODE_SHADOW)
            shadow.setOffset(0, 4)
            self.setGraphicsEffect(shadow)

        # Precompute elided title text for paint()
        fm = QFontMetrics(Theme.FONT_NODE_TITLE)
        max_title_w = Theme.NODE_WIDTH - 36
        self._display_title = fm.elidedText(
            self.node_name, Qt.ElideRight, int(max_title_w))

    def _layout(self):
        if self._collapsed:
            # Collapsed: just the header bar, no ports visible
            total_height = Theme.NODE_HEADER_HEIGHT
            self.setRect(0, 0, Theme.NODE_WIDTH, total_height)
            for port in self.left_ports + self.right_ports:
                port.setVisible(False)
                port._label.setVisible(False)
        else:
            max_rows = max(len(self.left_ports), len(self.right_ports), 1)
            body_height = Theme.PORT_MARGIN_TOP + max_rows * Theme.PORT_SPACING + 12
            total_height = Theme.NODE_HEADER_HEIGHT + body_height

            self.setRect(0, 0, Theme.NODE_WIDTH, total_height)

            y_start = Theme.NODE_HEADER_HEIGHT + Theme.PORT_MARGIN_TOP
            for i, port in enumerate(self.left_ports):
                port.setVisible(True)
                port._label.setVisible(True)
                port.setPos(0, y_start + i * Theme.PORT_SPACING)
                port.update_label_pos()
            for i, port in enumerate(self.right_ports):
                port.setVisible(True)
                port._label.setVisible(True)
                port.setPos(Theme.NODE_WIDTH, y_start + i * Theme.PORT_SPACING)
                port.update_label_pos()

    def has_connections(self) -> bool:
        """Return True if any port on this node has an active connection."""
        for port in self.left_ports + self.right_ports:
            if port.connections:
                return True
        return False

    def toggle_collapse(self):
        """Toggle collapsed state. Nodes with connections stay expanded."""
        if not self._collapsed:
            # Can only collapse if no connections
            if self.has_connections():
                return
            self._collapsed = True
        else:
            self._collapsed = False
        self._finish_init()
        # Update all connections on neighboring nodes
        for port in self.left_ports + self.right_ports:
            for conn in port.connections:
                conn.update_path()

    def mouseDoubleClickEvent(self, event):
        """Double-click to toggle collapse/expand."""
        if event.button() == Qt.LeftButton:
            self.toggle_collapse()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        radius = Theme.NODE_CORNER_RADIUS

        if self.isSelected():
            border_color = Theme.NODE_BORDER_SELECTED
            border_width = 1.5
        elif self._hovered:
            border_color = Theme.NODE_BORDER_HOVER
            border_width = 1.0
        else:
            border_color = Theme.NODE_BORDER
            border_width = 0.75

        # ── Body — translucent dark glass ─────────────────────────────────
        body_path = QPainterPath()
        body_path.addRoundedRect(rect, radius, radius)
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(Theme.NODE_BG))
        painter.drawPath(body_path)

        # ── Header band — subtle tinted strip ─────────────────────────────
        header_rect = QRectF(rect.x(), rect.y(),
                             rect.width(), Theme.NODE_HEADER_HEIGHT)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, radius, radius)
        # clip bottom corners of header to body
        clip_rect = QRectF(rect.x(), rect.y() + radius,
                           rect.width(), Theme.NODE_HEADER_HEIGHT - radius)
        header_path.addRect(clip_rect)

        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.header_color))
        painter.setClipPath(body_path)
        painter.drawPath(header_path)
        painter.restore()

        # ── Thin separator line ───────────────────────────────────────────
        sep_y = rect.y() + Theme.NODE_HEADER_HEIGHT
        painter.setPen(QPen(Theme.SEPARATOR_COLOR, 0.5))
        painter.drawLine(QPointF(rect.x() + 1, sep_y),
                         QPointF(rect.right() - 1, sep_y))

        # ── Title text — light, airy ─────────────────────────────────────
        title = getattr(self, '_display_title', self.node_name)
        painter.save()
        painter.setFont(Theme.FONT_NODE_TITLE)
        painter.setPen(Theme.TEXT_ON_HEADER)
        painter.setBrush(Qt.NoBrush)
        title_rect = QRectF(rect.x() + 12, rect.y(),
                            rect.width() - 32, Theme.NODE_HEADER_HEIGHT)
        painter.drawText(title_rect, Qt.AlignVCenter | Qt.AlignLeft, title)
        painter.restore()

        # ── Collapse indicator — minimal ──────────────────────────────────
        if self._collapsed:
            indicator = "›"
        else:
            indicator = "‹"
        painter.setPen(QPen(Theme.COLLAPSE_INDICATOR_COLOR))
        painter.setFont(QFont(Theme.FONT_FAMILY, 9))
        painter.drawText(
            QRectF(rect.right() - 22, rect.y(), 18, Theme.NODE_HEADER_HEIGHT),
            Qt.AlignCenter, indicator)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for port in self.left_ports + self.right_ports:
                for conn in port.connections:
                    conn.update_path()
        return super().itemChange(change, value)

    def get_port(self, name: str) -> Optional[Port]:
        for p in self.left_ports + self.right_ports:
            if p.port_def.name == name:
                return p
        return None

    def cleanup(self):
        """Override in subclasses to stop timers etc."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 9P Service Node - represents /n/rioa, /n/riob, etc.
# ═══════════════════════════════════════════════════════════════════════════

def _node_header_9p():
    return Theme.NODE_HEADER_9P

class NinePNode(BaseNode):
    """
    A node representing a 9P filesystem service (rioa, riob, llm, etc.).
    Ports map to files under the service's mount point.
    Terminal ports are added after construction via add_terminal().
    """

    PORT_TERM = QColor(100, 160, 200)

    def __init__(self, service_name: str, mount_path: str,
                 left_port_defs: List[PortDef],
                 right_port_defs: List[PortDef],
                 scene_ref: 'OperatorGraphScene',
                 description: str = ""):
        super().__init__(service_name, _node_header_9p(), scene_ref)

        self.service_name = service_name
        self.mount_path = mount_path
        self.description = description
        self._terminal_ids: List[str] = []

        for pdef in left_port_defs:
            fp = os.path.join(mount_path, pdef.name)
            pd = PortDef(pdef.name, pdef.direction, pdef.color,
                         file_path=fp, description=pdef.description)
            self.left_ports.append(Port(pd, self))

        for pdef in right_port_defs:
            fp = os.path.join(mount_path, pdef.name)
            pd = PortDef(pdef.name, pdef.direction, pdef.color,
                         file_path=fp, description=pdef.description)
            self.right_ports.append(Port(pd, self))

        self._finish_init()

    def add_terminal(self, term_id: str):
        """Add ports for a discovered terminal directory."""
        if term_id in self._terminal_ids:
            return
        self._terminal_ids.append(term_id)

        short = term_id[:12] if len(term_id) > 12 else term_id
        term_base = f"{self.mount_path}/terms/{term_id}"

        # stdin: writable (left) — send commands to shell
        pd = PortDef(f"term_{short}/stdin", PortDirection.LEFT,
                     NinePNode.PORT_TERM,
                     file_path=f"{term_base}/stdin",
                     description=f"Terminal shell input")
        self.left_ports.append(Port(pd, self))

        # stdout: readable (right) — blocking shell output
        pd = PortDef(f"term_{short}/stdout", PortDirection.RIGHT,
                     NinePNode.PORT_TERM,
                     file_path=f"{term_base}/stdout",
                     description=f"Terminal shell output (blocking)")
        self.right_ports.append(Port(pd, self))

        # output: readable (right)
        pd = PortDef(f"term_{short}/output", PortDirection.RIGHT,
                     Theme.PORT_OUTPUT,
                     file_path=f"{term_base}/output",
                     description=f"Terminal output stream")
        self.right_ports.append(Port(pd, self))

        # routes: readable (right)
        pd = PortDef(f"term_{short}/routes", PortDirection.RIGHT,
                     Theme.PORT_DEFAULT,
                     file_path=f"{term_base}/routes",
                     description=f"Active routes (cat to view)")
        self.right_ports.append(Port(pd, self))

        self._finish_init()

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(_menu_stylesheet())

        header = menu.addAction(f"⛁  {self.service_name}  ({self.mount_path})")
        header.setEnabled(False)
        if self.description:
            desc = menu.addAction(f"    {self.description}")
            desc.setEnabled(False)
        menu.addSeparator()
        inspect = menu.addAction("🔍  Inspect mount")

        action = menu.exec(event.screenPos())
        if action == inspect:
            self._inspect_mount()

    def _inspect_mount(self):
        info = [f"Service: {self.service_name}",
                f"Mount: {self.mount_path}", ""]
        info.append("── Ports ──")
        for p in self.left_ports:
            info.append(f"  ← {p.port_def.name}")
        for p in self.right_ports:
            info.append(f"  → {p.port_def.name}")
        QMessageBox.information(None, f"9P: {self.service_name}",
                                "\n".join(info))


# ═══════════════════════════════════════════════════════════════════════════
# Agent Node
# ═══════════════════════════════════════════════════════════════════════════

class AgentNode(BaseNode):
    """
    A rectangular node representing an LLMFS agent.

    Structure:
      ┌──────────────────────────┐
      │  ■  agent_name           │  ← colored header
      ├──────────────────────────┤
      │ ● input          output ●│  ← ports on left/right edges
      │ ● system         errors ●│
      │ ● config                 │
      │ ● rules                  │
      │ ● history                │
      │ ● ctl                    │
      │ ● rioa           rioa  ●│  ← supplementary outputs
      └──────────────────────────┘
    """

    # Standard files that map to the fixed port definitions
    _STANDARD_LEFT = {p.name for p in AGENT_LEFT_PORTS}
    _STANDARD_RIGHT = {p.name for p in AGENT_RIGHT_PORTS}
    _STANDARD_ALL = _STANDARD_LEFT | _STANDARD_RIGHT

    def __init__(self, agent_name: str, agent_path: str,
                 scene_ref: 'OperatorGraphScene',
                 header_color: QColor = None,
                 known_files: List[str] = None):
        """
        Args:
            known_files: Pre-scanned list of filenames in this agent directory.
                         If None, only standard ports are created.
                         Supplementary files (anything not in the standard set)
                         become extra right-side ports.
        """
        super().__init__(agent_name,
                         header_color or Theme.NODE_HEADER_AGENT,
                         scene_ref)

        self.agent_name = agent_name
        self.agent_path = agent_path
        self._state_text = ""
        self._model_text = ""
        self._known_files = known_files or []

        # Build ports from agent filesystem
        self._build_ports()
        self._finish_init()

        # State badge
        self._badge = QGraphicsSimpleTextItem("", self)
        self._badge.setFont(Theme.FONT_BADGE)
        self._badge.setBrush(QBrush(Theme.TEXT_BADGE))

        # Periodic state refresh (via background thread)
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh_state)
        self._refresh_timer.start(3000)

    def _build_ports(self):
        """Create ports based on agent directory contents."""
        # Standard left ports (inputs)
        for pdef in AGENT_LEFT_PORTS:
            fp = os.path.join(self.agent_path, pdef.name)
            pd = PortDef(pdef.name, pdef.direction, pdef.color,
                         file_path=fp, description=pdef.description)
            port = Port(pd, self)
            self.left_ports.append(port)

        # Standard right ports (outputs)
        for pdef in AGENT_RIGHT_PORTS:
            fp = os.path.join(self.agent_path, pdef.name)
            pd = PortDef(pdef.name, pdef.direction, pdef.color,
                         file_path=fp, description=pdef.description)
            port = Port(pd, self)
            self.right_ports.append(port)

        # Supplementary output files – anything in the directory that is
        # NOT one of the standard files.
        for fname in self._known_files:
            if fname not in self._STANDARD_ALL:
                fp = os.path.join(self.agent_path, fname)
                pd = PortDef(fname, PortDirection.RIGHT,
                             Theme.PORT_SUPPLEMENTARY,
                             file_path=fp,
                             description=f"Supplementary: {fname}")
                port = Port(pd, self)
                self.right_ports.append(port)

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)

        # State indicator dot (agent-specific)
        if self._state_text:
            rect = self.rect()
            dot_color = {
                "idle": QColor(90, 90, 100),
                "streaming": QColor(110, 160, 125),
                "done": QColor(140, 150, 170),
                "error": QColor(160, 90, 90),
                "cancelled": QColor(155, 135, 95),
            }.get(self._state_text, QColor(90, 90, 100))

            dot_x = rect.right() - 18
            dot_y = Theme.NODE_HEADER_HEIGHT / 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(dot_color))
            painter.drawEllipse(QPointF(dot_x, dot_y), 4, 4)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(_menu_stylesheet())

        header = menu.addAction(f"⬢  {self.agent_name}")
        header.setEnabled(False)
        menu.addSeparator()

        set_system = menu.addAction("📄  Set System Prompt (.md)...")
        set_model = menu.addAction("🔧  Set Model...")
        set_temp = menu.addAction("🌡  Set Temperature...")
        menu.addSeparator()

        clear_hist = menu.addAction("🗑  Clear History")
        cancel_gen = menu.addAction("⏹  Cancel Generation")
        retry_gen = menu.addAction("🔄  Retry Last")
        menu.addSeparator()

        add_rule = menu.addAction("📐  Add Plumbing Rule...")
        menu.addSeparator()

        inspect = menu.addAction("🔍  Inspect")
        delete = menu.addAction("✕   Delete Agent")

        action = menu.exec(event.screenPos())
        if not action:
            return

        if action == set_system:
            self._set_system_prompt()
        elif action == set_model:
            self._set_model()
        elif action == set_temp:
            self._set_temperature()
        elif action == clear_hist:
            self._write_ctl("clear")
        elif action == cancel_gen:
            self._write_ctl("cancel")
        elif action == retry_gen:
            self._write_ctl("retry")
        elif action == add_rule:
            self._add_plumbing_rule()
        elif action == inspect:
            self._inspect()
        elif action == delete:
            self.scene_ref.delete_agent_node(self)

    # ── Agent filesystem operations ─────────────────────────────────────

    def _write_file(self, filename: str, content: str):
        """Write content to an agent file via background thread."""
        fp = os.path.join(self.agent_path, filename)

        def _do_write():
            with open(fp, 'w') as f:
                f.write(content)

        panel = self._find_panel()
        if panel and panel._fs_worker:
            panel._fs_worker.run_async(_do_write)
        else:
            # Fallback: direct write (may block briefly for writes,
            # which are typically fast on 9P)
            try:
                _do_write()
            except Exception as e:
                QMessageBox.warning(None, "Write Error",
                                    f"Could not write to {fp}:\n{e}")

    def _read_file_async(self, filename: str, callback: Callable):
        """Read content from an agent file on a background thread."""
        fp = os.path.join(self.agent_path, filename)
        panel = self._find_panel()
        if panel and panel._fs_worker:
            panel._fs_worker.run_async(_fs_read, fp, callback=callback)
        else:
            # Fallback: direct non-blocking read
            callback(_fs_read(fp))

    def _read_file(self, filename: str) -> str:
        """
        Synchronous read using O_NONBLOCK.
        Use _read_file_async for potentially blocking files (output, history).
        Safe for small metadata files (ctl, config).
        """
        return _fs_read(os.path.join(self.agent_path, filename))

    def _write_ctl(self, command: str):
        """Send a control command."""
        self._write_file("ctl", command)

    def _set_system_prompt(self):
        """Open file dialog to select a .md file as system prompt."""
        path, _ = QFileDialog.getOpenFileName(
            None, "Select System Prompt",
            os.path.expanduser("~"),
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'r') as f:
                    content = f.read()
                self._write_file("system", content)
                QMessageBox.information(None, "System Prompt Set",
                                        f"Loaded {os.path.basename(path)} "
                                        f"({len(content)} chars)")
            except Exception as e:
                QMessageBox.warning(None, "Error", str(e))

    def _set_model(self):
        """Dialog to set the model."""
        current = ""
        try:
            config_text = self._read_file("config")
            if config_text:
                cfg = json.loads(config_text)
                current = cfg.get("model", "")
        except Exception:
            pass

        model, ok = QInputDialog.getText(
            None, "Set Model", "Model name:", text=current)
        if ok and model:
            self._write_ctl(f"model {model}")

    def _set_temperature(self):
        """Dialog to set temperature."""
        temp, ok = QInputDialog.getDouble(
            None, "Set Temperature", "Temperature (0.0 - 2.0):",
            value=0.7, min=0.0, max=2.0, decimals=2)
        if ok:
            self._write_ctl(f"temperature {temp}")

    def _add_plumbing_rule(self):
        """Dialog to add a plumbing rule."""
        rule, ok = QInputDialog.getText(
            None, "Add Plumbing Rule",
            "Format: pattern -> {capture_name}\n"
            "Example: ```(?P<bash>\\S+)\\n(?P<code>.*?)``` -> {bash}")
        if ok and rule:
            self._write_file("rules", rule)
            # Refresh ports to show new supplementary output
            QTimer.singleShot(500, self._refresh_ports)

    def _inspect(self):
        """Show detailed info about this agent."""
        info_parts = [f"Agent: {self.agent_name}", f"Path: {self.agent_path}", ""]

        # Read ctl for status
        ctl = self._read_file("ctl")
        if ctl:
            info_parts.append("── Status ──")
            info_parts.append(ctl)
            info_parts.append("")

        # Read config
        config = self._read_file("config")
        if config:
            info_parts.append("── Config ──")
            info_parts.append(config)
            info_parts.append("")

        # Read system (truncated)
        system = self._read_file("system")
        if system:
            info_parts.append("── System Prompt ──")
            if len(system) > 200:
                info_parts.append(system[:200] + "...")
            else:
                info_parts.append(system)
            info_parts.append("")

        # Read rules
        rules = self._read_file("rules")
        if rules:
            info_parts.append("── Rules ──")
            info_parts.append(rules)

        info_parts.append(f"\nPorts: {len(self.left_ports)} in, "
                          f"{len(self.right_ports)} out")
        total_conns = sum(len(p.connections)
                         for p in self.left_ports + self.right_ports)
        info_parts.append(f"Connections: {total_conns}")

        QMessageBox.information(None, f"Agent: {self.agent_name}",
                                "\n".join(info_parts))

    def _refresh_state(self):
        """
        Read agent ctl status on a background thread and update visual state.
        Uses O_NONBLOCK reads to avoid deadlocking.
        """
        panel = self._find_panel()
        if panel and panel._fs_worker:
            ctl_path = os.path.join(self.agent_path, 'ctl')
            panel._fs_worker.run_async(
                _fs_read, ctl_path,
                callback=self._on_state_read
            )

    def _on_state_read(self, result):
        """Callback when ctl read completes."""
        if isinstance(result, Exception) or not result:
            return
        text = result if isinstance(result, str) else ""
        # Parse "state <value>" line
        for line in text.splitlines():
            if line.startswith("state "):
                self._state_text = line.split(None, 1)[1].strip()
            elif line.startswith("model "):
                self._model_text = line.split(None, 1)[1].strip()
        self.update()

    def _refresh_ports(self):
        """
        Rescan agent directory on a background thread for new supplementary ports.
        """
        panel = self._find_panel()
        if panel and panel._fs_worker:
            panel._fs_worker.run_async(
                _fs_listdir, self.agent_path,
                callback=self._on_ports_scanned
            )

    def _on_ports_scanned(self, result):
        """Callback when agent dir listing completes."""
        if isinstance(result, Exception) or not result:
            return
        files = result if isinstance(result, list) else []
        existing_names = {p.port_def.name for p in self.right_ports}
        changed = False
        for fname in files:
            if fname not in self._STANDARD_ALL and fname not in existing_names:
                fp = os.path.join(self.agent_path, fname)
                pd = PortDef(fname, PortDirection.RIGHT,
                             Theme.PORT_SUPPLEMENTARY,
                             file_path=fp,
                             description=f"Supplementary: {fname}")
                self.right_ports.append(Port(pd, self))
                changed = True
        if changed:
            self._finish_init()

    def _find_panel(self) -> Optional['OperatorPanel']:
        """Walk up to find the OperatorPanel (for FSWorker access)."""
        app = QApplication.instance()
        if app is None:
            return None
        for widget in app.allWidgets():
            if isinstance(widget, OperatorPanel) and widget.scene is self.scene_ref:
                return widget
        return None

    def get_port(self, name: str) -> Optional[Port]:
        """Find a port by name."""
        for p in self.left_ports + self.right_ports:
            if p.port_def.name == name:
                return p
        return None

    def cleanup(self):
        """Stop timers before removal."""
        if self._refresh_timer:
            self._refresh_timer.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Text Node - read/write file node with embedded text area
# ═══════════════════════════════════════════════════════════════════════════

def _node_header_text():
    return Theme.NODE_HEADER_TEXT

class TextNode(BaseNode):
    """
    A utility node with an embedded text area and two ports:
      - input  (left):  the file to read from
      - output (right): the file to write to

    Buttons:
      - Read:  reads the input port's file into the text area
      - Write: writes the text area content to the output port's file

    A periodic-write timer with an enable checkbox can trigger
    automatic writes at a configurable interval (useful as a trigger).

    The node body is painted with a transparent background so the
    canvas grid shows through.
    """

    TEXT_NODE_WIDTH = 320
    TEXT_NODE_MIN_HEIGHT = 280

    def __init__(self, node_name: str, scene_ref: 'OperatorGraphScene',
                 input_path: str = "", output_path: str = ""):
        # We call BaseNode.__init__ but skip _finish_init — we do custom layout
        super().__init__(node_name, _node_header_text(), scene_ref)

        # Ensure the base QGraphicsRectItem has no fill
        self.setBrush(Qt.NoBrush)
        self.setPen(Qt.NoPen)

        self._input_path = input_path
        self._output_path = output_path

        # ── Ports ────────────────────────────────────────────────────────
        pd_in = PortDef("input", PortDirection.LEFT, Theme.PORT_INPUT,
                         file_path=input_path, description="File to read from")
        self.left_ports.append(Port(pd_in, self))

        pd_out = PortDef("output", PortDirection.RIGHT, Theme.PORT_OUTPUT,
                          file_path=output_path, description="File to write to")
        self.right_ports.append(Port(pd_out, self))

        # ── Embedded widget (proxy) ──────────────────────────────────────
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setZValue(4)

        container = QWidget()
        container.setAttribute(Qt.WA_TranslucentBackground)
        container.setAttribute(Qt.WA_NoSystemBackground)
        container.setStyleSheet("background: transparent;")
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(8, 4, 8, 6)
        vlayout.setSpacing(4)

        # ── Text area ────────────────────────────────────────────────────
        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText("Text content…")
        self._text_edit.setFont(QFont(Theme.FONT_FAMILY_MONO, 9))
        self._text_edit.setMinimumHeight(120)
        self._text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.TEXT_EDIT_BG};
                color: {Theme.TEXT_PRIMARY.name()};
                border: 1px solid {Theme.TEXT_EDIT_BORDER};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {Theme.TEXT_EDIT_SELECTION};
            }}
        """)
        vlayout.addWidget(self._text_edit)

        # ── Button row ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_style_read = f"""
            QPushButton {{
                background-color: {Theme.BTN_READ_BG};
                color: {Theme.TEXT_PRIMARY.name()}; border: 1px solid {Theme.BTN_READ_BORDER};
                border-radius: 4px;
                padding: 4px 14px; font-size: 10px; font-weight: 500;
            }}
            QPushButton:hover {{ background-color: {Theme.BTN_READ_HOVER}; }}
            QPushButton:pressed {{ background-color: {Theme.BTN_READ_PRESS}; }}
        """
        btn_style_write = f"""
            QPushButton {{
                background-color: {Theme.BTN_WRITE_BG};
                color: {Theme.TEXT_PRIMARY.name()}; border: 1px solid {Theme.BTN_WRITE_BORDER};
                border-radius: 4px;
                padding: 4px 14px; font-size: 10px; font-weight: 500;
            }}
            QPushButton:hover {{ background-color: {Theme.BTN_WRITE_HOVER}; }}
            QPushButton:pressed {{ background-color: {Theme.BTN_WRITE_PRESS}; }}
        """

        self._read_btn = QPushButton("▼ Read")
        self._read_btn.setCursor(Qt.PointingHandCursor)
        self._read_btn.setStyleSheet(btn_style_read)
        self._read_btn.clicked.connect(self._do_read)
        btn_row.addWidget(self._read_btn)

        self._write_btn = QPushButton("▲ Write")
        self._write_btn.setCursor(Qt.PointingHandCursor)
        self._write_btn.setStyleSheet(btn_style_write)
        self._write_btn.clicked.connect(self._do_write)
        btn_row.addWidget(self._write_btn)

        btn_row.addStretch()
        vlayout.addLayout(btn_row)

        # ── Timer / periodic-write row ───────────────────────────────────
        from PySide6.QtWidgets import QCheckBox
        timer_row = QHBoxLayout()
        timer_row.setSpacing(6)

        self._timer_enable = QCheckBox("Auto-write")
        self._timer_enable.setStyleSheet(f"""
            QCheckBox {{
                color: {Theme.TEXT_SECONDARY.name()};
                font-size: 10px;
            }}
            QCheckBox::indicator {{
                width: 13px; height: 13px;
                border: 1px solid {Theme.CHECKBOX_BORDER};
                border-radius: 3px;
                background: {Theme.CHECKBOX_BG};
            }}
            QCheckBox::indicator:checked {{
                background: {Theme.CHECKBOX_CHECKED_BG};
                border-color: {Theme.CHECKBOX_CHECKED_BORDER};
            }}
        """)
        self._timer_enable.toggled.connect(self._on_timer_toggled)
        timer_row.addWidget(self._timer_enable)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 3600)
        self._interval_spin.setValue(5)
        self._interval_spin.setSuffix("s")
        self._interval_spin.setFixedWidth(70)
        self._interval_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Theme.SPINBOX_BG};
                color: {Theme.TEXT_PRIMARY.name()};
                border: 1px solid {Theme.SPINBOX_BORDER};
                border-radius: 3px; padding: 2px 4px; font-size: 10px;
            }}
        """)
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        timer_row.addWidget(self._interval_spin)

        self._countdown_label = QLabel("")
        self._countdown_label.setStyleSheet(
            f"color: {Theme.TEXT_SECONDARY.name()}; font-size: 10px;")
        timer_row.addWidget(self._countdown_label)

        timer_row.addStretch()
        vlayout.addLayout(timer_row)

        container.setFixedWidth(self.TEXT_NODE_WIDTH - 16)
        self._proxy.setWidget(container)

        # ── Timer internals ──────────────────────────────────────────────
        self._periodic_timer = QTimer()
        self._periodic_timer.timeout.connect(self._tick)
        self._remaining = 0

        # ── Finish layout ────────────────────────────────────────────────
        self._finish_init()

    def _finish_init(self):
        """Custom _finish_init that skips the drop shadow for full transparency."""
        self._layout()

        # Title text — create or update (no shadow)
        if not hasattr(self, '_title') or self._title is None:
            self._title = QGraphicsSimpleTextItem(self.node_name, self)
            self._title.setFont(Theme.FONT_NODE_TITLE)
            self._title.setBrush(QBrush(Theme.TEXT_ON_HEADER))
            self._title.setZValue(5)

        tr = self._title.boundingRect()
        max_title_w = self.TEXT_NODE_WIDTH - 36
        if tr.width() > max_title_w:
            fm = QFontMetrics(Theme.FONT_NODE_TITLE)
            elided = fm.elidedText(self.node_name, Qt.ElideRight,
                                    int(max_title_w))
            self._title.setText(elided)
            tr = self._title.boundingRect()
        self._title.setPos(12, (Theme.NODE_HEADER_HEIGHT - tr.height()) / 2)

    # ── Layout override ──────────────────────────────────────────────────

    def _layout(self):
        """Custom layout: ports + embedded widget below the header."""
        proxy_h = self._proxy.widget().sizeHint().height() if self._proxy.widget() else 160
        port_rows = max(len(self.left_ports), len(self.right_ports), 1)
        port_section_h = Theme.PORT_MARGIN_TOP + port_rows * Theme.PORT_SPACING + 8

        body_height = port_section_h + proxy_h + 8
        total_height = max(Theme.NODE_HEADER_HEIGHT + body_height,
                           self.TEXT_NODE_MIN_HEIGHT)

        self.setRect(0, 0, self.TEXT_NODE_WIDTH, total_height)

        # Ports
        y_start = Theme.NODE_HEADER_HEIGHT + Theme.PORT_MARGIN_TOP
        for i, port in enumerate(self.left_ports):
            port.setPos(0, y_start + i * Theme.PORT_SPACING)
            port.update_label_pos()
        for i, port in enumerate(self.right_ports):
            port.setPos(self.TEXT_NODE_WIDTH, y_start + i * Theme.PORT_SPACING)
            port.update_label_pos()

        # Proxy widget sits below the port area
        self._proxy.setPos(8, Theme.NODE_HEADER_HEIGHT + port_section_h)

    # ── Paint override — transparent body ────────────────────────────────

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        radius = Theme.NODE_CORNER_RADIUS

        if self.isSelected():
            border_color = Theme.NODE_BORDER_SELECTED
            border_width = 1.5
        elif self._hovered:
            border_color = Theme.NODE_BORDER_HOVER
            border_width = 1.0
        else:
            border_color = Theme.NODE_BORDER
            border_width = 0.75

        # Translucent body
        body_path = QPainterPath()
        body_path.addRoundedRect(rect, radius, radius)
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(Theme.NODE_BODY_TRANSLUCENT))
        painter.drawPath(body_path)

        # Header (opaque-ish)
        header_rect = QRectF(rect.x(), rect.y(),
                             rect.width(), Theme.NODE_HEADER_HEIGHT)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, radius, radius)
        clip_rect = QRectF(rect.x(), rect.y() + radius,
                           rect.width(), Theme.NODE_HEADER_HEIGHT - radius)
        header_path.addRect(clip_rect)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.header_color))
        painter.setClipPath(body_path)
        painter.drawPath(header_path)
        painter.setClipping(False)

        # Separator line
        sep_y = rect.y() + Theme.NODE_HEADER_HEIGHT
        painter.setPen(QPen(Theme.SEPARATOR_COLOR, 0.5))
        painter.drawLine(QPointF(rect.x() + 1, sep_y),
                         QPointF(rect.right() - 1, sep_y))

    # ── Read / Write ─────────────────────────────────────────────────────

    def _resolve_input_path(self) -> str:
        """Return the file path to read from — the input port's connected source
        or the port's own file_path."""
        port = self.left_ports[0] if self.left_ports else None
        if port and port.connections:
            # Use the source port's file path of the first connection
            conn = port.connections[0]
            src = conn.source_port if conn.source_port != port else conn.target_port
            return src.port_def.file_path
        return port.port_def.file_path if port else self._input_path

    def _resolve_output_path(self) -> str:
        """Return the file path to write to — the output port's connected target
        or the port's own file_path."""
        port = self.right_ports[0] if self.right_ports else None
        if port and port.connections:
            conn = port.connections[0]
            dst = conn.target_port if conn.target_port != port else conn.source_port
            return dst.port_def.file_path
        return port.port_def.file_path if port else self._output_path

    def _do_read(self):
        """Read the input file on a background thread and display contents."""
        path = self._resolve_input_path()
        if not path:
            self._flash_status("No input path")
            return
        
        # Find the panel's FSWorker
        panel = self._find_panel()
        if panel and panel._fs_worker:
            self._flash_status("Reading…")
            panel._fs_worker.run_async(
                _fs_read, path, 65536,
                callback=self._on_read_done
            )
        else:
            # Fallback: direct non-blocking read
            content = _fs_read(path, 65536)
            self._text_edit.setPlainText(content)
            self._flash_status(f"Read {len(content)} chars")

    def _on_read_done(self, result):
        """Callback when background read completes."""
        if isinstance(result, Exception):
            self._flash_status("Read error")
            return
        content = result if isinstance(result, str) else ""
        self._text_edit.setPlainText(content)
        self._flash_status(f"Read {len(content)} chars")

    def _do_write(self):
        """Write the text area content to the output file via background thread."""
        path = self._resolve_output_path()
        if not path:
            self._flash_status("No output path")
            return
        content = self._text_edit.toPlainText()

        def _write():
            with open(path, 'w') as f:
                f.write(content)
            return len(content)

        panel = self._find_panel()
        if panel and panel._fs_worker:
            self._flash_status("Writing…")
            panel._fs_worker.run_async(
                _write,
                callback=lambda n: self._flash_status(
                    f"Wrote {n} chars" if not isinstance(n, Exception)
                    else "Write error")
            )
        else:
            try:
                _write()
                self._flash_status(f"Wrote {len(content)} chars")
            except Exception as e:
                self._flash_status("Write error")
                QMessageBox.warning(None, "Write Error",
                                    f"Could not write to {path}:\n{e}")

    def _find_panel(self) -> Optional['OperatorPanel']:
        """Walk up to find OperatorPanel for FSWorker access."""
        app = QApplication.instance()
        if app is None:
            return None
        for widget in app.allWidgets():
            if isinstance(widget, OperatorPanel) and widget.scene is self.scene_ref:
                return widget
        return None

    def _flash_status(self, msg: str):
        """Briefly show a status message in the countdown label."""
        self._countdown_label.setText(msg)
        QTimer.singleShot(2000, lambda: (
            self._countdown_label.setText("")
            if not self._timer_enable.isChecked() else None))

    # ── Timer / periodic write ───────────────────────────────────────────

    def _on_timer_toggled(self, enabled: bool):
        if enabled:
            self._remaining = self._interval_spin.value()
            self._countdown_label.setText(f"{self._remaining}s")
            self._periodic_timer.start(1000)
        else:
            self._periodic_timer.stop()
            self._remaining = 0
            self._countdown_label.setText("")

    def _on_interval_changed(self, val: int):
        if self._timer_enable.isChecked():
            # Reset countdown to the new interval
            self._remaining = val

    def _tick(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self._do_write()
            self._remaining = self._interval_spin.value()
        self._countdown_label.setText(f"{self._remaining}s")

    # ── Context menu ─────────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(_menu_stylesheet())

        header = menu.addAction(f"📝  {self.node_name}")
        header.setEnabled(False)
        menu.addSeparator()

        read_act = menu.addAction("▼  Read input")
        write_act = menu.addAction("▲  Write output")
        menu.addSeparator()
        set_in = menu.addAction("📂  Set input path…")
        set_out = menu.addAction("💾  Set output path…")
        menu.addSeparator()
        delete_act = menu.addAction("✕   Delete")

        action = menu.exec(event.screenPos())
        if action == read_act:
            self._do_read()
        elif action == write_act:
            self._do_write()
        elif action == set_in:
            self._set_path("input")
        elif action == set_out:
            self._set_path("output")
        elif action == delete_act:
            self.scene_ref.delete_text_node(self)

    def _set_path(self, which: str):
        current = (self._input_path if which == "input" else self._output_path)
        path, ok = QInputDialog.getText(
            None, f"Set {which} path",
            f"Filesystem path for the {which} port:",
            text=current)
        if ok and path:
            if which == "input":
                self._input_path = path
                if self.left_ports:
                    self.left_ports[0].port_def.file_path = path
            else:
                self._output_path = path
                if self.right_ports:
                    self.right_ports[0].port_def.file_path = path

    def cleanup(self):
        self._periodic_timer.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Debug Node - input-only sink that routes messages to a main-window overlay
# ═══════════════════════════════════════════════════════════════════════════

def _node_header_debug():
    return Theme.NODE_HEADER_DEBUG


class DebugNode(BaseNode):
    """
    A special input-only node used as a debug message sink.

    Features:
      - N input ports (adjustable via a spinbox on the node body).
      - No output ports.
      - When data arrives via a connection, the message is forwarded to a
        DebugOverlayWidget on the main RioWindow (top-right corner).
      - The overlay auto-hides when no connections exist on this node.
      - Each message is tagged with the input port name it arrived on.
    """

    DEBUG_NODE_WIDTH = 240
    DEBUG_NODE_MIN_HEIGHT = 140

    # Class-level signal bridge so that any instance can reach the overlay.
    # Populated by OperatorPanel / RioWindow on construction.
    _overlay_ref = None  # type: Optional[Any]  # DebugOverlayWidget

    def __init__(self, node_name: str, scene_ref: 'OperatorGraphScene',
                 initial_inputs: int = 2):
        super().__init__(node_name, _node_header_debug(), scene_ref)

        self.setBrush(Qt.NoBrush)
        self.setPen(Qt.NoPen)

        self._num_inputs = max(1, initial_inputs)

        # ── Embedded spinbox widget ──────────────────────────────────────
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setZValue(4)

        container = QWidget()
        container.setAttribute(Qt.WA_TranslucentBackground)
        container.setAttribute(Qt.WA_NoSystemBackground)
        container.setStyleSheet(f"background: {Theme.CONTAINER_BG};")
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(8, 4, 8, 6)
        vlayout.setSpacing(4)

        # Input count row
        row = QHBoxLayout()
        row.setSpacing(6)

        lbl = QLabel("Inputs:")
        lbl.setStyleSheet(f"color: {Theme.TEXT_SECONDARY.name()}; font-size: 10px;")
        row.addWidget(lbl)

        self._input_spin = QSpinBox()
        self._input_spin.setRange(1, 16)
        self._input_spin.setValue(self._num_inputs)
        self._input_spin.setFixedWidth(60)
        self._input_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Theme.SPINBOX_BG};
                color: {Theme.TEXT_PRIMARY.name()};
                border: 1px solid {Theme.SPINBOX_BORDER};
                border-radius: 3px; padding: 2px 4px; font-size: 10px;
            }}
        """)
        self._input_spin.valueChanged.connect(self._on_input_count_changed)
        row.addWidget(self._input_spin)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            f"color: {Theme.TEXT_SECONDARY.name()}; font-size: 10px;")
        row.addWidget(self._status_label)

        row.addStretch()
        vlayout.addLayout(row)

        container.setFixedWidth(self.DEBUG_NODE_WIDTH - 16)
        self._proxy.setWidget(container)

        # Build initial ports
        self._rebuild_input_ports()
        self._finish_init()

        # Periodic check: poll connected sources and push to overlay
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_inputs)
        self._poll_timer.start(2000)

        # Track what we've already shown (to avoid re-showing identical content)
        self._last_content: Dict[str, str] = {}

    # ── Port management ──────────────────────────────────────────────────

    def _rebuild_input_ports(self):
        """Rebuild left (input) ports to match self._num_inputs."""
        # Remove old ports and their connections
        for port in self.left_ports[:]:
            for conn in port.connections[:]:
                scene = self.scene()
                if scene and isinstance(scene, OperatorGraphScene):
                    scene.remove_connection(conn)
        self.left_ports.clear()

        for i in range(self._num_inputs):
            name = f"in_{i}"
            pd = PortDef(name, PortDirection.LEFT, Theme.PORT_INPUT,
                         file_path="", description=f"Debug input {i}")
            self.left_ports.append(Port(pd, self))

        # Never any right ports
        self.right_ports.clear()

    def _on_input_count_changed(self, val: int):
        self._num_inputs = val
        self._rebuild_input_ports()
        self._finish_init()
        self._update_overlay_visibility()

    # ── Layout / paint ───────────────────────────────────────────────────

    def _finish_init(self):
        """Custom _finish_init – transparent body, no drop shadow."""
        self._layout()

        if not hasattr(self, '_title') or self._title is None:
            self._title = QGraphicsSimpleTextItem(self.node_name, self)
            self._title.setFont(Theme.FONT_NODE_TITLE)
            self._title.setBrush(QBrush(Theme.TEXT_ON_HEADER))
            self._title.setZValue(5)

        tr = self._title.boundingRect()
        max_title_w = self.DEBUG_NODE_WIDTH - 36
        if tr.width() > max_title_w:
            fm = QFontMetrics(Theme.FONT_NODE_TITLE)
            elided = fm.elidedText(self.node_name, Qt.ElideRight,
                                    int(max_title_w))
            self._title.setText(elided)
            tr = self._title.boundingRect()
        self._title.setPos(12, (Theme.NODE_HEADER_HEIGHT - tr.height()) / 2)

    def _layout(self):
        proxy_h = self._proxy.widget().sizeHint().height() if self._proxy.widget() else 60
        port_rows = max(len(self.left_ports), 1)
        port_section_h = Theme.PORT_MARGIN_TOP + port_rows * Theme.PORT_SPACING + 8

        body_height = port_section_h + proxy_h + 8
        total_height = max(Theme.NODE_HEADER_HEIGHT + body_height,
                           self.DEBUG_NODE_MIN_HEIGHT)

        self.setRect(0, 0, self.DEBUG_NODE_WIDTH, total_height)

        y_start = Theme.NODE_HEADER_HEIGHT + Theme.PORT_MARGIN_TOP
        for i, port in enumerate(self.left_ports):
            port.setPos(0, y_start + i * Theme.PORT_SPACING)
            port.update_label_pos()

        self._proxy.setPos(8, Theme.NODE_HEADER_HEIGHT + port_section_h)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        radius = Theme.NODE_CORNER_RADIUS

        if self.isSelected():
            border_color = Theme.NODE_BORDER_SELECTED
            border_width = 1.5
        elif self._hovered:
            border_color = Theme.NODE_BORDER_HOVER
            border_width = 1.0
        else:
            border_color = Theme.NODE_BORDER
            border_width = 0.75

        body_path = QPainterPath()
        body_path.addRoundedRect(rect, radius, radius)
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(Theme.NODE_BODY_TRANSLUCENT))
        painter.drawPath(body_path)

        # Header
        header_rect = QRectF(rect.x(), rect.y(),
                             rect.width(), Theme.NODE_HEADER_HEIGHT)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, radius, radius)
        clip_rect = QRectF(rect.x(), rect.y() + radius,
                           rect.width(), Theme.NODE_HEADER_HEIGHT - radius)
        header_path.addRect(clip_rect)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.header_color))
        painter.setClipPath(body_path)
        painter.drawPath(header_path)
        painter.setClipping(False)

        sep_y = rect.y() + Theme.NODE_HEADER_HEIGHT
        painter.setPen(QPen(Theme.SEPARATOR_COLOR, 0.5))
        painter.drawLine(QPointF(rect.x() + 1, sep_y),
                         QPointF(rect.right() - 1, sep_y))

    # ── Polling connected sources ────────────────────────────────────────

    def _poll_inputs(self):
        """Kick off background reads for each connected input.

        All filesystem I/O is dispatched to the FSWorker thread so that
        blocking 9P reads (e.g. scene/STDERR) never freeze the UI.
        """
        self._update_overlay_visibility()

        overlay = DebugNode._overlay_ref
        if overlay is None:
            return

        panel = self._find_panel()
        worker = panel._fs_worker if panel else None

        for port in self.left_ports:
            if not port.connections:
                continue
            for conn in port.connections:
                src = conn.source_port if conn.source_port != port else conn.target_port
                src_path = src.port_def.file_path
                if not src_path:
                    continue

                # Build the tag now (on the main thread where Qt objects are safe)
                src_label = f"{src.parent_node.node_name}/{src.port_def.name}"
                tag = f"{port.port_def.name} ← {src_label}"
                key = f"{port.port_def.name}:{src_path}"

                if worker:
                    # Dispatch to background thread — never blocks the UI
                    worker.run_async(
                        _fs_read, src_path, 4096,
                        callback=lambda content, _t=tag, _k=key:
                            self._on_poll_read_done(_t, _k, content)
                    )
                else:
                    # No worker available — skip rather than risk blocking
                    pass

    def _on_poll_read_done(self, tag: str, key: str, result):
        """Callback (Qt main thread) after a background poll read finishes."""
        if isinstance(result, Exception) or not result:
            return
        content = result if isinstance(result, str) else ""
        if not content:
            return

        # Deduplicate: only push if content changed
        if self._last_content.get(key) == content:
            return
        self._last_content[key] = content

        overlay = DebugNode._overlay_ref
        if overlay:
            overlay.push_message(tag, content)

    def _has_any_connections(self) -> bool:
        for port in self.left_ports:
            if port.connections:
                return True
        return False

    def _update_overlay_visibility(self):
        overlay = DebugNode._overlay_ref
        if overlay is None:
            return
        if self._has_any_connections():
            overlay.setVisible(True)
        else:
            overlay.setVisible(False)

    # ── Context menu ─────────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(_menu_stylesheet())

        header = menu.addAction(f"🐛  {self.node_name}")
        header.setEnabled(False)
        menu.addSeparator()

        clear_act = menu.addAction("🗑  Clear overlay")
        menu.addSeparator()
        delete_act = menu.addAction("✕   Delete")

        action = menu.exec(event.screenPos())
        if action == clear_act:
            overlay = DebugNode._overlay_ref
            if overlay:
                overlay.clear_messages()
        elif action == delete_act:
            self.scene_ref.delete_debug_node(self)

    def _find_panel(self) -> Optional['OperatorPanel']:
        app = QApplication.instance()
        if app is None:
            return None
        for widget in app.allWidgets():
            if isinstance(widget, OperatorPanel) and widget.scene is self.scene_ref:
                return widget
        return None

    def cleanup(self):
        self._poll_timer.stop()
        # Hide overlay when the debug node is removed
        overlay = DebugNode._overlay_ref
        if overlay:
            overlay.setVisible(False)
            overlay.clear_messages()


# ═══════════════════════════════════════════════════════════════════════════
# Graph Scene
# ═══════════════════════════════════════════════════════════════════════════

class OperatorGraphScene(QGraphicsScene):
    """
    Manages the node graph: agent nodes, ports, connections.
    Handles drag-to-connect between ports.
    """

    node_selected = Signal(object)
    connection_created = Signal(str, str, str, str)  # src_agent, src_port, dst_agent, dst_port
    agent_created = Signal(str)
    agent_deleted = Signal(str)

    def __init__(self):
        super().__init__()
        self.setSceneRect(-3000, -3000, 6000, 6000)

        self.agent_nodes: Dict[str, AgentNode] = {}
        self.ninep_nodes: Dict[str, NinePNode] = {}
        self.text_nodes: Dict[str, TextNode] = {}
        self.debug_nodes: Dict[str, DebugNode] = {}
        self.connections: List[Connection] = []

        # Connection drag state
        self._drag_source_port: Optional[Port] = None
        self._temp_conn: Optional[TempConnection] = None

        # Reference to the terminal widget for writing routes
        self._terminal_widget = None

        self.setBackgroundBrush(Qt.NoBrush)

    def set_terminal_widget(self, tw):
        """Set a reference to the active terminal widget for route creation."""
        self._terminal_widget = tw

    # ── Drawing ──────────────────────────────────────────────────────────

    def drawBackground(self, painter, rect):
        # Do NOT call super — skip the background fill entirely
        painter.setRenderHint(QPainter.Antialiasing, False)

        gs = Theme.GRID_SIZE
        major = Theme.GRID_MAJOR_EVERY

        left = int(rect.left()) - (int(rect.left()) % gs)
        top = int(rect.top()) - (int(rect.top()) % gs)

        # Draw faint dots at intersections
        minor_pen = QPen(Theme.GRID_DOT, 1.0)
        major_pen = QPen(Theme.GRID_LINE_MAJOR, 1.2)

        x = left
        while x <= rect.right():
            y = top
            while y <= rect.bottom():
                is_major = (int(x / gs) % major == 0 and
                            int(y / gs) % major == 0)
                painter.setPen(major_pen if is_major else minor_pen)
                painter.drawPoint(QPointF(x, y))
                y += gs
            x += gs

    # ── Node Management ──────────────────────────────────────────────────

    def add_agent_node(self, agent_name: str, agent_path: str,
                       pos: QPointF = None,
                       header_color: QColor = None,
                       known_files: List[str] = None) -> AgentNode:
        """Add an agent node to the graph."""
        if agent_name in self.agent_nodes:
            return self.agent_nodes[agent_name]

        node = AgentNode(agent_name, agent_path, self, header_color,
                         known_files=known_files)
        if pos:
            node.setPos(pos)
        else:
            # Place in a grid-like pattern
            idx = len(self.agent_nodes)
            cols = 3
            col = idx % cols
            row = idx // cols
            x = col * 320 - (cols * 320) / 2
            y = row * 280 - 200
            node.setPos(x, y)

        self.addItem(node)
        self.agent_nodes[agent_name] = node
        return node

    def add_ninep_node(self, service_name: str, mount_path: str,
                       left_ports: List[PortDef],
                       right_ports: List[PortDef],
                       pos: QPointF = None,
                       description: str = "") -> NinePNode:
        """Add a 9P service node to the graph."""
        if service_name in self.ninep_nodes:
            return self.ninep_nodes[service_name]

        node = NinePNode(service_name, mount_path, left_ports, right_ports,
                         self, description)
        if pos:
            node.setPos(pos)

        self.addItem(node)
        self.ninep_nodes[service_name] = node
        return node

    def delete_agent_node(self, node: AgentNode):
        """Remove an agent node and all its connections."""
        # Confirm
        reply = QMessageBox.question(
            None, "Delete Agent",
            f"Delete agent '{node.agent_name}' from the filesystem?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # Remove connections
        for port in node.left_ports + node.right_ports:
            for conn in port.connections[:]:
                self.remove_connection(conn)

        # Cleanup timers
        node.cleanup()

        # Remove from scene
        self.removeItem(node)
        del self.agent_nodes[node.agent_name]

        # Delete from filesystem
        ctl_path = os.path.join(os.path.dirname(node.agent_path), ".ctl")
        try:
            with open(ctl_path, 'w') as f:
                f.write(f"delete {node.agent_name}")
        except Exception:
            pass  # May not have ctl at agents level

        self.agent_deleted.emit(node.agent_name)

    # ── Text Node Management ─────────────────────────────────────────────

    def add_text_node(self, name: str, pos: QPointF = None,
                      input_path: str = "", output_path: str = "") -> 'TextNode':
        """Add a text node to the graph."""
        if name in self.text_nodes:
            return self.text_nodes[name]

        node = TextNode(name, self, input_path, output_path)
        if pos:
            node.setPos(pos)
        self.addItem(node)
        self.text_nodes[name] = node
        return node

    def delete_text_node(self, node: 'TextNode'):
        """Remove a text node and its connections."""
        reply = QMessageBox.question(
            None, "Delete Text Node",
            f"Delete text node '{node.node_name}'?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        for port in node.left_ports + node.right_ports:
            for conn in port.connections[:]:
                self.remove_connection(conn)

        node.cleanup()
        self.removeItem(node)
        if node.node_name in self.text_nodes:
            del self.text_nodes[node.node_name]

    # ── Debug Node Management ────────────────────────────────────────────

    def add_debug_node(self, name: str, pos: QPointF = None,
                       initial_inputs: int = 2) -> 'DebugNode':
        """Add a debug node to the graph."""
        if name in self.debug_nodes:
            return self.debug_nodes[name]

        node = DebugNode(name, self, initial_inputs)
        if pos:
            node.setPos(pos)
        self.addItem(node)
        self.debug_nodes[name] = node
        return node

    def delete_debug_node(self, node: 'DebugNode'):
        """Remove a debug node and its connections."""
        reply = QMessageBox.question(
            None, "Delete Debug Node",
            f"Delete debug node '{node.node_name}'?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        for port in node.left_ports + node.right_ports:
            for conn in port.connections[:]:
                self.remove_connection(conn)

        node.cleanup()
        self.removeItem(node)
        if node.node_name in self.debug_nodes:
            del self.debug_nodes[node.node_name]

    # ── Connection via Port Drag ─────────────────────────────────────────

    def start_port_connection(self, port: Port):
        """Begin dragging a connection from a port."""
        self._drag_source_port = port
        self._temp_conn = TempConnection(port)
        self.addItem(self._temp_conn)

    def finish_port_connection(self, scene_pos: QPointF):
        """Complete or cancel the connection drag."""
        if not self._drag_source_port or not self._temp_conn:
            return

        # Find port under cursor
        target_port = self._find_port_at(scene_pos)

        if (target_port and target_port != self._drag_source_port
                and target_port.parent_node != self._drag_source_port.parent_node):
            # Ensure output→input direction
            src = self._drag_source_port
            dst = target_port
            if src.port_def.direction == PortDirection.LEFT:
                src, dst = dst, src

            self._create_connection(src, dst)

        # Cleanup temp
        self.removeItem(self._temp_conn)
        self._temp_conn = None
        self._drag_source_port = None

    def _find_port_at(self, scene_pos: QPointF) -> Optional[Port]:
        """Find a Port item near a scene position."""
        items = self.items(scene_pos, Qt.IntersectsItemShape,
                          Qt.DescendingOrder)
        for item in items:
            if isinstance(item, Port):
                return item
        # Broader search across all nodes
        all_nodes = (list(self.agent_nodes.values()) +
                     list(self.ninep_nodes.values()) +
                     list(self.text_nodes.values()) +
                     list(self.debug_nodes.values()))
        for node in all_nodes:
            for port in node.left_ports + node.right_ports:
                pc = port.scene_center()
                dist = (pc - scene_pos).manhattanLength()
                if dist < Theme.PORT_HIT_RADIUS * 2:
                    return port
        return None

    def _create_connection(self, source: Port, target: Port):
        """Create a connection and the corresponding filesystem route."""
        # Check for duplicates
        for conn in self.connections:
            if (conn.source_port == source and conn.target_port == target):
                return

        # Build route command
        src_path = source.port_def.file_path
        dst_path = target.port_def.file_path
        route_cmd = f"{src_path} -> {dst_path}"

        # Write route to /n/rioa/routes via the filesystem
        self._write_route_to_filesystem(src_path, dst_path)

        conn = Connection(source, target, route_cmd)
        self.addItem(conn)
        self.connections.append(conn)

        # Emit signal for filesystem route creation
        self.connection_created.emit(
            source.parent_node.node_name, source.port_def.name,
            target.parent_node.node_name, target.port_def.name)

    def _write_route_to_filesystem(self, src_path: str, dst_path: str):
        """
        Create a route by writing to /n/rioa/routes via a background
        subprocess — the exact equivalent of:
            echo 'src -> dst' > /n/rioa/routes
        Runs off the Qt thread so FUSE I/O cannot freeze the UI.
        """
        routes_path = self._find_routes_file(src_path, dst_path)
        if routes_path is None:
            return
        route_line = f"{src_path} -> {dst_path}"
        import subprocess
        threading.Thread(
            target=lambda: subprocess.run(
                ["sh", "-c", f"echo '{route_line}' > {routes_path}"],
                timeout=5,
            ),
            daemon=True,
        ).start()

    def _remove_route_from_filesystem(self, src_path: str, dst_path: str = ""):
        """
        Remove a route:  echo '-/path/to/source' > /n/rioa/routes
        """
        routes_path = self._find_routes_file(src_path, dst_path)
        if routes_path is None:
            return
        import subprocess
        threading.Thread(
            target=lambda: subprocess.run(
                ["sh", "-c", f"echo '-{src_path}' > {routes_path}"],
                timeout=5,
            ),
            daemon=True,
        ).start()

    def _find_routes_file(self, src_path: str, dst_path: str = "") -> Optional[str]:
        """
        Return the path to the /n/<svc>/routes file that should own this route.
        Checks which NinePNode mount is a prefix of source or dest.
        Falls back to the first available service.
        """
        for node in self.ninep_nodes.values():
            mp = node.mount_path
            if src_path.startswith(mp + "/") or dst_path.startswith(mp + "/"):
                return os.path.join(mp, "routes")
        # Fallback: first service
        for node in self.ninep_nodes.values():
            return os.path.join(node.mount_path, "routes")
        return None

    def remove_connection(self, conn: Connection):
        """Remove a connection and its filesystem route."""
        if conn.source_port:
            if conn in conn.source_port.connections:
                conn.source_port.connections.remove(conn)
        if conn.target_port:
            if conn in conn.target_port.connections:
                conn.target_port.connections.remove(conn)

        if conn in self.connections:
            self.connections.remove(conn)
        self.removeItem(conn)

        # Remove the route via the filesystem routes file
        if conn.source_port:
            src_path = conn.source_port.port_def.file_path
            dst_path = conn.target_port.port_def.file_path if conn.target_port else ""
            self._remove_route_from_filesystem(src_path, dst_path)

    # ── Import existing attachments as connections ────────────────────────

    def import_existing_attachments(self, attachments: dict):
        """
        Import existing Plan9Attachments from the terminal widget
        and create visual Connection objects for them.
        """
        for source_path, attachment in attachments.items():
            dest_path = attachment.destination
            self._import_route(source_path, dest_path)

    def import_routes(self, routes: List[Tuple[str, str]]):
        """
        Import routes as (source_path, dest_path) pairs and create
        visual Connection objects.  Called after all nodes and ports
        have been created so that _find_port_by_path can match.
        """
        for src_path, dst_path in routes:
            self._import_route(src_path, dst_path)

    def _import_route(self, src_path: str, dst_path: str):
        """Create a visual connection for a single source→dest route."""
        src_port = self._find_port_by_path(src_path)
        dst_port = self._find_port_by_path(dst_path)

        if src_port and dst_port:
            already = any(c.source_port == src_port and c.target_port == dst_port
                          for c in self.connections)
            if not already:
                conn = Connection(src_port, dst_port,
                                  f"{src_path} -> {dst_path}",
                                  is_existing=True)
                self.addItem(conn)
                self.connections.append(conn)

    def _find_port_by_path(self, file_path: str) -> Optional[Port]:
        """Find a port whose file_path matches the given path."""
        all_nodes = (list(self.agent_nodes.values()) +
                     list(self.ninep_nodes.values()) +
                     list(self.text_nodes.values()) +
                     list(self.debug_nodes.values()))
        for node in all_nodes:
            for port in node.left_ports + node.right_ports:
                if port.port_def.file_path == file_path:
                    return port
        return None

    # ── Mouse Events ─────────────────────────────────────────────────────

    def mouseMoveEvent(self, event):
        if self._temp_conn:
            self._temp_conn.set_end(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._temp_conn and event.button() == Qt.LeftButton:
            self.finish_port_connection(event.scenePos())
        super().mouseReleaseEvent(event)

    # ── Keyboard Events ──────────────────────────────────────────────────

    def keyPressEvent(self, event):
        """Delete selected connections (and nodes) on Delete / Backspace."""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected = self.selectedItems()
            if not selected:
                super().keyPressEvent(event)
                return

            # Gather connections to remove (directly selected, or owned by
            # selected nodes)
            conns_to_remove: List[Connection] = []
            nodes_to_remove: List[BaseNode] = []

            for item in selected:
                if isinstance(item, Connection):
                    conns_to_remove.append(item)
                elif isinstance(item, (AgentNode, TextNode, DebugNode)):
                    nodes_to_remove.append(item)

            # Remove connections first
            for conn in conns_to_remove:
                self.remove_connection(conn)

            # Remove nodes (each has its own confirmation dialog)
            for node in nodes_to_remove:
                if isinstance(node, AgentNode):
                    self.delete_agent_node(node)
                elif isinstance(node, TextNode):
                    self.delete_text_node(node)
                elif isinstance(node, DebugNode):
                    self.delete_debug_node(node)

            event.accept()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        # Check if we're on a node or connection
        item = self.itemAt(event.scenePos(), QTransform())
        if isinstance(item, (AgentNode, NinePNode, TextNode, DebugNode, Port, Connection)):
            super().contextMenuEvent(event)
            return

        # Canvas context menu
        menu = QMenu()
        menu.setStyleSheet(_menu_stylesheet())

        create_action = menu.addAction("＋  Create Agent...")
        create_text_action = menu.addAction("📝  Add Text Node...")
        create_debug_action = menu.addAction("🐛  Add Debug Node...")
        menu.addSeparator()
        layout_action = menu.addAction("⊞   Auto Layout")
        refresh_action = menu.addAction("↻   Refresh")
        menu.addSeparator()
        fit_action = menu.addAction("◻   Fit All")

        action = menu.exec(event.screenPos())
        if action == create_action:
            self._create_agent_dialog(event.scenePos())
        elif action == create_text_action:
            self._create_text_node_dialog(event.scenePos())
        elif action == create_debug_action:
            self._create_debug_node_dialog(event.scenePos())
        elif action == layout_action:
            self._auto_layout()
        elif action == refresh_action:
            self._refresh_all()
        elif action == fit_action:
            self._fit_all()

    def _create_agent_dialog(self, pos: QPointF):
        """Show dialog to create a new agent."""
        dlg = CreateAgentDialog()
        if dlg.exec() == QDialog.Accepted:
            name = dlg.agent_name
            system_file = dlg.system_file
            model = dlg.model_name

            # Create in filesystem
            self._create_agent_fs(name, system_file, model)

            # Choose color
            colors = [Theme.NODE_HEADER_AGENT, Theme.NODE_HEADER_AGENT_ALT,
                      Theme.NODE_HEADER_SPECIAL, Theme.NODE_HEADER_FILE]
            color = colors[len(self.agent_nodes) % len(colors)]

            agents_dir = self._get_agents_dir()
            agent_path = os.path.join(agents_dir, name)
            self.add_agent_node(name, agent_path, pos, color)
            self.agent_created.emit(name)

    def _create_text_node_dialog(self, pos: QPointF):
        """Show a dialog to create a new text node."""
        dlg = QDialog()
        dlg.setWindowTitle("Add Text Node")
        dlg.setMinimumWidth(380)
        dlg.setStyleSheet(_dialog_stylesheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(12)
        lay.setContentsMargins(24, 24, 24, 24)

        title_lbl = QLabel("Add Text Node")
        title_lbl.setFont(QFont(Theme.FONT_FAMILY, 14, QFont.DemiBold))
        title_lbl.setStyleSheet(f"color: {Theme.TEXT_PRIMARY.name()};")
        lay.addWidget(title_lbl)

        form = QFormLayout()
        form.setSpacing(10)

        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g. notepad, trigger, buffer")
        name_input.setText(f"text_{len(self.text_nodes) + 1}")
        form.addRow("Name:", name_input)

        in_input = QLineEdit()
        in_input.setPlaceholderText("Optional: /n/mux/llm/agents/foo/output")
        form.addRow("Input path:", in_input)

        out_input = QLineEdit()
        out_input.setPlaceholderText("Optional: /n/mux/llm/agents/bar/input")
        form.addRow("Output path:", out_input)

        lay.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.Accepted:
            name = name_input.text().strip() or f"text_{len(self.text_nodes) + 1}"
            self.add_text_node(name, pos,
                               in_input.text().strip(),
                               out_input.text().strip())

    def _create_debug_node_dialog(self, pos: QPointF):
        """Show a dialog to create a new debug node."""
        dlg = QDialog()
        dlg.setWindowTitle("Add Debug Node")
        dlg.setMinimumWidth(340)
        dlg.setStyleSheet(_dialog_stylesheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(12)
        lay.setContentsMargins(24, 24, 24, 24)

        title_lbl = QLabel("Add Debug Node")
        title_lbl.setFont(QFont(Theme.FONT_FAMILY, 14, QFont.DemiBold))
        title_lbl.setStyleSheet(f"color: {Theme.TEXT_PRIMARY.name()};")
        lay.addWidget(title_lbl)

        form = QFormLayout()
        form.setSpacing(10)

        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g. debug, monitor")
        name_input.setText(f"debug_{len(self.debug_nodes) + 1}")
        form.addRow("Name:", name_input)

        inputs_spin = QSpinBox()
        inputs_spin.setRange(1, 16)
        inputs_spin.setValue(2)
        form.addRow("Inputs:", inputs_spin)

        lay.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.Accepted:
            name = name_input.text().strip() or f"debug_{len(self.debug_nodes) + 1}"
            self.add_debug_node(name, pos, inputs_spin.value())

    def _create_agent_fs(self, name: str, system_file: str = "",
                         model: str = ""):
        """Create agent in the LLMFS filesystem."""
        agents_dir = self._get_agents_dir()
        ctl_path = os.path.join(agents_dir, ".ctl")

        # Create agent via ctl
        try:
            with open(ctl_path, 'w') as f:
                f.write(f"create {name}")
        except Exception as e:
            # Fallback: try mkdir
            try:
                os.makedirs(os.path.join(agents_dir, name), exist_ok=True)
            except Exception:
                QMessageBox.warning(None, "Error",
                                    f"Could not create agent: {e}")
                return

        agent_path = os.path.join(agents_dir, name)

        # Set system prompt from file
        if system_file and os.path.isfile(system_file):
            try:
                with open(system_file, 'r') as src:
                    content = src.read()
                system_path = os.path.join(agent_path, "system")
                with open(system_path, 'w') as dst:
                    dst.write(content)
            except Exception:
                pass

        # Set model
        if model:
            ctl = os.path.join(agent_path, "ctl")
            try:
                with open(ctl, 'w') as f:
                    f.write(f"model {model}")
            except Exception:
                pass

    def _get_agents_dir(self) -> str:
        """Get the agents directory path from the parent panel or terminal widget."""
        for view in self.views():
            parent = view.parent()
            while parent:
                if isinstance(parent, OperatorPanel):
                    return os.path.join(parent.llmfs_mount, "agents")
                parent = parent.parent() if hasattr(parent, 'parent') else None
        # Fallback: try terminal widget
        if self._terminal_widget and hasattr(self._terminal_widget, 'llmfs_mount'):
            return os.path.join(self._terminal_widget.llmfs_mount, "agents")
        return "/n/mux/llm/agents"

    def _auto_layout(self):
        """Arrange nodes: 9P services on left, agents on right, text nodes below, debug nodes further below."""
        ninep_list = list(self.ninep_nodes.values())
        agent_list = list(self.agent_nodes.values())
        text_list = list(self.text_nodes.values())
        debug_list = list(self.debug_nodes.values())

        spacing_x = 380
        spacing_y = 300

        # 9P nodes in a column on the left
        for i, node in enumerate(ninep_list):
            x = -spacing_x
            y = i * spacing_y - (len(ninep_list) * spacing_y) / 2
            node.setPos(x, y)

        # Agent nodes in a grid on the right
        if agent_list:
            cols = max(1, int(math.ceil(math.sqrt(len(agent_list)))))
            for i, node in enumerate(agent_list):
                col = i % cols
                row = i // cols
                x = spacing_x / 2 + col * (Theme.NODE_WIDTH + 120)
                y = row * spacing_y - (max(1, len(agent_list) // cols) * spacing_y) / 2
                node.setPos(x, y)

        # Text nodes below the agent grid
        if text_list:
            agent_bottom = 0
            if agent_list:
                agent_bottom = max(n.pos().y() + n.rect().height()
                                   for n in agent_list) + 80
            for i, node in enumerate(text_list):
                x = i * (TextNode.TEXT_NODE_WIDTH + 60) - (len(text_list) * (TextNode.TEXT_NODE_WIDTH + 60)) / 2
                node.setPos(x, agent_bottom)

        # Debug nodes below text nodes
        if debug_list:
            bottom_y = 0
            if text_list:
                bottom_y = max(n.pos().y() + n.rect().height()
                               for n in text_list) + 80
            elif agent_list:
                bottom_y = max(n.pos().y() + n.rect().height()
                               for n in agent_list) + 80
            for i, node in enumerate(debug_list):
                x = i * (DebugNode.DEBUG_NODE_WIDTH + 60) - (len(debug_list) * (DebugNode.DEBUG_NODE_WIDTH + 60)) / 2
                node.setPos(x, bottom_y)

    def _refresh_all(self):
        """Refresh all nodes."""
        for node in self.agent_nodes.values():
            node._refresh_state()
            node._refresh_ports()

    def _fit_all(self):
        """Fit all nodes in view."""
        if self.agent_nodes or self.ninep_nodes or self.text_nodes or self.debug_nodes:
            all_items_rect = self.itemsBoundingRect()
            for view in self.views():
                view.fitInView(all_items_rect.adjusted(-50, -50, 50, 50),
                               Qt.KeepAspectRatio)

    def clear_graph(self):
        """Clear all nodes and connections from the scene only."""
        for conn in self.connections[:]:
            self.removeItem(conn)
        for node in list(self.agent_nodes.values()):
            node.cleanup()
            self.removeItem(node)
        for node in list(self.ninep_nodes.values()):
            node.cleanup()
            self.removeItem(node)
        for node in list(self.text_nodes.values()):
            node.cleanup()
            self.removeItem(node)
        for node in list(self.debug_nodes.values()):
            node.cleanup()
            self.removeItem(node)
        self.agent_nodes.clear()
        self.ninep_nodes.clear()
        self.text_nodes.clear()
        self.debug_nodes.clear()
        self.connections.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Create Agent Dialog
# ═══════════════════════════════════════════════════════════════════════════

class CreateAgentDialog(QDialog):
    """Dialog for creating a new agent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Agent")
        self.setMinimumWidth(420)
        self.setStyleSheet(_dialog_stylesheet())

        self.agent_name = ""
        self.system_file = ""
        self.model_name = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Create New Agent")
        title.setFont(QFont(Theme.FONT_FAMILY, 14, QFont.DemiBold))
        title.setStyleSheet(f"color: {Theme.TEXT_PRIMARY.name()};")
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g. claude, coder, reviewer")
        form.addRow("Name:", self._name_input)

        # System file selector
        file_row = QHBoxLayout()
        self._system_input = QLineEdit()
        self._system_input.setPlaceholderText("Optional: path to .md file")
        self._system_input.setReadOnly(True)
        file_row.addWidget(self._system_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_system)
        file_row.addWidget(browse_btn)

        form.addRow("System:", file_row)

        # Model
        self._model_input = QComboBox()
        self._model_input.setEditable(True)
        self._model_input.addItems([
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
        ])
        form.addRow("Model:", self._model_input)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_system(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select System Prompt",
            os.path.expanduser("~"),
            "Markdown (*.md);;Text (*.txt);;All (*)")
        if path:
            self._system_input.setText(path)

    def _accept(self):
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Agent name is required.")
            return
        if not name.isidentifier():
            QMessageBox.warning(self, "Error",
                                "Agent name must be a valid identifier "
                                "(letters, digits, underscores).")
            return

        self.agent_name = name
        self.system_file = self._system_input.text().strip()
        self.model_name = self._model_input.currentText().strip()
        self.accept()


# ═══════════════════════════════════════════════════════════════════════════
# Operator Panel - Main Widget
# ═══════════════════════════════════════════════════════════════════════════

class OperatorPanel(QWidget):
    """
    Main operator panel widget.
    Provides a toolbar, graph view, and status bar for full LLMFS control.
    
    All filesystem I/O on /n/ is performed on a background thread via
    FSWorker so that blocking 9P reads never freeze the Qt event loop.
    """

    def __init__(self, llmfs_mount="/n/mux/llm", rio_mount="/n/mux/default",
                 terminal_widget=None, parent=None):
        super().__init__(parent)
        self.llmfs_mount = llmfs_mount
        self.rio_mount = rio_mount
        self._terminal_widget = terminal_widget

        # Determine the root to scan for 9P services.
        # With riomux:  llmfs_mount = /n/mux/llm  → scan_root = /n/mux
        # Standalone:    llmfs_mount = /n/llm      → scan_root = /n
        self.scan_root = os.path.dirname(llmfs_mount)

        # Background filesystem worker – all 9P I/O goes through this
        self._fs_worker = FSWorker(self)

        # Sync theme mode with the terminal widget's current dark mode state
        # (the panel may be created after the app has already toggled dark mode)
        if terminal_widget and getattr(terminal_widget, '_is_dark_mode', False):
            Theme.set_mode(True)
        else:
            Theme.set_mode(False)

        self._init_ui()
        self._setup_signals()

        # Initial scan
        QTimer.singleShot(200, self.scan_architecture)

    def set_terminal_widget(self, tw):
        """Set or update the terminal widget reference."""
        self._terminal_widget = tw
        self.scene.set_terminal_widget(tw)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ──────────────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setFixedHeight(46)
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.TOOLBAR_BG_RGBA};
                border-bottom: 1px solid {Theme.TOOLBAR_BORDER.name()};
            }}
        """)

        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        tb_layout.setSpacing(8)

        # Logo / title
        title = QLabel("operator")
        title.setFont(QFont(Theme.FONT_FAMILY, 12, QFont.Normal))
        title.setStyleSheet(f"background: transparent; color: {Theme.TEXT_PRIMARY.name()}; letter-spacing: {Theme.TITLE_LETTER_SPACING};")
        tb_layout.addWidget(title)

        sep = QLabel("·")
        sep.setStyleSheet(f"background: transparent; color: {Theme.SEP_COLOR_RGBA}; "
                          f"font-size: 14px; margin: 0 6px;")
        tb_layout.addWidget(sep)

        subtitle = QLabel("9p")
        subtitle.setFont(QFont(Theme.FONT_FAMILY, 10))
        subtitle.setStyleSheet(f"background: transparent; color: {Theme.TEXT_SECONDARY.name()}; letter-spacing: 1px;")
        tb_layout.addWidget(subtitle)

        tb_layout.addStretch()

        # Toolbar buttons
        self._add_toolbar_btn(tb_layout, "＋ Agent",
                              self._create_agent, primary=True)
        self._add_toolbar_btn(tb_layout, "📝 Text",
                              self._create_text_node)
        self._add_toolbar_btn(tb_layout, "🐛 Debug",
                              self._create_debug_node)
        self._add_toolbar_btn(tb_layout, "↻ Scan",
                              self.scan_architecture)
        self._add_toolbar_btn(tb_layout, "⊞ Layout",
                              self._auto_layout)
        self._add_toolbar_btn(tb_layout, "◻ Fit",
                              self._fit_all)

        # Zoom buttons
        self._add_toolbar_btn(tb_layout, "−", self._zoom_out)
        self._add_toolbar_btn(tb_layout, "+", self._zoom_in)

        layout.addWidget(toolbar)

        # ── Graph View ───────────────────────────────────────────────────
        self.scene = OperatorGraphScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setStyleSheet("QGraphicsView { border: none; background: transparent; }")
        self.view.viewport().setAutoFillBackground(False)
        self.view.setAttribute(Qt.WA_TranslucentBackground)
        self.view.setAutoFillBackground(False)

        layout.addWidget(self.view)

        # ── Floating Node Visibility Overlay ─────────────────────────────
        # Parented to the OperatorPanel itself so it stays fixed on screen
        # regardless of scene pan/zoom. Repositioned in resizeEvent.
        self._node_overlay = QWidget(self)
        self._node_overlay.setStyleSheet("background: transparent;")
        self._node_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._node_overlay.raise_()

        overlay_layout = QVBoxLayout(self._node_overlay)
        overlay_layout.setContentsMargins(4, 4, 4, 4)
        overlay_layout.setSpacing(0)

        self._checkbox_container = QWidget()
        self._checkbox_container.setStyleSheet("background: transparent;")
        self._checkbox_layout = QVBoxLayout(self._checkbox_container)
        self._checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox_layout.setSpacing(0)

        overlay_layout.addWidget(self._checkbox_container)
        self._node_overlay.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._node_overlay.show()

        # Tracks: node_key -> QCheckBox
        self._node_checkboxes: Dict[str, QCheckBox] = {}
        # Tracks which nodes the user explicitly hid (survives rescan)
        self._hidden_nodes: set = set()

        # ── Status Bar ───────────────────────────────────────────────────
        status = QFrame()
        status.setFixedHeight(28)
        status.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.STATUS_BG_RGBA};
                border-top: 1px solid {Theme.STATUS_BORDER.name()};
            }}
        """)

        st_layout = QHBoxLayout(status)
        st_layout.setContentsMargins(16, 0, 16, 0)

        self._status_label = QLabel("Ready")
        self._status_label.setFont(Theme.FONT_STATUS)
        self._status_label.setStyleSheet(
            f"background: transparent; color: {Theme.TEXT_SECONDARY.name()};")
        st_layout.addWidget(self._status_label)

        st_layout.addStretch()

        self._mount_label = QLabel(f"⛁ {self.scan_root}")
        self._mount_label.setFont(QFont(Theme.FONT_FAMILY_MONO, 9))
        self._mount_label.setStyleSheet(
            f"background: transparent; color: {Theme.TEXT_SECONDARY.name()};")
        st_layout.addWidget(self._mount_label)

        self._count_label = QLabel("0 agents · 0 routes")
        self._count_label.setFont(Theme.FONT_STATUS)
        self._count_label.setStyleSheet(
            f"background: transparent; color: {Theme.TEXT_SECONDARY.name()}; margin-left: 20px;")
        st_layout.addWidget(self._count_label)

        layout.addWidget(status)

        # Overall widget style — transparent background
        self.setObjectName("OperatorPanel")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet(f"""
            #OperatorPanel {{
                background: transparent;
                font-family: '{Theme.FONT_FAMILY}';
            }}
        """)

    def _add_toolbar_btn(self, layout, text, callback, primary=False):
        btn = QPushButton(text)
        btn.setFont(Theme.FONT_TOOLBAR)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(28)

        if primary:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Theme.BTN_PRIMARY_BG};
                    color: {Theme.TEXT_PRIMARY.name()};
                    border: 1px solid {Theme.BTN_PRIMARY_BORDER};
                    border-radius: 5px;
                    padding: 0 14px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {Theme.BTN_PRIMARY_HOVER_BG};
                    border-color: {Theme.BTN_PRIMARY_HOVER_BORDER};
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Theme.BTN_DEFAULT_BG_RGBA};
                    color: {Theme.TEXT_SECONDARY.name()};
                    border: 1px solid {Theme.BTN_DEFAULT_BORDER_RGBA};
                    border-radius: 5px;
                    padding: 0 12px;
                }}
                QPushButton:hover {{
                    background-color: {Theme.BTN_DEFAULT_HOVER_BG_RGBA};
                    color: {Theme.TEXT_PRIMARY.name()};
                    border-color: {Theme.BTN_DEFAULT_HOVER_BORDER_RGBA};
                }}
            """)

        btn.clicked.connect(callback)
        layout.addWidget(btn)

    def _setup_signals(self):
        self.scene.agent_created.connect(self._on_agent_created)
        self.scene.agent_deleted.connect(self._on_agent_deleted)
        self.scene.connection_created.connect(self._on_connection_created)

    # ── Toolbar Actions ──────────────────────────────────────────────────

    def _create_agent(self):
        """Show create agent dialog and add at center of view."""
        center = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene._create_agent_dialog(center)

    def _create_text_node(self):
        """Show create text node dialog and add at center of view."""
        center = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene._create_text_node_dialog(center)

    def _create_debug_node(self):
        """Show create debug node dialog and add at center of view."""
        center = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene._create_debug_node_dialog(center)

    def _auto_layout(self):
        self.scene._auto_layout()
        self._update_counts()

    def _fit_all(self):
        self.scene._fit_all()

    def _zoom_in(self):
        """Zoom the canvas in."""
        self.view.scale(1.25, 1.25)

    def _zoom_out(self):
        """Zoom the canvas out."""
        self.view.scale(1 / 1.25, 1 / 1.25)

    # ── Node Visibility Sidebar ──────────────────────────────────────────

    def _rebuild_sidebar_checkboxes(self):
        """Rebuild the floating overlay checkboxes to match current scene nodes.

        Rules:
          - New nodes (not previously seen) start checked (visible).
          - Nodes that the user previously unchecked stay unchecked
            (tracked in self._hidden_nodes).
          - Stale checkboxes for nodes no longer in the scene are removed.
        """
        # Collect all current node keys
        current_keys: Dict[str, object] = {}
        for name, node in self.scene.agent_nodes.items():
            current_keys[f"agent:{name}"] = node
        for name, node in self.scene.ninep_nodes.items():
            current_keys[f"9p:{name}"] = node
        for name, node in self.scene.text_nodes.items():
            current_keys[f"text:{name}"] = node
        for name, node in self.scene.debug_nodes.items():
            current_keys[f"debug:{name}"] = node

        # Remove stale checkboxes (block signals so deletion doesn't
        # fire toggled() and corrupt _hidden_nodes)
        stale = [k for k in self._node_checkboxes if k not in current_keys]
        for k in stale:
            cb = self._node_checkboxes.pop(k)
            cb.blockSignals(True)
            self._checkbox_layout.removeWidget(cb)
            cb.deleteLater()

        # Add new checkboxes / update existing
        for key, node in current_keys.items():
            if key in self._node_checkboxes:
                # Already exists — just re-apply visibility to the (new) node
                if key in self._hidden_nodes:
                    self._set_node_visible(key, False)
                continue

            initially_visible = key not in self._hidden_nodes
            display = key.split(":", 1)[1]

            cb = QCheckBox(display)
            cb.blockSignals(True)  # don't fire toggled during init
            cb.setChecked(initially_visible)
            cb.blockSignals(False)
            cb.setFont(QFont(Theme.FONT_FAMILY, 8))
            cb.setStyleSheet(self._checkbox_stylesheet())
            cb.toggled.connect(lambda checked, k=key: self._on_node_checkbox_toggled(k, checked))
            self._checkbox_layout.addWidget(cb)
            self._node_checkboxes[key] = cb

            if not initially_visible:
                self._set_node_visible(key, False)

        # Let the overlay grow/shrink to fit all checkboxes
        self._checkbox_container.adjustSize()
        self._node_overlay.adjustSize()

    def _checkbox_stylesheet(self) -> str:
        return f"""
            QCheckBox {{
                background: transparent;
                color: {Theme.TEXT_PRIMARY.name()};
                font-size: 11px;
                spacing: 5px;
                padding: 2px 0;
            }}
            QCheckBox::indicator {{
                width: 12px; height: 12px;
                border: 1px solid {Theme.CHECKBOX_BORDER};
                border-radius: 2px;
                background: {Theme.CHECKBOX_BG};
            }}
            QCheckBox::indicator:checked {{
                background: {Theme.CHECKBOX_CHECKED_BG};
                border-color: {Theme.CHECKBOX_CHECKED_BORDER};
            }}
        """

    def _on_node_checkbox_toggled(self, key: str, checked: bool):
        """Handle a node visibility checkbox toggle."""
        if checked:
            self._hidden_nodes.discard(key)
        else:
            self._hidden_nodes.add(key)
        self._set_node_visible(key, checked)

    def _set_node_visible(self, key: str, visible: bool):
        """Show or hide a node and all its connections in the scene."""
        node = self._resolve_node(key)
        if node is None:
            return
        node.setVisible(visible)
        # Hide/show connections attached to this node
        for port in node.left_ports + node.right_ports:
            for conn in port.connections:
                if visible:
                    # Only show connection if BOTH endpoints are visible
                    other = conn.target_port if conn.source_port in (node.left_ports + node.right_ports) else conn.source_port
                    other_node = other.parentItem()
                    conn.setVisible(other_node.isVisible())
                else:
                    conn.setVisible(False)

    def _resolve_node(self, key: str):
        """Resolve a sidebar key like 'agent:master' to a scene node."""
        prefix, name = key.split(":", 1)
        if prefix == "agent":
            return self.scene.agent_nodes.get(name)
        elif prefix == "9p":
            return self.scene.ninep_nodes.get(name)
        elif prefix == "text":
            return self.scene.text_nodes.get(name)
        elif prefix == "debug":
            return self.scene.debug_nodes.get(name)
        return None

    # ── Scanning ─────────────────────────────────────────────────────────

    def scan_architecture(self):
        """
        Scan the LLMFS and 9P mounts on a background thread, then
        build the graph when results arrive.
        """
        self._status_label.setText("Scanning filesystem…")
        self.scene.clear_graph()

        # Pass terminal widget reference to scene
        if self._terminal_widget:
            self.scene.set_terminal_widget(self._terminal_widget)

        # Kick off two parallel background scans:
        #   1. Scan scan_root (e.g. /n or /n/mux) for 9P services
        #   2. Scan {llmfs_mount}/agents/ for agents
        self._pending_scans = 2
        self._scanned_services = []
        self._scanned_agents = []

        self._fs_worker.run_async(
            _fs_scan_9p_root, self.scan_root,
            callback=self._on_9p_scan_done
        )

        agents_dir = os.path.join(self.llmfs_mount, "agents")
        self._fs_worker.run_async(
            _fs_scan_agents_dir, agents_dir,
            callback=self._on_agents_scan_done
        )

    def _on_9p_scan_done(self, result):
        """Callback when /n/ scan completes."""
        if isinstance(result, Exception):
            self._scanned_services = []
        else:
            self._scanned_services = result or []
        self._pending_scans -= 1
        if self._pending_scans <= 0:
            self._build_graph()

    def _on_agents_scan_done(self, result):
        """Callback when agents/ scan completes."""
        if isinstance(result, Exception):
            self._scanned_agents = []
        else:
            self._scanned_agents = result or []
        self._pending_scans -= 1
        if self._pending_scans <= 0:
            self._build_graph()

    def _build_graph(self):
        """
        Build the node graph from filesystem scan results.
        Called on the Qt main thread after all background scans complete.
        """
        # ── 1. Create filesystem nodes from recursive scan ──
        self._build_9p_nodes()

        # ── 2. Create agent nodes ──
        agents_dir = os.path.join(self.llmfs_mount, "agents")
        colors = [Theme.NODE_HEADER_AGENT, Theme.NODE_HEADER_AGENT_ALT,
                  Theme.NODE_HEADER_SPECIAL, Theme.NODE_HEADER_FILE]

        for i, agent_info in enumerate(self._scanned_agents):
            name = agent_info['name']
            agent_path = os.path.join(agents_dir, name)
            color = colors[i % len(colors)]
            self.scene.add_agent_node(
                name, agent_path,
                header_color=color,
                known_files=agent_info.get('files', [])
            )

        # ── 3. Import routes ──
        self._import_all_routes()

        self.scene._auto_layout()
        self._update_counts()
        self._status_label.setText("Architecture loaded")

        # Rebuild the sidebar checkboxes to reflect new nodes
        self._rebuild_sidebar_checkboxes()

        QTimer.singleShot(100, self._fit_all)

    def _import_all_routes(self):
        """
        Import routes from:
          1. routes files found in the recursive scan (primary)
          2. TerminalWidget.attachments in-memory dicts (fallback)
        """
        # Collect all paths that have a 'routes' file
        mount_paths = []
        for svc_info in self._scanned_services:
            files = svc_info.get('files', [])
            if 'routes' in files:
                mount_paths.append(svc_info['path'])

        if mount_paths:
            self._fs_worker.run_async(
                self._read_all_routes, mount_paths,
                callback=self._on_routes_read
            )
        else:
            self._import_tw_attachments()

    @staticmethod
    def _read_all_routes(mount_paths: List[str]) -> List[Tuple[str, str]]:
        """Read routes from all service mount paths. Runs on bg thread."""
        all_routes = []
        for mount_path in mount_paths:
            all_routes.extend(_fs_read_routes(mount_path))
        return all_routes

    def _on_routes_read(self, result):
        """Callback when routes have been read from the filesystem."""
        if isinstance(result, Exception) or not result:
            # Fall back to in-memory attachments
            self._import_tw_attachments()
            return

        self.scene.import_routes(result)

        # Also import in-memory attachments as a supplement
        # (catches any routes that aren't yet persisted to /n/rioa/routes)
        self._import_tw_attachments()

        self._update_counts()

    def _import_tw_attachments(self):
        """Import routes from TerminalWidget in-memory attachments."""
        for tw in self._find_all_terminal_widgets():
            if hasattr(tw, 'attachments') and tw.attachments:
                self.scene.import_existing_attachments(tw.attachments)

    def _build_9p_nodes(self):
        """
        Create nodes from the recursive filesystem scan.
        
        Each directory becomes a node. Files in the directory become ports.
        Subdirectories are separate nodes (created by the recursive scan).
        
        Port side heuristic:
          - ctl, input, parse, system, stdin → left (writable/input)
          - everything else → right (readable/output)
        """
        # Build a lookup: path → scan info
        path_lookup = {s['path']: s for s in self._scanned_services}

        # The scan root itself doesn't need a node — its children do
        scan_root = self.scan_root

        for svc_info in self._scanned_services:
            path = svc_info['path']
            name = svc_info['name']
            files = svc_info.get('files', [])
            depth = svc_info.get('depth', 0)

            # Skip the scan root itself (e.g. /n/mux) — just a container
            if path == scan_root:
                continue

            # Skip agents dir and individual agent dirs — those are AgentNodes
            agents_dir = os.path.join(self.llmfs_mount, "agents")
            if path == agents_dir or path.startswith(agents_dir + "/"):
                continue

            # Skip the llmfs_mount itself — agents are shown individually
            if path == self.llmfs_mount:
                continue

            # Skip directories that contain only subdirectories (no files).
            # These are pure containers (e.g. "terms/") and don't need nodes.
            # Their children that DO have files will get their own nodes.
            subdirs = svc_info.get('subdirs', [])
            if not files and subdirs:
                continue

            # Build a display name: relative path from scan_root
            rel = os.path.relpath(path, scan_root)
            display_name = rel.replace(os.sep, '/')

            # Check if this is a known service type
            svc_def = _lookup_service(name, path)
            if svc_def is not None:
                self.scene.add_ninep_node(
                    display_name, path,
                    svc_def["left_ports"], svc_def["right_ports"],
                    description=svc_def.get("description", "")
                )
                continue

            # Generic: create ports from files
            left_ports = []
            right_ports = []

            LEFT_NAMES = {'ctl', 'input', 'parse', 'system', 'stdin', 'config', 'rules'}

            for fname in files:
                if fname in LEFT_NAMES:
                    left_ports.append(PortDef(
                        fname, PortDirection.LEFT, Theme.PORT_CTL,
                        description=fname))
                else:
                    right_ports.append(PortDef(
                        fname, PortDirection.RIGHT, Theme.PORT_DEFAULT,
                        description=fname))

            # Only create the node if it has files (ports)
            # or it's a top-level directory (depth 1) even without files
            if left_ports or right_ports or depth <= 1:
                self.scene.add_ninep_node(
                    display_name, path, left_ports, right_ports,
                    description=f"{path}")

    def _scan_9p_services(self):
        """Legacy shim — now handled by scan_architecture pipeline."""
        pass

    def _get_agent_names(self) -> list:
        """
        Get agent names from the filesystem.
        Falls back to terminal widget state if FS scan not available.
        """
        agents_dir = os.path.join(self.llmfs_mount, "agents")
        names = _fs_listdir(agents_dir)
        if names:
            return [n for n in names if _fs_isdir(os.path.join(agents_dir, n))]

        # Fallback: terminal widget in-memory state
        result = set()
        for tw in self._find_all_terminal_widgets():
            if hasattr(tw, 'known_agents') and tw.known_agents:
                result.update(tw.known_agents)
            if hasattr(tw, 'connected_agent') and tw.connected_agent:
                result.add(tw.connected_agent)
        return sorted(result)

    def _find_all_terminal_widgets(self):
        """
        Find ALL TerminalWidget instances in the running QApplication.
        Used as a fallback for attachment import and agent discovery.
        """
        widgets = []
        app = QApplication.instance()
        if app is None:
            if self._terminal_widget is not None:
                return [self._terminal_widget]
            return []

        # Import locally to avoid circular imports at module level
        try:
            from .terminal_widget import TerminalWidget as TW_class
        except (ImportError, SystemError):
            try:
                from terminal_widget import TerminalWidget as TW_class
            except ImportError:
                TW_class = None

        if TW_class is not None:
            for widget in app.allWidgets():
                if isinstance(widget, TW_class):
                    widgets.append(widget)
        
        # Always include our own terminal widget if not already found
        if self._terminal_widget is not None and self._terminal_widget not in widgets:
            widgets.append(self._terminal_widget)

        return widgets

    def register_service(self, name: str, mount: str,
                         left_ports: List[PortDef],
                         right_ports: List[PortDef],
                         description: str = ""):
        """Register a custom 9P service for display."""
        NINE_P_SERVICES[name] = {
            "description": description,
            "mount": mount,
            "left_ports": left_ports,
            "right_ports": right_ports,
        }

    # ── Signal Handlers ──────────────────────────────────────────────────

    def set_dark_mode(self, enabled: bool, duration_steps: int = 50):
        """Switch the operator panel between dark and light mode.

        Called by RioWindow.toggle_dark_mode() (via TerminalWidget)
        to keep the panel in sync with the rest of the application.

        Sets the Theme proxy, then rebuilds all stylesheets and forces
        a full repaint of every node, port, connection, and chrome widget.
        """
        Theme.set_mode(enabled)

        # ── Rebuild toolbar chrome ────────────────────────────────────────
        # Find the toolbar and status frames by walking children
        for child in self.findChildren(QFrame):
            ss = child.styleSheet()
            if 'border-bottom' in ss:
                # Toolbar
                child.setStyleSheet(f"""
                    QFrame {{
                        background-color: {Theme.TOOLBAR_BG_RGBA};
                        border-bottom: 1px solid {Theme.TOOLBAR_BORDER.name()};
                    }}
                """)
            elif 'border-top' in ss:
                # Status bar
                child.setStyleSheet(f"""
                    QFrame {{
                        background-color: {Theme.STATUS_BG_RGBA};
                        border-top: 1px solid {Theme.STATUS_BORDER.name()};
                    }}
                """)

        # ── Rebuild toolbar button styles ────────────────────────────────
        for btn in self.findChildren(QPushButton):
            ss = btn.styleSheet()
            if 'font-weight: 500' in ss or 'font-weight:500' in ss:
                # Primary button
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Theme.BTN_PRIMARY_BG};
                        color: {Theme.TEXT_PRIMARY.name()};
                        border: 1px solid {Theme.BTN_PRIMARY_BORDER};
                        border-radius: 5px;
                        padding: 0 14px;
                        font-weight: 500;
                    }}
                    QPushButton:hover {{
                        background-color: {Theme.BTN_PRIMARY_HOVER_BG};
                        border-color: {Theme.BTN_PRIMARY_HOVER_BORDER};
                    }}
                """)
            elif 'border-radius: 5px' in ss or 'border-radius:5px' in ss:
                # Default toolbar button
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Theme.BTN_DEFAULT_BG_RGBA};
                        color: {Theme.TEXT_SECONDARY.name()};
                        border: 1px solid {Theme.BTN_DEFAULT_BORDER_RGBA};
                        border-radius: 5px;
                        padding: 0 12px;
                    }}
                    QPushButton:hover {{
                        background-color: {Theme.BTN_DEFAULT_HOVER_BG_RGBA};
                        color: {Theme.TEXT_PRIMARY.name()};
                        border-color: {Theme.BTN_DEFAULT_HOVER_BORDER_RGBA};
                    }}
                """)

        # ── Rebuild label styles ─────────────────────────────────────────
        for label in self.findChildren(QLabel):
            text = label.text()
            if text == "operator":
                label.setStyleSheet(f"background: transparent; color: {Theme.TEXT_PRIMARY.name()}; letter-spacing: {Theme.TITLE_LETTER_SPACING};")
            elif text == "·":
                label.setStyleSheet(f"background: transparent; color: {Theme.SEP_COLOR_RGBA}; font-size: 14px; margin: 0 6px;")
            elif text == "9p":
                label.setStyleSheet(f"background: transparent; color: {Theme.TEXT_SECONDARY.name()}; letter-spacing: 1px;")
            elif label is self._status_label:
                label.setStyleSheet(f"background: transparent; color: {Theme.TEXT_SECONDARY.name()};")
            elif label is self._mount_label:
                label.setStyleSheet(f"background: transparent; color: {Theme.TEXT_SECONDARY.name()};")
            elif label is self._count_label:
                label.setStyleSheet(f"background: transparent; color: {Theme.TEXT_SECONDARY.name()}; margin-left: 20px;")

        # ── Update node header colors for the new theme ──────────────────
        for node in self.scene.agent_nodes.values():
            # Re-map header color from the theme
            idx = list(self.scene.agent_nodes.keys()).index(node.agent_name)
            colors = [Theme.NODE_HEADER_AGENT, Theme.NODE_HEADER_AGENT_ALT,
                      Theme.NODE_HEADER_SPECIAL, Theme.NODE_HEADER_FILE]
            node.header_color = colors[idx % len(colors)]

        for node in self.scene.ninep_nodes.values():
            node.header_color = Theme.NODE_HEADER_9P

        for node in self.scene.text_nodes.values():
            node.header_color = Theme.NODE_HEADER_TEXT

        for node in self.scene.debug_nodes.values():
            node.header_color = Theme.NODE_HEADER_DEBUG

        # ── Update port visuals ──────────────────────────────────────────
        all_nodes = (list(self.scene.agent_nodes.values()) +
                     list(self.scene.ninep_nodes.values()) +
                     list(self.scene.text_nodes.values()) +
                     list(self.scene.debug_nodes.values()))
        for node in all_nodes:
            for port in node.left_ports + node.right_ports:
                port.setPen(QPen(Theme.PORT_BORDER, 1.5))
                port._label.setBrush(QBrush(Theme.TEXT_PORT))
            # Update shadow
            effect = node.graphicsEffect()
            if effect and hasattr(effect, 'setColor'):
                effect.setColor(Theme.NODE_SHADOW)

        # ── Force full scene repaint ─────────────────────────────────────
        self.scene.update()

        # ── Restyle overlay checkboxes ───────────────────────────────────
        for cb in self._node_checkboxes.values():
            cb.setStyleSheet(self._checkbox_stylesheet())

    def _on_agent_created(self, name: str):
        self._status_label.setText(f"Agent '{name}' created")
        self._update_counts()

    def _on_agent_deleted(self, name: str):
        self._status_label.setText(f"Agent '{name}' deleted")
        self._update_counts()

    def _on_connection_created(self, src_agent, src_port,
                               dst_agent, dst_port):
        self._status_label.setText(
            f"Route: {src_agent}/{src_port} → {dst_agent}/{dst_port}")
        self._update_counts()

    def _update_counts(self):
        n = len(self.scene.agent_nodes)
        s = len(self.scene.ninep_nodes)
        t = len(self.scene.text_nodes)
        d = len(self.scene.debug_nodes)
        c = len(self.scene.connections)
        parts = []
        if n:
            parts.append(f"{n} agent{'s' if n != 1 else ''}")
        if s:
            parts.append(f"{s} service{'s' if s != 1 else ''}")
        if t:
            parts.append(f"{t} text")
        if d:
            parts.append(f"{d} debug")
        parts.append(f"{c} route{'s' if c != 1 else ''}")
        self._count_label.setText(" · ".join(parts))

    # ── Overlay positioning ─────────────────────────────────────────────

    def _reposition_overlay(self):
        """Pin the node-visibility overlay to the top-left of the graph view."""
        if not hasattr(self, '_node_overlay'):
            return
        # The view's position inside the OperatorPanel layout
        view_pos = self.view.mapTo(self, self.view.rect().topLeft())
        self._node_overlay.move(view_pos.x() + 6, view_pos.y() + 6)
        self._node_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlay()

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition_overlay()

    # ── Zoom ─────────────────────────────────────────────────────────────

    def wheelEvent(self, event):
        """Zoom with scroll wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.view.scale(factor, factor)
        else:
            self.view.scale(1 / factor, 1 / factor)
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════
# Stylesheets
# ═══════════════════════════════════════════════════════════════════════════

def _menu_stylesheet() -> str:
    return f"""
        QMenu {{
            background-color: {Theme.MENU_BG};
            color: {Theme.TEXT_PRIMARY.name()};
            border: 1px solid {Theme.MENU_BORDER};
            border-radius: 6px;
            padding: 4px 0;
            font-family: '{Theme.FONT_FAMILY}';
            font-size: 12px;
        }}
        QMenu::item {{
            padding: 7px 22px;
        }}
        QMenu::item:selected {{
            background-color: {Theme.MENU_ITEM_HOVER};
            color: {Theme.TEXT_PRIMARY.name()};
        }}
        QMenu::item:disabled {{
            color: {Theme.TEXT_SECONDARY.name()};
        }}
        QMenu::separator {{
            height: 1px;
            background: {Theme.MENU_SEP};
            margin: 3px 10px;
        }}
    """

def _dialog_stylesheet() -> str:
    return f"""
        QDialog {{
            background-color: {Theme.DIALOG_BG};
            font-family: '{Theme.FONT_FAMILY}';
        }}
        QLabel {{
            color: {Theme.TEXT_PRIMARY.name()};
            font-size: 12px;
        }}
        QLineEdit, QComboBox {{
            background-color: {Theme.DIALOG_INPUT_BG};
            color: {Theme.TEXT_PRIMARY.name()};
            border: 1px solid {Theme.DIALOG_INPUT_BORDER};
            border-radius: 5px;
            padding: 7px 10px;
            font-size: 12px;
        }}
        QLineEdit:focus, QComboBox:focus {{
            border-color: {Theme.DIALOG_INPUT_FOCUS_BORDER};
        }}
        QPushButton {{
            background-color: {Theme.DIALOG_BTN_BG};
            color: {Theme.TEXT_PRIMARY.name()};
            border: 1px solid {Theme.DIALOG_BTN_BORDER};
            border-radius: 5px;
            padding: 7px 14px;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: {Theme.DIALOG_BTN_HOVER};
            border-color: {Theme.DIALOG_BTN_HOVER_BORDER};
        }}
        QDialogButtonBox QPushButton[text="OK"] {{
            background-color: {Theme.DIALOG_OK_BG};
            color: {Theme.TEXT_PRIMARY.name()};
            border: 1px solid {Theme.DIALOG_OK_BORDER};
        }}
        QDialogButtonBox QPushButton[text="OK"]:hover {{
            background-color: {Theme.DIALOG_OK_HOVER};
        }}
    """


# ═══════════════════════════════════════════════════════════════════════════
# Main (for testing)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # For testing, use a local mock directory
    panel = OperatorPanel(llmfs_mount="/n/mux/llm", rio_mount="/n/mux/default")
    panel.setWindowTitle("Operator · LLMFS")
    panel.resize(1400, 900)
    panel.show()

    sys.exit(app.exec())
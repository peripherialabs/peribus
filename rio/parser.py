"""
Parser and Executor for Rio — Parsed Elements Only

All Qt objects created by code execution are registered as parsed items
via scene_manager.register_parsed_item(). This is the sole entry point
into the version system.

No force_sync(), no blind scene sweeping. The parser explicitly registers
every object it creates.
"""

import sys
import traceback
import asyncio
from typing import Any, Optional, Callable, Dict, List
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of executing code"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    items_registered: List[int] = None
    widgets_created: List[Any] = None

    def __post_init__(self):
        if self.items_registered is None:
            self.items_registered = []
        if self.widgets_created is None:
            self.widgets_created = []


class ExecutionContext:
    """
    Execution context with full Python and PySide6 support.
    Maintains persistent variables across executions.

    After execution, discovers new Qt objects in the namespace and
    registers them as parsed items via scene_manager.register_parsed_item().
    """

    def __init__(self, scene_manager: 'SceneManager', main_window=None,
                 graphics_scene=None, graphics_view=None):
        self.scene_manager = scene_manager
        self.main_window = main_window
        self.graphics_scene = graphics_scene
        self.graphics_view = graphics_view

        # Persistent namespace across executions
        self._namespace = {}

        # Track state before execution
        self._scene_items_before = set()  # Qt scene items
        self._namespace_widgets_before = set()  # id() of widgets in namespace

        # Build initial namespace
        self._build_namespace()

    def _build_namespace(self) -> None:
        """Build the execution namespace with all PySide6 imports"""

        import builtins

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        self._namespace['__builtins__'] = builtins.__dict__.copy()
        self._namespace['__name__'] = '__main__'

        if self.main_window:
            self._namespace['main_window'] = self.main_window
        if self.graphics_scene:
            self._namespace['graphics_scene'] = self.graphics_scene
        if self.graphics_view:
            self._namespace['graphics_view'] = self.graphics_view

        self._namespace['scene_manager'] = self.scene_manager

        try:
            from PySide6 import QtWidgets, QtCore, QtGui

            self._namespace['QtWidgets'] = QtWidgets
            self._namespace['QtCore'] = QtCore
            self._namespace['QtGui'] = QtGui

            for name in dir(QtWidgets):
                if not name.startswith('_'):
                    obj = getattr(QtWidgets, name)
                    if isinstance(obj, type):
                        self._namespace[name] = obj

            self._namespace['Qt'] = QtCore.Qt
            self._namespace['QTimer'] = QtCore.QTimer
            self._namespace['QRect'] = QtCore.QRect
            self._namespace['QRectF'] = QtCore.QRectF
            self._namespace['QPoint'] = QtCore.QPoint
            self._namespace['QPointF'] = QtCore.QPointF
            self._namespace['QSize'] = QtCore.QSize
            self._namespace['QSizeF'] = QtCore.QSizeF
            self._namespace['Signal'] = QtCore.Signal
            self._namespace['Slot'] = QtCore.Slot

            self._namespace['QColor'] = QtGui.QColor
            self._namespace['QBrush'] = QtGui.QBrush
            self._namespace['QPen'] = QtGui.QPen
            self._namespace['QFont'] = QtGui.QFont
            self._namespace['QPixmap'] = QtGui.QPixmap
            self._namespace['QImage'] = QtGui.QImage
            self._namespace['QPainter'] = QtGui.QPainter

        except ImportError as e:
            print(f"Warning: PySide6 import failed: {e}")

        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self._namespace['QWebEngineView'] = QWebEngineView
        except ImportError:
            pass

        import math
        self._namespace.update({
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'pi': math.pi,
            'e': math.e,
        })

        self._namespace['asyncio'] = asyncio

        try:
            import numpy as np
            self._namespace['np'] = np
            self._namespace['numpy'] = np
        except ImportError:
            pass

        try:
            import pandas as pd
            self._namespace['pd'] = pd
            self._namespace['pandas'] = pd
        except ImportError:
            pass

        import json, os
        self._namespace['json'] = json
        self._namespace['os'] = os
        self._namespace['sys'] = sys

    def get_namespace(self) -> Dict[str, Any]:
        return self._namespace

    def track_items_before(self):
        """
        Snapshot current state before execution.

        We record:
          1. The set of Qt items currently on the graphics scene
          2. The id() of every widget already in the namespace
        """
        # All Qt items currently on the scene
        if self.graphics_scene:
            try:
                self._scene_items_before = set(self.graphics_scene.items())
            except Exception:
                self._scene_items_before = set()
        else:
            self._scene_items_before = set()

        # All widgets currently in namespace
        self._namespace_widgets_before = set()
        for v in self._namespace.values():
            if self._is_qt_graphics_item(v) or self._is_qt_widget(v):
                self._namespace_widgets_before.add(id(v))

    def track_items_after(self) -> tuple:
        """
        Discover new Qt objects after execution and register them as parsed items.

        Two detection strategies (both needed):
          1. New items on the graphics scene (added via scene.addItem / addWidget)
          2. New widgets in the namespace (proxy widgets get caught by #1,
             but we also catch any widget the user created and assigned)

        Returns:
            (new_item_ids, new_widgets)
        """
        new_item_ids = []
        new_widgets = []

        sm = self.scene_manager
        if not sm:
            return new_item_ids, new_widgets

        # --- Strategy 1: new items on the graphics scene ---
        if self.graphics_scene:
            try:
                current_scene_items = set(self.graphics_scene.items())
            except Exception:
                current_scene_items = set()

            new_scene_items = current_scene_items - self._scene_items_before

            for qt_item in new_scene_items:
                # Skip if already registered (e.g. infrastructure)
                existing = sm.get_item_by_qt(qt_item)
                if existing and existing.parsed:
                    continue

                item_id = sm.register_parsed_item(qt_item)
                new_item_ids.append(item_id)

                # Also track the underlying widget for proxy items
                if hasattr(qt_item, 'widget'):
                    try:
                        w = qt_item.widget()
                        if w:
                            new_widgets.append(w)
                            # Enable translucent background so that CSS
                            # border-radius is honoured inside the proxy.
                            # Without this the proxy paints an opaque rect
                            # that covers the rounded corners.
                            from PySide6.QtCore import Qt as _Qt
                            w.setAttribute(_Qt.WA_TranslucentBackground, True)
                    except Exception:
                        pass

        # --- Strategy 2: new widgets/graphics items in namespace ---
        for varname, obj in self._namespace.items():
            if varname.startswith('_'):
                continue
            if isinstance(obj, type):
                continue

            obj_id = id(obj)
            if obj_id in self._namespace_widgets_before:
                continue

            # Check if it's a QGraphicsItem or QWidget
            if self._is_qt_graphics_item(obj):
                existing = sm.get_item_by_qt(obj)
                if not existing:
                    # It's in the namespace but not on the scene — might have been
                    # added to the scene under a different reference. Check scene.
                    if self.graphics_scene and obj.scene() == self.graphics_scene:
                        item_id = sm.register_parsed_item(obj)
                        new_item_ids.append(item_id)
                elif not existing.parsed:
                    # Upgrade infrastructure → parsed
                    item_id = sm.register_parsed_item(obj)
                    new_item_ids.append(item_id)

            elif self._is_qt_widget(obj):
                # Check if embedded via proxy on the scene
                if self.graphics_scene:
                    for scene_item in self.graphics_scene.items():
                        if hasattr(scene_item, 'widget') and scene_item.widget() == obj:
                            existing = sm.get_item_by_qt(scene_item)
                            if not existing:
                                item_id = sm.register_parsed_item(scene_item)
                                new_item_ids.append(item_id)
                                new_widgets.append(obj)
                            elif not existing.parsed:
                                item_id = sm.register_parsed_item(scene_item)
                                new_item_ids.append(item_id)
                                new_widgets.append(obj)
                            break

        # Deduplicate
        new_item_ids = list(dict.fromkeys(new_item_ids))

        return new_item_ids, new_widgets

    def _is_qt_widget(self, obj) -> bool:
        """Check if object is a Qt widget"""
        try:
            from PySide6.QtWidgets import QWidget
            return isinstance(obj, QWidget)
        except ImportError:
            return False

    def _is_qt_graphics_item(self, obj) -> bool:
        """Check if object is a QGraphicsItem"""
        try:
            from PySide6.QtWidgets import QGraphicsItem
            return isinstance(obj, QGraphicsItem)
        except ImportError:
            return False


class Executor:
    """
    Executor that runs plain Python code.
    Registers new Qt objects as parsed items and takes versioned snapshots.
    """

    def __init__(
        self,
        context: ExecutionContext,
        error_callback: Optional[Callable[[str], Any]] = None
    ):
        self.context = context
        self.error_callback = error_callback

    async def execute(self, code: str) -> ExecutionResult:
        """Execute Python code."""

        if not code or not code.strip():
            return ExecutionResult(success=True, result=None)

        try:
            # Track scene items before execution
            self.context.track_items_before()

            namespace = self.context.get_namespace()

            # Execute code
            result = None

            try:
                compiled = compile(code, "<rio>", "eval")
                result = eval(compiled, namespace, namespace)
            except SyntaxError:
                compiled = compile(code, "<rio>", "exec")
                exec(compiled, namespace, namespace)
                result = None

            # Discover and register new parsed items
            new_items, new_widgets = self.context.track_items_after()

            # Annotate items with variable names
            self._annotate_with_variable_names(new_widgets, new_items, namespace)
            self._annotate_all_items(namespace)

            if new_items:
                print(f"Registered {len(new_items)} new parsed item(s): {new_items}")
            if new_widgets:
                print(f"Created {len(new_widgets)} widget(s)")

            # Take a snapshot after successful execution
            sm = self.context.scene_manager
            if sm:
                label = code.strip()[:80]
                sm.take_snapshot(label=label, code=code, namespace=namespace)

            return ExecutionResult(
                success=True,
                result=result,
                items_registered=new_items,
                widgets_created=new_widgets,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n"
            error_msg += traceback.format_exc()

            if self.error_callback:
                await self._report_error(error_msg)

            return ExecutionResult(
                success=False,
                error=error_msg
            )

    async def _report_error(self, error: str):
        if self.error_callback:
            result = self.error_callback(error)
            if asyncio.iscoroutine(result):
                await result

    def _annotate_with_variable_names(self, new_widgets: List[Any],
                                       new_item_ids: List[int],
                                       namespace: Dict[str, Any]):
        """Find variable names for newly created widgets and annotate scene items."""

        obj_to_varnames = self._build_obj_varname_map(namespace)

        for item_id in new_item_ids:
            item = self.context.scene_manager.get_item(item_id)
            if not item or not item.qt_item:
                continue

            varnames = self._find_varnames_for_item(item, obj_to_varnames)

            if varnames:
                item.metadata['variable_names'] = varnames
                item.metadata['primary_name'] = varnames[0]
                print(f"  • Item {item_id} -> {varnames[0]}")

    def _annotate_all_items(self, namespace: Dict[str, Any]):
        """
        Re-scan ALL parsed items and update variable name associations.
        """
        sm = self.context.scene_manager
        if not sm:
            return

        obj_to_varnames = self._build_obj_varname_map(namespace)

        for item_id, item in sm.parsed_items().items():
            if not item.qt_item:
                continue

            varnames = self._find_varnames_for_item(item, obj_to_varnames)

            if varnames:
                old_names = item.metadata.get('variable_names', [])
                if varnames != old_names:
                    item.metadata['variable_names'] = varnames
                    item.metadata['primary_name'] = varnames[0]

    def _build_obj_varname_map(self, namespace: Dict[str, Any]) -> Dict[int, List[str]]:
        """Build reverse mapping from python object id to variable names."""
        obj_to_varnames = {}

        skip_names = {
            '__builtins__', '__name__', 'scene_manager',
            'main_window', 'graphics_scene', 'graphics_view',
            'QtWidgets', 'QtCore', 'QtGui', 'asyncio',
            'json', 'os', 'sys', 'np', 'numpy', 'pd', 'pandas',
            'math', 'sin', 'cos', 'tan', 'sqrt', 'pi', 'e',
            'Qt', 'QTimer', 'QRect', 'QRectF', 'QPoint', 'QPointF',
            'QSize', 'QSizeF', 'Signal', 'Slot',
            'QColor', 'QBrush', 'QPen', 'QFont', 'QPixmap',
            'QImage', 'QPainter', 'QWebEngineView',
        }

        for varname, obj in namespace.items():
            if varname.startswith('_'):
                continue
            if varname in skip_names:
                continue
            if isinstance(obj, type):
                continue

            obj_id = id(obj)
            if obj_id not in obj_to_varnames:
                obj_to_varnames[obj_id] = []
            obj_to_varnames[obj_id].append(varname)

        return obj_to_varnames

    def _find_varnames_for_item(self, item: 'SceneItem',
                                 obj_to_varnames: Dict[int, List[str]]) -> List[str]:
        """Find variable names associated with a scene item."""
        varnames = []

        qt_item_id = id(item.qt_item)
        direct_names = obj_to_varnames.get(qt_item_id, [])
        varnames.extend(direct_names)

        # For proxy widgets, also check the underlying widget
        if hasattr(item.qt_item, 'widget'):
            try:
                widget = item.qt_item.widget()
                if widget:
                    widget_id = id(widget)
                    widget_names = obj_to_varnames.get(widget_id, [])
                    if widget_names:
                        varnames = widget_names + [n for n in varnames if n not in widget_names]
            except:
                pass

        return varnames


class StreamingParser:
    """
    Streaming parser for plain Python code.
    Accumulates code across multiple writes and executes when flushed.
    """

    def __init__(self):
        self._buffer = ""

    def feed(self, text: str) -> Optional[str]:
        self._buffer += text
        return None

    def flush(self) -> Optional[str]:
        if self._buffer.strip():
            code = self._buffer
            self._buffer = ""
            return code
        return None

    def reset(self):
        self._buffer = ""

    def get_buffer(self) -> str:
        return self._buffer

    def has_content(self) -> bool:
        return bool(self._buffer.strip())
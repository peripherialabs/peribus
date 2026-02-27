"""
Scene Manager for Rio - Parsed Elements Only

The version system tracks ONLY elements created by parsed code
(written to /n/rioa/scene/parse). Infrastructure widgets like
terminals, operator panels, and other UI chrome are never tracked,
never snapshotted, and never affected by undo/redo.

Design:
  - register_parsed_item()  → tracked, versioned, affected by undo/redo
  - register_infrastructure() → remembered (for z-ordering etc.) but
    completely invisible to the version system
  - force_sync() is REMOVED — we no longer sweep the Qt scene blindly.
    The parser is the sole entry point for tracked items.
  - clear() only removes parsed items; infrastructure stays.
"""

import asyncio
import json
import copy
import time
import pickle
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from pathlib import Path


@dataclass
class SceneItem:
    """
    Tracker for a Qt object placed on the graphics scene.
    Only items registered via the parser are versioned.
    """
    item_id: int
    qt_item: Any  # The actual Qt object
    metadata: Dict[str, Any] = field(default_factory=dict)
    parsed: bool = True  # True = from parser (versioned), False = infrastructure

    def to_dict(self) -> Dict[str, Any]:
        """Export basic info about the item"""
        result = {
            "id": self.item_id,
            "metadata": self.metadata,
            "parsed": self.parsed,
        }

        if "primary_name" in self.metadata:
            result["variable"] = self.metadata["primary_name"]
        if "variable_names" in self.metadata:
            result["variables"] = self.metadata["variable_names"]

        if self.qt_item:
            try:
                result["class"] = type(self.qt_item).__name__

                if hasattr(self.qt_item, 'pos'):
                    pos = self.qt_item.pos()
                    result["x"] = pos.x()
                    result["y"] = pos.y()

                if hasattr(self.qt_item, 'boundingRect'):
                    rect = self.qt_item.boundingRect()
                    result["width"] = rect.width()
                    result["height"] = rect.height()

                if hasattr(self.qt_item, 'geometry'):
                    geom = self.qt_item.geometry()
                    result["x"] = geom.x()
                    result["y"] = geom.y()
                    result["width"] = geom.width()
                    result["height"] = geom.height()

                if hasattr(self.qt_item, 'zValue'):
                    result["z"] = self.qt_item.zValue()

                if hasattr(self.qt_item, 'opacity'):
                    result["opacity"] = self.qt_item.opacity()

                if hasattr(self.qt_item, 'isVisible'):
                    result["visible"] = self.qt_item.isVisible()

                if hasattr(self.qt_item, 'widget'):
                    widget = self.qt_item.widget()
                    if widget:
                        result["widget_class"] = type(widget).__name__
            except:
                pass

        return result

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot of this item's state.
        Only called for parsed items.
        """
        snap = {
            "item_id": self.item_id,
            "metadata": copy.deepcopy(self.metadata),
        }

        if self.qt_item:
            try:
                if hasattr(self.qt_item, 'pos'):
                    pos = self.qt_item.pos()
                    snap["x"] = pos.x()
                    snap["y"] = pos.y()

                if hasattr(self.qt_item, 'geometry'):
                    geom = self.qt_item.geometry()
                    snap["geometry"] = {
                        "x": geom.x(),
                        "y": geom.y(),
                        "width": geom.width(),
                        "height": geom.height()
                    }

                if hasattr(self.qt_item, 'zValue'):
                    snap["z"] = self.qt_item.zValue()
                if hasattr(self.qt_item, 'opacity'):
                    snap["opacity"] = self.qt_item.opacity()
                if hasattr(self.qt_item, 'isVisible'):
                    snap["visible"] = self.qt_item.isVisible()
            except:
                pass

        return snap

    def restore_from_snapshot(self, snap: Dict[str, Any]):
        """Restore position/visibility/z/geometry from a snapshot"""
        if not self.qt_item:
            return

        try:
            if 'x' in snap or 'y' in snap:
                if hasattr(self.qt_item, 'setPos'):
                    from PySide6.QtCore import QPointF
                    x = snap.get('x', 0)
                    y = snap.get('y', 0)
                    self.qt_item.setPos(QPointF(x, y))

            if 'geometry' in snap:
                if hasattr(self.qt_item, 'setGeometry'):
                    from PySide6.QtCore import QRect
                    g = snap['geometry']
                    self.qt_item.setGeometry(QRect(g['x'], g['y'], g['width'], g['height']))

            if 'z' in snap and hasattr(self.qt_item, 'setZValue'):
                self.qt_item.setZValue(snap['z'])

            if 'opacity' in snap and hasattr(self.qt_item, 'setOpacity'):
                self.qt_item.setOpacity(snap['opacity'])

            if 'visible' in snap and hasattr(self.qt_item, 'setVisible'):
                self.qt_item.setVisible(snap['visible'])

            if 'metadata' in snap:
                self.metadata = copy.deepcopy(snap['metadata'])
        except Exception as e:
            print(f"Warning: Could not restore item {self.item_id}: {e}")


@dataclass
class SceneSnapshot:
    """
    A snapshot of parsed scene elements at a point in time.

    Only parsed items are captured — infrastructure is invisible here.
    """
    version: int
    timestamp: float
    label: str = ""
    code: str = ""  # The code that was executed to create this version

    # Item snapshots keyed by item_id (parsed items only)
    item_snapshots: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Which parsed item IDs existed at this point
    item_ids: List[int] = field(default_factory=list)

    # Serializable namespace variables
    namespace_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Scene properties
    width: int = 1920
    height: int = 1080
    background_color: str = "#FAFAFA"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "label": self.label,
            "code": self.code,
            "item_count": len(self.item_ids),
            "item_ids": self.item_ids,
            "namespace_vars": list(self.namespace_snapshot.keys()),
            "width": self.width,
            "height": self.height,
            "background": self.background_color,
        }


class VersionManager:
    """
    Manages scene snapshots for undo/redo.

    Keeps a linear history of snapshots. When you undo and then make
    a new change, the redo history is discarded (standard undo behavior).
    """

    def __init__(self, max_versions: int = 100):
        self.snapshots: List[SceneSnapshot] = []
        self.current_index: int = -1
        self._next_version: int = 1
        self.max_versions: int = max_versions

    def save(self, snapshot: SceneSnapshot) -> int:
        """
        Save a new snapshot. Discards any redo history.
        Returns the version number.
        """
        snapshot.version = self._next_version
        self._next_version += 1

        # Discard redo history
        if self.current_index < len(self.snapshots) - 1:
            self.snapshots = self.snapshots[:self.current_index + 1]

        self.snapshots.append(snapshot)
        self.current_index = len(self.snapshots) - 1

        # Trim if too many
        if len(self.snapshots) > self.max_versions:
            excess = len(self.snapshots) - self.max_versions
            self.snapshots = self.snapshots[excess:]
            self.current_index -= excess
            if self.current_index < 0:
                self.current_index = 0

        return snapshot.version

    def can_undo(self) -> bool:
        return self.current_index > 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.snapshots) - 1

    def undo(self) -> Optional[SceneSnapshot]:
        """Move back one version. Returns the snapshot to restore to."""
        if not self.can_undo():
            return None
        self.current_index -= 1
        return self.snapshots[self.current_index]

    def redo(self) -> Optional[SceneSnapshot]:
        """Move forward one version. Returns the snapshot to restore to."""
        if not self.can_redo():
            return None
        self.current_index += 1
        return self.snapshots[self.current_index]

    def goto(self, version: int) -> Optional[SceneSnapshot]:
        """Jump to a specific version number."""
        for idx, snap in enumerate(self.snapshots):
            if snap.version == version:
                self.current_index = idx
                return snap
        return None

    @property
    def current_version(self) -> int:
        if 0 <= self.current_index < len(self.snapshots):
            return self.snapshots[self.current_index].version
        return 0

    def get_current_snapshot(self) -> Optional[SceneSnapshot]:
        if 0 <= self.current_index < len(self.snapshots):
            return self.snapshots[self.current_index]
        return None

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions with metadata"""
        return [snap.to_dict() for snap in self.snapshots]


class SceneManager:
    """
    Scene Manager — parsed elements only.

    Two classes of objects live on the Qt scene:

      1. PARSED items — created by code written to /n/rioa/scene/parse.
         These are tracked, versioned, and affected by undo/redo/clear.
         Registered via register_parsed_item().

      2. INFRASTRUCTURE — terminals, panels, overlays, menus.
         These are known to the manager (for listing, z-order queries)
         but completely invisible to the version system.
         Registered via register_infrastructure().

    force_sync() is gone. The parser explicitly registers every object
    it creates — no blind sweeping of the Qt scene.
    """

    def __init__(self, width: int = 3840, height: int = 2160):
        self.width = width
        self.height = height
        self.background_color = "#FAFAFA"

        # All items (parsed + infrastructure), keyed by item_id
        self.items: Dict[int, SceneItem] = {}
        self._next_id = 1

        # Reverse lookup: qt_object → item_id
        self._qt_item_to_id: Dict[Any, int] = {}

        # Qt references
        self._qt_scene = None
        self._qt_window = None

        # Sync state
        self._lock = asyncio.Lock()

        # Version control (operates on parsed items only)
        self.versions = VersionManager()

        # Subscribers
        self._subscribers: List[Callable] = []

    # ================================================================
    # Qt Attachment
    # ================================================================

    def attach_qt(self, graphics_scene, main_window=None):
        """Attach Qt scene and optionally the main window reference."""
        self._qt_scene = graphics_scene
        self._qt_window = main_window

        if self._qt_scene:
            from PySide6.QtCore import QRectF
            self._qt_scene.setSceneRect(QRectF(0, 0, self.width, self.height))
            self.refresh_background_color()

        print(f"✓ Qt attached - Scene: {self.width}x{self.height}")

    def detach_qt(self):
        """Detach Qt references"""
        self._qt_scene = None
        self._qt_window = None

    # ================================================================
    # Subscribers
    # ================================================================

    def subscribe(self, callback: Callable):
        """Subscribe to scene change notifications"""
        self._subscribers.append(callback)

    def add_listener(self, callback: Callable):
        """Backward compatibility alias for subscribe()"""
        self.subscribe(callback)

    def remove_listener(self, callback: Callable):
        """Remove a change listener"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _notify(self, event: str, item_id: int, data: Any):
        """Notify subscribers of changes"""
        for callback in self._subscribers:
            try:
                callback(event, item_id, data)
            except Exception as e:
                print(f"Warning: Subscriber error: {e}")

    # ================================================================
    # Item Registration
    # ================================================================

    def register_parsed_item(self, qt_item: Any, metadata: Dict[str, Any] = None) -> int:
        """
        Register a parsed element for tracking and versioning.

        This is the ONLY way items enter the version system.
        Called by the parser/executor after creating Qt objects from
        code written to /n/rioa/scene/parse.

        Args:
            qt_item: The Qt graphics item or proxy widget
            metadata: Variable names, creation context, etc.

        Returns:
            The assigned item ID
        """
        # If already registered, return existing ID
        if qt_item in self._qt_item_to_id:
            existing_id = self._qt_item_to_id[qt_item]
            existing = self.items[existing_id]
            # Upgrade infrastructure → parsed if re-registered through parser
            if not existing.parsed:
                existing.parsed = True
                if metadata:
                    existing.metadata.update(metadata)
            return existing_id

        item_id = self._next_id
        self._next_id += 1

        item = SceneItem(
            item_id=item_id,
            qt_item=qt_item,
            parsed=True,
            metadata=metadata or {},
        )

        self.items[item_id] = item
        self._qt_item_to_id[qt_item] = item_id

        self._notify("add", item_id, item.to_dict())
        return item_id

    def register_infrastructure(self, qt_item: Any, label: str = "") -> int:
        """
        Register an infrastructure widget (terminal, panel, overlay).

        These are known to the manager but NEVER affected by
        undo/redo/clear/snapshots.

        Args:
            qt_item: The Qt object (proxy widget, etc.)
            label: Human-readable label (e.g. "terminal", "operator_panel")

        Returns:
            The assigned item ID
        """
        if qt_item in self._qt_item_to_id:
            return self._qt_item_to_id[qt_item]

        item_id = self._next_id
        self._next_id += 1

        item = SceneItem(
            item_id=item_id,
            qt_item=qt_item,
            parsed=False,
            metadata={"label": label, "infrastructure": True},
        )

        self.items[item_id] = item
        self._qt_item_to_id[qt_item] = item_id
        return item_id

    # Backward compat — callers that haven't been updated yet.
    # Defaults to infrastructure so nothing accidentally enters the version system.
    def register_item(self, qt_item: Any, location: str = "scene") -> int:
        """Legacy shim — registers as infrastructure by default."""
        return self.register_infrastructure(qt_item, label=f"legacy-{location}")

    def unregister_item(self, item_id: int):
        """Unregister any item (parsed or infrastructure)"""
        item = self.items.get(item_id)
        if not item:
            return

        if item.qt_item in self._qt_item_to_id:
            del self._qt_item_to_id[item.qt_item]

        was_parsed = item.parsed
        del self.items[item_id]

        if was_parsed:
            self._notify("remove", item_id, None)

    def get_item(self, item_id: int) -> Optional[SceneItem]:
        return self.items.get(item_id)

    def get_item_by_qt(self, qt_item: Any) -> Optional[SceneItem]:
        item_id = self._qt_item_to_id.get(qt_item)
        if item_id is not None:
            return self.items.get(item_id)
        return None

    # ================================================================
    # Listing — filtered views
    # ================================================================

    def list_items(self) -> List[int]:
        """List ALL tracked item IDs (parsed + infrastructure)"""
        return list(self.items.keys())

    def list_parsed_items(self) -> List[int]:
        """List only parsed (versioned) item IDs"""
        return [iid for iid, item in self.items.items() if item.parsed]

    def list_infrastructure(self) -> List[int]:
        """List only infrastructure item IDs"""
        return [iid for iid, item in self.items.items() if not item.parsed]

    def parsed_items(self) -> Dict[int, SceneItem]:
        """Return dict of parsed items only (for iteration)"""
        return {iid: item for iid, item in self.items.items() if item.parsed}

    # ================================================================
    # Scene Operations
    # ================================================================

    def refresh_background_color(self):
        if not self._qt_scene:
            return
        try:
            from PySide6.QtGui import QColor, QBrush
            bg_color = QColor(self.background_color)
            self._qt_scene.setBackgroundBrush(QBrush(bg_color))
        except Exception as e:
            print(f"Warning: Could not update background: {e}")

    async def clear(self):
        """
        Clear ONLY parsed items from the scene.
        Infrastructure (terminals, panels) is untouched.
        """
        async with self._lock:
            parsed_ids = self.list_parsed_items()

            for item_id in parsed_ids:
                item = self.items.get(item_id)
                if not item:
                    continue

                # Remove from Qt scene
                if item.qt_item and self._qt_scene:
                    try:
                        self._qt_scene.removeItem(item.qt_item)
                    except Exception:
                        pass

                # Clean up tracking
                if item.qt_item in self._qt_item_to_id:
                    del self._qt_item_to_id[item.qt_item]
                del self.items[item_id]

            self._notify("clear", 0, {"cleared": len(parsed_ids)})

    def refresh(self):
        if self._qt_scene:
            self._qt_scene.update()

    def to_json(self) -> str:
        """Export parsed items as JSON (infrastructure excluded)"""
        parsed = self.parsed_items()
        return json.dumps({
            "width": self.width,
            "height": self.height,
            "background": self.background_color,
            "items": [item.to_dict() for item in parsed.values()]
        }, indent=2)

    # ================================================================
    # Snapshot / Version Support  (parsed items only)
    # ================================================================

    def take_snapshot(self, label: str = "", code: str = "",
                      namespace: Dict[str, Any] = None) -> SceneSnapshot:
        """
        Take a snapshot of parsed scene elements.

        Infrastructure is completely excluded.

        Args:
            label: Human-readable label for this version
            code: The code that was executed
            namespace: The execution namespace (serializable vars extracted)

        Returns:
            The saved SceneSnapshot
        """
        parsed = self.parsed_items()

        snap = SceneSnapshot(
            version=0,  # Set by VersionManager
            timestamp=time.time(),
            label=label,
            code=code,
            item_ids=list(parsed.keys()),
            width=self.width,
            height=self.height,
            background_color=self.background_color,
        )

        for item_id, item in parsed.items():
            snap.item_snapshots[item_id] = item.snapshot()

        if namespace:
            snap.namespace_snapshot = self._serialize_namespace(namespace)

        self.versions.save(snap)
        return snap

    def restore_snapshot(self, snap: SceneSnapshot) -> bool:
        """
        Restore parsed scene state from a snapshot.

        Only parsed items are affected. Infrastructure is untouched.

        Returns True if restore was successful.
        """
        if not snap:
            return False

        try:
            # Restore scene properties
            self.width = snap.width
            self.height = snap.height
            self.background_color = snap.background_color

            if self._qt_scene:
                from PySide6.QtCore import QRectF
                self._qt_scene.setSceneRect(QRectF(0, 0, self.width, self.height))

            self.refresh_background_color()

            snapshot_ids = set(snap.item_ids)
            current_parsed = self.parsed_items()
            current_parsed_ids = set(current_parsed.keys())

            # Restore items that exist in both current and snapshot
            for item_id in snapshot_ids & current_parsed_ids:
                item = current_parsed[item_id]
                item_snap = snap.item_snapshots.get(item_id)
                if item_snap:
                    item.restore_from_snapshot(item_snap)

            # Hide parsed items that were added after the snapshot
            for item_id in current_parsed_ids - snapshot_ids:
                item = current_parsed[item_id]
                if item.qt_item and hasattr(item.qt_item, 'setVisible'):
                    item.qt_item.setVisible(False)

            # Show parsed items that should be visible in the snapshot
            for item_id in snapshot_ids & current_parsed_ids:
                item_snap = snap.item_snapshots.get(item_id)
                if item_snap and item_snap.get('visible', True):
                    item = current_parsed[item_id]
                    if item.qt_item and hasattr(item.qt_item, 'setVisible'):
                        item.qt_item.setVisible(True)

            self.refresh()
            self._notify("restore", snap.version, snap.to_dict())

            return True
        except Exception as e:
            print(f"Error restoring snapshot: {e}")
            import traceback
            traceback.print_exc()
            return False

    def undo(self) -> Optional[SceneSnapshot]:
        """Undo to previous version. Only parsed items affected."""
        snap = self.versions.undo()
        if snap:
            self.restore_snapshot(snap)
        return snap

    def redo(self) -> Optional[SceneSnapshot]:
        """Redo to next version. Only parsed items affected."""
        snap = self.versions.redo()
        if snap:
            self.restore_snapshot(snap)
        return snap

    def goto_version(self, version: int) -> Optional[SceneSnapshot]:
        """Jump to a specific version. Only parsed items affected."""
        snap = self.versions.goto(version)
        if snap:
            self.restore_snapshot(snap)
        return snap

    def _serialize_namespace(self, namespace: Dict[str, Any]) -> Dict[str, Any]:
        """Extract serializable variables from namespace for snapshotting."""
        result = {}
        for key, value in namespace.items():
            if key.startswith('_') or key in ('__builtins__', '__name__'):
                continue
            if hasattr(value, '__file__') or isinstance(value, type):
                continue

            try:
                if isinstance(value, (int, float, str, bool, type(None))):
                    result[key] = value
                elif isinstance(value, (list, dict, tuple)):
                    json.dumps(value)
                    result[key] = value
                else:
                    result[key] = f"<{type(value).__name__}>"
            except (TypeError, ValueError):
                result[key] = f"<{type(value).__name__}>"

        return result

    # ================================================================
    # Scene Queries
    # ================================================================

    def get_items_at(self, x: float, y: float) -> List[int]:
        """Get parsed item IDs at a position (infrastructure excluded)"""
        if not self._qt_scene:
            return []
        try:
            from PySide6.QtCore import QPointF
            point = QPointF(x, y)
            qt_items = self._qt_scene.items(point)

            item_ids = []
            for qt_item in qt_items:
                item_id = self._qt_item_to_id.get(qt_item)
                if item_id is not None:
                    item = self.items.get(item_id)
                    if item and item.parsed:
                        item_ids.append(item_id)
            return item_ids
        except Exception as e:
            print(f"Warning: Could not get items at position: {e}")
            return []

    def remove_item_from_scene(self, item_id: int):
        """Remove a parsed item from the scene. Refuses to remove infrastructure."""
        item = self.get_item(item_id)
        if not item or not item.qt_item:
            return

        if not item.parsed:
            print(f"Warning: refusing to remove infrastructure item {item_id}")
            return

        try:
            if self._qt_scene:
                self._qt_scene.removeItem(item.qt_item)
            self.unregister_item(item_id)
        except Exception as e:
            print(f"Warning: Could not remove item: {e}")

    # ================================================================
    # Save/Load for Crash Recovery
    # ================================================================

    def save_state(self, filepath: str = None) -> bool:
        """
        Save version history to disk for crash recovery.
        Only parsed element snapshots are persisted.
        """
        if filepath is None:
            filepath = str(Path.home() / ".rio_state.pkl")

        try:
            state = {
                "width": self.width,
                "height": self.height,
                "background_color": self.background_color,
                "next_id": self._next_id,
                "versions": {
                    "snapshots": [
                        {
                            "version": s.version,
                            "timestamp": s.timestamp,
                            "label": s.label,
                            "code": s.code,
                            "item_ids": s.item_ids,
                            "item_snapshots": s.item_snapshots,
                            "namespace_snapshot": s.namespace_snapshot,
                            "width": s.width,
                            "height": s.height,
                            "background_color": s.background_color,
                        }
                        for s in self.versions.snapshots
                    ],
                    "current_index": self.versions.current_index,
                    "next_version": self.versions._next_version,
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            print(f"State saved to {filepath}")
            return True

        except Exception as e:
            print(f"Error saving state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_state(self, filepath: str = None) -> bool:
        """
        Load version history from disk.

        Note: Only metadata and version history are restored.
        Qt objects must be recreated by re-executing the stored code.
        """
        if filepath is None:
            filepath = str(Path.home() / ".rio_state.pkl")

        if not os.path.exists(filepath):
            print(f"No state file found at {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.width = state["width"]
            self.height = state["height"]
            self.background_color = state["background_color"]
            self._next_id = state["next_id"]

            versions_data = state["versions"]
            self.versions.snapshots = []
            for snap_dict in versions_data["snapshots"]:
                snap = SceneSnapshot(
                    version=snap_dict["version"],
                    timestamp=snap_dict["timestamp"],
                    label=snap_dict["label"],
                    code=snap_dict["code"],
                    item_ids=snap_dict["item_ids"],
                    item_snapshots=snap_dict["item_snapshots"],
                    namespace_snapshot=snap_dict["namespace_snapshot"],
                    width=snap_dict["width"],
                    height=snap_dict["height"],
                    background_color=snap_dict["background_color"],
                )
                self.versions.snapshots.append(snap)

            self.versions.current_index = versions_data["current_index"]
            self.versions._next_version = versions_data["next_version"]

            print(f"  State loaded from {filepath}")
            print(f"  Restored {len(self.versions.snapshots)} versions")
            print(f"  NOTE: Qt objects need to be recreated by re-executing code")

            return True

        except Exception as e:
            print(f"Error loading state: {e}")
            import traceback
            traceback.print_exc()
            return False
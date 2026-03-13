"""
Mapbox Vector Tile Map — High-Performance ModernGL Edition v8
==============================================================
v8 "AAA STREAMING" OVERHAUL — key changes from v7:

 1. Predictive Pre-Tessellation ("Loading Horizon") —
    Based on camera velocity + heading in immersive/car mode, tiles
    are requested 2–3 seconds ahead of the flight path.  No more
    waiting until the camera physically enters a new tile boundary.

 2. Time-Sliced GPU Uploads — tile VBO uploads are queued and
    spread across multiple frames (1-2 per 8ms timer tick) instead
    of uploading all geometry in a single frame.  Keeps per-frame
    GPU work under ~4 ms, eliminating the "tile hitch".

 3. Terrain Texture Atlas (Slot-Based) — terrain DEM tiles are
    written into individual slots of a pre-allocated 1024×1024
    R32F atlas texture via write(viewport=...) instead of
    re-stitching a new np.zeros array + creating a new
    moderngl.Texture on every tile boundary crossing.

 4. VBO Pool (Ring Buffer) — REMOVED after testing. moderngl
    VAOs derive their vertex count from buffer size, so pre-
    allocated oversized buffers caused rendering of stale data
    (garbage vertices / wrong colors). Exact-sized ctx.buffer()
    is used instead; the time-sliced upload queue (Opt 2)
    handles the frame-time budget problem that VBO pooling was
    meant to solve.

 5. Decoupled Physics Thread (120 Hz) — immersive-mode walk/car
    physics run on a dedicated Python thread at 120 Hz.  Only the
    final (wx, wy, yaw, speed) coordinates are shared via a Lock.
    Camera movement stays buttery smooth even when the main thread
    is busy decoding tiles or uploading VBOs.

Previous v7 features (all kept):
- Numpy row-flip replaces QImage.mirrored()
- Cached label overlay
- QPainterPath halo replaced with multi-offset drawText
- Cached POI extraction + VBO rebuild
- Network concurrency 12, render workers 10
- GPU-direct geometry rendering (tessellated VBOs)
- Geometry VBO caching per tile+style
- Parent zoom fallback with GL scissor clipping
- Frustum-culled tile set, Hybrid LOD

Controls: (same as v5)
    Left-drag       pan (bearing-aware)
    Right-drag      tilt (up/down) + rotate bearing (left/right)
    Scroll          zoom toward cursor
    +/-             zoom in / out
    S               cycle styles
    R               reset view
    B               toggle 3D buildings
    T               reset tilt to 0
    F               toggle labels
    Middle-drag     rotate bearing
    N               toggle route mode (click to place waypoints)
    C               clear route
    M               cycle route profile (driving/walking/cycling)
    L               toggle building shadows
    H               cycle heatmap (POI → traffic → building → off)
    O               toggle POI markers (restaurants, cafes, shops…)
    V               toggle tile visibility (transparent to scene)
    /               open search bar (type destination, Enter to navigate)
    Y               cycle isochrone overlay (driving → walking → cycling → off)

Immersive + Car mode:
    I               enter immersive (click to place)
    WASD            walk / Space+Shift up/down
    O               toggle POI markers (3D billboards)
    Up/Down (+/-)   cycle through nearby POIs
    Enter           navigate to selected POI (auto-route)
    E               open POI in Google Maps (when popup visible)
    X               search POI on web (when popup visible)
    J               open Wikipedia page (when popup visible)
    G               toggle car mode (spawns car at your feet)
    W/S             gas / brake (car mode)
    A/D             steer (car mode)
    Space           handbrake (car mode)
    Mouse           look (walk) / orbit camera (car)
    Scroll          adjust speed (walk)
    P               toggle traffic (NPC cars on roads)
    V               hide tiles (city reconstruction)
    ESC             exit
"""

import os
import math
import struct
import gzip
import threading
import time
import datetime
import json
import numpy as np
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional, List, Any
from io import BytesIO
import queue

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QFrame, QGraphicsItem, QCheckBox, QLineEdit, QSpinBox, QComboBox,
)
from PySide6.QtCore import Qt, QTimer, QPoint, QPointF, Signal, QObject, QRectF, QUrl
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QCursor,
    QPolygonF, QPainterPath, QLinearGradient, QImage, QPixmap,
    QTransform, QDesktopServices,
)
from PySide6.QtNetwork import (
    QNetworkAccessManager, QNetworkRequest, QNetworkReply,
    QNetworkDiskCache,
)

import moderngl

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
TILE_SIZE = 256
MVT_EXTENT = 4096

# ---------------------------------------------------------------------------
#  Coordinate helpers
# ---------------------------------------------------------------------------

def _world_size(zoom: float) -> float:
    return TILE_SIZE * (2.0 ** zoom)

def _lat_to_wy(lat: float, zoom: float) -> float:
    lat_r = math.radians(max(-85.051129, min(85.051129, lat)))
    s = math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi
    return (1.0 - s) / 2.0 * _world_size(zoom)

def _lon_to_wx(lon: float, zoom: float) -> float:
    return (lon + 180.0) / 360.0 * _world_size(zoom)

def _wx_to_lon(wx: float, zoom: float) -> float:
    return wx / _world_size(zoom) * 360.0 - 180.0

def _wy_to_lat(wy: float, zoom: float) -> float:
    ws = _world_size(zoom)
    n = math.pi * (1.0 - 2.0 * wy / ws)
    return math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))

def _clamp_zoom(z: float) -> float:
    return max(1.0, min(20.0, z))

def _clamp_pitch(p: float) -> float:
    return max(0.0, min(60.0, p))

def _normalize_bearing(b: float) -> float:
    b %= 360.0
    if b >= 180.0:
        b -= 360.0
    elif b < -180.0:
        b += 360.0
    return b


# ===========================================================================
#  Smooth Camera Animation ("flyTo") — Mapbox GL JS style
# ===========================================================================

def _ease_in_out_cubic(t):
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0

def _ease_out_cubic(t):
    return 1.0 - (1.0 - t) ** 3


class FlyToAnimation:
    """State for an in-progress flyTo camera transition.

    All positions are stored as lat/lon (zoom-independent).  Each frame,
    the interpolated (lat, lon, zoom) is converted to world-pixels by the
    caller.  This avoids floating-point drift that occurs when accumulating
    world-pixel deltas across zoom changes."""
    __slots__ = (
        'active',
        'start_lat', 'start_lon', 'start_zoom', 'start_bearing', 'start_pitch',
        'end_lat', 'end_lon', 'end_zoom', 'end_bearing', 'end_pitch',
        'duration', 'elapsed',
        'min_zoom',       # how far to pull back during the arc
        'ease_fn',
        'suppress_tiles', # skip intermediate tile requests
    )

    def __init__(self):
        self.active = False
        self.duration = 2.0; self.elapsed = 0.0
        self.min_zoom = 1.0
        self.ease_fn = _ease_in_out_cubic
        self.suppress_tiles = False
        for attr in ('start_lat','start_lon','start_zoom','start_bearing','start_pitch',
                     'end_lat','end_lon','end_zoom','end_bearing','end_pitch'):
            setattr(self, attr, 0.0)

    def begin(self, s_lat, s_lon, s_zoom, s_bearing, s_pitch,
              e_lat, e_lon, e_zoom, e_bearing=None, e_pitch=None,
              duration=None, speed=1.2):
        self.start_lat = s_lat; self.start_lon = s_lon
        self.start_zoom = s_zoom
        self.start_bearing = s_bearing; self.start_pitch = s_pitch
        self.end_lat = e_lat; self.end_lon = e_lon
        self.end_zoom = e_zoom
        self.end_bearing = e_bearing if e_bearing is not None else s_bearing
        self.end_pitch = e_pitch if e_pitch is not None else s_pitch

        dlat = abs(e_lat - s_lat)
        dlon = abs(e_lon - s_lon)
        if dlon > 180.0: dlon = 360.0 - dlon
        dist_deg = math.hypot(dlat, dlon)

        if duration is not None:
            self.duration = max(0.2, min(8.0, duration))
        else:
            zoom_diff = abs(e_zoom - s_zoom)
            t = dist_deg / 10.0 * 1.5 + zoom_diff * 0.15
            self.duration = max(0.3, min(8.0, t / speed))

        if dist_deg < 0.01:
            self.min_zoom = min(s_zoom, e_zoom)
        else:
            pull_back = min(8.0, max(1.0, math.log2(dist_deg + 1.0) * 2.5))
            self.min_zoom = max(1.0, min(s_zoom, e_zoom) - pull_back)

        self.elapsed = 0.0
        self.active = True
        self.suppress_tiles = dist_deg > 0.5 or abs(e_zoom - s_zoom) > 2.0
        self.ease_fn = _ease_in_out_cubic

    def step(self, dt):
        """Advance animation.  Returns (lat, lon, zoom, bearing, pitch, finished)."""
        if not self.active:
            return (self.end_lat, self.end_lon, self.end_zoom,
                    self.end_bearing, self.end_pitch, True)

        self.elapsed += dt
        raw_t = min(1.0, self.elapsed / max(self.duration, 0.01))
        t = self.ease_fn(raw_t)
        finished = raw_t >= 1.0

        # Lat/lon — linear interp (handles lon wrapping)
        lat = self.start_lat + (self.end_lat - self.start_lat) * t
        dlon = self.end_lon - self.start_lon
        if dlon > 180.0: dlon -= 360.0
        elif dlon < -180.0: dlon += 360.0
        lon = self.start_lon + dlon * t

        # Zoom — parabolic arc through min_zoom
        z_s, z_e, z_m = self.start_zoom, self.end_zoom, self.min_zoom
        if abs(z_m - min(z_s, z_e)) < 0.5:
            zoom = z_s + (z_e - z_s) * t
        else:
            a = 2.0 * (z_s + z_e - 2.0 * z_m)
            b = (z_e - z_s) - a
            zoom = a * t * t + b * t + z_s
        zoom = _clamp_zoom(zoom)

        # Bearing — shortest path
        bd = self.end_bearing - self.start_bearing
        if bd > 180.0: bd -= 360.0
        elif bd < -180.0: bd += 360.0
        bearing = _normalize_bearing(self.start_bearing + bd * t)

        pitch = self.start_pitch + (self.end_pitch - self.start_pitch) * t
        pitch = _clamp_pitch(pitch)

        if finished:
            self.active = False
            return (self.end_lat, self.end_lon, self.end_zoom,
                    self.end_bearing, self.end_pitch, True)

        return (lat, lon, zoom, bearing, pitch, False)

    def cancel(self):
        self.active = False

def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = shift = 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7
    return result, pos

def _decode_zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)

def _parse_pbf_field(buf: bytes, pos: int):
    tag, pos = _read_varint(buf, pos)
    field_num = tag >> 3; wire_type = tag & 0x07
    if wire_type == 0:
        val, pos = _read_varint(buf, pos)
    elif wire_type == 1:
        val = struct.unpack('<d', buf[pos:pos+8])[0]; pos += 8
    elif wire_type == 2:
        length, pos = _read_varint(buf, pos)
        val = buf[pos:pos+length]; pos += length
    elif wire_type == 5:
        val = struct.unpack('<f', buf[pos:pos+4])[0]; pos += 4
    else:
        raise ValueError(f"Unsupported wire type {wire_type}")
    return field_num, wire_type, val, pos

def _parse_packed_varints(buf: bytes) -> List[int]:
    result = []; pos = 0
    while pos < len(buf):
        val, pos = _read_varint(buf, pos)
        result.append(val)
    return result


class MVTFeature:
    __slots__ = ('fid', 'tags', 'geom_type', 'geometry', 'properties', '_decoded_rings')
    def __init__(self):
        self.fid = 0; self.tags: List[int] = []; self.geom_type = 0
        self.geometry: List[int] = []; self.properties: Dict[str, Any] = {}
        self._decoded_rings = None

class MVTLayer:
    __slots__ = ('name', 'extent', 'features', 'keys', 'values', 'version')
    def __init__(self):
        self.name = ""; self.extent = MVT_EXTENT
        self.features: List[MVTFeature] = []
        self.keys: List[str] = []; self.values: List[Any] = []
        self.version = 2

class MVTTile:
    __slots__ = ('layers',)
    def __init__(self):
        self.layers: Dict[str, MVTLayer] = {}


def _parse_value(buf: bytes):
    pos = 0
    while pos < len(buf):
        fn, wt, val, pos = _parse_pbf_field(buf, pos)
        if fn == 1: return val.decode('utf-8', errors='replace')
        elif fn in (2, 3, 4, 5): return val
        elif fn == 6: return _decode_zigzag(val)
        elif fn == 7: return bool(val)
    return None

def _parse_feature(buf: bytes) -> MVTFeature:
    feat = MVTFeature(); pos = 0
    while pos < len(buf):
        fn, wt, val, pos = _parse_pbf_field(buf, pos)
        if fn == 1: feat.fid = val
        elif fn == 2: feat.tags = _parse_packed_varints(val) if isinstance(val, (bytes, bytearray)) else [val]
        elif fn == 3: feat.geom_type = val
        elif fn == 4: feat.geometry = _parse_packed_varints(val) if isinstance(val, (bytes, bytearray)) else [val]
    return feat


def _parse_layer(buf: bytes) -> MVTLayer:
    layer = MVTLayer(); pos = 0; raw_features = []
    while pos < len(buf):
        fn, wt, val, pos = _parse_pbf_field(buf, pos)
        if fn == 1:
            layer.name = val.decode('utf-8', errors='replace') if isinstance(val, (bytes, bytearray)) else str(val)
        elif fn == 2: raw_features.append(val)
        elif fn == 3:
            layer.keys.append(val.decode('utf-8', errors='replace') if isinstance(val, (bytes, bytearray)) else str(val))
        elif fn == 4:
            layer.values.append(_parse_value(val) if isinstance(val, (bytes, bytearray)) else val)
        elif fn == 5: layer.extent = val
        elif fn == 15: layer.version = val
    for fb in raw_features:
        feat = _parse_feature(fb)
        i = 0
        while i + 1 < len(feat.tags):
            ki, vi = feat.tags[i], feat.tags[i+1]
            if ki < len(layer.keys) and vi < len(layer.values):
                feat.properties[layer.keys[ki]] = layer.values[vi]
            i += 2
        layer.features.append(feat)
    return layer

def decode_mvt(data: bytes) -> MVTTile:
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    tile = MVTTile(); pos = 0
    while pos < len(data):
        fn, wt, val, pos = _parse_pbf_field(data, pos)
        if fn == 3 and isinstance(val, (bytes, bytearray)):
            layer = _parse_layer(val)
            tile.layers[layer.name] = layer
    return tile


def decode_geometry(commands: List[int], extent: int = MVT_EXTENT):
    rings = []; current = []; cx = cy = 0; i = 0
    while i < len(commands):
        cmd_int = commands[i]; cmd_id = cmd_int & 0x07; cmd_count = cmd_int >> 3; i += 1
        if cmd_id == 1:
            for _ in range(cmd_count):
                if i + 1 >= len(commands): break
                cx += _decode_zigzag(commands[i]); cy += _decode_zigzag(commands[i+1]); i += 2
                if current: rings.append(current)
                current = [(cx, cy)]
        elif cmd_id == 2:
            for _ in range(cmd_count):
                if i + 1 >= len(commands): break
                cx += _decode_zigzag(commands[i]); cy += _decode_zigzag(commands[i+1]); i += 2
                current.append((cx, cy))
        elif cmd_id == 7:
            if current and len(current) > 1:
                current.append(current[0])
            if current: rings.append(current); current = []
    if current: rings.append(current)
    return rings

def _get_rings(feat, ext=MVT_EXTENT):
    if feat._decoded_rings is not None: return feat._decoded_rings
    feat._decoded_rings = decode_geometry(feat.geometry, ext)
    return feat._decoded_rings


# ===========================================================================
#  Style definitions
# ===========================================================================

class MapStyle:
    __slots__ = ('name', 'bg', 'layers')
    def __init__(self, name, bg, layers):
        self.name = name; self.bg = bg; self.layers = layers

def _make_styles():
    dark = MapStyle("dark-v10", QColor(14, 16, 22), [
        {"match": "water", "type": "fill", "fill": QColor(22, 34, 56), "stroke": QColor(22, 34, 56, 80), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(22, 26, 32), "stroke": Qt.NoPen},
        {"match": "landuse_overlay", "type": "fill", "fill": QColor(18, 38, 24, 120), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(18, 20, 28), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(0, 0, 0, 30), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(32, 36, 48, 200), "stroke": QColor(42, 48, 62, 160), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "subtype": "tunnel", "stroke": QColor(38, 42, 55, 140), "stroke_w": 1.5, "dash": [4, 4]},
        {"match": "road", "type": "line", "stroke": QColor(55, 62, 80), "stroke_w": 1.2,
         "class_widths": {"motorway": 3.5, "trunk": 3.0, "primary": 2.4, "secondary": 1.8, "tertiary": 1.4, "street": 1.0, "service": 0.6, "path": 0.4, "pedestrian": 0.5},
         "class_colors": {"motorway": QColor(90, 70, 45), "trunk": QColor(75, 65, 42), "primary": QColor(60, 60, 55)}},
        {"match": "bridge", "type": "line", "stroke": QColor(70, 78, 100), "stroke_w": 2.0},
        {"match": "road", "type": "line", "subtype": "rail", "stroke": QColor(60, 60, 75, 160), "stroke_w": 0.8, "dash": [6, 4]},
        {"match": "waterway", "type": "line", "stroke": QColor(22, 38, 62), "stroke_w": 1.2},
        {"match": "admin", "type": "line", "stroke": QColor(80, 60, 90, 120), "stroke_w": 1.0, "dash": [6, 3]},
        {"match": "place_label", "type": "label", "color": QColor(170, 175, 190), "font_size": 11, "halo": QColor(14, 16, 22, 200)},
        {"match": "poi_label", "type": "label", "color": QColor(140, 150, 160, 180), "font_size": 8, "halo": QColor(14, 16, 22, 160)},
        {"match": "transit", "type": "line", "stroke": QColor(50, 55, 70, 100), "stroke_w": 0.8},
    ])
    streets = MapStyle("streets-v12", QColor(242, 239, 233), [
        {"match": "water", "type": "fill", "fill": QColor(170, 210, 223), "stroke": QColor(150, 195, 212, 120), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(230, 228, 220), "stroke": Qt.NoPen},
        {"match": "landuse_overlay", "type": "fill", "fill": QColor(200, 220, 190, 140), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(242, 239, 233), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(0, 0, 0, 15), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(218, 211, 205, 220), "stroke": QColor(200, 195, 188, 160), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(255, 255, 255), "stroke_w": 1.4,
         "class_widths": {"motorway": 4.0, "trunk": 3.5, "primary": 2.8, "secondary": 2.2, "tertiary": 1.6, "street": 1.2, "service": 0.7, "path": 0.4, "pedestrian": 0.5},
         "class_colors": {"motorway": QColor(240, 195, 100), "trunk": QColor(248, 210, 130), "primary": QColor(255, 255, 255)}},
        {"match": "bridge", "type": "line", "stroke": QColor(255, 255, 255), "stroke_w": 2.5},
        {"match": "waterway", "type": "line", "stroke": QColor(170, 210, 223), "stroke_w": 1.0},
        {"match": "admin", "type": "line", "stroke": QColor(160, 140, 170, 150), "stroke_w": 1.2, "dash": [6, 3]},
        {"match": "place_label", "type": "label", "color": QColor(60, 60, 70), "font_size": 11, "halo": QColor(255, 255, 255, 220)},
        {"match": "poi_label", "type": "label", "color": QColor(100, 105, 115, 200), "font_size": 8, "halo": QColor(255, 255, 255, 180)},
        {"match": "transit", "type": "line", "stroke": QColor(190, 185, 180, 120), "stroke_w": 0.8},
    ])
    satellite = MapStyle("satellite-v9", QColor(30, 40, 30), [
        {"match": "water", "type": "fill", "fill": QColor(15, 30, 55), "stroke": QColor(10, 25, 50, 100), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(35, 50, 32, 80), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(40, 48, 35), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(0, 0, 0, 40), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(60, 55, 50, 180), "stroke": QColor(80, 75, 65, 140), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(120, 110, 90, 180), "stroke_w": 1.0,
         "class_widths": {"motorway": 3.0, "trunk": 2.5, "primary": 2.0, "secondary": 1.4, "tertiary": 1.0, "street": 0.7}},
        {"match": "waterway", "type": "line", "stroke": QColor(15, 30, 55), "stroke_w": 1.0},
        {"match": "place_label", "type": "label", "color": QColor(220, 220, 210), "font_size": 11, "halo": QColor(20, 20, 20, 200)},
    ])
    outdoors = MapStyle("outdoors-v12", QColor(238, 236, 228), [
        {"match": "water", "type": "fill", "fill": QColor(160, 200, 215), "stroke": QColor(140, 185, 205, 120), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(225, 225, 215), "stroke": Qt.NoPen},
        {"match": "landuse_overlay", "type": "fill", "fill": QColor(185, 210, 170, 160), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(238, 236, 228), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(0, 0, 0, 25), "stroke": Qt.NoPen},
        {"match": "contour", "type": "line", "stroke": QColor(180, 170, 155, 80), "stroke_w": 0.5},
        {"match": "building", "type": "fill", "fill": QColor(210, 205, 198, 200), "stroke": QColor(195, 190, 182, 150), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(255, 255, 255), "stroke_w": 1.2,
         "class_widths": {"motorway": 3.5, "trunk": 3.0, "primary": 2.4, "secondary": 1.8, "tertiary": 1.3, "street": 1.0, "path": 0.5, "pedestrian": 0.5},
         "class_colors": {"motorway": QColor(240, 195, 100), "path": QColor(190, 100, 80, 160)}},
        {"match": "waterway", "type": "line", "stroke": QColor(160, 200, 215), "stroke_w": 1.0},
        {"match": "admin", "type": "line", "stroke": QColor(150, 130, 160, 140), "stroke_w": 1.0, "dash": [6, 3]},
        {"match": "place_label", "type": "label", "color": QColor(55, 55, 65), "font_size": 11, "halo": QColor(238, 236, 228, 220)},
        {"match": "poi_label", "type": "label", "color": QColor(90, 95, 100, 200), "font_size": 8, "halo": QColor(238, 236, 228, 180)},
    ])
    nav_night = MapStyle("navigation-night-v1", QColor(6, 8, 14), [
        {"match": "water", "type": "fill", "fill": QColor(16, 28, 48), "stroke": QColor(16, 28, 48, 80), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(16, 18, 26), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(12, 14, 22), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(26, 30, 42, 200), "stroke": QColor(36, 42, 56, 140), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(45, 55, 75), "stroke_w": 1.4,
         "class_widths": {"motorway": 4.0, "trunk": 3.5, "primary": 2.8, "secondary": 2.0, "tertiary": 1.4, "street": 1.0},
         "class_colors": {"motorway": QColor(70, 90, 140), "trunk": QColor(60, 75, 110), "primary": QColor(50, 60, 85)}},
        {"match": "waterway", "type": "line", "stroke": QColor(16, 28, 48), "stroke_w": 1.0},
        {"match": "place_label", "type": "label", "color": QColor(150, 160, 180), "font_size": 11, "halo": QColor(6, 8, 14, 200)},
    ])
    light = MapStyle("light-v11", QColor(250, 249, 246), [
        {"match": "water", "type": "fill", "fill": QColor(190, 220, 232), "stroke": QColor(170, 205, 220, 120), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(240, 238, 232), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(250, 249, 246), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(228, 224, 218, 200), "stroke": QColor(215, 210, 204, 150), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(255, 255, 255), "stroke_w": 1.2,
         "class_widths": {"motorway": 3.5, "trunk": 3.0, "primary": 2.5, "secondary": 1.8, "tertiary": 1.3, "street": 1.0},
         "class_colors": {"motorway": QColor(250, 210, 120), "trunk": QColor(252, 220, 145)}},
        {"match": "waterway", "type": "line", "stroke": QColor(190, 220, 232), "stroke_w": 1.0},
        {"match": "place_label", "type": "label", "color": QColor(70, 70, 80), "font_size": 11, "halo": QColor(250, 249, 246, 220)},
    ])
    cyberpunk = MapStyle("cyberpunk", QColor(8, 2, 18), [
        {"match": "water", "type": "fill", "fill": QColor(5, 8, 35), "stroke": QColor(0, 200, 255, 50), "stroke_w": 1.0},
        {"match": "landuse", "type": "fill", "fill": QColor(12, 5, 25, 180), "stroke": Qt.NoPen},
        {"match": "landuse_overlay", "type": "fill", "fill": QColor(20, 40, 10, 80), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(10, 4, 20), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(80, 0, 120, 20), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(18, 8, 40, 220), "stroke": QColor(120, 0, 255, 100), "stroke_w": 0.8, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(255, 0, 120), "stroke_w": 1.4,
         "class_widths": {"motorway": 4.0, "trunk": 3.5, "primary": 2.8, "secondary": 2.0, "tertiary": 1.4, "street": 0.8, "service": 0.5, "path": 0.3},
         "class_colors": {"motorway": QColor(255, 0, 200), "trunk": QColor(200, 0, 255), "primary": QColor(255, 50, 150)}},
        {"match": "bridge", "type": "line", "stroke": QColor(0, 255, 200), "stroke_w": 2.5},
        {"match": "waterway", "type": "line", "stroke": QColor(0, 150, 255, 120), "stroke_w": 1.5},
        {"match": "admin", "type": "line", "stroke": QColor(255, 0, 100, 80), "stroke_w": 1.0, "dash": [4, 4]},
        {"match": "transit", "type": "line", "stroke": QColor(0, 255, 180, 80), "stroke_w": 0.8},
        {"match": "place_label", "type": "label", "color": QColor(0, 255, 220), "font_size": 11, "halo": QColor(8, 2, 18, 220)},
        {"match": "poi_label", "type": "label", "color": QColor(200, 0, 255, 180), "font_size": 8, "halo": QColor(8, 2, 18, 180)},
    ])
    blueprint = MapStyle("blueprint", QColor(15, 30, 80), [
        {"match": "water", "type": "fill", "fill": QColor(10, 22, 60), "stroke": QColor(60, 120, 200, 100), "stroke_w": 0.8},
        {"match": "landuse", "type": "fill", "fill": QColor(18, 35, 85, 120), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(15, 30, 80), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(0, 0, 0, 20), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(20, 40, 100, 100), "stroke": QColor(80, 160, 255, 120), "stroke_w": 0.8, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(70, 140, 220, 200), "stroke_w": 1.0,
         "class_widths": {"motorway": 3.0, "trunk": 2.5, "primary": 2.0, "secondary": 1.5, "tertiary": 1.0, "street": 0.7, "path": 0.4},
         "class_colors": {"motorway": QColor(100, 180, 255), "trunk": QColor(90, 170, 245)}},
        {"match": "bridge", "type": "line", "stroke": QColor(100, 180, 255), "stroke_w": 2.0},
        {"match": "waterway", "type": "line", "stroke": QColor(50, 100, 180, 150), "stroke_w": 1.0},
        {"match": "admin", "type": "line", "stroke": QColor(100, 160, 255, 100), "stroke_w": 1.0, "dash": [8, 4]},
        {"match": "place_label", "type": "label", "color": QColor(150, 200, 255), "font_size": 11, "halo": QColor(15, 30, 80, 220)},
        {"match": "poi_label", "type": "label", "color": QColor(100, 160, 220, 160), "font_size": 8, "halo": QColor(15, 30, 80, 180)},
    ])
    sepia = MapStyle("sepia", QColor(235, 220, 195), [
        {"match": "water", "type": "fill", "fill": QColor(170, 185, 170), "stroke": QColor(150, 165, 150, 100), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(225, 210, 185), "stroke": Qt.NoPen},
        {"match": "landuse_overlay", "type": "fill", "fill": QColor(200, 195, 165, 120), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(235, 220, 195), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(80, 60, 30, 20), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(210, 195, 170, 200), "stroke": QColor(180, 165, 140, 140), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(245, 235, 215), "stroke_w": 1.2,
         "class_widths": {"motorway": 3.5, "trunk": 3.0, "primary": 2.4, "secondary": 1.8, "tertiary": 1.2, "street": 0.8},
         "class_colors": {"motorway": QColor(190, 150, 90), "trunk": QColor(200, 170, 110)}},
        {"match": "waterway", "type": "line", "stroke": QColor(160, 175, 160), "stroke_w": 1.0},
        {"match": "admin", "type": "line", "stroke": QColor(160, 130, 100, 120), "stroke_w": 1.0, "dash": [6, 3]},
        {"match": "place_label", "type": "label", "color": QColor(100, 70, 40), "font_size": 11, "halo": QColor(235, 220, 195, 220)},
        {"match": "poi_label", "type": "label", "color": QColor(140, 110, 80, 180), "font_size": 8, "halo": QColor(235, 220, 195, 180)},
    ])
    nord = MapStyle("nord", QColor(46, 52, 64), [
        {"match": "water", "type": "fill", "fill": QColor(59, 66, 82), "stroke": QColor(94, 129, 172, 80), "stroke_w": 0.5},
        {"match": "landuse", "type": "fill", "fill": QColor(50, 56, 70), "stroke": Qt.NoPen},
        {"match": "landuse_overlay", "type": "fill", "fill": QColor(55, 75, 65, 100), "stroke": Qt.NoPen},
        {"match": "land", "type": "fill", "fill": QColor(46, 52, 64), "stroke": Qt.NoPen},
        {"match": "hillshade", "type": "fill", "fill": QColor(0, 0, 0, 20), "stroke": Qt.NoPen},
        {"match": "building", "type": "fill", "fill": QColor(59, 66, 82, 200), "stroke": QColor(76, 86, 106, 140), "stroke_w": 0.5, "building": True},
        {"match": "road", "type": "line", "stroke": QColor(76, 86, 106), "stroke_w": 1.2,
         "class_widths": {"motorway": 3.5, "trunk": 3.0, "primary": 2.4, "secondary": 1.8, "tertiary": 1.3, "street": 1.0, "service": 0.6, "path": 0.4},
         "class_colors": {"motorway": QColor(191, 97, 106), "trunk": QColor(208, 135, 112), "primary": QColor(163, 190, 140)}},
        {"match": "bridge", "type": "line", "stroke": QColor(136, 192, 208), "stroke_w": 2.0},
        {"match": "waterway", "type": "line", "stroke": QColor(94, 129, 172, 150), "stroke_w": 1.2},
        {"match": "admin", "type": "line", "stroke": QColor(180, 142, 173, 120), "stroke_w": 1.0, "dash": [6, 3]},
        {"match": "transit", "type": "line", "stroke": QColor(76, 86, 106, 80), "stroke_w": 0.8},
        {"match": "place_label", "type": "label", "color": QColor(216, 222, 233), "font_size": 11, "halo": QColor(46, 52, 64, 220)},
        {"match": "poi_label", "type": "label", "color": QColor(136, 192, 208, 180), "font_size": 8, "halo": QColor(46, 52, 64, 180)},
    ])
    return {s.name: s for s in [dark, streets, satellite, outdoors, nav_night, light,
                                 cyberpunk, blueprint, sepia, nord]}

ALL_STYLES = _make_styles()

LAYER_MATCH = {
    "water": ["water"], "waterway": ["waterway"],
    "land": ["land", "landcover"], "landuse": ["landuse"],
    "landuse_overlay": ["landuse_overlay", "landcover"],
    "hillshade": ["hillshade"], "building": ["building"],
    "road": ["road", "transportation"], "bridge": ["bridge"],
    "tunnel": ["tunnel"], "transit": ["transit_stop", "transit"],
    "admin": ["admin", "boundary"], "contour": ["contour"],
    "place_label": ["place_label"], "poi_label": ["poi_label"],
}

def _match_layer(layer_name, match):
    candidates = LAYER_MATCH.get(match, [match])
    ln = layer_name.lower()
    if ln in candidates: return True
    for c in candidates:
        if ln.startswith(c + "_") or ln.endswith("_" + c): return True
    return False


# ===========================================================================
#  GPU Geometry Tessellation — replaces QPainter tile rendering
# ===========================================================================
#
#  Instead of rendering each tile to a QImage via QPainter, we tessellate
#  MVT geometry into GPU vertex buffers. Each tile produces:
#    - fill_verts: triangulated polygons (x, y, r, g, b, a)
#    - line_verts: thick-line quads (x, y, r, g, b, a)
#    - label_candidates: same as before
#    - building data: same as before
#
#  Coordinates are in tile-local space (0..1), transformed to world space
#  by the tile_offset + tile_scale uniforms at draw time.
#  This means geometry is zoom-independent and can be reused across
#  fractional zoom changes.

def _get_feature_class(feat):
    return str(feat.properties.get("class", feat.properties.get("type", "")))


try:
    import mapbox_earcut as _earcut
    _HAS_EARCUT = True
except ImportError:
    _HAS_EARCUT = False


def _triangulate_polygon(rings_list):
    """Triangulate a polygon (outer ring + optional holes) into triangles.
    Uses mapbox_earcut for correct handling of concave shapes and holes.
    Falls back to fan triangulation if earcut unavailable.
    Input: list of rings, each ring is list of (x,y) tuples.
           First ring is outer, rest are holes.
    Returns: list of (x, y) triples forming triangles."""
    if not rings_list or len(rings_list[0]) < 3:
        return []

    if _HAS_EARCUT:
        # Build flat coordinate array and ring-size array for earcut
        all_pts = []
        ring_sizes = []
        for ring in rings_list:
            pts = ring
            # Remove closing duplicate
            if len(pts) >= 4 and abs(pts[-1][0] - pts[0][0]) < 0.5 and abs(pts[-1][1] - pts[0][1]) < 0.5:
                pts = pts[:-1]
            if len(pts) < 3:
                if not all_pts:
                    return []  # outer ring too small
                continue  # skip degenerate hole
            all_pts.extend(pts)
            ring_sizes.append(len(pts))

        if not all_pts or not ring_sizes:
            return []

        coords = np.array(all_pts, dtype=np.float64)
        rings_arr = np.array(ring_sizes, dtype=np.int32)
        try:
            indices = _earcut.triangulate_float64(coords, rings_arr)
        except Exception:
            # Fallback to fan for this polygon
            return _fan_triangulate_ring(rings_list[0])

        result = []
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
            if i0 < len(all_pts) and i1 < len(all_pts) and i2 < len(all_pts):
                result.extend([all_pts[i0], all_pts[i1], all_pts[i2]])
        return result
    else:
        return _fan_triangulate_ring(rings_list[0])


def _fan_triangulate_ring(ring):
    """Ear-clipping triangulation fallback for when mapbox_earcut is unavailable.
    Handles concave polygons correctly (unlike simple fan triangulation)."""
    pts = list(ring)
    if len(pts) >= 4 and abs(pts[-1][0] - pts[0][0]) < 0.5 and abs(pts[-1][1] - pts[0][1]) < 0.5:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    # For very small polygons, fan is fine
    if len(pts) <= 4:
        tris = []
        p0 = pts[0]
        for i in range(1, len(pts) - 1):
            tris.extend([p0, pts[i], pts[i+1]])
        return tris

    # Basic ear-clipping for concave polygons
    # Determine overall winding (negative area = CW in Y-down)
    area = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _point_in_triangle(p, a, b, c):
        d1 = _cross(p, a, b)
        d2 = _cross(p, b, c)
        d3 = _cross(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    indices = list(range(len(pts)))
    tris = []
    max_iter = len(pts) * len(pts)  # safety limit
    iterations = 0
    while len(indices) > 2 and iterations < max_iter:
        iterations += 1
        found_ear = False
        n = len(indices)
        for i in range(n):
            prev_i = indices[(i - 1) % n]
            curr_i = indices[i]
            next_i = indices[(i + 1) % n]
            a, b, c = pts[prev_i], pts[curr_i], pts[next_i]
            cross = _cross(a, b, c)
            # Check if this is a convex vertex (ear candidate)
            # For CW winding (area < 0), convex ears have cross < 0
            # For CCW winding (area > 0), convex ears have cross > 0
            if area < 0:
                is_convex = cross < 0
            else:
                is_convex = cross > 0
            if not is_convex:
                continue
            # Check no other vertex is inside this triangle
            is_ear = True
            for j in range(n):
                if j in (i, (i - 1) % n, (i + 1) % n):
                    continue
                if _point_in_triangle(pts[indices[j]], a, b, c):
                    is_ear = False
                    break
            if is_ear:
                tris.extend([a, b, c])
                indices.pop(i)
                found_ear = True
                break
        if not found_ear:
            # Degenerate polygon — fall back to simple fan for remainder
            if len(indices) >= 3:
                p0 = pts[indices[0]]
                for i in range(1, len(indices) - 1):
                    tris.extend([p0, pts[indices[i]], pts[indices[i+1]]])
            break
    return tris


def _thicken_line(ring, half_width):
    """Convert a polyline to a triangle-strip quad ribbon with miter-clamped joins.
    Returns list of (x, y) vertices (6 per segment = 2 triangles)."""
    if len(ring) < 2:
        return []
    pts = ring
    n = len(pts)
    # Remove duplicate consecutive points
    cleaned = [pts[0]]
    for i in range(1, n):
        dx = pts[i][0] - cleaned[-1][0]
        dy = pts[i][1] - cleaned[-1][1]
        if dx*dx + dy*dy > 1e-8:
            cleaned.append(pts[i])
    pts = cleaned
    n = len(pts)
    if n < 2:
        return []

    # Compute per-segment direction normals
    seg_normals = []
    for i in range(n - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        l = math.sqrt(dx*dx + dy*dy)
        if l > 1e-8:
            seg_normals.append((-dy/l, dx/l))
        else:
            seg_normals.append((0, 1))

    # Compute per-vertex normals (average of adjacent segments, miter clamped)
    normals = []
    for i in range(n):
        if i == 0:
            nx, ny = seg_normals[0]
        elif i == n - 1:
            nx, ny = seg_normals[-1]
        else:
            # Average the two adjacent segment normals
            nx0, ny0 = seg_normals[i-1]
            nx1, ny1 = seg_normals[i]
            nx = nx0 + nx1; ny = ny0 + ny1
            l = math.sqrt(nx*nx + ny*ny)
            if l > 1e-8:
                nx /= l; ny /= l
                # Miter limit: clamp offset to 2x the half-width
                dot = nx * seg_normals[i][0] + ny * seg_normals[i][1]
                if dot > 0.1:
                    miter_scale = 1.0 / dot
                    if miter_scale > 2.0:
                        miter_scale = 2.0
                    nx *= miter_scale; ny *= miter_scale
                # else: bevel-like clamping
            else:
                nx, ny = seg_normals[i]
        normals.append((nx * half_width, ny * half_width))

    verts = []
    for i in range(n - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i+1]
        nx0, ny0 = normals[i]; nx1, ny1 = normals[i+1]
        # Two triangles forming a quad segment
        verts.extend([
            (x0 + nx0, y0 + ny0), (x0 - nx0, y0 - ny0), (x1 + nx1, y1 + ny1),
            (x0 - nx0, y0 - ny0), (x1 - nx1, y1 - ny1), (x1 + nx1, y1 + ny1),
        ])
    return verts


def _tessellate_tile(tile, style, z):
    """Tessellate MVT tile into GPU-ready vertex arrays.
    Returns (fill_data, line_data, label_candidates, buildings).
    fill_data/line_data are numpy float32 arrays with columns: x, y, r, g, b, a
    Coordinates are in normalised tile space (0..extent mapped to 0..1)."""
    fill_verts = []  # (x, y, r, g, b, a)
    line_verts = []

    for ls in style.layers:
        match_name = ls["match"]; ltype = ls["type"]
        if ls.get("building", False) or ltype == "label":
            continue
        matched = [layer for ln, layer in tile.layers.items() if _match_layer(ln, match_name)]
        if not matched:
            continue

        for layer in matched:
            ext = layer.extent or MVT_EXTENT
            inv_ext = 1.0 / ext

            for feat in layer.features:
                fclass = _get_feature_class(feat)
                if "subtype" in ls:
                    sub = ls["subtype"]
                    if sub == "tunnel" and feat.properties.get("brunnel") != "tunnel" and "tunnel" not in layer.name.lower():
                        continue
                    if sub == "rail" and fclass not in ("rail", "railway", "transit"):
                        continue

                rings = _get_rings(feat, ext)
                if not rings:
                    continue

                if ltype == "fill" and feat.geom_type == 3:
                    fill_c = ls["fill"]
                    r = fill_c.red() / 255.0
                    g = fill_c.green() / 255.0
                    b = fill_c.blue() / 255.0
                    a = fill_c.alpha() / 255.0
                    # Skip nearly-invisible fills (alpha < 5%) to avoid
                    # artifacts from mis-tessellated translucent polygons
                    if a < 0.02:
                        continue
                    # MVT polygons: outer rings are CW, holes are CCW
                    # In MVT's Y-down coordinate system, the standard
                    # shoelace formula yields NEGATIVE area for CW (outer)
                    # rings and POSITIVE for CCW (hole) rings.
                    poly_groups = []
                    current_group = None
                    for ring in rings:
                        if len(ring) < 3:
                            continue
                        # Compute signed area (shoelace) to determine winding
                        area = 0.0
                        for j in range(len(ring) - 1):
                            area += ring[j][0] * ring[j+1][1] - ring[j+1][0] * ring[j][1]
                        # Skip degenerate slivers (near-zero area)
                        if abs(area) < 1.0:
                            continue
                        if area <= 0:
                            # Negative (or zero) area = CW in Y-down = outer ring
                            current_group = [ring]
                            poly_groups.append(current_group)
                        else:
                            # Positive area = CCW in Y-down = hole
                            if current_group is not None:
                                current_group.append(ring)
                            # else: orphan hole, skip
                    for group in poly_groups:
                        tris = _triangulate_polygon(group)
                        for px, py in tris:
                            fill_verts.append((px * inv_ext, py * inv_ext, r, g, b, a))
                    # Fallback: if winding detection found no outer rings
                    # but we have rings, treat the first ring as outer.
                    if not poly_groups and rings:
                        valid_rings = [rr for rr in rings if len(rr) >= 3]
                        if valid_rings:
                            tris = _triangulate_polygon(valid_rings)
                            for px, py in tris:
                                fill_verts.append((px * inv_ext, py * inv_ext, r, g, b, a))

                elif ltype == "line" and feat.geom_type in (2, 3):
                    stroke_c = ls.get("class_colors", {}).get(fclass, ls["stroke"])
                    if isinstance(stroke_c, int) and stroke_c == 0:
                        continue  # Qt.NoPen
                    width = ls.get("class_widths", {}).get(fclass, ls.get("stroke_w", 1.0))
                    width *= max(0.5, min(2.0, z / 14.0))
                    # Line width in extent-space units.
                    # At draw time, tile_scale converts extent-space to screen pixels.
                    # tile_scale = tile_px, and positions are in [0..1] normalised.
                    # So 1 unit in normalised = tile_px screen pixels.
                    # We want width in screen pixels, so:
                    #   half_width_normalised = (width_px / 2) / tile_px_at_this_zoom
                    # But tile_px varies fractionally. Use the base tile size for
                    # the integer zoom level: TILE_SIZE (256).
                    hw = width * 0.5 / TILE_SIZE
                    rc = stroke_c.red() / 255.0
                    gc = stroke_c.green() / 255.0
                    bc = stroke_c.blue() / 255.0
                    ac = stroke_c.alpha() / 255.0
                    for ring in rings:
                        if len(ring) < 2:
                            continue
                        lv = _thicken_line(ring, hw * ext)  # scale to extent coords
                        for px, py in lv:
                            line_verts.append((px * inv_ext, py * inv_ext, rc, gc, bc, ac))

    # Build numpy arrays
    if fill_verts:
        fill_data = np.array(fill_verts, dtype='f4')
    else:
        fill_data = np.empty((0, 6), dtype='f4')

    if line_verts:
        line_data = np.array(line_verts, dtype='f4')
    else:
        line_data = np.empty((0, 6), dtype='f4')

    # Extract labels (same as v5)
    label_candidates = []
    for ls in style.layers:
        if ls["type"] != "label":
            continue
        match_name = ls["match"]
        for ln, layer in tile.layers.items():
            if not _match_layer(ln, match_name):
                continue
            ext = layer.extent or MVT_EXTENT
            s_norm = 1.0 / ext
            for feat in layer.features:
                if feat.geom_type != 1 or not feat.geometry:
                    continue
                name = feat.properties.get("name", feat.properties.get("name_en", ""))
                if not name or not isinstance(name, str):
                    continue
                name = str(name).strip()
                if not name:
                    continue
                rings = _get_rings(feat, ext)
                if not rings:
                    continue
                pt = rings[0][0]
                nx, ny = pt[0] * s_norm, pt[1] * s_norm
                if nx < -0.2 or nx > 1.2 or ny < -0.1 or ny > 1.1:
                    continue
                rank = feat.properties.get("filterrank", feat.properties.get("symbolrank", feat.properties.get("rank", 50)))
                try:
                    rank = int(rank)
                except:
                    rank = 50
                is_place = "place" in match_name
                priority = rank if is_place else rank + 100
                max_rank = max(2, min(20, int(z * 1.5)))
                if not is_place and rank > max_rank:
                    continue
                if is_place and rank > max_rank + 5:
                    continue
                label_candidates.append((priority, name, nx, ny, ls))

    # Extract buildings
    buildings = _extract_buildings(tile)

    return fill_data, line_data, label_candidates, buildings


def _extract_buildings(tile):
    """Extract building geometry from MVT tile."""
    buildings = []
    for ln, layer in tile.layers.items():
        if "building" not in ln.lower(): continue
        ext = layer.extent or MVT_EXTENT
        for feat in layer.features:
            if feat.geom_type != 3: continue
            rings = _get_rings(feat, ext)
            if not rings: continue
            h_val = feat.properties.get("height", feat.properties.get("render_height", feat.properties.get("extrude", 0)))
            try: h = float(h_val) if h_val else 0
            except: h = 0
            min_h_val = feat.properties.get("min_height", 0)
            try: min_h = float(min_h_val) if min_h_val else 0
            except: min_h = 0
            buildings.append({"rings": rings, "height": h, "min_height": min_h, "extent": ext})
    return buildings


def _mvt_tile_url(token, tileset, z, x, y):
    n = 2 ** z; x = x % n
    if y < 0 or y >= n: return ""
    return f"https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}.mvt?access_token={token}"


def _terrain_rgb_url(token, z, x, y):
    """Mapbox terrain-RGB v1 raster DEM tile URL (256px PNG)."""
    n = 2 ** z; x = x % n
    if y < 0 or y >= n: return ""
    return (f"https://api.mapbox.com/v4/mapbox.terrain-rgb/"
            f"{z}/{x}/{y}.pngraw?access_token={token}")


def _decode_terrain_rgb(png_bytes):
    """Decode Mapbox terrain-RGB PNG into a 2D numpy elevation array (meters).
    Formula: height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
    Returns float32 array of shape (H, W)."""
    try:
        img = QImage()
        img.loadFromData(png_bytes)
        if img.isNull():
            return None
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        elevation = -10000.0 + (r * 65536.0 + g * 256.0 + b) * 0.1
        return elevation
    except Exception:
        return None


def _sample_terrain_bilinear(terrain_cache, wx, wy, zoom):
    """Bilinear-interpolated terrain elevation (meters) at world-pixel coords.
    Returns 0.0 if no terrain data available."""
    if not terrain_cache:
        return 0.0
    z_int = max(0, min(15, int(math.floor(zoom + 0.5))))
    tile_px = TILE_SIZE * (2.0 ** (zoom - z_int))
    tx = int(math.floor(wx / tile_px))
    ty = int(math.floor(wy / tile_px))
    n = 2 ** z_int
    tx = tx % n
    tkey = (z_int, tx, ty)
    elev = terrain_cache.get(tkey)
    if elev is None:
        return 0.0
    frac_x = (wx / tile_px) - math.floor(wx / tile_px)
    frac_y = (wy / tile_px) - math.floor(wy / tile_px)
    h, w = elev.shape
    fx = frac_x * (w - 1)
    fy = frac_y * (h - 1)
    x0 = int(fx); y0 = int(fy)
    x1 = min(x0 + 1, w - 1); y1 = min(y0 + 1, h - 1)
    dx = fx - x0; dy = fy - y0
    e00 = elev[y0, x0]; e10 = elev[y0, x1]
    e01 = elev[y1, x0]; e11 = elev[y1, x1]
    return float((e00 * (1 - dx) * (1 - dy) + e10 * dx * (1 - dy) +
                   e01 * (1 - dx) * dy + e11 * dx * dy))


def _terrain_elev_wp(terrain_cache, wx, wy, zoom, mpp):
    """Get terrain elevation in world-pixels at (wx, wy)."""
    elev_m = _sample_terrain_bilinear(terrain_cache, wx, wy, zoom)
    return elev_m / max(mpp, 0.0001)


# ===========================================================================
#  Thread-safe signals
# ===========================================================================

class _TileSignals(QObject):
    tile_ready = Signal(tuple, object, object, list, list)
    terrain_ready = Signal(tuple, object)  # (z,x,y), elevation_array
    # key, fill_data(np), line_data(np), labels, buildings


# ===========================================================================
#  Optimization 4: VBO Pool (Ring Buffer) — reuse VBOs instead of alloc/free
# ===========================================================================

class VBOPool:
    """Pre-allocated pool of fixed-size VBOs.  write() overwrites an existing
    buffer instead of ctx.buffer() + old.release(), avoiding OpenGL driver
    overhead and memory fragmentation.  Unused slots are tracked in a FIFO."""

    def __init__(self, ctx, pool_size=24, slot_bytes=4 * 1024 * 1024):
        self._ctx = ctx
        self._pool_size = pool_size
        self._slot_bytes = slot_bytes
        self._free: deque = deque()
        self._all: list = []
        # Pre-allocate the pool
        for _ in range(pool_size):
            buf = ctx.buffer(reserve=slot_bytes)
            self._free.append(buf)
            self._all.append(buf)

    def acquire(self, data_bytes: bytes):
        """Return a VBO filled with *data_bytes*.
        If the data exceeds the fixed slot size, create a one-off buffer
        (still tracked for release).  If the pool is exhausted, also create
        a one-off.  Callers should eventually call release(buf)."""
        nbytes = len(data_bytes)
        if nbytes == 0:
            return None
        if nbytes <= self._slot_bytes and self._free:
            buf = self._free.popleft()
            # Overwrite existing buffer — much faster than alloc+free
            buf.write(data_bytes)
            return buf
        # Oversized or pool exhausted — allocate a new buffer
        buf = self._ctx.buffer(data_bytes)
        self._all.append(buf)
        return buf

    def release(self, buf):
        """Return a buffer to the pool (or truly release if oversized)."""
        if buf is None:
            return
        if buf.size == self._slot_bytes:
            self._free.append(buf)
        else:
            try:
                buf.release()
            except Exception:
                pass
            try:
                self._all.remove(buf)
            except ValueError:
                pass

    def release_all(self):
        for buf in self._all:
            try:
                buf.release()
            except Exception:
                pass
        self._all.clear()
        self._free.clear()


# ===========================================================================
#  Optimization 2: Time-Sliced GPU Upload Queue
# ===========================================================================

class GPUUploadQueue:
    """Queues tile geometry for time-sliced upload to the GPU.
    Instead of uploading fill + line + building VBOs in a single frame,
    the main thread pops one upload task per frame, keeping frame time
    under ~16ms.

    Each entry is a dict:
      { 'gkey', 'key', 'fill_data', 'line_data', 'labels', 'bldgs' }
    """

    def __init__(self):
        self._queue: deque = deque()
        self._pending_keys: set = set()

    def enqueue(self, gkey, key, fill_data, line_data, labels, bldgs):
        if gkey in self._pending_keys:
            # Replace: remove older entry for same gkey
            self._queue = deque(e for e in self._queue if e['gkey'] != gkey)
        self._pending_keys.add(gkey)
        self._queue.append({
            'gkey': gkey, 'key': key,
            'fill_data': fill_data, 'line_data': line_data,
            'labels': labels, 'bldgs': bldgs,
        })

    def pop(self):
        """Return the next upload task, or None."""
        if not self._queue:
            return None
        entry = self._queue.popleft()
        self._pending_keys.discard(entry['gkey'])
        return entry

    def __len__(self):
        return len(self._queue)

    @property
    def pending(self):
        return bool(self._queue)


# ===========================================================================
#  Optimization 5: High-Frequency Physics Thread (120 Hz)
# ===========================================================================

class PhysicsThread(threading.Thread):
    """Runs immersive-mode camera and car physics at 120 Hz on a
    dedicated thread.  Only the final (wx, wy, yaw, speed) coordinates
    are shared with the main/render thread via a Lock, keeping camera
    movement buttery smooth even during tile decode stalls.

    The main _tick() reads the latest state snapshot instead of running
    physics inline."""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._running = True
        self._active = False  # set True when immersive mode is entered

        # ---- Shared state (read by main thread) ----
        self.walk_wx = 0.0
        self.walk_wy = 0.0
        self.walk_yaw = 0.0
        self.walk_pitch = 0.0
        self.walk_eye_h = 5.0
        self.car_wx = 0.0
        self.car_wy = 0.0
        self.car_yaw = 0.0
        self.car_speed = 0.0
        self.car_steer_angle = 0.0

        # ---- Input state (written by main thread) ----
        self.car_mode = False
        self.keys_held: set = set()
        self.car_throttle = 0.0
        self.car_brake = 0.0
        self.car_steer = 0.0
        self.move_speed = 3.0
        self.car_max_speed = 0.0
        self.car_accel_rate = 0.0
        self.car_mpp = 0.001

        # Cached collision callback (set by main thread)
        self._collision_check_fn = None

    def snapshot(self):
        """Return a copy of the current physics state (thread-safe)."""
        with self._lock:
            return {
                'walk_wx': self.walk_wx, 'walk_wy': self.walk_wy,
                'walk_yaw': self.walk_yaw, 'walk_pitch': self.walk_pitch,
                'walk_eye_h': self.walk_eye_h,
                'car_wx': self.car_wx, 'car_wy': self.car_wy,
                'car_yaw': self.car_yaw, 'car_speed': self.car_speed,
                'car_steer_angle': self.car_steer_angle,
            }

    def set_walk_pos(self, wx, wy, yaw, pitch, eye_h):
        with self._lock:
            self.walk_wx = wx; self.walk_wy = wy
            self.walk_yaw = yaw; self.walk_pitch = pitch
            self.walk_eye_h = eye_h

    def set_car_pos(self, wx, wy, yaw, speed, steer_angle):
        with self._lock:
            self.car_wx = wx; self.car_wy = wy
            self.car_yaw = yaw; self.car_speed = speed
            self.car_steer_angle = steer_angle

    def run(self):
        last_t = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            dt = min(now - last_t, 1.0 / 20.0)
            last_t = now

            if self._active:
                with self._lock:
                    if self.car_mode:
                        self._step_car(dt)
                    else:
                        self._step_walk(dt)

            # Sleep to target ~120 Hz
            elapsed = time.perf_counter() - now
            sleep_t = max(0.0, (1.0 / 120.0) - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _step_walk(self, dt):
        """Advance walk-mode physics (called with lock held)."""
        keys = self.keys_held
        if not keys:
            return
        yaw_rad = math.radians(self.walk_yaw)
        fwd_x = math.sin(yaw_rad)
        fwd_y = -math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_y = math.sin(yaw_rad)
        speed = self.move_speed * dt
        dx = dy = 0.0
        if Qt.Key_W in keys:
            dx += fwd_x * speed; dy += fwd_y * speed
        if Qt.Key_S in keys:
            dx -= fwd_x * speed; dy -= fwd_y * speed
        if Qt.Key_A in keys:
            dx -= right_x * speed; dy -= right_y * speed
        if Qt.Key_D in keys:
            dx += right_x * speed; dy += right_y * speed
        if Qt.Key_Space in keys:
            self.walk_eye_h += speed * 0.5
        if Qt.Key_Shift in keys:
            self.walk_eye_h = max(1.0, self.walk_eye_h - speed * 0.5)
        self.walk_wx += dx
        self.walk_wy += dy

    def _step_car(self, dt):
        """Advance car physics (called with lock held)."""
        mpp = self.car_mpp
        accel = self.car_accel_rate

        if self.car_throttle > 0:
            self.car_speed = min(self.car_max_speed,
                                 self.car_speed + accel * dt * self.car_throttle)
        if self.car_brake > 0:
            self.car_speed = max(0, self.car_speed - accel * 3.0 * dt * self.car_brake)
        if self.car_throttle == 0 and self.car_brake == 0:
            self.car_speed = max(0, self.car_speed - accel * 0.25 * dt)

        # Smoothed steering
        speed_mps = self.car_speed * mpp
        if abs(self.car_steer) > 0.01 and self.car_speed > 0.001:
            speed_factor = min(1.0, 8.0 / max(speed_mps, 1.0))
            target_steer_rate = self.car_steer * 140.0 * speed_factor
        else:
            target_steer_rate = 0.0
        steer_smooth = 1.0 - math.exp(-12.0 * dt)
        self.car_steer_angle += (target_steer_rate - self.car_steer_angle) * steer_smooth
        self.car_yaw = (self.car_yaw + self.car_steer_angle * dt) % 360.0

        yaw_rad = math.radians(self.car_yaw)
        displacement = self.car_speed * dt
        self.car_wx += math.sin(yaw_rad) * displacement
        self.car_wy -= math.cos(yaw_rad) * displacement

    def stop(self):
        self._running = False


# ===========================================================================
#  GL Shaders
# ===========================================================================

# --- Geometry shader: renders tessellated fills and lines directly ---
GEO_VERT = """
#version 330
uniform mat4 view_proj;
uniform vec2 tile_offset;
uniform float tile_scale;
in vec2 in_position;
in vec4 in_color;
out vec4 v_color;
void main() {
    vec2 wp = tile_offset + in_position * tile_scale;
    gl_Position = view_proj * vec4(wp, 0.0, 1.0);
    v_color = in_color;
}
"""

GEO_FRAG = """
#version 330
in vec4 v_color;
out vec4 frag;
void main() {
    frag = v_color;
}
"""

# --- Fallback raster tile shader (for parent/child zoom fallback) ---
TILE_VERT = """
#version 330
uniform mat4 view_proj;
uniform vec2 tile_offset;
uniform float tile_scale;
uniform vec2 uv_offset;
uniform vec2 uv_scale;
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    vec2 wp = tile_offset + in_position * tile_scale;
    gl_Position = view_proj * vec4(wp, 0.0, 1.0);
    v_uv = uv_offset + in_uv * uv_scale;
}
"""

TILE_FRAG = """
#version 330
uniform sampler2D tile_tex;
in vec2 v_uv;
out vec4 frag;
void main() {
    frag = texture(tile_tex, v_uv);
}
"""

BLDG_VERT = """
#version 330
uniform mat4 mvp;
uniform float height_scale;
in vec2 in_position;
in float in_height;
in float in_norm_height;
in vec3 in_normal;
in vec4 in_color;
out vec4 v_color;
out float v_norm_h;
out vec3 v_normal;
out float v_height;
void main() {
    vec4 pos = mvp * vec4(in_position, 0.0, 1.0);
    pos.y += in_height * height_scale;
    pos.z = -in_height * height_scale * 0.001;
    gl_Position = pos;
    v_color = in_color;
    v_norm_h = in_norm_height;
    v_normal = in_normal;
    v_height = in_height;
}
"""

BLDG_FRAG = """
#version 330
uniform vec3 light_dir;
uniform float ambient_strength;
uniform float ao_strength;
uniform float top_highlight;
in vec4 v_color;
in float v_norm_h;
in vec3 v_normal;
in float v_height;
out vec4 frag;
void main() {
    vec3 base = v_color.rgb;
    vec3 n = normalize(v_normal);
    float ndotl = dot(n, light_dir);
    float diffuse = clamp(ndotl * 0.5 + 0.5, 0.0, 1.0);
    float light = ambient_strength + (1.0 - ambient_strength) * diffuse * 0.7;
    float ao = 1.0 - ao_strength * (1.0 - v_norm_h) * (1.0 - v_norm_h);
    float edge_glow = smoothstep(0.85, 1.0, v_norm_h) * top_highlight;
    vec3 lit = base * light * ao + edge_glow * vec3(1.0, 1.0, 1.0);
    if (n.z > 0.5) {
        float roof_light = ambient_strength + 0.45;
        float roof_ndotl = dot(n, light_dir) * 0.15 + 0.85;
        lit = base * roof_light * roof_ndotl + vec3(0.035, 0.03, 0.025);
        float h_factor = clamp(v_height / 120.0, 0.0, 1.0);
        lit += vec3(0.01, 0.015, 0.025) * h_factor;
    } else {
        float h_factor = clamp(v_height / 150.0, 0.0, 1.0);
        lit += vec3(-0.005, 0.0, 0.012) * h_factor;
    }
    frag = vec4(clamp(lit, 0.0, 1.0), v_color.a);
}
"""

SHADOW_VERT = """
#version 330
uniform mat4 mvp;
uniform vec2 shadow_offset;
in vec2 in_position;
in float in_height;
out float v_alpha;
void main() {
    vec2 shadow_pos = in_position + shadow_offset * in_height;
    vec4 pos = mvp * vec4(shadow_pos, 0.0, 1.0);
    gl_Position = pos;
    gl_Position.z = 0.5;
    v_alpha = clamp(in_height / 80.0, 0.05, 0.35);
}
"""

SHADOW_FRAG = """
#version 330
in float v_alpha;
out vec4 frag;
void main() {
    frag = vec4(0.0, 0.0, 0.0, v_alpha);
}
"""

ROUTE_VERT = """
#version 330
uniform mat4 view_proj;
in vec2 in_position;
in float in_side;
in float in_progress;
out float v_progress;
out float v_side;
void main() {
    gl_Position = view_proj * vec4(in_position, 0.0, 1.0);
    gl_Position.z = -0.1;
    v_progress = in_progress;
    v_side = in_side;
}
"""

ROUTE_FRAG = """
#version 330
uniform vec4 route_color;
uniform float time;
uniform float dash_scale;
in float v_progress;
in float v_side;
out vec4 frag;
void main() {
    float dash = smoothstep(0.45, 0.5, fract(v_progress * dash_scale - time * 0.8));
    float edge_fade = 1.0 - smoothstep(0.5, 1.0, abs(v_side));
    float glow = exp(-4.0 * v_side * v_side);
    vec4 col = route_color;
    col.a *= mix(0.3, 1.0, dash) * edge_fade;
    col.rgb += vec3(0.15) * glow * dash;
    frag = col;
}
"""

HEAT_VERT = """
#version 330
uniform mat4 view_proj;
uniform float point_radius;
in vec2 in_position;
in vec2 in_quad;
in float in_weight;
out vec2 v_quad;
out float v_weight;
void main() {
    vec2 world_pos = in_position + in_quad * point_radius;
    gl_Position = view_proj * vec4(world_pos, 0.0, 1.0);
    gl_Position.z = -0.05;
    v_quad = in_quad;
    v_weight = in_weight;
}
"""

HEAT_FRAG = """
#version 330
uniform float intensity;
uniform float min_heat;
in vec2 v_quad;
in float v_weight;
out vec4 frag;
void main() {
    float dist = length(v_quad);
    if (dist > 1.0) discard;
    float d = 1.0 - dist;
    float heat = d * d * v_weight * intensity;
    if (heat < min_heat) discard;
    // Remap remaining range to 0..1 for smooth color gradient
    float t = clamp((heat - min_heat) / max(1.0 - min_heat, 0.001), 0.0, 1.0);
    vec3 col;
    if (t < 0.25) col = mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.8, 1.0), t * 4.0);
    else if (t < 0.5) col = mix(vec3(0.0, 0.8, 1.0), vec3(0.0, 1.0, 0.2), (t - 0.25) * 4.0);
    else if (t < 0.75) col = mix(vec3(0.0, 1.0, 0.2), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
    else col = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.1, 0.0), (t - 0.75) * 4.0);
    float alpha = clamp(t * 1.5, 0.0, 0.7);
    frag = vec4(col, alpha);
}
"""


# ===========================================================================
#  Immersive Mode (First-Person) Shaders
# ===========================================================================

# --- First-person building shader: true 3D perspective with world-space coords ---
FP_BLDG_VERT = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
uniform sampler2D u_terrain;
uniform vec4 u_terrain_bounds;
uniform float u_terrain_scale;
uniform float u_terrain_active;
in vec3 in_pos;      // x,y = world px, z = height in world px
in vec3 in_normal;
in vec4 in_color;
out vec4 v_color;
out vec3 v_normal;
out vec3 v_world_pos;
out float v_height;
void main() {
    vec3 p = in_pos;
    float terrain_z = 0.0;
    if (u_terrain_active > 0.5) {
        vec2 uv = (p.xy - u_terrain_bounds.xy) / (u_terrain_bounds.zw - u_terrain_bounds.xy);
        uv = clamp(uv, 0.0, 1.0);
        terrain_z = texture(u_terrain, uv).r * u_terrain_scale;
    }
    vec3 world = vec3(-p.x, p.z + terrain_z, -p.y);
    vec3 rel = world - u_cam_pos;
    gl_Position = u_proj * u_view * vec4(rel, 1.0);
    v_color = in_color;
    v_normal = vec3(-in_normal.x, in_normal.z, -in_normal.y);
    v_world_pos = rel;
    v_height = in_pos.z;
}
"""

FP_BLDG_FRAG = """
#version 330
uniform vec3 u_light_dir;
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_time;
uniform float u_win_glow;      // window glow intensity  (0..2, default 1.0)
uniform float u_ao_strength;   // ambient occlusion      (0..2, default 1.0)
uniform float u_wall_opacity;  // overall wall opacity    (0..1, default 1.0)
uniform vec3  u_bld_tint;      // building color tint     (default 1,1,1)
in vec4 v_color;
in vec3 v_normal;
in vec3 v_world_pos;
in float v_height;
out vec4 frag;
void main() {
    vec3 base = v_color.rgb * u_bld_tint;
    vec3 n = normalize(v_normal);
    // Wrap diffuse
    float ndotl = dot(n, u_light_dir);
    float diffuse = clamp(ndotl * 0.5 + 0.5, 0.0, 1.0);
    float light = 0.35 + 0.65 * diffuse;
    vec3 lit = base * light;
    // Roof brighten (only truly horizontal faces)
    if (n.y > 0.8) {
        lit += vec3(0.06, 0.05, 0.04);
    }
    // Ambient occlusion at ground level
    float ao = clamp(v_height * 0.1, 0.0, 1.0);
    float ao_factor = mix(1.0, 0.6 + 0.4 * ao, u_ao_strength);
    lit *= ao_factor;
    // Window glow — strictly on vertical walls only (n.y close to 0)
    if (abs(n.y) < 0.15 && v_height > 3.0 && u_win_glow > 0.01) {
        float wx = fract(v_world_pos.x * 0.08);
        float wy = fract(v_height * 0.12);
        float wz = fract(v_world_pos.z * 0.08);
        // Pick the wall axis — use the dominant normal component
        float wall_frac;
        if (abs(n.x) > abs(n.z)) {
            wall_frac = wz;  // wall faces X, use Z for windows
        } else {
            wall_frac = wx;  // wall faces Z, use X for windows
        }
        float win = step(0.2, wall_frac) * step(wall_frac, 0.42)
                   * step(0.22, wy) * step(wy, 0.68);
        // Hash for random on/off per window cell
        float cell_id = floor(v_world_pos.x * 0.08) * 127.1
                       + floor(v_world_pos.z * 0.08) * 113.3
                       + floor(v_height * 0.12) * 311.7;
        float h1 = fract(sin(cell_id) * 43758.5);
        float lit_chance = step(0.4, h1);
        // Warm / cool color variation
        float warm = step(0.5, fract(h1 * 7.3));
        vec3 win_color = mix(vec3(0.45, 0.6, 0.95), vec3(1.0, 0.85, 0.5), warm);
        float glow = win * lit_chance * 0.45 * u_win_glow;
        // Subtle flicker
        float flicker = 0.88 + 0.12 * sin(u_time * 0.4 + h1 * 30.0);
        lit += win_color * glow * flicker;
    }
    // Distance fog
    float dist = length(v_world_pos);
    float fog_t = clamp((dist - u_fog_near) / max(u_fog_far - u_fog_near, 1.0), 0.0, 1.0);
    fog_t = fog_t * fog_t;
    lit = mix(lit, u_fog_color, fog_t);
    float alpha = v_color.a * u_wall_opacity;
    frag = vec4(clamp(lit, 0.0, 1.0), alpha);
}
"""

# --- First-person ground plane shader ---
FP_GROUND_VERT = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
uniform sampler2D u_terrain;
uniform vec4 u_terrain_bounds;  // (min_wx, min_wy, max_wx, max_wy)
uniform float u_terrain_scale;  // meters-to-worldpx: 1.0/mpp
uniform float u_terrain_active; // 0 or 1
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
out vec3 v_world_pos;
void main() {
    vec3 p = in_pos;
    if (u_terrain_active > 0.5) {
        vec2 uv = (p.xy - u_terrain_bounds.xy) / (u_terrain_bounds.zw - u_terrain_bounds.xy);
        uv = clamp(uv, 0.0, 1.0);
        float elev_m = texture(u_terrain, uv).r;
        p.z += elev_m * u_terrain_scale;
    }
    vec3 world = vec3(-p.x, p.z, -p.y);
    vec3 rel = world - u_cam_pos;
    gl_Position = u_proj * u_view * vec4(rel, 1.0);
    v_color = in_color;
    v_world_pos = rel;
}
"""

FP_GROUND_FRAG = """
#version 330
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
in vec4 v_color;
in vec3 v_world_pos;
out vec4 frag;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec3 col = v_color.rgb;
    // Very subtle terrain noise for organic feel
    vec2 uv = v_world_pos.xz * 0.005;
    float noise = hash(floor(uv)) * 0.03 - 0.015;
    col += noise;
    // Fog
    float dist = length(v_world_pos);
    float fog_t = clamp((dist - u_fog_near) / max(u_fog_far - u_fog_near, 1.0), 0.0, 1.0);
    fog_t = fog_t * fog_t;
    col = mix(col, u_fog_color, fog_t);
    frag = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# --- First-person water shader (animated surface) ---
FP_WATER_VERT = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
uniform float u_time;
uniform sampler2D u_terrain;
uniform vec4 u_terrain_bounds;
uniform float u_terrain_scale;
uniform float u_terrain_active;
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
out vec3 v_world_pos;
out vec2 v_water_uv;
void main() {
    vec3 p = in_pos;
    float terrain_z = 0.0;
    if (u_terrain_active > 0.5) {
        vec2 uv = (p.xy - u_terrain_bounds.xy) / (u_terrain_bounds.zw - u_terrain_bounds.xy);
        uv = clamp(uv, 0.0, 1.0);
        terrain_z = texture(u_terrain, uv).r * u_terrain_scale;
    }
    // Gentle wave displacement
    float wave1 = sin(p.x * 0.08 + u_time * 0.7) * cos(p.y * 0.06 + u_time * 0.5) * 0.35;
    float wave2 = sin(p.x * 0.15 - u_time * 0.9 + 1.3) * cos(p.y * 0.12 + u_time * 0.3) * 0.15;
    float wave = wave1 + wave2;
    vec3 world = vec3(-p.x, p.z + terrain_z + wave + 0.08, -p.y);
    vec3 rel = world - u_cam_pos;
    gl_Position = u_proj * u_view * vec4(rel, 1.0);
    v_color = in_color;
    v_world_pos = rel;
    v_water_uv = p.xy * 0.02;
}
"""

FP_WATER_FRAG = """
#version 330
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_time;
uniform vec3 u_cam_pos;
in vec4 v_color;
in vec3 v_world_pos;
in vec2 v_water_uv;
out vec4 frag;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p *= 2.1;
        a *= 0.5;
    }
    return v;
}

void main() {
    vec3 base_col = v_color.rgb;

    // Animated water UVs
    vec2 uv1 = v_water_uv * 3.0 + vec2(u_time * 0.015, u_time * 0.008);
    vec2 uv2 = v_water_uv * 5.0 - vec2(u_time * 0.012, u_time * 0.018);

    float wave_pattern = fbm(uv1) * 0.6 + fbm(uv2) * 0.4;

    // Water color variation
    vec3 deep = base_col * 0.7;
    vec3 shallow = base_col * 1.3 + vec3(0.05, 0.08, 0.12);
    vec3 col = mix(deep, shallow, wave_pattern);

    // Specular highlights (fake sun reflection)
    float spec_wave = fbm(v_water_uv * 8.0 + vec2(u_time * 0.03, -u_time * 0.02));
    float spec = pow(max(0.0, spec_wave - 0.45) * 2.5, 3.0);
    col += vec3(0.6, 0.65, 0.7) * spec * 0.4;

    // Subtle caustic ripple pattern
    float caustic1 = fbm(v_water_uv * 12.0 + vec2(u_time * 0.04, u_time * 0.025));
    float caustic2 = fbm(v_water_uv * 12.0 - vec2(u_time * 0.03, u_time * 0.04) + 3.7);
    float caustic = pow(max(0.0, caustic1 + caustic2 - 0.8), 2.0);
    col += base_col * caustic * 0.15;

    // Fresnel-like edge darkening based on view distance
    float dist = length(v_world_pos);
    float near_bright = 1.0 - smoothstep(0.0, 80.0, dist) * 0.15;
    col *= near_bright;

    // Foam hints at edges (high wave gradient areas)
    float foam_noise = fbm(v_water_uv * 20.0 + vec2(u_time * 0.05, 0.0));
    float foam = smoothstep(0.62, 0.72, foam_noise) * 0.15;
    col += vec3(foam);

    // Fog
    float fog_t = clamp((dist - u_fog_near) / max(u_fog_far - u_fog_near, 1.0), 0.0, 1.0);
    fog_t = fog_t * fog_t;
    col = mix(col, u_fog_color, fog_t);

    frag = vec4(clamp(col, 0.0, 1.0), 0.92);
}
"""

# --- First-person road shader (slightly above ground) ---
# Basic version (also used for route lines — no lane markings)
FP_ROAD_VERT = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
uniform sampler2D u_terrain;
uniform vec4 u_terrain_bounds;
uniform float u_terrain_scale;
uniform float u_terrain_active;
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
out vec3 v_world_pos;
void main() {
    vec3 p = in_pos;
    float terrain_z = 0.0;
    if (u_terrain_active > 0.5) {
        vec2 uv = (p.xy - u_terrain_bounds.xy) / (u_terrain_bounds.zw - u_terrain_bounds.xy);
        uv = clamp(uv, 0.0, 1.0);
        terrain_z = texture(u_terrain, uv).r * u_terrain_scale;
    }
    vec3 world = vec3(-p.x, p.z + terrain_z + 0.15, -p.y);
    vec3 rel = world - u_cam_pos;
    gl_Position = u_proj * u_view * vec4(rel, 1.0);
    v_color = in_color;
    v_world_pos = rel;
}
"""

FP_ROAD_FRAG = """
#version 330
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
in vec4 v_color;
in vec3 v_world_pos;
out vec4 frag;
void main() {
    vec3 col = v_color.rgb;
    // Subtle asphalt texture variation
    float grain = fract(sin(dot(floor(v_world_pos.xz * 2.0), vec2(12.9, 78.2))) * 43758.5) * 0.02 - 0.01;
    col += grain;
    // Fog
    float dist = length(v_world_pos);
    float fog_t = clamp((dist - u_fog_near) / max(u_fog_far - u_fog_near, 1.0), 0.0, 1.0);
    fog_t = fog_t * fog_t;
    col = mix(col, u_fog_color, fog_t);
    frag = vec4(clamp(col, 0.0, 1.0), v_color.a);
}
"""

# --- Enhanced road shader with lane markings (navigation-SDK style) ---
FP_ROAD_NAV_VERT = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
uniform sampler2D u_terrain;
uniform vec4 u_terrain_bounds;
uniform float u_terrain_scale;
uniform float u_terrain_active;
in vec3 in_pos;      // x, y = world px, z = height offset
in vec4 in_color;    // road surface color
in vec2 in_uv;       // u = distance along road (world px), v = -1..+1 across road
out vec4 v_color;
out vec3 v_world_pos;
out vec2 v_uv;
void main() {
    vec3 p = in_pos;
    float terrain_z = 0.0;
    if (u_terrain_active > 0.5) {
        vec2 uv = (p.xy - u_terrain_bounds.xy) / (u_terrain_bounds.zw - u_terrain_bounds.xy);
        uv = clamp(uv, 0.0, 1.0);
        terrain_z = texture(u_terrain, uv).r * u_terrain_scale;
    }
    vec3 world = vec3(-p.x, p.z + terrain_z + 0.15, -p.y);
    vec3 rel = world - u_cam_pos;
    gl_Position = u_proj * u_view * vec4(rel, 1.0);
    v_color = in_color;
    v_world_pos = rel;
    v_uv = in_uv;
}
"""

FP_ROAD_NAV_FRAG = """
#version 330
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_road_marking_scale;
uniform float u_time;
in vec4 v_color;
in vec3 v_world_pos;
in vec2 v_uv;
out vec4 frag;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec3 base_col = v_color.rgb;
    float base_alpha = v_color.a;

    float v_abs = abs(v_uv.y);
    float u_dist = v_uv.x;

    float mscale = max(u_road_marking_scale, 0.001);
    vec3 col = base_col;

    // asphalt grain
    float grain = hash(floor(v_world_pos.xz * 3.0)) * 0.03 - 0.015;
    col += vec3(grain);

    // slight crown / center brighten
    float crown = 1.0 - smoothstep(0.0, 1.0, v_abs);
    col += vec3(0.03) * crown;

    // lane markings only for "main surface" roads encoded by alpha >= 0.98
    if (base_alpha >= 0.98) {
        vec3 white_col = vec3(0.96, 0.96, 0.92);
        vec3 yellow_col = vec3(0.96, 0.82, 0.18);

        // center dashed line
        float center_band = smoothstep(0.08, 0.015, v_abs);
        float dash_len = 3.5 / mscale;
        float gap_len  = 4.0 / mscale;
        float period = dash_len + gap_len;
        float dash_t = mod(u_dist + u_time * 0.0, period);
        float dash_on = 1.0 - smoothstep(dash_len - 0.5 / mscale, dash_len + 0.5 / mscale, dash_t);
        float center_mark = center_band * dash_on;

        // edge lines
        float edge_center = 0.86;
        float edge_width = 0.035;
        float edge_l = smoothstep(edge_center + edge_width, edge_center + edge_width * 0.25, v_abs) *
                       (1.0 - smoothstep(edge_center - edge_width * 0.25, edge_center - edge_width, v_abs));

        // shoulder darkening near outer edges
        float shoulder = smoothstep(0.88, 1.0, v_abs);
        col *= 1.0 - shoulder * 0.14;

        col = mix(col, yellow_col, center_mark * 0.95);
        col = mix(col, white_col, edge_l * 0.85);

        // faint lane separation hint for wider roads
        float lane_sep_pos = 0.33;
        float lane_sep = smoothstep(lane_sep_pos + 0.02, lane_sep_pos, v_abs) *
                         (1.0 - smoothstep(lane_sep_pos, lane_sep_pos - 0.02, v_abs));
        col += vec3(0.035) * lane_sep * 0.5;
    }

    float dist = length(v_world_pos);
    float fog_t = clamp((dist - u_fog_near) / max(u_fog_far - u_fog_near, 1.0), 0.0, 1.0);
    fog_t = fog_t * fog_t;
    col = mix(col, u_fog_color, fog_t);

    frag = vec4(clamp(col, 0.0, 1.0), clamp(base_alpha, 0.0, 1.0));
}
"""

# --- Sky dome shader for immersive mode ---
FP_SKY_VERT = """
#version 330
uniform mat4 u_view_rot;
uniform mat4 u_proj;
in vec3 in_pos;
out vec3 v_dir;
void main() {
    gl_Position = u_proj * u_view_rot * vec4(in_pos * 1000.0, 1.0);
    gl_Position.z = gl_Position.w - 0.001;  // push to far plane
    v_dir = in_pos;
}
"""

FP_SKY_FRAG = """
#version 330
uniform vec3 u_sky_top;
uniform vec3 u_sky_horizon;
uniform vec3 u_sun_dir;
in vec3 v_dir;
out vec4 frag;

// Simple hash for star field
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec3 d = normalize(v_dir);
    float t = clamp(d.y * 1.5 + 0.1, 0.0, 1.0);
    vec3 sky = mix(u_sky_horizon, u_sky_top, t);
    // Sun glow (multi-layer for realism)
    float sun_dot = max(dot(d, normalize(u_sun_dir)), 0.0);
    sky += vec3(1.0, 0.9, 0.65) * pow(sun_dot, 128.0) * 2.5;  // sun disk
    sky += vec3(1.0, 0.85, 0.6) * pow(sun_dot, 32.0) * 0.8;    // inner corona
    sky += vec3(0.8, 0.5, 0.2) * pow(sun_dot, 6.0) * 0.25;     // outer glow
    // Stars (only visible when looking up in dark conditions)
    if (d.y > 0.1 && u_sky_top.r < 0.15) {
        vec2 st = d.xz / (d.y + 0.001) * 40.0;
        float star = hash(floor(st));
        float brightness = step(0.985, star);
        float twinkle = 0.7 + 0.3 * sin(star * 100.0 + d.x * 50.0);
        sky += vec3(0.9, 0.9, 1.0) * brightness * twinkle * t;
    }
    // Below horizon darken
    if (d.y < 0.0) {
        sky = mix(sky, u_sky_horizon * 0.4, clamp(-d.y * 5.0, 0.0, 1.0));
    }
    frag = vec4(clamp(sky, 0.0, 1.0), 1.0);
}
"""

# --- Car model shader (3rd person vehicle) ---
FP_CAR_VERT = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
in vec4 in_color;
out vec4 v_color;
out vec3 v_normal;
out vec3 v_world_pos;
void main() {
    vec4 local4 = u_model * vec4(in_pos, 1.0);
    vec3 rel = vec3(-local4.x, local4.z, -local4.y);
    gl_Position = u_proj * u_view * vec4(rel, 1.0);
    v_color = in_color;
    vec3 mn = mat3(u_model) * in_normal;
    v_normal = vec3(-mn.x, mn.z, -mn.y);
    v_world_pos = rel;
}
"""

FP_CAR_FRAG = """
#version 330
uniform vec3 u_light_dir;
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
in vec4 v_color;
in vec3 v_normal;
in vec3 v_world_pos;
out vec4 frag;
void main() {
    vec3 base = v_color.rgb;
    vec3 n = normalize(v_normal);
    float ndotl = dot(n, u_light_dir);
    float diffuse = clamp(ndotl * 0.5 + 0.5, 0.0, 1.0);
    float light = 0.4 + 0.6 * diffuse;
    vec3 lit = base * light;
    // Car paint specular (metallic gloss)
    vec3 V = normalize(-v_world_pos);
    vec3 H = normalize(u_light_dir + V);
    float spec = pow(max(dot(n, H), 0.0), 64.0) * 0.8;
    // Fresnel edge highlight for car body
    float fresnel = pow(1.0 - max(dot(n, V), 0.0), 3.0) * 0.3;
    lit += vec3(1.0, 0.95, 0.9) * spec;
    lit += vec3(0.6, 0.7, 0.8) * fresnel;
    // Headlight self-glow (emit from headlight-colored verts)
    if (v_color.r > 0.8 && v_color.g > 0.7 && v_color.b > 0.5 && v_color.b < 0.9) {
        // Headlight vertices — make them glow
        lit = base * 2.5;
        lit += vec3(0.3, 0.25, 0.15);
    }
    // Taillight glow
    if (v_color.r > 0.7 && v_color.g < 0.2 && v_color.b < 0.15) {
        lit = base * 2.0;
        lit += vec3(0.4, 0.05, 0.02);
    }
    // Fog
    float dist = length(v_world_pos);
    float fog_t = clamp((dist - u_fog_near) / max(u_fog_far - u_fog_near, 1.0), 0.0, 1.0);
    fog_t = fog_t * fog_t;
    lit = mix(lit, u_fog_color, fog_t);
    frag = vec4(clamp(lit, 0.0, 1.0), v_color.a);
}
"""

# --- First-person POI ground halo shader (cylinder ring on the floor) ---
FP_POI_VERT = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
uniform sampler2D u_terrain;
uniform vec4 u_terrain_bounds;
uniform float u_terrain_scale;
uniform float u_terrain_active;
in vec3 in_pos;       // world-pixel position of vertex (x, y, z)
in vec3 in_normal;    // normal for lighting (0,1,0 for ground)
in vec4 in_color;     // marker color
in float in_height;   // normalized height (0=base, 1=top of cylinder)
in float in_icon_id;  // category hash
out vec4 v_color;
out vec3 v_world_pos;
out vec3 v_normal;
out float v_height;
out float v_icon_id;
void main() {
    vec3 p = in_pos;
    float terrain_z = 0.0;
    if (u_terrain_active > 0.5) {
        vec2 uv = (p.xy - u_terrain_bounds.xy) / (u_terrain_bounds.zw - u_terrain_bounds.xy);
        uv = clamp(uv, 0.0, 1.0);
        terrain_z = texture(u_terrain, uv).r * u_terrain_scale;
    }
    vec3 world = vec3(-p.x, p.z + terrain_z, -p.y);
    vec3 rel = world - u_cam_pos;
    gl_Position = u_proj * u_view * vec4(rel, 1.0);
    v_color = in_color;
    v_world_pos = rel;
    v_normal = vec3(-in_normal.x, in_normal.z, -in_normal.y);
    v_height = in_height;
    v_icon_id = in_icon_id;
}
"""

FP_POI_FRAG = """
#version 330
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_time;
in vec4 v_color;
in vec3 v_world_pos;
in vec3 v_normal;
in float v_height;
in float v_icon_id;
out vec4 frag;

void main() {
    vec3 col = v_color.rgb;
    vec3 n = normalize(v_normal);

    // Vertical glow: brighter at base, fading to top
    float glow = 1.0 - v_height;
    glow = glow * glow;  // quadratic falloff

    // Animated upward pulse wave
    float wave = sin(v_height * 12.0 - u_time * 3.0 + v_icon_id * 1.7) * 0.5 + 0.5;
    wave *= (1.0 - v_height);  // stronger at bottom
    col += col * wave * 0.35;

    // Wall lighting (soft wrap diffuse)
    float light = 0.4 + 0.6 * clamp(dot(n, vec3(0.0, 1.0, 0.0)) * 0.3 + 0.7, 0.0, 1.0);
    col *= light;

    // Emissive boost for the base ring
    float base_ring = smoothstep(0.05, 0.0, v_height);
    col += v_color.rgb * base_ring * 1.5;

    // Top edge glow
    float top_edge = smoothstep(0.85, 1.0, v_height);
    col += v_color.rgb * top_edge * 0.4;

    // Fog
    float dist = length(v_world_pos);
    float fog_t = clamp((dist - u_fog_near) / max(u_fog_far - u_fog_near, 1.0), 0.0, 1.0);
    fog_t = fog_t * fog_t;
    col = mix(col, u_fog_color, fog_t);

    // Alpha: solid at base, transparent at top, plus distance fade
    float alpha = v_color.a * (glow * 0.7 + 0.3) * (1.0 - fog_t * 0.7);
    // Subtle pulse
    alpha *= 0.85 + 0.15 * sin(u_time * 1.5 + v_icon_id * 2.3);

    frag = vec4(clamp(col, 0.0, 1.0), clamp(alpha, 0.0, 1.0));
}
"""

# --- 2D map POI marker shader (rendered via GL) ---
POI_2D_VERT = """
#version 330
uniform mat4 view_proj;
uniform float point_radius;
in vec2 in_position;
in vec2 in_quad;
in vec4 in_color;
in float in_icon_id;
out vec2 v_quad;
out vec4 v_color;
out float v_icon_id;
void main() {
    vec2 world_pos = in_position + in_quad * point_radius;
    gl_Position = view_proj * vec4(world_pos, 0.0, 1.0);
    gl_Position.z = -0.08;
    v_quad = in_quad;
    v_color = in_color;
    v_icon_id = in_icon_id;
}
"""

POI_2D_FRAG = """
#version 330
uniform float u_time;
in vec2 v_quad;
in vec4 v_color;
in float v_icon_id;
out vec4 frag;
void main() {
    float dist_sq = v_quad.x * v_quad.x + v_quad.y * v_quad.y;
    if (dist_sq > 1.0) discard;
    float edge = smoothstep(1.0, 0.7, dist_sq);
    vec3 col = v_color.rgb;
    // White border ring
    float ring = smoothstep(1.0, 0.88, dist_sq) * smoothstep(0.75, 0.82, dist_sq);
    col = mix(col, vec3(1.0), ring * 0.9);
    // Inner darken
    float inner = smoothstep(0.45, 0.35, dist_sq);
    col = mix(col, col * 0.35 + vec3(0.06), inner * 0.6);
    // Drop shadow
    float shadow = smoothstep(1.3, 0.9, dist_sq + 0.15);
    float pulse = 0.93 + 0.07 * sin(u_time * 1.5 + v_icon_id * 2.7);
    frag = vec4(clamp(col, 0.0, 1.0), edge * v_color.a * pulse);
}
"""


# ===========================================================================
#  Car 3D model geometry builder
# ===========================================================================

def _build_car_model(scale=1.0):
    """Build a low-poly car model. Returns float32 array with 10 floats/vert:
    x, y, z, nx, ny, nz, r, g, b, a.
    Car faces +Y, centered at origin, wheels on z=0."""
    all_verts = []
    s = scale

    # Dimensions (meters, scaled)
    L = 4.5 * s; W = 1.85 * s
    H_body = 0.75 * s; H_roof = 1.42 * s
    H_wheel = 0.32 * s; ground_clear = 0.15 * s

    # Colors
    body_c = (0.12, 0.14, 0.22, 1.0)
    roof_c = (0.08, 0.08, 0.12, 1.0)
    win_c = (0.15, 0.25, 0.45, 0.85)
    tire_c = (0.05, 0.05, 0.05, 1.0)
    rim_c = (0.6, 0.6, 0.65, 1.0)
    hl_c = (1.0, 0.95, 0.8, 1.0)
    tl_c = (0.9, 0.1, 0.05, 1.0)
    bump_c = (0.08, 0.08, 0.1, 1.0)
    under_c = (0.04, 0.04, 0.04, 1.0)

    def _q(p0, p1, p2, p3, nx, ny, nz, r, g, b, a):
        return [(*p0,nx,ny,nz,r,g,b,a), (*p1,nx,ny,nz,r,g,b,a), (*p2,nx,ny,nz,r,g,b,a),
                (*p0,nx,ny,nz,r,g,b,a), (*p2,nx,ny,nz,r,g,b,a), (*p3,nx,ny,nz,r,g,b,a)]

    def _t(p0, p1, p2, nx, ny, nz, r, g, b, a):
        return [(*p0,nx,ny,nz,r,g,b,a), (*p1,nx,ny,nz,r,g,b,a), (*p2,nx,ny,nz,r,g,b,a)]

    hw = W / 2; hl = L / 2
    zb = ground_clear; zm = H_body

    # Main body box
    all_verts += _q((-hw,hl,zb),(hw,hl,zb),(hw,hl,zm),(-hw,hl,zm), 0,1,0, *body_c)  # front
    all_verts += _q((hw,-hl,zb),(-hw,-hl,zb),(-hw,-hl,zm),(hw,-hl,zm), 0,-1,0, *body_c)  # rear
    all_verts += _q((-hw,-hl,zb),(-hw,hl,zb),(-hw,hl,zm),(-hw,-hl,zm), -1,0,0, *body_c)  # left
    all_verts += _q((hw,hl,zb),(hw,-hl,zb),(hw,-hl,zm),(hw,hl,zm), 1,0,0, *body_c)  # right
    all_verts += _q((-hw,-hl,zb),(hw,-hl,zb),(hw,hl,zb),(-hw,hl,zb), 0,0,-1, *under_c)  # bottom
    all_verts += _q((-hw,-hl,zm),(-hw,hl,zm),(hw,hl,zm),(hw,-hl,zm), 0,0,1, *body_c)  # top

    # Cabin / greenhouse
    cf = hl * 0.25; cr = -hl * 0.4; chw = hw * 0.88; zt = H_roof
    # Windshields
    all_verts += _q((-chw,cf,zm),(chw,cf,zm),(chw*0.92,cf-L*0.08,zt),(-chw*0.92,cf-L*0.08,zt), 0,0.7,0.7, *win_c)
    all_verts += _q((chw,cr,zm),(-chw,cr,zm),(-chw*0.92,cr+L*0.06,zt),(chw*0.92,cr+L*0.06,zt), 0,-0.7,0.7, *win_c)
    # Side windows
    all_verts += _q((-chw,cr,zm),(-chw,cf,zm),(-chw*0.92,cf-L*0.08,zt),(-chw*0.92,cr+L*0.06,zt), -1,0,0.2, *win_c)
    all_verts += _q((chw,cf,zm),(chw,cr,zm),(chw*0.92,cr+L*0.06,zt),(chw*0.92,cf-L*0.08,zt), 1,0,0.2, *win_c)
    # Roof
    rw = chw*0.92; rf = cf-L*0.08; rr = cr+L*0.06
    all_verts += _q((-rw,rr,zt),(-rw,rf,zt),(rw,rf,zt),(rw,rr,zt), 0,0,1, *roof_c)

    # Hood & trunk
    all_verts += _q((-hw,cf,zm),(hw,cf,zm),(hw,hl,zm-0.04*s),(-hw,hl,zm-0.04*s), 0,0.1,1.0, *body_c)
    all_verts += _q((hw,cr,zm),(-hw,cr,zm),(-hw,-hl,zm-0.06*s),(hw,-hl,zm-0.06*s), 0,-0.1,1.0, *body_c)

    # Bumpers
    bh = ground_clear + 0.1*s
    all_verts += _q((-hw*1.02,hl,ground_clear),(hw*1.02,hl,ground_clear),(hw*1.02,hl,bh),(-hw*1.02,hl,bh), 0,1,0, *bump_c)
    all_verts += _q((hw*1.02,-hl,ground_clear),(-hw*1.02,-hl,ground_clear),(-hw*1.02,-hl,bh),(hw*1.02,-hl,bh), 0,-1,0, *bump_c)

    # Headlights & taillights
    lw = 0.22*s; lh = 0.12*s
    for side in [-1, 1]:
        cx = side * hw * 0.7
        all_verts += _q((cx-lw,hl+0.01*s,zm-lh*2),(cx+lw,hl+0.01*s,zm-lh*2),(cx+lw,hl+0.01*s,zm-lh),(cx-lw,hl+0.01*s,zm-lh), 0,1,0, *hl_c)
        all_verts += _q((cx+lw,-hl-0.01*s,zm-lh*2.5),(cx-lw,-hl-0.01*s,zm-lh*2.5),(cx-lw,-hl-0.01*s,zm-lh),(cx+lw,-hl-0.01*s,zm-lh), 0,-1,0, *tl_c)

    # Wheels (octagonal cylinders)
    wr = 0.33*s; ww = 0.22*s; n_seg = 10
    wheel_pos = [(-hw-0.05*s,hl*0.6,H_wheel),(hw+0.05*s,hl*0.6,H_wheel),
                 (-hw-0.05*s,-hl*0.6,H_wheel),(hw+0.05*s,-hl*0.6,H_wheel)]
    for wx,wy,wz in wheel_pos:
        is_left = wx < 0
        x_in = wx + (ww/2 if is_left else -ww/2)
        x_out = wx - (ww/2 if is_left else -ww/2)
        for i in range(n_seg):
            a0 = 2*math.pi*i/n_seg; a1 = 2*math.pi*(i+1)/n_seg
            y0=wy+math.cos(a0)*wr; z0=wz+math.sin(a0)*wr
            y1=wy+math.cos(a1)*wr; z1=wz+math.sin(a1)*wr
            nm_y=(math.cos(a0)+math.cos(a1))*0.5; nm_z=(math.sin(a0)+math.sin(a1))*0.5
            nl=math.sqrt(nm_y*nm_y+nm_z*nm_z)
            if nl>0.001: nm_y/=nl; nm_z/=nl
            all_verts += _q((x_in,y0,z0),(x_out,y0,z0),(x_out,y1,z1),(x_in,y1,z1), 0,nm_y,nm_z, *tire_c)
            all_verts += _t((x_out,wy,wz),(x_out,y0,z0),(x_out,y1,z1), 1 if not is_left else -1,0,0, *rim_c)

    return np.array(all_verts, dtype='f4')


def _build_npc_car_model(scale=1.0, color_idx=0):
    """Build a simplified NPC car model with varied colors.
    Returns float32 array with 10 floats/vert: x, y, z, nx, ny, nz, r, g, b, a.
    Car faces +Y, centered at origin."""
    all_verts = []
    s = scale
    L = 4.2 * s; W = 1.8 * s
    H_body = 0.7 * s; H_roof = 1.35 * s
    ground_clear = 0.15 * s

    # NPC car color palette
    npc_colors = [
        (0.7, 0.12, 0.1, 1.0),   # red
        (0.15, 0.25, 0.6, 1.0),  # blue
        (0.6, 0.6, 0.6, 1.0),    # silver
        (0.1, 0.1, 0.1, 1.0),    # black
        (0.8, 0.75, 0.2, 1.0),   # yellow (taxi)
        (0.95, 0.95, 0.9, 1.0),  # white
        (0.2, 0.5, 0.2, 1.0),    # green
        (0.5, 0.2, 0.1, 1.0),    # brown
    ]
    body_c = npc_colors[color_idx % len(npc_colors)]
    roof_c = (body_c[0]*0.7, body_c[1]*0.7, body_c[2]*0.7, 1.0)
    win_c = (0.12, 0.2, 0.38, 0.85)
    tire_c = (0.05, 0.05, 0.05, 1.0)
    hl_c = (1.0, 0.95, 0.8, 1.0)
    tl_c = (0.9, 0.1, 0.05, 1.0)
    under_c = (0.04, 0.04, 0.04, 1.0)

    def _q(p0, p1, p2, p3, nx, ny, nz, r, g, b, a):
        return [(*p0,nx,ny,nz,r,g,b,a), (*p1,nx,ny,nz,r,g,b,a), (*p2,nx,ny,nz,r,g,b,a),
                (*p0,nx,ny,nz,r,g,b,a), (*p2,nx,ny,nz,r,g,b,a), (*p3,nx,ny,nz,r,g,b,a)]

    def _t(p0, p1, p2, nx, ny, nz, r, g, b, a):
        return [(*p0,nx,ny,nz,r,g,b,a), (*p1,nx,ny,nz,r,g,b,a), (*p2,nx,ny,nz,r,g,b,a)]

    hw = W / 2; hl = L / 2
    zb = ground_clear; zm = H_body

    # Body box
    all_verts += _q((-hw,hl,zb),(hw,hl,zb),(hw,hl,zm),(-hw,hl,zm), 0,1,0, *body_c)
    all_verts += _q((hw,-hl,zb),(-hw,-hl,zb),(-hw,-hl,zm),(hw,-hl,zm), 0,-1,0, *body_c)
    all_verts += _q((-hw,-hl,zb),(-hw,hl,zb),(-hw,hl,zm),(-hw,-hl,zm), -1,0,0, *body_c)
    all_verts += _q((hw,hl,zb),(hw,-hl,zb),(hw,-hl,zm),(hw,hl,zm), 1,0,0, *body_c)
    all_verts += _q((-hw,-hl,zb),(hw,-hl,zb),(hw,hl,zb),(-hw,hl,zb), 0,0,-1, *under_c)
    all_verts += _q((-hw,-hl,zm),(-hw,hl,zm),(hw,hl,zm),(hw,-hl,zm), 0,0,1, *body_c)

    # Cabin
    cf = hl * 0.25; cr = -hl * 0.4; chw = hw * 0.88; zt = H_roof
    all_verts += _q((-chw,cf,zm),(chw,cf,zm),(chw*0.9,cf-L*0.08,zt),(-chw*0.9,cf-L*0.08,zt), 0,0.7,0.7, *win_c)
    all_verts += _q((chw,cr,zm),(-chw,cr,zm),(-chw*0.9,cr+L*0.06,zt),(chw*0.9,cr+L*0.06,zt), 0,-0.7,0.7, *win_c)
    all_verts += _q((-chw,cr,zm),(-chw,cf,zm),(-chw*0.9,cf-L*0.08,zt),(-chw*0.9,cr+L*0.06,zt), -1,0,0.2, *win_c)
    all_verts += _q((chw,cf,zm),(chw,cr,zm),(chw*0.9,cr+L*0.06,zt),(chw*0.9,cf-L*0.08,zt), 1,0,0.2, *win_c)
    rw = chw*0.9; rf = cf-L*0.08; rr = cr+L*0.06
    all_verts += _q((-rw,rr,zt),(-rw,rf,zt),(rw,rf,zt),(rw,rr,zt), 0,0,1, *roof_c)

    # Hood & trunk
    all_verts += _q((-hw,cf,zm),(hw,cf,zm),(hw,hl,zm-0.04*s),(-hw,hl,zm-0.04*s), 0,0.1,1.0, *body_c)
    all_verts += _q((hw,cr,zm),(-hw,cr,zm),(-hw,-hl,zm-0.06*s),(hw,-hl,zm-0.06*s), 0,-0.1,1.0, *body_c)

    # Headlights & taillights
    lw = 0.2*s; lh = 0.1*s
    for side in [-1, 1]:
        cx = side * hw * 0.7
        all_verts += _q((cx-lw,hl+0.01*s,zm-lh*2),(cx+lw,hl+0.01*s,zm-lh*2),(cx+lw,hl+0.01*s,zm-lh),(cx-lw,hl+0.01*s,zm-lh), 0,1,0, *hl_c)
        all_verts += _q((cx+lw,-hl-0.01*s,zm-lh*2.5),(cx-lw,-hl-0.01*s,zm-lh*2.5),(cx-lw,-hl-0.01*s,zm-lh),(cx+lw,-hl-0.01*s,zm-lh), 0,-1,0, *tl_c)

    # Simplified wheels (just boxes for NPCs to save verts)
    wr = 0.3*s; ww = 0.2*s
    wheel_pos = [(-hw,hl*0.6,wr),(hw,hl*0.6,wr),(-hw,-hl*0.6,wr),(hw,-hl*0.6,wr)]
    for wx,wy,wz in wheel_pos:
        all_verts += _q((wx-ww/2,wy-wr,0),(wx+ww/2,wy-wr,0),(wx+ww/2,wy+wr,0),(wx-ww/2,wy+wr,0), 0,0,-1, *tire_c)
        all_verts += _q((wx-ww/2,wy-wr,wz*2),(wx+ww/2,wy-wr,wz*2),(wx+ww/2,wy+wr,wz*2),(wx-ww/2,wy+wr,wz*2), 0,0,1, *tire_c)

    return np.array(all_verts, dtype='f4')


# ===========================================================================
#  POI Services — extraction, categorization, and billboard geometry
# ===========================================================================

# Category definitions: (category_key, display_name, color_rgb, icon_char)
POI_CATEGORIES = {
    # Food & Drink
    "restaurant": ("Restaurant", (0.95, 0.35, 0.25), "\U0001F37D"),
    "cafe": ("Café", (0.65, 0.40, 0.20), "\u2615"),
    "bar": ("Bar", (0.55, 0.20, 0.60), "\U0001F378"),
    "fast_food": ("Fast Food", (1.0, 0.55, 0.10), "\U0001F354"),
    "food": ("Food", (0.90, 0.40, 0.20), "\U0001F37D"),
    "pub": ("Pub", (0.60, 0.30, 0.15), "\U0001F37A"),
    "bakery": ("Bakery", (0.85, 0.65, 0.30), "\U0001F35E"),
    "ice_cream": ("Ice Cream", (0.85, 0.55, 0.70), "\U0001F366"),
    # Shopping
    "shop": ("Shop", (0.20, 0.60, 0.85), "\U0001F6CD"),
    "mall": ("Mall", (0.25, 0.50, 0.80), "\U0001F3EC"),
    "supermarket": ("Supermarket", (0.20, 0.70, 0.35), "\U0001F6D2"),
    "clothing": ("Clothing", (0.70, 0.30, 0.70), "\U0001F455"),
    "convenience": ("Convenience", (0.30, 0.65, 0.40), "\U0001F3EA"),
    "department_store": ("Dept Store", (0.35, 0.45, 0.75), "\U0001F3EC"),
    # Accommodation
    "hotel": ("Hotel", (0.15, 0.45, 0.75), "\U0001F3E8"),
    "hostel": ("Hostel", (0.25, 0.55, 0.65), "\U0001F3E8"),
    "motel": ("Motel", (0.30, 0.50, 0.60), "\U0001F3E8"),
    # Health & Services
    "pharmacy": ("Pharmacy", (0.15, 0.70, 0.40), "\U0001F48A"),
    "hospital": ("Hospital", (0.85, 0.15, 0.15), "\U0001F3E5"),
    "clinic": ("Clinic", (0.75, 0.25, 0.25), "\U0001F3E5"),
    "dentist": ("Dentist", (0.50, 0.70, 0.80), "\U0001F9B7"),
    "bank": ("Bank", (0.25, 0.45, 0.30), "\U0001F3E6"),
    "atm": ("ATM", (0.30, 0.50, 0.35), "\U0001F4B3"),
    "post_office": ("Post Office", (0.80, 0.60, 0.15), "\U0001F4EE"),
    # Transport
    "fuel": ("Gas Station", (0.80, 0.50, 0.10), "\u26FD"),
    "parking": ("Parking", (0.20, 0.45, 0.70), "\U0001F17F"),
    "bus_stop": ("Bus Stop", (0.35, 0.55, 0.25), "\U0001F68F"),
    "station": ("Station", (0.40, 0.40, 0.65), "\U0001F689"),
    # Leisure & Culture
    "museum": ("Museum", (0.55, 0.35, 0.60), "\U0001F3DB"),
    "attraction": ("Attraction", (0.85, 0.55, 0.20), "\u2B50"),
    "park": ("Park", (0.20, 0.65, 0.25), "\U0001F333"),
    "cinema": ("Cinema", (0.50, 0.25, 0.55), "\U0001F3AC"),
    "theatre": ("Theatre", (0.60, 0.20, 0.40), "\U0001F3AD"),
    "gym": ("Gym", (0.40, 0.60, 0.30), "\U0001F3CB"),
    "library": ("Library", (0.45, 0.35, 0.25), "\U0001F4DA"),
    "place_of_worship": ("Worship", (0.50, 0.50, 0.60), "\U0001F6D0"),
    # Landmarks & Other
    "viewpoint": ("Viewpoint", (0.70, 0.40, 0.20), "\U0001F304"),
    "monument": ("Monument", (0.55, 0.40, 0.30), "\U0001F3DB"),
    "tower": ("Tower", (0.50, 0.50, 0.60), "\U0001F3D7"),
    "school": ("School", (0.35, 0.50, 0.70), "\U0001F3EB"),
    "college": ("College", (0.30, 0.45, 0.65), "\U0001F393"),
    "university": ("University", (0.25, 0.40, 0.70), "\U0001F393"),
    "fire_station": ("Fire Station", (0.85, 0.20, 0.15), "\U0001F692"),
    "police": ("Police", (0.20, 0.30, 0.60), "\U0001F46E"),
    "toilets": ("Restroom", (0.40, 0.50, 0.55), "\U0001F6BB"),
    "playground": ("Playground", (0.50, 0.70, 0.30), "\U0001F3A0"),
    "swimming_pool": ("Pool", (0.20, 0.60, 0.80), "\U0001F3CA"),
    "golf": ("Golf", (0.25, 0.60, 0.30), "\u26F3"),
    "garden": ("Garden", (0.30, 0.65, 0.30), "\U0001F33B"),
    "information": ("Info", (0.30, 0.50, 0.70), "\u2139"),
}

# Category groupings for color coding
_POI_COLOR_MAP = {
    "food": (0.95, 0.35, 0.25, 0.90),     # red-orange
    "drink": (0.55, 0.20, 0.60, 0.90),     # purple
    "shop": (0.20, 0.60, 0.85, 0.90),      # blue
    "hotel": (0.15, 0.45, 0.75, 0.90),     # dark blue
    "health": (0.15, 0.70, 0.40, 0.90),    # green
    "finance": (0.25, 0.45, 0.30, 0.85),   # dark green
    "transport": (0.80, 0.50, 0.10, 0.85), # orange
    "culture": (0.55, 0.35, 0.60, 0.90),   # violet
    "leisure": (0.20, 0.65, 0.25, 0.85),   # green
    "default": (0.40, 0.55, 0.70, 0.80),   # visible teal-grey
}

def _poi_category_group(cat_type):
    """Map a POI type string to a group for color-coding."""
    food_types = {"restaurant", "cafe", "fast_food", "food", "bakery", "ice_cream"}
    drink_types = {"bar", "pub"}
    shop_types = {"shop", "mall", "supermarket", "clothing", "convenience", "department_store"}
    hotel_types = {"hotel", "hostel", "motel"}
    health_types = {"pharmacy", "hospital", "clinic", "dentist"}
    finance_types = {"bank", "atm", "post_office"}
    transport_types = {"fuel", "parking", "bus_stop", "station"}
    culture_types = {"museum", "attraction", "cinema", "theatre", "library", "place_of_worship",
                     "monument", "tower", "viewpoint"}
    leisure_types = {"park", "gym", "playground", "swimming_pool", "golf", "garden"}
    education_types = {"school", "college", "university"}
    ct = cat_type.lower()
    if ct in food_types: return "food"
    if ct in drink_types: return "drink"
    if ct in shop_types: return "shop"
    if ct in hotel_types: return "hotel"
    if ct in health_types: return "health"
    if ct in finance_types: return "finance"
    if ct in transport_types: return "transport"
    if ct in culture_types: return "culture"
    if ct in leisure_types: return "leisure"
    if ct in education_types: return "culture"
    return "default"


def _poi_color(cat_type):
    """Get RGBA color for a POI category."""
    group = _poi_category_group(cat_type)
    return _POI_COLOR_MAP.get(group, _POI_COLOR_MAP["default"])


def _poi_icon(cat_type):
    """Get display icon/emoji for a POI type."""
    info = POI_CATEGORIES.get(cat_type.lower())
    if info:
        return info[2]
    return "\U0001F4CD"  # pin fallback


def _poi_display_name(cat_type):
    """Get human-readable name for a POI type."""
    info = POI_CATEGORIES.get(cat_type.lower())
    if info:
        return info[0]
    return cat_type.replace("_", " ").title()


def _extract_pois_from_tiles(tile_cache, visible_keys, tile_px, zoom):
    """Extract POI features from cached MVT tiles.
    Returns list of dicts: {name, type, group, wx, wy, lat, lon, icon, color}."""
    pois = []
    seen_names = set()  # deduplicate by name+position

    for tkey in visible_keys:
        mvt = tile_cache.get(tkey)
        if mvt is None:
            continue
        tz, tx_w, ty = tkey
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)
        n = 2 ** tz

        for ln, layer in mvt.layers.items():
            lnl = ln.lower()
            is_poi = "poi" in lnl
            # Also extract named features from place_label (landmarks, parks, etc.)
            is_place = "place_label" in lnl or "place" == lnl
            # Also check mountain_peak, natural_label, etc.
            is_natural = "natural" in lnl or "mountain" in lnl
            if not (is_poi or is_place or is_natural):
                continue
            ext = layer.extent or MVT_EXTENT
            inv_ext = tile_px / ext

            for feat in layer.features:
                if feat.geom_type != 1:  # only POINT features
                    continue
                name = feat.properties.get("name", feat.properties.get("name_en", ""))
                if not name or not isinstance(name, str):
                    continue
                name = str(name).strip()
                if not name:
                    continue

                cat = str(feat.properties.get("type", feat.properties.get("class", ""))).lower()
                # Try to match to a known service/commerce category
                group = _poi_category_group(cat)
                if group == "default" and cat not in POI_CATEGORIES:
                    # Try maki icon as fallback category
                    maki = str(feat.properties.get("maki", "")).lower()
                    if maki in POI_CATEGORIES:
                        cat = maki
                        group = _poi_category_group(cat)
                    elif maki in ("restaurant", "cafe", "bar", "shop", "grocery",
                                   "fuel", "parking", "bank", "hospital", "pharmacy",
                                   "hotel", "cinema", "theatre", "museum", "park",
                                   "bus", "rail", "bakery", "lodging", "clothing-store",
                                   "grocery", "beer", "art-gallery", "dentist",
                                   "doctor", "library", "garden", "swimming",
                                   "school", "college", "monument", "viewpoint",
                                   "fire-station", "police", "golf", "playground",
                                   "information", "toilets", "tower", "zoo",
                                   "attraction", "stadium", "fitness-centre"):
                        cat = maki.replace("-", "_")
                        group = _poi_category_group(cat)
                    elif cat in ("food_and_drink", "food_and_drink_stores"):
                        cat = "restaurant"; group = "food"
                    elif cat in ("shop", "commercial_services", "store"):
                        cat = "shop"; group = "shop"
                    elif cat in ("lodging"):
                        cat = "hotel"; group = "hotel"
                    elif cat in ("medical"):
                        cat = "hospital"; group = "health"
                    elif cat in ("arts_and_entertainment"):
                        cat = "attraction"; group = "culture"
                    elif cat in ("education", "school"):
                        cat = "library"; group = "culture"
                    elif cat in ("sport_and_leisure", "fitness"):
                        cat = "gym"; group = "leisure"
                    elif cat in ("religious"):
                        cat = "place_of_worship"; group = "culture"
                    # Accept any remaining named POI with a generic marker
                    # (don't skip — all named POIs are useful)

                rings = _get_rings(feat, ext)
                if not rings:
                    continue
                pt = rings[0][0]
                wx = tile_ox + pt[0] * inv_ext
                wy = tile_oy + pt[1] * inv_ext

                # Deduplicate
                dedup_key = (name.lower(), int(wx / 10), int(wy / 10))
                if dedup_key in seen_names:
                    continue
                seen_names.add(dedup_key)

                # Compute lat/lon
                fx = (tx_w + pt[0] / ext)
                fy = (ty + pt[1] / ext)
                lon = fx / n * 360.0 - 180.0
                try:
                    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy / n)))
                    lat = math.degrees(lat_rad)
                except:
                    lat = 0.0

                color = _poi_color(cat)
                icon = _poi_icon(cat)
                display = _poi_display_name(cat)

                pois.append({
                    "name": name,
                    "type": cat,
                    "group": group,
                    "display_type": display,
                    "wx": wx,
                    "wy": wy,
                    "lat": lat,
                    "lon": lon,
                    "icon": icon,
                    "color": color,
                })

    return pois


def _build_poi_billboards_fp(pois, cam_wx, cam_wy, mpp, max_dist_m=300.0, max_pois=80):
    """Build cylindrical halo geometry for POIs in first-person immersive mode.
    Each POI gets a glowing cylinder ring on the ground.
    Returns float32 array: x, y, z, nx, ny, nz, r, g, b, a, height_norm, icon_id (12 floats per vert).
    """
    if not pois:
        return np.empty((0, 12), dtype='f4'), []

    max_dist_px = max_dist_m / max(mpp, 0.001)

    # Sort by distance, filter by range
    scored = []
    for poi in pois:
        dx = poi["wx"] - cam_wx
        dy = poi["wy"] - cam_wy
        dist = math.hypot(dx, dy)
        if dist < max_dist_px:
            scored.append((dist, poi))
    scored.sort(key=lambda x: x[0])
    scored = scored[:max_pois]

    if not scored:
        return np.empty((0, 12), dtype='f4'), []

    all_verts = []
    visible_pois = []

    n_seg = 24  # cylinder segments (smooth enough)
    cylinder_h = 10.0 / max(mpp, 0.001)  # ~10m tall cylinder
    cylinder_r_base = 2.5 / max(mpp, 0.001)  # ~2.5m radius

    for idx, (dist, poi) in enumerate(scored):
        wx, wy = poi["wx"], poi["wy"]
        r, g, b, a = poi["color"]
        # Boost alpha for visibility, taper with distance
        dist_factor = max(0.4, min(1.0, 80.0 / max(mpp, 0.001) / max(dist, 1.0)))
        a_eff = min(0.85, a * dist_factor)
        # Slightly vary radius with category
        radius = cylinder_r_base * (0.85 + 0.3 * ((hash(poi["type"]) % 7) / 7.0))
        h = cylinder_h * dist_factor

        icon_id = float(hash(poi["type"]) % 100)

        # Build cylinder wall (n_seg quads = n_seg * 6 verts)
        for i in range(n_seg):
            a0 = 2.0 * math.pi * i / n_seg
            a1 = 2.0 * math.pi * (i + 1) / n_seg
            cos0, sin0 = math.cos(a0), math.sin(a0)
            cos1, sin1 = math.cos(a1), math.sin(a1)

            # Vertex positions (world pixel coords, z = height)
            x0 = wx + cos0 * radius
            y0 = wy + sin0 * radius
            x1 = wx + cos1 * radius
            y1 = wy + sin1 * radius

            # Outward normal for this segment
            nmx = (cos0 + cos1) * 0.5
            nmy = (sin0 + sin1) * 0.5
            nl = math.sqrt(nmx * nmx + nmy * nmy)
            if nl > 1e-6:
                nmx /= nl; nmy /= nl

            # Two triangles per quad (bottom-left, bottom-right, top-right, bottom-left, top-right, top-left)
            # Vertex: x, y, z, nx, ny, nz, r, g, b, a, height_norm, icon_id
            all_verts.extend([
                (x0, y0, 0.0,   nmx, nmy, 0.0, r, g, b, a_eff, 0.0, icon_id),
                (x1, y1, 0.0,   nmx, nmy, 0.0, r, g, b, a_eff, 0.0, icon_id),
                (x1, y1, h,     nmx, nmy, 0.0, r, g, b, a_eff, 1.0, icon_id),
                (x0, y0, 0.0,   nmx, nmy, 0.0, r, g, b, a_eff, 0.0, icon_id),
                (x1, y1, h,     nmx, nmy, 0.0, r, g, b, a_eff, 1.0, icon_id),
                (x0, y0, h,     nmx, nmy, 0.0, r, g, b, a_eff, 1.0, icon_id),
            ])

        visible_pois.append(poi)

    return np.array(all_verts, dtype='f4'), visible_pois


def _build_poi_markers_2d(pois, zoom, max_pois=80):
    """Build 2D marker geometry for POIs on the overview map.
    Re-projects lat/lon to current zoom's world-pixel coords.
    Returns float32 array: wx, wy, qx, qy, r, g, b, a, icon_id (9 floats per vert).
    Each POI = 6 verts (1 quad)."""
    if not pois:
        return np.empty((0, 9), dtype='f4')

    used = pois[:max_pois]
    all_verts = []
    for poi in used:
        r, g, b, a = poi["color"]
        icon_id = float(hash(poi["type"]) % 100)
        # Reproject from lat/lon at the CURRENT zoom (not the extraction zoom)
        wx = _lon_to_wx(poi["lon"], zoom)
        wy = _lat_to_wy(poi["lat"], zoom)
        # Update the cached wx/wy so hit-testing and popups use correct coords
        poi["wx"] = wx
        poi["wy"] = wy
        quad_offsets = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
        for qx, qy in quad_offsets:
            all_verts.append((wx, wy, qx, qy, r, g, b, a, icon_id))

    return np.array(all_verts, dtype='f4')


class NPCCar:
    """A single traffic NPC car driving along a road segment."""
    __slots__ = ('wx', 'wy', 'yaw', 'speed', 'target_speed', 'color_idx',
                 'path', 'path_idx', 'road_class', 'alive', 'steer_timer',
                 'length_m', 'width_m', 'braking')
    def __init__(self):
        self.wx = 0.0; self.wy = 0.0; self.yaw = 0.0
        self.speed = 0.0; self.target_speed = 0.0
        self.color_idx = 0; self.path = []; self.path_idx = 0
        self.road_class = "street"; self.alive = True
        self.steer_timer = 0.0; self.length_m = 4.2; self.width_m = 1.8
        self.braking = False


def _extract_road_paths(tile_cache, visible_keys, tile_px, meters_per_px):
    """Extract road polylines as world-pixel paths for NPC spawning."""
    paths = []
    road_speeds = {
        "motorway": 25.0, "trunk": 20.0, "primary": 15.0,
        "secondary": 12.0, "tertiary": 10.0, "street": 8.0,
        "service": 5.0,
    }
    for tkey in visible_keys:
        mvt = tile_cache.get(tkey)
        if mvt is None:
            continue
        tz, tx_w, ty = tkey
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)
        for ln, layer in mvt.layers.items():
            if not _match_layer(ln, "road"):
                continue
            ext = layer.extent or MVT_EXTENT
            inv_ext = tile_px / ext
            for feat in layer.features:
                if feat.geom_type not in (2,):
                    continue
                fclass = _get_feature_class(feat)
                if fclass in ("path", "pedestrian", "track", "rail", "railway"):
                    continue
                tgt_speed = road_speeds.get(fclass, 8.0)
                rings = _get_rings(feat, ext)
                for ring in rings:
                    if len(ring) < 2:
                        continue
                    wp = [(tile_ox + px * inv_ext, tile_oy + py * inv_ext) for px, py in ring]
                    # Only keep paths longer than ~20m
                    total_len = sum(math.hypot(wp[i+1][0]-wp[i][0], wp[i+1][1]-wp[i][1])
                                    for i in range(len(wp)-1))
                    if total_len * meters_per_px > 20.0:
                        paths.append((wp, fclass, tgt_speed))
    return paths


def _spawn_npc_cars(road_paths, mpp, max_cars=40, rng_seed=None):
    """Spawn NPC cars along road paths based on road class density."""
    import random
    rng = random.Random(rng_seed)
    density = {
        "motorway": 0.6, "trunk": 0.5, "primary": 0.4,
        "secondary": 0.3, "tertiary": 0.2, "street": 0.15,
        "service": 0.05,
    }
    cars = []
    for wp, fclass, tgt_speed in road_paths:
        d = density.get(fclass, 0.1)
        total_len = sum(math.hypot(wp[i+1][0]-wp[i][0], wp[i+1][1]-wp[i][1])
                        for i in range(len(wp)-1))
        total_m = total_len * mpp
        n_cars = max(0, int(total_m * d / 50.0))  # ~1 car per 50m * density
        n_cars = min(n_cars, 3)  # max 3 per segment
        for _ in range(n_cars):
            if len(cars) >= max_cars:
                return cars
            t = rng.random()
            # Find position along path at fraction t
            target_dist = t * total_len
            accum = 0.0
            for i in range(len(wp) - 1):
                seg_len = math.hypot(wp[i+1][0]-wp[i][0], wp[i+1][1]-wp[i][1])
                if accum + seg_len >= target_dist and seg_len > 0.01:
                    frac = (target_dist - accum) / seg_len
                    car = NPCCar()
                    car.wx = wp[i][0] + frac * (wp[i+1][0] - wp[i][0])
                    car.wy = wp[i][1] + frac * (wp[i+1][1] - wp[i][1])
                    dx = wp[i+1][0] - wp[i][0]
                    dy = wp[i+1][1] - wp[i][1]
                    car.yaw = math.degrees(math.atan2(dx, -dy)) % 360.0
                    # Randomize: some go opposite direction
                    if rng.random() < 0.4:
                        car.yaw = (car.yaw + 180.0) % 360.0
                        car.path = list(reversed(wp))
                        car.path_idx = len(wp) - 1 - i
                    else:
                        car.path = list(wp)
                        car.path_idx = i
                    speed_var = rng.uniform(0.7, 1.1)
                    car.target_speed = tgt_speed * speed_var / mpp
                    car.speed = car.target_speed * rng.uniform(0.5, 1.0)
                    car.color_idx = rng.randint(0, 7)
                    car.road_class = fclass
                    # Offset to right side of road
                    perp_x = math.cos(math.radians(car.yaw))
                    perp_y = math.sin(math.radians(car.yaw))
                    lane_offset = 3.0 / mpp  # ~3m to the right
                    car.wx += perp_x * lane_offset
                    car.wy += perp_y * lane_offset
                    cars.append(car)
                    break
                accum += seg_len
    return cars


def _tick_npc_cars(cars, dt, mpp, car_wx=None, car_wy=None, view_dist=500.0):
    """Update all NPC cars — follow paths, avoid collisions with each other."""
    cull_dist_sq = (view_dist / mpp) ** 2 if mpp > 0 else 1e18
    for car in cars:
        if not car.alive:
            continue
        # Cull far-away NPCs (stop updating)
        if car_wx is not None:
            dx = car.wx - car_wx; dy = car.wy - car_wy
            if dx*dx + dy*dy > cull_dist_sq * 4:
                continue

        # Accelerate / decelerate toward target speed
        accel = 5.0 / max(mpp, 0.001) * dt
        if car.speed < car.target_speed:
            car.speed = min(car.target_speed, car.speed + accel)
        elif car.speed > car.target_speed:
            car.speed = max(car.target_speed, car.speed - accel * 2.0)

        # Follow path
        if car.path and car.path_idx < len(car.path) - 1:
            tx, ty = car.path[car.path_idx + 1]
            dx = tx - car.wx; dy = ty - car.wy
            dist = math.hypot(dx, dy)
            if dist < 2.0 / max(mpp, 0.001):
                car.path_idx += 1
                if car.path_idx >= len(car.path) - 1:
                    car.alive = False
                    continue
            else:
                target_yaw = math.degrees(math.atan2(dx, -dy)) % 360.0
                diff = (target_yaw - car.yaw + 180.0) % 360.0 - 180.0
                max_turn = 90.0 * dt
                car.yaw = (car.yaw + max(-max_turn, min(max_turn, diff))) % 360.0

        # Move
        yaw_rad = math.radians(car.yaw)
        car.wx += math.sin(yaw_rad) * car.speed * dt
        car.wy -= math.cos(yaw_rad) * car.speed * dt

    # Check NPC-to-NPC braking (simple)
    for i, c1 in enumerate(cars):
        if not c1.alive:
            continue
        c1.braking = False
        for j, c2 in enumerate(cars):
            if i == j or not c2.alive:
                continue
            dx = c2.wx - c1.wx; dy = c2.wy - c1.wy
            dist = math.hypot(dx, dy) * mpp
            if dist < 12.0:  # within 12m
                ahead_dot = math.sin(math.radians(c1.yaw)) * dx + (-math.cos(math.radians(c1.yaw))) * dy
                if ahead_dot > 0:  # car ahead
                    c1.speed = min(c1.speed, c2.speed * 0.8)
                    c1.braking = True

def _build_fp_buildings(building_cache, visible_bld_keys, tile_px,
                        default_h, h_exag, meters_per_px,
                        wall_color, roof_color, terrain_cache=None, zoom=13.0):
    """Build first-person building geometry in WORLD-PIXEL coords.
    Returns flat float32 array with vertex layout:
    x, y, z, nx, ny, nz, r, g, b, a (10 floats).
    x,y are world-pixel coords, z is height in world-pixels.
    Terrain displacement is handled by GPU shader — geometry stays at z=0 base."""
    all_verts = []
    wr, wg, wb, wa = wall_color
    rr, rg, rb, ra = roof_color

    for tkey in visible_bld_keys:
        bldgs = building_cache.get(tkey)
        if not bldgs:
            continue
        # Sort by height, limit count
        if len(bldgs) > 300:
            bldgs = sorted(bldgs, key=lambda b: b["height"], reverse=True)[:300]
        tx_w, ty = tkey[1], tkey[2]
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)

        for bld in bldgs:
            ext = bld["extent"]
            s_px = tile_px / ext
            bh = bld["height"]
            # Convert meters to world-pixel height
            h_meters = (bh if bh > 0 else default_h) * h_exag
            h_px = h_meters / meters_per_px  # convert to world pixels

            for ring in bld["rings"]:
                if len(ring) < 3:
                    continue
                pts = np.array(ring, dtype='f4')
                pts[:, 0] = tile_ox + pts[:, 0] * s_px
                pts[:, 1] = tile_oy + pts[:, 1] * s_px
                n_pts = len(pts)

                # Sample terrain elevation at building centroid
                base_z = 0.0
                # (terrain displacement now handled in GPU vertex shader)

                # Walls
                p0 = pts[:-1]; p1 = pts[1:]
                edges = p1 - p0
                edge_lens = np.sqrt(edges[:, 0]**2 + edges[:, 1]**2)
                valid = edge_lens > 0.01
                if not np.any(valid):
                    continue
                p0v = p0[valid]; p1v = p1[valid]; ev = edges[valid]; elv = edge_lens[valid]
                # Normal in xz plane (Y-up): for wall facing outward
                nx = -ev[:, 1] / elv
                ny = ev[:, 0] / elv
                nw = len(p0v)

                wall_data = np.empty((nw * 6, 10), dtype='f4')
                # Bottom-left, bottom-right, top-right, bottom-left, top-right, top-left
                wall_data[0::6, 0] = p0v[:, 0]; wall_data[0::6, 1] = p0v[:, 1]; wall_data[0::6, 2] = base_z
                wall_data[1::6, 0] = p1v[:, 0]; wall_data[1::6, 1] = p1v[:, 1]; wall_data[1::6, 2] = base_z
                wall_data[2::6, 0] = p1v[:, 0]; wall_data[2::6, 1] = p1v[:, 1]; wall_data[2::6, 2] = base_z + h_px
                wall_data[3::6, 0] = p0v[:, 0]; wall_data[3::6, 1] = p0v[:, 1]; wall_data[3::6, 2] = base_z
                wall_data[4::6, 0] = p1v[:, 0]; wall_data[4::6, 1] = p1v[:, 1]; wall_data[4::6, 2] = base_z + h_px
                wall_data[5::6, 0] = p0v[:, 0]; wall_data[5::6, 1] = p0v[:, 1]; wall_data[5::6, 2] = base_z + h_px
                for k in range(6):
                    wall_data[k::6, 3] = nx   # normal x
                    wall_data[k::6, 4] = 0.0  # normal y (Y-up, walls are vertical)
                    wall_data[k::6, 5] = ny   # normal z (mapped from old y)
                wall_data[:, 6] = wr; wall_data[:, 7] = wg
                wall_data[:, 8] = wb; wall_data[:, 9] = wa
                all_verts.append(wall_data)

                # Roof
                if n_pts >= 3:
                    roof_pts = pts
                    if n_pts >= 4:
                        d = pts[-1] - pts[0]
                        if abs(d[0]) < 0.01 and abs(d[1]) < 0.01:
                            roof_pts = pts[:-1]
                    n_roof = len(roof_pts)
                    if n_roof >= 3:
                        nt = n_roof - 2
                        roof_data = np.empty((nt * 3, 10), dtype='f4')
                        i0 = np.zeros(nt, dtype=np.int32)
                        i1 = np.arange(1, nt + 1, dtype=np.int32)
                        i2 = np.arange(2, nt + 2, dtype=np.int32)
                        roof_data[0::3, 0] = roof_pts[i0, 0]; roof_data[0::3, 1] = roof_pts[i0, 1]
                        roof_data[1::3, 0] = roof_pts[i1, 0]; roof_data[1::3, 1] = roof_pts[i1, 1]
                        roof_data[2::3, 0] = roof_pts[i2, 0]; roof_data[2::3, 1] = roof_pts[i2, 1]
                        roof_data[:, 2] = base_z + h_px
                        roof_data[:, 3] = 0.0; roof_data[:, 4] = 1.0; roof_data[:, 5] = 0.0  # up normal
                        roof_data[:, 6] = rr; roof_data[:, 7] = rg
                        roof_data[:, 8] = rb; roof_data[:, 9] = ra
                        all_verts.append(roof_data)

    if not all_verts:
        return np.empty((0, 10), dtype='f4')
    return np.concatenate(all_verts, axis=0)


def _build_fp_ground(cx, cy, radius, tile_px, style, terrain_cache=None, zoom=13.0, mpp=1.0):
    """Build a ground plane grid. Terrain displacement happens in the GPU shader.
    Returns float32 array: x, y, z, r, g, b, a (7 floats per vert)."""
    bg = style.bg
    r, g, b = bg.red() / 255.0, bg.green() / 255.0, bg.blue() / 255.0
    r = min(1.0, r * 1.15 + 0.02)
    g = min(1.0, g * 1.15 + 0.02)
    b = min(1.0, b * 1.15 + 0.02)

    # Always use a grid (for GPU terrain displacement to look smooth)
    grid_n = 80  # 80x80 grid — enough resolution, fast to generate
    step = (radius * 2.0) / grid_n
    x0 = cx - radius
    y0 = cy - radius

    # Pre-allocate all vertices: 6 verts per cell (2 triangles)
    n_verts = grid_n * grid_n * 6
    verts = np.empty((n_verts, 7), dtype='f4')

    # Build grid coords using numpy broadcasting
    ix = np.arange(grid_n, dtype='f4')
    iy = np.arange(grid_n, dtype='f4')
    gx, gy = np.meshgrid(ix, iy)  # (grid_n, grid_n)
    gx = gx.ravel(); gy = gy.ravel()
    n_cells = grid_n * grid_n

    wx0 = x0 + gx * step
    wx1 = x0 + (gx + 1) * step
    wy0 = y0 + gy * step
    wy1 = y0 + (gy + 1) * step

    # 6 verts per quad: v0(x0,y0), v1(x1,y0), v2(x1,y1), v3(x0,y0), v4(x1,y1), v5(x0,y1)
    base = np.arange(n_cells) * 6
    for i, (vx, vy) in enumerate([
        (wx0, wy0), (wx1, wy0), (wx1, wy1),
        (wx0, wy0), (wx1, wy1), (wx0, wy1),
    ]):
        verts[base + i, 0] = vx
        verts[base + i, 1] = vy
        verts[base + i, 2] = 0.0  # z=0, GPU shader adds terrain
        verts[base + i, 3] = r
        verts[base + i, 4] = g
        verts[base + i, 5] = b
        verts[base + i, 6] = 1.0

    return verts


def _build_fp_water(tile_cache, visible_keys, tile_px, style):
    """Extract water polygons from MVT tiles and tessellate into flat quads.
    Returns float32 array: x, y, z, r, g, b, a (7 floats per vert)."""
    water_layers = [ls for ls in style.layers if ls["type"] == "fill" and ls["match"] == "water"]
    if not water_layers:
        return np.empty((0, 7), dtype='f4')

    # Get water color from style
    wc = water_layers[0]["fill"]
    wr, wg, wb = wc.red() / 255.0, wc.green() / 255.0, wc.blue() / 255.0

    all_verts = []
    for tkey in visible_keys:
        mvt = tile_cache.get(tkey)
        if mvt is None:
            continue
        tz, tx_w, ty = tkey
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)

        for ls in water_layers:
            matched = [layer for ln, layer in mvt.layers.items() if _match_layer(ln, "water")]
            for layer in matched:
                ext = layer.extent or MVT_EXTENT
                inv_ext = tile_px / ext
                for feat in layer.features:
                    if feat.geom_type not in (3,):  # polygons only
                        continue
                    rings = _get_rings(feat, ext)
                    if not rings:
                        continue
                    for ring in rings:
                        if len(ring) < 3:
                            continue
                        # Transform to world pixels
                        wp = [(tile_ox + px * inv_ext, tile_oy + py * inv_ext) for px, py in ring]
                        # Fan triangulation from first vertex
                        p0 = wp[0]
                        for j in range(1, len(wp) - 1):
                            p1 = wp[j]
                            p2 = wp[j + 1]
                            all_verts.append((p0[0], p0[1], 0.1, wr, wg, wb, 0.92))
                            all_verts.append((p1[0], p1[1], 0.1, wr, wg, wb, 0.92))
                            all_verts.append((p2[0], p2[1], 0.1, wr, wg, wb, 0.92))

    if not all_verts:
        return np.empty((0, 7), dtype='f4')
    return np.array(all_verts, dtype='f4')


def _build_fp_roads(tile_cache, visible_keys, tile_px, style, zoom, meters_per_px, terrain_cache=None):
    """Build nav-style road geometry: casing + shoulder + asphalt layers with UVs.
    Returns float32 array: x, y, z, r, g, b, a, u_along, v_across (9 floats)."""
    all_verts = []
    road_layers = [ls for ls in style.layers if ls["type"] == "line" and ls["match"] in ("road", "bridge")]
    if not road_layers:
        return np.empty((0, 9), dtype='f4')

    meter_widths = {
        "motorway": 18.0, "trunk": 15.0, "primary": 12.0,
        "secondary": 10.0, "tertiary": 8.0, "street": 7.0,
        "service": 5.0, "path": 2.0, "pedestrian": 3.0,
        "track": 3.5, "rail": 3.0, "railway": 3.0,
    }
    markable_classes = {"motorway", "trunk", "primary", "secondary", "tertiary", "street"}

    bg = style.bg
    bg_lum = (bg.red() + bg.green() + bg.blue()) / (3.0 * 255.0)
    is_dark = bg_lum < 0.3

    for tkey in visible_keys:
        mvt = tile_cache.get(tkey)
        if mvt is None:
            continue
        tz, tx_w, ty = tkey
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)

        for ls in road_layers:
            matched = [layer for ln, layer in mvt.layers.items() if _match_layer(ln, ls["match"])]
            for layer in matched:
                ext = layer.extent or MVT_EXTENT
                inv_ext = tile_px / ext

                for feat in layer.features:
                    if feat.geom_type not in (2, 3):
                        continue
                    fclass = _get_feature_class(feat)
                    if fclass in ("rail", "railway"):
                        continue

                    rings = _get_rings(feat, ext)
                    if not rings:
                        continue

                    road_meters = meter_widths.get(fclass, 7.0)
                    hw_px = (road_meters / max(meters_per_px, 0.001)) * 0.5
                    has_markings = fclass in markable_classes

                    if fclass in ("path", "pedestrian", "track"):
                        if is_dark:
                            asphalt = (0.32, 0.30, 0.26, 0.88)
                            casing  = (0.18, 0.17, 0.15, 0.70)
                        else:
                            asphalt = (0.73, 0.70, 0.64, 0.92)
                            casing  = (0.58, 0.55, 0.50, 0.70)
                    else:
                        if is_dark:
                            asphalt = (0.18, 0.19, 0.22, 1.00)
                            casing  = (0.08, 0.09, 0.11, 0.95)
                            shoulder = (0.11, 0.12, 0.14, 0.58)
                        else:
                            asphalt = (0.29, 0.30, 0.32, 1.00)
                            casing  = (0.15, 0.16, 0.17, 0.95)
                            shoulder = (0.78, 0.76, 0.72, 0.42)

                    for ring in rings:
                        if len(ring) < 2:
                            continue
                        wp = [(tile_ox + px * inv_ext, tile_oy + py * inv_ext) for px, py in ring]

                        cum_dist = [0.0]
                        for i in range(1, len(wp)):
                            dx = wp[i][0] - wp[i - 1][0]
                            dy = wp[i][1] - wp[i - 1][1]
                            cum_dist.append(cum_dist[-1] + math.sqrt(dx * dx + dy * dy))

                        # 1) outer casing
                        casing_verts = _thicken_line_uv(wp, hw_px * 1.22, cum_dist)
                        for px, py, u_along, v_across in casing_verts:
                            all_verts.append((px, py, -0.03, casing[0], casing[1], casing[2], casing[3], u_along, v_across))

                        # 2) shoulder / soft border (main roads only)
                        if fclass not in ("path", "pedestrian", "track"):
                            shoulder_verts = _thicken_line_uv(wp, hw_px * 1.10, cum_dist)
                            for px, py, u_along, v_across in shoulder_verts:
                                all_verts.append((px, py, -0.01, shoulder[0], shoulder[1], shoulder[2], shoulder[3], u_along, v_across))

                        # 3) main asphalt surface
                        road_verts = _thicken_line_uv(wp, hw_px, cum_dist)
                        main_alpha = 1.0 if has_markings else 0.92
                        for px, py, u_along, v_across in road_verts:
                            all_verts.append((px, py, 0.05, asphalt[0], asphalt[1], asphalt[2], main_alpha, u_along, v_across))

    if not all_verts:
        return np.empty((0, 9), dtype='f4')
    return np.array(all_verts, dtype='f4')


def _thicken_line_uv(ring, half_width, cum_dist):
    """Like _thicken_line but returns (x, y, u_along, v_across) per vertex.
    v_across ranges from -1 (left) to +1 (right) across the road.
    u_along is the cumulative distance along the polyline at that vertex."""
    if len(ring) < 2:
        return []
    pts = ring
    n = len(pts)
    # Remove duplicate consecutive points (keep cum_dist in sync)
    cleaned = [pts[0]]
    cleaned_dist = [cum_dist[0]]
    for i in range(1, n):
        dx = pts[i][0] - cleaned[-1][0]
        dy = pts[i][1] - cleaned[-1][1]
        if dx*dx + dy*dy > 1e-8:
            cleaned.append(pts[i])
            cleaned_dist.append(cum_dist[i])
    pts = cleaned
    dists = cleaned_dist
    n = len(pts)
    if n < 2:
        return []

    # Compute per-segment direction normals
    seg_normals = []
    for i in range(n - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        l = math.sqrt(dx*dx + dy*dy)
        if l > 1e-8:
            seg_normals.append((-dy/l, dx/l))
        else:
            seg_normals.append((0, 1))

    # Compute per-vertex normals (average of adjacent segments, miter clamped)
    normals = []
    for i in range(n):
        if i == 0:
            nx, ny = seg_normals[0]
        elif i == n - 1:
            nx, ny = seg_normals[-1]
        else:
            nx0, ny0 = seg_normals[i-1]
            nx1, ny1 = seg_normals[i]
            nx = nx0 + nx1; ny = ny0 + ny1
            l = math.sqrt(nx*nx + ny*ny)
            if l > 1e-8:
                nx /= l; ny /= l
                dot = nx * seg_normals[i][0] + ny * seg_normals[i][1]
                if dot > 0.1:
                    miter_scale = 1.0 / dot
                    if miter_scale > 2.0:
                        miter_scale = 2.0
                    nx *= miter_scale; ny *= miter_scale
            else:
                nx, ny = seg_normals[i]
        normals.append((nx * half_width, ny * half_width))

    verts = []
    for i in range(n - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i+1]
        nx0, ny0 = normals[i]; nx1, ny1 = normals[i+1]
        d0 = dists[i]; d1 = dists[i+1]
        # Two triangles forming a quad segment
        # Left side = v=-1, right side = v=+1
        verts.extend([
            (x0 + nx0, y0 + ny0, d0, -1.0), (x0 - nx0, y0 - ny0, d0, 1.0), (x1 + nx1, y1 + ny1, d1, -1.0),
            (x0 - nx0, y0 - ny0, d0, 1.0), (x1 - nx1, y1 - ny1, d1, 1.0), (x1 + nx1, y1 + ny1, d1, -1.0),
        ])
    return verts


# ===========================================================================
#  v7: Trees in green areas (parks, grass, wood, etc.)
# ===========================================================================

_GREEN_CLASSES = frozenset({
    "park", "grass", "cemetery", "pitch", "wood", "forest",
    "scrub", "garden", "recreation_ground", "nature_reserve",
    "meadow", "village_green", "playground",
})


def _extract_green_polygons(tile_cache, visible_keys, tile_px):
    """Extract polygons from landuse/landcover layers that represent green areas.
    Returns list of (centroid_wx, centroid_wy, approx_area_sq_px) for tree placement."""
    greens = []
    for tkey in visible_keys:
        mvt = tile_cache.get(tkey)
        if mvt is None:
            continue
        tz, tx_w, ty = tkey
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)

        for ln, layer in mvt.layers.items():
            lnl = ln.lower()
            if not ("landuse" in lnl or "landcover" in lnl or "land" in lnl):
                continue
            ext = layer.extent or MVT_EXTENT
            inv_ext = tile_px / ext
            for feat in layer.features:
                if feat.geom_type != 3:  # POLYGON only
                    continue
                fclass = str(feat.properties.get("class", feat.properties.get("type", ""))).lower()
                if fclass not in _GREEN_CLASSES:
                    continue
                rings = _get_rings(feat, ext)
                if not rings:
                    continue
                ring = rings[0]
                if len(ring) < 3:
                    continue
                # Compute centroid and area
                sum_x = sum_y = area2 = 0.0
                for i in range(len(ring) - 1):
                    x0, y0 = ring[i]; x1, y1 = ring[i+1]
                    cross = x0 * y1 - x1 * y0
                    area2 += cross
                    sum_x += (x0 + x1) * cross
                    sum_y += (y0 + y1) * cross
                if abs(area2) < 1e-6:
                    continue
                cx_t = sum_x / (3.0 * area2) * inv_ext + tile_ox
                cy_t = sum_y / (3.0 * area2) * inv_ext + tile_oy
                area_px = abs(area2) * 0.5 * inv_ext * inv_ext
                # Also collect polygon boundary points for tree scatter
                poly_pts = [(tile_ox + px * inv_ext, tile_oy + py * inv_ext) for px, py in ring]
                greens.append((cx_t, cy_t, area_px, poly_pts, fclass))
    return greens


def _scatter_trees_in_polygon(poly_pts, area_px, mpp, density=0.002, max_trees=40, rng_seed=42):
    """Scatter tree positions inside a polygon using rejection sampling.
    density: trees per square meter of area.
    Returns list of (wx, wy) world-pixel positions."""
    import random
    rng = random.Random(rng_seed + int(poly_pts[0][0] * 7 + poly_pts[0][1] * 13))

    area_m2 = area_px * mpp * mpp
    n_target = min(max_trees, max(1, int(area_m2 * density)))

    # Bounding box
    xs = [p[0] for p in poly_pts]; ys = [p[1] for p in poly_pts]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)

    # Simple point-in-polygon (ray casting)
    def _pip(px, py):
        inside = False
        j = len(poly_pts) - 1
        for i in range(len(poly_pts)):
            xi, yi = poly_pts[i]; xj, yj = poly_pts[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    trees = []
    attempts = 0
    while len(trees) < n_target and attempts < n_target * 10:
        attempts += 1
        px = rng.uniform(xmin, xmax)
        py = rng.uniform(ymin, ymax)
        if _pip(px, py):
            trees.append((px, py))
    return trees


def _build_fp_trees(tile_cache, visible_keys, tile_px, mpp, style):
    """Build low-poly tree geometry for immersive mode.
    Each tree = truncated cone trunk + two sphere-ish canopy cones.
    Returns float32 array with building-compatible layout:
    x, y, z, nx, ny, nz, r, g, b, a (10 floats per vert)."""
    greens = _extract_green_polygons(tile_cache, visible_keys, tile_px)
    if not greens:
        return np.empty((0, 10), dtype='f4')

    all_verts = []
    bg = style.bg
    bg_lum = (bg.red() + bg.green() + bg.blue()) / (3.0 * 255.0)
    is_dark = bg_lum < 0.3

    # Tree colors — vary by style
    if is_dark:
        trunk_c = (0.18, 0.12, 0.06, 1.0)
        leaf_colors = [
            (0.06, 0.22, 0.08, 0.92),
            (0.08, 0.28, 0.06, 0.92),
            (0.05, 0.18, 0.10, 0.92),
            (0.10, 0.25, 0.05, 0.92),
        ]
    else:
        trunk_c = (0.35, 0.22, 0.10, 1.0)
        leaf_colors = [
            (0.15, 0.50, 0.12, 0.92),
            (0.20, 0.55, 0.10, 0.92),
            (0.12, 0.42, 0.18, 0.92),
            (0.25, 0.52, 0.08, 0.92),
        ]

    import random
    n_seg = 6  # hexagonal cross-section for trunks/canopy

    def _add_cone(cx, cy, base_z, top_z, base_r, top_r, color):
        """Add a truncated cone (or full cone if top_r=0)."""
        r, g, b, a = color
        verts = []
        for i in range(n_seg):
            a0 = 2.0 * math.pi * i / n_seg
            a1 = 2.0 * math.pi * (i + 1) / n_seg
            cos0, sin0 = math.cos(a0), math.sin(a0)
            cos1, sin1 = math.cos(a1), math.sin(a1)

            # Bottom ring
            bx0 = cx + cos0 * base_r; by0 = cy + sin0 * base_r
            bx1 = cx + cos1 * base_r; by1 = cy + sin1 * base_r
            # Top ring
            tx0 = cx + cos0 * top_r; ty0 = cy + sin0 * top_r
            tx1 = cx + cos1 * top_r; ty1 = cy + sin1 * top_r

            # Normal (approximate — outward radial)
            nmx = (cos0 + cos1) * 0.5
            nmy = (sin0 + sin1) * 0.5
            nl = math.sqrt(nmx*nmx + nmy*nmy)
            if nl > 0.001: nmx /= nl; nmy /= nl
            nmz = (base_r - top_r) / max(abs(top_z - base_z), 0.01) * 0.3

            # Two triangles for the side quad
            verts.extend([
                (bx0, by0, base_z, nmx, nmy, nmz, r, g, b, a),
                (bx1, by1, base_z, nmx, nmy, nmz, r, g, b, a),
                (tx1, ty1, top_z,  nmx, nmy, nmz, r, g, b, a),
                (bx0, by0, base_z, nmx, nmy, nmz, r, g, b, a),
                (tx1, ty1, top_z,  nmx, nmy, nmz, r, g, b, a),
                (tx0, ty0, top_z,  nmx, nmy, nmz, r, g, b, a),
            ])

        # Top cap (if top_r > 0)
        if top_r > 0.01:
            for i in range(n_seg):
                a0 = 2.0 * math.pi * i / n_seg
                a1 = 2.0 * math.pi * (i + 1) / n_seg
                verts.extend([
                    (cx, cy, top_z, 0, 0, 1, r, g, b, a),
                    (cx + math.cos(a0)*top_r, cy + math.sin(a0)*top_r, top_z, 0, 0, 1, r, g, b, a),
                    (cx + math.cos(a1)*top_r, cy + math.sin(a1)*top_r, top_z, 0, 0, 1, r, g, b, a),
                ])
        return verts

    tree_count = 0
    max_total_trees = 300  # performance cap

    for _cx, _cy, area_px, poly_pts, fclass in greens:
        if tree_count >= max_total_trees:
            break

        # Density varies by class
        if fclass in ("wood", "forest", "scrub"):
            dens = 0.004
        elif fclass in ("park", "garden", "recreation_ground", "village_green"):
            dens = 0.0015
        elif fclass in ("cemetery",):
            dens = 0.001
        else:
            dens = 0.001

        positions = _scatter_trees_in_polygon(poly_pts, area_px, mpp, density=dens,
                                               max_trees=min(40, max_total_trees - tree_count))
        rng = random.Random(int(_cx * 7 + _cy * 13))

        for wx, wy in positions:
            if tree_count >= max_total_trees:
                break
            tree_count += 1

            # Randomize tree dimensions (in meters, converted to world-px)
            trunk_h_m = rng.uniform(2.0, 5.0)
            trunk_r_m = rng.uniform(0.15, 0.3)
            canopy_h_m = rng.uniform(3.0, 7.0)
            canopy_r_m = rng.uniform(1.5, 3.5)

            # Convert to world pixels
            inv_mpp = 1.0 / max(mpp, 0.001)
            trunk_h = trunk_h_m * inv_mpp
            trunk_r = trunk_r_m * inv_mpp
            canopy_h = canopy_h_m * inv_mpp
            canopy_r = canopy_r_m * inv_mpp

            leaf_c = leaf_colors[rng.randint(0, len(leaf_colors)-1)]

            # Trunk: truncated cone from ground to canopy base
            all_verts.extend(_add_cone(wx, wy, 0, trunk_h, trunk_r, trunk_r * 0.7, trunk_c))

            # Lower canopy: wide cone
            all_verts.extend(_add_cone(wx, wy, trunk_h * 0.6, trunk_h + canopy_h * 0.6,
                                       canopy_r, canopy_r * 0.5, leaf_c))

            # Upper canopy: narrower cone on top
            leaf_c2 = (min(1.0, leaf_c[0]+0.03), min(1.0, leaf_c[1]+0.05),
                       leaf_c[2], leaf_c[3])
            all_verts.extend(_add_cone(wx, wy, trunk_h + canopy_h * 0.3,
                                       trunk_h + canopy_h,
                                       canopy_r * 0.6, canopy_r * 0.05, leaf_c2))

    if not all_verts:
        return np.empty((0, 10), dtype='f4')
    return np.array(all_verts, dtype='f4')


# ===========================================================================
#  v7: Route path geometry for immersive mode
# ===========================================================================

def _build_fp_route(route_geometry, zoom, mpp):
    """Convert route lat/lon points to nav-style glowing ribbon geometry.
    Returns float32 array: x, y, z, r, g, b, a, u_along, v_across (9 floats)."""
    if not route_geometry or len(route_geometry) < 2:
        return np.empty((0, 9), dtype='f4')

    wp = [(_lon_to_wx(lon, zoom), _lat_to_wy(lat, zoom)) for lat, lon in route_geometry]
    cum_dist = [0.0]
    for i in range(1, len(wp)):
        dx = wp[i][0] - wp[i - 1][0]
        dy = wp[i][1] - wp[i - 1][1]
        cum_dist.append(cum_dist[-1] + math.sqrt(dx * dx + dy * dy))

    all_verts = []

    glow_hw = (7.0 / max(mpp, 0.001)) * 0.5
    core_hw = (4.0 / max(mpp, 0.001)) * 0.5

    # Outer glow layer
    glow = _thicken_line_uv(wp, glow_hw, cum_dist)
    for px, py, u_along, v_across in glow:
        all_verts.append((px, py, 0.22, 0.10, 0.55, 1.00, 0.30, u_along, v_across))

    # Core ribbon
    core = _thicken_line_uv(wp, core_hw, cum_dist)
    for px, py, u_along, v_across in core:
        all_verts.append((px, py, 0.28, 0.18, 0.68, 1.00, 0.95, u_along, v_across))

    return np.array(all_verts, dtype='f4') if all_verts else np.empty((0, 9), dtype='f4')


def _make_sky_sphere():
    """Create a simple sphere for the sky dome. Returns float32 array of xyz positions."""
    verts = []
    stacks = 16
    slices = 24
    for i in range(stacks):
        phi0 = math.pi * i / stacks - math.pi / 2
        phi1 = math.pi * (i + 1) / stacks - math.pi / 2
        for j in range(slices):
            theta0 = 2 * math.pi * j / slices
            theta1 = 2 * math.pi * (j + 1) / slices
            # Four corners
            p00 = (math.cos(phi0)*math.cos(theta0), math.sin(phi0), math.cos(phi0)*math.sin(theta0))
            p10 = (math.cos(phi1)*math.cos(theta0), math.sin(phi1), math.cos(phi1)*math.sin(theta0))
            p01 = (math.cos(phi0)*math.cos(theta1), math.sin(phi0), math.cos(phi0)*math.sin(theta1))
            p11 = (math.cos(phi1)*math.cos(theta1), math.sin(phi1), math.cos(phi1)*math.sin(theta1))
            verts.extend([p00, p10, p11, p00, p11, p01])
    return np.array(verts, dtype='f4')


def _fp_perspective(fov_deg, aspect, near, far):
    """Build a perspective projection matrix (column-major for GL)."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype='f4')
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _fp_look_at(eye, center, up):
    """Build a look-at view matrix."""
    f = np.array(center, dtype='f4') - np.array(eye, dtype='f4')
    f = f / np.linalg.norm(f)
    u = np.array(up, dtype='f4')
    s = np.cross(f, u)
    s_len = np.linalg.norm(s)
    if s_len > 1e-6:
        s = s / s_len
    else:
        s = np.array([1, 0, 0], dtype='f4')
    u = np.cross(s, f)
    m = np.eye(4, dtype='f4')
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.eye(4, dtype='f4')
    t[0, 3] = -eye[0]
    t[1, 3] = -eye[1]
    t[2, 3] = -eye[2]
    return m @ t


def _fp_look_at_rot(eye, center, up):
    """Build a rotation-only view matrix (no eye translation baked in).
    Used with camera-relative rendering to avoid float32 jitter."""
    f = np.array(center, dtype='f4') - np.array(eye, dtype='f4')
    f = f / np.linalg.norm(f)
    u = np.array(up, dtype='f4')
    s = np.cross(f, u)
    s_len = np.linalg.norm(s)
    if s_len > 1e-6:
        s = s / s_len
    else:
        s = np.array([1, 0, 0], dtype='f4')
    u = np.cross(s, f)
    m = np.eye(4, dtype='f4')
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    return m


# ===========================================================================
#  Sun position calculator
# ===========================================================================

def _sun_position(lat, lon, dt=None):
    if dt is None:
        dt = datetime.datetime.now(datetime.timezone.utc)
    doy = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    gamma = 2.0 * math.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2*gamma) + 0.000907 * math.sin(2*gamma))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                     - 0.014615 * math.cos(2*gamma) - 0.04089 * math.sin(2*gamma))
    solar_time = hour_utc * 60.0 + eqt + lon * 4.0
    hour_angle = math.radians((solar_time / 4.0) - 180.0)
    lat_r = math.radians(lat)
    sin_elev = (math.sin(lat_r) * math.sin(decl) +
                math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle))
    elevation = math.degrees(math.asin(max(-1, min(1, sin_elev))))
    cos_elev = math.cos(math.radians(elevation))
    if cos_elev > 0.001:
        cos_azi = (math.sin(decl) - math.sin(lat_r) * sin_elev) / (math.cos(lat_r) * cos_elev)
        cos_azi = max(-1, min(1, cos_azi))
        azimuth = math.degrees(math.acos(cos_azi))
        if hour_angle > 0:
            azimuth = 360.0 - azimuth
    else:
        azimuth = 0.0
    return azimuth, elevation


# ===========================================================================
#  Settings Panel
# ===========================================================================

class SettingsPanel(QFrame):
    value_changed = Signal()
    toggle_changed = Signal(str, bool)  # key, state

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            "SettingsPanel { background: #12141e; border-left: 1px solid #2a3050; }")
        self._sliders = {}; self._labels = {}; self._checks = {}
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 14, 12, 14); main_layout.setSpacing(6)

        title = QLabel("Map Controls")
        title.setStyleSheet("color: #c0c8e0; font: bold 13px 'Segoe UI', sans-serif; padding-bottom: 4px;")
        main_layout.addWidget(title)

        # --- Toggle section ---
        sep1 = QLabel("Toggles")
        sep1.setStyleSheet("color: #6070a0; font: bold 9px monospace; padding-top: 6px;")
        main_layout.addWidget(sep1)

        toggle_defs = [
            ("buildings_3d", "3D Buildings", True),
            ("labels_visible", "Labels", True),
            ("shadows", "Building shadows", True),
            ("tiles_visible", "Show tiles", True),
            ("poi_markers", "POI markers (O)", False),
        ]
        for key, text, default in toggle_defs:
            cb = QCheckBox(text)
            cb.setChecked(default)
            cb.setStyleSheet(
                "QCheckBox { color: #a0a8c0; font: 10px 'Segoe UI', sans-serif; spacing: 6px; }"
                "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #4060a0; "
                "border-radius: 3px; background: #1a1e2e; }"
                "QCheckBox::indicator:checked { background: #4080d0; border-color: #5090e0; }"
                "QCheckBox::indicator:hover { border-color: #6090d0; }")
            cb.toggled.connect(lambda v, k=key: self._on_toggle(k, v))
            main_layout.addWidget(cb)
            self._checks[key] = cb

        # --- Slider section ---
        sep2 = QLabel("Parameters")
        sep2.setStyleSheet("color: #6070a0; font: bold 9px monospace; padding-top: 10px;")
        main_layout.addWidget(sep2)

        # --- Isochrone section ---
        sep_iso = QLabel("Isochrone (Y)")
        sep_iso.setStyleSheet("color: #6070a0; font: bold 9px monospace; padding-top: 10px;")
        main_layout.addWidget(sep_iso)

        # Profile combo
        iso_row1 = QHBoxLayout(); iso_row1.setSpacing(6)
        iso_lbl1 = QLabel("Profile"); iso_lbl1.setFixedWidth(60)
        iso_lbl1.setStyleSheet("color: #9098b0; font: 9px monospace;")
        iso_row1.addWidget(iso_lbl1)
        self._iso_combo = QComboBox()
        self._iso_combo.addItems(["driving", "walking", "cycling"])
        self._iso_combo.setStyleSheet(
            "QComboBox { background: #1e2230; color: #a0b0d0; border: 1px solid #3a4060; "
            "border-radius: 4px; padding: 2px 6px; font: 9px monospace; }"
            "QComboBox::drop-down { border: none; width: 16px; }"
            "QComboBox QAbstractItemView { background: #1a1e2e; color: #a0b0d0; "
            "selection-background-color: #3060a0; border: 1px solid #3a4060; }")
        iso_row1.addWidget(self._iso_combo, 1)
        main_layout.addLayout(iso_row1)

        # Time spinboxes
        iso_row2 = QHBoxLayout(); iso_row2.setSpacing(4)
        iso_lbl2 = QLabel("Minutes"); iso_lbl2.setFixedWidth(60)
        iso_lbl2.setStyleSheet("color: #9098b0; font: 9px monospace;")
        iso_row2.addWidget(iso_lbl2)

        spin_style = (
            "QSpinBox { background: #1e2230; color: #80c0ff; border: 1px solid #3a4060; "
            "border-radius: 3px; padding: 1px 4px; font: 9px monospace; min-width: 38px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 12px; }")

        self._iso_spin1 = QSpinBox(); self._iso_spin1.setRange(1, 60); self._iso_spin1.setValue(5)
        self._iso_spin1.setStyleSheet(spin_style); iso_row2.addWidget(self._iso_spin1)
        self._iso_spin2 = QSpinBox(); self._iso_spin2.setRange(1, 60); self._iso_spin2.setValue(10)
        self._iso_spin2.setStyleSheet(spin_style); iso_row2.addWidget(self._iso_spin2)
        self._iso_spin3 = QSpinBox(); self._iso_spin3.setRange(1, 60); self._iso_spin3.setValue(15)
        self._iso_spin3.setStyleSheet(spin_style); iso_row2.addWidget(self._iso_spin3)
        main_layout.addLayout(iso_row2)

        # Slider parameters section header
        sep2b = QLabel("Map Parameters")
        sep2b.setStyleSheet("color: #6070a0; font: bold 9px monospace; padding-top: 10px;")
        main_layout.addWidget(sep2b)

        slider_defs = [
            ("bld_height", "Building height", 1, 20, 4, 1),
            ("bld_scale", "Height exaggeration", 1, 30, 8, 100),
            ("bld_min_zoom", "Building min zoom", 13, 20, 16, 1),
            ("bld_max", "Max buildings/tile", 50, 2000, 500, 1),
            ("bld_opacity", "Building opacity", 10, 100, 100, 100),
            ("bld_win_glow", "Window glow", 0, 200, 100, 100),
            ("bld_ao", "Ambient occlusion", 0, 200, 100, 100),
            ("bld_tint_r", "Bld tint R", 50, 150, 100, 100),
            ("bld_tint_g", "Bld tint G", 50, 150, 100, 100),
            ("bld_tint_b", "Bld tint B", 50, 150, 100, 100),
            ("max_pitch", "Max pitch", 10, 80, 60, 1),
            ("max_tiles", "Max tiles/axis", 4, 16, 8, 1),
            ("render_conc", "Render threads", 1, 16, 10, 1),
            ("heat_filter", "Heatmap level filter", 0, 90, 0, 100),
            ("heat_intensity", "Heatmap intensity", 10, 200, 60, 100),
            ("heat_radius", "Heatmap radius", 5, 200, 50, 1),
        ]
        for key, label_text, smin, smax, sdef, div in slider_defs:
            row = QHBoxLayout(); row.setSpacing(6)
            lbl = QLabel(label_text); lbl.setFixedWidth(130)
            lbl.setStyleSheet("color: #9098b0; font: 9px monospace;"); row.addWidget(lbl)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(smin); slider.setMaximum(smax); slider.setValue(sdef)
            slider.setFixedHeight(16)
            slider.setStyleSheet(
                "QSlider::groove:horizontal{background:#2a2e3e;height:4px;border-radius:2px;}"
                "QSlider::handle:horizontal{background:#6080c0;width:10px;height:10px;margin:-3px 0;border-radius:5px;}"
                "QSlider::sub-page:horizontal{background:#4060a0;border-radius:2px;}")
            row.addWidget(slider, 1)
            val_lbl = QLabel(self._format_val(sdef, div)); val_lbl.setFixedWidth(36)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("color: #80c0ff; font: 9px monospace;"); row.addWidget(val_lbl)
            main_layout.addLayout(row)
            self._sliders[key] = slider; self._labels[key] = val_lbl
            slider.valueChanged.connect(lambda v, k=key, d=div: self._on_slider(k, v, d))

        main_layout.addStretch(1)

        # --- Keyboard legend ---
        sep3 = QLabel("Keyboard")
        sep3.setStyleSheet("color: #6070a0; font: bold 9px monospace; padding-top: 10px;")
        main_layout.addWidget(sep3)
        shortcuts = [
            "S  cycle style", "B  toggle 3D", "T  reset tilt",
            "R  reset view", "N  route mode", "C  clear route",
            "M  route profile", "L  shadows", "H  heatmap",
            "O  POI markers", "V  toggle tiles", "F  toggle labels", "+/-  zoom",
            "/  search destination", "Y  isochrone (place)",
            "I  immersive mode", "G  car mode (immersive)",
            "P  toggle traffic (immersive)",
            "\u2191\u2193 select POI  \u23CE navigate",
            "ESC  exit immersive/car",
        ]
        for s in shortcuts:
            sl = QLabel(s)
            sl.setStyleSheet("color: #707898; font: 8px monospace;")
            main_layout.addWidget(sl)

        self.setFixedWidth(260)

    def _format_val(self, v, div):
        if div == 1: return str(v)
        elif div >= 100: return f"{v/div:.2f}"
        else: return f"{v/div:.1f}"
    def _on_slider(self, key, val, div):
        self._labels[key].setText(self._format_val(val, div)); self.value_changed.emit()
    def _on_toggle(self, key, val):
        self.toggle_changed.emit(key, val)
    def get(self, key):
        s = self._sliders.get(key)
        return float(s.value()) if s else 0.0
    def get_check(self, key):
        cb = self._checks.get(key)
        return cb.isChecked() if cb else True
    def set_check(self, key, val):
        cb = self._checks.get(key)
        if cb:
            cb.blockSignals(True)
            cb.setChecked(val)
            cb.blockSignals(False)


# ===========================================================================
#  Numpy-vectorized building VBO construction
# ===========================================================================

def _build_buildings_numpy(building_cache, visible_bld_keys, tile_px,
                           default_h, h_exag, z_scale, pitch_factor,
                           max_per_tile, wall_color, roof_color):
    """Numpy-vectorized building VBO. Returns flat float32 array.
    Vertex: x,y, h, nh, nx,ny,nz, r,g,b,a (11 floats)."""
    all_verts = []
    wr, wg, wb, wa = wall_color
    rr, rg, rb, ra = roof_color

    for tkey in visible_bld_keys:
        bldgs = building_cache.get(tkey)
        if not bldgs: continue
        if len(bldgs) > max_per_tile:
            bldgs = sorted(bldgs, key=lambda b: b["height"], reverse=True)[:max_per_tile]
        tx_w, ty = tkey[1], tkey[2]
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)

        for bld in bldgs:
            ext = bld["extent"]; s_px = tile_px / ext
            bh = bld["height"]
            h_px = (bh if bh > 0 else default_h) * h_exag * z_scale
            h_px = max(1.0, min(h_px, 200.0)) * pitch_factor

            for ring in bld["rings"]:
                if len(ring) < 3: continue
                pts = np.array(ring, dtype='f4')
                pts[:, 0] = tile_ox + pts[:, 0] * s_px
                pts[:, 1] = tile_oy + pts[:, 1] * s_px
                n_pts = len(pts)

                p0 = pts[:-1]; p1 = pts[1:]
                edges = p1 - p0
                edge_lens = np.sqrt(edges[:, 0]**2 + edges[:, 1]**2)
                valid = edge_lens > 0.01
                if not np.any(valid): continue
                p0v = p0[valid]; p1v = p1[valid]; ev = edges[valid]; elv = edge_lens[valid]
                nx = -ev[:, 1] / elv; ny = ev[:, 0] / elv
                nw = len(p0v)

                wall_data = np.empty((nw * 6, 11), dtype='f4')
                wall_data[0::6, 0] = p0v[:, 0]; wall_data[0::6, 1] = p0v[:, 1]
                wall_data[0::6, 2] = 0.0;       wall_data[0::6, 3] = 0.0
                wall_data[1::6, 0] = p1v[:, 0]; wall_data[1::6, 1] = p1v[:, 1]
                wall_data[1::6, 2] = 0.0;       wall_data[1::6, 3] = 0.0
                wall_data[2::6, 0] = p1v[:, 0]; wall_data[2::6, 1] = p1v[:, 1]
                wall_data[2::6, 2] = h_px;      wall_data[2::6, 3] = 1.0
                wall_data[3::6, 0] = p0v[:, 0]; wall_data[3::6, 1] = p0v[:, 1]
                wall_data[3::6, 2] = 0.0;       wall_data[3::6, 3] = 0.0
                wall_data[4::6, 0] = p1v[:, 0]; wall_data[4::6, 1] = p1v[:, 1]
                wall_data[4::6, 2] = h_px;      wall_data[4::6, 3] = 1.0
                wall_data[5::6, 0] = p0v[:, 0]; wall_data[5::6, 1] = p0v[:, 1]
                wall_data[5::6, 2] = h_px;      wall_data[5::6, 3] = 1.0
                for k in range(6):
                    wall_data[k::6, 4] = nx; wall_data[k::6, 5] = ny; wall_data[k::6, 6] = 0.0
                wall_data[:, 7] = wr; wall_data[:, 8] = wg
                wall_data[:, 9] = wb; wall_data[:, 10] = wa
                all_verts.append(wall_data)

                if n_pts >= 3:
                    roof_pts = pts
                    if n_pts >= 4:
                        d = pts[-1] - pts[0]
                        if abs(d[0]) < 0.01 and abs(d[1]) < 0.01:
                            roof_pts = pts[:-1]
                    n_roof = len(roof_pts)
                    if n_roof >= 3:
                        nt = n_roof - 2
                        roof_data = np.empty((nt * 3, 11), dtype='f4')
                        i0 = np.zeros(nt, dtype=np.int32)
                        i1 = np.arange(1, nt + 1, dtype=np.int32)
                        i2 = np.arange(2, nt + 2, dtype=np.int32)
                        roof_data[0::3, 0] = roof_pts[i0, 0]; roof_data[0::3, 1] = roof_pts[i0, 1]
                        roof_data[1::3, 0] = roof_pts[i1, 0]; roof_data[1::3, 1] = roof_pts[i1, 1]
                        roof_data[2::3, 0] = roof_pts[i2, 0]; roof_data[2::3, 1] = roof_pts[i2, 1]
                        roof_data[:, 2] = h_px; roof_data[:, 3] = 1.0
                        roof_data[:, 4] = 0.0; roof_data[:, 5] = 0.0; roof_data[:, 6] = 1.0
                        roof_data[:, 7] = rr; roof_data[:, 8] = rg
                        roof_data[:, 9] = rb; roof_data[:, 10] = ra
                        all_verts.append(roof_data)

    if not all_verts:
        return np.empty((0, 11), dtype='f4')
    return np.concatenate(all_verts, axis=0)


def _build_shadow_numpy(building_cache, visible_bld_keys, tile_px,
                         default_h, h_exag, z_scale, pitch_factor,
                         max_per_tile, sun_azimuth, sun_elevation):
    if sun_elevation <= 0:
        return np.empty((0, 3), dtype='f4')
    shadow_len = 1.0 / max(math.tan(math.radians(max(sun_elevation, 2.0))), 0.04)
    shadow_len = min(shadow_len, 8.0)
    all_verts = []
    for tkey in visible_bld_keys:
        bldgs = building_cache.get(tkey)
        if not bldgs: continue
        if len(bldgs) > max_per_tile:
            bldgs = sorted(bldgs, key=lambda b: b["height"], reverse=True)[:max_per_tile]
        tx_w, ty = tkey[1], tkey[2]
        tile_ox = float(tx_w * tile_px)
        tile_oy = float(ty * tile_px)
        for bld in bldgs:
            ext = bld["extent"]; s_px = tile_px / ext
            bh = bld["height"]
            h_px = (bh if bh > 0 else default_h) * h_exag * z_scale
            h_px = max(1.0, min(h_px, 200.0)) * pitch_factor
            for ring in bld["rings"]:
                if len(ring) < 3: continue
                pts = np.array(ring, dtype='f4')
                pts[:, 0] = tile_ox + pts[:, 0] * s_px
                pts[:, 1] = tile_oy + pts[:, 1] * s_px
                roof_pts = pts
                n_pts = len(pts)
                if n_pts >= 4:
                    d = pts[-1] - pts[0]
                    if abs(d[0]) < 0.01 and abs(d[1]) < 0.01:
                        roof_pts = pts[:-1]
                n_roof = len(roof_pts)
                if n_roof < 3: continue
                nt = n_roof - 2
                shadow_data = np.empty((nt * 3, 3), dtype='f4')
                i0 = np.zeros(nt, dtype=np.int32)
                i1 = np.arange(1, nt + 1, dtype=np.int32)
                i2 = np.arange(2, nt + 2, dtype=np.int32)
                shadow_data[0::3, 0] = roof_pts[i0, 0]
                shadow_data[0::3, 1] = roof_pts[i0, 1]
                shadow_data[1::3, 0] = roof_pts[i1, 0]
                shadow_data[1::3, 1] = roof_pts[i1, 1]
                shadow_data[2::3, 0] = roof_pts[i2, 0]
                shadow_data[2::3, 1] = roof_pts[i2, 1]
                shadow_data[:, 2] = h_px * shadow_len
                all_verts.append(shadow_data)
    if not all_verts:
        return np.empty((0, 3), dtype='f4')
    return np.concatenate(all_verts, axis=0)


def _build_route_vbo(waypoints_world, line_width=6.0):
    if len(waypoints_world) < 2:
        return np.empty((0, 4), dtype='f4')
    pts = np.array(waypoints_world, dtype='f4')
    n = len(pts)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_dist[-1]
    if total < 0.001:
        return np.empty((0, 4), dtype='f4')
    progress = cum_dist / total
    normals = np.zeros((n, 2), dtype='f4')
    for i in range(n):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i+1] - pts[i-1]
        l = math.sqrt(d[0]**2 + d[1]**2)
        if l > 0.001:
            normals[i] = [-d[1]/l, d[0]/l]
    hw = line_width * 0.5
    verts = np.empty(((n - 1) * 6, 4), dtype='f4')
    for i in range(n - 1):
        p0 = pts[i]; p1 = pts[i+1]
        n0 = normals[i]; n1 = normals[i+1]
        pr0 = progress[i]; pr1 = progress[i+1]
        l0 = p0 + n0 * hw; r0 = p0 - n0 * hw
        l1 = p1 + n1 * hw; r1 = p1 - n1 * hw
        base = i * 6
        verts[base+0] = [l0[0], l0[1], 1.0, pr0]
        verts[base+1] = [r0[0], r0[1], -1.0, pr0]
        verts[base+2] = [l1[0], l1[1], 1.0, pr1]
        verts[base+3] = [r0[0], r0[1], -1.0, pr0]
        verts[base+4] = [r1[0], r1[1], -1.0, pr1]
        verts[base+5] = [l1[0], l1[1], 1.0, pr1]
    return verts


def _build_heatmap_vbo(points_world, radius=40.0):
    if not points_world:
        return np.empty((0, 5), dtype='f4')
    n = len(points_world)
    verts = np.empty((n * 6, 5), dtype='f4')
    for i, (wx, wy, weight) in enumerate(points_world):
        base = i * 6
        for j, (qx, qy) in enumerate([(-1,-1),(1,-1),(1,1), (-1,-1),(1,1),(-1,1)]):
            verts[base+j] = [wx, wy, qx, qy, weight]
    return verts


# ===========================================================================
#  MapboxWidget — v6 Performance Rewrite
# ===========================================================================

class MapboxWidget(QWidget):

    STYLES = [
        "mapbox/dark-v10", "mapbox/streets-v12", "mapbox/satellite-v9",
        "mapbox/outdoors-v12", "mapbox/light-v11", "mapbox/navigation-night-v1",
        "custom/cyberpunk", "custom/blueprint", "custom/sepia", "custom/nord",
    ]
    TILESETS = [
        "mapbox.mapbox-streets-v8", "mapbox.mapbox-streets-v8",
        "mapbox.mapbox-terrain-v2,mapbox.mapbox-streets-v8",
        "mapbox.mapbox-terrain-v2,mapbox.mapbox-streets-v8",
        "mapbox.mapbox-streets-v8", "mapbox.mapbox-streets-v8",
        "mapbox.mapbox-streets-v8", "mapbox.mapbox-streets-v8",
        "mapbox.mapbox-streets-v8", "mapbox.mapbox-streets-v8",
    ]

    def __init__(self, token="", style="mapbox/dark-v10",
                 lat=37.7749, lon=-122.4194, zoom=13.0,
                 width=900, height=600, parent=None):
        super().__init__(parent)
        self.token = token or os.environ.get("MAPBOX_TOKEN", "")
        self._style_idx = self.STYLES.index(style) if style in self.STYLES else 0
        self._style = self.STYLES[self._style_idx]

        self._zoom = _clamp_zoom(zoom); self._target_zoom = self._zoom
        self._cx = _lon_to_wx(lon, self._zoom)
        self._cy = _lat_to_wy(lat, self._zoom)
        self._pitch = 0.0; self._target_pitch = 0.0
        self._bearing = 0.0; self._target_bearing = 0.0

        # --- v9: Smooth camera animation ("flyTo") ---
        self._fly = FlyToAnimation()
        self._fly_last_tick = time.perf_counter()
        # Smooth scroll-zoom state (zoom toward cursor)
        self._scroll_zoom_target = self._zoom
        self._scroll_zoom_active = False
        self._scroll_anchor_lat = 0.0   # lat/lon under cursor (zoom-invariant)
        self._scroll_anchor_lon = 0.0
        self._scroll_anchor_sx = 0.0    # screen offset from center (bearing-rotated)
        self._scroll_anchor_sy = 0.0
        self._buildings_3d = True
        self._labels_visible = True  # Toggle with 'F' — show/hide map labels
        self._tiles_visible = True  # Toggle with 'V' — hides tile layers to show scene behind

        # --- Caches ---
        self._mvt_cache: OrderedDict = OrderedDict()   # (z,x,y) -> MVTTile
        self._mvt_cache_max = 350

        # v6: GPU geometry cache — tessellated VBOs per tile+style
        # Key: (z, x, y, style_name) -> {fill_vbo, fill_vao, fill_count,
        #                                  line_vbo, line_vao, line_count}
        self._geo_cache: OrderedDict = OrderedDict()
        self._geo_cache_max = 180

        self._building_cache: Dict = {}
        self._label_cache: Dict = {}
        self._building_cache_max = 300

        # --- Network ---
        self._pending: set = set()
        self._rendering: set = set()
        self._pending_replies: Dict = {}
        self._nam = QNetworkAccessManager(self)
        self._max_concurrent_net = 6  # hotfix: reduced from 12 to lower contention
        self._max_concurrent_render = 6  # hotfix: reduced from 10

        # --- Zoom-level tracking for tile stability ---
        self._last_tile_z = -1

        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._signals = _TileSignals()
        self._signals.tile_ready.connect(self._on_tile_rendered, Qt.QueuedConnection)
        self._signals.terrain_ready.connect(self._on_terrain_data_ready, Qt.QueuedConnection)

        # Deferred render queue
        self._render_queue: List = []
        self._render_queue_set: set = set()

        self._mvt_lock = threading.Lock()

        # --- GL state ---
        self._gl_ready = False
        self._ctx = None; self._fbo = None
        self._fbo_w = 0; self._fbo_h = 0
        self._gl_dirty = True
        self._cached_frame: Optional[QImage] = None
        self._raw_bytes = None

        # Building GPU data
        self._bld_vbo = None; self._bld_vao = None; self._bld_vert_count = 0
        self._bld_gpu_key = ()

        # Shadow GPU data
        self._shadow_vbo = None; self._shadow_vao = None; self._shadow_vert_count = 0
        self._shadow_gpu_key = ()
        self._shadows_enabled = True
        self._sun_azimuth = 0.0; self._sun_elevation = 45.0
        self._sun_time = None

        # Route data
        self._route_vbo = None; self._route_vao = None; self._route_vert_count = 0
        self._route_points = []
        self._route_world = []
        self._route_geometry = []
        self._route_mode = False
        self._route_color = (0.2, 0.6, 1.0, 0.85)
        self._route_dirty = True
        self._route_total_dist = 0.0
        self._route_duration = 0.0
        self._route_profile = "driving"
        self._route_pending = False
        self._route_reply = None

        # Heatmap data — extracted from real MVT tile data
        self._heatmap_vbo = None; self._heatmap_vao = None; self._heatmap_vert_count = 0
        self._heatmap_enabled = False
        self._heatmap_points_latlon = []  # [(lat, lon, weight), ...]
        self._heatmap_dirty = True
        self._heatmap_modes = ["poi", "traffic", "building"]
        self._heatmap_mode_idx = 0
        self._heatmap_mode = "poi"
        self._heatmap_generated = False
        self._heatmap_gen_lat = 0.0  # center lat when heatmap was generated
        self._heatmap_gen_lon = 0.0  # center lon when heatmap was generated
        self._heatmap_gen_zoom = 0.0  # zoom when heatmap was generated
        # --- Isochrone overlay ---
        self._iso_enabled = False
        self._iso_placing = False              # Y enters placement mode, click to place
        self._iso_profile = "driving"         # driving / walking / cycling
        self._iso_contours: List[Dict] = []   # [{minutes, color, polygon_latlon}, ...]
        self._iso_reply = None                # pending API request
        self._iso_center = (0.0, 0.0)         # (lat, lon) center of current isochrones
        self._iso_dirty = True                # need to redraw overlay
        self._iso_minutes = [5, 10, 15]       # time thresholds
        self._iso_colors = [
            QColor(67, 6, 206, 70),           # 5 min — purple
            QColor(4, 232, 19, 55),           # 10 min — green
            QColor(66, 134, 244, 40),         # 15 min — blue
        ]
        self._iso_outline_colors = [
            QColor(67, 6, 206, 180),
            QColor(4, 232, 19, 150),
            QColor(66, 134, 244, 120),
        ]
        # Traffic tile cache (separate from main MVT)
        self._traffic_cache: OrderedDict = OrderedDict()
        self._traffic_cache_max = 200
        self._traffic_pending: set = set()
        self._traffic_replies: Dict = {}

        # --- Immersive (first-person) mode ---
        self._immersive = False
        self._immersive_entering = False  # True when waiting for click to place camera
        self._fp_cx = 0.0  # camera world-pixel X
        self._fp_cy = 0.0  # camera world-pixel Y
        self._fp_yaw = 0.0    # degrees, 0 = north, CW positive
        self._fp_pitch_angle = 0.0   # degrees, positive = look up
        self._fp_eye_height = 5.0  # world-pixels (will be computed from meters)
        self._fp_move_speed = 3.0  # world-pixels per second (multiplied by dt in physics thread)
        self._fp_keys_held = set()  # WASD keys currently held
        self._fp_mouse_captured = False
        self._fp_mouse_last = None
        self._fp_zoom_at_enter = 16.0  # zoom level when entering
        self._fp_saved_cx = 0.0
        self._fp_saved_cy = 0.0
        self._fp_saved_zoom = 0.0
        self._fp_saved_pitch = 0.0
        self._fp_saved_bearing = 0.0
        # FP GPU resources
        self._fp_bldg_vbo = None; self._fp_bldg_vao = None; self._fp_bldg_count = 0
        self._fp_ground_vbo = None; self._fp_ground_vao = None; self._fp_ground_count = 0
        self._fp_water_vbo = None; self._fp_water_vao = None; self._fp_water_count = 0
        self._fp_road_vbo = None; self._fp_road_vao = None; self._fp_road_count = 0
        self._fp_sky_vbo = None; self._fp_sky_vao = None; self._fp_sky_count = 0
        self._fp_dirty = True
        self._fp_bldg_key = ()  # cache key for building VBO
        self._fp_geo_key = None  # (z, tile_x, tile_y) — triggers geometry rebuild on tile change
        # v7: Trees in green areas
        self._fp_tree_vbo = None; self._fp_tree_vao = None; self._fp_tree_count = 0
        self._trees_enabled = True  # toggled by panel (future) or always on
        # v7: Route in immersive mode
        self._fp_route_vbo = None; self._fp_route_vao = None; self._fp_route_count = 0
        self._fp_route_key = None  # cache key to detect route changes
        # FP shader programs (created lazily)
        self._fp_prog_bldg = None
        self._fp_prog_ground = None
        self._fp_prog_water = None
        self._fp_prog_road = None
        self._fp_prog_road_nav = None
        self._fp_prog_sky = None

        # --- Car mode (3rd-person driving within immersive) ---
        self._car_mode = False
        self._car_wx = 0.0           # car world-pixel X (float64)
        self._car_wy = 0.0           # car world-pixel Y (float64)
        self._car_yaw = 0.0          # heading degrees, 0=north CW
        self._car_speed = 0.0        # world-pixels per second
        self._car_max_speed = 0.0    # computed from mpp
        self._car_accel_rate = 0.0   # acceleration in wp/s²
        self._car_throttle = 0.0     # 0..1
        self._car_brake = 0.0        # 0..1
        self._car_steer = 0.0        # -1..1
        self._car_steer_angle = 0.0  # smoothed steering angle (degrees/s)
        self._car_cam_dist = 0.0     # camera distance behind (world px)
        self._car_cam_height = 0.0   # camera height (world px)
        self._car_cam_yaw_off = 0.0  # camera orbit offset deg
        self._car_mpp = 0.001        # meters per pixel at car
        self._car_last_tick = 0.0    # time.perf_counter() of last physics tick

        # --- Terrain DEM cache ---
        self._terrain_cache = {}     # (z, x, y) -> np.float32 elevation array
        self._terrain_pending = set()
        self._terrain_enabled = True
        self._terrain_tex = None     # moderngl.Texture (R32F heightmap)
        self._terrain_tex_dirty = True
        self._terrain_tex_bounds = (0.0, 0.0, 1.0, 1.0)  # (min_wx, min_wy, max_wx, max_wy)
        # Car GPU
        self._car_vbo = None; self._car_vao = None; self._car_vert_count = 0
        self._fp_prog_car = None

        # --- Traffic NPC system ---
        self._traffic_enabled = False
        self._traffic_cars: List[NPCCar] = []
        self._traffic_vbo = None; self._traffic_vao = None; self._traffic_vert_count = 0
        self._traffic_road_paths = []
        self._traffic_spawn_key = None  # cache key for road paths
        self._traffic_npc_model_cache = {}  # color_idx -> np.array
        self._traffic_rebuild_needed = True
        # Collision state
        self._collision_flash = 0.0  # screen flash on collision (0..1)
        self._collision_cooldown = 0.0  # seconds until next collision check
        self._collision_speed_penalty = 0.0  # speed reduction on collision

        # --- POI Service Markers ---
        self._poi_enabled = False            # toggled with 'O'
        self._poi_cache: List = []           # extracted POI dicts from MVT tiles
        self._poi_cache_key = None           # (z_int, cam_tx, cam_ty) for cache invalidation
        self._poi_visible_list: List = []    # nearby POIs visible in immersive mode
        # FP (immersive) POI GPU
        self._fp_poi_vbo = None; self._fp_poi_vao = None; self._fp_poi_count = 0
        self._fp_prog_poi = None
        self._fp_poi_rebuild = True
        # 2D map POI GPU
        self._poi_2d_vbo = None; self._poi_2d_vao = None; self._poi_2d_count = 0
        self._prog_poi_2d = None
        self._poi_2d_dirty = True
        # POI detail popup state
        self._poi_popup: Optional[Dict] = None   # currently shown POI detail dict
        self._poi_popup_timer = 0.0               # auto-dismiss timer (seconds)
        self._poi_popup_fade = 1.0                # fade animation (1.0 = full, 0 = gone)
        self._poi_popup_screen_pos = (0, 0)       # screen position for 2D popup
        self._poi_popup_geocode: Optional[Dict] = None   # geocoding result (address etc.)
        self._poi_popup_geocode_reply = None       # pending geocoding request
        self._poi_proximity_poi: Optional[Dict] = None   # last auto-proximity POI (avoid repeat)
        self._poi_proximity_cooldown = 0.0         # cooldown to avoid popup spam
        # POI detail enrichment — Wikipedia summary + photo
        self._poi_detail_cache: Dict[str, Dict] = {}     # name -> {summary, photo_url, photo_pixmap, ...}
        self._poi_detail_reply = None                     # pending Wikipedia API request
        self._poi_photo_reply = None                      # pending photo download
        self._poi_photo_pixmap: Optional[QPixmap] = None  # loaded photo pixmap for current popup
        self._poi_hover_poi: Optional[Dict] = None        # POI under mouse cursor (2D hover)
        self._poi_hover_timer = 0.0                       # hover dwell time before showing popup
        # POI selector (immersive) — cycle through nearby POIs with +/- and navigate with Enter
        self._poi_select_idx = -1                          # -1 = no selection, 0..N-1 = selected POI
        self._poi_sorted_list: List = []                   # distance-sorted copy for stable selection
        self._poi_select_active = False                    # whether selection mode is active
        # Search bar (/) — geocoding search + route to destination
        self._search_active = False                        # whether search bar is visible
        self._search_text = ""                             # current search input
        self._search_results: List[Dict] = []              # geocoding results
        self._search_selected = 0                          # selected result index
        self._search_reply = None                          # pending geocoding request
        self._search_cursor_blink = 0.0                    # cursor blink timer

        # PBO double-buffered readback
        self._pbo = [None, None]
        self._pbo_idx = 0
        self._pbo_frame_count = 0

        # --- Repaint coalescing ---
        self._tiles_arrived_since_paint = 0
        self._coalesce_timer = QTimer(self)
        self._coalesce_timer.setSingleShot(True)
        self._coalesce_timer.setInterval(8)  # v7: faster tile appearance (was 20)
        self._coalesce_timer.timeout.connect(self._coalesced_update)

        # --- Zoom velocity tracking ---
        self._zoom_velocity = 0.0
        self._prev_zoom = self._zoom

        # --- Drag state ---
        self._drag_start = None; self._drag_cx0 = 0.0; self._drag_cy0 = 0.0; self._drag_last = None
        self._drag_lat0 = 0.0; self._drag_lon0 = 0.0  # zoom-invariant anchor
        self._drag_zoom0 = 0.0  # zoom at drag start
        self._rdrag_start = None; self._rdrag_pitch0 = 0.0; self._rdrag_bearing0 = 0.0
        self._rdrag_cx0 = 0.0; self._rdrag_cy0 = 0.0
        self._rdrag_lat0 = 0.0; self._rdrag_lon0 = 0.0
        self._mdrag_start = None; self._mdrag_bearing0 = 0.0
        self._mdrag_cx0 = 0.0; self._mdrag_cy0 = 0.0
        self._mdrag_lat0 = 0.0; self._mdrag_lon0 = 0.0

        self._frame = 0

        # --- Widget setup ---
        self.setFixedSize(width, height)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        # Settings panel — created externally, referenced here
        self._panel = None  # Will be set via set_panel()
        self._panel_visible = True  # Panel is always accessible (side panel)

        # === v8 Streaming Pipeline State ===

        # Opt 1: Predictive Pre-Tessellation — "Loading Horizon"
        self._prefetch_tile_keys: set = set()  # tiles currently being pre-fetched
        self._last_prefetch_key = None         # debounce: last (z,tx,ty,yaw_bucket)

        # Opt 2: Time-Sliced GPU Upload Queue
        self._gpu_upload_queue = GPUUploadQueue()
        # Upload timer: drains 1 task per frame (~60 Hz)
        self._upload_timer = QTimer(self)
        self._upload_timer.timeout.connect(self._drain_gpu_upload)
        self._upload_timer.start(12)  # hotfix: steadier frame pacing than 8ms

        # Opt 3: Terrain Texture Array (slot-based)
        self._terrain_tex_slots = 16            # number of DEM tile slots
        self._terrain_tex_slot_size = 256       # DEM tile resolution
        self._terrain_tex_array = None          # moderngl.Texture (2D array or atlas)
        self._terrain_slot_map: Dict[tuple, int] = {}  # (z,x,y) -> slot index
        self._terrain_slot_lru: list = []       # LRU list of slot indices
        self._terrain_slots_free: deque = deque()  # free slot indices

        # Opt 4: VBO Pool — created lazily after GL init
        self._vbo_pool: Optional[VBOPool] = None

        # Opt 5: Physics Thread
        self._physics_thread = PhysicsThread()
        self._physics_thread.start()

        # Animation / idle timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

        # --- Preload base tiles (z0-z3) ---
        self._base_preload_max_z = 3
        self._preload_queue = []
        QTimer.singleShot(100, self._preload_base_tiles)

    # -----------------------------------------------------------------
    #  GL initialisation (lazy)
    # -----------------------------------------------------------------

    def _ensure_gl(self):
        if self._gl_ready: return
        self._gl_ready = True
        self._ctx = moderngl.create_context(standalone=True)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # v6: geometry shader for direct MVT rendering
        self._prog_geo = self._ctx.program(vertex_shader=GEO_VERT, fragment_shader=GEO_FRAG)

        # Fallback raster tile shader (kept for parent zoom fallback)
        self._prog_tile = self._ctx.program(vertex_shader=TILE_VERT, fragment_shader=TILE_FRAG)
        quad = np.array([
            0,0, 0,0,  1,0, 1,0,  1,1, 1,1,
            0,0, 0,0,  1,1, 1,1,  0,1, 0,1,
        ], dtype='f4').tobytes()
        self._quad_vbo = self._ctx.buffer(quad)
        self._vao_tile = self._ctx.vertex_array(
            self._prog_tile, [(self._quad_vbo, '2f 2f', 'in_position', 'in_uv')])

        self._prog_bldg = self._ctx.program(vertex_shader=BLDG_VERT, fragment_shader=BLDG_FRAG)
        self._prog_shadow = self._ctx.program(vertex_shader=SHADOW_VERT, fragment_shader=SHADOW_FRAG)
        self._prog_route = self._ctx.program(vertex_shader=ROUTE_VERT, fragment_shader=ROUTE_FRAG)
        self._prog_heat = self._ctx.program(vertex_shader=HEAT_VERT, fragment_shader=HEAT_FRAG)

        # Immersive mode shaders
        self._fp_prog_bldg = self._ctx.program(vertex_shader=FP_BLDG_VERT, fragment_shader=FP_BLDG_FRAG)
        self._fp_prog_ground = self._ctx.program(vertex_shader=FP_GROUND_VERT, fragment_shader=FP_GROUND_FRAG)
        self._fp_prog_water = self._ctx.program(vertex_shader=FP_WATER_VERT, fragment_shader=FP_WATER_FRAG)
        self._fp_prog_road = self._ctx.program(vertex_shader=FP_ROAD_VERT, fragment_shader=FP_ROAD_FRAG)
        self._fp_prog_road_nav = self._ctx.program(vertex_shader=FP_ROAD_NAV_VERT, fragment_shader=FP_ROAD_NAV_FRAG)
        self._fp_prog_sky = self._ctx.program(vertex_shader=FP_SKY_VERT, fragment_shader=FP_SKY_FRAG)
        self._fp_prog_car = self._ctx.program(vertex_shader=FP_CAR_VERT, fragment_shader=FP_CAR_FRAG)

        # POI marker shaders
        self._fp_prog_poi = self._ctx.program(vertex_shader=FP_POI_VERT, fragment_shader=FP_POI_FRAG)
        self._prog_poi_2d = self._ctx.program(vertex_shader=POI_2D_VERT, fragment_shader=POI_2D_FRAG)

        # Build sky sphere VBO
        sky_data = _make_sky_sphere()
        self._fp_sky_vbo = self._ctx.buffer(np.ascontiguousarray(sky_data).tobytes())
        self._fp_sky_vao = self._ctx.vertex_array(
            self._fp_prog_sky, [(self._fp_sky_vbo, '3f', 'in_pos')])
        self._fp_sky_count = len(sky_data)

        # Opt 4: VBO Pool REMOVED — moderngl VAOs derive vertex count from
        # buffer size, so oversized pooled buffers render garbage vertices.
        # Exact-sized ctx.buffer() is used instead. The time-sliced upload
        # queue (Opt 2) handles the "hitch" problem.
        self._vbo_pool = None

        # Opt 3: Initialise terrain texture atlas (16 slots of 256×256 R32F)
        n_slots = self._terrain_tex_slots
        slot_sz = self._terrain_tex_slot_size
        # Use a 4×4 grid atlas: 1024×1024 R32F
        atlas_w = 4 * slot_sz  # 1024
        atlas_h = 4 * slot_sz  # 1024
        self._terrain_atlas_cols = 4
        self._terrain_atlas_rows = 4
        self._terrain_tex_array = self._ctx.texture(
            (atlas_w, atlas_h), 1, dtype='f4',
            data=np.zeros(atlas_w * atlas_h, dtype='f4').tobytes())
        self._terrain_tex_array.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._terrain_tex_array.repeat_x = False
        self._terrain_tex_array.repeat_y = False
        self._terrain_slots_free = deque(range(n_slots))

        self._resize_fbo(max(self.width(), 320), max(self.height(), 200))

    def resizeGL(self, w, h):
        """QOpenGLWidget resize callback — just mark dirty."""
        self._gl_dirty = True

    def _resize_fbo(self, w, h):
        if w == self._fbo_w and h == self._fbo_h and self._fbo: return
        if self._fbo: self._fbo.release()
        self._fbo_w = w; self._fbo_h = h
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((w, h), 4)],
            depth_attachment=self._ctx.depth_renderbuffer((w, h)),
        )
        self._gl_dirty = True

    # -----------------------------------------------------------------
    #  Base tile preloading (z0–z3 at startup)
    # -----------------------------------------------------------------

    def _preload_base_tiles(self):
        self._preload_queue = []
        for z in range(0, self._base_preload_max_z + 1):
            n = 2 ** z
            for x in range(n):
                for y in range(n):
                    self._preload_queue.append((z, x, y))
        self._preload_batch_next()

    def _preload_batch_next(self):
        batch = 8  # v6: increased batch size
        fired = 0
        while self._preload_queue and fired < batch:
            z, x, y = self._preload_queue.pop(0)
            key = (z, x, y)
            if key in self._pending or key in self._rendering:
                continue
            with self._mvt_lock:
                if key in self._mvt_cache:
                    self._submit_render(key)
                    continue
            if not self.token:
                continue
            self._pending.add(key)
            url_str = _mvt_tile_url(self.token, self._current_tileset, z, x, y)
            if not url_str:
                self._pending.discard(key); continue
            req = QNetworkRequest(url_str)
            reply = self._nam.get(req)
            self._pending_replies[key] = reply
            reply.finished.connect(lambda r=reply, k=key: self._on_mvt_data(r, k))
            fired += 1
        if self._preload_queue:
            QTimer.singleShot(60, self._preload_batch_next)

    # -----------------------------------------------------------------
    #  Properties
    # -----------------------------------------------------------------

    @property
    def centre_lon(self): return _wx_to_lon(self._cx, self._zoom)
    @property
    def centre_lat(self): return _wy_to_lat(self._cy, self._zoom)
    @property
    def _current_style(self):
        key = self._style.split("/")[-1]
        return ALL_STYLES.get(key, list(ALL_STYLES.values())[0])
    @property
    def _current_tileset(self): return self.TILESETS[self._style_idx]

    def _geo_cache_key(self, z, x, y): return (z, x, y, self._style)

    # -----------------------------------------------------------------
    #  Smooth Camera Animation API
    # -----------------------------------------------------------------

    def flyTo(self, lat, lon, zoom=None, bearing=None, pitch=None,
              duration=None, speed=1.2):
        """Animate to a new position with Mapbox-style zoom-out arc."""
        if self._immersive:
            return
        cur_lat = _wy_to_lat(self._cy, self._zoom)
        cur_lon = _wx_to_lon(self._cx, self._zoom)
        self._fly.begin(
            cur_lat, cur_lon, self._zoom, self._bearing, self._pitch,
            lat, lon,
            _clamp_zoom(zoom if zoom is not None else self._zoom),
            bearing, pitch,
            duration=duration, speed=speed)
        self._scroll_zoom_active = False
        self._fly_last_tick = time.perf_counter()
        self._timer.setInterval(16)
        self._gl_dirty = True; self.update()

    def easeTo(self, lat, lon, zoom=None, bearing=None, pitch=None,
               duration=0.6):
        """Smooth ease to new position without zoom-out arc."""
        if self._immersive:
            return
        cur_lat = _wy_to_lat(self._cy, self._zoom)
        cur_lon = _wx_to_lon(self._cx, self._zoom)
        self._fly.begin(
            cur_lat, cur_lon, self._zoom, self._bearing, self._pitch,
            lat, lon,
            _clamp_zoom(zoom if zoom is not None else self._zoom),
            bearing, pitch,
            duration=duration)
        self._fly.min_zoom = min(self._zoom,
                                 _clamp_zoom(zoom if zoom is not None else self._zoom))
        self._fly.suppress_tiles = False
        self._fly.ease_fn = _ease_out_cubic
        self._scroll_zoom_active = False
        self._fly_last_tick = time.perf_counter()
        self._timer.setInterval(16)
        self._gl_dirty = True; self.update()

    def cancelFlight(self):
        self._fly.cancel()
        self._target_zoom = self._zoom

    @property
    def isFlying(self):
        return self._fly.active

    # -----------------------------------------------------------------
    #  External panel binding
    # -----------------------------------------------------------------

    def set_panel(self, panel):
        """Connect an external SettingsPanel to this map widget."""
        self._panel = panel
        self._panel_visible = True
        panel.value_changed.connect(self._on_settings_changed)
        panel.toggle_changed.connect(self._on_panel_toggle)
        # Sync initial state
        panel.set_check("buildings_3d", self._buildings_3d)
        panel.set_check("labels_visible", self._labels_visible)
        panel.set_check("shadows", self._shadows_enabled)
        panel.set_check("tiles_visible", self._tiles_visible)
        panel.set_check("poi_markers", self._poi_enabled)

    def _on_panel_toggle(self, key, val):
        if key == "buildings_3d":
            self._buildings_3d = val
        elif key == "labels_visible":
            self._labels_visible = val
        elif key == "shadows":
            self._shadows_enabled = val
            self._shadow_gpu_key = ()
        elif key == "tiles_visible":
            self._tiles_visible = val
        elif key == "poi_markers":
            self._poi_enabled = val
            self._poi_2d_dirty = True
            self._fp_poi_rebuild = True
        self._gl_dirty = True; self.update()

    # -----------------------------------------------------------------
    #  Settings changed
    # -----------------------------------------------------------------

    def _on_settings_changed(self):
        self._bld_gpu_key = ()
        self._heatmap_dirty = True  # radius changes need VBO rebuild
        self._gl_dirty = True
        self.update()

    def _get_panel_val(self, key, default=0.0):
        """Safe panel value getter — returns default when panel is absent."""
        if self._panel is None:
            return default
        return self._panel.get(key)

    # -----------------------------------------------------------------
    #  Opt 1: Predictive "Loading Horizon" Pre-Tessellation
    # -----------------------------------------------------------------

    def _prefetch_tiles_ahead(self, wx, wy, yaw_deg, speed_wpf, lookahead_sec=3.0):
        """Pre-request tiles along the predicted flight path.
        *speed_wpf* is speed in world-pixels per frame (~16ms).
        Looks 2-3 seconds ahead and triggers _request_tile for tiles
        the camera hasn't reached yet."""
        if speed_wpf < 0.5:
            return  # not moving fast enough to bother
        z = self._zoom
        z_int = max(0, min(20, int(math.floor(z + 0.5))))
        tile_px = TILE_SIZE * (2.0 ** (z - z_int))
        n = 2 ** z_int

        yaw_rad = math.radians(yaw_deg)
        sin_y = math.sin(yaw_rad)
        cos_y = -math.cos(yaw_rad)

        # Frames per second ≈ 60; speed_wpf = wp/frame
        speed_wps = speed_wpf * 60.0  # world-pixels per second

        # Debounce: bucket yaw to 15° and tile position
        cam_tx = int(math.floor(wx / tile_px))
        cam_ty = int(math.floor(wy / tile_px))
        yaw_bucket = int(yaw_deg / 15.0)
        pfkey = (z_int, cam_tx, cam_ty, yaw_bucket)
        if pfkey == self._last_prefetch_key:
            return
        self._last_prefetch_key = pfkey

        # Sample 3 points along the predicted path: 1s, 2s, 3s ahead
        for t in (1.0, 2.0, 3.0):
            fwd_wx = wx + sin_y * speed_wps * t
            fwd_wy = wy + cos_y * speed_wps * t
            ftx = int(math.floor(fwd_wx / tile_px))
            fty = int(math.floor(fwd_wy / tile_px))
            # Request a 3×3 around each predicted tile
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    self._request_tile(z_int, (ftx + dx) % n, fty + dy)
            # Also pre-fetch terrain
            if self._terrain_enabled:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        self._request_terrain_tile(z_int, (ftx + dx) % n, fty + dy)

    # -----------------------------------------------------------------
    #  Opt 2: Time-Sliced GPU Upload — drain 1-2 items per frame
    # -----------------------------------------------------------------

    def _drain_gpu_upload(self):
        """Called by upload timer (~120 Hz).  Pops pending tile
        geometry uploads and executes them on the main thread.
        During initial load (many queued), drains up to 4 per tick.
        During steady-state streaming, drains 1-2 per tick to keep
        per-frame GPU work under ~4 ms."""
        if not self._gl_ready:
            return
        # Drain more aggressively when backlog is large (initial load)
        max_per_tick = 4 if len(self._gpu_upload_queue) > 8 else 2
        uploaded = 0
        for _ in range(max_per_tick):
            entry = self._gpu_upload_queue.pop()
            if entry is None:
                break
            self._upload_tile_to_gpu(entry)
            uploaded += 1
        # Trigger repaint if we uploaded anything
        if uploaded > 0:
            self._gl_dirty = True
            self.update()
        # If there are still pending uploads, keep the render loop hot
        elif self._gpu_upload_queue.pending:
            self._gl_dirty = True
            self.update()

    def _upload_tile_to_gpu(self, entry):
        """Upload a single tile's geometry to the GPU.
        Uses exact-sized ctx.buffer() — VBO pool removed because moderngl
        VAOs derive vertex count from buffer size, so oversized pooled
        buffers cause rendering of stale/garbage vertices."""
        gkey = entry['gkey']
        fill_data = entry['fill_data']
        line_data = entry['line_data']
        labels = entry['labels']
        bldgs = entry['bldgs']
        key = entry['key']

        # Release old GPU resources for this key
        old = self._geo_cache.pop(gkey, None)
        if old:
            for k in ('fill_vbo', 'fill_vao', 'line_vbo', 'line_vao'):
                obj = old.get(k)
                if obj:
                    try: obj.release()
                    except: pass

        geo_entry = {'fill_count': 0, 'line_count': 0,
                     'fill_vbo': None, 'fill_vao': None,
                     'line_vbo': None, 'line_vao': None}

        if self._gl_ready and len(fill_data) > 0:
            buf = self._ctx.buffer(np.ascontiguousarray(fill_data).tobytes())
            vao = self._ctx.vertex_array(
                self._prog_geo, [(buf, '2f 4f', 'in_position', 'in_color')])
            geo_entry['fill_vbo'] = buf
            geo_entry['fill_vao'] = vao
            geo_entry['fill_count'] = len(fill_data)

        if self._gl_ready and len(line_data) > 0:
            buf = self._ctx.buffer(np.ascontiguousarray(line_data).tobytes())
            vao = self._ctx.vertex_array(
                self._prog_geo, [(buf, '2f 4f', 'in_position', 'in_color')])
            geo_entry['line_vbo'] = buf
            geo_entry['line_vao'] = vao
            geo_entry['line_count'] = len(line_data)

        self._geo_cache[gkey] = geo_entry

        # Evict old entries
        while len(self._geo_cache) > self._geo_cache_max:
            old_k = next(iter(self._geo_cache))
            old_e = self._geo_cache.pop(old_k)
            for k in ('fill_vbo', 'fill_vao', 'line_vbo', 'line_vao'):
                obj = old_e.get(k)
                if obj:
                    try: obj.release()
                    except: pass

        # Cache labels and buildings
        self._label_cache[gkey] = labels
        self._building_cache[key] = bldgs
        while len(self._building_cache) > self._building_cache_max:
            del self._building_cache[next(iter(self._building_cache))]
        self._bld_gpu_key = ()

        # Force immersive geometry rebuild when new building data arrives
        if self._immersive and bldgs:
            self._fp_geo_key = None

        self._gl_dirty = True
        self._tiles_arrived_since_paint += 1

    # -----------------------------------------------------------------
    #  Tick — animation interpolation
    # -----------------------------------------------------------------

    def _tick(self):
        self._frame += 1
        changed = False

        # POI popup timer and proximity check
        dt_tick = 1.0 / 60.0
        self._poi_tick_popup(dt_tick)

        # Search bar cursor blink
        if self._search_active:
            self._search_cursor_blink += dt_tick
            if self._frame % 8 == 0:  # redraw periodically for cursor blink
                self._gl_dirty = True; self.update()

        if self._immersive and self._poi_enabled:
            self._poi_check_proximity_immersive()
        elif not self._immersive and self._poi_enabled:
            # 2D hover detection — check cursor position each tick
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(cursor_pos):
                self._poi_hover_check(cursor_pos.x(), cursor_pos.y())

        # --- Immersive mode camera movement (Opt 5: read from physics thread) ---
        if self._immersive:
            pt = self._physics_thread
            if self._car_mode:
                # Sync physics thread inputs
                pt.car_mode = True
                pt.car_throttle = self._car_throttle
                pt.car_brake = self._car_brake
                pt.car_steer = self._car_steer
                pt.car_max_speed = self._car_max_speed
                pt.car_accel_rate = self._car_accel_rate
                pt.car_mpp = self._car_mpp
                # Read back latest state
                snap = pt.snapshot()
                self._car_wx = snap['car_wx']
                self._car_wy = snap['car_wy']
                self._car_yaw = snap['car_yaw']
                self._car_speed = snap['car_speed']
                self._car_steer_angle = snap['car_steer_angle']
                # Still run main-thread car logic for collisions, traffic, camera
                self._tick_car()
                changed = True
                # Opt 1: Predictive prefetch along car heading
                if self._car_speed > 0.5:
                    self._prefetch_tiles_ahead(
                        self._car_wx, self._car_wy,
                        self._car_yaw, self._car_speed / 60.0)
            else:
                # Walk mode — sync inputs to physics thread
                pt.car_mode = False
                pt.keys_held = set(self._fp_keys_held)
                pt.move_speed = self._fp_move_speed
                # Read back state
                snap = pt.snapshot()
                if self._fp_keys_held:
                    self._fp_cx = snap['walk_wx']
                    self._fp_cy = snap['walk_wy']
                    self._fp_eye_height = snap['walk_eye_h']
                    self._cx = self._fp_cx; self._cy = self._fp_cy
                    self._gl_dirty = True; self.update()
                    changed = True
                    # Opt 1: Predictive prefetch in walk direction
                    self._prefetch_tiles_ahead(
                        self._fp_cx, self._fp_cy,
                        self._fp_yaw, self._fp_move_speed / 60.0)

        self._zoom_velocity = abs(self._zoom - self._prev_zoom)
        self._prev_zoom = self._zoom

        # --- v9: Compute real dt for animations ---
        now = time.perf_counter()
        fly_dt = min(now - self._fly_last_tick, 1.0 / 15.0)
        self._fly_last_tick = now

        # --- v9: FlyTo animation (highest priority — drives all camera state) ---
        if self._fly.active:
            lat, lon, zoom, bearing, pitch, finished = self._fly.step(fly_dt)
            # Convert lat/lon at the interpolated zoom to world-pixels
            self._zoom = zoom
            self._cx = _lon_to_wx(lon, zoom)
            self._cy = _lat_to_wy(lat, zoom)
            self._bearing = _normalize_bearing(bearing)
            self._pitch = _clamp_pitch(pitch)
            if finished:
                self._target_zoom = zoom
                self._target_bearing = bearing
                self._target_pitch = pitch
            changed = True

        # --- v9: Smooth scroll-zoom toward cursor ---
        elif self._scroll_zoom_active:
            diff = self._scroll_zoom_target - self._zoom
            if abs(diff) > 0.003:
                step = diff * min(1.0, 8.0 * fly_dt)
                new_zoom = _clamp_zoom(self._zoom + step)
                # Recompute cx/cy so the anchor lat/lon stays under the cursor.
                # anchor_wx(z) = _lon_to_wx(anchor_lon, z)
                # cx = anchor_wx - screen_offset  (screen_offset is zoom-invariant)
                anchor_wx = _lon_to_wx(self._scroll_anchor_lon, new_zoom)
                anchor_wy = _lat_to_wy(self._scroll_anchor_lat, new_zoom)
                self._cx = anchor_wx - self._scroll_anchor_sx
                self._cy = anchor_wy - self._scroll_anchor_sy
                self._zoom = new_zoom
                self._target_zoom = self._scroll_zoom_target
                changed = True
            else:
                # Snap to final value
                anchor_wx = _lon_to_wx(self._scroll_anchor_lon, self._scroll_zoom_target)
                anchor_wy = _lat_to_wy(self._scroll_anchor_lat, self._scroll_zoom_target)
                self._cx = anchor_wx - self._scroll_anchor_sx
                self._cy = anchor_wy - self._scroll_anchor_sy
                self._zoom = self._scroll_zoom_target
                self._target_zoom = self._scroll_zoom_target
                self._scroll_zoom_active = False
                changed = True

        # --- Fallback: simple target-zoom lerp (keyboard +/-) ---
        elif abs(self._zoom - self._target_zoom) > 0.001:
            old_zoom = self._zoom
            self._zoom += (self._target_zoom - self._zoom) * 0.25
            self._zoom = _clamp_zoom(self._zoom)
            # Only scale cx/cy if no drag is active; drags recompute from lat/lon
            if self._drag_start is None and self._rdrag_start is None and self._mdrag_start is None:
                scale = 2.0 ** (self._zoom - old_zoom)
                self._cx *= scale
                self._cy *= scale
            changed = True

        # Pitch + bearing lerp (only when NOT in flyTo)
        if not self._fly.active:
            max_p = self._get_panel_val("max_pitch", 60.0)
            if abs(self._pitch - self._target_pitch) > 0.05:
                self._pitch += (self._target_pitch - self._pitch) * 0.22
                self._pitch = max(0.0, min(max_p, self._pitch)); changed = True

            bd = self._target_bearing - self._bearing
            if bd > 180: bd -= 360
            if bd < -180: bd += 360
            if abs(bd) > 0.05:
                self._bearing += bd * 0.22
                self._bearing = _normalize_bearing(self._bearing); changed = True

        if changed:
            self._gl_dirty = True
            if self._route_geometry:
                self._route_world = [
                    (_lon_to_wx(lon, self._zoom), _lat_to_wy(lat, self._zoom))
                    for lat, lon in self._route_geometry
                ]
                self._route_dirty = True
            elif self._route_points:
                self._route_world = [
                    (_lon_to_wx(lon, self._zoom), _lat_to_wy(lat, self._zoom))
                    for lat, lon in self._route_points
                ]
                self._route_dirty = True
            if self._heatmap_enabled and self._heatmap_points_latlon:
                self._heatmap_dirty = True
            self.update()

        has_route = len(self._route_world) >= 2
        is_busy = bool(self._pending or self._rendering)
        is_animating = self._fly.active or self._scroll_zoom_active
        need_anim = changed or is_busy or has_route or is_animating
        new_iv = 16 if need_anim else 50
        if self._timer.interval() != new_iv:
            self._timer.setInterval(new_iv)
        if has_route:
            self._gl_dirty = True
            self.update()

    # -----------------------------------------------------------------
    #  Tile network — prioritised & throttled
    # -----------------------------------------------------------------

    def _visible_tile_range(self):
        """v6+hotfix: Use floor(zoom) for stability during interactive zoom,
        but floor(zoom+0.5) in immersive mode for higher-res road tiles.
        Adds padding for smoother tile pop-in, tighter pitch extra."""
        # Immersive/car mode: round up for sharper roads; normal: floor for less churn
        if self._immersive:
            z_int = max(0, min(20, int(math.floor(self._zoom + 0.5))))
        else:
            z_int = max(0, min(20, int(math.floor(self._zoom))))
        tile_px = TILE_SIZE * (2.0 ** (self._zoom - z_int))
        w, h = self.width(), self.height()
        half_w, half_h = w / 2.0, h / 2.0

        b_rad = math.radians(self._bearing)
        sin_b, cos_b = abs(math.sin(b_rad)), abs(math.cos(b_rad))
        eff_hw = half_w * cos_b + half_h * sin_b
        eff_hh = half_w * sin_b + half_h * cos_b

        pitch_extra = 1 if self._pitch > 12.0 else (2 if self._pitch > 35.0 else 0)

        pad = 96.0
        col_min = int(math.floor((self._cx - eff_hw - pad) / tile_px))
        col_max = int(math.ceil((self._cx + eff_hw + pad) / tile_px))
        row_min = int(math.floor((self._cy - eff_hh - pad) / tile_px)) - pitch_extra
        row_max = int(math.ceil((self._cy + eff_hh + pad) / tile_px))
        n = 2 ** z_int

        max_ax = int(self._get_panel_val("max_tiles", 8))
        max_ax = max(4, min(max_ax, 10))
        if (col_max - col_min) > max_ax:
            mid = (col_min + col_max) // 2
            col_min = mid - max_ax // 2
            col_max = col_min + max_ax
        if (row_max - row_min) > max_ax:
            mid = (row_min + row_max) // 2
            row_min = mid - max_ax // 2
            row_max = row_min + max_ax

        return z_int, tile_px, n, col_min, col_max, row_min, row_max

    def _request_tile(self, z, x, y):
        n = 2 ** z; x = x % n
        if y < 0 or y >= n: return
        key = (z, x, y)
        gkey = self._geo_cache_key(z, x, y)
        # Opt 2: Also check the GPU upload queue — tiles there are "in flight"
        if (gkey in self._geo_cache or key in self._pending or key in self._rendering
                or gkey in self._gpu_upload_queue._pending_keys):
            return
        with self._mvt_lock:
            if key in self._mvt_cache:
                self._submit_render(key)
                return
        if not self.token: return
        if len(self._pending) >= self._max_concurrent_net:
            # v7: queue for later instead of dropping
            if key not in self._render_queue_set:
                self._render_queue.append(key)
                self._render_queue_set.add(key)
            return
        self._pending.add(key)
        url_str = _mvt_tile_url(self.token, self._current_tileset, z, x, y)
        if not url_str:
            self._pending.discard(key); return
        req = QNetworkRequest(url_str)
        reply = self._nam.get(req)
        self._pending_replies[key] = reply
        reply.finished.connect(lambda r=reply, k=key: self._on_mvt_data(r, k))

    def _cancel_stale_requests(self, current_z):
        base_z = self._base_preload_max_z
        stale = [k for k in self._pending if k[0] != current_z and k[0] > base_z]
        for k in stale:
            reply = self._pending_replies.pop(k, None)
            if reply is not None:
                try: reply.abort(); reply.deleteLater()
                except: pass
            self._pending.discard(k)

    def _on_mvt_data(self, reply, key):
        self._pending.discard(key)
        self._pending_replies.pop(key, None)
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = bytes(reply.readAll())
            if data:
                style = self._current_style
                z = key[0]
                signals = self._signals

                def _decode_and_tessellate():
                    try:
                        mvt = decode_mvt(data)
                        fill_data, line_data, labels, bldgs = _tessellate_tile(mvt, style, z)
                        with self._mvt_lock:
                            self._mvt_cache[key] = mvt
                            while len(self._mvt_cache) > self._mvt_cache_max:
                                self._mvt_cache.popitem(last=False)
                        signals.tile_ready.emit(key, fill_data, line_data, labels, bldgs)
                    except Exception:
                        signals.tile_ready.emit(key, None, None, [], [])

                self._rendering.add(key)
                self._executor.submit(_decode_and_tessellate)
        reply.deleteLater()

        # Hotfix: drain queued requests now that a slot freed up
        while self._render_queue and len(self._pending) < self._max_concurrent_net:
            qkey = self._render_queue.pop(0)
            self._render_queue_set.discard(qkey)
            if qkey in self._pending or qkey in self._rendering:
                continue
            z2, x2, y2 = qkey
            self._request_tile(z2, x2, y2)

    def _submit_render(self, key):
        """Re-tessellate an already-cached MVT tile for current style."""
        if key in self._rendering: return
        # Opt 2: Also skip if already in the upload queue
        gkey = self._geo_cache_key(key[0], key[1], key[2])
        if gkey in self._gpu_upload_queue._pending_keys:
            return
        max_conc = int(self._get_panel_val("render_conc", self._max_concurrent_render))
        if len(self._rendering) >= max_conc:
            if key not in self._render_queue_set:
                self._render_queue.append(key)
                self._render_queue_set.add(key)
            return
        with self._mvt_lock:
            mvt = self._mvt_cache.get(key)
        if mvt is None: return
        self._rendering.add(key)
        self._render_queue_set.discard(key)
        style = self._current_style; z = key[0]
        signals = self._signals

        def _do():
            try:
                fill_data, line_data, labels, bldgs = _tessellate_tile(mvt, style, z)
                signals.tile_ready.emit(key, fill_data, line_data, labels, bldgs)
            except:
                signals.tile_ready.emit(key, None, None, [], [])
        self._executor.submit(_do)

    def _on_tile_rendered(self, key, fill_data, line_data, labels, bldgs):
        """Main-thread callback: queue tessellated geometry for time-sliced
        GPU upload instead of uploading everything in one frame (Opt 2)."""
        self._rendering.discard(key)

        if fill_data is not None:
            gkey = self._geo_cache_key(key[0], key[1], key[2])
            # Enqueue for time-sliced upload
            self._gpu_upload_queue.enqueue(gkey, key, fill_data, line_data, labels, bldgs)

        if not self._coalesce_timer.isActive():
            self._coalesce_timer.start()

        # Drain deferred render queue
        while self._render_queue:
            qkey = self._render_queue[0]
            if qkey in self._rendering:
                self._render_queue.pop(0)
                self._render_queue_set.discard(qkey)
                continue
            max_conc = int(self._get_panel_val("render_conc", self._max_concurrent_render))
            if len(self._rendering) >= max_conc:
                break
            self._render_queue.pop(0)
            self._render_queue_set.discard(qkey)
            self._submit_render(qkey)

    def _coalesced_update(self):
        self._tiles_arrived_since_paint = 0
        self.update()

    # -----------------------------------------------------------------
    #  Terrain DEM tile fetching
    # -----------------------------------------------------------------

    def _request_terrain_tile(self, z, x, y):
        """Request a terrain-RGB DEM tile (raster PNG)."""
        n = 2 ** z; x = x % n
        if y < 0 or y >= n: return
        tkey = (z, x, y)
        if tkey in self._terrain_cache or tkey in self._terrain_pending:
            return
        if not self.token: return
        self._terrain_pending.add(tkey)
        url_str = _terrain_rgb_url(self.token, z, x, y)
        if not url_str:
            self._terrain_pending.discard(tkey); return
        req = QNetworkRequest(QUrl(url_str))
        reply = self._nam.get(req)
        reply.finished.connect(lambda r=reply, k=tkey: self._on_terrain_png(r, k))

    def _on_terrain_png(self, reply, key):
        """Handle terrain-RGB PNG response — decode on worker thread."""
        self._terrain_pending.discard(key)
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = bytes(reply.readAll())
            if data:
                signals = self._signals
                def _decode():
                    try:
                        elev = _decode_terrain_rgb(data)
                        signals.terrain_ready.emit(key, elev)
                    except Exception:
                        signals.terrain_ready.emit(key, None)
                self._executor.submit(_decode)
        reply.deleteLater()

    def _on_terrain_data_ready(self, key, elev_array):
        """Main-thread: store decoded terrain DEM and update texture slot (Opt 3).
        Instead of rebuilding the entire stitched texture, write only one
        slot in the pre-allocated terrain atlas."""
        if elev_array is not None:
            self._terrain_cache[key] = elev_array
            while len(self._terrain_cache) > 200:
                del self._terrain_cache[next(iter(self._terrain_cache))]

            # Opt 3: Write to a single slot in the terrain atlas
            self._terrain_write_slot(key, elev_array)

            # Still mark dirty for the stitched fallback path
            self._terrain_tex_dirty = True
            if self._immersive:
                self._gl_dirty = True
                self.update()

    def _terrain_write_slot(self, key, elev_array):
        """Write a single DEM tile into one slot of the terrain atlas texture.
        This replaces the old full-stitch approach for incremental updates."""
        if not self._gl_ready or self._terrain_tex_array is None:
            return
        slot_sz = self._terrain_tex_slot_size

        # Already have a slot for this key?
        if key in self._terrain_slot_map:
            slot_idx = self._terrain_slot_map[key]
        elif self._terrain_slots_free:
            slot_idx = self._terrain_slots_free.popleft()
            self._terrain_slot_map[key] = slot_idx
        else:
            # Evict LRU slot
            if self._terrain_slot_lru:
                evict_key = self._terrain_slot_lru.pop(0)
                slot_idx = self._terrain_slot_map.pop(evict_key, None)
                if slot_idx is None:
                    return
                self._terrain_slot_map[key] = slot_idx
            else:
                return

        # Update LRU
        if key in self._terrain_slot_lru:
            self._terrain_slot_lru.remove(key)
        self._terrain_slot_lru.append(key)

        # Compute atlas position from slot index
        col = slot_idx % self._terrain_atlas_cols
        row = slot_idx // self._terrain_atlas_cols
        px = col * slot_sz
        py = row * slot_sz

        # Resample to slot size if needed
        eh, ew = elev_array.shape
        if eh != slot_sz or ew != slot_sz:
            # Simple nearest-neighbor resample
            ys = np.linspace(0, eh - 1, slot_sz).astype(int)
            xs = np.linspace(0, ew - 1, slot_sz).astype(int)
            tile_data = elev_array[np.ix_(ys, xs)].astype('f4')
        else:
            tile_data = elev_array.astype('f4')

        # Write to the atlas texture using viewport
        try:
            self._terrain_tex_array.write(
                tile_data.tobytes(),
                viewport=(px, py, slot_sz, slot_sz))
        except Exception:
            pass  # texture not ready

    def _request_terrain_around(self, wx, wy, radius_tiles=2):
        """Request terrain tiles around a world-pixel position."""
        if not self._terrain_enabled or not self.token:
            return
        z = self._zoom
        z_int = max(0, min(15, int(math.floor(z + 0.5))))
        tile_px = TILE_SIZE * (2.0 ** (z - z_int))
        tx = int(math.floor(wx / tile_px))
        ty = int(math.floor(wy / tile_px))
        n = 2 ** z_int
        for dy in range(-radius_tiles, radius_tiles + 1):
            for dx in range(-radius_tiles, radius_tiles + 1):
                self._request_terrain_tile(z_int, (tx + dx) % n, ty + dy)

    def _prefetch_terrain_ahead(self, wx, wy, yaw_deg, ahead_tiles=4):
        """Prefetch terrain tiles ahead of car movement direction."""
        if not self._terrain_enabled or not self.token:
            return
        z = self._zoom
        z_int = max(0, min(15, int(math.floor(z + 0.5))))
        tile_px = TILE_SIZE * (2.0 ** (z - z_int))
        n = 2 ** z_int
        yaw_rad = math.radians(yaw_deg)
        sin_y = math.sin(yaw_rad)
        cos_y = -math.cos(yaw_rad)
        for dist in range(1, ahead_tiles + 1):
            fwd_wx = wx + sin_y * tile_px * dist
            fwd_wy = wy + cos_y * tile_px * dist
            tx = int(math.floor(fwd_wx / tile_px)) % n
            ty = int(math.floor(fwd_wy / tile_px))
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    self._request_terrain_tile(z_int, (tx + dx) % n, ty + dy)

    def _rebuild_terrain_texture(self, cam_wx, cam_wy, radius):
        """Stitch nearby terrain DEM tiles into a single R32F GL texture.
        The texture covers a world-pixel bounding box around the camera."""
        if not self._gl_ready or not self._terrain_cache:
            return
        z = self._zoom
        z_int = max(0, min(15, int(math.floor(z + 0.5))))
        tile_px = TILE_SIZE * (2.0 ** (z - z_int))
        n = 2 ** z_int

        # Determine which tiles cover the visible radius
        tx_min = int(math.floor((cam_wx - radius) / tile_px))
        tx_max = int(math.ceil((cam_wx + radius) / tile_px))
        ty_min = int(math.floor((cam_wy - radius) / tile_px))
        ty_max = int(math.ceil((cam_wy + radius) / tile_px))

        # Clamp tile range
        ty_min = max(0, ty_min); ty_max = min(n, ty_max)
        ntx = tx_max - tx_min
        nty = ty_max - ty_min
        if ntx <= 0 or nty <= 0:
            return

        # Get the DEM tile size (all tiles should be same size)
        sample_tile = None
        for key, elev in self._terrain_cache.items():
            sample_tile = elev; break
        if sample_tile is None:
            return
        dem_h, dem_w = sample_tile.shape

        # Stitch into one array
        tex_w = ntx * dem_w
        tex_h = nty * dem_h
        # Cap texture size for performance
        if tex_w > 2048 or tex_h > 2048:
            return
        stitched = np.zeros((tex_h, tex_w), dtype='f4')

        for tyi, ty in enumerate(range(ty_min, ty_max)):
            for txi, tx_raw in enumerate(range(tx_min, tx_max)):
                tx = tx_raw % n
                tkey = (z_int, tx, ty)
                elev = self._terrain_cache.get(tkey)
                if elev is not None:
                    py = tyi * dem_h
                    px = txi * dem_w
                    eh, ew = elev.shape
                    # Paste (handle size mismatch gracefully)
                    ch = min(eh, tex_h - py)
                    cw = min(ew, tex_w - px)
                    stitched[py:py+ch, px:px+cw] = elev[:ch, :cw]

        # Bounds in world-pixel coords
        bounds_min_wx = float(tx_min * tile_px)
        bounds_min_wy = float(ty_min * tile_px)
        bounds_max_wx = float(tx_max * tile_px)
        bounds_max_wy = float(ty_max * tile_px)
        self._terrain_tex_bounds = (bounds_min_wx, bounds_min_wy, bounds_max_wx, bounds_max_wy)

        # Upload to GL texture
        if self._terrain_tex:
            try: self._terrain_tex.release()
            except: pass
        self._terrain_tex = self._ctx.texture((tex_w, tex_h), 1, dtype='f4',
                                               data=stitched.tobytes())
        self._terrain_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._terrain_tex.repeat_x = False
        self._terrain_tex.repeat_y = False
        self._terrain_tex_dirty = False

    # -----------------------------------------------------------------
    #  Building VBO — WORLD-SPACE, numpy-vectorized
    # -----------------------------------------------------------------

    def _rebuild_building_vbo(self, z_int, tile_px, visible_bld_keys,
                               default_h, h_exag, z_scale, pitch_factor, max_per_tile):
        # Hotfix: cap to 250 buildings per tile to reduce rebuild cost
        max_per_tile = min(int(max_per_tile), 250)
        style = self._current_style
        bld_style = None
        for ls in style.layers:
            if ls.get("building"):
                bld_style = ls; break
        if not bld_style: return

        fc = bld_style["fill"]
        base_r, base_g, base_b = fc.red()/255.0, fc.green()/255.0, fc.blue()/255.0
        base_a = min(0.92, fc.alpha()/255.0)
        wall_color = (base_r * 0.85, base_g * 0.85, base_b * 0.88, base_a)
        roof_color = (min(1.0, base_r * 1.25 + 0.08), min(1.0, base_g * 1.25 + 0.08),
                      min(1.0, base_b * 1.20 + 0.06), base_a)

        verts = _build_buildings_numpy(
            self._building_cache, visible_bld_keys, tile_px,
            default_h, h_exag, z_scale, pitch_factor, max_per_tile,
            wall_color, roof_color)

        if len(verts) == 0:
            self._bld_vert_count = 0; return
        data = np.ascontiguousarray(verts).tobytes()
        if self._bld_vbo:
            try: self._bld_vbo.release()
            except: pass
        if self._bld_vao:
            try: self._bld_vao.release()
            except: pass
        self._bld_vbo = self._ctx.buffer(data)
        self._bld_vao = self._ctx.vertex_array(
            self._prog_bldg,
            [(self._bld_vbo, '2f 1f 1f 3f 4f',
              'in_position', 'in_height', 'in_norm_height', 'in_normal', 'in_color')])
        self._bld_vert_count = len(verts)

    def _rebuild_shadow_vbo(self, visible_bld_keys, tile_px,
                             default_h, h_exag, z_scale, pitch_factor, max_per_tile):
        az, el = self._sun_azimuth, self._sun_elevation
        verts = _build_shadow_numpy(
            self._building_cache, visible_bld_keys, tile_px,
            default_h, h_exag, z_scale, pitch_factor, max_per_tile,
            az, el)
        if len(verts) == 0:
            self._shadow_vert_count = 0; return
        data = np.ascontiguousarray(verts).tobytes()
        if self._shadow_vbo:
            try: self._shadow_vbo.release()
            except: pass
        if self._shadow_vao:
            try: self._shadow_vao.release()
            except: pass
        self._shadow_vbo = self._ctx.buffer(data)
        self._shadow_vao = self._ctx.vertex_array(
            self._prog_shadow,
            [(self._shadow_vbo, '2f 1f', 'in_position', 'in_height')])
        self._shadow_vert_count = len(verts)

    def _rebuild_route_vbo(self):
        if len(self._route_world) < 2:
            self._route_vert_count = 0; return
        line_w = max(3.0, 8.0 * (2.0 ** (self._zoom - 15)))
        verts = _build_route_vbo(self._route_world, line_w)
        if len(verts) == 0:
            self._route_vert_count = 0; return
        data = np.ascontiguousarray(verts).tobytes()
        if self._route_vbo:
            try: self._route_vbo.release()
            except: pass
        if self._route_vao:
            try: self._route_vao.release()
            except: pass
        self._route_vbo = self._ctx.buffer(data)
        self._route_vao = self._ctx.vertex_array(
            self._prog_route,
            [(self._route_vbo, '2f 1f 1f', 'in_position', 'in_side', 'in_progress')])
        self._route_vert_count = len(verts)
        self._route_dirty = False

    def _rebuild_heatmap_vbo(self):
        if not self._heatmap_points_latlon:
            self._heatmap_vert_count = 0; return
        # Project lat/lon points to current zoom's world-pixel coordinates
        z = self._zoom
        points_world = []
        for lat, lon, w in self._heatmap_points_latlon:
            wx = _lon_to_wx(lon, z)
            wy = _lat_to_wy(lat, z)
            points_world.append((wx, wy, w))
        heat_radius_base = self._get_panel_val("heat_radius", 50.0)
        radius = max(15.0, heat_radius_base * (2.0 ** (self._zoom - 14)))
        verts = _build_heatmap_vbo(points_world, radius)
        if len(verts) == 0:
            self._heatmap_vert_count = 0; return
        data = np.ascontiguousarray(verts).tobytes()
        if self._heatmap_vbo:
            try: self._heatmap_vbo.release()
            except: pass
        if self._heatmap_vao:
            try: self._heatmap_vao.release()
            except: pass
        self._heatmap_vbo = self._ctx.buffer(data)
        self._heatmap_vao = self._ctx.vertex_array(
            self._prog_heat,
            [(self._heatmap_vbo, '2f 2f 1f', 'in_position', 'in_quad', 'in_weight')])
        self._heatmap_vert_count = len(verts)
        self._heatmap_dirty = False

    def _generate_heatmap_data(self):
        """Extract real heatmap data from cached MVT tiles for the visible area."""
        self._heatmap_points_latlon.clear()
        center_lat = _wy_to_lat(self._cy, self._zoom)
        center_lon = _wx_to_lon(self._cx, self._zoom)
        self._heatmap_gen_lat = center_lat
        self._heatmap_gen_lon = center_lon
        self._heatmap_gen_zoom = self._zoom

        z_int = max(0, min(20, int(math.floor(self._zoom + 0.5))))
        tile_px = TILE_SIZE * (2.0 ** (self._zoom - z_int))
        w, h = self.width(), self.height()
        half_w, half_h = w / 2.0, h / 2.0
        col_min = int(math.floor((self._cx - half_w * 1.5) / tile_px))
        col_max = int(math.ceil((self._cx + half_w * 1.5) / tile_px))
        row_min = int(math.floor((self._cy - half_h * 1.5) / tile_px))
        row_max = int(math.ceil((self._cy + half_h * 1.5) / tile_px))
        n = 2 ** z_int

        mode = self._heatmap_mode

        if mode == "traffic":
            self._extract_traffic_heatmap(z_int, n, col_min, col_max, row_min, row_max)
        else:
            self._extract_streets_heatmap(z_int, n, col_min, col_max, row_min, row_max, mode)

        self._heatmap_dirty = True
        self._heatmap_generated = True

    def _extract_streets_heatmap(self, z_int, n, col_min, col_max, row_min, row_max, mode):
        """Extract POI or building density from cached mapbox-streets MVT tiles."""
        if mode == "poi":
            target_layers = ("poi_label",)
        elif mode == "building":
            target_layers = ("building",)
        else:
            return

        with self._mvt_lock:
            cached_keys = list(self._mvt_cache.keys())

        for key in cached_keys:
            tz, tx, ty = key
            if tz != z_int:
                continue
            rx = tx % n
            if rx < (col_min % n) and rx > (col_max % n):
                continue
            if ty < row_min or ty > row_max:
                continue

            mvt = None
            with self._mvt_lock:
                mvt = self._mvt_cache.get(key)
            if mvt is None:
                continue

            for lname in target_layers:
                layer = mvt.layers.get(lname)
                if not layer:
                    continue
                ext = layer.extent or MVT_EXTENT
                for feat in layer.features:
                    if mode == "poi":
                        # POI points: geom_type 1 = POINT
                        if feat.geom_type == 1:
                            rings = _get_rings(feat, ext)
                            for ring in rings:
                                for px, py in ring:
                                    fx = (tx + px / ext)
                                    fy = (ty + py / ext)
                                    lon = fx / n * 360.0 - 180.0
                                    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy / n)))
                                    lat = math.degrees(lat_rad)
                                    # Weight by category importance
                                    cat = feat.properties.get("type", "")
                                    w = 1.0
                                    if cat in ("restaurant", "cafe", "bar", "fast_food", "food"):
                                        w = 1.0
                                    elif cat in ("shop", "mall", "supermarket", "clothing"):
                                        w = 0.9
                                    elif cat in ("hotel", "museum", "attraction", "park"):
                                        w = 0.8
                                    else:
                                        w = 0.6
                                    self._heatmap_points_latlon.append((lat, lon, w))

                    elif mode == "building":
                        # Buildings: use centroid of polygon, weight by size
                        if feat.geom_type == 3:  # POLYGON
                            rings = _get_rings(feat, ext)
                            if not rings:
                                continue
                            ring = rings[0]
                            if len(ring) < 3:
                                continue
                            # Compute centroid and approximate area
                            sum_x = sum_y = 0
                            area2 = 0.0
                            for i in range(len(ring) - 1):
                                x0, y0 = ring[i]
                                x1, y1 = ring[i + 1]
                                cross = x0 * y1 - x1 * y0
                                area2 += cross
                                sum_x += (x0 + x1) * cross
                                sum_y += (y0 + y1) * cross
                            if abs(area2) < 1e-6:
                                cx_t = ring[0][0]; cy_t = ring[0][1]
                            else:
                                cx_t = sum_x / (3.0 * area2)
                                cy_t = sum_y / (3.0 * area2)

                            fx = (tx + cx_t / ext)
                            fy = (ty + cy_t / ext)
                            lon = fx / n * 360.0 - 180.0
                            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy / n)))
                            lat = math.degrees(lat_rad)

                            # Weight by building footprint area
                            norm_area = abs(area2) / (ext * ext)
                            height = feat.properties.get("height", 0)
                            if isinstance(height, (int, float)) and height > 0:
                                w = min(1.0, 0.3 + norm_area * 200 + height / 200.0)
                            else:
                                w = min(1.0, 0.2 + norm_area * 300)
                            self._heatmap_points_latlon.append((lat, lon, w))

    def _extract_traffic_heatmap(self, z_int, n, col_min, col_max, row_min, row_max):
        """Extract traffic congestion data from traffic MVT tiles.
        Falls back to road density from streets tiles if traffic tiles unavailable."""

        # First, try to request traffic tiles for visible area
        traffic_found = False
        for ty in range(max(0, row_min), min(n, row_max + 1)):
            for tx in range(col_min, col_max + 1):
                rx = tx % n
                key = (z_int, rx, ty)
                if key in self._traffic_cache:
                    traffic_found = True
                elif key not in self._traffic_pending:
                    self._request_traffic_tile(z_int, rx, ty)

        # Extract from traffic cache if available
        if traffic_found:
            for key, mvt in list(self._traffic_cache.items()):
                tz, tx, ty = key
                if tz != z_int:
                    continue
                layer = mvt.layers.get("traffic")
                if not layer:
                    continue
                ext = layer.extent or MVT_EXTENT
                for feat in layer.features:
                    if feat.geom_type != 2:  # LINESTRING
                        continue
                    congestion = feat.properties.get("congestion", "")
                    cw = {"low": 0.2, "moderate": 0.5, "heavy": 0.8,
                          "severe": 1.0}.get(congestion, 0.1)
                    rings = _get_rings(feat, ext)
                    for ring in rings:
                        # Sample points along road segments
                        step = max(1, len(ring) // 4)
                        for i in range(0, len(ring), step):
                            px, py = ring[i]
                            fx = (tx + px / ext)
                            fy = (ty + py / ext)
                            lon = fx / n * 360.0 - 180.0
                            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy / n)))
                            lat = math.degrees(lat_rad)
                            self._heatmap_points_latlon.append((lat, lon, cw))
            return

        # Fallback: use road density from streets tiles
        with self._mvt_lock:
            cached_keys = list(self._mvt_cache.keys())

        for key in cached_keys:
            tz, tx, ty = key
            if tz != z_int:
                continue

            mvt = None
            with self._mvt_lock:
                mvt = self._mvt_cache.get(key)
            if mvt is None:
                continue

            layer = mvt.layers.get("road")
            if not layer:
                continue
            ext = layer.extent or MVT_EXTENT
            for feat in layer.features:
                if feat.geom_type != 2:  # LINESTRING
                    continue
                road_class = feat.properties.get("class", "")
                cw = {"motorway": 1.0, "trunk": 0.9, "primary": 0.8,
                      "secondary": 0.6, "tertiary": 0.5, "street": 0.3,
                      "service": 0.15, "path": 0.05}.get(road_class, 0.2)
                rings = _get_rings(feat, ext)
                for ring in rings:
                    step = max(1, len(ring) // 3)
                    for i in range(0, len(ring), step):
                        px, py = ring[i]
                        fx = (tx + px / ext)
                        fy = (ty + py / ext)
                        lon = fx / n * 360.0 - 180.0
                        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy / n)))
                        lat = math.degrees(lat_rad)
                        self._heatmap_points_latlon.append((lat, lon, cw))

    def _request_traffic_tile(self, z, x, y):
        """Fetch a traffic tile from mapbox.mapbox-traffic-v1."""
        n = 2 ** z; x = x % n
        if y < 0 or y >= n: return
        key = (z, x, y)
        if key in self._traffic_cache or key in self._traffic_pending:
            return
        if not self.token: return
        if len(self._traffic_pending) >= 4:
            return
        self._traffic_pending.add(key)
        url_str = f"https://api.mapbox.com/v4/mapbox.mapbox-traffic-v1/{z}/{x}/{y}.mvt?access_token={self.token}"
        req = QNetworkRequest(url_str)
        reply = self._nam.get(req)
        self._traffic_replies[key] = reply
        reply.finished.connect(lambda r=reply, k=key: self._on_traffic_data(r, k))

    def _on_traffic_data(self, reply, key):
        """Handle traffic tile response."""
        self._traffic_pending.discard(key)
        self._traffic_replies.pop(key, None)
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = bytes(reply.readAll())
            if data:
                try:
                    mvt = decode_mvt(data)
                    self._traffic_cache[key] = mvt
                    if len(self._traffic_cache) > self._traffic_cache_max:
                        self._traffic_cache.pop(next(iter(self._traffic_cache)), None)
                    # Re-extract if currently in traffic mode
                    if self._heatmap_enabled and self._heatmap_mode == "traffic":
                        self._generate_heatmap_data()
                        self._gl_dirty = True; self.update()
                except Exception:
                    pass
        reply.deleteLater()

    def _screen_to_world(self, sx, sy):
        w, h = self.width(), self.height()
        hw, hh = w / 2.0, h / 2.0
        dx_s, dy_s = sx - hw, sy - hh
        if self._pitch > 0.5:
            pitch_rad = math.radians(self._pitch)
            cos_p = math.cos(pitch_rad)
            fov_f = 1.0 - 0.6 * math.sin(pitch_rad)
            dy_s = (dy_s - h * 0.35 * math.sin(pitch_rad)) / (cos_p * fov_f + (1.0 - cos_p) * 0.6)
        b_r = math.radians(self._bearing)
        cb, sb = math.cos(b_r), math.sin(b_r)
        wx = self._cx + dx_s * cb - dy_s * sb
        wy = self._cy + dx_s * sb + dy_s * cb
        return wx, wy

    # -----------------------------------------------------------------
    #  Mapbox Directions API
    # -----------------------------------------------------------------

    def _request_directions(self):
        if not self.token or len(self._route_points) < 2:
            return
        if self._route_reply is not None:
            try:
                self._route_reply.abort()
                self._route_reply.deleteLater()
            except:
                pass
            self._route_reply = None
        pts = self._route_points[:25]
        coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in pts)
        profile = self._route_profile
        url = (f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords}"
               f"?geometries=geojson&overview=full&access_token={self.token}")
        self._route_pending = True
        req = QNetworkRequest(url)
        reply = self._nam.get(req)
        self._route_reply = reply
        reply.finished.connect(lambda r=reply: self._on_directions_response(r))

    def _on_directions_response(self, reply):
        self._route_pending = False
        self._route_reply = None
        if reply.error() != QNetworkReply.NetworkError.NoError:
            self._route_geometry = list(self._route_points)
            self._route_world = [
                (_lon_to_wx(lon, self._zoom), _lat_to_wy(lat, self._zoom))
                for lat, lon in self._route_geometry
            ]
            self._route_dirty = True
            self._gl_dirty = True; self.update()
            reply.deleteLater()
            return
        try:
            data = json.loads(bytes(reply.readAll()).decode('utf-8'))
            routes = data.get("routes", [])
            if routes:
                route = routes[0]
                coords = route["geometry"]["coordinates"]
                self._route_geometry = [(lat, lon) for lon, lat in coords]
                self._route_total_dist = route.get("distance", 0.0)
                self._route_duration = route.get("duration", 0.0)
                self._route_world = [
                    (_lon_to_wx(lon, self._zoom), _lat_to_wy(lat, self._zoom))
                    for lat, lon in self._route_geometry
                ]
            else:
                self._route_geometry = list(self._route_points)
                self._route_world = [
                    (_lon_to_wx(lon, self._zoom), _lat_to_wy(lat, self._zoom))
                    for lat, lon in self._route_geometry
                ]
        except Exception:
            self._route_geometry = list(self._route_points)
            self._route_world = [
                (_lon_to_wx(lon, self._zoom), _lat_to_wy(lat, self._zoom))
                for lat, lon in self._route_geometry
            ]
        self._route_dirty = True
        self._gl_dirty = True; self.update()
        reply.deleteLater()

    # -----------------------------------------------------------------
    #  POI Detail Popup — Geocoding & Interaction
    # -----------------------------------------------------------------

    def _poi_show_popup(self, poi, screen_x=0, screen_y=0):
        """Show a detail popup for a POI. Triggers geocoding and Wikipedia enrichment."""
        self._poi_popup = poi
        self._poi_popup_timer = 15.0  # longer auto-dismiss (was 8s) for reading detail
        self._poi_popup_fade = 1.0
        self._poi_popup_screen_pos = (screen_x, screen_y)
        self._poi_popup_geocode = None
        self._poi_photo_pixmap = None
        # Check if we already have a cached photo
        name = poi.get("name", "")
        cache_key = name.lower().strip()
        cached = self._poi_detail_cache.get(cache_key)
        if cached and cached.get("photo_pixmap"):
            self._poi_photo_pixmap = cached["photo_pixmap"]
        # Fire reverse geocode request
        self._request_poi_geocode(poi["lon"], poi["lat"])
        # Fire Wikipedia detail request (will use geocode city if available after delay)
        self._poi_request_detail(poi)
        self._gl_dirty = True; self.update()

    def _poi_dismiss_popup(self):
        """Dismiss the current POI popup."""
        self._poi_popup = None
        self._poi_popup_geocode = None
        self._poi_popup_timer = 0.0
        self._poi_photo_pixmap = None
        if self._poi_popup_geocode_reply:
            try: self._poi_popup_geocode_reply.abort(); self._poi_popup_geocode_reply.deleteLater()
            except: pass
            self._poi_popup_geocode_reply = None
        if self._poi_detail_reply:
            try: self._poi_detail_reply.abort(); self._poi_detail_reply.deleteLater()
            except: pass
            self._poi_detail_reply = None
        if self._poi_photo_reply:
            try: self._poi_photo_reply.abort(); self._poi_photo_reply.deleteLater()
            except: pass
            self._poi_photo_reply = None
        self._gl_dirty = True; self.update()

    def _request_poi_geocode(self, lon, lat):
        """Use Mapbox Geocoding API to get address details for a location."""
        if not self.token:
            return
        if self._poi_popup_geocode_reply:
            try: self._poi_popup_geocode_reply.abort(); self._poi_popup_geocode_reply.deleteLater()
            except: pass
        url = (f"https://api.mapbox.com/geocoding/v5/mapbox.places/"
               f"{lon:.6f},{lat:.6f}.json"
               f"?types=address,poi&limit=1&access_token={self.token}")
        req = QNetworkRequest(url)
        reply = self._nam.get(req)
        self._poi_popup_geocode_reply = reply
        reply.finished.connect(lambda r=reply: self._on_poi_geocode_response(r))

    def _on_poi_geocode_response(self, reply):
        """Handle geocoding API response."""
        self._poi_popup_geocode_reply = None
        if reply.error() != QNetworkReply.NetworkError.NoError:
            reply.deleteLater()
            return
        try:
            data = json.loads(bytes(reply.readAll()).decode('utf-8'))
            features = data.get("features", [])
            if features:
                feat = features[0]
                props = feat.get("properties", {})
                context = feat.get("context", [])
                address = feat.get("place_name", "")
                # Extract structured address components
                street = ""
                city = ""
                postcode = ""
                country = ""
                category = props.get("category", "")
                phone = ""
                # Mapbox Places API response
                if "address" in feat:
                    street = feat.get("address", "") + " " + feat.get("text", "")
                elif feat.get("text"):
                    street = feat.get("text", "")
                for ctx in context:
                    ctx_id = ctx.get("id", "")
                    if ctx_id.startswith("place"):
                        city = ctx.get("text", "")
                    elif ctx_id.startswith("postcode"):
                        postcode = ctx.get("text", "")
                    elif ctx_id.startswith("country"):
                        country = ctx.get("text", "")
                # Properties from POI result
                if "properties" in feat:
                    phone = feat["properties"].get("tel", "")
                    if not category:
                        category = feat["properties"].get("category", "")

                self._poi_popup_geocode = {
                    "address": address,
                    "street": street.strip(),
                    "city": city,
                    "postcode": postcode,
                    "country": country,
                    "category": category,
                    "phone": phone,
                }
        except Exception:
            pass
        reply.deleteLater()
        self._gl_dirty = True; self.update()

    # -----------------------------------------------------------------
    #  POI Detail Enrichment — Wikipedia summary + photo
    # -----------------------------------------------------------------

    def _poi_request_detail(self, poi):
        """Fetch Wikipedia summary and thumbnail for a POI using the Wikipedia API.
        Results are cached by POI name to avoid repeat fetches."""
        name = poi.get("name", "")
        if not name:
            return
        cache_key = name.lower().strip()
        if cache_key in self._poi_detail_cache:
            # Already fetched — apply cached data immediately
            cached = self._poi_detail_cache[cache_key]
            if cached.get("photo_pixmap"):
                self._poi_photo_pixmap = cached["photo_pixmap"]
            self._gl_dirty = True; self.update()
            return

        # Cancel any pending detail request
        if self._poi_detail_reply:
            try: self._poi_detail_reply.abort(); self._poi_detail_reply.deleteLater()
            except: pass
            self._poi_detail_reply = None
        if self._poi_photo_reply:
            try: self._poi_photo_reply.abort(); self._poi_photo_reply.deleteLater()
            except: pass
            self._poi_photo_reply = None
        self._poi_photo_pixmap = None

        # Wikipedia API — search for the POI name, get summary + thumbnail
        import urllib.parse
        city = ""
        if self._poi_popup_geocode and self._poi_popup_geocode.get("city"):
            city = self._poi_popup_geocode["city"]
        search_term = f"{name} {city}".strip() if city else name
        encoded = urllib.parse.quote(search_term)
        url = (f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
               f"?redirect=true")
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"User-Agent", b"MapboxPOIViewer/1.0")
        reply = self._nam.get(req)
        self._poi_detail_reply = reply
        reply.finished.connect(lambda r=reply: self._on_poi_detail_response(r, cache_key))

    def _on_poi_detail_response(self, reply, cache_key):
        """Handle Wikipedia summary API response."""
        self._poi_detail_reply = None
        detail = {"summary": "", "photo_url": "", "photo_pixmap": None,
                  "wiki_url": "", "description": ""}
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                # Extract summary text
                extract = data.get("extract", "")
                if extract:
                    # Trim to reasonable length for popup display
                    if len(extract) > 300:
                        # Find sentence break near 300 chars
                        cut = extract[:300].rfind(". ")
                        if cut > 100:
                            extract = extract[:cut + 1]
                        else:
                            extract = extract[:297] + "..."
                    detail["summary"] = extract
                # Short description
                detail["description"] = data.get("description", "")
                # Wikipedia page URL
                content_urls = data.get("content_urls", {})
                if content_urls.get("desktop", {}).get("page"):
                    detail["wiki_url"] = content_urls["desktop"]["page"]
                # Thumbnail photo
                thumb = data.get("thumbnail", {})
                if thumb.get("source"):
                    detail["photo_url"] = thumb["source"]
            except Exception:
                pass
        reply.deleteLater()
        self._poi_detail_cache[cache_key] = detail
        # If we got a photo URL, start downloading it
        if detail["photo_url"]:
            self._poi_download_photo(detail["photo_url"], cache_key)
        self._gl_dirty = True; self.update()

    def _poi_download_photo(self, url, cache_key):
        """Download a photo thumbnail from a URL."""
        if self._poi_photo_reply:
            try: self._poi_photo_reply.abort(); self._poi_photo_reply.deleteLater()
            except: pass
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"User-Agent", b"MapboxPOIViewer/1.0")
        reply = self._nam.get(req)
        self._poi_photo_reply = reply
        reply.finished.connect(lambda r=reply: self._on_poi_photo_response(r, cache_key))

    def _on_poi_photo_response(self, reply, cache_key):
        """Handle photo download response."""
        self._poi_photo_reply = None
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                img_data = bytes(reply.readAll())
                pix = QPixmap()
                pix.loadFromData(img_data)
                if not pix.isNull():
                    # Scale to fit popup (max 260×140)
                    pix = pix.scaled(260, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    if cache_key in self._poi_detail_cache:
                        self._poi_detail_cache[cache_key]["photo_pixmap"] = pix
                    self._poi_photo_pixmap = pix
            except Exception:
                pass
        reply.deleteLater()
        self._gl_dirty = True; self.update()

    def _poi_get_current_detail(self):
        """Get the cached detail dict for the currently shown popup POI."""
        if not self._poi_popup:
            return None
        name = self._poi_popup.get("name", "")
        cache_key = name.lower().strip()
        return self._poi_detail_cache.get(cache_key)

    def _poi_open_wiki(self, poi=None):
        """Open the Wikipedia page for a POI in the default browser."""
        target = poi or self._poi_popup
        if target is None:
            return
        name = target.get("name", "")
        cache_key = name.lower().strip()
        detail = self._poi_detail_cache.get(cache_key)
        if detail and detail.get("wiki_url"):
            QDesktopServices.openUrl(QUrl(detail["wiki_url"]))
        else:
            # Fallback: search Wikipedia
            import urllib.parse
            q = urllib.parse.quote(name)
            QDesktopServices.openUrl(QUrl(f"https://en.wikipedia.org/wiki/Special:Search/{q}"))

    # -----------------------------------------------------------------
    #  2D hover detection for POI popups
    # -----------------------------------------------------------------

    def _poi_hover_check(self, mouse_x, mouse_y):
        """Check if mouse is hovering over a POI in 2D mode. Triggers popup after dwell."""
        if self._immersive or not self._poi_enabled or self._route_mode:
            self._poi_hover_poi = None
            self._poi_hover_timer = 0.0
            return
        hit = self._poi_hit_test_2d(mouse_x, mouse_y)
        if hit:
            if self._poi_hover_poi and self._poi_hover_poi.get("name") == hit.get("name"):
                # Same POI — accumulate dwell time
                self._poi_hover_timer += 1.0 / 60.0
                if self._poi_hover_timer > 0.35 and self._poi_popup is None:
                    # Show popup after 0.35s hover dwell
                    self._poi_show_popup(hit, mouse_x, mouse_y)
            else:
                # New POI under cursor
                self._poi_hover_poi = hit
                self._poi_hover_timer = 0.0
        else:
            if self._poi_hover_poi:
                self._poi_hover_poi = None
                self._poi_hover_timer = 0.0

    def _poi_hit_test_2d(self, screen_x, screen_y):
        """Test if a screen position hits a POI marker on the 2D map.
        Returns the POI dict if hit, None otherwise."""
        if not self._poi_enabled or not self._poi_cache:
            return None
        z = self._zoom
        bearing_rad = math.radians(self._bearing)
        pitch_rad = math.radians(self._pitch)
        cos_b = math.cos(bearing_rad); sin_b = math.sin(bearing_rad)
        cos_p = math.cos(pitch_rad) if self._pitch > 0.5 else 1.0
        fov_f = (1.0 - 0.6 * math.sin(pitch_rad)) if self._pitch > 0.5 else 1.0
        w, h = self.width(), self.height()
        hw, hh = w / 2.0, h / 2.0
        # Hit radius in screen pixels — generous for easy clicking
        hit_radius = max(8.0, 12.0 * min(1.5, z / 14.0))
        best_dist = hit_radius * hit_radius
        best_poi = None
        for poi in self._poi_cache:
            wx, wy = poi["wx"], poi["wy"]
            dx = wx - self._cx; dy = wy - self._cy
            sx = dx * cos_b + dy * sin_b
            sy = -dx * sin_b + dy * cos_b
            if self._pitch > 0.5:
                sy = sy * (cos_p * fov_f + (1.0 - cos_p) * 0.6) + h * 0.35 * math.sin(pitch_rad)
            px = hw + sx; py = hh + sy
            d2 = (px - screen_x) ** 2 + (py - screen_y) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_poi = poi
        return best_poi

    def _poi_check_proximity_immersive(self):
        """In immersive mode, check if player is near a POI and auto-show popup.
        If already showing a popup for a nearby POI, keep it alive while within the halo."""
        if not self._poi_enabled or not self._poi_visible_list:
            return
        cam_wx = self._car_wx if self._car_mode else self._fp_cx
        cam_wy = self._car_wy if self._car_mode else self._fp_cy
        z = self._zoom
        lat = _wy_to_lat(cam_wy, z)
        mpp = (40_075_000 * math.cos(math.radians(lat))) / _world_size(z)
        trigger_dist_m = 25.0 if not self._car_mode else 35.0
        # Stay-alive radius is larger than trigger radius — keeps popup open while near halo
        stay_dist_m = 35.0 if not self._car_mode else 50.0
        trigger_dist_px = trigger_dist_m / max(mpp, 0.001)
        stay_dist_px = stay_dist_m / max(mpp, 0.001)

        # If currently showing a popup, check if player is still within stay radius
        if self._poi_popup is not None:
            dx = self._poi_popup["wx"] - cam_wx
            dy = self._poi_popup["wy"] - cam_wy
            dist = math.hypot(dx, dy)
            if dist < stay_dist_px:
                # Still near — keep the popup alive (reset timer to prevent dismiss)
                if self._poi_popup_timer < 5.0:
                    self._poi_popup_timer = 5.0
                    self._poi_popup_fade = 1.0
                return
            # Walked away from the POI — let the timer run out naturally

        if self._poi_proximity_cooldown > 0:
            return

        for poi in self._poi_visible_list:
            dx = poi["wx"] - cam_wx
            dy = poi["wy"] - cam_wy
            dist = math.hypot(dx, dy)
            if dist < trigger_dist_px:
                # Don't re-trigger the same POI
                if self._poi_proximity_poi and self._poi_proximity_poi.get("name") == poi.get("name"):
                    if self._poi_popup is not None:
                        return  # already showing this one
                self._poi_proximity_poi = poi
                self._poi_proximity_cooldown = 3.0  # don't trigger again for 3s
                self._poi_show_popup(poi, self.width() // 2, 80)
                return

        # Background prefetch: request Wikipedia details for nearby POIs
        # so their photos are ready for billboard display (one per tick cycle)
        prefetch_dist_m = 100.0
        prefetch_dist_px = prefetch_dist_m / max(mpp, 0.001)
        for poi in self._poi_visible_list:
            name = poi.get("name", "")
            if not name:
                continue
            cache_key = name.lower().strip()
            if cache_key in self._poi_detail_cache:
                continue  # already fetched
            dx = poi["wx"] - cam_wx
            dy = poi["wy"] - cam_wy
            dist = math.hypot(dx, dy)
            if dist < prefetch_dist_px:
                # Only one fetch at a time
                if self._poi_detail_reply is None and self._poi_photo_reply is None:
                    self._poi_request_detail(poi)
                break  # only one per tick

    def _poi_rebuild_sorted_list(self):
        """Rebuild the distance-sorted POI list for selection cycling."""
        src = self._poi_visible_list if self._immersive else self._poi_cache
        if not src:
            self._poi_sorted_list = []
            return
        cam_wx = self._car_wx if (self._immersive and self._car_mode) else (self._fp_cx if self._immersive else self._cx)
        cam_wy = self._car_wy if (self._immersive and self._car_mode) else (self._fp_cy if self._immersive else self._cy)
        z = self._zoom
        lat = _wy_to_lat(cam_wy, z)
        mpp = (40_075_000 * math.cos(math.radians(lat))) / _world_size(z)
        scored = []
        for poi in src:
            dx = poi["wx"] - cam_wx
            dy = poi["wy"] - cam_wy
            dist_m = math.hypot(dx, dy) * mpp
            scored.append((dist_m, poi))
        scored.sort(key=lambda x: x[0])
        self._poi_sorted_list = scored

    def _poi_select_next(self):
        """Select the next POI in the sorted list (+ key / Down)."""
        if not self._poi_enabled:
            return
        self._poi_rebuild_sorted_list()
        if not self._poi_sorted_list:
            return
        self._poi_select_active = True
        max_idx = min(len(self._poi_sorted_list), 12) - 1
        if self._poi_select_idx < max_idx:
            self._poi_select_idx += 1
        else:
            self._poi_select_idx = 0  # wrap around
        self._poi_show_selected()

    def _poi_select_prev(self):
        """Select the previous POI in the sorted list (- key / Up)."""
        if not self._poi_enabled:
            return
        self._poi_rebuild_sorted_list()
        if not self._poi_sorted_list:
            return
        self._poi_select_active = True
        max_idx = min(len(self._poi_sorted_list), 12) - 1
        if self._poi_select_idx > 0:
            self._poi_select_idx -= 1
        else:
            self._poi_select_idx = max_idx  # wrap around
        self._poi_show_selected()

    def _poi_show_selected(self):
        """Show popup for the currently selected POI."""
        if not self._poi_sorted_list or self._poi_select_idx < 0:
            return
        idx = min(self._poi_select_idx, len(self._poi_sorted_list) - 1)
        _, poi = self._poi_sorted_list[idx]
        self._poi_show_popup(poi, 12, 12)
        self._gl_dirty = True; self.update()

    def _poi_navigate_to_selected(self):
        """Create a route from the current position to the selected POI."""
        if not self._poi_sorted_list or self._poi_select_idx < 0:
            return
        idx = min(self._poi_select_idx, len(self._poi_sorted_list) - 1)
        _, poi = self._poi_sorted_list[idx]

        # Get current position
        if self._immersive:
            cam_wx = self._car_wx if self._car_mode else self._fp_cx
            cam_wy = self._car_wy if self._car_mode else self._fp_cy
            z = self._zoom
            start_lat = _wy_to_lat(cam_wy, z)
            start_lon = _wx_to_lon(cam_wx, z)
        else:
            start_lat = _wy_to_lat(self._cy, self._zoom)
            start_lon = _wx_to_lon(self._cx, self._zoom)

        dest_lat = poi["lat"]
        dest_lon = poi["lon"]

        # Set up route points and request directions
        self._route_points = [(start_lat, start_lon), (dest_lat, dest_lon)]
        self._route_world = [
            (_lon_to_wx(start_lon, self._zoom), _lat_to_wy(start_lat, self._zoom)),
            (_lon_to_wx(dest_lon, self._zoom), _lat_to_wy(dest_lat, self._zoom)),
        ]
        self._route_profile = "driving" if self._car_mode else "walking"
        self._route_dirty = True
        self._request_directions()
        self._gl_dirty = True; self.update()

    def _poi_clear_selection(self):
        """Clear the POI selection."""
        self._poi_select_idx = -1
        self._poi_select_active = False
        self._gl_dirty = True; self.update()

    # -----------------------------------------------------------------
    #  Search bar — geocoding search + route to destination
    # -----------------------------------------------------------------

    def _search_open(self):
        """Open the search bar overlay."""
        self._search_active = True
        self._search_text = ""
        self._search_results = []
        self._search_selected = 0
        self._search_cursor_blink = 0.0
        self._gl_dirty = True; self.update()

    def _search_close(self):
        """Close the search bar and cancel any pending request."""
        self._search_active = False
        self._search_text = ""
        self._search_results = []
        if self._search_reply:
            try: self._search_reply.abort(); self._search_reply.deleteLater()
            except: pass
            self._search_reply = None
        self._gl_dirty = True; self.update()

    def _search_submit(self):
        """Submit the current search text for geocoding."""
        if not self._search_text.strip():
            return
        if not self.token:
            return
        # Cancel any pending request
        if self._search_reply:
            try: self._search_reply.abort(); self._search_reply.deleteLater()
            except: pass

        # Get current camera location for proximity bias
        if self._immersive:
            cam_wx = self._car_wx if self._car_mode else self._fp_cx
            cam_wy = self._car_wy if self._car_mode else self._fp_cy
            prox_lon = _wx_to_lon(cam_wx, self._zoom)
            prox_lat = _wy_to_lat(cam_wy, self._zoom)
        else:
            prox_lon = _wx_to_lon(self._cx, self._zoom)
            prox_lat = _wy_to_lat(self._cy, self._zoom)

        import urllib.parse
        q = urllib.parse.quote(self._search_text.strip())
        url = (f"https://api.mapbox.com/geocoding/v5/mapbox.places/{q}.json"
               f"?proximity={prox_lon:.6f},{prox_lat:.6f}"
               f"&limit=5&access_token={self.token}")
        req = QNetworkRequest(QUrl(url))
        reply = self._nam.get(req)
        self._search_reply = reply
        reply.finished.connect(lambda r=reply: self._on_search_response(r))

    def _on_search_response(self, reply):
        """Handle geocoding search results."""
        self._search_reply = None
        self._search_results = []
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                for feat in data.get("features", [])[:5]:
                    coords = feat.get("center", [0, 0])  # [lon, lat]
                    self._search_results.append({
                        "name": feat.get("text", ""),
                        "place_name": feat.get("place_name", ""),
                        "lat": coords[1],
                        "lon": coords[0],
                    })
            except Exception:
                pass
        reply.deleteLater()
        self._search_selected = 0
        self._gl_dirty = True; self.update()

    def _search_navigate_to_selected(self):
        """Create a route from current position to the selected search result."""
        if not self._search_results:
            return
        idx = min(self._search_selected, len(self._search_results) - 1)
        result = self._search_results[idx]

        # Get current position
        if self._immersive:
            cam_wx = self._car_wx if self._car_mode else self._fp_cx
            cam_wy = self._car_wy if self._car_mode else self._fp_cy
            z = self._zoom
            start_lat = _wy_to_lat(cam_wy, z)
            start_lon = _wx_to_lon(cam_wx, z)
        else:
            start_lat = _wy_to_lat(self._cy, self._zoom)
            start_lon = _wx_to_lon(self._cx, self._zoom)

        dest_lat = result["lat"]
        dest_lon = result["lon"]

        # Set up route and request directions
        self._route_points = [(start_lat, start_lon), (dest_lat, dest_lon)]
        self._route_world = [
            (_lon_to_wx(start_lon, self._zoom), _lat_to_wy(start_lat, self._zoom)),
            (_lon_to_wx(dest_lon, self._zoom), _lat_to_wy(dest_lat, self._zoom)),
        ]
        self._route_profile = "driving" if (self._immersive and self._car_mode) else "walking"
        self._route_dirty = True
        self._request_directions()

        # Close search bar
        self._search_close()

        # Fly camera to destination
        if not self._immersive:
            self.flyTo(dest_lat, dest_lon, zoom=max(self._zoom, 15.0))

    def _search_handle_key(self, event):
        """Handle keyboard input while the search bar is active.
        Returns True if the event was consumed."""
        k = event.key()

        if k == Qt.Key_Escape:
            self._search_close()
            return True

        if k == Qt.Key_Return or k == Qt.Key_Enter:
            if self._search_results:
                self._search_navigate_to_selected()
            else:
                # No results yet — submit the search
                self._search_submit()
            return True

        if k == Qt.Key_Down:
            if self._search_results:
                self._search_selected = min(self._search_selected + 1, len(self._search_results) - 1)
                self._gl_dirty = True; self.update()
            return True

        if k == Qt.Key_Up:
            if self._search_results:
                self._search_selected = max(self._search_selected - 1, 0)
                self._gl_dirty = True; self.update()
            return True

        if k == Qt.Key_Backspace:
            if self._search_text:
                self._search_text = self._search_text[:-1]
                # Auto-search after typing (debounced via results clearing)
                self._search_results = []
                self._search_selected = 0
                if len(self._search_text) >= 2:
                    self._search_submit()
                self._gl_dirty = True; self.update()
            return True

        # Regular text input
        text = event.text()
        if text and text.isprintable() and len(text) == 1:
            self._search_text += text
            self._search_results = []
            self._search_selected = 0
            # Auto-search when 3+ characters typed
            if len(self._search_text) >= 3:
                self._search_submit()
            self._gl_dirty = True; self.update()
            return True

        return False

    def _draw_search_bar(self, painter, w, h):
        """Draw the search bar overlay."""
        if not self._search_active:
            return

        # Dim background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.drawRect(0, 0, w, h)

        # Search bar dimensions
        bar_w = min(420, w - 40)
        bar_h = 44
        bar_x = (w - bar_w) // 2
        bar_y = 60

        # Bar background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 22, 32, 245))
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 12, 12)
        # Border
        painter.setPen(QPen(QColor(60, 120, 220, 180), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 12, 12)

        # Search icon
        painter.setPen(QColor(100, 160, 255, 200))
        painter.setFont(QFont("sans-serif", 16))
        painter.drawText(bar_x + 12, bar_y + 4, 32, bar_h - 8, Qt.AlignVCenter, "\U0001F50D")

        # Input text
        painter.setPen(QColor(235, 240, 255))
        painter.setFont(QFont("sans-serif", 14))
        display_text = self._search_text
        # Blinking cursor
        self._search_cursor_blink += 1.0 / 60.0
        if int(self._search_cursor_blink * 2) % 2 == 0:
            display_text += "\u2502"  # thin vertical bar cursor
        if not self._search_text:
            painter.setPen(QColor(100, 120, 160, 160))
            display_text = "Search destination..."
        painter.drawText(bar_x + 48, bar_y, bar_w - 60, bar_h,
                        Qt.AlignLeft | Qt.AlignVCenter, display_text)

        # Results dropdown
        if self._search_results:
            res_x = bar_x
            res_y = bar_y + bar_h + 4
            res_w = bar_w
            res_row_h = 52
            res_h = len(self._search_results) * res_row_h + 8

            # Dropdown background
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(14, 18, 28, 245))
            painter.drawRoundedRect(res_x, res_y, res_w, res_h, 10, 10)
            painter.setPen(QPen(QColor(50, 80, 140, 120), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(res_x, res_y, res_w, res_h, 10, 10)

            for i, result in enumerate(self._search_results):
                ry = res_y + 4 + i * res_row_h
                is_sel = (i == self._search_selected)

                # Selection highlight
                if is_sel:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(40, 80, 160, 80))
                    painter.drawRoundedRect(res_x + 4, ry, res_w - 8, res_row_h - 2, 6, 6)

                # Location pin
                painter.setPen(QColor(80, 160, 255) if is_sel else QColor(100, 120, 160))
                painter.setFont(QFont("sans-serif", 14))
                painter.drawText(res_x + 12, ry, 28, res_row_h,
                                Qt.AlignVCenter, "\U0001F4CD")

                # Name (primary line)
                name = result["name"]
                if len(name) > 40:
                    name = name[:39] + "\u2026"
                painter.setPen(QColor(230, 240, 255) if is_sel else QColor(190, 200, 220))
                painter.setFont(QFont("sans-serif", 11, QFont.Bold if is_sel else QFont.Normal))
                painter.drawText(res_x + 44, ry + 2, res_w - 60, 24,
                                Qt.AlignLeft | Qt.AlignVCenter, name)

                # Full place name (secondary line)
                place = result["place_name"]
                if len(place) > 55:
                    place = place[:54] + "\u2026"
                painter.setPen(QColor(120, 140, 175) if is_sel else QColor(90, 105, 135))
                painter.setFont(QFont("sans-serif", 8))
                painter.drawText(res_x + 44, ry + 24, res_w - 60, 22,
                                Qt.AlignLeft | Qt.AlignVCenter, place)

            # Hint at bottom
            hint_y = res_y + res_h + 4
            painter.setPen(QColor(70, 90, 130, 160))
            painter.setFont(QFont("monospace", 8))
            painter.drawText(res_x, hint_y, res_w, 18,
                            Qt.AlignCenter,
                            "\u2191\u2193 select   \u23CE navigate   ESC cancel")

        elif self._search_text and len(self._search_text) >= 3 and self._search_reply:
            # Loading indicator
            dots = "." * ((int(self._search_cursor_blink * 4)) % 4 + 1)
            painter.setPen(QColor(100, 130, 180, 180))
            painter.setFont(QFont("monospace", 9))
            painter.drawText(bar_x, bar_y + bar_h + 8, bar_w, 20,
                            Qt.AlignCenter, f"searching{dots}")
        elif self._search_text and len(self._search_text) >= 3 and not self._search_results:
            # No results
            painter.setPen(QColor(140, 100, 100, 180))
            painter.setFont(QFont("sans-serif", 9))
            painter.drawText(bar_x, bar_y + bar_h + 8, bar_w, 20,
                            Qt.AlignCenter, "No results found")
        else:
            # Typing hint
            painter.setPen(QColor(70, 90, 130, 140))
            painter.setFont(QFont("monospace", 8))
            painter.drawText(bar_x, bar_y + bar_h + 8, bar_w, 18,
                            Qt.AlignCenter, "type to search   \u23CE search   ESC cancel")

    # -----------------------------------------------------------------
    #  Isochrone overlay — reachability polygons via Mapbox Isochrone API
    # -----------------------------------------------------------------

    def _iso_toggle(self):
        """Y key: if isochrones showing, disable. If in 2D, enter placement mode. If immersive, place at current pos."""
        if self._iso_enabled:
            # Disable isochrones
            self._iso_enabled = False
            self._iso_placing = False
            self._iso_contours = []
            self._iso_dirty = True
            if not self._immersive:
                self.setCursor(QCursor(Qt.OpenHandCursor))
            self._gl_dirty = True; self.update()
        elif self._iso_placing:
            # Cancel placement (2D only)
            self._iso_placing = False
            self.setCursor(QCursor(Qt.OpenHandCursor))
            self._gl_dirty = True; self.update()
        elif self._immersive:
            # In immersive: place at current position immediately
            cam_wx = self._car_wx if self._car_mode else self._fp_cx
            cam_wy = self._car_wy if self._car_mode else self._fp_cy
            lat = _wy_to_lat(cam_wy, self._zoom)
            lon = _wx_to_lon(cam_wx, self._zoom)
            self._iso_place_at(lat, lon)
        else:
            # 2D: enter placement mode
            self._iso_placing = True
            self.setCursor(QCursor(Qt.CrossCursor))
            self._gl_dirty = True; self.update()

    def _iso_place_at(self, lat, lon):
        """Place isochrone at given coordinates and request from API."""
        self._iso_placing = False
        self._iso_enabled = True
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self._iso_center = (lat, lon)
        self._iso_read_panel_params()
        self._iso_request()

    def _iso_read_panel_params(self):
        """Read isochrone parameters from the settings panel."""
        if self._panel:
            try:
                combo = self._panel._iso_combo
                self._iso_profile = combo.currentText()
            except: pass
            try:
                m1 = self._panel._iso_spin1.value()
                m2 = self._panel._iso_spin2.value()
                m3 = self._panel._iso_spin3.value()
                # Deduplicate and sort
                mins = sorted(set([m1, m2, m3]))
                if mins:
                    self._iso_minutes = mins
            except: pass

    def _iso_request(self):
        """Request isochrones from the Mapbox Isochrone API."""
        if not self.token:
            return
        # Cancel pending
        if self._iso_reply:
            try: self._iso_reply.abort(); self._iso_reply.deleteLater()
            except: pass
            self._iso_reply = None

        lat, lon = self._iso_center
        minutes = ",".join(str(m) for m in self._iso_minutes)
        url = (f"https://api.mapbox.com/isochrone/v1/mapbox/{self._iso_profile}/"
               f"{lon:.6f},{lat:.6f}"
               f"?contours_minutes={minutes}&polygons=true&denoise=1"
               f"&access_token={self.token}")
        req = QNetworkRequest(QUrl(url))
        reply = self._nam.get(req)
        self._iso_reply = reply
        reply.finished.connect(lambda r=reply: self._on_iso_response(r))

    def _on_iso_response(self, reply):
        """Handle isochrone API response — parse GeoJSON polygons."""
        self._iso_reply = None
        self._iso_contours = []
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                features = data.get("features", [])
                # Features come in order: largest contour first (15min), smallest last (5min)
                for feat in features:
                    props = feat.get("properties", {})
                    minutes = props.get("contour", 0)
                    geom = feat.get("geometry", {})
                    coords_list = geom.get("coordinates", [])
                    # Polygon: coordinates is [ [ring], [hole], ... ]
                    if geom.get("type") == "Polygon" and coords_list:
                        ring = coords_list[0]  # outer ring
                        polygon = [(lat, lon) for lon, lat in ring]
                        self._iso_contours.append({
                            "minutes": minutes,
                            "polygon": polygon,
                        })
                    elif geom.get("type") == "MultiPolygon" and coords_list:
                        # Take the largest polygon
                        for poly_coords in coords_list:
                            if poly_coords:
                                ring = poly_coords[0]
                                polygon = [(lat, lon) for lon, lat in ring]
                                self._iso_contours.append({
                                    "minutes": minutes,
                                    "polygon": polygon,
                                })
                                break
            except Exception:
                pass
        reply.deleteLater()
        self._iso_dirty = True
        self._gl_dirty = True; self.update()

    def _draw_isochrone_overlay(self, painter, w, h):
        """Draw isochrone polygons on the 2D map."""
        if not self._iso_enabled or not self._iso_contours:
            return
        z = self._zoom
        bearing_rad = math.radians(self._bearing)
        pitch_rad = math.radians(self._pitch)
        cos_b = math.cos(bearing_rad); sin_b = math.sin(bearing_rad)
        cos_p = math.cos(pitch_rad) if self._pitch > 0.5 else 1.0
        fov_f = (1.0 - 0.6 * math.sin(pitch_rad)) if self._pitch > 0.5 else 1.0
        hw, hh = w / 2.0, h / 2.0

        painter.setRenderHint(QPainter.Antialiasing, True)

        # Draw from largest to smallest (so smaller ones overlay on top)
        for i, contour in enumerate(self._iso_contours):
            polygon = contour["polygon"]
            minutes = contour["minutes"]
            if not polygon:
                continue

            # Map contour to a color index based on minutes
            color_idx = -1
            for ci, m in enumerate(self._iso_minutes):
                if minutes == m:
                    color_idx = ci; break
            if color_idx < 0:
                color_idx = min(i, len(self._iso_colors) - 1)

            fill = self._iso_colors[min(color_idx, len(self._iso_colors) - 1)]
            outline = self._iso_outline_colors[min(color_idx, len(self._iso_outline_colors) - 1)]

            # Project polygon to screen
            qpoly = QPolygonF()
            for lat, lon in polygon:
                wx = _lon_to_wx(lon, z)
                wy = _lat_to_wy(lat, z)
                dx = wx - self._cx; dy = wy - self._cy
                sx = dx * cos_b + dy * sin_b
                sy = -dx * sin_b + dy * cos_b
                if self._pitch > 0.5:
                    sy = sy * (cos_p * fov_f + (1.0 - cos_p) * 0.6) + h * 0.35 * math.sin(pitch_rad)
                qpoly.append(QPointF(hw + sx, hh + sy))

            # Fill
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            painter.drawPolygon(qpoly)

            # Outline
            painter.setPen(QPen(outline, 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(qpoly)

        # Center pin marker
        clat, clon = self._iso_center
        cwx = _lon_to_wx(clon, z); cwy = _lat_to_wy(clat, z)
        cdx = cwx - self._cx; cdy = cwy - self._cy
        csx = cdx * cos_b + cdy * sin_b
        csy = -cdx * sin_b + cdy * cos_b
        if self._pitch > 0.5:
            csy = csy * (cos_p * fov_f + (1.0 - cos_p) * 0.6) + h * 0.35 * math.sin(pitch_rad)
        cpx, cpy = hw + csx, hh + csy
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(120, 50, 200, 220))
        painter.drawEllipse(QPointF(cpx, cpy), 8, 8)
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.drawEllipse(QPointF(cpx, cpy), 4, 4)

        # Label badge
        profile_icons = {"driving": "\U0001F697", "walking": "\U0001F6B6", "cycling": "\U0001F6B2"}
        icon = profile_icons.get(self._iso_profile, "\U0001F697")
        if self._iso_contours:
            label = f"{icon} {self._iso_profile.title()} isochrone"
        elif self._iso_reply:
            label = f"{icon} Loading..."
        else:
            label = f"{icon} No data"
        lw = len(label) * 8 + 20
        lx = w // 2 - lw // 2
        ly = 10
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(20, 10, 40, 220))
        painter.drawRoundedRect(lx, ly, lw, 24, 12, 12)
        painter.setPen(QColor(180, 140, 255))
        painter.setFont(QFont("sans-serif", 9, QFont.Bold))
        painter.drawText(lx, ly, lw, 24, Qt.AlignCenter, label)

        # Legend (small labels for each contour ring)
        legend_x = w // 2 - 80
        legend_y = 38
        for ci, m in enumerate(self._iso_minutes):
            fill = self._iso_colors[min(ci, len(self._iso_colors) - 1)]
            outline = self._iso_outline_colors[min(ci, len(self._iso_outline_colors) - 1)]
            bx = legend_x + ci * 55
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(fill.red(), fill.green(), fill.blue(), 160))
            painter.drawRoundedRect(bx, legend_y, 50, 18, 9, 9)
            painter.setPen(outline)
            painter.setFont(QFont("monospace", 8, QFont.Bold))
            painter.drawText(bx, legend_y, 50, 18, Qt.AlignCenter, f"{m} min")

    def _poi_tick_popup(self, dt):
        """Tick the POI popup timer for auto-dismiss and fade."""
        if self._poi_popup is None:
            return
        self._poi_popup_timer -= dt
        if self._poi_popup_timer <= 0:
            self._poi_popup_fade -= dt * 3.0
            if self._poi_popup_fade <= 0:
                self._poi_dismiss_popup()
            else:
                self._gl_dirty = True; self.update()
        if self._poi_proximity_cooldown > 0:
            self._poi_proximity_cooldown -= dt

    def _poi_open_web(self, poi=None):
        """Open a POI's website or a Google Maps search in the default browser."""
        target = poi or self._poi_popup
        if target is None:
            return
        name = target.get("name", "")
        lat = target.get("lat", 0.0)
        lon = target.get("lon", 0.0)
        cat = target.get("display_type", target.get("type", ""))

        # Try Google Maps search with name + coordinates for precise results
        import urllib.parse
        query = urllib.parse.quote(f"{name} {cat}")
        url = f"https://www.google.com/maps/search/{query}/@{lat:.6f},{lon:.6f},18z"
        QDesktopServices.openUrl(QUrl(url))

    def _poi_search_web(self, poi=None):
        """Open a web search for ordering / website of the POI."""
        target = poi or self._poi_popup
        if target is None:
            return
        name = target.get("name", "")
        cat = target.get("display_type", target.get("type", ""))
        city = ""
        geo = self._poi_popup_geocode
        if geo:
            city = geo.get("city", "")

        import urllib.parse
        # Use a search query that's likely to find their website / ordering page
        search_terms = f"{name} {cat} {city}".strip()
        query = urllib.parse.quote(search_terms)
        url = f"https://www.google.com/search?q={query}"
        QDesktopServices.openUrl(QUrl(url))

    def _draw_poi_popup(self, painter, w, h):
        """Draw the enhanced POI detail popup card with photo and Wikipedia description."""
        poi = self._poi_popup
        if poi is None:
            return
        alpha = max(0, min(1.0, self._poi_popup_fade))
        if alpha <= 0:
            return

        # --- Gather all available data ---
        geo = self._poi_popup_geocode
        detail = self._poi_get_current_detail()
        has_address = geo and geo.get("address")
        has_phone = geo and geo.get("phone")
        has_city = geo and geo.get("city")
        has_summary = detail and detail.get("summary")
        has_description = detail and detail.get("description")
        has_photo = self._poi_photo_pixmap is not None and not self._poi_photo_pixmap.isNull()

        # --- Calculate dynamic popup height ---
        pw = 300
        ph = 60  # header (name + type badge + distance)

        # Photo section
        photo_h = 0
        if has_photo:
            photo_h = self._poi_photo_pixmap.height() + 8
            ph += photo_h

        # Wikipedia short description line
        if has_description:
            ph += 18

        # Address block
        addr_lines = 0
        if has_address: addr_lines += 1
        if has_city: addr_lines += 1
        if has_phone: addr_lines += 1
        ph += addr_lines * 18
        if addr_lines > 0:
            ph += 6  # spacing before address block

        # Coordinates
        ph += 18

        # Wikipedia summary text (word-wrapped)
        summary_lines = 0
        summary_wrapped = []
        if has_summary:
            # Wrap text to fit popup width (approx 45 chars per line at font size 8)
            text = detail["summary"]
            chars_per_line = 48
            words = text.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > chars_per_line:
                    summary_wrapped.append(line)
                    line = word
                else:
                    line = f"{line} {word}" if line else word
            if line:
                summary_wrapped.append(line)
            summary_lines = min(len(summary_wrapped), 8)  # max 8 lines
            summary_wrapped = summary_wrapped[:8]
            ph += 8 + summary_lines * 15  # spacing + line height

        # Loading indicator space
        if not geo and not detail:
            ph += 20

        # Bottom padding + hint
        ph += 22

        # --- Position: always top-left corner for maximum visibility ---
        px = 12
        py = 12

        a_int = int(alpha * 255)
        cr, cg, cb, _ = poi["color"]
        accent = QColor(int(cr * 255), int(cg * 255), int(cb * 255), a_int)

        # --- Shadow ---
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, int(alpha * 80)))
        painter.drawRoundedRect(px + 4, py + 4, pw, ph, 14, 14)

        # --- Background ---
        painter.setBrush(QColor(12, 15, 24, int(alpha * 245)))
        painter.drawRoundedRect(px, py, pw, ph, 14, 14)

        # --- Accent bar (left edge) ---
        painter.setBrush(accent)
        painter.drawRoundedRect(px, py, 5, ph, 3, 3)

        # --- Accent glow at top ---
        glow = QLinearGradient(px, py, px, py + 40)
        glow.setColorAt(0, QColor(int(cr * 255), int(cg * 255), int(cb * 255), int(alpha * 50)))
        glow.setColorAt(1, QColor(int(cr * 255), int(cg * 255), int(cb * 255), 0))
        painter.setBrush(glow)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(px, py, pw, 40, 14, 14)

        y_cur = py + 10

        # --- Category icon dot ---
        painter.setBrush(accent)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(px + 20, y_cur + 10), 9, 9)
        painter.setPen(QColor(255, 255, 255, a_int))
        painter.setFont(QFont("sans-serif", 10))
        painter.drawText(px + 11, y_cur + 1, 18, 18, Qt.AlignCenter, poi.get("icon", "\U0001F4CD"))

        # --- Name ---
        name = poi.get("name", "Unknown")
        painter.setPen(QColor(240, 245, 255, a_int))
        painter.setFont(QFont("sans-serif", 12, QFont.Bold))
        # Truncate long names
        if len(name) > 30:
            name = name[:29] + "\u2026"
        painter.drawText(px + 36, y_cur, pw - 46, 22, Qt.AlignLeft | Qt.AlignVCenter, name)
        y_cur += 24

        # --- Type badge + distance on same row ---
        display_type = poi.get("display_type", poi.get("type", ""))
        if display_type:
            badge_w = min(len(display_type) * 7 + 14, pw // 2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(int(cr * 80), int(cg * 80), int(cb * 80), int(alpha * 200)))
            painter.drawRoundedRect(px + 14, y_cur, badge_w, 18, 9, 9)
            painter.setPen(QColor(int(cr * 255), int(cg * 255), int(cb * 255), a_int))
            painter.setFont(QFont("sans-serif", 8, QFont.Bold))
            painter.drawText(px + 14, y_cur, badge_w, 18, Qt.AlignCenter, display_type)

        # Distance (right-aligned)
        cam_wx = self._car_wx if (self._immersive and self._car_mode) else (self._fp_cx if self._immersive else self._cx)
        cam_wy = self._car_wy if (self._immersive and self._car_mode) else (self._fp_cy if self._immersive else self._cy)
        z = self._zoom
        lat_c = _wy_to_lat(cam_wy, z)
        mpp_c = (40_075_000 * math.cos(math.radians(lat_c))) / _world_size(z)
        dx_d = poi["wx"] - cam_wx; dy_d = poi["wy"] - cam_wy
        dist_m = math.hypot(dx_d, dy_d) * mpp_c
        dist_str = f"{dist_m:.0f}m" if dist_m < 1000 else f"{dist_m / 1000:.1f}km"
        painter.setPen(QColor(120, 155, 200, a_int))
        painter.setFont(QFont("monospace", 8))
        painter.drawText(px + pw - 75, y_cur, 65, 18, Qt.AlignRight | Qt.AlignVCenter, dist_str)
        y_cur += 24

        # --- Photo ---
        if has_photo:
            pix = self._poi_photo_pixmap
            img_x = px + (pw - pix.width()) // 2
            # Draw photo with rounded clip
            painter.setOpacity(alpha)
            painter.save()
            clip_path = QPainterPath()
            clip_path.addRoundedRect(float(img_x), float(y_cur), float(pix.width()), float(pix.height()), 8, 8)
            painter.setClipPath(clip_path)
            painter.drawPixmap(img_x, y_cur, pix)
            painter.restore()
            painter.setOpacity(1.0)
            y_cur += pix.height() + 8

        # --- Wikipedia short description ---
        if has_description:
            desc = detail["description"]
            if len(desc) > 55:
                desc = desc[:54] + "\u2026"
            painter.setPen(QColor(160, 175, 200, a_int))
            painter.setFont(QFont("sans-serif", 8))
            painter.drawText(px + 14, y_cur, pw - 28, 16,
                            Qt.AlignLeft | Qt.AlignVCenter, desc)
            y_cur += 18

        # --- Address block ---
        if geo:
            if has_address:
                addr_text = geo["address"]
                if len(addr_text) > 45:
                    addr_text = addr_text[:44] + "\u2026"
                painter.setPen(QColor(180, 190, 210, a_int))
                painter.setFont(QFont("sans-serif", 8))
                painter.drawText(px + 14, y_cur, pw - 28, 16,
                                Qt.AlignLeft | Qt.AlignVCenter, f"\U0001F4CD {addr_text}")
                y_cur += 18
            if has_city:
                city_text = geo["city"]
                if geo.get("postcode"):
                    city_text += f" ({geo['postcode']})"
                if geo.get("country"):
                    city_text += f", {geo['country']}"
                painter.setPen(QColor(140, 155, 180, a_int))
                painter.setFont(QFont("sans-serif", 8))
                painter.drawText(px + 14, y_cur, pw - 28, 16,
                                Qt.AlignLeft | Qt.AlignVCenter, f"\U0001F30D {city_text}")
                y_cur += 18
            if has_phone:
                painter.setPen(QColor(100, 200, 160, a_int))
                painter.setFont(QFont("sans-serif", 8))
                painter.drawText(px + 14, y_cur, pw - 28, 16,
                                Qt.AlignLeft | Qt.AlignVCenter, f"\U0001F4DE {geo['phone']}")
                y_cur += 18

        # --- Coordinates ---
        coord_str = f"{poi['lat']:.5f}, {poi['lon']:.5f}"
        painter.setPen(QColor(80, 100, 130, a_int))
        painter.setFont(QFont("monospace", 7))
        painter.drawText(px + 14, y_cur, pw - 28, 16, Qt.AlignLeft | Qt.AlignVCenter, coord_str)
        y_cur += 18

        # --- Wikipedia summary text ---
        if summary_wrapped:
            # Separator line
            painter.setPen(QPen(QColor(60, 70, 90, int(alpha * 100)), 1))
            painter.drawLine(px + 14, y_cur, px + pw - 14, y_cur)
            y_cur += 6
            painter.setPen(QColor(170, 180, 200, a_int))
            painter.setFont(QFont("sans-serif", 8))
            for line in summary_wrapped:
                painter.drawText(px + 14, y_cur, pw - 28, 14,
                                Qt.AlignLeft | Qt.AlignVCenter, line)
                y_cur += 15

        # --- Loading indicator ---
        if not geo and not detail:
            dots = "." * ((self._frame // 10) % 4 + 1)
            painter.setPen(QColor(100, 120, 150, a_int))
            painter.setFont(QFont("monospace", 8))
            painter.drawText(px + 14, y_cur, pw - 28, 16,
                            Qt.AlignLeft | Qt.AlignVCenter, f"fetching details{dots}")
        elif not has_photo and detail and detail.get("photo_url") and not self._poi_photo_pixmap:
            # Photo is loading
            dots = "." * ((self._frame // 8) % 4 + 1)
            painter.setPen(QColor(90, 110, 140, a_int))
            painter.setFont(QFont("monospace", 7))
            painter.drawText(px + 14, y_cur, pw - 28, 14,
                            Qt.AlignLeft | Qt.AlignVCenter, f"loading photo{dots}")

        # --- Bottom hint ---
        if not self._immersive:
            painter.setPen(QColor(60, 75, 100, int(alpha * 130)))
            painter.setFont(QFont("monospace", 7))
            painter.drawText(px + 8, py + ph - 16, pw - 16, 14,
                            Qt.AlignCenter, "click elsewhere to dismiss")

        # --- Action buttons bar ---
        btn_y = py + ph + 5
        btn_h = 24

        # "Open Map" button
        map_bw = 80
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(20, 55, 110, int(alpha * 220)))
        painter.drawRoundedRect(px, btn_y, map_bw, btn_h, 8, 8)
        painter.setPen(QColor(100, 180, 255, a_int))
        painter.setFont(QFont("sans-serif", 8, QFont.Bold))
        key_map = "E" if self._immersive else "dbl-click"
        painter.drawText(px, btn_y, map_bw, btn_h, Qt.AlignCenter, f"\U0001F5FA ({key_map})")

        # "Search Web" button
        web_bw = 80
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(75, 25, 95, int(alpha * 220)))
        painter.drawRoundedRect(px + map_bw + 4, btn_y, web_bw, btn_h, 8, 8)
        painter.setPen(QColor(200, 140, 255, a_int))
        painter.setFont(QFont("sans-serif", 8, QFont.Bold))
        key_web = "X" if self._immersive else "shift-click"
        painter.drawText(px + map_bw + 4, btn_y, web_bw, btn_h, Qt.AlignCenter, f"\U0001F310 ({key_web})")

        # "Wikipedia" button (only if we have wiki data)
        wiki_bw = 70
        if detail and detail.get("wiki_url"):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(30, 80, 60, int(alpha * 220)))
            painter.drawRoundedRect(px + map_bw + web_bw + 8, btn_y, wiki_bw, btn_h, 8, 8)
            painter.setPen(QColor(120, 220, 170, a_int))
            painter.setFont(QFont("sans-serif", 8, QFont.Bold))
            key_wiki = "J" if self._immersive else "ctrl-click"
            painter.drawText(px + map_bw + web_bw + 8, btn_y, wiki_bw, btn_h,
                            Qt.AlignCenter, f"\U0001F4D6 ({key_wiki})")

        # Dismiss hint
        painter.setPen(QColor(60, 75, 100, int(alpha * 110)))
        painter.setFont(QFont("monospace", 7))
        dismiss_hint = "ESC dismiss" if self._immersive else "click dismiss"
        hint_x = px + map_bw + web_bw + (wiki_bw + 12 if detail and detail.get("wiki_url") else 8)
        painter.drawText(hint_x, btn_y, pw - (hint_x - px), btn_h,
                        Qt.AlignLeft | Qt.AlignVCenter, dismiss_hint)

    # -----------------------------------------------------------------
    #  Paint — v6: GPU-direct geometry rendering
    # -----------------------------------------------------------------

    def paintEvent(self, event):
        self._ensure_gl()
        w, h = self.width(), self.height()
        self._resize_fbo(w, h)

        # --- IMMERSIVE MODE RENDER PATH ---
        if self._immersive:
            self._paint_immersive(w, h)
            return

        # Show "click to enter" overlay when entering
        z_int, tile_px, n, col_min, col_max, row_min, row_max = self._visible_tile_range()
        half_w, half_h = w / 2.0, h / 2.0

        # --- Build MVP matrices ---
        bearing_rad = math.radians(self._bearing)
        pitch_rad = math.radians(self._pitch)
        cos_b = math.cos(bearing_rad); sin_b = math.sin(bearing_rad)

        R = np.eye(4, dtype='f4')
        R[0,0] = cos_b; R[0,1] = sin_b; R[1,0] = -sin_b; R[1,1] = cos_b

        T = np.eye(4, dtype='f4')
        T[0,3] = -float(self._cx); T[1,3] = -float(self._cy)

        S = np.eye(4, dtype='f4')
        S[0,0] = 2.0 / w; S[1,1] = -2.0 / h

        P = np.eye(4, dtype='f4')
        if self._pitch > 0.5:
            cos_p = math.cos(pitch_rad)
            fov_f = 1.0 - 0.6 * math.sin(pitch_rad)
            P[1,1] = cos_p * fov_f + (1.0 - cos_p) * 0.6
            P[1,3] = 0.35 * math.sin(pitch_rad)

        VP = P @ S @ R @ T
        VP_gl = np.ascontiguousarray(VP.T).astype('f4')

        # --- Determine visible tiles ---
        tiles_to_draw = []   # (gkey, off_x, off_y, tile_scale)
        tiles_needed = []
        all_label_data = []
        screen_cx_tile = self._cx / tile_px
        screen_cy_tile = self._cy / tile_px

        # Cancel stale requests on zoom change
        if z_int != self._last_tile_z:
            self._cancel_stale_requests(z_int)
            self._last_tile_z = z_int

        for ty in range(row_min, row_max):
            if ty < 0 or ty >= n: continue
            for tx in range(col_min, col_max):
                tx_w = tx % n
                if tx_w < 0: tx_w += n
                gkey = self._geo_cache_key(z_int, tx_w, ty)
                off_x = float(tx * tile_px)
                off_y = float(ty * tile_px)

                geo = self._geo_cache.get(gkey)
                if geo is not None:
                    tiles_to_draw.append((gkey, off_x, off_y, tile_px, False))
                    labels = self._label_cache.get(gkey, [])
                    if labels:
                        all_label_data.append((off_x, off_y, tile_px, labels))
                else:
                    # Parent zoom fallback: find a cached parent and mark
                    # which region of it we need (for scissored drawing)
                    drawn = False
                    for dz in range(1, 5):
                        fz = z_int - dz
                        if fz < 0: break
                        fx = tx_w >> dz; fy = ty >> dz
                        fgkey = self._geo_cache_key(fz, fx, fy)
                        fgeo = self._geo_cache.get(fgkey)
                        if fgeo:
                            parent_tile_px = TILE_SIZE * (2.0 ** (self._zoom - fz))
                            parent_off_x = float(fx * parent_tile_px)
                            parent_off_y = float(fy * parent_tile_px)
                            # Store with scissor rect info for this child cell
                            tiles_to_draw.append((
                                fgkey, parent_off_x, parent_off_y, parent_tile_px,
                                True,  # is_parent_fallback
                                off_x, off_y, tile_px  # scissor region
                            ))
                            drawn = True
                            break
                    tiles_needed.append((z_int, tx_w, ty))

        # --- GL rendering ---
        if self._gl_dirty:
            self._fbo.use()
            self._ctx.viewport = (0, 0, w, h)
            bg = self._current_style.bg
            bg_alpha = 1.0 if self._tiles_visible else 0.0
            self._ctx.clear(bg.red()/255.0, bg.green()/255.0, bg.blue()/255.0, bg_alpha)
            self._ctx.disable(moderngl.DEPTH_TEST)

            # v6: Draw tiles via geometry shader (direct VBO rendering)
            # Skip tile geometry entirely when tiles are hidden
            if self._tiles_visible:
                self._prog_geo['view_proj'].write(VP_gl.tobytes())

            # Separate exact tiles from parent fallbacks
            exact_tiles = []
            parent_fallbacks = []
            for item in tiles_to_draw:
                if len(item) > 4 and item[4]:
                    parent_fallbacks.append(item)
                else:
                    exact_tiles.append(item)

            # Pass 1: Draw parent fallbacks first (they go behind exact tiles)
            # Use GL scissor to clip parent geometry to the child cell it fills
            if self._tiles_visible:
              for item in parent_fallbacks:
                gkey, off_x, off_y, t_scale = item[0], item[1], item[2], item[3]
                child_ox, child_oy, child_sz = item[5], item[6], item[7]
                geo = self._geo_cache.get(gkey)
                if geo is None:
                    continue

                # Compute screen-space scissor rect for the child tile cell
                # Transform the 4 corners of the child cell through VP
                corners_world = np.array([
                    [child_ox, child_oy],
                    [child_ox + child_sz, child_oy],
                    [child_ox, child_oy + child_sz],
                    [child_ox + child_sz, child_oy + child_sz],
                ], dtype='f4')

                # Apply VP to get clip coords, then to screen
                scissor_ok = True
                sx_min, sx_max = w, 0
                sy_min, sy_max = h, 0
                for cx, cy in corners_world:
                    clip = VP @ np.array([cx, cy, 0.0, 1.0], dtype='f4')
                    if abs(clip[3]) < 1e-6:
                        scissor_ok = False; break
                    ndc_x = clip[0] / clip[3]
                    ndc_y = clip[1] / clip[3]
                    scr_x = (ndc_x * 0.5 + 0.5) * w
                    scr_y = (1.0 - (ndc_y * 0.5 + 0.5)) * h
                    sx_min = min(sx_min, scr_x)
                    sx_max = max(sx_max, scr_x)
                    sy_min = min(sy_min, scr_y)
                    sy_max = max(sy_max, scr_y)

                if scissor_ok:
                    # Clamp to viewport
                    ix = max(0, int(math.floor(sx_min)))
                    iy = max(0, int(math.floor(sy_min)))
                    ix2 = min(w, int(math.ceil(sx_max)))
                    iy2 = min(h, int(math.ceil(sy_max)))
                    sw = ix2 - ix; sh = iy2 - iy
                    if sw > 0 and sh > 0:
                        # GL scissor Y is from bottom; setting ctx.scissor enables scissor test
                        self._ctx.scissor = (ix, h - iy2, sw, sh)

                self._prog_geo['tile_offset'].write(
                    np.array([off_x, off_y], dtype='f4').tobytes())
                self._prog_geo['tile_scale'].value = float(t_scale)

                if geo['fill_vao'] and geo['fill_count'] > 0:
                    geo['fill_vao'].render(moderngl.TRIANGLES)
                if geo['line_vao'] and geo['line_count'] > 0:
                    geo['line_vao'].render(moderngl.TRIANGLES)

                if scissor_ok:
                    # Restore full viewport (disables scissor)
                    self._ctx.scissor = None

            # Pass 2: Draw exact tiles (fills then lines for proper layering)
            if self._tiles_visible:
              for item in exact_tiles:
                gkey, off_x, off_y, t_scale = item[0], item[1], item[2], item[3]
                geo = self._geo_cache.get(gkey)
                if geo is None:
                    continue

                self._prog_geo['tile_offset'].write(
                    np.array([off_x, off_y], dtype='f4').tobytes())
                self._prog_geo['tile_scale'].value = float(t_scale)

                if geo['fill_vao'] and geo['fill_count'] > 0:
                    geo['fill_vao'].render(moderngl.TRIANGLES)

              for item in exact_tiles:
                gkey, off_x, off_y, t_scale = item[0], item[1], item[2], item[3]
                geo = self._geo_cache.get(gkey)
                if geo is None:
                    continue

                self._prog_geo['tile_offset'].write(
                    np.array([off_x, off_y], dtype='f4').tobytes())
                self._prog_geo['tile_scale'].value = float(t_scale)

                if geo['line_vao'] and geo['line_count'] > 0:
                    geo['line_vao'].render(moderngl.TRIANGLES)

            # Draw 3D buildings (with shadows)
            bld_min_zoom = int(self._get_panel_val("bld_min_zoom", 16))
            if self._buildings_3d and z_int >= bld_min_zoom:
                self._ctx.enable(moderngl.DEPTH_TEST)
                self._ctx.depth_func = '<='
                bld_max = int(self._get_panel_val("bld_max", 500))
                visible_bld_keys = []
                for ty in range(row_min, row_max):
                    if ty < 0 or ty >= n: continue
                    for tx in range(col_min, col_max):
                        tx_w = tx % n
                        if tx_w < 0: tx_w += n
                        tkey = (z_int, tx_w, ty)
                        if tkey in self._building_cache and self._building_cache[tkey]:
                            visible_bld_keys.append(tkey)

                if visible_bld_keys:
                    default_h = self._get_panel_val("bld_height", 4.0)
                    h_exag = self._get_panel_val("bld_scale", 0.08 * 100.0) / 100.0
                    pitch_factor = max(1.0, 1.0 + math.sin(pitch_rad) * 3.0)
                    z_scale = max(0.5, (2 ** (self._zoom - 13)) * 0.8)

                    bld_key = (z_int, tuple(visible_bld_keys), int(default_h*10),
                               int(h_exag*100), int(pitch_factor*10), bld_max)
                    if bld_key != self._bld_gpu_key:
                        self._rebuild_building_vbo(z_int, tile_px, visible_bld_keys,
                                                    default_h, h_exag, z_scale, pitch_factor, bld_max)
                        self._bld_gpu_key = bld_key

                    # Shadows
                    if self._shadows_enabled and self._pitch > 2.0:
                        self._sun_azimuth, self._sun_elevation = _sun_position(
                            self.centre_lat, self.centre_lon, self._sun_time)

                        if self._sun_elevation > 0:
                            shadow_key = (bld_key, int(self._sun_azimuth), int(self._sun_elevation))
                            if shadow_key != self._shadow_gpu_key:
                                self._rebuild_shadow_vbo(
                                    visible_bld_keys, tile_px,
                                    default_h, h_exag, z_scale, pitch_factor, bld_max)
                                self._shadow_gpu_key = shadow_key

                            if self._shadow_vao and self._shadow_vert_count > 0:
                                self._ctx.disable(moderngl.DEPTH_TEST)
                                self._prog_shadow['mvp'].write(VP_gl.tobytes())
                                sun_az_rad = math.radians(self._sun_azimuth)
                                shadow_scale = 0.012 / max(math.tan(math.radians(
                                    max(self._sun_elevation, 5.0))), 0.1)
                                shadow_scale = min(shadow_scale, 0.06)
                                sx_off = math.sin(sun_az_rad) * shadow_scale
                                sy_off = -math.cos(sun_az_rad) * shadow_scale
                                self._prog_shadow['shadow_offset'].write(
                                    np.array([sx_off, sy_off], dtype='f4').tobytes())
                                self._shadow_vao.render(moderngl.TRIANGLES)
                                self._ctx.enable(moderngl.DEPTH_TEST)

                    if self._bld_vao and self._bld_vert_count > 0:
                        self._prog_bldg['mvp'].write(VP_gl.tobytes())
                        self._prog_bldg['height_scale'].value = 2.0 / h

                        if self._shadows_enabled and self._sun_elevation > 0:
                            sun_az_rad = math.radians(self._sun_azimuth)
                            sun_el_rad = math.radians(self._sun_elevation)
                            light = np.array([
                                -math.sin(sun_az_rad) * math.cos(sun_el_rad),
                                math.cos(sun_az_rad) * math.cos(sun_el_rad),
                                math.sin(sun_el_rad)
                            ], dtype='f4')
                            light /= np.linalg.norm(light)
                        else:
                            light = np.array([-0.5, -0.7, 0.5], dtype='f4')
                            light /= np.linalg.norm(light)
                        self._prog_bldg['light_dir'].write(light.tobytes())

                        style_bg = self._current_style.bg
                        bg_lum = (style_bg.red() + style_bg.green() + style_bg.blue()) / (3.0 * 255.0)
                        is_dark = bg_lum < 0.3
                        self._prog_bldg['ambient_strength'].value = 0.55 if is_dark else 0.45
                        self._prog_bldg['ao_strength'].value = 0.35 if is_dark else 0.25
                        self._prog_bldg['top_highlight'].value = 0.08 if is_dark else 0.04

                        self._bld_vao.render(moderngl.TRIANGLES)

                self._ctx.disable(moderngl.DEPTH_TEST)

            # Heatmap (only when tiles visible)
            if self._heatmap_enabled and self._tiles_visible:
                if not self._heatmap_generated:
                    self._generate_heatmap_data()
                elif self._heatmap_points_latlon:
                    # Auto-regenerate if view has panned far or zoom changed significantly
                    cur_lat = _wy_to_lat(self._cy, self._zoom)
                    cur_lon = _wx_to_lon(self._cx, self._zoom)
                    view_span = 360.0 / (2.0 ** self._zoom) * 2.0
                    dlat = abs(cur_lat - self._heatmap_gen_lat)
                    dlon = abs(cur_lon - self._heatmap_gen_lon)
                    dzoom = abs(self._zoom - self._heatmap_gen_zoom)
                    if dlat > view_span or dlon > view_span or dzoom > 1.5:
                        self._heatmap_generated = False
                        self._generate_heatmap_data()
                if self._heatmap_dirty:
                    self._rebuild_heatmap_vbo()
                if self._heatmap_vao and self._heatmap_vert_count > 0:
                    self._ctx.enable(moderngl.BLEND)
                    self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
                    self._prog_heat['view_proj'].write(VP_gl.tobytes())
                    # Read panel values (with defaults when panel is hidden)
                    heat_radius_base = self._get_panel_val("heat_radius", 50.0)
                    heat_intensity = self._get_panel_val("heat_intensity", 60.0) / 100.0
                    heat_filter = self._get_panel_val("heat_filter", 0.0) / 100.0
                    radius_world = max(15.0, heat_radius_base * (2.0 ** (self._zoom - 14)))
                    self._prog_heat['point_radius'].value = float(radius_world)
                    self._prog_heat['intensity'].value = float(heat_intensity)
                    self._prog_heat['min_heat'].value = float(heat_filter)
                    self._heatmap_vao.render(moderngl.TRIANGLES)
                    self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

            # POI 2D markers (only when tiles visible and not in immersive)
            if self._poi_enabled and self._tiles_visible and self._prog_poi_2d:
                z_int_2d = max(0, min(20, int(math.floor(self._zoom + 0.5))))
                tp_2d = TILE_SIZE * (2.0 ** (self._zoom - z_int_2d))
                n_2d = 2 ** z_int_2d

                # v7: Only re-extract POIs when the visible tile key set changes
                with self._mvt_lock:
                    vis_k = []
                    for ty_2d in range(max(0, row_min), min(n_2d, row_max)):
                        for tx_2d in range(col_min, col_max):
                            tx_w_2d = tx_2d % n_2d
                            if tx_w_2d < 0: tx_w_2d += n_2d
                            tk = (z_int_2d, tx_w_2d, ty_2d)
                            if tk in self._mvt_cache:
                                vis_k.append(tk)
                    poi_tile_key = (z_int_2d, tuple(sorted(vis_k)))
                    need_poi_extract = (poi_tile_key != getattr(self, '_poi_tile_key_v7', None))
                    if need_poi_extract:
                        snap = {k: self._mvt_cache[k] for k in vis_k if k in self._mvt_cache}
                if need_poi_extract:
                    self._poi_cache = _extract_pois_from_tiles(snap, vis_k, tp_2d, self._zoom)
                    self._poi_tile_key_v7 = poi_tile_key
                    self._poi_2d_dirty = True

                # v7: Only rebuild VBO when POI data changed or zoom changed meaningfully
                poi_vbo_key = (len(self._poi_cache), int(self._zoom * 4))
                if self._poi_2d_dirty or poi_vbo_key != getattr(self, '_poi_vbo_key_v7', None):
                    poi_2d_verts = _build_poi_markers_2d(self._poi_cache, self._zoom)
                    for attr_2d in ('_poi_2d_vbo', '_poi_2d_vao'):
                        o_2d = getattr(self, attr_2d)
                        if o_2d:
                            try: o_2d.release()
                            except: pass
                    if len(poi_2d_verts) > 0:
                        self._poi_2d_vbo = self._ctx.buffer(
                            np.ascontiguousarray(poi_2d_verts).tobytes())
                        self._poi_2d_vao = self._ctx.vertex_array(
                            self._prog_poi_2d,
                            [(self._poi_2d_vbo, '2f 2f 4f 1f',
                              'in_position', 'in_quad', 'in_color', 'in_icon_id')])
                        self._poi_2d_count = len(poi_2d_verts)
                    else:
                        self._poi_2d_vbo = None; self._poi_2d_vao = None
                        self._poi_2d_count = 0
                    self._poi_2d_dirty = False
                    self._poi_vbo_key_v7 = poi_vbo_key

                if self._poi_2d_vao and self._poi_2d_count > 0:
                    self._ctx.enable(moderngl.BLEND)
                    self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                    self._prog_poi_2d['view_proj'].write(VP_gl.tobytes())
                    poi_rad = max(5.0, 8.0 * (2.0 ** (self._zoom - 15)))
                    self._prog_poi_2d['point_radius'].value = float(poi_rad)
                    self._prog_poi_2d['u_time'].value = float(time.time() % 100.0)
                    self._poi_2d_vao.render(moderngl.TRIANGLES)

            # Route
            if self._route_world and len(self._route_world) >= 2:
                if self._route_dirty:
                    self._rebuild_route_vbo()
                if self._route_vao and self._route_vert_count > 0:
                    self._prog_route['view_proj'].write(VP_gl.tobytes())
                    rc = self._route_color
                    self._prog_route['route_color'].write(
                        np.array(rc, dtype='f4').tobytes())
                    self._prog_route['time'].value = float(time.time() % 100.0)
                    self._prog_route['dash_scale'].value = max(5.0, 40.0 * (2.0 ** (self._zoom - 15)))
                    self._route_vao.render(moderngl.TRIANGLES)

            # --- Optimized FBO readback — only when dirty ---
            raw = self._fbo.color_attachments[0].read()
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
            flipped = np.ascontiguousarray(arr[::-1])
            self._cached_frame = QImage(flipped.data, w, h, w*4, QImage.Format_RGBA8888).copy()
            self._gl_dirty = False

        # --- Blit cached frame + overlays ---
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self._cached_frame and not self._cached_frame.isNull():
            painter.drawImage(0, 0, self._cached_frame)

        # Labels (only when tiles visible and labels enabled)
        if self._tiles_visible and self._labels_visible:
            self._draw_labels_overlay(painter, all_label_data, w, h, tile_px)

        # Route overlay
        self._draw_route_overlay(painter, w, h)

        # POI name labels overlay (2D map)
        if self._poi_enabled and self._tiles_visible and self._poi_cache:
            self._draw_poi_labels_2d(painter, w, h)

        # Heatmap legend
        if self._heatmap_enabled and self._tiles_visible:
            self._draw_heatmap_legend(painter, w, h)

        # Isochrone overlay
        if self._iso_enabled:
            self._draw_isochrone_overlay(painter, w, h)

        self._paint_hud(painter, w, h, len(tiles_needed))

        # POI detail popup (2D map)
        if self._poi_popup and not self._immersive:
            self._draw_poi_popup(painter, w, h)

        # Immersive mode entering overlay
        if self._immersive_entering:
            msg = "IMMERSIVE MODE — Click anywhere to enter first-person view (ESC to cancel)"
            pw = len(msg) * 7 + 20
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 60, 160, 210))
            painter.drawRoundedRect(w // 2 - pw // 2, h // 2 - 16, pw, 32, 16, 16)
            painter.setPen(QColor(255, 255, 255)); painter.setFont(QFont("monospace", 10, QFont.Bold))
            painter.drawText(w // 2 - pw // 2, h // 2 - 16, pw, 32, Qt.AlignCenter, msg)
            # Crosshair
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1))
            mx, my = self.mapFromGlobal(QCursor.pos()).x(), self.mapFromGlobal(QCursor.pos()).y()
            painter.drawLine(mx - 15, my, mx + 15, my)
            painter.drawLine(mx, my - 15, mx, my + 15)

        # Isochrone placement overlay
        if self._iso_placing and not self._immersive:
            profile_icons = {"driving": "\U0001F697", "walking": "\U0001F6B6", "cycling": "\U0001F6B2"}
            self._iso_read_panel_params()
            icon = profile_icons.get(self._iso_profile, "\U0001F697")
            mins_str = "/".join(str(m) for m in self._iso_minutes)
            msg = f"ISOCHRONE — Click to place ({icon} {self._iso_profile}, {mins_str} min)  |  Y cancel"
            pw = len(msg) * 7 + 20
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(60, 10, 120, 210))
            painter.drawRoundedRect(w // 2 - pw // 2, h // 2 - 16, pw, 32, 16, 16)
            painter.setPen(QColor(255, 255, 255)); painter.setFont(QFont("monospace", 10, QFont.Bold))
            painter.drawText(w // 2 - pw // 2, h // 2 - 16, pw, 32, Qt.AlignCenter, msg)
            # Crosshair
            painter.setPen(QPen(QColor(180, 120, 255, 200), 1.5))
            mx, my = self.mapFromGlobal(QCursor.pos()).x(), self.mapFromGlobal(QCursor.pos()).y()
            painter.drawLine(mx - 18, my, mx + 18, my)
            painter.drawLine(mx, my - 18, mx, my + 18)
            painter.setPen(QPen(QColor(180, 120, 255, 100), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(mx, my), 25, 25)

        # Search bar overlay (on top of everything)
        if self._search_active:
            self._draw_search_bar(painter, w, h)

        painter.end()

        # --- Request missing tiles (prioritised by distance from center) ---
        # v9: During flyTo, suppress tile requests for intermediate zoom levels.
        # Existing geometry stretches continuously via fractional tile_px.
        suppress = self._fly.active and self._fly.suppress_tiles
        if tiles_needed and not suppress:
            tiles_needed.sort(
                key=lambda t: (t[1] - screen_cx_tile)**2 + (t[2] - screen_cy_tile)**2)
            is_zooming_fast = self._zoom_velocity > 0.1 or self._scroll_zoom_active
            max_requests = 4 if is_zooming_fast else len(tiles_needed)
            for z, x, y in tiles_needed[:max_requests]:
                self._request_tile(z, x, y)

    # -----------------------------------------------------------------
    #  Immersive (First-Person) Rendering
    # -----------------------------------------------------------------

    def _paint_immersive(self, w, h):
        """Render the scene in first-person perspective."""
        z = self._zoom
        z_int = max(0, min(20, int(math.floor(z + 0.5))))
        tile_px = TILE_SIZE * (2.0 ** (z - z_int))
        n = 2 ** z_int

        # Camera world-pixel position
        cam_wx = self._fp_cx
        cam_wy = self._fp_cy
        lat = _wy_to_lat(cam_wy, z)
        lon = _wx_to_lon(cam_wx, z)
        mpp = (40_075_000 * math.cos(math.radians(lat))) / _world_size(z)

        eye_h = self._fp_eye_height

        # Build view + projection matrices
        # 3D space: (-wx, height, -wy) — right-handed Y-up
        # +Z = north (-wy), -X = east (+wx), +Y = up
        yaw_rad = math.radians(self._fp_yaw)
        pitch_rad = math.radians(self._fp_pitch_angle)

        if self._car_mode:
            # Use car position as the camera-relative origin for precision.
            # All scene geometry is offset relative to the car, not the camera.
            # This prevents float32 quantization jitter while driving.
            origin_wx = float(self._car_wx)  # float64
            origin_wy = float(self._car_wy)  # float64
            car_h = 1.0 / max(mpp, 0.001)

            # Terrain elevation at car position (world-pixels)
            _terr_elev = _terrain_elev_wp(self._terrain_cache, origin_wx, origin_wy, z, mpp) if self._terrain_enabled else 0.0

            # Camera eye relative to the car origin in cam_pos_gl-subtracted space.
            # cam_pos_gl.y = _terr_elev, so ground is at rel.y ≈ 0 in shader space.
            # The view matrix operates in this same space, so eye_h is height above ground.
            eye_rel_x = float(-cam_wx) - float(-origin_wx)
            eye_rel_y = float(eye_h)  # above ground (terrain already in cam_pos_gl)
            eye_rel_z = float(-cam_wy) - float(-origin_wy)

            # Look target: car roof above ground (ground = Y=0 in rel space)
            target_rel = np.array([0.0, car_h, 0.0], dtype='f4')
            eye_rel = np.array([eye_rel_x, eye_rel_y, eye_rel_z], dtype='f4')

            # Look direction from eye to car (small relative coords, no precision loss)
            look_dir = target_rel - eye_rel
            look_len = np.linalg.norm(look_dir)
            if look_len > 0.001:
                look_dir /= look_len
            else:
                look_dir = np.array([-math.sin(yaw_rad), 0, math.cos(yaw_rad)], dtype='f4')

            # View matrix: rotation + small eye offset from origin
            center_rel = eye_rel + look_dir * np.float32(100.0)
            up = np.array([0, 1, 0], dtype='f4')
            view_mat = _fp_look_at(eye_rel, center_rel, up)  # full look-at with small translation

            # cam_pos_gl = car position in swizzled world space (with terrain for fog)
            cam_pos_gl = np.array([-origin_wx, float(_terr_elev), -origin_wy], dtype='f4')
        else:
            look_dx = -math.sin(yaw_rad) * math.cos(pitch_rad)
            look_dy = math.sin(pitch_rad)
            look_dz = math.cos(yaw_rad) * math.cos(pitch_rad)
            eye_origin = np.array([0, 0, 0], dtype='f4')
            center_dir = np.array([look_dx, look_dy, look_dz], dtype='f4') * 100.0
            up = np.array([0, 1, 0], dtype='f4')
            view_mat = _fp_look_at_rot(eye_origin, center_dir, up)
            _terr_elev_walk = _terrain_elev_wp(self._terrain_cache, cam_wx, cam_wy, z, mpp) if self._terrain_enabled else 0.0
            eye = np.array([-cam_wx, eye_h + _terr_elev_walk, -cam_wy], dtype='f4')
            cam_pos_gl = eye.copy()

        aspect = w / max(h, 1)
        view_dist = 200.0 / max(mpp, 0.001)
        near = 0.3 / max(mpp, 0.001)
        far = view_dist * 2.5
        proj_mat = _fp_perspective(70.0, aspect, near, far)

        view_gl = np.ascontiguousarray(view_mat.T).astype('f4')
        proj_gl = np.ascontiguousarray(proj_mat.T).astype('f4')

        # --- Geometry rebuild: only when camera crosses a tile boundary ---
        cam_tx = int(math.floor(cam_wx / tile_px))
        cam_ty = int(math.floor(cam_wy / tile_px))
        geo_key = (z_int, cam_tx, cam_ty)

        if geo_key != getattr(self, '_fp_geo_key', None):
            self._fp_geo_key = geo_key
            self._request_terrain_around(cam_wx, cam_wy, radius_tiles=2)
            # Small radius: just 1 tile around camera
            radius_tiles = 1
            col_min = cam_tx - radius_tiles; col_max = cam_tx + radius_tiles + 1
            row_min = cam_ty - radius_tiles; row_max = cam_ty + radius_tiles + 1

            visible_bld_keys = []
            visible_mvt_keys = []
            with self._mvt_lock:
                for ty in range(max(0, row_min), min(n, row_max)):
                    for tx in range(col_min, col_max):
                        tx_w = tx % n
                        tkey = (z_int, tx_w, ty)
                        if tkey in self._building_cache and self._building_cache[tkey]:
                            visible_bld_keys.append(tkey)
                        if tkey in self._mvt_cache:
                            visible_mvt_keys.append(tkey)

            # Request missing tiles (non-blocking)
            for ty in range(max(0, row_min), min(n, row_max)):
                for tx in range(col_min, col_max):
                    self._request_tile(z_int, tx % n, ty)

            # Rebuild buildings
            if visible_bld_keys:
                style = self._current_style
                bld_style = None
                for ls in style.layers:
                    if ls.get("building"):
                        bld_style = ls; break
                if bld_style:
                    fc = bld_style["fill"]
                    br, bg_, bb = fc.red()/255.0, fc.green()/255.0, fc.blue()/255.0
                    ba = min(0.95, fc.alpha()/255.0)
                    wall_color = (br*0.85, bg_*0.85, bb*0.88, ba)
                    roof_color = (min(1.0, br*1.25+0.08), min(1.0, bg_*1.25+0.08),
                                  min(1.0, bb*1.20+0.06), ba)
                    # Immersive: use realistic heights — no exaggeration
                    default_h = 10.0  # ~3 stories for buildings without height data
                    h_exag = 1.0      # no exaggeration in first-person
                    verts = _build_fp_buildings(
                        self._building_cache, visible_bld_keys, tile_px,
                        default_h, h_exag, mpp, wall_color, roof_color,
                        terrain_cache=self._terrain_cache, zoom=z)
                    for a in ('_fp_bldg_vbo', '_fp_bldg_vao'):
                        o = getattr(self, a)
                        if o:
                            try: o.release()
                            except: pass
                    if len(verts) > 0:
                        self._fp_bldg_vbo = self._ctx.buffer(np.ascontiguousarray(verts).tobytes())
                        self._fp_bldg_vao = self._ctx.vertex_array(
                            self._fp_prog_bldg,
                            [(self._fp_bldg_vbo, '3f 3f 4f', 'in_pos', 'in_normal', 'in_color')])
                        self._fp_bldg_count = len(verts)
                    else:
                        self._fp_bldg_vbo = None; self._fp_bldg_vao = None; self._fp_bldg_count = 0

            # Rebuild ground (only on tile change)
            ground_radius = view_dist * 1.5
            gv = _build_fp_ground(cam_wx, cam_wy, ground_radius, tile_px, self._current_style,
                                  terrain_cache=self._terrain_cache, zoom=z, mpp=mpp)
            for a in ('_fp_ground_vbo', '_fp_ground_vao'):
                o = getattr(self, a)
                if o:
                    try: o.release()
                    except: pass
            self._fp_ground_vbo = self._ctx.buffer(np.ascontiguousarray(gv).tobytes())
            self._fp_ground_vao = self._ctx.vertex_array(
                self._fp_prog_ground, [(self._fp_ground_vbo, '3f 4f', 'in_pos', 'in_color')])
            self._fp_ground_count = len(gv)

            # Snapshot MVT tiles once for water + roads
            with self._mvt_lock:
                mvt_snap = {k: self._mvt_cache[k] for k in visible_mvt_keys if k in self._mvt_cache}

            # Rebuild water (only on tile change)
            wv = _build_fp_water(mvt_snap, visible_mvt_keys, tile_px, self._current_style)
            for a in ('_fp_water_vbo', '_fp_water_vao'):
                o = getattr(self, a)
                if o:
                    try: o.release()
                    except: pass
            if len(wv) > 0:
                self._fp_water_vbo = self._ctx.buffer(np.ascontiguousarray(wv).tobytes())
                self._fp_water_vao = self._ctx.vertex_array(
                    self._fp_prog_water, [(self._fp_water_vbo, '3f 4f', 'in_pos', 'in_color')])
                self._fp_water_count = len(wv)
            else:
                self._fp_water_vbo = None; self._fp_water_vao = None; self._fp_water_count = 0

            # Rebuild roads (only on tile change)
            rv = _build_fp_roads(mvt_snap, visible_mvt_keys, tile_px, self._current_style, z, mpp,
                                terrain_cache=self._terrain_cache)
            for a in ('_fp_road_vbo', '_fp_road_vao'):
                o = getattr(self, a)
                if o:
                    try: o.release()
                    except: pass
            if len(rv) > 0:
                self._fp_road_vbo = self._ctx.buffer(np.ascontiguousarray(rv).tobytes())
                self._fp_road_vao = self._ctx.vertex_array(
                    self._fp_prog_road_nav,
                    [(self._fp_road_vbo, '3f 4f 2f', 'in_pos', 'in_color', 'in_uv')])
                self._fp_road_count = len(rv)
            else:
                self._fp_road_vbo = None; self._fp_road_vao = None; self._fp_road_count = 0

            # v7: Rebuild trees in green areas (only on tile change)
            if self._trees_enabled:
                tree_verts = _build_fp_trees(mvt_snap, visible_mvt_keys, tile_px, mpp,
                                             self._current_style)
                for a in ('_fp_tree_vbo', '_fp_tree_vao'):
                    o = getattr(self, a)
                    if o:
                        try: o.release()
                        except: pass
                if len(tree_verts) > 0:
                    self._fp_tree_vbo = self._ctx.buffer(np.ascontiguousarray(tree_verts).tobytes())
                    self._fp_tree_vao = self._ctx.vertex_array(
                        self._fp_prog_bldg,  # reuse building shader — same vertex layout
                        [(self._fp_tree_vbo, '3f 3f 4f', 'in_pos', 'in_normal', 'in_color')])
                    self._fp_tree_count = len(tree_verts)
                else:
                    self._fp_tree_vbo = None; self._fp_tree_vao = None; self._fp_tree_count = 0

        # --- GL Rendering (runs every frame, but geometry is cached) ---

        # v7: Rebuild route geometry when route changes (independent of tile crossings)
        if self._route_geometry and len(self._route_geometry) >= 2:
            route_key = (len(self._route_geometry), id(self._route_geometry),
                         int(self._zoom * 4))
            if route_key != self._fp_route_key:
                self._fp_route_key = route_key
                route_verts = _build_fp_route(self._route_geometry, z, mpp)
                for a in ('_fp_route_vbo', '_fp_route_vao'):
                    o = getattr(self, a)
                    if o:
                        try: o.release()
                        except: pass
                if len(route_verts) > 0:
                    self._fp_route_vbo = self._ctx.buffer(
                        np.ascontiguousarray(route_verts).tobytes())
                    self._fp_route_vao = self._ctx.vertex_array(
                        self._fp_prog_road_nav,  # use nav shader for glow+markings
                        [(self._fp_route_vbo, '3f 4f 2f', 'in_pos', 'in_color', 'in_uv')])
                    self._fp_route_count = len(route_verts)
                else:
                    self._fp_route_vbo = None; self._fp_route_vao = None
                    self._fp_route_count = 0
        elif self._fp_route_count > 0:
            # Route was cleared — release GPU resources
            for a in ('_fp_route_vbo', '_fp_route_vao'):
                o = getattr(self, a)
                if o:
                    try: o.release()
                    except: pass
            self._fp_route_vbo = None; self._fp_route_vao = None
            self._fp_route_count = 0; self._fp_route_key = None
        self._fbo.use()
        self._ctx.viewport = (0, 0, w, h)
        bg = self._current_style.bg
        bg_lum = (bg.red() + bg.green() + bg.blue()) / (3.0 * 255.0)
        is_dark = bg_lum < 0.3

        if is_dark:
            fog_color = np.array([0.03, 0.04, 0.08], dtype='f4')
            sky_top = np.array([0.02, 0.03, 0.12], dtype='f4')
            sky_horiz = np.array([0.06, 0.08, 0.15], dtype='f4')
        else:
            fog_color = np.array([0.7, 0.75, 0.85], dtype='f4')
            sky_top = np.array([0.3, 0.5, 0.9], dtype='f4')
            sky_horiz = np.array([0.75, 0.82, 0.92], dtype='f4')

        self._ctx.clear(fog_color[0], fog_color[1], fog_color[2], 1.0 if self._tiles_visible else 0.0)

        fog_near = view_dist * 0.4
        fog_far = view_dist * 1.2

        # Sun direction
        sun_az, sun_el = self._sun_azimuth, self._sun_elevation
        if self._frame % 60 == 0:  # update sun only every ~1s
            sun_az, sun_el = _sun_position(lat, lon, self._sun_time)
            self._sun_azimuth = sun_az; self._sun_elevation = sun_el
        sun_rad_az = math.radians(sun_az); sun_rad_el = math.radians(max(sun_el, 10))
        light_dir = np.array([
            -math.sin(sun_rad_az) * math.cos(sun_rad_el),
            math.sin(sun_rad_el),
            math.cos(sun_rad_az) * math.cos(sun_rad_el)
        ], dtype='f4')
        light_dir /= np.linalg.norm(light_dir)

        # 1. Sky dome (no depth)
        if self._tiles_visible:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._fp_prog_sky['u_view_rot'].write(view_gl.tobytes())
            self._fp_prog_sky['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_sky['u_sky_top'].write(sky_top.tobytes())
            self._fp_prog_sky['u_sky_horizon'].write(sky_horiz.tobytes())
            self._fp_prog_sky['u_sun_dir'].write(light_dir.tobytes())
            if self._fp_sky_vao and self._fp_sky_count > 0:
                self._fp_sky_vao.render(moderngl.TRIANGLES)

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.depth_func = '<='

        # --- Terrain heightmap texture (rebuild when dirty) ---
        terrain_active = 0.0
        terrain_scale = 1.0 / max(mpp, 0.0001)
        terrain_bounds = np.array([0.0, 0.0, 1.0, 1.0], dtype='f4')
        if self._terrain_enabled and self._terrain_cache:
            if self._terrain_tex_dirty or self._terrain_tex is None:
                self._rebuild_terrain_texture(cam_wx, cam_wy, view_dist * 2.0)
            if self._terrain_tex is not None:
                terrain_active = 1.0
                b = self._terrain_tex_bounds
                terrain_bounds = np.array([b[0], b[1], b[2], b[3]], dtype='f4')

        def _set_terrain_uniforms(prog):
            """Set terrain uniforms on a shader program."""
            try:
                if self._terrain_tex and terrain_active > 0.5:
                    self._terrain_tex.use(location=4)
                    prog['u_terrain'].value = 4
                prog['u_terrain_bounds'].write(terrain_bounds.tobytes())
                prog['u_terrain_scale'].value = float(terrain_scale)
                prog['u_terrain_active'].value = float(terrain_active)
            except Exception:
                pass  # uniform not found — shader doesn't have terrain

        # 2. Ground (only when tiles visible)
        if self._tiles_visible and self._fp_ground_vao and self._fp_ground_count > 0:
            self._fp_prog_ground['u_view'].write(view_gl.tobytes())
            self._fp_prog_ground['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_ground['u_cam_pos'].write(cam_pos_gl.tobytes())
            self._fp_prog_ground['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_ground['u_fog_near'].value = float(fog_near)
            self._fp_prog_ground['u_fog_far'].value = float(fog_far)
            _set_terrain_uniforms(self._fp_prog_ground)
            self._fp_ground_vao.render(moderngl.TRIANGLES)

        # 2b. Water (animated surface, rendered after ground with blending)
        if self._tiles_visible and self._fp_water_vao and self._fp_water_count > 0:
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._fp_prog_water['u_view'].write(view_gl.tobytes())
            self._fp_prog_water['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_water['u_cam_pos'].write(cam_pos_gl.tobytes())
            self._fp_prog_water['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_water['u_fog_near'].value = float(fog_near)
            self._fp_prog_water['u_fog_far'].value = float(fog_far)
            self._fp_prog_water['u_time'].value = float(time.time() % 1000.0)
            _set_terrain_uniforms(self._fp_prog_water)
            self._fp_water_vao.render(moderngl.TRIANGLES)

        # 3. Roads (only when tiles visible) — navigation-SDK-style with lane markings
        if self._tiles_visible and self._fp_road_vao and self._fp_road_count > 0:
            self._fp_prog_road_nav['u_view'].write(view_gl.tobytes())
            self._fp_prog_road_nav['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_road_nav['u_cam_pos'].write(cam_pos_gl.tobytes())
            self._fp_prog_road_nav['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_road_nav['u_fog_near'].value = float(fog_near)
            self._fp_prog_road_nav['u_fog_far'].value = float(fog_far)
            self._fp_prog_road_nav['u_road_marking_scale'].value = float(mpp)
            if 'u_time' in self._fp_prog_road_nav:
                self._fp_prog_road_nav['u_time'].value = float(time.time() % 1000.0)
            _set_terrain_uniforms(self._fp_prog_road_nav)
            self._fp_road_vao.render(moderngl.TRIANGLES)

        # 3b. v7: Route path (rendered above roads, uses road shader)
        if self._fp_route_vao and self._fp_route_count > 0:
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._fp_prog_road_nav['u_view'].write(view_gl.tobytes())
            self._fp_prog_road_nav['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_road_nav['u_cam_pos'].write(cam_pos_gl.tobytes())
            self._fp_prog_road_nav['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_road_nav['u_fog_near'].value = float(fog_near)
            self._fp_prog_road_nav['u_fog_far'].value = float(fog_far)
            self._fp_prog_road_nav['u_road_marking_scale'].value = float(mpp)
            if 'u_time' in self._fp_prog_road_nav:
                self._fp_prog_road_nav['u_time'].value = float(time.time() % 1000.0)
            _set_terrain_uniforms(self._fp_prog_road_nav)
            self._fp_route_vao.render(moderngl.TRIANGLES)

        # 4. Buildings
        if self._fp_bldg_vao and self._fp_bldg_count > 0:
            self._fp_prog_bldg['u_view'].write(view_gl.tobytes())
            self._fp_prog_bldg['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_bldg['u_cam_pos'].write(cam_pos_gl.tobytes())
            self._fp_prog_bldg['u_light_dir'].write(light_dir.tobytes())
            self._fp_prog_bldg['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_bldg['u_fog_near'].value = float(fog_near)
            self._fp_prog_bldg['u_fog_far'].value = float(fog_far)
            self._fp_prog_bldg['u_time'].value = float(time.time() % 1000.0)
            _set_terrain_uniforms(self._fp_prog_bldg)
            # Building visual controls from panel
            self._fp_prog_bldg['u_wall_opacity'].value = float(
                self._get_panel_val("bld_opacity", 100.0)) / 100.0
            self._fp_prog_bldg['u_win_glow'].value = float(
                self._get_panel_val("bld_win_glow", 100.0)) / 100.0
            self._fp_prog_bldg['u_ao_strength'].value = float(
                self._get_panel_val("bld_ao", 100.0)) / 100.0
            tint_r = float(self._get_panel_val("bld_tint_r", 100.0)) / 100.0
            tint_g = float(self._get_panel_val("bld_tint_g", 100.0)) / 100.0
            tint_b = float(self._get_panel_val("bld_tint_b", 100.0)) / 100.0
            self._fp_prog_bldg['u_bld_tint'].write(
                np.array([tint_r, tint_g, tint_b], dtype='f4').tobytes())
            # Enable blending for opacity < 1
            bld_opacity = float(self._get_panel_val("bld_opacity", 100.0)) / 100.0
            if bld_opacity < 0.99:
                self._ctx.enable(moderngl.BLEND)
                self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._fp_bldg_vao.render(moderngl.TRIANGLES)
            if bld_opacity < 0.99:
                self._ctx.disable(moderngl.BLEND)

        # 4b. v7: Trees (reuse building shader — same vertex layout, same lighting)
        if self._trees_enabled and self._fp_tree_vao and self._fp_tree_count > 0:
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._fp_prog_bldg['u_view'].write(view_gl.tobytes())
            self._fp_prog_bldg['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_bldg['u_cam_pos'].write(cam_pos_gl.tobytes())
            self._fp_prog_bldg['u_light_dir'].write(light_dir.tobytes())
            self._fp_prog_bldg['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_bldg['u_fog_near'].value = float(fog_near)
            self._fp_prog_bldg['u_fog_far'].value = float(fog_far)
            self._fp_prog_bldg['u_time'].value = float(time.time() % 1000.0)
            self._fp_prog_bldg['u_wall_opacity'].value = 1.0
            self._fp_prog_bldg['u_win_glow'].value = 0.0  # no windows on trees
            self._fp_prog_bldg['u_ao_strength'].value = 0.5
            self._fp_prog_bldg['u_bld_tint'].write(
                np.array([1.0, 1.0, 1.0], dtype='f4').tobytes())
            self._fp_tree_vao.render(moderngl.TRIANGLES)

        # 5. Car model (3rd person) — origin is car position, rotation only
        #    Terrain is already handled by cam_pos_gl offset — ground/buildings are at rel.y≈0
        #    so car model at rel.y=0 sits on the ground correctly.
        if self._car_mode and self._car_vao and self._car_vert_count > 0:
            car_yaw_rad = math.radians(self._car_yaw) + math.pi  # +180° so front faces forward
            cos_y = math.cos(car_yaw_rad); sin_y = math.sin(car_yaw_rad)
            M = np.eye(4, dtype='f4')
            M[0,0] = cos_y; M[0,1] = -sin_y
            M[1,0] = sin_y; M[1,1] = cos_y
            M_gl = np.ascontiguousarray(M.T).astype('f4')
            self._fp_prog_car['u_view'].write(view_gl.tobytes())
            self._fp_prog_car['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_car['u_model'].write(M_gl.tobytes())
            self._fp_prog_car['u_light_dir'].write(light_dir.tobytes())
            self._fp_prog_car['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_car['u_fog_near'].value = float(fog_near)
            self._fp_prog_car['u_fog_far'].value = float(fog_far)
            self._car_vao.render(moderngl.TRIANGLES)

        # 6. Traffic NPC cars (rendered as a batch, same shader as player car)
        if self._traffic_enabled and self._car_mode and self._traffic_vao and self._traffic_vert_count > 0:
            # NPC cars are pre-transformed relative to car origin, so use identity model
            M_id = np.eye(4, dtype='f4')
            M_id_gl = np.ascontiguousarray(M_id.T).astype('f4')
            self._fp_prog_car['u_view'].write(view_gl.tobytes())
            self._fp_prog_car['u_proj'].write(proj_gl.tobytes())
            self._fp_prog_car['u_model'].write(M_id_gl.tobytes())
            self._fp_prog_car['u_light_dir'].write(light_dir.tobytes())
            self._fp_prog_car['u_fog_color'].write(fog_color.tobytes())
            self._fp_prog_car['u_fog_near'].value = float(fog_near)
            self._fp_prog_car['u_fog_far'].value = float(fog_far)
            self._traffic_vao.render(moderngl.TRIANGLES)

        # 7. POI service billboards (floating markers above ground)
        if self._poi_enabled and self._fp_prog_poi:
            # Rebuild POI data when camera crosses tile boundary or forced
            poi_key = (z_int, cam_tx, cam_ty)
            if poi_key != self._poi_cache_key or self._fp_poi_rebuild:
                self._poi_cache_key = poi_key
                self._fp_poi_rebuild = False
                with self._mvt_lock:
                    vis_keys = []
                    n_t = 2 ** z_int
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            tkey = (z_int, (cam_tx + dx) % n_t, max(0, min(n_t - 1, cam_ty + dy)))
                            if tkey in self._mvt_cache:
                                vis_keys.append(tkey)
                    mvt_snap = {k: self._mvt_cache[k] for k in vis_keys if k in self._mvt_cache}
                self._poi_cache = _extract_pois_from_tiles(mvt_snap, vis_keys, tile_px, z)
                # Build billboard geometry
                poi_verts, self._poi_visible_list = _build_poi_billboards_fp(
                    self._poi_cache, cam_wx, cam_wy, mpp)
                for attr in ('_fp_poi_vbo', '_fp_poi_vao'):
                    o = getattr(self, attr)
                    if o:
                        try: o.release()
                        except: pass
                if len(poi_verts) > 0:
                    self._fp_poi_vbo = self._ctx.buffer(
                        np.ascontiguousarray(poi_verts).tobytes())
                    self._fp_poi_vao = self._ctx.vertex_array(
                        self._fp_prog_poi,
                        [(self._fp_poi_vbo, '3f 3f 4f 1f 1f',
                          'in_pos', 'in_normal', 'in_color', 'in_height', 'in_icon_id')])
                    self._fp_poi_count = len(poi_verts)
                else:
                    self._fp_poi_vbo = None; self._fp_poi_vao = None
                    self._fp_poi_count = 0

            if self._fp_poi_vao and self._fp_poi_count > 0:
                self._ctx.enable(moderngl.BLEND)
                self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

                self._fp_prog_poi['u_view'].write(view_gl.tobytes())
                self._fp_prog_poi['u_proj'].write(proj_gl.tobytes())
                self._fp_prog_poi['u_cam_pos'].write(cam_pos_gl.tobytes())
                self._fp_prog_poi['u_fog_color'].write(fog_color.tobytes())
                self._fp_prog_poi['u_fog_near'].value = float(fog_near)
                self._fp_prog_poi['u_fog_far'].value = float(fog_far)
                self._fp_prog_poi['u_time'].value = float(time.time() % 1000.0)
                _set_terrain_uniforms(self._fp_prog_poi)
                self._fp_poi_vao.render(moderngl.TRIANGLES)

        self._ctx.disable(moderngl.DEPTH_TEST)

        # --- Optimized FBO readback ---
        raw = self._fbo.color_attachments[0].read()
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        flipped = np.ascontiguousarray(arr[::-1])
        self._cached_frame = QImage(flipped.data, w, h, w*4, QImage.Format_RGBA8888).copy()

        # --- QPainter overlay ---
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self._cached_frame and not self._cached_frame.isNull():
            painter.drawImage(0, 0, self._cached_frame)

        self._paint_immersive_hud(painter, w, h, lat, lon)

        # POI detail popup (immersive proximity)
        if self._poi_popup and self._immersive:
            self._draw_poi_popup(painter, w, h)

        # Search bar overlay (on top of everything)
        if self._search_active:
            self._draw_search_bar(painter, w, h)

        painter.end()

    def _paint_immersive_hud(self, painter, w, h, lat, lon):
        """Draw the immersive mode HUD overlay."""
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self._car_mode:
            # --- COLLISION FLASH (red screen overlay) ---
            if self._collision_flash > 0:
                flash_alpha = int(self._collision_flash * 120)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 40, 20, flash_alpha))
                painter.drawRect(0, 0, w, h)
                # Screen shake effect via text
                if self._collision_flash > 0.5:
                    painter.setPen(QColor(255, 60, 40, 220))
                    painter.setFont(QFont("monospace", 14, QFont.Bold))
                    painter.drawText(w // 2 - 50, h // 2 - 20, 100, 40,
                                    Qt.AlignCenter, "CRASH!")

            # --- CAR MODE HUD ---
            # Speedometer (enhanced with RPM-style gauge)
            speed_mps = self._car_speed * self._car_mpp
            speed_kmh = speed_mps * 3.6
            gw = 190; gh = 72
            gx = w // 2 - gw // 2; gy = h - gh - 28

            # Gauge background with gradient
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(10, 12, 20, 230))
            painter.drawRoundedRect(gx, gy, gw, gh, 14, 14)
            # Gauge border glow based on speed
            if speed_kmh > 80:
                glow_c = QColor(255, 80, 40, int(min(1.0, (speed_kmh - 80) / 40.0) * 100))
            elif speed_kmh > 0:
                glow_c = QColor(60, 180, 255, 40)
            else:
                glow_c = QColor(40, 60, 100, 30)
            painter.setPen(QPen(glow_c, 2)); painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(gx+1, gy+1, gw-2, gh-2, 13, 13)

            # Speed text
            painter.setPen(QColor(255, 255, 255)); painter.setFont(QFont("monospace", 28, QFont.Bold))
            painter.drawText(gx + 8, gy + 4, gw - 60, gh - 18, Qt.AlignCenter | Qt.AlignVCenter, f"{speed_kmh:.0f}")
            painter.setPen(QColor(120, 155, 200)); painter.setFont(QFont("monospace", 9))
            painter.drawText(gx + gw - 55, gy + gh // 2 - 8, "km/h")

            # Speed bar
            bx = gx + 12; by = gy + gh - 14; bw = gw - 24
            fill = min(1.0, speed_kmh / 120.0)
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(30, 35, 50))
            painter.drawRoundedRect(bx, by, bw, 6, 3, 3)
            if fill > 0:
                if speed_kmh < 60:
                    bc = QColor(60, 220, 120)
                elif speed_kmh < 90:
                    bc = QColor(255, 210, 50)
                else:
                    bc = QColor(255, 50, 30)
                painter.setBrush(bc)
                painter.drawRoundedRect(bx, by, int(bw * fill), 6, 3, 3)

            # Damage indicator (collision penalty)
            if self._collision_speed_penalty > 0:
                dmg_pct = int(self._collision_speed_penalty * 100)
                painter.setPen(Qt.NoPen); painter.setBrush(QColor(60, 10, 5, 200))
                painter.drawRoundedRect(gx, gy - 24, gw, 20, 10, 10)
                painter.setPen(QColor(255, 80, 60)); painter.setFont(QFont("monospace", 8, QFont.Bold))
                painter.drawText(gx, gy - 24, gw, 20, Qt.AlignCenter, f"DAMAGE {dmg_pct}%")
                # Damage bar
                painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 50, 30, 180))
                painter.drawRoundedRect(gx + 8, gy - 6, int((gw - 16) * self._collision_speed_penalty), 3, 1, 1)

            # Mode badge
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 60, 30, 220))
            painter.drawRoundedRect(w // 2 - 55, 8, 110, 26, 13, 13)
            painter.setPen(QColor(80, 255, 160)); painter.setFont(QFont("monospace", 10, QFont.Bold))
            painter.drawText(w // 2 - 55, 8, 110, 26, Qt.AlignCenter, "\u25cf CAR MODE")

            # Traffic badge
            if self._traffic_enabled:
                n_alive = len(self._traffic_cars)
                painter.setPen(Qt.NoPen); painter.setBrush(QColor(50, 30, 0, 220))
                painter.drawRoundedRect(w // 2 - 55, 38, 110, 22, 11, 11)
                painter.setPen(QColor(255, 200, 80)); painter.setFont(QFont("monospace", 8, QFont.Bold))
                painter.drawText(w // 2 - 55, 38, 110, 22, Qt.AlignCenter, f"TRAFFIC: {n_alive}")

            # Brake/throttle indicators (left side)
            ind_x = 16; ind_y = h // 2 - 60
            # Throttle
            t_fill = self._car_throttle
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(20, 25, 35, 180))
            painter.drawRoundedRect(ind_x, ind_y, 12, 50, 4, 4)
            if t_fill > 0:
                painter.setBrush(QColor(60, 220, 120, 200))
                h_fill = int(50 * t_fill)
                painter.drawRoundedRect(ind_x, ind_y + 50 - h_fill, 12, h_fill, 4, 4)
            painter.setPen(QColor(100, 130, 160, 150)); painter.setFont(QFont("monospace", 6))
            painter.drawText(ind_x - 2, ind_y + 54, "GAS")

            # Brake
            b_fill = self._car_brake
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(20, 25, 35, 180))
            painter.drawRoundedRect(ind_x, ind_y + 70, 12, 50, 4, 4)
            if b_fill > 0:
                painter.setBrush(QColor(255, 60, 40, 200))
                h_fill = int(50 * b_fill)
                painter.drawRoundedRect(ind_x, ind_y + 70 + 50 - h_fill, 12, h_fill, 4, 4)
            painter.setPen(QColor(100, 130, 160, 150)); painter.setFont(QFont("monospace", 6))
            painter.drawText(ind_x - 4, ind_y + 124, "BRK")

            # Steering indicator
            steer_cx = w // 2; steer_cy = h - gh - 38
            steer_w = 80
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(20, 25, 35, 120))
            painter.drawRoundedRect(steer_cx - steer_w//2, steer_cy, steer_w, 6, 3, 3)
            dot_x = steer_cx + int(self._car_steer * steer_w * 0.4)
            painter.setBrush(QColor(80, 200, 255, 200))
            painter.drawEllipse(QPointF(dot_x, steer_cy + 3), 4, 4)

            # Mini-map / rearview hint
            painter.setPen(QColor(80, 90, 110, 120)); painter.setFont(QFont("monospace", 7))
            painter.drawText(w - 100, h - 16, "P toggle traffic")

            # Controls
            hints = "W/S gas/brake  A/D steer  Space brake  G exit  \u2191\u2193 POI  \u23CE nav  E/X/J map/web/wiki"
            hw_px = len(hints) * 5.5 + 20
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(int(w // 2 - hw_px // 2), h - 18, int(hw_px), 16, 8, 8)
            painter.setPen(QColor(160, 165, 180, 180)); painter.setFont(QFont("monospace", 7))
            painter.drawText(int(w // 2 - hw_px // 2), h - 18, int(hw_px), 16, Qt.AlignCenter, hints)

        else:
            # --- WALKING MODE HUD ---
            # Crosshair
            cx, cy = w // 2, h // 2
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
            painter.drawLine(cx - 12, cy, cx - 4, cy)
            painter.drawLine(cx + 4, cy, cx + 12, cy)
            painter.drawLine(cx, cy - 12, cx, cy - 4)
            painter.drawLine(cx, cy + 4, cx, cy + 12)

            # Info bar
            info_text = f"IMMERSIVE  {lat:.5f}, {lon:.5f}  yaw {self._fp_yaw:.0f}\u00b0  pitch {self._fp_pitch_angle:.0f}\u00b0"
            pw = len(info_text) * 7 + 24
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 200))
            painter.drawRoundedRect(w // 2 - pw // 2, 8, pw, 26, 13, 13)
            painter.setPen(QColor(100, 200, 255)); painter.setFont(QFont("monospace", 10, QFont.Bold))
            painter.drawText(w // 2 - pw // 2, 8, pw, 26, Qt.AlignCenter, info_text)

            # Car hint
            car_hint = "G  enter car mode  |  P  traffic"
            chw = len(car_hint) * 7 + 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 50, 35, 190))
            painter.drawRoundedRect(8, 42, chw, 20, 10, 10)
            painter.setPen(QColor(80, 255, 160)); painter.setFont(QFont("monospace", 8))
            painter.drawText(8, 42, chw, 20, Qt.AlignCenter, car_hint)

            # Controls
            hints = "WASD move  |  G car  |  O POI  |  \u2191\u2193 select  \u23CE nav  |  E/X/J  |  ESC exit"
            hw_px = len(hints) * 6 + 20
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(w // 2 - hw_px // 2, h - 36, hw_px, 24, 12, 12)
            painter.setPen(QColor(180, 180, 200, 200)); painter.setFont(QFont("monospace", 9))
            painter.drawText(w // 2 - hw_px // 2, h - 36, hw_px, 24, Qt.AlignCenter, hints)

        # Compass (shared)
        compass_x, compass_y, compass_r = w - 50, 50, 28
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
        painter.drawEllipse(QPointF(compass_x, compass_y), compass_r + 4, compass_r + 4)
        disp_yaw = self._car_yaw if self._car_mode else self._fp_yaw
        yr = math.radians(disp_yaw)
        nx = compass_x - math.sin(yr) * compass_r * 0.8
        ny = compass_y - math.cos(yr) * compass_r * 0.8
        painter.setPen(QPen(QColor(255, 80, 80), 2))
        painter.drawLine(QPointF(compass_x, compass_y), QPointF(nx, ny))
        painter.setPen(QColor(255, 80, 80)); painter.setFont(QFont("sans-serif", 8, QFont.Bold))
        painter.drawText(int(nx - 4), int(ny - 4), "N")
        sx = compass_x + math.sin(yr) * compass_r * 0.6
        sy = compass_y + math.cos(yr) * compass_r * 0.6
        painter.setPen(QPen(QColor(180, 180, 200, 120), 1))
        painter.drawLine(QPointF(compass_x, compass_y), QPointF(sx, sy))

        # --- POI Nearby Services Panel (right side) ---
        if self._poi_enabled and self._poi_visible_list:
            panel_w = 200; panel_x = w - panel_w - 12
            panel_y = 90; row_h = 22
            max_show = min(12, len(self._poi_visible_list))

            # Panel background
            panel_h = 28 + max_show * row_h + 24  # +24 for nav hint
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(8, 10, 18, 210))
            painter.drawRoundedRect(panel_x, panel_y, panel_w, panel_h, 10, 10)
            # Panel border
            painter.setPen(QPen(QColor(60, 80, 120, 100), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(panel_x + 1, panel_y + 1, panel_w - 2, panel_h - 2, 9, 9)

            # Title
            painter.setPen(QColor(120, 180, 255)); painter.setFont(QFont("sans-serif", 9, QFont.Bold))
            painter.drawText(panel_x + 10, panel_y + 6, panel_w - 20, 20,
                            Qt.AlignLeft | Qt.AlignVCenter,
                            f"\U0001F4CD Nearby ({len(self._poi_visible_list)})")

            # Separator
            painter.setPen(QPen(QColor(40, 60, 100, 120), 1))
            painter.drawLine(panel_x + 10, panel_y + 26, panel_x + panel_w - 10, panel_y + 26)

            # Compute distances for each visible POI
            cam_wx_d = self._car_wx if self._car_mode else self._fp_cx
            cam_wy_d = self._car_wy if self._car_mode else self._fp_cy
            z_d = self._zoom
            lat_d = _wy_to_lat(cam_wy_d, z_d)
            mpp_d = (40_075_000 * math.cos(math.radians(lat_d))) / _world_size(z_d)

            poi_with_dist = []
            for poi in self._poi_visible_list:
                dx = poi["wx"] - cam_wx_d
                dy = poi["wy"] - cam_wy_d
                dist_m = math.hypot(dx, dy) * mpp_d
                poi_with_dist.append((dist_m, poi))
            poi_with_dist.sort(key=lambda x: x[0])

            for i, (dist_m, poi) in enumerate(poi_with_dist[:max_show]):
                ry = panel_y + 30 + i * row_h

                # Selection highlight
                is_selected = (self._poi_select_active and i == self._poi_select_idx)
                if is_selected:
                    cr_s, cg_s, cb_s, _ = poi["color"]
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(int(cr_s * 255), int(cg_s * 255), int(cb_s * 255), 50))
                    painter.drawRoundedRect(panel_x + 4, ry - 1, panel_w - 8, row_h, 4, 4)
                    # Selection arrow
                    painter.setPen(QColor(255, 255, 255, 230))
                    painter.setFont(QFont("sans-serif", 9, QFont.Bold))
                    painter.drawText(panel_x + 4, ry, 12, row_h, Qt.AlignVCenter, "\u25B6")

                # Category color dot
                cr, cg, cb, ca = poi["color"]
                dot_color = QColor(int(cr * 255), int(cg * 255), int(cb * 255), 255 if is_selected else 220)
                painter.setPen(Qt.NoPen); painter.setBrush(dot_color)
                painter.drawEllipse(QPointF(panel_x + 16, ry + 8), 4, 4)

                # Icon
                painter.setPen(QColor(200, 210, 230)); painter.setFont(QFont("sans-serif", 9))
                painter.drawText(panel_x + 24, ry, 16, row_h, Qt.AlignVCenter, poi["icon"])

                # Name (truncated)
                disp_name = poi["name"]
                if len(disp_name) > 16:
                    disp_name = disp_name[:15] + "\u2026"
                painter.setPen(QColor(200, 210, 230)); painter.setFont(QFont("sans-serif", 8))
                painter.drawText(panel_x + 40, ry, panel_w - 90, row_h,
                                Qt.AlignLeft | Qt.AlignVCenter, disp_name)

                # Distance
                if dist_m < 1000:
                    dist_str = f"{dist_m:.0f}m"
                else:
                    dist_str = f"{dist_m / 1000:.1f}km"
                painter.setPen(QColor(100, 140, 180, 180)); painter.setFont(QFont("monospace", 7))
                painter.drawText(panel_x + panel_w - 48, ry, 40, row_h,
                                Qt.AlignRight | Qt.AlignVCenter, dist_str)

            # Navigation hint at bottom of panel
            hint_y = panel_y + 30 + max_show * row_h + 2
            painter.setPen(QColor(80, 110, 160, 160)); painter.setFont(QFont("monospace", 7))
            painter.drawText(panel_x + 6, hint_y, panel_w - 12, 16,
                            Qt.AlignCenter, "\u2191\u2193 select  \u23CE navigate")

        # --- Floating POI name labels above halos (projected to screen) ---
        if self._poi_enabled and self._poi_visible_list:
            cam_wx_l = self._car_wx if self._car_mode else self._fp_cx
            cam_wy_l = self._car_wy if self._car_mode else self._fp_cy
            z_l = self._zoom
            lat_l = _wy_to_lat(cam_wy_l, z_l)
            mpp_l = (40_075_000 * math.cos(math.radians(lat_l))) / _world_size(z_l)
            yaw_l = self._car_yaw if self._car_mode else self._fp_yaw
            pitch_l = self._fp_pitch_angle
            yaw_rad_l = math.radians(yaw_l)
            pitch_rad_l = math.radians(pitch_l)
            fwd_x_l = math.sin(yaw_rad_l)
            fwd_y_l = -math.cos(yaw_rad_l)
            eye_h = self._fp_eye_height
            terrain_ok = bool(self._terrain_enabled and self._terrain_cache)

            # Use actual camera position for projection
            proj_cx = self._fp_cx
            proj_cy = self._fp_cy

            painter.setRenderHint(QPainter.Antialiasing, True)
            label_count = 0
            for poi in self._poi_visible_list:
                if label_count >= 20:
                    break
                dx_l = poi["wx"] - cam_wx_l
                dy_l = poi["wy"] - cam_wy_l
                dist_l = math.hypot(dx_l, dy_l)
                dist_m_l = dist_l * mpp_l
                if dist_m_l > 120.0 or dist_m_l < 1.0:
                    continue
                # Check if in front of camera
                dot_l = dx_l * fwd_x_l + dy_l * fwd_y_l
                if dot_l < 0:
                    continue
                # Get terrain elevation
                elev_z = 0.0
                if terrain_ok:
                    elev_m = _sample_terrain_bilinear(self._terrain_cache, poi["wx"], poi["wy"], z_l)
                    elev_z = elev_m / max(mpp_l, 0.001)
                # Label floats above the halo cylinder (~12m above ground)
                label_z = elev_z + 12.0 / max(mpp_l, 0.001)
                # Project to screen
                pdx = poi["wx"] - proj_cx
                pdy = poi["wy"] - proj_cy
                rel_x = -(pdx)
                rel_y = label_z - eye_h
                rel_z = -(pdy)
                # Yaw
                cos_y = math.cos(-yaw_rad_l); sin_y = math.sin(-yaw_rad_l)
                rx = rel_x * cos_y - rel_z * sin_y
                rz = rel_x * sin_y + rel_z * cos_y
                # Pitch
                cos_p = math.cos(pitch_rad_l); sin_p = math.sin(pitch_rad_l)
                ry = rel_y * cos_p - rz * sin_p
                rz2 = rel_y * sin_p + rz * cos_p
                if rz2 < 0.5:
                    continue
                fov = 70.0; aspect = w / max(h, 1)
                f = 1.0 / math.tan(math.radians(fov / 2.0))
                sx = int(w / 2 + rx * f / rz2 * w / 2)
                sy = int(h / 2 - ry * f * aspect / rz2 * h / 2)
                if sx < -100 or sx > w + 100 or sy < -50 or sy > h + 50:
                    continue
                # Distance-based alpha and size
                alpha_l = max(0.3, min(1.0, 1.0 - dist_m_l / 130.0))
                font_sz = max(7, min(12, int(10 * (50.0 / max(dist_m_l, 10.0)))))
                a_int = int(alpha_l * 255)
                # Background pill
                name_l = poi["name"]
                icon_l = poi["icon"]
                label_text = f"{icon_l} {name_l}"
                if len(label_text) > 25:
                    label_text = label_text[:24] + "\u2026"
                tw = len(label_text) * (font_sz - 1) + 14
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(10, 14, 24, int(alpha_l * 200)))
                painter.drawRoundedRect(sx - tw // 2, sy - font_sz, tw, font_sz + 8, 6, 6)
                # Category accent bar
                cr_l, cg_l, cb_l, _ = poi["color"]
                painter.setBrush(QColor(int(cr_l * 255), int(cg_l * 255), int(cb_l * 255), a_int))
                painter.drawRoundedRect(sx - tw // 2, sy - font_sz, 3, font_sz + 8, 2, 2)
                # Text
                painter.setPen(QColor(220, 230, 250, a_int))
                painter.setFont(QFont("sans-serif", font_sz, QFont.Bold))
                painter.drawText(sx - tw // 2 + 6, sy - font_sz, tw - 12, font_sz + 8,
                                Qt.AlignCenter, label_text)
                # Distance below
                dist_str_l = f"{dist_m_l:.0f}m" if dist_m_l < 1000 else f"{dist_m_l / 1000:.1f}km"
                painter.setPen(QColor(120, 155, 200, int(alpha_l * 160)))
                painter.setFont(QFont("monospace", max(6, font_sz - 2)))
                painter.drawText(sx - 25, sy + 4, 50, 12, Qt.AlignCenter, dist_str_l)
                label_count += 1

        # POI badge (when enabled)
        if self._poi_enabled:
            badge_y = 38 if not (self._car_mode and self._traffic_enabled) else 64
            if self._car_mode:
                badge_y = 64 if self._traffic_enabled else 38
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(40, 20, 60, 220))
            painter.drawRoundedRect(w // 2 - 55, badge_y, 110, 22, 11, 11)
            n_pois = len(self._poi_visible_list) if self._poi_visible_list else len(self._poi_cache)
            painter.setPen(QColor(200, 140, 255)); painter.setFont(QFont("monospace", 8, QFont.Bold))
            painter.drawText(w // 2 - 55, badge_y, 110, 22, Qt.AlignCenter,
                            f"\U0001F4CD POI: {n_pois}")

        # Loading indicator
        loading_count = len(self._pending) + len(self._rendering) + len(self._gpu_upload_queue)
        if loading_count > 0:
            dots = "." * (self._frame % 4 + 1)
            txt = f"loading {loading_count}{dots}"
            tw = len(txt) * 7 + 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(8, 8, tw, 22, 11, 11)
            painter.setPen(QColor(255, 200, 80)); painter.setFont(QFont("monospace", 9))
            painter.drawText(8, 8, tw, 22, Qt.AlignCenter, txt)

        # Attribution
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawRect(0, h - 16, w, 16)
        painter.setPen(QColor(180, 180, 180, 140)); painter.setFont(QFont("monospace", 7))
        painter.drawText(6, h - 3, "(c) Mapbox  (c) OpenStreetMap  [Immersive v6.2]")

    # -----------------------------------------------------------------
    #  Route overlay
    # -----------------------------------------------------------------

    def _draw_route_overlay(self, painter, w, h):
        if not self._route_points and not self._route_mode:
            return
        bearing_rad = math.radians(self._bearing)
        pitch_rad = math.radians(self._pitch)
        cos_b = math.cos(bearing_rad); sin_b = math.sin(bearing_rad)
        cos_p = math.cos(pitch_rad) if self._pitch > 0.5 else 1.0
        fov_f = (1.0 - 0.6 * math.sin(pitch_rad)) if self._pitch > 0.5 else 1.0
        hw, hh = w / 2.0, h / 2.0

        painter.setRenderHint(QPainter.Antialiasing, True)

        for i, (lat, lon) in enumerate(self._route_points):
            wx = _lon_to_wx(lon, self._zoom)
            wy = _lat_to_wy(lat, self._zoom)
            dx = wx - self._cx; dy = wy - self._cy
            sx = dx * cos_b + dy * sin_b
            sy = -dx * sin_b + dy * cos_b
            if self._pitch > 0.5:
                sy = sy * (cos_p * fov_f + (1.0 - cos_p) * 0.6) + h * 0.35 * math.sin(pitch_rad)
            px, py = hw + sx, hh + sy
            if px < -20 or px > w + 20 or py < -20 or py > h + 20:
                continue
            r = 8 if (i == 0 or i == len(self._route_points) - 1) else 5
            if i == 0:
                color = QColor(40, 200, 80)
            elif i == len(self._route_points) - 1:
                color = QColor(220, 60, 60)
            else:
                color = QColor(60, 140, 255)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(px, py), r, r)
            if i == 0 or i == len(self._route_points) - 1:
                label = "A" if i == 0 else chr(65 + min(i, 25))
                painter.setPen(QColor(255, 255, 255))
                font = QFont("sans-serif", 7, QFont.Bold)
                painter.setFont(font)
                painter.drawText(int(px - 4), int(py + 3), label)

        if len(self._route_points) >= 2 and self._route_total_dist > 0:
            dist = self._route_total_dist
            dist_str = f"{dist:.0f} m" if dist < 1000 else f"{dist/1000:.2f} km"
            dur = self._route_duration
            if dur > 0:
                dur_str = f"{dur:.0f}s" if dur < 60 else (f"{dur/60:.0f}min" if dur < 3600 else f"{dur/3600:.1f}h")
            else:
                dur_str = "..."
            profile_icon = {"driving": "\u2b62", "walking": "\u2b65", "cycling": "\u2b64"}.get(self._route_profile, "")
            info = f"{profile_icon} {dist_str}  {dur_str}  ({self._route_profile})"
            if self._route_pending:
                info = "Calculating route..."
            pw = len(info) * 7 + 20
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 190))
            painter.drawRoundedRect(w // 2 - pw // 2, h - 65, pw, 22, 11, 11)
            painter.setPen(QColor(100, 200, 255)); painter.setFont(QFont("monospace", 9))
            painter.drawText(w // 2 - pw // 2, h - 65, pw, 22, Qt.AlignCenter, info)

        if self._route_mode:
            msg = f"ROUTE [{self._route_profile}]  Click: add pt  C: clear  M: mode  N: exit"
            pw = len(msg) * 7 + 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 80, 180, 200))
            painter.drawRoundedRect(w // 2 - pw // 2, 56, pw, 22, 11, 11)
            painter.setPen(QColor(255, 255, 255)); painter.setFont(QFont("monospace", 9))
            painter.drawText(w // 2 - pw // 2, 56, pw, 22, Qt.AlignCenter, msg)

    # -----------------------------------------------------------------
    #  Heatmap legend
    # -----------------------------------------------------------------

    def _draw_heatmap_legend(self, painter, w, h):
        lx, ly, lw, lh = w - 160, h - 110, 140, 65
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
        painter.drawRoundedRect(lx, ly, lw, lh, 8, 8)
        bar_y = ly + 8; bar_h = 12
        colors = [(0, 0, 128), (0, 204, 255), (0, 255, 50), (255, 255, 0), (255, 25, 0)]
        step = (lw - 20) / (len(colors) - 1)
        for i in range(len(colors) - 1):
            grad = QLinearGradient(lx + 10 + i * step, 0, lx + 10 + (i+1) * step, 0)
            c1, c2 = colors[i], colors[i+1]
            grad.setColorAt(0, QColor(*c1)); grad.setColorAt(1, QColor(*c2))
            painter.setBrush(QBrush(grad)); painter.setPen(Qt.NoPen)
            painter.drawRect(int(lx + 10 + i * step), bar_y, int(step) + 1, bar_h)
        # Draw filter threshold marker on gradient bar
        heat_filter = self._get_panel_val("heat_filter", 0.0) / 100.0
        if heat_filter > 0.01:
            marker_x = int(lx + 10 + heat_filter * (lw - 20))
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.drawLine(marker_x, bar_y - 2, marker_x, bar_y + bar_h + 2)
            # Dim the region below the filter
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 120))
            painter.drawRect(lx + 10, bar_y, marker_x - lx - 10, bar_h)
        painter.setPen(QColor(200, 200, 200)); painter.setFont(QFont("monospace", 7))
        painter.drawText(lx + 10, ly + 28, "Low")
        painter.drawText(lx + lw - 35, ly + 28, "High")
        painter.setPen(QColor(140, 180, 220)); painter.setFont(QFont("monospace", 8))
        mode_labels = {"poi": "POI density", "traffic": "Traffic density", "building": "Population density"}
        label = mode_labels.get(self._heatmap_mode, self._heatmap_mode)
        painter.drawText(lx + 10, ly + 42, label)
        if heat_filter > 0.01:
            painter.setPen(QColor(255, 180, 100)); painter.setFont(QFont("monospace", 7))
            painter.drawText(lx + 10, ly + 56, f"filter: \u2265{heat_filter:.0%}")

    # -----------------------------------------------------------------
    #  Label overlay (cross-tile deduplication)
    # -----------------------------------------------------------------

    def _draw_labels_overlay(self, painter, all_label_data, w, h, tile_px):
        if not all_label_data or not self._labels_visible: return

        # Hotfix: while interacting, just reuse the cached label image
        if (self._drag_start is not None or
            self._rdrag_start is not None or
            self._mdrag_start is not None or
            self._fly.active or
            self._scroll_zoom_active or
            self._zoom_velocity > 0.03):
            if getattr(self, "_label_overlay_img", None) is not None:
                painter.drawImage(0, 0, self._label_overlay_img)
            return

        # v7: Cache labels to a QImage — only rebuild when view changes meaningfully
        z = self._zoom
        z_int = int(math.floor(z + 0.5))
        bearing_snap = round(self._bearing, 0)
        pitch_snap = round(self._pitch, 0)
        label_key = (z_int, bearing_snap, pitch_snap, len(all_label_data),
                     round(self._cx / 100), round(self._cy / 100))
        if (hasattr(self, '_label_overlay_key') and self._label_overlay_key == label_key
                and hasattr(self, '_label_overlay_img') and self._label_overlay_img is not None):
            painter.drawImage(0, 0, self._label_overlay_img)
            return

        bearing_rad = math.radians(self._bearing)
        pitch_rad = math.radians(self._pitch)
        cos_b = math.cos(bearing_rad); sin_b = math.sin(bearing_rad)
        cos_p = math.cos(pitch_rad) if self._pitch > 0.5 else 1.0
        fov_f = (1.0 - 0.6 * math.sin(pitch_rad)) if self._pitch > 0.5 else 1.0

        hw, hh = w / 2.0, h / 2.0

        candidates = []
        for tile_ox, tile_oy, t_px, labels in all_label_data:
            for priority, name, nx, ny, ls in labels:
                wx = tile_ox + nx * t_px
                wy = tile_oy + ny * t_px
                dx = wx - self._cx; dy = wy - self._cy
                sx = dx * cos_b + dy * sin_b
                sy = -dx * sin_b + dy * cos_b
                if self._pitch > 0.5:
                    sy = sy * (cos_p * fov_f + (1.0 - cos_p) * 0.6) + h * 0.35 * math.sin(pitch_rad)
                px = hw + sx; py = hh + sy
                if px < -80 or px > w + 80 or py < -30 or py > h + 30: continue
                candidates.append((priority, name, px, py, ls))

        if not candidates:
            self._label_overlay_key = label_key
            self._label_overlay_img = None
            return
        candidates.sort(key=lambda c: c[0])

        # Render to transparent QImage cache
        label_img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        label_img.fill(QColor(0, 0, 0, 0))
        lp = QPainter(label_img)
        lp.setRenderHint(QPainter.Antialiasing, True)

        LABEL_CELL = 48
        grid_cols = (w + LABEL_CELL - 1) // LABEL_CELL
        occupied = set(); seen = set()
        max_labels = max(4, min(25, int(z * 2))); placed = 0

        for priority, name, px, py, ls in candidates:
            if placed >= max_labels: break
            nl = name.lower()
            if nl in seen: continue
            fs = ls.get("font_size", 10)
            fs = max(7, int(fs * min(1.4, z / 12.0)))
            aw = len(name) * fs * 0.62; ah = fs * 1.4
            c0 = max(0, int((px - aw*0.3) / LABEL_CELL))
            c1 = min(grid_cols-1, int((px + aw*0.7) / LABEL_CELL))
            r0 = max(0, int((py - ah) / LABEL_CELL))
            r1 = min(grid_cols-1, int((py + ah*0.3) / LABEL_CELL))
            collision = False; cells = []
            for gc in range(c0, c1+1):
                for gr in range(r0, r1+1):
                    cell = (gc, gr)
                    if cell in occupied: collision = True; break
                    cells.append(cell)
                if collision: break
            if collision: continue
            for cell in cells: occupied.add(cell)
            seen.add(nl); placed += 1
            font = QFont("sans-serif", fs); font.setWeight(QFont.Medium); lp.setFont(font)
            halo = ls.get("halo")
            if halo:
                # v7: multi-offset drawText replaces QPainterPath — ~10x faster
                lp.setPen(halo)
                ipx = int(px); ipy = int(py)
                sname = str(name)
                for ox, oy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1)):
                    lp.drawText(ipx+ox, ipy+oy, sname)
                lp.setPen(ls["color"])
                lp.drawText(ipx, ipy, sname)
            else:
                lp.setPen(ls["color"]); lp.setBrush(Qt.NoBrush)
                lp.drawText(int(px), int(py), str(name))

        lp.end()

        self._label_overlay_key = label_key
        self._label_overlay_img = label_img
        painter.drawImage(0, 0, label_img)

    def _draw_poi_labels_2d(self, painter, w, h):
        """Draw POI name labels on the 2D map overlay."""
        if not self._poi_cache:
            return
        z = self._zoom
        bearing_rad = math.radians(self._bearing)
        pitch_rad = math.radians(self._pitch)
        cos_b = math.cos(bearing_rad); sin_b = math.sin(bearing_rad)
        cos_p = math.cos(pitch_rad) if self._pitch > 0.5 else 1.0
        fov_f = (1.0 - 0.6 * math.sin(pitch_rad)) if self._pitch > 0.5 else 1.0
        hw_s, hh_s = w / 2.0, h / 2.0

        # Place labels with collision detection
        CELL = 40
        grid_cols = (w + CELL - 1) // CELL
        occupied = set(); placed = 0
        max_labels = max(5, min(30, int(z * 2.5)))

        painter.setRenderHint(QPainter.Antialiasing, True)

        for poi in self._poi_cache:
            if placed >= max_labels:
                break
            wx, wy = poi["wx"], poi["wy"]
            dx = wx - self._cx; dy = wy - self._cy
            sx_s = dx * cos_b + dy * sin_b
            sy_s = -dx * sin_b + dy * cos_b
            if self._pitch > 0.5:
                sy_s = sy_s * (cos_p * fov_f + (1.0 - cos_p) * 0.6) + h * 0.35 * math.sin(pitch_rad)
            px = hw_s + sx_s; py = hh_s + sy_s
            if px < -40 or px > w + 40 or py < -20 or py > h + 20:
                continue

            name = poi["name"]
            icon = poi["icon"]
            display = f"{icon} {name}"
            if len(display) > 22:
                display = display[:21] + "\u2026"

            fs = max(7, int(8 * min(1.3, z / 14.0)))
            aw = len(display) * fs * 0.55 + 8
            ah = fs * 1.6

            # Label offset (above the marker dot)
            lx = px - aw * 0.5
            ly = py - 16

            # Collision check
            c0 = max(0, int(lx / CELL))
            c1 = min(grid_cols - 1, int((lx + aw) / CELL))
            r0 = max(0, int(ly / CELL))
            r1 = min(grid_cols - 1, int((ly + ah) / CELL))
            collision = False; cells = []
            for gc in range(c0, c1 + 1):
                for gr in range(r0, r1 + 1):
                    cell = (gc, gr)
                    if cell in occupied:
                        collision = True; break
                    cells.append(cell)
                if collision:
                    break
            if collision:
                continue
            for cell in cells:
                occupied.add(cell)
            placed += 1

            # Draw label pill
            cr, cg, cb, ca = poi["color"]
            bg_col = QColor(int(cr * 50), int(cg * 50), int(cb * 50), 200)
            painter.setPen(Qt.NoPen); painter.setBrush(bg_col)
            painter.drawRoundedRect(int(lx), int(ly), int(aw), int(ah), 6, 6)
            # Border
            painter.setPen(QPen(QColor(int(cr * 180), int(cg * 180), int(cb * 180), 120), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(int(lx) + 1, int(ly) + 1, int(aw) - 2, int(ah) - 2, 5, 5)
            # Text
            font = QFont("sans-serif", fs)
            painter.setFont(font)
            painter.setPen(QColor(230, 235, 245))
            painter.drawText(int(lx + 4), int(ly), int(aw - 8), int(ah),
                            Qt.AlignCenter | Qt.AlignVCenter, display)

    # -----------------------------------------------------------------
    #  HUD
    # -----------------------------------------------------------------

    def _paint_hud(self, painter, w, h, missing):
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRect(0, h - 20, w, 20)
        painter.setPen(QColor(200, 200, 200, 160)); painter.setFont(QFont("monospace", 8))
        painter.drawText(6, h - 5, "(c) Mapbox  (c) OpenStreetMap  [MVT/GL]")

        pill_w = 380
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
        painter.drawRoundedRect(8, 8, pill_w, 22, 11, 11)
        painter.setPen(QColor(160, 220, 160)); painter.setFont(QFont("monospace", 9))
        info = f"z{self._zoom:.1f}  p{self._pitch:.0f}  b{self._bearing:.0f}  {self.centre_lat:.4f}, {self.centre_lon:.4f}"
        painter.drawText(8, 8, pill_w, 22, Qt.AlignCenter, info)

        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
        painter.drawRoundedRect(w - 150, 8, 142, 22, 11, 11)
        painter.setPen(QColor(180, 180, 220)); painter.setFont(QFont("monospace", 9))
        painter.drawText(w - 150, 8, 142, 22, Qt.AlignCenter, self._style.split("/")[-1])

        if self._buildings_3d or self._pitch > 0.5 or abs(self._bearing) > 0.5:
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
            painter.drawRoundedRect(w - 150, 34, 142, 22, 11, 11)
            painter.setPen(QColor(100, 200, 255)); painter.setFont(QFont("monospace", 9))
            tag = "3D" if self._buildings_3d else ""
            if self._pitch > 0.5: tag += f" tilt {self._pitch:.0f}"
            if abs(self._bearing) > 0.5: tag += f" rot {self._bearing:.0f}"
            painter.drawText(w - 150, 34, 142, 22, Qt.AlignCenter, tag.strip())

        badge_y = 60
        features = []
        if not self._tiles_visible:
            features.append(("tiles hidden", QColor(255, 120, 255), "transparent"))
        if not self._labels_visible:
            features.append(("labels off", QColor(255, 160, 100), ""))
        if self._shadows_enabled:
            features.append(("shadows", QColor(255, 200, 80),
                             f"sun el {self._sun_elevation:.0f}\u00b0"))
        if self._heatmap_enabled:
            features.append(("heatmap:" + self._heatmap_mode, QColor(255, 100, 50),
                             f"{len(self._heatmap_points_latlon)} pts"))
        if self._route_mode:
            features.append(("route", QColor(80, 200, 255), f"{len(self._route_points)} pts"))
        if self._poi_enabled:
            features.append(("\U0001F4CD POI", QColor(200, 140, 255), f"{len(self._poi_cache)} places"))
        for label, color, extra in features:
            txt = f"{label} {extra}".strip()
            tw = len(txt) * 7 + 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
            painter.drawRoundedRect(w - tw - 8, badge_y, tw, 18, 9, 9)
            painter.setPen(color); painter.setFont(QFont("monospace", 8))
            painter.drawText(w - tw - 8, badge_y, tw, 18, Qt.AlignCenter, txt)
            badge_y += 22

        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
        painter.drawRoundedRect(w - 150, h - 42, 142, 18, 9, 9)
        painter.setPen(QColor(80, 255, 120)); painter.setFont(QFont("monospace", 8))
        painter.drawText(w - 150, h - 42, 142, 18, Qt.AlignCenter, "GPU Direct v6.1")

        self._draw_scale(painter, w, h)

        loading_count = len(self._pending) + len(self._rendering)
        if missing > 0 or loading_count > 0:
            dots = "." * (self._frame % 4 + 1)
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 140))
            pw = 160
            painter.drawRoundedRect(w//2 - pw//2, 30, pw, 20, 8, 8)
            painter.setPen(QColor(255, 200, 80, 200)); painter.setFont(QFont("monospace", 9))
            txt = f"loading {loading_count}{dots}" if loading_count > 0 else dots
            painter.drawText(w//2 - pw//2, 30, pw, 20, Qt.AlignCenter, txt)

        if self._frame < 360:
            a = 255 if self._frame < 280 else int(255 * (360 - self._frame) / 80)
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, int(a * 0.55)))
            painter.drawRoundedRect(w - 290, h - 202, 282, 196, 10, 10)
            painter.setPen(QColor(210, 210, 210, a)); painter.setFont(QFont("monospace", 8))
            for i, txt in enumerate([
                "Left-drag: pan   Scroll: zoom",
                "Right-drag: tilt + rotate",
                "Middle-drag: rotate bearing",
                "S: style  B: 3D  T: reset tilt",
                "F: labels  R: reset  +/-: zoom",
                "N: route mode  C: clear route",
                "M: drive/walk/cycle  L: shadows",
                "H: heatmap (POI/traffic/building)",
                "V: toggle tiles (see-through)",
                "I: IMMERSIVE first-person mode",
            ]):
                painter.drawText(w - 284, h - 194 + i*18, 270, 18,
                                 Qt.AlignLeft | Qt.AlignVCenter, txt)

    def _draw_scale(self, painter, w, h):
        lat = self.centre_lat
        mpp = (40_075_000 * math.cos(math.radians(lat))) / _world_size(self._zoom)
        if mpp <= 0: return
        target_m = mpp * 80
        mag = 10 ** math.floor(math.log10(max(target_m, 1)))
        scale_m = mag * min([1,2,5,10], key=lambda n: abs(n*mag - target_m))
        bar_px = int(scale_m / mpp)
        label = f"{scale_m:.0f} m" if scale_m < 1000 else f"{scale_m/1000:.0f} km"
        bx, by = 12, h - 38
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(bx-4, by-2, bar_px+8, 18, 5, 5)
        painter.setPen(QPen(QColor(255,255,255,200), 2))
        painter.drawLine(bx, by+10, bx+bar_px, by+10)
        painter.drawLine(bx, by+6, bx, by+14)
        painter.drawLine(bx+bar_px, by+6, bx+bar_px, by+14)
        painter.setPen(QColor(255,255,255,200)); painter.setFont(QFont("monospace", 8))
        painter.drawText(bx, by-1, bar_px, 10, Qt.AlignCenter, label)

    # -----------------------------------------------------------------
    #  Invalidation
    # -----------------------------------------------------------------

    def _invalidate_visible(self):
        # Release all GPU geometry
        for entry in self._geo_cache.values():
            for k in ('fill_vbo', 'fill_vao', 'line_vbo', 'line_vao'):
                obj = entry.get(k)
                if obj:
                    try: obj.release()
                    except: pass
        self._geo_cache.clear()
        self._building_cache.clear()
        self._label_cache.clear()
        self._render_queue.clear()
        self._render_queue_set.clear()
        self._bld_gpu_key = ()
        self._shadow_gpu_key = ()
        self._heatmap_dirty = True
        self._heatmap_generated = False  # Force re-extraction from new tiles
        self._route_dirty = True
        self._poi_cache_key = None; self._poi_2d_dirty = True; self._fp_poi_rebuild = True
        self._rendering.clear()
        self._gl_dirty = True
        # v7: invalidate label and POI caches
        self._label_overlay_key = None; self._label_overlay_img = None
        self._poi_tile_key_v7 = None; self._poi_vbo_key_v7 = None

        # Re-tessellate all cached MVT tiles for new style
        with self._mvt_lock:
            all_keys = list(self._mvt_cache.keys())
        for key in all_keys:
            self._submit_render(key)

        self.update()

    # -----------------------------------------------------------------
    #  Interaction
    # -----------------------------------------------------------------

    def keyReleaseEvent(self, event):
        k = event.key()
        self._fp_keys_held.discard(k)
        if self._car_mode:
            if k == Qt.Key_W: self._car_throttle = 0.0
            elif k in (Qt.Key_S, Qt.Key_Space): self._car_brake = 0.0
            elif k in (Qt.Key_A, Qt.Key_D): self._car_steer = 0.0
        event.accept()

    def _enter_immersive(self, wx, wy):
        """Enter immersive first-person mode at world-pixel position (wx, wy)."""
        self._fp_saved_cx = self._cx
        self._fp_saved_cy = self._cy
        self._fp_saved_zoom = self._zoom
        self._fp_saved_pitch = self._pitch
        self._fp_saved_bearing = self._bearing

        self._immersive = True
        self._immersive_entering = False
        self._fp_cx = wx
        self._fp_cy = wy
        self._fp_yaw = self._bearing  # inherit map bearing
        self._fp_pitch_angle = 0.0
        self._fp_keys_held.clear()
        self._fp_mouse_captured = True
        self._fp_mouse_last = None

        # Set zoom to 17 for good building data
        self._fp_zoom_at_enter = max(self._zoom, 16.0)
        self._zoom = self._fp_zoom_at_enter
        self._target_zoom = self._fp_zoom_at_enter

        # Compute eye height: ~1.7 meters converted to world pixels
        lat = _wy_to_lat(wy, self._zoom)
        mpp = (40_075_000 * math.cos(math.radians(lat))) / _world_size(self._zoom)
        self._fp_eye_height = 1.7 / max(mpp, 0.001)
        self._fp_move_speed = 30.0 / max(mpp, 0.001)  # ~30 m/s flight speed

        # Center map on camera to load surrounding tiles
        self._cx = wx; self._cy = wy
        self._pitch = 0.0; self._target_pitch = 0.0
        self._bearing = 0.0; self._target_bearing = 0.0

        self._fp_dirty = True
        self._fp_bldg_key = ()
        self._fp_geo_key = None  # force full rebuild on enter
        self._fp_route_key = None  # v7: force route rebuild at immersive zoom
        self._gl_dirty = True

        # Request terrain DEM tiles around entry point
        self._request_terrain_around(wx, wy, radius_tiles=3)

        # Opt 5: Activate physics thread for immersive mode
        pt = self._physics_thread
        pt.set_walk_pos(wx, wy, self._fp_yaw, 0.0, self._fp_eye_height)
        pt.move_speed = self._fp_move_speed
        pt.car_mode = False
        pt._active = True

        self.setCursor(Qt.BlankCursor)
        self.setMouseTracking(True)
        self.grabKeyboard()

        # Boost loading performance for immersive mode
        self._max_concurrent_net = 12
        self._max_concurrent_render = 10
        self._mvt_cache_max = 600
        self._geo_cache_max = 300
        self._building_cache_max = 600
        self._upload_timer.setInterval(8)

        self.update()

    def _exit_immersive(self):
        """Exit immersive mode, restore previous map view."""
        if self._car_mode:
            self._exit_car_mode()
        self._immersive = False
        self._immersive_entering = False
        self._fp_keys_held.clear()
        self._fp_mouse_captured = False
        self._fp_mouse_last = None
        self._poi_clear_selection()
        if self._search_active:
            self._search_close()
        # Opt 5: Deactivate physics thread
        self._physics_thread._active = False

        # Restore saved view (but centered on where camera ended up)
        self._zoom = self._fp_saved_zoom
        self._target_zoom = self._fp_saved_zoom
        self._cx = self._fp_cx
        self._cy = self._fp_cy
        # Re-project center to new zoom
        lat = _wy_to_lat(self._fp_cy, self._fp_zoom_at_enter)
        lon = _wx_to_lon(self._fp_cx, self._fp_zoom_at_enter)
        self._cx = _lon_to_wx(lon, self._zoom)
        self._cy = _lat_to_wy(lat, self._zoom)

        self._pitch = self._fp_saved_pitch
        self._target_pitch = self._fp_saved_pitch
        self._bearing = self._fp_saved_bearing
        self._target_bearing = self._fp_saved_bearing

        # Release FP GPU resources
        for attr in ('_fp_bldg_vbo', '_fp_bldg_vao', '_fp_ground_vbo',
                      '_fp_ground_vao', '_fp_water_vbo', '_fp_water_vao',
                      '_fp_road_vbo', '_fp_road_vao',
                      '_traffic_vbo', '_traffic_vao',
                      '_fp_poi_vbo', '_fp_poi_vao'):
            obj = getattr(self, attr, None)
            if obj:
                try: obj.release()
                except: pass
                setattr(self, attr, None)
        self._fp_bldg_count = 0
        self._fp_ground_count = 0
        self._fp_water_count = 0
        self._fp_road_count = 0
        self._traffic_vert_count = 0
        self._traffic_cars.clear()
        self._traffic_spawn_key = None
        self._traffic_npc_model_cache.clear()
        self._fp_poi_count = 0
        self._poi_visible_list.clear()
        self._poi_cache_key = None
        self._fp_poi_rebuild = True

        self._gl_dirty = True
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setMouseTracking(False)
        self.releaseKeyboard()

        # Restore conservative loading defaults for normal browsing
        self._max_concurrent_net = 6
        self._max_concurrent_render = 6
        self._mvt_cache_max = 350
        self._geo_cache_max = 180
        self._building_cache_max = 300
        self._upload_timer.setInterval(12)

        self.update()

    # -----------------------------------------------------------------
    #  Car mode — free driving on city streets
    # -----------------------------------------------------------------

    def _enter_car_mode(self):
        """Spawn a car at the current walking position."""
        self._car_mode = True
        self._car_wx = float(self._fp_cx)   # float64 precision
        self._car_wy = float(self._fp_cy)
        self._car_yaw = self._fp_yaw
        self._car_speed = 0.0
        self._car_throttle = 0.0
        self._car_brake = 0.0
        self._car_steer = 0.0
        self._car_steer_angle = 0.0
        self._car_cam_yaw_off = 0.0
        self._car_last_tick = time.perf_counter()

        # Physics params from meters-per-pixel
        lat = _wy_to_lat(self._car_wy, self._zoom)
        mpp = (40_075_000 * math.cos(math.radians(lat))) / _world_size(self._zoom)
        self._car_mpp = max(mpp, 0.0001)
        self._car_max_speed = 30.0 / mpp    # 30 m/s (in wp/s)
        self._car_accel_rate = 10.0 / mpp   # 10 m/s² (in wp/s²)
        self._car_cam_dist = 14.0 / mpp     # 14m behind
        self._car_cam_height = 5.0 / mpp    # 5m up
        self._fp_eye_height = self._car_cam_height

        # Snap camera immediately behind car (no lerp lag on spawn)
        cam_yaw_rad = math.radians(self._car_yaw)
        self._fp_cx = self._car_wx - math.sin(cam_yaw_rad) * self._car_cam_dist
        self._fp_cy = self._car_wy + math.cos(cam_yaw_rad) * self._car_cam_dist

        # Request terrain around car
        self._request_terrain_around(self._car_wx, self._car_wy)

        # Build car model VBO
        car_scale = 1.0 / mpp
        car_verts = _build_car_model(car_scale)
        for a in ('_car_vbo', '_car_vao'):
            o = getattr(self, a)
            if o:
                try: o.release()
                except: pass
        if len(car_verts) > 0:
            self._car_vbo = self._ctx.buffer(np.ascontiguousarray(car_verts).tobytes())
            self._car_vao = self._ctx.vertex_array(
                self._fp_prog_car,
                [(self._car_vbo, '3f 3f 4f', 'in_pos', 'in_normal', 'in_color')])
            self._car_vert_count = len(car_verts)

        self._fp_geo_key = None  # force geometry rebuild
        # Opt 5: Initialise physics thread with car state
        pt = self._physics_thread
        pt.set_car_pos(self._car_wx, self._car_wy, self._car_yaw, 0.0, 0.0)
        pt.car_max_speed = self._car_max_speed
        pt.car_accel_rate = self._car_accel_rate
        pt.car_mpp = self._car_mpp
        pt.car_mode = True
        pt._active = True
        self._gl_dirty = True; self.update()

    def _exit_car_mode(self):
        """Exit car, go back to walking at the car's position."""
        self._car_mode = False
        self._car_speed = 0.0
        self._car_steer_angle = 0.0
        self._fp_cx = self._car_wx
        self._fp_cy = self._car_wy
        self._fp_yaw = self._car_yaw
        mpp = self._car_mpp
        self._fp_eye_height = 1.7 / mpp
        self._fp_move_speed = 30.0 / mpp
        # Opt 5: Sync physics thread back to walk mode
        pt = self._physics_thread
        pt.car_mode = False
        pt.set_walk_pos(self._fp_cx, self._fp_cy, self._fp_yaw,
                        self._fp_pitch_angle, self._fp_eye_height)
        pt.move_speed = self._fp_move_speed
        for a in ('_car_vbo', '_car_vao'):
            o = getattr(self, a)
            if o:
                try: o.release()
                except: pass
                setattr(self, a, None)
        self._car_vert_count = 0
        # Clean up traffic
        self._traffic_cars.clear()
        self._traffic_spawn_key = None
        for a in ('_traffic_vbo', '_traffic_vao'):
            o = getattr(self, a)
            if o:
                try: o.release()
                except: pass
                setattr(self, a, None)
        self._traffic_vert_count = 0
        self._traffic_npc_model_cache.clear()
        self._collision_flash = 0.0
        self._collision_cooldown = 0.0
        self._collision_speed_penalty = 0.0
        self._gl_dirty = True; self.update()

    def _tick_car(self):
        """Update car collisions, traffic, and camera follow.
        Physics (speed, steering, position) are now handled by PhysicsThread (Opt 5).
        This method reads car_wx/wy/yaw/speed from self (already synced from thread)
        and handles the main-thread-only work: collision detection, NPC traffic,
        3rd-person camera smoothing."""
        now = time.perf_counter()
        raw_dt = now - self._car_last_tick if self._car_last_tick > 0 else (1.0 / 60.0)
        self._car_last_tick = now
        dt = min(raw_dt, 1.0 / 20.0)  # cap at 50ms
        mpp = self._car_mpp

        # --- Collision cooldown ---
        if self._collision_cooldown > 0:
            self._collision_cooldown -= dt
        if self._collision_flash > 0:
            self._collision_flash = max(0, self._collision_flash - dt * 3.0)
        if self._collision_speed_penalty > 0:
            self._collision_speed_penalty = max(0, self._collision_speed_penalty - dt * 0.5)

        # --- Building collision detection (main thread only — accesses building cache) ---
        if self._collision_cooldown <= 0:
            collided = self._check_building_collision(self._car_wx, self._car_wy)
            if collided:
                self._collision_flash = 1.0
                self._collision_cooldown = 0.5
                self._collision_speed_penalty = min(1.0, self._collision_speed_penalty + 0.3)
                # Bounce: modify physics thread state
                yaw_rad = math.radians(self._car_yaw)
                self._car_speed *= -0.3
                self._car_wx -= math.sin(yaw_rad) * abs(self._car_speed) * dt * 2
                self._car_wy += math.cos(yaw_rad) * abs(self._car_speed) * dt * 2
                self._car_speed = abs(self._car_speed) * 0.1
                # Push corrected state back to physics thread
                pt = self._physics_thread
                pt.set_car_pos(self._car_wx, self._car_wy,
                               self._car_yaw, self._car_speed, self._car_steer_angle)

        # --- Traffic NPC collision ---
        if self._traffic_enabled and self._collision_cooldown <= 0:
            car_hw = 2.2 / max(mpp, 0.001)  # half-width in world px
            for npc in self._traffic_cars:
                if not npc.alive:
                    continue
                dx = npc.wx - self._car_wx
                dy = npc.wy - self._car_wy
                dist = math.hypot(dx, dy) * mpp
                if dist < 5.0:  # collision within ~5m
                    self._collision_flash = 1.0
                    self._collision_cooldown = 1.0
                    self._collision_speed_penalty = min(1.0, self._collision_speed_penalty + 0.5)
                    # Kill NPC, slow player
                    npc.alive = False
                    self._traffic_rebuild_needed = True
                    self._car_speed *= 0.1
                    break

        # --- Traffic NPC spawning & updating ---
        if self._traffic_enabled:
            z = self._zoom
            z_int = max(0, min(20, int(math.floor(z + 0.5))))
            tile_px = TILE_SIZE * (2.0 ** (z - z_int))
            cam_tx = int(math.floor(self._car_wx / tile_px))
            cam_ty = int(math.floor(self._car_wy / tile_px))
            spawn_key = (z_int, cam_tx, cam_ty)

            if spawn_key != self._traffic_spawn_key:
                self._traffic_spawn_key = spawn_key
                n = 2 ** z_int
                visible_mvt_keys = []
                with self._mvt_lock:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            tkey = (z_int, (cam_tx + dx) % n, cam_ty + dy)
                            if tkey in self._mvt_cache:
                                visible_mvt_keys.append(tkey)
                    mvt_snap = {k: self._mvt_cache[k] for k in visible_mvt_keys if k in self._mvt_cache}
                self._traffic_road_paths = _extract_road_paths(
                    mvt_snap, visible_mvt_keys, tile_px, mpp)
                self._traffic_cars = _spawn_npc_cars(
                    self._traffic_road_paths, mpp, max_cars=40,
                    rng_seed=hash(spawn_key))
                self._traffic_rebuild_needed = True

            # Update NPC positions
            _tick_npc_cars(self._traffic_cars, dt, mpp,
                          self._car_wx, self._car_wy, 200.0)
            # Remove dead cars
            alive_before = len(self._traffic_cars)
            self._traffic_cars = [c for c in self._traffic_cars if c.alive]
            if len(self._traffic_cars) != alive_before:
                self._traffic_rebuild_needed = True

            # Rebuild traffic VBO periodically
            if self._traffic_rebuild_needed or self._frame % 3 == 0:
                self._rebuild_traffic_vbo()

        # Tight 3rd-person camera follow — exponential smoothing (frame-rate independent)
        cam_yaw_rad = math.radians(self._car_yaw + self._car_cam_yaw_off)
        target_cx = self._car_wx - math.sin(cam_yaw_rad) * self._car_cam_dist
        target_cy = self._car_wy + math.cos(cam_yaw_rad) * self._car_cam_dist
        cam_lerp = 1.0 - math.exp(-16.0 * dt)  # ~95% in 0.19s
        self._fp_cx += (target_cx - self._fp_cx) * cam_lerp
        self._fp_cy += (target_cy - self._fp_cy) * cam_lerp
        self._fp_yaw = self._car_yaw + self._car_cam_yaw_off
        self._fp_pitch_angle = -12.0

        # Request terrain tiles around car + ahead of movement direction
        if self._frame % 15 == 0:
            self._request_terrain_around(self._car_wx, self._car_wy, radius_tiles=2)
            if self._car_speed > 0.1:
                self._prefetch_terrain_ahead(self._car_wx, self._car_wy, self._car_yaw, ahead_tiles=4)

        # Keep map tiles loading around car
        self._cx = self._car_wx
        self._cy = self._car_wy
        self._gl_dirty = True; self.update()

    def _check_building_collision(self, wx, wy):
        """Check if position (wx, wy) is inside any building footprint."""
        z = self._zoom
        z_int = max(0, min(20, int(math.floor(z + 0.5))))
        tile_px = TILE_SIZE * (2.0 ** (z - z_int))
        tx = int(math.floor(wx / tile_px))
        ty = int(math.floor(wy / tile_px))
        n = 2 ** z_int
        mpp = self._car_mpp
        car_radius = 2.5 / max(mpp, 0.001)  # ~2.5m collision radius

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tkey = (z_int, (tx + dx) % n, ty + dy)
                bldgs = self._building_cache.get(tkey)
                if not bldgs:
                    continue
                tile_ox = float(((tx + dx) % n) * tile_px)
                tile_oy = float((ty + dy) * tile_px)
                for bld in bldgs[:100]:  # check up to 100 buildings
                    ext = bld["extent"]
                    s_px = tile_px / ext
                    for ring in bld["rings"]:
                        if len(ring) < 3:
                            continue
                        # Quick AABB check
                        pts = [(tile_ox + px * s_px, tile_oy + py * s_px) for px, py in ring]
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        if wx < min(xs) - car_radius or wx > max(xs) + car_radius:
                            continue
                        if wy < min(ys) - car_radius or wy > max(ys) + car_radius:
                            continue
                        # Point-in-polygon (ray casting)
                        inside = False
                        n_pts = len(pts)
                        j = n_pts - 1
                        for i in range(n_pts):
                            if ((pts[i][1] > wy) != (pts[j][1] > wy) and
                                wx < (pts[j][0] - pts[i][0]) * (wy - pts[i][1]) /
                                     (pts[j][1] - pts[i][1]) + pts[i][0]):
                                inside = not inside
                            j = i
                        if inside:
                            return True
        return False

    def _rebuild_traffic_vbo(self):
        """Rebuild the combined VBO for all alive NPC cars."""
        self._traffic_rebuild_needed = False
        mpp = self._car_mpp
        car_scale = 1.0 / max(mpp, 0.001)

        all_verts = []
        for npc in self._traffic_cars:
            if not npc.alive:
                continue
            # Get or build NPC model for this color
            if npc.color_idx not in self._traffic_npc_model_cache:
                self._traffic_npc_model_cache[npc.color_idx] = _build_npc_car_model(
                    car_scale, npc.color_idx)
            model = self._traffic_npc_model_cache[npc.color_idx]
            if len(model) == 0:
                continue
            # Transform: rotate by yaw + 180 (same flip as player), then translate
            yaw_rad = math.radians(npc.yaw) + math.pi
            cos_y = math.cos(yaw_rad); sin_y = math.sin(yaw_rad)
            # Each vert is (x,y,z, nx,ny,nz, r,g,b,a) — 10 floats
            transformed = model.copy()
            # Rotate position
            ox = transformed[:, 0] * cos_y - transformed[:, 1] * sin_y
            oy = transformed[:, 0] * sin_y + transformed[:, 1] * cos_y
            transformed[:, 0] = ox + (npc.wx - self._car_wx)
            transformed[:, 1] = oy + (npc.wy - self._car_wy)
            # Rotate normals
            nx = transformed[:, 3] * cos_y - transformed[:, 4] * sin_y
            ny = transformed[:, 3] * sin_y + transformed[:, 4] * cos_y
            transformed[:, 3] = nx; transformed[:, 4] = ny
            # Brake light boost (brighten tail lights when braking)
            if npc.braking:
                # Identify taillights by red channel > 0.5 and green < 0.2
                tl_mask = (model[:, 6] > 0.5) & (model[:, 7] < 0.2)
                transformed[tl_mask, 6] = 1.0  # bright red
                transformed[tl_mask, 7] = 0.2
                transformed[tl_mask, 8] = 0.1
            all_verts.append(transformed)

        for a in ('_traffic_vbo', '_traffic_vao'):
            o = getattr(self, a)
            if o:
                try: o.release()
                except: pass
                setattr(self, a, None)
        self._traffic_vert_count = 0

        if all_verts:
            combined = np.vstack(all_verts).astype('f4')
            if len(combined) > 0:
                self._traffic_vbo = self._ctx.buffer(
                    np.ascontiguousarray(combined).tobytes())
                self._traffic_vao = self._ctx.vertex_array(
                    self._fp_prog_car,
                    [(self._traffic_vbo, '3f 3f 4f', 'in_pos', 'in_normal', 'in_color')])
                self._traffic_vert_count = len(combined)

    def mousePressEvent(self, event):
        # Car mode: mouse for camera orbit
        if self._immersive and self._car_mode:
            if event.button() == Qt.LeftButton:
                self._fp_mouse_captured = True
                self._fp_mouse_last = event.pos()
                self.setCursor(Qt.BlankCursor)
            event.accept(); return

        # Immersive entering: click to place camera
        if self._immersive_entering and event.button() == Qt.LeftButton:
            wx, wy = self._screen_to_world(event.pos().x(), event.pos().y())
            self._enter_immersive(wx, wy)
            event.accept(); return

        # In immersive mode, capture mouse for look
        if self._immersive:
            if event.button() == Qt.LeftButton:
                self._fp_mouse_captured = True
                self._fp_mouse_last = event.pos()
                self.setCursor(Qt.BlankCursor)
            event.accept(); return

        if event.button() == Qt.LeftButton:
            # Cancel any active animation when user grabs the map
            if self._fly.active:
                self._fly.cancel()
            self._scroll_zoom_active = False
            self._target_zoom = self._zoom
            self._drag_start = event.pos(); self._drag_last = event.pos()
            self._drag_cx0 = self._cx; self._drag_cy0 = self._cy
            self._drag_lat0 = _wy_to_lat(self._cy, self._zoom)
            self._drag_lon0 = _wx_to_lon(self._cx, self._zoom)
            self._drag_zoom0 = self._zoom
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif event.button() == Qt.RightButton:
            if self._fly.active:
                self._fly.cancel()
            self._scroll_zoom_active = False
            self._target_zoom = self._zoom
            self._rdrag_start = event.pos(); self._rdrag_pitch0 = self._pitch
            self._rdrag_bearing0 = self._bearing
            self._rdrag_cx0 = self._cx; self._rdrag_cy0 = self._cy
            self._rdrag_lat0 = _wy_to_lat(self._cy, self._zoom)
            self._rdrag_lon0 = _wx_to_lon(self._cx, self._zoom)
        elif event.button() == Qt.MiddleButton:
            if self._fly.active:
                self._fly.cancel()
            self._scroll_zoom_active = False
            self._target_zoom = self._zoom
            self._mdrag_start = event.pos(); self._mdrag_bearing0 = self._bearing
            self._mdrag_cx0 = self._cx; self._mdrag_cy0 = self._cy
            self._mdrag_lat0 = _wy_to_lat(self._cy, self._zoom)
            self._mdrag_lon0 = _wx_to_lon(self._cx, self._zoom)

    def mouseMoveEvent(self, event):
        # Car mode: orbit camera around car
        if self._immersive and self._car_mode and self._fp_mouse_captured:
            if self._fp_mouse_last is not None:
                d = event.pos() - self._fp_mouse_last
                self._car_cam_yaw_off = max(-90.0, min(90.0,
                    self._car_cam_yaw_off + d.x() * 0.3))
                self._gl_dirty = True; self.update()
            self._fp_mouse_last = event.pos()
            center = QPoint(self.width() // 2, self.height() // 2)
            if abs(event.pos().x() - center.x()) > 100 or abs(event.pos().y() - center.y()) > 100:
                self._fp_mouse_last = center
                QCursor.setPos(self.mapToGlobal(center))
            return

        # Immersive mode: mouse look
        if self._immersive and self._fp_mouse_captured:
            if self._fp_mouse_last is not None:
                d = event.pos() - self._fp_mouse_last
                self._fp_yaw = (self._fp_yaw + d.x() * 0.25) % 360.0
                self._fp_pitch_angle = max(-80.0, min(80.0, self._fp_pitch_angle - d.y() * 0.20))
                # Opt 5: Sync look direction to physics thread
                pt = self._physics_thread
                with pt._lock:
                    pt.walk_yaw = self._fp_yaw
                    pt.walk_pitch = self._fp_pitch_angle
                self._gl_dirty = True; self.update()
            self._fp_mouse_last = event.pos()
            # Warp cursor to center to enable infinite look
            center = QPoint(self.width() // 2, self.height() // 2)
            if abs(event.pos().x() - center.x()) > 100 or abs(event.pos().y() - center.y()) > 100:
                self._fp_mouse_last = center
                QCursor.setPos(self.mapToGlobal(center))
            return

        if self._drag_start is not None:
            # Recompute anchor in current zoom's world-pixels (zoom-safe)
            cx0 = _lon_to_wx(self._drag_lon0, self._zoom)
            cy0 = _lat_to_wy(self._drag_lat0, self._zoom)
            d = event.pos() - self._drag_start
            dx_s, dy_s = float(d.x()), float(d.y())
            b_r = math.radians(self._bearing)
            cb, sb = math.cos(b_r), math.sin(b_r)
            self._cx = cx0 - (dx_s * cb - dy_s * sb)
            self._cy = cy0 - (dx_s * sb + dy_s * cb)
            self._drag_last = event.pos()
            self._gl_dirty = True; self.update()

        if self._rdrag_start is not None:
            # Recompute anchor in current zoom's world-pixels (zoom-safe)
            rdrag_cx0 = _lon_to_wx(self._rdrag_lon0, self._zoom)
            rdrag_cy0 = _lat_to_wy(self._rdrag_lat0, self._zoom)
            d = event.pos() - self._rdrag_start
            max_p = self._get_panel_val("max_pitch", 60.0)
            new_pitch = max(0.0, min(max_p, self._rdrag_pitch0 - d.y() * 0.3))
            new_bearing = _normalize_bearing(self._rdrag_bearing0 + d.x() * 0.4)
            if self._rdrag_pitch0 > 1.0:
                p0r = math.radians(self._rdrag_pitch0)
                look_dist = self.height() * 0.35 * math.sin(p0r)
                b0r = math.radians(self._rdrag_bearing0)
                gx = rdrag_cx0 - math.sin(b0r) * look_dist
                gy = rdrag_cy0 + math.cos(b0r) * look_dist
                bnr = math.radians(new_bearing)
                self._cx = gx + math.sin(bnr) * look_dist
                self._cy = gy - math.cos(bnr) * look_dist
            self._pitch = new_pitch; self._target_pitch = new_pitch
            self._bearing = new_bearing; self._target_bearing = new_bearing
            self._gl_dirty = True; self.update()

        if self._mdrag_start is not None:
            # Recompute anchor in current zoom's world-pixels (zoom-safe)
            mdrag_cx0 = _lon_to_wx(self._mdrag_lon0, self._zoom)
            mdrag_cy0 = _lat_to_wy(self._mdrag_lat0, self._zoom)
            d = event.pos() - self._mdrag_start
            new_bearing = _normalize_bearing(self._mdrag_bearing0 + d.x() * 0.5)
            if self._pitch > 1.0:
                pr = math.radians(self._pitch)
                look_dist = self.height() * 0.35 * math.sin(pr)
                b0r = math.radians(self._mdrag_bearing0)
                gx = mdrag_cx0 - math.sin(b0r) * look_dist
                gy = mdrag_cy0 + math.cos(b0r) * look_dist
                bnr = math.radians(new_bearing)
                self._cx = gx + math.sin(bnr) * look_dist
                self._cy = gy - math.cos(bnr) * look_dist
            self._bearing = new_bearing; self._target_bearing = new_bearing
            self._gl_dirty = True; self.update()

    def mouseReleaseEvent(self, event):
        if self._immersive:
            event.accept(); return
        if event.button() == Qt.LeftButton:
            # --- Isochrone placement mode ---
            if self._iso_placing and self._drag_start is not None:
                d = event.pos() - self._drag_start
                if abs(d.x()) < 5 and abs(d.y()) < 5:
                    wx, wy = self._screen_to_world(event.pos().x(), event.pos().y())
                    lat = _wy_to_lat(wy, self._zoom)
                    lon = _wx_to_lon(wx, self._zoom)
                    self._iso_place_at(lat, lon)
                self._drag_start = None; self._drag_last = None
                event.accept(); return

            if self._route_mode and self._drag_start is not None:
                d = event.pos() - self._drag_start
                if abs(d.x()) < 4 and abs(d.y()) < 4:
                    wx, wy = self._screen_to_world(event.pos().x(), event.pos().y())
                    lat = _wy_to_lat(wy, self._zoom)
                    lon = _wx_to_lon(wx, self._zoom)
                    self._route_points.append((lat, lon))
                    if len(self._route_points) >= 2:
                        self._request_directions()
                    else:
                        self._route_world = [
                            (_lon_to_wx(lo, self._zoom), _lat_to_wy(la, self._zoom))
                            for la, lo in self._route_points
                        ]
                    self._route_dirty = True
                    self._gl_dirty = True; self.update()
            elif not self._route_mode and self._drag_start is not None:
                # Check for click (not drag) — try POI hit test
                d = event.pos() - self._drag_start
                if abs(d.x()) < 5 and abs(d.y()) < 5:
                    hit_poi = self._poi_hit_test_2d(event.pos().x(), event.pos().y())
                    if hit_poi:
                        # Shift-click: open web search directly
                        if event.modifiers() & Qt.ShiftModifier:
                            self._poi_show_popup(hit_poi, event.pos().x(), event.pos().y())
                            self._poi_search_web(hit_poi)
                        # Ctrl-click: open Wikipedia page
                        elif event.modifiers() & Qt.ControlModifier:
                            self._poi_show_popup(hit_poi, event.pos().x(), event.pos().y())
                            self._poi_open_wiki(hit_poi)
                        else:
                            self._poi_show_popup(hit_poi, event.pos().x(), event.pos().y())
                    elif self._poi_popup is not None:
                        self._poi_dismiss_popup()
            self._drag_start = None; self._drag_last = None
            if self._iso_placing:
                self.setCursor(QCursor(Qt.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.CrossCursor if self._route_mode else Qt.OpenHandCursor))
        elif event.button() == Qt.RightButton:
            self._rdrag_start = None
        elif event.button() == Qt.MiddleButton:
            self._mdrag_start = None

    def mouseDoubleClickEvent(self, event):
        """Double-click on a POI marker opens it in Google Maps."""
        if self._immersive:
            event.accept(); return
        if event.button() == Qt.LeftButton:
            hit_poi = self._poi_hit_test_2d(event.pos().x(), event.pos().y())
            if hit_poi:
                self._poi_show_popup(hit_poi, event.pos().x(), event.pos().y())
                self._poi_open_web(hit_poi)
                event.accept(); return
        # Fall through to default
        event.accept()

    def wheelEvent(self, event):
        if self._immersive:
            delta = event.angleDelta().y()
            factor = 1.2 if delta > 0 else 0.83
            self._fp_move_speed = max(0.5, min(6000.0, self._fp_move_speed * factor))
            event.accept(); return

        # Cancel flyTo on manual zoom
        if self._fly.active:
            self._fly.cancel()

        delta = event.angleDelta().y()
        step = delta / 120.0 * 0.5

        # Compute the bearing-rotated screen offset from center
        cur = event.position()
        cx_off = cur.x() - self.width() / 2
        cy_off = cur.y() - self.height() / 2
        b_r = math.radians(self._bearing)
        cb, sb = math.cos(b_r), math.sin(b_r)
        sx = cx_off * cb - cy_off * sb
        sy = cx_off * sb + cy_off * cb

        # World-pixel position under cursor at current zoom
        wx_cur = self._cx + sx
        wy_cur = self._cy + sy

        # Convert to lat/lon (zoom-invariant anchor)
        anchor_lat = _wy_to_lat(wy_cur, self._zoom)
        anchor_lon = _wx_to_lon(wx_cur, self._zoom)

        # Store anchor and screen offset for smooth animation
        self._scroll_anchor_lat = anchor_lat
        self._scroll_anchor_lon = anchor_lon
        self._scroll_anchor_sx = sx
        self._scroll_anchor_sy = sy

        # Accumulate zoom target
        if not self._scroll_zoom_active:
            self._scroll_zoom_target = self._zoom
            self._scroll_zoom_active = True

        self._scroll_zoom_target = _clamp_zoom(self._scroll_zoom_target + step)
        self._target_zoom = self._scroll_zoom_target
        self._gl_dirty = True; self.update(); event.accept()

    def keyPressEvent(self, event):
        k = event.key()

        # --- Search bar intercepts ALL keys when active ---
        if self._search_active:
            if self._search_handle_key(event):
                event.accept(); return
            # If not consumed, still block most keys
            event.accept(); return

        # --- "/" key opens search bar (works in all modes) ---
        if k == Qt.Key_Slash:
            self._search_open()
            event.accept(); return

        # --- Immersive mode key handling ---
        if self._immersive:
            # --- POI selection cycling (works in both car and walking mode) ---
            if self._poi_enabled:
                if k in (Qt.Key_Down, Qt.Key_Minus):
                    self._poi_select_next()
                    event.accept(); return
                if k in (Qt.Key_Up, Qt.Key_Plus, Qt.Key_Equal):
                    self._poi_select_prev()
                    event.accept(); return
                if k == Qt.Key_Return or k == Qt.Key_Enter:
                    if self._poi_select_active and self._poi_select_idx >= 0:
                        self._poi_navigate_to_selected()
                        event.accept(); return

            # --- Car mode keys ---
            if self._car_mode:
                # POI popup actions (E=map, X=web, Esc=dismiss if popup shown)
                if self._poi_popup is not None:
                    if k == Qt.Key_E:
                        self._poi_open_web()
                        event.accept(); return
                    if k == Qt.Key_X:
                        self._poi_search_web()
                        event.accept(); return
                    if k == Qt.Key_J:
                        self._poi_open_wiki()
                        event.accept(); return
                    if k == Qt.Key_Escape:
                        self._poi_dismiss_popup()
                        event.accept(); return
                if k == Qt.Key_W:
                    self._car_throttle = 1.0; event.accept(); return
                if k == Qt.Key_S:
                    self._car_brake = 1.0; event.accept(); return
                if k == Qt.Key_A:
                    self._car_steer = -1.0; event.accept(); return
                if k == Qt.Key_D:
                    self._car_steer = 1.0; event.accept(); return
                if k == Qt.Key_Space:
                    self._car_brake = 1.0; event.accept(); return
                if k == Qt.Key_G:
                    self._exit_car_mode(); event.accept(); return
                if k == Qt.Key_Escape:
                    self._exit_car_mode(); self._exit_immersive()
                    event.accept(); return
                if k == Qt.Key_V:
                    self._tiles_visible = not self._tiles_visible
                    if self._panel: self._panel.set_check("tiles_visible", self._tiles_visible)
                    self._fp_geo_key = None
                    self._gl_dirty = True; self.update()
                if k == Qt.Key_P:
                    self._traffic_enabled = not self._traffic_enabled
                    if self._traffic_enabled:
                        self._traffic_spawn_key = None  # force respawn
                    else:
                        self._traffic_cars.clear()
                        self._traffic_rebuild_needed = True
                    self._gl_dirty = True; self.update()
                if k == Qt.Key_O:
                    self._poi_enabled = not self._poi_enabled
                    if self._panel: self._panel.set_check("poi_markers", self._poi_enabled)
                    self._poi_cache_key = None
                    self._fp_poi_rebuild = True
                    self._gl_dirty = True; self.update()
                if k == Qt.Key_Y:
                    self._iso_toggle()
                event.accept(); return

            # --- Normal immersive (walking) keys ---
            # POI popup actions (intercept before WASD)
            if self._poi_popup is not None:
                if k == Qt.Key_E:
                    self._poi_open_web()
                    event.accept(); return
                if k == Qt.Key_X:
                    self._poi_search_web()
                    event.accept(); return
                if k == Qt.Key_J:
                    self._poi_open_wiki()
                    event.accept(); return
                if k == Qt.Key_Escape:
                    self._poi_dismiss_popup()
                    event.accept(); return
            if k in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Space, Qt.Key_Shift):
                self._fp_keys_held.add(k)
                event.accept(); return
            if k == Qt.Key_Escape:
                self._exit_immersive()
                event.accept(); return
            if k == Qt.Key_Q:
                self._fp_yaw = (self._fp_yaw - 3.0) % 360.0
                self._gl_dirty = True; self.update()
            if k == Qt.Key_V:
                self._tiles_visible = not self._tiles_visible
                if self._panel: self._panel.set_check("tiles_visible", self._tiles_visible)
                self._fp_geo_key = None  # force rebuild
                self._gl_dirty = True; self.update()
            if k == Qt.Key_G:
                self._enter_car_mode()
                event.accept(); return
            if k == Qt.Key_P:
                self._traffic_enabled = not self._traffic_enabled
                if self._traffic_enabled:
                    self._traffic_spawn_key = None
                else:
                    self._traffic_cars.clear()
                    self._traffic_rebuild_needed = True
                self._gl_dirty = True; self.update()
            if k == Qt.Key_O:
                self._poi_enabled = not self._poi_enabled
                if self._panel: self._panel.set_check("poi_markers", self._poi_enabled)
                self._poi_cache_key = None
                self._fp_poi_rebuild = True
                self._gl_dirty = True; self.update()
            if k == Qt.Key_Y:
                self._iso_toggle()
            event.accept(); return

        # Cancel isochrone placement
        if self._iso_placing:
            if k == Qt.Key_Escape:
                self._iso_placing = False
                self.setCursor(QCursor(Qt.OpenHandCursor))
                self._gl_dirty = True; self.update()
                event.accept(); return

        if self._immersive_entering:
            if k == Qt.Key_Escape:
                self._immersive_entering = False
                self.setCursor(QCursor(Qt.OpenHandCursor))
                self._gl_dirty = True; self.update()
                event.accept(); return

        if k == Qt.Key_I:
            if not self._immersive and not self._immersive_entering:
                self._immersive_entering = True
                self.setCursor(QCursor(Qt.CrossCursor))
                self._gl_dirty = True; self.update()
            elif self._immersive:
                self._exit_immersive()
            event.accept(); return

        if k in (Qt.Key_Plus, Qt.Key_Equal):
            cur_lat = _wy_to_lat(self._cy, self._zoom)
            cur_lon = _wx_to_lon(self._cx, self._zoom)
            self.easeTo(cur_lat, cur_lon, zoom=_clamp_zoom(self._zoom + 1), duration=0.3)
        elif k == Qt.Key_Minus:
            cur_lat = _wy_to_lat(self._cy, self._zoom)
            cur_lon = _wx_to_lon(self._cx, self._zoom)
            self.easeTo(cur_lat, cur_lon, zoom=_clamp_zoom(self._zoom - 1), duration=0.3)
        elif k == Qt.Key_S:
            old_tileset = self._current_tileset
            self._style_idx = (self._style_idx + 1) % len(self.STYLES)
            self._style = self.STYLES[self._style_idx]
            new_tileset = self._current_tileset
            if new_tileset != old_tileset:
                with self._mvt_lock:
                    self._mvt_cache.clear()
                self._invalidate_visible()
                QTimer.singleShot(50, self._preload_base_tiles)
            else:
                self._invalidate_visible()
        elif k == Qt.Key_B:
            self._buildings_3d = not self._buildings_3d
            if self._panel: self._panel.set_check("buildings_3d", self._buildings_3d)
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_T:
            self._target_pitch = 0.0; self._target_bearing = 0.0
        elif k == Qt.Key_F:
            self._labels_visible = not self._labels_visible
            if self._panel: self._panel.set_check("labels_visible", self._labels_visible)
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_R:
            self.flyTo(37.7749, -122.4194, zoom=13.0, bearing=0.0, pitch=0.0)
        elif k == Qt.Key_N:
            self._route_mode = not self._route_mode
            self.setCursor(QCursor(Qt.CrossCursor if self._route_mode else Qt.OpenHandCursor))
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_C:
            self._route_points.clear()
            self._route_world.clear()
            self._route_geometry.clear()
            self._route_dirty = True
            self._route_total_dist = 0.0
            self._route_duration = 0.0
            if self._route_reply is not None:
                try: self._route_reply.abort(); self._route_reply.deleteLater()
                except: pass
                self._route_reply = None
            self._route_pending = False
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_M:
            profiles = ["driving", "walking", "cycling"]
            idx = profiles.index(self._route_profile) if self._route_profile in profiles else 0
            self._route_profile = profiles[(idx + 1) % len(profiles)]
            if len(self._route_points) >= 2:
                self._request_directions()
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_L:
            self._shadows_enabled = not self._shadows_enabled
            if self._panel: self._panel.set_check("shadows", self._shadows_enabled)
            self._shadow_gpu_key = ()
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_H:
            if not self._heatmap_enabled:
                # First press: enable with first mode
                self._heatmap_mode_idx = 0
                self._heatmap_mode = self._heatmap_modes[0]
                self._heatmap_enabled = True
                self._heatmap_generated = False
                self._generate_heatmap_data()
            else:
                # Cycle through modes, disable after last mode
                self._heatmap_mode_idx += 1
                if self._heatmap_mode_idx >= len(self._heatmap_modes):
                    self._heatmap_mode_idx = 0
                    self._heatmap_enabled = False
                    self._heatmap_generated = False
                else:
                    self._heatmap_mode = self._heatmap_modes[self._heatmap_mode_idx]
                    self._heatmap_generated = False
                    self._generate_heatmap_data()
            self._heatmap_dirty = True
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_V:
            self._tiles_visible = not self._tiles_visible
            if self._panel: self._panel.set_check("tiles_visible", self._tiles_visible)
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_O:
            self._poi_enabled = not self._poi_enabled
            if self._panel: self._panel.set_check("poi_markers", self._poi_enabled)
            self._poi_cache_key = None  # force rebuild
            self._poi_2d_dirty = True
            self._fp_poi_rebuild = True
            self._gl_dirty = True; self.update()
        elif k == Qt.Key_Y:
            self._iso_toggle()
        event.accept()


# ===========================================================================
#  Drop-in integration
# ===========================================================================

try:
    graphics_scene.removeItem(map_proxy)
except:
    pass

try:
    graphics_scene.removeItem(panel_proxy)
except:
    pass

try:
    graphics_scene.removeItem(token_proxy)
except:
    pass

# --- Token discovery: .env file → environment variable → prompt ---
def _find_mapbox_token():
    """Try to load token from .env file or environment."""
    # 1) Try .env file in current directory and common locations
    for env_path in [".env", os.path.expanduser("~/.env"),
                     os.path.join(os.path.dirname(__file__), ".env") if '__file__' in dir() else None]:
        if env_path and os.path.isfile(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k == "MAPBOX_TOKEN" and v and not v.startswith("pk.your_"):
                            return v
            except Exception:
                pass
    # 2) Environment variable
    val = os.environ.get("MAPBOX_TOKEN", "")
    if val and not val.startswith("pk.your_"):
        return val
    return ""

TOKEN = _find_mapbox_token()

if TOKEN:
    # --- Normal startup: map + side panel ---
    side_panel = SettingsPanel()
    side_panel.setFixedHeight(820)

    map_widget = MapboxWidget(
        token  = TOKEN,
        style  = "mapbox/dark-v10",
        lat    = 37.7749,
        lon    = -122.4194,
        zoom   = 13.0,
        width  = 900,
        height = 600,
    )
    map_widget.set_panel(side_panel)

    panel_proxy = graphics_scene.addWidget(side_panel)
    panel_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)

    map_proxy = graphics_scene.addWidget(map_widget)
    map_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)

    view = graphics_scene.views()[0]
    vr   = view.viewport().rect()
    sr   = view.mapToScene(vr).boundingRect()
    total_w = 260 + 900
    panel_proxy.setPos(sr.center().x() - total_w / 2, sr.center().y() - 300)
    map_proxy.setPos(sr.center().x() - total_w / 2 + 260, sr.center().y() - 300)

else:
    # --- No token: show a minimal input prompt ---
    class _TokenPrompt(QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedSize(480, 200)
            self.setStyleSheet(
                "QFrame { background: #1a1a2e; border: 1px solid #333; border-radius: 12px; }"
                "QLabel { color: #ccc; border: none; }"
                "QLineEdit { background: #0f0f1a; color: #eee; border: 1px solid #555;"
                "            border-radius: 6px; padding: 8px; font-size: 13px; }"
                "QLineEdit:focus { border-color: #4a9eff; }"
            )
            layout = QVBoxLayout(self)
            layout.setContentsMargins(28, 24, 28, 24)
            layout.setSpacing(12)

            title = QLabel("Mapbox Token Required")
            title.setFont(QFont("sans-serif", 15, QFont.Bold))
            title.setStyleSheet("color: #e0e0e0;")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)

            msg = QLabel(
                "No MAPBOX_TOKEN found in .env or environment.\n"
                "Paste your token below and press Enter."
            )
            msg.setFont(QFont("sans-serif", 10))
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            layout.addWidget(msg)

            self._input = QLineEdit()
            self._input.setPlaceholderText("pk.eyJ1Ijoi...")
            self._input.returnPressed.connect(self._on_submit)
            layout.addWidget(self._input)

            self._error = QLabel("")
            self._error.setFont(QFont("sans-serif", 9))
            self._error.setStyleSheet("color: #ff6666;")
            self._error.setAlignment(Qt.AlignCenter)
            layout.addWidget(self._error)

        def _on_submit(self):
            token = self._input.text().strip()
            if not token or not token.startswith("pk."):
                self._error.setText("Token must start with 'pk.' — try again.")
                return
            self._error.setText("")

            # Remove prompt, launch the full app
            try:
                graphics_scene.removeItem(token_proxy)
            except:
                pass

            side_panel = SettingsPanel()
            side_panel.setFixedHeight(820)

            map_widget = MapboxWidget(
                token  = token,
                style  = "mapbox/dark-v10",
                lat    = 37.7749,
                lon    = -122.4194,
                zoom   = 13.0,
                width  = 900,
                height = 600,
            )
            map_widget.set_panel(side_panel)

            pp = graphics_scene.addWidget(side_panel)
            pp.setFlag(QGraphicsItem.ItemIsMovable, True)

            mp = graphics_scene.addWidget(map_widget)
            mp.setFlag(QGraphicsItem.ItemIsMovable, True)

            v = graphics_scene.views()[0]
            vr = v.viewport().rect()
            sr = v.mapToScene(vr).boundingRect()
            total_w = 260 + 900
            pp.setPos(sr.center().x() - total_w / 2, sr.center().y() - 300)
            mp.setPos(sr.center().x() - total_w / 2 + 260, sr.center().y() - 300)

            # Store references globally so they don't get GC'd
            globals()['panel_proxy'] = pp
            globals()['map_proxy'] = mp
            globals()['side_panel'] = side_panel
            globals()['map_widget'] = map_widget

    _prompt = _TokenPrompt()
    token_proxy = graphics_scene.addWidget(_prompt)
    token_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)

    view = graphics_scene.views()[0]
    vr   = view.viewport().rect()
    sr   = view.mapToScene(vr).boundingRect()
    token_proxy.setPos(sr.center().x() - 240, sr.center().y() - 100)
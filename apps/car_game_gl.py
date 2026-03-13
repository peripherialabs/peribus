"""
Car Game — Pure Python / ModernGL / Offscreen FBO
Replaces QWebEngineView + Three.js with native OpenGL rendering.
Renders to an offscreen framebuffer via standalone ModernGL context,
then blits to a plain QWidget — works inside QGraphicsProxyWidget.
Drops into the same parser/scene system via QGraphicsScene.addWidget().

Dependencies: PySide6, moderngl, PyGLM, numpy
"""

import math
import time
import random
import struct
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsItem
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage,
    QLinearGradient
)

import moderngl
import glm

# ─────────────────────────────────────────────
#  Shader sources
# ─────────────────────────────────────────────

VERT_SHADER = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;
out float v_fog;

void main() {
    vec4 world = model * vec4(in_position, 1.0);
    v_world = world.xyz;
    gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = mat3(model) * in_normal;
    v_color = in_color;
    float dist = length(world.xyz - vec3(0.0, 5.0, 0.0));
    v_fog = clamp(dist / 1400.0, 0.0, 1.0);
}
"""

FRAG_SHADER = """
#version 330
uniform vec3 light_dir;
uniform vec3 ambient;
uniform vec3 fog_color;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_world;
in float v_fog;
out vec4 frag;

void main() {
    vec3 n = normalize(v_normal);
    float diff = max(dot(n, normalize(light_dir)), 0.0);
    vec3 col = v_color * (ambient + diff * vec3(1.0, 0.75, 0.5));
    col = mix(col, fog_color, v_fog * v_fog * 0.7);
    frag = vec4(col, 1.0);
}
"""

# Unlit shader for emissive things (lights, sky, windows)
UNLIT_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
out float v_fog;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
    float dist = length(in_position - vec3(0.0, 5.0, 0.0));
    v_fog = clamp(dist / 1400.0, 0.0, 1.0);
}
"""

UNLIT_FRAG = """
#version 330
uniform vec3 fog_color;
in vec3 v_color;
in float v_fog;
out vec4 frag;
void main() {
    vec3 col = mix(v_color, fog_color, v_fog * v_fog * 0.5);
    frag = vec4(col, 1.0);
}
"""

# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def _hex_to_rgb(h):
    return ((h >> 16) & 0xFF) / 255.0, ((h >> 8) & 0xFF) / 255.0, (h & 0xFF) / 255.0

def make_box(w, h, d, color_hex):
    """Generate box vertices: position(3) + normal(3) + color(3)"""
    r, g, b = _hex_to_rgb(color_hex)
    hw, hh, hd = w/2, h/2, d/2
    faces = [
        # front (+Z)
        ((-hw,-hh,hd),(hw,-hh,hd),(hw,hh,hd),(-hw,hh,hd),(0,0,1)),
        # back (-Z)
        ((hw,-hh,-hd),(-hw,-hh,-hd),(-hw,hh,-hd),(hw,hh,-hd),(0,0,-1)),
        # right (+X)
        ((hw,-hh,hd),(hw,-hh,-hd),(hw,hh,-hd),(hw,hh,hd),(1,0,0)),
        # left (-X)
        ((-hw,-hh,-hd),(-hw,-hh,hd),(-hw,hh,hd),(-hw,hh,-hd),(-1,0,0)),
        # top (+Y)
        ((-hw,hh,hd),(hw,hh,hd),(hw,hh,-hd),(-hw,hh,-hd),(0,1,0)),
        # bottom (-Y)
        ((-hw,-hh,-hd),(hw,-hh,-hd),(hw,-hh,hd),(-hw,-hh,hd),(0,-1,0)),
    ]
    verts = []
    for v0,v1,v2,v3,n in faces:
        for v in (v0,v1,v2, v0,v2,v3):
            verts.extend(v)
            verts.extend(n)
            verts.extend((r,g,b))
    return verts

def make_plane(w, d, color_hex):
    """Horizontal plane at y=0"""
    r, g, b = _hex_to_rgb(color_hex)
    hw, hd = w/2, d/2
    verts = []
    for v in ((-hw,0,hd),(hw,0,hd),(hw,0,-hd),(-hw,0,hd),(hw,0,-hd),(-hw,0,-hd)):
        verts.extend(v)
        verts.extend((0,1,0))
        verts.extend((r,g,b))
    return verts

def make_cylinder(radius, height, segs, color_hex):
    """Upright cylinder"""
    r, g, b = _hex_to_rgb(color_hex)
    verts = []
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i+1) / segs
        x0, z0 = math.cos(a0)*radius, math.sin(a0)*radius
        x1, z1 = math.cos(a1)*radius, math.sin(a1)*radius
        nx0, nz0 = math.cos(a0), math.sin(a0)
        nx1, nz1 = math.cos(a1), math.sin(a1)
        # side
        for v in ((x0,0,z0),(x1,0,z1),(x1,height,z1),(x0,0,z0),(x1,height,z1),(x0,height,z0)):
            verts.extend(v)
            verts.extend((nx0,0,nz0))
            verts.extend((r,g,b))
        # top cap
        for v in ((0,height,0),(x0,height,z0),(x1,height,z1)):
            verts.extend(v)
            verts.extend((0,1,0))
            verts.extend((r,g,b))
    return verts

def make_cone(radius, height, segs, color_hex):
    """Upright cone, base at y=0, tip at y=height"""
    r, g, b = _hex_to_rgb(color_hex)
    verts = []
    for i in range(segs):
        a0 = 2*math.pi*i/segs
        a1 = 2*math.pi*(i+1)/segs
        x0, z0 = math.cos(a0)*radius, math.sin(a0)*radius
        x1, z1 = math.cos(a1)*radius, math.sin(a1)*radius
        # side face
        # approximate normal
        ny = radius / math.sqrt(radius*radius + height*height)
        for v in ((0,height,0),(x0,0,z0),(x1,0,z1)):
            verts.extend(v)
            verts.extend((0, ny, 0))  # simplified normal
            verts.extend((r,g,b))
        # base
        for v in ((0,0,0),(x1,0,z1),(x0,0,z0)):
            verts.extend(v)
            verts.extend((0,-1,0))
            verts.extend((r,g,b))
    return verts

def make_sphere_approx(radius, segs_h, segs_v, color_hex):
    """Quick UV sphere"""
    r, g, b = _hex_to_rgb(color_hex)
    verts = []
    for j in range(segs_v):
        t0 = math.pi * j / segs_v
        t1 = math.pi * (j+1) / segs_v
        for i in range(segs_h):
            p0 = 2*math.pi*i/segs_h
            p1 = 2*math.pi*(i+1)/segs_h
            # 4 corners
            def sv(t, p):
                return (radius*math.sin(t)*math.cos(p),
                        radius*math.cos(t),
                        radius*math.sin(t)*math.sin(p))
            v00 = sv(t0,p0); v10 = sv(t1,p0); v11 = sv(t1,p1); v01 = sv(t0,p1)
            def nrm(v):
                l = math.sqrt(v[0]**2+v[1]**2+v[2]**2)+1e-9
                return (v[0]/l,v[1]/l,v[2]/l)
            for v in (v00,v10,v11, v00,v11,v01):
                verts.extend(v)
                verts.extend(nrm(v))
                verts.extend((r,g,b))
    return verts

# ─────────────────────────────────────────────
#  Mesh batching
# ─────────────────────────────────────────────

class MeshBatch:
    """Collects geometry, uploads to GPU as a single VBO."""
    def __init__(self):
        self.verts = []  # flat list of floats
        self.count = 0

    def add(self, geom_verts, transform=None):
        """Add geometry. transform is a glm.mat4 or None."""
        stride = 9  # pos(3) + normal(3) + color(3)
        n = len(geom_verts) // stride
        if transform is not None:
            for i in range(n):
                off = i * stride
                p = glm.vec3(geom_verts[off], geom_verts[off+1], geom_verts[off+2])
                norm = glm.vec3(geom_verts[off+3], geom_verts[off+4], geom_verts[off+5])
                p = glm.vec3(transform * glm.vec4(p, 1.0))
                norm = glm.vec3(glm.mat3(transform) * norm)
                self.verts.extend([p.x, p.y, p.z, norm.x, norm.y, norm.z,
                                   geom_verts[off+6], geom_verts[off+7], geom_verts[off+8]])
        else:
            self.verts.extend(geom_verts)
        self.count += n

    def build(self, ctx, prog):
        if self.count == 0:
            return None
        data = np.array(self.verts, dtype='f4').tobytes()
        vbo = ctx.buffer(data)
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        return vao

class UnlitBatch:
    """For emissive / unlit geometry: pos(3) + color(3)"""
    def __init__(self):
        self.verts = []
        self.count = 0

    def add_tri(self, p0, p1, p2, color_hex):
        r, g, b = _hex_to_rgb(color_hex)
        for p in (p0, p1, p2):
            self.verts.extend(p)
            self.verts.extend((r, g, b))
        self.count += 3

    def add_box(self, w, h, d, color_hex, transform=None):
        r, g, b = _hex_to_rgb(color_hex)
        hw, hh, hd = w/2, h/2, d/2
        corners = [
            (-hw,-hh,hd),(hw,-hh,hd),(hw,hh,hd),(-hw,hh,hd),
            (-hw,-hh,-hd),(hw,-hh,-hd),(hw,hh,-hd),(-hw,hh,-hd)
        ]
        if transform:
            corners = [glm.vec3(transform * glm.vec4(*c, 1.0)) for c in corners]
            corners = [(c.x,c.y,c.z) for c in corners]
        indices = [0,1,2,0,2,3, 5,4,7,5,7,6, 1,5,6,1,6,2, 4,0,3,4,3,7, 3,2,6,3,6,7, 4,5,1,4,1,0]
        for idx in indices:
            self.verts.extend(corners[idx])
            self.verts.extend((r,g,b))
        self.count += len(indices)

    def add_plane(self, w, d, color_hex, transform=None):
        r, g, b = _hex_to_rgb(color_hex)
        hw, hd = w/2, d/2
        pts = [(-hw,0,hd),(hw,0,hd),(hw,0,-hd),(-hw,0,-hd)]
        if transform:
            pts = [glm.vec3(transform * glm.vec4(*p, 1.0)) for p in pts]
            pts = [(p.x,p.y,p.z) for p in pts]
        for v in (pts[0],pts[1],pts[2],pts[0],pts[2],pts[3]):
            self.verts.extend(v)
            self.verts.extend((r,g,b))
        self.count += 6

    def build(self, ctx, prog):
        if self.count == 0:
            return None
        data = np.array(self.verts, dtype='f4').tobytes()
        vbo = ctx.buffer(data)
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
        return vao

# ─────────────────────────────────────────────
#  World builder
# ─────────────────────────────────────────────

RW = 18.0
LANE_W = RW / 4
SW = 4.5

def get_bridge_y(z):
    if 200 <= z <= 350: return ((z-200)/150)*0.5
    if 350 < z <= 1350: return 0.5
    if 1350 < z <= 1500: return 0.5-((z-1350)/150)*0.5
    return 0.0

class WorldData:
    """All static geometry + collider data"""
    def __init__(self):
        self.colliders = []
        self.traffic_light_positions = []

    def build_static(self):
        """Returns (lit_batch, unlit_batch)"""
        lit = MeshBatch()
        unlit = UnlitBatch()

        # Ground
        t = glm.translate(glm.mat4(1), glm.vec3(0, -0.1, 500))
        lit.add(make_plane(700, 4500, 0x141e14), t)

        # Water plane
        t = glm.translate(glm.mat4(1), glm.vec3(0, -5.5, 850))
        lit.add(make_plane(700, 1300, 0x082a4a), t)

        # Roads
        for pos_z, length, elev, color in [(-400,1200,0.02,0x2a2a2a),(275,150,0.02,0x2a2a2a),(850,1000,0.52,0x333333),(1950,1200,0.02,0x2a2a2a)]:
            t = glm.translate(glm.mat4(1), glm.vec3(0, elev, pos_z))
            lit.add(make_plane(RW if color==0x2a2a2a else RW+2, length, color), t)

        # Bridge approach ramps (smooth transition strips)
        for rz in range(200, 355, 5):
            ry = get_bridge_y(rz)
            t = glm.translate(glm.mat4(1), glm.vec3(0, ry, rz))
            lit.add(make_plane(RW+1, 6, 0x2e2e2e), t)

        # Road dashes (center line + lane dividers)
        for z0,z1,elev_fn in [(-1000,350,lambda z: get_bridge_y(z)),(350,1350,lambda z:0.5),(1350,2500,lambda z: get_bridge_y(z))]:
            d = z0
            while d < z1:
                ez = elev_fn(d)
                t = glm.translate(glm.mat4(1), glm.vec3(0, ez+0.03, d))
                unlit.add_plane(0.18, 3.2, 0xccaa44, t)
                for lx in [-LANE_W, LANE_W]:
                    t2 = glm.translate(glm.mat4(1), glm.vec3(lx, ez+0.03, d))
                    unlit.add_plane(0.12, 2.5, 0x999999, t2)
                d += 7

        # Edge lines
        for z0,z1,elev_fn in [(-1000,350,lambda z: get_bridge_y(z)),(350,1350,lambda z:0.5),(1350,2500,lambda z: get_bridge_y(z))]:
            segs = max(1, int((z1-z0)/30))
            for si in range(segs):
                seg_z0 = z0 + si*(z1-z0)/segs
                seg_z1 = z0 + (si+1)*(z1-z0)/segs
                seg_mid = (seg_z0+seg_z1)/2
                seg_elev = elev_fn(seg_mid)
                for s in [-1,1]:
                    t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2-0.3), seg_elev+0.03, seg_mid))
                    unlit.add_plane(0.18, seg_z1-seg_z0, 0x555555, t)

        # Sidewalks + curbs
        for s in [-1,1]:
            for pz, plen in [(-400,1200),(1950,1200)]:
                t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+SW/2), 0.05, pz))
                lit.add(make_plane(SW, plen, 0x3d3030), t)
                t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+0.15), 0.075, pz))
                lit.add(make_box(0.3, 0.15, plen, 0x4a4040), t)
            # approach area sidewalks
            t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+SW/2), 0.05, 275))
            lit.add(make_plane(SW, 150, 0x3d3030), t)

        # Bridge exit ramp (Marin side, z=1350 to 1500)
        for rz in range(1345, 1505, 5):
            ry = get_bridge_y(rz)
            t = glm.translate(glm.mat4(1), glm.vec3(0, ry, rz))
            lit.add(make_plane(RW+1, 6, 0x2e2e2e), t)

        # Bridge structure
        self._build_bridge(lit, unlit)

        # Buildings
        self._build_buildings(lit, unlit)

        # Street lamps (simplified)
        for lz in range(-950, 200, 28):
            for s, xoff in [(1, RW/2+0.8), (-1, -RW/2-0.8)]:
                self._add_lamp(lit, unlit, xoff, lz + (0 if s>0 else 14))
        for lz in range(1420, 2400, 28):
            for s, xoff in [(1, RW/2+0.8), (-1, -RW/2-0.8)]:
                self._add_lamp(lit, unlit, xoff, lz + (0 if s>0 else 14))

        # Trees
        random.seed(42)
        for z0, z1, prob in [(-920,180,0.7), (1440,2380,0.75)]:
            tz = z0
            while tz < z1:
                if random.random() < prob:
                    self._add_tree(lit, RW/2+SW-0.3, tz)
                    self._add_tree(lit, -RW/2-SW+0.3, tz+6)
                tz += 12 + random.random()*7

        # Hills
        for _ in range(14):
            hx = (random.random()-0.5)*350
            hz = 1500 + random.random()*800
            if abs(hx) < 28: continue
            hr = 25 + random.random()*45
            t = glm.translate(glm.mat4(1), glm.vec3(hx, 0, hz))
            t = glm.scale(t, glm.vec3(1, 0.25+random.random()*0.35, 1))
            lit.add(make_sphere_approx(hr, 10, 6, 0x1a2a18), t)

        # Mountains
        for _ in range(6):
            mx = (random.random()-0.5)*500
            mz = 2000 + random.random()*600
            if abs(mx) < 50: mx += math.copysign(50, mx)
            mr = 60 + random.random()*80
            t = glm.translate(glm.mat4(1), glm.vec3(mx, 0, mz))
            lit.add(make_cone(mr, mr*0.6, 8, 0x0f1a12), t)

        # Traffic lights (with housing and colored indicators)
        for tz in [-800,-580,-360,-140,60,1500,1720,1940,2160,2380]:
            for s, fd in [(1,-1),(-1,1)]:
                x = s*(RW/2+1)
                z = tz + (0 if s>0 else 2)
                # Pole
                t = glm.translate(glm.mat4(1), glm.vec3(x, 3.25, z))
                lit.add(make_cylinder(0.1, 6.5, 6, 0x3a3a3a), t)
                # Housing box
                t = glm.translate(glm.mat4(1), glm.vec3(x, 8.2, z))
                lit.add(make_box(0.7, 2.0, 0.7, 0x222222), t)
                # Three indicator spheres (green, yellow, red from bottom)
                for li, ly, lc in [(0, 7.5, 0x00cc44), (1, 8.2, 0xddaa00), (2, 8.9, 0xdd2222)]:
                    t = glm.translate(glm.mat4(1), glm.vec3(x, ly, z + fd*0.36))
                    unlit.add_box(0.28, 0.28, 0.05, lc, t)

                self.traffic_light_positions.append({'x': x, 'z': z, 'dir': fd,
                    'state': random.randint(0,2), 'timer': random.random()*8,
                    'green_dur': 8+random.random()*5, 'yellow_dur': 2.5,
                    'red_dur': 9+random.random()*4})

        return lit, unlit

    def _build_bridge(self, lit, unlit):
        bCol = 0xcc3311
        # Deck
        t = glm.translate(glm.mat4(1), glm.vec3(0, 0, 850))
        lit.add(make_box(RW+4, 0.5, 1000, 0x282828), t)
        # Under-deck
        t = glm.translate(glm.mat4(1), glm.vec3(0, -1, 850))
        lit.add(make_box(RW+6, 1.5, 1000, 0x1a1a1a), t)
        # Railings
        for s in [-1,1]:
            t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), 1.45, 850))
            lit.add(make_box(0.25, 1.9, 1000, bCol), t)
            t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), 2.4, 850))
            lit.add(make_box(0.4, 0.12, 1000, 0xdd4422), t)
        # Towers
        for tz in [600, 1100]:
            for s in [-1,1]:
                t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), 29, tz))
                lit.add(make_box(2.8, 58, 3.2, bCol), t)
                t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), 60, tz))
                lit.add(make_box(2.4, 4, 2.8, 0xdd4422), t)
                t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), 63.5, tz))
                lit.add(make_box(1.8, 3, 2.2, bCol), t)
                # Deco bands
                for y in [14,28,42,55]:
                    t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), y, tz))
                    lit.add(make_box(3.2, 1.4, 3.8, 0x991a08), t)
                # Warning light
                t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), 65.5, tz))
                lit.add(make_sphere_approx(0.4, 6, 4, 0xff2200), t)
            # Cross beams
            for y in [14,28,42,55,60]:
                t = glm.translate(glm.mat4(1), glm.vec3(0, y, tz))
                lit.add(make_box(RW+5, 1.2, 1.8, bCol), t)
            # Arch
            t = glm.translate(glm.mat4(1), glm.vec3(0, 8, tz))
            lit.add(make_box(RW+5, 1, 3, 0x991a08), t)

        # Cables (simplified as thin boxes)
        for s in [-1,1]:
            cx = s*(RW/2+1.5)
            for z1,z2,y1,y2,ymid in [(310,600,10,62,5),(600,1100,62,62,14),(1100,1390,62,10,5)]:
                segs = 20
                for i in range(segs):
                    t0 = i/segs; t1 = (i+1)/segs
                    za = z1+t0*(z2-z1); zb = z1+t1*(z2-z1)
                    ya = y1+t0*(y2-y1)+(ymid-(y1+y2)/2)*4*t0*(1-t0)
                    yb = y1+t1*(y2-y1)+(ymid-(y1+y2)/2)*4*t1*(1-t1)
                    mid_z = (za+zb)/2
                    mid_y = (ya+yb)/2
                    seg_len = math.sqrt((zb-za)**2+(yb-ya)**2)
                    angle = math.atan2(yb-ya, zb-za)
                    t = glm.translate(glm.mat4(1), glm.vec3(cx, mid_y, mid_z))
                    t = glm.rotate(t, -angle, glm.vec3(1,0,0))
                    lit.add(make_box(0.08, 0.08, seg_len, 0xdd4422), t)

        # Bridge deck lights
        for blz in range(355, 1345, 14):
            for s in [-1,1]:
                t = glm.translate(glm.mat4(1), glm.vec3(s*(RW/2+1.5), 2.55, blz))
                lit.add(make_sphere_approx(0.14, 5, 3, 0xffdd88), t)

    def _build_buildings(self, lit, unlit):
        random.seed(123)
        bPal = [0x3a2828,0x2d2838,0x383028,0x282d38,0x443333,0x334444,
                0x383838,0x4a3030,0x30304a,0x3a3a2a,0x2a2a3e,0x3e2a2a]
        bEdge = RW/2 + SW + 1

        def mk_bldg(bx, bz, bw, bd, bh, col):
            t = glm.translate(glm.mat4(1), glm.vec3(bx, bh/2, bz))
            lit.add(make_box(bw, bh, bd, col), t)
            self.colliders.append((bx-bw/2-0.3, bx+bw/2+0.3, bz-bd/2-0.3, bz+bd/2+0.3))

            # Windows (as small unlit planes on the facing side)
            facing = -1 if bx > 0 else 1
            wc_pool = [0x88bbff, 0xffdd88, 0xffeedd, 0x0c0a08]
            for fl in np.arange(4, bh-2, 3.5):
                for wx in np.arange(-bw/2+2, bw/2-1, 3):
                    wc = random.choice(wc_pool[:3]) if random.random() > 0.12 else 0x0c0a08
                    wt = glm.translate(glm.mat4(1), glm.vec3(bx+wx, fl, bz+facing*bd/2+facing*0.05))
                    unlit.add_plane(1.1, 1.5, wc, wt)

            # Rooftop details
            if bh > 25 and random.random() > 0.4:
                rh = 2 + random.random()*4
                t = glm.translate(glm.mat4(1), glm.vec3(bx, bh+rh/2, bz))
                lit.add(make_box(bw*0.3, rh, bd*0.3, 0x222222), t)

        # City blocks (before bridge)
        for s in [-1, 1]:
            bz = -950.0
            while bz < 180:
                bw = 12 + random.random()*22
                bd = 14 + random.random()*24
                bh = 15 + random.random()*70
                gap = 1.5 + random.random()*3
                mk_bldg(s*(bEdge+bw/2+random.random()*5), bz+bd/2, bw, bd, bh, random.choice(bPal))
                bz += bd + gap

        # After bridge
        for s in [-1, 1]:
            bz = 1420.0
            while bz < 2400:
                bw = 10 + random.random()*16
                bd = 12 + random.random()*18
                bh = 8 + random.random()*28
                gap = 3 + random.random()*6
                mk_bldg(s*(bEdge+bw/2+random.random()*4), bz+bd/2, bw, bd, bh, random.choice(bPal))
                bz += bd + gap

    def _add_lamp(self, lit, unlit, x, z):
        t = glm.translate(glm.mat4(1), glm.vec3(x, 2.9, z))
        lit.add(make_cylinder(0.09, 5.8, 6, 0x3a3a3a), t)
        # Arm extending toward road
        arm_dir = -1 if x > 0 else 1
        t = glm.translate(glm.mat4(1), glm.vec3(x + arm_dir*1.1, 8.5, z))
        lit.add(make_box(2.2, 0.1, 0.1, 0x3a3a3a), t)
        # Bulb (brighter, emissive)
        bx = x + arm_dir*2.2
        t = glm.translate(glm.mat4(1), glm.vec3(bx, 8.35, z))
        lit.add(make_sphere_approx(0.22, 5, 3, 0xffeeaa), t)
        # Glow on ground (unlit warm circle)
        t = glm.translate(glm.mat4(1), glm.vec3(bx, 0.04, z))
        unlit.add_plane(3.5, 4.5, 0x332a10, t)
        self.colliders.append((x-0.2, x+0.2, z-0.2, z+0.2))

    def _add_tree(self, lit, x, z):
        th = 1.5 + random.random()*1.2
        t = glm.translate(glm.mat4(1), glm.vec3(x, th/2, z))
        lit.add(make_cylinder(0.12, th, 5, 0x3d2817), t)
        leaf_col = random.choice([0x1a3a1a, 0x1f4420, 0x163316, 0x1a4428])
        for r, h, y_off in [(1.5,2,th+0.8),(1.05,1.6,th+2.2),(0.55,1.3,th+3.2)]:
            t = glm.translate(glm.mat4(1), glm.vec3(x, y_off, z))
            lit.add(make_cone(r, h, 7, leaf_col), t)

# ─────────────────────────────────────────────
#  Dynamic car model
# ─────────────────────────────────────────────

def build_car_verts(body_col, cab_col, is_player=False):
    """Build car geometry (lit). Returns vertex data list."""
    verts = []
    # Body
    t = glm.translate(glm.mat4(1), glm.vec3(0, 0.52, 0))
    verts.extend(_transform_verts(make_box(2.15, 0.65, 4.5, body_col), t))
    # Cabin
    t = glm.translate(glm.mat4(1), glm.vec3(0, 1.02, -0.25))
    verts.extend(_transform_verts(make_box(1.85, 0.58, 2.1, cab_col), t))
    # Wheels
    for wx, wy, wz in [(-1.1,0.33,1.3),(1.1,0.33,1.3),(-1.1,0.33,-1.3),(1.1,0.33,-1.3)]:
        t = glm.translate(glm.mat4(1), glm.vec3(wx, wy, wz))
        verts.extend(_transform_verts(make_cylinder(0.33, 0.26, 8, 0x1a1a1a), t))
    return verts

def build_car_light_verts(is_player=False, braking=False):
    """Build car light geometry (unlit). Returns pos(3)+color(3) vertex data.
    Headlights (front, white/yellow) and taillights (rear, red)."""
    verts = []
    # Headlights - two bright quads at front of car
    hl_color = 0xffffee if is_player else 0xeeeedd
    for hx in [-0.7, 0.7]:
        # Headlight bulb on front face
        hw, hh = 0.28, 0.18
        y = 0.58
        z = 2.25  # front of car body
        r, g, b = _hex_to_rgb(hl_color)
        pts = [(hx-hw, y-hh, z+0.01), (hx+hw, y-hh, z+0.01),
               (hx+hw, y+hh, z+0.01), (hx-hw, y+hh, z+0.01)]
        for v in (pts[0], pts[1], pts[2], pts[0], pts[2], pts[3]):
            verts.extend(v)
            verts.extend((r, g, b))
        # Headlight glow (larger, dimmer)
        gw, gh = 0.45, 0.28
        gr, gg, gb = r*0.5, g*0.45, b*0.3
        pts2 = [(hx-gw, y-gh, z+0.02), (hx+gw, y-gh, z+0.02),
                (hx+gw, y+gh, z+0.02), (hx-gw, y+gh, z+0.02)]
        for v in (pts2[0], pts2[1], pts2[2], pts2[0], pts2[2], pts2[3]):
            verts.extend(v)
            verts.extend((gr, gg, gb))

    # Taillights - red, brighter when braking
    tl_color = 0xff2222 if braking else 0xcc1111
    for hx in [-0.75, 0.75]:
        tw, th = 0.3, 0.14
        y = 0.55
        z = -2.25  # rear of car body
        r, g, b = _hex_to_rgb(tl_color)
        pts = [(hx-tw, y-th, z-0.01), (hx+tw, y-th, z-0.01),
               (hx+tw, y+th, z-0.01), (hx-tw, y+th, z-0.01)]
        # Wind order reversed for back face
        for v in (pts[2], pts[1], pts[0], pts[3], pts[2], pts[0]):
            verts.extend(v)
            verts.extend((r, g, b))
        # Tail glow
        gw, gh = 0.42, 0.22
        br_mult = 0.6 if braking else 0.3
        pts2 = [(hx-gw, y-gh, z-0.02), (hx+gw, y-gh, z-0.02),
                (hx+gw, y+gh, z-0.02), (hx-gw, y+gh, z-0.02)]
        for v in (pts2[2], pts2[1], pts2[0], pts2[3], pts2[2], pts2[0]):
            verts.extend(v)
            verts.extend((r*br_mult, g*br_mult, b*br_mult))

    return verts

def _transform_verts(verts, transform):
    """Apply transform to pos+normal vertex data"""
    stride = 9
    n = len(verts) // stride
    out = []
    for i in range(n):
        off = i*stride
        p = glm.vec3(verts[off], verts[off+1], verts[off+2])
        norm = glm.vec3(verts[off+3], verts[off+4], verts[off+5])
        p = glm.vec3(transform * glm.vec4(p, 1.0))
        norm = glm.vec3(glm.mat3(transform) * norm)
        out.extend([p.x, p.y, p.z, norm.x, norm.y, norm.z,
                     verts[off+6], verts[off+7], verts[off+8]])
    return out

# ─────────────────────────────────────────────
#  Sky dome (unlit)
# ─────────────────────────────────────────────

def build_sky_verts():
    """Build a hemisphere of colored triangles for the sky"""
    verts = []
    R = 550
    segs_h, segs_v = 24, 12
    for j in range(segs_v):
        t0 = math.pi * j / segs_v
        t1 = math.pi * (j+1) / segs_v
        for i in range(segs_h):
            p0 = 2*math.pi*i/segs_h
            p1 = 2*math.pi*(i+1)/segs_h
            def sv(t, p):
                return (R*math.sin(t)*math.cos(p), R*math.cos(t), R*math.sin(t)*math.sin(p))
            def sky_color(y_norm):
                """Sunset gradient based on normalized height"""
                h = (y_norm/R + 1)*0.5
                if h < 0.3: return (0.98, 0.35, 0.06)
                elif h < 0.42:
                    t = (h-0.3)/0.12
                    return (0.98-t*0.45, 0.35-t*0.05, 0.06+t*0.2)
                elif h < 0.55:
                    t = (h-0.42)/0.13
                    return (0.53-t*0.2, 0.30-t*0.1, 0.26+t*0.08)
                elif h < 0.72:
                    t = (h-0.55)/0.17
                    return (0.33-t*0.18, 0.20-t*0.08, 0.34-t*0.04)
                else:
                    t = (h-0.72)/0.28
                    return (0.15-t*0.08, 0.12-t*0.06, 0.30-t*0.14)

            pts = [sv(t0,p0), sv(t1,p0), sv(t1,p1), sv(t0,p1)]
            for tri in [(0,1,2),(0,2,3)]:
                for idx in tri:
                    px,py,pz = pts[idx]
                    r,g,b = sky_color(py)
                    verts.extend([px,py,pz, r,g,b])
    return verts

# ─────────────────────────────────────────────
#  NPC car
# ─────────────────────────────────────────────

class NPCCar:
    def __init__(self, x, z, direction, speed, max_speed):
        self.x = x
        self.z = z
        self.y = 0.0
        self.dir = direction  # 1=same direction, -1=opposite
        self.speed = speed
        self.max_speed = max_speed
        self.braking = False
        self.body_col = random.choice([0x2255aa,0xdddd33,0x22aa44,0xeeeeee,
            0x222222,0xcc6600,0x8833aa,0x33aaaa,0x888888,0xaa2255])
        self.cab_col = max(0x111111, self.body_col - 0x221100)

# ─────────────────────────────────────────────
#  Main game widget
# ─────────────────────────────────────────────

class CarGameGL(QWidget):
    """
    Pure OpenGL car game widget — no browser, no JS.

    Uses ModernGL with a standalone (headless) context and an offscreen
    framebuffer.  Each frame is rendered to the FBO, read back as pixels,
    and painted onto this plain QWidget via QPainter.  This sidesteps
    the QGraphicsProxyWidget limitation where a GL widget never gets a
    real native surface.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        # Game state
        self.player_x = -LANE_W
        self.player_z = -850.0
        self.player_y = 0.0
        self.speed = 0.0
        self.rot = 0.0
        self.max_speed = 95.0
        self.accel = 32.0
        self.brake_force = 50.0
        self.friction = 6.5
        self.turn_speed = 1.9
        self.distance = 0.0
        self.top_speed = 0.0
        self.nitro = 100.0
        self.nitro_active = False
        self.collision_cooldown = 0.0
        self.cam_mode = 0

        # Camera
        self.cam_x = self.player_x
        self.cam_y = 4.0
        self.cam_z = self.player_z - 10

        # Keys
        self.keys = {'w':False,'s':False,'a':False,'d':False,'space':False,'shift':False}

        # NPCs
        self.npcs = []
        random.seed(99)
        for _ in range(45):
            same_dir = random.random() > 0.35
            if same_dir:
                nx = -LANE_W - (LANE_W if random.random()>0.5 else 0)
                nz = self.player_z + 50 + random.random()*500
                npc = NPCCar(nx, nz, 1, 11+random.random()*14, 14+random.random()*12)
            else:
                nx = LANE_W + (LANE_W if random.random()>0.5 else 0)
                nz = self.player_z - 50 - random.random()*500
                npc = NPCCar(nx, nz, -1, 13+random.random()*16, 15+random.random()*14)
            self.npcs.append(npc)

        # World
        self.world = WorldData()

        # GL objects — created lazily on first paint
        self._gl_ready = False
        self.ctx = None
        self.fbo = None
        self.prog_lit = None
        self.prog_unlit = None
        self.vao_static = None
        self.vao_unlit_static = None
        self.vao_sky = None
        self._fbo_w = 0
        self._fbo_h = 0

        # Rendered QImage (updated each frame)
        self._frame_image = None

        # Timing
        self.frame_count = 0
        self.game_time = 0.0
        self.last_time = None

        # Timer for game loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(16)  # ~60 FPS

    # ── GL init (called once, lazily) ──────────────────────

    def _ensure_gl(self):
        if self._gl_ready:
            return
        self._gl_ready = True

        # Standalone context — works without any native window surface
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        # Compile shaders
        self.prog_lit = self.ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)
        self.prog_unlit = self.ctx.program(vertex_shader=UNLIT_VERT, fragment_shader=UNLIT_FRAG)

        # Build world geometry
        lit_batch, unlit_batch = self.world.build_static()
        self.vao_static = lit_batch.build(self.ctx, self.prog_lit)
        self.vao_unlit_static = unlit_batch.build(self.ctx, self.prog_unlit)

        # Sky
        sky_verts = build_sky_verts()
        sky_data = np.array(sky_verts, dtype='f4').tobytes()
        sky_vbo = self.ctx.buffer(sky_data)
        self.vao_sky = self.ctx.vertex_array(self.prog_unlit,
            [(sky_vbo, '3f 3f', 'in_position', 'in_color')])
        self.sky_vert_count = len(sky_verts) // 6

        # Player car
        player_verts = build_car_verts(0xcc2222, 0x991515, True)
        self.player_car_vert_count = len(player_verts) // 9
        player_data = np.array(player_verts, dtype='f4').tobytes()
        player_vbo = self.ctx.buffer(player_data)
        self.vao_player = self.ctx.vertex_array(self.prog_lit,
            [(player_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

        # Player car lights (unlit)
        player_light_verts = build_car_light_verts(is_player=True)
        self.player_light_count = len(player_light_verts) // 6
        pl_data = np.array(player_light_verts, dtype='f4').tobytes()
        pl_vbo = self.ctx.buffer(pl_data)
        self.vao_player_lights = self.ctx.vertex_array(self.prog_unlit,
            [(pl_vbo, '3f 3f', 'in_position', 'in_color')])

        # NPC cars
        self.npc_vaos = []
        for npc in self.npcs:
            npc_verts = build_car_verts(npc.body_col, npc.cab_col, False)
            npc_data = np.array(npc_verts, dtype='f4').tobytes()
            npc_vbo = self.ctx.buffer(npc_data)
            npc_vao = self.ctx.vertex_array(self.prog_lit,
                [(npc_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
            # NPC lights (default off-state)
            npc_light_verts = build_car_light_verts(is_player=False, braking=False)
            npc_ld = np.array(npc_light_verts, dtype='f4').tobytes()
            npc_lvbo = self.ctx.buffer(npc_ld)
            npc_lvao = self.ctx.vertex_array(self.prog_unlit,
                [(npc_lvbo, '3f 3f', 'in_position', 'in_color')])
            # Also create braking version
            npc_brake_verts = build_car_light_verts(is_player=False, braking=True)
            npc_bd = np.array(npc_brake_verts, dtype='f4').tobytes()
            npc_bvbo = self.ctx.buffer(npc_bd)
            npc_bvao = self.ctx.vertex_array(self.prog_unlit,
                [(npc_bvbo, '3f 3f', 'in_position', 'in_color')])
            self.npc_vaos.append((npc_vao, len(npc_verts)//9, npc_lvao, npc_bvao, len(npc_light_verts)//6))

        # Headlight beam on road (a warm trapezoid projected ahead of the car)
        beam_verts = []
        r, g, b = 0.22, 0.18, 0.08
        # Two overlapping quads: inner bright, outer dim
        # Inner beam
        ri, gi, bi = 0.28, 0.22, 0.10
        inner = [(-1.2, 0.06, 1.0), (1.2, 0.06, 1.0), (2.5, 0.06, 18.0), (-2.5, 0.06, 18.0)]
        for v in (inner[0], inner[1], inner[2], inner[0], inner[2], inner[3]):
            beam_verts.extend(v)
            beam_verts.extend((ri, gi, bi))
        # Outer beam
        ro, go, bo = 0.12, 0.10, 0.05
        outer = [(-2.5, 0.05, 3.0), (2.5, 0.05, 3.0), (4.5, 0.05, 28.0), (-4.5, 0.05, 28.0)]
        for v in (outer[0], outer[1], outer[2], outer[0], outer[2], outer[3]):
            beam_verts.extend(v)
            beam_verts.extend((ro, go, bo))
        beam_data = np.array(beam_verts, dtype='f4').tobytes()
        beam_vbo = self.ctx.buffer(beam_data)
        self.vao_beam = self.ctx.vertex_array(self.prog_unlit,
            [(beam_vbo, '3f 3f', 'in_position', 'in_color')])
        self.beam_vert_count = len(beam_verts) // 6

        # Create initial FBO
        self._resize_fbo(max(self.width(), 320), max(self.height(), 200))

        self.last_time = time.perf_counter()
        self.timer.start()

    def _resize_fbo(self, w, h):
        """(Re)create the offscreen framebuffer at the given size."""
        if w == self._fbo_w and h == self._fbo_h and self.fbo:
            return
        if self.fbo:
            self.fbo.release()
        self._fbo_w = w
        self._fbo_h = h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)),
        )

    # ── Game loop ─────────────────────────────────────────

    def _tick(self):
        now = time.perf_counter()
        dt = min(now - self.last_time, 0.05) if self.last_time else 0.016
        self.last_time = now
        self.game_time += dt
        self.frame_count += 1

        self._update_physics(dt)
        self._update_npcs(dt)
        self._update_traffic_lights(dt)
        self._update_camera(dt)
        self._render_gl_frame()
        self.update()  # triggers paintEvent → draws cached image + HUD

    def _update_physics(self, dt):
        ks = self.keys
        # Nitro
        nitro_boost = 1.0
        if ks['shift'] and self.nitro > 0 and self.speed > 5:
            self.nitro_active = True
            self.nitro = max(0, self.nitro - 35*dt)
            nitro_boost = 1.7
        else:
            self.nitro_active = False
            self.nitro = min(100, self.nitro + 6*dt)

        # Acceleration
        if ks['w']:
            self.speed += self.accel * nitro_boost * dt
        elif ks['s']:
            self.speed -= self.accel * 0.65 * dt
        else:
            if self.speed > 0: self.speed = max(0, self.speed - self.friction*dt)
            else: self.speed = min(0, self.speed + self.friction*dt)

        # Handbrake
        if ks['space']:
            if self.speed > 0: self.speed = max(0, self.speed - self.brake_force*2*dt)
            else: self.speed = min(0, self.speed + self.brake_force*2*dt)

        eff_max = self.max_speed * 1.5 if self.nitro_active else self.max_speed
        self.speed = max(-self.max_speed/3, min(eff_max, self.speed))

        # Steering
        is_drifting = ks['space'] and abs(self.speed) > 12 and (ks['a'] or ks['d'])
        if abs(self.speed) > 0.5:
            tf = min(1.0, abs(self.speed)/22)
            turn_mult = 1.6 if is_drifting else 1.0
            if ks['a']: self.rot += self.turn_speed * tf * dt * (1 if self.speed>0 else -1) * turn_mult
            if ks['d']: self.rot -= self.turn_speed * tf * dt * (1 if self.speed>0 else -1) * turn_mult
        self.rot *= 0.985 if is_drifting else 0.97
        self.rot = max(-0.42, min(0.42, self.rot))

        # Movement
        mx = math.sin(self.rot) * self.speed * dt
        mz = math.cos(self.rot) * self.speed * dt
        nx = self.player_x + mx
        nz = self.player_z + mz

        on_bridge = 200 < nz < 1500
        mx_lim = RW/2+0.5 if on_bridge else RW/2-1.3
        nx = max(-mx_lim, min(mx_lim, nx))

        if self.collision_cooldown > 0:
            self.collision_cooldown -= dt
            self.player_x = max(-mx_lim, min(mx_lim, self.player_x + mx))
            self.player_z += mz
        else:
            if self._check_collision(nx, nz):
                self.speed *= -0.28
                self.collision_cooldown = 0.3
            else:
                self.player_x = nx
                self.player_z = nz

        self.player_y = get_bridge_y(self.player_z)
        self.distance += abs(self.speed * dt)
        cur_spd = abs(self.speed * 3.6)
        if cur_spd > self.top_speed:
            self.top_speed = cur_spd

    def _check_collision(self, nx, nz):
        hw, hd = 1.15, 2.35
        for (minx, maxx, minz, maxz) in self.world.colliders:
            if nx+hw > minx and nx-hw < maxx and nz+hd > minz and nz-hd < maxz:
                return True
        for npc in self.npcs:
            dx = nx - npc.x
            dz = nz - npc.z
            if dx*dx + dz*dz < 8.5:
                return True
        if 340 < nz < 1360 and (nx < -(RW/2+0.8) or nx > (RW/2+0.8)):
            return True
        return False

    def _update_npcs(self, dt):
        for npc in self.npcs:
            # Check traffic light stops
            stop = self._npc_should_stop(npc)
            ahead = False
            for other in self.npcs:
                if other is npc: continue
                dz = (other.z - npc.z) * npc.dir
                if 0 < dz < 12 and abs(other.x - npc.x) < 2.5:
                    ahead = True; break
            # Player ahead check
            pdz = (self.player_z - npc.z) * npc.dir
            if 0 < pdz < 14 and abs(self.player_x - npc.x) < 2.5:
                ahead = True

            if stop or ahead:
                npc.speed = max(0, npc.speed - 32*dt)
                npc.braking = True
            else:
                npc.speed = min(npc.max_speed, npc.speed + 10*dt)
                npc.braking = False

            npc.z += npc.dir * npc.speed * dt
            npc.y = get_bridge_y(npc.z)

            # Respawn if too far
            df = npc.z - self.player_z
            if df < -280 or df > 580:
                npc.z = self.player_z + (npc.dir * (120 + random.random()*380))
                if npc.dir > 0:
                    npc.x = -LANE_W - (LANE_W if random.random()>0.5 else 0)
                else:
                    npc.x = LANE_W + (LANE_W if random.random()>0.5 else 0)
                npc.speed = 11 + random.random()*14

    def _npc_should_stop(self, npc):
        for tl in self.world.traffic_light_positions:
            if npc.dir == 1 and tl['dir'] != -1: continue
            if npc.dir == -1 and tl['dir'] != 1: continue
            dist = (tl['z'] - npc.z) * npc.dir
            if 0 < dist < 22 and (tl['state'] == 2 or (tl['state'] == 1 and dist < 10)):
                return True
        return False

    def _update_traffic_lights(self, dt):
        for tl in self.world.traffic_light_positions:
            tl['timer'] += dt
            if tl['state'] == 0: dur = tl['green_dur']
            elif tl['state'] == 1: dur = tl['yellow_dur']
            else: dur = tl['red_dur']
            if tl['timer'] >= dur:
                tl['timer'] = 0
                tl['state'] = (tl['state'] + 1) % 3

    def _update_camera(self, dt):
        px, py, pz = self.player_x, self.player_y, self.player_z
        if self.cam_mode == 0:
            # Chase cam
            off = glm.vec3(0, 3.8, -10)
            off = glm.vec3(glm.rotate(glm.mat4(1), self.rot, glm.vec3(0,1,0)) * glm.vec4(off, 1.0))
            ix, iy, iz = px + off.x, py + 3.8, pz + off.z
            lerp_f = 1 - math.pow(0.018, dt)
            self.cam_x += (ix - self.cam_x) * lerp_f
            self.cam_y += (iy - self.cam_y) * lerp_f
            self.cam_z += (iz - self.cam_z) * lerp_f
        elif self.cam_mode == 1:
            # High cam
            off = glm.vec3(0, 18, -22)
            off = glm.vec3(glm.rotate(glm.mat4(1), self.rot, glm.vec3(0,1,0)) * glm.vec4(off, 1.0))
            ix, iy, iz = px + off.x, py + 18, pz + off.z
            lerp_f = 1 - math.pow(0.022, dt)
            self.cam_x += (ix - self.cam_x) * lerp_f
            self.cam_y += (iy - self.cam_y) * lerp_f
            self.cam_z += (iz - self.cam_z) * lerp_f
        else:
            # Hood cam
            off = glm.vec3(0, 1.45, 0.85)
            off = glm.vec3(glm.rotate(glm.mat4(1), self.rot, glm.vec3(0,1,0)) * glm.vec4(off, 1.0))
            ix, iy, iz = px + off.x, py + off.y, pz + off.z
            lerp_f = 1 - math.pow(0.006, dt)
            self.cam_x += (ix - self.cam_x) * lerp_f
            self.cam_y += (iy - self.cam_y) * lerp_f
            self.cam_z += (iz - self.cam_z) * lerp_f

    def _render_gl_frame(self):
        """Render the 3D scene to the offscreen FBO and cache as QImage."""
        if not self._gl_ready:
            return

        w, h = max(self.width(), 320), max(self.height(), 200)
        self._resize_fbo(w, h)

        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.18, 0.08, 0.28, 1.0)

        # Projection
        aspect = w / h
        spd_pct = abs(self.speed) / self.max_speed
        fov = 68 + spd_pct * 15 + (8 if self.nitro_active else 0)
        proj = glm.perspective(glm.radians(fov), aspect, 0.5, 1500.0)

        # View
        target = glm.vec3(self.player_x, self.player_y + 1.0, self.player_z + 3.5)
        if self.cam_mode == 2:
            target = glm.vec3(self.player_x, self.player_y + 0.3, self.player_z + 20)
        view = glm.lookAt(
            glm.vec3(self.cam_x, self.cam_y, self.cam_z),
            target,
            glm.vec3(0, 1, 0)
        )

        vp = proj * view
        fog_col = (0.18, 0.08, 0.28)

        # --- Draw sky (unlit, no depth write) ---
        self.ctx.disable(moderngl.DEPTH_TEST)
        sky_model = glm.translate(glm.mat4(1), glm.vec3(self.player_x, 0, self.player_z))
        sky_mvp = proj * view * sky_model
        self.prog_unlit['mvp'].write(sky_mvp)
        self.prog_unlit['fog_color'].write(glm.vec3(*fog_col))
        self.vao_sky.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # --- Draw static world (lit) ---
        identity = glm.mat4(1)
        self.prog_lit['mvp'].write(vp)
        self.prog_lit['model'].write(identity)
        self.prog_lit['light_dir'].write(glm.vec3(0.3, 0.7, 0.5))
        self.prog_lit['ambient'].write(glm.vec3(0.45, 0.35, 0.32))
        self.prog_lit['fog_color'].write(glm.vec3(*fog_col))
        if self.vao_static:
            self.vao_static.render(moderngl.TRIANGLES)

        # --- Draw static unlit (windows, markings) ---
        self.prog_unlit['mvp'].write(vp)
        self.prog_unlit['fog_color'].write(glm.vec3(*fog_col))
        if self.vao_unlit_static:
            self.vao_unlit_static.render(moderngl.TRIANGLES)

        # --- Draw player car ---
        car_model = glm.translate(glm.mat4(1), glm.vec3(self.player_x, self.player_y, self.player_z))
        car_model = glm.rotate(car_model, self.rot, glm.vec3(0,1,0))
        car_mvp = vp * car_model
        self.prog_lit['mvp'].write(car_mvp)
        self.prog_lit['model'].write(car_model)
        self.vao_player.render(moderngl.TRIANGLES)

        # Player car lights (unlit)
        self.prog_unlit['mvp'].write(car_mvp)
        self.vao_player_lights.render(moderngl.TRIANGLES)

        # Headlight beam on road
        beam_model = glm.translate(glm.mat4(1), glm.vec3(self.player_x, self.player_y, self.player_z))
        beam_model = glm.rotate(beam_model, self.rot, glm.vec3(0,1,0))
        beam_mvp = vp * beam_model
        self.prog_unlit['mvp'].write(beam_mvp)
        self.vao_beam.render(moderngl.TRIANGLES)

        # --- Draw NPC cars ---
        for i, npc in enumerate(self.npcs):
            dz = npc.z - self.player_z
            if dz < -300 or dz > 600:
                continue
            npc_model = glm.translate(glm.mat4(1), glm.vec3(npc.x, npc.y, npc.z))
            if npc.dir < 0:
                npc_model = glm.rotate(npc_model, math.pi, glm.vec3(0,1,0))
            npc_mvp = vp * npc_model
            self.prog_lit['mvp'].write(npc_mvp)
            self.prog_lit['model'].write(npc_model)
            vao, count, lvao, bvao, lcount = self.npc_vaos[i]
            vao.render(moderngl.TRIANGLES)

            # NPC car lights (unlit)
            self.prog_unlit['mvp'].write(npc_mvp)
            if npc.braking:
                bvao.render(moderngl.TRIANGLES)
            else:
                lvao.render(moderngl.TRIANGLES)

        # --- Read pixels back as QImage ---
        raw = self.fbo.color_attachments[0].read()
        self._frame_image = QImage(raw, w, h, w * 4, QImage.Format_RGBA8888).mirrored(False, True)

    def paintEvent(self, event):
        """Paint the cached GL frame + HUD overlay."""
        self._ensure_gl()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Draw the 3D scene image
        if self._frame_image and not self._frame_image.isNull():
            painter.drawImage(0, 0, self._frame_image)
        else:
            painter.fillRect(0, 0, w, h, QColor(46, 20, 72))

        # Speed box (bottom-right)
        box_w, box_h = 180, 160
        bx, by = w - box_w - 20, h - box_h - 20
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(5, 2, 12, 230))
        painter.drawRoundedRect(bx, by, box_w, box_h, 18, 18)

        # Border
        painter.setPen(QPen(QColor(255, 120, 60, 50), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(bx, by, box_w, box_h, 18, 18)

        # Speed number
        spd = abs(round(self.speed * 3.6))
        font_big = QFont("monospace", 42, QFont.Black)
        painter.setFont(font_big)
        painter.setPen(QColor(255, 204, 102))
        painter.drawText(bx, by+10, box_w, 60, Qt.AlignCenter, str(spd))

        # KM/H label
        font_sm = QFont("monospace", 9)
        painter.setFont(font_sm)
        painter.setPen(QColor(255, 180, 120, 80))
        painter.drawText(bx, by+65, box_w, 16, Qt.AlignCenter, "K M / H")

        # Gear
        if self.speed < -0.5: gear = "R"
        elif abs(self.speed) < 0.5: gear = "P"
        elif spd < 40: gear = "GEAR 1"
        elif spd < 80: gear = "GEAR 2"
        elif spd < 140: gear = "GEAR 3"
        elif spd < 220: gear = "GEAR 4"
        else: gear = "GEAR 5"
        font_gear = QFont("monospace", 11, QFont.Bold)
        painter.setFont(font_gear)
        painter.setPen(QColor(255, 136, 68))
        painter.drawText(bx, by+82, box_w, 20, Qt.AlignCenter, gear)

        # RPM bar
        rpm_pct = min(1.0, spd / 300)
        rpm_y = by + 110
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255,255,255,15))
        painter.drawRoundedRect(bx+15, rpm_y, box_w-30, 5, 3, 3)
        grad = QLinearGradient(bx+15, 0, bx+15+(box_w-30)*rpm_pct, 0)
        grad.setColorAt(0, QColor(34, 255, 102))
        grad.setColorAt(0.5, QColor(255, 204, 34))
        grad.setColorAt(1.0, QColor(255, 34, 68))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(bx+15, rpm_y, int((box_w-30)*rpm_pct), 5, 3, 3)

        # Nitro bar
        nitro_y = rpm_y + 12
        painter.setBrush(QColor(255,255,255,10))
        painter.drawRoundedRect(bx+15, nitro_y, box_w-30, 3, 2, 2)
        n_grad = QLinearGradient(bx+15, 0, bx+15+(box_w-30)*self.nitro/100, 0)
        n_grad.setColorAt(0, QColor(34, 102, 255))
        n_grad.setColorAt(1, QColor(68, 204, 255))
        painter.setBrush(QBrush(n_grad))
        painter.drawRoundedRect(bx+15, nitro_y, int((box_w-30)*self.nitro/100), 3, 2, 2)

        font_tiny = QFont("monospace", 7)
        painter.setFont(font_tiny)
        painter.setPen(QColor(80, 160, 255, 100))
        painter.drawText(bx+15, nitro_y+5, box_w-30, 12, Qt.AlignCenter, "N I T R O")

        # Info pills (top-left)
        pill_font = QFont("monospace", 10)
        painter.setFont(pill_font)
        pills = [
            (QColor(255, 136, 68), f"{len(self.npcs)} vehicles"),
            (QColor(68, 255, 136), self._zone_label()),
            (QColor(68, 136, 255), f"{self.distance/1000:.1f} km"),
            (QColor(255, 68, 170), f"Top: {self.top_speed:.0f} km/h"),
        ]
        py_off = 16
        for dot_col, text in pills:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(5, 2, 12, 215))
            painter.drawRoundedRect(16, py_off, 180, 28, 14, 14)
            painter.setPen(QPen(QColor(255, 120, 60, 30), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(16, py_off, 180, 28, 14, 14)
            # Dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(dot_col)
            painter.drawEllipse(28, py_off+11, 6, 6)
            # Text
            painter.setPen(QColor(255, 192, 128))
            painter.drawText(42, py_off, 148, 28, Qt.AlignVCenter, text)
            py_off += 36

        # Controls (bottom-left)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(5, 2, 12, 210))
        painter.drawRoundedRect(16, h-90, 320, 74, 12, 12)
        ctrl_font = QFont("monospace", 9)
        painter.setFont(ctrl_font)
        painter.setPen(QColor(255, 212, 160))
        painter.drawText(28, h-82, 300, 20, Qt.AlignVCenter,
            "[W] Gas  [S] Brake  [A][D] Steer  [SPACE] Handbrake")
        painter.drawText(28, h-60, 300, 20, Qt.AlignVCenter,
            "[C] Camera  [SHIFT] Nitro")

        # Drift indicator
        is_drifting = self.keys['space'] and abs(self.speed) > 12 and (self.keys['a'] or self.keys['d'])
        if is_drifting:
            drift_font = QFont("monospace", 16, QFont.Black)
            painter.setFont(drift_font)
            painter.setPen(QColor(255, 68, 68))
            painter.drawText(w-220, h-200, 180, 30, Qt.AlignRight|Qt.AlignVCenter, "DRIFT!")

        # Bridge label
        pz = self.player_z
        if 260 < pz < 500:
            opacity = max(0, 1 - (pz - 260) / 240)
            bridge_font = QFont("monospace", 24, QFont.Black)
            painter.setFont(bridge_font)
            painter.setPen(QColor(255, 136, 68, int(opacity*255)))
            painter.drawText(0, int(h*0.35), w, 40, Qt.AlignCenter, "GOLDEN GATE BRIDGE")
            sub_font = QFont("monospace", 10)
            painter.setFont(sub_font)
            painter.setPen(QColor(255, 160, 100, int(opacity*128)))
            painter.drawText(0, int(h*0.42), w, 20, Qt.AlignCenter, "SAN FRANCISCO, CALIFORNIA")

        # Minimap (top-right)
        mm_x, mm_y, mm_w, mm_h = w-176, 16, 160, 160
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(5, 2, 12, 225))
        painter.drawRoundedRect(mm_x, mm_y, mm_w, mm_h, 14, 14)
        painter.setPen(QPen(QColor(255, 120, 60, 40), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(mm_x, mm_y, mm_w, mm_h, 14, 14)

        # Minimap content
        ms = 0.036
        cx, cy = mm_x + 80, mm_y + 80

        # Water
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(8, 42, 74, 90))
        wy1 = cy - (300-pz)*ms
        wy2 = cy - (1400-pz)*ms
        painter.drawRect(mm_x, int(min(wy1,wy2)), mm_w, int(abs(wy2-wy1)))

        # Road
        painter.setPen(QPen(QColor(42, 42, 42), 4))
        painter.drawLine(cx, int(cy-(-1000-pz)*ms), cx, int(cy-(2500-pz)*ms))

        # Bridge
        painter.setPen(QPen(QColor(204, 51, 17), 5))
        painter.drawLine(cx, int(cy-(350-pz)*ms), cx, int(cy-(1350-pz)*ms))

        # NPC dots
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(51, 102, 170))
        for npc in self.npcs:
            nmx = cx + npc.x * ms * 2
            nmy = cy - (npc.z - pz) * ms
            if mm_y-3 < nmy < mm_y+mm_h+3:
                painter.drawRect(int(nmx-1), int(nmy-1.5), 3, 4)

        # Player dot
        painter.setBrush(QColor(255, 68, 68))
        painter.drawEllipse(int(cx + self.player_x*ms*2 - 3.5), int(cy-3.5), 7, 7)

        # Direction
        painter.setPen(QPen(QColor(255, 68, 68, 128), 1))
        px_m = cx + self.player_x*ms*2
        painter.drawLine(int(px_m), cy, int(px_m + math.sin(self.rot)*8), int(cy - math.cos(self.rot)*8))

        # Vignette effect (simple dark corners)
        vig_grad = QLinearGradient(0, 0, 0, h)
        vig_grad.setColorAt(0, QColor(0, 0, 0, 40))
        vig_grad.setColorAt(0.3, QColor(0, 0, 0, 0))
        vig_grad.setColorAt(0.7, QColor(0, 0, 0, 0))
        vig_grad.setColorAt(1, QColor(0, 0, 0, 70))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(vig_grad))
        painter.drawRect(0, 0, w, h)

        painter.end()

    def _zone_label(self):
        pz = self.player_z
        if pz < 200: return "SF City Streets"
        elif pz < 350: return "Bridge Approach"
        elif pz < 1350: return "Golden Gate Bridge"
        elif pz < 1500: return "Marin Exit"
        else: return "Marin County"

    # --- Input ---
    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_W, Qt.Key_Up): self.keys['w'] = True
        if k in (Qt.Key_S, Qt.Key_Down): self.keys['s'] = True
        if k in (Qt.Key_A, Qt.Key_Left): self.keys['a'] = True
        if k in (Qt.Key_D, Qt.Key_Right): self.keys['d'] = True
        if k == Qt.Key_Space: self.keys['space'] = True
        if k == Qt.Key_Shift: self.keys['shift'] = True
        if k == Qt.Key_C: self.cam_mode = (self.cam_mode + 1) % 3
        event.accept()

    def keyReleaseEvent(self, event):
        k = event.key()
        if k in (Qt.Key_W, Qt.Key_Up): self.keys['w'] = False
        if k in (Qt.Key_S, Qt.Key_Down): self.keys['s'] = False
        if k in (Qt.Key_A, Qt.Key_Left): self.keys['a'] = False
        if k in (Qt.Key_D, Qt.Key_Right): self.keys['d'] = False
        if k == Qt.Key_Space: self.keys['space'] = False
        if k == Qt.Key_Shift: self.keys['shift'] = False
        event.accept()


# ─────────────────────────────────────────────
#  Integration point (same as car_3.py)
# ─────────────────────────────────────────────

# Remove old instances
for name in ['game_proxy2', 'game_proxy3', 'game_proxy_gl']:
    try:
        graphics_scene.removeItem(eval(name))
    except:
        pass

game_container_gl = QWidget()
game_container_gl.setFixedSize(1400, 900)
game_container_gl.setStyleSheet("background: black; border-radius: 15px;")
layout_gl = QVBoxLayout(game_container_gl)
layout_gl.setContentsMargins(0, 0, 0, 0)

game_widget = CarGameGL()
layout_gl.addWidget(game_widget)

game_proxy_gl = graphics_scene.addWidget(game_container_gl)
view = graphics_scene.views()[0]
vr = view.viewport().rect()
sr = view.mapToScene(vr).boundingRect()
game_proxy_gl.setPos(sr.center().x() - 700, sr.center().y() - 450)
game_proxy_gl.setFlag(QGraphicsItem.ItemIsMovable, True)
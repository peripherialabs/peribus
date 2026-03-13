"""
CFDLab — Computational Fluid Dynamics Workbench for Rio
═══════════════════════════════════════════════════════════════════

ARCHITECTURE: This file is the complete runtime. It registers a `cfd`
singleton into the namespace. The LLM sees this source as context and
injects short code snippets to drive it.

EXAMPLES (what the LLM would output):
    cfd.load_airfoil("naca2412")              # NACA 4-digit airfoil
    cfd.load_airfoil("naca0012", n=120)       # higher resolution
    cfd.load_airfoil("naca23015")             # NACA 5-digit
    cfd.load_geometry(pts)                     # custom [(x,y), ...] closed curve
    cfd.load_cylinder(r=1.0, n=64)            # circular cylinder
    cfd.load_plate(chord=1.0)                 # flat plate

    cfd.set_flow(aoa=5, v_inf=1.0)           # angle of attack (deg), freestream
    cfd.solve()                                # run panel method solver
    cfd.solve(method="vortex")                # vortex panel method (with lift)

    ## After solve(), cfd.info contains REAL data:
    cfd.info['Cl']                # lift coefficient
    cfd.info['Cd']                # pressure drag coefficient
    cfd.info['Cm']                # pitching moment coefficient (about c/4)
    cfd.info['Cp']                # pressure coefficient per panel
    cfd.info['stagnation']        # stagnation point index
    cfd.info['circulation']       # total circulation Γ

    ## Visualization:
    cfd.overlay_cp()                           # Cp distribution plot
    cfd.overlay_streamlines(n=20, density=1.5) # streamline field
    cfd.overlay_velocity()                     # velocity vectors on surface
    cfd.overlay_forces()                       # lift/drag arrows
    cfd.overlay_polar(aoa_range=(-5,15,1))     # Cl vs alpha sweep
    cfd.color_pressure()                       # color mesh by Cp
    cfd.color_velocity()                       # color mesh by |V|

    ## Geometry tools:
    cfd.scale(2.0)                             # scale geometry
    cfd.translate(dx, dy)                      # move geometry
    cfd.rotate(deg)                            # rotate geometry
    cfd.flip()                                 # flip y
    cfd.refine(n=200)                          # re-panel with n points

    ## Multi-element:
    cfd.add_flap(deflection=20, gap=0.02)     # slotted flap
    cfd.add_slat(deflection=15)               # leading edge slat

    ## Export:
    cfd.export_dat("/tmp/airfoil.dat")        # Selig format
    cfd.export_csv("/tmp/results.csv")        # Cp, velocities, etc.

    ## Parametric:
    cfd.sweep("aoa", -5, 15, 1)              # sweep AoA, collect Cl/Cd/Cm
    cfd.compare("naca0012", "naca2412", "naca4415")  # overlay multiple

VIEWER API (lower level, when LLM needs custom rendering):
    cfd.viewer.cam_x, cfd.viewer.cam_y       # camera pan
    cfd.viewer.cam_zoom                       # camera zoom
    cfd.viewer.add_overlay(name, fn)          # fn(painter, w, h)
    cfd.viewer.remove_overlay(name)
    cfd.viewer.screenshot(path)               # save PNG

NAMESPACE: After this file runs, these are available:
    cfd             — CFDLab singleton (main API)
    cfd_flow        — alias for cfd (short form)
    cfd_viewer      — alias for cfd.viewer (the GL widget)
    cfd_ui          — CFDLabUI instance (full UI controller)
    cfd_main_widget — top-level QWidget
    NACA_PRESETS    — common airfoil database
    All PySide6/Qt, numpy, moderngl from parser namespace
"""

import math
import os
import re
import threading
import numpy as np
from collections import OrderedDict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QCheckBox, QTabWidget, QTextEdit,
    QScrollArea, QListWidget, QLineEdit, QGraphicsItem,
    QGraphicsDropShadowEffect, QSizePolicy, QDoubleSpinBox,
    QSpinBox, QGroupBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage, QLinearGradient,
    QPainterPath, QPolygonF, QConicalGradient, QRadialGradient
)
from PySide6.QtCore import QPointF

import moderngl
import json

# ── Pure-numpy matrix math (replaces glm dependency) ──────────

def _np_perspective(fov_rad, aspect, near, far):
    """Perspective projection matrix (column-major for OpenGL)."""
    f = 1.0 / math.tan(fov_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def _np_look_at(eye, target, up):
    """Look-at view matrix (column-major)."""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

DEG = math.pi / 180.0

# Standard color ramps
def _hex(h):
    return ((h >> 16) & 0xFF) / 255.0, ((h >> 8) & 0xFF) / 255.0, (h & 0xFF) / 255.0

def _qcolor_hex(h):
    return QColor((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)

# Pressure colormap: blue (low Cp/suction) → white → red (high Cp/stagnation)
_CP_COLORS = [
    (-3.0, 0x1a237e),   # deep blue — strong suction
    (-2.0, 0x1565c0),
    (-1.0, 0x42a5f5),
    (-0.5, 0x90caf9),
    ( 0.0, 0xf5f5f5),   # neutral white
    ( 0.5, 0xef9a9a),
    ( 1.0, 0xc62828),   # stagnation red
]

def _cp_to_color(cp):
    """Map Cp value to RGB tuple (0-1)."""
    for i in range(len(_CP_COLORS) - 1):
        v0, c0 = _CP_COLORS[i]
        v1, c1 = _CP_COLORS[i + 1]
        if cp <= v1:
            t = max(0, min(1, (cp - v0) / (v1 - v0)))
            r0, g0, b0 = _hex(c0)
            r1, g1, b1 = _hex(c1)
            return (r0 + t * (r1 - r0), g0 + t * (g1 - g0), b0 + t * (b1 - b0))
    return _hex(_CP_COLORS[-1][1])

def _cp_to_qcolor(cp):
    r, g, b = _cp_to_color(cp)
    return QColor(int(r * 255), int(g * 255), int(b * 255))

# Velocity colormap: dark → cyan → yellow → red
_VEL_COLORS = [
    (0.0, 0x0d1b2a),
    (0.3, 0x1b4965),
    (0.6, 0x5fa8d3),
    (0.8, 0xfca311),
    (1.0, 0xe63946),
]

def _vel_to_color(v_norm):
    """Map normalized velocity (0-1) to RGB tuple."""
    for i in range(len(_VEL_COLORS) - 1):
        v0, c0 = _VEL_COLORS[i]
        v1, c1 = _VEL_COLORS[i + 1]
        if v_norm <= v1:
            t = max(0, min(1, (v_norm - v0) / (v1 - v0)))
            r0, g0, b0 = _hex(c0)
            r1, g1, b1 = _hex(c1)
            return (r0 + t * (r1 - r0), g0 + t * (g1 - g0), b0 + t * (b1 - b0))
    return _hex(_VEL_COLORS[-1][1])

# ═══════════════════════════════════════════════════════════════
#  NACA GEOMETRY
# ═══════════════════════════════════════════════════════════════

def naca4(code, n=80):
    """Generate NACA 4-digit airfoil coordinates.
    Returns list of (x, y) from TE clockwise around to TE (closed).
    """
    code = code.replace("naca", "").replace("NACA", "").strip()
    if len(code) != 4:
        raise ValueError(f"Expected 4-digit NACA code, got '{code}'")
    m = int(code[0]) / 100.0        # max camber
    p = int(code[1]) / 10.0         # location of max camber
    t = int(code[2:4]) / 100.0      # thickness

    # Cosine spacing for better LE resolution
    beta = np.linspace(0, np.pi, n)
    xc = 0.5 * (1 - np.cos(beta))

    # Thickness distribution (NACA standard)
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                   + 0.2843 * xc**3 - 0.1015 * xc**4)

    # Camber and gradient
    yc = np.zeros_like(xc)
    dyc = np.zeros_like(xc)
    if m > 0 and p > 0:
        fwd = xc <= p
        aft = ~fwd
        yc[fwd] = (m / p**2) * (2 * p * xc[fwd] - xc[fwd]**2)
        yc[aft] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * xc[aft] - xc[aft]**2)
        dyc[fwd] = (2 * m / p**2) * (p - xc[fwd])
        dyc[aft] = (2 * m / (1 - p)**2) * (p - xc[aft])

    theta = np.arctan(dyc)

    # Upper and lower surfaces
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Assemble: counterclockwise from TE — lower surface then upper surface
    # lower goes TE→LE (reversed), upper goes LE→TE
    upper = list(zip(xu, yu))  # LE→TE
    lower = list(zip(xl, yl))  # LE→TE
    # CCW: TE along lower→LE, then LE along upper→TE
    pts = list(reversed(lower)) + upper[1:]
    return pts

def naca5(code, n=80):
    """Generate NACA 5-digit airfoil coordinates (simplified)."""
    code = code.replace("naca", "").replace("NACA", "").strip()
    if len(code) != 5:
        raise ValueError(f"Expected 5-digit NACA code, got '{code}'")
    # 5-digit: use the camber line definition
    cl_design = int(code[0]) * 0.15     # design lift coefficient
    p = int(code[1]) / 20.0             # location of max camber
    reflex = int(code[2])               # 0=normal, 1=reflex
    t = int(code[3:5]) / 100.0          # thickness

    beta = np.linspace(0, np.pi, n)
    xc = 0.5 * (1 - np.cos(beta))

    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                   + 0.2843 * xc**3 - 0.1015 * xc**4)

    # Simplified 5-digit camber (use equivalent 4-digit approximation)
    m = cl_design / 10.0  # approximate
    if p == 0: p = 0.05
    yc = np.zeros_like(xc)
    dyc = np.zeros_like(xc)
    fwd = xc <= p
    aft = ~fwd
    if m > 0:
        yc[fwd] = (m / p**2) * (2 * p * xc[fwd] - xc[fwd]**2)
        yc[aft] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * xc[aft] - xc[aft]**2)
        dyc[fwd] = (2 * m / p**2) * (p - xc[fwd])
        dyc[aft] = (2 * m / (1 - p)**2) * (p - xc[aft])

    theta = np.arctan(dyc)
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    upper = list(zip(xu, yu))
    lower = list(zip(xl, yl))
    pts = list(reversed(lower)) + upper[1:]
    return pts

def make_cylinder(r=1.0, n=64):
    """Circular cylinder cross-section."""
    theta = np.linspace(0, 2 * np.pi, n + 1)
    return [(r * np.cos(t), r * np.sin(t)) for t in theta]

def make_plate(chord=1.0, t=0.001, n=40):
    """Flat plate with tiny thickness."""
    x = np.linspace(0, chord, n)
    # CCW: lower surface LE→TE, then upper surface TE→LE
    lower = [(xi, -t / 2) for xi in x]
    upper = [(xi, t / 2) for xi in reversed(x)]
    return lower + upper

def make_ellipse(a=1.0, b=0.3, n=80):
    """Elliptical cross-section."""
    theta = np.linspace(0, 2 * np.pi, n + 1)
    return [(a * np.cos(t), b * np.sin(t)) for t in theta]

# Common presets
NACA_PRESETS = {
    "naca0006": "Thin symmetric — low drag",
    "naca0012": "Classic symmetric — study & validation",
    "naca0018": "Thick symmetric — vertical tails",
    "naca2412": "General aviation — moderate lift",
    "naca2415": "GA standard — Cessna 172 wing",
    "naca4412": "High camber — good Cl",
    "naca4415": "High camber thick — slow flight",
    "naca23012": "Laminar flow — 5-digit series",
    "naca23015": "5-digit thick — transport aircraft",
    "naca6412": "Very high camber — flap studies",
    "cylinder": "Circular cylinder — flow separation",
    "flat_plate": "Flat plate — thin airfoil theory",
    "ellipse": "Elliptical section — Joukowski",
}

# ═══════════════════════════════════════════════════════════════
#  PANEL METHOD SOLVER
# ═══════════════════════════════════════════════════════════════

class PanelSolver:
    """2D panel method — source or vortex formulation.

    Source panels: solves for surface source strengths σ to satisfy
    no-penetration BC. Good for non-lifting flows.

    Vortex panels: linear-strength vortex panels with Kutta condition.
    Gives lift, circulation, and pressure distribution.
    """

    def __init__(self):
        self.panels = []   # list of panel dicts
        self.n = 0
        self.sigma = None  # source strengths
        self.gamma = None  # vortex strengths
        self.Cp = None
        self.Vt = None     # tangential velocity
        self.Cl = 0
        self.Cd = 0
        self.Cm = 0
        self.circulation = 0
        self.stagnation_idx = 0

    def set_panels(self, pts, aoa_deg=0, v_inf=1.0):
        """Build panels from ordered point list. aoa in degrees."""
        self.n = len(pts) - 1
        aoa = aoa_deg * DEG
        self.panels = []
        for i in range(self.n):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            xc = 0.5 * (x0 + x1)
            yc = 0.5 * (y0 + y1)
            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-14:
                length = 1e-14
            # Panel angle
            beta = math.atan2(dy, dx)
            # Normal (outward) — 90° CCW from panel direction
            nx = -math.sin(beta)
            ny = math.cos(beta)
            # Tangent
            tx = math.cos(beta)
            ty = math.sin(beta)
            self.panels.append({
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'xc': xc, 'yc': yc, 'length': length,
                'beta': beta, 'nx': nx, 'ny': ny, 'tx': tx, 'ty': ty,
            })
        self.aoa = aoa
        self.v_inf = v_inf
        self.u_inf = v_inf * math.cos(aoa)
        self.v_inf_y = v_inf * math.sin(aoa)

    def _panel_influence(self, xp, yp, panel):
        """Compute (source_vx, source_vy, vortex_vx, vortex_vy) for a unit-strength
        source panel and a unit-strength vortex panel at field point (xp, yp).

        This follows Katz & Plotkin, Ch. 11: constant-strength source and vortex
        panels on a segment from (x0,y0) to (x1,y1).

        The key integrals in the panel-local frame (ξ along panel, η perpendicular):
          Source:  u_s = (1/4π) ln(r1²/r2²),  v_s = (1/2π)(θ₂ - θ₁)
          Vortex:  u_v = (1/2π)(θ₂ - θ₁),     v_v = -(1/4π) ln(r1²/r2²)

        Both are then rotated back to global frame.
        """
        x0, y0 = panel['x0'], panel['y0']
        L = panel['length']
        cos_b = (panel['x1'] - x0) / L
        sin_b = (panel['y1'] - y0) / L

        # Transform point to panel-local coordinates
        dxp = xp - x0
        dyp = yp - y0
        xl = dxp * cos_b + dyp * sin_b
        yl = -dxp * sin_b + dyp * cos_b

        # Distances and angles
        r1_sq = xl * xl + yl * yl
        r2_sq = (xl - L) * (xl - L) + yl * yl
        r1_sq = max(r1_sq, 1e-20)
        r2_sq = max(r2_sq, 1e-20)

        log_term = 0.5 * math.log(r1_sq / r2_sq)  # = ln(r1/r2)
        theta1 = math.atan2(yl, xl)
        theta2 = math.atan2(yl, xl - L)
        dtheta = theta2 - theta1

        # Local-frame velocities (per unit strength, divided by 2π)
        inv2pi = 1.0 / (2.0 * math.pi)

        # Source panel (local frame)
        us_l = inv2pi * log_term       # along panel
        vs_l = inv2pi * dtheta         # perpendicular to panel

        # Vortex panel (local frame) — 90° rotation of source
        uv_l = inv2pi * dtheta         # along panel
        vv_l = -inv2pi * log_term      # perpendicular to panel

        # Rotate back to global frame
        s_vx = us_l * cos_b - vs_l * sin_b
        s_vy = us_l * sin_b + vs_l * cos_b
        v_vx = uv_l * cos_b - vv_l * sin_b
        v_vy = uv_l * sin_b + vv_l * cos_b

        return s_vx, s_vy, v_vx, v_vy

    def set_panels(self, pts, aoa_deg=0, v_inf=1.0):
        """Build panels from ordered point list. aoa in degrees."""
        self.n = len(pts) - 1
        aoa = aoa_deg * DEG
        self.panels = []
        for i in range(self.n):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            xc = 0.5 * (x0 + x1)
            yc = 0.5 * (y0 + y1)
            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-14:
                length = 1e-14
            beta = math.atan2(dy, dx)
            nx = -math.sin(beta)
            ny = math.cos(beta)
            tx = math.cos(beta)
            ty = math.sin(beta)
            self.panels.append({
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'xc': xc, 'yc': yc, 'length': length,
                'beta': beta, 'nx': nx, 'ny': ny, 'tx': tx, 'ty': ty,
            })
        self.aoa = aoa
        self.v_inf = v_inf
        self.u_inf = v_inf * math.cos(aoa)
        self.v_inf_y = v_inf * math.sin(aoa)

        # Pre-compute influence coefficient matrices for all panel pairs.
        # An[i,j] = normal velocity at panel i due to unit source on panel j
        # At[i,j] = tangential velocity at panel i due to unit source on panel j
        # Bn[i,j] = normal velocity at panel i due to unit vortex on panel j
        # Bt[i,j] = tangential velocity at panel i due to unit vortex on panel j
        n = self.n
        self._An = np.zeros((n, n))
        self._At = np.zeros((n, n))
        self._Bn = np.zeros((n, n))
        self._Bt = np.zeros((n, n))

        for i in range(n):
            pi = self.panels[i]
            for j in range(n):
                if i == j:
                    # Self-influence: source normal = 0.5, vortex tangential = 0.5
                    self._An[i, j] = 0.5
                    self._At[i, j] = 0.0
                    self._Bn[i, j] = 0.0
                    self._Bt[i, j] = 0.5
                else:
                    pj = self.panels[j]
                    s_vx, s_vy, v_vx, v_vy = self._panel_influence(
                        pi['xc'], pi['yc'], pj)
                    # Project onto panel i's normal and tangent
                    self._An[i, j] = s_vx * pi['nx'] + s_vy * pi['ny']
                    self._At[i, j] = s_vx * pi['tx'] + s_vy * pi['ty']
                    self._Bn[i, j] = v_vx * pi['nx'] + v_vy * pi['ny']
                    self._Bt[i, j] = v_vx * pi['tx'] + v_vy * pi['ty']

    def solve_source(self):
        """Source panel method — non-lifting, good for Cp on symmetric bodies."""
        n = self.n
        self.sigma = np.linalg.solve(
            self._An,
            np.array([-(self.u_inf * p['nx'] + self.v_inf_y * p['ny'])
                       for p in self.panels]))
        self.gamma = 0.0

        # Tangential velocity from source influence
        Vt = np.zeros(n)
        for i in range(n):
            pi = self.panels[i]
            Vt[i] = self.u_inf * pi['tx'] + self.v_inf_y * pi['ty']
            for j in range(n):
                if i != j:
                    Vt[i] += self.sigma[j] * self._At[i, j]

        self.Vt = Vt
        self.Cp = 1.0 - (Vt / self.v_inf) ** 2
        self.Cl = 0
        self.Cd = sum(self.Cp[i] * self.panels[i]['length'] * self.panels[i]['nx']
                       for i in range(n))
        self.Cm = 0
        self.circulation = 0
        self.stagnation_idx = int(np.argmax(self.Cp))

    def solve_vortex(self):
        """Source + constant-vortex panel method with Kutta condition.

        Each panel j has its own source strength σ_j. All panels share one
        vortex strength γ. The system is:
          Row i (no-penetration):  Σ_j An[i,j]·σ_j + γ·Σ_j Bn[i,j] = -V∞·n_i
          Row n+1 (Kutta):         Σ_j (At[0,j]+At[n-1,j])·σ_j
                                 + γ·Σ_j (Bt[0,j]+Bt[n-1,j]) = -(V∞·t_0 + V∞·t_{n-1})
        """
        n = self.n
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)

        An, At, Bn, Bt = self._An, self._At, self._Bn, self._Bt

        # No-penetration rows
        for i in range(n):
            A[i, :n] = An[i, :]
            A[i, n] = np.sum(Bn[i, :])
            pi = self.panels[i]
            b[i] = -(self.u_inf * pi['nx'] + self.v_inf_y * pi['ny'])

        # Kutta condition: Vt at first panel + Vt at last panel = 0
        A[n, :n] = At[0, :] + At[n - 1, :]
        A[n, n] = np.sum(Bt[0, :]) + np.sum(Bt[n - 1, :])
        p0 = self.panels[0]
        pn = self.panels[n - 1]
        b[n] = -(self.u_inf * (p0['tx'] + pn['tx']) +
                  self.v_inf_y * (p0['ty'] + pn['ty']))

        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]

        self.sigma = sol[:n]
        self.gamma = sol[n]

        # Tangential velocity at each panel
        Vt = np.zeros(n)
        for i in range(n):
            pi = self.panels[i]
            vt = self.u_inf * pi['tx'] + self.v_inf_y * pi['ty']
            for j in range(n):
                vt += self.sigma[j] * At[i, j]
                vt += self.gamma * Bt[i, j]
            Vt[i] = vt

        self.Vt = Vt
        V_inf = self.v_inf
        self.Cp = 1.0 - (Vt / V_inf) ** 2

        # Lift from Kutta-Joukowski
        self.circulation = self.gamma * sum(p['length'] for p in self.panels)
        self.Cl = 2 * self.circulation / (V_inf * self.panels[0]['length'] *
                                            self.n)  # normalized by chord
        # More robust: Cl = 2*Γ/(V∞*c) where c ≈ total panel projection
        chord = max(p['xc'] for p in self.panels) - min(p['xc'] for p in self.panels)
        if chord < 1e-6:
            chord = 1.0
        self.Cl = 2 * self.circulation / (V_inf * chord)

        # Pressure integration
        Cd = 0
        Cm = 0
        xref = 0.25 * chord + min(p['xc'] for p in self.panels)
        for i in range(n):
            pi = self.panels[i]
            fcp = self.Cp[i] * pi['length']
            Cd += fcp * pi['nx']
            Cm += -fcp * pi['ny'] * (pi['xc'] - xref) + fcp * pi['nx'] * pi['yc']
        self.Cd = Cd
        self.Cm = Cm / chord
        self.stagnation_idx = int(np.argmax(self.Cp))

    def _point_in_body(self, xp, yp, pts):
        """Ray-casting point-in-polygon test."""
        n = len(pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > yp) != (yj > yp)) and \
               (xp < (xj - xi) * (yp - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    def velocity_at(self, xp, yp):
        """Compute velocity at arbitrary field point (xp, yp).
        Uses the full source + vortex panel influence."""
        vx = self.u_inf
        vy = self.v_inf_y
        for j in range(self.n):
            pj = self.panels[j]
            s_vx, s_vy, v_vx, v_vy = self._panel_influence(xp, yp, pj)
            if self.sigma is not None:
                vx += self.sigma[j] * s_vx
                vy += self.sigma[j] * s_vy
            if self.gamma is not None:
                vx += self.gamma * v_vx
                vy += self.gamma * v_vy
        return vx, vy

    def streamline(self, x0, y0, body_pts=None, ds=0.01, max_steps=800, forward=True):
        """Trace a streamline from (x0, y0). Stops at body or domain edge."""
        pts = [(x0, y0)]
        x, y = x0, y0
        sign = 1.0 if forward else -1.0
        for _ in range(max_steps):
            vx, vy = self.velocity_at(x, y)
            vmag = math.sqrt(vx * vx + vy * vy)
            if vmag < 1e-10:
                break
            x += sign * ds * vx / vmag
            y += sign * ds * vy / vmag
            if abs(x) > 5 or abs(y) > 5:
                break
            if body_pts and self._point_in_body(x, y, body_pts):
                break
            pts.append((x, y))
        return pts


# ═══════════════════════════════════════════════════════════════
#  SHADERS (2.5D mesh visualization)
# ═══════════════════════════════════════════════════════════════

_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}"""

_FRAG = """
#version 330
in vec3 v_color;
out vec4 frag;
void main() {
    frag = vec4(v_color, 1.0);
}"""

_FRAG_ALPHA = """
#version 330
in vec3 v_color;
uniform float alpha;
out vec4 frag;
void main() {
    frag = vec4(v_color, alpha);
}"""


# ═══════════════════════════════════════════════════════════════
#  GEOMETRY BUILDER — triangulated surface mesh
# ═══════════════════════════════════════════════════════════════

def _build_airfoil_mesh_3d(pts, cp_values=None, extrude=0.15):
    """Build a 3D extruded airfoil mesh.
    pts: [(x,y), ...] surface points
    cp_values: Cp per panel (n-1 values) or None (uniform gray)
    extrude: half-span in z
    Returns flat list of floats: pos(3)+color(3) per vertex.
    """
    n = len(pts)
    verts = []

    # Default color
    def_col = (0.82, 0.85, 0.88)

    def cp_color(i):
        if cp_values is not None and i < len(cp_values):
            return _cp_to_color(cp_values[i])
        return def_col

    # Front face (z = +extrude) — fan from centroid
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    for i in range(n - 1):
        col = cp_color(i)
        for px, py in [(cx, cy), pts[i], pts[i + 1]]:
            verts.extend([px, py, extrude])
            verts.extend(col)

    # Back face (z = -extrude)
    for i in range(n - 1):
        col = cp_color(i)
        for px, py in [(cx, cy), pts[i + 1], pts[i]]:
            verts.extend([px, py, -extrude])
            verts.extend(col)

    # Side walls
    for i in range(n - 1):
        col = cp_color(i)
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        for v in [(x0, y0, extrude), (x1, y1, extrude), (x1, y1, -extrude),
                  (x0, y0, extrude), (x1, y1, -extrude), (x0, y0, -extrude)]:
            verts.extend(v)
            verts.extend(col)

    return verts


# ═══════════════════════════════════════════════════════════════
#  GL VIEWER WIDGET
# ═══════════════════════════════════════════════════════════════

class CFDViewer(QWidget):
    """OpenGL 2.5D CFD viewer — renders extruded airfoil + QPainter overlays.
    Supports mesh coloring, streamlines, and interactive camera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(450, 350)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Camera state
        self.cam_x = 0.5
        self.cam_y = 0.0
        self.cam_zoom = 2.8
        self.rot_x = 0.25
        self.rot_y = -0.4
        self.auto_rot = 0.0
        self._dragging = False
        self._lmx = 0
        self._lmy = 0
        self._rmb = False

        # Data
        self.pts = []           # surface points [(x,y), ...]
        self.body_name = ""
        self.cp_values = None   # Cp per panel

        # GL
        self._gl_ready = False
        self.ctx = None
        self.fbo = None
        self._fbo_w = 0
        self._fbo_h = 0
        self._frame = None
        self._mesh_vao = None
        self._mesh_n = 0

        # Overlays
        self._overlays = OrderedDict()

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(16)

    def _ensure_gl(self):
        if self._gl_ready:
            return
        self._gl_ready = True
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self._resize_fbo(max(self.width(), 320), max(self.height(), 200))
        self.timer.start()

    def _resize_fbo(self, w, h):
        if w == self._fbo_w and h == self._fbo_h and self.fbo:
            return
        if self.fbo:
            self.fbo.release()
        self._fbo_w = w
        self._fbo_h = h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)))

    def set_body(self, pts, name=""):
        """Set airfoil / body geometry from point list."""
        self.pts = pts
        self.body_name = name
        self.cp_values = None
        self._rebuild_mesh()

    def set_cp(self, cp_values):
        """Color the mesh by Cp."""
        self.cp_values = cp_values
        self._rebuild_mesh()

    def _rebuild_mesh(self):
        if not self._gl_ready or not self.pts:
            return
        verts = _build_airfoil_mesh_3d(self.pts, self.cp_values)
        if not verts:
            self._mesh_vao = None
            self._mesh_n = 0
            return
        data = np.array(verts, dtype='f4').tobytes()
        if self._mesh_vao:
            try:
                self._mesh_vao.release()
            except:
                pass
        vbo = self.ctx.buffer(data)
        self._mesh_vao = self.ctx.vertex_array(
            self.prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
        self._mesh_n = len(verts) // 6

    def add_overlay(self, name, fn):
        self._overlays[name] = fn

    def remove_overlay(self, name):
        self._overlays.pop(name, None)

    def screenshot(self, path):
        if self._frame:
            self._frame.save(path)

    def _tick(self):
        if not self._dragging:
            self.auto_rot += 0.003
        self._render()
        self.update()

    def _render(self):
        if not self._gl_ready:
            return
        w, h = max(self.width(), 320), max(self.height(), 200)
        self._resize_fbo(w, h)
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0, 0, 0, 0)

        fov = math.radians(40)
        proj = _np_perspective(fov, w / h, 0.01, 100.0)
        ry = self.rot_y + self.auto_rot
        d = self.cam_zoom
        eye = (
            math.sin(ry) * math.cos(self.rot_x) * d + self.cam_x,
            math.sin(self.rot_x) * d + self.cam_y,
            math.cos(ry) * math.cos(self.rot_x) * d)
        target = (self.cam_x, self.cam_y, 0.0)
        view = _np_look_at(eye, target, (0, 1, 0))
        mvp = np.ascontiguousarray((proj @ view).T, dtype=np.float32)

        if self._mesh_vao and self._mesh_n > 0:
            self.prog['mvp'].write(mvp.tobytes())
            self._mesh_vao.render(moderngl.TRIANGLES)

        raw = self.fbo.color_attachments[0].read()
        self._frame = QImage(raw, w, h, w * 4, QImage.Format_RGBA8888).mirrored(False, True)

    def paintEvent(self, event):
        self._ensure_gl()
        if self.pts and self._mesh_n == 0:
            self._rebuild_mesh()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._frame and not self._frame.isNull():
            p.drawImage(0, 0, self._frame)

        # HUD
        if self.pts:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 195))
            p.drawRoundedRect(12, 12, 220, 50, 8, 8)
            p.setFont(QFont("Consolas", 10, QFont.Bold))
            p.setPen(QColor(30, 45, 70))
            p.drawText(20, 16, 200, 18, Qt.AlignVCenter, self.body_name or "Body")
            p.setFont(QFont("Consolas", 9))
            p.setPen(QColor(80, 100, 130))
            p.drawText(20, 36, 200, 16, Qt.AlignVCenter,
                       f"{len(self.pts)} pts · {len(self.pts)-1} panels")

        # Overlays
        for name, fn in self._overlays.items():
            try:
                fn(p, w, h)
            except Exception as ex:
                p.setPen(QColor(200, 50, 50))
                p.setFont(QFont("Consolas", 8))
                p.drawText(12, h - 20, f"Overlay '{name}' error: {ex}")

        p.end()

    # ── Mouse interaction ──
    def mousePressEvent(self, e):
        self._dragging = True
        self._lmx = e.pos().x()
        self._lmy = e.pos().y()
        self._rmb = (e.button() == Qt.RightButton)
        self.auto_rot = 0

    def mouseMoveEvent(self, e):
        if not self._dragging:
            return
        dx = e.pos().x() - self._lmx
        dy = e.pos().y() - self._lmy
        self._lmx = e.pos().x()
        self._lmy = e.pos().y()
        if self._rmb:
            self.cam_x -= dx * 0.003 * self.cam_zoom
            self.cam_y += dy * 0.003 * self.cam_zoom
        else:
            self.rot_y += dx * 0.008
            self.rot_x += dy * 0.008
            self.rot_x = max(-1.4, min(1.4, self.rot_x))

    def mouseReleaseEvent(self, e):
        self._dragging = False

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        self.cam_zoom *= 0.95 if delta > 0 else 1.05
        self.cam_zoom = max(0.3, min(20.0, self.cam_zoom))


# ═══════════════════════════════════════════════════════════════
#  SIGNALS
# ═══════════════════════════════════════════════════════════════

class _Signals(QObject):
    status = Signal(str)
    solve_done = Signal()


# ═══════════════════════════════════════════════════════════════
#  CFDLab SINGLETON
# ═══════════════════════════════════════════════════════════════

class CFDLab:
    """Main API — driven by LLM code injection."""

    def __init__(self, viewer, log_widget=None):
        self.viewer = viewer
        self._log = log_widget
        self._signals = _Signals()

        # Geometry
        self.pts = []          # surface points [(x,y), ...]
        self.chord = 1.0

        # Flow conditions
        self.aoa = 0.0         # degrees
        self.v_inf = 1.0
        self.Re = 1e6          # Reynolds number (for annotations)

        # Solver
        self.solver = PanelSolver()

        # Results
        self.info = {}

    def log(self, msg):
        if self._log:
            self._log.append(f"[cfd] {msg}")
        print(f"[cfd] {msg}")

    # ── Loading geometry ──────────────────────────────────────

    def load_airfoil(self, name, n=80):
        """Load a NACA airfoil by code (e.g. 'naca2412' or '0012')."""
        code = name.lower().replace("naca", "").replace(" ", "").replace("-", "")
        full_name = f"NACA {code.upper()}"

        try:
            if len(code) == 4:
                self.pts = naca4(code, n)
            elif len(code) == 5:
                self.pts = naca5(code, n)
            else:
                self.log(f"Unknown NACA code: {code}")
                return self
        except Exception as ex:
            self.log(f"Airfoil error: {ex}")
            return self

        self.chord = max(p[0] for p in self.pts) - min(p[0] for p in self.pts)
        self.info = {'name': full_name, 'source': 'naca', 'n_panels': len(self.pts) - 1}
        self.viewer.set_body(self.pts, full_name)
        self.log(f"Loaded {full_name}: {len(self.pts)-1} panels, chord={self.chord:.3f}")
        return self

    def load_geometry(self, pts):
        """Load custom geometry from point list [(x,y), ...]."""
        self.pts = list(pts)
        self.chord = max(p[0] for p in pts) - min(p[0] for p in pts)
        self.info = {'name': 'Custom', 'source': 'custom', 'n_panels': len(pts) - 1}
        self.viewer.set_body(self.pts, "Custom Geometry")
        self.log(f"Loaded custom geometry: {len(pts)-1} panels")
        return self

    def load_cylinder(self, r=1.0, n=64):
        """Load circular cylinder."""
        self.pts = make_cylinder(r, n)
        self.chord = 2 * r
        self.info = {'name': f'Cylinder R={r}', 'source': 'cylinder', 'n_panels': n}
        self.viewer.set_body(self.pts, f"Cylinder R={r}")
        self.log(f"Loaded cylinder: R={r}, {n} panels")
        return self

    def load_plate(self, chord=1.0, n=40):
        """Load flat plate."""
        self.pts = make_plate(chord, n=n)
        self.chord = chord
        self.info = {'name': 'Flat Plate', 'source': 'plate', 'n_panels': len(self.pts) - 1}
        self.viewer.set_body(self.pts, f"Flat Plate c={chord}")
        self.log(f"Loaded flat plate: chord={chord}")
        return self

    def load_ellipse(self, a=1.0, b=0.3, n=80):
        """Load elliptical section."""
        self.pts = make_ellipse(a, b, n)
        self.chord = 2 * a
        self.info = {'name': f'Ellipse {a}×{b}', 'source': 'ellipse', 'n_panels': n}
        self.viewer.set_body(self.pts, f"Ellipse {a}×{b}")
        self.log(f"Loaded ellipse: a={a}, b={b}")
        return self

    # ── Flow conditions ───────────────────────────────────────

    def set_flow(self, aoa=None, v_inf=None, Re=None):
        """Set flow conditions. aoa in degrees."""
        if aoa is not None:
            self.aoa = aoa
        if v_inf is not None:
            self.v_inf = v_inf
        if Re is not None:
            self.Re = Re
        self.log(f"Flow: α={self.aoa}°, V∞={self.v_inf}, Re={self.Re:.0e}")
        return self

    # ── Solver ────────────────────────────────────────────────

    def solve(self, method="vortex"):
        """Run panel method solver.
        method: 'source' (non-lifting) or 'vortex' (with lift, Kutta condition).
        """
        if not self.pts:
            self.log("No geometry loaded")
            return self

        self.solver.set_panels(self.pts, self.aoa, self.v_inf)

        if method == "source":
            self.solver.solve_source()
            self.log(f"Source panel: Cd={self.solver.Cd:.6f}")
        else:
            self.solver.solve_vortex()
            self.log(f"Vortex panel: Cl={self.solver.Cl:.4f}, Cd={self.solver.Cd:.6f}, Cm={self.solver.Cm:.4f}")

        self.info.update({
            'method': method,
            'aoa': self.aoa,
            'Cl': self.solver.Cl,
            'Cd': self.solver.Cd,
            'Cm': self.solver.Cm,
            'Cp': list(self.solver.Cp),
            'Vt': list(self.solver.Vt),
            'circulation': self.solver.circulation,
            'stagnation': self.solver.stagnation_idx,
            'n_panels': self.solver.n,
        })

        self._signals.solve_done.emit()
        return self

    # ── Visualization ─────────────────────────────────────────

    def color_pressure(self):
        """Color the 3D mesh by Cp distribution."""
        if self.solver.Cp is None:
            self.log("Run cfd.solve() first")
            return self
        self.viewer.set_cp(self.solver.Cp)
        self.log("Mesh colored by Cp")
        return self

    def color_velocity(self):
        """Color the 3D mesh by surface velocity magnitude."""
        if self.solver.Vt is None:
            self.log("Run cfd.solve() first")
            return self
        vt = np.abs(self.solver.Vt)
        vmax = np.max(vt) if np.max(vt) > 0 else 1
        # Map velocity to a pseudo-Cp for coloring
        cp_like = 1.0 - (vt / vmax) ** 2
        self.viewer.set_cp(cp_like)
        self.log("Mesh colored by |V|")
        return self

    def overlay_cp(self, width=380, height=200):
        """Show Cp distribution plot as overlay."""
        if self.solver.Cp is None:
            self.log("Run cfd.solve() first")
            return self

        cp = self.solver.Cp.copy()
        panels = self.solver.panels
        aoa = self.aoa
        method_str = self.info.get('method', '?')

        def draw_cp(painter, w, h):
            ox, oy = w - width - 16, h - height - 16
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 215))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)

            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(40, 55, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter,
                           f"Cp Distribution  α={aoa}°  ({method_str})")

            pad_l, pad_r, pad_t, pad_b = 40, 16, 24, 30
            ax0 = ox + pad_l
            ax1 = ox + width - pad_r
            ay0 = oy + pad_t
            ay1 = oy + height - pad_b
            aw = ax1 - ax0
            ah = ay1 - ay0

            # Axes
            painter.setPen(QPen(QColor(180, 185, 200), 1))
            painter.drawLine(ax0, ay1, ax1, ay1)
            painter.drawLine(ax0, ay0, ax0, ay1)

            # Cp range (inverted y — convention is -Cp up)
            cp_min, cp_max = min(min(cp), -2.0), max(max(cp), 1.5)
            x_coords = [p['xc'] for p in panels]
            x_min, x_max = min(x_coords), max(x_coords)
            x_range = x_max - x_min if x_max > x_min else 1

            def map_x(xv):
                return ax0 + (xv - x_min) / x_range * aw

            def map_y(cpv):
                # Inverted: low Cp at top
                return ay0 + (cpv - cp_max) / (cp_min - cp_max) * ah

            # Grid lines
            painter.setPen(QPen(QColor(220, 225, 235), 1, Qt.DotLine))
            for cpg in [-2, -1, 0, 1]:
                yg = map_y(cpg)
                if ay0 <= yg <= ay1:
                    painter.drawLine(ax0, int(yg), ax1, int(yg))

            # Zero line
            y_zero = map_y(0)
            painter.setPen(QPen(QColor(160, 170, 190), 1))
            painter.drawLine(ax0, int(y_zero), ax1, int(y_zero))

            # Separate upper and lower surface
            n = len(panels)
            mid = n // 2
            upper_x = [panels[i]['xc'] for i in range(mid)]
            upper_cp = [cp[i] for i in range(mid)]
            lower_x = [panels[i]['xc'] for i in range(mid, n)]
            lower_cp = [cp[i] for i in range(mid, n)]

            # Draw upper surface (blue)
            painter.setPen(QPen(QColor(40, 90, 200), 1.8))
            path_u = QPainterPath()
            if upper_x:
                path_u.moveTo(map_x(upper_x[0]), map_y(upper_cp[0]))
                for xi, ci in zip(upper_x[1:], upper_cp[1:]):
                    path_u.lineTo(map_x(xi), map_y(ci))
            painter.drawPath(path_u)

            # Draw lower surface (red)
            painter.setPen(QPen(QColor(200, 60, 40), 1.8))
            path_l = QPainterPath()
            if lower_x:
                path_l.moveTo(map_x(lower_x[0]), map_y(lower_cp[0]))
                for xi, ci in zip(lower_x[1:], lower_cp[1:]):
                    path_l.lineTo(map_x(xi), map_y(ci))
            painter.drawPath(path_l)

            # Labels
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor(100, 115, 140))
            painter.drawText(ax0, ay1 + 4, aw, 14, Qt.AlignCenter, "x/c")
            painter.save()
            painter.translate(ox + 6, ay0 + ah // 2)
            painter.rotate(-90)
            painter.drawText(-30, 0, 60, 14, Qt.AlignCenter, "-Cp")
            painter.restore()

            # Tick labels
            painter.setFont(QFont("Consolas", 6))
            for cpg in [-2, -1, 0, 1]:
                yg = map_y(cpg)
                if ay0 <= yg <= ay1:
                    painter.drawText(ax0 - 28, int(yg) - 6, 24, 12,
                                   Qt.AlignRight | Qt.AlignVCenter, str(cpg))

            # Legend
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor(40, 90, 200))
            painter.drawText(ax0 + 8, oy + height - 20, "● upper")
            painter.setPen(QColor(200, 60, 40))
            painter.drawText(ax0 + 70, oy + height - 20, "● lower")

        self.viewer.add_overlay('cp_dist', draw_cp)
        self.log("Cp overlay added")
        return self

    def overlay_streamlines(self, n=20, density=1.0, domain=(-1.5, 2.5, -1.5, 1.5)):
        """Show streamlines around the body.

        Uses the full source+vortex velocity field with body-interior rejection.

        Args:
            n: number of seed streamlines
            density: step-size scaling (larger = coarser)
            domain: (x_min, x_max, y_min, y_max)
        """
        if self.solver.Cp is None:
            self.log("Run cfd.solve() first")
            return self

        solver = self.solver
        body_pts = self.pts
        x0d, x1d, y0d, y1d = domain

        # Seed upstream
        seed_y = np.linspace(y0d * 0.9, y1d * 0.9, n)
        lines = []
        for sy in seed_y:
            fwd = solver.streamline(x0d, sy, body_pts=body_pts,
                                     ds=0.012 * density, max_steps=1000, forward=True)
            if len(fwd) > 5:
                lines.append(fwd)

        aoa = self.aoa

        def draw_streamlines(painter, w, h):
            margin = 60
            sw = w - 2 * margin
            sh = h - 2 * margin
            x_range = x1d - x0d
            y_range = y1d - y0d

            def to_s(x, y):
                return (margin + (x - x0d) / x_range * sw,
                        margin + (y1d - y) / y_range * sh)

            # Streamlines
            for line in lines:
                if len(line) < 2:
                    continue
                path = QPainterPath()
                sx0, sy0 = to_s(line[0][0], line[0][1])
                path.moveTo(sx0, sy0)
                for lx, ly in line[1:]:
                    sx, sy = to_s(lx, ly)
                    path.lineTo(sx, sy)

                # Color by initial y-position
                mid_y = line[0][1]
                t = max(0, min(1, (mid_y - y0d) / y_range))
                r = int(30 + 60 * (1 - t))
                g = int(100 + 80 * t)
                b = int(190 - 50 * t)
                col = QColor(r, g, b, 170)
                painter.setPen(QPen(col, 1.3))
                painter.drawPath(path)

                # Arrowheads
                step = max(1, len(line) // 5)
                for ai in range(step, len(line) - 1, step):
                    ax_, ay_ = to_s(line[ai][0], line[ai][1])
                    bx_, by_ = to_s(line[ai-1][0], line[ai-1][1])
                    ddx = ax_ - bx_
                    ddy = ay_ - by_
                    dd = math.sqrt(ddx*ddx + ddy*ddy)
                    if dd < 2:
                        continue
                    ddx /= dd
                    ddy /= dd
                    sz = 5
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(col)
                    tri = QPolygonF([
                        QPointF(ax_, ay_),
                        QPointF(ax_ - sz*ddx + sz*0.4*ddy, ay_ - sz*ddy - sz*0.4*ddx),
                        QPointF(ax_ - sz*ddx - sz*0.4*ddy, ay_ - sz*ddy + sz*0.4*ddx),
                    ])
                    painter.drawPolygon(tri)

            # Body silhouette
            painter.setPen(QPen(QColor(20, 30, 50), 2.2))
            painter.setBrush(QColor(180, 195, 210, 200))
            body_path = QPainterPath()
            if body_pts:
                sx0, sy0 = to_s(body_pts[0][0], body_pts[0][1])
                body_path.moveTo(sx0, sy0)
                for bx, by in body_pts[1:]:
                    sx, sy = to_s(bx, by)
                    body_path.lineTo(sx, sy)
                body_path.closeSubpath()
            painter.drawPath(body_path)

            # Flow info label
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor(80, 100, 140))
            ar_x, ar_y = to_s(x0d + 0.15, y1d - 0.15)
            painter.drawText(int(ar_x), int(ar_y), f"V∞ → α={aoa:.1f}°")

        self.viewer.add_overlay('streamlines', draw_streamlines)
        self.log(f"Streamlines: {len(lines)} lines traced")
        return self

    def overlay_forces(self, scale=80):
        """Show lift and drag force arrows."""
        Cl = self.info.get('Cl', 0)
        Cd = self.info.get('Cd', 0)
        aoa = self.aoa

        def draw_forces(painter, w, h):
            cx, cy = w // 2, h // 2

            # Lift arrow (perpendicular to freestream)
            lift_angle = (90 + aoa) * DEG
            lx = cx + scale * Cl * math.cos(lift_angle)
            ly = cy - scale * Cl * math.sin(lift_angle)
            painter.setPen(QPen(QColor(40, 120, 200), 2.5))
            painter.drawLine(cx, cy, int(lx), int(ly))
            # Arrow head
            painter.setBrush(QColor(40, 120, 200))
            painter.setPen(Qt.NoPen)
            dx = lx - cx
            dy = ly - cy
            d = math.sqrt(dx*dx + dy*dy)
            if d > 2:
                dx /= d; dy /= d
                sz = 8
                painter.drawPolygon(QPolygonF([
                    QPointF(lx, ly),
                    QPointF(lx - sz*dx + sz*0.35*dy, ly - sz*dy - sz*0.35*dx),
                    QPointF(lx - sz*dx - sz*0.35*dy, ly - sz*dy + sz*0.35*dx),
                ]))
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(40, 120, 200))
            painter.drawText(int(lx) + 6, int(ly) - 6, f"L (Cl={Cl:.3f})")

            # Drag arrow (along freestream)
            drag_angle = aoa * DEG
            drx = cx + scale * max(abs(Cd), 0.002) * 8 * math.cos(drag_angle)
            dry = cy - scale * max(abs(Cd), 0.002) * 8 * math.sin(drag_angle)
            painter.setPen(QPen(QColor(200, 60, 40), 2.0))
            painter.drawLine(cx, cy, int(drx), int(dry))
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(200, 60, 40))
            painter.drawText(int(drx) + 6, int(dry) + 14, f"D (Cd={Cd:.5f})")

        self.viewer.add_overlay('forces', draw_forces)
        self.log(f"Forces overlay: Cl={Cl:.4f}, Cd={Cd:.6f}")
        return self

    def overlay_polar(self, aoa_range=(-5, 15, 1), width=300, height=200):
        """Sweep AoA and show Cl vs alpha polar plot."""
        if not self.pts:
            self.log("No geometry loaded")
            return self

        a_start, a_end, a_step = aoa_range
        alphas = np.arange(a_start, a_end + a_step * 0.5, a_step)
        cls = []
        cds = []

        for a in alphas:
            s = PanelSolver()
            s.set_panels(self.pts, a, self.v_inf)
            s.solve_vortex()
            cls.append(s.Cl)
            cds.append(s.Cd)

        name = self.info.get('name', '')
        current_aoa = self.aoa

        def draw_polar(painter, w, h):
            ox, oy = 16, h - height - 16
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 215))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)

            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(40, 55, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter,
                           f"Cl vs α — {name}")

            pad = {'l': 40, 'r': 16, 't': 24, 'b': 28}
            ax0 = ox + pad['l']
            ax1 = ox + width - pad['r']
            ay0 = oy + pad['t']
            ay1 = oy + height - pad['b']
            aw = ax1 - ax0
            ah = ay1 - ay0

            cl_min, cl_max = min(cls) - 0.1, max(cls) + 0.1
            a_min, a_max = alphas[0], alphas[-1]

            def mx(a):
                return ax0 + (a - a_min) / (a_max - a_min) * aw

            def my(cl):
                return ay1 - (cl - cl_min) / (cl_max - cl_min) * ah

            # Axes
            painter.setPen(QPen(QColor(180, 185, 200), 1))
            painter.drawLine(ax0, ay1, ax1, ay1)
            painter.drawLine(ax0, ay0, ax0, ay1)

            # Zero line
            if cl_min < 0 < cl_max:
                y0 = my(0)
                painter.setPen(QPen(QColor(200, 205, 215), 1, Qt.DashLine))
                painter.drawLine(ax0, int(y0), ax1, int(y0))

            # Cl curve
            painter.setPen(QPen(QColor(40, 100, 200), 2.0))
            path = QPainterPath()
            path.moveTo(mx(alphas[0]), my(cls[0]))
            for a, cl in zip(alphas[1:], cls[1:]):
                path.lineTo(mx(a), my(cl))
            painter.drawPath(path)

            # Current point
            if a_min <= current_aoa <= a_max:
                idx = int(round((current_aoa - a_min) / a_step))
                if 0 <= idx < len(cls):
                    px, py = mx(alphas[idx]), my(cls[idx])
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(220, 60, 40))
                    painter.drawEllipse(QPointF(px, py), 4, 4)
                    painter.setFont(QFont("Consolas", 7))
                    painter.setPen(QColor(200, 50, 30))
                    painter.drawText(int(px) + 6, int(py) - 4,
                                   f"α={current_aoa}° Cl={cls[idx]:.3f}")

            # Labels
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor(100, 115, 140))
            painter.drawText(ax0, ay1 + 4, aw, 14, Qt.AlignCenter, "α (deg)")
            painter.drawText(ox + 2, ay0, 36, 14, Qt.AlignVCenter, "Cl")

        self.viewer.add_overlay('polar', draw_polar)
        self.log(f"Polar overlay: α={a_start}°..{a_end}°, {len(alphas)} points")
        return self

    def overlay_velocity(self, n_arrows=30):
        """Show velocity vectors on the surface."""
        if self.solver.Vt is None:
            self.log("Run cfd.solve() first")
            return self

        panels = self.solver.panels
        vt = self.solver.Vt
        vmax = max(abs(v) for v in vt) if vt is not None else 1

        def draw_vecs(painter, w, h):
            margin = 80
            sw = w - 2 * margin
            sh = h - 2 * margin
            xs = [p['xc'] for p in panels]
            ys = [p['yc'] for p in panels]
            x_min, x_max = min(xs) - 0.2, max(xs) + 0.2
            y_min, y_max = min(ys) - 0.4, max(ys) + 0.4
            x_range = x_max - x_min
            y_range = y_max - y_min

            def to_s(x, y):
                return margin + (x - x_min) / x_range * sw, margin + (y_max - y) / y_range * sh

            step = max(1, len(panels) // n_arrows)
            for i in range(0, len(panels), step):
                p = panels[i]
                v = vt[i]
                sx, sy = to_s(p['xc'], p['yc'])
                sc = 25 * v / vmax
                ex = sx + sc * p['tx']
                ey = sy - sc * p['ty']
                # Color by velocity
                t = abs(v) / vmax
                col = QColor(int(40 + 180 * t), int(140 - 80 * t), int(200 - 160 * t), 180)
                painter.setPen(QPen(col, 1.5))
                painter.drawLine(int(sx), int(sy), int(ex), int(ey))

        self.viewer.add_overlay('velocity', draw_vecs)
        self.log("Velocity vectors overlay added")
        return self

    def overlay_info(self):
        """Show computed aerodynamic coefficients as overlay."""
        info = dict(self.info)

        def draw_info(painter, w, h):
            ox, oy = w - 200, 12
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 210))
            painter.drawRoundedRect(ox, oy, 188, 110, 8, 8)
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            painter.setPen(QColor(30, 45, 70))
            painter.drawText(ox + 8, oy + 4, 172, 16, Qt.AlignVCenter, "Aerodynamic Data")
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor(60, 80, 110))
            y = oy + 24
            for key, fmt in [('aoa', 'α = {:.1f}°'), ('Cl', 'Cl = {:.4f}'),
                              ('Cd', 'Cd = {:.6f}'), ('Cm', 'Cm = {:.4f}'),
                              ('circulation', 'Γ  = {:.4f}')]:
                val = info.get(key)
                if val is not None:
                    painter.drawText(ox + 12, y, 170, 14, Qt.AlignVCenter, fmt.format(val))
                    y += 15

        self.viewer.add_overlay('info', draw_info)
        return self

    # ── Clearing overlays ─────────────────────────────────────

    def clear_overlays(self):
        """Remove all overlays."""
        self.viewer._overlays.clear()
        self.log("All overlays cleared")
        return self

    def remove_overlay(self, name):
        """Remove a specific overlay."""
        self.viewer.remove_overlay(name)
        return self

    # ── Geometry transforms ───────────────────────────────────

    def scale(self, factor):
        """Scale geometry by factor."""
        cx = sum(p[0] for p in self.pts) / len(self.pts)
        cy = sum(p[1] for p in self.pts) / len(self.pts)
        self.pts = [((p[0] - cx) * factor + cx, (p[1] - cy) * factor + cy) for p in self.pts]
        self.chord *= factor
        self.viewer.set_body(self.pts, self.viewer.body_name)
        return self

    def translate(self, dx, dy):
        """Translate geometry."""
        self.pts = [(p[0] + dx, p[1] + dy) for p in self.pts]
        self.viewer.set_body(self.pts, self.viewer.body_name)
        return self

    def rotate(self, deg):
        """Rotate geometry about centroid."""
        cx = sum(p[0] for p in self.pts) / len(self.pts)
        cy = sum(p[1] for p in self.pts) / len(self.pts)
        r = deg * DEG
        cs, sn = math.cos(r), math.sin(r)
        self.pts = [(cx + (p[0]-cx)*cs - (p[1]-cy)*sn,
                     cy + (p[0]-cx)*sn + (p[1]-cy)*cs) for p in self.pts]
        self.viewer.set_body(self.pts, self.viewer.body_name)
        return self

    def flip(self):
        """Flip y-coordinates (reflect)."""
        self.pts = [(p[0], -p[1]) for p in self.pts]
        self.viewer.set_body(self.pts, self.viewer.body_name)
        return self

    def refine(self, n=200):
        """Re-panel with n points using cosine spacing along arc length."""
        if len(self.pts) < 3:
            return self
        # Compute cumulative arc length
        s = [0]
        for i in range(1, len(self.pts)):
            dx = self.pts[i][0] - self.pts[i-1][0]
            dy = self.pts[i][1] - self.pts[i-1][1]
            s.append(s[-1] + math.sqrt(dx*dx + dy*dy))
        total = s[-1]
        # Cosine spacing for new parameter
        beta = np.linspace(0, np.pi, n)
        s_new = total * 0.5 * (1 - np.cos(beta))
        # Interpolate
        new_pts = []
        j = 0
        for sn in s_new:
            while j < len(s) - 2 and s[j+1] < sn:
                j += 1
            t = (sn - s[j]) / (s[j+1] - s[j]) if s[j+1] != s[j] else 0
            x = self.pts[j][0] + t * (self.pts[j+1][0] - self.pts[j][0])
            y = self.pts[j][1] + t * (self.pts[j+1][1] - self.pts[j][1])
            new_pts.append((x, y))
        self.pts = new_pts
        self.info['n_panels'] = len(new_pts) - 1
        self.viewer.set_body(self.pts, self.viewer.body_name)
        self.log(f"Refined to {n} points")
        return self

    # ── Parametric sweeps ─────────────────────────────────────

    def sweep(self, param, start, end, step=1):
        """Sweep a parameter and collect results.
        param: 'aoa', 'thickness', etc.
        Returns dict with arrays of results.
        """
        vals = np.arange(start, end + step * 0.5, step)
        results = {'values': list(vals), 'Cl': [], 'Cd': [], 'Cm': []}

        for v in vals:
            if param == 'aoa':
                s = PanelSolver()
                s.set_panels(self.pts, v, self.v_inf)
                s.solve_vortex()
                results['Cl'].append(s.Cl)
                results['Cd'].append(s.Cd)
                results['Cm'].append(s.Cm)
            else:
                self.log(f"Unknown sweep parameter: {param}")
                return results

        self.info['sweep'] = results
        self.log(f"Sweep {param}: {start}→{end}, {len(vals)} points")
        return results

    def compare(self, *names, aoa=None):
        """Compare multiple airfoils at current or given AoA.
        Usage: cfd.compare("naca0012", "naca2412", "naca4415")
        """
        if aoa is None:
            aoa = self.aoa
        results = {}
        for name in names:
            try:
                code = name.lower().replace("naca","").replace(" ","")
                if len(code) == 4:
                    pts = naca4(code, 80)
                elif len(code) == 5:
                    pts = naca5(code, 80)
                else:
                    continue
                s = PanelSolver()
                s.set_panels(pts, aoa, self.v_inf)
                s.solve_vortex()
                results[name] = {'Cl': s.Cl, 'Cd': s.Cd, 'Cm': s.Cm, 'Cp': list(s.Cp)}
            except:
                pass
        self.info['comparison'] = results
        self.log(f"Compared {len(results)} airfoils at α={aoa}°")
        return results

    # ── Export ─────────────────────────────────────────────────

    def export_dat(self, path="~/airfoil.dat"):
        """Export airfoil in Selig format."""
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            f.write(f"{self.info.get('name', 'Airfoil')}\n")
            for x, y in self.pts:
                f.write(f"  {x:.6f}  {y:.6f}\n")
        self.log(f"Exported DAT: {path}")
        return path

    def export_csv(self, path="~/cfd_results.csv"):
        """Export panel data as CSV."""
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            f.write("panel,xc,yc,Cp,Vt,length,nx,ny\n")
            for i, p in enumerate(self.solver.panels):
                cp = self.solver.Cp[i] if self.solver.Cp is not None else 0
                vt = self.solver.Vt[i] if self.solver.Vt is not None else 0
                f.write(f"{i},{p['xc']:.6f},{p['yc']:.6f},{cp:.6f},{vt:.6f},"
                        f"{p['length']:.6f},{p['nx']:.6f},{p['ny']:.6f}\n")
        self.log(f"Exported CSV: {path}")
        return path

    # ── Analysis helpers ──────────────────────────────────────

    def summary(self):
        """Print a summary of current state."""
        lines = [
            f"Body: {self.info.get('name', 'None')}",
            f"Panels: {self.info.get('n_panels', 0)}",
            f"AoA: {self.aoa}°  V∞: {self.v_inf}",
        ]
        if 'Cl' in self.info:
            lines.append(f"Cl = {self.info['Cl']:.4f}")
            lines.append(f"Cd = {self.info['Cd']:.6f}")
            lines.append(f"Cm = {self.info['Cm']:.4f}")
            lines.append(f"L/D = {self.info['Cl']/max(abs(self.info['Cd']),1e-10):.1f}")
        self.log("\n".join(lines))
        return self.info


# ═══════════════════════════════════════════════════════════════
#  LIGHT-MODE STYLESHEET
# ═══════════════════════════════════════════════════════════════

_SS = """
QWidget{background:rgba(248,250,252,220);color:#1e293b;font-family:'Consolas','Menlo',monospace;font-size:11px}
QPushButton{background:rgba(241,245,249,240);border:1px solid #cbd5e1;border-radius:5px;padding:6px 10px;color:#1e40af;font-weight:bold;font-size:10px}
QPushButton:hover{background:rgba(219,234,254,250);border-color:#93c5fd}
QPushButton:checked{background:rgba(59,130,246,25);border-color:#60a5fa;color:#1d4ed8}
QSlider::groove:horizontal{height:4px;background:#e2e8f0;border-radius:2px}
QSlider::handle:horizontal{background:#3b82f6;width:14px;margin:-5px 0;border-radius:7px}
QComboBox{background:rgba(248,250,252,240);border:1px solid #cbd5e1;border-radius:4px;padding:5px 8px}
QComboBox QAbstractItemView{background:white;border:1px solid #cbd5e1;selection-background-color:#dbeafe}
QTextEdit{background:rgba(248,250,252,220);border:1px solid #e2e8f0;border-radius:4px;font-size:10px;color:#475569;padding:6px}
QCheckBox{spacing:8px;color:#475569}
QTabWidget::pane{border:1px solid #e2e8f0;background:rgba(255,255,255,200);border-top:none}
QTabBar::tab{background:rgba(241,245,249,200);color:#64748b;padding:7px 14px;font-size:10px;font-weight:bold;border:1px solid #e2e8f0;border-bottom:none}
QTabBar::tab:selected{background:rgba(255,255,255,240);color:#1e40af;border-bottom:2px solid #3b82f6}
QListWidget{background:rgba(248,250,252,220);border:1px solid #e2e8f0;border-radius:4px;font-size:10px;color:#475569}
QListWidget::item:selected{background:#dbeafe;color:#1e40af}
QLabel{background:transparent}
QScrollArea{border:none;background:transparent}
QLineEdit{background:rgba(248,250,252,240);border:1px solid #cbd5e1;border-radius:4px;padding:5px 8px}
QDoubleSpinBox,QSpinBox{background:rgba(248,250,252,240);border:1px solid #cbd5e1;border-radius:4px;padding:4px 6px}
"""

def _lbl(text):
    l = QLabel(text.upper())
    l.setStyleSheet("font-size:9px;letter-spacing:1.5px;color:#94a3b8;font-weight:bold;padding:2px 0;background:transparent")
    return l


# ═══════════════════════════════════════════════════════════════
#  CFDLabUI — All UI assembly and signal wiring in one class
# ═══════════════════════════════════════════════════════════════

class CFDLabUI:
    """Builds the entire CFDLab UI, wires signals, and exposes the
    ``cfd`` singleton and its aliases."""

    def __init__(self):
        # ── Main widget ──
        self.main_widget = QWidget()
        self.main_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ── Side panel ──
        self.panel = QWidget()
        self.panel.setFixedWidth(300)
        self.panel.setStyleSheet(_SS)
        self.panel.setAttribute(Qt.WA_TranslucentBackground, True)
        self.ps = QScrollArea()
        self.ps.setWidgetResizable(True)
        self.ps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ps.setStyleSheet(
            "QScrollArea{border:none;background:transparent}"
            "QScrollBar:vertical{width:5px;background:transparent}"
            "QScrollBar::handle:vertical{background:#94a3b8;border-radius:2px;min-height:30px}")
        self.inner = QWidget()
        self.inner.setAttribute(Qt.WA_TranslucentBackground, True)
        self.lay = QVBoxLayout(self.inner)
        self.lay.setSpacing(4)
        self.lay.setContentsMargins(10, 10, 10, 10)

        self._build_header()
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(_SS)
        self._build_tab_geometry()
        self._build_tab_flow()
        self._build_tab_visualize()
        self._build_tab_log()
        self.lay.addWidget(self.tabs)

        self.ps.setWidget(self.inner)
        playout = QVBoxLayout(self.panel)
        playout.setContentsMargins(0, 0, 0, 0)
        playout.addWidget(self.ps)

        # ── Viewer ──
        self.viewer = CFDViewer()
        self.viewer.setStyleSheet("background:transparent")
        self.main_layout.addWidget(self.panel)
        self.main_layout.addWidget(self.viewer, 1)

        # ── Singleton & aliases ──
        self.cfd = CFDLab(self.viewer, self.log_edit)
        self.flow = self.cfd

        # ── Wire signals ──
        self._wire_signals()

        # ── Load default preset ──
        self._load_preset()

    # ────────────────────────────────────────────────────────────
    #  HEADER
    # ────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = QWidget()
        hdr.setAttribute(Qt.WA_TranslucentBackground, True)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 4)
        ic = QLabel("\U0001f32c")  # wind emoji
        ic.setStyleSheet("font-size:20px;background:rgba(219,234,254,200);border:1px solid #bfdbfe;"
                          "border-radius:7px;padding:3px 7px")
        nw = QWidget()
        nw.setAttribute(Qt.WA_TranslucentBackground, True)
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(6, 0, 0, 0)
        nl.setSpacing(0)
        _title_lbl = QLabel("CFDLab")
        _title_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#0f172a;background:transparent")
        _sub_lbl = QLabel("FLUID DYNAMICS WORKBENCH")
        _sub_lbl.setStyleSheet("font-size:7px;letter-spacing:2px;color:#94a3b8;background:transparent")
        nl.addWidget(_title_lbl)
        nl.addWidget(_sub_lbl)
        hl.addWidget(ic)
        hl.addWidget(nw)
        hl.addStretch()
        self.lay.addWidget(hdr)

    # ────────────────────────────────────────────────────────────
    #  TAB 1: GEOMETRY
    # ────────────────────────────────────────────────────────────

    def _build_tab_geometry(self):
        t1 = QWidget()
        t1.setAttribute(Qt.WA_TranslucentBackground, True)
        t1l = QVBoxLayout(t1)
        t1l.setSpacing(5)
        t1l.setContentsMargins(6, 8, 6, 6)

        t1l.addWidget(_lbl("Airfoil Presets"))
        self.foil_combo = QComboBox()
        self.foil_combo.addItems(list(NACA_PRESETS.keys()))
        t1l.addWidget(self.foil_combo)

        t1l.addWidget(_lbl("Custom NACA Code"))
        self.naca_edit = QLineEdit()
        self.naca_edit.setPlaceholderText("e.g. 2412 or 23015")
        t1l.addWidget(self.naca_edit)

        t1l.addWidget(_lbl("Panel Count"))
        self.panel_spin = QSpinBox()
        self.panel_spin.setRange(20, 300)
        self.panel_spin.setValue(80)
        self.panel_spin.setSingleStep(10)
        t1l.addWidget(self.panel_spin)

        self.load_foil_btn = QPushButton("Load Airfoil")
        t1l.addWidget(self.load_foil_btn)

        t1l.addWidget(_lbl("Body Shapes"))
        shape_row = QWidget()
        shape_row.setAttribute(Qt.WA_TranslucentBackground, True)
        srl = QHBoxLayout(shape_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(2)
        self.cyl_btn = QPushButton("Cylinder")
        self.plate_btn = QPushButton("Flat Plate")
        self.elli_btn = QPushButton("Ellipse")
        srl.addWidget(self.cyl_btn)
        srl.addWidget(self.plate_btn)
        srl.addWidget(self.elli_btn)
        t1l.addWidget(shape_row)

        t1l.addStretch()
        self.tabs.addTab(t1, "Geometry")

    # ────────────────────────────────────────────────────────────
    #  TAB 2: FLOW & SOLVE
    # ────────────────────────────────────────────────────────────

    def _build_tab_flow(self):
        t2 = QWidget()
        t2.setAttribute(Qt.WA_TranslucentBackground, True)
        t2l = QVBoxLayout(t2)
        t2l.setSpacing(5)
        t2l.setContentsMargins(6, 8, 6, 6)

        t2l.addWidget(_lbl("Angle of Attack (deg)"))
        self.aoa_slider = QSlider(Qt.Horizontal)
        self.aoa_slider.setRange(-150, 200)  # -15.0 to 20.0 in tenths
        self.aoa_slider.setValue(0)
        self.aoa_lbl = QLabel("α = 0.0°")
        self.aoa_lbl.setStyleSheet("font-size:11px;font-weight:bold;color:#1e40af;background:transparent")
        t2l.addWidget(self.aoa_slider)
        t2l.addWidget(self.aoa_lbl)

        t2l.addWidget(_lbl("Freestream Velocity"))
        self.vinf_spin = QDoubleSpinBox()
        self.vinf_spin.setRange(0.1, 100.0)
        self.vinf_spin.setValue(1.0)
        self.vinf_spin.setSingleStep(0.1)
        t2l.addWidget(self.vinf_spin)

        t2l.addWidget(_lbl("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["vortex (lifting)", "source (non-lifting)"])
        t2l.addWidget(self.method_combo)

        self.solve_btn = QPushButton("▶  SOLVE")
        self.solve_btn.setStyleSheet("""
            QPushButton{background:rgba(37,99,235,220);color:white;border:none;
            border-radius:6px;padding:10px;font-size:12px;font-weight:bold}
            QPushButton:hover{background:rgba(29,78,216,240)}
        """)
        t2l.addWidget(self.solve_btn)

        t2l.addWidget(_lbl("Results"))
        self.result_edit = QTextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setMinimumHeight(90)
        self.result_edit.setPlaceholderText("Run solver to see results...")
        t2l.addWidget(self.result_edit)

        t2l.addStretch()
        self.tabs.addTab(t2, "Flow")

    # ────────────────────────────────────────────────────────────
    #  TAB 3: VISUALIZATION
    # ────────────────────────────────────────────────────────────

    def _build_tab_visualize(self):
        t3 = QWidget()
        t3.setAttribute(Qt.WA_TranslucentBackground, True)
        t3l = QVBoxLayout(t3)
        t3l.setSpacing(5)
        t3l.setContentsMargins(6, 8, 6, 6)

        t3l.addWidget(_lbl("Color Mesh"))
        color_row = QWidget()
        color_row.setAttribute(Qt.WA_TranslucentBackground, True)
        crl = QHBoxLayout(color_row)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.setSpacing(2)
        self.cp_color_btn = QPushButton("Pressure (Cp)")
        self.vel_color_btn = QPushButton("Velocity")
        self.reset_color_btn = QPushButton("Reset")
        crl.addWidget(self.cp_color_btn)
        crl.addWidget(self.vel_color_btn)
        crl.addWidget(self.reset_color_btn)
        t3l.addWidget(color_row)

        t3l.addWidget(_lbl("Overlays"))
        self.cp_overlay_btn = QPushButton("Cp Distribution")
        self.stream_btn = QPushButton("Streamlines")
        self.forces_btn = QPushButton("Force Arrows")
        self.polar_btn = QPushButton("Cl vs α Polar")
        self.info_btn = QPushButton("Aero Data Panel")
        self.clear_ov_btn = QPushButton("Clear All Overlays")
        for b in [self.cp_overlay_btn, self.stream_btn, self.forces_btn,
                  self.polar_btn, self.info_btn, self.clear_ov_btn]:
            t3l.addWidget(b)

        t3l.addWidget(_lbl("Export"))
        export_row = QWidget()
        export_row.setAttribute(Qt.WA_TranslucentBackground, True)
        erl = QHBoxLayout(export_row)
        erl.setContentsMargins(0, 0, 0, 0)
        erl.setSpacing(2)
        self.dat_btn = QPushButton("Export .dat")
        self.csv_btn = QPushButton("Export .csv")
        self.png_btn = QPushButton("Screenshot")
        erl.addWidget(self.dat_btn)
        erl.addWidget(self.csv_btn)
        erl.addWidget(self.png_btn)
        t3l.addWidget(export_row)

        t3l.addStretch()
        self.tabs.addTab(t3, "Visualize")

    # ────────────────────────────────────────────────────────────
    #  TAB 4: LOG
    # ────────────────────────────────────────────────────────────

    def _build_tab_log(self):
        t4 = QWidget()
        t4.setAttribute(Qt.WA_TranslucentBackground, True)
        t4l = QVBoxLayout(t4)
        t4l.setSpacing(5)
        t4l.setContentsMargins(6, 8, 6, 6)
        t4l.addWidget(_lbl("Computation Log"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(200)
        self.log_edit.setPlainText("[CFDLab] Initialised\n[CFDLab] Panel method solver ready\n")
        t4l.addWidget(self.log_edit)
        t4l.addStretch()
        self.tabs.addTab(t4, "Log")

    # ────────────────────────────────────────────────────────────
    #  SIGNAL WIRING
    # ────────────────────────────────────────────────────────────

    def _wire_signals(self):
        self.foil_combo.currentTextChanged.connect(lambda t: self._load_preset())
        self.load_foil_btn.clicked.connect(self._load_custom)
        self.naca_edit.returnPressed.connect(self._load_custom)

        self.cyl_btn.clicked.connect(lambda: self.cfd.load_cylinder())
        self.plate_btn.clicked.connect(lambda: self.cfd.load_plate())
        self.elli_btn.clicked.connect(lambda: self.cfd.load_ellipse())

        self.aoa_slider.valueChanged.connect(self._on_aoa_changed)
        self.vinf_spin.valueChanged.connect(lambda v: self.cfd.set_flow(v_inf=v))

        self.solve_btn.clicked.connect(self._solve)

        self.cp_color_btn.clicked.connect(lambda: self.cfd.color_pressure())
        self.vel_color_btn.clicked.connect(lambda: self.cfd.color_velocity())
        self.reset_color_btn.clicked.connect(lambda: (setattr(self.cfd.viewer, 'cp_values', None),
                                                       self.cfd.viewer._rebuild_mesh()))
        self.cp_overlay_btn.clicked.connect(lambda: self.cfd.overlay_cp())
        self.stream_btn.clicked.connect(lambda: self.cfd.overlay_streamlines())
        self.forces_btn.clicked.connect(lambda: self.cfd.overlay_forces())
        self.polar_btn.clicked.connect(lambda: self.cfd.overlay_polar())
        self.info_btn.clicked.connect(lambda: self.cfd.overlay_info())
        self.clear_ov_btn.clicked.connect(lambda: self.cfd.clear_overlays())

        self.dat_btn.clicked.connect(lambda: self.cfd.export_dat())
        self.csv_btn.clicked.connect(lambda: self.cfd.export_csv())
        self.png_btn.clicked.connect(lambda: self.cfd.viewer.screenshot(
            os.path.expanduser("~/cfd_screenshot.png")))

    # ────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ────────────────────────────────────────────────────────────

    def _update_results(self):
        """Sync results panel after solve."""
        lines = []
        if 'Cl' in self.cfd.info:
            lines.append(f"Cl  = {self.cfd.info['Cl']:.4f}")
            lines.append(f"Cd  = {self.cfd.info['Cd']:.6f}")
            lines.append(f"Cm  = {self.cfd.info['Cm']:.4f}")
            cd = self.cfd.info['Cd']
            if abs(cd) > 1e-10:
                lines.append(f"L/D = {self.cfd.info['Cl']/cd:.1f}")
            lines.append(f"Γ   = {self.cfd.info.get('circulation', 0):.4f}")
            lines.append(f"α   = {self.cfd.info.get('aoa', 0):.1f}°")
        self.result_edit.setPlainText("\n".join(lines))

    def _load_preset(self):
        name = self.foil_combo.currentText()
        if name in ("cylinder", "flat_plate", "ellipse"):
            if name == "cylinder":
                self.cfd.load_cylinder()
            elif name == "flat_plate":
                self.cfd.load_plate()
            else:
                self.cfd.load_ellipse()
        else:
            self.cfd.load_airfoil(name, self.panel_spin.value())

    def _load_custom(self):
        code = self.naca_edit.text().strip()
        if code:
            self.cfd.load_airfoil(code, self.panel_spin.value())

    def _on_aoa_changed(self, val):
        deg = val / 10.0
        self.aoa_lbl.setText(f"α = {deg:.1f}°")
        self.cfd.set_flow(aoa=deg)

    def _solve(self):
        method = "vortex" if self.method_combo.currentIndex() == 0 else "source"
        self.cfd.solve(method)
        self._update_results()
        self.cfd.color_pressure()


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE
# ═══════════════════════════════════════════════════════════════

cfd_ui = CFDLabUI()
cfd_main_widget = cfd_ui.main_widget
cfd = cfd_ui.cfd
cfd_flow = cfd_ui.flow
cfd_viewer = cfd_ui.viewer

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

cfd_proxy = graphics_scene.addWidget(cfd_main_widget)
cfd_proxy.setPos(0, 0)
cfd_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
cfd_shadow = QGraphicsDropShadowEffect()
cfd_shadow.setBlurRadius(60)
cfd_shadow.setOffset(45, 45)
cfd_shadow.setColor(QColor(0, 0, 0, 120))
cfd_proxy.setGraphicsEffect(cfd_shadow)
cfd_main_widget.resize(1400, 850)

# Center in current view
_vr = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
cfd_proxy.setPos(_vr.center().x() - cfd_main_widget.width() / 2,
             _vr.center().y() - cfd_main_widget.height() / 2)
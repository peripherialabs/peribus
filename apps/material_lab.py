"""
MatLab — Materials Science & Solid State Physics Workbench for Rio
═══════════════════════════════════════════════════════════════════

ARCHITECTURE: This file is the complete runtime. It registers a `crystal`
singleton into the namespace. The LLM sees this source as context and
injects short code snippets to drive it.

EXAMPLES (what the LLM would output):
    matlab_app.matlab_crystal.load("silicon")                     # load preset crystal
    matlab_app.matlab_crystal.load("nacl")                        # rock salt
    matlab_app.matlab_crystal.load_cif("/path/to/perovskite.cif") # from CIF file
    matlab_app.matlab_crystal.load_poscar("/path/to/POSCAR")      # VASP format
    matlab_app.matlab_crystal.supercell(2, 2, 2)                  # expand to 2×2×2
    matlab_app.matlab_crystal.style("ballstick")                  # ballstick/spacefill/polyhedra
    matlab_app.matlab_crystal.select(0)                           # highlight atom 0
    matlab_app.matlab_crystal.measure(0, 1)                       # distance between atoms
    matlab_app.matlab_crystal.show_unitcell(True)                 # toggle unit cell wireframe
    matlab_app.matlab_crystal.show_axes(True)                     # toggle lattice vector arrows
    matlab_app.matlab_crystal.cleave(1, 1, 0, layers=4)          # cleave (110) surface slab
    matlab_app.matlab_crystal.vacancy(3)                          # remove atom 3 (point defect)
    matlab_app.matlab_crystal.substitute(3, "Al")                 # substitute atom 3 with Al

    ## ANALYSIS (ASE / numpy built-in):
    matlab_app.matlab_crystal.calculate()                         # LJ/Morse quick energy
    matlab_app.matlab_crystal.calculate("EMT")                    # EMT potential (metals)
    matlab_app.matlab_crystal.radial_distribution(nbins=100)      # compute & overlay g(r)
    matlab_app.matlab_crystal.xrd(wavelength=1.5406)              # compute & overlay powder XRD
    matlab_app.matlab_crystal.overlay_dos(energies, dos)          # overlay density of states
    matlab_app.matlab_crystal.overlay_bands(kpath, bands)         # overlay band structure

    ## After calculate(), matlab_app.matlab_crystal.info contains:
    matlab_app.matlab_crystal.info['energy']          # total energy in eV
    matlab_app.matlab_crystal.info['energy_per_atom'] # eV/atom
    matlab_app.matlab_crystal.info['volume']          # cell volume in Å³
    matlab_app.matlab_crystal.info['density']         # g/cm³
    matlab_app.matlab_crystal.info['spacegroup']      # detected space group (if spglib)
    matlab_app.matlab_crystal.info['formula']         # reduced formula
    matlab_app.matlab_crystal.info['lattice']         # lattice parameters [a, b, c, α, β, γ]
    matlab_app.matlab_crystal.info['n_atoms']         # number of atoms

    ## EXPORT:
    matlab_app.matlab_crystal.export_poscar("/tmp/POSCAR")        # VASP format
    matlab_app.matlab_crystal.export_cif("/tmp/structure.cif")     # CIF format
    matlab_app.matlab_crystal.export_xyz("/tmp/crystal.xyz")       # XYZ snapshot
    matlab_app.matlab_crystal.export_qe_input("/tmp/pw.in")       # Quantum ESPRESSO input

VIEWER API (lower level):
    matlab_app.matlab_viewer.set_crystal(atoms, cell, name)
    matlab_app.matlab_viewer.set_style(style)
    matlab_app.matlab_viewer.add_overlay(name, fn)
    matlab_app.matlab_viewer.remove_overlay(name)
    matlab_app.matlab_viewer.cam_dist = 12.0
    matlab_app.matlab_viewer.rot_x, matlab_app.matlab_viewer.rot_y
    matlab_app.matlab_viewer.screenshot(path)

NAMESPACE: After this file runs, these are available:
    matlab_app  — MaterialLabApp instance (main entry point)
    matlab_app.matlab_crystal — MatLab singleton (main API)
    matlab_app.matlab_viewer  — the CrystalViewer widget
    matlab_app.matlab_mat     — alias for matlab_app.matlab_crystal
    ELEMENTS    — element data dict
    LATTICES    — preset crystal structures
    All PySide6/Qt, numpy, moderngl, glm from parser namespace
"""

import math
import os
import re
import subprocess
import threading
import numpy as np
from collections import OrderedDict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QCheckBox, QTabWidget, QTextEdit,
    QScrollArea, QListWidget, QLineEdit, QGraphicsItem,
    QGraphicsDropShadowEffect, QSizePolicy, QSpinBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage, QLinearGradient, QPainterPath
)

import moderngl
import struct

# ── GLM compatibility shim ─────────────────────────────────────
# Works with PyGLM, or falls back to pure-numpy if glm is missing
# or partial (some envs inject a stripped 'glm' module).

class _Vec3:
    __slots__ = ('x','y','z')
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)
    def __mul__(self, s): return _Vec3(self.x*s, self.y*s, self.z*s)
    def __rmul__(self, s): return self.__mul__(s)
    def __add__(self, o): return _Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return _Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def to_bytes(self): return struct.pack('3f', self.x, self.y, self.z)

class _Mat4:
    """Column-major 4×4 stored as list-of-4-lists (columns)."""
    __slots__ = ('cols',)
    def __init__(self, cols=None):
        if cols is None:
            self.cols = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        else:
            self.cols = [list(c) for c in cols]
    def __getitem__(self, i): return self.cols[i]
    def __mul__(self, other):
        if isinstance(other, _Mat4):
            r = [[0]*4 for _ in range(4)]
            for c in range(4):
                for rr in range(4):
                    r[c][rr] = sum(self.cols[k][rr]*other.cols[c][k] for k in range(4))
            return _Mat4(r)
        return NotImplemented
    def to_bytes(self):
        flat = []
        for c in self.cols:
            for v in c: flat.append(v)
        return struct.pack('16f', *flat)
    def write(self, *a, **kw):
        """For uniform.write() compatibility — returns bytes."""
        return self.to_bytes()

def _glm_vec3(x=0.0, y=0.0, z=0.0):
    if isinstance(x, (int,float)) and isinstance(y, (int,float)):
        return _Vec3(x, y, z)
    return _Vec3(float(x), float(y), float(z))

def _glm_mat4(v=1.0):
    if isinstance(v, (int,float)):
        return _Mat4([[v,0,0,0],[0,v,0,0],[0,0,v,0],[0,0,0,1]])
    return _Mat4()

def _glm_translate(m, v):
    r = _Mat4([list(c) for c in m.cols])
    r.cols[3][0] = m.cols[0][0]*v.x + m.cols[1][0]*v.y + m.cols[2][0]*v.z + m.cols[3][0]
    r.cols[3][1] = m.cols[0][1]*v.x + m.cols[1][1]*v.y + m.cols[2][1]*v.z + m.cols[3][1]
    r.cols[3][2] = m.cols[0][2]*v.x + m.cols[1][2]*v.y + m.cols[2][2]*v.z + m.cols[3][2]
    r.cols[3][3] = m.cols[0][3]*v.x + m.cols[1][3]*v.y + m.cols[2][3]*v.z + m.cols[3][3]
    return r

def _glm_rotate(m, angle, axis):
    c = math.cos(angle); s = math.sin(angle); t = 1.0 - c
    ax = axis.x; ay = axis.y; az = axis.z
    l = math.sqrt(ax*ax+ay*ay+az*az)+1e-12
    ax/=l; ay/=l; az/=l
    rot = _Mat4([
        [t*ax*ax+c, t*ax*ay+s*az, t*ax*az-s*ay, 0],
        [t*ax*ay-s*az, t*ay*ay+c, t*ay*az+s*ax, 0],
        [t*ax*az+s*ay, t*ay*az-s*ax, t*az*az+c, 0],
        [0, 0, 0, 1]])
    return m * rot

def _glm_perspective(fovy, aspect, near, far):
    f = 1.0/math.tan(fovy/2.0)
    nf = near - far
    return _Mat4([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/nf, -1],
        [0, 0, 2*far*near/nf, 0]])

def _glm_lookAt(eye, center, up):
    fx=center.x-eye.x; fy=center.y-eye.y; fz=center.z-eye.z
    fl=math.sqrt(fx*fx+fy*fy+fz*fz)+1e-12
    fx/=fl; fy/=fl; fz/=fl
    sx=fy*up.z-fz*up.y; sy=fz*up.x-fx*up.z; sz=fx*up.y-fy*up.x
    sl=math.sqrt(sx*sx+sy*sy+sz*sz)+1e-12
    sx/=sl; sy/=sl; sz/=sl
    ux=sy*fz-sz*fy; uy=sz*fx-sx*fz; uz=sx*fy-sy*fx
    return _Mat4([
        [sx, ux, -fx, 0],
        [sy, uy, -fy, 0],
        [sz, uz, -fz, 0],
        [-(sx*eye.x+sy*eye.y+sz*eye.z),
         -(ux*eye.x+uy*eye.y+uz*eye.z),
         (fx*eye.x+fy*eye.y+fz*eye.z), 1]])

def _glm_normalize(v):
    l = math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)+1e-12
    return _Vec3(v.x/l, v.y/l, v.z/l)

def _glm_dot(a, b): return a.x*b.x+a.y*b.y+a.z*b.z
def _glm_cross(a, b): return _Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x)
def _glm_length(v): return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)
def _glm_radians(d): return d * math.pi / 180.0

class _GLMShim:
    """Drop-in for PyGLM subset used by this module."""
    vec3 = staticmethod(_glm_vec3)
    mat4 = staticmethod(_glm_mat4)
    translate = staticmethod(_glm_translate)
    rotate = staticmethod(_glm_rotate)
    perspective = staticmethod(_glm_perspective)
    lookAt = staticmethod(_glm_lookAt)
    normalize = staticmethod(_glm_normalize)
    dot = staticmethod(_glm_dot)
    cross = staticmethod(_glm_cross)
    length = staticmethod(_glm_length)
    radians = staticmethod(_glm_radians)

# Try real PyGLM first, fall back to shim
try:
    import glm as _glm_real
    _glm_real.translate  # probe for the function we need
    glm = _glm_real
    _USE_SHIM = False
except (ImportError, AttributeError):
    glm = _GLMShim()
    _USE_SHIM = True

def _uw(uniform, obj):
    """Write a glm object to a moderngl uniform, handling both PyGLM and shim."""
    if _USE_SHIM:
        uniform.write(obj.to_bytes())
    else:
        uniform.write(obj)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

BOHR = 0.529177249

def _hex(h):
    return ((h >> 16) & 0xFF) / 255.0, ((h >> 8) & 0xFF) / 255.0, (h & 0xFF) / 255.0

ELEMENTS = {
    'H':  {'color':0xEEEEEE,'r':0.31,'cov':0.31,'mass':1.008,'Z':1,'name':'Hydrogen'},
    'He': {'color':0xD9FFFF,'r':0.28,'cov':0.28,'mass':4.003,'Z':2,'name':'Helium'},
    'Li': {'color':0xCC80FF,'r':1.52,'cov':1.28,'mass':6.941,'Z':3,'name':'Lithium'},
    'Be': {'color':0xC2FF00,'r':1.12,'cov':0.96,'mass':9.012,'Z':4,'name':'Beryllium'},
    'B':  {'color':0xFFB5B5,'r':0.87,'cov':0.84,'mass':10.81,'Z':5,'name':'Boron'},
    'C':  {'color':0x333333,'r':0.77,'cov':0.77,'mass':12.01,'Z':6,'name':'Carbon'},
    'N':  {'color':0x2040D8,'r':0.75,'cov':0.71,'mass':14.01,'Z':7,'name':'Nitrogen'},
    'O':  {'color':0xDD0000,'r':0.73,'cov':0.66,'mass':16.00,'Z':8,'name':'Oxygen'},
    'F':  {'color':0x70D040,'r':0.71,'cov':0.57,'mass':19.00,'Z':9,'name':'Fluorine'},
    'Na': {'color':0xAB5CF2,'r':1.86,'cov':1.66,'mass':22.99,'Z':11,'name':'Sodium'},
    'Mg': {'color':0x8AFF00,'r':1.60,'cov':1.41,'mass':24.31,'Z':12,'name':'Magnesium'},
    'Al': {'color':0xBFA6A6,'r':1.43,'cov':1.21,'mass':26.98,'Z':13,'name':'Aluminum'},
    'Si': {'color':0xF0C8A0,'r':1.17,'cov':1.11,'mass':28.09,'Z':14,'name':'Silicon'},
    'P':  {'color':0xFF8000,'r':1.10,'cov':1.07,'mass':30.97,'Z':15,'name':'Phosphorus'},
    'S':  {'color':0xDDDD00,'r':1.04,'cov':1.05,'mass':32.07,'Z':16,'name':'Sulfur'},
    'Cl': {'color':0x1FF01F,'r':0.99,'cov':1.02,'mass':35.45,'Z':17,'name':'Chlorine'},
    'K':  {'color':0x8F40D4,'r':2.27,'cov':2.03,'mass':39.10,'Z':19,'name':'Potassium'},
    'Ca': {'color':0x3DFF00,'r':1.97,'cov':1.76,'mass':40.08,'Z':20,'name':'Calcium'},
    'Ti': {'color':0xBFC2C7,'r':1.47,'cov':1.60,'mass':47.87,'Z':22,'name':'Titanium'},
    'V':  {'color':0xA6A6AB,'r':1.34,'cov':1.53,'mass':50.94,'Z':23,'name':'Vanadium'},
    'Cr': {'color':0x8A99C7,'r':1.28,'cov':1.39,'mass':52.00,'Z':24,'name':'Chromium'},
    'Mn': {'color':0x9C7AC7,'r':1.27,'cov':1.39,'mass':54.94,'Z':25,'name':'Manganese'},
    'Fe': {'color':0xE06633,'r':1.26,'cov':1.32,'mass':55.85,'Z':26,'name':'Iron'},
    'Co': {'color':0xF090A0,'r':1.25,'cov':1.26,'mass':58.93,'Z':27,'name':'Cobalt'},
    'Ni': {'color':0x50D050,'r':1.24,'cov':1.24,'mass':58.69,'Z':28,'name':'Nickel'},
    'Cu': {'color':0xC88033,'r':1.28,'cov':1.32,'mass':63.55,'Z':29,'name':'Copper'},
    'Zn': {'color':0x7D80B0,'r':1.34,'cov':1.22,'mass':65.38,'Z':30,'name':'Zinc'},
    'Ga': {'color':0xC28F8F,'r':1.35,'cov':1.22,'mass':69.72,'Z':31,'name':'Gallium'},
    'Ge': {'color':0x668F8F,'r':1.22,'cov':1.20,'mass':72.63,'Z':32,'name':'Germanium'},
    'As': {'color':0xBD80E3,'r':1.19,'cov':1.19,'mass':74.92,'Z':33,'name':'Arsenic'},
    'Se': {'color':0xFFA100,'r':1.20,'cov':1.20,'mass':78.96,'Z':34,'name':'Selenium'},
    'Br': {'color':0xA62929,'r':1.20,'cov':1.20,'mass':79.90,'Z':35,'name':'Bromine'},
    'Sr': {'color':0x00FF00,'r':2.15,'cov':1.95,'mass':87.62,'Z':38,'name':'Strontium'},
    'Y':  {'color':0x94FFFF,'r':1.80,'cov':1.90,'mass':88.91,'Z':39,'name':'Yttrium'},
    'Zr': {'color':0x94E0E0,'r':1.60,'cov':1.75,'mass':91.22,'Z':40,'name':'Zirconium'},
    'Nb': {'color':0x73C2C9,'r':1.46,'cov':1.64,'mass':92.91,'Z':41,'name':'Niobium'},
    'Mo': {'color':0x54B5B5,'r':1.39,'cov':1.54,'mass':95.95,'Z':42,'name':'Molybdenum'},
    'Ru': {'color':0x248F8F,'r':1.34,'cov':1.46,'mass':101.1,'Z':44,'name':'Ruthenium'},
    'Pd': {'color':0x006985,'r':1.37,'cov':1.39,'mass':106.4,'Z':46,'name':'Palladium'},
    'Ag': {'color':0xC0C0C0,'r':1.44,'cov':1.45,'mass':107.9,'Z':47,'name':'Silver'},
    'Sn': {'color':0x668080,'r':1.39,'cov':1.39,'mass':118.7,'Z':50,'name':'Tin'},
    'I':  {'color':0x940094,'r':1.39,'cov':1.39,'mass':126.9,'Z':53,'name':'Iodine'},
    'Ba': {'color':0x00C900,'r':2.17,'cov':2.15,'mass':137.3,'Z':56,'name':'Barium'},
    'La': {'color':0x70D4FF,'r':1.87,'cov':2.07,'mass':138.9,'Z':57,'name':'Lanthanum'},
    'W':  {'color':0x2194D6,'r':1.39,'cov':1.62,'mass':183.8,'Z':74,'name':'Tungsten'},
    'Pt': {'color':0xD0D0E0,'r':1.39,'cov':1.36,'mass':195.1,'Z':78,'name':'Platinum'},
    'Au': {'color':0xFFD123,'r':1.44,'cov':1.36,'mass':197.0,'Z':79,'name':'Gold'},
    'Pb': {'color':0x575961,'r':1.75,'cov':1.46,'mass':207.2,'Z':82,'name':'Lead'},
    'Bi': {'color':0x9E4FB5,'r':1.56,'cov':1.48,'mass':209.0,'Z':83,'name':'Bismuth'},
}

Z_TO_EL = {v['Z']: k for k, v in ELEMENTS.items()}

# ═══════════════════════════════════════════════════════════════
#  PRESET CRYSTAL STRUCTURES
# ═══════════════════════════════════════════════════════════════
# Each preset: { 'cell': [[ax,ay,az],[bx,by,bz],[cx,cy,cz]],
#                'atoms': [{'el':..,'fx':..,'fy':..,'fz':..},...] }
# Coordinates are FRACTIONAL. Converted to Cartesian on load.

_a_si = 5.431
_a_nacl = 5.640
_a_cu = 3.615
_a_fe = 2.870
_a_diamond = 3.567
_a_gaas = 5.653
_a_batio3 = 4.00
_c_batio3 = 4.036
_a_graphite = 2.461
_c_graphite = 6.708
_a_mgo = 4.212
_a_au = 4.078
_a_tio2_a = 3.784
_c_tio2   = 9.515

LATTICES = {
    "silicon": {
        'cell': [[_a_si,0,0],[0,_a_si,0],[0,0,_a_si]],
        'atoms': [
            {'el':'Si','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Si','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'Si','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'Si','fx':0.0,'fy':0.5,'fz':0.5},
            {'el':'Si','fx':0.25,'fy':0.25,'fz':0.25},
            {'el':'Si','fx':0.75,'fy':0.75,'fz':0.25},
            {'el':'Si','fx':0.75,'fy':0.25,'fz':0.75},
            {'el':'Si','fx':0.25,'fy':0.75,'fz':0.75},
        ],
    },
    "nacl": {
        'cell': [[_a_nacl,0,0],[0,_a_nacl,0],[0,0,_a_nacl]],
        'atoms': [
            {'el':'Na','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Na','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'Na','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'Na','fx':0.0,'fy':0.5,'fz':0.5},
            {'el':'Cl','fx':0.5,'fy':0.0,'fz':0.0},
            {'el':'Cl','fx':0.0,'fy':0.5,'fz':0.0},
            {'el':'Cl','fx':0.0,'fy':0.0,'fz':0.5},
            {'el':'Cl','fx':0.5,'fy':0.5,'fz':0.5},
        ],
    },
    "copper": {
        'cell': [[_a_cu,0,0],[0,_a_cu,0],[0,0,_a_cu]],
        'atoms': [
            {'el':'Cu','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Cu','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'Cu','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'Cu','fx':0.0,'fy':0.5,'fz':0.5},
        ],
    },
    "iron_bcc": {
        'cell': [[_a_fe,0,0],[0,_a_fe,0],[0,0,_a_fe]],
        'atoms': [
            {'el':'Fe','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Fe','fx':0.5,'fy':0.5,'fz':0.5},
        ],
    },
    "diamond": {
        'cell': [[_a_diamond,0,0],[0,_a_diamond,0],[0,0,_a_diamond]],
        'atoms': [
            {'el':'C','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'C','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'C','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'C','fx':0.0,'fy':0.5,'fz':0.5},
            {'el':'C','fx':0.25,'fy':0.25,'fz':0.25},
            {'el':'C','fx':0.75,'fy':0.75,'fz':0.25},
            {'el':'C','fx':0.75,'fy':0.25,'fz':0.75},
            {'el':'C','fx':0.25,'fy':0.75,'fz':0.75},
        ],
    },
    "gaas": {
        'cell': [[_a_gaas,0,0],[0,_a_gaas,0],[0,0,_a_gaas]],
        'atoms': [
            {'el':'Ga','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Ga','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'Ga','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'Ga','fx':0.0,'fy':0.5,'fz':0.5},
            {'el':'As','fx':0.25,'fy':0.25,'fz':0.25},
            {'el':'As','fx':0.75,'fy':0.75,'fz':0.25},
            {'el':'As','fx':0.75,'fy':0.25,'fz':0.75},
            {'el':'As','fx':0.25,'fy':0.75,'fz':0.75},
        ],
    },
    "batio3": {
        'cell': [[_a_batio3,0,0],[0,_a_batio3,0],[0,0,_c_batio3]],
        'atoms': [
            {'el':'Ba','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Ti','fx':0.5,'fy':0.5,'fz':0.52},
            {'el':'O','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'O','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'O','fx':0.0,'fy':0.5,'fz':0.5},
        ],
    },
    "graphite": {
        'cell': [[_a_graphite,0,0],
                 [-_a_graphite*0.5, _a_graphite*math.sqrt(3)/2, 0],
                 [0, 0, _c_graphite]],
        'atoms': [
            {'el':'C','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'C','fx':1/3,'fy':2/3,'fz':0.0},
            {'el':'C','fx':0.0,'fy':0.0,'fz':0.5},
            {'el':'C','fx':2/3,'fy':1/3,'fz':0.5},
        ],
    },
    "mgo": {
        'cell': [[_a_mgo,0,0],[0,_a_mgo,0],[0,0,_a_mgo]],
        'atoms': [
            {'el':'Mg','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Mg','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'Mg','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'Mg','fx':0.0,'fy':0.5,'fz':0.5},
            {'el':'O','fx':0.5,'fy':0.0,'fz':0.0},
            {'el':'O','fx':0.0,'fy':0.5,'fz':0.0},
            {'el':'O','fx':0.0,'fy':0.0,'fz':0.5},
            {'el':'O','fx':0.5,'fy':0.5,'fz':0.5},
        ],
    },
    "gold": {
        'cell': [[_a_au,0,0],[0,_a_au,0],[0,0,_a_au]],
        'atoms': [
            {'el':'Au','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Au','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'Au','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'Au','fx':0.0,'fy':0.5,'fz':0.5},
        ],
    },
    "tio2_rutile": {
        'cell': [[_a_tio2_a,0,0],[0,_a_tio2_a,0],[0,0,_c_tio2]],
        'atoms': [
            {'el':'Ti','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Ti','fx':0.5,'fy':0.5,'fz':0.5},
            {'el':'O','fx':0.305,'fy':0.305,'fz':0.0},
            {'el':'O','fx':0.695,'fy':0.695,'fz':0.0},
            {'el':'O','fx':0.195,'fy':0.805,'fz':0.5},
            {'el':'O','fx':0.805,'fy':0.195,'fz':0.5},
        ],
    },
    "perovskite_srtio3": {
        'cell': [[3.905,0,0],[0,3.905,0],[0,0,3.905]],
        'atoms': [
            {'el':'Sr','fx':0.0,'fy':0.0,'fz':0.0},
            {'el':'Ti','fx':0.5,'fy':0.5,'fz':0.5},
            {'el':'O','fx':0.5,'fy':0.5,'fz':0.0},
            {'el':'O','fx':0.5,'fy':0.0,'fz':0.5},
            {'el':'O','fx':0.0,'fy':0.5,'fz':0.5},
        ],
    },
    "wurtzite_zno": {
        'cell': [[3.250,0,0],
                 [-3.250*0.5, 3.250*math.sqrt(3)/2, 0],
                 [0, 0, 5.207]],
        'atoms': [
            {'el':'Zn','fx':1/3,'fy':2/3,'fz':0.0},
            {'el':'Zn','fx':2/3,'fy':1/3,'fz':0.5},
            {'el':'O','fx':1/3,'fy':2/3,'fz':0.382},
            {'el':'O','fx':2/3,'fy':1/3,'fz':0.882},
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
#  SHADERS
# ═══════════════════════════════════════════════════════════════

_VERT_LIT = """
#version 330
uniform mat4 mvp; uniform mat4 model;
in vec3 in_position; in vec3 in_normal; in vec3 in_color;
out vec3 v_normal, v_color, v_world;
void main() {
    vec4 w = model * vec4(in_position, 1.0);
    v_world = w.xyz; gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = mat3(model) * in_normal; v_color = in_color;
}"""

_FRAG_LIT = """
#version 330
uniform vec3 light_dir, ambient, view_pos;
in vec3 v_normal, v_color, v_world;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal), l = normalize(light_dir);
    float diff = max(dot(n, l), 0.0);
    vec3 vd = normalize(view_pos - v_world);
    vec3 h = normalize(l + vd);
    float spec = pow(max(dot(n, h), 0.0), 64.0) * 0.35;
    vec3 col = v_color * (ambient + diff * vec3(1.0, 0.98, 0.95)) + vec3(spec);
    frag = vec4(col / (col + vec3(1.0)), 1.0);
}"""

_FRAG_WIRE = """
#version 330
uniform vec3 light_dir, ambient, view_pos;
uniform float alpha;
in vec3 v_normal, v_color, v_world;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal), l = normalize(light_dir);
    float diff = abs(dot(n, l)) * 0.5 + 0.5;
    vec3 vd = normalize(view_pos - v_world);
    float rim = 1.0 - abs(dot(n, vd));
    vec3 col = v_color * (ambient * 0.6 + diff * 0.8) + vec3(rim * 0.08);
    frag = vec4(col, alpha);
}"""

# ═══════════════════════════════════════════════════════════════
#  GEOMETRY (numpy-accelerated)
# ═══════════════════════════════════════════════════════════════

def _make_sphere(radius, segs_h, segs_v, color_hex):
    r, g, b = _hex(color_hex)
    theta = np.linspace(0, np.pi, segs_v + 1)
    phi = np.linspace(0, 2 * np.pi, segs_h + 1)
    x = np.outer(np.sin(theta), np.cos(phi)) * radius
    y = np.outer(np.cos(theta), np.ones(segs_h + 1)) * radius
    z = np.outer(np.sin(theta), np.sin(phi)) * radius
    verts = np.empty((segs_v * segs_h * 6, 9), dtype=np.float32)
    idx = 0
    for j in range(segs_v):
        for i in range(segs_h):
            p00 = np.array([x[j, i], y[j, i], z[j, i]])
            p10 = np.array([x[j+1, i], y[j+1, i], z[j+1, i]])
            p11 = np.array([x[j+1, i+1], y[j+1, i+1], z[j+1, i+1]])
            p01 = np.array([x[j, i+1], y[j, i+1], z[j, i+1]])
            for p in (p00, p10, p11, p00, p11, p01):
                n = p / (np.linalg.norm(p) + 1e-9)
                verts[idx] = [p[0], p[1], p[2], n[0], n[1], n[2], r, g, b]
                idx += 1
    return verts[:idx].flatten().tolist()

def _make_cylinder(radius, h, segs, color_hex):
    cr, cg, cb = _hex(color_hex)
    verts = []; hh = h / 2
    for i in range(segs):
        a0 = 2 * math.pi * i / segs; a1 = 2 * math.pi * (i + 1) / segs
        x0, z0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
        nx0, nz0 = math.cos(a0), math.sin(a0)
        for v in ((x0, -hh, z0), (x1, -hh, z1), (x1, hh, z1), (x0, -hh, z0), (x1, hh, z1), (x0, hh, z0)):
            verts.extend(v); verts.extend((nx0, 0, nz0)); verts.extend((cr, cg, cb))
        for v in ((0, hh, 0), (x0, hh, z0), (x1, hh, z1)):
            verts.extend(v); verts.extend((0, 1, 0)); verts.extend((cr, cg, cb))
        for v in ((0, -hh, 0), (x1, -hh, z1), (x0, -hh, z0)):
            verts.extend(v); verts.extend((0, -1, 0)); verts.extend((cr, cg, cb))
    return verts

def _make_arrow(length, radius, head_r, head_h, segs, color_hex):
    """Arrow along +Y axis, base at origin."""
    shaft = _make_cylinder(radius, length - head_h, segs, color_hex)
    # shift shaft up
    arr = np.array(shaft, dtype=np.float32).reshape(-1, 9)
    arr[:, 1] += (length - head_h) / 2
    # cone head
    cr, cg, cb = _hex(color_hex)
    cone = []
    tip_y = length
    base_y = length - head_h
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i + 1) / segs
        x0, z0 = math.cos(a0) * head_r, math.sin(a0) * head_r
        x1, z1 = math.cos(a1) * head_r, math.sin(a1) * head_r
        # side
        nx = math.cos((a0 + a1) / 2)
        nz = math.sin((a0 + a1) / 2)
        ny = head_r / head_h
        nl = math.sqrt(nx*nx + ny*ny + nz*nz) + 1e-9
        nx /= nl; ny /= nl; nz /= nl
        for v in ((0, tip_y, 0), (x0, base_y, z0), (x1, base_y, z1)):
            cone.extend(v); cone.extend((nx, ny, nz)); cone.extend((cr, cg, cb))
        # base
        for v in ((0, base_y, 0), (x1, base_y, z1), (x0, base_y, z0)):
            cone.extend(v); cone.extend((0, -1, 0)); cone.extend((cr, cg, cb))
    cone_arr = np.array(cone, dtype=np.float32).reshape(-1, 9)
    return np.vstack([arr, cone_arr]).flatten().tolist()

def _transform_verts(verts_list, mat):
    arr = np.array(verts_list, dtype=np.float32).reshape(-1, 9)
    if arr.shape[0] == 0: return []
    m4 = np.array([[mat[c][r] for c in range(4)] for r in range(4)], dtype=np.float32)
    pos = np.hstack([arr[:, :3], np.ones((arr.shape[0], 1), dtype=np.float32)])
    pos_t = (m4 @ pos.T).T[:, :3]
    nrm_t = (m4[:3, :3] @ arr[:, 3:6].T).T
    return np.hstack([pos_t, nrm_t, arr[:, 6:9]]).flatten().tolist()

# ═══════════════════════════════════════════════════════════════
#  PARSERS
# ═══════════════════════════════════════════════════════════════

def frac_to_cart(fx, fy, fz, cell):
    """Fractional → Cartesian using cell matrix (rows = vectors)."""
    a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    return fx * a + fy * b + fz * c

def cart_to_frac(x, y, z, cell):
    """Cartesian → fractional."""
    M = np.array(cell).T  # columns = vectors
    return np.linalg.solve(M, np.array([x, y, z]))

def cell_volume(cell):
    a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    return abs(np.dot(a, np.cross(b, c)))

def cell_params(cell):
    """Return [a, b, c, alpha, beta, gamma] from cell matrix."""
    va, vb, vc = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    a = np.linalg.norm(va); b = np.linalg.norm(vb); c = np.linalg.norm(vc)
    alpha = math.degrees(math.acos(np.clip(np.dot(vb, vc) / (b * c), -1, 1)))
    beta  = math.degrees(math.acos(np.clip(np.dot(va, vc) / (a * c), -1, 1)))
    gamma = math.degrees(math.acos(np.clip(np.dot(va, vb) / (a * b), -1, 1)))
    return [a, b, c, alpha, beta, gamma]

def detect_bonds(atoms, factor=1.2):
    """Detect bonds by covalent radii with tolerance factor."""
    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a, b = atoms[i], atoms[j]
            d = math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 + (a['z'] - b['z'])**2)
            ci = ELEMENTS.get(a['el'], {'cov': 0.77})['cov']
            cj = ELEMENTS.get(b['el'], {'cov': 0.77})['cov']
            if 0.4 < d < (ci + cj) * factor:
                bonds.append((i, j))
    return bonds

def parse_cif(text):
    """Minimal CIF parser → cell + fractional atoms."""
    lines = text.split('\n')
    cell_a = cell_b = cell_c = 5.0
    alpha = beta = gamma = 90.0
    atoms = []
    in_loop = False; loop_keys = []; loop_data = False

    for line in lines:
        s = line.strip()
        if s.startswith('#') or not s: continue
        # Cell parameters
        if s.startswith('_cell_length_a'):
            cell_a = float(re.split(r'[\s(]+', s)[1])
        elif s.startswith('_cell_length_b'):
            cell_b = float(re.split(r'[\s(]+', s)[1])
        elif s.startswith('_cell_length_c'):
            cell_c = float(re.split(r'[\s(]+', s)[1])
        elif s.startswith('_cell_angle_alpha'):
            alpha = float(re.split(r'[\s(]+', s)[1])
        elif s.startswith('_cell_angle_beta'):
            beta = float(re.split(r'[\s(]+', s)[1])
        elif s.startswith('_cell_angle_gamma'):
            gamma = float(re.split(r'[\s(]+', s)[1])
        elif s == 'loop_':
            in_loop = True; loop_keys = []; loop_data = False
        elif in_loop and s.startswith('_'):
            loop_keys.append(s)
        elif in_loop and not s.startswith('_'):
            loop_data = True
            parts = s.split()
            # Check if this is an atom loop
            key_lower = [k.lower() for k in loop_keys]
            has_fract = any('fract_x' in k for k in key_lower)
            if has_fract and len(parts) >= len(loop_keys):
                try:
                    ix = next(i for i, k in enumerate(key_lower) if 'fract_x' in k)
                    iy = next(i for i, k in enumerate(key_lower) if 'fract_y' in k)
                    iz = next(i for i, k in enumerate(key_lower) if 'fract_z' in k)
                    # symbol
                    ie = None
                    for tag in ['_atom_site_type_symbol', '_atom_site_label']:
                        tag_l = tag.lower()
                        for i, k in enumerate(key_lower):
                            if tag_l in k: ie = i; break
                        if ie is not None: break
                    if ie is None: ie = 0
                    el_raw = parts[ie].strip("'\"")
                    el = re.match(r'([A-Z][a-z]?)', el_raw)
                    el = el.group(1) if el else el_raw[:2]
                    fx = float(re.split(r'[(]', parts[ix])[0])
                    fy = float(re.split(r'[(]', parts[iy])[0])
                    fz = float(re.split(r'[(]', parts[iz])[0])
                    atoms.append({'el': el, 'fx': fx, 'fy': fy, 'fz': fz})
                except:
                    pass
            elif not has_fract and loop_data and not parts[0].startswith('_'):
                # Non-atom loop data, skip
                pass
        elif loop_data and not s.startswith('_') and not s.startswith('loop_'):
            # Continue reading loop data
            parts = s.split()
            key_lower = [k.lower() for k in loop_keys]
            has_fract = any('fract_x' in k for k in key_lower)
            if has_fract and len(parts) >= len(loop_keys):
                try:
                    ix = next(i for i, k in enumerate(key_lower) if 'fract_x' in k)
                    iy = next(i for i, k in enumerate(key_lower) if 'fract_y' in k)
                    iz = next(i for i, k in enumerate(key_lower) if 'fract_z' in k)
                    ie = None
                    for tag in ['_atom_site_type_symbol', '_atom_site_label']:
                        tag_l = tag.lower()
                        for i, k in enumerate(key_lower):
                            if tag_l in k: ie = i; break
                        if ie is not None: break
                    if ie is None: ie = 0
                    el_raw = parts[ie].strip("'\"")
                    el = re.match(r'([A-Z][a-z]?)', el_raw)
                    el = el.group(1) if el else el_raw[:2]
                    fx = float(re.split(r'[(]', parts[ix])[0])
                    fy = float(re.split(r'[(]', parts[iy])[0])
                    fz = float(re.split(r'[(]', parts[iz])[0])
                    atoms.append({'el': el, 'fx': fx, 'fy': fy, 'fz': fz})
                except:
                    pass
        else:
            if loop_data: in_loop = False; loop_data = False

    # Build cell from parameters
    cell = _cell_from_params(cell_a, cell_b, cell_c, alpha, beta, gamma)
    return cell, atoms

def _cell_from_params(a, b, c, alpha, beta, gamma):
    """Construct cell matrix from lattice parameters."""
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)
    cos_a, cos_b, cos_g = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sin_g = math.sin(gamma_r)
    v1 = [a, 0, 0]
    v2 = [b * cos_g, b * sin_g, 0]
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0))
    v3 = [cx, cy, cz]
    return [v1, v2, v3]

def parse_poscar(text):
    """Parse VASP POSCAR/CONTCAR format → cell + fractional atoms."""
    lines = text.strip().split('\n')
    if len(lines) < 8: return None, None
    scale = float(lines[1].strip())
    cell = []
    for i in range(2, 5):
        parts = lines[i].split()
        cell.append([float(x) * scale for x in parts[:3]])
    # Species line
    species_line = lines[5].split()
    try:
        counts = [int(x) for x in species_line]
        species = [f"X{i}" for i in range(len(counts))]
        count_line_idx = 5
    except ValueError:
        species = species_line
        counts = [int(x) for x in lines[6].split()]
        count_line_idx = 6
    # Coordinate type
    coord_line = lines[count_line_idx + 1].strip()
    is_direct = coord_line[0].upper() in ('D', 'F')
    atoms = []
    atom_idx = count_line_idx + 2
    for sp, cnt in zip(species, counts):
        for j in range(cnt):
            parts = lines[atom_idx].split()
            fx, fy, fz = float(parts[0]), float(parts[1]), float(parts[2])
            if not is_direct:
                # Cartesian → fractional
                f = cart_to_frac(fx, fy, fz, cell)
                fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
            atoms.append({'el': sp, 'fx': fx, 'fy': fy, 'fz': fz})
            atom_idx += 1
    return cell, atoms

def parse_xyz_crystal(text):
    """Parse extended XYZ with lattice in comment line."""
    lines = text.strip().split('\n')
    if len(lines) < 3: return None, None
    natoms = int(lines[0].strip())
    comment = lines[1]
    # Try to extract Lattice="..." from comment
    cell = None
    m = re.search(r'Lattice="([^"]+)"', comment, re.I)
    if m:
        vals = [float(x) for x in m.group(1).split()]
        if len(vals) == 9:
            cell = [[vals[0], vals[1], vals[2]],
                    [vals[3], vals[4], vals[5]],
                    [vals[6], vals[7], vals[8]]]
    atoms_cart = []
    for line in lines[2:2+natoms]:
        parts = line.strip().split()
        if len(parts) >= 4:
            el = parts[0]
            if el not in ELEMENTS:
                el = re.match(r'([A-Z][a-z]?)', el)
                el = el.group(1) if el else 'X'
            atoms_cart.append({'el': el, 'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3])})
    if cell and atoms_cart:
        atoms = []
        for a in atoms_cart:
            f = cart_to_frac(a['x'], a['y'], a['z'], cell)
            atoms.append({'el': a['el'], 'fx': float(f[0]), 'fy': float(f[1]), 'fz': float(f[2])})
        return cell, atoms
    return cell, None

# ═══════════════════════════════════════════════════════════════
#  GL VIEWER WIDGET
# ═══════════════════════════════════════════════════════════════

class CrystalViewer(QWidget):
    """OpenGL crystal structure viewer — offscreen FBO → QImage → QPainter.
    Supports crystal rendering, unit cell wireframe, lattice axes, overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # State
        self.atoms = []; self.bonds = []; self.mol_name = ""
        self.cell = None
        self.rot_x = 0.35; self.rot_y = 0.3; self.auto_rot = 0.0; self.cam_dist = 14.0
        self._dragging = False; self._lmx = 0; self._lmy = 0
        self.selected_atom = -1; self.render_style = 'ballstick'
        self.show_bonds_flag = True; self.show_cell = True; self.show_axes = True
        self._center = (0, 0, 0)
        # GL
        self._gl_ready = False; self.ctx = None; self.fbo = None
        self._fbo_w = 0; self._fbo_h = 0; self._frame = None
        self._mol_vao = None; self._mol_n = 0
        self._wire_vao = None; self._wire_n = 0
        # Overlays
        self._overlays = OrderedDict()
        self._measurements = []
        # Timer
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.setInterval(16)

    def _ensure_gl(self):
        if self._gl_ready: return
        self._gl_ready = True
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog_lit = self.ctx.program(vertex_shader=_VERT_LIT, fragment_shader=_FRAG_LIT)
        self.prog_wire = self.ctx.program(vertex_shader=_VERT_LIT, fragment_shader=_FRAG_WIRE)
        self._resize_fbo(max(self.width(), 320), max(self.height(), 200))
        self.timer.start()

    def _resize_fbo(self, w, h):
        if w == self._fbo_w and h == self._fbo_h and self.fbo: return
        if self.fbo: self.fbo.release()
        self._fbo_w = w; self._fbo_h = h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)))

    def set_crystal(self, atoms, cell, name=""):
        """Set atoms (Cartesian) and cell for rendering."""
        self.atoms = atoms; self.cell = cell
        self.bonds = detect_bonds(atoms) if atoms else []
        self.mol_name = name; self.selected_atom = -1
        if atoms:
            cx = sum(a['x'] for a in atoms) / len(atoms)
            cy = sum(a['y'] for a in atoms) / len(atoms)
            cz = sum(a['z'] for a in atoms) / len(atoms)
            self._center = (cx, cy, cz)
            md = max(math.sqrt((a['x'] - cx)**2 + (a['y'] - cy)**2 + (a['z'] - cz)**2) for a in atoms)
            self.cam_dist = max(md * 3.0, 8.0)
        self.auto_rot = 0.0
        self._rebuild_mol()
        self._rebuild_wireframe()

    def set_style(self, style):
        self.render_style = style
        self._rebuild_mol()

    def add_overlay(self, name, fn):
        self._overlays[name] = fn

    def remove_overlay(self, name):
        self._overlays.pop(name, None)

    def screenshot(self, path):
        if self._frame: self._frame.save(path)

    def _rebuild_wireframe(self):
        """Build unit cell edges + lattice axis arrows."""
        if not self._gl_ready or not self.cell: return
        cx, cy, cz = self._center
        a, b, c = np.array(self.cell[0]), np.array(self.cell[1]), np.array(self.cell[2])
        o = np.zeros(3)
        # 12 edges of parallelepiped
        edges = [
            (o, a), (o, b), (o, c),
            (a, a+b), (a, a+c), (b, a+b), (b, b+c),
            (c, a+c), (c, b+c),
            (a+b, a+b+c), (a+c, a+b+c), (b+c, a+b+c),
        ]
        all_v = []
        for p1, p2 in edges:
            d = p2 - p1
            length = np.linalg.norm(d)
            if length < 0.01: continue
            cyl = _make_cylinder(0.03, length, 6, 0x6080A0)
            mid = (p1 + p2) / 2 - np.array([cx, cy, cz])
            dn = d / length
            up = glm.vec3(0, 1, 0)
            gd = glm.vec3(float(dn[0]), float(dn[1]), float(dn[2]))
            if abs(glm.dot(up, gd)) > 0.999:
                rm = glm.mat4(1) if gd.y > 0 else glm.rotate(glm.mat4(1), math.pi, glm.vec3(1, 0, 0))
            else:
                ax = glm.normalize(glm.cross(up, gd))
                ang = math.acos(max(-1, min(1, glm.dot(up, gd))))
                rm = glm.rotate(glm.mat4(1), ang, ax)
            t = glm.translate(glm.mat4(1), glm.vec3(float(mid[0]), float(mid[1]), float(mid[2]))) * rm
            all_v.extend(_transform_verts(cyl, t))
        # Arrows for axes
        if self.show_axes:
            colors = [0xDD3030, 0x30AA30, 0x3050DD]  # a=red, b=green, c=blue
            vecs = [a, b, c]
            for vi, (vec, col) in enumerate(zip(vecs, colors)):
                length = np.linalg.norm(vec)
                if length < 0.01: continue
                arrow = _make_arrow(length * 0.35, 0.06, 0.15, 0.3, 8, col)
                dn = vec / length
                start = -np.array([cx, cy, cz])
                gd = glm.vec3(float(dn[0]), float(dn[1]), float(dn[2]))
                up = glm.vec3(0, 1, 0)
                if abs(glm.dot(up, gd)) > 0.999:
                    rm = glm.mat4(1) if gd.y > 0 else glm.rotate(glm.mat4(1), math.pi, glm.vec3(1, 0, 0))
                else:
                    ax = glm.normalize(glm.cross(up, gd))
                    ang = math.acos(max(-1, min(1, glm.dot(up, gd))))
                    rm = glm.rotate(glm.mat4(1), ang, ax)
                t = glm.translate(glm.mat4(1), glm.vec3(float(start[0]), float(start[1]), float(start[2]))) * rm
                all_v.extend(_transform_verts(arrow, t))

        if not all_v:
            self._wire_vao = None; self._wire_n = 0; return
        data = np.array(all_v, dtype='f4').tobytes()
        if self._wire_vao:
            try: self._wire_vao.release()
            except: pass
        vbo = self.ctx.buffer(data)
        self._wire_vao = self.ctx.vertex_array(self.prog_wire, [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        self._wire_n = len(all_v) // 9

    def _rebuild_mol(self):
        if not self._gl_ready or not self.atoms: return
        cx, cy, cz = self._center
        all_v = []
        scale = 2.0 if self.render_style == 'spacefill' else 1.0
        segs = 20 if len(self.atoms) > 60 else 28
        for idx, a in enumerate(self.atoms):
            el = ELEMENTS.get(a['el'], {'color': 0x888888, 'r': 0.5})
            r = el['r'] * 0.35 * scale if self.render_style != 'wireframe' else 0.10
            col = el['color']
            if self.selected_atom == idx:
                rr, gg, bb = _hex(col)
                col = (min(int((rr + 0.3) * 255), 255) << 16) | (min(int((gg + 0.3) * 255), 255) << 8) | min(int((bb + 0.3) * 255), 255)
            sp = _make_sphere(r, segs, segs // 2, col)
            t = glm.translate(glm.mat4(1), glm.vec3(a['x'] - cx, a['y'] - cy, a['z'] - cz))
            all_v.extend(_transform_verts(sp, t))
        if self.show_bonds_flag and self.render_style != 'spacefill':
            br = 0.02 if self.render_style == 'wireframe' else 0.05
            for i, j in self.bonds:
                a1, a2 = self.atoms[i], self.atoms[j]
                s = glm.vec3(a1['x'] - cx, a1['y'] - cy, a1['z'] - cz)
                e = glm.vec3(a2['x'] - cx, a2['y'] - cy, a2['z'] - cz)
                mid = (s + e) * 0.5; diff = e - s; length = glm.length(diff)
                if length < 0.01: continue
                d = glm.normalize(diff)
                cyl = _make_cylinder(br, length, 6, 0x889AAA)
                up = glm.vec3(0, 1, 0)
                if abs(glm.dot(up, d)) > 0.999:
                    rm = glm.mat4(1) if d.y > 0 else glm.rotate(glm.mat4(1), math.pi, glm.vec3(1, 0, 0))
                else:
                    ax = glm.normalize(glm.cross(up, d))
                    ang = math.acos(max(-1, min(1, glm.dot(up, d))))
                    rm = glm.rotate(glm.mat4(1), ang, ax)
                all_v.extend(_transform_verts(cyl, glm.translate(glm.mat4(1), mid) * rm))
        if not all_v: self._mol_vao = None; self._mol_n = 0; return
        data = np.array(all_v, dtype='f4').tobytes()
        if self._mol_vao:
            try: self._mol_vao.release()
            except: pass
        vbo = self.ctx.buffer(data)
        self._mol_vao = self.ctx.vertex_array(self.prog_lit, [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        self._mol_n = len(all_v) // 9

    def _tick(self):
        if not self._dragging: self.auto_rot += 0.004
        self._render(); self.update()

    def _render(self):
        if not self._gl_ready: return
        w, h = max(self.width(), 320), max(self.height(), 200)
        self._resize_fbo(w, h); self.fbo.use(); self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0, 0, 0, 0)
        proj = glm.perspective(glm.radians(45), w / h, 0.1, 300.0)
        ry = self.rot_y + self.auto_rot
        eye = glm.vec3(math.sin(ry) * math.cos(self.rot_x) * self.cam_dist,
                       math.sin(self.rot_x) * self.cam_dist,
                       math.cos(ry) * math.cos(self.rot_x) * self.cam_dist)
        view = glm.lookAt(eye, glm.vec3(0), glm.vec3(0, 1, 0))
        vp = proj * view; identity = glm.mat4(1)
        # Atoms + bonds (opaque)
        if self._mol_vao and self._mol_n > 0:
            _uw(self.prog_lit['mvp'], vp); _uw(self.prog_lit['model'], identity)
            _uw(self.prog_lit['light_dir'], glm.normalize(glm.vec3(0.5, 0.8, 0.6)))
            _uw(self.prog_lit['ambient'], glm.vec3(0.42, 0.41, 0.43))
            _uw(self.prog_lit['view_pos'], eye)
            self._mol_vao.render(moderngl.TRIANGLES)
        # Wireframe (unit cell + axes, semi-transparent)
        if self._wire_vao and self._wire_n > 0 and self.show_cell:
            self.ctx.disable(moderngl.CULL_FACE)
            _uw(self.prog_wire['mvp'], vp); _uw(self.prog_wire['model'], identity)
            _uw(self.prog_wire['light_dir'], glm.normalize(glm.vec3(0.5, 0.8, 0.6)))
            _uw(self.prog_wire['ambient'], glm.vec3(0.5, 0.48, 0.52))
            _uw(self.prog_wire['view_pos'], eye)
            self.prog_wire['alpha'].value = 0.7
            self._wire_vao.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.CULL_FACE)
        raw = self.fbo.color_attachments[0].read()
        self._frame = QImage(raw, w, h, w * 4, QImage.Format_RGBA8888).mirrored(False, True)

    def paintEvent(self, event):
        self._ensure_gl()
        if self.atoms and self._mol_n == 0: self._rebuild_mol()
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self._frame and not self._frame.isNull(): p.drawImage(0, 0, self._frame)
        # HUD — crystal info
        if self.atoms:
            p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 195))
            p.drawRoundedRect(12, 12, 220, 76, 8, 8)
            p.setFont(QFont("Consolas", 9, QFont.Bold)); p.setPen(QColor(50, 55, 70))
            p.drawText(20, 22, 200, 16, Qt.AlignVCenter, self.mol_name or "Crystal")
            p.setFont(QFont("Consolas", 8)); p.setPen(QColor(80, 90, 110))
            counts = {}
            for a in self.atoms: counts[a['el']] = counts.get(a['el'], 0) + 1
            formula = ''.join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in sorted(counts.keys()))
            p.drawText(20, 38, 200, 14, Qt.AlignVCenter, f"{len(self.atoms)} atoms · {formula}")
            if self.cell:
                vol = cell_volume(self.cell)
                p.drawText(20, 52, 200, 14, Qt.AlignVCenter, f"V = {vol:.2f} Å³")
                cp = cell_params(self.cell)
                p.drawText(20, 64, 200, 14, Qt.AlignVCenter, f"a={cp[0]:.2f} b={cp[1]:.2f} c={cp[2]:.2f}")
        # Axis legend
        if self.show_axes:
            p.setFont(QFont("Consolas", 9, QFont.Bold))
            ly = h - 50
            for label, col in [("a", QColor(220, 50, 50)), ("b", QColor(50, 170, 50)), ("c", QColor(50, 80, 220))]:
                p.setPen(col)
                p.drawText(16, ly, 30, 14, Qt.AlignVCenter, label)
                ly += 15
        # Selected atom
        if 0 <= self.selected_atom < len(self.atoms):
            a = self.atoms[self.selected_atom]
            el = ELEMENTS.get(a['el'], {'name': a['el'], 'Z': '?', 'mass': 0})
            p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 210))
            p.drawRoundedRect(w - 185, 12, 173, 75, 8, 8)
            p.setFont(QFont("Consolas", 10, QFont.Bold)); p.setPen(QColor(40, 45, 60))
            p.drawText(w - 175, 18, 155, 16, Qt.AlignVCenter, f"{el['name']} #{self.selected_atom}")
            p.setFont(QFont("Consolas", 9)); p.setPen(QColor(90, 100, 120))
            p.drawText(w - 175, 36, 155, 13, Qt.AlignVCenter, f"{a['el']} Z={el['Z']}  {el['mass']:.3f} u")
            p.drawText(w - 175, 50, 155, 13, Qt.AlignVCenter, f"({a['x']:+.3f}, {a['y']:+.3f}, {a['z']:+.3f})")
        # Measurements
        p.setFont(QFont("Consolas", 9)); p.setPen(QColor(40, 100, 160))
        my = h - 60
        for mtype, idxs, val in self._measurements:
            if mtype == 'dist':
                p.drawText(12, my, 200, 14, Qt.AlignVCenter, f"d({idxs[0]}-{idxs[1]}) = {val:.4f} Å")
            elif mtype == 'angle':
                p.drawText(12, my, 250, 14, Qt.AlignVCenter, f"∠({idxs[0]}-{idxs[1]}-{idxs[2]}) = {val:.1f}°")
            my -= 16
        # Custom overlays
        for name, fn in self._overlays.items():
            try: fn(p, w, h)
            except: pass
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self._dragging = True; self._lmx = e.x(); self._lmy = e.y()
        e.accept()
    def mouseReleaseEvent(self, e): self._dragging = False; e.accept()
    def mouseMoveEvent(self, e):
        if self._dragging:
            self.rot_y += (e.x() - self._lmx) * 0.008
            self.rot_x = max(-1.4, min(1.4, self.rot_x + (e.y() - self._lmy) * 0.008))
            self._lmx = e.x(); self._lmy = e.y()
        e.accept()
    def wheelEvent(self, e):
        self.cam_dist = max(3, min(80, self.cam_dist - e.angleDelta().y() * 0.008))
        e.accept()

# ═══════════════════════════════════════════════════════════════
#  MATLAB — MAIN API SINGLETON
# ═══════════════════════════════════════════════════════════════

class _Signals(QObject):
    status = Signal(str)

class MatLab:
    """
    Main materials science API. Registered as `crystal` in the namespace.

    QUICK REFERENCE (for LLM context):
        crystal.load(name)                  — load preset: silicon/nacl/copper/iron_bcc/diamond/gaas/batio3/graphite/mgo/gold/tio2_rutile/perovskite_srtio3/wurtzite_zno
        crystal.load_cif(path_or_text)      — load from CIF file or text
        crystal.load_poscar(path_or_text)   — load VASP POSCAR
        crystal.load_xyz(path_or_text)      — load extended XYZ with Lattice
        crystal.supercell(na, nb, nc)       — expand to na×nb×nc supercell
        crystal.style(s)                    — ballstick/spacefill/wireframe
        crystal.select(i)                   — highlight atom i
        crystal.measure(i, j)               — distance between atoms i-j
        crystal.angle(i, j, k)             — angle i-j-k
        crystal.show_unitcell(bool)         — toggle unit cell wireframe
        crystal.show_axes(bool)             — toggle lattice vector arrows
        crystal.cleave(h, k, l, layers=4, vacuum=10.0)  — cleave surface slab
        crystal.vacancy(i)                  — remove atom i (point defect)
        crystal.substitute(i, new_el)       — substitute element at atom i
        crystal.calculate(method="LJ")      — quick energy (LJ/EMT)
        crystal.radial_distribution(rmax=10, nbins=100) — g(r) overlay
        crystal.xrd(wavelength=1.5406)      — powder XRD overlay
        crystal.overlay_dos(energies, dos)  — density of states overlay
        crystal.overlay_bands(kpath, bands) — band structure overlay
        crystal.export_poscar(path)         — VASP POSCAR output
        crystal.export_cif(path)            — CIF output
        crystal.export_xyz(path)            — XYZ snapshot
        crystal.export_qe_input(path)       — Quantum ESPRESSO pw.x input
        crystal.atoms                       — atom list [{'el','x','y','z','fx','fy','fz'},...]
        crystal.cell                        — [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
        crystal.info                        — dict of properties
        crystal.viewer                      — the CrystalViewer widget
        crystal.log(msg)                    — append to log panel
    """

    def __init__(self, viewer, log_widget=None):
        self.viewer = viewer
        self._log = log_widget
        self.atoms = []       # Cartesian + fractional: {'el','x','y','z','fx','fy','fz'}
        self.cell = None      # [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
        self.bonds = []
        self.info = {}
        self._signals = _Signals()
        self._frac_atoms = [] # fractional-only copy of base unit cell

    def log(self, msg):
        if self._log: self._log.append(f"[mat] {msg}")
        print(f"[mat] {msg}")

    # ── Loading ────────────────────────────────────────────────

    def _set_from_frac(self, cell, frac_atoms, name=""):
        """Set crystal from cell + fractional atoms, compute Cartesian."""
        self.cell = [list(v) for v in cell]
        self._frac_atoms = [dict(a) for a in frac_atoms]
        self.atoms = []
        for a in frac_atoms:
            pos = frac_to_cart(a['fx'], a['fy'], a['fz'], cell)
            self.atoms.append({
                'el': a['el'], 'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                'fx': a['fx'], 'fy': a['fy'], 'fz': a['fz']
            })
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_crystal(self.atoms, self.cell, name)
        self._update_info(name)
        self.log(f"Loaded: {name} ({len(self.atoms)} atoms)")
        return self

    def _update_info(self, name=""):
        counts = {}
        for a in self.atoms: counts[a['el']] = counts.get(a['el'], 0) + 1
        formula = ''.join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in sorted(counts.keys()))
        vol = cell_volume(self.cell) if self.cell else 0
        cp = cell_params(self.cell) if self.cell else [0]*6
        total_mass = sum(ELEMENTS.get(a['el'], {'mass': 0})['mass'] for a in self.atoms)
        density = (total_mass * 1.6605) / vol if vol > 0 else 0  # g/cm³ (1.6605e-24 g/amu, 1e-24 cm³/ų)
        self.info.update({
            'name': name, 'formula': formula, 'n_atoms': len(self.atoms),
            'volume': vol, 'density': density,
            'lattice': cp,
            'cell': self.cell,
        })
        # Try spglib for space group
        try:
            import spglib
            lat = np.array(self.cell)
            pos = np.array([[a['fx'], a['fy'], a['fz']] for a in self._frac_atoms])
            numbers = [ELEMENTS.get(a['el'], {'Z': 0})['Z'] for a in self._frac_atoms]
            sg = spglib.get_spacegroup((lat, pos, numbers), symprec=0.1)
            self.info['spacegroup'] = sg
        except:
            self.info['spacegroup'] = None

    def load(self, name):
        """Load a preset crystal structure by name."""
        key = name.lower().replace(' ', '').replace('-', '').replace('_', '')
        for k, data in LATTICES.items():
            kn = k.lower().replace('_', '')
            if key in kn or kn in key:
                return self._set_from_frac(data['cell'], data['atoms'], k)
        self.log(f"Unknown preset: {name}. Available: {', '.join(LATTICES.keys())}")
        return self

    def load_cif(self, path_or_text):
        """Load from CIF file or CIF text string."""
        if os.path.isfile(os.path.expanduser(str(path_or_text))):
            path = os.path.expanduser(path_or_text)
            with open(path, 'r') as f: text = f.read()
            name = os.path.basename(path)
        else:
            text = str(path_or_text)
            name = "CIF input"
        cell, atoms = parse_cif(text)
        if not atoms:
            self.log("No atoms found in CIF"); return self
        return self._set_from_frac(cell, atoms, name)

    def load_poscar(self, path_or_text):
        """Load from VASP POSCAR/CONTCAR."""
        if os.path.isfile(os.path.expanduser(str(path_or_text))):
            path = os.path.expanduser(path_or_text)
            with open(path, 'r') as f: text = f.read()
            name = os.path.basename(path)
        else:
            text = str(path_or_text)
            name = "POSCAR input"
        cell, atoms = parse_poscar(text)
        if not atoms:
            self.log("No atoms found in POSCAR"); return self
        return self._set_from_frac(cell, atoms, name)

    def load_xyz(self, path_or_text):
        """Load from extended XYZ with Lattice in comment."""
        if os.path.isfile(os.path.expanduser(str(path_or_text))):
            path = os.path.expanduser(path_or_text)
            with open(path, 'r') as f: text = f.read()
            name = os.path.basename(path)
        else:
            text = str(path_or_text)
            name = "XYZ input"
        cell, atoms = parse_xyz_crystal(text)
        if not atoms:
            self.log("No atoms found in XYZ"); return self
        return self._set_from_frac(cell, atoms, name)

    # ── Structure manipulation ─────────────────────────────────

    def supercell(self, na=2, nb=2, nc=2):
        """Expand current structure to na×nb×nc supercell."""
        if not self.cell or not self._frac_atoms:
            self.log("No crystal loaded"); return self
        new_atoms = []
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    for a in self._frac_atoms:
                        new_atoms.append({
                            'el': a['el'],
                            'fx': (a['fx'] + ia) / na,
                            'fy': (a['fy'] + ib) / nb,
                            'fz': (a['fz'] + ic) / nc,
                        })
        new_cell = [
            [self.cell[0][j] * na for j in range(3)],
            [self.cell[1][j] * nb for j in range(3)],
            [self.cell[2][j] * nc for j in range(3)],
        ]
        name = f"{self.info.get('name', 'crystal')} {na}×{nb}×{nc}"
        return self._set_from_frac(new_cell, new_atoms, name)

    def cleave(self, h, k, l, layers=4, vacuum=10.0):
        """Cleave a (hkl) surface slab with given layers and vacuum.
        Simplified: works best for low-index cubic surfaces."""
        if not self.cell or not self._frac_atoms:
            self.log("No crystal loaded"); return self
        # Build a supercell large enough, then slice
        n = max(layers + 2, 4)
        # For simplicity, expand along c if (001), permute for others
        na, nb, nc = n, n, n
        all_frac = []
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    for a in self._frac_atoms:
                        all_frac.append({
                            'el': a['el'],
                            'fx': a['fx'] + ia,
                            'fy': a['fy'] + ib,
                            'fz': a['fz'] + ic,
                        })
        # Convert to Cartesian
        big_cell = [
            [self.cell[0][j] * na for j in range(3)],
            [self.cell[1][j] * nb for j in range(3)],
            [self.cell[2][j] * nc for j in range(3)],
        ]
        cart_atoms = []
        for a in all_frac:
            pos = frac_to_cart(a['fx'] / na, a['fy'] / nb, a['fz'] / nc, big_cell)
            cart_atoms.append({'el': a['el'], 'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])})
        # Miller plane normal
        a_vec = np.array(self.cell[0]); b_vec = np.array(self.cell[1]); c_vec = np.array(self.cell[2])
        # Reciprocal vectors
        V = np.dot(a_vec, np.cross(b_vec, c_vec))
        ra = np.cross(b_vec, c_vec) / V
        rb = np.cross(c_vec, a_vec) / V
        rc = np.cross(a_vec, b_vec) / V
        normal = h * ra + k * rb + l * rc
        normal = normal / np.linalg.norm(normal)
        # Project all atoms onto normal
        projs = [np.dot(np.array([a['x'], a['y'], a['z']]), normal) for a in cart_atoms]
        pmin = min(projs)
        # d-spacing
        d = 1.0 / np.linalg.norm(h * ra + k * rb + l * rc)
        thickness = layers * d
        # Select atoms within [pmin, pmin+thickness]
        slab_atoms = []
        for a, proj in zip(cart_atoms, projs):
            if pmin <= proj <= pmin + thickness + 0.1:
                slab_atoms.append(a)
        if not slab_atoms:
            self.log(f"No atoms in ({h}{k}{l}) slab"); return self
        # Build new slab cell — keep a,b, set c along normal with vacuum
        # For cubic: use original a,b and set c = thickness+vacuum along normal
        slab_c = normal * (thickness + vacuum)
        slab_cell = [list(a_vec), list(b_vec), list(slab_c)]
        # Shift atoms so min is at 0
        for a in slab_atoms:
            a['x'] -= pmin * normal[0]
            a['y'] -= pmin * normal[1]
            a['z'] -= pmin * normal[2]
        # Convert to fractional
        new_frac = []
        for a in slab_atoms:
            f = cart_to_frac(a['x'], a['y'], a['z'], slab_cell)
            new_frac.append({'el': a['el'], 'fx': float(f[0]) % 1.0, 'fy': float(f[1]) % 1.0, 'fz': float(f[2]) % 1.0})
        name = f"({h}{k}{l}) slab · {layers} layers"
        return self._set_from_frac(slab_cell, new_frac, name)

    def vacancy(self, idx):
        """Remove atom at index (point defect)."""
        if idx >= len(self.atoms) or idx < 0:
            self.log(f"Invalid atom index: {idx}"); return self
        el = self.atoms[idx]['el']
        self.atoms.pop(idx)
        self._frac_atoms.pop(idx)
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_crystal(self.atoms, self.cell, f"V_{el} defect")
        self._update_info(f"V_{el} defect")
        self.log(f"Vacancy: removed {el} #{idx}")
        return self

    def substitute(self, idx, new_el):
        """Substitute element at atom index."""
        if idx >= len(self.atoms) or idx < 0:
            self.log(f"Invalid atom index: {idx}"); return self
        old_el = self.atoms[idx]['el']
        self.atoms[idx]['el'] = new_el
        self._frac_atoms[idx]['el'] = new_el
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_crystal(self.atoms, self.cell, f"{old_el}→{new_el} sub")
        self._update_info(f"{old_el}→{new_el} substitution")
        self.log(f"Substituted #{idx}: {old_el} → {new_el}")
        return self

    # ── Rendering ──────────────────────────────────────────────

    def style(self, s):
        """Set render style: 'ballstick', 'spacefill', or 'wireframe'."""
        self.viewer.set_style(s)
        return self

    def select(self, i):
        self.viewer.selected_atom = i
        self.viewer._rebuild_mol()
        return self

    def show_unitcell(self, val=True):
        self.viewer.show_cell = val
        self.viewer._rebuild_wireframe()
        return self

    def show_axes_toggle(self, val=True):
        self.viewer.show_axes = val
        self.viewer._rebuild_wireframe()
        return self

    # ── Measurements ───────────────────────────────────────────

    def measure(self, i, j):
        if max(i, j) >= len(self.atoms): return 0
        a, b = self.atoms[i], self.atoms[j]
        d = math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 + (a['z'] - b['z'])**2)
        self.viewer._measurements.append(('dist', (i, j), d))
        self.log(f"Distance {i}-{j}: {d:.4f} Å")
        return d

    def angle(self, i, j, k):
        if max(i, j, k) >= len(self.atoms): return 0
        a, b, c = self.atoms[i], self.atoms[j], self.atoms[k]
        v1 = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
        v2 = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        ang = math.degrees(math.acos(max(-1, min(1, cos_a))))
        self.viewer._measurements.append(('angle', (i, j, k), ang))
        self.log(f"Angle {i}-{j}-{k}: {ang:.1f}°")
        return ang

    def clear_measurements(self):
        self.viewer._measurements.clear(); return self

    # ── Compute ────────────────────────────────────────────────

    def calculate(self, method="LJ"):
        """Quick energy calculation.
        method: 'LJ' (Lennard-Jones, universal), 'EMT' (effective medium, metals only, needs ASE).
        """
        if not self.atoms:
            self.log("No crystal loaded"); return self

        if method.upper() == "EMT":
            try:
                from ase import Atoms as ASE_Atoms
                from ase.calculators.emt import EMT
                positions = [(a['x'], a['y'], a['z']) for a in self.atoms]
                symbols = [a['el'] for a in self.atoms]
                ase_atoms = ASE_Atoms(symbols=symbols, positions=positions,
                                       cell=self.cell, pbc=True)
                ase_atoms.calc = EMT()
                energy = ase_atoms.get_potential_energy()
                self.info['energy'] = energy
                self.info['energy_per_atom'] = energy / len(self.atoms)
                self.info['method'] = 'EMT'
                self.log(f"EMT Energy: {energy:.4f} eV ({energy/len(self.atoms):.4f} eV/atom)")
                return self
            except ImportError:
                self.log("ASE not installed: pip install ase — falling back to LJ")
            except Exception as ex:
                self.log(f"EMT failed: {ex} — falling back to LJ")

        # Lennard-Jones (universal, approximate)
        sigma = 2.5  # Å
        epsilon = 0.01  # eV
        energy = 0.0
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                a, b = self.atoms[i], self.atoms[j]
                r = math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 + (a['z'] - b['z'])**2)
                if r < 0.5: r = 0.5
                if r < 3 * sigma:
                    sr6 = (sigma / r)**6
                    energy += 4 * epsilon * (sr6 * sr6 - sr6)
        self.info['energy'] = energy
        self.info['energy_per_atom'] = energy / len(self.atoms)
        self.info['method'] = 'LJ'
        self.log(f"LJ Energy: {energy:.4f} eV ({energy/len(self.atoms):.4f} eV/atom)")
        return self

    # ── Analysis overlays ──────────────────────────────────────

    def radial_distribution(self, rmax=10.0, nbins=100):
        """Compute and overlay pair distribution function g(r)."""
        if len(self.atoms) < 2:
            self.log("Need ≥2 atoms for g(r)"); return self
        dr = rmax / nbins
        hist = np.zeros(nbins)
        n = len(self.atoms)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.atoms[i], self.atoms[j]
                r = math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 + (a['z'] - b['z'])**2)
                if r < rmax:
                    bi = int(r / dr)
                    if bi < nbins: hist[bi] += 2  # count i-j and j-i
        # Normalize
        vol = cell_volume(self.cell) if self.cell else (4/3*math.pi*rmax**3)
        rho = n / vol if vol > 0 else 1
        r_arr = np.arange(nbins) * dr + dr / 2
        for bi in range(nbins):
            shell_vol = 4 * math.pi * r_arr[bi]**2 * dr
            if shell_vol > 0 and rho > 0:
                hist[bi] /= (n * rho * shell_vol)

        width, height = 340, 170
        def draw_gr(painter, w, h):
            ox, oy = w - width - 16, h - height - 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 215))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter, "g(r) — Radial Distribution")
            pad_l, pad_r, pad_t, pad_b = 36, 12, 22, 28
            ax0 = ox + pad_l; ax1 = ox + width - pad_r
            ay0 = oy + pad_t; ay1 = oy + height - pad_b
            aw = ax1 - ax0; ah = ay1 - ay0
            painter.setPen(QPen(QColor(180, 185, 200), 1))
            painter.drawLine(ax0, ay1, ax1, ay1); painter.drawLine(ax0, ay0, ax0, ay1)
            peak = max(hist) if max(hist) > 0 else 1
            path = QPainterPath(); path.moveTo(ax0, ay1)
            for bi in range(nbins):
                px = ax0 + bi / nbins * aw
                py = ay1 - (hist[bi] / peak) * ah * 0.9
                path.lineTo(px, py)
            path.lineTo(ax1, ay1); path.closeSubpath()
            grad = QLinearGradient(ax0, ay0, ax0, ay1)
            grad.setColorAt(0, QColor(70, 160, 120, 160)); grad.setColorAt(1, QColor(70, 160, 120, 20))
            painter.setBrush(grad); painter.setPen(QPen(QColor(40, 130, 90), 1.2))
            painter.drawPath(path)
            # x-axis labels
            painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(100, 110, 130))
            for rv in range(0, int(rmax) + 1, max(1, int(rmax) // 5)):
                tx = ax0 + rv / rmax * aw
                painter.drawText(int(tx) - 10, ay1 + 4, 20, 14, Qt.AlignCenter, f"{rv}")
            painter.drawText(ax0, ay1 + 14, aw, 12, Qt.AlignCenter, "r (Å)")
        self.viewer.add_overlay('rdf', draw_gr)
        self.log(f"g(r) overlay: {nbins} bins, rmax={rmax} Å")
        return self

    def xrd(self, wavelength=1.5406, two_theta_max=90):
        """Compute and overlay simulated powder X-ray diffraction pattern."""
        if not self.cell or not self.atoms:
            self.log("No crystal for XRD"); return self
        a_v = np.array(self.cell[0]); b_v = np.array(self.cell[1]); c_v = np.array(self.cell[2])
        V = np.dot(a_v, np.cross(b_v, c_v))
        # Reciprocal vectors
        ra = np.cross(b_v, c_v) / V
        rb = np.cross(c_v, a_v) / V
        rc = np.cross(a_v, b_v) / V
        # Generate reflections
        hkl_max = 6
        peaks = []  # (2theta, intensity)
        for h in range(-hkl_max, hkl_max + 1):
            for k in range(-hkl_max, hkl_max + 1):
                for l in range(-hkl_max, hkl_max + 1):
                    if h == 0 and k == 0 and l == 0: continue
                    g = h * ra + k * rb + l * rc
                    d = 1.0 / np.linalg.norm(g)
                    sin_theta = wavelength / (2 * d)
                    if abs(sin_theta) > 1: continue
                    two_theta = 2 * math.degrees(math.asin(sin_theta))
                    if two_theta > two_theta_max: continue
                    # Structure factor
                    F_real = 0; F_imag = 0
                    for a in self._frac_atoms:
                        Z = ELEMENTS.get(a['el'], {'Z': 6})['Z']
                        phase = 2 * math.pi * (h * a['fx'] + k * a['fy'] + l * a['fz'])
                        F_real += Z * math.cos(phase)
                        F_imag += Z * math.sin(phase)
                    intensity = F_real**2 + F_imag**2
                    if intensity > 0.1:
                        # Lorentz-polarization factor
                        theta_r = math.radians(two_theta / 2)
                        lp = (1 + math.cos(2 * theta_r)**2) / (math.sin(theta_r)**2 * math.cos(theta_r) + 1e-10)
                        peaks.append((two_theta, intensity * lp, f"({h}{k}{l})"))

        # Merge nearby peaks
        peaks.sort()
        merged = []
        for tt, I, hkl in peaks:
            found = False
            for i, (mt, mI, mhkl) in enumerate(merged):
                if abs(tt - mt) < 0.3:
                    merged[i] = (mt, mI + I, mhkl if mI > I else hkl)
                    found = True; break
            if not found:
                merged.append((tt, I, hkl))
        if not merged:
            self.log("No XRD peaks found"); return self
        Imax = max(I for _, I, _ in merged)

        width, height = 380, 180
        def draw_xrd(painter, w, h):
            ox, oy = w - width - 16, 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 215))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter, f"Powder XRD (λ={wavelength:.4f} Å)")
            pad_l, pad_r, pad_t, pad_b = 36, 12, 22, 28
            ax0 = ox + pad_l; ax1 = ox + width - pad_r
            ay0 = oy + pad_t; ay1 = oy + height - pad_b
            aw = ax1 - ax0; ah = ay1 - ay0
            painter.setPen(QPen(QColor(180, 185, 200), 1))
            painter.drawLine(ax0, ay1, ax1, ay1)
            # Draw peaks as sticks
            painter.setPen(QPen(QColor(60, 80, 160), 1.5))
            for tt, I, hkl in merged:
                px = ax0 + tt / two_theta_max * aw
                py = ay1 - (I / Imax) * ah * 0.9
                painter.drawLine(int(px), ay1, int(px), int(py))
            # Top labels for strongest peaks
            painter.setFont(QFont("Consolas", 6)); painter.setPen(QColor(160, 50, 50))
            top_peaks = sorted(merged, key=lambda x: -x[1])[:8]
            for tt, I, hkl in top_peaks:
                if I / Imax > 0.08:
                    px = ax0 + tt / two_theta_max * aw
                    painter.drawText(int(px) - 16, ay0 - 2, 32, 12, Qt.AlignCenter, hkl)
            # x-axis
            painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(100, 110, 130))
            for ang in range(0, two_theta_max + 1, 15):
                tx = ax0 + ang / two_theta_max * aw
                painter.drawText(int(tx) - 10, ay1 + 4, 20, 14, Qt.AlignCenter, f"{ang}°")
            painter.drawText(ax0, ay1 + 14, aw, 12, Qt.AlignCenter, "2θ")
        self.viewer.add_overlay('xrd', draw_xrd)
        self.log(f"XRD overlay: {len(merged)} peaks, λ={wavelength} Å")
        return self

    def overlay_dos(self, energies, dos, label="DOS"):
        """Overlay a density of states plot. energies, dos: lists/arrays of same length."""
        energies = np.array(energies); dos = np.array(dos)
        width, height = 170, 300
        def draw_dos(painter, w, h):
            ox, oy = w - width - 16, h // 2 - height // 2
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 210))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter, label)
            pad = 24
            ax0 = ox + pad; ax1 = ox + width - 12
            ay0 = oy + pad; ay1 = oy + height - 20
            aw = ax1 - ax0; ah = ay1 - ay0
            emin, emax = energies.min(), energies.max()
            dmax = dos.max() if dos.max() > 0 else 1
            # Draw DOS curve (energy on Y, DOS on X)
            path = QPainterPath()
            path.moveTo(ax0, ay1)
            for i in range(len(energies)):
                px = ax0 + (dos[i] / dmax) * aw * 0.9
                py = ay1 - (energies[i] - emin) / (emax - emin) * ah
                if i == 0: path.moveTo(ax0, py)
                path.lineTo(px, py)
            path.lineTo(ax0, ay0)
            grad = QLinearGradient(ax0, oy, ax1, oy)
            grad.setColorAt(0, QColor(100, 50, 180, 120)); grad.setColorAt(1, QColor(100, 50, 180, 30))
            painter.setBrush(grad); painter.setPen(QPen(QColor(80, 40, 160), 1.2))
            painter.drawPath(path)
            # Fermi level (if 0 is in range)
            if emin <= 0 <= emax:
                fy = ay1 - (0 - emin) / (emax - emin) * ah
                painter.setPen(QPen(QColor(200, 50, 50), 1, Qt.DashLine))
                painter.drawLine(ax0, int(fy), ax1, int(fy))
                painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(200, 50, 50))
                painter.drawText(ax1 - 20, int(fy) - 12, 24, 12, Qt.AlignCenter, "Ef")
        self.viewer.add_overlay('dos', draw_dos)
        self.log(f"DOS overlay added")
        return self

    def overlay_bands(self, kpath, bands, labels=None):
        """Overlay band structure. kpath: 1D array of k-distances, bands: 2D [n_bands × n_kpts]."""
        kpath = np.array(kpath); bands = np.array(bands)
        width, height = 320, 260
        def draw_bands(painter, w, h):
            ox, oy = 16, h - height - 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 215))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter, "Band Structure")
            pad_l, pad_r, pad_t, pad_b = 36, 12, 24, 24
            ax0 = ox + pad_l; ax1 = ox + width - pad_r
            ay0 = oy + pad_t; ay1 = oy + height - pad_b
            aw = ax1 - ax0; ah = ay1 - ay0
            kmin, kmax = kpath.min(), kpath.max()
            emin, emax = bands.min(), bands.max()
            # Axes
            painter.setPen(QPen(QColor(180, 185, 200), 1))
            painter.drawLine(ax0, ay1, ax1, ay1); painter.drawLine(ax0, ay0, ax0, ay1)
            # Fermi level
            if emin <= 0 <= emax:
                fy = ay1 - (0 - emin) / (emax - emin) * ah
                painter.setPen(QPen(QColor(200, 50, 50), 1, Qt.DashLine))
                painter.drawLine(ax0, int(fy), ax1, int(fy))
            # Bands
            colors_cycle = [QColor(40, 80, 180), QColor(180, 60, 40), QColor(40, 150, 80),
                           QColor(160, 120, 40), QColor(120, 40, 160)]
            for bi in range(bands.shape[0]):
                painter.setPen(QPen(colors_cycle[bi % len(colors_cycle)], 1.2))
                path = QPainterPath()
                for ki in range(bands.shape[1]):
                    px = ax0 + (kpath[ki] - kmin) / (kmax - kmin + 1e-10) * aw
                    py = ay1 - (bands[bi, ki] - emin) / (emax - emin + 1e-10) * ah
                    if ki == 0: path.moveTo(px, py)
                    else: path.lineTo(px, py)
                painter.setBrush(Qt.NoBrush); painter.drawPath(path)
            # k-labels
            if labels:
                painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(80, 90, 110))
                for kl, lbl in labels:
                    tx = ax0 + (kl - kmin) / (kmax - kmin + 1e-10) * aw
                    painter.drawText(int(tx) - 10, ay1 + 4, 20, 14, Qt.AlignCenter, lbl)
            # y-axis label
            painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(100, 110, 130))
            painter.drawText(ox + 2, oy + height // 2 - 6, 30, 12, Qt.AlignCenter, "E(eV)")
        self.viewer.add_overlay('bands', draw_bands)
        self.log(f"Band structure overlay: {bands.shape[0]} bands, {bands.shape[1]} k-points")
        return self

    def remove_overlay(self, name):
        self.viewer.remove_overlay(name); return self

    # ── Labels ─────────────────────────────────────────────────

    def label_atoms(self):
        def draw_labels(painter, w, h):
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor(60, 60, 80))
            for i, a in enumerate(self.atoms):
                painter.drawText(w // 2, 80 + i * 13, f"{i}: {a['el']} ({a['x']:.2f},{a['y']:.2f},{a['z']:.2f})")
        self.viewer.add_overlay('labels', draw_labels)
        return self

    def label_elements(self):
        """Show element symbols near each atom in the HUD."""
        def draw_el(painter, w, h):
            painter.setFont(QFont("Consolas", 8)); painter.setPen(QColor(60, 80, 120))
            y = 100
            seen = set()
            for a in self.atoms:
                if a['el'] not in seen:
                    seen.add(a['el'])
                    el = ELEMENTS.get(a['el'], {})
                    painter.drawText(w // 2, y, f"{a['el']} — {el.get('name','?')}, Z={el.get('Z','?')}")
                    y += 14
        self.viewer.add_overlay('elements', draw_el)
        return self

    # ── Export ─────────────────────────────────────────────────

    def export_poscar(self, path="~/POSCAR"):
        """Export current structure as VASP POSCAR."""
        path = os.path.expanduser(path)
        species_order = []
        for a in self.atoms:
            if a['el'] not in species_order: species_order.append(a['el'])
        counts = [sum(1 for a in self.atoms if a['el'] == sp) for sp in species_order]
        with open(path, 'w') as f:
            f.write(f"{self.info.get('name', 'crystal')}\n")
            f.write("1.0\n")
            for v in self.cell:
                f.write(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n")
            f.write("  ".join(species_order) + "\n")
            f.write("  ".join(str(c) for c in counts) + "\n")
            f.write("Direct\n")
            for sp in species_order:
                for a in self.atoms:
                    if a['el'] == sp:
                        f.write(f"  {a.get('fx', 0):.10f}  {a.get('fy', 0):.10f}  {a.get('fz', 0):.10f}\n")
        self.log(f"Exported POSCAR: {path}")
        return path

    def export_cif(self, path="~/structure.cif"):
        """Export as minimal CIF."""
        path = os.path.expanduser(path)
        cp = cell_params(self.cell)
        with open(path, 'w') as f:
            f.write("data_crystal\n")
            f.write(f"_cell_length_a {cp[0]:.6f}\n")
            f.write(f"_cell_length_b {cp[1]:.6f}\n")
            f.write(f"_cell_length_c {cp[2]:.6f}\n")
            f.write(f"_cell_angle_alpha {cp[3]:.4f}\n")
            f.write(f"_cell_angle_beta {cp[4]:.4f}\n")
            f.write(f"_cell_angle_gamma {cp[5]:.4f}\n")
            f.write("loop_\n")
            f.write("_atom_site_type_symbol\n")
            f.write("_atom_site_fract_x\n")
            f.write("_atom_site_fract_y\n")
            f.write("_atom_site_fract_z\n")
            for a in self._frac_atoms:
                f.write(f"{a['el']}  {a['fx']:.8f}  {a['fy']:.8f}  {a['fz']:.8f}\n")
        self.log(f"Exported CIF: {path}")
        return path

    def export_xyz(self, path="~/crystal.xyz"):
        """Export as extended XYZ with lattice."""
        path = os.path.expanduser(path)
        c = self.cell
        lat_str = " ".join(f"{c[i][j]:.6f}" for i in range(3) for j in range(3))
        with open(path, 'w') as f:
            f.write(f"{len(self.atoms)}\n")
            f.write(f'Lattice="{lat_str}" Properties=species:S:1:pos:R:3\n')
            for a in self.atoms:
                f.write(f"{a['el']}  {a['x']:.6f}  {a['y']:.6f}  {a['z']:.6f}\n")
        self.log(f"Exported XYZ: {path}")
        return path

    def export_qe_input(self, path="~/pw.in", ecutwfc=40, ecutrho=320, kpoints=(4,4,4)):
        """Export Quantum ESPRESSO pw.x input file."""
        path = os.path.expanduser(path)
        species = list(set(a['el'] for a in self.atoms))
        cp = cell_params(self.cell)
        with open(path, 'w') as f:
            f.write("&CONTROL\n  calculation = 'scf'\n  prefix = 'crystal'\n  outdir = './tmp'\n  pseudo_dir = './pseudo'\n/\n")
            f.write(f"&SYSTEM\n  ibrav = 0\n  nat = {len(self.atoms)}\n  ntyp = {len(species)}\n")
            f.write(f"  ecutwfc = {ecutwfc}\n  ecutrho = {ecutrho}\n/\n")
            f.write("&ELECTRONS\n  mixing_beta = 0.7\n  conv_thr = 1.0d-8\n/\n")
            f.write("ATOMIC_SPECIES\n")
            for sp in species:
                mass = ELEMENTS.get(sp, {'mass': 1.0})['mass']
                f.write(f"  {sp}  {mass:.4f}  {sp}.UPF\n")
            f.write("CELL_PARAMETERS angstrom\n")
            for v in self.cell:
                f.write(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n")
            f.write("ATOMIC_POSITIONS crystal\n")
            for a in self._frac_atoms:
                f.write(f"  {a['el']}  {a['fx']:.10f}  {a['fy']:.10f}  {a['fz']:.10f}\n")
            f.write(f"K_POINTS automatic\n  {kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0\n")
        self.log(f"Exported QE input: {path}")
        return path

    # ── Analysis helpers ───────────────────────────────────────

    def formula(self):
        counts = {}
        for a in self.atoms: counts[a['el']] = counts.get(a['el'], 0) + 1
        return ''.join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in sorted(counts.keys()))

    def mass(self):
        return sum(ELEMENTS.get(a['el'], {'mass': 0})['mass'] for a in self.atoms)

    def nearest_neighbors(self, idx, n=6):
        """Find n nearest neighbors of atom idx. Returns [(neighbor_idx, distance), ...]."""
        if idx >= len(self.atoms): return []
        a = self.atoms[idx]
        dists = []
        for j, b in enumerate(self.atoms):
            if j == idx: continue
            d = math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 + (a['z'] - b['z'])**2)
            dists.append((j, d))
        dists.sort(key=lambda x: x[1])
        return dists[:n]

    def coordination_number(self, idx, cutoff=None):
        """Count atoms within cutoff distance. Auto-detects cutoff if None."""
        if idx >= len(self.atoms): return 0
        if cutoff is None:
            # Use 1.3 × sum of covalent radii as default
            el = self.atoms[idx]['el']
            r0 = ELEMENTS.get(el, {'cov': 1.0})['cov']
            cutoff = 3.0  # reasonable default
        a = self.atoms[idx]
        count = 0
        for j, b in enumerate(self.atoms):
            if j == idx: continue
            d = math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 + (a['z'] - b['z'])**2)
            if d < cutoff: count += 1
        return count


# ═══════════════════════════════════════════════════════════════
#  LIGHT-MODE STYLESHEET
# ═══════════════════════════════════════════════════════════════

_SS = """
QWidget{background:rgba(255,255,255,220);color:#2a2e3a;font-family:'Consolas','Menlo',monospace;font-size:11px}
QPushButton{background:rgba(245,247,250,240);border:1px solid #d0d5e0;border-radius:5px;padding:6px 10px;color:#3a6090;font-weight:bold;font-size:10px}
QPushButton:hover{background:rgba(230,238,248,250);border-color:#a0b8d0}
QPushButton:checked{background:rgba(59,130,200,30);border-color:#5a9fd0;color:#2a70b0}
QSlider::groove:horizontal{height:4px;background:#d8dce6;border-radius:2px}
QSlider::handle:horizontal{background:#5a9fd0;width:14px;margin:-5px 0;border-radius:7px}
QComboBox{background:rgba(250,251,253,240);border:1px solid #d0d5e0;border-radius:4px;padding:5px 8px}
QComboBox QAbstractItemView{background:white;border:1px solid #d0d5e0;selection-background-color:#e8f0fa}
QTextEdit{background:rgba(250,251,253,220);border:1px solid #dce0e8;border-radius:4px;font-size:10px;color:#4a5568;padding:6px}
QCheckBox{spacing:8px;color:#4a5568}
QTabWidget::pane{border:1px solid #d8dce6;background:rgba(255,255,255,200);border-top:none}
QTabBar::tab{background:rgba(240,242,246,200);color:#6a7a8a;padding:7px 14px;font-size:10px;font-weight:bold;border:1px solid #d8dce6;border-bottom:none}
QTabBar::tab:selected{background:rgba(255,255,255,240);color:#3a6090;border-bottom:2px solid #5a9fd0}
QListWidget{background:rgba(250,251,253,220);border:1px solid #dce0e8;border-radius:4px;font-size:10px;color:#4a5568}
QListWidget::item:selected{background:#e8f0fa;color:#2a60a0}
QLabel{background:transparent}
QScrollArea{border:none;background:transparent}
QLineEdit{background:rgba(250,251,253,240);border:1px solid #d0d5e0;border-radius:4px;padding:5px 8px}
QSpinBox{background:rgba(250,251,253,240);border:1px solid #d0d5e0;border-radius:4px;padding:4px 6px}
"""

def _lbl(text):
    l = QLabel(text.upper())
    l.setStyleSheet("font-size:9px;letter-spacing:1.5px;color:#8a96a8;font-weight:bold;padding:2px 0;background:transparent")
    return l


# ═══════════════════════════════════════════════════════════════
#  MaterialLabApp — ALL UI ASSEMBLY & WIRING IN ONE CLASS
# ═══════════════════════════════════════════════════════════════

class MaterialLabApp:
    """Encapsulates the entire MatLab UI: widgets, signals, file browser, and crystal singleton."""

    def __init__(self):
        self._cur_dir = os.path.expanduser("~")
        self._build_ui()
        self._create_singleton()
        self._wire_signals()
        self._refresh_files()
        self.matlab_crystal.load(self.matlab_struct_combo.currentText())

    # ── UI Construction ────────────────────────────────────────

    def _build_ui(self):
        self.matlab_main_widget = QWidget()
        self.matlab_main_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.matlab_main_layout = QHBoxLayout(self.matlab_main_widget)
        self.matlab_main_layout.setContentsMargins(0, 0, 0, 0); self.matlab_main_layout.setSpacing(0)

        # ── Panel ──
        self.matlab_panel = QWidget(); self.matlab_panel.setFixedWidth(290); self.matlab_panel.setStyleSheet(_SS)
        self.matlab_panel.setAttribute(Qt.WA_TranslucentBackground, True)
        self.matlab_scroll = QScrollArea(); self.matlab_scroll.setWidgetResizable(True); self.matlab_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.matlab_scroll.setStyleSheet("QScrollArea{border:none;background:transparent}QScrollBar:vertical{width:5px;background:transparent}QScrollBar::handle:vertical{background:#c0c8d4;border-radius:2px;min-height:30px}")
        self.matlab_inner = QWidget(); self.matlab_inner.setAttribute(Qt.WA_TranslucentBackground, True)
        self.matlab_lay = QVBoxLayout(self.matlab_inner); self.matlab_lay.setSpacing(4); self.matlab_lay.setContentsMargins(10, 10, 10, 10)

        # Header
        matlab_hdr = QWidget(); matlab_hdr.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_hl = QHBoxLayout(matlab_hdr); matlab_hl.setContentsMargins(0, 0, 0, 4)
        matlab_ic = QLabel("🔬"); matlab_ic.setStyleSheet("font-size:20px;background:rgba(230,240,250,200);border:1px solid #d0d8e4;border-radius:7px;padding:3px 7px")
        matlab_nw = QWidget(); matlab_nw.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_nl = QVBoxLayout(matlab_nw); matlab_nl.setContentsMargins(6, 0, 0, 0); matlab_nl.setSpacing(0)
        self.matlab_title_lbl = QLabel("MatLab"); self.matlab_title_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#1a2a40;background:transparent")
        self.matlab_sub_lbl = QLabel("MATERIALS SCIENCE WORKBENCH"); self.matlab_sub_lbl.setStyleSheet("font-size:7px;letter-spacing:2px;color:#8a96a8;background:transparent")
        matlab_nl.addWidget(self.matlab_title_lbl); matlab_nl.addWidget(self.matlab_sub_lbl)
        matlab_hl.addWidget(matlab_ic); matlab_hl.addWidget(matlab_nw); matlab_hl.addStretch()
        self.matlab_lay.addWidget(matlab_hdr)

        self.matlab_tabs = QTabWidget(); self.matlab_tabs.setStyleSheet(_SS)

        # ── Tab 1: Structure ──
        matlab_t1 = QWidget(); matlab_t1.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_t1l = QVBoxLayout(matlab_t1); matlab_t1l.setSpacing(5); matlab_t1l.setContentsMargins(6, 8, 6, 6)
        matlab_t1l.addWidget(_lbl("Presets"))
        self.matlab_struct_combo = QComboBox(); self.matlab_struct_combo.addItems(list(LATTICES.keys())); matlab_t1l.addWidget(self.matlab_struct_combo)

        matlab_t1l.addWidget(_lbl("Style"))
        matlab_sw = QWidget(); matlab_sw.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_sl = QHBoxLayout(matlab_sw); matlab_sl.setContentsMargins(0, 0, 0, 0); matlab_sl.setSpacing(2)
        self.matlab_style_btns = {}
        for sn, slb in [("ballstick", "Ball&Stick"), ("spacefill", "SpaceFill"), ("wireframe", "Wire")]:
            b = QPushButton(slb); b.setCheckable(True); b.setChecked(sn == "ballstick"); self.matlab_style_btns[sn] = b; matlab_sl.addWidget(b)
        matlab_t1l.addWidget(matlab_sw)

        self.matlab_cell_cb = QCheckBox("Unit Cell"); self.matlab_cell_cb.setChecked(True); matlab_t1l.addWidget(self.matlab_cell_cb)
        self.matlab_axes_cb = QCheckBox("Lattice Axes"); self.matlab_axes_cb.setChecked(True); matlab_t1l.addWidget(self.matlab_axes_cb)
        self.matlab_bonds_cb = QCheckBox("Show Bonds"); self.matlab_bonds_cb.setChecked(True); matlab_t1l.addWidget(self.matlab_bonds_cb)

        matlab_t1l.addWidget(_lbl("Supercell"))
        matlab_sc_w = QWidget(); matlab_sc_w.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_sc_l = QHBoxLayout(matlab_sc_w); matlab_sc_l.setContentsMargins(0, 0, 0, 0); matlab_sc_l.setSpacing(4)
        self.matlab_sc_a = QSpinBox(); self.matlab_sc_a.setRange(1, 6); self.matlab_sc_a.setValue(1); self.matlab_sc_a.setPrefix("a×")
        self.matlab_sc_b = QSpinBox(); self.matlab_sc_b.setRange(1, 6); self.matlab_sc_b.setValue(1); self.matlab_sc_b.setPrefix("b×")
        self.matlab_sc_c = QSpinBox(); self.matlab_sc_c.setRange(1, 6); self.matlab_sc_c.setValue(1); self.matlab_sc_c.setPrefix("c×")
        self.matlab_sc_btn = QPushButton("Build")
        matlab_sc_l.addWidget(self.matlab_sc_a); matlab_sc_l.addWidget(self.matlab_sc_b); matlab_sc_l.addWidget(self.matlab_sc_c); matlab_sc_l.addWidget(self.matlab_sc_btn)
        matlab_t1l.addWidget(matlab_sc_w)

        matlab_t1l.addStretch(); self.matlab_tabs.addTab(matlab_t1, "Structure")

        # ── Tab 2: Analysis ──
        matlab_t2 = QWidget(); matlab_t2.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_t2l = QVBoxLayout(matlab_t2); matlab_t2l.setSpacing(5); matlab_t2l.setContentsMargins(6, 8, 6, 6)
        matlab_t2l.addWidget(_lbl("Quick Compute"))
        self.matlab_calc_combo = QComboBox(); self.matlab_calc_combo.addItems(["LJ", "EMT"]); matlab_t2l.addWidget(self.matlab_calc_combo)
        self.matlab_calc_btn = QPushButton("Calculate Energy"); matlab_t2l.addWidget(self.matlab_calc_btn)
        matlab_t2l.addWidget(_lbl("Overlays"))
        self.matlab_rdf_btn = QPushButton("g(r) — Radial Dist."); matlab_t2l.addWidget(self.matlab_rdf_btn)
        self.matlab_xrd_btn = QPushButton("Powder XRD"); matlab_t2l.addWidget(self.matlab_xrd_btn)
        self.matlab_clear_overlays_btn = QPushButton("Clear Overlays"); matlab_t2l.addWidget(self.matlab_clear_overlays_btn)
        matlab_t2l.addStretch(); self.matlab_tabs.addTab(matlab_t2, "Analysis")

        # ── Tab 3: Import/Export ──
        matlab_t3 = QWidget(); matlab_t3.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_t3l = QVBoxLayout(matlab_t3); matlab_t3l.setSpacing(5); matlab_t3l.setContentsMargins(6, 8, 6, 6)
        matlab_t3l.addWidget(_lbl("Load .cif / POSCAR / .xyz"))
        self.matlab_path_edit = QLineEdit(); self.matlab_path_edit.setPlaceholderText("~/path/to/structure.cif"); matlab_t3l.addWidget(self.matlab_path_edit)
        self.matlab_file_list = QListWidget(); self.matlab_file_list.setMinimumHeight(100); self.matlab_file_list.setMaximumHeight(140); matlab_t3l.addWidget(self.matlab_file_list)
        self.matlab_load_btn = QPushButton("Load Selected"); matlab_t3l.addWidget(self.matlab_load_btn)
        matlab_t3l.addWidget(_lbl("Export"))
        matlab_ew = QWidget(); matlab_ew.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_el = QHBoxLayout(matlab_ew); matlab_el.setContentsMargins(0, 0, 0, 0); matlab_el.setSpacing(2)
        self.matlab_poscar_btn = QPushButton("POSCAR"); self.matlab_cif_btn = QPushButton("CIF"); self.matlab_xyz_btn = QPushButton("XYZ"); self.matlab_qe_btn = QPushButton("QE .in")
        matlab_el.addWidget(self.matlab_poscar_btn); matlab_el.addWidget(self.matlab_cif_btn); matlab_el.addWidget(self.matlab_xyz_btn); matlab_el.addWidget(self.matlab_qe_btn)
        matlab_t3l.addWidget(matlab_ew)
        self.matlab_status_lbl = QLabel(""); self.matlab_status_lbl.setWordWrap(True); self.matlab_status_lbl.setStyleSheet("color:#6a7a8a;font-size:10px;background:transparent"); matlab_t3l.addWidget(self.matlab_status_lbl)
        matlab_t3l.addStretch(); self.matlab_tabs.addTab(matlab_t3, "Import")

        # ── Tab 4: Info ──
        matlab_t4 = QWidget(); matlab_t4.setAttribute(Qt.WA_TranslucentBackground, True)
        matlab_t4l = QVBoxLayout(matlab_t4); matlab_t4l.setSpacing(5); matlab_t4l.setContentsMargins(6, 8, 6, 6)
        matlab_t4l.addWidget(_lbl("Properties"))
        self.matlab_info_edit = QTextEdit(); self.matlab_info_edit.setReadOnly(True); self.matlab_info_edit.setMinimumHeight(90); matlab_t4l.addWidget(self.matlab_info_edit)
        matlab_t4l.addWidget(_lbl("Atoms"))
        self.matlab_atom_list = QListWidget(); self.matlab_atom_list.setMinimumHeight(100); matlab_t4l.addWidget(self.matlab_atom_list)
        matlab_t4l.addWidget(_lbl("Log"))
        self.matlab_log_edit = QTextEdit(); self.matlab_log_edit.setReadOnly(True)
        self.matlab_log_edit.setPlainText("[MatLab] Initialised\n[MatLab] Renderer: ModernGL\n"); matlab_t4l.addWidget(self.matlab_log_edit)
        matlab_t4l.addStretch(); self.matlab_tabs.addTab(matlab_t4, "Info")

        self.matlab_lay.addWidget(self.matlab_tabs); self.matlab_scroll.setWidget(self.matlab_inner)
        matlab_playout = QVBoxLayout(self.matlab_panel); matlab_playout.setContentsMargins(0, 0, 0, 0); matlab_playout.addWidget(self.matlab_scroll)
        self.matlab_viewer = CrystalViewer()
        self.matlab_viewer.setStyleSheet("background:transparent")
        self.matlab_main_layout.addWidget(self.matlab_panel); self.matlab_main_layout.addWidget(self.matlab_viewer, 1)

    # ── Singleton & Wiring ─────────────────────────────────────

    def _create_singleton(self):
        self.matlab_crystal = MatLab(self.matlab_viewer, self.matlab_log_edit)
        self.matlab_mat = self.matlab_crystal  # alias

    def _update_ui(self):
        """Sync UI with crystal state."""
        self.matlab_atom_list.clear()
        for i, a in enumerate(self.matlab_crystal.atoms):
            el = ELEMENTS.get(a['el'], {'name': '?'})
            self.matlab_atom_list.addItem(f"{i:3d} {a['el']:2s} ({a['x']:+.3f},{a['y']:+.3f},{a['z']:+.3f})")
        info_lines = [f"Name: {self.matlab_crystal.info.get('name', '')}"]
        info_lines.append(f"Formula: {self.matlab_crystal.info.get('formula', '')}")
        info_lines.append(f"Atoms: {self.matlab_crystal.info.get('n_atoms', 0)}")
        vol = self.matlab_crystal.info.get('volume', 0)
        if vol: info_lines.append(f"Volume: {vol:.2f} ų")
        density = self.matlab_crystal.info.get('density', 0)
        if density: info_lines.append(f"Density: {density:.3f} g/cm³")
        cp = self.matlab_crystal.info.get('lattice')
        if cp:
            info_lines.append(f"a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f}")
            info_lines.append(f"α={cp[3]:.1f}° β={cp[4]:.1f}° γ={cp[5]:.1f}°")
        sg = self.matlab_crystal.info.get('spacegroup')
        if sg: info_lines.append(f"Space group: {sg}")
        energy = self.matlab_crystal.info.get('energy')
        if energy is not None: info_lines.append(f"Energy: {energy:.4f} eV ({self.matlab_crystal.info.get('method','')})")
        epa = self.matlab_crystal.info.get('energy_per_atom')
        if epa is not None: info_lines.append(f"E/atom: {epa:.4f} eV")
        self.matlab_info_edit.setPlainText('\n'.join(info_lines))

    def _wire_signals(self):
        crystal = self.matlab_crystal

        # Wrap methods to auto-update UI
        _orig_load = crystal.load
        def _load_wrap(name): _orig_load(name); self._update_ui(); return crystal
        crystal.load = _load_wrap

        _orig_load_cif = crystal.load_cif
        def _load_cif_wrap(p): _orig_load_cif(p); self._update_ui(); return crystal
        crystal.load_cif = _load_cif_wrap

        _orig_load_poscar = crystal.load_poscar
        def _load_poscar_wrap(p): _orig_load_poscar(p); self._update_ui(); return crystal
        crystal.load_poscar = _load_poscar_wrap

        _orig_load_xyz = crystal.load_xyz
        def _load_xyz_wrap(p): _orig_load_xyz(p); self._update_ui(); return crystal
        crystal.load_xyz = _load_xyz_wrap

        _orig_supercell = crystal.supercell
        def _supercell_wrap(a, b, c): _orig_supercell(a, b, c); self._update_ui(); return crystal
        crystal.supercell = _supercell_wrap

        _orig_vacancy = crystal.vacancy
        def _vacancy_wrap(i): _orig_vacancy(i); self._update_ui(); return crystal
        crystal.vacancy = _vacancy_wrap

        _orig_substitute = crystal.substitute
        def _sub_wrap(i, el): _orig_substitute(i, el); self._update_ui(); return crystal
        crystal.substitute = _sub_wrap

        _orig_calculate = crystal.calculate
        def _calc_wrap(m="LJ"): _orig_calculate(m); self._update_ui(); return crystal
        crystal.calculate = _calc_wrap

        _orig_cleave = crystal.cleave
        def _cleave_wrap(h, k, l, layers=4, vacuum=10.0): _orig_cleave(h, k, l, layers, vacuum); self._update_ui(); return crystal
        crystal.cleave = _cleave_wrap

        # UI signal wiring
        self.matlab_struct_combo.currentTextChanged.connect(lambda t: crystal.load(t))

        def _on_style(sn):
            for k, b in self.matlab_style_btns.items(): b.setChecked(k == sn)
            crystal.style(sn)
        for sn, bt in self.matlab_style_btns.items(): bt.clicked.connect(lambda c, s=sn: _on_style(s))

        self.matlab_cell_cb.toggled.connect(lambda c: crystal.show_unitcell(c))
        self.matlab_axes_cb.toggled.connect(lambda c: crystal.show_axes_toggle(c))
        self.matlab_bonds_cb.toggled.connect(lambda c: (setattr(self.matlab_viewer, 'show_bonds_flag', c), self.matlab_viewer._rebuild_mol()))

        self.matlab_sc_btn.clicked.connect(lambda: crystal.supercell(self.matlab_sc_a.value(), self.matlab_sc_b.value(), self.matlab_sc_c.value()))
        self.matlab_calc_btn.clicked.connect(lambda: crystal.calculate(self.matlab_calc_combo.currentText()))
        self.matlab_rdf_btn.clicked.connect(lambda: crystal.radial_distribution())
        self.matlab_xrd_btn.clicked.connect(lambda: crystal.xrd())
        self.matlab_clear_overlays_btn.clicked.connect(lambda: (self.matlab_viewer._overlays.clear(), self.matlab_viewer.update()))

        self.matlab_poscar_btn.clicked.connect(lambda: (crystal.export_poscar(), self.matlab_status_lbl.setText("✓ Saved ~/POSCAR")))
        self.matlab_cif_btn.clicked.connect(lambda: (crystal.export_cif(), self.matlab_status_lbl.setText("✓ Saved ~/structure.cif")))
        self.matlab_xyz_btn.clicked.connect(lambda: (crystal.export_xyz(), self.matlab_status_lbl.setText("✓ Saved ~/crystal.xyz")))
        self.matlab_qe_btn.clicked.connect(lambda: (crystal.export_qe_input(), self.matlab_status_lbl.setText("✓ Saved ~/pw.in")))

        self.matlab_atom_list.currentRowChanged.connect(lambda r: crystal.select(r) if 0 <= r < len(crystal.atoms) else None)

        # File browser
        self.matlab_file_list.itemDoubleClicked.connect(self._on_file_dblclick)
        self.matlab_path_edit.returnPressed.connect(lambda: self._refresh_files(self.matlab_path_edit.text().strip()) if os.path.isdir(self.matlab_path_edit.text().strip()) else None)
        self.matlab_load_btn.clicked.connect(self._on_load_selected)

    # ── File Browser ───────────────────────────────────────────

    def _refresh_files(self, d=None):
        if d: self._cur_dir = d
        self.matlab_path_edit.setText(self._cur_dir)
        self.matlab_file_list.clear()
        try:
            self.matlab_file_list.addItem("← ..")
            entries = sorted(os.listdir(self._cur_dir))
            dirs = [e for e in entries if os.path.isdir(os.path.join(self._cur_dir, e)) and not e.startswith('.')]
            files = [e for e in entries if os.path.isfile(os.path.join(self._cur_dir, e)) and not e.startswith('.')]
            shown = [f for f in files if f.endswith(('.cif', '.vasp', '.xyz', '.poscar', '.contcar'))]
            if not shown: shown = [f for f in files if any(f.upper().startswith(x) for x in ['POSCAR', 'CONTCAR'])]
            if not shown: shown = files[:30]
            for d in dirs[:20]: self.matlab_file_list.addItem(f"📁 {d}")
            for f in shown: self.matlab_file_list.addItem(f"🟦 {f}" if f.endswith('.cif') else f"📄 {f}")
        except: self.matlab_file_list.addItem("(error)")

    def _on_file_dblclick(self, item):
        name = item.text().split(" ", 1)[-1].strip() if " " in item.text() else item.text().strip()
        if item.text().startswith("←"):
            parent = os.path.dirname(self._cur_dir)
            if parent != self._cur_dir: self._refresh_files(parent)
            return
        full = os.path.join(self._cur_dir, name)
        if os.path.isdir(full): self._refresh_files(full)

    def _on_load_selected(self):
        item = self.matlab_file_list.currentItem()
        if not item: self.matlab_status_lbl.setText("⚠ Select a file"); return
        name = item.text().split(" ", 1)[-1].strip() if " " in item.text() else item.text().strip()
        full = os.path.join(self._cur_dir, name)
        if os.path.isfile(full):
            ext = os.path.splitext(full)[1].lower()
            upper = os.path.basename(full).upper()
            if ext == '.cif':
                self.matlab_crystal.load_cif(full)
            elif ext == '.xyz':
                self.matlab_crystal.load_xyz(full)
            elif ext in ('.vasp',) or upper.startswith('POSCAR') or upper.startswith('CONTCAR'):
                self.matlab_crystal.load_poscar(full)
            else:
                # Try CIF first, then POSCAR
                try:
                    self.matlab_crystal.load_cif(full)
                except:
                    self.matlab_crystal.load_poscar(full)
            self.matlab_status_lbl.setText(f"✓ Loaded {name}")
        else:
            self.matlab_status_lbl.setText("⚠ Not a file")


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE
# ═══════════════════════════════════════════════════════════════

matlab_app = MaterialLabApp()

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

matlab_proxy = graphics_scene.addWidget(matlab_app.matlab_main_widget)
matlab_proxy.setPos(0, 0)
matlab_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
matlab_shadow = QGraphicsDropShadowEffect()
matlab_shadow.setBlurRadius(60); matlab_shadow.setOffset(45, 45); matlab_shadow.setColor(QColor(0, 0, 0, 120))
matlab_proxy.setGraphicsEffect(matlab_shadow)
matlab_app.matlab_main_widget.resize(1400, 850)

# Center in current view
_vr = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
matlab_proxy.setPos(_vr.center().x() - matlab_app.matlab_main_widget.width() / 2,
             _vr.center().y() - matlab_app.matlab_main_widget.height() / 2)
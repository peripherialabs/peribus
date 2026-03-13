"""
ChemLab — Molecular Chemistry Workbench for Rio
═══════════════════════════════════════════════════════════════════

ARCHITECTURE: This file is the complete runtime. It registers a `chem`
singleton into the namespace. The LLM sees this source as context and
injects short code snippets to drive it.

EXAMPLES (what the LLM would output):
    chem.load("water")                          # load preset
    chem.load_xyz("O 0 0 0\\nH 0 0.76 -0.5\\nH 0 -0.76 -0.5")
    chem.load_smiles("c1ccccc1")                # benzene from SMILES
    chem.load_cube("/path/to/homo.cube")        # real orbital from ORCA
    chem.style("spacefill")                     # change render
    chem.measure(0, 1)                          # distance between atoms
    chem.angle(0, 1, 2)                         # bond angle
    chem.label_charges()                        # show Mulliken charges
    chem.export_xyz("/tmp/mol.xyz")             # save geometry

    ## REAL QUANTUM CHEMISTRY (PySCF built-in):
    chem.calculate()                            # B3LYP/sto-3g on current molecule
    chem.calculate("B3LYP", "def2-svp")         # specify method + basis
    chem.calculate("HF", "sto-3g", freq=True)   # with vibrational frequencies
    chem.calculate("B3LYP", "def2-svp", cube=True) # + generate HOMO/LUMO cube files
    chem.calculate_full()                        # density + ESP + HOMO/LUMO in one go
    chem.calculate_full(elf=True)                # + electron localization function
    chem.show_homo()                            # show HOMO isosurface (after calculate)
    chem.show_lumo()                            # show LUMO isosurface
    chem.show_orbital(n)                        # show MO #n isosurface
    chem.overlay_ir()                           # show IR spectrum (after freq=True)

    ## VOLUMETRIC DATA (sections 7.1–7.7):
    # 7.1 — Electron density + Electrostatic potential:
    chem.show_esp(density_iso=0.05)              # ESP color-mapped onto density surface
    chem.load_esp("dens.cube", "esp.cube")       # from external cube files
    chem.show_volume_slice(axis=1, pos=0.5)      # 2D cross-section through volume
    chem.clear_volume_slice()

    # 7.2 — Localized (Wannier) orbitals:
    chem.show_localized_orbitals()               # Boys-localized MOs (bonds + lone pairs)

    # 7.3 — Electron Localization Function:
    chem.show_elf(isovalue=0.85)                 # ELF from PySCF
    chem.load_elf("elf.cube")                    # from external cube file

    # 7.4 — Density differences / cube arithmetic:
    chem.density_difference("a.cube", "b.cube")  # visualize A - B
    chem.cube_math('subtract', cubeA, cubeB)     # general cube math
    chem.cube_math('add', cubeA, cubeB, scale=2) # weighted addition
    chem.save_cube('density', "~/out.cube")       # save any cube dataset

    # 7.5 — Bulk / periodic systems:
    chem.load_bulk("density.cube", "esp.cube")   # with centering + inbox
    chem.show_bulk_esp()                         # multi-layer ESP isosurfaces

    # 7.6 — Animated isosurfaces:
    chem.load_animation("frames/*.cube")         # load cube sequence
    chem.load_animation(["f1.cube","f2.cube"])   # or explicit list
    chem.play(interval_ms=200)                   # start playback
    chem.stop()                                  # pause
    chem.frame(5)                                # jump to frame

    # 7.7 — Spin density (open-shell):
    chem.show_spin_density()                     # alpha - beta density (after UHF/UKS calc)
    chem.load_spin_density("alpha.cube","beta.cube")  # from external cubes

    # Multi-cube overlay:
    chem.add_isosurface("extra.cube", iso=0.01, color=0xFF8000, opacity=0.4)
    chem.clear_layers()                          # remove all extra layers

    ## After calculate(), chem.info contains REAL data:
    chem.info['energy']         # total energy in Hartree
    chem.info['homo_energy']    # HOMO energy in eV
    chem.info['lumo_energy']    # LUMO energy in eV
    chem.info['gap']            # HOMO-LUMO gap in eV
    chem.info['dipole']         # dipole moment in Debye
    chem.info['charges']        # Mulliken charges per atom
    chem.info['frequencies']    # vibrational frequencies in cm-1 (if freq=True)
    chem.info['mo_energies']    # all MO energies in eV
    chem.info['n_occ']          # number of occupied MOs

    ## ORCA (external, for larger systems):
    chem.export_orca_input("/tmp/job.inp", method="B3LYP def2-TZVP Opt Freq")
    chem.run_orca("/tmp/job.inp")               # run ORCA (async)
    chem.parse_orca("/tmp/job.out")             # parse results

VIEWER API (lower level, when LLM needs custom rendering):
    chem.viewer.set_molecule(atoms, name)
    chem.viewer.set_cube(cube_data, isovalue)
    chem.viewer.set_style(style)                # ballstick/spacefill/wireframe
    chem.viewer.add_overlay(name, fn)           # fn(painter, w, h) for custom QPainter
    chem.viewer.remove_overlay(name)
    chem.viewer.cam_dist = 8.0                  # camera distance
    chem.viewer.rot_x, chem.viewer.rot_y        # camera angles
    chem.viewer.screenshot(path)                # save PNG

NAMESPACE: After this file runs, these are available:
    chem        — ChemLab singleton (main API)
    viewer      — alias for chem.viewer (the 3D GL widget)
    mol         — alias for chem (short form)
    ELEMENTS    — element data dict
    All PySide6/Qt, numpy, moderngl, glm from parser namespace
"""

import math
import time
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
    QGraphicsDropShadowEffect, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage, QLinearGradient
)

import moderngl
import glm
import json

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

BOHR = 0.529177249  # Bohr to Angstrom

def _hex(h):
    return ((h>>16)&0xFF)/255.0, ((h>>8)&0xFF)/255.0, (h&0xFF)/255.0

ELEMENTS = {
    'H':  {'color':0xEEEEEE,'r':0.31,'cov':0.31,'mass':1.008,'Z':1,'name':'Hydrogen'},
    'He': {'color':0xD9FFFF,'r':0.28,'cov':0.28,'mass':4.003,'Z':2,'name':'Helium'},
    'Li': {'color':0xCC80FF,'r':1.28,'cov':1.28,'mass':6.941,'Z':3,'name':'Lithium'},
    'Be': {'color':0xC2FF00,'r':0.96,'cov':0.96,'mass':9.012,'Z':4,'name':'Beryllium'},
    'B':  {'color':0xFFB5B5,'r':0.84,'cov':0.84,'mass':10.81,'Z':5,'name':'Boron'},
    'C':  {'color':0x333333,'r':0.76,'cov':0.77,'mass':12.01,'Z':6,'name':'Carbon'},
    'N':  {'color':0x2040D8,'r':0.71,'cov':0.71,'mass':14.01,'Z':7,'name':'Nitrogen'},
    'O':  {'color':0xDD0000,'r':0.66,'cov':0.66,'mass':16.00,'Z':8,'name':'Oxygen'},
    'F':  {'color':0x70D040,'r':0.57,'cov':0.57,'mass':19.00,'Z':9,'name':'Fluorine'},
    'Ne': {'color':0xB3E3F5,'r':0.58,'cov':0.58,'mass':20.18,'Z':10,'name':'Neon'},
    'Na': {'color':0xAB5CF2,'r':1.66,'cov':1.66,'mass':22.99,'Z':11,'name':'Sodium'},
    'Mg': {'color':0x8AFF00,'r':1.41,'cov':1.41,'mass':24.31,'Z':12,'name':'Magnesium'},
    'Al': {'color':0xBFA6A6,'r':1.21,'cov':1.21,'mass':26.98,'Z':13,'name':'Aluminum'},
    'Si': {'color':0xF0C8A0,'r':1.11,'cov':1.11,'mass':28.09,'Z':14,'name':'Silicon'},
    'P':  {'color':0xFF8000,'r':1.07,'cov':1.07,'mass':30.97,'Z':15,'name':'Phosphorus'},
    'S':  {'color':0xDDDD00,'r':1.05,'cov':1.05,'mass':32.07,'Z':16,'name':'Sulfur'},
    'Cl': {'color':0x1FF01F,'r':1.02,'cov':1.02,'mass':35.45,'Z':17,'name':'Chlorine'},
    'Ar': {'color':0x80D1E3,'r':1.06,'cov':1.06,'mass':39.95,'Z':18,'name':'Argon'},
    'K':  {'color':0x8F40D4,'r':2.03,'cov':2.03,'mass':39.10,'Z':19,'name':'Potassium'},
    'Ca': {'color':0x3DFF00,'r':1.76,'cov':1.76,'mass':40.08,'Z':20,'name':'Calcium'},
    'Ti': {'color':0xBFC2C7,'r':1.47,'cov':1.47,'mass':47.87,'Z':22,'name':'Titanium'},
    'Fe': {'color':0xE06633,'r':1.32,'cov':1.32,'mass':55.85,'Z':26,'name':'Iron'},
    'Co': {'color':0xF090A0,'r':1.26,'cov':1.26,'mass':58.93,'Z':27,'name':'Cobalt'},
    'Ni': {'color':0x50D050,'r':1.24,'cov':1.24,'mass':58.69,'Z':28,'name':'Nickel'},
    'Cu': {'color':0xC88033,'r':1.32,'cov':1.32,'mass':63.55,'Z':29,'name':'Copper'},
    'Zn': {'color':0x7D80B0,'r':1.22,'cov':1.22,'mass':65.38,'Z':30,'name':'Zinc'},
    'Br': {'color':0xA62929,'r':1.20,'cov':1.20,'mass':79.90,'Z':35,'name':'Bromine'},
    'Ru': {'color':0x248F8F,'r':1.46,'cov':1.46,'mass':101.1,'Z':44,'name':'Ruthenium'},
    'Pd': {'color':0x006985,'r':1.39,'cov':1.39,'mass':106.4,'Z':46,'name':'Palladium'},
    'I':  {'color':0x940094,'r':1.39,'cov':1.39,'mass':126.9,'Z':53,'name':'Iodine'},
    'Pt': {'color':0xD0D0E0,'r':1.36,'cov':1.36,'mass':195.1,'Z':78,'name':'Platinum'},
}

Z_TO_EL = {v['Z']: k for k, v in ELEMENTS.items()}

# Comprehensive fallback for atomic numbers not in ELEMENTS
_Z_SYMBOL_FULL = {
    1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
    11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',
    19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',
    27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',
    35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',
    43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',
    51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',
    72:'Hf',73:'Ta',74:'W',75:'Re',76:'Os',77:'Ir',78:'Pt',79:'Au',
    80:'Hg',81:'Tl',82:'Pb',83:'Bi',92:'U',
}
# Merge: ELEMENTS entries take priority, fallback fills the gaps
for _z, _sym in _Z_SYMBOL_FULL.items():
    if _z not in Z_TO_EL:
        Z_TO_EL[_z] = _sym
        # Add minimal ELEMENTS entry so rendering doesn't crash
        if _sym not in ELEMENTS:
            ELEMENTS[_sym] = {'color':0xAAAAAA,'r':1.2,'cov':1.2,'mass':_z*2.0,'Z':_z,'name':_sym}

# ── Preset molecules (Angstrom) ──
PRESETS = {
    "water": [{"el":"O","x":0,"y":0,"z":0.117},{"el":"H","x":0,"y":0.757,"z":-0.469},{"el":"H","x":0,"y":-0.757,"z":-0.469}],
    "methane": [{"el":"C","x":0,"y":0,"z":0},{"el":"H","x":0.629,"y":0.629,"z":0.629},{"el":"H","x":-0.629,"y":-0.629,"z":0.629},{"el":"H","x":-0.629,"y":0.629,"z":-0.629},{"el":"H","x":0.629,"y":-0.629,"z":-0.629}],
    "ammonia": [{"el":"N","x":0,"y":0,"z":0.38},{"el":"H","x":0,"y":0.94,"z":-0.127},{"el":"H","x":0.813,"y":-0.47,"z":-0.127},{"el":"H","x":-0.813,"y":-0.47,"z":-0.127}],
    "co2": [{"el":"C","x":0,"y":0,"z":0},{"el":"O","x":-1.16,"y":0,"z":0},{"el":"O","x":1.16,"y":0,"z":0}],
    "benzene": (lambda: [{"el":"C","x":math.cos(i*math.pi*2/6)*1.39,"y":math.sin(i*math.pi*2/6)*1.39,"z":0} for i in range(6)] + [{"el":"H","x":math.cos(i*math.pi*2/6)*2.48,"y":math.sin(i*math.pi*2/6)*2.48,"z":0} for i in range(6)])(),
    "ethanol": [{"el":"C","x":-0.748,"y":-0.015,"z":0.024},{"el":"C","x":0.748,"y":0.015,"z":-0.024},{"el":"O","x":1.165,"y":1.356,"z":0.034},{"el":"H","x":-1.145,"y":-0.524,"z":-0.862},{"el":"H","x":-1.145,"y":-0.524,"z":0.909},{"el":"H","x":-1.127,"y":1.01,"z":0.024},{"el":"H","x":1.145,"y":-0.524,"z":0.862},{"el":"H","x":1.145,"y":-0.524,"z":-0.909},{"el":"H","x":2.12,"y":1.356,"z":0.034}],
    "formaldehyde": [{"el":"C","x":0,"y":0,"z":-0.529},{"el":"O","x":0,"y":0,"z":0.677},{"el":"H","x":0,"y":0.935,"z":-1.109},{"el":"H","x":0,"y":-0.935,"z":-1.109}],
    "hcn": [{"el":"H","x":0,"y":0,"z":-1.63},{"el":"C","x":0,"y":0,"z":-0.563},{"el":"N","x":0,"y":0,"z":0.601}],
    "acetylene": [{"el":"C","x":-0.602,"y":0,"z":0},{"el":"C","x":0.602,"y":0,"z":0},{"el":"H","x":-1.664,"y":0,"z":0},{"el":"H","x":1.664,"y":0,"z":0}],
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

_FRAG_ORBITAL = """
#version 330
uniform vec3 light_dir, ambient, view_pos;
uniform float alpha;
in vec3 v_normal, v_color, v_world;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal), l = normalize(light_dir);
    float diff = abs(dot(n, l)) * 0.6 + 0.4;
    vec3 vd = normalize(view_pos - v_world);
    float spec = pow(max(abs(dot(n, normalize(l + vd))), 0.0), 32.0) * 0.25;
    vec3 col = v_color * (ambient * 0.8 + diff) + vec3(spec);
    float rim = 1.0 - abs(dot(n, vd));
    col += v_color * rim * rim * 0.3;
    frag = vec4(col, alpha);
}"""

# Volume-slice shader: renders a textured quad colored by a scalar field
_VERT_SLICE = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_uv = in_uv;
}"""

_FRAG_SLICE = """
#version 330
uniform sampler2D slice_tex;
uniform float alpha;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec4 c = texture(slice_tex, v_uv);
    frag = vec4(c.rgb, c.a * alpha);
}"""

# ═══════════════════════════════════════════════════════════════
#  GEOMETRY (numpy-accelerated)
# ═══════════════════════════════════════════════════════════════

def _make_sphere(radius, segs_h, segs_v, color_hex):
    r, g, b = _hex(color_hex)
    theta = np.linspace(0, np.pi, segs_v+1)
    phi = np.linspace(0, 2*np.pi, segs_h+1)
    x = np.outer(np.sin(theta), np.cos(phi)) * radius
    y = np.outer(np.cos(theta), np.ones(segs_h+1)) * radius
    z = np.outer(np.sin(theta), np.sin(phi)) * radius
    verts = np.empty((segs_v * segs_h * 6, 9), dtype=np.float32)
    idx = 0
    for j in range(segs_v):
        for i in range(segs_h):
            p00 = np.array([x[j,i],y[j,i],z[j,i]])
            p10 = np.array([x[j+1,i],y[j+1,i],z[j+1,i]])
            p11 = np.array([x[j+1,i+1],y[j+1,i+1],z[j+1,i+1]])
            p01 = np.array([x[j,i+1],y[j,i+1],z[j,i+1]])
            for p in (p00,p10,p11,p00,p11,p01):
                n = p / (np.linalg.norm(p)+1e-9)
                verts[idx] = [p[0],p[1],p[2],n[0],n[1],n[2],r,g,b]
                idx += 1
    return verts[:idx].flatten().tolist()

def _make_cylinder(radius, h, segs, color_hex):
    cr, cg, cb = _hex(color_hex)
    verts = []; hh = h/2
    for i in range(segs):
        a0 = 2*math.pi*i/segs; a1 = 2*math.pi*(i+1)/segs
        x0,z0 = math.cos(a0)*radius, math.sin(a0)*radius
        x1,z1 = math.cos(a1)*radius, math.sin(a1)*radius
        nx0,nz0 = math.cos(a0), math.sin(a0)
        for v in ((x0,-hh,z0),(x1,-hh,z1),(x1,hh,z1),(x0,-hh,z0),(x1,hh,z1),(x0,hh,z0)):
            verts.extend(v); verts.extend((nx0,0,nz0)); verts.extend((cr,cg,cb))
        for v in ((0,hh,0),(x0,hh,z0),(x1,hh,z1)):
            verts.extend(v); verts.extend((0,1,0)); verts.extend((cr,cg,cb))
        for v in ((0,-hh,0),(x1,-hh,z1),(x0,-hh,z0)):
            verts.extend(v); verts.extend((0,-1,0)); verts.extend((cr,cg,cb))
    return verts

def _transform_verts(verts_list, mat):
    arr = np.array(verts_list, dtype=np.float32).reshape(-1, 9)
    if arr.shape[0] == 0: return []
    m4 = np.array([[mat[c][r] for c in range(4)] for r in range(4)], dtype=np.float32)
    pos = np.hstack([arr[:,:3], np.ones((arr.shape[0],1), dtype=np.float32)])
    pos_t = (m4 @ pos.T).T[:,:3]
    nrm_t = (m4[:3,:3] @ arr[:,3:6].T).T
    return np.hstack([pos_t, nrm_t, arr[:,6:9]]).flatten().tolist()

# ═══════════════════════════════════════════════════════════════
#  PARSERS
# ═══════════════════════════════════════════════════════════════

def detect_bonds(atoms):
    bonds = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            a, b = atoms[i], atoms[j]
            d = math.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2+(a['z']-b['z'])**2)
            ci = ELEMENTS.get(a['el'],{'cov':0.77})['cov']
            cj = ELEMENTS.get(b['el'],{'cov':0.77})['cov']
            if 0.4 < d < (ci+cj)*1.3: bonds.append((i, j))
    return bonds

def parse_xyz(text):
    """Parse XYZ format text → list of atom dicts."""
    lines = text.strip().split('\n'); atoms = []; start = 0
    if len(lines) > 2:
        try: int(lines[0].strip()); start = 2
        except: pass
    for line in lines[start:]:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] in ELEMENTS:
            try: atoms.append({'el':parts[0],'x':float(parts[1]),'y':float(parts[2]),'z':float(parts[3])})
            except: pass
    return atoms

def parse_orca_output(text):
    """Parse ORCA .out file → dict with atoms, orbital_energies, charges, energy, etc."""
    lines = text.split('\n')
    result = {'atoms':[],'orb_labels':[],'orb_energies':[],'orb_occ':[],
              'total_energy':None,'mulliken':[],'scf_ok':False,'basis':'',
              'method':'','point_group':'','dipole':None,'frequencies':[],'ir_intensities':[]}
    in_coords=in_orbs=in_mull=in_freq=False; orb_dashes=0
    for i, line in enumerate(lines):
        if '!' in line and '|' in line:
            m = line.split('!');
            if len(m)>1: result['method'] = m[1].strip()
        if 'Your calculation utilizes the basis:' in line:
            p = line.split(':');
            if len(p)>1: result['basis'] = p[1].strip()
        if 'SCF CONVERGED' in line: result['scf_ok'] = True
        if 'FINAL SINGLE POINT ENERGY' in line:
            for p in line.split():
                try: result['total_energy'] = float(p)
                except: pass
        # Coordinates
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line: in_coords=True; continue
        if in_coords:
            s = line.strip()
            if s=='' and result['atoms']: in_coords=False; continue
            if '---' in s: continue
            parts = s.split()
            if len(parts)>=4 and parts[0] in ELEMENTS:
                try: result['atoms'].append({'el':parts[0],'x':float(parts[1]),'y':float(parts[2]),'z':float(parts[3])})
                except: pass
        # Orbitals
        if 'ORBITAL ENERGIES' in line: in_orbs=True; orb_dashes=0; continue
        if in_orbs:
            if '---' in line: orb_dashes+=1; continue
            if orb_dashes>=2:
                parts = line.strip().split()
                if len(parts)>=4:
                    try:
                        occ=float(parts[1]); eV=float(parts[2])
                        result['orb_labels'].append(f"MO {parts[0]} ({'occ' if occ>0.5 else 'virt'})")
                        result['orb_energies'].append(eV); result['orb_occ'].append(occ)
                    except: pass
                if line.strip()=='': in_orbs=False
        # Mulliken
        if 'MULLIKEN ATOMIC CHARGES' in line: in_mull=True; continue
        if in_mull:
            if 'Sum of Mulliken' in line: in_mull=False; continue
            m = re.match(r'\s*(\d+)\s+(\w+)\s*:\s*([-\d.]+)', line)
            if m: result['mulliken'].append({'idx':int(m.group(1)),'el':m.group(2),'charge':float(m.group(3))})
        # Frequencies
        fm = re.match(r'\s*(\d+):\s+([-\d.]+)\s+cm\*\*-1', line)
        if fm: result['frequencies'].append(float(fm.group(2)))
        im = re.match(r'\s*(\d+):\s+([-\d.]+)\s+km/mol', line)
        if im: result['ir_intensities'].append(float(im.group(2)))
        # Point group, dipole
        pgm = re.search(r'Point Group:\s*(\S+)', line, re.I)
        if pgm: result['point_group'] = pgm.group(1)
        dm = re.search(r'Magnitude \(Debye\)\s*:\s*([\d.]+)', line)
        if dm: result['dipole'] = float(dm.group(1))
    return result

def parse_cube(text):
    """Parse Gaussian cube file → dict with atoms, grid, origin, axes, npts."""
    lines = text.strip().split('\n')
    if len(lines) < 7: return None
    parts = lines[2].split()
    natoms = abs(int(parts[0]))
    origin = np.array([float(parts[1]),float(parts[2]),float(parts[3])]) * BOHR
    axes = []; npts = []
    for i in range(3):
        p = lines[3+i].split()
        npts.append(abs(int(p[0])))
        axes.append(np.array([float(p[1]),float(p[2]),float(p[3])]) * BOHR)
    atoms = []
    for i in range(natoms):
        p = lines[6+i].split()
        z = int(float(p[0]))
        if z <= 0:
            continue  # skip dummy atoms / ghost centres (Z=0)
        el = Z_TO_EL.get(z, None)
        if el is None:
            # Last resort: generate symbol from Z
            el = f"Z{z}"
            ELEMENTS[el] = {'color':0x888888,'r':0.8,'cov':0.8,'mass':float(z)*2,'Z':z,'name':f'Element {z}'}
        atoms.append({'el':el,'x':float(p[2])*BOHR,'y':float(p[3])*BOHR,'z':float(p[4])*BOHR})
    data_start = 6 + natoms
    if int(lines[2].split()[0]) < 0: data_start += 1
    values = []
    for line in lines[data_start:]:
        for v in line.split():
            try: values.append(float(v))
            except: pass
    n1, n2, n3 = npts
    if len(values) < n1*n2*n3: return None
    grid = np.array(values[:n1*n2*n3]).reshape(n1, n2, n3)
    return {'atoms':atoms,'grid':grid,'origin':origin,'axes':axes,'npts':npts}

# ═══════════════════════════════════════════════════════════════
#  CUBE FILE ARITHMETIC & UTILITIES
# ═══════════════════════════════════════════════════════════════

def cube_add(cube_a, cube_b, scale_a=1.0, scale_b=1.0):
    """Return a new cube dict = scale_a * A + scale_b * B (grids must match)."""
    g = cube_a['grid'] * scale_a + cube_b['grid'] * scale_b
    return {**cube_a, 'grid': g}

def cube_subtract(cube_a, cube_b):
    """A - B."""
    return cube_add(cube_a, cube_b, 1.0, -1.0)

def cube_scale(cube, factor):
    """Multiply grid values by factor."""
    return {**cube, 'grid': cube['grid'] * factor}

def cube_write(cube_data, path):
    """Write a cube dict back to a Gaussian cube file."""
    g = cube_data['grid']; o = cube_data['origin']; ax = cube_data['axes']; n = cube_data['npts']
    atoms = cube_data['atoms']
    with open(path, 'w') as f:
        f.write("Cube file written by ChemLab\n\n")
        f.write(f" {len(atoms):5d} {o[0]/BOHR:12.6f} {o[1]/BOHR:12.6f} {o[2]/BOHR:12.6f}\n")
        for i in range(3):
            f.write(f" {n[i]:5d} {ax[i][0]/BOHR:12.6f} {ax[i][1]/BOHR:12.6f} {ax[i][2]/BOHR:12.6f}\n")
        for a in atoms:
            Z = ELEMENTS.get(a['el'], {'Z':0})['Z']
            f.write(f" {Z:5d} {float(Z):12.6f} {a['x']/BOHR:12.6f} {a['y']/BOHR:12.6f} {a['z']/BOHR:12.6f}\n")
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    f.write(f" {g[i,j,k]:13.5E}")
                    if (k + 1) % 6 == 0 or k == n[2] - 1:
                        f.write("\n")
    return path

def cube_center_inbox(cube_data):
    """Center atoms inside the cube box and shift the grid origin accordingly."""
    atoms = cube_data['atoms']
    if not atoms:
        return cube_data
    o = cube_data['origin']; ax = cube_data['axes']; n = cube_data['npts']
    # Box extent
    box_diag = np.array(ax[0]) * n[0] + np.array(ax[1]) * n[1] + np.array(ax[2]) * n[2]
    box_center = o + box_diag * 0.5
    # Atom centroid
    cx = np.mean([a['x'] for a in atoms])
    cy = np.mean([a['y'] for a in atoms])
    cz = np.mean([a['z'] for a in atoms])
    shift = box_center - np.array([cx, cy, cz])
    new_atoms = [{'el':a['el'], 'x':a['x']+shift[0], 'y':a['y']+shift[1], 'z':a['z']+shift[2]} for a in atoms]
    return {**cube_data, 'atoms': new_atoms}

def _colormap_rwb(val, vmin, vmax):
    """Red-white-blue colormap. val in [vmin, vmax] → (r,g,b) in [0,1]."""
    if vmax == vmin:
        return (1.0, 1.0, 1.0)
    t = (val - vmin) / (vmax - vmin)  # 0..1
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t * 2.0  # 0..1 from blue→white
        return (s, s, 1.0)
    else:
        s = (t - 0.5) * 2.0  # 0..1 from white→red
        return (1.0, 1.0 - s, 1.0 - s)

def cube_colormapped_isosurface(density_cube, property_cube, isovalue=0.05, vmin=None, vmax=None):
    """Generate isosurface mesh from density_cube, colored by property_cube values.
    Returns flat list of pos(3)+normal(3)+color(3) floats.
    This is the VMD 'Volume' coloring method — e.g. ESP mapped onto electron density."""
    g = density_cube['grid']; o = density_cube['origin']
    ax = [np.array(a) for a in density_cube['axes']]; n = density_cube['npts']
    pg = property_cube['grid']
    # Determine color range
    if vmin is None: vmin = float(pg.min())
    if vmax is None: vmax = float(pg.max())
    # Symmetric range for ESP
    vm = max(abs(vmin), abs(vmax))
    if vm > 0:
        vmin, vmax = -vm, vm

    corner_offsets = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    n1, n2, n3 = n
    verts = []
    for i in range(n1-1):
        for j in range(n2-1):
            for k in range(n3-1):
                vals = [g[i+di,j+dj,k+dk] for di,dj,dk in corner_offsets]
                signs = [1 if v >= isovalue else 0 for v in vals]
                ci = sum(s<<idx for idx,s in enumerate(signs))
                if ci == 0 or ci == 255: continue
                positions = [o+(i+di)*ax[0]+(j+dj)*ax[1]+(k+dk)*ax[2] for di,dj,dk in corner_offsets]
                pvals = [pg[i+di,j+dj,k+dk] for di,dj,dk in corner_offsets]
                edge_pts = []; edge_colors = []
                for c1,c2 in edges:
                    if signs[c1] != signs[c2]:
                        t = (isovalue-vals[c1])/(vals[c2]-vals[c1]+1e-12)
                        t = max(0.0,min(1.0,t))
                        pt = positions[c1]+t*(positions[c2]-positions[c1])
                        pv = pvals[c1]+t*(pvals[c2]-pvals[c1])
                        edge_pts.append(pt)
                        edge_colors.append(_colormap_rwb(pv, vmin, vmax))
                if len(edge_pts) < 3: continue
                centroid = np.mean(edge_pts, axis=0)
                v1 = edge_pts[1]-edge_pts[0]; v2 = edge_pts[2]-edge_pts[0]
                normal = np.cross(v1,v2)
                nl = np.linalg.norm(normal)
                normal = normal/nl if nl > 1e-10 else np.array([0,0,1])
                avg_col = (np.mean([c[0] for c in edge_colors]),
                           np.mean([c[1] for c in edge_colors]),
                           np.mean([c[2] for c in edge_colors]))
                for ei in range(len(edge_pts)):
                    for p_idx, p in enumerate([centroid, edge_pts[ei], edge_pts[(ei+1)%len(edge_pts)]]):
                        verts.extend(p.tolist()); verts.extend(normal.tolist())
                        if p_idx == 0:
                            verts.extend(avg_col)
                        elif p_idx == 1:
                            verts.extend(edge_colors[ei])
                        else:
                            verts.extend(edge_colors[(ei+1)%len(edge_colors)])
    return verts

def _make_volume_slice(cube_data, axis=1, pos=0.5):
    """Generate a slice image (numpy RGBA array) through a cube volume.
    axis: 0=X, 1=Y, 2=Z. pos: fractional position [0,1] along that axis.
    Returns (rgba_image, quad_verts) where quad_verts are the 3D corner positions."""
    g = cube_data['grid']; o = cube_data['origin']
    ax = [np.array(a) for a in cube_data['axes']]; n = cube_data['npts']
    idx = int(pos * (n[axis]-1))
    idx = max(0, min(n[axis]-1, idx))
    if axis == 0:
        sl = g[idx, :, :]; u_ax, v_ax = 1, 2; u_n, v_n = n[1], n[2]
        origin_sl = o + ax[0] * idx
    elif axis == 1:
        sl = g[:, idx, :]; u_ax, v_ax = 0, 2; u_n, v_n = n[0], n[2]
        origin_sl = o + ax[1] * idx
    else:
        sl = g[:, :, idx]; u_ax, v_ax = 0, 1; u_n, v_n = n[0], n[1]
        origin_sl = o + ax[2] * idx
    # Normalize to RGBA image
    vmin, vmax = float(sl.min()), float(sl.max())
    if vmax == vmin: vmax = vmin + 1e-10
    img = np.zeros((u_n, v_n, 4), dtype=np.uint8)
    for i in range(u_n):
        for j in range(v_n):
            r, gc, b = _colormap_rwb(sl[i, j], vmin, vmax)
            img[i, j] = [int(r*255), int(gc*255), int(b*255), 200]
    # Quad corners in 3D
    c00 = origin_sl
    c10 = origin_sl + ax[u_ax] * u_n
    c01 = origin_sl + ax[v_ax] * v_n
    c11 = origin_sl + ax[u_ax] * u_n + ax[v_ax] * v_n
    return img, [c00, c10, c01, c11]

# ═══════════════════════════════════════════════════════════════
#  MARCHING CUBES (simplified — isosurface from cube data)
# ═══════════════════════════════════════════════════════════════

def _marching_cubes(grid, origin, axes, npts, isovalue):
    """Extract triangle mesh where grid crosses isovalue."""
    n1,n2,n3 = npts; ax = [np.array(a) for a in axes]
    corner_offsets = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    verts = []
    for i in range(n1-1):
        for j in range(n2-1):
            for k in range(n3-1):
                vals = [grid[i+di,j+dj,k+dk] for di,dj,dk in corner_offsets]
                signs = [1 if v >= isovalue else 0 for v in vals]
                ci = sum(s<<idx for idx,s in enumerate(signs))
                if ci == 0 or ci == 255: continue
                positions = [origin+(i+di)*ax[0]+(j+dj)*ax[1]+(k+dk)*ax[2] for di,dj,dk in corner_offsets]
                edge_pts = []
                for c1,c2 in edges:
                    if signs[c1] != signs[c2]:
                        t = (isovalue-vals[c1])/(vals[c2]-vals[c1]+1e-12)
                        t = max(0.0,min(1.0,t))
                        edge_pts.append(positions[c1]+t*(positions[c2]-positions[c1]))
                if len(edge_pts) < 3: continue
                centroid = np.mean(edge_pts, axis=0)
                v1 = edge_pts[1]-edge_pts[0]; v2 = edge_pts[2]-edge_pts[0]
                normal = np.cross(v1,v2)
                nl = np.linalg.norm(normal)
                normal = normal/nl if nl > 1e-10 else np.array([0,0,1])
                for ei in range(len(edge_pts)):
                    for p in [centroid, edge_pts[ei], edge_pts[(ei+1)%len(edge_pts)]]:
                        verts.extend(p.tolist()); verts.extend(normal.tolist())
    return verts

def cube_isosurface(cube_data, isovalue=0.02):
    """Generate colored triangle mesh from cube data. Returns pos(3)+normal(3)+color(3) floats."""
    g,o,ax,n = cube_data['grid'],cube_data['origin'],cube_data['axes'],cube_data['npts']
    col_pos, col_neg = _hex(0x3070CC), _hex(0xCC3030)
    pos_raw = _marching_cubes(g, o, ax, n, isovalue)
    neg_raw = _marching_cubes(g, o, ax, n, -isovalue)
    result = []
    for i in range(0, len(pos_raw), 6):
        result.extend(pos_raw[i:i+6]); result.extend(col_pos)
    for i in range(0, len(neg_raw), 6):
        result.extend(neg_raw[i:i+6]); result.extend(col_neg)
    return result

# ═══════════════════════════════════════════════════════════════
#  ORBITAL LOBE MESH (approximate — for presets without cube)
# ═══════════════════════════════════════════════════════════════

def _spherical_harmonic(l, m, theta, phi):
    ct, st = math.cos(theta), math.sin(theta)
    if l==0: return 0.282
    if l==1:
        if m==0: return 0.489*ct
        if m==1: return 0.489*st*math.cos(phi)
        if m==-1: return 0.489*st*math.sin(phi)
    if l==2:
        if m==0: return 0.315*(3*ct*ct-1)
        if m==1: return 1.092*st*ct*math.cos(phi)
        if m==-1: return 1.092*st*ct*math.sin(phi)
        if m==2: return 0.546*st*st*math.cos(2*phi)
        if m==-2: return 0.546*st*st*math.sin(2*phi)
    return 0

def _make_lobe_mesh(l, m, scale, center, col_pos_hex, col_neg_hex, segs_t=20, segs_p=28):
    """Triangle mesh of |Y_lm| angular surface centered at center."""
    cx,cy,cz = center
    rp,gp,bp = _hex(col_pos_hex); rn,gn,bn = _hex(col_neg_hex)
    thetas = [(j+0.5)/segs_t*math.pi for j in range(segs_t)]
    phis = [i/segs_p*2*math.pi for i in range(segs_p+1)]
    grid = []
    for j in range(segs_t):
        row = []
        for i in range(segs_p+1):
            Y = _spherical_harmonic(l, m, thetas[j], phis[i])
            r = max(abs(Y)*scale, 0.01)
            st,ct = math.sin(thetas[j]),math.cos(thetas[j])
            sp,cp = math.sin(phis[i]),math.cos(phis[i])
            row.append((cx+r*st*cp, cy+r*ct, cz+r*st*sp, 1 if Y>=0 else -1))
        grid.append(row)
    verts = []
    for j in range(segs_t-1):
        for i in range(segs_p):
            p00,p01,p10,p11 = grid[j][i][:3],grid[j][i+1][:3],grid[j+1][i][:3],grid[j+1][i+1][:3]
            sign = grid[j][i][3]
            v1 = (p10[0]-p00[0],p10[1]-p00[1],p10[2]-p00[2])
            v2 = (p11[0]-p00[0],p11[1]-p00[1],p11[2]-p00[2])
            nx=v1[1]*v2[2]-v1[2]*v2[1]; ny=v1[2]*v2[0]-v1[0]*v2[2]; nz=v1[0]*v2[1]-v1[1]*v2[0]
            nl=math.sqrt(nx*nx+ny*ny+nz*nz)+1e-9; nx/=nl; ny/=nl; nz/=nl
            col = (rp,gp,bp) if sign>0 else (rn,gn,bn)
            for tri_p in (p00,p10,p11, p00,p11,p01):
                verts.extend(tri_p); verts.extend((nx,ny,nz)); verts.extend(col)
    return verts

# ═══════════════════════════════════════════════════════════════
#  GL VIEWER WIDGET
# ═══════════════════════════════════════════════════════════════

class MolViewer(QWidget):
    """OpenGL molecular viewer — offscreen FBO → QImage → QPainter.
    Supports molecule rendering, orbital isosurfaces, custom overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # State
        self.atoms=[]; self.bonds=[]; self.mol_name=""
        self.rot_x=0.3; self.rot_y=0.0; self.auto_rot=0.0; self.cam_dist=8.0
        self._dragging=False; self._lmx=0; self._lmy=0
        self.selected_atom=-1; self.render_style='ballstick'; self.show_bonds=True
        self._center=(0,0,0)
        self.orbital_opacity=0.55
        # GL
        self._gl_ready=False; self.ctx=None; self.fbo=None
        self._fbo_w=0; self._fbo_h=0; self._frame=None
        self._mol_vao=None; self._mol_n=0
        self._orb_vao=None; self._orb_n=0
        # Multi-isosurface layers: list of (vao, n_tris, opacity, label)
        self._iso_layers = []
        # Volume slice state
        self._slice_tex = None
        self._slice_vao = None
        self._slice_n = 0
        self._slice_alpha = 0.7
        # Animation: list of cube dicts for frame cycling
        self._anim_cubes = []
        self._anim_frame = 0
        self._anim_playing = False
        self._anim_timer = None
        self._anim_isovalue = 0.02
        # Overlays: dict of name → fn(painter, w, h)
        self._overlays = OrderedDict()
        # Measurements: list of (type, indices, value)
        self._measurements = []
        # Timer
        self.timer=QTimer(self); self.timer.timeout.connect(self._tick); self.timer.setInterval(16)

    def _ensure_gl(self):
        if self._gl_ready: return
        self._gl_ready=True
        self.ctx=moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func=(moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog_lit=self.ctx.program(vertex_shader=_VERT_LIT, fragment_shader=_FRAG_LIT)
        self.prog_orb=self.ctx.program(vertex_shader=_VERT_LIT, fragment_shader=_FRAG_ORBITAL)
        self._resize_fbo(max(self.width(),320),max(self.height(),200))
        self.timer.start()

    def _resize_fbo(self,w,h):
        if w==self._fbo_w and h==self._fbo_h and self.fbo: return
        if self.fbo: self.fbo.release()
        self._fbo_w=w; self._fbo_h=h
        self.fbo=self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w,h),4)],
            depth_attachment=self.ctx.depth_renderbuffer((w,h)))

    def set_molecule(self, atoms, name=""):
        self.atoms=atoms; self.bonds=detect_bonds(atoms) if atoms else []
        self.mol_name=name; self.selected_atom=-1
        if atoms:
            cx=sum(a['x'] for a in atoms)/len(atoms)
            cy=sum(a['y'] for a in atoms)/len(atoms)
            cz=sum(a['z'] for a in atoms)/len(atoms)
            self._center=(cx,cy,cz)
            md=max(math.sqrt((a['x']-cx)**2+(a['y']-cy)**2+(a['z']-cz)**2) for a in atoms)
            self.cam_dist=max(md*3.2,5.0)
        self.auto_rot=0.0
        self._rebuild_mol()

    def set_orbital_mesh(self, verts):
        """Set orbital as raw triangle verts: pos(3)+normal(3)+color(3)."""
        if not self._gl_ready or not verts:
            self._orb_vao=None; self._orb_n=0; return
        data=np.array(verts,dtype='f4').tobytes()
        if self._orb_vao:
            try: self._orb_vao.release()
            except: pass
        vbo=self.ctx.buffer(data)
        self._orb_vao=self.ctx.vertex_array(self.prog_orb,[(vbo,'3f 3f 3f','in_position','in_normal','in_color')])
        self._orb_n=len(verts)//9

    def clear_orbital(self):
        self._orb_vao=None; self._orb_n=0

    # ── Multi-layer isosurface support ──────────────────────────

    def add_iso_layer(self, verts, opacity=0.55, label=""):
        """Add an additional transparent isosurface layer.
        verts: pos(3)+normal(3)+color(3) flat list."""
        if not self._gl_ready or not verts: return
        data = np.array(verts, dtype='f4').tobytes()
        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(self.prog_orb, [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        self._iso_layers.append((vao, len(verts)//9, opacity, label))

    def clear_iso_layers(self):
        """Remove all extra isosurface layers."""
        for vao, _, _, _ in self._iso_layers:
            try: vao.release()
            except: pass
        self._iso_layers.clear()

    def set_colormapped_surface(self, density_cube, property_cube, isovalue=0.05, vmin=None, vmax=None):
        """Render density isosurface color-coded by a property (e.g. ESP).
        This is the VMD 'Volume' coloring method."""
        verts = cube_colormapped_isosurface(density_cube, property_cube, isovalue, vmin, vmax)
        cx, cy, cz = self._center
        if verts and (cx or cy or cz):
            arr = np.array(verts, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            verts = arr.flatten().tolist()
        self.set_orbital_mesh(verts)

    # ── Volume slice ───────────────────────────────────────────

    def set_volume_slice(self, cube_data, axis=1, pos=0.5):
        """Display a 2D cross-section through a volumetric dataset.
        axis: 0=X, 1=Y, 2=Z. pos: fractional position [0,1]."""
        img, corners = _make_volume_slice(cube_data, axis, pos)
        cx, cy, cz = self._center
        # Create 2D texture from the slice image
        h, w = img.shape[:2]
        # Store as a QPainter overlay for simplicity (no extra GL pipeline needed)
        self._slice_cube = cube_data
        self._slice_axis = axis
        self._slice_pos = pos
        def draw_slice(painter, pw, ph):
            qimg = QImage(img.data, w, h, w*4, QImage.Format_RGBA8888)
            # Draw as thumbnail in bottom-left
            sw, sh = min(180, pw//3), min(180, ph//3)
            painter.setOpacity(self._slice_alpha)
            painter.drawImage(12, ph-sh-80, qimg.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            painter.setOpacity(1.0)
            painter.setFont(QFont("Consolas", 8)); painter.setPen(QColor(60,70,90))
            ax_name = ['X','Y','Z'][axis]
            painter.drawText(12, ph-sh-90, f"Volume slice: {ax_name}={pos:.2f}")
        self.add_overlay('volume_slice', draw_slice)

    def clear_volume_slice(self):
        self.remove_overlay('volume_slice')

    # ── Animation (frame-cycling isosurfaces) ──────────────────

    def set_animation_cubes(self, cube_list, isovalue=0.02):
        """Load a sequence of cube dicts for animated isosurface playback."""
        self._anim_cubes = cube_list
        self._anim_isovalue = isovalue
        self._anim_frame = 0
        if cube_list:
            self._show_anim_frame(0)

    def _show_anim_frame(self, idx):
        """Render frame idx of the animation."""
        if not self._anim_cubes or idx >= len(self._anim_cubes): return
        self._anim_frame = idx
        cube = self._anim_cubes[idx]
        verts = cube_isosurface(cube, self._anim_isovalue)
        cx, cy, cz = self._center
        if verts and (cx or cy or cz):
            arr = np.array(verts, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            verts = arr.flatten().tolist()
        self.set_orbital_mesh(verts)

    def play_animation(self, interval_ms=200):
        """Start animated playback of loaded cube sequence."""
        if not self._anim_cubes: return
        self._anim_playing = True
        if self._anim_timer is None:
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._anim_step)
        self._anim_timer.setInterval(interval_ms)
        self._anim_timer.start()

    def stop_animation(self):
        """Pause animation."""
        self._anim_playing = False
        if self._anim_timer: self._anim_timer.stop()

    def _anim_step(self):
        if not self._anim_cubes: self.stop_animation(); return
        self._anim_frame = (self._anim_frame + 1) % len(self._anim_cubes)
        self._show_anim_frame(self._anim_frame)

    def set_cube(self, cube_data, isovalue=0.02):
        """Render isosurface from cube file data."""
        verts = cube_isosurface(cube_data, isovalue)
        cx,cy,cz = self._center
        if verts and (cx or cy or cz):
            arr = np.array(verts, dtype=np.float32).reshape(-1,9)
            arr[:,0]-=cx; arr[:,1]-=cy; arr[:,2]-=cz
            verts = arr.flatten().tolist()
        self.set_orbital_mesh(verts)

    def add_overlay(self, name, fn):
        """Register a QPainter overlay: fn(painter, width, height)."""
        self._overlays[name] = fn

    def remove_overlay(self, name):
        self._overlays.pop(name, None)

    def screenshot(self, path):
        """Save current frame as PNG."""
        if self._frame: self._frame.save(path)

    def _rebuild_mol(self):
        if not self._gl_ready or not self.atoms: return
        cx,cy,cz=self._center; all_v=[]
        scale=2.2 if self.render_style=='spacefill' else 1.0
        segs=24 if len(self.atoms)>30 else 32
        for idx,a in enumerate(self.atoms):
            el=ELEMENTS.get(a['el'],{'color':0x888888,'r':0.5})
            r=el['r']*0.4*scale if self.render_style!='wireframe' else 0.12
            col=el['color']
            if self.selected_atom==idx:
                rr,gg,bb=_hex(col)
                col=(min(int((rr+0.3)*255),255)<<16)|(min(int((gg+0.3)*255),255)<<8)|min(int((bb+0.3)*255),255)
            sp=_make_sphere(r,segs,segs//2,col)
            t=glm.translate(glm.mat4(1),glm.vec3(a['x']-cx,a['y']-cy,a['z']-cz))
            all_v.extend(_transform_verts(sp,t))
        if self.show_bonds and self.render_style!='spacefill':
            br=0.025 if self.render_style=='wireframe' else 0.06
            for i,j in self.bonds:
                a1,a2=self.atoms[i],self.atoms[j]
                s=glm.vec3(a1['x']-cx,a1['y']-cy,a1['z']-cz)
                e=glm.vec3(a2['x']-cx,a2['y']-cy,a2['z']-cz)
                mid=(s+e)*0.5; diff=e-s; length=glm.length(diff)
                if length<0.01: continue
                d=glm.normalize(diff); cyl=_make_cylinder(br,length,8,0x778899)
                up=glm.vec3(0,1,0)
                if abs(glm.dot(up,d))>0.999:
                    rm=glm.mat4(1) if d.y>0 else glm.rotate(glm.mat4(1),math.pi,glm.vec3(1,0,0))
                else:
                    ax=glm.normalize(glm.cross(up,d))
                    ang=math.acos(max(-1,min(1,glm.dot(up,d))))
                    rm=glm.rotate(glm.mat4(1),ang,ax)
                all_v.extend(_transform_verts(cyl,glm.translate(glm.mat4(1),mid)*rm))
        if not all_v: self._mol_vao=None; self._mol_n=0; return
        data=np.array(all_v,dtype='f4').tobytes()
        if self._mol_vao:
            try: self._mol_vao.release()
            except: pass
        vbo=self.ctx.buffer(data)
        self._mol_vao=self.ctx.vertex_array(self.prog_lit,[(vbo,'3f 3f 3f','in_position','in_normal','in_color')])
        self._mol_n=len(all_v)//9

    def _tick(self):
        if not self._dragging: self.auto_rot+=0.006
        self._render(); self.update()

    def _render(self):
        if not self._gl_ready: return
        w,h=max(self.width(),320),max(self.height(),200)
        self._resize_fbo(w,h); self.fbo.use(); self.ctx.viewport=(0,0,w,h)
        self.ctx.clear(0,0,0,0)
        proj=glm.perspective(glm.radians(45),w/h,0.1,200.0)
        ry=self.rot_y+self.auto_rot
        eye=glm.vec3(math.sin(ry)*math.cos(self.rot_x)*self.cam_dist,
                      math.sin(self.rot_x)*self.cam_dist,
                      math.cos(ry)*math.cos(self.rot_x)*self.cam_dist)
        view=glm.lookAt(eye,glm.vec3(0),glm.vec3(0,1,0))
        vp=proj*view; identity=glm.mat4(1)
        if self._mol_vao and self._mol_n>0:
            self.prog_lit['mvp'].write(vp); self.prog_lit['model'].write(identity)
            self.prog_lit['light_dir'].write(glm.normalize(glm.vec3(0.5,0.8,0.6)))
            self.prog_lit['ambient'].write(glm.vec3(0.45,0.44,0.46))
            self.prog_lit['view_pos'].write(eye)
            self._mol_vao.render(moderngl.TRIANGLES)
        if self._orb_vao and self._orb_n>0:
            self.ctx.disable(moderngl.CULL_FACE); self.ctx.disable(moderngl.DEPTH_TEST)
            self.prog_orb['mvp'].write(vp); self.prog_orb['model'].write(identity)
            self.prog_orb['light_dir'].write(glm.normalize(glm.vec3(0.5,0.8,0.6)))
            self.prog_orb['ambient'].write(glm.vec3(0.5,0.48,0.52))
            self.prog_orb['view_pos'].write(eye); self.prog_orb['alpha'].value=self.orbital_opacity
            self._orb_vao.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.enable(moderngl.CULL_FACE)
        # Render extra isosurface layers
        for layer_vao, layer_n, layer_alpha, _ in self._iso_layers:
            if layer_vao and layer_n > 0:
                self.ctx.disable(moderngl.CULL_FACE); self.ctx.disable(moderngl.DEPTH_TEST)
                self.prog_orb['mvp'].write(vp); self.prog_orb['model'].write(identity)
                self.prog_orb['light_dir'].write(glm.normalize(glm.vec3(0.5,0.8,0.6)))
                self.prog_orb['ambient'].write(glm.vec3(0.5,0.48,0.52))
                self.prog_orb['view_pos'].write(eye); self.prog_orb['alpha'].value=layer_alpha
                layer_vao.render(moderngl.TRIANGLES)
                self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.enable(moderngl.CULL_FACE)
        raw=self.fbo.color_attachments[0].read()
        _img=QImage(raw,w,h,w*4,QImage.Format_RGBA8888)
        # Qt6 compatibility: mirrored() returns a new QImage (deprecated but works);
        # mirror() is in-place and returns None in some PySide6 versions.
        # We always use mirrored() which reliably returns the flipped copy.
        self._frame=_img.mirrored(False,True)

    def paintEvent(self, event):
        self._ensure_gl()
        if self.atoms and self._mol_n==0: self._rebuild_mol()
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height()
        if self._frame and not self._frame.isNull(): p.drawImage(0,0,self._frame)
        # HUD
        if self.atoms:
            p.setPen(Qt.NoPen); p.setBrush(QColor(255,255,255,195))
            p.drawRoundedRect(12,12,200,64,8,8)
            p.setFont(QFont("Consolas",9,QFont.Bold)); p.setPen(QColor(50,55,70))
            p.drawText(20,22,180,16,Qt.AlignVCenter,self.mol_name or "Molecule")
            p.setFont(QFont("Consolas",9)); p.setPen(QColor(80,90,110))
            mass=sum(ELEMENTS.get(a['el'],{'mass':0})['mass'] for a in self.atoms)
            counts={}
            for a in self.atoms: counts[a['el']]=counts.get(a['el'],0)+1
            keys=sorted(counts.keys(),key=lambda x:('A' if x=='C' else 'B' if x=='H' else x))
            formula=''.join(f"{k}{counts[k] if counts[k]>1 else ''}" for k in keys)
            p.drawText(20,38,180,14,Qt.AlignVCenter,f"{len(self.atoms)} atoms · {formula}")
            p.drawText(20,52,180,14,Qt.AlignVCenter,f"{mass:.2f} g/mol")
        # Selected atom
        if 0<=self.selected_atom<len(self.atoms):
            a=self.atoms[self.selected_atom]; el=ELEMENTS.get(a['el'],{'name':a['el'],'Z':'?','mass':0})
            p.setPen(Qt.NoPen); p.setBrush(QColor(255,255,255,210))
            p.drawRoundedRect(w-180,12,168,75,8,8)
            p.setFont(QFont("Consolas",10,QFont.Bold)); p.setPen(QColor(40,45,60))
            p.drawText(w-170,18,155,16,Qt.AlignVCenter,f"{el['name']} #{self.selected_atom}")
            p.setFont(QFont("Consolas",9)); p.setPen(QColor(90,100,120))
            p.drawText(w-170,36,155,13,Qt.AlignVCenter,f"{a['el']} Z={el['Z']}  {el['mass']:.3f} u")
            p.drawText(w-170,50,155,13,Qt.AlignVCenter,f"({a['x']:+.3f}, {a['y']:+.3f}, {a['z']:+.3f})")
        # Measurements
        p.setFont(QFont("Consolas",9)); p.setPen(QColor(40,100,160))
        my=h-60
        for mtype,idxs,val in self._measurements:
            if mtype=='dist': p.drawText(12,my,200,14,Qt.AlignVCenter,f"d({idxs[0]}-{idxs[1]}) = {val:.4f} \u00c5")
            elif mtype=='angle': p.drawText(12,my,250,14,Qt.AlignVCenter,f"\u2220({idxs[0]}-{idxs[1]}-{idxs[2]}) = {val:.1f}\u00b0")
            my-=16
        # Custom overlays
        for name, fn in self._overlays.items():
            try: fn(p, w, h)
            except: pass
        p.end()

    @staticmethod
    def _epos(e):
        """Extract mouse position compatible with both old and new PySide6."""
        try:
            p = e.position()  # Qt6.1+
            return p.x(), p.y()
        except AttributeError:
            return e.x(), e.y()

    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton:
            self._dragging=True
            self._lmx, self._lmy = self._epos(e)
        e.accept()
    def mouseReleaseEvent(self,e): self._dragging=False; e.accept()
    def mouseMoveEvent(self,e):
        if self._dragging:
            mx, my = self._epos(e)
            self.rot_y+=(mx-self._lmx)*0.008
            self.rot_x=max(-1.4,min(1.4,self.rot_x+(my-self._lmy)*0.008))
            self._lmx=mx; self._lmy=my
        e.accept()
    def wheelEvent(self,e):
        self.cam_dist=max(2,min(50,self.cam_dist-e.angleDelta().y()*0.005)); e.accept()

# ═══════════════════════════════════════════════════════════════
#  CHEMLAB — MAIN API SINGLETON
# ═══════════════════════════════════════════════════════════════

class _Signals(QObject):
    """Thread-safe signals for async operations."""
    orca_done = Signal(str)  # path to output file
    status = Signal(str)

class ChemLab:
    """
    Main chemistry API. Registered as `chem` in the namespace.

    QUICK REFERENCE (for LLM context):
        chem.load(name)                    — load preset: water/methane/benzene/co2/ammonia/ethanol/formaldehyde/hcn/acetylene
        chem.load_xyz(text)                — load from XYZ text
        chem.load_file(path)               — load .xyz, .cube, or ORCA .out
        chem.load_cube(path_or_data)       — load cube file (path string or parsed dict)
        chem.style(s)                      — ballstick/spacefill/wireframe
        chem.select(i)                     — highlight atom i
        chem.measure(i, j)                 — show distance i-j
        chem.angle(i, j, k)               — show angle i-j-k
        chem.dihedral(i, j, k, l)         — show dihedral
        chem.clear_measurements()          — clear all measurements
        chem.show_mo(idx)                  — show LCAO approx MO (for presets)
        chem.clear_mo()                    — hide orbital
        chem.overlay(name, fn)             — add QPainter overlay fn(painter, w, h)
        chem.remove_overlay(name)          — remove overlay
        chem.label_charges(charges)        — show charge labels on atoms
        chem.label_atoms()                 — show element+index labels
        chem.export_xyz(path)              — save current geometry as XYZ
        chem.export_orca_input(path, method="B3LYP def2-SVP", charge=0, mult=1, extra="")
        chem.run_orca(inp_path, orca_cmd="orca")  — run ORCA (async, result in chem.orca_result)
        chem.parse_orca(out_path)          — parse ORCA output, updates chem.info
        ## Volumetric (sections 7.1-7.7):
        chem.calculate_full()              — full calc: density + ESP + HOMO/LUMO cubes
        chem.show_esp(density_iso)         — ESP color-mapped onto density surface
        chem.load_esp(dens_path, esp_path) — load external density + ESP cube files
        chem.show_volume_slice(axis, pos)  — 2D cross-section through volume data
        chem.clear_volume_slice()          — remove volume slice
        chem.show_localized_orbitals()     — Boys-localized MOs (bonds + lone pairs)
        chem.show_elf(isovalue=0.85)       — electron localization function
        chem.load_elf(path)                — load external ELF cube
        chem.density_difference(a, b, iso) — display A - B density difference
        chem.cube_math(op, a, b, scale)    — general cube arithmetic (add/subtract/scale/write)
        chem.save_cube(key_or_dict, path)  — save cube dataset to file
        chem.load_bulk(dens, esp)          — load bulk system with centering
        chem.show_bulk_esp()               — multi-layer ESP isosurfaces for bulk
        chem.load_animation(paths)         — load cube sequence for animation
        chem.play(interval_ms)             — start animated playback
        chem.stop()                        — stop animation
        chem.frame(n)                      — jump to frame n
        chem.show_spin_density(iso)        — alpha-beta spin density (open-shell)
        chem.load_spin_density(a, b)       — from external alpha/beta cubes
        chem.add_isosurface(cube, iso, color, opacity)  — add extra transparent layer
        chem.clear_layers()                — remove all extra layers
        chem.atoms                         — current atom list
        chem.bonds                         — current bond list
        chem.info                          — dict of parsed properties
        chem.viewer                        — the MolViewer widget
        chem.log(msg)                      — append to log panel
    """

    def __init__(self, viewer, log_widget=None):
        self.viewer = viewer
        self._log = log_widget
        self.atoms = []
        self.bonds = []
        self.info = {}  # parsed ORCA results, charges, energies, etc.
        self.orca_result = None  # last ORCA parse result
        self._signals = _Signals()
        self._mo_defs = []  # LCAO MO definitions for current molecule
        self._cube_data = None

    def log(self, msg):
        if self._log: self._log.append(f"[chem] {msg}")
        print(f"[chem] {msg}")

    # ── Loading ────────────────────────────────────────────────

    def load(self, name):
        """Load a preset molecule by name."""
        key = name.lower().replace(' ','').replace('-','')
        # Try exact match first, then fuzzy
        for k, atoms in PRESETS.items():
            if key in k or k in key:
                self.atoms = [dict(a) for a in atoms]
                self.bonds = detect_bonds(self.atoms)
                self.viewer.set_molecule(self.atoms, name)
                self.info = {'name': name, 'source': 'preset'}
                self.log(f"Loaded preset: {name} ({len(self.atoms)} atoms)")
                return self
        self.log(f"Unknown preset: {name}. Available: {', '.join(PRESETS.keys())}")
        return self

    def load_xyz(self, text):
        """Load molecule from XYZ format text."""
        self.atoms = parse_xyz(text)
        if not self.atoms:
            self.log("No atoms found in XYZ text"); return self
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, "XYZ input")
        self.info = {'name': 'XYZ input', 'source': 'xyz'}
        self.log(f"Loaded {len(self.atoms)} atoms from XYZ")
        return self

    def load_file(self, path):
        """Load from file — auto-detects format by extension."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.log(f"File not found: {path}"); return self
        with open(path, 'r') as f: text = f.read()
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.cube', '.cub'):
            return self.load_cube(path)
        elif ext == '.xyz':
            self.atoms = parse_xyz(text)
            self.bonds = detect_bonds(self.atoms)
            self.viewer.set_molecule(self.atoms, os.path.basename(path))
            self.info = {'name': os.path.basename(path), 'source': 'xyz'}
            self.log(f"Loaded {os.path.basename(path)}: {len(self.atoms)} atoms")
        elif ext == '.out':
            return self.parse_orca(path)
        elif ext == '.inp':
            # Try to extract geometry from ORCA input
            self.atoms = parse_xyz(text)
            self.bonds = detect_bonds(self.atoms)
            self.viewer.set_molecule(self.atoms, os.path.basename(path))
        else:
            # Try XYZ
            self.atoms = parse_xyz(text)
            if self.atoms:
                self.bonds = detect_bonds(self.atoms)
                self.viewer.set_molecule(self.atoms, os.path.basename(path))
        return self

    def load_cube(self, path_or_data, isovalue=0.02):
        """Load cube file — from file path (str) or pre-parsed dict."""
        if isinstance(path_or_data, str):
            path = os.path.expanduser(path_or_data)
            with open(path, 'r') as f: text = f.read()
            cube = parse_cube(text)
            name = os.path.basename(path)
        elif isinstance(path_or_data, dict):
            cube = path_or_data
            name = "cube data"
        else:
            self.log("load_cube: pass a file path or parsed dict"); return self
        if cube is None:
            self.log("Failed to parse cube file"); return self
        self._cube_data = cube
        self.atoms = cube['atoms']
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, name)
        self.viewer.set_cube(cube, isovalue)
        n = cube['npts']
        self.info = {'name': name, 'source': 'cube', 'grid': f"{n[0]}x{n[1]}x{n[2]}"}
        self.log(f"Loaded cube: {name} ({n[0]}×{n[1]}×{n[2]} grid)")
        return self

    def load_smiles(self, smiles):
        """Load from SMILES string (requires obabel in PATH)."""
        try:
            result = subprocess.run(['obabel', '-:'+smiles, '-oxyz', '--gen3d'],
                                     capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return self.load_xyz(result.stdout)
            else:
                self.log(f"obabel failed: {result.stderr[:100]}")
        except FileNotFoundError:
            self.log("obabel not found. Install Open Babel: apt install openbabel")
        except Exception as ex:
            self.log(f"SMILES error: {ex}")
        return self

    # ── Rendering ──────────────────────────────────────────────

    def style(self, s):
        """Set render style: 'ballstick', 'spacefill', or 'wireframe'."""
        self.viewer.render_style = s
        self.viewer._rebuild_mol()
        return self

    def select(self, i):
        """Highlight atom by index."""
        self.viewer.selected_atom = i
        self.viewer._rebuild_mol()
        return self

    # ── Measurements ───────────────────────────────────────────

    def measure(self, i, j):
        """Measure and display distance between atoms i and j (Angstrom)."""
        if i >= len(self.atoms) or j >= len(self.atoms): return 0
        a, b = self.atoms[i], self.atoms[j]
        d = math.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2+(a['z']-b['z'])**2)
        self.viewer._measurements.append(('dist', (i, j), d))
        self.log(f"Distance {i}-{j}: {d:.4f} \u00c5")
        return d

    def angle(self, i, j, k):
        """Measure angle i-j-k in degrees."""
        if max(i,j,k) >= len(self.atoms): return 0
        a,b,c = self.atoms[i],self.atoms[j],self.atoms[k]
        v1 = np.array([a['x']-b['x'],a['y']-b['y'],a['z']-b['z']])
        v2 = np.array([c['x']-b['x'],c['y']-b['y'],c['z']-b['z']])
        cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-10)
        ang = math.degrees(math.acos(max(-1,min(1,cos_a))))
        self.viewer._measurements.append(('angle', (i, j, k), ang))
        self.log(f"Angle {i}-{j}-{k}: {ang:.1f}\u00b0")
        return ang

    def dihedral(self, i, j, k, l):
        """Measure dihedral i-j-k-l in degrees."""
        if max(i,j,k,l) >= len(self.atoms): return 0
        p = [np.array([self.atoms[x]['x'],self.atoms[x]['y'],self.atoms[x]['z']]) for x in (i,j,k,l)]
        b1=p[1]-p[0]; b2=p[2]-p[1]; b3=p[3]-p[2]
        n1=np.cross(b1,b2); n2=np.cross(b2,b3)
        m1=np.cross(n1,b2/np.linalg.norm(b2))
        x=np.dot(n1,n2); y=np.dot(m1,n2)
        d = math.degrees(math.atan2(y,x))
        self.log(f"Dihedral {i}-{j}-{k}-{l}: {d:.1f}\u00b0")
        return d

    def clear_measurements(self):
        self.viewer._measurements.clear(); return self

    # ── Orbitals ───────────────────────────────────────────────

    def show_mo(self, idx, lobes=None):
        """Show MO by index from LCAO definitions, or custom lobes list.
        lobes: [(atom_idx, l, m, scale, sign), ...] — each lobe is an angular shape on an atom."""
        if lobes:
            verts = []
            cx,cy,cz = self.viewer._center
            for ai,l,m,scale,sign in lobes:
                if ai >= len(self.atoms): continue
                a = self.atoms[ai]
                ac = (a['x']-cx, a['y']-cy, a['z']-cz)
                cp,cn = (0x3070CC,0xCC3030) if sign>=0 else (0xCC3030,0x3070CC)
                verts.extend(_make_lobe_mesh(l, m, scale, ac, cp, cn))
            self.viewer.set_orbital_mesh(verts)
        self.log(f"Showing MO {idx}")
        return self

    def clear_mo(self):
        self.viewer.clear_orbital(); return self

    def set_isovalue(self, val):
        """Update isovalue for current cube data."""
        if self._cube_data:
            self.viewer.set_cube(self._cube_data, val)
        return self

    # ── Labels & Overlays ──────────────────────────────────────

    def label_charges(self, charges=None):
        """Show Mulliken charges as floating labels. Uses self.info['mulliken'] if charges=None."""
        if charges is None:
            charges = {m['idx']: m['charge'] for m in self.info.get('mulliken', [])}
        if not charges: self.log("No charges available"); return self
        def draw_charges(painter, w, h):
            painter.setFont(QFont("Consolas", 8))
            for idx, charge in charges.items():
                if idx < len(self.atoms):
                    col = QColor(200,50,50) if charge > 0 else QColor(50,50,200)
                    painter.setPen(col)
                    sign = '+' if charge > 0 else ''
                    painter.drawText(w//2, 80 + idx*14, f"{self.atoms[idx]['el']}{idx}: {sign}{charge:.3f}")
        self.viewer.add_overlay('charges', draw_charges)
        return self

    def label_atoms(self):
        """Show element+index labels."""
        def draw_labels(painter, w, h):
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor(60,60,80))
            for i, a in enumerate(self.atoms):
                painter.drawText(w//2, 80 + i*13, f"{i}: {a['el']} ({a['x']:.2f},{a['y']:.2f},{a['z']:.2f})")
        self.viewer.add_overlay('labels', draw_labels)
        return self

    def overlay(self, name, fn):
        """Add custom overlay: fn(painter, width, height)."""
        self.viewer.add_overlay(name, fn); return self

    def remove_overlay(self, name):
        self.viewer.remove_overlay(name); return self

    # ── Export ─────────────────────────────────────────────────

    def export_xyz(self, path="~/molecule.xyz"):
        """Save current geometry as XYZ file."""
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            f.write(f"{len(self.atoms)}\nExported from ChemLab\n")
            for a in self.atoms:
                f.write(f"{a['el']}  {a['x']:.6f}  {a['y']:.6f}  {a['z']:.6f}\n")
        self.log(f"Exported XYZ: {path}")
        return path

    def export_orca_input(self, path="~/job.inp", method="B3LYP def2-SVP", charge=0, mult=1, extra=""):
        """Generate ORCA input file.
        method: ORCA simple input line (e.g. "B3LYP def2-TZVP Opt Freq")
        extra: additional blocks (e.g. '%scf MaxIter 200 end')
        """
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            f.write(f"! {method}\n")
            if extra: f.write(f"{extra}\n")
            f.write(f"\n*xyz {charge} {mult}\n")
            for a in self.atoms:
                f.write(f"  {a['el']}  {a['x']:.6f}  {a['y']:.6f}  {a['z']:.6f}\n")
            f.write("*\n")
        self.log(f"ORCA input: {path}")
        return path

    def export_orca_cube_input(self, path="~/job.inp", method="B3LYP def2-SVP",
                                charge=0, mult=1, orbitals=None, dim=60):
        """Generate ORCA input that produces cube files for orbitals.
        orbitals: list of (name, orbital_index) e.g. [("homo",4),("lumo",5)]
        """
        path = os.path.expanduser(path)
        if orbitals is None: orbitals = [("homo", "0,0"), ("lumo", "1,0")]
        with open(path, 'w') as f:
            f.write(f"! {method}\n\n%plots\n  Format Gaussian_Cube\n")
            f.write(f"  dim1 {dim}\n  dim2 {dim}\n  dim3 {dim}\n")
            for name, orb_spec in orbitals:
                f.write(f'  MO("{name}.cube", {orb_spec})\n')
            f.write("end\n")
            f.write(f"\n*xyz {charge} {mult}\n")
            for a in self.atoms:
                f.write(f"  {a['el']}  {a['x']:.6f}  {a['y']:.6f}  {a['z']:.6f}\n")
            f.write("*\n")
        self.log(f"ORCA+cube input: {path}")
        return path

    # ── ORCA ───────────────────────────────────────────────────

    def run_orca(self, inp_path, orca_cmd="orca"):
        """Run ORCA calculation asynchronously. Result in chem.orca_result when done."""
        inp_path = os.path.expanduser(inp_path)
        out_path = inp_path.replace('.inp', '.out')
        self.log(f"Running ORCA: {inp_path} ...")
        def _run():
            try:
                result = subprocess.run([orca_cmd, inp_path], capture_output=True, text=True, timeout=3600)
                with open(out_path, 'w') as f: f.write(result.stdout)
                self._signals.orca_done.emit(out_path)
            except Exception as ex:
                self._signals.status.emit(f"ORCA error: {ex}")
        threading.Thread(target=_run, daemon=True).start()
        self._signals.orca_done.connect(lambda p: self.parse_orca(p))
        return self

    def parse_orca(self, out_path):
        """Parse ORCA output file, update atoms, info, and display."""
        out_path = os.path.expanduser(out_path)
        with open(out_path, 'r') as f: text = f.read()
        result = parse_orca_output(text)
        self.orca_result = result
        if result['atoms']:
            self.atoms = result['atoms']
            self.bonds = detect_bonds(self.atoms)
            self.viewer.set_molecule(self.atoms, os.path.basename(out_path))
        self.info.update({
            'energy': result['total_energy'],
            'method': result['method'],
            'basis': result['basis'],
            'scf_ok': result['scf_ok'],
            'point_group': result['point_group'],
            'dipole': result['dipole'],
            'mulliken': result['mulliken'],
            'frequencies': result['frequencies'],
            'ir_intensities': result['ir_intensities'],
            'orb_labels': result['orb_labels'],
            'orb_energies': result['orb_energies'],
        })
        self.log(f"Parsed ORCA: {len(self.atoms)} atoms, E={result['total_energy']}")
        return self

    # ── PySCF COMPUTE ENGINE (real quantum chemistry) ──────────

    def calculate(self, method="B3LYP", basis="sto-3g", charge=0, mult=1,
                  freq=False, cube=False, cube_nx=50, opt=False):
        """Run a real quantum chemistry calculation using PySCF.

        Args:
            method: 'HF', 'B3LYP', 'PBE', 'PBE0', 'M06-2X', 'MP2', etc.
            basis:  'sto-3g', 'def2-svp', 'def2-tzvp', 'cc-pvdz', etc.
            charge: molecular charge (0 for neutral)
            mult:   spin multiplicity (1 for singlet)
            freq:   if True, compute vibrational frequencies + IR
            cube:   if True, generate HOMO/LUMO cube files and load HOMO
            cube_nx: cube file grid resolution per axis
            opt:    if True, optimize geometry first

        After calling, chem.info is populated with REAL computed data:
            energy, homo_energy, lumo_energy, gap, dipole, charges,
            mo_energies, n_occ, frequencies (if freq=True)

        Returns: self (for chaining)
        """
        if not self.atoms:
            self.log("No molecule loaded"); return self

        try:
            from pyscf import gto, scf, dft
            import numpy as np
        except ImportError:
            self.log("PySCF not installed: pip install pyscf"); return self

        self.log(f"Calculating {method}/{basis} on {len(self.atoms)} atoms ...")
        self.log(f"  Elements: {set(a['el'] for a in self.atoms)}")

        # Build PySCF molecule — filter out dummy/unknown atoms that PySCF can't handle
        _pyscf_valid = set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar '
                          'K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr '
                          'Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe '
                          'Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi U'.split())
        calc_atoms = [a for a in self.atoms if a['el'] in _pyscf_valid]
        if not calc_atoms:
            self.log("No valid atoms for PySCF — all atoms are unknown/dummy")
            self.log(f"  Atom elements: {[a['el'] for a in self.atoms]}")
            return self
        if len(calc_atoms) < len(self.atoms):
            skipped = len(self.atoms) - len(calc_atoms)
            bad_els = set(a['el'] for a in self.atoms if a['el'] not in _pyscf_valid)
            self.log(f"Skipping {skipped} atom(s) with unsupported elements: {bad_els}")
        atom_str = "; ".join(f"{a['el']} {a['x']} {a['y']} {a['z']}" for a in calc_atoms)
        try:
            mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=mult-1, unit='Angstrom', verbose=0)
        except Exception as ex:
            err_msg = str(ex)
            if 'Basis' in err_msg or 'basis' in err_msg:
                # Basis set doesn't cover some element — try per-element basis assignment
                self.log(f"Basis '{basis}' failed: {err_msg}")
                unique_els = set(a['el'] for a in calc_atoms)
                self.log(f"  Elements: {unique_els}. Trying fallback with def2-svp...")
                try:
                    mol = gto.M(atom=atom_str, basis='def2-svp', charge=charge,
                                spin=mult-1, unit='Angstrom', verbose=0)
                    basis = 'def2-svp'
                    self.log(f"  Fallback to def2-svp succeeded")
                except Exception as ex2:
                    self.log(f"Fallback basis also failed: {ex2}"); return self
            else:
                self.log(f"PySCF error building molecule: {ex}"); return self

        # Choose method
        meth_up = method.upper()
        if meth_up == 'HF':
            if mult > 1: mf = scf.UHF(mol)
            else: mf = scf.RHF(mol)
        elif meth_up == 'MP2':
            mf = scf.RHF(mol).run(verbose=0)
            from pyscf import mp
            pt = mp.MP2(mf).run(verbose=0)
            # MP2 doesn't have orbital-level output, use HF orbitals
            self.info['mp2_corr'] = pt.e_corr
            self.log(f"MP2 correlation: {pt.e_corr:.6f} Eh")
            # Fall through to use mf for orbitals
        else:
            # DFT
            if mult > 1: mf = dft.UKS(mol)
            else: mf = dft.RKS(mol)
            mf.xc = method.lower()

        # Geometry optimization
        if opt:
            self.log("Optimizing geometry...")
            try:
                from pyscf.geomopt import geometric_solver
                mol_eq = geometric_solver.optimize(mf, verbose=0)
                # Update atoms with optimized coordinates
                coords = mol_eq.atom_coords(unit='Angstrom')
                for i, a in enumerate(self.atoms):
                    a['x'], a['y'], a['z'] = float(coords[i][0]), float(coords[i][1]), float(coords[i][2])
                self.bonds = detect_bonds(self.atoms)
                self.viewer.set_molecule(self.atoms, self.viewer.mol_name)
                # Rebuild mol/mf with optimized geometry
                atom_str = "; ".join(f"{a['el']} {a['x']} {a['y']} {a['z']}" for a in self.atoms)
                mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=mult-1, unit='Angstrom', verbose=0)
                if meth_up == 'HF':
                    mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
                else:
                    mf = dft.RKS(mol) if mult == 1 else dft.UKS(mol)
                    mf.xc = method.lower()
                self.log("Geometry optimized")
            except Exception as ex:
                self.log(f"Optimization failed: {ex}, continuing with current geometry")

        # Run SCF
        mf.run(verbose=0)

        if not mf.converged:
            self.log("WARNING: SCF did not converge!")

        # Extract results
        n_occ = mol.nelectron // 2
        mo_eV = mf.mo_energy * 27.2114  # Hartree to eV

        # Mulliken charges
        try:
            charges_raw = mf.mulliken_pop(verbose=0)[1]
            charges = [float(c) for c in charges_raw]
        except:
            charges = [0.0] * len(self.atoms)

        # Dipole
        try:
            dip = mf.dip_moment(verbose=0)
            dipole_mag = float(np.linalg.norm(dip))
        except:
            dipole_mag = 0.0

        # Store everything
        self.info.update({
            'energy': float(mf.e_tot),
            'method': f"{method}/{basis}",
            'basis': basis,
            'scf_ok': mf.converged,
            'homo_energy': float(mo_eV[n_occ-1]),
            'lumo_energy': float(mo_eV[n_occ]) if n_occ < len(mo_eV) else None,
            'gap': float(mo_eV[n_occ] - mo_eV[n_occ-1]) if n_occ < len(mo_eV) else None,
            'dipole': dipole_mag,
            'charges': charges,
            'mulliken': [{'idx':i,'el':self.atoms[i]['el'],'charge':charges[i]} for i in range(len(charges))],
            'mo_energies': [float(e) for e in mo_eV],
            'n_occ': n_occ,
            'n_mo': len(mo_eV),
        })
        self._pyscf_mf = mf   # keep reference for orbital generation
        self._pyscf_mol = mol

        self.log(f"Energy: {mf.e_tot:.8f} Eh")
        self.log(f"HOMO: {mo_eV[n_occ-1]:.3f} eV  LUMO: {mo_eV[n_occ]:.3f} eV  gap: {mo_eV[n_occ]-mo_eV[n_occ-1]:.3f} eV")
        self.log(f"Dipole: {dipole_mag:.4f} Debye")

        # Frequencies
        if freq:
            self.log("Computing frequencies...")
            try:
                h = mf.Hessian().kernel()
                from pyscf.hessian import thermo
                results = thermo.harmonic_analysis(mol, h)
                freqs_all = results['freq_wavenumber']
                real_freqs = [float(f) for f in freqs_all if f > 0]
                # IR intensities from dipole derivatives (approximate)
                ir_ints = [10.0] * len(real_freqs)  # PySCF doesn't compute IR intensities directly
                # But we can get them from dipole derivative if available
                try:
                    from pyscf.prop import infrared
                    ir = infrared.Infrared(mf).kernel()
                    if hasattr(ir, 'ir_inten'):
                        ir_ints = [float(x) for x in ir.ir_inten[:len(real_freqs)]]
                except:
                    pass
                self.info['frequencies'] = real_freqs
                self.info['ir_intensities'] = ir_ints
                self.log(f"Frequencies: {', '.join(f'{f:.0f}' for f in real_freqs[:6])}{'...' if len(real_freqs)>6 else ''} cm-1")
            except Exception as ex:
                self.log(f"Frequency calculation failed: {ex}")

        # Cube files
        if cube:
            self.log("Generating orbital cube files...")
            try:
                from pyscf.tools import cubegen
                import tempfile
                td = tempfile.mkdtemp(prefix="chemlab_")
                homo_path = os.path.join(td, "homo.cube")
                lumo_path = os.path.join(td, "lumo.cube")
                cubegen.orbital(mol, homo_path, mf.mo_coeff[:,n_occ-1], nx=cube_nx, ny=cube_nx, nz=cube_nx)
                cubegen.orbital(mol, lumo_path, mf.mo_coeff[:,n_occ], nx=cube_nx, ny=cube_nx, nz=cube_nx)
                self.info['homo_cube'] = homo_path
                self.info['lumo_cube'] = lumo_path
                self.info['cube_dir'] = td
                # Auto-load HOMO
                self.load_cube(homo_path)
                self.log(f"HOMO/LUMO cubes: {td}")
            except Exception as ex:
                self.log(f"Cube generation failed: {ex}")

        return self

    def show_homo(self, isovalue=0.02):
        """Show HOMO isosurface (requires calculate(cube=True) first, or generates on the fly)."""
        if 'homo_cube' in self.info:
            self.load_cube(self.info['homo_cube'], isovalue)
        elif hasattr(self, '_pyscf_mf'):
            self.log("Generating HOMO cube...")
            from pyscf.tools import cubegen
            import tempfile
            td = tempfile.mkdtemp(prefix="chemlab_")
            path = os.path.join(td, "homo.cube")
            n_occ = self.info.get('n_occ', self._pyscf_mol.nelectron//2)
            cubegen.orbital(self._pyscf_mol, path, self._pyscf_mf.mo_coeff[:,n_occ-1], nx=50, ny=50, nz=50)
            self.info['homo_cube'] = path
            self.load_cube(path, isovalue)
        else:
            self.log("Run chem.calculate() first")
        return self

    def show_lumo(self, isovalue=0.02):
        """Show LUMO isosurface."""
        if 'lumo_cube' in self.info:
            self.load_cube(self.info['lumo_cube'], isovalue)
        elif hasattr(self, '_pyscf_mf'):
            self.log("Generating LUMO cube...")
            from pyscf.tools import cubegen
            import tempfile
            td = tempfile.mkdtemp(prefix="chemlab_")
            path = os.path.join(td, "lumo.cube")
            n_occ = self.info.get('n_occ', self._pyscf_mol.nelectron//2)
            cubegen.orbital(self._pyscf_mol, path, self._pyscf_mf.mo_coeff[:,n_occ], nx=50, ny=50, nz=50)
            self.info['lumo_cube'] = path
            self.load_cube(path, isovalue)
        else:
            self.log("Run chem.calculate() first")
        return self

    def show_orbital(self, n, isovalue=0.02):
        """Show any MO by index (0-based). Generates cube on the fly."""
        if not hasattr(self, '_pyscf_mf'):
            self.log("Run chem.calculate() first"); return self
        from pyscf.tools import cubegen
        import tempfile
        td = tempfile.mkdtemp(prefix="chemlab_")
        path = os.path.join(td, f"mo_{n}.cube")
        cubegen.orbital(self._pyscf_mol, path, self._pyscf_mf.mo_coeff[:,n], nx=50, ny=50, nz=50)
        self.load_cube(path, isovalue)
        n_occ = self.info.get('n_occ', 0)
        label = "HOMO" if n == n_occ-1 else "LUMO" if n == n_occ else f"MO {n}"
        eV = self.info.get('mo_energies', [None]*(n+1))[n]
        self.log(f"Showing {label} (MO {n}): {eV:.3f} eV" if eV else f"Showing MO {n}")
        return self

    # ── Analysis helpers ───────────────────────────────────────

    def formula(self):
        counts = {}
        for a in self.atoms: counts[a['el']] = counts.get(a['el'],0)+1
        keys = sorted(counts.keys(), key=lambda x: ('A' if x=='C' else 'B' if x=='H' else x))
        return ''.join(f"{k}{counts[k] if counts[k]>1 else ''}" for k in keys)

    def mass(self):
        return sum(ELEMENTS.get(a['el'],{'mass':0})['mass'] for a in self.atoms)

    def center_of_mass(self):
        total = 0; cx=cy=cz=0
        for a in self.atoms:
            m = ELEMENTS.get(a['el'],{'mass':1})['mass']
            cx+=a['x']*m; cy+=a['y']*m; cz+=a['z']*m; total+=m
        return (cx/total, cy/total, cz/total) if total else (0,0,0)

    def overlay_ir(self, width=350, height=180):
        """Show IR spectrum as overlay. Uses real frequencies from calculate(freq=True)."""
        freqs = self.info.get('frequencies', [])
        ints = self.info.get('ir_intensities', [])
        if not freqs: self.log("No frequency data — run chem.calculate(freq=True)"); return self
        # Normalize intensities
        if not ints or len(ints) != len(freqs):
            ints = [50.0] * len(freqs)
        imax = max(ints) if max(ints) > 0 else 1.0
        method = self.info.get('method', '?')
        def draw_ir(painter, w, h):
            ox, oy = w-width-16, h-height-16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,210))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas",8,QFont.Bold)); painter.setPen(QColor(50,60,80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, f"IR Spectrum ({method})")
            pad_l,pad_r,pad_t,pad_b = 36,12,22,28
            ax0=ox+pad_l; ax1=ox+width-pad_r; ay0=oy+pad_t; ay1=oy+height-pad_b
            aw=ax1-ax0; ah=ay1-ay0
            painter.setPen(QPen(QColor(180,185,200),1))
            painter.drawLine(ax0,ay1,ax1,ay1)
            fmin,fmax = min(min(freqs)-100, 400), max(max(freqs)+100, 3200)
            # Lorentzian broadening
            sigma = max(15, (fmax-fmin)/80)
            pts = []
            for xi in range(aw):
                f = fmin + xi/aw*(fmax-fmin)
                val = sum(I/imax / (1+((f-fq)/sigma)**2) for fq,I in zip(freqs,ints))
                pts.append(val)
            peak = max(pts) if pts else 1
            from PySide6.QtGui import QPainterPath
            path = QPainterPath(); path.moveTo(ax0, ay1)
            for xi, val in enumerate(pts):
                path.lineTo(ax0+xi, ay1-(val/peak)*ah*0.85)
            path.lineTo(ax1, ay1); path.closeSubpath()
            grad = QLinearGradient(ax0,ay0,ax0,ay1)
            grad.setColorAt(0, QColor(70,130,200,160)); grad.setColorAt(1, QColor(70,130,200,20))
            painter.setBrush(grad); painter.setPen(QPen(QColor(50,100,180),1.2))
            painter.drawPath(path)
            # Peak labels
            painter.setFont(QFont("Consolas",7)); painter.setPen(QColor(180,60,60))
            for fq, I in zip(freqs, ints):
                if I/imax > 0.15:
                    tx = ax0+(fq-fmin)/(fmax-fmin)*aw
                    painter.drawText(int(tx)-18, ay0-2, 36, 12, Qt.AlignCenter, f"{fq:.0f}")
            # X label
            painter.setFont(QFont("Consolas",7)); painter.setPen(QColor(100,110,130))
            painter.drawText(ax0, ay1+4, aw, 14, Qt.AlignCenter, "cm\u207b\u00b9")
        self.viewer.add_overlay('ir_spectrum', draw_ir)
        self.log(f"IR overlay: {len(freqs)} modes")
        return self

    # ══════════════════════════════════════════════════════════
    #  VOLUMETRIC DATA — ESP, ELF, density diffs, spin density
    # ══════════════════════════════════════════════════════════

    # ── 7.1: Electrostatic Potential mapped onto density ──────

    def show_esp(self, density_iso=0.05, vmin=None, vmax=None):
        """Render ESP color-mapped onto the electron density isosurface.
        Requires calculate(cube=True, esp=True) or externally loaded cubes.
        Like VMD's 'Volume' coloring method from section 7.1."""
        dens = self.info.get('_density_cube')
        esp = self.info.get('_esp_cube')
        if dens is None or esp is None:
            self.log("Need density + ESP cubes. Run calculate(cube=True, esp=True) or load_esp()")
            return self
        self.viewer.set_colormapped_surface(dens, esp, density_iso, vmin, vmax)
        self.log(f"ESP mapped onto density (iso={density_iso})")
        return self

    def load_esp(self, density_path, esp_path, density_iso=0.05):
        """Load external density and ESP cube files and display ESP-mapped surface.
        Equivalent to h2o-dens.cube + h2o-pot.cube from the tutorial."""
        dens_path = os.path.expanduser(density_path)
        esp_path_exp = os.path.expanduser(esp_path)
        with open(dens_path) as f: dens = parse_cube(f.read())
        with open(esp_path_exp) as f: esp = parse_cube(f.read())
        if not dens or not esp:
            self.log("Failed to parse cube files"); return self
        self.info['_density_cube'] = dens
        self.info['_esp_cube'] = esp
        self.atoms = dens['atoms']; self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, "ESP map")
        self.viewer.set_colormapped_surface(dens, esp, density_iso)
        self.log(f"ESP mapped onto density surface")
        return self

    def show_volume_slice(self, axis=1, pos=0.5, source='esp'):
        """Show a 2D cross-section through volumetric data.
        axis: 0=X, 1=Y, 2=Z. pos: fractional [0,1].
        source: 'esp', 'density', 'elf', or a cube dict."""
        if isinstance(source, dict):
            cube = source
        elif source == 'esp':
            cube = self.info.get('_esp_cube')
        elif source == 'density':
            cube = self.info.get('_density_cube')
        elif source == 'elf':
            cube = self.info.get('_elf_cube')
        else:
            cube = self._cube_data
        if cube is None:
            self.log(f"No {source} cube data available"); return self
        self.viewer.set_volume_slice(cube, axis, pos)
        self.log(f"Volume slice: {['X','Y','Z'][axis]}={pos:.2f}")
        return self

    def clear_volume_slice(self):
        """Remove volume slice display."""
        self.viewer.clear_volume_slice()
        return self

    # ── 7.2: Localized (Wannier) orbitals ────────────────────

    def show_localized_orbitals(self, isovalue=0.02):
        """Compute and display localized (Boys/Pipek-Mezey) orbitals using PySCF.
        Renders all occupied localized orbitals as stacked isosurfaces."""
        if not hasattr(self, '_pyscf_mf'):
            self.log("Run chem.calculate() first"); return self
        try:
            from pyscf import lo
            from pyscf.tools import cubegen
            import tempfile
        except ImportError:
            self.log("PySCF localization module not available"); return self
        mol = self._pyscf_mol; mf = self._pyscf_mf
        n_occ = self.info.get('n_occ', mol.nelectron//2)
        occ_coeffs = mf.mo_coeff[:, :n_occ]
        self.log(f"Localizing {n_occ} occupied orbitals (Boys)...")
        loc_orb = lo.Boys(mol, occ_coeffs).kernel()
        td = tempfile.mkdtemp(prefix="chemlab_loc_")
        self.viewer.clear_iso_layers()
        colors = [(0x3070CC, 0xCC3030), (0x30CC70, 0xCC7030), (0x7030CC, 0x30CCCC), (0xCC30CC, 0x70CC30)]
        all_verts = []
        for i in range(n_occ):
            path = os.path.join(td, f"loc_{i}.cube")
            cubegen.orbital(mol, path, loc_orb[:, i], nx=40, ny=40, nz=40)
            with open(path) as f: cube = parse_cube(f.read())
            if cube is None: continue
            col_p, col_n = colors[i % len(colors)]
            g, o, ax, n = cube['grid'], cube['origin'], cube['axes'], cube['npts']
            pos_raw = _marching_cubes(g, o, ax, n, isovalue)
            neg_raw = _marching_cubes(g, o, ax, n, -isovalue)
            rp, gp, bp = _hex(col_p); rn, gn, bn = _hex(col_n)
            verts = []
            for vi in range(0, len(pos_raw), 6):
                verts.extend(pos_raw[vi:vi+6]); verts.extend((rp, gp, bp))
            for vi in range(0, len(neg_raw), 6):
                verts.extend(neg_raw[vi:vi+6]); verts.extend((rn, gn, bn))
            if verts:
                cx, cy, cz = self.viewer._center
                arr = np.array(verts, dtype=np.float32).reshape(-1, 9)
                arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
                self.viewer.add_iso_layer(arr.flatten().tolist(), 0.45, f"loc_{i}")
            self.info[f'loc_orb_{i}_cube'] = path
        self.log(f"Showing {n_occ} localized orbitals")
        return self

    # ── 7.3: Electron Localization Function (ELF) ────────────

    def show_elf(self, isovalue=0.85):
        """Compute and display the Electron Localization Function.
        ELF describes chemical bonding topologically — attractors
        correspond to bonds, lone pairs, and atomic shells."""
        if not hasattr(self, '_pyscf_mf'):
            self.log("Run chem.calculate() first"); return self
        try:
            from pyscf import dft as pyscf_dft
            from pyscf.tools import cubegen
            import tempfile
        except ImportError:
            self.log("PySCF not available"); return self
        mol = self._pyscf_mol; mf = self._pyscf_mf
        self.log("Computing ELF...")
        # Compute electron density and kinetic energy density on a grid
        td = tempfile.mkdtemp(prefix="chemlab_elf_")
        nx = 50
        # Generate density cube for the grid
        n_occ = self.info.get('n_occ', mol.nelectron // 2)
        dens_path = os.path.join(td, "density.cube")
        cubegen.density(mol, dens_path, mf.make_rdm1(), nx=nx, ny=nx, nz=nx)
        with open(dens_path) as f: dens_cube = parse_cube(f.read())
        if dens_cube is None:
            self.log("Failed to generate density cube"); return self
        # Compute ELF on the same grid via orbital evaluation
        g = dens_cube['grid']; o = dens_cube['origin']
        ax_arr = dens_cube['axes']; npts = dens_cube['npts']
        n1, n2, n3 = npts
        # Build real-space grid coords
        coords = np.zeros((n1*n2*n3, 3))
        idx = 0
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    coords[idx] = (o + np.array(ax_arr[0])*i + np.array(ax_arr[1])*j + np.array(ax_arr[2])*k) / BOHR
                    idx += 1
        # Evaluate orbitals and their gradients
        from pyscf.dft import numint
        ao = numint.eval_ao(mol, coords, deriv=1)  # (4, ngrids, nao) — value, dx, dy, dz
        mo_coeff = mf.mo_coeff[:, :n_occ]
        mo_val = ao[0] @ mo_coeff       # (ngrids, n_occ)
        mo_dx  = ao[1] @ mo_coeff
        mo_dy  = ao[2] @ mo_coeff
        mo_dz  = ao[3] @ mo_coeff
        # Kinetic energy density τ = 0.5 Σ |∇ψ_i|²
        tau = 0.5 * np.sum(mo_dx**2 + mo_dy**2 + mo_dz**2, axis=1)
        # Electron density ρ = 2 Σ |ψ_i|² (factor 2 for closed shell)
        rho = 2.0 * np.sum(mo_val**2, axis=1)
        rho = np.maximum(rho, 1e-30)
        # Thomas-Fermi kinetic energy density
        cf = (3.0/10.0) * (3.0 * np.pi**2)**(2.0/3.0)
        tau_tf = cf * rho**(5.0/3.0)
        # von Weizsäcker kinetic energy density
        grad_rho_sq = (2.0 * np.sum(mo_val * mo_dx, axis=1))**2 + \
                      (2.0 * np.sum(mo_val * mo_dy, axis=1))**2 + \
                      (2.0 * np.sum(mo_val * mo_dz, axis=1))**2
        tau_w = grad_rho_sq / (8.0 * rho)
        # ELF = 1 / (1 + (D/D0)²) where D = τ - τ_W, D0 = τ_TF
        D = np.maximum(tau - tau_w, 0.0)
        D0 = np.maximum(tau_tf, 1e-30)
        chi = D / D0
        elf_vals = 1.0 / (1.0 + chi**2)
        elf_grid = elf_vals.reshape(n1, n2, n3)
        elf_cube = {**dens_cube, 'grid': elf_grid}
        self.info['_elf_cube'] = elf_cube
        self.info['_density_cube'] = dens_cube
        # Save ELF cube file
        elf_path = os.path.join(td, "elf.cube")
        cube_write(elf_cube, elf_path)
        self.info['elf_cube'] = elf_path
        # Display: ELF isosurface with a green-yellow palette
        verts = _marching_cubes(elf_grid, o, ax_arr, npts, isovalue)
        rg, gg, bg = _hex(0x40C060)
        result = []
        for vi in range(0, len(verts), 6):
            result.extend(verts[vi:vi+6]); result.extend((rg, gg, bg))
        if result:
            cx, cy, cz = self.viewer._center
            arr = np.array(result, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            self.viewer.set_orbital_mesh(arr.flatten().tolist())
        self.log(f"ELF computed (isovalue={isovalue})")
        return self

    def load_elf(self, path, isovalue=0.85):
        """Load a pre-computed ELF cube file."""
        path = os.path.expanduser(path)
        with open(path) as f: cube = parse_cube(f.read())
        if cube is None:
            self.log("Failed to parse ELF cube"); return self
        self.info['_elf_cube'] = cube
        self.atoms = cube['atoms']; self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, "ELF")
        verts = _marching_cubes(cube['grid'], cube['origin'], cube['axes'], cube['npts'], isovalue)
        rg, gg, bg = _hex(0x40C060)
        result = []
        for vi in range(0, len(verts), 6):
            result.extend(verts[vi:vi+6]); result.extend((rg, gg, bg))
        if result:
            cx, cy, cz = self.viewer._center
            arr = np.array(result, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            self.viewer.set_orbital_mesh(arr.flatten().tolist())
        self.log(f"Loaded ELF: {path}")
        return self

    # ── 7.4: Density difference (response to perturbation) ───

    def density_difference(self, cube_a, cube_b, isovalue=0.002):
        """Display the difference A - B of two density cube files/dicts.
        Shows areas of increased (yellow) and decreased (green) electron density.
        Like the cubman 'subtract' workflow from section 7.4."""
        if isinstance(cube_a, str):
            with open(os.path.expanduser(cube_a)) as f: cube_a = parse_cube(f.read())
        if isinstance(cube_b, str):
            with open(os.path.expanduser(cube_b)) as f: cube_b = parse_cube(f.read())
        if not cube_a or not cube_b:
            self.log("Failed to parse cube files"); return self
        diff = cube_subtract(cube_a, cube_b)
        self.info['_diff_cube'] = diff
        self.atoms = diff['atoms']; self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, "Density difference")
        g, o, ax, n = diff['grid'], diff['origin'], diff['axes'], diff['npts']
        # Positive = increased density (yellow), negative = decreased (green)
        pos_raw = _marching_cubes(g, o, ax, n, isovalue)
        neg_raw = _marching_cubes(g, o, ax, n, -isovalue)
        rp, gp, bp = _hex(0xDDCC30)  # yellow
        rn, gn, bn = _hex(0x30CC70)  # green
        result = []
        for vi in range(0, len(pos_raw), 6):
            result.extend(pos_raw[vi:vi+6]); result.extend((rp, gp, bp))
        for vi in range(0, len(neg_raw), 6):
            result.extend(neg_raw[vi:vi+6]); result.extend((rn, gn, bn))
        if result:
            cx, cy, cz = self.viewer._center
            arr = np.array(result, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            self.viewer.set_orbital_mesh(arr.flatten().tolist())
        self.log(f"Density difference (iso=±{isovalue})")
        return self

    def cube_math(self, op, cube_a, cube_b=None, scale=1.0):
        """General cube arithmetic. op: 'add', 'subtract', 'scale', 'write'.
        Returns the resulting cube dict (or path for 'write')."""
        if isinstance(cube_a, str):
            with open(os.path.expanduser(cube_a)) as f: cube_a = parse_cube(f.read())
        if cube_b and isinstance(cube_b, str):
            with open(os.path.expanduser(cube_b)) as f: cube_b = parse_cube(f.read())
        if op == 'add': return cube_add(cube_a, cube_b)
        elif op == 'subtract': return cube_subtract(cube_a, cube_b)
        elif op == 'scale': return cube_scale(cube_a, scale)
        elif op == 'write':
            path = os.path.expanduser(cube_b) if isinstance(cube_b, str) else "~/result.cube"
            return cube_write(cube_a, os.path.expanduser(path))
        return cube_a

    # ── 7.5: Bulk systems ────────────────────────────────────

    def load_bulk(self, density_path, esp_path=None, center=True, inbox=True):
        """Load volumetric data for a periodic/bulk system.
        Optionally centers atoms and applies inbox correction (section 7.5)."""
        dens_path = os.path.expanduser(density_path)
        with open(dens_path) as f: dens = parse_cube(f.read())
        if not dens:
            self.log("Failed to parse density cube"); return self
        if inbox: dens = cube_center_inbox(dens)
        self.info['_density_cube'] = dens
        self.atoms = dens['atoms']; self.bonds = detect_bonds(self.atoms)
        name = os.path.basename(density_path)
        self.viewer.set_molecule(self.atoms, f"Bulk: {name}")
        self.viewer.set_cube(dens, 0.05)
        if esp_path:
            esp_path_exp = os.path.expanduser(esp_path)
            with open(esp_path_exp) as f: esp = parse_cube(f.read())
            if esp:
                if inbox: esp = cube_center_inbox(esp)
                self.info['_esp_cube'] = esp
                self.log("ESP data loaded — use show_esp() or show_bulk_esp()")
        self.log(f"Loaded bulk system: {len(self.atoms)} atoms")
        return self

    def show_bulk_esp(self, density_iso=0.05, esp_isovals=None):
        """Display bulk system with multiple ESP isosurfaces at different values.
        esp_isovals: list of (value, color_hex) pairs. Default: blue/red/pink."""
        dens = self.info.get('_density_cube')
        esp = self.info.get('_esp_cube')
        if not dens:
            self.log("Load bulk system first"); return self
        self.viewer.clear_iso_layers()
        if esp_isovals is None:
            esp_isovals = [(-0.02, 0x3060CC), (0.02, 0xCC3030), (0.05, 0xDD80AA)]
        if esp:
            cx, cy, cz = self.viewer._center
            for val, col in esp_isovals:
                rp, gp, bp = _hex(col)
                raw = _marching_cubes(esp['grid'], esp['origin'], esp['axes'], esp['npts'], val)
                verts = []
                for vi in range(0, len(raw), 6):
                    verts.extend(raw[vi:vi+6]); verts.extend((rp, gp, bp))
                if verts:
                    arr = np.array(verts, dtype=np.float32).reshape(-1, 9)
                    arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
                    self.viewer.add_iso_layer(arr.flatten().tolist(), 0.4, f"ESP_{val}")
        # Density surface (green, semi-transparent)
        rg, gg, bg = _hex(0x40CC60)
        raw = _marching_cubes(dens['grid'], dens['origin'], dens['axes'], dens['npts'], density_iso)
        verts = []
        for vi in range(0, len(raw), 6):
            verts.extend(raw[vi:vi+6]); verts.extend((rg, gg, bg))
        if verts:
            arr = np.array(verts, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            self.viewer.add_iso_layer(arr.flatten().tolist(), 0.25, "density")
        self.log(f"Bulk ESP display: {len(esp_isovals)} ESP layers + density")
        return self

    # ── 7.6: Animated isosurfaces (trajectory) ───────────────

    def load_animation(self, cube_paths, isovalue=0.02):
        """Load a sequence of cube files for animated isosurface playback.
        cube_paths: list of file paths, or a glob pattern string."""
        if isinstance(cube_paths, str):
            import glob
            cube_paths = sorted(glob.glob(os.path.expanduser(cube_paths)))
        cubes = []
        for p in cube_paths:
            with open(p) as f: c = parse_cube(f.read())
            if c: cubes.append(c)
        if not cubes:
            self.log("No valid cube files found"); return self
        # Use first frame's atoms for the molecule display
        self.atoms = cubes[0]['atoms']; self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, f"Animation ({len(cubes)} frames)")
        self.viewer.set_animation_cubes(cubes, isovalue)
        self.log(f"Loaded animation: {len(cubes)} frames")
        return self

    def play(self, interval_ms=200):
        """Start animated isosurface playback."""
        self.viewer.play_animation(interval_ms)
        self.log("Animation playing")
        return self

    def stop(self):
        """Stop animation."""
        self.viewer.stop_animation()
        self.log("Animation stopped")
        return self

    def frame(self, n):
        """Jump to animation frame n."""
        self.viewer._show_anim_frame(n)
        self.log(f"Frame {n}")
        return self

    # ── 7.7 extended: Spin density ───────────────────────────

    def show_spin_density(self, isovalue=0.005):
        """Compute and display alpha-beta spin density difference.
        Requires an open-shell calculation (mult > 1).
        Shows where unpaired electrons are localized (section 7.7)."""
        if not hasattr(self, '_pyscf_mf'):
            self.log("Run chem.calculate() first"); return self
        mf = self._pyscf_mf; mol = self._pyscf_mol
        try:
            from pyscf.tools import cubegen
            import tempfile
        except ImportError:
            self.log("PySCF not available"); return self
        # Check if UHF/UKS
        dm = mf.make_rdm1()
        if isinstance(dm, np.ndarray) and dm.ndim == 3 and dm.shape[0] == 2:
            # dm[0] = alpha, dm[1] = beta
            spin_dm = dm[0] - dm[1]
        elif hasattr(mf, 'nelec') and hasattr(mf.nelec, '__len__'):
            # UHF/UKS with separate alpha/beta
            spin_dm = dm[0] - dm[1]
        else:
            self.log("Spin density requires open-shell (UHF/UKS). Use mult > 1")
            return self
        td = tempfile.mkdtemp(prefix="chemlab_spin_")
        path = os.path.join(td, "spin_density.cube")
        cubegen.density(mol, path, spin_dm, nx=50, ny=50, nz=50)
        with open(path) as f: cube = parse_cube(f.read())
        if cube is None:
            self.log("Failed to generate spin density cube"); return self
        self.info['_spin_cube'] = cube
        self.info['spin_density_cube'] = path
        g, o, ax, n = cube['grid'], cube['origin'], cube['axes'], cube['npts']
        # Alpha excess (blue), beta excess (red)
        pos_raw = _marching_cubes(g, o, ax, n, isovalue)
        neg_raw = _marching_cubes(g, o, ax, n, -isovalue)
        rp, gp, bp = _hex(0x3060CC)  # alpha = blue
        rn, gn, bn = _hex(0xCC3030)  # beta = red
        result = []
        for vi in range(0, len(pos_raw), 6):
            result.extend(pos_raw[vi:vi+6]); result.extend((rp, gp, bp))
        for vi in range(0, len(neg_raw), 6):
            result.extend(neg_raw[vi:vi+6]); result.extend((rn, gn, bn))
        if result:
            cx, cy, cz = self.viewer._center
            arr = np.array(result, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            self.viewer.set_orbital_mesh(arr.flatten().tolist())
        self.log(f"Spin density displayed (iso=±{isovalue})")
        return self

    def load_spin_density(self, alpha_path, beta_path, isovalue=0.005):
        """Load alpha and beta density cubes, display their difference."""
        return self.density_difference(alpha_path, beta_path, isovalue)

    # ── Multi-cube overlay ───────────────────────────────────

    def add_isosurface(self, cube_or_path, isovalue=0.02, color=0x3070CC, opacity=0.45, label=""):
        """Add an additional isosurface layer on top of the current view.
        Multiple calls stack up transparent layers."""
        if isinstance(cube_or_path, str):
            with open(os.path.expanduser(cube_or_path)) as f:
                cube = parse_cube(f.read())
        else:
            cube = cube_or_path
        if not cube:
            self.log("Failed to parse cube"); return self
        g, o, ax, n = cube['grid'], cube['origin'], cube['axes'], cube['npts']
        raw = _marching_cubes(g, o, ax, n, isovalue)
        rp, gp, bp = _hex(color)
        verts = []
        for vi in range(0, len(raw), 6):
            verts.extend(raw[vi:vi+6]); verts.extend((rp, gp, bp))
        if verts:
            cx, cy, cz = self.viewer._center
            arr = np.array(verts, dtype=np.float32).reshape(-1, 9)
            arr[:,0] -= cx; arr[:,1] -= cy; arr[:,2] -= cz
            self.viewer.add_iso_layer(arr.flatten().tolist(), opacity, label)
        self.log(f"Added isosurface layer: {label or 'unnamed'}")
        return self

    def clear_layers(self):
        """Remove all extra isosurface layers."""
        self.viewer.clear_iso_layers()
        return self

    # ── Extended calculate() with ESP + density cubes ─────────

    def calculate_full(self, method="B3LYP", basis="sto-3g", charge=0, mult=1,
                       freq=False, cube=True, esp=True, elf=False, cube_nx=50):
        """Full quantum chemistry calculation with density, ESP, and optionally ELF.
        This is the all-in-one method that generates everything from section 7.1-7.3.

        After calling:
            chem.show_esp()              — ESP mapped onto density
            chem.show_elf()              — electron localization function
            chem.show_homo()             — HOMO isosurface
            chem.show_volume_slice()     — 2D cross-section
        """
        # Run base calculation with cube generation
        self.calculate(method=method, basis=basis, charge=charge, mult=mult,
                      freq=freq, cube=cube, cube_nx=cube_nx)

        if not hasattr(self, '_pyscf_mf'):
            return self

        mol = self._pyscf_mol; mf = self._pyscf_mf
        td = self.info.get('cube_dir')
        if not td:
            import tempfile
            td = tempfile.mkdtemp(prefix="chemlab_full_")
            self.info['cube_dir'] = td

        # Generate density cube
        if cube or esp:
            self.log("Generating density cube...")
            try:
                from pyscf.tools import cubegen
                dens_path = os.path.join(td, "density.cube")
                cubegen.density(mol, dens_path, mf.make_rdm1(), nx=cube_nx, ny=cube_nx, nz=cube_nx)
                with open(dens_path) as f: dens = parse_cube(f.read())
                self.info['_density_cube'] = dens
                self.info['density_cube'] = dens_path
                self.log(f"Density cube: {dens_path}")
            except Exception as ex:
                self.log(f"Density cube failed: {ex}")

        # Generate ESP cube
        if esp:
            self.log("Computing electrostatic potential...")
            try:
                from pyscf.tools import cubegen
                esp_path = os.path.join(td, "esp.cube")
                # PySCF ESP on grid
                cubegen.mep(mol, esp_path, mf.make_rdm1(), nx=cube_nx, ny=cube_nx, nz=cube_nx)
                with open(esp_path) as f: esp_cube = parse_cube(f.read())
                self.info['_esp_cube'] = esp_cube
                self.info['esp_cube'] = esp_path
                self.log(f"ESP cube: {esp_path}")
            except Exception as ex:
                self.log(f"ESP computation failed: {ex}")

        # ELF
        if elf:
            self.show_elf()

        return self

    # ── Cube file I/O helpers ────────────────────────────────

    def save_cube(self, cube_or_key, path):
        """Save a cube dataset to file.
        cube_or_key: cube dict, or string key like 'density', 'esp', 'elf', 'diff'."""
        if isinstance(cube_or_key, str):
            key_map = {'density': '_density_cube', 'esp': '_esp_cube',
                       'elf': '_elf_cube', 'diff': '_diff_cube', 'spin': '_spin_cube'}
            cube = self.info.get(key_map.get(cube_or_key, cube_or_key))
        else:
            cube = cube_or_key
        if cube is None:
            self.log(f"No cube data for '{cube_or_key}'"); return self
        p = os.path.expanduser(path)
        cube_write(cube, p)
        self.log(f"Saved cube: {p}")
        return p


# ═══════════════════════════════════════════════════════════════
#  LIGHT-MODE STYLESHEET
# ═══════════════════════════════════════════════════════════════

class ChemLabApp(QWidget):
    """Complete ChemLab UI — panel + 3D viewer + signal wiring.
    Instantiate once and access `app.chem` for the API."""

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
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._cur_dir = os.path.expanduser("~")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)

        # ── Panel ──
        panel = QWidget(); panel.setFixedWidth(290); panel.setStyleSheet(self._SS)
        panel.setAttribute(Qt.WA_TranslucentBackground, True)
        ps = QScrollArea(); ps.setWidgetResizable(True)
        ps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ps.setStyleSheet(
            "QScrollArea{border:none;background:transparent}"
            "QScrollBar:vertical{width:5px;background:transparent}"
            "QScrollBar::handle:vertical{background:#c0c8d4;border-radius:2px;min-height:30px}")
        inner = QWidget(); inner.setAttribute(Qt.WA_TranslucentBackground, True)
        lay = QVBoxLayout(inner); lay.setSpacing(4); lay.setContentsMargins(10, 10, 10, 10)

        # Header
        hdr = QWidget(); hdr.setAttribute(Qt.WA_TranslucentBackground, True)
        hl = QHBoxLayout(hdr); hl.setContentsMargins(0, 0, 0, 4)
        ic = QLabel("\u269b")
        ic.setStyleSheet("font-size:20px;background:rgba(230,240,250,200);border:1px solid #d0d8e4;border-radius:7px;padding:3px 7px")
        nw = QWidget(); nw.setAttribute(Qt.WA_TranslucentBackground, True)
        nl = QVBoxLayout(nw); nl.setContentsMargins(6, 0, 0, 0); nl.setSpacing(0)
        tl = QLabel("ChemLab"); tl.setStyleSheet("font-size:14px;font-weight:bold;color:#1a2a40;background:transparent")
        sl = QLabel("MOLECULAR WORKBENCH"); sl.setStyleSheet("font-size:7px;letter-spacing:2px;color:#8a96a8;background:transparent")
        nl.addWidget(tl); nl.addWidget(sl)
        hl.addWidget(ic); hl.addWidget(nw); hl.addStretch()
        lay.addWidget(hdr)

        tabs = QTabWidget(); tabs.setStyleSheet(self._SS)

        # ── Tab: Molecule ──
        t1 = QWidget(); t1.setAttribute(Qt.WA_TranslucentBackground, True)
        t1l = QVBoxLayout(t1); t1l.setSpacing(5); t1l.setContentsMargins(6, 8, 6, 6)
        t1l.addWidget(self._lbl("Presets"))
        self.mol_combo = QComboBox(); self.mol_combo.addItems(list(PRESETS.keys())); t1l.addWidget(self.mol_combo)
        t1l.addWidget(self._lbl("Style"))
        sw = QWidget(); sw.setAttribute(Qt.WA_TranslucentBackground, True)
        swl = QHBoxLayout(sw); swl.setContentsMargins(0, 0, 0, 0); swl.setSpacing(2)
        self.style_btns = {}
        for sn, slb in [("ballstick", "Ball&Stick"), ("spacefill", "SpaceFill"), ("wireframe", "Wire")]:
            b = QPushButton(slb); b.setCheckable(True); b.setChecked(sn == "ballstick")
            self.style_btns[sn] = b; swl.addWidget(b)
        t1l.addWidget(sw)
        self.bonds_cb = QCheckBox("Show Bonds"); self.bonds_cb.setChecked(True); t1l.addWidget(self.bonds_cb)
        t1l.addWidget(self._lbl("Export"))
        ew = QWidget(); ew.setAttribute(Qt.WA_TranslucentBackground, True)
        el = QHBoxLayout(ew); el.setContentsMargins(0, 0, 0, 0); el.setSpacing(2)
        self.xyz_btn = QPushButton("Export .xyz"); self.orca_btn = QPushButton("ORCA .inp")
        el.addWidget(self.xyz_btn); el.addWidget(self.orca_btn)
        t1l.addWidget(ew); t1l.addStretch(); tabs.addTab(t1, "Molecule")

        # ── Tab: Orbitals ──
        t2 = QWidget(); t2.setAttribute(Qt.WA_TranslucentBackground, True)
        t2l = QVBoxLayout(t2); t2l.setSpacing(5); t2l.setContentsMargins(6, 8, 6, 6)
        t2l.addWidget(self._lbl("Molecular Orbitals"))
        self.mo_list = QListWidget(); self.mo_list.setMinimumHeight(160); t2l.addWidget(self.mo_list)
        t2l.addWidget(self._lbl("Isovalue (for cube)"))
        self.iso_sl = QSlider(Qt.Horizontal); self.iso_sl.setRange(1, 100); self.iso_sl.setValue(20); t2l.addWidget(self.iso_sl)
        self.clear_btn = QPushButton("Clear Orbital"); t2l.addWidget(self.clear_btn)
        t2l.addWidget(self._lbl("Legend"))
        ll = QLabel("\u25cf \u03c8>0 (blue)  \u25cf \u03c8<0 (red)")
        ll.setStyleSheet("color:#5a6a7a;font-size:9px;background:transparent")
        t2l.addWidget(ll); t2l.addStretch(); tabs.addTab(t2, "Orbitals")

        # ── Tab: Import ──
        t3 = QWidget(); t3.setAttribute(Qt.WA_TranslucentBackground, True)
        t3l = QVBoxLayout(t3); t3l.setSpacing(5); t3l.setContentsMargins(6, 8, 6, 6)
        t3l.addWidget(self._lbl("Load .cube / .xyz / ORCA .out"))
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText("~/path/to/file.cube"); t3l.addWidget(self.path_edit)
        self.file_list = QListWidget(); self.file_list.setMinimumHeight(120); self.file_list.setMaximumHeight(160); t3l.addWidget(self.file_list)
        self.load_btn = QPushButton("Load Selected"); t3l.addWidget(self.load_btn)
        t3l.addWidget(self._lbl("ORCA Output (paste)"))
        self.orca_edit = QTextEdit(); self.orca_edit.setPlaceholderText("Paste ORCA .out here..."); self.orca_edit.setMinimumHeight(60); t3l.addWidget(self.orca_edit)
        self.parse_btn = QPushButton("Parse ORCA Output"); t3l.addWidget(self.parse_btn)
        self.status_lbl = QLabel(""); self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#6a7a8a;font-size:10px;background:transparent"); t3l.addWidget(self.status_lbl)
        t3l.addStretch(); tabs.addTab(t3, "Import")

        # ── Tab: Info ──
        t4 = QWidget(); t4.setAttribute(Qt.WA_TranslucentBackground, True)
        t4l = QVBoxLayout(t4); t4l.setSpacing(5); t4l.setContentsMargins(6, 8, 6, 6)
        t4l.addWidget(self._lbl("Properties"))
        self.info_edit = QTextEdit(); self.info_edit.setReadOnly(True); self.info_edit.setMinimumHeight(80); t4l.addWidget(self.info_edit)
        t4l.addWidget(self._lbl("Atoms"))
        self.atom_list = QListWidget(); self.atom_list.setMinimumHeight(100); t4l.addWidget(self.atom_list)
        t4l.addWidget(self._lbl("Log"))
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setPlainText("[ChemLab] Initialised\n[ChemLab] Renderer: ModernGL\n"); t4l.addWidget(self.log_edit)
        t4l.addStretch(); tabs.addTab(t4, "Info")

        # ── Tab: Volume ──
        t5 = QWidget(); t5.setAttribute(Qt.WA_TranslucentBackground, True)
        t5l = QVBoxLayout(t5); t5l.setSpacing(5); t5l.setContentsMargins(6, 8, 6, 6)
        t5l.addWidget(self._lbl("Volumetric Data"))
        t5l.addWidget(self._lbl("PySCF Compute"))
        self.calc_full_btn = QPushButton("Full Calc (dens+ESP+MOs)"); t5l.addWidget(self.calc_full_btn)
        self.elf_btn = QPushButton("Show ELF"); t5l.addWidget(self.elf_btn)
        self.spin_btn = QPushButton("Show Spin Density"); t5l.addWidget(self.spin_btn)
        self.loc_btn = QPushButton("Localized Orbitals"); t5l.addWidget(self.loc_btn)
        t5l.addWidget(self._lbl("Display"))
        self.esp_btn = QPushButton("ESP on Density"); t5l.addWidget(self.esp_btn)
        self.diff_btn = QPushButton("Density Difference"); t5l.addWidget(self.diff_btn)
        self.bulk_btn = QPushButton("Bulk ESP Layers"); t5l.addWidget(self.bulk_btn)
        t5l.addWidget(self._lbl("Volume Slice"))
        slice_w = QWidget(); slice_w.setAttribute(Qt.WA_TranslucentBackground, True)
        slice_lay = QHBoxLayout(slice_w); slice_lay.setContentsMargins(0, 0, 0, 0); slice_lay.setSpacing(2)
        self.slice_axis_combo = QComboBox(); self.slice_axis_combo.addItems(["X", "Y", "Z"]); self.slice_axis_combo.setCurrentIndex(1)
        slice_lay.addWidget(self.slice_axis_combo)
        self.slice_pos_sl = QSlider(Qt.Horizontal); self.slice_pos_sl.setRange(0, 100); self.slice_pos_sl.setValue(50)
        slice_lay.addWidget(self.slice_pos_sl)
        t5l.addWidget(slice_w)
        sw2 = QWidget(); sw2.setAttribute(Qt.WA_TranslucentBackground, True)
        sw2l = QHBoxLayout(sw2); sw2l.setContentsMargins(0, 0, 0, 0); sw2l.setSpacing(2)
        self.slice_btn = QPushButton("Show Slice"); self.clear_slice_btn = QPushButton("Clear Slice")
        sw2l.addWidget(self.slice_btn); sw2l.addWidget(self.clear_slice_btn)
        t5l.addWidget(sw2)
        t5l.addWidget(self._lbl("Animation"))
        aw = QWidget(); aw.setAttribute(Qt.WA_TranslucentBackground, True)
        awl = QHBoxLayout(aw); awl.setContentsMargins(0, 0, 0, 0); awl.setSpacing(2)
        self.anim_play_btn = QPushButton("\u25b6 Play"); self.anim_stop_btn = QPushButton("\u25a0 Stop")
        awl.addWidget(self.anim_play_btn); awl.addWidget(self.anim_stop_btn)
        t5l.addWidget(aw)
        self.clear_vol_btn = QPushButton("Clear All Volumes"); t5l.addWidget(self.clear_vol_btn)
        t5l.addStretch(); tabs.addTab(t5, "Volume")

        lay.addWidget(tabs); ps.setWidget(inner)
        playout = QVBoxLayout(panel); playout.setContentsMargins(0, 0, 0, 0); playout.addWidget(ps)

        # ── 3D Viewer ──
        self.viewer = MolViewer()
        self.viewer.setStyleSheet("background:transparent")
        main_layout.addWidget(panel); main_layout.addWidget(self.viewer, 1)

        # ── Create ChemLab singleton ──
        self.chem = ChemLab(self.viewer, self.log_edit)

        # ── Wire signals ──
        self._wire_signals()
        self._refresh_files()
        self.chem.load(self.mol_combo.currentText())

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _lbl(text):
        l = QLabel(text.upper())
        l.setStyleSheet("font-size:9px;letter-spacing:1.5px;color:#8a96a8;font-weight:bold;padding:2px 0;background:transparent")
        return l

    def _update_ui(self):
        """Sync UI with chem state after any load."""
        self.atom_list.clear()
        for i, a in enumerate(self.chem.atoms):
            el = ELEMENTS.get(a['el'], {'name': '?'})
            self.atom_list.addItem(f"{i:3d} {a['el']:2s} ({a['x']:+.3f},{a['y']:+.3f},{a['z']:+.3f}) {el['name']}")
        info_lines = [f"Name: {self.chem.info.get('name', '')}"]
        info_lines.append(f"Formula: {self.chem.formula()}")
        info_lines.append(f"Mass: {self.chem.mass():.3f} g/mol")
        if self.chem.info.get('energy'):
            info_lines.append(f"Energy: {self.chem.info['energy']:.8f} Eh")
        if self.chem.info.get('method'):
            info_lines.append(f"Method: {self.chem.info['method']}")
        if self.chem.info.get('dipole'):
            info_lines.append(f"Dipole: {self.chem.info['dipole']:.4f} D")
        if self.chem.info.get('point_group'):
            info_lines.append(f"Point group: {self.chem.info['point_group']}")
        self.info_edit.setPlainText('\n'.join(info_lines))

    # ── Signal wiring ──────────────────────────────────────────

    def _wire_signals(self):
        c = self.chem; v = self.viewer

        # Wrap chem methods to auto-update UI
        _orig_load = c.load
        def _load_wrap(name): _orig_load(name); self._update_ui(); return c
        c.load = _load_wrap

        _orig_load_xyz = c.load_xyz
        def _load_xyz_wrap(text): _orig_load_xyz(text); self._update_ui(); return c
        c.load_xyz = _load_xyz_wrap

        _orig_load_file = c.load_file
        def _load_file_wrap(path): _orig_load_file(path); self._update_ui(); return c
        c.load_file = _load_file_wrap

        _orig_load_cube = c.load_cube
        def _load_cube_wrap(p, iso=0.02): _orig_load_cube(p, iso); self._update_ui(); return c
        c.load_cube = _load_cube_wrap

        _orig_parse_orca = c.parse_orca
        def _parse_orca_wrap(p): _orig_parse_orca(p); self._update_ui(); return c
        c.parse_orca = _parse_orca_wrap

        # Molecule tab
        self.mol_combo.currentTextChanged.connect(lambda t: c.load(t))
        for sn, bt in self.style_btns.items():
            bt.clicked.connect(lambda checked, s=sn: self._on_style(s))
        self.bonds_cb.toggled.connect(lambda checked: (setattr(v, 'show_bonds', checked), v._rebuild_mol()))
        self.xyz_btn.clicked.connect(lambda: (c.export_xyz(), self.status_lbl.setText("\u2713 Saved ~/molecule.xyz")))
        self.orca_btn.clicked.connect(lambda: (c.export_orca_input(), self.status_lbl.setText("\u2713 Saved ~/job.inp")))

        # Import tab
        self.file_list.itemDoubleClicked.connect(self._on_file_dblclick)
        self.path_edit.returnPressed.connect(
            lambda: self._refresh_files(self.path_edit.text().strip())
            if os.path.isdir(self.path_edit.text().strip()) else None)
        self.load_btn.clicked.connect(self._on_load_selected)
        self.parse_btn.clicked.connect(self._on_parse_paste)

        # Orbitals tab
        self.iso_sl.valueChanged.connect(lambda val: c.set_isovalue(val / 1000.0))
        self.clear_btn.clicked.connect(lambda: c.clear_mo())
        self.atom_list.currentRowChanged.connect(
            lambda r: c.select(r) if 0 <= r < len(c.atoms) else None)

        # Volume tab
        self.calc_full_btn.clicked.connect(lambda: (c.calculate_full(), self._update_ui()))
        self.elf_btn.clicked.connect(lambda: c.show_elf())
        self.spin_btn.clicked.connect(lambda: c.show_spin_density())
        self.loc_btn.clicked.connect(lambda: c.show_localized_orbitals())
        self.esp_btn.clicked.connect(lambda: c.show_esp())
        self.diff_btn.clicked.connect(lambda: c.density_difference(
            c.info.get('_density_cube', {}), c.info.get('_esp_cube', {}), 0.002)
            if c.info.get('_density_cube') else c.log("Run Full Calc first"))
        self.bulk_btn.clicked.connect(lambda: c.show_bulk_esp())
        self.slice_btn.clicked.connect(lambda: c.show_volume_slice(
            axis=self.slice_axis_combo.currentIndex(), pos=self.slice_pos_sl.value() / 100.0))
        self.clear_slice_btn.clicked.connect(lambda: c.clear_volume_slice())
        self.slice_pos_sl.valueChanged.connect(
            lambda val: c.show_volume_slice(
                axis=self.slice_axis_combo.currentIndex(), pos=val / 100.0)
            if v._overlays.get('volume_slice') else None)
        self.anim_play_btn.clicked.connect(lambda: c.play())
        self.anim_stop_btn.clicked.connect(lambda: c.stop())
        self.clear_vol_btn.clicked.connect(
            lambda: (c.clear_mo(), c.clear_layers(), c.clear_volume_slice(), c.stop()))

    def _on_style(self, sn):
        for k, b in self.style_btns.items():
            b.setChecked(k == sn)
        self.chem.style(sn)

    # ── File browser ───────────────────────────────────────────

    def _refresh_files(self, d=None):
        if d:
            self._cur_dir = d
        self.path_edit.setText(self._cur_dir)
        self.file_list.clear()
        try:
            self.file_list.addItem("\u2190 ..")
            entries = sorted(os.listdir(self._cur_dir))
            dirs = [e for e in entries if os.path.isdir(os.path.join(self._cur_dir, e)) and not e.startswith('.')]
            files = [e for e in entries if os.path.isfile(os.path.join(self._cur_dir, e)) and not e.startswith('.')]
            shown = [f for f in files if f.endswith(('.cube', '.cub', '.xyz', '.out', '.inp'))]
            if not shown:
                shown = files[:40]
            for d in dirs[:25]:
                self.file_list.addItem(f"\U0001f4c1 {d}")
            for f in shown:
                self.file_list.addItem(
                    f"\U0001f7e6 {f}" if f.endswith(('.cube', '.cub')) else f"\U0001f4c4 {f}")
        except Exception:
            self.file_list.addItem("(error)")

    def _on_file_dblclick(self, item):
        name = item.text().split(" ", 1)[-1].strip() if " " in item.text() else item.text().strip()
        if item.text().startswith("\u2190"):
            parent = os.path.dirname(self._cur_dir)
            if parent != self._cur_dir:
                self._refresh_files(parent)
            return
        full = os.path.join(self._cur_dir, name)
        if os.path.isdir(full):
            self._refresh_files(full)

    def _on_load_selected(self):
        item = self.file_list.currentItem()
        if not item:
            self.status_lbl.setText("\u26a0 Select a file"); return
        name = item.text().split(" ", 1)[-1].strip() if " " in item.text() else item.text().strip()
        full = os.path.join(self._cur_dir, name)
        if os.path.isfile(full):
            self.chem.load_file(full)
            self.status_lbl.setText(f"\u2713 Loaded {name}")
        else:
            self.status_lbl.setText("\u26a0 Not a file")

    def _on_parse_paste(self):
        text = self.orca_edit.toPlainText()
        if not text.strip():
            self.status_lbl.setText("\u26a0 Paste ORCA output first"); return
        result = parse_orca_output(text)
        if result['atoms']:
            self.chem.atoms = result['atoms']
            self.chem.bonds = detect_bonds(self.chem.atoms)
            self.chem.viewer.set_molecule(self.chem.atoms, "ORCA paste")
            self.chem.info.update({
                'energy': result['total_energy'],
                'method': result['method'],
                'mulliken': result['mulliken']
            })
            self._update_ui()
            self.status_lbl.setText(f"\u2713 Parsed {len(self.chem.atoms)} atoms")
        else:
            self.status_lbl.setText("\u26a0 No coordinates found")


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE
# ═══════════════════════════════════════════════════════════════

main_widget = ChemLabApp()
main_widget.resize(1400, 850)

# Convenience aliases for LLM / scripting namespace
chem = main_widget.chem
mol = chem
viewer = main_widget.viewer

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

chem_proxy = graphics_scene.addWidget(main_widget)
chem_proxy.setPos(0, 0)
chem_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
chem_shadow = QGraphicsDropShadowEffect()
chem_shadow.setBlurRadius(60); chem_shadow.setOffset(45, 45); chem_shadow.setColor(QColor(0,0,0,120))
chem_proxy.setGraphicsEffect(chem_shadow)
main_widget.resize(1400, 850)

# Center in current view
_vr = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
chem_proxy.setPos(_vr.center().x() - main_widget.width() / 2,
             _vr.center().y() - main_widget.height() / 2)
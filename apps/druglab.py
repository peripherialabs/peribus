"""
DrugLab — Drug Discovery & Molecular Biology Workbench for Rio
═══════════════════════════════════════════════════════════════════

ARCHITECTURE: This file is the complete runtime. It registers a `drug`
singleton into the namespace. The LLM sees this source as context and
injects short code snippets to drive it.

EXAMPLES (what the LLM would output):
    drug.load("aspirin")                        # load preset drug molecule
    drug.load_smiles("CC(=O)Oc1ccccc1C(=O)O")  # from SMILES
    drug.load_pdb("1BNA")                       # fetch PDB structure
    drug.load_sdf("/path/to/ligand.sdf")        # load SDF/MOL file
    drug.load_xyz("C 0 0 0\\nO 1.2 0 0")       # raw coordinates

    ## DRUG PROPERTIES (computed via RDKit or built-in):
    drug.lipinski()                             # Lipinski Rule-of-Five report
    drug.descriptors()                          # MW, LogP, HBD, HBA, TPSA, RotBonds
    drug.admet()                                # ADMET property predictions
    drug.toxicity()                             # toxicity flag estimates
    drug.druglikeness()                         # composite druglikeness score

    ## VISUALIZATION:
    drug.style("ballstick")                     # ballstick/spacefill/wireframe
    drug.color_by("element")                    # element/charge/hydrophobicity/bfactor
    drug.select(i)                              # highlight atom i
    drug.measure(i, j)                          # distance
    drug.show_pharmacophore()                   # color-coded pharmacophore features
    drug.show_hbonds()                          # highlight H-bond donors/acceptors
    drug.show_surface(prop="hydrophobicity")    # molecular surface colored by property

    ## PROTEIN / DOCKING:
    drug.load_protein("2HU4")                   # fetch protein from PDB
    drug.show_binding_pocket(radius=8.0)        # show pocket around ligand
    drug.dock(ligand_smiles, protein_pdb)       # simplified docking score
    drug.overlay_contacts()                     # show protein-ligand contacts
    drug.overlay_ramachandran()                 # Ramachandran plot overlay

    ## OVERLAYS & ANALYSIS:
    drug.overlay_descriptors()                  # property radar chart
    drug.overlay_lipinski()                     # Ro5 compliance panel
    drug.overlay_logp_map()                     # atom-wise LogP contribution
    drug.compare(smiles_list)                   # side-by-side property comparison

    ## After loading, drug.info contains data:
    drug.info['name']              # molecule name
    drug.info['smiles']            # canonical SMILES
    drug.info['formula']           # molecular formula
    drug.info['mw']                # molecular weight
    drug.info['logp']              # computed LogP
    drug.info['hbd']               # H-bond donors
    drug.info['hba']               # H-bond acceptors
    drug.info['tpsa']              # topological polar surface area
    drug.info['rotatable_bonds']   # rotatable bond count
    drug.info['lipinski_pass']     # True/False Ro5 compliance
    drug.info['druglikeness']      # composite score 0-1

VIEWER API (lower level):
    drug.viewer.set_molecule(atoms, name)
    drug.viewer.set_surface(verts)             # surface mesh
    drug.viewer.set_style(style)
    drug.viewer.add_overlay(name, fn)
    drug.viewer.remove_overlay(name)
    drug.viewer.cam_dist = 12.0
    drug.viewer.rot_x, drug.viewer.rot_y
    drug.viewer.screenshot(path)

NAMESPACE: After this file runs, these are available:
    drug        — DrugLab singleton (main API)
    drug_viewer — alias for drug.viewer (the 3D GL widget)
    AMINO_ACIDS — amino acid data dict
    DRUG_PRESETS — preset drug molecules
    All PySide6/Qt, numpy, moderngl, glm from parser namespace
"""

import math
import os
import re
import subprocess
import threading
import json
import hashlib
import time
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
    QPainter, QColor, QFont, QPen, QBrush, QImage, QLinearGradient,
    QPainterPath
)

import moderngl
import glm

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

def _hex(h):
    return ((h>>16)&0xFF)/255.0, ((h>>8)&0xFF)/255.0, (h&0xFF)/255.0

def _hexq(h, alpha=255):
    return QColor((h>>16)&0xFF, (h>>8)&0xFF, h&0xFF, alpha)

# Element data (subset for drug molecules)
ELEMENTS = {
    'H':  {'color':0xEEEEEE,'r':0.31,'cov':0.31,'mass':1.008,'Z':1,'name':'Hydrogen'},
    'C':  {'color':0x333333,'r':0.76,'cov':0.77,'mass':12.01,'Z':6,'name':'Carbon'},
    'N':  {'color':0x2040D8,'r':0.71,'cov':0.71,'mass':14.01,'Z':7,'name':'Nitrogen'},
    'O':  {'color':0xDD0000,'r':0.66,'cov':0.66,'mass':16.00,'Z':8,'name':'Oxygen'},
    'F':  {'color':0x70D040,'r':0.57,'cov':0.57,'mass':19.00,'Z':9,'name':'Fluorine'},
    'P':  {'color':0xFF8000,'r':1.07,'cov':1.07,'mass':30.97,'Z':15,'name':'Phosphorus'},
    'S':  {'color':0xDDDD00,'r':1.05,'cov':1.05,'mass':32.07,'Z':16,'name':'Sulfur'},
    'Cl': {'color':0x1FF01F,'r':1.02,'cov':1.02,'mass':35.45,'Z':17,'name':'Chlorine'},
    'Br': {'color':0xA62929,'r':1.20,'cov':1.20,'mass':79.90,'Z':35,'name':'Bromine'},
    'I':  {'color':0x940094,'r':1.39,'cov':1.39,'mass':126.9,'Z':53,'name':'Iodine'},
    'Na': {'color':0xAB5CF2,'r':1.66,'cov':1.66,'mass':22.99,'Z':11,'name':'Sodium'},
    'Mg': {'color':0x8AFF00,'r':1.41,'cov':1.41,'mass':24.31,'Z':12,'name':'Magnesium'},
    'Ca': {'color':0x3DFF00,'r':1.76,'cov':1.76,'mass':40.08,'Z':20,'name':'Calcium'},
    'Fe': {'color':0xE06633,'r':1.32,'cov':1.32,'mass':55.85,'Z':26,'name':'Iron'},
    'Zn': {'color':0x7D80B0,'r':1.22,'cov':1.22,'mass':65.38,'Z':30,'name':'Zinc'},
    'Se': {'color':0xFFA100,'r':1.20,'cov':1.20,'mass':78.97,'Z':34,'name':'Selenium'},
}

Z_TO_EL = {v['Z']: k for k, v in ELEMENTS.items()}

# Amino acid data for protein visualization
AMINO_ACIDS = {
    'ALA': {'code':'A','name':'Alanine','hydrophobicity':1.8,'color':0x8CFF8C,'mw':89.1},
    'ARG': {'code':'R','name':'Arginine','hydrophobicity':-4.5,'color':0x00007C,'mw':174.2},
    'ASN': {'code':'N','name':'Asparagine','hydrophobicity':-3.5,'color':0xFF7C70,'mw':132.1},
    'ASP': {'code':'D','name':'Aspartate','hydrophobicity':-3.5,'color':0xA00042,'mw':133.1},
    'CYS': {'code':'C','name':'Cysteine','hydrophobicity':2.5,'color':0xFFFF70,'mw':121.2},
    'GLN': {'code':'Q','name':'Glutamine','hydrophobicity':-3.5,'color':0xFF4C4C,'mw':146.1},
    'GLU': {'code':'E','name':'Glutamate','hydrophobicity':-3.5,'color':0x660000,'mw':147.1},
    'GLY': {'code':'G','name':'Glycine','hydrophobicity':-0.4,'color':0xEBEBEB,'mw':75.0},
    'HIS': {'code':'H','name':'Histidine','hydrophobicity':-3.2,'color':0x7070FF,'mw':155.2},
    'ILE': {'code':'I','name':'Isoleucine','hydrophobicity':4.5,'color':0x004C00,'mw':131.2},
    'LEU': {'code':'L','name':'Leucine','hydrophobicity':3.8,'color':0x455E45,'mw':131.2},
    'LYS': {'code':'K','name':'Lysine','hydrophobicity':-3.9,'color':0x4747B8,'mw':146.2},
    'MET': {'code':'M','name':'Methionine','hydrophobicity':1.9,'color':0xB8A042,'mw':149.2},
    'PHE': {'code':'F','name':'Phenylalanine','hydrophobicity':2.8,'color':0x534C52,'mw':165.2},
    'PRO': {'code':'P','name':'Proline','hydrophobicity':-1.6,'color':0x525252,'mw':115.1},
    'SER': {'code':'S','name':'Serine','hydrophobicity':-0.8,'color':0xFF7042,'mw':105.1},
    'THR': {'code':'T','name':'Threonine','hydrophobicity':-0.7,'color':0xB84C00,'mw':119.1},
    'TRP': {'code':'W','name':'Tryptophan','hydrophobicity':-0.9,'color':0x4F4600,'mw':204.2},
    'TYR': {'code':'Y','name':'Tyrosine','hydrophobicity':-1.3,'color':0x8C704C,'mw':181.2},
    'VAL': {'code':'V','name':'Valine','hydrophobicity':4.2,'color':0x005000,'mw':117.1},
}

# ── Preset drug molecules (SMILES + known properties) ──
DRUG_PRESETS = {
    "aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "name": "Aspirin (Acetylsalicylic acid)", "mw": 180.16,
        "logp": 1.2, "hbd": 1, "hba": 4, "tpsa": 63.6, "rot": 3,
        "target": "COX-1/COX-2", "class": "NSAID",
        "atoms": [
            {"el":"C","x":0.0,"y":0.0,"z":0.0},{"el":"C","x":1.21,"y":0.70,"z":0.0},
            {"el":"O","x":1.21,"y":1.91,"z":0.0},{"el":"O","x":2.32,"y":0.0,"z":0.0},
            {"el":"C","x":3.53,"y":0.70,"z":0.0},{"el":"C","x":3.53,"y":2.10,"z":0.0},
            {"el":"C","x":4.74,"y":2.80,"z":0.0},{"el":"C","x":5.95,"y":2.10,"z":0.0},
            {"el":"C","x":5.95,"y":0.70,"z":0.0},{"el":"C","x":4.74,"y":0.0,"z":0.0},
            {"el":"C","x":2.32,"y":2.80,"z":0.0},{"el":"O","x":2.32,"y":4.01,"z":0.0},
            {"el":"O","x":1.21,"y":2.10,"z":0.5},
            {"el":"H","x":-0.51,"y":-0.51,"z":0.88},{"el":"H","x":-0.51,"y":-0.51,"z":-0.88},
            {"el":"H","x":-0.51,"y":0.88,"z":0.0},{"el":"H","x":4.74,"y":3.88,"z":0.0},
            {"el":"H","x":6.88,"y":2.63,"z":0.0},{"el":"H","x":6.88,"y":0.17,"z":0.0},
            {"el":"H","x":4.74,"y":-1.08,"z":0.0},{"el":"H","x":1.39,"y":4.30,"z":0.0},
        ],
    },
    "caffeine": {
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "name": "Caffeine (1,3,7-Trimethylxanthine)", "mw": 194.19,
        "logp": -0.07, "hbd": 0, "hba": 6, "tpsa": 58.4, "rot": 0,
        "target": "Adenosine A2A receptor", "class": "Stimulant",
        "atoms": [
            {"el":"C","x":0.0,"y":0.0,"z":0.0},{"el":"N","x":1.15,"y":0.72,"z":0.0},
            {"el":"C","x":1.15,"y":2.10,"z":0.0},{"el":"O","x":0.0,"y":2.80,"z":0.0},
            {"el":"C","x":2.40,"y":2.80,"z":0.0},{"el":"C","x":3.55,"y":2.10,"z":0.0},
            {"el":"N","x":3.55,"y":0.72,"z":0.0},{"el":"C","x":4.80,"y":0.17,"z":0.0},
            {"el":"N","x":5.65,"y":1.25,"z":0.0},{"el":"C","x":4.80,"y":2.45,"z":0.0},
            {"el":"N","x":2.40,"y":4.10,"z":0.0},{"el":"C","x":3.55,"y":4.80,"z":0.0},
            {"el":"O","x":3.55,"y":6.0,"z":0.0},{"el":"N","x":4.80,"y":4.10,"z":0.0},
            {"el":"C","x":4.80,"y":3.45,"z":0.8},
            {"el":"C","x":1.15,"y":4.80,"z":0.0},{"el":"C","x":2.40,"y":0.0,"z":0.0},
            {"el":"H","x":-0.36,"y":-0.36,"z":0.95},{"el":"H","x":-0.36,"y":-0.36,"z":-0.95},
            {"el":"H","x":-0.36,"y":0.88,"z":0.0},
        ],
    },
    "ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "name": "Ibuprofen", "mw": 206.28,
        "logp": 3.97, "hbd": 1, "hba": 2, "tpsa": 37.3, "rot": 4,
        "target": "COX-1/COX-2", "class": "NSAID",
        "atoms": [
            {"el":"C","x":0.0,"y":0.0,"z":0.0},{"el":"C","x":1.40,"y":0.0,"z":0.0},
            {"el":"C","x":2.10,"y":1.21,"z":0.0},{"el":"C","x":3.50,"y":1.21,"z":0.0},
            {"el":"C","x":4.20,"y":0.0,"z":0.0},{"el":"C","x":3.50,"y":-1.21,"z":0.0},
            {"el":"C","x":2.10,"y":-1.21,"z":0.0},{"el":"C","x":5.72,"y":0.0,"z":0.0},
            {"el":"C","x":6.42,"y":1.21,"z":0.0},{"el":"C","x":7.94,"y":1.21,"z":0.0},
            {"el":"O","x":8.64,"y":2.10,"z":0.0},{"el":"O","x":8.64,"y":0.0,"z":0.0},
            {"el":"C","x":6.42,"y":-1.21,"z":0.0},{"el":"C","x":-0.70,"y":1.21,"z":0.0},
            {"el":"C","x":-0.70,"y":-1.21,"z":0.0},
            {"el":"H","x":1.60,"y":2.15,"z":0.0},{"el":"H","x":4.04,"y":2.15,"z":0.0},
            {"el":"H","x":4.04,"y":-2.15,"z":0.0},{"el":"H","x":1.60,"y":-2.15,"z":0.0},
            {"el":"H","x":9.56,"y":0.0,"z":0.0},
        ],
    },
    "penicillin": {
        "smiles": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",
        "name": "Penicillin G (Benzylpenicillin)", "mw": 334.39,
        "logp": 1.83, "hbd": 2, "hba": 6, "tpsa": 112.0, "rot": 5,
        "target": "PBP (Penicillin-binding proteins)", "class": "Beta-lactam antibiotic",
        "atoms": [
            {"el":"C","x":0.0,"y":0.0,"z":0.0},{"el":"C","x":1.2,"y":0.7,"z":0.3},
            {"el":"C","x":2.4,"y":0.0,"z":0.0},{"el":"S","x":2.4,"y":-1.6,"z":0.0},
            {"el":"C","x":1.0,"y":-1.8,"z":0.5},{"el":"N","x":0.0,"y":-1.0,"z":0.8},
            {"el":"C","x":-1.2,"y":-1.2,"z":0.2},{"el":"O","x":-1.2,"y":-2.4,"z":-0.1},
            {"el":"C","x":-2.4,"y":-0.3,"z":0.0},{"el":"N","x":-2.4,"y":0.9,"z":0.5},
            {"el":"C","x":-3.6,"y":1.6,"z":0.2},{"el":"O","x":-3.6,"y":2.8,"z":0.4},
            {"el":"C","x":-4.8,"y":0.9,"z":-0.2},
            {"el":"C","x":-6.0,"y":1.6,"z":-0.2},{"el":"C","x":-7.2,"y":0.9,"z":0.0},
            {"el":"C","x":-7.2,"y":-0.5,"z":0.2},{"el":"C","x":-6.0,"y":-1.2,"z":0.2},
            {"el":"C","x":-4.8,"y":-0.5,"z":0.0},
            {"el":"C","x":3.4,"y":0.5,"z":-0.6},{"el":"O","x":3.4,"y":1.7,"z":-0.9},
            {"el":"O","x":4.5,"y":-0.2,"z":-0.8},{"el":"C","x":0.8,"y":-2.9,"z":1.2},
            {"el":"C","x":1.2,"y":2.0,"z":0.8},
        ],
    },
    "metformin": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "name": "Metformin", "mw": 129.16,
        "logp": -1.43, "hbd": 3, "hba": 5, "tpsa": 91.5, "rot": 2,
        "target": "AMPK / Complex I", "class": "Biguanide (Antidiabetic)",
        "atoms": [
            {"el":"C","x":0.0,"y":0.0,"z":0.0},{"el":"N","x":1.2,"y":0.7,"z":0.0},
            {"el":"C","x":2.4,"y":0.0,"z":0.0},{"el":"N","x":2.4,"y":-1.3,"z":0.0},
            {"el":"N","x":3.6,"y":0.7,"z":0.0},{"el":"C","x":4.8,"y":0.0,"z":0.0},
            {"el":"N","x":4.8,"y":-1.3,"z":0.0},{"el":"N","x":6.0,"y":0.7,"z":0.0},
            {"el":"C","x":6.0,"y":2.0,"z":0.0},{"el":"C","x":7.3,"y":0.0,"z":0.0},
            {"el":"H","x":-0.5,"y":-0.5,"z":0.87},{"el":"H","x":-0.5,"y":-0.5,"z":-0.87},
            {"el":"H","x":-0.5,"y":0.9,"z":0.0},
        ],
    },
    "paclitaxel": {
        "smiles": "CC1=C2[C@@]([C@H]3[C@H]([C@H]([C@@H]([C@]1(C(=O)[C@@H]2OC(=O)C)O)O)OC(=O)C4=CC=CC=C4)(CC5=CC=CC=C5)OC(=O)[C@H](O)C(C)(C)C)(C)CC3OC(=O)C",
        "name": "Paclitaxel (Taxol)", "mw": 853.91,
        "logp": 3.0, "hbd": 4, "hba": 14, "tpsa": 221.3, "rot": 14,
        "target": "Tubulin (microtubule stabilizer)", "class": "Taxane (Chemotherapy)",
        "atoms": [
            {"el":"C","x":i*0.8*math.cos(i*0.6),"y":i*0.8*math.sin(i*0.6),"z":math.sin(i*0.3)*0.5}
            for i in range(30)
        ] + [
            {"el":"O","x":i*0.8*math.cos(i*0.6+1),"y":i*0.8*math.sin(i*0.6+1),"z":math.cos(i*0.3)*0.5}
            for i in range(14)
        ] + [
            {"el":"N","x":3.0,"y":5.0,"z":0.0},
        ],
    },
    "atorvastatin": {
        "smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(c2ccc(F)cc2)c(c1C(=O)Nc3ccccc3)c4ccccc4",
        "name": "Atorvastatin (Lipitor)", "mw": 558.64,
        "logp": 6.36, "hbd": 4, "hba": 7, "tpsa": 111.8, "rot": 12,
        "target": "HMG-CoA reductase", "class": "Statin (Cholesterol-lowering)",
        "atoms": [
            {"el":"C","x":i*0.7*math.cos(i*0.5),"y":i*0.7*math.sin(i*0.5),"z":0.0}
            for i in range(25)
        ] + [
            {"el":"O","x":2+i*1.2,"y":3.0,"z":0.0} for i in range(5)
        ] + [
            {"el":"N","x":0.0,"y":-2.0,"z":0.0},{"el":"N","x":2.5,"y":0.0,"z":0.0},
            {"el":"F","x":-4.0,"y":1.0,"z":0.0},
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

_FRAG_SURFACE = """
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
    lines = text.strip().split('\n'); atoms = []; start = 0
    if len(lines) > 2:
        try: int(lines[0].strip()); start = 2
        except: pass
    for line in lines[start:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            el = parts[0]
            if el in ELEMENTS:
                try: atoms.append({'el':el,'x':float(parts[1]),'y':float(parts[2]),'z':float(parts[3])})
                except: pass
    return atoms

def parse_pdb(text):
    """Parse PDB format → list of atom dicts with residue info."""
    atoms = []
    residues = {}
    for line in text.split('\n'):
        if line.startswith(('ATOM  ', 'HETATM')):
            try:
                el = line[76:78].strip()
                if not el:
                    el = line[12:14].strip()
                    el = el[0] if el else 'C'
                if el not in ELEMENTS:
                    el = el[0] if el[0] in ELEMENTS else 'C'
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                resnum = int(line[22:26].strip()) if line[22:26].strip() else 0
                atomname = line[12:16].strip()
                bfactor = float(line[60:66].strip()) if line[60:66].strip() else 0.0
                is_hetatm = line.startswith('HETATM')
                atoms.append({
                    'el': el, 'x': x, 'y': y, 'z': z,
                    'resname': resname, 'chain': chain, 'resnum': resnum,
                    'atomname': atomname, 'bfactor': bfactor, 'hetatm': is_hetatm,
                })
                key = f"{chain}:{resname}{resnum}"
                if key not in residues:
                    residues[key] = {'resname': resname, 'chain': chain, 'resnum': resnum, 'atoms': []}
                residues[key]['atoms'].append(len(atoms)-1)
            except:
                pass
    return atoms, residues

def parse_sdf(text):
    """Parse SDF/MOL format → list of atom dicts."""
    lines = text.strip().split('\n')
    atoms = []
    if len(lines) < 4: return atoms
    try:
        counts = lines[3].split()
        natoms = int(counts[0])
        for i in range(4, 4 + natoms):
            parts = lines[i].split()
            el = parts[3] if len(parts) > 3 else 'C'
            if el not in ELEMENTS: el = 'C'
            atoms.append({'el': el, 'x': float(parts[0]), 'y': float(parts[1]), 'z': float(parts[2])})
    except:
        pass
    return atoms

# ═══════════════════════════════════════════════════════════════
#  DRUG PROPERTY CALCULATIONS (built-in, no RDKit dependency)
# ═══════════════════════════════════════════════════════════════

def _count_hbd(atoms, bonds):
    """Count H-bond donors: N-H, O-H, S-H groups."""
    count = 0
    for i, a in enumerate(atoms):
        if a['el'] in ('N', 'O', 'S'):
            for bi, bj in bonds:
                partner = bj if bi == i else (bi if bj == i else -1)
                if partner >= 0 and atoms[partner]['el'] == 'H':
                    count += 1; break
    return count

def _count_hba(atoms):
    """Count H-bond acceptors: N and O atoms."""
    return sum(1 for a in atoms if a['el'] in ('N', 'O'))

def _count_rotatable(bonds, atoms):
    """Estimate rotatable bonds (single bonds between non-terminal heavy atoms)."""
    count = 0
    for i, j in bonds:
        if atoms[i]['el'] == 'H' or atoms[j]['el'] == 'H': continue
        # Count heavy neighbors of each
        ni = sum(1 for bi, bj in bonds if (bi == i or bj == i) and atoms[bi if bj == i else bj]['el'] != 'H')
        nj = sum(1 for bi, bj in bonds if (bi == j or bj == j) and atoms[bi if bj == j else bj]['el'] != 'H')
        if ni >= 2 and nj >= 2: count += 1
    return count

def _compute_tpsa(atoms, bonds):
    """Topological Polar Surface Area — simplified Ertl method for N and O atoms."""
    tpsa = 0.0
    for i, a in enumerate(atoms):
        if a['el'] == 'N':
            nh = sum(1 for bi, bj in bonds if (bi == i or bj == i) and atoms[bi if bj == i else bj]['el'] == 'H')
            nbonds = sum(1 for bi, bj in bonds if bi == i or bj == i)
            if nh == 0 and nbonds <= 2: tpsa += 12.36  # =N-
            elif nh == 0 and nbonds == 3: tpsa += 3.24   # N(R3)
            elif nh == 1: tpsa += 23.85                   # NH
            elif nh == 2: tpsa += 26.02                   # NH2
            else: tpsa += 12.0
        elif a['el'] == 'O':
            nh = sum(1 for bi, bj in bonds if (bi == i or bj == i) and atoms[bi if bj == i else bj]['el'] == 'H')
            nbonds = sum(1 for bi, bj in bonds if bi == i or bj == i)
            if nh == 0 and nbonds == 1: tpsa += 17.07  # =O
            elif nh == 0 and nbonds == 2: tpsa += 9.23   # -O-
            elif nh == 1: tpsa += 20.23                    # OH
            else: tpsa += 15.0
    return tpsa

def _estimate_logp(atoms):
    """Wildman-Crippen LogP estimate from atom counts."""
    counts = {}
    for a in atoms:
        counts[a['el']] = counts.get(a['el'], 0) + 1
    nc = counts.get('C', 0)
    nn = counts.get('N', 0)
    no = counts.get('O', 0)
    nf = counts.get('F', 0)
    ncl = counts.get('Cl', 0)
    nbr = counts.get('Br', 0)
    ns = counts.get('S', 0)
    # Simplified: each C contributes ~+0.12, N ~-0.57, O ~-0.47, halogens ~+0.37
    logp = nc * 0.12 - nn * 0.57 - no * 0.47 + (nf + ncl + nbr) * 0.37 + ns * 0.0
    return round(logp, 2)

def _identify_pharmacophore(atoms, bonds):
    """Identify pharmacophore features from atom types and connectivity."""
    features = []
    for i, a in enumerate(atoms):
        if a['el'] == 'H': continue
        if a['el'] in ('N', 'O'):
            has_h = any(
                atoms[bj if bi == i else bi]['el'] == 'H'
                for bi, bj in bonds if bi == i or bj == i
            )
            if has_h:
                features.append({'idx': i, 'type': 'donor', 'pos': (a['x'], a['y'], a['z'])})
            features.append({'idx': i, 'type': 'acceptor', 'pos': (a['x'], a['y'], a['z'])})
        # Aromatic: simplified — carbon with 3 bonds to non-H
        if a['el'] == 'C':
            heavy_n = sum(1 for bi, bj in bonds
                         if (bi == i or bj == i) and atoms[bi if bj == i else bj]['el'] != 'H')
            if heavy_n == 3:
                features.append({'idx': i, 'type': 'aromatic', 'pos': (a['x'], a['y'], a['z'])})
        # Hydrophobic: C or S with no polar neighbors
        if a['el'] in ('C', 'S'):
            neighbors = [atoms[bj if bi == i else bi]['el'] for bi, bj in bonds if bi == i or bj == i]
            if all(n in ('C', 'H', 'S') for n in neighbors):
                features.append({'idx': i, 'type': 'hydrophobic', 'pos': (a['x'], a['y'], a['z'])})
        # Positive/negative charge centers (very simplified)
        if a['el'] == 'N':
            nh = sum(1 for bi, bj in bonds if (bi == i or bj == i) and atoms[bi if bj == i else bj]['el'] == 'H')
            nbonds = sum(1 for bi, bj in bonds if bi == i or bj == i)
            if nh >= 2 and nbonds >= 3:
                features.append({'idx': i, 'type': 'positive', 'pos': (a['x'], a['y'], a['z'])})
    return features

# Pharmacophore feature colors
PHARMA_COLORS = {
    'donor':       0x3399FF,  # blue
    'acceptor':    0xFF3333,  # red
    'aromatic':    0xFF9900,  # orange
    'hydrophobic': 0x33CC33,  # green
    'positive':    0x6633FF,  # purple
    'negative':    0xFF33CC,  # pink
}

# ═══════════════════════════════════════════════════════════════
#  GL VIEWER WIDGET
# ═══════════════════════════════════════════════════════════════

class MolViewer(QWidget):
    """OpenGL molecular viewer — offscreen FBO → QImage → QPainter.
    Supports molecule rendering, surface meshes, custom overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # State
        self.atoms=[]; self.bonds=[]; self.mol_name=""
        self.rot_x=0.3; self.rot_y=0.0; self.auto_rot=0.0; self.cam_dist=10.0
        self._dragging=False; self._lmx=0; self._lmy=0
        self.selected_atom=-1; self.render_style='ballstick'; self.show_bonds=True
        self._center=(0,0,0)
        self.surface_opacity=0.45
        self._color_mode = 'element'  # element/charge/hydrophobicity/bfactor/pharmacophore
        self._atom_colors = None  # override per-atom colors
        # GL
        self._gl_ready=False; self.ctx=None; self.fbo=None
        self._fbo_w=0; self._fbo_h=0; self._frame=None
        self._mol_vao=None; self._mol_n=0
        self._surf_vao=None; self._surf_n=0
        # Overlays
        self._overlays = OrderedDict()
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
        self.prog_surf=self.ctx.program(vertex_shader=_VERT_LIT, fragment_shader=_FRAG_SURFACE)
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
        self.mol_name=name; self.selected_atom=-1; self._atom_colors=None
        if atoms:
            cx=sum(a['x'] for a in atoms)/len(atoms)
            cy=sum(a['y'] for a in atoms)/len(atoms)
            cz=sum(a['z'] for a in atoms)/len(atoms)
            self._center=(cx,cy,cz)
            md=max(math.sqrt((a['x']-cx)**2+(a['y']-cy)**2+(a['z']-cz)**2) for a in atoms)
            self.cam_dist=max(md*3.2,6.0)
        self.auto_rot=0.0
        self._rebuild_mol()

    def set_surface(self, verts):
        """Set surface mesh: raw triangle verts pos(3)+normal(3)+color(3)."""
        if not self._gl_ready or not verts:
            self._surf_vao=None; self._surf_n=0; return
        data=np.array(verts,dtype='f4').tobytes()
        if self._surf_vao:
            try: self._surf_vao.release()
            except: pass
        vbo=self.ctx.buffer(data)
        self._surf_vao=self.ctx.vertex_array(self.prog_surf,[(vbo,'3f 3f 3f','in_position','in_normal','in_color')])
        self._surf_n=len(verts)//9

    def clear_surface(self):
        self._surf_vao=None; self._surf_n=0

    def add_overlay(self, name, fn):
        self._overlays[name] = fn

    def remove_overlay(self, name):
        self._overlays.pop(name, None)

    def screenshot(self, path):
        if self._frame: self._frame.save(path)

    def set_style(self, style):
        self.render_style = style
        self._rebuild_mol()

    def _get_atom_color(self, idx, atom):
        """Get color for atom based on current color mode."""
        if self._atom_colors and idx < len(self._atom_colors):
            return self._atom_colors[idx]
        return ELEMENTS.get(atom['el'], {'color': 0x888888})['color']

    def _rebuild_mol(self):
        if not self._gl_ready or not self.atoms: return
        cx,cy,cz=self._center; all_v=[]
        scale=2.2 if self.render_style=='spacefill' else 1.0
        segs=24 if len(self.atoms)>50 else 32
        if len(self.atoms) > 200: segs = 12
        if len(self.atoms) > 1000: segs = 8
        for idx,a in enumerate(self.atoms):
            el=ELEMENTS.get(a['el'],{'color':0x888888,'r':0.5})
            r=el['r']*0.4*scale if self.render_style!='wireframe' else 0.12
            # For large proteins, reduce sphere detail
            if len(self.atoms) > 500: r *= 0.5
            col=self._get_atom_color(idx, a)
            if self.selected_atom==idx:
                rr,gg,bb=_hex(col)
                col=(min(int((rr+0.3)*255),255)<<16)|(min(int((gg+0.3)*255),255)<<8)|min(int((bb+0.3)*255),255)
            sp=_make_sphere(r,segs,segs//2,col)
            t=glm.translate(glm.mat4(1),glm.vec3(a['x']-cx,a['y']-cy,a['z']-cz))
            all_v.extend(_transform_verts(sp,t))
        if self.show_bonds and self.render_style!='spacefill':
            br=0.025 if self.render_style=='wireframe' else 0.06
            if len(self.atoms) > 500: br *= 0.5
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
        if not self._dragging: self.auto_rot+=0.004
        self._render(); self.update()

    def _render(self):
        if not self._gl_ready: return
        w,h=max(self.width(),320),max(self.height(),200)
        self._resize_fbo(w,h); self.fbo.use(); self.ctx.viewport=(0,0,w,h)
        self.ctx.clear(0,0,0,0)
        proj=glm.perspective(glm.radians(45),w/h,0.1,500.0)
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
        if self._surf_vao and self._surf_n>0:
            self.ctx.disable(moderngl.CULL_FACE); self.ctx.disable(moderngl.DEPTH_TEST)
            self.prog_surf['mvp'].write(vp); self.prog_surf['model'].write(identity)
            self.prog_surf['light_dir'].write(glm.normalize(glm.vec3(0.5,0.8,0.6)))
            self.prog_surf['ambient'].write(glm.vec3(0.5,0.48,0.52))
            self.prog_surf['view_pos'].write(eye); self.prog_surf['alpha'].value=self.surface_opacity
            self._surf_vao.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.enable(moderngl.CULL_FACE)
        raw=self.fbo.color_attachments[0].read()
        self._frame=QImage(raw,w,h,w*4,QImage.Format_RGBA8888).mirrored(False,True)

    def paintEvent(self, event):
        self._ensure_gl()
        if self.atoms and self._mol_n==0: self._rebuild_mol()
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height()
        if self._frame and not self._frame.isNull(): p.drawImage(0,0,self._frame)
        # HUD
        if self.atoms:
            p.setPen(Qt.NoPen); p.setBrush(QColor(255,255,255,195))
            p.drawRoundedRect(12,12,220,76,8,8)
            p.setFont(QFont("Consolas",10,QFont.Bold)); p.setPen(QColor(40,60,90))
            name = self.mol_name[:28] if self.mol_name else "Molecule"
            p.drawText(20,22,200,16,Qt.AlignVCenter, name)
            p.setFont(QFont("Consolas",9)); p.setPen(QColor(80,90,110))
            heavy = sum(1 for a in self.atoms if a['el'] != 'H')
            counts={}
            for a in self.atoms: counts[a['el']]=counts.get(a['el'],0)+1
            keys=sorted(counts.keys(),key=lambda x:('A' if x=='C' else 'B' if x=='H' else x))
            formula=''.join(f"{k}{counts[k] if counts[k]>1 else ''}" for k in keys)
            p.drawText(20,38,200,14,Qt.AlignVCenter,f"{len(self.atoms)} atoms ({heavy} heavy) · {formula}")
            mass=sum(ELEMENTS.get(a['el'],{'mass':0})['mass'] for a in self.atoms)
            p.drawText(20,52,200,14,Qt.AlignVCenter,f"MW: {mass:.2f} g/mol")
            # Show residue info if protein
            resnames = set(a.get('resname','') for a in self.atoms if a.get('resname'))
            if resnames:
                nres = len(set(f"{a.get('chain','')}{a.get('resnum',0)}" for a in self.atoms if a.get('resname')))
                p.drawText(20,66,200,14,Qt.AlignVCenter,f"{nres} residues")
        # Selected atom
        if 0<=self.selected_atom<len(self.atoms):
            a=self.atoms[self.selected_atom]
            el=ELEMENTS.get(a['el'],{'name':a['el'],'Z':'?','mass':0})
            p.setPen(Qt.NoPen); p.setBrush(QColor(255,255,255,210))
            bh = 90 if a.get('resname') else 75
            p.drawRoundedRect(w-195,12,183,bh,8,8)
            p.setFont(QFont("Consolas",10,QFont.Bold)); p.setPen(QColor(40,45,60))
            p.drawText(w-185,18,170,16,Qt.AlignVCenter,f"{el['name']} #{self.selected_atom}")
            p.setFont(QFont("Consolas",9)); p.setPen(QColor(90,100,120))
            p.drawText(w-185,36,170,13,Qt.AlignVCenter,f"{a['el']} Z={el['Z']}  {el['mass']:.3f} u")
            p.drawText(w-185,50,170,13,Qt.AlignVCenter,f"({a['x']:+.3f}, {a['y']:+.3f}, {a['z']:+.3f})")
            if a.get('resname'):
                p.drawText(w-185,64,170,13,Qt.AlignVCenter,
                          f"{a.get('resname','')} {a.get('resnum','')} {a.get('chain','')} · {a.get('atomname','')}")
                if a.get('bfactor'):
                    p.drawText(w-185,78,170,13,Qt.AlignVCenter,f"B-factor: {a['bfactor']:.1f}")
        # Measurements
        p.setFont(QFont("Consolas",9)); p.setPen(QColor(40,100,160))
        my=h-60
        for mtype,idxs,val in self._measurements:
            if mtype=='dist': p.drawText(12,my,220,14,Qt.AlignVCenter,f"d({idxs[0]}-{idxs[1]}) = {val:.4f} \u00c5")
            elif mtype=='angle': p.drawText(12,my,260,14,Qt.AlignVCenter,f"\u2220({idxs[0]}-{idxs[1]}-{idxs[2]}) = {val:.1f}\u00b0")
            my-=16
        # Custom overlays
        for name, fn in self._overlays.items():
            try: fn(p, w, h)
            except: pass
        p.end()

    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton: self._dragging=True; self._lmx=e.x(); self._lmy=e.y()
        e.accept()
    def mouseReleaseEvent(self,e): self._dragging=False; e.accept()
    def mouseMoveEvent(self,e):
        if self._dragging:
            self.rot_y+=(e.x()-self._lmx)*0.008
            self.rot_x=max(-1.4,min(1.4,self.rot_x+(e.y()-self._lmy)*0.008))
            self._lmx=e.x(); self._lmy=e.y()
        e.accept()
    def wheelEvent(self,e):
        self.cam_dist=max(2,min(200,self.cam_dist-e.angleDelta().y()*0.01)); e.accept()

# ═══════════════════════════════════════════════════════════════
#  DRUGLAB — MAIN API SINGLETON
# ═══════════════════════════════════════════════════════════════

class _Signals(QObject):
    status = Signal(str)
    fetch_done = Signal(str)

class DrugLab:
    """
    Main drug discovery API. Registered as `drug` in the namespace.

    QUICK REFERENCE (for LLM context):
        drug.load(name)                    — load preset: aspirin/caffeine/ibuprofen/penicillin/metformin/paclitaxel/atorvastatin
        drug.load_smiles(smiles)           — load from SMILES (needs obabel)
        drug.load_pdb(code_or_path)        — fetch PDB by code or load file
        drug.load_sdf(path)                — load SDF/MOL file
        drug.load_xyz(text)                — load from XYZ text
        drug.load_atoms(atoms)             — load atom list directly

        drug.style(s)                      — ballstick/spacefill/wireframe
        drug.color_by(mode)                — element/hydrophobicity/bfactor/pharmacophore
        drug.select(i)                     — highlight atom i
        drug.measure(i, j)                 — distance Å
        drug.angle(i, j, k)               — angle degrees
        drug.clear_measurements()          — clear all

        drug.lipinski()                    — Ro5 analysis
        drug.descriptors()                 — full descriptor dict
        drug.admet()                       — ADMET predictions
        drug.druglikeness()                — composite score 0-1
        drug.toxicity()                    — toxicity flags

        drug.show_pharmacophore()          — color by pharmacophore features
        drug.show_hbonds()                 — highlight H-bond sites
        drug.overlay_lipinski()            — Ro5 compliance panel overlay
        drug.overlay_descriptors()         — property radar chart overlay
        drug.overlay_admet()               — ADMET panel overlay
        drug.overlay_contacts(radius)      — show intermolecular contacts
        drug.overlay_ramachandran()        — Ramachandran plot (for proteins)

        drug.compare([smiles_list])        — multi-molecule comparison overlay

        drug.atoms                         — current atom list
        drug.bonds                         — current bond list
        drug.info                          — properties dict
        drug.residues                      — residue dict (for proteins)
        drug.pharmacophore                 — pharmacophore feature list
        drug.viewer                        — MolViewer widget
        drug.log(msg)                      — append to log
    """

    def __init__(self, viewer, log_widget=None):
        self.viewer = viewer
        self._log = log_widget
        self.atoms = []
        self.bonds = []
        self.info = {}
        self.residues = {}
        self.pharmacophore = []
        self._signals = _Signals()
        self._protein_atoms = []  # separate protein structure
        self._ligand_atoms = []   # separate ligand

    def log(self, msg):
        if self._log: self._log.append(f"[drug] {msg}")
        print(f"[drug] {msg}")

    # ── Loading ────────────────────────────────────────────────

    def load(self, name):
        """Load a preset drug molecule by name."""
        key = name.lower().replace(' ','').replace('-','')
        for k, data in DRUG_PRESETS.items():
            if key in k or k in key:
                self.atoms = [dict(a) for a in data['atoms']]
                self.bonds = detect_bonds(self.atoms)
                self.viewer.set_molecule(self.atoms, data.get('name', name))
                self.info = {
                    'name': data.get('name', name),
                    'smiles': data.get('smiles', ''),
                    'source': 'preset',
                    'mw': data.get('mw', self.mass()),
                    'logp': data.get('logp', _estimate_logp(self.atoms)),
                    'hbd': data.get('hbd', _count_hbd(self.atoms, self.bonds)),
                    'hba': data.get('hba', _count_hba(self.atoms)),
                    'tpsa': data.get('tpsa', _compute_tpsa(self.atoms, self.bonds)),
                    'rotatable_bonds': data.get('rot', _count_rotatable(self.bonds, self.atoms)),
                    'target': data.get('target', ''),
                    'drug_class': data.get('class', ''),
                }
                self.info['lipinski_pass'] = self._check_lipinski()
                self.info['druglikeness'] = self._compute_druglikeness()
                self.pharmacophore = _identify_pharmacophore(self.atoms, self.bonds)
                self.log(f"Loaded: {data.get('name', name)} ({len(self.atoms)} atoms)")
                return self
        self.log(f"Unknown preset: {name}. Available: {', '.join(DRUG_PRESETS.keys())}")
        return self

    def load_smiles(self, smiles):
        """Load from SMILES string (requires obabel in PATH)."""
        try:
            result = subprocess.run(['obabel', '-:'+smiles, '-oxyz', '--gen3d'],
                                     capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                self.atoms = parse_xyz(result.stdout)
                self.bonds = detect_bonds(self.atoms)
                self.viewer.set_molecule(self.atoms, f"SMILES: {smiles[:30]}")
                self._compute_all_properties(smiles=smiles)
                self.log(f"Loaded SMILES: {smiles[:40]} ({len(self.atoms)} atoms)")
            else:
                self.log(f"obabel failed: {result.stderr[:100]}")
        except FileNotFoundError:
            self.log("obabel not found. Install: apt install openbabel")
        except Exception as ex:
            self.log(f"SMILES error: {ex}")
        return self

    def load_pdb(self, code_or_path):
        """Load protein/molecule from PDB code (fetches from RCSB) or file path."""
        if os.path.isfile(os.path.expanduser(code_or_path)):
            path = os.path.expanduser(code_or_path)
            with open(path, 'r') as f: text = f.read()
            self.atoms, self.residues = parse_pdb(text)
            self.bonds = detect_bonds(self.atoms)
            self.viewer.set_molecule(self.atoms, os.path.basename(path))
            self.info = {'name': os.path.basename(path), 'source': 'pdb_file'}
            self.log(f"Loaded PDB file: {path} ({len(self.atoms)} atoms)")
        else:
            code = code_or_path.strip().upper()
            self.log(f"Fetching PDB: {code} from RCSB...")
            try:
                import urllib.request
                url = f"https://files.rcsb.org/download/{code}.pdb"
                req = urllib.request.Request(url, headers={'User-Agent': 'DrugLab/1.0'})
                resp = urllib.request.urlopen(req, timeout=30)
                text = resp.read().decode('utf-8')
                self.atoms, self.residues = parse_pdb(text)
                self.bonds = detect_bonds(self.atoms)
                self.viewer.set_molecule(self.atoms, f"PDB: {code}")
                self.info = {'name': f"PDB: {code}", 'source': 'rcsb', 'pdb_code': code}
                # Save locally
                cache_path = os.path.expanduser(f"~/{code}.pdb")
                with open(cache_path, 'w') as f: f.write(text)
                self.log(f"Fetched PDB {code}: {len(self.atoms)} atoms, {len(self.residues)} residues")
            except Exception as ex:
                self.log(f"PDB fetch failed: {ex}")
        return self

    def load_sdf(self, path):
        """Load from SDF/MOL file."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.log(f"File not found: {path}"); return self
        with open(path, 'r') as f: text = f.read()
        self.atoms = parse_sdf(text)
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, os.path.basename(path))
        self._compute_all_properties()
        self.log(f"Loaded SDF: {os.path.basename(path)} ({len(self.atoms)} atoms)")
        return self

    def load_xyz(self, text):
        """Load molecule from XYZ format text."""
        self.atoms = parse_xyz(text)
        if not self.atoms:
            self.log("No atoms found in XYZ text"); return self
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, "XYZ input")
        self._compute_all_properties()
        self.log(f"Loaded {len(self.atoms)} atoms from XYZ")
        return self

    def load_atoms(self, atoms):
        """Load from a list of atom dicts: [{'el':'C','x':0,'y':0,'z':0}, ...]"""
        self.atoms = [dict(a) for a in atoms]
        self.bonds = detect_bonds(self.atoms)
        self.viewer.set_molecule(self.atoms, "Custom")
        self._compute_all_properties()
        return self

    # ── Property computation ───────────────────────────────────

    def _compute_all_properties(self, smiles=''):
        """Compute all drug properties from current atoms."""
        mw = self.mass()
        logp = _estimate_logp(self.atoms)
        hbd = _count_hbd(self.atoms, self.bonds)
        hba = _count_hba(self.atoms)
        tpsa = _compute_tpsa(self.atoms, self.bonds)
        rot = _count_rotatable(self.bonds, self.atoms)
        self.info.update({
            'mw': mw, 'logp': logp, 'hbd': hbd, 'hba': hba,
            'tpsa': tpsa, 'rotatable_bonds': rot, 'formula': self.formula(),
        })
        if smiles: self.info['smiles'] = smiles
        self.info['lipinski_pass'] = self._check_lipinski()
        self.info['druglikeness'] = self._compute_druglikeness()
        self.pharmacophore = _identify_pharmacophore(self.atoms, self.bonds)

    def _check_lipinski(self):
        """Check Lipinski Rule of Five."""
        violations = 0
        if self.info.get('mw', 0) > 500: violations += 1
        if self.info.get('logp', 0) > 5: violations += 1
        if self.info.get('hbd', 0) > 5: violations += 1
        if self.info.get('hba', 0) > 10: violations += 1
        return violations <= 1

    def _compute_druglikeness(self):
        """Composite druglikeness score 0-1."""
        score = 1.0
        mw = self.info.get('mw', 0)
        logp = self.info.get('logp', 0)
        hbd = self.info.get('hbd', 0)
        hba = self.info.get('hba', 0)
        tpsa = self.info.get('tpsa', 0)
        rot = self.info.get('rotatable_bonds', 0)
        # Penalize for being outside ideal ranges
        if mw > 500: score -= min(0.3, (mw - 500) / 1000)
        if mw < 150: score -= 0.1
        if logp > 5: score -= min(0.3, (logp - 5) / 5)
        if logp < -2: score -= 0.1
        if hbd > 5: score -= min(0.2, (hbd - 5) / 10)
        if hba > 10: score -= min(0.2, (hba - 10) / 10)
        if tpsa > 140: score -= min(0.2, (tpsa - 140) / 200)
        if rot > 10: score -= min(0.2, (rot - 10) / 10)
        return max(0.0, min(1.0, score))

    # ── Public property methods ────────────────────────────────

    def lipinski(self):
        """Return Lipinski Rule-of-Five analysis dict."""
        mw = self.info.get('mw', self.mass())
        logp = self.info.get('logp', _estimate_logp(self.atoms))
        hbd = self.info.get('hbd', _count_hbd(self.atoms, self.bonds))
        hba = self.info.get('hba', _count_hba(self.atoms))
        result = {
            'mw': {'value': mw, 'limit': 500, 'pass': mw <= 500},
            'logp': {'value': logp, 'limit': 5, 'pass': logp <= 5},
            'hbd': {'value': hbd, 'limit': 5, 'pass': hbd <= 5},
            'hba': {'value': hba, 'limit': 10, 'pass': hba <= 10},
        }
        violations = sum(1 for v in result.values() if not v['pass'])
        result['violations'] = violations
        result['pass'] = violations <= 1
        self.log(f"Lipinski: {violations} violations → {'PASS' if result['pass'] else 'FAIL'}")
        return result

    def descriptors(self):
        """Return full molecular descriptor dictionary."""
        d = {
            'formula': self.formula(),
            'mw': self.info.get('mw', self.mass()),
            'logp': self.info.get('logp', _estimate_logp(self.atoms)),
            'hbd': self.info.get('hbd', _count_hbd(self.atoms, self.bonds)),
            'hba': self.info.get('hba', _count_hba(self.atoms)),
            'tpsa': self.info.get('tpsa', _compute_tpsa(self.atoms, self.bonds)),
            'rotatable_bonds': self.info.get('rotatable_bonds', _count_rotatable(self.bonds, self.atoms)),
            'heavy_atoms': sum(1 for a in self.atoms if a['el'] != 'H'),
            'num_atoms': len(self.atoms),
            'num_bonds': len(self.bonds),
            'rings_approx': max(0, len(self.bonds) - len(self.atoms) + 1),  # Euler
            'druglikeness': self.info.get('druglikeness', self._compute_druglikeness()),
        }
        self.log(f"Descriptors: MW={d['mw']:.1f} LogP={d['logp']:.2f} TPSA={d['tpsa']:.1f}")
        return d

    def admet(self):
        """Estimate ADMET properties (simplified heuristic model)."""
        mw = self.info.get('mw', self.mass())
        logp = self.info.get('logp', _estimate_logp(self.atoms))
        tpsa = self.info.get('tpsa', 60)
        hbd = self.info.get('hbd', 0)
        result = {
            'absorption': {
                'oral_bioavailability': 'likely' if (mw < 500 and logp < 5 and tpsa < 140) else 'poor',
                'caco2_permeability': 'high' if logp > 0 and tpsa < 90 else 'low',
                'intestinal_absorption': 'good' if tpsa < 140 else 'limited',
            },
            'distribution': {
                'bbb_penetration': 'yes' if (tpsa < 90 and mw < 400 and logp > 0) else 'no',
                'plasma_protein_binding': 'high' if logp > 3 else ('moderate' if logp > 1 else 'low'),
                'vd_estimate': 'high' if logp > 3 else 'moderate',
            },
            'metabolism': {
                'cyp_substrate_risk': 'high' if logp > 4 else 'moderate',
                'metabolic_stability': 'stable' if mw < 400 else 'moderate',
            },
            'excretion': {
                'route': 'renal' if logp < 1 and mw < 300 else 'hepatic',
                'half_life_estimate': 'short' if mw < 250 else ('medium' if mw < 500 else 'long'),
            },
            'toxicity': self.toxicity(),
        }
        self.log(f"ADMET: oral={'likely' if result['absorption']['oral_bioavailability']=='likely' else 'poor'} "
                 f"BBB={'yes' if result['distribution']['bbb_penetration']=='yes' else 'no'}")
        return result

    def toxicity(self):
        """Estimate toxicity flags (simplified rule-based)."""
        mw = self.info.get('mw', self.mass())
        logp = self.info.get('logp', _estimate_logp(self.atoms))
        tpsa = self.info.get('tpsa', 60)
        flags = {}
        # Structural alerts (simplified)
        elem_counts = {}
        for a in self.atoms: elem_counts[a['el']] = elem_counts.get(a['el'], 0) + 1
        flags['herg_risk'] = 'moderate' if logp > 3 and mw > 350 else 'low'
        flags['hepatotoxicity'] = 'flag' if logp > 5 else 'ok'
        flags['mutagenicity'] = 'check' if elem_counts.get('N', 0) > 4 else 'likely_ok'
        flags['reactive_metabolites'] = 'flag' if elem_counts.get('S', 0) > 1 else 'ok'
        flags['overall'] = 'concern' if any(v in ('flag', 'moderate') for v in flags.values()) else 'acceptable'
        return flags

    def druglikeness(self):
        """Return composite druglikeness score 0-1."""
        score = self._compute_druglikeness()
        self.log(f"Druglikeness score: {score:.3f}")
        return score

    # ── Visualization ──────────────────────────────────────────

    def style(self, s):
        """Set render style: 'ballstick', 'spacefill', or 'wireframe'."""
        self.viewer.set_style(s)
        return self

    def color_by(self, mode):
        """Color atoms by: 'element', 'hydrophobicity', 'bfactor', 'pharmacophore'."""
        self.viewer._color_mode = mode
        if mode == 'element':
            self.viewer._atom_colors = None
        elif mode == 'hydrophobicity':
            colors = []
            for a in self.atoms:
                resname = a.get('resname', '')
                if resname in AMINO_ACIDS:
                    h = AMINO_ACIDS[resname]['hydrophobicity']
                    t = (h + 4.5) / 9.0  # -4.5..+4.5 → 0..1
                    r = int(min(255, t * 510))
                    b = int(min(255, (1-t) * 510))
                    colors.append((r << 16) | (50 << 8) | b)
                else:
                    # Use element-based hydrophobicity estimate
                    if a['el'] in ('C', 'S'): colors.append(0xDD8800)
                    elif a['el'] in ('N', 'O'): colors.append(0x3388DD)
                    else: colors.append(0x888888)
            self.viewer._atom_colors = colors
        elif mode == 'bfactor':
            bfactors = [a.get('bfactor', 0.0) for a in self.atoms]
            bmax = max(bfactors) if bfactors else 1.0
            if bmax < 0.01: bmax = 1.0
            colors = []
            for bf in bfactors:
                t = min(1.0, bf / bmax)
                r = int(t * 255)
                b = int((1-t) * 255)
                colors.append((r << 16) | (50 << 8) | b)
            self.viewer._atom_colors = colors
        elif mode == 'pharmacophore':
            self.show_pharmacophore()
            return self
        self.viewer._rebuild_mol()
        self.log(f"Color mode: {mode}")
        return self

    def select(self, i):
        """Highlight atom by index."""
        self.viewer.selected_atom = i
        self.viewer._rebuild_mol()
        return self

    def measure(self, i, j):
        """Measure distance between atoms i and j (Angstrom)."""
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

    def clear_measurements(self):
        self.viewer._measurements.clear(); return self

    # ── Pharmacophore ──────────────────────────────────────────

    def show_pharmacophore(self):
        """Color atoms by pharmacophore feature type."""
        if not self.pharmacophore:
            self.pharmacophore = _identify_pharmacophore(self.atoms, self.bonds)
        # Build color map
        colors = [0x888888] * len(self.atoms)  # default gray
        for feat in self.pharmacophore:
            idx = feat['idx']
            col = PHARMA_COLORS.get(feat['type'], 0x888888)
            colors[idx] = col
        self.viewer._atom_colors = colors
        self.viewer._rebuild_mol()
        # Add legend overlay
        def draw_legend(painter, w, h):
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,200))
            painter.drawRoundedRect(w-170, h-145, 160, 135, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(50,60,80))
            painter.drawText(w-160, h-140, 140, 14, Qt.AlignVCenter, "Pharmacophore")
            y = h - 122
            for fname, fcol in PHARMA_COLORS.items():
                painter.setPen(Qt.NoPen)
                painter.setBrush(_hexq(fcol))
                painter.drawEllipse(w-158, y+2, 10, 10)
                painter.setPen(QColor(60,70,90))
                painter.setFont(QFont("Consolas", 8))
                painter.drawText(w-142, y, 130, 14, Qt.AlignVCenter, fname.capitalize())
                y += 18
        self.viewer.add_overlay('pharmacophore_legend', draw_legend)
        self.log(f"Pharmacophore: {len(self.pharmacophore)} features")
        return self

    def show_hbonds(self):
        """Highlight H-bond donor and acceptor atoms."""
        colors = [ELEMENTS.get(a['el'], {'color': 0x888888})['color'] for a in self.atoms]
        for i, a in enumerate(self.atoms):
            if a['el'] in ('N', 'O'):
                has_h = any(
                    self.atoms[bj if bi == i else bi]['el'] == 'H'
                    for bi, bj in self.bonds if bi == i or bj == i
                )
                if has_h:
                    colors[i] = 0x3399FF  # donor = blue
                else:
                    colors[i] = 0xFF3333  # acceptor = red
        self.viewer._atom_colors = colors
        self.viewer._rebuild_mol()
        self.log("Showing H-bond donors (blue) and acceptors (red)")
        return self

    # ── Overlays ───────────────────────────────────────────────

    def overlay_lipinski(self, width=280, height=150):
        """Show Lipinski Rule-of-Five compliance panel as overlay."""
        ro5 = self.lipinski()
        def draw_ro5(painter, w, h):
            ox, oy = w - width - 16, 100
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,220))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 9, QFont.Bold)); painter.setPen(QColor(40,55,80))
            status = "\u2705 PASS" if ro5['pass'] else f"\u274c FAIL ({ro5['violations']} violations)"
            painter.drawText(ox+10, oy+6, width-20, 16, Qt.AlignVCenter, f"Lipinski Ro5: {status}")
            painter.setFont(QFont("Consolas", 8)); y = oy + 28
            for key in ('mw', 'logp', 'hbd', 'hba'):
                entry = ro5[key]
                icon = "\u2705" if entry['pass'] else "\u274c"
                label = {'mw': 'MW', 'logp': 'LogP', 'hbd': 'HBD', 'hba': 'HBA'}[key]
                painter.setPen(QColor(50,60,80))
                painter.drawText(ox+10, y, width-20, 14, Qt.AlignVCenter,
                    f"{icon} {label}: {entry['value']:.1f}  (limit: {entry['limit']})")
                # Progress bar
                frac = min(1.0, entry['value'] / entry['limit'])
                bar_w = 80; bar_x = ox + width - bar_w - 14
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(230,235,240))
                painter.drawRoundedRect(bar_x, y+2, bar_w, 10, 3, 3)
                col = QColor(80,180,100) if entry['pass'] else QColor(220,80,80)
                painter.setBrush(col)
                painter.drawRoundedRect(bar_x, y+2, int(bar_w*frac), 10, 3, 3)
                y += 22
            # Druglikeness bar
            dl = self.info.get('druglikeness', 0)
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(40,55,80))
            painter.drawText(ox+10, y+4, width-20, 14, Qt.AlignVCenter,
                            f"Druglikeness: {dl:.2f}")
            bar_x = ox + width - 94; bar_w = 80
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(230,235,240))
            painter.drawRoundedRect(bar_x, y+6, bar_w, 10, 3, 3)
            col = QColor(80, int(180*dl), int(220*(1-dl)))
            painter.setBrush(col)
            painter.drawRoundedRect(bar_x, y+6, int(bar_w*dl), 10, 3, 3)
        self.viewer.add_overlay('lipinski', draw_ro5)
        return self

    def overlay_descriptors(self, width=220, height=220):
        """Show molecular descriptors as radar chart overlay."""
        desc = self.descriptors()
        # Normalize to 0-1 for radar
        axes = [
            ('MW', desc['mw'] / 500, desc['mw']),
            ('LogP', (desc['logp'] + 2) / 7, desc['logp']),
            ('HBD', desc['hbd'] / 5, desc['hbd']),
            ('HBA', desc['hba'] / 10, desc['hba']),
            ('TPSA', desc['tpsa'] / 140, desc['tpsa']),
            ('RotB', desc['rotatable_bonds'] / 10, desc['rotatable_bonds']),
        ]
        n = len(axes)
        def draw_radar(painter, w, h):
            ox, oy = 16, h - height - 16
            cx, cy = ox + width // 2, oy + height // 2 + 10
            r_max = min(width, height) // 2 - 25
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,210))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(40,55,80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, "Descriptor Profile")
            # Draw rings
            for ring in (0.25, 0.5, 0.75, 1.0):
                painter.setPen(QPen(QColor(200,210,220), 0.5))
                rr = int(r_max * ring)
                painter.drawEllipse(cx-rr, cy-rr, 2*rr, 2*rr)
            # Draw axes and fill polygon
            pts = []
            for i, (name, norm_val, raw_val) in enumerate(axes):
                angle = -math.pi/2 + 2*math.pi*i/n
                # Axis line
                ex, ey = cx + r_max*math.cos(angle), cy + r_max*math.sin(angle)
                painter.setPen(QPen(QColor(180,190,200), 0.8))
                painter.drawLine(int(cx), int(cy), int(ex), int(ey))
                # Label
                lx, ly = cx + (r_max+14)*math.cos(angle)-16, cy + (r_max+14)*math.sin(angle)-6
                painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(70,80,100))
                painter.drawText(int(lx), int(ly), 50, 12, Qt.AlignCenter, f"{name}\n{raw_val:.1f}")
                # Point
                val = max(0, min(1, norm_val))
                px, py = cx + r_max*val*math.cos(angle), cy + r_max*val*math.sin(angle)
                pts.append((px, py))
            # Fill polygon
            if pts:
                path = QPainterPath()
                path.moveTo(pts[0][0], pts[0][1])
                for px, py in pts[1:]: path.lineTo(px, py)
                path.closeSubpath()
                painter.setPen(QPen(QColor(60,130,200,180), 1.5))
                painter.setBrush(QColor(60,130,200,60))
                painter.drawPath(path)
                # Draw points
                for px, py in pts:
                    painter.setPen(Qt.NoPen); painter.setBrush(QColor(60,130,200))
                    painter.drawEllipse(int(px)-3, int(py)-3, 6, 6)
        self.viewer.add_overlay('descriptors', draw_radar)
        return self

    def overlay_admet(self, width=260, height=200):
        """Show ADMET prediction panel as overlay."""
        admet = self.admet()
        def draw_admet(painter, w, h):
            ox, oy = w - width - 16, h - height - 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,220))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 9, QFont.Bold)); painter.setPen(QColor(40,55,80))
            painter.drawText(ox+10, oy+6, width-20, 14, Qt.AlignVCenter, "ADMET Predictions")
            y = oy + 26
            painter.setFont(QFont("Consolas", 8))
            sections = [
                ("Absorption", [
                    f"Oral: {admet['absorption']['oral_bioavailability']}",
                    f"Caco-2: {admet['absorption']['caco2_permeability']}",
                ]),
                ("Distribution", [
                    f"BBB: {admet['distribution']['bbb_penetration']}",
                    f"PPB: {admet['distribution']['plasma_protein_binding']}",
                ]),
                ("Metabolism", [
                    f"CYP risk: {admet['metabolism']['cyp_substrate_risk']}",
                ]),
                ("Toxicity", [
                    f"hERG: {admet['toxicity']['herg_risk']}",
                    f"Hepato: {admet['toxicity']['hepatotoxicity']}",
                ]),
            ]
            for section_name, items in sections:
                painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(60,90,140))
                painter.drawText(ox+10, y, width-20, 13, Qt.AlignVCenter, section_name)
                y += 15
                painter.setFont(QFont("Consolas", 8)); painter.setPen(QColor(60,70,85))
                for item in items:
                    # Color code the value
                    val = item.split(': ')[1] if ': ' in item else ''
                    c = QColor(60,150,60) if val in ('likely','high','yes','ok','stable','low','likely_ok','acceptable') \
                        else QColor(200,120,40) if val in ('moderate','medium','check') \
                        else QColor(200,60,60)
                    painter.setPen(c)
                    painter.drawText(ox+18, y, width-28, 13, Qt.AlignVCenter, item)
                    y += 14
                y += 4
        self.viewer.add_overlay('admet', draw_admet)
        return self

    def overlay_contacts(self, radius=4.0):
        """Show close contacts between different chains/residue groups."""
        contacts = []
        for i in range(len(self.atoms)):
            for j in range(i+1, len(self.atoms)):
                a, b = self.atoms[i], self.atoms[j]
                if a['el'] == 'H' or b['el'] == 'H': continue
                # Different residue check
                ri = a.get('resnum', 0)
                rj = b.get('resnum', 0)
                ci = a.get('chain', '')
                cj = b.get('chain', '')
                if ri == rj and ci == cj: continue
                d = math.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2+(a['z']-b['z'])**2)
                if d < radius:
                    ctype = 'hbond' if (a['el'] in ('N','O') and b['el'] in ('N','O') and d < 3.5) \
                            else 'hydrophobic' if (a['el'] == 'C' and b['el'] == 'C' and d < 4.0) \
                            else 'vdw'
                    contacts.append({'i': i, 'j': j, 'dist': d, 'type': ctype})
        def draw_contacts(painter, w, h):
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,200))
            painter.drawRoundedRect(12, h-80, 200, 70, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(40,55,80))
            painter.drawText(20, h-76, 180, 14, Qt.AlignVCenter, "Intermolecular Contacts")
            painter.setFont(QFont("Consolas", 8)); painter.setPen(QColor(60,70,85))
            hb = sum(1 for c in contacts if c['type'] == 'hbond')
            hp = sum(1 for c in contacts if c['type'] == 'hydrophobic')
            vw = sum(1 for c in contacts if c['type'] == 'vdw')
            painter.drawText(20, h-58, 180, 13, Qt.AlignVCenter, f"H-bonds: {hb}")
            painter.drawText(20, h-44, 180, 13, Qt.AlignVCenter, f"Hydrophobic: {hp}")
            painter.drawText(20, h-30, 180, 13, Qt.AlignVCenter, f"van der Waals: {vw}")
        self.viewer.add_overlay('contacts', draw_contacts)
        self.log(f"Contacts within {radius}\u00c5: {len(contacts)} found")
        return self

    def overlay_ramachandran(self, width=200, height=200):
        """Show Ramachandran plot for protein residues (phi/psi angles)."""
        # Collect backbone atoms per residue
        phi_psi = []
        ca_atoms = {}
        for i, a in enumerate(self.atoms):
            if a.get('atomname') in ('N', 'CA', 'C'):
                key = (a.get('chain',''), a.get('resnum',0))
                if key not in ca_atoms: ca_atoms[key] = {}
                ca_atoms[key][a['atomname']] = i
        # Compute phi/psi for consecutive residues
        sorted_keys = sorted(ca_atoms.keys())
        for ki in range(1, len(sorted_keys)-1):
            prev, curr, nxt = sorted_keys[ki-1], sorted_keys[ki], sorted_keys[ki+1]
            if prev[0] != curr[0] or curr[0] != nxt[0]: continue  # same chain
            try:
                # phi: C(i-1)-N(i)-CA(i)-C(i)
                c_prev = ca_atoms[prev].get('C')
                n_curr = ca_atoms[curr].get('N')
                ca_curr = ca_atoms[curr].get('CA')
                c_curr = ca_atoms[curr].get('C')
                n_nxt = ca_atoms[nxt].get('N')
                if None in (c_prev, n_curr, ca_curr, c_curr, n_nxt): continue
                def _dihedral(i1,i2,i3,i4):
                    p = [np.array([self.atoms[x]['x'],self.atoms[x]['y'],self.atoms[x]['z']]) for x in (i1,i2,i3,i4)]
                    b1=p[1]-p[0]; b2=p[2]-p[1]; b3=p[3]-p[2]
                    n1=np.cross(b1,b2); n2=np.cross(b2,b3)
                    m1=np.cross(n1,b2/(np.linalg.norm(b2)+1e-10))
                    return math.degrees(math.atan2(np.dot(m1,n2),np.dot(n1,n2)))
                phi = _dihedral(c_prev, n_curr, ca_curr, c_curr)
                psi = _dihedral(n_curr, ca_curr, c_curr, n_nxt)
                phi_psi.append((phi, psi, curr[1]))
            except:
                pass
        if not phi_psi:
            self.log("No backbone dihedral angles found"); return self
        def draw_rama(painter, w, h):
            ox, oy = 16, 100
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,220))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(40,55,80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, "Ramachandran Plot")
            # Plot area
            px, py = ox+30, oy+24
            pw, ph = width-45, height-40
            painter.setPen(QPen(QColor(200,210,220),0.5)); painter.setBrush(QColor(245,248,252))
            painter.drawRect(px, py, pw, ph)
            # Grid lines at 0
            cx = px + pw/2; cy = py + ph/2
            painter.setPen(QPen(QColor(180,190,200),0.5, Qt.DashLine))
            painter.drawLine(int(cx), py, int(cx), py+ph)
            painter.drawLine(px, int(cy), px+pw, int(cy))
            # Points
            for phi, psi, resnum in phi_psi:
                sx = px + (phi + 180) / 360 * pw
                sy = py + (180 - psi - 0) / 360 * ph  # flip y
                # Color by region (helix vs sheet)
                if -160 < phi < -20 and -80 < psi < 0:
                    col = QColor(60, 130, 200, 180)  # helix region
                elif -180 < phi < -40 and 50 < psi < 180:
                    col = QColor(200, 100, 50, 180)  # sheet region
                else:
                    col = QColor(100, 100, 100, 150)
                painter.setPen(Qt.NoPen); painter.setBrush(col)
                painter.drawEllipse(int(sx)-2, int(sy)-2, 5, 5)
            # Axis labels
            painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(100,110,130))
            painter.drawText(px, py+ph+2, pw, 12, Qt.AlignCenter, "\u03c6 (-180\u00b0 to 180\u00b0)")
            painter.save(); painter.translate(ox+8, py+ph//2+30)
            painter.rotate(-90)
            painter.drawText(0, 0, 60, 12, Qt.AlignCenter, "\u03c8")
            painter.restore()
        self.viewer.add_overlay('ramachandran', draw_rama)
        self.log(f"Ramachandran: {len(phi_psi)} residues plotted")
        return self

    def compare(self, names_or_smiles):
        """Compare multiple drugs side by side. Accepts preset names or SMILES."""
        rows = []
        for item in names_or_smiles:
            key = item.lower().replace(' ','').replace('-','')
            found = False
            for k, data in DRUG_PRESETS.items():
                if key in k or k in key:
                    rows.append({
                        'name': data.get('name', k)[:20],
                        'mw': data.get('mw', 0),
                        'logp': data.get('logp', 0),
                        'hbd': data.get('hbd', 0),
                        'hba': data.get('hba', 0),
                        'tpsa': data.get('tpsa', 0),
                    })
                    found = True; break
            if not found:
                rows.append({'name': item[:20], 'mw': 0, 'logp': 0, 'hbd': 0, 'hba': 0, 'tpsa': 0})
        if not rows: return self
        height = 60 + len(rows) * 18
        width = 380
        def draw_compare(painter, w, h):
            ox, oy = (w - width) // 2, h - height - 20
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255,255,255,230))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(40,55,80))
            painter.drawText(ox+10, oy+6, width-20, 14, Qt.AlignVCenter, "Drug Comparison")
            # Header
            y = oy + 24
            headers = ['Name', 'MW', 'LogP', 'HBD', 'HBA', 'TPSA']
            xs = [ox+10, ox+120, ox+180, ox+230, ox+270, ox+320]
            painter.setFont(QFont("Consolas", 7, QFont.Bold)); painter.setPen(QColor(80,90,120))
            for xi, hdr in zip(xs, headers):
                painter.drawText(xi, y, 60, 12, Qt.AlignLeft, hdr)
            y += 16
            # Rows
            painter.setFont(QFont("Consolas", 8)); painter.setPen(QColor(50,60,80))
            for row in rows:
                vals = [row['name'], f"{row['mw']:.0f}", f"{row['logp']:.1f}",
                        str(row['hbd']), str(row['hba']), f"{row['tpsa']:.0f}"]
                for xi, val in zip(xs, vals):
                    painter.drawText(xi, y, 80, 13, Qt.AlignLeft, val)
                y += 18
        self.viewer.add_overlay('compare', draw_compare)
        self.log(f"Comparing {len(rows)} molecules")
        return self

    def overlay(self, name, fn):
        """Add custom overlay: fn(painter, width, height)."""
        self.viewer.add_overlay(name, fn); return self

    def remove_overlay(self, name):
        self.viewer.remove_overlay(name); return self

    def clear_overlays(self):
        """Remove all overlays."""
        self.viewer._overlays.clear(); return self

    # ── Protein helpers ────────────────────────────────────────

    def show_binding_pocket(self, center_atom=None, radius=8.0):
        """Show atoms within radius of a center atom (defaults to first HETATM)."""
        if center_atom is None:
            # Find first HETATM
            for i, a in enumerate(self.atoms):
                if a.get('hetatm', False) and a['el'] != 'H':
                    center_atom = i; break
        if center_atom is None:
            center_atom = 0
        ca = self.atoms[center_atom]
        pocket_atoms = []
        for i, a in enumerate(self.atoms):
            d = math.sqrt((a['x']-ca['x'])**2+(a['y']-ca['y'])**2+(a['z']-ca['z'])**2)
            if d <= radius:
                pocket_atoms.append(i)
        # Color pocket atoms brighter
        colors = [0x444444] * len(self.atoms)  # dim everything
        for i in pocket_atoms:
            a = self.atoms[i]
            if a.get('hetatm'):
                colors[i] = 0xFF6633  # ligand atoms orange
            else:
                colors[i] = ELEMENTS.get(a['el'], {'color': 0x888888})['color']
        self.viewer._atom_colors = colors
        self.viewer._rebuild_mol()
        self.log(f"Binding pocket: {len(pocket_atoms)} atoms within {radius}\u00c5 of atom {center_atom}")
        return self

    def dock(self, ligand_smiles=None, score_only=True):
        """Simplified docking score estimation.
        Returns estimated binding affinity (kcal/mol) based on pharmacophore matching."""
        if not self.atoms:
            self.log("No molecule loaded"); return None
        # Compute interaction score based on pharmacophore features
        pharma = self.pharmacophore or _identify_pharmacophore(self.atoms, self.bonds)
        n_donors = sum(1 for f in pharma if f['type'] == 'donor')
        n_acceptors = sum(1 for f in pharma if f['type'] == 'acceptor')
        n_hydrophobic = sum(1 for f in pharma if f['type'] == 'hydrophobic')
        n_aromatic = sum(1 for f in pharma if f['type'] == 'aromatic')
        # Simplified scoring function
        score = -(n_donors * 1.2 + n_acceptors * 0.8 + n_hydrophobic * 0.3 + n_aromatic * 0.5)
        score -= min(3.0, len(self.atoms) * 0.02)  # size bonus
        result = {
            'estimated_affinity_kcal': round(score, 2),
            'pharmacophore_contacts': {
                'hbond_donors': n_donors, 'hbond_acceptors': n_acceptors,
                'hydrophobic': n_hydrophobic, 'aromatic': n_aromatic,
            },
            'note': 'Simplified scoring — use AutoDock Vina for rigorous docking'
        }
        self.log(f"Dock estimate: {score:.2f} kcal/mol ({n_donors}D {n_acceptors}A {n_hydrophobic}H {n_aromatic}Ar)")
        return result

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

    # ── Export ─────────────────────────────────────────────────

    def export_xyz(self, path="~/molecule.xyz"):
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            f.write(f"{len(self.atoms)}\nExported from DrugLab\n")
            for a in self.atoms:
                f.write(f"{a['el']}  {a['x']:.6f}  {a['y']:.6f}  {a['z']:.6f}\n")
        self.log(f"Exported: {path}")
        return path

    def export_pdb(self, path="~/molecule.pdb"):
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            for i, a in enumerate(self.atoms):
                rec = 'HETATM' if a.get('hetatm') else 'ATOM  '
                resname = a.get('resname', 'UNK')[:3]
                chain = a.get('chain', 'A')
                resnum = a.get('resnum', 1)
                atomname = a.get('atomname', a['el'])
                f.write(f"{rec}{i+1:5d} {atomname:<4s} {resname:3s} {chain}{resnum:4d}    "
                        f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00  0.00           {a['el']:>2s}\n")
            f.write("END\n")
        self.log(f"Exported PDB: {path}")
        return path

    def export_sdf(self, path="~/molecule.sdf"):
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            f.write(f"{self.info.get('name','molecule')}\n  DrugLab\n\n")
            f.write(f"{len(self.atoms):3d}{len(self.bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n")
            for a in self.atoms:
                f.write(f"{a['x']:10.4f}{a['y']:10.4f}{a['z']:10.4f} {a['el']:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n")
            for i, j in self.bonds:
                f.write(f"{i+1:3d}{j+1:3d}  1  0  0  0  0\n")
            f.write("M  END\n$$$$\n")
        self.log(f"Exported SDF: {path}")
        return path

    def screenshot(self, path="~/druglab_screenshot.png"):
        self.viewer.screenshot(os.path.expanduser(path))
        self.log(f"Screenshot: {path}")
        return self


# ═══════════════════════════════════════════════════════════════
#  LIGHT-MODE STYLESHEET (teal/medical theme)
# ═══════════════════════════════════════════════════════════════

_SS = """
QWidget{background:rgba(255,255,255,220);color:#2a3040;font-family:'Consolas','Menlo',monospace;font-size:11px}
QPushButton{background:rgba(240,248,250,240);border:1px solid #b0d4d8;border-radius:5px;padding:6px 10px;color:#2a6868;font-weight:bold;font-size:10px}
QPushButton:hover{background:rgba(220,242,246,250);border-color:#80bcc4}
QPushButton:checked{background:rgba(40,160,160,30);border-color:#40a0a0;color:#1a6060}
QSlider::groove:horizontal{height:4px;background:#d0e0e4;border-radius:2px}
QSlider::handle:horizontal{background:#40a0a0;width:14px;margin:-5px 0;border-radius:7px}
QComboBox{background:rgba(248,252,253,240);border:1px solid #b0d4d8;border-radius:4px;padding:5px 8px}
QComboBox QAbstractItemView{background:white;border:1px solid #b0d4d8;selection-background-color:#e0f4f6}
QTextEdit{background:rgba(248,252,253,220);border:1px solid #c8dce0;border-radius:4px;font-size:10px;color:#3a4a58;padding:6px}
QCheckBox{spacing:8px;color:#3a4a58}
QTabWidget::pane{border:1px solid #c0d8dc;background:rgba(255,255,255,200);border-top:none}
QTabBar::tab{background:rgba(235,248,250,200);color:#5a7a80;padding:7px 14px;font-size:10px;font-weight:bold;border:1px solid #c0d8dc;border-bottom:none}
QTabBar::tab:selected{background:rgba(255,255,255,240);color:#1a6868;border-bottom:2px solid #40a0a0}
QListWidget{background:rgba(248,252,253,220);border:1px solid #c8dce0;border-radius:4px;font-size:10px;color:#3a4a58}
QListWidget::item:selected{background:#e0f4f6;color:#1a5858}
QLabel{background:transparent}
QScrollArea{border:none;background:transparent}
QLineEdit{background:rgba(248,252,253,240);border:1px solid #b0d4d8;border-radius:4px;padding:5px 8px}
"""

def _lbl(text):
    l=QLabel(text.upper())
    l.setStyleSheet("font-size:9px;letter-spacing:1.5px;color:#6a8a90;font-weight:bold;padding:2px 0;background:transparent")
    return l


# ═══════════════════════════════════════════════════════════════
#  DRUGLAB APP — UI ASSEMBLY + WIRING (class-based)
# ═══════════════════════════════════════════════════════════════

class DrugLabApp:
    """Encapsulates the entire DrugLab UI: panel, viewer, tabs, and signal wiring."""

    def __init__(self):
        # ── Main widget ──
        self.main_widget = QWidget()
        self.main_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ── Panel ──
        self.panel = QWidget()
        self.panel.setFixedWidth(300)
        self.panel.setStyleSheet(_SS)
        self.panel.setAttribute(Qt.WA_TranslucentBackground, True)
        ps = QScrollArea()
        ps.setWidgetResizable(True)
        ps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ps.setStyleSheet("QScrollArea{border:none;background:transparent}QScrollBar:vertical{width:5px;background:transparent}QScrollBar::handle:vertical{background:#a0c8cc;border-radius:2px;min-height:30px}")
        inner = QWidget()
        inner.setAttribute(Qt.WA_TranslucentBackground, True)
        lay = QVBoxLayout(inner)
        lay.setSpacing(4)
        lay.setContentsMargins(10, 10, 10, 10)

        # Header
        hdr = QWidget()
        hdr.setAttribute(Qt.WA_TranslucentBackground, True)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 4)
        ic = QLabel("\U0001f9ec")
        ic.setStyleSheet("font-size:20px;background:rgba(220,245,240,200);border:1px solid #b0d4d8;border-radius:7px;padding:3px 7px")
        nw = QWidget()
        nw.setAttribute(Qt.WA_TranslucentBackground, True)
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(6, 0, 0, 0)
        nl.setSpacing(0)
        _title_lbl = QLabel("DrugLab")
        _title_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#1a3a40;background:transparent")
        _sub_lbl = QLabel("DRUG DISCOVERY WORKBENCH")
        _sub_lbl.setStyleSheet("font-size:7px;letter-spacing:2px;color:#6a8a90;background:transparent")
        nl.addWidget(_title_lbl)
        nl.addWidget(_sub_lbl)
        hl.addWidget(ic)
        hl.addWidget(nw)
        hl.addStretch()
        lay.addWidget(hdr)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(_SS)

        # ── Tab 1: Molecules ──
        t1 = QWidget()
        t1.setAttribute(Qt.WA_TranslucentBackground, True)
        t1l = QVBoxLayout(t1)
        t1l.setSpacing(5)
        t1l.setContentsMargins(6, 8, 6, 6)
        t1l.addWidget(_lbl("Drug Presets"))
        self.mol_combo = QComboBox()
        self.mol_combo.addItems(list(DRUG_PRESETS.keys()))
        t1l.addWidget(self.mol_combo)
        t1l.addWidget(_lbl("Style"))
        sw = QWidget()
        sw.setAttribute(Qt.WA_TranslucentBackground, True)
        sl = QHBoxLayout(sw)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)
        self.style_btns = {}
        for sn, slb in [("ballstick", "Ball&Stick"), ("spacefill", "SpaceFill"), ("wireframe", "Wire")]:
            b = QPushButton(slb)
            b.setCheckable(True)
            b.setChecked(sn == "ballstick")
            self.style_btns[sn] = b
            sl.addWidget(b)
        t1l.addWidget(sw)
        self.bonds_cb = QCheckBox("Show Bonds")
        self.bonds_cb.setChecked(True)
        t1l.addWidget(self.bonds_cb)
        t1l.addWidget(_lbl("Color Mode"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["element", "hydrophobicity", "bfactor", "pharmacophore"])
        t1l.addWidget(self.color_combo)
        t1l.addWidget(_lbl("Export"))
        ew = QWidget()
        ew.setAttribute(Qt.WA_TranslucentBackground, True)
        el = QHBoxLayout(ew)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(2)
        self.xyz_btn = QPushButton("Export .xyz")
        self.pdb_btn = QPushButton("Export .pdb")
        self.sdf_btn = QPushButton("Export .sdf")
        el.addWidget(self.xyz_btn)
        el.addWidget(self.pdb_btn)
        el.addWidget(self.sdf_btn)
        t1l.addWidget(ew)
        t1l.addStretch()
        self.tabs.addTab(t1, "Molecule")

        # ── Tab 2: Analysis ──
        t2 = QWidget()
        t2.setAttribute(Qt.WA_TranslucentBackground, True)
        t2l = QVBoxLayout(t2)
        t2l.setSpacing(5)
        t2l.setContentsMargins(6, 8, 6, 6)
        t2l.addWidget(_lbl("Drug Properties"))
        self.lipinski_btn = QPushButton("Lipinski Ro5 Panel")
        t2l.addWidget(self.lipinski_btn)
        self.desc_btn = QPushButton("Descriptor Radar")
        t2l.addWidget(self.desc_btn)
        self.admet_btn = QPushButton("ADMET Predictions")
        t2l.addWidget(self.admet_btn)
        self.pharma_btn = QPushButton("Pharmacophore Map")
        t2l.addWidget(self.pharma_btn)
        self.hbond_btn = QPushButton("H-Bond Network")
        t2l.addWidget(self.hbond_btn)
        t2l.addWidget(_lbl("Protein Analysis"))
        self.contacts_btn = QPushButton("Show Contacts")
        t2l.addWidget(self.contacts_btn)
        self.rama_btn = QPushButton("Ramachandran Plot")
        t2l.addWidget(self.rama_btn)
        self.pocket_btn = QPushButton("Binding Pocket")
        t2l.addWidget(self.pocket_btn)
        self.dock_btn = QPushButton("Estimate Binding")
        t2l.addWidget(self.dock_btn)
        t2l.addWidget(_lbl("Clear"))
        self.clear_btn = QPushButton("Clear All Overlays")
        t2l.addWidget(self.clear_btn)
        t2l.addStretch()
        self.tabs.addTab(t2, "Analysis")

        # ── Tab 3: Import ──
        t3 = QWidget()
        t3.setAttribute(Qt.WA_TranslucentBackground, True)
        t3l = QVBoxLayout(t3)
        t3l.setSpacing(5)
        t3l.setContentsMargins(6, 8, 6, 6)
        t3l.addWidget(_lbl("SMILES Input"))
        self.smiles_edit = QLineEdit()
        self.smiles_edit.setPlaceholderText("e.g. c1ccccc1 (benzene)")
        t3l.addWidget(self.smiles_edit)
        self.smiles_btn = QPushButton("Load SMILES")
        t3l.addWidget(self.smiles_btn)
        t3l.addWidget(_lbl("PDB Code / File"))
        self.pdb_edit = QLineEdit()
        self.pdb_edit.setPlaceholderText("e.g. 2HU4 or ~/protein.pdb")
        t3l.addWidget(self.pdb_edit)
        self.pdb_load_btn = QPushButton("Load PDB")
        t3l.addWidget(self.pdb_load_btn)
        t3l.addWidget(_lbl("Load .sdf / .mol / .xyz"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("~/ligand.sdf")
        t3l.addWidget(self.path_edit)
        self.file_load_btn = QPushButton("Load File")
        t3l.addWidget(self.file_load_btn)
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#5a7a80;font-size:10px;background:transparent")
        t3l.addWidget(self.status_lbl)
        t3l.addStretch()
        self.tabs.addTab(t3, "Import")

        # ── Tab 4: Info ──
        t4 = QWidget()
        t4.setAttribute(Qt.WA_TranslucentBackground, True)
        t4l = QVBoxLayout(t4)
        t4l.setSpacing(5)
        t4l.setContentsMargins(6, 8, 6, 6)
        t4l.addWidget(_lbl("Properties"))
        self.info_edit = QTextEdit()
        self.info_edit.setReadOnly(True)
        self.info_edit.setMinimumHeight(100)
        t4l.addWidget(self.info_edit)
        t4l.addWidget(_lbl("Atoms"))
        self.atom_list = QListWidget()
        self.atom_list.setMinimumHeight(80)
        t4l.addWidget(self.atom_list)
        t4l.addWidget(_lbl("Log"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlainText("[DrugLab] Initialised\n[DrugLab] Renderer: ModernGL\n")
        t4l.addWidget(self.log_edit)
        t4l.addStretch()
        self.tabs.addTab(t4, "Info")

        lay.addWidget(self.tabs)
        ps.setWidget(inner)
        playout = QVBoxLayout(self.panel)
        playout.setContentsMargins(0, 0, 0, 0)
        playout.addWidget(ps)

        self.viewer = MolViewer()
        self.viewer.setStyleSheet("background:transparent")
        self.main_layout.addWidget(self.panel)
        self.main_layout.addWidget(self.viewer, 1)

        # ── Create singleton & wire signals ──
        self.drug = DrugLab(self.viewer, self.log_edit)
        self._wire_signals()
        self._wrap_load_methods()

        # Load default
        self.drug.load(self.mol_combo.currentText())

    def _update_ui(self):
        """Sync UI with drug state after any load."""
        self.atom_list.clear()
        for i, a in enumerate(self.drug.atoms[:200]):  # cap display for large proteins
            el = ELEMENTS.get(a['el'], {'name': '?'})
            res_info = f" {a['resname']}{a['resnum']}" if a.get('resname') else ''
            self.atom_list.addItem(f"{i:3d} {a['el']:2s} ({a['x']:+.3f},{a['y']:+.3f},{a['z']:+.3f}){res_info}")
        if len(self.drug.atoms) > 200:
            self.atom_list.addItem(f"... ({len(self.drug.atoms) - 200} more atoms)")
        info_lines = [f"Name: {self.drug.info.get('name', '')}"]
        info_lines.append(f"Formula: {self.drug.formula()}")
        info_lines.append(f"MW: {self.drug.mass():.2f} g/mol")
        if self.drug.info.get('smiles'):
            info_lines.append(f"SMILES: {self.drug.info['smiles'][:50]}")
        if self.drug.info.get('logp') is not None:
            info_lines.append(f"LogP: {self.drug.info['logp']:.2f}")
        if self.drug.info.get('hbd') is not None:
            info_lines.append(f"HBD: {self.drug.info['hbd']}  HBA: {self.drug.info.get('hba', 0)}")
        if self.drug.info.get('tpsa') is not None:
            _ang = "\u00c5\u00b2"
            info_lines.append(f"TPSA: {self.drug.info['tpsa']:.1f} {_ang}")
        if self.drug.info.get('lipinski_pass') is not None:
            _lip = "PASS \u2705" if self.drug.info['lipinski_pass'] else "FAIL \u274c"
            info_lines.append(f"Lipinski: {_lip}")
        if self.drug.info.get('druglikeness') is not None:
            info_lines.append(f"Druglikeness: {self.drug.info['druglikeness']:.3f}")
        if self.drug.info.get('target'):
            info_lines.append(f"Target: {self.drug.info['target']}")
        if self.drug.info.get('drug_class'):
            info_lines.append(f"Class: {self.drug.info['drug_class']}")
        if self.drug.info.get('pdb_code'):
            info_lines.append(f"PDB: {self.drug.info['pdb_code']}")
        self.info_edit.setPlainText('\n'.join(info_lines))

    def _wrap_load_methods(self):
        """Wrap load methods to auto-update UI after each call."""
        _orig_load = self.drug.load
        def _load_wrap(name): _orig_load(name); self._update_ui(); return self.drug
        self.drug.load = _load_wrap

        _orig_load_smiles = self.drug.load_smiles
        def _load_smiles_wrap(s): _orig_load_smiles(s); self._update_ui(); return self.drug
        self.drug.load_smiles = _load_smiles_wrap

        _orig_load_pdb = self.drug.load_pdb
        def _load_pdb_wrap(p): _orig_load_pdb(p); self._update_ui(); return self.drug
        self.drug.load_pdb = _load_pdb_wrap

        _orig_load_sdf = self.drug.load_sdf
        def _load_sdf_wrap(p): _orig_load_sdf(p); self._update_ui(); return self.drug
        self.drug.load_sdf = _load_sdf_wrap

        _orig_load_xyz = self.drug.load_xyz
        def _load_xyz_wrap(t): _orig_load_xyz(t); self._update_ui(); return self.drug
        self.drug.load_xyz = _load_xyz_wrap

        _orig_load_atoms = self.drug.load_atoms
        def _load_atoms_wrap(a): _orig_load_atoms(a); self._update_ui(); return self.drug
        self.drug.load_atoms = _load_atoms_wrap

    def _wire_signals(self):
        """Connect all UI signals to drug API methods."""
        # Molecule combo
        self.mol_combo.currentTextChanged.connect(lambda t: self.drug.load(t))

        # Style buttons
        def _on_style(sn):
            for k, b in self.style_btns.items():
                b.setChecked(k == sn)
            self.drug.style(sn)
        for sn, bt in self.style_btns.items():
            bt.clicked.connect(lambda c, s=sn: _on_style(s))

        # Bonds checkbox
        self.bonds_cb.toggled.connect(lambda c: (setattr(self.viewer, 'show_bonds', c), self.viewer._rebuild_mol()))

        # Color combo
        self.color_combo.currentTextChanged.connect(lambda t: self.drug.color_by(t))

        # Export buttons
        self.xyz_btn.clicked.connect(lambda: (self.drug.export_xyz(), self.status_lbl.setText("\u2713 Saved ~/molecule.xyz")))
        self.pdb_btn.clicked.connect(lambda: (self.drug.export_pdb(), self.status_lbl.setText("\u2713 Saved ~/molecule.pdb")))
        self.sdf_btn.clicked.connect(lambda: (self.drug.export_sdf(), self.status_lbl.setText("\u2713 Saved ~/molecule.sdf")))

        # Analysis buttons
        self.lipinski_btn.clicked.connect(lambda: self.drug.overlay_lipinski())
        self.desc_btn.clicked.connect(lambda: self.drug.overlay_descriptors())
        self.admet_btn.clicked.connect(lambda: self.drug.overlay_admet())
        self.pharma_btn.clicked.connect(lambda: self.drug.show_pharmacophore())
        self.hbond_btn.clicked.connect(lambda: self.drug.show_hbonds())
        self.contacts_btn.clicked.connect(lambda: self.drug.overlay_contacts())
        self.rama_btn.clicked.connect(lambda: self.drug.overlay_ramachandran())
        self.pocket_btn.clicked.connect(lambda: self.drug.show_binding_pocket())
        self.dock_btn.clicked.connect(lambda: self.drug.dock())
        self.clear_btn.clicked.connect(lambda: self.drug.clear_overlays())

        # Import buttons
        self.smiles_btn.clicked.connect(lambda: (
            self.drug.load_smiles(self.smiles_edit.text().strip()) if self.smiles_edit.text().strip()
            else self.status_lbl.setText("\u26a0 Enter SMILES first")
        ))
        self.pdb_load_btn.clicked.connect(lambda: (
            self.drug.load_pdb(self.pdb_edit.text().strip()) if self.pdb_edit.text().strip()
            else self.status_lbl.setText("\u26a0 Enter PDB code or path")
        ))
        self.file_load_btn.clicked.connect(self._on_file_load)

        # Atom list selection
        self.atom_list.currentRowChanged.connect(
            lambda r: self.drug.select(r) if 0 <= r < len(self.drug.atoms) else None
        )

    def _on_file_load(self):
        p = self.path_edit.text().strip()
        if not p:
            self.status_lbl.setText("\u26a0 Enter file path")
            return
        p = os.path.expanduser(p)
        if p.endswith('.sdf') or p.endswith('.mol'):
            self.drug.load_sdf(p)
        elif p.endswith('.pdb'):
            self.drug.load_pdb(p)
        else:
            self.drug.load_xyz(open(p).read() if os.path.isfile(p) else p)
        self.status_lbl.setText(f"\u2713 Loaded {os.path.basename(p)}")


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE APP & EXPOSE GLOBALS
# ═══════════════════════════════════════════════════════════════

drug_app = DrugLabApp()
drug = drug_app.drug
drug_viewer = drug_app.viewer

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

drug_proxy = graphics_scene.addWidget(drug_app.main_widget)
drug_proxy.setPos(0, 0)
drug_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
drug_shadow = QGraphicsDropShadowEffect()
drug_shadow.setBlurRadius(60); drug_shadow.setOffset(45, 45); drug_shadow.setColor(QColor(0,0,0,120))
drug_proxy.setGraphicsEffect(drug_shadow)
drug_app.main_widget.resize(1400, 850)

# Center in current view
_vr = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
drug_proxy.setPos(_vr.center().x() - drug_app.main_widget.width() / 2,
             _vr.center().y() - drug_app.main_widget.height() / 2)
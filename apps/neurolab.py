"""
NeuroLab — Brain Imaging Workbench for Rio
═══════════════════════════════════════════════════════════════

ARCHITECTURE: This file is the complete runtime. It registers a `brain`
singleton into the namespace. The LLM sees this source as context and
injects short code snippets to drive it.

EXAMPLES (what the LLM would output):
    brain.load("mni152")                        # load MNI152 template
    brain.load("fsaverage")                      # load FreeSurfer average brain
    brain.load_nifti("/path/to/scan.nii.gz")     # load any NIfTI volume
    brain.load_surface("/path/to/lh.pial")       # load FreeSurfer surface
    brain.slice("axial", z=45)                   # show axial slice at z=45
    brain.slice("sagittal", x=90)                # sagittal slice
    brain.slice("coronal", y=110)                # coronal slice
    brain.slice_3plane(x=90, y=110, z=45)        # all three planes at once
    brain.colormap("hot")                        # change colormap: hot/cool/gray/viridis/rdbu
    brain.opacity(0.7)                           # set surface opacity
    brain.threshold(2.3)                         # threshold overlay at t>2.3
    brain.screenshot("/tmp/brain.png")           # save current view

    ## OVERLAYS (statistical maps, activations):
    brain.overlay_nifti("/path/to/zstat.nii.gz") # overlay stat map
    brain.overlay_atlas("harvard-oxford")        # overlay parcellation
    brain.overlay_roi(region="precentral")       # highlight named ROI
    brain.clear_overlay()                        # remove overlay

    ## ANALYSIS (NiBabel + nilearn built-in):
    brain.extract_roi(region="hippocampus")      # extract ROI timeseries
    brain.smooth(fwhm=6)                         # spatial smoothing
    brain.resample(target_mm=2)                  # resample to isotropic
    brain.segment()                              # tissue segmentation (GM/WM/CSF)
    brain.compute_glm(events, tr=2.0)            # first-level GLM
    brain.connectivity(seeds=[(0,0,0)])          # seed-based connectivity
    brain.parcellate(atlas="aal")                # parcellate into regions

    ## After analysis, brain.info contains results:
    brain.info['shape']          # volume dimensions
    brain.info['voxel_size']     # voxel dimensions in mm
    brain.info['affine']         # 4x4 affine matrix
    brain.info['dtype']          # data type
    brain.info['range']          # (min, max) of volume data
    brain.info['n_voxels']       # total voxels
    brain.info['n_nonzero']      # nonzero voxels (for masks/stats)
    brain.info['roi_stats']      # dict of ROI statistics
    brain.info['glm_results']    # GLM output (betas, t-stats, etc.)

    ## MNE-Python (EEG/MEG, if installed):
    brain.load_eeg("/path/to/raw.fif")           # load EEG/MEG
    brain.plot_eeg(channels=["Cz","Fz"])         # overlay ERP waveforms
    brain.topomap(time=0.1)                      # show scalp topography at 100ms

VIEWER API (lower level, when LLM needs custom rendering):
    brain.viewer.set_volume(data_3d, affine, name)
    brain.viewer.set_overlay(data_3d, cmap, alpha)
    brain.viewer.set_surface(verts, faces, name)
    brain.viewer.set_slice_axis(axis, index)
    brain.viewer.add_overlay(name, fn)           # fn(painter, w, h) for custom QPainter
    brain.viewer.remove_overlay(name)
    brain.viewer.cam_dist = 250.0
    brain.viewer.rot_x, brain.viewer.rot_y
    brain.viewer.screenshot(path)

NAMESPACE: After this file runs, these are available:
    brain       — NeuroLab singleton (main API)
    viewer      — alias for brain.viewer (the 3D widget)
    nlab        — alias for brain (short form)
    ATLASES     — atlas registry dict
    COLORMAPS   — available colormaps
    All PySide6/Qt, numpy, moderngl, glm from parser namespace
"""

import math
import time
import os
import re
import subprocess
import threading
import struct
import gzip
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
    QPainter, QColor, QFont, QPen, QBrush, QImage, QLinearGradient,
    QPainterPath
)

# moderngl: prefer parser-injected, fallback to import
import json
try:
    _test_mgl = moderngl.create_context
except NameError:
    import moderngl

# ── Pure-numpy replacements for PyGLM (avoids import conflicts) ──

def _radians(deg):
    return deg * math.pi / 180.0

def _perspective(fov_rad, aspect, near, far):
    """4x4 perspective projection matrix (row-major numpy)."""
    f = 1.0 / math.tan(fov_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def _look_at(eye, center, up):
    """4x4 look-at view matrix (row-major numpy)."""
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye; f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up); s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m

def _gl(m):
    """Convert row-major 4x4 numpy matrix to column-major bytes for OpenGL."""
    return np.asarray(m, dtype=np.float32).T.tobytes()

_IDENTITY_GL = np.eye(4, dtype=np.float32).tobytes()

def _vec3_bytes(x, y, z):
    """Pack 3 floats as bytes for uniform upload."""
    return np.array([x, y, z], dtype=np.float32).tobytes()

def _normalize3(x, y, z):
    """Normalize a 3-vector and return as bytes."""
    v = np.array([x, y, z], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.tobytes()

def _translate_mat4(tx, ty, tz):
    """Return a 4x4 translation as a row-major numpy matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m

# ═══════════════════════════════════════════════════════════════
#  PURE-NUMPY HELPERS (replaces scipy.ndimage dependency)
# ═══════════════════════════════════════════════════════════════

def _zoom_nearest(arr, target_shape):
    """Resize a 3D array to target_shape using nearest-neighbor interpolation."""
    sx, sy, sz = arr.shape
    tx, ty, tz = target_shape
    xi = (np.arange(tx) * sx / tx).astype(int).clip(0, sx-1)
    yi = (np.arange(ty) * sy / ty).astype(int).clip(0, sy-1)
    zi = (np.arange(tz) * sz / tz).astype(int).clip(0, sz-1)
    return arr[np.ix_(xi, yi, zi)]

def _gaussian_kernel_1d(sigma, radius=None):
    """Create a 1D Gaussian kernel."""
    if radius is None:
        radius = int(math.ceil(sigma * 3))
    radius = max(radius, 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / max(sigma, 0.01)) ** 2)
    return k / k.sum()

def _gaussian_smooth_3d(vol, sigma=1.0):
    """Apply separable Gaussian smoothing to a 3D volume (pure numpy)."""
    if sigma <= 0: return vol
    k = _gaussian_kernel_1d(sigma)
    r = len(k) // 2
    out = vol.astype(np.float32).copy()
    # Pad and convolve along each axis
    for axis in range(3):
        out = np.moveaxis(out, axis, 0)
        shape_1d = out.shape
        n = shape_1d[0]
        padded = np.pad(out, ((r, r),) + ((0, 0),) * (out.ndim - 1), mode='reflect')
        result = np.zeros_like(out)
        for i in range(len(k)):
            result += padded[i:i+n] * k[i]
        out = np.moveaxis(result, 0, axis)
    return out

def _zoom_linear_3d(arr, zoom_factors):
    """Resize a 3D array using trilinear interpolation."""
    sx, sy, sz = arr.shape[:3]
    tx = max(1, int(round(sx * zoom_factors[0])))
    ty = max(1, int(round(sy * zoom_factors[1])))
    tz = max(1, int(round(sz * zoom_factors[2])))
    # Source coordinates
    xi = np.linspace(0, sx - 1, tx).astype(np.float32)
    yi = np.linspace(0, sy - 1, ty).astype(np.float32)
    zi = np.linspace(0, sz - 1, tz).astype(np.float32)
    x0 = np.floor(xi).astype(int).clip(0, sx-2)
    y0 = np.floor(yi).astype(int).clip(0, sy-2)
    z0 = np.floor(zi).astype(int).clip(0, sz-2)
    xf = xi - x0; yf = yi - y0; zf = zi - z0
    # Trilinear
    out = np.zeros((tx, ty, tz), dtype=np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((1-xf+dx*(2*xf-1))[:, None, None] *
                     (1-yf+dy*(2*yf-1))[None, :, None] *
                     (1-zf+dz*(2*zf-1))[None, None, :])
                out += arr[(x0+dx).clip(0,sx-1)][:, (y0+dy).clip(0,sy-1)][:, :, (z0+dz).clip(0,sz-1)] * w
    return out

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & COLORMAPS
# ═══════════════════════════════════════════════════════════════

def _hex(h):
    return ((h>>16)&0xFF)/255.0, ((h>>8)&0xFF)/255.0, (h&0xFF)/255.0

def _lerp_color(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))

def _cmap_lookup(cmap_name, val):
    """Map val in [0,1] to (r,g,b) in [0,1]."""
    cmap = COLORMAPS.get(cmap_name, COLORMAPS['gray'])
    val = max(0.0, min(1.0, val))
    n = len(cmap) - 1
    idx = val * n
    lo = int(idx)
    hi = min(lo + 1, n)
    t = idx - lo
    return _lerp_color(cmap[lo], cmap[hi], t)

COLORMAPS = {
    'gray':    [(0,0,0), (1,1,1)],
    'hot':     [(0,0,0), (0.5,0,0), (1,0.3,0), (1,0.7,0), (1,1,0.5), (1,1,1)],
    'cool':    [(0,1,1), (1,0,1)],
    'viridis': [(0.267,0.004,0.329),(0.282,0.140,0.458),(0.245,0.287,0.531),
                (0.190,0.407,0.556),(0.127,0.566,0.551),(0.199,0.718,0.488),
                (0.454,0.842,0.346),(0.741,0.933,0.178),(0.993,0.906,0.144)],
    'rdbu':    [(0.2,0.2,0.7),(0.4,0.5,0.9),(0.7,0.8,1.0),(0.95,0.95,0.95),
                (1.0,0.8,0.7),(0.9,0.4,0.3),(0.7,0.15,0.15)],
    'plasma':  [(0.050,0.030,0.528),(0.294,0.011,0.631),(0.492,0.012,0.658),
                (0.658,0.134,0.588),(0.797,0.280,0.470),(0.903,0.432,0.343),
                (0.972,0.604,0.222),(0.993,0.801,0.162),(0.940,0.975,0.131)],
    'bone':    [(0,0,0),(0.25,0.25,0.35),(0.5,0.55,0.6),(0.75,0.78,0.78),(1,1,1)],
}

# ── Brain region data for built-in atlases ──
BRAIN_REGIONS = {
    'frontal':      {'color': 0xE06060, 'center': (25, 60, 45), 'desc': 'Frontal Lobe'},
    'parietal':     {'color': 0x60A0E0, 'center': (25, 35, 60), 'desc': 'Parietal Lobe'},
    'temporal':     {'color': 0x60E060, 'center': (15, 40, 25), 'desc': 'Temporal Lobe'},
    'occipital':    {'color': 0xE0E060, 'center': (25, 15, 40), 'desc': 'Occipital Lobe'},
    'cerebellum':   {'color': 0xE060E0, 'center': (25, 10, 20), 'desc': 'Cerebellum'},
    'hippocampus':  {'color': 0xFF8C00, 'center': (20, 32, 25), 'desc': 'Hippocampus'},
    'amygdala':     {'color': 0xFF4500, 'center': (18, 36, 20), 'desc': 'Amygdala'},
    'thalamus':     {'color': 0x00CED1, 'center': (25, 38, 35), 'desc': 'Thalamus'},
    'caudate':      {'color': 0x9370DB, 'center': (22, 44, 35), 'desc': 'Caudate Nucleus'},
    'putamen':      {'color': 0x20B2AA, 'center': (20, 40, 30), 'desc': 'Putamen'},
    'insula':       {'color': 0xCD853F, 'center': (14, 42, 30), 'desc': 'Insular Cortex'},
    'cingulate':    {'color': 0x87CEEB, 'center': (25, 42, 48), 'desc': 'Cingulate Cortex'},
    'precentral':   {'color': 0xDC143C, 'center': (22, 48, 58), 'desc': 'Precentral Gyrus (Motor)'},
    'postcentral':  {'color': 0x4169E1, 'center': (22, 40, 58), 'desc': 'Postcentral Gyrus (Sensory)'},
    'broca':        {'color': 0xFF6347, 'center': (10, 52, 30), 'desc': "Broca's Area"},
    'wernicke':     {'color': 0x48D1CC, 'center': (10, 28, 32), 'desc': "Wernicke's Area"},
    'corpus_callosum': {'color': 0xFFFFFF, 'center': (25, 38, 38), 'desc': 'Corpus Callosum'},
}

ATLASES = {
    'harvard-oxford': 'Harvard-Oxford Cortical/Subcortical Atlas',
    'aal':           'Automated Anatomical Labeling (AAL)',
    'desikan':       'Desikan-Killiany Atlas (FreeSurfer)',
    'destrieux':     'Destrieux Atlas (FreeSurfer)',
    'schaefer':      'Schaefer 2018 Parcellation',
    'brodmann':      'Brodmann Areas',
    'mni':           'MNI Structural Atlas',
}

# ═══════════════════════════════════════════════════════════════
#  SHADERS
# ═══════════════════════════════════════════════════════════════

_VERT_BRAIN = """
#version 330
uniform mat4 mvp; uniform mat4 model;
in vec3 in_position; in vec3 in_normal; in vec3 in_color;
out vec3 v_normal, v_color, v_world;
void main() {
    vec4 w = model * vec4(in_position, 1.0);
    v_world = w.xyz; gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = mat3(model) * in_normal; v_color = in_color;
}"""

_FRAG_BRAIN = """
#version 330
uniform vec3 light_dir, ambient, view_pos;
uniform float alpha;
in vec3 v_normal, v_color, v_world;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal), l = normalize(light_dir);
    float diff = max(dot(n, l), 0.0) * 0.6 + 0.4;
    vec3 vd = normalize(view_pos - v_world);
    vec3 h = normalize(l + vd);
    float spec = pow(max(dot(n, h), 0.0), 48.0) * 0.25;
    // Subsurface scattering approximation
    float sss = max(0.0, dot(-n, l)) * 0.12;
    vec3 col = v_color * (ambient + diff * vec3(1.0, 0.98, 0.96)) + vec3(spec) + v_color * sss;
    // Rim light for depth
    float rim = 1.0 - max(dot(n, vd), 0.0);
    col += vec3(0.85, 0.88, 0.95) * rim * rim * 0.15;
    col = col / (col + vec3(1.0));  // Reinhard tonemap
    frag = vec4(col, alpha);
}"""

_FRAG_SLICE = """
#version 330
uniform vec3 light_dir, ambient, view_pos;
uniform float alpha;
in vec3 v_normal, v_color, v_world;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal);
    float ao = abs(dot(n, vec3(0.0, 0.0, 1.0))) * 0.1 + 0.9;
    frag = vec4(v_color * ao, 1.0);
}"""

# ═══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
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

def _transform_verts(verts_list, mat):
    arr = np.array(verts_list, dtype=np.float32).reshape(-1, 9)
    if arr.shape[0] == 0: return []
    # Accept numpy 4x4 array directly, or glm-style [col][row] objects
    if isinstance(mat, np.ndarray):
        m4 = mat.astype(np.float32)
    else:
        m4 = np.array([[mat[c][r] for c in range(4)] for r in range(4)], dtype=np.float32)
    pos = np.hstack([arr[:,:3], np.ones((arr.shape[0],1), dtype=np.float32)])
    pos_t = (m4 @ pos.T).T[:,:3]
    nrm_t = (m4[:3,:3] @ arr[:,3:6].T).T
    return np.hstack([pos_t, nrm_t, arr[:,6:9]]).flatten().tolist()

# ═══════════════════════════════════════════════════════════════
#  NIFTI PARSER (standalone — no nibabel needed for basic loading)
# ═══════════════════════════════════════════════════════════════

def _read_nifti(path):
    """Parse NIfTI-1 (.nii or .nii.gz) → dict with data, affine, header info.
    Handles the most common formats without nibabel dependency."""
    path = os.path.expanduser(path)
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f: raw = f.read()
    else:
        with open(path, 'rb') as f: raw = f.read()

    # NIfTI-1 header (348 bytes)
    if len(raw) < 352: return None
    sizeof_hdr = struct.unpack_from('<i', raw, 0)[0]
    if sizeof_hdr != 348:
        # Try big endian
        sizeof_hdr = struct.unpack_from('>i', raw, 0)[0]
        if sizeof_hdr != 348: return None
        endian = '>'
    else:
        endian = '<'

    # Dimensions
    ndim = struct.unpack_from(f'{endian}h', raw, 40)[0]
    dims = struct.unpack_from(f'{endian}8h', raw, 40)  # dim[0..7]
    ndim = dims[0]
    nx, ny, nz = dims[1], dims[2], dims[3]
    nt = dims[4] if ndim >= 4 else 1

    # Data type
    datatype = struct.unpack_from(f'{endian}h', raw, 70)[0]
    bitpix = struct.unpack_from(f'{endian}h', raw, 72)[0]

    # Voxel sizes
    pixdim = struct.unpack_from(f'{endian}8f', raw, 76)

    # Affine from sform
    sform_code = struct.unpack_from(f'{endian}h', raw, 254)[0]
    srow_x = struct.unpack_from(f'{endian}4f', raw, 280)
    srow_y = struct.unpack_from(f'{endian}4f', raw, 296)
    srow_z = struct.unpack_from(f'{endian}4f', raw, 312)

    if sform_code > 0:
        affine = np.array([
            list(srow_x),
            list(srow_y),
            list(srow_z),
            [0, 0, 0, 1]
        ], dtype=np.float64)
    else:
        # Fallback: qform or simple scaling
        affine = np.diag([pixdim[1], pixdim[2], pixdim[3], 1.0])

    # Data offset
    vox_offset = struct.unpack_from(f'{endian}f', raw, 108)[0]
    offset = max(int(vox_offset), 352)
    scl_slope = struct.unpack_from(f'{endian}f', raw, 112)[0]
    scl_inter = struct.unpack_from(f'{endian}f', raw, 116)[0]

    # Read data
    dtype_map = {
        2: (f'{endian}u1', 1), 4: (f'{endian}i2', 2), 8: (f'{endian}i4', 4),
        16: (f'{endian}f4', 4), 64: (f'{endian}f8', 8),
        256: (f'{endian}i1', 1), 512: (f'{endian}u2', 2), 768: (f'{endian}u4', 4),
    }

    if datatype not in dtype_map: return None
    dt, bpp = dtype_map[datatype]

    n_voxels = nx * ny * nz * max(nt, 1)
    data_raw = raw[offset:offset + n_voxels * bpp]
    data = np.frombuffer(data_raw, dtype=np.dtype(dt))

    if nt > 1:
        data = data[:nx*ny*nz*nt].reshape((nx, ny, nz, nt))
    else:
        data = data[:nx*ny*nz].reshape((nx, ny, nz))

    data = data.astype(np.float32)
    if scl_slope != 0 and not (scl_slope == 1 and scl_inter == 0):
        data = data * scl_slope + scl_inter

    return {
        'data': data, 'affine': affine,
        'shape': (nx, ny, nz), 'nt': nt,
        'voxel_size': (abs(pixdim[1]), abs(pixdim[2]), abs(pixdim[3])),
        'datatype': datatype, 'endian': endian,
    }

# ═══════════════════════════════════════════════════════════════
#  SYNTHETIC BRAIN VOLUME GENERATOR
# ═══════════════════════════════════════════════════════════════

def _generate_mni152_phantom(shape=(91, 109, 91)):
    """Generate a synthetic brain-like volume for demonstration.
    Produces anatomically-plausible intensity distribution with
    gray matter, white matter, CSF, and ventricles."""
    nx, ny, nz = shape
    vol = np.zeros(shape, dtype=np.float32)

    # Coordinate grids normalized to [-1, 1]
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Brain mask: elongated ellipsoid
    brain_r = np.sqrt((X/0.65)**2 + (Y/0.85)**2 + (Z/0.60)**2)
    brain_mask = brain_r < 1.0

    # White matter core: smaller ellipsoid, higher intensity
    wm_r = np.sqrt((X/0.40)**2 + (Y/0.55)**2 + (Z/0.38)**2)
    wm_mask = wm_r < 1.0

    # Ventricles: butterfly shape
    vent_l = np.sqrt(((X+0.08)/0.06)**2 + ((Y-0.05)/0.25)**2 + ((Z+0.05)/0.08)**2)
    vent_r = np.sqrt(((X-0.08)/0.06)**2 + ((Y-0.05)/0.25)**2 + ((Z+0.05)/0.08)**2)
    vent_mask = (vent_l < 1.0) | (vent_r < 1.0)

    # Cerebellum: lower posterior blob
    cere_r = np.sqrt((X/0.45)**2 + ((Y+0.55)/0.30)**2 + ((Z+0.10)/0.30)**2)
    cere_mask = cere_r < 1.0

    # Brainstem: cylinder going down
    stem_r = np.sqrt((X/0.10)**2 + ((Z+0.05)/0.10)**2)
    stem_mask = (stem_r < 1.0) & (Y < -0.35) & (Y > -0.85)

    # Assign intensities (T1-like contrast: WM bright, GM mid, CSF dark)
    vol[brain_mask] = 120.0   # Gray matter shell
    vol[wm_mask] = 180.0      # White matter core
    vol[vent_mask & brain_mask] = 30.0   # CSF in ventricles
    vol[cere_mask] = 130.0    # Cerebellum
    vol[stem_mask] = 160.0    # Brainstem

    # Cortical ribbon: just inside the brain boundary
    cortex_r = (brain_r > 0.82) & (brain_r < 1.0)
    vol[cortex_r] = 100.0

    # Add sulcal folding pattern (noise modulated by position)
    np.random.seed(42)
    fold_noise = np.random.randn(nx//4+1, ny//4+1, nz//4+1).astype(np.float32)
    fold_noise = _zoom_nearest(fold_noise, (nx, ny, nz))
    vol[brain_mask] += fold_noise[brain_mask] * 15

    # Smooth slightly for realism
    vol = _gaussian_smooth_3d(vol, sigma=0.8)
    vol[~(brain_mask | cere_mask | stem_mask)] = 0

    # Standard MNI152 affine (2mm isotropic)
    affine = np.array([
        [-2, 0, 0, 90],
        [0, 2, 0, -126],
        [0, 0, 2, -72],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    return vol, affine

def _generate_activation_map(brain_vol, regions=None, noise=0.3):
    """Generate a synthetic activation map matching the brain volume shape.
    regions: list of region names from BRAIN_REGIONS to 'activate'."""
    if regions is None:
        regions = ['precentral', 'broca']
    shape = brain_vol.shape[:3]
    nx, ny, nz = shape
    act = np.zeros(shape, dtype=np.float32)

    for rname in regions:
        reg = BRAIN_REGIONS.get(rname)
        if not reg: continue
        cx, cy, cz = reg['center']
        # Scale center to volume coordinates
        sx, sy, sz = cx/50*nx, cy/80*ny, cz/60*nz
        x = np.arange(nx) - sx
        y = np.arange(ny) - sy
        z = np.arange(nz) - sz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt(X**2/64 + Y**2/64 + Z**2/49)
        blob = np.exp(-r**2 * 2.0) * 5.0
        act += blob

    # Add noise
    np.random.seed(7)
    act += np.random.randn(*shape).astype(np.float32) * noise
    # Mask to brain
    act[brain_vol < 10] = 0
    return act

def _generate_surface_mesh(vol, threshold=60, step=3):
    """Generate a coarse brain surface mesh from volume data using
    a simplified marching-cubes-like approach. Returns (verts, faces)
    as numpy arrays suitable for GL rendering."""
    nx, ny, nz = vol.shape[:3]
    vertices = []
    normals = []

    # Sample surface points where volume crosses threshold
    smooth = _gaussian_smooth_3d(vol.astype(np.float32), sigma=1.5)

    # Compute gradient for normals
    gy, gx, gz = np.gradient(smooth)

    for i in range(1, nx-1, step):
        for j in range(1, ny-1, step):
            for k in range(1, nz-1, step):
                v = smooth[i, j, k]
                # Check if this is near the surface
                neighbors = [
                    smooth[i-1,j,k], smooth[i+1,j,k],
                    smooth[i,j-1,k], smooth[i,j+1,k],
                    smooth[i,j,k-1], smooth[i,j,k+1]
                ]
                if v >= threshold and any(n < threshold for n in neighbors):
                    vertices.append([float(i), float(j), float(k)])
                    # Normal from gradient
                    nx_ = -gx[i,j,k]; ny_ = -gy[i,j,k]; nz_ = -gz[i,j,k]
                    nl = math.sqrt(nx_**2 + ny_**2 + nz_**2) + 1e-9
                    normals.append([nx_/nl, ny_/nl, nz_/nl])

    if not vertices:
        return np.zeros((0,3)), np.zeros((0,3))

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  BRAIN VIEWER WIDGET
# ═══════════════════════════════════════════════════════════════

class BrainViewer(QWidget):
    """OpenGL brain viewer — offscreen FBO → QImage → QPainter.
    Supports volume slicing, surface rendering, and custom overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Volume state
        self._vol = None         # 3D float32 array
        self._vol_name = ""
        self._affine = np.eye(4)
        self._overlay_vol = None  # overlay stat map
        self._overlay_cmap = 'hot'
        self._overlay_alpha = 0.65
        self._threshold = 0.0

        # Slice state
        self._slice_axis = 'axial'   # axial/sagittal/coronal
        self._slice_idx = {'axial': 45, 'sagittal': 45, 'coronal': 55}
        self._show_3plane = False

        # Surface state
        self._surf_verts = None
        self._surf_normals = None
        self._surf_name = ""

        # View mode
        self._view_mode = 'slice'  # 'slice' or 'surface'
        self._cmap = 'bone'

        # Camera
        self.rot_x = 0.25; self.rot_y = 0.0; self.auto_rot = 0.0
        self.cam_dist = 150.0
        self._dragging = False; self._lmx = 0; self._lmy = 0
        self._center = (45, 55, 45)

        # GL
        self._gl_ready = False; self.ctx = None; self.fbo = None
        self._fbo_w = 0; self._fbo_h = 0; self._frame = None
        self._surf_vao = None; self._surf_n = 0
        self._slice_vao = None; self._slice_n = 0

        # Overlays: dict of name → fn(painter, w, h)
        self._overlays = OrderedDict()

        # Markers: list of (x,y,z, color_hex, label)
        self._markers = []

        # Crosshair
        self._show_crosshair = True

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(16)

    def _ensure_gl(self):
        if self._gl_ready: return
        self._gl_ready = True
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog_brain = self.ctx.program(vertex_shader=_VERT_BRAIN, fragment_shader=_FRAG_BRAIN)
        self.prog_slice = self.prog_brain  # reuse — avoids attribute stripping issues
        self._resize_fbo(max(self.width(), 320), max(self.height(), 200))
        self.timer.start()

    def _resize_fbo(self, w, h):
        if w == self._fbo_w and h == self._fbo_h and self.fbo: return
        if self.fbo: self.fbo.release()
        self._fbo_w = w; self._fbo_h = h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)))

    def set_volume(self, data, affine=None, name=""):
        """Set the primary volume data."""
        self._vol = data.astype(np.float32) if data is not None else None
        self._vol_name = name
        if affine is not None: self._affine = affine
        if data is not None:
            nx, ny, nz = data.shape[:3]
            self._center = (nx//2, ny//2, nz//2)
            self._slice_idx = {'axial': nz//2, 'sagittal': nx//2, 'coronal': ny//2}
            self.cam_dist = max(nx, ny, nz) * 1.6
        self._rebuild_slice()

    def set_overlay(self, data, cmap='hot', alpha=0.65):
        """Set overlay volume (stat map, activation, atlas)."""
        self._overlay_vol = data.astype(np.float32) if data is not None else None
        self._overlay_cmap = cmap
        self._overlay_alpha = alpha
        self._rebuild_slice()

    def set_surface_data(self, verts, normals, name="", color_hex=0xD0C0B0):
        """Set surface mesh from vertex/normal arrays."""
        self._surf_verts = verts
        self._surf_normals = normals
        self._surf_name = name
        self._rebuild_surface(color_hex)

    def set_slice_axis(self, axis, index=None):
        self._slice_axis = axis
        if index is not None:
            self._slice_idx[axis] = index
        self._show_3plane = False
        self._rebuild_slice()

    def add_overlay(self, name, fn):
        self._overlays[name] = fn

    def remove_overlay(self, name):
        self._overlays.pop(name, None)

    def screenshot(self, path):
        if self._frame: self._frame.save(path)

    # ── Surface mesh building ──

    def _rebuild_surface(self, color_hex=0xD0C0B0):
        if not self._gl_ready or self._surf_verts is None: return
        verts = self._surf_verts
        norms = self._surf_normals
        r, g, b = _hex(color_hex)
        cx, cy, cz = self._center

        # Build triangle list from point cloud using nearest-neighbor triangulation
        # For a point cloud, we create small oriented quads at each vertex
        all_v = []
        size = 1.2  # half-size of each surface patch

        for i in range(len(verts)):
            vx, vy, vz = verts[i] - np.array([cx, cy, cz], dtype=np.float32)
            nx_, ny_, nz_ = norms[i]

            # Create a small oriented quad
            # Find two tangent vectors
            n_vec = np.array([nx_, ny_, nz_])
            if abs(nx_) < 0.9:
                t1 = np.cross(n_vec, [1, 0, 0])
            else:
                t1 = np.cross(n_vec, [0, 1, 0])
            t1 = t1 / (np.linalg.norm(t1) + 1e-9) * size
            t2 = np.cross(n_vec, t1)
            t2 = t2 / (np.linalg.norm(t2) + 1e-9) * size

            p = np.array([vx, vy, vz])
            p00 = p - t1 - t2
            p10 = p + t1 - t2
            p11 = p + t1 + t2
            p01 = p - t1 + t2

            for tri_p in (p00, p10, p11, p00, p11, p01):
                all_v.extend(tri_p.tolist())
                all_v.extend([nx_, ny_, nz_])
                all_v.extend([r, g, b])

        if not all_v:
            self._surf_vao = None; self._surf_n = 0; return

        data = np.array(all_v, dtype='f4').tobytes()
        if self._surf_vao:
            try: self._surf_vao.release()
            except: pass
        vbo = self.ctx.buffer(data)
        self._surf_vao = self.ctx.vertex_array(
            self.prog_brain, [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        self._surf_n = len(all_v) // 9

    # ── Slice mesh building ──

    def _rebuild_slice(self):
        if not self._gl_ready or self._vol is None: return

        all_v = []
        axes_to_draw = ['axial', 'sagittal', 'coronal'] if self._show_3plane else [self._slice_axis]

        nx, ny, nz = self._vol.shape[:3]
        cx, cy, cz = self._center
        vmin = float(self._vol[self._vol > 0].min()) if np.any(self._vol > 0) else 0
        vmax = float(self._vol.max())
        vrange = vmax - vmin if vmax > vmin else 1.0

        for axis in axes_to_draw:
            idx = self._slice_idx.get(axis, 45)
            if axis == 'axial':
                idx = max(0, min(nz-1, idx))
                sl = self._vol[:, :, idx]
                ov = self._overlay_vol[:, :, idx] if self._overlay_vol is not None else None
                # Quad corners in volume space, centered
                corners = [
                    (0-cx, 0-cy, idx-cz),
                    (nx-cx, 0-cy, idx-cz),
                    (nx-cx, ny-cy, idx-cz),
                    (0-cx, ny-cy, idx-cz),
                ]
                norm = [0, 0, 1]
            elif axis == 'sagittal':
                idx = max(0, min(nx-1, idx))
                sl = self._vol[idx, :, :]
                ov = self._overlay_vol[idx, :, :] if self._overlay_vol is not None else None
                corners = [
                    (idx-cx, 0-cy, 0-cz),
                    (idx-cx, ny-cy, 0-cz),
                    (idx-cx, ny-cy, nz-cz),
                    (idx-cx, 0-cy, nz-cz),
                ]
                norm = [1, 0, 0]
            else:  # coronal
                idx = max(0, min(ny-1, idx))
                sl = self._vol[:, idx, :]
                ov = self._overlay_vol[:, idx, :] if self._overlay_vol is not None else None
                corners = [
                    (0-cx, idx-cy, 0-cz),
                    (nx-cx, idx-cy, 0-cz),
                    (nx-cx, idx-cy, nz-cz),
                    (0-cx, idx-cy, nz-cz),
                ]
                norm = [0, 1, 0]

            # Rasterize the slice into a textured quad using per-pixel color
            sh = sl.shape
            step = max(1, min(sh[0], sh[1]) // 90)  # resolution control

            for i in range(0, sh[0]-step, step):
                for j in range(0, sh[1]-step, step):
                    val = float(sl[i, j])
                    if val <= 0: continue

                    # Base color from structural volume
                    t = (val - vmin) / vrange
                    r, g, b = _cmap_lookup(self._cmap, t)

                    # Blend overlay if present
                    if ov is not None:
                        ov_val = float(ov[i, j])
                        if abs(ov_val) > self._threshold:
                            ov_t = min(1.0, abs(ov_val) / max(abs(ov.max()), abs(ov.min()), 1))
                            or_, og, ob = _cmap_lookup(self._overlay_cmap, ov_t)
                            a = self._overlay_alpha
                            r = r * (1-a) + or_ * a
                            g = g * (1-a) + og * a
                            b = b * (1-a) + ob * a

                    # Compute pixel position by interpolating corners
                    u0 = i / sh[0]; u1 = (i+step) / sh[0]
                    v0 = j / sh[1]; v1 = (j+step) / sh[1]

                    def _interp(u, v):
                        # Bilinear interp of corners
                        p0 = [corners[0][k]*(1-u)*(1-v) + corners[1][k]*u*(1-v) +
                              corners[2][k]*u*v + corners[3][k]*(1-u)*v for k in range(3)]
                        return p0

                    p00 = _interp(u0, v0)
                    p10 = _interp(u1, v0)
                    p11 = _interp(u1, v1)
                    p01 = _interp(u0, v1)

                    for tri_p in (p00, p10, p11, p00, p11, p01):
                        all_v.extend(tri_p)
                        all_v.extend(norm)
                        all_v.extend([r, g, b])

        if not all_v:
            self._slice_vao = None; self._slice_n = 0; return

        data = np.array(all_v, dtype='f4').tobytes()
        if self._slice_vao:
            try: self._slice_vao.release()
            except: pass
        vbo = self.ctx.buffer(data)
        self._slice_vao = self.ctx.vertex_array(
            self.prog_slice, [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        self._slice_n = len(all_v) // 9

    # ── Rendering ──

    def _tick(self):
        if not self._dragging and self._view_mode == 'surface':
            self.auto_rot += 0.004
        self._render(); self.update()

    def _render(self):
        if not self._gl_ready: return
        w, h = max(self.width(), 320), max(self.height(), 200)
        self._resize_fbo(w, h); self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0, 0, 0, 0)

        # Build matrices (row-major numpy, _gl() converts to column-major bytes)
        proj = _perspective(_radians(45), w/h, 0.1, 1000.0)
        ry = self.rot_y + self.auto_rot
        ex = math.sin(ry) * math.cos(self.rot_x) * self.cam_dist
        ey = math.sin(self.rot_x) * self.cam_dist
        ez = math.cos(ry) * math.cos(self.rot_x) * self.cam_dist
        view = _look_at((ex, ey, ez), (0, 0, 0), (0, 1, 0))
        eye_bytes = _vec3_bytes(ex, ey, ez)
        vp_gl = _gl(proj @ view)  # row-major multiply, then to column-major bytes
        light = _normalize3(0.4, 0.7, 0.5)

        # Draw surface
        if self._surf_vao and self._surf_n > 0 and self._view_mode == 'surface':
            self.prog_brain['mvp'].write(vp_gl)
            self.prog_brain['model'].write(_IDENTITY_GL)
            self.prog_brain['light_dir'].write(light)
            self.prog_brain['ambient'].write(_vec3_bytes(0.42, 0.40, 0.44))
            self.prog_brain['view_pos'].write(eye_bytes)
            self.prog_brain['alpha'].value = 0.92
            self._surf_vao.render(moderngl.TRIANGLES)

        # Draw slices
        if self._slice_vao and self._slice_n > 0 and self._view_mode == 'slice':
            self.prog_slice['mvp'].write(vp_gl)
            self.prog_slice['model'].write(_IDENTITY_GL)
            self.prog_slice['light_dir'].write(light)
            self.prog_slice['ambient'].write(_vec3_bytes(0.5, 0.5, 0.5))
            self.prog_slice['view_pos'].write(eye_bytes)
            self.prog_slice['alpha'].value = 1.0
            self._slice_vao.render(moderngl.TRIANGLES)

        # Draw markers as spheres
        if self._markers:
            cx, cy, cz = self._center
            for mx, my, mz, mcol, mlbl in self._markers:
                sp = _make_sphere(2.0, 10, 6, mcol)
                t_mat = _translate_mat4(mx-cx, my-cy, mz-cz)
                sp_t = _transform_verts(sp, t_mat)
                if sp_t:
                    data = np.array(sp_t, dtype='f4').tobytes()
                    vbo = self.ctx.buffer(data)
                    vao = self.ctx.vertex_array(
                        self.prog_brain, [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
                    self.prog_brain['mvp'].write(vp_gl)
                    self.prog_brain['model'].write(_IDENTITY_GL)
                    self.prog_brain['light_dir'].write(light)
                    self.prog_brain['ambient'].write(_vec3_bytes(0.5, 0.5, 0.5))
                    self.prog_brain['view_pos'].write(eye_bytes)
                    self.prog_brain['alpha'].value = 1.0
                    vao.render(moderngl.TRIANGLES)
                    vbo.release(); vao.release()

        raw = self.fbo.color_attachments[0].read()
        self._frame = QImage(raw, w, h, w*4, QImage.Format_RGBA8888).mirrored(False, True)

    def paintEvent(self, event):
        self._ensure_gl()
        # Deferred rebuild: GL wasn't ready when data was loaded
        if self._vol is not None and self._slice_n == 0:
            self._rebuild_slice()
        if self._surf_verts is not None and self._surf_n == 0:
            self._rebuild_surface()
        # Force a render if we have data but no frame yet
        if self._frame is None and (self._slice_n > 0 or self._surf_n > 0):
            self._render()
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self._frame and not self._frame.isNull():
            p.drawImage(0, 0, self._frame)

        # ── HUD ──
        if self._vol is not None:
            shape = self._vol.shape[:3]
            p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 195))
            p.drawRoundedRect(12, 12, 220, 72, 8, 8)
            p.setFont(QFont("Consolas", 9, QFont.Bold)); p.setPen(QColor(50, 55, 70))
            p.drawText(20, 22, 200, 16, Qt.AlignVCenter, self._vol_name or "Brain Volume")
            p.setFont(QFont("Consolas", 9)); p.setPen(QColor(80, 90, 110))
            p.drawText(20, 38, 200, 14, Qt.AlignVCenter,
                f"{shape[0]}×{shape[1]}×{shape[2]} voxels")
            vsize = f"{abs(self._affine[0,0]):.1f}×{abs(self._affine[1,1]):.1f}×{abs(self._affine[2,2]):.1f}"
            p.drawText(20, 52, 200, 14, Qt.AlignVCenter, f"Voxel: {vsize} mm")
            # Slice info
            axis = self._slice_axis
            idx = self._slice_idx.get(axis, 0)
            p.drawText(20, 66, 200, 14, Qt.AlignVCenter,
                f"Slice: {axis} #{idx}")

        # Slice position indicator
        if self._view_mode == 'slice' and self._vol is not None:
            axis = self._slice_axis
            idx = self._slice_idx.get(axis, 0)
            shape = self._vol.shape[:3]
            maxdim = {'axial': shape[2], 'sagittal': shape[0], 'coronal': shape[1]}[axis]
            # Mini navigator bar
            bar_w = 150; bar_h = 8
            bx = w - bar_w - 16; by = h - 30
            p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 160))
            p.drawRoundedRect(bx-4, by-12, bar_w+8, 32, 6, 6)
            p.setBrush(QColor(180, 190, 210))
            p.drawRoundedRect(bx, by, bar_w, bar_h, 3, 3)
            # Position indicator
            pos = idx / max(maxdim-1, 1) * bar_w
            p.setBrush(QColor(70, 130, 200))
            p.drawRoundedRect(int(bx + pos - 3), by-2, 6, bar_h+4, 3, 3)
            p.setFont(QFont("Consolas", 7)); p.setPen(QColor(80, 90, 110))
            p.drawText(bx, by+10, bar_w, 12, Qt.AlignCenter,
                f"{axis} {idx}/{maxdim-1}")

        # Crosshair on 2D slice rendering
        if self._show_crosshair and self._view_mode == 'slice' and not self._show_3plane:
            pass  # Crosshair handled in 3D space

        # Marker labels
        p.setFont(QFont("Consolas", 8, QFont.Bold))
        for mx, my, mz, mcol, mlbl in self._markers:
            if mlbl:
                p.setPen(QColor((mcol>>16)&0xFF, (mcol>>8)&0xFF, mcol&0xFF))
                p.drawText(w//2, 80 + self._markers.index((mx,my,mz,mcol,mlbl))*14,
                    200, 14, Qt.AlignVCenter, mlbl)

        # Custom overlays
        for name, fn in self._overlays.items():
            try: fn(p, w, h)
            except: pass

        p.end()

    # ── Mouse interaction ──

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True; self._lmx = e.x(); self._lmy = e.y()
        e.accept()

    def mouseReleaseEvent(self, e):
        self._dragging = False; e.accept()

    def mouseMoveEvent(self, e):
        if self._dragging:
            self.rot_y += (e.x() - self._lmx) * 0.008
            self.rot_x = max(-1.4, min(1.4, self.rot_x + (e.y() - self._lmy) * 0.008))
            self._lmx = e.x(); self._lmy = e.y()
        e.accept()

    def wheelEvent(self, e):
        self.cam_dist = max(20, min(500, self.cam_dist - e.angleDelta().y() * 0.15))
        e.accept()


# ═══════════════════════════════════════════════════════════════
#  NEUROLAB — MAIN API SINGLETON
# ═══════════════════════════════════════════════════════════════

class _Signals(QObject):
    status = Signal(str)
    compute_done = Signal(str)

class NeuroLab:
    """
    Main brain imaging API. Registered as `brain` in the namespace.

    QUICK REFERENCE (for LLM context):
        brain.load(name)                     — load preset: mni152/fsaverage
        brain.load_nifti(path)               — load NIfTI volume (.nii/.nii.gz)
        brain.load_surface(path)             — load surface mesh
        brain.slice(axis, idx)               — set slice: axial/sagittal/coronal + index
        brain.slice_3plane(x,y,z)            — show all three orthogonal slices
        brain.view_surface()                 — switch to 3D surface view
        brain.view_slice()                   — switch to slice view
        brain.colormap(name)                 — set colormap: bone/hot/cool/gray/viridis/rdbu/plasma
        brain.opacity(val)                   — set overlay opacity 0-1
        brain.threshold(val)                 — set overlay threshold
        brain.overlay_nifti(path)            — overlay statistical map
        brain.overlay_activation(regions)    — show synthetic activation
        brain.overlay_atlas(name)            — overlay parcellation atlas
        brain.overlay_roi(region)            — highlight named brain region
        brain.clear_overlay()                — remove all overlays
        brain.add_marker(x,y,z,label,color)  — add 3D marker point
        brain.clear_markers()                — remove markers
        brain.smooth(fwhm)                   — smooth volume
        brain.resample(mm)                   — resample to isotropic
        brain.segment()                      — tissue segmentation
        brain.voxel(x,y,z)                   — query voxel value
        brain.roi_stats(mask_or_region)      — compute ROI statistics
        brain.histogram(nbins)               — overlay intensity histogram
        brain.screenshot(path)               — save view as PNG
        brain.export_nifti(path)             — save volume as NIfTI
        brain.info                           — dict of volume properties
        brain.viewer                         — the BrainViewer widget
        brain.log(msg)                       — append to log panel
    """

    def __init__(self, viewer, log_widget=None):
        self.viewer = viewer
        self._log = log_widget
        self.info = {}
        self._vol = None
        self._affine = np.eye(4)
        self._overlay = None
        self._signals = _Signals()

    def log(self, msg):
        if self._log: self._log.append(f"[brain] {msg}")
        print(f"[brain] {msg}")

    # ── Loading ────────────────────────────────────────────────

    def load(self, name):
        """Load a preset brain template by name."""
        key = name.lower().replace(' ', '').replace('-', '').replace('_', '')
        if 'mni' in key or '152' in key:
            self.log("Generating MNI152 phantom (91×109×91)...")
            vol, affine = _generate_mni152_phantom()
            self._vol = vol; self._affine = affine
            self.viewer.set_volume(vol, affine, "MNI152 (2mm)")
            self._update_info("MNI152")
            # Also generate surface
            verts, norms = _generate_surface_mesh(vol, threshold=55, step=2)
            self.viewer.set_surface_data(verts, norms, "MNI152 surface")
            self.log(f"Loaded MNI152: {vol.shape}, surface: {len(verts)} vertices")
            return self
        elif 'fsaverage' in key or 'freesurfer' in key:
            self.log("Generating fsaverage phantom...")
            vol, affine = _generate_mni152_phantom(shape=(80, 100, 80))
            self._vol = vol; self._affine = affine
            self.viewer.set_volume(vol, affine, "fsaverage")
            verts, norms = _generate_surface_mesh(vol, threshold=55, step=2)
            self.viewer.set_surface_data(verts, norms, "fsaverage surface")
            self._update_info("fsaverage")
            return self
        else:
            # Try to load as NiBabel dataset
            try:
                self.log(f"Attempting nilearn dataset: {name}")
                from nilearn import datasets
                if hasattr(datasets, f'fetch_{name}'):
                    ds = getattr(datasets, f'fetch_{name}')()
                    if hasattr(ds, 'maps'):
                        return self.load_nifti(ds.maps)
                    elif hasattr(ds, 'filenames'):
                        return self.load_nifti(ds.filenames[0])
            except Exception as ex:
                self.log(f"Could not fetch dataset: {ex}")
            self.log(f"Unknown preset: {name}. Available: mni152, fsaverage")
            return self

    def load_nifti(self, path):
        """Load a NIfTI volume from file path."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.log(f"File not found: {path}"); return self

        # Try nibabel first, fall back to built-in parser
        try:
            import nibabel as nib
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            affine = img.affine
            self.log(f"Loaded with nibabel: {os.path.basename(path)}")
        except ImportError:
            result = _read_nifti(path)
            if result is None:
                self.log(f"Failed to parse NIfTI: {path}"); return self
            data = result['data']; affine = result['affine']
            self.log(f"Loaded with built-in parser: {os.path.basename(path)}")

        # Handle 4D (take first volume)
        if data.ndim == 4:
            self.log(f"4D volume ({data.shape[3]} timepoints), using first volume")
            data = data[:,:,:,0]

        self._vol = data; self._affine = affine
        self.viewer.set_volume(data, affine, os.path.basename(path))
        self._update_info(os.path.basename(path))

        # Generate surface
        thr = float(np.percentile(data[data > 0], 30)) if np.any(data > 0) else 0
        verts, norms = _generate_surface_mesh(data, threshold=thr, step=2)
        if len(verts) > 0:
            self.viewer.set_surface_data(verts, norms, os.path.basename(path))
            self.log(f"Surface extracted: {len(verts)} vertices")

        return self

    def load_surface(self, path):
        """Load a FreeSurfer surface file."""
        path = os.path.expanduser(path)
        try:
            import nibabel as nib
            surf = nib.freesurfer.read_geometry(path)
            verts, faces = surf[0], surf[1]
            # Compute normals
            norms = np.zeros_like(verts)
            for f in faces:
                v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
                n = np.cross(v1-v0, v2-v0)
                norms[f[0]] += n; norms[f[1]] += n; norms[f[2]] += n
            lens = np.linalg.norm(norms, axis=1, keepdims=True) + 1e-9
            norms = norms / lens
            self.viewer.set_surface_data(verts, norms, os.path.basename(path))
            self.viewer._view_mode = 'surface'
            self.log(f"Loaded surface: {os.path.basename(path)} ({len(verts)} vertices)")
        except ImportError:
            self.log("nibabel required for FreeSurfer surfaces: pip install nibabel")
        except Exception as ex:
            self.log(f"Surface load error: {ex}")
        return self

    def _update_info(self, name=""):
        """Update info dict from current volume."""
        if self._vol is None: return
        shape = self._vol.shape[:3]
        mask = self._vol > 0
        self.info = {
            'name': name,
            'shape': shape,
            'voxel_size': tuple(abs(self._affine[i, i]) for i in range(3)),
            'affine': self._affine.copy(),
            'dtype': str(self._vol.dtype),
            'range': (float(self._vol.min()), float(self._vol.max())),
            'n_voxels': int(np.prod(shape)),
            'n_nonzero': int(mask.sum()),
            'mean': float(self._vol[mask].mean()) if mask.any() else 0,
            'std': float(self._vol[mask].std()) if mask.any() else 0,
        }

    # ── Slice navigation ──────────────────────────────────────

    def slice(self, axis='axial', idx=None, **kwargs):
        """Set slice view. axis: 'axial'/'sagittal'/'coronal'. idx or x=/y=/z= keyword."""
        axis = axis.lower()
        if axis not in ('axial', 'sagittal', 'coronal'):
            self.log(f"Unknown axis: {axis}. Use axial/sagittal/coronal"); return self
        if idx is None:
            idx = kwargs.get('x', kwargs.get('y', kwargs.get('z', None)))
        if idx is None and self._vol is not None:
            idx = self._vol.shape[{'axial':2,'sagittal':0,'coronal':1}[axis]] // 2
        self.viewer.set_slice_axis(axis, idx)
        self.viewer._view_mode = 'slice'
        self.log(f"Slice: {axis} #{idx}")
        return self

    def slice_3plane(self, x=None, y=None, z=None):
        """Show all three orthogonal planes simultaneously."""
        if self._vol is None: return self
        shape = self._vol.shape[:3]
        if x is not None: self.viewer._slice_idx['sagittal'] = x
        if y is not None: self.viewer._slice_idx['coronal'] = y
        if z is not None: self.viewer._slice_idx['axial'] = z
        self.viewer._show_3plane = True
        self.viewer._view_mode = 'slice'
        self.viewer._rebuild_slice()
        self.log(f"3-plane view: sag={self.viewer._slice_idx['sagittal']}, "
                 f"cor={self.viewer._slice_idx['coronal']}, "
                 f"axi={self.viewer._slice_idx['axial']}")
        return self

    def view_surface(self):
        """Switch to 3D surface rendering mode."""
        self.viewer._view_mode = 'surface'
        self.viewer.auto_rot = 0.0
        self.log("Switched to surface view")
        return self

    def view_slice(self):
        """Switch to slice viewing mode."""
        self.viewer._view_mode = 'slice'
        self.log("Switched to slice view")
        return self

    # ── Visual settings ───────────────────────────────────────

    def colormap(self, name):
        """Set colormap: bone/hot/cool/gray/viridis/rdbu/plasma."""
        if name not in COLORMAPS:
            self.log(f"Unknown colormap: {name}. Available: {', '.join(COLORMAPS.keys())}")
            return self
        self.viewer._cmap = name
        self.viewer._rebuild_slice()
        self.log(f"Colormap: {name}")
        return self

    def opacity(self, val):
        """Set overlay opacity (0-1)."""
        self.viewer._overlay_alpha = max(0, min(1, val))
        self.viewer._rebuild_slice()
        return self

    def threshold(self, val):
        """Set overlay threshold (voxels below this are hidden)."""
        self.viewer._threshold = float(val)
        self.viewer._rebuild_slice()
        self.log(f"Threshold: {val}")
        return self

    # ── Overlays ──────────────────────────────────────────────

    def overlay_nifti(self, path):
        """Overlay a statistical map (z-stat, t-stat, etc.)."""
        path = os.path.expanduser(path)
        try:
            import nibabel as nib
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
        except ImportError:
            result = _read_nifti(path)
            if result is None:
                self.log(f"Failed to load overlay: {path}"); return self
            data = result['data']

        if data.ndim == 4: data = data[:,:,:,0]

        # Resample to match base volume if needed
        if self._vol is not None and data.shape != self._vol.shape[:3]:
            zoom_factors = [s/d for s, d in zip(self._vol.shape[:3], data.shape[:3])]
            data = _zoom_linear_3d(data, zoom_factors)

        self._overlay = data
        self.viewer.set_overlay(data, self.viewer._overlay_cmap, self.viewer._overlay_alpha)
        self.log(f"Overlay: {os.path.basename(path)} ({data.shape})")
        return self

    def overlay_activation(self, regions=None):
        """Generate and overlay a synthetic activation map.
        regions: list of region names, e.g. ['precentral', 'broca', 'hippocampus']"""
        if self._vol is None: self.log("Load a brain volume first"); return self
        if regions is None: regions = ['precentral', 'broca']
        act = _generate_activation_map(self._vol, regions)
        self._overlay = act
        self.viewer.set_overlay(act, 'hot', 0.65)
        self.log(f"Activation overlay: {', '.join(regions)}")
        return self

    def overlay_atlas(self, name='harvard-oxford'):
        """Overlay a parcellation atlas (synthetic demonstration)."""
        if self._vol is None: self.log("Load a brain volume first"); return self
        shape = self._vol.shape[:3]
        atlas = np.zeros(shape, dtype=np.float32)
        # Generate labeled regions
        for i, (rname, rdata) in enumerate(BRAIN_REGIONS.items()):
            cx, cy, cz = rdata['center']
            sx = cx/50*shape[0]; sy = cy/80*shape[1]; sz = cz/60*shape[2]
            x = np.arange(shape[0]) - sx
            y = np.arange(shape[1]) - sy
            z = np.arange(shape[2]) - sz
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            r = np.sqrt(X**2/100 + Y**2/100 + Z**2/64)
            mask = (r < 1.0) & (self._vol > 20)
            atlas[mask] = (i + 1) / len(BRAIN_REGIONS)
        self._overlay = atlas
        self.viewer.set_overlay(atlas, 'viridis', 0.5)
        self.log(f"Atlas overlay: {name} ({len(BRAIN_REGIONS)} regions)")
        return self

    def overlay_roi(self, region='hippocampus'):
        """Highlight a specific named brain region."""
        if self._vol is None: self.log("Load a brain volume first"); return self
        reg = BRAIN_REGIONS.get(region.lower())
        if not reg:
            self.log(f"Unknown region: {region}. Available: {', '.join(BRAIN_REGIONS.keys())}")
            return self
        act = _generate_activation_map(self._vol, [region.lower()], noise=0.0)
        self._overlay = act
        r, g, b = _hex(reg['color'])
        # Use custom colormap for this region
        self.viewer.set_overlay(act, 'hot', 0.7)
        # Add marker at region center
        cx, cy, cz = reg['center']
        sx = cx/50*self._vol.shape[0]
        sy = cy/80*self._vol.shape[1]
        sz = cz/60*self._vol.shape[2]
        self.add_marker(sx, sy, sz, f"{reg['desc']}", reg['color'])
        self.log(f"ROI: {reg['desc']}")
        return self

    def clear_overlay(self):
        """Remove overlay and markers."""
        self._overlay = None
        self.viewer.set_overlay(None)
        self.viewer._markers.clear()
        self.viewer._rebuild_slice()
        self.log("Overlay cleared")
        return self

    # ── Markers ───────────────────────────────────────────────

    def add_marker(self, x, y, z, label="", color=0xFF4444):
        """Add a 3D marker point at voxel coordinates."""
        self.viewer._markers.append((x, y, z, color, label))
        return self

    def clear_markers(self):
        self.viewer._markers.clear()
        return self

    # ── Analysis ──────────────────────────────────────────────

    def voxel(self, x, y, z):
        """Query voxel value at integer coordinates."""
        if self._vol is None: return None
        shape = self._vol.shape[:3]
        x, y, z = int(x), int(y), int(z)
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            val = float(self._vol[x, y, z])
            self.log(f"Voxel ({x},{y},{z}) = {val:.2f}")
            return val
        self.log(f"Out of bounds: ({x},{y},{z})")
        return None

    def smooth(self, fwhm=6.0):
        """Apply Gaussian smoothing to the volume.
        fwhm: full width at half maximum in mm."""
        if self._vol is None: return self
        voxel_size = abs(self._affine[0, 0])
        sigma = fwhm / (2.355 * max(voxel_size, 0.5))
        self._vol = _gaussian_smooth_3d(self._vol, sigma=sigma)
        self.viewer.set_volume(self._vol, self._affine, self.viewer._vol_name + " (smoothed)")
        self._update_info(self.viewer._vol_name)
        self.log(f"Smoothed: FWHM={fwhm}mm (σ={sigma:.2f} voxels)")
        return self

    def resample(self, target_mm=2.0):
        """Resample volume to isotropic resolution."""
        if self._vol is None: return self
        current_voxel = [abs(self._affine[i, i]) for i in range(3)]
        zoom_factors = [cv / target_mm for cv in current_voxel]
        self._vol = _zoom_linear_3d(self._vol, zoom_factors)
        # Update affine
        new_affine = self._affine.copy()
        for i in range(3):
            new_affine[i, i] = np.sign(self._affine[i, i]) * target_mm
        self._affine = new_affine
        self.viewer.set_volume(self._vol, self._affine, self.viewer._vol_name)
        self._update_info(self.viewer._vol_name)
        self.log(f"Resampled to {target_mm}mm isotropic: {self._vol.shape}")
        return self

    def segment(self):
        """Simple tissue segmentation into GM, WM, CSF using intensity thresholds.
        Creates brain.info['segmentation'] with labeled volume."""
        if self._vol is None: return self
        seg = np.zeros_like(self._vol, dtype=np.int8)
        mask = self._vol > 10
        p33 = np.percentile(self._vol[mask], 33)
        p66 = np.percentile(self._vol[mask], 66)
        seg[(self._vol > 10) & (self._vol <= p33)] = 1   # CSF
        seg[(self._vol > p33) & (self._vol <= p66)] = 2  # GM
        seg[(self._vol > p66)] = 3                        # WM
        self.info['segmentation'] = seg
        n_csf = int((seg == 1).sum())
        n_gm = int((seg == 2).sum())
        n_wm = int((seg == 3).sum())
        self.info['tissue_volumes'] = {'CSF': n_csf, 'GM': n_gm, 'WM': n_wm}
        self.log(f"Segmentation: CSF={n_csf}, GM={n_gm}, WM={n_wm} voxels")

        # Overlay segmentation
        seg_float = seg.astype(np.float32) / 3.0
        self.viewer.set_overlay(seg_float, 'viridis', 0.4)
        return self

    def roi_stats(self, region_or_mask):
        """Compute statistics for a brain region or binary mask.
        region_or_mask: string (region name) or 3D binary array."""
        if self._vol is None: return {}
        if isinstance(region_or_mask, str):
            reg = BRAIN_REGIONS.get(region_or_mask.lower())
            if not reg:
                self.log(f"Unknown region: {region_or_mask}"); return {}
            # Generate mask from region
            shape = self._vol.shape[:3]
            cx, cy, cz = reg['center']
            sx = cx/50*shape[0]; sy = cy/80*shape[1]; sz = cz/60*shape[2]
            x = np.arange(shape[0]) - sx
            y = np.arange(shape[1]) - sy
            z = np.arange(shape[2]) - sz
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            r = np.sqrt(X**2/100 + Y**2/100 + Z**2/49)
            mask = (r < 1.0) & (self._vol > 10)
            name = reg['desc']
        else:
            mask = region_or_mask.astype(bool)
            name = "custom ROI"

        vals = self._vol[mask]
        stats = {
            'name': name,
            'n_voxels': int(mask.sum()),
            'mean': float(vals.mean()) if len(vals) > 0 else 0,
            'std': float(vals.std()) if len(vals) > 0 else 0,
            'min': float(vals.min()) if len(vals) > 0 else 0,
            'max': float(vals.max()) if len(vals) > 0 else 0,
            'median': float(np.median(vals)) if len(vals) > 0 else 0,
        }
        self.info['roi_stats'] = stats
        self.log(f"ROI {name}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, n={stats['n_voxels']}")
        return stats

    def histogram(self, nbins=50, width=340, height=160):
        """Overlay an intensity histogram."""
        if self._vol is None: return self
        mask = self._vol > 0
        vals = self._vol[mask].flatten()
        hist, edges = np.histogram(vals, bins=nbins)
        hmax = float(hist.max())

        def draw_hist(painter, w, h):
            ox, oy = w - width - 16, h - height - 50
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 210))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, "Intensity Histogram")
            pad = 30
            aw = width - pad*2; ah = height - 50
            ax0 = ox + pad; ay1 = oy + height - 24
            # Bars
            bw = max(1, aw // nbins)
            for i, count in enumerate(hist):
                bh = int(count / hmax * ah) if hmax > 0 else 0
                t = i / nbins
                r, g, b = _cmap_lookup('viridis', t)
                painter.setBrush(QColor(int(r*255), int(g*255), int(b*255), 180))
                painter.setPen(Qt.NoPen)
                painter.drawRect(ax0 + i*bw, ay1 - bh, max(bw-1, 1), bh)
            # Axis label
            painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(100, 110, 130))
            painter.drawText(ax0, ay1+4, aw, 12, Qt.AlignLeft, f"{edges[0]:.0f}")
            painter.drawText(ax0, ay1+4, aw, 12, Qt.AlignRight, f"{edges[-1]:.0f}")

        self.viewer.add_overlay('histogram', draw_hist)
        self.log(f"Histogram overlay: {nbins} bins")
        return self

    # ── QPainter overlays ─────────────────────────────────────

    def overlay(self, name, fn):
        """Add custom QPainter overlay: fn(painter, width, height)."""
        self.viewer.add_overlay(name, fn); return self

    def remove_overlay_painter(self, name):
        self.viewer.remove_overlay(name); return self

    # ── Connectivity (nilearn-powered) ────────────────────────

    def connectivity(self, seeds=None, atlas='aal', n_components=5):
        """Compute seed-based or atlas-based connectivity (requires nilearn).
        seeds: list of (x,y,z) MNI coordinates for seed-based.
        Returns connectivity matrix in brain.info['connectivity']."""
        if self._vol is None: return self
        try:
            from nilearn import connectome
            from nilearn.maskers import NiftiSpheresMasker
            import nibabel as nib
            # This requires a 4D timeseries — show placeholder if 3D
            if self._vol.ndim < 4:
                self.log("Connectivity requires 4D data. Generating demo correlation matrix...")
                n = len(BRAIN_REGIONS)
                np.random.seed(42)
                corr = np.corrcoef(np.random.randn(n, 100))
                self.info['connectivity'] = corr
                self.info['connectivity_labels'] = list(BRAIN_REGIONS.keys())
                self._overlay_conn_matrix(corr, list(BRAIN_REGIONS.keys()))
                return self
        except ImportError:
            self.log("nilearn required for connectivity: pip install nilearn")
            # Generate demo matrix anyway
            n = len(BRAIN_REGIONS)
            np.random.seed(42)
            corr = np.corrcoef(np.random.randn(n, 100))
            self.info['connectivity'] = corr
            self.info['connectivity_labels'] = list(BRAIN_REGIONS.keys())
            self._overlay_conn_matrix(corr, list(BRAIN_REGIONS.keys()))
        return self

    def _overlay_conn_matrix(self, corr, labels, width=280, height=280):
        """Overlay a connectivity matrix."""
        n = len(labels)
        def draw_conn(painter, w, h):
            ox, oy = w - width - 16, 90
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, "Connectivity Matrix")
            pad = 40
            cell = min((width - pad*2) // n, (height - pad - 20) // n)
            mx0 = ox + pad; my0 = oy + 24
            for i in range(n):
                for j in range(n):
                    val = corr[i, j]
                    # RdBu colormap
                    t = (val + 1) / 2  # [-1,1] → [0,1]
                    r, g, b = _cmap_lookup('rdbu', t)
                    painter.setBrush(QColor(int(r*255), int(g*255), int(b*255)))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(mx0 + j*cell, my0 + i*cell, cell-1, cell-1)
            # Labels
            painter.setFont(QFont("Consolas", 5)); painter.setPen(QColor(80, 90, 110))
            for i, lbl in enumerate(labels):
                short = lbl[:6]
                painter.save()
                painter.translate(mx0 + i*cell + cell//2, my0 - 2)
                painter.rotate(-45)
                painter.drawText(0, 0, short)
                painter.restore()

        self.viewer.add_overlay('connectivity', draw_conn)
        self.log(f"Connectivity matrix: {n}×{n}")

    # ── GLM (nilearn-powered) ─────────────────────────────────

    def compute_glm(self, events=None, tr=2.0, n_scans=200):
        """Compute first-level GLM (requires nilearn).
        events: list of dicts with 'onset', 'duration', 'trial_type'.
        If None, generates a demo block design."""
        self.log("Computing GLM...")
        if events is None:
            # Generate demo block design
            events = []
            for i in range(0, n_scans*int(tr), 30):
                events.append({'onset': i, 'duration': 15, 'trial_type': 'task'})

        try:
            from nilearn.glm.first_level import make_first_level_design_matrix
            import pandas as pd
            frame_times = np.arange(n_scans) * tr
            events_df = pd.DataFrame(events)
            dm = make_first_level_design_matrix(frame_times, events_df)
            self.info['glm_design'] = dm
            self.info['glm_events'] = events
            self._overlay_design_matrix(dm)
            self.log(f"GLM design: {dm.shape[0]} timepoints, {dm.shape[1]} regressors")
        except ImportError:
            self.log("nilearn required for GLM. Showing demo activation instead.")
            self.overlay_activation(['precentral', 'broca', 'wernicke'])
        return self

    def _overlay_design_matrix(self, dm, width=220, height=180):
        """Overlay the GLM design matrix."""
        cols = list(dm.columns)[:6]
        n_tp = len(dm)
        vals = dm[cols].values

        def draw_dm(painter, w, h):
            ox, oy = w - width - 16, h - height - 50
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 210))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, "Design Matrix")
            pad = 12
            cw = (width - pad*2) // len(cols)
            ch = (height - 40) / n_tp
            ax0 = ox + pad; ay0 = oy + 22
            vmax = abs(vals).max() if abs(vals).max() > 0 else 1
            for ci, col in enumerate(cols):
                col_data = vals[:, ci]
                for ti in range(0, n_tp, max(1, n_tp//100)):
                    t = (col_data[ti] / vmax + 1) / 2
                    r, g, b = _cmap_lookup('rdbu', t)
                    painter.setBrush(QColor(int(r*255), int(g*255), int(b*255)))
                    painter.drawRect(int(ax0 + ci*cw), int(ay0 + ti*ch), cw-1, max(int(ch), 1))
            # Column labels
            painter.setFont(QFont("Consolas", 6)); painter.setPen(QColor(80, 90, 110))
            for ci, col in enumerate(cols):
                painter.drawText(ax0 + ci*cw, oy + height - 14, cw, 12, Qt.AlignCenter, col[:8])

        self.viewer.add_overlay('design_matrix', draw_dm)

    # ── EEG/MEG (MNE-powered) ─────────────────────────────────

    def load_eeg(self, path):
        """Load EEG/MEG data (requires MNE-Python)."""
        try:
            import mne
            raw = mne.io.read_raw(path, preload=True, verbose=False)
            self.info['eeg_raw'] = raw
            self.info['eeg_channels'] = raw.ch_names
            self.info['eeg_sfreq'] = raw.info['sfreq']
            self.info['eeg_duration'] = raw.times[-1]
            self.log(f"EEG: {len(raw.ch_names)} channels, {raw.info['sfreq']}Hz, {raw.times[-1]:.1f}s")
        except ImportError:
            self.log("MNE-Python required for EEG: pip install mne")
        except Exception as ex:
            self.log(f"EEG load error: {ex}")
        return self

    def plot_eeg(self, channels=None, t_start=0, t_end=2, width=380, height=200):
        """Show EEG waveforms as overlay.
        channels: list of channel names, e.g. ['Cz', 'Fz', 'Pz']."""
        raw = self.info.get('eeg_raw')
        if raw is None:
            # Generate demo EEG
            self.log("No EEG loaded. Generating demo waveforms...")
            sfreq = 256
            t = np.arange(0, 2, 1/sfreq)
            channels = channels or ['Cz', 'Fz', 'Pz', 'O1']
            demo_data = {}
            np.random.seed(11)
            for i, ch in enumerate(channels):
                # Simulate alpha + noise + ERP
                sig = (np.sin(2*np.pi*10*t) * 5 * np.exp(-(t-0.5)**2/0.05) +
                       np.random.randn(len(t)) * 3 +
                       np.sin(2*np.pi*(8+i)*t) * 2)
                demo_data[ch] = sig
        else:
            import mne
            channels = channels or raw.ch_names[:4]
            sfreq = raw.info['sfreq']
            start_idx = int(t_start * sfreq)
            end_idx = int(t_end * sfreq)
            demo_data = {}
            for ch in channels:
                if ch in raw.ch_names:
                    idx = raw.ch_names.index(ch)
                    demo_data[ch] = raw.get_data(picks=[idx])[0, start_idx:end_idx]
            t = np.arange(len(list(demo_data.values())[0])) / sfreq + t_start

        eeg_colors = [0x2196F3, 0xE91E63, 0x4CAF50, 0xFF9800, 0x9C27B0, 0x00BCD4]

        def draw_eeg(painter, w, h):
            ox, oy = w - width - 16, h - height - 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, "EEG Waveforms")
            pad_l, pad_r, pad_t, pad_b = 40, 12, 24, 20
            aw = width - pad_l - pad_r; ah = height - pad_t - pad_b
            ax0 = ox + pad_l; ay0 = oy + pad_t; ay1 = oy + height - pad_b

            ch_h = ah / max(len(demo_data), 1)
            for ci, (ch, sig) in enumerate(demo_data.items()):
                cy = ay0 + ci * ch_h + ch_h / 2
                col = eeg_colors[ci % len(eeg_colors)]
                painter.setPen(QPen(QColor((col>>16)&0xFF, (col>>8)&0xFF, col&0xFF), 1.2))
                sig_min, sig_max = sig.min(), sig.max()
                sig_range = sig_max - sig_min if sig_max > sig_min else 1
                path = QPainterPath()
                for si in range(0, len(sig), max(1, len(sig)//aw)):
                    px = ax0 + si / len(sig) * aw
                    py = cy - (sig[si] - (sig_min+sig_max)/2) / sig_range * ch_h * 0.7
                    if si == 0: path.moveTo(px, py)
                    else: path.lineTo(px, py)
                painter.drawPath(path)
                # Channel label
                painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(80, 90, 110))
                painter.drawText(ox+6, int(cy-6), 32, 12, Qt.AlignVCenter, ch)
            # Time axis
            painter.setPen(QPen(QColor(180, 185, 200), 1))
            painter.drawLine(ax0, ay1, ax0+aw, ay1)
            painter.setFont(QFont("Consolas", 7)); painter.setPen(QColor(100, 110, 130))
            painter.drawText(ax0, ay1+2, aw, 12, Qt.AlignCenter, "Time (s)")

        self.viewer.add_overlay('eeg', draw_eeg)
        self.log(f"EEG overlay: {list(demo_data.keys())}")
        return self

    def topomap(self, time_point=0.1, width=160, height=180):
        """Show a scalp topography map overlay at a given time point."""
        # Generate demo topography
        self.log(f"Topomap at {time_point}s...")
        ch_positions = {
            'Fp1':(-0.3,0.8), 'Fp2':(0.3,0.8), 'F3':(-0.4,0.5), 'Fz':(0,0.5), 'F4':(0.4,0.5),
            'C3':(-0.5,0), 'Cz':(0,0), 'C4':(0.5,0),
            'P3':(-0.4,-0.5), 'Pz':(0,-0.5), 'P4':(0.4,-0.5),
            'O1':(-0.3,-0.8), 'Oz':(0,-0.8), 'O2':(0.3,-0.8),
        }
        np.random.seed(int(time_point * 1000) % 100)
        values = {ch: np.random.randn() * 5 for ch in ch_positions}

        def draw_topo(painter, w, h):
            ox = 16; oy = h - height - 16
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold)); painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, f"Topo @ {time_point}s")
            # Head circle
            cx_h = ox + width//2; cy_h = oy + height//2 + 10
            radius = min(width, height) // 2 - 20
            painter.setPen(QPen(QColor(100, 110, 130), 1.5))
            painter.setBrush(QColor(240, 242, 248))
            painter.drawEllipse(cx_h - radius, cy_h - radius, radius*2, radius*2)
            # Nose
            painter.drawLine(cx_h, cy_h - radius, cx_h, cy_h - radius - 8)
            # Electrodes
            vmin = min(values.values()); vmax = max(values.values())
            vrange = vmax - vmin if vmax > vmin else 1
            for ch, (px, py) in ch_positions.items():
                ex = cx_h + int(px * radius * 0.85)
                ey = cy_h - int(py * radius * 0.85)
                t = (values[ch] - vmin) / vrange
                r, g, b = _cmap_lookup('rdbu', t)
                painter.setBrush(QColor(int(r*255), int(g*255), int(b*255)))
                painter.setPen(QPen(QColor(60, 60, 80), 0.8))
                painter.drawEllipse(ex-5, ey-5, 10, 10)
                painter.setFont(QFont("Consolas", 5)); painter.setPen(QColor(40, 40, 60))
                painter.drawText(ex-10, ey+7, 20, 8, Qt.AlignCenter, ch)

        self.viewer.add_overlay('topomap', draw_topo)
        self.log(f"Topomap overlay at {time_point}s")
        return self

    # ── Export ─────────────────────────────────────────────────

    def export_nifti(self, path="~/brain_export.nii.gz"):
        """Save current volume as NIfTI file (requires nibabel)."""
        if self._vol is None: self.log("No volume loaded"); return self
        path = os.path.expanduser(path)
        try:
            import nibabel as nib
            img = nib.Nifti1Image(self._vol, self._affine)
            nib.save(img, path)
            self.log(f"Exported: {path}")
        except ImportError:
            self.log("nibabel required for NIfTI export: pip install nibabel")
        return path

    def screenshot(self, path="/tmp/brain.png"):
        """Save current viewer as PNG."""
        self.viewer.screenshot(os.path.expanduser(path))
        self.log(f"Screenshot: {path}")
        return self


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
#  NEUROLAB APP — UI ASSEMBLY + SINGLETON + SIGNAL WIRING
# ═══════════════════════════════════════════════════════════════

class NeuroLabApp:
    """Encapsulates the entire NeuroLab UI assembly, singleton creation,
    and signal wiring. Exposes brain, viewer, and main_widget for external use."""

    def __init__(self):
        self._build_ui()
        self._create_singleton()
        self._wire_signals()
        self.brain.load(self.template_combo.currentText())

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        self.main_widget = QWidget()
        self.main_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        main_layout = QHBoxLayout(self.main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)

        # ── Panel ──
        panel = QWidget(); panel.setFixedWidth(300); panel.setStyleSheet(_SS)
        panel.setAttribute(Qt.WA_TranslucentBackground, True)
        ps = QScrollArea(); ps.setWidgetResizable(True); ps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ps.setStyleSheet("QScrollArea{border:none;background:transparent}QScrollBar:vertical{width:5px;background:transparent}QScrollBar::handle:vertical{background:#c0c8d4;border-radius:2px;min-height:30px}")
        inner = QWidget(); inner.setAttribute(Qt.WA_TranslucentBackground, True)
        lay = QVBoxLayout(inner); lay.setSpacing(4); lay.setContentsMargins(10, 10, 10, 10)

        # Header
        hdr = QWidget(); hdr.setAttribute(Qt.WA_TranslucentBackground, True)
        hl = QHBoxLayout(hdr); hl.setContentsMargins(0, 0, 0, 4)
        ic = QLabel("\U0001f9e0"); ic.setStyleSheet("font-size:20px;background:rgba(230,240,250,200);border:1px solid #d0d8e4;border-radius:7px;padding:3px 7px")
        nw = QWidget(); nw.setAttribute(Qt.WA_TranslucentBackground, True)
        nl = QVBoxLayout(nw); nl.setContentsMargins(6, 0, 0, 0); nl.setSpacing(0)
        _title_lbl = QLabel("NeuroLab"); _title_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#1a2a40;background:transparent")
        _sub_lbl = QLabel("BRAIN IMAGING WORKBENCH"); _sub_lbl.setStyleSheet("font-size:7px;letter-spacing:2px;color:#8a96a8;background:transparent")
        nl.addWidget(_title_lbl); nl.addWidget(_sub_lbl)
        hl.addWidget(ic); hl.addWidget(nw); hl.addStretch()
        lay.addWidget(hdr)

        tabs = QTabWidget(); tabs.setStyleSheet(_SS)

        # ── Tab 1: Volume ──
        t1 = QWidget(); t1.setAttribute(Qt.WA_TranslucentBackground, True)
        t1l = QVBoxLayout(t1); t1l.setSpacing(5); t1l.setContentsMargins(6, 8, 6, 6)
        t1l.addWidget(_lbl("Template"))
        self.template_combo = QComboBox()
        self.template_combo.addItems(["mni152", "fsaverage"])
        t1l.addWidget(self.template_combo)

        t1l.addWidget(_lbl("View Mode"))
        vm_w = QWidget(); vm_w.setAttribute(Qt.WA_TranslucentBackground, True)
        vm_l = QHBoxLayout(vm_w); vm_l.setContentsMargins(0, 0, 0, 0); vm_l.setSpacing(2)
        self.view_btns = {}
        for vn, vlb in [("slice", "Slice"), ("surface", "Surface")]:
            b = QPushButton(vlb); b.setCheckable(True); b.setChecked(vn == "slice")
            self.view_btns[vn] = b; vm_l.addWidget(b)
        t1l.addWidget(vm_w)

        t1l.addWidget(_lbl("Slice Axis"))
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["axial", "sagittal", "coronal"])
        t1l.addWidget(self.axis_combo)

        t1l.addWidget(_lbl("Slice Position"))
        self.slice_slider = QSlider(Qt.Horizontal); self.slice_slider.setRange(0, 90); self.slice_slider.setValue(45)
        t1l.addWidget(self.slice_slider)

        t1l.addWidget(_lbl("Colormap"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLORMAPS.keys()))
        self.cmap_combo.setCurrentText("bone")
        t1l.addWidget(self.cmap_combo)
        t1l.addStretch()
        tabs.addTab(t1, "Volume")

        # ── Tab 2: Overlays ──
        t2 = QWidget(); t2.setAttribute(Qt.WA_TranslucentBackground, True)
        t2l = QVBoxLayout(t2); t2l.setSpacing(5); t2l.setContentsMargins(6, 8, 6, 6)
        t2l.addWidget(_lbl("Activation Regions"))
        self.region_list = QListWidget(); self.region_list.setMinimumHeight(120)
        self.region_list.setSelectionMode(QListWidget.MultiSelection)
        for rname, rdata in BRAIN_REGIONS.items():
            self.region_list.addItem(f"\u25cf {rdata['desc']} ({rname})")
        t2l.addWidget(self.region_list)

        self.act_btn = QPushButton("Show Activation"); t2l.addWidget(self.act_btn)
        self.atlas_btn = QPushButton("Show Atlas"); t2l.addWidget(self.atlas_btn)
        self.clear_ov_btn = QPushButton("Clear Overlay"); t2l.addWidget(self.clear_ov_btn)

        t2l.addWidget(_lbl("Threshold"))
        self.thresh_slider = QSlider(Qt.Horizontal); self.thresh_slider.setRange(0, 100); self.thresh_slider.setValue(0)
        t2l.addWidget(self.thresh_slider)

        t2l.addWidget(_lbl("Overlay Opacity"))
        self.opacity_slider = QSlider(Qt.Horizontal); self.opacity_slider.setRange(0, 100); self.opacity_slider.setValue(65)
        t2l.addWidget(self.opacity_slider)
        t2l.addStretch()
        tabs.addTab(t2, "Overlays")

        # ── Tab 3: Analysis ──
        t3 = QWidget(); t3.setAttribute(Qt.WA_TranslucentBackground, True)
        t3l = QVBoxLayout(t3); t3l.setSpacing(5); t3l.setContentsMargins(6, 8, 6, 6)
        t3l.addWidget(_lbl("Processing"))
        self.smooth_btn = QPushButton("Smooth (6mm)"); t3l.addWidget(self.smooth_btn)
        self.seg_btn = QPushButton("Segment Tissues"); t3l.addWidget(self.seg_btn)
        self.conn_btn = QPushButton("Connectivity"); t3l.addWidget(self.conn_btn)
        self.hist_btn = QPushButton("Histogram"); t3l.addWidget(self.hist_btn)

        t3l.addWidget(_lbl("ROI Query"))
        self.roi_combo = QComboBox()
        self.roi_combo.addItems(list(BRAIN_REGIONS.keys()))
        t3l.addWidget(self.roi_combo)
        self.roi_btn = QPushButton("ROI Stats"); t3l.addWidget(self.roi_btn)

        t3l.addWidget(_lbl("EEG / MEG"))
        self.eeg_btn = QPushButton("Demo EEG Waveforms"); t3l.addWidget(self.eeg_btn)
        self.topo_btn = QPushButton("Demo Topomap"); t3l.addWidget(self.topo_btn)
        t3l.addStretch()
        tabs.addTab(t3, "Analysis")

        # ── Tab 4: Import / Info ──
        t4 = QWidget(); t4.setAttribute(Qt.WA_TranslucentBackground, True)
        t4l = QVBoxLayout(t4); t4l.setSpacing(5); t4l.setContentsMargins(6, 8, 6, 6)
        t4l.addWidget(_lbl("Load NIfTI"))
        self.nifti_path_edit = QLineEdit(); self.nifti_path_edit.setPlaceholderText("~/path/to/scan.nii.gz")
        t4l.addWidget(self.nifti_path_edit)
        self.load_nifti_btn = QPushButton("Load NIfTI"); t4l.addWidget(self.load_nifti_btn)

        t4l.addWidget(_lbl("Properties"))
        self.info_edit = QTextEdit(); self.info_edit.setReadOnly(True); self.info_edit.setMinimumHeight(80)
        t4l.addWidget(self.info_edit)

        t4l.addWidget(_lbl("Log"))
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setPlainText("[NeuroLab] Initialised\n[NeuroLab] Renderer: ModernGL\n")
        t4l.addWidget(self.log_edit)

        self.status_lbl = QLabel(""); self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#6a7a8a;font-size:10px;background:transparent")
        t4l.addWidget(self.status_lbl)
        t4l.addStretch()
        tabs.addTab(t4, "Import")

        lay.addWidget(tabs); ps.setWidget(inner)
        playout = QVBoxLayout(panel); playout.setContentsMargins(0, 0, 0, 0); playout.addWidget(ps)
        self.viewer = BrainViewer()
        self.viewer.setStyleSheet("background:transparent")
        main_layout.addWidget(panel); main_layout.addWidget(self.viewer, 1)

    # ── Singleton creation ────────────────────────────────────

    def _create_singleton(self):
        self.brain = NeuroLab(self.viewer, self.log_edit)
        self.nlab = self.brain  # alias

    # ── UI update helper ──────────────────────────────────────

    def _update_ui(self):
        """Sync UI with brain state after any load/change."""
        info_lines = [f"Name: {self.brain.info.get('name', '')}"]
        shape = self.brain.info.get('shape')
        if shape: info_lines.append(f"Shape: {shape[0]}×{shape[1]}×{shape[2]}")
        vs = self.brain.info.get('voxel_size')
        if vs: info_lines.append(f"Voxel: {vs[0]:.1f}×{vs[1]:.1f}×{vs[2]:.1f} mm")
        rng = self.brain.info.get('range')
        if rng: info_lines.append(f"Range: [{rng[0]:.1f}, {rng[1]:.1f}]")
        n = self.brain.info.get('n_nonzero')
        if n: info_lines.append(f"Non-zero: {n:,} voxels")
        m = self.brain.info.get('mean')
        if m: info_lines.append(f"Mean: {m:.1f} ± {self.brain.info.get('std',0):.1f}")
        if self.brain.info.get('tissue_volumes'):
            tv = self.brain.info['tissue_volumes']
            info_lines.append(f"GM: {tv['GM']:,}  WM: {tv['WM']:,}  CSF: {tv['CSF']:,}")
        self.info_edit.setPlainText('\n'.join(info_lines))
        # Update slider range
        if self.brain._vol is not None:
            axis = self.axis_combo.currentText()
            maxdim = {'axial': self.brain._vol.shape[2]-1,
                      'sagittal': self.brain._vol.shape[0]-1,
                      'coronal': self.brain._vol.shape[1]-1}[axis]
            self.slice_slider.setRange(0, maxdim)

    # ── Signal wiring ─────────────────────────────────────────

    def _wire_signals(self):
        brain = self.brain

        # Wrap brain methods to auto-update UI
        _orig_load = brain.load
        def _load_wrap(name): _orig_load(name); self._update_ui(); return brain
        brain.load = _load_wrap

        _orig_load_nifti = brain.load_nifti
        def _load_nifti_wrap(path): _orig_load_nifti(path); self._update_ui(); return brain
        brain.load_nifti = _load_nifti_wrap

        _orig_smooth = brain.smooth
        def _smooth_wrap(fwhm=6.0): _orig_smooth(fwhm); self._update_ui(); return brain
        brain.smooth = _smooth_wrap

        _orig_segment = brain.segment
        def _segment_wrap(): _orig_segment(); self._update_ui(); return brain
        brain.segment = _segment_wrap

        _orig_resample = brain.resample
        def _resample_wrap(mm=2.0): _orig_resample(mm); self._update_ui(); return brain
        brain.resample = _resample_wrap

        # ── UI signal wiring ──
        self.template_combo.currentTextChanged.connect(lambda t: brain.load(t))

        def _on_view_mode(vn):
            for k, b in self.view_btns.items(): b.setChecked(k == vn)
            if vn == 'surface': brain.view_surface()
            else: brain.view_slice()
        for vn, bt in self.view_btns.items():
            bt.clicked.connect(lambda c, v=vn: _on_view_mode(v))

        self.axis_combo.currentTextChanged.connect(lambda a: (
            brain.slice(a, self.slice_slider.value()),
            self._update_ui()
        ))

        self.slice_slider.valueChanged.connect(lambda v: brain.slice(self.axis_combo.currentText(), v))
        self.cmap_combo.currentTextChanged.connect(lambda c: brain.colormap(c))

        def _on_activation():
            selected = self.region_list.selectedItems()
            if not selected:
                brain.overlay_activation()
            else:
                regions = []
                for item in selected:
                    # Extract region name from "● Description (name)"
                    text = item.text()
                    m = re.search(r'\((\w+)\)', text)
                    if m: regions.append(m.group(1))
                brain.overlay_activation(regions)

        self.act_btn.clicked.connect(_on_activation)
        self.atlas_btn.clicked.connect(lambda: brain.overlay_atlas())
        self.clear_ov_btn.clicked.connect(lambda: brain.clear_overlay())
        self.thresh_slider.valueChanged.connect(lambda v: brain.threshold(v / 10.0))
        self.opacity_slider.valueChanged.connect(lambda v: brain.opacity(v / 100.0))

        self.smooth_btn.clicked.connect(lambda: brain.smooth(6.0))
        self.seg_btn.clicked.connect(lambda: brain.segment())
        self.conn_btn.clicked.connect(lambda: brain.connectivity())
        self.hist_btn.clicked.connect(lambda: brain.histogram())
        self.roi_btn.clicked.connect(lambda: (
            brain.roi_stats(self.roi_combo.currentText()),
            self._update_ui()
        ))
        self.eeg_btn.clicked.connect(lambda: brain.plot_eeg())
        self.topo_btn.clicked.connect(lambda: brain.topomap())

        self.load_nifti_btn.clicked.connect(lambda: (
            brain.load_nifti(self.nifti_path_edit.text().strip()) if self.nifti_path_edit.text().strip() else
            self.status_lbl.setText("\u26a0 Enter a NIfTI path first")
        ))


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE APP & EXPOSE NAMESPACE
# ═══════════════════════════════════════════════════════════════

neuro_app = NeuroLabApp()
neuro_brain = neuro_app.brain
neuro_nlab = neuro_app.nlab
neuro_viewer = neuro_app.viewer

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

neuro_proxy = graphics_scene.addWidget(neuro_app.main_widget)
neuro_proxy.setPos(0, 0)
neuro_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
neuro_shadow = QGraphicsDropShadowEffect()
neuro_shadow.setBlurRadius(60); neuro_shadow.setOffset(45, 45); neuro_shadow.setColor(QColor(0, 0, 0, 120))
neuro_proxy.setGraphicsEffect(neuro_shadow)
neuro_app.main_widget.resize(1400, 850)

# Center in current view
_vr = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
neuro_proxy.setPos(_vr.center().x() - neuro_app.main_widget.width() / 2,
             _vr.center().y() - neuro_app.main_widget.height() / 2)
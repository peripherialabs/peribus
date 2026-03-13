"""
CircuitLab — Electrical Engineering Workbench for Rio
═══════════════════════════════════════════════════════════════════

ARCHITECTURE: This file is the complete runtime. It registers a `circ`
singleton into the namespace. The LLM sees this source as context and
injects short code snippets to drive it.

EXAMPLES (what the LLM would output):
    circ.load("rc_lowpass")                   # load preset circuit
    circ.add("R1", "resistor", 1000)          # 1kΩ resistor
    circ.add("C1", "capacitor", 100e-9)       # 100nF capacitor
    circ.add("L1", "inductor", 10e-3)         # 10mH inductor
    circ.add("V1", "vsrc", 5.0)              # 5V DC source
    circ.add("V1", "vsrc", 5.0, ac_amp=1.0)  # 5V DC + 1V AC source
    circ.add("I1", "isrc", 0.001)            # 1mA current source
    circ.add("D1", "diode")                   # ideal diode
    circ.add("Q1", "npn")                     # NPN transistor (B,C,E pins)
    circ.add("U1", "opamp")                   # ideal op-amp (+,-,out pins)
    circ.connect("R1.1", "V1.+")             # wire pin to pin
    circ.connect("R1.2", "C1.1")
    circ.ground("C1.2")                       # connect to ground (node 0)
    circ.ground("V1.-")

    ## SIMULATION:
    circ.dc()                                 # DC operating point
    circ.ac(start=1, stop=1e6, points=200)   # AC sweep (Hz)
    circ.tran(tstep=1e-6, tstop=1e-3)        # transient analysis
    circ.dc_sweep("V1", 0, 5, 0.1)          # DC sweep source V1

    ## DISPLAY:
    circ.overlay_waveform("node_out")         # show voltage waveform
    circ.overlay_bode("node_out")             # show Bode plot (magnitude+phase)
    circ.overlay_dc_table()                   # show DC operating point table
    circ.label_voltages()                     # label all node voltages
    circ.label_currents()                     # label branch currents
    circ.remove_overlay("waveform")

    ## After dc()/ac()/tran(), circ.results contains REAL data:
    circ.results['dc']['node_voltages']       # {node_name: voltage}
    circ.results['dc']['branch_currents']     # {src_name: current}
    circ.results['ac']['freqs']               # frequency array
    circ.results['ac']['node_out']            # complex voltage array at node_out
    circ.results['tran']['time']              # time array
    circ.results['tran']['node_out']          # voltage array at node_out

    ## NETLIST I/O:
    circ.export_spice("/tmp/circuit.cir")     # save SPICE netlist
    circ.load_netlist(text)                    # parse SPICE netlist
    circ.clear()                               # clear all components

    ## COMPONENT QUERY:
    circ.components                            # dict of all components
    circ.nodes                                 # set of all node names
    circ.describe()                            # human-readable summary

VIEWER API (lower level, when LLM needs custom rendering):
    circ.viewer.cam_x, circ.viewer.cam_y      # camera pan
    circ.viewer.zoom                           # camera zoom
    circ.viewer.add_overlay(name, fn)          # fn(painter, w, h)
    circ.viewer.remove_overlay(name)
    circ.viewer.screenshot(path)               # save PNG

NAMESPACE: After this file runs, these are available:
    circ        — CircuitLab singleton (main API)
    circuit_viewer      — alias for circ.viewer
    circuit     — alias for circ (long form)
    All PySide6/Qt, numpy, moderngl, glm from parser namespace
"""

import math
import os
import re
import numpy as np
from collections import OrderedDict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QCheckBox, QTabWidget, QTextEdit,
    QScrollArea, QListWidget, QLineEdit, QGraphicsItem,
    QGraphicsDropShadowEffect, QSizePolicy, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage, QLinearGradient,
    QPainterPath, QPolygonF
)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & STYLING
# ═══════════════════════════════════════════════════════════════

# SI prefix formatting
def _si(val, unit=""):
    """Format value with SI prefix."""
    if val == 0: return f"0 {unit}"
    prefixes = [(1e12,'T'),(1e9,'G'),(1e6,'M'),(1e3,'k'),(1,''),
                (1e-3,'m'),(1e-6,'μ'),(1e-9,'n'),(1e-12,'p')]
    for thresh, prefix in prefixes:
        if abs(val) >= thresh:
            v = val / thresh
            if v == int(v): return f"{int(v)} {prefix}{unit}"
            return f"{v:.3g} {prefix}{unit}"
    return f"{val:.3g} {unit}"

# Component colors
COMP_COLORS = {
    'resistor':  QColor(180, 120, 60),
    'capacitor': QColor(50, 130, 180),
    'inductor':  QColor(130, 60, 160),
    'vsrc':      QColor(200, 60, 60),
    'isrc':      QColor(60, 160, 80),
    'diode':     QColor(160, 80, 40),
    'npn':       QColor(100, 100, 160),
    'pnp':       QColor(100, 100, 160),
    'opamp':     QColor(60, 60, 80),
    'wire':      QColor(80, 90, 110),
    'ground':    QColor(100, 110, 120),
}

# Pin definitions per component type
COMP_PINS = {
    'resistor':  ['1', '2'],
    'capacitor': ['1', '2'],
    'inductor':  ['1', '2'],
    'vsrc':      ['+', '-'],
    'isrc':      ['+', '-'],
    'diode':     ['A', 'K'],      # anode, cathode
    'npn':       ['B', 'C', 'E'],  # base, collector, emitter
    'pnp':       ['B', 'C', 'E'],
    'opamp':     ['+', '-', 'out'],
}

# ═══════════════════════════════════════════════════════════════
#  COMPONENT DATA MODEL
# ═══════════════════════════════════════════════════════════════

class Component:
    """A single circuit component."""
    def __init__(self, name, ctype, value=0, **kwargs):
        self.name = name
        self.ctype = ctype
        self.value = value
        self.pins = {p: None for p in COMP_PINS.get(ctype, ['1','2'])}
        self.params = kwargs  # ac_amp, ac_freq, model params, etc.
        # Layout position (auto-placed)
        self.x = 0.0
        self.y = 0.0
        self.rotation = 0  # 0, 90, 180, 270

    def pin_names(self):
        return list(self.pins.keys())

    def describe(self):
        unit = {'resistor':'Ω','capacitor':'F','inductor':'H',
                'vsrc':'V','isrc':'A'}.get(self.ctype, '')
        val_str = _si(self.value, unit) if self.value else self.ctype
        conns = ', '.join(f"{p}→{n}" for p,n in self.pins.items() if n is not None)
        return f"{self.name}: {val_str} [{conns}]"


# ═══════════════════════════════════════════════════════════════
#  SPICE-LIKE NETLIST PARSER
# ═══════════════════════════════════════════════════════════════

def _parse_spice_value(s):
    """Parse SPICE value notation: 1k, 100n, 4.7u, 10MEG, etc."""
    s = s.strip().upper()
    multipliers = {'F':1e-15,'P':1e-12,'N':1e-9,'U':1e-6,'M':1e-3,
                   'K':1e3,'MEG':1e6,'G':1e9,'T':1e12}
    # Try direct float first
    try: return float(s)
    except: pass
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            try: return float(s[:-len(suffix)]) * mult
            except: pass
    return 0.0

def parse_spice_netlist(text):
    """Parse a SPICE netlist → list of (name, type, nodes, value, params)."""
    components = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
        parts = line.split()
        if len(parts) < 3: continue
        name = parts[0]
        prefix = name[0].upper()
        if prefix == 'R':
            # R1 node1 node2 value
            if len(parts) >= 4:
                components.append((name, 'resistor', [parts[1], parts[2]], _parse_spice_value(parts[3]), {}))
        elif prefix == 'C':
            if len(parts) >= 4:
                components.append((name, 'capacitor', [parts[1], parts[2]], _parse_spice_value(parts[3]), {}))
        elif prefix == 'L':
            if len(parts) >= 4:
                components.append((name, 'inductor', [parts[1], parts[2]], _parse_spice_value(parts[3]), {}))
        elif prefix == 'V':
            val = _parse_spice_value(parts[3]) if len(parts) >= 4 else 0
            params = {}
            rest = ' '.join(parts[4:]) if len(parts) > 4 else ''
            ac_match = re.search(r'AC\s+([\d.eE+-]+)', rest, re.I)
            if ac_match: params['ac_amp'] = float(ac_match.group(1))
            components.append((name, 'vsrc', [parts[1], parts[2]], val, params))
        elif prefix == 'I':
            val = _parse_spice_value(parts[3]) if len(parts) >= 4 else 0
            components.append((name, 'isrc', [parts[1], parts[2]], val, {}))
        elif prefix == 'D':
            if len(parts) >= 3:
                components.append((name, 'diode', [parts[1], parts[2]], 0, {}))
        elif prefix == 'Q':
            if len(parts) >= 4:
                components.append((name, 'npn', [parts[1], parts[2], parts[3]], 0, {}))
    return components


# ═══════════════════════════════════════════════════════════════
#  SIMULATION ENGINE — Modified Nodal Analysis (MNA)
# ═══════════════════════════════════════════════════════════════

class MNASolver:
    """
    Modified Nodal Analysis engine for DC, AC, and Transient simulation.

    Builds the MNA matrix [G B; C D] x [v; i] = [s1; s2] and solves.
    Supports: R, C, L, V, I sources, ideal diodes (iterative), and
    ideal op-amps.
    """

    def __init__(self, components, ground_node='0'):
        self.components = components
        self.ground_node = ground_node

    def _collect_nodes(self):
        """Get all unique node names (excluding ground)."""
        nodes = set()
        for comp in self.components.values():
            for pin, node in comp.pins.items():
                if node is not None and node != self.ground_node:
                    nodes.add(node)
        return sorted(nodes)

    def _voltage_sources(self):
        """Get components that need MNA current variables (V sources, inductors)."""
        vsrcs = []
        for name, comp in self.components.items():
            if comp.ctype in ('vsrc', 'inductor', 'opamp'):
                vsrcs.append(name)
        return vsrcs

    def dc(self):
        """DC operating point analysis. Returns {node_name: voltage}, {src: current}."""
        nodes = self._collect_nodes()
        vsrcs = self._voltage_sources()
        n = len(nodes)
        m = len(vsrcs)
        size = n + m

        if size == 0:
            return {'node_voltages': {}, 'branch_currents': {}}

        node_idx = {name: i for i, name in enumerate(nodes)}
        vsrc_idx = {name: n + i for i, name in enumerate(vsrcs)}

        A = np.zeros((size, size))
        b = np.zeros(size)

        for name, comp in self.components.items():
            p_nodes = comp.pins  # {pin_name: node_name}
            p_list = list(p_nodes.values())

            if comp.ctype == 'resistor':
                if comp.value == 0: continue
                g = 1.0 / comp.value
                n1, n2 = p_list[0], p_list[1]
                i1 = node_idx.get(n1)
                i2 = node_idx.get(n2)
                if i1 is not None: A[i1, i1] += g
                if i2 is not None: A[i2, i2] += g
                if i1 is not None and i2 is not None:
                    A[i1, i2] -= g
                    A[i2, i1] -= g

            elif comp.ctype == 'capacitor':
                # DC: capacitor = open circuit (do nothing)
                pass

            elif comp.ctype == 'inductor':
                # DC: inductor = short circuit (wire) — modeled as 0V source
                n1, n2 = p_list[0], p_list[1]
                i1 = node_idx.get(n1)
                i2 = node_idx.get(n2)
                vi = vsrc_idx[name]
                if i1 is not None: A[i1, vi] += 1; A[vi, i1] += 1
                if i2 is not None: A[i2, vi] -= 1; A[vi, i2] -= 1
                b[vi] = 0  # V = 0 for DC (short circuit)

            elif comp.ctype == 'vsrc':
                n_pos, n_neg = p_list[0], p_list[1]
                ip = node_idx.get(n_pos)
                im = node_idx.get(n_neg)
                vi = vsrc_idx[name]
                if ip is not None: A[ip, vi] += 1; A[vi, ip] += 1
                if im is not None: A[im, vi] -= 1; A[vi, im] -= 1
                b[vi] = comp.value

            elif comp.ctype == 'isrc':
                n_pos, n_neg = p_list[0], p_list[1]
                ip = node_idx.get(n_pos)
                im = node_idx.get(n_neg)
                if ip is not None: b[ip] += comp.value
                if im is not None: b[im] -= comp.value

            elif comp.ctype == 'opamp':
                # Ideal op-amp: V(out) = A*(V+ - V-), A→∞
                # MNA: add voltage source from out to gnd, controlled by V+ - V-
                # Approximation: very large gain
                n_plus = p_nodes.get('+')
                n_minus = p_nodes.get('-')
                n_out = p_nodes.get('out')
                vi = vsrc_idx[name]
                io = node_idx.get(n_out)
                ip = node_idx.get(n_plus)
                im = node_idx.get(n_minus)
                if io is not None: A[io, vi] += 1
                # V(out) - A*(V+ - V-) = 0 → stamp into vsrc row
                # With ideal: V+ = V- constraint
                if ip is not None: A[vi, ip] += 1
                if im is not None: A[vi, im] -= 1
                # No voltage on output vsrc row (V+ - V- = 0 is the constraint)

            elif comp.ctype == 'diode':
                # Handled in iterative solve below
                pass

            elif comp.ctype in ('npn', 'pnp'):
                # Handled in iterative solve below (Ebers-Moll)
                pass

        # Check for nonlinear devices
        has_nonlinear = any(c.ctype in ('diode', 'npn', 'pnp')
                           for c in self.components.values())

        if has_nonlinear:
            x = self._dc_nonlinear(A, b, nodes, node_idx, vsrc_idx, size)
        else:
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Singular — try pseudoinverse
                x = np.linalg.lstsq(A, b, rcond=None)[0]

        node_voltages = {name: float(x[idx]) for name, idx in node_idx.items()}
        node_voltages[self.ground_node] = 0.0
        branch_currents = {name: float(x[idx]) for name, idx in vsrc_idx.items()}

        return {'node_voltages': node_voltages, 'branch_currents': branch_currents}

    def _dc_nonlinear(self, A_base, b_base, nodes, node_idx, vsrc_idx, size):
        """Iterative DC solve with diode + BJT Newton-Raphson linearization."""
        x = np.zeros(size)
        # Initial guess: set all nodes to small positive voltage to help convergence
        for i in range(len(nodes)):
            x[i] = 0.1
        Is = 1e-14   # diode saturation current
        Vt = 0.026   # thermal voltage
        beta_f = 100  # forward current gain for BJTs

        for iteration in range(80):
            A = A_base.copy()
            b = b_base.copy()

            for name, comp in self.components.items():
                if comp.ctype == 'diode':
                    p_list = list(comp.pins.values())
                    na, nk = p_list[0], p_list[1]
                    ia = node_idx.get(na)
                    ik = node_idx.get(nk)
                    va = x[ia] if ia is not None else 0
                    vk = x[ik] if ik is not None else 0
                    vd = va - vk

                    exp_vd = min(np.exp(vd / Vt), 1e15)
                    gd = max(Is / Vt * exp_vd, 1e-12)
                    id_val = Is * (exp_vd - 1)
                    ieq = id_val - gd * vd

                    if ia is not None: A[ia, ia] += gd
                    if ik is not None: A[ik, ik] += gd
                    if ia is not None and ik is not None:
                        A[ia, ik] -= gd; A[ik, ia] -= gd
                    if ia is not None: b[ia] -= ieq
                    if ik is not None: b[ik] += ieq

                elif comp.ctype in ('npn', 'pnp'):
                    # Ebers-Moll linearized model
                    p_nodes = comp.pins
                    nb = node_idx.get(p_nodes.get('B'))
                    nc = node_idx.get(p_nodes.get('C'))
                    ne = node_idx.get(p_nodes.get('E'))

                    vb = x[nb] if nb is not None else 0
                    vc = x[nc] if nc is not None else 0
                    ve = x[ne] if ne is not None else 0

                    if comp.ctype == 'npn':
                        vbe = vb - ve
                        vbc = vb - vc
                    else:  # pnp: reversed
                        vbe = ve - vb
                        vbc = vc - vb

                    # BE junction
                    exp_be = min(np.exp(vbe / Vt), 1e15)
                    gbe = max(Is / Vt * exp_be, 1e-12)
                    ibe = Is * (exp_be - 1)
                    ieq_be = ibe - gbe * vbe

                    # Transconductance: Ic = beta * Ibe
                    gm = beta_f * gbe
                    ic_eq = beta_f * ieq_be

                    if comp.ctype == 'npn':
                        # BE junction conductance (B to E)
                        if nb is not None: A[nb, nb] += gbe
                        if ne is not None: A[ne, ne] += gbe
                        if nb is not None and ne is not None:
                            A[nb, ne] -= gbe; A[ne, nb] -= gbe
                        if nb is not None: b[nb] -= ieq_be
                        if ne is not None: b[ne] += ieq_be

                        # Collector current: Ic = gm*(Vb - Ve) + ic_eq
                        if nc is not None and nb is not None: A[nc, nb] += gm
                        if nc is not None and ne is not None: A[nc, ne] -= gm
                        if ne is not None and nb is not None: A[ne, nb] -= gm
                        if ne is not None and ne is not None: A[ne, ne] += gm
                        if nc is not None: b[nc] += ic_eq
                        if ne is not None: b[ne] -= ic_eq
                    else:
                        # PNP: reversed current directions
                        if ne is not None: A[ne, ne] += gbe
                        if nb is not None: A[nb, nb] += gbe
                        if ne is not None and nb is not None:
                            A[ne, nb] -= gbe; A[nb, ne] -= gbe
                        if ne is not None: b[ne] -= ieq_be
                        if nb is not None: b[nb] += ieq_be

                        if nc is not None and ne is not None: A[nc, ne] += gm
                        if nc is not None and nb is not None: A[nc, nb] -= gm
                        if nb is not None and ne is not None: A[nb, ne] -= gm
                        if nb is not None: A[nb, nb] += gm
                        if nc is not None: b[nc] += ic_eq
                        if nb is not None: b[nb] -= ic_eq

            try:
                x_new = np.linalg.solve(A, b)
            except:
                x_new = np.linalg.lstsq(A, b, rcond=None)[0]

            if np.max(np.abs(x_new - x)) < 1e-9:
                return x_new
            # Damping for convergence stability
            x = 0.7 * x_new + 0.3 * x

        return x

    def ac(self, freqs):
        """AC analysis. Returns {node_name: complex_voltage_array} for each freq."""
        nodes = self._collect_nodes()
        vsrcs = self._voltage_sources()
        n = len(nodes)
        m = len(vsrcs)
        size = n + m
        if size == 0: return {}

        node_idx = {name: i for i, name in enumerate(nodes)}
        vsrc_idx = {name: n + i for i, name in enumerate(vsrcs)}

        results = {name: np.zeros(len(freqs), dtype=complex) for name in nodes}
        results[self.ground_node] = np.zeros(len(freqs), dtype=complex)

        for fi, freq in enumerate(freqs):
            w = 2 * np.pi * freq
            s = 1j * w

            A = np.zeros((size, size), dtype=complex)
            b = np.zeros(size, dtype=complex)

            for name, comp in self.components.items():
                p_list = list(comp.pins.values())

                if comp.ctype == 'resistor':
                    if comp.value == 0: continue
                    g = 1.0 / comp.value
                    n1, n2 = p_list[0], p_list[1]
                    i1 = node_idx.get(n1)
                    i2 = node_idx.get(n2)
                    if i1 is not None: A[i1, i1] += g
                    if i2 is not None: A[i2, i2] += g
                    if i1 is not None and i2 is not None:
                        A[i1, i2] -= g; A[i2, i1] -= g

                elif comp.ctype == 'capacitor':
                    y = s * comp.value  # admittance = sC
                    n1, n2 = p_list[0], p_list[1]
                    i1 = node_idx.get(n1)
                    i2 = node_idx.get(n2)
                    if i1 is not None: A[i1, i1] += y
                    if i2 is not None: A[i2, i2] += y
                    if i1 is not None and i2 is not None:
                        A[i1, i2] -= y; A[i2, i1] -= y

                elif comp.ctype == 'inductor':
                    # Inductor: V = sL * I → stamp as voltage source with V = sL*I
                    n1, n2 = p_list[0], p_list[1]
                    i1 = node_idx.get(n1)
                    i2 = node_idx.get(n2)
                    vi = vsrc_idx[name]
                    if i1 is not None: A[i1, vi] += 1; A[vi, i1] += 1
                    if i2 is not None: A[i2, vi] -= 1; A[vi, i2] -= 1
                    A[vi, vi] -= s * comp.value  # V = sL*I → V - sL*I = 0

                elif comp.ctype == 'vsrc':
                    n_pos, n_neg = p_list[0], p_list[1]
                    ip = node_idx.get(n_pos)
                    im = node_idx.get(n_neg)
                    vi = vsrc_idx[name]
                    if ip is not None: A[ip, vi] += 1; A[vi, ip] += 1
                    if im is not None: A[im, vi] -= 1; A[vi, im] -= 1
                    # AC source amplitude
                    ac_amp = comp.params.get('ac_amp', 0)
                    b[vi] = ac_amp if ac_amp else 0

                elif comp.ctype == 'isrc':
                    n_pos, n_neg = p_list[0], p_list[1]
                    ip = node_idx.get(n_pos)
                    im = node_idx.get(n_neg)
                    ac_amp = comp.params.get('ac_amp', 0)
                    if ac_amp:
                        if ip is not None: b[ip] += ac_amp
                        if im is not None: b[im] -= ac_amp

                elif comp.ctype == 'opamp':
                    n_plus = comp.pins.get('+')
                    n_minus = comp.pins.get('-')
                    n_out = comp.pins.get('out')
                    vi = vsrc_idx[name]
                    io = node_idx.get(n_out)
                    ip_n = node_idx.get(n_plus)
                    im_n = node_idx.get(n_minus)
                    if io is not None: A[io, vi] += 1
                    if ip_n is not None: A[vi, ip_n] += 1
                    if im_n is not None: A[vi, im_n] -= 1

            try:
                x = np.linalg.solve(A, b)
            except:
                x = np.linalg.lstsq(A, b, rcond=None)[0]

            for name, idx in node_idx.items():
                results[name][fi] = x[idx]

        return results

    def tran(self, tstep, tstop, tstart=0):
        """Transient analysis using backward Euler integration.
        Returns {node_name: voltage_array}, time_array."""
        nodes = self._collect_nodes()
        vsrcs = self._voltage_sources()
        n = len(nodes)
        m = len(vsrcs)
        size = n + m
        if size == 0: return {}, np.array([])

        node_idx = {name: i for i, name in enumerate(nodes)}
        vsrc_idx = {name: n + i for i, name in enumerate(vsrcs)}

        time_pts = np.arange(tstart, tstop + tstep/2, tstep)
        results = {name: np.zeros(len(time_pts)) for name in nodes}
        results[self.ground_node] = np.zeros(len(time_pts))

        x_prev = np.zeros(size)
        h = tstep

        for ti, t in enumerate(time_pts):
            A = np.zeros((size, size))
            b = np.zeros(size)

            for name, comp in self.components.items():
                p_list = list(comp.pins.values())

                if comp.ctype == 'resistor':
                    if comp.value == 0: continue
                    g = 1.0 / comp.value
                    n1, n2 = p_list[0], p_list[1]
                    i1 = node_idx.get(n1)
                    i2 = node_idx.get(n2)
                    if i1 is not None: A[i1, i1] += g
                    if i2 is not None: A[i2, i2] += g
                    if i1 is not None and i2 is not None:
                        A[i1, i2] -= g; A[i2, i1] -= g

                elif comp.ctype == 'capacitor':
                    # Backward Euler: I = C*(V_n - V_n-1)/h → G_eq = C/h, I_eq = C/h * V_n-1
                    geq = comp.value / h
                    n1, n2 = p_list[0], p_list[1]
                    i1 = node_idx.get(n1)
                    i2 = node_idx.get(n2)
                    v1_prev = x_prev[i1] if i1 is not None else 0
                    v2_prev = x_prev[i2] if i2 is not None else 0
                    ieq = geq * (v1_prev - v2_prev)

                    if i1 is not None: A[i1, i1] += geq
                    if i2 is not None: A[i2, i2] += geq
                    if i1 is not None and i2 is not None:
                        A[i1, i2] -= geq; A[i2, i1] -= geq
                    if i1 is not None: b[i1] += ieq
                    if i2 is not None: b[i2] -= ieq

                elif comp.ctype == 'inductor':
                    # Backward Euler: V = L*(I_n - I_n-1)/h → V - L/h*I_n = -L/h*I_n-1
                    n1, n2 = p_list[0], p_list[1]
                    i1 = node_idx.get(n1)
                    i2 = node_idx.get(n2)
                    vi = vsrc_idx[name]
                    if i1 is not None: A[i1, vi] += 1; A[vi, i1] += 1
                    if i2 is not None: A[i2, vi] -= 1; A[vi, i2] -= 1
                    A[vi, vi] -= comp.value / h
                    b[vi] = -(comp.value / h) * x_prev[vi]

                elif comp.ctype == 'vsrc':
                    n_pos, n_neg = p_list[0], p_list[1]
                    ip = node_idx.get(n_pos)
                    im = node_idx.get(n_neg)
                    vi = vsrc_idx[name]
                    if ip is not None: A[ip, vi] += 1; A[vi, ip] += 1
                    if im is not None: A[im, vi] -= 1; A[vi, im] -= 1
                    # Time-varying source
                    v = comp.value
                    ac_amp = comp.params.get('ac_amp', 0)
                    ac_freq = comp.params.get('ac_freq', 1000)
                    if ac_amp:
                        v += ac_amp * math.sin(2 * math.pi * ac_freq * t)
                    # Pulse source support
                    if 'pulse_v1' in comp.params:
                        prm = comp.params
                        period = prm.get('pulse_period', tstop)
                        t_mod = t % period
                        if t_mod < prm.get('pulse_rise', 0):
                            v = prm['pulse_v1'] + (prm['pulse_v2']-prm['pulse_v1'])*t_mod/max(prm['pulse_rise'],1e-15)
                        elif t_mod < prm.get('pulse_rise',0)+prm.get('pulse_width',period/2):
                            v = prm['pulse_v2']
                        elif t_mod < prm.get('pulse_rise',0)+prm.get('pulse_width',period/2)+prm.get('pulse_fall',0):
                            tf = t_mod - prm['pulse_rise'] - prm['pulse_width']
                            v = prm['pulse_v2'] + (prm['pulse_v1']-prm['pulse_v2'])*tf/max(prm['pulse_fall'],1e-15)
                        else:
                            v = prm['pulse_v1']
                    b[vi] = v

                elif comp.ctype == 'isrc':
                    n_pos, n_neg = p_list[0], p_list[1]
                    ip = node_idx.get(n_pos)
                    im = node_idx.get(n_neg)
                    val = comp.value
                    ac_amp = comp.params.get('ac_amp', 0)
                    ac_freq = comp.params.get('ac_freq', 1000)
                    if ac_amp:
                        val += ac_amp * math.sin(2 * math.pi * ac_freq * t)
                    if ip is not None: b[ip] += val
                    if im is not None: b[im] -= val

                elif comp.ctype == 'opamp':
                    n_plus = comp.pins.get('+')
                    n_minus = comp.pins.get('-')
                    n_out = comp.pins.get('out')
                    vi = vsrc_idx[name]
                    io = node_idx.get(n_out)
                    ip_n = node_idx.get(n_plus)
                    im_n = node_idx.get(n_minus)
                    if io is not None: A[io, vi] += 1
                    if ip_n is not None: A[vi, ip_n] += 1
                    if im_n is not None: A[vi, im_n] -= 1

                elif comp.ctype == 'diode':
                    na, nk = p_list[0], p_list[1]
                    ia = node_idx.get(na)
                    ik = node_idx.get(nk)
                    va = x_prev[ia] if ia is not None else 0
                    vk = x_prev[ik] if ik is not None else 0
                    vd = va - vk
                    Is_d = 1e-14; Vt_d = 0.026
                    exp_vd = min(np.exp(vd / Vt_d), 1e15)
                    gd = max(Is_d / Vt_d * exp_vd, 1e-12)
                    id_val = Is_d * (exp_vd - 1)
                    ieq = id_val - gd * vd
                    if ia is not None: A[ia, ia] += gd
                    if ik is not None: A[ik, ik] += gd
                    if ia is not None and ik is not None:
                        A[ia, ik] -= gd; A[ik, ia] -= gd
                    if ia is not None: b[ia] -= ieq
                    if ik is not None: b[ik] += ieq

                elif comp.ctype in ('npn', 'pnp'):
                    Is_t = 1e-14; Vt_t = 0.026; beta_f = 100
                    p_nodes = comp.pins
                    nb = node_idx.get(p_nodes.get('B'))
                    nc = node_idx.get(p_nodes.get('C'))
                    ne = node_idx.get(p_nodes.get('E'))
                    vb = x_prev[nb] if nb is not None else 0
                    vc = x_prev[nc] if nc is not None else 0
                    ve = x_prev[ne] if ne is not None else 0
                    vbe = (vb - ve) if comp.ctype == 'npn' else (ve - vb)
                    exp_be = min(np.exp(vbe / Vt_t), 1e15)
                    gbe = max(Is_t / Vt_t * exp_be, 1e-12)
                    ibe = Is_t * (exp_be - 1)
                    ieq_be = ibe - gbe * vbe
                    gm = beta_f * gbe
                    ic_eq = beta_f * ieq_be
                    if comp.ctype == 'npn':
                        if nb is not None: A[nb, nb] += gbe
                        if ne is not None: A[ne, ne] += gbe
                        if nb is not None and ne is not None:
                            A[nb, ne] -= gbe; A[ne, nb] -= gbe
                        if nb is not None: b[nb] -= ieq_be
                        if ne is not None: b[ne] += ieq_be
                        if nc is not None and nb is not None: A[nc, nb] += gm
                        if nc is not None and ne is not None: A[nc, ne] -= gm
                        if ne is not None and nb is not None: A[ne, nb] -= gm
                        if ne is not None: A[ne, ne] += gm
                        if nc is not None: b[nc] += ic_eq
                        if ne is not None: b[ne] -= ic_eq
                    else:
                        if ne is not None: A[ne, ne] += gbe
                        if nb is not None: A[nb, nb] += gbe
                        if ne is not None and nb is not None:
                            A[ne, nb] -= gbe; A[nb, ne] -= gbe
                        if ne is not None: b[ne] -= ieq_be
                        if nb is not None: b[nb] += ieq_be
                        if nc is not None and ne is not None: A[nc, ne] += gm
                        if nc is not None and nb is not None: A[nc, nb] -= gm
                        if nb is not None and ne is not None: A[nb, ne] -= gm
                        if nb is not None: A[nb, nb] += gm
                        if nc is not None: b[nc] += ic_eq
                        if nb is not None: b[nb] -= ic_eq

            try:
                x = np.linalg.solve(A, b)
            except:
                x = np.linalg.lstsq(A, b, rcond=None)[0]

            for name, idx in node_idx.items():
                results[name][ti] = float(x[idx].real)
            x_prev = x.copy()

        return results, time_pts


# ═══════════════════════════════════════════════════════════════
#  SCHEMATIC VIEWER — QPainter-based 2D circuit rendering
# ═══════════════════════════════════════════════════════════════

class SchematicViewer(QWidget):
    """2D schematic viewer with topology-aware layout, drag-to-move, click-to-connect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(500, 350)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        self.components = OrderedDict()
        self.node_positions = {}

        # Camera
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom = 1.0

        # Interaction state
        self._mode = 'idle'  # idle, pan, drag_comp, wire_start
        self._lmx = 0; self._lmy = 0
        self._drag_comp = None          # component being dragged
        self._drag_offset = (0, 0)
        self._wire_start_pin = None     # (comp_name, pin_name) for wire drawing
        self._hover_pin = None          # (comp_name, pin_name) under cursor
        self._hover_comp = None         # comp name under cursor
        self._mouse_world = (0, 0)      # current mouse in world coords

        # Overlays
        self._overlays = OrderedDict()

        # Visual
        self.show_grid = True
        self.grid_size = 60
        self._placed = set()  # track which comps have been manually/auto placed

        # Repaint timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.setInterval(33)
        self.timer.start()

        # Callback to parent for wiring (set by CircuitLab)
        self.on_connect = None  # fn(comp_a, pin_a, comp_b, pin_b)

    def set_circuit(self, components, ground_node='0'):
        """Update displayed components. Only auto-layouts NEW components."""
        self.components = components
        self._auto_layout_new()
        self.update()

    def _auto_layout_new(self):
        """Topology-aware layout: chain components along signal path, loop ground back."""
        comps = list(self.components.values())
        new_comps = [c for c in comps if c.name not in self._placed]
        if not new_comps and self._placed:
            return  # everything already placed

        if not comps:
            self._placed.clear()
            return

        # Build node → components adjacency
        node_to_comps = {}
        for comp in comps:
            for pin, node in comp.pins.items():
                if node is not None:
                    node_to_comps.setdefault(node, []).append((comp.name, pin))

        # If this is a fresh layout (nothing placed yet), do full topological placement
        if not self._placed:
            # Find signal chain: start from first voltage source or first component
            start = None
            for c in comps:
                if c.ctype == 'vsrc':
                    start = c; break
            if not start:
                start = comps[0]

            placed_order = []
            visited = set()

            def _chain(comp, depth=0):
                if comp.name in visited or depth > 50: return
                visited.add(comp.name)
                placed_order.append(comp)
                # Follow connections from right pin(s) to find next component
                for pin, node in comp.pins.items():
                    if node is None: continue
                    for cn, cp in node_to_comps.get(node, []):
                        if cn not in visited and cn in self.components:
                            _chain(self.components[cn], depth + 1)

            _chain(start)
            # Add any unvisited components
            for c in comps:
                if c.name not in visited:
                    placed_order.append(c)

            # Layout: sources on left, series chain going right, return on bottom
            # Separate ground-connected and signal-path components
            gnd_comps = set()
            sig_comps = []
            for c in placed_order:
                has_gnd = any(n == '0' for n in c.pins.values() if n is not None)
                if c.ctype in ('vsrc', 'isrc') and has_gnd:
                    gnd_comps.add(c.name)
                sig_comps.append(c)

            # Place in an L-shaped or rectangular path
            spacing = 180
            n = len(sig_comps)

            if n <= 4:
                # Simple loop: top row left→right, source on left with ground return below
                src_comps = [c for c in sig_comps if c.ctype in ('vsrc', 'isrc')]
                other_comps = [c for c in sig_comps if c.ctype not in ('vsrc', 'isrc')]

                # Place source(s) on the left, rotated vertical
                sx = 0
                for i, sc in enumerate(src_comps):
                    sc.x = sx
                    sc.y = 0
                    sc.rotation = 90
                    self._placed.add(sc.name)

                # Place other components in a horizontal chain to the right
                for i, c in enumerate(other_comps):
                    c.x = spacing + i * spacing
                    c.y = 0
                    c.rotation = 0
                    self._placed.add(c.name)
            else:
                # Larger circuits: 2 rows
                top_n = (n + 1) // 2
                for i, c in enumerate(sig_comps):
                    if i < top_n:
                        c.x = i * spacing
                        c.y = 0
                    else:
                        c.x = (top_n - 1 - (i - top_n)) * spacing
                        c.y = spacing
                    c.rotation = 0
                    self._placed.add(c.name)

        else:
            # Incremental: place new components offset from existing
            max_x = max((c.x for c in comps if c.name in self._placed), default=0)
            for i, c in enumerate(new_comps):
                c.x = max_x + 180 + i * 180
                c.y = 0
                c.rotation = 0
                self._placed.add(c.name)

        # Center camera on all components
        if comps:
            cx = sum(c.x for c in comps) / len(comps)
            cy = sum(c.y for c in comps) / len(comps)
            self.cam_x = -cx + 150
            self.cam_y = -cy + 100

    def _comp_size(self, comp):
        """Get (w, h) for a component's body."""
        if comp.ctype in ('npn', 'pnp', 'opamp'):
            return 80, 70
        return 80, 50

    def _pin_pos(self, comp, pin_name):
        """Get absolute position of a component pin in world coordinates, with rotation."""
        pins = list(comp.pins.keys())
        idx = pins.index(pin_name) if pin_name in pins else 0
        n_pins = len(pins)
        w, h = self._comp_size(comp)
        lead = 20  # lead wire length

        # Compute unrotated pin position relative to component origin
        lx, ly = 0, 0
        if n_pins == 2:
            if idx == 0: lx, ly = -lead, h/2
            else: lx, ly = w + lead, h/2
        elif n_pins == 3:
            if comp.ctype in ('npn', 'pnp'):
                if pin_name == 'B': lx, ly = -lead, h/2
                elif pin_name == 'C': lx, ly = w + lead, 12
                else: lx, ly = w + lead, h - 12
            else:  # opamp
                if pin_name == '+': lx, ly = -lead, 18
                elif pin_name == '-': lx, ly = -lead, h - 18
                else: lx, ly = w + lead, h/2
        else:
            lx, ly = 0, 0

        # Apply rotation around center of component body
        rot = getattr(comp, 'rotation', 0)
        if rot:
            cx, cy = w/2, h/2
            dx, dy = lx - cx, ly - cy
            rad = math.radians(rot)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            lx = cx + dx * cos_r - dy * sin_r
            ly = cy + dx * sin_r + dy * cos_r

        return comp.x + lx, comp.y + ly

    def _all_pin_positions(self):
        """Get list of (comp_name, pin_name, x, y) for all pins."""
        result = []
        for comp in self.components.values():
            for pin in comp.pins:
                px, py = self._pin_pos(comp, pin)
                result.append((comp.name, pin, px, py))
        return result

    def _hit_pin(self, wx, wy, radius=12):
        """Find pin under world coordinates. Returns (comp_name, pin_name) or None."""
        best = None; best_d = radius
        for cn, pn, px, py in self._all_pin_positions():
            d = math.sqrt((wx-px)**2 + (wy-py)**2)
            if d < best_d:
                best = (cn, pn); best_d = d
        return best

    def _hit_comp(self, wx, wy):
        """Find component whose body contains world point."""
        for comp in reversed(list(self.components.values())):
            w, h = self._comp_size(comp)
            margin = 10
            if comp.rotation == 90:
                bx, by, bw, bh = comp.x - margin, comp.y - margin, h + 2*margin, w + 2*margin
            else:
                bx, by, bw, bh = comp.x - margin, comp.y - margin, w + 2*margin, h + 2*margin
            if bx <= wx <= bx + bw and by <= wy <= by + bh:
                return comp
        return None

    def _screen_to_world(self, sx, sy):
        """Convert screen coords to world coords."""
        w, h = self.width(), self.height()
        wx = (sx - w/2 - self.cam_x) / self.zoom
        wy = (sy - h/2 - self.cam_y) / self.zoom
        return wx, wy

    def _snap_to_grid(self, x, y, grid=20):
        return round(x / grid) * grid, round(y / grid) * grid

    def add_overlay(self, name, fn):
        self._overlays[name] = fn

    def remove_overlay(self, name):
        self._overlays.pop(name, None)

    def screenshot(self, path):
        img = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        self._paint_scene(p, self.width(), self.height())
        p.end()
        img.save(path)

    # ── Paint ──────────────────────────────────────────────────

    def _paint_scene(self, p, w, h):
        p.setRenderHint(QPainter.Antialiasing)
        p.save()
        p.translate(w/2 + self.cam_x, h/2 + self.cam_y)
        p.scale(self.zoom, self.zoom)

        # Grid
        if self.show_grid:
            p.setPen(QPen(QColor(225, 228, 235), 0.5))
            gs = self.grid_size
            # Only draw grid in visible area
            for gx in range(-20, 30):
                p.drawLine(gx*gs, -20*gs, gx*gs, 20*gs)
            for gy in range(-20, 30):
                p.drawLine(-20*gs, gy*gs, 20*gs, gy*gs)

        # Wires (connections between pins sharing a node)
        self._draw_wires(p)

        # Components
        for name, comp in self.components.items():
            is_hover = (name == self._hover_comp)
            self._draw_component(p, comp, highlight=is_hover)

        # Pin dots (show all pins with connection state)
        self._draw_pins(p)

        # Ground symbols at ground node positions
        self._draw_ground_nodes(p)

        # Wire-in-progress (user is drawing a wire)
        if self._mode == 'wire_start' and self._wire_start_pin:
            cn, pn = self._wire_start_pin
            if cn in self.components:
                sx, sy = self._pin_pos(self.components[cn], pn)
                mx, my = self._mouse_world
                p.setPen(QPen(QColor(60, 160, 80, 180), 2.0, Qt.DashLine))
                # Manhattan preview
                p.drawLine(QPointF(sx, sy), QPointF(mx, sy))
                p.drawLine(QPointF(mx, sy), QPointF(mx, my))

        p.restore()

        # Overlays (screen coords)
        for name, fn in self._overlays.items():
            try: fn(p, w, h)
            except: pass

        # HUD
        self._draw_hud(p, w, h)

    def _draw_wires(self, p):
        """Draw wires between pins that share the same node. Uses Manhattan routing."""
        # Group pins by node
        node_pins = {}
        for comp in self.components.values():
            for pin, node in comp.pins.items():
                if node is not None:
                    px, py = self._pin_pos(comp, pin)
                    node_pins.setdefault(node, []).append((px, py, comp.name, pin))

        for node, pin_list in node_pins.items():
            if len(pin_list) < 2: continue

            # Color: ground wires darker, signal wires blue-gray
            if node == '0':
                p.setPen(QPen(QColor(100, 115, 140), 2.0))
            else:
                p.setPen(QPen(QColor(70, 90, 120), 2.0))

            # Sort pins left-to-right for cleaner routing
            pin_list.sort(key=lambda t: (t[0], t[1]))

            # Chain routing: connect each pin to the next in sorted order
            # with Manhattan paths (horizontal first, then vertical)
            for i in range(len(pin_list) - 1):
                x1, y1 = pin_list[i][0], pin_list[i][1]
                x2, y2 = pin_list[i+1][0], pin_list[i+1][1]

                if abs(y1 - y2) < 2:
                    # Same height — straight horizontal
                    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                else:
                    # Manhattan: horizontal to midpoint, then vertical, then horizontal
                    mid_x = (x1 + x2) / 2
                    p.drawLine(QPointF(x1, y1), QPointF(mid_x, y1))
                    p.drawLine(QPointF(mid_x, y1), QPointF(mid_x, y2))
                    p.drawLine(QPointF(mid_x, y2), QPointF(x2, y2))

            # Junction dots where 3+ wires meet
            if len(pin_list) >= 3:
                for x, y, _, _ in pin_list[1:-1]:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(60, 80, 120))
                    p.drawEllipse(QPointF(x, y), 3.5, 3.5)

    def _draw_pins(self, p):
        """Draw pin indicators — small circles, colored by state."""
        for comp in self.components.values():
            for pin, node in comp.pins.items():
                px, py = self._pin_pos(comp, pin)
                is_connected = node is not None
                is_hovered = (self._hover_pin == (comp.name, pin))
                is_wire_start = (self._wire_start_pin == (comp.name, pin))

                if is_wire_start:
                    p.setPen(QPen(QColor(60, 180, 80), 2))
                    p.setBrush(QColor(60, 180, 80, 100))
                    p.drawEllipse(QPointF(px, py), 6, 6)
                elif is_hovered:
                    p.setPen(QPen(QColor(50, 130, 200), 2))
                    p.setBrush(QColor(50, 130, 200, 80))
                    p.drawEllipse(QPointF(px, py), 6, 6)
                elif is_connected:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(60, 80, 120))
                    p.drawEllipse(QPointF(px, py), 3, 3)
                else:
                    # Unconnected pin — red warning dot
                    p.setPen(QPen(QColor(200, 80, 60), 1.5))
                    p.setBrush(QColor(200, 80, 60, 40))
                    p.drawEllipse(QPointF(px, py), 5, 5)

    def _draw_ground_nodes(self, p):
        """Draw ground symbol at every pin connected to node '0'."""
        drawn_positions = set()
        for comp in self.components.values():
            for pin, node in comp.pins.items():
                if node == '0':
                    px, py = self._pin_pos(comp, pin)
                    # Snap to avoid duplicate symbols at same spot
                    key = (round(px/10)*10, round(py/10)*10)
                    if key not in drawn_positions:
                        drawn_positions.add(key)
                        self._draw_ground_symbol(p, px, py)

    def _draw_hud(self, p, w, h):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 200))
        p.drawRoundedRect(12, 12, 200, 55, 8, 8)
        p.setFont(QFont("Consolas", 9, QFont.Bold))
        p.setPen(QColor(50, 55, 70))
        n_comp = len(self.components)
        n_nodes = len(set(n for c in self.components.values() for n in c.pins.values() if n))
        p.drawText(20, 16, 185, 16, Qt.AlignVCenter, f"{n_comp} components · {n_nodes} nodes")
        p.setFont(QFont("Consolas", 8))
        p.setPen(QColor(100, 110, 130))
        types = {}
        for c in self.components.values():
            types[c.ctype] = types.get(c.ctype, 0) + 1
        summary = ', '.join(f"{v}{k[0].upper()}" for k, v in types.items())
        p.drawText(20, 32, 185, 14, Qt.AlignVCenter, summary or "Empty circuit")
        # Mode indicator
        p.setFont(QFont("Consolas", 7))
        if self._mode == 'wire_start':
            p.setPen(QColor(60, 160, 80))
            p.drawText(20, 46, 185, 12, Qt.AlignVCenter, "● WIRING — click target pin (Esc to cancel)")
        elif self._drag_comp:
            p.setPen(QColor(50, 100, 180))
            p.drawText(20, 46, 185, 12, Qt.AlignVCenter, f"● Moving {self._drag_comp.name}")
        else:
            p.setPen(QColor(140, 150, 170))
            p.drawText(20, 46, 185, 12, Qt.AlignVCenter, "Drag=move · Click pin=wire · RMB=pan")

    def _draw_component(self, p, comp, highlight=False):
        """Draw a single component with rotation support."""
        x, y = comp.x, comp.y
        w, h = self._comp_size(comp)
        rot = getattr(comp, 'rotation', 0)

        color = COMP_COLORS.get(comp.ctype, QColor(120, 120, 140))

        # Highlight border when hovered (in untransformed space for correct bounds)
        if highlight:
            p.save()
            if rot:
                p.translate(x + w/2, y + h/2)
                p.rotate(rot)
                p.translate(-(x + w/2), -(y + h/2))
            p.setPen(QPen(QColor(50, 130, 200, 120), 3))
            p.setBrush(QColor(50, 130, 200, 15))
            p.drawRoundedRect(int(x) - 6, int(y) - 6, w + 12, h + 12, 8, 8)
            p.restore()

        # Apply rotation around component center
        p.save()
        if rot:
            p.translate(x + w/2, y + h/2)
            p.rotate(rot)
            p.translate(-(x + w/2), -(y + h/2))

        if comp.ctype == 'resistor':
            self._draw_resistor(p, x, y, w, h, color, comp)
        elif comp.ctype == 'capacitor':
            self._draw_capacitor(p, x, y, w, h, color, comp)
        elif comp.ctype == 'inductor':
            self._draw_inductor(p, x, y, w, h, color, comp)
        elif comp.ctype in ('vsrc', 'isrc'):
            self._draw_source(p, x, y, w, h, color, comp)
        elif comp.ctype == 'diode':
            self._draw_diode(p, x, y, w, h, color, comp)
        elif comp.ctype in ('npn', 'pnp'):
            self._draw_transistor(p, x, y, w, h, color, comp)
        elif comp.ctype == 'opamp':
            self._draw_opamp(p, x, y, w, h, color, comp)
        else:
            p.setPen(QPen(color, 2))
            p.setBrush(QColor(255, 255, 255, 200))
            p.drawRoundedRect(int(x), int(y), w, h, 6, 6)

        # Label (drawn rotated with the component)
        unit = {'resistor':'Ω','capacitor':'F','inductor':'H','vsrc':'V','isrc':'A'}.get(comp.ctype,'')
        val_str = _si(comp.value, unit) if comp.value else ''
        p.setFont(QFont("Consolas", 8, QFont.Bold))
        p.setPen(QColor(40, 50, 70))
        p.drawText(int(x), int(y) - 14, w, 14, Qt.AlignCenter, comp.name)
        if val_str:
            p.setFont(QFont("Consolas", 7))
            p.setPen(QColor(90, 100, 120))
            p.drawText(int(x), int(y) + h + 2, w, 12, Qt.AlignCenter, val_str)

        p.restore()

    def _draw_resistor(self, p, x, y, w, h, color, comp):
        """Zigzag resistor symbol."""
        cx, cy = x + w/2, y + h/2
        p.setPen(QPen(color, 2.2))
        # Lead lines (extend to pin positions at -20 and w+20)
        p.drawLine(QPointF(x - 20, cy), QPointF(x + 10, cy))
        p.drawLine(QPointF(x + w - 10, cy), QPointF(x + w + 20, cy))
        # Zigzag body
        path = QPainterPath()
        path.moveTo(x + 10, cy)
        zw = (w - 20) / 6
        for i in range(6):
            xp = x + 10 + (i + 0.5) * zw
            yp = cy + (12 if i % 2 == 0 else -12)
            path.lineTo(xp, yp)
        path.lineTo(x + w - 10, cy)
        p.drawPath(path)

    def _draw_capacitor(self, p, x, y, w, h, color, comp):
        cx, cy = x + w/2, y + h/2
        p.setPen(QPen(color, 2.2))
        # Leads
        p.drawLine(QPointF(x - 20, cy), QPointF(cx - 6, cy))
        p.drawLine(QPointF(cx + 6, cy), QPointF(x + w + 20, cy))
        # Plates
        p.setPen(QPen(color, 3))
        p.drawLine(QPointF(cx - 6, cy - 16), QPointF(cx - 6, cy + 16))
        p.drawLine(QPointF(cx + 6, cy - 16), QPointF(cx + 6, cy + 16))

    def _draw_inductor(self, p, x, y, w, h, color, comp):
        cx, cy = x + w/2, y + h/2
        p.setPen(QPen(color, 2.2))
        p.setBrush(Qt.NoBrush)
        # Leads
        p.drawLine(QPointF(x - 20, cy), QPointF(x + 12, cy))
        p.drawLine(QPointF(x + w - 12, cy), QPointF(x + w + 20, cy))
        # Coil arcs
        n_coils = 4
        coil_w = (w - 24) / n_coils
        for i in range(n_coils):
            rx = x + 12 + i * coil_w
            p.drawArc(int(rx), int(cy - 10), int(coil_w), 20, 0, 180 * 16)

    def _draw_source(self, p, x, y, w, h, color, comp):
        cx, cy = x + w/2, y + h/2
        r = min(w, h) / 2 - 4
        p.setPen(QPen(color, 2))
        p.setBrush(QColor(255, 255, 255, 220))
        p.drawEllipse(QPointF(cx, cy), r, r)
        # Leads
        p.drawLine(QPointF(x - 20, cy), QPointF(cx - r, cy))
        p.drawLine(QPointF(cx + r, cy), QPointF(x + w + 20, cy))
        # + / - or arrow
        p.setFont(QFont("Consolas", 12, QFont.Bold))
        p.setPen(color)
        if comp.ctype == 'vsrc':
            p.drawText(int(cx - 12), int(cy - 14), 12, 14, Qt.AlignCenter, "+")
            p.drawText(int(cx + 2), int(cy - 14), 12, 14, Qt.AlignCenter, "−")
        else:
            # Current arrow
            p.setPen(QPen(color, 2))
            p.drawLine(QPointF(cx - 8, cy), QPointF(cx + 8, cy))
            p.drawLine(QPointF(cx + 4, cy - 4), QPointF(cx + 8, cy))
            p.drawLine(QPointF(cx + 4, cy + 4), QPointF(cx + 8, cy))

    def _draw_diode(self, p, x, y, w, h, color, comp):
        cx, cy = x + w/2, y + h/2
        p.setPen(QPen(color, 2.2))
        # Leads
        p.drawLine(QPointF(x - 20, cy), QPointF(cx - 10, cy))
        p.drawLine(QPointF(cx + 10, cy), QPointF(x + w + 20, cy))
        # Triangle
        tri = QPolygonF([QPointF(cx - 10, cy - 12), QPointF(cx - 10, cy + 12), QPointF(cx + 10, cy)])
        p.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
        p.drawPolygon(tri)
        # Cathode bar
        p.drawLine(QPointF(cx + 10, cy - 12), QPointF(cx + 10, cy + 12))

    def _draw_transistor(self, p, x, y, w, h, color, comp):
        cx, cy = x + w/2, y + h/2
        p.setPen(QPen(color, 2))
        p.setBrush(QColor(255, 255, 255, 200))
        p.drawEllipse(QPointF(cx, cy), 24, 28)
        # Base line
        p.drawLine(QPointF(x - 20, cy), QPointF(cx - 12, cy))
        p.drawLine(QPointF(cx - 12, cy - 16), QPointF(cx - 12, cy + 16))
        # Collector
        p.drawLine(QPointF(cx - 12, cy - 10), QPointF(cx + 12, cy - 20))
        p.drawLine(QPointF(cx + 12, cy - 20), QPointF(x + w + 20, y + 12))
        # Emitter
        p.drawLine(QPointF(cx - 12, cy + 10), QPointF(cx + 12, cy + 20))
        p.drawLine(QPointF(cx + 12, cy + 20), QPointF(x + w + 20, y + h - 12))
        # Arrow on emitter (NPN)
        if comp.ctype == 'npn':
            p.drawLine(QPointF(cx + 6, cy + 18), QPointF(cx + 12, cy + 20))
            p.drawLine(QPointF(cx + 10, cy + 13), QPointF(cx + 12, cy + 20))

    def _draw_opamp(self, p, x, y, w, h, color, comp):
        cx, cy = x + w/2, y + h/2
        p.setPen(QPen(color, 2))
        p.setBrush(QColor(255, 255, 255, 220))
        # Triangle body
        tri = QPolygonF([QPointF(x + 8, y), QPointF(x + 8, y + h), QPointF(x + w - 8, cy)])
        p.drawPolygon(tri)
        # Input labels
        p.setFont(QFont("Consolas", 10, QFont.Bold))
        p.setPen(color)
        p.drawText(int(x + 12), int(y + 8), 20, 20, Qt.AlignCenter, "+")
        p.drawText(int(x + 12), int(y + h - 28), 20, 20, Qt.AlignCenter, "−")
        # Leads
        p.setPen(QPen(color, 2))
        p.drawLine(QPointF(x - 20, y + 18), QPointF(x + 8, y + 18))
        p.drawLine(QPointF(x - 20, y + h - 18), QPointF(x + 8, y + h - 18))
        p.drawLine(QPointF(x + w - 8, cy), QPointF(x + w + 20, cy))

    def _draw_ground_symbol(self, p, x, y):
        p.setPen(QPen(QColor(100, 110, 130), 2))
        p.drawLine(QPointF(x, y), QPointF(x, y + 8))
        p.drawLine(QPointF(x - 10, y + 8), QPointF(x + 10, y + 8))
        p.drawLine(QPointF(x - 6, y + 13), QPointF(x + 6, y + 13))
        p.drawLine(QPointF(x - 2, y + 18), QPointF(x + 2, y + 18))

    def paintEvent(self, event):
        p = QPainter(self)
        self._paint_scene(p, self.width(), self.height())
        p.end()

    def mousePressEvent(self, e):
        wx, wy = self._screen_to_world(e.x(), e.y())

        if e.button() == Qt.RightButton or e.button() == Qt.MiddleButton:
            # Pan
            self._mode = 'pan'
            self._lmx = e.x(); self._lmy = e.y()
            e.accept(); return

        if e.button() == Qt.LeftButton:
            # Check if clicking a pin (wire mode)
            hit_pin = self._hit_pin(wx, wy)

            if self._mode == 'wire_start' and self._wire_start_pin:
                # Second click — complete wire
                if hit_pin and hit_pin != self._wire_start_pin:
                    cn_a, pn_a = self._wire_start_pin
                    cn_b, pn_b = hit_pin
                    if self.on_connect:
                        self.on_connect(cn_a, pn_a, cn_b, pn_b)
                self._mode = 'idle'
                self._wire_start_pin = None
                e.accept(); return

            if hit_pin:
                # Start wiring from this pin
                self._mode = 'wire_start'
                self._wire_start_pin = hit_pin
                e.accept(); return

            # Check if clicking a component (drag mode)
            hit_comp = self._hit_comp(wx, wy)
            if hit_comp:
                self._mode = 'drag_comp'
                self._drag_comp = hit_comp
                self._drag_offset = (wx - hit_comp.x, wy - hit_comp.y)
                self._lmx = e.x(); self._lmy = e.y()
                e.accept(); return

            # Nothing hit — pan
            self._mode = 'pan'
            self._lmx = e.x(); self._lmy = e.y()
        e.accept()

    def mouseReleaseEvent(self, e):
        if self._mode == 'drag_comp' and self._drag_comp:
            # Snap to grid
            self._drag_comp.x, self._drag_comp.y = self._snap_to_grid(
                self._drag_comp.x, self._drag_comp.y)
        if self._mode != 'wire_start':
            self._mode = 'idle'
        self._drag_comp = None
        e.accept()

    def mouseMoveEvent(self, e):
        wx, wy = self._screen_to_world(e.x(), e.y())
        self._mouse_world = (wx, wy)

        if self._mode == 'pan':
            self.cam_x += (e.x() - self._lmx)
            self.cam_y += (e.y() - self._lmy)
            self._lmx = e.x(); self._lmy = e.y()
        elif self._mode == 'drag_comp' and self._drag_comp:
            self._drag_comp.x = wx - self._drag_offset[0]
            self._drag_comp.y = wy - self._drag_offset[1]
        else:
            # Hover detection
            self._hover_pin = self._hit_pin(wx, wy)
            comp = self._hit_comp(wx, wy)
            self._hover_comp = comp.name if comp else None
        e.accept()

    def wheelEvent(self, e):
        # Zoom toward cursor
        old_zoom = self.zoom
        factor = 1.12 if e.angleDelta().y() > 0 else 0.89
        self.zoom = max(0.15, min(6.0, self.zoom * factor))
        # Adjust cam so point under cursor stays fixed
        sx, sy = e.position().x(), e.position().y()
        w, h = self.width(), self.height()
        self.cam_x = sx - w/2 - (sx - w/2 - self.cam_x) * self.zoom / old_zoom
        self.cam_y = sy - h/2 - (sy - h/2 - self.cam_y) * self.zoom / old_zoom
        e.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._mode = 'idle'
            self._wire_start_pin = None
            self._drag_comp = None
        elif e.key() == Qt.Key_R and self._hover_comp:
            # Rotate component
            comp = self.components.get(self._hover_comp)
            if comp:
                comp.rotation = (comp.rotation + 90) % 360
        e.accept()


# ═══════════════════════════════════════════════════════════════
#  CIRCUITLAB — MAIN API SINGLETON
# ═══════════════════════════════════════════════════════════════

class CircuitLab:
    """
    Main circuit API. Registered as `circ` in the namespace.

    QUICK REFERENCE (for LLM context):
        circ.add(name, type, value, **kw)      — add component
        circ.connect(pin_a, pin_b)              — wire two pins
        circ.ground(pin)                        — connect pin to ground
        circ.remove(name)                       — remove component
        circ.clear()                            — remove all
        circ.load(preset)                       — load preset circuit
        circ.load_netlist(text)                 — parse SPICE netlist

        circ.dc()                               — DC operating point
        circ.ac(start, stop, points)           — AC sweep
        circ.tran(tstep, tstop)                — transient sim
        circ.dc_sweep(src, v1, v2, step)       — DC sweep

        circ.overlay_waveform(node, sim)       — show voltage trace
        circ.overlay_bode(node)                — show Bode plot
        circ.overlay_dc_table()                — show DC OP table
        circ.label_voltages()                  — label node voltages
        circ.label_currents()                  — label branch currents
        circ.remove_overlay(name)

        circ.export_spice(path)                — save netlist
        circ.components                        — component dict
        circ.nodes                             — set of node names
        circ.results                           — simulation results
        circ.describe()                        — summary string
        circ.viewer                            — SchematicViewer widget
        circ.log(msg)                          — append to log
    """

    def __init__(self, viewer, log_widget=None):
        self.viewer = viewer
        self._log = log_widget
        self.components = OrderedDict()
        self._node_counter = 1
        self._auto_nodes = {}  # (pin_a, pin_b) → node_name
        self.results = {'dc': {}, 'ac': {}, 'tran': {}}

    def log(self, msg):
        if self._log:
            self._log.append(f"[circ] {msg}")
        print(f"[circ] {msg}")

    # ── Component Management ──────────────────────────────────

    def add(self, name, ctype, value=0, **kwargs):
        """Add a component.
        Types: resistor, capacitor, inductor, vsrc, isrc, diode, npn, pnp, opamp
        """
        comp = Component(name, ctype, value, **kwargs)
        self.components[name] = comp
        self.viewer.set_circuit(self.components)
        self.log(f"Added {name}: {ctype} = {_si(value, '')}")
        return self

    def remove(self, name):
        """Remove a component."""
        if name in self.components:
            del self.components[name]
            self.viewer.set_circuit(self.components)
            self.log(f"Removed {name}")
        return self

    def clear(self):
        """Remove all components and reset."""
        self.components.clear()
        self._node_counter = 1
        self._auto_nodes.clear()
        self.results = {'dc': {}, 'ac': {}, 'tran': {}}
        self.viewer._placed.clear()
        self.viewer.set_circuit(self.components)
        self.viewer._overlays.clear()
        self.log("Circuit cleared")
        return self

    def connect(self, pin_a_str, pin_b_str):
        """Connect two pins. Format: 'CompName.PinName' e.g. 'R1.1', 'V1.+'
        Pins connected together share the same node."""
        comp_a, pin_a = self._resolve_pin(pin_a_str)
        comp_b, pin_b = self._resolve_pin(pin_b_str)
        if not comp_a or not comp_b:
            self.log(f"Connect failed: {pin_a_str} → {pin_b_str}")
            return self

        node_a = comp_a.pins.get(pin_a)
        node_b = comp_b.pins.get(pin_b)

        if node_a and node_b and node_a != node_b:
            # Merge: rename all node_b → node_a
            for comp in self.components.values():
                for p in comp.pins:
                    if comp.pins[p] == node_b:
                        comp.pins[p] = node_a
            node = node_a
        elif node_a:
            node = node_a
        elif node_b:
            node = node_b
        else:
            node = f"n{self._node_counter}"
            self._node_counter += 1

        comp_a.pins[pin_a] = node
        comp_b.pins[pin_b] = node
        self.viewer.set_circuit(self.components)
        self.log(f"Connected {pin_a_str} — {pin_b_str} → {node}")
        return self

    def ground(self, pin_str):
        """Connect a pin to ground (node '0')."""
        comp, pin = self._resolve_pin(pin_str)
        if not comp:
            self.log(f"Ground failed: {pin_str}")
            return self

        old_node = comp.pins.get(pin)
        if old_node and old_node != '0':
            # Merge old_node into ground
            for c in self.components.values():
                for p in c.pins:
                    if c.pins[p] == old_node:
                        c.pins[p] = '0'
        comp.pins[pin] = '0'
        self.viewer.set_circuit(self.components)
        self.log(f"Grounded {pin_str}")
        return self

    def node(self, pin_str, node_name):
        """Assign a named node to a pin. e.g. circ.node('R1.2', 'out')"""
        comp, pin = self._resolve_pin(pin_str)
        if not comp:
            self.log(f"Node assign failed: {pin_str}")
            return self

        old_node = comp.pins.get(pin)
        if old_node and old_node != node_name:
            for c in self.components.values():
                for p in c.pins:
                    if c.pins[p] == old_node:
                        c.pins[p] = node_name
        comp.pins[pin] = node_name
        self.viewer.set_circuit(self.components)
        self.log(f"Node {pin_str} → {node_name}")
        return self

    def _resolve_pin(self, pin_str):
        """Parse 'CompName.PinName' → (Component, pin_name) or (None, None)."""
        parts = pin_str.split('.', 1)
        if len(parts) != 2:
            self.log(f"Invalid pin format: {pin_str} (use 'CompName.PinName')")
            return None, None
        comp_name, pin_name = parts
        comp = self.components.get(comp_name)
        if not comp:
            self.log(f"Component not found: {comp_name}")
            return None, None
        if pin_name not in comp.pins:
            self.log(f"Pin '{pin_name}' not found on {comp_name}. Available: {list(comp.pins.keys())}")
            return None, None
        return comp, pin_name

    @property
    def nodes(self):
        """Get all unique node names."""
        s = set()
        for comp in self.components.values():
            for n in comp.pins.values():
                if n is not None: s.add(n)
        return s

    def describe(self):
        """Human-readable circuit summary."""
        lines = [f"Circuit: {len(self.components)} components, {len(self.nodes)} nodes\n"]
        for comp in self.components.values():
            lines.append(f"  {comp.describe()}")
        return '\n'.join(lines)

    # ── Presets ────────────────────────────────────────────────

    PRESETS = {
        "rc_lowpass": [
            ("V1", "vsrc", 5.0, {'+': 'vin', '-': '0'}, {'ac_amp': 1.0}),
            ("R1", "resistor", 1000, {'1': 'vin', '2': 'out'}, {}),
            ("C1", "capacitor", 100e-9, {'1': 'out', '2': '0'}, {}),
        ],
        "rl_highpass": [
            ("V1", "vsrc", 5.0, {'+': 'vin', '-': '0'}, {'ac_amp': 1.0}),
            ("C1", "capacitor", 100e-9, {'1': 'vin', '2': 'out'}, {}),
            ("R1", "resistor", 1000, {'1': 'out', '2': '0'}, {}),
        ],
        "voltage_divider": [
            ("V1", "vsrc", 10.0, {'+': 'vin', '-': '0'}, {}),
            ("R1", "resistor", 10000, {'1': 'vin', '2': 'out'}, {}),
            ("R2", "resistor", 10000, {'1': 'out', '2': '0'}, {}),
        ],
        "rlc_bandpass": [
            ("V1", "vsrc", 0, {'+': 'vin', '-': '0'}, {'ac_amp': 1.0}),
            ("R1", "resistor", 100, {'1': 'vin', '2': 'n1'}, {}),
            ("L1", "inductor", 10e-3, {'1': 'n1', '2': 'out'}, {}),
            ("C1", "capacitor", 100e-9, {'1': 'out', '2': '0'}, {}),
        ],
        "inverting_amp": [
            ("V1", "vsrc", 0, {'+': 'vin', '-': '0'}, {'ac_amp': 1.0}),
            ("R1", "resistor", 10000, {'1': 'vin', '2': 'inv'}, {}),
            ("R2", "resistor", 100000, {'1': 'inv', '2': 'out'}, {}),
            ("U1", "opamp", 0, {'+': '0', '-': 'inv', 'out': 'out'}, {}),
        ],
        "half_wave_rect": [
            ("V1", "vsrc", 0, {'+': 'vin', '-': '0'}, {'ac_amp': 5.0, 'ac_freq': 60}),
            ("D1", "diode", 0, {'A': 'vin', 'K': 'out'}, {}),
            ("R1", "resistor", 1000, {'1': 'out', '2': '0'}, {}),
            ("C1", "capacitor", 10e-6, {'1': 'out', '2': '0'}, {}),
        ],
        "common_emitter": [
            ("V1", "vsrc", 12.0, {'+': 'vcc', '-': '0'}, {}),
            ("V2", "vsrc", 0, {'+': 'vin', '-': '0'}, {'ac_amp': 0.01, 'ac_freq': 1000}),
            ("R1", "resistor", 100000, {'1': 'vcc', '2': 'base'}, {}),
            ("R2", "resistor", 22000, {'1': 'base', '2': '0'}, {}),
            ("RC", "resistor", 4700, {'1': 'vcc', '2': 'out'}, {}),
            ("RE", "resistor", 1000, {'1': 'emit', '2': '0'}, {}),
            ("Q1", "npn", 0, {'B': 'base', 'C': 'out', 'E': 'emit'}, {}),
            ("Cin", "capacitor", 1e-6, {'1': 'vin', '2': 'base'}, {}),
        ],
    }

    def load(self, name):
        """Load a preset circuit by name."""
        key = name.lower().replace(' ', '').replace('-', '').replace('_', '')
        for k, preset in self.PRESETS.items():
            if key in k.replace('_', '') or k.replace('_', '') in key:
                self.clear()
                for item in preset:
                    comp_name, ctype, value, pins, params = item
                    comp = Component(comp_name, ctype, value, **params)
                    comp.pins = pins
                    self.components[comp_name] = comp
                self.viewer.set_circuit(self.components)
                self.log(f"Loaded preset: {name} ({len(self.components)} components)")
                return self
        available = ', '.join(self.PRESETS.keys())
        self.log(f"Unknown preset: {name}. Available: {available}")
        return self

    def load_netlist(self, text):
        """Parse SPICE-format netlist text."""
        self.clear()
        parsed = parse_spice_netlist(text)
        for name, ctype, node_list, value, params in parsed:
            comp = Component(name, ctype, value, **params)
            pins = list(comp.pins.keys())
            for i, node in enumerate(node_list):
                if i < len(pins):
                    comp.pins[pins[i]] = node
            self.components[name] = comp
        self.viewer.set_circuit(self.components)
        self.log(f"Loaded netlist: {len(self.components)} components")
        return self

    # ── Simulation ─────────────────────────────────────────────

    def dc(self):
        """Run DC operating point analysis."""
        solver = MNASolver(self.components)
        result = solver.dc()
        self.results['dc'] = result
        nv = result['node_voltages']
        bc = result['branch_currents']
        self.log(f"DC OP: {', '.join(f'{k}={v:.4f}V' for k,v in nv.items() if k != '0')}")
        if bc:
            self.log(f"  Currents: {', '.join(f'{k}={_si(v,chr(65))}' for k,v in bc.items())}")
        return self

    def ac(self, start=1, stop=1e6, points=200):
        """Run AC frequency sweep."""
        freqs = np.logspace(np.log10(max(start, 0.01)), np.log10(stop), points)
        solver = MNASolver(self.components)
        ac_data = solver.ac(freqs)
        self.results['ac'] = {'freqs': freqs}
        self.results['ac'].update(ac_data)
        self.log(f"AC sweep: {_si(start,'Hz')} → {_si(stop,'Hz')}, {points} pts")
        return self

    def tran(self, tstep=1e-6, tstop=1e-3, tstart=0):
        """Run transient simulation."""
        solver = MNASolver(self.components)
        tran_data, time_pts = solver.tran(tstep, tstop, tstart)
        self.results['tran'] = {'time': time_pts}
        self.results['tran'].update(tran_data)
        self.log(f"Transient: {_si(tstart,'s')} → {_si(tstop,'s')}, step={_si(tstep,'s')}, {len(time_pts)} pts")
        return self

    def dc_sweep(self, src_name, v1, v2, step):
        """Sweep a voltage source and record all node voltages."""
        if src_name not in self.components:
            self.log(f"Source not found: {src_name}")
            return self
        original_val = self.components[src_name].value
        voltages = np.arange(v1, v2 + step/2, step)
        sweep_results = {'sweep_voltages': voltages}

        for v in voltages:
            self.components[src_name].value = v
            solver = MNASolver(self.components)
            dc = solver.dc()
            for node, voltage in dc['node_voltages'].items():
                sweep_results.setdefault(node, []).append(voltage)

        # Convert lists to arrays
        for k in sweep_results:
            if isinstance(sweep_results[k], list):
                sweep_results[k] = np.array(sweep_results[k])

        self.components[src_name].value = original_val
        self.results['dc_sweep'] = sweep_results
        self.log(f"DC sweep {src_name}: {v1}V → {v2}V, {len(voltages)} pts")
        return self

    # ── Overlays ───────────────────────────────────────────────

    def overlay_waveform(self, node_name, sim='tran', width=380, height=200):
        """Show waveform overlay for a node."""
        data = self.results.get(sim, {})
        if sim == 'tran':
            time = data.get('time')
            vals = data.get(node_name)
            if time is None or vals is None:
                self.log(f"No transient data for {node_name}. Run circ.tran() first.")
                return self

            def draw_wave(painter, w, h):
                ox, oy = w - width - 16, h - height - 16
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 255, 220))
                painter.drawRoundedRect(ox, oy, width, height, 8, 8)
                painter.setFont(QFont("Consolas", 8, QFont.Bold))
                painter.setPen(QColor(50, 60, 80))
                painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter,
                                f"V({node_name}) — Transient")
                pad_l, pad_r, pad_t, pad_b = 50, 16, 24, 30
                ax0 = ox + pad_l; ax1 = ox + width - pad_r
                ay0 = oy + pad_t; ay1 = oy + height - pad_b
                aw = ax1 - ax0; ah = ay1 - ay0
                # Axes
                painter.setPen(QPen(QColor(180, 185, 200), 1))
                painter.drawLine(ax0, ay1, ax1, ay1)
                painter.drawLine(ax0, ay0, ax0, ay1)
                # Data
                vmin, vmax = float(np.min(vals)), float(np.max(vals))
                if vmax - vmin < 1e-12: vmax = vmin + 1
                tmin, tmax = float(time[0]), float(time[-1])
                path = QPainterPath()
                path.moveTo(ax0, ay1 - (float(vals[0]) - vmin)/(vmax - vmin) * ah)
                for i in range(1, len(time)):
                    x = ax0 + (float(time[i]) - tmin) / (tmax - tmin) * aw
                    y = ay1 - (float(vals[i]) - vmin) / (vmax - vmin) * ah
                    path.lineTo(x, y)
                painter.setPen(QPen(QColor(50, 120, 200), 1.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
                # Labels
                painter.setFont(QFont("Consolas", 7))
                painter.setPen(QColor(100, 110, 130))
                painter.drawText(ax0 - 46, ay0, 44, 12, Qt.AlignRight | Qt.AlignVCenter, f"{vmax:.2g}V")
                painter.drawText(ax0 - 46, ay1 - 6, 44, 12, Qt.AlignRight | Qt.AlignVCenter, f"{vmin:.2g}V")
                painter.drawText(ax0, ay1 + 4, aw, 14, Qt.AlignCenter, f"Time ({_si(tmax - tmin, 's')})")

            self.viewer.add_overlay('waveform', draw_wave)
            self.log(f"Waveform overlay: V({node_name})")

        elif sim == 'dc_sweep':
            sweep_v = data.get('sweep_voltages')
            vals = data.get(node_name)
            if sweep_v is None or vals is None:
                self.log(f"No DC sweep data for {node_name}")
                return self

            def draw_dc(painter, w, h):
                ox, oy = w - width - 16, h - height - 16
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 255, 220))
                painter.drawRoundedRect(ox, oy, width, height, 8, 8)
                painter.setFont(QFont("Consolas", 8, QFont.Bold))
                painter.setPen(QColor(50, 60, 80))
                painter.drawText(ox+8, oy+4, width-16, 14, Qt.AlignVCenter, f"V({node_name}) — DC Sweep")
                pad_l, pad_r, pad_t, pad_b = 50, 16, 24, 30
                ax0=ox+pad_l; ax1=ox+width-pad_r; ay0=oy+pad_t; ay1=oy+height-pad_b
                aw=ax1-ax0; ah=ay1-ay0
                painter.setPen(QPen(QColor(180,185,200),1))
                painter.drawLine(ax0,ay1,ax1,ay1); painter.drawLine(ax0,ay0,ax0,ay1)
                vmin, vmax = float(np.min(vals)), float(np.max(vals))
                if vmax-vmin<1e-12: vmax=vmin+1
                xmin, xmax = float(sweep_v[0]), float(sweep_v[-1])
                path = QPainterPath()
                for i, (sv, v) in enumerate(zip(sweep_v, vals)):
                    x = ax0 + (float(sv)-xmin)/(xmax-xmin)*aw
                    y = ay1 - (float(v)-vmin)/(vmax-vmin)*ah
                    if i==0: path.moveTo(x,y)
                    else: path.lineTo(x,y)
                painter.setPen(QPen(QColor(200,80,60),1.5)); painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
                painter.setFont(QFont("Consolas",7)); painter.setPen(QColor(100,110,130))
                painter.drawText(ax0-46,ay0,44,12,Qt.AlignRight|Qt.AlignVCenter,f"{vmax:.2g}V")
                painter.drawText(ax0-46,ay1-6,44,12,Qt.AlignRight|Qt.AlignVCenter,f"{vmin:.2g}V")
                painter.drawText(ax0,ay1+4,aw,14,Qt.AlignCenter,"Sweep V")

            self.viewer.add_overlay('waveform', draw_dc)
            self.log(f"DC sweep overlay: V({node_name})")

        return self

    def overlay_bode(self, node_name, width=380, height=240):
        """Show Bode plot (magnitude + phase) overlay."""
        ac_data = self.results.get('ac', {})
        freqs = ac_data.get('freqs')
        vals = ac_data.get(node_name)
        if freqs is None or vals is None:
            self.log(f"No AC data for {node_name}. Run circ.ac() first.")
            return self

        mag_db = 20 * np.log10(np.abs(vals) + 1e-30)
        phase_deg = np.angle(vals, deg=True)

        def draw_bode(painter, w, h):
            ox, oy = w - width - 16, h - height - 16
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 225))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter,
                            f"Bode: V({node_name})")

            pad_l, pad_r, pad_t, pad_b = 50, 16, 22, 28
            ax0 = ox + pad_l; ax1 = ox + width - pad_r
            aw = ax1 - ax0
            mid_h = height // 2

            # Magnitude plot (top half)
            may0 = oy + pad_t; may1 = oy + mid_h - 8
            mah = may1 - may0
            painter.setPen(QPen(QColor(180, 185, 200), 0.5))
            painter.drawLine(ax0, may1, ax1, may1)
            painter.drawLine(ax0, may0, ax0, may1)

            mmin, mmax = float(np.min(mag_db)), float(np.max(mag_db))
            if mmax - mmin < 1: mmax = mmin + 1
            path_m = QPainterPath()
            log_fmin, log_fmax = np.log10(freqs[0]), np.log10(freqs[-1])
            for i in range(len(freqs)):
                x = ax0 + (np.log10(freqs[i]) - log_fmin) / (log_fmax - log_fmin) * aw
                y = may1 - (float(mag_db[i]) - mmin) / (mmax - mmin) * mah
                if i == 0: path_m.moveTo(x, y)
                else: path_m.lineTo(x, y)
            painter.setPen(QPen(QColor(50, 120, 200), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path_m)
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor(100, 110, 130))
            painter.drawText(ax0 - 46, may0, 44, 12, Qt.AlignRight | Qt.AlignVCenter, f"{mmax:.0f}dB")
            painter.drawText(ax0 - 46, may1 - 6, 44, 12, Qt.AlignRight | Qt.AlignVCenter, f"{mmin:.0f}dB")

            # Phase plot (bottom half)
            pay0 = oy + mid_h + 6; pay1 = oy + height - pad_b
            pah = pay1 - pay0
            painter.setPen(QPen(QColor(180, 185, 200), 0.5))
            painter.drawLine(ax0, pay1, ax1, pay1)
            painter.drawLine(ax0, pay0, ax0, pay1)

            pmin, pmax = float(np.min(phase_deg)), float(np.max(phase_deg))
            if pmax - pmin < 1: pmax = pmin + 1
            path_p = QPainterPath()
            for i in range(len(freqs)):
                x = ax0 + (np.log10(freqs[i]) - log_fmin) / (log_fmax - log_fmin) * aw
                y = pay1 - (float(phase_deg[i]) - pmin) / (pmax - pmin) * pah
                if i == 0: path_p.moveTo(x, y)
                else: path_p.lineTo(x, y)
            painter.setPen(QPen(QColor(200, 80, 60), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path_p)
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor(100, 110, 130))
            painter.drawText(ax0 - 46, pay0, 44, 12, Qt.AlignRight | Qt.AlignVCenter, f"{pmax:.0f}°")
            painter.drawText(ax0 - 46, pay1 - 6, 44, 12, Qt.AlignRight | Qt.AlignVCenter, f"{pmin:.0f}°")
            painter.drawText(ax0, pay1 + 4, aw, 14, Qt.AlignCenter, "Frequency (Hz)")

        self.viewer.add_overlay('bode', draw_bode)
        self.log(f"Bode overlay: V({node_name})")
        return self

    def overlay_dc_table(self, width=220, height=None):
        """Show DC operating point as a table overlay."""
        dc = self.results.get('dc', {})
        nv = dc.get('node_voltages', {})
        bc = dc.get('branch_currents', {})
        if not nv:
            self.log("No DC data. Run circ.dc() first.")
            return self

        n_lines = len(nv) + len(bc) + 1
        if height is None:
            height = 26 + n_lines * 15

        def draw_table(painter, w, h):
            ox, oy = 12, h - height - 12
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(ox, oy, width, height, 8, 8)
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox + 8, oy + 4, width - 16, 14, Qt.AlignVCenter, "DC Operating Point")
            y = oy + 22
            painter.setFont(QFont("Consolas", 8))
            for node, voltage in sorted(nv.items()):
                painter.setPen(QColor(50, 100, 180))
                painter.drawText(ox + 10, y, width - 20, 13, Qt.AlignVCenter,
                                f"V({node}) = {voltage:.4f} V")
                y += 14
            painter.setPen(QPen(QColor(200, 200, 210), 0.5))
            painter.drawLine(ox + 10, y, ox + width - 10, y)
            y += 4
            for src, current in sorted(bc.items()):
                painter.setPen(QColor(180, 80, 50))
                painter.drawText(ox + 10, y, width - 20, 13, Qt.AlignVCenter,
                                f"I({src}) = {_si(current, 'A')}")
                y += 14

        self.viewer.add_overlay('dc_table', draw_table)
        self.log("DC table overlay")
        return self

    def label_voltages(self):
        """Show node voltage labels on the schematic (requires dc() first)."""
        nv = self.results.get('dc', {}).get('node_voltages', {})
        if not nv:
            self.log("Run circ.dc() first")
            return self

        def draw_vlabels(painter, w, h):
            painter.setFont(QFont("Consolas", 8))
            for node, pos in self.viewer.node_positions.items():
                if node in nv:
                    v = nv[node]
                    tx = w/2 + (pos[0] + self.viewer.cam_x) * self.viewer.zoom
                    ty = h/2 + (pos[1] + self.viewer.cam_y - 18) * self.viewer.zoom
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(255, 255, 230, 210))
                    painter.drawRoundedRect(int(tx) - 30, int(ty) - 6, 60, 14, 3, 3)
                    painter.setPen(QColor(40, 80, 150))
                    painter.drawText(int(tx) - 30, int(ty) - 6, 60, 14, Qt.AlignCenter,
                                    f"{v:.3f}V")

        self.viewer.add_overlay('voltage_labels', draw_vlabels)
        return self

    def label_currents(self):
        """Show branch current labels."""
        bc = self.results.get('dc', {}).get('branch_currents', {})
        if not bc:
            self.log("Run circ.dc() first")
            return self

        def draw_ilabels(painter, w, h):
            painter.setFont(QFont("Consolas", 7))
            for comp_name, current in bc.items():
                comp = self.components.get(comp_name)
                if not comp: continue
                tx = w/2 + (comp.x + 40 + self.viewer.cam_x) * self.viewer.zoom
                ty = h/2 + (comp.y + 55 + self.viewer.cam_y) * self.viewer.zoom
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 240, 240, 210))
                painter.drawRoundedRect(int(tx) - 35, int(ty) - 6, 70, 14, 3, 3)
                painter.setPen(QColor(180, 60, 40))
                painter.drawText(int(tx) - 35, int(ty) - 6, 70, 14, Qt.AlignCenter,
                                f"I={_si(current, 'A')}")

        self.viewer.add_overlay('current_labels', draw_ilabels)
        return self

    def remove_overlay(self, name):
        """Remove an overlay by name."""
        self.viewer.remove_overlay(name)
        return self

    # ── Export ─────────────────────────────────────────────────

    def export_spice(self, path="~/circuit.cir"):
        """Export circuit as SPICE netlist."""
        path = os.path.expanduser(path)
        with open(path, 'w') as f:
            f.write(f"* CircuitLab Export\n")
            for name, comp in self.components.items():
                pins = list(comp.pins.values())
                nodes = ' '.join(str(p) if p else 'NC' for p in pins)
                if comp.ctype == 'resistor':
                    f.write(f"{name} {nodes} {comp.value}\n")
                elif comp.ctype == 'capacitor':
                    f.write(f"{name} {nodes} {comp.value}\n")
                elif comp.ctype == 'inductor':
                    f.write(f"{name} {nodes} {comp.value}\n")
                elif comp.ctype == 'vsrc':
                    ac = f" AC {comp.params.get('ac_amp', 0)}" if comp.params.get('ac_amp') else ''
                    f.write(f"{name} {nodes} {comp.value}{ac}\n")
                elif comp.ctype == 'isrc':
                    f.write(f"{name} {nodes} {comp.value}\n")
                elif comp.ctype == 'diode':
                    f.write(f"{name} {nodes} DMOD\n")
                elif comp.ctype in ('npn', 'pnp'):
                    f.write(f"{name} {nodes} QMOD\n")
            f.write(".end\n")
        self.log(f"Exported SPICE: {path}")
        return path


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
"""

def _lbl(text):
    l = QLabel(text.upper())
    l.setStyleSheet("font-size:9px;letter-spacing:1.5px;color:#8a96a8;font-weight:bold;padding:2px 0;background:transparent")
    return l


# ═══════════════════════════════════════════════════════════════
#  APP CLASS — UI ASSEMBLY + SINGLETON WIRING
# ═══════════════════════════════════════════════════════════════

class CircuitLabApp:
    """Encapsulates the entire CircuitLab UI, signal wiring, and singleton."""

    def __init__(self):
        # ── Main widget ──
        self.main_widget = QWidget()
        self.main_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ── Panel ──
        self.panel = QWidget()
        self.panel.setFixedWidth(290)
        self.panel.setStyleSheet(_SS)
        self.panel.setAttribute(Qt.WA_TranslucentBackground, True)
        self.ps = QScrollArea()
        self.ps.setWidgetResizable(True)
        self.ps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ps.setStyleSheet("QScrollArea{border:none;background:transparent}QScrollBar:vertical{width:5px;background:transparent}QScrollBar::handle:vertical{background:#c0c8d4;border-radius:2px;min-height:30px}")
        self.inner = QWidget()
        self.inner.setAttribute(Qt.WA_TranslucentBackground, True)
        self.lay = QVBoxLayout(self.inner)
        self.lay.setSpacing(4)
        self.lay.setContentsMargins(10, 10, 10, 10)

        # Header
        self._build_header()

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(_SS)

        # Build tabs
        self._build_circuit_tab()
        self._build_simulate_tab()
        self._build_netlist_tab()
        self._build_info_tab()

        self.lay.addWidget(self.tabs)
        self.ps.setWidget(self.inner)
        playout = QVBoxLayout(self.panel)
        playout.setContentsMargins(0, 0, 0, 0)
        playout.addWidget(self.ps)

        self.viewer = SchematicViewer()
        self.viewer.setStyleSheet("background:rgba(248,250,253,255)")
        self.main_layout.addWidget(self.panel)
        self.main_layout.addWidget(self.viewer, 1)

        # ── Create singleton & wire signals ──
        self.circ = CircuitLab(self.viewer, self.log_edit)

        # Wire interactive connection callback from viewer → circ.connect
        def _on_viewer_connect(cn_a, pn_a, cn_b, pn_b):
            self.circ.connect(f"{cn_a}.{pn_a}", f"{cn_b}.{pn_b}")
        self.viewer.on_connect = _on_viewer_connect

        # Wrap methods to auto-update UI
        for _method_name in ('add', 'remove', 'clear', 'connect', 'ground', 'node',
                              'load', 'load_netlist', 'dc', 'ac', 'tran', 'dc_sweep'):
            _orig = getattr(self.circ, _method_name)
            def _make_wrap(orig_fn):
                def _wrap(*args, **kwargs):
                    result = orig_fn(*args, **kwargs)
                    self._update_ui()
                    return result
                return _wrap
            setattr(self.circ, _method_name, _make_wrap(_orig))

        # UI signal wiring
        self.preset_combo.currentTextChanged.connect(lambda t: self.circ.load(t))

        self.dc_btn.clicked.connect(self._on_dc)
        self.ac_btn.clicked.connect(self._on_ac)
        self.tran_btn.clicked.connect(self._on_tran)

        self.wave_btn.clicked.connect(lambda: self.circ.overlay_waveform(self.node_combo.currentText()) if self.node_combo.currentText() else None)
        self.bode_btn.clicked.connect(lambda: self.circ.overlay_bode(self.node_combo.currentText()) if self.node_combo.currentText() else None)
        self.dc_tbl_btn.clicked.connect(lambda: self.circ.overlay_dc_table())
        self.clear_ov_btn.clicked.connect(lambda: (self.viewer._overlays.clear(), self.viewer.update()))

        self.grid_cb.toggled.connect(lambda c: (setattr(self.viewer, 'show_grid', c), self.viewer.update()))

        self.add_r_btn.clicked.connect(lambda: self._quick_add('R', 'resistor', 1000))
        self.add_c_btn.clicked.connect(lambda: self._quick_add('C', 'capacitor', 100e-9))
        self.add_l_btn.clicked.connect(lambda: self._quick_add('L', 'inductor', 10e-3))
        self.add_v_btn.clicked.connect(lambda: self._quick_add('V', 'vsrc', 5.0))

        self.load_nl_btn.clicked.connect(self._on_load_netlist)
        self.export_btn.clicked.connect(lambda: (self.circ.export_spice(), self.status_lbl.setText("✓ Saved ~/circuit.cir")))

        # Load default preset
        self.circ.load(self.preset_combo.currentText())

    # ── Header builder ────────────────────────────────────────

    def _build_header(self):
        hdr = QWidget()
        hdr.setAttribute(Qt.WA_TranslucentBackground, True)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 4)
        ic = QLabel("⚡")
        ic.setStyleSheet("font-size:20px;background:rgba(250,240,225,200);border:1px solid #d8d0c0;border-radius:7px;padding:3px 7px")
        nw = QWidget()
        nw.setAttribute(Qt.WA_TranslucentBackground, True)
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(6, 0, 0, 0)
        nl.setSpacing(0)
        _title_lbl = QLabel("CircuitLab")
        _title_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#1a2a40;background:transparent")
        _sub_lbl = QLabel("CIRCUIT WORKBENCH")
        _sub_lbl.setStyleSheet("font-size:7px;letter-spacing:2px;color:#8a96a8;background:transparent")
        nl.addWidget(_title_lbl)
        nl.addWidget(_sub_lbl)
        hl.addWidget(ic)
        hl.addWidget(nw)
        hl.addStretch()
        self.lay.addWidget(hdr)

    # ── Tab builders ──────────────────────────────────────────

    def _build_circuit_tab(self):
        t1 = QWidget()
        t1.setAttribute(Qt.WA_TranslucentBackground, True)
        t1l = QVBoxLayout(t1)
        t1l.setSpacing(5)
        t1l.setContentsMargins(6, 8, 6, 6)
        t1l.addWidget(_lbl("Presets"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(CircuitLab.PRESETS.keys()))
        t1l.addWidget(self.preset_combo)
        t1l.addWidget(_lbl("Components"))
        self.comp_list = QListWidget()
        self.comp_list.setMinimumHeight(120)
        t1l.addWidget(self.comp_list)
        t1l.addWidget(_lbl("Quick Add"))
        qadd_w = QWidget()
        qadd_w.setAttribute(Qt.WA_TranslucentBackground, True)
        qadd_l = QHBoxLayout(qadd_w)
        qadd_l.setContentsMargins(0, 0, 0, 0)
        qadd_l.setSpacing(2)
        self.add_r_btn = QPushButton("+ R")
        self.add_c_btn = QPushButton("+ C")
        self.add_l_btn = QPushButton("+ L")
        self.add_v_btn = QPushButton("+ V")
        qadd_l.addWidget(self.add_r_btn)
        qadd_l.addWidget(self.add_c_btn)
        qadd_l.addWidget(self.add_l_btn)
        qadd_l.addWidget(self.add_v_btn)
        t1l.addWidget(qadd_w)
        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        t1l.addWidget(self.grid_cb)
        t1l.addStretch()
        self.tabs.addTab(t1, "Circuit")

    def _build_simulate_tab(self):
        t2 = QWidget()
        t2.setAttribute(Qt.WA_TranslucentBackground, True)
        t2l = QVBoxLayout(t2)
        t2l.setSpacing(5)
        t2l.setContentsMargins(6, 8, 6, 6)
        t2l.addWidget(_lbl("Analysis"))
        sim_btns_w = QWidget()
        sim_btns_w.setAttribute(Qt.WA_TranslucentBackground, True)
        sim_l = QVBoxLayout(sim_btns_w)
        sim_l.setContentsMargins(0, 0, 0, 0)
        sim_l.setSpacing(3)
        self.dc_btn = QPushButton("DC Operating Point")
        self.ac_btn = QPushButton("AC Sweep (1Hz–1MHz)")
        self.tran_btn = QPushButton("Transient (1ms)")
        sim_l.addWidget(self.dc_btn)
        sim_l.addWidget(self.ac_btn)
        sim_l.addWidget(self.tran_btn)
        t2l.addWidget(sim_btns_w)
        t2l.addWidget(_lbl("Overlay Node"))
        self.node_combo = QComboBox()
        t2l.addWidget(self.node_combo)
        overlay_btns_w = QWidget()
        overlay_btns_w.setAttribute(Qt.WA_TranslucentBackground, True)
        ov_l = QHBoxLayout(overlay_btns_w)
        ov_l.setContentsMargins(0, 0, 0, 0)
        ov_l.setSpacing(2)
        self.wave_btn = QPushButton("Waveform")
        self.bode_btn = QPushButton("Bode")
        self.dc_tbl_btn = QPushButton("DC Table")
        ov_l.addWidget(self.wave_btn)
        ov_l.addWidget(self.bode_btn)
        ov_l.addWidget(self.dc_tbl_btn)
        t2l.addWidget(overlay_btns_w)
        self.clear_ov_btn = QPushButton("Clear Overlays")
        t2l.addWidget(self.clear_ov_btn)
        t2l.addStretch()
        self.tabs.addTab(t2, "Simulate")

    def _build_netlist_tab(self):
        t3 = QWidget()
        t3.setAttribute(Qt.WA_TranslucentBackground, True)
        t3l = QVBoxLayout(t3)
        t3l.setSpacing(5)
        t3l.setContentsMargins(6, 8, 6, 6)
        t3l.addWidget(_lbl("SPICE Netlist"))
        self.netlist_edit = QTextEdit()
        self.netlist_edit.setPlaceholderText("Paste SPICE netlist here...\ne.g.:\nV1 vin 0 5 AC 1\nR1 vin out 1k\nC1 out 0 100n")
        self.netlist_edit.setMinimumHeight(120)
        t3l.addWidget(self.netlist_edit)
        self.load_nl_btn = QPushButton("Load Netlist")
        t3l.addWidget(self.load_nl_btn)
        self.export_btn = QPushButton("Export SPICE .cir")
        t3l.addWidget(self.export_btn)
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#6a7a8a;font-size:10px;background:transparent")
        t3l.addWidget(self.status_lbl)
        t3l.addStretch()
        self.tabs.addTab(t3, "Netlist")

    def _build_info_tab(self):
        t4 = QWidget()
        t4.setAttribute(Qt.WA_TranslucentBackground, True)
        t4l = QVBoxLayout(t4)
        t4l.setSpacing(5)
        t4l.setContentsMargins(6, 8, 6, 6)
        t4l.addWidget(_lbl("Circuit Info"))
        self.info_edit = QTextEdit()
        self.info_edit.setReadOnly(True)
        self.info_edit.setMinimumHeight(100)
        t4l.addWidget(self.info_edit)
        t4l.addWidget(_lbl("Log"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlainText("[CircuitLab] Initialised\n[CircuitLab] Engine: MNA (Modified Nodal Analysis)\n")
        t4l.addWidget(self.log_edit)
        t4l.addStretch()
        self.tabs.addTab(t4, "Info")

    # ── UI update ─────────────────────────────────────────────

    def _update_ui(self):
        """Sync UI with circ state."""
        self.comp_list.clear()
        for name, comp in self.circ.components.items():
            self.comp_list.addItem(comp.describe())
        info_lines = [f"Components: {len(self.circ.components)}"]
        info_lines.append(f"Nodes: {', '.join(sorted(self.circ.nodes)) if self.circ.nodes else 'none'}")
        dc = self.circ.results.get('dc', {})
        if dc.get('node_voltages'):
            info_lines.append("\nDC Operating Point:")
            for n, v in sorted(dc['node_voltages'].items()):
                info_lines.append(f"  V({n}) = {v:.4f} V")
        self.info_edit.setPlainText('\n'.join(info_lines))
        # Update node combo
        self.node_combo.clear()
        nodes = sorted(self.circ.nodes - {'0'}) if self.circ.nodes else []
        self.node_combo.addItems(nodes)

    # ── Signal handlers ───────────────────────────────────────

    def _on_dc(self):
        self.status_lbl.setText("Running DC analysis...")
        self.status_lbl.repaint()
        self.circ.dc()
        self.circ.overlay_dc_table()
        self.circ.label_voltages()
        nv = self.circ.results.get('dc',{}).get('node_voltages',{})
        n_nodes = len([k for k in nv if k != '0'])
        self.status_lbl.setText(f"✓ DC done — {n_nodes} node voltages solved")

    def _get_overlay_node(self):
        """Get node to overlay — use combo selection or pick first non-ground node."""
        node = self.node_combo.currentText()
        if node: return node
        nodes = sorted(self.circ.nodes - {'0'}) if self.circ.nodes else []
        return nodes[0] if nodes else None

    def _on_ac(self):
        self.status_lbl.setText("Running AC sweep...")
        self.status_lbl.repaint()
        self.circ.ac()
        node = self._get_overlay_node()
        if node:
            self.circ.overlay_bode(node)
            self.status_lbl.setText(f"✓ AC done — Bode plot: V({node})")
        else:
            self.status_lbl.setText("✓ AC done — no signal nodes to plot")

    def _on_tran(self):
        self.status_lbl.setText("Running transient sim...")
        self.status_lbl.repaint()
        self.circ.tran()
        node = self._get_overlay_node()
        if node:
            self.circ.overlay_waveform(node)
            self.status_lbl.setText(f"✓ Transient done — waveform: V({node})")
        else:
            self.status_lbl.setText("✓ Transient done — no signal nodes to plot")

    def _quick_add(self, prefix, ctype, default_val):
        # Find next available index that doesn't collide
        existing = set(self.circ.components.keys())
        idx = 1
        while f"{prefix}{idx}" in existing:
            idx += 1
        name = f"{prefix}{idx}"
        self.circ.add(name, ctype, default_val)

    def _on_load_netlist(self):
        text = self.netlist_edit.toPlainText()
        if not text.strip():
            self.status_lbl.setText("⚠ Paste a netlist first")
            return
        self.circ.load_netlist(text)
        self.status_lbl.setText(f"✓ Loaded {len(self.circ.components)} components")


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE APP
# ═══════════════════════════════════════════════════════════════

circuit_app = CircuitLabApp()
circuit_circ = circuit_app.circ
circuit_circuit = circuit_circ  # alias (long form)
circuit_viewer = circuit_app.viewer

# Short aliases for LLM convenience
circ = circuit_circ
circuit = circuit_circuit

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

circuit_app.main_widget.resize(1400, 850)
circuit_proxy = graphics_scene.addWidget(circuit_app.main_widget)
circuit_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)

# Center in current view
_vr = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
circuit_proxy.setPos(_vr.center().x() - circuit_app.main_widget.width() / 2,
             _vr.center().y() - circuit_app.main_widget.height() / 2)

circuit_shadow = QGraphicsDropShadowEffect()
circuit_shadow.setBlurRadius(60)
circuit_shadow.setOffset(45, 45)
circuit_shadow.setColor(QColor(0, 0, 0, 120))
circuit_proxy.setGraphicsEffect(circuit_shadow)
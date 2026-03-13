# Rio — Apps

A collection of GPU-accelerated, LLM-driven apps built on **PySide6 + ModernGL**. Each app is a self-contained module that registers into the Rio canvas as a draggable widget. The suite spans scientific workbenches, a full map engine, a spreadsheet, a music generator, and a driving game. An LLM can drive every app programmatically through short code snippets.

---

## Setup

### Install dependencies

```bash
pip install -r requirements_apps.txt
```

### Environment variables

| Variable | Purpose |
|---|---|
| `MAPBOX_TOKEN` | Required by SuperMap. Your Mapbox access token for vector tile rendering. |
| `GOOGLE_API_KEY` | Required by Lyria. Your Google AI API key for the Lyria real-time music model. |

```bash
export MAPBOX_TOKEN="pk.your_token_here"
export GOOGLE_API_KEY="your_key_here"
```

> **Shortcut:** SuperMap also reads `MAPBOX_TOKEN` from a `.env` file in the working directory if you prefer not to export it globally.

---

## Apps — Science & Engineering

### CFDLab — `cfdlab.py`

**Computational Fluid Dynamics workbench.** Panel-method aerodynamics solver with real-time airfoil analysis.

- **Geometry** — Load NACA 4/5-digit airfoils, cylinders, flat plates, ellipses, or arbitrary closed curves. Refine panelling, scale, rotate, flip, and translate geometry on the fly.
- **Solver** — Source panel (non-lifting) and vortex panel (lifting) methods. Computes pressure coefficient (Cp), lift (Cl), drag (Cd), pitching moment (Cm), circulation, and stagnation point.
- **Multi-element** — Add slotted flaps and leading-edge slats with configurable deflection and gap.
- **Visualization** — Cp distribution plots, streamline fields, velocity vectors, force arrows, Cl-vs-α polar sweeps, and pressure/velocity colormapped meshes.
- **Parametric** — Sweep angle of attack or compare multiple airfoil profiles in a single overlay.
- **Export** — Selig `.dat` format, CSV results, and PNG screenshots.

**Singleton:** `cfd` &nbsp;|&nbsp; **Viewer:** `cfd_viewer`

---

### ChemLab — `chemlab.py`

**Molecular chemistry workbench with built-in quantum chemistry (PySCF).**

- **Input** — Load molecules by name, XYZ string, SMILES, or `.cube` orbital files. Preset library included.
- **Quantum chemistry** — Run HF, B3LYP, or other DFT methods with configurable basis sets. Compute energies, dipole moments, Mulliken charges, HOMO/LUMO gaps, and vibrational frequencies.
- **Volumetric visualization** — Isosurfaces for HOMO/LUMO/arbitrary MOs, electrostatic potential (ESP) mapped onto electron density, electron localization function (ELF), spin density, density differences, and animated cube sequences.
- **Localized orbitals** — Boys-localized molecular orbitals showing bonds and lone pairs.
- **Analysis** — Bond distances, angles, IR spectra overlay, and 2D cross-section slices through volumetric data.
- **ORCA integration** — Export input files, run ORCA jobs asynchronously, and parse output.
- **Rendering** — Ball-and-stick, spacefill, and wireframe styles with a full 3D OpenGL viewport.

**Singleton:** `chem` / `mol` &nbsp;|&nbsp; **Viewer:** `viewer`

---

### CircuitLab — `circuit_lab.py`

**Electrical engineering workbench with SPICE-class circuit simulation.**

- **Components** — Resistors, capacitors, inductors, DC/AC voltage and current sources, diodes, NPN transistors, and ideal op-amps. Connect pins, assign grounds, and build netlists interactively.
- **Simulation** — DC operating point, AC frequency sweep, transient analysis, and DC source sweep. All results are stored as numpy arrays for further manipulation.
- **Visualization** — Waveform overlays, Bode plots (magnitude + phase), DC operating point tables, and automatic node-voltage / branch-current labels on the schematic.
- **Presets** — RC low-pass, RL high-pass, RLC bandpass, voltage divider, inverting/non-inverting op-amp, and more.
- **Netlist I/O** — Export and import SPICE-format netlists.

**Singleton:** `circ` / `circuit` &nbsp;|&nbsp; **Viewer:** `circuit_viewer`

---

### DrugLab — `druglab.py`

**Drug discovery and molecular biology workbench.**

- **Input** — Load drugs by name (aspirin, caffeine, etc.), SMILES, PDB ID, SDF/MOL file, or raw XYZ coordinates.
- **Drug properties** — Lipinski Rule-of-Five, molecular descriptors (MW, LogP, HBD, HBA, TPSA, rotatable bonds), ADMET predictions, toxicity flags, and composite druglikeness scores.
- **Protein** — Fetch proteins from PDB, visualize binding pockets, compute simplified docking scores, and overlay protein–ligand contacts and Ramachandran plots.
- **Pharmacophore** — Color-coded pharmacophore feature display, H-bond donor/acceptor highlighting, and molecular surfaces colored by hydrophobicity or other properties.
- **Comparison** — Side-by-side property comparison of multiple molecules, descriptor radar charts, and atom-wise LogP contribution maps.

**Singleton:** `drug` &nbsp;|&nbsp; **Viewer:** `drug_viewer`

---

### GenomeLab — `genom_lab.py`

**Genomics and bioinformatics workbench.**

- **Input** — Load VCF (variant call format), FASTA references, BED interval files, and GFF3 gene annotations. Built-in demo datasets (BRCA, etc.).
- **Navigation** — Jump to genomic coordinates, switch chromosomes, zoom in/out, and display full ideogram karyotypes.
- **Variant analysis** — Filter variants by chromosome, quality, gene, or predicted impact. Functional annotation, Manhattan plots, allele frequency histograms, and quality score distributions.
- **Sequence tools** — Display nucleotide sequences, compute GC content, find restriction enzyme motifs, and perform six-frame translation.
- **Gene models** — Overlay gene models in the current view, look up gene details, and show read-depth coverage plots.
- **Export** — Filtered BED, FASTA subsequences, filtered VCF, and genome-wide statistics.

**Singleton:** `genome` / `gen` &nbsp;|&nbsp; **Viewer:** `viewer`

---

### MaterialLab — `material_lab.py`

**Materials science and solid-state physics workbench.**

- **Input** — Load preset crystal structures (silicon, NaCl, diamond, perovskites, etc.), CIF files, or VASP POSCAR files.
- **Crystal tools** — Build supercells, cleave Miller-index surfaces, introduce point defects (vacancies, substitutions), and measure interatomic distances.
- **Analysis** — Quick energy calculations via Lennard-Jones/Morse or EMT potentials. Compute and overlay radial distribution functions g(r) and simulated powder XRD patterns. Overlay density of states and band structures from external data.
- **Properties** — Total energy, energy per atom, cell volume, density, space group detection (via spglib if available), and lattice parameters.
- **Rendering** — Ball-and-stick, spacefill, and polyhedra styles with unit cell wireframes and lattice vector arrows.
- **Export** — VASP POSCAR, CIF, XYZ, and Quantum ESPRESSO input files.

**Singleton:** `matlab_app.matlab_crystal` &nbsp;|&nbsp; **Viewer:** `matlab_app.matlab_viewer`

---

### NeuroLab — `neurolab.py`

**Brain imaging workbench for neuroimaging and EEG/MEG.**

- **Input** — Load MNI152 template, FreeSurfer average brain, arbitrary NIfTI volumes, or FreeSurfer surface meshes.
- **Slicing** — Axial, sagittal, and coronal slices at arbitrary positions, or all three planes simultaneously.
- **Overlays** — Statistical maps (z-stat NIfTI), atlas parcellations (Harvard-Oxford, AAL), and named ROI highlighting.
- **Analysis** — ROI time-series extraction, spatial smoothing, resampling, tissue segmentation (GM/WM/CSF), first-level GLM, seed-based connectivity, and atlas-based parcellation.
- **EEG/MEG** — Load `.fif` files, plot ERP waveforms, and display scalp topography maps (requires MNE-Python).
- **Colormaps** — Hot, cool, gray, viridis, RdBu, and custom thresholding.

**Singleton:** `brain` / `nlab` &nbsp;|&nbsp; **Viewer:** `viewer`

---

### QuantView — `quantview.py`

**Financial terminal with 3D volatility surface rendering.**

- **Market data** — Simulated OHLCV data for major tickers (AAPL, TSLA, NVDA, JPM, AMZN, MSFT, SPY) with configurable volatility, drift, and seed.
- **Technical indicators** — SMA, EMA, and Bollinger Bands computed on the fly.
- **Options** — Full Black-Scholes pricing engine with Greeks (delta, gamma, theta, vega, rho). Interactive 3D implied volatility surface with smile/skew modeling.
- **Portfolio** — Sample portfolio tracker with position-level P&L, weights, and aggregate statistics.
- **Rendering** — OpenGL 3D volatility surface with lighting, fog, and grid overlays. 2D candlestick charts with volume bars and indicator overlays via QPainter.

---

## Apps — Productivity & Creative

### SuperMap — `supermap.py`

**High-performance vector tile map engine powered by Mapbox + ModernGL.** A full-featured GIS viewer with immersive first-person and car-driving modes.

Requires `MAPBOX_TOKEN` in your environment.

- **Rendering** — GPU-tessellated Mapbox vector tiles with geometry VBO caching, parent-zoom fallback with GL scissor clipping, frustum culling, and hybrid LOD. Time-sliced GPU uploads keep per-frame work under ~4 ms for stutter-free streaming.
- **3D** — Extruded buildings, tilt/bearing rotation, terrain DEM with a slot-based texture atlas, and building shadows.
- **Styles** — Cycle through map styles with `S`. Toggle buildings (`B`), labels (`F`), POI markers (`O`), and tile visibility (`V`).
- **Routing** — Toggle route mode (`N`), click to place waypoints, clear with `C`, and cycle profiles (driving/walking/cycling) with `M`.
- **Heatmaps & Isochrones** — Cycle heatmap layers with `H` (POI → traffic → building density → off). Cycle isochrone overlays with `Y` (driving → walking → cycling → off).
- **Search** — Press `/` to open the search bar, type a destination, and press Enter to navigate.
- **Immersive mode** — Press `I` then click to drop into first-person. Walk with `WASD`, look with the mouse, and scroll to adjust speed. Browse nearby POIs with `Up`/`Down`, auto-route to one with `Enter`, open in Google Maps (`E`), web-search (`X`), or Wikipedia (`J`).
- **Car mode** — While in immersive mode, press `G` to spawn a car. `W`/`S` gas/brake, `A`/`D` steer, `Space` handbrake. Toggle NPC traffic with `P`. Physics runs on a dedicated 120 Hz thread for smooth driving even during tile loads.

**Keyboard shortcuts at a glance:**

| Key | Action | Key | Action |
|---|---|---|---|
| Left-drag | Pan | `S` | Cycle styles |
| Right-drag | Tilt + rotate | `B` | Toggle 3D buildings |
| Scroll | Zoom | `T` | Reset tilt |
| `+` / `-` | Zoom in/out | `F` | Toggle labels |
| `R` | Reset view | `O` | Toggle POI markers |
| `N` | Route mode | `H` | Cycle heatmaps |
| `C` | Clear route | `Y` | Cycle isochrones |
| `M` | Cycle route profile | `L` | Toggle shadows |
| `/` | Search bar | `V` | Toggle tile visibility |
| `I` | Immersive mode | `G` | Car mode (in immersive) |
| `P` | Toggle NPC traffic | `Esc` | Exit immersive |

---

### Gridion — `supersheet.py`

**Advanced spreadsheet engine with a polished light-mode UI.**

- **Formula engine** — Supports cell references (absolute and relative), ranges, and a rich function library: `SUM`, `AVERAGE`, `MIN`, `MAX`, `COUNT`, `COUNTA`, `IF`, `AND`, `OR`, `NOT`, `CONCATENATE`, `LEN`, `LEFT`, `RIGHT`, `MID`, `UPPER`, `LOWER`, `TRIM`, `ROUND`, `ABS`, `POWER`, `SQRT`, `MOD`, `INT`, `VLOOKUP`, `HLOOKUP`, `INDEX`, `MATCH`, `SUMIF`, `COUNTIF`, `AVERAGEIF`, `NOW`, `TODAY`, `DATE`, `YEAR`, `MONTH`, `DAY`, `ISNUMBER`, `ISTEXT`, `ISBLANK`, `IFERROR`, `TEXT`, `VALUE`, `PI`, `RAND`, and more. Nested formulas and circular reference detection are built-in.
- **Editing** — Undo/redo, find & replace, freeze panes, auto-fill, drag-select, and cell styling (fonts, colors, borders, alignment).
- **Conditional formatting** — Rule-based cell highlighting with color scales.
- **Charts** — Built-in charting from selected data ranges.
- **Multi-sheet** — Tabbed sheets with add/remove/rename.
- **Import / Export** — CSV and JSON.

---

### Lyria — `lyria.py`

**Real-time AI music generator powered by Google's Lyria model.**

Requires `GOOGLE_API_KEY` in your environment.

- **Live streaming** — Connects to Google's `lyria-realtime-exp` model via the GenAI live music API. Audio streams in real-time through PyAudio at 48 kHz stereo.
- **Controls** — Rotary knobs for BPM (60–200) and Top-K sampling. Play, pause, and stop buttons with a live connection status display.
- **Prompts** — Add multiple weighted text prompts (e.g., "Piano", "Ambient Synth", "Jazz Drums") with per-prompt weight sliders. Prompts can be added, removed, and reweighted while music is playing.
- **Scales** — Choose from a full set of musical scales (major, minor, and all modal variants across all keys).
- **Mixer UI** — DAW-style vertical fader strips that animate in as prompts are added. Each fader controls the weight of one prompt in the mix.

---

### Car Game — `car_game_gl.py`

**3D driving game set on the Golden Gate Bridge.** Pure Python / ModernGL / offscreen FBO — no browser or Three.js.

- **Driving physics** — Acceleration, braking, steering with speed-dependent turn radius, handbrake drifting, and nitro boost. Collision detection against NPC traffic and guard rails.
- **World** — Procedurally built San Francisco city streets leading onto the Golden Gate Bridge and into Marin County. Includes buildings, trees, streetlights, lane markings, bridge towers and cables, and a water plane.
- **NPC traffic** — Multiple AI-controlled vehicles in both lanes with braking behavior and headlights/taillights.
- **Camera** — Three camera modes (chase, cockpit, cinematic) cycled with `C`.
- **HUD** — Speedometer, nitro gauge, zone labels, minimap with NPC dots, drift indicator, and a controls legend.
- **Rendering** — Lit + unlit shaders, fog, headlight beam projection, and a procedural gradient sky.

**Controls:**

| Key | Action |
|---|---|
| `W` / `↑` | Gas |
| `S` / `↓` | Brake / Reverse |
| `A` / `←` | Steer left |
| `D` / `→` | Steer right |
| `Space` | Handbrake (hold + steer for drift) |
| `Shift` | Nitro boost |
| `C` | Cycle camera mode |

---
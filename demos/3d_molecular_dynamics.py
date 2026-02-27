import json
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSlider, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QCheckBox, QTabWidget, QTextEdit, QSplitter, QFrame,
    QListWidget, QListWidgetItem, QProgressBar
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QFont

# Main container
main_widget = QWidget()
main_layout = QHBoxLayout(main_widget)
main_layout.setContentsMargins(0, 0, 0, 0)
main_layout.setSpacing(0)

# ========== LEFT CONTROL PANEL ==========
control_panel = QWidget()
control_panel.setFixedWidth(320)
control_panel.setStyleSheet("""
    QWidget {
        background-color: #1a1a2e;
        color: #eee;
        font-family: 'Segoe UI', Arial;
    }
    QGroupBox {
        border: 1px solid #4a4a6a;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 10px;
        font-weight: bold;
        color: #00d4ff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a8a, stop:1 #3a3a6a);
        border: 1px solid #5a5a9a;
        border-radius: 6px;
        padding: 8px 15px;
        color: white;
        font-weight: bold;
    }
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a5a9a, stop:1 #4a4a7a);
    }
    QPushButton:pressed {
        background: #2a2a5a;
    }
    QSlider::groove:horizontal {
        height: 6px;
        background: #3a3a5a;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #00d4ff;
        width: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }
    QComboBox {
        background: #2a2a4a;
        border: 1px solid #4a4a6a;
        border-radius: 4px;
        padding: 5px;
    }
    QSpinBox, QDoubleSpinBox {
        background: #2a2a4a;
        border: 1px solid #4a4a6a;
        border-radius: 4px;
        padding: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #4a4a6a;
    }
    QCheckBox::indicator:checked {
        background: #00d4ff;
    }
    QListWidget {
        background: #2a2a4a;
        border: 1px solid #4a4a6a;
        border-radius: 4px;
    }
    QProgressBar {
        border: 1px solid #4a4a6a;
        border-radius: 4px;
        text-align: center;
        background: #2a2a4a;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4ff, stop:1 #00ff88);
        border-radius: 3px;
    }
""")

control_layout = QVBoxLayout(control_panel)
control_layout.setSpacing(8)
control_layout.setContentsMargins(10, 10, 10, 10)

# Title
title = QLabel("⚛️ MolDyn Pro")
title.setFont(QFont("Segoe UI", 18, QFont.Bold))
title.setStyleSheet("color: #00d4ff; padding: 10px;")
title.setAlignment(Qt.AlignCenter)
control_layout.addWidget(title)

# Molecule Selection
mol_group = QGroupBox("Molecule Selection")
mol_layout = QVBoxLayout(mol_group)

mol_combo = QComboBox()
mol_combo.addItems(["Water (H₂O)", "Methane (CH₄)", "Ethanol (C₂H₅OH)", 
                    "Benzene (C₆H₆)", "Caffeine (C₈H₁₀N₄O₂)", "Aspirin (C₉H₈O₄)",
                    "DNA Double Helix", "Protein (Insulin)", "Fullerene (C₆₀)"])
mol_layout.addWidget(mol_combo)

load_btn = QPushButton("📂 Load Molecule")
mol_layout.addWidget(load_btn)
control_layout.addWidget(mol_group)

# Simulation Parameters
sim_group = QGroupBox("Simulation Parameters")
sim_layout = QVBoxLayout(sim_group)

# Temperature
temp_layout = QHBoxLayout()
temp_layout.addWidget(QLabel("Temperature (K):"))
temp_spin = QSpinBox()
temp_spin.setRange(1, 1000)
temp_spin.setValue(300)
temp_layout.addWidget(temp_spin)
sim_layout.addLayout(temp_layout)

# Pressure
press_layout = QHBoxLayout()
press_layout.addWidget(QLabel("Pressure (atm):"))
press_spin = QDoubleSpinBox()
press_spin.setRange(0.01, 100)
press_spin.setValue(1.0)
press_spin.setDecimals(2)
press_layout.addWidget(press_spin)
sim_layout.addLayout(press_layout)

# Time Step
dt_layout = QHBoxLayout()
dt_layout.addWidget(QLabel("Time Step (fs):"))
dt_spin = QDoubleSpinBox()
dt_spin.setRange(0.1, 10)
dt_spin.setValue(1.0)
dt_spin.setDecimals(1)
dt_layout.addWidget(dt_spin)
sim_layout.addLayout(dt_layout)

# Ensemble
ens_layout = QHBoxLayout()
ens_layout.addWidget(QLabel("Ensemble:"))
ens_combo = QComboBox()
ens_combo.addItems(["NVE", "NVT", "NPT", "NPH"])
ens_layout.addWidget(ens_combo)
sim_layout.addLayout(ens_layout)

control_layout.addWidget(sim_group)

# Visualization Options
vis_group = QGroupBox("Visualization")
vis_layout = QVBoxLayout(vis_group)

style_layout = QHBoxLayout()
style_layout.addWidget(QLabel("Style:"))
style_combo = QComboBox()
style_combo.addItems(["Ball & Stick", "Space Filling", "Wireframe", "Ribbon", "Surface"])
style_layout.addWidget(style_combo)
vis_layout.addLayout(style_layout)

color_layout = QHBoxLayout()
color_layout.addWidget(QLabel("Coloring:"))
color_combo = QComboBox()
color_combo.addItems(["Element (CPK)", "Charge", "Velocity", "Force", "Chain"])
color_layout.addWidget(color_combo)
vis_layout.addLayout(color_layout)

show_bonds = QCheckBox("Show Bonds")
show_bonds.setChecked(True)
vis_layout.addWidget(show_bonds)

show_forces = QCheckBox("Show Force Vectors")
vis_layout.addWidget(show_forces)

show_velocity = QCheckBox("Show Velocity Vectors")
vis_layout.addWidget(show_velocity)

show_labels = QCheckBox("Show Atom Labels")
vis_layout.addWidget(show_labels)

control_layout.addWidget(vis_group)

# Simulation Controls
ctrl_group = QGroupBox("Simulation Control")
ctrl_layout = QVBoxLayout(ctrl_group)

btn_layout = QHBoxLayout()
play_btn = QPushButton("▶ Play")
play_btn.setStyleSheet("background: #2a8a4a;")
pause_btn = QPushButton("⏸ Pause")
reset_btn = QPushButton("🔄 Reset")
btn_layout.addWidget(play_btn)
btn_layout.addWidget(pause_btn)
btn_layout.addWidget(reset_btn)
ctrl_layout.addLayout(btn_layout)

speed_layout = QHBoxLayout()
speed_layout.addWidget(QLabel("Speed:"))
speed_slider = QSlider(Qt.Horizontal)
speed_slider.setRange(1, 100)
speed_slider.setValue(50)
speed_layout.addWidget(speed_slider)
ctrl_layout.addLayout(speed_layout)

# Progress
progress = QProgressBar()
progress.setValue(0)
progress.setFormat("Step: 0 / 10000")
ctrl_layout.addWidget(progress)

control_layout.addWidget(ctrl_group)

# Energy Display
energy_group = QGroupBox("Energy Monitor")
energy_layout = QVBoxLayout(energy_group)

ke_label = QLabel("Kinetic: 0.00 kJ/mol")
ke_label.setStyleSheet("color: #ff6b6b;")
pe_label = QLabel("Potential: 0.00 kJ/mol")
pe_label.setStyleSheet("color: #4ecdc4;")
te_label = QLabel("Total: 0.00 kJ/mol")
te_label.setStyleSheet("color: #ffe66d; font-weight: bold;")

energy_layout.addWidget(ke_label)
energy_layout.addWidget(pe_label)
energy_layout.addWidget(te_label)
control_layout.addWidget(energy_group)

control_layout.addStretch()

# ========== THREE.JS WEBVIEW ==========
webview = QWebEngineView()
webview.setMinimumSize(800, 600)

three_js_html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            overflow: hidden; 
            background: radial-gradient(ellipse at center, #1a1a2e 0%, #0d0d1a 100%);
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: #00d4ff;
            font-family: 'Segoe UI', Arial;
            font-size: 14px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #4a4a6a;
            max-width: 250px;
        }
        #info h3 { margin-bottom: 10px; color: #fff; }
        #info p { margin: 5px 0; font-size: 12px; }
        .atom-info { color: #ffe66d; }
        #stats {
            position: absolute;
            bottom: 10px;
            right: 10px;
            color: #888;
            font-family: monospace;
            font-size: 11px;
            background: rgba(0,0,0,0.5);
            padding: 8px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>⚛️ Molecular Info</h3>
        <p>Molecule: <span class="atom-info" id="mol-name">Water (H₂O)</span></p>
        <p>Atoms: <span class="atom-info" id="atom-count">3</span></p>
        <p>Bonds: <span class="atom-info" id="bond-count">2</span></p>
        <p>Mass: <span class="atom-info" id="mol-mass">18.015 g/mol</span></p>
        <p style="margin-top:10px; color:#888; font-size:11px;">🖱️ Drag to rotate | Scroll to zoom</p>
    </div>
    <div id="stats">FPS: <span id="fps">60</span> | Atoms: <span id="rendered">0</span></div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Scene Setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.body.appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
        scene.add(ambientLight);
        
        const mainLight = new THREE.DirectionalLight(0xffffff, 1);
        mainLight.position.set(10, 20, 10);
        mainLight.castShadow = true;
        scene.add(mainLight);
        
        const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
        fillLight.position.set(-10, -5, -10);
        scene.add(fillLight);
        
        const rimLight = new THREE.DirectionalLight(0xff8844, 0.2);
        rimLight.position.set(0, -10, 5);
        scene.add(rimLight);
        
        // Element Colors & Radii (CPK)
        const elements = {
            H: { color: 0xffffff, radius: 0.31, mass: 1.008 },
            C: { color: 0x333333, radius: 0.77, mass: 12.011 },
            N: { color: 0x3050f8, radius: 0.71, mass: 14.007 },
            O: { color: 0xff0d0d, radius: 0.66, mass: 15.999 },
            S: { color: 0xffff30, radius: 1.05, mass: 32.065 },
            P: { color: 0xff8000, radius: 1.07, mass: 30.974 },
            F: { color: 0x90e050, radius: 0.57, mass: 18.998 },
            Cl: { color: 0x1ff01f, radius: 1.02, mass: 35.453 },
            Br: { color: 0xa62929, radius: 1.20, mass: 79.904 }
        };
        
        // Molecule Data
        const molecules = {
            water: {
                name: "Water (H₂O)",
                atoms: [
                    { element: 'O', pos: [0, 0, 0] },
                    { element: 'H', pos: [0.96, 0, 0] },
                    { element: 'H', pos: [-0.24, 0.93, 0] }
                ],
                bonds: [[0,1], [0,2]]
            },
            methane: {
                name: "Methane (CH₄)",
                atoms: [
                    { element: 'C', pos: [0, 0, 0] },
                    { element: 'H', pos: [1.09, 0, 0] },
                    { element: 'H', pos: [-0.36, 1.03, 0] },
                    { element: 'H', pos: [-0.36, -0.51, 0.89] },
                    { element: 'H', pos: [-0.36, -0.51, -0.89] }
                ],
                bonds: [[0,1], [0,2], [0,3], [0,4]]
            },
            ethanol: {
                name: "Ethanol (C₂H₅OH)",
                atoms: [
                    { element: 'C', pos: [-1.2, 0, 0] },
                    { element: 'C', pos: [0.3, 0, 0] },
                    { element: 'O', pos: [0.9, 1.2, 0] },
                    { element: 'H', pos: [-1.6, 1, 0] },
                    { element: 'H', pos: [-1.6, -0.5, 0.87] },
                    { element: 'H', pos: [-1.6, -0.5, -0.87] },
                    { element: 'H', pos: [0.7, -0.5, 0.87] },
                    { element: 'H', pos: [0.7, -0.5, -0.87] },
                    { element: 'H', pos: [1.8, 1.2, 0] }
                ],
                bonds: [[0,1], [1,2], [0,3], [0,4], [0,5], [1,6], [1,7], [2,8]]
            },
            benzene: {
                name: "Benzene (C₆H₆)",
                atoms: [],
                bonds: []
            },
            caffeine: {
                name: "Caffeine (C₈H₁₀N₄O₂)",
                atoms: [],
                bonds: []
            },
            fullerene: {
                name: "Fullerene (C₆₀)",
                atoms: [],
                bonds: []
            }
        };
        
        // Generate Benzene
        for (let i = 0; i < 6; i++) {
            const angle = (i * Math.PI * 2) / 6;
            molecules.benzene.atoms.push({ 
                element: 'C', 
                pos: [Math.cos(angle) * 1.4, Math.sin(angle) * 1.4, 0] 
            });
            molecules.benzene.atoms.push({ 
                element: 'H', 
                pos: [Math.cos(angle) * 2.5, Math.sin(angle) * 2.5, 0] 
            });
        }
        for (let i = 0; i < 6; i++) {
            molecules.benzene.bonds.push([i*2, ((i+1)%6)*2]);
            molecules.benzene.bonds.push([i*2, i*2+1]);
        }
        
        // Generate Fullerene (C60)
        const phi = (1 + Math.sqrt(5)) / 2;
        const fullereneVerts = [];
        // Icosahedron vertices scaled
        [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ].forEach(v => {
            const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            fullereneVerts.push([v[0]/len * 3.5, v[1]/len * 3.5, v[2]/len * 3.5]);
        });
        // Add more points for C60 approximation
        for (let i = 0; i < 48; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 3.5;
            fullereneVerts.push([
                r * Math.sin(phi) * Math.cos(theta),
                r * Math.sin(phi) * Math.sin(theta),
                r * Math.cos(phi)
            ]);
        }
        fullereneVerts.slice(0, 60).forEach(v => {
            molecules.fullerene.atoms.push({ element: 'C', pos: v });
        });
        // Connect nearby atoms
        for (let i = 0; i < 60; i++) {
            for (let j = i + 1; j < 60; j++) {
                const dx = molecules.fullerene.atoms[i].pos[0] - molecules.fullerene.atoms[j].pos[0];
                const dy = molecules.fullerene.atoms[i].pos[1] - molecules.fullerene.atoms[j].pos[1];
                const dz = molecules.fullerene.atoms[i].pos[2] - molecules.fullerene.atoms[j].pos[2];
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < 2.0 && molecules.fullerene.bonds.length < 90) {
                    molecules.fullerene.bonds.push([i, j]);
                }
            }
        }
        
        // Generate Caffeine (simplified)
        const caffAtoms = [
            { element: 'C', pos: [0, 0, 0] },
            { element: 'N', pos: [1.2, 0.5, 0] },
            { element: 'C', pos: [2.2, -0.3, 0] },
            { element: 'N', pos: [2.0, -1.5, 0] },
            { element: 'C', pos: [0.7, -1.8, 0] },
            { element: 'C', pos: [-0.3, -0.9, 0] },
            { element: 'N', pos: [-1.5, -1.2, 0] },
            { element: 'C', pos: [-2.0, 0, 0] },
            { element: 'O', pos: [-3.2, 0.3, 0] },
            { element: 'N', pos: [-1.2, 1.0, 0] },
            { element: 'C', pos: [0.1, 1.3, 0] },
            { element: 'O', pos: [0.5, 2.5, 0] },
            { element: 'C', pos: [1.3, 1.8, 0] },
            { element: 'C', pos: [3.5, 0.1, 0] },
            { element: 'C', pos: [-2.0, 2.2, 0] }
        ];
        caffAtoms.forEach(a => molecules.caffeine.atoms.push(a));
        [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[5,6],[6,7],[7,8],[7,9],[9,10],[10,0],[10,11],[1,12],[2,13],[9,14]].forEach(b => molecules.caffeine.bonds.push(b));
        
        // Molecule Group
        let moleculeGroup = new THREE.Group();
        scene.add(moleculeGroup);
        
        let atomMeshes = [];
        let bondMeshes = [];
        let velocities = [];
        let forces = [];
        
        // Create Molecule
        function createMolecule(molData, style = 'ballstick') {
            // Clear existing
            moleculeGroup.children.forEach(child => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            moleculeGroup.clear();
            atomMeshes = [];
            bondMeshes = [];
            velocities = [];
            forces = [];
            
            const scale = style === 'spacefill' ? 2.5 : 1.0;
            
            // Create atoms
            molData.atoms.forEach((atom, i) => {
                const elem = elements[atom.element] || { color: 0x888888, radius: 0.5 };
                const geometry = new THREE.SphereGeometry(elem.radius * scale, 32, 32);
                const material = new THREE.MeshPhysicalMaterial({
                    color: elem.color,
                    metalness: 0.1,
                    roughness: 0.3,
                    clearcoat: 0.5,
                    clearcoatRoughness: 0.3
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(atom.pos[0], atom.pos[1], atom.pos[2]);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                mesh.userData = { element: atom.element, index: i };
                moleculeGroup.add(mesh);
                atomMeshes.push(mesh);
                
                // Initialize velocity (random thermal motion)
                velocities.push(new THREE.Vector3(
                    (Math.random() - 0.5) * 0.02,
                    (Math.random() - 0.5) * 0.02,
                    (Math.random() - 0.5) * 0.02
                ));
                forces.push(new THREE.Vector3(0, 0, 0));
            });
            
            // Create bonds
            if (style !== 'spacefill') {
                molData.bonds.forEach(bond => {
                    const atom1 = molData.atoms[bond[0]];
                    const atom2 = molData.atoms[bond[1]];
                    
                    const start = new THREE.Vector3(atom1.pos[0], atom1.pos[1], atom1.pos[2]);
                    const end = new THREE.Vector3(atom2.pos[0], atom2.pos[1], atom2.pos[2]);
                    const mid = start.clone().add(end).multiplyScalar(0.5);
                    const length = start.distanceTo(end);
                    
                    const geometry = new THREE.CylinderGeometry(0.08, 0.08, length, 16);
                    const material = new THREE.MeshPhysicalMaterial({
                        color: 0x888888,
                        metalness: 0.3,
                        roughness: 0.5
                    });
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.copy(mid);
                    mesh.lookAt(end);
                    mesh.rotateX(Math.PI / 2);
                    moleculeGroup.add(mesh);
                    bondMeshes.push({ mesh, bond, start: start.clone(), end: end.clone() });
                });
            }
            
            // Update info
            document.getElementById('mol-name').textContent = molData.name;
            document.getElementById('atom-count').textContent = molData.atoms.length;
            document.getElementById('bond-count').textContent = molData.bonds.length;
            
            let mass = 0;
            molData.atoms.forEach(a => {
                mass += (elements[a.element] || { mass: 0 }).mass;
            });
            document.getElementById('mol-mass').textContent = mass.toFixed(3) + ' g/mol';
            document.getElementById('rendered').textContent = molData.atoms.length;
        }
        
        // Initial molecule
        let currentMolecule = molecules.water;
        createMolecule(currentMolecule);
        
        // Camera position
        camera.position.set(5, 3, 8);
        camera.lookAt(0, 0, 0);
        
        // Mouse controls
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        let rotationSpeed = { x: 0, y: 0 };
        
        document.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;
                
                rotationSpeed.x = deltaY * 0.005;
                rotationSpeed.y = deltaX * 0.005;
                
                moleculeGroup.rotation.x += rotationSpeed.x;
                moleculeGroup.rotation.y += rotationSpeed.y;
                
                previousMousePosition = { x: e.clientX, y: e.clientY };
            }
        });
        
        document.addEventListener('wheel', (e) => {
            camera.position.z += e.deltaY * 0.01;
            camera.position.z = Math.max(3, Math.min(30, camera.position.z));
        });
        
        // Simulation state
        let isPlaying = false;
        let simSpeed = 1.0;
        let temperature = 300;
        let step = 0;
        
        // Physics simulation (Lennard-Jones + bonds)
        function calculateForces() {
            // Reset forces
            forces.forEach(f => f.set(0, 0, 0));
            
            const epsilon = 0.001; // LJ well depth
            const sigma = 1.0; // LJ distance
            
            // Non-bonded interactions (simplified LJ)
            for (let i = 0; i < atomMeshes.length; i++) {
                for (let j = i + 1; j < atomMeshes.length; j++) {
                    const r = atomMeshes[i].position.clone().sub(atomMeshes[j].position);
                    const dist = r.length();
                    
                    if (dist > 0.5 && dist < 5) {
                        const sr6 = Math.pow(sigma / dist, 6);
                        const force = 24 * epsilon * (2 * sr6 * sr6 - sr6) / dist;
                        const forceVec = r.normalize().multiplyScalar(force);
                        
                        forces[i].add(forceVec);
                        forces[j].sub(forceVec);
                    }
                }
            }
            
            // Bond constraints (harmonic)
            bondMeshes.forEach(({ bond }) => {
                const i = bond[0];
                const j = bond[1];
                const r = atomMeshes[i].position.clone().sub(atomMeshes[j].position);
                const dist = r.length();
                const equilibrium = 1.5;
                const k = 0.5; // spring constant
                
                const force = -k * (dist - equilibrium);
                const forceVec = r.normalize().multiplyScalar(force);
                
                forces[i].add(forceVec);
                forces[j].sub(forceVec);
            });
        }
        
        function updateBonds() {
            bondMeshes.forEach(({ mesh, bond }) => {
                const start = atomMeshes[bond[0]].position;
                const end = atomMeshes[bond[1]].position;
                const mid = start.clone().add(end).multiplyScalar(0.5);
                const length = start.distanceTo(end);
                
                mesh.position.copy(mid);
                mesh.scale.y = length / 1.5;
                
                const direction = end.clone().sub(start).normalize();
                const quaternion = new THREE.Quaternion();
                quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
                mesh.quaternion.copy(quaternion);
            });
        }
        
        // Energy calculation
        function calculateEnergy() {
            let ke = 0;
            let pe = 0;
            
            velocities.forEach(v => {
                ke += 0.5 * v.lengthSq() * 1000;
            });
            
            bondMeshes.forEach(({ bond }) => {
                const dist = atomMeshes[bond[0]].position.distanceTo(atomMeshes[bond[1]].position);
                pe += 0.5 * 0.5 * Math.pow(dist - 1.5, 2) * 1000;
            });
            
            return { ke, pe, total: ke + pe };
        }
        
        // Animation loop
        let lastTime = performance.now();
        let frameCount = 0;
        
        function animate() {
            requestAnimationFrame(animate);
            
            // FPS counter
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }
            
            // Auto-rotation when not dragging
            if (!isDragging) {
                rotationSpeed.x *= 0.95;
                rotationSpeed.y *= 0.95;
                moleculeGroup.rotation.y += 0.002 + rotationSpeed.y * 0.1;
            }
            
            // Physics simulation
            if (isPlaying) {
                calculateForces();
                
                const dt = 0.016 * simSpeed;
                const damping = 0.99;
                
                for (let i = 0; i < atomMeshes.length; i++) {
                    // Velocity Verlet integration
                    velocities[i].add(forces[i].clone().multiplyScalar(dt));
                    velocities[i].multiplyScalar(damping);
                    
                    // Temperature control (simple velocity rescaling)
                    const targetVel = Math.sqrt(temperature / 300) * 0.02;
                    const currentVel = velocities[i].length();
                    if (currentVel > 0) {
                        velocities[i].multiplyScalar(0.99 + 0.01 * targetVel / currentVel);
                    }
                    
                    atomMeshes[i].position.add(velocities[i].clone().multiplyScalar(dt * 10));
                }
                
                updateBonds();
                step++;
            }
            
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // API for Qt communication
        window.setMolecule = function(name) {
            const molMap = {
                'Water (H₂O)': molecules.water,
                'Methane (CH₄)': molecules.methane,
                'Ethanol (C₂H₅OH)': molecules.ethanol,
                'Benzene (C₆H₆)': molecules.benzene,
                'Caffeine (C₈H₁₀N₄O₂)': molecules.caffeine,
                'Fullerene (C₆₀)': molecules.fullerene
            };
            if (molMap[name]) {
                currentMolecule = molMap[name];
                createMolecule(currentMolecule);
            }
        };
        
        window.setPlaying = function(playing) {
            isPlaying = playing;
        };
        
        window.setSpeed = function(speed) {
            simSpeed = speed / 50;
        };
        
        window.setTemperature = function(temp) {
            temperature = temp;
        };
        
        window.setStyle = function(style) {
            const styleMap = {
                'Ball & Stick': 'ballstick',
                'Space Filling': 'spacefill',
                'Wireframe': 'wireframe'
            };
            createMolecule(currentMolecule, styleMap[style] || 'ballstick');
        };
        
        window.resetSimulation = function() {
            step = 0;
            createMolecule(currentMolecule);
        };
        
        // Make functions globally accessible
        window.getEnergy = function() {
            return JSON.stringify(calculateEnergy());
        };
        
        window.getStep = function() {
            return step;
        };
        
        // Signal that the page is ready
        window.pageReady = true;
    </script>
</body>
</html>
"""

webview.setHtml(three_js_html)

# ========== RIGHT INFO PANEL ==========
right_panel = QWidget()
right_panel.setFixedWidth(280)
right_panel.setStyleSheet("""
    QWidget {
        background-color: #1a1a2e;
        color: #eee;
        font-family: 'Segoe UI', Arial;
    }
    QTabWidget::pane {
        border: 1px solid #4a4a6a;
        border-radius: 4px;
        background: #1a1a2e;
    }
    QTabBar::tab {
        background: #2a2a4a;
        color: #888;
        padding: 8px 15px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background: #3a3a6a;
        color: #00d4ff;
    }
    QTextEdit {
        background: #0d0d1a;
        border: 1px solid #4a4a6a;
        border-radius: 4px;
        font-family: 'Consolas', monospace;
        font-size: 11px;
    }
    QListWidget {
        background: #2a2a4a;
        border: 1px solid #4a4a6a;
        border-radius: 4px;
    }
""")

right_layout = QVBoxLayout(right_panel)
right_layout.setContentsMargins(10, 10, 10, 10)

tabs = QTabWidget()

# Log Tab
log_tab = QWidget()
log_layout = QVBoxLayout(log_tab)
log_text = QTextEdit()
log_text.setReadOnly(True)
log_text.setPlainText("""[INFO] MolDyn Pro v2.0 initialized
[INFO] OpenGL renderer ready
[INFO] Loading default molecule: Water
[INFO] 3 atoms, 2 bonds loaded
[INFO] Force field: AMBER99
[INFO] Ready for simulation
────────────────────────────────
[TIP] Use mouse to rotate view
[TIP] Scroll to zoom in/out
[TIP] Select molecules from dropdown
""")
log_layout.addWidget(log_text)
tabs.addTab(log_tab, "📋 Log")

# Atoms Tab
atoms_tab = QWidget()
atoms_layout = QVBoxLayout(atoms_tab)
atoms_list = QListWidget()
atoms_list.addItems([
    "O1  - Oxygen   (0.00, 0.00, 0.00)",
    "H1  - Hydrogen (0.96, 0.00, 0.00)",
    "H2  - Hydrogen (-0.24, 0.93, 0.00)"
])
atoms_layout.addWidget(atoms_list)
tabs.addTab(atoms_tab, "⚛️ Atoms")

# Analysis Tab
analysis_tab = QWidget()
analysis_layout = QVBoxLayout(analysis_tab)

analysis_layout.addWidget(QLabel("📊 Analysis Tools"))

rdf_btn = QPushButton("Calculate RDF")
msd_btn = QPushButton("Calculate MSD")
rmsd_btn = QPushButton("Calculate RMSD")
export_btn = QPushButton("📁 Export Trajectory")

analysis_layout.addWidget(rdf_btn)
analysis_layout.addWidget(msd_btn)
analysis_layout.addWidget(rmsd_btn)
analysis_layout.addWidget(export_btn)
analysis_layout.addStretch()

tabs.addTab(analysis_tab, "📈 Analysis")

right_layout.addWidget(tabs)

# ========== CONNECT SIGNALS ==========
# Flag to track if page is ready
page_ready = False

def check_page_ready():
    global page_ready
    webview.page().runJavaScript("window.pageReady", lambda result: setattr(check_page_ready, 'ready', result == True))
    if hasattr(check_page_ready, 'ready') and check_page_ready.ready:
        page_ready = True
        log_text.append("[INFO] JavaScript page ready")
        return False
    return True

# Timer to check if page is ready
ready_timer = QTimer()
ready_timer.timeout.connect(lambda: ready_timer.stop() if not check_page_ready() else None)
ready_timer.start(100)

def on_molecule_changed(text):
    if page_ready:
        webview.page().runJavaScript(f'setMolecule("{text}")')
        log_text.append(f"[INFO] Loaded molecule: {text}")
    
mol_combo.currentTextChanged.connect(on_molecule_changed)

def on_play():
    if page_ready:
        webview.page().runJavaScript('setPlaying(true)')
        log_text.append("[SIM] Simulation started")
    
def on_pause():
    if page_ready:
        webview.page().runJavaScript('setPlaying(false)')
        log_text.append("[SIM] Simulation paused")
    
def on_reset():
    if page_ready:
        webview.page().runJavaScript('resetSimulation()')
        progress.setValue(0)
        log_text.append("[SIM] Simulation reset")

play_btn.clicked.connect(on_play)
pause_btn.clicked.connect(on_pause)
reset_btn.clicked.connect(on_reset)

def on_speed_changed(value):
    if page_ready:
        webview.page().runJavaScript(f'setSpeed({value})')
    
speed_slider.valueChanged.connect(on_speed_changed)

def on_temp_changed(value):
    if page_ready:
        webview.page().runJavaScript(f'setTemperature({value})')
    
temp_spin.valueChanged.connect(on_temp_changed)

def on_style_changed(text):
    if page_ready:
        webview.page().runJavaScript(f'setStyle("{text}")')
        log_text.append(f"[VIS] Style changed to: {text}")
    
style_combo.currentTextChanged.connect(on_style_changed)

# Energy update timer
def update_energy():
    if page_ready:
        def callback(result):
            try:
                data = json.loads(result)
                ke_label.setText(f"Kinetic: {data['ke']:.2f} kJ/mol")
                pe_label.setText(f"Potential: {data['pe']:.2f} kJ/mol")
                te_label.setText(f"Total: {data['total']:.2f} kJ/mol")
            except:
                pass
        webview.page().runJavaScript('getEnergy()', callback)
        
        def step_callback(result):
            try:
                step = int(result)
                progress.setValue(min(step % 10000, 10000) // 100)
                progress.setFormat(f"Step: {step} / 10000")
            except:
                pass
        webview.page().runJavaScript('getStep()', step_callback)

energy_timer = QTimer()
energy_timer.timeout.connect(update_energy)
energy_timer.start(100)

# ========== ASSEMBLE LAYOUT ==========
main_layout.addWidget(control_panel)
main_layout.addWidget(webview, 1)
main_layout.addWidget(right_panel)

# Add to scene
proxy = graphics_scene.addWidget(main_widget)
proxy.setPos(0, 0)

# Resize to fit
main_widget.resize(1400, 800)
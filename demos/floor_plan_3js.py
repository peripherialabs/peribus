from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider

# Create container widget for the 3D floor plan
class FloorPlan3DViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1200, 800)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create WebEngineView for Three.js
        self.web_view = QWebEngineView()
        self.web_view.setStyleSheet("background: transparent; border-radius: 15px;")
        self.web_view.page().setBackgroundColor(Qt.transparent)
        
        # Three.js HTML with 3D Floor Plan
        html_content = '''
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            overflow: hidden; 
            background: transparent;
            border-radius: 15px;
        }
        canvas { 
            display: block; 
            border-radius: 15px;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div id="info">
        ð±ï¸ Left-click + drag: Rotate | Scroll: Zoom | Right-click + drag: Pan
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        // Camera
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(15, 20, 15);
        camera.lookAt(0, 0, 0);
        
        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.body.appendChild(renderer.domElement);
        
        // Orbit Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2.1;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);
        
        const pointLight = new THREE.PointLight(0xffaa00, 0.5, 20);
        pointLight.position.set(0, 5, 0);
        scene.add(pointLight);
        
        // Materials
        const floorMaterial = new THREE.MeshLambertMaterial({ color: 0x8B7355 });
        const wallMaterial = new THREE.MeshLambertMaterial({ color: 0xe8e8e8 });
        const wallMaterialDark = new THREE.MeshLambertMaterial({ color: 0xcccccc });
        const windowMaterial = new THREE.MeshLambertMaterial({ color: 0x87CEEB, transparent: true, opacity: 0.5 });
        const doorMaterial = new THREE.MeshLambertMaterial({ color: 0x654321 });
        const furnitureMaterial = new THREE.MeshLambertMaterial({ color: 0x4a4a4a });
        const sofaMaterial = new THREE.MeshLambertMaterial({ color: 0x2c5f2d });
        const tableMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
        const bedMaterial = new THREE.MeshLambertMaterial({ color: 0x4169E1 });
        const kitchenMaterial = new THREE.MeshLambertMaterial({ color: 0xf5f5f5 });
        const rugMaterial = new THREE.MeshLambertMaterial({ color: 0x8B0000 });
        
        // Floor dimensions
        const floorWidth = 20;
        const floorDepth = 16;
        const wallHeight = 3.5;
        const wallThickness = 0.2;
        
        // Create floor
        const floorGeometry = new THREE.BoxGeometry(floorWidth, 0.2, floorDepth);
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.position.y = -0.1;
        floor.receiveShadow = true;
        scene.add(floor);
        
        // Create walls function
        function createWall(width, height, depth, x, y, z, material = wallMaterial) {
            const geometry = new THREE.BoxGeometry(width, height, depth);
            const wall = new THREE.Mesh(geometry, material);
            wall.position.set(x, y, z);
            wall.castShadow = true;
            wall.receiveShadow = true;
            scene.add(wall);
            return wall;
        }
        
        // Exterior walls
        // Front wall (with door opening)
        createWall(7, wallHeight, wallThickness, -6.5, wallHeight/2, -floorDepth/2);
        createWall(7, wallHeight, wallThickness, 6.5, wallHeight/2, -floorDepth/2);
        createWall(6, wallHeight - 2.5, wallThickness, 0, wallHeight - 0.5, -floorDepth/2);
        
        // Back wall (with windows)
        createWall(6, wallHeight, wallThickness, -7, wallHeight/2, floorDepth/2);
        createWall(6, wallHeight, wallThickness, 7, wallHeight/2, floorDepth/2);
        createWall(2, 1, wallThickness, 0, 0.5, floorDepth/2);
        createWall(2, 1, wallThickness, 0, wallHeight - 0.5, floorDepth/2);
        createWall(4, wallHeight, wallThickness, 2, wallHeight/2, floorDepth/2);
        
        // Left wall
        createWall(wallThickness, wallHeight, floorDepth, -floorWidth/2, wallHeight/2, 0);
        
        // Right wall
        createWall(wallThickness, wallHeight, floorDepth, floorWidth/2, wallHeight/2, 0);
        
        // Interior walls
        // Living room / Kitchen divider (partial)
        createWall(wallThickness, wallHeight, 6, -3, wallHeight/2, 5);
        
        // Bedroom wall
        createWall(8, wallHeight, wallThickness, 6, wallHeight/2, 2);
        createWall(wallThickness, wallHeight, 6, 2, wallHeight/2, 5);
        
        // Bathroom walls
        createWall(4, wallHeight, wallThickness, 8, wallHeight/2, -3);
        createWall(wallThickness, wallHeight, 5, 6, wallHeight/2, -5.5);
        
        // Door
        const doorGeometry = new THREE.BoxGeometry(2, 2.5, 0.1);
        const door = new THREE.Mesh(doorGeometry, doorMaterial);
        door.position.set(0, 1.25, -floorDepth/2 + 0.15);
        scene.add(door);
        
        // Windows (back wall)
        const windowGeometry = new THREE.BoxGeometry(4, 2, 0.1);
        const window1 = new THREE.Mesh(windowGeometry, windowMaterial);
        window1.position.set(-4, 2, floorDepth/2 - 0.05);
        scene.add(window1);
        
        // Living Room Furniture
        // Sofa
        function createSofa(x, z, rotation = 0) {
            const sofaGroup = new THREE.Group();
            
            // Base
            const baseGeom = new THREE.BoxGeometry(3, 0.4, 1);
            const base = new THREE.Mesh(baseGeom, sofaMaterial);
            base.position.y = 0.2;
            sofaGroup.add(base);
            
            // Back
            const backGeom = new THREE.BoxGeometry(3, 0.8, 0.2);
            const back = new THREE.Mesh(backGeom, sofaMaterial);
            back.position.set(0, 0.6, -0.4);
            sofaGroup.add(back);
            
            // Arms
            const armGeom = new THREE.BoxGeometry(0.2, 0.5, 1);
            const leftArm = new THREE.Mesh(armGeom, sofaMaterial);
            leftArm.position.set(-1.4, 0.45, 0);
            sofaGroup.add(leftArm);
            
            const rightArm = new THREE.Mesh(armGeom, sofaMaterial);
            rightArm.position.set(1.4, 0.45, 0);
            sofaGroup.add(rightArm);
            
            // Cushions
            const cushionGeom = new THREE.BoxGeometry(0.9, 0.15, 0.7);
            const cushionMat = new THREE.MeshLambertMaterial({ color: 0x3d8b3d });
            for (let i = -1; i <= 1; i++) {
                const cushion = new THREE.Mesh(cushionGeom, cushionMat);
                cushion.position.set(i * 0.95, 0.48, 0.1);
                sofaGroup.add(cushion);
            }
            
            sofaGroup.position.set(x, 0, z);
            sofaGroup.rotation.y = rotation;
            sofaGroup.castShadow = true;
            scene.add(sofaGroup);
        }
        
        createSofa(-6, 4, 0);
        
        // Coffee table
        function createCoffeeTable(x, z) {
            const tableGroup = new THREE.Group();
            
            const topGeom = new THREE.BoxGeometry(1.5, 0.1, 0.8);
            const top = new THREE.Mesh(topGeom, tableMaterial);
            top.position.y = 0.45;
            tableGroup.add(top);
            
            const legGeom = new THREE.BoxGeometry(0.1, 0.4, 0.1);
            const positions = [[-0.6, 0.2, -0.3], [0.6, 0.2, -0.3], [-0.6, 0.2, 0.3], [0.6, 0.2, 0.3]];
            positions.forEach(pos => {
                const leg = new THREE.Mesh(legGeom, tableMaterial);
                leg.position.set(...pos);
                tableGroup.add(leg);
            });
            
            tableGroup.position.set(x, 0, z);
            scene.add(tableGroup);
        }
        
        createCoffeeTable(-6, 6);
        
        // TV Stand
        const tvStandGeom = new THREE.BoxGeometry(2.5, 0.6, 0.5);
        const tvStand = new THREE.Mesh(tvStandGeom, furnitureMaterial);
        tvStand.position.set(-6, 0.3, 0);
        scene.add(tvStand);
        
        // TV
        const tvGeom = new THREE.BoxGeometry(2, 1.2, 0.1);
        const tvMat = new THREE.MeshLambertMaterial({ color: 0x111111 });
        const tv = new THREE.Mesh(tvGeom, tvMat);
        tv.position.set(-6, 1.3, 0);
        scene.add(tv);
        
        // TV Screen
        const screenGeom = new THREE.BoxGeometry(1.8, 1, 0.05);
        const screenMat = new THREE.MeshLambertMaterial({ color: 0x222244 });
        const screen = new THREE.Mesh(screenGeom, screenMat);
        screen.position.set(-6, 1.3, 0.08);
        scene.add(screen);
        
        // Rug
        const rugGeom = new THREE.BoxGeometry(3, 0.02, 2);
        const rug = new THREE.Mesh(rugGeom, rugMaterial);
        rug.position.set(-6, 0.01, 5);
        scene.add(rug);
        
        // Kitchen
        // Counter
        const counterGeom = new THREE.BoxGeometry(4, 1, 0.6);
        const counter = new THREE.Mesh(counterGeom, kitchenMaterial);
        counter.position.set(-6, 0.5, -4);
        scene.add(counter);
        
        // Upper cabinets
        const cabinetGeom = new THREE.BoxGeometry(4, 0.8, 0.4);
        const cabinet = new THREE.Mesh(cabinetGeom, kitchenMaterial);
        cabinet.position.set(-6, 2.5, -4.1);
        scene.add(cabinet);
        
        // Stove
        const stoveGeom = new THREE.BoxGeometry(0.8, 0.05, 0.5);
        const stoveMat = new THREE.MeshLambertMaterial({ color: 0x333333 });
        const stove = new THREE.Mesh(stoveGeom, stoveMat);
        stove.position.set(-5, 1.03, -4);
        scene.add(stove);
        
        // Sink
        const sinkGeom = new THREE.BoxGeometry(0.6, 0.15, 0.4);
        const sinkMat = new THREE.MeshLambertMaterial({ color: 0xaaaaaa });
        const sink = new THREE.Mesh(sinkGeom, sinkMat);
        sink.position.set(-7, 0.95, -4);
        scene.add(sink);
        
        // Dining table
        function createDiningTable(x, z) {
            const group = new THREE.Group();
            
            const topGeom = new THREE.BoxGeometry(1.8, 0.1, 1.2);
            const top = new THREE.Mesh(topGeom, tableMaterial);
            top.position.y = 0.8;
            group.add(top);
            
            const legGeom = new THREE.BoxGeometry(0.1, 0.75, 0.1);
            const positions = [[-0.8, 0.375, -0.5], [0.8, 0.375, -0.5], [-0.8, 0.375, 0.5], [0.8, 0.375, 0.5]];
            positions.forEach(pos => {
                const leg = new THREE.Mesh(legGeom, tableMaterial);
                leg.position.set(...pos);
                group.add(leg);
            });
            
            group.position.set(x, 0, z);
            scene.add(group);
        }
        
        createDiningTable(-6, -1);
        
        // Dining chairs
        function createChair(x, z, rotation = 0) {
            const group = new THREE.Group();
            
            const seatGeom = new THREE.BoxGeometry(0.5, 0.05, 0.5);
            const seat = new THREE.Mesh(seatGeom, furnitureMaterial);
            seat.position.y = 0.5;
            group.add(seat);
            
            const backGeom = new THREE.BoxGeometry(0.5, 0.6, 0.05);
            const back = new THREE.Mesh(backGeom, furnitureMaterial);
            back.position.set(0, 0.8, -0.225);
            group.add(back);
            
            const legGeom = new THREE.BoxGeometry(0.05, 0.5, 0.05);
            const positions = [[-0.2, 0.25, -0.2], [0.2, 0.25, -0.2], [-0.2, 0.25, 0.2], [0.2, 0.25, 0.2]];
            positions.forEach(pos => {
                const leg = new THREE.Mesh(legGeom, furnitureMaterial);
                leg.position.set(...pos);
                group.add(leg);
            });
            
            group.position.set(x, 0, z);
            group.rotation.y = rotation;
            scene.add(group);
        }
        
        createChair(-5.2, -1, Math.PI/2);
        createChair(-6.8, -1, -Math.PI/2);
        createChair(-6, -0.2, Math.PI);
        createChair(-6, -1.8, 0);
        
        // Bedroom
        // Bed
        function createBed(x, z, rotation = 0) {
            const group = new THREE.Group();
            
            // Frame
            const frameGeom = new THREE.BoxGeometry(2.2, 0.4, 2.5);
            const frame = new THREE.Mesh(frameGeom, tableMaterial);
            frame.position.y = 0.2;
            group.add(frame);
            
            // Mattress
            const mattressGeom = new THREE.BoxGeometry(2, 0.3, 2.3);
            const mattress = new THREE.Mesh(mattressGeom, bedMaterial);
            mattress.position.y = 0.55;
            group.add(mattress);
            
            // Pillows
            const pillowGeom = new THREE.BoxGeometry(0.6, 0.15, 0.4);
            const pillowMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
            const pillow1 = new THREE.Mesh(pillowGeom, pillowMat);
            pillow1.position.set(-0.5, 0.78, -0.9);
            group.add(pillow1);
            
            const pillow2 = new THREE.Mesh(pillowGeom, pillowMat);
            pillow2.position.set(0.5, 0.78, -0.9);
            group.add(pillow2);
            
            // Headboard
            const headboardGeom = new THREE.BoxGeometry(2.2, 1, 0.15);
            const headboard = new THREE.Mesh(headboardGeom, tableMaterial);
            headboard.position.set(0, 0.9, -1.2);
            group.add(headboard);
            
            group.position.set(x, 0, z);
            group.rotation.y = rotation;
            scene.add(group);
        }
        
        createBed(7, 5.5, 0);
        
        // Nightstands
        function createNightstand(x, z) {
            const group = new THREE.Group();
            
            const bodyGeom = new THREE.BoxGeometry(0.5, 0.5, 0.4);
            const body = new THREE.Mesh(bodyGeom, tableMaterial);
            body.position.y = 0.25;
            group.add(body);
            
            // Lamp
            const lampBaseGeom = new THREE.CylinderGeometry(0.08, 0.1, 0.05, 16);
            const lampBase = new THREE.Mesh(lampBaseGeom, furnitureMaterial);
            lampBase.position.y = 0.525;
            group.add(lampBase);
            
            const lampPoleGeom = new THREE.CylinderGeometry(0.02, 0.02, 0.3, 8);
            const lampPole = new THREE.Mesh(lampPoleGeom, furnitureMaterial);
            lampPole.position.y = 0.7;
            group.add(lampPole);
            
            const lampShadeGeom = new THREE.CylinderGeometry(0.15, 0.1, 0.15, 16);
            const lampShadeMat = new THREE.MeshLambertMaterial({ color: 0xffffcc });
            const lampShade = new THREE.Mesh(lampShadeGeom, lampShadeMat);
            lampShade.position.y = 0.925;
            group.add(lampShade);
            
            group.position.set(x, 0, z);
            scene.add(group);
        }
        
        createNightstand(5.5, 4);
        createNightstand(8.5, 4);
        
        // Wardrobe
        const wardrobeGeom = new THREE.BoxGeometry(1.5, 2.5, 0.6);
        const wardrobe = new THREE.Mesh(wardrobeGeom, tableMaterial);
        wardrobe.position.set(9, 1.25, 5);
        scene.add(wardrobe);
        
        // Bathroom
        // Toilet
        function createToilet(x, z, rotation = 0) {
            const group = new THREE.Group();
            
            const baseGeom = new THREE.BoxGeometry(0.5, 0.4, 0.6);
            const baseMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
            const base = new THREE.Mesh(baseGeom, baseMat);
            base.position.y = 0.2;
            group.add(base);
            
            const tankGeom = new THREE.BoxGeometry(0.4, 0.5, 0.2);
            const tank = new THREE.Mesh(tankGeom, baseMat);
            tank.position.set(0, 0.55, -0.2);
            group.add(tank);
            
            group.position.set(x, 0, z);
            group.rotation.y = rotation;
            scene.add(group);
        }
        
        createToilet(8, -6, 0);
        
        // Bathroom sink
        const bathSinkGeom = new THREE.BoxGeometry(0.8, 0.1, 0.5);
        const bathSinkMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
        const bathSink = new THREE.Mesh(bathSinkGeom, bathSinkMat);
        bathSink.position.set(7, 0.9, -7.5);
        scene.add(bathSink);
        
        // Bathroom vanity
        const vanityGeom = new THREE.BoxGeometry(0.8, 0.85, 0.5);
        const vanity = new THREE.Mesh(vanityGeom, kitchenMaterial);
        vanity.position.set(7, 0.425, -7.5);
        scene.add(vanity);
        
        // Shower
        const showerBaseGeom = new THREE.BoxGeometry(1.2, 0.1, 1.2);
        const showerBase = new THREE.Mesh(showerBaseGeom, bathSinkMat);
        showerBase.position.set(9, 0.05, -6.5);
        scene.add(showerBase);
        
        // Shower glass
        const showerGlassGeom = new THREE.BoxGeometry(0.05, 2.2, 1.2);
        const showerGlassMat = new THREE.MeshLambertMaterial({ color: 0xaaddff, transparent: true, opacity: 0.3 });
        const showerGlass = new THREE.Mesh(showerGlassGeom, showerGlassMat);
        showerGlass.position.set(8.35, 1.1, -6.5);
        scene.add(showerGlass);
        
        // Plants
        function createPlant(x, z) {
            const group = new THREE.Group();
            
            const potGeom = new THREE.CylinderGeometry(0.15, 0.12, 0.25, 16);
            const potMat = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
            const pot = new THREE.Mesh(potGeom, potMat);
            pot.position.y = 0.125;
            group.add(pot);
            
            const plantGeom = new THREE.SphereGeometry(0.25, 16, 16);
            const plantMat = new THREE.MeshLambertMaterial({ color: 0x228B22 });
            const plant = new THREE.Mesh(plantGeom, plantMat);
            plant.position.y = 0.4;
            group.add(plant);
            
            group.position.set(x, 0, z);
            scene.add(group);
        }
        
        createPlant(-8.5, 6);
        createPlant(4, -6);
        
        // Grid helper (optional, for reference)
        // const gridHelper = new THREE.GridHelper(30, 30, 0x444444, 0x222222);
        // scene.add(gridHelper);
        
        // Room labels (using sprites)
        function createLabel(text, x, y, z) {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 256;
            canvas.height = 64;
            
            context.fillStyle = 'rgba(0, 0, 0, 0.7)';
            context.roundRect(0, 0, 256, 64, 10);
            context.fill();
            
            context.font = 'bold 28px Arial';
            context.fillStyle = 'white';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(text, 128, 32);
            
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.position.set(x, y, z);
            sprite.scale.set(2, 0.5, 1);
            scene.add(sprite);
        }
        
        createLabel('Living Room', -6, 0.5, 4);
        createLabel('Kitchen', -6, 0.5, -3);
        createLabel('Bedroom', 7, 0.5, 5);
        createLabel('Bathroom', 8, 0.5, -5);
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // JavaScript injection function
        function executeJS(code) {
            try {
                eval(code);
            } catch(e) {
                console.error('Error executing JS:', e);
            }
        }
        window.executeJS = executeJS;
    </script>
</body>
</html>
        '''
        
        self.web_view.setHtml(html_content)
        layout.addWidget(self.web_view)
        
        # Control panel
        control_panel = QWidget()
        control_panel.setStyleSheet("""
            QWidget {
                background: rgba(30, 30, 50, 0.9);
                border-radius: 10px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a90d9, stop:1 #357abd);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a9fe9, stop:1 #458acd);
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 5, 10, 5)
        
        # View buttons
        btn_top = QPushButton("Top View")
        btn_top.clicked.connect(lambda: self.set_camera_view("top"))
        control_layout.addWidget(btn_top)
        
        btn_front = QPushButton("Front View")
        btn_front.clicked.connect(lambda: self.set_camera_view("front"))
        control_layout.addWidget(btn_front)
        
        btn_side = QPushButton("Side View")
        btn_side.clicked.connect(lambda: self.set_camera_view("side"))
        control_layout.addWidget(btn_side)
        
        btn_perspective = QPushButton("3D View")
        btn_perspective.clicked.connect(lambda: self.set_camera_view("perspective"))
        control_layout.addWidget(btn_perspective)
        
        control_layout.addStretch()
        
        # Zoom slider
        zoom_label = QLabel("Zoom:")
        control_layout.addWidget(zoom_label)
        
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setMinimum(5)
        zoom_slider.setMaximum(50)
        zoom_slider.setValue(20)
        zoom_slider.setFixedWidth(150)
        zoom_slider.valueChanged.connect(self.set_zoom)
        control_layout.addWidget(zoom_slider)
        
        layout.addWidget(control_panel)
    
    def set_camera_view(self, view_type):
        if view_type == "top":
            js = "camera.position.set(0, 30, 0.1); camera.lookAt(0, 0, 0); controls.update();"
        elif view_type == "front":
            js = "camera.position.set(0, 10, 25); camera.lookAt(0, 0, 0); controls.update();"
        elif view_type == "side":
            js = "camera.position.set(25, 10, 0); camera.lookAt(0, 0, 0); controls.update();"
        else:  # perspective
            js = "camera.position.set(15, 20, 15); camera.lookAt(0, 0, 0); controls.update();"
        
        self.web_view.page().runJavaScript(js)
    
    def set_zoom(self, value):
        js = f"camera.position.normalize().multiplyScalar({value}); controls.update();"
        self.web_view.page().runJavaScript(js)

# Create the floor plan viewer
floor_plan_viewer = FloorPlan3DViewer()

# Add to graphics scene
floor_plan_proxy = graphics_scene.addWidget(floor_plan_viewer)

# Center in view
view = graphics_scene.views()[0]
viewport_rect = view.viewport().rect()
scene_rect = view.mapToScene(viewport_rect).boundingRect()
center_x = scene_rect.center().x() - floor_plan_viewer.width() / 2
center_y = scene_rect.center().y() - floor_plan_viewer.height() / 2
floor_plan_proxy.setPos(center_x, center_y)

# Make it movable
floor_plan_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
floor_plan_proxy.setFlag(QGraphicsItem.ItemIsSelectable, True)
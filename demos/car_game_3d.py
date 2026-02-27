from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout

# Create container widget for the game
game_container = QWidget()
game_container.setFixedSize(1200, 800)
game_container.setStyleSheet("background: transparent; border-radius: 15px;")

layout = QVBoxLayout(game_container)
layout.setContentsMargins(0, 0, 0, 0)

# Create WebEngineView for Three.js game
game_view = QWebEngineView()
game_view.setStyleSheet("background: transparent; border-radius: 15px;")

# HTML with Three.js 3D car driving game
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
            top: 20px;
            left: 20px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 16px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            z-index: 100;
        }
        #speed {
            position: absolute;
            bottom: 30px;
            right: 30px;
            color: white;
            font-family: 'Courier New', monospace;
            font-size: 32px;
            background: rgba(0,0,0,0.8);
            padding: 20px 30px;
            border-radius: 15px;
            border: 2px solid #00ff00;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="info">
        <b>Controls:</b><br>
        W / â : Accelerate<br>
        S / â : Brake/Reverse<br>
        A / â : Turn Left<br>
        D / â : Turn Right<br>
        Space : Handbrake
    </div>
    <div id="speed">0 km/h</div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x87CEEB);
        scene.fog = new THREE.Fog(0x87CEEB, 100, 500);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.body.appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
        sunLight.position.set(100, 100, 50);
        sunLight.castShadow = true;
        sunLight.shadow.mapSize.width = 2048;
        sunLight.shadow.mapSize.height = 2048;
        sunLight.shadow.camera.near = 0.5;
        sunLight.shadow.camera.far = 500;
        sunLight.shadow.camera.left = -150;
        sunLight.shadow.camera.right = 150;
        sunLight.shadow.camera.top = 150;
        sunLight.shadow.camera.bottom = -150;
        scene.add(sunLight);
        
        // Ground
        const groundGeometry = new THREE.PlaneGeometry(1000, 1000);
        const groundMaterial = new THREE.MeshLambertMaterial({ color: 0x3d5c3d });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        scene.add(ground);
        
        // Road system
        function createRoad(x, z, width, length, rotation = 0) {
            const roadGeometry = new THREE.PlaneGeometry(width, length);
            const roadMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
            const road = new THREE.Mesh(roadGeometry, roadMaterial);
            road.rotation.x = -Math.PI / 2;
            road.rotation.z = rotation;
            road.position.set(x, 0.01, z);
            road.receiveShadow = true;
            scene.add(road);
            
            // Road lines
            const lineGeometry = new THREE.PlaneGeometry(0.3, length - 2);
            const lineMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
            const line = new THREE.Mesh(lineGeometry, lineMaterial);
            line.rotation.x = -Math.PI / 2;
            line.rotation.z = rotation;
            line.position.set(x, 0.02, z);
            scene.add(line);
            
            return road;
        }
        
        // Create city grid roads
        const roadWidth = 15;
        const blockSize = 80;
        
        for (let i = -3; i <= 3; i++) {
            createRoad(i * blockSize, 0, roadWidth, 600);
            createRoad(0, i * blockSize, 600, roadWidth);
        }
        
        // Building creation
        function createBuilding(x, z, width, depth, height, color) {
            const group = new THREE.Group();
            
            // Main building
            const buildingGeometry = new THREE.BoxGeometry(width, height, depth);
            const buildingMaterial = new THREE.MeshLambertMaterial({ color: color });
            const building = new THREE.Mesh(buildingGeometry, buildingMaterial);
            building.position.y = height / 2;
            building.castShadow = true;
            building.receiveShadow = true;
            group.add(building);
            
            // Windows
            const windowMaterial = new THREE.MeshBasicMaterial({ color: 0x88ccff });
            const windowSize = 2;
            const windowSpacing = 5;
            
            for (let floor = 3; floor < height - 2; floor += windowSpacing) {
                for (let wx = -width/2 + 3; wx < width/2 - 2; wx += windowSpacing) {
                    // Front windows
                    const windowGeom = new THREE.PlaneGeometry(windowSize, windowSize * 1.5);
                    const windowMesh = new THREE.Mesh(windowGeom, windowMaterial);
                    windowMesh.position.set(wx, floor, depth/2 + 0.1);
                    group.add(windowMesh);
                    
                    // Back windows
                    const windowBack = windowMesh.clone();
                    windowBack.position.z = -depth/2 - 0.1;
                    windowBack.rotation.y = Math.PI;
                    group.add(windowBack);
                }
                
                for (let wz = -depth/2 + 3; wz < depth/2 - 2; wz += windowSpacing) {
                    // Side windows
                    const windowGeom = new THREE.PlaneGeometry(windowSize, windowSize * 1.5);
                    const windowMesh = new THREE.Mesh(windowGeom, windowMaterial);
                    windowMesh.position.set(width/2 + 0.1, floor, wz);
                    windowMesh.rotation.y = Math.PI / 2;
                    group.add(windowMesh);
                    
                    const windowOther = windowMesh.clone();
                    windowOther.position.x = -width/2 - 0.1;
                    windowOther.rotation.y = -Math.PI / 2;
                    group.add(windowOther);
                }
            }
            
            group.position.set(x, 0, z);
            scene.add(group);
            return group;
        }
        
        // Create city buildings
        const buildingColors = [0x8899aa, 0x667788, 0x556677, 0x778899, 0x99aabb, 0x445566];
        
        for (let bx = -3; bx <= 3; bx++) {
            for (let bz = -3; bz <= 3; bz++) {
                if (Math.abs(bx) <= 0 && Math.abs(bz) <= 0) continue;
                
                const baseX = bx * blockSize;
                const baseZ = bz * blockSize;
                const offset = blockSize / 2 - roadWidth;
                
                // Random buildings in each block
                const numBuildings = 1 + Math.floor(Math.random() * 2);
                for (let b = 0; b < numBuildings; b++) {
                    const bWidth = 15 + Math.random() * 20;
                    const bDepth = 15 + Math.random() * 20;
                    const bHeight = 20 + Math.random() * 60;
                    const color = buildingColors[Math.floor(Math.random() * buildingColors.length)];
                    
                    const posX = baseX + (Math.random() - 0.5) * (offset - bWidth/2);
                    const posZ = baseZ + (Math.random() - 0.5) * (offset - bDepth/2);
                    
                    createBuilding(posX, posZ, bWidth, bDepth, bHeight, color);
                }
            }
        }
        
        // Create car
        function createCar() {
            const car = new THREE.Group();
            
            // Car body
            const bodyGeometry = new THREE.BoxGeometry(2, 0.8, 4.5);
            const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0xff0000 });
            const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
            body.position.y = 0.6;
            body.castShadow = true;
            car.add(body);
            
            // Car top/cabin
            const cabinGeometry = new THREE.BoxGeometry(1.8, 0.7, 2.2);
            const cabinMaterial = new THREE.MeshLambertMaterial({ color: 0xcc0000 });
            const cabin = new THREE.Mesh(cabinGeometry, cabinMaterial);
            cabin.position.set(0, 1.15, -0.3);
            cabin.castShadow = true;
            car.add(cabin);
            
            // Windshield
            const windshieldGeometry = new THREE.PlaneGeometry(1.6, 0.6);
            const windshieldMaterial = new THREE.MeshBasicMaterial({ 
                color: 0x88ccff, 
                transparent: true, 
                opacity: 0.7,
                side: THREE.DoubleSide
            });
            const windshield = new THREE.Mesh(windshieldGeometry, windshieldMaterial);
            windshield.position.set(0, 1.2, 0.85);
            windshield.rotation.x = Math.PI / 6;
            car.add(windshield);
            
            // Rear window
            const rearWindow = windshield.clone();
            rearWindow.position.set(0, 1.2, -1.45);
            rearWindow.rotation.x = -Math.PI / 6;
            car.add(rearWindow);
            
            // Headlights
            const headlightGeometry = new THREE.CircleGeometry(0.15, 16);
            const headlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffcc });
            
            const headlightL = new THREE.Mesh(headlightGeometry, headlightMaterial);
            headlightL.position.set(-0.6, 0.5, 2.26);
            car.add(headlightL);
            
            const headlightR = headlightL.clone();
            headlightR.position.x = 0.6;
            car.add(headlightR);
            
            // Taillights
            const taillightMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const taillightL = new THREE.Mesh(headlightGeometry, taillightMaterial);
            taillightL.position.set(-0.6, 0.5, -2.26);
            taillightL.rotation.y = Math.PI;
            car.add(taillightL);
            
            const taillightR = taillightL.clone();
            taillightR.position.x = 0.6;
            car.add(taillightR);
            
            // Wheels
            const wheelGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
            const wheelMaterial = new THREE.MeshLambertMaterial({ color: 0x222222 });
            const hubMaterial = new THREE.MeshLambertMaterial({ color: 0xcccccc });
            
            const wheelPositions = [
                [-1.1, 0.4, 1.3],
                [1.1, 0.4, 1.3],
                [-1.1, 0.4, -1.3],
                [1.1, 0.4, -1.3]
            ];
            
            car.wheels = [];
            wheelPositions.forEach(pos => {
                const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
                wheel.rotation.z = Math.PI / 2;
                wheel.position.set(...pos);
                wheel.castShadow = true;
                car.add(wheel);
                car.wheels.push(wheel);
                
                // Hub cap
                const hubGeometry = new THREE.CircleGeometry(0.25, 8);
                const hub = new THREE.Mesh(hubGeometry, hubMaterial);
                hub.position.set(pos[0] + (pos[0] > 0 ? 0.16 : -0.16), pos[1], pos[2]);
                hub.rotation.y = pos[0] > 0 ? Math.PI / 2 : -Math.PI / 2;
                car.add(hub);
            });
            
            return car;
        }
        
        const car = createCar();
        car.position.set(0, 0, 0);
        scene.add(car);
        
        // Car physics
        const carState = {
            speed: 0,
            maxSpeed: 80,
            acceleration: 30,
            braking: 40,
            friction: 10,
            turnSpeed: 2.5,
            rotation: 0,
            wheelRotation: 0
        };
        
        // Controls
        const keys = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            brake: false
        };
        
        document.addEventListener('keydown', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w': case 'arrowup': keys.forward = true; break;
                case 's': case 'arrowdown': keys.backward = true; break;
                case 'a': case 'arrowleft': keys.left = true; break;
                case 'd': case 'arrowright': keys.right = true; break;
                case ' ': keys.brake = true; break;
            }
        });
        
        document.addEventListener('keyup', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w': case 'arrowup': keys.forward = false; break;
                case 's': case 'arrowdown': keys.backward = false; break;
                case 'a': case 'arrowleft': keys.left = false; break;
                case 'd': case 'arrowright': keys.right = false; break;
                case ' ': keys.brake = false; break;
            }
        });
        
        // Camera settings
        const cameraOffset = new THREE.Vector3(0, 6, -12);
        const cameraLookOffset = new THREE.Vector3(0, 2, 10);
        
        // Speed display
        const speedDisplay = document.getElementById('speed');
        
        // Animation loop
        const clock = new THREE.Clock();
        
        function animate() {
            requestAnimationFrame(animate);
            
            const delta = Math.min(clock.getDelta(), 0.1);
            
            // Car physics
            if (keys.forward) {
                carState.speed += carState.acceleration * delta;
            } else if (keys.backward) {
                carState.speed -= carState.acceleration * delta;
            } else {
                // Friction
                if (carState.speed > 0) {
                    carState.speed = Math.max(0, carState.speed - carState.friction * delta);
                } else if (carState.speed < 0) {
                    carState.speed = Math.min(0, carState.speed + carState.friction * delta);
                }
            }
            
            // Handbrake
            if (keys.brake) {
                if (carState.speed > 0) {
                    carState.speed = Math.max(0, carState.speed - carState.braking * 2 * delta);
                } else {
                    carState.speed = Math.min(0, carState.speed + carState.braking * 2 * delta);
                }
            }
            
            // Clamp speed
            carState.speed = Math.max(-carState.maxSpeed / 3, Math.min(carState.maxSpeed, carState.speed));
            
            // Turning (only when moving)
            if (Math.abs(carState.speed) > 0.5) {
                const turnFactor = Math.min(1, Math.abs(carState.speed) / 20);
                if (keys.left) {
                    carState.rotation += carState.turnSpeed * turnFactor * delta * Math.sign(carState.speed);
                }
                if (keys.right) {
                    carState.rotation -= carState.turnSpeed * turnFactor * delta * Math.sign(carState.speed);
                }
            }
            
            // Update car position
            const moveX = Math.sin(carState.rotation) * carState.speed * delta;
            const moveZ = Math.cos(carState.rotation) * carState.speed * delta;
            car.position.x += moveX;
            car.position.z += moveZ;
            car.rotation.y = carState.rotation;
            
            // Rotate wheels
            carState.wheelRotation += carState.speed * delta * 0.5;
            car.wheels.forEach(wheel => {
                wheel.rotation.x = carState.wheelRotation;
            });
            
            // Update speed display
            const displaySpeed = Math.abs(Math.round(carState.speed * 3.6));
            speedDisplay.textContent = displaySpeed + ' km/h';
            
            // Camera follow
            const idealOffset = cameraOffset.clone();
            idealOffset.applyAxisAngle(new THREE.Vector3(0, 1, 0), carState.rotation);
            idealOffset.add(car.position);
            
            const idealLookAt = cameraLookOffset.clone();
            idealLookAt.applyAxisAngle(new THREE.Vector3(0, 1, 0), carState.rotation);
            idealLookAt.add(car.position);
            
            camera.position.lerp(idealOffset, 5 * delta);
            
            const lookAtTarget = new THREE.Vector3();
            lookAtTarget.copy(car.position);
            lookAtTarget.y += 1.5;
            camera.lookAt(lookAtTarget);
            
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Focus for keyboard input
        document.body.tabIndex = 0;
        document.body.focus();
        
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

game_view.setHtml(html_content)
layout.addWidget(game_view)

# Add to graphics scene
game_proxy = graphics_scene.addWidget(game_container)

# Center on screen
view = graphics_scene.views()[0]
viewport_rect = view.viewport().rect()
scene_rect = view.mapToScene(viewport_rect).boundingRect()

center_x = scene_rect.center().x() - 600
center_y = scene_rect.center().y() - 400

game_proxy.setPos(center_x, center_y)
game_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
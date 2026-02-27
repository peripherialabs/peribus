from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout

game_container = QWidget()
game_container.setFixedSize(1200, 800)
game_container.setStyleSheet("background: transparent; border-radius: 15px;")

layout = QVBoxLayout(game_container)
layout.setContentsMargins(0, 0, 0, 0)

game_view = QWebEngineView()
game_view.setStyleSheet("background: transparent; border-radius: 15px;")

html_content = '''
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { overflow:hidden; background:#1a0a2e; border-radius:15px; font-family:Arial,sans-serif; }
        canvas { display:block; border-radius:15px; }
        #hud { position:absolute; top:0;left:0;right:0;bottom:0; pointer-events:none; z-index:100; }
        #speed-box {
            position:absolute; bottom:24px; right:24px;
            background:rgba(12,4,20,0.85); padding:18px 28px; border-radius:14px;
            border:1px solid rgba(255,150,80,0.2); text-align:center; min-width:120px;
        }
        #speed { color:#ffcc66; font-family:monospace; font-size:40px; font-weight:900;
            text-shadow:0 0 20px rgba(255,180,80,0.5); line-height:1; }
        #speed-unit { color:rgba(255,180,120,0.4); font-family:monospace; font-size:10px; letter-spacing:3px; margin-top:4px; }
        #gear { color:#ff8844; font-family:monospace; font-size:11px; margin-top:5px; letter-spacing:2px; }
        #controls {
            position:absolute; bottom:24px; left:24px; color:#ffd4a0; font-size:12px;
            background:rgba(12,4,20,0.75); padding:10px 16px; border-radius:10px;
            border:1px solid rgba(255,150,80,0.15); line-height:1.7;
        }
        #controls kbd {
            background:rgba(255,150,80,0.18); border:1px solid rgba(255,150,80,0.3);
            border-radius:3px; padding:1px 5px; font-family:monospace; font-size:10px; color:#ffcc88;
        }
        .pill {
            position:absolute; top:18px; left:18px;
            background:rgba(12,4,20,0.75); padding:8px 16px; border-radius:20px;
            border:1px solid rgba(255,150,80,0.15); color:#ffc080; font-size:13px;
            display:flex; align-items:center; gap:6px;
        }
        .pill .dot { width:5px;height:5px;border-radius:50%;background:#ff8844;box-shadow:0 0 5px #ff8844; }
        #snow-pill {
            position:absolute; top:18px; left:160px;
            background:rgba(12,4,20,0.75); padding:8px 16px; border-radius:20px;
            border:1px solid rgba(255,150,80,0.15); color:#ffc080; font-size:13px;
            display:none; align-items:center; gap:6px;
        }
        #collision-flash {
            position:absolute; top:0;left:0;right:0;bottom:0;
            background:radial-gradient(circle,rgba(255,60,20,0.4),transparent 70%);
            opacity:0; transition:opacity 0.1s; pointer-events:none;
        }
    </style>
</head>
<body>
    <div id="hud">
        <div class="pill"><div class="dot"></div><span id="npc-count">0</span></div>
        <div id="snow-pill"><div class="dot" style="background:#adf;box-shadow:0 0 5px #adf"></div>Snow</div>
        <div id="speed-box"><div id="speed">0</div><div id="speed-unit">KM/H</div><div id="gear">P</div></div>
        <div id="controls"><kbd>W</kbd> Gas <kbd>S</kbd> Brake <kbd>A</kbd><kbd>D</kbd> Steer <kbd>SPACE</kbd> Handbrake</div>
        <div id="collision-flash"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function(){
        var scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a0a2e);
        scene.fog = new THREE.Fog(0x2a1535, 40, 320);
        var camera = new THREE.PerspectiveCamera(68, window.innerWidth/window.innerHeight, 0.1, 500);
        var renderer = new THREE.WebGLRenderer({ antialias:true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.body.appendChild(renderer.domElement);

        // === SKY ===
        var skyGeo = new THREE.SphereGeometry(380,24,16);
        var sp = skyGeo.attributes.position.array;
        var sc = new Float32Array(sp.length);
        for(var i=0;i<sp.length;i+=3){
            var h=(sp[i+1]/380+1)*0.5;
            if(h<0.35){sc[i]=1.0;sc[i+1]=0.35;sc[i+2]=0.1;}
            else if(h<0.5){var t=(h-0.35)/0.15;sc[i]=1.0-t*0.6;sc[i+1]=0.35-t*0.12;sc[i+2]=0.1+t*0.16;}
            else if(h<0.7){var t=(h-0.5)/0.2;sc[i]=0.4-t*0.2;sc[i+1]=0.23-t*0.08;sc[i+2]=0.26+t*0.06;}
            else{var t=(h-0.7)/0.3;sc[i]=0.2-t*0.14;sc[i+1]=0.15-t*0.1;sc[i+2]=0.32-t*0.15;}
        }
        skyGeo.setAttribute('color',new THREE.Float32BufferAttribute(sc,3));
        var skyMesh=new THREE.Mesh(skyGeo,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.BackSide,depthWrite:false,fog:false}));
        scene.add(skyMesh);
        // Sun - directly ahead on the road (+Z)
        var sunG1=new THREE.Mesh(new THREE.SphereGeometry(40,12,12),new THREE.MeshBasicMaterial({color:0xffdd44,transparent:true,opacity:0.45,fog:false}));
        sunG1.position.set(0,22,370);scene.add(sunG1);
        var sunG2=new THREE.Mesh(new THREE.SphereGeometry(18,12,12),new THREE.MeshBasicMaterial({color:0xffffaa,transparent:true,opacity:0.8,fog:false}));
        sunG2.position.set(0,22,370);scene.add(sunG2);
        // Horizon haze
        var hazeGeo=new THREE.PlaneGeometry(800,60);
        var hazeMat=new THREE.MeshBasicMaterial({color:0xff8833,transparent:true,opacity:0.15,fog:false,side:THREE.DoubleSide});
        var haze=new THREE.Mesh(hazeGeo,hazeMat);
        haze.position.set(0,8,370);scene.add(haze);

        // === LIGHTING ===
        scene.add(new THREE.AmbientLight(0x665555,0.5));
        var sunLight=new THREE.DirectionalLight(0xff9955,0.9);
        sunLight.position.set(20,40,150);
        sunLight.castShadow=true;
        sunLight.shadow.mapSize.width=1024;sunLight.shadow.mapSize.height=1024;
        sunLight.shadow.camera.near=1;sunLight.shadow.camera.far=200;
        sunLight.shadow.camera.left=-60;sunLight.shadow.camera.right=60;
        sunLight.shadow.camera.top=60;sunLight.shadow.camera.bottom=-60;
        sunLight.shadow.bias=-0.003;
        scene.add(sunLight);scene.add(sunLight.target);
        scene.add(new THREE.HemisphereLight(0xff6633,0x222244,0.25));
        var headlight=new THREE.PointLight(0xffeedd,0.7,30,2);
        scene.add(headlight);

        // === ROAD CONFIG ===
        var RW=18; // road width
        var SW=4;  // sidewalk width
        var laneW=RW/4;
        var colliders=[];
        var npcCars=[];

        // === GROUND (dark grass/dirt on sides) ===
        var gnd=new THREE.Mesh(new THREE.PlaneGeometry(600,2000),new THREE.MeshLambertMaterial({color:0x151f15}));
        gnd.rotation.x=-Math.PI/2;gnd.position.set(0,0,0);gnd.receiveShadow=true;scene.add(gnd);

        // === SINGLE LONG ROAD (Z direction, toward sunset) ===
        var roadLen=2000;
        var roadMat=new THREE.MeshLambertMaterial({color:0x2a2a2a});
        var road=new THREE.Mesh(new THREE.PlaneGeometry(RW,roadLen),roadMat);
        road.rotation.x=-Math.PI/2;road.position.set(0,0.01,0);road.receiveShadow=true;scene.add(road);

        // Center dashes
        var dashMat=new THREE.MeshBasicMaterial({color:0xccaa44});
        for(var d=-roadLen/2;d<roadLen/2;d+=6){
            var dm=new THREE.Mesh(new THREE.PlaneGeometry(0.22,3),dashMat);
            dm.rotation.x=-Math.PI/2;dm.position.set(0,0.025,d);scene.add(dm);
        }
        // Lane lines (2 lanes each direction)
        var laneMat=new THREE.MeshBasicMaterial({color:0x444444});
        [-1,1].forEach(function(s){
            // Inner lane divider
            var ln=new THREE.Mesh(new THREE.PlaneGeometry(0.1,roadLen),laneMat);
            ln.rotation.x=-Math.PI/2;ln.position.set(s*laneW,0.025,0);scene.add(ln);
        });
        // Edge lines
        var edgeMat=new THREE.MeshBasicMaterial({color:0x555555});
        [-1,1].forEach(function(s){
            var el=new THREE.Mesh(new THREE.PlaneGeometry(0.15,roadLen),edgeMat);
            el.rotation.x=-Math.PI/2;el.position.set(s*(RW/2-0.2),0.025,0);scene.add(el);
        });

        // Sidewalks
        var swMat=new THREE.MeshLambertMaterial({color:0x3d3030});
        var curbMat=new THREE.MeshLambertMaterial({color:0x6b6058});
        [-1,1].forEach(function(s){
            var sw=new THREE.Mesh(new THREE.PlaneGeometry(SW,roadLen),swMat);
            sw.rotation.x=-Math.PI/2;sw.position.set(s*(RW/2+SW/2),0.05,0);sw.receiveShadow=true;scene.add(sw);
            var curb=new THREE.Mesh(new THREE.BoxGeometry(0.3,0.15,roadLen),curbMat);
            curb.position.set(s*(RW/2),0.075,0);scene.add(curb);
            var curb2=new THREE.Mesh(new THREE.BoxGeometry(0.3,0.15,roadLen),curbMat);
            curb2.position.set(s*(RW/2+SW),0.075,0);scene.add(curb2);
        });

        // === BUILDINGS along both sides ===
        var bPal=[0x3a2828,0x2d2838,0x383028,0x282d38,0x443333,0x334444,0x383838,0x4a3030,0x30304a,0x3a3a2a,0x332233,0x2a3333];

        function mkBldg(x,z,w,d,h,col){
            var g=new THREE.Group();
            var box=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshLambertMaterial({color:col}));
            box.position.y=h/2;box.castShadow=true;box.receiveShadow=true;g.add(box);
            // Roof
            if(h>20&&Math.random()>0.3){
                var rh=1.5+Math.random()*3;
                var rb=new THREE.Mesh(new THREE.BoxGeometry(w*0.35,rh,d*0.35),new THREE.MeshLambertMaterial({color:0x222222}));rb.position.y=h+rh/2;g.add(rb);
            }
            // AC units
            if(Math.random()>0.4){
                for(var a=0;a<1+Math.floor(Math.random()*3);a++){
                    var ac=new THREE.Mesh(new THREE.BoxGeometry(1.2,0.8,1.2),new THREE.MeshLambertMaterial({color:0x555555}));
                    ac.position.set((Math.random()-0.5)*w*0.5,h+0.4,(Math.random()-0.5)*d*0.5);g.add(ac);
                }
            }
            // Windows
            var wGeo=new THREE.PlaneGeometry(1.4,2);
            var wSp=4.2;
            for(var fl=3.5;fl<h-2;fl+=wSp){
                for(var wx=-w/2+2;wx<w/2-1;wx+=wSp){
                    var lit=Math.random()>0.15;
                    var cool=Math.random()>0.7;
                    var wc=lit?(cool?0x88bbff:0xffdd88):0x181210;
                    var wm=new THREE.MeshBasicMaterial({color:wc});
                    // Street-facing side
                    var facing=(x>0)?-1:1;
                    var f=new THREE.Mesh(wGeo,wm);f.position.set(wx,fl,facing*d/2+facing*0.05);
                    if(facing<0)f.rotation.y=Math.PI;g.add(f);
                    // Back side
                    var b=new THREE.Mesh(wGeo,wm);b.position.set(wx,fl,-facing*d/2-facing*0.05);
                    if(facing>0)b.rotation.y=Math.PI;g.add(b);
                }
                for(var wz=-d/2+2;wz<d/2-1;wz+=wSp){
                    var lit=Math.random()>0.15;
                    var wc=lit?(Math.random()>0.7?0x88bbff:0xffdd88):0x181210;
                    var wm=new THREE.MeshBasicMaterial({color:wc});
                    var r=new THREE.Mesh(wGeo,wm);r.position.set(w/2+0.05,fl,wz);r.rotation.y=Math.PI/2;g.add(r);
                    var l=new THREE.Mesh(wGeo,wm);l.position.set(-w/2-0.05,fl,wz);l.rotation.y=-Math.PI/2;g.add(l);
                }
            }
            // Shop front glow (street-facing)
            if(h>10){
                var shopMat=new THREE.MeshBasicMaterial({color:0xffcc77,transparent:true,opacity:0.5});
                var facing=(x>0)?-1:1;
                var sg=new THREE.PlaneGeometry(w*0.7,2.5);
                var sf=new THREE.Mesh(sg,shopMat);
                sf.position.set(0,1.8,facing*d/2+facing*0.06);
                if(facing<0)sf.rotation.y=Math.PI;
                g.add(sf);
            }
            g.position.set(x,0,z);scene.add(g);
            colliders.push({minX:x-w/2-0.3,maxX:x+w/2+0.3,minZ:z-d/2-0.3,maxZ:z+d/2+0.3});
        }

        // Generate buildings along both sides of the road
        var buildingEdge=RW/2+SW+1;
        for(var side=-1;side<=1;side+=2){
            var bz=-800;
            while(bz<800){
                var bw=12+Math.random()*22;
                var bd=14+Math.random()*25;
                var bh=18+Math.random()*65;
                var gap=1+Math.random()*4; // small gap between buildings
                var bx=side*(buildingEdge+bw/2+Math.random()*5);
                mkBldg(bx,bz+bd/2,bw,bd,bh,bPal[Math.floor(Math.random()*bPal.length)]);
                bz+=bd+gap;
            }
        }

        // === STREET LIGHTS (visual only) ===
        function mkLamp(x,z,rotY){
            var g=new THREE.Group();
            var pm=new THREE.MeshLambertMaterial({color:0x444444});
            var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.13,5.5,5),pm);
            pole.position.y=2.75;g.add(pole);
            var arm=new THREE.Mesh(new THREE.BoxGeometry(2.5,0.08,0.08),pm);
            arm.position.set(1.25,5.5,0);g.add(arm);
            var hous=new THREE.Mesh(new THREE.BoxGeometry(1,0.25,0.4),new THREE.MeshLambertMaterial({color:0x333333}));
            hous.position.set(2.2,5.38,0);g.add(hous);
            var gp=new THREE.Mesh(new THREE.PlaneGeometry(0.8,0.3),new THREE.MeshBasicMaterial({color:0xffdd88,transparent:true,opacity:0.85,side:THREE.DoubleSide}));
            gp.position.set(2.2,5.22,0);gp.rotation.x=Math.PI/2;g.add(gp);
            var bulb=new THREE.Mesh(new THREE.SphereGeometry(0.2,6,6),new THREE.MeshBasicMaterial({color:0xffeeaa}));
            bulb.position.set(2.2,5.15,0);g.add(bulb);
            var pool=new THREE.Mesh(new THREE.CircleGeometry(3.5,8),new THREE.MeshBasicMaterial({color:0xffdd88,transparent:true,opacity:0.06,side:THREE.DoubleSide}));
            pool.rotation.x=-Math.PI/2;pool.position.set(2.2,0.03,0);g.add(pool);
            var cone=new THREE.Mesh(new THREE.ConeGeometry(2.5,5,8,1,true),new THREE.MeshBasicMaterial({color:0xffdd66,transparent:true,opacity:0.03,side:THREE.DoubleSide}));
            cone.position.set(2.2,2.7,0);g.add(cone);
            g.position.set(x,0,z);g.rotation.y=rotY||0;scene.add(g);
            colliders.push({minX:x-0.25,maxX:x+0.25,minZ:z-0.25,maxZ:z+0.25});
        }
        for(var lz=-800;lz<800;lz+=30){
            mkLamp(RW/2+1,lz,0);
            mkLamp(-RW/2-1,lz+15,Math.PI);
        }

        // === TREES (between sidewalk and buildings) ===
        function mkTree(x,z){
            var g=new THREE.Group();
            var trunk=new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.2,2.2,5),new THREE.MeshLambertMaterial({color:0x3d2817}));
            trunk.position.y=1.1;g.add(trunk);
            var lc=[0x1a3a1a,0x1f4420,0x163316][Math.floor(Math.random()*3)];
            var lm=new THREE.MeshLambertMaterial({color:lc});
            [{r:1.8,h:2.5,y:3.2},{r:1.3,h:2,y:4.8},{r:0.8,h:1.6,y:6}].forEach(function(s){
                var c=new THREE.Mesh(new THREE.ConeGeometry(s.r,s.h,6),lm);c.position.y=s.y;c.castShadow=true;g.add(c);
            });
            g.position.set(x,0,z);scene.add(g);
            colliders.push({minX:x-0.3,maxX:x+0.3,minZ:z-0.3,maxZ:z+0.3});
        }
        for(var tz=-780;tz<780;tz+=14+Math.random()*10){
            if(Math.random()>0.35){
                mkTree(RW/2+SW-0.5,tz);
                mkTree(-RW/2-SW+0.5,tz+7);
            }
        }

        // === PLAYER CAR ===
        function mkCar(bodyCol,cabCol){
            var car=new THREE.Group();
            var body=new THREE.Mesh(new THREE.BoxGeometry(2.2,0.7,4.6),new THREE.MeshLambertMaterial({color:bodyCol}));
            body.position.set(0,0.55,0);body.castShadow=true;car.add(body);
            var under=new THREE.Mesh(new THREE.BoxGeometry(2.3,0.18,4.7),new THREE.MeshLambertMaterial({color:0x111111}));
            under.position.set(0,0.24,0);car.add(under);
            var cab=new THREE.Mesh(new THREE.BoxGeometry(1.85,0.6,2.2),new THREE.MeshLambertMaterial({color:cabCol}));
            cab.position.set(0,1.05,-0.3);cab.castShadow=true;car.add(cab);
            var wsMat=new THREE.MeshBasicMaterial({color:0x88bbdd,transparent:true,opacity:0.55,side:THREE.DoubleSide});
            var ws=new THREE.Mesh(new THREE.PlaneGeometry(1.65,0.5),wsMat);ws.position.set(0,1.1,0.85);ws.rotation.x=Math.PI/6;car.add(ws);
            var rw=new THREE.Mesh(new THREE.PlaneGeometry(1.65,0.5),wsMat);rw.position.set(0,1.1,-1.45);rw.rotation.x=-Math.PI/6;car.add(rw);
            var hlM=new THREE.MeshBasicMaterial({color:0xffffee});
            var hl1=new THREE.Mesh(new THREE.CircleGeometry(0.16,8),hlM);hl1.position.set(-0.65,0.5,2.31);car.add(hl1);
            var hl2=new THREE.Mesh(new THREE.CircleGeometry(0.16,8),hlM);hl2.position.set(0.65,0.5,2.31);car.add(hl2);
            var beamMat=new THREE.MeshBasicMaterial({color:0xffeecc,transparent:true,opacity:0.1,side:THREE.DoubleSide});
            var beam=new THREE.Mesh(new THREE.PlaneGeometry(1.8,8),beamMat);beam.position.set(0,0.15,6.5);beam.rotation.x=-Math.PI/2;car.add(beam);
            var tlM=new THREE.MeshBasicMaterial({color:0xff2200});
            var tl1=new THREE.Mesh(new THREE.BoxGeometry(0.28,0.12,0.04),tlM);tl1.position.set(-0.72,0.5,-2.31);car.add(tl1);
            var tl2=new THREE.Mesh(new THREE.BoxGeometry(0.28,0.12,0.04),tlM);tl2.position.set(0.72,0.5,-2.31);car.add(tl2);
            var tgM=new THREE.MeshBasicMaterial({color:0xff3300,transparent:true,opacity:0.06,side:THREE.DoubleSide});
            var tgP=new THREE.Mesh(new THREE.PlaneGeometry(1.5,4),tgM);tgP.position.set(0,0.15,-4.5);tgP.rotation.x=-Math.PI/2;car.add(tgP);
            var wg=new THREE.CylinderGeometry(0.35,0.35,0.26,8);
            var wm=new THREE.MeshLambertMaterial({color:0x1a1a1a});
            car.wheels=[];
            [[-1.1,0.35,1.3],[1.1,0.35,1.3],[-1.1,0.35,-1.3],[1.1,0.35,-1.3]].forEach(function(p){
                var w=new THREE.Mesh(wg,wm);w.rotation.z=Math.PI/2;w.position.set(p[0],p[1],p[2]);
                car.add(w);car.wheels.push(w);
                var hub=new THREE.Mesh(new THREE.CylinderGeometry(0.18,0.18,0.27,5),new THREE.MeshLambertMaterial({color:0x888888}));
                hub.rotation.z=Math.PI/2;hub.position.set(p[0]+(p[0]>0?0.14:-0.14),p[1],p[2]);car.add(hub);
            });
            return car;
        }
        var playerCar=mkCar(0xcc2222,0xaa1818);
        playerCar.position.set(-laneW,0,-600); // start in right lane, far back
        scene.add(playerCar);

        // === NPC CARS (same direction + oncoming) ===
        var npcCol=[[0x2255aa,0x1a4488],[0xdddd33,0xbbbb22],[0x22aa44,0x188833],[0xeeeeee,0xcccccc],[0x222222,0x111111],[0xcc6600,0xaa5500],[0x8833aa,0x6622aa],[0x33aaaa,0x228888],[0xaa3355,0x882244],[0x4488cc,0x336699]];

        function spawnNPC(){
            var c=npcCol[Math.floor(Math.random()*npcCol.length)];
            var npc=mkCar(c[0],c[1]);
            // Same direction (right lanes) or oncoming (left lanes)
            var sameDir=Math.random()>0.35;
            if(sameDir){
                // Right side, going +Z (toward sunset)
                var lane=-laneW-(Math.random()>0.5?laneW:0);
                npc.position.set(lane,0,playerCar.position.z-100-Math.random()*600);
                npc.userData={speed:14+Math.random()*12,dir:1,axis:'z'};
                npc.rotation.y=0;
            } else {
                // Left side, going -Z (toward us)
                var lane=laneW+(Math.random()>0.5?laneW:0);
                npc.position.set(lane,0,playerCar.position.z+50+Math.random()*600);
                npc.userData={speed:14+Math.random()*16,dir:-1,axis:'z'};
                npc.rotation.y=Math.PI;
            }
            scene.add(npc);npcCars.push(npc);
        }
        for(var i=0;i<30;i++) spawnNPC();

        // === SNOW ===
        var snowOn=false,snowTimer=0,snowDur=0,nextSnow=15+Math.random()*30;
        var SN=2000;
        var sGeo=new THREE.BufferGeometry();
        var sPos=new Float32Array(SN*3);
        var sVel=[];
        for(var i=0;i<SN;i++){
            sPos[i*3]=(Math.random()-0.5)*80;sPos[i*3+1]=Math.random()*40;sPos[i*3+2]=(Math.random()-0.5)*80;
            sVel.push({x:(Math.random()-0.5)*1.5,y:-(1+Math.random()*2.5),z:(Math.random()-0.5)*1.5});
        }
        sGeo.setAttribute('position',new THREE.BufferAttribute(sPos,3));
        var sPts=new THREE.Points(sGeo,new THREE.PointsMaterial({color:0xffffff,size:0.2,transparent:true,opacity:0.7,depthWrite:false}));
        sPts.visible=false;scene.add(sPts);

        // === CONTROLS ===
        var cs={speed:0,maxSpd:90,accel:30,brake:45,fric:6,turnSpd:1.8,rot:0,wRot:0,colCD:0};
        var ks={f:false,b:false,l:false,r:false,brk:false};
        document.addEventListener('keydown',function(e){
            switch(e.key.toLowerCase()){case'w':case'arrowup':ks.f=true;break;case's':case'arrowdown':ks.b=true;break;case'a':case'arrowleft':ks.l=true;break;case'd':case'arrowright':ks.r=true;break;case' ':ks.brk=true;e.preventDefault();break;}
        });
        document.addEventListener('keyup',function(e){
            switch(e.key.toLowerCase()){case'w':case'arrowup':ks.f=false;break;case's':case'arrowdown':ks.b=false;break;case'a':case'arrowleft':ks.l=false;break;case'd':case'arrowright':ks.r=false;break;case' ':ks.brk=false;break;}
        });

        // === COLLISION ===
        var flashEl=document.getElementById('collision-flash');
        function checkCol(nx,nz){
            var hw=1.2,hd=2.5,sR=Math.sin(cs.rot),cR=Math.cos(cs.rot);
            var x0=nx-hw-Math.abs(sR)*hd,x1=nx+hw+Math.abs(sR)*hd;
            var z0=nz-hd-Math.abs(cR)*hw,z1=nz+hd+Math.abs(cR)*hw;
            for(var i=0;i<colliders.length;i++){var c=colliders[i];if(x1>c.minX&&x0<c.maxX&&z1>c.minZ&&z0<c.maxZ)return true;}
            for(var i=0;i<npcCars.length;i++){var dx=nx-npcCars[i].position.x,dz=nz-npcCars[i].position.z;if(dx*dx+dz*dz<12)return true;}
            return false;
        }
        function doCol(){
            flashEl.style.opacity=String(Math.min(1,Math.abs(cs.speed)*0.018));
            setTimeout(function(){flashEl.style.opacity='0';},150);
            cs.speed*=-0.3;cs.colCD=0.3;
        }

        // === INFINITE SCROLLING ===
        // When player moves far along Z, teleport everything back to keep coordinates manageable
        // and recycle buildings/lamps/trees that are far behind
        var worldOffset=0;

        // === MAIN LOOP ===
        var clock=new THREE.Clock();
        var spdEl=document.getElementById('speed'),gearEl=document.getElementById('gear');
        var npcEl=document.getElementById('npc-count'),snowPill=document.getElementById('snow-pill');
        // Camera: lower, closer, more cinematic
        var camOff=new THREE.Vector3(0,3.5,-9);
        var fc=0;
        camera.position.set(-laneW,3.5,-609);
        camera.lookAt(-laneW,1.5,-600);

        function animate(){
            requestAnimationFrame(animate);
            var dt=Math.min(clock.getDelta(),0.05);fc++;

            // Physics
            if(cs.colCD>0)cs.colCD-=dt;
            if(ks.f)cs.speed+=cs.accel*dt;else if(ks.b)cs.speed-=cs.accel*0.7*dt;
            else{if(cs.speed>0)cs.speed=Math.max(0,cs.speed-cs.fric*dt);else if(cs.speed<0)cs.speed=Math.min(0,cs.speed+cs.fric*dt);}
            if(ks.brk){if(cs.speed>0)cs.speed=Math.max(0,cs.speed-cs.brake*2*dt);else cs.speed=Math.min(0,cs.speed+cs.brake*2*dt);}
            cs.speed=Math.max(-cs.maxSpd/3,Math.min(cs.maxSpd,cs.speed));

            // Gentle steering (limited so you stay roughly on the road)
            if(Math.abs(cs.speed)>0.5){
                var tf=Math.min(1,Math.abs(cs.speed)/25);
                if(ks.l)cs.rot+=cs.turnSpd*tf*dt*Math.sign(cs.speed);
                if(ks.r)cs.rot-=cs.turnSpd*tf*dt*Math.sign(cs.speed);
            }
            // Auto-straighten slowly
            cs.rot*=0.97;
            // Clamp rotation so car can't turn fully sideways
            cs.rot=Math.max(-0.4,Math.min(0.4,cs.rot));

            var mx=Math.sin(cs.rot)*cs.speed*dt;
            var mz=Math.cos(cs.rot)*cs.speed*dt;
            var nx=playerCar.position.x+mx,nz=playerCar.position.z+mz;

            // Keep on road (soft boundary)
            nx=Math.max(-RW/2+1.5,Math.min(RW/2-1.5,nx));

            if(cs.colCD<=0){if(checkCol(nx,nz))doCol();else{playerCar.position.x=nx;playerCar.position.z=nz;}}
            else{playerCar.position.x=Math.max(-RW/2+1.5,Math.min(RW/2-1.5,playerCar.position.x+mx));playerCar.position.z+=mz;}
            playerCar.rotation.y=cs.rot;
            cs.wRot+=cs.speed*dt*0.5;
            playerCar.wheels.forEach(function(w){w.rotation.x=cs.wRot;});

            // Headlight
            var hlOff=new THREE.Vector3(0,1.5,5);
            hlOff.applyAxisAngle(new THREE.Vector3(0,1,0),cs.rot);
            headlight.position.set(playerCar.position.x+hlOff.x,playerCar.position.y+hlOff.y,playerCar.position.z+hlOff.z);

            // NPCs - recycle when too far behind or ahead
            npcCars.forEach(function(n){
                var d=n.userData;
                n.position.z+=d.dir*d.speed*dt;
                n.wheels.forEach(function(w){w.rotation.x+=d.speed*dt*0.5;});
                // Recycle if too far behind player
                if(n.position.z<playerCar.position.z-200){
                    // Respawn ahead
                    n.position.z=playerCar.position.z+150+Math.random()*400;
                    if(d.dir<0){
                        // Oncoming - left lanes
                        n.position.x=laneW+(Math.random()>0.5?laneW:0);
                    } else {
                        // Same dir - right lanes
                        n.position.x=-laneW-(Math.random()>0.5?laneW:0);
                    }
                }
                // Also recycle oncoming that went too far behind
                if(d.dir<0&&n.position.z>playerCar.position.z+600){
                    n.position.z=playerCar.position.z+150+Math.random()*300;
                    n.position.x=laneW+(Math.random()>0.5?laneW:0);
                }
            });

            // Snow
            snowTimer+=dt;
            if(!snowOn&&snowTimer>=nextSnow){snowOn=true;snowTimer=0;snowDur=12+Math.random()*22;sPts.visible=true;snowPill.style.display='flex';}
            if(snowOn){
                if(snowTimer>=snowDur){snowOn=false;snowTimer=0;nextSnow=18+Math.random()*35;sPts.visible=false;snowPill.style.display='none';}
                var pa=sGeo.getAttribute('position');
                for(var i=0;i<SN;i++){
                    pa.array[i*3]+=sVel[i].x*dt;pa.array[i*3+1]+=sVel[i].y*dt;pa.array[i*3+2]+=sVel[i].z*dt;
                    if(pa.array[i*3+1]<0){
                        pa.array[i*3]=playerCar.position.x+(Math.random()-0.5)*60;
                        pa.array[i*3+1]=20+Math.random()*20;
                        pa.array[i*3+2]=playerCar.position.z+(Math.random()-0.5)*60;
                    }
                }
                pa.needsUpdate=true;
            }

            // HUD
            var dSpd=Math.abs(Math.round(cs.speed*3.6));
            spdEl.textContent=dSpd;
            if(cs.speed<-0.5)gearEl.textContent='R';else if(Math.abs(cs.speed)<0.5)gearEl.textContent='P';
            else if(dSpd<40)gearEl.textContent='1';else if(dSpd<80)gearEl.textContent='2';else if(dSpd<150)gearEl.textContent='3';else gearEl.textContent='4';
            if(fc%20===0)npcEl.textContent=npcCars.length+' cars';

            // Camera - low cinematic follow
            var ideal=camOff.clone().applyAxisAngle(new THREE.Vector3(0,1,0),cs.rot);
            ideal.add(playerCar.position);
            if(fc<10){camera.position.copy(ideal);}
            else{camera.position.lerp(ideal,1-Math.pow(0.02,dt));}
            var look=playerCar.position.clone();look.y+=1.2;look.z+=4;
            camera.lookAt(look);

            // Sun/sky follow player Z
            sunLight.position.set(playerCar.position.x+20,40,playerCar.position.z+150);
            sunLight.target.position.set(playerCar.position.x,0,playerCar.position.z);
            skyMesh.position.set(playerCar.position.x,0,playerCar.position.z);
            sunG1.position.set(playerCar.position.x,22,playerCar.position.z+370);
            sunG2.position.set(playerCar.position.x,22,playerCar.position.z+370);
            haze.position.set(playerCar.position.x,8,playerCar.position.z+370);
            // Move ground to follow
            gnd.position.set(playerCar.position.x,0,playerCar.position.z);

            renderer.render(scene,camera);
        }
        animate();
        window.addEventListener('resize',function(){camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);});
        document.body.tabIndex=0;document.body.focus();
        window.executeJS=function(code){try{eval(code);}catch(e){console.error(e);}};
    })();
    </script>
</body>
</html>
'''

game_view.setHtml(html_content)
layout.addWidget(game_view)

game_proxy = graphics_scene.addWidget(game_container)
view = graphics_scene.views()[0]
viewport_rect = view.viewport().rect()
scene_rect = view.mapToScene(viewport_rect).boundingRect()
center_x = scene_rect.center().x() - 600
center_y = scene_rect.center().y() - 400
game_proxy.setPos(center_x, center_y)
game_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
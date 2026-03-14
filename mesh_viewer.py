#!/usr/bin/env python3
"""
MeshViewer - Python wrapper for Three.js 3D mesh viewer
Provides standalone HTML viewers with localStorage-based communication.

Usage:
    viewer = MeshViewerComponent(mesh_path, title="NEAR Mesh")
    html_path = viewer.generate_viewer_html_file("near_points")
    # Open html_path in browser; user picks points, clicks Save & Close
    # Points are stored in localStorage under the key "near_points"
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Install with: pip install trimesh")


class MeshViewerComponent:
    """
    Web-based 3D mesh viewer using Three.js
    Supports large meshes with automatic decimation and interactive point selection.
    Generates standalone HTML files that communicate via localStorage.
    """
    
    def __init__(self, mesh_path: str, max_points: int = 100000, title: str = None):
        """
        Initialize mesh viewer
        
        Args:
            mesh_path: Path to mesh file (PLY, OBJ, STL, etc.)
            max_points: Maximum number of vertices to display (decimation if exceeded)
            title: Display title for the mesh
        """
        self.mesh_path = str(mesh_path)
        self.max_points = max_points
        self.title = title or Path(mesh_path).stem
        self.selected_points = []
        
        # Loaded mesh data
        self.vertices = None
        self.colors = None
        self.faces = None
        self.mesh_obj = None
        
    def load_mesh(self) -> Dict[str, Any]:
        """
        Load mesh from file and prepare data for Three.js
        
        Returns:
            Dictionary with vertices, colors, faces for Three.js viewer
        """
        print(f"[MeshViewer] Loading mesh from: {self.mesh_path}")
        
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for mesh loading")
        
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
        
        # Load with trimesh
        try:
            mesh = trimesh.load(self.mesh_path, force='mesh', process=False)
            print(f"[MeshViewer] Loaded as mesh: {type(mesh)}")
        except Exception as e:
            print(f"[MeshViewer] Mesh load failed: {e}, trying point cloud...")
            try:
                mesh = trimesh.load(self.mesh_path, force='pointcloud', process=False)
                print(f"[MeshViewer] Loaded as point cloud: {type(mesh)}")
            except Exception as e2:
                print(f"[MeshViewer] Both mesh and point cloud load failed!")
                raise
        
        # Handle Scene (multiple geometries)
        if isinstance(mesh, trimesh.Scene):
            print(f"[MeshViewer] Handling Scene with {len(mesh.geometry)} geometries")
            meshes = []
            for name, geom in mesh.geometry.items():
                if isinstance(geom, (trimesh.Trimesh, trimesh.PointCloud)):
                    meshes.append(geom)
            
            if len(meshes) == 0:
                raise ValueError("No valid geometries found in scene")
            
            if isinstance(meshes[0], trimesh.Trimesh):
                mesh = trimesh.util.concatenate(meshes)
            else:
                all_vertices = []
                all_colors = []
                for m in meshes:
                    all_vertices.append(np.array(m.vertices))
                    if hasattr(m, 'colors') and m.colors is not None:
                        all_colors.append(np.array(m.colors)[:, :3])
                
                vertices = np.vstack(all_vertices)
                colors = np.vstack(all_colors) if all_colors else None
                mesh = trimesh.PointCloud(vertices=vertices, colors=colors)
        
        self.mesh_obj = mesh
        
        # Extract vertices
        vertices = np.array(mesh.vertices)
        print(f"[MeshViewer] Extracted {len(vertices)} vertices")
        
        # Extract colors if available
        colors = None
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            if hasattr(mesh.visual, 'vertex_colors'):
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) > 0:
                    colors = np.array(vc)[:, :3] / 255.0
        
        # Extract faces if mesh (not point cloud)
        faces = None
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
            faces = np.array(mesh.faces)
            print(f"[MeshViewer] Extracted {len(faces)} faces")
        
        # Decimate if too many vertices
        if len(vertices) > self.max_points:
            print(f"[MeshViewer] Decimating mesh: {len(vertices)} -> {self.max_points} vertices")
            
            if isinstance(mesh, trimesh.Trimesh):
                try:
                    decimated = mesh.simplify_quadric_decimation(self.max_points)
                    vertices = np.array(decimated.vertices)
                    faces = np.array(decimated.faces)
                    print(f"[MeshViewer] Quadric decimation: {len(vertices)} verts, {len(faces)} faces")
                    
                    if hasattr(decimated, 'visual') and decimated.visual is not None:
                        if hasattr(decimated.visual, 'vertex_colors'):
                            vc = decimated.visual.vertex_colors
                            if vc is not None:
                                colors = np.array(vc)[:, :3] / 255.0
                except Exception as e:
                    print(f"[MeshViewer] Quadric decimation failed: {e}, using sampling")
                    ratio = self.max_points / len(vertices)
                    step = max(1, int(1.0 / ratio))
                    
                    sampled_indices = np.arange(0, len(vertices), step)[:self.max_points]
                    vertices = vertices[sampled_indices]
                    
                    if colors is not None and len(colors) >= len(sampled_indices):
                        colors = colors[sampled_indices]
                    
                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}
                    
                    new_faces = []
                    if faces is not None:
                        for face in faces:
                            if all(v in old_to_new for v in face):
                                new_faces.append([old_to_new[v] for v in face])
                    
                    if new_faces:
                        faces = np.array(new_faces)
                    else:
                        faces = None
            else:
                indices = np.random.choice(len(vertices), self.max_points, replace=False)
                vertices = vertices[indices]
                if colors is not None:
                    colors = colors[indices]
        
        self.vertices = vertices
        self.colors = colors
        self.faces = faces
        
        print(f"[MeshViewer] Final: {len(vertices)} verts, {len(faces) if faces is not None else 0} faces")
        
        mesh_data = {
            'vertices': vertices.tolist(),
            'colors': None,  # Omit colors to reduce HTML file size
            'faces': faces.tolist() if faces is not None else None,
            'title': self.title
        }
        
        return mesh_data

    def generate_viewer_html_file(self, storage_key: str, output_dir: str = None) -> str:
        """
        Generate a standalone HTML file for the mesh viewer.
        
        The viewer includes:
        - Three.js scene with the mesh
        - Point picking via raycasting (click to select)
        - Undo / Clear buttons
        - "Save & Close" button that writes points to localStorage and closes the tab
        
        Args:
            storage_key: localStorage key to store points under (e.g. "near_points")
            output_dir: Directory to save the HTML file. Defaults to pipeline/static/
            
        Returns:
            Absolute path to the generated HTML file
        """
        mesh_data = self.load_mesh()
        
        if output_dir is None:
            output_dir = str(Path(__file__).parent / "static")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"viewer_{storage_key}.html")
        
        mesh_json = json.dumps(mesh_data)
        
        html = self._build_viewer_html(storage_key, mesh_json)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"[MeshViewer] Viewer HTML saved to: {output_path}")
        return output_path

    def _build_viewer_html(self, storage_key: str, mesh_json: str) -> str:
        """Build the complete standalone HTML string for the viewer."""
        
        title = self.title
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Point Picker</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{ width: 100%; height: 100%; overflow: hidden; background: #1a1a2e; font-family: Arial, sans-serif; }}
        #canvas-container {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; }}

        #info-panel {{
            position: absolute; top: 12px; left: 12px;
            background: rgba(0,0,0,0.85); padding: 14px 18px; border-radius: 8px;
            color: #fff; font-size: 13px; z-index: 100; min-width: 220px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #info-panel h2 {{ margin: 0 0 10px 0; font-size: 16px; color: #4fc3f7; }}
        .info-row {{ margin: 4px 0; display: flex; justify-content: space-between; }}
        .info-label {{ color: #aaa; }}
        .info-value {{ color: #fff; font-weight: bold; }}

        #controls {{
            position: absolute; top: 12px; right: 12px; z-index: 100;
            display: flex; flex-direction: column; gap: 6px;
        }}
        .ctrl-btn {{
            padding: 10px 16px; border: none; border-radius: 6px;
            font-weight: 700; cursor: pointer; font-size: 12px;
            text-transform: uppercase; letter-spacing: 0.5px;
            transition: transform 0.1s, box-shadow 0.15s;
        }}
        .ctrl-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        .ctrl-btn:active {{ transform: translateY(0); }}
        .btn-pick   {{ background: #4fc3f7; color: #000; }}
        .btn-pick.active {{ background: #f44336; color: #fff; }}
        .btn-undo   {{ background: #ff9800; color: #000; }}
        .btn-clear  {{ background: #f44336; color: #fff; }}
        .btn-fit    {{ background: #66bb6a; color: #000; }}
        .btn-save   {{
            background: linear-gradient(135deg, #00e676, #00c853);
            color: #000; font-size: 14px; padding: 14px 20px;
            margin-top: 10px; border: 2px solid #00e676;
        }}
        .btn-save:hover {{ box-shadow: 0 6px 20px rgba(0,230,118,0.4); }}

        #points-list {{
            position: absolute; bottom: 12px; right: 12px;
            background: rgba(0,0,0,0.85); padding: 12px; border-radius: 8px;
            font-family: 'Courier New', monospace; font-size: 11px;
            color: #fff; max-height: 220px; overflow-y: auto; z-index: 100;
            min-width: 240px; border: 1px solid rgba(255,255,255,0.1);
        }}
        #points-list h3 {{ margin: 0 0 8px 0; color: #4fc3f7; font-size: 13px; }}
        .point-item {{
            background: rgba(255,255,255,0.08); padding: 5px 8px;
            margin: 4px 0; border-radius: 4px; border-left: 3px solid #4fc3f7;
        }}

        #instructions {{
            position: absolute; bottom: 12px; left: 12px;
            background: rgba(0,0,0,0.85); padding: 12px 16px; border-radius: 8px;
            color: #ccc; font-size: 12px; z-index: 100; max-width: 300px;
            border: 1px solid rgba(255,255,255,0.1); line-height: 1.6;
        }}
        #instructions strong {{ color: #4fc3f7; }}

        #crosshair {{
            position: absolute; top: 50%; left: 50%;
            width: 20px; height: 20px;
            transform: translate(-50%, -50%);
            pointer-events: none; z-index: 50; display: none;
        }}
        #crosshair::before, #crosshair::after {{
            content: ''; position: absolute; background: rgba(255,255,255,0.6);
        }}
        #crosshair::before {{ width: 1px; height: 100%; left: 50%; }}
        #crosshair::after  {{ height: 1px; width: 100%; top: 50%; }}
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    <div id="crosshair"></div>

    <div id="info-panel">
        <h2>{title}</h2>
        <div class="info-row"><span class="info-label">Vertices:</span> <span class="info-value" id="vert-count">0</span></div>
        <div class="info-row"><span class="info-label">Selected:</span> <span class="info-value" id="pt-count">0 / 4</span></div>
        <div class="info-row"><span class="info-label">Mode:</span>     <span class="info-value" id="mode-lbl">View</span></div>
    </div>

    <div id="controls">
        <button class="ctrl-btn btn-pick" id="btn-pick" onclick="togglePick()">Enable Picking</button>
        <button class="ctrl-btn btn-undo"  onclick="undoPoint()">Undo Last</button>
        <button class="ctrl-btn btn-clear" onclick="clearPoints()">Clear All</button>
        <button class="ctrl-btn btn-fit"   onclick="fitView()">Fit View</button>
        <button class="ctrl-btn btn-save"  onclick="saveAndClose()">💾  Save &amp; Close</button>
    </div>

    <div id="points-list">
        <h3>Selected Points</h3>
        <div id="pt-list-items"><em style="color:#888;">Click mesh to select points</em></div>
    </div>

    <div id="instructions">
        <strong>Instructions:</strong><br>
        1. Click <strong>Enable Picking</strong><br>
        2. Click on the mesh to select <strong>4 landmarks</strong>:<br>
        &nbsp;&nbsp;① Head &nbsp; ② L. Shoulder<br>
        &nbsp;&nbsp;③ R. Shoulder &nbsp; ④ Pelvis<br>
        3. Click <strong>Save &amp; Close</strong>
    </div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    // ── State ──────────────────────────────────────────────
    const STORAGE_KEY = "{storage_key}";
    const meshData    = {mesh_json};

    let scene, camera, renderer, controls;
    let meshGroup, pickingEnabled = false;
    const selectedPoints = [];
    const markerGroup    = new THREE.Group();
    const raycaster      = new THREE.Raycaster();
    const mouse          = new THREE.Vector2();
    let meshObject       = null;   // the object we raycast against

    // ── Init ───────────────────────────────────────────────
    function init() {{
        const container = document.getElementById('canvas-container');

        // Renderer
        renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x1a1a2e);
        container.appendChild(renderer.domElement);

        // Scene
        scene = new THREE.Scene();
        
        // Better Lighting Setup
        // Ambient + Hemisphere
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
        hemiLight.position.set(0, 20, 0);
        scene.add(hemiLight);

        // Main Directional Light (Key Light)
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 7);
        scene.add(dirLight);

        // Rim Light (Back Light for edge definition)
        const rimLight = new THREE.DirectionalLight(0xffffff, 0.4);
        rimLight.position.set(-5, 5, -5);
        scene.add(rimLight);

        scene.add(markerGroup);

        // Camera
        camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1000);

        // Controls
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.12;

        // Load mesh
        loadMesh(meshData);

        // Events
        window.addEventListener('resize', onResize);
        renderer.domElement.addEventListener('click', onCanvasClick);

        animate();
    }}

    // ── Load mesh data ────────────────────────────────────
    function loadMesh(data) {{
        const verts = data.vertices;
        const faces = data.faces;
        document.getElementById('vert-count').textContent = verts.length.toLocaleString();

        // Build BufferGeometry
        const geometry = new THREE.BufferGeometry();
        const posArr = new Float32Array(verts.length * 3);
        for (let i = 0; i < verts.length; i++) {{
            posArr[i*3]   = verts[i][0];
            posArr[i*3+1] = verts[i][1];
            posArr[i*3+2] = verts[i][2];
        }}
        geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3));

        // Use uniform color (light gray) instead of height-based coloring
        const colArr = new Float32Array(verts.length * 3);
        const cVal = 0.8;
        for (let i = 0; i < verts.length; i++) {{
            colArr[i*3]   = cVal;
            colArr[i*3+1] = cVal;
            colArr[i*3+2] = cVal;
        }}
        geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3));

        if (faces && faces.length > 0) {{
            const idxArr = new Uint32Array(faces.length * 3);
            for (let i = 0; i < faces.length; i++) {{
                idxArr[i*3]   = faces[i][0];
                idxArr[i*3+1] = faces[i][1];
                idxArr[i*3+2] = faces[i][2];
            }}
            geometry.setIndex(new THREE.BufferAttribute(idxArr, 1));
            geometry.computeVertexNormals();

            // Use Standard Material for PBR shading
            const material = new THREE.MeshStandardMaterial({{
                vertexColors: true,
                side: THREE.DoubleSide,
                flatShading: false,
                roughness: 0.6,
                metalness: 0.1,
            }});
            meshObject = new THREE.Mesh(geometry, material);

        }} else {{
            // Point cloud
            const material = new THREE.PointsMaterial({{
                size: 0.003,
                vertexColors: true,
                sizeAttenuation: true
            }});
            meshObject = new THREE.Points(geometry, material);
        }}

        scene.add(meshObject);
        fitView();
    }}

    // ── Camera fit ────────────────────────────────────────
    window.fitView = function() {{
        if (!meshObject) return;
        const box = new THREE.Box3().setFromObject(meshObject);
        const center = box.getCenter(new THREE.Vector3());
        const sz     = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(sz.x, sz.y, sz.z);
        const dist   = maxDim * 1.8;

        camera.position.set(center.x + dist * 0.5, center.y + dist * 0.3, center.z + dist);
        camera.lookAt(center);
        controls.target.copy(center);
        controls.update();
    }};

    // ── Picking ───────────────────────────────────────────
    window.togglePick = function() {{
        pickingEnabled = !pickingEnabled;
        const btn = document.getElementById('btn-pick');
        btn.textContent = pickingEnabled ? 'Disable Picking' : 'Enable Picking';
        btn.classList.toggle('active', pickingEnabled);
        document.getElementById('mode-lbl').textContent = pickingEnabled ? 'PICK' : 'View';
        document.getElementById('crosshair').style.display = pickingEnabled ? 'block' : 'none';
        controls.enabled = !pickingEnabled;
    }};

    function onCanvasClick(e) {{
        if (!pickingEnabled || !meshObject) return;

        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
        mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);

        let point = null;

        if (meshObject.isMesh) {{
            const hits = raycaster.intersectObject(meshObject, false);
            if (hits.length > 0) {{
                point = hits[0].point.clone();
                // Snap to nearest vertex for accuracy
                const posAttr = meshObject.geometry.getAttribute('position');
                let bestDist = Infinity, bestIdx = -1;
                const tmpV = new THREE.Vector3();
                // search vertices of the hit face first
                const face = hits[0].face;
                for (const vi of [face.a, face.b, face.c]) {{
                    tmpV.fromBufferAttribute(posAttr, vi);
                    const d = tmpV.distanceTo(point);
                    if (d < bestDist) {{ bestDist = d; bestIdx = vi; }}
                }}
                if (bestIdx >= 0) {{
                    point.fromBufferAttribute(posAttr, bestIdx);
                }}
            }}
        }} else {{
            // Point cloud — find closest point near ray
            raycaster.params.Points = {{ threshold: 0.02 }};
            const hits = raycaster.intersectObject(meshObject, false);
            if (hits.length > 0) {{
                const posAttr = meshObject.geometry.getAttribute('position');
                point = new THREE.Vector3().fromBufferAttribute(posAttr, hits[0].index);
            }}
        }}

        if (point) {{
            addPoint(point);
        }}
    }}

    function addPoint(pt) {{
        selectedPoints.push([pt.x, pt.y, pt.z]);
        const idx = selectedPoints.length;

        // Marker sphere
        const sphereGeo = new THREE.SphereGeometry(0.006, 16, 16);
        const colors = [0x00ff00, 0xffff00, 0xff00ff, 0x00ffff, 0xff8000, 0x8000ff];
        const col = colors[(idx - 1) % colors.length];
        const sphereMat = new THREE.MeshBasicMaterial({{ color: col }});
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.copy(pt);
        sphere.name = 'marker_' + idx;
        markerGroup.add(sphere);

        // Billboard label
        const canvas = document.createElement('canvas');
        canvas.width = 64; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.beginPath(); ctx.arc(32, 32, 28, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 36px Arial';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(idx), 32, 34);
        const tex = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({{ map: tex, depthTest: false }});
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(0.03, 0.03, 1);
        sprite.position.set(pt.x, pt.y + 0.015, pt.z);
        sprite.name = 'label_' + idx;
        markerGroup.add(sprite);

        updateUI();
    }}

    // ── Undo / Clear ──────────────────────────────────────
    window.undoPoint = function() {{
        if (selectedPoints.length === 0) return;
        selectedPoints.pop();
        rebuildMarkers();
        updateUI();
    }};

    window.clearPoints = function() {{
        selectedPoints.length = 0;
        rebuildMarkers();
        updateUI();
    }};

    function rebuildMarkers() {{
        while (markerGroup.children.length) {{
            const c = markerGroup.children[0];
            markerGroup.remove(c);
            if (c.geometry) c.geometry.dispose();
            if (c.material) {{
                if (c.material.map) c.material.map.dispose();
                c.material.dispose();
            }}
        }}
        const pts = [...selectedPoints];
        selectedPoints.length = 0;
        pts.forEach(p => addPoint(new THREE.Vector3(p[0], p[1], p[2])));
    }}

    // ── Save & Close ──────────────────────────────────────
    window.saveAndClose = function() {{
        if (selectedPoints.length < 1) {{
            alert('Please select at least 1 point before saving.');
            return;
        }}
        try {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(selectedPoints));
            console.log('[MeshViewer] Saved ' + selectedPoints.length + ' points to localStorage key: ' + STORAGE_KEY);
        }} catch (e) {{
            alert('Failed to save points: ' + e.message);
            return;
        }}
        window.close();
    }};

    // ── UI helpers ────────────────────────────────────────
    function updateUI() {{
        document.getElementById('pt-count').textContent = selectedPoints.length + ' / 4';
        const listEl = document.getElementById('pt-list-items');
        if (selectedPoints.length === 0) {{
            listEl.innerHTML = '<em style="color:#888;">Click mesh to select points</em>';
            return;
        }}
        const names = ['HEAD', 'L. SHOULDER', 'R. SHOULDER', 'PELVIS'];
        listEl.innerHTML = selectedPoints.map((p, i) => {{
            const lbl = names[i] || ('Point ' + (i+1));
            return '<div class="point-item">' +
                   '<strong>' + (i+1) + '. ' + lbl + '</strong><br>' +
                   'X: ' + p[0].toFixed(4) + '  Y: ' + p[1].toFixed(4) + '  Z: ' + p[2].toFixed(4) +
                   '</div>';
        }}).join('');
    }}

    // ── Resize / Animate ──────────────────────────────────
    function onResize() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }}

    function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }}

    // ── Start ─────────────────────────────────────────────
    init();
    </script>
</body>
</html>'''
        
        return html

    # ── Legacy convenience methods ──────────────────────────
    
    def add_selected_point(self, point: List[float]):
        """Add a selected point [x, y, z]"""
        if len(point) == 3:
            self.selected_points.append(point)
    
    def get_selected_points(self) -> List[List[float]]:
        """Get all selected points"""
        return self.selected_points.copy()
    
    def clear_selection(self):
        """Clear all selected points"""
        self.selected_points = []
    
    def remove_last_point(self):
        """Remove the last selected point"""
        if self.selected_points:
            self.selected_points.pop()

    def create_viewer_html(self, width: str = "100%", height: str = "100vh") -> str:
        """
        Create HTML string for embedding (legacy compatibility).
        For the new workflow, use generate_viewer_html_file() instead.
        """
        mesh_data = self.load_mesh()
        mesh_json = json.dumps(mesh_data)
        return self._build_viewer_html("__embedded__", mesh_json)

    @staticmethod
    def load_mesh_simple(mesh_path: str, max_points: int = 100000) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simple mesh loading function (replacement for Open3D version)
        
        Args:
            mesh_path: Path to mesh file
            max_points: Maximum points to load
            
        Returns:
            (vertices, colors) as numpy arrays
        """
        viewer = MeshViewerComponent(mesh_path, max_points)
        data = viewer.load_mesh()
        
        vertices = np.array(data['vertices'])
        colors = np.array(data['colors']) if data['colors'] else None
        
        return vertices, colors


def create_mesh_viewer(mesh_path: str, title: str = None, max_points: int = 100000) -> str:
    """
    Create HTML for a mesh viewer (for use in Gradio gr.HTML component)
    
    Args:
        mesh_path: Path to mesh file
        title: Display title
        max_points: Maximum vertices to display
        
    Returns:
        HTML string for embedding in Gradio
    """
    viewer = MeshViewerComponent(mesh_path, max_points, title)
    return viewer.create_viewer_html()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mesh_viewer.py <mesh_file> [storage_key]")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    storage_key = sys.argv[2] if len(sys.argv) > 2 else "test_points"
    
    viewer = MeshViewerComponent(mesh_path)
    html_path = viewer.generate_viewer_html_file(storage_key)
    
    print(f"Viewer saved to: {html_path}")
    print("Opening in browser...")
    
    import webbrowser
    webbrowser.open(f'file://{html_path}')

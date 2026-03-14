"""
slab_viewer_gradio_v2.py

Requirements:
    pip install trimesh numpy plotly gradio scipy
"""

import json
import numpy as np
import trimesh
import plotly.graph_objs as go
import gradio as gr
from typing import Optional, Tuple

# ---------- Utilities ----------

def load_mesh_vertices(filepath: str):
    """
    Load mesh/pointcloud from filepath and return Nx3 numpy array of vertices.
    """
    try:
        # force='mesh' attempts to load as mesh; if it's just points, trimesh handles it usually
        mesh = trimesh.load(filepath, force='mesh')
    except Exception as e:
        raise RuntimeError(f"Could not load mesh from {filepath}: {e}")

    if mesh is None:
        raise RuntimeError("trimesh returned None.")
    
    # trimesh can return a Scene; convert to mesh
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise RuntimeError("Empty scene.")
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        
    # If mesh has no vertices but has point cloud
    if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
        verts = np.asarray(mesh.vertices, dtype=float)
        return verts
    else:
        raise RuntimeError("No vertices found in uploaded file.")

def parse_plane_from_file(filepath: str) -> Optional[np.ndarray]:
    """
    Try to read file as JSON to find plane coefficients.
    Returns [a,b,c,d] or None.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # simple check if it looks like json
            if not content.strip().startswith('{'):
                return None
            
            j = json.loads(content)
            
            # CASE 0: Your specific format (nested plane_equation)
            if 'plane_equation' in j:
                pe = j['plane_equation']
                if all(key in pe for key in ['a', 'b', 'c', 'd']):
                    return np.array([pe['a'], pe['b'], pe['c'], pe['d']], dtype=float)

            # CASE 1: Standard arrays
            if 'plane' in j:
                p = np.array(j['plane'], dtype=float)
                if p.shape == (4,): return p
            if 'coefficients' in j:
                p = np.array(j['coefficients'], dtype=float)
                if p.shape == (4,): return p
            
            # CASE 2: Normal + Point
            if 'normal' in j and 'point' in j:
                n = np.array(j['normal'], dtype=float)
                pt = np.array(j['point'], dtype=float)
                n = n / np.linalg.norm(n)
                d = -float(n.dot(pt))
                return np.concatenate([n, [d]])

    except json.JSONDecodeError:
        return None 
    except Exception:
        return None
    return None

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        raise RuntimeError("Need at least 3 points to fit a plane.")
    centroid = points.mean(axis=0)
    A = points - centroid
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    normal = vh[-1, :]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid

def plane_from_coeffs(coeffs: np.ndarray) -> Tuple[np.ndarray, float]:
    a, b, c, d = coeffs
    n = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise RuntimeError("Zero normal in plane coefficients.")
    return n / norm, float(d) / norm

def rotation_matrix_align_y_to_vector(target_y: np.ndarray) -> np.ndarray:
    """
    Builds a rotation matrix where the new Y axis aligns with target_y.
    """
    y = np.array(target_y, dtype=float)
    y = y / np.linalg.norm(y)
    
    arb = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(arb, y)) > 0.95:
        arb = np.array([1.0, 0.0, 0.0])
        
    x = np.cross(arb, y)
    x = x / np.linalg.norm(x)
    z = np.cross(x, y) 
    
    R = np.vstack([x, y, z])
    return R

# ---------- Main processing ----------

def compute_slab_points(mesh_path: str, plane_path: str, plane_coeffs_text: str,
                        threshold_fraction: float = 0.1,
                        sample_mesh_frac: float = 0.02):
    
    # 1) Load mesh
    try:
        mesh_verts = load_mesh_vertices(mesh_path)
    except Exception as e:
        return None, f"Error loading mesh: {e}", None

    # 2) Determine ground plane
    plane_coeffs = None
    plane_point = None
    plane_normal = None

    # Option A: Text Input
    if plane_coeffs_text and plane_coeffs_text.strip():
        try:
            nums = [float(x) for x in plane_coeffs_text.replace(';',',').split(',') if x.strip()!='']
            if len(nums) == 4:
                plane_coeffs = np.array(nums, dtype=float)
                plane_normal, d = plane_from_coeffs(plane_coeffs)
                plane_point = -d * plane_normal
        except Exception as e:
            return None, f"Failed to parse text coefficients: {e}", None

    # Option B: File Input
    if plane_coeffs is None and plane_path is not None:
        is_json = plane_path.lower().endswith('.json')
        parsed = parse_plane_from_file(plane_path)
        
        if parsed is not None:
            plane_coeffs = parsed
            plane_normal, d = plane_from_coeffs(plane_coeffs)
            plane_point = -d * plane_normal
        elif is_json:
            return None, "Error: JSON file format not recognized (missing 'plane_equation', 'plane', or 'normal').", None
        else:
            try:
                plane_verts = load_mesh_vertices(plane_path)
                n, pt = fit_plane_svd(plane_verts)
                plane_normal = n
                plane_point = pt
            except Exception as e:
                return None, f"Error reading ground plane file: {e}", None

    if plane_normal is None:
        return None, "No valid ground plane provided.", None

    # 3) Alignment
    # NOTE: Your normal is [0, -1, 0]. If the mesh is upright, this points DOWN.
    # To view the mesh upright, we align the Y-axis to -normal (which would be UP [0,1,0]).
    # We check the sign: if normal.y is negative, we probably want -normal as the "up" vector.
    
    desired_y = plane_normal
    # Heuristic: If normal points predominantly down, flip it so the plot isn't upside down.
    if desired_y[1] < -0.9: 
        desired_y = -desired_y

    R = rotation_matrix_align_y_to_vector(desired_y)
    local = (R @ (mesh_verts.T - plane_point.reshape(3,1))).T

    # 4) Compute H
    # Filter for slightly positive Y to avoid noise below ground
    valid_heights = local[:, 1][local[:, 1] >= -0.01] 
    if len(valid_heights) == 0: 
        H = 0.0
    else: 
        H = float(np.max(valid_heights))

    # 5) Slab calculation
    slab_limit = threshold_fraction * H
    in_slab_mask = (local[:, 1] >= 0.0) & (local[:, 1] <= slab_limit)
    slab_points = mesh_verts[in_slab_mask]

    # 6) Visualization
    num_slab = slab_points.shape[0]
    total = mesh_verts.shape[0]
    
    sample_n = max(1, int(total * sample_mesh_frac))
    rng = np.random.default_rng(0)
    idx = rng.choice(total, size=min(sample_n, total), replace=False)
    sampled = mesh_verts[idx]

    slab_local = (R @ (slab_points.T - plane_point.reshape(3,1))).T
    sampled_local = (R @ (sampled.T - plane_point.reshape(3,1))).T

    trace_sampled = go.Scatter3d(
        x=sampled_local[:,0], y=sampled_local[:,1], z=sampled_local[:,2],
        mode='markers', marker=dict(size=2, color='gray', opacity=0.2),
        name='Context Mesh'
    )
    trace_slab = go.Scatter3d(
        x=slab_local[:,0], y=slab_local[:,1], z=slab_local[:,2],
        mode='markers', marker=dict(size=3, color='red'),
        name=f'Slab Points'
    )

    grid_size = max(H * 0.5, 1.0)
    gx = np.linspace(-grid_size, grid_size, 2)
    gz = np.linspace(-grid_size, grid_size, 2)
    gx, gz = np.meshgrid(gx, gz)
    gy = np.zeros_like(gx)
    plane_grid = go.Surface(x=gx, y=gy, z=gz, showscale=False, opacity=0.3, colorscale='Greens', name='Ground Plane')

    layout = go.Layout(
        scene=dict(xaxis_title='X', yaxis_title='Y (Height)', zaxis_title='Z', aspectmode='data'),
        height=700,
        title=f"Slab Analysis (H={H:.3f}, Threshold={slab_limit:.3f})"
    )

    fig = go.Figure(data=[trace_sampled, trace_slab, plane_grid], layout=layout)
    info_text = f"Total Vertices: {total}\nMax Height (H): {H:.4f}\nSlab Limit: {slab_limit:.4f}\nPoints in Slab: {num_slab}"

    return fig, info_text, str(num_slab)

# ---------- Gradio UI ----------

with gr.Blocks(title="Mesh Slab Viewer") as demo:
    gr.Markdown("## Mesh Slab Viewer")
    gr.Markdown("Upload a mesh and a ground plane JSON. The tool aligns the mesh based on the plane normal.")

    with gr.Row():
        mesh_file = gr.File(label="Input Mesh", file_count="single", type="filepath")
        plane_file = gr.File(label="Ground Plane (JSON or Mesh)", file_count="single", type="filepath")

    plane_txt = gr.Textbox(label="Or Plane Coeffs (a,b,c,d)", placeholder="0,1,0,0")
    
    with gr.Row():
        thresh = gr.Slider(0.01, 1.0, value=0.1, label="Slab Threshold (fraction of H)")
        sample_frac = gr.Slider(0.001, 0.1, value=0.02, step=0.001, label="Vis Sampling Rate")

    run_btn = gr.Button("Calculate Slab", variant="primary")

    with gr.Row():
        result_plot = gr.Plot(label="3D Visualization")
        with gr.Column():
            info = gr.Textbox(label="Statistics")
            count_out = gr.Textbox(label="Slab Point Count")

    def _wrap(mf, pf, txt, th, sf):
        if mf is None:
            return None, "Please upload a mesh.", "0"
        return compute_slab_points(mf, pf, txt, th, sf)

    run_btn.click(_wrap, inputs=[mesh_file, plane_file, plane_txt, thresh, sample_frac], outputs=[result_plot, info, count_out])

if __name__ == "__main__":
    demo.launch()
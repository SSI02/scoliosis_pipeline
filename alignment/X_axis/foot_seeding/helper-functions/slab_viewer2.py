"""
slab_viewer_gradio_v2_fixed.py

Requirements:
    pip install trimesh numpy plotly gradio scipy
"""

import json
import numpy as np
import trimesh
import plotly.graph_objs as go
import gradio as gr
from typing import Optional, Tuple
import io

# ---------- Utilities ----------

def load_mesh_vertices(filepath: str):
    """
    Load mesh/pointcloud from filepath and return Nx3 numpy array of vertices.
    """
    try:
        mesh = trimesh.load(filepath, force='mesh')
    except Exception as e:
        raise RuntimeError(f"Could not load mesh from {filepath}: {e}")

    if mesh is None:
        raise RuntimeError("trimesh returned None.")
    
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise RuntimeError("Empty scene.")
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        
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
            if not content.strip().startswith('{'):
                return None
            j = json.loads(content)
            # CASE 0: nested plane_equation
            if 'plane_equation' in j:
                pe = j['plane_equation']
                if all(key in pe for key in ['a', 'b', 'c', 'd']):
                    return np.array([pe['a'], pe['b'], pe['c'], pe['d']], dtype=float)
            # CASE 1: arrays
            if 'plane' in j:
                p = np.array(j['plane'], dtype=float)
                if p.shape == (4,): return p
            if 'coefficients' in j:
                p = np.array(j['coefficients'], dtype=float)
                if p.shape == (4,): return p
            # CASE 2: normal + point
            if 'normal' in j and 'point' in j:
                n = np.array(j['normal'], dtype=float)
                pt = np.array(j['point'], dtype=float)
                n = n / np.linalg.norm(n)
                d = -float(n.dot(pt))
                return np.concatenate([n, [d]])
    except Exception:
        return None
    return None

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit plane with SVD. Returns (unit_normal, centroid).
    Note: normal orientation sign is arbitrary.
    """
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
    Builds a rotation matrix where the new Y axis aligns with target_y (unit vector).
    Rows are [x', y', z'] so local = R @ (world - origin).
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

def export_points_to_ply(points: np.ndarray) -> bytes:
    """
    Simple PLY writer for point clouds (ASCII). Returns bytes.
    """
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    body = "\n".join(f"{p[0]} {p[1]} {p[2]}" for p in points)
    data = (header + body).encode('utf-8')
    return data

# ---------- Main processing ----------

def compute_slab_points(mesh_path: str, plane_path: str, plane_coeffs_text: str,
                        threshold_fraction: float = 0.1,
                        sample_mesh_frac: float = 0.02):
    """
    Uses signed distance to plane to select slab points.
    Returns: (plotly_fig, info_text, slab_count, slab_ply_bytes_or_None)
    """
    # 1) Load mesh vertices
    try:
        mesh_verts = load_mesh_vertices(mesh_path)
    except Exception as e:
        return None, f"Error loading mesh: {e}", "0", None

    total = mesh_verts.shape[0]

    # 2) Determine ground plane normal & a point on plane
    plane_normal = None
    plane_point = None
    plane_coeffs = None

    # Option A: text coefficients
    if plane_coeffs_text and plane_coeffs_text.strip():
        try:
            nums = [float(x) for x in plane_coeffs_text.replace(';',',').split(',') if x.strip()!='']
            if len(nums) == 4:
                plane_coeffs = np.array(nums, dtype=float)
                plane_normal, d = plane_from_coeffs(plane_coeffs)
                plane_point = -d * plane_normal
        except Exception as e:
            return None, f"Failed to parse text coefficients: {e}", "0", None

    # Option B: file input
    if plane_normal is None and plane_path:
        parsed = parse_plane_from_file(plane_path)
        if parsed is not None:
            plane_coeffs = parsed
            plane_normal, d = plane_from_coeffs(plane_coeffs)
            plane_point = -d * plane_normal
        else:
            # treat plane_path as mesh/point cloud and fit plane
            try:
                plane_verts = load_mesh_vertices(plane_path)
                n, pt = fit_plane_svd(plane_verts)
                plane_normal = n
                plane_point = pt
            except Exception as e:
                return None, f"Error reading plane file: {e}", "0", None

    if plane_normal is None:
        return None, "No valid ground plane provided (file or text).", "0", None

    # 3) Diagnostics: compare plane_normal to mesh PCA normal (helps catch wrong plane)
    try:
        mesh_pca_n, _ = fit_plane_svd(mesh_verts)
        # angle between normals (use absolute so orientation sign doesn't matter)
        cosang = np.clip(abs(np.dot(mesh_pca_n, plane_normal)), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cosang)))
    except Exception:
        angle_deg = -1.0

    # 4) Signed distance to plane (robust for bent poses)
    # plane: n.x + d = 0  ->  distance(v) = n . v + d
    # if we have a plane_point p: d = -n.dot(p)
    d = -float(plane_normal.dot(plane_point))
    distances = (mesh_verts @ plane_normal) + d  # shape (N,)

    # Keep numerical epsilon tolerance
    eps = 1e-8
    distances = distances.astype(float)

    # 5) Compute H and slab limit
    # We consider only distances >= 0 (above or on plane). H = max(distance)
    positive_dist = distances[distances >= -1e-6]  # allow tiny negative noise
    if positive_dist.size == 0:
        H = 0.0
    else:
        H = float(np.max(positive_dist))

    slab_limit = threshold_fraction * H

    # 6) If H is tiny, warn and expand slab_limit to an absolute min (helps debugging)
    absolute_min_slab = 1e-4  # 0.1 mm
    if slab_limit < absolute_min_slab:
        expanded = max(slab_limit, absolute_min_slab)
        slab_note = f"(slab limit was tiny {slab_limit:.6f}, using min {expanded:.6f} for visualization)"
        slab_limit = expanded
    else:
        slab_note = ""

    # 7) Slab mask: points between plane (distance>=0) and distance<=slab_limit
    in_slab_mask = (distances >= 0.0 - eps) & (distances <= slab_limit + eps)
    slab_points = mesh_verts[in_slab_mask]
    num_slab = slab_points.shape[0]

    # 8) Build visualization in a local frame where Y = up (i.e., -ground_normal considered up)
    # We want local Y to show height above ground. Choose desired_y = -plane_normal (so Y points "up")
    desired_y = -plane_normal
    R = rotation_matrix_align_y_to_vector(desired_y)
    # translate so plane lies at y=0
    local_all = (R @ (mesh_verts.T - plane_point.reshape(3,1))).T
    local_slab = (R @ (slab_points.T - plane_point.reshape(3,1))).T

    # Sample context points for plotting
    sample_n = max(1, int(total * sample_mesh_frac))
    rng = np.random.default_rng(0)
    idx = rng.choice(total, size=min(sample_n, total), replace=False)
    sampled_local = local_all[idx]

    trace_context = go.Scatter3d(
        x=sampled_local[:,0], y=sampled_local[:,1], z=sampled_local[:,2],
        mode='markers', marker=dict(size=2, opacity=0.25), name='Context Mesh'
    )
    trace_slab = go.Scatter3d(
        x=local_slab[:,0], y=local_slab[:,1], z=local_slab[:,2],
        mode='markers', marker=dict(size=3, color='red'), name=f'Slab Points ({num_slab})'
    )

    # Ground grid (in local coords at y=0)
    grid_size = max(0.5, H * 0.5)
    gx = np.linspace(-grid_size, grid_size, 2)
    gz = np.linspace(-grid_size, grid_size, 2)
    gx, gz = np.meshgrid(gx, gz)
    gy = np.zeros_like(gx)
    plane_grid = go.Surface(x=gx, y=gy, z=gz, showscale=False, opacity=0.25, name='Ground Plane')

    layout = go.Layout(
        scene=dict(xaxis_title='X', yaxis_title='Y (height above ground)', zaxis_title='Z', aspectmode='data'),
        height=700,
        title=f"Slab Analysis (H={H:.4f}, slab_limit={slab_limit:.6f})"
    )

    fig = go.Figure(data=[trace_context, trace_slab, plane_grid], layout=layout)

    info_lines = [
        f"Total vertices: {total}",
        f"H (max distance above plane): {H:.6f}",
        f"Slab threshold fraction: {threshold_fraction:.6f}",
        f"Slab limit used (distance): {slab_limit:.6f} {slab_note}",
        f"Points in slab: {num_slab}",
    ]
    if angle_deg >= 0:
        info_lines.append(f"Angle between mesh PCA normal and plane normal: {angle_deg:.2f}° (large angle -> plane may be wrong)")

    info_text = "\n".join(info_lines)

    # 9) prepare download of slab as PLY if any points exist
    slab_bytes = None
    if num_slab > 0:
        try:
            slab_bytes = export_points_to_ply(slab_points)
        except Exception:
            slab_bytes = None

    # Return figure + info + count + slab bytes (as a downloadable artifact)
    return fig, info_text, str(num_slab), slab_bytes

# ---------- Gradio UI ----------

with gr.Blocks(title="Mesh Slab Viewer (signed-distance)") as demo:
    gr.Markdown("## Mesh Slab Viewer (signed-distance)\n"
                "Upload a mesh and a ground-plane JSON/mesh. The slab selection uses signed distance to the plane, "
                "which is robust for forward-bent postures.")

    with gr.Row():
        mesh_file = gr.File(label="Input Mesh (ply/obj/stl/...)", file_count="single", type="filepath")
        plane_file = gr.File(label="Ground Plane (JSON or plane mesh)", file_count="single", type="filepath")

    plane_txt = gr.Textbox(label="Or Plane Coeffs (a,b,c,d)", placeholder="0,1,0,0")
    with gr.Row():
        thresh = gr.Slider(0.001, 1.0, value=0.1, label="Slab Threshold (fraction of H)")
        sample_frac = gr.Slider(0.001, 0.1, value=0.02, step=0.001, label="Context sample fraction")

    run_btn = gr.Button("Calculate Slab", variant="primary")

    with gr.Row():
        result_plot = gr.Plot(label="3D Visualization")
        with gr.Column():
            info = gr.Textbox(label="Statistics", lines=6)
            count_out = gr.Textbox(label="Slab Point Count")
            download_btn = gr.File(label="Download slab PLY", visible=False)

    def _wrap(mf, pf, txt, th, sf):
        if mf is None:
            return None, "Please upload a mesh.", "0", None
        mesh_path = mf  # filepath string
        plane_path = pf if pf is not None else ""
        fig, info_text, count_str, slab_bytes = compute_slab_points(mesh_path, plane_path, txt, threshold_fraction=float(th), sample_mesh_frac=float(sf))
        # handle slab download: if slab_bytes exists, return a temp file via gr.File by writing to /tmp-like location
        if slab_bytes is not None:
            # create an in-memory file object for Gradio's file output
            # Gradio's File component expects a path - write to a temp file
            tmp_path = "/tmp/slab_points.ply"
            with open(tmp_path, "wb") as f:
                f.write(slab_bytes)
            return fig, info_text, count_str, tmp_path
        else:
            return fig, info_text, count_str, None

    run_btn.click(_wrap, inputs=[mesh_file, plane_file, plane_txt, thresh, sample_frac], outputs=[result_plot, info, count_out, download_btn])

if __name__ == "__main__":
    demo.launch()

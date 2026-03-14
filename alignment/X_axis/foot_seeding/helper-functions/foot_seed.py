"""
leg_seed_gui.py

Requirements:
    pip install trimesh numpy plotly gradio scipy scikit-image

Run:
    python leg_seed_gui.py
"""

import json
import numpy as np
import trimesh
import plotly.graph_objs as go
import gradio as gr
from typing import Optional, Tuple
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
import os

# ---------- Utilities ----------

def load_mesh_vertices(filepath: str) -> np.ndarray:
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
        return np.asarray(mesh.vertices, dtype=float), mesh
    raise RuntimeError("No vertices found in file.")

def parse_plane_from_file(filepath: str) -> Optional[np.ndarray]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip().startswith('{'):
            return None
        j = json.loads(content)
        if 'plane' in j:
            p = np.array(j['plane'], dtype=float)
            if p.shape == (4,): return p
        if 'coefficients' in j:
            p = np.array(j['coefficients'], dtype=float)
            if p.shape == (4,): return p
        if 'normal' in j and 'point' in j:
            n = np.array(j['normal'], dtype=float)
            pt = np.array(j['point'], dtype=float)
            n = n / np.linalg.norm(n)
            d = -float(n.dot(pt))
            return np.concatenate([n, [d]])
        if 'plane_equation' in j:
            pe = j['plane_equation']
            if all(k in pe for k in ('a','b','c','d')):
                return np.array([pe['a'], pe['b'], pe['c'], pe['d']], dtype=float)
    except Exception:
        return None
    return None

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

# ---------- Hole-based seeding ----------



def hole_based_seeds_from_slab(
    slab_points_local,         # Nx3 (local coords where y is up; x,z are plane axes)
    slab_points_world,         # Nx3 matching slab_points_local in world coords
    torso_center_local=None,   # optional (x,z) center to bias selection (e.g., body midline)
    grid_res=0.01,             # meters per pixel (0.01 = 10 mm)
    closing_radius_px=2,       # morphological closing radius in pixels
    min_hole_area_px=50,       # ignore tiny holes
    keep_k=2,                  # number of holes/seeds to return
    hole_proximity_sigma=0.5,  # how much to prefer holes near torso center (m)
    largest_to_second_ratio=3.0,  # NEW: threshold for "one big hole only"
    kdtree_mesh=None           # optional KDTree over full mesh verts to snap seeds
):
    """
    Returns:
        seeds_world : list of seed positions (Mx3)
        hole_list   : metadata for holes
    """

    # -----------------------------
    # 1) Project to 2D ground plane
    # -----------------------------
    uv = slab_points_local[:, [0, 2]]  # Nx2  -> using (x_local, z_local)
    if uv.shape[0] == 0:
        return [], []

    # -----------------------------
    # 2) Grid quantization
    # -----------------------------
    min_uv = uv.min(axis=0) - 1e-8
    uv_shift = uv - min_uv
    ij = np.floor(uv_shift / grid_res).astype(int)
    ix, iy = ij[:,0], ij[:,1]

    W = int(ix.max()) + 1
    H = int(iy.max()) + 1

    occ = np.zeros((H, W), dtype=np.uint8)
    occ[iy, ix] = 1

    # -----------------------------
    # 3) Morphological closing
    # -----------------------------
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, closing_radius_px)
    occ_closed = ndimage.binary_closing(occ, structure=struct)

    # -----------------------------
    # 4) Identify HOLES (invert occ)
    # -----------------------------
    empty = ~occ_closed
    labels = measure.label(empty, connectivity=2)
    props = measure.regionprops(labels)

    avg_y = float(slab_points_local[:, 1].mean())
    hole_list = []

    for p in props:
        if p.area < min_hole_area_px:
            continue

        coords = p.coords  # (row, col)
        is_border = np.any(coords[:,0] == 0) or np.any(coords[:,0] == H-1) or \
                    np.any(coords[:,1] == 0) or np.any(coords[:,1] == W-1)

        cy, cx = p.centroid
        u_cent = (cx + 0.5)*grid_res + min_uv[0]
        v_cent = (cy + 0.5)*grid_res + min_uv[1]

        centroid_local = np.array([u_cent, avg_y, v_cent])

        # Find K nearest neighbors for LSQ mapping to world space
        K = 15
        dists_sq = np.sum((slab_points_local[:, [0,2]] - [u_cent, v_cent])**2, axis=1)
        k_idx = np.argsort(dists_sq)[:K]

        src_subset = slab_points_local[k_idx]
        dst_subset = slab_points_world[k_idx]

        # Solve affine: dst = src*M + T
        A = np.hstack([src_subset, np.ones((len(src_subset),1))])
        B = dst_subset
        try:
            X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            centroid_world = np.hstack([centroid_local, 1]) @ X
        except Exception:
            centroid_world = dst_subset.mean(axis=0)

        hole_list.append({
            'label': p.label,
            'area_px': p.area,
            'is_border': is_border,
            'centroid_grid_uv': (u_cent, v_cent),
            'centroid_local': centroid_local,
            'centroid_world': centroid_world,
            'n_pts': int(p.area)
        })

    if len(hole_list) == 0:
        return [], []

    # -----------------------------
    # 5) Ranking holes
    # -----------------------------
    def hole_score(h):
        score = h['area_px']

        if h['is_border']:
            score *= 0.0001  # massively downweight border holes

        if torso_center_local is not None:
            tc = np.array(torso_center_local)[[0,2]]
            hole_uv = h['centroid_local'][[0,2]]
            dist = np.linalg.norm(hole_uv - tc)
            score *= np.exp(- (dist**2) / (2*(hole_proximity_sigma**2)))

        return score

    hole_list = sorted(hole_list, key=lambda h: hole_score(h), reverse=True)

    # -----------------------------
    # 6) NEW STRONG LOGIC
    #    If biggest hole >> second hole, return only the biggest one
    # -----------------------------
    if len(hole_list) == 1:
        chosen = [hole_list[0]]
    else:
        def effective_area(h):
            return h['area_px'] * (0.0001 if h['is_border'] else 1.0)

        a1 = effective_area(hole_list[0])
        a2 = effective_area(hole_list[1])

        if a2 <= 0:
            ratio = float('inf')
        else:
            ratio = a1 / (a2 + 1e-12)

        if ratio >= largest_to_second_ratio:
            # One giant merged hole → only one seed
            chosen = [hole_list[0]]
        else:
            chosen = hole_list[:keep_k]

    # -----------------------------
    # 7) Return chosen world centroids
    # -----------------------------
    seeds_world = [h['centroid_world'] for h in chosen]
    return seeds_world, chosen


# ---------- Main compute function ----------

def compute_feet_seeds_and_viz(far_mesh_path: str,
                               plane_path: str,
                               plane_coeffs_text: str,
                               slab_fraction: float = 0.02,
                               min_slab_mm: float = 5.0,
                               grid_res: float = 0.01,
                               closing_radius_px: int = 2,
                               min_hole_area_px: int = 50,
                               keep_k: int = 2,
                               sample_frac: float = 0.02,
                               save_path: str = ""):
    # load far mesh
    try:
        verts, mesh_obj = load_mesh_vertices(far_mesh_path)
    except Exception as e:
        return None, f"Error loading far mesh: {e}", ""

    total = verts.shape[0]

    # parse plane
    plane_normal = None
    plane_point = None
    plane_coeffs = None
    if plane_coeffs_text and plane_coeffs_text.strip():
        try:
            nums = [float(x) for x in plane_coeffs_text.replace(';',',').split(',') if x.strip()!='']
            if len(nums) == 4:
                plane_coeffs = np.array(nums, dtype=float)
                plane_normal, d = plane_from_coeffs(plane_coeffs)
                plane_point = -d * plane_normal
        except Exception as e:
            return None, f"Failed to parse plane text: {e}", ""

    if plane_normal is None and plane_path:
        parsed = parse_plane_from_file(plane_path)
        if parsed is not None:
            plane_coeffs = parsed
            plane_normal, d = plane_from_coeffs(plane_coeffs)
            plane_point = -d * plane_normal
        else:
            # try fitting plane to uploaded plane mesh/pointcloud
            try:
                plane_verts, _ = load_mesh_vertices(plane_path)
                n, pt = fit_plane_svd(plane_verts)
                plane_normal, plane_point = n, pt
            except Exception as e:
                return None, f"Error reading plane file: {e}", ""

    if plane_normal is None:
        return None, "No valid ground plane supplied.", ""

    # signed distances: plane eqn n.x + d = 0 => d = -n.dot(p)
    d_val = -float(plane_normal.dot(plane_point))
    distances = verts @ plane_normal + d_val  # signed distance (positive above plane)

    # compute H = max(distance >= 0)
    positive = distances[distances >= -1e-6]
    if positive.size == 0:
        H = 0.0
    else:
        H = float(np.max(positive))

    min_slab = float(min_slab_mm) / 1000.0
    slab_limit = max(slab_fraction * H, min_slab)

    # find slab points (allow tiny negative epsilon)
    slab_mask = (distances >= -1e-6) & (distances <= slab_limit + 1e-8)
    slab_points = verts[slab_mask]
    if slab_points.shape[0] == 0:
        return None, "No slab points found with given parameters.", ""

    # build local frame where Y is up = -plane_normal (so local Y shows height above plane)
    desired_y = -plane_normal
    R = rotation_matrix_align_y_to_vector(desired_y)
    # local coords (plane at y=0)
    local_all = (R @ (verts.T - plane_point.reshape(3,1))).T
    local_slab = (R @ (slab_points.T - plane_point.reshape(3,1))).T

    # torso center estimate (use mesh centroid projected to local x,z)
    torso_center_local = local_all.mean(axis=0)

    # hole-based seeds
    seeds_world, holes = hole_based_seeds_from_slab(local_slab, slab_points,
                                                   grid_res=grid_res,
                                                   closing_radius_px=closing_radius_px,
                                                   min_hole_area_px=min_hole_area_px,
                                                   keep_k=keep_k,
                                                   torso_center_local=torso_center_local)

    # visualization: sample some points from full mesh for context
    sample_n = max(1, int(total * sample_frac))
    rng = np.random.default_rng(1)
    idx = rng.choice(total, size=min(sample_n, total), replace=False)
    sampled_local = local_all[idx]

    trace_context = go.Scatter3d(
        x=sampled_local[:,0], y=sampled_local[:,1], z=sampled_local[:,2],
        mode='markers', marker=dict(size=2, opacity=0.18, color='gray'), name='Context mesh'
    )
    trace_slab = go.Scatter3d(
        x=local_slab[:,0], y=local_slab[:,1], z=local_slab[:,2],
        mode='markers', marker=dict(size=3, color='red'), name='Slab points'
    )

    seed_traces = []
    for i, s in enumerate(seeds_world):
        # convert seed to local for plotting
        s_local = (R @ (np.array(s).reshape(3,1) - plane_point.reshape(3,1))).reshape(3,)
        seed_traces.append(go.Scatter3d(
            x=[s_local[0]], y=[s_local[1]], z=[s_local[2]],
            mode='markers', marker=dict(size=6, color='lime', symbol='diamond'),
            name=f'seed_{i+1}'
        ))

    # ground plane grid
    grid_sz = max(0.5, H * 0.5)
    gx = np.linspace(-grid_sz, grid_sz, 2)
    gz = np.linspace(-grid_sz, grid_sz, 2)
    gx, gz = np.meshgrid(gx, gz)
    gy = np.zeros_like(gx)
    plane_grid = go.Surface(x=gx, y=gy, z=gz, showscale=False, opacity=0.25, name='Ground plane')

    data = [trace_context, trace_slab] + seed_traces + [plane_grid]
    layout = go.Layout(scene=dict(xaxis_title='X', yaxis_title='Y (height)', zaxis_title='Z', aspectmode='data'),
                       height=720, title=f"Feet seeds (slab_limit={slab_limit:.3f} m)")

    fig = go.Figure(data=data, layout=layout)

    # info text
    info_lines = [
        f"Total vertices: {total}",
        f"H (max distance above plane): {H:.4f} m",
        f"Slab limit used: {slab_limit:.4f} m (min {min_slab:.4f} m or fraction {slab_fraction})",
        f"Slab points: {slab_points.shape[0]}",
        f"Found seeds: {len(seeds_world)}"
    ]
    for i, h in enumerate(holes):
        info_lines.append(f"hole {i+1}: area_px={h['area_px']}, n_pts={h['n_pts']}, centroid_local={np.round(h['centroid_local'],3)}")

    info = "\n".join(info_lines)

    # return plot and info; also return seeds in json string for user
    seeds_json = json.dumps([s.tolist() for s in seeds_world])
    
    if save_path and save_path.strip():
        try:
            with open(save_path, 'w') as f:
                f.write(seeds_json)
            info += f"\n\nSaved seeds to: {save_path}"
        except Exception as e:
            info += f"\n\nError saving seeds: {e}"
            
    return fig, info, seeds_json

import tkinter as tk
from tkinter import filedialog

def browse_filename():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filename = filedialog.asksaveasfilename(defaultextension=".json", 
                                              filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        root.destroy()
        return filename
    except Exception as e:
        return f"Error opening dialog: {e}"

# ---------- Gradio UI ----------

with gr.Blocks(title="Leg Seeding from Slab (hole-based)") as demo:
    gr.Markdown("### Leg seeding from slab holes\nUpload FAR mesh and ground plane (JSON or mesh). "
                "This tool finds holes in the slab projection and returns candidate foot seed points (green).")

    with gr.Row():
        mesh_file = gr.File(label="Far mesh (ply/obj/...)", file_count="single", type="filepath")
        plane_file = gr.File(label="Ground plane (JSON or mesh)", file_count="single", type="filepath")

    plane_txt = gr.Textbox(label="Or plane coefficients (a,b,c,d)", placeholder="0,1,0,0")
    with gr.Row():
        slab_fraction = gr.Slider(0.001, 0.1, value=0.02, step=0.001, label="Slab fraction of H (or use min mm below)")
        min_slab_mm = gr.Slider(1, 30, value=5, step=1, label="Min slab thickness (mm)")
    with gr.Row():
        grid_res = gr.Slider(0.002, 0.05, value=0.01, step=0.001, label="Occupancy grid resolution (m)")
        closing_px = gr.Slider(0, 6, value=2, step=1, label="Morphological closing radius (px)")
    with gr.Row():
        min_area = gr.Slider(10, 1000, value=50, step=10, label="Min hole area (px)")
        keep_k = gr.Slider(1, 4, value=2, step=1, label="Number of seeds to return")
        
    with gr.Row():
        save_path_input = gr.Textbox(label="Save seeds to JSON file", placeholder="/path/to/seeds.json", scale=3)
        browse_btn = gr.Button("Browse...", scale=1)
        
    browse_btn.click(fn=browse_filename, outputs=save_path_input)

    run_btn = gr.Button("Compute seeds", variant="primary")

    with gr.Row():
        plot_out = gr.Plot(label="Visualization")
        with gr.Column():
            info_out = gr.Textbox(label="Info", lines=8)
            json_out = gr.Textbox(label="Seed coordinates (JSON)", lines=4)

    def _wrap(mesh_f, plane_f, plane_txt_val, slab_frac_val, min_slab_mm_val, grid_res_val, closing_px_val, min_area_val, keep_k_val, save_path_val):
        if mesh_f is None:
            return None, "Upload far mesh first.", ""
        mesh_path = mesh_f
        plane_path = plane_f if plane_f is not None else ""
        try:
            fig, info, seeds_json = compute_feet_seeds_and_viz(
                mesh_path, plane_path, plane_txt_val,
                slab_fraction=float(slab_frac_val),
                min_slab_mm=float(min_slab_mm_val),
                grid_res=float(grid_res_val),
                closing_radius_px=int(closing_px_val),
                min_hole_area_px=int(min_area_val),
                keep_k=int(keep_k_val),
                sample_frac=0.02,
                save_path=save_path_val
            )
            return fig, info, seeds_json
        except Exception as e:
            return None, f"Error during processing: {e}", ""

    run_btn.click(_wrap, inputs=[mesh_file, plane_file, plane_txt, slab_fraction, min_slab_mm, grid_res, closing_px, min_area, keep_k, save_path_input],
                  outputs=[plot_out, info_out, json_out])

if __name__ == "__main__":
    demo.launch()

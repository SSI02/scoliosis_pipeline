#!/usr/bin/env python3
"""
Calf ROI & Ground-Plane Inspector (Gradio GUI)
----------------------------------------------
- Upload a mesh / pointcloud (PLY/OBJ/STL/PCD/XYZ) and optionally a JSON file containing a ground-plane.
- Supported JSON variants (robust):
    * { "ground_plane": [a,b,c,d] }
    * { "ground_plane": { "a":..., "b":..., "c":..., "d":... } }
    * { "plane_equation": { "a":..., "b":..., "c":..., "d":... } }
    * { "plane_equation": { "normal":[nx,ny,nz], "d": ... } }
    * { "a":..., "b":..., "c":..., "d":... }
    * Root array [a,b,c,d]
    * Variants with numeric tokens embedded in strings (loose parsing)
    * Your pipeline's structure like the console snippet is supported.
- If no JSON / typed coefficients given, RANSAC estimates a ground plane.
- Move the plane along -Y with the slider (positive = move plane upward). The displayed fraction
  is (maxY - plane_y) / h where h = maxY - minY (i.e., fraction from feet if feet = maxY).
- Visualizes the ROI slice between plane_y - high_frac*h and plane_y - low_frac*h.
- Dependencies: open3d, numpy, plotly, gradio

Run:
    python calf_plane_inspector_full.py
"""

import os
import json
import re
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import gradio as gr

# ---------------- Robust JSON plane parsing helpers ----------------

_num_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def _to_float_loose(x):
    """Convert x to float tolerantly (accept numeric strings with extra chars)."""
    if x is None:
        raise ValueError("None cannot be converted to float")
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            m = _num_re.search(s)
            if m:
                return float(m.group(0))
            raise ValueError(f"No numeric token found in string: {s!r}")
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _to_float_loose(x[0])
    raise ValueError(f"Unsupported type for float conversion: {type(x)}")

def parse_plane_from_json(json_path: str) -> Optional[Tuple[float,float,float,float]]:
    """
    Parse many JSON formats to extract (a,b,c,d) of plane ax+by+cz+d=0.
    Returns (a,b,c,d) or None.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    def _from_list_like(x):
        if isinstance(x, (list, tuple)) and len(x) == 4:
            try:
                return (_to_float_loose(x[0]), _to_float_loose(x[1]), _to_float_loose(x[2]), _to_float_loose(x[3]))
            except Exception:
                return None
        return None

    # Root array [a,b,c,d]
    out = _from_list_like(data)
    if out is not None:
        return out

    # plane_equation variant
    if isinstance(data, dict) and "plane_equation" in data:
        pe = data["plane_equation"]
        out = _from_list_like(pe)
        if out is not None:
            return out
        if isinstance(pe, dict):
            # keys a,b,c,d
            if all(k in pe for k in ("a","b","c","d")):
                try:
                    return (_to_float_loose(pe["a"]), _to_float_loose(pe["b"]), _to_float_loose(pe["c"]), _to_float_loose(pe["d"]))
                except Exception:
                    pass
            # normal + d
            if "normal" in pe and ("d" in pe or "D" in pe):
                try:
                    norm = pe.get("normal")
                    dval = pe.get("d", pe.get("D"))
                    nx = _to_float_loose(norm[0]); ny = _to_float_loose(norm[1]); nz = _to_float_loose(norm[2])
                    dd = _to_float_loose(dval)
                    return (nx, ny, nz, dd)
                except Exception:
                    pass
            # attempt to collect numeric-like values inside
            vals = []
            for v in pe.values():
                try:
                    vals.append(_to_float_loose(v))
                except Exception:
                    continue
            if len(vals) >= 4:
                return tuple(vals[:4])

    # ground_plane variant
    if isinstance(data, dict) and "ground_plane" in data:
        gp = data["ground_plane"]
        out = _from_list_like(gp)
        if out is not None:
            return out
        if isinstance(gp, dict) and all(k in gp for k in ("a","b","c","d")):
            try:
                return (_to_float_loose(gp["a"]), _to_float_loose(gp["b"]), _to_float_loose(gp["c"]), _to_float_loose(gp["d"]))
            except Exception:
                pass
        if isinstance(gp, dict) and "normal" in gp and "d" in gp:
            try:
                nx = _to_float_loose(gp["normal"][0]); ny = _to_float_loose(gp["normal"][1]); nz = _to_float_loose(gp["normal"][2])
                dd = _to_float_loose(gp["d"])
                return (nx, ny, nz, dd)
            except Exception:
                pass

    # flat a,b,c,d at root
    if isinstance(data, dict) and all(k in data for k in ("a","b","c","d")):
        try:
            return (_to_float_loose(data["a"]), _to_float_loose(data["b"]), _to_float_loose(data["c"]), _to_float_loose(data["d"]))
        except Exception:
            pass

    # fallback: scan for numeric-like values
    def _scan_for_four(d):
        vals = []
        if isinstance(d, dict):
            for v in d.values():
                try:
                    vals.append(_to_float_loose(v))
                except Exception:
                    if isinstance(v, dict):
                        rec = _scan_for_four(v)
                        if rec:
                            return rec
                    continue
        return vals if len(vals) >= 4 else None

    scanned = _scan_for_four(data)
    if scanned:
        return tuple(scanned[:4])

    return None

# ----------------- Point cloud / plane utilities ----------------

def load_mesh_or_cloud(path: str, sample_points: int = 50000) -> np.ndarray:
    """Load mesh or point cloud and return Nx3 numpy array of points (samples mesh if needed)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".pts", ".txt"]:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError("Point cloud is empty or failed to load.")
        pts = np.asarray(pcd.points)
    else:
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            pcd = o3d.io.read_point_cloud(path)
            if pcd.is_empty():
                raise ValueError("Failed to load mesh or point cloud.")
            pts = np.asarray(pcd.points)
        else:
            sample_n = min(sample_points, max(len(mesh.vertices), sample_points))
            pcd = mesh.sample_points_uniformly(number_of_points=sample_n)
            pts = np.asarray(pcd.points)
    return pts

def estimate_ground_plane_from_pcd(points: np.ndarray, distance_threshold: float = 0.01, ransac_n: int = 3, num_iterations: int = 2000):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    return tuple(plane_model), len(inliers)

def plane_point_from_coeffs(plane: Tuple[float,float,float,float]) -> np.ndarray:
    a,b,c,d = plane
    if abs(b) > 1e-8:
        return np.array([0.0, -(a*0 + c*0 + d)/b, 0.0])
    if abs(c) > 1e-8:
        return np.array([0.0, 0.0, -(a*0 + b*0 + d)/c])
    if abs(a) > 1e-8:
        return np.array([-(b*0 + c*0 + d)/a, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])

def bbox_of_points(points: np.ndarray):
    min_b = points.min(axis=0)
    max_b = points.max(axis=0)
    return min_b, max_b

# ----------------- Plotting (Plotly) ----------------

def make_plotly_figure(points: np.ndarray,
                       plane_y: float,
                       low_frac: float,
                       high_frac: float,
                       show_plane_size_factor: float = 1.2):
    """
    Build Plotly figure with:
     - all points (gray),
     - ROI points (red) between plane_y - high_frac*h and plane_y - low_frac*h,
     - horizontal plane rectangle at Y = plane_y,
     - annotation showing fraction_from_feet = (maxY - plane_y)/h
    """
    pts = points.copy()
    min_b, max_b = bbox_of_points(pts)
    h = max_b[1] - min_b[1]
    if h <= 1e-9:
        raise ValueError("Point cloud has near-zero height in Y.")
    fraction_from_feet = (max_b[1] - plane_y) / h

    # Visual ROI: between plane_y - high_frac*h  and plane_y - low_frac*h
    low_y_vis = plane_y - high_frac * h
    high_y_vis = plane_y - low_frac * h
    mask_roi = (pts[:,1] >= low_y_vis) & (pts[:,1] <= high_y_vis)

    N = pts.shape[0]
    downsample_rate = 1
    if N > 120000:
        downsample_rate = int(np.ceil(N / 120000))
    plot_pts = pts[::downsample_rate]

    trace_all = go.Scatter3d(
        x=plot_pts[:,0], y=plot_pts[:,1], z=plot_pts[:,2],
        mode='markers',
        marker=dict(size=1, color='lightgray', opacity=0.6),
        name='All points'
    )
    roi_pts = pts[mask_roi]
    trace_roi = go.Scatter3d(
        x=roi_pts[:,0], y=roi_pts[:,1], z=roi_pts[:,2],
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.9),
        name='ROI slice'
    )

    xmid = 0.5 * (min_b[0] + max_b[0])
    zmid = 0.5 * (min_b[2] + max_b[2])
    x_extent = (max_b[0] - min_b[0]) * show_plane_size_factor
    z_extent = (max_b[2] - min_b[2]) * show_plane_size_factor
    xs = np.linspace(xmid - x_extent/2, xmid + x_extent/2, 2)
    zs = np.linspace(zmid - z_extent/2, zmid + z_extent/2, 2)
    xs2, zs2 = np.meshgrid(xs, zs)
    ys2 = np.ones_like(xs2) * plane_y

    plane_surf = go.Surface(
        x=xs2, y=ys2, z=zs2,
        showscale=False,
        opacity=0.35,
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        name='Ground plane'
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y (vertical)'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(y=0.9)
    )

    fig = go.Figure(data=[trace_all, trace_roi, plane_surf], layout=layout)
    fig.update_layout(annotations=[dict(
        text=f"plane_y={plane_y:.4f}  •  fraction_from_feet={fraction_from_feet:.3f} of h",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.01, y=0.98, align="left",
        font=dict(size=12)
    )])
    return fig, fraction_from_feet, low_y_vis, high_y_vis

# ----------------- Gradio app logic ----------------

def app_build_figure(pcd_file, plane_json_file, plane_coeffs, slider_offset, low_frac, high_frac):
    """
    Inputs:
      - pcd_file: uploaded mesh/pointcloud
      - plane_json_file: optional JSON file containing ground_plane or plane_equation
      - plane_coeffs: optional typed 'a b c d'
      - slider_offset: meters to move plane along -Y (positive -> move plane upward)
      - low_frac, high_frac: ROI fractions
    """
    if pcd_file is None:
        return None, "Upload a mesh/pointcloud first."
    path = pcd_file.name if hasattr(pcd_file, "name") else pcd_file
    try:
        pts = load_mesh_or_cloud(path)
    except Exception as e:
        return None, f"Failed to load mesh/pointcloud: {e}"

    min_b, max_b = bbox_of_points(pts)
    h = max_b[1] - min_b[1]
    if h <= 0:
        return None, "Invalid point cloud (zero or negative height)."

    plane = None
    base_plane_y = None

    # 1) JSON precedence
    if plane_json_file is not None:
        json_path = plane_json_file.name if hasattr(plane_json_file, "name") else plane_json_file
        parsed = parse_plane_from_json(json_path)
        if parsed is None:
            return None, ("Provided JSON could not be parsed as ground plane. "
                          "Expected formats: plane_equation/ground_plane/root a,b,c,d etc.")
        plane = parsed
        base_point = plane_point_from_coeffs(plane)
        base_plane_y = float(base_point[1])
    else:
        # 2) typed coefficients fallback
        if plane_coeffs and len(plane_coeffs.strip()) > 0:
            try:
                arr = [float(x) for x in plane_coeffs.replace(',', ' ').split()]
                if len(arr) != 4:
                    return None, "Typed plane must have 4 numbers: a b c d"
                plane = tuple(arr)
                base_point = plane_point_from_coeffs(plane)
                base_plane_y = float(base_point[1])
            except Exception as e:
                return None, f"Error parsing typed plane coefficients: {e}"
        else:
            # 3) auto-estimate via RANSAC
            try:
                plane, inliers = estimate_ground_plane_from_pcd(pts, distance_threshold=max(0.005, 0.005*h), num_iterations=2000)
                base_point = plane_point_from_coeffs(plane)
                base_plane_y = float(base_point[1])
            except Exception:
                # fallback: plane at maxY (feet)
                base_plane_y = float(max_b[1])
                plane = (0.0, 1.0, 0.0, -base_plane_y)

    # slider_offset moves plane along -Y: new_plane_y = base_plane_y - slider_offset
    plane_y = float(base_plane_y) - float(slider_offset)

    try:
        fig, fraction, low_y_vis, high_y_vis = make_plotly_figure(pts, plane_y, float(low_frac), float(high_frac))
    except Exception as e:
        return None, f"Visualization failed: {e}"

    info = (f"h = {h:.4f} m  |  minY={min_b[1]:.4f}  maxY={max_b[1]:.4f}\n"
            f"Plane Y = {plane_y:.4f}  |  fraction_from_feet = {fraction:.4f}\n"
            f"ROI visual Y range: [{low_y_vis:.4f}, {high_y_vis:.4f}]  (from plane)")

    return fig, info

# ----------------- Gradio UI ----------------

with gr.Blocks(title="Calf ROI & Ground Plane Inspector (full)") as demo:
    gr.Markdown("## Calf ROI & Ground-Plane inspector\n"
                "Upload mesh/pointcloud, optionally upload a JSON with ground-plane (takes precedence), "
                "or type coefficients. Move the slider to move the plane along **-Y** (positive -> move plane upward).")
    with gr.Row():
        with gr.Column(scale=2):
            pcd_input = gr.File(label="Upload mesh / pointcloud (PLY/OBJ/STL/PCD)", file_count="single")
            plane_json = gr.File(label="Optional JSON with ground plane (takes precedence)", file_count="single")
            plane_text = gr.Textbox(label="Optional typed ground plane coefficients (a b c d)", placeholder="e.g. 0 1 0 -1.2")
            low_frac = gr.Number(value=0.05, label="ROI low fraction (from plane, e.g. 0.05)", precision=3)
            high_frac = gr.Number(value=0.25, label="ROI high fraction (from plane, e.g. 0.25)", precision=3)
            slider = gr.Slider(minimum=-0.5, maximum=0.5, step=0.001, value=0.0, label="Move plane along -Y (meters)")
            run_btn = gr.Button("Show / Update")
            info_out = gr.Textbox(label="Info", interactive=False)
        with gr.Column(scale=3):
            plot_out = gr.Plot(label="3D view (Plotly)")

    def on_update(file, json_file, plane_coeffs, slider_offset, lowf, highf):
        if file is None:
            return None, "Upload mesh first."
        return app_build_figure(file, json_file, plane_coeffs, slider_offset, lowf, highf)

    run_btn.click(on_update, inputs=[pcd_input, plane_json, plane_text, slider, low_frac, high_frac], outputs=[plot_out, info_out])
    slider.change(on_update, inputs=[pcd_input, plane_json, plane_text, slider, low_frac, high_frac], outputs=[plot_out, info_out])

# Launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)

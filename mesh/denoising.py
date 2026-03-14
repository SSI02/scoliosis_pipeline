# gradio_ply_denoise_fixed.py
# Paste and run: python gradio_ply_denoise_fixed.py
# Requires: open3d, numpy, plotly, gradio, scikit-learn
# pip install open3d numpy plotly gradio scikit-learn

import os
import tempfile
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import gradio as gr
from sklearn.neighbors import KDTree
import threading

# ---------------------------
# Utility & denoise methods
# ---------------------------
def load_ply_bytes(ply_bytes):
    """Load a .ply file from bytes (Gradio upload returns file-like object)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.write(ply_bytes.read())
    tmp.flush()
    tmp.close()
    pcd = o3d.io.read_point_cloud(tmp.name)
    os.unlink(tmp.name)
    return pcd

def pcd_to_numpy(pcd):
    return np.asarray(pcd.points), (np.asarray(pcd.colors) if np.asarray(pcd.colors).size else None)

def numpy_to_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def denoise_statistical(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def denoise_radius(pcd, nb_points=16, radius=0.05):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pcd.select_by_index(ind)

def denoise_dbscan_open3d(pcd, eps=0.05, min_points=10, keep_largest=True):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    mask = labels >= 0
    if not np.any(mask):
        return o3d.geometry.PointCloud()
    kept_idx = np.where(mask)[0]
    if keep_largest:
        unique, counts = np.unique(labels[mask], return_counts=True)
        if len(unique) > 0:
            largest_label = unique[np.argmax(counts)]
            kept_idx = np.where(labels == largest_label)[0]
    return pcd.select_by_index(kept_idx)

def combined_pipeline(pcd, use_sor, sor_n, sor_std, use_ror, ror_n, ror_radius, use_dbscan, db_eps, db_minpts, db_keep_largest):
    cur = pcd
    if use_sor:
        cur = denoise_statistical(cur, nb_neighbors=int(sor_n), std_ratio=float(sor_std))
    if use_ror:
        cur = denoise_radius(cur, nb_points=int(ror_n), radius=float(ror_radius))
    if use_dbscan:
        cur = denoise_dbscan_open3d(cur, eps=float(db_eps), min_points=int(db_minpts), keep_largest=db_keep_largest)
    return cur

# ---------------------------
# Robust autosuggest helpers (fixed)
# ---------------------------
def compute_nn_stats(points, k=6):
    """
    Robust computation of nearest-neighbour stats.
    - clamps k to available points
    - adds tiny jitter if all points identical / zero distances
    Returns dict with median_nn and mean_nn (floats).
    """
    pts = np.asarray(points)
    N = pts.shape[0]
    if N < 2:
        # Not enough points: return safe defaults
        return {'median_nn': 1e-6, 'mean_nn': 1e-6}

    # clamp k so query always valid
    k_eff = max(1, min(k, N - 1))
    # if all points identical (rare), add tiny jitter so KDTree works
    if np.allclose(np.var(pts, axis=0), 0.0):
        pts = pts + (np.random.RandomState(0).randn(*pts.shape) * 1e-9)

    tree = KDTree(pts)
    try:
        dists, idx = tree.query(pts, k=k_eff + 1)
    except Exception:
        # fallback: use k=1
        dists, idx = tree.query(pts, k=2)

    # ignore the self-dist (first column)
    nn = dists[:, 1:]
    median_nn = float(np.median(nn))
    mean_nn = float(np.mean(nn))
    # safety floors
    if median_nn <= 0.0:
        median_nn = 1e-6
    if mean_nn <= 0.0:
        mean_nn = median_nn
    return {'median_nn': median_nn, 'mean_nn': mean_nn}

def autosuggest_parameters(points):
    """
    Heuristics for parameter suggestions, robust for small clouds.
    Returns a dictionary of suggested numeric values (floats / ints).
    """
    pts = np.asarray(points)
    n = max(0, pts.shape[0])
    if n < 10:
        # Very small cloud — return small, safe defaults
        med = 1e-3
        return {
            'median_nn': med,
            'suggest_sor_n': 8,
            'suggest_sor_std': 1.5,
            'suggest_ror_n': 6,
            'suggest_ror_radius': max(1e-6, med * 2.0),
            'suggest_db_eps': max(1e-6, med * 1.5),
            'suggest_db_min': 6
        }

    stats = compute_nn_stats(pts, k=20)
    med = stats['median_nn']

    # heuristics (robust clamping)
    ror_radius = float(max(1e-6, med * 2.0))
    db_eps = float(max(1e-6, med * 1.5))
    sor_n = int(max(8, min(70, int(np.sqrt(n) / 2))))
    ror_n = int(max(6, min(40, int(np.sqrt(n) / 10))))
    db_min = int(max(6, min(30, int(np.sqrt(n) / 20) * 2 + 6)))

    # final safety clamps
    sor_n = max(1, sor_n)
    ror_n = max(1, ror_n)
    db_min = max(1, db_min)

    return {
        'median_nn': med,
        'suggest_sor_n': sor_n,
        'suggest_sor_std': 1.5,
        'suggest_ror_n': ror_n,
        'suggest_ror_radius': ror_radius,
        'suggest_db_eps': db_eps,
        'suggest_db_min': db_min
    }

# ---------------------------
# Plotly preview helpers
# ---------------------------
def make_plotly_scatter(points, colors=None, max_points_preview=200000, name="cloud"):
    """
    Create a Plotly scatter3d figure for interactive preview. Downsamples to max_points_preview.
    Returns plotly Figure.
    """
    n = points.shape[0]
    if n > max_points_preview:
        # downsample uniformly
        idx = np.linspace(0, n-1, max_points_preview, dtype=int)
        pts = points[idx]
        if colors is not None:
            cols = colors[idx]
        else:
            cols = None
    else:
        pts = points
        cols = colors

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if cols is not None:
        # plotly expects colors as rgb strings OR color arrays
        r = (cols[:, 0] * 255).astype(np.uint8)
        g = (cols[:, 1] * 255).astype(np.uint8)
        b = (cols[:, 2] * 255).astype(np.uint8)
        rgb = ["rgb({}, {}, {})".format(int(rr), int(gg), int(bb)) for rr, gg, bb in zip(r, g, b)]
        marker = dict(size=1.5, color=rgb)
    else:
        marker = dict(size=1.5, color='gray')

    fig = go.Figure(data=[
        go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker, name=name)
    ])
    fig.update_layout(scene=dict(aspectmode='auto'), margin=dict(l=0, r=0, b=0, t=0))
    return fig

# ---------------------------
# Gradio App callbacks and state
# ---------------------------
STATE = {
    'original_pcd': None,
    'original_pts': None,
    'original_cols': None,
    'clean_pcd': None
}

def upload_and_prepare(file_obj):
    """Load .ply, store in STATE and return preview + suggested params."""
    if file_obj is None:
        return (None, "No file uploaded", None, None, None, None, None, None, None)
    try:
        pcd = load_ply_bytes(file_obj)
        pts, cols = pcd_to_numpy(pcd)
        STATE['original_pcd'] = pcd
        STATE['original_pts'] = pts
        STATE['original_cols'] = cols
        STATE['clean_pcd'] = pcd
        n = pts.shape[0]
        sug = autosuggest_parameters(pts)
        before_fig = make_plotly_scatter(pts, cols)
        # Return: figure, message, sor_n, sor_std, ror_n, ror_radius, db_eps, db_min
        return (before_fig, None, f"Loaded: {n} points",
                sug['suggest_sor_n'], sug['suggest_sor_std'],
                sug['suggest_ror_n'], sug['suggest_ror_radius'],
                sug['suggest_db_eps'], sug['suggest_db_min'])
    except Exception as e:
        return (None, None, f"Failed to load: {e}", None, None, None, None, None, None)

def do_autosuggest():
    """Return suggested values and text for UI update (or None if missing)."""
    pts = STATE.get('original_pts', None)
    if pts is None:
        return None
    sug = autosuggest_parameters(pts)
    out_text = (f"median_nn={sug['median_nn']:.6f}, "
                f"suggest_sor_n={sug['suggest_sor_n']}, "
                f"suggest_ror_n={sug['suggest_ror_n']}, ror_radius={sug['suggest_ror_radius']:.6f}, "
                f"db_eps={sug['suggest_db_eps']:.6f}, db_min={sug['suggest_db_min']}")
    return (sug['suggest_sor_n'], sug['suggest_sor_std'],
            sug['suggest_ror_n'], sug['suggest_ror_radius'],
            sug['suggest_db_eps'], sug['suggest_db_min'], out_text)

def apply_filters(use_sor, sor_n, sor_std, use_ror, ror_n, ror_radius, use_dbscan, db_eps, db_minpts, db_keep_largest):
    pcd = STATE.get('original_pcd', None)
    if pcd is None:
        return (None, None, "No point cloud loaded")
    # copy to preserve original
    pcd_copy = o3d.geometry.PointCloud(pcd)
    try:
        cleaned = combined_pipeline(pcd_copy, use_sor, sor_n, sor_std, use_ror, ror_n, ror_radius, use_dbscan, db_eps, db_minpts, db_keep_largest)
        STATE['clean_pcd'] = cleaned
        pts, cols = pcd_to_numpy(cleaned)
        after_fig = make_plotly_scatter(pts, cols)
        before_fig = make_plotly_scatter(np.asarray(STATE['original_pts']), STATE['original_cols'])
        status = f"Done. Before: {len(STATE['original_pts'])} pts | After: {len(pts)} pts"
        return (before_fig, after_fig, status)
    except Exception as e:
        return (None, None, f"Error: {e}")

def save_cleaned_to_path(filename):
    pcd = STATE.get('clean_pcd', None)
    if pcd is None:
        return "No cleaned cloud to save."
    try:
        o3d.io.write_point_cloud(filename, pcd)
        return f"Saved cleaned PLY to: {filename}"
    except Exception as e:
        return f"Save failed: {e}"

def open_full_in_open3d():
    pcd = STATE.get('clean_pcd', None)
    if pcd is None:
        return "No cleaned cloud to open."
    # spawn a thread so Gradio doesn't block
    def _vis():
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Full-resolution Open3D Viewer")
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        vis.run()
        vis.destroy_window()
    t = threading.Thread(target=_vis, daemon=True)
    t.start()
    return "Opened full-res cloud in Open3D visualizer (desktop window)."

# ---------------------------
# Gradio Interface
# ---------------------------
with gr.Blocks(title="PLY Denoiser (Gradio + Plotly + Open3D) - Fixed Autosuggest") as demo:
    gr.Markdown("## PLY Denoiser — fast interactive preview + autosuggest fixes\n"
                "Upload a PLY, press **Auto-suggest** to get recommended parameters, tweak them, then **Apply**.")
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Upload .ply", file_types=['.ply'])
            load_btn = gr.Button("Load and Preview")
            autosugg_btn = gr.Button("Auto-suggest parameters")
            # SOR controls
            use_sor = gr.Checkbox(label="Use Statistical Outlier Removal (SOR)", value=True)
            sor_n = gr.Slider(5, 200, value=30, step=1, label="SOR: nb_neighbors (k)")
            sor_std = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="SOR: std_ratio")
            # ROR controls
            use_ror = gr.Checkbox(label="Use Radius Outlier Removal (ROR)", value=True)
            ror_n = gr.Slider(1, 200, value=16, step=1, label="ROR: nb_points (min neigh)")
            ror_radius = gr.Number(value=0.05, label="ROR: radius (scene units)")
            # DBSCAN controls
            use_db = gr.Checkbox(label="Use DBSCAN (cluster removal)", value=False)
            db_eps = gr.Number(value=0.05, label="DBSCAN: eps (scene units)")
            db_min = gr.Slider(1, 200, value=10, step=1, label="DBSCAN: min_points")
            db_keep = gr.Checkbox(label="Keep only largest cluster", value=True)
            apply_btn = gr.Button("Apply filters and preview")
            save_btn = gr.Button("Save cleaned .ply")
            save_path = gr.Textbox(label="Save filename (e.g. cleaned.ply)", value="cleaned_output.ply")
            open_open3d_btn = gr.Button("Open full-res in Open3D (desktop)")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            before_plot = gr.Plot(label="Before (preview)")
            after_plot = gr.Plot(label="After (preview)")

    # wiring
    def _load_and_prepare(file):
        if file is None:
            return (None, None, "Upload a PLY file first", None, None, None, None, None, None)
        fobj = open(file.name, 'rb')
        before_fig, after_fig, msg, sug_sor_n, sug_sor_std, sug_ror_n, sug_ror_rad, sug_db_eps, sug_db_min = upload_and_prepare(fobj)
        fobj.close()
        if before_fig is None:
            return (None, None, msg, None, None, None, None, None, None)
        # return the Plotly figure and suggested params to populate fields
        return (before_fig, after_fig, msg, sug_sor_n, sug_sor_std, sug_ror_n, sug_ror_rad, sug_db_eps, sug_db_min)

    load_btn.click(_load_and_prepare, inputs=[file_in], outputs=[before_plot, after_plot, status, sor_n, sor_std, ror_n, ror_radius, db_eps, db_min])

    def _on_autosuggest():
        res = do_autosuggest()
        if res is None:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), status.update(value="No cloud loaded. Load a PLY first."))
        s_sor_n, s_sor_std, s_ror_n, s_ror_rad, s_db_eps, s_db_min, text = res
        return (sor_n.update(value=s_sor_n), sor_std.update(value=s_sor_std),
                ror_n.update(value=s_ror_n), ror_radius.update(value=s_ror_rad),
                db_eps.update(value=s_db_eps), db_min.update(value=s_db_min),
                status.update(value=text))

    autosugg_btn.click(_on_autosuggest, inputs=None, outputs=[sor_n, sor_std, ror_n, ror_radius, db_eps, db_min, status])

    def _apply(*args):
        (use_sor_v, sor_n_v, sor_std_v, use_ror_v, ror_n_v, ror_radius_v,
        use_db_v, db_eps_v, db_min_v, db_keep_v) = args
        before_fig, after_fig, stat = apply_filters(use_sor_v, sor_n_v, sor_std_v, use_ror_v, ror_n_v, ror_radius_v, use_db_v, db_eps_v, db_min_v, db_keep_v)
        return before_fig, after_fig, stat

    apply_btn.click(_apply, inputs=[use_sor, sor_n, sor_std, use_ror, ror_n, ror_radius, use_db, db_eps, db_min, db_keep], outputs=[before_plot, after_plot, status])

    def _save(path):
        return save_cleaned_to_path(path)

    save_btn.click(_save, inputs=[save_path], outputs=[status])

    open_open3d_btn.click(lambda: open_full_in_open3d(), outputs=[status])

demo.launch(server_name="0.0.0.0", share=False)

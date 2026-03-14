"""
midline_plane_relative_params.py

Updated mid-sagittal plane extraction module.
All parameters are chosen relative to the mesh size (bounding-box diagonal) so you won't accidentally collapse the cloud
with a too-large voxel size. The script includes:
 - safe loading & adaptive downsampling (voxel_frac)
 - slice-midpoint plane extraction with slice_thickness_frac
 - symmetry optimization (optional) with sensible defaults
 - automatic fallback guards and informative prints
 - functions returning plane as (n, d) and midline points
 - visualization via Open3D

Usage:
    python midline_plane_relative_params.py /path/to/mesh.ply

Author: generated for Scoliosis-2 pipeline
"""

import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------
# Utilities (safe loaders)
# ----------------------

def compute_bbox_diag(pts):
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    return np.linalg.norm(bbox_max - bbox_min), bbox_min, bbox_max


def load_and_downsample_relative(mesh_path,
                                 target_points=100000,
                                 voxel_frac=0.005,
                                 min_points=2000,
                                 use_vertices_threshold=50000):
    """
    Loads mesh and downsample in a safe, relative manner.

    Parameters chosen relative to model size (bbox diagonal).
    - voxel_frac: voxel_size = diag * voxel_frac
    - target_points: maximum points to sample from mesh if mesh is dense
    - min_points: minimum acceptable final point count; if downsampling collapses below this, skip it
    - use_vertices_threshold: if mesh vertex count < threshold, use vertices directly instead of uniform sampling

    Returns: open3d.geometry.PointCloud
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Failed to load mesh or mesh has no vertices: {mesh_path}")

    verts = np.asarray(mesh.vertices)
    diag, bbox_min, bbox_max = compute_bbox_diag(verts)
    if diag <= 0:
        raise RuntimeError("Degenerate bounding box (zero diagonal). Check mesh coordinates.")

    # choose voxel relative to diag
    voxel_size = max(1e-9, diag * float(voxel_frac))
    print(f"Mesh bbox diag: {diag:.6e}, voxel_frac: {voxel_frac}, chosen voxel_size: {voxel_size:.6e}")

    n_verts = verts.shape[0]
    # decide sampling strategy
    if n_verts <= use_vertices_threshold:
        # small / medium mesh: use vertices directly to preserve detail
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
        print(f"Using raw vertices (count={n_verts}) as point cloud.")
    else:
        n_samples = min(int(target_points), n_verts)
        print(f"Mesh has {n_verts} vertices; sampling uniformly to {n_samples} points.")
        pcd = mesh.sample_points_uniformly(number_of_points=n_samples)

    before = np.asarray(pcd.points).shape[0]
    # only downsample if it meaningfully reduces point count and not too aggressive
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    after = np.asarray(pcd_ds.points).shape[0]
    print(f"Points before voxel_down_sample: {before}, after: {after}")

    if after < min_points:
        print("Voxel downsample reduced points below min_points. Skipping downsample to preserve points.")
        pcd_ds = pcd
    final_pts = np.asarray(pcd_ds.points).shape[0]
    if final_pts < 50:
        raise RuntimeError(f"Too few points after preprocessing ({final_pts}). Try decreasing voxel_frac or use_vertices_threshold.")
    return pcd_ds


# ----------------------
# Slice-midpoint method (relative params)
# ----------------------

def slice_midpoint_plane_relative(points,
                                  height_axis=2,
                                  slice_thickness_frac=0.01,
                                  n_slices=200,
                                  percentiles=(5, 95),
                                  min_pts_per_slice=30):
    """
    Computes mid-sagittal plane by slicing along height axis and taking robust left-right midpoints.
    
    Standard axis convention:
    - X-axis (axis 0): from left shoulder to right shoulder
    - Y-axis (axis 1): high value towards upper body, low value towards foot
    - Z-axis (axis 2): high value towards hips, low value towards neck (spine axis)
    
    height_axis=2 means slicing along Z-axis (spine), which is correct for the standard convention.

    slice_thickness_frac: fraction of bbox diagonal used as slice thickness
    n_slices: maximum number of slices to attempt across height range (will be adapted by bbox)
    """
    P = np.asarray(points)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be Nx3")

    diag, bbox_min, bbox_max = compute_bbox_diag(P)
    mn = P[:, height_axis].min()
    mx = P[:, height_axis].max()
    if mx <= mn:
        raise RuntimeError("Invalid height range in points.")

    # slice thickness proportional to diag
    slice_thickness = max(1e-9, diag * float(slice_thickness_frac))
    # fallback logic to ensure we produce at most n_slices
    total_height = mx - mn
    if slice_thickness <= 0:
        slice_thickness = total_height / float(n_slices)
    else:
        # compute slices from thickness, limit to n_slices
        num_possible = int(np.ceil(total_height / slice_thickness))
        if num_possible > n_slices:
            slice_thickness = total_height / float(n_slices)

    zs = np.arange(mn, mx, slice_thickness)
    other_axes = [0, 1, 2]
    other_axes.remove(height_axis)

    midpoints = []
    for z in zs:
        mask = (P[:, height_axis] >= z) & (P[:, height_axis] < z + slice_thickness)
        slice_pts = P[mask]
        if slice_pts.shape[0] < min_pts_per_slice:
            continue
        x_vals = slice_pts[:, other_axes[0]]
        left = np.percentile(x_vals, percentiles[0])
        right = np.percentile(x_vals, percentiles[1])
        mid_x = 0.5 * (left + right)
        depth_median = np.median(slice_pts[:, other_axes[1]])
        mid = np.zeros(3)
        mid[height_axis] = z + 0.5 * slice_thickness
        mid[other_axes[0]] = mid_x
        mid[other_axes[1]] = depth_median
        midpoints.append(mid)

    midpoints = np.array(midpoints)
    if midpoints.shape[0] < 3:
        raise RuntimeError("Not enough midpoints found. Try changing slice params or crop mesh.")

    centroid = midpoints.mean(axis=0)
    U, S, Vt = np.linalg.svd(midpoints - centroid, full_matrices=False)
    normal = Vt[-1, :]
    normal = normal / np.linalg.norm(normal)
    d = -normal.dot(centroid)
    return normal, d, midpoints


# ----------------------
# Symmetry optimization (reflection + NN)
# ----------------------

def reflect_points(P, n, d):
    signed = np.dot(P, n) + d
    return P - 2.0 * np.outer(signed, n)


def symmetry_cost_params(params, P_sample, trim_fraction=0.1):
    theta, phi, d = params
    n = np.array([np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    Pref = reflect_points(P_sample, n, d)
    tree_P = cKDTree(P_sample)
    tree_Ref = cKDTree(Pref)
    d1, _ = tree_Ref.query(P_sample, k=1)
    d2, _ = tree_P.query(Pref, k=1)
    errors = np.concatenate([d1, d2])
    if trim_fraction > 0 and trim_fraction < 0.4:
        k = int(len(errors) * (1.0 - trim_fraction))
        if k < 1:
            return np.mean(errors**2)
        errors = np.partition(errors, k)[:k]
    return np.mean(errors**2)


def safe_pca_init(P_sample):
    if P_sample.shape[0] < 10:
        # fallback axis: left-right (X) if small cloud
        print("PCA init: too few points; using fallback normal [1,0,0]")
        n0 = np.array([1.0, 0.0, 0.0])
        d0 = -n0.dot(P_sample.mean(axis=0))
        return n0, d0
    cov = np.cov(P_sample.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n0 = eigvecs[:, np.argmin(eigvals)]
    n0 = n0 / np.linalg.norm(n0)
    d0 = -n0.dot(P_sample.mean(axis=0))
    return n0, d0


def symmetry_optimize_plane_relative(points,
                                     init_n=None,
                                     init_d=None,
                                     sample_pts_frac=0.2,
                                     max_sample_pts=20000,
                                     trim_fraction=0.08):
    P = np.asarray(points)
    N = P.shape[0]
    if N <= 3:
        raise RuntimeError("Too few points for symmetry optimization.")

    # sample proportionally
    sample_pts = min(int(max(1000, N * float(sample_pts_frac))), max_sample_pts)
    if N > sample_pts:
        idx = np.random.choice(N, sample_pts, replace=False)
        P_sample = P[idx]
    else:
        P_sample = P

    if init_n is None or init_d is None:
        init_n, init_d = safe_pca_init(P_sample)

    # convert init_n to spherical
    def n_to_angles(nvec):
        nvec = nvec / np.linalg.norm(nvec)
        theta = np.arccos(np.clip(nvec[2], -1.0, 1.0))
        phi = np.arctan2(nvec[1], nvec[0])
        return theta, phi

    th0, ph0 = n_to_angles(init_n)
    x0 = np.array([th0, ph0, float(init_d)])
    bounds = [(0.0, np.pi), (-np.pi, np.pi), (None, None)]

    res = minimize(lambda x: symmetry_cost_params(x, P_sample, trim_fraction),
                   x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-6})

    theta_opt, phi_opt, d_opt = res.x
    n_opt = np.array([np.sin(theta_opt) * np.cos(phi_opt),
                      np.sin(theta_opt) * np.sin(phi_opt),
                      np.cos(theta_opt)])
    n_opt = n_opt / np.linalg.norm(n_opt)
    return n_opt, d_opt, res


# ----------------------
# Plane mesh helper
# ----------------------

def plane_to_point_and_normal(n, d):
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    p0 = -d * n

    return p0, n


def save_plane_to_json(out_path,
                       normal,
                       d,
                       method,
                       bbox_diag,
                       height_axis,
                       voxel_frac,
                       slice_thickness_frac,
                       notes=""):
    """
    Save plane parameters in a reproducible JSON format.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # closest point on plane to origin
    point_on_plane = (-d * normal).tolist()

    data = {
        "method": method,
        "plane_equation": {
            "normal": normal.tolist(),
            "d": float(d)
        },
        "point_on_plane": point_on_plane,
        "bbox_diag": float(bbox_diag),
        "height_axis": int(height_axis),
        "voxel_frac": float(voxel_frac),
        "slice_thickness_frac": float(slice_thickness_frac),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "notes": notes
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Mid-sagittal plane saved to: {out_path}")


def save_detailed_stats(out_path, entries):
    """
    Save list of objects with Quantity, Definition, Value keys.
    """
    # Validate structure roughly
    for e in entries:
        if "Quantity" not in e:
            e["Quantity"] = "Unknown"
        if "Quantity Definition" not in e:
            e["Quantity Definition"] = ""
        if "Quantity Estimated Value" not in e:
            e["Quantity Estimated Value"] = None

    out_path = Path(out_path)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=4)
    print(f"[OK] Detailed stats saved to: {out_path}")


def visualize_2d_with_legend(pcd, plane_n, plane_d, midpoints, title, save_path):
    """
    Generate a 2D projection (Top View / X-Z plane typically for spine) 
    using Matplotlib to include a clear legend.
    """
    pts = np.asarray(pcd.points)
    
    # 2D Projection: X (Width) vs Z (Spine Axis)
    # This corresponds to Coronal view if Y is depth.
    # Check axes: 
    # X: Left-Right
    # Y: Depth (Back-Front)
    # Z: Sup-Inf (Spine Length)
    # 
    # Top view usually means X-Y (Transverse).
    # Coronal view means X-Z (Frontal).
    # Sagittal view means Y-Z.
    # 
    # For "Mid-Sagittal Plane", we want to see the symmetry, which is best viewed in Coronal (X-Z) 
    # or Transverse (X-Y). 
    # Since the plane splits Left/Right, viewing from the Front (X-Z) or Top (X-Y) shows the split.
    # The spine is along Z. So X-Z is the "Front" view of the person standing.
    
    plt.figure(figsize=(10, 12))
    
    # Subsample for speed/clarity
    step = max(1, len(pts) // 10000)
    plt.scatter(pts[::step, 0], pts[::step, 2], c='gray', s=1, alpha=0.3, label='Scan Points (X-Z Projection)')
    
    # Check plane orientation
    # Plane: n.x*x + n.y*y + n.z*z + d = 0
    # We want to plot the intersection line on X-Z plane. 
    # Assume Y is roughly constant or we project everything. 
    # If we just project the plane equation to X-Z (assuming n.y is small for a sagittal plane):
    # n.x*x + n.z*z + d ~ 0 (ignoring Y). 
    # Better: Plot the projected line of the plane.
    
    # Create line points for the plane in X-Z bounds
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    
    # The plane is roughly X = constant (Sagittal). 
    # n ~ [1, 0, 0]. 
    # If n[2] is small (plane parallel to Z axis), we can solve for X given Z.
    # n.x * x + n.z * z + n.y * mean_y + d = 0
    mean_y = pts[:, 1].mean()
    
    z_vals = np.linspace(z_min, z_max, 100)
    # n.x * x = - (n.z * z + n.y * mean_y + d)
    # x = - (n.z * z + n.y * mean_y + d) / n.x
    
    if abs(plane_n[0]) > 1e-3:
        x_vals = - (plane_n[2] * z_vals + plane_n[1] * mean_y + plane_d) / plane_n[0]
        plt.plot(x_vals, z_vals, 'r-', linewidth=2, label='Mid-Sagittal Plane (Projected)')
    else:
        # Plane is horizontal?? Unlikely for sagittal plane.
        plt.axhline(y=-plane_d/plane_n[2], color='r', label='Plane (Horizontal?)')

    if midpoints is not None and len(midpoints) > 0:
        plt.scatter(midpoints[:, 0], midpoints[:, 2], c='green', s=10, label='Estimated Midline Points')

    plt.title(f"{title}\n(Frontal Projection X-Z)")
    plt.xlabel("X (Left-Right) [mm]")
    plt.ylabel("Z (Spine Axis) [mm]")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[OK] 2D Visualization saved to: {save_path}")
        
    plt.close() # Close to avoid memory leaks



def make_plane_mesh(n, d, size=500.0):
    p0, nvec = plane_to_point_and_normal(n, d)
    if abs(nvec[2]) < 0.9:
        t1 = np.cross(nvec, np.array([0, 0, 1.0]))
    else:
        t1 = np.cross(nvec, np.array([0, 1.0, 0]))
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(nvec, t1)
    half = size * 0.5
    corners = [
        p0 +  t1 * half +  t2 * half,
        p0 +  t1 * half -  t2 * half,
        p0 -  t1 * half -  t2 * half,
        p0 -  t1 * half +  t2 * half,
    ]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(corners))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2],[2,3,0]]))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.9, 0.2, 0.2])
    return mesh


# ----------------------
# Visualization and I/O
# ----------------------

def visualize_result(pcd, plane_n, plane_d, midpoints=None, title="Mid-sagittal plane", save_path=None):
    """Visualize point cloud, computed plane and midpoints with an embedded color-legend.

    - Grey point cloud = input scan
    - Red plane = computed mid-sagittal plane
    - Green points = slice midpoints (estimated midline samples)

    Note: Open3D has no built-in legend UI, so this function creates small colored spheres placed just outside
    the model bbox as a visual legend and prints a short legend in the console as well.
    """
    geoms = []
    pcd_temp = pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geoms.append(pcd_temp)

    # plane mesh centered and sized relative to point cloud diag
    plane_mesh = make_plane_mesh(plane_n, plane_d, size=diag_scale(pcd) * 0.8)
    plane_mesh.compute_vertex_normals()
    plane_mesh.paint_uniform_color([0.9, 0.2, 0.2])
    geoms.append(plane_mesh)

    if midpoints is not None and len(midpoints) > 0:
        mid_pcd = o3d.geometry.PointCloud()
        mid_pcd.points = o3d.utility.Vector3dVector(np.asarray(midpoints))
        mid_pcd.paint_uniform_color([0.0, 0.8, 0.0])
        mid_pcd.estimate_normals()
        geoms.append(mid_pcd)

    # create a small legend: three spheres placed slightly outside bbox
    pts = np.asarray(pcd.points)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    offset = bbox_size * 0.08
    legend_origin = bbox_max + offset

    def make_sphere(center, radius, color):
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sph.translate(center)
        sph.compute_vertex_normals()
        sph.paint_uniform_color(color)
        return sph

    diag = np.linalg.norm(bbox_size)
    sph_r = max(diag * 0.01, 1e-4)
    # positions for legend spheres
    s1 = legend_origin + np.array([0.0, 0.0, 0.0])
    s2 = legend_origin + np.array([sph_r * 4.0, 0.0, 0.0])
    s3 = legend_origin + np.array([sph_r * 8.0, 0.0, 0.0])

    # red sphere -> plane
    legend_plane = make_sphere(s1, sph_r, [0.9, 0.2, 0.2])
    # green sphere -> midpoints
    legend_mid = make_sphere(s2, sph_r, [0.0, 0.8, 0.0])
    # blue sphere -> cloud (for clarity)
    legend_cloud = make_sphere(s3, sph_r, [0.6, 0.6, 0.6])

    geoms.extend([legend_plane, legend_mid, legend_cloud])

    # Print legend to console for clarity
    print('\n--- Visualization legend ---')
    print('Red plane  : computed mid-sagittal plane (n,d)')
    print('Green dots : slice midpoints (estimated midline samples)')
    print('Grey cloud : input point cloud')
    print('--- End legend ---\n')

    # Also print plane centroid (closest point to origin on plane) to help debug placement
    p0, _ = plane_to_point_and_normal(plane_n, plane_d)
    print(f'Plane center (p0): {p0}, plane normal: {plane_n}, d: {plane_d}')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    for g in geoms:
        vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()

    if save_path:
        # Render a few frames to ensure everything is loaded/positioned
        for _ in range(5):
            vis.poll_events()
            vis.update_renderer()
        vis.capture_screen_image(str(save_path))
        print(f"[OK] Visualization PNG saved to: {save_path}")

    vis.run()
    vis.destroy_window()



def diag_scale(pcd):
    pts = np.asarray(pcd.points)
    diag, _, _ = compute_bbox_diag(pts)
    return diag * 1.5 if diag > 0 else 1.0


def compute_midline_stats(midpoints):
    """
    Compute statistics for the midline points.
    """
    if midpoints is None or len(midpoints) == 0:
        return None

    stats = {
        "count": int(len(midpoints)),
        "mean_x": float(np.mean(midpoints[:, 0])),
        "mean_y": float(np.mean(midpoints[:, 1])),
        "mean_z": float(np.mean(midpoints[:, 2])),
        "std_x": float(np.std(midpoints[:, 0])),
        "std_y": float(np.std(midpoints[:, 1])),
        "std_z": float(np.std(midpoints[:, 2])),
        "min_x": float(np.min(midpoints[:, 0])),
        "min_y": float(np.min(midpoints[:, 1])),
        "min_z": float(np.min(midpoints[:, 2])),
        "max_x": float(np.max(midpoints[:, 0])),
        "max_y": float(np.max(midpoints[:, 1])),
        "max_z": float(np.max(midpoints[:, 2]))
    }
    return stats


# ----------------------
# Main flow
# ----------------------

def main(mesh_path,
         use_symmetry_optimize=True,
         voxel_frac=0.005,
         slice_thickness_frac=0.01,
         height_axis=2,
         no_vis=False):
    """
    Main function for spine AIX estimation.
    
    Standard axis convention:
    - X-axis (axis 0): from left shoulder to right shoulder
    - Y-axis (axis 1): high value towards upper body, low value towards foot
    - Z-axis (axis 2): high value towards hips, low value towards neck (spine axis)
    
    height_axis=2 means slicing along Z-axis (spine), which is correct.
    """
    pcd = load_and_downsample_relative(mesh_path, voxel_frac=voxel_frac)

    points = np.asarray(pcd.points)
    bbox_diag, _, _ = compute_bbox_diag(points)
    print("Points after preprocess:", points.shape[0])

    input_dir = Path(mesh_path).parent
    mesh_name = Path(mesh_path).stem

    # method 1
    try:
        normal1, d1, mids = slice_midpoint_plane_relative(points,
                                                         height_axis=height_axis,
                                                         slice_thickness_frac=slice_thickness_frac)
        print("Slice-midpoint plane normal:", normal1, "d:", d1)

        # Prepare Detailed Stats for Method 1
        stats_entries = [
            {
                "Quantity": "Method",
                "Quantity Definition": "Algorithm used for plane estimation",
                "Quantity Estimated Value": "Slice-Midpoint"
            },
            {
                "Quantity": "Plane Normal",
                "Quantity Definition": "Unit normal vector of the mid-sagittal plane (approx [1, 0, 0])",
                "Quantity Estimated Value": normal1.tolist()
            },
            {
                "Quantity": "Plane Constant (d)",
                "Quantity Definition": "Plane equation constant d in nx*x + ny*y + nz*z + d = 0",
                "Quantity Estimated Value": float(d1)
            }
        ]
        
        if mids is not None:
             stats_entries.append({
                "Quantity": "Midline Points Count",
                "Quantity Definition": "Number of identified midline points used",
                "Quantity Estimated Value": int(len(mids))
             })
             stats_entries.append({
                "Quantity": "Midline Mean X",
                "Quantity Definition": "Average X-coordinate of midline points",
                "Quantity Estimated Value": float(np.mean(mids[:, 0]))
             })
             
        out_path_slice_json = input_dir / f"{mesh_name}_mid_sagittal_slice_midpoint.json"
        save_detailed_stats(out_path_slice_json, stats_entries)

        out_path_slice_png = input_dir / f"{mesh_name}_mid_sagittal_slice_midpoint_2d.png"
        visualize_2d_with_legend(pcd, normal1, d1, midpoints=mids, title="Slice-midpoint Result", save_path=out_path_slice_png)
        
        if not no_vis:
            # Keep original Open3D vis? Maybe rename it or keep as 3D option?
            out_path_slice_3d = input_dir / f"{mesh_name}_mid_sagittal_slice_midpoint_3d.png"
            visualize_result(pcd, normal1, d1, midpoints=mids, title="Slice-midpoint Result", save_path=out_path_slice_3d)

    except Exception as e:
        print("Slice-midpoint method failed:", e)
        import traceback
        traceback.print_exc()
        normal1 = None
        d1 = None
        mids = None

    # method 2
    if use_symmetry_optimize:
        try:
            init_n = normal1 if normal1 is not None else None
            init_d = d1 if d1 is not None else None
            n_opt, d_opt, res = symmetry_optimize_plane_relative(points,
                                                                init_n=init_n,
                                                                init_d=init_d)
            print("Optimized plane normal:", n_opt, "d:", d_opt)
            print("Optimization success:", res.success, res.message)

            # Prepare Detailed Stats for Method 2
            stats_entries_opt = [
                {
                    "Quantity": "Method",
                    "Quantity Definition": "Algorithm used for plane estimation",
                    "Quantity Estimated Value": "Symmetry-Optimized"
                },
                {
                    "Quantity": "Plane Normal",
                    "Quantity Definition": "Unit normal vector of the mid-sagittal plane",
                    "Quantity Estimated Value": n_opt.tolist()
                },
                {
                    "Quantity": "Plane Constant (d)",
                    "Quantity Definition": "Plane equation constant d",
                    "Quantity Estimated Value": float(d_opt)
                }
            ]
            if mids is not None:
                 stats_entries_opt.append({
                    "Quantity": "Midline Points Used",
                    "Quantity Definition": "Number of midline points from initial step used as hint/vis",
                    "Quantity Estimated Value": int(len(mids))
                 })

            out_path_opt = input_dir / f"{mesh_name}_mid_sagittal_symmetry_optimized.json"
            save_detailed_stats(out_path_opt, stats_entries_opt)

            out_path_opt_png = input_dir / f"{mesh_name}_mid_sagittal_symmetry_optimized_2d.png"
            visualize_2d_with_legend(pcd, n_opt, d_opt, midpoints=mids, title="Symmetry Optimization Result", save_path=out_path_opt_png)
            
            if not no_vis:
                out_path_opt_3d = input_dir / f"{mesh_name}_mid_sagittal_symmetry_optimized_3d.png"
                visualize_result(pcd, n_opt, d_opt, midpoints=mids, title="Symmetry Optimization Result", save_path=out_path_opt_3d)

        except Exception as e:
            print("Symmetry optimization failed:", e)
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Spine AIX Estimation")
    parser.add_argument("mesh_path", help="Path to the input mesh PLY")
    parser.add_argument("--no-vis", action="store_true", help="Disable Open3D visualization (prevents segfaults in headless envs)")
    
    args = parser.parse_args()
    
    main(args.mesh_path, no_vis=args.no_vis)

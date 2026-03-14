#!/usr/bin/env python3
"""
skeleton_legseed.py

Full self-contained Gradio app that:
 - Loads a mesh/pointcloud (.ply/.pcd/.obj/.xyz)
 - Accepts an optional ground-plane JSON (many formats supported)
 - Uses ROIs relative to object height (scale-invariant)
 - Voxelizes the ROI
 - Estimates leg seeds from far mesh ∩ ground-plane (with small upward offsets)
 - Computes geodesic distances from leg seeds to prune disconnected components (hands)
 - Prunes components not touching ground (relative tolerance)
 - Skeletonizes the pruned volume and clusters skeleton branches
 - Picks two leg branches (with verticality / ground-fraction / torso-distance heuristics)
 - Intersects branch centerlines with ground plane, computes mediolateral axis,
   rotates the full cloud so that this axis aligns with +X
 - Saves outputs and produces a combined PLY for visualization in Gradio Model3D

Usage:
    python skeleton_legseed.py
Dependencies:
    pip install numpy open3d scikit-image scipy scikit-learn gradio plotly
"""

import os
import json
import re
import math
import numpy as np
import open3d as o3d
from scipy import ndimage
import gradio as gr
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Use unified skeletonize API
try:
    from skimage.morphology import skeletonize as _skel_unified
    def skeletonize(grid):
        return _skel_unified(grid)
except Exception:
    try:
        from skimage.morphology import skeletonize_3d as _skel3d_impl
        def skeletonize(grid):
            return _skel3d_impl(grid)
    except Exception:
        from skimage.morphology import skeletonize as _skel_unified2
        def skeletonize(grid):
            return _skel_unified2(grid)

# ---------------------- Robust JSON plane parsing ----------------------
_num_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def _to_float_loose(x):
    if x is None:
        raise ValueError("None")
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
            raise ValueError(f"No numeric token in {s!r}")
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _to_float_loose(x[0])
    raise ValueError(f"Unsupported type {type(x)}")

def parse_plane_from_json(json_path: str):
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

    out = _from_list_like(data)
    if out is not None:
        return out

    # Handle various JSON structures
    candidates = [data.get("plane_equation"), data.get("ground_plane"), data]
    for cand in candidates:
        if not cand: continue
        out = _from_list_like(cand)
        if out: return out
        if isinstance(cand, dict):
            # Check a,b,c,d
            if all(k in cand for k in ("a","b","c","d")):
                try:
                    return (_to_float_loose(cand["a"]), _to_float_loose(cand["b"]), 
                            _to_float_loose(cand["c"]), _to_float_loose(cand["d"]))
                except: pass
            # Check normal, d
            if "normal" in cand and ("d" in cand or "D" in cand):
                try:
                    nx, ny, nz = cand["normal"][:3]
                    d_val = cand.get("d", cand.get("D"))
                    return (_to_float_loose(nx), _to_float_loose(ny), _to_float_loose(nz), _to_float_loose(d_val))
                except: pass

    return None

# ---------------------- IO helpers ----------------------

def load_mesh_or_cloud(path: str, sample_points: int = 50000):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".pts", ".txt"]:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError("Point cloud load failed or empty.")
        return np.asarray(pcd.points)
    else:
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            pcd = o3d.io.read_point_cloud(path)
            if pcd.is_empty():
                raise ValueError("Failed to load mesh or point cloud.")
            return np.asarray(pcd.points)
        else:
            sample_n = min(sample_points, max(len(mesh.vertices), sample_points))
            pcd = mesh.sample_points_uniformly(number_of_points=sample_n)
            return np.asarray(pcd.points)

# ---------------------- Geometry & voxel helpers ----------------------

def axis_aligned_bounds(points: np.ndarray):
    min_b = points.min(axis=0)
    max_b = points.max(axis=0)
    return min_b, max_b

def voxelize_pointcloud_to_grid(points: np.ndarray, voxel_size: float, padding: int = 3):
    min_b, max_b = axis_aligned_bounds(points)
    min_b = min_b - padding * voxel_size
    max_b = max_b + padding * voxel_size
    dims = np.ceil((max_b - min_b) / voxel_size).astype(int) + 1
    
    indices = np.floor((points - min_b) / voxel_size).astype(int)
    indices = np.clip(indices, 0, dims - 1)
    
    grid = np.zeros(dims, dtype=np.uint8)
    grid[indices[:,0], indices[:,1], indices[:,2]] = 1
    return grid, min_b, voxel_size

def grid_indices_to_points(indices, min_b, voxel_size):
    return indices * voxel_size + min_b + voxel_size * 0.5

def world_to_voxel_index(pt_world, min_b, voxel_size, grid_shape):
    idx = np.floor((pt_world - min_b) / voxel_size).astype(int)
    idx = np.clip(idx, 0, np.array(grid_shape) - 1)
    return tuple(idx)

# ---------------------- Geodesic computation ----------------------

def compute_geodesic_from_seeds(binary_grid, min_b, voxel_size, seed_world_points):
    """
    Compute geodesic shortest-path distances (meters) from seeds to all occupied voxels.
    """
    occ_idx = np.argwhere(binary_grid > 0)
    n_occ = len(occ_idx)
    if n_occ == 0:
        return np.full(binary_grid.shape, np.inf, dtype=float), []

    # mapping voxel->id
    id_of = {tuple(v): i for i, v in enumerate(map(tuple, occ_idx))}
    centers = (occ_idx * voxel_size) + (min_b + voxel_size*0.5)

    rows, cols, data = [], [], []
    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    
    # Build graph
    for i, v in enumerate(occ_idx):
        for dx,dy,dz in neigh:
            nb = (v[0]+dx, v[1]+dy, v[2]+dz)
            jid = id_of.get(nb, None)
            if jid is not None:
                # Euclidean distance between centers (usually just voxel_size)
                w = np.linalg.norm(centers[i] - centers[jid])
                rows.append(i); cols.append(jid); data.append(w)
    
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_occ, n_occ))

    # Identify seed nodes
    seed_ids = []
    for hw in seed_world_points:
        vi = world_to_voxel_index(np.array(hw), min_b, voxel_size, binary_grid.shape)
        
        # If specific voxel empty, search small radius
        if tuple(vi) not in id_of:
            found = None
            for r in range(1, 6):
                mins = np.maximum(np.array(vi) - r, 0)
                maxs = np.minimum(np.array(vi) + r, np.array(binary_grid.shape)-1)
                
                candidates = []
                for x in range(mins[0], maxs[0]+1):
                    for y in range(mins[1], maxs[1]+1):
                        for z in range(mins[2], maxs[2]+1):
                            if binary_grid[x,y,z]:
                                candidates.append((x,y,z))
                
                if candidates:
                    # Find closest candidate to world seed
                    cand_pts = np.array(candidates)
                    cand_world = cand_pts * voxel_size + (min_b + voxel_size*0.5)
                    dists = np.linalg.norm(cand_world - hw, axis=1)
                    found = candidates[np.argmin(dists)]
                    break
            
            if found:
                seed_ids.append(id_of[tuple(found)])
        else:
            seed_ids.append(id_of[tuple(vi)])

    if not seed_ids:
        return np.full(binary_grid.shape, np.inf, dtype=float), []

    # Dijkstra
    dist_matrix = dijkstra(csgraph=A, directed=False, indices=seed_ids, return_predecessors=False)
    
    if dist_matrix.ndim == 1:
        dist_from_seeds = dist_matrix
    else:
        dist_from_seeds = np.min(dist_matrix, axis=0)

    distances_grid = np.full(binary_grid.shape, np.inf, dtype=float)
    for i, v in enumerate(occ_idx):
        distances_grid[tuple(v)] = dist_from_seeds[i]

    return distances_grid, seed_ids

# ---------------------- Leg seeds from ground intersection ----------------------

def compute_leg_seeds_from_ground(points, ground_plane, H,
                                  max_offset_frac=0.02,
                                  ground_tol_frac=0.01,
                                  dbscan_eps_frac=0.02,
                                  dbscan_min_samples=5,
                                  n_expected=2):
    """
    Compute leg seed world points by intersecting the mesh with the ground plane.
    """
    if ground_plane is None:
        return []

    a,b,c,d = ground_plane
    norm_len = math.sqrt(a*a + b*b + c*c)
    if norm_len < 1e-12: return []

    # Convert relative fracs to absolutes
    max_offset = max_offset_frac * H
    ground_tol = ground_tol_frac * H
    dbscan_eps = dbscan_eps_frac * H

    offsets = np.linspace(0.0, max_offset, num=3) if max_offset > 0 else [0.0]

    for off in offsets:
        # Shift plane up
        d_shifted = d + off * norm_len
        # Compute signed distances
        dists = (points @ np.array([a,b,c]) + d_shifted) / norm_len
        mask = np.abs(dists) <= ground_tol
        near_pts = points[mask]
        
        if near_pts.shape[0] < 10:
            continue

        # Cluster points (Footprints)
        try:
            eps_val = max(1e-6, dbscan_eps)
            cl = DBSCAN(eps=eps_val, min_samples=max(2, int(dbscan_min_samples))).fit(near_pts)
            labels = cl.labels_
            unique_labels = [lab for lab in np.unique(labels) if lab != -1]
            
            if not unique_labels:
                # If no clear clusters, but we have points, return centroid if asking for 1, or skip
                if near_pts.shape[0] > n_expected * 10:
                     return [near_pts.mean(axis=0)]
                continue

            # Sort clusters by size (largest first)
            clusters_sorted = sorted(unique_labels, key=lambda l: -np.sum(labels==l))
            
            seeds = []
            for lab in clusters_sorted[:n_expected]:
                cpts = near_pts[labels==lab]
                seeds.append(cpts.mean(axis=0))
            
            if len(seeds) > 0:
                return seeds
                
        except Exception:
            continue

    # Fallback: Projection extremes in XZ
    try:
        nvec = np.array([a,b,c], dtype=float)
        denom = (np.dot(nvec, nvec) + 1e-12)
        # Project all points to ground
        projs = points - np.outer((points @ nvec + d) / denom, nvec)
        p_xz = projs[:, [0,2]]
        pca = PCA(n_components=1)
        pca.fit(p_xz)
        proj_vals = (p_xz - pca.mean_).dot(pca.components_[0])
        idx_min = np.argmin(proj_vals)
        idx_max = np.argmax(proj_vals)
        return [projs[idx_min], projs[idx_max]]
    except Exception:
        return []

# ---------------------- Pruning Logic ----------------------

def keep_components_touching_ground_with_geodesic(binary_grid, min_b, voxel_size,
                                                  ground_plane=None, ground_tol=0.01,
                                                  distances_grid=None, geodesic_thresh=0.5,
                                                  verticality_thresh=2.0):
    structure = np.ones((3,3,3), dtype=int)
    labeled, ncomp = ndimage.label(binary_grid, structure=structure)
    if ncomp == 0:
        return binary_grid

    # Pre-calculate voxel centers
    xs, ys, zs = np.where(binary_grid==1)
    coords = np.vstack([xs, ys, zs]).T
    voxel_centers = grid_indices_to_points(coords, min_b, voxel_size)

    # Define ground touch check
    if ground_plane is None:
        ground_y = voxel_centers[:,1].min()
        def touching(center): return (center[1] - ground_y) <= (ground_tol + 1e-9)
    else:
        a,b,c,d = ground_plane
        def touching(center):
            x,y,z = center
            dist = abs(a*x + b*y + c*z + d) / np.sqrt(a*a + b*b + c*c)
            return dist <= (ground_tol + 1e-9)

    keep_labels = set()

    for lab in range(1, ncomp+1):
        idxs = np.argwhere(labeled == lab)
        if idxs.size == 0: continue
        
        centers_lab = grid_indices_to_points(idxs, min_b, voxel_size)
        
        # 1. Ground Proximity
        # Check the lowest point in the component
        lowest_idx = np.argmin(centers_lab[:,1])
        near_ground = touching(centers_lab[lowest_idx])
        
        # 2. Verticality
        verticality = 0.0
        try:
            pca = PCA(n_components=3)
            pca.fit(centers_lab)
            eigs = np.sort(pca.explained_variance_)[::-1]
            verticality = (eigs[0] / (eigs[1] + 1e-9))
        except: pass

        # 3. Geodesic Connectivity to Leg Seeds
        min_geo = np.inf
        if distances_grid is not None:
            # Check min distance of any voxel in this component to a seed
            # Optimization: just check a subset if component is huge
            sample_idxs = idxs if len(idxs) < 100 else idxs[::int(len(idxs)/100)]
            dvals = [distances_grid[tuple(v)] for v in sample_idxs]
            if dvals:
                min_geo = min(dvals)

        # Decision Logic
        keep = False
        # A: Close to ground AND connected to a leg seed (walking up the leg)
        if near_ground and (min_geo <= geodesic_thresh):
            keep = True
        # B: Close to ground AND highly vertical (standing leg)
        elif near_ground and (verticality >= verticality_thresh):
            keep = True
        # C: Fallback for large ground objects
        elif near_ground and (len(idxs) > 500):
            keep = True

        if keep:
            keep_labels.add(lab)

    kept_mask = np.isin(labeled, list(keep_labels))
    pruned = np.zeros_like(binary_grid)
    pruned[kept_mask] = 1
    return pruned

# ---------------------- Skeleton/Line Logic ----------------------

def run_skeletonization(binary_grid):
    return skeletonize(binary_grid > 0).astype(np.uint8)

def skeleton_to_points(skeleton_grid, min_b, voxel_size):
    xs, ys, zs = np.where(skeleton_grid > 0)
    return grid_indices_to_points(np.vstack([xs, ys, zs]).T, min_b, voxel_size)

def cluster_skeleton_points_filtered(pts, eps, min_samples, n_clusters_expected, 
                                     ground_plane, ground_tol, height,
                                     verticality_thresh, min_cluster_frac,
                                     torso_center, torso_radius_frac, required_ground_fraction):
    if len(pts) == 0: return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = clustering.labels_
    unique = [l for l in np.unique(labels) if l != -1]
    
    total_pts = len(pts)
    torso_radius = torso_radius_frac * height if height else 1.0

    # Ground check helper
    if ground_plane:
        a,b,c,d = ground_plane
        def is_near_ground(p):
            return (abs(a*p[0] + b*p[1] + c*p[2] + d) / np.sqrt(a*a+b*b+c*c)) <= ground_tol
    else:
        gy = pts[:,1].min()
        def is_near_ground(p): return (p[1] - gy) <= ground_tol

    clusters_info = []
    for lab in unique:
        cpts = pts[labels == lab]
        size = len(cpts)
        if size < max(3, int(min_cluster_frac * total_pts)):
            continue
        
        # Metrics
        centroid = cpts.mean(axis=0)
        dist_xz = np.sqrt((centroid[0]-torso_center[0])**2 + (centroid[2]-torso_center[1])**2)
        
        near_ground_count = sum(1 for p in cpts if is_near_ground(p))
        ground_frac = near_ground_count / size
        
        vert = 0.0
        try:
            pca = PCA(n_components=3).fit(cpts)
            eigs = np.sort(pca.explained_variance_)[::-1]
            vert = eigs[0] / (eigs[1] + 1e-9)
        except: pass

        # Score
        # Bonus for touching ground, verticality, size, and being under torso
        g_bonus = 2.0 if ground_frac >= required_ground_fraction else 0.0
        v_score = min(vert, 10.0) / 10.0
        sz_score = min(size/total_pts, 0.5) / 0.5
        prox_score = max(0.0, 1.0 - (dist_xz / (torso_radius + 1e-9)))
        
        score = (g_bonus * 1.5) + (0.8 * v_score) + (0.5 * sz_score) + (0.8 * prox_score)
        
        clusters_info.append({
            "pts": cpts, "score": score, "centroid": centroid
        })

    clusters_info.sort(key=lambda x: x["score"], reverse=True)
    
    # Selection Strategy
    selected = []
    # Take top N
    for c in clusters_info[:n_clusters_expected]:
        selected.append(c["pts"])
        
    return selected

def fit_line_through_points(pts):
    if len(pts) < 2: return None, None
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    dir_vec = eigvecs[:, -1] # Main direction
    return mean, dir_vec

def line_plane_intersection(mean, dir_vec, plane):
    a,b,c,d = plane
    n = np.array([a,b,c], dtype=float)
    denom = n.dot(dir_vec)
    if abs(denom) < 1e-8: return None
    t = -(n.dot(mean) + d) / denom
    return mean + t * dir_vec

def rotate_pointcloud_to_xaxis(pcd, origin_point, target_vector):
    v = target_vector / (np.linalg.norm(target_vector) + 1e-12)
    target = np.array([1.0, 0.0, 0.0])
    cross = np.cross(v, target)
    dot = np.dot(v, target)
    
    if np.linalg.norm(cross) < 1e-8:
        R = np.eye(3) if dot > 0 else np.eye(3) # Simplified for parallel
    else:
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        
    pcd_copy = pcd.translate(-origin_point, relative=False)
    pcd_copy.rotate(R, center=(0,0,0))
    pcd_copy.translate(origin_point, relative=False)
    return pcd_copy, R

# ---------------------- Main Pipeline ----------------------

def skeleton_main(pcd_path,
                  low_frac=0.05, high_frac=0.25,
                  voxel_frac=0.01,
                  ground_plane=None, ground_tol_frac=0.02,
                  dbscan_eps_frac=0.015, dbscan_min_samples=5,
                  verticality_thresh=2.0, min_cluster_frac=0.005,
                  geodesic_thresh=0.5,
                  leg_seed_max_offset_frac=0.02,
                  save_prefix="out"):
    
    os.makedirs(save_prefix, exist_ok=True)
    points = load_mesh_or_cloud(pcd_path)
    min_b, max_b = axis_aligned_bounds(points)
    H = max_b[1] - min_b[1]
    if H <= 0: raise RuntimeError("Invalid height H.")

    # 1. Ground Plane Estimation
    if ground_plane is None:
        pcd_temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        gp, _ = pcd_temp.segment_plane(distance_threshold=ground_tol_frac*H, ransac_n=3, num_iterations=1000)
        ground_plane = tuple(gp)
        print(f"Auto-estimated ground plane: {ground_plane}")

    # Params
    voxel_size = voxel_frac * H
    ground_tol = ground_tol_frac * H
    dbscan_eps = dbscan_eps_frac * H

    # 2. Leg Seeding (Intersect mesh with ground)
    leg_seeds = compute_leg_seeds_from_ground(points, ground_plane, H,
                                              max_offset_frac=leg_seed_max_offset_frac,
                                              ground_tol_frac=ground_tol_frac,
                                              dbscan_eps_frac=dbscan_eps_frac,
                                              dbscan_min_samples=dbscan_min_samples)
    if not leg_seeds:
        # Fallback seeds: Centroid
        leg_seeds = [points.mean(axis=0)]

    # 3. Slice
    y_feet = max_b[1] # Assumption: Up is roughly +Y or we slice relative to bounds
    # Note: User provided code assumes Y is Up. 
    # Logic: Slice from feet (maxY? minY? Depends on coord sys). 
    # Usually standard biology scans: Y is up. If Y is up, feet are at minY?
    # The prompt code assumes feet ~ max_b[1] in the slicing logic: `y1 = y_feet - low_frac`.
    # Let's stick to the prompt's logic.
    y1 = y_feet - (low_frac * H)
    y2 = y_feet - (high_frac * H)
    if y2 > y1: y1, y2 = y2, y1
    
    mask = (points[:,1] <= y1) & (points[:,1] >= y2)
    sliced_pts = points[mask]
    if len(sliced_pts) == 0: raise RuntimeError("Slice empty.")
    
    o3d.io.write_point_cloud(os.path.join(save_prefix, "sliced.ply"), 
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sliced_pts)))

    # 4. Voxelize
    binary_grid, min_grid_b, vx_used = voxelize_pointcloud_to_grid(sliced_pts, voxel_size)

    # 5. Geodesic Pruning
    dist_grid, _ = compute_geodesic_from_seeds(binary_grid, min_grid_b, vx_used, leg_seeds)
    pruned = keep_components_touching_ground_with_geodesic(binary_grid, min_grid_b, vx_used,
                                                           ground_plane, ground_tol,
                                                           distances_grid=dist_grid,
                                                           geodesic_thresh=geodesic_thresh,
                                                           verticality_thresh=verticality_thresh)

    # 6. Skeletonize
    skel_grid = run_skeletonization(pruned)
    skel_pts = skeleton_to_points(skel_grid, min_grid_b, vx_used)
    
    # 7. Cluster
    # Estimate torso center (XZ center of bounding box)
    torso_c = np.array([(min_b[0]+max_b[0])/2, (min_b[2]+max_b[2])/2])
    
    clusters = cluster_skeleton_points_filtered(skel_pts, dbscan_eps, int(dbscan_min_samples), 2,
                                                ground_plane, ground_tol, H,
                                                verticality_thresh, min_cluster_frac,
                                                torso_c, 0.6, 0.05)

    # 8. Fit Lines & Find Origin
    line_params = []
    ground_points = []
    leg_center_lines = [] # (start, end) tuples for viz

    for cpts in clusters[:2]:
        mean, dv = fit_line_through_points(cpts)
        if mean is not None:
            # Ground intersect
            inter = None
            if ground_plane:
                inter = line_plane_intersection(mean, dv, ground_plane)
            
            if inter is None: inter = mean # Fallback
            
            # Line segment for viz (project extent)
            vecs = cpts - mean
            projs = vecs.dot(dv)
            p_s = mean + projs.min() * dv
            p_e = mean + projs.max() * dv
            
            line_params.append((mean, dv))
            ground_points.append(inter)
            leg_center_lines.append((p_s, p_e))
        else:
            leg_center_lines.append(None) # Keep index sync

    # Fallback if < 2 legs found
    if len(ground_points) < 2:
        # Fallback: Extremes of skeleton
        if len(skel_pts) > 2:
            pca = PCA(n_components=1).fit(skel_pts[:,[0,2]])
            proj = (skel_pts[:,[0,2]] - pca.mean_).dot(pca.components_[0])
            p1 = skel_pts[np.argmin(proj)]
            p2 = skel_pts[np.argmax(proj)]
            ground_points = [p1, p2]
            leg_center_lines = [(p1, p2)] # Visual fallback line
        else:
            # Total fail fallback
            p1 = points.mean(axis=0)
            ground_points = [p1, p1 + np.array([0.1,0,0])]

    # 9. Rotation
    gp_arr = np.vstack(ground_points)
    if len(gp_arr) >= 2:
        # Pick 2 furthest points if > 2
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(gp_arr))
        i, j = np.unravel_index(np.argmax(D), D.shape)
        p1, p2 = gp_arr[i], gp_arr[j]
    else:
        p1, p2 = gp_arr[0], gp_arr[1]

    origin = (p1 + p2) / 2.0
    vec = p2 - p1
    
    orig_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rot_pcd, R = rotate_pointcloud_to_xaxis(orig_pcd, origin, vec)

    # 10. Centering shift (move centroid of object to 0,0,0 relative to new frame)
    centroid_shift = np.zeros(3)
    # Simple centering:
    centroid_shift = -np.asarray(rot_pcd.points).mean(axis=0)
    rot_pcd.translate(centroid_shift, relative=True)

    # Save
    o3d.io.write_point_cloud(os.path.join(save_prefix, "rotated_full.ply"), rot_pcd)
    np.save(os.path.join(save_prefix, "rotation_matrix.npy"), R)
    np.save(os.path.join(save_prefix, "origin_point.npy"), origin)
    np.save(os.path.join(save_prefix, "centroid_shift.npy"), centroid_shift)

    return {
        "rotated_pcd": rot_pcd,
        "leg_seeds": leg_seeds,
        "leg_center_lines": leg_center_lines,
        "ground_points": ground_points,
        "rotation_matrix": R,
        "origin_point": origin,
        "centroid_shift": centroid_shift,
        "leg_clusters": clusters
    }

# ---------------------- Visualization ----------------------

def create_plotly_visualization(results):
    figs_data = []
    
    # Transforms
    origin = results.get("origin_point", np.zeros(3))
    R = results.get("rotation_matrix", np.eye(3))
    shift = results.get("centroid_shift", np.zeros(3))

    def transform(p):
        p = np.asarray(p)
        if p.ndim == 1:
            return (R @ (p - origin)) + origin + shift
        return (p - origin) @ R.T + origin + shift

    # 1. Far Mesh (Rotated)
    pcd = results.get("rotated_pcd")
    if pcd:
        pts = np.asarray(pcd.points)
        if len(pts) > 15000:
            pts = pts[np.random.choice(len(pts), 15000, replace=False)]
        figs_data.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers', marker=dict(size=1.5, color='gray', opacity=0.3),
            name="Far Mesh (Body)"
        ))

    # 2. Leg Seeds
    seeds = results.get("leg_seeds", [])
    if len(seeds) > 0:
        t_seeds = transform(np.vstack(seeds))
        figs_data.append(go.Scatter3d(
            x=t_seeds[:,0], y=t_seeds[:,1], z=t_seeds[:,2],
            mode='markers', marker=dict(size=6, color='magenta', symbol='diamond'),
            name="Leg Seeds"
        ))

    # 3. Center Lines
    lines = results.get("leg_center_lines", [])
    colors = ['green', 'blue', 'orange']
    for i, line in enumerate(lines):
        if line is None: continue
        p_s, p_e = line
        seg = transform(np.vstack([p_s, p_e]))
        col = colors[i % len(colors)]
        figs_data.append(go.Scatter3d(
            x=seg[:,0], y=seg[:,1], z=seg[:,2],
            mode='lines', line=dict(color=col, width=10),
            name=f"Leg {i+1} Axis"
        ))

    # 4. Axes
    center = origin + shift
    axis_len = 0.3
    axes = [([axis_len,0,0], 'red', 'X'), ([0,axis_len,0], 'green', 'Y'), ([0,0,axis_len], 'blue', 'Z')]
    for vec, col, name in axes:
        end = center + np.array(vec)
        figs_data.append(go.Scatter3d(
            x=[center[0], end[0]], y=[center[1], end[1]], z=[center[2], end[2]],
            mode='lines', line=dict(color=col, width=5),
            name=f"Axis {name}"
        ))

    layout = go.Layout(
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0, y=1),
        showlegend=True
    )
    return go.Figure(data=figs_data, layout=layout)

# ---------------------- Interface ----------------------

def run_interface(mesh_file, voxel_frac, low_frac, high_frac, 
                  save_dir, output_filename, ground_plane_file, 
                  ground_tol_frac, dbscan_eps_frac, dbscan_min, 
                  vert_thresh, min_clust_frac, geo_thresh, leg_off_frac):
    
    if not mesh_file: return None, "No mesh file."
    
    path = mesh_file.name
    if not save_dir: save_dir = "skeleton_out"
    
    plane = None
    if ground_plane_file:
        plane = parse_plane_from_json(ground_plane_file.name)

    log = []
    try:
        res = skeleton_main(path, low_frac, high_frac, voxel_frac, plane,
                            ground_tol_frac, dbscan_eps_frac, dbscan_min,
                            vert_thresh, min_clust_frac, geo_thresh, 
                            leg_off_frac, save_dir)
        
        log.append("Success.")
        log.append(f"Leg seeds found: {len(res.get('leg_seeds', []))}")
        log.append(f"Leg axes found: {len([l for l in res.get('leg_center_lines',[]) if l])}")
        
        fig = create_plotly_visualization(res)
        return fig, "\n".join(log)
        
    except Exception as e:
        import traceback
        return None, traceback.format_exc()

with gr.Blocks() as demo:
    gr.Markdown("## Skeleton Leg Seed Aligner")
    with gr.Row():
        with gr.Column():
            f_in = gr.File(label="Mesh")
            gp_in = gr.File(label="Ground JSON (Optional)")
            
            gr.Markdown("### Params")
            v_frac = gr.Slider(0.005, 0.05, 0.01, label="Voxel Frac")
            l_frac = gr.Slider(0.0, 0.5, 0.05, label="Low Crop")
            h_frac = gr.Slider(0.0, 1.0, 0.25, label="High Crop")
            geo_thresh = gr.Number(0.5, label="Geodesic Thresh (m)")
            leg_off = gr.Slider(0.0, 0.1, 0.02, label="Leg Seed Offset Frac")
            
            btn = gr.Button("Run")
        
        with gr.Column():
            plot = gr.Plot()
            out_log = gr.Textbox(label="Log")

    btn.click(run_interface, 
              inputs=[f_in, v_frac, l_frac, h_frac, gr.Textbox(value="out", visible=False), 
                      gr.Textbox(value="aligned.ply", visible=False), gp_in,
                      gr.Number(0.02, visible=False), gr.Number(0.015, visible=False), 
                      gr.Number(5, visible=False), gr.Number(2.0, visible=False), 
                      gr.Number(0.005, visible=False), geo_thresh, leg_off],
              outputs=[plot, out_log])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
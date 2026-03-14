#!/usr/bin/env python3
"""
skeleton_gradio_relative_pelvis_fork.py

Updated from your last script:
 - Replaces slab-based hip seeding with hybrid fork-based pelvis detector (preferred approach C).
 - Builds skeleton graph, finds junctions (degree>=3), extracts branches, computes descriptive features,
   scores forks and selects the pelvis fork. If possible it returns two hip seeds from the two longest child branches.
 - Falls back to adaptive slab / two-centroid method if fork detection fails.
 - Improves robustness mapping hip seeds to sliced voxel grid for geodesic by expanding search radius adaptively.
 - Keeps Gradio UI and Plotly visualization from your last version, returning figure + logs.

Notes:
 - This is a large self-contained script — tune parameters (weights/thresholds) to your dataset.
 - Performance: building the full skeleton graph on high-resolution voxel grids can be heavy. Reduce voxel_frac if memory/time is an issue.

Author: adapted for Vaibhav's project
"""

import os
import json
import re
import tempfile
import math
import numpy as np
import open3d as o3d
from scipy import ndimage
# unified skeletonize import (works across skimage versions)
try:
    from skimage.morphology import skeletonize as _skel_unified
    def skeletonize(grid): return _skel_unified(grid)
except Exception:
    try:
        from skimage.morphology import skeletonize_3d as _skel3d_impl
        def skeletonize(grid): return _skel3d_impl(grid)
    except Exception:
        from skimage.morphology import skeletonize as _skel_unified2
        def skeletonize(grid): return _skel_unified2(grid)

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import gradio as gr
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import plotly.graph_objs as go

# ---------------------- Utils & parsing ----------------------

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
    if isinstance(data, dict) and "plane_equation" in data:
        pe = data["plane_equation"]
        out = _from_list_like(pe)
        if out is not None:
            return out
        if isinstance(pe, dict):
            if all(k in pe for k in ("a","b","c","d")):
                try:
                    return (_to_float_loose(pe["a"]), _to_float_loose(pe["b"]), _to_float_loose(pe["c"]), _to_float_loose(pe["d"]))
                except Exception:
                    pass
            if "normal" in pe and ("d" in pe or "D" in pe):
                try:
                    nx = _to_float_loose(pe["normal"][0]); ny = _to_float_loose(pe["normal"][1]); nz = _to_float_loose(pe["normal"][2])
                    dd = _to_float_loose(pe.get("d", pe.get("D")))
                    return (nx, ny, nz, dd)
                except Exception:
                    pass
            vals = []
            for v in pe.values():
                try:
                    vals.append(_to_float_loose(v))
                except Exception:
                    continue
            if len(vals) >= 4:
                return tuple(vals[:4])
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
    if isinstance(data, dict) and all(k in data for k in ("a","b","c","d")):
        try:
            return (_to_float_loose(data["a"]), _to_float_loose(data["b"]), _to_float_loose(data["c"]), _to_float_loose(data["d"]))
        except Exception:
            pass
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

# ---------------------- Geometry helpers ----------------------

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
    # Clip indices to grid
    indices[:,0] = np.clip(indices[:,0], 0, dims[0]-1)
    indices[:,1] = np.clip(indices[:,1], 0, dims[1]-1)
    indices[:,2] = np.clip(indices[:,2], 0, dims[2]-1)
    grid = np.zeros(dims, dtype=np.uint8)
    grid[indices[:,0], indices[:,1], indices[:,2]] = 1
    return grid, min_b, voxel_size

def grid_indices_to_points(indices, min_b, voxel_size):
    return indices * voxel_size + min_b + voxel_size * 0.5

def world_to_voxel_index(pt_world, min_b, voxel_size, grid_shape):
    idx = np.floor((pt_world - min_b) / voxel_size).astype(int)
    idx[0] = np.clip(idx[0], 0, grid_shape[0]-1)
    idx[1] = np.clip(idx[1], 0, grid_shape[1]-1)
    idx[2] = np.clip(idx[2], 0, grid_shape[2]-1)
    return tuple(idx)

# ---------------------- Old hip estimator (fallback) ----------------------

def estimate_hip_seeds_from_pcd_slab(pts, min_b, max_b, h, voxel_size):
    """
    Old slab-based centroid fallback. Kept as fallback if fork-based fails.
    """
    low_y = min_b[1] + 0.45 * h
    high_y = min_b[1] + 0.65 * h
    slab_mask = (pts[:,1] >= low_y) & (pts[:,1] <= high_y)
    slab_pts = pts[slab_mask]
    if slab_pts.shape[0] == 0:
        return []
    try:
        db = DBSCAN(eps=max(0.03, 0.6 * voxel_size), min_samples=10).fit(slab_pts)
        labels = db.labels_
        unique = [l for l in np.unique(labels) if l != -1]
        if len(unique) == 0:
            return [slab_pts.mean(axis=0)]
        best = sorted(unique, key=lambda l: -np.sum(labels==l))
        seeds = []
        for l in best[:2]:
            seeds.append(slab_pts[labels==l].mean(axis=0))
        return seeds
    except Exception:
        return [slab_pts.mean(axis=0)]

# ---------------------- Skeleton graph & branch extraction ----------------------

def build_skeleton_graph_from_grid(skel_grid):
    """
    Build adjacency dict for skeleton voxels.
    Input: skel_grid - 3D binary numpy array (1=voxel present).
    Returns:
      - nodes: list of (x,y,z) tuples
      - neighbors: dict node -> list of neighbor nodes (6- or 26-neighborhood used)
      - degrees: dict node -> degree
    """
    occ = np.argwhere(skel_grid > 0)
    nodes = [tuple(x) for x in occ]
    node_set = set(nodes)
    neighbors = {}
    # use 26-neighborhood for connectivity (makes branches more continuous)
    neigh_offsets = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if not (dx==0 and dy==0 and dz==0)]
    for n in nodes:
        nb = []
        x,y,z = n
        for dx,dy,dz in neigh_offsets:
            cand = (x+dx, y+dy, z+dz)
            if cand in node_set:
                nb.append(cand)
        neighbors[n] = nb
    degrees = {n: len(neighbors[n]) for n in nodes}
    return nodes, neighbors, degrees

def extract_branches_from_fork(fork_node, neighbors, degrees, max_walk=5000):
    """
    For a fork node, follow each neighbor until hitting:
      - an endpoint (degree==1), or
      - another fork (degree>=3), or
      - max_walk steps
    Returns list of branch node-lists, each branch includes fork_node as first element.
    """
    branches = []
    for nb in neighbors[fork_node]:
        branch = [fork_node, nb]
        prev = fork_node
        cur = nb
        steps = 0
        while True:
            steps += 1
            if steps > max_walk:
                break
            d = degrees.get(cur, 0)
            if d == 0:
                break
            # stop if endpoint or another fork
            if d == 1 or d >= 3:
                break
            # else continue to the next neighbor that's not previous
            next_nodes = [n for n in neighbors[cur] if n != prev]
            if not next_nodes:
                break
            prev, cur = cur, next_nodes[0]
            branch.append(cur)
        branches.append(branch)
    return branches

# ---------------------- Fork features & scoring ----------------------

def compute_branch_length_world(branch_nodes, min_b, voxel_size):
    pts = grid_indices_to_points(np.array(branch_nodes), min_b, voxel_size)
    if len(pts) < 2:
        return 0.0
    # approximate geodesic length by chain distances
    dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return dists.sum()

def branch_verticality(branch_nodes, min_b, voxel_size):
    pts = grid_indices_to_points(np.array(branch_nodes), min_b, voxel_size)
    if len(pts) < 3:
        return 0.0
    try:
        pca = PCA(n_components=1)
        # primary direction from PCA on (x,y,z)
        pca.fit(pts)
        dir_vec = pca.components_[0]
        vertical_unit = np.array([0.0, 1.0, 0.0])
        return abs(np.dot(dir_vec, vertical_unit))
    except Exception:
        return 0.0

def branch_endpoint_world(branch_nodes, min_b, voxel_size):
    pts = grid_indices_to_points(np.array(branch_nodes), min_b, voxel_size)
    return pts[-1]

def local_volume_around_voxel(binary_grid, voxel_idx, radius_vox):
    # count occupied voxels within manhattan/cubic radius
    x,y,z = voxel_idx
    mins = np.maximum(np.array([x,y,z]) - radius_vox, 0)
    maxs = np.minimum(np.array([x,y,z]) + radius_vox, np.array(binary_grid.shape)-1)
    sub = binary_grid[mins[0]:maxs[0]+1, mins[1]:maxs[1]+1, mins[2]:maxs[2]+1]
    return int(sub.sum())

def score_fork_candidate(fork_node, branches, min_b, voxel_size, binary_grid_full, H, ground_plane=None, torso_center_xz=None):
    """
    Compute a composite score for fork candidate. Higher = more likely pelvis fork.
    Returns (score, feature_dict)
    """
    # features
    Ls = []
    verts = []
    endpoints = []
    masses = []
    for br in branches:
        L = compute_branch_length_world(br, min_b, voxel_size)
        Ls.append(L)
        verts.append(branch_verticality(br, min_b, voxel_size))
        endpoints.append(branch_endpoint_world(br, min_b, voxel_size))
        masses.append(len(br))
    Ls = np.array(Ls) if Ls else np.array([0.0])
    verts = np.array(verts) if verts else np.array([0.0])
    masses = np.array(masses) if masses else np.array([0])
    # branch-based metrics
    L_med = np.median(Ls)
    L_max = Ls.max() if Ls.size>0 else 0.0
    verticality_mean = float(np.mean(verts))
    # ground proximity: fraction of branches whose endpoint is near ground
    ground_count = 0
    ground_tol = max(0.02 * H, 0.01 * H)
    if ground_plane is None:
        # use global minimum Y
        occ = np.argwhere(binary_grid_full > 0)
        ground_y = (occ * voxel_size + (min_b + voxel_size*0.5))[:,1].min() if occ.size>0 else None
        def is_near_ground(pt): 
            if ground_y is None: return False
            return (pt[1] - ground_y) <= (ground_tol + 1e-9)
    else:
        a,b,c,d = ground_plane
        def is_near_ground(pt):
            dist = abs(a*pt[0] + b*pt[1] + c*pt[2] + d) / np.sqrt(a*a + b*b + c*c)
            return dist <= (ground_tol + 1e-9)
    for e in endpoints:
        if is_near_ground(e):
            ground_count += 1
    ground_frac = ground_count / float(max(1, len(endpoints)))
    # local volume (voxels) around fork
    fork_vox = tuple(fork_node)
    radius_vox = max(1, int(round(0.03 * H / voxel_size)))
    local_vol = local_volume_around_voxel(binary_grid_full, fork_vox, radius_vox)
    # torso proximity (XZ)
    fork_world = grid_indices_to_points(np.array([fork_node]), min_b, voxel_size)[0]
    if torso_center_xz is None:
        torso_center_xz = np.array([0.0, 0.0])
    
    # Split torso distance into X (lateral) and Z (depth)
    # Assuming torso_center_xz is [mean_x, mean_z]
    dx = abs(fork_world[0] - torso_center_xz[0])
    dz = abs(fork_world[2] - torso_center_xz[1])
    d_torso_lat = dx # lateral offset from midline
    
    # symmetry: if at least two long branches have roughly symmetric xz positions
    # compute centroids of two largest branches
    sorted_idx = np.argsort(-Ls)
    sym_score = 0.0
    if Ls.size >= 2 and np.isfinite(Ls[sorted_idx[0]]) and np.isfinite(Ls[sorted_idx[1]]):
        c0 = grid_indices_to_points(np.array(branches[sorted_idx[0]]), min_b, voxel_size).mean(axis=0)
        c1 = grid_indices_to_points(np.array(branches[sorted_idx[1]]), min_b, voxel_size).mean(axis=0)
        lateral_sep = np.linalg.norm((c0 - c1)[[0,2]])
        sym_score = lateral_sep
    
    # mass score: sum of top-2 branch mass
    top2mass = masses[sorted_idx[:2]].sum() if masses.size>=2 else masses.sum()

    # normalize features
    # Dist-to-ground: smaller is better -> ground_score in [0,1]
    # Use L_med normalized by H
    feat = {}
    feat['L_med'] = L_med
    feat['L_max'] = L_max
    feat['verticality_mean'] = verticality_mean
    feat['ground_frac'] = ground_frac
    feat['local_vol'] = local_vol
    feat['d_torso_lat'] = d_torso_lat
    feat['sym_score'] = sym_score
    feat['top2mass'] = top2mass

    # normalization helpers
    def norm_clip(x, scale):
        return float(np.clip(x / (scale + 1e-12), 0.0, 1.0))

    L_score = norm_clip(L_med, 0.25 * H)  # median branch longer -> better
    vert_score = norm_clip(verticality_mean, 1.0)  # 0..1
    ground_score = ground_frac  # already 0..1
    vol_score = norm_clip(local_vol, 200.0)  # depends on grid size; 200 voxels chosen as heuristic
    
    # Penalize lateral offset from torso center (hands are lateral, pelvis is central)
    lat_score = 1.0 - norm_clip(d_torso_lat, 0.25 * H) 
    
    sym_score_n = norm_clip(sym_score, 0.2 * H)
    mass_score = norm_clip(top2mass, 500.0)

    # weight sum
    w = {
        'ground': 2.0,      # Highest weight as per spec
        'length': 1.2,
        'vertical': 1.2,    # Hands are often non-vertical
        'sym': 1.0,
        'vol': 0.8,         # Pelvis is bulky
        'lat': 1.0,         # Prefer central
        'mass': 0.7
    }
    
    score = (w['ground']*ground_score + 
             w['length']*L_score + 
             w['vertical']*vert_score +
             w['sym']*sym_score_n + 
             w['vol']*vol_score + 
             w['lat']*lat_score + 
             w['mass']*mass_score)

    return float(score), feat

# ---------------------- Pelvis fork detector (hybrid strategy C) ----------------------

def detect_pelvis_fork_and_hip_seeds(points, voxel_frac, H, ground_plane=None, debug_save_prefix=None):
    """
    Detect pelvis fork on full point cloud skeleton and return hip seed(s) in world coordinates.
    Hybrid strategy:
      - Build coarse voxelization of full body
      - Skeletonize, build graph, find fork nodes (degree >= 3)
      - For each fork compute features and score
      - pick best fork; try to extract two hip seeds from its two largest child branches
      - fallback: adaptive slab + two-centroid method
    """
    # Voxelization on full body (coarser than sliced voxel to save mem)
    voxel_size_full = max(voxel_frac * H, 0.005 * H)  # ensure minimum relative resolution
    binary_full, min_b_full, vs_full = voxelize_pointcloud_to_grid(points, voxel_size_full, padding=2)
    # compute skeleton of full binary volume
    try:
        skel_full = skeletonize(binary_full.astype(bool)).astype(np.uint8)
    except Exception:
        # fallback: morphological skeletonization 3d could be slow. If fails, return empty.
        skel_full = skeletonize(binary_full.astype(bool)).astype(np.uint8)
    # build graph
    nodes, neighbors, degrees = build_skeleton_graph_from_grid(skel_full)
    forks = [n for n,d in degrees.items() if d >= 3]
    torso_center_xz = np.array([(points[:,0].mean()), (points[:,2].mean())])
    candidates = []
    # compute scores
    for f in forks:
        branches = extract_branches_from_fork(f, neighbors, degrees, max_walk=2000)
        score, feat = score_fork_candidate(f, branches, min_b_full, vs_full, binary_full, H, ground_plane=ground_plane, torso_center_xz=torso_center_xz)
        candidates.append((score, f, branches, feat))
    # choose best candidate
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        best_score, best_f, best_branches, best_feat = candidates[0]
        # pick two hip seeds if possible: take two longest branches from best_f and choose points along them near fork (e.g., 10%-30% down branch)
        Ls = [compute_branch_length_world(br, min_b_full, vs_full) for br in best_branches]
        idx_sorted = np.argsort(-np.array(Ls))
        hip_seeds = []
        
        # Strategy: try to get 2 seeds from top 2 branches
        if len(idx_sorted) >= 2 and Ls[idx_sorted[0]] > 0 and Ls[idx_sorted[1]] > 0:
            for k in idx_sorted[:2]:
                br = best_branches[k]
                L_br = Ls[k]
                # t = clamp(0.06*H, 0.02*H..0.3*branch_length)
                target_len = max(0.02 * H, min(0.06 * H, 0.3 * L_br))
                
                # walk branch cumulative distances and pick corresponding node
                pts = grid_indices_to_points(np.array(br), min_b_full, vs_full)
                dists = np.concatenate([[0.0], np.linalg.norm(pts[1:] - pts[:-1], axis=1).cumsum()])
                # find index where dist >= target_len
                idx = int(np.searchsorted(dists, target_len))
                idx = np.clip(idx, 0, len(br)-1)
                seed_world = pts[idx]
                hip_seeds.append(seed_world)
            
            # ensure two seeds are distinct and separated enough
            if np.linalg.norm(hip_seeds[0] - hip_seeds[1]) < 0.02 * H:
                # too close; fallback to symmetric offset from fork
                hip_seeds = [] # reset to trigger fallback below

        if len(hip_seeds) < 2:
            # Fallback: Symmetric pair fallback
            # If only one long branch or seeds too close, produce symmetric pair around fork
            centroid = grid_indices_to_points(np.array([best_f]), min_b_full, vs_full)[0]
            offset_dist = 0.02 * H # 0.01-0.03 * H
            hip_seeds = [centroid + np.array([offset_dist, 0, 0]), centroid + np.array([-offset_dist, 0, 0])]
        # Optionally save debug volumes/points
        if debug_save_prefix:
            np.save(os.path.join(debug_save_prefix, "skel_full.npy"), skel_full)
            np.save(os.path.join(debug_save_prefix, "fork_candidates.npy"), np.array([(c[0], c[1]) for c in candidates], dtype=object))
        return hip_seeds, best_score, best_f, best_feat, (binary_full, min_b_full, vs_full, skel_full)
    # fallback: two-centroid adaptive slab
    hip_seeds_slab = fallback_two_centroids(points, H, voxel_frac)
    return hip_seeds_slab, 0.0, None, {}, (binary_full, min_b_full, vs_full, None)

def fallback_two_centroids(points, H, voxel_frac):
    """
    If fork detection fails, search adaptively for a horizontal slab with two lateral peaks.
    Scan slab centers from 0.35H to 0.75H and pick slab with two-peaks in XZ density, then kmeans with k=2.
    """
    min_b, max_b = axis_aligned_bounds(points)
    best_seeds = []
    best_score = -1.0
    for center_frac in np.linspace(0.35, 0.75, 9):
        slab_center_y = min_b[1] + center_frac * H
        slab_h = 0.12 * H
        slab_mask = (points[:,1] >= slab_center_y - slab_h/2) & (points[:,1] <= slab_center_y + slab_h/2)
        slab_pts = points[slab_mask]
        if slab_pts.shape[0] < 50:
            continue
        # project to XZ and do coarse DBSCAN to find two large clusters
        try:
            db = DBSCAN(eps=max(0.02*H, 0.6*voxel_frac*H), min_samples=8).fit(slab_pts[:, [0,2]])
            labels = db.labels_
            unique = [l for l in np.unique(labels) if l != -1]
            if len(unique) >= 2:
                # pick top-2 by size
                best = sorted(unique, key=lambda l: -np.sum(labels==l))[:2]
                seeds = [slab_pts[labels==best[0]].mean(axis=0), slab_pts[labels==best[1]].mean(axis=0)]
                sep = np.linalg.norm((seeds[0] - seeds[1])[[0,2]])
                # score by separation
                if sep > best_score:
                    best_score = sep
                    best_seeds = seeds
        except Exception:
            continue
    if not best_seeds:
        # last resort centroid
        return [points.mean(axis=0)]
    return best_seeds

# ---------------------- Geodesic computation (with adaptive search radius) ----------------------

def compute_geodesic_from_hips(binary_grid, min_b, voxel_size, hip_world_points, max_nodes_for_graph=350_000):
    """
    Compute geodesic shortest-path distances (meters) from hip seeds to all occupied voxels.
    If mapping hip_world_points to voxel indices fails, expand search radius adaptively.
    """
    occ_idx = np.argwhere(binary_grid > 0)
    n_occ = len(occ_idx)
    if n_occ == 0:
        return np.full(binary_grid.shape, np.inf, dtype=float), []

    id_of = {tuple(v): i for i, v in enumerate(map(tuple, occ_idx))}
    centers = (occ_idx * voxel_size) + (min_b + voxel_size*0.5)

    rows = []
    cols = []
    data = []
    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for i, v in enumerate(occ_idx):
        vx, vy, vz = v
        for dx,dy,dz in neigh:
            nb = (vx+dx, vy+dy, vz+dz)
            jid = id_of.get(nb, None)
            if jid is not None:
                w = np.linalg.norm(centers[i] - centers[jid])
                rows.append(i); cols.append(jid); data.append(w)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_occ, n_occ))

    seed_ids = []
    # adaptive search radius in voxels (depends on H/voxel_size)
    # allow search up to 10% of H in voxels
    max_search_vox = max(5, int(round(0.1 * (np.max(binary_grid.shape) ))))
    for hw in hip_world_points:
        vi = world_to_voxel_index(np.array(hw), min_b, voxel_size, binary_grid.shape)
        found_id = None
        # first try exact mapping
        if tuple(vi) in id_of:
            seed_ids.append(id_of[tuple(vi)])
            continue
        # expand search radius
        for r in range(1, max_search_vox+1):
            mins = np.maximum(np.array(vi) - r, 0)
            maxs = np.minimum(np.array(vi) + r, np.array(binary_grid.shape)-1)
            candidates = []
            for x in range(mins[0], maxs[0]+1):
                for y in range(mins[1], maxs[1]+1):
                    for z in range(mins[2], maxs[2]+1):
                        if binary_grid[x,y,z]:
                            candidates.append((x,y,z))
            if candidates:
                sub_pts = np.array(candidates)
                sub_centers = sub_pts * voxel_size + (min_b + voxel_size*0.5)
                dists = np.linalg.norm(sub_centers - hw, axis=1)
                chosen = candidates[np.argmin(dists)]
                found_id = id_of.get(tuple(chosen), None)
                if found_id is not None:
                    seed_ids.append(found_id)
                    break
        # if not found, ignore this seed
    if len(seed_ids) == 0:
        return np.full(binary_grid.shape, np.inf, dtype=float), []

    dist_matrix = dijkstra(csgraph=A, directed=False, indices=seed_ids, return_predecessors=False)
    if dist_matrix.ndim == 1:
        dist_from_seeds = dist_matrix
    else:
        dist_from_seeds = np.min(dist_matrix, axis=0)

    distances_grid = np.full(binary_grid.shape, np.inf, dtype=float)
    for i, v in enumerate(occ_idx):
        distances_grid[tuple(v)] = dist_from_seeds[i]

    return distances_grid, seed_ids

# ---------------------- Ground-connected pruning with geodesic (unchanged) ----------------------

def keep_components_touching_ground_with_geodesic(binary_grid, min_b, voxel_size,
                                                  ground_plane=None, ground_tol=0.01, height=None,
                                                  distances_grid=None, geodesic_thresh=0.5,
                                                  verticality_thresh=2.0):
    structure = np.ones((3,3,3), dtype=int)
    labeled, ncomp = ndimage.label(binary_grid, structure=structure)
    if ncomp == 0:
        return binary_grid

    xs, ys, zs = np.where(binary_grid==1)
    coords = np.vstack([xs, ys, zs]).T
    voxel_centers = grid_indices_to_points(coords, min_b, voxel_size)

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
        if idxs.size == 0:
            continue
        centers_lab = grid_indices_to_points(idxs, min_b, voxel_size)
        size = centers_lab.shape[0]
        
        # Check ground proximity
        near_ground = touching(centers_lab[np.argmin(centers_lab[:,1])])
        
        # Check verticality
        verticality = 0.0
        try:
            pca = PCA(n_components=3)
            pca.fit(centers_lab)
            eigs = pca.explained_variance_
            eigs_sorted = np.sort(eigs)[::-1]
            verticality = (eigs_sorted[0] / (eigs_sorted[1] + 1e-9)) if eigs_sorted[1] > 0 else 999.0
        except Exception:
            verticality = 0.0

        # Check geodesic distance
        min_geo = np.inf
        if distances_grid is not None:
            try:
                mins = []
                for v in idxs:
                    t = tuple(v)
                    dval = distances_grid[t]
                    mins.append(dval)
                if len(mins) > 0:
                    min_geo = float(np.min(mins))
            except Exception:
                min_geo = np.inf

        keep = False
        if near_ground and (min_geo <= geodesic_thresh):
            keep = True
        if near_ground and (verticality >= verticality_thresh):
            keep = True
        if near_ground and (size > max(100, int(0.01 * np.prod(binary_grid.shape)))):
            keep = True

        if keep:
            keep_labels.add(lab)

    kept_mask = np.isin(labeled, list(keep_labels))
    pruned = np.zeros_like(binary_grid)
    pruned[kept_mask] = 1
    return pruned

# ---------------------- Skeletonization helpers ----------------------

def run_skeletonization(binary_grid):
    bool_grid = (binary_grid > 0)
    skeleton = skeletonize(bool_grid).astype(np.uint8)
    return skeleton

def skeleton_to_points(skeleton_grid, min_b, voxel_size):
    xs, ys, zs = np.where(skeleton_grid > 0)
    pts = grid_indices_to_points(np.vstack([xs, ys, zs]).T, min_b, voxel_size)
    return pts

# ---------------------- Clustering & filtering (unchanged) ----------------------

def cluster_skeleton_points_filtered(pts, eps, min_samples, n_clusters_expected, ground_plane=None,
                                     ground_tol=0.01, height=None, verticality_thresh=2.0,
                                     min_cluster_frac=0.005, torso_center=None, torso_radius_frac=0.6,
                                     required_ground_fraction=0.05):
    if len(pts) == 0:
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = clustering.labels_
    unique = [lab for lab in np.unique(labels) if lab != -1]
    clusters_info = []
    total_pts = max(1, len(pts))

    if ground_plane is None:
        ground_y = pts[:,1].min()
        def is_near_ground(p): return (p[1] - ground_y) <= (ground_tol + 1e-9)
    else:
        a,b,c,d = ground_plane
        def is_near_ground(p):
            dist = abs(a*p[0] + b*p[1] + c*p[2] + d) / np.sqrt(a*a + b*b + c*c)
            return dist <= (ground_tol + 1e-9)

    if torso_center is None:
        torso_center = np.array([np.mean(pts[:,0]), np.mean(pts[:,2])])
    torso_radius = (torso_radius_frac * height) if height is not None else None

    for lab in unique:
        cpts = pts[labels == lab]
        size = len(cpts)
        if size < max(3, int(min_cluster_frac * total_pts)):
            continue
        try:
            pca = PCA(n_components=3).fit(cpts)
            eigs = np.sort(pca.explained_variance_)[::-1]
            verticality = (eigs[0] / (eigs[1] + 1e-9))
        except Exception:
            verticality = 0.0
        centroid = cpts.mean(axis=0)
        min_y = cpts[:,1].min()
        near_ground_count = np.count_nonzero([is_near_ground(p) for p in cpts])
        ground_fraction = near_ground_count / float(size)
        dx = centroid[0] - torso_center[0]
        dz = centroid[2] - torso_center[1]
        dist_xz = np.sqrt(dx*dx + dz*dz)
        clusters_info.append({
            "lab": lab, "pts": cpts, "size": size, "verticality": verticality,
            "centroid": centroid, "min_y": min_y, "ground_fraction": ground_fraction,
            "dist_xz": dist_xz
        })

    if not clusters_info:
        return []

    def score_cluster(c):
        g_bonus = 2.0 if c["ground_fraction"] >= required_ground_fraction else 0.0
        vscore = min(c["verticality"], 10.0) / 10.0
        ssize = min(c["size"] / max(1, total_pts), 0.5) / 0.5
        prox = 1.0
        if torso_radius is not None:
            prox = max(0.0, 1.0 - (c["dist_xz"] / (torso_radius + 1e-9)))
        return (g_bonus * 1.2) + (0.8 * vscore) + (0.6 * ssize) + (0.6 * prox)

    clusters_info.sort(key=lambda x: score_cluster(x), reverse=True)

    selected = []
    for c in clusters_info:
        if len(selected) >= n_clusters_expected:
            break
        if (c["ground_fraction"] >= required_ground_fraction) or (c["verticality"] >= verticality_thresh):
            if (torso_radius is None) or (c["dist_xz"] <= max(2.0 * torso_radius, torso_radius + 1e-9)):
                selected.append(c["pts"])

    if len(selected) < n_clusters_expected:
        for c in clusters_info:
            if c["pts"] in selected: continue
            selected.append(c["pts"])
            if len(selected) >= n_clusters_expected: break

    if len(selected) >= 2 and height is not None:
        pcs = [np.array(s).mean(axis=0) for s in selected[:2]]
        lateral_sep = np.linalg.norm((pcs[0] - pcs[1])[[0,2]])
        min_sep = max(0.02 * height, 0.01)
        if lateral_sep < min_sep:
            alt = None
            for c in clusters_info:
                if any(np.allclose(c["pts"], s) for s in selected): continue
                cand_centroid = c["centroid"]
                cand_sep = np.linalg.norm((cand_centroid - pcs[0])[[0,2]])
                if cand_sep >= min_sep:
                    alt = c["pts"]
                    break
            if alt is not None:
                selected[1] = alt

    return [np.array(s) for s in selected[:n_clusters_expected]]

# ---------------------- Line fit, intersection, rotation (unchanged) ----------------------

def fit_line_through_points(pts):
    if len(pts) < 2:
        return None, None
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    dir_vec = eigvecs[:, -1]
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    return mean, dir_vec

def line_plane_intersection(point_on_line, dir_vec, plane):
    a,b,c,d = plane
    n = np.array([a,b,c], dtype=float)
    denom = n.dot(dir_vec)
    if abs(denom) < 1e-8:
        return None
    t = -(n.dot(point_on_line) + d) / denom
    return point_on_line + t * dir_vec

def rotate_pointcloud_to_xaxis(pcd, origin_point, target_vector):
    v = np.array(target_vector, dtype=float)
    v = v / np.linalg.norm(v)
    target = np.array([1.0, 0.0, 0.0])
    cross = np.cross(v, target)
    dot = np.dot(v, target)
    if np.linalg.norm(cross) < 1e-8:
        if dot < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.pi * np.array([0,1,0]))
        else:
            R = np.eye(3)
    else:
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    pcd_copy = pcd.translate(-origin_point, relative=False)
    pcd_copy.rotate(R, center=(0,0,0))
    pcd_copy.translate(origin_point, relative=False)
    return pcd_copy, R

# ---------------------- Main pipeline (integrated changes) ----------------------

def estimate_ground_plane(pcd, ransac_n=3, dist_threshold=0.01, n_iterations=2000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=n_iterations)
    return tuple(plane_model), inliers

def skeleton_main(pcd_path,
                  low_frac=0.05, high_frac=0.25,
                  voxel_frac=0.01,
                  ground_plane=None, ground_tol_frac=0.02,
                  dbscan_eps_frac=0.015, dbscan_min_samples=5,
                  verticality_thresh=2.0, min_cluster_frac=0.005,
                  geodesic_thresh=0.5,
                  save_prefix="out",
                  debug=False,
                  use_pelvis_fork=True):
    os.makedirs(save_prefix, exist_ok=True)

    points = load_mesh_or_cloud(pcd_path)
    min_b, max_b = axis_aligned_bounds(points)
    H = max_b[1] - min_b[1]
    if H <= 0:
        raise RuntimeError("Invalid height H.")

    # Auto-estimate ground plane if not provided
    if ground_plane is None:
        try:
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(points)
            gp, _ = estimate_ground_plane(pcd_temp, dist_threshold=ground_tol_frac * H)
            ground_plane = gp
            print(f"Auto-estimated ground plane: {ground_plane}")
        except Exception as e:
            print(f"Ground plane estimation failed: {e}; proceeding without explicit plane.")

    # Convert relative fractions to absolute values
    voxel_size = voxel_frac * H
    ground_tol = ground_tol_frac * H
    dbscan_eps = dbscan_eps_frac * H

    # Feet-relative slicing: assume feet ~ maxY
    y_feet = max_b[1]
    y1 = y_feet - (low_frac * H)   # upper (closer to feet)
    y2 = y_feet - (high_frac * H)  # lower (towards knees)
    if y2 > y1:
        y1, y2 = y2, y1

    mask = (points[:,1] <= y1) & (points[:,1] >= y2)
    sliced_pts = points[mask]
    if sliced_pts.shape[0] == 0:
        raise RuntimeError("Slice produced zero points. Try changing fractions or check axis.")
    sliced_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sliced_pts))
    o3d.io.write_point_cloud(os.path.join(save_prefix, "sliced.ply"), sliced_pcd)

    # Pelvis fork detection (hybrid)
    hip_seeds = []
    fork_score = 0.0
    best_f_node = None
    best_feat = {}
    
    if use_pelvis_fork:
        hip_seeds, fork_score, best_f_node, best_feat, debug_vol = detect_pelvis_fork_and_hip_seeds(points, voxel_frac, H, ground_plane=ground_plane, debug_save_prefix=save_prefix if debug else None)
        if debug:
            print("Pelvis detection fork_score:", fork_score, "best_node:", best_f_node, "features:", best_feat)
            print("Hip seeds (world):", hip_seeds)
    else:
        if debug: print("Skipping fork detection (use_pelvis_fork=False). Using fallback slab.")
        hip_seeds = fallback_two_centroids(points, H, voxel_frac)

    # Voxelize sliced region (for geodesic pruning we operate on the ROI)
    binary_grid, min_grid_b, voxel_size_used = voxelize_pointcloud_to_grid(sliced_pts, voxel_size=voxel_size)
    # Geodesic Pruning: compute distances from hip seeds, mapping adaptively
    if len(hip_seeds) == 0:
        hip_seeds = [points.mean(axis=0)] # Fallback
    distances_grid, _ = compute_geodesic_from_hips(binary_grid, min_grid_b, voxel_size_used, hip_seeds)

    pruned = keep_components_touching_ground_with_geodesic(
        binary_grid, min_grid_b, voxel_size_used,
        ground_plane=ground_plane, ground_tol=ground_tol, height=H,
        distances_grid=distances_grid, geodesic_thresh=geodesic_thresh,
        verticality_thresh=verticality_thresh
    )

    skeleton_grid = run_skeletonization(pruned)
    skel_pts = skeleton_to_points(skeleton_grid, min_grid_b, voxel_size_used)
    if len(skel_pts) == 0:
        raise RuntimeError("No skeleton points extracted. Try larger voxel_frac or larger ground_tol_frac.")

    o3d.io.write_point_cloud(os.path.join(save_prefix, "skeleton_points_vis.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(skel_pts)))

    clusters = cluster_skeleton_points_filtered(skel_pts,
                                               eps=dbscan_eps,
                                               min_samples=max(2, int(dbscan_min_samples)),
                                               n_clusters_expected=2,
                                               ground_plane=ground_plane,
                                               ground_tol=ground_tol,
                                               height=H,
                                               verticality_thresh=verticality_thresh,
                                               min_cluster_frac=min_cluster_frac,
                                               torso_center=np.array([(min_b[0]+max_b[0])/2.0, (min_b[2]+max_b[2])/2.0]),
                                               torso_radius_frac=0.6,
                                               required_ground_fraction=0.05)
    # Fit lines and intersect with ground
    ground_points = []
    line_params = []
    if len(clusters) >= 2:
        for cpts in clusters[:2]:
            mean, dir_vec = fit_line_through_points(cpts)
            if mean is None: continue
            inter = None
            if ground_plane is not None:
                inter = line_plane_intersection(mean, dir_vec, ground_plane)
            if inter is None:
                if ground_plane is not None and abs(ground_plane[1]) > 1e-8:
                    a,b,c,d = ground_plane
                    gy = -(a*mean[0] + c*mean[2] + d) / b
                    inter = np.array([mean[0], gy, mean[2]])
                else:
                    inter = mean.copy()
            line_params.append((mean, dir_vec))
            ground_points.append(inter)
    else:
        # fallback: use extremes from skeleton points
        if len(skel_pts) < 2:
            raise RuntimeError("Not enough skeleton points for fallback.")
        pca = PCA(n_components=2)
        pca.fit(skel_pts[:, [0,2]])
        proj = (skel_pts[:, [0,2]] - pca.mean_).dot(pca.components_[0])
        pmin = skel_pts[np.argmin(proj)]
        pmax = skel_pts[np.argmax(proj)]
        if ground_plane is not None:
            a,b,c,d = ground_plane
            if abs(b) > 1e-8:
                gy1 = -(a*pmin[0] + c*pmin[2] + d)/b
                gy2 = -(a*pmax[0] + c*pmax[2] + d)/b
                ground_points = [np.array([pmin[0], gy1, pmin[2]]), np.array([pmax[0], gy2, pmax[2]])]
            else:
                ground_points = [pmin, pmax]
        else:
            ground_points = [pmin, pmax]

    if len(ground_points) < 2:
        raise RuntimeError("Could not find two ground intersection points. Try loosening clustering or change slice.")

    gpts = np.vstack(ground_points)
    if gpts.shape[0] > 2:
        dists = np.sum((gpts[:,None,:] - gpts[None,:,:])**2, axis=-1)
        i,j = np.unravel_index(np.argmax(dists), dists.shape)
        p1, p2 = gpts[i], gpts[j]
    else:
        p1, p2 = gpts[0], gpts[1]

    vector = p2 - p1
    vnorm = np.linalg.norm(vector)
    if vnorm < 1e-9:
        raise RuntimeError("Zero vector between ground points; invalid geometry.")
    vector = vector / vnorm
    origin_point = (p1 + p2) / 2.0

    # rotate original full cloud
    original_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotated_pcd, R = rotate_pointcloud_to_xaxis(original_pcd, origin_point, vector)

    # --- NEW: Shift origin to centroid of non-ground mesh ---
    centroid_shift = np.zeros(3)
    if ground_plane is not None:
        a, b, c, d = ground_plane
        denom = np.sqrt(a*a + b*b + c*c)
        dists = np.abs(points @ np.array([a,b,c]) + d) / denom
        non_ground_mask = dists > ground_tol
        
        if np.sum(non_ground_mask) > 0:
            rot_pts = np.asarray(rotated_pcd.points)
            subset = rot_pts[non_ground_mask]
            centroid_rot = subset.mean(axis=0)
            centroid_shift = -centroid_rot
        else:
            centroid_shift = -np.asarray(rotated_pcd.points).mean(axis=0)
    else:
        centroid_shift = -np.asarray(rotated_pcd.points).mean(axis=0)

    rotated_pcd.translate(centroid_shift, relative=True)

    # Save outputs
    o3d.io.write_point_cloud(os.path.join(save_prefix, "rotated_full.ply"), rotated_pcd)
    np.save(os.path.join(save_prefix, "skeleton_points.npy"), skel_pts)
    np.save(os.path.join(save_prefix, "ground_points.npy"), np.vstack(ground_points))
    np.save(os.path.join(save_prefix, "rotation_matrix.npy"), R)
    np.save(os.path.join(save_prefix, "origin_point.npy"), origin_point)
    np.save(os.path.join(save_prefix, "centroid_shift.npy"), centroid_shift)

    return {
        "sliced_pcd": sliced_pcd,
        "rotated_pcd": rotated_pcd,
        "skeleton_points": skel_pts,
        "ground_points": np.vstack(ground_points),
        "rotation_matrix": R,
        "origin_point": origin_point,
        "centroid_shift": centroid_shift,
        "save_prefix": save_prefix,
        "leg_clusters": clusters,
        "hip_seeds": hip_seeds,
        "line_params": line_params,
        "fork_detection": {
            "score": float(fork_score),
            "best_node": best_f_node,
            "best_feat": best_feat
        }
    }

# ---------------------- Visualization helpers (Plotly) ----------------------

def create_plotly_visualization(results):
    figs_data = []
    rotated_pcd = results.get("rotated_pcd")
    if rotated_pcd is not None:
        pts = np.asarray(rotated_pcd.points)
        if len(pts) > 15000:
            indices = np.random.choice(len(pts), 15000, replace=False)
            pts = pts[indices]
        figs_data.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=1.5, color='lightgray', opacity=0.6),
            name='Body Cloud'
        ))
    origin = results.get("origin_point", np.zeros(3))
    R = results.get("rotation_matrix", np.eye(3))
    shift = results.get("centroid_shift", np.zeros(3))
    def transform_pts(p):
        p = np.asarray(p)
        if p.ndim == 1:
            return (R @ (p - origin)) + origin + shift
        else:
            return (p - origin) @ R.T + origin + shift
    # Leg clusters
    leg_clusters = results.get("leg_clusters", [])
    line_params = results.get("line_params", [])
    colors = ['green', 'blue']
    for i, cluster in enumerate(leg_clusters):
        if i >= len(colors): break
        c_pts = np.asarray(cluster)
        if len(c_pts) < 2: continue
        t_pts = transform_pts(c_pts)
        try:
            pca = PCA(n_components=1)
            pca.fit(t_pts[:, [0,2]])
            proj = (t_pts[:, [0,2]] - pca.mean_).dot(pca.components_[0])
            t_pts_sorted = t_pts[np.argsort(proj)]
        except:
            t_pts_sorted = t_pts
        figs_data.append(go.Scatter3d(
            x=t_pts_sorted[:,0], y=t_pts_sorted[:,1], z=t_pts_sorted[:,2],
            mode='lines+markers',
            marker=dict(size=2, color=colors[i], opacity=0.6),
            line=dict(color=colors[i], width=3),
            name=f'Leg {i+1} Skeleton'
        ))
        if i < len(line_params):
            mean, dir_vec = line_params[i]
            if mean is not None and dir_vec is not None:
                vecs = cluster - mean
                projs = vecs.dot(dir_vec)
                min_t, max_t = projs.min(), projs.max()
                p_start = mean + min_t * dir_vec
                p_end = mean + max_t * dir_vec
                line_seg = np.vstack([p_start, p_end])
                t_line_seg = transform_pts(line_seg)
                figs_data.append(go.Scatter3d(
                    x=t_line_seg[:,0], y=t_line_seg[:,1], z=t_line_seg[:,2],
                    mode='lines',
                    line=dict(color=colors[i], width=10),
                    name=f'Leg {i+1} Axis'
                ))
    # Ground connection lines
    ground_points = results.get("ground_points", [])
    for i, cluster in enumerate(leg_clusters):
        if i >= len(ground_points): break
        c_pts = np.asarray(cluster)
        centroid = c_pts.mean(axis=0)
        t_centroid = transform_pts(centroid)
        gp = ground_points[i]
        t_gp = transform_pts(gp)
        figs_data.append(go.Scatter3d(
            x=[t_centroid[0], t_gp[0]], y=[t_centroid[1], t_gp[1]], z=[t_centroid[2], t_gp[2]],
            mode='lines+markers',
            marker=dict(size=4, color='orange'),
            line=dict(color='orange', width=3, dash='dash'),
            name=f'Ground Conn {i+1}'
        ))
        figs_data.append(go.Scatter3d(
            x=[t_gp[0]], y=[t_gp[1]], z=[t_gp[2]],
            mode='markers',
            marker=dict(size=6, color='red'),
            name=f'Ground Pt {i+1}'
        ))
    # Hip seeds
    hip_seeds = results.get("hip_seeds", [])
    if len(hip_seeds) > 0:
        t_hips = transform_pts(np.array(hip_seeds))
        figs_data.append(go.Scatter3d(
            x=t_hips[:,0], y=t_hips[:,1], z=t_hips[:,2],
            mode='markers',
            marker=dict(size=8, color='magenta', symbol='diamond'),
            name='Hip Seeds'
        ))
    # axes at visual center
    center_vis = origin + shift
    axis_len = 0.2
    if rotated_pcd is not None:
        pts = np.asarray(rotated_pcd.points)
        if len(pts) > 0:
            extent = pts.max(axis=0) - pts.min(axis=0)
            axis_len = np.linalg.norm(extent) * 0.08
    axes_data = [
        ([axis_len, 0, 0], 'red', 'X'),
        ([0, axis_len, 0], 'green', 'Y'),
        ([0, 0, axis_len], 'blue', 'Z')
    ]
    for vec, color, label in axes_data:
        end_pt = center_vis + np.array(vec)
        figs_data.append(go.Scatter3d(
            x=[center_vis[0], end_pt[0]],
            y=[center_vis[1], end_pt[1]],
            z=[center_vis[2], end_pt[2]],
            mode='lines',
            line=dict(color=color, width=6),
            name=f'Axis {label}'
        ))
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X', visible=True),
            yaxis=dict(title='Y', visible=True),
            zaxis=dict(title='Z', visible=True),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(y=0.9)
    )
    return go.Figure(data=figs_data, layout=layout)

# ---------------------- Gradio wrapper ----------------------

def run_skeleton_alignment_interface(mesh_file, voxel_frac, low_frac, high_frac,
                                     save_dir, output_filename, ground_plane_file, ground_tol_frac,
                                     dbscan_eps_frac, dbscan_min_samples, verticality_thresh, min_cluster_frac,
                                     geodesic_thresh, use_pelvis_fork):
    if mesh_file is None:
        return None, "Please upload a mesh file."
    try:
        input_path = mesh_file.name
        if not save_dir:
            save_dir = os.path.join(os.path.dirname(input_path), "skeleton_out")
        os.makedirs(save_dir, exist_ok=True)

        plane = None
        log_text = f"Running Skeleton Alignment\nInput: {input_path}\n"
        if ground_plane_file is not None:
            try:
                parsed = parse_plane_from_json(ground_plane_file.name)
                if parsed is None:
                    log_text += "Warning: Could not parse JSON plane; will auto-estimate.\n"
                else:
                    plane = parsed
                    log_text += f"Using provided ground plane: {plane}\n"
            except Exception as e:
                log_text += f"Warning: Failed to read ground JSON: {e}\n"

        log_text += f"Relative params: voxel_frac={voxel_frac}, ground_tol_frac={ground_tol_frac}, eps_frac={dbscan_eps_frac}\n"
        log_text += f"ROI fractions: low={low_frac}, high={high_frac}\n"
        log_text += f"Geodesic Threshold: {geodesic_thresh} m\n"

        results = skeleton_main(
            pcd_path=input_path,
            low_frac=low_frac, high_frac=high_frac,
            voxel_frac=voxel_frac,
            ground_plane=plane,
            ground_tol_frac=ground_tol_frac,
            dbscan_eps_frac=dbscan_eps_frac,
            dbscan_min_samples=int(dbscan_min_samples),
            verticality_thresh=verticality_thresh,
            min_cluster_frac=min_cluster_frac,
            geodesic_thresh=geodesic_thresh,
            save_prefix=save_dir,
            debug=True,
            use_pelvis_fork=use_pelvis_fork
        )

        log_text += "Alignment complete.\n"
        log_text += f"Rotation matrix saved in: {os.path.join(save_dir, 'rotation_matrix.npy')}\n"
        shift = results.get("centroid_shift", np.zeros(3))
        log_text += f"Centroid shift applied: {shift}\n"
        fork_info = results.get("fork_detection", {})
        log_text += f"Fork detection score: {fork_info.get('score', 0.0)}; best_node: {fork_info.get('best_node')}\n"

        if not output_filename:
            output_filename = "aligned_mesh.ply"
        if not output_filename.lower().endswith(".ply"):
            output_filename += ".ply"
        out_path = os.path.join(save_dir, output_filename)
        try:
            mesh_in = o3d.io.read_triangle_mesh(input_path)
            if not mesh_in.is_empty() and len(mesh_in.triangles) > 0:
                origin = results["origin_point"]
                R = results["rotation_matrix"]
                mesh_in.translate(-origin, relative=False)
                mesh_in.rotate(R, center=(0,0,0))
                mesh_in.translate(origin, relative=False)
                mesh_in.translate(shift, relative=True)
                o3d.io.write_triangle_mesh(out_path, mesh_in, write_ascii=True)
                log_text += f"Saved aligned MESH (with faces) to: {out_path}\n"
            else:
                o3d.io.write_point_cloud(out_path, results["rotated_pcd"], write_ascii=True)
                log_text += f"Saved aligned POINT CLOUD (no faces found in input) to: {out_path}\n"
        except Exception as e:
            log_text += f"Warning: Failed to save as mesh ({e}). Saving point cloud instead.\n"
            o3d.io.write_point_cloud(out_path, results["rotated_pcd"], write_ascii=True)

        fig = create_plotly_visualization(results)
        return fig, log_text

    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}\n{traceback.format_exc()}"

# ---------------------- Gradio UI ----------------------

with gr.Blocks(title="Skeleton Leg Aligner — Pelvis Fork Detection (Hybrid)") as demo:
    gr.Markdown("# Skeleton Leg Aligner — pelvis-fork detector (hybrid)")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Point Cloud / Mesh", file_types=[".ply", ".pcd", ".xyz", ".obj"])
            gr.Markdown("### Relative parameters (fractions of object height H = maxY - minY)")
            voxel_frac_input = gr.Slider(label="Voxel size (fraction of H)", minimum=0.002, maximum=0.05, value=0.01, step=0.001)
            ground_tol_frac_input = gr.Slider(label="Ground tolerance (fraction of H)", minimum=0.002, maximum=0.1, value=0.02, step=0.001)
            dbscan_eps_frac_input = gr.Slider(label="DBSCAN eps (fraction of H)", minimum=0.002, maximum=0.05, value=0.015, step=0.001)

            gr.Markdown("### ROI (fractions from feet = maxY)")
            low_frac_input = gr.Slider(label="Low fraction (near feet)", minimum=0.0, maximum=0.5, value=0.05, step=0.01)
            high_frac_input = gr.Slider(label="High fraction (towards knees)", minimum=0.0, maximum=1.0, value=0.25, step=0.01)

            gr.Markdown("### Clustering & filtering")
            dbscan_min_samples_input = gr.Number(label="DBSCAN min samples", value=5, step=1)
            verticality_thresh_input = gr.Number(label="Verticality threshold (PCA ratio)", value=2.0, step=0.1)
            min_cluster_frac_input = gr.Number(label="Min cluster fraction (skeleton pts)", value=0.005, step=0.001)
            geodesic_thresh_input = gr.Number(label="Geodesic Threshold (m)", value=0.5, step=0.1)
            use_pelvis_fork_input = gr.Checkbox(label="Use Pelvis Fork Detection", value=True)

            gr.Markdown("### Ground plane (optional)")
            gr.Markdown("Upload JSON (many formats supported). If omitted, plane estimated by RANSAC inside pipeline.")
            ground_plane_input = gr.File(label="Ground Plane JSON", file_types=[".json"])

            gr.Markdown("### Output")
            save_dir_input = gr.Textbox(label="Save directory", value="skeleton_out")
            output_filename_input = gr.Textbox(label="Output filename (.ply)", value="aligned_mesh.ply")

            run_btn = gr.Button("Run Alignment", variant="primary")

        with gr.Column():
            model_out = gr.Plot(label="3D Visualization")
            logs_out = gr.Textbox(label="Logs", lines=16)

    run_btn.click(fn=run_skeleton_alignment_interface,
                  inputs=[file_input, voxel_frac_input, low_frac_input, high_frac_input,
                          save_dir_input, output_filename_input, ground_plane_input, ground_tol_frac_input,
                          dbscan_eps_frac_input, dbscan_min_samples_input, verticality_thresh_input, min_cluster_frac_input,
                          geodesic_thresh_input, use_pelvis_fork_input],
                  outputs=[model_out, logs_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)

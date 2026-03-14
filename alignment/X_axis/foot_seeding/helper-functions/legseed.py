#!/usr/bin/env python3
"""
skeleton_gradio_relative.py

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
    python skeleton_gradio_relative.py
Dependencies:
    pip install numpy open3d scikit-image scipy scikit-learn gradio plotly
"""

import os
import json
import re
import tempfile
import math
import numpy as np
import open3d as o3d
from scipy import ndimage
# Use unified skeletonize API. If skeletonize_3d exists in older skimage, prefer it.
try:
    # Newer skimage: skeletonize handles 2D and 3D
    from skimage.morphology import skeletonize as _skel_unified
    def skeletonize(grid):
        return _skel_unified(grid)
except Exception:
    # Last-resort fallback to older name
    try:
        from skimage.morphology import skeletonize_3d as _skel3d_impl
        def skeletonize(grid):
            return _skel3d_impl(grid)
    except Exception:
        # Further fallback will be provided later if needed
        from skimage.morphology import skeletonize as _skel_unified2
        def skeletonize(grid):
            return _skel_unified2(grid)

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import gradio as gr
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import plotly.graph_objs as go

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

    # last attempt: scan for numeric-like values
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
def normalize_plane(plane):
    """
    Accept many plane representations and return (a,b,c,d) floats or None.
    Handles lists/tuples, dicts with keys a/b/c/d or {'normal':[x,y,z], 'd':d}, or nested dicts.
    """
    if plane is None:
        return None
    # already a list/tuple of numbers
    if isinstance(plane, (list, tuple)) and len(plane) == 4:
        try:
            return (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
        except Exception:
            return None
    if isinstance(plane, dict):
        # direct a,b,c,d keys
        if all(k in plane for k in ("a","b","c","d")):
            try:
                return (float(plane["a"]), float(plane["b"]), float(plane["c"]), float(plane["d"]))
            except Exception:
                pass
        # normal + d
        if "normal" in plane and ("d" in plane or "D" in plane):
            try:
                normal = plane["normal"]
                dval = plane.get("d", plane.get("D"))
                return (float(normal[0]), float(normal[1]), float(normal[2]), float(dval))
            except Exception:
                pass
        # plane might itself be nested or contain numeric values; attempt to extract numbers
        vals = []
        def rec_extract(x):
            if x is None: return
            if isinstance(x, (int, float, np.floating, np.integer)):
                vals.append(float(x)); return
            if isinstance(x, str):
                try:
                    vals.append(float(x)); return
                except Exception:
                    return
            if isinstance(x, (list,tuple)):
                for y in x:
                    rec_extract(y)
            elif isinstance(x, dict):
                for y in x.values():
                    rec_extract(y)
        rec_extract(plane)
        if len(vals) >= 4:
            return (vals[0], vals[1], vals[2], vals[3])
    # otherwise fail
    return None

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
    # Map points to indices
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


# ---------------------- Geodesic computation (reused) ----------------------

def compute_geodesic_from_hips(binary_grid, min_b, voxel_size, hip_world_points, max_nodes_for_graph=350_000):
    """
    Compute geodesic shortest-path distances (meters) from hip/leg seeds to all occupied voxels.
    (Re-used function name for compatibility; accepts any set of seed world points)
    """
    occ_idx = np.argwhere(binary_grid > 0)
    n_occ = len(occ_idx)
    if n_occ == 0:
        return np.full(binary_grid.shape, np.inf, dtype=float), []

    # mapping voxel->id
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
    for hw in hip_world_points:
        vi = world_to_voxel_index(np.array(hw), min_b, voxel_size, binary_grid.shape)
        if tuple(vi) not in id_of:
            found = None
            for r in range(1,6):
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
                    found = chosen
                    break
            if found is not None:
                nid = id_of[tuple(found)]
                seed_ids.append(nid)
        else:
            seed_ids.append(id_of[tuple(vi)])

    if len(seed_ids) == 0:
        return np.full(binary_grid.shape, np.inf, dtype=float), []

    # run multi-source dijkstra
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

from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt

def compute_leg_seeds_plane2d(slab_pts, plane, H,
                              voxel_px_frac=0.002, keep_k=3,
                              min_area_frac=0.0005, torso_radius_frac=0.35,
                              min_compactness=0.35, max_offset_frac=0.02):
    # Project slab points
    projs = project_to_plane(slab_pts, plane)
    # grid resolution in meters
    voxel_px = voxel_px_frac * H
    grid, grid_meta = rasterize_contacts(projs, voxel_px=voxel_px)
    # multi-offset accumulator (simple: try offsets by reprojecting with shifted d)
    # (for brevity do single pass here; you can iterate offsets and sum grids)
    labeled, labels_chosen = largest_plane_components(grid, keep_k=keep_k, closing_iter=2)
    seeds = []
    min_area_px = max(4, int(min_area_frac * grid.size))
    min_torso_radius = torso_radius_frac * H
    # torso center XZ
    min_x, min_z, px = grid_meta
    torso_center_xz = np.array([(slab_pts[:,0].min()+slab_pts[:,0].max())/2.0,
                                (slab_pts[:,2].min()+slab_pts[:,2].max())/2.0])
    for lab in labels_chosen:
        mask = (labeled == lab)
        area = mask.sum()
        if area < min_area_px: continue
        # get grid indices of mask -> world XZ coords
        inds = np.argwhere(mask)
        xs = min_x + (inds[:,0] - 0) * px
        zs = min_z + (inds[:,1] - 0) * px
        pts_xz = np.vstack([xs, zs]).T
        # compactness via convex hull
        try:
            hull = ConvexHull(pts_xz)
            hull_area = hull.volume
            compactness = area / (hull_area / (px*px) + 1e-9)  # normalized
        except Exception:
            compactness = 1.0
        if compactness < min_compactness: continue
        # centroid in XZ
        cx = xs.mean(); cz = zs.mean()
        # map centroid back to world point: we need Y on plane at (cx,cz)
        a,b,c,d = plane
        if abs(b) > 1e-9:
            cy = -(a*cx + c*cz + d) / b
        else:
            # compute using average Y of nearby slab points projected near centroid
            # fallback:
            dist2 = (projs[:,0]-cx)**2 + (projs[:,2]-cz)**2
            idx = np.argmin(dist2)
            cy = projs[idx,1]
        centroid_world = np.array([cx, cy, cz])
        # torso check
        dx = cx - torso_center_xz[0]; dz = cz - torso_center_xz[1]
        if math.hypot(dx, dz) > 1.5 * min_torso_radius:
            # likely far away -> skip
            continue
        seeds.append((centroid_world, area, compactness))
    # If we have >=2 seeds, sort by area and pick two
    if len(seeds) >= 2:
        seeds.sort(key=lambda x: -x[1])
        return [s[0] for s in seeds[:2]]
    # else fallback to PCA extremes
    # ... (same as your previous fallback)
    return fallback_pca_seeds(slab_pts, plane, H)


# ---------------------- Ground-connected pruning ----------------------

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
        # lenient size-based keep
        if near_ground and (size > max(100, int(0.01 * np.prod(binary_grid.shape)))):
            keep = True

        if keep:
            keep_labels.add(lab)

    kept_mask = np.isin(labeled, list(keep_labels))
    pruned = np.zeros_like(binary_grid)
    pruned[kept_mask] = 1
    return pruned


# ---------------------- Skeletonization & skeleton -> points ----------------------

def run_skeletonization(binary_grid):
    bool_grid = (binary_grid > 0)
    # Use unified skeletonize for 3D
    skeleton = skeletonize(bool_grid).astype(np.uint8)
    return skeleton

def skeleton_to_points(skeleton_grid, min_b, voxel_size):
    xs, ys, zs = np.where(skeleton_grid > 0)
    pts = grid_indices_to_points(np.vstack([xs, ys, zs]).T, min_b, voxel_size)
    return pts


# ---------------------- Skeleton clustering & filtering ----------------------

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


# ---------------------- Line fit, intersection, rotation ----------------------

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


# ---------------------- Main pipeline (relative parameters) ----------------------

def estimate_ground_plane(pcd, ransac_n=3, dist_threshold=0.01, n_iterations=2000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=n_iterations)
    return tuple(plane_model), inliers

from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt

def compute_leg_seeds_from_ground(points, ground_plane, H,
                                  max_offset_frac=0.02,
                                  ground_tol_frac=0.01,
                                  dbscan_eps_frac=0.02,
                                  dbscan_min_samples=5,
                                  n_expected=2,
                                  torso_center_xz=None,
                                  torso_radius_frac=0.6,
                                  min_lateral_sep_frac=0.02,
                                  debug=False):
    """
    Robust leg seeding with optional rich debug output.
    Returns: (seeds_list, debug_info) where debug_info is None unless debug=True.
    debug_info contains per-offset near_pts (Nx3), projected pts (Nx3), component polygons (list of XZ arrays),
    per-component centroids, chosen seeds, and grid metadata.
    """
    debug_info = {"offsets": []} if debug else None

    if ground_plane is None:
        return [], debug_info

    # ensure numeric plane
    gp_norm = normalize_plane(ground_plane)
    if gp_norm is None:
        # invalid plane format — bail out with empty seeds (caller uses fallbacks)
        if debug:
            debug_info["error"] = "ground_plane not numeric after normalization"
        return [], debug_info
    a,b,c,d = gp_norm
    nvec = np.array([a,b,c], dtype=float)

    norm_len = np.linalg.norm(nvec)
    if norm_len < 1e-12:
        return [], debug_info

    # Absolute parameters
    max_offset = max_offset_frac * H
    ground_tol = ground_tol_frac * H
    dbscan_eps = max(1e-6, dbscan_eps_frac * H)
    min_lateral_sep = max(1e-6, min_lateral_sep_frac * H)
    torso_center_xz = np.array(torso_center_xz) if torso_center_xz is not None else None
    torso_radius = torso_radius_frac * H if torso_radius_frac is not None else None

    # raster resolution (meters)
    voxel_px = max(1e-4, 0.002 * H)

    offsets = np.linspace(0.0, max_offset, num=3) if max_offset > 0 else [0.0]
    candidate_components = []

    # Compute torso center fallback from full points if not provided
    if torso_center_xz is None:
        torso_center_xz = np.array([(points[:,0].min() + points[:,0].max())/2.0,
                                    (points[:,2].min() + points[:,2].max())/2.0])

    for off in offsets:
        d_shifted = d + off * norm_len
        plane_dists = (points @ nvec + d_shifted) / norm_len
        mask = np.abs(plane_dists) <= ground_tol
        near_pts = points[mask]
        if near_pts.shape[0] < 6:
            if debug:
                debug_info["offsets"].append({"offset": float(off), "n_near": int(near_pts.shape[0]), "components": []})
            continue

        # Project the near_pts to plane
        denom = (nvec @ nvec) + 1e-12
        projs = near_pts - np.outer((near_pts @ nvec + d_shifted) / denom, nvec)  # Nx3

        # Rasterize projected XZ to grid
        xs = projs[:,0]; zs = projs[:,2]
        min_x, max_x = xs.min(), xs.max()
        min_z, max_z = zs.min(), zs.max()
        pad = 4
        gx = int(np.ceil((max_x - min_x) / voxel_px)) + 1 + 2*pad
        gz = int(np.ceil((max_z - min_z) / voxel_px)) + 1 + 2*pad
        # If grid would be huge (rare), fallback to DBSCAN clustering on projs directly
        if gx <= 0 or gz <= 0 or gx*gz > 6_000_000:
            cl = DBSCAN(eps=dbscan_eps, min_samples=max(2, int(dbscan_min_samples))).fit(projs)
            labels = cl.labels_
            unique = [lab for lab in np.unique(labels) if lab != -1]
            comps = []
            for lab in unique:
                cpts = projs[labels==lab]
                centroid = cpts.mean(axis=0)
                min_gd = float(np.min(np.abs((cpts @ nvec + d_shifted)) / norm_len))
                comps.append({"centroid_world": centroid, "pts": cpts, "min_ground_dist": min_gd})
                candidate_components.append({"centroid_world": centroid,
                                             "area_m2": cpts.shape[0] * (voxel_px**2),
                                             "min_ground_dist": min_gd,
                                             "compactness": 1.0,
                                             "offset": float(off),
                                             "pts_xz": cpts[:, [0,2]]})
            if debug:
                debug_info["offsets"].append({"offset": float(off), "n_near": int(near_pts.shape[0]), "components": comps,
                                             "projs": projs})
            continue

        ix = np.floor((xs - min_x) / voxel_px).astype(int) + pad
        iz = np.floor((zs - min_z) / voxel_px).astype(int) + pad
        valid = (ix>=0)&(ix<gx)&(iz>=0)&(iz<gz)
        grid = np.zeros((gx, gz), dtype=np.uint8)
        grid[ix[valid], iz[valid]] = 1

        # Morphological closing to fill small holes and remove thin noise
        gridc = ndimage.binary_closing(grid, structure=np.ones((3,3)), iterations=2)
        labeled, nlab = ndimage.label(gridc)
        if nlab == 0:
            if debug:
                debug_info["offsets"].append({"offset": float(off), "n_near": int(near_pts.shape[0]), "components": [], "grid_meta": {"min_x": min_x, "min_z": min_z, "voxel_px": voxel_px}})
            continue
        sizes = ndimage.sum(gridc, labeled, range(1, nlab+1))
        order = np.argsort(-sizes)

        comps_out = []
        for idx in order:
            lab = idx + 1
            mask_grid = (labeled == lab)
            area_px = int(mask_grid.sum())
            if area_px < max(4, int(0.0005 * grid.size)):
                continue
            coords = np.argwhere(mask_grid)
            xs_grid = min_x + (coords[:,0] - pad + 0.5) * voxel_px
            zs_grid = min_z + (coords[:,1] - pad + 0.5) * voxel_px
            pts_xz = np.vstack([xs_grid, zs_grid]).T
            # compactness via convex hull area
            compactness = 1.0
            hull_area = None
            try:
                hull = ConvexHull(pts_xz)
                hull_area = float(hull.volume)
                compactness = (area_px * (voxel_px**2)) / (hull_area + 1e-12)
            except Exception:
                compactness = 1.0
            # centroid world Y computed from plane (use shifted d)
            cx = xs_grid.mean(); cz = zs_grid.mean()
            if abs(b) > 1e-9:
                cy = -(a*cx + c*cz + d_shifted) / b
            else:
                # fallback: pick nearest projected point's Y
                dist2 = (projs[:,0]-cx)**2 + (projs[:,2]-cz)**2
                cy = projs[np.argmin(dist2),1]
            centroid_world = np.array([cx, cy, cz])
            # get min ground dist among projs (absolute)
            min_ground_dist = float(np.min(np.abs((projs @ nvec + d_shifted)) / norm_len))
            comp = {"centroid_world": centroid_world,
                    "area_m2": area_px * (voxel_px**2),
                    "min_ground_dist": min_ground_dist,
                    "compactness": compactness,
                    "pts_xz": pts_xz,
                    "hull_area": hull_area}
            comps_out.append(comp)
            candidate_components.append({"centroid_world": centroid_world,
                                         "area_m2": area_px * (voxel_px**2),
                                         "min_ground_dist": min_ground_dist,
                                         "compactness": compactness,
                                         "offset": float(off),
                                         "pts_xz": pts_xz})
        if debug:
            debug_info["offsets"].append({"offset": float(off), "n_near": int(near_pts.shape[0]), "components": comps_out,
                                         "grid_meta": {"min_x": float(min_x), "min_z": float(min_z), "voxel_px": float(voxel_px)},
                                         "projs": projs})
    # end offsets loop

    if len(candidate_components) == 0:
        # fallback: projected PCA extremes
        try:
            denom = (nvec @ nvec) + 1e-12
            projs_all = points - np.outer((points @ nvec + d) / denom, nvec)
            p_xz = projs_all[:, [0,2]]
            pca = PCA(n_components=1); pca.fit(p_xz)
            proj_vals = (p_xz - pca.mean_).dot(pca.components_[0])
            idx_min = np.argmin(proj_vals); idx_max = np.argmax(proj_vals)
            seeds = [projs_all[idx_min], projs_all[idx_max]]
            if debug:
                debug_info["chosen"] = [seeds[0].tolist(), seeds[1].tolist()]
            return seeds[:n_expected], debug_info
        except Exception:
            if debug:
                debug_info["chosen"] = []
            return [], debug_info

    # Rank: prefer smallest ground dist, then largest area, then compactness
    candidate_components.sort(key=lambda c: (c["min_ground_dist"], -c["area_m2"], -c.get("compactness", 0.0)))

    # torso proximity filter
    filtered = []
    for c in candidate_components:
        cx, cz = c["centroid_world"][0], c["centroid_world"][2]
        if torso_center_xz is not None and torso_radius is not None:
            dx = cx - torso_center_xz[0]; dz = cz - torso_center_xz[1]
            dist_xz = math.hypot(dx, dz)
            if dist_xz > max(2.0 * torso_radius, torso_radius + 1e-9):
                continue
        filtered.append(c)
    if len(filtered) == 0:
        filtered = candidate_components

    # pick seeds enforcing lateral separation
    chosen = []
    for cand in filtered:
        if len(chosen) == 0:
            chosen.append(cand); 
            if len(chosen) >= n_expected: break
            continue
        ok = True
        for ch in chosen:
            sep = np.linalg.norm((cand["centroid_world"] - ch["centroid_world"])[[0,2]])
            if sep < min_lateral_sep:
                ok = False; break
        if ok:
            chosen.append(cand)
        if len(chosen) >= n_expected:
            break

    if len(chosen) < n_expected:
        for cand in filtered:
            if any(np.allclose(cand["centroid_world"], ch["centroid_world"]) for ch in chosen): continue
            chosen.append(cand)
            if len(chosen) >= n_expected: break

    seeds = [c["centroid_world"] for c in chosen[:n_expected]]

    if debug:
        # Provide detailed debug arrays: per-offset near_pts, per-component polygons (world XYZ), chosen seeds
        debug_info["components_all"] = []
        for comp in candidate_components:
            poly_xz = comp.get("pts_xz", None)
            poly_world = None
            if poly_xz is not None:
                # compute Y for each XZ using plane (use original d)
                xs = poly_xz[:,0]; zs = poly_xz[:,1]
                if abs(b) > 1e-9:
                    ys = (-(a*xs + c*zs + d)) / b
                else:
                    ys = np.zeros_like(xs) + (points[:,1].mean())
                poly_world = np.vstack([xs, ys, zs]).T
            debug_info["components_all"].append({"centroid": comp["centroid_world"].tolist(),
                                                 "area_m2": comp["area_m2"],
                                                 "min_ground_dist": comp["min_ground_dist"],
                                                 "compactness": comp.get("compactness", None),
                                                 "poly_world": (poly_world.tolist() if poly_world is not None else None),
                                                 "offset": comp.get("offset", None)})
        debug_info["chosen"] = [s.tolist() for s in seeds]
        debug_info["torso_center_xz"] = torso_center_xz.tolist()
        debug_info["voxel_px"] = float(voxel_px)

    return seeds, debug_info

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

    # normalize whatever we got (from parse or auto-estimate)
    ground_plane = normalize_plane(ground_plane)
    if ground_plane is None:
        print("Warning: ground_plane could not be normalized -> seeding will use fallbacks.")

    # Convert relative fractions to absolute values
    voxel_size = voxel_frac * H
    ground_tol = ground_tol_frac * H
    dbscan_eps = dbscan_eps_frac * H

    # Feet-relative slicing: assume feet ~ maxY
    y_feet = max_b[1]
    y1 = y_feet - (low_frac * H)
    y2 = y_feet - (high_frac * H)
    if y2 > y1:
        y1, y2 = y2, y1

    mask = (points[:,1] <= y1) & (points[:,1] >= y2)
    sliced_pts = points[mask]
    if sliced_pts.shape[0] == 0:
        raise RuntimeError("Slice produced zero points. Try changing fractions or check axis.")
    sliced_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sliced_pts))
    o3d.io.write_point_cloud(os.path.join(save_prefix, "sliced.ply"), sliced_pcd)

    # --- NEW: Leg Seeding from ground intersection (replaces hip seeding) ---
    leg_seeds = []
    try:
        # Use the foot slab (sliced_pts) as the focus for seed extraction to avoid head/hand contamination
        torso_center_xz = np.array([(min_b[0] + max_b[0]) / 2.0, (min_b[2] + max_b[2]) / 2.0])
        leg_seeds, _ = compute_leg_seeds_from_ground(
            sliced_pts, ground_plane, H,
            max_offset_frac=leg_seed_max_offset_frac,
            ground_tol_frac=ground_tol_frac,
            dbscan_eps_frac=dbscan_eps_frac,
            dbscan_min_samples=dbscan_min_samples,
            n_expected=2,
            torso_center_xz=torso_center_xz,
            torso_radius_frac=0.6,
            min_lateral_sep_frac=0.02
        )

    except Exception as e:
        print(f"Leg seed computation failed: {e}")
        leg_seeds = []

    if len(leg_seeds) == 0:
        # As a final fallback (should be rare), use two extremes in XZ of the sliced pts projected onto ground
        try:
            if ground_plane is not None:
                a,b,c,d = ground_plane
                nvec = np.array([a,b,c], dtype=float)
                denom = (np.dot(nvec, nvec) + 1e-12)
                projs = points - np.outer((points @ nvec + d) / denom, nvec)
                p_xz = projs[:, [0,2]]
                pca = PCA(n_components=1)
                pca.fit(p_xz)
                proj_vals = (p_xz - pca.mean_).dot(pca.components_[0])
                idx_min = np.argmin(proj_vals)
                idx_max = np.argmax(proj_vals)
                leg_seeds = [projs[idx_min], projs[idx_max]]
            else:
                leg_seeds = [points.mean(axis=0)]
        except Exception:
            leg_seeds = [points.mean(axis=0)]

    # Voxelization (on sliced points)
    binary_grid, min_grid_b, voxel_size_used = voxelize_pointcloud_to_grid(sliced_pts, voxel_size=voxel_size)
    
    # Geodesic Pruning (use leg seeds)
    distances_grid, _ = compute_geodesic_from_hips(binary_grid, min_grid_b, voxel_size_used, leg_seeds)
    
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
    leg_center_lines = []

    # Process exactly 2 legs (or fewer if not found)
    for i, cpts in enumerate(clusters[:2]):
        mean, dir_vec = fit_line_through_points(cpts)
        if mean is not None:
            # Calculate ground intersection
            inter = None
            if ground_plane is not None:
                inter = line_plane_intersection(mean, dir_vec, ground_plane)
            if inter is None:
                # Fallback to projecting onto a horizontal plane if explicit plane missing/vertical
                if ground_plane is not None and abs(ground_plane[1]) > 1e-8:
                    a,b,c,d = ground_plane
                    gy = -(a*mean[0] + c*mean[2] + d) / b
                    inter = np.array([mean[0], gy, mean[2]])
                else:
                    inter = mean.copy()
            
            # Calculate center line segment
            vecs = cpts - mean
            projs = vecs.dot(dir_vec)
            p_start = mean + projs.min() * dir_vec
            p_end = mean + projs.max() * dir_vec
            
            line_params.append((mean, dir_vec))
            ground_points.append(inter)
            leg_center_lines.append((p_start, p_end))
        else:
            # Placeholder to keep indices aligned if fit fails
            line_params.append((None, None))
            ground_points.append(cpts.mean(axis=0)) # Fallback to centroid
            leg_center_lines.append(None)

    # Fallback if we didn't get 2 valid legs
    if len(ground_points) < 2:
        # Fallback: use extremes from skeleton points
        pca = PCA(n_components=2)
        pca.fit(skel_pts[:, [0,2]])
        proj = (skel_pts[:, [0,2]] - pca.mean_).dot(pca.components_[0])
        pmin = skel_pts[np.argmin(proj)]
        pmax = skel_pts[np.argmax(proj)]
        
        # Create a fake "center line" connecting min and max for visualization
        leg_center_lines = [(pmin, pmax), (pmax, pmin)]

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
        # Emergency fallback if points coincide
        vector = np.array([1.0, 0.0, 0.0]) 
    else:
        vector = vector / vnorm
    
    origin_point = (p1 + p2) / 2.0

    # rotate original full cloud
    original_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotated_pcd, R = rotate_pointcloud_to_xaxis(original_pcd, origin_point, vector)

    # --- Shift origin to centroid of non-ground mesh ---
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

    # Apply shift to the rotated point cloud
    rotated_pcd.translate(centroid_shift, relative=True)

    # Save outputs (note: use "leg_seeds" naming)
    o3d.io.write_point_cloud(os.path.join(save_prefix, "rotated_full.ply"), rotated_pcd)
    np.save(os.path.join(save_prefix, "skeleton_points.npy"), skel_pts)
    np.save(os.path.join(save_prefix, "ground_points.npy"), np.vstack(ground_points))
    np.save(os.path.join(save_prefix, "rotation_matrix.npy"), R)
    np.save(os.path.join(save_prefix, "origin_point.npy"), origin_point)
    np.save(os.path.join(save_prefix, "centroid_shift.npy"), centroid_shift)
    np.save(os.path.join(save_prefix, "leg_seeds.npy"), np.vstack(leg_seeds) if len(leg_seeds)>0 else np.zeros((0,3)))

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
        "leg_seeds": leg_seeds,
        "line_params": line_params,
        "leg_center_lines": leg_center_lines,
        "H": H,
        "ground_plane": ground_plane,
        "min_b": min_b,
        "max_b": max_b
    }


# ---------------------- Gradio wrapper ----------------------

def create_voxel_mesh(points, color, voxel_size, max_points=10000):
    """Creates a combined mesh of cubes for the given points."""
    if len(points) == 0:
        return o3d.geometry.TriangleMesh()
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    n_pts = len(points)
    half = voxel_size / 2.0
    base_verts = np.array([
        [-half, -half, -half], [half, -half, -half], [half, half, -half], [-half, half, -half],
        [-half, -half, half], [half, -half, half], [half, half, half], [-half, half, half]
    ])
    base_tris = np.array([
        [0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4], [0, 4, 7], [0, 7, 3],
        [6, 5, 1], [6, 1, 2], [6, 2, 3], [6, 3, 7], [6, 7, 4], [6, 4, 5]
    ], dtype=np.int32)
    
    verts = (points[:, np.newaxis, :] + base_verts[np.newaxis, :, :]).reshape(-1, 3)
    offsets = np.arange(n_pts, dtype=np.int32) * 8
    tris = (base_tris[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]).reshape(-1, 3)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

# ---------- Visualization helpers: cylinder & polyline ----------

def create_plotly_visualization(results):
    """
    Creates a Plotly Figure showing:
      - The rotated body point cloud (subsampled)
      - The skeleton leg branches (lines)
      - The ground connection lines
      - The leg seeds
    """
    figs_data = []

    # 1. Body Point Cloud
    rotated_pcd = results.get("rotated_pcd")
    if rotated_pcd is not None:
        pts = np.asarray(rotated_pcd.points)
        # Subsample for performance
        if len(pts) > 15000:
            indices = np.random.choice(len(pts), 15000, replace=False)
            pts = pts[indices]
        
        figs_data.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=1.5, color='lightgray', opacity=0.5),
            name='Body Cloud'
        ))

    # Helpers for transformation
    origin = results.get("origin_point", np.zeros(3))
    R = results.get("rotation_matrix", np.eye(3))
    shift = results.get("centroid_shift", np.zeros(3))

    def transform_pts(p):
        p = np.asarray(p)
        if p.ndim == 1:
            return (R @ (p - origin)) + origin + shift
        else:
            return (p - origin) @ R.T + origin + shift

    # 2. Leg Skeletons & Fitted Center Lines
    leg_clusters = results.get("leg_clusters", [])
    leg_center_lines = results.get("leg_center_lines", [])
    
    # Define colors for legs (Leg 1: Green, Leg 2: Blue)
    colors = ['green', 'blue']
    
    # Iterate exactly 2 times (or max cluster count) to ensure we try to plot everything
    loop_count = max(len(leg_clusters), len(leg_center_lines))
    if loop_count == 0 and len(leg_center_lines) > 0: loop_count = len(leg_center_lines) # Fallback handling

    for i in range(loop_count):
        if i >= len(colors): break # Only support 2 main colors for now
        
        # --- A) Polyline (Skeleton points) ---
        if i < len(leg_clusters):
            c_pts = np.asarray(leg_clusters[i])
            if len(c_pts) > 1:
                t_pts = transform_pts(c_pts)
                
                # Sort for line connectivity (PCA)
                try:
                    pca = PCA(n_components=1)
                    pca.fit(t_pts[:, [0,2]]) 
                    proj = (t_pts[:, [0,2]] - pca.mean_).dot(pca.components_[0])
                    t_pts_sorted = t_pts[np.argsort(proj)]
                except:
                    t_pts_sorted = t_pts
                    
                figs_data.append(go.Scatter3d(
                    x=t_pts_sorted[:,0], y=t_pts_sorted[:,1], z=t_pts_sorted[:,2],
                    mode='markers',
                    marker=dict(size=2, color=colors[i], opacity=0.5),
                    name=f'Leg {i+1} Skeleton'
                ))

        # --- B) Fitted Center Line (Axis) ---
        if i < len(leg_center_lines) and leg_center_lines[i] is not None:
            p_start, p_end = leg_center_lines[i]
            
            line_seg = np.vstack([p_start, p_end])
            t_line_seg = transform_pts(line_seg)
            
            figs_data.append(go.Scatter3d(
                x=t_line_seg[:,0], y=t_line_seg[:,1], z=t_line_seg[:,2],
                mode='lines',
                line=dict(color=colors[i], width=10), # Thick line
                name=f'Leg {i+1} Axis'
            ))
        else:
             figs_data.append(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='lines',
                line=dict(color=colors[i], width=10),
                name=f'Leg {i+1} Axis (Missing)',
                visible='legendonly'
            ))

    # 3. Ground Lines
    ground_points = results.get("ground_points", [])
    for i in range(min(len(leg_clusters), len(ground_points))):
        c_pts = np.asarray(leg_clusters[i])
        centroid = c_pts.mean(axis=0)
        t_centroid = transform_pts(centroid)
        
        gp = ground_points[i]
        t_gp = transform_pts(gp)
        
        line_pts = np.vstack([t_centroid, t_gp])
        
        figs_data.append(go.Scatter3d(
            x=line_pts[:,0], y=line_pts[:,1], z=line_pts[:,2],
            mode='lines+markers',
            marker=dict(size=4, color='orange'),
            line=dict(color='orange', width=4, dash='dash'),
            name=f'Ground Conn {i+1}'
        ))

    # 4. Leg Seeds
    leg_seeds = results.get("leg_seeds", [])
    if len(leg_seeds) > 0:
        t_legs = transform_pts(leg_seeds)
        figs_data.append(go.Scatter3d(
            x=t_legs[:,0], y=t_legs[:,1], z=t_legs[:,2],
            mode='markers',
            marker=dict(size=6, color='magenta', symbol='diamond'),
            name='Leg Seeds'
        ))

    # 5. Coordinate Axes (RGB) at Center
    center_vis = origin + shift
    axis_len = 0.2
    if rotated_pcd is not None:
        pts = np.asarray(rotated_pcd.points)
        if len(pts) > 0:
            extent = pts.max(axis=0) - pts.min(axis=0)
            axis_len = np.linalg.norm(extent) * 0.1

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
        legend=dict(y=0.9),
        showlegend=True
    )
    
    return go.Figure(data=figs_data, layout=layout)

def run_skeleton_alignment_interface(mesh_file, voxel_frac, low_frac, high_frac,
                                     save_dir, output_filename, ground_plane_file, ground_tol_frac,
                                     dbscan_eps_frac, dbscan_min_samples, verticality_thresh, min_cluster_frac,
                                     geodesic_thresh, leg_seed_max_offset_frac):
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
                parsed_raw = parse_plane_from_json(ground_plane_file.name)
                parsed = normalize_plane(parsed_raw)
                if parsed is None:
                    log_text += "Warning: Could not parse/normalize JSON plane; will auto-estimate.\n"
                else:
                    plane = parsed
                    log_text += f"Using provided ground plane: {plane}\n"
            except Exception as e:
                log_text += f"Warning: Failed to read ground JSON: {e}\n"

        log_text += f"Relative params: voxel_frac={voxel_frac}, ground_tol_frac={ground_tol_frac}, eps_frac={dbscan_eps_frac}\n"
        log_text += f"ROI fractions: low={low_frac}, high={high_frac}\n"
        log_text += f"Leg seed max offset (fraction of H): {leg_seed_max_offset_frac}\n"
        
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
            leg_seed_max_offset_frac=leg_seed_max_offset_frac,
            save_prefix=save_dir
        )

        log_text += "Alignment complete.\n"
        shift = results.get("centroid_shift", np.zeros(3))
        log_text += f"Centroid shift applied: {shift}\n"

        # --- Debug Info for Logs ---
        leg_center_lines = results.get("leg_center_lines", [])
        leg_seeds = results.get("leg_seeds", [])
        log_text += f"DEBUG: Found {len(leg_center_lines)} leg center lines.\n"
        log_text += f"DEBUG: Computed {len(leg_seeds)} leg seeds: {leg_seeds}\n"
        for i, line in enumerate(leg_center_lines):
            log_text += f"DEBUG: Line {i}: {line}\n"

        # --- SAVE OUTPUT ---
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
                mesh_in.translate(results["centroid_shift"], relative=True)
                o3d.io.write_triangle_mesh(out_path, mesh_in, write_ascii=True)
                log_text += f"Saved aligned MESH to: {out_path}\n"
            else:
                o3d.io.write_point_cloud(out_path, results["rotated_pcd"], write_ascii=True)
                log_text += f"Saved aligned POINT CLOUD to: {out_path}\n"
        except Exception as e:
            log_text += f"Warning: Failed to save as mesh ({e}). Saving point cloud instead.\n"
            o3d.io.write_point_cloud(out_path, results["rotated_pcd"], write_ascii=True)

        # Unpack results for debug visualization
        slab_dbg = np.asarray(results["sliced_pcd"].points)
        plane_dbg = results["ground_plane"]
        H_all = results["H"]
        min_b_all = results["min_b"]
        max_b_all = results["max_b"]

        torso_center_xz = np.array([(min_b_all[0] + max_b_all[0]) / 2.0, (min_b_all[2] + max_b_all[2]) / 2.0])

        # --- RUN enhanced debug seeding ---
        seeds_dbg, dbg_info = compute_leg_seeds_from_ground(
            slab_dbg, plane_dbg, H_all,
            max_offset_frac=leg_seed_max_offset_frac,
            ground_tol_frac=ground_tol_frac,
            dbscan_eps_frac=dbscan_eps_frac,
            dbscan_min_samples=dbscan_min_samples,
            n_expected=2,
            torso_center_xz=torso_center_xz,
            torso_radius_frac=0.6,
            min_lateral_sep_frac=0.02,
            debug=True
        )

        # Build detailed Plotly figure from dbg_info
        figs = []
        # slab points (subsample)
        if slab_dbg.shape[0] > 25000:
            idxs = np.random.choice(len(slab_dbg), 25000, replace=False)
            slab_vis = slab_dbg[idxs]
        else:
            slab_vis = slab_dbg
        figs.append(go.Scatter3d(x=slab_vis[:,0], y=slab_vis[:,1], z=slab_vis[:,2],
                                 mode='markers', marker=dict(size=1, color='lightgray', opacity=0.45),
                                 name='Slab points'))

        # colors for offsets/components
        palette = ['red','orange','green','blue','purple','brown']
        # Plot per-offset projected near points (if dbg_info stored projs)
        for oi, off_entry in enumerate(dbg_info.get("offsets", [])):
            color = palette[oi % len(palette)]
            projs = off_entry.get("projs", None)
            if projs is not None and len(projs) > 0:
                # convert projs (Nx3) back to small jittered Y for display -> keep real Y
                p = np.array(projs)
                if p.shape[0] > 15000:
                    idxs = np.random.choice(len(p), 15000, replace=False)
                    p = p[idxs]
                figs.append(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2],
                                         mode='markers', marker=dict(size=1.8, color=color, opacity=0.35),
                                         name=f'Offset {off_entry["offset"]:.4f} near-pts'))
            # also plot components listed for this offset if present
            comps = off_entry.get("components", [])
            for ci, comp in enumerate(comps):
                # comp may be dict with 'pts_xz' or 'poly_world' (depending on path)
                pts_xz = comp.get("pts_xz", None)
                poly_world = None
                if pts_xz is not None:
                    # map XZ -> world (compute Y using plane)
                    xs = pts_xz[:,0]; zs = pts_xz[:,1]
                    if abs(plane_dbg[1]) > 1e-9:
                        ys = (-(plane_dbg[0]*xs + plane_dbg[2]*zs + plane_dbg[3])) / plane_dbg[1]
                    else:
                        ys = np.full_like(xs, slab_dbg[:,1].mean())
                    poly_world = np.vstack([xs, ys, zs]).T
                elif comp.get("poly_world", None) is not None:
                    poly_world = np.array(comp["poly_world"])
                if poly_world is not None:
                    # outline polygon - use first N points to reduce plot size
                    N = poly_world.shape[0]
                    step = max(1, N//300)
                    poly_small = poly_world[::step]
                    figs.append(go.Scatter3d(x=poly_small[:,0], y=poly_small[:,1], z=poly_small[:,2],
                                             mode='lines', line=dict(color=color, width=4),
                                             name=f'Comp {oi}-{ci} outline'))
                    # centroid
                    cx,cy,cz = comp.get("centroid_world", comp.get("centroid", [None,None,None]))
                    if cx is not None:
                        figs.append(go.Scatter3d(x=[cx], y=[cy], z=[cz],
                                                 mode='markers', marker=dict(size=5, color=color, symbol='circle'),
                                                 name=f'Centroid {oi}-{ci}'))
        # Plot all candidate polygons (components_all) semi-transparent
        for i, comp in enumerate(dbg_info.get("components_all", [])):
            poly = comp.get("poly_world", None)
            if poly is None: continue
            poly = np.array(poly)
            if poly.shape[0] < 3: continue
            # reduce resolution for plotting
            step = max(1, poly.shape[0]//400)
            poly_s = poly[::step]
            figs.append(go.Scatter3d(x=poly_s[:,0], y=poly_s[:,1], z=poly_s[:,2],
                                     mode='lines', line=dict(color='rgba(0,150,200,0.6)', width=3),
                                     name=f'Comp_all_{i}'))
        # chosen seeds (magenta diamonds)
        chosen = dbg_info.get("chosen", [])
        if chosen:
            chosen_arr = np.array(chosen)
            figs.append(go.Scatter3d(x=chosen_arr[:,0], y=chosen_arr[:,1], z=chosen_arr[:,2],
                                     mode='markers', marker=dict(size=8, color='magenta', symbol='diamond'),
                                     name='Chosen seeds'))
        # torso center display (XZ -> world approx)
        torso_xz = np.array(dbg_info.get("torso_center_xz", torso_center_xz.tolist()))
        torso_y = (min_b_all[1] + max_b_all[1]) / 2.0
        figs.append(go.Scatter3d(x=[torso_xz[0]], y=[torso_y], z=[torso_xz[1]],
                                 mode='markers', marker=dict(size=7, color='black', symbol='x'),
                                 name='Torso center (XZ)'))
        layout_dbg = go.Layout(scene=dict(aspectmode='data',
                                          xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')),
                               margin=dict(l=0,r=0,t=0,b=0))
        fig = go.Figure(data=figs, layout=layout_dbg)
        
        return fig, log_text

    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}\n{traceback.format_exc()}"

# ---------------------- Gradio UI ----------------------

with gr.Blocks(title="Skeleton Leg Aligner (Relative Params)") as demo:
    gr.Markdown("# Skeleton Leg Aligner — scale-invariant")
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

            gr.Markdown("### Ground plane (optional)")
            gr.Markdown("Upload JSON (many formats supported). If omitted, plane estimated by RANSAC inside pipeline.")
            ground_plane_input = gr.File(label="Ground Plane JSON", file_types=[".json"])

            gr.Markdown("### Leg seed options")
            leg_seed_offset_frac_input = gr.Slider(label="Leg seed plane offset (max, fraction of H)", minimum=0.0, maximum=0.05, value=0.02, step=0.001)

            gr.Markdown("### Output")
            save_dir_input = gr.Textbox(label="Save directory", value="skeleton_out")
            output_filename_input = gr.Textbox(label="Output filename (.ply)", value="aligned_mesh.ply")

            run_btn = gr.Button("Run Alignment", variant="primary")

        with gr.Column():
            model_out = gr.Plot(label="3D Visualization")
            logs_out = gr.Textbox(label="Logs", lines=12)

    run_btn.click(fn=run_skeleton_alignment_interface,
                  inputs=[file_input, voxel_frac_input, low_frac_input, high_frac_input,
                          save_dir_input, output_filename_input, ground_plane_input, ground_tol_frac_input,
                          dbscan_eps_frac_input, dbscan_min_samples_input, verticality_thresh_input, min_cluster_frac_input,
                          geodesic_thresh_input, leg_seed_offset_frac_input],
                  outputs=[model_out, logs_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)

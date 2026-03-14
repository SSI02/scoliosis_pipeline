#!/usr/bin/env python3
"""
skeleton_gradio_relative.py

Full self-contained Gradio app that:
 - Loads a mesh/pointcloud (.ply/.pcd/.obj/.xyz)
 - Accepts an optional ground-plane JSON (many formats supported)
 - Uses ROIs relative to object height (scale-invariant)
 - Voxelizes the ROI
 - Estimates hip seeds and computes geodesic distances to prune disconnected components (hands)
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
from skimage import measure
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

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

# ---------------------- Hip Estimation & Geodesic ----------------------

# Hip estimation function removed as we use feet seeding now.


def compute_geodesic_from_seeds(binary_grid, min_b, voxel_size, seed_world_points, max_nodes_for_graph=350_000):
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
    seed_ids = []
    for hw in seed_world_points:
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

def skeleton_main(pcd_path,
                  low_frac=0.05, high_frac=0.25,
                  voxel_frac=0.01,
                  ground_plane=None, ground_tol_frac=0.02,
                  dbscan_eps_frac=0.015, dbscan_min_samples=5,
                  verticality_thresh=2.0, min_cluster_frac=0.005,
                  geodesic_thresh=0.5,
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

    # [1] ROI extraction and saving (add after loading and computing H):
    # Use ground-aligned ROI extraction if possible
    if ground_plane is not None:
        a, b, c, d = ground_plane
        normal = np.array([a, b, c])
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        distances = (points @ normal) + d  # signed distances
        
        roi_lower = 0.05 * H
        roi_upper = 0.25 * H
        roi_mask = (distances >= roi_lower) & (distances <= roi_upper)
    else:
        # Fallback to axis-aligned
        roi_mask = (points[:, 1] >= min_b[1] + 0.05 * H) & (points[:, 1] <= min_b[1] + 0.25 * H)

    roi_points = points[roi_mask]
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    o3d.io.write_point_cloud(os.path.join(save_prefix, "roi.ply"), roi_pcd, write_ascii=True)

    # Convert relative fractions to absolute values
    voxel_size = voxel_frac * H

    # [2] ROI skeletonization and saving (add after voxelizing ROI):
    roi_grid, roi_min_b, roi_voxel = voxelize_pointcloud_to_grid(roi_points, voxel_size=voxel_size)
    roi_skeleton_grid = run_skeletonization(roi_grid)
    roi_skeleton_pts = skeleton_to_points(roi_skeleton_grid, roi_min_b, roi_voxel)
    roi_skel_pcd = o3d.geometry.PointCloud()
    roi_skel_pcd.points = o3d.utility.Vector3dVector(roi_skeleton_pts)
    o3d.io.write_point_cloud(os.path.join(save_prefix, "roi_skeleton.ply"), roi_skel_pcd, write_ascii=True)
    ground_tol = ground_tol_frac * H
    dbscan_eps = dbscan_eps_frac * H

    # Feet-relative slicing: use ground plane distance!
    # "Slab" for seeding should be very thin near the ground plane
    # If we don't have a ground plane, we can't reliably do this "hole" method properly
    # (since we need to project to a specific plane)
    # But we attempted to auto-estimate above.
    
    if ground_plane is None:
        # Fallback to crude bottom slice if estimation failed completely
        # Assume feet are at min Y
        y_min = points[:,1].min()
        y_slab_top = y_min + (low_frac * H) 
        # But wait, original code assumed max_b[1]? That's suspicious. 
        # Typically feet are at min Y. Let's assume min Y here for fallback.
        mask = (points[:,1] >= y_min) & (points[:,1] <= y_slab_top)
    else:
        # Use ground plane distance
        a,b,c,d = ground_plane
        plane_normal = np.array([a,b,c])
        # Distance = n.p + d (if |n|=1)
        norm_val = np.linalg.norm(plane_normal)
        if norm_val > 1e-6:
            plane_normal = plane_normal / norm_val
            d = d / norm_val
        
        # d is -n.dot(point_on_plane)
        # signed distance
        dists = points @ plane_normal + d
        
        # We want points "close" to the plane. 
        # But "above" or "below"? Feet are usually "above" the floor.
        # But depending on normal orientation, "above" might be positive or negative distance.
        # usually we look for abs(dist) < threshold
        # foot_seed.py checked "positive" distance.
        
        # Heuristic: check where the bulk of points are.
        # If mean(dists) > 0, then body is on positive side.
        # If mean(dists) < 0, body is on negative side.
        if np.mean(dists) < 0:
            dists = -dists # Flip so body is positive
            
        slab_thick = max(0.02 * H, 0.005) # 2% or 5mm
        mask = (dists >= -0.005) & (dists <= slab_thick)

    sliced_pts = points[mask]
    if sliced_pts.shape[0] < 10:
         # Fallback to a simpler slice if plane logic was empty (e.g. plane was far away)
         print("Warning: Plane slab was empty, falling back to min-Y slice.")
         y_min = points[:,1].min()
         mask = points[:,1] <= (y_min + 0.05 * H)
         sliced_pts = points[mask]

    if sliced_pts.shape[0] == 0:
        # Final fallback
        sliced_pts = points
        print("Warning: Using full cloud for seeding (slice failed).")

    if sliced_pts.shape[0] == 0:
        raise RuntimeError("Slice produced zero points. Try changing fractions or check axis.")
    sliced_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sliced_pts))
    o3d.io.write_point_cloud(os.path.join(save_prefix, "sliced.ply"), sliced_pcd)

    # Hip Estimation (Replaced with Feet Seeding)
    
    # Prepare for feet seeding
    # We need slab_points_local where Y is up (aligned to -normal of ground)
    # If ground_plane is available, use it. Else assume Y is up.
    if ground_plane is not None:
        # Match foot_seed.py logic EXACTLY
        a,b,c,d = ground_plane
        plane_normal = np.array([a,b,c], dtype=float)
        norm_val = np.linalg.norm(plane_normal)
        if norm_val == 0:
            raise RuntimeError("Invalid plane normal 0")
        plane_normal = plane_normal / norm_val
        d_val = d / norm_val
        
        # In foot_seed.py: plane_point = -d * plane_normal
        # checking consistency: n.p + d = 0 => n.(-d*n) + d = -d(n.n) + d = -d + d = 0. Correct.
        plane_point = -d_val * plane_normal
        
        # foot_seed.py sets desired_y = -plane_normal
        # This rotation matrix R maps World -> Local
        target_y = -plane_normal
        R_align = rotation_matrix_align_y_to_vector(target_y)

        # Transform to local: subtract plane point first!
        # local_all = (R @ (verts.T - plane_point.reshape(3,1))).T
        # We need this for the seeds to be in the same local frame
        slab_local = (R_align @ (sliced_pts.T - plane_point.reshape(3,1))).T
    else:
        # Fallback if ground_plane is None (though we tried to estimate it)
        slab_local = sliced_pts
        
        
    # Call feet seeding
    kdtree_slab = KDTree(sliced_pts)
    
    leg_seeds, _ = hole_based_seeds_from_slab(
        slab_points_local=slab_local,
        slab_points_world=sliced_pts,
        grid_res=voxel_size, 
        min_hole_area_px=20,
        keep_k=2,
        kdtree_mesh=kdtree_slab
    )
    
    # Use these as the seeds for geodesic
    seeds_for_geodesic = leg_seeds 
    print(f"Found {len(seeds_for_geodesic)} leg seeds via hole detection.")

    # [3] Foot seeding slab save with seeds (after slicing slab):
    slab_pcd = o3d.geometry.PointCloud()
    slab_pcd.points = o3d.utility.Vector3dVector(sliced_pts)
    slab_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    seed_pcd = o3d.geometry.PointCloud()
    seed_pcd.points = o3d.utility.Vector3dVector(np.array(leg_seeds))
    seed_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    combined_slab = slab_pcd + seed_pcd
    o3d.io.write_point_cloud(os.path.join(save_prefix, "foot_seeding_slab.ply"), combined_slab, write_ascii=True)

    # [2a] Geodesic Pruning on ROI Skeleton
    # User Plan: Full Voxel Grid -> Geodesic Distance -> Filter ROI Skeleton

    # 1. Voxelize the FULL cloud to ensure connectivity from feet (bottom) to ROI (up)
    full_grid, full_min_b, full_voxel_used = voxelize_pointcloud_to_grid(points, voxel_size=voxel_size)
    
    # 2. Compute Geodesic Distances on FULL GRID
    # Seeds are already in world, so they map correctly to full grid
    if len(seeds_for_geodesic) == 0:
        seeds_for_geodesic = [points.mean(axis=0)] # Fallback
    
    # Note: this is expensive but necessary for connectivity
    full_dists_grid, _ = compute_geodesic_from_seeds(full_grid, full_min_b, full_voxel_used, seeds_for_geodesic)
    
    # 3. Skeletonize the ROI Grid
    roi_skeleton_grid = run_skeletonization(roi_grid)
    raw_skel_pts = skeleton_to_points(roi_skeleton_grid, roi_min_b, roi_voxel)
    
    # 4. Prune the SKELETON based on geodesic distance in FULL GRID
    pruned_pts = []
    
    for p in raw_skel_pts:
        # map to full grid
        idx = world_to_voxel_index(p, full_min_b, full_voxel_used, full_grid.shape)
        # Check boundary just in case (though points are from subset)
        if (0 <= idx[0] < full_grid.shape[0] and 
            0 <= idx[1] < full_grid.shape[1] and 
            0 <= idx[2] < full_grid.shape[2]):
            
            dist = full_dists_grid[idx]
            if dist <= geodesic_thresh:
                pruned_pts.append(p)
        else:
             # Should not happen if ROI is subset of Points, but floating point tolerance...
             pass
            
    skel_pts = np.array(pruned_pts)
    
    if len(skel_pts) == 0:
        raise RuntimeError("No skeleton points extracted. Try larger voxel_frac, larger geodesic_thresh, or check seeds.")

    o3d.io.write_point_cloud(os.path.join(save_prefix, "skeleton_points_vis.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(skel_pts)))

    # [4] Filtered skeleton after geodesic pruning:
    filtered_skel_pcd = o3d.geometry.PointCloud()
    filtered_skel_pcd.points = o3d.utility.Vector3dVector(skel_pts)
    o3d.io.write_point_cloud(os.path.join(save_prefix, "skeleton_filtered.ply"), filtered_skel_pcd, write_ascii=True)

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
    # ---- Replace PCA-based fit/intersection with plane-projection + circle-fit ----
    def _make_plane_basis(n):
        # n must be unit
        n = n / (np.linalg.norm(n) + 1e-12)
        # pick arbitrary vector not parallel to n
        arb = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(arb, n)) > 0.9:
            arb = np.array([1.0, 0.0, 0.0])
        u = np.cross(arb, n)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    def project_to_plane_pts(pts, plane_n, plane_d, plane_point, u, v):
        # returns Nx2 coordinates in (u,v) basis relative to plane_point
        # plane_n must be unit
        rel = pts - plane_point[None, :]
        # orthogonal projection already done by removing normal component if needed,
        # but here we'll project each point onto plane point-by-point:
        dists = (pts @ plane_n) + plane_d
        proj = pts - np.outer(dists, plane_n)  # projected 3D points on plane
        rel_proj = proj - plane_point[None, :]
        xy = np.stack([rel_proj.dot(u), rel_proj.dot(v)], axis=1)
        return xy, proj

    def fit_circle_kasa(xy):
        # Kasa linear least-squares circle fit: x^2 + y^2 + A x + B y + C = 0
        if xy.shape[0] < 6:
            return None  # not enough points
        X = np.column_stack([xy[:,0], xy[:,1], np.ones(len(xy))])
        Y = -(xy[:,0]**2 + xy[:,1]**2)
        try:
            sol, *_ = np.linalg.lstsq(X, Y, rcond=None)
            A, B, C = sol
            cx = -A/2.0
            cy = -B/2.0
            rad_sq = cx*cx + cy*cy - C
            if rad_sq <= 0 or not np.isfinite(rad_sq):
                return None
            r = np.sqrt(rad_sq)
            return (cx, cy, r)
        except Exception:
            return None

    # prepare normalized plane normal and plane point
    if ground_plane is not None:
        a,b,c,d = ground_plane
        n = np.array([a,b,c], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise RuntimeError("Invalid ground plane normal.")
        n = n / n_norm
        d = d / n_norm
        plane_point = -d * n  # one point on the plane
    else:
        # fallback: assume Y-up world and plane at min Y
        n = np.array([0.0, 1.0, 0.0])
        plane_point = np.array([0.0, points[:,1].min(), 0.0])

    # build orthonormal basis (u,v) spanning the plane
    u, v = _make_plane_basis(n)

    ground_centers_3d = []
    center_debug = []  # for optional diagnostics

    for i, cpts in enumerate(clusters[:2]):
        if cpts is None or len(cpts) < 6:
            # too few points — fallback to PCA method below
            center_debug.append({"cluster": i, "reason": "not_enough_points", "size": 0})
            ground_centers_3d.append(None)
            continue

        # Project pruned-skeleton cluster points to plane and get 2D coords
        xy, proj3d = project_to_plane_pts(cpts, n, d, plane_point, u, v)

        # Optionally filter out outliers in XY (RANSAC-like pruning)
        # compute centroid and keep points within 2*sigma
        med = np.median(xy, axis=0)
        dists2 = np.sum((xy - med)**2, axis=1)
        thr = np.percentile(dists2, 90) * 4.0
        mask = dists2 <= (thr + 1e-12)
        if np.count_nonzero(mask) < max(6, int(0.5 * len(xy))):
            mask = np.ones(len(xy), dtype=bool)  # if too aggressive, keep all

        xy_f = xy[mask]

        # Fit circle in plane coordinates
        circ = fit_circle_kasa(xy_f)
        if circ is None:
            # fitting failed -> fallback
            center_debug.append({"cluster": i, "reason": "circle_fit_failed", "n_pts": len(xy_f)})
            ground_centers_3d.append(None)
            continue

        cx, cy, r = circ
        # convert center back to 3D: plane_point + cx*u + cy*v
        center3d = plane_point + (u * cx) + (v * cy)
        ground_centers_3d.append(center3d)
        center_debug.append({"cluster": i, "reason": "ok", "cx": cx, "cy": cy, "r": r, "n_pts": len(xy_f)})

    # If circle-based centers insufficient (None), fallback to PCA-based intersection for those clusters
    for i in range(len(ground_centers_3d)):
        if ground_centers_3d[i] is None:
            # try PCA / line intersection as before
            try:
                cpts = clusters[i]
                mean, dir_vec = fit_line_through_points(cpts)
                if mean is not None:
                    denom = n.dot(dir_vec)
                    if abs(denom) < 1e-8:
                        # fallback to use mean projected to plane
                        proj_mean = mean - ( (mean @ n) + d ) * n
                        ground_centers_3d[i] = proj_mean
                        center_debug.append({"cluster": i, "reason": "pca_fallback_proj_mean"})
                    else:
                        t = - (n.dot(mean) + d) / denom
                        inter = mean + t * dir_vec
                        ground_centers_3d[i] = inter
                        center_debug.append({"cluster": i, "reason": "pca_fallback_intersect", "denom": float(denom)})
                else:
                    center_debug.append({"cluster": i, "reason": "pca_fit_failed"})
                    ground_centers_3d[i] = None
            except Exception as e:
                center_debug.append({"cluster": i, "reason": "pca_exception", "err": str(e)})
                ground_centers_3d[i] = None

    # collect valid centers
    valid_centers = [c for c in ground_centers_3d if c is not None]
    if len(valid_centers) >= 2:
        # pick two farthest centers if more than 2
        gpts_arr = np.vstack(valid_centers)
        if gpts_arr.shape[0] > 2:
            dmat = np.sum((gpts_arr[:, None, :] - gpts_arr[None, :, :])**2, axis=-1)
            i1, j1 = np.unravel_index(np.argmax(dmat), dmat.shape)
            p1 = gpts_arr[i1]; p2 = gpts_arr[j1]
        else:
            p1, p2 = gpts_arr[0], gpts_arr[1]
    else:
        # ultimate fallback: use PCA lateral extremes on skel_pts
        try:
            pca = PCA(n_components=2)
            pca.fit(skel_pts[:, [0,2]])
            proj = (skel_pts[:, [0,2]] - pca.mean_).dot(pca.components_[0])
            pmin = skel_pts[np.argmin(proj)]
            pmax = skel_pts[np.argmax(proj)]
            p1, p2 = pmin, pmax
            center_debug.append({"reason": "ultimate_fallback_pca_extremes"})
        except Exception:
            raise RuntimeError("Failed to compute foot centers by circle fit or fallbacks.")

    # now p1,p2 are the ground centers (use them further as before)
    ground_points = [p1, p2]
    # the rest of your pipeline (mediolateral vector, origin, rotation) can continue using p1,p2

    # optional: save debug info to disk for inspection
    try:
        import json
        with open(os.path.join(save_prefix, "circle_fit_debug.json"), "w") as jf:
            json.dump(center_debug, jf, default=lambda x: float(x) if isinstance(x, np.floating) else (x.tolist() if isinstance(x, np.ndarray) else x), indent=2)
    except Exception:
        pass

    # [5] Ground intersection points:
    gpts = np.vstack(ground_points)
    ground_inter_pcd = o3d.geometry.PointCloud()
    ground_inter_pcd.points = o3d.utility.Vector3dVector(gpts)
    o3d.io.write_point_cloud(os.path.join(save_prefix, "ground_intersection_points.ply"), ground_inter_pcd, write_ascii=True)
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

    # Save outputs
    o3d.io.write_point_cloud(os.path.join(save_prefix, "rotated_full.ply"), rotated_pcd)
    np.save(os.path.join(save_prefix, "skeleton_points.npy"), skel_pts)
    np.save(os.path.join(save_prefix, "ground_points.npy"), np.vstack(ground_points))
    np.save(os.path.join(save_prefix, "rotation_matrix.npy"), R)
    np.save(os.path.join(save_prefix, "origin_point.npy"), origin_point)
    np.save(os.path.join(save_prefix, "centroid_shift.npy"), centroid_shift)
    
    # [Verification] Save aligned mesh with foot points
    # Transform ground points to aligned frame
    gpts = np.vstack(ground_points)
    gpts_pcd = o3d.geometry.PointCloud()
    gpts_pcd.points = o3d.utility.Vector3dVector(gpts)
    # Apply R, origin, and shift to match rotated_pcd
    gpts_pcd.translate(-origin_point, relative=False)
    gpts_pcd.rotate(R, center=(0,0,0))
    gpts_pcd.translate(origin_point, relative=False)
    gpts_pcd.translate(centroid_shift, relative=True)
    
    gpts_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red
    
    mesh_copy = type(rotated_pcd)() 
    mesh_copy.points = rotated_pcd.points
    mesh_copy.colors = rotated_pcd.colors
    # If rotated_pcd is just a point cloud, we can use copy.
    # Note: open3d copy might be safer.
    import copy
    mesh_copy = copy.deepcopy(rotated_pcd)
    mesh_copy.paint_uniform_color([0.7, 0.7, 0.7]) # Light Gray
    
    combined = mesh_copy + gpts_pcd
    o3d.io.write_point_cloud(os.path.join(save_prefix, "aligned_mesh_with_footpoints.ply"), combined, write_ascii=True)

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
        "leg_seeds": seeds_for_geodesic,
        "line_params": line_params,
        "leg_center_lines": leg_center_lines
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
      - The hip seeds
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
        # Robustly check if line data exists for this index
        if i < len(leg_center_lines) and leg_center_lines[i] is not None:
            p_start, p_end = leg_center_lines[i]
            
            line_seg = np.vstack([p_start, p_end])
            t_line_seg = transform_pts(line_seg)
            
            figs_data.append(go.Scatter3d(
                x=t_line_seg[:,0], y=t_line_seg[:,1], z=t_line_seg[:,2],
                mode='lines',
                opacity=1.0,
                line=dict(color=colors[i], width=10), # Thick line
                name=f'Leg {i+1} Axis'
            ))
        else:
             # FORCE a legend entry even if missing, so user knows it failed
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
        t_seeds = transform_pts(leg_seeds)
        figs_data.append(go.Scatter3d(
            x=t_seeds[:,0], y=t_seeds[:,1], z=t_seeds[:,2],
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
                                     geodesic_thresh):
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
            save_prefix=save_dir
        )

        log_text += "Alignment complete.\n"
        shift = results.get("centroid_shift", np.zeros(3))
        log_text += f"Centroid shift applied: {shift}\n"

        # --- Debug Info for Logs ---
        leg_center_lines = results.get("leg_center_lines", [])
        log_text += f"DEBUG: Found {len(leg_center_lines)} leg center lines.\n"
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
                mesh_in.translate(shift, relative=True)
                o3d.io.write_triangle_mesh(out_path, mesh_in, write_ascii=True)
                log_text += f"Saved aligned MESH to: {out_path}\n"
            else:
                o3d.io.write_point_cloud(out_path, results["rotated_pcd"], write_ascii=True)
                log_text += f"Saved aligned POINT CLOUD to: {out_path}\n"
        except Exception as e:
            log_text += f"Warning: Failed to save as mesh ({e}). Saving point cloud instead.\n"
            o3d.io.write_point_cloud(out_path, results["rotated_pcd"], write_ascii=True)

        fig = create_plotly_visualization(results)
        
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
                          geodesic_thresh_input],
                  outputs=[model_out, logs_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
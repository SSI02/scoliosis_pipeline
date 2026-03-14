#!/usr/bin/env python3
"""
CLI wrapper for point cloud denoising.
Usage:
    python denoising_cli.py --input /path/to/input.ply --output /path/to/output.ply
    python denoising_cli.py --input /path/to/input.ply --output /path/to/output.ply --use-sor --use-ror
"""

import os
import argparse
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def load_ply(path: str) -> o3d.geometry.PointCloud:
    """Load a PLY file as point cloud."""
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud from {path}")
    return pcd


def denoise_statistical(pcd: o3d.geometry.PointCloud, 
                       nb_neighbors: int = 20, 
                       std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """Statistical Outlier Removal (SOR)."""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def denoise_radius(pcd: o3d.geometry.PointCloud, 
                   nb_points: int = 16, 
                   radius: float = 0.05) -> o3d.geometry.PointCloud:
    """Radius Outlier Removal (ROR)."""
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pcd.select_by_index(ind)


def denoise_dbscan(pcd: o3d.geometry.PointCloud, 
                   eps: float = 0.05, 
                   min_points: int = 10,
                   keep_largest: bool = True) -> o3d.geometry.PointCloud:
    """DBSCAN clustering-based denoising."""
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


def compute_nn_stats(points: np.ndarray, k: int = 6) -> dict:
    """Compute nearest neighbor statistics for auto-parameter suggestion."""
    pts = np.asarray(points)
    N = pts.shape[0]
    
    if N < 2:
        return {'median_nn': 1e-6, 'mean_nn': 1e-6}
    
    k_eff = max(1, min(k, N - 1))
    
    if np.allclose(np.var(pts, axis=0), 0.0):
        pts = pts + (np.random.RandomState(0).randn(*pts.shape) * 1e-9)
    
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k_eff + 1)
    nn = dists[:, 1:]
    
    median_nn = float(np.median(nn))
    mean_nn = float(np.mean(nn))
    
    if median_nn <= 0.0:
        median_nn = 1e-6
    if mean_nn <= 0.0:
        mean_nn = median_nn
    
    return {'median_nn': median_nn, 'mean_nn': mean_nn}


def autosuggest_parameters(points: np.ndarray) -> dict:
    """Suggest denoising parameters based on point cloud statistics."""
    pts = np.asarray(points)
    n = max(0, pts.shape[0])
    
    if n < 10:
        med = 1e-3
        return {
            'sor_n': 8, 'sor_std': 1.5,
            'ror_n': 6, 'ror_radius': max(1e-6, med * 2.0),
            'db_eps': max(1e-6, med * 1.5), 'db_min': 6
        }
    
    stats = compute_nn_stats(pts, k=20)
    med = stats['median_nn']
    
    return {
        'sor_n': int(max(8, min(70, int(np.sqrt(n) / 2)))),
        'sor_std': 1.5,
        'ror_n': int(max(6, min(40, int(np.sqrt(n) / 10)))),
        'ror_radius': float(max(1e-6, med * 2.0)),
        'db_eps': float(max(1e-6, med * 1.5)),
        'db_min': int(max(6, min(30, int(np.sqrt(n) / 20) * 2 + 6)))
    }


def denoise_point_cloud(input_path: str, output_path: str,
                       use_sor: bool = True, sor_n: int = None, sor_std: float = None,
                       use_ror: bool = True, ror_n: int = None, ror_radius: float = None,
                       use_dbscan: bool = False, db_eps: float = None, db_min: int = None,
                       db_keep_largest: bool = True,
                       auto_params: bool = True) -> dict:
    """
    Main denoising pipeline.
    
    Args:
        input_path: Path to input PLY file
        output_path: Path for output PLY file
        use_sor: Enable Statistical Outlier Removal
        use_ror: Enable Radius Outlier Removal
        use_dbscan: Enable DBSCAN clustering
        auto_params: Auto-suggest parameters if not provided
        
    Returns:
        dict: Statistics about the denoising process
    """
    # Load point cloud
    print(f"Loading: {input_path}")
    pcd = load_ply(input_path)
    original_count = len(pcd.points)
    print(f"Original points: {original_count}")
    
    # Auto-suggest parameters
    if auto_params:
        suggested = autosuggest_parameters(np.asarray(pcd.points))
        print(f"Suggested parameters: {suggested}")
        
        if sor_n is None:
            sor_n = suggested['sor_n']
        if sor_std is None:
            sor_std = suggested['sor_std']
        if ror_n is None:
            ror_n = suggested['ror_n']
        if ror_radius is None:
            ror_radius = suggested['ror_radius']
        if db_eps is None:
            db_eps = suggested['db_eps']
        if db_min is None:
            db_min = suggested['db_min']
    else:
        # Defaults
        sor_n = sor_n or 30
        sor_std = sor_std or 2.0
        ror_n = ror_n or 16
        ror_radius = ror_radius or 0.05
        db_eps = db_eps or 0.05
        db_min = db_min or 10
    
    current = pcd
    
    # Apply SOR
    if use_sor:
        print(f"Applying SOR (k={sor_n}, std={sor_std})...")
        current = denoise_statistical(current, nb_neighbors=int(sor_n), std_ratio=float(sor_std))
        print(f"  After SOR: {len(current.points)} points")
    
    # Apply ROR
    if use_ror:
        print(f"Applying ROR (n={ror_n}, radius={ror_radius:.4f})...")
        current = denoise_radius(current, nb_points=int(ror_n), radius=float(ror_radius))
        print(f"  After ROR: {len(current.points)} points")
    
    # Apply DBSCAN
    if use_dbscan:
        print(f"Applying DBSCAN (eps={db_eps:.4f}, min={db_min})...")
        current = denoise_dbscan(current, eps=float(db_eps), min_points=int(db_min), 
                                keep_largest=db_keep_largest)
        print(f"  After DBSCAN: {len(current.points)} points")
    
    final_count = len(current.points)
    removed = original_count - final_count
    
    # Save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    o3d.io.write_point_cloud(output_path, current)
    print(f"Saved: {output_path}")
    
    return {
        "original_points": original_count,
        "final_points": final_count,
        "removed_points": removed,
        "removal_percentage": (removed / original_count * 100) if original_count > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Point cloud denoising with SOR/ROR/DBSCAN")
    parser.add_argument("--input", "-i", required=True, help="Input PLY file")
    parser.add_argument("--output", "-o", required=True, help="Output PLY file")
    
    # Filter selection
    parser.add_argument("--use-sor", action="store_true", default=True,
                        help="Use Statistical Outlier Removal (default: True)")
    parser.add_argument("--no-sor", action="store_true", help="Disable SOR")
    parser.add_argument("--use-ror", action="store_true", default=True,
                        help="Use Radius Outlier Removal (default: True)")
    parser.add_argument("--no-ror", action="store_true", help="Disable ROR")
    parser.add_argument("--use-dbscan", action="store_true", 
                        help="Use DBSCAN clustering (default: False)")
    
    # SOR parameters
    parser.add_argument("--sor-n", type=int, help="SOR neighbor count")
    parser.add_argument("--sor-std", type=float, help="SOR std ratio")
    
    # ROR parameters
    parser.add_argument("--ror-n", type=int, help="ROR min neighbor count")
    parser.add_argument("--ror-radius", type=float, help="ROR radius")
    
    # DBSCAN parameters
    parser.add_argument("--db-eps", type=float, help="DBSCAN epsilon")
    parser.add_argument("--db-min", type=int, help="DBSCAN min points")
    parser.add_argument("--db-keep-largest", action="store_true", default=True,
                        help="Keep only largest cluster in DBSCAN")
    
    parser.add_argument("--no-auto-params", action="store_true",
                        help="Disable auto parameter suggestion")
    
    args = parser.parse_args()
    
    use_sor = not args.no_sor
    use_ror = not args.no_ror
    
    print(f"\n{'='*60}")
    print("Point Cloud Denoising")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"SOR: {use_sor}, ROR: {use_ror}, DBSCAN: {args.use_dbscan}")
    print(f"{'='*60}\n")
    
    results = denoise_point_cloud(
        input_path=args.input,
        output_path=args.output,
        use_sor=use_sor,
        sor_n=args.sor_n,
        sor_std=args.sor_std,
        use_ror=use_ror,
        ror_n=args.ror_n,
        ror_radius=args.ror_radius,
        use_dbscan=args.use_dbscan,
        db_eps=args.db_eps,
        db_min=args.db_min,
        db_keep_largest=args.db_keep_largest,
        auto_params=not args.no_auto_params
    )
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Original: {results['original_points']} points")
    print(f"Final:    {results['final_points']} points")
    print(f"Removed:  {results['removed_points']} ({results['removal_percentage']:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

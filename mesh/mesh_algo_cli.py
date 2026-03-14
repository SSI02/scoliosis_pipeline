#!/usr/bin/env python3
"""
CLI wrapper for point cloud to mesh reconstruction.
Usage:
    python mesh_algo_cli.py --input /path/to/input.ply --output /path/to/output.ply
"""

import os
import argparse
import numpy as np

try:
    import pymeshlab
except ImportError:
    print("PyMeshLab not installed. Install with: pip install pymeshlab")
    exit(1)


def reconstruct_mesh(input_path: str, output_path: str,
                    normal_neighbours: int = 500,
                    normal_smooth_iter: int = 5,
                    poisson_depth: int = 8,
                    hc_iterations: int = 2) -> dict:
    """
    Reconstruct mesh from point cloud using Screened Poisson Surface Reconstruction.
    
    Args:
        input_path: Path to input point cloud (PLY/XYZ/PTS/PCD)
        output_path: Path for output mesh (PLY)
        normal_neighbours: Number of neighbors for normal computation
        normal_smooth_iter: Smoothing iterations for normals
        poisson_depth: Octree depth for Poisson reconstruction
        hc_iterations: HC Laplacian smoothing iterations
        
    Returns:
        dict: Statistics about the reconstruction
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Initialize MeshLab MeshSet
    ms = pymeshlab.MeshSet()
    
    # Load point cloud
    print(f"Loading point cloud: {input_path}")
    ms.load_new_mesh(input_path)
    initial_vertices = ms.current_mesh().vertex_number()
    print(f"  Loaded {initial_vertices} points")
    
    # Step 1: Compute Normals
    print(f"Computing normals (k={normal_neighbours}, smooth={normal_smooth_iter})...")
    ms.compute_normal_for_point_clouds(
        k=normal_neighbours,
        smoothiter=normal_smooth_iter,
        flipflag=False,
        viewpos=np.array([0.0, 0.0, 0.0])
    )
    print("  ✓ Normals computed")
    
    # Step 2: Screened Poisson Surface Reconstruction
    print(f"Running Screened Poisson reconstruction (depth={poisson_depth})...")
    ms.generate_surface_reconstruction_screened_poisson(
        depth=poisson_depth,
        fulldepth=5,
        cgdepth=0,
        scale=1.1,
        samplespernode=1.5,
        pointweight=4.0,
        iters=8,
        confidence=False,
        preclean=False
    )
    post_poisson_vertices = ms.current_mesh().vertex_number()
    post_poisson_faces = ms.current_mesh().face_number()
    print(f"  ✓ Mesh created: {post_poisson_vertices} vertices, {post_poisson_faces} faces")
    
    # Step 3: Laplacian Smoothing
    print(f"Applying Laplacian smoothing ({hc_iterations} iterations)...")
    ms.apply_coord_laplacian_smoothing(
        stepsmoothnum=hc_iterations,
        cotangentweight=False,
        selected=False
    )
    print("  ✓ Smoothing complete")
    
    # Save mesh
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ms.save_current_mesh(output_path)
    print(f"Saved mesh to: {output_path}")
    
    final_vertices = ms.current_mesh().vertex_number()
    final_faces = ms.current_mesh().face_number()
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    return {
        "input_points": initial_vertices,
        "output_vertices": final_vertices,
        "output_faces": final_faces,
        "file_size_mb": file_size_mb
    }


def main():
    parser = argparse.ArgumentParser(description="Point cloud to mesh reconstruction")
    parser.add_argument("--input", "-i", required=True,
                        help="Input point cloud file (PLY/XYZ/PTS/PCD/OBJ/STL)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output mesh file (PLY)")
    
    # Normal computation parameters
    parser.add_argument("--normal-neighbours", type=int, default=500,
                        help="Neighbors for normal computation (default: 500)")
    parser.add_argument("--normal-smooth-iter", type=int, default=5,
                        help="Normal smoothing iterations (default: 5)")
    
    # Poisson parameters
    parser.add_argument("--poisson-depth", type=int, default=8,
                        help="Octree depth for Poisson (default: 8, higher=more detail)")
    
    # Smoothing parameters
    parser.add_argument("--hc-iterations", type=int, default=2,
                        help="HC Laplacian smoothing iterations (default: 2)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Point Cloud to Mesh Reconstruction")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Parameters:")
    print(f"  Normal neighbors:     {args.normal_neighbours}")
    print(f"  Normal smooth iter:   {args.normal_smooth_iter}")
    print(f"  Poisson depth:        {args.poisson_depth}")
    print(f"  HC smooth iterations: {args.hc_iterations}")
    print(f"{'='*60}\n")
    
    results = reconstruct_mesh(
        input_path=args.input,
        output_path=args.output,
        normal_neighbours=args.normal_neighbours,
        normal_smooth_iter=args.normal_smooth_iter,
        poisson_depth=args.poisson_depth,
        hc_iterations=args.hc_iterations
    )
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Input points:    {results['input_points']:,}")
    print(f"Output vertices: {results['output_vertices']:,}")
    print(f"Output faces:    {results['output_faces']:,}")
    print(f"File size:       {results['file_size_mb']:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

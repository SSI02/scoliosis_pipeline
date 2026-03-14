#!/usr/bin/env python3
"""
verify_3d_skeleton_cylinder.py

Create a synthetic cylinder volume, run true 3D skeletonization (skimage.morphology.skeletonize_3d),
extract ordered centerline, save outputs (PLY + NPY), and optionally visualize.

Usage:
    python verify_3d_skeleton_cylinder.py [--radius R] [--height H] [--voxel V]
                                         [--outdir DIR] [--no-plot]

Dependencies:
    pip install numpy scikit-image scikit-learn matplotlib
"""
import os
import argparse
import numpy as np

# try importing true volumetric skeletonizer
try:
    from skimage.morphology import skeletonize_3d
except Exception as e:
    raise ImportError(
        "skimage.morphology.skeletonize_3d is required for true 3D skeletonization.\n"
        "Please install/upgrade scikit-image (e.g. `pip install -U scikit-image`)."
    ) from e

from sklearn.decomposition import PCA

# Optional plotting imports (only if user requests)
def write_ply_points(path, points):
    """Write Nx3 points to a simple ASCII PLY (no colors)."""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{float(p[0])} {float(p[1])} {float(p[2])}\n")

def build_cylinder_volume(radius, height, voxel_size, margin_voxels=2):
    """
    Returns:
      cyl_volume: ndarray (Ny, Nx, Nz) boolean voxel occupancy (True = occupied)
      grid_axes: (x_array, y_array, z_array) the coordinates of voxel grid cell *starts* (min values)
                 use + voxel_size*0.5 to get centers
    Notes:
      - We use meshgrid with indexing='xy' to produce array ordering (Y, X, Z).
    """
    margin = voxel_size * margin_voxels
    x = np.arange(-radius - margin, radius + margin + 1e-12, voxel_size)
    y = np.arange(-height/2 - margin, height/2 + margin + 1e-12, voxel_size)
    z = np.arange(-radius - margin, radius + margin + 1e-12, voxel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')  # yields shape (Ny, Nx, Nz)
    cyl = (X**2 + Z**2) <= (radius + 1e-12)**2
    return cyl.astype(np.uint8), (x, y, z)

def indices_to_centers(indices, axes, voxel_size):
    """
    Map index triples (iy, ix, iz) to world-space voxel centers (x,y,z).
    axes = (x_array, y_array, z_array) where each array gives the coordinate of the voxel origin (min).
    We return Nx3 array of centers.
    """
    x_axis, y_axis, z_axis = axes
    ix = indices[:,1]
    iy = indices[:,0]
    iz = indices[:,2]
    xs = x_axis[ix] + voxel_size*0.5
    ys = y_axis[iy] + voxel_size*0.5
    zs = z_axis[iz] + voxel_size*0.5
    return np.column_stack([xs, ys, zs])

def compute_ordered_centerline(skel_pts):
    """
    Given skeleton points (Mx3), compute PCA main axis and return skeleton points
    sorted along projection onto that axis (produces an ordered centerline).
    """
    if len(skel_pts) == 0:
        return skel_pts, None
    pca = PCA(n_components=3)
    pca.fit(skel_pts)
    main_axis = pca.components_[0]
    proj = (skel_pts - skel_pts.mean(axis=0)).dot(main_axis)
    order = np.argsort(proj)
    return skel_pts[order], main_axis

def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify 3D skeletonize_3d on a synthetic cylinder")
    parser.add_argument("--radius", type=float, default=0.5, help="Cylinder radius (meters)")
    parser.add_argument("--height", type=float, default=2.0, help="Cylinder height (meters)")
    parser.add_argument("--voxel", type=float, default=0.02, help="Voxel size (meters)")
    parser.add_argument("--outdir", type=str, default="cylinder_skel_verify", help="Output directory")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Do not show matplotlib plot")
    parser.add_argument("--plot", dest="plot", action="store_true", help="Show matplotlib plot (default: show)")
    parser.set_defaults(plot=True)
    args = parser.parse_args(argv)

    radius = float(args.radius)
    height = float(args.height)
    voxel_size = float(args.voxel)
    out_dir = args.outdir
    do_plot = bool(args.plot)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Building cylinder: radius={radius} m, height={height} m, voxel={voxel_size} m")
    cyl_vol, axes = build_cylinder_volume(radius, height, voxel_size)
    print("Volume shape (Ny, Nx, Nz):", cyl_vol.shape)
    print("Occupied voxels (count):", int(cyl_vol.sum()))

    # Run true 3D skeletonization
    print("Running skeletonize_3d (this may take a moment)...")
    skel = skeletonize_3d(cyl_vol)
    skel_count = int(skel.sum())
    print("Skeleton voxels (count):", skel_count)

    # Extract indices and convert to world-space centers
    occ_idx = np.argwhere(cyl_vol > 0)   # (iy, ix, iz)
    skel_idx = np.argwhere(skel > 0)

    occ_pts = indices_to_centers(occ_idx, axes, voxel_size)
    skel_pts = indices_to_centers(skel_idx, axes, voxel_size)

    # Save PLYs and numpy arrays
    ply_occ = os.path.join(out_dir, "cylinder_voxels.ply")
    ply_skel = os.path.join(out_dir, "cylinder_skeleton_points.ply")
    np_occ = os.path.join(out_dir, "cylinder_voxels.npy")
    np_skel = os.path.join(out_dir, "cylinder_skeleton_pts.npy")

    print(f"Writing PLYs to: {out_dir}")
    write_ply_points(ply_occ, occ_pts)
    write_ply_points(ply_skel, skel_pts)

    np.save(np_occ, occ_pts)
    np.save(np_skel, skel_pts)

    # Compute ordered centerline
    centerline_ordered, main_axis = compute_ordered_centerline(skel_pts)
    np.save(os.path.join(out_dir, "cylinder_centerline_ordered.npy"), centerline_ordered)

    # Diagnostics
    print("Occupied points shape:", occ_pts.shape)
    print("Skeleton points shape:", skel_pts.shape)
    if main_axis is not None:
        print("PCA main axis (unit):", np.array2string(main_axis, precision=6))
    if centerline_ordered.shape[0] > 0:
        print("Centerline endpoints (first,last):")
        print(centerline_ordered[0], centerline_ordered[-1])

    print("\nSaved files:")
    print(" - Cylinder voxels PLY:", ply_occ)
    print(" - Cylinder skeleton PLY:", ply_skel)
    print(" - Numpy arrays:", np_occ, np_skel,
          os.path.join(out_dir, "cylinder_centerline_ordered.npy"))

    # Optional plotting
    if do_plot:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # ensures 3d projection available
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')

            # Subsample for plotting if too many points
            max_plot_occ = 8000
            if occ_pts.shape[0] > max_plot_occ:
                idx = np.random.choice(occ_pts.shape[0], max_plot_occ, replace=False)
                occ_plot = occ_pts[idx]
            else:
                occ_plot = occ_pts

            ax.scatter(occ_plot[:,0], occ_plot[:,1], occ_plot[:,2], s=1, label='voxels')
            if skel_pts.shape[0] > 0:
                ax.scatter(skel_pts[:,0], skel_pts[:,1], skel_pts[:,2], s=20, label='skeleton')
            if centerline_ordered.shape[0] > 0:
                ax.plot(centerline_ordered[:,0], centerline_ordered[:,1], centerline_ordered[:,2],
                        linewidth=2, label='centerline')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Cylinder voxels (subsampled), skeleton points, and ordered centerline')
            ax.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Plotting failed (matplotlib may be missing):", e)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
neck_hip_offset_relative.py

Fully relative, scale-invariant estimation of horizontal offset
between neck center and hip center using sliding Y-normal slabs.

Axis convention (standard):
- X-axis: from left shoulder to right shoulder
- Y-axis: high value towards upper body, low value towards foot
- Z-axis: high value towards hips, low value towards neck (spine axis)

All parameters are derived from mesh statistics.
Safe guards prevent empty reductions.
"""

import argparse
import numpy as np
import open3d as o3d

# --------------------------------------------------
# Loading utilities
# --------------------------------------------------

def load_as_pcd(path, target_points_frac=0.15):
    geom = o3d.io.read_triangle_mesh(path)
    if geom.has_triangles():
        geom.compute_vertex_normals()
        n_vertices = np.asarray(geom.vertices).shape[0]
        n_sample = max(int(target_points_frac * n_vertices), 5000)
        pcd = geom.sample_points_uniformly(number_of_points=n_sample)
    else:
        pcd = o3d.io.read_point_cloud(path)
    return pcd

# --------------------------------------------------
# Core computation (fully relative)
# --------------------------------------------------

def compute_neck_hip_relative(pcd, slice_frac=1/50):
    pts = np.asarray(pcd.points)

    if pts.shape[0] < 200:
        raise RuntimeError("Point cloud too sparse for reliable estimation")

    # ---- mesh statistics ----
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    x_min, y_min, z_min = bbox_min
    x_max, y_max, z_max = bbox_max

    X_range = x_max - x_min
    Y_range = y_max - y_min

    if Y_range <= 0 or X_range <= 0:
        raise RuntimeError("Degenerate bounding box detected")

    N = pts.shape[0]

    # ---- fully relative parameters ----
    slice_thickness = Y_range * slice_frac
    min_pts = max(int(0.001 * N), 40)   # relative density threshold

    # --------------------------------------------------
    # HIP CENTER (low Y = foot region, but we need to find hips)
    # Actually, hips are at high Z, so we should slice along Z-axis
    # But this code uses Y-axis slicing, so we'll adapt:
    # In standard convention: Y is vertical (high = upper body, low = foot)
    # Hips are typically in the middle-lower Y region
    # --------------------------------------------------
    hip_center = None
    # Search lower 30% of Y-range (towards foot, but hips are above feet)
    y = y_min + 0.30 * Y_range
    y_stop = y_min + 0.10 * Y_range

    while y > y_stop:
        slab = pts[(pts[:,1] >= y - slice_thickness) & (pts[:,1] < y)]
        if slab.shape[0] >= min_pts:
            hip_center = slab.mean(axis=0)
            break
        y -= slice_thickness

    if hip_center is None:
        raise RuntimeError(
            "Hip center not found. Mesh may be incomplete near pelvis or too sparse."
        )

    # --------------------------------------------------
    # NECK CENTER (bottleneck detection)
    # Search upper torso band (high Y = upper body/head region)
    # --------------------------------------------------
    records = []

    y_low = y_max - 0.70 * Y_range  # Start from upper body
    y_high = y_max - 0.15 * Y_range  # Towards head

    y = y_low
    while y < y_high:
        slab = pts[(pts[:,1] >= y) & (pts[:,1] < y + slice_thickness)]
        if slab.shape[0] >= min_pts:
            x_range = slab[:,0].max() - slab[:,0].min()
            records.append((x_range, slab.mean(axis=0)))
        y += slice_thickness

    if len(records) < 5:
        raise RuntimeError(
            f"Neck detection failed: only {len(records)} valid slices found. "
            "Increase slice_frac or check mesh continuity."
        )

    # Minimum X-range corresponds to neck bottleneck
    neck_center = min(records, key=lambda r: r[0])[1]

    return hip_center, neck_center

# --------------------------------------------------
# Normalization utilities (relative)
# --------------------------------------------------

def bbox_diagonal(pts):
    return np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))

# --------------------------------------------------
# Visualization
# --------------------------------------------------

def visualize(pcd, hip, neck, save_png_path=None, img_size=2048):
    """
    Visualization with projection-consistent offset depiction.

    - Always shows interactive visualization
    - Optionally saves a high-quality PNG
    - Mesh rendered in uniform gray
    - All marker sizes relative to mesh scale
    - Camera looks along +Z, flipped so head (min Y) is at top
    """

    # -----------------------------
    # Relative scale parameters
    # -----------------------------
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    scale = max(extent)

    sphere_radius = 0.01 * scale

    # -----------------------------
    # Helper geometry creators
    # -----------------------------
    def make_sphere(center, color):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        s.translate(center)
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        return s

    def make_line(points, color):
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        ls.colors = o3d.utility.Vector3dVector([color])
        return ls

    # -----------------------------
    # Markers
    # -----------------------------
    hip_sphere = make_sphere(hip, [0, 1, 0])    # green
    neck_sphere = make_sphere(neck, [1, 0, 0])  # red

    y0, y1 = bbox.min_bound[1], bbox.max_bound[1]

    hip_y_line = make_line([
        [hip[0], y0, hip[2]],
        [hip[0], y1, hip[2]],
    ], [0, 1, 0])

    neck_y_line = make_line([
        [neck[0], y0, neck[2]],
        [neck[0], y1, neck[2]],
    ], [1, 0, 0])

    y_mid = 0.5 * (hip[1] + neck[1])
    offset_line = make_line([
        [hip[0], y_mid, hip[2]],
        [neck[0], y_mid, hip[2]],
    ], [1, 1, 0])

    # -----------------------------
    # Interactive visualization
    # -----------------------------
    o3d.visualization.draw_geometries([
        pcd,
        hip_sphere,
        neck_sphere,
        hip_y_line,
        neck_y_line,
        offset_line,
    ])

    if save_png_path is None:
        return

    # -----------------------------
    # Offscreen PNG rendering
    # -----------------------------
    import open3d.visualization.rendering as rendering

    renderer = rendering.OffscreenRenderer(img_size, img_size)
    scene = renderer.scene
    scene.set_background([0, 0, 0, 1])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.7, 0.7, 0.7, 1.0]

    scene.add_geometry("mesh", pcd, mat)
    scene.add_geometry("hip", hip_sphere, mat)
    scene.add_geometry("neck", neck_sphere, mat)
    scene.add_geometry("hip_y", hip_y_line, mat)
    scene.add_geometry("neck_y", neck_y_line, mat)
    scene.add_geometry("offset", offset_line, mat)

    center = bbox.get_center()
    cam_distance = 2.5 * scale
    eye = center + np.array([0, 0, cam_distance])
    up = np.array([0, -1, 0])  # flip Y so head is at top

    scene.camera.look_at(center, eye, up)

    scene.camera.set_projection(
        rendering.Camera.Projection.Ortho,
        -0.6 * scale, 0.6 * scale,
        -0.9 * scale, 0.9 * scale,
        0.1, 10 * scale
    )

    img = renderer.render_to_image()
    o3d.io.write_image(save_png_path, img)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", help="Input mesh or point cloud")
    ap.add_argument("--slice-frac", type=float, default=1/50,
                    help="Slice thickness as fraction of Y-range")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--save-png", type=str, default=None,
                    help="Path to save Z-axis camera PNG snapshot")
    args = ap.parse_args()

    pcd = load_as_pcd(args.mesh)
    pts = np.asarray(pcd.points)

    hip, neck = compute_neck_hip_relative(pcd, args.slice_frac)

    dx = neck[0] - hip[0]
    dx_norm = dx / bbox_diagonal(pts)

    print("Hip center :", hip)
    print("Neck center:", neck)
    print(f"Δx raw      : {dx:.6f}")
    print(f"Δx norm     : {dx_norm:.6f}")

    if not args.no_vis:
        visualize(pcd, hip, neck, save_png_path=args.save_png)
if __name__ == "__main__":
    main()


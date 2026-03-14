#!/usr/bin/env python3
"""
hip_aix_bend.py

Absolute estimation of horizontal offset between neck center and hip center
for a bending posture (Adam's Forward Bend Test).

Axis convention (standard):
- X-axis: from left shoulder to right shoulder
- Y-axis: high value towards upper body, low value towards foot
- Z-axis: high value towards hips, low value towards neck (spine axis)

Logic:
- Slices the mesh along the Z-axis (Spine).
- Hips are identified at high Z (towards hips).
- Neck is identified at low Z (towards neck) via bottleneck detection.
- Computes absolute Δx (Horizontal Offset).
"""

import argparse
import numpy as np
import open3d as o3d

# --------------------------------------------------
# Loading utilities
# --------------------------------------------------

def load_as_pcd(path):
    geom = o3d.io.read_triangle_mesh(path)
    if geom.has_triangles() and len(geom.vertices) > 0:
        # Use existing vertices directly for deterministic behavior
        # geom.compute_vertex_normals() # Optional, but good for visualization if using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = geom.vertices
        if geom.has_vertex_colors():
             pcd.colors = geom.vertex_colors
    else:
        pcd = o3d.io.read_point_cloud(path)
    
    if len(pcd.points) == 0:
         raise RuntimeError(f"Loaded mesh/pcd from {path} has no points.")
         
    return pcd

# --------------------------------------------------
# Core computation (Z-axis slicing)
# --------------------------------------------------

def compute_neck_hip_absolute(pcd, slice_frac=1/50):
    pts = np.asarray(pcd.points)

    if pts.shape[0] < 200:
        raise RuntimeError("Point cloud too sparse for reliable estimation")

    # ---- mesh statistics ----
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    x_min, y_min, z_min = bbox_min
    x_max, y_max, z_max = bbox_max

    X_range = x_max - x_min
    Y_range = y_max - y_min  # Depth/Height range
    Z_range = z_max - z_min  # Spine Length range

    if Z_range <= 0 or X_range <= 0:
        raise RuntimeError("Degenerate bounding box detected")

    N = pts.shape[0]

    # ---- Slicing along Z now ----
    slice_thickness = Z_range * slice_frac
    min_pts = max(int(0.001 * N), 40)

    # --------------------------------------------------
    # HIP CENTER (High Z -> towards hips)
    # Search top 20% of Z-range (high Z = hips)
    # --------------------------------------------------
    hip_center = None
    z = z_max
    z_stop = z_max - 0.20 * Z_range

    while z > z_stop:
        # Slice indices 0:X, 1:Y, 2:Z
        slab = pts[(pts[:,2] >= z - slice_thickness) & (pts[:,2] < z)]
        if slab.shape[0] >= min_pts:
            hip_center = slab.mean(axis=0)
            break
        z -= slice_thickness

    if hip_center is None:
        raise RuntimeError(
            "Hip center not found. Mesh may be incomplete near pelvis (high Z) or too sparse."
        )

    # --------------------------------------------------
    # NECK CENTER (Trend-based detection)
    # Solution 3: Directional + Trend constraints
    # Search for first stable narrow region followed by shoulder expansion.
    # --------------------------------------------------
    records = [] # (z_center, x_range, centroid)

    z = z_min
    while z < z_max:
        slab = pts[(pts[:,2] >= z) & (pts[:,2] < z + slice_thickness)]
        if slab.shape[0] >= min_pts:
            x_range = slab[:,0].max() - slab[:,0].min()
            z_center = z + 0.5 * slice_thickness
            records.append((z_center, x_range, slab.mean(axis=0)))
        z += slice_thickness

    if len(records) < 5:
        raise RuntimeError(
            f"Neck detection failed: only {len(records)} valid slices found. "
            "Increase slice_frac or check mesh continuity along Z-axis."
        )

    # Sort by Z (ascending, from Neck towards Hips)
    records.sort(key=lambda r: r[0])

    neck_center = None
    
    # Parameters for trend detection
    k_lookahead = 4      # Look at next 4 slices
    epsilon = 0.02 * X_range # Width must increase by at least 2% of total body width (defined above)
    
    # we only search within the first 60% of the valid slices (assuming neck is in upper/low-Z half)
    # to avoid falsely detecting waist indentation as neck if shoulders are missing.
    search_limit_idx = int(0.60 * len(records))
    
    found_idx = -1

    # Iterate to find "width take-off"
    for i in range(search_limit_idx - k_lookahead):
        curr_width = records[i][1]
        
        # Check if next k slices are consistently wider
        is_bottleneck = True
        for j in range(1, k_lookahead + 1):
            next_width = records[i+j][1]
            if next_width < curr_width + epsilon:
                is_bottleneck = False
                break
        
        if is_bottleneck:
            # Found the start of shoulder expansion
            found_idx = i
            break
    
    if found_idx != -1:
        # Robustness: Average this slice and its immediate neighbor
        candidates = [records[found_idx][2]]
        if found_idx + 1 < len(records):
            candidates.append(records[found_idx+1][2])
        neck_center = np.mean(candidates, axis=0)
    else:
        # Fallback: If no clear shoulder expansion found (e.g. armless/shoulderless crop),
        # revert to finding the narrowest slice in the detected upper region (Low Z).
        print("Warning: Trend-based neck detection failed (no clear shoulder expansion). "
              "Fallback to narrowest slice in top 40% of captured spine.")
        
        subset_limit = max(5, int(0.4 * len(records)))
        subset = records[:subset_limit]
        nearest_narrowest = min(subset, key=lambda r: r[1])
        neck_center = nearest_narrowest[2]

    return hip_center, neck_center

# --------------------------------------------------
# Visualization
# --------------------------------------------------

def visualize(pcd, hip, neck, save_png_path=None, img_size=2048):
    """
    Visualization for Bending Posture:
    - Z-axis is Spine (Horizontal-ish)
    - Y-axis is Ground Normal
    - Camera looks along Y (Top-down view of the back)
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

    # Lines should span the Z-length now? Or show the X-offset?
    # Usually we show the offset relative to the spine axis.
    # In original: Lines were Vertical (Y).
    # Here: Lines should be Longitudinal (Z).
    
    z0, z1 = bbox.min_bound[2], bbox.max_bound[2]

    # Line through Hip parallel to Z
    hip_z_line = make_line([
        [hip[0], hip[1], z0],
        [hip[0], hip[1], z1],
    ], [0, 1, 0])

    # Line through Neck parallel to Z
    neck_z_line = make_line([
        [neck[0], neck[1], z0],
        [neck[0], neck[1], z1],
    ], [1, 0, 0])

    # Offset line between them at mid-Z geometry?
    # Original: y_mid. Here: z_mid.
    z_mid = 0.5 * (hip[2] + neck[2])
    
    # We want to show the X-offset.
    # Connecting the two lines at z_mid
    offset_line = make_line([
        [hip[0], hip[1], z_mid],
        [neck[0], hip[1], z_mid], # Keep Y constant? Neck and Hip might have different Y (depth).
        # To show pure X-offset, we should project to same plane?
        # Original: [hip[0], y_mid, hip[2]], [neck[0], y_mid, hip[2]] -> Only X changed. Z constant.
        # Here: X changes. Y? Z constant.
    ], [1, 1, 0])
    
    # Better offset line: Connect the projected points
    offset_line = make_line([
        [hip[0], hip[1], z_mid],
        [neck[0], hip[1], z_mid] 
    ], [1, 1, 0])


    # -----------------------------
    # Interactive visualization (Open3D)
    # -----------------------------
    print("Close the Open3D window to proceed to 2D Top-View Plot...")
    o3d.visualization.draw_geometries([
        pcd,
        hip_sphere,
        neck_sphere,
        hip_z_line,
        neck_z_line,
        offset_line,
    ])

    # -----------------------------
    # Matplotlib 2D Visualization (Top View with Legends)
    # -----------------------------
    import matplotlib.pyplot as plt

    # Extract points
    pts = np.asarray(pcd.points)
    
    # "Top view" (X vs Z)
    plt.figure(figsize=(10, 8))
    
    # Main mesh points - use scatter
    plot_step = 1
    if pts.shape[0] > 10000:
        plot_step = max(1, pts.shape[0] // 10000)
        
    plt.scatter(pts[::plot_step, 0], pts[::plot_step, 2], s=1, c='gray', alpha=0.3, label='Mesh Points')
    
    # Plot Hip and Neck
    plt.scatter([hip[0]], [hip[2]], c='green', s=150, marker='o', edgecolors='black', label='Hip Center', zorder=10)
    plt.scatter([neck[0]], [neck[2]], c='red', s=150, marker='o', edgecolors='black', label='Neck Center', zorder=10)
    
    # Connect them
    plt.plot([hip[0], neck[0]], [hip[2], neck[2]], c='blue', linestyle='--', linewidth=2, label='Connection', zorder=5)
    
    plt.title(f"Top View (X-Z Projection)\nNeck and Hip Center Identification")
    plt.xlabel("X [mm or units]")
    plt.ylabel("Z (Spine Axis) [mm or units]")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Save if requested
    if save_png_path:
        plt.savefig(save_png_path, dpi=150)
        print(f"Saved Top-View PNG to: {save_png_path}")

    # Show interactive plot
    plt.tight_layout()
    plt.show()


def save_geometric_visualization(pcd, hip, neck, dx, out_path):
    """
    Save a 2D Top-View visualization focusing on the Geometric Offset (Delta X).
    """
    import matplotlib.pyplot as plt
    
    pts = np.asarray(pcd.points)
    
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plot_step = max(1, pts.shape[0] // 10000)
    plt.scatter(pts[::plot_step, 0], pts[::plot_step, 2], s=1, c='lightgray', alpha=0.5, label='Mesh Points')
    
    # Plot Hip and Neck
    plt.scatter([hip[0]], [hip[2]], c='green', s=150, marker='o', edgecolors='black', label='Hip Center', zorder=10)
    plt.scatter([neck[0]], [neck[2]], c='red', s=150, marker='o', edgecolors='black', label='Neck Center', zorder=10)
    
    # Draw reference lines for Delta X
    # Vertical line at Hip X
    plt.axvline(x=hip[0], color='green', linestyle=':', alpha=0.7, label='Hip X-ref')
    # Vertical line at Neck X
    plt.axvline(x=neck[0], color='red', linestyle=':', alpha=0.7, label='Neck X-ref')
    
    # Draw horizontal arrow showing Delta X
    z_mid = (hip[2] + neck[2]) / 2
    plt.annotate(
        '', xy=(neck[0], z_mid), xytext=(hip[0], z_mid),
        arrowprops=dict(arrowstyle='<->', color='blue', lw=2)
    )
    plt.text(
        (hip[0] + neck[0]) / 2, z_mid + 5,
        f"Δx = {dx:.2f} mm", ha='center', va='bottom', color='blue', fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    
    plt.title(f"Geometric Metric Visualization\nHorizontal Offset (Neck - Hip)")
    plt.xlabel("X (Left-Right) [mm]")
    plt.ylabel("Z (Spine Axis) [mm]")
    
    # Description
    desc = ("Red: Neck Center. Green: Hip Center.\n"
            "This plot calculates the horizontal displacement (Delta X) "
            "between the top and bottom of the functional spine.\n"
            "Large Delta X indicates lateral deviation/imbalance.")
    plt.figtext(0.5, 0.01, desc, wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.subplots_adjust(bottom=0.15)
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Geometric Visualization to: {out_path}")


def save_angular_visualization(pcd, hip, neck, angle_3d, angle_proj, out_path):
    """
    Save a visualziation focusing on Angular Metrics.
    Top-down view (XZ) for projected angle.
    """
    import matplotlib.pyplot as plt
    
    pts = np.asarray(pcd.points)
    
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plot_step = max(1, pts.shape[0] // 10000)
    plt.scatter(pts[::plot_step, 0], pts[::plot_step, 2], s=1, c='lightgray', alpha=0.5, label='Mesh Points')
    
    # Plot Hip and Neck
    plt.scatter([hip[0]], [hip[2]], c='green', s=150, marker='o', edgecolors='black', label='Hip Center', zorder=10)
    plt.scatter([neck[0]], [neck[2]], c='red', s=150, marker='o', edgecolors='black', label='Neck Center', zorder=10)
    
    # Draw Spine Axis (Z-axis) passing through Hip? 
    # Or just a vertical reference line.
    # We compare the Neck-Hip vector to the Z-axis.
    # So we can draw a vertical line from Hip.
    
    # Vector line
    plt.plot([hip[0], neck[0]], [hip[2], neck[2]], c='blue', linewidth=2, label='Neck-Hip Vector')
    
    # Reference Z-line from Hip
    z_ref_end = neck[2] # Extend to neck depth
    plt.plot([hip[0], hip[0]], [hip[2], z_ref_end], c='black', linestyle='--', linewidth=1.5, label='Reference Z-axis')
    
    # Annotate Angle
    # This is a schematic representation
    plt.text(
        hip[0], (hip[2] + neck[2])/2,
        f"Projected Angle: {angle_proj:.1f}°",
        color='blue', fontweight='bold', ha='right', va='center', rotation=90,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    
    # Add text box for 3D angle
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = f"3D Angle (w.r.t Z-axis): {angle_3d:.2f}°"
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
             
    # Description
    desc = ("Red: Neck Center. Green: Hip Center. Blue Line: Neck-Hip Vector.\n"
            "Calculates the angle of the full spine axis relative to the vertical (Z-axis).\n"
            "Higher angle indicates global tilt or deviation.")
    plt.figtext(0.5, 0.01, desc, wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.subplots_adjust(bottom=0.15)

    plt.title(f"Angular Metric Visualization\nSpine Axis Tilt")
    plt.xlabel("X (Left-Right) [mm]")
    plt.ylabel("Z (Spine Axis) [mm]")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Angular Visualization to: {out_path}")



# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Absolute delta_x calculator for bending posture")
    ap.add_argument("--mesh", required=True, help="Input mesh or point cloud")
    ap.add_argument("--slice-frac", type=float, default=1/50,
                    help="Slice thickness as fraction of Z-range")
    ap.add_argument("--no-vis", action="store_true")
    args = ap.parse_args()

    # Pass '1.0' or just rely on default argument signature change
    # Note: load_as_pcd signature changed, no longer takes ratio arg, but we should check call site.
    pcd = load_as_pcd(args.mesh)
    
    hip, neck = compute_neck_hip_absolute(pcd, args.slice_frac)

    # Absolute difference
    dx = neck[0] - hip[0]
    
    # Vector from Neck to Hip (Head -> Back, roughly +Z)
    vec = hip - neck
    
    # 1. 3D Angle with Z-axis
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 1e-6:
        cos_theta = vec[2] / vec_norm
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg_3d = np.degrees(np.arccos(cos_theta))
    else:
        angle_deg_3d = 0.0

    # 2. Projected Angle on Ground Plane (XZ plane, perp to Y)
    # Project vector onto XZ plane simply by setting Y component to 0
    vec_xz = np.array([vec[0], 0, vec[2]])
    vec_xz_norm = np.linalg.norm(vec_xz)
    
    # Angle between Projected Vector and Z-axis [0, 0, 1]
    # In XZ plane, Z-axis is [0, 1] effectively.
    if vec_xz_norm > 1e-6:
        cos_theta_xz = vec_xz[2] / vec_xz_norm
        cos_theta_xz = np.clip(cos_theta_xz, -1.0, 1.0)
        angle_deg_xz = np.degrees(np.arccos(cos_theta_xz))
    else:
        angle_deg_xz = 0.0

    # Note: User requested "angle between projected line and y axis". 
    # Since Y is normal to the projection plane, that angle is mathematically 90 degrees.
    # We assume the user meant Z-axis (Spine Axis) as the reference in the projected plane.

    # Output to console
    print(f"Hip center : {hip}")
    print(f"Neck center: {neck}")
    print(f"Δx (signed): {dx:.6f}")
    print(f"|Δx| (abs) : {abs(dx):.6f}")
    print(f"Angle 3D w/ Z     : {angle_deg_3d:.2f} degrees")
    print(f"Angle XZ (proj) w/ Z: {angle_deg_xz:.2f} degrees")

    # Save stats to JSON
    import json
    import os
    
    # Structured output as requested
    stats = [
        {
            "Quantity": "Hip Center (XYZ)",
            "Definition": "Centroid of the mesh slice at the detected hip region (Top 20% of Z-range)",
            "Value": hip.tolist()
        },
        {
            "Quantity": "Neck Center (XYZ)",
            "Definition": "Centroid of the mesh slice at the detected neck bottleneck (Minimum X-width region)",
            "Value": neck.tolist()
        },
        {
            "Quantity": "Delta X",
            "Definition": "Signed horizontal offset along X-axis (Neck_X - Hip_X)",
            "Value": dx
        },
        {
            "Quantity": "Absolute Delta X",
            "Definition": "Magnitude of the horizontal offset |Neck_X - Hip_X|",
            "Value": abs(dx)
        },
        {
            "Quantity": "3D Asymmetry Angle",
            "Definition": "Angle (degrees) between the 3D vector (Neck->Hip) and the Spine Axis (Z-axis)",
            "Value": angle_deg_3d
        },
        {
            "Quantity": "Projected Asymmetry Angle",
            "Definition": "Angle (degrees) between the vector (Neck->Hip) projected onto the Coronal (XZ) plane and the Spine Axis (Z-axis)",
            "Value": angle_deg_xz
        }
    ]
    
    # Construct output path: same folder as mesh, suffix _stats.json
    mesh_dir = os.path.dirname(os.path.abspath(args.mesh))
    mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
    json_path = os.path.join(mesh_dir, f"{mesh_name}_hip_stats.json")
    png_path = os.path.join(mesh_dir, f"{mesh_name}_neck_hip_view.png")
    
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Stats saved to: {json_path}")

    save_geometric_visualization(
        pcd, hip, neck, dx,
        out_path=os.path.join(mesh_dir, f"{mesh_name}_hip_geometric_vis.png")
    )

    save_angular_visualization(
        pcd, hip, neck, angle_deg_3d, angle_deg_xz,
        out_path=os.path.join(mesh_dir, f"{mesh_name}_hip_angular_vis.png")
    )

    if not args.no_vis:
        visualize(pcd, hip, neck, save_png_path=png_path)

if __name__ == "__main__":
    main()

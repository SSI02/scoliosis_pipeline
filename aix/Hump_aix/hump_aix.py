import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

# --------------------------------------------------
# Plane utilities
# --------------------------------------------------

def load_plane(json_path):
    with open(json_path, "r") as f:
        p = json.load(f)
    
    n = None
    d = None

    # Handle new list-based format from spine_aix.py
    if isinstance(p, list):
        for item in p:
            q = item.get("Quantity", "")
            val = item.get("Quantity Estimated Value", None)
            
            if q == "Plane Normal" and val is not None:
                n = np.array(val, float)
            elif q == "Plane Constant (d)" and val is not None:
                d = float(val)
        
        if n is None:
            # Fallback for older list formats or if keys are missing
            print(f"Warning: Could not find 'Plane Normal' in {json_path}. Using default.")
            n = np.array([0., 0., 1.])
        if d is None:
             d = 0.0

    # Handle legacy dictionary formats
    elif isinstance(p, dict):
        if "plane_equation" in p:
            pe = p["plane_equation"]
            if "normal" in pe:
                n = np.array(pe["normal"], float)
            elif "equation" in pe:
                 # Fallback if normal not explicitly in plane_equation
                 n = np.array([pe.get("a", 0), pe.get("b", 0), pe.get("c", 1)], float)
            else:
                 # Try top level normal_vector if present
                 if "normal_vector" in p:
                     n = np.array(p["normal_vector"], float)
                 else:
                     # Last resort
                     n = np.array(p.get("normal", [0,0,1]), float)
            
            d = float(pe.get("d", 0.0))
        else:
            # Flat format
            n = np.array(p.get("normal", p.get("normal_vector", [0,0,1])), float)
            d = float(p.get("d", 0.0))
    
    else:
        # Unknown format
        print(f"Error: Unknown JSON format in {json_path}")
        n = np.array([0., 0., 1.]) 
        d = 0.0

    if n is not None:
        n /= np.linalg.norm(n)
        
    return n, d


def fit_plane_pca(pts):
    if len(pts) < 3:
        return np.array([0., 0., 1.]), np.mean(pts, axis=0) if len(pts)>0 else np.zeros(3)
    c = pts.mean(axis=0)
    U = pts - c
    _, _, Vt = np.linalg.svd(U, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    return n, c


def estimate_back_normal(pts, n_mid):
    """
    Returns inward normal to the back.
    Convention: +Y points INTO the body.
    """

    # PCA normal
    c = pts.mean(axis=0)
    U = pts - c
    _, _, Vt = np.linalg.svd(U, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n)

    # Use mid-sagittal plane to decide inward direction
    # Project points slightly along ±n and see which goes "inside"
    test = pts + 0.01 * n
    side = test @ n_mid

    # Inside of body lies closer to mid-sagittal plane
    if np.mean(np.abs(side)) > np.mean(np.abs(pts @ n_mid)):
        n = -n

    return n


def angle_between(n1, n2):
    return np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1, 1)))


def save_side_view_angle_plot(n_Y, hump_n, Z_roi, Y_roi, angle_deg, out_path, hump_pts=None):
    """
    Side view (Z-Y plane) showing angle between back plane and hump plane.
    Z: neck (low) -> hips (high)
    Y: foot (low) -> upper body (high)
    """

    # Project normals to Z-Y plane (X=0 plane)
    # n_Y and hump_n are 3D normals, we project to Z-Y plane by taking Y and Z components
    v_back = np.array([n_Y[2], n_Y[1]])  # [Z, Y] components
    v_hump = np.array([hump_n[2], hump_n[1]])  # [Z, Y] components

    # Normalize
    v_back /= np.linalg.norm(v_back)
    v_hump /= np.linalg.norm(v_hump)

    # Axis limits from data
    # Z: low = neck, high = hips (use as-is)
    zmin, zmax = np.percentile(Z_roi, [5, 95])
    # Y: low = foot, high = upper body (use as-is)
    ymin, ymax = np.percentile(Y_roi, [5, 95])

    z0 = 0.5 * (zmin + zmax)
    y0 = 0.5 * (ymin + ymax)
    L = 0.4 * (zmax - zmin)

    plt.figure(figsize=(6, 6))

    # Plot actual points if provided (Validation!)
    if hump_pts is not None and len(hump_pts) > 0:
        # Z is index 2, Y is index 1
        plt.scatter(hump_pts[:, 2], hump_pts[:, 1], s=5, c='orange', alpha=0.5, label='Hump Points', zorder=1)

    # Back plane line
    plt.plot(
        [z0 - L * v_back[0], z0 + L * v_back[0]],
        [y0 - L * v_back[1], y0 + L * v_back[1]],
        color="green", linewidth=3, label="Back plane ref"
    )

    # Hump plane line
    # Center the hump line at the centroid of hump points if possible, else plot center
    if hump_pts is not None and len(hump_pts) > 0:
        center_z = np.mean(hump_pts[:, 2])
        center_y = np.mean(hump_pts[:, 1])
        # Use a slightly smaller L for the fitted line to keep it local
        plt.plot(
            [center_z - L * v_hump[0], center_z + L * v_hump[0]],
            [center_y - L * v_hump[1], center_y + L * v_hump[1]],
            color="red", linewidth=2, linestyle="--", label="Fitted Hump Plane"
        )
    else:
        plt.plot(
            [z0 - L * v_hump[0], z0 + L * v_hump[0]],
            [y0 - L * v_hump[1], y0 + L * v_hump[1]],
            color="red", linewidth=2, linestyle="--", label="Hump Plane Normal"
        )

    # Angle annotation with more detail
    plt.text(
        z0, ymax,
        f"Angle: {angle_deg:.1f}°",
        color="black",
        fontsize=12,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )
    
    # Detailed text box
    details = f"Back Normal (proj): (0, 1)\nHump Normal (proj): ({v_hump[1]:.2f}, {v_hump[0]:.2f})"
    plt.text(0.05, 0.05, details, transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel("Spine axis (Z)  [Neck ← Hips]")
    plt.ylabel("Vertical (Y)  [Foot ↑ Upper Body]")
    plt.title(f"Rib Hump Angle Verification\nSide View (Z-Y Projection)")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.xlim(zmin, zmax)
    # y limits might need expansion to show lines
    plt.ylim(ymin - 0.2*(ymax-ymin), ymax + 0.2*(ymax-ymin))

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()



def create_plane_mesh(n, point, size=0.4, color=[0,1,0]):
    """
    Create a rectangular Open3D plane mesh.
    n     : plane normal (unit)
    point : a point on plane
    """
    # Create orthogonal basis
    n = n / np.linalg.norm(n)
    v1 = np.array([1.,0,0])
    if abs(v1 @ n) > 0.9:
        v1 = np.array([0,1.,0])
    v1 -= (v1 @ n) * n
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)

    corners = [
        point + size*( v1 + v2),
        point + size*( v1 - v2),
        point + size*(-v1 - v2),
        point + size*(-v1 + v2),
    ]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_arrow(origin, direction, length=0.15, color=[0, 0, 1]):
    """
    Create an Open3D arrow showing a vector direction.
    """
    direction = direction / np.linalg.norm(direction)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003,
        cone_radius=0.006,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()

    # Rotate arrow (default points +Z)
    z_axis = np.array([0, 0, 1])
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.cross(z_axis, direction)
    )
    arrow.rotate(R, center=np.zeros(3))

    # Move to origin
    arrow.translate(origin)

    return arrow


def text3d(text, pos):
    # Check if create_text is available (requires newer Open3D)
    if hasattr(o3d.geometry.TriangleMesh, 'create_text'):
        try:
            t = o3d.geometry.TriangleMesh.create_text(text, depth=0.01)
            t.translate(pos)
            t.paint_uniform_color([1,1,1])
            return t
        except Exception:
            return o3d.geometry.TriangleMesh()
    else:
        # Return empty mesh if not supported, avoid cluttering with spheres/warnings
        return o3d.geometry.TriangleMesh()


def save_ridge_comparison(z_vals, ridge_L, ridge_R, out_path):
    """
    Visualize Left vs Right ridge lines with shaded area (Volume Asymmetry).
    """
    valid = ~np.isnan(ridge_L) & ~np.isnan(ridge_R)
    if np.sum(valid) < 5:
        return

    z = z_vals[valid]
    L = ridge_L[valid]
    R = ridge_R[valid]

    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(z, L, color='blue', label='Left Ridge', linewidth=2)
    plt.plot(z, R, color='red', label='Right Ridge', linewidth=2, linestyle='--')
    
    # Fill areas
    # Where L > R (Left Dominant)
    plt.fill_between(z, L, R, where=(L > R), interpolate=True, color='blue', alpha=0.2, label='Left Prominence')
    # Where R > L (Right Dominant)
    plt.fill_between(z, L, R, where=(R > L), interpolate=True, color='red', alpha=0.2, label='Right Prominence')
    
    # Calculate simple area stats for annotation
    area_L = np.trapz(np.maximum(L - R, 0), z)
    area_R = np.trapz(np.maximum(R - L, 0), z)
    
    plt.text(0.05, 0.95, f"Left Dominance Area: {area_L:.2f}\nRight Dominance Area: {area_R:.2f}", 
             transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel("Spine Axis (Z) [Neck -> Hips]")
    plt.ylabel("Posterior Height (Y)")
    plt.title("Ridge Line Comparison & Volume Asymmetry")
    
    # Description
    desc = ("Red line: Right dorsal profile. Blue line: Left dorsal profile.\n"
            "Shaded area represents the volume difference (asymmetry).\n"
            "Higher area value indicates greater prominence on that side.")
    plt.figtext(0.5, 0.01, desc, wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.subplots_adjust(bottom=0.15) # Make room for text

    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Ridge Comparison to: {out_path}")


def save_symmetry_heatmap(X, Z, Y, out_path):
    """
    Generate a heatmap of depth difference (Asymmetry) across the back.
    We interpolate the point cloud to a grid and compute (Z_right_mirrored - Z_left).
    Actually, simpler: Just bin statistics of Y on (X, Z) grid, then flip and subtract.
    As standard convention (pipeline): X is Left(-)-to-Right(+)
    """
    # 1. Bin data to grid
    target_bins = len(X) // 10
    nx = nz = int(np.sqrt(target_bins))
    nx = np.clip(nx, 100, 300)
    
    # Define symmetric grid centered on 0
    xmax = np.percentile(np.abs(X), 95)
    zmin, zmax = Z.min(), Z.max()
    
    xb = np.linspace(-xmax, xmax, nx)
    zb = np.linspace(zmin, zmax, nz)
    
    # Grid for Right side (positive X)
    xb_right = np.linspace(0, xmax, nx//2)
    
    # Bin separate sides? No, simpler:
    # Bin everything to full grid
    ret = binned_statistic_2d(X, Z, Y, statistic='mean', bins=[xb, zb])
    grid_Y = ret.statistic.T # shape (nz-1, nx-1)
    
    # Split into Left and Right halves
    # Center index of X
    cx = (nx-1) // 2
    
    # Left half (indices 0 to cx), Right half (indices cx to end)
    # We need to mirror Left onto Right to compare.
    # Left X are negative. 
    # Left side: grid_Y[:, :cx] (columns 0..cx-1) -> corresponds to X < 0
    # Right side: grid_Y[:, cx:] (columns cx..end) -> corresponds to X > 0
    
    # We need strictly symmetric comparison. 
    # Let's resample at specific symmetric X points.
    
    from scipy.interpolate import griddata
    
    # Create target symmetric grid points
    num_pts = 100
    x_sym = np.linspace(0, xmax * 0.9, num_pts)
    z_sym = np.linspace(zmin, zmax, num_pts)
    XX, ZZ = np.meshgrid(x_sym, z_sym)
    
    # Interpolate Right side (X > 0)
    mask_R = X > 0
    grid_R = griddata((X[mask_R], Z[mask_R]), Y[mask_R], (XX, ZZ), method='linear')
    
    # Interpolate Left side (X < 0) - but treat X as positive (mirrored) for query
    mask_L = X < 0
    grid_L = griddata((-X[mask_L], Z[mask_L]), Y[mask_L], (XX, ZZ), method='linear')
    
    # Difference (Right - Left)? Or Left - Right?
    # Usually "Hump" is positive.
    # Deviation = |L - R| or signed?
    # Signed allows seeing which side is higher.
    # Convention: Positive = Right side is higher?
    # Let's do (Right - Left). Positive -> Right Hump. Negative -> Left Hump.
    
    diff_map = grid_R - grid_L
    
    plt.figure(figsize=(8, 7)) # Increased height for text
    # Plot extent: X (distance from spine), Z (spine axis)
    extent = [0, xmax*0.9, zmin, zmax]
    
    # Use diverging colormap centered at 0
    vmax = np.nanpercentile(np.abs(diff_map), 99)
    plt.imshow(diff_map, origin='lower', extent=extent, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(label="Height Diff (Right - Left) [mm]")
    
    plt.xlabel("Lateral Distance from Spine (X) [mm]")
    plt.ylabel("Spine Axis (Z) [mm]")
    plt.title("Back Asymmetry Heatmap\n(Red=Right Higher, Blue=Left Higher)")
    
    # Description
    desc = ("Heatmap showing height difference between mirrored points on Left/Right back.\n"
            "Red (+): Right side is higher than Left.\n"
            "Blue (-): Left side is higher than Right.\n"
            "Intensity indicates magnitude of asymmetry.")
    plt.figtext(0.5, 0.01, desc, wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.subplots_adjust(bottom=0.18)

    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Symmetry Heatmap to: {out_path}")


def save_plane_fit_debug(points, n_plane, n_ref, out_path):
    """
    Visualize points used for plane fitting in 2D to verify 'hump' selection.
    Projects points onto the fitted plane normal direction? 
    Or just Top View (X-Z) or Side View (Y-Z).
    Side view (Y-Z) is best to show the "slope" of the hump.
    """
    if points is None or len(points) == 0:
        return

    # Project to Y-Z (Side View)
    y = points[:, 1]
    z = points[:, 2]
    
    # Create plane line for visualization in Y-Z
    # Plane equation: nx*x + ny*y + nz*z + d = 0
    # Centroid
    c = np.mean(points, axis=0)
    d = -np.dot(n_plane, c)
    
    # In Y-Z projection (assuming X is integrated/ignored or taken at centroid X)
    # ny*y + nz*z + (nx*cx + d) = 0
    # y = -(nz*z + nx*cx + d) / ny
    
    z_line = np.linspace(z.min(), z.max(), 100)
    # Solve for y
    if abs(n_plane[1]) > 1e-3:
        y_line = -(n_plane[2]*z_line + n_plane[0]*c[0] + d) / n_plane[1]
    else:
        y_line = np.full_like(z_line, np.nan)

    plt.figure(figsize=(8, 7))
    plt.scatter(z, y, s=10, c='orange', alpha=0.6, label='Hump Points Selected')
    plt.plot(z_line, y_line, color='red', linewidth=2, label='Fitted Plane (Profile)')
    
    # Reference (vertical/back plane direction?)
    # Just show data
    
    plt.xlabel("Spine Axis (Z)")
    plt.ylabel("Height (Y)")
    plt.title("Hump Plane Fit Debug (Side Profile)")
    
    desc = ("Visualizes the side profile of the selected 'Hump' points (Orange).\n"
            "The Red Line shows the plane fitted to calculate Hump Angle.\n"
            "Steeper slope = Larger Hump Angle.")
    plt.figtext(0.5, 0.01, desc, wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.subplots_adjust(bottom=0.15)
    
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Plane Fit Debug to: {out_path}")


def save_hump_selection_vis(all_pts, sel_pts, out_path):
    """
    Top-down view showing exactly where the 'Hump' points were selected on the back.
    """
    if all_pts is None or len(all_pts) == 0:
        return
        
    plt.figure(figsize=(8, 9))
    
    # Plot all ROI points
    # Subsample for speed/clarity
    step = max(1, len(all_pts) // 5000)
    plt.scatter(all_pts[::step, 0], all_pts[::step, 2], s=1, c='lightgray', label='Back ROI')
    
    # Plot Selected Hump Points
    if sel_pts is not None and len(sel_pts) > 0:
        plt.scatter(sel_pts[:, 0], sel_pts[:, 2], s=5, c='red', alpha=0.6, label='Selected Hump Area')
        
        # Draw bounding circle/box?
        # Just centroid
        c = np.mean(sel_pts, axis=0)
        plt.scatter([c[0]], [c[2]], marker='x', c='black', s=100, label='Hump Centroid')
        plt.text(c[0], c[2], "  HUMP", color='black', fontsize=10, fontweight='bold')

    plt.xlabel("Lateral Position (X) [mm]\n<-- Left   Right -->")
    plt.ylabel("Spine Axis (Z) [mm]\nNeck (Lower Z) -> Hips (Higher Z)") # Check Z direction logic again?
    # Logic: Z axis usually neck low, hips high or vice versa. 
    # Pipeline convention: "Z-axis: high value towards hips, low value towards neck"
    
    plt.title("Hump Selection Visualization (Top View)")
    
    desc = ("Top-down view of the back showing the region selected as the 'Rib Hump' (Red).\n"
            "This region is automatically detected as the highest prominence.\n"
            "Used to calculate Hump Angle and Plane.")
    plt.figtext(0.5, 0.02, desc, wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.subplots_adjust(bottom=0.15)
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Hump Selection Top View to: {out_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(mesh_path, mid_json, out_prefix="rib", save_vis=False):
    # ---- Load ----
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pts = np.asarray(mesh.vertices)

    # ---- 1. Standard Coordinate System Setup ----
    # Convention (mesh is already aligned by pipeline):
    # - X-axis: from left shoulder to right shoulder (points[:, 0])
    # - Y-axis: high value towards upper body, low value towards foot (points[:, 1])
    # - Z-axis: high value towards hips, low value towards neck (points[:, 2])
    
    # Use mesh coordinates directly (assuming pipeline has already aligned them)
    X = pts[:, 0]  # Left-to-right (left shoulder < right shoulder)
    Y = pts[:, 1]  # Vertical: high = upper body/head, low = foot
    Z = pts[:, 2]  # Spine axis: high = hips, low = neck
    
    # Load mid-sagittal plane for reference (used for some calculations)
    n_mid, d_mid = load_plane(mid_json)
    
    # Define axis vectors for projections (if needed)
    n_X = np.array([1.0, 0.0, 0.0])  # X-axis unit vector
    n_Y = np.array([0.0, 1.0, 0.0])  # Y-axis unit vector  
    n_Z = np.array([0.0, 0.0, 1.0])  # Z-axis unit vector

    # ---- 3. Thoracic ROI ----
    # Select central region along the spine axis (Z) to isolate the thoracic area.
    z0, z1 = np.percentile(Z, [30, 75])
    
    roi = (Z > z0) & (Z < z1)
    
    if np.sum(roi) < 100:
        roi = np.ones(len(Z), dtype=bool) # fallback
        
    X_roi, Y_roi, Z_roi = X[roi], Y[roi], Z[roi]
    pts_roi = pts[roi]

    # ---- 4. Grid Generation (Vectorized) ----
    target_bins = len(X_roi) // 10
    nx = nz = int(np.sqrt(target_bins))
    nx = np.clip(nx, 150, 500)
    
    
    xmin, xmax = X_roi.min(), X_roi.max()
    zmin, zmax = Z_roi.min(), Z_roi.max()
    
    margin = 0.05
    xmin -= margin; xmax += margin
    zmin -= margin; zmax += margin

    xb = np.linspace(xmin, xmax, nx+1)
    zb = np.linspace(zmin, zmax, nz+1)

    # Depth map: 90th percentile Y per bin (robust peak)
    depth_res = binned_statistic_2d(X_roi, Z_roi, Y_roi, 
                                    statistic=lambda v: np.percentile(v, 90), 
                                    bins=[xb, zb])
    
    depth = depth_res.statistic.T 
    
    # Fill gaps for smoothing
    depth_filled = depth.copy()
    mask_nan = np.isnan(depth_filled)
    min_val = np.nanmin(depth) if not np.all(mask_nan) else 0
    depth_filled[mask_nan] = min_val
    
    depth_smooth = gaussian_filter(depth_filled, sigma=2)

    # ---- Save depth map (Contour Version) ----
    plt.figure(figsize=(7,6))
    
    # Contour fill
    levels = np.linspace(np.nanmin(depth_smooth), np.nanmax(depth_smooth), 20)
    # Filter nan for contouring
    depth_clean = np.nan_to_num(depth_smooth, nan=np.nanmin(depth_smooth))
    
    # Use imshow extent for contours
    # X_roi (cols) corresponds to xb, Z_roi (rows) to zb
    # depth_smooth is (nz, nx). 
    # meshgrid for contour
    xc = 0.5 * (xb[:-1] + xb[1:])
    zc = 0.5 * (zb[:-1] + zb[1:])
    XC, ZC = np.meshgrid(xc, zc)
    
    cf = plt.contourf(XC, ZC, depth_smooth, levels=levels, cmap="viridis", extend='both')
    cbar = plt.colorbar(cf, label="Depth (Y: Posterior Height) [mm]")
    
    # Add contour lines
    cl = plt.contour(XC, ZC, depth_smooth, levels=levels[::2], colors='k', linewidths=0.5, alpha=0.5)
    plt.clabel(cl, inline=True, fontsize=8, fmt='%1.0f')
    
    # Mark max point
    max_idx = np.unravel_index(np.argmax(np.nan_to_num(depth_smooth, nan=-np.inf)), depth_smooth.shape)
    z_max_val = zc[max_idx[0]]
    x_max_val = xc[max_idx[1]]
    plt.scatter([x_max_val], [z_max_val], c='red', marker='x', s=100, label='Max Hump Height', zorder=10)
    
    plt.xlabel("Lateral Position (X) [mm]")
    plt.ylabel("Spine Axis (Z) [mm]")
    plt.title("Back Surface Topography (Contour Map)")
    plt.legend(loc='lower right')
    plt.savefig(f"{out_prefix}_depth_map.png", dpi=200)
    plt.close()

    # ---- 5. Volume Asymmetry ----
    dx = (xmax - xmin) / nx
    dz = (zmax - zmin) / nz

    x_centers = xc
    # In standard convention: X-axis from left shoulder to right shoulder
    # Left side: X < 0, Right side: X > 0
    left_mask_cols = x_centers < 0
    right_mask_cols = x_centers > 0
    
    # Volume calculation: Sum of positive depth (bulge) relative to reference plane.
    # Shifting not strictly necessary if Y=0 is the plane, but clipping negatives ensures we measure "humps".
    
    depth_vol = np.clip(depth_smooth, 0, None)
    
    V_L = np.sum(depth_vol[:, left_mask_cols]) * dx * dz
    V_R = np.sum(depth_vol[:, right_mask_cols]) * dx * dz
    
    volume_asym = abs(V_L - V_R) / (V_L + V_R + 1e-8)
    
    print(f"Volume asymmetry index: {volume_asym:.4f}")

    # ---- 6. Ridge Lines ----
    ridge_L = []
    ridge_R = []
    z_vals_plot = []
    
    for j in range(nz):
        row = depth_smooth[j, :]
        
        if np.any(left_mask_cols):
             ridge_L.append(np.max(row[left_mask_cols]))
        else:
             ridge_L.append(np.nan)
             
        if np.any(right_mask_cols):
             ridge_R.append(np.max(row[right_mask_cols]))
        else:
             ridge_R.append(np.nan)
             
        z_vals_plot.append(0.5 * (zb[j] + zb[j+1]))

    ridge_L = np.array(ridge_L)
    ridge_R = np.array(ridge_R)
    z_vals_plot = np.array(z_vals_plot)

    # (Ridge plot code remains similar or can be removed if save_ridge_comparison covers it. 
    # I'll keep the original simple one or replace it? User said "visualisations... utilize them to max potential".
    # The new save_ridge_comparison is better. I will comment out or remove the old simple plot to avoid clutter/confusion?
    # Or keep it as "simple raw data". I'll keep it for now but improve labels.)
    
    plt.figure(figsize=(6,4))
    plt.plot(z_vals_plot, ridge_L, label="Left hump", color="blue")
    plt.plot(z_vals_plot, ridge_R, label="Right hump", color="red")
    plt.xlabel("Spine axis (Z) [mm]")
    plt.ylabel("Posterior Elevation (Y) [mm]")
    plt.title("Rib Hump Ridge Lines (Raw)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_ridge_lines.png", dpi=200)
    plt.close()

    # ... (Hump processing code) ...




    # --------------------------------------------------
    # Identify left & right rib hump crest points
    # --------------------------------------------------

    # Find Z slice with maximum asymmetry
    valid = ~np.isnan(ridge_L) & ~np.isnan(ridge_R)
    
    if np.sum(valid) == 0:
        # No valid ridge data - use fallback: center of ROI
        print("Warning: No valid ridge data found. Using ROI center as fallback.")
        z_star = np.median(Z_roi)
        
        # Find left and right points from entire ROI
        # In standard convention: X-axis from left shoulder to right shoulder
        # Left side: X < 0, Right side: X > 0
        left_mask_roi = X_roi < 0
        right_mask_roi = X_roi > 0
        
        if np.sum(left_mask_roi) > 0:
            p_left = pts_roi[left_mask_roi][np.argmin(Y_roi[left_mask_roi])]
        else:
            p_left = pts_roi[0]
        
        if np.sum(right_mask_roi) > 0:
            p_right = pts_roi[right_mask_roi][np.argmin(Y_roi[right_mask_roi])]
        else:
            p_right = pts_roi[-1]
    else:
        idx = np.argmax(np.abs(ridge_L[valid] - ridge_R[valid]))
        z_star = z_vals_plot[valid][idx]

        # Find points near this Z slice
        dz_slice = 0.5 * (zb[1] - zb[0])
        slice_mask = np.abs(Z_roi - z_star) < dz_slice

        slice_pts = pts_roi[slice_mask]
        slice_X = X_roi[slice_mask]
        slice_Y = Y_roi[slice_mask]

        # Left hump = max posterior on left
        # In standard convention: X-axis from left shoulder to right shoulder
        # Left side: X < 0, Right side: X > 0
        left_mask = slice_X < 0
        right_mask = slice_X > 0

        if np.sum(left_mask) == 0:
            print("Warning: No left-side points found in slice. Using fallback.")
            left_mask_roi = X_roi < 0
            if np.sum(left_mask_roi) > 0:
                p_left = pts_roi[left_mask_roi][np.argmin(Y_roi[left_mask_roi])]
            else:
                p_left = pts_roi[0]
        else:
            p_left = slice_pts[left_mask][np.argmin(slice_Y[left_mask])]
        
        if np.sum(right_mask) == 0:
            print("Warning: No right-side points found in slice. Using fallback.")
            right_mask_roi = X_roi > 0
            if np.sum(right_mask_roi) > 0:
                p_right = pts_roi[right_mask_roi][np.argmin(Y_roi[right_mask_roi])]
            else:
                p_right = pts_roi[-1]
        else:
            p_right = slice_pts[right_mask][np.argmin(slice_Y[right_mask])]

    # Rib-hump line direction
    # Force Left -> Right direction (p_right - p_left)
    rib_line = p_right - p_left
    rib_line_norm = np.linalg.norm(rib_line)
    if rib_line_norm < 1e-6:
        print("Warning: Left and right points are too close. Using X-axis as fallback.")
        rib_line = n_X
    else:
        rib_line /= rib_line_norm


    # --------------------------------------------------
    # 6.5 Pointwise rib-hump asymmetry statistics
    # --------------------------------------------------

    # ---- 6.5 Pointwise rib-hump asymmetry statistics ----

    # Ensure same valid mask
    valid = ~np.isnan(ridge_L) & ~np.isnan(ridge_R)

    # Prepare definitions for statistics
    definitions = {
        "volume_asymmetry": "Global asymmetry index calculated from the difference in depth volume between the left and right sides of the back.",
        "mean_diff": "The average difference in height between the left and right ridge lines along the spine.",
        "std_diff": "The standard deviation of the difference between the left and right ridge lines.",
        "rms_diff": "Root Mean Square (RMS) of the difference between the left and right ridge lines, indicating overall magnitude of asymmetry.",
        "mean_abs_diff": "The average of the absolute differences between the left and right ridge lines.",
        "max_abs_diff": "The maximum absolute height difference observed between the left and right humps.",
        "num_slices": "The number of vertical (Z-axis) slices where valid ridge data was available for calculation.",
        "rib_hump_angle": "The local tilt angle of the prominent rib hump plane relative to the general back plane."
    }

    stats_list = []
    
    # Volume asymmetry (calculated earlier)
    stats_list.append({
        "Quantity": "volume_asymmetry",
        "Definition": definitions["volume_asymmetry"],
        "Quantity Estimated Value": float(volume_asym)
    })

    if np.sum(valid) > 5:
        diff = ridge_L[valid] - ridge_R[valid]

        calc_stats = {
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "rms_diff": float(np.sqrt(np.mean(diff**2))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "num_slices": int(np.sum(valid))
        }

        print("\nRib Hump Pointwise Statistics:")
        for k, v in calc_stats.items():
            print(f"  {k}: {v:.4f}")
            stats_list.append({
                "Quantity": k,
                "Definition": definitions.get(k, "Statistical metric"),
                "Quantity Estimated Value": v
            })

        # Optional visualization
        plt.figure(figsize=(8, 6)) # Increased size for detail
        plt.plot(z_vals_plot[valid], diff, color="purple", linewidth=2, label="Difference (L - R)")
        plt.axhline(0, color="black", linestyle="--", alpha=0.5, label="Symmetry Line")
        
        # Add stats to plot
        stat_text = (f"Max Abs Diff: {calc_stats['max_abs_diff']:.2f}\n"
                     f"RMS Diff: {calc_stats['rms_diff']:.2f}")
        plt.text(0.05, 0.95, stat_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel("Spine axis (Z)  [Neck → Hips]")
        plt.ylabel("Left - Right Elevation (Y)")
        plt.title(f"Pointwise Rib Hump Asymmetry\nMean Abs Diff: {calc_stats['mean_abs_diff']:.3f}")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.savefig(f"{out_prefix}_hump_diff_curve.png", dpi=300) # Higher DPI
        plt.close()

    else:
        print("Warning: Not enough valid ridge data for statistics.")
        # Add placeholders or empty
        pass

    # ---- 7. Hump Plane Angle ----
    is_left_dominant = V_L > V_R
    
    # Select points in the "hump" region
    # ... (same as before)
    side_mask = (X_roi < 0) if is_left_dominant else (X_roi > 0)
    side_pts = pts_roi[side_mask]
    Y_side = Y_roi[side_mask] 
    
    if len(side_pts) > 0:
        thr = np.percentile(Y_side, 85)
        hump_crest_mask = Y_side >= thr
        hump_pts_sel = side_pts[hump_crest_mask]
    else:
        hump_pts_sel = []

    if len(hump_pts_sel) > 10:
        hump_n, _ = fit_plane_pca(hump_pts_sel)
        hump_angle = angle_between(hump_n, n_Y)
        print(f"Dot(n_hump, n_Y): {hump_n @ n_Y:.4f}")
    else:
        hump_angle = 0.0
        hump_n = n_Y
        print("Warning: Not enough points to fit hump plane.")

    print(f"Hump plane vs back normal angle (deg): {hump_angle:.2f}")

    # ---- Add angles to stats and save ----
    stats_list.append({
        "Quantity": "rib_hump_angle",
        "Definition": definitions["rib_hump_angle"],
        "Quantity Estimated Value": float(hump_angle)
    })

    # Save stats
    with open(f"{out_prefix}_rib_hump_stats.json", "w") as f:
        json.dump(stats_list, f, indent=4)

    save_side_view_angle_plot(
        n_Y=n_Y,
        hump_n=hump_n,
        Z_roi=Z_roi,
        Y_roi=Y_roi,
        angle_deg=hump_angle,
        out_path=f"{out_prefix}_side_view_angle.png",
        hump_pts=hump_pts_sel
    )

    # --------------------------------------------------
    # NEW VISUALIZATIONS
    # --------------------------------------------------
    
    # 1. Ridge Comparison with Volume Shading
    save_ridge_comparison(
        z_vals_plot, ridge_L, ridge_R, 
        out_path=f"{out_prefix}_ridge_comparison.png"
    )

    # 2. Symmetry Heatmap
    save_symmetry_heatmap(
        X_roi, Z_roi, Y_roi, 
        out_path=f"{out_prefix}_symmetry_heatmap.png"
    )

    # 3. Plane Fit Debug (Side View Profile)
    save_plane_fit_debug(
        hump_pts_sel, hump_n, n_Y,
        out_path=f"{out_prefix}_plane_fit_debug.png"
    )

    # 4. Hump Selection Top View (Where is it?)
    save_hump_selection_vis(
        pts_roi, hump_pts_sel, 
        out_path=f"{out_prefix}_hump_selection_top_view.png"
    )

    # ---- 9. 3D Visualization: ROI + planes ----
    print("Launching 3D visualization...")

    # Base mesh (grey)
    import copy
    mesh_vis = copy.deepcopy(mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])

    # ROI point cloud (red)
    pcd_roi = o3d.geometry.PointCloud()
    pcd_roi.points = o3d.utility.Vector3dVector(pts_roi)
    pcd_roi.paint_uniform_color([1.0, 0.0, 0.0])

    # Back plane (green)
    back_center = pts.mean(axis=0)
    plane_back = create_plane_mesh(
        n=n_Y,
        point=back_center,
        size=0.5 * np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()),
        color=[0,1,0]
    )

    # Hump plane (grey)
    plane_hump = create_plane_mesh(
        n=hump_n,
        point=np.mean(hump_pts_sel, axis=0) if len(hump_pts_sel)>0 else back_center,
        size=0.4 * np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()),
        color=[0.6,0.6,0.6]
    )

    # Visualize Y-axis (back outward normal)
    origin_Y = pts_roi.mean(axis=0)
    arrow_Y = create_arrow(
        origin=origin_Y,
        direction=n_Y,
        length=0.25 * np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()),
        color=[0, 0, 1]  # BLUE = Y axis
    )

    # --------------------------------------------------
    # NEW VISUALIZATIONS
    # --------------------------------------------------
    
    # 1. Ridge Comparison with Volume Shading
    save_ridge_comparison(
        z_vals_plot, ridge_L, ridge_R, 
        out_path=f"{out_prefix}_ridge_comparison.png"
    )

    # 2. Symmetry Heatmap
    # We need the full depth map difference for this. 
    # Re-calculate or pass depth_smooth, but flipped?
    # Actually, depth_smooth is (Nz, Nx). We need to subtract left from right mirrored?
    # Simplified: interpolate to compare L vs R
    save_symmetry_heatmap(
        X_roi, Z_roi, Y_roi, 
        out_path=f"{out_prefix}_symmetry_heatmap.png"
    )

    # 3. Plane Fit Debug (Visualization of points used)
    save_plane_fit_debug(
        hump_pts_sel, hump_n, n_Y,
        out_path=f"{out_prefix}_plane_fit_debug.png"
    )

    # --------------------------------------------------
    # Interactive 3D Visualization
    # --------------------------------------------------

    # Ground plane
    plane_ground = create_plane_mesh(
        n=np.array([0, 1, 0]),
        point=pts.mean(axis=0),
        size=1.2 * np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()),
        color=[0.3, 0.3, 0.9]
    )

    # Rib-hump line
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p_left, p_right]),
        lines=o3d.utility.Vector2iVector([[0,1]])
    )
    line.colors = o3d.utility.Vector3dVector([[1,0,0]])

    # Text legends
    # Check if text is supported, otherwise print legend
    if not hasattr(o3d.geometry.TriangleMesh, 'create_text'):
        print("\n" + "="*40)
        print("  3D Visualization Legend")
        print("="*40)
        print("  [Green Plane] : Back Reference Plane")
        print("  [Gray Plane]  : Rib Hump Plane")
        print("  [Blue Arrow]  : Y-Axis (Back Normal)")
        print("  [Blue Plane]  : Ground Plane")
        print("  [Red Points]  : Thoracic ROI")
        print("="*40 + "\n")

    t_Y = text3d("Blue: Y axis (Back normal)", origin_Y)
    t_Back = text3d("Green: Back plane", back_center)
    t_Hump = text3d("Gray: Rib hump plane", np.mean(hump_pts_sel,0) if len(hump_pts_sel)>0 else back_center)
    t_Gnd = text3d("Blue plane: Ground", back_center)

    # Interactive Visualizer (Only for saving screenshot now)
    if save_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Rib Hump Analysis", width=1024, height=768, visible=False)

        for g in [mesh_vis, pcd_roi, plane_back, plane_hump, arrow_Y, plane_ground, line, t_Y, t_Back, t_Hump, t_Gnd]:
            vis.add_geometry(g)

        # Setup Camera
        ctr = vis.get_view_control()
        # Look along -X (from right side), Up is Y (vertical)
        ctr.set_front(-n_X)
        ctr.set_up(n_Y)
        ctr.set_lookat(origin_Y)
        ctr.set_zoom(0.8)

        vis.poll_events()
        vis.update_renderer()
        out_vis_path = f"{out_prefix}_roi_vis.png"
        vis.capture_screen_image(out_vis_path)
        print(f"Saved 3D visualization to {out_vis_path}")

        # vis.run() # Disabled per user request
        vis.destroy_window()

    print(f"Done. Outputs: {out_prefix}_depth_map.png, {out_prefix}_ridge_lines.png")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Rib Hump Analysis AIX")
    parser.add_argument("--mesh_path", help="Path to input mesh (.ply/.obj)")
    parser.add_argument("--midline_json", help="Path to midline JSON file")
    parser.add_argument("--out_prefix", nargs="?", help="Output prefix (default: derived from mesh path)")
    parser.add_argument("--save-vis", action="store_true", help="Save 3D visualization as PNG")

    args = parser.parse_args()

    if not args.out_prefix:
        d = os.path.dirname(args.mesh_path)
        base = os.path.splitext(os.path.basename(args.mesh_path))[0]
        prefix = os.path.join(d, base + "_rib")
    else:
        prefix = args.out_prefix

    main(args.mesh_path, args.midline_json, prefix, save_vis=args.save_vis)


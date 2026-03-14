#!/usr/bin/env python3
"""
skeletonize_full.py

A Gradio app that:
1. Loads a mesh/pointcloud.
2. Voxelizes the ENTIRE cloud (no ROI slicing).
3. Skeletonizes the volume.
4. Visualizes the original cloud and the full skeleton using Plotly.

Usage:
    python skeletonize_full.py
"""

import os
import numpy as np
import open3d as o3d
import gradio as gr
import plotly.graph_objs as go


# Use true volumetric 3D skeletonization only
# ----------------------
try:
    # Use the dedicated 3D skeletonizer from scikit-image
    from skimage.morphology import skeletonize_3d as skeletonize
except Exception as e:
    # Fail early and clearly if skeletonize_3d is not available
    raise ImportError(
        "skimage.morphology.skeletonize_3d is required for true 3D volumetric skeletonization. "
        "Please install or upgrade scikit-image so that `skeletonize_3d` is available."
    ) from e



# ---------------------- IO & Geometry Helpers ----------------------

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

# ---------------------- Skeletonization ----------------------

def run_skeletonization(binary_grid):
    bool_grid = (binary_grid > 0)
    skeleton = skeletonize(bool_grid).astype(np.uint8)
    return skeleton

def skeleton_to_points(skeleton_grid, min_b, voxel_size):
    xs, ys, zs = np.where(skeleton_grid > 0)
    if len(xs) == 0:
        return np.zeros((0,3))
    pts = grid_indices_to_points(np.vstack([xs, ys, zs]).T, min_b, voxel_size)
    return pts

# ---------------------- Main Logic ----------------------

def process_full_skeleton(mesh_file, voxel_frac):
    if mesh_file is None:
        return None, "Please upload a file."
    
    try:
        input_path = mesh_file.name
        points = load_mesh_or_cloud(input_path)
        
        min_b, max_b = axis_aligned_bounds(points)
        H = max_b[1] - min_b[1]
        if H <= 0:
            return None, "Invalid object height."
            
        voxel_size = voxel_frac * H
        
        # Voxelize
        grid, min_grid_b, voxel_size_used = voxelize_pointcloud_to_grid(points, voxel_size)
        
        # Skeletonize
        skel_grid = run_skeletonization(grid)
        skel_pts = skeleton_to_points(skel_grid, min_grid_b, voxel_size_used)
        
        # Visualization
        fig = create_plotly_viz(points, skel_pts)
        
        log = f"Processed {len(points)} points.\n"
        log += f"Height: {H:.4f} m\n"
        log += f"Voxel Size: {voxel_size:.4f} m ({voxel_frac} * H)\n"
        log += f"Skeleton Points: {len(skel_pts)}\n"
        
        return fig, log
        
    except Exception as e:
        import traceback
        return None, f"Error:\n{traceback.format_exc()}"

def create_plotly_viz(body_pts, skel_pts):
    figs_data = []
    
    # 1. Body Cloud (Subsampled)
    if len(body_pts) > 15000:
        indices = np.random.choice(len(body_pts), 15000, replace=False)
        pts = body_pts[indices]
    else:
        pts = body_pts
        
    figs_data.append(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='markers',
        marker=dict(size=1.5, color='lightgray', opacity=0.3),
        name='Body Cloud'
    ))
    
    # 2. Skeleton Points
    if len(skel_pts) > 0:
        figs_data.append(go.Scatter3d(
            x=skel_pts[:,0], y=skel_pts[:,1], z=skel_pts[:,2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Skeleton'
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

# ---------------------- Gradio UI ----------------------

with gr.Blocks(title="Full Skeletonization Viewer") as demo:
    gr.Markdown("# Full Mesh Skeletonization")
    gr.Markdown("Upload a mesh to see its full internal skeleton (no pruning).")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Input Mesh/PCD", file_types=[".ply", ".obj", ".pcd", ".xyz"])
            voxel_slider = gr.Slider(label="Voxel Size (fraction of H)", minimum=0.005, maximum=0.1, value=0.015, step=0.001)
            run_btn = gr.Button("Skeletonize", variant="primary")
            log_out = gr.Textbox(label="Logs", lines=10)
            
        with gr.Column(scale=3):
            plot_out = gr.Plot(label="3D Visualization")
            
    run_btn.click(fn=process_full_skeleton, 
                  inputs=[file_in, voxel_slider], 
                  outputs=[plot_out, log_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)

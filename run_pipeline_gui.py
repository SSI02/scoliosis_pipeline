#!/usr/bin/env python3
"""
GUI-Based 3D Body Scanning Pipeline
====================================
Complete interactive Gradio interface with:
- Fully automated AI-based 3D reconstruction processing
- Real-time visualization after each stage
- Progress tracking AND detailed logs
- Manual alignment step included

Usage:
    python run_pipeline_gui.py
    
Then open http://localhost:7860 in your browser.
"""

import os
import sys
import subprocess
import shutil
import time
import json
import threading
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional, Tuple
import numpy as np
import gradio as gr
import plotly.graph_objects as go

# For post-processing NEAR mesh (quality filtering)
try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False

try:
    from skimage.filters import threshold_otsu
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Pipeline configuration - Environment names from last_readme.md"""
    
    # Conda environments for each stage (from last_readme.md)
    ENVS = {
        'preprocess': 'vv_vggt',       # video->frames, AI-based 3D reconstruction, glb_to_ply
        'denoise': 'vv_denoise',        # denoising
        'mesh': 'vv_meshlab',           # mesh reconstruction
        'ground': 'vv_gnd_estimate',    # ground plane estimation
        'xaxis': 'vv_aix',              # x-axis alignment
        'alignment': 'vv_mesh_alignment', # manual alignment v2
        'aix': 'vv_aix'                 # all AIX estimation
    }
    
    # Preprocessing
    NUM_FRAMES = 5
    
    # AI-based 3D reconstruction parameters - separate thresholds for FAR and NEAR
    VGGT_CONF_THRESH_FAR = 75.0   # Confidence threshold for FAR (full body)
    VGGT_CONF_THRESH_NEAR = 85.0  # Confidence threshold for NEAR (close-up, may need lower)
    VGGT_MASK_BLACK = True
    VGGT_MASK_WHITE = True
    VGGT_SHOW_CAM = False
    VGGT_MASK_SKY = False
    
    # Ground plane alignment
    GROUND_ALIGN_TARGET = [0, 1, 0]  # Target axis: Negative Y (feet at max Y, head at min Y)
    
    # X-axis alignment (skeleton-based)
    XAXIS_LOW_FRAC = 0.05    # ROI starts at 5% above ground
    XAXIS_HIGH_FRAC = 0.25   # ROI ends at 25% (below knees)
    XAXIS_VOXEL_FRAC = 0.01  # Voxel size = 1% of height
    XAXIS_GROUND_TOL = 0.02  # Ground tolerance = 2% of height
    XAXIS_GEODESIC_THRESH = 0.5  # Geodesic distance threshold (meters)
    
    # Denoising - Adaptive parameters (computed from point cloud)
    # These are fallback values only, actual values computed dynamically
    USE_SOR = True
    USE_ROR = True
    USE_DBSCAN = True  # Enable DBSCAN for better cluster-based denoising
    
    # FAR mesh denoising (heavy denoising for noisy far scans)
    FAR_DENOISE_AGGRESSIVE = True
    FAR_SOR_STD_MULTIPLIER = 1.0   # Lower = more aggressive (removes more points)
    FAR_ROR_RADIUS_MULTIPLIER = 1.5  # Multiplier on median_nn distance
    
    # NEAR mesh denoising (light denoising)
    NEAR_DENOISE_ENABLED = True   # Enable light denoising for near mesh
    NEAR_SOR_STD_MULTIPLIER = 2.0  # Higher = more lenient
    NEAR_ROR_RADIUS_MULTIPLIER = 2.5  # More lenient radius
    
    # Mesh reconstruction  
    NORMAL_NEIGHBORS = 500
    NORMAL_SMOOTH = 5
    POISSON_DEPTH = 8
    HC_ITERATIONS = 2
    
    # X-axis alignment
    VOXEL_FRAC = 0.01
    GROUND_TOL_FRAC = 0.02
    GEODESIC_THRESH = 0.5
    
    # Hip AIX
    SLICE_FRAC = 0.02


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def create_plotly_visualization(mesh_path: str, title: str = "3D Visualization", max_points: int = 80000):
    """Create Plotly figure for 3D visualization - returns a Plotly Figure object"""
    try:
        if not mesh_path:
            print(f"Visualization: No path provided")
            return create_empty_figure(title, "No file path provided")
        
        if not os.path.exists(mesh_path):
            print(f"Visualization: File not found: {mesh_path}")
            return create_empty_figure(title, f"File not found: {os.path.basename(mesh_path)}")
        
        # Try loading as mesh first, then as point cloud
        points = None
        colors = None
        
        # Try trimesh first (more reliable, no display issues)
        try:
            import trimesh
            mesh = trimesh.load(mesh_path, force='mesh')
            
            if isinstance(mesh, trimesh.PointCloud):
                points = np.array(mesh.vertices)
                if hasattr(mesh, 'colors') and mesh.colors is not None:
                    colors = np.array(mesh.colors)[:, :3] / 255.0
            elif isinstance(mesh, trimesh.Scene):
                # Scene with multiple geometries
                all_verts = []
                all_colors = []
                for g in mesh.geometry.values():
                    if hasattr(g, 'vertices') and len(g.vertices) > 0:
                        all_verts.append(np.array(g.vertices))
                        if hasattr(g, 'visual') and hasattr(g.visual, 'vertex_colors'):
                            all_colors.append(np.array(g.visual.vertex_colors)[:, :3] / 255.0)
                if all_verts:
                    points = np.vstack(all_verts)
                    if all_colors and len(all_colors) == len(all_verts):
                        colors = np.vstack(all_colors)
            elif hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                points = np.array(mesh.vertices)
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                    vc = mesh.visual.vertex_colors
                    if vc is not None and len(vc) > 0:
                        colors = np.array(vc)[:, :3] / 255.0
        except Exception as e:
            print(f"Trimesh load failed: {e}, trying alternative methods...")
        
        # Fallback: try loading as point cloud with trimesh
        if points is None or len(points) == 0:
            try:
                import trimesh
                # Try loading as point cloud
                pcd = trimesh.load(mesh_path, force='pointcloud')
                if hasattr(pcd, 'vertices') and len(pcd.vertices) > 0:
                    points = np.array(pcd.vertices)
                    if hasattr(pcd, 'colors') and pcd.colors is not None:
                        colors = np.array(pcd.colors)[:, :3] / 255.0
            except Exception as e:
                print(f"Alternative load methods also failed: {e}")
        
        if points is None or len(points) == 0:
            print(f"Visualization: No points loaded from {mesh_path}")
            return create_empty_figure(title, "Could not load mesh/point cloud")
        
        # Subsample for performance
        n = len(points)
        if n > max_points:
            indices = np.random.choice(n, max_points, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
        
        print(f"Visualization: Loaded {len(points)} points from {os.path.basename(mesh_path)}")
        
        # Create plotly figure
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Color by height if no vertex colors
        if colors is not None and len(colors) == len(points):
            r = (colors[:, 0] * 255).astype(np.uint8)
            g = (colors[:, 1] * 255).astype(np.uint8)
            b = (colors[:, 2] * 255).astype(np.uint8)
            color_str = [f'rgb({int(rr)},{int(gg)},{int(bb)})' for rr, gg, bb in zip(r, g, b)]
            marker = dict(size=2, color=color_str, opacity=0.85)
        else:
            # Color by height (Z) for better visualization
            z_min, z_max = z.min(), z.max()
            if z_max > z_min:
                norm_z = (z - z_min) / (z_max - z_min)
                # Blue to Red gradient
                color_vals = np.zeros((len(z), 3))
                color_vals[:, 0] = norm_z        # Red
                color_vals[:, 2] = 1 - norm_z    # Blue
                color_vals[:, 1] = 0.3           # Green
                color_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                            for r, g, b in color_vals]
                marker = dict(size=2, color=color_str, opacity=0.85)
            else:
                marker = dict(size=2, color='steelblue', opacity=0.85)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=marker,
            name=title,
            hovertemplate='X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>'
        )])
        
        # Add coordinate axes at center
        center = np.mean(points, axis=0)
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        axis_len = np.max(extent) * 0.15
        
        # X axis (red)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0] + axis_len], y=[center[1], center[1]], z=[center[2], center[2]],
            mode='lines', line=dict(color='red', width=6), name='X', showlegend=False
        ))
        # Y axis (green)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0]], y=[center[1], center[1] + axis_len], z=[center[2], center[2]],
            mode='lines', line=dict(color='green', width=6), name='Y', showlegend=False
        ))
        # Z axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0]], y=[center[1], center[1]], z=[center[2], center[2] + axis_len],
            mode='lines', line=dict(color='blue', width=6), name='Z', showlegend=False
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b> ({len(points):,} pts)", font=dict(size=14)),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='data',
                bgcolor='#1a1a2e',
                xaxis=dict(gridcolor='#444', zerolinecolor='#666'),
                yaxis=dict(gridcolor='#444', zerolinecolor='#666'),
                zaxis=dict(gridcolor='#444', zerolinecolor='#666')
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=500,
            paper_bgcolor='#f0f0f0',
            showlegend=False
        )
        
        return fig
    except Exception as e:
        print(f"Visualization error for {mesh_path}: {e}")
        import traceback
        traceback.print_exc()
        return create_empty_figure(title, f"Error: {str(e)[:50]}")


def create_empty_figure(title: str, message: str):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title=title,
        height=500,
        paper_bgcolor='#f0f0f0'
    )
    return fig


def create_mesh_visualization_html(mesh_path: str, title: str = "3D Visualization") -> str:
    """Create HTML for 3D mesh visualization using plotly"""
    fig = create_plotly_visualization(mesh_path, title)
    if fig:
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    return f"<p style='color:#888; text-align:center;'>Could not load visualization for: {mesh_path}</p>"


def create_frames_gallery_html(frames_dir: str, max_frames: int = 12) -> str:
    """Create HTML gallery of extracted frames"""
    try:
        frames = sorted(Path(frames_dir).glob("*.png"))[:max_frames]
        if not frames:
            return "<p>No frames found</p>"
        
        html = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
        for frame in frames:
            html += f'<img src="file={frame}" style="width: 120px; height: auto; border-radius: 4px;">'
        html += '</div>'
        return html
    except Exception as e:
        return f"<p>Gallery error: {str(e)}</p>"


# ============================================================
# WEB-BASED MANUAL ALIGNMENT (Similar to Original Flow)
# ============================================================

class WebBasedAlignment:
    """
    Web-based manual alignment following the original manual_alignment_v2.py flow:
    1. First select points on SOURCE mesh
    2. Then select corresponding points on TARGET mesh
    3. Compute transformation and apply
    """
    
    @staticmethod
    def load_mesh_points(mesh_path: str, max_points: int = 50000) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load mesh/point cloud and return points and colors"""
        import trimesh
        
        points = None
        colors = None
        
        # Try loading as mesh first
        try:
            mesh = trimesh.load(str(mesh_path), force='mesh')
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                if len(mesh.vertices) > max_points:
                    # Sample points uniformly from mesh surface
                    points = mesh.sample(max_points)
                else:
                    points = np.array(mesh.vertices)
                
                # Get colors if available
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                    vc = mesh.visual.vertex_colors
                    if vc is not None and len(vc) > 0:
                        colors = np.array(vc)[:, :3] / 255.0
        except:
            pass
        
        # Try as point cloud if mesh loading failed
        if points is None or len(points) == 0:
            try:
                pcd = trimesh.load(str(mesh_path), force='pointcloud')
                if hasattr(pcd, 'vertices') and len(pcd.vertices) > 0:
                    points = np.array(pcd.vertices)
                    if hasattr(pcd, 'colors') and pcd.colors is not None:
                        colors = np.array(pcd.colors)[:, :3] / 255.0
            except:
                pass
        
        # Subsample if needed
        if points is not None and len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
        
        return points, colors
    
    @staticmethod
    def create_single_mesh_figure(mesh_path: str, mesh_name: str, selected_points: list = None,
                                   point_color: str = 'lime', max_points: int = 50000) -> 'go.Figure':
        """Create Plotly figure for a single mesh with selected points marked"""
        
        pts, colors = WebBasedAlignment.load_mesh_points(mesh_path, max_points)
        
        if pts is None:
            return None
        
        # Color by height for easier identification (like original)
        z_vals = pts[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        if z_max > z_min:
            norm_z = (z_vals - z_min) / (z_max - z_min)
            # Blue (low) to Red (high) gradient
            color_arr = np.zeros((len(pts), 3))
            color_arr[:, 0] = norm_z        # Red increases with height
            color_arr[:, 2] = 1 - norm_z    # Blue decreases with height
            color_arr[:, 1] = 0.3           # Slight green
            color_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                        for r, g, b in color_arr]
        else:
            color_str = 'rgba(100, 100, 200, 0.8)'
        
        fig = go.Figure()
        
        # Main point cloud with prominent coordinate display
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=color_str, opacity=0.8),
            name=mesh_name,
            # Make coordinates easy to copy - show them prominently
            hovertemplate=(
                '<b>COPY THESE VALUES:</b><br>'
                '<b>X:</b> %{x:.4f}<br>'
                '<b>Y:</b> %{y:.4f}<br>'
                '<b>Z:</b> %{z:.4f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Add selected points with large numbered markers
        if selected_points and len(selected_points) > 0:
            sp_arr = np.array(selected_points)
            # Different colors for each point
            point_colors = ['#00ff00', '#ffff00', '#ff00ff', '#00ffff', '#ff8000', '#8000ff', '#ff0080', '#0080ff']
            
            for i, pt in enumerate(selected_points):
                color = point_colors[i % len(point_colors)]
                fig.add_trace(go.Scatter3d(
                    x=[pt[0]],
                    y=[pt[1]],
                    z=[pt[2]],
                    mode='markers+text',
                    marker=dict(size=15, color=color, symbol='diamond', 
                               line=dict(width=2, color='black')),
                    text=[f'<b>{i+1}</b>'],
                    textposition='top center',
                    textfont=dict(size=14, color='black'),
                    name=f'Point {i+1}',
                    showlegend=True
                ))
        
        # Add coordinate frame at center
        center = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        axis_len = extent.max() * 0.15
        
        # X axis (red)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0] + axis_len], y=[center[1], center[1]], z=[center[2], center[2]],
            mode='lines', line=dict(color='red', width=5), name='X-axis', showlegend=False
        ))
        # Y axis (green)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0]], y=[center[1], center[1] + axis_len], z=[center[2], center[2]],
            mode='lines', line=dict(color='green', width=5), name='Y-axis', showlegend=False
        ))
        # Z axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0]], y=[center[1], center[1]], z=[center[2], center[2] + axis_len],
            mode='lines', line=dict(color='blue', width=5), name='Z-axis', showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>{mesh_name}</b> - Click to view coordinates, then enter below',
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                bgcolor='#1a1a2e'
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=550,
            paper_bgcolor='#f0f0f0',
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
        )
        
        return fig
    
    @staticmethod
    def compute_similarity_transform(source_pts: np.ndarray, target_pts: np.ndarray, 
                                      allow_scale: bool = True) -> Tuple[np.ndarray, float]:
        """Compute optimal similarity transformation (rotation, translation, scale)"""
        source_pts = np.array(source_pts)
        target_pts = np.array(target_pts)
        
        # Center the points
        source_center = source_pts.mean(axis=0)
        target_center = target_pts.mean(axis=0)
        
        source_centered = source_pts - source_center
        target_centered = target_pts - target_center
        
        # Compute scale
        if allow_scale:
            source_rms = np.sqrt(np.mean(np.sum(source_centered**2, axis=1)))
            target_rms = np.sqrt(np.mean(np.sum(target_centered**2, axis=1)))
            scale = target_rms / source_rms if source_rms > 1e-8 else 1.0
        else:
            scale = 1.0
        
        # Scale source
        source_scaled = source_centered * scale
        
        # Compute rotation using SVD
        H = source_scaled.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_center - scale * R @ source_center
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t
        
        return T, scale
    
    @staticmethod
    def apply_transformation(mesh_path: str, transform: np.ndarray, output_path: str) -> bool:
        """Apply transformation matrix to mesh and save"""
        import trimesh
        
        try:
            # Try loading as mesh first
            try:
                mesh = trimesh.load(str(mesh_path), force='mesh')
                if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                    mesh.apply_transform(transform)
                    mesh.export(str(output_path))
                    return True
            except:
                pass
            
            # Try as point cloud
            try:
                pcd = trimesh.load(str(mesh_path), force='pointcloud')
                if hasattr(pcd, 'vertices') and len(pcd.vertices) > 0:
                    pcd.apply_transform(transform)
                    pcd.export(str(output_path))
                    return True
            except:
                pass
            
            return False
        except Exception as e:
            print(f"Error applying transformation: {e}")
            return False
    
    @staticmethod
    def compute_alignment_error(source_pts: np.ndarray, target_pts: np.ndarray, 
                                 transform: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Compute RMSE and max error after transformation"""
        source_pts = np.array(source_pts)
        target_pts = np.array(target_pts)
        
        source_h = np.hstack([source_pts, np.ones((len(source_pts), 1))])
        source_transformed = (transform @ source_h.T).T[:, :3]
        
        errors = np.linalg.norm(source_transformed - target_pts, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = errors.max()
        
        return rmse, max_error, errors
    
    @staticmethod
    def create_alignment_preview_figure(source_path: str, target_path: str, 
                                         source_pts: list, target_pts: list,
                                         max_points: int = 30000) -> 'go.Figure':
        """Create figure showing both meshes with correspondence lines"""
        
        src_cloud, _ = WebBasedAlignment.load_mesh_points(source_path, max_points)
        tgt_cloud, _ = WebBasedAlignment.load_mesh_points(target_path, max_points)
        
        if src_cloud is None or tgt_cloud is None:
            return None
        
        # Offset meshes side by side
        src_range = src_cloud.max(axis=0) - src_cloud.min(axis=0)
        tgt_range = tgt_cloud.max(axis=0) - tgt_cloud.min(axis=0)
        offset = max(src_range.max(), tgt_range.max()) * 1.5
        
        fig = go.Figure()
        
        # Source cloud (blue)
        fig.add_trace(go.Scatter3d(
            x=src_cloud[:, 0] - offset/2, y=src_cloud[:, 1], z=src_cloud[:, 2],
            mode='markers', marker=dict(size=1.5, color='steelblue', opacity=0.5),
            name='SOURCE (to align)'
        ))
        
        # Target cloud (orange)
        fig.add_trace(go.Scatter3d(
            x=tgt_cloud[:, 0] + offset/2, y=tgt_cloud[:, 1], z=tgt_cloud[:, 2],
            mode='markers', marker=dict(size=1.5, color='darkorange', opacity=0.5),
            name='TARGET (reference)'
        ))
        
        # Correspondence lines and points
        colors = ['#00ff00', '#ffff00', '#ff00ff', '#00ffff', '#ff8000', '#8000ff']
        for i, (sp, tp) in enumerate(zip(source_pts, target_pts)):
            color = colors[i % len(colors)]
            # Line
            fig.add_trace(go.Scatter3d(
                x=[sp[0] - offset/2, tp[0] + offset/2],
                y=[sp[1], tp[1]],
                z=[sp[2], tp[2]],
                mode='lines', line=dict(color=color, width=4),
                name=f'Corr {i+1}', showlegend=False
            ))
            # Source point
            fig.add_trace(go.Scatter3d(
                x=[sp[0] - offset/2], y=[sp[1]], z=[sp[2]],
                mode='markers+text', marker=dict(size=12, color=color, symbol='diamond'),
                text=[str(i+1)], textposition='top center',
                name=f'Src {i+1}', showlegend=False
            ))
            # Target point  
            fig.add_trace(go.Scatter3d(
                x=[tp[0] + offset/2], y=[tp[1]], z=[tp[2]],
                mode='markers+text', marker=dict(size=12, color=color, symbol='diamond'),
                text=[str(i+1)], textposition='top center',
                name=f'Tgt {i+1}', showlegend=False
            ))
        
        fig.update_layout(
            title='Correspondence Preview - Lines show matched points',
            scene=dict(aspectmode='data', bgcolor='#f0f0f0'),
            height=500, margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig


# Global state for step-by-step alignment (like original code)
alignment_state = {
    'source_path': None,
    'target_path': None,
    'source_points': [],      # Points selected on source mesh
    'target_points': [],      # Corresponding points on target mesh
    'current_step': 'idle',   # 'idle', 'selecting_source', 'selecting_target', 'preview', 'done'
    'num_points': 4,          # Number of points to select
    'transformation': None,
    'output_path': None
}


def reset_alignment_state():
    """Reset alignment state to start fresh"""
    global alignment_state
    alignment_state = {
        'source_path': None,
        'target_path': None,
        'source_points': [],
        'target_points': [],
        'current_step': 'idle',
        'num_points': 4,
        'transformation': None,
        'output_path': None
    }


# ============================================================
# PIPELINE EXECUTOR
# ============================================================

class PipelineExecutor:
    """Pipeline execution with progress tracking and visualization"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.work_dir = None
        self.outputs = {}
        self.log_lines = []
        self.current_stage = 0
        self.total_stages = 10  # Updated: added VGGT and manual alignment
        self.visualizations = {}
        
    def setup_working_dir(self):
        """Create working directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = self.base_dir / "working_directory" / f"run_{timestamp}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.dirs = {
            'frames_far': self.work_dir / "01_frames" / "far",
            'frames_near': self.work_dir / "01_frames" / "near",
            'vggt_far': self.work_dir / "02_vggt" / "far",
            'vggt_near': self.work_dir / "02_vggt" / "near",
            'ply_far': self.work_dir / "03_pointcloud" / "far",
            'ply_near': self.work_dir / "03_pointcloud" / "near",
            'denoised': self.work_dir / "04_denoised",
            'mesh_far': self.work_dir / "05_mesh" / "far",
            'mesh_near': self.work_dir / "05_mesh" / "near",
            'ground_aligned': self.work_dir / "06_ground_aligned",
            'manual_aligned': self.work_dir / "07_manual_aligned",
            'aix': self.work_dir / "08_aix_results",
            'logs': self.work_dir / "logs",
            'visualizations': self.work_dir / "visualizations"
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        return str(self.work_dir)
    
    def log(self, message: str, level: str = "INFO"):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Status indicators
        prefixes = {
            "SUCCESS": "[SUCCESS]",
            "ERROR": "[ERROR]",
            "WARNING": "[WARNING]",
            "PROGRESS": "[PROGRESS]",
            "STAGE": "[STAGE]",
            "FILE": "[FILE]",
            "VIS": "[VIS]",
            "INFO": "[INFO]"
        }
        prefix = prefixes.get(level, "[INFO]")
        
        line = f"[{timestamp}] {prefix} {message}"
        self.log_lines.append(line)
        
        # Write to log file
        if self.work_dir:
            log_file = self.dirs['logs'] / "pipeline.log"
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] [{level}] {message}\n")
        
        return line
    
    def get_log_text(self) -> str:
        """Get all log lines as text"""
        return "\n".join(self.log_lines[-150:])  # Last 150 lines
    
    def get_progress(self) -> float:
        """Get overall progress (0-1)"""
        return self.current_stage / self.total_stages
    
    def get_progress_text(self) -> str:
        """Get progress as text"""
        pct = int(self.get_progress() * 100)
        bar_filled = int(pct / 5)
        bar_empty = 20 - bar_filled
        bar = "█" * bar_filled + "░" * bar_empty
        return f"[{bar}] {pct}% - Stage {self.current_stage}/{self.total_stages}"
    
    def run_command(self, cmd: list, env_name: str = None, timeout: int = 3600) -> tuple:
        """Run a shell command with proper conda environment handling"""
        if env_name:
            cmd_str = ' '.join(str(c) for c in cmd)
            # Use bash with conda initialization for proper environment handling
            # This fixes "python: command not found" issues
            full_cmd = f"""bash -c 'source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null; conda activate {env_name} && {cmd_str}'"""
        else:
            full_cmd = ' '.join(str(c) for c in cmd)
        
        self.log(f"Running: {cmd[1] if len(cmd) > 1 else full_cmd[:80]}...", "PROGRESS")
        
        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                executable='/bin/bash'
            )
            
            # Log output if verbose
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:
                    if line.strip():
                        self.log(f"  stdout: {line[:100]}")
            
            if result.returncode != 0 and result.stderr:
                for line in result.stderr.strip().split('\n')[-3:]:
                    if line.strip():
                        self.log(f"  stderr: {line[:100]}", "WARNING")
            
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, '', 'Command timed out'
        except Exception as e:
            return False, '', str(e)
    
    def save_stage_output(self, stage_name: str, data: dict):
        """Save stage output"""
        output_file = self.dirs['logs'] / f"{stage_name}_output.json"
        data['timestamp'] = datetime.now().isoformat()
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.outputs[stage_name] = data
    
    def add_visualization(self, stage_name: str, file_path: str, vis_type: str = "mesh"):
        """Add visualization for a stage"""
        self.visualizations[stage_name] = {
            'path': file_path,
            'type': vis_type
        }


# ============================================================
# PIPELINE STAGES
# ============================================================

def run_stage_video_to_frames(executor: PipelineExecutor, far_video: str, near_video: str) -> bool:
    """Stage 1: Extract frames from videos"""
    executor.current_stage = 1
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 1/10: VIDEO TO FRAMES", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['preprocess']}")
    
    success = True
    
    for video_path, frames_dir, label in [
        (far_video, executor.dirs['frames_far'], "FAR"),
        (near_video, executor.dirs['frames_near'], "NEAR")
    ]:
        executor.log(f"Processing {label} video: {os.path.basename(video_path)}", "PROGRESS")
        
        temp_dir = executor.work_dir / f"temp_{label.lower()}"
        temp_dir.mkdir(exist_ok=True)
        shutil.copy(video_path, temp_dir / os.path.basename(video_path))
        
        cmd = [
            "python", executor.base_dir / "prepro" / "vid_to_frame_cli.py",
            "--input_folder", str(temp_dir),
            "--output_folder", str(frames_dir.parent),
            "--num_frames", str(Config.NUM_FRAMES)
        ]
        
        ok, out, err = executor.run_command(cmd, Config.ENVS['preprocess'])
        
        if ok:
            video_name = Path(video_path).stem
            actual_output = frames_dir.parent / video_name
            if actual_output.exists():
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)
                shutil.move(str(actual_output), str(frames_dir))
            
            frame_count = len(list(frames_dir.glob("*.png")))
            executor.log(f"{label} video: {frame_count} frames extracted", "SUCCESS")
            executor.log(f"Saved to: {frames_dir}", "FILE")
        else:
            executor.log(f"{label} video processing failed: {err[:200]}", "ERROR")
            success = False
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    executor.save_stage_output('01_frames', {
        'far_frames': str(executor.dirs['frames_far']),
        'near_frames': str(executor.dirs['frames_near']),
        'success': success
    })
    
    return success


def run_stage_vggt(executor: PipelineExecutor, 
                    far_threshold: float = None, 
                    near_threshold: float = None) -> bool:
    """Stage 2: AI-based 3D Reconstruction (Automated)"""
    executor.current_stage = 2
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 2/10: AI-BASED 3D RECONSTRUCTION", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['preprocess']}")
    
    # Use provided thresholds or defaults
    far_conf = far_threshold if far_threshold is not None else Config.VGGT_CONF_THRESH_FAR
    near_conf = near_threshold if near_threshold is not None else Config.VGGT_CONF_THRESH_NEAR
    
    executor.log(f"FAR confidence threshold: {far_conf}")
    executor.log(f"NEAR confidence threshold: {near_conf}")
    executor.log("Running AI-based 3D reconstruction automatically (this may take a few minutes)...", "PROGRESS")
    
    success = True
    
    # Create a Python script to run VGGT programmatically
    vggt_script = executor.work_dir / "run_vggt.py"
    
    vggt_code = '''
import os
import sys
import shutil
import numpy as np

# Add AI-based 3D reconstruction path
sys.path.insert(0, "{vggt_path}")
sys.path.insert(0, "{vggt_path}/vggt")

os.chdir("{vggt_path}")

import torch
import cv2
import glob
from datetime import datetime

# Import AI-based 3D reconstruction modules
from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {{device}}")

# Load model
print("Loading AI-based 3D reconstruction model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)
print("Model loaded successfully")

def run_vggt_on_frames(frames_dir, output_dir, label, conf_thres=70.0):
    """Run AI-based 3D reconstruction on a directory of frames with specified confidence threshold"""
    print(f"Processing {{label}} frames from: {{frames_dir}}")
    print(f"Using confidence threshold: {{conf_thres}}")
    
    # Create output directory structure
    target_dir = os.path.join(output_dir, f"vggt_{{label}}")
    images_dir = os.path.join(target_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Copy frames to target dir
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    print(f"Found {{len(frame_files)}} frames")
    
    for i, f in enumerate(frame_files):
        shutil.copy(f, os.path.join(images_dir, f"{{i:06d}}.png"))
    
    # Load and preprocess
    image_names = sorted(glob.glob(os.path.join(images_dir, "*")))
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {{images.shape}}")
    
    # Run inference
    print("Running AI-based 3D reconstruction inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    # Process predictions
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions['pose_enc_list'] = None
    
    # Generate world points
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    # Save predictions
    np.savez(os.path.join(target_dir, "predictions.npz"), **predictions)
    
    # Generate GLB - threshold depends on which mesh (far vs near)
    glb_name = f"{{label}}_reconstruction.glb"
    glb_path = os.path.join(target_dir, glb_name)
    
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,  # Use the threshold passed to this function
        filter_by_frames="All",
        mask_black_bg={mask_black},
        mask_white_bg={mask_white},
        show_cam={show_cam},
        mask_sky={mask_sky},
        target_dir=target_dir,
        prediction_mode="Depthmap and Camera Branch",
    )
    glbscene.export(file_obj=glb_path)
    
    print(f"Saved GLB to: {{glb_path}}")
    
    # Cleanup
    torch.cuda.empty_cache()
    
    return glb_path

# Process FAR with FAR threshold
far_frames = "{far_frames}"
far_output = "{far_output}"
far_conf_thres = {far_conf_thres}
print(f"\\nFAR confidence threshold: {{far_conf_thres}}")
if os.path.isdir(far_frames):
    glb_far = run_vggt_on_frames(far_frames, far_output, "far", conf_thres=far_conf_thres)
    print(f"FAR_GLB={{glb_far}}")

# Process NEAR with NEAR threshold  
near_frames = "{near_frames}"
near_output = "{near_output}"
near_conf_thres = {near_conf_thres}
print(f"\\nNEAR confidence threshold: {{near_conf_thres}}")
if os.path.isdir(near_frames):
    glb_near = run_vggt_on_frames(near_frames, near_output, "near", conf_thres=near_conf_thres)
    print(f"NEAR_GLB={{glb_near}}")

print("AI-based 3D reconstruction processing complete!")
'''.format(
        vggt_path=str(executor.base_dir / "prepro" / "vggt"),
        far_frames=str(executor.dirs['frames_far']),
        near_frames=str(executor.dirs['frames_near']),
        far_output=str(executor.dirs['vggt_far'].parent),
        near_output=str(executor.dirs['vggt_near'].parent),
        far_conf_thres=far_conf,
        near_conf_thres=near_conf,
        mask_black=Config.VGGT_MASK_BLACK,
        mask_white=Config.VGGT_MASK_WHITE,
        show_cam=Config.VGGT_SHOW_CAM,
        mask_sky=Config.VGGT_MASK_SKY
    )
    
    with open(vggt_script, 'w') as f:
        f.write(vggt_code)
    
    # Run AI-based 3D reconstruction
    cmd = ["python", str(vggt_script)]
    ok, out, err = executor.run_command(cmd, Config.ENVS['preprocess'], timeout=1800)  # 30 min timeout
    
    if ok:
        # Find and copy GLB files
        for label, vggt_dir, ply_dir in [
            ("far", executor.dirs['vggt_far'], executor.dirs['ply_far']),
            ("near", executor.dirs['vggt_near'], executor.dirs['ply_near'])
        ]:
            # Look for GLB in AI-based 3D reconstruction output
            vggt_subdir = vggt_dir.parent / f"vggt_{label}"
            glb_files = list(vggt_subdir.glob("*.glb")) if vggt_subdir.exists() else []
            
            if glb_files:
                glb_file = glb_files[0]
                shutil.copy(glb_file, ply_dir / glb_file.name)
                executor.log(f"{label.upper()} GLB created: {glb_file.name}", "SUCCESS")
                executor.outputs[f'{label}_glb'] = str(ply_dir / glb_file.name)
                executor.add_visualization(f'vggt_{label}', str(ply_dir / glb_file.name), 'glb')
            else:
                executor.log(f"No {label.upper()} GLB found in AI-based 3D reconstruction output", "ERROR")
                success = False
    else:
        executor.log(f"AI-based 3D reconstruction failed: {err[:300]}", "ERROR")
        success = False
    
    executor.save_stage_output('02_vggt', {
        'far_glb': executor.outputs.get('far_glb'),
        'near_glb': executor.outputs.get('near_glb'),
        'success': success
    })
    
    return success


def run_stage_glb_to_ply(executor: PipelineExecutor) -> bool:
    """Stage 3: Convert GLB to PLY"""
    executor.current_stage = 3
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 3/10: GLB TO PLY CONVERSION", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['preprocess']}")
    
    success = True
    
    for label, ply_dir in [("far", executor.dirs['ply_far']), ("near", executor.dirs['ply_near'])]:
        glb_files = list(ply_dir.glob("*.glb"))
        
        if glb_files:
            glb_file = glb_files[0]
            executor.log(f"Converting {label.upper()}: {glb_file.name}", "PROGRESS")
            
            ply_file = ply_dir / f"{label}_pointcloud.ply"
            
            cmd = [
                "python", executor.base_dir / "prepro" / "glb_to_ply_cli.py",
                "--input_file", str(glb_file),
                "--output_file", str(ply_file)
            ]
            
            ok, _, err = executor.run_command(cmd, Config.ENVS['preprocess'])
            
            if ok and ply_file.exists():
                executor.log(f"{label.upper()} PLY created", "SUCCESS")
                executor.log(f"Saved to: {ply_file}", "FILE")
                executor.outputs[f'{label}_ply'] = str(ply_file)
                executor.add_visualization(f'ply_{label}', str(ply_file), 'mesh')
            else:
                executor.log(f"{label.upper()} GLB conversion failed: {err[:200]}", "ERROR")
                success = False
        else:
            executor.log(f"No {label.upper()} GLB file found", "ERROR")
            success = False
    
    executor.save_stage_output('03_glb_to_ply', {
        'far_ply': executor.outputs.get('far_ply'),
        'near_ply': executor.outputs.get('near_ply'),
        'success': success
    })
    
    return success


def run_stage_denoising(executor: PipelineExecutor) -> bool:
    """Stage 4: Denoise point clouds with ADAPTIVE parameters"""
    executor.current_stage = 4
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 4/10: POINT CLOUD DENOISING (ADAPTIVE)", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['denoise']}")
    
    # Create adaptive denoising script
    denoise_script = executor.work_dir / "run_adaptive_denoise.py"
    
    denoise_code = '''
import os
import sys
import json
import numpy as np
import trimesh
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN

def compute_nn_stats(points, k=20):
    """Compute nearest neighbor statistics for adaptive parameters."""
    pts = np.asarray(points)
    N = pts.shape[0]
    
    if N < 10:
        return {{'median_nn': 1e-3, 'mean_nn': 1e-3, 'std_nn': 1e-4}}
    
    k_eff = max(1, min(k, N - 1))
    
    # Add tiny jitter if all points identical
    if np.allclose(np.var(pts, axis=0), 0.0):
        pts = pts + np.random.RandomState(0).randn(*pts.shape) * 1e-9
    
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k_eff + 1)
    nn = dists[:, 1:]  # Exclude self-distance
    
    median_nn = float(np.median(nn))
    mean_nn = float(np.mean(nn))
    std_nn = float(np.std(nn))
    
    # Safety floor
    if median_nn <= 0.0:
        median_nn = 1e-6
    if mean_nn <= 0.0:
        mean_nn = median_nn
    
    return {{'median_nn': median_nn, 'mean_nn': mean_nn, 'std_nn': std_nn}}

def compute_adaptive_params(points, is_far=True, aggressive=True):
    """Compute adaptive denoising parameters based on point cloud analysis."""
    pts = np.asarray(points)
    n = pts.shape[0]
    
    if n < 10:
        return {{
            'sor_n': 8, 'sor_std': 1.5,
            'ror_n': 6, 'ror_radius': 0.01,
            'db_eps': 0.01, 'db_min': 6
        }}
    
    stats = compute_nn_stats(pts, k=20)
    med = stats['median_nn']
    std = stats['std_nn']
    
    print(f"Point cloud statistics:")
    print(f"  - Total points: {{n}}")
    print(f"  - Median NN distance: {{med:.6f}}")
    print(f"  - Mean NN distance: {{stats['mean_nn']:.6f}}")
    print(f"  - Std NN distance: {{std:.6f}}")
    
    # Adaptive heuristics based on point cloud size and density
    if is_far:
        # FAR mesh: More aggressive denoising
        if aggressive:
            sor_std_mult = {far_sor_std_mult}  # From config
            ror_radius_mult = {far_ror_radius_mult}
        else:
            sor_std_mult = 1.5
            ror_radius_mult = 2.0
            
        # Scale neighbors based on point count
        sor_n = int(max(15, min(80, np.sqrt(n) / 1.5)))
        ror_n = int(max(10, min(50, np.sqrt(n) / 8)))
        
        # Tighter radius for far (more noise)
        ror_radius = med * ror_radius_mult
        db_eps = med * 1.2
        
        # Lower std_ratio = more aggressive removal
        sor_std = sor_std_mult
        
    else:
        # NEAR mesh: Light denoising (less noise expected)
        sor_std_mult = {near_sor_std_mult}
        ror_radius_mult = {near_ror_radius_mult}
        
        sor_n = int(max(10, min(50, np.sqrt(n) / 2.5)))
        ror_n = int(max(6, min(30, np.sqrt(n) / 12)))
        
        # More lenient radius for near
        ror_radius = med * ror_radius_mult
        db_eps = med * 2.0
        sor_std = sor_std_mult
    
    # DBSCAN min points based on density
    db_min = int(max(8, min(40, np.sqrt(n) / 15)))
    
    # Safety clamps
    sor_n = max(5, sor_n)
    ror_n = max(3, ror_n)
    ror_radius = max(1e-6, ror_radius)
    db_eps = max(1e-6, db_eps)
    db_min = max(3, db_min)
    
    return {{
        'sor_n': sor_n,
        'sor_std': sor_std,
        'ror_n': ror_n,
        'ror_radius': ror_radius,
        'db_eps': db_eps,
        'db_min': db_min,
        'median_nn': med
    }}

def remove_statistical_outlier(points, k, std_ratio):
    """Remove statistical outliers using KDTree."""
    if len(points) < k + 1:
        return points, np.arange(len(points))
    
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k+1)
    mean_dists = np.mean(dists[:, 1:], axis=1)  # Exclude self
    mean_global = np.mean(mean_dists)
    std_global = np.std(mean_dists)
    threshold = mean_global + std_ratio * std_global
    inliers = mean_dists < threshold
    return points[inliers], np.where(inliers)[0]

def remove_radius_outlier(points, nb_points, radius):
    """Remove radius outliers using KDTree."""
    if len(points) < nb_points:
        return points, np.arange(len(points))
    
    tree = KDTree(points)
    counts = tree.query_radius(points, r=radius, count_only=True)
    inliers = counts >= nb_points
    return points[inliers], np.where(inliers)[0]

def denoise_point_cloud(input_path, output_path, is_far=True, aggressive=True,
                       use_sor=True, use_ror=True, use_dbscan=True):
    """Adaptive denoising pipeline."""
    print(f"\\nLoading: {{input_path}}")
    try:
        pcd = trimesh.load(input_path, force='pointcloud')
        if not hasattr(pcd, 'vertices') or len(pcd.vertices) == 0:
            print("ERROR: Failed to load point cloud")
            return False
        original_pts = np.array(pcd.vertices)
    except Exception as e:
        print(f"ERROR: Failed to load point cloud: {{e}}")
        return False
    
    original_count = len(original_pts)
    print(f"Original points: {{original_count}}")
    
    # Compute adaptive parameters
    params = compute_adaptive_params(original_pts, is_far=is_far, aggressive=aggressive)
    
    print(f"\\nAdaptive parameters for {{'FAR' if is_far else 'NEAR'}} mesh:")
    print(f"  SOR: n={{params['sor_n']}}, std={{params['sor_std']:.2f}}")
    print(f"  ROR: n={{params['ror_n']}}, radius={{params['ror_radius']:.6f}}")
    print(f"  DBSCAN: eps={{params['db_eps']:.6f}}, min={{params['db_min']}}")
    
    current_pts = original_pts.copy()
    
    # Stage 1: Statistical Outlier Removal (SOR)
    if use_sor:
        print(f"\\nApplying SOR (k={{params['sor_n']}}, std={{params['sor_std']:.2f}})...")
        current_pts, _ = remove_statistical_outlier(
            current_pts,
            k=int(params['sor_n']),
            std_ratio=float(params['sor_std'])
        )
        after_sor = len(current_pts)
        removed_sor = original_count - after_sor
        print(f"  After SOR: {{after_sor}} points (removed {{removed_sor}}, {{removed_sor/original_count*100:.1f}}%)")
    
    # Stage 2: Radius Outlier Removal (ROR)
    if use_ror:
        before_ror = len(current_pts)
        print(f"\\nApplying ROR (n={{params['ror_n']}}, radius={{params['ror_radius']:.6f}})...")
        current_pts, _ = remove_radius_outlier(
            current_pts,
            nb_points=int(params['ror_n']),
            radius=float(params['ror_radius'])
        )
        after_ror = len(current_pts)
        removed_ror = before_ror - after_ror
        print(f"  After ROR: {{after_ror}} points (removed {{removed_ror}}, {{removed_ror/before_ror*100:.1f}}%)")
    
    # Stage 3: DBSCAN clustering (keep largest cluster)
    if use_dbscan and is_far:  # Only apply DBSCAN to FAR mesh
        before_db = len(current_pts)
        print(f"\\nApplying DBSCAN (eps={{params['db_eps']:.6f}}, min={{params['db_min']}})...")
        
        clustering = DBSCAN(eps=float(params['db_eps']), min_samples=int(params['db_min']))
        labels = clustering.fit_predict(current_pts)
        
        # Keep only valid clusters (label >= 0)
        valid_mask = labels >= 0
        if np.any(valid_mask):
            # Find and keep largest cluster
            unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
            if len(unique_labels) > 0:
                largest_label = unique_labels[np.argmax(counts)]
                keep_mask = labels == largest_label
                current_pts = current_pts[keep_mask]
                
                after_db = len(current_pts)
                removed_db = before_db - after_db
                print(f"  Found {{len(unique_labels)}} clusters, kept largest ({{np.max(counts)}} points)")
                print(f"  After DBSCAN: {{after_db}} points (removed {{removed_db}}, {{removed_db/before_db*100:.1f}}%)")
        else:
            print("  DBSCAN: No valid clusters found, keeping all points")
    
    # Final statistics
    final_count = len(current_pts)
    total_removed = original_count - final_count
    
    print("\\n" + "="*50)
    print("DENOISING SUMMARY")
    print("="*50)
    print(f"Original: {{original_count}} points")
    print(f"Final:    {{final_count}} points")
    print(f"Removed:  {{total_removed}} ({{total_removed/original_count*100:.1f}}%)")
    print("="*50)
    
    # Save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    pcd_out = trimesh.PointCloud(vertices=current_pts)
    pcd_out.export(output_path)
    print(f"\\nSaved: {{output_path}}")
    
    # Save parameters to JSON
    params_file = output_path.replace('.ply', '_params.json')
    with open(params_file, 'w') as f:
        json.dump({{
            'is_far': is_far,
            'original_points': original_count,
            'final_points': final_count,
            'removed_points': total_removed,
            'removal_percentage': total_removed/original_count*100 if original_count > 0 else 0,
            'params': params
        }}, f, indent=2)
    
    return True

# Process FAR
far_ply = "{far_ply}"
far_output = "{far_output}"
if os.path.exists(far_ply):
    print("\\n" + "="*60)
    print("DENOISING FAR POINT CLOUD (Aggressive)")
    print("="*60)
    denoise_point_cloud(
        far_ply, far_output,
        is_far=True,
        aggressive={far_aggressive},
        use_sor={use_sor},
        use_ror={use_ror},
        use_dbscan={use_dbscan}
    )

# Process NEAR
near_ply = "{near_ply}"
near_output = "{near_output}"
near_enabled = {near_enabled}
if near_enabled and os.path.exists(near_ply):
    print("\\n" + "="*60)
    print("DENOISING NEAR POINT CLOUD (Light)")
    print("="*60)
    denoise_point_cloud(
        near_ply, near_output,
        is_far=False,
        aggressive=False,
        use_sor={use_sor},
        use_ror={use_ror},
        use_dbscan=False  # No DBSCAN for near mesh
    )
elif os.path.exists(near_ply):
    # Just copy if denoising disabled
    import shutil
    shutil.copy(near_ply, near_output)
    print(f"NEAR denoising disabled, copied: {{near_output}}")

print("\\nDenoising complete!")
'''.format(
        far_ply=executor.outputs.get('far_ply', ''),
        near_ply=executor.outputs.get('near_ply', ''),
        far_output=str(executor.dirs['denoised'] / "far_denoised.ply"),
        near_output=str(executor.dirs['denoised'] / "near_denoised.ply"),
        far_aggressive=str(Config.FAR_DENOISE_AGGRESSIVE),
        far_sor_std_mult=Config.FAR_SOR_STD_MULTIPLIER,
        far_ror_radius_mult=Config.FAR_ROR_RADIUS_MULTIPLIER,
        near_sor_std_mult=Config.NEAR_SOR_STD_MULTIPLIER,
        near_ror_radius_mult=Config.NEAR_ROR_RADIUS_MULTIPLIER,
        near_enabled=str(Config.NEAR_DENOISE_ENABLED),
        use_sor=str(Config.USE_SOR),
        use_ror=str(Config.USE_ROR),
        use_dbscan=str(Config.USE_DBSCAN)
    )
    
    with open(denoise_script, 'w') as f:
        f.write(denoise_code)
    
    far_ply = executor.outputs.get('far_ply')
    if not far_ply or not os.path.exists(far_ply):
        executor.log("Far PLY not found, skipping denoising", "WARNING")
        return False
    
    executor.log("Running ADAPTIVE denoising...", "PROGRESS")
    executor.log("Parameters will be computed from point cloud statistics", "INFO")
    executor.log(f"FAR: Aggressive={Config.FAR_DENOISE_AGGRESSIVE}, SOR_std_mult={Config.FAR_SOR_STD_MULTIPLIER}, ROR_radius_mult={Config.FAR_ROR_RADIUS_MULTIPLIER}")
    executor.log(f"NEAR: Enabled={Config.NEAR_DENOISE_ENABLED}, SOR_std_mult={Config.NEAR_SOR_STD_MULTIPLIER}")
    
    cmd = ["python", str(denoise_script)]
    ok, out, err = executor.run_command(cmd, Config.ENVS['denoise'], timeout=600)
    
    # Parse and log output
    if out:
        for line in out.split('\n'):
            if any(k in line for k in ['points', 'SOR:', 'ROR:', 'DBSCAN:', 'Median', 'removed', 'After', 'SUMMARY']):
                executor.log(f"  {line.strip()}")
    
    denoised_far = executor.dirs['denoised'] / "far_denoised.ply"
    denoised_near = executor.dirs['denoised'] / "near_denoised.ply"
    
    if ok and denoised_far.exists():
        executor.log("FAR point cloud denoised (adaptive)", "SUCCESS")
        executor.log(f"Saved to: {denoised_far}", "FILE")
        executor.outputs['denoised_far'] = str(denoised_far)
        executor.add_visualization('denoised_far', str(denoised_far), 'mesh')
        
        # Log the parameters used
        params_file = str(denoised_far).replace('.ply', '_params.json')
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
            executor.log(f"FAR removal: {params.get('removal_percentage', 0):.1f}%")
    else:
        executor.log(f"FAR denoising failed: {err[:200]}", "ERROR")
        return False
    
    if denoised_near.exists():
        executor.log("NEAR point cloud processed", "SUCCESS")
        executor.log(f"Saved to: {denoised_near}", "FILE")
        executor.outputs['denoised_near'] = str(denoised_near)
        executor.add_visualization('denoised_near', str(denoised_near), 'mesh')
        
        params_file = str(denoised_near).replace('.ply', '_params.json')
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
            executor.log(f"NEAR removal: {params.get('removal_percentage', 0):.1f}%")
    elif executor.outputs.get('near_ply'):
        # Copy near if denoising didn't process it
        near_ply = executor.outputs.get('near_ply')
        shutil.copy(near_ply, denoised_near)
        executor.outputs['denoised_near'] = str(denoised_near)
        executor.log(f"NEAR copied (no denoising): {denoised_near}", "FILE")
    
    executor.save_stage_output('04_denoising', {
        'denoised_far': executor.outputs.get('denoised_far'),
        'denoised_near': executor.outputs.get('denoised_near'),
        'success': True,
        'method': 'adaptive'
    })
    
    return True


def run_stage_mesh_reconstruction(executor: PipelineExecutor) -> bool:
    """Stage 5: Mesh reconstruction"""
    executor.current_stage = 5
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 5/10: MESH RECONSTRUCTION", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['mesh']}")
    
    success = True
    
    for label, denoised_key, mesh_dir in [
        ("FAR", 'denoised_far', executor.dirs['mesh_far']),
        ("NEAR", 'denoised_near', executor.dirs['mesh_near'])
    ]:
        denoised = executor.outputs.get(denoised_key)
        if denoised and os.path.exists(denoised):
            executor.log(f"Reconstructing {label} mesh...", "PROGRESS")
            executor.log(f"Poisson depth: {Config.POISSON_DEPTH}")
            
            mesh_file = mesh_dir / f"{label.lower()}_mesh.ply"
            
            cmd = [
                "python", executor.base_dir / "mesh" / "mesh_algo_cli.py",
                "--input", denoised,
                "--output", str(mesh_file),
                "--normal-neighbours", str(Config.NORMAL_NEIGHBORS),
                "--poisson-depth", str(Config.POISSON_DEPTH)
            ]
            
            ok, out, err = executor.run_command(cmd, Config.ENVS['mesh'])
            
            if ok and mesh_file.exists():
                executor.log(f"{label} mesh reconstructed", "SUCCESS")
                executor.log(f"Saved to: {mesh_file}", "FILE")
                executor.outputs[f'{label.lower()}_mesh'] = str(mesh_file)
                executor.add_visualization(f'mesh_{label.lower()}', str(mesh_file), 'mesh')
            else:
                executor.log(f"{label} mesh failed: {err[:200]}", "ERROR")
                success = False
    
    executor.save_stage_output('05_mesh', {
        'far_mesh': executor.outputs.get('far_mesh'),
        'near_mesh': executor.outputs.get('near_mesh'),
        'success': success
    })
    
    return success


def run_stage_post_process_near_mesh(executor: PipelineExecutor) -> bool:
    """
    Stage 5.5: Post-process NEAR mesh to remove Poisson reconstruction artifacts.
    
    Based on temp.py logic:
    - Loads PLY with quality attribute (from Poisson reconstruction)
    - Uses Otsu threshold (auto) to filter vertices
    - Keeps largest connected component
    - Removes degenerate/duplicated triangles and non-manifold edges
    """
    executor.current_stage = 5.5
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 5.5/8: POST-PROCESSING NEAR MESH (Poisson Artifact Removal)", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['mesh']}")
    
    near_mesh = executor.outputs.get('near_mesh')
    if not near_mesh or not os.path.exists(near_mesh):
        executor.log("NEAR mesh not found, skipping post-processing", "WARNING")
        return True  # Not a failure, just skip
    
    if not PLYFILE_AVAILABLE:
        executor.log("plyfile not available, skipping quality-based post-processing", "WARNING")
        executor.log("Install with: pip install plyfile", "INFO")
        return True
    
    if not SKIMAGE_AVAILABLE:
        executor.log("scikit-image not available, skipping Otsu threshold", "WARNING")
        executor.log("Install with: pip install scikit-image", "INFO")
        return True
    
    executor.log(f"Loading NEAR mesh: {near_mesh}", "PROGRESS")
    
    try:
        import trimesh
        
        # Try to load PLY with plyfile to check for quality attribute
        try:
            ply = PlyData.read(near_mesh)
            v = ply["vertex"].data
            
            if "quality" not in v.dtype.names:
                executor.log("PLY does not contain 'quality' attribute", "WARNING")
                executor.log("Computing quality from mesh curvature instead...", "INFO")
                # Fall back to curvature-based quality (from existing post_process_near_mesh)
                return _post_process_near_mesh_curvature(executor, near_mesh)
            
            # Extract vertices and quality
            vertices = np.vstack([v["x"], v["y"], v["z"]]).T
            quality = v["quality"].astype(np.float64)
            
            # Extract faces if available
            if "face" in ply:
                faces = np.vstack(ply["face"].data["vertex_indices"])
            else:
                faces = None
            
            executor.log(f"Loaded {len(vertices):,} vertices with quality attribute", "INFO")
            
        except Exception as e:
            executor.log(f"Could not read PLY with plyfile: {e}", "WARNING")
            executor.log("Falling back to trimesh and computing quality from curvature...", "INFO")
            return _post_process_near_mesh_curvature(executor, near_mesh)
        
        # Use Otsu threshold (auto mode from temp.py)
        executor.log("Computing Otsu threshold for quality filtering...", "PROGRESS")
        threshold = threshold_otsu(quality)
        executor.log(f"Otsu threshold: {threshold:.6f}", "INFO")
        
        # Filter vertices above threshold
        keep_v = quality >= threshold
        n_keep = np.sum(keep_v)
        n_remove = len(keep_v) - n_keep
        removal_pct = (n_remove / len(keep_v)) * 100
        
        executor.log(f"Keeping {n_keep:,} vertices ({100-removal_pct:.1f}%), removing {n_remove:,} ({removal_pct:.1f}%)", "INFO")
        
        if n_keep < 100:
            executor.log("Too few vertices remaining, skipping post-processing", "WARNING")
            return True
        
        # Reindex vertices
        old_to_new = -np.ones(len(keep_v), dtype=int)
        old_to_new[keep_v] = np.arange(n_keep)
        
        V2 = vertices[keep_v]
        
        # Filter faces (keep only faces where all vertices are kept)
        if faces is not None:
            face_mask = np.all(keep_v[faces], axis=1)
            F2 = old_to_new[faces[face_mask]]
        else:
            F2 = None
        
        # Build mesh with trimesh
        if F2 is not None:
            mesh = trimesh.Trimesh(vertices=V2, faces=F2)
        else:
            mesh = trimesh.PointCloud(vertices=V2)
        
        # Keep largest connected component (from temp.py)
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            executor.log("Keeping largest connected component...", "PROGRESS")
            components = mesh.split(only_watertight=False)
            if len(components) > 0:
                # Keep largest component
                largest = max(components, key=lambda m: len(m.vertices))
                mesh = largest
                executor.log(f"Kept largest component with {len(mesh.vertices):,} vertices", "INFO")
        
        # Final cleanup (from temp.py)
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            executor.log("Removing degenerate/duplicated triangles and non-manifold edges...", "PROGRESS")
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            # Trimesh doesn't have direct non-manifold edge removal, but we can fix it
            mesh.fix_normals()
        
        # Save cleaned mesh
        output_dir = executor.dirs['mesh_near']
        output_path = output_dir / "near_mesh_postprocessed.ply"
        os.makedirs(output_dir, exist_ok=True)
        
        mesh.export(str(output_path))
        
        executor.log("NEAR mesh post-processed successfully", "SUCCESS")
        executor.log(f"Saved to: {output_path}", "FILE")
        executor.outputs['near_mesh'] = str(output_path)  # Update to use post-processed mesh
        
        executor.save_stage_output('05.5_postprocess_near', {
            'original_mesh': near_mesh,
            'postprocessed_mesh': str(output_path),
            'threshold': float(threshold),
            'vertices_kept': int(n_keep),
            'vertices_removed': int(n_remove),
            'removal_percentage': float(removal_pct),
            'success': True
        })
        
        return True
        
    except Exception as e:
        executor.log(f"Post-processing failed: {e}", "ERROR")
        executor.log(traceback.format_exc(), "ERROR")
        return False


def _post_process_near_mesh_curvature(executor: PipelineExecutor, mesh_path: str) -> bool:
    """
    Fallback: Post-process NEAR mesh using curvature-based quality (when quality attribute not available).
    This uses the existing post_process_near_mesh logic.
    """
    try:
        import trimesh
        from scipy.spatial import cKDTree
        
        executor.log("Loading mesh with trimesh...", "PROGRESS")
        mesh = trimesh.load(mesh_path, force='mesh')
        
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            executor.log("Empty mesh, skipping post-processing", "WARNING")
            return True
        
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            executor.log("Mesh has no triangles, skipping post-processing", "WARNING")
            return True
        
        vertices = np.array(mesh.vertices)
        n_vertices = len(vertices)
        executor.log(f"Computing curvature-based quality for {n_vertices:,} vertices...", "PROGRESS")
        
        # Compute curvature-based quality
        # Trimesh automatically computes vertex normals when needed
        
        k = min(20, n_vertices // 10)
        if k < 3:
            k = 3
        
        tree = cKDTree(vertices)
        quality = np.zeros(n_vertices)
        
        for i in range(n_vertices):
            distances, indices = tree.query(vertices[i], k=min(k+1, n_vertices))
            if len(indices) < 2:
                quality[i] = 1.0
                continue
            
            neighbor_indices = indices[1:] if indices[0] == i else indices[:k]
            neighbor_pts = vertices[neighbor_indices]
            center = vertices[i]
            centered = neighbor_pts - center
            
            if len(centered) >= 3:
                cov = np.cov(centered.T)
                eigenvals = np.linalg.eigvals(cov)
                eigenvals = np.sort(eigenvals)[::-1]
                if eigenvals[0] > 1e-10:
                    variation = eigenvals[-1] / np.sum(eigenvals)
                    quality[i] = 1.0 - min(variation * 10, 1.0)
                else:
                    quality[i] = 1.0
            else:
                quality[i] = 1.0
        
        # Use Otsu threshold
        if SKIMAGE_AVAILABLE:
            threshold = threshold_otsu(quality)
        else:
            # Fallback: use percentile
            threshold = np.percentile(quality, 10)
        
        executor.log(f"Quality threshold: {threshold:.6f}", "INFO")
        
        keep_mask = quality >= threshold
        n_keep = np.sum(keep_mask)
        n_remove = n_vertices - n_keep
        
        if n_keep < 100:
            executor.log("Too few vertices remaining, skipping", "WARNING")
            return True
        
        # Filter mesh
        keep_indices = np.where(keep_mask)[0]
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        
        filtered_vertices = vertices[keep_mask]
        triangles = np.array(mesh.faces)
        keep_set = set(keep_indices)
        valid_triangles = np.array([all(v in keep_set for v in tri) for tri in triangles])
        filtered_triangles = triangles[valid_triangles]
        remapped_triangles = np.array([[vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]] 
                                       for tri in filtered_triangles])
        
        cleaned_mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=remapped_triangles)
        
        # Keep largest connected component
        if len(cleaned_mesh.faces) > 0:
            components = cleaned_mesh.split(only_watertight=False)
            if len(components) > 0:
                largest = max(components, key=lambda m: len(m.vertices))
                cleaned_mesh = largest
        
        # Cleanup
        cleaned_mesh.remove_duplicate_faces()
        cleaned_mesh.remove_unreferenced_vertices()
        cleaned_mesh.fix_normals()
        
        output_path = executor.dirs['mesh_near'] / "near_mesh_postprocessed.ply"
        cleaned_mesh.export(str(output_path))
        
        executor.log("NEAR mesh post-processed (curvature-based)", "SUCCESS")
        executor.log(f"Saved to: {output_path}", "FILE")
        executor.outputs['near_mesh'] = str(output_path)
        
        return True
        
    except Exception as e:
        executor.log(f"Curvature-based post-processing failed: {e}", "ERROR")
        return False


def run_stage_ground_alignment(executor: PipelineExecutor) -> bool:
    """Stage 6: Ground plane estimation and Y-axis alignment"""
    executor.current_stage = 6
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 6/10: GROUND PLANE ESTIMATION & ALIGNMENT", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['ground']}")
    
    far_mesh = executor.outputs.get('far_mesh')
    if not far_mesh or not os.path.exists(far_mesh):
        executor.log("Far mesh not found", "ERROR")
        return False
    
    executor.log("Estimating ground plane and aligning to Y-axis...", "PROGRESS")
    executor.log(f"Target axis: {Config.GROUND_ALIGN_TARGET} (feet at max Y)", "INFO")
    
    # Create a script that estimates ground plane AND aligns to target axis
    ground_script = executor.work_dir / "run_ground_align.py"
    
    ground_code = '''#!/usr/bin/env python3
import trimesh
import numpy as np
import json
import os

def voxel_downsample(points, voxel_size):
    """Voxel downsampling using numpy."""
    if len(points) == 0:
        return points
    
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    
    # Compute voxel indices
    voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)
    
    # Use dictionary to keep first point in each voxel
    voxel_dict = {{}}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = i
    
    return points[list(voxel_dict.values())]

def ransac_plane_segmentation(points, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """RANSAC plane segmentation."""
    best_inliers = []
    best_model = None
    best_score = 0
    
    n_points = len(points)
    if n_points < ransac_n:
        return None, []
    
    for _ in range(num_iterations):
        # Randomly sample ransac_n points
        sample_indices = np.random.choice(n_points, ransac_n, replace=False)
        sample_points = points[sample_indices]
        
        # Fit plane through these points
        if ransac_n == 3:
            # Use three points to define plane
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -np.dot(normal, sample_points[0])
        else:
            # Use SVD for more points
            centered = sample_points - sample_points.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]
            d = -np.dot(normal, sample_points.mean(axis=0))
        
        # Count inliers
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.where(distances < distance_threshold)[0]
        
        if len(inliers) > best_score:
            best_score = len(inliers)
            best_inliers = inliers
            best_model = [normal[0], normal[1], normal[2], d]
    
    return best_model, best_inliers

def rotation_matrix_from_axis_angle(axis, angle):
    """Compute rotation matrix from axis-angle representation."""
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]], 
                   [axis[2], 0, -axis[0]], 
                   [-axis[1], axis[0], 0]])
    R = cos_a * np.eye(3) + sin_a * K + (1 - cos_a) * np.outer(axis, axis)
    return R

mesh_path = "{mesh_path}"
output_dir = "{output_dir}"
target_axis = {target_axis}

print(f"Loading mesh: {{mesh_path}}")
mesh = trimesh.load(mesh_path, force='mesh')
if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
    raise ValueError("Empty mesh")

print(f"Mesh has {{len(mesh.vertices)}} vertices")

# Get vertices for RANSAC
points = np.array(mesh.vertices)

# Downsample
points_down = voxel_downsample(points, voxel_size=0.01)
if len(points_down) < 100:
    points_down = points
print(f"Processing {{len(points_down)}} points")

# Find ground plane using RANSAC
plane_model, inliers = ransac_plane_segmentation(
    points_down,
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000
)

if plane_model is None:
    raise ValueError("Failed to find plane")

[a, b, c, d] = plane_model
print(f"Plane equation: {{a:.4f}}x + {{b:.4f}}y + {{c:.4f}}z + {{d:.4f}} = 0")

# Get non-ground points (the object)
object_mask = np.ones(len(points_down), dtype=bool)
object_mask[inliers] = False
object_points = points_down[object_mask]

# Correct normal orientation (should point towards object)
normal = np.array([a, b, c])
normal = normal / np.linalg.norm(normal)
if len(object_points) > 0:
    dists = np.dot(object_points, normal) + d
    if np.sum(dists > 0) < len(dists) / 2:
        print("Flipping normal orientation...")
        normal = -normal
        plane_model = [-a, -b, -c, -d]
        d = plane_model[3]

# Now align to target axis
target = np.array(target_axis, dtype=float)
target = target / np.linalg.norm(target)

# Calculate rotation matrix (from normal to target)
v = np.cross(normal, target)
c_val = np.dot(normal, target)

if np.linalg.norm(v) < 1e-6:
    if c_val > 0:
        R = np.eye(3)
    else:
        # Opposite, rotate 180 deg
        if abs(normal[2]) < 0.9:
            axis = np.cross(normal, [0, 0, 1])
        else:
            axis = np.cross(normal, [1, 0, 0])
        axis = axis / np.linalg.norm(axis)
        R = rotation_matrix_from_axis_angle(axis, np.pi)
else:
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c_val) / (s**2))

print(f"Rotation to align normal {{normal}} to target {{target}}")
print(f"Rotation matrix:\\n{{R}}")

# Apply rotation
center = mesh.centroid
transform = np.eye(4)
transform[:3, :3] = R
transform[:3, 3] = center - R @ center
mesh.apply_transform(transform)

# Update plane parameters after rotation
dist_to_center = np.dot(center, normal) + d
P = center - dist_to_center * normal  # Point on plane
P_new = center - dist_to_center * target  # After rotation
d_new = -np.dot(P_new, target)
new_plane_model = [target[0], target[1], target[2], d_new]

# Save results
os.makedirs(output_dir, exist_ok=True)

# Save aligned mesh
aligned_path = os.path.join(output_dir, "far_ground_aligned.ply")
mesh.export(aligned_path)
print(f"Saved aligned mesh: {{aligned_path}}")

# Save ground plane JSON
json_data = {{
    "mesh_file": mesh_path,
    "original_plane_equation": {{"a": float(a), "b": float(b), "c": float(c), "d": float(d)}},
    "aligned_plane_equation": {{
        "a": float(new_plane_model[0]),
        "b": float(new_plane_model[1]),
        "c": float(new_plane_model[2]),
        "d": float(new_plane_model[3])
    }},
    "target_axis": target_axis,
    "rotation_matrix": R.tolist(),
    "method": "ransac_with_alignment"
}}

json_path = os.path.join(output_dir, "ground_plane.json")
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"Saved plane info: {{json_path}}")

# Also save rotation matrix for later use
np.save(os.path.join(output_dir, "ground_rotation.npy"), R)

print("Ground alignment complete!")
'''.format(
        mesh_path=far_mesh,
        output_dir=str(executor.dirs['ground_aligned']),
        target_axis=Config.GROUND_ALIGN_TARGET
    )
    
    with open(ground_script, 'w') as f:
        f.write(ground_code)
    
    cmd = ["python", str(ground_script)]
    ok, out, err = executor.run_command(cmd, Config.ENVS['ground'], timeout=300)
    
    aligned_mesh = executor.dirs['ground_aligned'] / "far_ground_aligned.ply"
    ground_json = executor.dirs['ground_aligned'] / "ground_plane.json"
    
    if ok and aligned_mesh.exists():
        executor.log("Ground plane estimated and aligned", "SUCCESS")
        executor.outputs['ground_json'] = str(ground_json)
        executor.outputs['y_aligned_far'] = str(aligned_mesh)
        executor.add_visualization('ground_aligned', str(aligned_mesh), 'mesh')
        executor.log(f"Aligned mesh: {aligned_mesh}", "FILE")
        
        # Log rotation info
        if ground_json.exists():
            with open(ground_json, 'r') as f:
                gnd_data = json.load(f)
            orig = gnd_data.get('original_plane_equation', {})
            executor.log(f"Original plane normal: [{orig.get('a',0):.3f}, {orig.get('b',0):.3f}, {orig.get('c',0):.3f}]")
            executor.log(f"Aligned to: {Config.GROUND_ALIGN_TARGET}")
        
        return True
    else:
        executor.log(f"Ground alignment failed: {err[:300]}", "ERROR")
        # Fallback: copy original mesh
        shutil.copy(far_mesh, executor.dirs['ground_aligned'] / "far_ground_aligned.ply")
        executor.outputs['y_aligned_far'] = str(executor.dirs['ground_aligned'] / "far_ground_aligned.ply")
        return False


def run_stage_xaxis_alignment(executor: PipelineExecutor) -> bool:
    """Stage 7: X-axis alignment - detect feet and align X-axis to line connecting them"""
    executor.current_stage = 7
    executor.log("=" * 50, "STAGE")
    # X-axis alignment removed - now integrated into manual alignment
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['xaxis']}")
    
    y_aligned = executor.outputs.get('y_aligned_far') or executor.outputs.get('far_mesh')
    if not y_aligned or not os.path.exists(y_aligned):
        executor.log("No mesh available for X-axis alignment", "ERROR")
        return False
    
    executor.log("Detecting feet using DBSCAN clustering...", "PROGRESS")
    
    # Create a simple, robust foot detection script
    xaxis_script = executor.work_dir / "run_xaxis.py"
    
    xaxis_code = '''#!/usr/bin/env python3
"""
X-Axis Alignment Script - Robust Foot Detection
================================================
After ground alignment (Y-axis aligned to gravity, feet at max Y):
1. Slice the foot region (bottom 8% of height)
2. Use DBSCAN to cluster points in X-Z plane
3. Find two largest clusters that are well-separated (the two feet)
4. Rotate around Y-axis to align foot-to-foot vector with X-axis
"""
import os
import json
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN

mesh_path = "{{mesh_path}}"
output_dir = "{{output_dir}}"

os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("X-AXIS ALIGNMENT - FOOT DETECTION")
print("=" * 60)

def load_mesh_or_cloud(path):
    """Load mesh or point cloud"""
    try:
        mesh = trimesh.load(path, force='mesh')
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            return np.array(mesh.vertices), mesh
    except:
        pass
    
    try:
        pcd = trimesh.load(path, force='pointcloud')
        if hasattr(pcd, 'vertices') and len(pcd.vertices) > 0:
            return np.array(pcd.vertices), None
    except:
        pass
    
    raise ValueError(f"Cannot load: {{path}}")

def find_foot_centers(points, foot_height_frac=0.10):
    """
    Find foot centers by clustering points in the foot region.
    After ground alignment: feet are at MAX Y (ground normal is -Y).
    """
    min_b = points.min(axis=0)
    max_b = points.max(axis=0)
    H = max_b[1] - min_b[1]  # Height (Y-axis)
    W = max(max_b[0] - min_b[0], max_b[2] - min_b[2])  # Width
    
    # Feet are at MAX Y after ground alignment
    y_feet = max_b[1]
    y_cutoff = y_feet - (foot_height_frac * H)
    
    print(f"Mesh bounds: Y=[{{min_b[1]:.3f}}, {{max_b[1]:.3f}}]")
    print(f"Height H: {{H:.3f}}, Width W: {{W:.3f}}")
    print(f"Foot region: Y > {{y_cutoff:.3f}} (top {{foot_height_frac*100:.0f}}% of height)")
    
    # Extract foot region
    foot_mask = points[:, 1] > y_cutoff
    foot_pts = points[foot_mask]
    print(f"Points in foot region: {{len(foot_pts)}}")
    
    if len(foot_pts) < 50:
        # Expand region if too few points
        y_cutoff = y_feet - (0.15 * H)
        foot_mask = points[:, 1] > y_cutoff
        foot_pts = points[foot_mask]
        print(f"Expanded region: {{len(foot_pts)}} points")
    
    if len(foot_pts) < 20:
        print("ERROR: Not enough points in foot region!")
        return None, None
    
    # Cluster in X-Z plane
    xz_pts = foot_pts[:, [0, 2]]
    
    # DBSCAN with adaptive eps based on mesh width
    eps = max(0.02 * W, 0.01)  # 2% of width or 1cm minimum
    print(f"DBSCAN eps: {{eps:.4f}}")
    
    clustering = DBSCAN(eps=eps, min_samples=10).fit(xz_pts)
    labels = clustering.labels_
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    print(f"Found {{len(unique_labels)}} clusters")
    
    if len(unique_labels) < 2:
        # Try with larger epsilon
        eps = 0.05 * W
        clustering = DBSCAN(eps=eps, min_samples=5).fit(xz_pts)
        labels = clustering.labels_
        unique_labels = [l for l in np.unique(labels) if l >= 0]
        print(f"Retry with eps={{eps:.4f}}: {{len(unique_labels)}} clusters")
    
    # Collect cluster info
    clusters = []
    for lab in unique_labels:
        mask = labels == lab
        cluster_pts = foot_pts[mask]
        center = cluster_pts.mean(axis=0)
        size = len(cluster_pts)
        clusters.append({{'label': lab, 'center': center, 'size': size}})
    
    # Sort by size
    clusters.sort(key=lambda c: -c['size'])
    
    # Select two largest clusters with good separation
    min_sep = 0.1 * W  # At least 10% of width apart
    selected = []
    
    for c in clusters:
        if len(selected) >= 2:
            break
        sep_ok = True
        for s in selected:
            dist_xz = np.sqrt((c['center'][0] - s['center'][0])**2 + 
                             (c['center'][2] - s['center'][2])**2)
            if dist_xz < min_sep:
                sep_ok = False
                break
        if sep_ok:
            selected.append(c)
    
    if len(selected) < 2:
        print("Could not find 2 separated foot clusters, using PCA fallback")
        # Fallback: use extremes of X-Z projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(xz_pts)
        proj = (xz_pts - pca.mean_).dot(pca.components_[0])
        p1_idx = np.argmin(proj)
        p2_idx = np.argmax(proj)
        
        p1 = foot_pts[p1_idx].copy()
        p2 = foot_pts[p2_idx].copy()
        p1[1] = y_feet  # Project to ground
        p2[1] = y_feet
        return p1, p2
    
    # Get foot centers (projected to ground level)
    foot1 = selected[0]['center'].copy()
    foot2 = selected[1]['center'].copy()
    foot1[1] = y_feet
    foot2[1] = y_feet
    
    print(f"Foot 1: [{{foot1[0]:.4f}}, {{foot1[1]:.4f}}, {{foot1[2]:.4f}}]")
    print(f"Foot 2: [{{foot2[0]:.4f}}, {{foot2[1]:.4f}}, {{foot2[2]:.4f}}]")
    
    return foot1, foot2

def compute_yaxis_rotation(foot1, foot2):
    """Compute rotation around Y-axis to align foot-to-foot vector with X-axis"""
    direction = foot2 - foot1
    direction[1] = 0  # Project to X-Z plane
    
    length = np.linalg.norm(direction)
    if length < 1e-6:
        print("Warning: Feet are at same position!")
        return np.eye(3), 0
    
    direction = direction / length
    print(f"Foot-to-foot direction: [{{direction[0]:.4f}}, 0, {{direction[2]:.4f}}]")
    print(f"Foot separation: {{length:.4f}}")
    
    # Angle to rotate (in X-Z plane, around Y)
    angle = np.arctan2(direction[2], direction[0])
    print(f"Rotation angle: {{np.degrees(angle):.1f}} degrees")
    
    # Rotation matrix around Y-axis
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    return R, np.degrees(angle)

# Main execution
try:
    print(f"\\nLoading: {{mesh_path}}")
    points, mesh = load_mesh_or_cloud(mesh_path)
    print(f"Loaded {{len(points)}} points")
    
    # Find feet
    print("\\nFinding foot positions...")
    foot1, foot2 = find_foot_centers(points)
    
    if foot1 is None or foot2 is None:
        print("\\nERROR: Could not detect feet, copying mesh as-is")
        import shutil
        shutil.copy(mesh_path, os.path.join(output_dir, "aligned_mesh.ply"))
        exit(0)
    
    # Compute rotation
    print("\\nComputing rotation...")
    R, angle = compute_yaxis_rotation(foot1, foot2)
    
    # Apply rotation around mesh center
    center = points.mean(axis=0)
    
    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = center - R @ center
    
    if mesh is not None:
        mesh.apply_transform(transform)
        output_path = os.path.join(output_dir, "aligned_mesh.ply")
        mesh.export(output_path)
    else:
        # Apply rotation to points
        points_centered = points - center
        points_rotated = (R @ points_centered.T).T + center
        pcd = trimesh.PointCloud(vertices=points_rotated)
        output_path = os.path.join(output_dir, "rotated_full.ply")
        pcd.export(output_path)
    
    print(f"\\nSaved: {{output_path}}")
    
    # Save transformation info
    transform_info = {{
        'rotation_matrix': R.tolist(),
        'center': center.tolist(),
        'foot1': foot1.tolist(),
        'foot2': foot2.tolist(),
        'rotation_angle_degrees': angle,
        'foot_separation': float(np.linalg.norm(foot2 - foot1))
    }}
    with open(os.path.join(output_dir, 'xaxis_transform.json'), 'w') as f:
        json.dump(transform_info, f, indent=2)
    
    print("\\n" + "=" * 60)
    print("X-AXIS ALIGNMENT COMPLETE!")
    print("=" * 60)

except Exception as e:
    import traceback
    print(f"\\nERROR: {{e}}")
    traceback.print_exc()
    
    # Fallback: copy input as-is
    print("\\nFallback: copying input mesh as-is...")
    import shutil
    shutil.copy(mesh_path, os.path.join(output_dir, "aligned_mesh.ply"))
'''.format(
        mesh_path=y_aligned,
        output_dir=str(executor.dirs['ground_aligned'])
    )
    
    with open(xaxis_script, 'w') as f:
        f.write(xaxis_code)
    
    cmd = ["python", str(xaxis_script)]
    ok, out, err = executor.run_command(cmd, Config.ENVS['xaxis'], timeout=300)
    
    # Log output
    if out:
        for line in out.split('\n'):
            if line.strip() and any(k in line.lower() for k in ['foot', 'cluster', 'rotation', 'align', 'height', 'region', 'saved']):
                executor.log(f"  {line.strip()[:100]}")
    
    # Check for output
    aligned_mesh = executor.dirs['ground_aligned'] / "aligned_mesh.ply"
    rotated_ply = executor.dirs['ground_aligned'] / "rotated_full.ply"
    
    output_mesh = None
    if aligned_mesh.exists():
        output_mesh = aligned_mesh
    elif rotated_ply.exists():
        output_mesh = rotated_ply
    
    if output_mesh:
        executor.outputs['aligned_far'] = str(output_mesh)
        executor.add_visualization('xaxis_aligned', str(output_mesh), 'mesh')
        executor.log("X-axis alignment complete", "SUCCESS")
        executor.log(f"Saved to: {output_mesh}", "FILE")
        
        # Log transformation info
        transform_json = executor.dirs['ground_aligned'] / "xaxis_transform.json"
        if transform_json.exists():
            with open(transform_json, 'r') as f:
                t_data = json.load(f)
            executor.log(f"Foot separation: {t_data.get('foot_separation', 0):.3f}m")
            executor.log(f"Rotation: {t_data.get('rotation_angle_degrees', 0):.1f}°")
        
        return True
    else:
        executor.log("X-axis alignment failed, using Y-aligned mesh", "WARNING")
        if err:
            executor.log(f"Error: {err[:200]}", "WARNING")
        xaxis_fallback = executor.dirs['ground_aligned'] / "far_xaxis_aligned.ply"
        shutil.copy(y_aligned, xaxis_fallback)
        executor.outputs['aligned_far'] = str(xaxis_fallback)
        return True


def run_stage_manual_alignment(executor: PipelineExecutor) -> Tuple[bool, str]:
    """Stage 7: Manual alignment - uses web-based point selection in Gradio"""
    executor.current_stage = 7
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 7/8: MANUAL ALIGNMENT (with X-axis alignment)", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['alignment']}")
    
    aligned_far = executor.outputs.get('aligned_far') or executor.outputs.get('y_aligned_far') or executor.outputs.get('far_mesh')
    near_mesh = executor.outputs.get('near_mesh')
    
    if not aligned_far or not near_mesh:
        executor.log("Missing meshes for manual alignment", "ERROR")
        return False, "Missing meshes"
    
    # Set up alignment state
    expected_output = executor.dirs['manual_aligned'] / 'near_aligned.ply'
    
    global alignment_state
    alignment_state['source_path'] = near_mesh
    alignment_state['target_path'] = aligned_far
    alignment_state['source_points'] = []
    alignment_state['target_points'] = []
    alignment_state['selecting_on'] = 'source'
    alignment_state['transformation'] = None
    alignment_state['output_path'] = str(expected_output)
    
    executor.log("MANUAL ALIGNMENT READY", "WARNING")
    executor.log("", "INFO")
    executor.log("Use the ALIGNMENT TAB below to select correspondence points.", "INFO")
    executor.log("", "INFO")
    executor.log("Instructions:", "INFO")
    executor.log("  1. Switch to the 'Manual Alignment' tab", "INFO")
    executor.log("  2. Click 'Load Meshes' to visualize both meshes", "INFO")
    executor.log("  3. Enter coordinates for 3-4 corresponding points on each mesh", "INFO")
    executor.log("     - Point 1: Top of head (vertex/crown)", "INFO")
    executor.log("     - Point 2: Left shoulder", "INFO")
    executor.log("     - Point 3: Right shoulder", "INFO")
    executor.log("     - Point 4: Hip center (optional)", "INFO")
    executor.log("  4. Click 'Compute & Apply Alignment'", "INFO")
    executor.log("  5. Then click 'Continue After Alignment'", "INFO")
    executor.log("", "INFO")
    executor.log(f"Source mesh (to align): {os.path.basename(near_mesh)}", "INFO")
    executor.log(f"Target mesh (reference): {os.path.basename(aligned_far)}", "INFO")
    executor.log(f"Output will be saved to: {expected_output}", "INFO")
    
    executor.save_stage_output('07_manual_alignment', {
        'source_mesh': near_mesh,
        'target_mesh': aligned_far,
        'expected_output': str(expected_output),
        'status': 'waiting_for_user'
    })
    
    return True, str(expected_output)


def check_manual_alignment_complete(executor: PipelineExecutor, expected_path: str) -> bool:
    """Check if manual alignment has been completed"""
    if os.path.exists(expected_path):
        executor.outputs['aligned_near'] = expected_path
        executor.add_visualization('manual_aligned', expected_path, 'mesh')
        executor.log("Manual alignment detected!", "SUCCESS")
        executor.log(f"Found: {expected_path}", "FILE")
        return True
    return False


def run_stage_aix(executor: PipelineExecutor) -> bool:
    """Stage 9-10: AIX estimation (Hip, Spine, Hump)"""
    executor.current_stage = 9
    executor.log("=" * 50, "STAGE")
    executor.log("STAGE 9-10/10: AIX ESTIMATION", "STAGE")
    executor.log("=" * 50, "STAGE")
    executor.log(f"Environment: {Config.ENVS['aix']}")
    
    # Use aligned near mesh, or fall back to near mesh, or far mesh
    input_mesh = executor.outputs.get('aligned_near') or executor.outputs.get('near_mesh')
    if not input_mesh:
        input_mesh = executor.outputs.get('aligned_far') or executor.outputs.get('far_mesh')
    
    if not input_mesh or not os.path.exists(input_mesh):
        executor.log("No mesh available for AIX estimation", "ERROR")
        return False
    
    executor.log(f"Using mesh: {os.path.basename(input_mesh)}")
    success = True
    
    # Create separate subfolders for each AIX
    hip_aix_dir = executor.dirs['aix'] / "Hip_aix"
    spine_aix_dir = executor.dirs['aix'] / "Spine_aix"
    hump_aix_dir = executor.dirs['aix'] / "Hump_aix"
    
    hip_aix_dir.mkdir(parents=True, exist_ok=True)
    spine_aix_dir.mkdir(parents=True, exist_ok=True)
    hump_aix_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_dir = Path(input_mesh).parent
    mesh_stem = Path(input_mesh).stem
    
    # ===== Hip AIX =====
    executor.log("Running Hip AIX...", "PROGRESS")
    
    cmd = [
        "python", executor.base_dir / "aix" / "Hip_aix" / "hip_aix_bend.py",
        "--mesh", input_mesh,
        "--slice-frac", str(Config.SLICE_FRAC),
        "--no-vis"
    ]
    
    ok, out, err = executor.run_command(cmd, Config.ENVS['aix'])
    
    if ok:
        # Parse and log results
        for line in out.split('\n'):
            if any(k in line for k in ['Δx', 'Hip center', 'Neck center', 'Angle', 'offset']):
                executor.log(f"  {line.strip()}")
        
        # Copy results to Hip_aix subfolder and clean up from mesh directory
        hip_json = mesh_dir / f"{mesh_stem}_hip_stats.json"
        if hip_json.exists():
            dest_json = hip_aix_dir / hip_json.name
            shutil.copy(hip_json, dest_json)
            os.remove(hip_json)  # Remove from mesh directory
            executor.outputs['hip_json'] = str(dest_json)
            executor.log(f"Saved to: {dest_json}", "FILE")
        
        # Move visualization PNGs
        for png_file in mesh_dir.glob(f"{mesh_stem}_hip_*.png"):
            dest_png = hip_aix_dir / png_file.name
            shutil.copy(png_file, dest_png)
            os.remove(png_file)
            executor.log(f"Saved vis: {dest_png}", "FILE")
        
        executor.log("Hip AIX completed", "SUCCESS")
    else:
        executor.log(f"Hip AIX failed: {err[:200]}", "ERROR")
        success = False
    
    # ===== Spine AIX =====
    executor.current_stage = 10
    executor.log("Running Spine AIX...", "PROGRESS")
    
    cmd = [
        "python", executor.base_dir / "aix" / "Spine_aix" / "spine_aix.py",
        input_mesh,
        "--no-vis"
    ]
    
    ok, out, err = executor.run_command(cmd, Config.ENVS['aix'])
    
    if ok:
        executor.log("Spine AIX completed", "SUCCESS")
        
        # Copy results to Spine_aix subfolder and clean up from mesh directory
        copied_files = []
        midline_json_candidates = []
        
        for pattern in ['*mid_sagittal*.json', '*midline*.json', '*spine*.json', '*mid_sagittal*.png']:
            for f in mesh_dir.glob(pattern):
                dest_file = spine_aix_dir / f.name
                shutil.copy(f, dest_file)
                os.remove(f)  # Remove from mesh directory
                copied_files.append(dest_file)
                if f.suffix == '.json' and ('mid_sagittal' in f.name or 'midline' in f.name):
                    midline_json_candidates.append(dest_file)
        
        # Prefer optimized midline JSON if available, otherwise use any midline JSON
        if midline_json_candidates:
            # Sort to prefer 'symmetry_optimized' over 'slice_midpoint'
            midline_json_candidates.sort(key=lambda p: 'optimized' not in str(p))
            executor.outputs['midline_json'] = str(midline_json_candidates[0])
            executor.log(f"Using midline: {midline_json_candidates[0].name}", "INFO")
        
        if copied_files:
            executor.log(f"Saved {len(copied_files)} file(s) to Spine_aix folder", "FILE")
    else:
        executor.log(f"Spine AIX failed: {err[:200]}", "ERROR")
        success = False
    
    # ===== Hump AIX =====
    midline_json = executor.outputs.get('midline_json')
    if midline_json and os.path.exists(midline_json):
        executor.log("Running Hump AIX...", "PROGRESS")
        
        out_prefix = str(hump_aix_dir / "hump")
        
        cmd = [
            "python", executor.base_dir / "aix" / "Hump_aix" / "hump_aix.py",
            "--mesh_path", input_mesh,
            "--midline_json", midline_json,
            "--out_prefix", out_prefix
        ]
        
        ok, out, err = executor.run_command(cmd, Config.ENVS['aix'])
        
        if ok:
            for line in out.split('\n'):
                if any(k in line for k in ['Volume', 'angle', 'Angle', 'asymmetry', 'hump']):
                    executor.log(f"  {line.strip()}")
            
            # Hump AIX outputs are already saved to the specified out_prefix directory
            # Verify files exist
            hump_files = list(hump_aix_dir.glob("hump_*"))
            if hump_files:
                executor.log(f"Saved {len(hump_files)} file(s) to Hump_aix folder", "FILE")
            
            executor.log("Hump AIX completed", "SUCCESS")
        else:
            executor.log("Hump AIX failed (non-critical)", "WARNING")
    else:
        executor.log("Skipping Hump AIX (no midline data)", "WARNING")
    
    # ===== Final combined SSI score =====
    try:
        executor.log("Computing final SSI (Scoliosis Severity Index) score...", "PROGRESS")
        ssi_path = compute_ssi_score(executor, input_mesh)
        if ssi_path:
            executor.outputs['ssi_json'] = ssi_path
            executor.log(f"SSI_score JSON saved: {ssi_path}", "SUCCESS")
        else:
            executor.log("SSI_score JSON was not created (see earlier errors).", "WARNING")
    except Exception as e:
        executor.log(f"SSI_score computation failed: {e}", "ERROR")
    
    executor.save_stage_output('09_aix', {
        'input_mesh': input_mesh,
        'results_dir': str(executor.dirs['aix']),
        'hip_dir': str(hip_aix_dir),
        'spine_dir': str(spine_aix_dir),
        'hump_dir': str(hump_aix_dir),
        'success': success
    })
    
    return success


# ============================================================
# GRADIO INTERFACE
# ============================================================

# Global executor
executor = None
manual_alignment_path = None


def start_pipeline(far_video, near_video, progress=gr.Progress()):
    """Main pipeline execution with progress updates and visualizations"""
    global executor, manual_alignment_path
    
    # Validate inputs
    if not far_video:
        yield "[ERROR] Please select a FAR video file", "", 0, "", None
        return
    if not near_video:
        yield "[ERROR] Please select a NEAR video file", "", 0, "", None
        return
    
    # Initialize executor
    executor = PipelineExecutor()
    work_dir = executor.setup_working_dir()
    
    # Use default thresholds (user can adjust and regenerate after)
    executor.vggt_far_threshold = Config.VGGT_CONF_THRESH_FAR
    executor.vggt_near_threshold = Config.VGGT_CONF_THRESH_NEAR
    
    executor.log("=" * 60)
    executor.log("SCOLIOSIS ASSESSMENT PIPELINE STARTED", "STAGE")
    executor.log("=" * 60)
    executor.log(f"Working directory: {work_dir}")
    executor.log(f"Far video: {far_video}")
    executor.log(f"AI-based 3D reconstruction FAR threshold: {executor.vggt_far_threshold}")
    executor.log(f"AI-based 3D reconstruction NEAR threshold: {executor.vggt_near_threshold}")
    executor.log(f"Near video: {near_video}")
    executor.log("")
    executor.log("Environments used:")
    for stage, env in Config.ENVS.items():
        executor.log(f"  {stage}: {env}")
    
    yield executor.get_log_text(), work_dir, 0, executor.get_progress_text(), None
    
    # Stage 1: Video to Frames
    progress(0.1, desc="Stage 1/8: Extracting frames...")
    run_stage_video_to_frames(executor, far_video, near_video)
    executor.log("Frames extracted - check output folder for preview", "VIS")
    yield executor.get_log_text(), work_dir, 0.10, executor.get_progress_text(), None
    
    # Stage 2: AI-based 3D reconstruction with fixed thresholds
    progress(0.15, desc="Stage 2/8: AI-based 3D reconstruction...")
    run_stage_vggt(executor, 
                   far_threshold=executor.vggt_far_threshold, 
                   near_threshold=executor.vggt_near_threshold)
    
    # Stage 3: Convert GLB to PLY
    progress(0.20, desc="Stage 3/8: Converting GLB to PLY...")
    run_stage_glb_to_ply(executor)
    
    # Count points (no visualization)
    far_points = 0
    near_points = 0
    
    if executor.outputs.get('far_ply'):
        try:
            import trimesh
            pcd = trimesh.load(executor.outputs['far_ply'], force='pointcloud')
            if hasattr(pcd, 'vertices'):
                far_points = len(pcd.vertices)
        except:
            pass
    
    if executor.outputs.get('near_ply'):
        try:
            import trimesh
            pcd = trimesh.load(executor.outputs['near_ply'], force='pointcloud')
            if hasattr(pcd, 'vertices'):
                near_points = len(pcd.vertices)
        except:
            pass
    
    executor.log("")
    executor.log("=" * 60, "STAGE")
    executor.log("AI-BASED 3D RECONSTRUCTION COMPLETE", "SUCCESS")
    executor.log("=" * 60, "STAGE")
    executor.log("")
    executor.log(f"RESULTS:")
    executor.log(f"  FAR:  {far_points:,} points (threshold={executor.vggt_far_threshold})")
    executor.log(f"  NEAR: {near_points:,} points (threshold={executor.vggt_near_threshold})")
    executor.log("")
    
    # Continue automatically to next stages
    executor.current_stage = 3
    
    # Stage 4: Denoising
    progress(0.35, desc="Stage 4/8: Denoising point clouds...")
    run_stage_denoising(executor)
    
    # Stage 5: Mesh reconstruction
    progress(0.5, desc="Stage 5/8: Reconstructing meshes...")
    run_stage_mesh_reconstruction(executor)
    
    # Stage 5.5: Post-process NEAR mesh (remove Poisson artifacts)
    progress(0.55, desc="Stage 5.5/8: Post-processing NEAR mesh...")
    run_stage_post_process_near_mesh(executor)
    
    # Stage 6: Ground alignment
    progress(0.6, desc="Stage 6/8: Ground plane estimation...")
    run_stage_ground_alignment(executor)
    
    # Stage 7: Manual alignment (pauses here for user interaction)
    progress(0.7, desc="Stage 7/8: Manual alignment required...")
    ok, manual_alignment_path = run_stage_manual_alignment(executor)
    
    executor.log("")
    executor.log("=" * 60, "STAGE")
    executor.log("MANUAL ALIGNMENT REQUIRED", "WARNING")
    executor.log("=" * 60, "STAGE")
    executor.log("")
    executor.log("Go to the 'Manual Alignment' tab", "INFO")
    executor.log("")
    executor.log("Steps:", "INFO")
    executor.log("  1. Go to the 'Manual Alignment' tab", "INFO")
    executor.log("  2. Click 'Start Alignment' button", "INFO")
    executor.log("  3. Select 4 landmarks on the NEAR mesh in the browser viewer:", "INFO")
    executor.log("     a. HEAD (top of head/crown)", "INFO")
    executor.log("     b. LEFT SHOULDER", "INFO")
    executor.log("     c. RIGHT SHOULDER", "INFO")
    executor.log("     d. PELVIS MIDPOINT (center of pelvis/hip)", "INFO")
    executor.log("  4. Click 'Save & Close' in the viewer tab", "INFO")
    executor.log("  5. Repeat for FAR mesh (same 4 points in same order)", "INFO")
    executor.log("  6. Alignment will be computed automatically", "INFO")
    executor.log("")
    executor.log("In the viewer tab:", "INFO")
    executor.log("  • Click on the mesh to select a landmark point", "INFO")
    executor.log("  • Use Undo to remove the last point", "INFO")
    executor.log("  • Click 'Save & Close' when all 4 points are selected", "INFO")
    executor.log("  • X-axis alignment will be computed automatically from shoulder vector", "INFO")
    executor.log("")
    executor.log("")
    executor.log("Pipeline PAUSED - waiting for manual alignment...", "WARNING")
    
    executor.current_stage = 7
    yield executor.get_log_text(), work_dir, 0.80, executor.get_progress_text(), work_dir


def regenerate_glb_with_threshold(predictions_path, target_dir, label, conf_thres):
    """
    Regenerate GLB from saved predictions with new confidence threshold.
    This doesn't re-run the model, just regenerates the output.
    """
    import sys
    
    # Add VGGT path for imports
    base_dir = Path(__file__).parent
    vggt_path = str(base_dir / "prepro" / "vggt")
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)
    
    # Also add the vggt subdirectory (some imports expect it)
    vggt_subpath = str(base_dir / "prepro" / "vggt" / "vggt")
    if vggt_subpath not in sys.path:
        sys.path.insert(0, vggt_subpath)
    
    try:
        from visual_util import predictions_to_glb
    except ImportError as e:
        raise ImportError(f"Could not import predictions_to_glb from {vggt_path}: {e}")
    
    # Load saved predictions
    loaded = np.load(predictions_path)
    key_list = [
        "pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
        "images", "extrinsic", "intrinsic", "world_points_from_depth"
    ]
    predictions = {}
    for key in key_list:
        if key in loaded:
            predictions[key] = np.array(loaded[key])
    
    # Generate new GLB with new threshold
    glb_path = os.path.join(target_dir, f"{label}_threshold_{conf_thres}.glb")
    
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames="All",
        mask_black_bg=Config.VGGT_MASK_BLACK,
        mask_white_bg=Config.VGGT_MASK_WHITE,
        show_cam=False,  # Don't show camera for cleaner preview
        mask_sky=Config.VGGT_MASK_SKY,
        target_dir=target_dir,
        prediction_mode="Depthmap and Camera Branch",
    )
    glbscene.export(file_obj=glb_path)
    
    return glb_path


def compute_ssi_score(executor: PipelineExecutor, input_mesh: str) -> Optional[str]:
    """
    Compute SSI_score (0–100) by combining trunk shift, rib hump angle, and volume asymmetry.
    
    Formula (scale-invariant via normalization by spine length):
        dx_raw       = |neck_x - hip_x|
        spine_length = ||neck_center - hip_center||
        D_norm       = dx_raw / spine_length
        D_ref        = 0.18461
        S_D          = 1 - exp(-D_norm / D_ref)
        S_theta      = 1 - exp(-rib_hump_angle / 15)
        S_V          = volume_asymmetry
        SSI          = 100 * (0.50 * S_D + 0.25 * S_theta + 0.25 * S_V)
    
    Saves JSON:
        <mesh_stem>_final_scoliosis_score.json
    in the main AIX results directory.
    """
    import math

    def _load_hip_metrics(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (dx_raw, spine_length, dx_norm_denom) from hip_stats JSON."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            executor.log(f"Failed to read hip stats JSON: {e}", "ERROR")
            return None, None, None
        
        hip_center = None
        neck_center = None
        dx_raw = None
        
        if not isinstance(data, list):
            executor.log("Hip stats JSON has unexpected format (expected list).", "ERROR")
            return None, None, None
        
        for entry in data:
            q = entry.get("Quantity", "")
            if q == "Hip Center (XYZ)":
                try:
                    hip_center = np.array(entry.get("Value", []), dtype=float)
                except Exception:
                    executor.log("Failed to parse Hip Center (XYZ) from hip stats.", "ERROR")
            elif q == "Neck Center (XYZ)":
                try:
                    neck_center = np.array(entry.get("Value", []), dtype=float)
                except Exception:
                    executor.log("Failed to parse Neck Center (XYZ) from hip stats.", "ERROR")
            elif q in ["Absolute Delta X", "Delta X"]:
                try:
                    val = float(entry.get("Value", 0.0))
                    # Always use absolute magnitude for dx_raw
                    dx_raw = abs(val)
                except Exception:
                    executor.log("Failed to parse Delta X from hip stats.", "ERROR")
        
        if hip_center is None or neck_center is None:
            executor.log("Hip or neck center not found in hip stats JSON.", "ERROR")
            return dx_raw, None, None
        
        if hip_center.shape[0] != 3 or neck_center.shape[0] != 3:
            executor.log("Hip/neck centers have invalid shape in hip stats JSON.", "ERROR")
            return dx_raw, None, None
        
        spine_vec = neck_center - hip_center
        spine_length = float(np.linalg.norm(spine_vec))
        
        if spine_length <= 0:
            executor.log("Spine length is zero or negative – cannot normalize trunk shift.", "ERROR")
            return dx_raw, None, None
        
        return dx_raw, spine_length, spine_length

    def _load_hump_metrics(path: Path) -> Tuple[Optional[float], Optional[float]]:
        """Return (rib_hump_angle_deg, volume_asymmetry) from hump_rib_hump_stats JSON."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            executor.log(f"Failed to read hump stats JSON: {e}", "ERROR")
            return None, None
        
        rib_hump_angle = None
        volume_asym = None
        
        if not isinstance(data, list):
            executor.log("Hump stats JSON has unexpected format (expected list).", "ERROR")
            return None, None
        
        for entry in data:
            q = entry.get("Quantity", "")
            if q == "rib_hump_angle":
                try:
                    rib_hump_angle = float(entry.get("Quantity Estimated Value", 0.0))
                except Exception:
                    executor.log("Failed to parse rib_hump_angle from hump stats.", "ERROR")
            elif q == "volume_asymmetry":
                try:
                    volume_asym = float(entry.get("Quantity Estimated Value", 0.0))
                except Exception:
                    executor.log("Failed to parse volume_asymmetry from hump stats.", "ERROR")
        
        return rib_hump_angle, volume_asym

    # ------------------------------------------------------------------
    # Locate input JSON files
    # ------------------------------------------------------------------
    hip_json_path = executor.outputs.get('hip_json')
    hip_path: Optional[Path] = None
    hump_path: Optional[Path] = None
    
    if hip_json_path and os.path.exists(hip_json_path):
        hip_path = Path(hip_json_path)
    else:
        # Fallback: search Hip_aix directory
        hip_dir = executor.dirs.get('aix') / "Hip_aix"
        if hip_dir and hip_dir.exists():
            candidates = sorted(hip_dir.glob("*hip_stats.json"))
            if candidates:
                hip_path = candidates[0]
    
    hump_dir = executor.dirs.get('aix') / "Hump_aix"
    if hump_dir and hump_dir.exists():
        hump_candidates = sorted(hump_dir.glob("*rib_hump_stats.json"))
        if hump_candidates:
            hump_path = hump_candidates[0]
    
    if hip_path is None:
        executor.log("Cannot compute SSI_score: hip stats JSON not found.", "ERROR")
    if hump_path is None:
        executor.log("Cannot fully compute SSI_score: hump stats JSON not found.", "ERROR")
    
    # Load metrics (some may be None)
    dx_raw = spine_length = denom = None
    if hip_path is not None:
        executor.log(f"Loading hip metrics from: {hip_path.name}", "INFO")
        dx_raw, spine_length, denom = _load_hip_metrics(hip_path)
    
    rib_hump_angle = volume_asym = None
    if hump_path is not None:
        executor.log(f"Loading hump metrics from: {hump_path.name}", "INFO")
        rib_hump_angle, volume_asym = _load_hump_metrics(hump_path)
    
    # ------------------------------------------------------------------
    # Compute normalized components
    # ------------------------------------------------------------------
    D_norm = None
    S_D = S_theta = S_V = None
    
    if dx_raw is not None and denom is not None and denom > 0:
        D_norm = float(dx_raw) / float(denom)
        D_ref = 0.18461  # Severe shift ≈ 18.461% of spine length
        S_D = 1.0 - math.exp(-D_norm / D_ref)
    else:
        executor.log("Skipping S_D (trunk shift score) – missing dx_raw or spine_length.", "WARNING")
    
    if rib_hump_angle is not None:
        # rib_hump_angle is in degrees; scale with characteristic angle 15°
        S_theta = 1.0 - math.exp(-float(rib_hump_angle) / 15.0)
    else:
        executor.log("Skipping S_theta (rib hump score) – rib_hump_angle not available.", "WARNING")
    
    if volume_asym is not None:
        S_V = float(volume_asym)
    else:
        executor.log("Skipping S_V (volume asymmetry score) – volume_asymmetry not available.", "WARNING")
    
    SSI_score = None
    if S_D is not None and S_theta is not None and S_V is not None:
        raw_SSI = 100.0 * (0.50 * S_D + 0.25 * S_theta + 0.25 * S_V)
        # Clamp to [0, 100]
        SSI_score = max(0.0, min(100.0, raw_SSI))
    else:
        executor.log("Insufficient components to compute full SSI_score.", "ERROR")
    
    # ------------------------------------------------------------------
    # Severity label
    # ------------------------------------------------------------------
    def _severity(score: Optional[float]) -> str:
        if score is None:
            return "Unknown"
        if score < 20:
            return "Normal"
        if score < 40:
            return "Mild"
        if score < 60:
            return "Moderate"
        if score < 80:
            return "Severe"
        return "Very Severe"
    
    severity_label = _severity(SSI_score)
    
    # ------------------------------------------------------------------
    # Save combined JSON
    # ------------------------------------------------------------------
    out_dir = executor.dirs.get('aix') or Path(os.path.dirname(input_mesh))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_stem = Path(input_mesh).stem
    out_path = out_dir / f"{mesh_stem}_final_scoliosis_score.json"
    
    payload = {
        "dx_raw": float(dx_raw) if dx_raw is not None else None,
        "dx_norm": float(D_norm) if D_norm is not None else None,
        "spine_length": float(spine_length) if spine_length is not None else None,
        "rib_hump_angle": float(rib_hump_angle) if rib_hump_angle is not None else None,
        "volume_asymmetry": float(volume_asym) if volume_asym is not None else None,
        "SSI_score": float(SSI_score) if SSI_score is not None else None,
        "severity_label": severity_label,
    }
    
    try:
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        executor.log(f"Saved final scoliosis score to: {out_path}", "FILE")
    except Exception as e:
        executor.log(f"Failed to write final scoliosis score JSON: {e}", "ERROR")
        return None
    
    return str(out_path)

def glb_to_ply_points(glb_path):
    """Extract points and colors from GLB file and return as numpy arrays"""
    import trimesh
    
    print(f"Loading GLB: {glb_path}")
    mesh = trimesh.load(glb_path)
    print(f"GLB type: {type(mesh)}")
    
    points = None
    colors = None
    
    if isinstance(mesh, trimesh.PointCloud):
        points = np.array(mesh.vertices)
        if hasattr(mesh, 'colors') and mesh.colors is not None and len(mesh.colors) > 0:
            colors = np.array(mesh.colors)[:, :3] / 255.0 if mesh.colors.max() > 1 else np.array(mesh.colors)[:, :3]
        print(f"PointCloud: {len(points)} points")
    elif isinstance(mesh, trimesh.Scene):
        all_verts = []
        all_colors = []
        for name, g in mesh.geometry.items():
            print(f"  Geometry '{name}': {type(g)}")
            if hasattr(g, 'vertices') and len(g.vertices) > 0:
                all_verts.append(np.array(g.vertices))
                # Try to get colors
                if hasattr(g, 'visual'):
                    if hasattr(g.visual, 'vertex_colors') and g.visual.vertex_colors is not None:
                        vc = np.array(g.visual.vertex_colors)[:, :3]
                        vc = vc / 255.0 if vc.max() > 1 else vc
                        all_colors.append(vc)
                    elif hasattr(g, 'colors') and g.colors is not None:
                        vc = np.array(g.colors)[:, :3]
                        vc = vc / 255.0 if vc.max() > 1 else vc
                        all_colors.append(vc)
        if all_verts:
            points = np.vstack(all_verts)
            if all_colors and sum(len(c) for c in all_colors) == len(points):
                colors = np.vstack(all_colors)
        print(f"Scene: {len(points) if points is not None else 0} total points from {len(all_verts)} geometries")
    elif hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
        points = np.array(mesh.vertices)
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) > 0:
                colors = np.array(vc)[:, :3]
                colors = colors / 255.0 if colors.max() > 1 else colors
        print(f"Mesh: {len(points)} vertices")
    
    if points is None:
        points = np.array([])
    
    return points, colors


def create_visualization_from_points(points, colors=None, title="3D Visualization", max_points=80000):
    """Create Plotly figure directly from numpy arrays - bypasses file I/O issues"""
    
    if points is None or len(points) == 0:
        return create_empty_figure(title, "No points to visualize")
    
    # Subsample if needed
    n = len(points)
    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Color handling
    if colors is not None and len(colors) == len(points):
        # Use provided colors
        if colors.max() <= 1:
            r = (colors[:, 0] * 255).astype(int)
            g = (colors[:, 1] * 255).astype(int)
            b = (colors[:, 2] * 255).astype(int)
        else:
            r, g, b = colors[:, 0].astype(int), colors[:, 1].astype(int), colors[:, 2].astype(int)
        color_str = [f'rgb({rr},{gg},{bb})' for rr, gg, bb in zip(r, g, b)]
        marker = dict(size=2, color=color_str, opacity=0.85)
    else:
        # Color by height
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            norm_z = (z - z_min) / (z_max - z_min)
            color_vals = np.zeros((len(z), 3))
            color_vals[:, 0] = norm_z        # Red
            color_vals[:, 2] = 1 - norm_z    # Blue
            color_vals[:, 1] = 0.3           # Green
            color_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in color_vals]
            marker = dict(size=2, color=color_str, opacity=0.85)
        else:
            marker = dict(size=2, color='steelblue', opacity=0.85)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=marker,
        name=title,
        hovertemplate='X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>'
    )])
    
    # Add coordinate axes
    center = np.mean(points, axis=0)
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    axis_len = np.max(extent) * 0.15
    
    fig.add_trace(go.Scatter3d(
        x=[center[0], center[0] + axis_len], y=[center[1], center[1]], z=[center[2], center[2]],
        mode='lines', line=dict(color='red', width=6), name='X', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[center[0], center[0]], y=[center[1], center[1] + axis_len], z=[center[2], center[2]],
        mode='lines', line=dict(color='green', width=6), name='Y', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[center[0], center[0]], y=[center[1], center[1]], z=[center[2], center[2] + axis_len],
        mode='lines', line=dict(color='blue', width=6), name='Z', showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> ({len(points):,} pts)", font=dict(size=14)),
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            bgcolor='#1a1a2e',
            xaxis=dict(gridcolor='#444', zerolinecolor='#666'),
            yaxis=dict(gridcolor='#444', zerolinecolor='#666'),
            zaxis=dict(gridcolor='#444', zerolinecolor='#666')
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        paper_bgcolor='#f0f0f0',
        showlegend=False
    )
    
    return fig


def save_points_as_ply(points, colors, ply_path):
    """Save points as PLY file using numpy (more reliable than trimesh for point clouds)"""
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    
    n_points = len(points)
    has_colors = colors is not None and len(colors) == n_points
    
    # Create PLY header
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
"""
    if has_colors:
        header += """property uchar red
property uchar green
property uchar blue
"""
    header += "end_header\n"
    
    with open(ply_path, 'w') as f:
        f.write(header)
        for i in range(n_points):
            if has_colors:
                r = int(colors[i, 0] * 255) if colors[i, 0] <= 1 else int(colors[i, 0])
                g = int(colors[i, 1] * 255) if colors[i, 1] <= 1 else int(colors[i, 1])
                b = int(colors[i, 2] * 255) if colors[i, 2] <= 1 else int(colors[i, 2])
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")
    
    print(f"Saved PLY: {ply_path} ({n_points} points)")


def preview_vggt_far(threshold):
    """Preview FAR point cloud with new threshold using saved predictions"""
    global executor
    
    if executor is None:
        return None, "[ERROR] No pipeline running. Start pipeline first."
    
    if executor.current_stage < 3:
        return None, "[ERROR] AI-based 3D reconstruction stage not complete yet. Wait for initial reconstruction."
    
    # Find predictions.npz for FAR
    # The AI-based 3D reconstruction script creates: output_dir/vggt_far/predictions.npz
    # where output_dir = executor.dirs['vggt_far'].parent = work_dir/02_vggt
    predictions_path = None
    search_paths = [
        executor.work_dir / "02_vggt" / "vggt_far" / "predictions.npz",  # Most likely location
        executor.dirs['vggt_far'].parent / "vggt_far" / "predictions.npz",
        executor.dirs['vggt_far'] / "vggt_far" / "predictions.npz",
        executor.dirs['vggt_far'] / "predictions.npz",
    ]
    
    for possible_path in search_paths:
        if possible_path.exists():
            predictions_path = possible_path
            break
    
    if predictions_path is None:
        executor.log(f"predictions.npz not found for FAR", "ERROR")
        executor.log(f"  Searched paths:", "INFO")
        for p in search_paths:
            executor.log(f"    - {p}", "INFO")
        return None, "[ERROR] FAR predictions not found. Re-run AI-based 3D reconstruction stage."
    
    executor.log(f"Regenerating FAR point cloud with threshold={threshold}...", "INFO")
    
    try:
        target_dir = str(predictions_path.parent)
        executor.log(f"  Found predictions at: {predictions_path}", "INFO")
        
        # Regenerate GLB with new threshold
        executor.log(f"  Regenerating GLB with threshold={threshold}...", "INFO")
        glb_path = regenerate_glb_with_threshold(
            str(predictions_path), target_dir, "far", threshold
        )
        executor.log(f"  GLB created: {glb_path}", "INFO")
        
        # Extract points and colors from GLB
        points, colors = glb_to_ply_points(glb_path)
        
        if points is not None and len(points) > 0:
            # Save as PLY
            ply_path = str(executor.dirs['ply_far'] / f'far_pointcloud_t{int(threshold)}.ply')
            save_points_as_ply(points, colors, ply_path)
            
            # Update executor state
            executor.outputs['far_ply'] = ply_path
            executor.outputs['far_glb'] = glb_path
            executor.vggt_far_threshold = threshold
            
            # Create visualization DIRECTLY from points (bypasses file read issues)
            fig = create_visualization_from_points(
                points, colors,
                f"FAR Point Cloud (Threshold={threshold})"
            )
            
            executor.log(f"FAR preview: {len(points):,} points with threshold={threshold}", "SUCCESS")
            return fig, f"FAR: {len(points):,} points (threshold={threshold})"
        else:
            executor.log(f"FAR: No points with threshold={threshold} (try lowering)", "WARNING")
            return None, f"[WARNING] No points! Threshold {threshold} too high. Try lowering it."
        
    except Exception as e:
        executor.log(f"FAR preview failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None, f"[ERROR] Error: {str(e)}"


def preview_vggt_near(threshold):
    """Preview NEAR point cloud with new threshold using saved predictions"""
    global executor
    
    if executor is None:
        return None, "[ERROR] No pipeline running. Start pipeline first."
    
    if executor.current_stage < 3:
        return None, "[ERROR] AI-based 3D reconstruction stage not complete yet. Wait for initial reconstruction."
    
    # Find predictions.npz for NEAR
    # The AI-based 3D reconstruction script creates: output_dir/vggt_near/predictions.npz
    # where output_dir = executor.dirs['vggt_near'].parent = work_dir/02_vggt
    predictions_path = None
    search_paths = [
        executor.work_dir / "02_vggt" / "vggt_near" / "predictions.npz",  # Most likely location
        executor.dirs['vggt_near'].parent / "vggt_near" / "predictions.npz",
        executor.dirs['vggt_near'] / "vggt_near" / "predictions.npz",
        executor.dirs['vggt_near'] / "predictions.npz",
    ]
    
    for possible_path in search_paths:
        if possible_path.exists():
            predictions_path = possible_path
            break
    
    if predictions_path is None:
        executor.log(f"predictions.npz not found for NEAR", "ERROR")
        executor.log(f"  Searched paths:", "INFO")
        for p in search_paths:
            executor.log(f"    - {p}", "INFO")
        return None, "[ERROR] NEAR predictions not found. Re-run AI-based 3D reconstruction stage."
    
    executor.log(f"Regenerating NEAR point cloud with threshold={threshold}...", "INFO")
    
    try:
        target_dir = str(predictions_path.parent)
        executor.log(f"  Found predictions at: {predictions_path}", "INFO")
        
        # Regenerate GLB with new threshold
        executor.log(f"  Regenerating GLB with threshold={threshold}...", "INFO")
        glb_path = regenerate_glb_with_threshold(
            str(predictions_path), target_dir, "near", threshold
        )
        executor.log(f"  GLB created: {glb_path}", "INFO")
        
        # Extract points and colors from GLB
        points, colors = glb_to_ply_points(glb_path)
        
        if points is not None and len(points) > 0:
            # Save as PLY
            ply_path = str(executor.dirs['ply_near'] / f'near_pointcloud_t{int(threshold)}.ply')
            save_points_as_ply(points, colors, ply_path)
            
            # Update executor state
            executor.outputs['near_ply'] = ply_path
            executor.outputs['near_glb'] = glb_path
            executor.vggt_near_threshold = threshold
            
            # Create visualization DIRECTLY from points (bypasses file read issues)
            fig = create_visualization_from_points(
                points, colors,
                f"NEAR Point Cloud (Threshold={threshold})"
            )
            
            executor.log(f"NEAR preview: {len(points):,} points with threshold={threshold}", "SUCCESS")
            return fig, f"NEAR: {len(points):,} points (threshold={threshold})"
        else:
            executor.log(f"NEAR: No points with threshold={threshold} (try lowering)", "WARNING")
            return None, f"[WARNING] No points! Threshold {threshold} too high. Try lowering it."
        
    except Exception as e:
        executor.log(f"NEAR preview failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None, f"[ERROR] Error: {str(e)}"


# continue_to_next_stage function removed - pipeline now runs continuously


def compute_xaxis_alignment_from_shoulders(shoulder_left, shoulder_right):
    """
    Compute rotation matrix to align left-to-right shoulder vector with +X axis.
    
    The left-to-right shoulder vector should point in the +X direction after rotation.
    
    Args:
        shoulder_left: 3D point for left shoulder [x, y, z]
        shoulder_right: 3D point for right shoulder [x, y, z]
        
    Returns:
        rotation_matrix: 3x3 rotation matrix (rotation around Y-axis)
        angle_degrees: Rotation angle in degrees
    """
    # Compute shoulder vector: from LEFT to RIGHT (should point to +X)
    shoulder_vec = np.array(shoulder_right) - np.array(shoulder_left)
    shoulder_vec_norm = np.linalg.norm(shoulder_vec)
    
    if shoulder_vec_norm < 1e-6:
        print("[WARNING] Shoulders are at same position, no rotation needed")
        return np.eye(3), 0.0
    
    # Project to X-Z plane (set Y=0) for rotation around Y-axis
    shoulder_vec_xz = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]])
    shoulder_vec_xz_norm = np.linalg.norm(shoulder_vec_xz)
    
    if shoulder_vec_xz_norm < 1e-6:
        # Shoulder vector is parallel to Y-axis, cannot align to X
        print("[WARNING] Shoulder vector is parallel to Y-axis, no rotation needed")
        return np.eye(3), 0.0
    
    # Normalize the X-Z projection
    shoulder_vec_xz = shoulder_vec_xz / shoulder_vec_xz_norm
    
    # Compute angle from current direction to +X axis
    # atan2(z, x) gives angle from +X axis in X-Z plane
    # We want to rotate by -angle to align with +X
    current_angle = np.arctan2(shoulder_vec_xz[2], shoulder_vec_xz[0])
    rotation_angle = current_angle  # positive to rotate towards +X
    angle_degrees = np.degrees(rotation_angle)
    
    print(f"Left shoulder: [{shoulder_left[0]:.4f}, {shoulder_left[1]:.4f}, {shoulder_left[2]:.4f}]")
    print(f"Right shoulder: [{shoulder_right[0]:.4f}, {shoulder_right[1]:.4f}, {shoulder_right[2]:.4f}]")
    print(f"Shoulder vector (left→right): [{shoulder_vec[0]:.4f}, {shoulder_vec[1]:.4f}, {shoulder_vec[2]:.4f}]")
    print(f"Shoulder vector (X-Z plane): [{shoulder_vec_xz[0]:.4f}, 0, {shoulder_vec_xz[2]:.4f}]")
    print(f"Current angle from +X: {np.degrees(current_angle):.2f}°")
    print(f"Rotation angle: {angle_degrees:.2f}°")
    
    # Rotation matrix around Y-axis to align shoulder vector with +X axis
    # Standard rotation matrix for rotation around Y-axis:
    # R_y(θ) = [[cos(θ), 0, sin(θ)],
    #            [0, 1, 0],
    #            [-sin(θ), 0, cos(θ)]]
    # This rotates points counter-clockwise around Y when viewed from above (positive Y)
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    # Verify: R @ shoulder_vec_xz should point to [1, 0, 0]
    rotated_vec = R @ shoulder_vec_xz
    print(f"Verification - After rotation, vector: [{rotated_vec[0]:.4f}, 0, {rotated_vec[2]:.4f}] (expected [1.0, 0, 0.0])")
    
    # Check if alignment is correct
    alignment_ok = abs(rotated_vec[0] - 1.0) < 0.01 and abs(rotated_vec[2]) < 0.01
    if not alignment_ok:
        print(f"[WARNING] Rotation verification failed. Expected [1, 0, 0], got [{rotated_vec[0]:.4f}, 0, {rotated_vec[2]:.4f}]")
        print(f"   Error: X component off by {abs(rotated_vec[0] - 1.0):.6f}, Z component is {rotated_vec[2]:.6f}")
    else:
        print(f"[SUCCESS] Rotation matrix verified: shoulder vector will align with +X axis")
    
    return R, angle_degrees


def apply_xaxis_alignment_to_mesh(mesh_path, rotation_matrix, center_point, output_path):
    """
    Apply X-axis rotation to a mesh.
    
    Args:
        mesh_path: Path to input mesh
        rotation_matrix: 3x3 rotation matrix
        center_point: Center point for rotation
        output_path: Path to save rotated mesh
        
    Returns:
        bool: True if successful
    """
    import trimesh
    
    try:
        # Try loading as mesh first
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                # Create transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = center_point - rotation_matrix @ center_point
                mesh.apply_transform(transform)
                mesh.export(output_path)
                return True
        except:
            pass
        
        # Try as point cloud
        try:
            pcd = trimesh.load(mesh_path, force='pointcloud')
            if hasattr(pcd, 'vertices') and len(pcd.vertices) > 0:
                # Apply rotation around center
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = center_point - rotation_matrix @ center_point
                pcd.apply_transform(transform)
                pcd.export(output_path)
                return True
        except:
            pass
        
        return False
    except Exception as e:
        print(f"Error applying X-axis alignment: {e}")
        return False


def center_mesh_bbox_at_origin(mesh_path, output_dir):
    """
    Center the mesh's bounding box at the origin (0, 0, 0).
    This replicates Meshlab's "center on layer bbox" operation.
    
    Process:
    1. Compute bounding box of the mesh
    2. Calculate the center of the bounding box
    3. Translate the mesh so the bounding box center is at origin
    
    Args:
        mesh_path: Path to input mesh
        output_dir: Directory to save the centered mesh
    
    Returns:
        str: Path to the centered mesh, or None if failed
    """
    import trimesh
    
    try:
        # Load mesh or point cloud
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            is_mesh = hasattr(mesh, 'vertices') and len(mesh.vertices) > 0
            if not is_mesh:
                raise ValueError("Not a mesh")
        except:
            try:
                pcd = trimesh.load(mesh_path, force='pointcloud')
                if not hasattr(pcd, 'vertices') or len(pcd.vertices) == 0:
                    print(f"ERROR: Cannot load mesh or point cloud from {mesh_path}")
                    return None
                mesh = pcd
                is_mesh = False
            except Exception as e:
                print(f"ERROR: Cannot load mesh or point cloud from {mesh_path}: {e}")
                return None
        
        # Compute bounding box
        bbox = mesh.bounds
        bbox_center = bbox.mean(axis=0)
        bbox_min = bbox[0]
        bbox_max = bbox[1]
        
        print(f"Bounding box center: [{bbox_center[0]:.6f}, {bbox_center[1]:.6f}, {bbox_center[2]:.6f}]")
        print(f"Bounding box min: [{bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f}]")
        print(f"Bounding box max: [{bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f}]")
        
        # Create translation matrix to move bbox center to origin
        # Translation = -bbox_center (move center to origin)
        translation = -bbox_center
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, 3] = translation
        
        # Apply transformation
        mesh.apply_transform(transform)
        
        # Verify: new bbox center should be at origin
        bbox_new = mesh.bounds
        bbox_center_new = bbox_new.mean(axis=0)
        
        print(f"New bounding box center: [{bbox_center_new[0]:.6f}, {bbox_center_new[1]:.6f}, {bbox_center_new[2]:.6f}]")
        print(f"Translation applied: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
        
        # Check if centering was successful (should be very close to origin)
        center_error = np.linalg.norm(bbox_center_new)
        if center_error > 1e-6:
            print(f"[WARNING] Bounding box center is not exactly at origin. Error: {center_error:.9f}")
        else:
            print(f"[SUCCESS] Bounding box successfully centered at origin")
        
        # Save centered mesh
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_stem = Path(mesh_path).stem
        output_path = output_dir / f"{input_stem}_centered.ply"
        
        mesh.export(str(output_path))
        
        print(f"[SUCCESS] Centered mesh saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"ERROR centering mesh bounding box: {e}")
        import traceback
        traceback.print_exc()
        return None


def post_process_near_mesh(aligned_mesh_path, original_pcd_path=None, output_dir=None):
    """
    Post-process NEAR mesh to remove Poisson reconstruction artifacts.
    
    Uses vertex quality filtering:
    1. Computes vertex quality (distance from original point cloud or curvature-based)
    2. Automatically determines threshold based on quality distribution
    3. Removes vertices below threshold
    4. Reconstructs mesh from remaining vertices
    
    Args:
        aligned_mesh_path: Path to aligned NEAR mesh
        original_pcd_path: Path to original point cloud (for distance-based quality)
        output_dir: Directory to save cleaned mesh
        
    Returns:
        Path to cleaned mesh, or None if failed
    """
    global executor
    import trimesh
    from scipy.spatial import cKDTree
    
    def log_msg(msg, level="INFO"):
        if executor:
            executor.log(msg, level)
        else:
            print(f"[{level}] {msg}")
    
    if executor is None:
        log_msg("Executor not available, skipping post-processing", "WARNING")
        return None
    
    try:
        log_msg(f"  Loading aligned mesh: {aligned_mesh_path}", "INFO")
        mesh = trimesh.load(aligned_mesh_path, force='mesh')
        
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            log_msg("  Empty mesh, skipping post-processing", "WARNING")
            return None
        
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            log_msg("  Mesh has no triangles, skipping post-processing", "WARNING")
            return None
        
        vertices = np.array(mesh.vertices)
        n_vertices = len(vertices)
        log_msg(f"  Mesh has {n_vertices:,} vertices", "INFO")
        
        # Compute vertex quality
        quality = np.zeros(n_vertices)
        
        # Method 1: Distance-based quality (if original point cloud available)
        if original_pcd_path and os.path.exists(original_pcd_path):
            log_msg(f"  Computing quality from original point cloud: {original_pcd_path}", "INFO")
            try:
                original_pcd = trimesh.load(original_pcd_path, force='pointcloud')
                if hasattr(original_pcd, 'vertices') and len(original_pcd.vertices) > 0:
                    original_points = np.array(original_pcd.vertices)
                    # Build KDTree for fast nearest neighbor search
                    tree = cKDTree(original_points)
                    # Find distance to nearest point in original cloud
                    distances, _ = tree.query(vertices, k=1)
                    # Quality is inverse of distance (closer = higher quality)
                    max_dist = np.percentile(distances, 95)  # Use 95th percentile as normalization
                    if max_dist > 1e-6:
                        quality = 1.0 - (distances / max_dist)
                        quality = np.clip(quality, 0.0, 1.0)
                    else:
                        quality = np.ones(n_vertices)
                    log_msg(f"  Distance-based quality computed (median: {np.median(quality):.3f})", "INFO")
                else:
                    raise ValueError("Empty original point cloud")
            except Exception as e:
                log_msg(f"  Distance-based quality failed: {e}, using curvature-based", "WARNING")
                original_pcd_path = None  # Fall back to curvature
        
        # Method 2: Curvature-based quality (if distance-based failed or not available)
        if original_pcd_path is None or np.all(quality == 0):
            log_msg("  Computing quality from mesh curvature...", "INFO")
            try:
                # Trimesh automatically computes vertex normals when needed
                
                # Estimate curvature using PCA of local neighborhood
                k = min(20, n_vertices // 10)  # Number of neighbors
                if k < 3:
                    k = 3
                
                # Build KDTree for neighbor search
                tree = cKDTree(vertices)
                
                for i in range(n_vertices):
                    # Find k nearest neighbors
                    distances, indices = tree.query(vertices[i], k=min(k+1, n_vertices))
                    if len(indices) < 2:
                        quality[i] = 1.0
                        continue
                    
                    # Get neighbor points (exclude self)
                    neighbor_indices = indices[1:] if indices[0] == i else indices[:k]
                    neighbor_pts = vertices[neighbor_indices]
                    
                    # Center points
                    center = vertices[i]
                    centered = neighbor_pts - center
                    
                    # PCA to find surface variation
                    if len(centered) >= 3:
                        cov = np.cov(centered.T)
                        eigenvals = np.linalg.eigvals(cov)
                        eigenvals = np.sort(eigenvals)[::-1]
                        if eigenvals[0] > 1e-10:
                            # Surface variation = smallest eigenvalue / sum
                            variation = eigenvals[-1] / np.sum(eigenvals)
                            # Low variation = smooth surface = high quality
                            quality[i] = 1.0 - min(variation * 10, 1.0)  # Scale variation
                        else:
                            quality[i] = 1.0
                    else:
                        quality[i] = 1.0
                
                log_msg(f"  Curvature-based quality computed (median: {np.median(quality):.3f})", "INFO")
            except Exception as e:
                log_msg(f"  Curvature computation failed: {e}, using uniform quality", "WARNING")
                quality = np.ones(n_vertices)
        
        # Automatically determine threshold based on quality distribution
        quality_sorted = np.sort(quality)
        # Use percentile-based threshold: remove bottom 5-15% of vertices
        # Adaptive: if quality range is large, be more aggressive
        quality_range = np.max(quality) - np.min(quality)
        
        if quality_range > 0.3:  # Large variation in quality
            percentile = 10  # Remove bottom 10%
        elif quality_range > 0.1:  # Medium variation
            percentile = 7   # Remove bottom 7%
        else:  # Small variation
            percentile = 5   # Remove bottom 5%
        
        threshold = np.percentile(quality, percentile)
        log_msg(f"  Quality range: {quality_range:.3f}, threshold (percentile {percentile}): {threshold:.3f}", "INFO")
        
        # Filter vertices above threshold
        keep_mask = quality >= threshold
        n_keep = np.sum(keep_mask)
        n_remove = n_vertices - n_keep
        removal_pct = (n_remove / n_vertices) * 100
        
        log_msg(f"  Keeping {n_keep:,} vertices ({100-removal_pct:.1f}%), removing {n_remove:,} ({removal_pct:.1f}%)", "INFO")
        
        if n_keep < 100:
            log_msg("  Too few vertices remaining, skipping post-processing", "WARNING")
            return None
        
        # Extract vertices and faces to keep
        keep_indices = np.where(keep_mask)[0]
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        
        # Filter vertices
        filtered_vertices = vertices[keep_mask]
        
        # Filter triangles (keep only triangles where all vertices are kept)
        triangles = np.array(mesh.faces)
        # Check if all three vertices of each triangle are in keep_indices
        keep_set = set(keep_indices)
        valid_triangles = np.array([all(v in keep_set for v in tri) for tri in triangles])
        filtered_triangles = triangles[valid_triangles]
        
        # Remap triangle indices to new vertex indices
        remapped_triangles = np.array([[vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]] 
                                       for tri in filtered_triangles])
        
        # Create new mesh with trimesh
        cleaned_mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=remapped_triangles)
        
        # Cleanup
        cleaned_mesh.remove_duplicate_faces()
        cleaned_mesh.remove_unreferenced_vertices()
        cleaned_mesh.fix_normals()
        cleaned_mesh.remove_duplicated_triangles()
        cleaned_mesh.remove_duplicated_vertices()
        cleaned_mesh.remove_non_manifold_edges()
        
        # Save cleaned mesh
        if output_dir is None:
            output_dir = os.path.dirname(aligned_mesh_path)
        else:
            output_dir = str(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        cleaned_path = os.path.join(output_dir, "near_aligned_cleaned.ply")
        cleaned_mesh.export(cleaned_path)
        
        log_msg(f"  Cleaned mesh saved: {cleaned_path}", "SUCCESS")
        log_msg(f"    Vertices: {n_vertices:,} → {len(cleaned_mesh.vertices):,}", "INFO")
        log_msg(f"    Triangles: {len(mesh.faces):,} → {len(cleaned_mesh.faces):,}", "INFO")
        
        return cleaned_path
        
    except Exception as e:
        log_msg(f"  Post-processing failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


# ===========================================================================
# NOTE: The old Open3D-based alignment functions (start_alignment_web,
# pick_points_open3d, add_source_point, etc.) have been replaced by the
# web-based browser-tab workflow in manual_alignment_automated.py.
# ===========================================================================


def _deprecated_alignment_placeholder():
    """Placeholder — all alignment logic is now in manual_alignment_automated.py"""
    pass



def regenerate_vggt(far_threshold, near_threshold, progress=gr.Progress()):
    """Regenerate AI-based 3D reconstruction with new thresholds"""
    global executor
    
    if executor is None:
        yield "❌ No pipeline running. Start a pipeline first.", "", 0, "", None, None
        return
    
    # Check if we have frames
    if not executor.dirs.get('frames_far') or not executor.dirs['frames_far'].exists():
        yield "[ERROR] No frames found. Run Stage 1 first.", str(executor.work_dir), 0.1, "", None
        return
    
    executor.log("")
    executor.log("=" * 60, "STAGE")
    executor.log("REGENERATING AI-BASED 3D RECONSTRUCTION WITH NEW THRESHOLDS", "STAGE")
    executor.log("=" * 60, "STAGE")
    executor.log(f"Previous thresholds: FAR={executor.vggt_far_threshold}, NEAR={executor.vggt_near_threshold}")
    executor.log(f"New thresholds: FAR={far_threshold}, NEAR={near_threshold}")
    executor.log("")
    
    yield executor.get_log_text(), str(executor.work_dir), 0.15, "Regenerating...", None, None
    
    # Update thresholds
    executor.vggt_far_threshold = far_threshold
    executor.vggt_near_threshold = near_threshold
    
    progress(0.16, desc="Running AI-based 3D reconstruction with new thresholds...")
    executor.log("Running AI-based 3D reconstruction with new thresholds (this may take a few minutes)...", "PROGRESS")
    
    # Re-run VGGT stage
    run_stage_vggt(executor, 
                   far_threshold=far_threshold, 
                   near_threshold=near_threshold)
    
    yield executor.get_log_text(), str(executor.work_dir), 0.18, "Converting to PLY...", None, None
    
    progress(0.18, desc="Converting GLB to PLY...")
    executor.log("Converting new GLB to PLY for visualization...", "PROGRESS")
    
    # Convert to PLY
    run_stage_glb_to_ply(executor)
    
    progress(0.20, desc="Generating visualization...")
    
    executor.log("")
    executor.log("=" * 60, "STAGE")
    executor.log("AI-BASED 3D RECONSTRUCTION REGENERATION COMPLETE", "SUCCESS")
    executor.log("=" * 60, "STAGE")
    executor.log("")
    executor.log("Happy with the result?", "INFO")
    executor.log("  → Pipeline will continue from where it paused", "INFO")
    executor.log("  → The new reconstruction will be used for all further stages", "INFO")
    executor.log("")
    executor.log("Still not right? Adjust thresholds and regenerate again!", "INFO")
    executor.log("")
    
    yield executor.get_log_text(), str(executor.work_dir), 0.20, executor.get_progress_text(), None, None


def continue_after_vggt(progress=gr.Progress()):
    """Continue pipeline from after AI-based 3D reconstruction (denoising onwards)"""
    global executor
    
    if executor is None:
        yield "❌ No pipeline running. Start a pipeline first.", "", 0, "", None, None
        return
    
    work_dir = str(executor.work_dir)
    
    # Check if AI-based 3D reconstruction has run
    if not executor.outputs.get('far_ply'):
        yield "❌ AI-based 3D reconstruction not completed yet. Wait for Stage 2 to finish.", work_dir, 0.1, "", None, None
        return
    
    executor.log("")
    executor.log("=" * 60, "STAGE")
    executor.log("CONTINUING PIPELINE FROM DENOISING...", "STAGE")
    executor.log("=" * 60, "STAGE")
    
    yield executor.get_log_text(), work_dir, 0.30, executor.get_progress_text(), None, None
    
    # Stage 4: Denoising
    progress(0.4, desc="Stage 4/10: Denoising point clouds...")
    run_stage_denoising(executor)
    
    yield executor.get_log_text(), work_dir, 0.40, executor.get_progress_text(), None, None
    
    # Stage 5: Mesh reconstruction
    progress(0.5, desc="Stage 5/8: Reconstructing meshes...")
    run_stage_mesh_reconstruction(executor)
    
    yield executor.get_log_text(), work_dir, 0.50, executor.get_progress_text(), None, None
    
    # Stage 5.5: Post-process NEAR mesh (remove Poisson artifacts)
    progress(0.55, desc="Stage 5.5/8: Post-processing NEAR mesh...")
    run_stage_post_process_near_mesh(executor)
    
    yield executor.get_log_text(), work_dir, 0.55, executor.get_progress_text(), None, None
    
    # Stage 6: Ground alignment
    progress(0.6, desc="Stage 6/8: Ground plane estimation...")
    run_stage_ground_alignment(executor)
    
    yield executor.get_log_text(), work_dir, 0.60, executor.get_progress_text(), None, None
    
    # Stage 7: Manual alignment - PAUSE HERE (X-axis alignment integrated)
    progress(0.7, desc="Stage 7/8: Manual alignment required...")
    ok, manual_alignment_path = run_stage_manual_alignment(executor)
    
    executor.log("")
    executor.log("=" * 60, "STAGE")
    executor.log("MANUAL ALIGNMENT REQUIRED", "WARNING")
    executor.log("=" * 60, "STAGE")
    executor.log("")
    executor.log("👉 Switch to the 'Manual Alignment' tab above!", "INFO")
    executor.log("")
    executor.log("Steps:", "INFO")
    executor.log("  1. Click 'Start Alignment'", "INFO")
    executor.log("  2. Select 4 landmarks on SOURCE mesh in this EXACT order:", "INFO")
    executor.log("     a. HEAD (top of head/crown)", "INFO")
    executor.log("     b. LEFT SHOULDER", "INFO")
    executor.log("     c. RIGHT SHOULDER", "INFO")
    executor.log("     d. PELVIS MIDPOINT (center of pelvis/hip)", "INFO")
    executor.log("  3. Select the SAME 4 landmarks on TARGET mesh in the SAME order", "INFO")
    executor.log("  4. X-axis alignment will be computed automatically from shoulder vector", "INFO")
    executor.log("  5. Return here and click 'Continue After Alignment'", "INFO")
    executor.log("")
    executor.log("⏸️ Pipeline PAUSED - waiting for manual alignment...", "WARNING")
    
    yield executor.get_log_text(), work_dir, 0.80, executor.get_progress_text(), None, work_dir


def continue_after_alignment(progress=gr.Progress()):
    """Continue pipeline after manual alignment"""
    global executor, manual_alignment_path
    
    if executor is None:
        yield "[ERROR] No pipeline running. Start a new pipeline first.", "", 0, "", None
        return
    
    work_dir = str(executor.work_dir)
    
    # Check if manual alignment is complete
    # Priority: 1) executor.outputs['aligned_near'] (should be centered mesh), 2) manual_alignment_path, 3) search for files
    aligned_found = False
    
    # First, check if executor already has the aligned_near (centered mesh) set
    if executor.outputs.get('aligned_near') and os.path.exists(executor.outputs['aligned_near']):
        aligned_mesh = executor.outputs['aligned_near']
        executor.log(f"Using aligned mesh (centered): {os.path.basename(aligned_mesh)}", "SUCCESS")
        aligned_found = True
    elif manual_alignment_path and os.path.exists(manual_alignment_path):
        executor.outputs['aligned_near'] = manual_alignment_path
        executor.log(f"Found aligned mesh: {os.path.basename(manual_alignment_path)}", "SUCCESS")
        aligned_found = True
    else:
        # Check for centered mesh first (preferred), then any aligned mesh
        centered_files = list(executor.dirs['manual_aligned'].glob("*_centered.ply"))
        if centered_files:
            executor.outputs['aligned_near'] = str(centered_files[0])
            executor.log(f"Found centered mesh: {centered_files[0].name}", "SUCCESS")
            aligned_found = True
        else:
            # Check for any .ply file in manual_aligned directory
            aligned_files = list(executor.dirs['manual_aligned'].glob("*.ply"))
            if aligned_files:
                executor.outputs['aligned_near'] = str(aligned_files[0])
                executor.log(f"Found aligned mesh: {aligned_files[0].name}", "SUCCESS")
                aligned_found = True
            else:
                # Check in mesh dir (manual_alignment_v2 saves there by default)
                near_mesh = executor.outputs.get('near_mesh')
                if near_mesh:
                    near_dir = Path(near_mesh).parent
                    aligned_files = list(near_dir.glob("*aligned*.ply")) + list(near_dir.glob("manual_aligned*.ply"))
                    if aligned_files:
                        # Copy to our directory
                        shutil.copy(aligned_files[0], executor.dirs['manual_aligned'] / aligned_files[0].name)
                        executor.outputs['aligned_near'] = str(executor.dirs['manual_aligned'] / aligned_files[0].name)
                        executor.log(f"Found and copied aligned mesh: {aligned_files[0].name}", "SUCCESS")
                        aligned_found = True
    
    if not aligned_found:
        executor.log("No aligned mesh found. Using NEAR mesh directly.", "WARNING")
        near_mesh = executor.outputs.get('near_mesh')
        if near_mesh:
            executor.outputs['aligned_near'] = near_mesh
    
    executor.log("Continuing pipeline after manual alignment...", "SUCCESS")
    
    yield executor.get_log_text(), work_dir, 0.85, executor.get_progress_text(), None
    
    # Stage 8: AIX
    progress(0.9, desc="Stage 8/8: AIX estimation...")
    run_stage_aix(executor)
    
    yield executor.get_log_text(), work_dir, 0.95, executor.get_progress_text(), None
    
    # Complete
    executor.log("")
    executor.log("=" * 60)
    executor.log("PIPELINE COMPLETED SUCCESSFULLY", "SUCCESS")
    executor.log("=" * 60)
    executor.log(f"Results saved to: {work_dir}")
    executor.log("")
    executor.log("Output files in AIX results:")
    for f in executor.dirs['aix'].glob("*"):
        executor.log(f"  {f.name}", "FILE")
    
    yield executor.get_log_text(), work_dir, 1.0, executor.get_progress_text(), str(executor.dirs['aix'])


def open_folder(folder_path):
    """Open folder in file explorer"""
    if folder_path and os.path.exists(folder_path):
        subprocess.Popen(['xdg-open', folder_path])
        return f"Opened: {folder_path}"
    return "No folder to open"


# ============================================================
# BUILD GRADIO APP
# ============================================================

def create_app():
    """Create Gradio interface with integrated manual alignment"""
    
    with gr.Blocks(
        title="Scoliosis Assessment Pipeline",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
        css="""
        .log-box textarea {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
            font-size: 11px !important;
            line-height: 1.3 !important;
            background-color: #1a1a2e !important;
            color: #eee !important;
        }
        .progress-text {
            font-family: monospace;
            font-size: 14px;
            color: #6366f1;
        }
        .stage-complete {
            color: #22c55e;
        }
        .alignment-info {
            background: #fef3c7;
            border: 1px solid #f59e0b;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .point-input {
            margin: 5px 0;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # Scoliosis Assessment Pipeline
        
        **Automated pipeline with integrated manual alignment for scoliosis assessment**
        
        ---
        """)
        
        with gr.Tabs():
            # ========== PIPELINE TAB ==========
            with gr.TabItem("Pipeline", id="pipeline"):
                with gr.Row():
                    # Left column - Inputs and Controls
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Videos")
                        
                        far_video = gr.File(
                            label="FAR Video (full body, environment shot)",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"],
                            type="filepath"
                        )
                        
                        near_video = gr.File(
                            label="NEAR Video (close-up of back/torso)",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"],
                            type="filepath"
                        )
                        
                        gr.Markdown("---")
                        
                        start_btn = gr.Button("Start Pipeline", variant="primary", size="lg")
                        
                        gr.Markdown("---")
                        gr.Markdown("### Post-Alignment Controls")
                        
                        continue_btn = gr.Button("Continue After Alignment", variant="primary", size="lg")
                        
                        gr.Markdown("---")
                        
                        gr.Markdown("### Progress")
                        progress_text = gr.Textbox(
                            label="",
                            value="[░░░░░░░░░░░░░░░░░░░░] 0% - Ready",
                            interactive=False,
                            elem_classes=["progress-text"]
                        )
                        
                        progress_bar = gr.Slider(
                            minimum=0, maximum=1, value=0,
                            label="Overall Progress",
                            interactive=False
                        )
                        
                        working_dir = gr.Textbox(label="Working Directory", interactive=False)
                        results_dir = gr.Textbox(label="Results Directory", interactive=False)
                        
                        open_btn = gr.Button("Open Results Folder")
                    
                    # Right column - Logs
                    with gr.Column(scale=2):
                        gr.Markdown("### Pipeline Log")
                        log_output = gr.Textbox(
                            label="",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            elem_classes=["log-box"],
                            show_copy_button=True
                        )
            
            # ========== MANUAL ALIGNMENT TAB ==========
            with gr.TabItem("Manual Alignment", id="alignment"):
                from manual_alignment_automated import create_automated_alignment_interface
                alignment_components = create_automated_alignment_interface()
        
        gr.Markdown("""
        ---
        
        ### Pipeline Stages & Environments
        
        | Stage | Description | Environment |
        |-------|-------------|-------------|
        | 1 | Video → Frames | `vv_vggt` |
        | 2 | AI-based 3D Reconstruction | `vv_vggt` |
        | 3 | GLB → PLY | `vv_vggt` |
        | 4 | Denoising | `vv_denoise` |
        | 5 | Mesh Reconstruction | `vv_meshlab` |
        | 6 | Ground Alignment | `vv_gnd_estimate` |
        | 7 | **Manual Alignment** (web-based) | `vv_mesh_alignment` |
        | 8-9 | AIX Estimation | `vv_aix` |
        
        ---
        """)
        
        # ========== EVENT HANDLERS ==========
        
        # Pipeline tab handlers
        start_btn.click(
            fn=start_pipeline,
            inputs=[far_video, near_video],
            outputs=[log_output, working_dir, progress_bar, progress_text, results_dir],
        )
        
        continue_btn.click(
            fn=continue_after_alignment,
            inputs=[],
            outputs=[log_output, working_dir, progress_bar, progress_text, results_dir],
            show_progress=True
        )
        
        open_btn.click(
            fn=open_folder,
            inputs=[results_dir],
            outputs=[log_output]
        )
    
    return app


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app = create_app()
    
    print("\n" + "="*60)
    print("  SCOLIOSIS ASSESSMENT PIPELINE - GUI")
    print("="*60)
    print("\n  Environments used (from last_readme.md):")
    for stage, env in Config.ENVS.items():
        print(f"    {stage}: {env}")
    print("\n  Starting Gradio server...")
    print("  Open http://localhost:7860 in your browser")
    print("\n" + "="*60 + "\n")
    
    # Add static dir to allowed_paths so Gradio can serve viewer HTML files
    # Also add project root and working directory to be safe
    project_root = Path(__file__).parent.resolve()
    static_dir = project_root / "static"
    working_dir = project_root / "working_directory"
    
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)
    
    allowed_dirs = [str(static_dir), str(project_root), str(working_dir)]
    print(f"  Allowed paths: {allowed_dirs}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=allowed_dirs
    )

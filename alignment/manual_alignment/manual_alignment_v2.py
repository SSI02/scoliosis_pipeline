#!/usr/bin/env python3
"""
Manual Alignment Tool V2 - With Enhanced Point Picker

Improved version with more accessible and intuitive correspondence point selection.
"""

import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
import os
import copy
from enhanced_point_picker import EnhancedPointPicker


class ManualAlignmentV2:
    def __init__(self):
        self.source_mesh = None
        self.target_mesh = None
        self.source_points = []
        self.target_points = []
        self.transformation = None
        self.aligned_mesh = None
        
    def select_file(self, title):
        """File selector"""
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Mesh files", "*.ply *.stl *.obj"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        root.destroy()
        return path if path else None
    
    def load_mesh(self, path, name="mesh"):
        """Load mesh"""
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if not mesh.has_vertices():
                print(f"Error: {name} has no vertices!")
                return None
            
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            print(f"✓ {name} loaded:")
            print(f"  File: {os.path.basename(path)}")
            print(f"  Vertices: {len(mesh.vertices):,}")
            print(f"  Triangles: {len(mesh.triangles):,}")
            return mesh
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return None
    
    def pick_correspondence_points(self, mesh, mesh_name, num_points, is_source=True):
        """Pick correspondence points with enhanced GUI"""
        print("\n" + "="*70)
        if is_source:
            print(f"SELECT {num_points} LANDMARKS ON SOURCE MESH")
        else:
            print(f"SELECT CORRESPONDING {num_points} LANDMARKS ON TARGET MESH")
        print("="*70)
        
        if is_source:
            print("\n💡 SUGGESTED ANATOMICAL LANDMARKS:")
            print("  1️⃣  Top of head (vertex/crown)")
            print("  2️⃣  Left shoulder (acromion)")
            print("  3️⃣  Right shoulder (acromion)")
            print("  4️⃣  Center of pelvis/hip")
            if num_points > 4:
                print("  5️⃣  Left knee (optional)")
                print("  6️⃣  Right knee (optional)")
                print("  7️⃣  Sternum (optional)")
                print("  8️⃣  Spine point (optional)")
        else:
            print("\n⚠️  IMPORTANT: Select the SAME landmarks in the SAME ORDER!")
            print("   Point 1 on target = Point 1 on source")
            print("   Point 2 on target = Point 2 on source")
            print("   etc.")
        
        print("\n" + "="*70)
        input("\nPress Enter when ready to select points...")
        
        # Use enhanced picker
        picker = EnhancedPointPicker(
            mesh,
            num_points_needed=num_points,
            mesh_name=mesh_name
        )
        
        points = picker.pick_points_interactive()
        
        if points is None or len(points) < 3:
            print(f"\n❌ Failed: Need at least 3 points!")
            return None
        
        print(f"\n✓ {len(points)} points selected on {mesh_name}")
        return points
    
    def compute_rigid_transformation(self, source_pts, target_pts, allow_scale=True):
        """Compute optimal transformation (with or without scale)"""
        source_pts = np.array(source_pts)
        target_pts = np.array(target_pts)
        
        # Center
        source_center = source_pts.mean(axis=0)
        target_center = target_pts.mean(axis=0)
        
        source_centered = source_pts - source_center
        target_centered = target_pts - target_center
        
        # Compute scale if allowed
        if allow_scale:
            source_rms = np.sqrt(np.mean(np.sum(source_centered**2, axis=1)))
            target_rms = np.sqrt(np.mean(np.sum(target_centered**2, axis=1)))
            scale = target_rms / source_rms if source_rms > 0 else 1.0
            print(f"  Computed scale: {scale:.6f}")
        else:
            scale = 1.0
        
        # Scale source for rotation
        source_scaled = source_centered * scale
        
        # Covariance
        H = source_scaled.T @ target_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            print("  Note: Correcting reflection...")
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = target_center - scale * R @ source_center
        
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t
        
        return T, scale
    
    def compute_alignment_error(self, source_pts, target_pts, T):
        """Compute RMSE"""
        source_pts = np.array(source_pts)
        target_pts = np.array(target_pts)
        
        # Transform
        source_h = np.hstack([source_pts, np.ones((len(source_pts), 1))])
        source_t = (T @ source_h.T).T[:, :3]
        
        # Distances
        distances = np.linalg.norm(source_t - target_pts, axis=1)
        rmse = np.sqrt(np.mean(distances**2))
        
        return rmse, distances
    
    def visualize_correspondences(self):
        """Visualize correspondences before alignment"""
        print("\n" + "="*70)
        print("VISUALIZING CORRESPONDENCES (Before Alignment)")
        print("="*70)
        print("\nShowing how points will be matched...")
        print("  Blue mesh = Source (will be aligned)")
        print("  Orange mesh = Target (reference)")
        print("  Green lines = Correspondence connections")
        print("  Numbered spheres = Selected points")
        print("\nClose window when done viewing.")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Correspondences Preview",
            width=1600,
            height=900
        )
        
        # Source in blue
        source_vis = copy.deepcopy(self.source_mesh)
        source_vis.paint_uniform_color([0.2, 0.6, 1.0])
        vis.add_geometry(source_vis)
        
        # Target in orange
        target_vis = copy.deepcopy(self.target_mesh)
        target_vis.paint_uniform_color([1.0, 0.6, 0.2])
        vis.add_geometry(target_vis)
        
        # Lines
        line_points = []
        line_lines = []
        for i in range(len(self.source_points)):
            line_points.append(self.source_points[i])
            line_points.append(self.target_points[i])
            line_lines.append([i*2, i*2+1])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_lines)
        line_set.paint_uniform_color([0, 1, 0])
        vis.add_geometry(line_set)
        
        # Spheres with numbers
        for i, (src_pt, tgt_pt) in enumerate(zip(self.source_points, self.target_points)):
            # Source (blue)
            sphere_src = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere_src.paint_uniform_color([0, 0, 1])
            sphere_src.translate(src_pt)
            vis.add_geometry(sphere_src)
            
            # Target (red)
            sphere_tgt = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere_tgt.paint_uniform_color([1, 0, 0])
            sphere_tgt.translate(tgt_pt)
            vis.add_geometry(sphere_tgt)
        
        vis.run()
        vis.destroy_window()
    
    def visualize_alignment(self):
        """Visualize final alignment"""
        print("\n" + "="*70)
        print("VISUALIZING ALIGNMENT RESULT")
        print("="*70)
        print("\n  Blue mesh = Aligned source")
        print("  Orange mesh = Target")
        print("  Green/Red spheres = Aligned correspondence points")
        print("  Yellow lines = Residual errors")
        print("\nClose window when done viewing.")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Alignment Result",
            width=1600,
            height=900
        )
        
        # Aligned source
        aligned_vis = copy.deepcopy(self.aligned_mesh)
        aligned_vis.paint_uniform_color([0.2, 0.6, 1.0])
        vis.add_geometry(aligned_vis)
        
        # Target
        target_vis = copy.deepcopy(self.target_mesh)
        target_vis.paint_uniform_color([1.0, 0.6, 0.2])
        vis.add_geometry(target_vis)
        
        # Aligned correspondence points
        source_transformed = np.array([
            (self.transformation @ np.append(pt, 1))[:3]
            for pt in self.source_points
        ])
        
        for i, (src_aligned, tgt_pt) in enumerate(zip(source_transformed, self.target_points)):
            # Aligned source point (green)
            sphere_src = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere_src.paint_uniform_color([0, 1, 0])
            sphere_src.translate(src_aligned)
            vis.add_geometry(sphere_src)
            
            # Target point (red)
            sphere_tgt = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere_tgt.paint_uniform_color([1, 0, 0])
            sphere_tgt.translate(tgt_pt)
            vis.add_geometry(sphere_tgt)
            
            # Error line
            error = np.linalg.norm(src_aligned - tgt_pt)
            if error > 0.001:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector([src_aligned, tgt_pt])
                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                line_set.paint_uniform_color([1, 1, 0])
                vis.add_geometry(line_set)
        
        # Coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(coord)
        
        vis.run()
        vis.destroy_window()
    
    def save_results(self, output_prefix="manual_aligned_v2"):
        """Save results"""
        if self.aligned_mesh is None:
            return False
        
        # Save mesh
        mesh_path = f"{output_prefix}.ply"
        o3d.io.write_triangle_mesh(mesh_path, self.aligned_mesh)
        print(f"\n✓ Saved aligned mesh: {mesh_path}")
        
        # Save data
        data = {
            "transformation_matrix": self.transformation.tolist(),
            "scale": float(self.scale),
            "uniform_sizing": True,
            "scale_description": "Source mesh scaled to match target size" if self.scale != 1.0 else "No scaling needed",
            "source_points": [pt.tolist() for pt in self.source_points],
            "target_points": [pt.tolist() for pt in self.target_points],
            "num_correspondences": len(self.source_points),
            "rmse": float(self.rmse),
            "max_error": float(self.max_error),
            "individual_errors": self.errors.tolist(),
            "method": "manual_correspondence_v2_with_scale",
            "version": "v2"
        }
        
        json_path = f"{output_prefix}_transform.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved transformation: {json_path}")
        
        # Print scale summary
        if self.scale != 1.0:
            print(f"\n📏 Uniform Sizing Applied:")
            print(f"  Scale factor: {self.scale:.6f}")
            if self.scale > 1:
                print(f"  Source enlarged {self.scale:.2f}x to match target")
            else:
                print(f"  Source reduced {1/self.scale:.2f}x to match target")
            print(f"  Both meshes now at uniform size")
        
        return True
    
    def run(self):
        """Main workflow"""
        print("\n" + "="*70)
        print("MANUAL CORRESPONDENCE ALIGNMENT V2")
        print("Enhanced with Accessible Point Selection & Uniform Sizing")
        print("="*70)
        
        # Load source
        print("\n📂 Step 1: Load SOURCE mesh (will be aligned)")
        print("="*70)
        source_path = self.select_file("Select Source Mesh")
        if not source_path:
            return
        
        self.source_mesh = self.load_mesh(source_path, "Source")
        if not self.source_mesh:
            return
        
        # Load target
        print("\n📂 Step 2: Load TARGET mesh (reference)")
        print("="*70)
        target_path = self.select_file("Select Target Mesh")
        if not target_path:
            return
        
        self.target_mesh = self.load_mesh(target_path, "Target")
        if not self.target_mesh:
            return
        
        # Get number of points
        print("\n🔢 Step 3: Number of correspondence points")
        print("="*70)
        num_str = input("\nHow many points? (3-10, default=4): ").strip()
        num_points = int(num_str) if num_str else 4
        num_points = max(3, min(10, num_points))
        print(f"  Will select {num_points} correspondence points")
        
        # Pick source points
        print("\n📍 Step 4: Select landmarks on SOURCE")
        print("="*70)
        self.source_points = self.pick_correspondence_points(
            self.source_mesh,
            "SOURCE",
            num_points,
            is_source=True
        )
        if not self.source_points:
            return
        
        # Pick target points
        print("\n📍 Step 5: Select CORRESPONDING landmarks on TARGET")
        print("="*70)
        self.target_points = self.pick_correspondence_points(
            self.target_mesh,
            "TARGET",
            num_points,
            is_source=False
        )
        if not self.target_points:
            return
        
        # Verify counts
        if len(self.source_points) != len(self.target_points):
            print(f"\n❌ Error: Point count mismatch!")
            print(f"   Source: {len(self.source_points)}")
            print(f"   Target: {len(self.target_points)}")
            return
        
        # Show correspondences
        print("\n👁️  Step 6: Preview correspondences")
        print("="*70)
        self.visualize_correspondences()
        
        # Compute transformation
        print("\n🔬 Step 7: Compute transformation (with uniform sizing)")
        print("="*70)
        print("\nComputing optimal similarity transformation...")
        
        self.transformation, self.scale = self.compute_rigid_transformation(
            self.source_points,
            self.target_points,
            allow_scale=True  # Enable uniform sizing
        )
        
        print("\n✓ Transformation matrix:")
        print(self.transformation)
        
        if self.scale != 1.0:
            print(f"\n📏 Uniform Sizing:")
            print(f"  Scale factor: {self.scale:.6f}")
            if self.scale > 1:
                print(f"  Source enlarged {self.scale:.2f}x to match target size")
            else:
                print(f"  Source reduced {1/self.scale:.2f}x to match target size")
        
        # Compute errors
        self.rmse, self.errors = self.compute_alignment_error(
            self.source_points,
            self.target_points,
            self.transformation
        )
        self.max_error = self.errors.max()
        
        print(f"\n📊 Alignment Quality:")
        print(f"  RMSE:      {self.rmse:.6f} m ({self.rmse*1000:.2f} mm)")
        print(f"  Max error: {self.max_error:.6f} m ({self.max_error*1000:.2f} mm)")
        print(f"\n  Individual errors:")
        for i, err in enumerate(self.errors):
            print(f"    Point {i+1}: {err:.6f} m ({err*1000:.2f} mm)")
        
        # Apply transformation
        self.aligned_mesh = copy.deepcopy(self.source_mesh)
        self.aligned_mesh.transform(self.transformation)
        
        # Visualize result
        print("\n👁️  Step 8: Visualize result")
        print("="*70)
        self.visualize_alignment()
        
        # Save
        print("\n💾 Step 9: Save results")
        print("="*70)
        self.save_results()
        
        print("\n" + "="*70)
        print("✅ COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nYou can now use the aligned mesh with:")
        print("  python compute_metrics_from_ground_plane.py")
        print("="*70 + "\n")


if __name__ == "__main__":
    tool = ManualAlignmentV2()
    tool.run()


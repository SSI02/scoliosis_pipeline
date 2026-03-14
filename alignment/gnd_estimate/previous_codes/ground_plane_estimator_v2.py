#!/usr/bin/env python3
"""
Ground Plane Estimator V2 - With Enhanced Point Picker

Improved version with more accessible and intuitive point selection GUI.
"""

import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
import os
from enhanced_point_picker import EnhancedPointPicker


class GroundPlaneEstimatorV2:
    def __init__(self):
        self.mesh = None
        self.mesh_path = None
        self.picked_points = []
        self.plane_params = None
        
    def select_mesh_file(self):
        """Open file dialog to select mesh file"""
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select Mesh File",
            filetypes=[
                ("PLY files", "*.ply"),
                ("STL files", "*.stl"),
                ("OBJ files", "*.obj"),
                ("All files", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        root.destroy()
        
        if file_path:
            self.mesh_path = file_path
            return True
        return False
    
    def load_mesh(self):
        """Load mesh from selected file"""
        if not self.mesh_path:
            print("No mesh file selected!")
            return False
        
        try:
            self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
            if not self.mesh.has_vertices():
                print("Error: Mesh has no vertices!")
                return False
            
            if not self.mesh.has_vertex_normals():
                self.mesh.compute_vertex_normals()
            
            print(f"\n✓ Mesh loaded successfully:")
            print(f"  File: {os.path.basename(self.mesh_path)}")
            print(f"  Vertices: {len(self.mesh.vertices):,}")
            print(f"  Triangles: {len(self.mesh.triangles):,}")
            
            return True
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False
    
    def pick_ground_points(self):
        """Pick points on ground using enhanced picker"""
        if self.mesh is None:
            print("No mesh loaded!")
            return False
        
        print("\n" + "="*70)
        print("GROUND POINT SELECTION")
        print("="*70)
        print("\n🎯 Goal: Select 5-10 points on the GROUND/FLOOR")
        print("\n💡 Tips:")
        print("  • Select points spread across the floor area")
        print("  • Blue points = ground level (select these!)")
        print("  • Zoom in close for precise selection")
        print("  • At least 3 points required, 5-10 recommended")
        print("="*70)
        
        input("\nPress Enter when ready to start selection...")
        
        # Use enhanced picker
        picker = EnhancedPointPicker(
            self.mesh,
            num_points_needed=5,  # Suggest 5, but can do more/less
            mesh_name="Ground/Floor"
        )
        
        self.picked_points = picker.pick_points_interactive()
        
        if self.picked_points is None or len(self.picked_points) < 3:
            print(f"\n❌ Need at least 3 points, but only {len(self.picked_points) if self.picked_points else 0} selected.")
            return False
        
        print(f"\n✓ {len(self.picked_points)} ground points selected!")
        return True
    
    def estimate_plane_ransac(self, distance_threshold=0.01):
        """Estimate plane using RANSAC"""
        points_array = np.array(self.picked_points)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_array)
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        return plane_model, inliers
    
    def estimate_plane_least_squares(self):
        """Estimate plane using least squares"""
        points_array = np.array(self.picked_points)
        
        centroid = np.mean(points_array, axis=0)
        centered = points_array - centroid
        
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]
        
        # Ensure upward normal
        if normal[2] < 0:
            normal = -normal
        
        d = -np.dot(normal, centroid)
        return [normal[0], normal[1], normal[2], d]
    
    def estimate_ground_plane(self):
        """Estimate ground plane"""
        print("\n" + "="*70)
        print("ESTIMATING GROUND PLANE")
        print("="*70)
        
        if len(self.picked_points) >= 3:
            print("\nUsing RANSAC method for robust estimation...")
            self.plane_params, inliers = self.estimate_plane_ransac()
            print(f"  RANSAC inliers: {len(inliers)}/{len(self.picked_points)}")
        else:
            print("\nUsing least squares method...")
            self.plane_params = self.estimate_plane_least_squares()
        
        a, b, c, d = self.plane_params
        
        print(f"\n✓ Ground plane estimated:")
        print(f"  Equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
        print(f"  Normal vector: [{a:.6f}, {b:.6f}, {c:.6f}]")
        
        # Check if roughly horizontal
        if abs(c) > 0.9:
            print(f"  ✓ Plane is approximately horizontal (good!)")
        else:
            print(f"  ⚠️  Warning: Plane is tilted (normal Z component = {c:.3f})")
            print(f"      Expected ~1.0 for horizontal plane")
        
        return True
    
    def create_plane_mesh(self, size=2.0):
        """Create visualization mesh for plane"""
        if self.plane_params is None:
            return None
        
        a, b, c, d = self.plane_params
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        centroid = np.mean(self.picked_points, axis=0)
        
        # Create perpendicular vectors
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, np.array([0, 0, 1]))
        else:
            v1 = np.cross(normal, np.array([1, 0, 0]))
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(normal, v1)
        
        # Create quad
        vertices = [
            centroid - size * v1 - size * v2,
            centroid + size * v1 - size * v2,
            centroid + size * v1 + size * v2,
            centroid - size * v1 + size * v2,
        ]
        
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        plane_mesh.paint_uniform_color([0, 0.8, 0])
        plane_mesh.compute_vertex_normals()
        
        return plane_mesh
    
    def visualize_result(self):
        """Visualize mesh with plane"""
        print("\n" + "="*70)
        print("VISUALIZING RESULT")
        print("="*70)
        print("\nOpening visualization window...")
        print("  Gray mesh = Original mesh")
        print("  Red spheres = Selected ground points")
        print("  Green plane = Estimated ground plane")
        print("\nClose window when done viewing.")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Ground Plane Estimation Result",
            width=1600,
            height=900
        )
        
        # Mesh
        self.mesh.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(self.mesh)
        
        # Points
        for i, pt in enumerate(self.picked_points):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(pt)
            vis.add_geometry(sphere)
        
        # Plane
        plane_mesh = self.create_plane_mesh(size=1.5)
        if plane_mesh:
            vis.add_geometry(plane_mesh)
        
        # Coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(coord)
        
        vis.run()
        vis.destroy_window()
    
    def save_plane_parameters(self, output_path=None):
        """Save plane parameters"""
        if self.plane_params is None:
            print("No plane to save!")
            return False
        
        if output_path is None:
            base_name = os.path.splitext(self.mesh_path)[0]
            output_path = f"{base_name}_ground_plane.json"
        
        a, b, c, d = self.plane_params
        
        data = {
            "mesh_file": self.mesh_path,
            "plane_equation": {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "d": float(d),
                "equation": f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"
            },
            "normal_vector": [float(a), float(b), float(c)],
            "picked_points": [pt.tolist() for pt in self.picked_points],
            "num_points": len(self.picked_points),
            "version": "v2_enhanced_picker"
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Saved plane parameters to:")
        print(f"  {output_path}")
        
        return True
    
    def run(self):
        """Main workflow"""
        print("\n" + "="*70)
        print("GROUND PLANE ESTIMATOR V2")
        print("Enhanced with Accessible Point Selection")
        print("="*70)
        
        # Step 1: Select file
        print("\n📁 Step 1: Select mesh file...")
        if not self.select_mesh_file():
            print("No file selected. Exiting.")
            return
        
        # Step 2: Load mesh
        print("\n📂 Step 2: Loading mesh...")
        if not self.load_mesh():
            print("Failed to load mesh. Exiting.")
            return
        
        # Step 3: Pick points
        print("\n📍 Step 3: Select ground points...")
        if not self.pick_ground_points():
            print("Failed to select points. Exiting.")
            return
        
        # Step 4: Estimate plane
        print("\n🔬 Step 4: Estimating ground plane...")
        if not self.estimate_ground_plane():
            print("Failed to estimate plane. Exiting.")
            return
        
        # Step 5: Visualize
        print("\n👁️  Step 5: Visualizing result...")
        self.visualize_result()
        
        # Step 6: Save
        print("\n💾 Step 6: Saving results...")
        self.save_plane_parameters()
        
        print("\n" + "="*70)
        print("✅ COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nYou can now use the ground plane with:")
        print("  python mesh_alignment_with_ground_plane.py")
        print("  python compute_metrics_from_ground_plane.py")
        print("="*70 + "\n")


if __name__ == "__main__":
    estimator = GroundPlaneEstimatorV2()
    estimator.run()


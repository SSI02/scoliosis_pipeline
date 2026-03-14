#!/usr/bin/env python3
"""
Automated Ground Plane Estimation
---------------------------------
Estimates the ground plane of a 3D mesh automatically using RANSAC and PCA-based verification.
Robust to rotation (does not rely on coordinate axes) and body pose (handles bending).

Algorithm:
1. Load Mesh & Preprocess (Downsample, Compute Normals).
2. RANSAC Plane Segmentation to find candidate planes (largest planar surfaces).
3. PCA Analysis of the object (outliers of the plane) to find principal axes.
4. Verification:
   - PC3 (Left-Right axis) should be roughly perpendicular to Ground Normal.
   - Object Spread along Normal should be significant (Height > Thickness).
5. Orientation: Ensure normal points towards the object.
6. Save & Visualize.
"""

import open3d as o3d
import numpy as np
import json
import os
import copy
import argparse

class AutomatedGroundPlaneEstimator:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.mesh = None
        self.plane_model = None
        self.plane_inliers = None
        self.object_pcd = None  # The mesh without the ground plane
        
    def load_mesh(self):
        """Load and preprocess the mesh."""
        if not os.path.exists(self.mesh_path):
            print(f"Error: File not found: {self.mesh_path}")
            return False
            
        try:
            self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
            if not self.mesh.has_vertices():
                print("Error: Empty mesh.")
                return False
                
            # Compute normals if needed
            if not self.mesh.has_vertex_normals():
                self.mesh.compute_vertex_normals()
                
            print(f"Loaded mesh: {len(self.mesh.vertices)} vertices")
            return True
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False

    def compute_pca(self, pcd):
        """Compute PCA for a point cloud."""
        points = np.asarray(pcd.points)
        mean = np.mean(points, axis=0)
        centered_points = points - mean
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (ascending)
        # eigenvectors[:, 0] corresponds to smallest eigenvalue (PC3 - likely Left-Right/Thickness)
        # eigenvectors[:, 2] corresponds to largest eigenvalue (PC1 - likely Height/Spine)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors

    def estimate_ground_plane(self):
        """
        Find the best ground plane candidate.
        Strategy:
        1. Iteratively remove planes using RANSAC.
        2. For each candidate, check if it's a plausible floor.
        """
        if self.mesh is None:
            return False

        # Convert to PointCloud for RANSAC
        pcd = o3d.geometry.PointCloud()
        pcd.points = self.mesh.vertices
        
        # Downsample for speed (optional, but good for large meshes)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.01) # 1cm voxel
        if len(pcd_down.points) < 100:
            pcd_down = pcd # Revert if too small
            
        print(f"Processing {len(pcd_down.points)} points...")

        # Try to find top 3 largest planes
        candidates = []
        remaining_pcd = copy.deepcopy(pcd_down)
        
        for i in range(3): # Check top 3 planes
            if len(remaining_pcd.points) < 100:
                break
                
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=0.02, # 2cm tolerance
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < 100:
                break
                
            # Extract plane and object
            plane_cloud = remaining_pcd.select_by_index(inliers)
            object_cloud = remaining_pcd.select_by_index(inliers, invert=True)
            
            # Analyze this candidate
            score, details = self.evaluate_plane_candidate(plane_model, object_cloud)
            
            candidates.append({
                "model": plane_model,
                "inliers_count": len(inliers),
                "score": score,
                "details": details,
                "object_cloud": object_cloud # Store for visualization/verification
            })
            
            # Prepare for next iteration (remove this plane)
            remaining_pcd = object_cloud

        if not candidates:
            print("No planes found.")
            return False
            
        # Select best candidate
        # Sort by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        
        print("\nPlane Evaluation Results:")
        for i, c in enumerate(candidates):
            print(f"Candidate {i+1}: Score={c['score']:.2f}, Inliers={c['inliers_count']}")
            print(f"  Details: {c['details']}")
            
        self.plane_model = best["model"]
        self.object_pcd = best["object_cloud"]
        
        # Ensure normal points towards the object
        self.correct_normal_orientation(best["object_cloud"])
        
        return True

    def evaluate_plane_candidate(self, plane_model, object_cloud):
        """
        Evaluate if a plane is a good ground candidate.
        Returns a score (higher is better).
        """
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        if len(object_cloud.points) < 10:
            return -1.0, "Too few object points"

        # 1. PCA of the object
        evals, evecs = self.compute_pca(object_cloud)
        
        # PC3 (Smallest variance) -> Usually Left-Right axis or Thickness
        # PC1 (Largest variance) -> Usually Spine/Height
        
        pc3 = evecs[:, 0] # Smallest
        pc1 = evecs[:, 2] # Largest
        
        # CHECK 1: Orthogonality with PC3
        # The floor normal should be PERPENDICULAR to the Left-Right axis (PC3).
        # So dot product should be close to 0.
        dot_pc3 = abs(np.dot(normal, pc3))
        ortho_score = 1.0 - dot_pc3 # 1.0 if perfectly perpendicular
        
        # CHECK 2: Spread along Normal
        # Project object points onto the normal
        points = np.asarray(object_cloud.points)
        # Signed distances: ax + by + cz + d
        dists = np.dot(points, normal) + d
        spread = np.max(dists) - np.min(dists)
        
        # Heuristic: Floor should have a large spread (Height of person)
        # Wall should have small spread (Thickness of person)
        # We can normalize this by the max extent of the object
        max_extent = np.sqrt(evals[2]) * 4 # roughly 2 sigma
        spread_ratio = spread / (max_extent + 1e-6)
        
        # Combine scores
        # We prioritize Orthogonality to PC3 (avoids side walls)
        # And Spread (avoids back walls)
        
        final_score = (ortho_score * 2.0) + spread_ratio
        
        details = f"Ortho_PC3={ortho_score:.2f}, Spread={spread:.3f}m"
        
        return final_score, details

    def correct_normal_orientation(self, object_cloud):
        """Ensure normal points towards the object (upwards relative to floor)."""
        [a, b, c, d] = self.plane_model
        normal = np.array([a, b, c])
        
        points = np.asarray(object_cloud.points)
        # Calculate signed distances
        dists = np.dot(points, normal) + d
        
        # Check if majority of points are positive
        positive_count = np.sum(dists > 0)
        total_count = len(dists)
        
        if positive_count < total_count / 2:
            print("Flipping normal orientation...")
            self.plane_model = [-a, -b, -c, -d]

    def save_results(self, output_path=None):
        """Save plane parameters to JSON."""
        if self.plane_model is None:
            return
            
        if output_path is None:
            base = os.path.splitext(self.mesh_path)[0]
            output_path = f"{base}_automated_ground_plane.json"
            
        [a, b, c, d] = self.plane_model
        
        data = {
            "mesh_file": self.mesh_path,
            "plane_equation": {
                "a": float(a), "b": float(b), "c": float(c), "d": float(d),
                "equation": f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"
            },
            "normal_vector": [float(a), float(b), float(c)],
            "method": "automated_ransac_pca",
            "version": "v1.0"
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved results to {output_path}")

    def align_to_ground(self, target_axis=[0, -1, 0]):
        """
        Rotate the mesh so that the ground plane normal aligns with the target axis.
        Default target is [0, -1, 0] (Negative Y).
        """
        if self.mesh is None or self.plane_model is None:
            return None, None

        [a, b, c, d] = self.plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        target = np.array(target_axis)
        target = target / np.linalg.norm(target)
        
        # Calculate rotation matrix
        # Rotation from 'normal' to 'target'
        # Axis of rotation = cross(normal, target)
        # Angle = arccos(dot(normal, target))
        
        v = np.cross(normal, target)
        c_val = np.dot(normal, target)
        
        if np.linalg.norm(v) < 1e-6:
            # Already aligned or opposite
            if c_val > 0:
                R = np.eye(3)
            else:
                # Opposite direction, rotate 180 deg around any perpendicular axis
                if abs(normal[2]) < 0.9:
                    axis = np.cross(normal, [0, 0, 1])
                else:
                    axis = np.cross(normal, [1, 0, 0])
                axis = axis / np.linalg.norm(axis)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
        else:
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + kmat + kmat @ kmat * ((1 - c_val) / (s**2))
            
        # Apply rotation to mesh
        self.mesh.rotate(R, center=self.mesh.get_center())
        
        # Update plane model (Normal becomes target, d changes)
        # New normal is target
        # We need to find new d. 
        # Pick a point on old plane, rotate it, find new d.
        # Old point p: dot(p, normal) + d = 0
        center = self.mesh.get_center() # This center has changed? No, rotate uses center.
        # Actually, let's just re-estimate or transform the plane.
        # Easier: Transform a point on the plane.
        
        # But wait, we rotated the mesh around its center.
        # So the plane also rotates around that center.
        
        # Let's just return the rotation matrix so the caller can use it if needed
        # And update internal state
        
        # Update plane parameters
        # New normal is 'target'
        # New d?
        # Let P be a point on the old plane.
        # P_new = R @ (P - center) + center
        # New plane equation: dot(P_new, target) + d_new = 0
        
        # Find a point on the old plane
        # Closest point to center on old plane
        dist = np.dot(center, normal) + d
        P = center - dist * normal
        
        P_new = center - dist * target # Since we rotated normal to target around center
        
        d_new = -np.dot(P_new, target)
        
        self.plane_model = [target[0], target[1], target[2], d_new]
        
        return R

    def create_normal_arrow(self, origin=None, scale=1.0):
        """Create an arrow geometry representing the normal."""
        if self.plane_model is None:
            return None
            
        [a, b, c, d] = self.plane_model
        normal = np.array([a, b, c])
        
        if origin is None:
            # Default origin: Center of mesh projected onto plane
            center = self.mesh.get_center()
            dist = np.dot(center, normal) + d
            origin = center - dist * normal
            
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02 * scale,
            cone_radius=0.04 * scale,
            cylinder_height=0.8 * scale,
            cone_height=0.2 * scale
        )
        
        # Align arrow (default +Z) to normal
        # Arrow points in +Z by default
        arrow_normal = np.array([0, 0, 1])
        
        v = np.cross(arrow_normal, normal)
        c_val = np.dot(arrow_normal, normal)
        
        if np.linalg.norm(v) < 1e-6:
            if c_val < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
                arrow.rotate(R, center=[0,0,0])
        else:
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + kmat + kmat @ kmat * ((1 - c_val) / (s**2))
            arrow.rotate(R, center=[0,0,0])
            
        arrow.translate(origin)
        arrow.paint_uniform_color([1, 0, 0]) # Red
        
        return arrow

    def visualize(self):
        """Visualize the result."""
        if self.mesh is None or self.plane_model is None:
            return

        print("Visualizing...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Automated Ground Plane Estimation")
        
        # Mesh
        self.mesh.compute_vertex_normals()
        vis.add_geometry(self.mesh)
        
        # Plane
        [a, b, c, d] = self.plane_model
        
        # Create a large quad for the plane
        # Find a point on the plane
        # If c != 0, z = -(ax + by + d)/c
        # We need a robust way to generate the quad basis
        normal = np.array([a, b, c])
        
        # Create basis vectors
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        # Center of the mesh
        center = self.mesh.get_center()
        # Project center onto plane
        dist = np.dot(center, normal) + d
        plane_center = center - dist * normal
        
        size = 2.0 # 2 meters
        points = [
            plane_center - size*v1 - size*v2,
            plane_center + size*v1 - size*v2,
            plane_center + size*v1 + size*v2,
            plane_center - size*v1 + size*v2
        ]
        
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(points)
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        plane_mesh.paint_uniform_color([0, 1, 0]) # Green
        plane_mesh.compute_vertex_normals()
        
        vis.add_geometry(plane_mesh)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        
        vis.run()
        vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Automated Ground Plane Estimation")
    parser.add_argument("mesh_path", help="Path to the input mesh (PLY/OBJ/STL)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    estimator = AutomatedGroundPlaneEstimator(args.mesh_path)
    if estimator.load_mesh():
        if estimator.estimate_ground_plane():
            estimator.save_results()
            if not args.no_vis:
                estimator.visualize()

if __name__ == "__main__":
    main()

import gradio as gr
import open3d as o3d
import numpy as np
import os
import json
import tempfile
import shutil
from auto_gnd_estimate import AutomatedGroundPlaneEstimator

def estimate_and_visualize(mesh_file, save_dir, save_name):
    if mesh_file is None:
        return None, None, "Please upload a mesh file."
    
    try:
        input_path = mesh_file.name
        
        # Run estimation
        estimator = AutomatedGroundPlaneEstimator(input_path)
        if not estimator.load_mesh():
            return None, None, "Failed to load mesh."
            
        if not estimator.estimate_ground_plane():
            return None, None, "Failed to estimate ground plane."
            
        # ALIGNMENT: Rotate so Normal aligns with Negative Y (0, -1, 0)
        estimator.align_to_ground(target_axis=[0, -1, 0])
        
        # Get results (now aligned)
        [a, b, c, d] = estimator.plane_model
        mesh = estimator.mesh
        
        # VISUALIZATION
        # 1. Plane Geometry
        normal = np.array([a, b, c]) # Should be [0, -1, 0] roughly
        center = mesh.get_center()
        dist = np.dot(center, normal) + d
        plane_center = center - dist * normal
        
        # Basis vectors for plane quad
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        size = 2.0
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
        
        # 2. Normal Arrow
        arrow = estimator.create_normal_arrow(origin=plane_center, scale=0.5)
        
        # Merge for visualization
        combined_mesh = mesh + plane_mesh
        if arrow:
            combined_mesh += arrow
        
        # Save temp file for viewer
        temp_dir = tempfile.gettempdir()
        output_mesh_path = os.path.join(temp_dir, "visualization_result.obj")
        o3d.io.write_triangle_mesh(output_mesh_path, combined_mesh)
        
        # Prepare JSON result
        result_json = {
            "plane_equation": {
                "a": float(a), "b": float(b), "c": float(c), "d": float(d),
                "equation": f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"
            },
            "normal": [float(a), float(b), float(c)],
            "aligned": True,
            "alignment_target": [0, -1, 0]
        }
        
        log_text = f"Success!\nAligned Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0\nNormal: [{a:.4f}, {b:.4f}, {c:.4f}]"
        
        # SAVING LOGIC
        if save_dir and save_name:
            try:
                os.makedirs(save_dir, exist_ok=True)
                
                # Save Mesh (Aligned)
                save_mesh_path = os.path.join(save_dir, f"{save_name}.ply")
                o3d.io.write_triangle_mesh(save_mesh_path, mesh) # Save ONLY the aligned mesh, not the vis helpers
                
                # Save JSON
                save_json_path = os.path.join(save_dir, f"{save_name}.json")
                with open(save_json_path, 'w') as f:
                    json.dump(result_json, f, indent=2)
                    
                log_text += f"\n\nSaved to:\n{save_mesh_path}\n{save_json_path}"
            except Exception as e:
                log_text += f"\n\nError saving: {str(e)}"
        
        return output_mesh_path, result_json, log_text
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Define Gradio Interface
with gr.Blocks(title="Ground Plane Estimator") as app:
    gr.Markdown("# Automated Ground Plane Estimator")
    gr.Markdown("Upload a mesh. The system will estimate the ground plane, **align the mesh** so the ground normal points to **Negative Y**, and visualize it.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Mesh", file_types=[".ply", ".obj", ".stl"])
            
            gr.Markdown("### Save Options")
            save_dir_input = gr.Textbox(label="Save Directory", placeholder="/path/to/save", value="/home/stud2/Shikhar/vv/Scoliosis/Changes/gnd_estimate/output")
            save_name_input = gr.Textbox(label="Filename (without extension)", placeholder="aligned_mesh")
            
            run_btn = gr.Button("Estimate, Align & Save", variant="primary")
            
        with gr.Column():
            model_output = gr.Model3D(label="Visualization (Aligned Mesh + Green Plane + Red Normal)", clear_color=[1.0, 1.0, 1.0, 1.0])
            json_output = gr.JSON(label="Plane Parameters")
            log_output = gr.Textbox(label="Logs")
            
    run_btn.click(
        fn=estimate_and_visualize,
        inputs=[file_input, save_dir_input, save_name_input],
        outputs=[model_output, json_output, log_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

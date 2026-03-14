import gradio as gr
import open3d as o3d
import numpy as np
import os

def flip_mesh(mesh_file, operation, save_dir, output_filename):
    if mesh_file is None:
        return "Error: No file uploaded."
    
    try:
        input_path = mesh_file.name
        mesh = o3d.io.read_triangle_mesh(input_path)
        if mesh.is_empty():
            # Try reading as point cloud if mesh fails
            pcd = o3d.io.read_point_cloud(input_path)
            if pcd.is_empty():
                return "Error: Could not read file as mesh or point cloud."
            geometry = pcd
            points = np.asarray(pcd.points)
        else:
            geometry = mesh
            points = np.asarray(mesh.vertices)
        
        # Apply operation
        new_points = points.copy()
        if operation == "Swap X and Z":
            # (x, y, z) -> (z, y, x)
            new_points[:, 0] = points[:, 2]
            new_points[:, 2] = points[:, 0]
        elif operation == "Negate X":
            new_points[:, 0] = -points[:, 0]
        elif operation == "Negate Z":
            new_points[:, 2] = -points[:, 2]
        elif operation == "Negate X and Z":
            new_points[:, 0] = -points[:, 0]
            new_points[:, 2] = -points[:, 2]
        
        # Update geometry
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            geometry.vertices = o3d.utility.Vector3dVector(new_points)
            geometry.compute_vertex_normals()
        else:
            geometry.points = o3d.utility.Vector3dVector(new_points)
            
        # Determine save path
        if not save_dir:
            save_dir = os.path.dirname(input_path)
        os.makedirs(save_dir, exist_ok=True)
        
        if not output_filename:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_filename = f"{base_name}_flipped.ply"
        
        if not output_filename.lower().endswith(('.ply', '.obj', '.stl')):
             output_filename += ".ply"
             
        out_path = os.path.join(save_dir, output_filename)
        
        # Save
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(out_path, geometry, write_ascii=True)
        else:
            o3d.io.write_point_cloud(out_path, geometry, write_ascii=True)
            
        return f"Success! Saved to: {out_path}"

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks(title="Mesh Axis Flipper") as demo:
    gr.Markdown("# Mesh Axis Flipper")
    gr.Markdown("Upload a mesh and choose an operation to transform its coordinates.")
    
    with gr.Row():
        file_input = gr.File(label="Upload Mesh (.ply, .obj, .stl)")
        
    with gr.Row():
        op_input = gr.Radio(
            choices=["Swap X and Z", "Negate X", "Negate Z", "Negate X and Z"],
            value="Swap X and Z",
            label="Operation"
        )
    
    with gr.Row():
        save_dir_input = gr.Textbox(label="Save Directory (Optional, defaults to input dir)", placeholder="/path/to/save")
        out_name_input = gr.Textbox(label="Output Filename (Optional)", placeholder="flipped_mesh.ply")
        
    run_btn = gr.Button("Flip and Save", variant="primary")
    output_text = gr.Textbox(label="Status")
    
    run_btn.click(
        fn=flip_mesh,
        inputs=[file_input, op_input, save_dir_input, out_name_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866)

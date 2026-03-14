import gradio as gr
import plotly.graph_objects as go
import trimesh
import numpy as np
import json
import os

def load_mesh(file_obj):
    """
    Loads a mesh or point cloud from a file object or path.
    """
    if file_obj is None:
        return None
    
    try:
        # Handle Gradio file object (which might be a path string or a file-like object)
        if isinstance(file_obj, str):
            filepath = file_obj
        elif hasattr(file_obj, 'name'):
            filepath = file_obj.name
        else:
            raise ValueError("Invalid file object")

        mesh = trimesh.load(filepath, force='mesh')
        
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None

def parse_plane(file_obj):
    """
    Parses a ground plane from a JSON file.
    Expected formats: 
    - {'plane': [a, b, c, d]}
    - {'coefficients': [a, b, c, d]}
    - {'normal': [nx, ny, nz], 'point': [px, py, pz]}
    - {'plane_equation': {'a': a, 'b': b, 'c': c, 'd': d}}
    """
    if file_obj is None:
        return None
    
    try:
        if isinstance(file_obj, str):
            filepath = file_obj
        elif hasattr(file_obj, 'name'):
            filepath = file_obj.name
        else:
            raise ValueError("Invalid file object")

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'plane' in data:
            return np.array(data['plane'])
        elif 'coefficients' in data:
            return np.array(data['coefficients'])
        elif 'normal' in data and 'point' in data:
            normal = np.array(data['normal'])
            point = np.array(data['point'])
            # Plane equation: ax + by + cz + d = 0
            # normal . (x - point) = 0 => normal . x - normal . point = 0
            # d = - normal . point
            d = -np.dot(normal, point)
            return np.concatenate((normal, [d]))
        elif 'plane_equation' in data:
            pe = data['plane_equation']
            return np.array([pe['a'], pe['b'], pe['c'], pe['d']])
        
        return None
    except Exception as e:
        print(f"Error parsing plane: {e}")
        return None

def create_plane_mesh(plane_coeffs, centroid, size=2.0):
    """
    Creates a visual representation of the plane centered at 'centroid'.
    """
    a, b, c, d = plane_coeffs
    
    # Create a grid of points
    x = np.linspace(centroid[0] - size, centroid[0] + size, 10)
    z = np.linspace(centroid[2] - size, centroid[2] + size, 10)
    X, Z = np.meshgrid(x, z)
    
    # Calculate Y based on plane equation: ax + by + cz + d = 0 => y = -(ax + cz + d) / b
    if abs(b) > 1e-6:
        Y = -(a * X + c * Z + d) / b
    elif abs(c) > 1e-6:
        # If b is near zero, try z = -(ax + by + d) / c
        # Re-mesh for X, Y
        y = np.linspace(centroid[1] - size, centroid[1] + size, 10)
        X, Y = np.meshgrid(x, y)
        Z = -(a * X + b * Y + d) / c
    else:
        # If b and c are near zero, plane is vertical (x = constant)
        # Re-mesh for Y, Z
        y = np.linspace(centroid[1] - size, centroid[1] + size, 10)
        Y, Z = np.meshgrid(y, z)
        X = -(b * Y + c * Z + d) / a

    return X, Y, Z

def visualize(mesh_file, plane_file):
    fig = go.Figure()
    
    # Load Mesh
    mesh = load_mesh(mesh_file)
    if mesh:
        # Subsample for performance if too large
        verts = mesh.vertices
        if len(verts) > 50000:
            indices = np.random.choice(len(verts), 50000, replace=False)
            verts = verts[indices]
            
        fig.add_trace(go.Scatter3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.5),
            name='Mesh'
        ))
        
        centroid = mesh.centroid
        mesh_size = np.max(mesh.extents)
    else:
        centroid = np.array([0, 0, 0])
        mesh_size = 1.0

    # Load Plane
    plane_coeffs = parse_plane(plane_file)
    if plane_coeffs is not None:
        X, Y, Z = create_plane_mesh(plane_coeffs, centroid, size=mesh_size)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.5,
            colorscale='Viridis',
            showscale=False,
            name='Ground Plane'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="Mesh and Ground Plane Visualization"
    )
    
    return fig

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Mesh and Ground Plane Visualization")
    
    with gr.Row():
        mesh_input = gr.File(label="Upload Mesh (PLY)", file_count="single", type="filepath")
        plane_input = gr.File(label="Upload Ground Plane (JSON)", file_count="single", type="filepath")
    
    plot_output = gr.Plot(label="Visualization")
    
    btn = gr.Button("Visualize")
    btn.click(fn=visualize, inputs=[mesh_input, plane_input], outputs=plot_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

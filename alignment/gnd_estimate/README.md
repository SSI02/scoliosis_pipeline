# Ground Plane Estimation & Alignment

This directory contains tools for automatically estimating the ground plane of 3D meshes and point clouds, performing alignment, and visualizing the results. The core logic uses RANSAC for plane segmentation and PCA (Principal Component Analysis) for verification.

## Files Description

### 1. `auto_gnd_estimate.py`
This is the **core library** file containing the `AutomatedGroundPlaneEstimator` class. 
- **Purpose**: It encapsulates the logic for loading a mesh, estimating the ground plane using RANSAC with PCA-based heuristic verification, and aligning the mesh.
- **Key Features**:
    - **RANSAC Plane Segmentation**: Identifies candidate planes.
    - **PCA Verification**: Checks if the "object" (non-plane points) is oriented correctly relative to the plane (e.g., perpendicular to the "thickness" axis).
    - **Alignment**: Can rotate the mesh so the ground normal points in a specific direction (default: negative Y).
    - **Visualization**: Includes Open3D-based visualization methods.
- **Usage**: Can be imported as a module or run as a standalone CLI script.
  ```bash
  python auto_gnd_estimate.py input_mesh.ply
  ```

### 2. `pcd_mesh_gnd_estimate.py` (Advanced Gradio App)
This script provides a comprehensive **Gradio Web Interface** capable of handling both **Point Clouds (.ply, .pcd, .txt)** and **Meshes (.obj, .ply, .stl)**.
- **Visualization**: Uses **Plotly** to render an interactive 3D scene directly in the browser. It visualizes the points/mesh, the estimated ground plane (as a green transparent quad), and the normal vector.
- **Features**:
    - Automatically distinguishes between mesh and point cloud inputs.
    - Estimates normals for point clouds if missing.
    - Aligns the object to the ground plane (Negative Y).
    - Saves the aligned file and JSON parameters.
- **Usage**:
  ```bash
  python pcd_mesh_gnd_estimate.py
  ```

### 3. `gradio_app.py` (Simple Gradio App)
A simpler **Gradio Web Interface** focused on **Meshes** only.
- **Visualization**: Uses Gradio's native `Model3D` component. It creates a temporary `.obj` file merging the aligned mesh, the ground plane, and an arrow indicator for visualization.
- **Features**:
    - Upload Mesh -> Estimate -> Align -> Visualize.
    - Saves the aligned mesh and JSON parameters.
- **Usage**:
  ```bash
  python gradio_app.py
  ```

## Dependencies
Ensure you have the following Python packages installed:

```bash
pip install open3d numpy gradio plotly
```

## How It Works
1. **Preprocessing**: The input is downsampled (if dense) and normals are computed.
2. **Candidate Generation**: RANSAC is used to find the largest planar surfaces in the data.
3. **Verification**: Candidates are evaluated based on the shape of the remaining "object" points.
    - The code assumes the object (e.g., a human body) should have its "thickness" axis (PC3 from PCA) roughly perpendicular to the floor normal.
    - It also checks for significant "spread" (height) aong the normal to distinguish floors from walls.
4. **Alignment**: The object is rotated so the ground plane is flat (normal aligns with [0, -1, 0]).

## Usage Examples

**Running the Core CLI:**
```bash
# Estimate and visualize
python auto_gnd_estimate.py data/scoliosis_mesh.ply

# Estimate without visualization
python auto_gnd_estimate.py data/scoliosis_mesh.ply --no-vis
```

**Running the Web Interface:**
```bash
# Start the advanced interface (Recommended for Point Clouds & Meshes)
python pcd_mesh_gnd_estimate.py
# Access at http://0.0.0.0:7860
```

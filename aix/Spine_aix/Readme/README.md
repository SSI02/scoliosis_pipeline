# Spine AIX Analysis - Mid-Sagittal Plane Estimation

This module (`spine_aix.py`) is designed to robustly estimate the mid-sagittal plane of a 3D human back scan (mesh or point cloud). It is a core component of the Scoliosis-2 pipeline, enabling the definition of a reference frame for further asymmetry analysis.

The script employs two primary strategies:
1.  **Slice-Midpoint Plane**: Cuts the mesh into horizontal slices (along the spine axis), calculates the midpoint of the torso at each slice, and fits a plane to these centroids.
2.  **Symmetry Optimization**: Refines the plane by maximizing the reflection symmetry of the point cloud across the plane.

## Key Features

*   **Adaptive Loading**: Handles large meshes by downsampling relative to the bounding box diagonal.
*   **Relative Parameters**: All internal parameters (voxel size, slice thickness) are scaled relative to the mesh size, preventing issues with unit scaling.
*   **Robustness**: Includes fallback mechanisms and safety checks for degenerate inputs.
*   **Visualization**: Auto-generates PNG visualizations of the estimated plane and midline.

## Usage

```bash
python spine_aix.py /path/to/mesh.ply
```

## Arguments

*   `mesh_path`: Absolute path to the input `.ply` mesh file.

## Method Details

### 1. Slice-Midpoint Method
This method is geometric and heuristic-based.
*   **Slicing**: The mesh is sliced along the Z-axis (spine axis).
*   **Midpoints**: For each slice, the left and right boundaries (percentiles) are found, and the midpoint is calculated.
*   **Plane Fit**: SVD (Singular Value Decomposition) is used to fit a plane to the collection of midpoints.

### 2. Symmetry Optimization
This method is an optimization-based refinement.
*   **Initialization**: Strictly initialized using the result from the Slice-Midpoint method.
*   **Objective**: Maximizes the overlap between the original point cloud and its reflection across the candidate plane.
*   **Cost Function**: Uses a trimmed Nearest Neighbor (NN) loss to be robust against outliers and asymmetric deformities (like a rib hump).

## Outputs

All outputs are saved in the same directory as the input mesh or in an `outputs/` subdirectory if configured.

### JSON Metadata
*   `*_mid_sagittal_slice_midpoint.json` / `*_mid_sagittal_symmetry_optimized.json`: Detailed JSON files containing a list of metrics. Each item includes:
    *   **Quantity**: Name of the parameter (e.g., "Plane Normal", "Midline Mean X").
    *   **Definition**: Description of the parameter.
    *   **Value**: The calculated value.
*   `*_midline_stats.json`: (Legacy) Statistical distribution of detected midline points.

### Visualizations
*   `*_mid_sagittal_slice_midpoint_2d.png` / `*_mid_sagittal_symmetry_optimized_2d.png`: **2D Frontal Projection (X-Z)** plot with a clear legend:
    *   **Red Line**: The computed Mid-Sagittal Plane.
    *   **Green Dots**: Estimate midline points.
    *   **Grey Cloud**: Input point cloud background.
*   `*_3d.png`: (Optional) 3D screenshot from Open3D visualizer.

## Dependencies
*   `numpy`
*   `open3d`
*   `scipy`
*   `matplotlib`

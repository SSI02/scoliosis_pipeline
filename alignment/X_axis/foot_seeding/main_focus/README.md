# Scale-Invariant Skeleton Leg Aligner

This tool provides a **Gradio-based GUI** for aligning 3D human scans (meshes or point clouds) by detecting leg structures and identifying foot placement on the ground. It uses a scale-invariant approach (parameters relative to object height) to be robust across different capture units (meters, millimeters, etc.).

## 🚀 Features

-   **Scale-Invariant Processing**: All major parameters are defined as fractions of the object's height ($H$), making the tool adaptable to different unit systems automatically.
-   **Robust Foot Seeding**: Uses a "negative mold" approach to find where feet touch the ground by identifying holes in a projected ground slab.
-   **Geodesic Pruning**: Filters out disconnected skeletal components (like hands touching knees) by computing geodesic distances from the detected foot seeds.
-   **Skeletonization**: Voxelizes a specific Region of Interest (ROI) and extracts a 1-pixel wide skeleton.
-   **Automatic Alignment**: aligns the mesh/cloud so that the vector connecting the two feet becomes the X-axis, correcting for rotation.
-   **Interactive Visualization**: Uses Plotly for 3D inspection of the skeleton, axis, and ground connections directly in the browser.

## 📦 Requirements

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
-   `numpy`, `scipy`
-   `open3d` (Geometry processing)
-   `gradio` (GUI)
-   `scikit-image` (Skeletonization)
-   `scikit-learn` (DBSCAN Clustering, PCA)
-   `plotly` (Visualization)

## 🏃 Usage

Run the script directly:

```bash
python skeleton_pca.py
```

This will launch a local server (default: `http://0.0.0.0:7865`). Open the link in your browser to access the interface.

## 🎛️ Parameters Guide

The interface allows you to tune various pipeline steps. Most values are fractions of the object's bounding box height ($H$).

### 1. Input & Ground Plane
-   **Upload Point Cloud/Mesh**: Supports `.ply`, `.pcd`, `.xyz`, `.obj`.
-   **Ground Plane JSON** (Optional): A JSON file defining the ground plane equation. If omitted, the tool attempts to auto-estimate the ground using RANSAC.

### 2. Relative Parameters
-   **Voxel size (fraction of H)**: Controls the resolution of the voxel grid used for skeletonization.
    -   *Default*: `0.01` (1% of height).
    -   *Smaller*: finer skeleton, slower.
    -   *Larger*: coarser skeleton, faster, may merge legs.
-   **Ground tolerance (fraction of H)**: Distance from the plane to consider points as "touching the ground".

### 3. Region of Interest (ROI)
Defines the vertical slice of the body to analyze for leg detection.
-   **Low fraction**: Start of ROI relative to feet (min Y). *Default*: `0.05` (start slightly above ankles).
-   **High fraction**: End of ROI. *Default*: `0.25` (end below knees/hips).

### 4. Clustering & Filtering
-   **DBSCAN eps (fraction of H)**: Epsilon neighbor distance for clustering skeleton points into branches.
-   **DBSCAN min samples**: Min points to form a cluster.
-   **Verticality threshold**: Ratio of 1st/2nd principal components. Higher values strictly enforce linear (vertical) leg branches.
-   **Geodesic Threshold (m)**: Max geodesic (walking) distance from foot seeds to keep a skeleton point. **Crucial for removing hands.**

## 🧠 Pipeline Overview

1.  **Preprocessing**: Loads data, computes height ($H$), and estimates/loads ground plane.
2.  **Foot Seeding (Hole Detection)**:
    -   Slices a thin slab (e.g., 5mm) near the ground.
    -   Projects points to 2D and performs morphological operations to find "holes" (where the feet are invalidating the empty space).
    -   Selects the best 2 seeds based on area, separation, and position.
3.  **Voxelization & Geodesic Calculation**:
    -   Voxelizes the full cloud.
    -   Computes geodesic distance map from the "Foot Seeds" to every voxel.
4.  **ROI Skeletonization**:
    -   Extracts the ROI slab ($0.05H$ to $0.25H$).
    -   Skeletonizes this slice using `skimage`.
5.  **Pruning**:
    -   Removes skeleton branches that are "too far" geodesically from seeds (removes hands).
    -   Clusters remaining points using DBSCAN.
6.  **Leg Fitting**:
    -   Selects top 2 clusters (legs).
    -   Projects clusters to ground and fits circles to estimate precise centers.
7.  **Alignment**:
    -   Computes vector between foot centers.
    -   Rotates space to align this vector with +X axis.

## 📂 Outputs

Files are saved to the specified `Save directory` (default: `skeleton_out/`):

-   `aligned_mesh.ply`: The final aligned mesh/cloud.
-   `foot_seeding_slab.ply`: Visual debug of the ground slab and detected seeds.
-   `roi_skeleton.ply`: The raw skeleton of the ROI.
-   `skeleton_filtered.ply`: The skeleton after geodesic pruning.
-   `ground_intersection_points.ply`: Calculated foot centers on the ground.
-   `rotated_full.ply`: The rotated point cloud (before centering shift).
-   `circle_fit_debug.json`: Debug info on circle fitting success/failure.
-   `.npy` files: `rotation_matrix.npy`, `origin_point.npy`, `centroid_shift.npy` for reuse.

## 🔧 Troubleshooting

-   **"No skeleton points extracted"**:
    -   Increase **Voxel size**.
    -   Increase **Geodesic Threshold** (points might be disconnected).
    -   Check if the **ROI** (Low/High fractions) actually covers the legs.
-   **"Slice produced zero points"**:
    -   The object might be flying too high above the ground plane.
    -   Check the input file scale.
-   **Legs detected but mixed with hands**:
    -   Decrease **Geodesic Threshold**.
    -   Ensure **Foot Seeding** is working (check `foot_seeding_slab.ply`).

# X-Axis Alignment

This directory contains the algorithms for verifying and correcting the **X-axis alignment** of the 3D scans.

## Objective

The goal is to align the **X-axis** with the **Left-Right axis** of the subject. 
This is achieved by identifying the **foot positions** on the ground plane. The vector connecting the centers of the two feet defines the precise X-axis direction.

## Implementation

The robust implementation for this logic is located in the `foot_seeding` subdirectory.

### Key Components

-   **Foot Seeding (`foot_seeding/`)**: 
    -   Contains the "Scale-Invariant Skeleton Leg Aligner".
    -   **Path**: `foot_seeding/main_focus/skeleton_pca.py`
    -   **Algorithm**:
        1.  **Ground Slab Extraction**: Identifies where the subject touches the ground.
        2.  **Foot Seeding**: Finds the "holes" in the ground slab to pinpoint foot locations.
        3.  **Skeletonization**: Extracts 3D skeleton lines for the legs using a geodesic distance map from the foot seeds (isolating legs from hands).
        4.  **Circle Fitting**: Projects leg clusters to the ground and fits circles to find precise foot centers.
        5.  **Alignment**: Rotates the mesh so the foot-to-foot vector aligns with the X-axis.

## Usage

To run the alignment tool, navigate to the main script location:

```bash
cd foot_seeding/main_focus
python skeleton_pca.py
```

See `foot_seeding/main_focus/README.md` for detailed parameters and troubleshooting.

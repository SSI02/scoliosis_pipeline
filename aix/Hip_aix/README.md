# Hip AIX (Bending Posture)

## Overview
`hip_aix_bend.py` calculates the **horizontal offset ($\Delta x$)** between the neck center and the hip center for a subject in a bending posture (Adam's Forward Bend Test). This measurement helps quantify asymmetry in the spinal column's alignment relative to the hips.

## Axis Convention
The script assumes the following coordinate system (standard for this pipeline):
- **X-axis**: Left-Right (from left shoulder to right shoulder).
- **Y-axis**: Depth (High value towards upper body/back, low value towards foot/floor).
- **Z-axis**: Spine Axis (High value towards hips, low value towards neck).

## Logic
1. **Z-Axis Slicing**: The mesh is sliced along the Z-axis (along the spine).
2. **Hip Detection**: Identifies the hip center by searching the top 20% of the Z-range (high Z).
3. **Neck Detection**: Identifies the neck center using a robust **trend-based detection** algorithm:
   - Scans slices from Neck (Low Z) towards Hips.
   - Looks for the "shoulder expansion signature": a stable narrow region followed by a consistent increase in width (shoulders).
   - This handles forward-bend postures where the mid-back might be narrower than the neck.
   - *Fallback*: If no shoulder expansion is found (e.g., cropped meshes), defaults to the narrowest slice in the upper spine region.
4. **Calculations**:
   - Computes the signed and absolute horizontal offset ($\Delta x = \text{Neck}_x - \text{Hip}_x$).
   - Calculates the 3D angle between the Neck-Hip vector and the Z-axis.
   - Calculates the projected angle (on the XZ plane) relative to the Z-axis.

## Usage

```bash
python hip_aix_bend.py --mesh <path_to_mesh.ply> [options]
```

### Arguments
- `--mesh`: (Required) Path to the input `.ply` or `.obj` mesh file.
- `--slice-frac`: (Optional) Slice thickness as a fraction of the Z-range (default: `0.02`).
- `--no-vis`: (Optional) Disable the 3D interactive visualization (and skip PNG saving).

### Outputs
1. **Console Output**: Prints the calculated centers, $\Delta x$, and angles.
2. **JSON Statistics**: Automatically saves `*_hip_stats.json`. This file contains a structured table of metrics, where each entry has:
   - **Quantity**: Name of the metric (e.g., "delta_x").
   - **Definition**: Brief explanation of what the metric represents.
   - **Value**: The estimated numerical value or coordinate.
3. **Visualization**:
   - **Interactive**: Opens an 3D visualization window (unless `--no-vis` is used).
   - **PNG Snapshot**: Automatically saves `*_neck_hip_view.png`. This is a **2D Top-View (X-Z)** plot with a clear legend indicating:
     - **Green Point**: Hip Center
     - **Red Point**: Neck Center
     - **Blue Dashed Line**: Connection vector
     - **Grey Points**: Background mesh showing the torso shape

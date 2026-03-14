# Hump AIX Analysis

This folder contains the `hump_aix.py` script, a specialized tool for analyzing rib hump asymmetry in 3D torso scans. It utilizes a **Torso-Centric Coordinate System** to derive clinically relevant metrics independent of patient posture or camera orientation.

## 🚀 Script: `hump_aix.py`

This script performs a comprehensive analysis of the posterior torso, calculating volume asymmetry, extracting ridge lines, and computing specific angles to quantify the rib hump.

### Key Features
*   **Depth Map Generation**: Visualizes the posterior surface topography.
*   **Ridge Line Extraction**: Identifies and compares the maximum posterior prominence on the left and right sides along the spine.
*   **Asymmetry Metrics**: Calculates Volume Asymmetry and Pointwise Statistics (RMS, Mean, Max differences).
*   **Angle Calculations**:
    *   **Hump Plane Angle**: The local tilt of the rib hump relative to the general back plane.
*   **Visualization**: Generates 2D plots for all metrics and an interactive 3D visualization of the analysis.

---

## 💻 Usage

```bash
python hump_aix.py <mesh_path> <midline_json> [out_prefix] [--save-vis]
```

### Arguments
*   `mesh_path`: **(Required)** Path to the input 3D mesh file (e.g., `.ply`, `.obj`).
*   `midline_json`: **(Required)** Path to the JSON file containing Mid-Sagittal Plane data (must include `plane_equation` or `normal_vector`).
*   `out_prefix`: **(Optional)** Prefix for output filenames.
    *   *Default*: Uses the input mesh filename in the same directory (e.g., `.../mesh_name_rib`).
*   `--save-vis`: **(Optional)** Flag to save a screenshot of the interactive 3D visualization window.

---

## 📐 Coordinate System & Conventions

The analysis assumes the input mesh is pre-aligned to a standard torso frame:

| Axis | Direction | Interpretation |
| :--- | :--- | :--- |
| **X** | **Left ↔ Right** | Lateral axis. (Left < 0, Right > 0) |
| **Y** | **Posterior ↔ Anterior** | Depth axis. **Positive Y** represents the **Posterior Bulge** (sticking out of the back). |
| **Z** | **Caudal ↔ Cranial** | Longitudinal Spine axis. (Low = Neck, High = Hips) |

*   **Transverse Plane**: Defined as the plane perpendicular to the Z-axis (Spine).
*   **Back Plane**: The general plane of the back, assumed to have a normal roughly along the Y-axis.

---

## 📊 Outputs

The script generates several files to document the analysis:

### 1. Quantitative Data (`*_rib_hump_stats.json`)
A JSON file containing a list of calculated metrics. Each entry provides:
*   `quantity`: The name of the metric (e.g., `volume_asymmetry`, `transverse_angle`).
*   `definition`: A clear description of what the metric represents.
*   `estimated_value`: The calculated numerical value.

**Metrics included:**
*   `volume_asymmetry`
*   `rib_hump_angle`
*   `mean_diff`, `std_diff`, `rms_diff`, `mean_abs_diff`, `max_abs_diff` (Pointwise asymmetry stats)
*   `num_slices`

### 2. Visualizations (Images)
*   **Depth Map** (`*_depth_map.png`): Heatmap of posterior depth (Y) across the back (X vs Z).
*   **Ridge Lines** (`*_ridge_lines.png`): Comparison of Left (Blue) and Right (Red) max-depth profiles along the spine.
*   **Asymmetry Curve** (`*_hump_diff_curve.png`): Plot of the difference (Left - Right) along the spine.
*   **Side View Angle** (`*_side_view_angle.png`): Profile view showing the tilt of the hump plane relative to the back plane.
*   **3D Visualization** (`*_roi_vis.png`): (Optional) Screenshot of the 3D scene.

### 3. Interactive 3D View
An interactive Open3D window will open showing:
*   **Grey Mesh**: The input torso.
*   **Red Points**: The Region of Interest (ROI) used for analysis.
*   **Green Plane**: The reference Back Plane.
*   **Grey Plane**: The fitted Hump Plane.
*   **Blue Arrow**: The Y-axis (Posterior Normal).
*   **Red Line**: The line connecting the left and right rib crests.

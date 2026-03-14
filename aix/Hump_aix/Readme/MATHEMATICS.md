# Mathematical Formulation of Hump AIX Analysis

This document details the mathematical framework, variable definitions, and step-by-step algorithms used in `hump_aix.py` to quantify rib hump asymmetry.

## 1. Coordinate System Definition

The analysis operates in a **Torso-Centric Coordinate System** derived from the input mesh.

*   **Variables**:
    *   $P = \{p_1, p_2, ..., p_N\}$: Set of $N$ vertices in the mesh, where $p_i = (x_i, y_i, z_i)$.
    *   $\mathbf{u}_x = [1, 0, 0]$: Unit vector for the **X-axis** (Left $\leftrightarrow$ Right).
    *   $\mathbf{u}_y = [0, 1, 0]$: Unit vector for the **Y-axis** (Posterior Normal / Depth).
    *   $\mathbf{u}_z = [0, 0, 1]$: Unit vector for the **Z-axis** (Caudal $\leftrightarrow$ Cranial / Spine Axis).

*   **Orientation**:
    *   $x$: Lateral position. $x < 0$ is Left, $x > 0$ is Right.
    *   $y$: Posterior depth. Higher $y$ values indicate points further "out" of the back (the hump).
    *   $z$: Longitudinal position. Lower $z$ is Neck, Higher $z$ is Hips.

---

## 2. Region of Interest (ROI) Selection

We isolate the thoracic region to focus the analysis.

*   **Step 1**: Calculate percentiles of the $z$-coordinates.
    $$z_{30} = P_{30}(Z), \quad z_{75} = P_{75}(Z)$$
*   **Step 2**: Define the ROI subset $P_{ROI}$.
    $$P_{ROI} = \{ p_i \in P \mid z_{30} < z_i < z_{75} \}$$
    *   $P_{30}, P_{75}$: The 30th and 75th percentiles of the Z-coordinates.

---

## 3. Depth Map Generation

We discretize the back surface into a 2D grid to analyze topography.

*   **Variables**:
    *   $n_x, n_z$: Number of bins along X and Z axes (approx $\sqrt{N_{ROI}/10}$).
    *   $B_{ij}$: The set of points falling into grid bin $(i, j)$.

*   **Step 1**: Binning.
    Divide the $x$ range $[x_{min}, x_{max}]$ and $z$ range $[z_{min}, z_{max}]$ into $n_x \times n_z$ bins.

*   **Step 2**: Raw Depth Calculation ($D_{raw}$).
    For each bin $(i, j)$, the depth is the 90th percentile of the $y$-coordinates of points in that bin (robust peak).
    $$D_{raw}(i, j) = P_{90}(\{ y_k \mid p_k \in B_{ij} \})$$

*   **Step 3**: Smoothing ($D_{smooth}$).
    Apply a Gaussian filter to reduce noise.
    $$D_{smooth} = D_{raw} * G(\sigma=2)$$
    *   $G$: Gaussian kernel.

---

## 4. Volume Asymmetry Index

Quantifies the difference in "bulge" volume between the left and right sides.

*   **Variables**:
    *   $V_L$: Volume of the left side.
    *   $V_R$: Volume of the right side.
    *   $\Delta x, \Delta z$: Physical dimensions of a grid cell.

*   **Step 1**: Integration.
    Sum the positive depths (bulges) for left ($x < 0$) and right ($x > 0$) bins.
    $$V_L = \sum_{i \in Left, j} \max(0, D_{smooth}(i, j)) \cdot \Delta x \cdot \Delta z$$
    $$V_R = \sum_{i \in Right, j} \max(0, D_{smooth}(i, j)) \cdot \Delta x \cdot \Delta z$$

*   **Step 2**: Asymmetry Index ($ASI$).
    $$ASI = \frac{|V_L - V_R|}{V_L + V_R}$$

---

## 5. Ridge Lines Extraction

Identifies the profile of the hump along the spine.

*   **Variables**:
    *   $R_L(z_j)$: Maximum depth of the left side at longitudinal level $j$.
    *   $R_R(z_j)$: Maximum depth of the right side at longitudinal level $j$.

*   **Formula**:
    $$R_L(z_j) = \max_{i \in Left} (D_{smooth}(i, j))$$
    $$R_R(z_j) = \max_{i \in Right} (D_{smooth}(i, j))$$

---

## 6. Rib Hump Crest Detection

Locates the specific points representing the peaks of the rib hump.

*   **Step 1**: Find the Z-level of maximum asymmetry ($z^*$).
    $$z^* = \arg \max_{z_j} |R_L(z_j) - R_R(z_j)|$$

*   **Step 2**: Identify Crest Points ($p_L, p_R$).
    Within the slice $z \approx z^*$:
    *   $p_L$: The point with the maximum $y$ value on the Left side ($x < 0$).
    *   $p_R$: The point with the maximum $y$ value on the Right side ($x > 0$).

---


## 7. Hump Plane Angle ($\theta_H$)

Measures the local tilt of the rib hump relative to the general back plane.

*   **Variables**:
    *   $S_{hump}$: Subset of points on the dominant (larger volume) side.
    *   $S_{crest}$: Top 15% of points in $S_{hump}$ based on $y$ value (the "peak" of the hump).
    *   $\mathbf{n}_{hump}$: Normal vector of the plane fitted to $S_{crest}$.
    *   $\mathbf{n}_{back} = \mathbf{u}_y = [0, 1, 0]$: Reference normal of the back plane.

*   **Step 1**: Plane Fitting (PCA).
    Compute the covariance matrix of centered points in $S_{crest}$ and perform Singular Value Decomposition (SVD). $\mathbf{n}_{hump}$ is the eigenvector corresponding to the smallest eigenvalue.

*   **Step 2**: Calculate Angle.
    $$\theta_H = \arccos( | \mathbf{n}_{hump} \cdot \mathbf{n}_{back} | )$$
    *   The absolute value handles the ambiguity of normal vector direction (inward vs outward).
    *   Result converted to degrees.

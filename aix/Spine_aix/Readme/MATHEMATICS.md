# Spine AIX: Mathematical Formulation

This document details the mathematical models and algorithms implemented in `spine_aix.py` for estimating the mid-sagittal plane of the human back.

## 1. Coordinate System

The analysis assumes a standard medical/anatomical coordinate system for the input point cloud $P = \{p_i \in \mathbb{R}^3 \mid i=1 \dots N\}$:

*   **X-axis**: Left-Right direction.
*   **Y-axis**: Posterior-Anterior direction (Depth).
*   **Z-axis**: Inferior-Superior direction (Spine axis, Height).

## 2. Method 1: Slice-Midpoint Estimation

This method serves as a robust initialization. It assumes the trunk is roughly symmetric and approximates the mid-sagittal plane by fitting a plane to the centroids of horizontal body slices.

### 2.1. Slicing
The point cloud is partitioned into $K$ horizontal slices along the Z-axis. For the $k$-th slice covering the interval $[z_k, z_k + \Delta z)$:

$$ S_k = \{ p \in P \mid z_k \le p_z < z_k + \Delta z \} $$

### 2.2. Midpoint Calculation
For each slice $S_k$, a robust midpoint $m_k$ is computed to define the "center" of the torso at that height. To avoid outliers (e.g., arms, artifacts), we use percentiles rather than min/max.

Let $X_k = \{ x \mid (x, y, z) \in S_k \}$ be the set of x-coordinates in the slice.
The left and right boundaries are estimated as:
$$ x_{left} = \text{Percentile}(X_k, 5) $$
$$ x_{right} = \text{Percentile}(X_k, 95) $$

The midpoint coordinates are:
$$ m_{k, x} = \frac{x_{left} + x_{right}}{2} $$
$$ m_{k, y} = \text{Median}(\{ y \mid (x, y, z) \in S_k \}) $$
$$ m_{k, z} = z_k + \frac{\Delta z}{2} $$

### 2.3. Plane Fitting (SVD)
We obtain a set of midpoints $M = \{m_1, \dots, m_K\}$. The mid-sagittal plane is defined by a normal vector $\mathbf{n}$ and scalar $d$ such that $\mathbf{n} \cdot x + d = 0$.

We compute the centroid of the midpoints:
$$ \bar{m} = \frac{1}{K} \sum_{k=1}^K m_k $$

We form the centered data matrix $A$:
$$ A = \begin{bmatrix} (m_1 - \bar{m})^T \\ \vdots \\ (m_K - \bar{m})^T \end{bmatrix} $$

We perform Singular Value Decomposition (SVD) on $A$:
$$ A = U \Sigma V^T $$

The normal $\mathbf{n}$ is the right singular vector corresponding to the smallest singular value (the last row of $V^T$ or last column of $V$). This minimizes the sum of squared orthogonal distances from the points to the plane.

The offset $d$ is determined by enforcing the plane to pass through the centroid:
$$ d = - \mathbf{n} \cdot \bar{m} $$

---

## 3. Method 2: Symmetry Optimization

This method refines the plane by maximizing the global reflection symmetry of the point cloud. It is an optimization problem over the plane parameters.

### 3.1. Plane Parameterization
The plane is parameterized by spherical coordinates for the normal vector and the offset $d$:
$$ \mathbf{n}(\theta, \phi) = \begin{bmatrix} \sin\theta \cos\phi \\ \sin\theta \sin\phi \\ \cos\theta \end{bmatrix} $$
The state vector is $\xi = [\theta, \phi, d]$.

### 3.2. Reflection Transformation
For a given plane $(\mathbf{n}, d)$, the reflection of a point $p$ across the plane is given by $p'$:

$$ p' = p - 2 (\mathbf{n} \cdot p + d) \mathbf{n} $$

This transforms the entire point cloud $P$ into a reflected cloud $P_{ref}$.

### 3.3. Cost Function (Symmetry Loss)
To measure symmetry, we compute the discrepancy between the original cloud $P$ and the reflected cloud $P_{ref}$. We use a bidirectional Chamfer-like distance calculated via Nearest Neighbors (NN).

Let $NN(q, S)$ be the Euclidean distance from query point $q$ to its nearest neighbor in set $S$.

$$ E(P, P_{ref}) = \sum_{p \in P} NN(p, P_{ref})^2 + \sum_{q \in P_{ref}} NN(q, P)^2 $$

**Robustness (Trimmed Loss):**
To handle asymmetries like scoliosis curves or rib humps (where perfect symmetry is impossible), we use a **Trimmed Mean of Squares**.
Let $D$ be the set of all squared distances calculated above. We sort $D$ and discard the top $T\%$ (outliers/large deformations).
The cost function $J(\xi)$ is the mean of the remaining distances.

### 3.4. Optimization
We minimize $J(\xi)$ using the L-BFGS-B algorithm.
*   **Initialization**: $\xi_0$ derived from the Method 1 (Slice-Midpoint) result.
*   **Bounds**: Constrained to keep the normal orientation consistent.

$$ \xi_{opt} = \arg \min_{\xi} J(\xi) $$

The resulting $\mathbf{n}(\xi_{opt})$ and $d(\xi_{opt})$ define the optimal symmetry plane.

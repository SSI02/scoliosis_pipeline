import numpy as np
import open3d as o3d
from plyfile import PlyData
from skimage.filters import threshold_otsu
import tkinter as tk
from tkinter import filedialog, messagebox
import os

class MeshLabQualityAutoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MeshLab Quality Cleanup (Auto Threshold)")

        self.vertices = None
        self.faces = None
        self.quality = None

        # ---------- UI ----------
        tk.Button(
            root, text="Load Poisson PLY (with quality)",
            width=45, command=self.load_ply
        ).pack(pady=6)

        self.mode = tk.StringVar(value="auto")

        tk.Radiobutton(
            root, text="Auto threshold (histogram valley)",
            variable=self.mode, value="auto"
        ).pack()

        tk.Radiobutton(
            root, text="Manual threshold (ratio × max quality)",
            variable=self.mode, value="manual"
        ).pack()

        self.slider = tk.Scale(
            root, from_=0.1, to=0.9, resolution=0.02,
            orient=tk.HORIZONTAL, length=360,
            label="Manual Threshold Ratio"
        )
        self.slider.set(0.5)
        self.slider.pack(pady=4)

        tk.Button(
            root, text="Run MeshLab Cleanup",
            width=45, command=self.run_cleanup
        ).pack(pady=10)

        self.status = tk.Label(root, text="No mesh loaded.", fg="blue")
        self.status.pack(pady=8)

    # --------------------------------------------------
    def load_ply(self):
        path = filedialog.askopenfilename(
            filetypes=[("PLY files", "*.ply")]
        )
        if not path:
            return

        ply = PlyData.read(path)
        v = ply["vertex"].data

        if "quality" not in v.dtype.names:
            messagebox.showerror(
                "Error",
                "PLY does not contain 'quality'.\n"
                "Cannot replicate MeshLab."
            )
            return

        self.vertices = np.vstack([v["x"], v["y"], v["z"]]).T
        self.quality = v["quality"].astype(np.float64)

        if "face" in ply:
            self.faces = np.vstack(ply["face"].data["vertex_indices"])
        else:
            self.faces = None

        self.status.config(text=f"Loaded: {os.path.basename(path)}")

    # --------------------------------------------------
    def run_cleanup(self):
        if self.vertices is None:
            messagebox.showerror("Error", "Load a PLY first.")
            return

        Q = self.quality

        # ---- AUTO histogram-based threshold ----
        if self.mode.get() == "auto":
            threshold = threshold_otsu(Q)
            threshold_info = f"Otsu threshold = {threshold:.6f}"
        else:
            ratio = float(self.slider.get())
            threshold = ratio * Q.max()
            threshold_info = f"Manual threshold = {ratio:.2f} × max"

        # ---- Vertex selection ----
        keep_v = Q >= threshold

        # ---- Reindex ----
        old_to_new = -np.ones(len(keep_v), dtype=int)
        old_to_new[keep_v] = np.arange(np.sum(keep_v))

        V2 = self.vertices[keep_v]

        if self.faces is not None:
            face_mask = np.all(keep_v[self.faces], axis=1)
            F2 = old_to_new[self.faces[face_mask]]
        else:
            F2 = None

        # ---- Build mesh ----
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V2)
        if F2 is not None:
            mesh.triangles = o3d.utility.Vector3iVector(F2)

        mesh.compute_vertex_normals()

        # ---- KEEP LARGEST CONNECTED COMPONENT ----
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)

        keep_cluster = counts.argmax()
        mesh.remove_triangles_by_mask(labels != keep_cluster)
        mesh.remove_unreferenced_vertices()

        # ---- Final cleanup ----
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()

        # ---- Save ----
        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir:
            return

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "near_quality_cleaned.ply")
        o3d.io.write_triangle_mesh(out_path, mesh)

        messagebox.showinfo(
            "Done",
            f"Cleanup complete\n\n"
            f"{threshold_info}\n"
            f"Saved to:\n{out_path}"
        )

# --------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MeshLabQualityAutoGUI(root)
    root.mainloop()

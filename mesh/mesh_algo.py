"""
Point Cloud to Mesh Reconstruction Tool
Applies MeshLab filters: Normal Computation, Poisson Reconstruction, and HC Laplacian Smoothing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
import time

try:
    import pymeshlab
except ImportError:
    print("PyMeshLab not installed. Install with: pip install pymeshlab")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("NumPy not installed. Install with: pip install numpy")
    exit(1)


class MeshReconstructionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud to Mesh Reconstruction")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.is_processing = False
        
        # Parameters
        self.normal_neighbours = tk.IntVar(value=500)
        self.normal_smooth_iter = tk.IntVar(value=5)
        self.poisson_depth = tk.IntVar(value=8)
        self.hc_iterations = tk.IntVar(value=2)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Point Cloud Mesh Reconstruction", 
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input File Section
        ttk.Label(main_frame, text="Input Point Cloud:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(input_frame, textvariable=self.input_path, width=60).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).grid(
            row=0, column=1)
        
        # Output File Section
        ttk.Label(main_frame, text="Output Mesh:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=5)
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        output_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_frame, textvariable=self.output_path, width=60).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(
            row=0, column=1)
        
        # Parameters Section
        params_label = ttk.Label(main_frame, text="Processing Parameters", 
                                font=("Arial", 12, "bold"))
        params_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(10, 10))
        
        # Normal Computation Parameters
        ttk.Label(main_frame, text="1. Normal Computation", 
                font=("Arial", 10, "bold"), foreground="blue").grid(
            row=6, column=0, columnspan=3, sticky=tk.W, pady=(5, 5))
        
        ttk.Label(main_frame, text="Neighbour Num:").grid(row=7, column=0, sticky=tk.W, padx=(20, 0))
        ttk.Spinbox(main_frame, from_=10, to=1000, textvariable=self.normal_neighbours, 
                width=15).grid(row=7, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(main_frame, text="Smooth Iterations:").grid(row=8, column=0, sticky=tk.W, padx=(20, 0))
        ttk.Spinbox(main_frame, from_=0, to=20, textvariable=self.normal_smooth_iter, 
                width=15).grid(row=8, column=1, sticky=tk.W, pady=2)
        
        # Poisson Reconstruction Parameters
        ttk.Label(main_frame, text="2. Screened Poisson Reconstruction", 
                font=("Arial", 10, "bold"), foreground="blue").grid(
            row=9, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        ttk.Label(main_frame, text="Octree Depth:").grid(row=10, column=0, sticky=tk.W, padx=(20, 0))
        ttk.Spinbox(main_frame, from_=5, to=12, textvariable=self.poisson_depth, 
                width=15).grid(row=10, column=1, sticky=tk.W, pady=2)
        ttk.Label(main_frame, text="(Default: 8, Higher = More detail)", 
                font=("Arial", 8), foreground="gray").grid(row=10, column=2, sticky=tk.W)
        
        # HC Laplacian Smooth Parameters
        ttk.Label(main_frame, text="3. HC Laplacian Smoothing", 
                font=("Arial", 10, "bold"), foreground="blue").grid(
            row=11, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        ttk.Label(main_frame, text="Iterations:").grid(row=12, column=0, sticky=tk.W, padx=(20, 0))
        ttk.Spinbox(main_frame, from_=1, to=10, textvariable=self.hc_iterations, 
                width=15).grid(row=12, column=1, sticky=tk.W, pady=2)
        
        # Process Button
        self.process_btn = ttk.Button(main_frame, text="Start Reconstruction", 
                                    command=self.start_processing, style="Accent.TButton")
        self.process_btn.grid(row=13, column=0, columnspan=3, pady=20)
        
        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=14, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Log Section
        ttk.Label(main_frame, text="Processing Log:", font=("Arial", 10, "bold")).grid(
            row=15, column=0, sticky=tk.W, pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=12, width=80, 
                                                font=("Consolas", 9))
        self.log_text.grid(row=16, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), 
                        pady=(0, 10))
        
        main_frame.rowconfigure(16, weight=1)
        
        # Status Bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Point Cloud File",
            filetypes=[
                ("Point Cloud Files", "*.ply *.xyz *.pts *.pcd *.obj *.stl"),
                ("PLY Files", "*.ply"),
                ("XYZ Files", "*.xyz"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.input_path.set(filename)
            # Auto-suggest output path
            base = os.path.splitext(filename)[0]
            self.output_path.set(f"{base}_reconstructed.ply")
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Mesh As",
            defaultextension=".ply",
            filetypes=[
                ("PLY Files", "*.ply"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.output_path.set(filename)
            
    def log(self, message):
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Processing", "Already processing a mesh!")
            return
            
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input point cloud file!")
            return
            
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output file path!")
            return
            
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input file does not exist!")
            return
            
        # Start processing in a separate thread
        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.progress.start(10)
        
        thread = threading.Thread(target=self.process_mesh)
        thread.daemon = True
        thread.start()
        
    def process_mesh(self):
        try:
            self.log("=" * 60)
            self.log("Starting mesh reconstruction pipeline...")
            self.update_status("Processing...")
            
            # Initialize MeshLab MeshSet
            ms = pymeshlab.MeshSet()
            
            # Load point cloud
            self.log(f"Loading point cloud: {self.input_path.get()}")
            ms.load_new_mesh(self.input_path.get())
            self.log(f"✓ Loaded {ms.current_mesh().vertex_number()} points")
            
            # Step 1: Compute Normals
            self.log("\n--- Step 1: Computing Normals ---")
            self.update_status("Computing normals...")
            
            ms.compute_normal_for_point_clouds(
                k=self.normal_neighbours.get(),
                smoothiter=self.normal_smooth_iter.get(),
                flipflag=False,
                viewpos=np.array([0.0, 0.0, 0.0])
            )
            self.log(f"✓ Normals computed (neighbours={self.normal_neighbours.get()}, "
                    f"smooth_iter={self.normal_smooth_iter.get()})")
            
            # Step 2: Screened Poisson Surface Reconstruction
            self.log("\n--- Step 2: Screened Poisson Reconstruction ---")
            self.update_status("Reconstructing surface...")
            
            ms.generate_surface_reconstruction_screened_poisson(
                depth=self.poisson_depth.get(),
                fulldepth=5,
                cgdepth=0,
                scale=1.1,
                samplespernode=1.5,
                pointweight=4.0,
                iters=8,
                confidence=False,
                preclean=False
            )
            self.log(f"✓ Surface reconstructed (depth={self.poisson_depth.get()})")
            self.log(f"  Mesh has {ms.current_mesh().vertex_number()} vertices, "
                    f"{ms.current_mesh().face_number()} faces")
            
            # Step 3: Laplacian Smoothing
            self.log("\n--- Step 3: Laplacian Smoothing ---")
            self.update_status("Smoothing mesh...")
            
            ms.apply_coord_laplacian_smoothing(
                stepsmoothnum=self.hc_iterations.get(),
                cotangentweight=False,
                selected=False
            )
            self.log(f"✓ Mesh smoothed ({self.hc_iterations.get()} iterations)")
            
            # Save mesh
            self.log("\n--- Saving Mesh ---")
            self.update_status("Saving mesh...")
            
            output_dir = os.path.dirname(self.output_path.get())
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            ms.save_current_mesh(self.output_path.get())
            self.log(f"✓ Mesh saved to: {self.output_path.get()}")
            
            # Final statistics
            final_vertices = ms.current_mesh().vertex_number()
            final_faces = ms.current_mesh().face_number()
            file_size = os.path.getsize(self.output_path.get()) / 1024 / 1024
            
            self.log("\n" + "=" * 60)
            self.log("RECONSTRUCTION COMPLETE!")
            self.log(f"Final mesh: {final_vertices} vertices, {final_faces} faces")
            self.log(f"Output file size: {file_size:.2f} MB")
            self.log("=" * 60)
            
            self.update_status("Complete!")
            messagebox.showinfo("Success", 
                            f"Mesh reconstruction complete!\n\n"
                            f"Vertices: {final_vertices:,}\n"
                            f"Faces: {final_faces:,}\n"
                            f"Saved to: {self.output_path.get()}")
            
        except Exception as e:
            self.log(f"\n✗ ERROR: {str(e)}")
            self.update_status("Error occurred")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            
        finally:
            self.is_processing = False
            self.process_btn.config(state='normal')
            self.progress.stop()


def main():
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = MeshReconstructionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

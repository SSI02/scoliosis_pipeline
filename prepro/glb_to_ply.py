import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import trimesh

def convert_glb_to_ply(input_path, output_path, binary=True):
    # """
    # Convert a .glb file to a .ply file using trimesh.
    # """
    try:
        scene_or_mesh = trimesh.load(input_path, force='scene')

        if isinstance(scene_or_mesh, trimesh.Scene):
            meshes = scene_or_mesh.dump(concatenate=False)
            if len(meshes) == 0:
                raise ValueError("No mesh geometry found in file.")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene_or_mesh, trimesh.Trimesh):
            mesh = scene_or_mesh
        else:
            raise TypeError("Unsupported GLB structure")

        mesh.merge_vertices()

        export_kwargs = {}
        if binary:
            export_kwargs['encoding'] = 'binary'

        ply_data = mesh.export(file_type='ply', **export_kwargs)
        mode = 'wb' if isinstance(ply_data, (bytes, bytearray)) else 'w'
        with open(output_path, mode) as f:
            f.write(ply_data)

        return f"✅ {os.path.basename(input_path)} → {os.path.basename(output_path)}"
    except Exception as e:
        return f"❌ {os.path.basename(input_path)}: {e}"


def browse_input_folder():
    folder = filedialog.askdirectory(title="Select Input Folder (with .glb files)")
    if folder:
        input_folder_var.set(folder)

def browse_output_folder():
    folder = filedialog.askdirectory(title="Select Output Folder (to save .ply files)")
    if folder:
        output_folder_var.set(folder)

def start_conversion():
    input_folder = input_folder_var.get()
    output_folder = output_folder_var.get()
    binary = binary_var.get()

    if not os.path.isdir(input_folder):
        messagebox.showerror("Error", "Please select a valid input folder.")
        return
    if not os.path.isdir(output_folder):
        messagebox.showerror("Error", "Please select a valid output folder.")
        return

    glb_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".glb")]
    if not glb_files:
        messagebox.showwarning("No Files", "No .glb files found in input folder.")
        return

    status_text.delete(1.0, tk.END)
    status_text.insert(tk.END, f"Starting conversion of {len(glb_files)} files...\n\n")

    for glb_file in glb_files:
        input_path = os.path.join(input_folder, glb_file)
        output_file = os.path.splitext(glb_file)[0] + ".ply"
        output_path = os.path.join(output_folder, output_file)

        msg = convert_glb_to_ply(input_path, output_path, binary=binary)
        status_text.insert(tk.END, msg + "\n")
        status_text.see(tk.END)
        root.update_idletasks()

    messagebox.showinfo("Done", "All conversions completed!")


# ---- UI Setup ----
root = tk.Tk()
root.title("GLB → PLY Converter")
root.geometry("650x420")
root.resizable(False, False)
root.configure(bg="#1e1e1e")

title_label = tk.Label(root, text="GLB → PLY Converter", font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#1e1e1e")
title_label.pack(pady=10)

frame = tk.Frame(root, bg="#1e1e1e")
frame.pack(pady=10)

input_folder_var = tk.StringVar()
output_folder_var = tk.StringVar()
binary_var = tk.BooleanVar(value=True)

# Input Folder
tk.Label(frame, text="Input Folder:", font=("Segoe UI", 11), fg="white", bg="#1e1e1e").grid(row=0, column=0, sticky="e", padx=5, pady=5)
tk.Entry(frame, textvariable=input_folder_var, width=50).grid(row=0, column=1, padx=5)
tk.Button(frame, text="Browse", command=browse_input_folder, bg="#3b3b3b", fg="white").grid(row=0, column=2, padx=5)

# Output Folder
tk.Label(frame, text="Output Folder:", font=("Segoe UI", 11), fg="white", bg="#1e1e1e").grid(row=1, column=0, sticky="e", padx=5, pady=5)
tk.Entry(frame, textvariable=output_folder_var, width=50).grid(row=1, column=1, padx=5)
tk.Button(frame, text="Browse", command=browse_output_folder, bg="#3b3b3b", fg="white").grid(row=1, column=2, padx=5)

# Binary/ASCII option
tk.Checkbutton(root, text="Export as Binary PLY (uncheck for ASCII)", variable=binary_var,
            font=("Segoe UI", 10), fg="white", bg="#1e1e1e", selectcolor="#2e2e2e").pack(pady=5)

# Start button
tk.Button(root, text="Start Conversion", font=("Segoe UI", 12, "bold"), command=start_conversion,
        bg="#007acc", fg="white", relief="flat", padx=20, pady=8).pack(pady=10)

# Status box
status_text = scrolledtext.ScrolledText(root, width=80, height=12, bg="#252526", fg="white", font=("Consolas", 10))
status_text.pack(padx=10, pady=5)

# Footer
tk.Label(root, text="Developed for VGGT Reconstructed Files", fg="#888", bg="#1e1e1e", font=("Segoe UI", 9)).pack(side="bottom", pady=5)

root.mainloop()

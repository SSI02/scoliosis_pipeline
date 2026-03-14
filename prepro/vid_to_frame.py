import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def select_input_folder():
    input_path.set(filedialog.askdirectory(title="Select Input Folder (with videos)"))

def select_output_folder():
    output_path.set(filedialog.askdirectory(title="Select Output Folder"))

def extract_frames_batch():
    input_folder = input_path.get()
    output_folder = output_path.get()

    if not input_folder or not output_folder:
        messagebox.showerror("Error", "Please select both input and output folders")
        return

    # Ask user how many frames they want
    try:
        num_frames = int(frame_count_entry.get())
        if num_frames <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid positive integer for frame count")
        return

    # Process each video in input folder
    videos = [f for f in os.listdir(input_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

    if not videos:
        messagebox.showerror("Error", "No video files found in the input folder")
        return

    total_videos = len(videos)
    processed_videos = 0

    for video in videos:
        video_file = os.path.join(input_folder, video)
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            continue

        # Calculate equispaced indices
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        # Create subfolder for each video
        video_name = os.path.splitext(video)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        saved = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                filename = os.path.join(video_output_folder, f"frame_{saved:04d}.png")
                cv2.imwrite(filename, frame)
                saved += 1

        cap.release()
        processed_videos += 1

    messagebox.showinfo("Success", f"Processed {processed_videos}/{total_videos} videos. Frames saved in {output_folder}")

# ------------------- GUI -------------------
root = tk.Tk()
root.title("Batch Video to Frames Extractor")

input_path = tk.StringVar()
output_path = tk.StringVar()

tk.Label(root, text="Input Folder (with videos):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
tk.Entry(root, textvariable=input_path, width=50).grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=select_input_folder).grid(row=0, column=2, padx=5, pady=5)

tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
tk.Entry(root, textvariable=output_path, width=50).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=select_output_folder).grid(row=1, column=2, padx=5, pady=5)

tk.Label(root, text="Number of Frames per Video:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
frame_count_entry = tk.Entry(root, width=10)
frame_count_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

tk.Button(root, text="Extract Frames (Batch)", command=extract_frames_batch, bg="lightblue").grid(row=3, column=0, columnspan=3, pady=10)

root.mainloop()

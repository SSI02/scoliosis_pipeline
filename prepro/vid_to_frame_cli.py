#!/usr/bin/env python3
"""
CLI wrapper for video to frame extraction.
Usage:
    python vid_to_frame_cli.py --input_folder /path/to/videos --output_folder /path/to/frames --num_frames 50
"""

import cv2
import os
import argparse


def extract_frames_batch(input_folder: str, output_folder: str, num_frames: int) -> dict:
    """
    Extract equispaced frames from all videos in input folder.
    
    Args:
        input_folder: Directory containing video files
        output_folder: Directory to save extracted frames
        num_frames: Number of frames to extract per video
        
    Returns:
        dict: Statistics about processing
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    videos = [f for f in os.listdir(input_folder) 
              if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    
    if not videos:
        raise ValueError(f"No video files found in {input_folder}")
    
    results = {
        "total_videos": len(videos),
        "processed_videos": 0,
        "failed_videos": [],
        "output_dirs": []
    }
    
    for video in videos:
        video_file = os.path.join(input_folder, video)
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            results["failed_videos"].append(video)
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
        results["processed_videos"] += 1
        results["output_dirs"].append(video_output_folder)
        print(f"✓ {video}: {saved} frames extracted -> {video_output_folder}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract equispaced frames from videos")
    parser.add_argument("--input_folder", "-i", required=True, 
                        help="Input folder containing video files")
    parser.add_argument("--output_folder", "-o", required=True,
                        help="Output folder for extracted frames")
    parser.add_argument("--num_frames", "-n", type=int, default=50,
                        help="Number of frames to extract per video (default: 50)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Video to Frame Extraction")
    print(f"{'='*60}")
    print(f"Input folder:  {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Frames/video:  {args.num_frames}")
    print(f"{'='*60}\n")
    
    results = extract_frames_batch(
        args.input_folder, 
        args.output_folder, 
        args.num_frames
    )
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Processed: {results['processed_videos']}/{results['total_videos']} videos")
    if results["failed_videos"]:
        print(f"Failed: {results['failed_videos']}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI wrapper for GLB to PLY conversion.
Usage:
    python glb_to_ply_cli.py --input_folder /path/to/glb --output_folder /path/to/ply
    python glb_to_ply_cli.py --input_file /path/to/file.glb --output_file /path/to/output.ply
"""

import os
import argparse
import trimesh


def convert_glb_to_ply(input_path: str, output_path: str, binary: bool = True) -> str:
    """
    Convert a .glb file to a .ply file using trimesh.
    
    Args:
        input_path: Path to input GLB file
        output_path: Path for output PLY file
        binary: Whether to export as binary PLY (default: True)
        
    Returns:
        str: Status message
    """
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
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, mode) as f:
            f.write(ply_data)
        
        return f"✓ {os.path.basename(input_path)} → {os.path.basename(output_path)}"
    
    except Exception as e:
        return f"✗ {os.path.basename(input_path)}: {e}"


def batch_convert(input_folder: str, output_folder: str, binary: bool = True) -> dict:
    """
    Convert all GLB files in a folder to PLY format.
    
    Args:
        input_folder: Directory containing GLB files
        output_folder: Directory for output PLY files
        binary: Whether to export as binary PLY
        
    Returns:
        dict: Conversion statistics
    """
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    glb_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".glb")]
    
    if not glb_files:
        raise ValueError(f"No .glb files found in {input_folder}")
    
    results = {
        "total": len(glb_files),
        "success": 0,
        "failed": [],
        "output_files": []
    }
    
    for glb_file in glb_files:
        input_path = os.path.join(input_folder, glb_file)
        output_file = os.path.splitext(glb_file)[0] + ".ply"
        output_path = os.path.join(output_folder, output_file)
        
        msg = convert_glb_to_ply(input_path, output_path, binary=binary)
        print(msg)
        
        if msg.startswith("✓"):
            results["success"] += 1
            results["output_files"].append(output_path)
        else:
            results["failed"].append(glb_file)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert GLB files to PLY format")
    parser.add_argument("--input_folder", "-i", 
                        help="Input folder containing GLB files")
    parser.add_argument("--output_folder", "-o",
                        help="Output folder for PLY files")
    parser.add_argument("--input_file", "-if",
                        help="Single input GLB file")
    parser.add_argument("--output_file", "-of",
                        help="Single output PLY file")
    parser.add_argument("--ascii", action="store_true",
                        help="Export as ASCII PLY instead of binary")
    
    args = parser.parse_args()
    
    binary = not args.ascii
    
    print(f"\n{'='*60}")
    print("GLB to PLY Converter")
    print(f"{'='*60}")
    
    # Single file mode
    if args.input_file:
        if not args.output_file:
            args.output_file = os.path.splitext(args.input_file)[0] + ".ply"
        
        print(f"Input:  {args.input_file}")
        print(f"Output: {args.output_file}")
        print(f"Binary: {binary}")
        print(f"{'='*60}\n")
        
        msg = convert_glb_to_ply(args.input_file, args.output_file, binary)
        print(msg)
        
    # Batch mode
    elif args.input_folder and args.output_folder:
        print(f"Input folder:  {args.input_folder}")
        print(f"Output folder: {args.output_folder}")
        print(f"Binary: {binary}")
        print(f"{'='*60}\n")
        
        results = batch_convert(args.input_folder, args.output_folder, binary)
        
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Converted: {results['success']}/{results['total']} files")
        if results["failed"]:
            print(f"Failed: {results['failed']}")
    else:
        parser.print_help()
        print("\nError: Provide either --input_file or both --input_folder and --output_folder")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

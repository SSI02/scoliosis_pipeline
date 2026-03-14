#!/usr/bin/env python3
"""
Automated 3D Body Scanning Pipeline
====================================
Fully automated sequential pipeline from video to AIX estimation.

Usage:
    python run_pipeline.py --far_video path/to/far.mp4 --near_video path/to/near.mp4
    python run_pipeline.py --far_video test_data/raw_video/p_env_far.mp4 --near_video test_data/raw_video/p_env_close.mp4
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import json
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Pipeline configuration"""
    
    # Conda environments for each stage
    ENVS = {
        'preprocess': 'vv_vggt',
        'denoise': 'vv_denoise', 
        'mesh': 'vv_meshlab',
        'ground': 'vv_gnd_estimate',
        'xaxis': 'vv_x_axis',
        'aix': 'vv_aix'
    }
    
    # Preprocessing
    NUM_FRAMES = 50
    
    # Denoising
    USE_SOR = True
    USE_ROR = True
    SOR_NEIGHBORS = 30
    SOR_STD = 2.0
    ROR_NEIGHBORS = 16
    ROR_RADIUS = 0.05
    
    # Mesh reconstruction  
    NORMAL_NEIGHBORS = 500
    NORMAL_SMOOTH = 5
    POISSON_DEPTH = 8
    HC_ITERATIONS = 2
    
    # X-axis alignment
    VOXEL_FRAC = 0.01
    GROUND_TOL_FRAC = 0.02
    GEODESIC_THRESH = 0.5
    
    # Hip AIX
    SLICE_FRAC = 0.02


# ============================================================
# CONSOLE OUTPUT HELPERS
# ============================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    """Print a major section header"""
    width = 70
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text.upper()}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}\n")


def print_stage(stage_num, total, name, env=None):
    """Print stage header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}[{stage_num}/{total}] {name}{Colors.END}")
    if env:
        print(f"    {Colors.YELLOW}Environment: {env}{Colors.END}")
    print(f"    {'-'*50}")


def print_progress(message):
    """Print progress message"""
    print(f"    {Colors.CYAN}► {message}{Colors.END}")


def print_success(message):
    """Print success message"""
    print(f"    {Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message):
    """Print error message"""
    print(f"    {Colors.RED}✗ {message}{Colors.END}")


def print_warning(message):
    """Print warning message"""
    print(f"    {Colors.YELLOW}⚠ {message}{Colors.END}")


def print_info(key, value):
    """Print info line"""
    print(f"    {Colors.BOLD}{key}:{Colors.END} {value}")


def print_file_saved(path):
    """Print file saved notification"""
    print(f"    {Colors.GREEN}💾 Saved: {path}{Colors.END}")


# ============================================================
# PIPELINE EXECUTION
# ============================================================

class Pipeline:
    """Automated pipeline executor"""
    
    def __init__(self, far_video: str, near_video: str, working_dir: str = None):
        self.far_video = os.path.abspath(far_video)
        self.near_video = os.path.abspath(near_video)
        self.base_dir = Path(__file__).parent
        
        # Setup working directory
        if working_dir:
            self.work_dir = Path(working_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.work_dir = self.base_dir / "working_directory" / f"run_{timestamp}"
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage directories
        self.dirs = {
            'frames_far': self.work_dir / "01_frames" / "far",
            'frames_near': self.work_dir / "01_frames" / "near",
            'ply_far': self.work_dir / "02_pointcloud" / "far",
            'ply_near': self.work_dir / "02_pointcloud" / "near",
            'denoised': self.work_dir / "03_denoised",
            'mesh_far': self.work_dir / "04_mesh" / "far",
            'mesh_near': self.work_dir / "04_mesh" / "near",
            'aligned': self.work_dir / "05_aligned",
            'aix': self.work_dir / "06_aix_results",
            'logs': self.work_dir / "logs"
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # Log file
        self.log_file = self.dirs['logs'] / "pipeline.log"
        
        # Track outputs
        self.outputs = {}
        
    def log(self, message: str):
        """Write to log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def run_command(self, cmd: list, env_name: str = None, 
                   cwd: str = None, capture: bool = True) -> tuple:
        """
        Run a command with real-time output.
        Returns (success, stdout, stderr)
        """
        if env_name:
            cmd_str = ' '.join(str(c) for c in cmd)
            full_cmd = f"conda run -n {env_name} {cmd_str}"
        else:
            full_cmd = ' '.join(str(c) for c in cmd)
        
        self.log(f"Running: {full_cmd}")
        
        try:
            if capture:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    cwd=cwd or str(self.base_dir),
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                # Log output
                if result.stdout:
                    self.log(f"STDOUT: {result.stdout}")
                if result.stderr:
                    self.log(f"STDERR: {result.stderr}")
                
                return result.returncode == 0, result.stdout, result.stderr
            else:
                # Stream output in real-time
                process = subprocess.Popen(
                    full_cmd,
                    shell=True,
                    cwd=cwd or str(self.base_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output = []
                for line in process.stdout:
                    line = line.rstrip()
                    output.append(line)
                    # Show condensed output
                    if any(k in line.lower() for k in ['error', 'success', 'saved', 'loaded', '✓', '✗', 'complete']):
                        print(f"      {line}")
                
                process.wait()
                return process.returncode == 0, '\n'.join(output), ''
                
        except subprocess.TimeoutExpired:
            print_error("Command timed out!")
            return False, '', 'Timeout'
        except Exception as e:
            print_error(f"Command failed: {e}")
            return False, '', str(e)
    
    def save_stage_output(self, stage_name: str, data: dict):
        """Save stage output metadata"""
        output_file = self.dirs['logs'] / f"{stage_name}_output.json"
        data['timestamp'] = datetime.now().isoformat()
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.outputs[stage_name] = data
    
    # --------------------------------------------------------
    # STAGE 1: Video to Frames
    # --------------------------------------------------------
    
    def stage_video_to_frames(self) -> bool:
        """Extract frames from videos"""
        print_stage(1, 9, "VIDEO TO FRAMES", Config.ENVS['preprocess'])
        
        success = True
        
        # Far video
        print_progress(f"Processing far video: {os.path.basename(self.far_video)}")
        cmd = [
            "python", self.base_dir / "prepro" / "vid_to_frame_cli.py",
            "--input_folder", os.path.dirname(self.far_video),
            "--output_folder", self.dirs['frames_far'].parent,
            "--num_frames", str(Config.NUM_FRAMES)
        ]
        
        # Need to handle single file - create temp dir
        temp_far = self.work_dir / "temp_far"
        temp_far.mkdir(exist_ok=True)
        shutil.copy(self.far_video, temp_far / os.path.basename(self.far_video))
        
        cmd = [
            "python", self.base_dir / "prepro" / "vid_to_frame_cli.py",
            "--input_folder", str(temp_far),
            "--output_folder", str(self.dirs['frames_far'].parent),
            "--num_frames", str(Config.NUM_FRAMES)
        ]
        
        ok, out, err = self.run_command(cmd, Config.ENVS['preprocess'])
        if ok:
            # Find output folder
            video_name = Path(self.far_video).stem
            actual_output = self.dirs['frames_far'].parent / video_name
            if actual_output.exists():
                # Move to expected location
                if self.dirs['frames_far'].exists():
                    shutil.rmtree(self.dirs['frames_far'])
                shutil.move(str(actual_output), str(self.dirs['frames_far']))
            
            frame_count = len(list(self.dirs['frames_far'].glob("*.png")))
            print_success(f"Far video: {frame_count} frames extracted")
            print_file_saved(str(self.dirs['frames_far']))
        else:
            print_error(f"Far video processing failed")
            success = False
        
        # Near video
        print_progress(f"Processing near video: {os.path.basename(self.near_video)}")
        
        temp_near = self.work_dir / "temp_near"
        temp_near.mkdir(exist_ok=True)
        shutil.copy(self.near_video, temp_near / os.path.basename(self.near_video))
        
        cmd = [
            "python", self.base_dir / "prepro" / "vid_to_frame_cli.py",
            "--input_folder", str(temp_near),
            "--output_folder", str(self.dirs['frames_near'].parent),
            "--num_frames", str(Config.NUM_FRAMES)
        ]
        
        ok, out, err = self.run_command(cmd, Config.ENVS['preprocess'])
        if ok:
            video_name = Path(self.near_video).stem
            actual_output = self.dirs['frames_near'].parent / video_name
            if actual_output.exists():
                if self.dirs['frames_near'].exists():
                    shutil.rmtree(self.dirs['frames_near'])
                shutil.move(str(actual_output), str(self.dirs['frames_near']))
            
            frame_count = len(list(self.dirs['frames_near'].glob("*.png")))
            print_success(f"Near video: {frame_count} frames extracted")
            print_file_saved(str(self.dirs['frames_near']))
        else:
            print_error(f"Near video processing failed")
            success = False
        
        # Cleanup temp dirs
        shutil.rmtree(temp_far, ignore_errors=True)
        shutil.rmtree(temp_near, ignore_errors=True)
        
        self.save_stage_output('01_frames', {
            'far_frames': str(self.dirs['frames_far']),
            'near_frames': str(self.dirs['frames_near']),
            'success': success
        })
        
        return success
    
    # --------------------------------------------------------
    # STAGE 2: AI-based 3D Reconstruction (Manual)
    # --------------------------------------------------------
    
    def stage_vggt(self) -> bool:
        """Run AI-based 3D reconstruction - requires manual intervention"""
        print_stage(2, 9, "AI-BASED 3D RECONSTRUCTION", Config.ENVS['preprocess'])
        
        print_warning("This stage requires manual execution via Gradio interface")
        print()
        print(f"    {Colors.BOLD}Instructions:{Colors.END}")
        print(f"    1. Open a new terminal")
        print(f"    2. Run: {Colors.CYAN}conda activate {Config.ENVS['preprocess']}{Colors.END}")
        print(f"    3. Run: {Colors.CYAN}cd {self.base_dir / 'prepro' / 'vggt'}{Colors.END}")
        print(f"    4. Run: {Colors.CYAN}python demo_gradio.py{Colors.END}")
        print()
        print(f"    {Colors.BOLD}Process these folders:{Colors.END}")
        print(f"    - FAR:  {Colors.GREEN}{self.dirs['frames_far']}{Colors.END}")
        print(f"    - NEAR: {Colors.GREEN}{self.dirs['frames_near']}{Colors.END}")
        print()
        print(f"    {Colors.BOLD}Save GLB outputs to:{Colors.END}")
        print(f"    - FAR GLB:  {Colors.GREEN}{self.dirs['ply_far'] / 'far_reconstruction.glb'}{Colors.END}")
        print(f"    - NEAR GLB: {Colors.GREEN}{self.dirs['ply_near'] / 'near_reconstruction.glb'}{Colors.END}")
        print()
        
        input(f"    {Colors.YELLOW}Press Enter when AI-based 3D reconstruction processing is complete...{Colors.END}")
        
        # Check if files exist
        far_glb = list(self.dirs['ply_far'].glob("*.glb"))
        near_glb = list(self.dirs['ply_near'].glob("*.glb"))
        
        if not far_glb:
            print_warning("No GLB file found for FAR mesh")
            far_path = input("    Enter path to FAR GLB file (or press Enter to skip): ").strip()
            if far_path and os.path.exists(far_path):
                shutil.copy(far_path, self.dirs['ply_far'] / "far_reconstruction.glb")
                far_glb = [self.dirs['ply_far'] / "far_reconstruction.glb"]
        
        if not near_glb:
            print_warning("No GLB file found for NEAR mesh")
            near_path = input("    Enter path to NEAR GLB file (or press Enter to skip): ").strip()
            if near_path and os.path.exists(near_path):
                shutil.copy(near_path, self.dirs['ply_near'] / "near_reconstruction.glb")
                near_glb = [self.dirs['ply_near'] / "near_reconstruction.glb"]
        
        success = bool(far_glb) and bool(near_glb)
        
        if success:
            print_success("AI-based 3D reconstruction files received")
        else:
            print_error("Missing AI-based 3D reconstruction output files")
        
        self.save_stage_output('02_vggt', {
            'far_glb': str(far_glb[0]) if far_glb else None,
            'near_glb': str(near_glb[0]) if near_glb else None,
            'success': success
        })
        
        return success
    
    # --------------------------------------------------------
    # STAGE 3: GLB to PLY Conversion
    # --------------------------------------------------------
    
    def stage_glb_to_ply(self) -> bool:
        """Convert GLB files to PLY"""
        print_stage(3, 9, "GLB TO PLY CONVERSION", Config.ENVS['preprocess'])
        
        success = True
        
        # Far GLB
        far_glb = list(self.dirs['ply_far'].glob("*.glb"))
        if far_glb:
            print_progress(f"Converting: {far_glb[0].name}")
            far_ply = self.dirs['ply_far'] / "far_pointcloud.ply"
            
            cmd = [
                "python", self.base_dir / "prepro" / "glb_to_ply_cli.py",
                "--input_file", str(far_glb[0]),
                "--output_file", str(far_ply)
            ]
            
            ok, _, _ = self.run_command(cmd, Config.ENVS['preprocess'])
            if ok and far_ply.exists():
                print_success(f"Far PLY created")
                print_file_saved(str(far_ply))
                self.outputs['far_ply'] = str(far_ply)
            else:
                print_error("Far GLB conversion failed")
                success = False
        else:
            print_error("No FAR GLB file found")
            success = False
        
        # Near GLB
        near_glb = list(self.dirs['ply_near'].glob("*.glb"))
        if near_glb:
            print_progress(f"Converting: {near_glb[0].name}")
            near_ply = self.dirs['ply_near'] / "near_pointcloud.ply"
            
            cmd = [
                "python", self.base_dir / "prepro" / "glb_to_ply_cli.py",
                "--input_file", str(near_glb[0]),
                "--output_file", str(near_ply)
            ]
            
            ok, _, _ = self.run_command(cmd, Config.ENVS['preprocess'])
            if ok and near_ply.exists():
                print_success(f"Near PLY created")
                print_file_saved(str(near_ply))
                self.outputs['near_ply'] = str(near_ply)
            else:
                print_error("Near GLB conversion failed")
                success = False
        else:
            print_error("No NEAR GLB file found")
            success = False
        
        self.save_stage_output('03_glb_to_ply', {
            'far_ply': self.outputs.get('far_ply'),
            'near_ply': self.outputs.get('near_ply'),
            'success': success
        })
        
        return success
    
    # --------------------------------------------------------
    # STAGE 4: Denoising (Far mesh only)
    # --------------------------------------------------------
    
    def stage_denoising(self) -> bool:
        """Denoise point clouds"""
        print_stage(4, 9, "POINT CLOUD DENOISING", Config.ENVS['denoise'])
        
        far_ply = self.outputs.get('far_ply')
        if not far_ply or not os.path.exists(far_ply):
            print_error("Far PLY not found, skipping denoising")
            return False
        
        print_progress(f"Denoising far point cloud...")
        print_info("SOR", f"k={Config.SOR_NEIGHBORS}, std={Config.SOR_STD}")
        print_info("ROR", f"n={Config.ROR_NEIGHBORS}, r={Config.ROR_RADIUS}")
        
        denoised_far = self.dirs['denoised'] / "far_denoised.ply"
        
        cmd = [
            "python", self.base_dir / "mesh" / "denoising_cli.py",
            "--input", far_ply,
            "--output", str(denoised_far),
            "--use-sor", "--use-ror",
            "--sor-n", str(Config.SOR_NEIGHBORS),
            "--sor-std", str(Config.SOR_STD),
            "--ror-n", str(Config.ROR_NEIGHBORS),
            "--ror-radius", str(Config.ROR_RADIUS)
        ]
        
        ok, out, _ = self.run_command(cmd, Config.ENVS['denoise'])
        
        if ok and denoised_far.exists():
            # Parse output for stats
            if 'Original:' in out and 'Final:' in out:
                for line in out.split('\n'):
                    if 'Original:' in line or 'Final:' in line or 'Removed:' in line:
                        print(f"      {line.strip()}")
            
            print_success("Far point cloud denoised")
            print_file_saved(str(denoised_far))
            self.outputs['denoised_far'] = str(denoised_far)
        else:
            print_error("Denoising failed")
            return False
        
        # Near mesh - light denoising or skip
        near_ply = self.outputs.get('near_ply')
        if near_ply and os.path.exists(near_ply):
            print_progress("Copying near point cloud (minimal noise expected)...")
            denoised_near = self.dirs['denoised'] / "near_denoised.ply"
            shutil.copy(near_ply, denoised_near)
            self.outputs['denoised_near'] = str(denoised_near)
            print_file_saved(str(denoised_near))
        
        self.save_stage_output('04_denoising', {
            'denoised_far': self.outputs.get('denoised_far'),
            'denoised_near': self.outputs.get('denoised_near'),
            'success': True
        })
        
        return True
    
    # --------------------------------------------------------
    # STAGE 5: Mesh Reconstruction
    # --------------------------------------------------------
    
    def stage_mesh_reconstruction(self) -> bool:
        """Reconstruct meshes from point clouds"""
        print_stage(5, 9, "MESH RECONSTRUCTION", Config.ENVS['mesh'])
        
        success = True
        
        # Far mesh
        denoised_far = self.outputs.get('denoised_far')
        if denoised_far and os.path.exists(denoised_far):
            print_progress("Reconstructing far mesh...")
            print_info("Poisson depth", Config.POISSON_DEPTH)
            print_info("Normal neighbors", Config.NORMAL_NEIGHBORS)
            
            far_mesh = self.dirs['mesh_far'] / "far_mesh.ply"
            
            cmd = [
                "python", self.base_dir / "mesh" / "mesh_algo_cli.py",
                "--input", denoised_far,
                "--output", str(far_mesh),
                "--normal-neighbours", str(Config.NORMAL_NEIGHBORS),
                "--normal-smooth-iter", str(Config.NORMAL_SMOOTH),
                "--poisson-depth", str(Config.POISSON_DEPTH),
                "--hc-iterations", str(Config.HC_ITERATIONS)
            ]
            
            ok, out, _ = self.run_command(cmd, Config.ENVS['mesh'])
            
            if ok and far_mesh.exists():
                # Show stats
                for line in out.split('\n'):
                    if 'vertices' in line.lower() or 'faces' in line.lower():
                        print(f"      {line.strip()}")
                
                print_success("Far mesh reconstructed")
                print_file_saved(str(far_mesh))
                self.outputs['far_mesh'] = str(far_mesh)
            else:
                print_error("Far mesh reconstruction failed")
                success = False
        
        # Near mesh
        denoised_near = self.outputs.get('denoised_near')
        if denoised_near and os.path.exists(denoised_near):
            print_progress("Reconstructing near mesh...")
            
            near_mesh = self.dirs['mesh_near'] / "near_mesh.ply"
            
            cmd = [
                "python", self.base_dir / "mesh" / "mesh_algo_cli.py",
                "--input", denoised_near,
                "--output", str(near_mesh),
                "--normal-neighbours", str(Config.NORMAL_NEIGHBORS),
                "--normal-smooth-iter", str(Config.NORMAL_SMOOTH),
                "--poisson-depth", str(Config.POISSON_DEPTH),
                "--hc-iterations", str(Config.HC_ITERATIONS)
            ]
            
            ok, out, _ = self.run_command(cmd, Config.ENVS['mesh'])
            
            if ok and near_mesh.exists():
                print_success("Near mesh reconstructed")
                print_file_saved(str(near_mesh))
                self.outputs['near_mesh'] = str(near_mesh)
            else:
                print_error("Near mesh reconstruction failed")
                success = False
        
        self.save_stage_output('05_mesh', {
            'far_mesh': self.outputs.get('far_mesh'),
            'near_mesh': self.outputs.get('near_mesh'),
            'success': success
        })
        
        return success
    
    # --------------------------------------------------------
    # STAGE 6: Ground Plane Estimation
    # --------------------------------------------------------
    
    def stage_ground_alignment(self) -> bool:
        """Estimate ground plane and align Y-axis"""
        print_stage(6, 9, "GROUND PLANE ESTIMATION", Config.ENVS['ground'])
        
        far_mesh = self.outputs.get('far_mesh')
        if not far_mesh or not os.path.exists(far_mesh):
            print_error("Far mesh not found")
            return False
        
        print_progress("Estimating ground plane using RANSAC...")
        
        cmd = [
            "python", self.base_dir / "alignment" / "gnd_estimate" / "auto_gnd_estimate.py",
            far_mesh,
            "--no-vis"
        ]
        
        ok, out, _ = self.run_command(cmd, Config.ENVS['ground'])
        
        # Find generated JSON
        mesh_stem = Path(far_mesh).stem
        mesh_dir = Path(far_mesh).parent
        ground_json = mesh_dir / f"{mesh_stem}_automated_ground_plane.json"
        
        if ok and ground_json.exists():
            # Copy to aligned folder
            shutil.copy(ground_json, self.dirs['aligned'] / ground_json.name)
            
            # Show plane info
            with open(ground_json) as f:
                plane_data = json.load(f)
            
            if 'plane_equation' in plane_data:
                eq = plane_data['plane_equation']
                print_info("Plane equation", eq.get('equation', 'N/A')[:50])
            
            print_success("Ground plane estimated")
            print_file_saved(str(ground_json))
            self.outputs['ground_json'] = str(ground_json)
        else:
            print_error("Ground plane estimation failed")
            return False
        
        self.save_stage_output('06_ground', {
            'ground_json': self.outputs.get('ground_json'),
            'success': True
        })
        
        return True
    
    # --------------------------------------------------------
    # STAGE 7: X-Axis Alignment (Foot Detection)
    # --------------------------------------------------------
    
    def stage_xaxis_alignment(self) -> bool:
        """Align X-axis using foot skeleton detection"""
        print_stage(7, 9, "X-AXIS ALIGNMENT", Config.ENVS['xaxis'])
        
        far_mesh = self.outputs.get('far_mesh')
        ground_json = self.outputs.get('ground_json')
        
        if not far_mesh or not os.path.exists(far_mesh):
            print_error("Far mesh not found")
            return False
        
        print_warning("This stage requires Gradio interface for skeleton alignment")
        print()
        print(f"    {Colors.BOLD}Instructions:{Colors.END}")
        print(f"    1. Open a new terminal")
        print(f"    2. Run: {Colors.CYAN}conda activate {Config.ENVS['xaxis']}{Colors.END}")
        print(f"    3. Run: {Colors.CYAN}cd {self.base_dir / 'alignment' / 'X_axis' / 'foot_seeding' / 'main_focus'}{Colors.END}")
        print(f"    4. Run: {Colors.CYAN}python skeleton_pca.py{Colors.END}")
        print()
        print(f"    {Colors.BOLD}Upload:{Colors.END}")
        print(f"    - Mesh: {Colors.GREEN}{far_mesh}{Colors.END}")
        if ground_json:
            print(f"    - Ground JSON: {Colors.GREEN}{ground_json}{Colors.END}")
        print()
        print(f"    {Colors.BOLD}Save aligned mesh to:{Colors.END}")
        print(f"    - {Colors.GREEN}{self.dirs['aligned'] / 'far_aligned.ply'}{Colors.END}")
        print()
        
        input(f"    {Colors.YELLOW}Press Enter when X-axis alignment is complete...{Colors.END}")
        
        # Check for aligned mesh
        aligned_far = self.dirs['aligned'] / "far_aligned.ply"
        
        if not aligned_far.exists():
            # Try to find any PLY in skeleton output
            skeleton_out = self.base_dir / "alignment" / "X_axis" / "foot_seeding" / "main_focus" / "skeleton_out"
            if skeleton_out.exists():
                aligned_files = list(skeleton_out.glob("aligned*.ply")) + list(skeleton_out.glob("rotated*.ply"))
                if aligned_files:
                    shutil.copy(aligned_files[0], aligned_far)
        
        if not aligned_far.exists():
            path = input("    Enter path to aligned far mesh (or press Enter to use original): ").strip()
            if path and os.path.exists(path):
                shutil.copy(path, aligned_far)
            else:
                # Use original mesh
                shutil.copy(far_mesh, aligned_far)
                print_warning("Using original mesh (no alignment applied)")
        
        if aligned_far.exists():
            print_success("X-axis aligned mesh saved")
            print_file_saved(str(aligned_far))
            self.outputs['aligned_far'] = str(aligned_far)
        
        self.save_stage_output('07_xaxis', {
            'aligned_far': self.outputs.get('aligned_far'),
            'success': aligned_far.exists()
        })
        
        return aligned_far.exists()
    
    # --------------------------------------------------------
    # STAGE 8: Manual Alignment (SKIP - Requires user interaction)
    # --------------------------------------------------------
    
    def stage_manual_alignment(self) -> bool:
        """Manual near-to-far mesh alignment - SKIPPED"""
        print_stage(8, 9, "MANUAL ALIGNMENT", Config.ENVS['mesh'])
        
        print_warning("Skipping manual alignment (requires user interaction)")
        print()
        print(f"    {Colors.BOLD}To run manually later:{Colors.END}")
        print(f"    1. {Colors.CYAN}conda activate {Config.ENVS['mesh']}{Colors.END}")
        print(f"    2. {Colors.CYAN}cd {self.base_dir / 'alignment' / 'manual_alignment'}{Colors.END}")
        print(f"    3. {Colors.CYAN}python manual_alignment_v2.py{Colors.END}")
        print()
        print(f"    Source (to align): {self.outputs.get('near_mesh', 'N/A')}")
        print(f"    Target (reference): {self.outputs.get('aligned_far', 'N/A')}")
        print()
        
        # For AIX, we'll use the near mesh directly
        near_mesh = self.outputs.get('near_mesh')
        if near_mesh and os.path.exists(near_mesh):
            aligned_near = self.dirs['aligned'] / "near_mesh.ply"
            shutil.copy(near_mesh, aligned_near)
            self.outputs['aligned_near'] = str(aligned_near)
            print_info("Near mesh copied to", str(aligned_near))
        
        self.save_stage_output('08_manual', {
            'skipped': True,
            'near_mesh': self.outputs.get('near_mesh'),
            'aligned_near': self.outputs.get('aligned_near')
        })
        
        return True
    
    # --------------------------------------------------------
    # STAGE 9: AIX Estimation
    # --------------------------------------------------------
    
    def stage_aix_estimation(self) -> bool:
        """Run all AIX estimations"""
        print_stage(9, 9, "AIX ESTIMATION", Config.ENVS['aix'])
        
        # Use aligned near mesh or fall back to regular near mesh
        input_mesh = self.outputs.get('aligned_near') or self.outputs.get('near_mesh')
        
        if not input_mesh or not os.path.exists(input_mesh):
            # Fall back to aligned far mesh
            input_mesh = self.outputs.get('aligned_far')
        
        if not input_mesh or not os.path.exists(input_mesh):
            print_error("No mesh available for AIX estimation")
            return False
        
        print_info("Input mesh", input_mesh)
        success = True
        
        # --- Hip AIX ---
        print_progress("Running Hip AIX (hip-neck offset)...")
        
        cmd = [
            "python", self.base_dir / "aix" / "Hip_aix" / "hip_aix_bend.py",
            "--mesh", input_mesh,
            "--slice-frac", str(Config.SLICE_FRAC),
            "--no-vis"
        ]
        
        ok, out, _ = self.run_command(cmd, Config.ENVS['aix'])
        
        if ok:
            # Show results
            for line in out.split('\n'):
                if any(k in line for k in ['Hip center', 'Neck center', 'Δx', 'Angle']):
                    print(f"      {line.strip()}")
            
            # Find and copy output JSON
            mesh_dir = Path(input_mesh).parent
            mesh_stem = Path(input_mesh).stem
            hip_json = mesh_dir / f"{mesh_stem}_hip_stats.json"
            if hip_json.exists():
                shutil.copy(hip_json, self.dirs['aix'] / hip_json.name)
                print_file_saved(str(self.dirs['aix'] / hip_json.name))
            
            print_success("Hip AIX completed")
        else:
            print_error("Hip AIX failed")
            success = False
        
        # --- Spine AIX ---
        print_progress("Running Spine AIX (mid-sagittal plane)...")
        
        cmd = [
            "python", self.base_dir / "aix" / "Spine_aix" / "spine_aix.py",
            input_mesh
        ]
        
        ok, out, _ = self.run_command(cmd, Config.ENVS['aix'])
        
        if ok:
            # Show results
            for line in out.split('\n'):
                if any(k in line for k in ['plane', 'normal', 'OK', 'saved']):
                    print(f"      {line.strip()}")
            
            # Find and copy output JSONs
            mesh_dir = Path(input_mesh).parent
            mesh_stem = Path(input_mesh).stem
            
            for pattern in ['*mid_sagittal*.json', '*midline*.json']:
                for f in mesh_dir.glob(pattern):
                    shutil.copy(f, self.dirs['aix'] / f.name)
                    self.outputs['midline_json'] = str(self.dirs['aix'] / f.name)
            
            print_success("Spine AIX completed")
        else:
            print_error("Spine AIX failed")
            success = False
        
        # --- Hump AIX ---
        midline_json = self.outputs.get('midline_json')
        
        if midline_json and os.path.exists(midline_json):
            print_progress("Running Hump AIX (rib hump angle)...")
            
            out_prefix = str(self.dirs['aix'] / "hump")
            
            cmd = [
                "python", self.base_dir / "aix" / "Hump_aix" / "hump_aix.py",
                "--mesh_path", input_mesh,
                "--midline_json", midline_json,
                "--out_prefix", out_prefix
            ]
            
            ok, out, _ = self.run_command(cmd, Config.ENVS['aix'])
            
            if ok:
                for line in out.split('\n'):
                    if any(k in line for k in ['Volume', 'angle', 'Angle', 'asymmetry']):
                        print(f"      {line.strip()}")
                
                print_success("Hump AIX completed")
            else:
                print_error("Hump AIX failed (non-critical)")
        else:
            print_warning("Skipping Hump AIX (no midline JSON)")
        
        self.save_stage_output('09_aix', {
            'input_mesh': input_mesh,
            'results_dir': str(self.dirs['aix']),
            'success': success
        })
        
        return success
    
    # --------------------------------------------------------
    # MAIN EXECUTION
    # --------------------------------------------------------
    
    def run(self) -> bool:
        """Execute the full pipeline"""
        start_time = time.time()
        
        print_header("3D BODY SCANNING PIPELINE")
        
        print_info("Far video", self.far_video)
        print_info("Near video", self.near_video)
        print_info("Working directory", str(self.work_dir))
        print_info("Log file", str(self.log_file))
        print()
        
        self.log("Pipeline started")
        self.log(f"Far video: {self.far_video}")
        self.log(f"Near video: {self.near_video}")
        
        stages = [
            ("Video to Frames", self.stage_video_to_frames),
            ("AI-based 3D Reconstruction", self.stage_vggt),
            ("GLB to PLY", self.stage_glb_to_ply),
            ("Denoising", self.stage_denoising),
            ("Mesh Reconstruction", self.stage_mesh_reconstruction),
            ("Ground Alignment", self.stage_ground_alignment),
            ("X-Axis Alignment", self.stage_xaxis_alignment),
            ("Manual Alignment", self.stage_manual_alignment),
            ("AIX Estimation", self.stage_aix_estimation),
        ]
        
        results = {}
        all_success = True
        
        for name, func in stages:
            try:
                success = func()
                results[name] = success
                if not success:
                    all_success = False
                    print_error(f"Stage '{name}' failed - continuing...")
            except Exception as e:
                results[name] = False
                all_success = False
                print_error(f"Stage '{name}' crashed: {e}")
                self.log(f"Stage '{name}' exception: {e}")
        
        # Summary
        elapsed = time.time() - start_time
        
        print_header("PIPELINE COMPLETE")
        
        print(f"\n    {Colors.BOLD}Stage Results:{Colors.END}")
        for name, success in results.items():
            status = f"{Colors.GREEN}✓ PASS{Colors.END}" if success else f"{Colors.RED}✗ FAIL{Colors.END}"
            print(f"    {status}  {name}")
        
        print(f"\n    {Colors.BOLD}Outputs saved to:{Colors.END} {self.work_dir}")
        print(f"    {Colors.BOLD}Log file:{Colors.END} {self.log_file}")
        print(f"    {Colors.BOLD}Total time:{Colors.END} {elapsed/60:.1f} minutes")
        
        # Save final summary
        summary = {
            'far_video': self.far_video,
            'near_video': self.near_video,
            'working_dir': str(self.work_dir),
            'stages': results,
            'outputs': self.outputs,
            'elapsed_seconds': elapsed,
            'success': all_success
        }
        
        with open(self.work_dir / "pipeline_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if all_success:
            print(f"\n    {Colors.GREEN}{Colors.BOLD}✓ Pipeline completed successfully!{Colors.END}")
        else:
            print(f"\n    {Colors.YELLOW}{Colors.BOLD}⚠ Pipeline completed with some failures{Colors.END}")
        
        print()
        
        self.log(f"Pipeline finished. Success: {all_success}")
        
        return all_success


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated 3D Body Scanning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python run_pipeline.py --far_video test_data/raw_video/p_env_far.mp4 --near_video test_data/raw_video/p_env_close.mp4
    python run_pipeline.py -f path/to/far.mp4 -n path/to/near.mp4 -w my_run
        """
    )
    
    parser.add_argument("--far_video", "-f", required=True,
                        help="Path to FAR video (environment shot)")
    parser.add_argument("--near_video", "-n", required=True,
                        help="Path to NEAR video (close-up)")
    parser.add_argument("--working_dir", "-w",
                        help="Working directory (default: working_directory/run_TIMESTAMP)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.far_video):
        print(f"{Colors.RED}Error: Far video not found: {args.far_video}{Colors.END}")
        sys.exit(1)
    
    if not os.path.exists(args.near_video):
        print(f"{Colors.RED}Error: Near video not found: {args.near_video}{Colors.END}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = Pipeline(
        far_video=args.far_video,
        near_video=args.near_video,
        working_dir=args.working_dir
    )
    
    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

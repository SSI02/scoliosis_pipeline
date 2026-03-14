#!/usr/bin/env python3
"""
FULLY AUTOMATED Web-Based Manual Alignment (Browser Tab + localStorage)
=======================================================================
Opens mesh viewers in separate browser tabs. Points are saved to localStorage
by the viewer's "Save & Close" button. The main Gradio tab polls localStorage
via JavaScript and auto-chains NEAR → FAR → alignment computation.

Integration:
    from manual_alignment_automated import create_automated_alignment_interface
    components = create_automated_alignment_interface(working_dir)
"""

import os
import sys
import json
import shutil
import traceback
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import gradio as gr

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from mesh_viewer import MeshViewerComponent


# ============================================================
# ALIGNMENT MATH  (same as WebBasedAlignment in run_pipeline_gui)
# ============================================================

def compute_similarity_transform(source_pts, target_pts, allow_scale=True):
    """Compute optimal similarity transformation (rotation, translation, scale) via SVD."""
    source_pts = np.array(source_pts, dtype=float)
    target_pts = np.array(target_pts, dtype=float)

    source_center = source_pts.mean(axis=0)
    target_center = target_pts.mean(axis=0)

    source_centered = source_pts - source_center
    target_centered = target_pts - target_center

    if allow_scale:
        source_rms = np.sqrt(np.mean(np.sum(source_centered**2, axis=1)))
        target_rms = np.sqrt(np.mean(np.sum(target_centered**2, axis=1)))
        scale = target_rms / source_rms if source_rms > 1e-8 else 1.0
    else:
        scale = 1.0

    source_scaled = source_centered * scale

    H = source_scaled.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = target_center - scale * R @ source_center

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T, scale


def compute_alignment_error(source_pts, target_pts, transform):
    """Compute RMSE and max error after transformation."""
    source_pts = np.array(source_pts, dtype=float)
    target_pts = np.array(target_pts, dtype=float)

    source_h = np.hstack([source_pts, np.ones((len(source_pts), 1))])
    source_transformed = (transform @ source_h.T).T[:, :3]

    errors = np.linalg.norm(source_transformed - target_pts, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    max_error = errors.max()

    return rmse, max_error, errors


def compute_xaxis_alignment_from_shoulders(shoulder_left, shoulder_right):
    """Compute rotation matrix to align left-to-right shoulder vector with +X axis."""
    shoulder_vec = np.array(shoulder_right) - np.array(shoulder_left)
    shoulder_vec_norm = np.linalg.norm(shoulder_vec)

    if shoulder_vec_norm < 1e-6:
        return np.eye(3), 0.0

    shoulder_vec_xz = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]])
    shoulder_vec_xz_norm = np.linalg.norm(shoulder_vec_xz)

    if shoulder_vec_xz_norm < 1e-6:
        return np.eye(3), 0.0

    shoulder_vec_xz = shoulder_vec_xz / shoulder_vec_xz_norm

    current_angle = np.arctan2(shoulder_vec_xz[2], shoulder_vec_xz[0])
    rotation_angle = current_angle
    angle_degrees = np.degrees(rotation_angle)

    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    return R, angle_degrees


def apply_xaxis_alignment_to_mesh(mesh_path, rotation_matrix, center_point, output_path):
    """Apply X-axis rotation to a mesh around a center point."""
    import trimesh

    try:
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = center_point - rotation_matrix @ center_point
                mesh.apply_transform(transform)
                mesh.export(output_path)
                return True
        except Exception:
            pass

        try:
            pcd = trimesh.load(mesh_path, force='pointcloud')
            if hasattr(pcd, 'vertices') and len(pcd.vertices) > 0:
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = center_point - rotation_matrix @ center_point
                pcd.apply_transform(transform)
                pcd.export(output_path)
                return True
        except Exception:
            pass

        return False
    except Exception as e:
        print(f"Error applying X-axis alignment: {e}")
        return False


def center_mesh_bbox_at_origin(mesh_path, output_dir):
    """Center the mesh's bounding box at the origin."""
    import trimesh

    try:
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            if not (hasattr(mesh, 'vertices') and len(mesh.vertices) > 0):
                raise ValueError("Not a mesh")
        except Exception:
            try:
                mesh = trimesh.load(mesh_path, force='pointcloud')
                if not (hasattr(mesh, 'vertices') and len(mesh.vertices) > 0):
                    return None
            except Exception:
                return None

        bbox_center = mesh.bounds.mean(axis=0)
        transform = np.eye(4)
        transform[:3, 3] = -bbox_center
        mesh.apply_transform(transform)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_stem = Path(mesh_path).stem
        output_path = output_dir / f"{input_stem}_centered.ply"
        mesh.export(str(output_path))
        print(f"[CENTER] Centered mesh saved to: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"ERROR centering mesh: {e}")
        traceback.print_exc()
        return None


# ============================================================
# FULL ALIGNMENT PIPELINE  (replaces start_alignment_web)
# ============================================================

def run_full_alignment(near_pts_json, far_pts_json, near_mesh_path, far_mesh_path, output_dir):
    """
    Run the complete alignment pipeline:
      1. Procrustes/SVD transform (NEAR → FAR)
      2. X-axis alignment from shoulder vector
      3. Center bounding box at origin
      4. Save aligned files

    Returns:
        (success: bool, message: str, stats: dict)
    """
    import trimesh

    try:
        near_pts = json.loads(near_pts_json) if isinstance(near_pts_json, str) else near_pts_json
        far_pts  = json.loads(far_pts_json)  if isinstance(far_pts_json, str)  else far_pts_json
    except Exception as e:
        return False, f"❌ Invalid point data: {e}", {}

    if len(near_pts) != 4:
        return False, f"❌ Need 4 NEAR points (have {len(near_pts)})", {}
    if len(far_pts) != 4:
        return False, f"❌ Need 4 FAR points (have {len(far_pts)})", {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_pts = np.array(near_pts, dtype=float)
    target_pts = np.array(far_pts, dtype=float)

    # ── Step 1: Compute Procrustes transform ───────────────
    transform, scale = compute_similarity_transform(source_pts, target_pts, allow_scale=True)
    rmse, max_error, errors = compute_alignment_error(source_pts, target_pts, transform)

    print(f"[ALIGN] Scale: {scale:.6f}, RMSE: {rmse*1000:.2f} mm, Max: {max_error*1000:.2f} mm")

    # ── Step 2: X-axis alignment from shoulder vector ──────
    target_left_shoulder  = target_pts[1]
    target_right_shoulder = target_pts[2]

    if target_left_shoulder[0] > target_right_shoulder[0]:
        target_left_shoulder, target_right_shoulder = target_right_shoulder, target_left_shoulder
        print("[ALIGN] Swapped shoulders to ensure left-to-right order")

    xaxis_rotation, xaxis_angle = compute_xaxis_alignment_from_shoulders(
        target_left_shoulder, target_right_shoulder
    )
    shoulder_center = (target_left_shoulder + target_right_shoulder) / 2.0
    print(f"[ALIGN] X-axis rotation: {xaxis_angle:.2f}°")

    # ── Step 3: Apply manual alignment to NEAR mesh ────────
    near_output = str(output_dir / 'near_aligned.ply')
    temp_path   = str(output_dir / 'near_aligned_temp.ply')

    try:
        mesh = trimesh.load(near_mesh_path, force='mesh')
        if not (hasattr(mesh, 'vertices') and len(mesh.vertices) > 0):
            raise ValueError("empty")
    except Exception:
        mesh = trimesh.load(near_mesh_path, force='pointcloud')

    mesh.apply_transform(transform)
    mesh.export(temp_path)
    print(f"[ALIGN] Manual alignment applied to NEAR → {temp_path}")

    # ── Step 4: Apply X-axis to FAR mesh ───────────────────
    far_xaxis_path = str(output_dir / 'far_xaxis_aligned.ply')
    success_far = apply_xaxis_alignment_to_mesh(
        far_mesh_path, xaxis_rotation, shoulder_center, far_xaxis_path
    )
    if not success_far:
        print("[ALIGN] X-axis alignment failed for FAR mesh, using original")
        shutil.copy(far_mesh_path, far_xaxis_path)

    # ── Step 5: Apply X-axis to NEAR mesh ──────────────────
    shoulder_center_h = np.append(shoulder_center, 1.0)
    transformed_shoulder_center = (transform @ shoulder_center_h)[:3]

    success_near = apply_xaxis_alignment_to_mesh(
        temp_path, xaxis_rotation, transformed_shoulder_center, near_output
    )
    if not success_near:
        print("[ALIGN] X-axis alignment failed for NEAR mesh, using manual-only")
        shutil.copy(temp_path, near_output)

    # Clean up temp
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # ── Step 6: Center bounding box at origin ──────────────
    centered_path = center_mesh_bbox_at_origin(near_output, str(output_dir))
    final_near_path = centered_path if centered_path and os.path.exists(centered_path) else near_output

    # ── Step 7: Save transform JSON ────────────────────────
    transform_path = near_output.replace('.ply', '_transform.json')
    with open(transform_path, 'w') as f:
        json.dump({
            'transformation': transform.tolist(),
            'scale': float(scale),
            'rmse': float(rmse),
            'max_error': float(max_error),
            'source_points': [p.tolist() if hasattr(p, 'tolist') else p for p in near_pts],
            'target_points': [p.tolist() if hasattr(p, 'tolist') else p for p in far_pts],
            'errors': errors.tolist(),
            'xaxis_angle_deg': float(xaxis_angle),
        }, f, indent=2)

    stats = {
        'scale': round(scale, 6),
        'rmse_mm': round(rmse * 1000, 2),
        'max_error_mm': round(max_error * 1000, 2),
        'xaxis_angle': round(xaxis_angle, 2),
        'near_output': final_near_path,
        'far_output': far_xaxis_path,
    }

    msg = (
        f"✅ ALIGNMENT COMPLETE\n\n"
        f"Scale: {scale:.6f}\n"
        f"RMSE: {rmse*1000:.2f} mm\n"
        f"Max error: {max_error*1000:.2f} mm\n"
        f"X-axis rotation: {xaxis_angle:.2f}°\n\n"
        f"Saved NEAR: {os.path.basename(final_near_path)}\n"
        f"Saved FAR:  {os.path.basename(far_xaxis_path)}\n\n"
        f"Click 'Continue After Alignment' on the Pipeline tab."
    )

    return True, msg, stats


# ============================================================
# GRADIO INTERFACE BUILDER
# ============================================================

def create_automated_alignment_interface(working_dir: str = None):
    """
    Create the fully automated manual alignment interface for integration
    into run_pipeline_gui.py.

    Workflow:
        1. User clicks "Start Alignment"
        2. NEAR mesh viewer opens in a new browser tab
        3. User picks 4 points → Save & Close
        4. Main tab detects near_points in localStorage → auto-opens FAR viewer
        5. User picks 4 points → Save & Close
        6. Main tab detects far_points → runs alignment automatically

    Args:
        working_dir: Working directory path (optional, for auto-populating paths)

    Returns:
        dict with keys: 'start_btn', 'status', 'near_pts_box', 'far_pts_box'
    """

    # --- Hidden state textboxes ---
    near_pts_box = gr.Textbox(value="", visible=False, elem_id="near-pts-hidden")
    far_pts_box  = gr.Textbox(value="", visible=False, elem_id="far-pts-hidden")
    # Stores the paths set by the pipeline executor
    near_mesh_state = gr.State(value=None)
    far_mesh_state  = gr.State(value=None)
    output_dir_state = gr.State(value=None)
    # Stores the viewer URLs for JS to open
    near_viewer_url_state = gr.Textbox(value="", visible=False, elem_id="near-viewer-url")
    far_viewer_url_state  = gr.Textbox(value="", visible=False, elem_id="far-viewer-url")

    # --- Visible UI ---
    gr.Markdown("""
    ## 🎯 Web-Based Manual Alignment

    **Select 4 corresponding landmarks in the SAME order on each mesh:**
    1. **HEAD** (top of head/crown)
    2. **LEFT SHOULDER**
    3. **RIGHT SHOULDER**
    4. **PELVIS MIDPOINT** (center of pelvis/hip)

    The viewers will open in **new browser tabs**.
    After saving points in both tabs, alignment runs automatically.
    """)

    status_box = gr.Textbox(
        label="Status",
        lines=12,
        interactive=False,
        value="Ready — click 'Start Alignment' after pipeline reaches Stage 7."
    )

    with gr.Row():
        start_btn = gr.Button("🚀 Start Alignment", variant="primary", size="lg")

    result_json = gr.JSON(label="Alignment Result", visible=False)

    # ==========================================================
    # CALLBACK: Start Alignment — generate NEAR HTML, return JS
    # ==========================================================
    def on_start_alignment():
        """Generate the NEAR viewer HTML and return JS to open it + start polling."""
        # Access the global executor from the main module.
        # When run_pipeline_gui.py is executed directly, it becomes __main__,
        # so 'import run_pipeline_gui' would create a separate module copy
        # where executor is still None.
        import sys
        main_mod = sys.modules.get('__main__')
        pipe_executor = getattr(main_mod, 'executor', None) if main_mod else None

        if pipe_executor is None:
            return (
                "❌ No pipeline running. Start the pipeline first.",
                gr.update(),  # result_json
                None, None, None,  # states
                "",  # near_pts_box
                "",  # far_pts_box
                "",  # near_viewer_url_state -> NOW HTML CONTENT
            )

        near_mesh = pipe_executor.outputs.get('near_mesh')
        aligned_far = pipe_executor.outputs.get('y_aligned_far') or pipe_executor.outputs.get('far_mesh')
        output_dir = str(pipe_executor.dirs.get('manual_aligned', ''))

        if not near_mesh or not os.path.exists(str(near_mesh)):
            return (
                f"❌ NEAR mesh not found: {near_mesh}",
                gr.update(), None, None, None, "", "", "",
            )
        if not aligned_far or not os.path.exists(str(aligned_far)):
            return (
                f"❌ FAR mesh not found: {aligned_far}",
                gr.update(), None, None, None, "", "", "",
            )

        # Generate the NEAR viewer HTML file (for backup/logging)
        try:
            viewer = MeshViewerComponent(str(near_mesh), max_points=100000, title="NEAR Mesh (Source)")
            near_html_path = viewer.generate_viewer_html_file("near_points")
            
            # Read CONTENT to pass to frontend (bypass 404)
            with open(near_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
        except Exception as e:
            return (
                f"❌ Failed to generate NEAR viewer: {e}",
                gr.update(), None, None, None, "", "", "",
            )

        # Pass FULL HTML CONTENT instead of URL
        viewer_content = html_content

        status_msg = (
            "🔵 Step 1/2: NEAR Mesh Viewer opening in new tab.\n\n"
            "   Select 4 landmarks then click 'Save & Close'.\n\n"
            "   Waiting for NEAR points..."
        )

        return (
            status_msg,
            gr.update(),      # result_json stays hidden
            near_mesh,        # near_mesh_state
            aligned_far,      # far_mesh_state
            output_dir,       # output_dir_state
            "",               # clear near_pts_box
            "",               # clear far_pts_box
            viewer_content,   # near_viewer_url_state (content)
        )

    start_btn.click(
        fn=on_start_alignment,
        inputs=[],
        outputs=[status_box, result_json,
                 near_mesh_state, far_mesh_state, output_dir_state,
                 near_pts_box, far_pts_box, near_viewer_url_state],
    ).then(
        # After Python returns, run JS to clear localStorage, open NEAR viewer (BLOB), start polling
        fn=None,
        inputs=[near_viewer_url_state],
        outputs=[],
        js="""
        (htmlContent) => {
            localStorage.removeItem('near_points');
            localStorage.removeItem('far_points');

            if (!htmlContent) {
                console.log('[Alignment] No content, skipping tab open.');
                return;
            }

            // Create Blob URL from HTML content
            const blob = new Blob([htmlContent], {type: 'text/html'});
            const url = URL.createObjectURL(blob);
            
            const w = window.open(url, '_blank');
            if (!w) {
                alert('Pop-up blocked! Please allow pop-ups for this site and click Start Alignment again.');
                return;
            }

            const poll = setInterval(() => {
                const data = localStorage.getItem('near_points');
                if (data) {
                    clearInterval(poll);
                    const el = document.querySelector('#near-pts-hidden textarea, #near-pts-hidden input');
                    if (el) {
                        el.value = data;
                        el.dispatchEvent(new Event('input', {bubbles: true}));
                    }
                }
            }, 500);
        }
        """
    )

    # ==========================================================
    # CALLBACK: NEAR points received → generate FAR viewer
    # ==========================================================
    def on_near_points_received(near_pts_json, far_mesh):
        """When NEAR points appear, generate the FAR viewer and return JS to open it."""
        if not near_pts_json or near_pts_json.strip() == "":
            return gr.update(), gr.update()  # no-op

        try:
            pts = json.loads(near_pts_json)
            if len(pts) < 1:
                return gr.update(), gr.update()
        except Exception:
            return gr.update(), gr.update()

        if not far_mesh or not os.path.exists(str(far_mesh)):
            return (
                f"🔵 NEAR points received ({len(pts)}).\n"
                f"❌ FAR mesh not found. Cannot continue.",
                ""
            )

        # Generate FAR viewer
        try:
            viewer = MeshViewerComponent(str(far_mesh), max_points=100000, title="FAR Mesh (Target)")
            far_html_path = viewer.generate_viewer_html_file("far_points")
            
            with open(far_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
        except Exception as e:
            return f"❌ Failed to generate FAR viewer: {e}", ""

        viewer_content = html_content

        status_msg = (
            f"✅ NEAR points received ({len(pts)} points).\n\n"
            f"🔵 Step 2/2: FAR Mesh Viewer opening in new tab.\n\n"
            f"   Select 4 landmarks then click 'Save & Close'.\n\n"
            f"   Waiting for FAR points..."
        )

        return status_msg, viewer_content

    near_pts_box.change(
        fn=on_near_points_received,
        inputs=[near_pts_box, far_mesh_state],
        outputs=[status_box, far_viewer_url_state],
    ).then(
        # Open the FAR viewer tab (BLOB) and start polling for far_points
        fn=None,
        inputs=[far_viewer_url_state],
        outputs=[],
        js="""
        (htmlContent) => {
            const nearData = localStorage.getItem('near_points');
            if (!nearData || !htmlContent) return;

            localStorage.removeItem('far_points');

            // Create Blob URL from HTML content
            const blob = new Blob([htmlContent], {type: 'text/html'});
            const url = URL.createObjectURL(blob);

            const w = window.open(url, '_blank');
            if (!w) {
                alert('Pop-up blocked! Please allow pop-ups for this site.');
                return;
            }

            const poll = setInterval(() => {
                const data = localStorage.getItem('far_points');
                if (data) {
                    clearInterval(poll);
                    const el = document.querySelector('#far-pts-hidden textarea, #far-pts-hidden input');
                    if (el) {
                        el.value = data;
                        el.dispatchEvent(new Event('input', {bubbles: true}));
                    }
                }
            }, 500);
        }
        """
    )
    
    # ... rest of the file ...
    
    # ==========================================================
    # CALLBACK: FAR points received → run alignment
    # ==========================================================
    def on_far_points_received(near_pts_json, far_pts_json, near_mesh, far_mesh, output_dir):
        """When FAR points appear, run the full alignment pipeline."""
        if not far_pts_json or far_pts_json.strip() == "":
            return gr.update(), gr.update()

        try:
            far_pts = json.loads(far_pts_json)
            if len(far_pts) < 1:
                return gr.update(), gr.update()
        except Exception:
            return gr.update(), gr.update()

        if not near_pts_json:
            return "❌ NEAR points missing.", gr.update()
        if not near_mesh or not far_mesh or not output_dir:
            return "❌ Mesh paths not set.", gr.update()

        status = (
            f"✅ FAR points received ({len(far_pts)} points).\n\n"
            f"⏳ Computing alignment..."
        )

        # Run alignment
        success, msg, stats = run_full_alignment(
            near_pts_json, far_pts_json,
            str(near_mesh), str(far_mesh), str(output_dir)
        )

        if success:
            # Update the global executor state
            try:
                import sys
                main_mod = sys.modules.get('__main__')
                pipe_executor = getattr(main_mod, 'executor', None) if main_mod else None

                if pipe_executor is not None:
                    pipe_executor.outputs['aligned_near'] = stats.get('near_output', '')
                    pipe_executor.outputs['aligned_far'] = stats.get('far_output', '')
                    main_mod.manual_alignment_path = stats.get('near_output', '')
                    pipe_executor.log("MANUAL ALIGNMENT COMPLETE", "SUCCESS")
                    pipe_executor.log(f"Scale: {stats['scale']}, RMSE: {stats['rmse_mm']} mm", "INFO")
                    pipe_executor.log(f"Saved: {stats['near_output']}", "FILE")
                    pipe_executor.log("Click 'Continue After Alignment' on Pipeline tab", "INFO")
            except Exception as e:
                print(f"[WARN] Could not update executor state: {e}")

            return msg, gr.update(value=stats, visible=True)
        else:
            return msg, gr.update(value=stats, visible=True) if stats else gr.update()

    far_pts_box.change(
        fn=on_far_points_received,
        inputs=[near_pts_box, far_pts_box, near_mesh_state, far_mesh_state, output_dir_state],
        outputs=[status_box, result_json],
    )

    return {
        'start_btn': start_btn,
        'status': status_box,
        'near_pts_box': near_pts_box,
        'far_pts_box': far_pts_box,
        'near_mesh_state': near_mesh_state,
        'far_mesh_state': far_mesh_state,
        'output_dir_state': output_dir_state,
    }


# ============================================================
# STANDALONE TEST MODE
# ============================================================

if __name__ == "__main__":
    with gr.Blocks(title="Automated Manual Alignment") as demo:
        create_automated_alignment_interface()

    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)

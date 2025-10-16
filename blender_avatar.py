#!/usr/bin/env python3
"""
blender_avatar.py - Wrapper for Blender avatar generation

FIXED: Now properly handles base64 photo data by writing to temp files
"""

import os
import sys
import base64
import subprocess
import tempfile
import json
import shutil
from pathlib import Path
from typing import Optional


def _write_b64_to_file(b64_string: str, output_dir: str, prefix: str) -> Optional[str]:
    """
    Write base64 image data to a temporary file.
    
    Args:
        b64_string: Base64 encoded image
        output_dir: Directory to write file to
        prefix: Filename prefix (e.g., "front", "side")
        
    Returns:
        File path or None if failed
    """
    if not b64_string:
        return None
    
    try:
        # Handle data URI format
        if b64_string.startswith('data:'):
            b64_string = b64_string.split(',', 1)[1]
        
        # Decode base64
        img_bytes = base64.b64decode(b64_string)
        
        # Write to temp file
        filepath = os.path.join(output_dir, f"{prefix}.jpg")
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
        
        return filepath
    except Exception as e:
        print(f"[BLENDER] Warning: Failed to write {prefix} photo: {e}", file=sys.stderr)
        return None


def run_blender_avatar(
    preset: str,
    height_m: float,
    measurements: dict,
    photos: dict,
    tex_res: int = 2048,
    photos_ranked: dict = None,
    high_detail: bool = False,
    pose_mode: str = "neutral",
    pose_angles: dict = None
) -> dict:
    """
    Generate a 3D avatar GLB using Blender.
    
    Args:
        preset: Body type preset ("male", "female", "neutral", "child", "baby")
        height_m: Height in meters
        measurements: Dict with optional keys: chest, waist, hips, shoulder, inseam, arm
        photos: Dict with keys: front, side, back (base64 strings)
        tex_res: Texture resolution (512-8192)
        photos_ranked: Optional dict with ranked photo lists per role (base64 strings)
        high_detail: Force high detail baking (≥4K)
        pose_mode: "neutral" or "auto"
        pose_angles: Dict with pose angles if pose_mode="auto"
        
    Returns:
        dict: {
            "ok": bool,
            "glb_b64": str (if success),
            "error": str (if failure),
            "log": str (Blender output),
            "file_size": int (if success)
        }
    """
    
    # Create unique temp directory for this avatar
    output_dir = tempfile.mkdtemp(prefix="avatar_")
    output_file = os.path.join(output_dir, "twin.glb")
    
    print(f"[BLENDER] Created temp directory: {output_dir}")
    print(f"[BLENDER] Target output file: {output_file}")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate Blender binary
    blender_bin = os.environ.get("BLENDER_BIN", "/blender/blender")
    if not os.path.exists(blender_bin):
        return {
            "ok": False,
            "error": f"Blender binary not found at {blender_bin}",
            "log": ""
        }
    
    # Locate base .blend file
    base_blend = f"/app/assets/base_{preset}.blend"
    if not os.path.exists(base_blend):
        print(f"[BLENDER] Warning: {base_blend} not found, falling back to base_female.blend")
        base_blend = "/app/assets/base_female.blend"
    
    if not os.path.exists(base_blend):
        return {
            "ok": False,
            "error": f"Base blend file not found: {base_blend}",
            "log": ""
        }
    
    print(f"[BLENDER] Using base blend: {base_blend}")
    
    # Write base64 photos to temp files
    photo_files = {}
    if photos:
        print(f"[BLENDER] Converting base64 photos to files...")
        if photos.get("front"):
            photo_files["front"] = _write_b64_to_file(photos["front"], output_dir, "front")
        if photos.get("side"):
            photo_files["side"] = _write_b64_to_file(photos["side"], output_dir, "side")
        if photos.get("back"):
            photo_files["back"] = _write_b64_to_file(photos["back"], output_dir, "back")
    
    # Build Blender command
    cmd = [
        blender_bin,
        "-b", base_blend,
        "--python", "/app/deform_avatar.py",
        "--"
    ]
    
    # Add core arguments
    cmd.extend(["--preset", preset])
    cmd.extend(["--height", str(height_m)])
    cmd.extend(["--texRes", str(tex_res)])
    
    # CRITICAL: Pass output path explicitly
    cmd.extend(["--out", output_file])
    
    # High detail flag
    if high_detail:
        cmd.append("--highDetail")
    
    # Add measurements (only if not None)
    measurement_keys = ["chest", "waist", "hips", "shoulder", "inseam", "arm"]
    for key in measurement_keys:
        value = measurements.get(key)
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    # Add single photo paths (from converted files)
    if photo_files.get("front"):
        cmd.extend(["--frontTex", photo_files["front"]])
        print(f"[BLENDER] Front photo: {photo_files['front']}")
    if photo_files.get("side"):
        cmd.extend(["--sideTex", photo_files["side"]])
        print(f"[BLENDER] Side photo: {photo_files['side']}")
    if photo_files.get("back"):
        cmd.extend(["--backTex", photo_files["back"]])
        print(f"[BLENDER] Back photo: {photo_files['back']}")
    
    # Handle multi-photo ranked lists (if available)
    # photos_ranked contains base64 strings, need to convert them
    if photos_ranked:
        print(f"[BLENDER] Processing ranked photos...")
        
        # Front photos
        if photos_ranked.get("front") and isinstance(photos_ranked["front"], list):
            front_files = []
            for idx, b64_photo in enumerate(photos_ranked["front"][:2]):  # Max 2
                if isinstance(b64_photo, str):
                    filepath = _write_b64_to_file(b64_photo, output_dir, f"front_{idx}")
                    if filepath:
                        front_files.append(filepath)
            
            if front_files:
                front_list = ";".join(front_files)
                cmd.extend(["--frontTexList", front_list])
                print(f"[BLENDER] Front ranked photos: {len(front_files)} images")
        
        # Side photos
        if photos_ranked.get("side") and isinstance(photos_ranked["side"], list):
            side_files = []
            for idx, b64_photo in enumerate(photos_ranked["side"][:2]):
                if isinstance(b64_photo, str):
                    filepath = _write_b64_to_file(b64_photo, output_dir, f"side_{idx}")
                    if filepath:
                        side_files.append(filepath)
            
            if side_files:
                side_list = ";".join(side_files)
                cmd.extend(["--sideTexList", side_list])
                print(f"[BLENDER] Side ranked photos: {len(side_files)} images")
        
        # Back photos
        if photos_ranked.get("back") and isinstance(photos_ranked["back"], list):
            back_files = []
            for idx, b64_photo in enumerate(photos_ranked["back"][:2]):
                if isinstance(b64_photo, str):
                    filepath = _write_b64_to_file(b64_photo, output_dir, f"back_{idx}")
                    if filepath:
                        back_files.append(filepath)
            
            if back_files:
                back_list = ";".join(back_files)
                cmd.extend(["--backTexList", back_list])
                print(f"[BLENDER] Back ranked photos: {len(back_files)} images")
    
    # Handle pose (if auto mode and angles provided)
    pose_json_path = None
    if pose_mode == "auto" and pose_angles:
        pose_json_path = os.path.join(output_dir, "pose.json")
        try:
            with open(pose_json_path, 'w') as f:
                json.dump(pose_angles, f, indent=2)
            cmd.extend(["--poseJson", pose_json_path])
            print(f"[BLENDER] Pose mode enabled with angles: {pose_angles}")
        except Exception as e:
            print(f"[BLENDER] Warning: Failed to write pose JSON: {e}")
    
    # Set environment variable as backup method
    env = os.environ.copy()
    env["OUTPUT_GLTF"] = output_file
    
    # Log command (truncated for security)
    cmd_display = ' '.join(cmd[:10]) + "..." if len(cmd) > 10 else ' '.join(cmd)
    print(f"[BLENDER] Starting avatar generation: preset={preset}, height={height_m}m, texRes={tex_res}")
    print(f"[BLENDER] Executing command: {cmd_display}")
    
    # Execute Blender with timeout
    timeout = 300  # 5 minutes
    print(f"[BLENDER] Timeout set to {timeout}s")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        # Combine stdout and stderr for log
        log = ""
        if result.stdout:
            log += "=== STDOUT ===\n" + result.stdout + "\n"
        if result.stderr:
            log += "=== STDERR ===\n" + result.stderr + "\n"
        
        print(f"[BLENDER] Blender process finished with return code: {result.returncode}")
        
        # Check if output file was created
        if not os.path.exists(output_file):
            # File not found - detailed error reporting
            print(f"[BLENDER] ERROR: Output file not found at {output_file}", file=sys.stderr)
            print(f"[BLENDER] Output directory exists: {os.path.exists(output_dir)}", file=sys.stderr)
            
            if os.path.exists(output_dir):
                dir_contents = os.listdir(output_dir)
                print(f"[BLENDER] Directory contents: {dir_contents}", file=sys.stderr)
            else:
                print(f"[BLENDER] Output directory does not exist!", file=sys.stderr)
            
            # Print last 50 lines of output for debugging
            if result.stdout:
                stdout_lines = result.stdout.split('\n')
                print(f"[BLENDER] Last 50 lines of stdout:", file=sys.stderr)
                for line in stdout_lines[-50:]:
                    print(f"  {line}", file=sys.stderr)
            
            if result.stderr:
                stderr_lines = result.stderr.split('\n')
                print(f"[BLENDER] Last 50 lines of stderr:", file=sys.stderr)
                for line in stderr_lines[-50:]:
                    print(f"  {line}", file=sys.stderr)
            
            return {
                "ok": False,
                "error": f"Blender completed with code {result.returncode} but output file not found at {output_file}",
                "log": log,
                "returncode": result.returncode,
                "output_dir": output_dir,
                "dir_contents": os.listdir(output_dir) if os.path.exists(output_dir) else []
            }
        
        # File exists - read and encode it
        file_size = os.path.getsize(output_file)
        print(f"[BLENDER] Success! GLB file created")
        print(f"[BLENDER] File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # Verify file is not empty
        if file_size == 0:
            return {
                "ok": False,
                "error": f"Output file created but is empty (0 bytes)",
                "log": log
            }
        
        # Read GLB file as binary
        try:
            with open(output_file, "rb") as f:
                glb_bytes = f.read()
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to read output file: {str(e)}",
                "log": log
            }
        
        # Encode to base64
        try:
            glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to encode GLB to base64: {str(e)}",
                "log": log
            }
        
        print(f"[BLENDER] GLB encoded to base64: {len(glb_b64)} characters")
        
        # Cleanup temp files
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            print("[BLENDER] Temp files cleaned up")
        except Exception as e:
            print(f"[BLENDER] Warning: Cleanup failed: {e}")
        
        # Return success
        return {
            "ok": True,
            "glb_b64": glb_b64,
            "log": log,
            "file_size": file_size,
            "returncode": result.returncode
        }
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"Blender execution timed out after {timeout} seconds"
        print(f"[BLENDER] ERROR: {error_msg}", file=sys.stderr)
        
        # Try to get partial output
        partial_log = ""
        if hasattr(e, 'stdout') and e.stdout:
            partial_log += "=== PARTIAL STDOUT ===\n" + e.stdout + "\n"
        if hasattr(e, 'stderr') and e.stderr:
            partial_log += "=== PARTIAL STDERR ===\n" + e.stderr + "\n"
        
        # Cleanup on timeout
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except Exception:
            pass
        
        return {
            "ok": False,
            "error": error_msg,
            "log": partial_log or "Process timed out before producing output"
        }
        
    except Exception as e:
        error_msg = f"Unexpected error during Blender execution: {str(e)}"
        print(f"[BLENDER] ERROR: {error_msg}", file=sys.stderr)
        
        # Cleanup on error
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except Exception:
            pass
        
        return {
            "ok": False,
            "error": error_msg,
            "log": str(e)
        }

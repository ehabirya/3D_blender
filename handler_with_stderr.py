#!/usr/bin/env python3
"""
handler_with_stderr.py

Add this to your handler/orchestrator script to capture full Blender errors.
This is the file that calls Blender as a subprocess.
"""

import subprocess
import os
import sys
from pathlib import Path


def run_blender_with_error_capture(cmd, temp_dir):
    """
    Run Blender command and capture both stdout and stderr.
    
    Args:
        cmd: List of command arguments
        temp_dir: Temporary directory for the avatar
        
    Returns:
        dict with success status, output, and error details
    """
    print(f"[BLENDER] Executing: {' '.join(cmd[:5])} ...")
    print(f"[BLENDER] Temp dir: {temp_dir}")
    
    # Create log files
    stdout_log = os.path.join(temp_dir, "blender_stdout.log")
    stderr_log = os.path.join(temp_dir, "blender_stderr.log")
    
    try:
        # Run Blender process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Get string output instead of bytes
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Capture output in real-time
        stdout_lines = []
        stderr_lines = []
        
        # Read both streams
        stdout_data, stderr_data = process.communicate()
        
        # Split into lines
        stdout_lines = stdout_data.splitlines() if stdout_data else []
        stderr_lines = stderr_data.splitlines() if stderr_data else []
        
        # Log stdout
        print("\n[BLENDER] STDOUT:")
        for line in stdout_lines:
            print(f"  {line}")
            
        # Save stdout to file
        with open(stdout_log, 'w') as f:
            f.write(stdout_data)
        print(f"[BLENDER] Stdout saved to: {stdout_log}")
        
        # Log stderr (THIS IS CRITICAL - often contains Python errors)
        if stderr_lines:
            print("\n[BLENDER] STDERR:")
            for line in stderr_lines:
                print(f"  {line}")
            
            # Save stderr to file
            with open(stderr_log, 'w') as f:
                f.write(stderr_data)
            print(f"[BLENDER] Stderr saved to: {stderr_log}")
        
        # Get return code
        return_code = process.returncode
        print(f"\n[BLENDER] Return code: {return_code}")
        
        # Parse output for specific errors
        error_indicators = [
            "Error:",
            "ERROR:",
            "Traceback",
            "Exception",
            "FATAL",
            "mesh verification failed",
            "Mesh verification failed"
        ]
        
        found_errors = []
        for line in stdout_lines + stderr_lines:
            for indicator in error_indicators:
                if indicator.lower() in line.lower():
                    found_errors.append(line)
        
        if found_errors:
            print("\n[BLENDER] Detected errors in output:")
            for err in found_errors:
                print(f"  ✗ {err}")
        
        return {
            "success": return_code == 0,
            "return_code": return_code,
            "stdout": stdout_lines,
            "stderr": stderr_lines,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "errors": found_errors
        }
        
    except Exception as e:
        print(f"[BLENDER] ✗ Exception running Blender: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "return_code": -1,
            "stdout": [],
            "stderr": [str(e)],
            "errors": [str(e)]
        }


def build_blender_command(preset, height, measurements, photos, tex_res, pose_json, output_glb):
    """
    Build the Blender command with all parameters.
    
    Args:
        preset: Avatar preset (male/female/neutral/child/baby)
        height: Height in meters
        measurements: Dict of body measurements
        photos: Dict with 'front', 'side', 'back' lists of photo paths
        tex_res: Texture resolution
        pose_json: Path to pose JSON (or None)
        output_glb: Output GLB file path
        
    Returns:
        List of command arguments
    """
    # Base command
    cmd = [
        "/blender/blender",  # Adjust path as needed
        "-b",  # Background mode
        f"/app/assets/base_{preset}.blend",  # Base blend file
        "--python", "/app/deform_avatar.py",  # Your script
        "--",  # Separator for script arguments
    ]
    
    # Basic parameters
    cmd.extend([
        "--preset", preset,
        "--height", str(height),
        "--texRes", str(tex_res),
        "--out", output_glb,
    ])
    
    # Add measurements
    for key, value in measurements.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    # Add photos
    if photos.get('front'):
        cmd.extend(["--frontTexList", ";".join(photos['front'])])
    if photos.get('side'):
        cmd.extend(["--sideTexList", ";".join(photos['side'])])
    if photos.get('back'):
        cmd.extend(["--backTexList", ";".join(photos['back'])])
    
    # Add pose if provided
    if pose_json and os.path.exists(pose_json):
        cmd.extend(["--poseJson", pose_json])
    
    return cmd


def main_handler(preset, height, measurements, photos, tex_res, pose, output_glb, temp_dir):
    """
    Main handler function that orchestrates the avatar generation.
    
    This replaces your existing handler function.
    """
    print("[HANDLER] Starting Blender render:")
    print(f"  preset={preset}, height={height}m, texRes={tex_res}, pose={pose}")
    
    # Build command
    cmd = build_blender_command(
        preset=preset,
        height=height,
        measurements=measurements,
        photos=photos,
        tex_res=tex_res,
        pose_json=pose if isinstance(pose, str) else None,
        output_glb=output_glb
    )
    
    # Run Blender with error capture
    result = run_blender_with_error_capture(cmd, temp_dir)
    
    # Check results
    if not result["success"]:
        print(f"\n[HANDLER] ✗ Blender failed with return code {result['return_code']}")
        
        # Print error summary
        if result["errors"]:
            print("\n[HANDLER] Error summary:")
            for err in result["errors"]:
                print(f"  {err}")
        
        print(f"\n[HANDLER] Check logs:")
        print(f"  stdout: {result['stdout_log']}")
        print(f"  stderr: {result['stderr_log']}")
        
        return False, "Blender execution failed"
    
    # Check if GLB was created
    if not os.path.exists(output_glb):
        print(f"\n[HANDLER] ✗ Blender finished (code {result['return_code']}) but GLB missing")
        print(f"[HANDLER] Expected: {output_glb}")
        
        # This is your current error!
        print("\n[HANDLER] This usually means:")
        print("  1. Mesh verification failed (check stderr)")
        print("  2. Export failed (check stderr)")
        print("  3. Wrong output path")
        
        return False, f"GLB not found at {output_glb}"
    
    # Success!
    size_mb = os.path.getsize(output_glb) / 1024 / 1024
    print(f"\n[HANDLER] ✓ SUCCESS: {size_mb:.2f} MB")
    print(f"[HANDLER] Output: {output_glb}")
    
    return True, output_glb


# Example usage:
if __name__ == "__main__":
    # Example parameters
    measurements = {
        "chest": 0.85,
        "waist": 0.72,
        "hips": 0.95,
        "shoulder": 0.42,
        "inseam": 0.78,
        "arm": 0.65,
    }
    
    photos = {
        "front": ["/path/to/front1.jpg", "/path/to/front2.jpg"],
        "side": ["/path/to/side1.jpg"],
        "back": ["/path/to/back1.jpg"],
    }
    
    success, result = main_handler(
        preset="female",
        height=1.72,
        measurements=measurements,
        photos=photos,
        tex_res=2048,
        pose="auto",
        output_glb="/tmp/avatar_test/twin.glb",
        temp_dir="/tmp/avatar_test"
    )
    
    if success:
        print(f"Avatar generated: {result}")
    else:
        print(f"Failed: {result}")
        sys.exit(1)

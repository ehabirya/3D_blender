# app/blender_avatar.py - IMPROVED VERSION WITH CRITICAL FIXES
"""
Wrapper to run Blender headless and produce a GLB avatar.

IMPROVEMENTS IN THIS VERSION:
- Added timeout to Blender subprocess (300s)
- Added GLB file size validation (< 7MB)
- Improved error messages with exit codes
- Added Blender binary path validation
- Better logging

- Preprocesses photos with OpenCV (white balance, CLAHE, denoise, unsharp, resize)
- Supports multiple photos per role (front/side/back)
- Calls deform_avatar.py which does projection + FaceMask + bake + (optional) pose-after-bake
- Returns base64-encoded GLB + last 4k lines of Blender log
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, subprocess, tempfile, base64, shutil, sys

# ---- OpenCV preprocessing ----
import cv2
import numpy as np

BLENDER = os.environ.get("BLENDER_BIN", "/blender/blender")
DEFORM_SCRIPT = "/app/deform_avatar.py"
ASSETS_DIR = "/app/assets"

# Validate Blender binary path at import time
if not os.path.exists(BLENDER):
    print(f"[WARNING] Blender binary not found at {BLENDER}", file=sys.stderr)
    print(f"[WARNING] Set BLENDER_BIN environment variable", file=sys.stderr)

BASES = {
    "male":    os.path.join(ASSETS_DIR, "base_male.blend"),
    "female":  os.path.join(ASSETS_DIR, "base_female.blend"),
    "neutral": os.path.join(ASSETS_DIR, "base_neutral.blend"),
    "child":   os.path.join(ASSETS_DIR, "base_child.blend"),
    "baby":    os.path.join(ASSETS_DIR, "base_baby.blend"),
}
REQUIRED_ASSETS = list(BASES.values())

# Configuration constants
MAX_GLB_SIZE_BYTES = 7_000_000  # 7 MB (base64 adds ~33% overhead)
BLENDER_TIMEOUT_SECONDS = 300   # 5 minutes


# ---------------- OpenCV helpers ----------------

def _b64_to_np(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    imgf = img.astype(np.float32) + 1e-6
    means = imgf.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    gain = gray / means
    imgf *= gain
    imgf = np.clip(imgf, 0, 255)
    return imgf.astype(np.uint8)

def _clahe_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab = cv2.merge([L, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

def _unsharp(img: np.ndarray, radius=2.5, amount=0.5) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img

def preprocess_and_save(b64_img: str, out_path: str, max_side: int) -> str:
    img = _b64_to_np(b64_img)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # IMPROVEMENT: Resize first to reduce memory footprint for large images
    img = _resize_max(img, max_side)
    
    img = _gray_world_white_balance(img)
    img = _clahe_contrast(img)
    img = _denoise(img)
    img = _unsharp(img, radius=2.5, amount=0.5)
    
    ok = cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Could not write preprocessed image to {out_path}")
    return out_path

def _prep_many(b64_list: List[str], tmpdir: str, tag: str, tex_res: int) -> List[str]:
    """Preprocess and save a list of b64 images; best-effort (skips failures)."""
    paths = []
    for j, b64img in enumerate(b64_list or []):
        if not b64img:
            continue
        out_png = os.path.join(tmpdir, f"{tag}_{j}.png")
        try:
            preprocess_and_save(b64img, out_png, max_side=int(tex_res))
            paths.append(out_png)
        except Exception as e:
            print(f"[WARN] Preprocess failed for {tag}[{j}]: {e}", file=sys.stderr)
    return paths


# ---------------- Blender runner ----------------

def _check_assets() -> Optional[str]:
    missing = [p for p in REQUIRED_ASSETS if not os.path.exists(p)]
    if missing:
        return "Missing base files: " + ", ".join(os.path.basename(m) for m in missing)
    if not os.path.exists(DEFORM_SCRIPT):
        return f"Missing deform script at {DEFORM_SCRIPT}"
    if not os.path.exists(BLENDER):
        return f"Blender binary not found at {BLENDER} (set BLENDER_BIN)"
    return None


def run_blender_avatar(
    *,
    preset: str,
    height_m: float,
    measurements: Dict[str, Optional[float]] | None,
    photos: Dict[str, Optional[str]],
    tex_res: int = 2048,
    photos_ranked: Optional[Dict[str, List[str]]] = None,
    high_detail: bool = False,
    pose_mode: str = "auto",
    pose_angles: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Execute Blender → deform_avatar.py and return {"ok": bool, "glb_b64"?, "log"?, "error"?}
    
    IMPROVEMENTS:
    - Timeout added (300s)
    - File size validation (< 7 MB)
    - Better error messages with exit codes
    - Enhanced logging
    
    Args:
        preset: Model preset (male/female/neutral/child/baby)
        height_m: Height in meters
        measurements: Optional body measurements dict
        photos: Dict of base64 image strings by role
        tex_res: Texture resolution (default 2048)
        photos_ranked: Ranked photos by role from calibration
        high_detail: Force high detail mode (>=4K)
        pose_mode: "auto" (apply pose after bake), "neutral" (no pose)
        pose_angles: Dict with pose angles (deg) if pose_mode=="auto"
    
    Returns:
        Dict with ok, glb_b64 (if success), error, log
    """
    print(f"[BLENDER] Starting avatar generation: preset={preset}, height={height_m}m, texRes={tex_res}")
    
    err = _check_assets()
    if err:
        return {"ok": False, "error": err}

    blend_path = BASES.get(preset.lower(), BASES["neutral"])
    if not os.path.exists(blend_path):
        return {"ok": False, "error": f"Base file for preset '{preset}' not found: {blend_path}"}

    tmpdir = tempfile.mkdtemp(prefix="avatar_")
    try:
        out_glb = os.path.join(tmpdir, "twin.glb")

        cmd = [
            BLENDER, "-b", blend_path, "--python", DEFORM_SCRIPT, "--",
            "--preset", preset, "--height", str(height_m),
            "--out", out_glb, "--texRes", str(int(tex_res))
        ]
        if high_detail:
            cmd += ["--highDetail"]

        # --- pose JSON when requested ---
        pose_json_path = None
        if (pose_mode or "auto").lower() == "auto" and pose_angles:
            import json
            pose_json_path = os.path.join(tmpdir, "pose.json")
            with open(pose_json_path, "w") as f:
                json.dump(pose_angles, f)
            cmd += ["--poseJson", pose_json_path]
            print(f"[BLENDER] Pose mode enabled with angles: {pose_angles}")

        # measurements
        for k in ("chest", "waist", "hips", "shoulder", "inseam", "arm"):
            v = (measurements or {}).get(k)
            if v is not None:
                cmd += [f"--{k}", str(float(v))]

        # photos
        def pick_list(role: str) -> List[str]:
            if photos_ranked and (lst := photos_ranked.get(role)):
                return lst[:2]
            b64 = (photos or {}).get(role)
            return [b64] if b64 else []

        front_paths = _prep_many(pick_list("front"), tmpdir, "front", tex_res)
        side_paths  = _prep_many(pick_list("side"),  tmpdir, "side",  tex_res)
        back_paths  = _prep_many(pick_list("back"),  tmpdir, "back",  tex_res)

        def join_or_empty(lst): return ";".join(lst) if lst else ""
        if front_paths: cmd += ["--frontTexList", join_or_empty(front_paths)]
        if side_paths:  cmd += ["--sideTexList",  join_or_empty(side_paths)]
        if back_paths:  cmd += ["--backTexList",  join_or_empty(back_paths)]

        # back-compat single flags
        if not front_paths and (photos or {}).get("front"):
            f1 = _prep_many([photos["front"]], tmpdir, "front_single", tex_res)
            if f1: cmd += ["--frontTex", f1[0]]
        if not side_paths and (photos or {}).get("side"):
            s1 = _prep_many([photos["side"]], tmpdir, "side_single", tex_res)
            if s1: cmd += ["--sideTex", s1[0]]
        if not back_paths and (photos or {}).get("back"):
            b1 = _prep_many([photos["back"]], tmpdir, "back_single", tex_res)
            if b1: cmd += ["--backTex", b1[0]]

        print(f"[BLENDER] Executing command: {' '.join(cmd[:5])}... (truncated)")
        print(f"[BLENDER] Timeout set to {BLENDER_TIMEOUT_SECONDS}s")
        
        # IMPROVEMENT: Add timeout to prevent hanging
        try:
            proc = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                timeout=BLENDER_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as e:
            print(f"[BLENDER] Process timed out after {BLENDER_TIMEOUT_SECONDS}s", file=sys.stderr)
            return {
                "ok": False, 
                "error": f"Blender rendering timed out after {BLENDER_TIMEOUT_SECONDS} seconds",
                "log": "Process killed due to timeout. Try reducing texRes or simplifying the model."
            }
        
        log_tail = proc.stdout[-4000:] if proc.stdout else ""
        
        # IMPROVEMENT: Better error messages with exit codes
        if proc.returncode != 0:
            print(f"[BLENDER] Process failed with exit code {proc.returncode}", file=sys.stderr)
            return {
                "ok": False, 
                "error": f"Blender process failed with exit code {proc.returncode}. Check log for details.",
                "log": log_tail
            }
        
        if not os.path.exists(out_glb):
            print(f"[BLENDER] Output file not found at {out_glb}", file=sys.stderr)
            return {
                "ok": False, 
                "error": f"Blender completed successfully but output file not found at {out_glb}",
                "log": log_tail
            }
        
        # IMPROVEMENT: Check file size before encoding
        glb_size = os.path.getsize(out_glb)
        print(f"[BLENDER] Generated GLB size: {glb_size / 1e6:.2f} MB")
        
        if glb_size > MAX_GLB_SIZE_BYTES:
            print(f"[BLENDER] Output file too large: {glb_size} bytes", file=sys.stderr)
            return {
                "ok": False,
                "error": f"Output file too large ({glb_size / 1e6:.1f} MB, max {MAX_GLB_SIZE_BYTES / 1e6:.1f} MB). Try lower texRes (current: {tex_res}).",
                "log": log_tail
            }
        
        with open(out_glb, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        print(f"[BLENDER] Success! Base64 length: {len(glb_b64)} chars")
        return {"ok": True, "glb_b64": glb_b64, "log": log_tail}
        
    except Exception as e:
        print(f"[BLENDER] Unexpected error: {e}", file=sys.stderr)
        import traceback
        return {
            "ok": False, 
            "error": f"Unexpected error: {str(e)}", 
            "log": traceback.format_exc()
        }
    finally:
        # Always cleanup temp files
        shutil.rmtree(tmpdir, ignore_errors=True)

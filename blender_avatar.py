#!/usr/bin/env python3
"""
blender_avatar.py - Wrapper for Blender avatar generation

- Handles base64 photos & data: URIs (chooses proper file extension)
- Supports single photos AND ranked photo lists (front/side/back)
- Silences ALSA with SDL_AUDIODRIVER=dummy for headless runs
- Exports GLB via deform_avatar.py and returns base64 + logs
"""

import os
import sys
import base64
import subprocess
import tempfile
import json
import shutil
import time
from pathlib import Path
from typing import Optional, Iterable

# Limits to keep serverless jobs predictable
MAX_TOTAL_PHOTOS = 10  # across all roles


# ----------------------------- helpers -----------------------------

def _guess_ext_from_header(header: str) -> str:
    """Pick extension from data: URI mime type."""
    if not header.startswith("data:"):
        return ".jpg"
    mime = header.split(";")[0].split(":", 1)[-1].lower()
    if "png" in mime:
        return ".png"
    if "webp" in mime:
        return ".webp"
    if "jpeg" in mime or "jpg" in mime:
        return ".jpg"
    return ".jpg"


def _sniff_ext_from_bytes(b: bytes) -> str:
    """Best-effort file type sniff from magic bytes."""
    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(b) >= 3 and b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(b) >= 12 and b[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _write_b64_to_file(b64_string: str, output_dir: str, prefix: str, ext_hint: Optional[str] = None) -> Optional[str]:
    """
    Write base64 image data (plain or data: URI) to a file with a sensible extension.
    Returns the path or None on failure.
    """
    if not b64_string:
        return None

    try:
        header = ""
        payload = b64_string
        if b64_string.startswith("data:"):
            # data:image/png;base64,AAAA...
            header, payload = b64_string.split(",", 1)

        img_bytes = base64.b64decode(payload)

        # Choose extension: header hint > bytes sniff > ext_hint > jpg
        ext = _guess_ext_from_header(header) if header else None
        if not ext:
            ext = _sniff_ext_from_bytes(img_bytes)
        if not ext and ext_hint:
            ext = ext_hint
        if not ext:
            ext = ".jpg"

        filepath = os.path.join(output_dir, f"{prefix}{ext}")
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        return filepath
    except Exception as e:
        print(f"[BLENDER] Warning: Failed to write {prefix} photo: {e}", file=sys.stderr)
        return None


def _iter_nonempty(v) -> Iterable[str]:
    """Normalize a value into an iterable of non-empty strings."""
    if v is None:
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str) and x.strip()]
    if isinstance(v, str) and v.strip():
        return [v]
    return []


def _unique_preserve_order(seq: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# ----------------------------- main entry -----------------------------

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

    Returns:
        dict with keys:
          ok: bool
          glb_b64: str (on success)
          error: str (on failure)
          log: str (blender stdout/stderr)
          file_size: int (on success)
          returncode: int
    """
    # Create unique temp directory for this avatar
    output_dir = tempfile.mkdtemp(prefix="avatar_")
    output_file = os.path.join(output_dir, "twin.glb")
    print(f"[BLENDER] Temp dir: {output_dir}")
    print(f"[BLENDER] Output file: {output_file}")

    # Resolve Blender binary and base .blend
    blender_bin = os.environ.get("BLENDER_BIN", "/blender/blender")
    if not os.path.exists(blender_bin):
        return {"ok": False, "error": f"Blender binary not found at {blender_bin}", "log": ""}

    base_blend = f"/app/assets/base_{preset}.blend"
    if not os.path.exists(base_blend):
        print(f"[BLENDER] Warning: {base_blend} not found; fallback to base_female.blend")
        base_blend = "/app/assets/base_female.blend"
    if not os.path.exists(base_blend):
        return {"ok": False, "error": f"Base blend not found: {base_blend}", "log": ""}

    # Convert photos (single + ranked) to files
    print("[BLENDER] Converting photos...")
    photo_paths = {"front": [], "side": [], "back": []}

    # Singles (base64 strings)
    if photos:
        for role in ("front", "side", "back"):
            for item in _iter_nonempty(photos.get(role)):
                path = _write_b64_to_file(item, output_dir, f"{role}")
                if path:
                    photo_paths[role].append(path)
                    break  # only need one from singles per role if present

    # Ranked lists (base64 strings) — keep up to 2 per role
    if photos_ranked:
        for role in ("front", "side", "back"):
            rank_list = list(_iter_nonempty(photos_ranked.get(role)))
            role_files = []
            for idx, b64_photo in enumerate(rank_list[:2]):
                fp = _write_b64_to_file(b64_photo, output_dir, f"{role}_{idx}")
                if fp:
                    role_files.append(fp)
            if role_files:
                # Prepend ranked results (higher priority)
                photo_paths[role] = _unique_preserve_order(role_files + photo_paths[role])

    # Cap total number of photos for safety
    total = sum(len(v) for v in photo_paths.values())
    if total > MAX_TOTAL_PHOTOS:
        print(f"[BLENDER] Trimming photos from {total} to {MAX_TOTAL_PHOTOS}")
        # Flatten in role order, trim, then rebuild
        flat = []
        for r in ("front", "side", "back"):
            for p in photo_paths[r]:
                flat.append((r, p))
        flat = flat[:MAX_TOTAL_PHOTOS]
        # Rebuild
        photo_paths = {"front": [], "side": [], "back": []}
        for r, p in flat:
            photo_paths[r].append(p)

    print("[BLENDER] Photo counts:",
          {k: len(v) for k, v in photo_paths.items()})

    # Build Blender command
    cmd = [
        blender_bin,
        "-b", base_blend,
        "--python", "/app/deform_avatar.py",
        "--",
        "--preset", preset,
        "--height", str(height_m),
        "--texRes", str(tex_res),
        "--out", output_file,
    ]

    if high_detail:
        cmd.append("--highDetail")

    # Measurements (only if provided)
    for key in ["chest", "waist", "hips", "shoulder", "inseam", "arm"]:
        val = (measurements or {}).get(key)
        if val is not None:
            cmd.extend([f"--{key}", str(val)])

    # Add multi-photo lists first (semicolon separated), then single fallbacks
    for role in ("front", "side", "back"):
        paths = photo_paths[role]
        if len(paths) >= 2:
            cmd.extend([f"--{role}TexList", ";".join(paths)])
        elif len(paths) == 1:
            cmd.extend([f"--{role}Tex", paths[0]])

    # Pose (apply AFTER bake in deform_avatar.py)
    pose_json_path = ""
    if pose_mode == "auto" and pose_angles:
        pose_json_path = os.path.join(output_dir, "pose.json")
        try:
            with open(pose_json_path, "w", encoding="utf-8") as f:
                json.dump(pose_angles, f)
            cmd.extend(["--poseJson", pose_json_path])
            print(f"[BLENDER] Pose: auto with {len(pose_angles)} angles")
        except Exception as e:
            print(f"[BLENDER] Warning: failed to write pose JSON: {e}", file=sys.stderr)

    # Environment
    env = os.environ.copy()
    env["OUTPUT_GLTF"] = output_file
    env["SDL_AUDIODRIVER"] = "dummy"  # silence ALSA in containers

    # Log command (partially)
    safe_preview = " ".join(cmd[:14]) + (" ..." if len(cmd) > 14 else "")
    print(f"[BLENDER] Executing: {safe_preview}")

    # Execute Blender
    timeout = int(os.environ.get("BLENDER_TIMEOUT", "300"))  # seconds
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        log = ""
        if result.stdout:
            log += "=== STDOUT ===\n" + result.stdout + "\n"
        if result.stderr:
            log += "=== STDERR ===\n" + result.stderr + "\n"

        print(f"[BLENDER] Return code: {result.returncode}")

        # Small FS settle helps avoid rare container races
        time.sleep(0.2)

        # Validate output file
        if not os.path.exists(output_file):
            print(f"[BLENDER] ERROR: GLB not found at {output_file}", file=sys.stderr)
            try:
                dir_contents = os.listdir(output_dir)
            except Exception:
                dir_contents = []
            err = {
                "ok": False,
                "error": f"Blender finished (code {result.returncode}) but GLB missing",
                "log": log,
                "returncode": result.returncode,
                "output_dir": output_dir,
                "dir_contents": dir_contents,
            }
            # keep temp dir for debugging
            return err

        size = os.path.getsize(output_file)
        if size == 0:
            return {"ok": False, "error": "GLB created but empty", "log": log, "returncode": result.returncode}

        # Check GLB header
        with open(output_file, "rb") as f:
            header = f.read(4)
        if header != b"glTF":
            print(f"[BLENDER] Warning: unexpected file header: {header}", file=sys.stderr)

        # Read + encode
        with open(output_file, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Cleanup on success
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass

        return {
            "ok": True,
            "glb_b64": glb_b64,
            "log": log,
            "file_size": size,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired as e:
        error_msg = f"Blender execution timed out after {timeout}s"
        print(f"[BLENDER] ERROR: {error_msg}", file=sys.stderr)
        partial_log = ""
        if getattr(e, "stdout", None):
            partial_log += "=== PARTIAL STDOUT ===\n" + e.stdout + "\n"
        if getattr(e, "stderr", None):
            partial_log += "=== PARTIAL STDERR ===\n" + e.stderr + "\n"
        # try to clean
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass
        return {"ok": False, "error": error_msg, "log": partial_log}

    except Exception as e:
        err = f"Unexpected error during Blender execution: {e}"
        print(f"[BLENDER] ERROR: {err}", file=sys.stderr)
        # try to clean
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass
        return {"ok": False, "error": err, "log": str(e)}

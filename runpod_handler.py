#!/usr/bin/env python3
"""
RunPod Serverless entrypoint (CPU) with strict photo QA gate.

IMPROVEMENTS IN THIS VERSION:
- Added input validation for texRes (512-8192 range)
- Added photo count validation (max 10 photos)
- Added height validation with better error messages
- Better logging throughout
- Graceful handling of missing photos

Flow:
1) Calibrate input (height required, photo QA, language, role report)
2) REQUIRE photos to be OK before rendering (front + side by default)
3) If OK, call Blender to deform + project + bake (neutral/A-pose), then pose after bake if poseMode="auto"
4) Return GLB (base64) + diagnostics

Override per request:
- "required_roles": ["front","side","back"]
- "allowPartial": true
- "poseMode": "auto" | "neutral"
- "highDetail": true (forces ≥4K bake)
"""

import traceback
import sys

from calibration import calibrate_input
from blender_avatar import run_blender_avatar

# RunPod bootstrap
try:
    import runpod
except Exception:
    class _Dummy:
        class serverless:
            @staticmethod
            def start(cfg):
                print("RunPod SDK not found. Local mode; not starting serverless loop.")
    runpod = _Dummy()

DEFAULT_REQUIRED_ROLES = ("front", "side")   # back optional by default

# Configuration constants
MIN_TEX_RES = 512
MAX_TEX_RES = 8192
DEFAULT_TEX_RES = 2048
MAX_PHOTOS = 10


def _build_photos_from_calibration(calib: dict, raw_input: dict) -> dict:
    """Prefer calibration-chosen photos; fall back to raw explicit ones."""
    chosen = calib.get("chosen_by_role") or {}
    photos = (raw_input.get("photos") or {})
    return {
        "front": chosen.get("front") or photos.get("front"),
        "side":  chosen.get("side")  or photos.get("side"),
        "back":  chosen.get("back")  or photos.get("back"),
    }


def _roles_not_ok(role_report: dict, required_roles) -> list:
    """Return list of roles that are NOT ok (missing or retry) among the required ones."""
    missing = []
    for r in required_roles:
        status = (role_report.get(r) or {}).get("status")
        if status != "ok":
            missing.append(r)
    return missing


def _validate_tex_res(tex_res_input, lang: str) -> tuple[int, dict | None]:
    """
    Validate texture resolution input.
    Returns: (validated_value, error_dict_or_None)
    """
    try:
        tex_res = int(tex_res_input or DEFAULT_TEX_RES)
    except (ValueError, TypeError):
        return DEFAULT_TEX_RES, {
            "ok": False,
            "lang": lang,
            "error": f"Invalid texRes value: {tex_res_input}. Must be an integer."
        }
    
    if not (MIN_TEX_RES <= tex_res <= MAX_TEX_RES):
        return DEFAULT_TEX_RES, {
            "ok": False,
            "lang": lang,
            "error": f"texRes must be between {MIN_TEX_RES} and {MAX_TEX_RES}, got {tex_res}. Using default {DEFAULT_TEX_RES}."
        }
    
    return tex_res, None


def _validate_photos(photos_input: dict, lang: str) -> dict | None:
    """
    Validate photos input structure and count.
    Returns: error_dict_or_None
    """
    if not photos_input:
        return None  # Empty photos is okay, will be caught in calibration
    
    if not isinstance(photos_input, dict):
        return {
            "ok": False,
            "lang": lang,
            "error": "photos must be a dictionary with 'front', 'side', 'back' keys or 'unordered' list"
        }
    
    # Count total photos
    all_photos = []
    if isinstance(photos_input.get("unordered"), list):
        all_photos = [p for p in photos_input["unordered"] if p]
    else:
        for key in ("front", "side", "back"):
            val = photos_input.get(key)
            if isinstance(val, str) and val:
                all_photos.append(val)
            elif isinstance(val, list):
                all_photos.extend([p for p in val if p])
    
    if len(all_photos) > MAX_PHOTOS:
        return {
            "ok": False,
            "lang": lang,
            "error": f"Too many photos provided ({len(all_photos)}). Maximum {MAX_PHOTOS} photos allowed."
        }
    
    return None


def handler(event):
    """
    Main RunPod handler function.
    
    IMPROVEMENTS:
    - Input validation before processing
    - Better error messages
    - Enhanced logging
    """
    try:
        data = event.get("input", event)
        print(f"[HANDLER] Received request with keys: {list(data.keys())}")
        
        # IMPROVEMENT: Early language detection for error messages
        from calibration import resolve_lang
        lang = resolve_lang(data)
        
        # IMPROVEMENT: Validate texture resolution early
        tex_res_input = data.get("texRes")
        tex_res, error = _validate_tex_res(tex_res_input, lang)
        if error:
            print(f"[HANDLER] texRes validation failed: {error['error']}", file=sys.stderr)
            return error
        
        # IMPROVEMENT: Validate photos structure and count
        photos_input = data.get("photos")
        error = _validate_photos(photos_input, lang)
        if error:
            print(f"[HANDLER] Photo validation failed: {error['error']}", file=sys.stderr)
            return error
        
        print(f"[HANDLER] Validated inputs: texRes={tex_res}, lang={lang}")

        # 1) Calibrate (validates height, analyzes + QA photos, resolves language)
        print("[HANDLER] Starting calibration...")
        calib = calibrate_input(data)
        
        if not calib.get("ok"):
            print(f"[HANDLER] Calibration failed: {calib.get('error')}", file=sys.stderr)
            return {k: calib.get(k) for k in ("ok", "error", "lang")}

        lang = calib.get("lang", "en")
        role_report = calib.get("role_report", {})
        retake_tips = calib.get("retake_tips", [])
        
        print(f"[HANDLER] Calibration complete: lang={lang}, role_report={list(role_report.keys())}")

        # 2) Photo QA gate FIRST
        required_roles = tuple(data.get("required_roles") or DEFAULT_REQUIRED_ROLES)
        allow_partial = bool(data.get("allowPartial") or False)
        not_ok = _roles_not_ok(role_report, required_roles)
        
        print(f"[HANDLER] Photo QA: required_roles={required_roles}, allow_partial={allow_partial}, not_ok={not_ok}")
        
        if not allow_partial and not_ok:
            print(f"[HANDLER] Photo QA failed: {not_ok} roles not okay", file=sys.stderr)
            return {
                "ok": False,
                "lang": lang,
                "error": "Photos didn't pass quality. Please retake the required views.",
                "required_roles": list(required_roles),
                "roles_not_ok": not_ok,
                "role_report": role_report,
                "retake_tips": retake_tips,
                "thresholds": calib.get("thresholds"),
            }

        # 3) Proceed to render
        height_m = calib["height_m"]
        measurements = {
            "chest": calib.get("chest"),
            "waist": calib.get("waist"),
            "hips": calib.get("hips"),
            "shoulder": calib.get("shoulder"),
            "inseam": calib.get("inseam"),
            "arm": calib.get("arm"),
        }
        photos = _build_photos_from_calibration(calib, data)
        preset = (data.get("preset") or data.get("gender_hint") or "neutral").strip().lower()

        # Posing mode & angles (we bake in neutral, then optionally pose after bake)
        pose_mode = (data.get("poseMode") or "auto").strip().lower()  # "auto" | "neutral"
        pose_angles = calib.get("pose_hint") if pose_mode == "auto" else None
        high_detail = bool(data.get("highDetail"))
        
        print(f"[HANDLER] Starting Blender render: preset={preset}, height={height_m}m, texRes={tex_res}, pose={pose_mode}, highDetail={high_detail}")

        result = run_blender_avatar(
            preset=preset,
            height_m=height_m,
            measurements=measurements,
            photos=photos,
            tex_res=tex_res,
            photos_ranked=calib.get("by_role_ranked"),
            high_detail=high_detail,
            pose_mode=pose_mode,
            pose_angles=pose_angles
        )

        payload = {
            "ok": bool(result.get("ok")),
            "lang": lang,
            "preset": preset,
            "height_m": height_m,
            "texRes": tex_res,
            "chosen_by_role": calib.get("chosen_by_role"),
            "by_role_ranked": calib.get("by_role_ranked"),
            "role_report": role_report,
            "retake_tips": retake_tips,
            "thresholds": calib.get("thresholds"),
            "accepted": calib.get("accepted"),
            "rejected": calib.get("rejected"),
            "required_roles": list(required_roles),
            "allowPartial": allow_partial,
            "poseMode": pose_mode
        }

        if result.get("ok"):
            glb_size = len(result["glb_b64"]) if result.get("glb_b64") else 0
            print(f"[HANDLER] Success! GLB base64 size: {glb_size} chars")
            payload["glb_b64"] = result["glb_b64"]
            payload["log"] = result.get("log", "")
        else:
            print(f"[HANDLER] Blender failed: {result.get('error')}", file=sys.stderr)
            payload["error"] = result.get("error", "Blender failed")
            payload["log"] = result.get("log", "")

        return payload

    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"[HANDLER] Unhandled exception: {error_msg}", file=sys.stderr)
        print(trace, file=sys.stderr)
        return {
            "ok": False, 
            "error": error_msg, 
            "trace": trace
        }


if __name__ == "__main__":
    print("[HANDLER] Starting RunPod serverless handler...")
    print(f"[HANDLER] Configuration: texRes range=[{MIN_TEX_RES}, {MAX_TEX_RES}], max_photos={MAX_PHOTOS}")
    runpod.serverless.start({"handler": handler})

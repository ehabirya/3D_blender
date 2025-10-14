#!/usr/bin/env python3
"""
RunPod Serverless entrypoint.

Pipeline:
1) Calibrate/validate inputs (height required).
2) If photos are unordered (list of base64 strings), auto-pick front/side/back + preset.
3) Call Blender runner to deform, project photos, bake, and export GLB.

Expected input JSON (examples):
{
  "height": 175,                      # cm or m; REQUIRED
  "chest": 92, "waist": 74, ... ,     # optional numeric measurements
  "preset": "female",                  # optional, will be auto-guessed if missing
  "gender_hint": "female",             # optional
  "age_hint": "adult|child|baby",      # optional
  "photos": {
    "unordered": ["<b64>", "<b64>", "<b64>"]   # preferred
    # OR explicit:
    # "front": "<b64>", "side": "<b64>", "back": "<b64>"
  },
  "texRes": 2048                       # optional bake resolution
}

Returns:
{ "ok": true, "glb_b64": "<...>", "log": "<last N lines>" }
or
{ "ok": false, "error": "...", "log": "...?" }
"""

import traceback

# 1) local modules
from calibration import calibrate_input
from photo_selector import analyze_photos
from blender_avatar import run_blender_avatar

# 2) RunPod bootstrap (works locally too)
try:
    import runpod  # provided in RunPod serverless
except Exception:  # allow local tests without the SDK
    class _Dummy:
        class serverless:
            @staticmethod
            def start(cfg):
                print("RunPod SDK not found. Local mode; not starting serverless loop.")
    runpod = _Dummy()


def _extract_photos_and_preset(data: dict):
    """
    Returns (photos_dict, preset_or_none).
    photos_dict has keys: front, side, back (values are base64 strings or None).
    """
    photos = data.get("photos") or {}
    preset_hint = (data.get("preset") or "").strip().lower() or None

    # If unordered photos provided, analyze and classify.
    unordered = photos.get("unordered")
    if isinstance(unordered, list) and len(unordered) >= 1:
        # use at most 4 to keep CPU reasonable
        unordered = unordered[:4]
        sel = analyze_photos(
            unordered,
            hints={
                "gender_hint": (data.get("gender_hint") or "").lower() or None,
                "age_hint": (data.get("age_hint") or "").lower() or None
            }
        )
        auto_photos = sel.get("by_role", {})
        auto_preset = sel.get("preset")
        photos_out = {
            "front": auto_photos.get("front") or photos.get("front"),
            "side":  auto_photos.get("side")  or photos.get("side"),
            "back":  auto_photos.get("back")  or photos.get("back"),
        }
        return photos_out, (preset_hint or auto_preset or None)

    # Else: use explicit roles (may be None)
    return {
        "front": photos.get("front"),
        "side":  photos.get("side"),
        "back":  photos.get("back")
    }, preset_hint


def handler(event):
    """
    RunPod handler — do NOT rename; Serverless looks for this.
    """
    try:
        data = event.get("input", event)

        # 1) Calibration / validation
        calib = calibrate_input(data)
        if not calib.get("ok"):
            # Pass through the reason
            return calib
        height_m = calib["height_m"]

        # Collect clean measurements if present (already numeric in calib's return)
        measurements = {
            k: calib.get(k) for k in ("chest", "waist", "hips", "shoulder", "inseam", "arm")
        }

        # 2) Photos + preset resolution
        photos, maybe_preset = _extract_photos_and_preset(data)
        preset = (maybe_preset or "neutral").lower()

        # 3) Render settings
        tex_res = int(data.get("texRes") or 2048)

        # 4) Call Blender headless (preprocess in blender_avatar, then deform/bake in deform_avatar)
        result = run_blender_avatar(
            preset=preset,
            height_m=height_m,
            measurements=measurements,
            photos=photos,
            tex_res=tex_res
        )
        return result

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


# Serverless entry
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

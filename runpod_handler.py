#!/usr/bin/env python3
"""
RunPod Serverless entrypoint (CPU) with strict photo QA gate.

Flow:
1) Calibrate input (height required, photo QA, language, role report)
2) REQUIRE photos to be OK before rendering (front + side by default)
3) If OK, call Blender to deform + project + bake + export GLB
4) Return GLB (base64) + diagnostics

You can override gating via input:
- "required_roles": ["front","side","back"]   # customize per-call
- "allowPartial": true                        # proceed even if some roles fail (default: false)
"""

import traceback

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

# Defaults for gating
DEFAULT_REQUIRED_ROLES = ("front", "side")   # back is optional by default


def _build_photos_from_calibration(calib: dict, raw_input: dict) -> dict:
    """Prefer calibration-chosen photos; fall back to raw explicit ones."""
    chosen = calib.get("chosen_by_role") or {}
    photos = raw_input.get("photos") or {}
    return {
        "front": chosen.get("front") or photos.get("front"),
        "side":  chosen.get("side")  or photos.get("side"),
        "back":  chosen.get("back")  or photos.get("back"),
    }


def _roles_not_ok(role_report: dict, required_roles) -> list:
    """
    Return list of roles that are NOT ok (missing or retry) among the required ones.
    """
    missing = []
    for r in required_roles:
        status = (role_report.get(r) or {}).get("status")
        if status != "ok":
            missing.append(r)
    return missing


def handler(event):
    try:
        data = event.get("input", event)

        # 1) Calibrate (validates height, analyzes + QA photos, resolves language)
        calib = calibrate_input(data)
        if not calib.get("ok"):
            return {k: calib.get(k) for k in ("ok", "error", "lang")}

        lang = calib.get("lang", "en")
        role_report = calib.get("role_report", {})
        retake_tips = calib.get("retake_tips", [])

        # 2) Photo QA gate (FIRST check if fotos are ok)
        required_roles = tuple(data.get("required_roles") or DEFAULT_REQUIRED_ROLES)
        allow_partial = bool(data.get("allowPartial") or False)

        not_ok = _roles_not_ok(role_report, required_roles)
        if not allow_partial and not_ok:
            # Block rendering; return actionable info
            return {
                "ok": False,
                "lang": lang,
                "error": "Photos didn’t pass quality. Please retake the required views.",
                "required_roles": list(required_roles),
                "roles_not_ok": not_ok,
                "role_report": role_report,
                "retake_tips": retake_tips,
                "thresholds": calib.get("thresholds"),
            }

        # 3) Proceed to render (even if partial if allowed)
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
        tex_res = int(data.get("texRes") or 2048)

        result = run_blender_avatar(
            preset=preset,
            height_m=height_m,
            measurements=measurements,
            photos=photos,
            tex_res=tex_res
        )

        payload = {
            "ok": bool(result.get("ok")),
            "lang": lang,
            "preset": preset,
            "height_m": height_m,
            "chosen_by_role": calib.get("chosen_by_role"),
            "role_report": role_report,
            "retake_tips": retake_tips,
            "thresholds": calib.get("thresholds"),
            "accepted": calib.get("accepted"),
            "rejected": calib.get("rejected"),
            "required_roles": list(required_roles),
            "allowPartial": allow_partial
        }

        if result.get("ok"):
            payload["glb_b64"] = result["glb_b64"]
            payload["log"] = result.get("log", "")
        else:
            payload["error"] = result.get("error", "Blender failed")
            payload["log"] = result.get("log", "")

        return payload

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

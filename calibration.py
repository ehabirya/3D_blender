# app/calibration.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import math

import numpy as np
import cv2

from vision import (
    b64_to_img, analyze_one, quality_ok, choose_roles, DEFAULT_THRESHOLDS
)

# ------------- numeric helpers -------------

def _to_float(x) -> Optional[float]:
    if x is None or x == "": return None
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

def _cm_to_m(v: Optional[float]) -> Optional[float]:
    if v is None: return None
    return v if v <= 3.0 else v / 100.0


# ------------- TIPS: reason -> user-friendly guidance -------------

def reason_to_tip(reason: str, analysis: Dict[str, Any]) -> str:
    """
    Turn a rejection reason into a plain-English tip.
    Uses analysis (e.g., camera distance) when available.
    """
    r = reason.lower()
    cam = analysis.get("camera", {}) if analysis else {}
    dist = cam.get("distance_m")

    if "low focus" in r:
        return "Hold the phone steady and tap to focus on the body. Good light helps reduce blur."
    if "shoulders not fully visible" in r or "subject too far" in r:
        return "Step back slightly and make sure both shoulders are fully in frame."
    if "camera tilt" in r:
        return "Hold the phone level (no tilt). Keep horizon vertical and frame upright."
    if "not facing camera enough for front view" in r:
        return "For the front photo, face the camera directly with shoulders level."
    if "not turned enough for side view" in r:
        return "For the side photo, turn 90° to the camera and keep posture straight."
    # distance-based nudges
    if dist is not None:
        # recommend a comfortable range ~1.5–2.5 m for full body
        if dist < 1.2:
            return "You’re a bit too close—step back about 0.5–1 m for a full-body shot."
        if dist > 3.0:
            return "You’re a bit far—step closer about 0.5–1 m so your body fills the frame."

    # fallback generic tip
    return "Stand straight in good light; keep the whole body in frame with the phone held level."

def tips_for_rejected(entry: Dict[str, Any]) -> List[str]:
    """
    Build a concise set of tips for a rejected photo entry.
    """
    tips = []
    reasons = entry.get("reasons", [])
    analysis = entry.get("analysis", {})
    for r in reasons:
        tip = reason_to_tip(r, analysis)
        if tip not in tips:
            tips.append(tip)
    # If no specific reasons produced tips, give a generic one
    if not tips:
        tips.append("Retake the photo with better lighting, steady hands, and the whole body visible.")
    return tips


# ------------- main API -------------

def calibrate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shared calibration:
    - height (required) normalized to meters
    - optional measurements normalized
    - analyze all provided photos with shared vision logic
    - quality gate: mark photos as accepted/rejected with reasons
    - choose best front/side/back (even if none accepted)
    - attach user-friendly retake tips to rejections and a deduped global list
    """
    # ---- 1) height required ----
    height = _to_float(data.get("height"))
    if height is None:
        return {"ok": False, "error": "height is required (cm or m)"}
    height_m = _cm_to_m(height)
    if height_m < 0.3 or height_m > 2.6:
        return {"ok": False, "error": f"unrealistic height: {height_m:.2f} m"}

    # ---- 2) optional measurements ----
    def cm(key): return _cm_to_m(_to_float(data.get(key)))
    chest    = cm("chest")
    waist    = cm("waist")
    hips     = cm("hips")
    shoulder = cm("shoulder")
    inseam   = cm("inseam")
    arm      = cm("arm")
    foot_len = cm("foot_length") or cm("footLength")
    foot_wid = cm("foot_width")  or cm("footWidth")
    foot_cat = (data.get("foot_width_category") or data.get("footWidthCategory") or "").strip() or None

    out: Dict[str, Any] = {
        "ok": True,
        "height_m": height_m,
        "chest": chest, "waist": waist, "hips": hips,
        "shoulder": shoulder, "inseam": inseam, "arm": arm,
        "foot_length_m": foot_len, "foot_width_m": foot_wid,
        "foot_width_category": foot_cat,
    }

    # ---- 3) gather photos ----
    photos = (data.get("photos") or {})
    unordered = photos.get("unordered") if isinstance(photos, dict) else None

    b64_list: List[str] = []
    if isinstance(unordered, list) and unordered:
        b64_list.extend([b for b in unordered if isinstance(b, str)])
    else:
        for k in ("front", "side", "back"):
            v = photos.get(k) if isinstance(photos, dict) else None
            if isinstance(v, str):
                b64_list.append(v)

    if not b64_list:
        out.update({"accepted": [], "rejected": [], "chosen_by_role": {"front": None, "side": None, "back": None},
                    "thresholds": DEFAULT_THRESHOLDS, "retake_tips": []})
        return out

    # ---- 4) analyze all photos ----
    analyses: List[Dict[str, Any]] = []
    for b64 in b64_list:
        img = b64_to_img(b64)
        if img is None:
            analyses.append({"decode_error": True, "role": "unknown"})
            continue
        analyses.append(analyze_one(img, height_m))

    # ---- 5) quality gate per photo ----
    accepted, rejected = [], []
    for idx, (b64, a) in enumerate(zip(b64_list, analyses)):
        role_hint = None
        ok, reasons = quality_ok(a, role_hint)
        entry = {"index": idx, "role_pred": a.get("role"), "reasons": reasons, "analysis": a}
        if ok:
            accepted.append(entry)
        else:
            entry["tips"] = tips_for_rejected(entry)  # << add tips here
            rejected.append(entry)

    # ---- 6) choose best front/side/back ----
    if accepted:
        chosen = choose_roles([a["analysis"] for a in accepted], [b64_list[e["index"]] for e in accepted])["by_role"]
    else:
        chosen = choose_roles(analyses, b64_list)["by_role"]

    # ---- 7) dedup global retake tips (ordered) ----
    retake_tips: List[str] = []
    for r in rejected:
        for tip in r.get("tips", []):
            if tip not in retake_tips:
                retake_tips.append(tip)

    out.update({
        "accepted": accepted,
        "rejected": rejected,
        "chosen_by_role": chosen,
        "thresholds": DEFAULT_THRESHOLDS,
        "retake_tips": retake_tips
    })
    return out

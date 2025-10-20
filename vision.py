#!/usr/bin/env python3
"""
vision.py - ENHANCED & COMPATIBLE VERSION

IMPROVEMENTS:
- ✅ Keeps all original functions (choose_roles, quality_ok with role_hint)
- ✅ Adds scale calibration (estimate_global_scale)
- ✅ Adds foot measurements (foot_length, foot_width)
- ✅ Maintains MediaPipe integration
- ✅ Compatible with existing calibration.py
- ✅ Backward compatible return structure
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import base64, math
import numpy as np
import cv2

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# ====================== NEW: Scale Calibration ======================

DEFAULT_SCALE_WEIGHTS: Dict[str, float] = {
    "height": 1.2,
    "chest": 1.0,
    "waist": 1.0,
    "hips": 1.0,
    "shoulder": 0.8,
    "inseam": 0.6,
    "arm": 0.6,
    "foot_length": 0.7,
    "foot_width": 0.5,
}

def _weighted_median(values: List[float], weights: List[float]) -> float:
    """Calculate weighted median for robust scale estimation."""
    if not values:
        return 1.0
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    denom = float(w.sum()) if w.sum() > 0 else 1.0
    cdf = np.cumsum(w) / denom
    idx = int(np.searchsorted(cdf, 0.5))
    if idx >= len(v):
        idx = len(v) - 1
    return float(v[idx])

def estimate_global_scale(
    user: Dict[str, Optional[float]],
    photo: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
    clamp: Tuple[float, float] = (0.85, 1.15),
) -> Dict[str, Any]:
    """
    Compare user vs photo measurements and return a robust global scale factor.
    
    This helps correct systematic errors in photo-based measurement extraction.
    
    Returns:
      - 'scale': global scaling factor to apply to all photo values
      - 'per_item': {k: s_k = user_k / photo_k}
      - 'errors': {k: 100 * (photo_k - user_k) / user_k}
      - 'used_pairs': [(key, user_val, photo_val)]
    """
    weights = weights or DEFAULT_SCALE_WEIGHTS
    per_item: Dict[str, float] = {}
    errors: Dict[str, float] = {}
    used_pairs: List[Tuple[str, float, float]] = []
    s_vals: List[float] = []
    w_vals: List[float] = []

    for k, w in weights.items():
        u = user.get(k)
        p = photo.get(k)
        if u and u > 0 and p and p > 0:
            s_k = float(u) / float(p)
            per_item[k] = s_k
            errors[k] = 100.0 * (float(p) - float(u)) / float(u)
            used_pairs.append((k, float(u), float(p)))
            s_vals.append(s_k)
            w_vals.append(w)

    if not s_vals:
        return {"scale": 1.0, "per_item": per_item, "errors": errors, "used_pairs": used_pairs}

    s_med = _weighted_median(s_vals, w_vals)
    s = float(np.clip(s_med, clamp[0], clamp[1]))
    
    return {
        "scale": s,
        "per_item": per_item,
        "errors": errors,
        "used_pairs": used_pairs
    }

def apply_scale_to_measurements(
    measurements: Dict[str, Optional[float]],
    scale: float
) -> Dict[str, Optional[float]]:
    """Apply global scale factor to all measurements."""
    out: Dict[str, Optional[float]] = {}
    for k, v in measurements.items():
        if v is not None and isinstance(v, (int, float)) and v > 0:
            out[k] = float(v) * scale
        else:
            out[k] = v
    return out

# ====================== Original Functions (Kept) ======================

_POSE_MODEL = None
_FACE_MODEL = None

def _get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        _POSE_MODEL = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    return _POSE_MODEL

def _get_face_model():
    global _FACE_MODEL
    if _FACE_MODEL is None:
        _FACE_MODEL = mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
    return _FACE_MODEL

def cleanup_models():
    global _POSE_MODEL, _FACE_MODEL
    if _POSE_MODEL:
        _POSE_MODEL.close()
        _POSE_MODEL = None
    if _FACE_MODEL:
        _FACE_MODEL.close()
        _FACE_MODEL = None

def b64_to_img(b64_str: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(b64_str.split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def focus_score(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def _shoulders(pose_lm, w: int, h: int):
    if not pose_lm:
        return None, None, None, None
    try:
        L = pose_lm.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        R = pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        pL = np.array([L.x * w, L.y * h], dtype=np.float32)
        pR = np.array([R.x * w, R.y * h], dtype=np.float32)
        vec = pR - pL
        length = float(np.linalg.norm(vec))
        roll_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
        mid = (pR + pL) * 0.5
        return vec, length, roll_deg, mid
    except Exception:
        return None, None, None, None

def _face_box_and_yaw(face_lm, w: int, h: int):
    if not face_lm:
        return None, None, None
    xs, ys = [], []
    for lm in face_lm.landmark:
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
    x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))
    box = [x1, y1, x2, y2]
    yaw = None
    abs_yaw = None
    try:
        nose = face_lm.landmark[4]
        left = face_lm.landmark[234]
        right = face_lm.landmark[454]
        denom = max(1e-6, (right.x - left.x))
        yaw = float((nose.x - 0.5 * (left.x + right.x)) / denom)
        abs_yaw = abs(yaw)
    except Exception:
        pass
    return box, yaw, abs_yaw

def _person_bbox_height_px(pose_lm, face_box, h: int):
    if pose_lm:
        try:
            top = min(
                pose_lm.landmark[mp_pose.PoseLandmark.NOSE].y,
                pose_lm.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            ) * h
            bottom = max(
                pose_lm.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                pose_lm.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            ) * h
            return float(max(1.0, bottom - top))
        except Exception:
            pass
    if face_box:
        return float((face_box[3] - face_box[1]) * 2.8)
    return None

def assign_view(abs_yaw, shoulder_vec, roll_deg) -> str:
    if abs_yaw is None or shoulder_vec is None or roll_deg is None:
        return "unknown"
    horiz = abs(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
    if abs_yaw < 0.15 and horiz < 20:
        return "front"
    if abs_yaw > 0.40 or horiz > 45:
        return "side"
    return "back"

def estimate_camera(h_img: int, bbox_h_px: Optional[float], height_m: float, vfov_deg: float = 49.0):
    vfov = math.radians(vfov_deg)
    f_pix = 0.5 * h_img / math.tan(0.5 * vfov)
    out = {"fov_v_deg": vfov_deg, "f_pix": float(f_pix)}
    if bbox_h_px and bbox_h_px > 0:
        dist = (height_m * f_pix) / bbox_h_px
        out["distance_m"] = float(dist)
        out["px_per_meter"] = float(bbox_h_px / height_m)
    return out

DEFAULT_THRESHOLDS = {
    "min_focus": 200.0,
    "min_shoulder_ratio": 0.20,
    "max_roll_deg": 10.0,
    "front_max_abs_yaw": 0.20,
    "side_min_abs_yaw": 0.35,
}

def _angle_2d(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    nba = ba / (np.linalg.norm(ba) + 1e-6)
    nbc = bc / (np.linalg.norm(bc) + 1e-6)
    cosang = float(np.clip(np.dot(nba, nbc), -1.0, 1.0))
    return math.degrees(math.acos(cosang))

def pose_angles_from_mediapipe(pose_lm, face_lm, w, h) -> dict:
    if not pose_lm:
        return {}
    def pt(idx):
        p = pose_lm.landmark[idx]
        return (p.x*w, p.y*h)
    LSH, RSH = pt(mp_pose.PoseLandmark.LEFT_SHOULDER), pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    LEL, REL = pt(mp_pose.PoseLandmark.LEFT_ELBOW), pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
    LWR, RWR = pt(mp_pose.PoseLandmark.LEFT_WRIST), pt(mp_pose.PoseLandmark.RIGHT_WRIST)
    left_elbow = _angle_2d(LSH, LEL, LWR)
    right_elbow = _angle_2d(RSH, REL, RWR)
    def shoulder_abd(sh, el):
        v = (el[0]-sh[0], el[1]-sh[1])
        ang = math.degrees(math.atan2(v[0], -v[1]))
        return ang
    left_sh_abd = shoulder_abd(LSH, LEL)
    right_sh_abd = shoulder_abd(RSH, REL)
    yaw = None
    if face_lm:
        try:
            nose = face_lm.landmark[4]
            left = face_lm.landmark[234]
            right = face_lm.landmark[454]
            denom = max(1e-6, (right.x - left.x))
            yaw = float((nose.x - 0.5*(left.x+right.x)) / denom) * 60.0
        except Exception:
            pass
    return {
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_shoulder_abd": left_sh_abd,
        "right_shoulder_abd": right_sh_abd,
        "head_yaw": yaw
    }

# ====================== ENHANCED: With Scale Calibration ======================

def analyze_one(
    img: np.ndarray,
    height_m: float,
    user_measurements: Optional[Dict[str, float]] = None,
    extract_measurements: bool = True
) -> Dict[str, Any]:
    """
    Analyze a single image with ENHANCED scale calibration.
    
    NEW: Compares photo-extracted vs user-provided measurements
    to estimate systematic error and apply correction.
    
    Args:
        img: Image array
        height_m: User's height (mandatory, used as reference)
        user_measurements: User-provided measurements (take priority + used for scale)
        extract_measurements: Whether to extract measurements from photo
        
    Returns:
        Analysis dict with measurements_scale added for debugging
    """
    h, w = img.shape[:2]
    f = focus_score(img)
    
    pose = _get_pose_model()
    face = _get_face_model()
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pose_res = pose.process(rgb)
    face_res = face.process(rgb)
    
    plm = getattr(pose_res, "pose_landmarks", None)
    flm = face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None
    
    svec, s_len, roll_deg, _ = _shoulders(plm, w, h)
    fbox, yaw, abs_yaw = _face_box_and_yaw(flm, w, h)
    bbox_h_px = _person_bbox_height_px(plm, fbox, h)
    role = assign_view(abs_yaw, svec, roll_deg)
    cam = estimate_camera(h, bbox_h_px, height_m)
    angles = pose_angles_from_mediapipe(plm, flm, w, h)
    
    result = {
        "focus": f,
        "shoulder_len_px": s_len,
        "shoulder_len_ratio": (s_len / w) if s_len else 0.0,
        "roll_deg": roll_deg,
        "yaw": yaw,
        "abs_yaw": abs_yaw if abs_yaw is not None else 9.0,
        "role": role,
        "bbox_face": fbox,
        "bbox_h_px": bbox_h_px,
        "camera": cam,
        "image_h": h,
        "image_w": w,
        "pose_angles": angles
    }
    
    # ===== ENHANCED: Extract measurements with scale calibration =====
    if extract_measurements and plm:
        try:
            from measurement_extractor import (
                extract_all_measurements,
                merge_measurements,
                validate_measurement_sanity
            )
            
            camera_info = {
                'distance_m': cam.get('distance_m', 2.5),
                'roll_deg': roll_deg if roll_deg is not None else 0.0
            }
            
            # Extract from photo
            photo_measurements = extract_all_measurements(
                plm, h, w, height_m, camera_info
            )
            
            # NEW: Estimate scale factor if user provided measurements
            user_meas = user_measurements or {}
            scale_info = estimate_global_scale(user_meas, photo_measurements)
            
            # Apply scale correction to photo measurements
            photo_measurements_scaled = apply_scale_to_measurements(
                photo_measurements,
                scale_info["scale"]
            )
            
            # Merge with user measurements (user takes priority)
            merged = merge_measurements(photo_measurements_scaled, user_meas)
            
            # Validate sanity
            validated = validate_measurement_sanity(merged, height_m)
            
            # Add to result
            result['measurements_photo'] = photo_measurements
            result['measurements_photo_scaled'] = photo_measurements_scaled
            result['measurements_merged'] = validated
            result['measurements_available'] = True
            
            # NEW: Add scale debugging info
            result['measurements_scale'] = {
                'scale_factor': scale_info['scale'],
                'per_feature': scale_info['per_item'],
                'errors_percent': scale_info['errors'],
                'pairs_used': scale_info['used_pairs']
            }
            
            print(f"[VISION] ✓ Measurements extracted for {role}")
            print(f"[VISION] Scale factor: {scale_info['scale']:.3f}")
            
        except Exception as e:
            print(f"[VISION] Warning: Measurement extraction failed: {e}")
            result['measurements_available'] = False
    else:
        result['measurements_available'] = False
    
    return result

# ====================== KEPT: Original API ======================

def quality_ok(
    analysis: Dict[str, Any],
    role_hint: Optional[str],
    thr: Dict[str, float] = DEFAULT_THRESHOLDS
) -> Tuple[bool, List[str]]:
    """Check if photo passes quality thresholds. KEPT ORIGINAL SIGNATURE."""
    reasons = []
    if analysis.get("focus", 0.0) < thr["min_focus"]:
        reasons.append(f"low focus ({analysis.get('focus', 0.0):.1f} < {thr['min_focus']})")
    if analysis.get("shoulder_len_ratio", 0.0) < thr["min_shoulder_ratio"]:
        reasons.append("shoulders not fully visible / subject too far")
    roll = abs(analysis.get("roll_deg") or 0.0)
    if roll > thr["max_roll_deg"]:
        reasons.append(f"camera tilt {roll:.1f}° > {thr['max_roll_deg']}°")
    if role_hint == "front":
        if (analysis.get("abs_yaw", 9.0)) > thr["front_max_abs_yaw"]:
            reasons.append("not facing camera enough for front view")
    if role_hint == "side":
        if (analysis.get("abs_yaw", 0.0)) < thr["side_min_abs_yaw"]:
            reasons.append("not turned enough for side view")
    return (len(reasons) == 0), reasons

def choose_roles(
    analyses: List[Dict[str, Any]],
    photos_b64: List[str]
) -> Dict[str, Any]:
    """Choose best photo for each role. KEPT ORIGINAL FUNCTION."""
    scored = []
    for idx, a in enumerate(analyses):
        q = a.get("focus", 0.0) * (1.0 + 0.4 * a.get("shoulder_len_ratio", 0.0))
        scored.append((idx, a.get("role", "unknown"), q, a.get("abs_yaw", 9.0)))
    
    def best_for(role: str) -> int:
        cands = [(i, q, yaw) for (i, r, q, yaw) in scored if r == role]
        if not cands:
            cands = [(i, q, yaw) for (i, r, q, yaw) in scored]
        if role == "side":
            cands.sort(key=lambda t: (-t[1], -t[2]))
        else:
            cands.sort(key=lambda t: (-t[1], t[2]))
        return cands[0][0]
    
    mapping = {}
    for role in ("front", "side", "back"):
        idx = best_for(role)
        mapping[role] = photos_b64[idx] if 0 <= idx < len(photos_b64) else None
    
    return {"by_role": mapping, "scores": analyses}

def get_best_measurements(
    analyses: List[Dict[str, Any]],
    user_measurements: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Get best measurements across all photos. KEPT ORIGINAL FUNCTION."""
    front_analyses = [a for a in analyses if a.get('role') == 'front']
    if not front_analyses:
        front_analyses = analyses
    
    best = max(front_analyses, key=lambda a: a.get('focus', 0) * a.get('shoulder_len_ratio', 0))
    
    measurements = best.get('measurements_merged', {})
    
    if user_measurements:
        for key, value in user_measurements.items():
            if value is not None and value > 0:
                measurements[key] = value
    
    return {k: v for k, v in measurements.items() if v is not None and v > 0}

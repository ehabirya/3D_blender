# app/vision.py
"""
Shared vision utilities (FINAL)
- CPU-only: MediaPipe Pose + FaceMesh
- Decoding, focus/quality metrics, role assignment (front/side/back)
- Camera distance estimate (from person bbox height + user height)
- Quality gating with DEFAULT_THRESHOLDS
- Role selection helpers (choose_roles)

Used by:
- calibration.py  (photo QA, tips, ranking)
- photo_selector.py (if you keep the thin wrapper)
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import base64
import math

import numpy as np
import cv2

# MediaPipe (CPU)
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh


# ------------------------- I/O helpers -------------------------

def b64_to_img(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 → BGR image (OpenCV). Returns None on failure."""
    try:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def focus_score(img: np.ndarray) -> float:
    """Laplacian variance focus metric (higher = sharper)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ------------------------- geometry helpers -------------------------

def _shoulders(pose_lm, w: int, h: int) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[np.ndarray]]:
    """
    Returns (vector R-L, length_px, roll_deg, mid_xy) or (None,...)
    roll ≈ arctan(dy/dx) in degrees: camera tilt proxy.
    """
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

def _face_box_and_yaw(face_lm, w: int, h: int) -> Tuple[Optional[List[int]], Optional[float], Optional[float]]:
    """
    Face bbox in pixels and a crude yaw proxy in normalized units.
    yaw sign: + right / - left; abs_yaw small ≈ front.
    """
    if not face_lm:
        return None, None, None
    xs, ys = [], []
    for lm in face_lm.landmark:
        xs.append(lm.x * w); ys.append(lm.y * h)
    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
    x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))
    box = [x1, y1, x2, y2]

    yaw = None
    abs_yaw = None
    try:
        # Mediapipe indices: 234 (left cheek), 454 (right cheek), 4 (nose tip)
        nose = face_lm.landmark[4]
        left = face_lm.landmark[234]
        right = face_lm.landmark[454]
        denom = max(1e-6, (right.x - left.x))
        yaw = float((nose.x - 0.5 * (left.x + right.x)) / denom)
        abs_yaw = abs(yaw)
    except Exception:
        pass

    return box, yaw, abs_yaw

def _person_bbox_height_px(pose_lm, face_box, h: int) -> Optional[float]:
    """Rough person bbox height in pixels using ankles/shoulders; fallback from face height."""
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
        # crude proxy when only face is visible
        return float((face_box[3] - face_box[1]) * 2.8)
    return None

def assign_view(abs_yaw: Optional[float], shoulder_vec: Optional[np.ndarray], roll_deg: Optional[float]) -> str:
    """
    Classify into front/side/back/unknown using yaw + shoulder orientation.
    """
    if abs_yaw is None or shoulder_vec is None or roll_deg is None:
        return "unknown"
    # approximate horizontal angle of shoulders; front ≈ horizontal
    horiz = abs(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
    if abs_yaw < 0.15 and horiz < 20:
        return "front"
    if abs_yaw > 0.40 or horiz > 45:
        return "side"
    return "back"

def estimate_camera(h_img: int, bbox_h_px: Optional[float], height_m: float, vfov_deg: float = 49.0) -> Dict[str, float]:
    """
    Estimate camera distance and focal in pixels assuming ~49° vertical FOV (typical phone).
    """
    vfov = math.radians(vfov_deg)
    f_pix = 0.5 * h_img / math.tan(0.5 * vfov)
    out = {"fov_v_deg": vfov_deg, "f_pix": float(f_pix)}
    if bbox_h_px and bbox_h_px > 0:
        dist = (height_m * f_pix) / bbox_h_px
        out["distance_m"] = float(dist)
        out["px_per_meter"] = float(bbox_h_px / height_m)
    return out


# ------------------------- thresholds & analysis -------------------------

DEFAULT_THRESHOLDS = {
    "min_focus": 200.0,           # Laplacian variance
    "min_shoulder_ratio": 0.20,   # shoulder_len_px / image_width
    "max_roll_deg": 10.0,         # camera tilt
    "front_max_abs_yaw": 0.20,    # front must face camera
    "side_min_abs_yaw": 0.35,     # side must be turned enough
}

def analyze_one(img: np.ndarray, height_m: float) -> Dict[str, Any]:
    """
    Analyze one photo:
    - focus, shoulder span/roll, face bbox + yaw
    - role prediction (front/side/back)
    - camera distance estimate
    """
    h, w = img.shape[:2]
    f = focus_score(img)

    with mp_pose.Pose(static_image_mode=True) as pose, mp_face.FaceMesh(static_image_mode=True, refine_landmarks=False) as face:
        pose_res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        face_res = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plm = getattr(pose_res, "pose_landmarks", None)
        flm = face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None

        svec, s_len, roll_deg, _ = _shoulders(plm, w, h)
        fbox, yaw, abs_yaw = _face_box_and_yaw(flm, w, h)
        bbox_h_px = _person_bbox_height_px(plm, fbox, h)
        role = assign_view(abs_yaw, svec, roll_deg)
        cam = estimate_camera(h, bbox_h_px, height_m)

        return {
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
            "image_w": w
        }

def quality_ok(analysis: Dict[str, Any], role_hint: Optional[str], thr: Dict[str, float] = DEFAULT_THRESHOLDS) -> Tuple[bool, List[str]]:
    """
    Decide if a photo passes quality thresholds; return (ok, reasons).
    role_hint: 'front'|'side'|'back'|None — tighten yaw checks if provided.
    """
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


# ------------------------- role selection -------------------------

def choose_roles(analyses: List[Dict[str, Any]], photos_b64: List[str]) -> Dict[str, Any]:
    """
    Pick single best photo per role; fallback to any if no role match.
    Heuristic scoring: focus * (1 + 0.4 * shoulder_ratio), tie-break by abs_yaw.
    Returns {"by_role": {"front": b64|None, "side": b64|None, "back": b64|None}, "scores": analyses}
    """
    scored = []
    for idx, a in enumerate(analyses):
        q = a.get("focus", 0.0) * (1.0 + 0.4 * a.get("shoulder_len_ratio", 0.0))
        scored.append((idx, a.get("role", "unknown"), q, a.get("abs_yaw", 9.0)))

    def best_for(role: str) -> int:
        cands = [(i, q, yaw) for (i, r, q, yaw) in scored if r == role]
        if not cands:
            cands = [(i, q, yaw) for (i, r, q, yaw) in scored]  # fallback: any
        # front/back prefer low yaw; side prefers high yaw
        if role == "side":
            cands.sort(key=lambda t: (-t[1], -t[2]))   # high quality, larger yaw
        else:
            cands.sort(key=lambda t: (-t[1], t[2]))    # high quality, smaller yaw
        return cands[0][0]

    mapping = {}
    for role in ("front", "side", "back"):
        idx = best_for(role)
        mapping[role] = photos_b64[idx] if 0 <= idx < len(photos_b64) else None

    return {"by_role": mapping, "scores": analyses}

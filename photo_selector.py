# app/photo_selector.py
"""
Photo selector for digital-twin pipeline.

- Input: 2–4 unordered base64 images (front/side/back of a person) + optional hints
- Output:
    {
      "by_role": {"front": b64|None, "side": b64|None, "back": b64|None},
      "scores": [
         {"index": 0, "role": "front|side|back|unknown", "focus": float,
          "abs_yaw": float|None, "shoulder_len": float|None, "face_bbox": [x1,y1,x2,y2]|None}
      ],
      "preset": "male|female|neutral|child|baby"
    }

Heuristics:
- View selection via face yaw (FaceMesh) and shoulder line orientation (Pose).
- Quality scoring via Laplacian focus; prefer visible shoulders.
- Preset guess via face-to-shoulder ratio + optional hints.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import base64
import cv2
import numpy as np

# MediaPipe (CPU)
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh


# ------------------------- utils -------------------------

def _b64_to_img(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid base64 image")
    return img

def _focus_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _shoulder_metrics(pose_landmarks, w: int, h: int) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Returns (vector R-L, length_px, mid_y)
    """
    if not pose_landmarks:
        return None, None, None
    try:
        L = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        R = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        pL = np.array([L.x * w, L.y * h])
        pR = np.array([R.x * w, R.y * h])
        vec = pR - pL
        length = float(np.linalg.norm(vec))
        midy = float((pR[1] + pL[1]) / 2.0)
        return vec, length, midy
    except Exception:
        return None, None, None

def _face_box_and_yaw(face_landmarks, w: int, h: int) -> Tuple[Optional[List[int]], Optional[float], Optional[float]]:
    if not face_landmarks:
        return None, None, None
    xs, ys = [], []
    for lm in face_landmarks.landmark:
        xs.append(lm.x * w); ys.append(lm.y * h)
    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
    x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))
    box = [x1, y1, x2, y2]
    # crude yaw proxy using nose vs ear-midline
    try:
        nose = face_landmarks.landmark[4]
        left = face_landmarks.landmark[234]
        right = face_landmarks.landmark[454]
        denom = max(1e-6, (right.x - left.x))
        yaw = ((nose.x - 0.5 * (left.x + right.x)) / denom)
        abs_yaw = abs(yaw)
    except Exception:
        yaw, abs_yaw = None, None
    return box, yaw, abs_yaw

def _assign_view(abs_yaw: Optional[float], shoulder_vec: Optional[np.ndarray]) -> str:
    """
    Use face yaw and shoulder-line tilt to pick front/side/back.
    """
    if abs_yaw is None or shoulder_vec is None:
        return "unknown"
    horiz_deg = abs(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
    # thresholds tuned for simple studio shots
    if abs_yaw < 0.15 and horiz_deg < 20:
        return "front"
    if abs_yaw > 0.40 or horiz_deg > 45:
        return "side"
    # if still ambiguous, call it back (useful when face is occluded/hair)
    return "back"

def _child_baby_guess(face_box: Optional[List[int]], shoulder_len: Optional[float]) -> Optional[str]:
    """
    Kids/babies have a larger face-to-shoulder ratio.
    """
    if not face_box or not shoulder_len or shoulder_len <= 0:
        return None
    fw = face_box[2] - face_box[0]
    ratio = fw / shoulder_len
    if ratio > 1.10:  # very large head vs shoulders
        return "baby"
    if ratio > 0.80:
        return "child"
    return None

def _gender_guess_hook(img: np.ndarray) -> Optional[str]:
    """
    Placeholder for an ONNX age/gender model if you add one.
    """
    return None


# ------------------------- main API -------------------------

def analyze_photos(photos_b64: List[str], hints: Dict[str, str] | None = None) -> Dict[str, Any]:
    """
    Analyze 2–4 unordered photos.
    Returns by_role mapping, per-photo scores, and preset guess.
    """
    if not photos_b64:
        raise ValueError("No photos provided")
    if len(photos_b64) == 1:
        photos_b64 = photos_b64 * 3  # replicate if only one provided

    imgs = []
    for b in photos_b64:
        try:
            imgs.append(_b64_to_img(b))
        except Exception:
            # skip bad image
            pass
    if not imgs:
        raise ValueError("Could not decode any image")

    results = []
    by_role = {"front": None, "side": None, "back": None}

    with mp_pose.Pose(static_image_mode=True) as pose, mp_face.FaceMesh(static_image_mode=True, refine_landmarks=False) as face:
        for idx, img in enumerate(imgs):
            h, w = img.shape[:2]
            focus = _focus_score(img)

            pose_res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            face_res = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            vec, shoulder_len, _ = _shoulder_metrics(getattr(pose_res, "pose_landmarks", None), w, h)
            fm = (face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None)
            fbox, yaw, abs_yaw = _face_box_and_yaw(fm, w, h)

            role = _assign_view(abs_yaw, vec)
            # prefer images where shoulders are visible and focus is high
            quality_bonus = (shoulder_len or 0.0) / max(1.0, w * 0.35)
            score = float(focus * (1.0 + 0.2 * quality_bonus))

            results.append({
                "index": idx, "role": role, "score": score,
                "focus": focus, "abs_yaw": (abs_yaw if abs_yaw is not None else 9.0),
                "shoulder_len": shoulder_len, "face_bbox": fbox
            })

    # choose best for each role
    def choose(role: str) -> Optional[int]:
        cands = [r for r in results if r["role"] == role]
        if not cands:
            cands = results[:]  # fallback: any
        cands.sort(key=lambda r: (-r["score"], r["abs_yaw"]))
        return cands[0]["index"] if cands else None

    for role in ("front", "side", "back"):
        idx = choose(role)
        if idx is not None and idx < len(photos_b64):
            by_role[role] = photos_b64[idx]

    # ---- preset guess ----
    preset = "neutral"
    if hints:
        age_hint = (hints.get("age_hint") or "").lower()
        gender_hint = (hints.get("gender_hint") or "").lower()
        if age_hint in ("child", "baby"):
            preset = age_hint
        if gender_hint in ("male", "female"):
            preset = gender_hint if preset == "neutral" else preset  # age overrides gender here

    if preset == "neutral":
        # infer from front image shape ratios
        front_info = next((r for r in results if r["role"] == "front"), None)
        child_guess = _child_baby_guess(front_info["face_bbox"] if front_info else None,
                                        front_info["shoulder_len"] if front_info else None)
        if child_guess:
            preset = child_guess
        else:
            g = _gender_guess_hook(imgs[0])
            if g in ("male", "female"):
                preset = g

    return {"by_role": by_role, "scores": results, "preset": preset}


# ------------------------- local test -------------------------
if __name__ == "__main__":
    import json, sys, pathlib

    print("photo_selector.py — CLI test")
    print("Usage: python photo_selector.py <img1.b64> <img2.b64> <img3.b64>")
    if len(sys.argv) > 1:
        b64s = []
        for p in sys.argv[1:]:
            pth = pathlib.Path(p)
            if pth.exists():
                b = base64.b64encode(pth.read_bytes()).decode("utf-8")
                b64s.append(b)
        out = analyze_photos(b64s)
        print(json.dumps(out, indent=2)[:2000])

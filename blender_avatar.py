import os, subprocess, tempfile, base64, shutil
from typing import Dict, Any

import cv2
import numpy as np

BLENDER = os.environ.get("BLENDER_BIN", "/blender/blender")
DEFORM_SCRIPT = "/app/deform_avatar.py"
ASSETS_DIR = "/app/assets"

BASES = {
    "male":   os.path.join(ASSETS_DIR, "base_male.blend"),
    "female": os.path.join(ASSETS_DIR, "base_female.blend"),
    "neutral":os.path.join(ASSETS_DIR, "base_neutral.blend"),
    "child":  os.path.join(ASSETS_DIR, "base_child.blend"),
    "baby":   os.path.join(ASSETS_DIR, "base_baby.blend"),
}

# -------------------- image helpers --------------------

def _b64_to_np(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    # simple gray-world: scale channels so their means match the global gray mean
    imgf = img.astype(np.float32) + 1e-6
    means = imgf.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
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
    # light denoise to preserve edges
    return cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

def _unsharp(img: np.ndarray, radius=3, amount=0.6) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    return sharp

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

    img = _gray_world_white_balance(img)
    img = _clahe_contrast(img)
    img = _denoise(img)
    img = _unsharp(img, radius=2.5, amount=0.5)
    img = _resize_max(img, max_side)

    ok = cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Could not write preprocessed image to {out_path}")
    return out_path

# ------------------------------------------------------

def run_blender_avatar(preset: str, height_m: float,
                       measurements: Dict[str, float],
                       photos: Dict[str, str],
                       tex_res: int = 2048) -> Dict[str, Any]:
    """Run Blender deformation headlessly and return GLB as base64."""
    blend_path = BASES.get(preset, BASES["neutral"])
    if not os.path.exists(blend_path):
        return {"ok": False, "error": f"missing base file for {preset}"}

    tmpdir = tempfile.mkdtemp(prefix="avatar_")
    try:
        out_glb = os.path.join(tmpdir, "twin.glb")

        cmd = [
            BLENDER, "-b", blend_path, "--python", DEFORM_SCRIPT, "--",
            "--preset", preset, "--height", str(height_m),
            "--out", out_glb, "--texRes", str(int(tex_res))
        ]

        for k, v in (measurements or {}).items():
            if v is not None:
                cmd += [f"--{k}", str(v)]

        # preprocess photos (if provided)
        for role in ("front", "side", "back"):
            b64img = photos.get(role)
            if b64img:
                out_png = os.path.join(tmpdir, f"{role}.png")
                try:
                    preprocess_and_save(b64img, out_png, max_side=int(tex_res))
                    cmd += [f"--{role}Tex", out_png]
                except Exception as e:
                    # continue without this texture; Blender will blend remaining ones
                    print(f"[WARN] Preprocess failed for {role}: {e}")

        # Execute Blender
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0 or not os.path.exists(out_glb):
            return {"ok": False, "error": "Blender failed", "log": proc.stdout[-4000:]}

        with open(out_glb, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"ok": True, "glb_b64": glb_b64, "log": proc.stdout[-4000:]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

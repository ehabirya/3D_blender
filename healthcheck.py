#!/usr/bin/env python3
import os, sys, subprocess, importlib

BLENDER_BIN = os.environ.get("BLENDER_BIN", "/blender/blender")

def ok(msg):  # print to stdout so Docker logs it
    print("[HEALTH] " + msg)

def fail(msg, code=1):
    print("[HEALTH] FAIL: " + msg, file=sys.stderr)
    sys.exit(code)

def check_blender():
    if not os.path.exists(BLENDER_BIN):
        fail(f"Blender not found at {BLENDER_BIN}")
    try:
        proc = subprocess.run([BLENDER_BIN, "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
        if proc.returncode != 0:
            fail("Blender returned non-zero exit")
        ok("Blender OK")
    except subprocess.TimeoutExpired:
        fail("Blender -v timed out")
    except Exception as e:
        fail(f"Blender check error: {e}")

def check_imports():
    try:
        importlib.import_module("cv2")
        ok("OpenCV OK")
    except Exception as e:
        fail(f"OpenCV import failed: {e}")
    # Mediapipe is heavier; keep it but don’t instantiate models
    try:
        importlib.import_module("mediapipe")
        ok("MediaPipe OK")
    except Exception as e:
        fail(f"MediaPipe import failed: {e}")

if __name__ == "__main__":
    check_blender()
    check_imports()
    ok("All checks passed")
    sys.exit(0)

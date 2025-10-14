# ===== CPU-only, slim image with Blender headless + Python =====
FROM python:3.10-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLENDER_VERSION=3.6.9 \
    BLENDER_DIR=/blender \
    BLENDER_BIN=/blender/blender

# ---- System deps for Blender + OpenCV/MediaPipe ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget xz-utils bzip2 \
    libgl1 libglib2.0-0 libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 \
    libxkbcommon0 libsm6 libice6 libxrandr2 libfontconfig1 libfreetype6 \
    libxext6 libxinerama1 libxtst6 libxcomposite1 libxdamage1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Blender (headless) ----
# LTS is safest for CLI baking/export. Change version via BLENDER_VERSION arg above.
RUN set -eux; \
    BLENDER_TAR="blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
    wget -q "https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/${BLENDER_TAR}" -O /tmp/blender.tar.xz; \
    mkdir -p "${BLENDER_DIR}"; \
    tar -xf /tmp/blender.tar.xz -C /tmp; \
    mv /tmp/blender-${BLENDER_VERSION}-linux-x64/* "${BLENDER_DIR}/"; \
    rm -rf /tmp/blender*; \
    "${BLENDER_BIN}" -v

# ---- Python deps ----
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- App code ----
# Make sure your repo copies these:
#   app/runpod_handler.py
#   app/blender_avatar.py
#   app/deform_avatar.py
#   app/calibration.py
#   app/vision.py
#   app/assets/*  (your base_male.blend, etc.)
COPY app /app

# Ensure assets folder exists even if empty (so you can mount/commit them later)
RUN mkdir -p /app/assets

# Environment for our code
ENV BLENDER_BIN=/blender/blender \
    PYTHONPATH="/app:${PYTHONPATH}"

# Default command for RunPod serverless;
# runpod will wrap this, but this keeps local docker run working too.
CMD ["python", "-u", "runpod_handler.py"]

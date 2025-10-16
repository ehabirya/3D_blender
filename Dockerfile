# =========================
# Blender + Python (CPU) images
# =========================
FROM python:3.10-slim AS runtime

ARG BLENDER_VERSION=3.6.9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLENDER_DIR=/blender \
    BLENDER_BIN=/blender/blender \
    PYTHONPATH="/app:${PYTHONPATH}"

# ---- OS deps for Blender, OpenCV, MediaPipe ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget xz-utils bzip2 \
    # GL + X deps required by Blender and OpenCV (headless still needs these libs)
    libgl1 libglib2.0-0 libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 \
    libxkbcommon0 libsm6 libice6 libxrandr2 libfontconfig1 libfreetype6 \
    libxext6 libxinerama1 libxtst6 libxcomposite1 libxdamage1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Blender (headless/CLI) ----
RUN set -eux; \
    BLENDER_TAR="blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
    wget -q "https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/${BLENDER_TAR}" -O /tmp/blender.tar.xz; \
    mkdir -p "${BLENDER_DIR}"; \
    tar -xf /tmp/blender.tar.xz -C /tmp; \
    mv /tmp/blender-${BLENDER_VERSION}-linux-x64/* "${BLENDER_DIR}/"; \
    rm -rf /tmp/blender*; \
    "${BLENDER_BIN}" -v

# ---- App workspace ----
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code (all Python files and assets from root directory)
COPY *.py /app/
COPY assets /app/assets

# Ensure assets dir exists (you'll commit your base_*.blend into this)
RUN mkdir -p /app/assets

# ---- Healthcheck script ----
COPY healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# ---- Non-root user (safer in serverless) ----
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# ---- Healthcheck (quick & safe) ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD ["python", "/app/healthcheck.py"]

# ---- Default command: RunPod serverless handler ----
ENV BLENDER_BIN=/blender/blender
CMD ["python", "-u", "runpod_handler.py"]

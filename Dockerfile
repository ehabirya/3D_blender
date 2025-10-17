# =========================
# Blender + Python (CPU) image - IMPROVED VERSIONs
# =========================
# FIXES:
# - Added retry logic for Blender download
# - Multiple mirror URLs with fallback
# - Better error handling and logging
# - Uses curl with proper user-agent
# =========================

FROM python:3.10-slim AS runtime

ARG BLENDER_VERSION=3.6.9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLENDER_DIR=/blender \
    BLENDER_BIN=/blender/blender \
    PYTHONPATH="/app:${PYTHONPATH:-}"

# ---- OS deps for Blender, OpenCV, MediaPipe ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget curl xz-utils bzip2 \
    libgl1 libglib2.0-0 libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 \
    libxkbcommon0 libsm6 libice6 libxrandr2 libfontconfig1 libfreetype6 \
    libxext6 libxinerama1 libxtst6 libxcomposite1 libxdamage1 libasound2 \
 && rm -rf /var/lib/apt/lists/*

# ---- Install Blender (with retry logic and multiple mirrors) ----
RUN set -eux; \
    BLENDER_TAR="blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
    BLENDER_MAJOR="${BLENDER_VERSION%.*}"; \
    \
    # List of mirror URLs to try
    MIRRORS="\
        https://mirror.clarkson.edu/blender/release/Blender${BLENDER_MAJOR}/${BLENDER_TAR} \
        https://ftp.nluug.nl/pub/graphics/blender/release/Blender${BLENDER_MAJOR}/${BLENDER_TAR} \
        https://download.blender.org/release/Blender${BLENDER_MAJOR}/${BLENDER_TAR} \
    "; \
    \
    echo "===== Downloading Blender ${BLENDER_VERSION} ====="; \
    DOWNLOADED=false; \
    \
    for MIRROR_URL in ${MIRRORS}; do \
        echo "Trying mirror: ${MIRROR_URL}"; \
        for attempt in 1 2 3; do \
            echo "  Attempt ${attempt}/3..."; \
            if curl -fsSL \
                --user-agent "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
                --connect-timeout 30 \
                --max-time 600 \
                --retry 2 \
                --retry-delay 5 \
                "${MIRROR_URL}" \
                -o /tmp/blender.tar.xz; then \
                echo "  ✓ Download successful from ${MIRROR_URL}!"; \
                DOWNLOADED=true; \
                break 2; \
            else \
                echo "  ✗ Download failed"; \
                if [ "$attempt" -lt 3 ]; then \
                    echo "  Retrying in 5 seconds..."; \
                    sleep 5; \
                fi; \
            fi; \
        done; \
        echo "  Failed all attempts for this mirror, trying next..."; \
    done; \
    \
    if [ "$DOWNLOADED" = false ]; then \
        echo "ERROR: Failed to download Blender from all mirrors!"; \
        echo "Please check your network connection and try again."; \
        exit 1; \
    fi; \
    \
    echo "===== Extracting Blender ====="; \
    FILE_SIZE=$(stat -f%z /tmp/blender.tar.xz 2>/dev/null || stat -c%s /tmp/blender.tar.xz); \
    echo "Downloaded file size: ${FILE_SIZE} bytes (~$(expr ${FILE_SIZE} / 1048576)MB)"; \
    \
    if [ ${FILE_SIZE} -lt 100000000 ]; then \
        echo "ERROR: Downloaded file is too small (corrupted download)"; \
        exit 1; \
    fi; \
    \
    mkdir -p "${BLENDER_DIR}"; \
    tar -xf /tmp/blender.tar.xz -C /tmp; \
    mv /tmp/blender-${BLENDER_VERSION}-linux-x64/* "${BLENDER_DIR}/"; \
    rm -rf /tmp/blender*; \
    \
    echo "===== Verifying Blender Installation ====="; \
    "${BLENDER_BIN}" --version || exit 1; \
    echo "✓ Blender ${BLENDER_VERSION} installed successfully!"

# ---- App workspace ----
WORKDIR /app

# Copy requirements first (leverage layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with progress output
RUN echo "===== Installing Python Dependencies =====" && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    echo "✓ Python dependencies installed successfully!"

# Copy application code and assets
COPY *.py /app/
COPY assets /app/assets
RUN mkdir -p /app/assets

# ---- Healthcheck script ----
COPY healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# ---- Non-root user ----
RUN useradd -m -u 10001 appuser && \
    chown -R appuser:appuser /app && \
    echo "✓ Non-root user created"

USER appuser

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD ["python", "/app/healthcheck.py"]

# ---- Runtime command ----
ENV BLENDER_BIN=/blender/blender

CMD ["python", "-u", "runpod_handler.py"]

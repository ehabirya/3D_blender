FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl xz-utils \
    libx11-6 libxi6 libxrender1 libxfixes3 libxrandr2 libxxf86vm1 \
    libgl1 libglu1-mesa libxkbcommon0 libxkbcommon-x11-0 \
  && rm -rf /var/lib/apt/lists/*

ARG BLENDER_VERSION=3.6.9
RUN curl -L -o /tmp/bl.t.xz https://download.blender.org/release/Blender3.6/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
 && mkdir -p /blender && tar -xJf /tmp/bl.t.xz -C /blender --strip-components=1 && rm /tmp/bl.t.xz

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENV PATH="/blender:${PATH}"
CMD ["python","-u","/app/runpod_handler.py"]

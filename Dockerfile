# Crash&Learn Grand Prix — multi-stage Docker image
#
# Stages:
#   base  — CPU-only, headless.  Used by: sim, train, tensorboard, viz.
#   gpu   — base + GL + pyglet.  Used by: viz-gpu (opt-in, requires nvidia-docker + DISPLAY).
#
# NOTE: f1tenth's setup.py is NOT used — it pins gym==0.19.0 and numpy<=1.22.
# Only the engine modules (f110_gym/envs/) are imported; deps are installed here.

FROM python:3.11-slim AS base

WORKDIR /app

# Build tools for numba + Pillow native extensions; tkinter for optional TkAgg
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps — loose pins, confirmed working on numpy 2.x + numba 0.65
RUN pip install --no-cache-dir \
        "numpy>=2.0" \
        "numba>=0.65" \
        "scipy>=1.7" \
        "Pillow>=9" \
        "pyyaml>=5.3" \
        "stable-baselines3>=2.0" \
        "gymnasium>=0.26" \
        "onnx" \
        "onnxruntime" \
        "imageio>=2.28" \
        "imageio-ffmpeg" \
        "matplotlib>=3.7" \
        "tensorboard"

COPY . /app

# Engine modules are imported by path; PYTHONPATH points at the gym/ subdirectory
ENV PYTHONPATH=/app/gym
# Default headless matplotlib backend — can be overridden at runtime for TkAgg
ENV MPLBACKEND=Agg

# ---------------------------------------------------------------------------
# gpu stage — extends base with OpenGL libraries and pyglet for viz-gpu
# ---------------------------------------------------------------------------
FROM base AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "pyglet<1.5" "PyOpenGL" "PyOpenGL_accelerate"

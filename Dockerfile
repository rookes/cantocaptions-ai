# cantocaptions-ai base image: CUDA 12.8 runtime + ffmpeg + the pipeline installed
# via uv. This image is meant to be used directly (CLI) or extended by the web
# worker (see the cantocaptions-web repo), which adds FastAPI/RQ on top.
#
# Notes:
#  * Linux x86_64 GPU target. Torch is pulled from the CUDA 12.8 index (pyproject).
#  * Extras installed: transformers_qwen (ASR) + ensemble + llm. Diarization needs no
#    extra: pyannote-audio is a base dependency. The
#    flash-attn extra is intentionally omitted — it compiles from source (needs the
#    full CUDA toolkit and many minutes) for little gain here, and the default
#    attention implementation is 'sdpa'. Add it later if you benchmark a win.
#  * Model weights are NOT baked by default (they need an HF token for the gated
#    pyannote diarization model and add ~6 GB). Enable with --build-arg PREFETCH=full
#    and --build-arg HF_TOKEN=hf_xxx, or mount a warm HF cache volume at runtime.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv (pinned copy of the standalone binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Resolve/lock layer first for better caching: copy only metadata, then sync.
COPY pyproject.toml uv.lock ./
# Bring in the rest of the project (package code, vendored assets, config).
COPY . .

# Install the pipeline + server-relevant extras into a project venv.
RUN uv sync --frozen \
        --extra transformers_qwen \
        --extra ensemble \
        --extra llm

# Optional: bake model weights into the image at build time.
ARG PREFETCH=none
ARG HF_TOKEN=""
RUN if [ "$PREFETCH" != "none" ]; then \
        HF_TOKEN="$HF_TOKEN" uv run python scripts/download_models.py \
            $([ "$PREFETCH" = "full" ] && echo --full) ; \
    fi

# Default: run the CLI. The web worker image overrides this CMD.
ENTRYPOINT ["uv", "run", "cantocaptions"]
CMD ["--help"]

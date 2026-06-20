# cantocaptions-ai

An end-to-end speech pipeline for generating high-quality, written Cantonese subtitles.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [ffmpeg](https://ffmpeg.org/) on your system PATH
- NVIDIA GPU with CUDA 12.8 and ≥ 8 GB VRAM (recommended), or run on CPU / Apple Silicon MPS

## Installation

```bash
git clone --recurse-submodules https://github.com/rookes/cantocaptions-ai
cd cantocaptions-ai
uv sync
```

`uv sync` installs all dependencies into an isolated virtual environment and pins exact versions. Torch is pulled from the PyTorch CUDA 12.8 index on Linux and Windows; the CPU build is used on macOS.

## Usage

```bash
cantocaptions audio.wav
```

This produces `audio.srt` in the current directory. The first run downloads model weights automatically (~6 GB).

### HuggingFace access token

The VAD model (`pyannote/segmentation`) requires accepting its terms of use on HuggingFace. Pass your token once:

```bash
cantocaptions audio.wav --hf_token hf_...
```

### Output directory

```bash
cantocaptions audio.wav -o ./subtitles
```

## Optional features

Install extras alongside the base package:

```bash
uv sync --extra ensemble   # Whisper ensemble correction (faster-whisper)
uv sync --extra llm        # LLM particle correction (bitsandbytes)
uv sync --extra diarize    # Speaker diarization — Linux only (NeMo)
uv sync --extra full       # All of the above
```

Enable at runtime:

```bash
cantocaptions audio.wav --ensemble_model whisper
cantocaptions audio.wav --llm_correction
cantocaptions audio.wav --diarize          # Linux only
```

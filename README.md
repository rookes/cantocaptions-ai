# cantocaptions-ai

An end-to-end speech pipeline for generating high-quality, timed written Cantonese (粵文) subtitles.

*This repository is currently in an early stage of development, and some modules may not yet be fully functional.*

## Prerequisites

- Python 3.10, 3.11, or 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [ffmpeg](https://ffmpeg.org/) on your system PATH
- NVIDIA GPU with CUDA 12.8 and ≥ 8 GB VRAM (recommended), or run on CPU / Apple Silicon MPS

## Installation

```bash
git clone https://github.com/rookes/cantocaptions-ai
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

### Custom options

* `-o [DIR_NAME]` - output directory for the SRT file
* `--vocal_isolation_method [OPTION]` - set to "none" for no vocal isolation, or "mbroformer" for full mbroformer vocal isolation
* `--log_file [FILE_PATH]` - simplify console logging and output full logs to designated file
* `--debug_dir [DIR_NAME]` - directory for intermediate processed data for debugging purposes
* `--load_debug_dir [DIR_NAME]` - load previously generated `--debug_dir` data from this directory to skip processing steps (such as VAD, vocal isolation, and transcription)

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

## Planned Updates

Current updates planned for the near future:

* Add Cantonese standardization and cleaning scripts (adapted from [rookes/canto-subtitle-cleaner](https://github.com/rookes/canto-subtitle-cleaner))
* Check for certain characters that are poorly-handled by Qwen3-ASR (i.e. "喎")
* Improve ensemble+LLM integration to allow for more consistent and error-free transcriptions
* Complete diarization implementation to separate lines from different speakers
* Add [SubER](https://github.com/apptek/SubER) metric calculation compatibility, and use its Levenshtein distance algorithm to parallelize ensemble subs
* Add some subtitle processing helper utilities, such as an SRT retiming utility

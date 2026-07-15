# cantocaptions-ai

An end-to-end speech pipeline for generating high-quality, timed written Cantonese (粵文) subtitles. 

*This repository is currently in an early stage of development. Please be patient as new features are rolled out.*

## How it Works

You can use this tool via command line to generate a subtitle file (default format: SRT) for a given Cantonese audio/video file.

This project is modeled after the [WhisperX ASR library](https://github.com/m-bain/whisperx), and shares some of the same [basic architecture](https://raw.githubusercontent.com/m-bain/whisperX/refs/heads/main/figures/pipeline.png). However, `cantocaptions_ai` uses Alibaba Cloud's [Qwen3-ASR models](https://github.com/QwenLM/Qwen3-ASR) for the transcription step, alvanlii's [wav2vec2-BERT-Cantonese model](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese) for the alignment step, and adds a wide array of subtitling improvements designed specifically for written Cantonese.

This library is currently designed to run locally on consumer hardware. I may implement some optional LLM API usage in the future, but for now the goal is to provide users with fully open access to generate their own Cantonese subtitles.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [ffmpeg](https://ffmpeg.org/) on your system PATH
- NVIDIA GPU with CUDA 12.8 and ≥ 8 GB VRAM (recommended), or run on CPU / Apple Silicon MPS

## Installation

```bash
git clone https://github.com/rookes/cantocaptions-ai
cd cantocaptions-ai
uv sync --extra transformers_qwen
```

This installs all dependencies plus the recommended ASR backend into an isolated virtual environment and pins exact versions. Torch is pulled from the PyTorch CUDA 12.8 index on Linux and Windows; the CPU build is used on macOS. 

Note that bare `uv sync` does **not** install a working ASR backend. You need to pick one explicitly:

```bash
uv sync --extra transformers_qwen   # ASR via official transformers Qwen3-ASR support (recommended)
uv sync --extra legacy              # ASR via the older qwen_asr package; mutually exclusive with transformers_qwen
```

## Usage

```bash
uv run cantocaptions_ai audio.wav
```

This produces `audio.srt` in the current directory. Note that the first run will take a while, as it downloads model weights automatically (~6 GB).

### HuggingFace access token

The VAD model (`pyannote/segmentation`) may require accepting its terms of use on HuggingFace. Pass your token once if necessary:

```bash
uv run cantocaptions_ai audio.wav --hf_token hf_...
```

### Custom options

You can update default command line arguments by editing the file `config/default.cfg`. Additionally, you can run using any config file's arguments by using its filename with the `--cfg` option (e.g. `--cfg cpu` to use the configuration in `config/cpu.cfg`).

* `--help` - show command line arguments and syntax
* `-o [DIR_NAME]` - output directory for the SRT file
* `--input_dir [DIR_NAME]` - run all 
* `--vocal_isolation_method [OPTION]` - set to "none" for no vocal isolation, or "mbroformer" for full mbroformer vocal isolation
* `--log_file [FILE_PATH]` - simplify console logging and output full logs to designated file
* `--debug_dir [DIR_NAME]` - directory for intermediate processed data for debugging purposes
* `--load_debug_dir [DIR_NAME]` - load previously generated `--debug_dir` data from this directory to skip processing steps (such as VAD, vocal isolation, and transcription)

## Planned Updates

Current updates planned for the near future:

- [x] Add Cantonese standardization and cleaning scripts (adapted from [rookes/canto-subtitle-cleaner](https://github.com/rookes/canto-subtitle-cleaner))
- [x] Add [SubER](https://github.com/apptek/SubER) metric calculation compatibility, and use its Levenshtein distance algorithm to parallelize ensemble subs
- [ ] Add more performant options for vocal isolation
- [ ] Implement the "retime" feature to accurately run alignment on existing subtitles (IN PROGRESS)
- [ ] Add an option to use Qwen LLM to do error-correction based on a reference standard Chinese subtitle file (IN PROGRESS)
- [ ] Check for certain characters that are poorly-handled by Qwen3-ASR (i.e. "喎")
- [ ] Add better multilingual recognition for Mandarin and English
- [ ] Complete diarization implementation to separate lines from different speakers

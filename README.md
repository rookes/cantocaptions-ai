# cantocaptions-ai

An end-to-end speech pipeline for generating high-quality, timed written Cantonese (粵文) subtitles. 

## How it Works

You can use this tool via command line to generate a subtitle file (default format: SRT) for a given Cantonese audio/video file. There are also several other modes for text alignment, diarizatin, and more.

This project is modeled after the [WhisperX ASR library](https://github.com/m-bain/whisperx), and shares some of the same [basic architecture](https://raw.githubusercontent.com/m-bain/whisperX/refs/heads/main/figures/pipeline.png). However, `cantocaptions_ai` uses Alibaba Cloud's [Qwen3-ASR models](https://github.com/QwenLM/Qwen3-ASR) for the transcription step, alvanlii's [wav2vec2-BERT-Cantonese model](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese) for the alignment step, and adds a wide array of subtitling improvements designed specifically for written Cantonese.

`cantocaptions-ai` is currently designed to run locally on consumer hardware. For the time being, the goal is to provide users with fully open access to generate their own Cantonese subtitles, so no LLM APIs are queried. Once you download the model weights, you can run this completely offline.

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

### Aligning an existing transcript

If you already have the words and only need the timings, you can skip ASR entirely:

```bash
uv run cantocaptions movie.mp4 --realign transcript.txt
```

`transcript.txt` is line-delimited — one subtitle cue per line, no timestamps. Those line
breaks are treated as the authoritative cue boundaries, so the output has one cue per line
(interjection-only lines aside, which the cleaning rules drop). Text cleaning and the
acoustic particle spot-checks (喇/啦, 呀/啊/吖, 咁/噉) run as they do on ASR output.

An SRT works too, in which case its timings are discarded and re-derived from scratch. That
is the difference from `--retime`, which *keeps* a subtitle's timings and nudges them, and so
needs one that is already roughly in sync.

* `--realign_anchor asr` — transcribe first and match the two texts, instead of searching
  acoustically. Slower, but it can leave a line unmatched rather than forcing it somewhere,
  which is what you want if the transcript may contain lines the recording does not.
* `--realign_min_score [SCORE]` — report lines with weak acoustic support. This detects a
  transcript that disagrees with the recording; it is **not** a check that the timings are
  right (see `CLAUDE.md`).
* `--audio_downmix center` — on a 5.1 source, align against the front-centre channel alone,
  which is largely the dialogue stem.

Measure a change to any of this with `scripts/eval_realign.py`, which strips the timings off a
known-good SRT, realigns its text, and reports how far each cue landed from where it belongs.

### Custom options

You can update default command line arguments by editing the file `config/default.cfg`. Additionally, you can run using any config file's arguments by using its filename with the `--cfg` option (e.g. `--cfg cpu` to use the configuration in `config/cpu.cfg`).

* `--help` - show command line arguments and syntax
* `-o [DIR_NAME]` - output directory for the SRT file
* `--input_dir [DIR_NAME]` - run all 
* `--vocal_isolation_method [OPTION]` - "mbroformer" enables Mel-Band RoFormer vocal isolation (helps on noisy/music-heavy audio); defaults to "none" as it is a heavy stage for little gain on clean speech
* `--audio_start [SECONDS]` / `--audio_end [SECONDS]` - transcribe only a clip of the input; output timestamps map back to the source timeline
* `--log_file [FILE_PATH]` - simplify console logging and output full logs to designated file
* `--debug_dir [DIR_NAME]` - directory for intermediate processed data for debugging purposes
* `--load_debug_dir [DIR_NAME]` - load previously generated `--debug_dir` data from this directory to skip processing steps (such as VAD, vocal isolation, and transcription)

## Planned Updates

Current updates planned for the near future:

- [x] Add Cantonese standardization and cleaning scripts (adapted from [rookes/canto-subtitle-cleaner](https://github.com/rookes/canto-subtitle-cleaner))
- [x] Add [SubER](https://github.com/apptek/SubER) metric calculation compatibility, and use its Levenshtein distance algorithm to parallelize ensemble subs
- [ ] Add more performant options for vocal isolation
- [x] Implement the "realign" feature to run alignment on an existing untimed transcript
- [x] Add error-correction based on a reference standard Chinese subtitle file
- [x] Check for certain characters that are poorly-handled by Qwen3-ASR (i.e. "喎")
- [ ] Add better multilingual recognition for Mandarin and English
- [x] Complete diarization implementation to separate lines from different speakers

# cantocaptions-ai

An end-to-end speech pipeline for generating high-quality, timed written Cantonese (粵文) subtitles. Generate subtitles from any 
Cantonese audio or video file. Runs locally on consumer hardware. Works entirely offline once models have been downloaded, so no API
tokens or web-based LLM usage required.

This project is modeled after the [WhisperX ASR library](https://github.com/m-bain/whisperx), and
shares some of the same [basic architecture](https://raw.githubusercontent.com/m-bain/whisperX/refs/heads/main/figures/pipeline.png).
However, `cantocaptions_ai` uses rookes's [cantocaptions-cantonese-asr model](https://huggingface.co/rookes/cantocaptions-cantonese-asr) 
for the transcription step, alvanlii's [wav2vec2-BERT-Cantonese model](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese) for the
alignment step, and adds a wide array of other subtitling improvements designed specifically for written
Cantonese. The target written Cantonese standard is the [CantoCaptions standard](https://cantocaptions.com).

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

This installs all dependencies plus the recommended ASR backend into an isolated virtual
environment and pins exact versions. Torch is pulled from the PyTorch CUDA 12.8 index on Linux and
Windows; the CPU build is used on macOS.

Note that bare `uv sync` does **not** install a working ASR backend. You need to pick one
explicitly:

```bash
uv sync --extra transformers_qwen   # ASR via official transformers Qwen3-ASR support (recommended)
uv sync --extra legacy              # ASR via the older qwen_asr package; mutually exclusive with transformers_qwen
```

## Basic Usage

To generate a subtitle file easily for a given audio or video file, run:

```bash
uv run cantocaptions_ai video.mkv
uv run cantocaptions_ai audio.wav
```

This will write `<your-file-name>.srt` into the `output/`. The first run also downloads the model 
weights (~6 GB), so it will take significantly longer than the ones after it.

Any file ffmpeg can read works: `.mkv`, `.mp4`, `.wav`, `.m4a` and so on. Add `--input_dir DIR` to
process a whole folder.

You can configure more extensively using command-line flags (see below), but, more conveniently, you can also 
**set your own defaults in `config/default.cfg`**. Any command line flags you add will override these defaults 
at runtime. Use the flag `--cfg NAME` to swap in a different file from `config/` (for example `--cfg cpu` to use 
the defaults from `config/cpu.cfg`).

Run `uv run cantocaptions_ai --help` to display the complete flag list.

_Note: If you are running out of VRAM when running, it's recommended to lower `batch_size` and `align_batch_size`._

## Configuration

### ASR Model selection

`model` picks the transcription model. The default, `cantocaptions-cantonese-ASR`, is
[a fine-tune of Qwen3-ASR](https://huggingface.co/rookes/cantocaptions-cantonese-asr) trained on the
CantoCaptions dataset. It already writes HK-style written Cantonese and even distinguishes sentence-final 
particles by tone (e.g. 啦 laa1 / 喇 laa3), so less post-processing is necessary to clean things up. This is
currently the best model by far for this framework, so it is recommended to keep as-is.

Alternatives are the stock `Qwen3-ASR` (1.7B) and `Qwen3-ASR-0.6B`. In order to avoid simplified Chinesee and 
non-standard written Cantonese, both of these models are put through extensive post-processing when used. 
Post-processing includes using the alignment model as a phonetic guide to check for certain variants 
such as gam2 噉 vs. gam3 咁.

### Speech detection (VAD)

Before transcribing, the pipeline finds where the speech in the audio is and cuts it into chunks. Three 
important settings to adjust if there are issues with dropped speech:

- `chunk_size` — the maximum length of an audio chunk to transcribe. The input audio is split into
  chunks based on this size.
- `vad_onset` / `vad_offset` — the detection thresholds. Lower them if speech is being missed entirely.
  Raise to detect less as speech and speed up inference.
- `vad_pad_onset` — audio kept before each detected region. It is deliberately large, because
  the detector's own onset lags about a second behind real speech after a silence. Raise it if the
  first word of lines is being clipped.

### Vocal isolation

`vocal_isolation_method = mbroformer` runs the audio through a Mel-Band RoFormer separator and
transcribes the isolated vocals. Improves subtitle quality significantly, but is very slow. Requires
~600 MB download on first use. Off by default.

### Alignment

To get an accurate timing for the subtitles, an alignment model is used (`no_align = True` to skip the timing step). 
By default, the model used is [alvinlii's wav2vec2-BERT model for Cantonese](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese).

## Post-Processing / Text Cleaning

After alignment, subtitles are split and re-merged to maintain output standards. Line segmentation can be manipulated with:

* `min_cue_duration` — the shortest subtitle allowed before it is merged into a neighbour
* `max_line_width` / `max_line_count` (default: 18 / 2) — control forced line breaks and how text is wrapped

More extensive text processing, such as OpenCC simplified -> traditional options and regex substitutions, are configurable via .toml
files in `cantocaptions-ai/cantonese`.

### Speaker separation (diarization)

Set `diarize = True` to attempt to check the speaker for each cue. By default, this will only be used to 
stop one subtitle from being used for two different speakers' dialogue. Lower `speaker_confidence` to split
more eagerly. Add `speaker_labels = True` if you also want each line prefixed with `[SPEAKER_00]:` 
(note: speaker identification is currently highly inaccurate).

Diarization requires a gated model download, so you need to accept its terms on HuggingFace and supply a 
token (see below).

### Debugging

`debug_dir` is off by default. Set it to a directory (e.g. `--debug_dir temp` for `./temp/`) and each 
stage's output will be saved there as it runs. Useful for testing multiple runs with different settings.
You can set `load_debug_dir` to the same directory to reload a previous run's data (if it exists) without having
to recompute anything.

Note that `debug_dir` also outputs segmented audio, so it can grow large quickly.

`--log_file FILE` keeps the console output brief and writes the full log to a file.

### HuggingFace access token

Some models — the diarization model, and possibly the VAD model — require accepting their terms of
use on HuggingFace. Pass a token once if you hit that:

```bash
uv run cantocaptions_ai video.mkv --hf_token hf_...
```

You can also set the `HF_TOKEN` environment variable. Prefer either over putting the token in
`config/default.cfg`, which is tracked by git.

To fetch the model weights ahead of time rather than on first run:

```bash
uv run python scripts/download_models.py
```

## Realign Feature

If you already have a transcript and only need the timings, you can skip ASR entirely:

```bash
uv run cantocaptions_ai movie.mp4 --realign transcript.txt
```

`transcript.txt` is line-delimited. One subtitle cue per line, no timestamps. Those line breaks are
treated as the authoritative cue boundaries, so the output has one cue per line. Text cleaning and 
the acoustic particle spot-checks (喇/啦, 呀/啊/吖, 咁/噉) run as they normally would for ASR output.

### Retiming a subtitle

Pass a subtitle that already has timings and `--realign` will keep them as a starting point instead
of discarding them:

```bash
uv run cantocaptions_ai test.mkv --realign misaligned_subtitle_file.srt
```

TThis process finds the lines it is confident about acoustically, then fits the simplest transform between 
the two timelines, and realigns using that transform rather than manually aligning every subtitle to the audio. 
The transform can express an offset, a speed difference (e.g. a PAL broadcast runs ~4.3% fast against its 23.976 fps master),
and cuts and insertions where one release has content the other does not, all of which it reports:

```
realign: 956 cue(s) mapped through 3 transform piece(s) from 848 anchor(s) (5 rejected)
realign: SPEED CHANGE of 1.0434x (close to a 25 -> 23.976 fps conversion)
realign: insertion of 2.8s at source 00:09:42 -- this recording has audio the subtitle does not cover
realign: cues moved by a median of +65.98s (largest +127.61s)
```

Text cleaning is **off** by default for a subtitle input, although punctuation is still normalized. 
Use `--realign_normalize False` to turn off all realign normalization.

### Other realign options

* `--realign_mode transcript` — ignore the input's timings after all and place every line from
  scratch.
* `--realign_mode adjust` — fit the same map, then re-time each cue from the audio within
  `--realign_adjust_tolerance` of it. Slower, and it does not preserve the input's proportions.
* `--realign_cut_policy keep` — where the recording is missing content the subtitle covers, keep
  those cues (collapsed onto the cut and flagged) instead of dropping them.
* `--realign_anchor asr` — transcribe first and match the two texts, instead of searching
  acoustically. Slower, but it can leave a line unmatched rather than forcing it somewhere, which is
  what you want if the transcript may contain lines the recording does not.
* `--realign_min_score SCORE` — report lines with weak acoustic support. This detects a transcript
  that disagrees with the recording; it is **not** a check that the timings are right.
* `--audio_downmix center` — on a 5.1 source, align against the front-centre channel alone, which is
  largely the dialogue stem.

With `--debug_dir`, `realign/transform.json` records every piece, every edit and where each cue moved
from and to, and `realign/changes.srt` holds just the cues that did something other than shift with
the rest of the file.

## Additional Features

* Measure a change to realignment with `scripts/eval_realign.py`, which strips the timings off a
known-good SRT, realigns its text, and reports how far each cue landed from where it belongs.
* Set `HF_XET_HIGH_PERFORMANCE=1` to trade RAM/CPU for more model download speed

Thank you to everyone from the CantoCaptions community and Discord for their support and testing on this project.

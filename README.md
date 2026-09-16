# cantocaptions-ai

An end-to-end speech pipeline for generating high-quality, timed written Cantonese (粵文) subtitles.

You point it at a Cantonese audio or video file and it writes a subtitle file. It runs locally on
consumer hardware — no APIs are queried, and once the model weights are downloaded it works
entirely offline.

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

## Usage

```bash
uv run cantocaptions_ai video.mkv
```

That is the whole thing. It writes `video.srt` into the `output/` directory. The first run also
downloads the model weights it needs (~6 GB), so it will take noticeably longer than the ones
after it.

Any file ffmpeg can read works — `.mkv`, `.mp4`, `.wav`, `.m4a` and so on. Add `--input_dir DIR` to
process a whole folder.

### Configuration

Everything below is a command-line flag, but you rarely want to retype flags.
**`config/default.cfg` is the file to edit.** It lists every setting with a short comment, grouped
by what it affects, and its values apply to every run. Anything you type on the command line still
wins over it, and `--cfg NAME` swaps in a different file from `config/` (for example `--cfg cpu`).

Run `uv run cantocaptions_ai --help` for the complete flag list.

The single most useful setting is `batch_size`. It is the main VRAM lever — if a run fails with an
out-of-memory error, lower it before changing anything else.

### 1. Model selection

`model` picks the transcriber. The default, `cantocaptions-cantonese-ASR`, is
[a fine-tune of Qwen3-ASR](https://huggingface.co/rookes/cantocaptions-cantonese-asr) trained on the
CantoCaptions dataset: it already writes HK-traditional Cantonese and distinguishes the
sentence-final particles by tone, so none of the rewriting rules are applied on top of it. The
alternatives are the stock `Qwen3-ASR` (1.7B) and `Qwen3-ASR-0.6B`, whose Mandarin-flavoured,
Simplified-derived output is put through the full cleaning chain instead. Prefer the default unless
you are comparing against a baseline.

### 2. Speech detection (VAD)

Before transcribing, the pipeline finds where the speech is and cuts it into chunks. Three settings
matter if it is getting that wrong:

- `chunk_size` (28 s) — the longest piece handed to the ASR model. Also a VRAM lever.
- `vad_onset` / `vad_offset` (0.45 / 0.30) — the detection thresholds. **Lower them if quiet or sung
  speech is being missed entirely**; raise them if music and effects are being picked up as dialogue.
- `vad_pad_onset` (1.0 s) — audio kept before each detected region. It is deliberately large, because
  the detector's own onset lags about a second behind real speech after a silence. Raise it if the
  first word of lines is being clipped.

### 3. Vocal isolation

`vocal_isolation_method = mbroformer` runs the audio through a Mel-Band RoFormer separator and
transcribes the isolated vocals. It is worth it on music-heavy or noisy sources — film and TV with a
score underneath the dialogue — and not worth it on clean speech, where it adds a slow stage and a
~600 MB download for very little. Off by default.

### 4. Alignment

Alignment is what puts each character on the timeline, and it runs by default. `no_align = True`
skips it and falls back to the ASR model's own rough timings: much faster, much less accurate, and
only sensible when you want a transcript rather than a subtitle.

The settings worth knowing are `min_cue_duration` (0.5 s), the shortest subtitle allowed before it
is merged into a neighbour, and `max_line_width` / `max_line_count` (18 / 2), which control how the
text is wrapped on screen.

### 5. Speaker separation (diarization)

`diarize = True` works out who is speaking. Its main job is to stop one subtitle spanning two
people — by default it changes where cues are split and nothing else. Add `speaker_labels = True` if
you also want each line prefixed with `[SPEAKER_00]:`.

This downloads a gated model, so you need to accept its terms on HuggingFace and supply a token (see
below). `speaker_confidence` (0.7) is the dial: lower it to split more eagerly.

### 6. Debugging

`debug_dir` is off by default. Set it to a directory (`temp`, say) and each stage's output is
saved there as it runs. Its real use is `load_debug_dir`: point that at a previous run's
`debug_dir` and the expensive stages (VAD, vocal isolation, transcription) are replayed from disk
instead of recomputed, so you can iterate on the later ones in seconds. It writes the segmented
audio too, so expect it to grow large.

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

### Faster downloads

Nothing to set up — `hf-xet` comes in automatically with `huggingface_hub`, so model
downloads are already Xet-backed (chunk-level deduplication, parallel transfer) on any
mainstream CPU architecture. `scripts/download_models.py` prints whether it is active:

```
[info] Xet: on (set HF_XET_HIGH_PERFORMANCE=1 to trade RAM/CPU for more speed)
```

If that line says `OFF`, either you are on an architecture with no `hf-xet` wheel or
`HF_HUB_DISABLE_XET` is set; downloads still work, just over plain HTTP. Setting
`HF_XET_HIGH_PERFORMANCE=1` raises throughput further at the cost of more RAM and CPU.

## Aligning an existing transcript

If you already have the words and only need the timings, you can skip ASR entirely:

```bash
uv run cantocaptions_ai movie.mp4 --realign transcript.txt
```

`transcript.txt` is line-delimited — one subtitle cue per line, no timestamps. Those line breaks are
treated as the authoritative cue boundaries, so the output has one cue per line (interjection-only
lines aside, which the cleaning rules drop). Text cleaning and the acoustic particle spot-checks
(喇/啦, 呀/啊/吖, 咁/噉) run as they do on ASR output.

### Retiming a subtitle onto a different release

Pass a subtitle that already has timings and `--realign` will keep them as a starting point instead
of discarding them:

```bash
uv run cantocaptions_ai bluray.mkv --realign broadcast.srt
```

This is the case where a subtitle was timed against one release and you want it on another — a
different broadcast, a Blu-ray, a version with the adverts cut out. It finds the lines it is
confident about acoustically, fits the simplest map between the two timelines, and moves every cue
through it, so the subtitle's own rhythm survives exactly. The map can express an offset, a speed
difference (a PAL broadcast runs ~4.3% fast against its 23.976 fps master), and cuts and insertions
where one release has content the other does not — all of which it reports:

```
realign: 956 cue(s) mapped through 3 transform piece(s) from 848 anchor(s) (5 rejected)
realign: SPEED CHANGE of 1.0434x (close to a 25 -> 23.976 fps conversion)
realign: insertion of 2.8s at source 00:09:42 -- this recording has audio the subtitle does not cover
realign: cues moved by a median of +65.98s (largest +127.61s)
```

Text cleaning is **off** for a subtitle input — it is already a finished subtitle, so its wording is
left alone — and on for a bare transcript. Punctuation is still normalized either way, which matters
because a halfwidth mark beside Chinese text is in neither the align vocabulary nor the pause
tokens, so the pause it stands for would otherwise go unmodelled. `--realign_normalize False` turns
even that off.

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
the rest of the file — load it beside the video and step through them.

## How it works

This project is modeled after the [WhisperX ASR library](https://github.com/m-bain/whisperx), and
shares some of the same
[basic architecture](https://raw.githubusercontent.com/m-bain/whisperX/refs/heads/main/figures/pipeline.png).
However, `cantocaptions_ai` uses Alibaba Cloud's
[Qwen3-ASR models](https://github.com/QwenLM/Qwen3-ASR) for the transcription step, alvanlii's
[wav2vec2-BERT-Cantonese model](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese) for the
alignment step, and adds a wide array of subtitling improvements designed specifically for written
Cantonese.

Stages run sequentially, and models are loaded and unloaded between them so the whole pipeline fits
in limited VRAM. `CLAUDE.md` documents each stage in depth, including what was measured and why the
defaults are what they are.

Measure a change to realignment with `scripts/eval_realign.py`, which strips the timings off a
known-good SRT, realigns its text, and reports how far each cue landed from where it belongs.

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

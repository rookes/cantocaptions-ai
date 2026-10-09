# cantocaptions-ai

An end-to-end speech pipeline for generating high-quality, timed written Cantonese (粵文) subtitles. Point to any audio or video file
with Cantonese speech to generate timed captions. Runs locally on consumer hardware, so no need for API tokens or web-based LLM usage.

This project is modeled after the [WhisperX ASR library](https://github.com/m-bain/whisperx), and
shares some of the same [basic architecture](https://raw.githubusercontent.com/m-bain/whisperX/refs/heads/main/figures/pipeline.png).
However, `cantocaptions_ai` uses rookes's [cantocaptions-cantonese-asr model](https://huggingface.co/rookes/cantocaptions-cantonese-asr) 
for the transcription step, alvanlii's [wav2vec2-BERT-Cantonese model](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese) for the
alignment step, and adds a wide array of other subtitling improvements designed specifically for written
Cantonese. The target written Cantonese standard is the [CantoCaptions standard](https://cantocaptions.com).

An optional proofreading pass through a hosted LLM is available, but it is off unless you turn it on
(see [Online proofreading](#online-proofreading-opt-in)); nothing else in the pipeline needs the network once
the models are downloaded.

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

This installs all dependencies into an isolated virtual environment and pins exact versions; every ASR
family (Qwen3-ASR, Whisper, CTC) runs on it. Torch is pulled from the PyTorch CUDA 12.8 index on Linux
and Windows; the CPU build is used on macOS. Optional extras: `compile` (triton, for `--compile`),
`ensemble`, `llm`, `flash-attn`, `proofread` (the hosted-LLM clients for `--proofread`), or `full` for all of them. `transformers_qwen` is the old name of
`compile` and still works.

## Basic Usage

To generate a subtitle file easily for a given audio or video file, run:

```bash
uv run cantocaptions_ai video.mkv
uv run cantocaptions_ai audio.wav
```

This will write `<your-file-name>.srt` into the `output/`. The first run also downloads the model 
weights (~6 GB), so it will take significantly longer than the ones after it.

Any file ffmpeg can read works: `.mkv`, `.mp4`, `.wav`, `.m4a` and so on. Add `--input_dir DIR` to
process a whole folder (and `--recursive` to include its subfolders). Output mirrors the input
tree, so `DIR/s1/ep01.mkv` and `DIR/s2/ep01.mkv` become `output/s1/ep01.srt` and `output/s2/ep01.srt`.

You can configure more extensively using command-line flags (see below), but, more conveniently, you can also
**put your own defaults in `user.cfg`** in your config directory (`config/` in the repo, which git
ignores; `~/.config/cantocaptions-ai/` for a pip install; or wherever `$CANTOCAPTIONS_CONFIG_DIR` points).
It has the same format as the shipped [`default.cfg`](cantocaptions_ai/presets/default.cfg), holding only
the keys you want to change, either in one `[pipeline]` block or grouped by section (`[vad]`, `[alignment]`,
... the same groups as `--help`). Settings are layered, each overriding the one before:

1. built-in defaults
2. `default.cfg` (shipped with the package; documents the defaults), or the preset picked with
   `--cfg NAME` (for example `--cfg cpu`, `--cfg fast_test`). A file of the same name in your config
   directory takes its place
3. `user.cfg`
4. the `--vocal_isolation` / `--asr` / `--align` presets
5. flags you type

`language` defaults to `yue`, the one language with a full language pack: it alone picks the ASR model,
alignment model, cleaning rules and subtitle conventions, so nothing else needs setting. Another language
runs as a raw pipeline -- transcribed, aligned, split into cues and laid out with that language's
conventions (spaces between words, Latin punctuation, word-wrapped lines), but not cleaned -- and needs you
to supply what its pack lacks: a `--model` that transcribes it, an `--align_model` if there is no built-in
default for it (most languages have one), and `--no_clean_text`. The error message names whatever is
missing. To add a language properly, see [docs/adding-a-language.md](docs/adding-a-language.md).

For video files, the audio track is chosen from the stream tags by language (for `yue`: a Cantonese
track, then any Chinese one). Pass `--audio_track N` when a release's tags are missing or wrong.

Run `uv run cantocaptions_ai --help` to display the complete flag list.

**Tips:**

* If you are running out of VRAM, it's recommended to lower `batch_size` and `align_batch_size`
* If you are running out of system RAM on a long `--input_dir` run, lower `files_per_group` (default 10). Each file's decoded audio is held until its subtitles are written, about 0.23 GB per hour of audio
* If you want to improve subtitle quality and don't mind extra compute time, turn on vocal isolation with `--vocal_isolation_method mbroformer`

## Configuration

### ASR Model selection

`model` picks the transcription model; left unset, the language's own is used. For Cantonese that is
`cantocaptions-cantonese-ASR`,
[a fine-tune of Qwen3-ASR](https://huggingface.co/rookes/cantocaptions-cantonese-asr) trained on the
CantoCaptions dataset. It already writes HK-style written Cantonese and even distinguishes sentence-final 
particles by tone (e.g. 啦 laa1 / 喇 laa3), so less post-processing is necessary to clean things up. This is
currently the best model by far for this framework, so it is recommended to keep as-is.

Alternatives are the stock `Qwen3-ASR` (1.7B) and `Qwen3-ASR-0.6B`. In order to avoid simplified Chinese and 
non-standard written Cantonese, both of these models are put through extensive post-processing when used. 
Post-processing includes using the alignment model as a phonetic guide to check for certain variants 
such as gam2 噉 vs. gam3 咁.

`model` also takes any Hugging Face hub id or local checkpoint path, and the ASR backend follows from the
checkpoint's own `config.json`:

| Family | Examples | Notes |
|---|---|---|
| Qwen3-ASR | the models above | The only backend that takes `--asr_context`. |
| Whisper | `whisper-large-v3`, `whisper-large-v3-turbo`, or any Whisper checkpoint | Forced to transcribe in `--language`. Under Cantonese it gets the same post-processing as stock Qwen3-ASR. It punctuates little, so the end of each phrase it times is written as a comma. |
| CTC (wav2vec2 family) | `alvanlii/wav2vec2-BERT-cantonese`; wav2vec2, wav2vec2-BERT, HuBERT, WavLM checkpoints | Decoded greedily. These models write no punctuation, so each pause of 0.3 s or more is written as a comma. |

Those commas are what give the pipeline somewhere to break cues; without them, each speech chunk (up to 28 s)
would be a single cue.

Only the fine-tune is tuned for Cantonese subtitles; the others are there for other languages, and for comparison.

### Speech detection (VAD)

Before transcribing, the pipeline finds where the speech in the audio is and cuts it into chunks. Three 
important settings to adjust if there are issues with dropped speech:

- `chunk_size` — the maximum length of an audio chunk to transcribe. The input audio is split into
  chunks based on this size.
- `vad_onset` / `vad_offset` — the detection thresholds. Lower them if speech is being missed entirely.
  Raise to detect less as speech and speed up inference.
- `vad_pad_onset` — audio kept before each detected region. Raise it if the first word of lines
  is being clipped.

`vad_method` picks the detector:

- `pyannote` (default) — pyannote segmentation-3.0. Fast on a GPU (~2.5 s per hour of audio) and
  the most accurate option.
- `silero` — Silero VAD v6, a small model built for CPUs (~23 s per hour of audio on two cores,
  where pyannote on a CPU takes ~38 s across every core). `--cfg cpu` uses it. It drops somewhat
  more speech than pyannote.

Both models' scores go through the same thresholds, padding and chunking, but they are calibrated
differently: the `vad_*` values in the default config are tuned for pyannote and the ones in
the `cpu` preset (`cantocaptions_ai/presets/cpu.cfg`) for silero, so copy the whole set when switching
models.

### Vocal isolation

`vocal_isolation_method = mbroformer` runs the audio through a Mel-Band RoFormer separator and
transcribes the isolated vocals. Improves subtitle quality significantly, but is very slow. Requires
~600 MB download on first use. Off by default.

On CUDA the separator is compiled with `torch.compile`, which roughly halves its time once
compiled. Compiling costs about 12 s per process (about 50 s the first time on a machine), so with
the default `vocal_isolation_compile = auto` it happens only when there's enough audio to repay
it (about 13 minutes of speech), or when the process has compiled it already. `on` and `off`
force it either way. It needs Triton: included on Linux; on Windows install the `compile` extra.
Without it, isolation runs uncompiled.

The separator works on overlapping 8 s windows, and pads each speech segment with reflected
audio so its edges get the same overlap. `vocal_isolation_span_gap` (default 1.0 s) isolates
segments that close together as one stretch instead, with the real audio between them as
context, which removes about 13% of the work at no measurable cost in accuracy. `0` isolates
every segment on its own.

### Alignment

To get an accurate timing for the subtitles, an alignment model is used (`no_align = True` to skip the timing step). 
By default, the model used is [alvanlii's wav2vec2-BERT model for Cantonese](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese).

### Post-Processing / Text Cleaning

After alignment, subtitles are split and re-merged to maintain output standards. Line segmentation can be manipulated with:

* `min_cue_duration` — the shortest subtitle allowed before it is merged into a neighbour
* `max_cue_duration` (default: 4) — the longest a subtitle may run: neighbours are not merged past it, and a
  longer cue (usually a clause the ASR never punctuated) is cut at its longest internal pause. `0` turns it off
* `max_line_width` / `max_line_count` (default: the language's width / 2) — control forced line breaks and how text is
  wrapped. Unset, the width is the language's own: 18 characters for Chinese, 42 for space-separated scripts. `0` never
  breaks a line

More extensive text processing, such as OpenCC simplified -> traditional options and regex substitutions, are configurable via .toml
files in `cantocaptions_ai/languages/yue/rules/` (point `--clean_rules_dir` at a copy to use your own).

### Speaker separation (diarization)

Set `diarize = True` to attempt to check the speaker for each cue. By default, this will only be used to 
stop one subtitle from being used for two different speakers' dialogue. Lower `speaker_confidence` to split
more eagerly. Add `speaker_labels = True` if you also want each line prefixed with `[SPEAKER_00]:` 
(note: speaker identification is currently highly inaccurate).
See [docs/diarization.md](docs/diarization.md) for measured accuracy and the scripts that score it.

For keeping two speakers out of one subtitle, `speaker_change = True` works better than `diarize`. It scores
each boundary between clauses for a change of voice directly, and on the eval episodes it catches about twice as
many speaker changes as diarization labels, for a few seconds per episode. It uses the same gated model, and either
or both can be on. `speaker_change_threshold` (default 0.8) trades splits for recall.

Diarization requires a gated model download, so you need to accept its terms on HuggingFace and supply a 
token (see below).

### Online proofreading (opt-in)

The finished subtitles can be sent to a hosted LLM (Gemini or Claude) for a proofreading pass. This is
the one stage that needs the network and costs money per run, so it is **never on by default**: you
enable it per run with `--proofread gemini`, or make it your own default in `config/user.cfg`:

```ini
[proofreading]
proofread = gemini
```

It needs the client libraries (`uv sync --inexact --extra proofread`; `--inexact` keeps whatever other extras you installed) and an API key in the environment,
`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`. Keys are read from the environment only; do not put them in a
config file.

```bash
uv run cantocaptions_ai episode.mkv --proofread gemini --reference_subtitle episode.chi.srt
```

The proofreader works best with `--reference_subtitle`: a same-content subtitle in another language (for
Cantonese, usually a Standard Chinese track) is interleaved with the cues by time, which is what lets it
recover names and mishearings the ASR could not. It is told to correct what was *said*, never to copy the
reference's wording. Its answer is structured (name decisions, per-cue edits, doubts) and is applied in code:
edits that would drift into a foreign register (Standard Written Chinese forms such as 的/這/沒 for Cantonese)
or that the model marks below `proofread_min_confidence` are refused and listed for review instead, and every
edited cue goes back through text cleaning. Cue timings and boundaries never change: the model may not move
words from one cue to another, and a pair of edits that does is listed for review rather than applied.

Sentence-final particles (啦/喇, 吖/啊, 㗎/𠿪 …) are left alone by default. Which one was said is a matter of
tone, which a text-only model cannot hear, and in testing its particle changes were wrong several times as
often as right. The prompt tells it not to change them, and any change it proposes anyway is listed in
`flags.srt` instead of being applied. `proofread_particles = allow` lifts this.

Measured on Cantonese episodes with a human-made reference transcript and a Standard Chinese reference
subtitle, using Gemini 3.7 Flash at medium effort: it removed roughly a quarter of the remaining character
errors, with about nine correct edits for every wrong one, at around 10-15 US cents and three to four minutes
per 25-minute episode. Keep `proofread_effort` at `medium`: at `high` a whole episode had not come back after
25 minutes. Results without a reference subtitle, and on other languages, have not been measured.

To proofread a subtitle you already have, with no audio and no ASR pass, give it as `--proofread_input`
instead of a media file. The reference subtitle is optional:

```bash
uv run cantocaptions_ai --proofread_input episode.srt --proofread gemini
uv run cantocaptions_ai --proofread_input episode.srt --proofread gemini --reference_subtitle episode.chi.srt
```

This writes `episode.proofread.srt` to `output_dir` and never overwrites the input. Cue timings, and every cue
the model leaves alone, come out exactly as they went in; edited cues are re-cleaned as in a normal run (add
`--no_clean_text` to apply the model's text as is). Several files can be given at once, but a reference
subtitle only with a single file. Without a reference, the model works from the text alone, so expect it to
catch fewer mishearings and names.

Useful settings:

* `proofread_max_cost` (default 1.0 USD) — refuse any request estimated above this. `None` removes it.
* `proofread_dry_run = True` — write the exact request and its cost estimate, send nothing.
* `proofread_preclean` (default True) — before proofreading an existing subtitle, apply only the basic cleaning:
  punctuation and the standard character variants, nothing that rewrites words or layout. The lines it changed go to
  `precleaned.srt` beside the review files.
* `proofread_chunk_cues N` — split a file into requests of N cues (0, the default, sends it whole), each shown
  `proofread_chunk_context` (6) read-only cues of its neighbours. The first chunk goes alone and the rest
  `proofread_parallel` (4) at a time, sharing one cached copy of the system prompt; a chunk whose request fails
  is skipped and the others still apply. Name fixes are still applied across the whole file.
* `proofread_context FILE` — a short paragraph about the show (setting, main characters), if you have one.
* `proofread_standard`, `proofread_conventions FILE`, `proofread_prompt FILE` — pick a different writing
  standard, or replace its conventions or the whole prompt template, without touching code. The CantoCaptions
  conventions ship at `cantocaptions_ai/languages/yue/proofread/cantocaptions.md`.

Every answer is saved, since it was paid for: under `debug_dir` when one is set (`{debug_dir}/{name}/proofread/`),
otherwise beside the output in `{output_dir}/{name}.proofread/`. There, `changes.srt` shows each edited cue over a
`[was]` line with its previous text, the changed characters coloured in both, `flags.srt` lists lines the model doubted but did not change (and particle or register changes it
was not allowed to make), for checking against the audio, and `summary.json` holds the edits, names and cost. A
`load_debug_dir` replay reuses a saved answer whenever the request is unchanged, so it is never billed twice.

Beside the proofread subtitle itself goes a Subtitle Edit bookmarks file (`episode.proofread.srt.SE.bookmarks`),
which Subtitle Edit loads on its own when it opens the subtitle: every changed line is bookmarked with
`[was] <old text>`, every flagged line with `[flag] <note>`. An existing bookmarks file is never overwritten; the
new one is named `episode.proofread (1).srt.SE.bookmarks` instead (rename the subtitle to match to load it). The bookmarks always
match the subtitle they sit beside: after a realign that proofread first, they are carried onto the realigned
output's own lines, whatever cues realign split, merged or dropped.

### Debugging

`debug_dir` is off by default. Set it to a directory (e.g. `--debug_dir temp` for `./temp/`) and each 
stage's output will be saved there as it runs. Useful for testing multiple runs with different settings.
You can set `load_debug_dir` to the same directory to reload a previous run's data (if it exists) without having
to recompute anything. Each stage's checkpoint records the settings that produced it (`meta.json`), so a stage
whose settings changed since, or whose inputs came from a stage that changed, is recomputed rather than replayed;
the log names the settings that differ.

Note that `debug_dir` also outputs segmented audio, so it can grow large quickly.

### Console output and the log file

The console shows one section per stage: a header with the time it started, a spinner or progress bar
that says what the stage is doing right now, and a line with its duration when it finishes. Every run
also writes a detailed log, with timestamps, every step's debug lines and the warnings that libraries
print, to `{output_dir}/logs/{input}-{date}.log`. The last console line gives its path.

- `--verbose` shows that detail on the console too.
- `--log_file FILE` writes the log somewhere else; `--log_file none` writes none.
- `--log_level warning` keeps the console to warnings and errors.

### HuggingFace access token

Some models — the diarization model, and possibly the VAD model — require accepting their terms of
use on HuggingFace. Pass a token once if you hit that:

```bash
uv run cantocaptions_ai video.mkv --hf_token hf_...
```

You can also set the `HF_TOKEN` environment variable. Prefer either over putting the token in a
config file; if you must, use your untracked `user.cfg`, never a shipped preset.

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

This process finds the lines it is confident about acoustically, then fits the simplest transform between 
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

### Proofreading while realigning

`--realign` and `--proofread` combine: a human-edited subtitle can be proofread against a reference and then put
on another release's timeline in one run. The reference subtitle has to be on the *same* timeline as the cues
it is compared with, so say which one it follows with `--reference_timing`:

```bash
# reference timed like the subtitle being realigned: proofread first, then realign the corrected copy
uv run cantocaptions_ai bluray.mkv --realign episode.srt --proofread gemini     --reference_subtitle episode.chi.srt --reference_timing subtitle

# reference timed to bluray.mkv: realign first, then proofread the realigned cues
uv run cantocaptions_ai bluray.mkv --realign episode.srt --proofread gemini     --reference_subtitle bluray.chi.srt --reference_timing media
```

`--reference_timing` defaults to `media`, so the second form needs no flag. With `subtitle`, the
corrected copy is also written as `episode.proofread.srt`, and its review files go to `episode.proofread/`
(or the debug dir). Without a reference, proofreading runs first. Add `--proofread_min_confidence medium` (or
`high`) to apply only the model's confident changes and list the rest as flags.

Every proofread run also checks that the reference really does share the cues' timeline: a matching reference
has 85-95% of its cues starting within half a second of a cue, and a shifted one, or one from another release,
falls to under half. Below 60% nothing is sent, and the error names any constant `--reference_offset` that
would line it up.

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

## Development

Run the test suite (CPU only; no model downloads):

```bash
uv sync --group dev
uv run pytest
```

`tests/test_pipeline_e2e.py` runs the whole pipeline with fake models (`tests/_pipeline_fakes.py`:
synthetic audio that encodes its own transcript, so alignment is exact) and compares the result to
`tests/golden/`. After a deliberate output change, regenerate the goldens with
`UPDATE_GOLDEN=1 uv run pytest tests/test_pipeline_e2e.py` and review the diff.

For a check on real models and real audio, `scripts/build_eval_episodes.py` stitches
cantocaptions-dataset test-split clips into episodes with an exact reference SRT, and
`scripts/score_subtitles.py` scores runs against it (CER, cue coverage, start error) or diffs two runs.

### Architecture

The pipeline runs a list of stages over every input file: VAD → vocal isolation (optional) → ASR →
ensemble / LLM correction (optional) → forced alignment → diarization (optional), then cue assembly, text
cleaning and the writers. The stages are objects in `cantocaptions_ai/pipeline/stages.py`; `build_stages`
picks the ones a config needs, and `_execute_pipeline` (`pipeline/transcribe.py`) runs them in order after
logging the plan, e.g. `Pipeline: VAD [cached] → Transcription [compute] → Alignment`. A model stage with a
debug checkpoint (`CachedStage`) computes, or reads every file back from `--load_debug_dir` when all of them
have a current checkpoint. Each model itself is a `PipelineStage` (`utils/model_utils.py`).

To add a stage, subclass `Stage` (`run(ctx, items) -> items`, plus `active(ctx)` if it is optional) or
`CachedStage` (`compute` and `from_cache`, plus a `checkpoint` key in `utils/checkpoints.py`), and add it
to `DEFAULT_STAGES` where it belongs. A caller can also pass `_execute_pipeline(..., stages=fn)`, where
`fn(ctx, default_stages)` returns the list to run, to insert or replace one without editing the package.

Everything that depends on the language is gathered into one **language pack** per language
(`cantocaptions_ai/languages/`): how it is written (`ScriptConfig`, `PunctuationConfig` from
`text_profiles.py`), its default ASR and alignment models, how each ASR model writes it (normalization,
particle spot-checks, cue markers, cleaning manifest), its cleaning rules and builtin steps, its audio-track
preference, and its LLM correction prompts and ensemble model. `get_language_pack(code).resolve(model)` gives
the stages what they need; a language with no pack gets a generic one, enough to run raw. The Cantonese pack
is `languages/yue/`; the old `cantocaptions_ai.cantonese` import paths still work.

Behaviour that depends on a particular model rather than a language is looked up per model:

* `pipeline/model_profiles.py` — per ASR model: where its weights are, which backend runs it, and which
  languages it is trained for. The backends (Qwen3-ASR, Whisper, CTC) are registered in `pipeline/asr.py`
  (`ASR_BACKENDS`); an unregistered model's backend is read from its checkpoint's `model_type`.
* `pipeline/align_profiles.py` — per alignment model: audio primer, hand-picked character substitutions
  and the default substitution level (homophones only for the Cantonese model), internal-gap
  splitting, minimum input length. The model's own processor and emission frame rate come with it from
  `load_align_model`.
* `pipeline/align_backends.py` — how each family of alignment model is loaded and run: any Hugging Face CTC
  checkpoint (batched), or a torchaudio pipeline bundle by name. The family is read from the model itself.

Substituting a character the align model has no token for (its variant form, or a homophone) needs to know
how the language sounds, so that comes from the language pack (`char_readings`; Jyutping for Cantonese).

Text cleaning is a language-independent engine (`cantocaptions_ai/cleaning/`: a manifest of TOML regex rule
files and coded builtin steps) that each pack supplies with rules; cue assembly, alignment, realign and line
layout take the pack's script and punctuation as arguments.

Thank you to everyone from the CantoCaptions community and Discord for their support and testing on this project.

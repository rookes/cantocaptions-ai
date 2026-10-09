# Cue assembly and speaker attribution

Design notes and measurements for `pipeline/segmentation.py`, `diarize.py` and `speaker_assign.py`. For diarization *accuracy*, see [diarization.md](diarization.md).

## Cue assembly (pipeline/segmentation.py)

Alignment deliberately over-splits: it cuts each ASR chunk at every punctuation mark so the Viterbi pass can place a pause at each clause boundary. `segmentation.py:assemble_cues()` — called from `transcribe.py:_merge_and_write()` — glues those fragments back into displayable cues in four passes:

| Pass | What it does | Key knobs |
|---|---|---|
| A. Adjacency merge | joins touching neighbours whose join boundary is punctuation-clean | `--align_merge_distance`, `--align_padding`, `MAX_CHARS` (one line) |
| B. Noise drop | drops sub-threshold cues whose *cleaned* text is `is_removable` | `--min_cue_duration` |
| C. Short-cue rescue | merges each too-short cue into the neighbour that reads best | `--min_cue_duration`, `--merge_gap`, `max_line_width × max_line_count` |
| D. Duration floor | extends whatever is left into the following silence | `--min_cue_duration`, `--align_padding` |

Passes B–D exist because CTC forced alignment is *peaky*: a discourse marker inside a continuous speech run ("嗱，", "喂，") gets only the frames where its label was emitted, so it surfaces as a 40–80 ms cue wedged one `align_padding` from its neighbours. The fix is to merge it, not retime it.

Pass C picks its direction **punctuation first, direction second, distance last**: a join whose boundary character reads cleanly wins even if it is the farther neighbour; among equally clean joins, a cue listed in the profile's `leading_markers` joins *forwards* onto the clause it introduces. Pass A defers those markers rather than absorbing them backwards, since it accumulates strictly left to right and would otherwise decide before pass C could weigh both sides. Unlike pass A, pass C does **not** treat `mergeable_chars` as a gate — rescuing a fragment stranded behind a full stop is the whole point.

**Which side a short fragment joins is decided by the *spoken* pause, not the cue-edge gap.**
`_sounded_bounds` takes a cue's first and last timed non-punctuation token, and `_sounded_pause`
measures the silence between two cues' sounded edges (falling back to the edge gap when either
side has no timed token). Cue edges are a poor proxy: punctuation mapped to blank dwells across
a pause and `align_padding`/release decisions move the edge, so a vocative or a short reply could
look closer to the wrong neighbour. Pass A defers a short cue forwards when the pause after it is
shorter than the pause before it by more than `FORWARD_PAUSE_MARGIN` (0.05 s) -- as well as for
`leading_markers` -- and pass C ranks candidates on the same measure. Measured on a ~900-cue
episode, it changed two cues, both towards the reference segmentation.

Compare `merge_segments()` in `utils/schema.py`, which does the actual joining and preserves keys later stages attach (notably `speaker`).

Text cleaning runs inside `transcribe.py:_merge_and_write()` on the assembled cues, just before the writer. It edits segment `text` only (word/char timings are left as alignment ground truth), drops noise-only subtitles (`is_removable`), and is skipped under `--realign_mode sync` and `adjust`, whose input is already a finished subtitle. When `--debug_dir` is set, a pre-cleaning SRT snapshot is written to `{debug_dir}/{stem}/pre_cleaning/` (not a loadable checkpoint — cleaning always re-runs so rule edits take effect during `--load_debug_dir` replay).

## Speaker attribution (pipeline/diarize.py + speaker_assign.py)

Diarization runs as **stage 5, after alignment**, so its speaker turns land on the same
over-split subsegments cue assembly is about to glue back together. Its only job in the
pipeline is to stop `assemble_cues()` merging two lines spoken by different people.

**Scope (`--diarize_scope`, default `segment`).** The merge gate only ever compares *adjacent*
cues, so the only question it asks is "is the voice in these two neighbouring clauses the same
one?" — a local question. `segment` scope therefore diarizes each VAD segment independently
(on the segment's own audio, already vocal-isolated when isolation ran) rather than clustering
the whole episode into a fixed global speaker set, whose errors otherwise spread across every
cue. `--diarize_scope file` restores the whole-file pass for comparison.

Because per-segment identities are established independently, labels are namespaced with the
VAD segment id — `S0007/SPEAKER_00` — and `_same_speaker` compares the scope *first*: two labels
from different segments are treated as "unknown" and permit the merge, since there is no
evidence either way. Vetoes therefore only ever happen inside one VAD segment. Segments shorter
than `MIN_SEGMENT_DURATION` (0.5 s) are skipped rather than guessed at, and `--min_speakers` /
`--max_speakers` apply *per segment* under this scope.

**VRAM: `--diarize_batch_size` (default 4).** The checkpoint's own `config.yaml` asks for
`embedding_batch_size: 32`, and that does not fit a consumer card. Measured on an RTX 3080,
one 28 s segment:

| batch | time | peak VRAM |
|---|---|---|
| 32 (model default) | ~60 s | 10,735 MB |
| 16 | 45 s | 10,002 MB |
| 8 | 585 s | 9,352 MB |
| **4** | **1.09 s** | **251 MB** |

The jump between 4 and 8 is a kernel-workspace threshold, not smooth scaling, so 4 sits on the
safe side of a cliff. Over budget, Windows does not raise -- WDDM pages the oversubscribed
allocation into host RAM, so diarization silently runs ~50x slower with the GPU pegged. Whole
file, 403 s of audio: 5.9 s (segment scope) / 8.1 s (file scope) at batch 4.

Two traps this leaves behind:

- **`torch.cuda.mem_get_info()` does not see it.** It reported 5.3 GB free while Task Manager
  showed dedicated memory full. Judge fit by `max_memory_allocated` (the in-call peak), which
  `--log_level debug` prints per segment -- never by `free`.
- **Live allocation is flat** (42 MB between segments). Diarization accumulates nothing across
  calls, so `empty_cache()` between them reclaims nothing; `SegmentDiarization.flush_every`
  defaults to 0. Only the in-call peak matters.

Segments still run longest-first, which costs nothing and makes the largest peak the first
thing attempted, so an over-budget batch size shows up immediately rather than midway through.

The two halves are deliberately separate:

- `diarize.py` runs `pyannote/speaker-diarization-community-1` at the chosen scope and
  returns speaker turns on the file timeline. It is checkpointed
  (`diarization/result.json`), so a `--load_debug_dir` replay never reloads the model.
- `speaker_assign.py` maps those turns onto subsegments and is the only place a *decision*
  is made. It always re-runs, so threshold edits take effect on a replay.

Attribution is by **overlap share**: intersect a subsegment with the turns, normalise each
speaker's overlap by the subsegment's *diarized* (not wall-clock) time, and label it only
when the leading speaker clears `--speaker_confidence` (0.7). Anything more ambiguous is
left **unlabeled**, which `segmentation._same_speaker` reads as "no objection to merging" —
the pre-diarization behaviour. That asymmetry is the whole design: a wrong label permanently
splits a sentence, a missing one costs nothing.

The runner-up's share is the experimental multi-speaker signal. Under
`--flag_speaker_conflicts`, a subsegment whose second speaker clears
`--speaker_conflict_share` (0.25) gets `speaker_conflict`, surfaced only in the diarization
debug output — it is diagnostic input for deciding whether splitting cues on diarization is
worth pursuing, and changes no timings or text.

Every boundary the gate actually holds is logged at INFO by `segmentation.py`, one line per
boundary plus a count:

```
Diarization held a cue boundary at 00:00:02,040: [SPEAKER_00] 你好嗎， | [SPEAKER_01] 幾好呀，
Diarization kept 1 cue boundary from merging
```

A boundary is only reported when diarization is the *sole* reason the merge did not happen —
punctuation-blocked joins and fragments that pass C rescued in the other direction stay
silent — so the log reads as "loosen `--speaker_confidence` and these would join".

Speaker labels stay **out of the subtitle text** unless `--speaker_labels` is passed
(`utils/output.py:_with_speaker`); `--diarize` alone changes cue boundaries, nothing else.

`merge_segments()` (`utils/schema.py`) takes the label from whichever side has one, since a
merge is only ever permitted between sides that don't contradict each other.


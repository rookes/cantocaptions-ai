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

**Widths, not character counts.** The line budgets in passes A and C (and the line breaker in
`cleaning/layout.py`) measure text with `ScriptConfig.width`. Under CJK an English letter, digit
or space counts half a Chinese character, so `Considerate, thoughtful` (22 characters, 11 wide)
fits one line. A Chinese-only line measures exactly its length, so Cantonese output is unchanged.
The long-cue split (pass E, `max_cue_duration`) cuts only between `words`, and alignment keeps
an English word as one (see [alignment.md](alignment.md#code-switched-english-pipelinealignmentpy)).
It also never cuts inside a Chinese word: `ScriptConfig.inside_word` asks the language pack
(`languages/yue/words.py`), so 樂觀 and 結果 stay whole. pycantonese's segmentation decides,
with one correction. It glues what it doesn't know into long tokens (演奏一首, 外觀同符合,
生蠔王豆沙包大), and trusting those left a long cue nowhere to cut but the wrong place. So only a
token of up to three characters is one word. Inside a longer one, a cut is protected only where
a word from its own dictionary (HKCanCor plus rime-cantonese) spans it: 演奏|一首 may be cut,
豆沙|包 may not. The dictionary is never consulted *across* the segmenter's boundaries, where it
over-protects: 現有 ("existing") is a word, but 我對佢嘅表現有啲失望 is 表現 / 有啲. The same
check guards `split_gapped_cues` and realign's pause split.

**Pass E leaves a clause read slowly alone.** A cue whose text fits on one line is long only
because it is spoken slowly, so pass E cuts it only at punctuation or at an aligned pause of
`ONE_LINE_MIN_PAUSE` (0.5 s). Alignment can't show a pause inside a clause anyway: CTC gives
each character a span reaching the next, so a learner's 一諗起下個月去旅行, read over 5.3 s,
reads 0 at every gap and was cut mid-word at 下個|月. Measured (2026-10) on bluey and Doraemon
against their references, the one-line cuts the references agree with sat at aligned gaps of
1.2 and 2.0 s, and the ones they disagree with (美景將可|以望見得到, 大亂中嘅|不倒金威) at 0.0.
Start error within 0.5 s is 78.5 % with the rule and 78.7 % without. The one agreed cut it loses
sits at `、`, which pass E doesn't treat as punctuation (Tension、|拖鞋).

**Text that fits the screen is cut only at a confident pause.** Pass E has three tiers by the
width of the cue's text:

| text | cut at |
|---|---|
| fits one line | punctuation, or an aligned gap of `ONE_LINE_MIN_PAUSE` (0.5 s) |
| fits the screen (`max_line_width × max_line_count`) | punctuation, or the longest pause of `CONFIDENT_PAUSE` (0.25 s); otherwise shown whole, over two lines |
| longer | wherever is least bad, as before; a cut can't be avoided |

A pause is the aligned gap or the extra time a word was held (`_held`), whichever is longer.
CTC folds a pause inside a clause into the earlier character's span, so the hold is where it
shows. In bluey's 跟住落嚟我將會演奏一首好出名嘅土耳其進行曲啊 every character runs about
0.16 s onset to onset and 奏 runs 0.48 s; the reference cuts after 演奏, and that is now the cut.
Before, with every gap at 0, pass E took the middlemost boundary (好|出名).

Swept (2026-10) on bluey and Doraemon against their references, with and without vocal
isolation (start / end within 0.5 s, both files pooled):

| | isolated | not isolated |
|---|---|---|
| before | 78.5 % / 77.1 % | 79.5 % / 77.1 % |
| `CONFIDENT_PAUSE` 0.20 or 0.25 s | 78.9 % / 77.4 % | 79.9 % / 77.4 % |
| 0.35 s | 78.4 % / 76.9 % | 79.4 % / 76.9 % |
| 0.25 s with `max_cue_duration` 4.5 s | 77.9 % / 76.6 % | 79.4 % / 76.9 % |

Every cut that changed in bluey now matches the reference (演奏|一首…, 對唔住啊|啱啱…,
真係成日都|令我…), as does Doraemon's 三冠王|豆沙包. The cap stays at 4 s: the references
do split the cues between 4 and 4.5 s. On a learner's vocabulary video, English sentences with
no confident pause stay one two-line subtitle where they were cut mid-sentence before.

**Every cue is at least `align_padding` from the next** on the ASR path. Alignment's
release/trim pass already does this, but pass E's cut made touching pairs when it cut where
alignment showed no pause (the head ended where the tail began). Pass E now ends the head
`align_padding` early, and `_keep_apart` checks the whole file once more after pass D. A cue too
short to give up the padding is left touching, never overlapping. Under `--realign`
(`merge=False`) the source's own chained cues are kept as they are.

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


# Reference-subtitle context for ASR

Design notes and measurements for `--reference_subtitle` / `--asr_context` (`pipeline/reference_context.py`) and the evaluation harness.


`--reference_subtitle` takes a same-content, different-language cue list (a standard
Chinese or English SRT). It has always fed **LLM correction** (stage 3c); `--asr_context`
additionally routes it into two earlier, independent places. Both are experimental and
off by default.

| Use | Where | Flag |
|---|---|---|
| Widen the VAD timeline over cues VAD missed | stage 1, pre-slice | `--asr_context_vad_expand` (default on) |
| Bias the ASR decode with the cues covering each segment | stage 3, prompt | `--asr_context` |
| Restrict that bias to the audio expansion recovered | stage 3, prompt | `--asr_context_scope expanded` |

**Scope.** `--asr_context_scope expanded` prompts only where the reference actually told
the pipeline something new. Stage 1 records, per segment, the sub-spans that exist solely
because a cue was unioned in (`VadAudioSegment['expanded']`, from
`expansion_only_spans()`); stage 3 then keeps only the cues overlapping one of them. A
segment VAD detected unaided decodes with no context at all.

That provenance is the one piece of segment state that is **persisted** through the VAD
and vocal-isolation manifests (`debug._PERSISTED_SEGMENT_KEYS`) rather than re-derived
like `context` is — it cannot be recomputed downstream, because by then the
pre-expansion timeline no longer exists. Note the padding asymmetry this creates: on the
Doraemon fixture 44 of 67 segments carry expansion provenance but only 23 earn a prompt,
because in the other 21 only `--asr_context_padding` spilled past the VAD boundary and no
cue *body* sits in the recovered span. That is the intended reading — nothing was
recovered there, so there is nothing to tell the model.

**VAD expansion is an inclusive OR.** `expand_intervals_to_reference()` runs inside
`VadProcessor.process()` between `merge_chunks` and the audio-slicing loop, so recovered
regions are ordinary VAD segments downstream and land in the VAD debug checkpoint. **Every**
cue is widened by `--asr_context_padding` (0.5 s) and unioned with the VAD regions — a span
survives if *either* source claims it. There is no confidence gate: the reference is a
human record of where dialogue is, and a cue VAD agrees with costs nothing to include
twice, while a cue VAD missed is exactly the unrecoverable speech.

> An earlier version gated cues on an absolute "uncovered seconds" tolerance. That is
> scale-blind — a 2 s cue can be 46% missing and still sit under a 1 s threshold — and it
> silently dropped the one line that motivated the whole feature. Do not reintroduce it.

`--chunk_size` is then honoured **unconditionally**; capping is the caller's decision.
`_split_one` chooses *where* to cut, in order of how little it damages:

1. a point covered by neither a VAD region nor a cue — pure padding, severs nothing;
2. a gap between two consecutive cues — no dialogue line spans it, even where VAD hears
   music or effects;
3. the lowest-scoring VAD frame in the window, penalising frames inside a cue so an
   equally quiet frame outside a dialogue line wins. This is `Binarize._split_long`'s
   min-cut idea; `vad.py` passes the raw probability curve in for it. Tier 3 logs a
   **warning** with the score at the cut, so a genuinely through-speech cut is visible.

The split window is `[start + chunk_size/2, start + chunk_size]`, giving the same
"every piece is at least half the budget" guarantee `Binarize` does. On both fixtures tier 3 never
fired — every cut landed in padding or a cue gap.

Two invariants the output must keep: **sorted and disjoint** (alignment's
`_find_vad_segment_idx` returns the *first* segment containing a timestamp, so overlaps
orphan the second one's words), and VAD's own *touching* boundaries — `Binarize._split_long`
min-cuts at the quietest frame — are reinstated wherever no cue crosses them, so the union
never re-segments stretches the reference did not touch.

**ASR context.** Qwen3-ASR reads free-form text in the *system* prompt as background
knowledge and tilts decoding toward it ("context biasing"). The effect is soft — a
probability nudge, not a constraint. `build_segment_contexts()` gives each VAD segment
the reference cues that overlap it (widen with `--asr_context_neighbours`), rendered
through `--asr_context_template` and capped at `--asr_context_max_chars` (400), since
context is a prefill cost paid per segment per batch and inflates the KV-cache estimate
in `_asr_native._warn_vram`.

Contexts are attached in `transcribe.py` *after* stage 2, not at VAD time: both the VAD
and vocal-isolation debug round-trips rebuild segment dicts from scratch and would drop
the key. They are cheap and always re-derived, so template edits take effect on a
`--load_debug_dir` replay — the same pattern as speaker assignment and text cleaning.

`overlapping_reference_indices()` is the single time-matcher; `llm_correction`'s
`match_reference_to_segments` is a thin wrapper over it (defaults preserve its original
`'，'` joiner and 2 s nearest-cue fallback, both wrong for 30 s VAD chunks).

**Measure it, do not assume it.** Qwen publishes no guidance on context format or length,
and the technical report gives no numbers for context biasing. Use
`scripts/eval_asr_context.py` (see "Evaluating ASR context" below).

## Evaluating ASR context

`scripts/eval_asr_context.py` A/Bs the experimental context feature against a hand-corrected
ground truth:

```bash
uv run python scripts/eval_asr_context.py     --audio experiments/fixtures/bluey/bluey_test.wav     --reference experiments/fixtures/bluey/bluey_standardchinese.srt     --groundtruth experiments/fixtures/bluey/bluey_groundtruth.srt --diff
uv run python scripts/eval_asr_context.py ... --sweep template
```

Two things the harness exists to get right:

- **Each variant gets its own `--debug_dir`.** The transcription checkpoint does not record
  whether context was used, so a shared debug dir replays the first variant's ASR output for
  every later one and reports a null result. Non-expanding variants are seeded with the
  baseline's `vad/` + `vocal_isolation/` checkpoints so only ASR re-runs; `context_expand`
  changes the VAD timeline and is never seeded.
- **The headline metric is `docCER`**, not `calculate_cer` — see the `suber.py` note under
  Module Reference.

`--diff` prints, per ground-truth cue, what each variant said over that timespan. Rows are
aligned **by time, not cue index**: variants legitimately produce different cue counts, so
an index-to-index comparison lines up unrelated lines and reads as noise even where the
variants agree. CER alone cannot tell "picked up the proper nouns" from "churned the
particles", and that distinction is the whole decision.

### Measured results (Doraemon 555-556, 2026-08-18)

Full clean rerun: corrected reference subtitle, `--reference_offset -1.0`, VAD expansion on
at defaults (union + 30 s cap, padding 0.5). Nothing cache-seeded.

| variant | docCER | SubER | segs | longest | GT speech missed |
|---|---|---|---|---|---|
| baseline | 16.27 | 50.92 | 61 | 27.9 s | 33.2 s (3.4 %) |
| `none` (VAD only) | 15.44 (−0.82) | 49.76 | 67 | 28.0 s | **1.2 s (0.1 %)** |
| `labelled` + scope `expanded` | **14.67 (−1.60)** | 49.37 | 67 | 28.0 s | 1.2 s (0.1 %) |
| `bare` | 15.08 (−1.18) | **48.79** | 67 | 28.0 s | 1.2 s (0.1 %) |
| `labelled` | **14.77 (−1.49)** | 48.99 | 67 | 28.0 s | 1.2 s (0.1 %) |
| `instruct` | 16.37 (+0.10) | 49.37 | 67 | 28.0 s | 1.2 s (0.1 %) |

**`none` is the control, and it is the row that makes the rest interpretable.** It runs
the VAD expansion and then decodes with no context prompt at all, so the gap between it
and the baseline is the expansion alone, and the gap between it and `labelled` is the
prompt alone. Roughly 55 % of the headline gain is VAD, 45 % is the prompt. Note that
neither half clears the 1-point noise band on its own — only the combination does.

Decomposing the document edit distance shows the two halves are not doing the same job
at all, and is also the whole reason `instruct` is kept only as a negative control:

| variant | sub | ins | del | edits |
|---|---|---|---|---|
| baseline | 281 | 70 | 281 | 632 |
| `none` (VAD only) | **324** | 75 | 201 | 600 |
| `expanded` scope | 306 | **84** | **180** | **570** |
| `bare` | 231 | 125 | 230 | 586 |
| `labelled` | **229** | 127 | 218 | 574 |
| `instruct` | 250 | **175** | 211 | 636 |

- **VAD expansion kills deletions and costs substitutions** (281 → 201 del, 281 → 324
  sub). It recovers speech that was dropped entirely, but the recovered audio is the
  *hard* audio — sung, shouted, music-bedded — and transcribes badly.
- **The prompt kills substitutions and costs insertions** (324 → 229 sub, 75 → 127 ins).
  It is largely cleaning up after the expansion.

So the prompt's real substitution effect is −95 against the correct control, not the
−52 it appears to be against the plain baseline — measuring the prompt against an
unexpanded baseline understates it by nearly half. The two signals are complementary,
not redundant; `--asr_context_template none` exists so that stays checkable.

`instruct` is where the insertion trade stops paying: it has 48 more insertions than
`labelled`, which is essentially the entire 1.6-point gap.

**`--asr_context_scope expanded` reaches the same docCER as `labelled` by the opposite
route** (14.67 vs 14.78 — 4 edits on 3885 characters, pure noise; do not read it as a
win). It prompts only the 23 of 67 segments holding reference-recovered audio, so it
keeps the recall (lowest deletions of any variant, 180) and avoids most of the copying
(84 insertions, a third fewer than `labelled`), while giving up the proper-noun fixes on
the 44 segments VAD found by itself (306 substitutions, near `none`'s 324). The two
differ on 39 segments — `labelled` better on 29 of them, but `expanded`'s 10 wins are
much larger, being the segments where `labelled` recited the reference instead of
transcribing.

The practical read: `expanded` is the *safer* setting and `labelled` the higher-ceiling
one. Choose by which failure costs more — Mandarin bleeding into the output (favour
`expanded`, less than half the marker rate) or proper nouns going unrecovered on audio
VAD already had (favour `labelled`).

**Do not pick a default template from one measurement.** An earlier sweep (old reference,
no expansion) had `bare` ahead of `labelled` 15.11 vs 15.73 and looked like grounds for
changing the default; with the corrected reference the order reverses. The two are close
and fixture-sensitive — `labelled` stays the default.

The `instruct` gap is a different matter and is *not* in the noise band. The three
templates order themselves by how strongly the prompt frames the context as text to use —
`labelled` (參考翻譯：, "reference translation") copies least, `bare` more, and `instruct`
(an English imperative) most — which is the expected failure for a model that treats the
system prompt as background knowledge rather than as instructions.

**Recitation is reduced, not gone.** Capping segments at 30 s removed the runaway *loop*
(`instruct` previously hit `max_new_tokens` repeating one sentence up to 7 times, for
+2.39 docCER): a segment that no longer ends mid-word gives the model nothing to complete
from. Single-shot recitation survives under `instruct` — it emits the context once
alongside the transcription, in either order:

```
seg 26  ctx  : 早安,女士 早…早安 小姐，隨便看，隨便買 對啊，今天天氣很好 這是什麼地方？
        out  : 早安，女士早…早安小姐，隨便看，隨便買，對呀 今天天氣很好，這是什麼地方？  ← context, verbatim
               Bonjour， séniorla唔做做Séniorita，隨便睬，隨便買…            ← then the actual audio
```

**`bare` and `labelled` do it too, just less often.** An earlier pass here claimed they
did not; that was measured with the wrong instrument (longest *contiguous* copied run,
plus adjacent-duplicate cue counts, which are 0 for every variant). Segment 64 is a plain
`labelled` recitation the SRT-level check misses entirely:

```
seg 64  truth    : 歡迎光臨啊 係，而家嘅喺 點解突然品多人　…
        labelled : 歡迎光臨，是，現在來了麻煩你…為什麼突然這麼多客人？…   ← context, in Mandarin
                   點解突然品多人ㄞ？係呢個影像投射鏡…                 ← then the audio
```

Use a Mandarin-marker rate over the whole output as the detector instead — it is cheap and
does not depend on finding a contiguous match:

| variant | Mandarin-marker rate |
|---|---|
| baseline | 0.14 % |
| `expanded` | 0.31 % |
| `labelled` | 0.68 % |
| `bare` | 0.73 % |
| `instruct` | 0.85 % |

Two places to see the effect directly:

```
1212-1221s
  truth    : 荒屋嘅豆沙包平時80円係買唔到㗎 | 喂，有冇搞錯啊？ | 總之非常好味，超值，係日本第一
  baseline : 今日豆沙包平時80英係買唔到㗎   | 喂，有冇搞錯啊？ | 總之，冰爽撈味，超值，人氣第一
  bare     : 荒屋嘅豆沙包平時80円係買唔到㗎 | 喂，有冇搞錯啊？ | 總之非常好味，超值，日本第一

652-690s (胖虎's song — pyannote scores sung voice near zero, so VAD drops it)
  truth    : 大力壯健，乜都咁威 | 我叫做胖虎 | 全心意，有high tension | 我係小霸王 | 喺全世界出道喇 | 胖虎嚟喇
  baseline : 楊三依 | 遊客聽信粗話 | 盤府佢哋真係亂咁嚟啊
  labelled : 大力壯健乜都咁威 | 我叫做胖虎 | 有high tension | 我係小霸王… | 全世界出道啦 | 胖虎嚟喇
```

The first shows context fixing homophones *toward Cantonese* (係買唔到㗎, 好味, 超值) rather than
copying the Mandarin reference. The second is the expansion recovering speech VAD never
detected at all.

Against the `none` control the prompt's contribution is specifically proper nouns and
domain vocabulary — the class of error context biasing is meant to fix (32 of the 46
segments where they differ favour `labelled`):

```
1185-1213s
  truth    : 荒屋嘅和菓子係和菓子嘅三冠王 … 豆沙包只係賡80円咂
  none     : 包裝嘅豆沙包只係黃果子嘅三倍黃 … 豆沙包只係賡80英咂？
  labelled : 荒屋嘅和菓子係和菓子嘅三冠王 … 豆沙包只係賡80円咂

1367-1395s
  truth    : 唔該，我想買金鍔啊 我要豆沙包同埋金鍔
  none     : 唔該，我想買金鱷啊 我要訂沙發同埋金鱷
  labelled : 唔該，我想買金鍔啊 我要豆沙包同埋金鍔
```

**Reference alignment is load-bearing.** The corrected OCR pass moved every cue ~1.13 s later,
which measured as −0.726 s against VAD speech onsets (ground truth: −0.030 s). Left
uncorrected it pads the wrong spans and attaches cues to the wrong segments.
`--reference_offset -1.0` restores it to +0.072 s. Check a new reference this way before
trusting a result from it. If the offset drifts rather than being constant, `--realign_mode sync` fits a transform instead of a single shift.

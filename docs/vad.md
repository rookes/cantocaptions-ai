# VAD segment boundaries

Design notes and measurements for `pipeline/vads/` (`Binarize` lives in `vads/curve.py`).

## Onset lag and the chunk-cap min-cut

A VAD segment occasionally starts (rarely, ends) audibly mid-utterance, with no correlation to
anything a listener would call "the loud part." Confirmed on two unrelated files (a podcast and
a dubbed drama), so neither cause below is content- or format-specific — they are two
independent, unrelated mechanisms, both structural rather than a detection failure.

**Onset lag.** `pyannote.audio.core.inference.Inference` scores the file with a 5 s sliding
window at a 0.5 s step, aggregated with a Hamming taper plus a 0.5 s "warm-up" zeroing at each
window's own edges (`0.1 * duration`). The aggregated speech-probability curve therefore does
not step sharply at a genuine onset following silence — it climbs over roughly 0.9-1.3 s,
measured on both fixtures. With `vad_pad_onset` at the old default (0.2 s), the recorded
segment start lands wherever that ramp happens to cross `vad_onset`, often solidly inside the
utterance. Ruled out before landing on this explanation: GPU nondeterminism (reran identical
audio through the model 3x, byte-identical output), sliding-window phase (re-sliced the same
decoded audio at 7 different sample offsets; the crossing time stayed within 0.3 s regardless),
and file format (same ramp shape and magnitude in a `.webm`/Opus podcast and an `.rmvb` dubbed
drama). `vad_pad_onset` is now **1.0 s** (was 0.2 s) — verified against a captured score curve
to recover the reported case's missing lead-in (463.252 s → 462.452 s, 0.8 s of genuine
near-silence) with no material growth in segment count. `vad_pad_offset` stays at 0.2 s: the
same ramp exists on the trailing edge too, but works in the pipeline's favor there — the
offset crossing happens promptly, while the score is still elevated, so segments do not lose
audible tails the way they lose leading silence.

**Chunk-cap min-cut.** `Binarize._split_long` must cut any region still longer than
`chunk_size`, and searched only `[start + chunk_size/2, start + chunk_size]` (WhisperX's
min-cut) for the lowest-scoring frame. In a genuinely continuous run — a monologue, dense
back-and-forth with no pause anywhere near the midpoint — every candidate in that half is still
deep inside active speech, so the cut is forced with nothing good to pick: not a detection
failure, a hard consequence of budgeting a fixed `chunk_size`. Measured frequency: 9 such
boundaries in an 11-minute file, 5 in a 16-minute one — roughly one every 1-2 minutes of dense
dialogue, though most land somewhere tolerable; only the worst are audible as a bad cut.

`Binarize` now takes `min_split_duration` (default `None`, which preserves the exact old
`max_duration / 2` floor). `Pyannote.merge_chunks` — the path ordinary segmentation runs
through — passes `pad_onset + pad_offset + min_duration_off` instead: the shortest span that
could hold a boundary the rest of the current configuration would already treat as real (a
genuine gap has to clear `min_duration_off` to survive `_smooth` without being bridged; the two
pads are the context the pipeline always attaches to one). This widens the search into the
*first* half of the window too, so a real, deeper dip earlier than the midpoint can be found
instead of a cut with nothing to recommend it over any other frame. `Vad.cover_chunks` (used by
`--realign`) builds its own zero-argument `Binarize` and is untouched by this — its "every
piece is at least half the budget" guarantee must not move; see its docstring.

Measured effect, holding padding constant to isolate this change alone: of the over-long
regions in the two fixtures, 3/6 (Doraemon) and 4/5 (podcast) got a different split at a
measurably lower (more silence-like) score — one region's split score dropped from 0.91 to
0.63, another from 0.30 to 0.23. The rest were unchanged because the old split point was
already the global minimum across the widened search too — there genuinely was no better
option there, which the widened search cannot manufacture and does not pretend to.

**Halving `Inference`'s step size does not sharpen the onset ramp, and this was measured, not
assumed.** The hypothesis: since `Inference` aggregates overlapping 5 s windows at a 0.5 s
step, sampling more of them (`step=0.25`, `step=0.1`) might let the aggregate track the true
onset more precisely, allowing a smaller `vad_pad_onset`. Disproved by calling `Inference.infer()`
directly on a single, unaggregated 5 s chunk (bypassing windowing/aggregation entirely) with
the same transition frame positioned at different offsets within it: at window-start
461.0 (2.7 s of post-transition speech left in the window) the model's own raw confidence for
that exact frame is 0.10-0.13; at window-start 461.5 (3.2 s of post-transition speech) it jumps
to 0.98 — a sharp, near-binary swing **inside a single forward pass**, nothing to do with
aggregation. `PyanNet`'s 4-layer BiLSTM apparently needs roughly 3+ seconds of clearly-speech
content within its own 5 s receptive field before it will commit a frame to "speech" — an
architectural/training property of window *duration*, invariant to `step`. The aggregate curve
is the continuous-limit integral of that fixed confident/unconfident window-start boundary,
Hamming-and-warm-up-weighted; a finer `step` only approximates that same integral with less
quantization noise. Measured on the Doraemon case: onset-crossing time moved by ~0.1-0.25 s
between `step=0.5/0.25/0.1` — noise, not the ~0.8 s this would need to matter — while wall time
for a 300 s clip went 0.26 s → 0.50 s → 1.24 s (linear in window count, as expected). Do not
retry this lever; if onset latency needs to improve further, it would have to come from a
different model or a smaller trained `duration`, not from re-sampling the existing one faster.


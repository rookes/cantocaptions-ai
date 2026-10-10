# Realign

Design notes and measurements for `--realign`: `pipeline/realign.py` and `pipeline/timefit.py`.

## Realigning a transcript (pipeline/realign.py)

`--realign` takes a complete, ordered transcript and puts it on **this** recording's timeline.
One cue per line. The input is either a line-delimited text file (a scraped film script, a
human transcription) or a subtitle that already has timings, and `--realign_mode` decides what
happens to those timings:

| mode | input timings | what places each cue | cost |
|---|---|---|---|
| `transcript` | none, or discarded | the anchor-and-fill search below | search + forced alignment |
| `sync` | the prior | the fitted transform, applied to every cue | search only |
| `adjust` | a prior, trusted ±tolerance | anchor-and-fill over transform brackets, clamped | search + forced alignment |

`auto` (the default) resolves to `transcript` for a bare transcript and `sync` for a subtitle
with timings, testing the file's *content* rather than its extension.

**`sync` never runs forced alignment.** The anchor search is its only acoustic work, so it
skips both `fill()` and stage 4 — measured at 39 s end to end on the 7-minute bluey fixture.
Cue durations come from the source cue scaled by the piece's slope, which is what "preserve
the proportions" means and is verified by test: on bluey every one of 132 cues moved by
exactly +0.074 s and **no cue's duration changed at all**.

**These three modes replace `--retime`, which is removed.** Retime searched a ±5 s window
around each cue's existing timestamp and carried a single rolling offset (`retime.py`), so a
rate difference was both unrepresentable and invisible — and a rate difference is the most
common reason a subtitle does not fit a different release. It also computed the file's
emissions twice, had no model of cuts or insertions, no debug output, no tunables and no
tests, and `--retime --no_align` raised `KeyError: 'time_stamps'`.

**Punctuation normalization is not cleaning, and it runs in every mode.**
`normalize_transcript_text` folds a halfwidth `, . ? ! ;` sitting beside Chinese text into its
fullwidth form and any ellipsis into a single `…`. Punctuation and whitespace only, never the
words. It runs even under `sync`, so "the input's text is preserved" is true of the *words* and
not literally of every byte — on the ReZero fixture it changes 5 of 956 cues, and 0 of 132 on
bluey and 0 of 489 on Doraemon. Each change is a typing slip corrected, and the aligner gets a
real pause token out of it, since a halfwidth mark is in neither the align vocabulary nor
`split_chars` and is otherwise dropped outright. `--realign_normalize False` turns it off and
hands the text through byte for byte, at that cost.

**Measured cost of the opt-out on the ReZero Blu-ray: none.** `--realign_normalize False` gave
956 of 956 cues byte-identical to the input (against 951 with it on) and **timings identical to
the millisecond** — the same 848 anchors, the same 5 rejected, the same three transform pieces.
Do not read that as "the opt-out is free": it is free *here* because only 5 of 956 cues carry a
halfwidth mark at all, so the anchor set could not have moved. On a subtitle punctuated in
halfwidth throughout, every one of those marks stops being a pause the aligner can spend, and
the cost is real.

> **The lookback skips punctuation, and knows about supplementary-plane ideographs.** Both are
> load-bearing and both were bugs. Judging each mark by its single preceding character splits a
> run — `咩話!?` came out `咩話！?`, two widths in one breath — and the same shape turned
> `美少...女` into `美少。..女`, the first dot being the only one with a Chinese character to its
> left. Separately, a range check capped at U+9FFF treats written Cantonese's plane-2
> characters as non-Chinese: 𠸏, 𠻹, 𠺢, 𡃁 appear in 62 cues across the fixtures (𠸏 alone 40
> times). Digits and letters are deliberately *not* skipped, which is what keeps `1.5` a decimal
> and `3,000` a separator inside an otherwise Chinese line, and what keeps the comma in
> "Thank you, sir." English.

**Text cleaning follows the input, not the feature.** `transcript` mode cleans (its input is a
raw transcript, which wants the rule files as much as ASR output does); `sync` and `adjust` do
not (their input is a finished subtitle whose text is the user's). This generalises what
`--retime` used to special-case.

**Intra-cue line breaks are preserved.** A break inside a cue is usually a change of speaker,
so flattening it changes the content rather than the formatting — on the ReZero fixture 4 of
the 7 two-line cues are dashed two-speaker exchanges. The mechanism is one entry:
`REALIGN_PUNCTUATION.split_chars` contains `"\n"`, so `_preprocess_segment` keeps it and
`line_tokens` maps it to the blank id, giving the aligner an explicit pause where the speaker
changes. It cannot split a cue, because `_get_sentence_spans` is never consulted under
`--realign`.

**Splitting a cue where the audio pauses (`--realign_split`).** A subtitler often holds two
sentences on screen as one cue across the pause between them, as two lines or run together
on one. `split_at_pauses` (run after stage-4 alignment, on the final character timings) cuts
a cue where the audio agrees, trying the input's own line breaks first:

* **at a line break** when the gap between the last timed character of one line and the
  first of the next is at least `--realign_split_gap` (0.3 s), and a half of
  `MAX_DETACHED_RUN` characters or fewer is not split off beyond `MAX_INTERNAL_GAP` (the
  "short and far away means misplaced" rule tightening uses);
* **between two characters written together** (`pauses` only) when the second starts at
  least `--realign_pause_gap` (0.7 s) after the first, and each piece keeps
  `PAUSE_SPLIT_MIN_CHARS` (3) timed characters. Nothing written between them (punctuation,
  a space, an untimed character) is allowed: an untimed character would count its own
  duration as pause;
* either way a piece of `SHORT_PIECE_CHARS` (4) timed characters or fewer needs
  `SHORT_PIECE_EXTRA` (0.2 s) more pause, so 0.5 s at a line break and 0.9 s between words. A
  short exchange reads easily as one cue, and a subtitler keeps it whole across a pause that
  would split two full sentences;
* each piece's median `peak` is at least `LINE_SPLIT_MIN_PEAK` (0.5), neither is
  smeared (`_plausible_span`), no line opens with a dialogue dash, and the cue carries no
  `realign_reason`. Each piece is tested again, strongest bare pause first.

The tail starts on its own first character; the head ends the way any cue before a pause
does, released `align_release` past its last character and stopping `align_padding` short
of the tail. `auto` (the default) is `pauses` in `adjust`, whose premise is already that the
input's own timing is not trusted, and `off` elsewhere; `lines` keeps only the first rule.

**Why the onset, not the gap.** Between two characters with nothing written between them
there is no pause token, and this trellis charges a character's held frames at the blank
probability, so the whole silence is folded into the first character's span: the measured
gap between the two spans is 0.00 s at 12 of the 14 bare boundaries the checker cut on
episode 2. The time from one character's onset to the next is what carries it. Two other
measures were tried on the same boundaries and are no better: a blank token inserted between
the characters takes exactly the frames after the first one's onset (its span is the onset
gap within a frame or two), and the longest near-silent run in the isolated vocals is 0.00 s
at several real cuts -- the pause is a breath or room tone, not silence.

**Why three characters a piece.** No acoustic measure here tells a pause from a drawn-out
syllable. On episode 2 every bare boundary the checker kept with an onset gap of 0.4 s or
more had one or two characters on one side: a held syllable before a final particle (好痛|啊,
冷淡|㗎, 魔法|寫) or a short opener (就算|莎緹拉…).

**Measured on two ReZero broadcast episodes**, each realigned in mode `adjust` from a badly
timed copy and scored against a hand-finished one:

| | line breaks | bare pauses ≥ 0.4 s | ≥ 0.5 s | ≥ 0.6 s | ≥ 0.7 s |
|---|---|---|---|---|---|
| ep 2 (to 07:59): cuts the checker also made / cuts made | 7 / 7 | 11 / 11 | 7 / 7 | 6 / 6 | 5 / 5 |
| ep 1 (whole file): same | 1 / 4 → **1 / 1** | 9 / 41 | 6 / 22 | 5 / 15 | 2 / 2 |

The two files disagree about pauses of 0.4-0.6 s inside a sentence and about two-line cues
with a 0.4 s gap: episode 2's checker split those run-ons, episode 1's finished file keeps
them whole. The line-break half of that disagreement turned out to be about length: the three
two-line cues episode 1 kept whole at 0.44 s all had a piece of 3-4 characters (嚇唔係呀？ /
呢個係最東㗎啦？, 我唔知呀 / 唔講喇, 好痛呀 / 好難受), while every cut either checker made with a
piece that short had 0.64 s or more (我冇事啊 / …, 喂 / …). The short-piece margin removes
exactly those three and changes nothing else: episode 2's output is byte-identical with and
without it, and episode 1 goes from 2 of 5 cuts matching to 2 of 2. The bare-pause half is
editorial rather than acoustic, so the default sits at 0.7 s,
where both files agree on every cut (7 of 7). `--realign_pause_gap 0.4` is the episode-2
reading and finds over twice as many of its cuts. Most of episode 1's own 67 hand cuts are
out of reach by design: they split off a one- or two-character interjection (呀|冇問題,
哦|通常…), which the three-character rule refuses. Timing is unaffected either way: start
error against the finished file is identical with and without splitting.

On episode 2 the line-break rule reproduced every hand split, with every edge within 0.1 s of
the hand timing (starts within 0.03 s). The peak gate refused one candidate (a 2.56 s gap,
halves at 0.31 / 0.48); the checker later found a line missing from the transcript inside
that gap.

**Top-of-screen cues (`{\an8}`) are a track of their own under `sync` and `adjust`.** The
override block is read off the cue (`SubtitleCue.style`), kept out of the text the aligner
and proofreader see, and written back by the SRT writer (WebVTT gets `line:0`). Such a cue
is almost always background speech running concurrently with the main line, and everything
in this module assumes one ordered, non-overlapping stream, so in sequence it was a line out
of order: the anchor search read it with its neighbours, forced alignment put it before or
after the cue it overlaps, and `map_cues`' overlap trim cut the main cue short where it began.

`split_tracks` takes them out before the anchor search; the main track is fitted and placed
as if they were absent. Each top cue is then mapped through the main fit on its own
(`_overlay_transform` empties `Transform.ids` and re-derives each cut's `cue_indices`, since
both number the *main* track's lines), and under `adjust` forced-aligned alone inside the
same leash and clamped to its prior. Its alignment chunks are separate, and the finished cues
travel on `item["overlay_segments"]`, are assembled as their own track, and are interleaved
by start only at the writer (`heapq.merge`, never a sort, so the main track's order cannot be
touched). Under `transcript` there is no prior to place a concurrent line by, so a top cue
stays in sequence and only its tag is carried.

> **A line spoken over other speech needs `concurrent_emission`.** The trellis charges every
> frame a line does not occupy at the *blank* probability. That is right over silence and
> wrong over someone else talking: the main line's frames are expensive as blank, so the
> cheapest path finishes the background line *before* the main speech starts, at the edge
> of its window, whatever was really said where. Concurrent segments may pass over any
> frame as filler at the cost of the best token not in the line. Only the blank column
> changes, so peaks and spot-checks are unaffected.

**Chunk reach under `adjust`.** `build_align_input` used to put each chunk boundary in the
middle of the gap between two lines and cap the chunk at `chunk_size` from its start. On the
same ReZero episode, the line after the 93 s opening was given the chunk [109.1, 137.1] while
placed at 155.95: the chunk ended 19 s before the line began, and forced alignment put it in
the theme song, 46 s early. The placement itself was right (−0.03 s from its prior); only
stage 4 moved it. Two changes:

* the `chunk_size` cap now comes out of the silence before a group's lines, never the lines
  (all modes: a chunk that does not hold its own line is wrong in any mode);
* `reach` caps how far into a gap a chunk extends from the line beside it. `adjust` passes
  its leash, so a chunk never holds more of an empty stretch than the transform allows the
  cue to move into. `transcript` keeps the midpoint until it is re-measured on the fixtures.

* under a reach, a gap wider than twice it also ends the chunk (the cut `bracket_blocks`
  already makes for the placement), so no chunk holds a wide gap in its middle either.

This is a contract fix rather than a scoring one. The leash already *is* adjust's bound on a
move; the fault was that stage 4 did not honour it. **`hold_to_placement` closes it at the
last step**: each cue's placement rides through alignment (`cue_placements` →
`realign_placement`), and a cue the final alignment moved further than the leash takes its
placement back, flagged `off_prior`. Stage 4 moves the median cue 0.000 s from its
placement on that episode; of the moves over 1 s whose truth is known, the three over 2 s
were all wrong (2.1-2.8 s early, placement within 0.1 s) and the two under it were
corrections (+1.6 s, +1.7 s), so the leash is where the line falls.

**The main track passes over the top-of-screen speech too.** Taking `{\an8}` lines out of
the sequence leaves their speech in the main track's audio with nothing to consume it, and
it pulled the next main line back across it (奇怪喇 and 嚟唔切喇, 2.2-2.3 s early). The main
track's chunks and its placement fill therefore carry `filler_spans` (the top cues'
placements, or their source spans mapped through the transform) and apply
`concurrent_emission` on those frames only.

**Measured end to end on that episode against the hand-corrected file (to 07:23, 119 cues):**

| | cues | worst start error | start p90 | top cues tagged | hand splits made |
|---|---|---|---|---|---|
| before | 398 | 46.80 s | 0.067 s | 0 of 5 | 0 of 6 |
| after | 413 | 0.67 s | 0.059 s | 5 of 5 | 6 of 6 (3 of 3 refusals agree) |

Three cues got worse: 奇怪喇 (+0.09 → −0.67 s; the background line before it ends 啦, which
competes for its 喇, and it is flagged `isolated`), 着住嘅運動衫… (−0.02 → +0.14 s) and 激氣啊
(0.00 → +0.06 s, held to its placement). Past the hand-checked part, three more cues moved
2-42 s, each onto its own source position (the 42 s one was the same wide-gap fault before
the ending).

## Fitting the transform between two releases (pipeline/timefit.py)

`sync` and `adjust` both rest on one question: given the lines we are confident about, what is
the *simplest* map from the subtitle's timeline onto this recording's? The map is piecewise
affine in **source** time, `phi(x) = a_p * x + b_p`, and the whole design is choosing how many
pieces to use and when a piece may take a slope other than 1. Too few and a real cut smears
its error over the whole file; too many and anchor jitter is "explained" by inventing edits
nobody made.

**The fit is a segmented regression solved by dynamic programming.** For anchors `[i, j)` the
cost is the cheaper of an offset-only model (`a = 1`, free) and a free-slope model (charged
`SLOPE_PENALTY`), plus `BREAK_PENALTY` per additional piece:

```
F[0] = -BREAK_PENALTY                                  # the first piece is free
F[j] = BREAK_PENALTY + min over i of ( F[i] + C(i, j) )
```

`C(i, j)` is O(1) from weighted prefix sums, so the whole thing is O(n²) — about 450k
evaluations at 950 anchors, one numpy call per `j`. Everything is in **seconds²**, the same
units as the residuals, so each penalty reads directly as "how much squared error a piece must
save to be worth having". They are deliberately *not* normalised by anchor count: a fixed
charge says a break must be supported by this much evidence, and the evidence for a break does
scale with the anchors it repairs.

**`SLOPE_PENALTY` at ~7x `BREAK_PENALTY` is what implements "prefer cuts and insertions to
speed changes".** A 5 s step halfway through a 600 s span could be absorbed by a slope of
1.008, comfortably inside the bound; the penalty is what makes it come out as two offset
pieces instead, because a real edit is what a human has to act on. Where a rate change is
genuine the offset staircase would need dozens of pieces and still carry residuals, so the
slope still wins by orders of magnitude and the preference only bites where it should.

**The slope gate is hard, not a soft prior, and that matters.** A continuous penalty on
`(a - 1)²` needs an arbitrary time scale to be dimensionally meaningful, and worse, it
*shrinks a genuine rate change*: 1e-5 of slope is 0.03 s at the end of a 2800 s file, so any
shrinkage is visible there. Instead a free slope requires `MIN_SLOPE_ANCHORS` (8) spanning
`MIN_SLOPE_SPAN` (120 s) — the dangerous failure is a slope fitted over a short run and then
extrapolated — and beyond `--realign_max_scale` it is **refused rather than clamped**, since a
clamped slope is a number the fit does not believe. `TransformError` carries
`scale_bound_hit` so the refusal names the bound instead of blaming a content mismatch.

**Robustness is a trim loop around the DP, not a robust loss inside it.** A Huber loss in
`C(i, j)` needs an IRLS per candidate segment and would make the DP cubic; squared error
inside plus a 4-MAD trim outside gets the same protection at the same cost.
`MIN_SEGMENT_ANCHORS` (3) is what stops the DP hiding an outlier in a piece of its own before
the trim can see it. There is deliberately **no global pre-screen** — fitting one robust line
and dropping whatever sits far from it is tempting and wrong, because a genuine cut makes half
the anchors outliers of that line.

**Three refusals, and the third catches what the other two cannot.** Too few anchors, and too
few inliers, are the obvious ones. But trimming only notices anchors that disagree with an
otherwise-good fit; when the anchors disagree with *each other* so thoroughly that the DP
explains them with a break every few anchors, nothing is trimmed — every piece fits its own
handful perfectly. `MAX_PIECE_SHARE` refuses that. A refused fit falls back to the **identity**
and warns loudly: unchanged timings are wrong in a way the user can see, whereas timings
mapped through a fit nobody believes are wrong *and* look deliberate.

**Anchors only, and starts only.** A fill placement is derived from the anchors bracketing it,
so including it would multiply each anchor's weight by however many lines it brackets —
backwards, since a wide bracket is *weaker* evidence and holds *more* lines. And an anchor's
start is an acoustic onset while its end is a display decision (padded, released to the next
cue, floored), so pairing on ends would feed the fit noise dressed as evidence. A subtitler's
systematic lead-in bias on the start is absorbed exactly into the per-piece offset, so it
costs nothing: on bluey that bias measures +0.074 s.

**A cut says nothing about speed, and estimating slope independently per piece throws away
most of the evidence for it.** An edit removes or adds content; it does not change how fast
the surviving footage plays, so the file's speed is one property of the whole timeline, not a
per-piece one. The DP above nonetheless lets each piece's model pick its own slope, and slope
needs both span *and* anchor count to pin down precisely — so splitting the anchor budget
across pieces multiplies the variance of each piece's own estimate, and that variance shows up
as *progressive drift within a piece that resets at the next cut*: an error that is near zero
right after a break (where the piece's own fit is naturally best-anchored) and grows steadily
as the cue in question sits further from that piece's own centroid.

Measured on the ReZero Blu-ray (reported against 14 hand-checked reference points spanning the
file): the three independently-fit slopes were 1.042982 / 1.043248 / 1.043691, disagreeing by
up to 7e-4 — worth several tenths of a second of drift within each piece. Correcting each
piece's slope for its *own* reported drift independently gives 1.043427 and 1.043404: the two
disagreeing pieces agree with each other to four decimal places once corrected, which is the
signature of one true shared speed and three noisy independent estimates of it, not three
different speeds.

`_shared_slope_refit` is a refinement pass, run once the DP above has settled on pieces and
breakpoints (that part needed no fixing — both cuts were found precisely). It fits **one slope
across every piece's anchors together**: an ANCOVA-style common-slope regression, centring
each piece's anchors on its own weighted mean first (which is what makes the per-piece
intercepts drop out of the pooled normal equations algebraically), then fitting one slope
through the pooled, centred residuals. A piece keeps its own independently-fit slope only when
doing so earns back `SLOPE_PENALTY` over the shared one, so a file that genuinely splices
footage at two different speeds is not forced together — only pieces whose own slope cannot
justify its keep default to the shared one. Re-run on the real file: **all three pieces
converged to 1.043477**, and the within-piece drift the fix targets dropped by 89% (measured
as the change in average error between two widely-separated windows inside the same piece).

**A remaining piece-level near-constant error survives the slope fix, and it is not (mainly)
the offset formula.** Once the drift was gone, each piece still carried a roughly constant
error of its own (+0.27 s / −0.23 s against further hand-checked points). The natural next
suspect was the offset estimator: a piece's shared-slope offset was `weighted_mean(y -
a_shared*x)`, and a minority of anchors reading consistently late is invisible to the earlier
4-MAD outlier trim while still dragging a mean off the typical cue's position. Switching to
`_weighted_median` (`_weighted_median` in `timefit.py`) is correct in principle — a
constructed synthetic case with a 12% skewed tail per piece shows a median landing within
0.012 s of true where a mean is pulled 0.03-0.07 s off — but **re-measured on the same
Blu-ray, it made no net difference**: mean |error| against the hand-checked points went 0.202 s
(shared slope alone) → 0.207 s (+ median offset), better on some points, worse on others, all
within about 0.03 s either way. The synthetic case proves the mechanism works when the skew is
real; it does not establish that this file's anchors are actually skewed enough for it to
matter. Kept anyway as a defensive measure (it costs nothing when the data is symmetric, and
protects a future file that genuinely does have a skewed tail), but the remaining ~0.1-0.3 s
per-piece bias is **not explained by this**, and is left as an open question — see below.

**What plausibly *is* left is the floor of what a purely linear map through acoustic anchors
can do**, not a bug still to find. The remaining bias sits inside or near each piece's own
reported residual band (piece 2/3: p50 0.18-0.24 s, p90 0.32-0.42 s), so a handful of
hand-checked points cannot yet distinguish genuine systematic bias from where those particular
anchors happened to land. There is also a real, separate candidate: a subtitler's lead-in
convention (padding a cue open a beat before the audio, more so after a pause) is a difference
between *where a human puts the cue* and *the acoustic onset an anchor measures*, and `sync`,
being a single straight line through every anchor, has no way to reproduce a convention it
never measured cue-by-cue. `adjust` mode exists for exactly this gap: it uses the fitted
transform only as a bracket and re-times each cue individually by forced alignment rather than
trusting one line through all of them.

**Confirmed: this is exactly what closes the gap.** Run on the same Blu-ray with the same 14
hand-checked points: `adjust` brings mean |error| from 0.207 s (`sync`) to **0.029 s**, and max
error from 0.333 s to **0.062 s** — every point now inside the "a frame early is fine, exactly
on it is fine, late is the one to avoid" window the points were checked against. This also
confirms the transform fixes above were a real prerequisite rather than cosmetic: `adjust`'s
brackets are only `±realign_adjust_tolerance` (2 s) wide, so the 10+ second internal drift the
unfixed per-piece slopes produced would have made the bracket construction itself unreliable
long before forced alignment got a chance to help. Getting the transform's global shape right
is what makes `adjust` mode's local precision possible, not a redundant step before it.

**That "~0.1-0.3 s is the floor of a linear map" framing was wrong, and the fix is `map_cues`
using each anchor's own position exactly rather than reading it off the piece's fitted line.**
A piece-wide fit is an *average*; even a correctly-estimated shared slope still leaves every
individual anchor scattered around that average by the piece's own residual, and 0.1-0.3 s is
squarely inside what viewers notice — more than two frames, and this project's standard
elsewhere is that even one frame late is a problem. There was no acoustic reason for a cue
that *is* a confidently-placed anchor to be re-derived from a line fitted through hundreds of
others; the anchor's own measured value is simply the better answer, and using it costs
nothing extra since the acoustic search was already run.

`map_cues` (`_AnchorLookup` in `timefit.py`) now works this way: a cue that is itself a kept
anchor (tracked via `Transform.ids`, a transcript-line index parallel to `.xs`/`.ys`/`.ws`
threaded through `fit_transform`'s sorting and trimming) gets that anchor's own start exactly.
Every other cue is placed by **linear interpolation between its two nearest anchors** — a
two-point local fit, not the whole piece's — and only falls back to the piece's own affine
formula where there is no local anchor pair to use at all: outside the anchor range, or where
the two nearest anchors sit in *different* pieces (interpolating across a cut or insertion
would blend straight through the discontinuity `locate_breaks` exists to find). A cue's
duration still comes from scaling its own (source_end − source_start) by whichever rate placed
its start — the local two-anchor slope, or the piece's own scale at an edge — extending
"anchors only, starts only" to the one endpoint that was never itself measured.

> **`Transform.ids` must stay empty unless a caller passes real line indices, never default
> to a sequential placeholder.** The first version of this defaulted a missing `ids` to
> `range(len(pairs))` — harmless-looking, since it only changes what `Transform.ids` reports —
> but `map_cues` treats a populated `ids` as "cue *i* is anchor *i*" by line index, and a
> placeholder `0..n-1` collides with **every** caller's own span indices whenever they happen
> to be small integers too (which they always are). That silently matched unrelated test cues
> to a stranger's anchor purely because both carried the same coincidental integer, and would
> have done the same to real cues on any run through `map_cues` without `ids` at all. Give a
> caller with no line indices to offer `Transform.ids == ()` instead, so the lookup is simply
> empty and every cue falls through to interpolation/fallback exactly as it did before this
> feature existed — see `test_without_ids_map_cues_behaves_exactly_as_before`.

**Measured on the same Blu-ray, before and after this fix, against the independent
transcript-mode ground truth** (498 lines with source time < 1550 s, before transcript mode's
own search loses its place further into the file — see below): mean |error| **0.165 s →
0.021 s**, an 8x reduction, with **462 of 498 (93%) now matching to the millisecond** — those
are exactly the lines that are themselves anchors, now using their own measured position
instead of a value read off the fitted line. Cue validity is unaffected: 956 cues, 0
zero-length, 0 out-of-order. The handful of lines that still show meaningful error (up to
4.1 s) are pre-existing acoustic-search failures unrelated to this fix — one is 如果到時真係成功嘅
(the repeated-phrase confusion reported separately), one is a stylised title-card reading, one
is a short common word (媽媽) — and none of them regressed; they disagree with transcript mode's
own anchor for the same reason transcript mode and sync mode's OWN search can each be wrong
independently on hard audio.

**That "trust every anchor exactly" design was itself wrong, and the reason is exactly the
kind of failure `sync` is supposed to be immune to.** Every anchor the acoustic search
accepted was now individually overriding a cue's position, and a *confidently*-wrong anchor
(a repeated phrase confusing CTC, a stylised reading) is indistinguishable from a correct one
by its own score. Measured directly: four hand-flagged bad lines on the ReZero Blu-ray score
0.945-0.99 — above the anchor set's own median (0.937) — and a 0.9 confidence floor, which
would already refuse a quarter of every anchor in the file, keeps every one of them. Raising
`--realign_min_score` cannot fix this class of error, because the model is not uncertain
about these lines; it is confident and wrong, which a score can never see.

**`timefit.prune_to_density` thins the anchors `map_cues` trusts directly down to a sparse,
residual-screened subset** (default ~3/minute of the piece's own span, always keeping each
piece's two edge anchors so interpolation still reaches the boundary `locate_breaks` trusted
them to place) — everything else is interpolated between whichever of those survive nearby,
never given its own individually-trusted acoustic opinion. The fitted scale and each piece's
median offset are left exactly as fitted; only which anchors get *direct* trust changes, so a
handful of bad anchors cannot corrupt the transform itself, only be excluded from having a
cue named directly after them.

**Measured on the ReZero Blu-ray at the default ~3/minute density**: all four hand-flagged
lines moved in the direction independently reported as needed (the two reported "too late"
moved 0.04-0.18 s earlier; the one reported "too early" moved 0.57 s later) — a real, honestly
positive signal, though not something this session can fully confirm without hearing the
audio, since the same repeated-phrase confusion this feature exists to route around also
infects `--realign_mode transcript`'s own independent search on these exact lines, so it
cannot serve as a clean check here the way it does everywhere else in this document. The
**trade-off is real and worth stating plainly**: file-wide mean |error| against
transcript-mode's placements (the same 498-line sample used above, unaffected by that shared
weakness) went from 0.021 s (trusting every anchor) to **0.127 s** at this density — worse
than the all-anchors result, though still better than the pre-anchor-exact baseline of
0.165 s. Density is exposed as `--realign_sync_anchor_density` precisely because this number
is a dial, not a fixed constant: raising it recovers more of the 0.021 s aggregate accuracy at
the cost of trusting more anchors individually again, and the right value for a given file is
a judgement call about which failure mode costs more, not something one fixture settles for
every input.

`map_cues` also gained two passes that run after every cue has its first-draft timing, in
this order: `_trim_overlaps` pulls a cue's end back whenever the next cue's (independently
derived) start would run into it, leaving `align_padding` as a gap rather than a flush cut;
then `_pad_starts` shifts every start earlier by `align_padding` as the last thing that
happens to any cue's timing at all — a subtitler forgives a cue appearing a frame early far
more readily than a frame late, so every start is nudged the same direction on principle, not
only the ones a fit happened to place late. A pair the trim pass actually touched comes out of
padding exactly flush (the same amount was subtracted from both sides of the join), which is
why the trim has to run first and the pad last, not the reverse.

Prefer `adjust` over `sync` for the reasons already measured above (it still wins on this
file, since forced alignment beats even a well-placed straight line between two anchors, and
is not exposed to the same aggregate-vs-robustness trade-off `sync`'s own anchor density is);
the gap between them is real, and how large it is now depends on where `sync`'s density is
set.

**The anchor search ignores the input's own timings, in every mode.** This looks wasteful when
a subtitle arrives roughly in sync and it is the single most important decision here. A search
narrowed around the prior can only return the prior, and the transform would then be fitted to
its own assumption — which is exactly how `--retime` failed. The prior is the thing being
measured; it cannot also be the ruler. It is not the cost either: the encoder pass dominates
and is needed anyway.

**Cuts and insertions are the sign of the jump at a breakpoint**, `delta = phi_next - phi_prev`:

| `delta` | name | meaning | source time with no audio |
|---|---|---|---|
| `< 0` | **cut** | the target is missing content the source has | `[c0, c1]`, length `abs(delta)/a` |
| `> 0` | **insert** | the target has audio the source never covered (OP/ED, restored scene, ad break) | none |

A breakpoint is placed where it does least damage, the same tier ordering
`reference_context._split_one` uses: an insertion lands in the middle of the widest cue gap in
its bracket, and a cut is swept over the O(k) positions where an interval edge sits flush
against a cue edge, minimising the **share of cue content destroyed** rather than the count —
counting cues would sacrifice one long cue to spare three short ones, which is backwards. A
cut claiming to delete more source time than its bracketing anchors leave room for is
*infeasible*: the fit is asserting that content two verified anchors say was spoken is not in
the recording, so the two pieces are merged and refitted instead.

**Measured, on the bluey fixture syncing the ground-truth SRT against its own audio** — the
identity regression, where the right answer is known exactly: 1 piece, scale `1.000000`, offset
`+0.074 s`, 117 anchors (1 trimmed), residual p50 0.05 s / p90 0.17 s, **0 breaks**. All 132
cues moved by exactly +0.074 s, **no cue's duration changed**, and all 9 two-line cues kept
their line breaks. Note how much tighter the residual is than the 0.54 s from pairing two
*different-language* subtitle files: real acoustic anchors are the thing that makes this work.

**Measured on `experiments/realign-tests/ReZero`**, one Cantonese subtitle (956 cues, timed against a
25 fps broadcast) synced onto three different sources of the same episode. The same command
and the same defaults each time; only the audio changed:

| target | pieces | scale | offset | breaks | residual p90 | median cue move | wall |
|---|---|---|---|---|---|---|---|
| `original-aligned-video.mkv` (the source it was timed to) | 1 | **1.000000** | +0.039 s | 0 | 0.08 s | +0.04 s | 2:13 |
| `realign-target-1-viuTV.mp4` (same master, 25 fps) | 1 | **1.000000** | −0.298 s | 0 | 0.29 s | −0.30 s | 2:13 |
| `realign-target-2-bluray.mkv` (23.976 fps) | 3 | **1.043405** | +0.59 / +3.21 / +5.15 s | 2 insertions | 0.09–0.39 s | +65.98 s | 2:15 |

The first two rows are the regression: a file that already matches must come back as scale
exactly 1.0 with no invented structure, and it does, twice. The third is the case the feature
exists for — a PAL speed-up that `--retime` could not express at all, since by the last cue the
subtitle is 127 s out. `named_ratio` reported "close to a 25 -> 23.976 fps conversion".

**`sync` beats `adjust` where the input's internal timing is already good**, and the two modes
are a genuine trade rather than a quality ladder. Syncing the bluey ground truth against its
own audio, scored against itself:

| mode | median | p90 | max | within 0.25 s |
|---|---|---|---|---|
| `sync` | 0.074 s | 0.074 s | 0.074 s | 100 % |
| `adjust` | 0.071 s | 0.231 s | **1.102 s** | 91 % |

`sync` reproduces the subtitler's own 0.074 s lead-in on every cue because a linear map cannot
do anything else. `adjust` discards that convention and re-derives each cue from the audio,
which buys per-cue alignment noise in exchange for nothing, since the input was already right.

> **Read that table for what it is.** It is the identity case, where `sync` wins *by
> construction* — the answer key is the input. It is not evidence that `sync` is better in
> general, and it cannot be: the fixture has no cue whose internal timing is wrong, which is
> the only condition under which `adjust` has anything to fix. Choose by what you distrust. If
> only the subtitle's global alignment is off (a different release, a frame-rate conversion),
> `sync` fixes exactly that and touches nothing else. If its cue-to-cue timing is also
> unreliable — OCR drift, a sloppy fansub — `adjust` re-derives it and the noise is the price.

**The two insertions on the Blu-ray are a real finding, not noise**: 2.8 s at source 09:42 and
2.6 s at 25:58, which is where a double-length episode's act breaks sit — content the broadcast
trimmed and the disc keeps. Both landed in cue gaps, so **0 cues were affected and none were
dropped**. Scored against the Blu-ray's own embedded reference track (934 cues, Standard
Chinese, so a different segmentation): **p50 0.17 s, p90 0.48 s, 95.5 % within 1 s, 99.7 %
within 2 s**, with 848 of 853 anchors kept.

> **Do not snap a fitted scale to a known frame-rate ratio.** `named_ratio` reports the
> neighbouring conversion in the log because it usually explains *why* a file is out of sync,
> and that is all it does. On the ReZero Blu-ray three independent estimates of the same scale
> — endpoints (1.0427), a RANSAC over paired cue starts (1.043494), and a grid search plus this
> fitter (1.046246) — **span 0.35%, which is ~10 s of drift end to end**. The embedded
> Standard-Chinese reference track pins the scale to about ±0.35% and no better, because it is
> a different language with different cue segmentation (934 cues against 956). Rounding to
> 25/23.976 = 1.042709 on the strength of that would put the last cue seconds wrong. The label
> is cosmetic; the number is not.

**The transcript's line breaks are the cue boundaries.** `split_chars` exists because Qwen
returns an undifferentiated block per segment that has to be cut up; a transcript arrives
pre-cut. So realign declares its cue structure to alignment through the new
`SingleSegment['cue_spans']` (inclusive index pairs into the joined text, consumed by
`_preprocess_segment`), and punctuation is demoted to what it acoustically is -- a pause.
Deriving cues from punctuation here would be actively wrong: real transcripts are punctuated
far too sparsely. The Police Story 2 transcript ends only 9.9% of its lines with a split
char, so pass A's mergeable-boundary test would hold almost nothing back and most of the film
would fuse into a handful of cues. `assemble_cues(merge=False)` therefore turns off the two
passes that *join* cues (A and C) while keeping the noise drop (B) and the duration floor (D).

**Three pieces, and each exists for a measured reason.**

| Piece | Where | Why |
|---|---|---|
| Split-only chunking | `vads/curve.py:cover_chunks` | a transcript line exists for speech VAD scores below threshold |
| One emission timeline | `alignment.py:EmissionTimeline` | a chunk edge must not be an alignment edge |
| Anchor-and-fill placement | `realign.py:assign_lines` | a forward-only sweep cannot recover from losing its place |
| Cue-edge tightening | `realign.py:tighten_cue_spans` | a blank token's dwell must not set a cue's start |
| Dwelling characters | `alignment.py:_reseat_dwelling_chars` | a character's *own* span must not swallow a pause |
| Transcript order | `realign.py:_sanitize`, `enforce_cue_order` | the line order is authoritative; nothing may permute it |
| Two chunk budgets | `realign.py:build_align_input` | time alone does not bound what a trellis can hold |
| Cue visibility repair | `realign.py:ensure_visible_cues` | a zero-length cue invalidates the SRT |
| Implausible-cue report | `realign.py:warn_on_implausible_cues` | a cue too long for its text was chunked wrong |

**Chunk grouping bounds characters as well as seconds**, and the second budget is not
optional. CTC needs an emission frame per token and the align model runs at ~25 fps, so a 30 s
chunk holds a few hundred characters at most. Group on time alone and a run of lines the
search stranded on one timestamp all lands in a single chunk, where the trellis has far more
tokens than frames, `backtrack` returns None, and `_align_segment` falls back to **one** cue
carrying every line in the group. On the first Police Story 2 run that was 1015 lines and 6492
characters in a single subtitle. `MAX_CHARS_PER_SECOND` (half the frame rate, leaving room for
the blanks between characters) is what stops it.

**Split-only chunking.** Ordinary VAD keeps speech and drops the rest, which is right when ASR
is going to read it. Under realign we hold a line for every utterance in the file *including*
the sung, shouted and music-bedded ones VAD scores low, and dropping that audio leaves the
line nothing to align against. `cover_chunks` reuses **stage 3 of `Binarize` and only stage 3**:
the single region `[0, duration]` goes straight to the min-cut, which splits at the
lowest-scoring frame in the second half of each over-long window. Because a split timestamp
both ends one piece and starts the next, the output tiles the file exactly -- sorted, disjoint,
gap-free, each piece between `chunk_size/2` and `chunk_size`.

**One emission timeline (`alignment.py:EmissionTimeline`).** The file's emissions are held as
one continuous timeline and sliced by time, so a segment may be aligned against audio that
spans a chunk join. This is a prerequisite for everything below, not an optimisation: 22-44% of
the spans placement wants to align in one piece cross a join (bluey 4/9, ps2 head 2/9, Doraemon
15/47). Before it, `_get_emission_for_segment` could only slice the *one* chunk holding a
segment's start, so `build_align_input` had to re-cut the file to keep every line inside a
single chunk -- and a coarse guess a second or two out then put the boundary *in front of* the
line it was meant to contain. That is what produced 「由今日開始」 as a 0.2 s cue at score 0.000,
its speech having ended 1.5 s before its chunk began.

Frame times come from each chunk's own frame count, not a global rate (chunks vary by up to 2%,
which would smear a cross-join slice), and emissions are held float16 and upcast per slice --
about 1 GB for a two-hour film against the 2 GB float32 copy `align()` used to hold. The same
timeline serves the placement search and the final alignment, so the encoder runs **once** over
the file where the acoustic anchor previously ran it twice.

**Anchor-and-fill placement.** The old search walked the file with one pointer, asking each
window "how much of the remaining transcript does this explain?" and advancing by whatever it
consumed. On audio the model cannot read the honest answer is "none of it", but the window still
had to move on -- so time advanced by a whole window while the line pointer barely moved, and
because the pointer only ever went forward it could never recover. Measured against ground
truth, that scored a **median 153 s** start error on Doraemon (only 37% of lines within half a
second), and on Police Story 2 it ran off the end of the film with 1015 lines unplaced.

The replacement is four passes:

1. **`acquire`** sweeps for anchors as before, with two changes. The line pointer may **jump**:
   when the probe at the pointer explains nothing, the same window is retried at pointer +
   `ANCHOR_OFFSETS`, so a search that has lost its place can re-acquire rather than stay lost.
   And progress in *time* is guaranteed -- each iteration either anchors or advances the window
   -- so the loop cannot grind on unreadable audio. Lines that are not anchored are **not
   committed**; they are filled later from their brackets.
2. **`sanitise_anchors`** keeps the heaviest set of anchors that can all be true at once:
   increasing in time, and leaving at least `chars_between / MAX_CHAR_RATE` seconds for the text
   between them. Only the *lower* bound is used -- capping the forward jump is the documented
   trap below.
3. **`verify_anchors`** forced-aligns overlapping blocks of consecutive anchors against the audio
   those anchors bracket and drops any interior anchor that moves more than `VERIFY_TOLERANCE`.
4. **`fill`** forced-aligns everything between two consecutive anchors against exactly the
   bracketed audio. Forced alignment is correct there precisely because the brackets are
   trusted, so every token is consumed and each line lands where the audio supports it.

A bad stretch is then trapped between two good anchors and cannot spread.

**Verification is what catches a confidently wrong anchor, and two obvious alternatives were
measured and do not work.** On the Police Story 2 head two police-radio lines anchored at
214.8 s and 223.8 s with scores 0.92 and 0.89 while belonging at 195.8 and 197.2. Nothing local
separates them -- the score is high and the implied speaking rate is fine. Asking whether the
15 s they skipped contains unaccounted *speech* fails twice over: the emission's blank
probability is not a voice-activity detector (CTC is peaky enough that a whole 360 s clip shows
24 s of non-blank frames, and the skipped stretch measured **0.00 s**, the same as a genuinely
wordless one), and the VAD speech spans do no better (1.41 s in the false jump against 1.52 s in
a real 64 s silence). Do not reintroduce either. What separates them is their neighbours, which
is what `verify_anchors` asks; dropping those two took the head from 94.7% to 97.4% within half
a second.

That needed `free_end` on both `get_trellis` and `backtrack`, and the two **must** agree --
`backtrack` raises rather than answering wrongly if they do not. The forced trellis seeds
`trellis[-num_tokens:, 0] = +inf` to terminate its backward walk, and the recurrence carries
that `+inf` diagonally, so by the final row it has flooded every token column but the last.
A free-end search over such a row has only one column to "choose" and silently degenerates
into a forced alignment. This cost two debugging rounds; do not remove the guard.

**Cue-edge tightening is not optional, and the reason is subtle.** `_align_segment` takes a
subsegment's start as the minimum over *all* its characters. Two ordinary facts then combine
badly: a character outside the align model's vocabulary carries no timing at all (interjections
are, and are exactly what a line tends to open with), and the punctuation after it was mapped
to the blank token, on which CTC will dwell for as long as the silence lasts. The cue then
inherits its start from a comma that "began" ten seconds earlier, in the preceding silence.

Measured on `experiments/fixtures/bluey`, before and after `tighten_cue_spans`:

| | max start error | within 1.0s | within 2.0s | mean signed |
|---|---|---|---|---|
| before | **8.809 s** | 97.0 % | 98.5 % | -0.066 s |
| after | **1.071 s** | 99.2 % | 100.0 % | +0.099 s |

Median (0.076 s) and p90 (0.246 s) are unchanged -- this only ever touched the tail. Anchoring
on the first *timed, non-blank* character costs the unmodellable interjection, so those cues
now start about a beat **late** instead of nine seconds early, which is the right trade for a
subtitle. The residual ~1 s on those lines is irreducible without vocabulary support.

**Only a *short* detached run is dropped, and only from the ends** (`MAX_DETACHED_RUN`, 2
characters). The first version of this kept whichever run was longest, which over-reaches: a
line with a real pause in the middle of it loses its correctly-aligned opening because the
other clause is longer. Police Story 2 cue 18, 「無錯喇，我哋係唔需要身手好嘅警察」 -- the
emission puts 無 at 296.120 s (hand-measured 296.114) and the comma after 喇 then dwells
1.400 s -- came out 1.77 s late for exactly that reason, on both anchors. Edge-trimming fixes
it (296.117 s on the film) and changes nothing on bluey or Doraemon, where every detached run
is a single character. A tie between two equal runs still resolves towards the opening.

> **The confidence score does not see this failure.** Those misplacements scored 0.96-0.99,
> *higher* than the median, while the lowest-scoring lines were placed to within 0.05 s. The
> model really is confident, just about the wrong frame. `--realign_min_score` is a transcript-
> versus-recording mismatch detector, not a timing check; do not read a clean run as proof the
> timings are good. Measure with `scripts/eval_realign.py`.

**What does see it is the geometry.** `find_implausible_cues` applies `_bounded_timing`'s test
to the *final* timings: a cue longer than its text could be spoken in (below `MIN_CHAR_RATE`)
means the line was grouped into an alignment chunk that does not hold its speech, and forced
alignment spread it over whatever the chunk did hold. On Police Story 2 an eight-character line
belonging at 368.8 s was grouped into the chunk starting at 372.9 s and came out as an 18.5 s
cue **at score 0.983**. The complementary case -- the line's audio ending *before* its chunk
begins -- leaves nothing to align against and does show up in the score (`MIN_CUE_SCORE`);
「由今日開始」 scored exactly 0.000 and got a 0.2 s cue. Both are reported, not repaired:
neither edge of such a cue is trustworthy, so the fix belongs upstream. 20 of 2098 cues are
flagged on the full film.

**Anchoring (`--realign_anchor`), and the acoustic anchor's one real failure mode.**
`acoustic` (default) is the search above and runs no ASR. `asr` runs the normal ASR stage and
matches the two character streams instead.

The acoustic search is excellent on audio the align model can read and fails hard on audio it
cannot, and the boundary between those is sharper than it sounds. Measured on the Doraemon
fixture, whose middle contains a stretch of Italian dialogue a Cantonese character model has
no way to represent:

| lines | drift |
|---|---|
| 0-186 (the first 594 s) | within **0.2 s** |
| 187 onwards | +38 s, then +166 s, then +300 s, ... |

The mechanism is worth understanding before touching this code. The search is strictly
forward-only, and the free end has **no incentive to consume tokens**: the trellis maximises
total score, so where the model cannot read the audio, dwelling on blanks scores better than
advancing through poorly-matching characters. The window then places its lines near its own
far edge, the pointer jumps, and the next window starts past the audio those lines belonged
to. Nothing downstream can undo it, because a forward-only search cannot look back.

`_bounded_timing` keeps a *single* line from claiming more audio than its text could be spoken
in, which stops the pointer walking off the end of the file and is why the failure now
degrades instead of exploding. It does not fix the drift itself.

A second, independent data point: **Police Story 2** (2 h 02 m action film, 2189 transcript
lines). The acoustic anchor consumed ~6.2 s of audio per line against a true rate of ~3.3 s,
ran out of film at line 1174 of 2189, and stranded the remaining 1015. Nothing exotic went
wrong -- an action film simply has long stretches with no dialogue, and each one is audio the
search advances through while consuming no lines. Long non-dialogue content is the same
failure as unreadable content, arrived at from the other direction.

> **A bound on the forward jump between lines was tried and is a trap.** The theory is sound
> -- a complete transcript should not leave two minutes unaccounted for -- but films contain
> genuinely wordless stretches, and clamping the first legitimate one leaves the pointer
> lagging, which makes the next correctly-placed line look stranded too. It cascades: median
> error on Doraemon went from 0.05 s to 127 s, with the whole transcript running ahead of the
> audio. The code says this too; do not reintroduce it.

Measured on the Doraemon fixture against ground truth, per transcript line (not per matched
cue -- every line is scored), with `scripts/bench_realign_placement.py`:

| | cues out | median | p90 | within 0.5 s | wall |
|---|---|---|---|---|---|
| old forward-only sweep | 211 / 489 | 0.049 s | 615.6 s | 36.8 % | 97 s |
| anchor-and-fill | **489 / 489** | **0.042 s** | **0.138 s** | **96.9 %** | 98 s |

Note what a broken run looks like: the old sweep's *median* is fine. It placed the lines before
it lost its place perfectly well, and half the file was still more than two minutes out. Median
is the easy statistic here; watch p90 and the share within 0.5 s.

Two degenerate inputs, also on Doraemon:

* **Transcript 40% longer than its audio** (the Police Story 2 failure): the 255 lines that do
  have audio still place at median 0.042 s, 96.5 % within half a second. The rest are laid out
  past the end and flagged `no_audio`, instead of the whole tail collapsing.
* **40 phantom lines injected**: median 0.043 s, 94.9 % within half a second. Anchors fall from
  406 to 292 and the fill absorbs the difference.

**`acoustic` is the default and is now the better anchor on every fixture measured**, including
the ones `asr` was added for. The full Police Story 2 film, scored against 17 hand-measured
lines: `asr` placed 3 within half a second, `acoustic` places 15. `asr` still exists because a
text match can leave a line *unmatched* where forced alignment must place it somewhere, but it
costs a full ASR pass (4:55 of a 7:55 run against 7:36 total for `acoustic`) and no longer wins
on accuracy.

**Spot-checks still apply.** The branch threads the run profile's `spotchecks` (from the language pack), so
the default `--model Qwen3-ASR` supplies 喇/啦, 呀/啊/吖, 咋/啫 and 咁→噉 (+0.8) even though no
ASR ran -- the reselection is acoustic and only ever needed the align model. Note 啊 and 咯 are
absent from their own candidate sets (`languages/yue/__init__.py`) and so are *always* rewritten.

**Saying which lines to distrust.** A run of 2189 lines is not usable unless the output says
which cues to check, so every line carries a *reason* rather than a bool. `LineTiming.reason` is
one of `no_audio`, `unreadable`, `no_vocabulary`, `isolated` or `implausible`/`silent_start`
(the last two added after alignment), it survives forced alignment onto the finished cue
(`SingleSegment['cue_reasons']` -> `SingleAlignedSegment['realign_reason']`), and it comes out
three ways: in `realign/result.json`, as a breakdown at the end of the run, and as
**`realign/suspect.srt`** -- only the doubtful cues, on their final timings, each prefixed with
its reason:

```
141
00:07:43,329 --> 00:07:43,741
[isolated] May呀
```

Load that beside the film and you step through the lines worth checking, in order. Same idea as
the diarization debug SRT.

**The per-cue trust signal is geometry, not confidence, and that was measured.**
`--realign_min_score` is a transcript-versus-recording *mismatch* detector and is logged only;
attaching it to cues was tried and is noise. On Police Story 2 it flagged 187 of 2172 lines,
almost all correctly-placed short interjections (「唔該」 scores 0.001 and is right to within
0.2 s), and it **missed** the one line that is badly wrong -- `May呀`, 48 s out at score 0.733,
above the median. What does catch it is `flag_unconstrained`: a line whose neighbours leave more
room than `UNCONSTRAINED_ROOM` x the most generous time its text could occupy was not pinned
down by anything, and forced alignment was free to put it anywhere in that gap.

It is precise about the catastrophic cases and blunt about small ones. On the fixtures it flags
2 % of Doraemon and catches its worst error (6.2 s) and the ps2 head's only bad line (49 s); it
misses errors in the 2-5 s range whose geometry looks plausible. On the full film it flags 213
of 2090 cues (210 `isolated`, 1 `no_vocabulary`, 2 `silent_start`) -- an action film has a lot
of genuinely wide gaps. Tune `UNCONSTRAINED_ROOM` by watching a `suspect.srt` and counting how
many flags were real, not by picking a tidy number.

`no_vocabulary` is counted apart in the summary, because it means nothing went wrong. It used
to be the *second* most common reason on this film (17 lines, almost all 嘿 and 欸); character
substitution takes it to 1 at the default level and to 0 at `near`. See "Characters the align
model has no token for" in [alignment.md](alignment.md).

`--realign_mode transcript` does **not** suppress text cleaning: its input is a raw transcript,
not a finished subtitle, so it wants the rule files as much as ASR output does.


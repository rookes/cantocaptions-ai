# Alignment

Design notes and measurements for `pipeline/alignment.py`, `align_profiles.py`, `align_checks.py` and `align_vocab.py`.

## Alignment: audio priming and the silent-start check (pipeline/align_profiles.py + align_checks.py)

`alvanlii/wav2vec2-BERT-cantonese` was fine-tuned on clips that begin at speech onset and
reproduces that prior: **given audio whose first frames are not speech, it emits the
utterance's first character at emission frame 0 and nowhere else.** Measured on one
segment, the character scores ~0.999 at frame 0 and ~1e-6 at its real onset 2.8 s later.
Forced alignment then has nothing to find and pins each VAD segment's first character to
the segment start.

This was invisible until `vad_pad_onset` (a0bbc8b) began handing alignment segments that
start ~0.2 s before the speech does. Before that, frame 0 *was* speech onset, so the prior
happened to be right.

**Two independent pieces, deliberately split:**

| | Scope | Where |
|---|---|---|
| Audio priming | per align model, opt-in via registry | `align_profiles.py` |
| Silent-start check | every align model, always on | `align_checks.py` |

**Priming (`align_profiles.py`).** `AlignProfile.primer` prepends speech-like left context
to each segment so the start-of-utterance prior fires on the primer instead of on the first
real word. `_compute_vad_emissions_batched` then discards the primer's frames, so nothing
downstream knows it existed. `TailPrimer` uses a reversed copy of the segment's own tail —
always available, same voice, and unmistakable for dialogue if it ever leaked.

Only *speech-like energy* satisfies the prior. Digital silence, low-level noise and masked
left-padding were all measured and all leave the character at frame 0, so a silent buffer
will not do. Reversed and forward tails agree on 12 of 14 segments.

> The primer reports only `prefix + audio`, never how much it prepended. The frame count to
> drop is measured from the encoder's own output length for the **unprimed** audio, which is
> exact whatever the prefix was — that is what keeps the strategy swappable (a fixed shared
> buffer, room tone sampled per file). Do not "optimise" this into a duration estimate: an
> off-by-one frame hands back the artifact the primer exists to remove, silently.

Adding an align model = one `ALIGN_PROFILES` entry. Every field is a no-op by default, so a
model with no entry runs unprimed rather than inheriting another model's fix.

**The check (`align_checks.py`) is not per-model and not optional.** `find_silent_starts()`
flags any cue whose start sits on audio at the region's noise floor with no sound within
`MIN_GAP_FRAMES` (4 frames = 160 ms). It reads only timings and audio — never the emission,
blank id, or vocabulary — so a new align model gets it for free. It runs at the end of
`align()`, after the release/trim pass, and doubles as the primer's post-condition: on the
fixtures it goes 5 → 0 hits once priming is on.

**It is precise, not exhaustive.** Over 96 segments in three files: ~94% precision, but
only ~48% recall. It sees only a genuinely near-silent lead-in, and the failure also happens
over music beds and room tone — speechless but not silent. A clean run means "nothing
obvious", not "alignment is sound". Two guards keep it quiet: regions with less than
`MIN_SEPARATION_DB` between floor and speech are skipped rather than guessed at (a music bed
sits a few dB under quiet speech, and every false positive seen during development was one),
and the gap threshold is 4 rather than 3 because CTC leads the onset by a frame or two and
unvoiced onsets (就, 三, 開) read as silence to an energy envelope.

Verify with `scripts/bench_align_primer.py temp/07`, which aligns a saved `--debug_dir` both
ways and reports per-segment deltas plus the check's counts. **A zero delta is only wrong
when the audio there is silent** — a VAD region that opens on speech legitimately starts at
0.000, which is why the Doraemon fixture shows many zero deltas and zero silent starts.

## A cue that is not one utterance (pipeline/align_checks.py)

`find_gapped_cues` flags any cue holding a silence over `MAX_INTERNAL_GAP` (1 s) between two
of its **directly adjacent** characters. CTC has to place every token it is given, so when a
cue's text contains something that was not said where the cue sits, the extra characters go
wherever they score least badly — leaving a hole in the middle of the cue. On `experiments/fixtures/bluey`,
ASR emitted 爸爸， once for what the reference has as two separate calls: the first 爸 landed
at 56.44 s and the second at 59.80 s, so the subtitle went up **3.35 s** before anyone spoke.

**It is the best per-cue trust signal on the ASR path**, and much better than the CTC score
(which is useless for this; see [realign.md](realign.md)). Measured over the two ASR fixtures,
557 cues with a hand-made reference:

| cues | n | median start error | within 0.5 s |
|---|---:|---|---|
| with a gap over 1 s | 12 | 1.25–3.35 s | 0–45 % |
| without | 545 | 0.05–0.09 s | 76–79 % |

**Adjacency is what makes it safe to leave on for every model.** Punctuation is mapped to
blank precisely so it can absorb a pause, and it does that by *spanning* it — the mark stays
contiguous with both neighbours, so an ordinary clause break produces no gap at all. On both
fixtures every gap found is between two real characters with nothing between them.

**At its 1 s reporting threshold it reports and does not repair, deliberately.** 45 % of the
cues it flags are still within half a second of the reference, so trimming or splitting *there*
would hurt about as often as it helped — and which side of the gap to keep has no general
answer. Realign's `_trim_detached_edges` resolves that tie towards the opening, which is right
for a whole transcript line and wrong for a two-character cue (it declines to trim 爸爸， for
exactly that reason). It lands as a cue `note`, so it shows up in `notes.srt` next to the
substitutions.

**`split_gapped_cues` is the opt-in repair, and it is a different bet at a different
threshold.** Where the gap is long enough that the two halves cannot plausibly be one
utterance, the honest fix is not to pick a side — it is to stop pretending the cue is one cue.
Both halves' own timings are the ones alignment actually measured; only the cue *spanning* them
has an edge nobody spoke. So the split needs no answer to "which side", which is exactly what
made trimming unworkable.

`SPLIT_INTERNAL_GAP` (1.5 s) is the suggested starting point rather than `MAX_INTERNAL_GAP`
(1.0 s) because the 45 % figure above is a statement about the reporting threshold: a break at
1 s would invent a cue boundary about as often as it found one. **This has not been measured on
a fixture the way the reporting threshold was** — 1.5 s is reasoned from the reporting numbers,
not swept — which is why nothing enables it by default.

The cut is exactly the silence and nothing is retimed: the head keeps the cue's own start and
ends on its last timed character before the gap, the tail begins on its first timed character
after it and keeps the cue's own end. It therefore runs **after** `align()`'s release/trim pass
(so it cuts against final edges, and every cue outside the split is untouched) and **before**
the checks, so those judge what the pipeline will actually consume. Each piece carries a
`split_gap:` note. An untimed character between the two neighbours goes with the *head*: it was
never placed, so nothing says it belongs to the clause after the silence.

**Enabling it is per align model, not per pipeline.** `AlignProfile.split_gap`
(`align_profiles.py`) holds the threshold, because how long a hole a model leaves between two
characters is a fact about that model's emission; a model with no measured number inherits
nobody else's. `--align_split_gap SECONDS` overrides the profile for one run, and `0` forces it
off. **No shipped profile sets one**, so the default behaviour is unchanged and pinned by test.
The detection is shared with `find_gapped_cues` — one scan, two verdicts — so the report and
the break can never disagree about what a gap is.

**It does not apply under `--realign`, which passes `0` regardless of the flag.** There the
transcript's line breaks *are* the cue boundaries and a cue is one whole line by contract; a
line whose audio has a hole in it is the `find_implausible_cues` case, whose fix is upstream in
chunking. Cue assembly cannot undo a split either way: the pieces are `split_gap` seconds
apart, far beyond both `--align_merge_distance` (0.08 s) and `--merge_gap` (0.25 s).

## Per-character `peak`: was this character heard? (pipeline/alignment.py)

`merge_repeats` reports a character's `score` as the mean probability over **every frame of its
path span**, and most of those frames are CTC's blank dwell. That makes the score track the
character's *duration* far more than whether its token ever fired, so it says almost nothing
about whether the character is right. Every aligned char (and word, as the min over its chars)
now also carries `peak`: the highest probability the character's *own* token reaches anywhere on
its path. A correctly transcribed character fires near 1.0; a wrong one the trellis was merely
forced through stays low. As an error locator over ASR output it found roughly half the wrong
characters where `score` found about a tenth -- but it shares the align model's blind spots:
low-confidence sentence-final particles (啊 especially) and characters substituted through
`VocabRepair` read low whether or not they are right. It is diagnostic only; nothing in the
pipeline acts on it.

## Characters the align model has no token for (pipeline/align_vocab.py)

A CTC align model can only place a character it holds a token for. `_preprocess_segment`
drops everything else, so an out-of-vocabulary character is not mistimed — it is **absent**,
contributing no evidence at all, and a line made only of such characters cannot be timed from
its own audio. On the Police Story 2 transcript against `alvanlii/wav2vec2-BERT-cantonese`
(2680 entries) that is **326 characters over 88 distinct ones, and 17 whole lines**.

`VocabRepair` gives the dictionary a token for them, in escalating tiers:

| tier | example | why it is sound |
|---|---|---|
| `variant` | 辉 → 輝 | the transcript used the wrong character set; nothing acoustic is wrong |
| `homophone` | 駒 (keoi1) → 區 (keoi1) | the same syllable, so the same acoustics |
| `near` | 悍 (hon5) → 漢 (hon3) | same syllable, different tone — weaker, but the alternative is no token |

**This is not a text edit.** The original character stays in `clean_char` and therefore in the
subtitle, exactly the way punctuation keeps its own character while being tokenised as blank.
Only the token id changes.

**The mechanism is the dictionary itself.** `augment()` writes `original → token id of the
replacement` into the align dictionary, so `_preprocess_segment`'s membership test,
`_align_segment`'s token lookup and `realign.line_tokens` all pick it up from the one dict
they already read. Nothing is threaded through a signature, and the coarse search and the
final alignment cannot end up disagreeing about which characters are alignable — the call is
idempotent, so realign runs it before its search and `align()`'s later call finds nothing left
to do.

**Digits and punctuation are deliberately out of scope.** `8` has a reading (baat3) and would
substitute happily, but a digit in a transcript may be spoken in Cantonese, in English or
digit by digit, and nothing in the text says which. Punctuation the aligner wants is already
mapped to blank by `split_chars`; a cue holding nothing else is noise and is dropped
downstream. `_repairable` therefore admits letters only.

Two details that are load-bearing:

- **A character with no reading of its own reads through its variant.** 撺 is unknown to the
  reading data, but its traditional form 攛 is cyun1, which resolves to 村. Without that
  fallback the four Simplified characters in this transcript whose traditional forms are also
  missing resolve to nothing.
- **Candidates are ranked by HKCanCor frequency, then by code point.** The commoner character
  is the better-trained token; the code point is there so a run is reproducible rather than
  dependent on dict order.

Measured on Police Story 2 (76 distinct CJK characters, 249 occurrences):

| level | distinct resolved | occurrences | lines left unalignable |
|---|---|---|---|
| `off` | 0 | 0 | 17 |
| `variant` | 4 | 4 | 17 |
| **`homophone`** (default) | **57** | **217** | **1** |
| `near` | 69 | 240 | 0 |

`homophone` is the default because an exact-reading substitution needs no justification
beyond "same syllable"; `near` trades tone accuracy for the last 12 characters and is opt-in.
One character carries 22% of the total on its own (駒, 72 uses, almost all 家駒), and 嘿 alone
accounts for 9 of the 17 unalignable lines.

**Three layers, and the order matters.** A substitution only means anything relative to one
vocabulary — 爹 → 弟 is useful only because *this* model holds 弟 and not 爹 — so the
hand-picked table belongs to the **align profile**, not to the pipeline:

| layer | where | beats |
|---|---|---|
| automatic tiers | `VocabRepair.resolve` | nothing |
| the model's bundled table | `AlignProfile.substitutions` → `pipeline/align_substitutions/*.toml` | the automatic tiers |
| the caller's file | `--align_substitutions FILE` | both |

The two files merge **per character**, so a user file can correct one entry without restating
the rest, and `"x" = ""` switches a single character off. `align_vocab.bundled_substitutions`
loads and caches the profile's table; `merge_substitutions` layers them.

**The bundled table is deliberately short**, and `scripts/report_align_vocab.py --toml`
output does *not* belong in it wholesale. Those are the automatic choices — they already
happen without the file, and freezing them there would hide them from the report and stop
them adapting to the transcript at hand. It holds only what the tiers cannot reach or get
wrong. Today that is three entries:

- **爹 (de1) and 嗲 (de2)**, which are the case the tiers *provably* cannot reach: no
  character in the 2680-entry vocabulary reads `de` at any tone, so both the homophone and
  the tone-relaxed tier come up empty. Both are common — 爹哋 "daddy", 收嗲 "shut up".
- **脅 → 協**, correcting a defensible-but-wrong automatic pick. pycantonese reads 歉 as
  hip3, which is not its standard reading (him3), and 歉 then wins the frequency tie-break.

**How 弟 was chosen, since nothing is a homophone.** Measured on Peppa Pig S2E08, which says
爹哋 in four cues. Standardising the emission **per character** over the whole file — each
character's own mean and standard deviation across all 7550 frames — and only then reading
the frames 爹 occupies puts d-initial characters at the top of three of the four (特 dak6,
弟 dai6, 電 din6, 地 dei6, 典 din2, all z ≈ 3.1–3.3). The model hears the /d/ onset; it
simply has no `de` rime to put it on. The whitening is what makes this visible — on raw
probabilities the window is dominated by leakage from the neighbouring syllables and by
whatever is merely common.

> **Do the whitening, not the raw scores.** Every character carries its own baseline, so
> ranking a single frame by probability mostly re-reports how common each character is. An
> earlier pass without it returned 去/啦/下 — the tails of the adjacent words — and 𤓓, which
> also ranks 8th in a control run on unrelated audio and is therefore a garbage-sink token,
> not a match. `temp/peppa_s2e08_emission_coeffs.csv` is an example coefficient dump.

Substituting recovers most of what skipping loses. Aligning those four cues:

| | cue start vs reference | worst |
|---|---|---|
| no substitution | **+0.190 s** late | +0.22 s |
| any candidate | **−0.031 s** | 0.08 s |

Note the second row: *every* candidate lands on identical frames with own-scores spanning
0.648–0.651. For 爹哋 the token is a placeholder soaking up the frames before 哋 anchors the
cue, so its identity does almost no acoustic work — which is why the whitening, not the
timing, is what picks between them. 弟 over 地 for a reason the scores do not show: 地 is a
homophone of 哋, so 爹哋 would become 地哋, two dei6 tokens in a row inviting them to compete
for the same frames.

**Curating the list.** `scripts/report_align_vocab.py` reads a transcript exactly the way
`--realign` does, asks the model's own tokenizer what it holds, and runs this resolver over
the rest — so it reports what a run would actually do. `--toml` emits a ready-to-edit
`[substitutions]` table, which comes back as `--align_substitutions FILE` and beats every
automatic tier (`"摷" = ""` switches one character off without switching off the feature). It
loads the tokenizer only, so it needs no GPU.

```bash
uv run python scripts/report_align_vocab.py transcript.txt --level near -o missing.md
uv run python scripts/report_align_vocab.py transcript.txt --toml > substitutions.toml
```

**Every substitution is recorded on the cues it touched**, as a `notes` entry
(`homophone:駒→區`), and a character that resolved to nothing gets `no_token:摷`. See "Cue
notes" below.

**Measured on the full film**, against the same run with substitution off (same VAD, isolation
off, nothing else changed):

| | off | `homophone` | `near` |
|---|---:|---:|---:|
| hand-measured lines within 0.5 s | 14/18 | 14/18 | 14/18 |
| cues | 2090 | 2090 | 2090 |
| longest cue | 5.40 s | 5.40 s | 5.40 s |
| cues > 8 s | 0 | 0 | 0 |
| `no_vocabulary` | **17** | **1** | **0** |
| `isolated` | 209 | 212 | 212 |
| `suspect.srt` | 225 | 213 | 213 |

The safety property is the important one: **17 of the 18 hand-measured lines are unchanged**,
and the one that moved contains a substituted character (巡, +0.12 s → -0.04 s). 114 cues moved
at all, 77 of them holding a substituted character; the largest movers are exactly the lines
that previously had nothing to align against (three 嘿 cues by 3.3-13.0 s, 家駒 by 2.7 s), all
of which had been sitting on the `min_cue_duration` floor because they were interpolated
between their neighbours rather than placed. `isolated` rising by 3 is the right direction: those
lines went from invisible to the trellis to placed-but-in-a-wide-gap, which is what
`flag_unconstrained` exists to say.

`near` on top of `homophone` moves **8 cues**, by at most 1.0 s (median 0.12 s), and improves
the one hand-measured line it touches (摷摷摷。, -0.45 s → -0.17 s). It is left opt-in anyway:
tone is part of a Jyutping reading, eight cues is a thin basis for loosening that in general,
and a wrong-tone token can attract the trellis to the wrong frame in a way a missing token
cannot. This fixture simply had no such case.

> **pycantonese carries one reading per character, and occasionally an odd one.** 脅 resolved
> to 歉, which its data reads as hip3; 協 (also hip3, and the runner-up) is the better choice.
> That is what `--align_substitutions` is for -- the automatic tiers are a starting point to
> curate, not an oracle. A polyphone is likewise matched on whichever reading the data holds.

**This is not a `--realign` feature.** It lives in `align()`, so it applies to the ordinary
ASR path and to every `--realign` mode on exactly the same terms: every `load_align_model` call site
passes the level, and `align()` augments the dictionary from whatever transcript it was
handed. Only the *rate* differs — ASR output is 0.86 % out-of-vocabulary against the
transcript's 2.27 %, because Qwen's text has already been through OpenCC and the HK-variant
rules, so the Simplified forms are gone before alignment sees them. `homophone` resolves 43 of
those 62 distinct characters, `near` 52.

The failure it fixes is also different in shape, and milder: an ASR cue always has other
characters to align on, so a dropped one costs a word timing and can drag a subsegment
boundary, rather than stranding a whole line. It is still worth having. On `experiments/fixtures/bluey`, the
one substantive change between substitution off and on is a **recovered cue boundary**:

```
groundtruth  139.155-140.163  唉，噉𠸏？          140.225-142.480  點解我隻手指會喺個鼻度㗎？
off          138.373-142.578  哎噉嘅，點解我隻手指會喺個秘道㗎？        (one 4.2 s cue)
homophone    138.373-140.176  嘿，哎噉嘅           140.216-142.578  點解我隻手指會喺個秘道㗎？
```

嘿 had no token, so alignment could not place the clause it opens and cue assembly glued both
clauses together. With one substituted character the split reappears **within 0.02 s of the
hand-measured boundary**. Nothing else in the file changed (125 cues against 124).

> **A substituted character must never be a spot-check candidate**, and `align()` enforces it
> via `align_vocab.filter_spotchecks`. A spot-check asks the emission which of two
> interchangeable particles the audio supports (喇/啦, 咁/噉); that question only means
> something while each candidate's token carries its *own* acoustics. Worse, two candidates in
> one set substituting to the same token would score identically and the candidate weight
> would decide the rewrite alone — a silent text change driven by nothing. Every spot-check
> character in the Cantonese language pack (`languages/yue/__init__.py`) is natively in `alvanlii/wav2vec2-BERT-cantonese`'s vocabulary,
> so this never fires today; it is there so a future align model or profile cannot turn an
> acoustic reselection into a coin toss.

## Cue notes (utils/schema.py + utils/debug.py)

`SingleAlignedSegment.notes` is the general annotation channel: a list of `kind:detail`
strings any stage may append to with `schema.add_note`. `merge_segments` unions them, and
`debug.write_segment_notes` renders every annotated cue to `{debug_dir}/{stem}/notes/notes.srt`
on any run with a debug dir — load it beside the video and step through what happened.

It is deliberately **not** `realign_reason`, and the two must not be merged. A reason is a
positive statement that a *timing is not to be trusted* and drives `realign/suspect.srt`; a
note is informational and says only that something happened here. Notes are also far more
numerous — one substituted character can touch a hundred cues — so folding them into
`suspect.srt` would drown the handful of lines that actually need checking.
`debug.write_labelled_srt` is the renderer both go through.


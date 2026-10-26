# Missing lines: recovering a line from the reference and the audio

Machine-made subtitles drop lines: VAD misses quiet or overlapped speech, or the recogniser
returns nothing. The reference subtitle often still has a line there. The proofreader
(cantocaptions-proofread: `ccp missing`, `ccp listen --missing`, `ccp flag add --missing`)
is to notice such *orphan* reference lines, decide whether the draft really lacks the line,
and propose what was said. This page measures each part of that before it was built.
Harness: `scripts/eval_missing_lines.py`; outputs in `../cantocaptions-eval/missing/all/`.

## Test set

Two kinds of draft, each paired with a hand-made strict file of the same release (the ground
truth) and a Standard Chinese reference:

- **Fresh drafts**: 8 strict episodes transcribed with the default pipeline (Frieren E01-E04
  and Frozen are held-out shows; AoT S1E05, Spy x Family S1E01 and Maid Sama E01 are not).
- **Old AI files**: the library's `[CCAI]` files whose episode also has a hand-made file of the
  same release (Maid Sama E01-E03, Spy x Family E01, Moonlight Resonance E01, Doraemon 555-556).
  These are the "messy" files the proofreader meets.

An orphan is a reference cue, shifted by the measured offset, that draft cues cover for less
than 30% of its length. Each was labelled from the hand-made file:

| label | meaning | all | fresh | old AI |
|---|---|---|---|---|
| P | a spoken hand-made line is there and the draft lacks it | 105 | 24 | 81 |
| C | covered: the draft has the line, timed elsewhere | 26 | 9 | 17 |
| S | on-screen text or bracketed lyrics | 214 | 146 | 68 |
| N | no hand-made line there either | 153 | 70 | 83 |

- The current pipeline rarely drops a line: three Frieren episodes had one between them (their
  orphans are the bracketed lyrics of the Japanese songs). Old AI files drop far more
  (Moonlight 39, Doraemon 26).
- **Brackets separate on-screen text cleanly.** Of the 214 bracketed orphans, 208 are S. Of the
  284 unbracketed ones, 101 are real missing lines (36%), 153 are N (54%) and 24 are covered by a
  neighbour (8%).
- **N is mostly songs** the hand-made files leave out (openings and endings in unbracketed
  references, as in Maid Sama), with a few cries. So the proofreader is told to rule out
  "already covered" and "a song" by meaning before listening.
- The reference span ±0.3 s covers at least 90% of the hand-made line in 79% of the P cases,
  and ±1.0 s in 90%.

## Hearing the span

For each P case the span (the reference span ± pad, never past the neighbouring draft cues)
was transcribed by the fine-tuned model (ft) and stock Qwen3-ASR (qw), plain, after vocal
isolation (mbroformer), and with the reference line as ASR context (`labelled` template).
Usable = CER at most 0.2 against the hand-made line (punctuation stripped, variants folded).

| method | usable | exact | empty | held-out (17) | trained (88) |
|---|---|---|---|---|---|
| ft, 0.3 s | 39% | 29% | 19% | 24% | 42% |
| ft, 0.3 s, isolated | 43% | 29% | 20% | 35% | 44% |
| ft, 0.3 s, reference as context | 47% | 36% | 11% | 29% | 50% |
| ft, 1.0 s | 27% | 24% | 32% | 41% | 24% |
| ft, 1.0 s, context | 32% | 26% | 23% | 47% | 30% |
| qw, 0.3 s | 22% | 13% | 24% | 12% | 24% |
| qw, 0.3 s, context | 29% | 16% | 15% | 24% | 30% |
| best of all the above, per case (ceiling) | 60% | 46% | 6% | 53% | 61% |

- The 1.0 s rows are penalised for including the neighbours' speech, which a reader drops.
- Context helps the recognisers find words: ft +7.7 points overall, at least +10 on two shows
  (Spy x Family +22, Doraemon +12). It recited the Mandarin reference in 1-5% of cases. It must
  still never count as confirmation (it can recite).
- Isolation is mixed: +12.5 on Frozen, +11 on Spy x Family, -33 on AoT (n=6).
- No single hearing wins everywhere; the ceiling says each one holds pieces the others miss.

## Writing the line, and scoring candidates

A blind writer (an LLM given only the reference, the neighbouring cues and every hearing above,
never the hand-made line) wrote 1-3 candidates per case, the way the proofreader would.

| method | usable | exact | held-out | trained |
|---|---|---|---|---|
| the writer's first candidate | **59%** | 42% | 47% | 61% |
| most likely of all candidates under ft (teacher-forced, `asr_scoring.score_texts`) | 44% | 31% | 35% | 45% |
| same, on the isolated clip | 44% | 31% | 35% | 45% |
| most likely under qw | 25% | 14% | 18% | 26% |
| ft + qw summed, isolated | 31% | 20% | 24% | 32% |
| best candidate, per case (ceiling) | 68% | 53% | 53% | 70% |

- The written line beats the plain fine-tuned hearing by 20 points, and by at least 18 on four
  shows (Frozen +25, Spy x Family +33, Moonlight +18, Doraemon +27).
- **Teacher-forced scoring does not pick well.** It prefers the recognisers' own transcripts,
  the same way the evidence tools vouch for the models' own errors. So the proofreader gets no
  "score these candidates" step for missing lines.

### Grading the written line

Agreement between the written line and the *close, unhinted* hearings (plain or isolated,
0.3 s) predicts whether it is right. Two definitions of "agrees", at most one character in
five different:

| close hearings that agree | whole hearing: n / usable | hearing *contains* the line: n / usable |
|---|---|---|
| both recognisers | 35 / 86% | 42 / 86% |
| only ft | 20 / 75% | 18 / 67% |
| only qw | 3 / 33% | 4 / 50% |
| neither | 47 / 34% | 41 / 29% |

"Contains" lets the hearing have extra words around the line (the edit distance of the line
against the hearing's best-matching stretch): a close span often runs into a neighbour, or into
the next missing line when several are missing in a row, and the whole-hearing rule then
refused lines both recognisers clearly heard. It separates better (more lines in the top band
at the same 86%), so `ccp flag add --missing` uses it: fix when both recognisers' close hearings
contain the line, suggest when the fine-tuned one's does, otherwise query (the line as a
candidate for the reviewer).

## Timing the inserted line

`timing`: the writer's line for every P case of one episode is inserted where the proofreader
puts it (the reference span inside the gap between the neighbouring cues, at least 0.75 s), and
the file is realigned as the cleanup does (`--realign_mode adjust --realign_split off`). Errors
against the hand-made line, seconds; "text right" = the inserted line is usable:

| episode | lines | placed at the reference: start median / p90 / within 0.5 s | after the realign | end p90: reference / realign |
|---|---|---|---|---|
| Doraemon 555-556, text right | 16 | 0.22 / 1.01 / 81% | 0.08 / 1.52 / 75% | 0.44 / 1.16 |
| Moonlight E01, text right | 20 | 0.26 / 0.55 / 85% | 0.05 / 0.14 / 95% | 1.02 / 1.73 |
| Doraemon, all | 26 | 0.27 / 1.13 / 69% | 0.17 / 1.62 / 65% | 0.81 / 2.86 |
| Moonlight, all | 39 | 0.19 / 1.08 / 82% | 0.06 / 1.36 / 79% | 0.95 / 2.10 |

- The realign kept the cues 1:1 (443 in and out, 1001 in and out): the proofreader can carry its
  records onto the realigned draft by position (`ccp.pipeline.cue_mapping`).
- Starts improve a lot in the median and on Moonlight in the tail; Doraemon's tail is worse.
  Ends are later than the hand-made ones after the realign in both.
- A line with the wrong text is placed badly either way, one more reason a query-graded line
  is not inserted until the reviewer picks its text.
- Keeping the reference placement for the cues the realign doubts (`realign/suspect.srt`)
  helped Doraemon's tail (start p90 1.52 -> 1.13) and hurt Moonlight's (0.14 -> 0.55): no
  fallback was added. Inserted lines get the realign every other cue gets, and its doubts show
  up as `T` lines.

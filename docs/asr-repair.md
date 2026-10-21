# ASR repair

Some ASR chunks go wrong in ways a subtitle reader can't miss: a phrase repeated until the token
budget runs out, or invented Cantonese over speech in another language. ASR repair finds those
chunks from the first pass's own output and re-transcribes them. The code is in
`pipeline/asr_repair.py` (pure functions) and `stages.AsrRepairStage` (model loading,
checkpoint). The measurements come from `scripts/eval_hallucination.py`.

## What goes wrong

The fine-tuned model (`cantocaptions-cantonese-ASR`, a LoRA on Qwen3-ASR-1.7B) is Cantonese-only.
Triumph in the Skies E01 (98 chunks of up to 28 s) shows every failure the repair targets:

| Chunk | First pass | What the audio is |
|---|---|---|
| 85 (36:21) | `…一、二、三OK！嘿，艾爾！艾爾，堅尼…放咗手先，放�` (looped after the retry) | Cantonese, then Italian |
| 86–90 | empty | Italian (the old stock-model run transcribed it) |
| 93 (40:14) | Cantonese, some Italian, then `等下先而家先…` | Cantonese and Italian alternating; the end is "Grazie, grazie" |
| 96 (42:05) | `…我哋揀咗喇，`×16 `我哦` | Italian |
| 73, 82 | budget hit, recovered by the retry, but with invented text | foreign speech and indistinct background voices |

Why the old handling missed them:

- **The loop flags were lost.** `budget_hit` and `retried` were dropped at alignment, and the
  checkpoint couldn't say whether the retry looped too.
- **The processor hid loops.** Its `_detect_and_fix_repetitions` folds only runs of 20+ copies,
  and silently.
- **A budget that ran out cut text mid-character.** That left U+FFFD in the output.
- **Text cleaning couldn't fold chunk 96.** Its `repeated_speech_strict.toml` runs per cue,
  after cue assembly, by which time the loop was spread across six cues.

## Detecting loops

`find_loop` finds the run of one phrase repeated back to back that covers the most text. Two
details:

- A last copy without its separator counts.
- A cut-off part-copy at the end marks the run truncated.

Each copy's phase is chosen so that it ends on its separator. Read as `晒，唔該` copies,
`唔該晒，唔該晒，唔該晒` would leave a false truncated tail.

A chunk is a loop (`loop_verdict`) if any of these hold:

- the retry also spent the whole budget (`retry_budget_hit`);
- the processor folded a loop (`raw_text` differs from the text);
- the text ends in U+FFFD;
- a run has at least `asr_loop_min_copies` (5) copies over at least `asr_loop_min_chars` (24)
  characters;
- a truncated run has at least 3 copies.

The thresholds were chosen with `eval_hallucination.py scan` over 88 debug checkpoints (8,945
chunks, about 50 h of speech: Journey to the West, Triumph in the Skies, Court!, Doraemon,
Bluey and others):

- **Runs of 5 or more copies** fired on about 1 chunk per hour. Every one inspected was a loop
  (`燒死佢…`×8, `你個頭啦，你個手啦`×14, `大…`×13).
- **Runs of 3–4 copies** are mostly real speech: `拜拜，拜拜，拜拜`, `唔該晒`×3, `早晨`×4,
  `喂，我啊…`×4. That is why the default isn't lower.
- **gzip ratio > 2.4** found the same loops and nothing extra, so it is off by default
  (`LoopRule.max_compression`).
- **U+FFFD** ended 13 chunks, each one a run that hit the budget.

## Stopping loops early

A loop costs far more than its own chunk. Generation runs a batch until every row finishes, so
one looping chunk held all the others in its batch (12 by default) to the full 200-token
budget, and then its penalty retry did the same again.

`_asr_native._LoopStop` is a per-row `StoppingCriteria`. It ends a row as soon as its last
tokens are one pattern of up to 24 tokens repeated 5 times (over at least 24 tokens). Five
copies matches the text rule above, since real speech repeats a phrase three or four times. Rows
that already emitted an EOS are left alone, because their padding repeats too. A stopped row
counts as a loop (`budget_hit`), exactly like one that ran out of budget. `asr_loop_stop`
(default on) switches it.

Measured on E01's 98 chunks, transcribed alone:

| | Time | Text |
|---|---|---|
| Without | 182 s | — |
| With | 114 s | identical on all 93 non-looping chunks |

Only the 5 loops changed, and they now end on a short, visible repeat instead of a full budget's
worth.

## Re-transcribing (the ladder)

A looping chunk is cut at its deepest pause and each piece transcribed again by the run's own
model. A chunk whose Latin-script text sits beside Chinese (it changes language part way) is cut
the same way. A piece that still loops, or is still mixed, is cut again, up to `asr_split_depth`
(3). No piece is ever shorter than `asr_split_min_piece` (4 s).

Where `split_point` cuts, in order of preference:

1. the longest gap between VAD's own speech regions;
2. the longest quiet stretch in the audio;
3. the smoothed minimum.

A piece's generation budget floor is 64 tokens rather than 200, and the loop stop applies, so a
looping 4 s piece wastes little.

When the run will re-transcribe anything, the transcription stage doesn't free its model. It
hands the model to the repair stage (`RunContext.resident_asr`), which frees it after the ladder
and before loading the multilingual model. This saves one load per run. The hand-over happens
only when the repair stage is in the run's stage list, and `run_stages` frees the model at the
end in any case.

The pieces are joined back into **one segment per VAD chunk** with `asr.join_clauses`, because
`ensemble_texts` and `reference_texts` are indexed by segment. Alignment aligns the joined text
against the whole chunk, as before.

On E01, depth 3 was needed for chunk 93. At depth 2, one piece still held both Cantonese and
Italian, and the multilingual model *translated* its Cantonese line into Italian (`講乜嘢啊？
佢覺得我哋好登對` came out as `Che ha detto? Ha detto che eravamo molto d'accordo`). At depth 3
the chunk comes apart into Cantonese / Italian / Cantonese / (the "Grazie" piece).

## Checking the language

The fine-tune writes **no language label**. The base model's `language X<asr_text>` prefix was
trained out of it, hinted or not (`eval_hallucination.py rerun --probe_lid`). So language comes
from base Qwen3-ASR (`asr_foreign_model`) in automatic mode, asked about every piece of every
planned chunk.

Base Qwen3-ASR's label alone isn't reliable:

- It labelled E01 chunk 94 (Italian) "English".
- It labelled chunk 96 "English", while writing `Non mi va`.
- It labelled mixed chunks 85 and 93 "Chinese" and "Cantonese".
- It labelled an Italian chunk (69) "Portuguese" and wrote a Portuguese rendering.

So `heard_language` corrects an English, Chinese or Cantonese label from the reading's own
Latin-script text, using a function-word check (`latin_language`, about 20 words per
language). Where the run's own model already wrote the text in that language, its text names it.

Which chunks are asked (`chunk_reasons`):

- loops;
- chunks whose Latin text is not English (`foreign_text`), or that a label-writing main model
  (stock Qwen3-ASR) said were in another language (`foreign_label`);
- empty chunks of 2 s or more, but only in a file where some chunk already shows another
  language (`RepairSettings.empty_needs_foreign`). Across the 8 Cantonese-only episodes below,
  all 21 empty chunks checked came back unchanged (music, effects), each costing a load of the
  multilingual model for nothing. E01's empty Italian chunks (90, 94) are still found, because
  chunks 69 and 91–93 hold Italian text.

Mixed-script text alone is *not* a reason. E01 has 26 chunks of ordinary English code-switching,
and sending all of them to a second model would cost a lot for nothing.

### What a piece ends up saying (`combine`)

- **Native readings never replace or drop anything.** Native means `asr_native_languages`; for
  yue that is Cantonese, any "Chinese" label, and English. English heard over Chinese text is
  only noted (`asr_lang_mismatch`).
- **A reading that is mostly Chinese is only noted** (`asr_lang_mixed`). The piece changes
  language part way.
- **A foreign reading in another script replaces the run's own text only if that text is
  suspect.** Suspect means empty, still looping, or already in that language. Otherwise it is a
  conflict, and the piece is asked again with the run's language as a hint:
  - Told to expect Cantonese, base writes pure Italian as Italian or nothing (E01 91, 92, 94).
  - It writes any Cantonese it hears as Cantonese.
  - If the hinted reading holds Chinese, the run's text stays and the conflict is noted
    (`asr_lang_conflict`).
  - This guards against the translation failure above. It is also what replaced chunk 96's
    invented `我哋唔理佢，快啲走啊！` with `che non vi va basta andiamo`.
- **`asr_foreign` decides the outcome:**
  - `transcribe` (default): use the other language's text.
  - `drop`: an empty piece, so no cue.
  - `note`: keep the run's text and note it.
  - `none`: no check, and the model never loads.

## Measured

E01, end to end. The original run's output is in `output/`; the repaired one is from
`--load_debug_dir temp`.

| Chunk | After repair |
|---|---|
| 85 | `…準備，一、二、三，OK。` then `Hey, Aizer! Aizer, vieni! Poza, vieni!` |
| 90 | `Si può scattare una foto? La festa? … Fai la foto. Scatta. Dai, scatta.` (was empty) |
| 93 | `佢以為我係你女朋友` / `Che peccato, pensavo che era tua ragazza…` / `佢覺得我哋好登對，即係類似噉樣，多謝你噉講啊。` / `係，得喇…` (the last piece is still wrong, noted `asr_lang_mismatch`) |
| 94 | `Rosé, até. Il nasce sulle gambe blu, la ragazza più rossa…` (was empty) |
| 96 | `che non vi va basta andiamo…`, then the rest, which base heard no speech in |
| 73, 82 | unchanged (base heard Cantonese/Chinese too), noted `asr_loop:recovered by a retry` |

**False positives.** All flagged chunks in 8 Cantonese-only episodes (Journey to the West E10,
E15, E21, E26; Triumph E16, E22; Court! S01E07, E08) went through the full ladder and language
check:

- **31 chunks.** No Cantonese was replaced or dropped.
- **The 21 empty or quiet chunks** were unchanged. Base heard English humming, Chinese filler, or
  nothing.
- **All 10 loop chunks** came back without the loop.
- **The one foreign verdict** was a real Italian song (Triumph E16 #44).

**Fixtures.** Bluey and Doraemon produce byte-identical subtitles with repair on and off
(`score_subtitles.py diff`):

- **Doraemon:** nothing to repair.
- **Bluey:** one empty chunk checked and left alone.
- **CER, coverage, start error:** unchanged.

**Cost.** E01 end to end, before and after the loop stop and the model hand-over:

| | Before | After |
|---|---|---|
| Transcription | 2:44 | 2:06 |
| Repair | 1:09 | 0:34 (3 chunks re-transcribed, 18 checked) |
| Both together | 3:53 | 2:40 |

That is less than transcription alone took before repair existed. A Cantonese-only file with no
loops and no foreign text loads no extra model at all.

## Timing other-language text

The Cantonese align model can't follow Italian. It has no tokens for `é`, `ò`, `ù`, and its
emissions over Italian speech are noise. So the times it gave Italian words were guesses, and
the cue cutters trusted the gaps between them:

- `split_gapped_cues` believes a pause only when both sides were "heard", but measures that from
  Chinese characters alone, so Latin-only text passes unchecked.
- Pass E cut long cues at those invented pauses.
- Split pieces are never rejoined.

On E01 this gave cues like `Rosé` / `até` / `Il` and `Si` / `può`.

Spreading the words evenly over their stretch was tried first, and was worse than the CTC
guesses. So these stretches now go to a **word aligner** (`pipeline/word_align.py`):

1. Repair records `foreign_spans` for every piece kept as another language: its text as written
   into the chunk, its piece's start and end, and its language.
2. The alignment stage, before the run's own align model loads, collects every span across all
   files. It gives each span's audio and its words (`word_align.spoken_tokens`: a Latin run
   whole, any other letter on its own, the same split alignment makes) to the word aligner
   (`align_foreign_model`, default `Qwen/Qwen3-ForcedAligner-0.6B-hf`) in one batched pass, then
   frees it. The times go into `span["words"]`.
3. After CTC alignment, `asr_repair.place_foreign` gives each span's words those times, matched
   by their place in the chunk's text, and drops their `peak`. Cue edges come from spoken words
   only, since spaces and marks keep the CTC times.
4. Alignment splits a chunk only at the run's own punctuation, so cues are cut at each Latin
   sentence end (`.` `?` `!`), and at a comma only when a sentence is wider than a line. Cue
   assembly merges sentences back up to a line where they fit.

The CTC times stand for any of these:

- a span whose text can't be found (pre-alignment cleaning changed it);
- a span whose word count doesn't match;
- a span in a language the aligner doesn't cover;
- the setting `align_foreign_model none`.

**The word aligner is general.** `WordAligner` takes any audio, any words and a language;
families are looked up by checkpoint architecture (`WORD_ALIGNERS`, `word_aligner_family`). Today
only `foreign_spans` use it, but a language with no CTC model of its own could align with one
outright.

Qwen3-ForcedAligner covers Chinese, Cantonese, English, French, German, Italian, Japanese,
Korean, Portuguese, Russian and Spanish. It takes up to 5 minutes of audio per job, and answers in
80 ms steps. Measured on an RTX 3080 in float16:

| Job | Time | Notes |
|---|---|---|
| E01's 8 Italian stretches | under 1 s | plus about 8 s to load |
| All 88 chunks of E01 (30 min of audio) | 5.3 s | peak 3.6 GB at batch 8 |
| The same chunks, CTC alignment | about 40 s | for comparison |

So it would hold up as a file's main aligner. Batching changes its answers by a mean of 2 ms
(fp16 noise).

**Piles.** A word the aligner can't place gets the same instant as its neighbours, so four
words land at once where speech is sung, invented or too fast. `word_align.unpile` spreads each
such run from that instant towards the next timed word, at no more than 0.4 s a word. Spreading
to the next word found put `Chinese food? Sì, thank you` ten seconds late.

**Related fixes:**

- **Accented letters are part of the word.** `text_profiles.LATIN_WORD` now covers them
  (`À–ɏ`, `Ḁ–ỿ`); before, `può` was the two words `pu` and `ò`.
- **Two Latin pieces are spaced when joined.** `ScriptConfig.join` on an unspaced script now
  spaces two runs of Latin text (`him?` + `Naturally`), while Latin beside Chinese stays unspaced
  (`我嚟到cafe喇`).

**E01 against the previous CTC timings.** Where the earlier stock-model subtitles give a
reference, the aligner lands closer:

| Line | Aligner | CTC | Earlier subtitles |
|---|---|---|---|
| `Hey, Aizer!` | 36:29.7–36:30.4 | in a cue stretched over 36:29.8–36:32.9 | — |
| `Basta che prendi qua` | 39:25.6 | 39:24.1, merged with the next line | 39:25.9 |
| `Ragazzi vi girate` | 39:26.8 | 39:24.1, merged with the previous line | 39:26.6 |

It lands worse where the ASR text is wrong. Chunk 69's `Chinese food? Sì. Thank you`
was really "Chinese lady? Sì, okay, thank you", around 29:44–29:48. The aligner, made to find
words that weren't said, put them at 30:00.7; CTC kept them near 29:44.

## Known limits

- **Text invented over indistinct background voices isn't caught** (E01 82: `係啊！放低啲嘅歌。
  唔好再唱啦…`). The multilingual model hears Chinese there too. The chunk is only noted
  (`asr_loop:recovered by a retry`). Untried: a cross-check against the align model's own greedy
  CTC decode of the cue, which is already in memory at alignment. The CTC *score* is no
  per-cue trust signal for timing (see known-gotchas), so this would need its own measurement.
- **A word aligner can only place the words it is given.** Where the ASR text is wrong (E01
  69), the aligner times invented words somewhere, sometimes far from the real speech.
- **Pauses in Latin-only text are believed without evidence.** `split_gapped_cues` measures
  "heard" from Chinese characters only. With aligner times the gaps are mostly real, but a sung
  line can still be cut mid-phrase (`e la` / `cantina buia dove noi`).
- **`drop` drops a whole piece.** A piece mixing English and Italian (E01 69:
  `Just gone? … Chinese food? … Thank you`) loses its English too.

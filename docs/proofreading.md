# LLM proofreading

Design notes and measurements for `pipeline/proofread/`.

## LLM proofreading (pipeline/proofread/)

**Off unless the user turns it on, and that is a contract, not a default.** The library runs
offline once models are downloaded; this is the one stage that needs the network and spends
money. `proofread = none` ships in `presets/default.cfg` and must stay that way -- a user enables
it per run (`--proofread gemini`) or in their own `config/user.cfg`. `load_proofreader` returns
`None` for `none`, and the provider SDKs (the `proofread` extra) are imported lazily, so an
offline run never touches them. API keys come from the environment only
(`GEMINI_API_KEY` / `ANTHROPIC_API_KEY`); never read them from config, log them or write them out.

It runs inside `_merge_and_write` after cue assembly, cleaning and the time offset -- on the
cues the user will actually receive -- and before the writer. `--proofread_input` runs the same
`_proofread` on subtitle files read from disk instead (`transcribe.proofread_files`, no audio,
no models): the output is `{name}.proofread.{ext}`, so it can never overwrite its input, and
only edited cues are re-cleaned, since the rest is the user's finished text:

| piece | file | job |
|---|---|---|
| prompt | `prompt.md` + `render.py` | language-neutral template filled from a `ProofreadStandard`; the cue stream with the reference interleaved by time |
| providers | `providers.py` | Gemini / Anthropic request shapes, structured JSON output, timeouts, cost estimate |
| apply | `apply.py` | validate the answer and apply it in code |
| driver | `__init__.py` | chunking, hash-keyed checkpoint replay, cost ceiling, dry run, review files |

**Everything language-specific is data on the language pack.** `LanguagePack.proofreading` maps
names to `ProofreadStandard`s (language name, description, typical-error bullets, a
conventions Markdown file, a name-variation example, and a `foreign_register` of forms an
edit may not introduce); `default_proofreading` picks one. CantoCaptions is one standard for
`yue` (`languages/yue/proofreading.py`, conventions at `languages/yue/proofread/cantocaptions.md`),
not *the* standard: another can be added beside it and chosen with `--proofread_standard`, and
`--proofread_conventions` / `--proofread_prompt` replace either half without code. A pack with
no standard gets a neutral one and proofreads for meaning only. Keep `prompt.md` free of
language-specific examples; they belong in the standard.

**The model's output is never trusted wholesale.** The schema asks for proper-noun decisions
*first*, then per-cue edits, then flags (doubts it could not resolve). `apply_answer`:

- refuses edits to unknown cue ids and demotes edits below `--proofread_min_confidence` to flags;
- refuses an edit that *introduces* items of the standard's `foreign_register` -- for Cantonese,
  Standard Written Chinese forms copied from the reference (的, 這, 沒, or 既 where Cantonese says
  唔止). This is the main residual failure mode: the model "corrects" a line to the reference's
  wording even where that does not sound like the audio. The guard catches the register; it
  cannot catch a same-register paraphrase, which is why the prompt says to correct what was
  *said*;
- under `--proofread_particles protect` (the default), undoes any change to a clause-final run
  of the standard's `final_particles` -- a swap, an addition, a drop -- keeps the rest of the
  edit and the answer's punctuation, and records the particle change as a flag
  (`apply.protect_particles`). The prompt carries the matching rule (`final_particle_rule`);
  `allow` swaps it for the error description (`final_particle_errors`) and turns the guard off;
- refuses a pair of edits that **moves words between adjacent cues** (`apply.moves_text`: one
  cue loses words at the shared edge and its neighbour gains them), whole, as a flag. The
  prompt no longer offers moves and the schema has no `moved` type. Boundaries follow the
  audio's timing; on whole episodes moves were rarely needed, multiplied with smaller chunks (9
  pairs in one 7-chunk run against 1 sent whole), and judged per cue each read as two errors;
- applies the name table **file-wide** after all chunks (min 2 characters, longest first,
  word-boundary-aware for spaced scripts), since a name misheard once is usually misheard the
  same way everywhere;
- notes each edited cue `proofread:<types>`, so it shows in `notes/notes.srt`.

**Why particles are protected, measured.** Over every whole-episode run, the model's
particle changes were wrong about five times as often as right (≈15 wrong occurrences against
≈3 right), clustered on the rarer contractions (𠿪→㗎, 𠾵→喳, 𡁜→喎). On human-checked text
they were the largest single class of over-correction. A *limited* rule was also tried --
change only between common particles, only where the reference clearly marks the function
(了 → 喇, 吧 → 啦) -- and the model then made **no** particle changes at all, the same as an
outright ban for a longer prompt; the one such change seen in earlier runs (受死啊 → 受死啦 on a
reference 吧) was wrong. The ceiling being given up is real but out of reach: on one ASR episode
an oracle fixing only particles would remove ~9 % of the remaining character errors, and the
model recovered almost none of it. Tone is what separates them, so the right tool for
particles is acoustic (the alignment spot-checks), not a text model.

Edited cues go back through the cleaner and the noise drop, so a correction cannot reintroduce
a convention violation.

**Measured (approximate; Cantonese, strict human ground truth, Standard Chinese reference,
Gemini 3.7 Flash at medium thinking, one request per ~25-minute episode):** about a quarter of
the remaining character errors removed, ~0.9 precision, ~US$0.10-0.16 and 3-4 minutes per
episode, confirmed end to end through the pipeline. **`high` effort is not usable on a whole
episode**: the request had not returned after 25 minutes and was abandoned. Keep `medium`. Several heavier designs were measured and are *not* worth shipping; do not re-add them
without new evidence:

- **Acoustic arbitration** (asking the align model, or an ASR model, to score candidate fixes):
  the acoustic models share the ASR's mishearings, so they vetoed correct fixes about as often as
  wrong ones. The same goes for gating edits on `peak`.
- **Two-pass review / tool calls**: more tokens, no accuracy gain, and an agentic loop can run
  away with the budget.
- **Telling the model to defer to the reference**: no measurable effect; removed.

Without a reference subtitle, and on other languages, it has not been measured.

**Checkpointing.** Each request is keyed by a hash of its full text;
`{debug_dir}/{stem}/proofread/` keeps the request, the answer and the key, and a
`--load_debug_dir` replay reuses an answer only for a byte-identical request -- so a replay never
re-bills, and any upstream change re-asks. **Answers are saved even without a debug dir**, in
`{output_dir}/{name}.proofread/` (`run(save_dir=...)`, set by `_proofread`): they were paid for.
`--proofread_dry_run` writes the request(s) and cost estimate and sends nothing;
`--proofread_max_cost` (per request) refuses an over-estimate before *anything* is sent.

**Chunking** (`--proofread_chunk_cues N`, `--proofread_chunk_context` 6 read-only cues each
side). The first chunk is sent alone and the rest `--proofread_parallel` (4) at a time: the
first is what warms the cache the others read. On Gemini, a multi-chunk file gets an explicit
cache of the system prompt (`providers.open_cache`, deleted afterwards; implicit caching is
best-effort and misses requests that start together); Anthropic's system block carries
`cache_control` on every request. Measured on one ~1800-cue file in four 500-cue chunks: the
whole ~5.6k-token system prompt was a cache hit on every chunk, but that saves only ~half a cent
per chunk -- thinking (17-26k tokens per chunk at medium) is ~85% of the bill, so chunking is a
*latency* lever, not a cost one. A chunk whose request fails is skipped with a warning and the
others still apply; only when every request fails is the file written unproofread. Names are
collected from every chunk and applied file-wide at the end, as for a single request.


## Proofreading with `--realign`, and which timeline the reference is on

The stream interleaves each reference line after the cue it overlaps, so the reference and
the cues **must share a timeline**. Under `--realign` there are two (the input subtitle's and
the media's), and `reference_timing` says which one the reference follows. It defaults to
`media`, the timeline a reference has in every non-realign run; a reference timed like the
input subtitle needs `subtitle`. A wrong choice pairs every line with the wrong one, which is
why every proofread checks the pairing before it sends anything (below). Setting it to `None`
in a cfg brings back the old requirement to choose explicitly: validation then refuses the
combination.

| `reference_timing` | proofreading runs | on |
|---|---|---|
| `subtitle` | **before** realign (`transcribe._proofread_realign_input`) | the `--realign` input, on its own timings |
| `media` | after realign, as in an ASR run | the realigned cues |
| (no reference) | before realign, if the input has timings | the input |

`proofreads_before_realign` decides it. Proofreading first writes the corrected copy as
`{input stem}.proofread.srt` (review files in `{input stem}.proofread/`), then swaps
`cfg.realign` for that copy and sets `proofread = none` -- at the very top of
`_execute_pipeline`, before anything reads `cfg.realign`, so every stage and checkpoint sees
the copy and nothing is proofread twice. No cleaning there: the realign run cleans the whole
file afterwards, or (sync/adjust) deliberately does not. A bare transcript has no timeline to
pair a reference with, so `subtitle` is refused for one.

**Every proofread checks the pairing before it spends anything** (`Proofreader.
_check_reference_timing`): the share of reference cues starting within 0.5 s of a cue start.
Measured both ways (reference starts near a cue start, and cue starts near a reference start)
and the better kept, so a reference that also subtitles the opening song or splits lines finer
is not mistaken for a different timeline: 83-94 % on a shared timeline over five real pairs,
17-46 % (the chance level of dense dialogue) shifted by a second or more or timed to another
release. One-way, a correct reference with 44 song-lyric cues the draft lacked scored 68 %. Below 60 % the
request is refused (`ProviderError`, so the file is left unproofread with a warning), and the
error names the constant shift within ±10 s that would line it up, for `--reference_offset`.
Overlap is the wrong test and was measured: dialogue is dense enough that a reference 30 s
out still overlaps some cue 76-89 % of the time.

## Review output: `changes.srt` and Subtitle Edit bookmarks (`proofread/review.py`)

`changes.srt` (in the review folder) holds each edited cue as two lines -- the new text, then
`[was]` and the old text -- with the changed characters in `<font color>` (green new, red old),
character-level, so a one-character fix lights up one character.

Every written SRT/VTT that was proofread gets a Subtitle Edit bookmarks sidecar,
`<subtitle file>.SE.bookmarks`, in the same folder (the name SE loads automatically). Format
from `libse/Common/BookmarkPersistence.cs`: UTF-8 **with** BOM, `{"bookmarks":[` CRLF, then
`{"idx":N,"txt":"..."}` entries joined by commas, `]}` CRLF. `idx` is the cue's **0-based**
position in the file, one less than its SRT number; `txt` escapes `\` and `"` and writes line
breaks as `<br />`. Notes are `[was] <old text>` for a changed cue and `[flag] <note>` per flag,
joined by line breaks when a cue has both.

The indices come from the cue list *as written*: `_proofread` keys each note by the segment
object, not its position, because re-cleaning can drop an edited cue (reduced to noise) and
shift every later index (`review.bookmark_marks`). An existing bookmarks file is never
overwritten: the next free `<stem> (N)<ext>.SE.bookmarks` is used instead.

**Bookmarks always describe the file they sit beside.** In an ASR run, or `--realign` with
`reference_timing media`, proofreading is the last thing before the writer -- after cue
assembly's splits and merges, cleaning and the time offset -- so its indices are the written
file's. With `reference_timing subtitle` proofreading runs *before* realign, on the input's
cues, which realign may then drop (cut policy, noise drop, cleaning), split or merge: neither
index nor timing survives, but the text and its order do. `review.remap_marks` therefore
carries each note across by aligning the two texts character by character (letters and digits
only, so cleaning and punctuation changes do not matter): a note lands on the output cue
holding the first surviving character of its source cue, a cue none of whose text survived
lands on the cue holding the text just before it, and notes meeting on one cue are joined.
The corrected copy (`{stem}.proofread.srt`) keeps its own bookmarks too. Checked on a whole
episode realigned onto a different release, with 40 cues dropped and a pair merged on top:
57 of 57 notes on the expected cue.

An edit that empties a cue removes the cue (`_proofread`): written, a blank cue is one some
editors skip, which would put every later bookmark one line out.

## Basic cleaning before proofreading (`proofread_preclean`, default on)

Text this run does not clean in full -- an existing subtitle under `--proofread_input` or
`--realign` (proofread first), or a sync/adjust realign's output (which keeps the subtitle's
own text) -- is put through the language's **basic** cleaning before it is proofread:
`CleaningSpec.basic_manifest`, for Cantonese `rules/pipeline_basic.toml` = `punctuation.toml`
(full-width marks, spacing, ellipses, no space after a dialogue dash) + `chars_hk.toml` (the
standard character variants) + `trim` (no stray comma at either end of a line). Nothing that
rewrites words or layout: no clause commas, numerals, interjection/repeat removal or line
breaking, since in a finished subtitle those are the editor's decisions.

It runs on each display line separately (`transcribe._preclean_text`): `punctuation.toml`'s
first rule folds line breaks into spaces, which on a whole cue would merge a two-speaker
cue's lines. A line it would empty is left as it was. ASR output is already fully cleaned and
is not cleaned twice. Off with `proofread_preclean = False` or `--no_clean_text`. What it
changed goes to `precleaned.srt` (new over `[was]` old) beside the review files.

## The run plan

`Pipeline:` is logged once, before anything runs, and covers the whole run
(`transcribe._describe_run`): the proofread-first step if there is one, the stages, cue
assembly and cleaning, the end-of-run proofread if that is where it runs, and the write. Each
proofread step names its provider, model, effort and reference -- `no reference`, or the
file and which timeline it was declared on -- so a run that will proofread without the
reference it was meant to have says so before it spends anything. The proofread-first step
itself runs only after the plan is logged.

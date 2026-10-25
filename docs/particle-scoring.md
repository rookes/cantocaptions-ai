# Scoring alternative wordings (particles)

`QwenPipelineNative.score_alternatives` (`pipeline/asr_scoring.py`) asks a Qwen3-ASR model how
likely it finds each of a few alternatives at given spots of a transcript, given the audio:
喇 or 啦 at the end of a clause, say. It is not part of the pipeline. The proofreading tools
(cantocaptions-proofread, `ccp particles`) use it with the fine-tuned model to weigh every
clause-final particle of a draft, and the proofreading agent may rest a particle correction on
a clear preference.

## Method

This is the cache-per-spot design of cantocaptions-dataset's probe (`probe.py`,
`_continuation_logprobs`):

1. One teacher-forced forward over prompt + transcript + EOS, keeping the KV cache. The
   prompt is exactly the transcription prompt (`apply_transcription_request` with the
   language). A model that writes `language X<asr_text>` before its text (stock Qwen3-ASR)
   gets that preamble prefilled, as the official SDK forces a language. The fine-tuned model
   writes no preamble. Which kind a model is gets read once, from eight greedy tokens of its own.
2. For each spot, latest first (a cache only crops shorter): crop the cache to just before
   it, widen it to one row per alternative, and score `alternative + the next 4 transcript
   characters`. The shared tail makes the options explain the same following text. The
   audio encoder never runs again.
3. A spot inside a multi-character token (a particle welded to its comma) branches at that
   token's start, and the token's leading characters lead every continuation.

The proofreader scores each evidence group: its cues joined the way the model writes (no line
breaks or speaker dashes, a comma between cues unless one ends in punctuation), on the group's
padded audio. That matches how the dataset's training chunks are written. An episode's 207
particles took about 1.5 minutes, including the model load.

**Check against full-sequence scoring** (`scripts/eval_particle_scoring.py --verify 4`): the
difference between two wordings, cached-tail versus whole-sequence, agreed within 0.15 nats
on every pair checked. That is fp16 noise plus the tail cut at 4 characters.

## Measurements

`scripts/eval_particle_scoring.py` scores every clause-final particle (in a confusion set,
`cantocaptions-proofread/config/particles.toml`) of the dataset's held-out shows, using raw
audio (`*_raw.jsonl`, the way the proofreader hears an episode), against all its alternatives.
- **False alarm:** on the real transcript, another particle beats the transcript's by at least T.
- **Detection:** for each alternative a, as if the draft had a by mistake, the real particle
  beats a by at least T.

Fine-tuned model (`rookes/cantocaptions-cantonese-asr`, 2026-10-25):

| | dev (Re:Zero, Your Name) | test (Frieren, Frozen, Haikyu, War of the Genders) |
|---|---|---|
| slots / clips | 1988 / 471 | 2314 / 574 |
| real particle ranked first | 93.5% | 91.8% |
| T = 2: false alarms / detection | 2.2% / 95.0% | 2.3% / 94.0% |
| T = 3 | 1.2% / 91.4% | 0.95% / 90.0% |
| **T = 4** | **0.35% / 85.8%** | **0.35% / 83.5%** |
| T = 5 | 0.05% / 76.8% | 0.09% / 74.4% |

The proofreader calls a preference of at least 4 nats *clear* and at least 2 a *lean*
(`[model]` in particles.toml).

Pairs (how often the real particle beats the alternative; median margin):
- 喇 vs 啦: 92–93% (4.0); 啦 vs 喇: 94–96% (4.1). The hardest common pair, still mostly
  right.
- 啊 vs 吖/呀/喇: 99%; 㗎 vs 嘎/𠸏: 98–100%.
- **吖 vs 啊: 77–79% (2.3–2.5); 嘅 vs 㗎: 65% (1.0).** The model tells these apart poorly,
  so the proofreader never calls its preference between them clear (`weak_pairs`).

Most of the 15 slots at T = 4 are real ambiguity, not noise: 今晚見啦/喇,
唯有下次先話佢知喇/啦, 等陣吖/啊. Some may be slips in the transcript itself.

**Collocations** (`ccp.colloc`, mined from strict subtitles with the held-out shows left out):
- When the strongest notable pull in a clause gives its particle at least 75%, it is right
  81–83% of the time. Where the model's top choice agrees, they are right 98% of the time.
- The scan's own rule (`colloc.against`: a strong phrase whose particle dominates, the
  draft's rare and under half its usual rate) doubts 1.2–1.3% of correct particles and
  catches only 14–16% of simulated wrong ones.

So a collocation is a reason to look, and supporting evidence when the meaning already points
there. It is much weaker than the model as a detector.

Rows and reports: `../cantocaptions-eval/particles/{dev,test}/`.

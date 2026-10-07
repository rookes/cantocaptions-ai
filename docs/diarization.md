# Speaker diarization: how accurate it is, and why

Investigation of 2026-10-07. It asks how much of the diarization error comes from the
pyannote model and how much from the way this pipeline drives it. All figures come from
Cantonese film/TV/anime audio with verbatim subtitles, over 22 shows: live action, anime and
Western animation dubbed into Cantonese.

## How the pipeline uses diarization

`pipeline/diarize.py` runs `pyannote/speaker-diarization-community-1` once per VAD segment
by default (`diarize_scope = segment`). VAD segments run about 20–28 s. pyannote's own
pipeline:

1. Its segmentation model labels up to 3 local speakers in sliding 10 s windows, with a 1 s
   step.
2. It embeds each (window, local speaker) with a WeSpeaker ResNet34 trained on VoxCeleb.
3. It clusters those embeddings: AHC, then VBx in PLDA space.
4. It rebuilds speaker turns from the result.

`pipeline/speaker_assign.py` labels an aligned subsegment only when one speaker holds 70% of
its diarized time (`speaker_confidence`). Cue assembly then refuses to merge two
subsegments whose labels differ (`segmentation._same_speaker`).

**No cosine similarity is computed in our code.** The only comparison of embeddings happens
inside pyannote's clustering:
- **AHC:** centroid linkage at Euclidean distance 0.6 on unit vectors, which means cosine
  ≥ 0.82.
- **VBx:** refines the clusters (Fa 0.07, Fb 0.8).
- **Assignment:** each window's speaker goes to its nearest centroid by cosine.

## Measuring without speaker labels

The question the pipeline asks is local: do these two adjacent stretches share a voice? So
it is scored at boundaries, using what subtitles already say
(`cantocaptions_ai/utils/speaker_eval.py`):

| boundary | truth | where it comes from |
|---|---|---|
| change | different speakers | between the lines of a hyphen two-speaker cue (`-兩蚊` / `-哦，好嘅`), placed in time by forced alignment |
| same | one speaker | the two halves of a one-speaker cue, split at the clause mark nearest its middle |
| adjacent | unknown | two consecutive cues; reported as the share called "different" |

"Recall" below is the share of change boundaries the pipeline calls different. "False
split" is the same share on same boundaries; each one is a sentence split in two. Both sides
of a boundary must hold at least 0.3 s of aligned speech. The sample is 474 change, 1,173
same and 3,796 adjacent boundaries from 1,166 clips of 12–28 s. Clips that size stand in
for VAD segments.

## Findings

### 1. The pipeline almost never detects a speaker change

| decision source | change recall | false split | adjacent called different |
|---|---|---|---|
| **pipeline today (segment scope)** | **7.2%** | 0.5% | 8.6% |
| same, raw top speaker (no 0.7 threshold) | 9.7% | 0.7% | 9.0% |

- **Errors are one-sided.** Diarization is not noisy, it under-splits: 85% of true changes
  are labeled *same speaker*, not unknown.
- **The threshold isn't the cause.** Dropping the 70% dominance rule recovers only a few
  points.
- **Neither is the "only one speaker" outcome.** 72% of clips containing a change were
  diarized as two or more speakers, yet the two sides of the change still got the same label.

### 2. Most of the loss is in pyannote's segmentation and turn reconstruction, not in the harness

All the variants below are replayed offline from cached segmentation and embeddings. The
replay reproduces the live pipeline exactly.

| variant | change recall | false split |
|---|---|---|
| segment scope, VBx as shipped | 7.2% | 0.5% |
| VBx Fa 0.3 | 14.1% | 1.7% |
| VBx min_speakers = 2 | 15.0% | 2.4% |
| cosine AHC (average linkage) instead of VBx, best threshold | 11.6–13.9% | 0.7–1.4% |
| **no clustering**: segmentation model's local speakers only | **24.7%** | 1.8% |
| **file scope** (whole episode, 27 episodes), VBx as shipped | 15.9% | 2.6% |
| file scope, cosine AHC 0.3 | 17.7% | 1.6% |

- **Clustering settings.** Every setting lies on roughly the same recall/false-split
  trade-off; none is clearly better. Fa ≥ 0.1 at file scope fragments episodes into dozens
  of clusters. At Fa 0.3, and with cosine AHC at 0.4, it exceeds 127, which overflows
  pyannote's int8 cluster labels.
- **File scope** doubles recall, because each window is clustered against the whole
  episode. False splits also rise to 1.6–2.6%.
- **Local segmentation** (before any clustering) catches the most changes, but tops out at
  about a quarter of them.

### 3. Embeddings of sub-second speech are weak; that is the floor

Two crops from the same one-speaker cue are compared against crops from different shows,
using the pipeline's WeSpeaker embedding.

| crop length | EER | AUC | same-speaker cosine (median / 10th pct) | different-speaker cosine (median / 90th pct) |
|---|---|---|---|---|
| 0.5 s | 26.6% | 0.81 | 0.21 / 0.04 | 0.06 / 0.18 |
| 0.75 s | 16.4% | 0.91 | 0.29 / 0.11 | 0.06 / 0.18 |
| 1.0 s | 13.0% | 0.94 | 0.36 / 0.15 | 0.06 / 0.20 |
| 1.5 s | 9.9% | 0.95 | 0.41 / 0.18 | 0.06 / 0.19 |

- **The usable gap is narrow.** Even same-speaker pairs score far below the 0.82 cosine
  pyannote's AHC needs to merge. Speaker turns in dialogue are short: at change boundaries
  the shorter side is under 0.6 s in 55% of cases.
- **Comparing the two sides directly is no better.** As a change detector, a plain cosine
  between the sides scores AUC 0.82. At the pipeline's 0.5% false-split rate it catches
  5.7% of changes, the same trade-off as pyannote.
- **Vocal isolation (MelBand RoFormer) changes nothing:** every row is within noise.
- **PLDA scoring** instead of cosine changes nothing either.

### 4. Longer context is where the signal is: speaker profiles across a file or a series

**Doraemon, 16 episodes, 6,102 one-speaker cues.** Every cue of 1.5 s or more was clustered
together (average-link cosine 0.4, no labels). That gives 18 clusters of 20 or more cues,
holding 75% of those cues.

- **Series-spanning clusters.** The biggest clusters each span 13–16 episodes. Read by their
  lines, they look like the main cast. One is whiny and self-pitying ("我點解成日都係噉樣𠸏"),
  likely Nobita. One gives advice ("你努力啲練習下就可以㗎啦"), likely Doraemon. One is bossy
  ("要我講幾多次你哋先至明白㗎"), likely Gian. One says "我就請爹哋識得嘅designer", likely
  Suneo. A Shizuka-like voice, and a mother who appears in only 3 episodes (an older
  production run), round them out. The opening and ending songs form their own clusters.
- **The signatures are stable across episodes.** Split the episodes in half and cluster
  each half on its own. The two clusterings agree at ARI 0.87–0.93 over 5 random splits.
  Matching clusters' centroids sit at cosine 0.94–0.97; the next-best cluster is at
  0.51–0.67.
- **Short cues are where it fails.** A cue was counted as confidently attributed when its
  nearest character centroid (built from other episodes) scored at least 0.5 and beat the
  next by 0.15:

  | cue length | confidently attributed |
  |---|---|
  | 0.8–1.0 s | 30% |
  | 1.0–1.5 s | 44% |
  | 1.5–2.5 s | 66% |
  | ≥ 2.5 s | 72% |

- **What's still unmeasured:** the clusters' purity against true characters. That needs
  labels.

**Per-file context helps the boundary decision too.** Each side's embedding was replaced by
itself plus the mean of its 10 nearest one-speaker cues elsewhere in the same file (never
the same clip). That raises change-vs-same AUC from 0.82 to 0.86, and recall at 2% false
split from 23% to 31%.

### 5. A Chinese-trained embedding model helps modestly

Same boundaries, cosine between the two sides:

| embedding model | change vs same AUC | recall @1% / 2% / 5% false split | same-cue vs cross-show EER |
|---|---|---|---|
| pyannote (WeSpeaker ResNet34, VoxCeleb) | 0.823 | 12% / 23% / 37% | 13.6% |
| WeSpeaker ResNet34-LM, CN-Celeb (same architecture) | 0.841 | 14% / 23% / 41% | 9.4% |
| CAM++ (3D-Speaker, 200k Chinese speakers) | 0.839 | 17% / 23% / 38% | 11.5% |

The CN-Celeb model is a drop-in for pyannote's embedding class: same architecture and same
features. Load its `model_5.pt` into `WeSpeakerResNet34` with a `resnet.` key prefix.

### 6. Multi-speaker cues: what does and doesn't show it

Over 5,100 cues of 1.2 s or more (825 of them multi-speaker), each detector scored alone:

| detector | AUC | multi-speaker | one-speaker |
|---|---|---|---|
| segmentation model hears ≥ 2 local speakers | 0.70 | 51% | 9.5% |
| overlapped-speech frames (≥ 2 active), share of cue | 0.68 | 20% | 3% |
| min cosine between adjacent 0.6 s windows | 0.69 | mean 0.08 | mean 0.16 |
| subtitle characters per second | 0.68 | 6.7 | 5.3 |
| **final diarization puts ≥ 2 speakers in the cue** | 0.55 | 11% | 1.3% |

**Multi-speaker cues are fast exchanges with overlap.** The segmentation model sees the
second voice in half of them; the final diarization keeps it in only one in nine. The
clustering and exclusive reconstruction are discarding evidence the model had.

A logistic model of the first four features, cross-validated by show, reaches AUC 0.80. It
catches 32% of multi-speaker cues at 2% false positives.

### 7. Combining the cheap signals at boundaries

Logistic regression, cross-validated by show (no show in both train and test):

| signals | AUC | recall @1% / 2% / 5% false split |
|---|---|---|
| pipeline today | — | 7.2% at 0.5% |
| local segmentation only | 0.74 | 16% / 20% / 36% |
| kNN-in-file cosine | 0.85 | 18% / 24% / 47% |
| local segmentation + kNN cosine | 0.88 | 27% / 34% / 51% |
| **local segmentation + kNN cosine + CN-Celeb cosine** | **0.90** | **31% / 38% / 53%** |

## Answers to the original questions

- **Model or harness?** Mostly the model, at this audio's turn lengths. Embeddings of
  0.3–1 s of expressive, dubbed, music-backed speech overlap heavily between speakers.
  pyannote's segmentation model sees only a quarter of the changes.
  - **Harness losses on top of that:** per-segment scope roughly halves recall against file
    scope. Using only the final clustered, exclusive turns throws away local evidence the
    segmentation model had: 25% recall locally, 7% after clustering.
- **Are larger portions different from segment-level embeddings?** Yes, a lot. Whole cues
  of ≥ 1.5 s, pooled over a file or a series, form stable voice profiles; 0.5–1 s pieces do
  not. Boundary decisions improve when each side is compared through its neighbours in the
  file rather than directly.
- **How far to trust the pairwise comparison?** Not far for short spans: same-speaker
  cosine at 1 s has a median of 0.36 with a long low tail. No threshold separates
  speakers cleanly below about 1.5 s.
- **Do multi-speaker cues show a detectable pattern?** Yes, but only partly. They show
  overlapped speech and a second local speaker in pyannote's segmentation output, a faster
  text rate, and a dip in adjacent-window similarity. Combined: AUC 0.80.

## Recommendations

In order of expected value:

1. **For the merge gate, stop using final diarization labels.** Score each boundary between
   aligned subsegments directly, from:
   - the segmentation model's local speaker posteriors over the boundary (before
     clustering);
   - each side's embedding, compared through file-level kNN context;
   - optionally, a CN-Celeb ResNet34 embedding.

   Calibrate one threshold on `scripts/eval_diarization.py` output. Expect about 30% of
   speaker changes caught at about 1% false splits, against 7% at 0.5% today. Everything
   needed is already computed in pyannote's pipeline; the local posteriors come from the
   `segmentation` hook.
2. **If the gate stays on pyannote's turns,** consider `diarize_scope = file`. It doubles
   recall (7% → 16%) at a false-split cost of 0.5% → 2.6%. Measure it on the eval episodes
   first, because the false splits land as split sentences.
3. **Speaker labels (`speaker_labels`) should stay experimental** for anything short.
   Labels are achievable for recurring characters with series-level profiles built from
   cues of ≥ 1.5 s: 66–72% of such cues attribute confidently. A profile store per series,
   reused across episodes, is the shape that works. Per-episode, per-segment labels are not.
4. **Not worth pursuing:** VBx/AHC parameter tuning, PLDA vs cosine scoring, and running
   diarization on isolated vocals.
5. **Watch for padding when embedding variable-length clips.** Zero-padding into a batch
   changes WeSpeaker's output even with masked pooling. Batch only equal lengths, or embed
   one at a time.

## Tools

- `scripts/eval_diarization.py`: scores change, same and adjacent boundaries from any
  audio file plus verbatim subtitles that use the hyphen convention. Diarization comes from
  a run's `--debug_dir` checkpoint, or from pyannote run in file or segment scope.
  `--embeddings` adds the side-vs-side cosine separability; `--out` dumps every boundary.
- `scripts/probe_speaker_embeddings.py`: embeds every one-speaker cue across several files
  and clusters them together. It prints each cluster's file spread, cohesion and sample
  lines, for checking whether recurring voices come out as profiles.
- `cantocaptions_ai/utils/speaker_eval.py`: the boundary definitions and metrics, with
  unit tests in `tests/test_speaker_eval.py`.

## Caveats

- **Change labels.** They come from the subtitler and are placed by forced alignment. A
  tight cue end can squeeze the second speaker's characters, so a few change boundaries are
  misplaced by a few hundred ms. Inspecting the highest-cosine change boundaries turned up
  sung duets and similar voices, not obvious label errors.
- **Same boundaries are easier than reality.** They sit inside one utterance (same breath,
  same prosody), while real same-speaker boundaries between sentences are harder. Treat
  false-split rates as lower bounds.
- **Dubbing.** Hong Kong dubbing reuses a small pool of voice actors, so different
  characters, and cross-show "different speaker" pairs, sometimes share a voice.
- **Unvalidated clusters.** Doraemon cluster identities were read from the lines, not
  checked against labels.

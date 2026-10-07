"""Speaker-change detection at the boundaries cue assembly decides on.

Cue assembly (``pipeline/segmentation.py``) glues aligned subsegments back into cues and asks
one question of each join: do the two sides share a voice? The diarization stage answers it
indirectly -- cluster every 10 s window's speakers, rebuild turns, label each subsegment, and
compare labels -- and on Cantonese film/TV that catches about 7% of real speaker changes
(docs/diarization.md). The evidence is lost in the clustering, not missing from the audio.

This stage asks the question directly, per boundary between two adjacent subsegments. It
computes three signals and weighs two of them:

``local``    pyannote's *segmentation* model, before any clustering: within each 10 s window
             that holds both sides, how different the two sides' local-speaker activity is
             (1 - overlap of the two activity distributions), averaged over windows. It sees
             a change inside one window without having to decide who anyone is.
``cos``      cosine between the two sides' speaker embeddings (the same WeSpeaker network
             the diarization pipeline uses). Logged for debugging, not weighted: given
             ``knn_cos`` it added nothing.
``knn_cos``  the same cosine after each side is pulled towards its nearest neighbours among
             the file's other subsegments (outside its own VAD segment). A 0.5 s side is a
             noisy sample of a voice; its neighbours elsewhere in the episode are cleaner
             samples of the same voice, so comparing the neighbourhoods is steadier.

A logistic model (``ChangeModel``) turns ``local`` and ``knn_cos`` into a probability; at or
above the threshold the right-hand subsegment gets ``speaker_break``, which ``segmentation``
treats as a hard veto on joining it to its left neighbour. The model was fitted on
subtitle-derived boundaries from 22 shows (two-speaker hyphen cues vs. clause splits inside
one-speaker cues), cross-validated by show; the shipped threshold catches about 29% of
speaker changes while splitting about 1% of one-speaker sentences.

The feature arithmetic is pure numpy (``local_change``, ``knn_pull``, ``boundary_pairs``);
``SpeakerChangeScorer`` wraps the two pyannote networks.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.schema import SingleAlignedSegment, VadAudioSegment

logger = get_logger(__name__)

Span = Tuple[float, float]

# Where both networks come from: the diarization pipeline the repo already uses, so enabling
# this needs no new download beyond the gated model diarization itself needs.
DEFAULT_SOURCE_MODEL = "pyannote/speaker-diarization-community-1"

# A side shorter than this carries too little voice to judge; such a boundary is not scored,
# which cue assembly reads as "no objection to merging".
MIN_SIDE = 0.3
# Only this much of each side, nearest the boundary, is embedded: a long subsegment's far end
# says little about who is speaking at the join, and may already be someone else.
MAX_SIDE = 4.0
# Boundaries across a gap wider than this are not scored: cue assembly never merges across
# one (align_merge_distance and merge_gap are both well under a second).
MAX_GAP = 1.0
# Subsegments at least this long, elsewhere in the file, form the neighbour pool.
POOL_MIN = 1.0
KNN = 10
# What ``local`` reads as when no segmentation window holds both sides (a boundary spanning
# two VAD segments, or sides together longer than the window): no evidence either way.
LOCAL_UNKNOWN = 0.5


@dataclass(frozen=True)
class ChangeModel:
    """Logistic combination of the boundary features; ``probability`` is P(speaker change).

    ``cos`` is computed and logged but not weighted: given ``knn_cos`` it added nothing.
    """
    bias: float
    local: float
    knn_cos: float

    def probability(self, features: Dict[str, float]) -> float:
        z = self.bias + self.local * features["local"] + self.knn_cos * features["knn_cos"]
        return 1.0 / (1.0 + math.exp(-z))


# Fitted on 1,647 subtitle-derived boundaries (474 speaker changes) from 22 shows, with the
# features exactly as computed here; refit if any feature's definition changes. Cross-
# validated by show (no show in both train and test), AUC 0.877.
DEFAULT_MODEL = ChangeModel(bias=0.906, local=3.177, knn_cos=-5.495)
# Out-of-fold, this threshold split 1.0% of one-speaker clause boundaries and caught 28.5% of
# speaker changes (0.85: 0.5% / 21.5%; 0.73: 2.0% / 34.6%). The diarization label gate it
# replaces caught 7.2% at 0.5%.
DEFAULT_THRESHOLD = 0.80


# --- Pure feature arithmetic --------------------------------------------------------------

def _l2n(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def boundary_pairs(
    segments: Sequence[SingleAlignedSegment],
    *,
    min_side: float = MIN_SIDE,
    max_gap: float = MAX_GAP,
) -> List[Tuple[int, Span, Span]]:
    """Scorable boundaries as ``(index of the right subsegment, left span, right span)``.

    ``segments`` must be in time order (alignment emits them that way). A boundary is skipped
    when either side is shorter than ``min_side`` or the gap exceeds ``max_gap``.
    """
    out = []
    for i in range(1, len(segments)):
        left, right = segments[i - 1], segments[i]
        if right["start"] - left["end"] > max_gap:
            continue
        if left["end"] - left["start"] < min_side or right["end"] - right["start"] < min_side:
            continue
        out.append((i, (left["start"], left["end"]), (right["start"], right["end"])))
    return out


def trim_toward(span: Span, boundary: float, max_len: float = MAX_SIDE) -> Span:
    """The part of ``span`` within ``max_len`` of the boundary it touches."""
    start, end = span
    if end - start <= max_len:
        return span
    return (end - max_len, end) if end <= boundary + 1e-9 else (start, start + max_len)


def local_change(
    activity: np.ndarray,
    chunk_starts: np.ndarray,
    chunk_duration: float,
    left: Span,
    right: Span,
) -> float:
    """1 - overlap of the two sides' local-speaker activity, averaged over every window that
    wholly contains both sides; ``LOCAL_UNKNOWN`` if none does.

    ``activity`` is the segmentation model's binarized output, ``(chunks, frames, speakers)``,
    with chunk ``k`` covering ``[chunk_starts[k], chunk_starts[k] + chunk_duration)`` on the
    same timeline as the spans. Local speaker indices are only meaningful within one window,
    which is why each window is compared on its own.
    """
    num_chunks, num_frames, _ = activity.shape
    votes = []
    for k in range(num_chunks):
        c0 = chunk_starts[k]
        if c0 > left[0] + 1e-9 or c0 + chunk_duration < right[1] - 1e-9:
            continue

        def frames(span: Span) -> np.ndarray:
            f0 = int((span[0] - c0) / chunk_duration * num_frames)
            f1 = int((span[1] - c0) / chunk_duration * num_frames)
            return activity[k, max(0, f0):min(num_frames, f1)].sum(0).astype(float)

        a, b = frames(left), frames(right)
        if a.sum() == 0 or b.sum() == 0:
            continue
        votes.append(1.0 - float((a / a.sum()) @ (b / b.sum())))
    return float(np.mean(votes)) if votes else LOCAL_UNKNOWN


def knn_pull(side: np.ndarray, pool: np.ndarray, k: int = KNN) -> np.ndarray:
    """``side`` plus the mean of its ``k`` most similar pool vectors, renormalised.

    Inputs are unit vectors. An empty pool returns ``side`` unchanged.
    """
    if len(pool) == 0:
        return side
    sims = pool @ side
    top = pool[np.argsort(-sims)[:k]]
    return _l2n(top.mean(0) + side)


# --- Models -------------------------------------------------------------------------------

def _segment_index(vad_segments: Sequence[VadAudioSegment], span: Span) -> Optional[int]:
    """The VAD segment wholly containing ``span`` (with a little slack), or None."""
    for i, seg in enumerate(vad_segments):
        if seg["start"] - 0.05 <= span[0] and span[1] <= seg["end"] + 0.05:
            return i
    return None


def _crop(seg: VadAudioSegment, span: Span) -> np.ndarray:
    i0 = max(0, int(round((span[0] - seg["start"]) * SAMPLE_RATE)))
    i1 = min(len(seg["audio"]), int(round((span[1] - seg["start"]) * SAMPLE_RATE)))
    return np.asarray(seg["audio"][i0:i1], dtype=np.float32)


class SpeakerChangeScorer:
    """The segmentation and embedding networks of a pyannote diarization pipeline."""

    def __init__(self, segmentation, embedding, device: str):
        self.segmentation = segmentation
        self.embedding = embedding
        self.device = device

    @classmethod
    def load(cls, model_name: str = DEFAULT_SOURCE_MODEL, *, device: str = "cpu",
             device_index: int = 0, token: Optional[str] = None,
             model_dir: Optional[str] = None,
             batch_size: Optional[int] = 4) -> "SpeakerChangeScorer":
        import torch

        from cantocaptions_ai.utils.audio import resolve_device
        # See diarize.load_diarization for why this import path.
        from pyannote.audio.core.pipeline import Pipeline

        pipeline = Pipeline.from_pretrained(model_name, token=token, cache_dir=model_dir)
        if pipeline is None:
            raise RuntimeError(
                f"Could not load '{model_name}'. It is a gated model: accept its terms on "
                "huggingface.co and pass a token via --hf_token."
            )
        if batch_size is not None:
            # Same in-call VRAM cliff as diarization (see PipelineConfig.diarize_batch_size).
            pipeline.segmentation_batch_size = batch_size
            pipeline.embedding_batch_size = batch_size
        device = resolve_device(device, device_index)
        pipeline.to(torch.device(device))
        return cls(pipeline._segmentation, pipeline._embedding, device)

    def activity(self, segment: VadAudioSegment):
        """Binarized local-speaker activity over one VAD segment, on the file timeline:
        ``(activity, chunk_starts, chunk_duration)``."""
        import torch

        audio = np.asarray(segment["audio"], dtype=np.float32)
        output = self.segmentation(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE}
        )
        window = output.sliding_window
        starts = segment["start"] + window.start + window.step * np.arange(output.data.shape[0])
        return np.asarray(output.data), starts, float(window.duration)

    def embed(self, chunks: Sequence[np.ndarray], batch: int = 32) -> np.ndarray:
        """Unit-norm embeddings. Batches only equal-length chunks: zero-padding a shorter
        one into a batch changes what the network's convolutions see, even with masks."""
        import torch

        out = np.zeros((len(chunks), self.embedding.dimension), np.float32)
        by_length: Dict[int, List[int]] = {}
        for i, chunk in enumerate(chunks):
            by_length.setdefault(len(chunk), []).append(i)
        for _, idx in by_length.items():
            for b in range(0, len(idx), batch):
                sub = idx[b:b + batch]
                waveforms = torch.from_numpy(np.stack([chunks[i] for i in sub]))[:, None]
                out[sub] = self.embedding(waveforms)
        return _l2n(out)

    def features(
        self,
        vad_segments: Sequence[VadAudioSegment],
        boundaries: Sequence[Tuple[Span, Span]],
        pool_spans: Sequence[Span],
    ) -> List[Optional[Dict[str, float]]]:
        """``{local, cos, knn_cos}`` per boundary; None where a side has no audio."""
        seg_of_pool = [_segment_index(vad_segments, span) for span in pool_spans]
        keep = [i for i, s in enumerate(seg_of_pool) if s is not None]
        pool = self.embed([
            _crop(vad_segments[seg_of_pool[i]], pool_spans[i]) for i in keep
        ]) if keep else np.zeros((0, self.embedding.dimension), np.float32)
        pool_seg = np.array([seg_of_pool[i] for i in keep])

        located = []
        sides: List[np.ndarray] = []
        for left, right in boundaries:
            boundary = (left[1] + right[0]) / 2
            left_t, right_t = trim_toward(left, boundary), trim_toward(right, boundary)
            sl, sr = _segment_index(vad_segments, left_t), _segment_index(vad_segments, right_t)
            if sl is None or sr is None:
                located.append(None)
                continue
            a, b = _crop(vad_segments[sl], left_t), _crop(vad_segments[sr], right_t)
            if len(a) < MIN_SIDE * SAMPLE_RATE or len(b) < MIN_SIDE * SAMPLE_RATE:
                located.append(None)
                continue
            located.append((sl, sr, left_t, right_t, len(sides)))
            sides += [a, b]
        side_emb = self.embed(sides) if sides else np.zeros((0, self.embedding.dimension))

        activity_cache: Dict[int, tuple] = {}
        out: List[Optional[Dict[str, float]]] = []
        for entry in located:
            if entry is None:
                out.append(None)
                continue
            sl, sr, left_t, right_t, j = entry
            ea, eb = side_emb[j], side_emb[j + 1]
            if sl == sr:
                if sl not in activity_cache:
                    activity_cache[sl] = self.activity(vad_segments[sl])
                act, starts, duration = activity_cache[sl]
                local = local_change(act, starts, duration, left_t, right_t)
            else:
                local = LOCAL_UNKNOWN
            others = pool[(pool_seg != sl) & (pool_seg != sr)] if len(pool) else pool
            out.append({
                "local": local,
                "cos": float(ea @ eb),
                "knn_cos": float(knn_pull(ea, others) @ knn_pull(eb, others)),
            })
        return out


# --- Applying the decision ----------------------------------------------------------------

def score_segments(
    scorer: SpeakerChangeScorer,
    segments: List[SingleAlignedSegment],
    vad_segments: Sequence[VadAudioSegment],
    model: ChangeModel = DEFAULT_MODEL,
) -> List[Dict]:
    """Attach ``speaker_change`` -- P(the voice changes at this subsegment's start) -- to
    every subsegment whose boundary with its left neighbour is scorable, in place.

    Returns one record per scored boundary (time, probability, features) for the debug
    snapshot; the segments carry only the probability.
    """
    ordered = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    for seg in ordered:
        seg.pop("speaker_change", None)
    pairs = boundary_pairs(ordered)
    pool_spans = [
        (seg["start"], seg["end"]) for seg in ordered if seg["end"] - seg["start"] >= POOL_MIN
    ]
    feats = scorer.features(vad_segments, [(l, r) for _, l, r in pairs], pool_spans)
    records = []
    for (index, left, right), f in zip(pairs, feats):
        if f is None:
            continue
        p = round(model.probability(f), 4)
        ordered[index]["speaker_change"] = p
        records.append({
            "time": round(right[0], 3), "left": list(left), "right": list(right),
            "probability": p, **{k: round(v, 4) for k, v in f.items()},
        })
    return records


def mark_breaks(segments: List[SingleAlignedSegment], threshold: float) -> int:
    """Set ``speaker_break`` where ``speaker_change`` reaches ``threshold``; clear it elsewhere.

    Kept apart from scoring so a threshold change is pure arithmetic. Returns the count.
    """
    count = 0
    for seg in segments:
        seg.pop("speaker_break", None)
        p = seg.get("speaker_change")
        if p is not None and p >= threshold:
            seg["speaker_break"] = True
            count += 1
    return count

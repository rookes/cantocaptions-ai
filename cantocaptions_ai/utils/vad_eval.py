"""Score a VAD run against known speech: is it dropping speech, and what does it keep?

Two questions, scored separately because they are asked of different outputs:

**Detection** (``score_detection``) -- how well the binarized speech regions match the
reference, before any padding or chunking. Frame-level precision / recall / F1 on a 10 ms
grid, with the usual collars around each reference boundary left unscored, because a
boundary is only known to within a few frames. This characterises the *detector*: the
model plus its onset/offset thresholds.

**Coverage** (``score_coverage``) -- what the pipeline actually hands to ASR: the chunks
``merge_chunks`` emits, after padding, gap bridging and grouping. This is where dropped
speech is decided, so it is scored strictly and per subtitle line:

* ``speech_recall``  share of reference speech that lands inside a chunk.
* ``cues_clipped``   lines that lose at least ``clip_tolerance`` seconds of their speech --
                     typically a first or last syllable.
* ``cues_missed``    lines that lose more than half their speech -- the subtitle is gone.
* ``cues_split``     lines with a chunk edge inside them. Mostly min-cut splits, where
                     nothing is dropped but ASR sees the line in two halves; a clipped line
                     counts here too.
* ``kept_frac``      share of all audio sent to ASR -- the cost side, since ASR time scales
                     with it and every second of non-speech is a chance to hallucinate.
* ``nonspeech_kept_frac`` / ``gap_kept_frac``  of the audio known to hold no speech (all
                     of it / the pure-negative gap clips alone), the share sent to ASR.

The reference (see scripts/build_vad_testset.py) is subtitle cue timing. Cue onsets are
accurate; cue *ends* carry some trailing padding, so the tail of every cue is treated as
uncertain: excluded from coverage's "core" speech (a VAD that stops where the voice stops
must not be charged for the subtitle lingering) and collared in detection.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

RES = 0.01  # seconds per grid frame

Span = Tuple[float, float]


@dataclass
class Tolerances:
    # Detection collar around a reference onset, either side.
    onset_collar: float = 0.2
    # A cue end is late by up to this much: the last ``end_slack`` seconds of a cue (at most
    # half of it) are uncertain -- unscored in detection, outside the core in coverage.
    end_slack: float = 0.5
    # Detection collar after a reference end.
    end_collar_after: float = 0.2
    # Coverage: a line counts as clipped when this much of its core speech is lost.
    clip_tolerance: float = 0.1


@dataclass
class StreamReference:
    """One stream's ground truth. ``speech`` rows are (start, end, cue_id); a cue_id groups
    the pieces of one subtitle line (a line cut across two clips arrives as two rows)."""
    duration: float
    speech: Sequence[Tuple[float, float, object]]
    dontcare: Sequence[Span] = ()
    negatives: Sequence[Span] = ()  # pure-negative clips (the dataset's gap clips)
    tol: Tolerances = field(default_factory=Tolerances)

    def __post_init__(self):
        n = int(np.ceil(self.duration / RES))
        self.n = n
        t = self.tol
        speech = _raster([(s, e) for s, e, _ in self.speech], n)
        dontcare = _raster(self.dontcare, n)

        # Detection scoring mask: everything except collars and don't-care audio.
        unscored = dontcare.copy()
        for s, e, _ in self.speech:
            slack = min(t.end_slack, 0.5 * (e - s))
            _paint(unscored, s - t.onset_collar, s + t.onset_collar)
            _paint(unscored, e - slack, e + t.end_collar_after)
        self.det_scored = ~unscored
        self.det_ref = speech

        # Coverage: the per-line core speech, and the audio known to be speech-free.
        cues: Dict[object, List[Span]] = {}
        for s, e, cid in self.speech:
            slack = min(t.end_slack, 0.5 * (e - s))
            cues.setdefault(cid, []).append((s, e - slack))
        self.cue_ids = list(cues)
        self.cue_cores = [cues[c] for c in self.cue_ids]
        self.cue_bounds = [(min(s for s, _ in v), max(e for _, e in v)) for v in self.cue_cores]
        core = _raster([sp for v in self.cue_cores for sp in v], n)
        self.core = core
        # Speech-free = not speech, not the uncertain tail after a cue, not don't-care.
        tails = np.zeros(n, dtype=bool)
        for s, e, _ in self.speech:
            _paint(tails, s, e + t.end_collar_after)
        self.nonspeech = ~(tails | dontcare)
        self.negative = _raster(self.negatives, n)

    def score_detection(self, regions: Iterable[Span]) -> Dict[str, float]:
        hyp = _raster(regions, self.n)
        m = self.det_scored
        return {
            "tp": RES * float(np.sum(hyp & self.det_ref & m)),
            "fp": RES * float(np.sum(hyp & ~self.det_ref & m)),
            "fn": RES * float(np.sum(~hyp & self.det_ref & m)),
            "tn": RES * float(np.sum(~hyp & ~self.det_ref & m)),
        }

    def score_coverage(self, chunks: Sequence[Span]) -> Dict[str, float]:
        kept = _raster(chunks, self.n)
        csum = np.concatenate([[0], np.cumsum(kept)])
        clipped = missed = 0
        for spans in self.cue_cores:
            total = lost = 0.0
            for s, e in spans:
                a, b = _idx(s, self.n), _idx(e, self.n)
                total += b - a
                lost += (b - a) - (csum[b] - csum[a])
            total, lost = total * RES, lost * RES
            if total > 0 and lost > 0.5 * total:
                missed += 1
            elif lost >= self.tol.clip_tolerance:
                clipped += 1
        # A chunk edge strictly inside a line's span (not within 50 ms of its ends).
        edges = np.array(sorted({e for c in chunks for e in c}))
        split = 0
        for s, e in self.cue_bounds:
            i = np.searchsorted(edges, s + 0.05, side="left")
            if i < len(edges) and edges[i] < e - 0.05:
                split += 1
        return {
            "core_s": RES * float(np.sum(self.core)),
            "core_kept_s": RES * float(np.sum(self.core & kept)),
            "cues": len(self.cue_cores),
            "cues_clipped": clipped,
            "cues_missed": missed,
            "cues_split": split,
            "kept_s": RES * float(np.sum(kept)),
            "total_s": RES * self.n,
            "nonspeech_s": RES * float(np.sum(self.nonspeech)),
            "nonspeech_kept_s": RES * float(np.sum(self.nonspeech & kept)),
            "gap_s": RES * float(np.sum(self.negative)),
            "gap_kept_s": RES * float(np.sum(self.negative & kept)),
            "chunks": len(chunks),
        }


def summarize(detection: Sequence[Dict[str, float]], coverage: Sequence[Dict[str, float]]):
    """Pool per-stream counts (micro-average: every second and every line weighs the same)."""
    d = {k: sum(x[k] for x in detection) for k in ("tp", "fp", "fn", "tn")}
    c = {k: sum(x[k] for x in coverage) for k in coverage[0]} if coverage else {}
    out = {}
    if detection:
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        out.update(
            det_precision=p,
            det_recall=r,
            det_f1=2 * p * r / (p + r) if p + r else 0.0,
            det_false_alarm=d["fp"] / (d["fp"] + d["tn"]) if d["fp"] + d["tn"] else 0.0,
        )
    if coverage:
        out.update(
            speech_recall=c["core_kept_s"] / c["core_s"] if c["core_s"] else 0.0,
            cues=c["cues"],
            cues_clipped=c["cues_clipped"] / c["cues"] if c["cues"] else 0.0,
            cues_missed=c["cues_missed"] / c["cues"] if c["cues"] else 0.0,
            cues_split=c["cues_split"] / c["cues"] if c["cues"] else 0.0,
            kept_frac=c["kept_s"] / c["total_s"] if c["total_s"] else 0.0,
            nonspeech_kept_frac=c["nonspeech_kept_s"] / c["nonspeech_s"] if c["nonspeech_s"] else 0.0,
            gap_kept_frac=c["gap_kept_s"] / c["gap_s"] if c["gap_s"] else float("nan"),
            chunks=c["chunks"],
            mean_chunk_s=c["kept_s"] / c["chunks"] if c["chunks"] else 0.0,
        )
    return out


def _idx(t: float, n: int) -> int:
    return min(max(int(round(t / RES)), 0), n)


def _paint(mask: np.ndarray, s: float, e: float):
    a, b = _idx(s, len(mask)), _idx(e, len(mask))
    if b > a:
        mask[a:b] = True


def _raster(spans: Iterable[Span], n: int) -> np.ndarray:
    """Boolean frame mask of a set of (possibly overlapping) spans, built in O(n)."""
    delta = np.zeros(n + 1, dtype=np.int32)
    for s, e in spans:
        a, b = _idx(s, n), _idx(e, n)
        if b > a:
            delta[a] += 1
            delta[b] -= 1
    return np.cumsum(delta[:-1]) > 0

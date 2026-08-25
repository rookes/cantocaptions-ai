"""Reference-subtitle context: VAD expansion + per-segment ASR context strings.

A reference subtitle is a same-content, different-language cue list for the audio
being transcribed (typically a standard-Chinese or English SRT). This module turns
one into two independent signals, both pure functions over dicts -- no models:

1. ``expand_intervals_to_reference`` -- cues mark where speech *is*, so a cue lying
   outside every VAD region is evidence VAD dropped something. Union-ing those cue
   intervals into the VAD timeline recovers the region before audio is ever sliced.

2. ``build_segment_contexts`` -- Qwen3-ASR accepts free-form background text in the
   system prompt and uses it as a soft prior while decoding ("context biasing").
   Handing each VAD segment the reference cues that overlap it biases proper nouns,
   names and homophone-confusable content *before* an error is emitted, rather than
   repairing it afterwards the way ``llm_correction`` does.

``overlapping_reference_indices`` is the shared time-matcher; ``llm_correction``'s
``match_reference_to_segments`` is a thin wrapper over it, so there is exactly one
implementation of "which reference cues belong to this segment" in the codebase.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.schema import SingleSegment

logger = get_logger(__name__)


# Rendering styles for the context string handed to the ASR system prompt.
#
# Qwen3-ASR is trained to read the system prompt as *background knowledge*, not as an
# instruction (the technical report calls it "context tokens ... as background
# knowledge"), so a bare or lightly-labelled reference should bias better than an
# imperative wrapper, which spends tokens the model was not trained to act on. That is
# a hypothesis, not a finding -- Qwen publishes no guidance on context format at all --
# so all three are selectable via --asr_context_template and measurable with
# scripts/eval_asr_context.py.
#
# "none" renders the empty string for every segment, which makes it the *control* for
# the whole feature rather than a fourth style: the reference still expands the VAD
# timeline, but no segment carries a context, so ``_infer_batch`` takes the ordinary
# ``apply_transcription_request`` path and the decode is byte-identical to a run with no
# reference at all. It exists to answer "is the measured gain the prompt, or just the
# extra speech VAD expansion recovered?" -- see "Measured results" in CLAUDE.md.
CONTEXT_TEMPLATES: Dict[str, str] = {
    "none": "",
    "bare": "{text}",
    "labelled": "參考翻譯：{text}",
    "instruct": "Use the following translation of this audio to assist transcription: {text}",
}

DEFAULT_CONTEXT_TEMPLATE = "labelled"


def shift_cues(cues: Sequence[SingleSegment], offset: float) -> List[SingleSegment]:
    """Shift every reference cue by *offset* seconds.

    A reference subtitle is frequently sourced separately from the audio (a different
    release, or OCR'd from burned-in subs) and can carry a constant offset. Both consumers
    are sensitive to it: expansion pads the wrong spans, and per-segment context attaches
    cues to the wrong segment near a boundary. A constant shift is the cheap fix; for a
    drifting or scene-by-scene offset use ``--retime``, which realigns acoustically.

    Cues are clamped at zero and any that end up entirely before the start are dropped.
    """
    if not offset:
        return list(cues)
    out: List[SingleSegment] = []
    for cue in cues:
        end = cue["end"] + offset
        if end <= 0:
            continue
        out.append({**cue, "start": max(0.0, cue["start"] + offset), "end": end})
    return out


# ---------------------------------------------------------------------------
# Shared time matching
# ---------------------------------------------------------------------------

def overlapping_reference_indices(
    segments: Sequence[dict],
    reference: Sequence[SingleSegment],
    fallback_window: float = 2.0,
) -> List[List[int]]:
    """Return, per segment, the indices of the reference cues that belong to it.

    A cue belongs to a segment when the two intervals overlap. When nothing overlaps
    and ``fallback_window`` is positive, the single nearest cue by midpoint distance is
    used if it lies within that many seconds -- a fallback tuned for cue-sized segments,
    which is why callers matching against whole VAD chunks pass ``fallback_window=0``.
    """
    out: List[List[int]] = []
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        overlapping = [
            i for i, r in enumerate(reference)
            if r["start"] < seg_end and r["end"] > seg_start
        ]
        if overlapping:
            out.append(overlapping)
            continue
        if not reference or fallback_window <= 0:
            out.append([])
            continue
        seg_mid = (seg_start + seg_end) / 2
        nearest = min(
            range(len(reference)),
            key=lambda i: abs((reference[i]["start"] + reference[i]["end"]) / 2 - seg_mid),
        )
        ref_mid = (reference[nearest]["start"] + reference[nearest]["end"]) / 2
        out.append([nearest] if abs(ref_mid - seg_mid) <= fallback_window else [])
    return out


# ---------------------------------------------------------------------------
# 1. VAD expansion
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: List[List[float]], touching: bool = True) -> List[List[float]]:
    """Sort and coalesce intervals.

    ``touching=True`` also joins intervals that merely abut (``a.end == b.start``),
    which is what a *coverage* test wants. ``touching=False`` requires a strict overlap
    and is what building the output timeline wants: ``Vad.merge_chunks`` deliberately
    emits touching chunks where ``Binarize._split_long`` cut a long run at its
    lowest-scoring (quietest) frame, and coalescing those would throw that choice away
    and re-split at an arbitrary point -- re-segmenting stretches of the file where no
    reference cue was added at all.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        overlaps = start <= merged[-1][1] if touching else start < merged[-1][1]
        if overlaps:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _subtract_intervals(
    minuend: List[List[float]], subtrahend: List[List[float]]
) -> List[List[float]]:
    """``minuend`` minus ``subtrahend``, both as interval lists. Sorted and disjoint."""
    keep = _merge_intervals(minuend, touching=True)
    cut = _merge_intervals(subtrahend, touching=True)
    out: List[List[float]] = []
    for start, end in keep:
        cursor = start
        for c_start, c_end in cut:
            if c_end <= cursor:
                continue
            if c_start >= end:
                break
            if c_start > cursor:
                out.append([cursor, min(c_start, end)])
            cursor = max(cursor, c_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append([cursor, end])
    return out


def padded_cue_spans(
    cues: Sequence[SingleSegment],
    padding: float = 0.5,
    upper_bound: Optional[float] = None,
) -> List[List[float]]:
    """Each cue widened by *padding* on both sides, clamped to ``[0, upper_bound]``.

    This is exactly the span set ``expand_intervals_to_reference`` unions into the VAD
    timeline; it is factored out so ``expansion_only_spans`` cannot drift from it.
    """
    spans: List[List[float]] = []
    for cue in cues:
        start = max(0.0, cue["start"] - padding)
        end = cue["end"] + padding
        if upper_bound is not None:
            end = min(end, upper_bound)
            start = min(start, end)
        if end > start:
            spans.append([start, end])
    return spans


def expansion_only_spans(
    intervals: Sequence[dict],
    cues: Sequence[SingleSegment],
    padding: float = 0.5,
    upper_bound: Optional[float] = None,
) -> List[List[float]]:
    """The spans the reference *added* -- padded cues minus what VAD already had.

    This is the provenance behind ``--asr_context_scope expanded``: audio inside one of
    these spans is audio the ASR would never have seen without the reference subtitle,
    so it is the only audio for which the reference is telling the model something the
    acoustic model did not already find on its own.

    Returns sorted, disjoint spans on the file timeline. Empty when every cue already
    sat inside a VAD region -- which is the honest answer, not a failure: nothing was
    recovered, so under ``expanded`` scope nothing gets a context prompt either.
    """
    if not cues:
        return []
    base = [[iv["start"], iv["end"]] for iv in intervals]
    return _subtract_intervals(padded_cue_spans(cues, padding, upper_bound), base)


def _uncovered_duration(start: float, end: float, covered: List[List[float]]) -> float:
    """Seconds of [start, end) not overlapped by any interval in *covered* (sorted, disjoint)."""
    total = max(0.0, end - start)
    for c_start, c_end in covered:
        if c_start >= end:
            break
        overlap = min(end, c_end) - max(start, c_start)
        if overlap > 0:
            total -= overlap
    return max(0.0, total)


# Shortest piece a budget split may produce, in seconds.
_MIN_PIECE = 1.0


def _free_zones(start: float, end: float, protected: List[List[float]]) -> List[List[float]]:
    """Sub-intervals of [start, end] covered by nothing in *protected* (sorted, disjoint)."""
    zones: List[List[float]] = []
    cursor = start
    for p_start, p_end in protected:
        if p_end <= start:
            continue
        if p_start >= end:
            break
        if p_start > cursor:
            zones.append([cursor, min(p_start, end)])
        cursor = max(cursor, p_end)
        if cursor >= end:
            break
    if cursor < end:
        zones.append([cursor, end])
    return [z for z in zones if z[1] > z[0]]


def _inter_cue_gaps(start: float, end: float, cues: Sequence[SingleSegment],
                    min_gap: float = 0.1) -> List[List[float]]:
    """Gaps between consecutive reference cues lying inside [start, end].

    Second-tier split candidates: the reference says no dialogue line spans these, even
    where VAD believes there is sound (music, laughter, overlapping effects).
    """
    gaps = []
    ordered = sorted(cues, key=lambda c: c["start"])
    for a, b in zip(ordered, ordered[1:]):
        gap_start, gap_end = a["end"], b["start"]
        if gap_end - gap_start < min_gap:
            continue
        lo, hi = max(gap_start, start), min(gap_end, end)
        if hi - lo >= min_gap:
            gaps.append([lo, hi])
    return gaps


def _score_at(t: float, times: "np.ndarray", scores: "np.ndarray") -> float:
    """VAD speech probability nearest time *t*."""
    import numpy as np
    return float(scores[int(np.clip(np.searchsorted(times, t), 0, len(scores) - 1))])


def _best_score_split(lo: float, hi: float, times, scores,
                      cues: Sequence[SingleSegment]) -> Optional[float]:
    """Lowest-scoring frame in [lo, hi], preferring points no dialogue line spans.

    The same idea as ``Binarize._split_long``'s min-cut -- cut where the model is least
    confident there is speech -- with an added penalty for landing inside a reference
    cue, so an equally quiet frame outside a dialogue line wins.
    """
    import numpy as np
    if times is None or scores is None or len(times) == 0:
        return None
    i0, i1 = int(np.searchsorted(times, lo)), int(np.searchsorted(times, hi))
    i0, i1 = max(0, i0), min(len(scores), max(i1, i0 + 1))
    if i1 <= i0:
        return None
    window = scores[i0:i1].astype(float).copy()
    inside = np.zeros(len(window), dtype=bool)
    for cue in cues:
        if cue["end"] < lo or cue["start"] > hi:
            continue
        seg = times[i0:i1]
        inside |= (seg >= cue["start"]) & (seg <= cue["end"])
    window[inside] += 1.0          # any frame outside a cue beats any frame inside one
    return float(times[i0 + int(np.argmin(window))])


def _split_one(start: float, end: float, chunk_size: float,
               protected: List[List[float]], cues: Sequence[SingleSegment],
               times=None, scores=None, report: Optional[list] = None) -> List[List[float]]:
    """Split [start, end] so every piece fits *chunk_size*, cutting where it hurts least.

    The cap is honoured unconditionally -- ``--chunk_size`` is the caller's decision and
    the ASR input budget -- so this always splits an over-long interval. What varies is
    *where*, tried in order of how little it damages:

    1. a point covered by neither a VAD region nor a reference cue (pure padding);
    2. a gap between two consecutive cues (no dialogue line spans it);
    3. the lowest-scoring VAD frame in the window, preferring frames outside any cue.

    The split must land in ``[start + chunk_size/2, start + chunk_size]`` so the leading
    piece both fits the budget and stays at least half of it -- the same guarantee
    ``Binarize._split_long`` gives. Each chosen point is appended to *report* with the VAD
    score there, so callers can tell whether a genuinely quiet cut was available.
    """
    if end - start <= chunk_size:
        return [[start, end]]

    lo, hi = start + chunk_size / 2, min(start + chunk_size, end - _MIN_PIECE)
    if hi <= lo:
        lo, hi = start + (end - start) / 2, min(start + chunk_size, end)

    def within(spans):
        out = []
        for a, b in spans:
            a2, b2 = max(a, lo), min(b, hi)
            if b2 > a2:
                out.append((a2 + b2) / 2)
        return out

    tier = None
    point = None
    free = within(_free_zones(start, end, protected))
    if free:
        tier, point = "padding", min(free, key=lambda c: abs(c - hi))
    else:
        gaps = within(_inter_cue_gaps(start, end, cues))
        if gaps:
            tier, point = "cue gap", min(gaps, key=lambda c: abs(c - hi))
        else:
            point = _best_score_split(lo, hi, times, scores, cues)
            tier = "min-score"
            if point is None:
                tier, point = "midpoint", (lo + hi) / 2

    point = min(max(point, lo), hi)
    if report is not None:
        report.append((point, tier,
                       _score_at(point, times, scores) if times is not None else None))
    return [[start, point]] + _split_one(point, end, chunk_size, protected, cues,
                                         times, scores, report)


def _split_to_budget(intervals: List[List[float]], chunk_size: float,
                     protected: List[List[float]], cues: Sequence[SingleSegment],
                     times=None, scores=None, report: Optional[list] = None) -> List[List[float]]:
    """Apply the chunk_size budget to every interval."""
    if chunk_size <= 0:
        return intervals
    out: List[List[float]] = []
    for start, end in intervals:
        out.extend(_split_one(start, end, chunk_size, protected, cues, times, scores, report))
    return out


def _touching_boundaries(intervals: List[List[float]]) -> List[float]:
    """Points where two input intervals abut exactly -- VAD's own min-cut split points."""
    ordered = sorted(intervals, key=lambda iv: iv[0])
    return [b[0] for a, b in zip(ordered, ordered[1:]) if abs(a[1] - b[0]) < 1e-9]


def _reinstate_boundaries(intervals: List[List[float]], points: Sequence[float]) -> List[List[float]]:
    """Re-cut *intervals* at each of *points* that falls strictly inside one."""
    out: List[List[float]] = []
    for start, end in intervals:
        cuts = sorted(p for p in points if start < p < end)
        cursor = start
        for p in cuts:
            out.append([cursor, p])
            cursor = p
        out.append([cursor, end])
    return out


def expand_intervals_to_reference(
    intervals: Sequence[dict],
    cues: Sequence[SingleSegment],
    padding: float = 0.5,
    chunk_size: float = 30.0,
    upper_bound: Optional[float] = None,
    times=None,
    scores=None,
) -> List[dict]:
    """Union reference-cue timings into a VAD timeline.

    **Every** cue is included, widened by *padding* on each side, and unioned with the VAD
    regions -- the output covers a span if *either* source claims it. There is no
    confidence gate: the reference is a human-authored record of where dialogue is, and a
    cue VAD happens to agree with costs nothing to include twice, while a cue VAD missed is
    exactly the speech that is otherwise unrecoverable.

    Returns ``[{'start', 'end'}, ...]``, sorted and disjoint. Disjointness is load-bearing:
    alignment's ``_find_vad_segment_idx`` returns the *first* segment containing a
    timestamp, so overlapping segments would silently orphan the second one's words.

    ``chunk_size`` is honoured unconditionally -- it is the caller's setting and the ASR
    input budget. When a unioned run exceeds it, ``_split_one`` picks the least damaging
    cut available, falling back to the lowest-scoring VAD frame when no silent gap exists;
    pass *times*/*scores* (the VAD probability curve) to enable that. Extra keys on the
    input dicts are dropped, since an output interval may be the union of several inputs
    plus a cue.
    """
    base = [[iv["start"], iv["end"]] for iv in intervals]
    if not cues:
        # Nothing to union in: hand VAD's own boundaries back untouched (sorted, but never
        # coalesced or re-split), so no region is moved that no cue went near.
        return [
            {"start": iv["start"], "end": iv["end"]}
            for iv in sorted(intervals, key=lambda x: x["start"])
        ]

    padded = padded_cue_spans(cues, padding, upper_bound)

    # A union merges spans that merely touch -- [0,5] u [5,12] is [0,12] -- so that a cue
    # abutting a VAD region does not leave a boundary mid-utterance.
    merged = _merge_intervals(base + padded, touching=True)
    # But VAD's own touching boundaries are Binarize._split_long min-cuts, placed at the
    # quietest frame; keep the ones no cue crosses, or the union would silently
    # re-segment stretches of the file the reference never touched.
    keep = [b for b in _touching_boundaries(base)
            if not any(p_start < b < p_end for p_start, p_end in padded)]
    merged = _reinstate_boundaries(merged, keep)
    # Never cut through acoustic speech (VAD) or a dialogue line (an unpadded cue).
    protected = _merge_intervals(base + [[c["start"], c["end"]] for c in cues])
    report: list = []
    result = _split_to_budget(merged, chunk_size, protected, cues, times, scores, report)

    added = sum(e - s for s, e in merged) - sum(e - s for s, e in _merge_intervals(base))
    logger.info(
        "Reference subtitle unioned %d cue(s) (padding %.2fs): %d VAD segment(s) -> %d, "
        "%.1fs of audio added",
        len(padded), padding, len(base), len(result), added,
    )
    # Surface every cut that had to go through speech, with how loud it was there --
    # a high score means no quiet point existed anywhere in the window.
    for point, tier, score in report:
        if tier in ("padding", "cue gap"):
            logger.debug("Budget split at %.2fs (%s)", point, tier)
        else:
            logger.warning(
                "Budget split at %.2fs fell inside speech (%s, VAD score %.2f): no silent "
                "point existed within the chunk window", point, tier,
                score if score is not None else float("nan"),
            )
    return [{"start": s, "end": e} for s, e in result]


# ---------------------------------------------------------------------------
# 2. Per-segment ASR context
# ---------------------------------------------------------------------------

def _truncate_at_cue_boundary(texts: List[str], joiner: str, max_chars: int) -> str:
    """Join *texts*, dropping whole trailing cues rather than cutting one mid-sentence.

    A cue chopped mid-way is worse than useless as biasing text -- it can bias toward a
    word fragment. Only a single over-long cue is cut hard, since there is nothing
    shorter to fall back to.
    """
    if max_chars <= 0:
        return joiner.join(texts)
    kept: List[str] = []
    length = 0
    for text in texts:
        addition = len(text) + (len(joiner) if kept else 0)
        if length + addition > max_chars:
            break
        kept.append(text)
        length += addition
    if not kept:
        return texts[0][:max_chars] if texts else ""
    return joiner.join(kept)


def build_segment_contexts(
    segments: Sequence[dict],
    cues: Sequence[SingleSegment],
    neighbours: int = 0,
    template: str = DEFAULT_CONTEXT_TEMPLATE,
    max_chars: int = 400,
    joiner: str = " ",
    restrict_to_spans: Optional[Sequence[Sequence[float]]] = None,
) -> List[str]:
    """Build one ASR context string per segment from the reference cues it covers.

    Each segment gets the cues that overlap it, widened by *neighbours* cues either
    side in cue order. Segments with no overlapping cue get ``""`` -- no nearest-cue
    fallback, because a VAD chunk that overlaps nothing is genuinely uncovered and a
    wrong context is worse than none.

    The result is capped at *max_chars*: context is a prefill cost paid for every
    segment in every batch, and it inflates the KV-cache high-water mark that
    ``_asr_native._warn_vram`` estimates.

    *restrict_to_spans* (``--asr_context_scope expanded``) narrows the selection to cues
    that overlap one of the given spans -- in practice ``expansion_only_spans()``, the
    audio the reference recovered that VAD never found. A segment VAD detected on its
    own then gets no context at all, and a segment that is *partly* recovered gets only
    the cues covering the recovered part. Filtering happens *after* neighbour widening,
    so ``--asr_context_neighbours`` cannot pull in a cue over audio VAD already had.
    """
    try:
        pattern = CONTEXT_TEMPLATES[template]
    except KeyError:
        raise ValueError(
            f"Unknown context template {template!r}; expected one of {sorted(CONTEXT_TEMPLATES)}"
        ) from None

    spans = _merge_intervals([list(sp) for sp in restrict_to_spans], touching=True)         if restrict_to_spans is not None else None

    def _eligible(cue: SingleSegment) -> bool:
        if spans is None:
            return True
        return any(cue["start"] < end and cue["end"] > start for start, end in spans)

    matches = overlapping_reference_indices(segments, cues, fallback_window=0.0)
    contexts: List[str] = []
    for indices in matches:
        if not indices:
            contexts.append("")
            continue
        first = max(0, indices[0] - neighbours)
        last = min(len(cues) - 1, indices[-1] + neighbours)
        texts = [
            cues[i]["text"]
            for i in range(first, last + 1)
            if cues[i]["text"] and _eligible(cues[i])
        ]
        if not texts:
            contexts.append("")
            continue
        contexts.append(pattern.format(text=_truncate_at_cue_boundary(texts, joiner, max_chars)))
    return contexts

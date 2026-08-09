"""Assembly of aligned subsegments into displayable subtitle cues.

Alignment deliberately over-splits: ``alignment.py`` cuts each ASR chunk at every
punctuation mark so the Viterbi pass can place a pause at each clause boundary, emitting one
subsegment per clause. Those fragments have to be glued back into cues that read naturally,
which is what this module does.

Four passes run in order over one file's segments:

  A. **Adjacency merge** — the classic rule: join neighbours that are touching and whose join
     boundary is punctuation-clean, up to a single-line character budget.
  B. **Noise drop** — discard sub-threshold cues whose text is pure interjection noise, before
     the rescue pass can glue them onto a neighbour.
  C. **Short-cue rescue** — CTC forced alignment is peaky, so a discourse marker sitting inside
     a continuous speech run ("嗱，", "喂，") collapses to a handful of frames and ends up as its
     own 40 ms cue. Those fragments are merged into whichever neighbour reads best, with the
     punctuation rule relaxed from a hard gate to a ranking signal.
  D. **Duration floor** — whatever is still too short to read, and had no neighbour to join,
     gets its end extended into the following silence.

Everything is a pure function over segment dicts: no models, no I/O, no config objects beyond
the plain value types from ``cantonese/text.py``. Set ``min_cue_duration=0`` to disable
passes B-D entirely.
"""

from typing import Callable, List, Optional, Sequence

from cantocaptions_ai.cantonese.text import (
    DEFAULT_PUNCTUATION,
    DEFAULT_SEGMENTATION,
    MAX_CHARS,
    PunctuationConfig,
    SegmentationConfig,
    boundary_is_mergeable,
    is_mergeable,
)
from cantocaptions_ai.utils.schema import SingleAlignedSegment, merge_segments

# Alignment writes cue ends as round(next_start - align_padding, 3), so a pair of touching
# cues is exactly one padding apart *by construction* -- but the rounded floats don't subtract
# cleanly (311.564 - 311.524 == 0.040000000000020464), which put the gap a hair over the
# threshold and blocked ~half of all intended merges. Compare with a tolerance.
GAP_EPS = 1e-6

# Trailing marks stripped when matching a cue against a profile's leading_markers, so that
# "嗱，" and "嗱。" both match the token "嗱". Covers the default split chars plus the tildes
# the ASR uses for drawn-out interjections ("嚇～").
_TOKEN_TRIM = "，。？！；…～~ "


def _duration(seg: SingleAlignedSegment) -> float:
    return seg["end"] - seg["start"]


def _text(seg: SingleAlignedSegment) -> str:
    return seg["text"].strip()


def _same_speaker(seg1: SingleAlignedSegment, seg2: SingleAlignedSegment) -> bool:
    """True when merging won't destroy a speaker attribution.

    Segments only carry ``speaker`` when diarization ran; if either side lacks a label there
    is nothing to contradict, so the merge is allowed.
    """
    spk1, spk2 = seg1.get("speaker"), seg2.get("speaker")
    if spk1 is None or spk2 is None:
        return True
    return spk1 == spk2


def _adjacency_merge(
    segments: Sequence[SingleAlignedSegment],
    punctuation: PunctuationConfig,
    align_merge_distance: float,
    align_padding: float,
    max_chars: int,
    leading_markers: frozenset,
    min_cue_duration: float,
) -> List[SingleAlignedSegment]:
    """Pass A: greedily join touching neighbours with a clean join boundary.

    One exception: a too-short cue that is a known *leading* marker is not absorbed backwards
    here. This pass accumulates strictly left to right, so it would otherwise glue "嗱，" onto
    the end of the preceding sentence before anything got to weigh the other direction. Held
    back, the marker reaches the rescue pass, which prefers to join it forwards onto the
    clause it introduces -- and still joins it backwards if forwards turns out impossible.
    """
    threshold = align_merge_distance - align_padding
    merged: List[SingleAlignedSegment] = []

    for segment in segments:
        if not merged:
            merged.append(segment)
            continue

        prev = merged[-1]
        gap = segment["start"] - prev["end"]
        defer = (
            _duration(segment) < min_cue_duration
            and _text(segment).strip(_TOKEN_TRIM) in leading_markers
        )
        if (
            not defer
            and gap <= threshold + GAP_EPS
            and _same_speaker(prev, segment)
            and is_mergeable(_text(prev), _text(segment), punctuation, max_chars=max_chars)
        ):
            merged[-1] = merge_segments(prev, segment)
        else:
            merged.append(segment)

    return merged


def _drop_noise(
    segments: Sequence[SingleAlignedSegment],
    is_noise: Callable[[str], bool],
    min_cue_duration: float,
) -> List[SingleAlignedSegment]:
    """Pass B: drop sub-threshold cues that are pure interjection noise.

    Runs before the rescue pass so a bare "哦，" is removed outright rather than glued onto the
    front of its neighbour's line. Only short cues are considered -- a noise word held for a
    full second is a deliberate beat and stays.
    """
    return [
        seg for seg in segments
        if not (_duration(seg) < min_cue_duration and is_noise(_text(seg)))
    ]


def _rescue_short_cues(
    segments: Sequence[SingleAlignedSegment],
    punctuation: PunctuationConfig,
    segmentation: SegmentationConfig,
    min_cue_duration: float,
    merge_gap: float,
    rescue_max_chars: int,
) -> List[SingleAlignedSegment]:
    """Pass C: merge each too-short cue into whichever neighbour reads best.

    For every cue under ``min_cue_duration`` both join directions are considered. A direction
    is *admissible* on gap, combined length and speaker agreement; punctuation is deliberately
    not a gate here, since the whole point is to rescue fragments stranded behind a full stop.
    Among admissible directions the choice is, in order:

    1. **Punctuation** -- the join whose boundary character reads cleanly wins, even if it is
       the farther neighbour.
    2. **Direction** -- a cue that is itself one of the profile's ``leading_markers`` joins
       forwards, because those markers introduce the clause that follows them ("嗱，你知啦"
       reads as one thought; "…嘅感覺，嗱，" strands the marker on the wrong sentence).
    3. **Distance** -- otherwise the smaller gap decides.

    Merging makes the result longer, so a rescued cue stops being a candidate; the loop
    therefore terminates. It restarts after each merge because indices shift.
    """
    cues = list(segments)
    tokens = frozenset(segmentation.leading_markers)

    merged_any = True
    while merged_any:
        merged_any = False

        for i, cue in enumerate(cues):
            if _duration(cue) >= min_cue_duration:
                continue

            # Known discourse markers get a wider window: they are reliably part of the
            # neighbouring utterance even across a slightly longer pause.
            is_marker = _text(cue).strip(_TOKEN_TRIM) in tokens
            gap_limit = merge_gap * 2 if is_marker else merge_gap

            candidates = []
            for left_idx in (i - 1, i):
                if left_idx < 0 or left_idx + 1 >= len(cues):
                    continue
                left, right = cues[left_idx], cues[left_idx + 1]
                if not _same_speaker(left, right):
                    continue
                if len(_text(left)) + len(_text(right)) > rescue_max_chars:
                    continue
                gap = right["start"] - left["end"]
                if gap > gap_limit + GAP_EPS:
                    continue
                rank = 0 if boundary_is_mergeable(_text(left), punctuation) else 1
                # left_idx == i means this cue is the left member, i.e. joining forwards.
                direction = 0 if (is_marker and left_idx == i) else 1
                candidates.append((rank, direction, gap, left_idx))

            if not candidates:
                continue

            candidates.sort()
            left_idx = candidates[0][-1]
            cues[left_idx:left_idx + 2] = [merge_segments(cues[left_idx], cues[left_idx + 1])]
            merged_any = True
            break

    return cues


def _apply_duration_floor(
    segments: Sequence[SingleAlignedSegment],
    min_cue_duration: float,
    align_padding: float,
) -> List[SingleAlignedSegment]:
    """Pass D: extend the remaining too-short cues into the silence that follows them.

    Only the cue's ``end`` moves. Word and character timings are left alone as alignment
    ground truth, matching how text cleaning treats them. A cue with no room to grow is left
    exactly as it was rather than being allowed to overlap its neighbour.
    """
    cues = list(segments)

    for i, cue in enumerate(cues):
        if _duration(cue) >= min_cue_duration:
            continue
        wanted = cue["start"] + min_cue_duration
        if i + 1 < len(cues):
            wanted = min(wanted, cues[i + 1]["start"] - align_padding)
        if wanted > cue["end"]:
            cue["end"] = round(wanted, 3)

    return cues


def assemble_cues(
    segments: Sequence[SingleAlignedSegment],
    *,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    segmentation: SegmentationConfig = DEFAULT_SEGMENTATION,
    align_merge_distance: float = 0.08,
    align_padding: float = 0.04,
    min_cue_duration: float = 0.5,
    merge_gap: float = 0.25,
    max_chars: int = MAX_CHARS,
    rescue_max_chars: Optional[int] = None,
    is_noise: Optional[Callable[[str], bool]] = None,
) -> List[SingleAlignedSegment]:
    """Turn aligned subsegments into displayable cues (passes A-D; see module docstring).

    ``punctuation`` and ``segmentation`` come from the ASR model's profile. ``max_chars`` caps
    an ordinary adjacency merge (one subtitle line); ``rescue_max_chars`` caps a short-cue
    rescue and defaults to ``max_chars`` -- callers pass the full multi-line budget
    (``max_line_width * max_line_count``) so a stranded marker can attach to a sentence that
    will simply be broken over two lines. ``is_noise`` decides pass B and is skipped when None.

    ``min_cue_duration=0`` reduces this to pass A alone.
    """
    if not segments:
        return []

    if rescue_max_chars is None:
        rescue_max_chars = max_chars

    # Deferring leading markers in pass A is only safe when the rescue pass runs to place
    # them; with passes B-D off, pass A must behave exactly as it always has.
    leading_markers = frozenset(segmentation.leading_markers) if min_cue_duration > 0 else frozenset()

    cues = _adjacency_merge(
        segments, punctuation, align_merge_distance, align_padding, max_chars,
        leading_markers, min_cue_duration,
    )

    if min_cue_duration > 0:
        if is_noise is not None:
            cues = _drop_noise(cues, is_noise, min_cue_duration)
        cues = _rescue_short_cues(
            cues, punctuation, segmentation, min_cue_duration, merge_gap, rescue_max_chars,
        )
        cues = _apply_duration_floor(cues, min_cue_duration, align_padding)

    return cues

"""Assembly of aligned subsegments into displayable subtitle cues.

Alignment deliberately over-splits: ``alignment.py`` cuts each ASR chunk at every
punctuation mark so the Viterbi pass can place a pause at each clause boundary, emitting one
subsegment per clause. Those fragments have to be glued back into cues that read naturally,
which is what this module does.

Five passes run in order over one file's segments:

  A. **Adjacency merge** — the classic rule: join neighbours that are touching and whose join
     boundary is punctuation-clean, up to a single-line character budget.
  B. **Noise drop** — discard sub-threshold cues whose text is pure interjection noise, before
     the rescue pass can glue them onto a neighbour.
  C. **Short-cue rescue** — CTC forced alignment is peaky, so a discourse marker sitting inside
     a continuous speech run ("嗱，", "喂，") collapses to a handful of frames and ends up as its
     own 40 ms cue. Those fragments are merged into whichever neighbour reads best, with the
     punctuation rule relaxed from a hard gate to a ranking signal.
  E. **Long-cue split** — a cue still longer than ``max_cue_duration`` (one clause the ASR
     never punctuated, typically) is cut at its best internal boundary: the longest pause
     between two of its timed words or characters, favouring punctuation and the middle.
  D. **Duration floor** — whatever is still too short to read, and had no neighbour to join,
     gets its end extended into the following silence.

Passes A and C also refuse a join that would make a cue longer than ``max_cue_duration``, so
pass E only ever sees a cue that alignment produced long.

Everything is a pure function over segment dicts: no models, no I/O, no config objects beyond
the plain value types from ``text_profiles.py``. Set ``min_cue_duration=0`` to disable
passes B-D entirely.
"""

from typing import Callable, List, Optional, Sequence

from cantocaptions_ai.text_profiles import (
    DEFAULT_PUNCTUATION,
    DEFAULT_SCRIPT,
    DEFAULT_SEGMENTATION,
    PunctuationConfig,
    ScriptConfig,
    SegmentationConfig,
    boundary_is_mergeable,
    is_mergeable,
)
from cantocaptions_ai.pipeline.align_checks import InternalGap, _split_one
from cantocaptions_ai.pipeline.speaker_assign import speaker_scope
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.schema import SingleAlignedSegment, merge_segments

logger = get_logger(__name__)

# Alignment writes cue ends as round(next_start - align_padding, 3), so a pair of touching
# cues is exactly one padding apart *by construction* -- but the rounded floats don't subtract
# cleanly (311.564 - 311.524 == 0.040000000000020464), which put the gap a hair over the
# threshold and blocked ~half of all intended merges. Compare with a tolerance.
GAP_EPS = 1e-6

# Trailing marks stripped when matching a cue against a profile's leading_markers, so that
# "嗱，" and "嗱。" both match the token "嗱". Covers the default split chars plus the tildes
# the ASR uses for drawn-out interjections ("嚇～").
_TOKEN_TRIM = "，。？！；…～~ "


# How much closer (seconds of sounded pause) a short fragment must sit to its next neighbour
# than to its previous one before pass A holds it back for the rescue pass to place.
FORWARD_PAUSE_MARGIN = 0.05


def _sounded_bounds(seg: SingleAlignedSegment) -> Optional[tuple]:
    """(start of the first, end of the last) *sounded* timed token of *seg*, or None.

    A cue's own start and end include punctuation, which alignment maps to blank and which
    therefore dwells across the pause beside it -- so two clauses joined by "，" touch at the
    cue edges whatever the silence between them. The pause a listener hears is between the
    last spoken character of one and the first of the next.
    """
    tokens = seg.get("chars") or seg.get("words") or []
    timed = [t for t in tokens
             if t.get("start") is not None and t.get("end") is not None
             and str(t.get("char", t.get("word", ""))).strip(_TOKEN_TRIM + "、：；「」『』（）")]
    if not timed:
        return None
    return timed[0]["start"], timed[-1]["end"]


def _sounded_pause(left: SingleAlignedSegment, right: SingleAlignedSegment) -> float:
    """Seconds of pause between *left*'s last spoken token and *right*'s first.

    Falls back to the gap between the cue edges when either side has no timed tokens.
    """
    a, b = _sounded_bounds(left), _sounded_bounds(right)
    if a is None or b is None:
        return right["start"] - left["end"]
    return b[0] - a[1]


def _duration(seg: SingleAlignedSegment) -> float:
    return seg["end"] - seg["start"]


def _text(seg: SingleAlignedSegment) -> str:
    return seg["text"].strip()


def _within_cap(left: SingleAlignedSegment, right: SingleAlignedSegment,
                max_cue_duration: float) -> bool:
    """Whether joining *left* and *right* keeps the cue within ``max_cue_duration`` (0: no cap)."""
    return not max_cue_duration or right["end"] - left["start"] <= max_cue_duration + GAP_EPS


# Longest cue text shown in a veto log line before it is elided.
_VETO_PREVIEW_CHARS = 14


class _MergeVetoLog:
    """Boundaries where the speaker gate blocked a merge that would otherwise have happened.

    Only merges that were admissible on every *other* count are recorded, so a line here
    always means "the speaker gate (diarization labels or a speaker-change mark) is the reason
    these two cues stayed separate" and never "punctuation would have kept them apart anyway".

    Keyed by boundary time because pass C rescans the whole cue list after each merge, so
    the same stranded cue is examined many times; blocked pairs never merge, so the boundary
    time is a stable identity for one veto.
    """

    def __init__(self):
        self._by_boundary = {}

    def record(self, left: SingleAlignedSegment, right: SingleAlignedSegment) -> None:
        self._by_boundary[right["start"]] = (left, right)

    @staticmethod
    def _preview(segment: SingleAlignedSegment) -> str:
        text = _text(segment)
        if len(text) > _VETO_PREVIEW_CHARS:
            text = text[:_VETO_PREVIEW_CHARS] + "\u2026"
        if segment.get("speaker") is not None:
            return f"[{segment.get('speaker')}] {text}"
        return text

    def report(self) -> None:
        """Log one line per held boundary, plus a count. Silent when nothing was blocked."""
        if not self._by_boundary:
            return
        # Imported lazily to keep this module's import graph to text_profiles + utils.schema.
        from cantocaptions_ai.utils.output import format_timestamp

        for boundary in sorted(self._by_boundary):
            left, right = self._by_boundary[boundary]
            if right.get("speaker_break"):
                change = right.get("speaker_change")
                source = "Speaker change" + (f" (p={change:.2f})" if change is not None else "")
            else:
                source = "Diarization"
            logger.info(
                "%s held a cue boundary at %s: %s | %s",
                source,
                format_timestamp(boundary, always_include_hours=True, decimal_marker=","),
                self._preview(left),
                self._preview(right),
            )
        held = len(self._by_boundary)
        logger.info(
            "The speaker gate kept %d cue %s from merging",
            held, "boundary" if held == 1 else "boundaries",
        )


def _same_speaker(seg1: SingleAlignedSegment, seg2: SingleAlignedSegment) -> bool:
    """True when merging won't join two voices, as far as anything upstream can tell.

    Two independent signals can say no:

    * ``speaker_break`` on the right-hand side, from the speaker-change stage
      (``pipeline/speaker_change.py``): the boundary at its start was scored as a change of
      voice. Merging keeps the left side's keys, so a merged cue carries the mark of its own
      first subsegment, which is always the one that faces its left neighbour -- the mark
      stays attached to the right boundary however the pieces were glued.
    * differing ``speaker`` labels, from diarization. Segments only carry ``speaker`` when
      diarization ran; if either side lacks a label there is nothing to contradict, so the
      merge is allowed. Under ``--diarize_scope segment``, labels are namespaced per VAD
      segment because speaker identity is only established within one (``S0003/SPEAKER_00``
      and ``S0004/SPEAKER_00`` are unrelated voices). Comparing across that boundary would
      veto every merge spanning a VAD segment on no evidence at all, so differing scopes read
      as "unknown" and permit the merge.
    """
    if seg2.get("speaker_break"):
        return False
    spk1, spk2 = seg1.get("speaker"), seg2.get("speaker")
    if spk1 is None or spk2 is None:
        return True
    if speaker_scope(spk1) != speaker_scope(spk2):
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
    vetoes: "_MergeVetoLog",
    script: ScriptConfig = DEFAULT_SCRIPT,
    max_cue_duration: float = 0.0,
) -> List[SingleAlignedSegment]:
    """Pass A: greedily join touching neighbours with a clean join boundary.

    Two exceptions, both too-short cues held back for the rescue pass instead of being
    absorbed backwards. This pass accumulates strictly left to right, so it would otherwise
    decide before anything got to weigh the other direction:

    * a known *leading* marker ("嗱，" glued onto the end of the preceding sentence);
    * any fragment whose spoken pause to the *next* cue is shorter than to the previous one.
      A short clause that opens the next utterance -- a name called before a sentence,
      "阿明，你過嚟" -- is one comma away from both neighbours, and the cue edges cannot tell
      the two directions apart because each comma's dwell fills its pause. The spoken
      characters can: here the pause before the name is the longer one.

    Held back, the fragment reaches the rescue pass, which weighs both sides and still joins
    it backwards if forwards turns out impossible.
    """
    threshold = align_merge_distance - align_padding
    merged: List[SingleAlignedSegment] = []

    for k, segment in enumerate(segments):
        if not merged:
            merged.append(segment)
            continue

        prev = merged[-1]
        gap = segment["start"] - prev["end"]
        short = _duration(segment) < min_cue_duration
        nxt = segments[k + 1] if k + 1 < len(segments) else None
        defer = short and (
            _text(segment).strip(_TOKEN_TRIM) in leading_markers
            or (nxt is not None
                and _sounded_pause(segment, nxt) + FORWARD_PAUSE_MARGIN
                < _sounded_pause(prev, segment))
        )
        # The speaker gate is checked last so a veto can be attributed to diarization
        # alone: everything else about this join already reads cleanly.
        mergeable = (
            not defer
            and gap <= threshold + GAP_EPS
            and _within_cap(prev, segment, max_cue_duration)
            and is_mergeable(_text(prev), _text(segment), punctuation, max_chars=max_chars,
                             script=script)
        )
        if mergeable and _same_speaker(prev, segment):
            merged[-1] = merge_segments(prev, segment, join=script.join)
        else:
            if mergeable:
                vetoes.record(prev, segment)
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
    vetoes: "_MergeVetoLog",
    script: ScriptConfig = DEFAULT_SCRIPT,
    max_cue_duration: float = 0.0,
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
    3. **Distance** -- otherwise the shorter *spoken* pause decides: the silence between
       the last spoken character on one side and the first on the other, not the gap
       between cue edges, which punctuation's dwell closes to nothing on both sides.

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
            speaker_blocked = []
            for left_idx in (i - 1, i):
                if left_idx < 0 or left_idx + 1 >= len(cues):
                    continue
                left, right = cues[left_idx], cues[left_idx + 1]
                if len(script.join(_text(left), _text(right))) > rescue_max_chars:
                    continue
                if not _within_cap(left, right, max_cue_duration):
                    continue
                gap = right["start"] - left["end"]
                if gap > gap_limit + GAP_EPS:
                    continue
                # Checked after the width and gap gates so a veto means this direction was
                # otherwise usable, and diarization is what closed it.
                if not _same_speaker(left, right):
                    speaker_blocked.append((left, right))
                    continue
                rank = 0 if boundary_is_mergeable(_text(left), punctuation) else 1
                # left_idx == i means this cue is the left member, i.e. joining forwards.
                direction = 0 if (is_marker and left_idx == i) else 1
                candidates.append((rank, direction, _sounded_pause(left, right), left_idx))

            if not candidates:
                # Only worth reporting when the cue is stranded outright. A direction the
                # speaker gate closed costs nothing if the other direction still rescued it.
                for left, right in speaker_blocked:
                    vetoes.record(left, right)
                continue

            candidates.sort()
            left_idx = candidates[0][-1]
            cues[left_idx:left_idx + 2] = [
                merge_segments(cues[left_idx], cues[left_idx + 1], join=script.join)]
            merged_any = True
            break

    return cues


# Pass E's choice of where to cut a long cue. The pause between two timed words dominates
# (seconds of silence, punctuation spanning it included); a punctuation mark at the cut is
# worth this much pause on top, and every 0.1 of the cue's length off-centre costs a tenth of
# BALANCE_WEIGHT seconds -- enough to pick the middlemost of several equal pauses, not to beat
# a real one.
PUNCTUATION_BONUS = 0.25
BALANCE_WEIGHT = 0.5


def _best_cut(seg: SingleAlignedSegment, punctuation: PunctuationConfig,
              min_piece: float) -> Optional[InternalGap]:
    """Where to cut *seg*: between two consecutive timed content words (CJK: characters),
    leaving at least ``min_piece`` seconds each side. None if nowhere qualifies."""
    split = set(punctuation.split_chars)
    words = seg.get("words") or []
    content = [
        (i, w) for i, w in enumerate(words)
        if w.get("start") is not None and w.get("end") is not None
        and str(w.get("word", "")).strip() and str(w.get("word", "")).strip() not in split
    ]
    start, end = float(seg["start"]), float(seg["end"])
    duration = end - start
    best, best_score = None, None
    for (i, a), (j, b) in zip(content, content[1:]):
        head, tail = float(a["end"]) - start, end - float(b["start"])
        if head < min_piece or tail < min_piece:
            continue
        pause = max(0.0, float(b["start"]) - float(a["end"]))
        marked = any(str(w.get("word", "")).strip()[-1:] in split for w in words[i:j])
        centre = (float(a["end"]) + float(b["start"])) / 2
        off_centre = abs(centre - (start + end) / 2) / duration
        score = pause + (PUNCTUATION_BONUS if marked else 0.0) - BALANCE_WEIGHT * off_centre
        if best_score is None or score > best_score:
            best_score = score
            best = InternalGap(i, j, round(pause, 3), round(float(a["end"]), 3),
                               str(a.get("word", "")), str(b.get("word", "")))
    return best


def _split_long_cues(
    segments: Sequence[SingleAlignedSegment],
    max_cue_duration: float,
    punctuation: PunctuationConfig,
) -> List[SingleAlignedSegment]:
    """Pass E: cut every cue longer than ``max_cue_duration`` until its pieces fit.

    Each cut leaves at least a quarter of the cap (a second, at 4 s) on either side, so a cue
    is never shaved into a fragment for pass D to stretch. A cue with nowhere to cut -- no
    word timings, or no boundary far enough from both ends -- is left as it is. The cut
    itself is ``align_checks._split_one``'s, so text, words and characters stay in step.
    """
    min_piece = max_cue_duration / 4
    out: List[SingleAlignedSegment] = []
    pending = list(segments)[::-1]
    cuts = 0
    while pending:
        seg = pending.pop()
        if _duration(seg) <= max_cue_duration + GAP_EPS:
            out.append(seg)
            continue
        cut = _best_cut(seg, punctuation, min_piece)
        pieces = _split_one(seg, [cut], kind="split_long") if cut is not None else None
        if not pieces:
            out.append(seg)
            continue
        cuts += 1
        pending.extend(reversed(pieces))
    if cuts:
        logger.info("Split %d cue(s) longer than %.1fs at their longest internal pause",
                    cuts, max_cue_duration)
    return out


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
    max_chars: Optional[int] = None,
    rescue_max_chars: Optional[int] = None,
    is_noise: Optional[Callable[[str], bool]] = None,
    merge: bool = True,
    script: ScriptConfig = DEFAULT_SCRIPT,
    max_cue_duration: float = 0.0,
) -> List[SingleAlignedSegment]:
    """Turn aligned subsegments into displayable cues (passes A-E; see module docstring).

    ``punctuation``, ``segmentation`` and ``script`` come from the ASR model's profile,
    resolved for the language. ``script`` decides how two cues' text is joined (nothing for
    CJK, a space otherwise). ``max_chars`` caps an ordinary adjacency merge (one subtitle line;
    the script's ``line_width`` by default); ``rescue_max_chars`` caps a short-cue
    rescue and defaults to ``max_chars`` -- callers pass the full multi-line budget
    (``max_line_width * max_line_count``) so a stranded marker can attach to a sentence that
    will simply be broken over two lines. ``is_noise`` decides pass B and is skipped when None.

    ``min_cue_duration=0`` reduces this to pass A alone.

    ``max_cue_duration`` caps how long a cue may run (0, the default here, means no cap): the
    joining passes respect it, and pass E cuts whatever alignment produced longer.

    ``merge=False`` turns off the two passes that *join* cues (A and B's rescue sibling C),
    leaving only the noise drop and the duration floor (pass E is off too: it cuts cues the
    same over-splitting made). Use it when the incoming cue
    boundaries are authoritative rather than an artifact of over-splitting -- under
    --realign the boundaries come from the source transcript's own line breaks, so there is
    nothing to glue back together, and a merge pass would instead destroy them. Note that
    such a transcript is typically punctuated too sparsely for pass A's mergeable-boundary
    test to hold anything back, so leaving it on would fuse most of the file.
    """
    if not segments:
        return []

    # A cue with no text can never be displayed, and leaving one in corrupts its neighbours
    # rather than merely wasting a line: pass A finds no punctuation at the join, reads it as
    # a clean boundary, and glues the cues on either side into one cue spanning the silence
    # between them. Dropped here rather than in pass B because that pass only drops a cue
    # *shorter* than min_cue_duration -- deliberately, since a long cue whose text is noisy
    # may still be speech -- and an empty one is commonly the longest cue in the file.
    segments = [seg for seg in segments if str(seg.get("text", "")).strip()]
    if not segments:
        return []

    if max_chars is None:
        max_chars = script.line_width
    if rescue_max_chars is None:
        rescue_max_chars = max_chars

    # Deferring leading markers in pass A is only safe when the rescue pass runs to place
    # them; with passes B-D off, pass A must behave exactly as it always has.
    leading_markers = frozenset(segmentation.leading_markers) if min_cue_duration > 0 else frozenset()

    vetoes = _MergeVetoLog()
    if merge:
        cues = _adjacency_merge(
            segments, punctuation, align_merge_distance, align_padding, max_chars,
            leading_markers, min_cue_duration, vetoes, script, max_cue_duration,
        )
    else:
        cues = [dict(seg) for seg in segments]

    if min_cue_duration > 0:
        if is_noise is not None:
            cues = _drop_noise(cues, is_noise, min_cue_duration)
        if merge:
            cues = _rescue_short_cues(
                cues, punctuation, segmentation, min_cue_duration, merge_gap, rescue_max_chars,
                vetoes, script, max_cue_duration,
            )
    if merge and max_cue_duration > 0:
        cues = _split_long_cues(cues, max_cue_duration, punctuation)
    if min_cue_duration > 0:
        cues = _apply_duration_floor(cues, min_cue_duration, align_padding)

    vetoes.report()
    return cues

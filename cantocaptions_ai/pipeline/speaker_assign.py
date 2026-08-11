"""Map diarization speaker turns onto aligned segments.

The diarization stage (``pipeline/diarize.py``) produces a flat list of speaker turns for a
whole file. This module turns those turns into per-segment attributions, and is the *only*
place that decides whether an attribution is trustworthy enough to act on.

The decision is deliberately conservative, because the consumer is cue assembly
(``pipeline/segmentation.py``), which refuses to merge two cues carrying different speakers.
A wrong label there permanently splits a sentence, while a missing label costs nothing -- an
unlabeled cue merges exactly as it did before diarization existed. So a segment is labeled
only when one speaker holds ``min_dominant_share`` of the segment's *diarized* time; anything
more ambiguous is left unlabeled and merges freely.

The runner-up's share of the same segment is the multi-speaker signal: a segment where two
speakers each hold a meaningful slice is one the aligner should probably have split. Flagging
those (``flag_conflicts``) is diagnostic only -- it changes no timings and no text.

Pure functions over segment dicts: no models, no I/O, no config objects beyond the frozen
dataclass below, so this is unit-testable without a GPU (see ``tests/test_speaker_assign.py``).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from cantocaptions_ai.utils.schema import (
    SingleAlignedSegment,
    SingleWordSegment,
    SpeakerTurn,
)

# Overlaps shorter than this contribute nothing. Guards against a segment boundary grazing a
# turn boundary by a rounding error and picking up a spurious second speaker.
_MIN_OVERLAP = 1e-3

# Separates a speaker label's scope from the label itself ("S0007/SPEAKER_00"). Chosen because
# pyannote's own labels contain "_" but never "/", so the split is unambiguous.
SCOPE_SEP = "/"


def segment_scope_id(index: int) -> str:
    """Stable scope id for the ``index``-th VAD segment of a file."""
    return f"S{index:04d}"


def scoped_speaker(scope: str, label: str) -> str:
    """Namespace a speaker label to the scope it was derived in."""
    return f"{scope}{SCOPE_SEP}{label}"


def speaker_scope(label: Optional[str]) -> Optional[str]:
    """The scope a speaker label belongs to, or None when it is globally scoped.

    Per-segment diarization assigns speaker identities independently per VAD segment, so
    ``S0003/SPEAKER_00`` and ``S0004/SPEAKER_00`` are unrelated. Callers comparing two labels
    must check this first; whole-file labels have no scope and compare directly.
    """
    if label is None:
        return None
    scope, separator, _ = label.partition(SCOPE_SEP)
    return scope if separator else None


@dataclass(frozen=True)
class SpeakerAssignmentConfig:
    """Thresholds governing how diarization turns become segment labels.

    min_dominant_share:
        Fraction of a segment's diarized time the leading speaker must hold before the
        segment is labeled at all. Below it the segment stays unlabeled (merge-permissive).
    conflict_share:
        Fraction the *runner-up* must hold for the segment to be flagged as containing more
        than one speaker. Only consulted when ``flag_conflicts`` is set.
    flag_conflicts:
        Experimental diagnostic; writes ``speaker_conflict`` onto qualifying segments.
    """
    min_dominant_share: float = 0.7
    conflict_share: float = 0.25
    flag_conflicts: bool = False


@dataclass(frozen=True)
class AssignmentStats:
    """Summary of one file's assignment pass, for logging."""
    speakers: Tuple[str, ...]
    total: int
    labeled: int
    conflicts: int

    @property
    def unlabeled(self) -> int:
        return self.total - self.labeled


def speaker_shares(
    turns: Sequence[SpeakerTurn],
    start: float,
    end: float,
    *,
    first: int = 0,
) -> Tuple[Dict[str, float], int]:
    """Seconds of ``[start, end)`` attributed to each speaker, plus a resume index.

    ``turns`` must be sorted by start time. ``first`` is an index to start scanning from;
    the returned index is the first turn that could still overlap a *later* window, so a
    caller walking a sorted list of windows can thread it through and pay O(n + m) overall
    instead of rescanning the turn list per window.

    Speakers contributing less than a millisecond are omitted.
    """
    shares: Dict[str, float] = {}
    # Skip turns that end at or before this window; since windows are consumed in start
    # order, they cannot overlap any later window either.
    while first < len(turns) and turns[first]["end"] <= start:
        first += 1

    for turn in turns[first:]:
        if turn["start"] >= end:
            break
        overlap = min(turn["end"], end) - max(turn["start"], start)
        if overlap > _MIN_OVERLAP:
            shares[turn["speaker"]] = shares.get(turn["speaker"], 0.0) + overlap

    return shares, first


def rank_shares(shares: Dict[str, float]) -> List[Tuple[str, float]]:
    """Speakers as ``(speaker, fraction_of_covered_time)``, most dominant first.

    Normalising by *covered* time rather than by segment duration is what makes the threshold
    mean "this segment is one speaker" rather than "this segment is mostly speech": a cue
    padded with silence should not be penalised for it. Ties break on speaker name so the
    result is deterministic.
    """
    covered = sum(shares.values())
    if covered <= 0:
        return []
    ranked = [(speaker, seconds / covered) for speaker, seconds in shares.items()]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def _apply(
    target: dict,
    ranked: List[Tuple[str, float]],
    config: SpeakerAssignmentConfig,
    *,
    with_conflict: bool,
) -> bool:
    """Write speaker keys onto ``target`` from its ranked shares. True if it was labeled.

    Stale keys are cleared first, so re-running assignment (which a ``--load_debug_dir``
    replay with different thresholds does) never leaves old and new attributions mixed.
    """
    target.pop("speaker", None)
    target.pop("speaker_confidence", None)
    target.pop("speaker_conflict", None)

    if not ranked:
        return False

    speaker, share = ranked[0]
    target["speaker_confidence"] = round(share, 3)
    labeled = share >= config.min_dominant_share
    if labeled:
        target["speaker"] = speaker

    if (
        with_conflict
        and config.flag_conflicts
        and len(ranked) > 1
        and ranked[1][1] >= config.conflict_share
    ):
        target["speaker_conflict"] = True

    return labeled


def _assign_words(
    words: Sequence[SingleWordSegment],
    turns: Sequence[SpeakerTurn],
    config: SpeakerAssignmentConfig,
    *,
    first: int = 0,
) -> None:
    """Label individual words in place, using the same threshold as segments.

    Words are short enough that one turn usually covers a whole word, so this is mostly
    exact. It exists to give a future "split a cue where the speaker changes mid-line"
    experiment something to cut on.

    ``first`` is the enclosing segment's resume index. Words lie inside that segment, so no
    earlier turn can reach them, and starting there keeps the whole pass linear.
    """
    cursor = first
    for word in words:
        start, end = word.get("start"), word.get("end")
        # Alignment leaves start/end off words it could not place (e.g. bare punctuation).
        if start is None or end is None:
            word.pop("speaker", None)
            continue
        shares, cursor = speaker_shares(turns, start, end, first=cursor)
        _apply(word, rank_shares(shares), config, with_conflict=False)


def assign_speakers(
    segments: List[SingleAlignedSegment],
    turns: Sequence[SpeakerTurn],
    config: SpeakerAssignmentConfig = SpeakerAssignmentConfig(),
) -> AssignmentStats:
    """Attribute each segment (and its words) to a speaker, in place.

    Both inputs are sorted by start time here, which lets the segment pass run as a single
    sweep. Segments with no diarized speech beneath them are left unlabeled rather than
    guessed at -- see the module docstring for why.
    """
    ordered_turns = sorted(turns, key=lambda turn: (turn["start"], turn["end"]))
    ordered_segments = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    cursor = 0
    labeled = 0
    conflicts = 0

    for segment in ordered_segments:
        shares, cursor = speaker_shares(
            ordered_turns, segment["start"], segment["end"], first=cursor
        )
        if _apply(segment, rank_shares(shares), config, with_conflict=True):
            labeled += 1
        if segment.get("speaker_conflict"):
            conflicts += 1
        words = segment.get("words")
        if words:
            _assign_words(words, ordered_turns, config, first=cursor)

    speakers = tuple(sorted({turn["speaker"] for turn in ordered_turns}))
    return AssignmentStats(
        speakers=speakers, total=len(segments), labeled=labeled, conflicts=conflicts
    )


def format_stats(stats: AssignmentStats) -> str:
    """One-line human summary for the stage log."""
    parts = [
        f"{len(stats.speakers)} speaker(s)",
        f"{stats.labeled}/{stats.total} cues confidently attributed",
    ]
    if stats.conflicts:
        parts.append(f"{stats.conflicts} flagged as multi-speaker")
    return ", ".join(parts)

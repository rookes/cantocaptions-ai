"""Model-agnostic sanity checks on alignment output.

Unlike ``pipeline/align_profiles.py``, nothing here is per-model and nothing here is
opt-in: these run for **every** alignment model, including one added later with no profile
entry. That is possible because the checks read only the aligned timings and the audio —
never the emission tensor, the blank id, or the model's vocabulary — so they stay correct
for any model that produces ``{start, end, text}``.

There are two of them. ``find_gapped_cues`` reads the timings alone and needs no audio;
``find_silent_starts`` needs the waveform. Neither repairs anything -- both say which cues
to distrust, which is the thing a 2000-cue file cannot be used without.

``find_silent_starts`` catches a cue whose start time lands on audio where nothing is
being said. Alignment placing a character over silence is always wrong, whatever produced
it; the failure that motivated the check was ``alvanlii/wav2vec2-BERT-cantonese`` pinning
each VAD segment's first character to frame 0 (see ``align_profiles.TailPrimer``), but the
check is written against the symptom rather than that cause.

**It is precise, not exhaustive.** Measured over 96 segments in three files: when it fires
it is a real error ~94% of the time, but it catches only about half of them. It can only
see a lead-in that is genuinely near-silent, and the failure it was built for also happens
over music beds and room tone, which are speechless but not silent. Treat a clean run as
"nothing obvious", not as proof the alignment is sound.
"""

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

# One frame per 40 ms, matching the ~25 fps the CTC emission runs at for the current align
# model. Nothing depends on the two agreeing — this is only the resolution the audio's
# energy envelope is measured at — but a comparable scale keeps "gap" counts meaningful.
FRAME_SECONDS = 0.04

# A frame counts as silent within this much of the region's own noise floor, so the check
# adapts to whatever the floor happens to be rather than assuming digital zero.
FLOOR_MARGIN_DB = 6.0

# Below this much spread between floor and speech level there is no usable silence floor
# (a music bed sits a few dB under quiet speech). Skip the region rather than guess: both
# false positives seen during development were music-bed segments, where "floor + margin"
# swallowed real speech.
MIN_SEPARATION_DB = 30.0

# How far a start must sit from *any* sound before it is suspicious. Two effects stack to
# make small gaps normal: CTC legitimately leads the acoustic onset by a frame or two, and
# an unvoiced onset (就, 三, 開 — affricates and fricatives) carries little energy, so the
# envelope reads as silence where speech has in fact begun. Swept over three files with the
# fix in place: 4 frames is the smallest value that never fires on correct output, while
# still catching errors at the scale of one vad_pad_onset (0.2s = 5 frames). At 3 it fires
# on correct cues; at 6 it only catches gross failures.
MIN_GAP_FRAMES = 4

# The least silence between two adjacent characters of one cue that says the cue is not a
# single continuous utterance. Well clear of an ordinary breath or an unvoiced onset, which
# is where find_silent_starts' 4-frame threshold sits; a whole second between two characters
# that are written next to each other means one of them was placed somewhere it was not said.
MAX_INTERNAL_GAP = 1.0


@dataclass(frozen=True)
class GappedCue:
    """A cue holding a silence between two of its own adjacent characters."""

    index: int
    gap: float
    at: float
    before: str
    after: str


def find_gapped_cues(
    segments: Sequence[Mapping],
    max_gap: float = MAX_INTERNAL_GAP,
    split_chars: Iterable[str] = (),
) -> List[GappedCue]:
    """Cues whose characters are not one continuous utterance.

    CTC must place every token it is given, so when a cue's text contains something that was
    not said where the cue sits, the extra characters are put wherever scores least badly --
    typically seconds away, leaving a hole in the middle of the cue. On `test/bluey`, ASR
    emitted 爸爸， once for what the reference has as two separate calls: the first 爸 landed
    at 56.44 s and the second at 59.80 s, so the subtitle went on screen **3.35 s** before
    anybody spoke (the reference cue starts at 59.790).

    Only *directly adjacent* characters are compared, which is what makes this safe to leave
    on. A punctuation mark is mapped to blank precisely so it can absorb a pause, and it does
    so by *spanning* it -- the mark is contiguous with both neighbours, so an ordinary clause
    break produces no gap at all. ``split_chars`` covers only the residual case where the mark
    itself sits inside the silence rather than filling it. Measured on the two ASR fixtures,
    every gap found is between two real characters with nothing between them.

    **This is a report, not a repair, and that is deliberate.** 45% of the cues it flags are
    still within half a second of the reference, so trimming or splitting on it would hurt
    about as often as it helped; and which side of the gap to keep has no general answer
    (realign's ``_trim_detached_edges`` resolves that tie towards the opening, which is right
    for a whole transcript line and wrong for a two-character cue). What it is good at is
    saying *which* cues to distrust, and it is much better at that than the CTC score:

    | cues | n | median start error | within 0.5 s |
    |---|---|---|---|
    | with a gap over 1 s | 12 | 1.25-3.35 s | 0-45% |
    | without | 545 | 0.05-0.09 s | 76-79% |
    """
    split = set(split_chars)
    hits: List[GappedCue] = []
    for index, segment in enumerate(segments):
        words = [w for w in (segment.get("words") or [])
                 if w.get("start") is not None and w.get("end") is not None]
        worst: Optional[GappedCue] = None
        for before, after in zip(words, words[1:]):
            if before["word"] in split or after["word"] in split:
                continue
            gap = float(after["start"]) - float(before["end"])
            if gap > max_gap and (worst is None or gap > worst.gap):
                worst = GappedCue(index, round(gap, 3), round(float(before["end"]), 3),
                                  before["word"], after["word"])
        if worst is not None:
            hits.append(worst)
    return hits


def warn_on_gapped_cues(
    segments: Sequence[Mapping],
    max_gap: float = MAX_INTERNAL_GAP,
    split_chars: Iterable[str] = (),
) -> List[GappedCue]:
    """find_gapped_cues, with a line per hit at debug and a count at info."""
    hits = find_gapped_cues(segments, max_gap, split_chars)
    for hit in hits:
        logger.debug(
            "Cue %d holds %.2fs of silence between %r and %r at %.3fs: %r",
            hit.index, hit.gap, hit.before, hit.after, hit.at,
            str(segments[hit.index].get("text", ""))[:40],
        )
    if hits:
        logger.info(
            "%d cue(s) contain a silence over %.1fs between two of their own characters; "
            "their timings are the least trustworthy in the file (see notes.srt)",
            len(hits), max_gap,
        )
    return hits


@dataclass(frozen=True)
class SilentStart:
    """One cue whose start time landed on silence."""

    time: float
    """Absolute start time of the cue, in seconds."""
    text: str
    gap: float
    """Seconds from that start to the nearest audio energy."""
    dbfs: float
    floor_dbfs: float
    index: int = -1
    """Position of the cue in the list that was checked, so a caller can annotate it."""


def frame_dbfs(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_seconds: float = FRAME_SECONDS
) -> np.ndarray:
    """RMS level per fixed-length frame, in dBFS.

    Frames are exactly ``frame_seconds`` long and start at the audio's first sample, so a
    time offset maps to a frame index by plain division — no windowing or hop bookkeeping.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    hop = max(int(round(frame_seconds * sample_rate)), 1)
    n = len(audio) // hop
    if n == 0:
        return np.empty(0)
    frames = audio[: n * hop].reshape(n, hop)
    return 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)


def _gap_to_energy(silent: np.ndarray) -> np.ndarray:
    """Frames from each position to the nearest non-silent frame, in either direction."""
    n = len(silent)
    far = n + 1
    gaps = np.full(n, far, dtype=np.int64)

    last = -far
    for t in range(n):
        if not silent[t]:
            last = t
        gaps[t] = t - last

    nxt = far
    for t in range(n - 1, -1, -1):
        if not silent[t]:
            nxt = t
        gaps[t] = min(gaps[t], nxt - t)
    return gaps


def find_silent_starts(
    aligned_segments: Sequence[Mapping],
    regions: Iterable[Mapping],
    sample_rate: int = SAMPLE_RATE,
    frame_seconds: float = FRAME_SECONDS,
    floor_margin_db: float = FLOOR_MARGIN_DB,
    min_separation_db: float = MIN_SEPARATION_DB,
    min_gap_frames: int = MIN_GAP_FRAMES,
) -> List[SilentStart]:
    """Cues whose start sits on near-silent audio, far from any sound.

    ``regions`` are dicts carrying ``start``/``end``/``audio`` — VAD segments as the
    pipeline builds them, or a single whole-file region. Each region establishes its own
    noise floor, since level varies widely between them.
    """
    hits: List[SilentStart] = []
    starts = [(i, seg.get("start")) for i, seg in enumerate(aligned_segments)]

    for region in regions:
        audio = region.get("audio")
        if audio is None:
            continue
        db = frame_dbfs(audio, sample_rate, frame_seconds)
        if db.size == 0:
            continue

        floor, speech = np.percentile(db, [5, 95])
        if speech - floor < min_separation_db:
            continue

        silent = db < floor + floor_margin_db
        gaps = _gap_to_energy(silent)
        r_start, r_end = region["start"], region["end"]

        for i, t in starts:
            if t is None or not (r_start <= t < r_end):
                continue
            idx = int((t - r_start) / frame_seconds)
            if not (0 <= idx < db.size) or not silent[idx] or gaps[idx] < min_gap_frames:
                continue
            hits.append(SilentStart(
                time=float(t),
                text=str(aligned_segments[i].get("text", "")),
                index=i,
                gap=float(gaps[idx] * frame_seconds),
                dbfs=float(db[idx]),
                floor_dbfs=float(floor),
            ))

    hits.sort(key=lambda h: h.time)
    return hits


def warn_on_silent_starts(
    aligned_segments: Sequence[Mapping],
    regions: Iterable[Mapping],
    max_examples: int = 3,
    **kwargs,
) -> List[SilentStart]:
    """Run ``find_silent_starts`` and log what it found. Returns the hits."""
    hits = find_silent_starts(aligned_segments, regions, **kwargs)
    if not hits:
        return hits

    from cantocaptions_ai.utils.output import format_timestamp

    logger.warning(
        "Alignment placed %d cue %s on silence — the audio there is at the segment's noise "
        "floor with no sound within %.2fs. Timings for %s are not trustworthy.",
        len(hits), "start" if len(hits) == 1 else "starts",
        max(h.gap for h in hits), "it" if len(hits) == 1 else "them",
    )
    for hit in hits[:max_examples]:
        logger.info(
            "  silent cue start at %s: %.2fs of silence before %r (%.0f dBFS, floor %.0f)",
            format_timestamp(hit.time, always_include_hours=True, decimal_marker=","),
            hit.gap, hit.text[:30], hit.dbfs, hit.floor_dbfs,
        )
    if len(hits) > max_examples:
        logger.info("  ...and %d more", len(hits) - max_examples)
    return hits


def region_levels(
    regions: Iterable[Mapping],
    sample_rate: int = SAMPLE_RATE,
    frame_seconds: float = FRAME_SECONDS,
) -> List[Tuple[float, np.ndarray]]:
    """Per-frame dBFS for each region, as (region start, levels), sorted by start.

    Levels only -- no thresholds and no verdict. Callers decide what quiet means, because the
    right comparison depends on the question.

    Be careful what you ask of it. Trimming a cue back off "silence" by level was tried and
    does not work on real material: inside the Police Story 2 cues that sit through a pause,
    the pause is continuous room tone about 12 dB under the dialogue and the first frame above
    any sane threshold is the cue's own first frame. What perceptually reads as silence is not
    silence to an envelope. The dwell those cues came from is visible in the *emission*
    instead -- see _align_segment's MAX_CHAR_DWELL_SECONDS.
    """
    out = []
    for region in regions:
        audio = region.get("audio")
        if audio is None:
            continue
        db = frame_dbfs(audio, sample_rate, frame_seconds)
        if db.size:
            out.append((float(region["start"]), db))
    out.sort(key=lambda item: item[0])
    return out


def whole_file_region(audio, duration: float) -> List[dict]:
    """Wrap a whole-file waveform as a single region for ``find_silent_starts``.

    Used when ``align()`` was handed a waveform instead of VAD segments; accepts the numpy
    array or torch tensor, mono or (1, N), that ``align()`` may be holding at that point.
    """
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.reshape(audio.shape[0], -1)[0]
    return [{"start": 0.0, "end": duration, "audio": audio}]

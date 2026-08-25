"""Model-agnostic sanity checks on alignment output.

Unlike ``pipeline/align_profiles.py``, nothing here is per-model and nothing here is
opt-in: these run for **every** alignment model, including one added later with no profile
entry. That is possible because the checks read only the aligned timings and the audio —
never the emission tensor, the blank id, or the model's vocabulary — so they stay correct
for any model that produces ``{start, end, text}``.

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
from typing import Iterable, List, Mapping, Sequence

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

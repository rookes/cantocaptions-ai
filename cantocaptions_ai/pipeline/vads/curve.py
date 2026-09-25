"""Score-curve VADs: every backend that emits a per-frame speech probability.

A backend's only job is to turn audio into a ``SlidingWindowFeature`` of speech
probabilities (one column). Everything after that -- hysteresis thresholds, padding,
gap bridging, the min-cut that caps a region at ``chunk_size`` -- lives here and is
shared, so ``vad_onset``/``vad_offset``/``vad_pad_*`` mean the same operation whichever
model produced the curve. Only the curve's *calibration* differs between models, which is
why each backend needs its own tuned thresholds (silero's are in config/cpu.cfg).
"""
import bisect
import math
from typing import Optional

import numpy as np

from cantocaptions_ai.pipeline.vads.base import Vad
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

class _Segment:
    def __init__(self, start: float, end: float, speaker: Optional[str] = None):
        self.start = start
        self.end = end
        self.speaker = speaker


def frame_middles(scores) -> list:
    """Centre time of every frame of a ``SlidingWindowFeature``, as a list of floats.

    The same arithmetic ``SlidingWindow.__getitem__(i).middle`` does, element for element, so
    the values are bit-identical to it -- without constructing a Segment per frame, which was
    the dominant cost of binarizing an hour of audio.
    """
    window = scores.sliding_window
    starts = window.start + np.arange(scores.data.shape[0]) * window.step
    return (0.5 * (starts + (starts + window.duration))).tolist()



class Binarize:
    """Binarize detection scores using hysteresis thresholding, then smooth, then cap length.

    The three stages run in this order, which matters:

    1. **Hysteresis** — a region opens when the score exceeds ``onset`` and closes when it
       drops below ``offset``.
    2. **Smoothing** — regions are widened by ``pad_onset``/``pad_offset``, then any pair
       separated by no more than ``min_duration_off`` is merged, then regions shorter than
       ``min_duration_on`` are dropped. Without this the model's frame-exact threshold
       crossings clip word onsets and tails, and a single sub-threshold frame (~17 ms) tears a
       region in half.
    3. **Min-cut** — any region still longer than ``max_duration`` is split at the
       lowest-scoring frame in the over-long window, repeatedly. By default the search is
       limited to the window's second half (WhisperX's min-cut); ``min_split_duration`` widens
       it to also cover the first half, down to that floor — see ``_split_long``. Splits are
       contiguous (the split timestamp ends one piece and starts the next), so no audio is
       dropped.

    Stage 3 runs *last* on purpose. It used to run inside the hysteresis loop, which made it
    mutually exclusive with stage 2 — setting any padding raised NotImplementedError whenever
    ``max_duration`` was finite, and ``max_duration`` is always finite here because it is the
    ASR chunk budget. Capping the *final* regions instead lets both apply, and guarantees the
    cap holds after padding has widened things.

    Parameters
    ----------
    onset : float, optional
        Onset threshold. Defaults to 0.5.
    offset : float, optional
        Offset threshold. Defaults to `onset`.
    min_duration_on : float, optional
        Remove active regions shorter than that many seconds. Defaults to 0s.
    min_duration_off : float, optional
        Fill inactive regions shorter than that many seconds. Defaults to 0s.
    pad_onset : float, optional
        Extend active regions by moving their start time by that many seconds.
        Defaults to 0s.
    pad_offset : float, optional
        Extend active regions by moving their end time by that many seconds.
        Defaults to 0s.
    max_duration: float
        The maximum length of an active segment, divides segment at timestamp with lowest score.
    min_split_duration : float, optional
        Search floor for the min-cut, in seconds. ``None`` (default) searches only the second
        half of the over-long window (``max_duration / 2``). A smaller value widens the search
        into the first half too, so a genuine low-score dip before the midpoint can be found
        instead of forcing a cut deep inside continuous speech. See ``_split_long``.
    Reference
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.

    Modified by Max Bain to include WhisperX's min-cut operation
    https://arxiv.org/abs/2303.00747

    Pyannote-audio
    """

    def __init__(
            self,
            onset: float = 0.5,
            offset: Optional[float] = None,
            min_duration_on: float = 0.0,
            min_duration_off: float = 0.0,
            pad_onset: float = 0.0,
            pad_offset: float = 0.0,
            max_duration: float = float('inf'),
            min_split_duration: Optional[float] = None,
    ):

        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration
        # None preserves the original WhisperX-style "second half only" search
        # (min_split_duration == max_duration / 2); see _split_long.
        self.min_split_duration = min_split_duration

    def _hysteresis(self, timestamps, k_scores):
        """Stage 1: raw (start, end) regions from hysteresis thresholding, no smoothing.

        A region opens on the first frame scoring above ``onset`` and closes on the first
        *later* frame scoring below ``offset``; the closing frame is never itself re-tested as
        an opening. Walks the threshold crossings rather than every frame, which is what makes
        a parameter sweep over hours of audio affordable -- the result is identical to the
        frame-by-frame loop it replaced (tests/test_vad_binarize.py pins that).
        """
        y = np.asarray(k_scores)
        n = len(y)
        if n == 0:
            return []
        ups = np.flatnonzero(y[1:] > self.onset) + 1
        downs = np.flatnonzero(y[1:] < self.offset) + 1

        regions = []
        if y[0] > self.onset:
            s = 0
        elif len(ups):
            s = int(ups[0])
        else:
            return regions
        while True:
            k = int(np.searchsorted(downs, s, side="right"))
            if k == len(downs):
                regions.append((float(timestamps[s]), float(timestamps[n - 1])))
                return regions
            e = int(downs[k])
            regions.append((float(timestamps[s]), float(timestamps[e])))
            k = int(np.searchsorted(ups, e, side="right"))
            if k == len(ups):
                return regions
            s = int(ups[k])

    def _smooth(self, regions, lower_bound, upper_bound):
        """Stage 2: pad outward, merge across short gaps, drop stragglers."""
        if not regions:
            return regions

        # Clamp so a padded region can never start before the audio (which would become a
        # negative sample index when the caller slices the waveform) or run past its end.
        padded = [
            (max(start - self.pad_onset, lower_bound), min(end + self.pad_offset, upper_bound))
            for start, end in regions
        ]

        merged = [padded[0]]
        for start, end in padded[1:]:
            prev_start, prev_end = merged[-1]
            # Negative gaps (overlaps created by padding) are <= min_duration_off too.
            if start - prev_end <= self.min_duration_off:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        if self.min_duration_on > 0:
            merged = [r for r in merged if r[1] - r[0] >= self.min_duration_on]
        return merged

    def _split_long(self, regions, timestamps, k_scores):
        """Stage 3: cap region length, cutting at the lowest-scoring frame.

        The search floor is ``max_duration / 2`` by default (WhisperX's min-cut: search only
        the second half of the over-long window, so each emitted piece is at least half the
        budget and the loop always makes progress). ``min_split_duration`` -- when the caller
        supplies one -- lowers that floor instead, so the search also covers the *first* half.

        This matters because ``max_duration/2`` is an argument about loop termination, not
        about the audio: it guarantees *some* cut exists, not a *good* one. In a long run of
        genuinely continuous speech (a monologue with no pause anywhere near the midpoint),
        the lowest-scoring frame in the second half is still deep inside active speech --
        there is no real pause there to find, so the cut lands wherever, indistinguishable
        from random. If a real, deeper dip exists earlier in the window (before the midpoint),
        forbidding the search from ever looking there is what forces the bad cut.

        ``min_split_duration`` still has to stop the search from reaching all the way back to
        ``start`` -- a floor of 0 would let a single noisy low-scoring frame right next to the
        previous cut immediately become the next one, fragmenting a region into many
        near-zero-length pieces before the loop's forward-progress guarantee (still technically
        respected, since ``split_t > start`` whenever ``min_split_duration > 0``) becomes
        practically meaningless. The caller is expected to derive it from the same
        onset/offset/pad/min_duration_off configuration already governing hysteresis and
        smoothing -- see ``CurveVad.merge_chunks`` -- rather than pass a new unrelated
        constant. ``None`` (the default, and what ``cover_chunks`` uses -- see its own
        docstring for why that guarantee must not move) preserves the original
        ``max_duration / 2`` behaviour exactly.

        The split timestamp both ends one piece and starts the next, so no audio is dropped.
        """
        if self.max_duration == float("inf"):
            return regions

        floor = self.min_split_duration if self.min_split_duration is not None else self.max_duration / 2

        out = []
        for start, end in regions:
            while end - start > self.max_duration:
                lo = bisect.bisect_left(timestamps, start + floor)
                hi = bisect.bisect_right(timestamps, start + self.max_duration)
                if hi <= lo:
                    break
                split_t = timestamps[lo + int(np.argmin(k_scores[lo:hi]))]
                if not (start < split_t < end):
                    break
                out.append((start, split_t))
                start = split_t
            out.append((start, end))
        return out

    def __call__(self, scores: "SlidingWindowFeature") -> "Annotation":
        """Binarize detection scores
        Parameters
        ----------
        scores : SlidingWindowFeature
            Detection scores.
        Returns
        -------
        active : Annotation
            Binarized scores.
        """
        from pyannote.core import Annotation, Segment

        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = frame_middles(scores)
        lower_bound = min(frames[0].start, timestamps[0])
        upper_bound = max(frames[num_frames - 1].end, timestamps[-1])

        # annotation meant to store 'active' regions
        active = Annotation()
        for k, k_scores in enumerate(scores.data.T):
            label = k if scores.labels is None else scores.labels[k]

            regions = self._hysteresis(timestamps, k_scores)
            regions = self._smooth(regions, lower_bound, upper_bound)
            regions = self._split_long(regions, timestamps, k_scores)

            for start, end in regions:
                active[Segment(start, end), k] = label

        return active



class CurveVad(Vad):
    """A VAD whose ``__call__`` returns a speech-probability ``SlidingWindowFeature``."""

    @staticmethod
    def cover_chunks(segments, chunk_size, duration):
        """Contiguous chunking: cut ``[0, duration]`` at quiet frames, keep every sample.

        Reuses stage 3 of Binarize -- and *only* stage 3. Hysteresis and smoothing are what
        decide which audio is speech, so skipping them is the whole point: the single region
        [0, duration] is handed straight to the min-cut, which recursively splits at the
        lowest-scoring frame in the second half of each over-long window. Because a split
        timestamp both ends one piece and starts the next, the result tiles the file exactly.

        Each piece is between ``chunk_size / 2`` and ``chunk_size`` seconds (the last one may
        be shorter), and cuts land in the quietest frame available, so a chunk boundary
        rarely falls mid-word.
        """
        assert chunk_size > 0
        timestamps = frame_middles(segments)
        # Column 0 is the speech probability (for pyannote, max-aggregated over its classes by
        # the pre-aggregation hook in load_vad_model); the same curve Binarize thresholds on.
        k_scores = segments.data[:, 0]

        binarize = Binarize(max_duration=chunk_size)
        regions = binarize._split_long([(0.0, duration)], timestamps, k_scores)
        # _split_long bails out of its loop when no candidate frame exists in the search
        # window (a duration shorter than the frame grid, say), which can leave a piece over
        # budget. Fall back to an even split there rather than handing alignment a chunk it
        # cannot hold in memory.
        out = []
        for start, end in regions:
            if end - start <= chunk_size:
                out.append((start, end))
                continue
            n = int(math.ceil((end - start) / chunk_size))
            step = (end - start) / n
            out.extend((start + i * step, start + (i + 1) * step) for i in range(n))
            out[-1] = (out[-1][0], end)
        return [{"start": s, "end": e, "segments": [(s, e)]} for s, e in out]

    @staticmethod
    def speech_regions(segments,
                       onset: float = 0.5,
                       offset: Optional[float] = None,
                       pad_onset: float = 0.0,
                       pad_offset: float = 0.0,
                       min_duration_off: float = 0.0,
                       min_duration_on: float = 0.0,
                       ):
        """Binarized speech turns as ``[(start, end)]``, ungrouped and uncapped.

        Stages 1 and 2 of Binarize only: no ``max_duration``, because nothing is being
        budgeted here -- the caller wants to know where the speech *is*, not how to cut it up.
        """
        binarize = Binarize(
            onset=onset, offset=offset, pad_onset=pad_onset, pad_offset=pad_offset,
            min_duration_off=min_duration_off, min_duration_on=min_duration_on,
        )
        return [(turn.start, turn.end) for turn in binarize(segments).get_timeline()]

    @staticmethod
    def merge_chunks(segments,
                     chunk_size,
                     onset: float = 0.5,
                     offset: Optional[float] = None,
                     pad_onset: float = 0.0,
                     pad_offset: float = 0.0,
                     min_duration_off: float = 0.0,
                     min_duration_on: float = 0.0,
                     ):
        assert chunk_size > 0
        # The min-cut's search floor: the shortest span that could hold a boundary the rest
        # of this configuration would treat as real -- a real gap has to clear
        # min_duration_off to not be bridged by _smooth, and pad_onset/pad_offset is the
        # context the pipeline always attaches to one. Anything shorter is indistinguishable
        # from noise, so the search still refuses to land there; anything at or beyond it is
        # fair game, including before max_duration/2. See _split_long for the full rationale.
        min_split_duration = pad_onset + pad_offset + min_duration_off
        binarize = Binarize(
            max_duration=chunk_size, onset=onset, offset=offset,
            pad_onset=pad_onset, pad_offset=pad_offset,
            min_duration_off=min_duration_off, min_duration_on=min_duration_on,
            min_split_duration=min_split_duration,
        )
        segments = binarize(segments)
        segments_list = []
        for speech_turn in segments.get_timeline():
            segments_list.append(_Segment(speech_turn.start, speech_turn.end, "UNKNOWN"))

        if len(segments_list) == 0:
            logger.warning("No active speech found in audio")
            return []
        return Vad.merge_chunks(segments_list, chunk_size)

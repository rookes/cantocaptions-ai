import bisect
import os
from typing import Optional, Union

import numpy as np
import torch

from cantocaptions_ai.pipeline.vads.base import Vad
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


class _Segment:
    def __init__(self, start: float, end: float, speaker: Optional[str] = None):
        self.start = start
        self.end = end
        self.speaker = speaker


def load_vad_model(device, token=None, model_fp=None):
    """Load the pyannote segmentation model as a frame-scoring Inference.

    Note this only produces the score curve; the onset/offset thresholds and all smoothing
    are applied later, in Binarize (via merge_chunks) — not here.
    """
    # Import only from pyannote.audio.core — avoids pyannote.audio.pipelines.__init__,
    # which eagerly loads SpeakerDiarization → speaker_verification → NeMo.
    from pyannote.audio.core.model import Model
    from pyannote.audio.core.inference import Inference

    model_dir = torch.hub._get_torch_home()

    # __file__ is cantocaptions_ai/pipeline/vads/pyannote.py
    # assets/ is at cantocaptions_ai/assets/ (3 levels up)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    os.makedirs(model_dir, exist_ok=True)
    if model_fp is None:
        model_fp = os.path.join(main_dir, "assets", "pytorch_model.bin")
        model_fp = os.path.abspath(model_fp)
    else:
        model_fp = os.path.abspath(model_fp)

    if not os.path.exists(model_fp):
        raise FileNotFoundError(f"Model file not found at {model_fp}")

    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f"{model_fp} exists and is not a regular file")

    vad_model = Model.from_pretrained(model_fp, token=token)
    return Inference(
        vad_model,
        device=torch.device(device),
        pre_aggregation_hook=lambda scores: np.max(scores, axis=-1, keepdims=True),
    )

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
       lowest-scoring frame in the second half of the over-long window, repeatedly. Splits are
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
            max_duration: float = float('inf')
    ):

        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration

    def _hysteresis(self, timestamps, k_scores):
        """Stage 1: raw (start, end) regions from hysteresis thresholding, no smoothing."""
        regions = []
        start = timestamps[0]
        is_active = k_scores[0] > self.onset
        t = start
        for t, y in zip(timestamps[1:], k_scores[1:]):
            if is_active:
                if y < self.offset:
                    regions.append((start, t))
                    is_active = False
            elif y > self.onset:
                start = t
                is_active = True
        if is_active:
            regions.append((start, t))
        return regions

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

        Searches the second half of the over-long window (as WhisperX's min-cut does) so each
        emitted piece is at least ``max_duration / 2`` and the loop always makes progress.
        The split timestamp both ends one piece and starts the next, so no audio is dropped.
        """
        if self.max_duration == float("inf"):
            return regions

        out = []
        for start, end in regions:
            while end - start > self.max_duration:
                lo = bisect.bisect_left(timestamps, start + self.max_duration / 2)
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
        timestamps = [frames[i].middle for i in range(num_frames)]
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


class Pyannote(Vad):

    def __init__(self, device, token=None, model_fp=None, **kwargs):
        logger.info("Performing voice activity detection using Pyannote...")
        super().__init__(kwargs['vad_onset'])
        self.vad_pipeline = load_vad_model(device, token=token, model_fp=model_fp)

    def __call__(self, audio, **kwargs):
        return self.vad_pipeline(audio)

    @staticmethod
    def preprocess_audio(audio):
        return torch.from_numpy(audio).unsqueeze(0)

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
        binarize = Binarize(
            max_duration=chunk_size, onset=onset, offset=offset,
            pad_onset=pad_onset, pad_offset=pad_offset,
            min_duration_off=min_duration_off, min_duration_on=min_duration_on,
        )
        segments = binarize(segments)
        segments_list = []
        for speech_turn in segments.get_timeline():
            segments_list.append(_Segment(speech_turn.start, speech_turn.end, "UNKNOWN"))

        if len(segments_list) == 0:
            logger.warning("No active speech found in audio")
            return []
        return Vad.merge_chunks(segments_list, chunk_size)

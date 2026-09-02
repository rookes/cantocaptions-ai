from typing import List, Optional, Union

import numpy as np
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.schema import ProgressCallback, SingleSegment, VadAudioSegment
from cantocaptions_ai.utils.model_utils import PipelineStage
from cantocaptions_ai.utils.debug import load_vad_debug, write_vad_debug
from cantocaptions_ai.utils.log_utils import get_logger

# get_logger initializes logging (including flop_counter suppression) before
# pyannote/lightning are imported below, which would otherwise emit a spurious
# triton-not-found warning from torch.utils.flop_counter.
logger = get_logger(__name__)

from cantocaptions_ai.pipeline.vads import Vad, Pyannote


class VadProcessor(PipelineStage["np.ndarray", "List[VadAudioSegment]"]):
    def __init__(
        self,
        vad_model: Vad,
        vad_onset: float,
        vad_offset: float,
        chunk_size: int,
        vad_pad_onset: float = 0.0,
        vad_pad_offset: float = 0.0,
        vad_min_duration_off: float = 0.0,
        reference_cues: Optional[List[SingleSegment]] = None,
        reference_padding: float = 0.5,
        cover_all: bool = False,
    ):
        self.vad_model = vad_model
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.chunk_size = chunk_size
        self.vad_pad_onset = vad_pad_onset
        self.vad_pad_offset = vad_pad_offset
        self.vad_min_duration_off = vad_min_duration_off
        self.reference_cues = reference_cues
        self.reference_padding = reference_padding
        # Emit a gap-free partition of the file instead of speech-only regions; see
        # Vad.cover_chunks. Reference expansion is meaningless here (nothing is dropped for
        # it to recover) and is skipped.
        self.cover_all = cover_all

    @staticmethod
    def read_debug(audio_path, debug_dir): return load_vad_debug(audio_path, debug_dir)

    @staticmethod
    def write_debug(audio_path, result, debug_dir): write_vad_debug(audio_path, result, debug_dir)

    @staticmethod
    def _extract(item):
        return load_audio(
            item['audio_path'],
            audio_track=item.get('audio_track', 0),
            downmix=item.get('audio_downmix', 'mix'),
        )

    @staticmethod
    def _pack(item, result):
        out = {'audio_path': item['audio_path'], 'vad_segments': result}
        if 'audio_track' in item:
            out['audio_track'] = item['audio_track']
        if 'audio_downmix' in item:
            out['audio_downmix'] = item['audio_downmix']
        return out

    def process(self, input: np.ndarray, *, progress_callback: ProgressCallback = None) -> List[VadAudioSegment]:
        """Run VAD on audio and return merged audio segments with timestamps."""
        logger.info("Performing voice activity detection...")
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(input)
            merge_chunks = self.vad_model.merge_chunks
            cover_chunks = self.vad_model.cover_chunks
            speech_regions = self.vad_model.speech_regions
        else:
            waveform = Pyannote.preprocess_audio(input)
            merge_chunks = Pyannote.merge_chunks
            cover_chunks = Pyannote.cover_chunks
            speech_regions = Pyannote.speech_regions

        duration = len(input) / SAMPLE_RATE
        raw_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        speech_spans: list = []
        if self.cover_all:
            # Split-only mode: VAD picks the cut points, but nothing is discarded. See
            # Vad.cover_chunks for why --realign cannot use the speech-only timeline.
            merged = cover_chunks(raw_segments, self.chunk_size, duration)
            logger.info(
                "Contiguous chunking: %d chunk(s) covering %.1fs (no audio discarded)",
                len(merged), duration,
            )
            # Split-only chunks keep every sample, which is the point, but it also means a
            # chunk says nothing about where inside itself anyone is speaking. Record the
            # speech turns separately: nothing is filtered by them, but the ASR anchor needs
            # them to turn a character stream into times (realign._hypothesis_stream), and
            # they cannot be recovered downstream once the score curve is gone.
            speech_spans = speech_regions(
                raw_segments,
                onset=self.vad_onset,
                offset=self.vad_offset,
                pad_onset=self.vad_pad_onset,
                pad_offset=self.vad_pad_offset,
                min_duration_off=self.vad_min_duration_off,
            )
            heard = sum(e - s for s, e in speech_spans)
            logger.info(
                "  ...of which %.1fs (%.0f%%) is speech, across %d region(s)",
                heard, 100 * heard / duration if duration else 0.0, len(speech_spans),
            )
        else:
            merged = merge_chunks(
                raw_segments,
                self.chunk_size,
                onset=self.vad_onset,
                offset=self.vad_offset,
                pad_onset=self.vad_pad_onset,
                pad_offset=self.vad_pad_offset,
                min_duration_off=self.vad_min_duration_off,
            )

        # Reference-cue expansion runs here -- after the timeline exists, before any
        # audio is sliced -- so the recovered regions are ordinary VAD segments to
        # everything downstream and land inside this stage's debug checkpoint.
        expanded_spans: list = []
        if self.reference_cues and not self.cover_all:
            from cantocaptions_ai.pipeline.reference_context import (
                expand_intervals_to_reference, expansion_only_spans,
            )
            # Record what the reference *added* before the union erases the distinction:
            # --asr_context_scope expanded prompts only over audio VAD never found.
            expanded_spans = expansion_only_spans(
                merged,
                self.reference_cues,
                padding=self.reference_padding,
                upper_bound=len(input) / SAMPLE_RATE,
            )
            # Hand over the raw probability curve too: when a unioned run exceeds
            # chunk_size and contains no silent gap, the least damaging cut is the
            # lowest-scoring frame -- the same signal Binarize._split_long uses.
            window = raw_segments.sliding_window
            times = np.array([window[i].middle for i in range(raw_segments.data.shape[0])])
            merged = expand_intervals_to_reference(
                merged,
                self.reference_cues,
                padding=self.reference_padding,
                chunk_size=self.chunk_size,
                upper_bound=len(input) / SAMPLE_RATE,
                times=times,
                scores=raw_segments.data[:, 0],
            )

        segments = []
        for seg in merged:
            f1 = int(seg['start'] * SAMPLE_RATE)
            f2 = int(seg['end'] * SAMPLE_RATE)
            out = {
                'start': seg['start'],
                'end': seg['end'],
                'audio': input[f1:f2],
            }
            # Clip the file-level expansion spans to this segment. Carried per segment
            # (rather than per file) so it survives the VAD and vocal-isolation debug
            # round-trips, which rebuild segment dicts from their manifests.
            overlap = [
                [max(sp[0], seg['start']), min(sp[1], seg['end'])]
                for sp in expanded_spans
                if sp[0] < seg['end'] and sp[1] > seg['start']
            ]
            if overlap:
                out['expanded'] = overlap
            heard = [
                [max(sp[0], seg['start']), min(sp[1], seg['end'])]
                for sp in speech_spans
                if sp[0] < seg['end'] and sp[1] > seg['start']
            ]
            if heard:
                out['speech'] = heard
            segments.append(out)
        if expanded_spans:
            recovered = sum(e - s for s, e in expanded_spans)
            logger.info(
                "Reference expansion added %.1fs of audio across %d span(s) that VAD "
                "did not detect", recovered, len(expanded_spans),
            )
        return segments

def load_vad(
    vad_method: str = "pyannote",
    device: str = "cpu",
    device_index: int = 0,
    vad_onset: float = 0.450,
    vad_offset: float = 0.300,
    chunk_size: int = 30,
    vad_model: Optional[Vad] = None,
    use_auth_token: Optional[Union[str, bool]] = None,
    vad_pad_onset: float = 0.20,
    vad_pad_offset: float = 0.20,
    vad_min_duration_off: float = 0.25,
    reference_cues: Optional[List[SingleSegment]] = None,
    reference_padding: float = 0.5,
    cover_all: bool = False,
) -> VadProcessor:
    """Load a VAD model and return a VadProcessor for audio segmentation."""
    if vad_model is not None:
        logger.info("Using manually assigned vad_model. vad_method is ignored.")
    else:
        if vad_method == "pyannote":
            vad_model = Pyannote(
                torch.device(resolve_device(device, device_index)),
                token=use_auth_token,
                vad_onset=vad_onset,
                vad_offset=vad_offset,
                chunk_size=chunk_size,
            )
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return VadProcessor(
        vad_model=vad_model,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        chunk_size=chunk_size,
        vad_pad_onset=vad_pad_onset,
        vad_pad_offset=vad_pad_offset,
        vad_min_duration_off=vad_min_duration_off,
        reference_cues=reference_cues,
        reference_padding=reference_padding,
        cover_all=cover_all,
    )

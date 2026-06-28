from typing import List, Optional, Union

import numpy as np
import torch

from cantocaptions_ai.utils.audio import SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.schema import ProgressCallback, VadAudioSegment
from cantocaptions_ai.utils.model_utils import PipelineStage
from cantocaptions_ai.pipeline.vads import Vad, Pyannote
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


class VadProcessor(PipelineStage["np.ndarray", "List[VadAudioSegment]"]):
    def __init__(
        self,
        vad_model: Vad,
        vad_onset: float,
        vad_offset: float,
        chunk_size: int,
    ):
        self.vad_model = vad_model
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.chunk_size = chunk_size

    def process(self, input: np.ndarray, *, progress_callback: ProgressCallback = None) -> List[VadAudioSegment]:
        """Run VAD on audio and return merged audio segments with timestamps."""
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(input)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(input)
            merge_chunks = Pyannote.merge_chunks

        raw_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        merged = merge_chunks(
            raw_segments,
            self.chunk_size,
            onset=self.vad_onset,
            offset=self.vad_offset,
        )

        segments = []
        for seg in merged:
            f1 = int(seg['start'] * SAMPLE_RATE)
            f2 = int(seg['end'] * SAMPLE_RATE)
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'audio': input[f1:f2],
            })
        return segments

def load_vad(
    vad_method: str = "pyannote",
    device: str = "cpu",
    device_index: int = 0,
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
    chunk_size: int = 30,
    vad_model: Optional[Vad] = None,
    use_auth_token: Optional[Union[str, bool]] = None,
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
    )

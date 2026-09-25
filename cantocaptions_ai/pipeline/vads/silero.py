"""Silero VAD v6 as a score-curve backend -- the CPU option (see config/cpu.cfg).

The model ships vendored in ``cantocaptions_ai/assets`` as the TorchScript file from the
silero-vad 6.2.3 wheel (MIT, see ``silero_vad_LICENSE.txt``), so there is no pip dependency
and nothing to download. It scores one 512-sample frame (32 ms at 16 kHz) at a time.

The model is recurrent and its state carries a long memory, so the file is scored in one
serial pass. Splitting it into pieces scored as a batch would be ~5-15x faster (the batch
itself is exact), but a piece that starts cold never re-converges onto the serial curve:
measured on film audio, even with 64 s of warm-up audio in front of it, ~3% of the
following minute's frames still differ by more than 0.05. That is a different detector,
not a faster one.
"""
import os

import numpy as np
import torch

from cantocaptions_ai.pipeline.vads.curve import CurveVad
from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

FRAME_SAMPLES = 512  # the only window size the 16 kHz model accepts
_THREADS = 2  # see _score

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "silero_vad_v6.jit",
)


class Silero(CurveVad):
    """Silero VAD. Always runs on CPU: the model steps one 32 ms frame at a time, so a GPU
    only adds a kernel launch per frame -- measured no faster than CPU, and it would take
    VRAM from the stages that need it."""

    def __init__(self, **kwargs):
        logger.info("Performing voice activity detection using Silero v6...")
        super().__init__(kwargs["vad_onset"])
        self.model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()

    def __call__(self, audio, **kwargs):
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        waveform = audio["waveform"]
        if audio.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE:
            raise ValueError(f"Silero expects {SAMPLE_RATE} Hz audio")
        probs = self._score(waveform.reshape(-1).float())
        step = FRAME_SAMPLES / SAMPLE_RATE
        return SlidingWindowFeature(
            probs[:, None], SlidingWindow(start=0.0, duration=step, step=step)
        )

    def _score(self, wav: torch.Tensor) -> np.ndarray:
        """One speech probability per 512-sample frame, the last frame zero-padded."""
        # Each step is a tiny matmul, so intra-op threads stop paying almost at once:
        # measured 36 s/h on 1 thread, 23 s/h on 2, and no faster on 4 or 12 -- which
        # burn 3-7x the CPU time for it. Capped for the call, then restored.
        threads = torch.get_num_threads()
        torch.set_num_threads(min(threads, _THREADS))
        try:
            with torch.inference_mode():
                self.model.reset_states()
                out = self.model.audio_forward(wav, SAMPLE_RATE)
        finally:
            torch.set_num_threads(threads)
        return out[0].numpy().astype(np.float32)

    @staticmethod
    def preprocess_audio(audio):
        return torch.from_numpy(audio).unsqueeze(0)

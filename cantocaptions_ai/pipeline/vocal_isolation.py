import io
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.schema import ProgressCallback, VadAudioSegment
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# _PACKAGE_DIR is cantocaptions_ai/pipeline/; repo root is two levels up
_DEFAULT_MBROFORMER_DIR = os.path.join(os.path.dirname(os.path.dirname(_PACKAGE_DIR)), "mb-roformer")

_MBROFORMER_MODEL_TYPE = "mel_band_roformer"
_MBROFORMER_CONFIG_NAME = "config_vocals_mel_band_roformer.yaml"
_MBROFORMER_CHECKPOINT_NAME = "MelBandRoformer.ckpt"


_DURATION_TOLERANCE_S = 0.005  # seconds


class VocalIsolationProcessor:
    """Base class for vocal isolation processors."""

    def isolate(self, segments: List[VadAudioSegment], progress_callback: ProgressCallback = None) -> List[VadAudioSegment]:
        """Run isolation then validate that segment durations are consistent."""
        result = self._isolate(segments, progress_callback=progress_callback)
        for seg in result:
            expected = seg["end"] - seg["start"]
            actual = len(seg["audio"]) / SAMPLE_RATE
            if abs(actual - expected) > _DURATION_TOLERANCE_S:
                logger.warning(
                    "Segment duration mismatch after vocal isolation: "
                    f"timestamps span {expected:.3f}s but audio is {actual:.3f}s "
                    f"(start={seg['start']:.3f}, end={seg['end']:.3f})"
                )
        return result

    def _isolate(self, segments: List[VadAudioSegment], progress_callback: ProgressCallback = None) -> List[VadAudioSegment]:
        """Replace each segment's audio with its vocals-only audio. Subclasses must override."""
        ...


class MbRoformerProcessor(VocalIsolationProcessor):
    """Vocal isolation processor backed by the Mel-Band RoFormer model."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: ConfigDict,
        device: torch.device,
        demix_track_fn,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model_sample_rate: int = config.model.sample_rate
        self._demix_track = demix_track_fn

    def _isolate(self, segments: List[VadAudioSegment], progress_callback: ProgressCallback = None) -> List[VadAudioSegment]:
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for vocal isolation: pip install librosa"
            )

        self.model.eval()
        result = []
        n = len(segments)

        for idx, seg in enumerate(segments):
            # Suppress demix_track's per-chunk stdout prints
            _old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                isolated = self._isolate_segment(seg["audio"], librosa)
            finally:
                sys.stdout = _old_stdout

            result.append({"start": seg["start"], "end": seg["end"], "audio": isolated})
            if progress_callback is not None:
                progress_callback((idx + 1) / n * 100)

        return result

    def _isolate_segment(self, audio: np.ndarray, librosa) -> np.ndarray:
        # Resample from project rate (16 kHz) to model's expected rate (44.1 kHz)
        if SAMPLE_RATE != self.model_sample_rate:
            audio_at_model_sr = librosa.resample(
                audio, orig_sr=SAMPLE_RATE, target_sr=self.model_sample_rate
            )
        else:
            audio_at_model_sr = audio

        # Build stereo tensor (channels, samples) as expected by the model
        stereo = np.stack([audio_at_model_sr, audio_at_model_sr], axis=0)
        mixture = torch.tensor(stereo, dtype=torch.float32)

        res, _ = self._demix_track(self.config, self.model, mixture, self.device, None)

        vocals = res[self.config.training.target_instrument]  # shape: (2, samples)

        # Average stereo channels to mono, then resample back to project rate
        vocals_mono = vocals.mean(axis=0).astype(np.float32)
        if SAMPLE_RATE != self.model_sample_rate:
            vocals_mono = librosa.resample(
                vocals_mono, orig_sr=self.model_sample_rate, target_sr=SAMPLE_RATE
            )

        return vocals_mono


def load_vocal_isolation(
    model_name: str,
    device: str,
    device_index: int = 0,
    model_dir: Optional[str] = None,
) -> VocalIsolationProcessor:
    """Load a vocal isolation model and return a processor."""
    if model_name != "mbroformer":
        raise ValueError(
            f"Unknown vocal isolation model '{model_name}'. Supported: 'mbroformer'"
        )

    model_dir = model_dir or _DEFAULT_MBROFORMER_DIR
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"mb-roformer directory not found: {model_dir}")

    # mb-roformer is not an installed package — add its root to sys.path so that
    # its local `utils` module and `models/` package can be imported.
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from utils import demix_track, get_model_from_config  # noqa: PLC0415

    config_path = os.path.join(model_dir, "configs", _MBROFORMER_CONFIG_NAME)
    checkpoint_path = os.path.join(model_dir, _MBROFORMER_CHECKPOINT_NAME)

    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    logger.info("Loading vocal isolation model (MelBandRoformer)...")
    torch_model = get_model_from_config(_MBROFORMER_MODEL_TYPE, config)

    if os.path.exists(checkpoint_path):
        torch_model.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device("cpu"))
        )
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")

    torch_device = (
        torch.device(f"cuda:{device_index}") if device == "cuda" else torch.device(device)
    )
    torch_model = torch_model.to(torch_device)

    return MbRoformerProcessor(
        model=torch_model,
        config=config,
        device=torch_device,
        demix_track_fn=demix_track,
    )

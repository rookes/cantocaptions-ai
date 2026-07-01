import io
import importlib.resources
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download

from cantocaptions_ai.pipeline.mbroformer.model import MelBandRoformer
from cantocaptions_ai.utils.audio import SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.schema import ProgressCallback, VadAudioSegment
from cantocaptions_ai.utils.model_utils import PipelineStage
from cantocaptions_ai.utils.debug import load_isolation_debug, write_isolation_debug
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_HF_REPO_ID = "KimberleyJSN/melbandroformer"
_HF_FILENAME = "MelBandRoformer.ckpt"

_DURATION_TOLERANCE_S = 0.005  # seconds


# ---------------------------------------------------------------------------
# Inference helpers (ported from mb-roformer/utils.py)
# ---------------------------------------------------------------------------

def _get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


def _demix_track(config, model, mix, device, first_chunk_time=None):
    C = config.inference.chunk_size
    N = config.inference.num_overlap
    step = C // N
    fade_size = C // 10
    border = C - step

    if mix.shape[1] > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    windowing_array = _get_windowing_array(C, fade_size, device)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1,) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

            i = 0
            total_length = mix.shape[1]
            num_chunks = (total_length + step - 1) // step

            if first_chunk_time is None:
                start_time = time.time()
                first_chunk = True
            else:
                start_time = None
                first_chunk = False

            while i < total_length:
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

                if first_chunk and i == 0:
                    chunk_start_time = time.time()

                x = model(part.unsqueeze(0))[0]

                window = windowing_array.clone()
                if i == 0:
                    window[:fade_size] = 1
                elif i + C >= total_length:
                    window[-fade_size:] = 1

                result[..., i:i + length] += x[..., :length] * window[..., :length]
                counter[..., i:i + length] += window[..., :length]
                i += step

                if first_chunk and i == step:
                    chunk_time = time.time() - chunk_start_time
                    first_chunk_time = chunk_time
                    estimated_total_time = chunk_time * num_chunks
                    print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
                    first_chunk = False

                if first_chunk_time is not None and i > step:
                    chunks_processed = i // step
                    time_remaining = first_chunk_time * (num_chunks - chunks_processed)
                    sys.stdout.write(f"\rEstimated time remaining: {time_remaining:.2f} seconds")
                    sys.stdout.flush()

            print()
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if mix.shape[1] > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}, first_chunk_time
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}, first_chunk_time


# ---------------------------------------------------------------------------
# Processor classes
# ---------------------------------------------------------------------------

class VocalIsolationProcessor(PipelineStage["List[VadAudioSegment]", "List[VadAudioSegment]"]):
    """Base class for vocal isolation processors."""

    @staticmethod
    def read_debug(audio_path, debug_dir): return load_isolation_debug(audio_path, debug_dir)

    @staticmethod
    def write_debug(audio_path, result, debug_dir): write_isolation_debug(audio_path, result, debug_dir)

    @staticmethod
    def _extract(item): return item['vad_segments']

    @staticmethod
    def _pack(item, result): return {'audio_path': item['audio_path'], 'vad_segments': result}

    def process(self, input: List[VadAudioSegment], *, progress_callback: ProgressCallback = None) -> List[VadAudioSegment]:
        """Run isolation then validate that segment durations are consistent."""
        logger.info("Performing vocal isolation...")
        result = self._isolate(input, progress_callback=progress_callback)
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
        config: DictConfig,
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
                progress_callback((idx + 1) / n)

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


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_vocal_isolation(
    model_name: str,
    device: str,
    device_index: int = 0,
    model_dir: Optional[str] = None,
) -> VocalIsolationProcessor:
    """Load a vocal isolation model and return a processor.

    The checkpoint is downloaded from HuggingFace on first use and cached.
    model_dir, if given, overrides the default HuggingFace cache directory.
    """
    if model_name != "mbroformer":
        raise ValueError(
            f"Unknown vocal isolation model '{model_name}'. Supported: 'mbroformer'"
        )

    # Load bundled config
    config_ref = importlib.resources.files("cantocaptions_ai.assets").joinpath(
        "config_vocals_mel_band_roformer.yaml"
    )
    with importlib.resources.as_file(config_ref) as config_path:
        config = OmegaConf.load(config_path)

    # Instantiate model — convert OmegaConf container to plain Python types so
    # beartype is satisfied, and restore the tuple expected by the constructor.
    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    model_kwargs["multi_stft_resolutions_window_sizes"] = tuple(
        model_kwargs["multi_stft_resolutions_window_sizes"]
    )
    torch_model = MelBandRoformer(**model_kwargs)

    # Download checkpoint from HuggingFace (cached after first download)
    logger.info("Loading vocal isolation model (MelBandRoformer)...")
    checkpoint_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_FILENAME,
        cache_dir=model_dir,
    )
    torch_model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))
    )

    torch_device = torch.device(resolve_device(device, device_index))
    torch_model = torch_model.to(torch_device)

    return MbRoformerProcessor(
        model=torch_model,
        config=config,
        device=torch_device,
        demix_track_fn=_demix_track,
    )

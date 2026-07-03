import importlib.resources
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download

from cantocaptions_ai.pipeline.mbroformer.model import MelBandRoformer
from cantocaptions_ai.utils.audio import SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.schema import ProgressCallback, VadAudioSegment
from cantocaptions_ai.utils.model_utils import PipelineStage, partition_by_cache, run_adaptive_batches
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
        # Vocal isolation batches chunks across files, so it overrides run() rather
        # than processing one file at a time via the base run()/process() path.
        raise NotImplementedError("VocalIsolationProcessor drives work through run(), not process()")


def _validate_segment_duration(start: float, end: float, audio: np.ndarray) -> None:
    expected = end - start
    actual = len(audio) / SAMPLE_RATE
    if abs(actual - expected) > _DURATION_TOLERANCE_S:
        logger.warning(
            "Segment duration mismatch after vocal isolation: "
            f"timestamps span {expected:.3f}s but audio is {actual:.3f}s "
            f"(start={start:.3f}, end={end:.3f})"
        )


class MbRoformerProcessor(VocalIsolationProcessor):
    """Vocal isolation processor backed by the Mel-Band RoFormer model.

    The model runs on fixed-size chunks of ``config.inference.chunk_size`` samples, so
    the batch unit is the chunk (identical size → no padding). Chunks from every
    to-compute segment across every file are packed into batches (cross-file backfill);
    each segment reconstructs via overlap-add into its own CPU buffer and is finalized
    (and its buffers freed) as soon as its last chunk completes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: DictConfig,
        device: torch.device,
        batch_size: Optional[int] = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model_sample_rate: int = config.model.sample_rate
        self._batch_size = batch_size
        self._C = config.inference.chunk_size
        self._step = self._C // config.inference.num_overlap
        self._fade = self._C // 10
        self._border = self._C - self._step

    def run(self, items, *, debug_dir=None, load_debug_dir=None, progress_callback: ProgressCallback = None):
        logger.info("Performing vocal isolation...")
        self.model.eval()

        cached, to_compute = partition_by_cache(items, self.read_debug, load_debug_dir)

        C, step, fade, border = self._C, self._step, self._fade, self._border
        base_window = _get_windowing_array(C, fade, torch.device("cpu")).numpy()

        # Per-segment overlap-add state keyed by (item_idx, seg_idx); jobs reference it.
        seg_state: Dict[Tuple[int, int], dict] = {}
        # Per-item finalized audio: idx -> {'n': int, 'segs': {seg_idx: np.ndarray}, 'audio_path': str}
        item_out: Dict[int, dict] = {}
        jobs: List[Tuple[Tuple[int, int], int]] = []

        for idx, item in to_compute:
            segs = item['vad_segments']
            item_out[idx] = {'n': len(segs), 'segs': {}, 'audio_path': item['audio_path']}
            for sdx, seg in enumerate(segs):
                mixture, total_length, padded = self._prepare_mixture(seg['audio'])
                key = (idx, sdx)
                offsets = list(range(0, total_length, step))
                seg_state[key] = {
                    'mixture': mixture,
                    'total_length': total_length,
                    'padded': padded,
                    'result': np.zeros((2, total_length), dtype=np.float32),
                    'counter': np.zeros((2, total_length), dtype=np.float32),
                    'remaining': len(offsets),
                    'start': seg['start'],
                    'end': seg['end'],
                }
                jobs.extend((key, off) for off in offsets)

        if progress_callback is not None:
            progress_callback.set_total(len(jobs), unit="chunk")

        def infer_fn(batch):
            parts = []
            for key, off in batch:
                part = seg_state[key]['mixture'][:, off:off + C]
                plen = part.shape[-1]
                if plen < C:
                    if plen > C // 2 + 1:
                        part = nn.functional.pad(part, (0, C - plen), mode='reflect')
                    else:
                        part = nn.functional.pad(part, (0, C - plen), mode='constant', value=0)
                parts.append(part)
            batch_t = torch.stack(parts, dim=0).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                with torch.no_grad():
                    out = self.model(batch_t)  # (B, 2, C) for the single-stem vocals model
            out = out.float().cpu().numpy()
            for bi, (key, off) in enumerate(batch):
                st = seg_state[key]
                total_length = st['total_length']
                length = min(C, total_length - off)
                window = base_window.copy()
                if off == 0:
                    window[:fade] = 1.0
                elif off + C >= total_length:
                    window[-fade:] = 1.0
                st['result'][:, off:off + length] += out[bi][:, :length] * window[:length]
                st['counter'][:, off:off + length] += window[:length]
                st['remaining'] -= 1
                if st['remaining'] == 0:
                    self._finalize_segment(key, st, item_out)
                    del seg_state[key]

        run_adaptive_batches(jobs, self._batch_size, infer_fn, reporter=progress_callback)

        # Assemble per-item results and write debug for freshly computed items.
        computed: Dict[int, List[VadAudioSegment]] = {}
        for idx, meta in item_out.items():
            ordered = [meta['segs'][s] for s in range(meta['n'])]
            computed[idx] = ordered
            if debug_dir is not None:
                self.write_debug(meta['audio_path'], ordered, debug_dir)

        result_items = []
        for idx, item in enumerate(items):
            segs_out = cached[idx] if idx in cached else computed[idx]
            result_items.append(self._pack(item, segs_out))
        return result_items

    def _prepare_mixture(self, audio: np.ndarray):
        """Resample a 16 kHz mono segment to the model rate, build the stereo mixture,
        and apply the reflect border pad. Returns (mixture[2,L], total_length, padded).

        Uses torchaudio (not librosa's default soxr_hq) to resample: soxr_hq pays a
        ~3.7s one-time cold-start cost on its first call and is ~5x slower per call
        thereafter than torchaudio.functional.resample.
        """
        audio_t = torch.from_numpy(audio)
        if SAMPLE_RATE != self.model_sample_rate:
            audio_t = torchaudio.functional.resample(audio_t, SAMPLE_RATE, self.model_sample_rate)
        mixture = torch.stack([audio_t, audio_t], dim=0).float()
        padded = False
        if mixture.shape[1] > 2 * self._border and self._border > 0:
            mixture = nn.functional.pad(mixture, (self._border, self._border), mode='reflect')
            padded = True
        return mixture, mixture.shape[1], padded

    def _finalize_segment(self, key, st, item_out) -> None:
        idx, sdx = key
        estimated = st['result'] / st['counter']
        np.nan_to_num(estimated, copy=False, nan=0.0)
        if st['padded']:
            estimated = estimated[:, self._border:-self._border]
        vocals_mono = estimated.mean(axis=0).astype(np.float32)
        if SAMPLE_RATE != self.model_sample_rate:
            vocals_mono = torchaudio.functional.resample(
                torch.from_numpy(vocals_mono), self.model_sample_rate, SAMPLE_RATE
            ).numpy()
        _validate_segment_duration(st['start'], st['end'], vocals_mono)
        item_out[idx]['segs'][sdx] = {'start': st['start'], 'end': st['end'], 'audio': vocals_mono}
        # Free the heavy buffers now that this segment is done.
        st['mixture'] = st['result'] = st['counter'] = None


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_vocal_isolation(
    model_name: str,
    device: str,
    device_index: int = 0,
    model_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> VocalIsolationProcessor:
    """Load a vocal isolation model and return a processor.

    The checkpoint is downloaded from HuggingFace on first use and cached.
    model_dir, if given, overrides the default HuggingFace cache directory.
    batch_size controls how many fixed-size chunks are run through the model at once.
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
        batch_size=batch_size,
    )

import importlib.resources
import time
from contextlib import contextmanager
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
from cantocaptions_ai.utils.model_utils import (
    PipelineStage,
    partition_by_cache,
    write_checkpoint,
    BatchExecutor,
    check_vram_headroom,
    ensure_hf_file_downloaded,
    guard_model_load,
    resolve_torch_compute_dtype,
)
from cantocaptions_ai.utils.debug import (_PERSISTED_SEGMENT_KEYS, load_isolation_debug,
                                          write_isolation_debug)
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_HF_REPO_ID = "KimberleyJSN/melbandroformer"
_HF_FILENAME = "MelBandRoformer.ckpt"

# Rough fp32 params + activation footprint for MelBandRoformer; used only for the
# preflight VRAM-headroom warning, not an exact bound.
_VOCAL_ISOLATION_VRAM_ESTIMATE_MB = 1200
_VOCAL_ISOLATION_REMEDIATION = "pass --vocal_isolation_method none to skip vocal isolation"

_DURATION_TOLERANCE_S = 0.005  # seconds

# Caps how many segments' worth of overlap-add buffers (mixture/result/counter,
# each a (2, total_length) float32 array) can be concurrently alive in
# MbRoformerProcessor.run(). The primary windowing unit is "one file" (see
# _iter_windows); this cap only kicks in defensively for a single file with more
# segments than this — VAD's max_duration bounds individual segment length
# (via --chunk_size) but nothing bounds how many segments one very long file can
# produce. At ~40MB/segment for a typical ~30s segment (see the RAM-exhaustion
# investigation this guards against), 128 segments is ~5GB worst case —
# comfortably under typical machine RAM and far above a normal ~50 segs/file, so
# it essentially never triggers in normal operation.
_MAX_SEGMENTS_PER_WINDOW = 128

# torch.compile for the chunked path (--vocal_isolation_compile). Every chunk is the same
# (batch, 2, chunk_size) shape, so one static graph serves a whole run, once each file's
# short last batch is padded up to size. Measured on a 3080 Ti at batch 4: eager ~152
# ms/chunk, compiled ~77 ms/chunk, output within ~1e-6. That is not batching paying off:
# eager spends most of its time on many small elementwise kernels and their launch
# overhead, which compiling fuses away. The cost is paid once per process: ~12 s with a
# warm Inductor disk cache, ~50 s the first time on a machine. A model loaded again in the
# same process (the worker's next job) reuses the compiled code in ~0.3 s.
#
# "auto" compiles once a run is long enough to repay a warm compile: ~75 ms saved per
# chunk against ~12 s is ~160 chunks, so 200 (about 13 minutes of speech at the 4 s
# step). It always compiles when this process already has, since that costs nothing.
_COMPILE_MIN_CHUNKS = 200
# (batch, dtype, device) shapes this process has compiled and run.
_COMPILED_SHAPES: set = set()
# Set when a compile fails, so the process stops trying (e.g. no working Triton).
_COMPILE_FAILED = False


@contextmanager
def _deterministic():
    """Deterministic algorithms for the compiled model's calls, and only those.

    Compiled, the model is not reproducible on its own: Inductor picks among reduction
    kernel configurations by benchmarking them as each process starts, and the choice
    changes the float rounding, so the same input gave a few different outputs across runs
    (up to ~4e-4). With deterministic algorithms on, Inductor uses one fixed configuration
    instead. Together with istft running eagerly (see mbroformer/model.py), output is
    bit-identical from run to run, at no measurable cost in speed. The setting is
    process-wide, so it is restored after each call rather than left on for the rest of the
    pipeline. Compiling happens inside the first call, so the graph is built under the same
    setting it runs under.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _compile_model(model):
    """torch.compile for one fixed chunk shape. A hook so tests can stand in for it."""
    return torch.compile(model, dynamic=False)


def _can_compile(device: torch.device) -> bool:
    """Whether "auto" may compile here: CUDA with a Triton that Inductor can use."""
    if device.type != "cuda" or _COMPILE_FAILED:
        return False
    try:
        from torch.utils._triton import has_triton
    except ImportError:
        return False
    return has_triton()


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

    debug_stage = "vocal_isolation"

    @staticmethod
    def read_debug(audio_path, debug_dir): return load_isolation_debug(audio_path, debug_dir)

    @staticmethod
    def write_debug(audio_path, result, debug_dir): write_isolation_debug(audio_path, result, debug_dir)

    @staticmethod
    def _extract(item): return item['vad_segments']

    @staticmethod
    def _pack(item, result): return {**item, 'vad_segments': result}

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

    Two modes, selected by ``config.inference.mode``:

    ``chunked`` (default) is the original sliding-window path, described below.

    ``whole`` runs one forward pass per segment at its natural length, dropping
    both the 2x overlap redundancy and the chunk seams for ~2.5x the throughput.
    It is not the default despite that: overlap-add averaging turns out to act as
    test-time ensembling, and losing it costs ~5% relative CER (see the yaml's
    ``mode`` comment for the measurement). Segments longer than ``whole_max_s``
    fall back to ``chunked`` per segment, bounding peak VRAM.

    The model runs on fixed-size chunks of ``config.inference.chunk_size`` samples, so
    the batch unit is the chunk (identical size → no padding). Segments are processed
    one file at a time (see ``_iter_windows``, sub-chunked defensively at
    ``_MAX_SEGMENTS_PER_WINDOW`` for a pathologically long single file): each window's
    segments get their overlap-add scratch buffers (mixture/result/counter) built,
    batched through the model, and freed via ``_finalize_segment`` before the next
    window starts — bounding peak host RAM to one window's segments instead of the
    entire dataset. This trades away batching chunks across file boundaries (at most
    ``batch_size - 1`` chunks go under-full in each file's last batch) in exchange for
    that bound.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: DictConfig,
        device: torch.device,
        batch_size: Optional[int] = None,
        segment_mode: Optional[str] = None,
        compile: str = "off",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model_sample_rate: int = config.model.sample_rate
        self._batch_size = batch_size
        if compile not in ("auto", "on", "off"):
            raise ValueError(f"unknown vocal isolation compile setting {compile!r} "
                             "(expected 'auto', 'on' or 'off')")
        self._compile = compile
        # Set per run by _choose_forward: the compiled model, or None to run eagerly.
        self._compiled = None
        self._compile_started = None
        # The batch size short batches are padded to. Halved when a padded batch runs out
        # of memory, following BatchExecutor's own retry down.
        self._pad_to = batch_size or 1
        inference = config.inference
        self._C = inference.chunk_size
        self._step = self._C // inference.num_overlap
        self._fade = self._C // 10

        self._mode = segment_mode or inference.get("mode", "chunked")
        if self._mode not in ("chunked", "whole"):
            raise ValueError(
                f"unknown vocal isolation mode {self._mode!r} (expected 'chunked' or 'whole')"
            )
        if self._mode == "whole":
            # Whole mode has no step, so the border is an explicit context pad
            # rather than the overlap-add ramp width.
            self._border = int(round(float(inference.get("border_s", 1.0)) * self.model_sample_rate))
            self._whole_max = int(round(float(inference.get("whole_max_s", 35.0)) * self.model_sample_rate))
        else:
            self._border = self._C - self._step
            self._whole_max = 0

    def run(self, items, *, debug_dir=None, load_debug_dir=None, progress_callback: ProgressCallback = None):
        logger.info("Performing vocal isolation...")
        self.model.eval()

        cached, to_compute = partition_by_cache(items, self, load_debug_dir)

        C, step, fade, border = self._C, self._step, self._fade, self._border
        base_window = _get_windowing_array(C, fade, torch.device("cpu")).numpy()

        # Per-item finalized audio: idx -> {'n': int, 'segs': {seg_idx: np.ndarray}, 'item': dict}
        # Bookkeeping only (no audio buffers) — safe to build for every to-compute item
        # up front regardless of dataset size; segments are filled in incrementally by
        # _finalize_segment as each window below completes.
        item_out: Dict[int, dict] = {
            idx: {'n': len(item['vad_segments']), 'segs': {}, 'item': item}
            for idx, item in to_compute
        }

        # Cheap approximate total (no resampling/allocation). It sizes the progress bar
        # as a single continuous bar across every window below instead of resetting
        # once per file (StageTimer._start_determinate closes/replaces the bar on
        # every call, so set_total must only be called once, up front), and decides
        # whether compiling pays.
        total_jobs = sum(
            self._estimate_num_offsets(seg)
            for _, item in to_compute
            for seg in item['vad_segments']
        )
        if progress_callback is not None:
            progress_callback.set_total(total_jobs, unit="chunk")
        self._compiled = self._choose_forward(total_jobs)

        for window in self._iter_windows(to_compute):
            seg_state: Dict[Tuple[int, int], dict] = {}
            jobs: List[Tuple[Tuple[int, int], int]] = []
            for idx, sdx, seg in window:
                mixture, total_length, padded = self._prepare_mixture(seg['audio'])
                key = (idx, sdx)
                if self._mode == "whole" and total_length <= self._whole_max:
                    # One forward pass at the segment's natural length. No offsets,
                    # no overlap-add buffers, no seams -- and no batching, since
                    # lengths vary and batching this model is measured to be flat
                    # (150 ms/chunk at every batch size that fits).
                    self._run_whole(key, seg, mixture, padded, item_out)
                    if progress_callback is not None:
                        progress_callback.advance(1)
                    continue
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
                    'src_len': len(seg['audio']),
                    **{k: seg[k] for k in _PERSISTED_SEGMENT_KEYS if k in seg},
                }
                jobs.extend((key, off) for off in offsets)

            def infer_fn(batch, seg_state=seg_state):
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
                out = self._separate_chunks(batch_t)  # (B, 2, C) for the single-stem vocals model
                out = out.float().cpu().numpy()
                for bi, (key, off) in enumerate(batch):
                    st = seg_state[key]
                    total_length = st['total_length']
                    length = min(C, total_length - off)
                    window_arr = base_window.copy()
                    if off == 0:
                        window_arr[:fade] = 1.0
                    elif off + C >= total_length:
                        window_arr[-fade:] = 1.0
                    st['result'][:, off:off + length] += out[bi][:, :length] * window_arr[:length]
                    st['counter'][:, off:off + length] += window_arr[:length]
                    st['remaining'] -= 1
                    if st['remaining'] == 0:
                        estimated = st['result'] / st['counter']
                        np.nan_to_num(estimated, copy=False, nan=0.0)
                        self._finalize_segment(key, st, estimated, item_out)
                        st['mixture'] = st['result'] = st['counter'] = None
                        del seg_state[key]

            # No order_key: chunks are all the fixed model chunk_size, so batches never
            # pad and GPU shapes don't grow — input order is fine (see docs/memory_batching_decoupling.md).
            BatchExecutor(self._batch_size).run(jobs, infer_fn, reporter=progress_callback)

        # Assemble per-item results and write debug for freshly computed items.
        computed: Dict[int, List[VadAudioSegment]] = {}
        for idx, meta in item_out.items():
            ordered = [meta['segs'][s] for s in range(meta['n'])]
            computed[idx] = ordered
            write_checkpoint(self, meta['item'], ordered, debug_dir)

        result_items = []
        for idx, item in enumerate(items):
            segs_out = cached[idx] if idx in cached else computed[idx]
            result_items.append(self._pack(item, segs_out))
        return result_items

    @staticmethod
    def _iter_windows(to_compute):
        """Yield windows of (item_idx, seg_idx, seg) triples, each window sized to
        bound how many segments' overlap-add buffers run() holds concurrently.

        One file per window, normally; sub-chunked at _MAX_SEGMENTS_PER_WINDOW if a
        single file alone exceeds it. This is what bounds run()'s peak memory
        independent of total dataset size — see _MAX_SEGMENTS_PER_WINDOW's docstring.
        """
        for idx, item in to_compute:
            segs = item['vad_segments']
            for start in range(0, len(segs), _MAX_SEGMENTS_PER_WINDOW):
                chunk = segs[start:start + _MAX_SEGMENTS_PER_WINDOW]
                yield [(idx, start + offset, seg) for offset, seg in enumerate(chunk)]

    def _estimate_num_offsets(self, seg) -> int:
        """Cheap approximate chunk count for a segment, for the progress bar's total
        only — mirrors _prepare_mixture's resample+pad arithmetic without actually
        resampling or allocating buffers. May be off by about one chunk vs. the real
        count (torchaudio's exact resampled length can differ slightly from a naive
        round()); harmless since only the progress bar consumes this, never buffer
        sizing (which always uses _prepare_mixture's real, exact total_length).
        """
        approx_len = round(len(seg['audio']) * self.model_sample_rate / SAMPLE_RATE)
        if approx_len > 2 * self._border and self._border > 0:
            approx_len += 2 * self._border
        if approx_len <= 0:
            return 0
        if self._mode == "whole" and approx_len <= self._whole_max:
            return 1
        return len(range(0, approx_len, self._step))

    def _shape_key(self, batch: int):
        params = getattr(self.model, "parameters", None)
        first = next(params(), None) if params is not None else None
        return (batch, first.dtype if first is not None else None, str(self.device))

    def _choose_forward(self, n_chunks: int):
        """The compiled model for this run's chunks, or None to run eagerly.

        Only chunked mode compiles: its chunks share one shape, while whole mode's
        segments are each a new length. A fixed batch size is needed for the same reason.
        """
        if (self._compile == "off" or self._mode != "chunked" or not self._batch_size
                or n_chunks == 0 or _COMPILE_FAILED):
            return None
        warm = self._shape_key(self._pad_to) in _COMPILED_SHAPES
        if self._compile == "auto" and not (
            _can_compile(self.device) and (warm or n_chunks >= _COMPILE_MIN_CHUNKS)
        ):
            return None
        if not warm:
            logger.info("Compiling the vocal isolation model for %d-chunk batches (once per "
                        "process; about 10-50 s)...", self._pad_to)
        # The model's rotary embeddings (rotary_embedding_torch) fill a frequency cache on
        # their first call, and the cache's length is part of what a compiled graph assumes,
        # so compiling before that call means compiling twice. One eager pass fills it; a
        # single chunk does, since the cache is sized by the chunk's length, not the batch.
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            with torch.no_grad():
                self.model(torch.zeros((1, 2, self._C), device=self.device))
        self._compile_started = None if warm else time.perf_counter()
        return _compile_model(self.model)

    def _separate_chunks(self, batch_t: torch.Tensor) -> torch.Tensor:
        """Run a batch of chunks through the model, compiled when the run chose it.

        The compiled model is fed one batch size only. A file's short last batch is padded
        with silent chunks and their output dropped, so it doesn't trigger a recompile.
        """
        global _COMPILE_FAILED
        n = batch_t.shape[0]
        compiled = self._compiled
        if compiled is not None:
            target = max(n, self._pad_to)
            if n < target:
                batch_t = torch.cat([batch_t, batch_t.new_zeros((target - n, *batch_t.shape[1:]))])
            try:
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    with torch.no_grad(), _deterministic():
                        out = compiled(batch_t)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    # BatchExecutor retries at half the batch; pad to that from now on.
                    self._pad_to = max(1, target // 2)
                    raise
                logger.warning("Compiling the vocal isolation model failed; running it "
                               "uncompiled for the rest of this process. %s: %s",
                               type(e).__name__, e)
                _COMPILE_FAILED = True
                self._compiled = None
                batch_t = batch_t[:n]
            else:
                _COMPILED_SHAPES.add(self._shape_key(target))
                if self._compile_started is not None:
                    logger.info("Compiled the vocal isolation model in %.0f s",
                                time.perf_counter() - self._compile_started)
                    self._compile_started = None
                return out[:n]
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            with torch.no_grad():
                return self.model(batch_t)

    def _run_whole(self, key, seg, mixture, padded, item_out) -> None:
        """Separate one segment in a single forward pass at its natural length.

        The chunked path exists because the reference harness slid a fixed
        chunk_size window; the model itself is a transformer over STFT frames and
        accepts any length. Running the segment whole removes the 2x overlap
        redundancy and the chunk seams at once.
        """
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            with torch.no_grad():
                out = self.model(mixture.unsqueeze(0).to(self.device))
        estimated = out[0].float().cpu().numpy()
        st = {
            'padded': padded,
            'start': seg['start'],
            'end': seg['end'],
            'src_len': len(seg['audio']),
            **{k: seg[k] for k in _PERSISTED_SEGMENT_KEYS if k in seg},
        }
        self._finalize_segment(key, st, estimated, item_out)

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

    def _finalize_segment(self, key, st, estimated, item_out) -> None:
        """Unpad, downmix to mono, resample back to 16 kHz and file the result.

        `estimated` is the separated (2, total_length) stereo signal: normalized
        overlap-add output in chunked mode, the raw model output in whole mode.
        """
        idx, sdx = key
        if st['padded']:
            estimated = estimated[:, self._border:-self._border]
        vocals_mono = estimated.mean(axis=0).astype(np.float32)
        if SAMPLE_RATE != self.model_sample_rate:
            vocals_mono = torchaudio.functional.resample(
                torch.from_numpy(vocals_mono), self.model_sample_rate, SAMPLE_RATE
            ).numpy()
        # Pin the result to the input's sample count. The 16k -> 44.1k -> 16k round
        # trip is not length-preserving (ratio 2.75625; each leg rounds independently),
        # which left segments up to ~9 ms short -- enough to trip
        # _validate_segment_duration and to walk the audio out of step with the
        # timestamps it is filed under. The drift is sub-frame, so clamping here is
        # exact, not a fudge.
        src_len = st.get('src_len')
        if src_len is not None and len(vocals_mono) != src_len:
            if len(vocals_mono) > src_len:
                vocals_mono = vocals_mono[:src_len]
            else:
                vocals_mono = np.pad(vocals_mono, (0, src_len - len(vocals_mono)))
        _validate_segment_duration(st['start'], st['end'], vocals_mono)
        rebuilt = {'start': st['start'], 'end': st['end'], 'audio': vocals_mono}
        # Isolation rewrites the audio but must not lose provenance the VAD stage
        # attached (currently 'expanded', which --asr_context_scope reads downstream).
        for key in _PERSISTED_SEGMENT_KEYS:
            if key in st:
                rebuilt[key] = st[key]
        item_out[idx]['segs'][sdx] = rebuilt


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_vocal_isolation(
    model_name: str,
    device: str,
    device_index: int = 0,
    model_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    compute_type: str = "float32",
    vram_checks: bool = True,
    local_files_only: bool = False,
    segment_mode: Optional[str] = None,
    compile: str = "off",
) -> VocalIsolationProcessor:
    """Load a vocal isolation model and return a processor.

    The checkpoint is downloaded from HuggingFace on first use and cached.
    model_dir, if given, overrides the default HuggingFace cache directory.
    batch_size controls how many fixed-size chunks are run through the model at once
    (chunked mode only; whole mode runs one segment per pass).
    segment_mode overrides config.inference.mode ("whole" or "chunked").
    compute_type="float16" halves the model's weight VRAM footprint; inference still
    runs under the existing autocast (see infer_fn) so activations/STFT stay numerically
    safe regardless of the stored weight dtype.
    compile is "on", "off" or "auto" (compile when the run is long enough to repay it);
    see _COMPILE_MIN_CHUNKS.
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
    try:
        ensure_hf_file_downloaded(_HF_REPO_ID, _HF_FILENAME, cache_dir=model_dir, local_files_only=local_files_only)
    except Exception as e:
        logger.warning("Could not download %r: %s — using cached version if available.", _HF_FILENAME, e)
    checkpoint_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_FILENAME,
        cache_dir=model_dir,
        local_files_only=local_files_only,
    )
    torch_model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))
    )

    resolved_device = resolve_device(device, device_index)
    torch_device = torch.device(resolved_device)
    dtype = resolve_torch_compute_dtype(compute_type, resolved_device, "vocal_isolation")
    check_vram_headroom(
        "Vocal isolation model load", torch_device,
        _VOCAL_ISOLATION_VRAM_ESTIMATE_MB, _VOCAL_ISOLATION_REMEDIATION,
        vram_checks=vram_checks,
    )
    torch_model = guard_model_load(
        "vocal isolation", _VOCAL_ISOLATION_REMEDIATION,
        lambda: torch_model.to(torch_device, dtype=dtype),
    )

    return MbRoformerProcessor(
        model=torch_model,
        config=config,
        device=torch_device,
        batch_size=batch_size,
        segment_mode=segment_mode,
        compile=compile,
    )

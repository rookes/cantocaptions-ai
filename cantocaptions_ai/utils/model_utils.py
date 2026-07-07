import gc
import torch
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Generic, List, Optional, Tuple, TypeVar

from cantocaptions_ai.utils.schema import ProgressCallback
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
_ModelT = TypeVar("_ModelT")


def resolve_torch_compute_dtype(compute_type: str, device: str, stage: str) -> torch.dtype:
    """Map a "float32"/"float16" compute_type option to a torch dtype.

    float16 falls back to float32 off CUDA (half precision is unsupported/unreliable
    for CPU ops these models rely on, e.g. FFT in vocal isolation).
    """
    if compute_type == "float16":
        if not device.startswith("cuda"):
            logger.warning(
                "%s compute_type=float16 requires a CUDA device; falling back to float32 on %s.",
                stage, device,
            )
            return torch.float32
        return torch.float16
    return torch.float32


def _load_or_compute(audio_path, load_debug_dir, debug_dir, load_fn, write_fn, compute_fn):
    """Load a stage result from the debug cache, or compute and optionally save it."""
    if load_debug_dir:
        cached = load_fn(audio_path, load_debug_dir)
        if cached is not None:
            return cached
    result = compute_fn()
    if debug_dir:
        write_fn(audio_path, result, debug_dir)
    return result


def partition_by_cache(
    items: List[dict],
    read_debug: Callable[[str, str], Any],
    load_debug_dir: Optional[str],
) -> Tuple[Dict[int, Any], List[Tuple[int, dict]]]:
    """Split items into cached results and items still needing compute.

    Returns ``(cached, to_compute)`` where ``cached`` maps an item's index → its
    loaded stage result, and ``to_compute`` is a list of ``(index, item)`` for the
    items whose cache was absent. Used by batched stages so a partial ``--load_debug_dir``
    (some files cached, some not) is handled without recomputing the cached ones.
    """
    cached: Dict[int, Any] = {}
    to_compute: List[Tuple[int, dict]] = []
    for idx, item in enumerate(items):
        if load_debug_dir:
            result = read_debug(item['audio_path'], load_debug_dir)
            if result is not None:
                cached[idx] = result
                continue
        to_compute.append((idx, item))
    return cached, to_compute


def run_adaptive_batches(
    jobs: List[Any],
    batch_size: Optional[int],
    infer_fn: Callable[[List[Any]], None],
    reporter: ProgressCallback = None,
) -> None:
    """Process a flat list of work units in fixed-size batches with OOM-adaptive halving.

    ``jobs`` is already flattened across all files, so successive batches pack work
    units from different files (cross-file backfill — no half-empty tail batch per
    file). ``infer_fn(batch)`` runs the model and scatters its results (via closures);
    it must not mutate shared state before the model call so an OOM retry is safe.
    On CUDA OOM the batch size is halved and the same batch retried; ``reporter`` is
    advanced by ``len(batch)`` after each successful batch.
    """
    total = len(jobs)
    if total == 0:
        return
    current = batch_size if (batch_size and batch_size >= 1) else total
    oom_warned = False
    i = 0
    while i < total:
        batch = jobs[i:i + current]
        try:
            infer_fn(batch)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current <= 1:
                raise RuntimeError(
                    "CUDA out of memory even at batch_size=1. "
                    "Try freeing VRAM or reducing --chunk_size."
                ) from e
            current = max(1, current // 2)
            if not oom_warned:
                logger.warning(
                    "CUDA out of memory — retrying with batch_size=%d. "
                    "Pass --batch_size %d to avoid this next time.",
                    current, current,
                )
                oom_warned = True
            continue
        i += len(batch)
        if reporter is not None:
            reporter.advance(len(batch))


class PipelineStage(ABC, Generic[InputT, OutputT]):
    """Abstract base for pipeline stages: implement process() to transform InputT → OutputT.

    Subclasses must also implement four static methods that plug into the run() machinery:
    - read_debug / write_debug: load and save stage checkpoints
    - _extract: pull this stage's input out of the pipeline item carrier dict
    - _pack: merge this stage's output back into the carrier dict
    """

    @abstractmethod
    def process(self, input: InputT, *, progress_callback: ProgressCallback = None) -> OutputT:
        ...

    @staticmethod
    @abstractmethod
    def read_debug(audio_path: str, debug_dir: str) -> Any:
        """Load this stage's checkpoint for audio_path from debug_dir, or None on miss."""
        ...

    @staticmethod
    @abstractmethod
    def write_debug(audio_path: str, result: Any, debug_dir: str) -> None:
        """Save this stage's result for audio_path to debug_dir."""
        ...

    @staticmethod
    @abstractmethod
    def _extract(item: dict) -> Any:
        """Extract this stage's input from the pipeline carrier dict.

        Called lazily (inside the compute lambda) — expensive ops like load_audio are safe.
        """
        ...

    @staticmethod
    @abstractmethod
    def _pack(item: dict, result: Any) -> dict:
        """Return a new carrier dict with this stage's result merged in."""
        ...

    @classmethod
    def load_cache(cls, items: List[dict], debug_dir: Optional[str]) -> List[dict]:
        """Load all items from the debug cache without running the model.

        Used when need_*=False (all items cached) and the stage was never instantiated.
        debug_dir must be non-None in practice (guarded by need_* checks in transcribe.py).
        """
        assert debug_dir, "load_cache called with no debug_dir"
        return [cls._pack(item, cls.read_debug(item['audio_path'], debug_dir)) for item in items]

    def run(
        self,
        items: List[dict],
        *,
        debug_dir: Optional[str] = None,
        load_debug_dir: Optional[str] = None,
        progress_callback: ProgressCallback = None,
    ) -> List[dict]:
        """Run this stage over all pipeline items with debug caching.

        For each item, _load_or_compute tries the cache first; _extract and process()
        are only called on a cache miss. Progress is reported per file (one unit per
        item) so the bar spans all files in the stage; stages that batch work units
        across files (ASR, vocal isolation) override run() for finer-grained progress.
        """
        cls = type(self)
        if progress_callback is not None:
            progress_callback.set_total(len(items), unit="file")
        result_items = []
        for item in items:
            audio_path = item['audio_path']
            result = _load_or_compute(
                audio_path, load_debug_dir, debug_dir,
                cls.read_debug, cls.write_debug,
                lambda item=item: self.process(cls._extract(item)),
            )
            result_items.append(cls._pack(item, result))
            if progress_callback is not None:
                progress_callback.advance(1)
        return result_items


@contextmanager
def model_scope(load_fn: Callable[..., _ModelT], *args, **kwargs) -> Generator[_ModelT, None, None]:
    """Load a model, yield it for use, then delete it and free CUDA memory."""
    model = load_fn(*args, **kwargs)
    try:
        yield model
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def flush_vram() -> None:
    """Collect garbage and free CUDA memory. Call after deleting model references."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_device(device) -> bool:
    """True if *device* refers to a CUDA device (None/int default to the current one)."""
    if device is None or isinstance(device, int):
        return True
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def vram_stats(device=None) -> Optional[Dict[str, float]]:
    """Return a snapshot of current VRAM usage, or None when unavailable/not applicable.

    ``free_mb``/``total_mb`` come from ``torch.cuda.mem_get_info()``, which reports
    real device-wide free memory (other processes included) — not
    ``total_memory - memory_reserved()``, which only reflects this process's own
    PyTorch caching-allocator reservation and is blind to VRAM held by other programs.

    Returns None for a non-CUDA *device* (e.g. "cpu", "mps") even when CUDA happens
    to be available elsewhere on the machine — callers pass through whatever device
    string the pipeline is actually configured to use (which may be "cpu" on a
    CUDA-capable box, e.g. via --device cpu), and torch.cuda.mem_get_info()/
    memory_allocated() raise if handed a non-CUDA device string.
    """
    if not torch.cuda.is_available() or not _is_cuda_device(device):
        return None
    idx = device if device is not None else 0
    allocated = torch.cuda.memory_allocated(idx)
    reserved  = torch.cuda.memory_reserved(idx)
    with torch.cuda.device(idx):
        free, total = torch.cuda.mem_get_info()
    return {
        'allocated_mb': allocated / 1e6,
        'reserved_mb':  reserved  / 1e6,
        'free_mb':      free  / 1e6,
        'total_mb':     total / 1e6,
    }


def check_vram_headroom(
    stage: str,
    device,
    estimated_mb: float,
    remediation: str,
    threshold: float = 0.85,
    vram_checks: bool = True,
) -> Optional[Dict[str, float]]:
    """Log estimated VRAM usage for *stage* against real device-wide headroom.

    Always logs the comparison (useful for diagnosing slow/stalled runs even when
    headroom is fine); warns with *remediation* — a stage-specific one-line CLI
    suggestion — when ``estimated_mb`` exceeds ``threshold`` fraction of free VRAM.
    Returns the ``vram_stats()`` snapshot so callers that also want to log it don't
    need to query it a second time.

    ``vram_checks=False`` (``--vram_checks False``) skips the ``vram_stats()`` call
    (and thus its ``torch.cuda.mem_get_info()`` driver round-trip) entirely, for
    zero-overhead runs where turnaround time matters more than OOM safety margins.
    """
    if not vram_checks:
        return None
    stats = vram_stats(device)
    if stats is None:
        return None
    pct = estimated_mb / stats['total_mb'] * 100
    logger.info(
        "%s VRAM estimate: %.0f MB (%.0f%% of %.0f MB total, %.0f MB free)",
        stage, estimated_mb, pct, stats['total_mb'], stats['free_mb'],
    )
    if estimated_mb > stats['free_mb'] * threshold:
        logger.warning(
            "%s may exceed available VRAM headroom (estimated %.0f MB vs %.0f MB free). "
            "Slowdown or an out-of-memory failure is likely. %s",
            stage, estimated_mb, stats['free_mb'], remediation,
        )
    return stats


def guard_model_load(stage: str, remediation: str, load_fn: Callable[[], _ModelT]) -> _ModelT:
    """Run a model-loading callable; on CUDA OOM, re-raise with an actionable message.

    Model loading (``.to(device)``/``from_pretrained`` moving weights to GPU) isn't
    batchable like inference, so unlike run_adaptive_batches this doesn't retry —
    it only turns an opaque CUDA OOM traceback into a clear, stage-specific one.
    """
    try:
        return load_fn()
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA out of memory while loading the {stage} model. {remediation}"
        ) from e


def ensure_hf_model_downloaded(repo_id: str, cache_dir=None, local_files_only: bool = False) -> None:
    """Download a full HF Hub repo snapshot to the local cache if not already present.

    Logs before/after the download so a first-run fetch (which can take a while and
    would otherwise produce no visible signal until it completes) doesn't look like a
    hang. Uses huggingface_hub.snapshot_download, which skips files already cached.
    """
    if local_files_only:
        return
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    probe = try_to_load_from_cache(repo_id, "config.json", cache_dir=cache_dir)
    if probe is not None:
        return

    try:
        import hf_xet  # noqa: F401
        xet_hint = ""
    except ImportError:
        xet_hint = " (tip: pip install hf_xet for faster downloads)"

    logger.info("Downloading %r from HuggingFace Hub%s", repo_id, xet_hint)
    snapshot_download(repo_id, cache_dir=cache_dir)
    logger.info("Download complete: %r", repo_id)


def ensure_hf_file_downloaded(repo_id: str, filename: str, cache_dir=None, local_files_only: bool = False) -> None:
    """Download a single file from an HF Hub repo to the local cache if not present.

    Same before/after logging as ensure_hf_model_downloaded, for repos where only one
    file (e.g. a checkpoint) is needed rather than a full snapshot.
    """
    if local_files_only:
        return
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    probe = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
    if probe is not None:
        return

    logger.info("Downloading %r from HuggingFace Hub", f"{repo_id}/{filename}")
    hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    logger.info("Download complete: %r", f"{repo_id}/{filename}")

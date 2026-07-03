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


def vram_stats(device=None) -> Optional[Dict[str, float]]:
    """Return a snapshot of current VRAM usage, or None when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    idx = device if device is not None else 0
    allocated = torch.cuda.memory_allocated(idx)
    reserved  = torch.cuda.memory_reserved(idx)
    total     = torch.cuda.get_device_properties(idx).total_memory
    return {
        'allocated_mb': allocated / 1e6,
        'reserved_mb':  reserved  / 1e6,
        'free_mb':      (total - reserved) / 1e6,
        'total_mb':     total / 1e6,
    }

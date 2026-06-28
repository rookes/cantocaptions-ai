import gc
import torch
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generic, TypeVar

from cantocaptions_ai.utils.schema import ProgressCallback

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PipelineStage(ABC, Generic[InputT, OutputT]):
    """Abstract base for pipeline stages: implement process() to transform InputT → OutputT."""

    @abstractmethod
    def process(self, input: InputT, *, progress_callback: ProgressCallback = None) -> OutputT:
        ...


@contextmanager
def model_scope(load_fn, *args, **kwargs):
    """Load a model, yield it for use, then delete it and free CUDA memory.

    If load_fn is None, yields None (useful for conditional loading where the
    pipeline may skip a stage due to cached debug output).
    """
    model = load_fn(*args, **kwargs) if load_fn is not None else None
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

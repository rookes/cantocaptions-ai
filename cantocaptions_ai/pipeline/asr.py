"""The ASR stage, and the registry of ASR backends that implement it.

Every backend turns VAD segments into one text per segment; what differs is the model
family behind it. Which backend runs is a property of the *model* (``ModelProfile.backend``,
else the checkpoint's own ``model_type``), not of which libraries happen to be installed:

    qwen3-asr  Qwen3-ASR through transformers' native qwen3_asr support (_asr_native.py)
    whisper    Whisper encoder-decoder models (_asr_whisper.py)
    ctc        wav2vec2-family CTC models: greedy decode of the emissions (_asr_ctc.py)

``AsrStage`` holds the transcription checkpoint plumbing every backend shares, and
``BatchedAsrStage`` the batching all three use: segments from every file are pooled,
run longest-first through ``BatchExecutor`` (OOM halves the batch), scattered back, and
normalized per the language pack. A backend supplies only ``_infer_batch``.
"""
import importlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from cantocaptions_ai.utils.schema import (
    ProgressCallback,
    SingleSegment,
    TranscriptionResult,
    VadAudioSegment,
)
from cantocaptions_ai.utils.model_utils import (
    BatchExecutor,
    MemoryPolicy,
    PipelineStage,
    partition_by_cache,
    write_checkpoint,
)
from cantocaptions_ai.utils.debug import load_transcription_debug, write_transcription_debug
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.output import LANGUAGES
from cantocaptions_ai.text_profiles import DEFAULT_NORMALIZATION, TextNormalization

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _has_native_qwen3asr() -> bool:
    """True when transformers ships native qwen3_asr support (official as of transformers>=5.13.0)."""
    import importlib.util
    return importlib.util.find_spec("transformers.models.qwen3_asr") is not None


def _resolve_normalization(model_name: Optional[str], language: Optional[str], normalization):
    """The post-ASR normalization to apply: the caller's, else the language pack's
    conventions for this model (e.g. OpenCC for stock Qwen3-ASR writing Cantonese)."""
    if normalization is not None:
        return normalization
    from cantocaptions_ai.languages import get_language_pack
    return get_language_pack(language).resolve(model_name).normalization


def _normalize_language(language: str) -> str:
    """Convert an ISO code or bare name to the canonical Qwen3-ASR form (e.g. 'yue' → 'Cantonese')."""
    if not language:
        raise ValueError("an ASR language is required (e.g. 'yue'); there is no auto-detection")
    longname = LANGUAGES.get(language, language)
    return longname[:1].upper() + longname[1:].lower()


def _torch_device(device: Union[int, str, "torch.device", None]) -> "torch.device":
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int) and device >= 0:
        return torch.device(f"cuda:{device}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage base classes
# ---------------------------------------------------------------------------

class AsrStage(PipelineStage["List[VadAudioSegment]", "TranscriptionResult"]):
    """Base for every ASR backend: the transcription checkpoint and the item carrier.

    Provides the debug-caching static methods required by PipelineStage (and used by
    ``AsrStage.load_cache`` in transcribe.py). Subclasses implement process().
    """

    debug_stage = "transcription"

    @staticmethod
    def read_debug(audio_path, debug_dir): return load_transcription_debug(audio_path, debug_dir)

    @staticmethod
    def write_debug(audio_path, result, debug_dir): write_transcription_debug(audio_path, result, debug_dir)

    @staticmethod
    def _extract(item): return item['vad_segments']

    @staticmethod
    def _pack(item, result):
        return {**item, 'result': result}

    @abstractmethod
    def process(
        self,
        input: "List[VadAudioSegment]",
        *,
        progress_callback: ProgressCallback = None,
    ) -> "TranscriptionResult":
        ...


# The name the stage had while Qwen3-ASR was the only family; kept for callers.
QwenPipeline = AsrStage


class BatchedAsrStage(AsrStage):
    """An ASR stage that transcribes batches of segments, pooled across files.

    Subclasses implement :meth:`_infer_batch` (one batch of audio arrays -> one text each)
    and may override :meth:`_backend_language` (the language label the model wants, which
    is also what the result records).
    """

    #: Shown in the "Performing transcription (... backend)" log line.
    backend_label = "batched"

    def __init__(
        self,
        device: Union[int, str, "torch.device", None] = None,
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
        print_progress: bool = False,
        verbose: bool = False,
        vram_checks: bool = True,
        normalization: TextNormalization = DEFAULT_NORMALIZATION,
    ):
        self.normalization = normalization
        self.device = _torch_device(device)
        self.preset_language = language
        self._batch_size = batch_size
        self.print_progress = print_progress
        self.verbose = verbose
        self.vram_checks = vram_checks
        self.policy = MemoryPolicy(vram_checks)

    def _backend_language(self, language: Optional[str]) -> str:
        """The language as this backend's model names it (default: the ISO code)."""
        if not language:
            raise ValueError("an ASR language is required (e.g. 'yue'); there is no auto-detection")
        return language

    @abstractmethod
    def _infer_batch(
        self, wavs: List, language: str, contexts: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """Transcribe one batch of 16 kHz audio arrays; one text per array, in order.

        Must raise RuntimeError on CUDA OOM (BatchExecutor retries at a smaller batch
        size) and mutate no shared state before the model call.
        """

    def _segments(self, segs, texts) -> List[SingleSegment]:
        from cantocaptions_ai.languages.yue.text import normalize_segment_text
        return [
            normalize_segment_text({'text': text or '', 'start': seg['start'], 'end': seg['end']}, self.normalization)
            for seg, text in zip(segs, texts)
        ]

    def run(self, items, *, debug_dir=None, load_debug_dir=None, progress_callback: ProgressCallback = None):
        """Transcribe all files, batching VAD segments across file boundaries.

        Segments from every to-compute file are flattened into one job stream, so
        batches pack work from different files (no half-empty tail batch per file).
        """
        logger.info("Performing transcription (%s backend)...", self.backend_label)
        language = self._backend_language(self.preset_language)
        cached, to_compute = partition_by_cache(items, self, load_debug_dir)

        # jobs are (item_idx, seg_idx); texts scattered back into per-item buffers.
        jobs: List = []
        buffers = {}  # idx -> {'segs': List[VadAudioSegment], 'texts': List[Optional[str]], 'item': dict}
        for idx, item in to_compute:
            segs = item['vad_segments']
            buffers[idx] = {'segs': segs, 'texts': [None] * len(segs), 'item': item}
            jobs.extend((idx, sdx) for sdx in range(len(segs)))

        if progress_callback is not None:
            progress_callback.set_total(len(jobs), unit="seg")

        def infer_fn(batch):
            wavs = [buffers[idx]['segs'][sdx]['audio'] for idx, sdx in batch]
            contexts = [buffers[idx]['segs'][sdx].get('context') for idx, sdx in batch]
            texts = self._infer_batch(wavs, language, contexts)
            for (idx, sdx), text in zip(batch, texts):
                buffers[idx]['texts'][sdx] = text

        # Longest-first: front-loads the largest allocation (the KV cache, for a generative
        # model) so the allocator's reserved pool is claimed once up front rather than
        # ratcheting up over the run (bench_asr_native.py --sort desc confirmed the win).
        # Texts scatter back by index, so output order is unaffected.
        BatchExecutor(
            self._batch_size,
            order_key=lambda job: len(buffers[job[0]]['segs'][job[1]]['audio']),
        ).run(jobs, infer_fn, reporter=progress_callback)

        computed = {}
        for idx, buf in buffers.items():
            result: TranscriptionResult = {
                "segments": self._segments(buf['segs'], buf['texts']), "language": language,
            }
            computed[idx] = result
            write_checkpoint(self, buf['item'], result, debug_dir)

        result_items = []
        for idx, item in enumerate(items):
            result = cached[idx] if idx in cached else computed[idx]
            result_items.append(self._pack(item, result))
        return result_items

    def process(
        self,
        input: List[VadAudioSegment],
        *,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Transcribe a single file's segments (library/single-file entry point)."""
        language = self._backend_language(self.preset_language)
        texts: List[Optional[str]] = [None] * len(input)
        jobs = list(range(len(input)))

        if progress_callback is not None:
            progress_callback.set_total(len(jobs), unit="seg")

        def infer_fn(batch):
            wavs = [input[i]['audio'] for i in batch]
            contexts = [input[i].get('context') for i in batch]
            for i, text in zip(batch, self._infer_batch(wavs, language, contexts)):
                texts[i] = text

        BatchExecutor(
            self._batch_size,
            order_key=lambda i: len(input[i]['audio']),
        ).run(jobs, infer_fn, reporter=progress_callback)

        return {"segments": self._segments(input, texts), "language": language}


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AsrBackend:
    """One ASR model family: how to load it, and what it supports."""
    name: str
    loader: str                 # "module:function", imported only when the backend is used
    supports_context: bool = False  # --asr_context (reference-subtitle biasing)

    def load(self, **kwargs) -> AsrStage:
        module, _, func = self.loader.partition(":")
        return getattr(importlib.import_module(module), func)(**kwargs)


ASR_BACKENDS: Dict[str, AsrBackend] = {
    backend.name: backend for backend in (
        AsrBackend("qwen3-asr", "cantocaptions_ai.pipeline._asr_native:load_model_native",
                   supports_context=True),
        AsrBackend("whisper", "cantocaptions_ai.pipeline._asr_whisper:load_model_whisper"),
        AsrBackend("ctc", "cantocaptions_ai.pipeline._asr_ctc:load_model_ctc"),
    )
}

# A checkpoint's config.json model_type -> backend, for models with no registered profile.
_MODEL_TYPE_BACKENDS = {
    "qwen3_asr": "qwen3-asr",
    "whisper": "whisper",
    **{model_type: "ctc" for model_type in (
        "wav2vec2", "wav2vec2-bert", "wav2vec2-conformer", "hubert", "wavlm",
        "data2vec-audio", "sew", "sew-d", "unispeech", "unispeech-sat",
    )},
}


def backend_for(model_name: Optional[str], language: Optional[str] = None, *,
                cache_dir: Optional[str] = None, local_files_only: bool = False) -> AsrBackend:
    """The backend that runs *model_name* (None: the language's default model).

    A registered model names its backend; any other is identified from its checkpoint's
    ``config.json`` (downloaded if needed, which is all that is read).
    """
    from cantocaptions_ai.pipeline.model_profiles import get_model_profile

    profile = get_model_profile(model_name, language)
    if profile.backend is not None:
        return ASR_BACKENDS[profile.backend]

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(profile.hf_id, cache_dir=cache_dir,
                                        local_files_only=local_files_only)
    name = _MODEL_TYPE_BACKENDS.get(config.model_type)
    if name is None:
        raise ValueError(
            f"ASR model {profile.hf_id!r} is a {config.model_type!r} model, which no ASR backend "
            f"runs (supported: {', '.join(sorted(set(_MODEL_TYPE_BACKENDS)))})"
        )
    return ASR_BACKENDS[name]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(
    model_name: Optional[str],
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    attn_implementation: str = "sdpa",
    language: Optional[str] = "yue",
    model=None,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
    use_auth_token: Optional[Union[str, bool]] = None,
    batch_size: Optional[int] = None,
    compile_enabled: bool = False,
    print_progress: bool = False,
    verbose: bool = False,
    vram_checks: bool = True,
    vram_headroom_mb: int = 512,
    processor=None,
    normalization=None,
) -> AsrStage:
    """Load the ASR model and the backend that runs it (see :func:`backend_for`).

    ``model_name`` None loads the language pack's default model. ``normalization`` None
    takes the pack's conventions for the model (see languages/base.py). ``model`` and
    ``processor`` may be passed pre-built (e.g. a merged LoRA checkpoint).

    Qwen3-ASR needs transformers' native qwen3_asr support (transformers>=5.13.0, the
    ``transformers_qwen`` extra); torch.compile is opt-in for it (compile_enabled /
    --compile), a net loss by default -- see _asr_native._compile_and_warmup.
    """
    backend = backend_for(model_name, language, cache_dir=download_root,
                          local_files_only=local_files_only)
    if backend.name == "qwen3-asr" and not _has_native_qwen3asr():
        raise ImportError(
            "Qwen3-ASR needs transformers' native qwen3_asr support (transformers>=5.13.0): "
            "run `uv sync --extra transformers_qwen`"
        )
    logger.info("ASR backend: %s", backend.name)
    kwargs = dict(
        model_name=model_name, device=device, device_index=device_index,
        compute_type=compute_type, attn_implementation=attn_implementation,
        language=language, model=model, download_root=download_root,
        local_files_only=local_files_only, batch_size=batch_size,
        print_progress=print_progress, verbose=verbose, vram_checks=vram_checks,
        vram_headroom_mb=vram_headroom_mb, processor=processor, normalization=normalization,
    )
    if backend.name == "qwen3-asr":
        kwargs["compile_enabled"] = compile_enabled
    return backend.load(**kwargs)

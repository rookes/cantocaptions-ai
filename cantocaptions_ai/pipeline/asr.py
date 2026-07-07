from abc import abstractmethod
from typing import List, Optional, Union

from cantocaptions_ai.utils.schema import TranscriptionResult, VadAudioSegment, ProgressCallback
from cantocaptions_ai.utils.model_utils import PipelineStage
from cantocaptions_ai.utils.debug import load_transcription_debug, write_transcription_debug
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.output import LANGUAGES

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _has_native_qwen3asr() -> bool:
    """True when transformers ships native qwen3_asr support (official as of transformers>=5.13.0)."""
    import importlib.util
    return importlib.util.find_spec("transformers.models.qwen3_asr") is not None


# ---------------------------------------------------------------------------
# Shared utilities used by both backends
# ---------------------------------------------------------------------------

def _normalize_language(language: str) -> str:
    """Convert an ISO code or bare name to the canonical Qwen3-ASR form (e.g. 'yue' → 'Cantonese')."""
    longname = LANGUAGES.get(language, language)
    return longname[:1].upper() + longname[1:].lower()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QwenPipeline(PipelineStage["List[VadAudioSegment]", "TranscriptionResult"]):
    """Base class for Qwen3-ASR pipeline backends.

    Provides the debug-caching static methods required by PipelineStage (and used
    by QwenPipeline.load_cache in transcribe.py).  Subclasses implement process().

    Concrete subclasses:
      QwenPipelineLegacy (_asr_legacy.py) — qwen_asr package, transformers==4.57.6 (`legacy` extra)
      QwenPipelineNative (_asr_native.py) — official transformers qwen3_asr support, -hf model (`transformers_qwen` extra)
    """

    @staticmethod
    def read_debug(audio_path, debug_dir): return load_transcription_debug(audio_path, debug_dir)

    @staticmethod
    def write_debug(audio_path, result, debug_dir): write_transcription_debug(audio_path, result, debug_dir)

    @staticmethod
    def _extract(item): return item['vad_segments']

    @staticmethod
    def _pack(item, result):
        return {'audio_path': item['audio_path'], 'result': result, 'vad_segments': item['vad_segments']}

    @abstractmethod
    def process(
        self,
        input: "List[VadAudioSegment]",
        *,
        progress_callback: ProgressCallback = None,
    ) -> "TranscriptionResult":
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    attn_implementation: str = "sdpa",
    asr_options: Optional[dict] = None,
    language: Optional[str] = "yue",
    vocal_isolation_method: Optional[str] = None,
    model=None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
    use_auth_token: Optional[Union[str, bool]] = None,
    batch_size: Optional[int] = None,
    compile_enabled: bool = False,
    print_progress: bool = False,
    verbose: bool = False,
    vram_checks: bool = True,
) -> QwenPipeline:
    """Load a Qwen3-ASR model, auto-selecting the backend based on the installed transformers.

    With transformers>=5.13.0 (uv sync --extra transformers_qwen, recommended):
      → QwenPipelineNative using Qwen/Qwen3-ASR-1.7B-hf. torch.compile is opt-in
        (compile_enabled=True / --compile) — benchmarked (scripts/bench_asr_compile.py)
        to be a net loss by default for this pipeline's essentially-unique VAD segment
        lengths: see _asr_native.py's _compile_and_warmup docstring for the full findings.

    With transformers==4.57.6 (uv sync --extra legacy):
      → QwenPipelineLegacy using Qwen/Qwen3-ASR-1.7B via the qwen_asr package.

    `transformers_qwen` and `legacy` are mutually exclusive installs (conflicting
    transformers pins) — see pyproject.toml.
    """
    if _has_native_qwen3asr():
        logger.info("Native qwen3_asr support detected — using native backend")
        from cantocaptions_ai.pipeline._asr_native import load_model_native
        return load_model_native(
            model_name=model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            attn_implementation=attn_implementation,
            language=language,
            model=model,
            download_root=download_root,
            local_files_only=local_files_only,
            batch_size=batch_size,
            compile_enabled=compile_enabled,
            print_progress=print_progress,
            verbose=verbose,
            vram_checks=vram_checks,
        )
    else:
        logger.info(
            "Using qwen_asr legacy backend — "
            "run `uv sync --extra transformers_qwen` for the recommended native backend"
        )
        from cantocaptions_ai.pipeline._asr_legacy import load_model_legacy
        return load_model_legacy(
            model_name=model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            attn_implementation=attn_implementation,
            language=language,
            download_root=download_root,
            local_files_only=local_files_only,
            batch_size=batch_size,
            print_progress=print_progress,
            verbose=verbose,
        )

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
    """True when git-main transformers with native qwen3_asr support is installed."""
    import importlib.util
    return importlib.util.find_spec("transformers.models.qwen3_asr") is not None


# ---------------------------------------------------------------------------
# Shared utilities used by both backends
# ---------------------------------------------------------------------------

def _normalize_language(language: str) -> str:
    """Convert an ISO code or bare name to the canonical Qwen3-ASR form (e.g. 'yue' → 'Cantonese')."""
    longname = LANGUAGES.get(language, language)
    return longname[:1].upper() + longname[1:].lower()


def _build_text_prompt(processor, language: str) -> str:
    """Build the chat-template prompt that forces text-only output in the given language."""
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return base + f"language {language}<asr_text>"


def _ensure_model_downloaded(model_id: str, cache_dir=None) -> None:
    """Download model weights to the local cache if not already present.

    Uses huggingface_hub.snapshot_download, which shows per-file progress bars and
    skips files that are already cached.  hf_xet accelerates downloads for repos
    hosted on Xet storage — install it with `pip install hf_xet` for a speed boost.
    """
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    probe = try_to_load_from_cache(model_id, "config.json", cache_dir=cache_dir)
    if probe is not None:
        return

    try:
        import hf_xet  # noqa: F401
        xet_hint = ""
    except ImportError:
        xet_hint = " (tip: pip install hf_xet for faster downloads)"

    logger.info("Downloading %r from HuggingFace Hub%s", model_id, xet_hint)
    snapshot_download(model_id, cache_dir=cache_dir)
    logger.info("Download complete: %r", model_id)


def _detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Remove hallucinated character/pattern repetitions from ASR output."""
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            res.append(s[i] if count > thresh else s[i:i + count])
            i += count
        return "".join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        if n < thresh * 2:
            return s
        i = 0
        result = []
        found = False
        while i <= n - thresh * 2:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                if all(s[i + r * k:i + r * k + k] == pattern for r in range(1, thresh)):
                    end = i + thresh * k
                    while end + k <= n and s[end:end + k] == pattern:
                        end += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end:], thresh, max_len))
                    i = n
                    found = True
                    break
            if not found:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return "".join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


_ASR_TEXT_TAG = "<asr_text>"


def _parse_asr_output(raw: str, user_language: Optional[str] = None):
    """Parse Qwen3-ASR raw output into (language, text).

    When user_language is supplied the model output is treated as plain text.
    """
    if not raw:
        return "", ""
    s = _detect_and_fix_repetitions(str(raw).strip())
    if not s:
        return "", ""
    if user_language:
        return user_language, s
    if _ASR_TEXT_TAG in s:
        _, text = s.split(_ASR_TEXT_TAG, 1)
        return "", text.strip()
    return "", s


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QwenPipeline(PipelineStage["List[VadAudioSegment]", "TranscriptionResult"]):
    """Base class for Qwen3-ASR pipeline backends.

    Provides the debug-caching static methods required by PipelineStage (and used
    by QwenPipeline.load_cache in transcribe.py).  Subclasses implement process().

    Concrete subclasses:
      QwenPipelineLegacy (_asr_legacy.py) — qwen_asr package, transformers 4.57.6
      QwenPipelineNative (_asr_native.py) — native transformers git-main, -hf model
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
    print_progress: bool = False,
    verbose: bool = False,
) -> QwenPipeline:
    """Load a Qwen3-ASR model, auto-selecting the backend based on the installed transformers.

    With git-main transformers (uv sync --extra compile):
      → QwenPipelineNative using Qwen/Qwen3-ASR-1.7B-hf; torch.compile applied when triton available.

    With transformers 4.57.6 (uv sync, default):
      → QwenPipelineLegacy using Qwen/Qwen3-ASR-1.7B via the qwen_asr package.
    """
    if _has_native_qwen3asr():
        logger.info("Native qwen3_asr support detected — using HF-native backend")
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
            print_progress=print_progress,
            verbose=verbose,
        )
    else:
        logger.info(
            "Using qwen_asr legacy backend — "
            "run `uv sync --extra compile` for HF-native backend with torch.compile"
        )
        from cantocaptions_ai.pipeline._asr_legacy import load_model_legacy
        return load_model_legacy(
            model_name=model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            language=language,
            download_root=download_root,
            local_files_only=local_files_only,
            batch_size=batch_size,
            print_progress=print_progress,
            verbose=verbose,
        )

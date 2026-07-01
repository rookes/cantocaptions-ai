"""Legacy ASR backend: uses qwen_asr.Qwen3ASRModel with transformers 4.57.6.

Loaded lazily by asr.load_model() when native qwen3_asr support is not detected
in the installed transformers.  Do NOT import this module at the top of any file
that may run under git-main transformers — the qwen_asr package uses a decorator
calling convention that changed between 4.57.6 and git-main.
"""
from typing import List, Optional

from cantocaptions_ai.pipeline.asr import QwenPipeline, _normalize_language, _ensure_model_downloaded
from cantocaptions_ai.utils.audio import resolve_device
from cantocaptions_ai.utils.schema import SingleSegment, TranscriptionResult, VadAudioSegment, ProgressCallback
from cantocaptions_ai.cantonese.text import normalize_segment_text
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_MODEL_IDS = {
    "Qwen3-ASR":      "Qwen/Qwen3-ASR-1.7B",
    "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
}


class QwenPipelineLegacy(QwenPipeline):
    """Legacy backend: wraps qwen_asr.Qwen3ASRModel.

    Qwen3ASRModel handles batching internally via max_inference_batch_size.
    process() passes all VAD segments in a single call.
    """

    def __init__(self, model, language: str = "yue"):
        self._model = model
        self._language = _normalize_language(language or "yue")

    def process(
        self,
        input: List[VadAudioSegment],
        *,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        logger.info("Performing transcription (legacy backend)...")
        language = self._language

        # Qwen3ASRModel.transcribe accepts List[(np.ndarray, sample_rate)].
        # VAD segments are already 16 kHz mono float32 arrays.
        audio_inputs = [(seg['audio'], 16000) for seg in input]
        transcriptions = self._model.transcribe(audio_inputs, language=language)

        segments: List[SingleSegment] = []
        for vad_seg, t in zip(input, transcriptions):
            segments.append(normalize_segment_text({
                'text': t.text,
                'start': vad_seg['start'],
                'end': vad_seg['end'],
            }))

        if progress_callback is not None:
            progress_callback(1.0)

        return {"segments": segments, "language": language}


def load_model_legacy(
    model_name: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    language: Optional[str] = "yue",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    batch_size: Optional[int] = None,
    print_progress: bool = False,
    verbose: bool = False,
) -> QwenPipelineLegacy:
    from qwen_asr import Qwen3ASRModel

    model_id = _MODEL_IDS.get(model_name, model_name)

    if not local_files_only:
        try:
            _ensure_model_downloaded(model_id, cache_dir=download_root)
        except Exception as e:
            logger.warning("Could not download %r: %s — using cached version if available.", model_id, e)

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.info("Compute type defaulting to %s for device %s", compute_type, device)

    logger.info("Loading ASR model %r (legacy backend)", model_id)

    hf_model = Qwen3ASRModel.from_pretrained(
        model_id,
        max_inference_batch_size=batch_size or 24,
        max_new_tokens=200,
        torch_dtype=compute_type,
        device_map=resolve_device(device, device_index),
        local_files_only=local_files_only,
        cache_dir=download_root,
    )

    return QwenPipelineLegacy(model=hf_model, language=language)

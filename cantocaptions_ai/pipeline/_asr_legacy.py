"""Legacy ASR backend: uses qwen_asr.Qwen3ASRModel with transformers==4.57.6.

Loaded lazily by asr.load_model() when native qwen3_asr support is not detected
in the installed transformers. Installed via the `legacy` extra, which is
mutually exclusive with `transformers_qwen` (see pyproject.toml) — the two pin
incompatible transformers versions. Do NOT import this module at the top of any
file that may run under a newer transformers — the qwen_asr package's decorator
calling convention is specific to 4.57.6.
"""
from typing import List, Optional

from cantocaptions_ai.pipeline.asr import QwenPipeline, _normalize_language
from cantocaptions_ai.utils.audio import resolve_device
from cantocaptions_ai.utils.schema import SingleSegment, TranscriptionResult, VadAudioSegment, ProgressCallback
from cantocaptions_ai.utils.model_utils import partition_by_cache, ensure_hf_model_downloaded
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

    def run(self, items, *, debug_dir=None, load_debug_dir=None, progress_callback: ProgressCallback = None):
        """Transcribe all files in a single Qwen3ASRModel.transcribe call.

        All to-compute files' segments are concatenated into one list; the model
        backfills them across its internal max_inference_batch_size (so the last
        batch of one file is packed with the next file's segments). Results are then
        split back per file. Progress is coarse (the model call is opaque): the bar
        jumps to full on completion.
        """
        logger.info("Performing transcription (legacy backend)...")
        language = self._language
        cached, to_compute = partition_by_cache(items, self.read_debug, load_debug_dir)

        total_segs = sum(len(item['vad_segments']) for _, item in to_compute)
        if progress_callback is not None:
            progress_callback.set_total(total_segs, unit="seg")

        computed = {}
        if to_compute:
            # Qwen3ASRModel.transcribe accepts List[(np.ndarray, sample_rate)];
            # VAD segments are already 16 kHz mono float32 arrays.
            audio_inputs = [
                (seg['audio'], 16000) for _, item in to_compute for seg in item['vad_segments']
            ]
            transcriptions = self._model.transcribe(audio_inputs, language=language)

            pos = 0
            for idx, item in to_compute:
                segs = item['vad_segments']
                chunk = transcriptions[pos:pos + len(segs)]
                pos += len(segs)
                segments: List[SingleSegment] = [
                    normalize_segment_text({'text': t.text, 'start': s['start'], 'end': s['end']})
                    for s, t in zip(segs, chunk)
                ]
                result: TranscriptionResult = {"segments": segments, "language": language}
                computed[idx] = result
                if debug_dir is not None:
                    self.write_debug(item['audio_path'], result, debug_dir)

            if progress_callback is not None:
                progress_callback.advance(total_segs)

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
            progress_callback.set_total(len(input), unit="seg")
            progress_callback.advance(len(input))

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
    attn_implementation: Optional[str] = "sdpa",
    print_progress: bool = False,
    verbose: bool = False,
) -> QwenPipelineLegacy:
    from qwen_asr import Qwen3ASRModel

    model_id = _MODEL_IDS.get(model_name, model_name)

    try:
        ensure_hf_model_downloaded(model_id, cache_dir=download_root, local_files_only=local_files_only)
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
        attn_implementation=attn_implementation
    )

    return QwenPipelineLegacy(model=hf_model, language=language)

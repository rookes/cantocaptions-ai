"""HF-native ASR backend: uses AutoModelForSpeechSeq2Seq with git-main transformers.

Loaded lazily by asr.load_model() when native qwen3_asr support is detected.
Requires `uv sync --extra compile` (installs git-main transformers + triton).
"""
from typing import List, Optional, Union

import torch

from cantocaptions_ai.pipeline.asr import (
    QwenPipeline,
    _normalize_language,
    _build_text_prompt,
    _ensure_model_downloaded,
    _parse_asr_output,
)
from cantocaptions_ai.utils.audio import resolve_device
from cantocaptions_ai.utils.schema import SingleSegment, TranscriptionResult, VadAudioSegment, ProgressCallback
from cantocaptions_ai.utils.model_utils import partition_by_cache, run_adaptive_batches
from cantocaptions_ai.cantonese.text import normalize_segment_text
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_MODEL_IDS = {
    "Qwen3-ASR":      "Qwen/Qwen3-ASR-1.7B-hf",
    "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B-hf",
}


def _apply_compile(model) -> None:
    # torch.compile is not viable for Qwen3-ASR-hf as implemented in transformers:
    #
    # model.model.language_model (Qwen3 decoder): RoPE inv_freq computation
    #   (inv_freq = 1 / theta ** ...) triggers pow_by_natural with a [-1,-1]
    #   exponent range in the symbolic shape system, crashing the trace.
    #
    # model.model.audio_tower: get_audio_cu_seqlens() uses max().item() to extract
    #   a Python int, iterates over a tensor in a Python for-loop, and builds a
    #   Python list from tensor arithmetic.  These are structural graph breaks that
    #   cause dynamo to retrace on every call, making inference 30x slower.
    #
    # The compile extra's value is git-main transformers (native qwen3_asr backend),
    # not torch.compile.  Triton is kept in the extra for future use.
    pass



def _warn_vram(inputs, batch_size: int, model, max_new_tokens: int, device) -> None:
    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory
    free_vram = total_vram - torch.cuda.memory_allocated(device)
    dtype_bytes = next(model.parameters()).element_size()
    seq_len = inputs["input_ids"].shape[1]
    input_bytes = sum(t.numel() * t.element_size() for t in inputs.values() if isinstance(t, torch.Tensor))
    try:
        text_cfg = model.config.thinker_config.text_config
        kv_bytes = (
            batch_size * (seq_len + max_new_tokens)
            * text_cfg.num_hidden_layers * 2
            * text_cfg.num_key_value_heads * text_cfg.head_dim
            * dtype_bytes
        )
    except AttributeError:
        kv_bytes = 0
    estimated = input_bytes + kv_bytes
    pct = estimated / total_vram * 100
    logger.info(
        "VRAM estimate — batch_size=%d, seq_len=%d: inputs=%.0f MB, kv_cache=%.0f MB, "
        "total_estimated=%.0f MB (%.0f%% of %.0f MB), free=%.0f MB",
        batch_size, seq_len,
        input_bytes / 1e6, kv_bytes / 1e6,
        estimated / 1e6, pct, total_vram / 1e6,
        free_vram / 1e6,
    )
    if estimated > free_vram * 0.85:
        logger.warning(
            "Estimated VRAM for this batch (%.0f MB) may exceed available headroom (%.0f MB free). "
            "Slowdown or failure is likely — consider reducing --batch_size.",
            estimated / 1e6, free_vram / 1e6,
        )


class QwenPipelineNative(QwenPipeline):
    """HF-native backend: uses AutoModelForSpeechSeq2Seq (Qwen3ASRForConditionalGeneration).

    Loads Qwen/Qwen3-ASR-1.7B-hf or -0.6B-hf from git-main transformers.
    """

    def __init__(
        self,
        model,
        processor,
        device: Union[int, str, "torch.device"],
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_new_tokens: int = 256,
        print_progress: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.processor = processor
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, int) and device >= 0:
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")
        self.preset_language = language
        self._batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.print_progress = print_progress
        self.verbose = verbose

    def run(self, items, *, debug_dir=None, load_debug_dir=None, progress_callback: ProgressCallback = None):
        """Transcribe all files, batching VAD segments across file boundaries.

        Segments from every to-compute file are flattened into one job stream, so
        batches pack work from different files (no half-empty tail batch per file).
        """
        logger.info("Performing transcription (HF-native backend)...")
        language = _normalize_language(self.preset_language or "yue")
        cached, to_compute = partition_by_cache(items, self.read_debug, load_debug_dir)

        # jobs are (item_idx, seg_idx); texts scattered back into per-item buffers.
        jobs: List = []
        buffers = {}  # idx -> {'segs': List[VadAudioSegment], 'texts': List[Optional[str]], 'audio_path': str}
        for idx, item in to_compute:
            segs = item['vad_segments']
            buffers[idx] = {'segs': segs, 'texts': [None] * len(segs), 'audio_path': item['audio_path']}
            jobs.extend((idx, sdx) for sdx in range(len(segs)))

        if progress_callback is not None:
            progress_callback.set_total(len(jobs), unit="seg")

        def infer_fn(batch):
            wavs = [buffers[idx]['segs'][sdx]['audio'] for idx, sdx in batch]
            texts = self._infer_batch(wavs, language)
            for (idx, sdx), text in zip(batch, texts):
                buffers[idx]['texts'][sdx] = text

        run_adaptive_batches(jobs, self._batch_size, infer_fn, reporter=progress_callback)

        computed = {}
        for idx, buf in buffers.items():
            segments: List[SingleSegment] = [
                normalize_segment_text({'text': text or '', 'start': seg['start'], 'end': seg['end']})
                for seg, text in zip(buf['segs'], buf['texts'])
            ]
            result: TranscriptionResult = {"segments": segments, "language": language}
            computed[idx] = result
            if debug_dir is not None:
                self.write_debug(buf['audio_path'], result, debug_dir)

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
        language = _normalize_language(self.preset_language or "yue")
        texts: List[Optional[str]] = [None] * len(input)
        jobs = list(range(len(input)))

        if progress_callback is not None:
            progress_callback.set_total(len(jobs), unit="seg")

        def infer_fn(batch):
            wavs = [input[i]['audio'] for i in batch]
            for i, text in zip(batch, self._infer_batch(wavs, language)):
                texts[i] = text

        run_adaptive_batches(jobs, self._batch_size, infer_fn, reporter=progress_callback)

        segments: List[SingleSegment] = [
            normalize_segment_text({'text': texts[i] or '', 'start': input[i]['start'], 'end': input[i]['end']})
            for i in range(len(input))
        ]
        return {"segments": segments, "language": language}

    def _infer_batch(self, wavs: List, language: str) -> List[str]:
        """Run one batch of audio arrays through the model, returning parsed texts.

        Raises RuntimeError on CUDA OOM (caught and retried at a smaller batch size by
        run_adaptive_batches); no shared state is mutated before the model call.
        """
        prompts = [_build_text_prompt(self.processor, language) for _ in wavs]
        inputs = self.processor(text=prompts, audio=wavs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        if self.device.type == "cuda":
            _warn_vram(inputs, len(wavs), self.model, self.max_new_tokens, self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        # generate() returns a plain tensor by default; handle both for safety
        sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
        n_input = inputs["input_ids"].shape[1]
        logger.debug("[TOKENS] batch=%d, new_tokens=%d, max=%d", len(wavs), sequences.shape[1] - n_input, self.max_new_tokens)

        decoded = self.processor.batch_decode(
            sequences[:, n_input:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        del inputs, generated

        texts: List[str] = []
        for raw in decoded:
            logger.debug("[RAW] %s", raw)
            _, text = _parse_asr_output(raw, user_language=language)
            texts.append(text)
        return texts


def load_model_native(
    model_name: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    attn_implementation: str = "sdpa",
    language: Optional[str] = "yue",
    model=None,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    batch_size: Optional[int] = None,
    print_progress: bool = False,
    verbose: bool = False,
) -> QwenPipelineNative:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_id = _MODEL_IDS.get(model_name, model_name)

    if not local_files_only:
        try:
            _ensure_model_downloaded(model_id, cache_dir=download_root)
        except Exception as e:
            logger.warning("Could not download %r: %s — using cached version if available.", model_id, e)

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.info("Compute type defaulting to %s for device %s", compute_type, device)

    device_map = resolve_device(device, device_index)
    pipeline_device = device_index if device == "cuda" else device

    logger.info("Loading ASR model %r (HF-native backend, attn=%s)", model_id, attn_implementation)

    hf_model = model
    if hf_model is None:
        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=compute_type,
            device_map=device_map,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            cache_dir=download_root,
        ).eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        cache_dir=download_root,
    )

    return QwenPipelineNative(
        model=hf_model,
        processor=processor,
        device=pipeline_device,
        language=language,
        batch_size=batch_size,
        max_new_tokens=256,
        print_progress=print_progress,
        verbose=verbose,
    )

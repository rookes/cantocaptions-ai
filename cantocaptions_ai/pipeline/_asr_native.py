"""Native ASR backend: uses AutoModelForMultimodalLM, transformers' official Qwen3-ASR support.

Loaded lazily by asr.load_model() when native qwen3_asr support is detected
(transformers>=5.13.0, installed via `uv sync --extra transformers_qwen`).
"""
from typing import List, Optional, Union

import numpy as np
import torch

from cantocaptions_ai.pipeline.asr import QwenPipeline, _normalize_language
from cantocaptions_ai.utils.audio import resolve_device
from cantocaptions_ai.utils.schema import SingleSegment, TranscriptionResult, VadAudioSegment, ProgressCallback
from cantocaptions_ai.utils.model_utils import (
    partition_by_cache,
    BatchExecutor,
    MemoryPolicy,
    ensure_hf_model_downloaded,
    guard_model_load,
)
from cantocaptions_ai.cantonese.text import normalize_segment_text
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_MODEL_IDS = {
    "Qwen3-ASR":      "Qwen/Qwen3-ASR-1.7B-hf",
    "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B-hf",
}


def _compile_and_warmup(model, processor, language: str, batch_size: Optional[int]) -> None:
    """Compile model.forward and run a short warmup so the first real batch isn't the one paying the compile cost.

    Opt-in only (--compile) — benchmarked in scripts/bench_asr_compile.py against this
    workload's actual shape variance (VAD segments are essentially unique-length) and found
    to be a net loss in every configuration tried:
      - Default (graph break allowed): get_audio_cu_seqlens() in the audio tower does
        `.max().item()` + a Python loop over tensor-derived lengths, which dynamo can't trace.
        Each new (batch_size, audio_length) shape burns up to `recompile_limit` (8) full
        compile attempts (~20-30s each) before giving up and falling back to eager for that
        code location — tens of seconds wasted per distinct shape, for at most a ~2x speedup
        on the rare batch that repeats an exact prior shape before the budget trips.
      - dynamic=True: made the first-compile cost dramatically worse (137-408s), not better.
      - Padding every segment to a fixed shape (chunk_size worth of samples) does eliminate
        the recompiles and gives a real, stable steady-state speedup once compiled — but only
        pays off once the workload is already VRAM-healthy; at a starved batch_size, eager
        itself was 10-50x slower (near-OOM allocator overhead unrelated to compile), which
        made compile look better than it is. At a batch_size with real headroom, eager alone
        is already fast and stable, and compile's edge shrinks to ~25-40% against a ~30s+
        per-process tax.
      - That per-process tax cannot be cached away: neither torch.compiler.save_cache_artifacts/
        load_cache_artifacts (torch's "Mega-Cache", meant for cross-machine cache portability)
        nor the default on-disk Inductor cache avoid it — the dominant cost is TorchDynamo's
        own bytecode tracing/graph-break handling for this model, which happens once per fresh
        Python process regardless of what's cached on disk, not the kernel compilation itself.

    Uses a small batch of silent audio at roughly the configured batch size — torch.compile
    specializes to the shapes it first sees, so warming up near the real batch size avoids an
    extra recompilation on the first genuine inference call.
    """
    model.forward = torch.compile(model.forward)
    warmup_batch = max(1, min(batch_size or 4, 8))
    silence = [np.zeros(3 * 16000, dtype=np.float32) for _ in range(warmup_batch)]
    inputs = processor.apply_transcription_request(audio=silence, language=language)
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=8, do_sample=False)


def _warn_vram(inputs, batch_size: int, model, max_new_tokens: int, device, policy: MemoryPolicy) -> None:
    """Estimate one batch's peak VRAM (input tensors + generation KV-cache high-water)
    and log it against real headroom via *policy*. Callers guard on ``policy.enabled``
    so this whole function — including the estimate math below — is skipped when VRAM
    checks are off (measurably faster on the hot path; see the diagnosis findings).
    """
    dtype_bytes = next(model.parameters()).element_size()
    seq_len = inputs["input_ids"].shape[1]
    input_bytes = sum(t.numel() * t.element_size() for t in inputs.values() if isinstance(t, torch.Tensor))
    try:
        text_cfg = model.config.text_config
        kv_bytes = (
            batch_size * (seq_len + max_new_tokens)
            * text_cfg.num_hidden_layers * 2
            * text_cfg.num_key_value_heads * text_cfg.head_dim
            * dtype_bytes
        )
    except AttributeError:
        kv_bytes = 0
    estimated = input_bytes + kv_bytes
    policy.warn(
        f"ASR batch (batch_size={batch_size}, seq_len={seq_len})",
        device,
        estimated / 1e6,
        "consider reducing --batch_size or using --asr_compute_type int8",
    )


class QwenPipelineNative(QwenPipeline):
    """Native backend: uses AutoModelForMultimodalLM (Qwen3ASRForConditionalGeneration).

    Loads Qwen/Qwen3-ASR-1.7B-hf or -0.6B-hf via transformers' official qwen3_asr support.
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
        vram_checks: bool = True,
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
        self.vram_checks = vram_checks
        self.policy = MemoryPolicy(vram_checks)

    def run(self, items, *, debug_dir=None, load_debug_dir=None, progress_callback: ProgressCallback = None):
        """Transcribe all files, batching VAD segments across file boundaries.

        Segments from every to-compute file are flattened into one job stream, so
        batches pack work from different files (no half-empty tail batch per file).
        """
        logger.info("Performing transcription (native backend)...")
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

        # Longest-first: front-loads the largest KV-cache allocation so the allocator's
        # reserved pool is claimed once up front rather than ratcheting up over the run
        # (bench_asr_native.py --sort desc confirmed the win). Texts scatter back by
        # index, so output order is unaffected.
        BatchExecutor(
            self._batch_size,
            order_key=lambda job: len(buffers[job[0]]['segs'][job[1]]['audio']),
        ).run(jobs, infer_fn, reporter=progress_callback)

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

        BatchExecutor(
            self._batch_size,
            order_key=lambda i: len(input[i]['audio']),
        ).run(jobs, infer_fn, reporter=progress_callback)

        segments: List[SingleSegment] = [
            normalize_segment_text({'text': texts[i] or '', 'start': input[i]['start'], 'end': input[i]['end']})
            for i in range(len(input))
        ]
        return {"segments": segments, "language": language}

    def _infer_batch(self, wavs: List, language: str) -> List[str]:
        """Run one batch of audio arrays through the model, returning parsed texts.

        Raises RuntimeError on CUDA OOM (caught and retried at a smaller batch size by
        BatchExecutor); no shared state is mutated before the model call.
        """
        inputs = self.processor.apply_transcription_request(audio=wavs, language=language)
        inputs = inputs.to(self.model.device, self.model.dtype)

        if self.device.type == "cuda" and self.policy.enabled:
            _warn_vram(inputs, len(wavs), self.model, self.max_new_tokens, self.device, self.policy)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        # generate() returns a plain tensor by default; handle both for safety
        sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
        n_input = inputs["input_ids"].shape[1]
        logger.debug("[TOKENS] batch=%d, new_tokens=%d, max=%d", len(wavs), sequences.shape[1] - n_input, self.max_new_tokens)

        # Qwen3ASRProcessor.decode(return_format="transcription_only") strips the
        # "language <LANG><asr_text>" prefix and applies the same repetition-collapsing
        # dehallucination pass the original qwen_asr package used — no extra
        # post-processing needed here.
        texts = self.processor.decode(
            sequences[:, n_input:],
            return_format="transcription_only",
            clean_up_tokenization_spaces=False,
        )
        del inputs, generated

        for raw in texts:
            logger.debug("[RAW] %s", raw)
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
    compile_enabled: bool = False,
    print_progress: bool = False,
    verbose: bool = False,
    vram_checks: bool = True,
    vram_headroom_mb: int = 512,
) -> QwenPipelineNative:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_id = _MODEL_IDS.get(model_name, model_name)

    try:
        ensure_hf_model_downloaded(model_id, cache_dir=download_root, local_files_only=local_files_only)
    except Exception as e:
        logger.warning("Could not download %r: %s — using cached version if available.", model_id, e)

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.info("Compute type defaulting to %s for device %s", compute_type, device)

    device_map = resolve_device(device, device_index)
    pipeline_device = device_index if device == "cuda" else device

    logger.info("Loading ASR model %r (native backend, attn=%s)", model_id, attn_implementation)

    hf_model = model
    if hf_model is None:
        hf_model = guard_model_load(
            "ASR",
            "consider --asr_compute_type int8 or a lower --batch_size",
            lambda: AutoModelForMultimodalLM.from_pretrained(
                model_id,
                dtype=compute_type,
                device_map=device_map,
                attn_implementation=attn_implementation,
                local_files_only=local_files_only,
                cache_dir=download_root,
            ).eval(),
        )

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        cache_dir=download_root,
    )

    # Cap the allocator now that the weights are resident, so the reading reflects
    # true post-load free VRAM. Keeps generation's growing KV-cache from silently
    # paging into host RAM; a near-OOM instead raises and the BatchExecutor halves.
    if device == "cuda":
        MemoryPolicy(vram_checks, vram_headroom_mb).cap_after_load(device_index)

    if device == "cuda" and compile_enabled:
        try:
            _compile_and_warmup(hf_model, processor, _normalize_language(language or "yue"), batch_size)
            logger.info("torch.compile enabled for ASR model")
        except Exception as e:
            logger.warning(
                "torch.compile failed (%s); falling back to eager mode. "
                "Install the transformers_qwen extra for triton support.",
                e,
            )

    return QwenPipelineNative(
        model=hf_model,
        processor=processor,
        device=pipeline_device,
        language=language,
        batch_size=batch_size,
        max_new_tokens=200,
        print_progress=print_progress,
        verbose=verbose,
        vram_checks=vram_checks,
    )

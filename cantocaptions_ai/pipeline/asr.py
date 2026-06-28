import os
from typing import Any, List, Optional, Union

import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from cantocaptions_ai.utils.audio import resolve_device
from cantocaptions_ai.utils.schema import SingleSegment, TranscriptionResult, VadAudioSegment, ProgressCallback
from cantocaptions_ai.utils.model_utils import PipelineStage
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.output import LANGUAGES

logger = get_logger(__name__)


def _warn_vram(inputs, batch_size: int, model, max_new_tokens: int, device) -> None:
    """Log VRAM estimate for the current batch and warn if headroom looks tight."""
    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory
    free_vram = total_vram - torch.cuda.memory_allocated(device)

    dtype_bytes = next(model.parameters()).element_size()
    seq_len = inputs["input_ids"].shape[1]

    # Input tensor footprint (already on device, in model dtype)
    input_bytes = sum(t.numel() * t.element_size() for t in inputs.values() if isinstance(t, torch.Tensor))

    # KV cache estimate: batch × (seq_len + new_tokens) × layers × 2 (K+V) × kv_heads × head_dim × dtype
    text_cfg = model.config.thinker_config.text_config
    kv_bytes = (
        batch_size
        * (seq_len + max_new_tokens)
        * text_cfg.num_hidden_layers
        * 2
        * text_cfg.num_key_value_heads
        * text_cfg.head_dim
        * dtype_bytes
    )

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


class QwenPipeline(Pipeline, PipelineStage["List[VadAudioSegment]", "TranscriptionResult"]):
    """
    Huggingface Pipeline wrapper for Qwen3ASRModel.

    Batching is handled in `transcribe()` via a manual loop that calls the
    model processor and generate() directly, bypassing the high-level
    Qwen3ASRModel.transcribe() to avoid redundant audio normalization and
    to free GPU memory between batches.
    """

    def __init__(
        self,
        model,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        batch_size: Optional[int] = None,
        print_progress: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = batch_size
        self.print_progress = print_progress
        self.verbose = verbose
        self._preprocess_params, self._forward_params, self._postprocess_params = {}, {}, {}
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, input_, **_preprocess_parameters):
        """Process a single audio segment into model inputs."""
        wav = input_['audio']
        language = input_.get('language') or self.preset_language or "Cantonese"
        prompt = self.model._build_text_prompt(context="", force_language=language)
        inputs = self.model.processor(text=[prompt], audio=[wav], return_tensors="pt")
        inputs['_language'] = language
        return inputs

    def _forward(self, input_tensors, **_forward_parameters) -> Any:
        """Run generation on a single set of model inputs."""
        language = input_tensors.pop('_language', self.preset_language or "Cantonese")
        inputs = input_tensors.to(self.model.model.device).to(self.model.model.dtype)
        with torch.no_grad():
            generated = self.model.model.generate(**inputs, max_new_tokens=self.model.max_new_tokens)
        n_input = inputs["input_ids"].shape[1]
        return {"sequences": generated.sequences, "n_input_tokens": n_input, "_language": language}

    def postprocess(self, model_outputs, **_postprocess_parameters):
        """Decode generated tokens to text."""
        from qwen_asr.inference.utils import parse_asr_output
        language = model_outputs.get('_language', self.preset_language or "Cantonese")
        sequences = model_outputs["sequences"]
        n_input = model_outputs["n_input_tokens"]
        decoded = self.model.processor.batch_decode(
            sequences[:, n_input:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        _, text = parse_asr_output(decoded[0], user_language=language)
        return {"text": text}

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model_iterator = PipelineIterator(dataset, self.forward, forward_params)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def process(
        self,
        input: List[VadAudioSegment],
        *,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Implements PipelineStage: delegates to transcribe() using constructor-stored config."""
        return self.transcribe(input, progress_callback=progress_callback)

    def transcribe(
        self,
        vad_segments: List[VadAudioSegment],
        language: Optional[str] = None,
        combined_progress: bool = False,
        progress_callback: ProgressCallback = None,
        use_native: bool = False,
    ) -> TranscriptionResult:
        from qwen_asr.inference.utils import parse_asr_output

        language = language or self.preset_language or "Cantonese"

        if use_native:
            language_longname = LANGUAGES.get(language, language)
            wavs = [(seg['audio'], 16000) for seg in vad_segments]
            results = self.model.transcribe(wavs, language=language_longname)
            segments = [
                {'text': r.text, 'start': seg['start'], 'end': seg['end']}
                for r, seg in zip(results, vad_segments)
            ]
            if progress_callback is not None:
                progress_callback(1.0)
            return {"segments": segments, "language": language}

        effective_batch = self._batch_size or self.model.max_inference_batch_size
        if not effective_batch or effective_batch < 1:
            effective_batch = len(vad_segments)

        segments: List[SingleSegment] = []
        total = len(vad_segments)
        current_batch = effective_batch
        oom_warned = False
        i = 0

        while i < total:
            batch = vad_segments[i:i + current_batch]
            wavs = [seg['audio'] for seg in batch]
            prompts = [self.model._build_text_prompt(context="", force_language=language) for _ in batch]

            try:
                inputs = self.model.processor(text=prompts, audio=wavs, return_tensors="pt", padding=True)
                inputs = inputs.to(self.model.model.device).to(self.model.model.dtype)

                if self.device.type == "cuda":
                    _warn_vram(inputs, len(batch), self.model.model, self.model.max_new_tokens, self.device)

                with torch.no_grad():
                    generated = self.model.model.generate(**inputs, max_new_tokens=self.model.max_new_tokens)

                n_input = inputs["input_ids"].shape[1]
                actual_new = generated.sequences.shape[1] - n_input
                logger.debug("[TOKENS] batch=%d, new_tokens=%d, max=%d", len(wavs), actual_new, self.model.max_new_tokens)

                decoded = self.model.processor.batch_decode(
                    generated.sequences[:, n_input:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                del inputs, generated, wavs

                for vad_seg, raw in zip(batch, decoded):
                    logger.debug("[RAW] %s", raw)
                    _, text = parse_asr_output(raw, user_language=language)
                    segments.append({
                        'text': text,
                        'start': vad_seg['start'],
                        'end': vad_seg['end'],
                    })

                i += len(batch)
                if progress_callback is not None:
                    progress_callback(i / total)

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                if current_batch <= 1:
                    raise RuntimeError(
                        "CUDA out of memory even at batch_size=1. "
                        "Try freeing VRAM or reducing --chunk_size."
                    ) from e
                current_batch = max(1, current_batch // 2)
                if not oom_warned:
                    logger.warning(
                        "CUDA out of memory — retrying with batch_size=%d. "
                        "Pass --batch_size %d to avoid this next time.",
                        current_batch, current_batch,
                    )
                    oom_warned = True

        return {"segments": segments, "language": language}


def load_model(
    model_name: str,
    device: str,
    device_index=0,
    compute_type="default",
    asr_options: Optional[dict] = None,
    language: Optional[str] = "yue",
    vocal_isolation_method: Optional[str] = None,
    model=None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
    use_auth_token: Optional[Union[str, bool]] = None,
    batch_size: Optional[int] = None,
    print_progress: bool = False,
    verbose: bool = False,
) -> QwenPipeline:
    """Load a Qwen ASR model for inference."""
    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.info(f"Compute type not specified, defaulting to {compute_type} for device {device}")

    device_map = resolve_device(device, device_index)
    pipeline_device = device_index if device == "cuda" else device

    from qwen_asr import Qwen3ASRModel
    qwen_model = model or Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=compute_type,
        device_map=device_map,
        attn_implementation="sdpa",
        max_inference_batch_size=batch_size or 24,
        max_new_tokens=256,
    )

    return QwenPipeline(
        model=qwen_model,
        device=pipeline_device,
        language=language,
        batch_size=batch_size,
        print_progress=print_progress,
        verbose=verbose,
    )

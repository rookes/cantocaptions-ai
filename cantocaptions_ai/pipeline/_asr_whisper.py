"""Whisper ASR backend: encoder-decoder transcription through transformers.

Each VAD segment (at most ``chunk_size``, 28 s by default) fits Whisper's 30 s window as
it is, so a segment is one forward pass with no long-form chunking. Batching, OOM handling,
checkpointing and the language pack's normalization are BatchedAsrStage's.

Whisper is asked for timestamps, but not for their times -- forced alignment does the
timing, as for every backend. What they are used for is where they fall: Whisper closes a
phrase with one, and on Cantonese it writes little punctuation of its own, so without
them a whole VAD chunk came out as one unpunctuated run and became one cue (131 cues
against 897 in the ground truth on the eval episodes). Each phrase boundary is written as
a clause mark instead (``asr.join_clauses``).

Whisper writes many languages in a generic standard form (Cantonese comes out much like
stock Qwen3-ASR's, Mandarin-flavoured), so its conventions per language live in the
language pack like any other model's.
"""
from typing import List, Optional, Union

import torch

from cantocaptions_ai.pipeline.asr import BatchedAsrStage, join_clauses
from cantocaptions_ai.pipeline.model_profiles import get_model_profile
from cantocaptions_ai.text_profiles import (
    DEFAULT_NORMALIZATION,
    DEFAULT_PUNCTUATION,
    DEFAULT_SCRIPT,
    PunctuationConfig,
    ScriptConfig,
    TextNormalization,
)
from cantocaptions_ai.utils.audio import SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.model_utils import (
    MemoryPolicy,
    ensure_hf_model_downloaded,
    guard_model_load,
    resolve_torch_compute_dtype,
)

logger = get_logger(__name__)


class WhisperAsr(BatchedAsrStage):
    """Whisper (``WhisperForConditionalGeneration``), forced to transcribe in one language."""

    backend_label = "whisper"

    def __init__(
        self,
        model,
        processor,
        device: Union[int, str, "torch.device"],
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
        print_progress: bool = False,
        verbose: bool = False,
        vram_checks: bool = True,
        normalization: TextNormalization = DEFAULT_NORMALIZATION,
        punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
        script: ScriptConfig = DEFAULT_SCRIPT,
    ):
        super().__init__(
            device=device, language=language, batch_size=batch_size,
            print_progress=print_progress, verbose=verbose, vram_checks=vram_checks,
            normalization=normalization,
        )
        self.model = model
        self.processor = processor
        self.punctuation = punctuation
        self.script = script
        # Token ids from <|0.00|> up are timestamps.
        self._timestamp_begin = processor.tokenizer.convert_tokens_to_ids("<|0.00|>")

    def _infer_batch(self, wavs: List, language: str, contexts=None) -> List[str]:
        """One batch of segments -> one text each, its phrases joined as clauses.

        ``language`` is the ISO code Whisper's tokenizer takes (``yue``, ``en``, ...); forcing
        it, rather than letting Whisper detect per segment, keeps a short or noisy segment
        from coming back in another language.
        """
        features = self.processor(
            wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True,
        )
        input_features = features["input_features"].to(self.model.device, self.model.dtype)
        attention_mask = features.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(
                input_features, attention_mask=attention_mask,
                language=language, task="transcribe", return_timestamps=True,
            )
        return [self._join_phrases(row.tolist()) for row in ids]

    def _join_phrases(self, ids: List[int]) -> str:
        phrases, current = [], []
        for token in ids:
            if token >= self._timestamp_begin:
                phrases.append(current)
                current = []
            else:
                current.append(token)
        phrases.append(current)
        texts = self.processor.batch_decode(phrases, skip_special_tokens=True)
        return join_clauses(texts, self.punctuation, self.script)


def _whisper_dtype(compute_type: str, device: str) -> torch.dtype:
    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
    if compute_type == "int8":
        logger.warning("Whisper does not run at int8 here; using float16 (float32 off CUDA).")
        compute_type = "float16"
    return resolve_torch_compute_dtype(compute_type, device, "ASR")


def load_model_whisper(
    model_name: Optional[str],
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
    vram_checks: bool = True,
    vram_headroom_mb: int = 512,
    processor=None,
    normalization=None,
) -> WhisperAsr:
    """Load a Whisper checkpoint (hub id or path, via the model registry) as an ASR stage."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    from cantocaptions_ai.pipeline.asr import _resolve_normalization

    from cantocaptions_ai.languages import get_language_pack

    model_id = get_model_profile(model_name, language).hf_id
    normalization = _resolve_normalization(model_name, language, normalization)
    run_profile = get_language_pack(language).resolve(model_name)

    if model is None or processor is None:
        try:
            ensure_hf_model_downloaded(model_id, cache_dir=download_root, local_files_only=local_files_only)
        except Exception as e:
            logger.warning("Could not download %r: %s — using cached version if available.", model_id, e)

    dtype = _whisper_dtype(compute_type, device)
    logger.info("ASR model: %s (whisper backend, %s)", model_id, dtype)
    if model is None:
        model = guard_model_load(
            "ASR",
            "consider a lower --batch_size",
            lambda: AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, dtype=dtype, attn_implementation=attn_implementation,
                local_files_only=local_files_only, cache_dir=download_root,
            ).to(resolve_device(device, device_index)).eval(),
        )
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=local_files_only, cache_dir=download_root,
        )
    if device == "cuda":
        MemoryPolicy(vram_checks, vram_headroom_mb).cap_after_load(device_index)

    return WhisperAsr(
        model=model, processor=processor,
        device=device_index if device == "cuda" else device,
        language=language, batch_size=batch_size, print_progress=print_progress,
        verbose=verbose, vram_checks=vram_checks, normalization=normalization,
        punctuation=run_profile.punctuation, script=run_profile.script,
    )

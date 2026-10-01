"""CTC ASR backend: greedy decoding of a wav2vec2-family model (``AutoModelForCTC``).

For many smaller languages a fine-tuned wav2vec2 / wav2vec2-BERT / HuBERT CTC model is the
only ASR there is. The forward pass is the alignment code's own
(``alignment._compute_vad_emissions_batched``), which already handles both input families
(fbank features and raw samples), batch padding and trimming each row back to its real
length; this backend takes the per-frame argmax and lets the processor collapse repeats,
drop blanks and turn the word delimiter into a space.

CTC output carries no punctuation, so cues split only at pauses -- expected for the family.
"""
from typing import List, Optional, Union

import torch

from cantocaptions_ai.pipeline.asr import BatchedAsrStage
from cantocaptions_ai.pipeline.model_profiles import get_model_profile
from cantocaptions_ai.text_profiles import DEFAULT_NORMALIZATION, TextNormalization
from cantocaptions_ai.utils.audio import SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.model_utils import (
    MemoryPolicy,
    ensure_hf_model_downloaded,
    guard_model_load,
    resolve_torch_compute_dtype,
)

logger = get_logger(__name__)


class CtcAsr(BatchedAsrStage):
    """A CTC model transcribing one segment per row, greedily."""

    backend_label = "ctc"

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
    ):
        super().__init__(
            device=device, language=language, batch_size=batch_size,
            print_progress=print_progress, verbose=verbose, vram_checks=vram_checks,
            normalization=normalization,
        )
        self.model = model
        self.processor = processor

    def _infer_batch(self, wavs: List, language: str, contexts=None) -> List[str]:
        from cantocaptions_ai.pipeline.alignment import (
            MIN_ALIGN_SAMPLES,
            _compute_vad_emissions_batched,
        )

        # Below one feature frame the extractor fails outright, and there is nothing to hear.
        usable = [i for i, w in enumerate(wavs) if len(w) >= MIN_ALIGN_SAMPLES]
        segments = [{"start": 0.0, "end": len(wavs[i]) / SAMPLE_RATE, "audio": wavs[i]}
                    for i in usable]
        emissions = _compute_vad_emissions_batched(
            segments, self.model, self.processor, self.model.device,
            batch_size=max(len(segments), 1), vram_checks=self.vram_checks,
        ) if segments else []
        texts = [""] * len(wavs)
        for i, (emission, _rate) in zip(usable, emissions):
            texts[i] = self.processor.decode(emission.argmax(dim=-1)).strip()
        return texts


def load_model_ctc(
    model_name: Optional[str],
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    attn_implementation: Optional[str] = None,
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
) -> CtcAsr:
    """Load a CTC checkpoint (hub id or path, via the model registry) as an ASR stage.

    ``attn_implementation`` is accepted for a uniform loader signature and ignored: the
    wav2vec2 family's attention choice is not worth a per-model compatibility table here.
    """
    from transformers import AutoModelForCTC, AutoProcessor

    from cantocaptions_ai.pipeline.asr import _resolve_normalization

    model_id = get_model_profile(model_name, language).hf_id
    normalization = _resolve_normalization(model_name, language, normalization)

    if model is None or processor is None:
        try:
            ensure_hf_model_downloaded(model_id, cache_dir=download_root, local_files_only=local_files_only)
        except Exception as e:
            logger.warning("Could not download %r: %s — using cached version if available.", model_id, e)

    # Loaded as alignment loads the same models: float32 unless float16 is asked for.
    dtype = resolve_torch_compute_dtype(compute_type, device, "ASR")
    logger.info("Loading ASR model %r (ctc backend, %s)", model_id, dtype)
    if model is None:
        model = guard_model_load(
            "ASR",
            "consider a lower --batch_size",
            lambda: AutoModelForCTC.from_pretrained(
                model_id, local_files_only=local_files_only, cache_dir=download_root,
            ).to(resolve_device(device, device_index), dtype=dtype).eval(),
        )
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=local_files_only, cache_dir=download_root,
        )
    if device == "cuda":
        MemoryPolicy(vram_checks, vram_headroom_mb).cap_after_load(device_index)

    return CtcAsr(
        model=model, processor=processor,
        device=device_index if device == "cuda" else device,
        language=language, batch_size=batch_size, print_progress=print_progress,
        verbose=verbose, vram_checks=vram_checks, normalization=normalization,
    )

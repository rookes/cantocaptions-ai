"""CTC ASR backend: greedy decoding of a wav2vec2-family model (``AutoModelForCTC``).

For many smaller languages a fine-tuned wav2vec2 / wav2vec2-BERT / HuBERT CTC model is the
only ASR there is. The forward pass is the alignment code's own
(``alignment._compute_vad_emissions_batched``), which already handles both input families
(fbank features and raw samples), batch padding and trimming each row back to its real
length; this backend takes the per-frame argmax and lets the processor collapse repeats,
drop blanks and turn the word delimiter into a space.

CTC output carries no punctuation, and punctuation is where alignment cuts a chunk into
clauses and cue assembly finds its boundaries: left bare, every VAD chunk (up to 28 s)
becomes one cue. The greedy path already says where the speaker paused, though -- a long
run of blank frames -- so each such pause is written as the language's mergeable clause
mark (``，`` / ``,``). That gives alignment the clause cuts punctuation would, while cue
assembly stays free to rejoin short clauses, since the mark is a mergeable one.
"""
from typing import List, Optional, Union

import torch

from cantocaptions_ai.pipeline.asr import BatchedAsrStage
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

# A blank run at least this long, inside a segment, is written as a clause mark.
PAUSE_SECONDS = 0.3


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
        punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
        script: ScriptConfig = DEFAULT_SCRIPT,
        pause_seconds: float = PAUSE_SECONDS,
    ):
        super().__init__(
            device=device, language=language, batch_size=batch_size,
            print_progress=print_progress, verbose=verbose, vram_checks=vram_checks,
            normalization=normalization,
        )
        self.model = model
        self.processor = processor
        self.script = script
        self.pause_mark = punctuation.mergeable_chars[0] if punctuation.mergeable_chars else None
        self.pause_seconds = pause_seconds
        tokenizer = processor.tokenizer
        # Frames that carry no character: CTC blank, and the word delimiter where there is one.
        self._silent_ids = {tokenizer.pad_token_id}
        delimiter = getattr(tokenizer, "word_delimiter_token_id", None)
        if delimiter is not None:
            self._silent_ids.add(delimiter)

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
        for i, (emission, rate) in zip(usable, emissions):
            texts[i] = self._decode(emission.argmax(dim=-1).tolist(), rate)
        return texts

    def _decode(self, ids: List[int], frame_rate: float) -> str:
        """Greedy CTC decode, with each long enough internal pause written as a clause mark."""
        pieces = [ids]
        if self.pause_mark is not None and frame_rate > 0:
            pieces = _split_at_pauses(ids, self._silent_ids,
                                      max(1, round(self.pause_seconds * frame_rate)))
        text = ""
        for piece in pieces:
            words = self.processor.decode(piece).strip()
            if not words:
                continue
            text = self.script.join(text + self.pause_mark, words) if text else words
        return text


def _split_at_pauses(ids: List[int], silent_ids, min_frames: int) -> List[List[int]]:
    """Cut a frame path in the middle of every silent run of at least *min_frames* frames
    that has speech on both sides. Cutting inside a blank run keeps CTC's repeat collapse
    unchanged: a character either side of it was already two characters."""
    pieces, start, run_start = [], 0, None
    spoke = False
    for t, token in enumerate(ids):
        if token in silent_ids:
            if run_start is None:
                run_start = t
            continue
        if run_start is not None and spoke and t - run_start >= min_frames:
            cut = (run_start + t) // 2
            pieces.append(ids[start:cut])
            start = cut
        run_start, spoke = None, True
    pieces.append(ids[start:])
    return pieces


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

    from cantocaptions_ai.languages import get_language_pack

    model_id = get_model_profile(model_name, language).hf_id
    normalization = _resolve_normalization(model_name, language, normalization)
    run_profile = get_language_pack(language).resolve(model_name)

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
        punctuation=run_profile.punctuation, script=run_profile.script,
    )

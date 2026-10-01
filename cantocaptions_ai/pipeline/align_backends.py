"""Forced-alignment model families: how each one is loaded and run.

Alignment itself (``pipeline/alignment.py``) is model-agnostic: it needs a character
dictionary, and log-softmax CTC emissions for the audio. Getting those out of a model is
the backend's job, and it is the only part that differs between families:

* ``huggingface`` -- any Hugging Face CTC checkpoint (wav2vec2, wav2vec2-BERT, HuBERT,
  WavLM, ...), through the Auto classes and the model's own processor. Batched, and the
  only family that takes an align profile's audio primer.
* ``torchaudio`` -- a torchaudio pipeline bundle by name (``WAV2VEC2_ASR_BASE_960H``, the
  VoxPopuli models): raw audio in, one segment at a time.

The names are the values ``load_align_model`` has always written to ``metadata["type"]``,
which callers outside this package read, so they are kept. ``align_backend_for`` decides a
model's family; ``get_align_backend`` looks one up by name. A new family is one subclass
and one ``ALIGN_BACKENDS`` entry.
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.model_utils import (
    BatchExecutor,
    _looks_offline,
    check_vram_headroom,
    ensure_hf_model_downloaded,
    guard_model_load,
)
from cantocaptions_ai.utils.schema import VadAudioSegment

if TYPE_CHECKING:
    from cantocaptions_ai.pipeline.align_profiles import AudioPrimer

logger = get_logger(__name__)

# Rough fp32 params + activation footprint for wav2vec2-BERT-cantonese; used only for
# the preflight VRAM-headroom warning, not an exact bound.
_ALIGN_MODEL_VRAM_ESTIMATE_MB = 1200
_ALIGN_REMEDIATION = "pass --no_align to skip alignment, or free VRAM used by other processes/stages"

# config.json model_type values of the Hugging Face CTC families. Alignment needs a CTC
# model, and so does the CTC ASR backend, which reads the same list.
CTC_MODEL_TYPES = (
    "wav2vec2", "wav2vec2-bert", "wav2vec2-conformer", "hubert", "wavlm",
    "data2vec-audio", "sew", "sew-d", "unispeech", "unispeech-sat",
)


# --- Hugging Face CTC: the batched path ------------------------------------------------

def _input_key(processor) -> str:
    """The model input a processor produces: ``input_features`` for wav2vec2-BERT (log-mel
    fbank frames), ``input_values`` for plain wav2vec2 (normalised raw samples)."""
    names = getattr(processor, "model_input_names", None)
    return names[0] if names else "input_features"


def _passes_attention_mask(processor) -> bool:
    """Whether the model should be *given* the padding mask.

    The mask is always requested -- its row sums are each segment's real length -- but the
    HF convention is that a feature extractor with ``return_attention_mask=False`` belongs to
    a model that must not see one (wav2vec2-base and other group-norm checkpoints, which are
    trained on zero-padded batches and degrade when masked).
    """
    extractor = getattr(processor, "feature_extractor", processor)
    return bool(getattr(extractor, "return_attention_mask", True))


def _warn_alignment_vram(input_features: torch.Tensor, model: torch.nn.Module, device: str) -> None:
    """Estimate one batch's peak VRAM use from its actual (padded) shape and log it
    against real headroom — mirrors _asr_native.py's _warn_vram, but for a
    bidirectional CTC encoder with no KV-cache instead of autoregressive generation.
    The caller guards on ``vram_checks`` so this (including the estimate math) is
    skipped entirely when checks are off.

    Rough proxy, not an exact bound: the padded input tensor itself, plus one
    conformer layer's transient self-attention score matrix (batch * heads *
    frames^2) and FFN intermediate activation (batch * frames * intermediate) —
    the dominant terms, since inference_mode lets earlier layers' activations be
    freed as later layers run, so peak memory tracks roughly one layer's working
    set rather than the sum across all layers.
    """
    dtype_bytes = input_features.element_size()
    batch, max_frames = input_features.shape[0], input_features.shape[1]
    input_bytes = input_features.numel() * dtype_bytes
    try:
        cfg = model.config
        attn_bytes = batch * cfg.num_attention_heads * max_frames * max_frames * dtype_bytes
        ffn_bytes = batch * max_frames * cfg.intermediate_size * dtype_bytes
        activation_bytes = attn_bytes + ffn_bytes
    except AttributeError:
        activation_bytes = 0
    check_vram_headroom(
        f"Alignment batch (batch_size={batch}, max_frames={max_frames})",
        device,
        (input_bytes + activation_bytes) / 1e6,
        "consider reducing --align_batch_size or --chunk_size",
    )


def _compute_vad_emissions_batched(
    vad_segments: List[VadAudioSegment],
    model: torch.nn.Module,
    processor,
    device: str,
    batch_size: int,
    vram_checks: bool = True,
    primer: Optional["AudioPrimer"] = None,
    dtype: Optional[torch.dtype] = None,
) -> List[Tuple[torch.Tensor, float]]:
    """Batch VAD segments through a Hugging Face CTC align model via BatchExecutor.

    ``processor`` is the align model's own (``align_metadata["processor"]``). Its first
    model input is what the model is fed -- fbank ``input_features`` for wav2vec2-BERT, raw
    ``input_values`` for plain wav2vec2 -- and the padding mask is passed on only where the
    extractor says the model expects one (see ``_passes_attention_mask``).

    ``primer`` (from the align model's profile, ``None`` for a model without one) prepends
    left context to each segment before the encoder sees it, and its frames are discarded
    here so nothing downstream knows it existed — see ``align_profiles.TailPrimer`` for why
    the current model needs it. The prefix length is deliberately *not* asked of the primer:
    the encoder's own output length for the **unprimed** audio is measured and that many
    frames are kept from the end, which is exact whatever the primer prepended and keeps a
    future primer free to change shape. That costs one extra feature-extraction pass per
    batch (cheap next to the forward) and is worth it — estimating the offset from duration
    instead would be a frame out often enough to reintroduce the artifact it removes.

    The processor pads each batch to its own longest segment and returns an attention_mask,
    whose per-row sum is the segment's real length **in model-input units** (feature frames for
    wav2vec2-BERT, samples for wav2vec2). That is not the same unit as the CTC emission: alvanlii/wav2vec2-BERT-cantonese
    sets add_adapter=True with adapter_stride=2, so the emission runs at half the feature rate
    (~25 fps vs ~50 fps). The mask length is therefore converted through the model's own
    _get_feat_extract_output_lengths before it is used to trim each row's emission back to its
    real (unpadded) length. That helper maps the model's input unit to emission frames for
    either family: the adapter convs for wav2vec2-BERT (an identity without an adapter), the
    conv feature encoder for wav2vec2.

    Getting this wrong is silent and costly: an over-long real_len makes the slice a no-op, so
    the segment keeps the whole batch-padded emission, _align_segment's
    `ratio = duration / (trellis.size(0) - 1)` divides the true duration by too many frames,
    and every timestamp in the segment compresses toward its start (seconds of drift by the
    end). Only the longest segment in each batch escapes. Hence the assertion below.

    Jobs are processed longest-segment-first (not VAD order) via BatchExecutor's
    order_key, for two reasons: VAD segments range from sub-second to the full
    --chunk_size (default 30s), and self-attention's O(frames^2) memory scaling
    means one long segment sharing a batch with several short ones pads all of them
    up to the long one's length — spiking peak VRAM well above what the batch_size
    alone suggests. Sorting by length groups similar-duration segments together
    instead, so no batch pads far past its own natural size. Processing longest-first
    also matters for the CUDA caching allocator: if batches were processed
    shortest-first, every batch that needs a new largest-yet shape would force a
    fresh, ever-larger cudaMalloc (old smaller cached blocks can't be reused for it
    and are never freed back to the driver mid-stage), so reserved VRAM would climb
    monotonically over the course of the stage even though each batch's actual usage
    stays small — until the device runs out and the driver falls back to slow memory
    paging. Starting with the largest batch makes the allocator's one big allocation
    happen up front, and every smaller batch after that reuses/splits the same
    cached block.
    """
    results: List[Optional[Tuple[torch.Tensor, float]]] = [None] * len(vad_segments)
    jobs = list(range(len(vad_segments)))

    model_dtype = next(model.parameters()).dtype
    input_key = _input_key(processor)
    pass_mask = _passes_attention_mask(processor)

    def _emission_lens(wavs) -> torch.Tensor:
        features = processor(
            wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True,
            padding=True,
        )
        # Model-input units -> emission frames (see docstring).
        return model._get_feat_extract_output_lengths(features["attention_mask"].sum(dim=-1))

    def infer_fn(batch: List[int]) -> None:
        wavs = [vad_segments[i]["audio"] for i in batch]
        model_inputs = [primer(w, SAMPLE_RATE) for w in wavs] if primer is not None else wavs
        with torch.inference_mode():
            # padding=True explicitly: the wav2vec2-BERT extractor pads by default, but the
            # plain wav2vec2 one does not and cannot tensorise a ragged batch without it.
            features = processor(
                model_inputs, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                return_attention_mask=True, padding=True,
            )
            input_features = features[input_key].to(device, dtype=model_dtype)
            attention_mask = features["attention_mask"].to(device) if pass_mask else None
            if vram_checks:
                _warn_alignment_vram(input_features, model, device)
            emissions = torch.log_softmax(
                model(input_features, attention_mask=attention_mask).logits, dim=-1
            )
            valid_lens = features["attention_mask"].sum(dim=-1)
            emission_lens = model._get_feat_extract_output_lengths(valid_lens)
            # How many frames the segment alone is worth; the rest of the row is primer.
            plain_lens = _emission_lens(wavs) if primer is not None else emission_lens
        for row, i in enumerate(batch):
            real_len = int(emission_lens[row].item())
            if real_len > emissions.shape[1]:
                raise RuntimeError(
                    f"Alignment emission trim is longer than the emission itself "
                    f"({real_len} > {emissions.shape[1]} frames). The feature-frame -> "
                    f"emission-frame conversion does not match this align model; timestamps "
                    f"would silently compress. Check the model's adapter config."
                )
            keep = int(plain_lens[row].item())
            if keep > real_len:
                raise RuntimeError(
                    f"Primed alignment emission is shorter than the unprimed segment it "
                    f"contains ({real_len} < {keep} frames). The primer must return its "
                    f"prefix followed by the original audio unchanged."
                )
            emission = emissions[row, real_len - keep:real_len, :].detach().to("cpu", dtype=dtype)
            vad_duration = vad_segments[i]["end"] - vad_segments[i]["start"]
            frame_rate = emission.size(0) / vad_duration if vad_duration > 0 else 0.0
            results[i] = (emission, frame_rate)

    BatchExecutor(
        batch_size, order_key=lambda i: len(vad_segments[i]["audio"]),
    ).run(jobs, infer_fn)

    # A correct run yields the model's constant frame rate for every segment regardless of
    # length. Spread means some emission still carries batch padding, which shows up as
    # timestamps compressed toward the segment start -- cheap to check, and otherwise silent.
    rates = [r[1] for r in results if r is not None and r[1] > 0]
    if rates:
        median_rate = sorted(rates)[len(rates) // 2]
        spread = max(abs(rate - median_rate) for rate in rates) / median_rate
        if spread > 0.02:
            logger.warning(
                "Alignment emission frame rate varies by %.1f%% across VAD segments "
                "(median %.2f fps, range %.2f-%.2f). Timestamps in the outlying segments are "
                "likely compressed; suspect the emission length conversion.",
                spread * 100, median_rate, min(rates), max(rates),
            )
    return results



# --- The backends ------------------------------------------------------------------------

class AlignBackend:
    """One family of forced-alignment models. Subclasses fill in every method."""

    name: str = ""
    # Whether emissions() applies an align profile's AudioPrimer.
    supports_primer: bool = False

    def load(self, model_name: str, device: str, dtype: torch.dtype, *, model_dir=None,
             cache_only: bool = False, vram_checks: bool = True):
        """Load *model_name* onto *device*: ``(model, processor or None, dictionary)``."""
        raise NotImplementedError

    def emissions(self, segments: List[VadAudioSegment], model, processor, device: str,
                  batch_size: int, *, vram_checks: bool = True, primer=None,
                  dtype: Optional[torch.dtype] = None) -> List[Tuple[torch.Tensor, float]]:
        """``(log-softmax emission, frames per second)`` for each segment, in order."""
        raise NotImplementedError

    def forward(self, model, processor, audio: torch.Tensor, device: str,
                lengths=None) -> torch.Tensor:
        """One forward pass over a ``(1, samples)`` waveform: log-softmax emissions."""
        raise NotImplementedError

    def frame_rate(self, model, processor, device: str) -> float:
        """Emission frames per second of audio."""
        raise NotImplementedError


class HuggingFaceCtcAlign(AlignBackend):
    name = "huggingface"
    supports_primer = True

    def load(self, model_name, device, dtype, *, model_dir=None, cache_only=False,
             vram_checks=True):
        # The Auto classes resolve the model family from the checkpoint's own config
        # (wav2vec2-bert -> Wav2Vec2BertForCTC + its fbank processor, wav2vec2 -> Wav2Vec2ForCTC
        # + its raw-sample processor, and so on) rather than from the repo name.
        from transformers import AutoModelForCTC, AutoProcessor
        try:
            ensure_hf_model_downloaded(model_name, cache_dir=model_dir, local_files_only=cache_only)
        except Exception as e:
            logger.warning("Could not download %r: %s — using cached version if available.", model_name, e)
        try:
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_dir, local_files_only=cache_only)
            model = AutoModelForCTC.from_pretrained(model_name, cache_dir=model_dir, local_files_only=cache_only)
        except Exception as e:
            logger.error("Error loading model from huggingface (%s): %s", model_name, e)
            raise ValueError(_not_found(model_name))
        check_vram_headroom("Alignment model load", device, _ALIGN_MODEL_VRAM_ESTIMATE_MB, _ALIGN_REMEDIATION, vram_checks=vram_checks)
        model = guard_model_load("alignment", _ALIGN_REMEDIATION, lambda: model.to(device, dtype=dtype))
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                f'align_model "{model_name}" has no CTC tokenizer (its processor is a bare '
                f'{type(processor).__name__}); alignment needs the character vocabulary'
            )
        dictionary = {char.lower(): code for char, code in tokenizer.get_vocab().items()}
        return model, processor, dictionary

    def emissions(self, segments, model, processor, device, batch_size, *, vram_checks=True,
                  primer=None, dtype=None):
        if processor is None:
            raise ValueError(
                "A Hugging Face align model needs its processor; pass align_metadata['processor']"
            )
        return _compute_vad_emissions_batched(
            segments, model, processor, device, batch_size,
            vram_checks=vram_checks, primer=primer, dtype=dtype,
        )

    def forward(self, model, processor, audio, device, lengths=None):
        model_dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            if processor is not None:
                # One 1-D array per row: the wav2vec2-BERT extractor would take the (1, n)
                # tensor as it is, but plain wav2vec2's wraps it into (1, 1, n), which its
                # conv front end cannot read.
                rows = list(audio.cpu().numpy()) if audio.dim() == 2 else audio
                features = processor(rows, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inputs = features[_input_key(processor)]
                emissions = model(inputs.to(device, dtype=model_dtype)).logits
            else:
                emissions = model(audio.to(device, dtype=model_dtype)).logits
            return torch.log_softmax(emissions, dim=-1)

    def frame_rate(self, model, processor, device):
        """From the model's own length arithmetic: the processor's feature length for 100 s
        of audio, run through ``_get_feat_extract_output_lengths`` -- exact, no forward pass."""
        seconds = 100
        silence = np.zeros(SAMPLE_RATE * seconds, dtype=np.float32)
        with torch.inference_mode():
            features = processor(silence, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                 return_attention_mask=True)
            frames = int(model._get_feat_extract_output_lengths(
                features["attention_mask"].sum(dim=-1))[0])
        return frames / seconds


class TorchaudioAlign(AlignBackend):
    name = "torchaudio"

    def load(self, model_name, device, dtype, *, model_dir=None, cache_only=False,
             vram_checks=True):
        import torchaudio
        bundle = torchaudio.pipelines.__dict__[model_name]
        check_vram_headroom("Alignment model load", device, _ALIGN_MODEL_VRAM_ESTIMATE_MB, _ALIGN_REMEDIATION, vram_checks=vram_checks)
        model = guard_model_load(
            "alignment", _ALIGN_REMEDIATION,
            lambda: bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device, dtype=dtype),
        )
        dictionary = {c.lower(): i for i, c in enumerate(bundle.get_labels())}
        return model, None, dictionary  # bundles take raw audio and have no processor

    def emissions(self, segments, model, processor, device, batch_size, *, vram_checks=True,
                  primer=None, dtype=None):
        """One segment at a time: a bundle has no processor to pad a batch through."""
        results = []
        for vad_seg in segments:
            seg_audio = vad_seg["audio"]
            if not torch.is_tensor(seg_audio):
                seg_audio = torch.from_numpy(seg_audio)
            if len(seg_audio.shape) == 1:
                seg_audio = seg_audio.unsqueeze(0)

            emissions = self.forward(model, processor, seg_audio, device)
            emission = emissions[0].detach().to("cpu", dtype=dtype)
            vad_duration = vad_seg["end"] - vad_seg["start"]
            frame_rate = emission.size(0) / vad_duration if vad_duration > 0 else 0.0
            results.append((emission, frame_rate))
        return results

    def forward(self, model, processor, audio, device, lengths=None):
        model_dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            emissions, _ = model(audio.to(device, dtype=model_dtype), lengths=lengths)
            return torch.log_softmax(emissions, dim=-1)

    def frame_rate(self, model, processor, device):
        """A bundle has no length helper, so ten seconds of silence go through the model once
        and the rate is rounded to the whole number every wav2vec2-family model runs at (edge
        frames lose one)."""
        seconds = 10
        silence = np.zeros(SAMPLE_RATE * seconds, dtype=np.float32)
        dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            emissions, _ = model(torch.from_numpy(silence)[None].to(device, dtype=dtype))
        return float(round(emissions.shape[1] / seconds))


ALIGN_BACKENDS: Dict[str, AlignBackend] = {
    backend.name: backend for backend in (HuggingFaceCtcAlign(), TorchaudioAlign())
}


def get_align_backend(name: str) -> AlignBackend:
    """The backend registered as *name* (an align metadata dict's ``"type"``)."""
    try:
        return ALIGN_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Align model of type {name!r} not supported "
            f"(known: {', '.join(sorted(ALIGN_BACKENDS))})"
        ) from None


def _not_found(model_name: str) -> str:
    return (
        f'The chosen align_model "{model_name}" could not be found in huggingface '
        f'(https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)'
    )


def align_backend_for(model_name: str, *, cache_dir=None,
                      local_files_only: bool = False) -> AlignBackend:
    """The backend that runs align model *model_name*: a torchaudio bundle name, else a
    Hugging Face checkpoint whose ``config.json`` says it is a CTC model.

    A hub that cannot be reached is re-raised as it came, so ``load_with_offline_fallback``
    can retry from the cache; any other failure to read the config is "not found".
    """
    import torchaudio
    if model_name in torchaudio.pipelines.__all__:
        return ALIGN_BACKENDS["torchaudio"]

    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir,
                                            local_files_only=local_files_only)
    except Exception as e:
        if _looks_offline(e):
            raise
        logger.error("Error loading model from huggingface (%s): %s", model_name, e)
        raise ValueError(_not_found(model_name)) from e
    if config.model_type not in CTC_MODEL_TYPES:
        raise ValueError(
            f"align_model {model_name!r} is a {config.model_type!r} model; alignment needs a "
            f"CTC model (one of: {', '.join(CTC_MODEL_TYPES)})"
        )
    return ALIGN_BACKENDS["huggingface"]

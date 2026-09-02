"""
Forced Alignment with Whisper
C. Max Bain
"""
import bisect
from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Union, List, Tuple

import numpy as np
import pandas as pd
import torch

from cantocaptions_ai.utils.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram, resolve_device
from cantocaptions_ai.utils.output import PUNKT_LANGUAGES
from cantocaptions_ai.utils.schema import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
    ProgressCallback,
    VadAudioSegment,
    add_note,
    interpolate_nans,
)
from cantocaptions_ai.cantonese.text import DEFAULT_PUNCTUATION, PunctuationConfig, SpotCheck
from cantocaptions_ai.pipeline.align_checks import (
    warn_on_gapped_cues,
    warn_on_silent_starts,
    whole_file_region,
)
from cantocaptions_ai.pipeline.align_profiles import (
    DEFAULT_ALIGN_PROFILE,
    AudioPrimer,
    get_align_profile,
)
from cantocaptions_ai.pipeline.align_vocab import (
    LEVEL_HOMOPHONE,
    VocabRepair,
    filter_spotchecks,
    substitution_notes,
)

from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.output import LANGUAGES_WITHOUT_SPACES
from cantocaptions_ai.utils.model_utils import (
    BatchExecutor,
    check_vram_headroom,
    ensure_hf_model_downloaded,
    guard_model_load,
    resolve_torch_compute_dtype,
)

logger = get_logger(__name__)

# Rough fp32 params + activation footprint for wav2vec2-BERT-cantonese; used only for
# the preflight VRAM-headroom warning, not an exact bound.
_ALIGN_MODEL_VRAM_ESTIMATE_MB = 1200
_ALIGN_REMEDIATION = "pass --no_align to skip alignment, or free VRAM used by other processes/stages"

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi-vlsp2020',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "yue": "alvanlii/wav2vec2-BERT-cantonese"
}

# https://huggingface.co/scottykwok/wav2vec2-large-xlsr-cantonese xlsr-53 + common voice
# scottykwok/wav2vec2-large-xlsr-cantonese xlsr-53 + common voice + more training
# wcfr/wav2vec2-conformer-rel-pos-base-cantonese
# alvanlii/wav2vec2-BERT-cantonese


# --- Dataclasses ---

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


# --- Low-level CTC alignment ---
# source: https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

def get_trellis(emission, tokens, blank_id=0, free_end: bool = False):
    """Forced-alignment trellis over *emission* for *tokens*.

    ``free_end`` must match the value later passed to ``backtrack``. It drops the
    ``trellis[-num_tokens:, 0] = +inf`` sentinel, which exists purely to terminate the
    backward walk of a *forced* alignment. That +inf propagates diagonally through the
    recurrence, so by the final row it has flooded every token column but the last -- which
    would leave a free-end search with only one column to "choose", silently turning it back
    into a forced walk. A partial walk does not need the sentinel: it enters at a column the
    forward pass actually reached, so it has the frames to walk back to the start.
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    if not free_end:
        trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0, free_end: bool = False):
    """Walk the trellis back to token 0, returning the best monotonic path.

    Without ``free_end`` (the default) the path must consume *every* token: it enters at the
    last token column and finds the frame that best completes it. That is right when the
    text is known to correspond to exactly this audio.

    With ``free_end=True`` the path instead enters at the *last frame* and takes whichever
    token column scores best there, so it consumes only as many tokens as the audio
    actually explains. --realign uses this to slide a window over a long recording and ask
    "how much of the remaining transcript fits in here?" without knowing the answer up
    front; see pipeline/realign.py. The trellis must have been built with the same
    ``free_end``, and is checked for it rather than being allowed to answer wrongly.
    """
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    if free_end:
        t_start = trellis.size(0) - 1
        row = trellis[t_start, 1:]
        if bool(torch.isposinf(row).any()):
            raise ValueError(
                "backtrack(free_end=True) needs a trellis built with get_trellis("
                "free_end=True); the forced-walk sentinel makes every column but the last "
                "infinite, which would silently reduce the search to a forced alignment."
            )
        # Column 0 is the start state -- entering there means nothing was consumed.
        j = 1 + torch.argmax(row).item()
    else:
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[T-1]` is the emission at time frame `T` of trellis dimension.
        # Score for token staying the same from time frame T-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from J-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None

    return path[::-1]


def get_score(emission, tokens, blank_id=0):
    """Return average score for a token sequence against emissions."""
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    if path is None:
        return float("-inf")
    return sum(p.score for p in path) / len(path)


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


# --- Private helpers ---

def _run_model_inference(
    model: torch.nn.Module,
    model_type: str,
    audio: torch.Tensor,
    bert_processor,
    device: str,
    lengths=None,
) -> torch.Tensor:
    """Single forward pass returning log-softmax emissions."""
    model_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        if model_type == "torchaudio":
            emissions, _ = model(audio.to(device, dtype=model_dtype), lengths=lengths)
        elif model_type == "huggingface":
            if bert_processor is not None:
                features = bert_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                emissions = model(features["input_features"].to(device, dtype=model_dtype)).logits
            else:
                emissions = model(audio.to(device, dtype=model_dtype)).logits
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")
        return torch.log_softmax(emissions, dim=-1)


def _get_blank_id(model_dictionary: dict) -> int:
    return next((code for char, code in model_dictionary.items() if char in ('[pad]', '<pad>')), 0)


def _get_sentence_spans(text: str, model_lang: str, punctuation: PunctuationConfig) -> List[Tuple[int, int]]:
    """Split text into sentence span tuples using language-appropriate tokenization."""
    if model_lang in ['yue', 'zh']:
        return punctuation.sentence_spans(text)

    import nltk
    from nltk.data import load as nltk_load
    punkt_lang = PUNKT_LANGUAGES.get(model_lang, 'english')
    try:
        sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
    return list(sentence_splitter.span_tokenize(text))


def _preprocess_segment(
    text: str, model_lang: str, model_dictionary: dict,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    spans: Optional[List[Tuple[int, int]]] = None,
) -> SegmentData:
    """Clean text and produce per-segment alignment metadata.

    ``spans`` lets a caller declare the cue boundaries as inclusive (start, end) index
    pairs into ``text`` instead of having them derived from punctuation. Alignment treats
    them as opaque, so the only requirement is that they index ``text``. Used by --realign,
    whose input arrives pre-split into cues; see utils/schema.py SingleSegment.cue_spans.
    """
    num_leading = len(text) - len(text.lstrip())
    num_trailing = len(text) - len(text.rstrip())

    per_word = text.split(" ") if model_lang not in LANGUAGES_WITHOUT_SPACES else text

    clean_char, clean_cdx = [], []
    for cdx, char in enumerate(text):
        char_ = char.lower()
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            char_ = char_.replace(" ", "|")
        if cdx < num_leading or cdx > len(text) - num_trailing - 1:
            continue
        if char_ in model_dictionary or char_ in punctuation.split_chars:
            clean_char.append(char_)
            clean_cdx.append(cdx)

    clean_wdx = [
        wdx for wdx, wrd in enumerate(per_word)
        if any(c in model_dictionary for c in wrd.lower())
    ]

    return {
        "clean_char": clean_char,
        "clean_cdx": clean_cdx,
        "clean_wdx": clean_wdx,
        "sentence_spans": (
            list(spans) if spans is not None
            else _get_sentence_spans(text, model_lang, punctuation)
        ),
    }

def _preprocess_transcript(
    transcript: List[SingleSegment],
    model_lang: str,
    model_dictionary: dict,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    print_progress: bool = False,
) -> dict:
    """First pass: build SegmentData for every transcript segment."""
    total = len(transcript)
    segment_data = {}
    for sdx, segment in enumerate(transcript):
        segment_data[sdx] = _preprocess_segment(
            segment["text"], model_lang, model_dictionary, punctuation,
            spans=segment.get("cue_spans"),
        )
    return segment_data


_TIMESTAMP_TOLERANCE_S = 0.005


def _find_vad_segment_idx(vad_segments: List[VadAudioSegment], t: float) -> Optional[int]:
    # Half-open [start - tol, end) so that a timestamp exactly at a segment boundary
    # belongs to the next segment rather than the one that just ended.
    for i, seg in enumerate(vad_segments):
        if seg["start"] - _TIMESTAMP_TOLERANCE_S <= t < seg["end"]:
            return i
    # Fallback: t is at or just past the end of the last segment.
    if vad_segments and t <= vad_segments[-1]["end"] + _TIMESTAMP_TOLERANCE_S:
        return len(vad_segments) - 1
    return None


def _compute_vad_emissions_sequential(
    vad_segments: List[VadAudioSegment],
    model: torch.nn.Module,
    model_type: str,
    bert_processor,
    device: str,
) -> List[Tuple[torch.Tensor, float]]:
    """Run inference on each VAD segment one at a time.

    Fallback used for model types this pipeline doesn't actually exercise in
    practice (torchaudio bundles, or a plain HF wav2vec2 model with no BERT
    feature-extractor) — see _compute_vad_emissions_batched for the real path.
    """
    results = []
    for vad_seg in vad_segments:
        seg_audio = vad_seg["audio"]
        if not torch.is_tensor(seg_audio):
            seg_audio = torch.from_numpy(seg_audio)
        if len(seg_audio.shape) == 1:
            seg_audio = seg_audio.unsqueeze(0)

        emissions = _run_model_inference(model, model_type, seg_audio, bert_processor, device)
        emission = emissions[0].cpu().detach()
        vad_duration = vad_seg["end"] - vad_seg["start"]
        frame_rate = emission.size(0) / vad_duration if vad_duration > 0 else 0.0
        results.append((emission, frame_rate))
    return results


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
    bert_processor,
    device: str,
    batch_size: int,
    vram_checks: bool = True,
    primer: Optional["AudioPrimer"] = None,
) -> List[Tuple[torch.Tensor, float]]:
    """Batch VAD segments through the HF Wav2Vec2-BERT model via BatchExecutor.

    ``primer`` (from the align model's profile, ``None`` for a model without one) prepends
    left context to each segment before the encoder sees it, and its frames are discarded
    here so nothing downstream knows it existed — see ``align_profiles.TailPrimer`` for why
    the current model needs it. The prefix length is deliberately *not* asked of the primer:
    the encoder's own output length for the **unprimed** audio is measured and that many
    frames are kept from the end, which is exact whatever the primer prepended and keeps a
    future primer free to change shape. That costs one extra feature-extraction pass per
    batch (cheap next to the forward) and is worth it — estimating the offset from duration
    instead would be a frame out often enough to reintroduce the artifact it removes.

    bert_processor's feature extractor pads each batch to its own longest segment and returns
    an attention_mask, whose per-row sum is the segment's real length **in input feature
    frames**. That is not the same unit as the CTC emission: alvanlii/wav2vec2-BERT-cantonese
    sets add_adapter=True with adapter_stride=2, so the emission runs at half the feature rate
    (~25 fps vs ~50 fps). The mask length is therefore converted through the model's own
    _get_feat_extract_output_lengths before it is used to trim each row's emission back to its
    real (unpadded) length — that helper applies the adapter convs and is an identity when a
    model has no adapter, so this stays correct for a different align model.

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

    def _emission_lens(wavs) -> torch.Tensor:
        features = bert_processor(
            wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True,
        )
        # Feature frames -> emission frames (see docstring); identity without an adapter.
        return model._get_feat_extract_output_lengths(features["attention_mask"].sum(dim=-1))

    def infer_fn(batch: List[int]) -> None:
        wavs = [vad_segments[i]["audio"] for i in batch]
        model_inputs = [primer(w, SAMPLE_RATE) for w in wavs] if primer is not None else wavs
        with torch.inference_mode():
            features = bert_processor(
                model_inputs, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = features["input_features"].to(device, dtype=model_dtype)
            attention_mask = features["attention_mask"].to(device)
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
            emission = emissions[row, real_len - keep:real_len, :].cpu().detach()
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


def _compute_vad_emissions(
    vad_segments: List[VadAudioSegment],
    model: torch.nn.Module,
    model_type: str,
    bert_processor,
    device: str,
    batch_size: int = 4,
    vram_checks: bool = True,
    primer: Optional["AudioPrimer"] = None,
) -> List[Tuple[torch.Tensor, float]]:
    """Run inference on each full VAD segment. Returns (log_softmax_emission, frame_rate) per segment.

    Logs before/after regardless of which path runs below, so a slow pass (a file
    with many/long VAD segments) is visibly explained rather than looking like a
    hang — this ran with no progress feedback at all before batching was added.

    ``primer`` reaches only the batched path. The sequential fallback exists for model
    types this pipeline doesn't exercise (torchaudio bundles, plain HF wav2vec2), none of
    which carry a profile primer today; priming there would need the same exact
    frame-length bookkeeping for no current caller, so it warns instead of half-doing it.
    """
    if not vad_segments:
        return []
    start = time.perf_counter()
    logger.info("Computing alignment emissions for %d VAD segments...", len(vad_segments))
    if model_type == "huggingface" and bert_processor is not None:
        results = _compute_vad_emissions_batched(
            vad_segments, model, bert_processor, device, batch_size,
            vram_checks=vram_checks, primer=primer,
        )
    else:
        if primer is not None:
            logger.warning(
                "Align model profile configures an audio primer, but this model runs the "
                "sequential path, which does not apply it. First-character timings may be "
                "pinned to each segment's start."
            )
        results = _compute_vad_emissions_sequential(vad_segments, model, model_type, bert_processor, device)
    logger.info("Alignment emissions computed in %.1fs", time.perf_counter() - start)
    return results


def compute_vad_emissions(vad_segments, model, model_type, bert_processor, device, batch_size: int = 4, vram_checks: bool = True, primer=None):
    """Public wrapper around _compute_vad_emissions for use by the retime pipeline."""
    return _compute_vad_emissions(vad_segments, model, model_type, bert_processor, device, batch_size, vram_checks=vram_checks, primer=primer)


class EmissionTimeline:
    """The file's emissions as one timeline, sliceable by time across chunk boundaries.

    The chunks are contiguous and gap-free (``Vad.cover_chunks``), so a slice spanning a join
    is exactly as valid as one inside a single chunk: the tear exists because the encoder
    cannot take a whole film at once, not because anything changes in the audio there.

    That matters more than it sounds. Before this, a transcript segment could only be aligned
    against the *one* chunk holding its start (``alignment._get_emission_for_segment``), so
    ``build_align_input`` had to re-cut the file to keep every line inside a single chunk --
    and a coarse guess a second or two out could then put the boundary *in front of* the line
    it was meant to contain, handing forced alignment a chunk that does not hold the line at
    all. On the Police Story 2 head that produced a 0.2 s cue at score 0.000 for a line whose
    speech had ended 1.5 s before the chunk began. Measured across the fixtures, 22-44% of the
    spans this module wants to align in one piece cross a chunk join, so the old constraint
    was not a corner case.

    Frame times come from each chunk's *own* frame count rather than from one global rate:
    ``_compute_vad_emissions_batched`` reports up to 2% variation between chunks, and a single
    rate would smear a cross-join slice at every boundary.

    Emissions are held as float16 and upcast per slice. Two hours is then about 1 GB against
    the 2 GB float32 copy ``align()`` holds today, and the same copy serves both the placement
    search and the final alignment -- one encoder pass over the file instead of two.
    """

    def __init__(
        self,
        vad_segments: Sequence[VadAudioSegment],
        compute_fn: Callable[[List[VadAudioSegment]], List[Tuple[torch.Tensor, float]]],
    ):
        self._segments = list(vad_segments)
        self._compute = compute_fn
        self._emissions: List[Optional[np.ndarray]] = [None] * len(self._segments)
        self._times: List[Optional[np.ndarray]] = [None] * len(self._segments)
        self._starts = [float(s["start"]) for s in self._segments]
        self._ends = [float(s["end"]) for s in self._segments]
        self.computed = 0

    @property
    def file_start(self) -> float:
        return self._starts[0] if self._starts else 0.0

    @property
    def file_end(self) -> float:
        return self._ends[-1] if self._ends else 0.0

    def chunk_range(self, t0: float, t1: float) -> Tuple[int, int]:
        """The half-open range of chunks covering [t0, t1]."""
        lo = max(0, bisect.bisect_right(self._starts, t0) - 1)
        hi = bisect.bisect_right(self._starts, t1)
        return lo, max(hi, lo + 1)

    def ensure(self, lo: int, hi: int) -> None:
        missing = [i for i in range(lo, min(hi, len(self._segments)))
                   if self._emissions[i] is None]
        if not missing:
            return
        results = self._compute([self._segments[i] for i in missing])
        for i, (emission, _rate) in zip(missing, results):
            arr = emission.detach().cpu().numpy().astype(np.float16, copy=False)
            self._emissions[i] = arr
            n = arr.shape[0]
            step = (self._ends[i] - self._starts[i]) / n if n else 0.0
            self._times[i] = self._starts[i] + np.arange(n, dtype=np.float64) * step
        self.computed += len(missing)

    def slice(self, t0: float, t1: float) -> Tuple[torch.Tensor, np.ndarray]:
        """(emission[frames, vocab] as float32, absolute start time of each frame)."""
        lo, hi = self.chunk_range(t0, t1)
        self.ensure(lo, hi)
        ems, ts = [], []
        for i in range(lo, min(hi, len(self._segments))):
            times = self._times[i]
            if times is None or not len(times):
                continue
            a = int(np.searchsorted(times, t0, side="left"))
            b = int(np.searchsorted(times, t1, side="right"))
            if b > a:
                ems.append(self._emissions[i][a:b])
                ts.append(times[a:b])
        if not ems:
            # A request narrower than one frame. Give back the frame covering t0 so no caller
            # has to special-case an empty emission.
            i = min(max(lo, 0), len(self._segments) - 1)
            self.ensure(i, i + 1)
            times = self._times[i]
            if times is None or not len(times):
                return torch.zeros((0, 0)), np.empty(0, dtype=np.float64)
            k = min(max(int(np.searchsorted(times, t0, side="right")) - 1, 0), len(times) - 1)
            ems, ts = [self._emissions[i][k:k + 1]], [times[k:k + 1]]
        emission = ems[0] if len(ems) == 1 else np.concatenate(ems)
        frame_times = ts[0] if len(ts) == 1 else np.concatenate(ts)
        return torch.from_numpy(np.asarray(emission, dtype=np.float32)), frame_times

    @classmethod
    def from_computed(cls, vad_segments, results):
        """Wrap emissions that have already been computed, so nothing is encoded twice."""
        by_start = {float(s["start"]): r for s, r in zip(vad_segments, results)}
        timeline = cls(vad_segments, lambda segs: [by_start[float(s["start"])] for s in segs])
        timeline.ensure(0, len(vad_segments))
        return timeline


def _get_emission_for_segment(
    t1: float,
    t2: float,
    audio,
    vad_segments: Optional[List[VadAudioSegment]],
    vad_seg_emissions: Optional[List[Tuple[torch.Tensor, float]]],
    model: torch.nn.Module,
    model_type: str,
    bert_processor,
    device: str,
    timeline=None,
) -> Optional[Tuple[torch.Tensor, Optional[np.ndarray]]]:
    """Return (emission, frame times) for one segment, or None if no audio matches.

    With a ``timeline`` (see ``realign.EmissionTimeline``) the emission is cut straight out of
    the file's timeline and may span chunk joins, and the exact absolute time of every frame
    comes back with it. Without one the old behaviour stands: the segment is sliced out of the
    single VAD chunk containing ``t1``, and the caller derives times from the segment's own
    duration. Frame times are ``None`` in that case.
    """
    if timeline is not None:
        emission, frame_times = timeline.slice(t1, t2)
        return (emission, frame_times) if emission.size(0) else None
    if vad_seg_emissions is not None:
        vad_idx = _find_vad_segment_idx(vad_segments, t1)
        if vad_idx is None:
            return None
        vad_seg = vad_segments[vad_idx]
        full_emission, frame_rate = vad_seg_emissions[vad_idx]
        t1_local = t1 - vad_seg["start"]
        t2_local = t2 - vad_seg["start"]
        e1 = int(t1_local * frame_rate)
        e2 = max(int(t2_local * frame_rate), e1 + 1)
        return full_emission[e1:e2, :], None

    f1 = int(t1 * SAMPLE_RATE)
    f2 = int(t2 * SAMPLE_RATE)
    waveform_segment = audio[:, f1:f2]
    if waveform_segment.shape[-1] < 400:
        lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
        waveform_segment = torch.nn.functional.pad(
            waveform_segment, (0, 400 - waveform_segment.shape[-1])
        )
    else:
        lengths = None
    emissions = _run_model_inference(model, model_type, waveform_segment, bert_processor, device, lengths=lengths)
    return emissions[0].cpu().detach(), None


# How long a character's aligned span may run before it is read as a dwell rather than a
# character.
#
# CTC is peaky: a character fires on a frame or two and the path then sits on it, emitting
# blank, until whatever comes next arrives. merge_repeats reports that whole wait as the
# character's span, and where the wait is long the span swallows it. Measured on Police
# Story 2:
#
#     我@330.26-344.19  覺@344.19-344.35  得@344.35-344.47  ...
#
# Thirteen seconds on the first character and a tenth of a second on each one after it, which
# put that subtitle on screen fourteen seconds before anyone spoke. The same shape at the far
# edge gave 「唔該警察叔叔」 an 11.8 s cue whose last nine seconds are a pause.
#
# No guard that compares *consecutive* characters can see this -- there is no gap between
# characters anywhere in those cues -- and neither can an energy envelope, because the pause
# is room tone a dozen dB under the dialogue rather than actual silence. The emission says it
# plainly: within the dwell, the character's own token peaks on one frame and is negligible on
# the rest.
#
# A real syllable, even drawn out, does not hold the CTC path for a second.
MAX_CHAR_DWELL_SECONDS = 1.0


def _reseat_dwelling_chars(char_segments, emission, tokens, blank_id, seconds_per_frame):
    """Move any character that merely waited onto the frame its own token peaks at.

    Its replacement span is the median length of the segment's other characters, so the
    character keeps a plausible duration instead of collapsing to a single frame, and it is
    clipped against the next character so the sequence stays ordered. See
    MAX_CHAR_DWELL_SECONDS.
    """
    if len(char_segments) != len(tokens) or seconds_per_frame <= 0:
        return 0
    limit = max(int(round(MAX_CHAR_DWELL_SECONDS / seconds_per_frame)), 2)
    lengths = [cs.end - cs.start for cs in char_segments]
    typical = max(int(np.median(lengths)), 1)
    moved = 0
    for idx, cs in enumerate(char_segments):
        if cs.end - cs.start <= limit or tokens[idx] == blank_id:
            continue
        window = emission[cs.start:cs.end, tokens[idx]]
        if window.numel() == 0:
            continue
        peak = cs.start + int(torch.argmax(window).item())
        ceiling = char_segments[idx + 1].start if idx + 1 < len(char_segments) else cs.end
        cs.start = peak
        cs.end = max(min(peak + typical, max(ceiling, peak + 1), cs.end), peak + 1)
        moved += 1
    if moved:
        logger.debug(
            "Re-seated %d character(s) that held the alignment path for more than %.1fs onto "
            "the frame their own token peaks at", moved, MAX_CHAR_DWELL_SECONDS,
        )
    return moved


def _align_segment(
    segment: SingleSegment,
    seg_data: SegmentData,
    emission: torch.Tensor,
    model_dictionary: dict,
    model_lang: str,
    blank_id: int,
    spacing_char_id: int,
    t1: float,
    t2: float,
    interpolate_method: str,
    return_char_alignments: bool,
    spotchecks: Mapping[str, SpotCheck],
    punctuation: PunctuationConfig,
    frame_times: Optional[np.ndarray] = None,
) -> List[dict]:
    """Align one transcript segment against its emission, returning subsegment dicts.

    ``frame_times`` gives the absolute time of each emission frame. When present it is used
    verbatim, which is what makes an emission spanning several VAD chunks correct: those
    chunks can differ in frame rate by a couple of percent, and stretching one rate across the
    join would smear every timestamp after it. Without it the old assumption holds -- the
    emission covers exactly [t1, t2] at a constant rate.
    """
    text = segment["text"]
    avg_logprob = segment.get("avg_logprob")

    base_seg: SingleAlignedSegment = {"start": t1, "end": t2, "text": text, "words": [], "chars": None}
    if avg_logprob is not None:
        base_seg["avg_logprob"] = avg_logprob
    if return_char_alignments:
        base_seg["chars"] = []

    if len(seg_data["clean_char"]) == 0:
        logger.warning(f'Failed to align segment ("{text}"): no characters in this segment found in model dictionary, resorting to original')
        return [base_seg]

    text_clean = "".join(seg_data["clean_char"])

    # Replace punctuation with spacing token to better align breaks at sentence ends
    split_chars = punctuation.split_chars
    tokens = [model_dictionary[c] if c not in split_chars else spacing_char_id for c in text_clean]

    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)

    # Spot checks: for each char with an interchangeable candidate set (per the model's
    # profile), pick the candidate whose acoustic log-prob at this char's aligned frame is
    # highest, plus any per-candidate bias weight. Empty `spotchecks` (the default for a
    # model whose output already uses the intended particles) makes this loop a no-op.
    if spotchecks and path is not None:
        logger.debug("Checking particle candidates for text: '%s'.", text_clean)

        lowercase_text = text.lower()
        t_i = 0
        for p_i, p in enumerate(text_clean):
            # Use t_i to mark the position in the base "text" var. Keep this updated to avoid conflicts.
            # TODO: roll text, text_clean, and seg_data["clean_char"] all up into a single dynamic type
            t_i = t_i + lowercase_text[t_i:].index(p) + 1

            sc = spotchecks.get(p)
            if sc is None or len(sc.candidates) <= 1:
                continue

            path_i = min(x.time_index for x in path if x.token_index == p_i)

            max_score = -math.inf
            best_candidate = None
            for c in sc.candidates:
                c_token = model_dictionary.get(c)
                if c_token is None:
                    logger.warning("Spot-check candidate %r absent from align model vocab; skipping.", c)
                    continue
                score = emission[path_i, c_token].item() + sc.weights.get(c, 0.0)
                if score > max_score:
                    best_candidate = c
                    max_score = score

            if best_candidate is None:
                continue

            if best_candidate != p:
                text_clean = text_clean[:p_i] + best_candidate + text_clean[p_i + 1:]
                text = text[:t_i - 1] + best_candidate + text[t_i:] # messy :(

            logger.debug("Best candidate for char '%d' ('%s'): '%s' (score %.3f).", p_i, p, best_candidate, max_score)

    seg_data["clean_char"] = [c for c in text_clean]

    if path is None:
        logger.warning(f'Failed to align segment ("{text}"): backtrack failed, resorting to original')
        return [base_seg]

    seconds_per_frame = (
        float(frame_times[1] - frame_times[0])
        if frame_times is not None and len(frame_times) > 1
        else (t2 - t1) / max(trellis.size(0) - 1, 1)
    )
    char_segments = merge_repeats(path, text_clean)
    _reseat_dwelling_chars(char_segments, emission, tokens, blank_id, seconds_per_frame)
    if frame_times is not None and len(frame_times):
        # merge_repeats reports an *exclusive* end, so the map needs one entry past the last
        # frame; extend by the final step rather than clamping, which would collapse the last
        # character to zero length.
        step = float(frame_times[-1] - frame_times[-2]) if len(frame_times) > 1 else 0.04
        edges = np.append(np.asarray(frame_times, dtype=np.float64), frame_times[-1] + step)
        last = len(edges) - 1

        def _at(frame: float) -> float:
            return float(edges[min(max(int(round(frame)), 0), last)])
    else:
        ratio = (t2 - t1) / max(trellis.size(0) - 1, 1)

        def _at(frame: float) -> float:
            return frame * ratio + t1

    char_segments_arr = []
    word_idx = 0
    for cdx, char in enumerate(text):
        start, end, score = None, None, None
        if cdx in seg_data["clean_cdx"]:
            char_seg = char_segments[seg_data["clean_cdx"].index(cdx)]
            start = round(_at(char_seg.start), 3)
            end = round(_at(char_seg.end), 3)
            score = round(char_seg.score, 3)
        char_segments_arr.append({"char": char, "start": start, "end": end, "score": score, "word-idx": word_idx})
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            word_idx += 1
        elif cdx == len(text) - 1 or text[cdx + 1] == " ":
            word_idx += 1

    char_segments_arr = pd.DataFrame(char_segments_arr)
    char_segments_arr["sentence-idx"] = None
    aligned_subsegments = []

    for sdx2, (sstart, send) in enumerate(seg_data["sentence_spans"]):
        mask = (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)
        curr_chars = char_segments_arr.loc[mask]
        char_segments_arr.loc[mask, "sentence-idx"] = sdx2

        end_chars = curr_chars[curr_chars["char"] != ' ']
        if len(end_chars) == 0:
            continue

        sentence_text = text[sstart:send + 1]
        sentence_start = curr_chars["start"].min()
        last_char = end_chars.iloc[-1]
        sentence_end = end_chars["end"].max()
        # Sentences ending on punctuation get their end time released (extended) later,
        # in align(), once the position relative to *all* subsegments in the file
        # (not just this transcript segment) is known — see release_from below.
        release_from = last_char["start"] if last_char["char"] in split_chars else None

        sentence_words = []
        for word_idx in curr_chars["word-idx"].unique():
            word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
            word_text = "".join(word_chars["char"].tolist()).strip()
            if not word_text:
                continue
            word_chars = word_chars[word_chars["char"] != " "]
            word_start = word_chars["start"].min()
            word_end = word_chars["end"].max()
            word_score = round(word_chars["score"].mean(), 3)
            word_segment = {"word": word_text}
            if not np.isnan(word_start):
                word_segment["start"] = word_start
            if not np.isnan(word_end):
                word_segment["end"] = word_end
            if not np.isnan(word_score):
                word_segment["score"] = word_score
            sentence_words.append(word_segment)

        subsegment = {
            "text": sentence_text,
            "start": sentence_start,
            "end": sentence_end,
            "words": sentence_words,
            "release_from": release_from,
        }
        # A caller that declared its own cue spans may also say why a cue is doubtful
        # (--realign). Carry it onto the finished cue: the reason is known before alignment
        # runs and there is nothing downstream that could reconstruct it.
        cue_reasons = segment.get("cue_reasons")
        if cue_reasons and sdx2 < len(cue_reasons) and cue_reasons[sdx2]:
            subsegment["realign_reason"] = cue_reasons[sdx2]
        if avg_logprob is not None:
            subsegment["avg_logprob"] = avg_logprob
        aligned_subsegments.append(subsegment)

        if return_char_alignments:
            chars_out = curr_chars[["char", "start", "end", "score"]].copy()
            chars_out.fillna(-1, inplace=True)
            aligned_subsegments[-1]["chars"] = [
                {k: v for k, v in row.items() if v != -1}
                for row in chars_out.to_dict("records")
            ]

    aligned_subsegments = pd.DataFrame(aligned_subsegments)
    aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
    aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)

    # Concatenate sentences with same timestamps
    if "realign_reason" not in aligned_subsegments.columns:
        aligned_subsegments["realign_reason"] = None
    agg_dict = {"text": " ".join, "words": "sum", "release_from": "first",
                "realign_reason": "first"}
    if model_lang in LANGUAGES_WITHOUT_SPACES:
        agg_dict["text"] = "".join
    if return_char_alignments:
        agg_dict["chars"] = "sum"
    if avg_logprob is not None:
        agg_dict["avg_logprob"] = "first"

    aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
    records = aligned_subsegments.to_dict("records")
    for row in records:
        # A column of Nones comes back from groupby as NaN, and NaN is *truthy* -- test the
        # type, not the value, or every cue in the file ends up carrying a float "reason".
        if not isinstance(row.get("realign_reason"), str) or not row["realign_reason"]:
            row.pop("realign_reason", None)
    return records


# --- Public functions ---

def load_align_model(
    language_code: str, device: str, device_index: int = 0, model_name: Optional[str] = None,
    model_dir=None, model_cache_only: bool = False, compute_type: str = "float32",
    vram_checks: bool = True,
    char_substitution: str = LEVEL_HOMOPHONE,
    substitution_overrides: Optional[Mapping[str, str]] = None,
):
    """Load the phoneme-alignment model.

    compute_type="float16" halves weight VRAM but the model is otherwise loaded and
    invoked exactly like float32 (no autocast) — inputs are cast to match in
    _run_model_inference/_compute_vad_emissions_batched, and this is deliberately
    opt-in with float32 as the default since it can measurably affect forced-alignment
    accuracy.
    """
    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            logger.error(
                f"No default alignment model for language: {language_code}. "
                f"Please find a wav2vec2.0 model finetuned on this language at https://huggingface.co/models, "
                f"then pass the model name via --align_model [MODEL_NAME]"
            )
            raise ValueError(f"No default align-model for language: {language_code}")

    device = resolve_device(device, device_index)
    dtype = resolve_torch_compute_dtype(compute_type, device, "align")

    import torchaudio
    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        check_vram_headroom("Alignment model load", device, _ALIGN_MODEL_VRAM_ESTIMATE_MB, _ALIGN_REMEDIATION, vram_checks=vram_checks)
        align_model = guard_model_load(
            "alignment", _ALIGN_REMEDIATION,
            lambda: bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device, dtype=dtype),
        )
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
        is_bert = 'wav2vec2-BERT' in model_name
        ProcessorClass = Wav2Vec2BertProcessor if is_bert else Wav2Vec2Processor
        ModelClass = Wav2Vec2BertForCTC if is_bert else Wav2Vec2ForCTC
        model_flavor = "wav2vec2-BERT" if is_bert else "wav2vec2.0"
        try:
            ensure_hf_model_downloaded(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
        except Exception as e:
            logger.warning("Could not download %r: %s — using cached version if available.", model_name, e)
        try:
            processor = ProcessorClass.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
            align_model = ModelClass.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
        except Exception as e:
            logger.error("Error loading model from huggingface (%s): %s", model_name, e)
            raise ValueError(
                f'The chosen align_model "{model_name}" could not be found in huggingface '
                f'(https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)'
            )
        pipeline_type = "huggingface"
        check_vram_headroom("Alignment model load", device, _ALIGN_MODEL_VRAM_ESTIMATE_MB, _ALIGN_REMEDIATION, vram_checks=vram_checks)
        align_model = guard_model_load("alignment", _ALIGN_REMEDIATION, lambda: align_model.to(device, dtype=dtype))
        align_dictionary = {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()}

    align_metadata = {
        "language": language_code,
        "dictionary": align_dictionary,
        "type": pipeline_type,
        # Built here, next to the dictionary it edits, so every stage that tokenises text
        # against this model shares one vocabulary. Resolves nothing until a caller hands it
        # some text -- see align_vocab.VocabRepair.
        "vocab_repair": VocabRepair(
            align_dictionary, char_substitution, substitution_overrides,
        ),
        # Resolved once here rather than in align(), which then has no idea which model it
        # is holding. Unknown models get the all-no-op default.
        "profile": get_align_profile(model_name),
    }
    return align_model, align_metadata


def load_bert_processor(model_dir=None, model_cache_only: bool = False):
    """Load the Wav2Vec2-BERT processor used for particle disambiguation during alignment.

    Shared by both the --retime and normal alignment paths in transcribe.py, which
    otherwise each held their own unlogged, cache_dir/local_files_only-blind
    from_pretrained call for the same repo.
    """
    from transformers import Wav2Vec2BertProcessor

    repo_id = "alvanlii/wav2vec2-BERT-cantonese"
    try:
        ensure_hf_model_downloaded(repo_id, cache_dir=model_dir, local_files_only=model_cache_only)
    except Exception as e:
        logger.warning("Could not download %r: %s — using cached version if available.", repo_id, e)
    return Wav2Vec2BertProcessor.from_pretrained(repo_id, cache_dir=model_dir, local_files_only=model_cache_only)


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor, List[VadAudioSegment]],
    device: str,
    bert_processor=None,
    align_padding: float = 0.04,
    align_release: float = 0.4,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    progress_callback: ProgressCallback = None,
    batch_size: int = 4,
    vram_checks: bool = True,
    spotchecks: Optional[Mapping[str, SpotCheck]] = None,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    timeline=None,
) -> AlignedTranscriptionResult:
    """Align phoneme recognition predictions to known transcription.

    ``spotchecks`` and ``punctuation`` come from the ASR model's profile (see
    ``pipeline/model_profiles.py``); their defaults (no spot checks, standard
    punctuation) keep alignment independent of any specific model.

    ``timeline`` (``realign.EmissionTimeline``) replaces the per-chunk emission set: segments
    are then cut out of one continuous timeline, so a segment may span chunk joins and the
    encoder is not run a second time over audio the caller has already encoded.
    """
    spotchecks = spotchecks or {}

    # --- Audio setup ---
    vad_segments: Optional[List[VadAudioSegment]] = None
    if isinstance(audio, list):
        vad_segments = audio
        MAX_DURATION = max(seg["end"] for seg in vad_segments) if vad_segments else 0.0
    else:
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]
    # .get so hand-built metadata dicts (tests, callers predating align_profiles) still work.
    profile = align_model_metadata.get("profile") or DEFAULT_ALIGN_PROFILE
    blank_id = _get_blank_id(model_dictionary)
    spacing_char_id = blank_id # model_dictionary['！']

    # One timeline for the file whichever way we got here. --realign's acoustic anchor hands
    # one in (already populated, so the encoder does not run twice over the same audio);
    # otherwise the emissions are computed here and wrapped. Either way a transcript segment
    # can be cut across chunk joins, which is what stops a boundary landing in front of the
    # line it was meant to contain.
    if timeline is None and vad_segments is not None:
        timeline = EmissionTimeline.from_computed(
            vad_segments,
            _compute_vad_emissions(
                vad_segments, model, model_type, bert_processor, device, batch_size,
                vram_checks=vram_checks, primer=profile.primer,
            ),
        )
    vad_seg_emissions = None

    # --- Preprocess transcript ---
    transcript = list(transcript)
    # Before anything reads the dictionary: give it a token for the characters it has none
    # for, so _preprocess_segment keeps them instead of dropping them. Idempotent, so under
    # --realign (where the coarse search already ran this over the same transcript against
    # the same dictionary) this is a no-op and the two passes cannot disagree.
    repair = align_model_metadata.get("vocab_repair")
    if repair is not None:
        repair.augment(segment.get("text", "") for segment in transcript)
        # A substituted character's token is some homophone's, so it cannot be asked which
        # of two particles the audio supports. Never fires for the shipped profiles.
        spotchecks = filter_spotchecks(spotchecks, repair.substitutions)
    segment_data = _preprocess_transcript(transcript, model_lang, model_dictionary, punctuation, print_progress)

    # --- Align each segment ---
    aligned_segments: List[SingleAlignedSegment] = []

    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")

        base_seg: SingleAlignedSegment = {"start": t1, "end": t2, "text": text, "words": [], "chars": None}
        if avg_logprob is not None:
            base_seg["avg_logprob"] = avg_logprob
        if return_char_alignments:
            base_seg["chars"] = []

        if t1 >= MAX_DURATION:
            logger.warning(f'Failed to align segment ("{text}"): original start time longer than audio duration, skipping')
            aligned_segments.append(base_seg)
            continue

        found = _get_emission_for_segment(
            t1, t2, audio, vad_segments, vad_seg_emissions,
            model, model_type, bert_processor, device, timeline=timeline,
        )
        emission, frame_times = found if found is not None else (None, None)
        if emission is None:
            logger.warning(f'Failed to align segment ("{text}"): no VAD segment found for start time {t1}, skipping')
            aligned_segments.append(base_seg)
            continue

        subsegments = _align_segment(
            segment, segment_data[sdx], emission,
            model_dictionary, model_lang, blank_id, spacing_char_id,
            t1, t2, interpolate_method, return_char_alignments,
            spotchecks, punctuation, frame_times=frame_times,
        )
        aligned_segments += subsegments

        if progress_callback is not None:
            progress_callback.advance(1)

    # --- Release punctuation-terminated ends, then trim overlaps against the next
    # subsegment's start. Done once over the whole file (not per transcript segment)
    # so that a released end can't collide with the first subsegment of the next
    # VAD segment, which _align_segment has no visibility into.
    if aligned_segments:
        starts = pd.Series([seg["start"] for seg in aligned_segments], dtype="float64")
        ends = pd.Series([seg["end"] for seg in aligned_segments], dtype="float64")
        release_froms = pd.Series(
            [seg.pop("release_from", None) for seg in aligned_segments], dtype="float64"
        )

        release_mask = release_froms.notna()
        ends[release_mask] = (release_froms[release_mask] + align_release).round(3)

        next_starts = starts.shift(-1)
        overlap = ends > next_starts - align_padding
        ends[overlap] = (next_starts[overlap] - align_padding).round(3)

        for seg, new_end in zip(aligned_segments, ends):
            seg["end"] = float(new_end)

    # --- Validate. Model-agnostic and always on: a cue start sitting on silence is wrong
    # whichever align model produced it. Runs on final timings, after the release/trim
    # pass, so it judges what the rest of the pipeline will actually consume.
    silent = warn_on_silent_starts(
        aligned_segments,
        vad_segments if vad_segments is not None else whole_file_region(audio, MAX_DURATION),
    )
    # Land the finding on the cue rather than only printing it: a warning that names no cue
    # cannot be acted on across a 2000-line file. setdefault so a reason set at placement
    # time ("no audio for this line") outranks a symptom of it.
    for hit in silent:
        if 0 <= hit.index < len(aligned_segments):
            aligned_segments[hit.index].setdefault("realign_reason", "silent_start")

    # ...and the same question asked of the timings alone: a cue holding a silence between
    # two of its own adjacent characters is not one utterance, and its edges are the least
    # trustworthy in the file. A note rather than a reason or a repair -- see
    # align_checks.find_gapped_cues for why it is deliberately not fixed here.
    for gapped in warn_on_gapped_cues(aligned_segments, split_chars=punctuation.split_chars):
        add_note(aligned_segments[gapped.index],
                 f"internal_gap:{gapped.gap:.1f}s after {gapped.before}")

    # Record on each cue which of its characters the model could not read as written. Done
    # here rather than at substitution time because a substitution is per *character* over
    # the whole file, while what a reader wants to see is the handful of cues it touched.
    if repair is not None and (repair.substitutions or repair.unresolved):
        for seg in aligned_segments:
            for note in substitution_notes(
                seg.get("text", ""), repair.substitutions, repair.unresolved,
            ):
                add_note(seg, note)

    # --- Collect word segments ---
    word_segments: List[SingleWordSegment] = [w for seg in aligned_segments for w in seg["words"]]
    return {"segments": aligned_segments, "word_segments": word_segments}

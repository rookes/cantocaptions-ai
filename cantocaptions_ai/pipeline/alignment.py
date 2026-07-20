"""
Forced Alignment with Whisper
C. Max Bain
"""
from dataclasses import dataclass
import math
import time
from typing import Iterable, Mapping, Optional, Union, List, Tuple

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
    interpolate_nans,
)
from cantocaptions_ai.cantonese.text import DEFAULT_PUNCTUATION, PunctuationConfig, SpotCheck

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

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
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
) -> SegmentData:
    """Clean text and produce per-segment alignment metadata."""
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
        "sentence_spans": _get_sentence_spans(text, model_lang, punctuation),
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
        segment_data[sdx] = _preprocess_segment(segment["text"], model_lang, model_dictionary, punctuation)
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
) -> List[Tuple[torch.Tensor, float]]:
    """Batch VAD segments through the HF Wav2Vec2-BERT model via BatchExecutor.

    bert_processor's feature extractor pads each batch to its own longest segment
    and returns an attention_mask; this model has add_adapter=False, so the mask's
    per-row valid length maps 1:1 onto the CTC emission's frame dimension with no
    extra downsampling to account for — used directly to trim each item's emission
    back to its real (unpadded) length before it's stored.

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

    def infer_fn(batch: List[int]) -> None:
        wavs = [vad_segments[i]["audio"] for i in batch]
        with torch.inference_mode():
            features = bert_processor(
                wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True,
            )
            input_features = features["input_features"].to(device, dtype=model_dtype)
            attention_mask = features["attention_mask"].to(device)
            if vram_checks:
                _warn_alignment_vram(input_features, model, device)
            emissions = torch.log_softmax(
                model(input_features, attention_mask=attention_mask).logits, dim=-1
            )
        valid_lens = features["attention_mask"].sum(dim=-1)
        for row, i in enumerate(batch):
            real_len = int(valid_lens[row].item())
            emission = emissions[row, :real_len, :].cpu().detach()
            vad_duration = vad_segments[i]["end"] - vad_segments[i]["start"]
            frame_rate = emission.size(0) / vad_duration if vad_duration > 0 else 0.0
            results[i] = (emission, frame_rate)

    BatchExecutor(
        batch_size, order_key=lambda i: len(vad_segments[i]["audio"]),
    ).run(jobs, infer_fn)
    return results


def _compute_vad_emissions(
    vad_segments: List[VadAudioSegment],
    model: torch.nn.Module,
    model_type: str,
    bert_processor,
    device: str,
    batch_size: int = 4,
    vram_checks: bool = True,
) -> List[Tuple[torch.Tensor, float]]:
    """Run inference on each full VAD segment. Returns (log_softmax_emission, frame_rate) per segment.

    Logs before/after regardless of which path runs below, so a slow pass (a file
    with many/long VAD segments) is visibly explained rather than looking like a
    hang — this ran with no progress feedback at all before batching was added.
    """
    if not vad_segments:
        return []
    start = time.perf_counter()
    logger.info("Computing alignment emissions for %d VAD segments...", len(vad_segments))
    if model_type == "huggingface" and bert_processor is not None:
        results = _compute_vad_emissions_batched(vad_segments, model, bert_processor, device, batch_size, vram_checks=vram_checks)
    else:
        results = _compute_vad_emissions_sequential(vad_segments, model, model_type, bert_processor, device)
    logger.info("Alignment emissions computed in %.1fs", time.perf_counter() - start)
    return results


def compute_vad_emissions(vad_segments, model, model_type, bert_processor, device, batch_size: int = 4, vram_checks: bool = True):
    """Public wrapper around _compute_vad_emissions for use by the retime pipeline."""
    return _compute_vad_emissions(vad_segments, model, model_type, bert_processor, device, batch_size, vram_checks=vram_checks)


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
) -> Optional[torch.Tensor]:
    """Return the emission tensor for one segment, or None if no VAD match is found."""
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
        return full_emission[e1:e2, :]

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
    return emissions[0].cpu().detach()


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
) -> List[dict]:
    """Align one transcript segment against its emission, returning subsegment dicts."""
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

    char_segments = merge_repeats(path, text_clean)
    duration = t2 - t1
    ratio = duration / (trellis.size(0) - 1)

    char_segments_arr = []
    word_idx = 0
    for cdx, char in enumerate(text):
        start, end, score = None, None, None
        if cdx in seg_data["clean_cdx"]:
            char_seg = char_segments[seg_data["clean_cdx"].index(cdx)]
            start = round(char_seg.start * ratio + t1, 3)
            end = round(char_seg.end * ratio + t1, 3)
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
    agg_dict = {"text": " ".join, "words": "sum", "release_from": "first"}
    if model_lang in LANGUAGES_WITHOUT_SPACES:
        agg_dict["text"] = "".join
    if return_char_alignments:
        agg_dict["chars"] = "sum"
    if avg_logprob is not None:
        agg_dict["avg_logprob"] = "first"

    aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
    return aligned_subsegments.to_dict("records")


# --- Public functions ---

def load_align_model(
    language_code: str, device: str, device_index: int = 0, model_name: Optional[str] = None,
    model_dir=None, model_cache_only: bool = False, compute_type: str = "float32",
    vram_checks: bool = True,
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

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}
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
) -> AlignedTranscriptionResult:
    """Align phoneme recognition predictions to known transcription.

    ``spotchecks`` and ``punctuation`` come from the ASR model's profile (see
    ``pipeline/model_profiles.py``); their defaults (no spot checks, standard
    punctuation) keep alignment independent of any specific model.
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
    blank_id = _get_blank_id(model_dictionary)
    spacing_char_id = blank_id # model_dictionary['！']

    vad_seg_emissions = (
        _compute_vad_emissions(vad_segments, model, model_type, bert_processor, device, batch_size, vram_checks=vram_checks)
        if vad_segments is not None
        else None
    )

    # --- Preprocess transcript ---
    transcript = list(transcript)
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

        emission = _get_emission_for_segment(
            t1, t2, audio, vad_segments, vad_seg_emissions,
            model, model_type, bert_processor, device,
        )
        if emission is None:
            logger.warning(f'Failed to align segment ("{text}"): no VAD segment found for start time {t1}, skipping')
            aligned_segments.append(base_seg)
            continue

        subsegments = _align_segment(
            segment, segment_data[sdx], emission,
            model_dictionary, model_lang, blank_id, spacing_char_id,
            t1, t2, interpolate_method, return_char_alignments,
            spotchecks, punctuation,
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

    # --- Collect word segments ---
    word_segments: List[SingleWordSegment] = [w for seg in aligned_segments for w in seg["words"]]
    return {"segments": aligned_segments, "word_segments": word_segments}

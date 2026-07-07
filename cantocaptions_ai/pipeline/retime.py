from __future__ import annotations

import os
from typing import List

import torch

from cantocaptions_ai.cantonese.text import SPLIT_CHARS
from cantocaptions_ai.pipeline.alignment import (
    _preprocess_segment,
    compute_vad_emissions,
    get_score,
)
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.schema import SingleSegment, VadAudioSegment

logger = get_logger(__name__)


def load_subtitle_file(path: str) -> List[SingleSegment]:
    """Parse a subtitle file (SRT, VTT, etc.) and return as a list of SingleSegment dicts.

    Times are in seconds. Text is the joined word strings from each subtitle block.
    """
    try:
        from suber.file_readers import read_input_file
    except ImportError as e:
        raise ImportError(
            "subtitle-edit-rate is required for --retime. Install with: pip install subtitle-edit-rate"
        ) from e

    ext = os.path.splitext(path)[1].upper().lstrip(".")
    if not ext:
        raise ValueError(f"Cannot determine subtitle format from path: {path}")

    subtitles = read_input_file(path, ext)
    segments: List[SingleSegment] = []
    for sub in subtitles:
        text = " ".join(w.string for w in sub.word_list).strip()
        if text:
            segments.append({"start": sub.start_time, "end": sub.end_time, "text": text})
    return segments


def _text_to_tokens(text: str, model_lang: str, model_dictionary: dict, blank_id: int) -> List[int]:
    """Convert subtitle text to alignment model token IDs for scoring.

    Mirrors the token sequence produced by _align_segment: known chars use their
    dict ID, SPLIT_CHARS (punctuation used as sentence boundaries) map to the
    spacing token (blank_id), unknown chars are dropped.
    """
    seg_data = _preprocess_segment(text, model_lang, model_dictionary)
    tokens = [
        model_dictionary[c] if c not in SPLIT_CHARS else blank_id
        for c in seg_data["clean_char"]
    ]
    return tokens


def retime_subtitles(
    subtitles: List[SingleSegment],
    vad_segments: List[VadAudioSegment],
    align_model: torch.nn.Module,
    align_metadata: dict,
    bert_processor,
    device: str,
    score_threshold: float = -5.0,
    search_window: float = 120.0,
    batch_size: int = 4,
) -> List[SingleSegment]:
    """Search the audio for each subtitle line and update its timing.

    For each subtitle, scans VAD segments within a search window to find the
    best acoustic match using the alignment model's CTC scoring. Returns
    SingleSegments whose start/end equal the matched VAD segment's bounds so
    that the standard align() pipeline can perform fine character-level timing.

    When no match exceeds score_threshold, the current rolling offset is applied
    and a warning is logged (hook for future missing-line detection).

    Args:
        subtitles: Parsed subtitle segments with original timings.
        vad_segments: VAD segments from the new audio.
        align_model: Loaded alignment model (Wav2Vec2-BERT).
        align_metadata: Metadata dict from load_align_model() — must contain
            'dictionary', 'language', 'type'.
        bert_processor: Wav2Vec2BertProcessor instance.
        device: Torch device string.
        score_threshold: Minimum average CTC score to accept a VAD match.
        search_window: Seconds to scan forward from the expected subtitle position.
        batch_size: VAD segments per batch when computing emissions.

    Returns:
        List of SingleSegments with updated start/end = matched VAD segment bounds,
        ready to pass to align().
    """
    if not vad_segments:
        logger.warning("No VAD segments found; returning subtitles with original timings.")
        return list(subtitles)

    model_dictionary: dict = align_metadata["dictionary"]
    model_lang: str = align_metadata["language"]
    model_type: str = align_metadata["type"]

    blank_id = next(
        (code for char, code in model_dictionary.items() if char in ("[pad]", "<pad>")), 0
    )

    logger.info("Computing VAD emissions for retime search...")
    vad_emissions = compute_vad_emissions(
        vad_segments, align_model, model_type, bert_processor, device, batch_size
    )

    current_offset: float = 0.0
    retimed: List[SingleSegment] = []

    for i, subtitle in enumerate(subtitles):
        tokens = _text_to_tokens(subtitle["text"], model_lang, model_dictionary, blank_id)

        if not tokens:
            logger.warning(
                f"Subtitle {i} has no alignable characters, applying offset: {subtitle['text'][:40]!r}"
            )
            retimed.append(_apply_offset(subtitle, current_offset))
            continue

        if i == 0:
            search_start = 0.0
        else:
            search_start = max(0.0, subtitle["start"] + current_offset - 5.0)
        search_end = search_start + search_window

        candidates = [
            (j, vad_segments[j])
            for j in range(len(vad_segments))
            if vad_segments[j]["end"] > search_start and vad_segments[j]["start"] < search_end
        ]

        best_score = float("-inf")
        best_vad_idx: int | None = None

        for j, vad_seg in candidates:
            full_emission, _ = vad_emissions[j]
            try:
                score = get_score(full_emission, tokens, blank_id=blank_id)
            except Exception:
                score = float("-inf")
            if score > best_score:
                best_score = score
                best_vad_idx = j

        # TODO: future — if best_score is still below threshold, try merging
        # adjacent VAD emissions to handle subtitles that span a short pause.

        if best_vad_idx is not None and best_score > score_threshold:
            matched = vad_segments[best_vad_idx]
            sub_center = (subtitle["start"] + subtitle["end"]) / 2
            vad_center = (matched["start"] + matched["end"]) / 2
            current_offset = vad_center - sub_center
            logger.debug(
                f"Subtitle {i} matched VAD seg {best_vad_idx} "
                f"[{matched['start']:.2f}–{matched['end']:.2f}] "
                f"score={best_score:.3f} offset={current_offset:+.2f}s"
            )
            retimed.append({**subtitle, "start": matched["start"], "end": matched["end"]})
        else:
            # TODO: future — detect missing/cut lines and handle them separately
            logger.warning(
                f"Subtitle {i} not found in audio (best score {best_score:.3f} < {score_threshold}), "
                f"applying rolling offset {current_offset:+.2f}s: {subtitle['text'][:40]!r}"
            )
            retimed.append(_apply_offset(subtitle, current_offset))

    return retimed


def _apply_offset(subtitle: SingleSegment, offset: float) -> SingleSegment:
    return {
        **subtitle,
        "start": max(0.0, subtitle["start"] + offset),
        "end": max(0.0, subtitle["end"] + offset),
    }

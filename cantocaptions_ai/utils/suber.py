from __future__ import annotations

import argparse
from typing import List

try:
    import regex
    from suber.data_types import LineBreak, Subtitle, TimedWord
    from suber.file_readers import read_input_file
    from suber.hyp_to_ref_alignment import levenshtein_align_hypothesis_to_reference
    from suber.lib_levenshtein import distance as _levenshtein_distance
    from suber.metrics.cer import calculate_character_error_rate
    from suber.metrics.suber import calculate_SubER
except ImportError as e:
    raise ImportError(
        "subtitle-edit-rate is required. Install with: pip install subtitle-edit-rate"
    ) from e

from cantocaptions_ai.utils.schema import (
    AlignedTranscriptionResult,
    SingleSegment,
    TranscriptionResult,
)


def _to_suber_subtitles(result: AlignedTranscriptionResult) -> List[Subtitle]:
    """Convert an AlignedTranscriptionResult to a list of SubER Subtitle objects.

    Text is split on whitespace. For CJK text without spaces, SubER's internal
    East Asian tokenization handles character-level segmentation during alignment.
    """
    subtitles = []
    for idx, seg in enumerate(result["segments"], start=1):
        tokens = seg["text"].split()
        words = [
            TimedWord(
                string=w,
                subtitle_start_time=seg["start"],
                subtitle_end_time=seg["end"],
            )
            for w in tokens
        ]
        if words:
            last = words[-1]
            words[-1] = TimedWord(
                string=last.string,
                line_break=LineBreak.END_OF_BLOCK,
                subtitle_start_time=seg["start"],
                subtitle_end_time=seg["end"],
            )
        subtitles.append(
            Subtitle(word_list=words, index=idx, start_time=seg["start"], end_time=seg["end"])
        )
    return subtitles


def calculate_suber(hyp_path: str, ref_path: str, metric: str = "SubER") -> float:
    """Calculate SubER between a hypothesis SRT file and a reference SRT file."""
    hyp = read_input_file(hyp_path, "SRT")
    ref = read_input_file(ref_path, "SRT")
    return calculate_SubER(hyp, ref, metric=metric)


def calculate_cer(hyp_path: str, ref_path: str, metric: str = "CER") -> float:
    """Calculate CER between a hypothesis SRT file and a reference SRT file.

    Levenshtein-aligns the hypothesis to the reference before computing CER,
    since SubER's CER requires segment counts to match.
    """
    hyp = read_input_file(hyp_path, "SRT")
    ref = read_input_file(ref_path, "SRT")
    aligned = levenshtein_align_hypothesis_to_reference(hyp, ref)
    return calculate_character_error_rate(aligned, ref, metric=metric)


def _document_text(path: str, cased: bool = False) -> str:
    """Whole-file text with cue boundaries and punctuation removed."""
    subtitles = read_input_file(path, "SRT")
    text = "".join(w.string for sub in subtitles for w in sub.word_list)
    text = regex.sub(r"\p{P}", "", text)
    return text if cased else text.lower()


def calculate_document_cer(hyp_path: str, ref_path: str, cased: bool = False) -> float:
    """Character error rate over the concatenated file, ignoring cue boundaries.

    ``calculate_cer`` pairs cues one-to-one after a Levenshtein alignment pass and then
    scores each pair independently, so a single inserted or dropped cue misaligns every
    later pair and inflates the score several-fold: on the Bluey fixture it reports
    ~77% where the true character error rate is ~17%. That makes it unusable for
    comparing two ASR configurations whose cue boundaries differ at all.

    This measures transcription accuracy alone. Use it when the question is "did the
    text get better"; use ``calculate_suber`` when the question is "did the subtitles
    get better", since that one is *supposed* to be sensitive to segmentation.

    Returned as a percentage, matching the other metrics here.
    """
    hyp = _document_text(hyp_path, cased)
    ref = _document_text(ref_path, cased)
    if not ref:
        return 0.0 if not hyp else 100.0
    return round(100 * _levenshtein_distance(hyp, ref) / len(ref), 3)


def align_to_reference(
    hypothesis: AlignedTranscriptionResult,
    reference: AlignedTranscriptionResult,
) -> TranscriptionResult:
    """Re-segment hypothesis text to match reference segment boundaries.

    Uses Levenshtein alignment to redistribute hypothesis words across reference
    segments. The output uses reference timing; word-level timing is not preserved
    since words are redistributed across new boundaries.

    Args:
        hypothesis: The transcription whose text will be redistributed.
        reference: The transcription whose segment boundaries (timing) will be used.

    Returns:
        TranscriptionResult with reference timing and hypothesis text.
    """
    hyp_subtitles = _to_suber_subtitles(hypothesis)
    ref_subtitles = _to_suber_subtitles(reference)
    aligned = levenshtein_align_hypothesis_to_reference(hyp_subtitles, ref_subtitles)

    segments: List[SingleSegment] = []
    for aligned_seg, ref_seg in zip(aligned, reference["segments"]):
        text = " ".join(w.string for w in aligned_seg.word_list)
        segments.append(
            {
                "start": ref_seg["start"],
                "end": ref_seg["end"],
                "text": text,
            }
        )

    return {"segments": segments, "language": hypothesis["language"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate SubER or CER between a hypothesis SRT and a reference SRT."
    )
    parser.add_argument("hyp", help="hypothesis SRT file")
    parser.add_argument("ref", help="reference SRT file")
    parser.add_argument(
        "--metric",
        choices=["SubER", "SubER-cased", "CER", "CER-cased", "docCER", "docCER-cased"],
        default="SubER",
        help="metric to compute (default: SubER)",
    )
    args = parser.parse_args()

    if args.metric.startswith("docCER"):
        score = calculate_document_cer(args.hyp, args.ref, cased=args.metric.endswith("-cased"))
    elif args.metric.startswith("CER"):
        score = calculate_cer(args.hyp, args.ref, metric=args.metric)
    else:
        score = calculate_suber(args.hyp, args.ref, metric=args.metric)

    print(f"{args.metric}: {score}")

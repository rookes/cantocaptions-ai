from typing import TYPE_CHECKING, Callable, TypedDict, Optional, List, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from cantocaptions_ai.utils.log_utils import ProgressReporter

# A pipeline stage receives a ProgressReporter (set_total / advance) rather than a
# bare float callback, so tqdm can render accurate throughput and ETA across all files.
ProgressCallback = Optional["ProgressReporter"]


def interpolate_nans(x, method: str = 'nearest'):
    """Fill NaN values in a pandas Series using interpolation."""
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class VadAudioSegment(TypedDict):
    """A VAD-detected speech segment with its extracted audio data."""
    start: float
    end: float
    audio: np.ndarray


class SingleWordSegment(TypedDict):
    """
    A single word of a speech.
    """
    word: str
    start: float
    end: float
    score: float
    speaker: NotRequired[str]  # attached by speaker_assign when diarization ran

class SingleCharSegment(TypedDict):
    """
    A single char of a speech.
    """
    char: str
    start: float
    end: float
    score: float

class TimeStampChar(TypedDict):
    """
    A single character of speech with start and end time.
    """
    char: str
    start: float
    end: float

class SingleSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech.
    """

    start: float
    end: float
    text: str
    time_stamps: NotRequired[List[TimeStampChar]]
    avg_logprob: NotRequired[float]
    speaker: NotRequired[str]
    speaker_confidence: NotRequired[float]
    speaker_conflict: NotRequired[bool]


class SegmentData(TypedDict):
    """
    Temporary processing data used during alignment.
    Contains cleaned and preprocessed data for each segment.
    """
    clean_char: List[str]  # Cleaned characters that exist in model dictionary
    clean_cdx: List[int]   # Original indices of cleaned characters
    clean_wdx: List[int]   # Indices of words containing valid characters
    sentence_spans: List[Tuple[int, int]]  # Start and end indices of sentences


class SingleAlignedSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """

    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float]
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]
    # Attached by the diarization stage (see pipeline/speaker_assign.py). ``speaker`` is only
    # set when one speaker holds a confident majority of the segment; a missing key means
    # "unknown", which cue assembly treats as "no objection to merging".
    speaker: NotRequired[str]
    speaker_confidence: NotRequired[float]
    speaker_conflict: NotRequired[bool]


class TranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """
    segments: List[SingleSegment]
    language: str


class AlignedTranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """
    segments: List[SingleAlignedSegment]
    word_segments: List[SingleWordSegment]
    language: str


class SpeakerTurn(TypedDict):
    """One contiguous stretch of audio attributed to a single speaker by diarization."""
    start: float
    end: float
    speaker: str


class DiarizationResult(TypedDict):
    """Raw output of the diarization stage for one file, before it is mapped onto segments.

    ``turns`` comes from pyannote's *exclusive* diarization (no overlapping speech), which is
    what speaker assignment consumes; ``overlap_turns`` keeps the unmodified diarization for
    debug inspection of crosstalk. Times are always on the file timeline.

    Under ``scope == "segment"`` each VAD segment was diarized independently, so speaker
    labels are namespaced (``S0007/SPEAKER_00``) and are only comparable within one segment;
    ``segment_speakers`` then carries the per-segment breakdown.
    """
    speakers: List[str]
    turns: List[SpeakerTurn]
    overlap_turns: List[SpeakerTurn]
    scope: NotRequired[str]
    segment_speakers: NotRequired[List[dict]]
    embeddings: NotRequired[Optional[dict]]


class VadItem(TypedDict):
    """Intermediate carrier for the VAD and vocal-isolation stages (before transcription)."""
    audio_path: str
    vad_segments: List[VadAudioSegment]
    audio_track: NotRequired[int]


class ProcessingItem(TypedDict):
    """Carries one audio file's data from transcription onwards."""
    audio_path: str
    result: Union[TranscriptionResult, AlignedTranscriptionResult]
    vad_segments: NotRequired[List[VadAudioSegment]]
    ensemble_texts: NotRequired[List[str]]  # index-aligned alternative ASR hypotheses
    reference_texts: NotRequired[List[str]]  # time-matched standard Chinese reference; one per segment
    audio_track: NotRequired[int]
    diarization: NotRequired[DiarizationResult]


def merge_segments(seg1: SingleAlignedSegment, seg2: SingleAlignedSegment) -> SingleAlignedSegment:
    """Merge two adjacent aligned segments into one.

    Starts from a copy of ``seg1`` so keys attached by later stages (and anything else a
    caller carries) survive the merge; only the fields that a merge actually redefines are
    overridden. ``avg_logprob`` is cleared because it is genuinely undefined for a merged cue.

    Speaker attribution needs more than ``dict(seg1)``: callers only merge when the two sides
    do not contradict each other (see ``segmentation._same_speaker``), so an unlabeled side
    means "unknown", not "nobody". Taking seg1's label unconditionally would drop seg2's
    label when merging (unknown, SPEAKER_01) and let a later merge cross a real speaker
    boundary, so whichever side carries a label wins. ``speaker_confidence`` is dropped for
    the same reason as ``avg_logprob``: it described one of the two originals, not the union.
    """
    s3_chars = (seg1.get("chars") or []) + (seg2.get("chars") or [])
    merged = dict(seg1)
    merged.update({
        "start": seg1["start"],
        "end": seg2["end"],
        "text": seg1["text"] + seg2["text"],
        "avg_logprob": None,
        "words": seg1["words"] + seg2["words"],
        "chars": s3_chars or None,
    })
    speaker = seg1.get("speaker") or seg2.get("speaker")
    if speaker is not None:
        merged["speaker"] = speaker
    merged.pop("speaker_confidence", None)
    if seg1.get("speaker_conflict") or seg2.get("speaker_conflict"):
        merged["speaker_conflict"] = True
    return merged

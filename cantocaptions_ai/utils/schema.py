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


def merge_segments(seg1: SingleAlignedSegment, seg2: SingleAlignedSegment) -> SingleAlignedSegment:
    """Merge two adjacent aligned segments into one."""
    s3_chars = (seg1.get("chars") or []) + (seg2.get("chars") or [])
    return {
        "start": seg1["start"],
        "end": seg2["end"],
        "text": seg1["text"] + seg2["text"],
        "avg_logprob": None,
        "words": seg1["words"] + seg2["words"],
        "chars": s3_chars or None,
    }

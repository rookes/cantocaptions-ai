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
    # Reference-subtitle text biasing this segment's ASR decode (--asr_context).
    # Attached in transcribe.py *after* the VAD/isolation cache load, because both
    # debug round-trips rebuild segment dicts from scratch and would drop it; it is
    # cheap to re-derive and always is. Consumed only by the ASR backends.
    context: NotRequired[str]
    # Sub-spans of this segment that exist only because a reference cue was unioned into
    # the VAD timeline -- audio VAD itself never detected (see reference_context.
    # expansion_only_spans). Unlike 'context' this is set at VAD time and *is* persisted
    # through the VAD/isolation manifests, because it cannot be re-derived later: the
    # pre-expansion timeline is gone by then. Read by --asr_context_scope expanded.
    expanded: NotRequired[List[List[float]]]
    # Where inside this segment VAD actually heard speech, as [[start, end], ...] on the file
    # timeline. Set only under --realign's split-only chunking, where the segment boundaries
    # are cut points rather than speech boundaries and so carry no such information
    # themselves. Persisted for the same reason as 'expanded': the score curve it came from
    # is gone by the time anything downstream wants it. Read by realign's ASR anchor.
    speech: NotRequired[List[List[float]]]


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
    # Cue boundaries declared by the caller, as inclusive (start, end) index pairs into
    # ``text`` -- the same shape PunctuationConfig.sentence_spans returns. When present,
    # alignment slices its output on these instead of deriving spans from punctuation.
    # Set by --realign, where the source transcript arrives pre-split into cues and its
    # line breaks, not its (sparse) punctuation, are the authoritative boundaries.
    cue_spans: NotRequired[List[Tuple[int, int]]]
    # Why each of those cues is not to be trusted, index-aligned with cue_spans (None where
    # it is fine). Carried through alignment onto the finished cue so the reason survives to
    # something the user can watch; see realign.REASON_HELP.
    cue_reasons: NotRequired[List[Optional[str]]]


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
    # Why this cue's timing is doubtful, from --realign. Written to realign/suspect.srt and
    # summarised at the end of the run; see realign.REASON_HELP.
    realign_reason: NotRequired[str]
    # Free-form annotations any stage may attach to a cue, as "kind:detail" strings. Unlike
    # realign_reason -- which says "do not trust this timing" and drives suspect.srt -- a note
    # is informational: it records something that happened to this cue which a person might
    # want to see, whether or not anything is wrong. Rendered by
    # debug.write_segment_notes into notes.srt, and deliberately open-ended so a new stage
    # needs no new field. First user: align_vocab's character substitutions.
    notes: NotRequired[List[str]]


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
    audio_downmix: NotRequired[str]


class ProcessingItem(TypedDict):
    """Carries one audio file's data from transcription onwards."""
    audio_path: str
    result: Union[TranscriptionResult, AlignedTranscriptionResult]
    vad_segments: NotRequired[List[VadAudioSegment]]
    audio_downmix: NotRequired[str]
    ensemble_texts: NotRequired[List[str]]  # index-aligned alternative ASR hypotheses
    reference_texts: NotRequired[List[str]]  # time-matched standard Chinese reference; one per segment
    audio_track: NotRequired[int]
    diarization: NotRequired[DiarizationResult]
    # One emission timeline over the whole file, set by --realign's placement stage and
    # reused by alignment so the encoder runs once rather than twice. Not serialisable and
    # never checkpointed; see realign.EmissionTimeline.
    emission_timeline: NotRequired[object]


def add_note(segment: SingleAlignedSegment, note: str) -> None:
    """Attach an informational annotation to a cue, ignoring an exact duplicate.

    Notes accumulate across stages and survive merges, so the same note arriving twice (the
    same substituted character in both halves of a merged cue, say) must not double up.
    """
    notes = segment.setdefault("notes", [])
    if note not in notes:
        notes.append(note)


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
    notes = list(seg1.get("notes") or [])
    for note in seg2.get("notes") or []:
        if note not in notes:
            notes.append(note)
    if notes:
        merged["notes"] = notes
    merged.pop("speaker_confidence", None)
    if seg1.get("speaker_conflict") or seg2.get("speaker_conflict"):
        merged["speaker_conflict"] = True
    return merged

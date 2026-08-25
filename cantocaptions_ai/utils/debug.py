import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from cantocaptions_ai.utils.audio import SAMPLE_RATE
from typing import Union

from cantocaptions_ai.utils.schema import (
    AlignedTranscriptionResult,
    DiarizationResult,
    SingleAlignedSegment,
    TranscriptionResult,
    VadAudioSegment,
)
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


def _stem(audio_path: str) -> str:
    """Directory name a file's checkpoints live under.

    Writers and readers must agree on this: transcribe.py decides whether to skip a
    stage from `_debug_stage_exists`, so a stem that only the writer strips would make
    "cached" and "loadable" disagree.
    """
    return Path(audio_path).stem.strip()


def _stage_dir(audio_path: str, stage: str, debug_dir: str) -> str:
    path = os.path.join(debug_dir, _stem(audio_path), stage)
    os.makedirs(path, exist_ok=True)
    return path


# Segment keys carried through the VAD / vocal-isolation manifests alongside the audio.
# An explicit allowlist rather than a blanket passthrough: segment dicts also hold numpy
# arrays and other non-JSON values, so copying everything would break json.dump.
_PERSISTED_SEGMENT_KEYS = ("expanded",)


def _write_audio_segments(segments: List[VadAudioSegment], stage_dir: str) -> list:
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for debug audio output: pip install soundfile")

    segment_records = []
    for i, seg in enumerate(segments):
        filename = f"segment_{i:04d}.wav"
        filepath = os.path.join(stage_dir, filename)
        audio = seg["audio"]
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        sf.write(filepath, audio, SAMPLE_RATE, subtype="PCM_16")
        record = {
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "file": filename,
        }
        for key in _PERSISTED_SEGMENT_KEYS:
            if key in seg:
                record[key] = seg[key]
        segment_records.append(record)
    return segment_records


def _write_segments_json(audio_path: str, segment_records: list, stage_dir: str) -> None:
    manifest = {
        "audio_path": os.path.abspath(audio_path),
        "sample_rate": SAMPLE_RATE,
        "segments": segment_records,
    }
    json_path = os.path.join(stage_dir, "segments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_vad_debug(audio_path: str, segments: List[VadAudioSegment], debug_dir: str) -> None:
    """Write VAD segments (audio + manifest) to {debug_dir}/{stem}/vad/."""
    stage_dir = _stage_dir(audio_path, "vad", debug_dir)
    segment_records = _write_audio_segments(segments, stage_dir)
    _write_segments_json(audio_path, segment_records, stage_dir)
    logger.info(f"VAD debug output written to {stage_dir} ({len(segments)} segments)")


def write_isolation_debug(audio_path: str, segments: List[VadAudioSegment], debug_dir: str) -> None:
    """Write vocal isolation segments (audio + manifest) to {debug_dir}/{stem}/vocal_isolation/."""
    stage_dir = _stage_dir(audio_path, "vocal_isolation", debug_dir)
    segment_records = _write_audio_segments(segments, stage_dir)
    _write_segments_json(audio_path, segment_records, stage_dir)
    logger.info(f"Vocal isolation debug output written to {stage_dir} ({len(segments)} segments)")


def write_transcription_debug(audio_path: str, result: TranscriptionResult, debug_dir: str) -> None:
    """Write normalized transcription result to {debug_dir}/{stem}/transcription/result.json."""
    stage_dir = _stage_dir(audio_path, "transcription", debug_dir)
    output = {
        "audio_path": os.path.abspath(audio_path),
        "language": result.get("language"),
        "segments": result["segments"],
    }
    json_path = os.path.join(stage_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Transcription debug output written to {json_path} ({len(result['segments'])} segments)")


# ---------------------------------------------------------------------------
# Load functions (inverse of the write functions above)
# ---------------------------------------------------------------------------

def _debug_stage_exists(audio_path: str, stage: str, debug_dir: str) -> bool:
    """Return True if the expected marker file for a stage exists in the debug directory."""
    stem = _stem(audio_path)
    stage_dir = os.path.join(debug_dir, stem, stage)
    marker = {
        "transcription": "result.json",
        "llm_correction": "result.json",
        "diarization": "result.json",
        "ensemble": "texts.json",
    }.get(stage, "segments.json")
    return os.path.isfile(os.path.join(stage_dir, marker))


def _load_audio_segments(stage_dir: str) -> Optional[List[VadAudioSegment]]:
    """Load VadAudioSegments from a stage directory. Returns None if the directory or manifest is absent."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for debug audio loading: pip install soundfile")

    manifest_path = os.path.join(stage_dir, "segments.json")
    if not os.path.isfile(manifest_path):
        return None

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    segments = []
    for rec in manifest["segments"]:
        wav_path = os.path.join(stage_dir, rec["file"])
        audio, _ = sf.read(wav_path, dtype="float32")
        seg = {"start": rec["start"], "end": rec["end"], "audio": audio}
        for key in _PERSISTED_SEGMENT_KEYS:
            if key in rec:
                seg[key] = rec[key]
        segments.append(seg)
    return segments


def load_vad_debug(audio_path: str, debug_dir: str) -> Optional[List[VadAudioSegment]]:
    """Load VAD segments from a previous debug run. Returns None if not present."""
    stem = _stem(audio_path)
    stage_dir = os.path.join(debug_dir, stem, "vad")
    segments = _load_audio_segments(stage_dir)
    if segments is not None:
        logger.info(f"Loaded {len(segments)} VAD segments from {stage_dir}")
    return segments


def load_isolation_debug(audio_path: str, debug_dir: str) -> Optional[List[VadAudioSegment]]:
    """Load vocal isolation segments from a previous debug run. Returns None if not present."""
    stem = _stem(audio_path)
    stage_dir = os.path.join(debug_dir, stem, "vocal_isolation")
    segments = _load_audio_segments(stage_dir)
    if segments is not None:
        logger.info(f"Loaded {len(segments)} vocal isolation segments from {stage_dir}")
    return segments


def load_transcription_debug(audio_path: str, debug_dir: str) -> Optional[TranscriptionResult]:
    """Load a normalized transcription result from a previous debug run. Returns None if not present."""
    stem = _stem(audio_path)
    json_path = os.path.join(debug_dir, stem, "transcription", "result.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    result: TranscriptionResult = {"segments": data["segments"], "language": data.get("language")}
    logger.info(f"Loaded transcription ({len(result['segments'])} segments) from {json_path}")
    return result


def write_ensemble_debug(audio_path: str, texts: List[str], debug_dir: str) -> None:
    """Write ensemble ASR texts to {debug_dir}/{stem}/ensemble/texts.json."""
    stage_dir = _stage_dir(audio_path, "ensemble", debug_dir)
    json_path = os.path.join(stage_dir, "texts.json")
    data = {"audio_path": os.path.abspath(audio_path), "texts": texts}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Ensemble debug output written to {json_path} ({len(texts)} segments)")


def load_ensemble_debug(audio_path: str, debug_dir: str) -> Optional[List[str]]:
    """Load ensemble ASR texts from a previous debug run. Returns None if not present."""
    stem = _stem(audio_path)
    json_path = os.path.join(debug_dir, stem, "ensemble", "texts.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        texts = json.load(f)["texts"]
    logger.info(f"Loaded ensemble texts ({len(texts)} segments) from {json_path}")
    return texts


def write_llm_correction_debug(audio_path: str, result: TranscriptionResult, debug_dir: str) -> None:
    """Write LLM-corrected transcription to {debug_dir}/{stem}/llm_correction/result.json."""
    stage_dir = _stage_dir(audio_path, "llm_correction", debug_dir)
    output = {
        "audio_path": os.path.abspath(audio_path),
        "language": result.get("language"),
        "segments": result["segments"],
    }
    json_path = os.path.join(stage_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"LLM correction debug output written to {json_path} ({len(result['segments'])} segments)")


def write_precleaning_debug(
    audio_path: str,
    result: Union[TranscriptionResult, AlignedTranscriptionResult],
    debug_dir: str,
) -> None:
    """Write the merged, pre-cleaning subtitles to {debug_dir}/{stem}/pre_cleaning/.

    Writes {stem}.srt (diffable against the final output) plus result.json for parity
    with other stages. This is a snapshot, not a loadable checkpoint: cleaning always
    re-runs so that rule-file edits take effect under --load_debug_dir.
    """
    from cantocaptions_ai.utils.output import WriteSRT

    stage_dir = _stage_dir(audio_path, "pre_cleaning", debug_dir)
    WriteSRT(stage_dir)(result, audio_path, {})
    output = {
        "audio_path": os.path.abspath(audio_path),
        "language": result.get("language"),
        "segments": result["segments"],
    }
    json_path = os.path.join(stage_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Pre-cleaning debug output written to {stage_dir} ({len(result['segments'])} segments)")


def load_llm_correction_debug(audio_path: str, debug_dir: str) -> Optional[TranscriptionResult]:
    """Load LLM-corrected transcription from a previous debug run. Returns None if not present."""
    stem = _stem(audio_path)
    json_path = os.path.join(debug_dir, stem, "llm_correction", "result.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    result: TranscriptionResult = {"segments": data["segments"], "language": data.get("language")}
    logger.info(f"Loaded LLM correction ({len(result['segments'])} segments) from {json_path}")
    return result


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def write_diarization_debug(audio_path: str, result: DiarizationResult, debug_dir: str) -> None:
    """Write raw diarization output to {debug_dir}/{stem}/diarization/result.json.

    This is the loadable half of the stage: it holds the model's answer (speaker count and
    turns), not the thresholded per-segment attribution, so a --load_debug_dir replay can
    skip the diarization model while still re-deriving labels at new thresholds.
    """
    stage_dir = _stage_dir(audio_path, "diarization", debug_dir)
    # Under segment scope, "speakers" is the set of per-segment local identities, so its
    # length is a count of voices-per-segment summed, not a count of people in the episode.
    # Name it for what it is rather than implying a global speaker count that wasn't computed.
    count_key = (
        "num_speakers" if result.get("scope", "file") == "file" else "num_local_speakers"
    )
    output = {
        "audio_path": os.path.abspath(audio_path),
        count_key: len(result["speakers"]),
        **result,
    }
    json_path = os.path.join(stage_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Diarization debug output written to {json_path} "
        f"({output[count_key]} {count_key.removeprefix('num_')}, {len(result['turns'])} turns)"
    )


def load_diarization_debug(audio_path: str, debug_dir: str) -> Optional[DiarizationResult]:
    """Load diarization output from a previous debug run. Returns None if not present."""
    stem = _stem(audio_path)
    json_path = os.path.join(debug_dir, stem, "diarization", "result.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    result: DiarizationResult = {
        "speakers": data["speakers"],
        "turns": data["turns"],
        "overlap_turns": data.get("overlap_turns", []),
    }
    # Optional keys are copied only when present so a checkpoint written by one scope
    # reloads as that scope, rather than silently reading back as a whole-file run.
    for key in ("scope", "segment_speakers", "embeddings"):
        if data.get(key) is not None:
            result[key] = data[key]
    logger.info(
        f"Loaded diarization ({len(result['speakers'])} speaker labels, "
        f"{len(result['turns'])} turns, scope: {result.get('scope', 'file')}) from {json_path}"
    )
    return result


def _assignment_label(segment: SingleAlignedSegment) -> str:
    """Bracketed diagnostic prefix describing one segment's speaker attribution."""
    confidence = segment.get("speaker_confidence")
    if segment.get("speaker") is None:
        # Unlabeled: either no diarized speech underneath, or no speaker cleared the
        # threshold. Both are merge-permissive, so both are worth seeing.
        marker = "?" if confidence is None else f"? {confidence:.2f}"
    else:
        marker = f"{segment['speaker']} {confidence:.2f}" if confidence is not None else segment["speaker"]
    if segment.get("speaker_conflict"):
        marker += " !MULTI"
    return f"[{marker}]"


def write_speaker_assignment_debug(
    audio_path: str,
    segments: List[SingleAlignedSegment],
    debug_dir: str,
) -> None:
    """Write per-subsegment speaker attribution to {debug_dir}/{stem}/diarization/.

    Produces assignments.json plus {stem}.srt, in which every cue is prefixed with the
    speaker, the confidence, and a !MULTI marker when the cue looks like it spans more than
    one speaker. Both are snapshots, not checkpoints: assignment is cheap and must re-run so
    threshold changes take effect on a --load_debug_dir replay.

    Note this runs on the *pre-assembly* subsegments, which is the level diarization acts at
    -- the final cues are what pre_cleaning/ shows.
    """
    from cantocaptions_ai.utils.output import WriteSRT

    stage_dir = _stage_dir(audio_path, "diarization", debug_dir)

    assignments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "speaker": segment.get("speaker"),
            "confidence": segment.get("speaker_confidence"),
            "conflict": bool(segment.get("speaker_conflict")),
        }
        for segment in segments
    ]
    json_path = os.path.join(stage_dir, "assignments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"audio_path": os.path.abspath(audio_path), "assignments": assignments},
            f, ensure_ascii=False, indent=2,
        )

    # Build a shadow result rather than reusing the live segments: the speaker key is folded
    # into the text here so the SRT shows confidence and conflicts too, and the writer's own
    # "[speaker]: " prefix would double up on it.
    shadow = {
        "language": None,
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": f"{_assignment_label(segment)} {segment['text'].strip()}",
            }
            for segment in segments
        ],
    }
    WriteSRT(stage_dir)(shadow, audio_path, {})

    flagged = sum(1 for a in assignments if a["conflict"])
    logger.info(
        f"Speaker assignment debug output written to {stage_dir} "
        f"({len(assignments)} subsegments, {flagged} flagged multi-speaker)"
    )

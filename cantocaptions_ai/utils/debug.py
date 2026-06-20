import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.schema import VadAudioSegment, TranscriptionResult
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


def _stage_dir(audio_path: str, stage: str, debug_dir: str) -> str:
    stem = Path(audio_path).stem
    path = os.path.join(debug_dir, stem, stage)
    os.makedirs(path, exist_ok=True)
    return path


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
        segment_records.append({
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "file": filename,
        })
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
    stem = Path(audio_path).stem
    stage_dir = os.path.join(debug_dir, stem, stage)
    marker = {
        "transcription": "result.json",
        "llm_correction": "result.json",
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
        segments.append({"start": rec["start"], "end": rec["end"], "audio": audio})
    return segments


def load_vad_debug(audio_path: str, debug_dir: str) -> Optional[List[VadAudioSegment]]:
    """Load VAD segments from a previous debug run. Returns None if not present."""
    stem = Path(audio_path).stem
    stage_dir = os.path.join(debug_dir, stem, "vad")
    segments = _load_audio_segments(stage_dir)
    if segments is not None:
        logger.info(f"Loaded {len(segments)} VAD segments from {stage_dir}")
    return segments


def load_isolation_debug(audio_path: str, debug_dir: str) -> Optional[List[VadAudioSegment]]:
    """Load vocal isolation segments from a previous debug run. Returns None if not present."""
    stem = Path(audio_path).stem
    stage_dir = os.path.join(debug_dir, stem, "vocal_isolation")
    segments = _load_audio_segments(stage_dir)
    if segments is not None:
        logger.info(f"Loaded {len(segments)} vocal isolation segments from {stage_dir}")
    return segments


def load_transcription_debug(audio_path: str, debug_dir: str) -> Optional[TranscriptionResult]:
    """Load a normalized transcription result from a previous debug run. Returns None if not present."""
    stem = Path(audio_path).stem
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
    stem = Path(audio_path).stem
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


def load_llm_correction_debug(audio_path: str, debug_dir: str) -> Optional[TranscriptionResult]:
    """Load LLM-corrected transcription from a previous debug run. Returns None if not present."""
    stem = Path(audio_path).stem
    json_path = os.path.join(debug_dir, stem, "llm_correction", "result.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    result: TranscriptionResult = {"segments": data["segments"], "language": data.get("language")}
    logger.info(f"Loaded LLM correction ({len(result['segments'])} segments) from {json_path}")
    return result

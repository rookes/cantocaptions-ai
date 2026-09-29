"""Every stage's debug checkpoint reads back as what was written.

``--load_debug_dir`` replays a stage from its checkpoint, so a field the writer drops or
the reader renames silently changes the replayed run. VAD and vocal isolation are covered
in test_segment_provenance; these are the remaining stages.
"""
import math

import pytest

from cantocaptions_ai.pipeline.realign import REASON_UNREADABLE, LineTiming, TranscriptLine
from cantocaptions_ai.utils import debug

SEGMENTS = [
    {"start": 1.0, "end": 2.5, "text": "你好"},
    {"start": 3.0, "end": 4.25, "text": "今日天氣幾好"},
]


@pytest.fixture
def d(tmp_path):
    return str(tmp_path)


def test_transcription(d):
    debug.write_transcription_debug("s1/ep01", {"segments": SEGMENTS, "language": "Cantonese"}, d)
    assert debug.load_transcription_debug("s1/ep01", d) == {
        "segments": SEGMENTS, "language": "Cantonese"}


def test_llm_correction(d):
    debug.write_llm_correction_debug("ep01", {"segments": SEGMENTS, "language": "yue"}, d)
    assert debug.load_llm_correction_debug("ep01", d)["segments"] == SEGMENTS


def test_ensemble(d):
    debug.write_ensemble_debug("ep01", ["你好", ""], d)
    assert debug.load_ensemble_debug("ep01", d) == ["你好", ""]


def test_diarization_keeps_its_scope(d):
    result = {
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "turns": [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        "overlap_turns": [],
        "scope": "segment",
        "segment_speakers": [{"start": 1.0, "end": 2.5, "speakers": ["SPEAKER_00"]}],
    }
    debug.write_diarization_debug("ep01", result, d)
    loaded = debug.load_diarization_debug("ep01", d)
    for key in ("speakers", "turns", "scope"):
        assert loaded[key] == result[key]


def test_realign_placements_and_reasons(d, tmp_path):
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("你好\n今日天氣幾好\n", encoding="utf-8")
    timings = [
        LineTiming(index=0, start=1.0, end=2.5, score=0.9),
        LineTiming(index=1, start=3.0, end=4.25, score=float("nan"), reason=REASON_UNREADABLE),
    ]
    lines = [TranscriptLine(index=0, text="你好"), TranscriptLine(index=1, text="今日天氣幾好")]
    debug.write_realign_debug("ep01", str(transcript), timings, lines, d)
    loaded = debug.load_realign_debug("ep01", str(transcript), d)
    assert [(t.index, t.start, t.end, t.reason) for t in loaded] == [
        (0, 1.0, 2.5, None), (1, 3.0, 4.25, REASON_UNREADABLE)]
    assert loaded[0].score == 0.9 and math.isnan(loaded[1].score)


def test_realign_checkpoint_for_another_transcript_is_refused(d, tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("你好\n", encoding="utf-8")
    b.write_text("你好\n", encoding="utf-8")
    debug.write_realign_debug("ep01", str(a), [LineTiming(0, 1.0, 2.0)],
                              [TranscriptLine(index=0, text="你好")], d)
    assert debug.load_realign_debug("ep01", str(b), d) is None


def test_a_missing_checkpoint_reads_as_none(d):
    for load in (debug.load_transcription_debug, debug.load_llm_correction_debug,
                 debug.load_ensemble_debug, debug.load_diarization_debug,
                 debug.load_vad_debug, debug.load_isolation_debug):
        assert load("never-written", d) is None

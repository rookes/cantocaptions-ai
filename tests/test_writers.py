"""Every output format, byte for byte.

Writers are the pipeline's contract with players and editing tools, so their exact output
is pinned here: timestamp formats, numbering, speaker labels, escaping, and one row per cue
for the row-oriented formats.
"""
import json

import pytest

from cantocaptions_ai.utils.output import get_writer, render_result

RESULT = {
    "language": "yue",
    "segments": [
        {"start": 1.0, "end": 2.5, "text": "你好", "speaker": "SPEAKER_00"},
        # Two lines (as the line-layout step writes them), a tab, and an SRT arrow.
        {"start": 3725.25, "end": 3727.0, "text": "今日\t天氣 --> 幾好\n我哋去公園"},
    ],
}
LABELS = {"speaker_labels": True}


def test_srt():
    assert render_result(RESULT, "srt", LABELS) == (
        "1\n00:00:01,000 --> 00:00:02,500\n[SPEAKER_00]: 你好\n\n"
        "2\n01:02:05,250 --> 01:02:07,000\n今日\t天氣 -> 幾好\n我哋去公園\n\n"
    )


def test_vtt():
    assert render_result(RESULT, "vtt", {}) == (
        "WEBVTT\n\n"
        "00:01.000 --> 00:02.500\n你好\n\n"
        "01:02:05.250 --> 01:02:07.000\n今日\t天氣 -> 幾好\n我哋去公園\n\n"
    )


def test_tsv_is_one_row_per_cue():
    # A two-line cue used to split its row in two, corrupting the file.
    assert render_result(RESULT, "tsv", {}) == (
        "start\tend\ttext\n"
        "1000\t2500\t你好\n"
        "3725250\t3727000\t今日 天氣 --> 幾好我哋去公園\n"
    )


def test_audacity_labels_are_one_row_per_cue():
    assert render_result(RESULT, "aud", LABELS) == (
        "1.0\t2.5\t[[SPEAKER_00]]你好\n"
        "3725.25\t3727.0\t今日 天氣 --> 幾好我哋去公園\n"
    )


def test_txt_and_json():
    assert render_result(RESULT, "txt", LABELS) == "[SPEAKER_00]: 你好\n今日\t天氣 --> 幾好\n我哋去公園\n"
    assert json.loads(render_result(RESULT, "json", {})) == RESULT


def test_speaker_labels_are_off_unless_asked_for():
    assert "SPEAKER_00" not in render_result(RESULT, "srt", {})


def test_empty_result_writes_empty_documents():
    empty = {"segments": [], "language": "yue"}
    assert render_result(empty, "srt", {}) == ""
    assert render_result(empty, "vtt", {}) == "WEBVTT\n\n"


def test_all_writes_every_format_under_the_name(tmp_path):
    get_writer("all", str(tmp_path))(RESULT, "s1/ep01", {})
    assert sorted(p.name for p in (tmp_path / "s1").iterdir()) == [
        "ep01.json", "ep01.srt", "ep01.tsv", "ep01.txt", "ep01.vtt"]


def test_all_cannot_render_to_one_string():
    with pytest.raises(ValueError):
        render_result(RESULT, "all", {})

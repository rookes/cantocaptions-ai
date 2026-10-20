"""Which embedded subtitle tracks are found, chosen and named (references/probe.py), and which
extracted track becomes the embedded reference (extract.best_chinese)."""
import pytest

from cantocaptions_ai.references.extract import ExtractedTrack, best_chinese
from cantocaptions_ai.references.probe import (
    SubtitleStream, canvas_for, is_chinese, lang_tag, ocr_language, parse_streams, script_of,
    select_streams,
)


def _stream(index, codec, language="", title="", forced=False):
    s = {"index": index, "codec_name": codec, "codec_type": "subtitle",
         "tags": {k: v for k, v in (("language", language), ("title", title)) if v},
         "disposition": {"forced": int(forced), "default": 0}}
    return s


# A Blu-ray remux: two ASS tracks, a Chinese PGS track; plus an untagged DVD track titled
# only in Chinese, a Simplified one, and teletext, which has nothing to extract.
PROBE = {"streams": [
    {"index": 0, "codec_name": "hevc", "codec_type": "video"},
    _stream(4, "ass", "eng", "Signs [OZC]"),
    _stream(5, "ass", "jpn", "Subtitles [OZC]"),
    _stream(6, "hdmv_pgs_subtitle", "chi", "CHT (HKBD PGS)"),
    _stream(7, "dvd_subtitle", "", "正體中文"),
    _stream(8, "dvd_subtitle", "chi", "Simplified Chinese [简体中文]"),
    _stream(9, "dvb_teletext", "chi"),
]}


@pytest.fixture
def streams():
    return parse_streams(PROBE)


def test_parse_keeps_only_subtitle_streams_with_their_kind(streams):
    assert [s.index for s in streams] == [4, 5, 6, 7, 8, 9]
    assert [s.kind for s in streams] == ["text", "text", "bitmap", "bitmap", "bitmap", "other"]
    assert streams[2].language == "chi" and streams[2].title == "CHT (HKBD PGS)"


def test_chinese_and_script(streams):
    by = {s.index: s for s in streams}
    assert not is_chinese(by[4]) and not is_chinese(by[5])
    assert is_chinese(by[6]) and script_of(by[6]) == "hant"
    assert is_chinese(by[7]) and script_of(by[7]) == "hant"      # untagged, titled 正體中文
    assert script_of(by[8]) == "hans"
    # "sc" inside a word is no script hint.
    assert script_of(SubtitleStream(1, "subrip", "text", "chi", "Disc 2")) is None


def test_tags_and_ocr_languages(streams):
    by = {s.index: s for s in streams}
    assert [lang_tag(by[i]) for i in (4, 5, 6, 8)] == ["en", "ja", "zh-Hant", "zh-Hans"]
    assert ocr_language(by[6]) == "chinese_cht"
    assert ocr_language(by[8]) == "ch"
    assert ocr_language(by[4]) == "en"
    assert ocr_language(SubtitleStream(1, "dvd_subtitle", "bitmap", "swe")) is None


def test_selection(streams):
    assert [s.index for s in select_streams(streams, "chinese")] == [6, 7, 8]
    assert [s.index for s in select_streams(streams, "all")] == [4, 5, 6, 7, 8]
    assert [s.index for s in select_streams(streams, "8, 4")] == [8, 4]
    with pytest.raises(ValueError, match="no subtitle stream"):
        select_streams(streams, "3")
    with pytest.raises(ValueError, match="dvb_teletext"):
        select_streams(streams, "9")
    with pytest.raises(ValueError, match="expected chinese"):
        select_streams(streams, "chi")


def test_canvas_is_the_formats_own_unless_the_video_is_bigger(streams):
    pgs, dvd = streams[2], streams[3]
    # A 1920x1040 encode of a Blu-ray: PGS is still authored at 1080 lines.
    assert canvas_for(pgs, (1920, 1040)) == (1920, 1080)
    assert canvas_for(pgs, (3840, 2160)) == (3840, 2160)
    assert canvas_for(dvd, None) == (720, 576)


def _track(index, method, cues, language="chi", title="", forced=False, path=True):
    return ExtractedTrack(
        stream={"index": index, "codec": "x", "kind": "text", "language": language,
                "title": title, "forced": forced, "default": False},
        method=method, lang="zh", path=f"t{index}.srt" if path else None, cues=cues)


def test_best_chinese_prefers_full_text_traditional_tracks():
    tracks = [
        _track(1, "ocr", 300, title="CHT"),
        _track(2, "text", 300, title="CHS"),
        _track(3, "text", 20, title="CHT Signs"),
        _track(4, "text", 290, title="繁體"),
        _track(5, "text", 400, language="eng"),
    ]
    assert best_chinese(tracks) == "t4.srt"
    # Text beats OCR whatever the script; a signs track only when nothing else is left.
    assert best_chinese(tracks[:3]) == "t2.srt"
    assert best_chinese([tracks[2], tracks[4]]) == "t3.srt"
    assert best_chinese([tracks[4], _track(6, "ocr", 0)]) is None

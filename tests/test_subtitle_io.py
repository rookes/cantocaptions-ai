"""Reading subtitle files: cue structure, line breaks, markup, and both formats.

The reader these cover replaced a suber-backed one that flattened every multi-line cue to a
single space-joined string. That is a content change, not a formatting one -- on the ReZero
fixture 5 of the 7 two-line cues are two-speaker dialogue pairs -- so the round trip of a
``\n`` is the property most of this file is about.
"""

import pytest

from cantocaptions_ai.utils.subtitles import (
    SubtitleFormatError,
    load_subtitle_file,
    read_subtitle_cues,
    strip_markup,
    subtitle_has_timings,
)

SRT = (
    "1\r\n"
    "00:00:05,407 --> 00:00:08,453\r\n"
    "弊喇，今次大件事喇\r\n"
    "\r\n"
    "2\r\n"
    "00:00:08,800 --> 00:00:10,542\r\n"
    "-講真嘠？\r\n"
    "-講真㗎！\r\n"
    "\r\n"
)


def _write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return str(path)


def test_parses_crlf_srt(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", SRT))
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(5.407)
    assert cues[0].end == pytest.approx(8.453)
    assert cues[0].text == "弊喇，今次大件事喇"


def test_two_line_cue_keeps_its_newline(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", SRT))
    assert cues[1].text == "-講真嘠？\n-講真㗎！"


def test_single_line_cue_gains_no_newline(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", SRT))
    assert "\n" not in cues[0].text


def test_utf8_bom_is_tolerated(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", SRT, encoding="utf-8-sig"))
    assert len(cues) == 2
    # A BOM left on the first line would make the cue number unparseable and eat the cue.
    assert cues[0].text == "弊喇，今次大件事喇"


def test_lf_line_endings_parse(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", SRT.replace("\r\n", "\n")))
    assert len(cues) == 2


def test_cue_indices_are_assigned_not_read(tmp_path):
    # An SRT in the wild is not reliably numbered from 1, and nothing downstream should
    # inherit a bad numbering.
    text = SRT.replace("1\r\n00:00:05", "7\r\n00:00:05")
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", text))
    assert [c.index for c in cues] == [0, 1]


def test_strips_html_tags_including_multi_character_ones():
    # suber's regex was `</?[^>]>`, which matches a one-character tag only: <i> went, and
    # <font color=...> was handed to the aligner as text.
    assert strip_markup('<i>hello</i>') == "hello"
    assert strip_markup('<font color="#ffffff">hello</font>') == "hello"
    assert strip_markup("<b>a</b><u>b</u>") == "ab"


def test_strips_ass_override_blocks():
    assert strip_markup(r"{\an8}top") == "top"
    assert strip_markup(r"{\pos(10,20)}x{\i1}y") == "xy"


def test_markup_stripping_leaves_ordinary_angle_brackets():
    # "<" followed by a non-letter is arithmetic or an emoticon, not a tag.
    assert strip_markup("a < b") == "a < b"
    assert strip_markup("3<5") == "3<5"


VTT = (
    "WEBVTT\n"
    "\n"
    "NOTE this is a comment\n"
    "spanning two lines\n"
    "\n"
    "cue-identifier\n"
    "00:00:05.407 --> 00:00:08.453 align:start line:90%\n"
    "first\n"
    "\n"
    "00:05.100 --> 00:06.200\n"
    "second\n"
)


def test_vtt_parses(tmp_path):
    # The old reader advertised VTT and raised ValueError from inside suber for it.
    cues = read_subtitle_cues(_write(tmp_path, "a.vtt", VTT))
    assert [c.text for c in cues] == ["first", "second"]


def test_vtt_cue_settings_are_not_read_as_timing(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.vtt", VTT))
    assert cues[0].end == pytest.approx(8.453)


def test_vtt_optional_hour(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.vtt", VTT))
    assert cues[1].start == pytest.approx(5.1)
    assert cues[1].end == pytest.approx(6.2)


def test_vtt_note_and_header_blocks_are_skipped(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.vtt", VTT))
    assert len(cues) == 2


def test_load_subtitle_file_flattens_for_legacy_callers(tmp_path):
    segments = load_subtitle_file(_write(tmp_path, "a.srt", SRT))
    assert segments[1]["text"] == "-講真嘠？ -講真㗎！"
    assert segments[0]["start"] == pytest.approx(5.407)


def test_empty_cues_are_dropped(tmp_path):
    text = SRT + "3\r\n00:00:11,000 --> 00:00:12,000\r\n\r\n"
    assert len(read_subtitle_cues(_write(tmp_path, "a.srt", text))) == 2


def test_a_file_with_no_cues_raises(tmp_path):
    with pytest.raises(SubtitleFormatError):
        read_subtitle_cues(_write(tmp_path, "a.srt", "just some text\nand more\n"))


def test_has_timings_true_for_srt(tmp_path):
    assert subtitle_has_timings(_write(tmp_path, "a.srt", SRT))


def test_has_timings_false_for_a_plain_transcript(tmp_path):
    assert not subtitle_has_timings(_write(tmp_path, "a.txt", "one\ntwo\n"))


def test_has_timings_tests_content_not_only_extension(tmp_path):
    # A transcript saved as .srt must not be "synced" against timings it does not have.
    assert not subtitle_has_timings(_write(tmp_path, "a.srt", "one\ntwo\n"))

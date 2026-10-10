"""Reading subtitle files: cue structure, line breaks, markup, and both formats.

The reader these cover replaced a suber-backed one that flattened every multi-line cue to a
single space-joined string. That is a content change, not a formatting one -- on the ReZero
fixture 5 of the 7 two-line cues are two-speaker dialogue pairs -- so the round trip of a
``\n`` is the property most of this file is about.
"""

import pytest

from cantocaptions_ai.utils.output import WriteSRT, WriteVTT
from cantocaptions_ai.utils.subtitles import (
    SubtitleFormatError,
    cue_segment,
    is_top_position,
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


# --- Position tags -------------------------------------------------------------------
#
# {\an8} is not styling: it says where the whole cue goes, and on a subtitle it almost always
# marks background speech running concurrently with the main line. Stripping it moved those
# cues onto the main line, where they collide with whatever is said there.

TOP_SRT = (
    "1\n00:00:01,000 --> 00:00:03,000\n主線\n\n"
    "2\n00:00:01,500 --> 00:00:02,500\n{\\an8}背景對白\n\n"
)


def test_a_leading_override_block_is_kept_apart_from_the_text(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", TOP_SRT))
    assert [c.style for c in cues] == ["", r"{\an8}"]
    assert cues[1].text == "背景對白"


def test_only_the_cue_opening_counts_as_its_style(tmp_path):
    srt = "1\n00:00:01,000 --> 00:00:02,000\n第一行\n{\\an8}第二行\n\n"
    cue = read_subtitle_cues(_write(tmp_path, "a.srt", srt))[0]
    assert cue.style == ""
    assert cue.text == "第一行\n第二行"


@pytest.mark.parametrize("style, top", [
    (r"{\an8}", True), (r"{\an7}", True), (r"{\an9}", True), (r"{\fs20\an8}", True),
    (r"{\a6}", True), (r"{\an2}", False), (r"{\an5}", False), (r"{\a10}", False),
    (r"{\i1}", False), ("", False),
])
def test_top_positions_are_recognised(style, top):
    assert is_top_position(style) is top


def test_cue_segment_carries_the_style(tmp_path):
    cues = read_subtitle_cues(_write(tmp_path, "a.srt", TOP_SRT))
    assert "style_tags" not in cue_segment(cues[0])
    assert cue_segment(cues[1])["style_tags"] == r"{\an8}"


def _written(writer, segments, tmp_path):
    writer(str(tmp_path))({"segments": segments, "language": "yue"}, "out", {})
    return (tmp_path / f"out.{writer.extension}").read_text(encoding="utf-8")


def test_srt_writer_puts_the_override_block_back(tmp_path):
    segments = [{"start": 1.0, "end": 3.0, "text": "主線"},
                {"start": 1.5, "end": 2.5, "text": "背景對白", "style_tags": r"{\an8}"}]
    out = _written(WriteSRT, segments, tmp_path)
    assert "\n{\\an8}背景對白\n" in out
    assert "\n主線\n" in out
    # ...and the round trip reads it back where it was.
    assert [c.style for c in read_subtitle_cues(str(tmp_path / "out.srt"))] == ["", r"{\an8}"]


def test_vtt_writer_says_top_with_a_cue_setting(tmp_path):
    segments = [{"start": 1.0, "end": 3.0, "text": "主線"},
                {"start": 1.5, "end": 2.5, "text": "背景對白", "style_tags": r"{\an8}"}]
    out = _written(WriteVTT, segments, tmp_path)
    assert "00:01.500 --> 00:02.500 line:0\n背景對白" in out
    assert "{" not in out
    assert "00:01.000 --> 00:03.000\n主線" in out

"""Bitmap subtitles from rendered frames to cues, without OCR or ffmpeg: frames are drawn here
and a fake recogniser stands in for PaddleOCR (references/raster.py, ocr.py, cues.py, worker.py)."""
import numpy as np
import pytest
from PIL import Image

from cantocaptions_ai.references import ocr, raster, worker
from cantocaptions_ai.references.cues import Cue, tidy, to_srt
from cantocaptions_ai.references.ocr import _join_boxes, _word_gap, normalize_text

W, H = 320, 180


def _frame(boxes=(), alpha=255, fill=(250, 250, 250), outline=(10, 10, 10)):
    """An RGBA canvas with each (x0, y0, x1, y1) box drawn as 'text': a fill inside a 2 px
    outline, the way subtitle bitmaps are drawn."""
    a = np.zeros((H, W, 4), np.uint8)
    for x0, y0, x1, y1 in boxes:
        a[y0:y1, x0:x1, :3] = outline
        a[y0 + 2:y1 - 2, x0 + 2:x1 - 2, :3] = fill
        a[y0:y1, x0:x1, 3] = alpha
    return a


def _write(tmp_path, frames):
    """[(seconds, rgba)] -> [(seconds, path)], named by microseconds as ffmpeg names them."""
    out = []
    for t, rgba in frames:
        path = tmp_path / f"{round(t * 1_000_000)}.png"
        Image.fromarray(rgba, "RGBA").save(path)
        out.append((t, str(path)))
    return out


LINE = [(40, 140, 280, 160)]
OTHER = [(60, 140, 260, 160)]


def test_events_follow_the_visible_text(tmp_path):
    frames = _write(tmp_path, [
        (1.0, _frame(LINE)), (1.999999, _frame(LINE)),        # shown, then re-sent
        (2.0, _frame()),                                       # cleared
        (3.0, _frame(OTHER)), (4.0, _frame(LINE)),             # replaced by another line
        (5.0, _frame(LINE, alpha=60)),                         # fading out: mostly transparent
        (6.0, _frame(OTHER)),                                  # still up when the stream ends
    ])
    events = raster.frame_events(frames, fallback=2.5)
    assert [(e.start, e.end) for e in events] == [(1.0, 2.0), (3.0, 4.0), (4.0, 5.0), (6.0, 8.5)]


def test_a_fade_is_one_event(tmp_path):
    frames = _write(tmp_path, [(1.0 + i / 10, _frame(LINE, alpha=a))
                               for i, a in enumerate((40, 100, 160, 220, 255, 255, 160, 40))])
    events = raster.frame_events(frames)
    assert [(e.start, round(e.end, 3)) for e in events] == [(1.2, 1.7)]


def test_lines_and_slivers():
    mask = np.zeros((100, 50), bool)
    mask[10:30] = True       # a line
    mask[34:37] = True       # a sliver just under it: the second stroke of 二
    mask[60:80] = True       # a second line
    assert raster.line_spans(mask) == [(10, 37), (60, 80)]
    assert raster.line_spans(np.zeros((10, 10), bool)) == []


def test_pieces_split_at_wide_gaps_only():
    mask = np.zeros((20, 300), bool)
    mask[:, 0:60] = True
    mask[:, 70:120] = True       # a 10 px gap: between two characters
    mask[:, 160:200] = True      # a 40 px gap (two line-heights): two speakers side by side
    assert raster.segment_spans(mask) == [("", 0, 120), (" ", 160, 200)]


@pytest.mark.parametrize("fill, outline", [((250, 250, 250), (10, 10, 10)),
                                           ((10, 10, 10), (250, 250, 250))])
def test_ink_keeps_the_fill_whichever_tone_it_is(fill, outline):
    rgba = _frame([(0, 0, 40, 20)], fill=fill, outline=outline)[0:20, 0:40]
    grey = np.asarray(raster._ink(rgba))
    pad = raster.PAD_PX
    assert grey[pad + 10, pad + 20] < 50          # the fill is dark ink
    assert grey[pad, pad + 20] > 200              # the outline is gone
    assert grey[0, 0] == 255                      # white border


def test_event_images_cut_lines_top_to_bottom(tmp_path):
    two = [(40, 20, 200, 40), (40, 140, 280, 160)]
    (t, path), = _write(tmp_path, [(1.0, _frame(two))])
    images = raster.event_images(raster.Event(1.0, 2.0, path))
    assert len(images.lines) == 2 and all(len(line) == 1 for line in images.lines)
    top, bottom = images.lines[0][0][1], images.lines[1][0][1]
    assert bottom.width > top.width
    assert raster.event_images(raster.Event(1.0, 2.0, _write(tmp_path, [(2.0, _frame())])[0][1])) is None


class FakeBackend(ocr.OcrBackend):
    """Reads a piece as its width, so the tests can tell pieces apart."""
    name = "fake"
    calls = 0

    def read(self, items):
        FakeBackend.calls += len(items)
        return ["\n".join("".join(j + f"w{im.width}" for j, im in line) for line in it.lines)
                for it in items]


def test_worker_reads_each_picture_once_and_writes_tidy_cues(tmp_path, monkeypatch):
    frames = _write(tmp_path, [(1.0, _frame(LINE)), (2.0, _frame()), (3.0, _frame(LINE)),
                               (4.0, _frame(OTHER)), (5.0, _frame())])
    monkeypatch.setattr(raster, "render_frames", lambda *a: frames)
    monkeypatch.setitem(ocr.OCR_BACKENDS, "fake", FakeBackend)
    FakeBackend.calls = 0
    out = tmp_path / "out.srt"
    result = worker.run({"backend": "fake", "jobs": [
        {"media": "x.mkv", "stream_index": 3, "canvas": [W, H], "lang": "ch", "out_srt": str(out)}]})
    assert result["jobs"][0]["ok"] and result["jobs"][0]["cues"] == 3
    assert FakeBackend.calls == 2          # the repeated picture is read once
    text = out.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in text and "00:00:04,000 --> 00:00:05,000" in text


def test_worker_reports_a_failed_job_and_carries_on(tmp_path, monkeypatch):
    def boom(*a):
        raise RuntimeError("no such stream")
    monkeypatch.setattr(raster, "render_frames", boom)
    monkeypatch.setitem(ocr.OCR_BACKENDS, "fake", FakeBackend)
    result = worker.run({"backend": "fake", "jobs": [
        {"media": "x", "stream_index": 3, "canvas": [W, H], "lang": "ch", "out_srt": str(tmp_path / "a.srt")}]})
    assert result["jobs"] == [{"ok": False, "error": "RuntimeError: no such stream"}]


def test_normalize_text():
    assert normalize_text("課長, 現場已經受控制") == "課長，現場已經受控制"
    assert normalize_text("你們是警察?") == "你們是警察？"
    assert normalize_text("調較到02:20後 等候") == "調較到02:20後 等候"
    assert normalize_text("OK, fine!") == "OK, fine!"
    assert normalize_text("少佐 ！") == "少佐！"


def test_join_boxes_reads_rows_left_to_right():
    res = [{"rec_texts": ["別再說話", "就別再聽、", "全員準備"],
            "rec_polys": [[[400, 5], [600, 5], [600, 50], [400, 50]],
                          [[0, 3], [380, 3], [380, 52], [0, 52]],
                          [[0, 70], [300, 70], [300, 115], [0, 115]]]}]
    assert _join_boxes(res) == "就別再聽、別再說話\n全員準備"


def test_word_gap_cuts_at_the_word_space_nearest_the_middle():
    ink = np.zeros((10, 100), bool)
    for x0, x1 in ((0, 20), (22, 40), (48, 70), (72, 90)):     # letters 2 apart, words 8 apart
        ink[:, x0:x1] = True
    assert _word_gap(ink) == 44
    assert _word_gap(np.zeros((10, 10), bool)) is None


def test_tidy_merges_repeats_and_never_emits_an_empty_or_inverted_cue():
    cues = tidy([Cue(5.0, 6.0, "b"), Cue(1.0, 2.0, "a"), Cue(2.02, 3.0, "a"),
                 Cue(4.0, 4.0, "c"), Cue(7.0, 8.0, "  ")])
    assert [(c.start, c.end, c.text) for c in cues] == [(1.0, 3.0, "a"), (4.0, 4.02, "c"),
                                                        (5.0, 6.0, "b")]
    assert to_srt(cues).startswith("1\n00:00:01,000 --> 00:00:03,000\na\n")

"""Embedded reference extraction on a real (generated) media file, through the library call,
the pipeline step's checkpoints and the CLI. No OCR runs: image tracks are faked at the probe
and the worker, since ffmpeg cannot encode one from text."""
import json
import shutil
import subprocess
import sys

import pytest

from cantocaptions_ai.references import extract
from cantocaptions_ai.references.probe import SubtitleStream

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

SRT_ZH = "1\n00:00:01,000 --> 00:00:02,500\n你好嗎？\n\n2\n00:00:03,000 --> 00:00:04,000\n走吧\n第二行\n"
ASS_EN = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Italic, Alignment, MarginL, MarginR, MarginV
Style: Default,Arial,20,&H00FFFFFF,0,0,2,10,10,10

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,{\\an8}How are you?
Dialogue: 1,0:00:01.00,0:00:02.50,Default,,0,0,0,,{\\an8}How are you?
Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,Let's go
"""


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("refs_media")
    (d / "zh.srt").write_text(SRT_ZH, encoding="utf-8")
    (d / "en.ass").write_text(ASS_EN, encoding="utf-8")
    out = d / "episode.mkv"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=black:s=320x180:d=5",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5",
         "-i", str(d / "zh.srt"), "-i", str(d / "en.ass"),
         "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map", "3:s",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-c:s", "copy",
         "-metadata:s:s:0", "language=chi", "-metadata:s:s:0", "title=繁體",
         "-metadata:s:s:1", "language=eng", "-shortest", str(out)],
        check=True)
    return str(out)


def _cues(path):
    from cantocaptions_ai.utils.subtitles import read_subtitle_cues
    return [(c.start, c.end, c.text) for c in read_subtitle_cues(path)]


def test_text_tracks_round_trip(media, tmp_path):
    result = extract.extract_references(media, str(tmp_path), "ep", tracks="all", ocr="off")
    paths = {t.lang: t.path for t in result.tracks}
    assert set(paths) == {"zh-Hant", "en"}
    assert _cues(paths["zh-Hant"]) == [(1.0, 2.5, "你好嗎？"), (3.0, 4.0, "走吧\n第二行")]
    # Styling dropped, the duplicate layer collapsed.
    assert _cues(paths["en"]) == [(1.0, 2.5, "How are you?"), (3.0, 4.0, "Let's go")]
    assert result.best_chinese == paths["zh-Hant"]
    manifest = json.loads((tmp_path / "ep.references.json").read_text(encoding="utf-8"))
    assert manifest["best_chinese"] == paths["zh-Hant"]
    assert [s["index"] for s in manifest["streams"]] == [2, 3]


def test_chinese_is_the_default_selection(media, tmp_path):
    result = extract.extract_references(media, str(tmp_path), "ep")
    assert [t.lang for t in result.tracks] == ["zh-Hant"]
    assert [s["reason"] for s in result.skipped] == ["not selected (reference_tracks)"]


def _with_a_bitmap_track(monkeypatch, media):
    real = extract.probe_subtitle_streams(media)
    pgs = SubtitleStream(7, "hdmv_pgs_subtitle", "bitmap", "chi", "CHT PGS")
    monkeypatch.setattr(extract, "probe_subtitle_streams", lambda m: real + [pgs])


def test_image_tracks_are_skipped_with_a_reason_when_ocr_is_missing(media, tmp_path, monkeypatch):
    _with_a_bitmap_track(monkeypatch, media)
    monkeypatch.setattr(extract, "ocr_available", lambda backend: "PaddleOCR is not installed (...)")
    result = extract.extract_references(media, str(tmp_path), "ep")
    assert [t.stream["index"] for t in result.tracks] == [2]
    assert result.skipped[-1]["reason"].startswith("bitmap track; PaddleOCR is not installed")


def test_image_tracks_go_to_the_worker_in_one_batch(media, tmp_path, monkeypatch):
    _with_a_bitmap_track(monkeypatch, media)
    monkeypatch.setattr(extract, "ocr_available", lambda backend: None)
    seen = {}

    def fake_worker(request, status, labels, reporter=None):
        seen.update(request)
        for job in request["jobs"]:
            with open(job["out_srt"], "w", encoding="utf-8") as f:
                f.write("1\n00:00:01,000 --> 00:00:02,000\n字幕\n")
        return {"jobs": [{"ok": True, "cues": 1, "seconds": 0.1}]}

    monkeypatch.setattr(extract, "_run_worker", fake_worker)
    result = extract.extract_references(media, str(tmp_path), "ep", device="cpu")
    (job,) = seen["jobs"]
    assert job["stream_index"] == 7 and job["lang"] == "chinese_cht"
    assert job["canvas"] == [1920, 1080] and seen["device"] == "cpu"
    ocr_track = next(t for t in result.tracks if t.method == "ocr")
    assert ocr_track.cues == 1 and ocr_track.path.endswith("ep.ref.zh-Hant.7.srt")
    # The text track still wins as the reference.
    assert result.best_chinese.endswith("ep.ref.zh-Hant.srt")


def _cfg(tmp_path, **kw):
    from cantocaptions_ai.pipeline.config import PipelineConfig
    return PipelineConfig(**{"device": "cpu", "output_dir": str(tmp_path / "out"), **kw})


def test_step_checkpoints_and_replays(media, tmp_path, monkeypatch):
    from cantocaptions_ai.references import step
    from cantocaptions_ai.utils.log_utils import TranscriptionSummary
    debug = str(tmp_path / "debug")
    cfg = _cfg(tmp_path, extract_references=True, debug_dir=debug)
    first = step.extract_for_inputs(cfg, [media], {media: "ep"}, TranscriptionSummary(enabled=False))
    assert (tmp_path / "debug" / "ep" / "references" / "ep.ref.zh-Hant.srt").is_file()

    def no_extraction(*a, **k):
        raise AssertionError("a current checkpoint must be reloaded, not extracted again")
    monkeypatch.setattr(extract, "extract_references", no_extraction)
    replay = _cfg(tmp_path, extract_references=True, load_debug_dir=debug,
                  output_dir=str(tmp_path / "out2"))
    again = step.extract_for_inputs(replay, [media], {media: "ep"}, TranscriptionSummary(enabled=False))
    assert _cues(again[media].best_chinese) == _cues(first[media].best_chinese)
    assert str(tmp_path / "out2") in again[media].best_chinese

    # Other settings: the checkpoint is stale, so extraction runs (and here, fails loudly).
    stale = _cfg(tmp_path, extract_references=True, load_debug_dir=debug, reference_tracks="all",
                 output_dir=str(tmp_path / "out3"))
    with pytest.raises(AssertionError, match="must be reloaded"):
        step.extract_for_inputs(stale, [media], {media: "ep"}, TranscriptionSummary(enabled=False))


def test_embedded_reference_needs_a_chinese_track(tmp_path):
    from cantocaptions_ai.errors import InputError
    from cantocaptions_ai.pipeline.transcribe import _use_embedded_reference
    with pytest.raises(InputError, match="no Chinese subtitle track"):
        _use_embedded_reference(_cfg(tmp_path), extract.ReferenceSet(media="m", name="ep"), "ep")
    cfg = _use_embedded_reference(_cfg(tmp_path), extract.ReferenceSet(
        media="m", name="ep", best_chinese="ep.ref.zh-Hant.srt"), "ep")
    assert cfg.reference_subtitle == "ep.ref.zh-Hant.srt"


def _cli(monkeypatch, *argv):
    from cantocaptions_ai import __main__ as main
    monkeypatch.setattr(sys, "argv", ["cantocaptions", *argv])
    main.cli()


def test_references_only_runs_no_pipeline(media, tmp_path, monkeypatch):
    from cantocaptions_ai.pipeline import transcribe
    monkeypatch.setattr(transcribe, "_execute_pipeline",
                        lambda *a, **k: pytest.fail("--references_only must not run the pipeline"))
    _cli(monkeypatch, media, "--references_only", "--reference_tracks", "all",
         "-o", str(tmp_path), "--print_progress", "False")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "episode.ref.en.srt", "episode.ref.zh-Hant.srt", "episode.references.json"]


@pytest.mark.parametrize("argv, message", [
    (["--references_only", "--reference_tracks", "chi"], "reference_tracks must be"),
    (["--reference_subtitle", "embedded", "--asr_context", "--reference_timing", "subtitle"],
     "follows the media's timeline"),
    (["--reference_subtitle", "embedded"], "requires llm_correction, asr_context or proofread"),
])
def test_bad_reference_settings_are_refused(media, tmp_path, monkeypatch, capsys, argv, message):
    with pytest.raises(SystemExit):
        _cli(monkeypatch, media, *argv, "-o", str(tmp_path), "--device", "cpu")
    assert message in capsys.readouterr().err


def test_embedded_reference_is_one_inputs(media, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli(monkeypatch, media, media, "--reference_subtitle", "embedded", "--asr_context",
             "-o", str(tmp_path), "--device", "cpu")
    assert "pass one file" in capsys.readouterr().err

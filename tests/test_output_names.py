"""One name per input, mirrored from --input_dir, keys every output and debug file.

Keyed by stem, ``s1/ep01.mkv`` and ``s2/ep01.mkv`` from a recursive run wrote the same
``ep01.srt`` and shared one debug cache, so the second silently replaced (or replayed) the
first.
"""
import os

import numpy as np
import pytest

from cantocaptions_ai.errors import ConfigError
from cantocaptions_ai.pipeline.transcribe import _merge_and_write
from cantocaptions_ai.utils.debug import load_vad_debug, write_vad_debug
from cantocaptions_ai.utils.output import get_writer, output_names
from cantocaptions_ai.utils.schema import item_name


def test_direct_files_are_named_by_stem():
    assert output_names(["/a/ep01.mkv", "/b/ep02.mp4"]) == {
        "/a/ep01.mkv": "ep01", "/b/ep02.mp4": "ep02",
    }


def test_input_dir_names_mirror_its_subfolders(tmp_path):
    root = str(tmp_path)
    paths = [os.path.join(root, "s1", "ep01.mkv"), os.path.join(root, "s2", "ep01.mkv"),
             os.path.join(root, "extra.wav")]
    assert list(output_names(paths, root).values()) == ["s1/ep01", "s2/ep01", "extra"]


def test_direct_files_that_would_clash_are_rejected_with_a_hint():
    with pytest.raises(ConfigError, match="--input_dir"):
        output_names(["/a/ep01.mkv", "/b/ep01.mkv"])


def test_same_stem_different_extension_still_clashes(tmp_path):
    root = str(tmp_path)
    with pytest.raises(ConfigError, match="ep01"):
        output_names([os.path.join(root, "ep01.mkv"), os.path.join(root, "ep01.mp4")], root)


def test_item_name_prefers_the_assigned_name():
    assert item_name({"audio_path": "/x/s1/ep01.mkv", "name": "s1/ep01"}) == "s1/ep01"
    assert item_name({"audio_path": "/x/s1/ep01.mkv"}) == "ep01"


def test_writer_creates_the_mirrored_subfolder(tmp_path):
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "你好"}], "language": "yue"}
    get_writer("srt", str(tmp_path))(result, "s1/ep01", {})
    assert (tmp_path / "s1" / "ep01.srt").read_text(encoding="utf-8").startswith("1\n")


def test_debug_checkpoints_are_separate_per_name(tmp_path):
    seg = lambda v: [{"start": 0.0, "end": 1.0, "audio": np.full(160, v, dtype=np.float32)}]
    write_vad_debug("s1/ep01", seg(0.1), str(tmp_path))
    write_vad_debug("s2/ep01", seg(0.2), str(tmp_path))
    assert load_vad_debug("s1/ep01", str(tmp_path))[0]["audio"][0] == pytest.approx(0.1, abs=1e-4)
    assert load_vad_debug("s2/ep01", str(tmp_path))[0]["audio"][0] == pytest.approx(0.2, abs=1e-4)


def test_merge_and_write_writes_each_item_under_its_own_name(tmp_path):
    def item(season, text):
        return {
            "audio_path": f"/media/{season}/ep01.mkv", "name": f"{season}/ep01",
            "result": {"language": "yue", "segments": [
                {"start": 0.0, "end": 1.0, "text": text, "words": []}]},
        }

    out_dir, debug_dir = tmp_path / "out", tmp_path / "debug"
    results = _merge_and_write(
        [item("s1", "第一季"), item("s2", "第二季")], get_writer("srt", str(out_dir)), "yue",
        0.12, 0.04, {}, debug_dir=str(debug_dir), collect=True,
    )
    assert "第一季" in (out_dir / "s1" / "ep01.srt").read_text(encoding="utf-8")
    assert "第二季" in (out_dir / "s2" / "ep01.srt").read_text(encoding="utf-8")
    assert (debug_dir / "s1" / "ep01" / "pre_cleaning" / "ep01.srt").is_file()
    assert [r["name"] for r in results] == ["s1/ep01", "s2/ep01"]
    assert [r["audio_path"] for r in results] == ["/media/s1/ep01.mkv", "/media/s2/ep01.mkv"]


def test_stage_packs_keep_the_name():
    """A stage that rebuilt the item dict from scratch would drop the name mid-pipeline."""
    from cantocaptions_ai.pipeline.asr import QwenPipeline
    from cantocaptions_ai.pipeline.vad import VadProcessor

    item = {"audio_path": "/m/s1/ep01.mkv", "name": "s1/ep01", "vad_segments": []}
    assert VadProcessor._pack(item, [])["name"] == "s1/ep01"
    assert QwenPipeline._pack(item, {"segments": []})["name"] == "s1/ep01"

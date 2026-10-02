"""Config sections: every setting belongs to one, and the flat names stay the only names.

The sections group settings in --help, in .cfg files ([vad] blocks beside or instead of
[pipeline]) and for callers that want one stage's settings; they never rename anything.
"""
import contextlib
import io
import json
from dataclasses import fields
from pathlib import Path

import pytest

from cantocaptions_ai.__main__ import build_parser
from cantocaptions_ai.pipeline.cli_config import describe_config, load_cfg_file
from cantocaptions_ai.pipeline.config import CONFIG_SECTIONS, SECTION_TITLES, PipelineConfig


@pytest.fixture(scope="module")
def parser():
    return build_parser()


def _load(tmp_path, body, parser):
    path = tmp_path / "x.cfg"
    path.write_text(body, encoding="utf-8")
    return load_cfg_file(path, parser)


def _refused(tmp_path, body, parser) -> str:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
        _load(tmp_path, body, parser)
    return err.getvalue()


def test_every_setting_is_in_exactly_one_section():
    listed = [key for keys in CONFIG_SECTIONS.values() for key in keys]
    assert sorted(listed) == sorted(f.name for f in fields(PipelineConfig))
    assert len(listed) == len(set(listed))
    assert set(SECTION_TITLES) == set(CONFIG_SECTIONS)


def test_each_flag_sits_in_its_sections_help_group(parser):
    group_of = {a.dest: g.title for g in parser._action_groups for a in g._group_actions}
    for section, keys in CONFIG_SECTIONS.items():
        for key in keys:
            assert group_of[key] == SECTION_TITLES[section], key


def test_section_views_and_lookup():
    cfg = PipelineConfig(device="cpu", vad_onset=0.3)
    assert PipelineConfig.section_of("vad_onset") == "vad"
    assert cfg.section("vad")["vad_onset"] == 0.3
    assert set(cfg.section("cues")) == set(CONFIG_SECTIONS["cues"])
    with pytest.raises(TypeError):
        cfg.section("vad")["vad_onset"] = 0.5
    with pytest.raises(KeyError, match="no pipeline setting"):
        PipelineConfig.section_of("vad_threshold")


FLAT = """[pipeline]
vad_onset = 0.3
min_cue_duration = 0.4
batch_size = 12
"""

SECTIONED = """[vad]
vad_onset = 0.3

[cues]
min_cue_duration = 0.4

[inference]
batch_size = 12
"""

MIXED = """[pipeline]
batch_size = 12

[vad]
vad_onset = 0.3

[cues]
min_cue_duration = 0.4
"""


def test_flat_sectioned_and_mixed_files_read_the_same(tmp_path, parser):
    flat = _load(tmp_path, FLAT, parser)
    assert flat == {"vad_onset": 0.3, "min_cue_duration": 0.4, "batch_size": 12}
    assert _load(tmp_path, SECTIONED, parser) == flat
    assert _load(tmp_path, MIXED, parser) == flat


def test_a_setting_in_the_wrong_section_names_the_right_one(tmp_path, parser):
    msg = _refused(tmp_path, "[alignment]\nmin_cue_duration = 0.4\n", parser)
    assert "'min_cue_duration' belongs in [cues], not [alignment]" in msg


def test_a_setting_given_twice_is_refused(tmp_path, parser):
    msg = _refused(tmp_path, "[pipeline]\nvad_onset = 0.2\n[vad]\nvad_onset = 0.3\n", parser)
    assert "'vad_onset' is set more than once" in msg


def test_an_unknown_section_is_refused(tmp_path, parser):
    assert "unknown section(s) [vads]" in _refused(tmp_path, "[vads]\nvad_onset = 0.3\n", parser)


def test_flatten_accepts_section_tables_and_flat_keys():
    nested = {"vad": {"vad_onset": 0.3}, "language": "yue", "cues": {"merge_gap": 0.2}}
    assert PipelineConfig.flatten(nested) == {"vad_onset": 0.3, "language": "yue",
                                              "merge_gap": 0.2}
    with pytest.raises(ValueError, match="belongs in \\[cues\\]"):
        PipelineConfig.flatten({"vad": {"merge_gap": 0.2}})
    with pytest.raises(ValueError, match="set twice"):
        PipelineConfig.flatten({"vad_onset": 0.2, "vad": {"vad_onset": 0.3}})
    with pytest.raises(ValueError, match="unknown pipeline setting"):
        PipelineConfig.flatten({"vad_threshold": 0.3})


def test_describe_config_is_plain_data_in_help_order(parser):
    described = describe_config(parser)
    assert [s["name"] for s in described] == list(CONFIG_SECTIONS)
    json.dumps(described)   # plain data, ready to hand to a UI
    vad = {s["name"]: s for s in described[[d["name"] for d in described].index("vad")]["settings"]}
    assert vad["vad_method"]["choices"] == ["pyannote", "silero"]
    assert vad["vad_onset"]["flag"] == "--vad_onset"
    assert vad["vad_onset"]["type"] == "float"
    assert vad["vad_onset"]["default"] == PipelineConfig.defaults()["vad_onset"]
    assert all(s["help"] for section in described for s in section["settings"])

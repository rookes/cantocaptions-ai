"""A debug checkpoint is only replayed under the settings that produced it.

Before this, ``--load_debug_dir`` keyed checkpoints by file name alone: change the ASR
model or a VAD threshold and a replay would load the old stage output as if it were current.
"""
import logging

import pytest

from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.utils import checkpoints as cp
from cantocaptions_ai.utils.checkpoints import (
    checkpoint_is_current,
    checkpoint_settings,
    fingerprint,
    write_checkpoint_meta,
)
from cantocaptions_ai.utils.model_utils import PipelineStage, partition_by_cache


def _fingerprints(**overrides):
    cfg = PipelineConfig(device="cpu", **overrides)
    return {stage: fingerprint(s) for stage, s in checkpoint_settings(cfg).items()}


def _changed(**overrides):
    base, new = _fingerprints(), _fingerprints(**overrides)
    return {stage for stage in base if base[stage] != new[stage]}


class TestWhatInvalidatesWhat:
    def test_vad_settings_invalidate_everything_downstream(self):
        assert _changed(vad_onset=0.3) == set(cp.STAGES)

    def test_asr_model_invalidates_transcription_and_its_consumers_only(self):
        assert _changed(model="Qwen3-ASR") == {"transcription", "llm_correction"}

    def test_vocal_isolation_invalidates_what_reads_its_audio(self):
        assert _changed(vocal_isolation_method="mbroformer") == set(cp.STAGES) - {"vad"}

    def test_the_default_model_spelled_out_is_the_same_checkpoint(self):
        # model=None means the language's own model; naming it must not invalidate caches.
        assert _changed(model="cantocaptions-cantonese-ASR") == set()
        assert _changed(align_model="alvanlii/wav2vec2-BERT-cantonese") == set()

    def test_throughput_knobs_invalidate_nothing(self):
        # A different batch size must not throw away an hour of ASR.
        assert _changed(batch_size=24, align_batch_size=6, diarize_batch_size=8,
                        vram_checks=True) == set()

    def test_context_settings_only_count_when_context_is_on(self):
        assert _changed(asr_context_template="bare") == set()
        on = dict(asr_context=True, reference_subtitle="ref.srt")
        base = _fingerprints(**on)
        assert _fingerprints(**on, asr_context_template="bare")["transcription"] != base["transcription"]


class TestCheckpointIsCurrent:
    def settings(self, **overrides):
        return checkpoint_settings(PipelineConfig(device="cpu", **overrides))["transcription"]

    def test_matching_settings_are_current(self, tmp_path):
        write_checkpoint_meta(str(tmp_path), "ep01", "transcription", self.settings())
        assert checkpoint_is_current(str(tmp_path), "ep01", "transcription", self.settings())

    def test_changed_settings_are_stale_and_the_change_is_named(self, tmp_path, caplog):
        write_checkpoint_meta(str(tmp_path), "ep02", "transcription", self.settings())
        with caplog.at_level(logging.WARNING):
            assert not checkpoint_is_current(
                str(tmp_path), "ep02", "transcription", self.settings(model="Qwen3-ASR"))
        assert "transcription.model" in caplog.text

    def test_a_checkpoint_without_metadata_is_accepted(self, tmp_path):
        # Written before checkpoints recorded their settings; reused, with a warning.
        assert checkpoint_is_current(str(tmp_path), "ep03", "vad", self.settings())

    def test_no_expected_settings_disables_the_check(self, tmp_path):
        write_checkpoint_meta(str(tmp_path), "ep04", "vad", {"vad.vad_onset": 0.1})
        assert checkpoint_is_current(str(tmp_path), "ep04", "vad", None)


class _EchoStage(PipelineStage):
    """A stage whose 'model' returns its input, with an in-memory checkpoint store."""

    debug_stage = "transcription"
    store: dict = {}

    def __init__(self):
        self.calls = 0

    def process(self, input, *, progress_callback=None):
        self.calls += 1
        return f"computed:{input}"

    @staticmethod
    def read_debug(name, debug_dir):
        return _EchoStage.store.get((debug_dir, name))

    @staticmethod
    def write_debug(name, result, debug_dir):
        _EchoStage.store[(debug_dir, name)] = result

    @staticmethod
    def _extract(item):
        return item["audio_path"]

    @staticmethod
    def _pack(item, result):
        return {**item, "result": result}


@pytest.fixture
def stage():
    _EchoStage.store = {}
    return _EchoStage()


def _item(**overrides):
    cfg = PipelineConfig(device="cpu", **overrides)
    return {"audio_path": "/m/ep01.mkv", "name": "ep01", "checkpoints": checkpoint_settings(cfg)}


def test_stage_replays_a_checkpoint_made_under_the_same_settings(stage, tmp_path):
    d = str(tmp_path)
    stage.run([_item()], debug_dir=d)
    stage.run([_item()], load_debug_dir=d)
    assert stage.calls == 1


def test_stage_recomputes_a_checkpoint_made_under_other_settings(stage, tmp_path):
    d = str(tmp_path)
    stage.run([_item()], debug_dir=d)
    out = stage.run([_item(model="Qwen3-ASR")], load_debug_dir=d)
    assert stage.calls == 2
    assert out[0]["result"] == "computed:/m/ep01.mkv"


def test_partition_by_cache_treats_stale_checkpoints_as_missing(stage, tmp_path):
    d = str(tmp_path)
    stage.run([_item()], debug_dir=d)
    cached, to_compute = partition_by_cache([_item(language="zh")], stage, d)
    assert cached == {} and [i for i, _ in to_compute] == [0]

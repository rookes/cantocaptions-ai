"""The orchestrator's stage list (pipeline/stages.py): which stages a config runs, what the
plan line says about the cache, and that a caller can insert a stage of its own.

The end-to-end goldens (tests/test_pipeline_e2e.py) are what pin the stages' *output*; this
file is about the list itself.
"""
import logging
import shutil
from pathlib import Path

import pytest

from cantocaptions_ai.languages import get_language_pack
from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.pipeline.stages import (
    RunContext,
    Stage,
    VadStage,
    build_stages,
    describe_plan,
)
from cantocaptions_ai.utils.checkpoints import checkpoint_settings, write_checkpoint_meta
from cantocaptions_ai.utils.log_utils import TranscriptionSummary


def ctx_for(audio=("a.wav",), cleaner=None, **cfg_kw) -> RunContext:
    cfg = PipelineConfig(**{"device": "cpu", **cfg_kw})
    pack = get_language_pack(cfg.language)
    return RunContext(
        cfg=cfg, audio_paths=list(audio), name_of={p: Path(p).stem for p in audio},
        checkpoints=checkpoint_settings(cfg), pack=pack, profile=pack.resolve(cfg.model),
        align_model_name=cfg.align_model or pack.default_align_model,
        summary=TranscriptionSummary(enabled=False), cleaner=cleaner,
    )


def names(ctx) -> list:
    return [type(stage).__name__ for stage in build_stages(ctx)]


# --- which stages a config runs -------------------------------------------------------

def test_the_default_run():
    assert names(ctx_for()) == ["VadStage", "TranscriptionStage", "AlignmentStage"]


def test_cleaning_adds_its_pre_alignment_steps():
    assert names(ctx_for(cleaner=object())) == [
        "VadStage", "TranscriptionStage", "PreAlignCleanStage", "AlignmentStage"]


def test_no_align_takes_the_asr_timings_instead():
    assert names(ctx_for(no_align=True)) == ["VadStage", "TranscriptionStage", "TimestampsStage"]


def test_vocal_isolation_and_diarization():
    assert names(ctx_for(vocal_isolation_method="mbroformer", diarize=True)) == [
        "VadStage", "VocalIsolationStage", "TranscriptionStage", "AlignmentStage",
        "DiarizationStage", "SpeakerAssignStage"]


def test_the_optional_second_opinions_follow_transcription():
    ctx = ctx_for(ensemble_model="tiny", llm_correction=True)
    ctx.reference_cues = [{"start": 0.0, "end": 1.0, "text": "x"}]
    assert names(ctx) == [
        "VadStage", "TranscriptionStage", "EnsembleStage", "ReferenceMatchStage",
        "LlmCorrectionStage", "AlignmentStage"]


def test_asr_context_needs_a_reference():
    assert "AsrContextStage" not in names(ctx_for(asr_context=True))
    ctx = ctx_for(asr_context=True)
    ctx.reference_cues = [{"start": 0.0, "end": 1.0, "text": "x"}]
    assert names(ctx)[:3] == ["VadStage", "AsrContextStage", "TranscriptionStage"]


def test_acoustic_realign_replaces_asr_through_alignment():
    ctx = ctx_for(realign="t.srt", realign_anchor="acoustic", cleaner=object(), diarize=True)
    assert names(ctx) == ["VadStage", "RealignAcousticStage", "DiarizationStage",
                          "SpeakerAssignStage"]


def test_asr_anchored_realign_rejoins_before_alignment():
    ctx = ctx_for(realign="t.srt", realign_anchor="asr", cleaner=object())
    assert names(ctx) == ["VadStage", "TranscriptionStage", "RealignAsrStage", "AlignmentStage"]


# --- the plan line and the cache ------------------------------------------------------

def _checkpoint(ctx, path, stage, tmp_path, marker="segments.json"):
    """Make *stage* look cached for *path* under tmp_path, as a previous run would leave it."""
    name = ctx.name_of[path]
    stage_dir = tmp_path / name / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / marker).write_text("{}")
    write_checkpoint_meta(str(tmp_path), name, stage, ctx.checkpoints[stage])


def test_the_plan_says_what_will_be_computed_and_what_read_back(tmp_path):
    ctx = ctx_for(load_debug_dir=str(tmp_path))
    _checkpoint(ctx, "a.wav", "vad", tmp_path)
    plan = describe_plan(ctx, build_stages(ctx))
    assert plan == "VAD [cached] → Transcription [compute] → Alignment"

    _checkpoint(ctx, "a.wav", "transcription", tmp_path, marker="result.json")
    assert "Transcription [cached]" in describe_plan(ctx, build_stages(ctx))


def test_one_uncached_input_means_the_stage_computes(tmp_path):
    ctx = ctx_for(audio=("a.wav", "b.wav"), load_debug_dir=str(tmp_path))
    _checkpoint(ctx, "a.wav", "vad", tmp_path)
    assert describe_plan(ctx, build_stages(ctx)).startswith("VAD [compute]")


def test_vad_is_skipped_where_vocal_isolation_is_cached(tmp_path, monkeypatch):
    ctx = ctx_for(vocal_isolation_method="mbroformer", load_debug_dir=str(tmp_path))
    _checkpoint(ctx, "a.wav", "vocal_isolation", tmp_path)
    assert describe_plan(ctx, build_stages(ctx)).startswith("VAD [cached] → Vocal isolation [cached]")

    from cantocaptions_ai.pipeline import vad

    def no_vad(**kwargs):
        raise AssertionError("VAD must not load for a file its isolation cache covers")

    monkeypatch.setattr(vad, "load_vad", no_vad)
    items = [{"audio_path": "a.wav", "name": "a"}]
    assert VadStage().run(ctx, items) is items


# --- a caller's own stage -------------------------------------------------------------

class _Witness(Stage):
    """Records the segments it is handed and passes the items on untouched."""

    name = "Witness"

    def __init__(self):
        self.seen = []

    def run(self, ctx, items):
        self.seen = [dict(seg) for item in items for seg in item["result"]["segments"]]
        return items


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_a_stage_inserted_through_the_hook_runs_in_place(tmp_path, monkeypatch, caplog):
    from _pipeline_fakes import ScriptedAudio, install
    from cantocaptions_ai.pipeline.transcribe import _execute_pipeline, validate_config

    scripted = ScriptedAudio()
    install(monkeypatch, scripted)
    media = scripted.write_wav(tmp_path / "episode.wav")
    cfg = PipelineConfig(device="cpu", print_progress=False, audio_normalize=False,
                         output_dir=str(tmp_path / "out"), no_clean_text=True)
    validate_config(cfg)

    def plain():
        return _execute_pipeline([media], cfg, collect=True)[0]["result"]["segments"]

    witness = _Witness()

    def after_transcription(ctx, default):
        i = [type(s).__name__ for s in default].index("TranscriptionStage")
        return [*default[:i + 1], witness, *default[i + 1:]]

    baseline = plain()
    with caplog.at_level(logging.INFO):
        watched = _execute_pipeline([media], cfg, collect=True,
                                    stages=after_transcription)[0]["result"]["segments"]
    assert "Transcription [compute] → Witness → Alignment" in caplog.text
    # It ran between ASR and alignment: it saw the ASR text, with no word timings yet.
    assert [seg["text"] for seg in witness.seen] == [t for _, t in scripted.script]
    assert not any(seg.get("words") for seg in witness.seen)
    assert watched == baseline


# --- the plan a progress sink receives -------------------------------------------------

def test_plan_entries_carry_stable_keys_and_what_will_run(tmp_path):
    from cantocaptions_ai.pipeline.stages import plan_entries

    ctx = ctx_for()
    assert plan_entries(ctx, build_stages(ctx)) == [
        {"key": "vad", "label": "VAD", "timed": True, "cached": False},
        {"key": "transcription", "label": "Transcription", "timed": True, "cached": False},
        {"key": "alignment", "label": "Alignment", "timed": True, "cached": None},
    ]

    cached = ctx_for(load_debug_dir=str(tmp_path))
    _checkpoint(cached, "a.wav", "vad", tmp_path)
    _checkpoint(cached, "a.wav", "transcription", tmp_path, marker="result.json")
    vad, asr, _ = plan_entries(cached, build_stages(cached))
    assert (vad["cached"], vad["timed"]) == (True, True)   # VAD's timer covers its cache load
    assert (asr["cached"], asr["timed"]) == (True, False)  # a cached model stage runs no timer


def test_every_stage_has_its_own_key():
    from cantocaptions_ai.pipeline.stages import DEFAULT_STAGES

    keys = [cls.key for cls in DEFAULT_STAGES]
    assert all(keys) and len(keys) == len(set(keys))


def test_realign_and_diarize_plans():
    from cantocaptions_ai.pipeline.stages import plan_entries

    ctx = ctx_for(realign="t.srt", realign_anchor="acoustic", diarize=True)
    assert [e["key"] for e in plan_entries(ctx, build_stages(ctx))] == [
        "vad", "realign", "diarization", "speaker_assign"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_a_sink_with_plan_gets_it_once_before_the_first_stage(tmp_path, monkeypatch):
    from _pipeline_fakes import ScriptedAudio, install
    from cantocaptions_ai.pipeline.transcribe import _execute_pipeline, validate_config

    scripted = ScriptedAudio()
    install(monkeypatch, scripted)
    media = scripted.write_wav(tmp_path / "episode.wav")
    cfg = PipelineConfig(device="cpu", print_progress=False, audio_normalize=False,
                         output_dir=str(tmp_path / "out"))
    validate_config(cfg)

    class Sink:
        def __init__(self):
            self.events = []

        def plan(self, stages):
            self.events.append(("plan", [s["key"] for s in stages],
                                [s["label"] for s in stages if s["timed"]]))

        def stage_start(self, name):
            self.events.append(("start", name))

        def stage_end(self, name):
            pass

        def set_total(self, total, unit="it"):
            pass

        def advance(self, n=1):
            pass

    sink = Sink()
    _execute_pipeline([media], cfg, collect=True, progress=sink)
    assert sink.events[0][:2] == ("plan", ["vad", "transcription", "pre_align_clean",
                                           "alignment"])
    assert [e for e in sink.events if e[0] == "plan"] == [sink.events[0]]
    started = [e[1] for e in sink.events if e[0] == "start"]
    assert started == sink.events[0][2]   # every timed stage reports, by its plan label

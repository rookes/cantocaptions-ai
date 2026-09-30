"""The whole pipeline, end to end, on CPU in seconds.

Real VAD, ASR and align models are swapped for the fakes in ``_pipeline_fakes`` (the audio
encodes its own transcript, so alignment is exact), and everything else -- orchestration,
debug checkpoints, cue assembly, cleaning, writers, the CLI and the service -- is the real
code. This is the regression net for refactoring ``_execute_pipeline``.

Goldens live in ``tests/golden/``. After a *deliberate* output change, regenerate them with
``UPDATE_GOLDEN=1 uv run pytest tests/test_pipeline_e2e.py`` and review the diff.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

from _pipeline_fakes import ScriptedAudio, StubAsr, StubVad, fake_align_model, install
from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.pipeline.transcribe import _execute_pipeline, _prepare_clips, validate_config
from cantocaptions_ai.utils.output import render_result

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

GOLDEN = Path(__file__).parent / "golden"


def _check_golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8"), f"{name} changed; see module docstring"


@pytest.fixture
def scripted():
    return ScriptedAudio()


@pytest.fixture
def media(tmp_path, scripted):
    return scripted.write_wav(tmp_path / "episode.wav")


@pytest.fixture
def fakes(monkeypatch, scripted):
    install(monkeypatch, scripted)


def _cfg(tmp_path, **overrides) -> PipelineConfig:
    base = dict(device="cpu", print_progress=False, audio_normalize=False,
                output_dir=str(tmp_path / "out"))
    cfg = PipelineConfig(**{**base, **overrides})
    validate_config(cfg)
    return cfg


def _run(paths, cfg, **kwargs):
    return _execute_pipeline(list(paths), cfg, collect=True, **kwargs)


def _srt(results) -> str:
    return render_result(results[0]["result"], "srt", {})


# --- the default path ----------------------------------------------------------------

def test_default_run_matches_golden(tmp_path, media, fakes):
    _check_golden("e2e_default.srt", _srt(_run([media], _cfg(tmp_path))))


def test_cues_start_where_the_script_speaks(tmp_path, media, fakes, scripted):
    segments = _run([media], _cfg(tmp_path))[0]["result"]["segments"]
    assert [round(s["start"], 3) for s in segments] == [start for start, _ in scripted.script]
    # Each cue holds exactly one utterance, cleaned of its closing full stop.
    assert [s["text"] for s in segments] == [t.rstrip("。") for _, t in scripted.script]
    for seg, chars in zip(segments, scripted.char_times):
        assert seg["end"] > chars[-1][1], "a cue must not end before its last character"


def test_writes_files_named_after_the_input(tmp_path, media, fakes):
    _execute_pipeline([media], _cfg(tmp_path, output_format="srt"))
    assert (tmp_path / "out" / "episode.srt").read_text(encoding="utf-8").startswith("1\n")


# --- options that change the path -----------------------------------------------------

def test_no_align_uses_the_vad_segments(tmp_path, media, fakes, scripted):
    # This path raised KeyError('time_stamps') on every run until it was fixed.
    cfg = _cfg(tmp_path, no_align=True, max_line_width=None, max_line_count=None)
    segments = _run([media], cfg)[0]["result"]["segments"]
    assert [round(s["start"], 3) for s in segments] == [start for start, _ in scripted.script]
    assert all("\n" not in s["text"] for s in segments)


def test_no_clean_text_still_breaks_long_lines(tmp_path, monkeypatch):
    # One clause longer than the line width: cue assembly never merges past one line, so
    # only a single over-long clause reaches the line breaker.
    long_line = ScriptedAudio(script=((1.0, "我哋今日去咗好多地方之後仲食咗好多嘢添。"),))
    install(monkeypatch, long_line)
    media = long_line.write_wav(tmp_path / "long.wav")
    cfg = _cfg(tmp_path, no_clean_text=True, max_line_width=12, max_line_count=2)
    segments = _run([media], cfg)[0]["result"]["segments"]
    # Uncleaned: the text is the ASR's, full stop included, only broken into lines. Where
    # the break falls is the line breaker's business (test_cantonese_cleaner).
    (text,) = [s["text"] for s in segments]
    first, second = text.split("\n")
    assert first + second == long_line.script[0][1]
    assert len(first) <= 12


def test_clip_is_mapped_back_onto_the_source_timeline(tmp_path, media, monkeypatch, scripted):
    install(monkeypatch, scripted, asr_offset=3.6)
    cfg = _cfg(tmp_path, audio_start=3.6, output_format="srt")
    paths, display, temps = _prepare_clips([media], cfg)
    try:
        results = _execute_pipeline(paths, cfg, collect=True, display_paths=display,
                                    audio_start_offset=cfg.audio_start)
    finally:
        for t in temps:
            os.remove(t)
    segments = results[0]["result"]["segments"]
    assert [round(s["start"], 3) for s in segments] == [4.0, 7.0]
    assert results[0]["audio_path"] == media and results[0]["name"] == "episode"


def test_realign_places_a_bare_transcript(tmp_path, media, fakes, scripted):
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("\n".join(t for _, t in scripted.script) + "\n", encoding="utf-8")
    segments = _run([media], _cfg(tmp_path, realign=str(transcript)))[0]["result"]["segments"]
    assert len(segments) == len(scripted.script)
    for seg, (start, _) in zip(segments, scripted.script):
        assert abs(seg["start"] - start) <= 0.1


def test_asr_model_is_released_before_alignment_loads(tmp_path, media, fakes, scripted,
                                                      monkeypatch):
    """Each stage's model must be gone before the next loads, or their VRAM stacks up.

    The name bound by ``with model_scope(...) as model`` used to outlive the block, keeping
    the ASR model resident on the GPU through alignment and diarization.
    """
    import weakref
    from cantocaptions_ai.pipeline import alignment, asr

    asr_refs, alive_at_alignment = [], []

    def load_model(*args, **kwargs):
        model = StubAsr(scripted)
        asr_refs.append(weakref.ref(model))
        return model

    def load_align_model(*args, **kwargs):
        alive_at_alignment.append(asr_refs[0]() is not None)
        return fake_align_model(scripted)

    monkeypatch.setattr(asr, "load_model", load_model)
    monkeypatch.setattr(alignment, "load_align_model", load_align_model)
    _run([media], _cfg(tmp_path))
    assert alive_at_alignment == [False]


# --- debug checkpoints ---------------------------------------------------------------

def test_replay_reuses_every_checkpoint(tmp_path, media, fakes):
    debug = str(tmp_path / "debug")
    first = _srt(_run([media], _cfg(tmp_path, debug_dir=debug)))
    assert (StubVad.calls, StubAsr.calls) == (1, 1)
    assert (tmp_path / "debug" / "episode" / "transcription" / "meta.json").is_file()

    replay = _srt(_run([media], _cfg(tmp_path, debug_dir=debug, load_debug_dir=debug)))
    assert (StubVad.calls, StubAsr.calls) == (1, 1), "nothing may be recomputed"
    assert replay == first


def test_changed_settings_recompute_only_what_they_affect(tmp_path, media, fakes):
    debug = str(tmp_path / "debug")
    _run([media], _cfg(tmp_path, debug_dir=debug))
    _run([media], _cfg(tmp_path, debug_dir=debug, load_debug_dir=debug, model="Qwen3-ASR"))
    assert (StubVad.calls, StubAsr.calls) == (1, 2), "a new ASR model keeps the VAD cut"
    _run([media], _cfg(tmp_path, debug_dir=debug, load_debug_dir=debug, model="Qwen3-ASR",
                       vad_onset=0.3))
    assert (StubVad.calls, StubAsr.calls) == (2, 3), "new VAD cuts invalidate the transcript"


# --- entry points --------------------------------------------------------------------

def test_cli_mirrors_input_subfolders(tmp_path, scripted, fakes, monkeypatch):
    from cantocaptions_ai import __main__ as cli_main
    from cantocaptions_ai.pipeline import cli_config

    for season in ("s1", "s2"):
        (tmp_path / "media" / season).mkdir(parents=True)
        scripted.write_wav(tmp_path / "media" / season / "ep01.wav")
    # Keep the developer's own config/user.cfg out of the test.
    monkeypatch.setattr(cli_config, "default_config_dir", lambda: tmp_path / "config")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "cantocaptions", "--input_dir", str(tmp_path / "media"), "--recursive",
        "--output_dir", str(out), "--device", "cpu", "--no_audio_normalize",
        "--print_progress", "False", "--output_format", "srt",
    ])
    cli_main.cli()
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.srt")) == [
        "s1/ep01.srt", "s2/ep01.srt"]


def test_service_renders_the_same_subtitles(tmp_path, media, fakes):
    from cantocaptions_ai.service import run_pipeline

    result = run_pipeline(media, _cfg(tmp_path, output_format="srt"))
    assert (result.num_segments, result.empty, result.language) == (3, False, "yue")
    _check_golden("e2e_default.srt", result.subtitle_text)


def test_resident_service_loads_vad_once_per_setting(tmp_path, media, fakes, monkeypatch):
    from cantocaptions_ai.pipeline import vad
    from cantocaptions_ai.service import PipelineService

    resident, seen = object(), []

    def load_vad(**kw):
        seen.append(kw.get("vad_model"))
        stub = StubVad(cover_all=kw.get("cover_all", False))
        stub.vad_model = resident
        return stub

    monkeypatch.setattr(vad, "load_vad", load_vad)
    service = PipelineService(resident=True)
    service.run(media, _cfg(tmp_path, output_format="srt"))
    service.run(media, _cfg(tmp_path, output_format="srt"))
    # One warm-up load, then each job reuses the resident model.
    assert seen == [None, resident, resident]
    service.run(media, _cfg(tmp_path, output_format="srt", vad_onset=0.3))
    assert seen[3:] == [None, resident], "new VAD settings reload the resident model"


def test_service_raises_instead_of_exiting(tmp_path, media, fakes):
    from cantocaptions_ai.errors import ConfigError, InputError
    from cantocaptions_ai.service import run_pipeline

    with pytest.raises(ConfigError):
        run_pipeline(media, PipelineConfig(device="cpu", output_format="all"))
    with pytest.raises(ConfigError):
        run_pipeline(media, PipelineConfig(device="cpu", language=None))
    with pytest.raises(InputError):
        run_pipeline(str(tmp_path / "missing.wav"), PipelineConfig(device="cpu"))


def test_service_reports_audio_without_speech_as_empty(tmp_path, fakes):
    from cantocaptions_ai.service import run_pipeline

    silent = ScriptedAudio(script=(), duration=3.0).write_wav(tmp_path / "silence.wav")
    result = run_pipeline(silent, _cfg(tmp_path, output_format="srt"))
    assert (result.empty, result.num_segments, result.subtitle_text) == (True, 0, "")


# --- a space-separated language ------------------------------------------------------

def test_english_takes_the_space_separated_path(tmp_path, monkeypatch):
    """The Cantonese assumptions are seams now: an English run joins words with spaces,
    splits sentences at Latin punctuation and breaks lines between words."""
    from cantocaptions_ai.pipeline import alignment

    # "hi." is too short to stand alone, so cue assembly rescues it into the next
    # sentence -- which is where a CJK join would have written "hi.how are you".
    english = ScriptedAudio(script=((1.0, "hi. how are you doing today?"),
                                    (5.0, "fine thanks.")), duration=8.0)
    install(monkeypatch, english)
    # No network in tests: sentence splitting falls back to Latin punctuation.
    monkeypatch.setattr(alignment, "_punkt_splitter", lambda lang: None)
    media = english.write_wav(tmp_path / "english.wav")
    cfg = _cfg(tmp_path, language="en", model="some/english-asr", no_clean_text=True,
               max_line_width=16, max_line_count=2)
    texts = [s["text"] for s in _run([media], cfg)[0]["result"]["segments"]]
    assert texts == ["hi. how are you\ndoing today?", "fine thanks."]


def test_english_needs_its_own_model_and_no_cantonese_cleaning(tmp_path):
    from cantocaptions_ai.errors import ConfigError

    with pytest.raises(ConfigError) as err:
        validate_config(PipelineConfig(device="cpu", language="en"))
    assert "--model" in str(err.value) and "--no_clean_text" in str(err.value)


def test_a_registered_pack_adds_a_fully_supported_language(tmp_path, monkeypatch):
    """One LanguagePack is all a new language needs: its cleaning rules, noise tokens and
    default model all take effect, and the raw-pipeline restrictions no longer apply."""
    from cantocaptions_ai import languages
    from cantocaptions_ai.languages import CleaningSpec, LanguagePack
    from cantocaptions_ai.text_profiles import LATIN_PUNCTUATION, SPACED_SCRIPT

    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "pipeline.toml").write_text(
        '[[steps]]\ntype = "rules"\nfile = "polite.toml"\n'
        '[[steps]]\ntype = "builtin"\nname = "shout"\n', encoding="utf-8")
    (rules / "polite.toml").write_text(
        '[[rules]]\npattern = "fine"\nreplace = "very well"\n', encoding="utf-8")
    pack = LanguagePack(
        "en", SPACED_SCRIPT, LATIN_PUNCTUATION,
        default_model="some/english-asr", default_align_model="some/english-align",
        cleaning=CleaningSpec(rules_dir=rules, builtin_steps=lambda: {"shout": str.upper},
                              noise_tokens=("UM.",)),
    )
    monkeypatch.setitem(languages._PACKS, "en", pack)
    monkeypatch.setattr("cantocaptions_ai.pipeline.alignment._punkt_splitter", lambda lang: None)

    english = ScriptedAudio(script=((1.0, "fine thanks."), (4.0, "um.")), duration=6.0)
    install(monkeypatch, english)
    media = english.write_wav(tmp_path / "english.wav")
    # No --model, no --no_clean_text: the pack supplies both.
    cfg = _cfg(tmp_path, language="en")
    texts = [s["text"] for s in _run([media], cfg)[0]["result"]["segments"]]
    assert texts == ["VERY WELL THANKS."]

import io
import logging
import os
import sys

import pytest

# Imported before conftest's fixture swaps it out of the module for the duration of a test.
from cantocaptions_ai.__main__ import _resolve_log_file
from cantocaptions_ai.utils import log_utils
from cantocaptions_ai.utils.log_utils import StageTimer, TranscriptionSummary


class _Tty(io.StringIO):
    """A console that says it is a terminal, so StageTimer animates."""

    def isatty(self):
        return True


class _Sink:
    def __init__(self):
        self.calls = []

    def stage_start(self, name): pass
    def stage_end(self, name): pass
    def set_total(self, total, unit="it"): self.calls.append(("total", total))
    def advance(self, n=1): self.calls.append(("advance", n))


def test_partial_progress_moves_the_bar_but_never_reaches_a_sink(monkeypatch):
    console = _Tty()
    monkeypatch.setattr(sys, "__stdout__", console)
    sink = _Sink()
    with StageTimer("VAD", TranscriptionSummary(), progress=sink) as timer:
        timer.reporter.set_total(1, unit="file")
        timer._last_partial = -1.0          # past the redraw throttle
        timer.reporter.partial(0.45)
        assert timer._bar.n == 0.45
        assert "0.45/1" in console.getvalue()
        timer._last_partial = -1.0
        timer.reporter.partial(1.0)         # a unit is only finished by advance()
        assert timer._bar.n == 0.99
        timer.reporter.advance(1)
        assert timer._bar.n == 1
    # The worker stores these as integers: a fraction must never get there.
    assert sink.calls == [("total", 1), ("advance", 1)]


def test_advance_after_partial_counts_whole_units(monkeypatch):
    monkeypatch.setattr(sys, "__stdout__", _Tty())
    with StageTimer("VAD", TranscriptionSummary()) as timer:
        timer.reporter.set_total(3, unit="file")
        for _ in range(3):
            timer._last_partial = -1.0
            timer.reporter.partial(0.5)
            timer.reporter.advance(1)
        assert timer._bar.n == 3


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def inductor(monkeypatch):
    """Inductor's logger with a handler standing in for its own stderr one; logging is
    reset afterwards, since setup_logging with a file swaps sys.stdout and sys.stderr."""
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    torch_side = _Collect()
    logging.getLogger("torch._inductor").addHandler(torch_side)
    yield logging.getLogger("torch._inductor.utils"), torch_side
    logging.getLogger("torch._inductor").removeHandler(torch_side)
    log_utils.setup_logging()


def _warn_sms(logger):
    logger.warning("Not enough SMs to use max_autotune_gemm mode",
                   extra={"min_sms": 68, "avail_sms": 34})


def test_the_sm_warning_goes_to_the_log_file_not_the_console(tmp_path, inductor):
    logger, torch_side = inductor
    log_file = tmp_path / "run.log"
    log_utils.setup_logging(level="info", log_file=str(log_file))
    _warn_sms(logger)
    logger.warning("some other inductor warning")
    assert torch_side.messages == ["some other inductor warning"]
    text = log_file.read_text(encoding="utf-8")
    assert "this GPU has 34 SMs, under the 68" in text and "not an error" in text


def test_without_a_log_file_the_sm_note_is_debug_only(inductor, monkeypatch):
    logger, torch_side = inductor
    ours = _Collect()
    for level, expected in (("info", 0), ("debug", 1)):
        log_utils.setup_logging(level=level)
        logging.getLogger("cantocaptions_ai").addHandler(ours)
        ours.setLevel(logging.getLogger("cantocaptions_ai").level)
        ours.messages.clear()
        _warn_sms(logger)
        assert len(ours.messages) == expected, level
        assert torch_side.messages == []


def test_the_filter_is_attached_once():
    log_utils.setup_logging()
    log_utils.setup_logging()
    filters = logging.getLogger("torch._inductor.utils").filters
    assert sum(isinstance(f, log_utils._InductorSmNote) for f in filters) == 1


def test_partial_is_ignored_with_console_output_off():
    with StageTimer("VAD", TranscriptionSummary(enabled=False)) as timer:
        timer.reporter.set_total(1)
        timer.reporter.partial(0.5)
        assert timer._bar is None


def test_a_stage_run_once_per_file_group_is_one_row():
    summary = TranscriptionSummary()
    summary.record("VAD", 1.0, 10.0, 300.0)
    summary.record("Transcription", 2.0, 60.0, 5800.0)
    summary.record("VAD", 1.5, 12.0, 280.0)
    summary.record("Transcription", None, 50.0, 6100.0)
    assert summary._stages == [
        ("VAD", 2.5, 22.0, 300.0),
        ("Transcription", 2.0, 110.0, 6100.0),
    ]


def test_stages_without_a_load_or_vram_stay_without_one():
    summary = TranscriptionSummary()
    summary.record("Speaker assignment", None, 0.1)
    summary.record("Speaker assignment", None, 0.2)
    ((label, load, run, vram),) = summary._stages
    assert (label, load, round(run, 3), vram) == ("Speaker assignment", None, 0.3, None)


# --- the console and the log file ----------------------------------------------------------

@pytest.fixture
def logs(tmp_path, monkeypatch):
    """setup_logging against a fake console and a log file; logging is reset afterwards."""
    console = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", console)
    log_file = tmp_path / "run.log"

    def setup(level="info"):
        log_utils.setup_logging(level=level, log_file=str(log_file))
        return console, log_file

    yield setup
    log_utils.setup_logging()


def test_console_lines_carry_no_timestamp_or_logger_name(logs):
    console, log_file = logs()
    logging.getLogger("cantocaptions_ai.pipeline.vad").info("68 chunks")
    logging.getLogger("cantocaptions_ai.pipeline.vad").debug("Performing VAD...")
    assert console.getvalue() == "  68 chunks\n"
    text = log_file.read_text(encoding="utf-8")
    assert "cantocaptions_ai.pipeline.vad - INFO - 68 chunks" in text
    assert "DEBUG - Performing VAD..." in text


def test_verbose_console_shows_detail_with_times(logs):
    console, _ = logs("debug")
    logging.getLogger("cantocaptions_ai.pipeline.vad").debug("Performing VAD...")
    line = console.getvalue()
    assert "pipeline.vad" in line and "DEBUG" in line and "Performing VAD..." in line


def test_warnings_are_labelled_and_lists_stay_indented(logs):
    console, _ = logs()
    logging.getLogger("cantocaptions_ai").warning("2 lines need review:\n  1  a\n  1  b")
    lines = console.getvalue().splitlines()
    assert lines[0].endswith("Warning: 2 lines need review:")
    assert lines[1:] == ["    1  a", "    1  b"]


def test_our_warnings_reach_the_console_and_library_ones_only_the_file(logs):
    import warnings
    from cantocaptions_ai.errors import CantocaptionsWarning
    console, log_file = logs()
    warnings.warn("speaker_labels has no effect without diarize", CantocaptionsWarning)
    warnings.warn("TensorFloat-32 (TF32) has been disabled", UserWarning)
    shown = console.getvalue()
    assert "Warning: speaker_labels has no effect without diarize" in shown
    assert "TF32" not in shown
    assert "TF32" in log_file.read_text(encoding="utf-8")


def test_verbose_shows_library_warnings_too(logs):
    import warnings
    console, _ = logs("debug")
    warnings.warn("TensorFloat-32 (TF32) has been disabled", UserWarning)
    assert "TF32" in console.getvalue()


class _StatusSink(_Sink):
    def __init__(self):
        super().__init__()
        self.statuses = []

    def status(self, text):
        self.statuses.append(text)


def test_a_stage_is_a_section_with_a_completion_line(monkeypatch):
    console = io.StringIO()   # not a terminal: plain lines, no animation
    monkeypatch.setattr(sys, "__stdout__", console)
    sink = _StatusSink()
    with StageTimer("Vocal isolation", TranscriptionSummary(), progress=sink) as timer:
        timer.status("compiling model")
        timer.reporter.set_total(378, unit="chunk")
        timer.reporter.advance(378)
        timer.reporter.note("compile 3:09")
    text = console.getvalue()
    lines = [line for line in text.splitlines() if line]
    assert lines[0].startswith("-- Vocal isolation ") or lines[0].startswith("── Vocal isolation ")
    assert "Vocal isolation: compiling model" in lines[1]
    assert "Vocal isolation: 378 chunks, compile 3:09 in" in lines[-1]
    assert "\r" not in text
    assert sink.statuses == ["compiling model", ""]


def test_a_failed_stage_says_so(monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", console)
    with pytest.raises(ValueError):
        with StageTimer("Diarization", TranscriptionSummary()):
            raise ValueError("gated")
    assert "Diarization failed after" in console.getvalue()


def test_no_console_output_with_progress_off(monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", console)
    sink = _StatusSink()
    with StageTimer("VAD", TranscriptionSummary(enabled=False), progress=sink) as timer:
        timer.status("loading model")
    assert console.getvalue() == ""
    assert sink.statuses == ["loading model", ""]


def test_a_title_changes_the_console_not_the_summary_row(monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", console)
    summary = TranscriptionSummary()
    with StageTimer("Proofreading", summary, title="Proofreading (realign input)"):
        pass
    assert "Proofreading (realign input) in" in console.getvalue()
    assert [row[0] for row in summary._stages] == ["Proofreading"]



def test_log_file_resolution():
    merged = {"output_dir": "out", "audio": [os.path.join("media", "Season 1", "02.mkv")]}
    path = _resolve_log_file(None, merged)
    assert os.path.dirname(path) == os.path.join("out", "logs")
    assert os.path.basename(path).startswith("02-") and path.endswith(".log")
    assert _resolve_log_file("none", merged) is None
    assert _resolve_log_file("mine.log", merged) == "mine.log"
    assert _resolve_log_file(None, {"output_dir": "out"}) is None   # no input: about to be refused

import io
import logging
import sys

import pytest

from cantocaptions_ai.utils import log_utils
from cantocaptions_ai.utils.log_utils import StageTimer, TranscriptionSummary


class _Sink:
    def __init__(self):
        self.calls = []

    def stage_start(self, name): pass
    def stage_end(self, name): pass
    def set_total(self, total, unit="it"): self.calls.append(("total", total))
    def advance(self, n=1): self.calls.append(("advance", n))


def test_partial_progress_moves_the_bar_but_never_reaches_a_sink(monkeypatch):
    console = io.StringIO()
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
    monkeypatch.setattr(sys, "__stdout__", io.StringIO())
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

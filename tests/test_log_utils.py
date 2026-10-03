import io
import sys

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

import io
import logging
import sys
import threading

from cantocaptions_ai.utils import model_utils
from cantocaptions_ai.utils.log_utils import StageTimer, TranscriptionSummary
from cantocaptions_ai.utils.model_utils import _DownloadProgressLogger


class _Tty(io.StringIO):
    def isatty(self):
        return True


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append((record.levelno, record.getMessage()))


def _in_worker(fn):
    """Run *fn* on a fresh thread, as snapshot_download's thread pool does: no contextvars."""
    t = threading.Thread(target=fn)
    t.start()
    t.join()


def _snapshot_like(timer):
    """The two bars snapshot_download makes from tqdm_class: bytes (fed from worker
    threads) and thread_map's file count (advanced on the calling thread)."""
    bytes_bar = _DownloadProgressLogger(desc="Downloading (incomplete total...)", total=0,
                                        unit="B", unit_scale=True)
    files_bar = _DownloadProgressLogger(total=12, desc="Fetching 12 files")
    bytes_bar.total += 4_000_000_000
    return bytes_bar, files_bar


def test_byte_progress_from_a_worker_thread_reaches_the_status_line(monkeypatch):
    monkeypatch.setattr(sys, "__stdout__", _Tty())
    log = _Collect()
    model_utils.logger.addHandler(log)
    try:
        with StageTimer("Transcription", TranscriptionSummary()) as timer:
            bytes_bar, files_bar = _snapshot_like(timer)
            _in_worker(lambda: bytes_bar.update(2_000_000_000))
            files_bar.update(1)
            assert timer.current_status == "downloading 2.00GB / 4.00GB (50%)"
    finally:
        model_utils.logger.removeHandler(log)
    # A live console carries the progress on the status line: nothing at INFO.
    assert all(level < logging.INFO for level, _ in log.messages)
    assert not any("incomplete total" in m or "12.0B" in m for _, m in log.messages)


def test_the_status_clock_times_the_whole_download(monkeypatch):
    monkeypatch.setattr(sys, "__stdout__", _Tty())
    with StageTimer("Transcription", TranscriptionSummary()) as timer:
        bytes_bar, _ = _snapshot_like(timer)
        bytes_bar.update(1_000)
        since = timer._status_since
        bytes_bar._last_status_t = -1.0          # past the redraw throttle
        bytes_bar.update(1_000_000_000)
        assert "1.00GB" in timer.current_status
        assert timer._status_since == since


def test_an_unmoved_byte_count_is_not_logged_again(monkeypatch):
    monkeypatch.setattr(sys, "__stdout__", io.StringIO())   # not a terminal: lines are INFO
    log = _Collect()
    model_utils.logger.addHandler(log)
    try:
        with StageTimer("Transcription", TranscriptionSummary()) as timer:
            bytes_bar, _ = _snapshot_like(timer)
            for n in (120_000, 0, 0, 1_000_000_000):
                bytes_bar._last_log_t = -1e9     # past the heartbeat interval
                _in_worker(lambda n=n: bytes_bar.update(n))
    finally:
        model_utils.logger.removeHandler(log)
    assert [m for _, m in log.messages] == [
        "Downloaded 120kB / 4.00GB (0%)",
        "Downloaded 1.00GB / 4.00GB (25%)",
    ]

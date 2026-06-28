import itertools
import logging
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tqdm import tqdm as _TqdmBar

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that routes output through tqdm.write() to avoid corrupting progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = None,
) -> None:
    logger = logging.getLogger("cantocaptions_ai")

    logger.handlers.clear()

    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        log_level = logging.WARNING
    logger.setLevel(log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = TqdmLoggingHandler(sys.__stdout__)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    if not log_file:
        warnings_terminal_handler = TqdmLoggingHandler(sys.__stdout__)
        warnings_terminal_handler.setFormatter(formatter)
        warnings_logger.addHandler(warnings_terminal_handler)
    warnings_logger.propagate = False

    if log_file:
        try:
            log_fh = open(log_file, "w", encoding="utf-8", buffering=1)

            file_handler = logging.StreamHandler(log_fh)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            warnings_file_handler = logging.StreamHandler(log_fh)
            warnings_file_handler.setFormatter(formatter)
            warnings_logger.addHandler(warnings_file_handler)

            sys.stdout = log_fh
            sys.stderr = log_fh
        except OSError as e:
            logger.warning(f"Failed to create log file '{log_file}': {e}")
            logger.warning("Continuing without log file")


def get_logger(name: str) -> logging.Logger:
    cantoqwenx_logger = logging.getLogger("cantocaptions_ai")
    if not cantoqwenx_logger.handlers:
        setup_logging()

    logger_name = "cantocaptions_ai" if name == "__main__" else name
    return logging.getLogger(logger_name)


class TranscriptionSummary:
    """Accumulates per-stage timing records and prints a formatted summary table."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._stages: list[tuple[str, float]] = []

    def record(self, label: str, elapsed: float) -> None:
        if self.enabled:
            self._stages.append((label, elapsed))

    def print_summary(self) -> None:
        if not self.enabled or not self._stages:
            return
        total = sum(e for _, e in self._stages)
        col_w = max(len(label) for label, _ in self._stages) + 2
        width = col_w + 12
        eq = "═" * width
        dash = "─" * width
        print(f"\n{eq}", file=sys.__stdout__)
        print(" Transcription complete", file=sys.__stdout__)
        print(eq, file=sys.__stdout__)
        for label, elapsed in self._stages:
            print(f" {label:<{col_w}}{elapsed:>7.2f} s", file=sys.__stdout__)
        print(dash, file=sys.__stdout__)
        print(f" {'Total':<{col_w}}{total:>7.2f} s", file=sys.__stdout__)
        print(f"{eq}\n", file=sys.__stdout__)


class StageTimer:
    """Context manager that times a pipeline stage and drives a tqdm progress bar."""

    def __init__(self, label: str, summary: TranscriptionSummary) -> None:
        self._label = label
        self._summary = summary
        self._start: float = 0.0
        self._bar: "Optional[_TqdmBar]" = None
        self._determinate: bool = False
        self._spinner_stop: threading.Event = threading.Event()
        self._spinner_thread: Optional[threading.Thread] = None

    def __enter__(self) -> "StageTimer":
        if not self._summary.enabled:
            self._start = time.perf_counter()
            return self
        from tqdm import tqdm
        self._start = time.perf_counter()
        self._spinner_stop.clear()
        self._bar = tqdm(
            desc=self._label,
            bar_format="{desc}",
            leave=True,
            file=sys.__stdout__,
            dynamic_ncols=True,
        )
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()
        return self

    def _spin(self) -> None:
        for char in itertools.cycle(r'\|/-'):
            if self._spinner_stop.is_set():
                break
            if not self._determinate and self._bar is not None:
                self._bar.set_description_str(f"{self._label} {char}")
                self._bar.refresh()
            self._spinner_stop.wait(0.12)

    def __exit__(self, *_: object) -> None:
        elapsed = time.perf_counter() - self._start
        self._spinner_stop.set()
        if self._spinner_thread is not None:
            self._spinner_thread.join(timeout=0.5)
        if self._bar is not None:
            if self._determinate:
                self._bar.n = 100
                self._bar.last_print_n = 100
            else:
                self._bar.set_description_str(f"{self._label}: Complete")
            self._bar.close()
        self._summary.record(self._label, elapsed)

    @property
    def callback(self):
        """A ProgressCallback (Callable[[float], None]) suitable for pipeline functions."""
        return self._update

    def _update(self, pct: float) -> None:
        if not self._summary.enabled:
            return
        if not self._determinate:
            self._spinner_stop.set()
            if self._spinner_thread is not None:
                self._spinner_thread.join(timeout=0.5)
            if self._bar is not None:
                self._bar.leave = False
                self._bar.close()
            from tqdm import tqdm
            self._bar = tqdm(
                total=100,
                desc=self._label,
                unit="%",
                leave=True,
                file=sys.__stdout__,
                dynamic_ncols=True,
            )
            self._determinate = True
        if self._bar is not None:
            self._bar.n = pct * 100
            self._bar.refresh()

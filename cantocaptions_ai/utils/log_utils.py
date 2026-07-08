import itertools
import logging
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from tqdm import tqdm as _TqdmBar

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that routes output through tqdm.write() to avoid corrupting progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
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

    # lightning.pytorch imports torch.utils.flop_counter at load time, which logs a
    # spurious warning about triton being absent on CUDA-only builds (no Windows wheels).
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

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
        self._stages: list[tuple[str, Optional[float], float, Optional[float]]] = []

    def record(self, label: str, load_time: Optional[float], run_time: float, vram_peak_mb: Optional[float] = None) -> None:
        if self.enabled:
            self._stages.append((label, load_time, run_time, vram_peak_mb))

    def print_summary(self, process_elapsed: Optional[float] = None) -> None:
        if not self.enabled or not self._stages:
            return
        show_vram = any(v is not None for _, _, _, v in self._stages)
        col_w = max(len(label) for label, _, _, _ in self._stages) + 2
        vram_col_w = 10 if show_vram else 0  # "  X.XX GB" = 10 chars
        # 1 leading space + col_w label + 3 × 11-char time columns + optional VRAM
        width = 1 + col_w + 33 + vram_col_w
        eq = "═" * width
        dash = "─" * width
        print(f"\n{eq}", file=sys.__stdout__)
        print(" Transcription complete", file=sys.__stdout__)
        print(eq, file=sys.__stdout__)
        vram_header = "Peak VRAM".center(10) if show_vram else ""
        print(
            f" {'':>{col_w}}{'Load Time'.center(11)}{'Run Time'.center(11)}{'Total'.center(11)}{vram_header}",
            file=sys.__stdout__,
        )
        for label, load_time, run_time, vram_mb in self._stages:
            load_str  = f" {load_time:>8.2f} s" if load_time is not None else f"{'—':^11}"
            run_str   = f" {run_time:>8.2f} s"
            total_str = f" {(load_time or 0.0) + run_time:>8.2f} s"
            vram_str  = f"  {vram_mb / 1000:>5.1f} GB" if vram_mb is not None else ""
            print(f" {label:<{col_w}}{load_str}{run_str}{total_str}{vram_str}", file=sys.__stdout__)
        if process_elapsed is not None:
            print(dash, file=sys.__stdout__)
            print(f" Total Process Time   {process_elapsed:.2f} s", file=sys.__stdout__)
        print(f"{eq}\n", file=sys.__stdout__)


class ProgressReporter:
    """Lightweight facade handed to pipeline stages so they can drive a stage's
    progress bar without touching StageTimer internals.

    A stage calls ``set_total(n, unit)`` once it knows how many work units it will
    process (segments, chunks, files, …), then ``advance(k)`` as it completes them.
    tqdm then renders accurate throughput (unit/s) and an ETA.
    """

    def __init__(self, timer: "StageTimer") -> None:
        self._timer = timer

    def set_total(self, total: int, unit: str = "it") -> None:
        self._timer._start_determinate(total, unit)

    def advance(self, n: int = 1) -> None:
        self._timer._advance(n)


class StageTimer:
    """Context manager that times a pipeline stage and drives a tqdm progress bar."""

    def __init__(self, label: str, summary: TranscriptionSummary) -> None:
        self._label = label
        self._summary = summary
        self._start: float = 0.0
        self._load_end: Optional[float] = None
        self._bar: "Optional[_TqdmBar]" = None
        self._determinate: bool = False
        self._total: Optional[int] = None
        self._reporter: "ProgressReporter" = ProgressReporter(self)
        self._spinner_stop: threading.Event = threading.Event()
        self._spinner_thread: Optional[threading.Thread] = None

    def __enter__(self) -> "StageTimer":
        if self._summary.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if not self._summary.enabled:
            self._start = time.perf_counter()
            return self
        
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
        end = time.perf_counter()
        vram_peak_mb = (
            torch.cuda.max_memory_allocated() / 1e6
            if self._summary.enabled and torch.cuda.is_available()
            else None
        )
        self._spinner_stop.set()
        if self._spinner_thread is not None:
            self._spinner_thread.join(timeout=0.5)
        if self._bar is not None:
            if self._determinate:
                if self._total is not None and self._bar.n < self._total:
                    self._bar.update(self._total - self._bar.n)
            else:
                self._bar.set_description_str(f"{self._label}: Complete")
            self._bar.close()
        if self._load_end is not None:
            load_time: Optional[float] = self._load_end - self._start
            run_time: float = end - self._load_end
        else:
            load_time = None
            run_time = end - self._start
        self._summary.record(self._label, load_time, run_time, vram_peak_mb)

    def mark_inference_start(self) -> None:
        """Record the boundary between model loading and inference within this stage."""
        self._load_end = time.perf_counter()

    @property
    def reporter(self) -> "ProgressReporter":
        """A ProgressReporter suitable for pipeline stages (set_total / advance)."""
        return self._reporter

    def _start_determinate(self, total: int, unit: str = "it") -> None:
        """Swap the indeterminate spinner for a determinate bar of *total* units.

        tqdm owns rate (unit/s) and ETA; we only feed it monotonic update() deltas.
        """
        if not self._summary.enabled:
            return
        self._spinner_stop.set()
        if self._spinner_thread is not None:
            self._spinner_thread.join(timeout=0.5)
        # Close the previous bar with leave=False (never disable=True:
        # disable=True skips _decr_instances() and leaks tqdm._instances).
        if self._bar is not None:
            self._bar.leave = False
            self._bar.close()

        self._total = total if total and total > 0 else None
        self._bar = tqdm(
            total=self._total,
            desc=self._label,
            unit=unit,
            leave=True,
            file=sys.__stdout__,
            dynamic_ncols=True,
        )
        self._determinate = True

    def _advance(self, n: int = 1) -> None:
        if not self._summary.enabled:
            return
        if not self._determinate:
            # advance() before set_total() → fall back to an unbounded bar
            self._start_determinate(0)
        if self._bar is not None:
            if self._total is not None:
                n = min(n, self._total - self._bar.n)
                if n <= 0:
                    return
            self._bar.update(n)

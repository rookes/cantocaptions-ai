import contextvars
import itertools
import logging
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

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


def _pick_time_unit(total_seconds: float) -> str:
    """Choose the unit the whole summary table renders in, from its longest total."""
    if total_seconds >= 3600:
        return "h"
    if total_seconds >= 60:
        return "m"
    return "s"


def _format_duration(seconds: float, unit: str) -> str:
    """Render *seconds* in ``unit`` (``"s"``/``"m"``/``"h"``, from `_pick_time_unit`).

    The unit is picked once per table rather than per cell so every row stays on
    the same scale and the columns remain comparable at a glance: a 32-second
    stage in an hour-long run reads ``0:00:32``, not ``32.00 s``.
    """
    if unit == "s":
        return f"{seconds:.2f} s"
    whole = int(round(seconds))
    if unit == "m":
        return f"{whole // 60}:{whole % 60:02d}"
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


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
        stage_totals = [(load or 0.0) + run for _, load, run, _ in self._stages]
        unit = _pick_time_unit(
            process_elapsed if process_elapsed is not None else sum(stage_totals)
        )
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
        for (label, load_time, run_time, vram_mb), stage_total in zip(self._stages, stage_totals):
            load_str  = f"{_format_duration(load_time, unit):>11}" if load_time is not None else f"{'—':^11}"
            run_str   = f"{_format_duration(run_time, unit):>11}"
            total_str = f"{_format_duration(stage_total, unit):>11}"
            vram_str  = f"  {vram_mb / 1000:>5.1f} GB" if vram_mb is not None else ""
            print(f" {label:<{col_w}}{load_str}{run_str}{total_str}{vram_str}", file=sys.__stdout__)
        if process_elapsed is not None:
            print(dash, file=sys.__stdout__)
            print(f" Total Process Time   {_format_duration(process_elapsed, unit)}", file=sys.__stdout__)
        print(f"{eq}\n", file=sys.__stdout__)


@runtime_checkable
class ProgressSink(Protocol):
    """Duck-typed sink a caller (e.g. a web worker) passes in to observe pipeline
    progress out-of-band from the console tqdm bars.

    ``StageTimer`` forwards to it independently of ``TranscriptionSummary.enabled``,
    so a headless server can suppress console output (``print_progress=False``) yet
    still stream per-stage progress to a client. All methods are optional-ish: a
    minimal implementation only needs the ones it cares about, but the protocol
    lists the full surface StageTimer will call.
    """

    def stage_start(self, name: str) -> None: ...
    def stage_end(self, name: str) -> None: ...
    def set_total(self, total: int, unit: str = "it") -> None: ...
    def advance(self, n: int = 1) -> None: ...


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


# The StageTimer currently "in scope" on this thread, so code nested arbitrarily deep inside
# a stage (e.g. a model download in model_utils.py) can quiet that stage's spinner for the
# duration of its own output without every intervening call site threading a StageTimer
# reference through. Set in __enter__/reset in __exit__; unset (None) outside any stage.
_active_stage_timer: "contextvars.ContextVar[Optional[StageTimer]]" = contextvars.ContextVar(
    "_active_stage_timer", default=None
)


def get_active_stage_timer() -> "Optional[StageTimer]":
    """The innermost StageTimer currently open on this thread, or None outside any stage."""
    return _active_stage_timer.get()


class StageTimer:
    """Context manager that times a pipeline stage and drives a tqdm progress bar."""

    def __init__(
        self,
        label: str,
        summary: TranscriptionSummary,
        progress: "Optional[ProgressSink]" = None,
    ) -> None:
        self._label = label
        self._summary = summary
        self._progress = progress
        self._start: float = 0.0
        self._load_end: Optional[float] = None
        self._bar: "Optional[_TqdmBar]" = None
        self._determinate: bool = False
        self._total: Optional[int] = None
        self._reporter: "ProgressReporter" = ProgressReporter(self)
        self._spinner_stop: threading.Event = threading.Event()
        self._spinner_thread: Optional[threading.Thread] = None
        self._cv_token: "Optional[contextvars.Token]" = None

    def __enter__(self) -> "StageTimer":
        self._cv_token = _active_stage_timer.set(self)
        # Notify the out-of-band sink regardless of console-summary state so a
        # headless caller still sees stage boundaries with print_progress=False.
        if self._progress is not None:
            self._progress.stage_start(self._label)
        if self._summary.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if not self._summary.enabled:
            self._start = time.perf_counter()
            return self

        self._start = time.perf_counter()
        self._start_spinner()
        return self

    def _start_spinner(self) -> None:
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

    def pause_spinner(self) -> None:
        """Stop the indeterminate spinner so other output (e.g. a model download's own
        progress lines) doesn't render interleaved with it. No-op once the stage has
        moved to a determinate bar, console output is off, or it's already paused."""
        if not self._summary.enabled or self._determinate or self._bar is None:
            return
        self._spinner_stop.set()
        if self._spinner_thread is not None:
            self._spinner_thread.join(timeout=0.5)
        self._bar.leave = False
        self._bar.close()
        self._bar = None

    def resume_spinner(self) -> None:
        """Restart the spinner after pause_spinner(), continuing the same stage label."""
        if not self._summary.enabled or self._determinate or self._bar is not None:
            return
        self._start_spinner()

    def _spin(self) -> None:
        for char in itertools.cycle(r'\|/-'):
            if self._spinner_stop.is_set():
                break
            if not self._determinate and self._bar is not None:
                self._bar.set_description_str(f"{self._label} {char}")
                self._bar.refresh()
            self._spinner_stop.wait(0.12)

    def __exit__(self, *_: object) -> None:
        if self._cv_token is not None:
            _active_stage_timer.reset(self._cv_token)
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
        if self._progress is not None:
            self._progress.stage_end(self._label)

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
        if self._progress is not None:
            self._progress.set_total(total, unit)
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
        if self._progress is not None:
            self._progress.advance(n)
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

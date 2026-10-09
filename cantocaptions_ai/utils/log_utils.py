"""Logging and console output.

Two audiences, two channels. The console is for a person watching the run: each stage is a
section with a header, a live spinner or bar carrying a status, and a completion line, and
log lines inside it are plain indented messages. The log file (``--log_file``, on by
default from the CLI) is for reading afterwards: every record at DEBUG, with timestamps
and logger names, plus the library warnings the console leaves out.

INFO is what the user needs to know (decisions, results, files written); DEBUG is how the
pipeline got there. Anything slow reports what it is doing through ``StageTimer.status``
rather than a "Performing X..." line.
"""
import contextvars
import itertools
import logging
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional, Protocol, runtime_checkable

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from tqdm import tqdm as _TqdmBar


class _StageBar(tqdm):
    """A stage's bar. Its count can stand partway through a unit (ProgressReporter.partial),
    which tqdm would print as a raw float, so it is shown to two decimals."""

    @property
    def format_dict(self):
        d = super().format_dict
        if d["n"] != int(d["n"]):
            d["n"] = round(d["n"], 2)
        return d

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_VERBOSE_CONSOLE_FORMAT = "%(asctime)s %(shortname)-24s %(levelname)-7s %(message)s"

# Logger that library warnings (anything not a CantocaptionsWarning) are written to, at DEBUG.
LIB_WARNINGS_LOGGER = "cantocaptions_ai.lib_warnings"


# --- glyphs --------------------------------------------------------------------------------

_UNICODE_GLYPHS = {
    "ok": "✓", "fail": "✗", "warn": "!", "error": "✗", "cached": "↺", "rule": "─",
    "group": "═", "dot": "·", "ellipsis": "…", "spin": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
}
_ASCII_GLYPHS = {
    "ok": "+", "fail": "x", "warn": "!", "error": "x", "cached": "~", "rule": "-",
    "group": "=", "dot": "-", "ellipsis": "...", "spin": "|/-\\",
}
_glyphs = _ASCII_GLYPHS


def _choose_glyphs() -> dict:
    """Unicode glyphs where the console can encode them, ASCII otherwise.

    A legacy Windows console on cp437/cp1252 has no ✓ or braille spinner; make_console_safe
    would print them as "?", which reads like corruption.
    """
    encoding = getattr(sys.__stdout__, "encoding", None) or "ascii"
    try:
        "".join(_UNICODE_GLYPHS.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return _ASCII_GLYPHS
    return _UNICODE_GLYPHS


def glyph(name: str) -> str:
    return _glyphs[name]


def _console_is_tty() -> bool:
    try:
        return bool(sys.__stdout__.isatty())
    except (AttributeError, ValueError):
        return False


def format_clock(seconds: float) -> str:
    """A duration for a console line: ``4.2 s`` under a minute, else ``m:ss`` / ``h:mm:ss``."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    whole = int(round(seconds))
    if whole < 3600:
        return f"{whole // 60}:{whole % 60:02d}"
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


# --- handlers and formatters ---------------------------------------------------------------

class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that routes output through tqdm.write() to avoid corrupting progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


class ConsoleFormatter(logging.Formatter):
    """The console's view of a record.

    Normally a bare message, indented under the stage section it belongs to, with warnings
    and errors labelled as such. Verbose (a DEBUG console) shows the time, the module and the
    level as well, since the reader then wants to know where a line came from.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(_VERBOSE_CONSOLE_FORMAT if verbose else None, datefmt="%H:%M:%S")
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        if self.verbose:
            record.shortname = record.name.removeprefix("cantocaptions_ai.")
            return super().format(record)
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        if record.levelno >= logging.ERROR:
            message = f"{glyph('error')} Error: {message}"
        elif record.levelno >= logging.WARNING:
            message = f"{glyph('warn')} Warning: {message}"
        return "\n".join("  " + line for line in message.split("\n"))


# The log file handler, while there is one: where _InductorSmNote, the section headers and
# the summary table write directly.
_log_file_handler: Optional[logging.Handler] = None
_log_file_path: Optional[str] = None


def log_file_path() -> Optional[str]:
    """The path of this process's log file, or None when there is none."""
    return _log_file_path


def _to_log_file(message: str, level: int = logging.INFO, name: str = "cantocaptions_ai") -> None:
    """Write *message* to the log file only (no console): for what the console shows its own
    way -- section headers, completion lines, the summary table."""
    if _log_file_handler is None:
        return
    record = logging.getLogger(name).makeRecord(name, level, "", 0, message, None, None)
    _log_file_handler.handle(record)


def console_line(text: str = "", log: bool = True) -> None:
    """Print *text* as-is on the console (no indent or level), and write it to the log file
    unless *log* is False (for what the file already holds in its own form)."""
    tqdm.write(text, file=sys.__stdout__)
    if log and text.strip():
        _to_log_file(text.strip())


class _InductorSmNote(logging.Filter):
    """Keep Inductor's "Not enough SMs to use max_autotune_gemm mode" off the console.

    Inductor logs it once per process when it compiles on a GPU with fewer than 68 SMs
    (a 4060 Ti or 5060 Ti has 34-36), in any compile mode, through its own stderr
    handler. It only means max-autotune's matmul templates are unavailable, which the
    default mode never uses, but on a console it reads as a failure: a user stopped a run
    over it. It is rewritten as a note for the log file, or for --log_level debug when there
    is none. Whether vocal isolation compiled is logged separately ("Compiled the vocal
    isolation model ..." or "... failed; running it uncompiled").
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.getMessage().startswith("Not enough SMs"):
            return True
        avail = getattr(record, "avail_sms", "too few")
        wanted = getattr(record, "min_sms", 68)
        note = logging.getLogger("cantocaptions_ai.torch_compile").makeRecord(
            "cantocaptions_ai.torch_compile", logging.INFO, record.pathname, record.lineno,
            "torch.compile: this GPU has %s SMs, under the %s Inductor wants for "
            "max-autotune matmul templates. Compiling goes ahead without them; this is not "
            "an error (Inductor's own warning, kept off the console).",
            (avail, wanted), None,
        )
        if _log_file_handler is not None:
            _log_file_handler.handle(note)
        else:
            note.levelno, note.levelname = logging.DEBUG, "DEBUG"
            logging.getLogger("cantocaptions_ai.torch_compile").handle(note)
        return False


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """``warnings.showwarning``: our own warnings are the user's, everything else is detail.

    A ``CantocaptionsWarning`` (an option that has no effect, a reference used for several
    files) goes to the console as a plain warning. Anything else comes from a library --
    pyannote turning TF32 off, torch's attention kernels, Inductor's complex ops -- and is
    benign to this pipeline but reads as a failure on a console, so it is written at DEBUG:
    to the log file, and to the console only under --verbose.
    """
    if file is not None:  # someone asked for a specific stream: stdlib behaviour
        _stdlib_showwarning(message, category, filename, lineno, file, line)
        return
    from cantocaptions_ai.errors import CantocaptionsWarning
    if issubclass(category, CantocaptionsWarning):
        logging.getLogger("cantocaptions_ai").warning("%s", message)
        return
    text = warnings.formatwarning(message, category, filename, lineno, line).rstrip()
    logging.getLogger(LIB_WARNINGS_LOGGER).debug("%s", text)


_stdlib_showwarning = warnings.showwarning


def _log_uncaught(exc_type, exc, tb) -> None:
    """sys.excepthook: put the traceback in the log file too, then the usual console one."""
    if _log_file_handler is not None and not issubclass(exc_type, KeyboardInterrupt):
        record = logging.getLogger("cantocaptions_ai").makeRecord(
            "cantocaptions_ai", logging.CRITICAL, "", 0, "Uncaught exception", None,
            (exc_type, exc, tb))
        _log_file_handler.handle(record)
    _previous_excepthook(exc_type, exc, tb)


_previous_excepthook = sys.excepthook


def make_console_safe() -> None:
    """Stop console output from raising on characters the stream cannot encode.

    Every console handler and the end-of-run summary write to ``sys.__stdout__``. Redirected
    to a file on Windows, that stream takes the locale's code page (cp1252), so the first
    log line carrying a CJK character, an arrow or the summary's box-drawing rule raised
    ``UnicodeEncodeError`` -- after every output had been written, turning a successful run
    into exit status 1. A redirected stream is switched to UTF-8 (what a log file of this
    pipeline's text needs anyway); a real console keeps its encoding and replaces what it
    cannot show.
    """
    for stream in (sys.__stdout__, sys.__stderr__):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # a closed or detached stream: nothing to protect
            pass


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = None,
) -> None:
    """Configure the console at *level* and, given *log_file*, a log file at DEBUG.

    A DEBUG console is the verbose one: it shows timestamps, module names and library
    warnings. Anything above shows the plain, sectioned console (see ConsoleFormatter).
    """
    global _glyphs, _log_file_handler, _log_file_path, _previous_excepthook

    make_console_safe()
    _glyphs = _choose_glyphs()
    logger = logging.getLogger("cantocaptions_ai")
    for handler in logger.handlers:
        if handler is _log_file_handler:
            handler.close()
    logger.handlers.clear()

    try:
        console_level = getattr(logging, level.upper())
    except AttributeError:
        console_level = logging.WARNING
    verbose = console_level <= logging.DEBUG

    console_handler = TqdmLoggingHandler(sys.__stdout__)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleFormatter(verbose=verbose))
    logger.addHandler(console_handler)
    logger.setLevel(console_level)
    logger.propagate = False

    if not verbose:
        # Both libraries configure their own stderr handler on import and read these
        # first; their warnings (generation flags, cache notices) are noise to a user here.
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")

    # lightning.pytorch imports torch.utils.flop_counter at load time, which logs a
    # spurious warning about triton being absent on CUDA-only builds (no Windows wheels).
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    inductor_utils = logging.getLogger("torch._inductor.utils")
    if not any(isinstance(f, _InductorSmNote) for f in inductor_utils.filters):
        inductor_utils.addFilter(_InductorSmNote())
    _log_file_handler = None
    _log_file_path = None

    # Our own routing instead of logging.captureWarnings: see _show_warning.
    logging.captureWarnings(False)
    warnings.showwarning = _show_warning

    if log_file:
        try:
            directory = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(directory, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            _log_file_handler = file_handler
            _log_file_path = log_file
        except OSError as e:
            logger.warning(f"Failed to create log file '{log_file}': {e}")
            logger.warning("Continuing without log file")

    if sys.excepthook is not _log_uncaught:
        _previous_excepthook = sys.excepthook
        sys.excepthook = _log_uncaught


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

    def __init__(self, enabled: bool = True, title: str = "Transcription complete") -> None:
        self.enabled = enabled
        self.title = title
        self._stages: list[tuple[str, Optional[float], float, Optional[float]]] = []
        self._amounts: dict[str, tuple[float, str]] = {}

    def add_amount(self, label: str, amount: float, fmt: str = "{:.3f}") -> None:
        """Add to a running total printed under the table (e.g. a paid stage's cost); one
        line per label, summed over files."""
        if not self.enabled or amount is None:
            return
        total, _ = self._amounts.get(label, (0.0, fmt))
        self._amounts[label] = (total + float(amount), fmt)

    def record(self, label: str, load_time: Optional[float], run_time: float, vram_peak_mb: Optional[float] = None) -> None:
        """Add a stage's timings. A stage that runs again (once per file group) is folded
        into its first row: times summed, VRAM peak the highest of them."""
        if not self.enabled:
            return
        for i, (seen, load, run, vram) in enumerate(self._stages):
            if seen == label:
                if load is not None or load_time is not None:
                    load = (load or 0.0) + (load_time or 0.0)
                peaks = [v for v in (vram, vram_peak_mb) if v is not None]
                self._stages[i] = (label, load, run + run_time, max(peaks) if peaks else None)
                return
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
        lines = [f"\n{eq}", f" {self.title}", eq]
        vram_header = "Peak VRAM".center(10) if show_vram else ""
        lines.append(
            f" {'':>{col_w}}{'Load Time'.center(11)}{'Run Time'.center(11)}{'Total'.center(11)}{vram_header}"
        )
        for (label, load_time, run_time, vram_mb), stage_total in zip(self._stages, stage_totals):
            load_str  = f"{_format_duration(load_time, unit):>11}" if load_time is not None else f"{'—':^11}"
            run_str   = f"{_format_duration(run_time, unit):>11}"
            total_str = f"{_format_duration(stage_total, unit):>11}"
            vram_str  = f"  {vram_mb / 1000:>5.1f} GB" if vram_mb is not None else ""
            lines.append(f" {label:<{col_w}}{load_str}{run_str}{total_str}{vram_str}")
        if process_elapsed is not None or self._amounts:
            lines.append(dash)
        for label, (total, fmt) in self._amounts.items():
            lines.append(f" {label:<20} {fmt.format(total)}")
        if process_elapsed is not None:
            lines.append(f" Total Process Time   {_format_duration(process_elapsed, unit)}")
        lines.append(f"{eq}\n")
        text = "\n".join(lines)
        print(text, file=sys.__stdout__)
        _to_log_file(text)


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

    # Optional, and not called by StageTimer: if a sink has it, _execute_pipeline calls it
    # once before the first stage with the run's plan -- a list of {"key", "label",
    # "timed", "cached"} dicts, one per stage in order (see pipeline.stages.plan_entries).
    # stage_start's name is the matching entry's label.
    # def plan(self, stages: list) -> None: ...

    # Optional: if a sink has it, StageTimer.status calls it with what the stage is doing
    # right now ("loading model", "compiling model", "waiting for gemini-3.7-flash"), or ""
    # once that is over. English, like the stage label.
    # def status(self, text: str) -> None: ...


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

    def partial(self, fraction: float) -> None:
        """How far through the current unit the stage is, from 0 to 1.

        For a stage whose units are few and slow: VAD counts files, so a group of one file
        would otherwise sit at 0% until it is done. It moves the console bar only. A
        ProgressSink never sees it, since its counts are whole units (the worker stores them
        as integers), and the next advance() discards it.
        """
        self._timer._partial(fraction)

    def status(self, text: str = "") -> None:
        """What the stage is doing now, shown beside its spinner or bar; "" clears it."""
        self._timer.status(text)

    def note(self, text: str) -> None:
        """A fact for the stage's completion line, e.g. ``compile 3:09`` or ``$0.076``."""
        self._timer.note(text)


class NullReporter:
    """A ProgressReporter that does nothing, for code run outside any stage."""

    def set_total(self, total: int, unit: str = "it") -> None: pass
    def advance(self, n: int = 1) -> None: pass
    def partial(self, fraction: float) -> None: pass
    def status(self, text: str = "") -> None: pass
    def note(self, text: str) -> None: pass


# The StageTimer currently "in scope" on this thread, so code nested arbitrarily deep inside
# a stage (e.g. a model download in model_utils.py) can report on that stage's status line
# without every intervening call site threading a StageTimer reference through. Set in
# __enter__/reset in __exit__; unset (None) outside any stage.
_active_stage_timer: "contextvars.ContextVar[Optional[StageTimer]]" = contextvars.ContextVar(
    "_active_stage_timer", default=None
)


def get_active_stage_timer() -> "Optional[StageTimer]":
    """The innermost StageTimer currently open on this thread, or None outside any stage."""
    return _active_stage_timer.get()


def stage_status(text: str = "") -> None:
    """Set the enclosing stage's status, if there is one (see StageTimer.status)."""
    stage = get_active_stage_timer()
    if stage is not None:
        stage.status(text)


def _section_header(label: str, when: Optional[float] = None) -> str:
    """``── label ─────────── 10:53:59``, as wide as the console allows (at most 80)."""
    stamp = time.strftime("%H:%M:%S", time.localtime(when)) if when is not None else ""
    try:
        columns = os.get_terminal_size(sys.__stdout__.fileno()).columns
    except (AttributeError, OSError, ValueError):
        columns = 80
    width = max(40, min(columns - 1, 80))
    rule = glyph("rule")
    head = f"{rule * 2} {label} "
    tail = f" {stamp}" if stamp else ""
    return head + rule * max(3, width - len(head) - len(tail)) + tail


_UNIT_NAMES = {"seg": "segments", "chunk": "chunks", "file": "files", "line": "lines"}


def _count(n: int, unit: str) -> Optional[str]:
    plural = _UNIT_NAMES.get(unit)
    if plural is None:
        return None
    return f"{n} {plural[:-1] if n == 1 else plural}"


class Section:
    """A console section without a timer row: header, its log lines, a completion line.

    For the pipeline's untimed steps (speaker assignment, cue assembly and writing), so
    every part of a run is delimited the same way as a timed stage. Silent when *enabled*
    is False (``print_progress = False``, or a server).
    """

    def __init__(self, label: str, enabled: bool = True) -> None:
        self._label = label
        self._enabled = enabled
        self._start = 0.0
        self._notes: list = []

    def note(self, text: str) -> None:
        self._notes.append(text)

    def __enter__(self) -> "Section":
        self._start = time.perf_counter()
        if self._enabled:
            console_line()
            console_line(_section_header(self._label, time.time()))
        return self

    def __exit__(self, exc_type, *_: object) -> None:
        if not self._enabled:
            return
        elapsed = time.perf_counter() - self._start
        if exc_type is not None:
            console_line(f"{glyph('fail')} {self._label} failed after {format_clock(elapsed)}")
            return
        detail = f": {', '.join(self._notes)}" if self._notes else ""
        console_line(f"{glyph('ok')} {self._label}{detail} in {format_clock(elapsed)}")


@contextmanager
def section(label: str, enabled: bool = True) -> Iterator[Section]:
    with Section(label, enabled) as s:
        yield s


class StageTimer:
    """Context manager that times a pipeline stage and shows it on the console.

    On the console a stage is a section: a header with its start time, its log lines, a
    spinner (or, once the stage calls ``set_total``, a bar) carrying the current status and
    elapsed time, and a completion line with the duration. ``nested`` makes it a sub-step
    of an enclosing section: no header, the spinner and completion line indented.

    When stdout is not a terminal nothing animates: a status change is printed once as a
    line, and a bar becomes an occasional ``n/total`` line.
    """

    _PLAIN_PROGRESS_INTERVAL_S = 30.0

    def __init__(
        self,
        label: str,
        summary: TranscriptionSummary,
        progress: "Optional[ProgressSink]" = None,
        track_vram: bool = True,
        nested: bool = False,
        title: Optional[str] = None,
    ) -> None:
        # The label names the stage to the summary table and a ProgressSink; the console
        # shows the title, which may say more (whose file, which input).
        self._label = label
        self._title = title or label
        self._summary = summary
        self._progress = progress
        # False for a stage that uses no GPU (proofreading is a network call), whose row
        # would otherwise show whatever the previous stage left allocated.
        self._track_vram = track_vram
        self._nested = nested
        self._indent = "  " if nested else ""
        self._start: float = 0.0
        self._load_end: Optional[float] = None
        self._bar: "Optional[_TqdmBar]" = None
        self._determinate: bool = False
        self._total: Optional[int] = None
        self._unit: str = "it"
        # Whole units advanced; the bar's own count may also hold a partial one on top.
        self._units: int = 0
        self._last_partial: float = 0.0
        self._last_plain: float = 0.0
        self._status: str = ""
        self._status_since: float = 0.0
        self._notes: list = []
        self._reporter: "ProgressReporter" = ProgressReporter(self)
        self._spinner_stop: threading.Event = threading.Event()
        self._spinner_thread: Optional[threading.Thread] = None
        self._cv_token: "Optional[contextvars.Token]" = None
        self._console = False
        self._animated = False

    @property
    def animated(self) -> bool:
        """Whether this stage draws a live spinner/bar (a console on a terminal)."""
        return self._animated

    def __enter__(self) -> "StageTimer":
        self._cv_token = _active_stage_timer.set(self)
        # Notify the out-of-band sink regardless of console-summary state so a
        # headless caller still sees stage boundaries with print_progress=False.
        if self._progress is not None:
            self._progress.stage_start(self._label)
        if self._summary.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start = time.perf_counter()
        self._status_since = self._start
        self._console = self._summary.enabled
        if not self._console:
            return self
        self._animated = _console_is_tty()
        if self._nested:
            _to_log_file(f"-- {self._title} --")
        else:
            console_line()
            console_line(_section_header(self._title, time.time()))
        if self._animated:
            self._start_spinner()
        return self

    # -- the spinner ------------------------------------------------------------------------

    def _spinner_text(self, char: str) -> str:
        elapsed = time.perf_counter() - (self._status_since if self._status else self._start)
        what = f"{self._title}: {self._status}" if self._status else self._title
        return f"{self._indent}  {char} {what}{glyph('ellipsis')} {format_clock(elapsed)}"

    def _start_spinner(self) -> None:
        self._spinner_stop.clear()
        self._bar = tqdm(
            desc=self._spinner_text(glyph("spin")[0]),
            bar_format="{desc}",
            leave=False,
            file=sys.__stdout__,
            dynamic_ncols=True,
        )
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self) -> None:
        self._spinner_stop.set()
        if self._spinner_thread is not None:
            self._spinner_thread.join(timeout=0.5)
            self._spinner_thread = None

    def pause_spinner(self) -> None:
        """Stop the indeterminate spinner so other output doesn't render interleaved with
        it. No-op once the stage has moved to a determinate bar, console output is off, or
        it's already paused."""
        if not self._animated or self._determinate or self._bar is None:
            return
        self._stop_spinner()
        bar, self._bar = self._bar, None
        bar.close()

    def resume_spinner(self) -> None:
        """Restart the spinner after pause_spinner(), continuing the same stage label."""
        if not self._animated or self._determinate or self._bar is not None:
            return
        self._start_spinner()

    def _spin(self) -> None:
        # Also keeps a determinate bar's clock moving: tqdm only redraws on update(), so a
        # stretch with no unit finishing (realign fitting its transform after the last
        # chunk is encoded) would otherwise freeze the elapsed time and look like a hang.
        for tick, char in enumerate(itertools.cycle(glyph("spin"))):
            if self._spinner_stop.is_set():
                break
            bar = self._bar
            if bar is not None:
                if not self._determinate:
                    bar.set_description_str(self._spinner_text(char), refresh=False)
                    bar.refresh()
                elif tick % 8 == 0:
                    bar.refresh()
            self._spinner_stop.wait(0.12)

    # -- status and notes -------------------------------------------------------------------

    def status(self, text: str = "") -> None:
        """What the stage is doing now: ``loading model``, ``compiling model``, ``waiting for
        gemini-3.7-flash (398 cues)``. Shown beside the spinner with the time it has taken
        so far, or as the bar's postfix; "" clears it. Also sent to a sink with ``status``.
        """
        if text == self._status:
            return
        self._status = text
        self._status_since = time.perf_counter()
        send = getattr(self._progress, "status", None)
        if callable(send):
            send(text)
        if text:
            _to_log_file(f"{self._title}: {text}", logging.DEBUG)
        if not self._console:
            return
        if not self._animated:
            if text:
                console_line(f"{self._indent}  {self._title}: {text}{glyph('ellipsis')}",
                             log=False)
            return
        if self._determinate and self._bar is not None:
            self._bar.set_postfix_str(text, refresh=True)

    @property
    def current_status(self) -> str:
        return self._status

    def note(self, text: str) -> None:
        """Add *text* to the completion line (``✓ Vocal isolation: 378 chunks, compile 3:09``)."""
        if text:
            self._notes.append(text)

    # -- end --------------------------------------------------------------------------------

    def __exit__(self, exc_type, *_: object) -> None:
        if self._cv_token is not None:
            _active_stage_timer.reset(self._cv_token)
        end = time.perf_counter()
        vram_peak_mb = (
            torch.cuda.max_memory_allocated() / 1e6
            if self._summary.enabled and self._track_vram and torch.cuda.is_available()
            else None
        )
        self._stop_spinner()
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        if self._load_end is not None:
            load_time: Optional[float] = self._load_end - self._start
            run_time: float = end - self._load_end
        else:
            load_time = None
            run_time = end - self._start
        self._summary.record(self._label, load_time, run_time, vram_peak_mb)
        if self._status:
            send = getattr(self._progress, "status", None)
            if callable(send):
                send("")
        if self._progress is not None:
            self._progress.stage_end(self._label)
        if self._console:
            console_line(self._completion_line(exc_type, end - self._start))

    def _completion_line(self, exc_type, elapsed: float) -> str:
        if exc_type is not None:
            what = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            return f"{self._indent}{glyph('fail')} {self._title} {what} after {format_clock(elapsed)}"
        details = []
        if self._determinate and self._total:
            counted = _count(self._total, self._unit)
            if counted:
                details.append(counted)
        details += self._notes
        detail = f": {', '.join(details)}" if details else ""
        return f"{self._indent}{glyph('ok')} {self._title}{detail} in {format_clock(elapsed)}"

    def mark_inference_start(self) -> None:
        """Record the boundary between model loading and inference within this stage."""
        self._load_end = time.perf_counter()
        self.status("")

    @property
    def reporter(self) -> "ProgressReporter":
        """A ProgressReporter suitable for pipeline stages (set_total / advance)."""
        return self._reporter

    # -- the bar ----------------------------------------------------------------------------

    def _start_determinate(self, total: int, unit: str = "it") -> None:
        """Swap the indeterminate spinner for a determinate bar of *total* units.

        tqdm owns rate (unit/s) and ETA; we only feed it monotonic update() deltas.
        """
        if self._progress is not None:
            self._progress.set_total(total, unit)
        self._total = total if total and total > 0 else None
        self._unit = unit
        self._units = 0
        self._determinate = True
        if not self._console:
            return
        if not self._animated:
            self._last_plain = time.perf_counter()
            return
        # Close the previous bar with leave=False (never disable=True:
        # disable=True skips _decr_instances() and leaks tqdm._instances). The spinner
        # thread keeps running and switches to refreshing the new bar (see _spin).
        previous, self._bar = self._bar, None
        if previous is not None:
            previous.leave = False
            previous.close()
        self._bar = _StageBar(
            total=self._total,
            desc=f"{self._indent}  {self._title}",
            unit=unit,
            leave=False,
            file=sys.__stdout__,
            dynamic_ncols=True,
        )
        if self._status:
            self._bar.set_postfix_str(self._status, refresh=True)

    def _advance(self, n: int = 1) -> None:
        if self._progress is not None:
            self._progress.advance(n)
        if not self._console:
            return
        if not self._determinate:
            # advance() before set_total() → fall back to an unbounded bar
            self._start_determinate(0)
        if self._total is not None:
            n = min(n, self._total - self._units)
            if n <= 0:
                return
        self._units += n
        if not self._animated:
            now = time.perf_counter()
            if now - self._last_plain >= self._PLAIN_PROGRESS_INTERVAL_S:
                self._last_plain = now
                of = f"/{self._total}" if self._total else ""
                console_line(f"{self._indent}  {self._title}: {self._units}{of} {self._unit}")
            return
        if self._bar is not None:
            # A finished unit replaces whatever partial progress was shown into it.
            self._bar.n = self._units - n
            self._bar.update(n)

    def _partial(self, fraction: float) -> None:
        if not self._animated or not self._determinate or self._bar is None:
            return
        if self._total is not None and self._units >= self._total:
            return
        # Redrawing costs a console write; a VAD hook fires ~70 times a file.
        now = time.perf_counter()
        if now - self._last_partial < 0.2:
            return
        self._last_partial = now
        # Never the whole unit: only advance() says it is done.
        self._bar.n = self._units + min(max(fraction, 0.0), 0.99)
        self._bar.refresh()

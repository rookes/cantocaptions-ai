"""Reading subtitle files: SRT and WebVTT in, cue structure intact.

This replaces ``pipeline/retime.py:load_subtitle_file``, which delegated to
``suber.file_readers.read_input_file`` and had three problems.

**It made a dev-only package a runtime dependency.** ``subtitle-edit-rate`` is declared in
``[dependency-groups] dev`` in pyproject.toml, and it is the right place for it -- it is a
scoring library used by ``utils/suber.py`` and the eval harnesses. But ``--realign`` and
``--reference_subtitle`` both read subtitles on the ordinary run path, so an install without
the dev group failed on a plain ``--realign file.srt``.

**It flattened multi-line cues.** suber joins every word of a cue with a single space, so a
two-speaker exchange written as two lines came back as one run-on line. That is a change to
the content, not to the formatting: on the ReZero fixture 5 of the 7 two-line cues are
dialogue pairs (``-講真嘠？`` / ``-講真㗎！``).

**It advertised a format it could not read.** The extension was passed through uppercased, and
``read_input_file`` accepts only ``"SRT"`` and ``"plain"`` -- so a ``.vtt``, which several help
strings offer, raised ``ValueError: Unknown file format: VTT`` from inside suber.

Cue text keeps its line breaks as ``"\\n"`` all the way to the writer. See the realign notes on
``REALIGN_PUNCTUATION`` for how the aligner is told to treat one as a pause.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from cantocaptions_ai.utils.schema import SingleSegment

# Extensions that carry a cue timeline, as opposed to a bare line-delimited transcript.
TIMED_EXTENSIONS = (".srt", ".vtt", ".webvtt")

# ``-->`` with optional decoration either side. WebVTT allows cue settings after the end
# timestamp (``align:start line:90%``), which are positioning hints and carry no timing.
_ARROW = re.compile(r"^(?P<start>[^\s]+)\s*-->\s*(?P<end>[^\s]+)(?:\s+.*)?$")

# hh:mm:ss,mmm / hh:mm:ss.mmm / mm:ss.mmm -- WebVTT makes the hour optional, and both formats
# are written with either decimal marker in the wild.
_TIMESTAMP = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})$")

# ``<i>``, ``</font>``, ``<font color="#ffffff">``. suber's regex was ``</?[^>]>``, which
# matches a *one-character* tag only, so it stripped ``<i>`` and left ``<font ...>`` behind to
# be fed to the aligner as text.
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

# ASS/SSA override blocks, which survive a careless conversion to SRT: ``{\an8}``, ``{\pos(..)}``.
_ASS_OVERRIDE = re.compile(r"\{\\[^}]*\}")


@dataclass
class SubtitleCue:
    """One cue, with its line breaks preserved."""
    index: int
    start: float
    end: float
    text: str


class SubtitleFormatError(ValueError):
    """The file is not a subtitle format this can read."""


def _parse_timestamp(value: str) -> Optional[float]:
    match = _TIMESTAMP.match(value.strip())
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    # "1" in the fraction field means a tenth, not a millisecond.
    millis = int(fraction.ljust(3, "0"))
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000.0


def strip_markup(text: str) -> str:
    """Remove styling that is markup rather than words."""
    text = _ASS_OVERRIDE.sub("", text)
    text = _HTML_TAG.sub("", text)
    return text


def _read_lines(path: str) -> List[str]:
    """Decode the file, tolerating a UTF-8 BOM and any mix of line endings."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw.decode("utf-8-sig", errors="replace").splitlines()


def read_subtitle_cues(path: str) -> List[SubtitleCue]:
    """Parse an SRT or WebVTT file into cues, keeping intra-cue line breaks as ``\\n``.

    The two formats differ only in ways this does not have to care about: a ``WEBVTT``
    header, an optional cue identifier line, cue settings trailing the timestamps, and
    ``NOTE``/``STYLE``/``REGION`` blocks. Everything else is "a timestamp line, then text
    until a blank line", so one scanner reads both. Cue numbering is assigned here rather
    than read, because an SRT in the wild is not reliably numbered from 1.
    """
    lines = _read_lines(path)
    cues: List[SubtitleCue] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        # A WebVTT block header introduces something that is not a cue; skip to the blank
        # line that ends it. (A cue's own text can never start with these, since a cue always
        # opens with its timestamp or identifier line.)
        if stripped.split(maxsplit=1)[0] in ("WEBVTT", "NOTE", "STYLE", "REGION"):
            while i < n and lines[i].strip():
                i += 1
            continue

        arrow = _ARROW.match(stripped)
        if arrow is None:
            # Either a cue number (SRT) or a cue identifier (VTT); the timestamp is next.
            i += 1
            if i >= n:
                break
            arrow = _ARROW.match(lines[i].strip())
            if arrow is None:
                # Not a cue after all -- a stray line. Skip to the next blank and resync
                # rather than aborting the whole file over it.
                while i < n and lines[i].strip():
                    i += 1
                continue
        start = _parse_timestamp(arrow.group("start"))
        end = _parse_timestamp(arrow.group("end"))
        i += 1
        if start is None or end is None:
            continue

        body: List[str] = []
        while i < n and lines[i].strip():
            body.append(strip_markup(lines[i]).strip())
            i += 1
        text = "\n".join(part for part in body if part)
        if text:
            cues.append(SubtitleCue(index=len(cues), start=start, end=end, text=text))
    if not cues:
        raise SubtitleFormatError(f"No subtitle cues found in: {path}")
    return cues


def load_subtitle_file(path: str) -> List[SingleSegment]:
    """Cues as ``{start, end, text}`` dicts, line breaks flattened to spaces.

    The shape the rest of the pipeline has always consumed. Callers that want the cue's own
    line structure -- ``--realign`` does, since a line break inside a cue is usually a change
    of speaker -- should use :func:`read_subtitle_cues` instead.
    """
    return [
        {"start": cue.start, "end": cue.end, "text": cue.text.replace("\n", " ")}
        for cue in read_subtitle_cues(path)
    ]


def subtitle_has_timings(path: str) -> bool:
    """True when *path* is a subtitle carrying a cue timeline, not a bare transcript.

    This decides what ``--realign_mode auto`` resolves to, so it tests the *content* and not
    only the extension: a transcript saved as ``.srt`` would otherwise be synced against
    timings it does not have, and a subtitle saved as ``.txt`` would have its timings thrown
    away in silence. The extension narrows the search; a cue arrow is what settles it.
    """
    if os.path.splitext(path)[1].lower() not in TIMED_EXTENSIONS:
        return False
    try:
        for line in _read_lines(path):
            if _ARROW.match(line.strip()):
                return True
    except OSError:
        return False
    return False

"""Reference cues: tidying and writing. Shared by the text and the OCR paths, and imported by
the OCR worker, so it imports nothing heavier than the standard library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from cantocaptions_ai.utils.output import format_timestamp

# Cues closer than this are one cue shown twice (a PGS re-send, a fade step).
MERGE_GAP_S = 0.05
# The least a cue may last; anything shorter is stretched, so an SRT reader never sees an
# empty or inverted span.
MIN_DURATION_S = 0.02


@dataclass
class Cue:
    start: float
    end: float
    text: str


def tidy(cues: Iterable[Cue]) -> List[Cue]:
    """Sorted by start, without empty cues, with repeats and touching copies of the same text
    merged into one, and every cue at least MIN_DURATION_S long. Different lines that overlap
    (a sign over dialogue) are both kept: a reference may show two things at once."""
    ordered = sorted((Cue(c.start, c.end, c.text.strip()) for c in cues if c.text.strip()),
                     key=lambda c: (c.start, c.end))
    out: List[Cue] = []
    for c in ordered:
        if out and out[-1].text == c.text and c.start <= out[-1].end + MERGE_GAP_S:
            out[-1].end = max(out[-1].end, c.end)
            continue
        out.append(c)
    for c in out:
        if c.end < c.start + MIN_DURATION_S:
            c.end = c.start + MIN_DURATION_S
    return out


def to_srt(cues: Iterable[Cue]) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        start = format_timestamp(c.start, always_include_hours=True, decimal_marker=",")
        end = format_timestamp(c.end, always_include_hours=True, decimal_marker=",")
        blocks.append(f"{i}\n{start} --> {end}\n{c.text}\n")
    return "\n".join(blocks)


def write_srt(cues: Iterable[Cue], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_srt(cues))

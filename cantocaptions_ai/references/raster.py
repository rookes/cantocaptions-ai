"""Bitmap subtitle tracks (PGS, DVB, VobSub) as pictures, one per on-screen event.

ffmpeg does the decoding. Routing a subtitle stream into a filtergraph turns on its
*sub2video* mode, which paints each subtitle update onto a transparent full-frame canvas with
the palette applied, whatever the codec. With ``-fps_mode passthrough`` it emits one frame per
update instead of one per video tick, and ``-frame_pts true`` names each PNG after its
presentation time in microseconds, which is the only timing that comes back. No ``-copyts``:
ffmpeg then rebases the times to the start of the file, as it does when it extracts a text
track, so both kinds of reference share one timeline.

The frames come in pairs, the last moment of one state and the first of the next (a caption
at 90.799000, the same caption again at 91.924999, a blank at 91.925000). An *event* is a run
of frames whose visible text mask is the same; the next frame that differs ends it. The mask
is the alpha channel thresholded at ALPHA_ON, so the steps of a fade-in or fade-out, which
differ only in transparency, stay one event, and a frame whose text is still mostly
transparent counts as blank.

Each event's picture is cut into text lines for OCR: rows of the canvas holding text, split
at blank rows, with slivers (the separate strokes of 二 or 三, a dash) merged back into the
nearest line. A top-of-screen sign and the dialogue at the bottom arrive in one frame, and
become two lines of one cue, top first.

This module imports numpy and Pillow only: the OCR worker loads it beside Paddle, and torch
must never be in that process (see worker.py).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Alpha at or above which a pixel counts as visible text.
ALPHA_ON = 128
# A caption still on screen when the stream ends is given this long.
FALLBACK_DURATION_S = 3.0
# A row run shorter than this share of the tallest one in the picture is part of its
# neighbour (a stroke of 二, a dash, the dots of an ellipsis), not a line of its own.
SLIVER_SHARE = 0.4
# White border added around each piece of text before it is read.
PAD_PX = 10
# Extra pixels kept around a line's mask, so the anti-aliased outline (below ALPHA_ON) is
# not cut off.
MARGIN_PX = 3
# A blank stretch of a line at least this many line-heights wide separates two pieces of
# text (two speakers side by side, "02:20後  等候訊號行動"). Each piece is read on its own and
# the pieces are joined with a space: read as one, the recogniser fills the gap with a
# character that is not there (妳, ，). The space between two characters is well under one.
SEGMENT_GAP_SHARE = 1.0


@dataclass
class Event:
    start: float
    end: float
    frame: str                      # a PNG holding the event's picture


@dataclass
class EventImages:
    """What OCR reads for one event: the whole picture, and each text line on its own, cut
    into pieces. ``lines[i]`` is line i's pieces, left to right, each ``(joiner, image)``:
    the text goes back together as ``"".join(joiner + text)``. ``key`` identifies the
    picture's pixels, so an identical picture is read only once."""
    key: str
    block: object                   # PIL.Image, mode "L"
    lines: List[List[Tuple[str, object]]] = field(default_factory=list)


def render_frames(media: str, stream_index: int, canvas: Tuple[int, int],
                  out_dir: str) -> List[Tuple[float, str]]:
    """Render one bitmap subtitle stream to PNGs in *out_dir*: [(seconds, path)], in time order."""
    os.makedirs(out_dir, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-max_error_rate", "1.0",
         "-canvas_size", f"{canvas[0]}x{canvas[1]}", "-i", media,
         "-filter_complex", f"[0:{stream_index}]null[out]", "-map", "[out]",
         "-fps_mode", "passthrough", "-frame_pts", "true", "-f", "image2",
         os.path.join(out_dir, "%d.png")],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    frames = []
    for name in os.listdir(out_dir):
        stem, ext = os.path.splitext(name)
        if ext == ".png" and stem.isdigit():
            frames.append((int(stem) / 1_000_000, os.path.join(out_dir, name)))
    if proc.returncode != 0 and not frames:
        raise RuntimeError(f"ffmpeg could not render subtitle stream {stream_index}: "
                           f"{proc.stderr.strip()[-300:]}")
    return sorted(frames)


_THRESHOLD = [255 if v >= ALPHA_ON else 0 for v in range(256)]


def frame_key(path: str) -> Optional[str]:
    """A key for the visible text of a rendered frame, or None when nothing is visible.

    Most of a frame is transparent, so the threshold and the bounding box are taken by Pillow
    in C and only the box is handed to numpy: on full-HD frames that is several times faster
    than thresholding the whole frame in numpy, and an episode has a thousand frames or more.
    """
    from PIL import Image
    with Image.open(path) as im:
        alpha = (im if im.mode == "RGBA" else im.convert("RGBA")).getchannel("A")
    visible = alpha.point(_THRESHOLD)
    box = visible.getbbox()
    if box is None:
        return None
    digest = hashlib.md5(np.packbits(np.asarray(visible.crop(box)) > 0).tobytes()).hexdigest()
    return "{},{},{},{}:{}".format(*box, digest)


def frame_events(frames: Sequence[Tuple[float, str]],
                 fallback: float = FALLBACK_DURATION_S) -> List[Event]:
    """Events from rendered frames: each run of frames with the same visible text."""
    events: List[Event] = []
    current: Optional[Event] = None
    current_key: Optional[str] = None
    # Decoding the PNGs is most of the time, and Pillow decodes without holding the GIL.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        keys = list(pool.map(frame_key, [path for _, path in frames]))
    for (t, path), key in zip(frames, keys):
        if key == current_key:
            continue
        if current is not None:
            current.end = t
            events.append(current)
        current = Event(t, t, path) if key is not None else None
        current_key = key
    if current is not None:
        current.end = current.start + fallback
        events.append(current)
    return events


def line_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[(y0, y1)] row ranges of the text lines in a frame's visible mask, top to bottom."""
    rows = mask.any(axis=1)
    runs: List[List[int]] = []
    y = 0
    n = rows.size
    while y < n:
        if rows[y]:
            start = y
            while y < n and rows[y]:
                y += 1
            runs.append([start, y])
        else:
            y += 1
    if not runs:
        return []
    tallest = max(b - a for a, b in runs)
    merged = True
    while merged and len(runs) > 1:
        merged = False
        for i, (a, b) in enumerate(runs):
            if b - a >= SLIVER_SHARE * tallest:
                continue
            # Join a sliver to the nearer neighbour.
            gap_up = a - runs[i - 1][1] if i > 0 else None
            gap_down = runs[i + 1][0] - b if i + 1 < len(runs) else None
            j = i - 1 if gap_down is None or (gap_up is not None and gap_up <= gap_down) else i + 1
            lo, hi = min(i, j), max(i, j)
            runs[lo] = [runs[lo][0], runs[hi][1]]
            del runs[hi]
            merged = True
            break
    return [(a, b) for a, b in runs]


def segment_spans(mask: np.ndarray) -> List[Tuple[str, int, int]]:
    """[(joiner, x0, x1)] the pieces of text one line's mask is read in, left to right: the
    line cut at every blank of SEGMENT_GAP_SHARE line-heights or more, joined back with a space."""
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return []
    min_gap = SEGMENT_GAP_SHARE * mask.shape[0]
    pieces = [["", int(cols[0]), int(cols[0]) + 1]]
    for x in cols[1:]:
        if x - pieces[-1][2] >= min_gap:
            pieces.append([" ", int(x), int(x) + 1])
        else:
            pieces[-1][2] = int(x) + 1
    return [(j, a, b) for j, a, b in pieces]


def _ink(rgba: np.ndarray):
    """A transparent RGBA crop as dark text on white, padded: a grayscale PIL image.

    Subtitle text is a fill inside an outline of the opposite tone (white in black, mostly;
    yellow in black; sometimes black in white). Only the fill is kept, as ink, weighted by
    its opacity. Flattening the whole thing onto grey instead, outline and all, leaves the
    recogniser a thick dark ring around every stroke: it then reads an outlined fullwidth
    comma as a middle dot (課長・現場 for 課長，現場), an ellipsis as ".…", and digits as their
    fullwidth forms. Which tone is the fill is read off the picture: the outline is what
    touches the transparent background.
    """
    from PIL import Image, ImageOps
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    lum = rgba[..., :3].astype(np.float32).mean(axis=2)
    opaque = alpha >= 0.5
    # Outside the crop counts as transparent, so text cut tight at the edge still has one.
    clear = np.pad(~opaque, 1, constant_values=True)
    near_clear = (clear[1:-1, 1:-1] | clear[:-2, 1:-1] | clear[2:, 1:-1]
                  | clear[1:-1, :-2] | clear[1:-1, 2:])
    edge = opaque & near_clear
    dark_outline = not edge.any() or float(np.median(lum[edge])) < 128
    tone = (lum - 64) / 160 if dark_outline else (192 - lum) / 160
    ink = alpha * np.clip(tone, 0.0, 1.0)
    grey = (255 - ink * 255).astype(np.uint8)
    return ImageOps.expand(Image.fromarray(grey, "L"), border=PAD_PX, fill=255)


def event_images(event: Event) -> Optional[EventImages]:
    from PIL import Image
    with Image.open(event.frame) as im:
        rgba = np.asarray(im.convert("RGBA"))
    mask = rgba[..., 3] >= ALPHA_ON
    spans = line_spans(mask)
    if not spans:
        return None
    h, w = mask.shape

    def crop(y0: int, y1: int, x0: int = 0, x1: int = w) -> np.ndarray:
        cols = x0 + np.flatnonzero(mask[y0:y1, x0:x1].any(axis=0))
        x0, x1 = max(0, cols[0] - MARGIN_PX), min(w, cols[-1] + 1 + MARGIN_PX)
        return rgba[max(0, y0 - MARGIN_PX):min(h, y1 + MARGIN_PX), x0:x1]

    block = crop(spans[0][0], spans[-1][1])
    key = hashlib.md5(np.ascontiguousarray(block).tobytes()).hexdigest()
    lines = [[(joiner, _ink(crop(y0, y1, x0, x1))) for joiner, x0, x1 in segment_spans(mask[y0:y1])]
             for y0, y1 in spans]
    return EventImages(key=key, block=_ink(block), lines=lines)


def iter_event_images(events: Sequence[Event]) -> Iterator[Tuple[Event, Optional[EventImages]]]:
    """(event, its images) in order, prepared on a few threads (see frame_events)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        yield from zip(events, pool.map(event_images, events))

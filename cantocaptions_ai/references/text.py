"""Text subtitle tracks (subrip, ASS/SSA, mov_text, WebVTT): ffmpeg converts the stream to SRT
and the cues are read back with the pipeline's own subtitle reader.

ASS styling (``{\\pos}``, ``{\\an8}``) is dropped: a reference is read for its words, and its
position on screen means nothing to the consumers. Sign and karaoke layers often repeat a line
several times over the same span; those copies collapse into one cue (``cues.tidy``).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List

from cantocaptions_ai.references.cues import Cue, tidy
from cantocaptions_ai.utils.subtitles import read_subtitle_cues, strip_markup


def extract_text_track(media: str, stream_index: int) -> List[Cue]:
    fd, tmp = tempfile.mkstemp(suffix=".srt", prefix="cantoref_")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", media, "-map", f"0:{stream_index}",
             "-c:s", "srt", tmp],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg could not extract subtitle stream {stream_index}: "
                               f"{proc.stderr.strip()[-300:]}")
        cues = read_subtitle_cues(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return tidy(Cue(c.start, c.end, strip_markup(c.text)) for c in cues)

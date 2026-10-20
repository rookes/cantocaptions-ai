"""Which subtitle tracks a media file carries, and which of them are worth extracting.

A track is either *text* (subrip, ASS, mov_text, WebVTT: ffmpeg converts it straight to SRT)
or *bitmap* (PGS, DVB, VobSub: pictures of text, rendered by ffmpeg and read by OCR). Text
formats this does not know, and teletext, are listed as ``other`` and never extracted.

Nothing here imports torch or an OCR library: it runs in the main process before any model.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from cantocaptions_ai.utils.audio import _CHINESE_LANG_CODES, _CHINESE_TITLE_KEYWORDS

TEXT_CODECS = frozenset({"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"})


@dataclass(frozen=True)
class BitmapCodec:
    """A bitmap subtitle format ffmpeg can render (``sub2video``; see raster.py).

    ``canvas`` is the largest picture the format is authored for. ffmpeg renders onto the
    *video's* size when not told otherwise and scales the subtitle to fit, so a 1920x1040
    encode of a 1920x1080 Blu-ray shrinks every PGS caption by 4%; rendering onto at least
    this canvas keeps the bitmaps at their native size.
    """
    name: str
    canvas: Tuple[int, int]


BITMAP_CODECS: Dict[str, BitmapCodec] = {
    "hdmv_pgs_subtitle": BitmapCodec("pgs", (1920, 1080)),
    "dvb_subtitle": BitmapCodec("dvb", (1920, 1080)),
    "dvd_subtitle": BitmapCodec("dvdsub", (720, 576)),
}

# Title words marking a script, for tracks tagged only "chi"/"zho". Checked in order, so a
# title naming both ("CHS&CHT") counts as Traditional.
_HANT_HINTS = ("cht", "tc", "big5", "traditional", "繁", "hant", "hk", "tw", "正體")
_HANS_HINTS = ("chs", "sc", "gb", "simplified", "简", "簡", "hans")
_HANT_TAGS = {"zh-hant", "zh-hk", "zh-tw", "zh-mo", "cht", "yue"}
_HANS_TAGS = {"zh-hans", "zh-cn", "zh-sg", "chs"}


@dataclass(frozen=True)
class SubtitleStream:
    """One subtitle stream as ffprobe reports it. ``index`` is ffmpeg's absolute stream
    index (``-map 0:N``), not the position among subtitle streams."""
    index: int
    codec: str
    kind: str                     # "text", "bitmap" or "other"
    language: str = ""
    title: str = ""
    forced: bool = False
    default: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _kind(codec: str) -> str:
    if codec in TEXT_CODECS:
        return "text"
    if codec in BITMAP_CODECS:
        return "bitmap"
    return "other"


def parse_streams(ffprobe_json: dict) -> List[SubtitleStream]:
    """The subtitle streams of an ``ffprobe -show_streams -of json`` result."""
    out = []
    for s in ffprobe_json.get("streams", []):
        if s.get("codec_type", "subtitle") != "subtitle":
            continue
        tags = s.get("tags") or {}
        disposition = s.get("disposition") or {}
        codec = s.get("codec_name") or ""
        out.append(SubtitleStream(
            index=int(s["index"]), codec=codec, kind=_kind(codec),
            language=(tags.get("language") or "").strip(), title=(tags.get("title") or "").strip(),
            forced=bool(disposition.get("forced")), default=bool(disposition.get("default")),
        ))
    return out


def probe_subtitle_streams(path: str) -> List[SubtitleStream]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
         "stream=index,codec_name,codec_type:stream_tags=language,title:"
         "stream_disposition=forced,default", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {proc.stderr.strip()[-300:]}")
    return parse_streams(json.loads(proc.stdout or "{}"))


def probe_video_size(path: str) -> Optional[Tuple[int, int]]:
    """The first video stream's (width, height), or None for audio-only input."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        s = json.loads(proc.stdout or "{}").get("streams", [])[0]
        return int(s["width"]), int(s["height"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def canvas_for(stream: SubtitleStream, video_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """The canvas a bitmap stream is rendered on: its format's own, or the video's if bigger."""
    w, h = BITMAP_CODECS[stream.codec].canvas
    if video_size:
        w, h = max(w, video_size[0]), max(h, video_size[1])
    return w, h


def is_chinese(stream: SubtitleStream) -> bool:
    if stream.language.lower() in _CHINESE_LANG_CODES | {"chs", "cht"}:
        return True
    if any(k in stream.title.lower() for k in _CHINESE_TITLE_KEYWORDS):
        return True
    # An untagged track titled only "CHT" or "繁體" is Chinese too.
    return stream.language.lower() in ("", "und") and script_of(stream) is not None


def script_of(stream: SubtitleStream) -> Optional[str]:
    """``"hant"``, ``"hans"`` or None (unknown) for a Chinese track, from its tag or title."""
    lang = stream.language.lower()
    if lang in _HANT_TAGS:
        return "hant"
    if lang in _HANS_TAGS:
        return "hans"
    words = stream.title.lower().replace("&", " ").replace("/", " ").replace("+", " ").split()
    title = stream.title.lower()
    for hints, script in ((_HANT_HINTS, "hant"), (_HANS_HINTS, "hans")):
        for hint in hints:
            # Latin hints must be whole words ("sc" is inside "disc"); CJK ones need not be.
            if (hint.isascii() and hint in words) or (not hint.isascii() and hint in title):
                return script
    return None


def lang_tag(stream: SubtitleStream) -> str:
    """A short language tag for file names: ``zh-Hant``, ``zh-Hans``, ``zh``, ``en``..."""
    if is_chinese(stream):
        script = script_of(stream)
        return {"hant": "zh-Hant", "hans": "zh-Hans"}.get(script, "zh")
    lang = stream.language.lower()
    return {"eng": "en", "jpn": "ja", "kor": "ko", "und": "und", "": "und"}.get(lang, lang)


def ocr_language(stream: SubtitleStream) -> Optional[str]:
    """The PaddleOCR language for a bitmap track, or None when it has no recogniser for it.

    The line recogniser (ocr.PADDLE_REC_MODEL) reads both scripts, so for Chinese this only
    matters to the ``detect`` mode, where PaddleOCR picks its models by language.
    """
    if is_chinese(stream):
        return "ch" if script_of(stream) == "hans" else "chinese_cht"
    return {"eng": "en", "en": "en", "jpn": "japan", "ja": "japan", "kor": "korean",
            "ko": "korean"}.get(stream.language.lower())


def select_streams(streams: Sequence[SubtitleStream], spec: str) -> List[SubtitleStream]:
    """The streams a ``--reference_tracks`` value asks for, in stream order.

    ``chinese`` (the default): every Chinese text or bitmap track. ``all``: every text or
    bitmap track. Otherwise a comma-separated list of absolute stream indices.
    """
    usable = [s for s in streams if s.kind != "other"]
    spec = (spec or "chinese").strip().lower()
    if spec == "chinese":
        return [s for s in usable if is_chinese(s)]
    if spec == "all":
        return usable
    try:
        wanted = [int(x) for x in spec.replace(" ", "").split(",") if x]
    except ValueError:
        raise ValueError(f"reference_tracks: expected chinese, all or stream indices, got {spec!r}")
    by_index = {s.index: s for s in streams}
    missing = [i for i in wanted if i not in by_index]
    if missing:
        raise ValueError(f"reference_tracks: no subtitle stream {missing} "
                         f"(subtitle streams: {sorted(by_index)})")
    bad = [i for i in wanted if by_index[i].kind == "other"]
    if bad:
        raise ValueError(f"reference_tracks: stream {bad} is {by_index[bad[0]].codec}, which "
                         "has no text or pictures to extract")
    return [by_index[i] for i in wanted]

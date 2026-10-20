"""Pull a media file's embedded subtitle tracks out as reference SRTs.

For each chosen track (probe.select_streams) this writes ``{name}.ref.{lang}.srt`` next to the
run's other outputs, and once per media file ``{name}.references.json``: every subtitle stream
found, what became of it (written, skipped and why), and ``best_chinese``, the track to use
as ``--reference_subtitle`` when the run asks for the embedded one.

Text tracks are converted in this process. Bitmap tracks go to the OCR worker (worker.py) as
one batch, so the OCR model loads once per run, in a process of its own.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from cantocaptions_ai.references.cues import write_srt
from cantocaptions_ai.references.ocr import DEFAULT_BACKEND, ocr_available
from cantocaptions_ai.references.probe import (
    SubtitleStream, canvas_for, is_chinese, lang_tag, ocr_language, probe_subtitle_streams,
    probe_video_size, script_of, select_streams,
)
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

MANIFEST_SUFFIX = ".references.json"
# Title words of tracks that carry only signs, songs or forced lines, not the dialogue.
_PARTIAL_TITLE_WORDS = ("sign", "song", "forced", "karaoke", "op/ed", "lyrics")


@dataclass
class ExtractedTrack:
    stream: dict
    method: str                       # "text" or "ocr"
    lang: str                         # the file-name tag: zh-Hant, zh, en...
    path: Optional[str] = None
    cues: int = 0
    ocr_lang: Optional[str] = None
    seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ReferenceSet:
    media: str
    name: str
    streams: List[dict] = field(default_factory=list)
    tracks: List[ExtractedTrack] = field(default_factory=list)
    skipped: List[dict] = field(default_factory=list)
    best_chinese: Optional[str] = None
    manifest: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def written(self) -> List[ExtractedTrack]:
        return [t for t in self.tracks if t.path and not t.error]


def manifest_path(out_dir: str, name: str) -> str:
    return os.path.normpath(os.path.join(out_dir, *name.split("/"))) + MANIFEST_SUFFIX


def load_manifest(path: str) -> ReferenceSet:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["tracks"] = [ExtractedTrack(**t) for t in data.get("tracks", [])]
    return ReferenceSet(**data)


def _partial(stream: dict) -> bool:
    title = (stream.get("title") or "").lower()
    return bool(stream.get("forced")) or any(w in title for w in _PARTIAL_TITLE_WORDS)


def best_chinese(tracks: List[ExtractedTrack]) -> Optional[str]:
    """The written Chinese track most worth reading as a reference: a full track before a
    signs-or-forced one, text before OCR (it has no recognition errors), Traditional before an
    unknown script before Simplified, then the one with more cues."""
    candidates = [t for t in tracks if t.path and not t.error and t.cues
                  and is_chinese(SubtitleStream(**t.stream))]
    if not candidates:
        return None
    rank = {"hant": 0, None: 1, "hans": 2}
    candidates.sort(key=lambda t: (_partial(t.stream), t.method != "text",
                                   rank[script_of(SubtitleStream(**t.stream))], -t.cues))
    return candidates[0].path


def _unique_path(out_dir: str, name: str, tag: str, stream_index: int, used: set) -> str:
    base = os.path.normpath(os.path.join(out_dir, *name.split("/")))
    path = f"{base}.ref.{tag}.srt"
    if path in used:
        path = f"{base}.ref.{tag}.{stream_index}.srt"
    used.add(path)
    return path


def _run_worker(request: dict, status: Optional[Callable[[str], None]],
                labels: List[str], reporter=None) -> dict:
    """Run the OCR worker on *request* and return its result. Progress drives *reporter*'s
    bar (set_total / advance), one bar per track, or else goes to *status* as text."""
    with tempfile.TemporaryDirectory(prefix="cantoref_") as tmp:
        req, res, err = (os.path.join(tmp, n) for n in ("request.json", "result.json", "stderr.log"))
        with open(req, "w", encoding="utf-8") as f:
            json.dump(request, f, ensure_ascii=False)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        with open(err, "w", encoding="utf-8", errors="replace") as err_f, subprocess.Popen(
                [sys.executable, "-m", "cantocaptions_ai.references.worker", req, res],
                stdout=subprocess.PIPE, stderr=err_f, stdin=subprocess.DEVNULL, env=env,
                text=True, encoding="utf-8", errors="replace") as proc:
            current, shown = None, 0
            for line in proc.stdout:
                parts = line.split()
                if len(parts) != 4 or parts[0] != "@progress":
                    continue
                job, done, total = (int(p) for p in parts[1:])
                if reporter is not None:
                    if job != current:
                        current, shown = job, 0
                        if status is not None:
                            status(f"OCR {labels[job]}")
                        reporter.set_total(total, unit="caption")
                    reporter.advance(done - shown)
                    shown = done
                elif status is not None:
                    status(f"OCR {labels[job]}: {done}/{total} captions")
            code = proc.wait()
        with open(err, encoding="utf-8", errors="replace") as f:
            log = f.read()
        if log.strip():
            logger.debug("OCR worker output:\n%s", log.strip()[-20000:])
        if code != 0 or not os.path.exists(res):
            tail = "\n".join(log.strip().splitlines()[-8:])
            raise RuntimeError(f"the OCR worker failed (exit {code}):\n{tail}")
        with open(res, encoding="utf-8") as f:
            return json.load(f)


def extract_references(
    media: str, out_dir: str, name: Optional[str] = None, *, tracks: str = "chinese",
    ocr: str = "auto", device: str = "auto", mode: str = "line", ocr_lang: Optional[str] = None,
    cache_only: bool = False, status: Optional[Callable[[str], None]] = None, reporter=None,
) -> ReferenceSet:
    """Extract *media*'s embedded subtitle tracks into *out_dir* and write the manifest.

    *tracks*: ``chinese``, ``all`` or stream indices (probe.select_streams). *ocr*: ``auto``
    reads bitmap tracks when an OCR engine is installed and skips them with a warning
    otherwise; ``off`` never runs OCR. *device* is the OCR worker's (``auto``, ``cuda``, ``cpu``).
    """
    name = name or os.path.splitext(os.path.basename(media))[0]
    os.makedirs(os.path.dirname(manifest_path(out_dir, name)) or out_dir, exist_ok=True)
    streams = probe_subtitle_streams(media)
    chosen = select_streams(streams, tracks)
    result = ReferenceSet(media=os.path.abspath(media), name=name,
                          streams=[s.as_dict() for s in streams])
    chosen_ids = {s.index for s in chosen}
    for s in streams:
        if s.index not in chosen_ids:
            reason = (f"{s.codec} has no text or pictures to extract" if s.kind == "other"
                      else "not selected (reference_tracks)")
            result.skipped.append({"stream": s.as_dict(), "reason": reason})

    used: set = set()
    jobs, job_tracks = [], []
    ocr_problem = None if ocr == "off" else ocr_available(DEFAULT_BACKEND)
    video_size = None
    for s in chosen:
        tag = lang_tag(s)
        if s.kind == "text":
            from cantocaptions_ai.references.text import extract_text_track
            t0 = time.perf_counter()
            track = ExtractedTrack(stream=s.as_dict(), method="text", lang=tag)
            try:
                cues = extract_text_track(media, s.index)
                track.path = _unique_path(out_dir, name, tag, s.index, used)
                write_srt(cues, track.path)
                track.cues = len(cues)
            except (RuntimeError, OSError, ValueError) as e:
                track.error = str(e)
            track.seconds = round(time.perf_counter() - t0, 1)
            result.tracks.append(track)
            continue
        if ocr == "off":
            result.skipped.append({"stream": s.as_dict(), "reason": "bitmap track; OCR is off"})
            continue
        if ocr_problem:
            result.skipped.append({"stream": s.as_dict(), "reason": f"bitmap track; {ocr_problem}"})
            continue
        lang = ocr_lang or ocr_language(s)
        if lang is None:
            result.skipped.append({"stream": s.as_dict(),
                                   "reason": f"no OCR language for {s.language or 'an untagged track'} "
                                             "(set ocr_lang)"})
            continue
        if video_size is None:
            video_size = probe_video_size(media) or (0, 0)
        track = ExtractedTrack(stream=s.as_dict(), method="ocr", lang=tag, ocr_lang=lang,
                               path=_unique_path(out_dir, name, tag, s.index, used))
        jobs.append({"media": os.path.abspath(media), "stream_index": s.index,
                     "canvas": list(canvas_for(s, video_size)), "lang": lang, "out_srt": track.path})
        job_tracks.append(track)

    if jobs:
        labels = [f"stream {j['stream_index']} ({t.lang})" for j, t in zip(jobs, job_tracks)]
        reply = _run_worker({"jobs": jobs, "backend": DEFAULT_BACKEND, "device": device,
                             "mode": mode, "cache_only": cache_only}, status, labels, reporter)
        for track, outcome in zip(job_tracks, reply["jobs"]):
            if outcome.get("ok"):
                track.cues, track.seconds = outcome["cues"], outcome.get("seconds")
                logger.debug("OCR of stream %s: %s", track.stream["index"], outcome)
            else:
                track.error = outcome.get("error") or "OCR failed"
                track.path = None
        result.tracks.extend(job_tracks)

    result.tracks.sort(key=lambda t: t.stream["index"])
    result.best_chinese = best_chinese(result.tracks)
    result.manifest = manifest_path(out_dir, name)
    with open(result.manifest, "w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, ensure_ascii=False, indent=2)
    return result


def describe(result: ReferenceSet) -> List[str]:
    """One line per track for the console: what was written, skipped or failed."""
    lines = []
    for t in result.tracks:
        s = t.stream
        what = f"stream {s['index']} {s['codec']}" + (f" {s['language']}" if s.get("language") else "") \
            + (f" \"{s['title']}\"" if s.get("title") else "")
        if t.error:
            lines.append(f"{what}: failed ({t.error})")
        else:
            lines.append(f"{what}: {t.cues} cues by {t.method} -> {os.path.normpath(t.path)}")
    for sk in result.skipped:
        s = sk["stream"]
        if sk["reason"].startswith("not selected"):
            continue
        lines.append(f"stream {s['index']} {s['codec']}: skipped, {sk['reason']}")
    return lines

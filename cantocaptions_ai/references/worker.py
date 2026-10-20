"""The OCR worker: renders bitmap subtitle streams and reads them, in a process of its own.

    python -m cantocaptions_ai.references.worker REQUEST.json RESULT.json

Paddle and torch never share a process. Each bundles its own CUDA, cuDNN and OpenMP runtimes,
and on Windows a second copy loaded into a process that already holds the other's fails or
misbehaves; Paddle's allocator also keeps its VRAM until the process exits. So the pipeline
starts this worker before VAD, waits for it, and only then loads its own models: by the time
torch touches the GPU, Paddle has released all of it.

The request names the jobs (media, stream, canvas, OCR language, output SRT) and the backend;
the parent works all of that out (extract.py), so this module needs neither ffprobe nor any
pipeline code beyond raster.py, ocr.py and cues.py. Progress goes to stdout as
``@progress JOB DONE TOTAL`` lines; everything else Paddle prints goes to stderr, which the
parent logs at DEBUG.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback


def _hide_torch() -> None:
    """Make torch look uninstalled to everything in this process.

    PaddleX imports modelscope, whose logger imports torch whenever
    ``importlib.util.find_spec("torch")`` finds it. With Paddle's CUDA runtime already loaded,
    torch's own DLLs then fail to load on Windows (``WinError 127`` loading ``torch\\lib\\shm.dll``)
    and the import of ``paddleocr`` dies. Nothing the OCR needs uses torch, so answering
    find_spec with None for it keeps torch out entirely. ``sys.modules["torch"] = None`` would
    be the usual way, but SciPy's array-API helpers then read that None as the module.
    """
    import importlib.util
    real = importlib.util.find_spec
    if getattr(real, "_hides_torch", False):
        return

    def find_spec(name, package=None):
        if name == "torch" or name.startswith("torch."):
            return None
        return real(name, package)

    find_spec._hides_torch = True
    importlib.util.find_spec = find_spec


def _progress(job: int, done: int, total: int) -> None:
    print(f"@progress {job} {done} {total}", flush=True)


def run(request: dict) -> dict:
    from cantocaptions_ai.references.cues import Cue, tidy, write_srt
    from cantocaptions_ai.references.ocr import OCR_BACKENDS
    from cantocaptions_ai.references.raster import frame_events, iter_event_images, render_frames

    backends: dict = {}
    results = []
    for j, job in enumerate(request["jobs"]):
        t0 = time.perf_counter()
        try:
            key = (request.get("backend", "paddle"), job["lang"])
            if key not in backends:
                cls = OCR_BACKENDS[key[0]]
                backends[key] = cls(job["lang"], device=request.get("device", "auto"),
                                    mode=request.get("mode", "line"),
                                    cache_only=bool(request.get("cache_only")))
            backend = backends[key]
            with tempfile.TemporaryDirectory(prefix="cantoref_frames_") as tmp:
                frames = render_frames(job["media"], job["stream_index"], tuple(job["canvas"]), tmp)
                events = frame_events(frames)
                _progress(j, 0, len(events))
                cache: dict = {}
                cues = []
                pending = []          # (event, images) not yet read

                def flush():
                    unread = list({im.key: im for _, im in pending if im.key not in cache}.values())
                    for im, text in zip(unread, backend.read(unread) if unread else []):
                        cache[im.key] = text
                    for event, im in pending:
                        cues.append(Cue(event.start, event.end, cache[im.key]))
                    pending.clear()

                for n, (event, images) in enumerate(iter_event_images(events), 1):
                    if images is not None:
                        pending.append((event, images))
                    if len(pending) >= 32:
                        flush()
                        _progress(j, n, len(events))
                flush()
                _progress(j, len(events), len(events))
            cues = tidy(cues)
            write_srt(cues, job["out_srt"])
            results.append({"ok": True, "cues": len(cues), "frames": len(frames),
                            "events": len(events), "seconds": round(time.perf_counter() - t0, 1),
                            "device": getattr(backend, "_device", None),
                            "mode": getattr(backend, "mode", None)})
        except Exception as e:     # one bad track must not lose the others
            traceback.print_exc(file=sys.stderr)
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return {"jobs": results}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    _hide_torch()
    with open(argv[0], encoding="utf-8") as f:
        request = json.load(f)
    result = run(request)
    with open(argv[1], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

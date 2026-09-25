"""Score VAD backends and settings against the test set from build_vad_testset.py.

Each backend scores every stream once; its speech-probability curve is cached, so a sweep
over thresholds and padding re-runs only the post-processing -- through the pipeline's own
``speech_regions``/``merge_chunks``, so what is scored is exactly what the pipeline would
do. The metrics are defined in cantocaptions_ai/utils/vad_eval.py.

    # the shipped settings, both backends
    uv run python scripts/eval_vad.py --backends pyannote silero

    # one explicit setting
    uv run python scripts/eval_vad.py --backends silero --onset 0.2 --offset 0.2 \\
        --pad_onset 0.25 --pad_offset 0.2 --min_duration_off 1.0

    # sweep the grid, write every row, print each backend's frontier
    uv run python scripts/eval_vad.py --backends pyannote silero --sweep --out sweep_dev.jsonl

Runtime is reported per backend as scoring seconds per hour of audio, measured on this
machine the first time a curve is computed (cached curves keep their original timing;
--recompute re-measures).
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cantocaptions_ai.pipeline.config import PipelineConfig  # noqa: E402
from cantocaptions_ai.utils.vad_eval import StreamReference, Tolerances, summarize  # noqa: E402

DEFAULT_TESTSET = os.path.join(REPO_ROOT, ".cache", "vad_testset")
DEFAULT_CACHE = os.path.join(REPO_ROOT, ".cache", "vad_curves")
BACKENDS = ("pyannote", "silero")

GRID = {
    "onset": [0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7],
    "offset_gap": [0.0, 0.15, 0.3],  # offset = onset - gap
    "pad_onset": [0.0, 0.25, 0.5, 1.0],
    "pad_offset": [0.1, 0.2, 0.4, 0.6],
    "min_duration_off": [0.25, 0.5, 1.0],
}


def load_streams(testset, split):
    with open(os.path.join(testset, split, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest, manifest["streams"]


def reference_for(stream, tol):
    return StreamReference(
        duration=stream["duration"],
        speech=[(s, e, (clip, idx)) for s, e, clip, idx in stream["speech"]],
        dontcare=[tuple(x) for x in stream["dontcare"]],
        negatives=[(c["start"], c["end"]) for c in stream["clips"] if c["kind"] == "gap"],
        tol=tol,
    )


def make_backend(name, args):
    from cantocaptions_ai.pipeline.vad import load_vad
    return load_vad(vad_method=name, device=args.device, vad_onset=0.5).vad_model


def compute_curves(name, streams, testset, split, args):
    """Score every stream with one backend (or read the cache). Returns curves + timing."""
    import soundfile as sf
    from pyannote.core import SlidingWindow, SlidingWindowFeature

    cache_dir = os.path.join(args.cache, split, name)
    os.makedirs(cache_dir, exist_ok=True)
    curves, timing = {}, {"score_s": 0.0, "audio_s": 0.0, "load_s": None, "peak_vram_mb": None}
    model = None
    for st in streams:
        path = os.path.join(cache_dir, st["episode_id"] + ".npz")
        if os.path.isfile(path) and not args.recompute:
            z = np.load(path)
            win = SlidingWindow(start=float(z["start"]), duration=float(z["duration"]),
                                step=float(z["step"]))
            curves[st["episode_id"]] = SlidingWindowFeature(z["data"], win)
            timing["score_s"] += float(z["score_s"])
            timing["audio_s"] += st["duration"]
            if "load_s" in z:
                timing["load_s"] = float(z["load_s"])
            if "peak_vram_mb" in z:
                timing["peak_vram_mb"] = float(z["peak_vram_mb"])
            continue
        import torch
        if model is None:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            model = make_backend(name, args)
            timing["load_s"] = time.perf_counter() - t
        audio, sr = sf.read(os.path.join(testset, split, st["audio"]), dtype="float32")
        wav = model.preprocess_audio(audio)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        curve = model({"waveform": wav, "sample_rate": sr})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        timing["peak_vram_mb"] = max(timing["peak_vram_mb"] or 0.0, vram)
        win = curve.sliding_window
        np.savez(path, data=curve.data.astype(np.float32), start=win.start,
                 duration=win.duration, step=win.step, score_s=dt, load_s=timing["load_s"],
                 peak_vram_mb=timing["peak_vram_mb"])
        curves[st["episode_id"]] = curve
        timing["score_s"] += dt
        timing["audio_s"] += st["duration"]
        print(f"    {name:10s} {st['episode_id']:55s} {dt:6.1f}s", file=sys.stderr)
    timing["s_per_hour"] = timing["score_s"] / (timing["audio_s"] / 3600)
    return curves, timing


def evaluate(curves, refs, streams, p, chunk_size):
    from cantocaptions_ai.pipeline.vads import CurveVad
    det, cov = [], []
    for st in streams:
        curve, ref = curves[st["episode_id"]], refs[st["episode_id"]]
        regions = CurveVad.speech_regions(
            curve, onset=p["onset"], offset=p["offset"], min_duration_off=p["min_duration_off"])
        det.append(ref.score_detection(regions))
        chunks = CurveVad.merge_chunks(
            curve, chunk_size, onset=p["onset"], offset=p["offset"],
            pad_onset=p["pad_onset"], pad_offset=p["pad_offset"],
            min_duration_off=p["min_duration_off"])
        cov.append(ref.score_coverage([(c["start"], c["end"]) for c in chunks]))
    return summarize(det, cov)


def grid_points():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        p = dict(zip(keys, values))
        p["offset"] = round(max(p["onset"] - p.pop("offset_gap"), 0.05), 3)
        yield p


def pareto(rows, better_x="kept_frac", better_y="speech_recall"):
    """Rows not beaten on both recall (higher) and kept audio (lower)."""
    rows = sorted(rows, key=lambda r: (r[better_x], -r[better_y]))
    front, best = [], -1.0
    for r in rows:
        if r[better_y] > best + 1e-9:
            front.append(r)
            best = r[better_y]
    return front


def fmt(r):
    return (f"on={r['onset']:.2f} off={r['offset']:.2f} pad={r['pad_onset']:.2f}/"
            f"{r['pad_offset']:.2f} gap={r['min_duration_off']:.2f} | "
            f"recall={r['speech_recall']:.4f} clipped={r['cues_clipped']:.4f} "
            f"missed={r['cues_missed']:.4f} split={r['cues_split']:.3f} "
            f"kept={r['kept_frac']:.3f} nonsp={r['nonspeech_kept_frac']:.3f} "
            f"gapkept={r['gap_kept_frac']:.3f} | detF1={r['det_f1']:.3f} "
            f"P={r['det_precision']:.3f} R={r['det_recall']:.3f} chunks={r['chunks']}")


def main():
    defaults = PipelineConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--testset", default=DEFAULT_TESTSET)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--backends", nargs="+", default=["pyannote"], choices=BACKENDS)
    ap.add_argument("--device", default="cuda", help="for pyannote; silero is always CPU")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="score-curve cache directory")
    ap.add_argument("--recompute", action="store_true", help="ignore cached curves")
    ap.add_argument("--chunk_size", type=int, default=defaults.chunk_size)
    ap.add_argument("--onset", type=float, default=defaults.vad_onset)
    ap.add_argument("--offset", type=float, default=defaults.vad_offset)
    ap.add_argument("--pad_onset", type=float, default=defaults.vad_pad_onset)
    ap.add_argument("--pad_offset", type=float, default=defaults.vad_pad_offset)
    ap.add_argument("--min_duration_off", type=float, default=defaults.vad_min_duration_off)
    ap.add_argument("--sweep", action="store_true", help="evaluate the whole GRID")
    ap.add_argument("--out", default=None, help="write every evaluated row here (JSONL)")
    ap.add_argument("--by_stratum", action="store_true",
                    help="also break the single-setting result down by stratum")
    args = ap.parse_args()

    manifest, streams = load_streams(args.testset, args.split)
    tol = Tolerances()
    refs = {st["episode_id"]: reference_for(st, tol) for st in streams}
    hours = sum(st["duration"] for st in streams) / 3600
    print(f"{args.split}: {len(streams)} streams, {hours:.2f} h, "
          f"{sum(len(r.cue_cores) for r in refs.values())} lines")

    out = open(args.out, "w", encoding="utf-8") if args.out else None
    for name in args.backends:
        curves, timing = compute_curves(name, streams, args.testset, args.split, args)
        vram = f", peak VRAM {timing['peak_vram_mb']:.0f} MB" if timing["peak_vram_mb"] else ""
        load = f"load {timing['load_s']:.1f}s, " if timing["load_s"] is not None else ""
        print(f"\n== {name}: {load}{timing['s_per_hour']:.1f} s per audio hour{vram}")
        points = list(grid_points()) if args.sweep else [{
            "onset": args.onset, "offset": args.offset, "pad_onset": args.pad_onset,
            "pad_offset": args.pad_offset, "min_duration_off": args.min_duration_off}]
        rows = []
        t = time.perf_counter()
        for p in points:
            row = {"backend": name, "split": args.split, **p,
                   **evaluate(curves, refs, streams, p, args.chunk_size),
                   "s_per_hour": timing["s_per_hour"]}
            rows.append(row)
            if out:
                out.write(json.dumps(row) + "\n")
        if not args.sweep:
            print("  " + fmt(rows[0]))
            if args.by_stratum:
                for stratum in sorted({st["stratum"] for st in streams}):
                    sub = [st for st in streams if st["stratum"] == stratum]
                    r = evaluate(curves, refs, sub, points[0], args.chunk_size)
                    print(f"    {stratum:14s} " + fmt({**points[0], **r}))
            continue
        print(f"  {len(rows)} settings in {time.perf_counter() - t:.0f}s; frontier "
              "(no setting keeps less audio AND recalls more speech):")
        for r in pareto(rows):
            print("  " + fmt(r))
    if out:
        out.close()


if __name__ == "__main__":
    main()

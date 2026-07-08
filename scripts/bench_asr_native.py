"""Diagnostic harness for the native Qwen3-ASR backend's per-batch throughput and VRAM.

Purpose: pin down the ~1.3-1.4x ASR slowdown that appeared *within* the native
backend after the VRAM-headroom work (same backend as the faster earlier runs —
a regression, not a native-vs-legacy difference), and that persists even with the
new VRAM features disabled (--vram_checks False / --vram_headroom_mb 0).

It drives the REAL QwenPipelineNative through the production load_model_native /
_infer_batch path (not a reimplementation), so running it at a different git
checkout measures *that checkout's* hot path — which is what makes it a bisection
tool (see the diagnosis ladder below). The batching shell here mirrors
QwenPipelineNative.run()'s lines ~148-159 (build a flat (seg_idx,) job list ->
optionally reorder -> run_adaptive_batches(jobs, batch_size, infer_fn) where
infer_fn calls the real pipe._infer_batch), keeping model/generate/processor
production-identical while exposing batch *order* as a knob (run() itself hardcodes
VAD order).

Toggles, and the question each answers:
  --warn-vram/--no-warn-vram   is the always-on _warn_vram pre-guard math (which
                               _infer_batch runs on CUDA even when vram_checks is
                               False) the cost? (default off = production w/ flags off)
  --cap-headroom-mb N          does capping the allocator (cap_cuda_memory) matter?
                               -1 = off (default); >=0 caps after load.
  --sort {none,desc,asc}       none = current production (VAD) order; desc = longest-
                               first (as alignment.py already does) — does reserved
                               VRAM climb under none and flatten under desc?

Per-batch it prints dt, ms/seg, max_audio_s, and the reserved_mb DELTA from the
previous batch (via the existing vram_stats helper in model_utils, not a new one),
so a monotonic reserved climb is directly visible. Aggregates: total ms/seg, peak
GB, and reserved first-batch vs last-batch.

CAVEAT (as in bench_asr_batching.py:11-14): random-noise audio makes generation
length/stopping undefined, so absolute ms/seg on noise is NOT representative and
may *understate* the sort effect (noise clips don't run to the token lengths real
speech does). Pass --audio-dir DIR of real 16 kHz wavs for a truthful number;
noise is the quick smoke path. Requires the model locally cached
(local_files_only=True); run a normal `cantocaptions` invocation once first if not.

Diagnosis ladder (run on the GPU, cheapest first):
  1. `git stash` the uncommitted changes, rerun. If ms/seg recovers, the culprit
     is in this session's working tree — unstash and narrow with the toggles above.
     If unchanged, the regression is already committed (in/around c596de7).
  2. If committed: run this script at c596de7 vs its parent (git worktree, or copy
     the script in — it calls stable production APIs) to confirm/locate it.
  3. At HEAD, toggle --warn-vram, --cap-headroom-mb 0 vs 512, --sort none vs desc.

Usage:
    uv run python scripts/bench_asr_native.py
    uv run python scripts/bench_asr_native.py --n-segments 120 --batch-size 8 --sort desc
    uv run python scripts/bench_asr_native.py --warn-vram --cap-headroom-mb 512
    uv run python scripts/bench_asr_native.py --audio-dir ./samples --n-segments 40
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cantocaptions_ai.pipeline._asr_native import load_model_native
from cantocaptions_ai.utils.audio import SAMPLE_RATE, load_audio
from cantocaptions_ai.utils.model_utils import cap_cuda_memory, run_adaptive_batches, vram_stats


def make_noise_segments(rng: np.random.Generator, n: int, min_s: float, max_s: float) -> list:
    """VAD-segment-like dicts of random noise, variable durations over the chunk_size range."""
    segments = []
    t = 0.0
    for _ in range(n):
        duration = float(rng.uniform(min_s, max_s))
        audio = rng.standard_normal(int(duration * SAMPLE_RATE)).astype(np.float32)
        segments.append({"start": t, "end": t + duration, "audio": audio})
        t += duration + float(rng.uniform(0.1, 1.0))
    return segments


def load_dir_segments(audio_dir: str, n: int, min_s: float, max_s: float) -> list:
    """Real 16 kHz wavs from *audio_dir*, sliced into <=max_s VAD-segment-like chunks.

    Gives a truthful ms/seg (generation runs to real token lengths) unlike noise;
    cycles through the directory's wavs until *n* segments are produced.
    """
    paths = sorted(p for p in Path(audio_dir).iterdir() if p.suffix.lower() in (".wav", ".flac", ".mp3"))
    if not paths:
        raise SystemExit(f"--audio-dir {audio_dir} has no .wav/.flac/.mp3 files")
    segments = []
    t = 0.0
    win = int(max_s * SAMPLE_RATE)
    for path in paths:
        audio = load_audio(str(path))
        for off in range(0, len(audio), win):
            chunk = audio[off:off + win]
            if len(chunk) < int(min_s * SAMPLE_RATE):
                continue
            duration = len(chunk) / SAMPLE_RATE
            segments.append({"start": t, "end": t + duration, "audio": chunk})
            t += duration + 0.3
            if len(segments) >= n:
                return segments
    if not segments:
        raise SystemExit(f"--audio-dir {audio_dir}: no chunks >= --min-s produced")
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-segments", type=int, default=80, help="VAD segments to transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sort", choices=("none", "desc", "asc"), default="none",
                        help="batch order: none=VAD/production, desc=longest-first, asc=shortest-first")
    parser.add_argument("--min-s", type=float, default=1.0, help="min segment duration (s)")
    parser.add_argument("--max-s", type=float, default=30.0, help="max segment duration (s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap-headroom-mb", type=float, default=-1.0,
                        help="-1 = no allocator cap (default); >=0 calls cap_cuda_memory after load")
    parser.add_argument("--warn-vram", dest="warn_vram", action="store_true",
                        help="run the _warn_vram pre-guard math + vram_stats round-trip (default off = production w/ flags off)")
    parser.add_argument("--no-warn-vram", dest="warn_vram", action="store_false")
    parser.set_defaults(warn_vram=False)
    parser.add_argument("--iters", type=int, default=1, help="timed repetitions over the full segment set")
    parser.add_argument("--model", default="Qwen3-ASR", help="model key or HF id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audio-dir", default=None, help="dir of real 16 kHz wavs (truthful ms/seg); default is noise")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU (pass --device cpu to run without one).")
    on_cuda = args.device == "cuda"

    # Loader cap off (vram_headroom_mb=0) so THIS harness owns capping via --cap-headroom-mb;
    # warn-vram wiring goes through vram_checks so --warn-vram exercises the real gate.
    pipe = load_model_native(
        model_name=args.model,
        device=args.device,
        device_index=0,
        local_files_only=True,
        batch_size=args.batch_size,
        vram_checks=args.warn_vram,
        vram_headroom_mb=0,
    )

    if on_cuda and args.cap_headroom_mb >= 0:
        cap_cuda_memory(0, args.cap_headroom_mb)

    if args.audio_dir:
        segments = load_dir_segments(args.audio_dir, args.n_segments, args.min_s, args.max_s)
    else:
        rng = np.random.default_rng(args.seed)
        segments = make_noise_segments(rng, args.n_segments, args.min_s, args.max_s)
    n = len(segments)

    from cantocaptions_ai.pipeline.asr import _normalize_language
    language = _normalize_language(pipe.preset_language or "yue")

    print(
        f"model={args.model}  n_segments={n}  batch_size={args.batch_size}  sort={args.sort}  "
        f"warn_vram={args.warn_vram}  cap_headroom_mb={args.cap_headroom_mb}  "
        f"src={'audio-dir' if args.audio_dir else 'noise'}"
    )
    if not args.audio_dir:
        print("  (noise audio: absolute ms/seg not representative; use --audio-dir for a truthful number)")

    # Mirror run()'s batching shell: flat job list over segment indices, optional reorder,
    # then run_adaptive_batches driving the real pipe._infer_batch.
    def build_jobs() -> list:
        jobs = list(range(n))
        if args.sort == "desc":
            jobs.sort(key=lambda i: len(segments[i]["audio"]), reverse=True)
        elif args.sort == "asc":
            jobs.sort(key=lambda i: len(segments[i]["audio"]))
        return jobs

    # Warm up (kernel compilation / cuDNN autotune / any compile warmup) before timing.
    pipe._infer_batch([segments[i]["audio"] for i in range(min(2, n))], language)
    if on_cuda:
        torch.cuda.synchronize()

    for it in range(args.iters):
        jobs = build_jobs()
        batch_no = [0]
        prev_reserved = [None]
        first_reserved = [None]
        last_reserved = [None]

        def infer_fn(batch):
            wavs = [segments[i]["audio"] for i in batch]
            if on_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe._infer_batch(wavs, language)
            if on_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            max_audio_s = max(len(w) for w in wavs) / SAMPLE_RATE
            reserved = None
            delta = float("nan")
            if on_cuda:
                stats = vram_stats(0)
                if stats is not None:
                    reserved = stats["reserved_mb"]
                    if prev_reserved[0] is not None:
                        delta = reserved - prev_reserved[0]
                    if first_reserved[0] is None:
                        first_reserved[0] = reserved
                    last_reserved[0] = reserved
                    prev_reserved[0] = reserved
            batch_no[0] += 1
            reserved_str = f"{reserved:8.0f}" if reserved is not None else "     n/a"
            print(
                f"  batch {batch_no[0]:>3} (n={len(batch):>2}): {dt*1000:8.1f} ms  "
                f"{dt/len(batch)*1000:7.1f} ms/seg  max_audio={max_audio_s:5.1f}s  "
                f"reserved={reserved_str} MB  d_reserved={delta:+8.1f} MB"
            )

        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_adaptive_batches(jobs, args.batch_size, infer_fn)
        if on_cuda:
            torch.cuda.synchronize()
        total = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if on_cuda else 0.0

        print(
            f"iter {it+1}/{args.iters}: {total:6.2f}s total, {total/n*1000:7.1f} ms/seg, "
            f"peak_mem={peak_gb:.2f} GB"
            + (
                f", reserved first={first_reserved[0]:.0f} MB last={last_reserved[0]:.0f} MB "
                f"(climb={last_reserved[0]-first_reserved[0]:+.0f} MB)"
                if on_cuda and first_reserved[0] is not None else ""
            )
        )
        print()


if __name__ == "__main__":
    main()

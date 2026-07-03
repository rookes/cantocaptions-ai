"""Benchmark whether the legacy ASR backend's cross-file batch backfill actually pays off.

Compares two call patterns against the real Qwen3-ASR model:
  - OLD: one Qwen3ASRModel.transcribe() call per file (pre-refactor behaviour;
    each file's segments are batched internally, but never combined across files)
  - NEW: one combined transcribe() call across all files' segments (current
    QwenPipelineLegacy.run() behaviour; lets a ragged tail batch from one file
    be filled out with the next file's segments)

Uses random noise as audio (no real speech), so absolute per-segment latency is
not meaningful (generation length/stopping is undefined for noise) — only the
OLD vs. NEW ratio is. Requires the model to be locally cached
(local_files_only=True); run once online first if not, e.g. via a normal
`cantocaptions` invocation.

Usage:
    uv run python scripts/bench_asr_batching.py
    uv run python scripts/bench_asr_batching.py --segments-per-file 12,30,45,18,25
"""

import argparse
import time

import numpy as np
import torch


def make_file(rng: np.random.Generator, n_segs: int, min_s: float = 1.0, max_s: float = 15.0) -> list:
    """A list of (audio, sample_rate) tuples mimicking variable-length VAD segments."""
    return [
        (rng.standard_normal(int(rng.uniform(min_s, max_s) * 16000)).astype(np.float32), 16000)
        for _ in range(n_segs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments-per-file", default="12,30,45,18,25",
        help="comma-separated segment counts, one per simulated file",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="HF model id (legacy backend)")
    parser.add_argument("--batch-size", type=int, default=24, help="max_inference_batch_size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")

    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=200,
        torch_dtype="float16",
        device_map="cuda",
        local_files_only=True,
    )

    rng = np.random.default_rng(args.seed)
    counts = [int(n) for n in args.segments_per_file.split(",")]
    files = [make_file(rng, n) for n in counts]
    total_segs = sum(len(f) for f in files)
    print(f"files: {counts}  total_segs={total_segs}  batch_size={args.batch_size}")

    # Warm up (model/kernel compilation, cuDNN autotune) before timing either path.
    model.transcribe(files[0][:4], language="Cantonese")
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for f in files:
        model.transcribe(f, language="Cantonese")
    torch.cuda.synchronize()
    old_time = time.perf_counter() - t0
    print(f"OLD (per-file calls): {old_time:6.2f}s total, {old_time/total_segs*1000:7.1f} ms/seg")

    torch.cuda.reset_peak_memory_stats()
    combined = [seg for f in files for seg in f]
    t0 = time.perf_counter()
    model.transcribe(combined, language="Cantonese")
    torch.cuda.synchronize()
    new_time = time.perf_counter() - t0
    print(f"NEW (combined call):  {new_time:6.2f}s total, {new_time/total_segs*1000:7.1f} ms/seg")

    print(f"speedup: {old_time/new_time:.2f}x")


if __name__ == "__main__":
    main()

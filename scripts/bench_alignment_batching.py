"""Benchmark whether batching VAD segments through the wav2vec2-BERT alignment
model actually speeds up the emission-precompute step in align().

Compares two call patterns against the real Cantonese alignment model:
  - OLD: _compute_vad_emissions_sequential() — one forward pass per VAD segment,
    one at a time (production behavior before batching was added; this is what
    caused a 30+ second stall between files in a batch run, with no progress
    feedback, on files with many VAD segments)
  - NEW: _compute_vad_emissions_batched() — segments grouped into batches via
    run_adaptive_batches(), one padded forward pass per batch

Both call the actual production functions in cantocaptions_ai.pipeline.alignment
(not reimplementations), so this measures the real code and can't drift out of
sync with it.

Uses random noise as audio (no real speech) at variable, VAD-segment-like
durations (0.5s-30s, spanning the real --chunk_size range), so absolute
per-segment latency isn't meaningful — only the OLD vs. NEW ratio and the
correctness check are. Requires the real alignment model to be locally cached
(local_files_only=True); run once online first if not, e.g. via a normal
`cantocaptions` invocation with alignment enabled.

Usage:
    uv run python scripts/bench_alignment_batching.py
    uv run python scripts/bench_alignment_batching.py --n-segments 120 --batch-sizes 1,2,4,8,16
"""

import argparse
import time

import numpy as np
import torch

from cantocaptions_ai.pipeline.alignment import (
    _compute_vad_emissions_batched,
    _compute_vad_emissions_sequential,
    load_align_model,
    load_bert_processor,
)
from cantocaptions_ai.utils.audio import SAMPLE_RATE


def make_vad_segments(rng: np.random.Generator, n: int, min_s: float = 0.5, max_s: float = 30.0) -> list:
    """VAD-segment-like dicts with variable durations spanning the real chunk_size range."""
    segments = []
    t = 0.0
    for _ in range(n):
        duration = float(rng.uniform(min_s, max_s))
        audio = rng.standard_normal(int(duration * SAMPLE_RATE)).astype(np.float32)
        segments.append({"start": t, "end": t + duration, "audio": audio})
        t += duration + float(rng.uniform(0.1, 1.0))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-segments", type=int, default=80, help="VAD segments in the simulated file")
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="comma-separated batch sizes to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-3, help="tolerance for the batched-vs-sequential correctness check")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU (pass --device cpu to run without one).")

    device = args.device
    model, _ = load_align_model("yue", device, model_cache_only=True)
    bert_processor = load_bert_processor(model_cache_only=True)
    model.eval()  # dropout must be off for the sequential/batched outputs to be comparable at all

    rng = np.random.default_rng(args.seed)
    segments = make_vad_segments(rng, args.n_segments)
    print(f"n_segments={args.n_segments}  device={device}")

    # Warm up (kernel compilation/cuDNN autotune) before timing either path.
    _compute_vad_emissions_sequential(segments[:2], model, "huggingface", bert_processor, device)
    if device == "cuda":
        torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sequential = _compute_vad_emissions_sequential(segments, model, "huggingface", bert_processor, device)
    if device == "cuda":
        torch.cuda.synchronize()
    old_time = time.perf_counter() - t0
    old_peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(
        f"OLD (sequential, one segment at a time): {old_time:6.2f}s total, "
        f"{old_time/args.n_segments*1000:7.1f} ms/seg, peak_mem={old_peak:.2f} GB"
    )
    print()

    for batch_size in (int(b) for b in args.batch_sizes.split(",")):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        batched = _compute_vad_emissions_batched(segments, model, bert_processor, device, batch_size)
        if device == "cuda":
            torch.cuda.synchronize()
        new_time = time.perf_counter() - t0
        new_peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

        ok = True
        for i, ((seq_em, _), (batch_em, _)) in enumerate(zip(sequential, batched)):
            if seq_em.shape != batch_em.shape:
                print(f"  segment {i}: SHAPE MISMATCH sequential={tuple(seq_em.shape)} batched={tuple(batch_em.shape)}")
                ok = False
                continue
            if not torch.allclose(seq_em, batch_em, atol=args.atol):
                max_diff = (seq_em - batch_em).abs().max().item()
                print(f"  segment {i}: VALUE MISMATCH max_diff={max_diff:.5f} (atol={args.atol})")
                ok = False

        print(
            f"batch_size={batch_size:>3}: correctness={'PASS' if ok else 'FAIL'}  "
            f"{new_time:6.2f}s total, {new_time/args.n_segments*1000:7.1f} ms/seg, "
            f"peak_mem={new_peak:.2f} GB, speedup={old_time/new_time:.2f}x"
        )

    print()


if __name__ == "__main__":
    main()

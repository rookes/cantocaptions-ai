"""Benchmark MelBandRoformer forward-pass cost as a function of batch size.

Answers: does batching chunks through the vocal isolation model actually reduce
per-chunk wall-clock time, or is the model already compute-saturated at batch=1?

Uses randomly initialized weights (no checkpoint download needed) since only
forward-pass timing is being measured, not output correctness.

Usage:
    uv run python scripts/bench_vocal_isolation_batching.py
    uv run python scripts/bench_vocal_isolation_batching.py --batch-sizes 1,2,4,8,16
"""

import argparse
import time

import torch
from omegaconf import OmegaConf
import importlib.resources

from cantocaptions_ai.pipeline.mbroformer.model import MelBandRoformer


def bench(model, C: int, batch_size: int, n_iters: int = 5, n_warmup: int = 2) -> None:
    x = torch.randn(batch_size, 2, C, device="cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(n_warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"batch={batch_size:>3}: {dt*1000:8.1f} ms/call, "
        f"{dt/batch_size*1000:8.1f} ms/chunk, peak_mem={peak_gb:.2f} GB"
    )
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="comma-separated batch sizes to test")
    parser.add_argument("--iters", type=int, default=5, help="timed iterations per batch size")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")

    config_ref = importlib.resources.files("cantocaptions_ai.assets").joinpath(
        "config_vocals_mel_band_roformer.yaml"
    )
    with importlib.resources.as_file(config_ref) as p:
        config = OmegaConf.load(p)

    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    model_kwargs["multi_stft_resolutions_window_sizes"] = tuple(
        model_kwargs["multi_stft_resolutions_window_sizes"]
    )
    model = MelBandRoformer(**model_kwargs).cuda().eval()
    C = config.inference.chunk_size

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"chunk_size C = {C}  params = {n_params:.1f}M  device = {torch.cuda.get_device_name(0)}")
    print()

    for batch_size in (int(b) for b in args.batch_sizes.split(",")):
        bench(model, C, batch_size, n_iters=args.iters)


if __name__ == "__main__":
    main()

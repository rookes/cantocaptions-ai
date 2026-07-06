"""Benchmark whether torch.compile is a net win for the native ASR backend despite the
audio-tower's get_audio_cu_seqlens() graph break (Tensor.item() + a Python loop over
tensor-derived lengths, untraceable by dynamo), and whether disabling that one function
from tracing changes anything.

Compares four configurations against the same sequence of batches (deliberately varying
batch size and audio length, with repeats, to probe whether dynamo actually recompiles the
pre-break graph section every time the audio shape changes, or reuses a cached compile):
  - eager: no compile, baseline.
  - compile: model.forward = torch.compile(model.forward), graph break allowed as-is.
  - compile+capture_scalar_outputs: same, with torch._dynamo.config.capture_scalar_outputs
    set first (what the graph-break warning itself suggests).
  - compile+disable_cu_seqlens: get_audio_cu_seqlens monkeypatched with
    torch.compiler.disable() before compiling, turning the break into an explicit boundary.

Uses random noise as audio (no real speech), so absolute per-segment latency is not
meaningful — only the relative pattern across configs and across repeated shapes within a
config is. Requires the model to be locally cached (local_files_only=True); run once online
first if not, e.g. via a normal `cantocaptions` invocation.

Usage:
    uv run python scripts/bench_asr_compile.py
"""

import argparse
import time

import numpy as np
import torch


def make_batches(rng: np.random.Generator, specs: list) -> list:
    """specs: list of (batch_size, audio_seconds) -> list of audio batches (list[np.ndarray])."""
    return [
        [rng.standard_normal(int(seconds * 16000)).astype(np.float32) for _ in range(batch_size)]
        for batch_size, seconds in specs
    ]


def run_config(name: str, model, processor, batches: list, max_new_tokens: int) -> list:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for wavs in batches:
        inputs = processor.apply_transcription_request(audio=wavs, language="Cantonese")
        inputs = inputs.to(model.device, model.dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n=== {name} ===")
    for i, (t, wavs) in enumerate(zip(times, batches)):
        print(f"  batch {i}: {t * 1000:7.1f} ms  (batch_size={len(wavs)}, samples={wavs[0].shape[0]})")
    print(f"  total: {sum(times):.2f}s  peak_vram: {peak_mb:.0f} MB")
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B-hf", help="HF model id (native backend)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32, help="kept small since this measures the compile/recompile pattern, not real transcription quality")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    rng = np.random.default_rng(args.seed)
    # Varying batch size / audio length, with repeats (batches 0/1 and 2/5) to check whether
    # a previously-seen shape re-triggers a fresh compile or reuses a cached one.
    specs = [
        (4, 3.0), (4, 3.0),
        (8, 6.0),
        (4, 3.0),
        (2, 12.0),
        (8, 6.0),
    ]
    batches = make_batches(rng, specs)

    print(f"Loading {args.model} (float16, cuda)...")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    def fresh_model():
        return AutoModelForMultimodalLM.from_pretrained(
            args.model, dtype="float16", device_map="cuda", local_files_only=True,
        ).eval()

    model = fresh_model()
    run_config("eager", model, processor, batches, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()

    model = fresh_model()
    model.forward = torch.compile(model.forward)
    run_config("compile (graph break allowed, as shipped)", model, processor, batches, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    torch._dynamo.config.capture_scalar_outputs = True
    model = fresh_model()
    model.forward = torch.compile(model.forward)
    run_config("compile + capture_scalar_outputs=True", model, processor, batches, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    torch._dynamo.config.capture_scalar_outputs = False

    import transformers.models.qwen3_asr.modeling_qwen3_asr as qwen3_asr_modeling
    qwen3_asr_modeling.get_audio_cu_seqlens = torch.compiler.disable(qwen3_asr_modeling.get_audio_cu_seqlens)
    model = fresh_model()
    model.forward = torch.compile(model.forward)
    run_config("compile + get_audio_cu_seqlens disabled", model, processor, batches, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # dynamic=True tells dynamo to treat shape dims symbolically up front instead of
    # specializing (and recompiling) per exact shape combination it happens to see.
    model = fresh_model()
    model.forward = torch.compile(model.forward, dynamic=True)
    run_config("compile(dynamic=True) + get_audio_cu_seqlens disabled (still patched)", model, processor, batches, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Restore the original function, then test dynamic=True alone (graph break allowed)
    # to isolate how much dynamic=True helps vs. the disable-patch.
    qwen3_asr_modeling.get_audio_cu_seqlens = qwen3_asr_modeling.get_audio_cu_seqlens.__wrapped__
    model = fresh_model()
    model.forward = torch.compile(model.forward, dynamic=True)
    run_config("compile(dynamic=True), graph break allowed", model, processor, batches, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Cap the number of failed compile attempts dynamo will burn on a bad/varying shape
    # before giving up and running eagerly, to bound the worst-case one-time tax.
    torch._dynamo.config.recompile_limit = 1
    model = fresh_model()
    model.forward = torch.compile(model.forward)
    run_config("compile, recompile_limit=1, graph break allowed", model, processor, batches, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 8


if __name__ == "__main__":
    main()

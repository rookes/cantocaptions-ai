"""Benchmark a static KV cache, with and without a compiled decode step, for the native ASR backend.

bench_asr_compile.py compiled the whole ``model.forward`` and found it a net loss: the audio
tower breaks the graph, and with the default growing cache every decode step is a new shape.
transformers can instead keep the prefill (audio tower included) eager and compile only the
one-token decode step, which has a fixed shape once the KV cache is static
(``cache_implementation="static"``, sized once by ``max_cache_len`` so calls reuse it). With
``mode="reduce-overhead"`` that step runs as a CUDA graph. This measures whether that pays:

    eager     the pipeline as it is (dynamic cache)
    static    static cache, decode step uncompiled (isolates the cache's own cost)
    compiled  static cache, decode step compiled

``run`` transcribes real ASR inputs -- the chunk audio a ``--debug_dir`` run saved (its
vocal_isolation checkpoint if there is one, else vad) -- through the ASR stage itself
(``AsrStage.run``: batches across files, longest first, loop retries, normalization), in a
fresh process per mode so the compile cost is what a pipeline run would pay. It records the
texts, per-batch timings, peak VRAM and how many graphs were compiled:

    uv run python scripts/bench_asr_static_cache.py run --mode eager --batch_size 12 \\
        --inputs temp/bluey_test temp/Doraemon.555-556 --out runs/eager_b12.json

``compare`` scores runs against ground-truth SRTs (CER over all chunk text, punctuation and
whitespace removed, as score_subtitles.py does) and against the first run (share of chunks
with identical text, and the CER between the two):

    uv run python scripts/bench_asr_static_cache.py compare runs/eager_b12.json runs/*.json \\
        --gt bluey_test=experiments/fixtures/bluey/bluey_groundtruth.srt --gt_dir episodes/gt

Peak VRAM matters as much as time: a mode that needs more memory per row forces a smaller
batch size on a smaller card, so compare modes at equal peak VRAM too (run each at several
--batch_size values).

Measured 2026-10-21 (RTX 3080 10 GB, torch 2.11, transformers 5.13; 324 chunks: Bluey,
Doraemon 555-556, Triumph in the Skies E01, five 10-minute build_eval_episodes.py shows;
GT CER over the seven with ground truth). Times are the ASR stage, warm Inductor cache:

    mode      bs   time   peak alloc  peak reserved  GT CER  chunks same as eager b12
    eager      8   397 s   5.6 GB      6.1 GB         8.97%   90.4%
    eager     12   290 s   6.4 GB      8.4 GB         8.91%   --
    eager     16   246 s   7.2 GB      9.9 GB         8.90%   91.4%
    static    12   299 s   7.2 GB      9.4 GB         8.91%   99.4%
    compiled   8   115 s   6.1 GB      6.8 GB         8.96%   89.8%
    compiled  10   103 s   6.7 GB      7.9 GB         8.94%   90.1%
    compiled  12    95 s   7.2 GB      9.2 GB         8.95%   94.4%
    compiled  16   104 s   8.3 GB     10.9 GB         8.92%   91.4%   (reserved > the card: WDDM spill)

  * Decode steps at batch 12: 12.6/s eager, 49/s compiled (3.9x); full batches 253 s -> 76 s.
    At equal or lower VRAM (compiled b8 vs eager b12) it is 2.5x faster.
  * The static cache costs batch x max_cache_len of KV up front (~0.8 GB at b12, ~600
    slots), held during the eager prefill, which sets the peak. Alone it is no faster.
  * Every new batch size compiles its own graph: ~40-60 s cold, ~7 s warm. Loop retries
    (1-2 rows), a short last batch and an OOM halving each are one.
  * Text: compiled b12 changes fewer chunks (5.6%, mostly punctuation; 6 of 324 in their
    characters) than eager does between batch sizes 8, 12 and 16 (~9-10%). GT CER spans
    8.90-8.97% over every run, so the change is numerical noise, not a quality shift.

End to end, the same 8 inputs through the pipeline (batch 12, isolation and diarization on,
VAD/isolation reloaded): Transcription 4:49 off -> 1:52 with --compile auto (14 s first
batch; 66 s the first time), the whole run 9:36 -> 6:29, peak 6.7 -> 7.5 GB. Final SRTs:
same cue counts, 2469 of 2479 cues identical, start-error and coverage unchanged, GT CER
8.86% -> 8.90%; one cue boundary moved where a 。 became ，. Two auto runs: identical.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

MODES = ("eager", "static", "compiled")


def _load_inputs(stem_dirs):
    """[(stem, segments)] from debug-dir stem folders: vocal_isolation's audio, else vad's."""
    from cantocaptions_ai.utils.debug import _load_audio_segments
    out = []
    for path in stem_dirs:
        path = os.path.normpath(path)
        for stage in ("vocal_isolation", "vad"):
            segs = _load_audio_segments(os.path.join(path, stage))
            if segs is not None:
                out.append((os.path.basename(path), stage, segs))
                break
        else:
            raise SystemExit(f"{path}: no vocal_isolation or vad checkpoint")
    return out


def _max_cache_len(stage, inputs) -> int:
    """Prompt length of the longest chunk plus the token budget generate gives it, so one
    static cache serves every call of the run."""
    from cantocaptions_ai.pipeline import _asr_native as native
    from cantocaptions_ai.utils.audio import SAMPLE_RATE
    longest = max((s["audio"] for _, _, segs in inputs for s in segs), key=len)
    prompt = stage.processor.apply_transcription_request(
        audio=[longest], language=native._normalize_language("yue"))
    budget = max(stage.max_new_tokens, int(len(longest) / SAMPLE_RATE * native._TOKENS_PER_SECOND))
    return int(prompt["input_ids"].shape[1]) + budget + 8


def cmd_run(args) -> None:
    import torch
    from torch._dynamo.utils import counters
    import transformers
    from transformers import CompileConfig

    from cantocaptions_ai.pipeline.asr import load_model

    inputs = _load_inputs(args.inputs)
    stage = load_model(None, "cuda", language="yue", batch_size=args.batch_size,
                       local_files_only=True, vram_checks=False,
                       vram_headroom_mb=args.vram_headroom_mb)
    model = stage.model
    gen = model.generation_config
    max_cache_len = None
    if args.mode != "eager":
        max_cache_len = args.max_cache_len or _max_cache_len(stage, inputs)
        gen.cache_implementation = "static"
        gen.max_cache_len = max_cache_len
        if args.mode == "static":
            gen.disable_compile = True
        else:
            gen.compile_config = CompileConfig(
                mode=args.compile_mode,
                dynamic={"none": None, "false": False, "true": True}[args.dynamic])

    # Per-batch wall time and shape, so a batch that paid a compile shows up as one.
    batches = []
    real_generate = model.generate

    def timed_generate(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = real_generate(*a, **kw)
        torch.cuda.synchronize()
        batches.append({"rows": int(kw["input_ids"].shape[0]),
                        "prompt": int(kw["input_ids"].shape[1]),
                        "new_tokens": int(out.shape[1] - kw["input_ids"].shape[1]),
                        "retry": "repetition_penalty" in kw,
                        "seconds": time.perf_counter() - t0})
        return out

    model.generate = timed_generate
    items = [{"audio_path": stem, "name": stem, "vad_segments": segs} for stem, _, segs in inputs]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    weights = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    results = stage.run(items)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    files = {}
    for (stem, source, segs), item in zip(inputs, results):
        files[stem] = {
            "source": source,
            "chunks": len(segs),
            "audio_seconds": sum(len(s["audio"]) for s in segs) / 16000,
            "texts": [s["text"] for s in item["result"]["segments"]],
            "budget_hit": [bool(s.get("budget_hit")) for s in item["result"]["segments"]],
        }
    report = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "max_cache_len": max_cache_len,
        "compile_mode": args.compile_mode if args.mode == "compiled" else None,
        "dynamic": args.dynamic if args.mode == "compiled" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "total_seconds": total,
        "weights_mb": weights / 2**20,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
        "batches": batches,
        "files": files,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    first = batches[0]["seconds"] if batches else 0.0
    print(f"{args.mode} b{args.batch_size}: {total:.1f} s ({first:.1f} s first batch), "
          f"peak allocated {report['peak_allocated_mb']:.0f} MB, "
          f"reserved {report['peak_reserved_mb']:.0f} MB, graphs {report['unique_graphs']}")


def _gt_texts(args):
    from cantocaptions_ai.utils.subtitles import load_subtitle_file
    from score_subtitles import _chars
    paths = {}
    for d in args.gt_dir or []:
        for f in os.listdir(d):
            if f.endswith(".srt"):
                paths[f[:-4]] = os.path.join(d, f)
    for pair in args.gt or []:
        stem, _, path = pair.partition("=")
        paths[stem] = path
    return {stem: _chars(load_subtitle_file(p)) for stem, p in paths.items()}


def cmd_compare(args) -> None:
    from score_subtitles import _STRIP, _cer
    gt = _gt_texts(args)
    runs = []
    for path in args.runs:
        with open(path, encoding="utf-8") as f:
            runs.append((os.path.basename(path), json.load(f)))
    base_name, base = runs[0]

    def doc(texts):
        return _STRIP.sub("", "".join(texts))

    print(f"{'run':28s} {'mode':9s} {'bs':>3s} {'total':>7s} {'1st':>6s} {'rest':>7s} "
          f"{'alloc MB':>8s} {'resv MB':>8s} {'graphs':>6s} {'loops':>5s} {'GT CER':>7s} "
          f"{'same':>6s} {'vs base':>7s}")
    for name, run in runs:
        b = run["batches"]
        first = b[0]["seconds"] if b else 0.0
        edits = chars = 0.0
        same = n = 0
        base_edits = base_chars = 0.0
        for stem, fr in run["files"].items():
            if stem in gt and gt[stem]:
                cer = _cer(gt[stem], doc(fr["texts"]))
                edits += cer * len(gt[stem])
                chars += len(gt[stem])
            bf = base["files"].get(stem)
            if bf is not None:
                same += sum(x == y for x, y in zip(fr["texts"], bf["texts"]))
                n += len(fr["texts"])
                ref = doc(bf["texts"])
                if ref:
                    base_edits += _cer(ref, doc(fr["texts"])) * len(ref)
                    base_chars += len(ref)
        loops = sum(sum(fr["budget_hit"]) for fr in run["files"].values())
        gt_cer = f"{edits / chars * 100:6.2f}%" if chars else "    n/a"
        vs = f"{base_edits / base_chars * 100:6.2f}%" if base_chars else "    n/a"
        print(f"{name[:28]:28s} {run['mode']:9s} {run['batch_size']:3d} "
              f"{run['total_seconds']:6.1f}s {first:5.1f}s {run['total_seconds'] - first:6.1f}s "
              f"{run['peak_allocated_mb']:8.0f} {run['peak_reserved_mb']:8.0f} "
              f"{run['unique_graphs']:6d} {loops:5d} {gt_cer} {same / n * 100 if n else 0:5.1f}% {vs}")

    if args.per_file:
        for name, run in runs:
            print(f"\n== {name}")
            for stem, fr in run["files"].items():
                cer = _cer(gt[stem], doc(fr["texts"])) if gt.get(stem) else float("nan")
                print(f"  {stem[:44]:44s} {fr['chunks']:4d} chunks  GT CER {cer * 100:6.2f}%")

    if args.show_diffs:
        for name, run in runs[1:]:
            print(f"\n== {name} vs {base_name}: differing chunks")
            shown = 0
            for stem, fr in run["files"].items():
                for i, (x, y) in enumerate(zip(base["files"][stem]["texts"], fr["texts"])):
                    if x != y and shown < args.show_diffs:
                        print(f"  {stem[:30]} #{i}\n    base: {x}\n    this: {y}")
                        shown += 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--batch_size", type=int, default=12)
    r.add_argument("--inputs", nargs="+", required=True, help="debug-dir stem folders")
    r.add_argument("--out", required=True)
    r.add_argument("--max_cache_len", type=int, default=0, help="0: sized from the longest chunk")
    r.add_argument("--compile_mode", default="reduce-overhead")
    r.add_argument("--dynamic", choices=("none", "false", "true"), default="false")
    r.add_argument("--vram_headroom_mb", type=int, default=0)
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="run JSONs; the first is the baseline")
    c.add_argument("--gt", action="append", metavar="STEM=SRT")
    c.add_argument("--gt_dir", action="append", metavar="DIR")
    c.add_argument("--per_file", action="store_true")
    c.add_argument("--show_diffs", type=int, default=0, metavar="N")
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

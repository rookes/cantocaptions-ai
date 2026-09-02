"""Score --realign's line placement against a ground-truth subtitle, with no model in the loop.

Placement is the expensive half of --realign and the half that goes wrong, but iterating on it
through the pipeline costs a model load and an encoder pass over the file every time. This
caches the align model's emissions once per fixture and then scores the placement search
against them, so a sweep takes seconds and the numbers are still real emissions against real
ground truth.

    # once per fixture (needs the align model, a few minutes)
    uv run python scripts/bench_realign_placement.py cache --name bluey \
        --audio test/bluey/bluey_test.wav --groundtruth test/bluey/bluey_groundtruth.srt

    # then as often as you like (seconds, no model)
    uv run python scripts/bench_realign_placement.py score --name bluey --worst 10

The ground-truth SRT supplies both halves of the answer: its text is the transcript --realign
is given, and its timings are what placement should recover. Cues are compared by position,
which is exact because placement returns one timing per input line.

**Median is the easy part and stays good even when the search has failed.** A broken run looks
like a large p90 next to a fine median: the shipped forward-only search scored a 0.049 s median
on Doraemon while half the file was more than two minutes out, because the lines it placed
before losing its place were placed well. Watch p90, the share within 0.5 s, and how many lines
come back carrying a reason.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cantocaptions_ai.pipeline.realign import (  # noqa: E402
    EmissionTimeline, TranscriptLine, assign_lines, load_transcript_lines,
)
from cantocaptions_ai.pipeline.retime import load_subtitle_file  # noqa: E402

CACHE = os.path.join(REPO_ROOT, "temp", "realign_placement_cache")
THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


def _paths(name):
    return (os.path.join(CACHE, name + ".emis.npy"),
            os.path.join(CACHE, name + ".meta.npz"),
            os.path.join(CACHE, name + ".vocab.json"))


def cache(args):
    """Run split-only VAD and the align encoder once, and store the result plus the answer key."""
    from cantocaptions_ai.pipeline.alignment import (
        _compute_vad_emissions_batched, load_align_model, load_bert_processor,
    )
    from cantocaptions_ai.pipeline.vad import load_vad

    os.makedirs(CACHE, exist_ok=True)
    vad = load_vad(chunk_size=args.chunk_size, device=args.device, cover_all=True)
    audio = vad._extract({"audio_path": args.audio, "audio_downmix": args.audio_downmix})
    segments = vad.process(audio)
    print(f"{len(segments)} contiguous chunks covering {segments[-1]['end']:.1f}s")

    processor = load_bert_processor()
    model, meta = load_align_model(args.language, args.device, compute_type="float32")
    results = _compute_vad_emissions_batched(
        segments, model, processor, args.device, args.batch_size,
        vram_checks=False, primer=meta["profile"].primer,
    )
    emissions = np.concatenate(
        [e.float().numpy().astype(np.float16) for e, _rate in results], axis=0)

    truth = load_subtitle_file(args.groundtruth)
    emis_path, meta_path, vocab_path = _paths(args.name)
    np.save(emis_path, emissions)
    np.savez(
        meta_path,
        starts=np.array([s["start"] for s in segments]),
        ends=np.array([s["end"] for s in segments]),
        counts=np.array([e.shape[0] for e, _rate in results]),
        texts=np.array([c["text"].replace("\n", " ") for c in truth], dtype=object),
        gt_start=np.array([c["start"] for c in truth]),
        gt_end=np.array([c["end"] for c in truth]),
    )
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(meta["dictionary"], fh, ensure_ascii=False)
    print(f"cached {emissions.shape} ({emissions.nbytes / 1e6:.0f} MB) and "
          f"{len(truth)} ground-truth cues under {CACHE}")


def score(args):
    emis_path, meta_path, vocab_path = _paths(args.name)
    if not os.path.isfile(emis_path):
        raise SystemExit(f"no cache for {args.name!r}; run the 'cache' subcommand first")
    emissions = np.load(emis_path, mmap_mode="r")
    meta = np.load(meta_path, allow_pickle=True)
    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)
    starts, ends, counts = meta["starts"], meta["ends"], meta["counts"]

    segments, offsets, frame = [], [], 0
    for start, end, count in zip(starts, ends, counts):
        segments.append({"start": float(start), "end": float(end),
                         "audio": np.zeros(1, dtype=np.float32)})
        offsets.append((frame, frame + int(count)))
        frame += int(count)
    index = {float(s): k for k, s in enumerate(starts)}

    def compute(segs):
        out = []
        for seg in segs:
            a, b = offsets[index[float(seg["start"])]]
            out.append((torch.from_numpy(np.asarray(emissions[a:b], dtype=np.float32)), 25.0))
        return out

    texts = [str(x) for x in meta["texts"]]
    if args.transcript:      # score a real transcript rather than the truth's own text
        texts = [line.text for line in load_transcript_lines(args.transcript)]
    truth = list(zip(meta["gt_start"].tolist(), meta["gt_end"].tolist()))
    lines = [TranscriptLine(i, text) for i, text in enumerate(texts)]

    timeline = EmissionTimeline(segments, compute)
    started = time.perf_counter()
    timings = assign_lines(lines, segments, timeline, vocab, args.language,
                           window_seconds=args.window, commit_margin=args.commit_margin)
    elapsed = time.perf_counter() - started

    n = min(len(timings), len(truth))
    err = sorted(abs(timings[i].start - truth[i][0]) for i in range(n))
    pct = lambda p: err[min(len(err) - 1, int(p * (len(err) - 1)))]
    reasons = {}
    for timing in timings:
        if timing.reason:
            reasons[timing.reason] = reasons.get(timing.reason, 0) + 1

    print(f"\n{args.name}: {len(lines)} lines, {timeline.file_end:.0f}s, "
          f"{len(segments)} chunks, {timeline.computed} encoded")
    print(f"  start error  median {pct(.5):.3f}s   p90 {pct(.9):.3f}s   max {err[-1]:.3f}s")
    print("  within       " + "   ".join(
        f"<={t}s {100 * sum(1 for e in err if e <= t) / len(err):5.1f}%" for t in THRESHOLDS))
    print(f"  flagged      {sum(reasons.values())} of {len(timings)}  {reasons or 'none'}")
    print(f"  wall         {elapsed:.1f}s")

    if args.worst:
        order = sorted(range(n), key=lambda i: -abs(timings[i].start - truth[i][0]))
        print(f"\n  worst {args.worst}:")
        for i in sorted(order[:args.worst]):
            print(f"    line {i:4d}  truth {truth[i][0]:8.2f}s  got {timings[i].start:8.2f}s  "
                  f"({timings[i].start - truth[i][0]:+7.2f}s)  "
                  f"{timings[i].reason or '-':<14} {texts[i][:34]}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cache", help="encode a fixture once and store its emissions")
    c.add_argument("--name", required=True)
    c.add_argument("--audio", required=True)
    c.add_argument("--groundtruth", required=True)
    c.add_argument("--chunk_size", type=int, default=28)
    c.add_argument("--audio_downmix", default="mix")
    c.add_argument("--batch_size", type=int, default=4)
    c.add_argument("--device", default="cuda")
    c.add_argument("--language", default="yue")
    c.set_defaults(func=cache)

    s = sub.add_parser("score", help="run the placement search against a cached fixture")
    s.add_argument("--name", required=True)
    s.add_argument("--transcript", default=None,
                   help="score a real transcript rather than the ground truth's own text")
    s.add_argument("--window", type=float, default=120.0)
    s.add_argument("--commit_margin", type=float, default=10.0)
    s.add_argument("--language", default="yue")
    s.add_argument("--worst", type=int, default=10)
    s.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

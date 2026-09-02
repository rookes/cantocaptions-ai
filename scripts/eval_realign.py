"""Measure --realign against a subtitle whose timings are already known to be right.

The trick this harness turns is that a ground-truth SRT contains both halves of the answer:
strip its timings and you have exactly the kind of untimed transcript --realign takes, and
keep them and you have the timings it should recover. So the recovered start of every cue can
be compared against the second it actually belongs at, which is the only honest measure of
whether the placement search works. Nothing else in the repo can tell you that -- SubER and
CER score *text*, and --realign never changes the text.

    uv run python scripts/eval_realign.py --audio test/bluey/bluey_test.wav \
        --groundtruth test/bluey/bluey_groundtruth.srt
    uv run python scripts/eval_realign.py ... --anchor both
    uv run python scripts/eval_realign.py ... --worst 20

Two things to know when reading the numbers:

* **Text cleaning is off by default**, so every transcript line survives to the output and the
  k-th cue out is the k-th line in. That keeps the comparison exact and isolates the question
  the harness exists to answer. Pass --clean to score the pipeline as it actually ships; cue
  counts then differ (the noise rules drop interjection-only lines) and cues are matched by
  text similarity instead, which is a looser instrument.
* **Start error is the headline, not end error.** A cue's end is set by the release/trim pass
  from the *next* cue's start (alignment.py), so end error largely restates start error one
  cue later rather than adding information.
"""

import argparse
import bisect
import difflib
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cantocaptions_ai.pipeline.retime import load_subtitle_file  # noqa: E402

THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def write_transcript(cues: Sequence[dict], path: str) -> None:
    """Write the cue texts as a line-delimited transcript -- the timings are the answer key."""
    with open(path, "w", encoding="utf-8") as fh:
        for cue in cues:
            fh.write(cue["text"].strip() + "\n")


def run_realign(
    audio: str, transcript: str, workdir: str, name: str,
    anchor: str, clean: bool, extra_args: Sequence[str],
) -> Tuple[str, float]:
    """Run the pipeline once; return (path to its SRT, wall seconds)."""
    out_dir = os.path.join(workdir, name)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "uv", "run", "cantocaptions", audio,
        "--realign", transcript,
        "--realign_anchor", anchor,
        "--output_dir", out_dir,
        "--output_format", "srt",
        "--debug_dir", os.path.join(out_dir, "debug"),
    ]
    if not clean:
        cmd += ["--no_clean_text"]
    cmd += list(extra_args)

    print(f"--- {name} ---")
    print("  " + " ".join(cmd))
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {proc.returncode}")

    srt = os.path.join(out_dir, f"{_stem(audio)}.srt")
    if not os.path.isfile(srt):
        raise SystemExit(f"{name} produced no SRT at {srt}")
    return srt, elapsed


def match_cues(truth: Sequence[dict], hyp: Sequence[dict]) -> List[Tuple[int, int]]:
    """Pair ground-truth cues with output cues.

    Equal counts mean --realign kept the 1:1 line-to-cue mapping it promises, so position is
    the correspondence and nothing is inferred. Otherwise cleaning dropped or fused lines and
    the pairing falls back to a monotonic text match, which can only ever be approximate.
    """
    if len(truth) == len(hyp):
        return [(i, i) for i in range(len(truth))]
    matcher = difflib.SequenceMatcher(
        a=[c["text"].strip() for c in truth], b=[c["text"].strip() for c in hyp], autojunk=False
    )
    pairs: List[Tuple[int, int]] = []
    for a0, b0, size in matcher.get_matching_blocks():
        pairs.extend((a0 + k, b0 + k) for k in range(size))
    return pairs


def score(truth: Sequence[dict], hyp: Sequence[dict]) -> Dict[str, object]:
    pairs = match_cues(truth, hyp)
    start_err = [abs(hyp[j]["start"] - truth[i]["start"]) for i, j in pairs]
    end_err = [abs(hyp[j]["end"] - truth[i]["end"]) for i, j in pairs]
    signed = [hyp[j]["start"] - truth[i]["start"] for i, j in pairs]

    out: Dict[str, object] = {
        "truth_cues": len(truth),
        "hyp_cues": len(hyp),
        "matched": len(pairs),
        "start_median": _percentile(start_err, 0.5),
        "start_p90": _percentile(start_err, 0.9),
        "start_max": max(start_err) if start_err else float("nan"),
        "end_median": _percentile(end_err, 0.5),
        # A consistent sign here is a systematic lag, which is a different (and easier)
        # problem than scattered error: it means every cue is off the same way.
        "bias": sum(signed) / len(signed) if signed else float("nan"),
        "pairs": pairs,
        "start_err": start_err,
    }
    for t in THRESHOLDS:
        out[f"within_{t}"] = (
            100.0 * sum(1 for e in start_err if e <= t) / len(start_err) if start_err else 0.0
        )
    return out


def report(name: str, res: Dict[str, object], elapsed: float) -> None:
    print(f"\n=== {name} ===")
    print(f"  cues            {res['hyp_cues']} out / {res['truth_cues']} in "
          f"({res['matched']} matched)")
    print(f"  start error     median {res['start_median']:.3f}s   "
          f"p90 {res['start_p90']:.3f}s   max {res['start_max']:.3f}s")
    print(f"  end error       median {res['end_median']:.3f}s")
    print(f"  mean signed     {res['bias']:+.3f}s  (systematic lag if far from 0)")
    within = "   ".join(f"<={t}s {res[f'within_{t}']:5.1f}%" for t in THRESHOLDS)
    print(f"  within          {within}")
    print(f"  wall            {elapsed:.1f}s")


def show_worst(truth, hyp, res, limit: int) -> None:
    pairs, errs = res["pairs"], res["start_err"]
    order = sorted(range(len(pairs)), key=lambda k: errs[k], reverse=True)[:limit]
    if not order:
        return
    print(f"\n  worst {len(order)} placements:")
    for k in sorted(order, key=lambda k: pairs[k][0]):
        i, j = pairs[k]
        drift = hyp[j]["start"] - truth[i]["start"]
        print(f"    truth {truth[i]['start']:8.2f}s  ->  got {hyp[j]['start']:8.2f}s "
              f"({drift:+7.3f}s)  {truth[i]['text'][:40]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", required=True, help="audio or video file to align against")
    parser.add_argument("--groundtruth", required=True,
                        help="SRT holding the correct timings; its text becomes the transcript")
    parser.add_argument("--workdir", default=os.path.join(REPO_ROOT, "temp", "realign_eval"))
    parser.add_argument("--anchor", default="acoustic",
                        choices=["acoustic", "asr", "both"],
                        help="which placement strategy to measure ('both' runs and compares)")
    parser.add_argument("--clean", action="store_true",
                        help="score with text cleaning on, as the pipeline actually ships; "
                             "cue counts then differ and matching becomes approximate")
    parser.add_argument("--worst", type=int, default=10,
                        help="list this many of the worst-placed cues (0 to skip)")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                        help="everything after this is passed through to the pipeline")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    truth = load_subtitle_file(args.groundtruth)
    transcript = os.path.join(args.workdir, f"{_stem(args.groundtruth)}.transcript.txt")
    write_transcript(truth, transcript)
    print(f"{len(truth)} ground-truth cues -> {transcript}")

    anchors = ["acoustic", "asr"] if args.anchor == "both" else [args.anchor]
    results = {}
    for anchor in anchors:
        srt, elapsed = run_realign(
            args.audio, transcript, args.workdir, anchor, anchor, args.clean, args.extra_args,
        )
        res = score(truth, load_subtitle_file(srt))
        results[anchor] = res
        report(anchor, res, elapsed)
        if args.worst:
            show_worst(truth, load_subtitle_file(srt), res, args.worst)

    if len(results) > 1:
        print("\n=== comparison ===")
        print(f"  {'anchor':10s} {'median':>9s} {'p90':>9s} {'<=0.5s':>9s}")
        for anchor, res in results.items():
            print(f"  {anchor:10s} {res['start_median']:8.3f}s {res['start_p90']:8.3f}s "
                  f"{res['within_0.5']:8.1f}%")


if __name__ == "__main__":
    main()

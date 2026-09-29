"""Score pipeline SRTs against ground truth, and diff two runs of the pipeline.

    # accuracy of one or more runs against a reference directory
    uv run python scripts/score_subtitles.py score --gt episodes/gt runs/baseline runs/new

    # what changed between two runs (no reference needed)
    uv run python scripts/score_subtitles.py diff runs/baseline runs/new

Files are paired by relative path (``<dir>/<name>.srt``), so mirrored subfolders work.

``score`` reports, per file and in total:
  * CER -- character error rate over the whole transcript, punctuation and whitespace removed.
  * coverage -- share of reference cues overlapped by any output cue.
  * start error -- for each reference cue, the output cue overlapping it most; median |start
    difference| and the share within 0.2 s / 0.5 s. Cue boundaries need not agree between
    the two, so this is a comparative number (run A vs run B), not an absolute one.
"""
import argparse
import os
import re
import statistics
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cantocaptions_ai.utils.subtitles import load_subtitle_file  # noqa: E402

# Punctuation and whitespace: CER measures the words, not how they were punctuated.
_STRIP = re.compile(r"[\s，。？！、；：…,.!?;:'\"「」『』（）()～~\-—]+")


def _srts(root: str) -> Dict[str, str]:
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".srt"):
                path = os.path.join(dirpath, f)
                out[os.path.relpath(path, root)[:-4]] = path
    return out


def _chars(cues: List[dict]) -> str:
    return _STRIP.sub("", "".join(c["text"] for c in cues))


def _cer(ref: str, hyp: str) -> float:
    import jiwer
    return jiwer.cer(ref, hyp) if ref else float("nan")


def _best_overlap(cue, candidates):
    best, best_ov = None, 0.0
    for c in candidates:
        ov = min(cue["end"], c["end"]) - max(cue["start"], c["start"])
        if ov > best_ov:
            best, best_ov = c, ov
    return best


def score_file(ref: List[dict], hyp: List[dict]) -> dict:
    ref_text, hyp_text = _chars(ref), _chars(hyp)
    cer = _cer(ref_text, hyp_text)
    errors = []
    for cue in ref:
        match = _best_overlap(cue, hyp)
        if match is not None:
            errors.append(abs(match["start"] - cue["start"]))
    return {
        "ref_chars": len(ref_text),
        "cer": cer,
        "edits": cer * len(ref_text),
        "ref_cues": len(ref),
        "hyp_cues": len(hyp),
        "covered": len(errors),
        "start_errors": errors,
    }


def _fmt(row: dict) -> str:
    errs = row["start_errors"]
    med = statistics.median(errs) if errs else float("nan")
    w02 = sum(e <= 0.2 for e in errs) / len(errs) if errs else float("nan")
    w05 = sum(e <= 0.5 for e in errs) / len(errs) if errs else float("nan")
    cov = row["covered"] / row["ref_cues"] if row["ref_cues"] else float("nan")
    return (f"CER {row['cer'] * 100:5.2f}%  cues {row['hyp_cues']:4d}/{row['ref_cues']:<4d} "
            f"coverage {cov * 100:5.1f}%  start |err| median {med:5.2f}s  "
            f"<=0.2s {w02 * 100:5.1f}%  <=0.5s {w05 * 100:5.1f}%")


def cmd_score(args) -> None:
    gt = _srts(args.gt)
    for run in args.runs:
        hyps = _srts(run)
        print(f"\n== {run}")
        total = {"ref_chars": 0, "edits": 0.0, "ref_cues": 0, "hyp_cues": 0, "covered": 0,
                 "start_errors": []}
        for name, ref_path in sorted(gt.items()):
            if name not in hyps:
                print(f"  {name:40s} MISSING")
                continue
            row = score_file(load_subtitle_file(ref_path), load_subtitle_file(hyps[name]))
            print(f"  {name:40s} {_fmt(row)}")
            for key in ("ref_chars", "edits", "ref_cues", "hyp_cues", "covered"):
                total[key] += row[key]
            total["start_errors"] += row["start_errors"]
        total["cer"] = total["edits"] / total["ref_chars"] if total["ref_chars"] else float("nan")
        print(f"  {'TOTAL':40s} {_fmt(total)}")


def cmd_diff(args) -> None:
    a, b = _srts(args.a), _srts(args.b)
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print(f"{name}: only in {'A' if name in a else 'B'}")
            continue
        ca, cb = load_subtitle_file(a[name]), load_subtitle_file(b[name])
        same = sum(1 for x, y in zip(ca, cb) if x == y) if len(ca) == len(cb) else None
        text_a, text_b = _chars(ca), _chars(cb)
        starts = [abs(x["start"] - y["start"]) for x, y in zip(ca, cb)] if len(ca) == len(cb) else []
        line = f"{name}: cues {len(ca)} -> {len(cb)}"
        if same is not None:
            line += f", identical {same}/{len(ca)}"
            if starts:
                line += f", start shift max {max(starts):.3f}s"
        line += f", text CER A->B {_cer(text_a, text_b) * 100:.2f}%"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--gt", required=True)
    s.add_argument("runs", nargs="+")
    s.set_defaults(func=cmd_score)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(func=cmd_diff)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

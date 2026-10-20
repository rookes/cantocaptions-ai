"""Compare extracted reference SRTs with a baseline extraction of the same tracks.

    uv run python scripts/eval_references.py HYP.srt BASE.srt [HYP2.srt BASE2.srt ...] [--diff N]

For each pair: cue counts, the share of hypothesis cue starts within 0.1 s of a baseline
start, docCER of the hypothesis against the baseline (`suber.calculate_document_cer`, the
whole text as one document, so cue splits do not count), and with --diff the first N cues
whose text differs from the baseline cue at the same time, for judging by eye which side
read the picture right. The baseline is not ground truth: for OCR it is another OCR.
"""
from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cantocaptions_ai.utils.subtitles import read_subtitle_cues  # noqa: E402


def compare(hyp: str, base: str, diff: int) -> dict:
    from cantocaptions_ai.utils.suber import calculate_document_cer
    h, b = read_subtitle_cues(hyp), read_subtitle_cues(base)
    starts = sorted(c.start for c in b)

    def near(t: float) -> float:
        i = bisect.bisect_left(starts, t)
        return min(abs(starts[j] - t) for j in (i - 1, i) if 0 <= j < len(starts))

    within = sum(near(c.start) <= 0.1 for c in h) / len(h) if h and b else 0.0
    by_start = {round(c.start, 1): c for c in b}
    diffs = []
    for c in h:
        other = by_start.get(round(c.start, 1))
        if other is not None and other.text.replace("\n", "") != c.text.replace("\n", ""):
            diffs.append((c.start, c.text.replace("\n", "⏎"), other.text.replace("\n", "⏎")))
    return {"hyp": hyp, "cues": len(h), "base_cues": len(b), "starts_within_0.1s": round(within, 3),
            "docCER": round(calculate_document_cer(hyp, base), 4), "text_diffs": len(diffs),
            "diffs": diffs[:diff]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="HYP BASE [HYP BASE ...]")
    ap.add_argument("--diff", type=int, default=0, help="list this many differing cues per pair")
    a = ap.parse_args()
    if len(a.pairs) % 2:
        ap.error("give the files in HYP BASE pairs")
    for hyp, base in zip(a.pairs[::2], a.pairs[1::2]):
        r = compare(hyp, base, a.diff)
        print(f"{Path(hyp).name}: {r['cues']} cues (baseline {r['base_cues']}), starts within 0.1 s "
              f"{r['starts_within_0.1s']:.1%}, docCER {r['docCER']:.2f}%, {r['text_diffs']} cues differ")
        for t, x, y in r["diffs"]:
            print(f"  {t:9.3f}  hyp  {x}\n             base {y}")


if __name__ == "__main__":
    main()

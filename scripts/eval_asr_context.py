"""A/B the experimental reference-subtitle ASR context feature against a ground truth.

What is being measured and why
------------------------------
`--asr_context` feeds a same-content, different-language subtitle into Qwen3-ASR's
context-biasing system prompt, and `--asr_context_vad_expand` additionally unions the
reference cue timings into the VAD timeline. Both are unproven: Qwen publishes no
guidance on context format or length, and the technical report gives no numbers for
context biasing at all. This script exists so the question is settled by measurement
rather than by plausibility.

Variants
--------
  baseline        no reference subtitle at all
  context         context biasing only (--asr_context_vad_expand False)
  context_expand  context biasing + VAD expansion

Caveats on the measurement's validity
-------------------------------------
* Each variant gets its **own --debug_dir**. The transcription checkpoint does not
  record whether context was used, so a shared debug dir would silently replay the
  first variant's ASR output for every later one and report a null result.
* `--reuse` seeds each non-expanding variant's VAD and vocal-isolation checkpoints from
  the baseline, so only ASR and downstream stages re-run. `context_expand` changes the
  VAD timeline itself and is therefore never seeded.
* The headline metric is **docCER** -- character error rate over the whole file with
  cue boundaries ignored. utils/suber.py's `calculate_cer` scores cue *pairs*, so one
  inserted cue misaligns every later pair: on the Bluey fixture it reports ~77% where
  the true character error rate is ~17%. That instrument cannot see a few-percent
  biasing effect, so it is reported only as a secondary column. SubER *is* meant to be
  segmentation-sensitive and is the one to watch when VAD expansion moves boundaries.
* Wall time is the whole subprocess, not ASR alone, and is only comparable between
  variants seeded the same way.

Usage:
    uv run python scripts/eval_asr_context.py \
        --audio test/bluey/bluey_test.wav \
        --reference test/bluey/bluey_standardchinese.srt \
        --groundtruth test/bluey/bluey_groundtruth.srt

    uv run python scripts/eval_asr_context.py ... --diff
    uv run python scripts/eval_asr_context.py ... --sweep template
    uv run python scripts/eval_asr_context.py ... --sweep neighbours --extra_args --no_align
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Stages whose checkpoints are independent of the ASR context, and so may be shared
# between variants that do not change the VAD timeline.
SEEDABLE_STAGES = ("vad", "vocal_isolation")

VARIANT_FLAGS: Dict[str, List[str]] = {
    "baseline": [],
    "context": ["--asr_context", "--asr_context_vad_expand", "False"],
    "context_expand": ["--asr_context", "--asr_context_vad_expand", "True"],
}
# Variants that alter the VAD timeline cannot inherit a cached VAD/isolation pass.
EXPANDS_VAD = {"context_expand"}


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].strip()


def _seed_cache(src_debug: str, dst_debug: str, stem: str) -> bool:
    """Copy reusable stage checkpoints from one debug tree to another."""
    copied = False
    for stage in SEEDABLE_STAGES:
        src = os.path.join(src_debug, stem, stage)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(dst_debug, stem, stage)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
        copied = True
    return copied


def run_variant(
    name: str,
    flags: List[str],
    audio: str,
    reference: str,
    workdir: str,
    seed_from: Optional[str],
    extra_args: List[str],
) -> Tuple[str, float]:
    """Run one pipeline configuration; return (path to its SRT, wall seconds)."""
    out_dir = os.path.join(workdir, name)
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "uv", "run", "cantocaptions", audio,
        "--output_dir", out_dir,
        "--output_format", "srt",
        "--debug_dir", debug_dir,
    ]
    if flags:
        cmd += ["--reference_subtitle", reference] + flags

    seeded = False
    if seed_from and name not in EXPANDS_VAD:
        seeded = _seed_cache(seed_from, debug_dir, _stem(audio))
        if seeded:
            cmd += ["--load_debug_dir", debug_dir]
    cmd += extra_args

    print(f"--- {name} ---")
    print("  " + " ".join(cmd))
    if seed_from and name in EXPANDS_VAD:
        print("  (not seeded: this variant changes the VAD timeline)")
    elif seeded:
        print(f"  (seeded {', '.join(SEEDABLE_STAGES)} from baseline)")

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {proc.returncode}")

    srt = os.path.join(out_dir, f"{_stem(audio)}.srt")
    if not os.path.isfile(srt):
        raise SystemExit(f"{name} produced no SRT at {srt}")
    return srt, elapsed


def score(hyp: str, ref: str) -> Dict[str, float]:
    from cantocaptions_ai.utils.suber import (
        calculate_cer,
        calculate_document_cer,
        calculate_suber,
    )

    return {
        "docCER": calculate_document_cer(hyp, ref),  # primary: text quality alone
        "SubER": calculate_suber(hyp, ref),          # subtitle quality, segmentation-aware
        "segCER": calculate_cer(hyp, ref),           # secondary; inflated by cue drift
    }


def load_cues(path: str) -> List[dict]:
    from cantocaptions_ai.pipeline.retime import load_subtitle_file

    return load_subtitle_file(path)


def cue_texts(path: str) -> List[str]:
    return [seg["text"] for seg in load_cues(path)]


def print_table(results: Dict[str, dict], baseline: str) -> None:
    print()
    print("=== results ===  (docCER is the headline; all figures are percentages)")
    header = (
        f"{'variant':<18}{'docCER':>9}{'SubER':>9}{'segCER':>9}"
        f"{'cues':>7}{'wall':>9}{'delta':>9}"
    )
    print(header)
    print("-" * len(header))
    base_cer = results.get(baseline, {}).get("docCER")
    for name, row in results.items():
        delta = ""
        if base_cer is not None and name != baseline:
            delta = f"{row['docCER'] - base_cer:+.2f}"
        print(
            f"{name:<18}{row['docCER']:>9.2f}{row['SubER']:>9.2f}{row['segCER']:>9.2f}"
            f"{row['cues']:>7}{row['wall']:>8.1f}s{delta:>9}"
        )
    print()
    if base_cer is None or len(results) < 2:
        return
    best_name, best_row = min(results.items(), key=lambda kv: kv[1]["docCER"])
    gain = base_cer - best_row["docCER"]
    if best_name == baseline:
        print("No variant beat the baseline on docCER.")
    elif gain < 1.0:
        # One percentage point of docCER is roughly 10 characters on the Bluey fixture:
        # below that, run-to-run variation is a likelier explanation than the feature.
        print(
            f"Best variant {best_name!r} improves docCER by only {gain:.2f} points "
            "(<1.0) - treat as noise, not a win."
        )
    else:
        print(f"Best variant: {best_name} (docCER {base_cer:.2f} -> {best_row['docCER']:.2f})")


def print_diff(paths: Dict[str, str], groundtruth: str, baseline: str, limit: int) -> None:
    """Show, per ground-truth cue, what each variant said there.

    Rows are aligned by **time overlap against the ground truth**, not by cue index:
    variants legitimately produce different cue counts (VAD expansion adds cues, and
    cue assembly merges differently), so an index-to-index comparison lines up
    unrelated lines and reads as noise even when the variants agree.

    CER alone cannot distinguish "picked up the proper nouns" from "churned the
    particles", and that distinction is what decides whether the feature is worth
    keeping -- watch particularly for Mandarin drift (的 for 嘅, 不 for 唔), which is
    the specific risk of biasing on a standard-Chinese reference.
    """
    from cantocaptions_ai.pipeline.reference_context import overlapping_reference_indices

    truth = load_cues(groundtruth)
    variants = {name: load_cues(path) for name, path in paths.items()}

    # For each ground-truth cue, the text every variant produced over that timespan.
    rows: Dict[str, List[str]] = {}
    for name, cues in variants.items():
        matched = overlapping_reference_indices(truth, cues, fallback_window=0.0)
        rows[name] = [" ".join(cues[i]["text"] for i in idxs) for idxs in matched]

    width = max(len(n) for n in rows)
    print()
    print("=== per ground-truth cue, where variants disagree (time-aligned) ===")
    shown = 0
    for i, cue in enumerate(truth):
        texts = {name: rows[name][i] for name in rows}
        if len(set(texts.values())) == 1:
            continue
        if shown >= limit:
            print(f"... ({limit} shown; raise --diff_limit for more)")
            break
        print(f"[{i}] {cue['start']:.2f}-{cue['end']:.2f}")
        print(f"  {'truth':<{width}} : {cue['text']}")
        for name in ([baseline] if baseline in texts else []) + [n for n in texts if n != baseline]:
            marker = " *" if texts[name] == cue["text"] else "  "
            print(f"  {name:<{width}} :{marker}{texts[name] or '<silence>'}")
        shown += 1
    if shown == 0:
        print("(no differences)")
    else:
        print()
        print("  * = exact match with the ground truth for that cue")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, help="audio file to transcribe")
    parser.add_argument("--reference", required=True, help="reference subtitle in another language (SRT/VTT)")
    parser.add_argument("--groundtruth", required=True, help="hand-corrected Cantonese SRT to score against")
    parser.add_argument("--workdir", default="temp/eval_asr_context", help="where variant outputs and debug trees go")
    parser.add_argument("--variants", default="baseline,context,context_expand", help="comma-separated subset of: " + ", ".join(VARIANT_FLAGS))
    parser.add_argument("--sweep", choices=["template", "neighbours"], help="instead of the variant set, sweep one context knob")
    parser.add_argument("--sweep_values", default=None, help="comma-separated values for --sweep (defaults per knob)")
    parser.add_argument("--sweep_vad_expand", action="store_true", help="run the sweep with VAD expansion ON, isolating the prompt against the 'none' control rather than against the unexpanded baseline")
    parser.add_argument("--reuse", action="store_true", default=True, help="seed non-expanding variants' VAD/isolation checkpoints from the baseline")
    parser.add_argument("--no_reuse", dest="reuse", action="store_false", help="run every variant from scratch")
    parser.add_argument("--diff", action="store_true", help="print per-cue differences against the baseline")
    parser.add_argument("--diff_limit", type=int, default=30, help="max differing cues to print")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[], help="everything after this is passed through to the pipeline")
    args = parser.parse_args()

    for path in (args.audio, args.reference, args.groundtruth):
        if not os.path.isfile(path):
            raise SystemExit(f"not found: {path}")

    expand = "True" if args.sweep_vad_expand else "False"
    if args.sweep == "template":
        default = "none,bare,labelled" if args.sweep_vad_expand else "bare,labelled,instruct"
        values = (args.sweep_values or default).split(",")
        plan = {"baseline": []}
        plan.update({
            f"tpl_{v}": ["--asr_context", "--asr_context_vad_expand", expand,
                         "--asr_context_template", v]
            for v in values
        })
    elif args.sweep == "neighbours":
        values = (args.sweep_values or "0,1,2").split(",")
        plan = {"baseline": []}
        plan.update({
            f"nb_{v}": ["--asr_context", "--asr_context_vad_expand", expand,
                        "--asr_context_neighbours", v]
            for v in values
        })
    else:
        names = [n.strip() for n in args.variants.split(",") if n.strip()]
        unknown = [n for n in names if n not in VARIANT_FLAGS]
        if unknown:
            raise SystemExit(f"unknown variant(s): {', '.join(unknown)}")
        plan = {n: VARIANT_FLAGS[n] for n in names}

    os.makedirs(args.workdir, exist_ok=True)
    baseline = "baseline" if "baseline" in plan else next(iter(plan))

    print("=== running ===")
    paths: Dict[str, str] = {}
    results: Dict[str, dict] = {}
    baseline_debug: Optional[str] = None
    # First expanded variant's debug tree: the timeline every later swept variant shares.
    sweep_debug: Optional[str] = None

    for name, flags in plan.items():
        seed_from = baseline_debug if args.reuse else None
        if args.sweep and args.sweep_vad_expand and name != baseline:
            # Every swept variant expands VAD, so none may inherit the baseline's
            # unexpanded timeline -- but they all share one timeline with each other.
            seed_from = sweep_debug
        srt, wall = run_variant(
            name, flags, args.audio, args.reference, args.workdir, seed_from, args.extra_args
        )
        paths[name] = srt
        if name == baseline:
            baseline_debug = os.path.join(args.workdir, name, "debug")
        elif sweep_debug is None:
            sweep_debug = os.path.join(args.workdir, name, "debug")
        row = score(srt, args.groundtruth)
        row["cues"] = len(cue_texts(srt))
        row["wall"] = wall
        results[name] = row

    print_table(results, baseline)
    if args.diff:
        print_diff(paths, args.groundtruth, baseline, args.diff_limit)


if __name__ == "__main__":
    main()

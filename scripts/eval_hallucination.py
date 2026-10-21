"""Measure ASR loop and foreign-language detection, and try repairs on chosen chunks.

What is being measured and why
------------------------------
ASR repair (``pipeline/asr_repair.py``) acts on chunks its detectors flag, so a detector
that fires on real speech -- 拜拜，拜拜，拜拜, a sung chorus -- would damage good
subtitles. Every threshold is chosen here, against two kinds of evidence:

* ``scan`` runs the text-only detectors over every transcription checkpoint under a debug
  dir and reports how often each setting fires, per hour of audio, with examples. Most
  checkpoints are real speech, so a flag rate is mostly a false-positive rate; ``--export``
  writes the flagged chunks to a TSV for labelling.
* ``score`` takes hand labels (``experiments/hallucination/labels.toml``) and reports
  precision and recall per detector setting.

``rerun`` loads chosen chunks' audio from a checkpoint (following the stage manifest, never
a glob) and runs them through models on the GPU: ``--probe_lid`` prints what the run's
model and base Qwen3-ASR each write and which language each names, with and without the
language hint; ``--ladder`` runs the repair ladder itself and prints before and after.

Usage:
    uv run python scripts/eval_hallucination.py scan --debug_dir temp
    uv run python scripts/eval_hallucination.py scan --debug_dir temp --export flagged.tsv
    uv run python scripts/eval_hallucination.py score --debug_dir temp \
        --labels experiments/hallucination/labels.toml
    uv run python scripts/eval_hallucination.py rerun --debug_dir temp \
        --name Triumph.in.the.Skies.2003.DVD.E01 --chunks 73,82,85-96 --probe_lid
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cantocaptions_ai.pipeline.asr_repair import (  # noqa: E402
    LoopRule,
    RepairSettings,
    chunk_reasons,
    compression_ratio,
    find_loop,
    foreign_text,
    loop_verdict,
)

NATIVE = ("Cantonese", "Chinese", "English")


def _ranges(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _results(debug_dir: str, stage: str = "transcription") -> Iterable[Tuple[str, dict, dict]]:
    """(episode name, result, meta settings) for every checkpoint of *stage*."""
    for path in sorted(glob.glob(os.path.join(debug_dir, "*", stage, "result.json"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        meta_path = os.path.join(os.path.dirname(path), "meta.json")
        settings = {}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                settings = json.load(f).get("settings", {})
        yield name, result, settings


def _model(settings: dict) -> str:
    return settings.get("transcription.model") or "unrecorded"


# --- scan -----------------------------------------------------------------------------------

def _grid() -> List[Tuple[str, LoopRule]]:
    rules = []
    for copies in (3, 4, 5, 6, 8):
        for chars in (12, 16, 24, 32, 40):
            rules.append((f"copies>={copies},chars>={chars}",
                          LoopRule(min_copies=copies, min_chars=chars, truncated_copies=99)))
    for trunc in (2, 3, 4):
        rules.append((f"truncated>={trunc}",
                      LoopRule(min_copies=999, min_chars=999, truncated_copies=trunc)))
    for ratio in (2.0, 2.2, 2.4, 2.6, 3.0):
        rules.append((f"gzip>{ratio}",
                      LoopRule(min_copies=999, min_chars=999, truncated_copies=999,
                               max_compression=ratio)))
    return rules


def cmd_scan(args) -> None:
    rules = _grid()
    hours: Dict[str, float] = defaultdict(float)
    chunks: Dict[str, int] = defaultdict(int)
    fired: Dict[Tuple[str, str], List[Tuple[str, int, str]]] = defaultdict(list)
    signals: Dict[Tuple[str, str], List[Tuple[str, int, str]]] = defaultdict(list)
    export_rows = []

    for name, result, settings in _results(args.debug_dir, args.stage):
        model = _model(settings)
        for i, seg in enumerate(result["segments"]):
            text = seg.get("text") or ""
            hours[model] += (seg["end"] - seg["start"]) / 3600
            chunks[model] += 1
            for key, rule in rules:
                why, _ = loop_verdict({"text": text}, rule)  # text alone, comparable across runs
                if why:
                    fired[(model, key)].append((name, i, text))
            for key, hit in (
                ("broken_char", "�" in text),
                ("foreign_text", bool(foreign_text(text, NATIVE))),
                ("empty>=2s", not text.strip() and seg["end"] - seg["start"] >= 2.0),
                ("budget_hit", bool(seg.get("budget_hit"))),
                ("retry_budget_hit", bool(seg.get("retry_budget_hit"))),
                ("raw_text(fold fired)", bool(seg.get("raw_text"))),
                ("asr_language!=hint", bool(seg.get("asr_language"))),
            ):
                if hit:
                    signals[(model, key)].append((name, i, text))
            reasons = chunk_reasons(seg, RepairSettings(native=NATIVE))
            if reasons:
                loop = find_loop(text)
                export_rows.append((name, i, seg["start"], seg["end"], ";".join(reasons),
                                    loop.copies if loop else 0, f"{compression_ratio(text):.2f}",
                                    seg.get("asr_language", ""), text.replace("\t", " ")))

    for model in sorted(hours):
        print(f"\n=== model: {model}  ({chunks[model]} chunks, {hours[model]:.1f} h of speech) ===")
        print(f"{'detector':32} {'chunks':>7} {'per h':>7}  examples")
        keys = [k for k, _ in rules] + sorted({k for (m, k) in signals if m == model})
        for key in keys:
            rows = fired.get((model, key)) or signals.get((model, key)) or []
            if not rows:
                continue
            examples = " | ".join(t[:28] for _, _, t in rows[:args.examples])
            print(f"{key:32} {len(rows):>7} {len(rows) / max(hours[model], 1e-9):>7.1f}  {examples}")

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            f.write("episode\tchunk\tstart\tend\treasons\tcopies\tgzip\tasr_language\ttext\n")
            for row in export_rows:
                f.write("\t".join(str(c) for c in row) + "\n")
        print(f"\nwrote {len(export_rows)} candidate rows to {args.export}")


# --- score ----------------------------------------------------------------------------------

def _load_labels(path: str) -> Dict[str, Dict[int, str]]:
    """{episode: {chunk index: label}}; label is a reason ("loop", "foreign", ...) or "ok"."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(path, "rb") as f:
        data = tomllib.load(f)
    out: Dict[str, Dict[int, str]] = {}
    for episode, table in data.items():
        labels: Dict[int, str] = {}
        for label, spec in table.items():
            for i in _ranges(spec if isinstance(spec, str) else ",".join(map(str, spec))):
                labels[i] = label
        out[episode] = labels
    return out


def cmd_score(args) -> None:
    labels = _load_labels(args.labels)
    rules = _grid() + [("default", LoopRule())]
    results = {name: result for name, result, _ in _results(args.debug_dir, args.stage)}
    print(f"{'detector':32} {'TP':>4} {'FP':>4} {'FN':>4}  precision  recall   (loop labels)")
    for key, rule in rules:
        tp = fp = fn = 0
        for episode, table in labels.items():
            result = results.get(episode)
            if result is None:
                continue
            for i, label in table.items():
                if i >= len(result["segments"]):
                    continue
                why, _ = loop_verdict(result["segments"][i], rule)
                is_bad = label.startswith("loop")
                if why and is_bad:
                    tp += 1
                elif why:
                    fp += 1
                elif is_bad:
                    fn += 1
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{key:32} {tp:>4} {fp:>4} {fn:>4}  {prec:9.2f}  {rec:6.2f}")

    print("\nchunk_reasons (default settings) against every label:")
    for episode, table in labels.items():
        result = results.get(episode)
        if result is None:
            print(f"  {episode}: no checkpoint")
            continue
        settings = RepairSettings(native=NATIVE)
        jobs = {i: r for i, seg in enumerate(result["segments"])
                if (r := chunk_reasons(seg, settings))}
        for i in sorted(table):
            mark = "flag" if i in jobs else "    "
            expect = table[i]
            ok = (i in jobs) == (expect != "ok")
            print(f"  {'  ' if ok else '!!'} {episode[:30]:30} {i:>4} {expect:>10} {mark} "
                  f"{','.join(jobs.get(i, []))}")


# --- rerun ----------------------------------------------------------------------------------

def _chunk_audio(debug_dir: str, name: str) -> Tuple[str, List[dict]]:
    """The audio the transcription stage ran on: the vocal isolation checkpoint's if there is
    one (the stage reads it, not VAD's), else VAD's. Follows the manifest."""
    from cantocaptions_ai.utils.debug import _load_audio_segments

    for stage in ("vocal_isolation", "vad"):
        segs = _load_audio_segments(os.path.join(debug_dir, name, stage))
        if segs:
            return stage, segs
    raise SystemExit(f"no vad/vocal_isolation checkpoint for {name!r} under {debug_dir}")


def _load(model_name: str, args):
    from cantocaptions_ai.pipeline._asr_native import load_model_native
    return load_model_native(model_name, device=args.device, language="yue",
                             batch_size=args.batch_size, local_files_only=True,
                             loop_retry_penalty=args.retry_penalty)


def _first_pass(debug_dir: str, name: str) -> Optional[List[dict]]:
    path = os.path.join(debug_dir, name, "transcription", "result.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["segments"]


def _chosen(spec: str, first: Optional[List[dict]], settings: RepairSettings) -> List[int]:
    """--chunks: explicit indices, or "auto" for every chunk repair would plan."""
    if spec != "auto":
        return _ranges(spec)
    if first is None:
        return []
    return [i for i, seg in enumerate(first) if chunk_reasons(seg, settings)]


def cmd_rerun(args) -> None:
    import torch

    settings = RepairSettings(
        native=tuple(args.native.split(",")), foreign=args.foreign,
        split_depth=args.split_depth, split_min_piece=args.split_min_piece)
    episodes = []
    for name in args.name.split(","):
        stage, segs = _chunk_audio(args.debug_dir, name)
        first = _first_pass(args.debug_dir, name)
        chunks = _chosen(args.chunks, first, settings)
        print(f"{name}: {len(chunks)} chunk(s) from {stage}/")
        episodes.append((name, segs, first, chunks))

    if args.probe_lid:
        for name, segs, first, chunks in episodes:
            rows: Dict[int, Dict[str, str]] = defaultdict(dict)
            for model_name in args.models.split(","):
                model = _load(model_name, args)
                wavs = [segs[i]["audio"] for i in chunks]
                for hint in ("Cantonese", None):
                    t0 = time.time()
                    texts = model.transcribe(wavs, hint)
                    dt = time.time() - t0
                    for i, text in zip(chunks, texts):
                        flag = "LOOP " if text.retry_budget_hit else "loop " if text.budget_hit else ""
                        rows[i][f"{model_name} / {hint or 'auto'}"] = (
                            f"[{text.language}] {flag}{str(text)[:args.width]}")
                    print(f"  {model_name} hint={hint or 'auto'}: {dt:.1f} s")
                del model
                torch.cuda.empty_cache()
            for i in chunks:
                seg = segs[i]
                print(f"\n#{i}  {seg['start']:.1f}-{seg['end']:.1f} s")
                if first is not None and i < len(first):
                    print(f"  {'first pass':34} {(first[i].get('text') or '')[:args.width]}")
                for key, line in rows[i].items():
                    print(f"  {key:34} {line}")

    if args.ladder:
        from cantocaptions_ai.languages import get_language_pack
        from cantocaptions_ai.pipeline.asr import join_clauses
        from cantocaptions_ai.pipeline.asr_repair import (
            finish, plan_chunks, run_ladder, run_language_check)

        pack = get_language_pack("yue")
        files, results = [], []
        for name, segs, first, chunks in episodes:
            if first is None:
                raise SystemExit(f"--ladder needs {name}'s transcription checkpoint")
            wanted = set(chunks)
            # Only the chosen chunks are planned; the rest stand in as plain text.
            result = {"segments": [dict(seg) if i in wanted else {**seg, "text": "x"}
                                   for i, seg in enumerate(first)], "language": "Cantonese"}
            files.append((result, segs))
            results.append(result)
        plan = [c for c in plan_chunks(files, settings) if c.index in set(episodes[c.file][3])]
        print(f"planned: {len(plan)} chunk(s), {sum(c.split for c in plan)} to split")
        own_name, foreign_name = args.models.split(",")[:2]
        if any(c.split for c in plan):
            t0 = time.time()
            model = _load(own_name, args)
            n = run_ladder([c for c in plan if c.split],
                           lambda wavs: model.transcribe(wavs, "Cantonese", token_floor=64),
                           settings)
            print(f"ladder: {n} piece(s) in {time.time() - t0:.1f} s")
            del model
            torch.cuda.empty_cache()
        if settings.check_language and plan:
            t0 = time.time()
            model = _load(foreign_name, args)
            n = run_language_check(plan, lambda wavs: model.transcribe(wavs, None, token_floor=64),
                                   settings.native,
                                   lambda wavs: model.transcribe(wavs, "Cantonese", token_floor=64))
            print(f"language check: {n} piece(s) in {time.time() - t0:.1f} s")
            del model
            torch.cuda.empty_cache()
        rows = finish(plan, results, settings,
                      lambda texts: join_clauses(texts, pack.punctuation, pack.script))
        outcomes: Dict[str, int] = defaultdict(int)
        for (f, start, line), chunk in zip(rows, plan):
            name, _, first, _ = episodes[f]
            outcomes[line.split(": ", 1)[1]] += 1
            seg = results[f]["segments"][chunk.index]
            if args.quiet and line.endswith(": unchanged"):
                continue
            print(f"\n{name[:30]} #{chunk.index}  {seg['start']:.1f}-{seg['end']:.1f} s  "
                  f"[{seg.get('repair')}]  {line}")
            print(f"  before: {(first[chunk.index].get('text') or '')[:args.width]}")
            print(f"  after:  {(seg.get('text') or '')[:args.width]}")
            if not args.quiet:
                for piece in seg.get("pieces", []):
                    print(f"     {piece}")
        print("\noutcomes:")
        for outcome, n in sorted(outcomes.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4}  {outcome}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="detector flag rates over every checkpoint")
    scan.add_argument("--debug_dir", default="temp")
    scan.add_argument("--stage", default="transcription")
    scan.add_argument("--examples", type=int, default=4)
    scan.add_argument("--export", default=None, help="write flagged chunks to this TSV")
    scan.set_defaults(fn=cmd_scan)

    score = sub.add_parser("score", help="precision/recall against hand labels")
    score.add_argument("--debug_dir", default="temp")
    score.add_argument("--stage", default="transcription")
    score.add_argument("--labels", default="experiments/hallucination/labels.toml")
    score.set_defaults(fn=cmd_score)

    rerun = sub.add_parser("rerun", help="re-run chosen chunks on the GPU")
    rerun.add_argument("--debug_dir", default="temp")
    rerun.add_argument("--name", required=True,
                       help="the episode's debug folder name; several, comma-separated, for --ladder")
    rerun.add_argument("--chunks", required=True,
                       help="e.g. 73,82,85-96, or 'auto' for every chunk repair would plan")
    rerun.add_argument("--quiet", action="store_true", help="--ladder: only changed chunks, no pieces")
    rerun.add_argument("--models", default="cantocaptions-cantonese-ASR,Qwen3-ASR")
    rerun.add_argument("--device", default="cuda")
    rerun.add_argument("--batch_size", type=int, default=4)
    rerun.add_argument("--retry_penalty", type=float, default=1.1)
    rerun.add_argument("--width", type=int, default=110)
    rerun.add_argument("--probe_lid", action="store_true")
    rerun.add_argument("--ladder", action="store_true")
    rerun.add_argument("--native", default=",".join(NATIVE))
    rerun.add_argument("--foreign", default="transcribe")
    rerun.add_argument("--split_depth", type=int, default=3)
    rerun.add_argument("--split_min_piece", type=float, default=4.0)
    rerun.set_defaults(fn=cmd_rerun)

    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.fn(args)


if __name__ == "__main__":
    main()

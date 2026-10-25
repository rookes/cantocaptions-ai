"""How far the proofreader can trust a Qwen3-ASR model's preference between particles.

The proofreading tools (`ccp particles`, cantocaptions-proofread) score every clause-final
particle of a draft against its alternatives (喇 vs 啦 vs 囖 ...) with
`QwenPipelineNative.score_alternatives`, and the review agent may rest a correction on a
*clear* preference. This measures, on the dataset's held-out shows (raw audio, the way the
tools hear an episode), what "clear" should mean:

  false alarm   on the real (correct) transcript, the model prefers another particle over
                the transcript's by at least T nats
  detection     for each alternative a, as if the draft had a by mistake: the model
                prefers the real particle over a by at least T

per margin threshold T, overall and per confusion set and pair; and, beside it, the strict-
corpus collocation prior (`ccp.colloc`, mined with the held-out shows excluded): where a
strong collocation fires, how often its particle is the real one.

Slots and alternatives come from the proofreader itself (`ccp.particles`), so this measures
exactly what the tool does:

    uv run python scripts/eval_particle_scoring.py --split dev --out ../cantocaptions-eval/particles/dev
    uv run python scripts/eval_particle_scoring.py --split test --rows-only ...   # then --report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
HELD_OUT_SPLITS = ("dev", "test")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def show_of(row: dict) -> str:
    return Path(row["audio"]).parent.name


def held_out_shows(export: Path) -> tuple[str, ...]:
    return tuple(sorted({show_of(r) for s in HELD_OUT_SPLITS for r in _read_jsonl(export / f"{s}_raw.jsonl")}))


def score_rows(args) -> list[dict]:
    from ccp import particles
    from cantocaptions_ai.pipeline.asr import load_model
    from cantocaptions_ai.pipeline.asr_scoring import Spot
    from cantocaptions_ai.utils.audio import load_audio

    rows = _read_jsonl(args.export / f"{args.split}_raw.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    stage = load_model(args.model, "cuda", language="yue")
    out = []
    for n, row in enumerate(rows, 1):
        text = row["text"].split("<asr_text>", 1)[-1]
        slots = [s for s in particles.slots(text) if particles.set_of(s.last)]
        if not slots:
            continue
        wav = load_audio(str(args.dataset / row["audio"]))
        spots = [Spot(s.start, s.end, particles.alternatives(s.run)) for s in slots]
        scored = stage.score_alternatives([{"audio": wav, "text": text, "spots": spots}], "yue")[0]
        by_start = {sc.start: sc for sc in scored}
        for s in slots:
            sc = by_start.get(s.start)
            if sc is None:
                continue
            out.append({"show": show_of(row), "audio": row["audio"], "start": s.start, "gold": s.run,
                        "clause": s.clause, "after": s.after, "set": particles.set_of(s.last),
                        "lp": sc.logprobs, "tokens": sc.tokens})
        if n % 25 == 0:
            print(f"{n}/{len(rows)} clips, {len(out)} slots", file=sys.stderr, flush=True)
    return out


def verify(args) -> None:
    """The cached-continuation scores against plain full-sequence scoring on a few clips:
    the difference between two wordings must agree."""
    import torch
    from ccp import particles
    from cantocaptions_ai.pipeline.asr import load_model
    from cantocaptions_ai.pipeline.asr_scoring import Spot
    from cantocaptions_ai.utils.audio import load_audio

    stage = load_model(args.model, "cuda", language="yue")
    proc, model, tok = stage.processor, stage.model, stage.processor.tokenizer
    checked = 0
    for row in _read_jsonl(args.export / f"{args.split}_raw.jsonl"):
        text = row["text"].split("<asr_text>", 1)[-1]
        slots = [s for s in particles.slots(text) if particles.set_of(s.last)]
        if not slots:
            continue
        wav = load_audio(str(args.dataset / row["audio"]))
        s = slots[0]
        alts = particles.alternatives(s.run)[:2]
        sc = stage.score_alternatives([{"audio": wav, "text": text, "spots": [Spot(s.start, s.end, alts)]}], "yue")[0][0]

        def full(t: str) -> float:
            inputs = proc.apply_transcription_request(audio=[wav], language="Cantonese").to(model.device, model.dtype)
            prompt = inputs["input_ids"]
            ids = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"].to(prompt.device)
            eos = torch.tensor([[tok.eos_token_id]], device=prompt.device)
            seq = torch.cat([prompt, ids, eos], dim=1)
            batch = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                logits = model(**batch, input_ids=seq, attention_mask=torch.ones_like(seq)).logits
            lp = torch.log_softmax(logits[0, prompt.shape[1] - 1:-1].float(), -1)
            return float(lp.gather(-1, seq[0, prompt.shape[1]:, None]).sum())

        base = full(text)
        for a in alts:
            if a not in sc.logprobs:
                continue
            want = full(particles.swap(text, s, a)) - base
            got = sc.logprobs[a] - sc.logprobs[s.run]
            print(f"{s.run}->{a}: full-sequence {want:+.2f}  cached {got:+.2f}  (tail differs: {abs(want - got):.2f})")
        checked += 1
        if checked >= args.verify:
            return


def report(rows: list[dict], held_out: tuple[str, ...], args) -> dict:
    from ccp import colloc

    def margins(r):
        gold = r["lp"].get(r["gold"])
        others = {w: lp for w, lp in r["lp"].items() if w != r["gold"]}
        return gold, others

    out: dict = {"slots": len(rows), "clips": len({r["audio"] for r in rows}),
                 "shows": sorted({r["show"] for r in rows})}
    usable = [r for r in rows if r["gold"] in r["lp"] and len(r["lp"]) > 1]
    out["argmax_is_gold"] = round(sum(max(r["lp"], key=r["lp"].get) == r["gold"] for r in usable) / len(usable), 4)

    table = []
    for t in THRESHOLDS:
        fa = sum(1 for r in usable if max(margins(r)[1].values()) - margins(r)[0] >= t)
        det = tot = 0
        for r in usable:
            gold, others = margins(r)
            for lp in others.values():
                tot += 1
                det += gold - lp >= t
        table.append({"T": t, "false_alarm_rate": round(fa / len(usable), 4), "false_alarms": fa,
                      "detection_rate": round(det / tot, 4), "simulated_errors": tot})
    out["by_threshold"] = table

    pairs = defaultdict(lambda: [0, 0, []])      # (gold, alt) -> [n, model prefers gold, margins]
    for r in usable:
        gold, others = margins(r)
        for w, lp in others.items():
            p = pairs[(r["gold"], w)]
            p[0] += 1
            p[1] += gold > lp
            p[2].append(gold - lp)
    out["pairs"] = [{"gold": g, "alt": a, "n": n, "prefers_gold": round(k / n, 3),
                     "median_margin": round(sorted(m)[len(m) // 2], 2)}
                    for (g, a), (n, k, m) in sorted(pairs.items(), key=lambda kv: -kv[1][0]) if n >= args.min_pair]

    by_set = defaultdict(list)
    for r in usable:
        by_set[r["set"]].append(r)
    out["by_set"] = {k: {"slots": len(v), "argmax_is_gold": round(
        sum(max(r["lp"], key=r["lp"].get) == r["gold"] for r in v) / len(v), 4)} for k, v in by_set.items()}

    # The collocations, mined without the held-out shows: the strongest pull as a prior, and
    # the scan's own rule (`colloc.against`) as a detector, like the model above.
    if not args.no_colloc:
        mined_out = colloc.mined("strict", held_out)
        fired = right = agree = agree_right = 0
        against_fa = against_det = against_tot = 0
        for r in usable:
            q = "？" in r["after"] or "?" in r["after"]
            gold = r["gold"][-1]
            pulls = colloc.pulls(r["clause"], gold, q, exclude=held_out)
            if pulls and pulls[0].share >= args.prior_share:
                p = pulls[0]
                fired += 1
                right += gold == p.best
                if max(r["lp"], key=r["lp"].get).endswith(p.best):
                    agree += 1
                    agree_right += gold == p.best
            against_fa += colloc.against(r["clause"], gold, q, exclude=held_out) is not None
            for w in r["lp"]:
                if w == r["gold"] or w[:-1] != r["gold"][:-1] or not colloc.set_of(w[-1]):
                    continue
                against_tot += 1
                hit = colloc.against(r["clause"], w[-1], q, exclude=held_out)
                against_det += hit is not None and hit.best == gold
        out["colloc_prior"] = {"clauses_mined": len(mined_out["clauses"]), "fires": fired,
                               "fires_rate": round(fired / len(usable), 4),
                               "prior_right": round(right / fired, 4) if fired else None,
                               "model_agrees": agree,
                               "right_when_model_agrees": round(agree_right / agree, 4) if agree else None}
        out["colloc_against"] = {"false_alarms": against_fa, "false_alarm_rate": round(against_fa / len(usable), 4),
                                 "simulated_errors": against_tot,
                                 "detection_rate": round(against_det / against_tot, 4) if against_tot else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=HELD_OUT_SPLITS, default="dev")
    ap.add_argument("--dataset", type=Path, default=ROOT.parent / "cantocaptions-dataset")
    ap.add_argument("--proofread", type=Path, default=ROOT.parent / "cantocaptions-proofread")
    ap.add_argument("--model", default="cantocaptions-cantonese-ASR")
    ap.add_argument("--out", type=Path, required=True, help="folder for rows.jsonl and report.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips")
    ap.add_argument("--verify", type=int, default=0, help="check the scorer against full-sequence scoring on N clips, then stop")
    ap.add_argument("--report", action="store_true", help="report from an existing rows.jsonl (no GPU)")
    ap.add_argument("--no-colloc", action="store_true")
    ap.add_argument("--prior-share", type=float, default=0.75)
    ap.add_argument("--min-pair", type=int, default=15)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(args.proofread / "src"))
    args.export = args.dataset / "data" / "export"

    if args.verify:
        verify(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.jsonl"
    if args.report:
        rows = _read_jsonl(rows_path)
    else:
        rows = score_rows(args)
        with open(rows_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    result = report(rows, held_out_shows(args.export), args)
    result.update(split=args.split, model=args.model)
    (args.out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Can a missing subtitle line be recovered from the reference and the audio?

Machine-made subtitles drop lines (VAD misses quiet or overlapped speech, ASR returns nothing).
The reference subtitle often still has a line there. The proofreader (cantocaptions-proofread)
is to notice such *orphan* reference lines, decide whether the draft really lacks the line,
and propose a transcription. This measures, before any of that is built, how well each way of
hearing the span recovers what the hand-made subtitles say there.

Test set: strict hand-made episodes with media and a Chinese reference (proofread workspaces),
each transcribed by the default pipeline into a machine draft. An orphan is a reference cue
(shifted by the measured offset) that draft cues cover for less than COVER_MIN of its length.
Each orphan is labelled from the hand-made file:

    P  a spoken hand-made line is there and the draft lacks it (the case to recover)
    C  covered: the draft has that line, timed elsewhere (the reference is mistimed)
    S  on-screen text (bracketed), hand-made line or not
    N  the hand-made file has no line there either

Steps (outputs in --out, default ../cantocaptions-eval/missing):

    build   orphans + labels                              -> cases.jsonl, prints label counts
    run     (GPU) hear each P span several ways            -> hyps.jsonl
    blind   what a proofreader would see, no ground truth  -> blind.jsonl (for candidate writing)
    rerank  (GPU) score every candidate with both models   -> rerank.jsonl
    score   CER of each method against the hand-made line  -> report.json, printed
    timing  (GPU) insert the written lines of one episode, realign it (adjust, --realign_split
            off), and measure where they land against the hand-made lines -> timing-<id>.json

Methods (run): ft / qw = the fine-tuned and the stock recogniser; p03 / p10 = 0.3 s or 1.0 s of
padding, clamped to the neighbouring draft cues; iso = after vocal isolation; ctx = the
reference line as ASR context (reference_context's `labelled` template).

    uv run python scripts/eval_missing_lines.py build --episodes ../cantocaptions-eval/missing/episodes.txt
    uv run python scripts/eval_missing_lines.py run
    uv run python scripts/eval_missing_lines.py blind     # then write candidates.jsonl
    uv run python scripts/eval_missing_lines.py rerank
    uv run python scripts/eval_missing_lines.py score

Results: docs/missing-lines.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
COVER_MIN = 0.3          # share of a reference cue the draft must cover for it to count as there
NEAR_S = 3.0             # how far a mistimed reference may sit from the draft line it matches
COVERED_SIM = 0.6        # text similarity (normalised) for "the draft has this line"
USABLE_CER = 0.2
BRACKETS = ("（）", "()", "【】", "［］", "[]", "《》")
# Standard Written Chinese characters a Cantonese transcription should not need: a context
# prompt that makes the recogniser recite the Mandarin reference shows up as these.
MANDARIN = set("的不是這沒他她們麼嗎什")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------------------- text

_CC = None


def norm(text: str) -> str:
    """Sound characters only, Traditional (HK), spelling variants and the stock recogniser's
    sound-merged spellings folded, so only a difference in what was said counts."""
    global _CC
    from ccp.jyutping import is_sound
    from ccp.render import FOLDS
    if _CC is None:
        import opencc
        _CC = opencc.OpenCC("s2hk")
    text = _CC.convert(text or "").translate(FOLDS["qw"])
    return "".join(ch.lower() for ch in text if is_sound(ch))


def cer(hyp: str, ref: str) -> float:
    from rapidfuzz.distance import Levenshtein
    h, r = norm(hyp), norm(ref)
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(h, r) / len(r)


def contained(want: str, heard: str) -> int:
    """Edit distance of *want* against the best-matching stretch of *heard* (words the hearing
    has around it are free); as `ccp.missing.agreement` measures it."""
    w, h = norm(want), norm(heard)
    prev = [0] * (len(h) + 1)
    for i, x in enumerate(w, 1):
        cur = [i]
        for j, y in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return min(prev)


def similarity(a: str, b: str) -> float:
    from rapidfuzz import fuzz
    a, b = norm(a), norm(b)
    return fuzz.partial_ratio(a, b) / 100 if a and b else 0.0


def bracketed(text: str) -> bool:
    t = text.strip()
    return any(t.startswith(o) and t.endswith(c) for o, c in BRACKETS)


# ------------------------------------------------------------------------------------ build

def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _cues(path: Path):
    from ccp.srtio import SrtFile
    return [c for c in SrtFile(path).cues if not c.an8 and c.end > c.start]


def episode_sources(eid: str, work: Path, drafts: Path) -> dict:
    """draft, human, reference, media, audio_track and show of a proofread workspace."""
    ep = json.loads((work / eid / "episode.json").read_text(encoding="utf-8"))
    outs = sorted(p for p in (drafts / eid).glob("*.srt"))
    if not outs:
        raise SystemExit(f"{eid}: no draft in {drafts / eid}")
    return {"id": eid, "draft": str(outs[0]), "human": ep["source_srt"], "reference": str(work / eid / "reference.srt"),
            "media": ep["media"], "audio_track": ep.get("audio_track", 0), "show": ep["show_slug"]}


def build_episode(src: dict) -> list[dict]:
    from ccp.reference import agreement, best_shift
    eid = src["id"]
    human = _cues(Path(src["human"]))
    ref = _cues(Path(src["reference"]))
    draft = _cues(Path(src["draft"]))
    starts = [c.start for c in draft]
    raw = agreement(starts, [c.start for c in ref])
    shift, best = best_shift(starts, [c.start for c in ref])
    offset = shift if best > raw + 0.05 else 0.0
    rows = []
    for k, r in enumerate(ref):
        r0, r1 = r.start + offset, r.end + offset
        dur = r1 - r0
        cover = sum(_overlap(r0, r1, d.start, d.end) for d in draft) / dur
        if cover >= COVER_MIN:
            continue
        hum = [h for h in human if _overlap(r0, r1, h.start, h.end) >= COVER_MIN * min(dur, h.end - h.start)]
        spoken = [h for h in hum if not bracketed(h.text)]
        missing = []
        for h in spoken:
            at_time = sum(_overlap(h.start, h.end, d.start, d.end) for d in draft) / (h.end - h.start)
            near = [d for d in draft if d.end > h.start - NEAR_S and d.start < h.end + NEAR_S]
            if at_time >= COVER_MIN or any(similarity(h.text, d.text) >= COVERED_SIM for d in near):
                continue
            missing.append(h)
        if not hum:
            label = "S" if bracketed(r.text) else "N"
        elif not spoken:
            label = "S"
        elif not missing:
            label = "C"
        else:
            label = "P"
        mid = (r0 + r1) / 2
        prev = [d for d in draft if (d.start + d.end) / 2 < mid]
        nxt = [d for d in draft if (d.start + d.end) / 2 >= mid]
        row = {"case": f"{eid}#{k + 1}", "episode": eid, "show": src["show"], "label": label,
               "media": src["media"], "audio_track": src.get("audio_track", 0),
               "ref": r.text, "ref_start": round(r0, 3), "ref_end": round(r1, 3), "cover": round(cover, 3),
               "prev": [{"text": d.text, "start": d.start, "end": d.end} for d in prev[-2:]],
               "next": [{"text": d.text, "start": d.start, "end": d.end} for d in nxt[:2]]}
        if label == "P":
            row["truth"] = "，".join(h.text.replace("\n", "，") for h in missing)
            row["truth_start"], row["truth_end"] = missing[0].start, missing[-1].end
        rows.append(row)
    print(f"{eid}: {len(ref)} reference cues, offset {offset:+.2f} s, {len(rows)} orphans "
          f"{dict(Counter(r['label'] for r in rows))}", file=sys.stderr)
    return rows


def span(case: dict, pad: float) -> tuple[float, float]:
    """The orphan's reference span ± pad, never reaching into a neighbouring draft cue."""
    lo, hi = case["ref_start"] - pad, case["ref_end"] + pad
    if case["prev"]:
        lo = max(lo, case["prev"][-1]["end"])
    if case["next"]:
        hi = min(hi, case["next"][0]["start"])
    if hi - lo < 0.3:
        lo, hi = case["ref_start"], case["ref_end"]
    return max(0.0, lo), hi


def cmd_build(a) -> None:
    """--episodes: proofread workspace ids, with their pipeline drafts in --drafts/<id>/;
    --manifest: a JSON list of {id, draft, human, reference, media, audio_track, show} for
    drafts made elsewhere (the library's AI-generated files)."""
    sources = []
    if a.episodes:
        sources += [episode_sources(eid, a.work, a.drafts or a.out / "drafts")
                    for eid in Path(a.episodes).read_text(encoding="utf-8").split()]
    if a.manifest:
        sources += json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    rows = [r for src in sources for r in build_episode(src)]
    _write_jsonl(a.out / "cases.jsonl", rows)
    by = Counter(r["label"] for r in rows)
    print(json.dumps({"orphans": len(rows), "labels": dict(by)}, ensure_ascii=False))


# -------------------------------------------------------------------------------------- run

def _clips(cases: list[dict], work: Path) -> dict:
    """{(case, pad): clip} for pads 0.3 and 1.0."""
    from cantocaptions_ai.utils.audio import load_audio
    out = {}
    by_ep = defaultdict(list)
    for c in cases:
        by_ep[c["episode"]].append(c)
    for eid, cs in by_ep.items():
        audio = load_audio(cs[0]["media"], audio_track=cs[0].get("audio_track", 0))
        for c in cs:
            for pad in (0.3, 1.0):
                s, e = span(c, pad)
                out[(c["case"], pad)] = audio[int(s * SR):int(e * SR)]
        del audio
    return out


def _isolate(clips: dict) -> dict:
    import gc
    import torch
    from cantocaptions_ai.pipeline.vocal_isolation import load_vocal_isolation
    proc = load_vocal_isolation("mbroformer", "cuda")
    keys = list(clips)
    items = [{"audio_path": f"clip{i}", "vad_segments": [{"start": 0.0, "end": len(clips[k]) / SR, "audio": clips[k]}]}
             for i, k in enumerate(keys)]
    done = proc.run(items)
    del proc
    gc.collect()
    torch.cuda.empty_cache()
    return {k: it["vad_segments"][0]["audio"] for k, it in zip(keys, done)}


MODELS = {"ft": "cantocaptions-cantonese-ASR", "qw": "Qwen3-ASR"}


def _load(tag: str):
    from cantocaptions_ai.pipeline.asr import load_model
    from cantocaptions_ai.pipeline.config import PipelineConfig
    return load_model(MODELS[tag], "cuda", language="yue", batch_size=PipelineConfig.batch_size, compile="off")


def _free(stage) -> None:
    import gc
    import torch
    del stage
    gc.collect()
    torch.cuda.empty_cache()


def cmd_run(a) -> None:
    from cantocaptions_ai.pipeline.reference_context import CONTEXT_TEMPLATES
    cases = [c for c in _read_jsonl(a.out / "cases.jsonl") if c["label"] == "P"][: a.limit or None]
    print(f"{len(cases)} cases", file=sys.stderr)
    plain = _clips(cases, a.work)
    iso = _isolate(plain)
    template = CONTEXT_TEMPLATES["labelled"]
    jobs = []      # (case, method, clip, context)
    for c in cases:
        ctx = template.format(text=c["ref"].replace("\n", " "))
        for pad, p in ((0.3, "p03"), (1.0, "p10")):
            jobs.append((c["case"], f"{p}", plain[(c["case"], pad)], None))
            jobs.append((c["case"], f"{p}_iso", iso[(c["case"], pad)], None))
            jobs.append((c["case"], f"{p}_ctx", plain[(c["case"], pad)], ctx))
        jobs.append((c["case"], "p10_iso_ctx", iso[(c["case"], 1.0)], ctx))
    hyps: dict[str, dict] = defaultdict(dict)
    for tag in ("ft", "qw"):
        stage = _load(tag)
        segs = [{"start": 0.0, "end": len(clip) / SR, "audio": clip, **({"context": ctx} if ctx else {})}
                for _, _, clip, ctx in jobs]
        texts = [s.get("text", "") for s in stage.process(segs)["segments"]]
        if len(texts) != len(jobs):
            raise SystemExit(f"{tag}: {len(texts)} texts for {len(jobs)} clips")
        for (case, method, _, _), text in zip(jobs, texts):
            hyps[case][f"{tag}_{method}"] = text
        _free(stage)
        print(f"{tag} done", file=sys.stderr)
    _write_jsonl(a.out / "hyps.jsonl", [{"case": k, "hyps": v} for k, v in hyps.items()])


# ------------------------------------------------------------------------------------ blind

def cmd_blind(a) -> None:
    """What the proofreader would have for each P case, without the hand-made line."""
    hyps = {r["case"]: r["hyps"] for r in _read_jsonl(a.out / "hyps.jsonl")}
    rows = []
    for c in _read_jsonl(a.out / "cases.jsonl"):
        if c["case"] not in hyps:
            continue
        heard = sorted({t.strip() for t in hyps[c["case"]].values() if t.strip()})
        rows.append({"case": c["case"], "reference": c["ref"], "seconds": round(c["ref_end"] - c["ref_start"], 1),
                     "before": [p["text"] for p in c["prev"]], "after": [n["text"] for n in c["next"]],
                     "heard": heard})
    # A folder of its own, away from the ground truth, for whoever writes the candidates.
    _write_jsonl(a.out / "blind" / "blind.jsonl", rows)
    print(f"{len(rows)} cases -> {a.out / 'blind' / 'blind.jsonl'}; write {a.out / 'blind' / 'candidates.jsonl'} "
          '({"case", "candidates": [...]}) from it alone', file=sys.stderr)


# ----------------------------------------------------------------------------------- rerank

def candidates_for(case: str, hyps: dict, written: dict) -> list[str]:
    seen, out = set(), []
    for t in list(hyps.get(case, {}).values()) + written.get(case, []):
        t = (t or "").strip()
        if t and norm(t) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def cmd_rerank(a) -> None:
    cases = {c["case"]: c for c in _read_jsonl(a.out / "cases.jsonl") if c["label"] == "P"}
    hyps = {r["case"]: r["hyps"] for r in _read_jsonl(a.out / "hyps.jsonl")}
    written = {r["case"]: r["candidates"] for r in _read_jsonl(a.out / "blind" / "candidates.jsonl")}
    todo = [cases[k] for k in hyps if k in cases]
    plain = _clips(todo, a.work)
    iso = _isolate(plain)
    cands = {c["case"]: candidates_for(c["case"], hyps, written) for c in todo}
    scores: dict[str, dict] = defaultdict(dict)
    for tag in ("ft", "qw"):
        stage = _load(tag)
        for clips, name in ((plain, "p03"), (iso, "p03_iso")):
            items = [{"audio": clips[(c["case"], 0.3)], "texts": cands[c["case"]]} for c in todo]
            for c, res in zip(todo, stage.score_texts(items, "yue")):
                scores[c["case"]][f"{tag}_{name}"] = [[round(lp, 3), n] for lp, n in res]
        _free(stage)
        print(f"{tag} scored", file=sys.stderr)
    _write_jsonl(a.out / "rerank.jsonl", [{"case": k, "candidates": cands[k], "scores": scores[k]} for k in cands])


# ------------------------------------------------------------------------------------ score

def _pick(cands: list[str], lps: list[list]) -> tuple[int, float]:
    """Index of the most likely candidate and its margin (nats) over the next."""
    order = sorted(range(len(cands)), key=lambda i: -lps[i][0])
    margin = lps[order[0]][0] - lps[order[1]][0] if len(order) > 1 else float("inf")
    return order[0], margin


def cmd_score(a) -> None:
    cases = {c["case"]: c for c in _read_jsonl(a.out / "cases.jsonl")}
    held_out = set(a.held_out.split(",")) if a.held_out else _held_out_shows(a.dataset)
    labels = Counter(c["label"] for c in cases.values())
    hyps = {r["case"]: r["hyps"] for r in _read_jsonl(a.out / "hyps.jsonl")}
    rerank = {r["case"]: r for r in _read_jsonl(a.out / "rerank.jsonl")}
    written = {r["case"]: r["candidates"] for r in _read_jsonl(a.out / "blind" / "candidates.jsonl")}
    groups = {"held_out": lambda c: c["show"] in held_out, "trained": lambda c: c["show"] not in held_out,
              "all": lambda c: True}
    for show in sorted({c["show"] for c in cases.values() if c["label"] == "P"}):
        groups[f"show:{show}"] = (lambda sh: lambda c: c["show"] == sh)(show)
    per_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    mandarin: dict[str, list[int]] = defaultdict(list)
    calib = []
    for key, hs in hyps.items():
        c = cases[key]
        rows = dict(hs)
        heard = [t for t in hs.values() if t.strip()]
        if heard:
            rows["oracle_heard"] = min(heard, key=lambda t: cer(t, c["truth"]))
        if key in rerank:
            rr = rerank[key]
            for name, lps in rr["scores"].items():
                i, margin = _pick(rr["candidates"], lps)
                rows[f"rerank_{name}"] = rr["candidates"][i]
                if name == "ft_p03_iso":
                    agree = sum(1 for t in hs.values() if t.strip() and cer(t, rr["candidates"][i]) <= USABLE_CER)
                    calib.append({"cer": cer(rr["candidates"][i], c["truth"]), "margin": margin, "agree": agree,
                                  "written": rr["candidates"][i] in written.get(key, [])})
            # both models together: the sum of their joint log-probabilities (same clip, same texts)
            both = [[a_[0] + b_[0], a_[1]] for a_, b_ in zip(rr["scores"]["ft_p03_iso"], rr["scores"]["qw_p03_iso"])]
            rows["rerank_both_iso"] = rr["candidates"][_pick(rr["candidates"], both)[0]]
            rows["oracle"] = min(rr["candidates"], key=lambda t: cer(t, c["truth"]))
            if written.get(key):
                rows["written_first"] = written[key][0]
        for m, t in rows.items():
            for g, f in groups.items():
                if f(c):
                    per_method[m][g].append(cer(t, c["truth"]))
            if "ctx" in m or m.startswith("rerank") or m == "written_first":
                mandarin[m].append(int(any(ch in MANDARIN for ch in t) and not any(ch in MANDARIN for ch in c["truth"])))

    def summ(xs):
        return {"n": len(xs), "usable": round(sum(x <= USABLE_CER for x in xs) / len(xs), 3) if xs else None,
                "exact": round(sum(x == 0 for x in xs) / len(xs), 3) if xs else None,
                "median_cer": round(statistics.median(xs), 3) if xs else None,
                "empty": round(sum(x >= 1.0 for x in xs) / len(xs), 3) if xs else None}

    table = {m: {g: summ(v) for g, v in gs.items()} for m, gs in per_method.items()}
    report = {"labels": dict(labels), "held_out_shows": sorted(held_out),
              "methods": dict(sorted(table.items(), key=lambda kv: -(kv[1].get("all", {}).get("usable") or 0))),
              "mandarin_rate": {m: round(sum(v) / len(v), 3) for m, v in mandarin.items() if v}}
    if calib:
        bands = {}
        for lo, hi in ((0, 1), (1, 3), (3, 6), (6, float("inf"))):
            xs = [r["cer"] for r in calib if lo <= r["margin"] < hi]
            bands[f"margin {lo}-{hi}"] = summ(xs)
        for k in (0, 1, 2, 3):
            xs = [r["cer"] for r in calib if (r["agree"] >= k if k == 3 else r["agree"] == k)]
            bands[f"agree {k}{'+' if k == 3 else ''}"] = summ(xs)
        bands["picked a written candidate"] = summ([r["cer"] for r in calib if r["written"]])
        bands["picked a recogniser's text"] = summ([r["cer"] for r in calib if not r["written"]])
        report["calibration_ft_p03_iso"] = bands
    # Grading the written line: which recognisers' close, unhinted hearings agree with it.
    agree_defs = {"whole": lambda t, w: bool(norm(t)) and cer(t, w) <= USABLE_CER,
                  "contained": lambda t, w: bool(norm(t)) and contained(w, t) <= USABLE_CER * max(1, len(norm(w)))}
    grading = {}
    for name, agree in agree_defs.items():
        bands = defaultdict(list)
        for key, cands in written.items():
            if key not in hyps or not cands:
                continue
            hs, w = hyps[key], cands[0]
            ft = any(agree(hs[m], w) for m in ("ft_p03", "ft_p03_iso"))
            qw = any(agree(hs[m], w) for m in ("qw_p03", "qw_p03_iso"))
            band = "both" if ft and qw else "only ft" if ft else "only qw" if qw else "neither"
            bands[band].append(cer(w, cases[key]["truth"]))
        grading[name] = {b: summ(v) for b, v in bands.items()}
    report["grading_written_line"] = grading
    (a.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_timing(a) -> None:
    """Insert each P case's first written candidate into the episode's draft where the
    proofreader would (the reference span, inside the gap, at least 0.75 s), realign the
    result as the cleanup does, and compare the inserted cues' times with the hand-made ones,
    before and after the realign. Also checks that the realign keeps the cues 1:1."""
    import os
    import subprocess
    from ccp.srtio import SrtFile, fmt_time
    cases = [c for c in _read_jsonl(a.out / "cases.jsonl") if c["label"] == "P" and c["episode"] == a.episode]
    written = {r["case"]: r["candidates"] for r in _read_jsonl(a.out / "blind" / "candidates.jsonl")}
    src = (next(m for m in json.loads(Path(a.manifest).read_text(encoding="utf-8")) if m["id"] == a.episode)
           if a.manifest else episode_sources(a.episode, a.work, a.drafts or a.out / "drafts"))
    draft = SrtFile(Path(src["draft"]))
    cues = [(c.start, c.end, c.text) for c in draft.cues]
    added = []
    for c in cases:
        lo = c["prev"][-1]["end"] if c["prev"] else 0.0
        hi = c["next"][0]["start"] if c["next"] else c["ref_end"] + 5
        s, e = max(lo, c["ref_start"]), min(hi, c["ref_end"])
        if e - s < 0.75:
            e = min(hi, s + 0.75)
            s = max(lo, min(s, e - 0.75))
        text = written.get(c["case"], [c["ref"]])[0]
        cues.append((s, e, text))
        added.append({"case": c["case"], "text": text, "placed": [round(s, 3), round(e, 3)],
                      "truth": [c["truth_start"], c["truth_end"]]})
    cues.sort(key=lambda x: x[0])
    work = a.out / "timing" / a.episode
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "inserted.srt"
    inp.write_text("".join(f"{i}\n{fmt_time(s)} --> {fmt_time(e)}\n{t}\n\n" for i, (s, e, t) in enumerate(cues, 1)),
                   encoding="utf-8")
    cmd = ["uv", "run", "--no-sync", "cantocaptions", src["media"], "--audio_track", str(src.get("audio_track", 0)),
           "--realign", str(inp), "--realign_mode", "adjust", "--realign_split", "off",
           "--output_dir", str(work), "--output_format", "srt", "--debug_dir", str(work / "debug")]
    with open(work / "run.log", "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    outs = [p for p in work.glob("*.srt") if p.name != "inserted.srt"]
    if code or not outs:
        raise SystemExit(f"realign failed ({code}): {work / 'run.log'}")
    out = SrtFile(outs[0]).cues
    by_text = defaultdict(list)
    for c in out:
        by_text[c.text].append(c)
    # Cues the realign does not trust (realign/suspect.srt): for those, the fallback keeps the
    # line where the reference put it.
    suspects = set()
    for sp in (work / "debug").rglob("suspect.srt"):
        suspects |= {(round(c.start, 2), round(c.end, 2)) for c in SrtFile(sp).cues}
    before, after, fallback, lost, right = [], [], [], 0, []
    truth_of = {c["case"]: c["truth"] for c in cases}
    for x in added:
        hit = [c for c in by_text.get(x["text"], [])]
        if not hit:
            lost += 1
            continue
        c = min(hit, key=lambda c: abs(c.start - x["placed"][0]))
        b = (x["placed"][0] - x["truth"][0], x["placed"][1] - x["truth"][1])
        r = (c.start - x["truth"][0], c.end - x["truth"][1])
        doubted = (round(c.start, 2), round(c.end, 2)) in suspects
        before.append(b)
        after.append(r)
        fallback.append(b if doubted else r)
        right.append(cer(x["text"], truth_of[x["case"]]) <= USABLE_CER)

    def summ(pairs):
        st = sorted(abs(p[0]) for p in pairs)
        en = sorted(abs(p[1]) for p in pairs)
        q = lambda xs, f: round(xs[min(len(xs) - 1, int(f * len(xs)))], 2) if xs else None
        return {"n": len(pairs), "start_median": q(st, 0.5), "start_p90": q(st, 0.9),
                "start_within_0.5s": round(sum(x <= 0.5 for x in st) / len(st), 2) if st else None,
                "end_median": q(en, 0.5), "end_p90": q(en, 0.9)}
    res = {"episode": a.episode, "inserted": len(added), "cues_in": len(cues), "cues_out": len(out),
           "inserted_lost": lost, "realign_doubted": sum(1 for b, f in zip(before, fallback) if b is f),
           "placed_at_reference": summ(before), "after_realign": summ(after),
           "realign_unless_doubted": summ(fallback),
           "text_right": {"n": sum(right),
                          "placed_at_reference": summ([b for b, k in zip(before, right) if k]),
                          "after_realign": summ([r for r, k in zip(after, right) if k]),
                          "realign_unless_doubted": summ([f for f, k in zip(fallback, right) if k])}}
    (a.out / f"timing-{a.episode}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def _held_out_shows(dataset: Path) -> set[str]:
    splits = json.loads((dataset / "data" / "splits.json").read_text(encoding="utf-8"))
    return {k for k, v in splits.items() if v["split"] in ("dev", "test")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run", "blind", "rerank", "score", "timing"])
    ap.add_argument("--episode", help="timing: the episode id (as in cases.jsonl)")
    ap.add_argument("--episodes", help="build: a file of proofread workspace ids")
    ap.add_argument("--manifest", help="build: a JSON list of {id, draft, human, reference, media, audio_track, show}")
    ap.add_argument("--drafts", type=Path, help="build: the machine drafts, one folder per id (default OUT/drafts)")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "cantocaptions-eval" / "missing")
    ap.add_argument("--proofread", type=Path, default=ROOT.parent / "cantocaptions-proofread")
    ap.add_argument("--dataset", type=Path, default=ROOT.parent / "cantocaptions-dataset")
    ap.add_argument("--held-out", help="score: comma-separated show slugs (default: the dataset's dev/test)")
    ap.add_argument("--limit", type=int, help="run: only the first N cases")
    a = ap.parse_args()
    a.work = a.proofread / "work"
    sys.path.insert(0, str(a.proofread / "src"))
    {"build": cmd_build, "run": cmd_run, "blind": cmd_blind, "rerank": cmd_rerank, "score": cmd_score,
     "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    main()

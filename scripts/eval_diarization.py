"""Score diarization at the boundaries cue assembly asks about, against reference subtitles.

The question diarization answers in this pipeline is "do these two adjacent stretches of speech
share a voice?" (see ``pipeline/segmentation._same_speaker``). This script builds boundaries
with a known answer from an ordinary subtitle file and scores how often that question is
answered correctly. No speaker labels are needed. The metrics are defined in
``cantocaptions_ai/utils/speaker_eval.py``:

  change    between the lines of a hyphen-convention two-speaker cue ("-你好\\n-早晨")
  same      the two halves of a one-speaker cue, split at the clause mark nearest its middle
  adjacent  two consecutive cues (truth unknown; reported as a rate)

Each cue is force-aligned with the pipeline's own aligner for --language, to place the change
and clause boundaries in time. The reference must be verbatim and reasonably well timed;
cue ends may be padded.

Diarization comes from one of three sources:

    # a real run: its checkpoint (--debug_dir DIR wrote DIR/<stem>/diarization/result.json)
    python scripts/eval_diarization.py --pair ep01.mkv ep01.srt --debug_dir runs/debug

    # pyannote over the whole file (--diarize_scope file)
    python scripts/eval_diarization.py --pair ep01.mkv ep01.srt --scope file

    # pyannote per pseudo-VAD segment (--diarize_scope segment). The segments are the
    # reference cues merged across gaps < --merge_gap, up to --chunk_size seconds -- close to
    # what VAD produces, without running it.
    python scripts/eval_diarization.py --pair ep01.mkv ep01.srt --pair ep02.mkv ep02.srt --scope segment

    # the speaker-change scorer (--speaker_change), on the same pseudo-VAD segments, with
    # the file's reference cues of 1 s or more as its neighbour pool
    python scripts/eval_diarization.py --pair ep01.mkv ep01.srt --speaker_change

--embeddings also embeds both sides of every boundary with the diarization model's own
embedding network and reports how well their cosine separates change from same, which is
the ceiling for any veto built on comparing the two sides directly. --out writes every
boundary (spans, decision, cosine) as JSONL for further analysis.
"""

import argparse
import json
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# pyannote averages over empty frame sets on very short segments; the result is unused.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cantocaptions_ai.pipeline.speaker_assign import speaker_scope  # noqa: E402
from cantocaptions_ai.utils.audio import SAMPLE_RATE, load_audio  # noqa: E402
from cantocaptions_ai.utils.speaker_eval import (  # noqa: E402
    Boundary,
    boundary_decision,
    decision_rates,
    format_rates,
    hyphen_turns,
    separability,
)
from cantocaptions_ai.utils.subtitles import read_subtitle_cues  # noqa: E402

# Clause and sentence marks a one-speaker cue may be split at. Splitting at a mark rather than
# an arbitrary character puts the boundary where the pipeline's aligned subsegments meet.
_CLAUSE_MARK = re.compile(r"[，,、；;：:。？?！!…]")
# Bracketed cues ("[歌詞]", "(笑)") are sound descriptions or lyrics, not dialogue.
_NON_DIALOGUE = re.compile(r"^\s*[\[［(（【♪]")

MIN_SIDE = 0.3          # seconds; shorter sides carry too little voice to judge
MIN_CHARS = 2           # aligned characters per side
MAX_ADJACENT_GAP = 1.5  # seconds between consecutive cues to count as adjacent
MIN_CHAR_SCORE = 0.05   # aligned characters scoring below this are guesses


class CueAligner:
    """Character timings for one cue's text, with the pipeline's aligner for a language."""

    def __init__(self, language: str, device: str):
        from cantocaptions_ai.pipeline.alignment import load_align_model

        self.device = device
        self.model, self.meta = load_align_model(language, device, vram_checks=False)
        self.dictionary = self.meta["dictionary"]

    def prepare(self, texts: Sequence[str]) -> None:
        """Map out-of-vocabulary characters onto readable ones, once for the whole run."""
        repair = self.meta.get("vocab_repair")
        if repair is not None:
            repair.augment(texts)

    def chars(self, audio: np.ndarray, text: str) -> Optional[List[Tuple[int, float, float, float]]]:
        """``(index in text, start, end, score)`` for every alignable character, or None."""
        import torch

        from cantocaptions_ai.pipeline.alignment import (
            _get_blank_id,
            _run_model_inference,
            backtrack,
            get_trellis,
            merge_repeats,
        )

        keep = [(i, ch) for i, ch in enumerate(text.lower()) if ch in self.dictionary and not ch.isspace()]
        if not keep or len(audio) < MIN_SIDE * SAMPLE_RATE:
            return None
        emission = _run_model_inference(
            self.model, self.meta["type"], torch.from_numpy(audio).unsqueeze(0),
            self.meta["processor"], self.device,
        )[0].cpu()
        tokens = [self.dictionary[ch] for _, ch in keep]
        blank = _get_blank_id(self.dictionary)
        path = backtrack(get_trellis(emission, tokens, blank), emission, tokens, blank)
        if path is None:
            return None
        segments = merge_repeats(path, [ch for _, ch in keep])
        frame = len(audio) / SAMPLE_RATE / emission.size(0)
        return [(i, s.start * frame, s.end * frame, s.score) for (i, _), s in zip(keep, segments)]


def _side(chars, lo: int, hi: int) -> Optional[Tuple[float, float]]:
    picked = [c for c in chars if lo <= c[0] < hi and c[3] > MIN_CHAR_SCORE]
    if len(picked) < MIN_CHARS:
        return None
    span = (picked[0][1], picked[-1][2])
    return span if span[1] - span[0] >= MIN_SIDE else None


def build_boundaries(audio: np.ndarray, cues, aligner: CueAligner, source: str) -> List[Boundary]:
    """Every scorable change/same/adjacent boundary in one file's reference cues."""
    boundaries: List[Boundary] = []
    extents = []  # (cue, aligned extent or None, is multi-speaker)
    for cue in cues:
        if _NON_DIALOGUE.match(cue.text):
            extents.append((cue, None, False))
            continue
        turns = hyphen_turns(cue.text)
        text = "".join(turns) if turns else cue.text.replace("\n", "")
        t0 = max(0.0, cue.start - 0.15)
        t1 = min(len(audio) / SAMPLE_RATE, cue.end + 0.3)
        chars = aligner.chars(audio[int(t0 * SAMPLE_RATE):int(t1 * SAMPLE_RATE)], text)
        if not chars:
            extents.append((cue, None, bool(turns)))
            continue
        chars = [(i, a + t0, b + t0, s) for i, a, b, s in chars]
        extents.append((cue, (chars[0][1], chars[-1][2]), bool(turns)))

        if turns:
            edges = np.cumsum([0] + [len(t) for t in turns])
            for k in range(len(turns) - 1):
                left = _side(chars, edges[k], edges[k + 1])
                right = _side(chars, edges[k + 1], edges[k + 2])
                if left and right:
                    boundaries.append(Boundary("change", left, right, source, {"text": cue.text}))
        else:
            marks = [m.start() for m in _CLAUSE_MARK.finditer(text) if 0 < m.start() < len(text) - 1]
            if marks:
                cut = min(marks, key=lambda i: abs(i - len(text) / 2))
                left, right = _side(chars, 0, cut), _side(chars, cut + 1, len(text))
                if left and right:
                    boundaries.append(Boundary("same", left, right, source, {"text": text}))

    for (c1, e1, m1), (c2, e2, m2) in zip(extents, extents[1:]):
        if e1 is None or e2 is None or m1 or m2:
            continue
        if e2[0] - e1[1] > MAX_ADJACENT_GAP or min(e1[1] - e1[0], e2[1] - e2[0]) < MIN_SIDE:
            continue
        boundaries.append(Boundary(
            "adjacent", e1, e2, source,
            {"text": f"{c1.text} | {c2.text}", "question": c1.text.rstrip().endswith(("？", "?"))},
        ))
    return boundaries


def pseudo_vad_segments(audio: np.ndarray, cues, merge_gap: float, chunk_size: float) -> List[Dict]:
    """Reference cues merged into VAD-like segments, for --scope segment without running VAD."""
    spans: List[List[float]] = []
    for cue in sorted(cues, key=lambda c: c.start):
        start, end = max(0.0, cue.start - 0.25), cue.end + 0.2
        if spans and start - spans[-1][1] <= merge_gap and end - spans[-1][0] <= chunk_size:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    total = len(audio) / SAMPLE_RATE
    return [
        {"start": s, "end": min(e, total), "audio": audio[int(s * SAMPLE_RATE):int(min(e, total) * SAMPLE_RATE)]}
        for s, e in spans if s < total
    ]


def scoped_decision(turns, boundary: Boundary, min_share: float) -> str:
    """boundary_decision, plus the merge gate's rule that labels from different segment scopes
    are not comparable (``segmentation._same_speaker``)."""
    from cantocaptions_ai.utils.speaker_eval import span_label

    left, _ = span_label(turns, boundary.left, min_share)
    right, _ = span_label(turns, boundary.right, min_share)
    if left is not None and right is not None and speaker_scope(left) != speaker_scope(right):
        return "unknown"
    return boundary_decision(turns, boundary, min_share)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("AUDIO", "SUBTITLES"),
                        help="an audio/video file and its verbatim reference subtitles (repeatable)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--debug_dir", help="score the diarization checkpoints of a real run")
    source.add_argument("--scope", choices=("file", "segment"), help="run pyannote here instead")
    source.add_argument("--speaker_change", action="store_true",
                        help="score boundaries with the speaker-change model instead of diarization")
    parser.add_argument("--speaker_change_threshold", type=float, default=None,
                        help="--speaker_change: decision threshold (default: the shipped one)")
    parser.add_argument("--language", default="yue", help="aligner language (default: yue)")
    parser.add_argument("--audio_track", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--diarize_model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--diarize_batch_size", type=int, default=4)
    parser.add_argument("--speaker_confidence", type=float, default=0.7,
                        help="min dominant share for a side to be labeled (the pipeline's setting)")
    parser.add_argument("--merge_gap", type=float, default=1.0, help="--scope segment: pseudo-VAD merge gap")
    parser.add_argument("--chunk_size", type=float, default=28.0, help="--scope segment: max segment length")
    parser.add_argument("--embeddings", action="store_true",
                        help="also score the embedding cosine between the two sides of each boundary")
    parser.add_argument("--out", help="write every boundary as JSONL")
    args = parser.parse_args(argv)

    aligner = CueAligner(args.language, args.device)
    files = []
    for audio_path, subtitle_path in args.pair:
        cues = read_subtitle_cues(subtitle_path)
        files.append((audio_path, cues))
    aligner.prepare([c.text for _, cues in files for c in cues])

    diarizer = scorer = None
    if args.speaker_change:
        from cantocaptions_ai.pipeline.speaker_change import SpeakerChangeScorer
        scorer = SpeakerChangeScorer.load(args.diarize_model, device=args.device,
                                          token=args.hf_token, batch_size=args.diarize_batch_size)
    if args.scope:
        from cantocaptions_ai.pipeline.diarize import load_diarization
        diarizer = load_diarization(
            device=args.device, model_name=args.diarize_model, token=args.hf_token,
            scope=args.scope, batch_size=args.diarize_batch_size, vram_checks=False,
        )

    all_boundaries: List[Boundary] = []
    decisions: List[str] = []
    side_audio: List[Tuple[np.ndarray, np.ndarray]] = []
    for audio_path, cues in files:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        audio = load_audio(audio_path, audio_track=args.audio_track)
        boundaries = build_boundaries(audio, cues, aligner, stem)

        if scorer is not None:
            decisions.extend(speaker_change_decisions(scorer, audio, cues, boundaries, args))
        else:
            if args.debug_dir:
                path = os.path.join(args.debug_dir, stem, "diarization", "result.json")
                with open(path, encoding="utf-8") as fh:
                    result = json.load(fh)
            elif args.scope == "file":
                result = diarizer.process(audio)
            else:
                result = diarizer.process(pseudo_vad_segments(audio, cues, args.merge_gap, args.chunk_size))
            turns = sorted(result["turns"], key=lambda t: (t["start"], t["end"]))
            decisions.extend(scoped_decision(turns, b, args.speaker_confidence) for b in boundaries)
        if args.embeddings:
            for b in boundaries:
                cut = lambda span: audio[int(span[0] * SAMPLE_RATE):int(span[1] * SAMPLE_RATE)]
                side_audio.append((cut(b.left), cut(b.right)))
        all_boundaries.extend(boundaries)
        counts = {k: sum(b.kind == k for b in boundaries) for k in ("change", "same", "adjacent")}
        print(f"{stem}: {counts}", flush=True)

    print("\nPipeline decision ("
          + ("speaker-change model" if scorer is not None
             else "diarization turns -> speaker_assign labels") + " -> merge gate):")
    for line in format_rates(decision_rates(all_boundaries, decisions)):
        print("  " + line)
    adjacent_q = [d for b, d in zip(all_boundaries, decisions) if b.kind == "adjacent" and b.meta.get("question")]
    if adjacent_q:
        print(f"  adjacent after a question: n={len(adjacent_q)}  diff {adjacent_q.count('diff') / len(adjacent_q):6.1%}")

    cosines: List[Optional[float]] = [None] * len(all_boundaries)
    if args.embeddings:
        cosines = embedding_cosines(side_audio, args)
        change = [c for b, c in zip(all_boundaries, cosines) if b.kind == "change"]
        same = [c for b, c in zip(all_boundaries, cosines) if b.kind == "same"]
        if change and same:
            sep = separability(change, same)
            recall = "  ".join(f"recall@{r:.0%} {v:.1%}" for r, v in sep.recall_at.items())
            print(f"\nSide-vs-side embedding cosine: AUC {sep.auc:.3f}  EER {sep.eer:.3f}  {recall}"
                  f"  (n change {sep.n_change}, same {sep.n_same})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for b, d, c in zip(all_boundaries, decisions, cosines):
                fh.write(json.dumps({"kind": b.kind, "source": b.source, "left": b.left, "right": b.right,
                                     "decision": d, "cosine": c, **b.meta}, ensure_ascii=False) + "\n")


def speaker_change_decisions(scorer, audio, cues, boundaries, args) -> List[str]:
    """"diff"/"same" per boundary from the speaker-change model ("unknown" if unscorable).

    Also records the probability on each boundary's meta, so --out carries it.
    """
    from cantocaptions_ai.pipeline.speaker_change import DEFAULT_MODEL, DEFAULT_THRESHOLD, POOL_MIN

    threshold = args.speaker_change_threshold or DEFAULT_THRESHOLD
    vad = pseudo_vad_segments(audio, cues, args.merge_gap, args.chunk_size)
    pool = [(c.start + 0.05, c.end - 0.1) for c in cues
            if c.end - c.start - 0.15 >= POOL_MIN and not _NON_DIALOGUE.match(c.text)]
    features = scorer.features(vad, [(b.left, b.right) for b in boundaries], pool)
    out = []
    for b, f in zip(boundaries, features):
        if f is None:
            out.append("unknown")
            continue
        p = DEFAULT_MODEL.probability(f)
        b.meta["speaker_change"] = round(p, 4)
        out.append("diff" if p >= threshold else "same")
    return out


def embedding_cosines(side_audio, args) -> List[float]:
    """Cosine between the two sides' embeddings, from the diarization pipeline's own model.

    Each side is embedded on its own, unpadded: zero-padding a shorter clip into a batch
    changes what the network's convolutions see even with masked pooling.
    """
    import torch
    from pyannote.audio.core.pipeline import Pipeline

    pipeline = Pipeline.from_pretrained(args.diarize_model, token=args.hf_token)
    model = pipeline._embedding
    model.to(torch.device(args.device))
    out = []
    for left, right in side_audio:
        a = model(torch.from_numpy(left)[None, None])[0]
        b = model(torch.from_numpy(right)[None, None])[0]
        out.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return out


if __name__ == "__main__":
    main()

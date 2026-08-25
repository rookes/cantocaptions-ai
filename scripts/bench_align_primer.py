"""Measure what the align-model audio primer does to first-character timings.

The unit tests for priming (tests/test_align_primer.py) use a fake model, so they can only
prove the *slicing* is exact — they cannot show the primer is needed at all. That is what
this script is for: it runs the real Cantonese alignment model over a saved --debug_dir and
compares alignment with and without the primer.

Background. ``alvanlii/wav2vec2-BERT-cantonese`` was fine-tuned on clips that begin at
speech onset, and reproduces that prior: given audio whose first frames are not speech, it
emits the utterance's first character at emission frame 0 and nowhere else. Forced
alignment then pins each VAD segment's first character to the segment start. Priming with
speech-like left context (see pipeline/align_profiles.py) removes the trigger.

Two numbers are reported per file:

  - **delta**: how far each VAD segment's first cue starts after the segment boundary.
    Unprimed, this collapses to 0.000 for most segments — that is the bug.
  - **silent starts**: what pipeline/align_checks.py finds. Priming should drive this to
    zero; a residue means the primer did not take, and is the post-condition worth watching
    when the primer or the align model changes.

Needs a debug dir holding ``transcription/`` plus ``vocal_isolation/`` (or ``vad/``)
checkpoints — i.e. a previous run with --debug_dir. The alignment model must be locally
cached; run a normal `cantocaptions` invocation with alignment enabled once first.

Usage:
    uv run python scripts/bench_align_primer.py temp/07
    uv run python scripts/bench_align_primer.py temp/07 temp/08 --stage vad
"""

import argparse
import io
import json
import os
import sys

import soundfile as sf

# The report and the align_checks log lines both carry Cantonese text, and this script is
# run straight from a console rather than through utils.log_utils.setup_logging.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from cantocaptions_ai.pipeline.align_checks import find_silent_starts
from cantocaptions_ai.pipeline.align_profiles import AlignProfile, TailPrimer
from cantocaptions_ai.pipeline.alignment import align, load_align_model, load_bert_processor
from cantocaptions_ai.pipeline.model_profiles import get_model_profile


def _load(debug_dir, stage):
    manifest = os.path.join(debug_dir, stage, "segments.json")
    transcription = os.path.join(debug_dir, "transcription", "result.json")
    for path in (manifest, transcription):
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint: {path}")

    metas = json.load(io.open(manifest, encoding="utf-8"))["segments"]
    segments = json.load(io.open(transcription, encoding="utf-8"))["segments"]
    vad_segments = [{
        "start": m["start"],
        "end": m["end"],
        "audio": sf.read(os.path.join(debug_dir, stage, m["file"]), dtype="float32")[0],
    } for m in metas]
    return vad_segments, segments


def _first_cue_deltas(vad_segments, aligned):
    """Seconds between each VAD segment's boundary and its first cue's start."""
    deltas = []
    for segment in vad_segments:
        cues = [c for c in aligned if segment["start"] - 0.02 <= c["start"] < segment["end"]]
        deltas.append(cues[0]["start"] - segment["start"] if cues else float("nan"))
    return deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("debug_dirs", nargs="+", help="debug dirs from a previous --debug_dir run")
    parser.add_argument("--stage", default="vocal_isolation", choices=["vocal_isolation", "vad"],
                        help="which stage's audio to align (default: vocal_isolation)")
    parser.add_argument("--language", default="yue")
    parser.add_argument("--asr-model", default="qwen3-asr",
                        help="ASR model profile supplying spot-checks and punctuation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--primer-seconds", type=float, default=1.0)
    parser.add_argument("--no-reverse", action="store_true", help="prime with the tail unreversed")
    args = parser.parse_args()

    align_model, metadata = load_align_model(args.language, args.device)
    bert_processor = load_bert_processor()
    asr_profile = get_model_profile(args.asr_model)
    primer = TailPrimer(seconds=args.primer_seconds, reverse=not args.no_reverse)

    for debug_dir in args.debug_dirs:
        vad_segments, transcript = _load(debug_dir, args.stage)
        runs = {}
        for label, configured in (("unprimed", None), ("primed", primer)):
            metadata = {**metadata, "profile": AlignProfile(primer=configured)}
            aligned = align(
                transcript, align_model, metadata, vad_segments, args.device,
                bert_processor=bert_processor,
                spotchecks=asr_profile.spotchecks, punctuation=asr_profile.punctuation,
            )["segments"]
            runs[label] = (_first_cue_deltas(vad_segments, aligned),
                           find_silent_starts(aligned, vad_segments))

        print(f"\n=== {debug_dir} ({len(vad_segments)} VAD segments, {args.stage} audio) ===")
        print(f"{'seg':<5}{'vad start':>11}{'unprimed':>11}{'primed':>11}   first cue")
        unprimed, primed = runs["unprimed"][0], runs["primed"][0]
        for i, (segment, a, b) in enumerate(zip(vad_segments, unprimed, primed)):
            flag = "  <-- pinned" if a < 0.001 else ""
            print(f"{i:<5}{segment['start']:>11.3f}{a:>11.3f}{b:>11.3f}{flag}")

        pinned = sum(1 for d in unprimed if d < 0.001)
        print(f"\n  first cue on the VAD boundary: {pinned}/{len(unprimed)} unprimed"
              f", {sum(1 for d in primed if d < 0.001)}/{len(primed)} primed")
        print("  (a zero delta is only wrong when the audio there is silent — which is what"
              " the next lines measure;\n   a segment whose VAD region opens on speech"
              " legitimately starts at 0.000)")
        for label in ("unprimed", "primed"):
            hits = runs[label][1]
            worst = f", worst {max(h.gap for h in hits):.2f}s" if hits else ""
            print(f"  align_checks silent starts, {label:<8}: {len(hits)}{worst}")


if __name__ == "__main__":
    main()

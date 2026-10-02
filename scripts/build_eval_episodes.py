"""Stitch cantocaptions-dataset clips into test episodes with a ground-truth SRT.

The dataset keeps per-segment clips (``data/wav/<show>/<segment_id>.wav``) whose cues are
known exactly, but not the full source media. Concatenating a show's clips with a short
silence between them gives an "episode" the whole pipeline can run on -- VAD, ASR,
alignment, cue assembly, cleaning -- and a reference SRT whose timings are exact by
construction: every cue sits at its clip's offset plus the cue's own offset inside the clip.

Uses the held-out ``test`` split by default, so a fine-tuned model is not scored on audio it
trained on, plus a few chosen shows from the other splits (``INCLUDE``), which can be.

    uv run python scripts/build_eval_episodes.py --out ../cantocaptions-eval/episodes
    # -> episodes/<show>.wav (16 kHz mono) and episodes/gt/<show>.srt
"""
import argparse
import json
import os
import wave
from collections import defaultdict

import numpy as np

SR = 16000


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_wav(path):
    with wave.open(path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR):
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def _ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# Test-split shows whose subtitles are not a transcript of what is said, so a model can't be
# scored against them. Echoes of the Rainbow's carry English lines with Chinese translations,
# glosses and pronunciations in brackets, romanised syllables, and spelling the dataset's
# standardization report rates nonstandard (0.23).
NOT_VERBATIM = ["echoes-of-the-rainbow-2010"]

# Shows from outside the split that are added to it, for more shows than the test split has.
# Both have verbatim subtitles (standardization 0.98 and 1.00). They are not held out:
# ReZero is in dev, used to pick checkpoints, and War of the Genders is in train, so
# rookes/cantocaptions-cantonese-asr has been fine-tuned on its audio and scores better on
# it than on unseen speech. Its timing and segmentation scores aren't affected.
INCLUDE = ["re-zero-starting-life-in-another-world-2016", "war-of-the-genders-2000"]


def build_show(data_dir, show, max_seconds, gap):
    """Return (pcm int16 array, [(start, end, text)]) for one show's clips in time order."""
    episodes = sorted(
        f[:-len(".jsonl")] for f in os.listdir(os.path.join(data_dir, "segments"))
        if f.startswith(show + "_") and f.endswith(".jsonl")
    )
    silence = np.zeros(int(gap * SR), dtype="<i2")
    pieces, cues, t = [], [], 0.0
    for episode in episodes:
        segments = _read_jsonl(os.path.join(data_dir, "segments", episode + ".jsonl"))
        cue_rows = {c["index"]: c for c in _read_jsonl(os.path.join(data_dir, "cues", episode + ".jsonl"))}
        for seg in segments:
            wav_path = os.path.join(data_dir, "wav", show, seg["segment_id"] + ".wav")
            if not os.path.isfile(wav_path):
                continue  # dropped or not cut
            pcm = _read_wav(wav_path)
            lo, hi = seg["cue_span"]
            for idx in range(lo, hi + 1):
                cue = cue_rows.get(idx)
                if cue is None:
                    continue
                start = t + max(cue["start"] - seg["start"], 0.0)
                end = t + min(cue["end"] - seg["start"], len(pcm) / SR)
                cues.append((start, end, cue["text"]))
            pieces += [pcm, silence]
            t += len(pcm) / SR + gap
            if max_seconds and t >= max_seconds:
                return np.concatenate(pieces), cues
    return (np.concatenate(pieces) if pieces else np.zeros(0, dtype="<i2")), cues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../cantocaptions-dataset/data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--minutes-per-show", type=float, default=10.0, help="0 = every clip")
    ap.add_argument("--gap", type=float, default=1.0, help="seconds of silence between clips")
    ap.add_argument("--exclude", nargs="*", default=NOT_VERBATIM, metavar="SHOW",
                    help="shows to leave out (default: those whose subtitles aren't verbatim)")
    ap.add_argument("--include", nargs="*", default=INCLUDE, metavar="SHOW",
                    help="shows from other splits to add (default: %(default)s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.dataset, "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)
    unknown = sorted(set(args.include) - set(splits))
    if unknown:
        ap.error(f"not in the dataset: {', '.join(unknown)}")
    shows = sorted(s for s, v in splits.items()
                   if (v["split"] == args.split or s in args.include) and s not in args.exclude)
    os.makedirs(os.path.join(args.out, "gt"), exist_ok=True)
    manifest = defaultdict(dict)
    for show in shows:
        pcm, cues = build_show(args.dataset, show, args.minutes_per_show * 60, args.gap)
        if not cues:
            print(f"{show}: no clips found, skipped")
            continue
        with wave.open(os.path.join(args.out, show + ".wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        with open(os.path.join(args.out, "gt", show + ".srt"), "w", encoding="utf-8") as f:
            for i, (start, end, text) in enumerate(cues, 1):
                f.write(f"{i}\n{_ts(start)} --> {_ts(end)}\n{text}\n\n")
        manifest[show] = {"seconds": round(len(pcm) / SR, 1), "cues": len(cues)}
        print(f"{show}: {len(pcm) / SR / 60:.1f} min, {len(cues)} cues")
    with open(os.path.join(args.out, "gt", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()

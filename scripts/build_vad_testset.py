"""Build the VAD test set from cantocaptions-dataset: long audio streams with known speech.

A VAD is judged on whole files, but the dataset holds only per-segment clips (the episode
media is not part of it). This rebuilds file-like audio from what is there: every clip of
one episode, in episode order, butted end to end into one stream, with that episode's
non-speech ``gaps`` clips interleaved. Each clip carries its own head and tail padding, so
every join falls in quiet audio; a 5 ms fade either side removes the click a hard join
would put there. The detector then sees minutes of continuous programme audio -- which
matters, because Silero's recurrent state carries minutes of memory and pyannote scores
10 s windows -- rather than a thousand isolated 15 s clips.

Ground truth, on each stream's timeline:

* ``speech``    every KEPT cue overlapping a clip, clipped to the clip. Cue timing in this
                corpus is accurate at the onset and carries some padding at the end, which
                the metrics allow for with a wider end collar (see utils/vad_eval.py).
* ``dontcare``  DROPPED cues that are audible speech nobody transcribed (other_language,
                unintelligible, lyrics). Scored neither way: finding them is not a false
                alarm and missing them is not a miss.
* ``gap`` clips pure-negative audio: nothing is marked in it, so anything detected there is
                a false alarm (music, effects, room tone). Caveat: the dataset mined these
                clips away from anything pyannote heard at onset 0.70 (dataset coverage.py),
                so they flatter pyannote; read that number per backend with this in mind.

Segments the dataset's own forced-alignment check rejected (align_fail / low_score /
low_coverage -- exactly what `export` drops) are left out: their text does not fit their
audio, so their cues do not mark where the speech is.

    uv run python scripts/build_vad_testset.py                      # dev + test splits
    uv run python scripts/build_vad_testset.py --split dev --out /tmp/vad

Writes ``<out>/<split>/<episode_id>.wav`` (16 kHz PCM16) and ``<out>/<split>/manifest.json``.
Deterministic: the same dataset state rebuilds the same bytes.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(os.path.dirname(REPO_ROOT), "cantocaptions-dataset")
DEFAULT_OUT = os.path.join(REPO_ROOT, ".cache", "vad_testset")

SAMPLE_RATE = 16000
FADE_S = 0.005
# verify flags that mean the text does not fit the audio; export drops these too.
REJECT_FLAGS = {"align_fail", "low_score", "low_coverage"}
# Dropped-cue reasons that are nonetheless someone audibly speaking or singing.
DONTCARE_REASONS = {"other_language", "unintelligible", "lyrics"}
# A clip whose wav disagrees with its segment span by more than this was cut from a
# different segmentation than the one on disk; its cue times would not line up.
MAX_LEN_MISMATCH_S = 0.01


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def overlapping(items, start, end):
    return [c for c in items if c["start"] < end and c["end"] > start]


def fade(audio):
    n = min(int(FADE_S * SAMPLE_RATE), len(audio) // 2)
    if n:
        ramp = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def dataset_revision(dataset):
    try:
        return subprocess.run(["git", "-C", dataset, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_episode(dataset, rec, gap_files, skipped):
    """One stream: this episode's usable clips in time order, gap clips spread among them."""
    eid, show = rec["episode_id"], rec["show_slug"]
    segments = read_jsonl(os.path.join(dataset, "data", "segments", f"{eid}.jsonl"))
    cues = read_jsonl(os.path.join(dataset, "data", "cues", f"{eid}.jsonl"))
    dropped = read_jsonl(os.path.join(dataset, "data", "cues", f"{eid}.dropped.jsonl"))
    verdicts = {r["segment_id"]: r for r in read_jsonl(
        os.path.join(dataset, "data", "verify", f"{eid}.jsonl"))}

    clips = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        wav = os.path.join(dataset, "data", "wav", show, f"{seg['segment_id']}.wav")
        if not os.path.isfile(wav):
            skipped["no_wav"] += 1
            continue
        verdict = verdicts.get(seg["segment_id"])
        if verdict is None:
            skipped["unverified"] += 1
            continue
        if REJECT_FLAGS & set(verdict["flags"]):
            skipped["verify_rejected"] += 1
            continue
        clips.append(("speech", seg, wav))
    if not clips:
        return None

    # Spread the gap clips evenly through the episode rather than bunching them at one end,
    # so each one sits in the same acoustic surroundings a real inter-scene gap would.
    for k, wav in enumerate(gap_files):
        at = round((k + 1) * len(clips) / (len(gap_files) + 1)) + k
        clips.insert(at, ("gap", None, wav))

    pieces, clip_rows, speech, dontcare = [], [], [], []
    offset = 0
    for kind, seg, wav in clips:
        audio, sr = sf.read(wav, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{wav}: expected {SAMPLE_RATE} Hz, got {sr}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / SAMPLE_RATE
        t0 = offset / SAMPLE_RATE
        if kind == "speech":
            if abs(dur - (seg["end"] - seg["start"])) > MAX_LEN_MISMATCH_S:
                skipped["length_mismatch"] += 1
                continue
            a, b = seg["start"], seg["start"] + dur

            def place(item):
                return [round(t0 + max(item["start"], a) - a, 4),
                        round(t0 + min(item["end"], b) - a, 4)]

            for cue in overlapping(cues, a, b):
                speech.append(place(cue) + [len(clip_rows), cue["index"]])
            for d in overlapping(dropped, a, b):
                if d["reason"] in DONTCARE_REASONS:
                    dontcare.append(place(d))
            clip_rows.append({"kind": "speech", "id": seg["segment_id"], "start": round(t0, 4),
                              "end": round(t0 + dur, 4), "source_start": seg["start"]})
        else:
            clip_rows.append({"kind": "gap", "id": os.path.basename(wav)[:-4],
                              "start": round(t0, 4), "end": round(t0 + dur, 4)})
        pieces.append(fade(audio.copy()))
        offset += len(audio)

    return {
        "episode_id": eid,
        "show_slug": show,
        "audio": np.concatenate(pieces),
        "clips": clip_rows,
        # [start, end, clip_index, cue_index]: the cue index identifies the subtitle line,
        # so a cue split across two clips is still one line to the cue-level metrics.
        "speech": speech,
        "dontcare": dontcare,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="cantocaptions-dataset checkout")
    ap.add_argument("--split", nargs="+", default=["dev", "test"],
                    help="dataset splits to build (the split is by show)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    args = ap.parse_args()

    data = os.path.join(args.dataset, "data")
    with open(os.path.join(data, "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)
    with open(os.path.join(data, "manifest.json"), encoding="utf-8") as f:
        records = json.load(f)["records"]
    gap_rows = read_jsonl(os.path.join(data, "gaps", "gaps.jsonl"))
    gap_paths = sorted(os.path.join(args.dataset, r["audio"]) for r in gap_rows)

    for split in args.split:
        shows = {s: v for s, v in splits.items() if v["split"] == split}
        if not shows:
            sys.exit(f"no shows in split {split!r}")
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        skipped = Counter()
        streams = []
        for rec in sorted(records, key=lambda r: r["episode_id"]):
            if rec["show_slug"] not in shows or rec.get("exclusion") is not None:
                continue
            eid = rec["episode_id"]
            gaps = [p for p in gap_paths
                    if os.path.basename(p).startswith(f"{eid}_gap") and os.path.isfile(p)]
            built = build_episode(args.dataset, rec, gaps, skipped)
            if built is None:
                continue
            audio = built.pop("audio")
            sf.write(os.path.join(out_dir, f"{eid}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
            built.update(audio=f"{eid}.wav", duration=round(len(audio) / SAMPLE_RATE, 4),
                         stratum=shows[rec["show_slug"]]["stratum"])
            streams.append(built)
            n_speech = sum(c["kind"] == "speech" for c in built["clips"])
            print(f"  {eid:55s} {built['duration'] / 60:6.1f} min  {n_speech:4d} clips  "
                  f"{len(gaps):2d} gaps  {len({s[3] for s in built['speech']}):4d} cues")

        manifest = {
            "built": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": os.path.abspath(args.dataset),
            "dataset_revision": dataset_revision(args.dataset),
            "split": split,
            "sample_rate": SAMPLE_RATE,
            "skipped_segments": dict(skipped),
            "streams": streams,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        hours = sum(s["duration"] for s in streams) / 3600
        gap_h = sum(c["end"] - c["start"] for s in streams for c in s["clips"]
                    if c["kind"] == "gap") / 3600
        print(f"{split}: {len(streams)} streams, {hours:.2f} h ({gap_h:.2f} h gap clips), "
              f"skipped {dict(skipped)} -> {out_dir}")


if __name__ == "__main__":
    main()

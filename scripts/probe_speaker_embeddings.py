"""Look for recurring voices: embed every reference cue across files and cluster them together.

A speaker embedding is only useful for attribution if one voice lands in the same place every
time it speaks -- across a scene, an episode, a series. This embeds each one-speaker cue of
one or more files (an audio file plus its subtitles each), clusters all of them together, and
prints every sizeable cluster with how many files it spans, how tight it is, and sample lines,
so the clusters can be named by reading what they say.

    # a series: do the main characters come out as clusters spanning every episode?
    python scripts/probe_speaker_embeddings.py --pair ep01.mkv ep01.srt --pair ep02.mkv ep02.srt ...

    # a lower bar (merge more): --similarity 0.3; only cues of 2 s or more: --min_duration 2

What to expect (see docs/diarization.md): with cues of about 1.5 s or more, a long-running
cartoon's main cast comes out as clusters that each span nearly every episode. At sub-second
length, the same voice scatters. Theme songs form their own clusters.

Hyphen-convention cues (two speakers in one cue) and bracketed non-dialogue cues are skipped.
--embedding_model picks the network: the diarization pipeline's own (default) or any
pyannote-loadable embedding checkpoint.
"""

import argparse
import collections
import json
import os
import sys
import warnings

import numpy as np

# pyannote averages over empty frame sets on very short segments; the result is unused.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cantocaptions_ai.utils.audio import SAMPLE_RATE, load_audio  # noqa: E402
from cantocaptions_ai.utils.speaker_eval import hyphen_turns  # noqa: E402
from cantocaptions_ai.utils.subtitles import read_subtitle_cues  # noqa: E402

_NON_DIALOGUE = ("[", "［", "(", "（", "【", "♪")


def load_embedder(model_name: str, device: str, token):
    import torch

    if model_name.startswith("pyannote/speaker-diarization"):
        from pyannote.audio.core.pipeline import Pipeline
        model = Pipeline.from_pretrained(model_name, token=token)._embedding
    else:
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        model = PretrainedSpeakerEmbedding(model_name, token=token)
    model.to(torch.device(device))

    def embed(chunk: np.ndarray) -> np.ndarray:
        # One at a time and unpadded: padding shifts the result (see eval_diarization.py).
        return model(torch.from_numpy(chunk)[None, None])[0]

    return embed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("AUDIO", "SUBTITLES"))
    parser.add_argument("--min_duration", type=float, default=1.5, help="shortest cue to embed (s)")
    parser.add_argument("--similarity", type=float, default=0.4,
                        help="average-linkage cosine similarity at which clusters stop merging")
    parser.add_argument("--min_cluster", type=int, default=20, help="smallest cluster to print")
    parser.add_argument("--samples", type=int, default=6, help="lines printed per cluster")
    parser.add_argument("--embedding_model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--audio_track", type=int, default=0)
    parser.add_argument("--out", help="write every cue's file, time, text and cluster as JSONL")
    args = parser.parse_args(argv)

    from scipy.cluster.hierarchy import fcluster, linkage

    embed = load_embedder(args.embedding_model, args.device, args.hf_token)
    rows, vectors = [], []
    for audio_path, subtitle_path in args.pair:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        audio = load_audio(audio_path, audio_track=args.audio_track)
        kept = 0
        for cue in read_subtitle_cues(subtitle_path):
            # Trim the onset slack and the padded tail subtitles usually carry.
            start, end = cue.start + 0.05, cue.end - 0.1
            if end - start < args.min_duration or cue.text.lstrip().startswith(_NON_DIALOGUE):
                continue
            if hyphen_turns(cue.text):
                continue
            chunk = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            if len(chunk) < args.min_duration * SAMPLE_RATE:
                continue
            vectors.append(embed(chunk))
            rows.append({"file": stem, "start": round(start, 3), "end": round(end, 3),
                         "text": cue.text.replace("\n", " ")})
            kept += 1
        print(f"{stem}: {kept} cues embedded", flush=True)

    if len(rows) < 2:
        sys.exit("Not enough cues to cluster.")
    X = np.asarray(vectors, dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = fcluster(linkage(X, method="average", metric="cosine"), 1 - args.similarity, criterion="distance")
    n_files = len({r["file"] for r in rows})
    sizes = collections.Counter(labels)
    big = [k for k, n in sizes.most_common() if n >= args.min_cluster]
    print(f"\n{len(rows)} cues from {n_files} file(s): {len(sizes)} clusters at similarity "
          f"{args.similarity}; {len(big)} with >= {args.min_cluster} cues, holding "
          f"{sum(sizes[k] for k in big) / len(rows):.0%} of all cues\n")
    for k in big:
        members = np.flatnonzero(labels == k)
        centroid = X[members].mean(0)
        centroid /= np.linalg.norm(centroid)
        files = collections.Counter(rows[i]["file"] for i in members)
        cohesion = float(np.mean(X[members] @ centroid))
        print(f"cluster {k}: {len(members)} cues, in {len(files)}/{n_files} files "
              f"(largest share {files.most_common(1)[0][1] / len(members):.0%}), "
              f"mean cosine to centroid {cohesion:.2f}")
        closest = members[np.argsort(-(X[members] @ centroid))[: args.samples]]
        for i in closest:
            print(f"    {rows[i]['file']} {rows[i]['start']:8.1f}s  {rows[i]['text']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for row, label in zip(rows, labels):
                fh.write(json.dumps({**row, "cluster": int(label)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

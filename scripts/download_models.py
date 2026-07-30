#!/usr/bin/env python
"""Pre-fetch model weights into the Hugging Face cache.

Run this at container-build time (or once on a fresh persistent volume) so the
first real transcription job doesn't pay the multi-GB download tax mid-request.
The pyannote VAD segmentation weights ship vendored in ``cantocaptions_ai/assets``,
so they are NOT fetched here — only the models the pipeline downloads from HF.

Usage:
    python scripts/download_models.py                 # always-on models (ASR + alignment)
    python scripts/download_models.py --full          # + roformer, ensemble, LLM, diarization
    python scripts/download_models.py --hf-token hf_xxx --cache-dir /models

The gated pyannote diarization model requires accepting its terms on HF and a
token (``--hf-token`` or the HF_TOKEN env var); it is only fetched with ``--full``.
"""
import argparse
import os
import sys


# Source-of-truth for these ids lives in the pipeline modules named in each comment;
# kept as literals here so this script imports no torch-heavy pipeline code.
ALWAYS_REPOS = [
    "alvanlii/wav2vec2-BERT-cantonese",   # alignment.py: align model + bert processor
]

# ASR model repo ids are read from the profile registry (model_profiles.py), which
# is a light import (no torch). Local-path profiles (e.g. a LoRA dir) are skipped.
FULL_REPOS = [
    "alvanlii/whisper-small-cantonese",   # ensemble.py
    "Qwen/Qwen3-4B",                      # config.py: default llm_model
    "pyannote/speaker-diarization-community-1",  # config.py: default diarize_model (GATED)
]

# (repo_id, filename) single-file downloads.
FULL_FILES = [
    ("KimberleyJSN/melbandroformer", "MelBandRoformer.ckpt"),  # vocal_isolation.py
]


def _asr_repos() -> list:
    from cantocaptions_ai.pipeline.model_profiles import MODEL_PROFILES
    repos = []
    for profile in MODEL_PROFILES.values():
        hf_id = profile.hf_id
        # Skip local directories / non-hub paths (e.g. a merged LoRA checkpoint).
        if os.path.sep in hf_id or (":" in hf_id and not hf_id.count("/") == 1) or os.path.isdir(hf_id):
            continue
        repos.append(hf_id)
    return repos


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-fetch cantocaptions-ai model weights.")
    ap.add_argument("--full", action="store_true", help="also fetch roformer, ensemble, LLM, and diarization models")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HF token for gated models (default: HF_TOKEN env)")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"), help="HF cache dir (sets HF_HOME; default: HF_HOME env or ~/.cache/huggingface)")
    args = ap.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    from huggingface_hub import hf_hub_download, snapshot_download
    token = args.hf_token or None

    repos = list(_asr_repos()) + list(ALWAYS_REPOS)
    files = []
    if args.full:
        repos += FULL_REPOS
        files += FULL_FILES

    failed = []
    for repo in repos:
        print(f"[download] snapshot: {repo}", flush=True)
        try:
            snapshot_download(repo_id=repo, token=token)
        except Exception as e:  # noqa: BLE001 — report and continue so one gated/missing repo doesn't abort the build
            print(f"[warn] failed {repo}: {e}", file=sys.stderr, flush=True)
            failed.append(repo)

    for repo, filename in files:
        print(f"[download] file: {repo}/{filename}", flush=True)
        try:
            hf_hub_download(repo_id=repo, filename=filename, token=token)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed {repo}/{filename}: {e}", file=sys.stderr, flush=True)
            failed.append(f"{repo}/{filename}")

    if failed:
        print(f"\nCompleted with {len(failed)} failure(s): {', '.join(failed)}", file=sys.stderr)
        print("Gated models (e.g. pyannote) need terms accepted + a valid --hf-token.", file=sys.stderr)
        return 1
    print("\nAll requested models fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _default_model_name() -> str:
    """The ``model`` a plain run uses, read off PipelineConfig without calling
    ``defaults()`` -- that resolves the ``device`` default_factory, which imports torch.
    It is equal to config/default.cfg's own value by test (test_cli_config.py).
    """
    from dataclasses import fields
    from cantocaptions_ai.pipeline.config import PipelineConfig

    return next(f.default for f in fields(PipelineConfig) if f.name == "model")


def _is_hub_id(hf_id: str) -> bool:
    """True for a fetchable ``org/name``, False for a local directory.

    Do NOT test ``os.path.sep in hf_id`` here -- it is "/" on Linux, which is where this
    script actually runs, so that skipped every hub id on the one platform that matters
    and pre-fetched no ASR weights at all.
    """
    return (
        "\\" not in hf_id
        and ":" not in hf_id
        and hf_id.count("/") == 1
        and not os.path.isdir(hf_id)
    )


def _asr_repos(model_name=None, every=False) -> list:
    """Repos for the ASR model(s) to fetch.

    Only ONE ASR model is fetched by default -- the one a plain run loads. Fetching
    every registered profile means ~8 GB of checkpoints a user will never load, which
    is the opposite of what a prefetch step is for. ``every=True`` restores that for a
    build that wants to serve any --model without a cold download.
    """
    from cantocaptions_ai.pipeline.model_profiles import MODEL_PROFILES, get_model_profile

    if every:
        wanted = [p.hf_id for p in MODEL_PROFILES.values()]
    else:
        wanted = [get_model_profile(model_name or _default_model_name()).hf_id]

    seen, repos = set(), []
    for hf_id in wanted:
        if _is_hub_id(hf_id) and hf_id not in seen:
            seen.add(hf_id)
            repos.append(hf_id)
    return repos


def _xet_status() -> str:
    """One line on whether Xet-backed downloads are active.

    hf-xet needs no setup: huggingface_hub depends on it unconditionally (gated on CPU
    architecture, not on an extra), so a plain install already has it. This exists for the
    cases where it is genuinely absent -- an architecture with no hf-xet wheel, or
    HF_HUB_DISABLE_XET set -- so "downloads are slow" is diagnosable rather than a mystery.
    """
    import importlib.util

    if importlib.util.find_spec("hf_xet") is None:
        return "Xet: OFF (hf-xet not installed -- no wheel for this architecture?); downloads use plain HTTP"

    try:
        from huggingface_hub import constants

        if getattr(constants, "HF_HUB_DISABLE_XET", False):
            return "Xet: OFF (HF_HUB_DISABLE_XET is set); downloads use plain HTTP"
        high_perf = getattr(constants, "HF_XET_HIGH_PERFORMANCE", False)
    except Exception:  # noqa: BLE001 -- a status line must never break the download
        high_perf = False

    extra = "" if high_perf else " (set HF_XET_HIGH_PERFORMANCE=1 to trade RAM/CPU for more speed)"
    return f"Xet: on{extra}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-fetch cantocaptions-ai model weights.")
    ap.add_argument("--full", action="store_true", help="also fetch roformer, ensemble, LLM, and diarization models")
    ap.add_argument("--model", default=None, help="ASR model name or hub id to fetch (default: whatever a plain run loads)")
    ap.add_argument("--all-asr", action="store_true", help="fetch every registered ASR model, not just the default one")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HF token for gated models (default: HF_TOKEN env)")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"), help="HF cache dir (sets HF_HOME; default: HF_HOME env or ~/.cache/huggingface)")
    args = ap.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    from huggingface_hub import hf_hub_download, snapshot_download
    token = args.hf_token or None

    print(f"[info] {_xet_status()}", flush=True)

    repos = list(_asr_repos(args.model, every=args.all_asr)) + list(ALWAYS_REPOS)
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

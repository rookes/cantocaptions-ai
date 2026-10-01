"""The ASR model registry: what each named model *is*, independent of language.

A ``ModelProfile`` says where a model's weights live (``hf_id``, a hub repo id or local
path), which ASR backend runs it (``backend``; see ``pipeline/asr.py``) and which languages
it can transcribe (``languages``; None for a multilingual model such as stock Qwen3-ASR). Everything about how a model *writes* a particular language --
OpenCC normalization, particle spot-checks, cue markers, which cleaning manifest -- belongs
to that language and lives in its pack (``languages/<code>``, ``LanguagePack.conventions``),
so stock Qwen3-ASR can carry Cantonese conventions under yue and none elsewhere.

``get_model_profile(None, language)`` resolves the language pack's default model, so a
config that leaves ``model`` unset gets that language's own model. An unregistered name is
treated as a raw hub id or path, whose backend is read off its own ``config.json``.

This module is a light import (no torch): the model download script and the worker's
preflight checks read it.
"""
import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Optional


@dataclass(frozen=True)
class ModelProfile:
    """One ASR model: where its weights are, what runs it, which languages it transcribes."""
    hf_id: str
    # None: multilingual (or unknown) -- no language is refused. A set: the model was
    # trained for those languages only, and validate_config refuses it for any other.
    languages: Optional[FrozenSet[str]] = None
    # A key of pipeline.asr.ASR_BACKENDS; None: identified from the checkpoint's config.
    backend: Optional[str] = None


# Env var pointing at a *local* merged-weights directory. Kept out of the source tree so no
# machine-specific path ships in git: the "Qwen3-ASR-lora" profile is only registered (and
# only offered as a --model choice) when this is set. It is the personal/debugging escape
# hatch for an unpublished build — the published checkpoint is "cantocaptions-cantonese-ASR",
# which is pulled from the hub and needs no env var. See _build_profiles.
_LORA_MODEL_DIR_ENV = "CANTOCAPTIONS_LORA_MODEL_DIR"

# The published CantoCaptions fine-tune, on the hub like any other model.
_CANTOCAPTIONS_HF_ID = "rookes/cantocaptions-cantonese-asr"

_CANTONESE_ONLY = frozenset({"yue"})
_QWEN = "qwen3-asr"


def _build_profiles() -> Dict[str, ModelProfile]:
    profiles: Dict[str, ModelProfile] = {
        "Qwen3-ASR": ModelProfile("Qwen/Qwen3-ASR-1.7B-hf", backend=_QWEN),
        "Qwen3-ASR-0.6B": ModelProfile("Qwen/Qwen3-ASR-0.6B-hf", backend=_QWEN),
        "cantocaptions-cantonese-ASR": ModelProfile(_CANTOCAPTIONS_HF_ID, _CANTONESE_ONLY, _QWEN),
    }
    # The same fine-tune, pointed at a local directory instead. Registered only when the
    # env var is set; otherwise --model Qwen3-ASR-lora is simply not a valid choice (clean
    # argparse error) rather than a broken hardcoded path.
    lora_dir = os.environ.get(_LORA_MODEL_DIR_ENV)
    if lora_dir:
        profiles["Qwen3-ASR-lora"] = ModelProfile(lora_dir, _CANTONESE_ONLY, _QWEN)
    return profiles


MODEL_PROFILES: Mapping[str, ModelProfile] = _build_profiles()


def resolve_model_name(name: Optional[str], language: Optional[str] = None) -> Optional[str]:
    """The model a run uses: *name*, or the language pack's default when it is None."""
    if name:
        return name
    from cantocaptions_ai.languages import get_language_pack
    return get_language_pack(language).default_model


def get_model_profile(name: Optional[str], language: Optional[str] = None) -> ModelProfile:
    """Return the profile for model *name*.

    None means "the default model for *language*" (yue when *language* is None too).
    An unregistered name is treated as a raw hub id or path.
    """
    resolved = resolve_model_name(name, language)
    if resolved is None:
        raise ValueError(f"language '{language}' has no default ASR model; name one with --model")
    return MODEL_PROFILES.get(resolved) or ModelProfile(hf_id=resolved)

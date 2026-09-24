"""Per-model downstream configuration registry.

A ``ModelProfile`` ties an ASR model name to the downstream behavior that depends on
how that particular model writes its output: post-ASR text normalization (OpenCC /
HK-variant rewriting), the punctuation set used for sentence splitting, the alignment
"spot checks" that swap interchangeable particles, and which text-cleaning manifest the
finished subtitles are folded through.

Every field defaults to a no-op, so:
  * a model **not** in ``MODEL_PROFILES`` (or a new entry that omits fields) runs no
    OpenCC, uses the default punctuation, performs no spot checks and cleans through
    the conservative default manifest — the right starting point for a freshly
    fine-tuned model that already outputs the target convention;
  * adding a model means adding one ``MODEL_PROFILES`` entry — no edits to the ASR or
    alignment code.

This replaces the former ``_MODEL_IDS`` dict in ``_asr_native.py``. The value objects
themselves (``TextNormalization``, ``PunctuationConfig``, ``SpotCheck``,
``CleaningConfig``) live in
``cantonese/text.py`` so that module stays free of any ``pipeline`` import.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping

from cantocaptions_ai.cantonese.text import (
    DEFAULT_CLEANING,
    DEFAULT_PUNCTUATION,
    DEFAULT_SEGMENTATION,
    CleaningConfig,
    PunctuationConfig,
    SegmentationConfig,
    SpotCheck,
    TextNormalization,
)


@dataclass(frozen=True)
class ModelProfile:
    """Everything downstream stages need to know about one ASR model's output.

    ``hf_id`` is the native-backend repo id or local path to load. The remaining fields
    all default to no-ops (see module docstring).
    """
    hf_id: str
    normalization: TextNormalization = TextNormalization()
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION
    spotchecks: Mapping[str, SpotCheck] = field(default_factory=dict)
    segmentation: SegmentationConfig = DEFAULT_SEGMENTATION
    cleaning: CleaningConfig = DEFAULT_CLEANING


# Vanilla Qwen3-ASR outputs Simplified characters and generic particles, so it needs the
# full OpenCC s2t + HK-variant normalization and the particle spot-check table (migrated
# verbatim from the former text.QWEN_PARTICLE_MAP, with the 咁->噉 acoustic bias that
# used to be hard-coded in alignment._align_segment now expressed as a candidate weight).
_QWEN_NORMALIZATION = TextNormalization(opencc_config="s2t_c.json", chars_hk=True)

_QWEN_SPOTCHECKS: Mapping[str, SpotCheck] = {
    "啊": SpotCheck(("吖", "呀")),
    "吖": SpotCheck(("吖", "呀")),
    "呀": SpotCheck(("吖", "呀")),
    "咯": SpotCheck(("喇", "啦", "囉")),
    "喇": SpotCheck(("喇", "啦")),
    "啦": SpotCheck(("喇", "啦")),
    "咋": SpotCheck(("咋", "啫")),
    "啫": SpotCheck(("咋", "啫")),
    "咁": SpotCheck(("咁", "噉"), weights={"噉": 0.8}),
}

# Qwen's raw output is Mandarin-flavoured and Simplified-derived, with generic final
# particles and Chinese numerals written out, so it needs the full legacy cleaning chain
# rather than the conservative default manifest a fine-tuned checkpoint gets.
_QWEN_CLEANING = CleaningConfig(manifest="pipeline_qwen.toml")

# Both Qwen and the fine-tunes built on it punctuate leading discourse markers off as their
# own clause ("嗱，你知啦，" -> "嗱，" + "你知啦，"), so alignment gives them a standalone
# subsegment that CTC then squeezes to a few frames. Listing them here rejoins them onto the
# sentence they introduce.
# Deliberately excludes 呀/啦/吓: those double as final particles, so they attach backwards
# about as often as forwards and are better left to the generic duration-based rescue.
_LEADING_MARKER_SEGMENTATION = SegmentationConfig(leading_markers=(
    "嗱", "喂", "咦", "哦", "唉", "誒", "哎",
    "哎呀", "哎吔", "哇", "嚇", "嗯", "好啦",
))


# The published fine-tune and the local LoRA build below share one profile shape: both
# already emit HK-traditional text and the CantoCaptions final-particle convention, so they
# need no OpenCC, default punctuation, no spot checks, and the conservative default cleaning
# manifest rather than Qwen's full chain. Cue assembly is the exception: they still clause
# off leading markers the way Qwen does, so they take Qwen's marker list too. Written once
# so the two cannot drift apart; copy it as the template when adding another fine-tuned
# checkpoint.
def _finetuned_profile(hf_id: str) -> ModelProfile:
    return ModelProfile(hf_id=hf_id, segmentation=_LEADING_MARKER_SEGMENTATION)


# Env var pointing at a *local* merged-weights directory. Kept out of the source tree so no
# machine-specific path ships in git: the "Qwen3-ASR-lora" profile is only registered (and
# only offered as a --model choice) when this is set. It is the personal/debugging escape
# hatch for an unpublished build — the published checkpoint is "cantocaptions-cantonese-ASR",
# which is pulled from the hub and needs no env var. See _build_profiles.
_LORA_MODEL_DIR_ENV = "CANTOCAPTIONS_LORA_MODEL_DIR"

# The published CantoCaptions fine-tune, on the hub like any other model.
_CANTOCAPTIONS_HF_ID = "rookes/cantocaptions-cantonese-asr"


def _build_profiles() -> Dict[str, ModelProfile]:
    profiles: Dict[str, ModelProfile] = {
        "Qwen3-ASR": ModelProfile(
            hf_id="Qwen/Qwen3-ASR-1.7B-hf",
            normalization=_QWEN_NORMALIZATION,
            spotchecks=_QWEN_SPOTCHECKS,
            segmentation=_LEADING_MARKER_SEGMENTATION,
            cleaning=_QWEN_CLEANING,
        ),
        "Qwen3-ASR-0.6B": ModelProfile(
            hf_id="Qwen/Qwen3-ASR-0.6B-hf",
            normalization=_QWEN_NORMALIZATION,
            spotchecks=_QWEN_SPOTCHECKS,
            segmentation=_LEADING_MARKER_SEGMENTATION,
            cleaning=_QWEN_CLEANING,
        ),
        "cantocaptions-cantonese-ASR": _finetuned_profile(_CANTOCAPTIONS_HF_ID),
    }
    # Same profile as the published fine-tune above, pointed at a local directory instead.
    # Registered only when the env var is set; otherwise --model Qwen3-ASR-lora is simply not
    # a valid choice (clean argparse error) rather than a broken hardcoded path.
    lora_dir = os.environ.get(_LORA_MODEL_DIR_ENV)
    if lora_dir:
        profiles["Qwen3-ASR-lora"] = _finetuned_profile(lora_dir)
    return profiles


MODEL_PROFILES: Mapping[str, ModelProfile] = _build_profiles()


def get_model_profile(name: str) -> ModelProfile:
    """Return the profile for ``name``.

    An unregistered name is treated as a raw HF id/path with all-default (no-op)
    downstream behavior, mirroring the old ``_MODEL_IDS.get(name, name)`` passthrough.
    """
    profile = MODEL_PROFILES.get(name)
    if profile is not None:
        return profile
    return ModelProfile(hf_id=name)

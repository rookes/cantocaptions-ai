"""Per-model downstream configuration registry.

A ``ModelProfile`` ties an ASR model name to the downstream behavior that depends on
how that particular model writes its output: post-ASR text normalization (OpenCC /
HK-variant rewriting), the punctuation set used for sentence splitting, and the
alignment "spot checks" that swap interchangeable particles.

Every field defaults to a no-op, so:
  * a model **not** in ``MODEL_PROFILES`` (or a new entry that omits fields) runs no
    OpenCC, uses the default punctuation, and performs no spot checks — the right
    starting point for a freshly fine-tuned model that already outputs the target
    convention;
  * adding a model means adding one ``MODEL_PROFILES`` entry — no edits to the ASR or
    alignment code.

This replaces the former ``_MODEL_IDS`` dict in ``_asr_native.py``. The value objects
themselves (``TextNormalization``, ``PunctuationConfig``, ``SpotCheck``) live in
``cantonese/text.py`` so that module stays free of any ``pipeline`` import.
"""
from dataclasses import dataclass, field
from typing import Mapping

from cantocaptions_ai.cantonese.text import (
    DEFAULT_PUNCTUATION,
    PunctuationConfig,
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


MODEL_PROFILES: Mapping[str, ModelProfile] = {
    "Qwen3-ASR": ModelProfile(
        hf_id="Qwen/Qwen3-ASR-1.7B-hf",
        normalization=_QWEN_NORMALIZATION,
        spotchecks=_QWEN_SPOTCHECKS,
    ),
    "Qwen3-ASR-0.6B": ModelProfile(
        hf_id="Qwen/Qwen3-ASR-0.6B-hf",
        normalization=_QWEN_NORMALIZATION,
        spotchecks=_QWEN_SPOTCHECKS,
    ),
    # Fine-tuned checkpoint: already emits HK-traditional text and custom final particles,
    # so it takes all defaults — no OpenCC, default punctuation, no spot checks. Copy this
    # entry as the template when adding a new model.
    "Qwen3-ASR-lora": ModelProfile(
        hf_id="<your custom model filepath>", # model training in progress - replace with local directory
    ),
}


def get_model_profile(name: str) -> ModelProfile:
    """Return the profile for ``name``.

    An unregistered name is treated as a raw HF id/path with all-default (no-op)
    downstream behavior, mirroring the old ``_MODEL_IDS.get(name, name)`` passthrough.
    """
    profile = MODEL_PROFILES.get(name)
    if profile is not None:
        return profile
    return ModelProfile(hf_id=name)

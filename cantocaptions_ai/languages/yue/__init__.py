"""Cantonese (yue): the language this project was built for, and the reference pack.

Everything Cantonese-specific the pipeline uses is declared here: written-Cantonese
conventions (CJK script and punctuation), the CantoCaptions fine-tune and the Cantonese
align model as defaults, how each supported ASR model writes Cantonese, the HK cleaning
rules, the audio-track preference, and the LLM correction prompts. The implementations live
beside it in this package (``text``, ``numbers``, ``linebreak``, ``rules/``, ...).

Module import must stay cheap: OpenCC, pycantonese and the cleaning builtins load on use.
"""
from typing import List, Mapping

from cantocaptions_ai.languages.base import CleaningSpec, LanguagePack, ModelConventions
from cantocaptions_ai.languages.yue.paths import RULES_DIR
from cantocaptions_ai.languages.yue.prompts import CORRECTION_PROMPTS
from cantocaptions_ai.languages.yue.text import REMOVE_STANDALONE_CHARS
from cantocaptions_ai.text_profiles import (
    CJK_PUNCTUATION,
    CJK_SCRIPT,
    SegmentationConfig,
    SpotCheck,
    TextNormalization,
)

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

# Qwen's raw output is Mandarin-flavoured and Simplified-derived, with generic final
# particles and Chinese numerals written out, so it needs the full legacy cleaning chain
# rather than the conservative default manifest a fine-tuned checkpoint gets.
_QWEN = ModelConventions(
    normalization=_QWEN_NORMALIZATION,
    spotchecks=_QWEN_SPOTCHECKS,
    segmentation=_LEADING_MARKER_SEGMENTATION,
    cleaning_manifest="pipeline_qwen.toml",
)

# The published fine-tune and the local LoRA build share one shape: both already emit
# HK-traditional text and the CantoCaptions final-particle convention, so they need no
# OpenCC, no spot checks and the conservative default cleaning manifest rather than Qwen's
# full chain. Cue assembly is the exception: they still clause off leading markers the way
# Qwen does, so they take Qwen's marker list too. Copy it as the template when adding
# another fine-tuned checkpoint.
_FINETUNED = ModelConventions(segmentation=_LEADING_MARKER_SEGMENTATION)


def _builtin_steps():
    from cantocaptions_ai.languages.yue.builtins import builtin_steps
    return builtin_steps()


def _select_track(streams: List[dict]) -> int:
    from cantocaptions_ai.utils.audio import select_cantonese_track
    return select_cantonese_track(streams)


YUE = LanguagePack(
    code="yue",
    script=CJK_SCRIPT,
    punctuation=CJK_PUNCTUATION,
    default_model="cantocaptions-cantonese-ASR",
    default_align_model="alvanlii/wav2vec2-BERT-cantonese",
    conventions={
        "Qwen3-ASR": _QWEN,
        "Qwen3-ASR-0.6B": _QWEN,
        "cantocaptions-cantonese-ASR": _FINETUNED,
        "Qwen3-ASR-lora": _FINETUNED,
    },
    cleaning=CleaningSpec(
        rules_dir=RULES_DIR,
        manifest="pipeline.toml",
        builtin_steps=_builtin_steps,
        noise_tokens=tuple(REMOVE_STANDALONE_CHARS),
    ),
    correction_prompts=CORRECTION_PROMPTS,
    ensemble_model=("alvanlii/whisper-small-cantonese", "cts"),
    track_selector=_select_track,
)

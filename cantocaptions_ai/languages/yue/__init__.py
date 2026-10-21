"""Cantonese (yue): the language this project was built for, and the reference pack.

Everything Cantonese-specific the pipeline uses is declared here: written-Cantonese
conventions (CJK script and punctuation), the CantoCaptions fine-tune and the Cantonese
align model as defaults, how each supported ASR model writes Cantonese, the HK cleaning
rules, the audio-track preference, the LLM correction prompts, and the CantoCaptions
standard for the optional proofreading stage. The implementations live
beside it in this package (``text``, ``numbers``, ``linebreak``, ``rules/``, ...).

Module import must stay cheap: OpenCC, pycantonese and the cleaning builtins load on use.
"""
from dataclasses import replace
from typing import List, Mapping

from cantocaptions_ai.languages.base import CleaningSpec, LanguagePack, ModelConventions
from cantocaptions_ai.languages.yue.paths import RULES_DIR
from cantocaptions_ai.languages.yue.proofreading import CANTOCAPTIONS
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


def _char_readings():
    from cantocaptions_ai.languages.yue.readings import JYUTPING
    return JYUTPING


def _select_track(streams: List[dict]) -> int:
    from cantocaptions_ai.utils.audio import select_cantonese_track
    return select_cantonese_track(streams)


# The ASR leaves out the punctuation after about 3% of final particles (啊 2.8%, 喇 4.4%), and
# where two utterances run together in one chunk, nothing else marks where they part
# (喺垃圾桶啊復原光線). The cue cutters cut there on a pause (align_checks.split_gapped_cues).
# Not text.PARTICLE_CHARS: that list keeps 嘅, 呢, 可, 咩 and 吓, which are words far more often
# than they end a sentence (嘅話, 呢個, 可以, 咩事). Those are continuations here, along with 嗬
# and 嗎: a particle followed by one may be a cluster (嚟㗎咩, 先啦吓). 囉 is left out because the
# conventions write the particle 囖, which leaves 囉 to 囉嗦 and 囉囉攣.
YUE_PUNCTUATION = replace(
    CJK_PUNCTUATION,
    final_particles=tuple("啊呀喇啦㗎喎嘛吖啫咋囖噃𠻹嘞咯𠺢吒啩"),
    particle_continuations=tuple("吓咩呢嗬嘅嗎哦"),
    particle_words=("喇叭", "啫喱"),
)


YUE = LanguagePack(
    code="yue",
    script=CJK_SCRIPT,
    punctuation=YUE_PUNCTUATION,
    default_model="cantocaptions-cantonese-ASR",
    default_align_model="alvanlii/wav2vec2-BERT-cantonese",
    conventions={
        "Qwen3-ASR": _QWEN,
        "Qwen3-ASR-0.6B": _QWEN,
        # Whisper's Cantonese is generic written Chinese too, so it takes Qwen's chain.
        "whisper-large-v3": _QWEN,
        "whisper-large-v3-turbo": _QWEN,
        "cantocaptions-cantonese-ASR": _FINETUNED,
        "Qwen3-ASR-lora": _FINETUNED,
    },
    cleaning=CleaningSpec(
        rules_dir=RULES_DIR,
        manifest="pipeline.toml",
        basic_manifest="pipeline_basic.toml",
        builtin_steps=_builtin_steps,
        noise_tokens=tuple(REMOVE_STANDALONE_CHARS),
    ),
    correction_prompts=CORRECTION_PROMPTS,
    ensemble_model=("alvanlii/whisper-small-cantonese", "cts"),
    track_selector=_select_track,
    char_readings=_char_readings,
    proofreading={CANTOCAPTIONS.name: CANTOCAPTIONS},
    default_proofreading=CANTOCAPTIONS.name,
    # Cantonese films and TV switch into English and Mandarin as a matter of course; those
    # stay as the ASR wrote them. Qwen3-ASR names every Chinese variety other than Cantonese
    # "Chinese" (asr_repair.is_native matches any label naming Chinese).
    native_asr_languages=("Cantonese", "Chinese", "English"),
)

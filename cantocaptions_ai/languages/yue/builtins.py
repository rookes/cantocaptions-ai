"""The Cantonese cleaning steps a manifest may name, and a cleaner preset for them.

``pipeline.toml`` and ``pipeline_qwen.toml`` (in ``rules/``) name these as ``builtin``
steps; ``linebreak`` is provided by the generic engine from the line settings.
"""
from typing import Callable, Mapping, Optional, Sequence

from cantocaptions_ai.cleaning.engine import SubtitleCleaner
from cantocaptions_ai.languages.yue.acronyms import format_acronyms
from cantocaptions_ai.languages.yue.linebreak import trim
from cantocaptions_ai.languages.yue.numbers import convert_chinese_numbers
from cantocaptions_ai.languages.yue.paths import RULES_DIR
from cantocaptions_ai.languages.yue.questions import clean_question_particles
from cantocaptions_ai.languages.yue.text import REMOVE_STANDALONE_CHARS
from cantocaptions_ai.text_profiles import DEFAULT_CLEANING

CANTONESE_BUILTIN_STEPS: Mapping[str, Callable[[str], str]] = {
    "acronyms": format_acronyms,
    "question_particles": clean_question_particles,
    "chinese_numbers": convert_chinese_numbers,
    "trim": trim,
}


def builtin_steps() -> Mapping[str, Callable[[str], str]]:
    """The loader ``CleaningSpec.builtin_steps`` calls (see languages/yue/__init__.py)."""
    return CANTONESE_BUILTIN_STEPS


class CantoneseCleaner(SubtitleCleaner):
    """A :class:`SubtitleCleaner` with the Cantonese pack's rules, steps and noise tokens.

    What ``cantonese.cleaner.SubtitleCleaner`` always was; every argument can still be
    overridden, and ``rules_dir`` may point at a user's own rule set.
    """

    BUILTIN_STEPS = CANTONESE_BUILTIN_STEPS

    def __init__(
        self,
        rules_dir: Optional[str] = None,
        line_max_length: int = 18,
        max_line_count: Optional[int] = 1,
        manifest: str = DEFAULT_CLEANING.manifest,
        builtin_steps: Optional[Mapping[str, Callable[[str], str]]] = None,
        noise_tokens: Sequence[str] = REMOVE_STANDALONE_CHARS,
        layout: str = "cjk",
    ) -> None:
        super().__init__(
            rules_dir=rules_dir if rules_dir is not None else RULES_DIR,
            line_max_length=line_max_length,
            max_line_count=max_line_count,
            manifest=manifest,
            builtin_steps=CANTONESE_BUILTIN_STEPS if builtin_steps is None else builtin_steps,
            noise_tokens=noise_tokens,
            layout=layout,
        )

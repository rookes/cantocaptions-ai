"""Subtitle line layout: breaking a cue's text over its lines.

A script names its line breaker (``ScriptConfig.layout``); this maps the name to the
function. ``"cjk"`` is the Cantonese breaker -- punctuation first, then a character
boundary pycantonese does not place inside a word -- and is what every CJK script uses
until another language brings its own segmenter. ``"word"`` breaks a space-separated line
at the space nearest its middle.
"""
from typing import Callable, Mapping, Optional

from cantocaptions_ai.text_profiles import word_wrap


def _cjk(text: str, line_max_length: int) -> str:
    # Imported on first use: the Cantonese breaker loads pycantonese, which the rest of
    # this module (and every non-CJK run) never needs.
    from cantocaptions_ai.languages.yue.linebreak import linebreak
    return linebreak(text, line_max_length)


LAYOUTS: Mapping[str, Callable[[str, int], str]] = {
    "cjk": _cjk,
    "word": word_wrap,
}


def linebreak_step(
    line_max_length: int, max_line_count: Optional[int], layout: str = "cjk",
) -> Optional[Callable[[str], str]]:
    """The line-layout step for these line limits, or None where no break is allowed.

    Shared by the cleaner's ``linebreak`` builtin and by the pipeline's layout pass when
    cleaning is off, so ``max_line_width``/``max_line_count`` mean the same either way.
    ``layout`` picks the line breaker from ``LAYOUTS`` (the script's ``layout``).
    """
    if max_line_count is not None and max_line_count < 2:
        return None  # a single-line output can't take a break
    if not line_max_length:
        return None  # 0 (or unset): lines are never broken
    breaker = LAYOUTS[layout]
    return lambda text: breaker(text, line_max_length)

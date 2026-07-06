"""Two-line subtitle line breaking and final trimming.

Long lines are split into two, preferring a break after delimiting punctuation in
the first half of the line, then after punctuation slightly past the midpoint, and
finally at any character boundary that does not fall inside a word (checked with
pycantonese word segmentation).
"""

import re

from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

RE_DELIMITING_PUNCTUATION = re.compile(r'[，？！…。：；]')

_RE_LEADING_TRIM = re.compile(r'^[，\s]+', re.MULTILINE)
_RE_TRAILING_TRIM = re.compile(r'[，\s]+$', re.MULTILINE)


def linebreak(text: str, line_max_length: int = 21, line_break_threshold: int = 1) -> str:
    """Split text into two lines if it is longer than ``line_max_length - 1``."""
    if '\n' in text:
        # Rule files are user-editable, so a replacement could introduce a newline;
        # never break an already-broken line further.
        return text

    length = len(text)

    if length <= line_max_length:
        return text

    # We always want two lines. Restrict first line shorter for aesthetic reasons.
    firstline_min_length = max(length // 4, 4)
    firstline_max_length = min(length // 2, line_max_length - 1)
    firstline_extended_length = min(length // 4 * 3, line_max_length)

    # If possible, split after delimiting punctuation in the first half of the line
    for i in range(firstline_max_length, firstline_min_length - 1, -1):
        if RE_DELIMITING_PUNCTUATION.match(text[i]):
            return text[:i + 1] + '\n' + text[i + 1:]

    # Otherwise, split after delimiting punctuation in the second half of the line
    for i in range(firstline_max_length, firstline_extended_length + 1):
        if RE_DELIMITING_PUNCTUATION.match(text[i]):
            return text[:i + 1] + '\n' + text[i + 1:]

    # Otherwise, split at the first non-punctuation character that's not mid-word
    import pycantonese  # deferred: loads corpus data on first segment() call

    from cantocaptions_ai.cantonese.text import is_punctuation

    for i in range(firstline_max_length, firstline_min_length - 1, -1):
        if is_punctuation(text[i]):
            return text[:i + 1] + '\n' + text[i + 1:]

        if len(pycantonese.segment(text[i:i + 2])) == 1:
            logger.debug(f"Skipping line break at char {i}: mid-word {text[i:i + 2]}")
            continue

        return text[:i + 1] + '\n' + text[i + 1:]

    logger.warning(f"No suitable line break found for line: {text}")
    return text


def trim(text: str) -> str:
    """Strip leading/trailing commas and whitespace from every line."""
    text = _RE_LEADING_TRIM.sub('', text)
    text = _RE_TRAILING_TRIM.sub('', text)
    return text

"""Two-line subtitle line breaking and final trimming.

Long lines are split into two, preferring a break after delimiting punctuation in
the first half of the line, then after punctuation slightly past the midpoint, and
finally at any character boundary that does not fall inside a word (checked with
pycantonese word segmentation).

Lengths are display widths in Chinese characters (``CJK_SCRIPT.width``): code-switched
English, digits and spaces count half. A run of English (``LATIN_WORD``) is never broken
inside; a space between two words is a break that is consumed, so neither line starts or
ends on one.
"""

import re
from typing import List, Optional, Set

from cantocaptions_ai.text_profiles import CJK_SCRIPT, LATIN_WORD, display_columns
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

RE_DELIMITING_PUNCTUATION = re.compile(r'[，？！…。：；]')
# English punctuation only delimits a clause when a space follows it ("Touched, moved"); a
# point inside a number or an abbreviation does not.
_ASCII_DELIMITERS = ",?!.;:"

_RE_LEADING_TRIM = re.compile(r'^[，,\s]+', re.MULTILINE)
_RE_TRAILING_TRIM = re.compile(r'[，,\s]+$', re.MULTILINE)


def _inside_words(text: str) -> Set[int]:
    """Indices *i* where a break after ``text[i]`` would fall inside an English word."""
    inside: Set[int] = set()
    for match in LATIN_WORD.finditer(text):
        inside.update(range(match.start(), match.end() - 1))
    return inside


def _is_delimiter(text: str, i: int) -> bool:
    if RE_DELIMITING_PUNCTUATION.match(text[i]):
        return True
    return text[i] in _ASCII_DELIMITERS and i + 1 < len(text) and text[i + 1] == ' '


def _split(text: str, i: int) -> str:
    """Break *text* after ``text[i]``, dropping any space the break lands next to."""
    return text[:i + 1].rstrip(' ') + '\n' + text[i + 1:].lstrip(' ')


def linebreak(text: str, line_max_length: int = 21, line_break_threshold: int = 1) -> str:
    """Split text into two lines if it is wider than ``line_max_length``."""
    if '\n' in text:
        # Rule files are user-editable, so a replacement could introduce a newline;
        # never break an already-broken line further.
        return text

    length = CJK_SCRIPT.width(text)

    if length <= line_max_length:
        return text

    # Width of text[:i + 1], i.e. of the first line when breaking after text[i]. For a line of
    # Chinese this is i + 1, which is what keeps the windows below picking the same break as
    # when they counted characters.
    prefix: List[float] = []
    total = 0
    for char in text:
        total += display_columns(char)
        prefix.append(total / CJK_SCRIPT.cell_columns)

    inside = _inside_words(text)
    last = len(text) - 1

    def allowed(i: int) -> bool:
        if i >= last or i in inside:
            return False
        # Neither line may end up empty once the spaces beside the break are dropped.
        return bool(text[:i + 1].strip()) and bool(text[i + 1:].strip())

    def window(lo: float, hi: float, descending: bool = True) -> List[int]:
        """Break positions whose first line is between lo + 1 and hi + 1 wide."""
        found = [i for i in range(last) if lo + 1 <= prefix[i] <= hi + 1 and allowed(i)]
        return found[::-1] if descending else found

    # We always want two lines. Restrict first line shorter for aesthetic reasons.
    firstline_min_length = max(length // 4, 4)
    firstline_max_length = min(length // 2, line_max_length - 1)
    firstline_extended_length = min(length // 4 * 3, line_max_length)

    # If possible, split after delimiting punctuation in the first half of the line
    for i in window(firstline_min_length, firstline_max_length):
        if _is_delimiter(text, i):
            return _split(text, i)

    # Otherwise, split after delimiting punctuation in the second half of the line
    for i in window(firstline_max_length, firstline_extended_length, descending=False):
        if _is_delimiter(text, i):
            return _split(text, i)

    # Otherwise, split at the first non-punctuation character that's not mid-word
    for i in window(firstline_min_length, firstline_max_length):
        if not _mid_word(text, i):
            return _split(text, i)

    # Nothing in the window: any boundary that is not mid-word, nearest the middle. A line
    # of English with no space near the middle lands here, as does one long word.
    candidates = [i for i in range(last) if allowed(i) and not _mid_word(text, i)]
    if candidates:
        best = min(candidates, key=lambda i: (prefix[i] > line_max_length,
                                              abs(prefix[i] - length / 2)))
        return _split(text, best)

    logger.warning(f"No suitable line break found for line: {text}")
    return text


def _mid_word(text: str, i: int) -> bool:
    """Whether a break after ``text[i]`` falls inside a Cantonese word.

    Only asked of two full-width characters: English words are protected by ``allowed``,
    and a boundary between English and Chinese is always a word boundary.
    """
    from cantocaptions_ai.languages.yue.text import is_punctuation

    if is_punctuation(text[i]):
        return False
    if display_columns(text[i]) < 2 or display_columns(text[i + 1]) < 2:
        return False

    import pycantonese  # deferred: loads corpus data on first segment() call

    if len(pycantonese.segment(text[i:i + 2])) == 1:
        logger.debug(f"Skipping line break at char {i}: mid-word {text[i:i + 2]}")
        return True
    return False


def trim(text: str) -> str:
    """Strip leading/trailing commas and whitespace from every line."""
    text = _RE_LEADING_TRIM.sub('', text)
    text = _RE_TRAILING_TRIM.sub('', text)
    return text

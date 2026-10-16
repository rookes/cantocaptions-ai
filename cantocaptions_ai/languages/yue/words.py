"""Where a Cantonese line may be cut without cutting a word.

Everything that splits a cue in two -- cue assembly's long-cue split, ``split_gapped_cues``,
realign's pause split -- asks :func:`inside_word` before it cuts, so 樂觀 is never left as 樂 on
one cue and 觀 on the next. A run of code-switched English is one word
(``text_profiles.LATIN_WORD``).

For Chinese, pycantonese's segmentation decides, with its dictionary as a check on the one
thing it gets wrong:

* **Its boundaries are trusted.** It reads 表現 / 有啲 where a dictionary alone would also
  find 現有 ("existing") across that boundary.
* **A token of up to three characters is one word**, including colloquial ones its dictionary
  lacks (早啲, 好啦).
* **A longer token is not trusted whole.** The segmenter glues what it does not know into one
  blob (演奏一首, 外觀同符合, 生蠔王豆沙包大, 喇豆沙包), and trusting those left a long cue
  nowhere to cut but the wrong place. Inside one, a cut is only inside a word where a
  dictionary word of 2-4 characters (the HKCanCor words and rime-cantonese entries the
  segmenter is built from) spans it: 演奏|一首 may be cut, 豆沙|包 may not.
"""
import re
from functools import lru_cache
from typing import FrozenSet, Tuple

from cantocaptions_ai.text_profiles import LATIN_WORD

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]+")

# The longest dictionary word looked for across a cut, and the longest segmenter token trusted.
_MAX_WORD = 4
_MAX_TRUSTED_TOKEN = 3


@lru_cache(maxsize=1)
def _dictionary() -> FrozenSet[str]:
    """Multi-character words pycantonese's segmenter knows. Loaded once, on first use."""
    from pycantonese.corpus import hkcancor
    from pycantonese.data.rime_cantonese import CHARS_TO_JYUTPING

    words = set(CHARS_TO_JYUTPING)
    for utterance in hkcancor().words(by_utterance=True):
        words.update(utterance)
    return frozenset(w for w in words if 2 <= len(w) <= _MAX_WORD)


@lru_cache(maxsize=4096)
def _token_spans(run: str) -> Tuple[Tuple[int, int], ...]:
    """(start, end) of each token pycantonese makes of *run* (empty if it rewrote the run)."""
    import pycantonese  # deferred: loads corpus data on first segment() call

    tokens = pycantonese.segment(run)
    if "".join(tokens) != run:
        return ()
    spans, at = [], 0
    for token in tokens:
        spans.append((at, at + len(token)))
        at += len(token)
    return tuple(spans)


def _inside_chinese_word(run: str, at: int) -> bool:
    for start, end in _token_spans(run):
        if not start < at < end:
            continue
        if end - start <= _MAX_TRUSTED_TOKEN:
            return True
        words = _dictionary()
        return any(run[lo:hi] in words
                   for lo in range(max(start, at - _MAX_WORD + 1), at)
                   for hi in range(at + 1, min(end, lo + _MAX_WORD) + 1))
    return False


def inside_word(text: str, at: int) -> bool:
    """Whether cutting *text* before ``text[at]`` would split a word."""
    if not 0 < at < len(text):
        return False
    for match in LATIN_WORD.finditer(text):
        if match.start() < at < match.end():
            return True
    for match in _CJK_RUN.finditer(text):
        if match.start() < at < match.end():
            return _inside_chinese_word(match.group(), at - match.start())
    return False

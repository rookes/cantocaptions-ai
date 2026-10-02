"""Cantonese character readings for alignment's vocab repair (``pipeline/align_vocab``).

Jyutping from pycantonese, character frequencies from its HKCanCor corpus, and the
Simplified/variant fold through this pack's OpenCC config. See ``languages.base.CharReadings``.
"""
import logging
from collections import Counter
from functools import lru_cache
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def reading_of(char: str) -> Optional[str]:
    """The Jyutping for a single character, or None if nothing knows it.

    Goes through pycantonese's public converter rather than its bundled dictionary, which is
    an internal module. Only one reading per character is available there, so a polyphone is
    matched on its commonest reading -- acceptable, given the alternative for these
    characters is no token at all.
    """
    try:
        import pycantonese
    except ImportError:  # pragma: no cover - pycantonese is a base dependency
        logger.warning("pycantonese is not installed; homophone substitution is unavailable.")
        return None
    got = pycantonese.characters_to_jyutping(char)
    reading = got[0][1] if got else None
    # A multi-syllable result means the converter re-segmented, which is not a character
    # reading.
    return reading if reading and " " not in reading else None


@lru_cache(maxsize=1)
def _hkcancor_frequencies() -> Mapping[str, int]:
    freq: "Counter[str]" = Counter()
    try:
        import pycantonese
        for word in pycantonese.hkcancor().words():
            freq.update(word)
    except Exception as exc:  # pragma: no cover - corpus ships with pycantonese
        logger.debug("No HKCanCor frequencies for substitution ranking (%s)", exc)
    return freq


class JyutpingReadings:
    """``CharReadings`` for Cantonese."""

    label = "Jyutping"

    def reading(self, char: str) -> Optional[str]:
        return reading_of(char)

    def toneless(self, reading: str) -> str:
        return reading.rstrip("123456")

    def variant(self, char: str) -> Optional[str]:
        from cantocaptions_ai.languages.yue.text import simplified_to_traditional
        try:
            converted = simplified_to_traditional(char)
        except Exception as exc:  # pragma: no cover - opencc is a base dependency
            logger.debug("OpenCC unavailable for variant folding (%s)", exc)
            return None
        return converted if len(converted) == 1 and converted != char else None

    def frequencies(self) -> Mapping[str, int]:
        return _hkcancor_frequencies()


JYUTPING = JyutpingReadings()

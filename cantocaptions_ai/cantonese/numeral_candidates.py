"""Arabic-digit -> Chinese-numeral reading candidates.

The reverse direction of :mod:`cantocaptions_ai.cantonese.numbers` (which reads
Chinese numerals BACK into digits for post-ASR display). This module exists for a
different problem: a forced-aligner's CTC vocabulary is built from real subtitle
text, which is overwhelmingly written with Chinese numerals, so it typically holds
tokens for every Chinese numeral character but only a handful of Arabic digits (a
model trained on this corpus's own convention has no reason to ever see "5" as a
token). A line like ``"1891年"`` then aligns with no acoustic evidence at all for
its digits.

:func:`numeral_readings` does not decide which reading is correct -- Cantonese
genuinely has two: a POSITIONAL reading (1891 -> 一千八百九十一) and a DIGIT-WISE
reading (1891 -> 一八九一, the common style for years, phone numbers, IDs). It
returns both, most-likely-first, so a caller can score each against real audio
(see :func:`cantocaptions_ai.pipeline.alignment.align_best_of`) and keep whichever
one actually fits -- never guessing the reading from the text alone. The reader's
own ground-truth text is never touched; this only ever produces a candidate for
scoring.

Digit runs that are NOT numerals are left alone:

- A jyutping pronunciation gloss (``踩(jaai2)``, ``m4(噉)``) is a corpus convention
  for annotating a character's reading, not a quantity. Detected by shape (1-6
  lowercase letters immediately followed by a tone digit 1-6, word-bounded) and
  masked out before the digit scanner ever sees it, so ``m4`` is never misread as
  the digit 4.

A leading zero (``007``, ``09``, ``012``) is never a positional quantity -- nobody
reads "012" as "twelve" -- so it is always forced digit-wise regardless of which
variant slot is being rendered. A decimal fraction is always read digit-by-digit
in Cantonese (``1.5`` -> 一點五, not "one and a half" as a single quantity), matching
the fix in :mod:`numbers.py` for the reverse direction. A thousands-separator comma
(``204，000``) is joined before the digit run is read, guarded to exactly 3 trailing
digits so an ordinary sentence comma between two numbers is never absorbed.

Deliberately NOT implemented (see the dataset repo's discovery notes): truncated
colloquial readings (百五 = 150, 千七 = 1700) and 廿/卅 substitution. These would
close a real but small gap (~5% of gold corpus numeral spellings) at the cost of
more candidates to score; add them if a case-by-case review of low-scoring
digit-bearing rows shows it is worth it. Negative numbers (負) are not handled --
a corpus survey found essentially no genuine cases (every ``-\\d`` match was a
false positive: "7-11", a multispeaker dash, a licence plate).
"""

import re

_DIGIT_CHARS = "零一二三四五六七八九"
_SECTION_UNIT = {4: "千", 3: "百", 2: "十", 1: ""}
_GROUP_UNIT = ["", "萬", "億", "兆"]

# A jyutping-shaped annotation: 1-6 lowercase letters immediately followed by a
# tone digit 1-6, word-bounded on both sides (never touching "m4" inside a longer
# alphanumeric token). Tried first in _RE_NUM_OR_JYUT below so it consumes the run
# before the bare-digit branch ever sees it -- one linear pass, no separate
# mask-then-scan step that could drift out of sync with the digit scanner.
_RE_JYUTPING = r"(?<![A-Za-z0-9])[a-z]{1,6}[1-6](?![A-Za-z0-9])"
_RE_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_RE_NUM_OR_JYUT = re.compile(f"(?:{_RE_JYUTPING})|(?P<num>{_RE_NUMBER})")

# Fullwidth digits and the fullwidth period both denote a number; the fullwidth
# COMMA is left alone here on purpose -- it is this corpus's ordinary sentence
# separator, and folding it in would misparse "...一百，五十人..." (two clauses) as
# one number. A genuine thousands separator is joined explicitly by
# _RE_THOUSANDS_SEP below, guarded to exactly 3 trailing digits.
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．", "0123456789.")

# Joins a thousands-grouped digit string ("204，000") into one token, but ONLY
# when the comma is followed by exactly 3 digits with no fourth -- so a sentence
# comma between two otherwise-adjacent numbers is never absorbed.
_RE_THOUSANDS_SEP = re.compile(r"(?<=[0-9])[,，](?=[0-9]{3}(?![0-9]))")


def _section(n: int) -> str:
    """Positional reading of 0 < n < 10000: 15 -> 一十五, 104 -> 一百零四."""
    if n == 0:
        return ""
    digits = str(n)
    length = len(digits)
    out = []
    pending_zero = False
    for i, ch in enumerate(digits):
        d = int(ch)
        if d == 0:
            pending_zero = True
            continue
        if pending_zero and out:
            out.append("零")
        pending_zero = False
        out.append(_DIGIT_CHARS[d] + _SECTION_UNIT[length - i])
    return "".join(out)


def _combined(n: int) -> str:
    """Positional reading of any non-negative integer: 12674 -> 一萬二千六百七十四."""
    if n == 0:
        return "零"
    groups = []
    m = n
    while m:
        groups.append(m % 10000)
        m //= 10000
    if len(groups) > len(_GROUP_UNIT):
        return _digitwise(str(n))  # beyond 兆 -- positional reading stops being useful
    parts = []
    pending_zero = False
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            if parts:
                pending_zero = True
            continue
        if parts and (pending_zero or g < 1000):
            parts.append("零")
        pending_zero = False
        parts.append(_section(g) + _GROUP_UNIT[i])
    s = "".join(parts)
    return s[1:] if s.startswith("一十") else s  # 一十X reads as 十X


def _digitwise(digits: str) -> str:
    """Digit-by-digit reading: "1891" -> 一八九一, "007" -> 零零七."""
    return "".join(_DIGIT_CHARS[int(c)] for c in digits)


_RE_LOENG = re.compile(r"二(?=[百千萬億兆])")


def _with_loeng(s: str) -> str:
    """兩 for 二百/二千/二萬/二億/二兆 -- never 二十, never a bare 二."""
    return _RE_LOENG.sub("兩", s)


def _token_readings(token: str) -> list[str]:
    """Ordered readings for one matched digit run, most likely first."""
    if "." in token:
        # The fractional side is always digit-by-digit (Cantonese never reads a
        # decimal fraction by place value). The integer side still has the usual
        # combined-vs-digitwise choice, same as a bare integer.
        int_part, _, frac_part = token.partition(".")
        tail = _digitwise(frac_part)
        if not int_part:
            return [f"零點{tail}"]
        combined_head = _with_loeng(_combined(int(int_part)))
        digitwise_head = _digitwise(int_part)
        heads = [combined_head] if combined_head == digitwise_head else [combined_head, digitwise_head]
        return [f"{h}點{tail}" for h in heads]

    if token.startswith("0") and len(token) > 1:
        # A leading zero is never a positional quantity (007, 09年, 012).
        return [_digitwise(token)]

    n = int(token)
    combined = _with_loeng(_combined(n))
    digitwise = _digitwise(token)
    return [combined] if combined == digitwise else [combined, digitwise]


def has_digits(text: str) -> bool:
    """True if `text` contains a real digit run once jyutping-shaped annotations
    (``m4``, ``teng1``) are excluded. Cheap pre-check so a caller can skip the
    candidate machinery entirely for the ~98% of lines with no digits at all."""
    t = text.translate(_FULLWIDTH_DIGITS)
    t = _RE_THOUSANDS_SEP.sub("", t)
    return any(m.group("num") for m in _RE_NUM_OR_JYUT.finditer(t))


def numeral_readings(text: str, max_variants: int = 2) -> list[str]:
    """Up to `max_variants` full-text renderings of `text` with every digit run
    replaced by a Chinese-numeral reading, most likely first. Every digit run in
    a given variant is rendered in the SAME style (all positional, or all
    digit-wise) -- a deliberate simplification: scoring per-line rather than
    per-number was measured to cost only ~0.03% mean acoustic score on a 400-line
    sample, far below the cost of the extra complexity. Returns `[text]`
    unchanged (a single "variant") if there are no real digits to convert.
    """
    t = text.translate(_FULLWIDTH_DIGITS)
    t = _RE_THOUSANDS_SEP.sub("", t)
    matches = [m for m in _RE_NUM_OR_JYUT.finditer(t) if m.group("num")]
    if not matches:
        return [text]

    per_token = [_token_readings(m.group("num")) for m in matches]
    depth = min(max(len(r) for r in per_token), max(max_variants, 1))

    variants, seen = [], set()
    for k in range(depth):
        pieces = []
        cursor = 0
        for m, readings in zip(matches, per_token):
            pieces.append(t[cursor:m.start()])
            pieces.append(readings[min(k, len(readings) - 1)])
            cursor = m.end()
        pieces.append(t[cursor:])
        variant = "".join(pieces)
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)
    return variants

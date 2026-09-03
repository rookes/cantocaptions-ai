"""Chinese numeral to Arabic numeral conversion for subtitle text.

Three contexts are recognised, in this order:

``<n>點`` / ``<n>點<n>``
    ``點`` is an hour, but it is also a point and a degree. A numeral on *both*
    sides fixes the reading, so both halves convert whatever their size --
    ``八點三十分`` -> ``8點30分``, ``八點六十分`` -> ``8點60分``, ``三十六點五度`` ->
    ``36點5度``. Where the pair could genuinely be a clock time (a following ``分``,
    hour in ``MIN_HOUR..MAX_HOUR``, minute <= ``MAX_MINUTE``) the minute is
    zero-padded to read as a clock face: ``八點零五分`` -> ``8點05分``.

    With nothing after the ``點`` the reading is not fixed, so the numeral converts
    only once it is too large to be an hour -- above ``MAX_BARE_HOUR``, a bare hour
    being written on the 12-hour clock. ``五點`` and ``十二點`` stay; ``十九點`` ->
    ``19點``.

    A hedge either side (``十九點幾``, ``十幾點``, ``十二點幾分鐘``), a colloquial ten
    (``廿三點``) or the quarter-hour count ``個字`` (``十點四個字``) suppresses the whole
    construct, and neither half converts.

``<n>月`` / ``<n>號``
    Dates and numeric names (``一月二十三號`` -> ``1月23號``, ``五號`` -> ``5號``).
    Converted unconditionally: the standard numeral is the convention here
    regardless of magnitude.

Anything else
    A plain quantity, converted only where writing it in digits is an improvement --
    see :func:`_should_convert`. Years written digit-by-digit (``一二三四年``) are
    converted digit-by-digit.
"""

import re
from typing import NamedTuple, Optional

_CHINESE_DIGITS = {
    '零': 0, '一': 1, '二': 2, '兩': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}
_SMALL_UNITS = {'十': 10, '百': 100, '千': 1000}
_BIG_UNITS = {'萬': 10 ** 4, '億': 10 ** 8}
_UNITS = {**_SMALL_UNITS, **_BIG_UNITS}

# 廿 (20) and 卅 (30) are written-Cantonese colloquialisms; a writer who reaches for
# one has chosen a register that "21" does not carry, so they suppress conversion.
_COLLOQUIAL_TENS = {'廿': 20, '卅': 30}

# A hedge on either side makes the quantity approximate: 幾十個, 十幾個, 四十幾.
_MODIFIERS = frozenset('幾數多餘約')

# Suffixes that make the number a date or a name rather than a quantity.
_NAMED_SUFFIXES = frozenset('月號')

_NUM_CHARS = ''.join(_CHINESE_DIGITS) + ''.join(_UNITS) + ''.join(_COLLOQUIAL_TENS)
_NUM = f'[{_NUM_CHARS}]+'

# 十點四個字: 個字 counts the minutes in fifths of an hour, so the 四 is not a number
# the reader wants in digits.
_QUARTER_HOUR_UNIT = '個字'

_POINT_PATTERN = re.compile(f'({_NUM})點({_NUM})?')
_NUMBER_PATTERN = re.compile(f'({_NUM})([{"".join(sorted(_NAMED_SUFFIXES))}]?)')
_YEAR_PATTERN = re.compile(r'[零一二三四五六七八九]{2,4}年')

# Only used to decide whether a converted 點 pair is a clock face worth zero-padding
# the minute of, never whether to convert it at all.
MIN_HOUR = 1  # 零點五分 is the decimal 0.5, not five past midnight
MAX_HOUR = 23
MAX_MINUTE = 59

# 點 is also "point" and "degree", so a numeral in front of a *bare* 點 is only left
# alone while it could still be read as an hour. A bare hour is written on the
# 12-hour clock in practice (十二點, 五點), so above this the 點 is a measurement and
# the numeral converts like any other: 十九點 -> 19點.
MAX_BARE_HOUR = 12

# At and above this, digit grouping is what makes the numeral readable at all.
_GROUPING_THRESHOLD = 10000


class _Number(NamedTuple):
    """A parsed Chinese numeral, plus what the *writing* of it told us."""

    value: int
    max_unit: int       # largest 十/百/千/萬/億 used; 0 when the numeral has none
    unit_is_last: bool  # ...and that largest unit is also the final character
    colloquial: bool    # 廿 / 卅
    has_digit: bool     # a digit was written, so this is not a bare 十/百/千


def _parse(text: str) -> Optional[_Number]:
    """Parse a run of numeral characters, or return ``None`` if it is not a number.

    Digits are accumulated into a section that the next 萬/億 multiplies whole, so a
    big unit applies to everything written in front of it: 二十五萬 is (20 + 5) x
    10,000, not 20 + 5 x 10,000.
    """
    total = 0        # completed 萬/億 groups
    section = 0      # accumulated below the next big unit
    digit: Optional[int] = None
    after_zero = False   # 零 is an explicit place-holder: 一百零八 is 108, not 180
    max_unit = 0
    last_unit = 0
    has_digit = False
    colloquial = False

    for char in text:
        if char == '零':
            if digit is not None or after_zero:
                return None  # 一零一 / 零零: a digit string, not a quantity
            after_zero = True
        elif char in _CHINESE_DIGITS:
            if digit is not None:
                return None  # 一二三: two digits with no unit between them
            digit = _CHINESE_DIGITS[char]
            has_digit = True
        elif char in _COLLOQUIAL_TENS:
            if digit is not None or section or total:
                return None
            section = _COLLOQUIAL_TENS[char]
            colloquial = True
            max_unit = max(max_unit, 10)
            last_unit = 10
            after_zero = False
        elif char in _SMALL_UNITS:
            unit = _SMALL_UNITS[char]
            if digit is None:
                if char != '十' or section or total or after_zero:
                    return None  # a unit with nothing in front of it: 三百百, 零十
                digit = 1  # 十五 = 15
            section += digit * unit
            digit = None
            after_zero = False
            max_unit = max(max_unit, unit)
            last_unit = unit
        elif char in _BIG_UNITS:
            unit = _BIG_UNITS[char]
            if digit is not None:
                section += digit
                digit = None
            if not section and not total:
                return None  # a bare 萬 / 億
            if unit == _BIG_UNITS['億']:
                total = (total + section) * unit
            else:
                total += section * unit
            section = 0
            after_zero = False
            max_unit = max(max_unit, unit)
            last_unit = unit
        else:
            return None

    value = total + section
    if digit is not None:
        # A trailing bare digit takes the place below the last unit written --
        # 三百五 is 350, 五十五萬一 is 551,000 -- unless a 零 said otherwise.
        value += digit if (after_zero or not last_unit) else digit * (last_unit // 10)

    return _Number(
        value=value,
        max_unit=max_unit,
        unit_is_last=bool(max_unit) and _UNITS.get(text[-1]) == max_unit,
        colloquial=colloquial,
        has_digit=has_digit,
    )


def _should_convert(number: _Number) -> bool:
    """Whether a plain quantity (no date, name or time context) reads better in digits."""
    if not number.max_unit:
        return False  # no 十/百/千/萬/億 at all: 八, 一二三
    if not number.has_digit:
        return False  # a bare 十 or 百 is a quantifier, not a written-out number
    if number.colloquial:
        return False
    if number.max_unit >= _BIG_UNITS['萬'] and number.unit_is_last:
        # The 萬/億 applies to everything in front of it and nothing follows it, so
        # the Chinese form *is* the compact one: 二十五萬 beats 250,000.
        return False
    return True


def _format(value: int) -> str:
    """Digits, grouped only where the grouping is what makes them readable."""
    return f'{value:,}' if value >= _GROUPING_THRESHOLD else str(value)


def convert_chinese_numbers(text: str) -> str:
    """Replace convertible Chinese numerals in text with Arabic numerals."""

    def point_replacer(match):
        """A 點 construct: a clock time, or a point/degree reading of the same shape."""
        left = _parse(match.group(1))
        right = _parse(match.group(2)) if match.group(2) else None
        if left is None or left.colloquial:
            return match.group(0)
        if match.group(2) and (right is None or right.colloquial):
            return match.group(0)

        source = match.string
        before = source[match.start() - 1] if match.start() else ''
        after = source[match.end():]
        if before in _MODIFIERS or after[:1] in _MODIFIERS:
            return match.group(0)  # 十二點幾分鐘, 十九點幾: an approximation
        if after.startswith(_QUARTER_HOUR_UNIT):
            return match.group(0)  # 十點四個字

        if right is None:
            # Nothing after the 點 to fix the reading, so it converts only once it is
            # too large to be an hour.
            return f'{left.value}點' if left.value > MAX_BARE_HOUR else match.group(0)

        if (after.startswith('分')
                and MIN_HOUR <= left.value <= MAX_HOUR
                and right.value <= MAX_MINUTE):
            # A clock face, so the minute is zero-padded: 8點05分, not 8點5分.
            return f'{left.value}點{right.value:02d}'
        return f'{left.value}點{right.value}'

    text = _POINT_PATTERN.sub(point_replacer, text)

    def replacer(match):
        chinese_num, suffix = match.group(1), match.group(2)
        number = _parse(chinese_num)
        if number is None or number.colloquial:
            return match.group(0)

        source = match.string
        before = source[match.start() - 1] if match.start() else ''
        after = source[match.end()] if match.end() < len(source) else ''
        if before in _MODIFIERS or after in _MODIFIERS:
            return match.group(0)  # 幾十個, 四十幾: an approximation, not a quantity

        if suffix:
            return f'{number.value}{suffix}'  # a date or a numeric name
        if '點' in (before, after):
            # The 點 pass has already had its chance at this numeral and declined it,
            # so a half-converted 十點4個字 is not on the table.
            return match.group(0)

        return _format(number.value) if _should_convert(number) else chinese_num

    text = _NUMBER_PATTERN.sub(replacer, text)

    def year_replacer(match):
        chinese_num = match.group(0)
        for old_char, new_char in _CHINESE_DIGITS.items():
            chinese_num = chinese_num.replace(old_char, str(new_char))
        return chinese_num

    return _YEAR_PATTERN.sub(year_replacer, text)

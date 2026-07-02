"""Chinese numeral to Arabic numeral conversion for subtitle text.

Converts unambiguous quantities (二十三 -> 23) while leaving idiomatic or uncertain
usages alone (一二三 counting, 幾十 "tens of", round numbers like 一百). Years written
digit-by-digit (一二三四年) are converted digit-by-digit (1234年).
"""

import re

_CHINESE_DIGITS = {
    '零': 0, '一': 1, '二': 2, '兩': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}
_CHINESE_UNITS = {'十': 10, '百': 100, '千': 1000, '萬': 10000}
_UNCERTAIN_WORDS = {'幾', '數', '多', '餘', '約'}
_NAMED_DATE_WORDS = {'月', '號'}

_NUMBER_PATTERN = re.compile(r'[零一二兩两三四五六七八九十百千萬]+([月號]?)')
_YEAR_PATTERN = re.compile(r'[零一二三四五六七八九]{2,4}年')

# Round numbers read more naturally in Chinese; leave them alone.
_ROUND_NUMBERS = {100, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000}


def _parse_chinese_number(chinese_num: str):
    if any(word in chinese_num for word in _UNCERTAIN_WORDS):
        return None

    if chinese_num[0] not in _CHINESE_DIGITS and chinese_num[0] != '十':
        return None

    num = 0
    temp = 0
    last_was_digit = False

    length = len(chinese_num)
    i = 0

    while i < length:
        char = chinese_num[i]
        if char in _CHINESE_DIGITS:
            if last_was_digit and chinese_num[i - 1] != '零':
                # Two digits in a row without unit = invalid (e.g. 一二三 counting),
                # except after the placeholder 零 (五千零一十 = 5010)
                return None
            temp = _CHINESE_DIGITS[char]
            i += 1
            last_was_digit = True
            if i < length:
                next_char = chinese_num[i]
                if next_char in _NAMED_DATE_WORDS:
                    num += temp
                    return str(num)
                elif next_char in _CHINESE_UNITS:
                    num += temp * _CHINESE_UNITS[next_char]
                    temp = 0
                    i += 1
                    last_was_digit = False
                else:
                    num += temp
                    temp = 0
            else:
                num += temp
        elif char in _CHINESE_UNITS:
            if i == 0:
                num += _CHINESE_UNITS[char]
            i += 1
            last_was_digit = False
        else:
            return None

    if 10 < num < 100000 and num not in _ROUND_NUMBERS:
        return str(num)
    else:
        return None


def convert_chinese_numbers(text: str) -> str:
    """Replace convertible Chinese numerals in text with Arabic numerals."""

    def replacer(match):
        chinese_num = match.group(0)
        arabic_num = _parse_chinese_number(chinese_num)
        if arabic_num is not None:
            return arabic_num + match.group(1)
        else:
            return chinese_num + match.group(1)

    text = _NUMBER_PATTERN.sub(replacer, text)

    def year_replacer(match):
        chinese_num = match.group(0)
        for old_char, new_char in _CHINESE_DIGITS.items():
            chinese_num = chinese_num.replace(old_char, str(new_char))
        return chinese_num

    return _YEAR_PATTERN.sub(year_replacer, text)

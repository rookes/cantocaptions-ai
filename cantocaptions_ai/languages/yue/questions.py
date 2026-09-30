"""Question detection and question-aware final-particle correction.

Written Cantonese distinguishes particle spellings by sentence type (e.g. 呀 in
questions vs 啊 in statements). This module splits a subtitle line into segments,
classifies each as question / question-marked / statement, and applies the
appropriate particle swaps. The regex post-processing that follows this step in
the cleaning pipeline lives in ``rules/question_post.toml``.
"""

import re
from typing import List

# Words whose presence (with a trailing ？) marks a segment as a genuine question.
QUESTION_WORDS = ['做乜', '係咪', '未', '有冇', '好冇', '邊', '咩', '邊個', '點解', '幾歲', '幾耐', '幾時', '邊度', '點', '點樣', '幾多', '乜嘢', '嗎']

# X唔X constructions (係唔係, 好唔好, ...) also mark questions.
RE_QUESTION_PAT = re.compile(r'([一-鿿])唔\1')

RE_QUESTION_DELIMITING_PUNCTUATION = re.compile(r'([？！。：；]+)')


def split_segments(line: str) -> List[str]:
    """Split a line into segments after sentence-delimiting punctuation."""
    line = RE_QUESTION_DELIMITING_PUNCTUATION.sub(r'\1,', line)
    return [x for x in line.split(',') if x]


def is_question(segment: str) -> bool:
    """True if the segment ends with ？ and contains a question word or X唔X pattern."""
    if segment and segment[-1] == '？':
        if RE_QUESTION_PAT.search(segment) or any(word in segment for word in QUESTION_WORDS):
            return True

    return False


def clean_question_particles(text: str) -> str:
    """Swap final-particle spellings per segment based on question context."""

    def _update_segment(s: str) -> str:
        if s == '':
            return s

        has_question_mark = s[-1] == '？'

        if is_question(s):              # has question mark and question word
            s = s.replace('呀', '啊')
            s = s.replace('嘎', '㗎')
            s = s.replace('啫？', '唧？')

            # 啊 -> 呀 in cases like "乜你覺得唔開心啊？"
            if len(s) > 1 and s[0] == '乜' and s[1] != '嘢':
                s = s.replace('啊？', '呀？')
        elif has_question_mark:         # has question mark and no question word
            s = s.replace('㗎？', '嘎？')
            s = s.replace('啊？', '呀？')
        else:                           # no question mark and no question word
            s = s.replace('呀', '啊')
            s = s.replace('嘎', '㗎')

        return s

    return ''.join(map(_update_segment, split_segments(text)))

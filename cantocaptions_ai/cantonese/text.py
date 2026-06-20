"""
A utility library for Cantonese parsing of raw text
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cantocaptions_ai.utils.schema import SingleSegment

from opencc import OpenCC
_OPENCC_CONFIG = str(Path(__file__).parent / "opencc" / "s2t_c.json")
cc = OpenCC(_OPENCC_CONFIG)

MAX_CHARS = 18

# To understand how we process final particles, see the preprocess/postprocess particles function below
PARTICLE_CHARS = [
    '吖', '啊', '呀', '噃', '㗎', '嘅', '吓', '可', '嗬', '啩', '囖', '囉', '咯', '啦', '喇', '嘞', '呢', '咧', '哩', '嗎', '嘛', '咩', '𠻹', '喎', '啫', '唧'
]

QWEN_PARTICLE_MAP = {
    #'': ['', '喎'], # Qwen often just excludes 喎
    '啊': ['吖', '呀'],
    '吖': ['吖', '呀'],
    '呀': ['吖', '呀'],
    '咯': ['喇', '啦', '囉'],
    '喇': ['喇', '啦', '囉'],
    '啦': ['喇', '啦', '囉'],
    #'呢': ['呢', '啦'],
    #'嘅': ['嘅', '㗎'],
    #'㗎': ['㗎', '㗎喎'],
    '咋': ['咋', '啫'],
    '啫': ['咋', '啫'],
    '咁': ['咁', '噉'],
    #'喎': ['喎'],
    #'嘅喎': ['㗎喎'],
    #'嘅喇': ['㗎喇', '㗎啦'],
    #'㗎喎': ['㗎喎'],
    #'㗎喇': ['㗎喇', '㗎啦'],
}

REMOVE_CHARS_TRANS = str.maketrans('', '', "~～.,!?;")
SPLIT_CHARS = ["，", "。", "？", "！", "；", "…"]
MERGEABLE_CHARS = ["，"]
REMOVE_STANDALONE_CHARS = ["噢", "嗯", "哦", "嘩", "嗌", "唉", "誒", "哎", "啊", "嘿", "吓"]
QUESTION_WORDS = ['做乜', '係咪', '未', '有冇', '好冇', '邊', '咩', '邊個', '點解', '幾歲', '幾耐', '幾時', '邊度', '點', '點樣', '幾多', '乜嘢', '嗎']

RE_QUESTION_DELIMITING_PUNCTUATION = re.compile(r'([？！。：；]+)')
ZH = r'[一-鿿]'

def _resub(text, pattern, repl):
    """Helper function to perform regex substitution."""

    return re.sub(pattern, repl, text)

def _resub(text, regex_list):
    """Helper function to perform multiple regex substitutions."""

    for pattern, repl in regex_list:
        text = re.sub(pattern, repl, text)

    return text

def _segments(text) -> List[str]:
    line = re.sub(RE_QUESTION_DELIMITING_PUNCTUATION, r'\1,', line) # add an English comma as our new delimiter
    line = [x for x in line.split(',') if x] # remove empty segments

    return line

def _clean_punctuation(text):
    regex_list = [
        (r'[\n\t]+', ' '), # Replace all line breaks and tabs with a space (to be removed later)
        (r'﹑', '\''), # restore normal apostrophe
        (r'([a-zA-Z])-([a-zA-Z])', r'\1\2'), # remove random hyphens
        (r'(?<![a-zA-Z])\s+(?![a-zA-Z])', ''), # remove spaces when not next to Latin characters
        (r'(?<=' + ZH + r')\s+', ''), # remove spaces next to Chinese characters
        (r'\s+(?=' + ZH + r')', ''),
        (r'\s+', ' '), # reduce all spaces to single space
        ('％', '%'), # Netflix standard uses half-width percent sign
        (r'\?', '？'),
        (r'\.\.\.', '…'),
        (r'[。.]$', ''),
        (r'^[。.]', ''),
        (r'[!！]', '，'),
        (r',', '，'),
        (r'^，', ''),
        (r'，$', ''),
        (r'([，？…])[，？…]+', r'\1') # remove repeated punctuation
    ]

    return _resub(text, regex_list)

def _clean_question_particles(text):
    # Smart replacement of final particles based on question context
    segments = _segments(text)

    def _is_question(s):
        if s[-1] == '？':
            if re.search(r'([一-鿿])唔\1', s) or any(word in s for word in QUESTION_WORDS):
                return True

        return False

    def _update_segment(s):
        if s == '':
            return s

        has_question_mark = s[-1] == '？'

        if _is_question(s):        # has question mark and question word
            s = s.replace('呀', '啊')
            s = s.replace('嘎', '㗎')
            s = s.replace('啫？', '唧？')

            # 啊 -> 呀 in cases like "乜你覺得唔開心啊？"
            if s[0] == '乜' and s[1] != '嘢':
                s = s.replace('啊？', '呀？')
        elif has_question_mark:         # has question mark and no question word
            s = s.replace('㗎？','嘎？')
            s = s.replace('啊？','呀？')
        else:                           # no question mark and no question word
            s = s.replace('呀', '啊')
            s = s.replace('嘎', '㗎')

        return s

    segments = map(_update_segment, segments)
    text = ''.join(segments)

    regex_list = [
        (r'(?<![，。！!?.;？；…])係咪(?=[呀啊吖？])', '，係咪'), # add comma to tag question 係咪
        (r'[㗎喇]㗎', '㗎'),
        (r'嘅？', '𠸏？'),
        ('啦啦聲', '嗱嗱聲'),
        (r'([啊喎喇啦㗎咋噃嗎嘛])(?![？\n！，…啊呀吖喇啦喎啝噃咩吒咋喳啫唧嗎嗱呢𠻹添㖭嘛嗎囉囖咯])', r'\1，'), #Add comma after final particles
        (r'^啊…', '') # Remove isolated 啊…
    ]

    text = _resub(text, regex_list)

    return text

def simplified_to_traditional(text: str) -> str:
    return cc.convert(text)

def standardize_chars_hk(text: str) -> str:
    regex_list = [
        ('爲', '為'),
        ('嬀', '媯'),
        ('僞', '偽'),
        ('潙', '溈'),
        ('蔿', '蒍'),
        ('搵', '揾'),
        ('溫', '温'),
        ('慍', '愠'),
        ('醞', '醖'),
        ('媼', '媪'),
        ('榲', '榅'),
        ('熅', '煴'),
        ('縕', '緼'),
        ('膃', '腽'),
        ('轀', '輼'),
        ('鰮', '鰛'),
        ('蒕', '蒀'),
        ('蘊', '藴'),
        ('氳', '氲'),
        ('兌', '兑'),
        ('說', '説'),
        ('脫', '脱'),
        ('稅', '税'),
        ('悅', '悦'),
        ('挩', '捝'),
        ('敓', '敚'),
        ('梲', '棁'),
        ('涗', '涚'),
        ('蛻', '蜕'),
        ('銳', '鋭'),
        ('閱', '閲'),
        ('㨂', '揀'),
        ('錬', '鍊'),
        ('床', '牀'),
        ('羣', '群'),
        ('裡', '裏'),
        ('麵', '麪'),
        ('敎', '教'),
        ('祕', '秘'),
        ('巿', '市'),
        ('衆', '眾'),
        ('潨', '潀'),
        ('溼', '濕'),
        ('鷄', '雞'),
        ('吿', '告'),
        ('汙', '污'),
        ('洩', '泄'),
        ('駡', '罵'),
        ('銹', '鏽'),
        ('鉤', '鈎'),
        ('衛', '衞'),
        ('蔥', '葱'),
        ('艷', '豔'),
        ('葯', '藥'),
        ('滙', '匯'),
        ('啟', '啓'),
        ('奬', '獎'),
        ('俾', '畀'),
        ('我地', '我哋'),
        ('你地', '你哋'),
        ('佢地', '佢哋'),
        ('人地', '人哋'),
        ('爹地', '爹哋'),
        ('妳', '你'),
        ('您', '你'),
        ('癐', '攰'),
        ('倆', '兩'),
        ('咧', '呢'),
        ('噶', '㗎'),
    ]

    s = _resub(text, regex_list)
    s.translate(REMOVE_CHARS_TRANS)

    return s

def normalize_segment_text(segment: "SingleSegment") -> "SingleSegment":
    """Return a copy of segment with text converted to HK traditional Chinese."""
    normalized = dict(segment)
    normalized['text'] = standardize_chars_hk(simplified_to_traditional(segment['text']))
    return normalized


def is_non_chinese(char):
    "Returns true if char contains only alphanumeric chars."
    return re.match(r'[A-Za-z\d]', char)

def is_punctuation(char):
    "Returns true if char contains only Chinese punctuation chars."
    return re.match(r'[，？！…：；\s\-]', char)

def is_mergeable(text1: str, text2: str) -> bool:
    "Returns true if text1 and text2 can be acceptably merged into a single line."
    if len(text1) == 0 or len(text2) == 0:
        return True

    if text1[-1] not in SPLIT_CHARS or text1[-1] in MERGEABLE_CHARS:
        if len(text1 + text2) <= MAX_CHARS:
            return True

    return False

def is_removable(text: str) -> bool:
    "Returns true if the subtitle line text can be removed completely (used for interjections and meaningless text)."
    if len(text) == 0:
        return True

    if text in REMOVE_STANDALONE_CHARS:
        return True

    return False

def _locate_particles(sentence: str) -> Tuple[Tuple[int, int], str]:
    particle_start_i = next((i+1 for i in range(len(sentence)-1, -1, -1) if sentence[i] not in PARTICLE_CHARS), 0)
    particle = sentence[particle_start_i:]

    if particle == "":
        return None

    return ((particle_start_i, len(sentence)), sentence[particle_start_i:])

def locate_particles(text: str) -> List[Tuple[Tuple[int, int], str]]:
    SPLIT_CHARS = ["，", "。", "？", "！", "；", "…"]
    split_indexes = [i for i, char in enumerate(text) if char in SPLIT_CHARS]
    sentence_spans = []

    cur_start = 0

    for val in split_indexes:
        sentence_spans.append((cur_start, val))
        cur_start = val + 1

    if cur_start <= len(text):
        sentence_spans.append((cur_start, len(text)))

    particle_locations = []

    for x, y in sentence_spans:
        sentence = text[x:y]
        t = _locate_particles(sentence)
        if t:
            particle_locations.append( ((t[0][0] + x, t[0][1] + x), t[1]) )

    return particle_locations

def get_particle_candidates(p: str, particle_map: Dict[str, List[str]] = QWEN_PARTICLE_MAP) -> List[str]:
    """Return a list of possible substitute particle candidates given an input set of particles."""
    return particle_map.get(p, [p])

def get_particles(sentence: str) -> str:
    """Get a string of particles from the end of the sentence"""
    particle_start_i = next((i+1 for i in range(len(sentence)-1, -1, -1) if sentence[i] not in PARTICLE_CHARS), 0)

    return sentence[particle_start_i:]

def separate_particles(sentence: str) -> Tuple[str, str]:
    "Extract a tuple (x, y) from the sentence, where x = pre-particle chars and y = particle chars."
    particle_start_i = next((i+1 for i in range(len(sentence)-1, -1, -1) if sentence[i] not in PARTICLE_CHARS), 0)

    return(sentence[:particle_start_i], sentence[particle_start_i:])

def particle_candidates(sentence: str, particle_map: Dict[str, List[str]] = QWEN_PARTICLE_MAP) -> List[str]:
    "Take a transcribed sentence and return a list of corrected candidate sentences and final particles."
    s, p = separate_particles(sentence)

    candidate_p = particle_map.get(p, [p])
    return [s + px for px in candidate_p]

def separate_particle_candidates(sentence: str, particle_map: Dict[str, List[str]] = QWEN_PARTICLE_MAP) -> List[Tuple[str, str]]:
    "Take a transcribed sentence and return a list of tuples containing corrected candidate sentences and candidate final particles."
    s, p = separate_particles(sentence)

    candidate_p = particle_map.get(p, [p])
    return [(s, px) for px in candidate_p]

def preprocess_line(text: str) -> str:
    return standardize_chars_hk(str)

def postprocess_line(text: str) -> str:
    return str

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
    '喇': ['喇', '啦'],
    '啦': ['喇', '啦'],
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

SPLIT_CHARS = ["，", "。", "？", "！", "；", "…"]
MERGEABLE_CHARS = ["，"]
REMOVE_STANDALONE_CHARS = ["噢", "嗯", "哦", "嘩", "嗌", "唉", "誒", "哎", "啊", "嘿", "吓"]

def simplified_to_traditional(text: str) -> str:
    return cc.convert(text)

def standardize_chars_hk(text: str) -> str:
    """Convert character variants to the Hong Kong standard forms (rules/chars_hk.toml)."""
    from cantocaptions_ai.cantonese.rules import apply_ruleset, get_builtin_ruleset
    return apply_ruleset(text, get_builtin_ruleset("chars_hk"))

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


"""
A utility library for Cantonese parsing of raw text
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cantocaptions_ai.utils.schema import SingleSegment

_OPENCC_DIR = Path(__file__).parent / "opencc"

MAX_CHARS = 18

# To understand how we process final particles, see the preprocess/postprocess particles function below
PARTICLE_CHARS = [
    '吖', '啊', '呀', '噃', '㗎', '嘅', '吓', '可', '嗬', '啩', '囖', '囉', '咯', '啦', '喇', '嘞', '呢', '咧', '哩', '嗎', '嘛', '咩', '𠻹', '喎', '啫', '唧'
]

SPLIT_CHARS = ["，", "。", "？", "！", "；", "…"]
MERGEABLE_CHARS = ["，"]
REMOVE_STANDALONE_CHARS = ["噢", "嗯", "哦", "嘩", "嗌", "唉", "誒", "哎", "啊", "嘿", "吓"]


# --- Per-model downstream config value objects ---
# These are plain data holders describing how a given ASR model's output should be
# treated downstream. They live here (not in pipeline/) so this module stays free of
# pipeline imports; pipeline/model_profiles.py bundles them per model.

@dataclass(frozen=True)
class TextNormalization:
    """Post-ASR text normalization to apply. Both steps default off (no-op)."""
    opencc_config: Optional[str] = None   # filename under cantonese/opencc/; None => skip OpenCC
    chars_hk: bool = False                # run the rules/chars_hk.toml HK-variant ruleset


@dataclass(frozen=True)
class PunctuationConfig:
    """Punctuation that drives sentence splitting, alignment token spacing and line merging."""
    split_chars: Tuple[str, ...] = tuple(SPLIT_CHARS)
    mergeable_chars: Tuple[str, ...] = tuple(MERGEABLE_CHARS)

    def sentence_spans(self, text: str) -> List[Tuple[int, int]]:
        """Return (start, end) index spans of text between split_chars (never mutates text)."""
        split_indexes = [i for i, ch in enumerate(text) if ch in self.split_chars]
        spans: List[Tuple[int, int]] = []
        cur_start = 0
        for val in split_indexes:
            spans.append((cur_start, val))
            cur_start = val + 1
        if cur_start <= len(text):
            spans.append((cur_start, len(text)))
        return spans


@dataclass(frozen=True)
class SpotCheck:
    """A set of interchangeable candidate characters for one source char, with optional
    additive log-prob biases applied on top of the acoustic score during alignment."""
    candidates: Tuple[str, ...]                              # incl. the source char
    weights: Mapping[str, float] = field(default_factory=dict)


DEFAULT_NORMALIZATION = TextNormalization()
DEFAULT_PUNCTUATION = PunctuationConfig()


@lru_cache(maxsize=None)
def _get_opencc(config_name: str):
    """Build (and cache) an OpenCC converter for a config file under cantonese/opencc/."""
    from opencc import OpenCC
    return OpenCC(str(_OPENCC_DIR / config_name))

def simplified_to_traditional(text: str, config_name: str = "s2t_c.json") -> str:
    return _get_opencc(config_name).convert(text)

def standardize_chars_hk(text: str) -> str:
    """Convert character variants to the Hong Kong standard forms (rules/chars_hk.toml)."""
    from cantocaptions_ai.cantonese.rules import apply_ruleset, get_builtin_ruleset
    return apply_ruleset(text, get_builtin_ruleset("chars_hk"))

def normalize_segment_text(
    segment: "SingleSegment", normalization: TextNormalization = DEFAULT_NORMALIZATION,
) -> "SingleSegment":
    """Return a copy of segment with the configured post-ASR text normalization applied.

    With the default (no-op) normalization the text is returned unchanged — models whose
    output already meets the target convention skip OpenCC/HK-variant rewriting entirely.
    """
    normalized = dict(segment)
    text = segment['text']
    if normalization.opencc_config:
        text = simplified_to_traditional(text, normalization.opencc_config)
    if normalization.chars_hk:
        text = standardize_chars_hk(text)
    normalized['text'] = text
    return normalized


def is_non_chinese(char):
    "Returns true if char contains only alphanumeric chars."
    return re.match(r'[A-Za-z\d]', char)

def is_punctuation(char):
    "Returns true if char contains only Chinese punctuation chars."
    return re.match(r'[，？！…：；\s\-]', char)

def is_mergeable(text1: str, text2: str, punctuation: PunctuationConfig = DEFAULT_PUNCTUATION) -> bool:
    "Returns true if text1 and text2 can be acceptably merged into a single line."
    if len(text1) == 0 or len(text2) == 0:
        return True

    if text1[-1] not in punctuation.split_chars or text1[-1] in punctuation.mergeable_chars:
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
    particle_locations = []

    for x, y in DEFAULT_PUNCTUATION.sentence_spans(text):
        sentence = text[x:y]
        t = _locate_particles(sentence)
        if t:
            particle_locations.append( ((t[0][0] + x, t[0][1] + x), t[1]) )

    return particle_locations

def get_particles(sentence: str) -> str:
    """Get a string of particles from the end of the sentence"""
    particle_start_i = next((i+1 for i in range(len(sentence)-1, -1, -1) if sentence[i] not in PARTICLE_CHARS), 0)

    return sentence[particle_start_i:]

def separate_particles(sentence: str) -> Tuple[str, str]:
    "Extract a tuple (x, y) from the sentence, where x = pre-particle chars and y = particle chars."
    particle_start_i = next((i+1 for i in range(len(sentence)-1, -1, -1) if sentence[i] not in PARTICLE_CHARS), 0)

    return(sentence[:particle_start_i], sentence[particle_start_i:])


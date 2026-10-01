"""
Cantonese text utilities: OpenCC / HK-variant normalization, final particles, noise lines.

Part of the yue language pack (``languages/yue``). The generic value types that used to be
defined here now live in ``cantocaptions_ai/text_profiles.py`` and are re-exported below.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cantocaptions_ai.utils.schema import SingleSegment

_OPENCC_DIR = Path(__file__).parent / "opencc"  # == paths.OPENCC_DIR

MAX_CHARS = 18

# To understand how we process final particles, see the preprocess/postprocess particles function below
PARTICLE_CHARS = [
    '吖', '啊', '呀', '噃', '㗎', '嘅', '吓', '可', '嗬', '啩', '囖', '囉', '咯', '啦', '喇', '嘞', '呢', '咧', '哩', '嗎', '嘛', '咩', '𠻹', '喎', '啫', '唧'
]

REMOVE_STANDALONE_CHARS = ["噢", "嗯", "哦", "嘩", "嗌", "唉", "誒", "哎", "啊", "嘿", "吓"]

# The generic value types moved to cantocaptions_ai/text_profiles.py; they are re-exported
# here so existing imports keep working.
from cantocaptions_ai.text_profiles import (  # noqa: E402,F401
    CJK_MERGEABLE_CHARS,
    CJK_SPLIT_CHARS,
    DEFAULT_CLEANING,
    DEFAULT_NORMALIZATION,
    DEFAULT_PUNCTUATION,
    DEFAULT_SEGMENTATION,
    CleaningConfig,
    PunctuationConfig,
    SegmentationConfig,
    SpotCheck,
    TextNormalization,
    boundary_is_mergeable,
    is_mergeable,
)

SPLIT_CHARS = list(CJK_SPLIT_CHARS)
MERGEABLE_CHARS = list(CJK_MERGEABLE_CHARS)


@lru_cache(maxsize=None)
def _get_opencc(config_name: str):
    """Build (and cache) an OpenCC converter for a config file under languages/yue/opencc/."""
    from opencc import OpenCC
    return OpenCC(str(_OPENCC_DIR / config_name))

def simplified_to_traditional(text: str, config_name: str = "s2t_c.json") -> str:
    return _get_opencc(config_name).convert(text)

def standardize_chars_hk(text: str) -> str:
    """Convert character variants to the Hong Kong standard forms (rules/chars_hk.toml)."""
    from cantocaptions_ai.cleaning.rules import apply_ruleset, load_ruleset_cached
    from cantocaptions_ai.languages.yue.paths import RULES_DIR
    return apply_ruleset(text, load_ruleset_cached(RULES_DIR / "chars_hk.toml"))

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

def is_removable(text: str, noise_tokens=REMOVE_STANDALONE_CHARS) -> bool:
    "Returns true if the subtitle line text can be removed completely (used for interjections and meaningless text)."
    return len(text) == 0 or text in noise_tokens

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


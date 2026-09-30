"""How a language and a model write text, as plain value types.

Everything downstream of ASR that depends on the *writing* rather than the audio -- where
sentences split, which punctuation lets two cues join, what joins two pieces of text, how
long a subtitle line is, how a long line is broken -- is described by the frozen
dataclasses here, and the stages take them as arguments instead of assuming Cantonese.
``pipeline/model_profiles.py`` pins them per ASR model; ``ModelProfile.for_language``
fills in whatever a profile leaves unset from the defaults for the run's language.

This module is deliberately free of any ``cantonese`` or ``pipeline`` import: it is the
vocabulary both of those speak, not an implementation of either.
"""
import re
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

# Languages written without spaces between words. Mirrors utils.output's list, which
# still serves the whisperX-era writers; this is the one the text conventions follow.
LANGUAGES_WITHOUT_SPACES = ("ja", "zh", "yue")

CJK_SPLIT_CHARS = ("，", "。", "？", "！", "；", "…")
CJK_MERGEABLE_CHARS = ("，",)
LATIN_SPLIT_CHARS = (",", ".", "?", "!", ";", "…")
LATIN_MERGEABLE_CHARS = (",",)


@dataclass(frozen=True)
class TextNormalization:
    """Post-ASR text normalization to apply. Both steps default off (no-op)."""
    opencc_config: Optional[str] = None   # filename under cantonese/opencc/; None => skip OpenCC
    chars_hk: bool = False                # run the rules/chars_hk.toml HK-variant ruleset


@dataclass(frozen=True)
class PunctuationConfig:
    """Punctuation that drives sentence splitting, alignment token spacing and line merging."""
    split_chars: Tuple[str, ...] = CJK_SPLIT_CHARS
    mergeable_chars: Tuple[str, ...] = CJK_MERGEABLE_CHARS

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


CJK_PUNCTUATION = PunctuationConfig()
LATIN_PUNCTUATION = PunctuationConfig(LATIN_SPLIT_CHARS, LATIN_MERGEABLE_CHARS)


@dataclass(frozen=True)
class ScriptConfig:
    """How a writing system lays text out.

    ``word_separator`` joins two pieces of text that were split apart (two cues merged back
    into one, two words of a line): nothing for CJK, a space for space-separated scripts.
    It also decides whether alignment treats a space as a word boundary. ``line_width`` is
    the default subtitle line length in characters, used where no ``max_line_width`` is set.
    ``layout`` names the line breaker (see ``cantonese.cleaner.LAYOUTS``): ``"cjk"`` breaks
    at punctuation or between words found by segmentation; ``"word"`` breaks at a space.
    """
    word_separator: str = ""
    line_width: int = 18
    layout: str = "cjk"

    @property
    def spaced(self) -> bool:
        return bool(self.word_separator)

    def join(self, left: str, right: str) -> str:
        """Join two pieces of text the way this script writes them."""
        if not self.word_separator or not left or not right:
            return left + right
        if left.endswith(self.word_separator) or right.startswith(self.word_separator):
            return left + right
        return left + self.word_separator + right


CJK_SCRIPT = ScriptConfig()
SPACED_SCRIPT = ScriptConfig(word_separator=" ", line_width=42, layout="word")


def script_for_language(language: Optional[str]) -> ScriptConfig:
    """The default script for a language code: CJK unless it separates words with spaces."""
    return CJK_SCRIPT if language in LANGUAGES_WITHOUT_SPACES else SPACED_SCRIPT


def punctuation_for_script(script: ScriptConfig) -> PunctuationConfig:
    return LATIN_PUNCTUATION if script.spaced else CJK_PUNCTUATION


@dataclass(frozen=True)
class SpotCheck:
    """A set of interchangeable candidate characters for one source char, with optional
    additive log-prob biases applied on top of the acoustic score during alignment."""
    candidates: Tuple[str, ...]                              # incl. the source char
    weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentationConfig:
    """Model-specific discourse markers that introduce the clause following them.

    Markers like 嗱/哎吔/喂 are punctuated off by the ASR as their own clause, but acoustically
    they sit inside a continuous speech run, so forced alignment collapses them to a few
    frames and they surface as sub-100 ms cues. Cue assembly (``pipeline/segmentation.py``)
    uses this list to rejoin them *forwards*, onto the sentence they introduce, rather than
    stranding them on the end of the previous one.

    Only list tokens that reliably *lead*. Final particles (呀, 啦, 吓) are ambiguous: they
    attach backwards just as often, so they are left to the generic duration-based rescue
    instead. Empty by default (a no-op), like every other model-profile field.
    """
    leading_markers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CleaningConfig:
    """Which cleaning step manifest this model's output should be folded through.

    ``manifest`` names a file in the rules directory (the packaged ``cantonese/rules/``,
    or a ``--clean_rules_dir`` override). The default manifest is conservative --
    character variants, punctuation, noise and line layout -- because a model fine-tuned to emit
    the target convention already writes what the heavier rules exist to impose.

    A model that writes generic Mandarin-flavoured output needs the full legacy chain
    instead (``pipeline_qwen.toml``): question particles, ASR error repair, numeral
    conversion and particle conventions on top. Like every other model-profile field,
    the default is the no-op-ish one and a model opts *in* to more work.
    """
    manifest: str = "pipeline.toml"


DEFAULT_NORMALIZATION = TextNormalization()
DEFAULT_PUNCTUATION = CJK_PUNCTUATION
DEFAULT_SEGMENTATION = SegmentationConfig()
DEFAULT_CLEANING = CleaningConfig()
DEFAULT_SCRIPT = CJK_SCRIPT


def boundary_is_mergeable(text1: str, punctuation: PunctuationConfig = DEFAULT_PUNCTUATION) -> bool:
    """Returns true if a line ending in ``text1`` reads acceptably joined to what follows.

    This is the punctuation half of :func:`is_mergeable`, split out so the cue-assembly
    passes can reuse one definition of the rule: the adjacency merge applies it as a hard
    gate, while the short-cue rescue applies it only to *rank* the two possible join
    directions (see ``pipeline/segmentation.py``).
    """
    if len(text1) == 0:
        return True

    return text1[-1] not in punctuation.split_chars or text1[-1] in punctuation.mergeable_chars


def is_mergeable(
    text1: str,
    text2: str,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    max_chars: int = DEFAULT_SCRIPT.line_width,
    script: ScriptConfig = DEFAULT_SCRIPT,
) -> bool:
    "Returns true if text1 and text2 can be acceptably merged into a single line."
    if len(text1) == 0 or len(text2) == 0:
        return True

    if boundary_is_mergeable(text1, punctuation):
        if len(script.join(text1, text2)) <= max_chars:
            return True

    return False


_SPACE_RUN = re.compile(r"\s+")


def word_wrap(text: str, line_max_length: int) -> str:
    """Break a space-separated line in two at the space nearest its middle.

    The ``"word"`` layout's counterpart to the CJK line breaker: never splits a word, never
    re-breaks text that already holds a line break, and leaves a line that fits (or has no
    space to break at) alone.
    """
    if "\n" in text or len(text) <= line_max_length:
        return text
    spaces = [m.start() for m in _SPACE_RUN.finditer(text)]
    if not spaces:
        return text
    middle = len(text) / 2
    best = min(spaces, key=lambda i: (len(text[:i]) > line_max_length, abs(i - middle)))
    return text[:best].rstrip() + "\n" + text[best:].lstrip()

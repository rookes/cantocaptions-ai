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
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Mapping, Optional, Tuple

# Languages written without spaces between words. Mirrors utils.output's list, which
# still serves the whisperX-era writers; this is the one the text conventions follow.
LANGUAGES_WITHOUT_SPACES = ("ja", "zh", "yue")

# One word of Latin-script text (or a number) written inside text that has no spaces between
# its own words: code-switched English in a Cantonese line. Alignment keeps each match as one
# word, so nothing that cuts a cue between words can cut inside it, and the line breaker never
# breaks inside one. An apostrophe, hyphen or point joins two runs (you've, open-minded, 3.5).
# Accented Latin letters belong to the word (café, Pokémon, the Italian può): left out, each
# was a word of its own, and a cue could be cut between "pu" and "ò".
_LATIN_LETTERS = "A-Za-z0-9\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff"
LATIN_WORD = re.compile(rf"[{_LATIN_LETTERS}]+(?:['’\-.][{_LATIN_LETTERS}]+)*")
# Where ScriptConfig.join spaces two runs of Latin text: one ends in a letter, digit or ASCII
# clause mark, the other starts with a letter or digit.
_LATIN_END = re.compile(rf"[{_LATIN_LETTERS},.?!;:'\"]$")
_LATIN_START = re.compile(rf"[{_LATIN_LETTERS}]")


@lru_cache(maxsize=4096)
def display_columns(char: str) -> int:
    """How many display columns *char* takes: 2 for a full-width character, 1 for a half-width one.

    Half-width is what ``east_asian_width`` calls narrow (ASCII letters, digits, space and
    punctuation) or half-width, plus a non-wide Latin letter or digit (é). Ambiguous-width
    punctuation (…, ·) counts as full: a CJK font draws it full-width, and counting it so is
    what keeps a line of Chinese measuring exactly its length in characters.
    """
    eaw = unicodedata.east_asian_width(char)
    if eaw in ("W", "F"):
        return 2
    if eaw in ("Na", "H"):
        return 1
    return 1 if unicodedata.category(char)[0] in "LN" else 2

CJK_SPLIT_CHARS = ("，", "。", "？", "！", "；", "…")
CJK_MERGEABLE_CHARS = ("，",)
LATIN_SPLIT_CHARS = (",", ".", "?", "!", ";", "…")
LATIN_MERGEABLE_CHARS = (",",)


@dataclass(frozen=True)
class TextNormalization:
    """Post-ASR text normalization to apply. Both steps default off (no-op)."""
    opencc_config: Optional[str] = None   # filename under languages/yue/opencc/; None => skip OpenCC
    chars_hk: bool = False                # run the rules/chars_hk.toml HK-variant ruleset


@dataclass(frozen=True)
class PunctuationConfig:
    """Punctuation that drives sentence splitting, alignment token spacing and line merging."""
    split_chars: Tuple[str, ...] = CJK_SPLIT_CHARS
    mergeable_chars: Tuple[str, ...] = CJK_MERGEABLE_CHARS
    # Sentence ends speech has and a transcript may leave unwritten: Cantonese final particles
    # (啊, 喇, 㗎). A cue cutter that finds a pause right after one reads it as the end of an
    # utterance even with no punctuation there (see ends_utterance). Empty -- any language
    # whose pack sets none -- means the text is no evidence either way.
    final_particles: Tuple[str, ...] = ()
    # Characters that continue a particle cluster (㗎咩, 先啦吓) but are ordinary words elsewhere
    # (咩事, 呢個), so a final particle followed by one is not known to end anything.
    particle_continuations: Tuple[str, ...] = ()
    # Words with a final particle inside them that the word check does not know (無啦啦, 喇叭).
    particle_words: Tuple[str, ...] = ()

    def ends_utterance(self, text: str, at: int) -> bool:
        """Whether ``text[:at]`` ends on a final particle that nothing after it continues: the
        boundary 喺垃圾桶啊|復原光線 has, written or not. Only the text's half of the question --
        whether the speaker paused there is the caller's to measure; a right-dislocated
        做咩啊|你 passes here and has no pause.
        """
        if not self.final_particles or not 0 < at < len(text):
            return False
        last, nxt = text[at - 1], text[at]
        if last not in self.final_particles or nxt.isspace() or nxt in self.split_chars:
            return False
        if nxt in self.final_particles or nxt in self.particle_continuations:
            return False
        if at >= 2 and text[at - 2] == last:   # 啦啦, 啊啊: a noise, not a sentence end
            return False
        return not any(text.startswith(word, lo)
                       for word in self.particle_words
                       for lo in range(max(0, at - len(word) + 1), at)
                       if lo + len(word) > at)

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
    the default subtitle line length, used where no ``max_line_width`` is set, in this
    script's own character cells (see :meth:`width`).
    ``layout`` names the line breaker (see ``cantonese.cleaner.LAYOUTS``): ``"cjk"`` breaks
    at punctuation or between words found by segmentation; ``"word"`` breaks at a space.
    ``cell_columns`` is how many display columns one cell is: 2 for CJK, whose cell is one
    Chinese character (so an English letter counts half), 1 for a space-separated script,
    whose cell is one letter (so a Chinese character counts two).
    """
    word_separator: str = ""
    # 17.5, not 18: the CantoCaptions style guide's "single line max length is 17.5
    # characters" -- 18 full-width characters is one too many, 17 plus a half-width one fits.
    line_width: float = 17.5
    layout: str = "cjk"
    cell_columns: int = 2

    @property
    def spaced(self) -> bool:
        return bool(self.word_separator)

    def width(self, text: str) -> float:
        """How wide *text* is on screen, in this script's cells (what ``line_width`` counts)."""
        return sum(display_columns(c) for c in text) / self.cell_columns

    def inside_word(self, text: str, at: int) -> bool:
        """Whether cutting *text* before ``text[at]`` would split a word (see
        ``cleaning.layout.WORD_CHECKS``, which this script's ``layout`` picks from)."""
        from cantocaptions_ai.cleaning.layout import WORD_CHECKS
        check = WORD_CHECKS.get(self.layout)
        return check(text, at) if check is not None else False

    def join(self, left: str, right: str) -> str:
        """Join two pieces of text the way this script writes them.

        A script without spaces still spaces two runs of Latin text it joins: merging the cues
        "So, do you believe him?" and "Naturally" must not give "him?Naturally". Latin beside
        Chinese stays unspaced, as written (我嚟到cafe喇).
        """
        if not left or not right:
            return left + right
        if not self.word_separator:
            if _LATIN_END.search(left) and _LATIN_START.match(right):
                return left + " " + right
            return left + right
        if left.endswith(self.word_separator) or right.startswith(self.word_separator):
            return left + right
        return left + self.word_separator + right


CJK_SCRIPT = ScriptConfig()
SPACED_SCRIPT = ScriptConfig(word_separator=" ", line_width=42, layout="word", cell_columns=1)


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

    ``manifest`` names a file in the rules directory (the language pack's, e.g. ``languages/yue/rules/``,
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
    max_chars: float = DEFAULT_SCRIPT.line_width,
    script: ScriptConfig = DEFAULT_SCRIPT,
) -> bool:
    """Returns true if text1 and text2 can be acceptably merged into a single line.

    ``max_chars`` is a width in the script's cells (:meth:`ScriptConfig.width`).
    """
    if len(text1) == 0 or len(text2) == 0:
        return True

    if boundary_is_mergeable(text1, punctuation):
        if script.width(script.join(text1, text2)) <= max_chars:
            return True

    return False


_SPACE_RUN = re.compile(r"\s+")


def word_wrap(text: str, line_max_length: int) -> str:
    """Break a space-separated line in two at the space nearest its middle.

    The ``"word"`` layout's counterpart to the CJK line breaker: never splits a word, never
    re-breaks text that already holds a line break, and leaves a line that fits (or has no
    space to break at) alone.
    """
    width = SPACED_SCRIPT.width
    if "\n" in text or width(text) <= line_max_length:
        return text
    spaces = [m.start() for m in _SPACE_RUN.finditer(text)]
    if not spaces:
        return text
    middle = width(text) / 2
    best = min(spaces, key=lambda i: (width(text[:i]) > line_max_length,
                                      abs(width(text[:i]) - middle)))
    return text[:best].rstrip() + "\n" + text[best:].lstrip()

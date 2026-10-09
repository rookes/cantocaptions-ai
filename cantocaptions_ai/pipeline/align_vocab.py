"""Give the align model a token for characters its vocabulary has no entry for.

A CTC align model can only place a character it holds a token for. ``_preprocess_segment``
drops everything else, so an out-of-vocabulary character contributes *no evidence at all* --
it is not mistimed, it is absent -- and a line made only of such characters cannot be timed
from its own audio (``--realign`` reports those as ``no_vocabulary``). Measured on the Police
Story 2 transcript against ``alvanlii/wav2vec2-BERT-cantonese``: 326 characters over 88
distinct ones, and 17 whole lines with nothing alignable in them.

Two substitutions recover almost all of it, tried in this order:

1. **A variant fold.** 辉, 师, 岛 are Simplified forms of characters the model does know.
   Nothing acoustic is wrong here; the transcript was typed with the wrong character set.
2. **A homophone.** 駒 (keoi1) is absent while 區 (keoi1) is present, and to an acoustic
   model those are the same syllable.

Both need to know the language -- its variant forms, how its characters sound -- which its
language pack supplies (``LanguagePack.char_readings``; Cantonese's is Jyutping, in
``languages/yue/readings.py``). A language without one gets only the overrides and the align
model's own substitution table.

**This is not a text edit.** The original character stays in ``clean_char`` and therefore in
the subtitle -- exactly the way punctuation keeps its own character while being tokenised as
blank. Only the token id changes.

The mechanism is the dictionary itself: ``VocabRepair.augment`` adds ``original -> token id
of the replacement`` to the align dictionary. Every consumer then picks it up with no
threading and no way for the coarse search and the final alignment to disagree about what is
alignable -- ``_preprocess_segment``'s membership test, ``_align_segment``'s token lookup and
``realign.line_tokens`` all read that one dict. Nothing else in the pipeline needs to know
this happened.

**Digits and punctuation are deliberately out of scope.** 8 has a Cantonese reading (baat3)
and would substitute happily, but a digit in a transcript may be read in Cantonese, in
English, or digit by digit, and nothing in the text says which. Punctuation the aligner wants
is already mapped to blank by ``PunctuationConfig.split_chars``; a cue holding nothing else is
noise and is dropped downstream. Only letters are repaired.

Measured coverage on Police Story 2 (76 distinct CJK characters, 249 occurrences):

| level | distinct resolved | occurrences | lines left unalignable |
|---|---|---|---|
| ``off`` | 0 | 0 | 17 |
| ``variant`` | 4 | 4 | 17 |
| ``homophone`` | 57 | 217 | 1 |
| ``near`` | 69 | 240 | 0 |
"""
import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from cantocaptions_ai.languages.base import CharReadings

logger = logging.getLogger(__name__)

# --- Levels ---------------------------------------------------------------------------

LEVEL_OFF = "off"
LEVEL_VARIANT = "variant"
LEVEL_HOMOPHONE = "homophone"
LEVEL_NEAR = "near"

# Ordered weakest to strongest; a level enables every tier at or below it.
LEVELS: Tuple[str, ...] = (LEVEL_OFF, LEVEL_VARIANT, LEVEL_HOMOPHONE, LEVEL_NEAR)

KIND_OVERRIDE = "override"
KIND_VARIANT = "variant"
KIND_HOMOPHONE = "homophone"
KIND_NEAR = "near-homophone"

# What a note on a cue means, for the run summary and for anyone reading notes.srt.
KIND_HELP = {
    KIND_OVERRIDE: "substituted by hand from --align_substitutions",
    KIND_VARIANT: "a Simplified or variant form of a character the model knows",
    KIND_HOMOPHONE: "substituted by a character with the same reading",
    KIND_NEAR: "substituted by a character with the same syllable, different tone",
}

# A character the model has no token for and nothing to substitute. The line is still placed
# -- from its neighbours -- but this character contributed nothing to placing it.
NOTE_NO_TOKEN = "no_token"


@dataclass(frozen=True)
class Substitution:
    """One character standing in for another, during alignment only."""

    original: str
    replacement: str
    kind: str
    reading: str = ""

    def note(self) -> str:
        """The per-cue annotation, e.g. ``homophone:駒→區``."""
        return f"{self.kind}:{self.original}→{self.replacement}"


@dataclass
class RepairReport:
    """What one ``augment`` call changed, for logging and for the caller's summary."""

    substitutions: Dict[str, Substitution] = field(default_factory=dict)
    unresolved: "Counter[str]" = field(default_factory=Counter)
    occurrences: "Counter[str]" = field(default_factory=Counter)

    @property
    def unresolved_occurrences(self) -> int:
        return sum(self.unresolved.values())

    def by_kind(self) -> "Counter[str]":
        return Counter(s.kind for s in self.substitutions.values())


def _repairable(char: str) -> bool:
    """Only letters. See the module docstring on digits and punctuation."""
    return (
        len(char) == 1
        and not char.isascii()
        and unicodedata.category(char).startswith("L")
    )


# --- The repair -----------------------------------------------------------------------

class VocabRepair:
    """Substitutions for one align dictionary, resolved on demand and applied to it.

    Built once per align-model load and shared by every stage that tokenises text against
    that dictionary, so ``--realign``'s coarse search and the final alignment can never
    disagree about which characters are alignable.
    """

    def __init__(
        self,
        dictionary: Dict[str, int],
        level: str = LEVEL_HOMOPHONE,
        overrides: Optional[Mapping[str, str]] = None,
        readings: Optional["CharReadings"] = None,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown substitution level {level!r}; expected one of {LEVELS}")
        self.dictionary = dictionary
        self.level = level
        self.overrides = dict(overrides or {})
        # The language's (languages.base.CharReadings). Without it the variant and homophone
        # tiers have nothing to go on, and only the overrides apply.
        self.readings = readings
        self.substitutions: Dict[str, Substitution] = {}
        self.unresolved: Dict[str, None] = {}
        self._exact: Optional[Dict[str, List[str]]] = None
        self._near: Optional[Dict[str, List[str]]] = None

    # -- candidate index --

    def _build_index(self) -> None:
        """Index the dictionary's own characters by reading. Lazy: only built if needed.

        Candidates are ranked by HKCanCor frequency, on the theory that the align model saw
        the commoner character more often and so holds a better-trained token for it, then by
        code point so that a run is reproducible.
        """
        freq = self.readings.frequencies()

        exact: Dict[str, List[str]] = {}
        near: Dict[str, List[str]] = {}
        for char in self.dictionary:
            if not _repairable(char):
                continue
            reading = self._reading(char)
            if not reading:
                continue
            exact.setdefault(reading, []).append(char)
            near.setdefault(self.readings.toneless(reading), []).append(char)
        for index in (exact, near):
            for reading in index:
                index[reading].sort(key=lambda c: (-freq.get(c, 0), ord(c)))
        self._exact, self._near = exact, near

    def _first_candidate(
        self, index: Optional[Dict[str, List[str]]], key: str, exclude: str,
    ) -> Optional[str]:
        for candidate in (index or {}).get(key, ()):
            if candidate != exclude:
                return candidate
        return None

    # -- resolution --

    def resolve(self, char: str) -> Optional[Substitution]:
        """Find a token the model does hold for *char*, or None if there is none."""
        if self.level == LEVEL_OFF or char in self.dictionary:
            return None

        forced = self.overrides.get(char)
        if forced is not None:
            # An empty override means "leave this one alone", so one bad automatic choice
            # can be switched off without switching off the feature.
            if forced and forced in self.dictionary:
                return Substitution(char, forced, KIND_OVERRIDE, self._reading(char) or "")
            if forced:
                logger.warning(
                    "Substitution override %r -> %r ignored: the align model has no token "
                    "for %r either.", char, forced, forced,
                )
            return None

        if not _repairable(char) or self.readings is None:
            return None

        # 1. A variant fold, which is not an acoustic substitution at all -- the transcript
        #    simply spelled a character the model knows in the wrong character set.
        variant = self._variant(char)
        if variant is not None and variant in self.dictionary:
            return Substitution(char, variant, KIND_VARIANT, self._reading(char) or "")
        if LEVELS.index(self.level) < LEVELS.index(LEVEL_HOMOPHONE):
            return None

        # 2. A homophone. Fall back to the *variant's* reading when the original has none:
        #    撺 is unknown to the reading data, but 攛 (cyun1) is not.
        reading = self._reading(char) or (self._reading(variant) if variant else None)
        if not reading:
            return None
        if self._exact is None:
            self._build_index()
        candidate = self._first_candidate(self._exact, reading, char)
        if candidate is not None:
            return Substitution(char, candidate, KIND_HOMOPHONE, reading)
        if LEVELS.index(self.level) < LEVELS.index(LEVEL_NEAR):
            return None

        # 3. The same syllable on a different tone. Weaker evidence, but the alternative is a
        #    character the trellis cannot see at all.
        candidate = self._first_candidate(self._near, self.readings.toneless(reading), char)
        if candidate is not None:
            return Substitution(char, candidate, KIND_NEAR, reading)
        return None

    def _reading(self, char: str) -> Optional[str]:
        return self.readings.reading(char) if self.readings is not None else None

    def _variant(self, char: str) -> Optional[str]:
        return self.readings.variant(char) if self.readings is not None else None

    # -- application --

    def augment(self, texts: Iterable[str]) -> RepairReport:
        """Add a token for every unknown character in *texts*, in place on the dictionary.

        Idempotent and cumulative: a character resolved for an earlier file -- or an earlier
        stage of the same file -- is already in the dictionary and is not looked at again, so
        ``--realign``'s search and the alignment that follows it share one vocabulary.
        """
        report = RepairReport()
        if self.level == LEVEL_OFF:
            return report
        for text in texts:
            for char in text or "":
                if char in self.dictionary or not _repairable(char):
                    continue
                report.occurrences[char] += 1
                if char in self.substitutions or char in self.unresolved:
                    continue
                substitution = self.resolve(char)
                if substitution is None:
                    self.unresolved[char] = None
                    continue
                report.substitutions[char] = substitution
                self.substitutions[char] = substitution
        for char, count in report.occurrences.items():
            if char not in report.substitutions:
                report.unresolved[char] = count
        # Applied last so the loop above tests membership against the untouched dictionary.
        for char, substitution in report.substitutions.items():
            self.dictionary[char] = self.dictionary[substitution.replacement]
        _log_report(report)
        return report


def _log_report(report: RepairReport) -> None:
    if not report.occurrences:
        return
    kinds = report.by_kind()
    unresolved = ""
    if report.unresolved:
        unresolved = "; no substitute for " + " ".join(
            f"{c}x{n}" if n > 1 else c for c, n in report.unresolved.most_common())
    logger.info(
        "Align vocabulary: %d of %d character(s) the align model has no token for were "
        "substituted (%s)%s",
        len(report.substitutions), len(report.occurrences),
        ", ".join(f"{n} {kind}" for kind, n in kinds.most_common()) or "none", unresolved,
    )
    logger.debug(
        "Align vocabulary: %d occurrence(s) without a token, %d left unresolved",
        sum(report.occurrences.values()), report.unresolved_occurrences,
    )
    for char, substitution in sorted(
        report.substitutions.items(), key=lambda kv: -report.occurrences[kv[0]]
    ):
        logger.debug(
            "  %s -> %s (%s, %s) x%d", char, substitution.replacement, substitution.kind,
            substitution.reading or "no reading", report.occurrences[char],
        )


def load_substitution_overrides(path: str) -> Dict[str, str]:
    """Read a hand-curated ``[substitutions]`` table from a TOML file.

    ::

        [substitutions]
        "駒" = "區"      # use this instead of whatever the automatic search picks
        "摷" = ""        # ...or refuse to substitute this one at all

    Overrides beat every automatic tier, so one bad choice can be corrected without turning
    the feature off.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    table = data.get("substitutions", data)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: expected a [substitutions] table of character pairs")
    overrides: Dict[str, str] = {}
    for original, replacement in table.items():
        if not isinstance(replacement, str):
            raise ValueError(
                f"{path}: substitution for {original!r} must be a string "
                f'(use "" to leave the character alone)'
            )
        if len(original) != 1 or len(replacement) > 1:
            raise ValueError(
                f"{path}: substitutions map one character to one character, got "
                f"{original!r} -> {replacement!r}"
            )
        overrides[original] = replacement
    logger.debug("Loaded %d substitution override(s) from %s", len(overrides), path)
    return overrides


def filter_spotchecks(spotchecks: Mapping, substituted: Iterable[str]) -> Mapping:
    """Drop spot-check candidates whose token now belongs to a different character.

    A spot-check asks the emission which of two interchangeable particles the audio actually
    supports (喇/啦, 咁/噉). That question only means anything while each candidate's token is
    its *own* acoustics. A substituted candidate's token is some homophone's, so its score
    says nothing about the candidate -- and if two candidates in one set happened to
    substitute to the same token, their scores would be identical and a candidate weight
    would decide the rewrite on its own, silently.

    No spot-check character in ``MODEL_PROFILES`` is out of ``alvanlii/wav2vec2-BERT-cantonese``'s
    vocabulary today, so this never fires; it is here so that a future align model or profile
    cannot quietly turn an acoustic reselection into a coin toss. A check left with fewer than
    two candidates is dropped, which is what an empty spotchecks entry already means.
    """
    substituted = set(substituted)
    if not substituted or not spotchecks:
        return spotchecks
    out, dropped = {}, []
    for char, check in spotchecks.items():
        candidates = tuple(c for c in check.candidates if c not in substituted)
        if len(candidates) == len(check.candidates):
            out[char] = check
        elif len(candidates) > 1:
            out[char] = replace(check, candidates=candidates)
            dropped.append(char)
        else:
            dropped.append(char)
    if dropped:
        logger.warning(
            "Spot-check(s) for %s reduced or dropped: a candidate is only in the align "
            "vocabulary through a substitution, so its score is another character's.",
            " ".join(sorted(dropped)),
        )
    return out


@lru_cache(maxsize=None)
def bundled_substitutions(filename: str) -> Mapping[str, str]:
    """A substitution table shipped with the align model's profile, or empty if absent.

    Cached: the file is small and is read once per process, not once per file aligned.
    """
    from cantocaptions_ai.pipeline.align_profiles import SUBSTITUTIONS_DIR

    path = SUBSTITUTIONS_DIR / filename
    if not path.is_file():
        logger.warning("Align profile names a substitution table that is missing: %s", path)
        return {}
    return load_substitution_overrides(str(path))


def merge_substitutions(*layers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Later layers win per character, so a user file need not restate the bundled ones."""
    merged: Dict[str, str] = {}
    for layer in layers:
        if layer:
            merged.update(layer)
    return merged


def substitution_notes(
    text: str,
    substitutions: Mapping[str, Substitution],
    unresolved: Iterable[str] = (),
) -> List[str]:
    """Notes describing what happened to *text*'s unknown characters, in first-use order."""
    unknown = set(unresolved)
    notes: List[str] = []
    for char in dict.fromkeys(text or ""):
        substitution = substitutions.get(char)
        if substitution is not None:
            notes.append(substitution.note())
        elif char in unknown:
            notes.append(f"{NOTE_NO_TOKEN}:{char}")
    return notes

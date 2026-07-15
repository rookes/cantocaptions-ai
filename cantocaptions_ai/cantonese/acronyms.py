"""English acronym formatting (e.g. "A P P" / "A.P.P" -> "APP").

Runs before any other cleaning step (see rules/pipeline.toml) because later steps
(punctuation.toml) rewrite or strip "." entirely, which would destroy the very
separators this step needs to see.
"""

import re

# A capital-letter run: starts and ends on [A-Z], with only capitals/periods/
# whitespace (including newlines) allowed in between.
_ACRONYM_RUN = re.compile(r"[A-Z](?:[A-Z.\s]*[A-Z])?")

# Zero-width split point inside a run: "<CAP>. <CAP>" is treated as an
# end-of-sentence abbreviation marker (e.g. "P. B") rather than an acronym
# separator, so it splits the run instead of being collapsed away.
_SENTENCE_SPLIT = re.compile(r"(?<=[A-Z]\.\s)(?=[A-Z])")


def _collapse_subrun(subrun: str) -> str:
    """Collapse a single acronym candidate, or leave it untouched if it has < 2 capitals."""
    caps = [i for i, ch in enumerate(subrun) if "A" <= ch <= "Z"]
    if len(caps) < 2:
        return subrun
    return subrun[:caps[0]] + "".join(subrun[i] for i in caps) + subrun[caps[-1] + 1:]


def _replace(m: "re.Match[str]", text: str) -> str:
    s = m.group(0)
    if len(s) <= 1:
        return s

    start, end = m.start(), m.end()
    # A capital touching a lowercase English letter with no separator (e.g. the
    # "B" in "Bsitter") belongs to a mixed-case word, not an acronym.
    leading_trim = start > 0 and "a" <= text[start - 1] <= "z"
    trailing_trim = end < len(text) and "a" <= text[end] <= "z"

    core_start = 1 if leading_trim else 0
    core_end = len(s) - 1 if trailing_trim else len(s)
    if core_start >= core_end:
        return s

    core = s[core_start:core_end]
    processed = "".join(_collapse_subrun(sr) for sr in _SENTENCE_SPLIT.split(core))
    return s[:core_start] + processed + s[core_end:]


def format_acronyms(text: str) -> str:
    """Collapse runs of 2+ capital letters into a single acronym token.

    Internal "." and whitespace (including newlines) between the capitals are
    removed. Any other character (comma, Chinese punctuation, etc.) breaks a run
    apart. A capital directly touching a lowercase English letter is excluded
    from consideration. A "<CAP>. <CAP>" boundary is preserved as an
    end-of-sentence marker rather than collapsed across.
    """
    return _ACRONYM_RUN.sub(lambda m: _replace(m, text), text)

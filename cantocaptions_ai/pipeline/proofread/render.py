"""The proofreading request: one system prompt and one cue stream per episode (or chunk).

The cue stream is plain numbered text rather than JSON -- one line per cue, ``id|text`` --
which costs a fraction of the tokens of an array of objects and is just as unambiguous. The
reference subtitle is interleaved by time, each reference line directly after the draft cue
it overlaps most, so the model never has to align two lists itself. Gaps between cues are
drawn as ``--`` (a pause) and ``==`` (a long gap, often a scene change), which is all the
timing a proofreader needs.

Nothing here knows any language: the language and its standard arrive as a
``ProofreadStandard`` and are spliced into the stage's own prompt (``prompt.md``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from cantocaptions_ai.languages.base import ProofreadStandard

PROMPT_FILE = Path(__file__).with_name("prompt.md")
PAUSE_S = 1.0     # a gap at least this long is drawn as "--"
SCENE_S = 3.0     # ...and at least this long as "=="

EDIT_TYPES = ["mishearing", "function_word", "missing", "extra", "punctuation", "spelling",
              "name", "other"]

# ``names`` comes first on purpose: generation follows the schema's property order, so the
# model settles every proper noun before it writes a single edit.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "use": {"type": "string"},
                    "draft_forms": {"type": "array", "items": {"type": "string"}},
                    "reference": {"type": "string"},
                },
                "required": ["use", "draft_forms", "reference"],
                "additionalProperties": False,
            },
        },
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": EDIT_TYPES},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["id", "text", "type", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "note": {"type": "string"}},
                "required": ["id", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["names", "edits", "flags"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Cue:
    """One draft cue as the proofreader sees it. ``id`` is 1-based within the file."""
    id: int
    start: float
    end: float
    text: str
    editable: bool = True


@dataclass(frozen=True)
class Request:
    system: str
    user: str


def system_prompt(standard: ProofreadStandard, template: Optional[str] = None,
                  conventions: Optional[str] = None, particles: str = "protect") -> str:
    """The stage's prompt with *standard* spliced in.

    ``template`` replaces the stage's own ``prompt.md`` and ``conventions`` the standard's
    conventions file -- the two user overrides. A template may use any of the placeholders
    or none of them. ``particles`` (``protect`` / ``allow``) picks which of the standard's
    two particle bullets is shown: the rule to leave them alone, or the error description.
    """
    text = template if template is not None else PROMPT_FILE.read_text(encoding="utf-8")
    if conventions is None and standard.conventions is not None:
        conventions = Path(standard.conventions).read_text(encoding="utf-8")
    protect = particles == "protect"
    errors = [standard.error_examples.rstrip()]
    if not protect and standard.final_particle_errors:
        errors.append(standard.final_particle_errors.rstrip())
    not_errors = standard.final_particle_rule.rstrip() if protect else ""
    parts = {
        "{LANGUAGE_NAME}": standard.language_name,
        "{DESCRIPTION}": ("\n" + standard.description.strip() + "\n") if standard.description else "",
        "{ERROR_EXAMPLES}": "\n".join(e for e in errors if e),
        "{NOT_ERRORS}": ("\n" + not_errors) if not_errors else "",
        "{NAME_EXAMPLE}": f" ({standard.name_example})" if standard.name_example else "",
        "{CONVENTIONS}": ("\n" + conventions.strip() + "\n") if conventions else "",
    }
    for key, value in parts.items():
        text = text.replace(key, value)
    return text


def _gap_marker(gap: float) -> Optional[str]:
    if gap >= SCENE_S:
        return "=="
    if gap >= PAUSE_S:
        return "--"
    return None


def _reference_after(cues: Sequence[Cue], reference: Sequence[dict]) -> List[tuple]:
    """(sort key, line) for every reference cue, keyed to land after its best draft cue."""
    out = []
    for r in reference:
        best, best_ov = None, 0.0
        for c in cues:
            ov = min(c.end, r["end"]) - max(c.start, r["start"])
            if ov > best_ov:
                best, best_ov = c, ov
        key = (best.start, 1) if best is not None else (r["start"], 1)
        out.append((key, "R " + str(r["text"]).replace("\n", " ")))
    return out


def render_stream(cues: Sequence[Cue], reference: Sequence[dict] = ()) -> str:
    items = [((c.start, 0), c) for c in cues] + _reference_after(cues, reference)
    items.sort(key=lambda t: t[0])
    out: List[str] = []
    prev_end = None
    for _, item in items:
        if isinstance(item, str):
            out.append(item)
            continue
        c: Cue = item
        if prev_end is not None:
            mark = _gap_marker(c.start - prev_end)
            if mark:
                out.append(mark)
        first, *rest = c.text.split("\n")
        out.append(f"{c.id}|{first}" if c.editable else f"~{c.id}|{first}")
        out.extend("  " + line for line in rest)
        prev_end = c.end
    return "\n".join(out)


def user_message(cues: Sequence[Cue], reference: Sequence[dict], title: str = "",
                 context: str = "") -> str:
    parts = [f"# {title}" if title else "# Subtitle"]
    if context:
        parts += ["", context.strip()]
    parts += [
        "", "## Format", "",
        "Each draft cue is one line, `<id>|<text>`; an indented line with no id is that cue's "
        "second display line. Lines starting `~` are context from just outside your section: "
        "read them, never edit them. "
        f"`--` marks a pause of {PAUSE_S:g}-{SCENE_S:g} s between cues, `==` a longer gap "
        "(often a scene change).",
    ]
    if reference:
        parts += ["", "Lines starting `R ` are a reference subtitle for the same audio in "
                      "another language or register, placed in time order among the draft "
                      "cues. Use them for meaning and names only; never copy their wording."]
    else:
        parts += ["", "There is no reference subtitle for this file. Work from the draft alone: "
                      "where the instructions mention the reference, it is simply not available, "
                      "and a name's spelling should follow the draft's most plausible form."]
    parts += ["", "## Draft", "", render_stream(cues, reference), "",
              "Proofread every cue that has an id without `~`. Return only the JSON object."]
    return "\n".join(parts)


AGREEMENT_TOLERANCE = 0.5   # seconds between a reference start and the nearest cue start
# Share of reference starts within the tolerance of a cue start. Measured: a reference on the
# cues' own timeline scores 86-94 %; the same reference shifted by a second or more, or one
# timed to a different release, falls to the 33-46 % chance level of dense dialogue.
MIN_AGREEMENT = 0.6


def reference_agreement(cues: Sequence[Cue], reference: Sequence[dict], shift: float = 0.0) -> float:
    """Share of reference cues starting within ``AGREEMENT_TOLERANCE`` of a cue start.

    The check that the reference and the cues share a timeline, which the stream's
    interleaving silently assumes. Start times, not overlap: dialogue is dense enough that
    a reference shifted by half a minute still overlaps some cue ~80 % of the time.
    """
    import bisect
    starts = sorted(c.start for c in cues)
    if not starts or not reference:
        return 0.0
    hit = 0
    for r in reference:
        t = float(r["start"]) + shift
        i = bisect.bisect_left(starts, t)
        near = min(abs(starts[j] - t) for j in (i - 1, i) if 0 <= j < len(starts))
        hit += near <= AGREEMENT_TOLERANCE
    return hit / len(reference)


def best_reference_shift(cues: Sequence[Cue], reference: Sequence[dict],
                         span: float = 10.0, step: float = 0.1) -> tuple:
    """``(shift, agreement)``: the constant shift within ±``span`` s that lines up best.

    Every shift within the tolerance of the true one scores the same, so the answer is the
    middle of the best-scoring run of shifts, not its first.
    """
    n = int(round(span / step))
    scored = [(round(k * step, 3), reference_agreement(cues, reference, k * step))
              for k in range(-n, n + 1)]
    top = max(score for _, score in scored)
    best = [shift for shift, score in scored if score >= top - 1e-9]
    return best[len(best) // 2], top


def approx_tokens(text: str) -> int:
    """Offline estimate, pessimistic for CJK (about one token per common Han character,
    more for a rare or supplementary-plane one; about four Latin characters per token).
    Measured against Gemini's own count it runs roughly 10-15% high."""
    cjk = sum(1 for ch in text if ord(ch) >= 0x2E80)
    rare = sum(1 for ch in text if ord(ch) >= 0x20000)
    return int(cjk * 1.15 + rare * 1.5 + (len(text) - cjk) / 3.6)

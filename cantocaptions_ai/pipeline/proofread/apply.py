"""Turn a proofreading answer into changed cues, deterministically.

The model's answer is a proposal. What reaches the subtitle is decided here, in code:

1. **Validation.** An edit must name an editable cue, carry text, and change something;
   one cue is edited once. Anything else is dropped and counted.
2. **Confidence.** Edits below ``min_confidence`` are kept as flags rather than applied.
3. **Names.** The answer's ``names`` list is applied to *every* cue, not only those the
   model edited. A recogniser repeats one wrong spelling of a name consistently, and a model
   shown that consistency tends to adopt it unless the name is decided once, globally.
4. **Register guard.** An edit that introduces a character of a register the standard
   excludes -- for written Cantonese, a Standard Written Chinese one borrowed from the
   reference -- is turned into a flag. Measured on Cantonese, it fired rarely and never on a
   good edit.
5. **Particle guard.** Under ``particles="protect"`` (the default), a change to a
   clause-final run of the standard's ``final_particles`` is undone and reported as a flag;
   the rest of the edit still applies. The prompt already tells the model to leave them
   alone; this holds when it does not.
6. **Cue boundaries.** A pair of edits that carries words from one cue into its neighbour
   (``moves_text``) is refused whole and reported as a flag. Boundaries come from the audio's
   timing; measured on whole episodes, moves were rarely right and, scored per cue, each
   one read as two errors.

Pure functions over plain data; no network, no models.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from cantocaptions_ai.languages.base import ProofreadStandard
from cantocaptions_ai.text_profiles import ScriptConfig

CONFIDENCE = {"low": 0, "medium": 1, "high": 2}
MIN_NAME_FORM_CHARS = 2      # a shorter form is too likely to be an ordinary word


@dataclass
class Outcome:
    """What applying one answer did."""
    texts: Dict[int, str] = field(default_factory=dict)       # cue id -> new text
    edits: List[dict] = field(default_factory=list)           # applied, as given (+ names)
    flags: List[dict] = field(default_factory=list)           # model flags + demoted edits
    invalid: List[dict] = field(default_factory=list)
    names_applied: Dict[int, List[str]] = field(default_factory=dict)


def validate(edits: Sequence[dict], current: Dict[int, str]) -> Tuple[List[dict], List[dict]]:
    """(usable, invalid) -- each invalid edit carrying its ``problem``."""
    usable, invalid, seen = [], [], set()
    for e in edits:
        cid = e.get("id")
        if cid not in current:
            invalid.append({**e, "problem": "not an editable cue id"})
        elif not isinstance(e.get("text"), str):
            invalid.append({**e, "problem": "missing text"})
        elif cid in seen:
            invalid.append({**e, "problem": "duplicate id"})
        elif e["text"] == current[cid]:
            invalid.append({**e, "problem": "no change"})
        else:
            seen.add(cid)
            usable.append(e)
    return usable, invalid


def name_table(names: Sequence[dict]) -> Dict[str, str]:
    """{draft form: name to use}, longest forms first so overlapping forms are safe.

    Forms shorter than ``MIN_NAME_FORM_CHARS``, forms equal to their target and forms
    contained in their target (阿明 -> 明 would rewrite the target itself) are left out.
    """
    table: Dict[str, str] = {}
    for n in names or []:
        use = str(n.get("use") or "").strip()
        if not use:
            continue
        for form in n.get("draft_forms") or []:
            form = str(form or "").strip()
            if len(form) >= MIN_NAME_FORM_CHARS and form != use and form not in use:
                table[form] = use
    return dict(sorted(table.items(), key=lambda kv: -len(kv[0])))


def apply_names(text: str, table: Dict[str, str], script: ScriptConfig) -> Tuple[str, List[str]]:
    """*text* with every name form replaced, and which replacements happened.

    In a space-separated script a form must match whole words, or "Ann" -> "Anne" would
    rewrite "Annual"; a CJK script has no word boundaries to respect.
    """
    used = []
    for form, use in table.items():
        if script.spaced:
            pattern = re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)")
            new = pattern.sub(use, text)
        else:
            new = text.replace(form, use)
        if new != text:
            used.append(f"{form}→{use}")
            text = new
    return text, used


def _sound(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PZSC")


def introduced_foreign(before: str, after: str, standard: ProofreadStandard,
                       script: ScriptConfig) -> List[str]:
    """Items of the standard's foreign register that *after* has more of than *before*."""
    if not standard.foreign_register:
        return []
    if script.spaced:
        words_a = re.findall(r"\w+", before.lower())
        words_b = re.findall(r"\w+", after.lower())
        return sorted({w for w in standard.foreign_register
                       if words_b.count(w.lower()) > words_a.count(w.lower())})
    a, b = _sound(before), _sound(after)
    return sorted({ch for ch in standard.foreign_register if b.count(ch) > a.count(ch)})


def _units(text: str, script: ScriptConfig) -> List[str]:
    """*text* as the units a particle is made of: words in a spaced script, else characters."""
    return re.findall(r"\w+|\W", text) if script.spaced else list(text)


def _is_break(unit: str) -> bool:
    return all(unicodedata.category(ch)[0] in "PZ" for ch in unit)


def _clause_final(units: Sequence[str], end: int) -> bool:
    """Whether a run ending before ``units[end]`` closes a clause (punctuation or the end)."""
    while end < len(units) and units[end].isspace():
        end += 1
    return end >= len(units) or _is_break(units[end])


def protect_particles(before: str, after: str, standard: ProofreadStandard,
                      script: ScriptConfig) -> Tuple[str, List[Tuple[str, str]]]:
    """*after* with every change to a clause-final particle run undone, and what was undone.

    A change is undone when, on both sides, the units it touches are all particles (or
    breaks) and it ends a clause: swapping 啦 for 喇, adding 吖, dropping 㗎 all qualify. A
    change that also touches an ordinary word is left alone -- the word is the point of it.
    """
    particles = {p.lower() for p in standard.final_particles}
    if not particles:
        return after, []
    a, b = _units(before, script), _units(after, script)

    def only_particles(units: Sequence[str]) -> bool:
        words = [u for u in units if not _is_break(u)]
        return all(u.lower() in particles for u in words)

    out, undone = [], []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        old, new = a[i1:i2], b[j1:j2]
        if (op != "equal" and only_particles(old) and only_particles(new)
                and any(not _is_break(u) for u in old + new)
                and _clause_final(a, i2) and _clause_final(b, j2)):
            # The draft's particles, the answer's punctuation: 你返嚟喇 -> 你返嚟嗱？ keeps
            # the question mark and comes out 你返嚟喇？.
            words = [u for u in old if not _is_break(u)]
            seg, placed = [], False
            for u in new:
                if not _is_break(u):
                    if not placed:
                        seg += words
                        placed = True
                else:
                    seg.append(u)
            out += seg if placed else words + seg
            undone.append(("".join(words), "".join(u for u in new if not _is_break(u))))
        else:
            out += new
    return "".join(out), undone


def moves_text(before_a: str, after_a: str, before_b: str, after_b: str) -> bool:
    """Whether edits to two adjacent cues (a, then b) carry words across their boundary.

    Judged on the sounded text alone (punctuation and spaces dropped): one cue loses words
    at the shared edge and the other gains those same words there. A pure move leaves the
    pair's combined text unchanged; a move with a fix elsewhere in either cue is caught too.
    """
    sa, ta, sb, tb = (_sound(t) for t in (before_a, after_a, before_b, after_b))
    if sa + sb == ta + tb and sa != ta:
        return True
    for k in range(1, min(len(sa), len(tb)) + 1):           # a's tail went to b's head
        x = sa[-k:]
        if not ta.endswith(x) and tb.startswith(x) and not sb.startswith(x):
            return True
    for k in range(1, min(len(sb), len(ta)) + 1):           # b's head went to a's tail
        x = sb[:k]
        if not tb.startswith(x) and ta.endswith(x) and not sa.endswith(x):
            return True
    return False


def apply_answer(current: Dict[int, str], answer: dict, standard: ProofreadStandard,
                 script: ScriptConfig, min_confidence: str = "low",
                 particles: str = "protect") -> Outcome:
    """Decide which of *answer*'s proposals change which cues. See the module docstring."""
    out = Outcome(flags=[dict(f) for f in answer.get("flags", []) if f.get("id") in current])
    usable, out.invalid = validate(answer.get("edits", []), current)
    floor = CONFIDENCE.get(min_confidence, 0)

    # Cue boundaries follow the audio and are not the model's to change: a pair of edits
    # that moves words between neighbours is refused whole and listed for review.
    proposed = {e["id"]: e["text"] for e in usable}
    moved = set()
    for cid in sorted(proposed):
        nxt = cid + 1
        if nxt in proposed and nxt in current and moves_text(
                current[cid], proposed[cid], current[nxt], proposed[nxt]):
            moved |= {cid, nxt}
            out.flags.append({"id": cid, "note": (
                f"suggested moving text between this cue and the next (「{proposed[cid]}」 / "
                f"「{proposed[nxt]}」); not applied, cue boundaries are kept")})

    for e in usable:
        cid = e["id"]
        if cid in moved:
            continue
        if CONFIDENCE.get(e.get("confidence", "high"), 2) < floor:
            out.flags.append({"id": cid, "note": (
                f"suggested 「{e['text']}」 ({e.get('confidence')} confidence: "
                f"{e.get('reason', '')})")})
            continue
        if particles == "protect":
            kept, undone = protect_particles(current[cid], e["text"], standard, script)
            if undone:
                swaps = "、".join(f"{o or '∅'}→{n or '∅'}" for o, n in undone)
                out.flags.append({"id": cid, "note": (
                    f"particle change {swaps} not applied (sentence-final particles are "
                    f"protected): {e.get('reason', '')}")})
                if kept == current[cid]:
                    continue
                e = dict(e, text=kept)
        foreign = introduced_foreign(current[cid], e["text"], standard, script)
        if foreign:
            out.flags.append({"id": cid, "note": (
                f"suggested 「{e['text']}」 was not applied: it introduces "
                f"{standard.foreign_register_name or 'excluded'} {'、'.join(foreign)}; "
                "check what was actually said")})
            continue
        out.texts[cid] = e["text"]
        out.edits.append(dict(e))

    table = name_table(answer.get("names", []))
    if table:
        for cid, before in current.items():
            text = out.texts.get(cid, before)
            new, used = apply_names(text, table, script)
            if not used:
                continue
            out.texts[cid] = new
            out.names_applied[cid] = used
            if not any(e["id"] == cid for e in out.edits):
                out.edits.append({"id": cid, "text": new, "type": "name", "confidence": "high",
                                  "reason": "name: " + ", ".join(used)})
            else:
                for e in out.edits:
                    if e["id"] == cid:
                        e["text"] = new
    # A name pass can turn an edit back into the original text; that is no edit at all.
    for cid in [c for c, t in out.texts.items() if t == current[c]]:
        del out.texts[cid]
    out.edits = [e for e in out.edits if e["id"] in out.texts]
    return out

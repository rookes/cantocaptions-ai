"""Files for a human to review a proofreading run with: an audit SRT and editor bookmarks.

``changes.srt`` holds every edited cue on its own timing, the new text over a ``[was]`` line
with the old text, the changed characters coloured in each. Players and editors that read
SRT markup (VLC, mpv, asbplayer, Subtitle Edit) show the colours; a text editor shows the tags.

The bookmarks file is Subtitle Edit's sidecar (``<subtitle file>.SE.bookmarks``, which it
loads on its own when it opens the subtitle): every changed cue is bookmarked with what it
was, every flagged cue with the flag. Format as Subtitle Edit writes it
(``libse/Common/BookmarkPersistence.cs``): UTF-8 with a byte-order mark, one JSON object
``{"bookmarks":[{"idx":N,"txt":"..."}, ...]}``, where ``idx`` is the cue's **0-based**
position in the file -- one less than its SRT number -- and a line break in ``txt`` is
written ``<br />``.
"""
from __future__ import annotations

import difflib
import os
from typing import Iterable, List, Optional, Sequence, Tuple

NEW_COLOUR, OLD_COLOUR = "#66ff66", "#ff6666"


def highlight_diff(before: str, after: str) -> Tuple[str, str]:
    """``(after, before)`` with the changed characters wrapped in SRT ``<font color>`` tags.

    Character-level, so a one-character fix lights up one character in each line.
    """
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    new, old = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            new.append(after[j1:j2])
            old.append(before[i1:i2])
            continue
        if j2 > j1:
            new.append(f'<font color="{NEW_COLOUR}">{after[j1:j2]}</font>')
        if i2 > i1:
            old.append(f'<font color="{OLD_COLOUR}">{before[i1:i2]}</font>')
    return "".join(new), "".join(old)


def write_changes_srt(path: str, rows: Iterable[Tuple[float, float, str, str]]) -> int:
    """Write ``changes.srt`` from ``(start, end, new text, old text)`` rows. Returns the count.

    Each cue is two lines: the new text, then ``[was]`` and the old one, both highlighted. A
    line break inside a cue is shown as `` / `` so each half stays on one line.
    """
    from cantocaptions_ai.utils.output import format_timestamp

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for start, end, new, old in rows:
            count += 1
            hi_new, hi_old = highlight_diff(old.replace("\n", " / "), new.replace("\n", " / "))
            stamp_a = format_timestamp(start, always_include_hours=True, decimal_marker=",")
            stamp_b = format_timestamp(max(end, start), always_include_hours=True,
                                       decimal_marker=",")
            f.write(f"{count}\n{stamp_a} --> {stamp_b}\n{hi_new}\n[was] {hi_old}\n\n")
    return count


def bookmarks_path(subtitle_path: str) -> str:
    """Where the bookmarks for *subtitle_path* go, without overwriting one already there.

    ``<subtitle>.SE.bookmarks`` when that is free -- the name Subtitle Edit loads by itself.
    Otherwise ``<stem> (1)<ext>.SE.bookmarks``, ``(2)`` and so on: rename the subtitle to
    match (``<stem> (1)<ext>``) and Subtitle Edit picks the bookmarks up again.
    """
    first = subtitle_path + ".SE.bookmarks"
    if not os.path.exists(first):
        return first
    stem, ext = os.path.splitext(subtitle_path)
    n = 1
    while os.path.exists(f"{stem} ({n}){ext}.SE.bookmarks"):
        n += 1
    return f"{stem} ({n}){ext}.SE.bookmarks"


def _encode(text: str) -> str:
    """Subtitle Edit's ``Json.EncodeJsonText``: escape ``\\`` and ``"``, line breaks as ``<br />``."""
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return text.replace("\r\n", "<br />").replace("\n", "<br />")


def write_bookmarks(subtitle_path: str, marks: Sequence[Tuple[int, str]]) -> Optional[str]:
    """Write Subtitle Edit bookmarks for *subtitle_path*: ``(0-based cue index, note)`` pairs.

    Returns the path written, or None when there is nothing to bookmark (Subtitle Edit itself
    writes no file then).
    """
    if not marks:
        return None
    path = bookmarks_path(subtitle_path)
    body = ",".join(f'{{"idx":{idx},"txt":"{_encode(txt)}"}}' for idx, txt in sorted(marks))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write('{"bookmarks":[\r\n' + body + ']}\r\n')
    return path


def _sounded(text: str) -> str:
    import unicodedata
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def remap_marks(source: Sequence[str], marks: Sequence[Tuple[int, str]],
                target: Sequence[str]) -> List[Tuple[int, str]]:
    """Carry ``(index, note)`` marks on *source* cues onto the *target* cues the text became.

    For a subtitle proofread first and then realigned: realign keeps the text and its order
    but not the cue list -- cues can be dropped (cuts, noise), split or merged, and cleaned --
    so neither index nor timing survives. Text does. Both sides are reduced to their letters
    and digits, aligned character by character, and each mark lands on the target cue holding
    the first character of its source cue that made it across. A cue none of whose characters
    survived goes to the target cue at the nearest surviving character before it (it was cut
    or merged away there). Marks meeting on one target cue are joined.
    """
    if not marks or not target:
        return []
    src_owner: List[int] = []
    tgt_owner: List[int] = []
    src = "".join(_sounded(t) for t in source)
    for i, t in enumerate(source):
        src_owner += [i] * len(_sounded(t))
    tgt = "".join(_sounded(t) for t in target)
    for i, t in enumerate(target):
        tgt_owner += [i] * len(_sounded(t))
    to_tgt = {}
    sm = difflib.SequenceMatcher(a=src, b=tgt, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            to_tgt[a + k] = b + k
    first_char = {}
    for pos, owner in enumerate(src_owner):
        first_char.setdefault(owner, pos)
    mapped = sorted(to_tgt)
    out = {}
    for idx, note in marks:
        start = first_char.get(idx)
        positions = [p for p in range(start, start + len(_sounded(source[idx])))
                     if p in to_tgt] if start is not None else []
        if positions:
            cue = tgt_owner[to_tgt[positions[0]]]
        else:
            # nothing of it survived: the nearest surviving character before where it was
            anchor = start if start is not None else 0
            before = [p for p in mapped if p < anchor]
            cue = tgt_owner[to_tgt[before[-1]]] if before else (tgt_owner[to_tgt[mapped[0]]] if mapped else 0)
        out.setdefault(cue, []).append(note)
    return sorted((cue, "\n".join(notes)) for cue, notes in out.items())


def bookmark_marks(segments: List[dict], notes: dict) -> List[Tuple[int, str]]:
    """``(index, note)`` for each of *segments* with notes; *notes* is keyed by ``id(segment)``.

    Keyed by object rather than position because re-cleaning can drop an edited cue between
    proofreading and writing, which shifts every later index; the index is taken from the
    list as it will be written.
    """
    return [(i, "\n".join(notes[id(seg)])) for i, seg in enumerate(segments) if notes.get(id(seg))]

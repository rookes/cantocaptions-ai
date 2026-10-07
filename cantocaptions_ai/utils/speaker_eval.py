"""Score speaker diarization where it is actually used: at boundaries between pieces of speech.

The pipeline never asks diarization "who is this?" across a whole episode. It asks cue assembly's
question, "do these two adjacent stretches share a voice?" (``segmentation._same_speaker``), and
it answers from per-segment labels that ``speaker_assign`` only writes when one speaker dominates.
So this scores *boundaries*, each a pair of time spans with a known (or proxy) answer:

``change``    the two spans are different speakers. The only source of this without hand
              labels is the subtitle convention for a two-speaker cue, one hyphen-led line
              per speaker (see :func:`hyphen_turns`); the change sits between the lines.
``same``      the two spans are one speaker: halves of a single-speaker cue, split at a clause
              mark. This is the boundary the merge gate meets most often, so its error rate is
              the cost of a speaker signal: every false "different" splits a sentence.
``adjacent``  two consecutive cues. Truth unknown -- reported only as a rate, because a
              diarizer that almost never calls these different is not separating speakers.

Two kinds of score:

* :func:`boundary_decision` -- the pipeline's own decision from diarization turns, via
  ``speaker_assign``'s dominance rule. :func:`decision_rates` tabulates it per kind.
* :func:`separability` -- for any continuous same-speaker score (an embedding cosine, say):
  AUC, EER, and the recall of ``change`` boundaries at a fixed false-split rate on ``same``
  ones, which is the operating point that matters for a merge veto.

Pure functions over plain data; no models, no I/O.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cantocaptions_ai.pipeline.speaker_assign import rank_shares, speaker_shares
from cantocaptions_ai.utils.schema import SpeakerTurn

Span = Tuple[float, float]

BOUNDARY_KINDS = ("change", "same", "adjacent")
DECISIONS = ("diff", "same", "unknown")

# A speaker line in a two-speaker cue: "-你好" / "- 你好", with the fullwidth and Unicode
# hyphens that turn up in Chinese subtitles. A lone hyphen-led line is not this convention --
# it is more often a dash in the text -- so a cue needs two or more.
_SPEAKER_HYPHEN = re.compile(r"^\s*[-－‐–]\s*")


def hyphen_turns(text: str) -> Optional[List[str]]:
    """The per-speaker lines of a hyphen-convention cue, or None for an ordinary cue.

    A cue qualifies when at least two of its lines open with a hyphen. A line without one
    continues the speaker before it (a wrapped line); one before the first hyphen starts the
    first speaker's turn. Lines are returned with the hyphen removed.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if sum(bool(_SPEAKER_HYPHEN.match(line)) for line in lines) < 2:
        return None
    turns: List[str] = []
    for line in lines:
        if _SPEAKER_HYPHEN.match(line) or not turns:
            turns.append(_SPEAKER_HYPHEN.sub("", line).strip())
        else:
            # A wrapped continuation of the current speaker's line.
            turns[-1] += line.strip()
    return turns if len(turns) >= 2 else None


@dataclass
class Boundary:
    """Two adjacent spans and what is known about whether they share a speaker."""
    kind: str
    left: Span
    right: Span
    source: str = ""
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in BOUNDARY_KINDS:
            raise ValueError(f"unknown boundary kind {self.kind!r}")

    @property
    def shorter_side(self) -> float:
        return min(self.left[1] - self.left[0], self.right[1] - self.right[0])


def span_label(
    turns: Sequence[SpeakerTurn], span: Span, min_share: float
) -> Tuple[Optional[str], Optional[str]]:
    """(label, top speaker) for a span, exactly as ``speaker_assign`` would label a segment.

    ``label`` is None when no speaker holds ``min_share`` of the span's diarized time; ``top``
    is the leading speaker regardless, so a caller can tell a threshold miss from no speech.
    ``turns`` must be sorted by start.
    """
    ranked = rank_shares(speaker_shares(turns, span[0], span[1])[0])
    if not ranked:
        return None, None
    speaker, share = ranked[0]
    return (speaker if share >= min_share else None), speaker


def boundary_decision(
    turns: Sequence[SpeakerTurn], boundary: Boundary, min_share: float = 0.7
) -> str:
    """"diff", "same" or "unknown": what the merge gate would conclude about this boundary."""
    left, _ = span_label(turns, boundary.left, min_share)
    right, _ = span_label(turns, boundary.right, min_share)
    if left is None or right is None:
        return "unknown"
    return "diff" if left != right else "same"


def decision_rates(
    boundaries: Sequence[Boundary], decisions: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Per boundary kind: count and the share of each decision.

    For ``change`` the "diff" share is recall; for ``same`` it is the false-split rate.
    """
    out: Dict[str, Dict[str, float]] = {}
    for kind in BOUNDARY_KINDS:
        picked = [d for b, d in zip(boundaries, decisions) if b.kind == kind]
        if not picked:
            continue
        row: Dict[str, float] = {"n": len(picked)}
        for decision in DECISIONS:
            row[decision] = picked.count(decision) / len(picked)
        out[kind] = row
    return out


@dataclass(frozen=True)
class Separability:
    """How well a same-speaker score separates ``change`` from ``same`` boundaries."""
    n_change: int
    n_same: int
    auc: float
    eer: float
    # false-split rate on ``same`` -> recall on ``change`` at that rate
    recall_at: Dict[float, float]


def separability(
    change_scores: Iterable[float],
    same_scores: Iterable[float],
    false_split_rates: Sequence[float] = (0.01, 0.02, 0.05),
) -> Separability:
    """Score where *higher means more likely the same speaker* (a cosine, a PLDA LLR...).

    ``recall_at[r]`` is the share of ``change`` boundaries scoring below the threshold that
    splits only ``r`` of the ``same`` ones -- the recall a merge veto would get if tuned to
    break at most that share of genuine one-speaker sentences.
    """
    change = np.asarray(list(change_scores), dtype=float)
    same = np.asarray(list(same_scores), dtype=float)
    if len(change) == 0 or len(same) == 0:
        raise ValueError("separability needs both change and same scores")

    # AUC as P(same-score > change-score), ties counted half (Mann-Whitney).
    scores = np.concatenate([same, change])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):  # average ranks over ties
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    auc = (ranks[: len(same)].sum() - len(same) * (len(same) + 1) / 2) / (len(same) * len(change))

    # EER: sweep thresholds; "split" when score < threshold.
    thresholds = np.unique(scores)
    miss = np.array([np.mean(change >= t) for t in thresholds])     # change not split
    false = np.array([np.mean(same < t) for t in thresholds])       # same split
    k = int(np.argmin(np.abs(miss - false)))
    eer = float((miss[k] + false[k]) / 2)

    recall_at = {}
    for rate in false_split_rates:
        threshold = np.quantile(same, rate)
        recall_at[rate] = float(np.mean(change < threshold))
    return Separability(len(change), len(same), float(auc), eer, recall_at)


def format_rates(rates: Dict[str, Dict[str, float]]) -> List[str]:
    """Human-readable lines for :func:`decision_rates`."""
    gloss = {"change": "recall", "same": "false split", "adjacent": "called different"}
    lines = []
    for kind, row in rates.items():
        lines.append(
            f"{kind:9s} n={int(row['n']):5d}  diff {row['diff']:6.1%} ({gloss[kind]})  "
            f"same {row['same']:6.1%}  unknown {row['unknown']:6.1%}"
        )
    return lines

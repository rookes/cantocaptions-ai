"""Fitting the transform between a subtitle's timeline and a different cut of the same content.

``--realign_mode sync`` and ``adjust`` need an answer to one question: given a set of places
where we are confident a subtitle line was spoken, what is the *simplest* map from the
subtitle's own timeline onto this recording's? The answer is almost never a single offset.
A PAL-speedup broadcast runs 4.27% fast against its Blu-ray master; a TV airing has its
adverts cut out; a home release restores a scene the broadcast dropped.

So the map is piecewise affine in **source** time::

    phi(x) = a_p * x + b_p     for x in [x0_p, x1_p)

and the whole design is about choosing how many pieces to use, and when a piece may take a
slope other than 1. Getting that choice wrong in either direction is bad in a different way:
too few pieces and a real cut smears its error across the whole file, too many and ordinary
anchor jitter is "explained" by inventing edits that were never made. The two penalties below
are the dials, and they are in the same units as the residuals they are weighed against.

**Everything here is pure** -- numpy only, no torch, no model, nothing imported from
realign.py -- which is what lets the algorithm be tested without a GPU. The acoustic half
(finding the anchors) lives in realign.py and hands this module nothing but numbers.

Read the note on monotonicity at :class:`Break` before changing how breakpoints work. The one
counter-intuitive thing in this module is that phi running *backwards* at a breakpoint is
correct, and is the definition of a cut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


# --- Constants -----------------------------------------------------------------------
#
# The two penalties are in seconds-squared, the same units as the sum of squared residuals
# they are compared against, so each reads directly as "how much squared error a piece must
# save to be worth having". They are deliberately NOT normalised by the anchor count: a fixed
# charge says "a break must be supported by this much evidence", and the evidence for a break
# genuinely does scale with the number of anchors it repairs.

# A piece cannot be fitted to fewer anchors than this. Without it the DP will happily
# manufacture a one-anchor piece that explains an outlier with zero residual and pays only
# BREAK_PENALTY for the privilege, which is cheaper than almost any honest fit. With it, an
# outlier inflates its segment's error, the trim below removes it, and the next round merges
# the scar away.
MIN_SEGMENT_ANCHORS = 3

# ...and a piece may only take a free slope with at least this many anchors, spanning at
# least this much source time. A slope fitted over a short anchor run and then extrapolated
# across a film is the dangerous failure in this module, and a hard gate is what prevents it.
# A soft prior would not: see the note on MAX_SCALE_DEV.
MIN_SLOPE_ANCHORS = 8
MIN_SLOPE_SPAN = 120.0

# Seconds-squared, charged once per additional piece. A 1 s jump costs each affected anchor
# about 1 s^2 that it cannot otherwise shed, so roughly four anchors justify a break; a 0.4 s
# jump needs about twenty-five. Within-piece anchor noise is around 0.15 s, i.e. 0.02 s^2 per
# anchor, so noise cannot buy a break at any plausible anchor count.
BREAK_PENALTY = 4.0

# Seconds-squared, charged once per piece that takes a free slope. At roughly seven times
# BREAK_PENALTY, a slope must beat about seven offset pieces -- which is what implements
# "prefer cuts and insertions to speed changes". Where a rate change is genuine the offset
# staircase would need dozens of pieces and still carry residuals, so the slope still wins by
# orders of magnitude; the preference only bites where it should.
SLOPE_PENALTY = 30.0

# Bound on |a - 1|. Beyond it the free-slope model is refused outright rather than clamped.
#
# Note this is a hard gate and not a soft prior, and that is deliberate. A continuous
# penalty on (a - 1)^2 needs an arbitrary time scale to be dimensionally meaningful, and
# worse, it *shrinks a genuine rate change*: the ReZero fixture needs the slope good to about
# 1e-5 or the end of the file is seconds out, and any shrinkage is visible there. A clamped
# slope is also a fit reporting a number it does not believe; the honest degradation is a
# staircase of offset pieces plus a warning.
MAX_SCALE_DEV = 0.25

# A smaller apparent jump than this is anchor noise rather than an edit, and the two pieces
# are merged. Below roughly this size a break cannot pay BREAK_PENALTY anyway.
MIN_BREAK_JUMP = 0.4

# Floor on the outlier-trim threshold, which is otherwise four scaled MADs. A very tight fit
# would otherwise start trimming anchors that sit at the edge of ordinary jitter.
OUTLIER_TOLERANCE = 0.5 # Changed from 1.0 for testing
MAX_TRIM_ROUNDS = 3

# Below either of these the fit is refused and the caller falls back (the identity for sync,
# plain placement for adjust). Timings mapped through a garbage fit are worse than unchanged
# timings, because they are wrong *and* they look deliberate.
MIN_FIT_ANCHORS = 4
MIN_FIT_INLIER_SHARE = 0.5

# ...and a third refusal, for the case the other two cannot see. Trimming only catches anchors
# that disagree with an otherwise-good fit; it says nothing when the anchors disagree with
# *each other* so thoroughly that the DP explains them with a break every few anchors. Nothing
# is trimmed there -- every piece fits its own handful perfectly -- but a map needing an edit
# every third anchor is not a map, it is a list of coincidences. A transform is a claim that
# two recordings are the same content, and this is the share of that claim that has to hold.
MAX_PIECE_SHARE = 0.15

# Share of its own duration a cue must lose to a cut interval before it counts as cut.
CUT_OVERLAP_SHARE = 0.5

REASON_CUT = "cut"
REASON_NO_AUDIO = "no_audio"

# --- sync mode's own anchor discipline ------------------------------------------------
#
# Using every anchor `map_cues` was handed exposes every cue that happens to score well to
# whatever acoustic mistake the search made for it, individually -- and a confidently wrong
# anchor (a repeated phrase confusing CTC, a stylised reading) scores exactly as well as a
# correct one. This guards against the case where the model is sometimes both confident and  
# wrong. What can is the fitted transform itself: an anchor with a large residual against 
# hundreds of others is exactly what raising the bar for direct trust should be measured 
# against, not the search's own opinion of itself.
#
# So sync keeps only a sparse, high-residual-screened subset of anchors for map_cues to trust
# directly (see prune_to_density), and everywhere else falls back to interpolation between
# whichever of those survive nearby -- the discipline the file-wide transform was fitted for
# in the first place, rather than 800-950 individually-trusted acoustic opinions.
TARGET_ANCHORS_PER_MINUTE = 3.0 # Changed from 3 for testing
MIN_ANCHORS_PER_PIECE = 3

# Common frame-rate conversions, for naming a fitted scale in the log. NAMING ONLY -- the fit
# is never snapped to one of these. On a test file the fixture the scale measures about 1.0435
# while 25/23.976 is 1.042709; that 7.9e-4 difference is 2.2 s of drift by the end of the
# file, so snapping would be actively wrong. Report the neighbour, keep the measurement.
_RATIOS = (
    ("25 -> 23.976 fps", 25.0 / (24000.0 / 1001.0)),
    ("25 -> 24 fps", 25.0 / 24.0),
    ("24 -> 23.976 fps", 24.0 / (24000.0 / 1001.0)),
    ("30 -> 29.97 fps", 30.0 / (30000.0 / 1001.0)),
)
# Relative, and tight enough to keep 25/23.976 and 25/24 (0.1% apart) from both matching.
_RATIO_TOLERANCE = 0.0008 # Changed from 0.0008 for testing


class TransformError(ValueError):
    """The anchors do not support a transform worth applying.

    ``scale_bound_hit`` distinguishes the two ways that happens, which call for opposite
    responses. Without it a source genuinely running at 1.5x reports as "probably not the same
    content" -- true of the *fit*, and actively misleading about the cause, since the one
    thing that would fix it is raising --realign_max_scale.
    """

    def __init__(self, message: str, *, scale_bound_hit: bool = False):
        super().__init__(message)
        self.scale_bound_hit = scale_bound_hit


# --- Value objects -------------------------------------------------------------------

@dataclass(frozen=True)
class Piece:
    """One affine stretch of the map, owning the source-time range [x0, x1)."""
    first: int          # inclusive anchor index
    last: int           # inclusive anchor index
    scale: float
    offset: float
    x0: float
    x1: float
    free_slope: bool
    residual_p50: float = 0.0
    residual_p90: float = 0.0

    @property
    def anchors(self) -> int:
        return self.last - self.first + 1

    def map(self, t: float) -> float:
        return self.scale * t + self.offset


@dataclass(frozen=True)
class Break:
    """A discontinuity between two pieces, stored as the source interval it really is.

    **A backwards jump is not a monotonicity violation.** It is what a cut looks like when the
    cut *interval* is collapsed to a point, and widening it back out restores monotonicity
    exactly. For two pieces with slope ``a`` and ``delta = b_next - b_prev < 0``, solving
    ``phi_prev(c0) == phi_next(c1)`` gives::

        c1 - c0 = |delta| / a

    so phi is continuous and non-decreasing with the open interval ``(c0, c1)`` excised from
    its domain -- that interval being source time this recording has no counterpart for. Do
    not "fix" this by forcing breakpoints to be non-decreasing: that deletes cuts from the
    model entirely and leaves nothing able to express the case the feature exists for.

    ``source_start == source_end`` for an insertion: the target gained audio, and no source
    time is lost.
    """
    source_start: float
    source_end: float
    jump: float          # target seconds; negative is a cut, positive an insertion
    kind: str            # "cut" | "insert"
    cue_indices: Tuple[int, ...] = ()

    @property
    def seconds(self) -> float:
        """Magnitude in target seconds -- runtime the target gained or lost."""
        return abs(self.jump)

    @property
    def source_seconds(self) -> float:
        """How much of the *subtitle's* timeline the cut covers. Zero for an insertion."""
        return self.source_end - self.source_start


@dataclass(frozen=True)
class Transform:
    pieces: Tuple[Piece, ...]
    breaks: Tuple[Break, ...] = ()
    used: Tuple[int, ...] = ()        # anchor indices the final fit kept
    rejected: Tuple[int, ...] = ()    # anchor indices trimmed as outliers
    cost: float = 0.0
    scale_bound_hit: bool = False     # a candidate slope exceeded max_scale_dev and was refused
    xs: Tuple[float, ...] = ()        # the anchors the fit was built from, for refitting
    ys: Tuple[float, ...] = ()
    ws: Tuple[float, ...] = ()
    # The transcript line index each surviving anchor in xs/ys/ws came from, same order and
    # length. What makes map_cues able to use an anchor's own measured position for the exact
    # line it was found on, rather than a smoothed line through hundreds of others -- see
    # map_cues for why that distinction turned out to matter far more than a piece-level fit
    # ever could, however precisely that fit is estimated.
    ids: Tuple[int, ...] = ()

    def piece_for(self, t: float) -> Piece:
        for piece in self.pieces:
            if t < piece.x1:
                return piece
        return self.pieces[-1]

    def __call__(self, t: float) -> float:
        """Map a source time onto the target timeline.

        Inside a cut the map is strictly undefined -- there is no audio for that source time.
        What comes back is the cut's own image, which is the continuous extension from both
        sides and is what the 'keep' cut policy stacks its orphaned cues on.
        """
        for brk in self.breaks:
            if brk.kind == "cut" and brk.source_start < t < brk.source_end:
                return self.piece_for(brk.source_start).map(brk.source_start)
        return self.piece_for(t).map(t)

    def map_span(self, start: float, end: float) -> Tuple[float, float]:
        lo = self(start)
        return lo, max(self(end), lo)

    @property
    def scale(self) -> float:
        """Anchor-weighted mean slope -- the one number that describes the whole fit."""
        total = sum(p.anchors for p in self.pieces)
        if total <= 0:
            return 1.0
        return sum(p.scale * p.anchors for p in self.pieces) / total

    @property
    def identity(self) -> bool:
        return (len(self.pieces) == 1 and not self.breaks
                and self.pieces[0].scale == 1.0 and self.pieces[0].offset == 0.0)


def identity_transform() -> Transform:
    """The map that changes nothing. What a refused fit falls back to."""
    return Transform(pieces=(Piece(0, -1, 1.0, 0.0, -math.inf, math.inf, False),))


def named_ratio(scale: float) -> Optional[str]:
    """The frame-rate conversion this scale sits near, for the log. Never snapped to."""
    best: Optional[str] = None
    best_err = _RATIO_TOLERANCE
    for name, value in _RATIOS:
        for candidate, label in ((value, name), (1.0 / value, name + " (inverted)")):
            err = abs(scale - candidate) / candidate
            if err < best_err:
                best, best_err = label, err
    return best


# --- The fit -------------------------------------------------------------------------

class _Sums:
    """Weighted prefix sums, with x and y centred so the normal equations stay conditioned.

    ``denom = W*XX - X*X`` loses most of its significant digits when every x in a segment sits
    near 2800 s and they span half a second, which is exactly the shape of a short segment
    late in a film. Centring costs one pass and removes the problem; the slope is invariant
    under it and the intercept is recovered on the way out.
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, ws: np.ndarray):
        self.xs = xs
        self.xm = float(np.average(xs, weights=ws)) if len(xs) else 0.0
        self.ym = float(np.average(ys, weights=ws)) if len(ys) else 0.0
        xc = xs - self.xm
        yc = ys - self.ym
        dc = yc - xc

        def cum(values: np.ndarray) -> np.ndarray:
            return np.concatenate(([0.0], np.cumsum(values)))

        self.W = cum(ws)
        self.X = cum(ws * xc)
        self.Y = cum(ws * yc)
        self.XX = cum(ws * xc * xc)
        self.XY = cum(ws * xc * yc)
        self.YY = cum(ws * yc * yc)
        # The a==1 model is fitted on d = y - x directly rather than as a constrained case of
        # the general one: it is exact, and it keeps the offset model well conditioned even
        # where the free-slope one is degenerate.
        self.D = cum(ws * dc)
        self.DD = cum(ws * dc * dc)


def _models(
    sums: _Sums, lo: np.ndarray, j: int, *,
    max_scale_dev: float, slope_penalty: float,
    min_slope_anchors: int, min_slope_span: float,
):
    """Both candidate models for every segment ``[lo, j)``, vectorised over lo.

    Returns (cost, use_free, scale, offset, bound_hit) as arrays.
    """
    W = sums.W[j] - sums.W[lo]
    X = sums.X[j] - sums.X[lo]
    Y = sums.Y[j] - sums.Y[lo]
    XX = sums.XX[j] - sums.XX[lo]
    XY = sums.XY[j] - sums.XY[lo]
    YY = sums.YY[j] - sums.YY[lo]
    D = sums.D[j] - sums.D[lo]
    DD = sums.DD[j] - sums.DD[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        sse_fixed = np.maximum(DD - D * D / W, 0.0)
        offset_fixed = sums.ym - sums.xm + D / W

        denom = W * XX - X * X
        # A segment whose anchors share one source time has no slope to estimate.
        solvable = denom > 1e-9 * np.maximum(np.abs(W * XX), 1.0)
        scale_free = np.where(solvable, (W * XY - X * Y) / np.where(solvable, denom, 1.0), 1.0)
        bc = (Y - scale_free * X) / W
        sse_free = np.maximum(YY - scale_free * XY - bc * Y, 0.0)
        offset_free = sums.ym - scale_free * sums.xm + bc

    count = j - lo
    span = sums.xs[j - 1] - sums.xs[lo]
    within = np.abs(scale_free - 1.0) <= max_scale_dev
    eligible = solvable & (count >= min_slope_anchors) & (span >= min_slope_span)
    gate = eligible & within
    bound_hit = bool(np.any(eligible & ~within))

    cost_free = np.where(gate, sse_free + slope_penalty, np.inf)
    use_free = cost_free < sse_fixed
    cost = np.where(use_free, cost_free, sse_fixed)
    scale = np.where(use_free, scale_free, 1.0)
    offset = np.where(use_free, offset_free, offset_fixed)
    return cost, use_free, scale, offset, bound_hit


def _fit_range(
    sums: _Sums, first: int, last: int, *,
    max_scale_dev: float, slope_penalty: float,
    min_slope_anchors: int, min_slope_span: float,
) -> Tuple[float, float, bool]:
    """Refit one anchor range on its own. Used after pieces are merged."""
    cost, use_free, scale, offset, _hit = _models(
        sums, np.array([first]), last + 1,
        max_scale_dev=max_scale_dev, slope_penalty=slope_penalty,
        min_slope_anchors=min_slope_anchors, min_slope_span=min_slope_span,
    )
    return float(scale[0]), float(offset[0]), bool(use_free[0])


def _segment(
    sums: _Sums, n: int, *,
    max_scale_dev: float, break_penalty: float, slope_penalty: float,
    min_segment_anchors: int, min_slope_anchors: int, min_slope_span: float,
) -> Tuple[List[Tuple[int, int, float, float, bool]], float, bool]:
    """The dynamic program. Returns (ranges, cost, scale_bound_hit).

    ``F[j]`` is the best cost for explaining the first j anchors::

        F[0] = -break_penalty                      # the first piece is free
        F[j] = break_penalty + min over i of ( F[i] + C(i, j) )

    where ``C(i, j)`` is the cheaper of the two models for anchors ``[i, j)``. C is O(1) from
    the prefix sums, so the whole thing is O(n^2) -- about 450k evaluations at n = 950, and
    one numpy call per j.
    """
    m = max(1, min_segment_anchors)
    if n < m:
        raise TransformError(
            f"a transform needs at least {m} anchors to fit a single piece; got {n}"
        )
    best = np.full(n + 1, np.inf)
    best[0] = -break_penalty
    back: List[Optional[Tuple[int, float, float, bool]]] = [None] * (n + 1)
    bound_hit = False

    for j in range(m, n + 1):
        lo = np.arange(0, j - m + 1)
        cost, use_free, scale, offset, hit = _models(
            sums, lo, j, max_scale_dev=max_scale_dev, slope_penalty=slope_penalty,
            min_slope_anchors=min_slope_anchors, min_slope_span=min_slope_span,
        )
        bound_hit = bound_hit or hit
        total = best[lo] + cost
        k = int(np.argmin(total))
        if not math.isfinite(float(total[k])):
            continue
        best[j] = break_penalty + float(total[k])
        back[j] = (int(lo[k]), float(scale[k]), float(offset[k]), bool(use_free[k]))

    if back[n] is None:
        raise TransformError("no feasible segmentation of the anchors")

    ranges: List[Tuple[int, int, float, float, bool]] = []
    j = n
    while j > 0:
        entry = back[j]
        if entry is None:
            raise TransformError("segmentation backtrack hit an unreachable state")
        first, scale, offset, free = entry
        ranges.append((first, j - 1, scale, offset, free))
        j = first
    ranges.reverse()
    return ranges, float(best[n]), bound_hit


def _merge_small_jumps(
    ranges: List[Tuple[int, int, float, float, bool]], sums: _Sums, *,
    min_break_jump: float, **fit_kwargs,
) -> List[Tuple[int, int, float, float, bool]]:
    """Merge neighbouring pieces the data cannot really tell apart.

    Two tests, and both have to pass. The jump at the boundary must be under
    ``min_break_jump`` -- anchor noise, not an edit -- *and* the slopes must agree closely
    enough that merging them does not move anything by more than the same amount over the
    merged span. The second test is what stops two pieces at genuinely different rates being
    fused just because they happen to cross near the boundary.
    """
    if len(ranges) < 2:
        return ranges
    out = list(ranges)
    changed = True
    while changed and len(out) > 1:
        changed = False
        for k in range(len(out) - 1):
            a_first, a_last, a_scale, a_offset, _a_free = out[k]
            b_first, b_last, b_scale, b_offset, _b_free = out[k + 1]
            boundary = (sums.xs[a_last] + sums.xs[b_first]) / 2.0
            jump = (b_scale * boundary + b_offset) - (a_scale * boundary + a_offset)
            span = max(sums.xs[b_last] - sums.xs[a_first], 1e-9)
            divergence = abs(a_scale - b_scale) * span
            if abs(jump) < min_break_jump and divergence < min_break_jump:
                scale, offset, free = _fit_range(sums, a_first, b_last, **fit_kwargs)
                out[k:k + 2] = [(a_first, b_last, scale, offset, free)]
                changed = True
                break
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The weighted median of *values*. Resists a skewed tail; a mean does not.

    A piece's offset only needs to describe where its *typical* anchor sits, not where
    the arithmetic average of all of them sits -- and those two disagree exactly when a
    handful of anchors read consistently high or low without being extreme enough to be
    trimmed as outliers by the earlier 4-MAD pass. A skew that size is invisible to the
    trim (nothing crosses its threshold) but still drags a mean-based offset off the
    typical cue's true position, one piece at a time -- which is the shape of a roughly
    constant, piece-dependent bias rather than of a slope or a breakpoint error.
    """
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    idx = min(int(np.searchsorted(cw, cw[-1] / 2.0)), len(v) - 1)
    return float(v[idx])


def _shared_slope_refit(
    ranges: List[Tuple[int, int, float, float, bool]],
    xs: np.ndarray, ys: np.ndarray, ws: np.ndarray,
    *, max_scale_dev: float, slope_penalty: float,
) -> List[Tuple[int, int, float, float, bool]]:
    """Re-fit every piece's offset against one slope shared across the whole file.

    A cut or an insertion says nothing about *speed* -- an edit removes or adds content,
    it does not change how fast the surviving footage plays. So the file's speed is a
    single property of the whole timeline, and estimating it independently inside each
    piece throws away most of the evidence for it: a short, sparsely-anchored piece pins
    its own slope far less precisely than the full file does.

    Measured on the ReZero Blu-ray (3 pieces cut apart by two insertions): the
    independently-fit slopes were 1.042982 / 1.043248 / 1.043691, disagreeing by up to
    7e-4 -- worth several tenths of a second of drift *within* each piece, growing away
    from that piece's own anchors and resetting at the next cut, exactly the symptom
    reported against hand-checked reference points. Correcting each piece's slope for its
    own reported drift independently gives 1.043427 and 1.043404 -- the two disagreeing
    pieces agree with each other to four decimal places once corrected, which is the
    signature of one true shared speed and three noisy independent estimates of it, not
    three different speeds.

    The pooled estimate is an ANCOVA-style common-slope regression: centre each piece's
    anchors on its own weighted mean first (this is what makes the per-piece intercepts
    drop out of the normal equations algebraically), then fit one slope through the
    pooled, centred residuals. A piece keeps its own independently-fit slope only when
    doing so earns back ``slope_penalty`` over the shared one -- so a file that genuinely
    splices footage at two different speeds is not forced together, but three noisy
    estimates of the same speed are.

    Each piece's own offset is then the **weighted median** of its residuals under
    ``a_shared``, not the mean: see ``_weighted_median``. Measured on the same Blu-ray
    after the slope fix alone, the progressive within-piece drift the slope sharing
    targets was 89% gone, but each piece still carried a roughly constant bias of its own
    (+0.27 s / -0.23 s) that a mean-based offset could not see past.
    """
    if len(ranges) < 2 or len(xs) == 0:
        return ranges

    cx = np.empty_like(xs)
    cy = np.empty_like(ys)
    centres: List[Tuple[float, float]] = []
    for first, last, *_rest in ranges:
        idx = slice(first, last + 1)
        w = ws[idx]
        xbar = float(np.average(xs[idx], weights=w))
        ybar = float(np.average(ys[idx], weights=w))
        centres.append((xbar, ybar))
        cx[idx] = xs[idx] - xbar
        cy[idx] = ys[idx] - ybar

    denom = float(np.sum(ws * cx * cx))
    if denom <= 1e-9:
        return ranges
    a_shared = float(np.sum(ws * cx * cy) / denom)
    if abs(a_shared - 1.0) > max_scale_dev:
        return ranges

    refined: List[Tuple[int, int, float, float, bool]] = []
    pooled = 0
    for k, (first, last, a_own, b_own, free_own) in enumerate(ranges):
        idx = slice(first, last + 1)
        w, x, y = ws[idx], xs[idx], ys[idx]
        b_shared = _weighted_median(y - a_shared * x, w)
        sse_shared = float(np.sum(w * (y - a_shared * x - b_shared) ** 2))
        sse_own = float(np.sum(w * (y - a_own * x - b_own) ** 2))
        cost_own = sse_own + (slope_penalty if free_own else 0.0)
        if sse_shared <= cost_own:
            refined.append((first, last, a_shared, b_shared, a_shared != 1.0))
            pooled += 1
        else:
            refined.append((first, last, a_own, b_own, free_own))

    if pooled >= 2:
        logger.info(
            "realign: %d of %d transform pieces shared one pooled speed (%.6fx) instead of "
            "each fitting its own -- an edit does not change playback speed, so a lone piece "
            "estimating it from a fraction of the file's anchors is the noisier answer",
            pooled, len(refined), a_shared,
        )
    return refined


def _residuals(
    ranges: Sequence[Tuple[int, int, float, float, bool]],
    xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    out = np.zeros(len(xs))
    for first, last, scale, offset, _free in ranges:
        idx = np.arange(first, last + 1)
        out[idx] = ys[idx] - (scale * xs[idx] + offset)
    return out


def fit_transform(
    pairs: Sequence[Tuple[float, float, float]],
    *,
    ids: Optional[Sequence[int]] = None,
    max_scale_dev: float = MAX_SCALE_DEV,
    break_penalty: float = BREAK_PENALTY,
    slope_penalty: float = SLOPE_PENALTY,
    min_segment_anchors: int = MIN_SEGMENT_ANCHORS,
    min_slope_anchors: int = MIN_SLOPE_ANCHORS,
    min_slope_span: float = MIN_SLOPE_SPAN,
    min_break_jump: float = MIN_BREAK_JUMP,
    outlier_tolerance: float = OUTLIER_TOLERANCE,
    max_trim_rounds: int = MAX_TRIM_ROUNDS,
    min_anchors: int = MIN_FIT_ANCHORS,
    min_inlier_share: float = MIN_FIT_INLIER_SHARE,
    max_piece_share: float = MAX_PIECE_SHARE,
) -> Transform:
    """Fit the simplest piecewise-affine map explaining *pairs*.

    Each pair is ``(source time, audio time, weight)``, in source order. ``ids``, if given,
    is a parallel array (a transcript line index per pair) that survives sorting and outlier
    trimming unchanged and comes back as ``Transform.ids`` -- what lets ``map_cues`` use an
    anchor's own measured position for the exact line it came from, see there. Raises
    :class:`TransformError` when the anchors do not support a fit worth applying, which the
    caller must treat as "change nothing" rather than as a failure to work around.

    **Robustness is a trim loop around the DP, not a robust loss inside it.** A Huber loss in
    ``C(i, j)`` needs an IRLS per candidate segment and would make the DP cubic. Squared error
    inside plus an outlier trim outside gets the same protection at the same cost, and
    ``MIN_SEGMENT_ANCHORS`` is what stops the DP hiding an outlier in a piece of its own
    before the trim can see it.

    There is deliberately **no global pre-screen** -- fitting one robust line and dropping
    whatever sits far from it is tempting and wrong, because a genuine cut makes half the
    anchors outliers of that line. The screening that matters has already happened in
    realign's ``sanitise_anchors`` and ``verify_anchors``, the latter being the only thing
    that catches a *confidently wrong* anchor, which is exactly a high-leverage outlier here.
    """
    if len(pairs) < max(min_anchors, min_segment_anchors):
        raise TransformError(
            f"need at least {max(min_anchors, min_segment_anchors)} anchors to fit a "
            f"transform; got {len(pairs)}"
        )
    all_xs = np.array([float(p[0]) for p in pairs], dtype=float)
    all_ys = np.array([float(p[1]) for p in pairs], dtype=float)
    all_ws = np.array([max(float(p[2]), 1e-6) for p in pairs], dtype=float)
    # A caller that gives no ids gets none back (Transform.ids == ()), never a sequential
    # placeholder. map_cues treats a populated ids as "cue i is anchor i" by line index, and
    # a made-up 0..n-1 default would collide with unrelated span indices in exactly that
    # lookup -- silently mapping some cue to a stranger's anchor merely because both happened
    # to carry the same small integer index.
    has_ids = ids is not None
    all_ids = np.array(list(ids), dtype=int) if has_ids else np.arange(len(pairs))
    if len(all_ids) != len(pairs):
        raise ValueError(f"ids has {len(all_ids)} entries for {len(pairs)} pairs")
    if np.any(np.diff(all_xs) < 0):
        order = np.argsort(all_xs, kind="stable")
        all_xs, all_ys, all_ws, all_ids = (
            all_xs[order], all_ys[order], all_ws[order], all_ids[order],
        )

    fit_kwargs = dict(
        max_scale_dev=max_scale_dev, slope_penalty=slope_penalty,
        min_slope_anchors=min_slope_anchors, min_slope_span=min_slope_span,
    )
    keep = np.arange(len(all_xs))
    ranges: List[Tuple[int, int, float, float, bool]] = []
    cost = 0.0
    bound_hit = False
    sums = _Sums(all_xs, all_ys, all_ws)

    for _round in range(max(1, max_trim_rounds)):
        xs, ys, ws = all_xs[keep], all_ys[keep], all_ws[keep]
        if len(xs) < max(min_anchors, min_segment_anchors):
            raise TransformError("outlier trimming left too few anchors to fit a transform")
        sums = _Sums(xs, ys, ws)
        ranges, cost, bound_hit = _segment(
            sums, len(xs), break_penalty=break_penalty,
            min_segment_anchors=min_segment_anchors, **fit_kwargs,
        )
        ranges = _merge_small_jumps(
            ranges, sums, min_break_jump=min_break_jump, **fit_kwargs,
        )
        residual = _residuals(ranges, xs, ys)
        spread = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
        threshold = max(outlier_tolerance, 4.0 * spread)
        inlier = np.abs(residual) <= threshold
        if bool(np.all(inlier)):
            break
        survivors = keep[inlier]
        if len(survivors) < max(min_anchors, min_segment_anchors):
            break
        # Only narrow the anchor set if there is another round left to refit it with.
        # ``ranges`` has to describe ``keep`` when the loop exits, or the piece boundaries
        # below index an array that no longer exists.
        if _round == max(1, max_trim_rounds) - 1:
            break
        keep = survivors

    share = len(keep) / float(len(all_xs))
    if share < min_inlier_share:
        raise TransformError(
            f"only {len(keep)} of {len(all_xs)} anchors ({share:.0%}) agree on any transform; "
            "the subtitle and this recording may not be the same content"
        )
    if len(ranges) > max(3, int(max_piece_share * len(keep))):
        if bound_hit:
            # The staircase is a symptom, not the disease: the data wanted one sloped piece
            # and the bound refused it, so the DP approximated the slope with steps.
            raise TransformError(
                f"the subtitle and this recording differ in speed by more than "
                f"--realign_max_scale ({max_scale_dev:.0%}) allows, so no transform was "
                f"fitted. Raise it if they really do run at that different a rate.",
                scale_bound_hit=True,
            )
        raise TransformError(
            f"explaining {len(keep)} anchors needed {len(ranges)} transform pieces, i.e. an "
            "edit every few lines; the subtitle and this recording are probably not the same "
            "content"
        )

    xs, ys, ws = all_xs[keep], all_ys[keep], all_ws[keep]
    # Breakpoints are settled; now ask whether the pieces they define actually run at
    # different speeds, or whether the same one true speed was just estimated noisily
    # three separate times. See _shared_slope_refit.
    ranges = _shared_slope_refit(
        ranges, xs, ys, ws, max_scale_dev=max_scale_dev, slope_penalty=slope_penalty,
    )
    ranges = _merge_small_jumps(
        ranges, _Sums(xs, ys, ws), min_break_jump=min_break_jump, **fit_kwargs,
    )
    residual = _residuals(ranges, xs, ys)
    pieces: List[Piece] = []
    for n_piece, (first, last, scale, offset, free) in enumerate(ranges):
        block = np.abs(residual[first:last + 1])
        pieces.append(Piece(
            first=first, last=last, scale=scale, offset=offset,
            # Provisional: the outer edges open up and the interior boundaries are placed
            # properly by locate_breaks, which is the pass that knows where the cues are.
            x0=-math.inf if n_piece == 0 else float(xs[first]),
            x1=math.inf if n_piece == len(ranges) - 1 else float(xs[last]),
            free_slope=free,
            residual_p50=float(np.median(block)) if len(block) else 0.0,
            residual_p90=float(np.percentile(block, 90)) if len(block) else 0.0,
        ))

    rejected = sorted(set(range(len(all_xs))) - set(int(k) for k in keep))
    return Transform(
        pieces=tuple(pieces),
        used=tuple(int(k) for k in keep),
        rejected=tuple(rejected),
        cost=cost,
        scale_bound_hit=bound_hit,
        xs=tuple(float(v) for v in xs),
        ys=tuple(float(v) for v in ys),
        ws=tuple(float(v) for v in all_ws[keep]),
        ids=tuple(int(v) for v in all_ids[keep]) if has_ids else (),
    )


# --- Placing the breakpoints ---------------------------------------------------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _damage(spans: Sequence[Tuple[float, float]], c0: float, c1: float) -> float:
    """Total fraction of cue content the interval [c0, c1] destroys.

    A *share* rather than a count, deliberately: counting cues lets one long cue be sacrificed
    to spare three short ones, which is backwards -- the long cue carries more of the script.
    """
    total = 0.0
    for start, end in spans:
        if end <= c0:
            continue
        if start >= c1:
            break
        duration = max(end - start, 1e-6)
        total += _overlap(start, end, c0, c1) / duration
    return total


def _largest_gap_midpoint(
    spans: Sequence[Tuple[float, float]], lo: float, hi: float,
) -> float:
    """The quietest point between lo and hi: the middle of the widest gap between cues."""
    edges = [lo]
    for start, end in spans:
        if end <= lo or start >= hi:
            continue
        edges.append(max(start, lo))
        edges.append(min(end, hi))
    edges.append(hi)
    # edges is [lo, start0, end0, start1, end1, ..., hi], so the gaps *between* cues are
    # exactly the consecutive (even, odd) pairs: (lo, start0), (end0, start1), ..., (endN, hi).
    best, best_width = (lo + hi) / 2.0, -1.0
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a > best_width:
            best, best_width = (a + b) / 2.0, b - a
    return min(max(best, lo), hi)


def _cut_candidates(
    spans: Sequence[Tuple[float, float]], lo: float, hi: float, length: float,
) -> List[float]:
    """Where a cut of *length* source-seconds could start, in order of interest.

    The optimum of a sum of overlaps is always attained with an edge of the interval flush
    against a cue edge, so only those O(k) positions need testing -- there is no need to
    sample the continuum.
    """
    out = {lo, max(lo, hi - length)}
    for start, end in spans:
        if end <= lo or start >= hi:
            continue
        for value in (start, end, start - length, end - length):
            if lo <= value <= hi:
                out.add(value)
    return sorted(out)


def locate_breaks(
    transform: Transform,
    spans: Sequence[Tuple[float, float]],
    *,
    cut_overlap_share: float = CUT_OVERLAP_SHARE,
    max_scale_dev: float = MAX_SCALE_DEV,
    slope_penalty: float = SLOPE_PENALTY,
    min_slope_anchors: int = MIN_SLOPE_ANCHORS,
    min_slope_span: float = MIN_SLOPE_SPAN,
) -> Transform:
    """Decide where each breakpoint actually sits, and what it costs.

    The dynamic program only knows a break lies *somewhere* between two anchors -- there are
    no anchors inside a cut, by definition. Within that bracket the timings say nothing more,
    so the break goes where it does least damage, which is the same tier ordering
    ``reference_context._split_one`` and ``build_align_input`` already use: land it in
    silence if silence exists, and otherwise destroy as little cue content as possible.

    Also repairs an **infeasible** cut: one claiming to delete more source time than exists
    between the two anchors bracketing it. That is the fit asserting that content two
    *verified* anchors say was spoken is not in the recording, so the two pieces are merged
    and refitted instead. A break that keeps coming back infeasible means the anchors are
    wrong, not the audio.
    """
    if len(transform.pieces) < 2 or not transform.xs:
        pieces = tuple(
            Piece(p.first, p.last, p.scale, p.offset, -math.inf, math.inf, p.free_slope,
                  p.residual_p50, p.residual_p90)
            for p in transform.pieces
        )
        return Transform(
            pieces=pieces, breaks=(), used=transform.used, rejected=transform.rejected,
            cost=transform.cost, scale_bound_hit=transform.scale_bound_hit,
            xs=transform.xs, ys=transform.ys, ws=transform.ws, ids=transform.ids,
        )

    ordered = sorted(spans, key=lambda s: s[0])
    xs = np.array(transform.xs)
    ys = np.array(transform.ys)
    ws = np.array(transform.ws)
    fit_kwargs = dict(
        max_scale_dev=max_scale_dev, slope_penalty=slope_penalty,
        min_slope_anchors=min_slope_anchors, min_slope_span=min_slope_span,
    )
    ranges = [(p.first, p.last, p.scale, p.offset, p.free_slope) for p in transform.pieces]

    # Merge-and-refit until every remaining break is feasible. Each pass can only shorten the
    # list, so this terminates.
    placements: List[Tuple[float, float, float, str]] = []
    for _attempt in range(len(ranges) + 1):
        placements = []
        infeasible: Optional[int] = None
        for k in range(len(ranges) - 1):
            a_first, a_last, a_scale, a_offset, _ = ranges[k]
            b_first, b_last, b_scale, b_offset, _ = ranges[k + 1]
            lo, hi = float(xs[a_last]), float(xs[b_first])
            if hi < lo:
                lo, hi = hi, lo

            def jump_at(t: float) -> float:
                return (b_scale * t + b_offset) - (a_scale * t + a_offset)

            probe = jump_at((lo + hi) / 2.0)
            if probe >= 0.0:
                point = _largest_gap_midpoint(ordered, lo, hi)
                placements.append((point, point, jump_at(point), "insert"))
                continue

            length = abs(probe) / max(b_scale, 1e-6)
            if length > hi - lo:
                infeasible = k
                break
            # Ties go to the cut nearest the middle of the bracket, which centres it in the
            # silence rather than pinning it against whichever cue happened to come first.
            middle = (lo + hi) / 2.0
            best: Optional[Tuple[float, float, float, float]] = None
            for c0 in _cut_candidates(ordered, lo, hi, length):
                delta = jump_at(c0)
                if delta >= 0.0:
                    continue
                c1 = c0 + abs(delta) / max(b_scale, 1e-6)
                key = (round(_damage(ordered, c0, c1), 9), abs((c0 + c1) / 2.0 - middle))
                if best is None or key < (best[0], best[1]):
                    best = (key[0], key[1], c0, c1)
            if best is None:
                point = _largest_gap_midpoint(ordered, lo, hi)
                placements.append((point, point, jump_at(point), "insert"))
                continue
            placements.append((best[2], best[3], jump_at(best[2]), "cut"))

        if infeasible is None:
            break
        a_first, a_last = ranges[infeasible][0], ranges[infeasible][1]
        b_last = ranges[infeasible + 1][1]
        sums = _Sums(xs, ys, ws)
        scale, offset, free = _fit_range(sums, a_first, b_last, **fit_kwargs)
        logger.warning(
            "realign: a fitted cut would delete more subtitle time than the anchors around it "
            "leave room for; merging those two transform pieces instead"
        )
        ranges[infeasible:infeasible + 2] = [(a_first, b_last, scale, offset, free)]

    residual = _residuals(ranges, xs, ys)
    pieces: List[Piece] = []
    breaks: List[Break] = []
    for k, (first, last, scale, offset, free) in enumerate(ranges):
        x0 = -math.inf if k == 0 else placements[k - 1][1]
        x1 = math.inf if k == len(ranges) - 1 else placements[k][0]
        block = np.abs(residual[first:last + 1])
        pieces.append(Piece(
            first=first, last=last, scale=scale, offset=offset, x0=x0, x1=x1,
            free_slope=free,
            residual_p50=float(np.median(block)) if len(block) else 0.0,
            residual_p90=float(np.percentile(block, 90)) if len(block) else 0.0,
        ))
    for c0, c1, jump, kind in placements:
        touched: Tuple[int, ...] = ()
        if kind == "cut":
            touched = tuple(
                i for i, (start, end) in enumerate(spans)
                if _overlap(start, end, c0, c1) > cut_overlap_share * max(end - start, 1e-6)
            )
        breaks.append(Break(source_start=c0, source_end=c1, jump=jump, kind=kind,
                            cue_indices=touched))

    return Transform(
        pieces=tuple(pieces), breaks=tuple(breaks),
        used=transform.used, rejected=transform.rejected, cost=transform.cost,
        scale_bound_hit=transform.scale_bound_hit,
        xs=transform.xs, ys=transform.ys, ws=transform.ws, ids=transform.ids,
    )


def prune_to_density(
    transform: Transform,
    *,
    target_per_minute: float = TARGET_ANCHORS_PER_MINUTE,
    min_anchors_per_piece: int = MIN_ANCHORS_PER_PIECE,
) -> Transform:
    """Thin the anchor set map_cues will trust directly down to a sparse, screened subset.

    Per piece: keep whichever anchors sit closest to the piece's own fit (smallest residual)
    up to a target count derived from the piece's own span, always keeping the two anchors
    nearest its edges regardless of their residual -- those are the ones ``locate_breaks``
    already trusted to place the cut or insertion next to this piece, and losing them would
    push map_cues's interpolation back from the boundary it is most useful right up against.

    The scale and per-piece offset are **not** re-estimated here. They were already fitted
    from every anchor the acoustic search found, including the ones this function is about
    to discard for the *separate* purpose of direct per-cue trust -- throwing them out of
    that fit too would only make the transform itself noisier for no benefit, since the whole
    point of a robust median offset and a many-anchor pooled slope is that a handful of bad
    anchors barely move either.

    Piece boundaries (``Piece.x0``/``.x1``, and every ``Break``) are untouched: this only
    changes which anchors ``Transform.xs``/``.ys``/``.ids`` expose, not where the cuts are or
    what the fitted line says. Call it after ``locate_breaks``, not before.
    """
    if not transform.xs or not transform.pieces:
        return transform
    xs = np.asarray(transform.xs, dtype=float)
    ys = np.asarray(transform.ys, dtype=float)
    ws = np.asarray(transform.ws, dtype=float)
    has_ids = bool(transform.ids)
    ids = np.asarray(transform.ids, dtype=int) if has_ids else None

    keep = np.zeros(len(xs), dtype=bool)
    for piece in transform.pieces:
        idx = np.arange(piece.first, piece.last + 1)
        if len(idx) == 0:
            continue
        residual = np.abs(ys[idx] - (piece.scale * xs[idx] + piece.offset))
        span_minutes = max((xs[idx[-1]] - xs[idx[0]]) / 60.0, 1e-9)
        target = max(min_anchors_per_piece, int(round(target_per_minute * span_minutes)))
        target = min(target, len(idx))
        order = idx[np.argsort(residual)]
        keep[order[:target]] = True
        keep[idx[0]] = True
        keep[idx[-1]] = True

    new_xs, new_ys, new_ws = xs[keep], ys[keep], ws[keep]
    new_ids = ids[keep] if has_ids else None
    new_position = np.cumsum(keep) - 1  # old index -> new index, valid wherever keep is True

    new_pieces = []
    for piece in transform.pieces:
        old_idx = np.arange(piece.first, piece.last + 1)
        survivors = old_idx[keep[old_idx]]
        if len(survivors) == 0:
            new_pieces.append(piece)  # not reachable given the edge guarantee above
            continue
        new_pieces.append(replace(
            piece, first=int(new_position[survivors[0]]), last=int(new_position[survivors[-1]]),
        ))

    return replace(
        transform, pieces=tuple(new_pieces),
        xs=tuple(float(v) for v in new_xs), ys=tuple(float(v) for v in new_ys),
        ws=tuple(float(v) for v in new_ws),
        ids=tuple(int(v) for v in new_ids) if has_ids else (),
    )


# --- Applying it ---------------------------------------------------------------------

class _AnchorLookup:
    """Answers "where does this cue's own evidence place it" for map_cues.

    Built once per call, not per cue. ``transform.xs``/``.ys`` are already source-sorted (see
    fit_transform), so every lookup here is a binary search, not a scan.
    """

    def __init__(self, transform: Transform):
        self.transform = transform
        self.xs = np.asarray(transform.xs, dtype=float)
        self.ys = np.asarray(transform.ys, dtype=float)
        self.by_line = {lid: k for k, lid in enumerate(transform.ids)}

    def _same_piece(self, piece: Piece, k: int) -> bool:
        return piece.x0 <= self.xs[k] < piece.x1

    def _local_scale(self, piece: Piece, lo: int, hi: int) -> float:
        """The slope two specific anchors imply directly, bypassing the piece's own fit."""
        dx = self.xs[hi] - self.xs[lo]
        return (self.ys[hi] - self.ys[lo]) / dx if dx > 1e-9 else piece.scale

    def position(self, line_index: int, source_start: float) -> Tuple[float, float]:
        """(start, local_scale) for *source_start*. Falls back to the piece's own fit only
        where there is no better evidence: outside the anchor range, or where the two
        nearest anchors sit in different pieces (never interpolate across a cut/insert)."""
        piece = self.transform.piece_for(source_start)
        n = len(self.xs)

        k = self.by_line.get(line_index)
        if k is not None:
            # This cue *is* an anchor: its own measured position is exactly right, and
            # nothing downstream should second-guess it with a line fitted through hundreds
            # of others. Only the duration/local-scale still needs a neighbour, since an
            # anchor's own end was never independently measured (starts only -- see
            # anchor_pairs).
            lo = k - 1 if k > 0 and self._same_piece(piece, k - 1) else k
            hi = k + 1 if k + 1 < n and self._same_piece(piece, k + 1) else k
            scale = self._local_scale(piece, lo, hi) if lo != hi else piece.scale
            return float(self.ys[k]), scale

        # Not an anchor: interpolate between the nearest anchor at/before and at/after,
        # exactly like a piecewise-linear spline through the anchors rather than one line
        # per piece. This is what keeps a filled cue close to its neighbours' own evidence
        # instead of a piece-wide average that can be off by a piece's own residual (which
        # measures in the tenths of a second, not the noise a rounded scale would add).
        hi = int(np.searchsorted(self.xs, source_start))
        lo = hi - 1
        if lo < 0 or hi >= n or not self._same_piece(piece, lo) or not self._same_piece(piece, hi):
            return piece.map(source_start), piece.scale
        scale = self._local_scale(piece, lo, hi)
        start = self.ys[lo] + (source_start - self.xs[lo]) * scale
        return start, scale


def _trim_overlaps(
    out: List[Optional[Tuple[float, float, Optional[str]]]], pad: float, min_visible: float,
) -> None:
    """Pull a cue's end back whenever its successor's start would land inside it.

    In place, and only ever moves an END earlier -- never a start, never past that cue's own
    start by more than ``min_visible`` allows -- so this can only shrink a cue, never invert
    or hide one. ``pad`` is the gap left behind rather than a flush cut, matching
    ``align_padding``'s existing meaning elsewhere in the pipeline (the release/trim pass in
    ``align()`` trims an end to ``next_start - align_padding`` for the same reason).
    """
    prev: Optional[int] = None
    for i, row in enumerate(out):
        if row is None:
            continue
        if prev is not None:
            p_start, p_end, p_reason = out[prev]
            if row[0] < p_end:
                new_end = max(p_start + min_visible, row[0] - pad)
                out[prev] = (p_start, round(new_end, 3), p_reason)
        prev = i


def _pad_starts(
    out: List[Optional[Tuple[float, float, Optional[str]]]], pad: float, file_start: float,
) -> None:
    """Shift every start earlier by *pad*, as the last thing done to any cue's timing.

    A viewer forgives a subtitle appearing a frame before the words start far more readily
    than one appearing a frame after -- see the CLAUDE.md note on why even one frame late is
    treated as a real defect here, not a rounding nuance. So every start is nudged the same
    direction on principle, not only the ones a fit happened to place late.
    """
    for i, row in enumerate(out):
        if row is None:
            continue
        start = max(file_start, row[0] - pad)
        out[i] = (round(start, 3), row[1], row[2])


def map_cues(
    transform: Transform,
    spans: Sequence[Tuple[float, float]],
    *,
    cut_policy: str = "drop",
    min_step: float = 0.04,
    min_visible: float = 0.08,
    file_start: float = 0.0,
    file_end: Optional[float] = None,
    align_padding: float = 0.0,
) -> List[Optional[Tuple[float, float, Optional[str]]]]:
    """Map every cue through *transform*. One entry per cue, in cue order.

    ``None`` means the cue was dropped, which only happens under ``cut_policy='drop'`` for a
    cue inside a cut. Otherwise each entry is ``(start, end, reason)``, reason being None for
    an ordinary mapping.

    **A cue that is itself a kept anchor gets its own measured start exactly**, not a value
    read off a line fitted through the whole piece. A piece-wide fit is an *average*: even a
    perfectly-estimated shared slope still leaves each piece's own anchors scattered around it
    by that piece's own residual (a few tenths of a second here), and that scatter is well
    past the threshold of being visible on screen. Every other cue is placed by linear
    interpolation between its two nearest anchors -- a two-point local fit, not the whole
    piece's -- and only falls back to the piece's own formula where there is no local anchor
    pair to use (outside the anchor range, or where the interpolation would cross a cut or
    insertion). ``spans[i]`` is assumed to belong to transcript line ``i``, matching every
    other convention in this module (``cut_of``/``i`` below, ``Transform.ids``).

    Durations still scale by the *local* rate implied by the nearest anchors (or the piece's
    own scale at the edges), so gaps and lengths track the actual local speed rather than a
    single piece-wide average -- the direct extension of "anchors only, starts only" to a
    cue's other endpoint, which was never itself an anchor.

    Two passes run after every cue has a first-draft timing, in this order and not the
    reverse: ``_trim_overlaps`` resolves any cue whose neighbour's own (independently derived)
    timing now runs into it, then ``_pad_starts`` shifts every start earlier by
    ``align_padding`` as the very last thing that happens. Doing the trim first means a pair
    the trim actually touched comes out of padding exactly flush (the same amount is
    subtracted from both sides of the join), not padding first and hoping the trim has
    nothing left to fix.
    """
    if cut_policy not in ("drop", "keep"):
        raise ValueError(f"unknown cut policy: {cut_policy!r}")
    cut_of: dict = {}
    for brk in transform.breaks:
        if brk.kind != "cut":
            continue
        for i in brk.cue_indices:
            cut_of[i] = brk

    lookup = _AnchorLookup(transform)
    out: List[Optional[Tuple[float, float, Optional[str]]]] = []
    stacked: dict = {}
    for i, (source_start, source_end) in enumerate(spans):
        brk = cut_of.get(i)
        if brk is not None:
            if cut_policy == "drop":
                out.append(None)
                continue
            # Stack the orphans on the cut's own image, a step apart, so a run of them stays
            # ordered and separable into cues rather than collapsing onto one instant.
            base = transform.piece_for(brk.source_start).map(brk.source_start)
            rank = stacked.get(id(brk), 0)
            stacked[id(brk)] = rank + 1
            start = base + rank * min_step
            out.append((round(start, 3), round(start + min_visible, 3), REASON_CUT))
            continue

        start, scale = lookup.position(i, source_start)
        end = start + (source_end - source_start) * scale
        start = max(start, file_start)
        end = max(end, start + min_visible)
        reason = None
        if file_end is not None and start >= file_end:
            reason = REASON_NO_AUDIO
        out.append((round(start, 3), round(end, 3), reason))

    _trim_overlaps(out, align_padding, min_visible)

    # Removing universal shift for now. Still keeping align_padding to use as padding
    # distance between subtitles.
    #
    # if align_padding:
    #    _pad_starts(out, align_padding, file_start)

    return out


# --- Reporting -----------------------------------------------------------------------

@dataclass(frozen=True)
class TransformReport:
    scale: float
    named_ratio: Optional[str]
    median_move: float
    max_move: float
    moved_over_threshold: int
    pieces: Tuple[Piece, ...]
    breaks: Tuple[Break, ...]
    cut_cues: int
    dropped_cues: int
    anchors_used: int
    anchors_rejected: int
    residual_p50: float
    residual_p90: float
    scale_bound_hit: bool = False
    refused: bool = False


def describe_transform(
    transform: Transform,
    spans: Sequence[Tuple[float, float]],
    mapped: Sequence[Optional[Tuple[float, float, Optional[str]]]],
    *,
    report_move: float = 30.0,
) -> TransformReport:
    """Summarise what the transform did, for the log and for transform.json."""
    moves = [
        row[0] - span[0]
        for row, span in zip(mapped, spans) if row is not None
    ]
    median_move = float(np.median(moves)) if moves else 0.0
    max_move = max(moves, key=abs) if moves else 0.0
    over = sum(1 for move in moves if abs(move - median_move) > report_move)
    anchors = sum(p.anchors for p in transform.pieces) or 1
    return TransformReport(
        scale=transform.scale,
        named_ratio=named_ratio(transform.scale),
        median_move=median_move,
        max_move=float(max_move),
        moved_over_threshold=over,
        pieces=transform.pieces,
        breaks=transform.breaks,
        cut_cues=sum(len(b.cue_indices) for b in transform.breaks if b.kind == "cut"),
        dropped_cues=sum(1 for row in mapped if row is None),
        anchors_used=len(transform.used),
        anchors_rejected=len(transform.rejected),
        residual_p50=sum(p.residual_p50 * p.anchors for p in transform.pieces) / anchors,
        residual_p90=max((p.residual_p90 for p in transform.pieces), default=0.0),
        scale_bound_hit=transform.scale_bound_hit,
        refused=transform.identity,
    )


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 3600):02d}:{int(seconds // 60) % 60:02d}:{int(seconds) % 60:02d}"


def report_transform(report: TransformReport, total_cues: int) -> None:
    """Say what happened, loudly where it was drastic. See CLAUDE.md on --realign_mode."""
    if report.refused:
        logger.warning(
            "realign: no usable transform was fitted; timings are unchanged. Check that the "
            "subtitle and this recording are the same content."
        )
        return

    logger.info(
        "realign: %d cue(s) mapped through %d transform piece(s) from %d anchor(s) "
        "(%d rejected)",
        total_cues, len(report.pieces), report.anchors_used, report.anchors_rejected,
    )
    for piece in report.pieces:
        logger.info(
            "realign:   [%s-%s] x%.6f %+.3fs  (%d anchors, residual p50 %.2fs p90 %.2fs)",
            _clock(piece.x0 if math.isfinite(piece.x0) else 0.0),
            _clock(piece.x1) if math.isfinite(piece.x1) else "end",
            piece.scale, piece.offset, piece.anchors,
            piece.residual_p50, piece.residual_p90,
        )
    if abs(report.scale - 1.0) > 0.0005:
        near = f" (close to a {report.named_ratio} conversion)" if report.named_ratio else ""
        logger.warning(
            "realign: SPEED CHANGE of %.4fx%s -- the recording runs %.2f%% %s than the "
            "subtitle was timed for",
            report.scale, near, abs(report.scale - 1.0) * 100.0,
            "slower" if report.scale > 1.0 else "faster",
        )
    if report.scale_bound_hit:
        logger.warning(
            "realign: a candidate speed change exceeded --realign_max_scale and was refused; "
            "raise it if the two sources really do differ in speed by that much"
        )
    logger.info(
        "realign: cues moved by a median of %+.2fs (largest %+.2fs)",
        report.median_move, report.max_move,
    )
    for brk in report.breaks:
        if brk.kind == "insert":
            logger.info(
                "realign: insertion of %.1fs at source %s -- this recording has audio the "
                "subtitle does not cover", brk.seconds, _clock(brk.source_start),
            )
        else:
            logger.warning(
                "realign: CUT of %.1fs at source %s (%.1fs of the subtitle's timeline) -- "
                "%d cue(s) have no audio here",
                brk.seconds, _clock(brk.source_start), brk.source_seconds,
                len(brk.cue_indices),
            )
    if report.dropped_cues:
        logger.warning(
            "realign: %d cue(s) dropped as having no audio in this recording (see "
            "realign/transform.json for the full list)", report.dropped_cues,
        )
    if report.residual_p90 > 1.0:
        logger.warning(
            "realign: the transform fits its anchors only loosely (residual p90 %.2fs); "
            "treat the output as approximate", report.residual_p90,
        )
    if len(report.pieces) > 8:
        logger.warning(
            "realign: %d transform pieces were needed. A file wanting this many edits is "
            "often not the same cut of the content.", len(report.pieces),
        )

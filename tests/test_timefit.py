"""The transform fitter: how many pieces, when a slope is allowed, and what a cut destroys.

Pure numbers, no model and no GPU -- which is the point of keeping timefit.py free of any
pipeline import. The acoustic half (whether the anchors are any good) is measured by
scripts/eval_realign.py against real audio; nothing here can tell you that.

Two tests carry most of the design's weight and are worth reading first:
``test_prefers_two_offset_pieces_over_one_slope`` is the "prefer cuts and insertions to speed
changes" requirement written down, and ``test_outlier_does_not_buy_itself_a_piece`` is why
MIN_SEGMENT_ANCHORS exists.
"""

import math

import numpy as np
import pytest

from cantocaptions_ai.pipeline.timefit import (
    MIN_SEGMENT_ANCHORS,
    TransformError,
    _pad_starts,
    _trim_overlaps,
    describe_transform,
    fit_transform,
    identity_transform,
    locate_breaks,
    map_cues,
    named_ratio,
    prune_to_density,
)

PAL = 25.0 / (24000.0 / 1001.0)   # 1.0427083..., the classic speed-up


def pairs(fn, count=40, step=20.0, start=0.0, weight=1.0):
    """Anchors sampled every *step* seconds of source time, mapped through *fn*."""
    return [(start + i * step, fn(start + i * step), weight) for i in range(count)]


def spans_from(xs, duration=1.5):
    return [(x, x + duration) for x in xs]


def cut_pairs(cut_at=400.0, length=30.0, count=40, step=20.0):
    """Anchors for a recording missing *length* seconds of source time from *cut_at*.

    The missing stretch carries **no anchors**, which is not a detail -- it is what a cut is.
    A fixture with anchors inside the cut is asserting that lines were both spoken and not
    spoken, and the feasibility check in locate_breaks is right to reject it.
    """
    out = []
    for i in range(count):
        x = i * step
        if cut_at <= x < cut_at + length:
            continue
        out.append((x, x if x < cut_at else x - length, 1.0))
    return out


# --- Fitting -------------------------------------------------------------------------

def test_pure_offset_is_one_piece_at_exactly_scale_one():
    t = fit_transform(pairs(lambda x: x + 3.0))
    assert len(t.pieces) == 1
    assert t.pieces[0].free_slope is False
    # Exactly 1.0, not merely close: the offset model fixes a and never estimates it.
    assert t.pieces[0].scale == 1.0
    assert t.pieces[0].offset == pytest.approx(3.0, abs=1e-9)


def test_offset_survives_jitter():
    jitter = [0.09, -0.11, 0.04, -0.07, 0.12, -0.03]
    t = fit_transform(pairs(lambda x: x + 3.0 + jitter[int(x / 20) % len(jitter)]))
    assert len(t.pieces) == 1
    assert t.pieces[0].offset == pytest.approx(3.0, abs=0.1)


def test_pal_speedup_is_recovered_as_a_single_sloped_piece():
    # The ReZero Blu-ray case: ~400 anchors over 2800s at the PAL ratio.
    t = fit_transform(pairs(lambda x: PAL * x + 3.018, count=400, step=7.0))
    assert len(t.pieces) == 1
    assert t.pieces[0].free_slope is True
    assert t.pieces[0].scale == pytest.approx(PAL, abs=1e-4)
    assert t.pieces[0].offset == pytest.approx(3.018, abs=0.05)


def test_a_slope_is_not_shrunk_toward_one():
    # A soft prior on (a-1) would bias this low. 1e-5 of slope is 0.03s at the end of a
    # 2800s file, so the fit has to be unbiased, not merely close.
    t = fit_transform(pairs(lambda x: PAL * x, count=400, step=7.0))
    assert t.pieces[0].scale == pytest.approx(PAL, abs=1e-6)


def test_slope_beyond_the_bound_is_refused_not_clamped():
    """A slope outside the bound must not be quietly clamped into range.

    Clamping would report a scale the fit does not believe, and map every cue through it. The
    honest outcome is a refusal that names the bound, because raising --realign_max_scale is
    the only thing that can fix it -- which is why TransformError carries scale_bound_hit
    rather than leaving the caller to blame a content mismatch.
    """
    with pytest.raises(TransformError) as excinfo:
        fit_transform(pairs(lambda x: 1.5 * x, count=400, step=7.0), max_scale_dev=0.25)
    assert excinfo.value.scale_bound_hit is True
    assert "max_scale" in str(excinfo.value)


def test_a_slope_inside_the_bound_is_taken():
    # The complementary case, so the bound is not just refusing everything.
    t = fit_transform(pairs(lambda x: 1.2 * x, count=400, step=7.0), max_scale_dev=0.25)
    assert len(t.pieces) == 1
    assert t.pieces[0].scale == pytest.approx(1.2, abs=1e-6)


def test_short_anchor_runs_may_not_take_a_slope():
    # MIN_SLOPE_SPAN: a slope fitted over a few seconds and extrapolated is the dangerous
    # failure, so the gate is on span and count, not on how well it happens to fit.
    t = fit_transform(pairs(lambda x: PAL * x + 1.0, count=40, step=0.5), min_slope_span=120.0)
    assert all(not p.free_slope for p in t.pieces)


def test_a_cut_becomes_two_offset_pieces():
    t = fit_transform(cut_pairs())
    assert len(t.pieces) == 2
    assert t.pieces[0].offset == pytest.approx(0.0, abs=0.01)
    assert t.pieces[1].offset == pytest.approx(-30.0, abs=0.01)


def test_an_insertion_becomes_two_offset_pieces():
    t = fit_transform(pairs(lambda x: x if x < 400 else x + 25.0, count=40, step=20.0))
    assert len(t.pieces) == 2
    assert t.pieces[1].offset == pytest.approx(25.0, abs=0.01)


def test_prefers_two_offset_pieces_over_one_slope():
    """The requirement: an edit explains an edit, a speed change does not.

    A 5 s step halfway through a 600 s span could be absorbed by a slope of 1.008, comfortably
    inside max_scale_dev. SLOPE_PENALTY is what stops it -- a real edit must be reported as
    an edit, because that is what a human has to act on.
    """
    t = fit_transform(pairs(lambda x: x if x < 300 else x - 5.0, count=61, step=10.0))
    assert len(t.pieces) == 2
    assert all(not p.free_slope for p in t.pieces)


def test_prefers_one_slope_over_an_offset_staircase():
    # The complementary case: a genuine rate change is not a sequence of cuts.
    t = fit_transform(pairs(lambda x: PAL * x, count=400, step=7.0))
    assert len(t.pieces) == 1
    assert t.pieces[0].free_slope is True


def test_a_small_jump_is_merged_away():
    t = fit_transform(cut_pairs(length=0.2))
    assert len(t.pieces) == 1


def test_outlier_does_not_buy_itself_a_piece():
    """One bad anchor must not become its own zero-residual piece. See MIN_SEGMENT_ANCHORS."""
    data = pairs(lambda x: x + 2.0, count=41, step=20.0)
    data[20] = (data[20][0], data[20][1] + 20.0, 1.0)
    t = fit_transform(data)
    assert len(t.pieces) == 1
    assert t.pieces[0].offset == pytest.approx(2.0, abs=0.01)
    assert 20 in t.rejected


def test_trimming_reports_which_anchors_it_dropped():
    data = pairs(lambda x: x + 2.0, count=41, step=20.0)
    data[7] = (data[7][0], data[7][1] - 15.0, 1.0)
    t = fit_transform(data)
    assert 7 in t.rejected
    assert 7 not in t.used
    assert len(t.used) + len(t.rejected) == len(data)


def test_too_few_anchors_raises():
    with pytest.raises(TransformError):
        fit_transform(pairs(lambda x: x + 1.0, count=MIN_SEGMENT_ANCHORS - 1))


def test_mostly_disagreeing_anchors_are_refused():
    # Half the anchors on one map, half scattered: no transform is worth applying, and
    # saying so is better than mapping everything through a fit nobody believes.
    data = pairs(lambda x: x + 1.0, count=40, step=20.0)
    for i in range(0, 40, 2):
        data[i] = (data[i][0], data[i][1] + (i % 7) * 40.0, 1.0)
    with pytest.raises(TransformError):
        fit_transform(data)


def test_degenerate_source_times_do_not_divide_by_zero():
    # Every anchor at the same source time: no slope is estimable, and that must not raise.
    data = [(100.0, 100.0 + i * 0.01, 1.0) for i in range(10)]
    t = fit_transform(data)
    assert all(not p.free_slope for p in t.pieces)
    assert math.isfinite(t.pieces[0].offset)


def test_a_cut_does_not_imply_a_speed_change():
    """Three pieces noisily estimating the SAME true speed must converge to one value.

    An edit removes or adds content; it does not change how fast the surviving footage
    plays. A short, sparsely-anchored piece pins its own slope far less precisely than the
    full file does, so fitting each piece's slope independently just adds sampling noise
    around one true value -- reproduced here at the same spans, anchor counts and noise
    level measured on the ReZero Blu-ray (three pieces cut apart by two insertions),
    whose independently-fit slopes disagreed by up to 7e-4 and, corrected for the
    hand-measured drift, converged to the same value to four decimal places.
    """
    rng = np.random.default_rng(0)
    true_scale = 1.0434
    pieces = [(0, 580, 127, 0.6), (620, 1600, 345, 3.2), (1650, 2900, 376, 5.15)]
    pairs = []
    for x0, x1, n, offset in pieces:
        xs = np.sort(rng.uniform(x0, x1, n))
        noise = rng.normal(0, 0.12, n)
        ys = true_scale * xs + offset + noise
        pairs.extend((float(x), float(y), 1.0) for x, y in zip(xs, ys))

    t = fit_transform(pairs)
    assert len(t.pieces) == 3
    scales = [p.scale for p in t.pieces]
    assert max(scales) - min(scales) < 1e-9, "pieces must share one slope, not three"
    assert scales[0] == pytest.approx(true_scale, abs=2e-4)


def test_a_genuine_partial_file_speed_change_is_not_forced_together():
    """Two pieces at *actually* different speeds must not be pooled into one.

    The shared-slope refit only wins when it fits at least as well as a piece's own
    independent slope; a compilation splicing two masters at different rates is real
    evidence a piece's own fit should keep winning that comparison.
    """
    rng = np.random.default_rng(1)
    xs1 = np.sort(rng.uniform(0, 500, 200))
    ys1 = 1.00 * xs1 + 0.5 + rng.normal(0, 0.05, 200)
    xs2 = np.sort(rng.uniform(540, 1200, 250))
    ys2 = 1.15 * xs2 - 60.0 + rng.normal(0, 0.05, 250)
    pairs = ([(float(x), float(y), 1.0) for x, y in zip(xs1, ys1)]
             + [(float(x), float(y), 1.0) for x, y in zip(xs2, ys2)])

    t = fit_transform(pairs)
    assert len(t.pieces) == 2
    assert abs(t.pieces[0].scale - t.pieces[1].scale) > 0.05
    assert t.pieces[0].scale == pytest.approx(1.0, abs=0.01)
    assert t.pieces[1].scale == pytest.approx(1.15, abs=0.01)


def test_shared_offset_resists_a_skewed_anchor_tail():
    """A piece's shared-slope offset is a weighted median, not a mean.

    A skewed minority of anchors reading consistently late is invisible to the earlier
    4-MAD outlier trim -- nothing in it is extreme enough to cross that threshold -- but a
    mean-based offset would still be dragged toward it. Measured on the ReZero Blu-ray
    after the slope-sharing fix alone: the progressive drift was 89% gone, but each piece
    still carried a roughly constant bias of its own (+0.27s / -0.23s) that turned out to
    be exactly this. Reproduced here with a 12% skewed tail per piece; a mean-based offset
    would land 0.03-0.07s off, a median-based one within a hundredth.
    """
    rng = np.random.default_rng(2)
    true_scale = 1.0434
    pieces_spec = [(0, 580, 127, 0.6), (620, 1600, 345, 3.2), (1650, 2900, 376, 5.15)]
    data = []
    for x0, x1, n, offset in pieces_spec:
        xs = np.sort(rng.uniform(x0, x1, n))
        noise = rng.normal(0, 0.1, n)
        skew = rng.random(n) < 0.12
        noise[skew] += rng.uniform(0.3, 0.7, int(skew.sum()))
        ys = true_scale * xs + offset + noise
        data.extend((float(x), float(y), 1.0) for x, y in zip(xs, ys))

    t = fit_transform(data)
    assert len(t.pieces) == 3
    for piece, (_x0, _x1, _n, true_offset) in zip(t.pieces, pieces_spec):
        assert piece.offset == pytest.approx(true_offset, abs=0.03)


def test_shared_slope_refit_is_a_noop_for_a_single_piece():
    # Nothing to pool across when there is only one piece; must not perturb the exact
    # scale==1.0 identity result the sync-mode regression tests depend on.
    t = fit_transform(pairs(lambda x: x + 3.0))
    assert len(t.pieces) == 1
    assert t.pieces[0].scale == 1.0


def test_weights_matter():
    data = pairs(lambda x: x + 1.0, count=40, step=20.0)
    data[10] = (data[10][0], data[10][1] + 4.0, 50.0)   # heavy and wrong
    light = fit_transform(data)
    # A heavy outlier still gets trimmed; the weight decides ties, not truth.
    assert light.pieces[0].offset == pytest.approx(1.0, abs=0.5)


# --- Breakpoint placement ------------------------------------------------------------

def test_a_cut_lands_in_a_cue_gap_when_one_exists():
    """A cut that fits in the silence destroys nothing, and must find it."""
    t = fit_transform(cut_pairs())
    # Cues everywhere except a 40 s hole at 390-430, which is where the cut belongs.
    cues = spans_from([x for x in range(0, 800, 10) if not 390 <= x < 430], duration=4.0)
    t = locate_breaks(t, cues)
    assert len(t.breaks) == 1
    brk = t.breaks[0]
    assert brk.kind == "cut"
    assert brk.seconds == pytest.approx(30.0, abs=0.5)
    assert brk.cue_indices == ()
    assert 385 <= brk.source_start <= 400


def test_a_cut_with_no_gap_takes_cues_and_reports_them():
    t = fit_transform(cut_pairs())
    cues = spans_from(list(range(0, 800, 10)), duration=9.0)   # wall-to-wall dialogue
    t = locate_breaks(t, cues)
    brk = t.breaks[0]
    assert brk.kind == "cut"
    assert len(brk.cue_indices) >= 2
    # The cut interval and the cues it claims have to agree.
    for i in brk.cue_indices:
        assert cues[i][0] < brk.source_end and cues[i][1] > brk.source_start


def test_cut_source_length_is_the_jump_over_the_slope():
    t = fit_transform(cut_pairs())
    t = locate_breaks(t, spans_from(list(range(0, 800, 10))))
    brk = t.breaks[0]
    assert brk.source_seconds == pytest.approx(brk.seconds / t.pieces[1].scale, abs=0.01)


def test_an_insertion_claims_no_cues():
    t = fit_transform(pairs(lambda x: x if x < 400 else x + 25.0, count=40, step=20.0))
    t = locate_breaks(t, spans_from(list(range(0, 800, 10))))
    brk = t.breaks[0]
    assert brk.kind == "insert"
    assert brk.cue_indices == ()
    assert brk.source_start == brk.source_end


def test_an_infeasible_cut_merges_its_pieces():
    """A cut cannot delete more source time than the anchors bracketing it leave room for."""
    # Anchors 20 s apart, but the second piece is 300 s earlier -- impossible.
    t = fit_transform(pairs(lambda x: x if x < 400 else x - 300.0, count=40, step=20.0))
    assert len(t.pieces) == 2
    merged = locate_breaks(t, spans_from(list(range(0, 800, 10))))
    assert len(merged.pieces) == 1
    assert merged.breaks == ()


def test_pieces_tile_the_whole_timeline():
    t = fit_transform(cut_pairs())
    t = locate_breaks(t, spans_from(list(range(0, 800, 10))))
    assert t.pieces[0].x0 == -math.inf
    assert t.pieces[-1].x1 == math.inf
    for a, b in zip(t.pieces, t.pieces[1:]):
        assert b.x0 >= a.x1      # the excised cut interval sits between them


# --- Mapping -------------------------------------------------------------------------

def test_map_preserves_proportions_inside_a_piece():
    """sync mode's whole contract: within a piece, every gap and duration scales alike."""
    t = locate_breaks(fit_transform(pairs(lambda x: PAL * x + 3.0, count=400, step=7.0)), [])
    cues = [(10.0, 12.0), (20.0, 21.0), (50.0, 56.0)]
    out = map_cues(t, cues)
    scale = t.pieces[0].scale
    for (s, e), row in zip(cues, out):
        assert (row[1] - row[0]) == pytest.approx((e - s) * scale, abs=1e-3)
    gap_in = cues[1][0] - cues[0][1]
    gap_out = out[1][0] - out[0][1]
    assert gap_out == pytest.approx(gap_in * scale, abs=1e-3)


def test_a_cue_that_is_an_anchor_gets_its_own_position_exactly():
    """An anchor is never re-derived from the line fitted through everyone else's.

    Even a well-estimated piece-wide scale still leaves each anchor scattered around it by
    that piece's own residual -- a few tenths of a second in practice, which is well past
    what's visible on screen. A cue that IS an anchor must skip that averaging entirely.
    """
    rng = np.random.default_rng(3)
    xs = np.arange(40) * 20.0
    noise = rng.normal(0, 0.15, 40)
    ys = 1.0 * xs + 3.0 + noise
    data = list(zip(xs.tolist(), ys.tolist(), [1.0] * 40))
    t = locate_breaks(fit_transform(data, ids=list(range(40))), [])
    spans = [(x, x + 1.5) for x in xs]
    out = map_cues(t, spans)
    for i in range(40):
        # map_cues rounds to milliseconds for SRT output; that rounding, not this test, is
        # the precision floor here.
        assert out[i][0] == pytest.approx(ys[i], abs=1e-3), \
            f"anchor {i} should be its own measured position, not the fitted line's"


def test_a_non_anchor_cue_interpolates_locally_not_off_the_whole_piece():
    """A filled cue is placed by its two nearest anchors, not a piece-wide average.

    Two anchors bracketing a real local dip in the data pull a non-anchor cue toward
    themselves; a piece-wide fit, blind to that local structure, would place it on the
    smooth line instead -- exactly the gap this design closes.
    """
    # Two clean anchors, offset by a known local jump, bracketing an unanchored line.
    pairs_ = [(0.0, 3.0, 1.0), (50.0, 53.5, 1.0), (100.0, 103.0, 1.0),
              (150.0, 153.0, 1.0), (200.0, 203.0, 1.0)]
    ids_ = [0, 1, 3, 4, 5]  # line 2 has no anchor of its own -- it must interpolate
    t = locate_breaks(fit_transform(pairs_, ids=ids_, min_segment_anchors=3), [])
    spans = {0: (0.0, 1.0), 1: (50.0, 51.0), 2: (75.0, 76.0), 3: (100.0, 101.0),
             4: (150.0, 151.0), 5: (200.0, 201.0)}
    ordered = [spans[i] for i in sorted(spans)]
    out = map_cues(t, ordered)
    # Line 2 (source 75) sits midway between anchors at 50->53.5 and 100->103: local
    # interpolation gives 53.5 + 0.5*(103-53.5) = 78.25, not whatever the piece-wide
    # (noisier) average slope would predict.
    assert out[2][0] == pytest.approx(78.25, abs=1e-6)


def test_interpolation_does_not_cross_a_cut():
    """The two nearest anchors by source time can straddle a cut; that must not blend them."""
    before = [(x, x + 1.0, 1.0) for x in (0.0, 20.0, 40.0, 60.0)]
    after = [(x, x - 30.0 + 1.0, 1.0) for x in (100.0, 120.0, 140.0, 160.0)]
    pairs_ = before + after
    ids_ = list(range(8))
    t = fit_transform(pairs_, ids=ids_, min_segment_anchors=3)
    spans = [(0.0, 1.0), (20.0, 21.0), (40.0, 41.0), (60.0, 61.0),
             (80.0, 81.0),  # unanchored line sitting inside the cut gap
             (100.0, 101.0), (120.0, 121.0), (140.0, 141.0), (160.0, 161.0)]
    t = locate_breaks(t, spans)
    assert len(t.breaks) == 1 and t.breaks[0].kind == "cut"
    out = map_cues(t, spans, cut_policy="keep")
    # The line inside the gap must fall back to a piece formula, not a straight blend
    # between an anchor before the cut and one after it.
    naive_blend = 1.0 + (80.0 - 0.0) / (100.0 - 0.0) * ((100.0 - 30.0 + 1.0) - 1.0)
    assert out[4][0] != pytest.approx(naive_blend, abs=1.0)


def test_duration_scales_by_the_local_rate_not_the_piece_average():
    """A cue's end still comes from its own duration, scaled by the *local* rate."""
    rng = np.random.default_rng(4)
    xs = np.arange(30) * 10.0
    ys = 1.2 * xs + 1.0 + rng.normal(0, 0.05, 30)
    data = list(zip(xs.tolist(), ys.tolist(), [1.0] * 30))
    t = locate_breaks(fit_transform(data, ids=list(range(30)), min_slope_span=50.0), [])
    spans = [(x, x + 3.0) for x in xs]
    out = map_cues(t, spans)
    for i in range(1, 29):
        local_scale = (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1])
        # Both endpoints round to milliseconds independently, so the duration can carry up
        # to 2x that rounding -- not a precision issue in the mapping itself.
        assert (out[i][1] - out[i][0]) == pytest.approx(3.0 * local_scale, abs=2e-3)


def test_without_ids_map_cues_behaves_exactly_as_before():
    """No ids given must never let a cue collide with an anchor by coincidental index.

    fit_transform used to default a missing ids to range(n), which happened to equal many
    callers' own span indices -- silently matching an unrelated cue to a stranger's anchor
    purely because both carried the same small integer. ids must come back empty instead.
    """
    t = fit_transform(pairs(lambda x: 1.05 * x + 2.0, count=40, step=20.0))
    assert t.ids == ()
    t = locate_breaks(t, [])
    spans = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]  # indices 0,1,2 exist in the anchor set too
    out = map_cues(t, spans)
    for (s, e), row in zip(spans, out):
        expected = t.pieces[0].map(s)
        assert row[0] == pytest.approx(expected, abs=1e-6)


# --- Thinning the anchor set sync mode trusts directly --------------------------------

def _noisy_transform(n=120, span=600.0, seed=5, noise=0.15):
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(0, span, n))
    ys = 1.02 * xs + 1.0 + rng.normal(0, noise, n)
    data = list(zip(xs.tolist(), ys.tolist(), [1.0] * n))
    return locate_breaks(fit_transform(data, ids=list(range(n))), [])


def test_prune_keeps_roughly_the_requested_density():
    t = _noisy_transform(n=200, span=600.0)  # 10 minutes
    pruned = prune_to_density(t, target_per_minute=3.0, min_anchors_per_piece=3)
    # ~30 anchors targeted (3/min * 10 min); a handful of slack for the two guaranteed edges.
    assert 25 <= len(pruned.xs) <= 35


def test_prune_never_drops_below_the_per_piece_floor():
    t = _noisy_transform(n=10, span=60.0)  # too short to reach the density target honestly
    pruned = prune_to_density(t, target_per_minute=3.0, min_anchors_per_piece=5)
    assert len(pruned.xs) >= 5


def test_prune_always_keeps_each_piece_edges():
    t = _noisy_transform(n=150, span=500.0)
    pruned = prune_to_density(t, target_per_minute=2.0, min_anchors_per_piece=2)
    assert min(pruned.xs) == pytest.approx(min(t.xs))
    assert max(pruned.xs) == pytest.approx(max(t.xs))


def test_prune_keeps_the_anchors_closest_to_the_fit():
    """The whole point: a confidently-wrong anchor is exactly what should be thinned away."""
    xs = np.arange(60) * 10.0
    ys = 1.0 * xs + 2.0
    # Off by enough to be the clear worst residual, but not so much that the segmentation DP
    # reads a single-point jump as a genuine edit rather than as one noisy anchor -- that
    # would confound this test with a different mechanism entirely.
    ys[30] += 3.0
    data = list(zip(xs.tolist(), ys.tolist(), [1.0] * 60))
    t = locate_breaks(fit_transform(data, ids=list(range(60)), outlier_tolerance=100.0), [])
    assert len(t.pieces) == 1 and 30 in t.ids, "fixture must stay one piece with 30 still in it"
    pruned = prune_to_density(t, target_per_minute=3.0, min_anchors_per_piece=3)
    assert 30 not in pruned.ids


def test_prune_does_not_change_the_fitted_scale_or_offset():
    t = _noisy_transform(n=200, span=600.0)
    pruned = prune_to_density(t, target_per_minute=3.0)
    assert pruned.pieces[0].scale == t.pieces[0].scale
    assert pruned.pieces[0].offset == t.pieces[0].offset


def test_prune_is_a_noop_on_an_identity_transform():
    # No anchors at all (a refused fit): must not raise, and must change nothing.
    t = identity_transform()
    assert prune_to_density(t).pieces == t.pieces


def test_pruned_anchors_fall_back_to_interpolation_not_exact_match():
    t = _noisy_transform(n=100, span=400.0, noise=0.3)
    pruned = prune_to_density(t, target_per_minute=3.0)
    dropped_ids = set(t.ids) - set(pruned.ids)
    assert dropped_ids, "fixture should have discarded at least one anchor"
    by_id = {lid: (x, y) for lid, x, y in zip(t.ids, t.xs, t.ys)}
    victim = next(iter(dropped_ids))
    vx, vy = by_id[victim]
    spans = [(vx, vx + 1.0)]
    out = map_cues(pruned, spans)
    # No longer exact -- it is now wherever interpolation between the survivors lands, which
    # is not the discarded anchor's own (possibly wrong) measured position.
    assert out[0][0] != pytest.approx(vy, abs=1e-6)


# --- Overlap trimming and the final start-padding step ---------------------------------

def test_trim_overlaps_pulls_the_earlier_cue_back():
    out = [(0.0, 5.0, None), (4.0, 6.0, None)]
    _trim_overlaps(out, pad=0.1, min_visible=0.08)
    assert out[0][1] == pytest.approx(3.9, abs=1e-6)
    assert out[1] == (4.0, 6.0, None)


def test_trim_overlaps_leaves_non_overlapping_cues_alone():
    out = [(0.0, 3.0, None), (3.5, 6.0, None)]
    before = list(out)
    _trim_overlaps(out, pad=0.1, min_visible=0.08)
    assert out == before


def test_trim_overlaps_never_inverts_a_cue():
    # The successor starts before the predecessor even begins -- an extreme case, but the
    # trimmed cue must still have end >= start + min_visible, never a negative duration.
    out = [(2.0, 5.0, None), (2.05, 6.0, None)]
    _trim_overlaps(out, pad=0.1, min_visible=0.08)
    assert out[0][1] - out[0][0] >= 0.08 - 1e-9


def test_trim_overlaps_skips_dropped_cues():
    out = [(0.0, 5.0, None), None, (4.0, 6.0, None)]
    _trim_overlaps(out, pad=0.1, min_visible=0.08)
    assert out[0][1] == pytest.approx(3.9, abs=1e-6)
    assert out[1] is None


def test_pad_starts_shifts_every_start_earlier():
    out = [(1.0, 2.0, None), (5.0, 6.0, "cut")]
    _pad_starts(out, pad=0.04, file_start=0.0)
    assert out[0] == (0.96, 2.0, None)
    assert out[1] == (4.96, 6.0, "cut")


def test_pad_starts_does_not_go_negative():
    out = [(0.02, 1.0, None)]
    _pad_starts(out, pad=0.04, file_start=0.0)
    assert out[0][0] == 0.0


def test_map_cues_with_padding_trims_then_pads_flush():
    """A pair the trim pass actually touches comes out exactly flush, not overlapping."""
    t = locate_breaks(fit_transform(pairs(lambda x: x + 1.0, count=40, step=20.0)), [])
    # Two spans placed so the first cue's own duration runs into the second's start.
    spans = [(5.0, 30.0), (25.0, 26.0)]
    out = map_cues(t, spans, align_padding=0.04)
    assert out[0][1] == pytest.approx(out[1][0], abs=1e-9)


def test_map_cues_without_padding_does_not_shift_starts():
    t = locate_breaks(fit_transform(pairs(lambda x: x + 1.0, count=40, step=20.0)), [])
    spans = [(5.0, 6.0)]
    out = map_cues(t, spans)
    assert out[0][0] == pytest.approx(6.0, abs=1e-6)


def test_map_is_the_identity_for_an_identity_transform():
    cues = [(10.0, 12.0), (20.0, 21.5)]
    out = map_cues(identity_transform(), cues)
    assert [(r[0], r[1]) for r in out] == cues


def test_drop_policy_removes_cut_cues_and_leaves_the_rest_alone():
    t = fit_transform(cut_pairs())
    cues = spans_from(list(range(0, 800, 10)), duration=9.0)
    t = locate_breaks(t, cues)
    dropped = map_cues(t, cues, cut_policy="drop")
    kept = map_cues(t, cues, cut_policy="keep")
    cut = set(t.breaks[0].cue_indices)
    assert cut, "this fixture is supposed to destroy some cues"
    assert [i for i, row in enumerate(dropped) if row is None] == sorted(cut)
    # Every cue that was not cut is identical under both policies, except the one immediately
    # before the cut under 'keep': the stacked cut cues right after it can only exist there
    # because that cue's own end was trimmed back to make room for them. That trim is the
    # correct behaviour (see _trim_overlaps), not a discrepancy between the two policies.
    before_cut = min(cut) - 1
    for i, (a, b) in enumerate(zip(dropped, kept)):
        if i not in cut and i != before_cut:
            assert a == b


def test_keep_policy_stacks_cut_cues_in_order():
    t = fit_transform(cut_pairs())
    cues = spans_from(list(range(0, 800, 10)), duration=9.0)
    t = locate_breaks(t, cues)
    kept = map_cues(t, cues, cut_policy="keep")
    cut = list(t.breaks[0].cue_indices)
    starts = [kept[i][0] for i in cut]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "a stacked run must stay separable into cues"
    for i in cut:
        assert kept[i][2] == "cut"
        assert kept[i][1] > kept[i][0], "a zero-length cue never displays"


def test_a_cue_past_the_end_of_the_audio_is_flagged():
    t = identity_transform()
    out = map_cues(t, [(10.0, 12.0), (900.0, 902.0)], file_end=500.0)
    assert out[0][2] is None
    assert out[1][2] == "no_audio"


def test_mapped_cues_are_never_zero_length():
    t = identity_transform()
    out = map_cues(t, [(10.0, 10.0)])
    assert out[0][1] > out[0][0]


def test_unknown_cut_policy_raises():
    with pytest.raises(ValueError):
        map_cues(identity_transform(), [(1.0, 2.0)], cut_policy="squeeze")


# --- Reporting -----------------------------------------------------------------------

def test_named_ratio_recognises_the_common_conversions():
    assert named_ratio(PAL) == "25 -> 23.976 fps"
    assert named_ratio(25.0 / 24.0) == "25 -> 24 fps"
    assert named_ratio(1.2) is None


def test_named_ratio_separates_neighbouring_conversions():
    # 25/23.976 and 25/24 are only 0.1% apart, so the tolerance has to be tight enough that
    # a scale near one is not reported as the other.
    assert named_ratio(PAL) != named_ratio(25.0 / 24.0)


def test_a_named_ratio_is_a_label_and_never_a_correction():
    """Naming the neighbouring conversion is useful; snapping the fit to it is not.

    The measured ReZero scale is about 1.0435 -- near PAL, but 7.9e-4 off it, which is 2.2 s
    of drift by the end of the file. Naming it helps a human understand *why* the file is out
    of sync. Rounding the fit to 1.042709 on the strength of that would put the last cue two
    seconds wrong, so the two must stay separate: the label is cosmetic, the number is not.
    """
    measured = 1.0435
    assert named_ratio(measured) == "25 -> 23.976 fps"
    t = fit_transform(pairs(lambda x: measured * x + 3.0, count=400, step=7.0))
    assert t.pieces[0].scale == pytest.approx(measured, abs=1e-6)
    assert t.pieces[0].scale != pytest.approx(PAL, abs=1e-5)


def test_describe_counts_what_changed():
    t = fit_transform(pairs(lambda x: x + 60.0))
    t = locate_breaks(t, [])
    cues = [(10.0, 12.0), (20.0, 22.0), (30.0, 32.0)]
    report = describe_transform(t, cues, map_cues(t, cues))
    assert report.median_move == pytest.approx(60.0, abs=0.01)
    assert report.dropped_cues == 0
    assert report.anchors_used > 0
    assert report.refused is False


def test_describe_marks_a_refused_fit():
    cues = [(10.0, 12.0)]
    t = identity_transform()
    report = describe_transform(t, cues, map_cues(t, cues))
    assert report.refused is True

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
    describe_transform,
    fit_transform,
    identity_transform,
    locate_breaks,
    map_cues,
    named_ratio,
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
    # Every cue that was not cut is identical under both policies.
    for i, (a, b) in enumerate(zip(dropped, kept)):
        if i not in cut:
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

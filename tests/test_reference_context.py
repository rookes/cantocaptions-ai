"""Tests for reference-subtitle context (pipeline/reference_context.py).

Covers:
- expand_intervals_to_reference: the tolerance gate, merging, the chunk_size budget,
  and the sorted-and-disjoint invariant alignment depends on
- build_segment_contexts: overlap selection, neighbour widening, templates, truncation
- overlapping_reference_indices: shared matcher, incl. the fallback llm_correction uses
"""
import unittest

from cantocaptions_ai.pipeline.reference_context import (
    CONTEXT_TEMPLATES,
    expansion_only_spans,
    _merge_intervals,
    shift_cues,
    _merge_intervals,
    _uncovered_duration,
    build_segment_contexts,
    expand_intervals_to_reference,
    overlapping_reference_indices,
)


def _iv(*pairs):
    return [{"start": s, "end": e} for s, e in pairs]


def _cues(*triples):
    return [{"start": s, "end": e, "text": t} for s, e, t in triples]


def _assert_sorted_disjoint(case, result, chunk_size=None):
    """The invariant alignment._find_vad_segment_idx relies on."""
    for prev, nxt in zip(result, result[1:]):
        case.assertLessEqual(prev["end"], nxt["start"], f"overlap in {result}")
    for seg in result:
        case.assertLess(seg["start"], seg["end"])
        if chunk_size is not None:
            case.assertLessEqual(seg["end"] - seg["start"], chunk_size + 1e-6)


# ---------------------------------------------------------------------------
# expand_intervals_to_reference
# ---------------------------------------------------------------------------

class TestExpandIntervals(unittest.TestCase):
    """Expansion is an inclusive OR: a span survives if VAD *or* the reference claims it."""

    def test_every_cue_is_unioned_in_even_when_already_covered(self):
        intervals = _iv((0.0, 10.0))
        result = expand_intervals_to_reference(
            intervals, _cues((2.0, 5.0, "hi")), padding=0.5
        )
        self.assertEqual(result, [{"start": 0.0, "end": 10.0}])

    def test_cue_extends_the_interval_on_both_sides(self):
        result = expand_intervals_to_reference(
            _iv((4.0, 4.5)), _cues((4.0, 5.0, "x")), padding=1.0
        )
        self.assertEqual(result, [{"start": 3.0, "end": 6.0}])

    def test_uncovered_cue_is_added(self):
        result = expand_intervals_to_reference(
            _iv((0.0, 5.0)), _cues((20.0, 25.0, "missed")), padding=0.5
        )
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[1]["start"], 19.5)
        self.assertAlmostEqual(result[1]["end"], 25.5)
        _assert_sorted_disjoint(self, result)

    def test_small_overhang_is_included_no_gate(self):
        # Under the old tolerance gate this cue was skipped; now it always counts.
        result = expand_intervals_to_reference(
            _iv((0.0, 10.0)), _cues((9.5, 10.5, "x")), padding=0.0
        )
        self.assertEqual(result, [{"start": 0.0, "end": 10.5}])

    def test_short_cue_mostly_uncovered_is_included(self):
        """Regression: the real Doraemon case an absolute tolerance used to drop.

        Reference cue 1216.640-1218.640 had 0.912 s (46%) outside every VAD region, which
        slipped under a 1.0 s gate and left 大雄，你好衰啊！ unrecoverable.
        """
        intervals = _iv((1168.627, 1217.728), (1218.948, 1246.129))
        cues = _cues((1216.640, 1218.640, "是日本第一大雄，你真壞"))
        result = expand_intervals_to_reference(intervals, cues, padding=0.5, chunk_size=30.0)
        covered = _merge_intervals([[r["start"], r["end"]] for r in result])
        self.assertAlmostEqual(
            _uncovered_duration(cues[0]["start"], cues[0]["end"], covered), 0.0, places=6
        )
        _assert_sorted_disjoint(self, result)

    def test_cue_bridging_two_regions_merges_not_overlaps(self):
        result = expand_intervals_to_reference(
            _iv((0.0, 5.0), (12.0, 20.0)), _cues((6.0, 11.0, "gap")), padding=1.0
        )
        self.assertEqual(result, [{"start": 0.0, "end": 20.0}])
        _assert_sorted_disjoint(self, result)

    def test_padding_is_clamped_at_zero_and_upper_bound(self):
        result = expand_intervals_to_reference(
            _iv((20.0, 25.0)), _cues((0.2, 1.0, "head"), (29.0, 30.0, "tail")),
            padding=2.0, upper_bound=30.5,
        )
        self.assertAlmostEqual(result[0]["start"], 0.0)
        self.assertAlmostEqual(result[-1]["end"], 30.5)

    def test_no_cues_is_a_passthrough(self):
        self.assertEqual(
            expand_intervals_to_reference(_iv((0.0, 5.0), (7.0, 9.0)), []),
            [{"start": 0.0, "end": 5.0}, {"start": 7.0, "end": 9.0}],
        )

    def test_touching_vad_chunks_are_not_coalesced_without_a_cue(self):
        intervals = _iv((38.88, 54.52), (54.52, 70.05), (70.05, 93.92))
        self.assertEqual(expand_intervals_to_reference(intervals, [], chunk_size=30.0), intervals)

    def test_extra_input_keys_are_dropped(self):
        intervals = [{"start": 0.0, "end": 5.0, "segments": ["ignored"]}]
        self.assertEqual(
            expand_intervals_to_reference(intervals, []), [{"start": 0.0, "end": 5.0}]
        )

    def test_unsorted_input_is_normalised(self):
        result = expand_intervals_to_reference(_iv((10.0, 15.0), (0.0, 5.0)), [])
        self.assertEqual(result, [{"start": 0.0, "end": 5.0}, {"start": 10.0, "end": 15.0}])


class TestSplitSafety(unittest.TestCase):
    """Over-budget is preferred to cutting mid-dialogue."""

    def test_splits_in_padding_not_through_speech(self):
        # Two 18s VAD runs joined by a cue whose padding spans the gap.
        intervals = _iv((0.0, 18.0), (22.0, 40.0))
        cues = _cues((19.0, 21.0, "bridge"))
        result = expand_intervals_to_reference(intervals, cues, padding=2.0, chunk_size=30.0)
        _assert_sorted_disjoint(self, result)
        # the split must land in the unprotected zone between VAD and the cue
        boundaries = [r["end"] for r in result[:-1]]
        for b in boundaries:
            self.assertTrue(18.0 <= b <= 22.0, f"split at {b} cuts through speech")

    def test_chunk_size_is_always_honoured(self):
        # 50s of unbroken VAD speech fully spanned by one cue: no silent point exists,
        # but the budget is the caller's setting and must still hold.
        intervals = _iv((0.0, 50.0))
        cues = _cues((0.0, 50.0, "one long line"))
        result = expand_intervals_to_reference(intervals, cues, padding=0.5, chunk_size=30.0)
        _assert_sorted_disjoint(self, result, chunk_size=30.0)
        self.assertGreater(len(result), 1)

    def test_splits_at_the_lowest_scoring_frame_when_no_gap_exists(self):
        import numpy as np
        intervals = _iv((0.0, 50.0))
        cues = _cues((0.0, 50.0, "one long line"))
        times = np.arange(0.0, 50.0, 0.1)
        scores = np.full(len(times), 0.9)
        scores[(times > 26.0) & (times < 26.5)] = 0.05      # the one quiet spot
        result = expand_intervals_to_reference(
            intervals, cues, padding=0.0, chunk_size=30.0, times=times, scores=scores
        )
        _assert_sorted_disjoint(self, result, chunk_size=30.0)
        self.assertAlmostEqual(result[0]["end"], 26.0, delta=0.6)

    def test_cue_gap_beats_a_quiet_frame_inside_speech(self):
        """A boundary between dialogue lines outranks a quiet frame mid-line."""
        import numpy as np
        intervals = _iv((0.0, 50.0))
        cues = _cues((0.0, 24.0, "a"), (24.4, 50.0, "b"))   # 0.4s gap at 24.2
        times = np.arange(0.0, 50.0, 0.1)
        scores = np.full(len(times), 0.9)
        scores[(times > 26.8) & (times < 27.2)] = 0.05      # quieter, but inside cue b
        result = expand_intervals_to_reference(
            intervals, cues, padding=0.0, chunk_size=30.0, times=times, scores=scores
        )
        self.assertAlmostEqual(result[0]["end"], 24.2, delta=0.15)

    def test_min_score_split_penalises_frames_inside_a_cue(self):
        """Tier 3 in isolation: equally quiet frames, one inside a cue and one not."""
        import numpy as np
        from cantocaptions_ai.pipeline.reference_context import _best_score_split
        times = np.arange(0.0, 50.0, 0.1)
        scores = np.full(len(times), 0.9)
        scores[(times > 19.8) & (times < 20.2)] = 0.05      # inside the cue
        scores[(times > 26.8) & (times < 27.2)] = 0.05      # outside every cue
        cues = _cues((0.0, 24.0, "a"))
        self.assertAlmostEqual(
            _best_score_split(15.0, 30.0, times, scores, cues), 27.0, delta=0.25
        )

    def test_falls_back_to_an_inter_cue_gap(self):
        # Continuous VAD, but the reference shows a gap between two dialogue lines.
        intervals = _iv((0.0, 50.0))
        cues = _cues((0.0, 24.0, "line one"), (26.0, 50.0, "line two"))
        result = expand_intervals_to_reference(intervals, cues, padding=0.0, chunk_size=30.0)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["end"], 25.0)   # midpoint of the 24-26 gap
        _assert_sorted_disjoint(self, result)

    def test_never_splits_inside_a_cue(self):
        intervals = _iv((0.0, 70.0))
        cues = _cues((0.0, 20.0, "a"), (21.0, 45.0, "b"), (46.0, 70.0, "c"))
        result = expand_intervals_to_reference(intervals, cues, padding=0.0, chunk_size=30.0)
        _assert_sorted_disjoint(self, result)
        for r in result[:-1]:
            for c in cues:
                self.assertFalse(
                    c["start"] < r["end"] < c["end"], f"split at {r['end']} is inside {c}"
                )


# ---------------------------------------------------------------------------
# build_segment_contexts
# ---------------------------------------------------------------------------

class TestBuildSegmentContexts(unittest.TestCase):
    def setUp(self):
        self.cues = _cues(
            (0.0, 1.0, "one"), (1.0, 2.0, "two"), (2.0, 3.0, "three"),
            (10.0, 11.0, "four"),
        )

    def test_overlapping_cues_only(self):
        segs = _iv((0.5, 2.5))
        self.assertEqual(
            build_segment_contexts(segs, self.cues, template="bare"), ["one two three"]
        )

    def test_no_overlap_yields_empty_string(self):
        # No nearest-cue fallback: a wrong context is worse than none.
        self.assertEqual(build_segment_contexts(_iv((5.0, 6.0)), self.cues), [""])

    def test_neighbours_widen_the_window(self):
        segs = _iv((1.2, 1.8))
        self.assertEqual(
            build_segment_contexts(segs, self.cues, neighbours=1, template="bare"),
            ["one two three"],
        )

    def test_neighbours_clamp_at_the_edges(self):
        segs = _iv((0.1, 0.2))
        self.assertEqual(
            build_segment_contexts(segs, self.cues, neighbours=5, template="bare"),
            ["one two three four"],
        )

    def test_templates_render(self):
        segs = _iv((0.1, 0.2))
        self.assertEqual(build_segment_contexts(segs, self.cues, template="bare"), ["one"])
        self.assertEqual(
            build_segment_contexts(segs, self.cues, template="labelled"), ["參考翻譯：one"]
        )
        self.assertTrue(
            build_segment_contexts(segs, self.cues, template="instruct")[0].endswith("one")
        )

    def test_unknown_template_raises(self):
        with self.assertRaises(ValueError):
            build_segment_contexts(_iv((0.1, 0.2)), self.cues, template="nope")

    def test_truncation_drops_whole_trailing_cues(self):
        segs = _iv((0.0, 3.0))
        # "one two" is 7 chars; adding " three" would reach 13.
        self.assertEqual(
            build_segment_contexts(segs, self.cues, template="bare", max_chars=10), ["one two"]
        )

    def test_single_oversized_cue_is_hard_truncated(self):
        cues = _cues((0.0, 1.0, "abcdefghij"))
        self.assertEqual(
            build_segment_contexts(_iv((0.0, 1.0)), cues, template="bare", max_chars=4), ["abcd"]
        )

    def test_empty_cue_text_is_skipped(self):
        cues = _cues((0.0, 1.0, ""), (1.0, 2.0, "real"))
        self.assertEqual(
            build_segment_contexts(_iv((0.0, 2.0)), cues, template="bare"), ["real"]
        )

    def test_one_context_per_segment(self):
        segs = _iv((0.0, 1.0), (5.0, 6.0), (10.0, 11.0))
        self.assertEqual(len(build_segment_contexts(segs, self.cues)), 3)

    def test_every_rendering_template_has_a_text_placeholder(self):
        # "none" is the control: it renders nothing, so it is the one template that
        # must NOT interpolate the reference text.
        for name, pattern in CONTEXT_TEMPLATES.items():
            if name == "none":
                self.assertEqual(pattern, "", name)
            else:
                self.assertIn("{text}", pattern, name)

    def test_none_template_renders_empty_for_covered_segments(self):
        # A segment that overlaps cues still gets "" -- that is what makes the decode
        # identical to a run with no reference at all, while VAD expansion still ran.
        segs = _iv((0.1, 0.2), (0.0, 3.0))
        self.assertEqual(build_segment_contexts(segs, self.cues, template="none"), ["", ""])

    def test_none_template_still_returns_one_entry_per_segment(self):
        segs = _iv((0.0, 1.0), (50.0, 51.0), (10.0, 11.0))
        self.assertEqual(len(build_segment_contexts(segs, self.cues, template="none")), 3)


# ---------------------------------------------------------------------------
# overlapping_reference_indices
# ---------------------------------------------------------------------------

class TestOverlappingReferenceIndices(unittest.TestCase):
    def test_fallback_window_picks_nearest(self):
        cues = _cues((0.0, 1.0, "a"), (10.0, 11.0, "b"))
        self.assertEqual(
            overlapping_reference_indices(_iv((1.5, 2.0)), cues, fallback_window=2.0), [[0]]
        )

    def test_fallback_window_zero_disables_it(self):
        cues = _cues((0.0, 1.0, "a"))
        self.assertEqual(
            overlapping_reference_indices(_iv((1.5, 2.0)), cues, fallback_window=0.0), [[]]
        )

    def test_fallback_respects_distance(self):
        cues = _cues((0.0, 1.0, "a"))
        self.assertEqual(
            overlapping_reference_indices(_iv((30.0, 31.0)), cues, fallback_window=2.0), [[]]
        )

    def test_empty_reference(self):
        self.assertEqual(overlapping_reference_indices(_iv((0.0, 1.0)), []), [[]])


class TestShiftCues(unittest.TestCase):
    def test_zero_offset_is_a_copy(self):
        cues = _cues((1.0, 2.0, "a"))
        self.assertEqual(shift_cues(cues, 0.0), cues)

    def test_negative_shift_moves_cues_earlier(self):
        self.assertEqual(
            shift_cues(_cues((5.0, 6.0, "a")), -1.0),
            [{"start": 4.0, "end": 5.0, "text": "a"}],
        )

    def test_start_is_clamped_at_zero(self):
        self.assertEqual(shift_cues(_cues((0.5, 2.0, "a")), -1.0)[0]["start"], 0.0)

    def test_cue_entirely_before_zero_is_dropped(self):
        self.assertEqual(shift_cues(_cues((0.2, 0.5, "a"), (9.0, 10.0, "b")), -1.0),
                         [{"start": 8.0, "end": 9.0, "text": "b"}])

    def test_other_keys_survive(self):
        out = shift_cues([{"start": 5.0, "end": 6.0, "text": "a", "extra": 1}], -1.0)
        self.assertEqual(out[0]["extra"], 1)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# expansion_only_spans + --asr_context_scope expanded
# ---------------------------------------------------------------------------

class TestExpansionOnlySpans(unittest.TestCase):
    """What the reference *added*, which is what 'expanded' scope prompts over."""

    def test_no_cues_adds_nothing(self):
        self.assertEqual(expansion_only_spans(_iv((0.0, 10.0)), [], padding=0.5), [])

    def test_cue_already_inside_vad_adds_nothing(self):
        # VAD covers 0-10; a padded cue at 2.5-5.5 is wholly inside it.
        self.assertEqual(
            expansion_only_spans(_iv((0.0, 10.0)), _cues((3.0, 5.0, "x")), padding=0.5), []
        )

    def test_cue_wholly_outside_vad_is_all_expansion(self):
        self.assertEqual(
            expansion_only_spans(_iv((0.0, 2.0)), _cues((5.0, 6.0, "x")), padding=0.5),
            [[4.5, 6.5]],
        )

    def test_partial_overlap_yields_only_the_uncovered_part(self):
        # VAD ends at 5.0; the padded cue runs 4.5-7.5, so only 5.0-7.5 is new.
        self.assertEqual(
            expansion_only_spans(_iv((0.0, 5.0)), _cues((5.0, 7.0, "x")), padding=0.5),
            [[5.0, 7.5]],
        )

    def test_cue_straddling_a_vad_gap_is_split_around_it(self):
        spans = expansion_only_spans(
            _iv((0.0, 3.0), (5.0, 8.0)), _cues((1.0, 7.0, "x")), padding=0.0
        )
        self.assertEqual(spans, [[3.0, 5.0]])

    def test_upper_bound_clamps_padding(self):
        self.assertEqual(
            expansion_only_spans(
                _iv((0.0, 1.0)), _cues((5.0, 6.0, "x")), padding=1.0, upper_bound=6.5
            ),
            [[4.0, 6.5]],
        )

    def test_spans_are_sorted_and_disjoint(self):
        spans = expansion_only_spans(
            _iv((0.0, 1.0)),
            _cues((2.0, 3.0, "a"), (3.2, 4.0, "b"), (10.0, 11.0, "c")),
            padding=0.5,
        )
        self.assertEqual(spans, sorted(spans))
        for earlier, later in zip(spans, spans[1:]):
            self.assertLessEqual(earlier[1], later[0])

    def test_matches_what_the_expander_actually_added(self):
        # The two must not drift: everything expand_intervals_to_reference emits beyond
        # the base timeline has to be accounted for by expansion_only_spans.
        base = _iv((0.0, 4.0), (20.0, 24.0))
        cues = _cues((6.0, 8.0, "a"), (12.0, 13.0, "b"))
        added = expansion_only_spans(base, cues, padding=0.5)
        out = expand_intervals_to_reference(base, cues, padding=0.5, chunk_size=30.0)
        covered = _merge_intervals(
            [[iv["start"], iv["end"]] for iv in out], touching=True
        )
        for start, end in added:
            self.assertTrue(
                any(c[0] <= start and end <= c[1] for c in covered),
                f"expansion span {[start, end]} is not in the expanded timeline",
            )


class TestRestrictToSpans(unittest.TestCase):
    """--asr_context_scope expanded: prompt only over what the reference recovered."""

    def setUp(self):
        self.cues = _cues((0.0, 1.0, "one"), (1.0, 2.0, "two"), (2.0, 3.0, "three"))

    def test_none_keeps_every_cue(self):
        self.assertEqual(
            build_segment_contexts(
                _iv((0.0, 3.0)), self.cues, template="bare", restrict_to_spans=None
            ),
            ["one two three"],
        )

    def test_only_cues_overlapping_a_span_survive(self):
        self.assertEqual(
            build_segment_contexts(
                _iv((0.0, 3.0)), self.cues, template="bare", restrict_to_spans=[[1.2, 1.8]]
            ),
            ["two"],
        )

    def test_empty_span_list_suppresses_all_context(self):
        # A segment VAD found entirely on its own recovers nothing, so it gets no prompt.
        self.assertEqual(
            build_segment_contexts(
                _iv((0.0, 3.0)), self.cues, template="bare", restrict_to_spans=[]
            ),
            [""],
        )

    def test_partly_recovered_segment_keeps_only_the_recovered_cues(self):
        # The span abuts "two" (1.0-2.0) without overlapping it and covers "three".
        self.assertEqual(
            build_segment_contexts(
                _iv((0.0, 3.0)), self.cues, template="bare", restrict_to_spans=[[2.0, 3.0]]
            ),
            ["three"],
        )

    def test_a_sliver_of_overlap_is_still_overlap(self):
        # 1.9-3.0 clips 0.1s of "two", so "two" is genuinely over recovered audio.
        self.assertEqual(
            build_segment_contexts(
                _iv((0.0, 3.0)), self.cues, template="bare", restrict_to_spans=[[1.9, 3.0]]
            ),
            ["two three"],
        )

    def test_touching_span_does_not_count_as_overlap(self):
        # A span ending exactly where a cue starts shares no audio with it.
        self.assertEqual(
            build_segment_contexts(
                _iv((0.0, 3.0)), self.cues, template="bare", restrict_to_spans=[[0.0, 1.0]]
            ),
            ["one"],
        )

    def test_filtering_applies_after_neighbour_widening(self):
        # neighbours would pull in "one" and "three"; the span must still exclude them.
        self.assertEqual(
            build_segment_contexts(
                _iv((1.2, 1.8)), self.cues, neighbours=1, template="bare",
                restrict_to_spans=[[1.2, 1.8]],
            ),
            ["two"],
        )

    def test_template_still_applies_to_the_filtered_text(self):
        self.assertEqual(
            build_segment_contexts(
                _iv((0.0, 3.0)), self.cues, template="labelled",
                restrict_to_spans=[[1.2, 1.8]],
            ),
            ["參考翻譯：two"],
        )

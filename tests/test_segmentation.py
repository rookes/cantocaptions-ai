"""Tests for cue assembly (pipeline/segmentation.py).

Covers the four passes that turn over-split aligned subsegments into displayable cues:
- pass A: adjacency merge, incl. the float-tolerance regression that used to block ~half of
  all intended merges
- pass B: noise drop ahead of the rescue pass
- pass C: short-cue rescue, its admissibility gates and its punctuation-first direction choice
- pass D: duration floor

All pure functions -- no models, no I/O.
"""

import unittest

from cantocaptions_ai.cantonese.text import SegmentationConfig
from cantocaptions_ai.pipeline.segmentation import assemble_cues


def seg(start, end, text, **extra):
    """Build a minimal aligned segment. Words/chars are irrelevant to assembly."""
    return {"start": start, "end": end, "text": text, "words": [], "chars": None, **extra}


def texts(cues):
    return [c["text"] for c in cues]


class TestAdjacencyMerge(unittest.TestCase):
    """Pass A."""

    def test_touching_cues_merge_despite_float_representation(self):
        # Regression: alignment writes end = round(next_start - align_padding, 3), so a
        # touching pair is exactly one padding apart by construction -- but
        # 311.564 - 311.524 == 0.040000000000020464, which is > the 0.04 threshold. These
        # real values come from the Triumph in the Skies debug run.
        cues = assemble_cues(
            [seg(311.2, 311.524, "唉，"), seg(311.564, 313.0, "你又想講咩緣分呀？")],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["唉，你又想講咩緣分呀？"])

    def test_hard_stop_blocks_merge(self):
        cues = assemble_cues(
            [seg(0.0, 1.0, "好呀。"), seg(1.04, 2.0, "我聽日返嚟。")],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["好呀。", "我聽日返嚟。"])

    def test_char_budget_blocks_merge(self):
        # 3 + 18 = 21 chars, over the single-line budget of 18.
        cues = assemble_cues(
            [seg(0.0, 1.0, "唔該，"), seg(1.04, 3.0, "我想問下呢班機今日幾點鐘先至會起飛呀")],
            min_cue_duration=0,
            max_chars=18,
        )
        self.assertEqual(len(cues), 2)

    def test_far_apart_cues_do_not_merge(self):
        cues = assemble_cues(
            [seg(0.0, 1.0, "唔該，"), seg(3.0, 4.0, "多謝晒")],
            min_cue_duration=0,
        )
        self.assertEqual(len(cues), 2)

    def test_trailing_whitespace_does_not_bypass_the_punctuation_gate(self):
        # A raw (unstripped) prev text ends in a space, which is in no split_chars set, so
        # the hard stop before it used to go unnoticed.
        cues = assemble_cues(
            [seg(0.0, 1.0, "好呀。 "), seg(1.04, 2.0, "我聽日返嚟。")],
            min_cue_duration=0,
        )
        self.assertEqual(len(cues), 2)


class TestNoiseDrop(unittest.TestCase):
    """Pass B."""

    def test_short_noise_cue_is_dropped_not_glued(self):
        cues = assemble_cues(
            [seg(0.0, 2.0, "你食咗飯未呀？"), seg(2.5, 2.54, "哦，"), seg(2.6, 4.0, "我啱啱食完。")],
            is_noise=lambda t: t == "哦，",
        )
        self.assertEqual(texts(cues), ["你食咗飯未呀？", "我啱啱食完。"])

    def test_long_noise_cue_is_kept(self):
        # A noise word held for a full second is a deliberate beat.
        cues = assemble_cues(
            [seg(0.0, 2.0, "你食咗飯未呀？"), seg(3.0, 4.5, "哦，")],
            is_noise=lambda t: t == "哦，",
        )
        self.assertIn("哦，", texts(cues))


class TestShortCueRescue(unittest.TestCase):
    """Pass C."""

    def test_rescue_crosses_a_hard_stop_that_pass_a_refuses(self):
        segments = [seg(0.0, 2.0, "你好嗎？"), seg(2.04, 2.08, "幾好。")]
        self.assertEqual(len(assemble_cues(segments, min_cue_duration=0)), 2)
        cues = assemble_cues(segments, min_cue_duration=0.5)
        self.assertEqual(texts(cues), ["你好嗎？幾好。"])

    def test_rescue_uses_the_wider_multiline_budget(self):
        # 2 + 17 = 19 chars: over the single-line cap, inside the two-line one.
        segments = [seg(0.0, 0.08, "嗱，"), seg(0.12, 3.0, "你試下諗下彩虹嘅盡頭究竟有啲咩嘢呀")]
        self.assertEqual(len(assemble_cues(segments, max_chars=18, rescue_max_chars=18)), 2)
        cues = assemble_cues(segments, max_chars=18, rescue_max_chars=36)
        self.assertEqual(texts(cues), ["嗱，你試下諗下彩虹嘅盡頭究竟有啲咩嘢呀"])

    def test_rescue_respects_the_gap_limit(self):
        cues = assemble_cues(
            [seg(0.0, 2.0, "你好嗎？"), seg(3.0, 3.04, "幾好。")],
            merge_gap=0.25,
        )
        self.assertEqual(len(cues), 2)

    def test_known_marker_gets_a_doubled_gap_limit(self):
        # 0.4s gap: past merge_gap, inside 2 * merge_gap.
        segments = [seg(0.0, 0.08, "嗱，"), seg(0.48, 3.0, "你知啦，")]
        self.assertEqual(len(assemble_cues(segments, merge_gap=0.25)), 2)
        cues = assemble_cues(
            segments,
            merge_gap=0.25,
            segmentation=SegmentationConfig(leading_markers=("嗱",)),
        )
        self.assertEqual(texts(cues), ["嗱，你知啦，"])

    def test_rescue_does_not_cross_a_speaker_change(self):
        cues = assemble_cues(
            [
                seg(0.0, 2.0, "你好嗎？", speaker="SPEAKER_00"),
                seg(2.04, 2.08, "幾好。", speaker="SPEAKER_01"),
                seg(2.12, 4.0, "多謝關心。", speaker="SPEAKER_01"),
            ],
        )
        self.assertEqual(texts(cues), ["你好嗎？", "幾好。多謝關心。"])
        self.assertEqual(cues[1]["speaker"], "SPEAKER_01")


class TestRescueDirection(unittest.TestCase):
    """Pass C's punctuation-first, distance-second direction choice."""

    def test_joins_forward_when_prev_ends_in_a_hard_stop(self):
        # Equidistant neighbours: the 嗱， boundary is clean, the 係咪？ boundary is not.
        cues = assemble_cues(
            [seg(0.0, 2.0, "舅仔都係仔嚟啫，係咪？"), seg(2.04, 2.08, "嗱，"), seg(2.12, 4.0, "你係型仔喎。")],
            rescue_max_chars=36,
        )
        self.assertEqual(texts(cues), ["舅仔都係仔嚟啫，係咪？", "嗱，你係型仔喎。"])

    def test_joins_backward_across_a_comma_even_when_next_is_closer(self):
        # next is 0.04s away, prev is 0.18s away -- but only the prev boundary reads cleanly.
        cues = assemble_cues(
            [seg(0.0, 2.0, "一塵不染噉話喇喎，"), seg(2.18, 2.22, "唔該。"), seg(2.26, 4.0, "OK，")],
            rescue_max_chars=36,
        )
        self.assertEqual(texts(cues), ["一塵不染噉話喇喎，唔該。", "OK，"])

    def test_known_marker_joins_forward_when_both_boundaries_are_clean(self):
        # Real E20 case: a 79 ms 嗱， between two comma-ended clauses. Both boundaries read
        # cleanly and both gaps are one padding, so without the direction rule the marker
        # gets stranded on the end of the preceding sentence.
        segments = [
            seg(1115.263, 1116.937, "好容易畀人有啲唔恭敬嘅感覺，"),
            seg(1116.977, 1117.056, "嗱，"),
            seg(1117.096, 1119.249, "甚至好容易畀人引起啲誤會喎。"),
        ]
        cues = assemble_cues(
            segments,
            rescue_max_chars=36,
            segmentation=SegmentationConfig(leading_markers=("嗱",)),
        )
        self.assertEqual(
            texts(cues),
            ["好容易畀人有啲唔恭敬嘅感覺，", "嗱，甚至好容易畀人引起啲誤會喎。"],
        )

    def test_unknown_short_cue_has_no_direction_preference(self):
        # Same shape, but the cue is not a listed marker: the tie falls back to distance,
        # and with equal gaps the earlier join wins.
        segments = [
            seg(1115.263, 1116.937, "好容易畀人有啲唔恭敬嘅感覺，"),
            seg(1116.977, 1117.056, "係，"),
            seg(1117.096, 1119.249, "甚至好容易畀人引起啲誤會喎。"),
        ]
        cues = assemble_cues(segments, rescue_max_chars=36)
        self.assertEqual(texts(cues)[0], "好容易畀人有啲唔恭敬嘅感覺，係，")

    def test_punctuation_outranks_the_direction_preference(self):
        # 吓？ is a listed marker, so it would prefer to join forwards -- but its own trailing
        # ？ makes that boundary a hard stop, while the preceding comma is clean. Punctuation
        # is the higher-priority key, so it joins backwards (which also reads correctly: 吓？
        # answers what came before rather than introducing what follows).
        cues = assemble_cues(
            [seg(0.0, 2.0, "我啱啱返到嚟，"), seg(2.04, 2.08, "吓？"), seg(2.12, 4.0, "你講咩話。")],
            rescue_max_chars=36,
            segmentation=SegmentationConfig(leading_markers=("吓",)),
        )
        self.assertEqual(texts(cues), ["我啱啱返到嚟，吓？", "你講咩話。"])

    def test_closer_neighbour_wins_when_both_boundaries_are_clean(self):
        cues = assemble_cues(
            [seg(0.0, 2.0, "喺度賴人，"), seg(2.30, 2.34, "係呀，"), seg(2.38, 4.0, "點呀")],
            rescue_max_chars=36,
        )
        self.assertEqual(texts(cues), ["喺度賴人，", "係呀，點呀"])

    def test_closer_neighbour_wins_when_neither_boundary_is_clean(self):
        cues = assemble_cues(
            [seg(0.0, 2.0, "你係咪做錯事？"), seg(2.30, 2.34, "係。"), seg(2.38, 4.0, "唔好意思。")],
            rescue_max_chars=36,
        )
        self.assertEqual(texts(cues), ["你係咪做錯事？", "係。唔好意思。"])


class TestDurationFloor(unittest.TestCase):
    """Pass D."""

    def test_isolated_short_cue_is_extended(self):
        cues = assemble_cues([seg(10.0, 10.04, "唉～")], min_cue_duration=0.5)
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0]["end"], 10.5)

    def test_extension_never_overlaps_the_next_cue(self):
        # Two cues that cannot merge (speaker change) and sit one padding apart: the first
        # has no room to grow, so it is left exactly as it was.
        cues = assemble_cues(
            [
                seg(0.0, 0.04, "呀！", speaker="A"),
                seg(0.08, 3.0, "始終真係香港啲嘢好食呀！", speaker="B"),
            ],
            min_cue_duration=0.5,
            align_padding=0.04,
        )
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0]["end"], 0.04)
        self.assertLessEqual(cues[0]["end"], cues[1]["start"])

    def test_extension_stops_short_of_the_next_cue(self):
        cues = assemble_cues(
            [seg(0.0, 0.04, "呀！", speaker="A"), seg(0.30, 3.0, "好食呀！", speaker="B")],
            min_cue_duration=0.5,
            align_padding=0.04,
        )
        self.assertAlmostEqual(cues[0]["end"], 0.26)


class TestPassToggles(unittest.TestCase):
    def test_min_cue_duration_zero_is_pass_a_only(self):
        segments = [seg(0.0, 2.0, "你好嗎？"), seg(2.04, 2.08, "幾好。")]
        cues = assemble_cues(segments, min_cue_duration=0, is_noise=lambda t: True)
        self.assertEqual(texts(cues), ["你好嗎？", "幾好。"])
        self.assertAlmostEqual(cues[1]["end"], 2.08)

    def test_empty_input(self):
        self.assertEqual(assemble_cues([]), [])

    def test_does_not_mutate_input_segments(self):
        segments = [seg(0.0, 2.0, "你好嗎？"), seg(2.04, 2.08, "幾好。")]
        before = [dict(s) for s in segments]
        assemble_cues(segments)
        self.assertEqual(segments, before)


if __name__ == "__main__":
    unittest.main()

"""Tests for cue assembly (pipeline/segmentation.py).

Covers the four passes that turn over-split aligned subsegments into displayable cues:
- pass A: adjacency merge, incl. the float-tolerance regression that used to block ~half of
  all intended merges
- pass B: noise drop ahead of the rescue pass
- pass C: short-cue rescue, its admissibility gates and its punctuation-first direction choice
- pass D: duration floor

All pure functions -- no models, no I/O.
"""

import logging
import unittest

from cantocaptions_ai.cantonese.text import SegmentationConfig
from cantocaptions_ai.pipeline.segmentation import assemble_cues
from cantocaptions_ai.utils.schema import merge_segments


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


class TestSpeakerGate(unittest.TestCase):
    """The merge veto diarization exists to drive (segmentation._same_speaker).

    The gate is deliberately one-directional: it only ever blocks a merge between two
    *differently labeled* cues. An unlabeled cue means "diarization was not confident here",
    which must behave exactly as it did before diarization existed.
    """

    def test_pass_a_will_not_merge_across_a_speaker_change(self):
        cues = assemble_cues(
            [
                seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"),
                seg(2.04, 4.0, "幾好呀，", speaker="SPEAKER_01"),
            ],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["你好嗎，", "幾好呀，"])

    def test_pass_a_merges_the_same_speaker(self):
        cues = assemble_cues(
            [
                seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"),
                seg(2.04, 4.0, "幾好呀，", speaker="SPEAKER_00"),
            ],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["你好嗎，幾好呀，"])

    def test_an_unlabeled_cue_does_not_block_a_merge(self):
        cues = assemble_cues(
            [seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"), seg(2.04, 4.0, "幾好呀，")],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["你好嗎，幾好呀，"])

    def test_labels_from_different_segments_are_not_compared(self):
        # Under --diarize_scope segment, SPEAKER_00 in one VAD segment is an unrelated voice
        # to SPEAKER_00 in the next. Vetoing on that would block every cross-segment merge on
        # no evidence, so differing scopes must read as "unknown".
        cues = assemble_cues(
            [
                seg(0.0, 2.0, "你好嗎，", speaker="S0003/SPEAKER_00"),
                seg(2.04, 4.0, "幾好呀，", speaker="S0004/SPEAKER_01"),
            ],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["你好嗎，幾好呀，"])

    def test_labels_within_one_segment_still_veto(self):
        cues = assemble_cues(
            [
                seg(0.0, 2.0, "你好嗎，", speaker="S0003/SPEAKER_00"),
                seg(2.04, 4.0, "幾好呀，", speaker="S0003/SPEAKER_01"),
            ],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["你好嗎，", "幾好呀，"])

    def test_same_scoped_speaker_merges(self):
        cues = assemble_cues(
            [
                seg(0.0, 2.0, "你好嗎，", speaker="S0003/SPEAKER_00"),
                seg(2.04, 4.0, "幾好呀，", speaker="S0003/SPEAKER_00"),
            ],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["你好嗎，幾好呀，"])

    def test_a_label_survives_a_merge_with_an_unlabeled_left_neighbour(self):
        # The regression this guards: taking seg1's speaker unconditionally dropped
        # SPEAKER_01 here, so the third cue -- a genuinely different speaker -- was then
        # free to merge in too.
        cues = assemble_cues(
            [
                seg(0.0, 2.0, "你好嗎，"),
                seg(2.04, 4.0, "幾好呀，", speaker="SPEAKER_01"),
                seg(4.04, 6.0, "唔該晒。", speaker="SPEAKER_00"),
            ],
            min_cue_duration=0,
        )
        self.assertEqual(texts(cues), ["你好嗎，幾好呀，", "唔該晒。"])
        self.assertEqual(cues[0]["speaker"], "SPEAKER_01")


class TestMergeSegmentsSpeakerKeys(unittest.TestCase):
    """utils/schema.py:merge_segments -- speaker bookkeeping across a join."""

    def test_right_hand_speaker_is_adopted_when_the_left_has_none(self):
        merged = merge_segments(seg(0.0, 1.0, "你好"), seg(1.0, 2.0, "嗎？", speaker="B"))
        self.assertEqual(merged["speaker"], "B")

    def test_left_hand_speaker_wins_when_both_are_present(self):
        merged = merge_segments(
            seg(0.0, 1.0, "你好", speaker="A"), seg(1.0, 2.0, "嗎？", speaker="A")
        )
        self.assertEqual(merged["speaker"], "A")

    def test_no_speaker_key_when_neither_side_has_one(self):
        merged = merge_segments(seg(0.0, 1.0, "你好"), seg(1.0, 2.0, "嗎？"))
        self.assertNotIn("speaker", merged)

    def test_conflict_flag_propagates_from_either_side(self):
        merged = merge_segments(
            seg(0.0, 1.0, "你好"), seg(1.0, 2.0, "嗎？", speaker_conflict=True)
        )
        self.assertTrue(merged["speaker_conflict"])

    def test_confidence_is_dropped_because_it_described_only_one_side(self):
        merged = merge_segments(
            seg(0.0, 1.0, "你好", speaker="A", speaker_confidence=0.9),
            seg(1.0, 2.0, "嗎？", speaker="A", speaker_confidence=0.72),
        )
        self.assertNotIn("speaker_confidence", merged)


class TestSpeakerVetoLogging(unittest.TestCase):
    """The INFO line reporting that diarization held two cues apart.

    A veto is only reported when diarization is the *sole* reason the merge did not happen,
    so the log can be read as "loosen --speaker_confidence and these would join".
    """

    LOGGER = "cantocaptions_ai.pipeline.segmentation"

    def veto_lines(self, segments, **kwargs):
        with self.assertLogs(self.LOGGER, level="INFO") as captured:
            # A log record of our own guarantees assertLogs never fails on an empty run.
            logging.getLogger(self.LOGGER).info("probe")
            assemble_cues(segments, **kwargs)
        return [m for m in captured.output if "held a cue boundary" in m]

    def test_pass_a_reports_a_blocked_merge(self):
        lines = self.veto_lines(
            [
                seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"),
                seg(2.04, 4.0, "幾好呀，", speaker="SPEAKER_01"),
            ],
            min_cue_duration=0,
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("00:00:02,040", lines[0])
        self.assertIn("SPEAKER_00", lines[0])
        self.assertIn("SPEAKER_01", lines[0])

    def test_silent_when_punctuation_would_have_blocked_the_merge_anyway(self):
        # 。 is a hard stop, so these never merge regardless of speaker. Reporting it would
        # send someone tuning --speaker_confidence after a boundary it does not control.
        self.assertEqual(
            self.veto_lines(
                [
                    seg(0.0, 2.0, "你好嗎。", speaker="SPEAKER_00"),
                    seg(2.04, 4.0, "幾好呀，", speaker="SPEAKER_01"),
                ],
                min_cue_duration=0,
            ),
            [],
        )

    def test_silent_for_the_same_speaker(self):
        self.assertEqual(
            self.veto_lines(
                [
                    seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"),
                    seg(2.04, 4.0, "幾好呀，", speaker="SPEAKER_00"),
                ],
                min_cue_duration=0,
            ),
            [],
        )

    def test_silent_when_diarization_did_not_run(self):
        self.assertEqual(
            self.veto_lines(
                [seg(0.0, 2.0, "你好嗎，"), seg(2.04, 4.0, "幾好呀，")], min_cue_duration=0
            ),
            [],
        )

    def test_pass_c_is_silent_when_the_other_direction_still_rescues_the_cue(self):
        # Gap 0.15 clears pass A's 0.04 threshold but not pass C's merge_gap, so anything
        # logged here comes from pass C alone. The fragment joins forwards, so the backward
        # direction diarization closed cost it nothing.
        self.assertEqual(
            self.veto_lines(
                [
                    seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"),
                    seg(2.15, 2.21, "係，", speaker="SPEAKER_01"),
                    seg(2.36, 5.0, "多謝關心。", speaker="SPEAKER_01"),
                ],
                min_cue_duration=0.5,
            ),
            [],
        )

    def test_pass_c_reports_a_fragment_stranded_in_both_directions(self):
        lines = self.veto_lines(
            [
                seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"),
                seg(2.15, 2.21, "係，", speaker="SPEAKER_01"),
                seg(2.36, 5.0, "多謝關心。", speaker="SPEAKER_02"),
            ],
            min_cue_duration=0.5,
        )
        self.assertEqual(len(lines), 2)

    def test_each_boundary_is_reported_once_despite_the_pass_c_rescan(self):
        # Pass C restarts its scan after every merge, so a stranded cue is re-examined many
        # times; the report must still name each boundary exactly once.
        segments = []
        start = 0.0
        for i in range(6):
            segments.append(seg(start, start + 0.10, f"字{i}，", speaker=f"SPEAKER_0{i % 2}"))
            start += 0.14
        lines = self.veto_lines(segments, min_cue_duration=0.5)
        self.assertEqual(len(lines), 5)
        self.assertEqual(len(set(lines)), 5)

    def test_summary_counts_the_held_boundaries(self):
        with self.assertLogs(self.LOGGER, level="INFO") as captured:
            assemble_cues(
                [
                    seg(0.0, 2.0, "你好嗎，", speaker="SPEAKER_00"),
                    seg(2.04, 4.0, "幾好呀，", speaker="SPEAKER_01"),
                ],
                min_cue_duration=0,
            )
        self.assertTrue(
            any("kept 1 cue boundary from merging" in m for m in captured.output),
            captured.output,
        )


class TestMergeDisabled(unittest.TestCase):
    """merge=False: the incoming cue boundaries are authoritative (--realign)."""

    def _cues(self):
        # Two touching cues with a punctuation-clean boundary -- pass A would join them --
        # plus a third that is too short to read.
        return [
            seg(1.0, 2.0, "你好"),
            seg(2.04, 3.0, "嗎"),
            seg(3.04, 3.14, "好呀"),
        ]

    def test_boundaries_survive_a_join_pass_a_would_have_made(self):
        merged = assemble_cues(self._cues(), min_cue_duration=0.0)
        kept = assemble_cues(self._cues(), min_cue_duration=0.0, merge=False)
        self.assertLess(len(merged), 3, "pass A should merge these when it is enabled")
        self.assertEqual(texts(kept), ["你好", "嗎", "好呀"])

    def test_the_duration_floor_still_applies(self):
        kept = assemble_cues(self._cues(), min_cue_duration=0.5, merge=False)
        self.assertEqual(len(kept), 3, "no cue should have been merged away")
        self.assertGreaterEqual(kept[-1]["end"] - kept[-1]["start"], 0.5 - 1e-6)

    def test_noise_is_still_dropped(self):
        cues = self._cues() + [seg(4.0, 4.1, "嗯")]
        kept = assemble_cues(
            cues, min_cue_duration=0.5, merge=False, is_noise=lambda t: t == "嗯",
        )
        self.assertNotIn("嗯", texts(kept))

    def test_the_input_segments_are_not_mutated(self):
        cues = self._cues()
        assemble_cues(cues, min_cue_duration=0.5, merge=False)
        self.assertEqual(cues[-1]["end"], 3.14)


if __name__ == "__main__":
    unittest.main()

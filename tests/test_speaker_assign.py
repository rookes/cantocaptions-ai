"""Tests for speaker attribution (pipeline/speaker_assign.py).

Covers the decision this module exists to make: when is a diarization result confident
enough to label a subsegment, and therefore to stop cue assembly merging it into its
neighbour. The bias is one-directional -- an unlabeled subsegment must merge exactly as it
did before diarization existed -- so most of these tests are about *not* labeling.

All pure functions -- no models, no I/O.
"""

import unittest

from cantocaptions_ai.pipeline.speaker_assign import (
    SpeakerAssignmentConfig,
    assign_speakers,
    format_stats,
    rank_shares,
    scoped_speaker,
    segment_scope_id,
    speaker_scope,
    speaker_shares,
)


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def seg(start, end, text="話", words=None):
    return {"start": start, "end": end, "text": text, "words": words or [], "chars": None}


def word(start, end, text="話"):
    return {"word": text, "start": start, "end": end, "score": 1.0}


class TestSpeakerShares(unittest.TestCase):
    def test_turn_covering_the_whole_window(self):
        shares, _ = speaker_shares([turn(0.0, 10.0, "A")], 2.0, 4.0)
        self.assertEqual(shares, {"A": 2.0})

    def test_partial_overlap_is_clipped_to_the_window(self):
        shares, _ = speaker_shares([turn(0.0, 3.0, "A"), turn(3.0, 9.0, "B")], 2.0, 4.0)
        self.assertAlmostEqual(shares["A"], 1.0)
        self.assertAlmostEqual(shares["B"], 1.0)

    def test_disjoint_turns_contribute_nothing(self):
        shares, _ = speaker_shares([turn(0.0, 1.0, "A"), turn(8.0, 9.0, "B")], 3.0, 5.0)
        self.assertEqual(shares, {})

    def test_sub_millisecond_graze_is_ignored(self):
        # A segment boundary landing a rounding error inside the next turn must not pick up
        # a spurious second speaker -- that would drag the dominant share below threshold.
        shares, _ = speaker_shares([turn(0.0, 4.0, "A"), turn(4.0, 9.0, "B")], 1.0, 4.0001)
        self.assertEqual(list(shares), ["A"])

    def test_same_speaker_across_two_turns_accumulates(self):
        shares, _ = speaker_shares([turn(0.0, 1.0, "A"), turn(2.0, 3.0, "A")], 0.0, 3.0)
        self.assertAlmostEqual(shares["A"], 2.0)

    def test_resume_index_skips_turns_that_ended_before_the_window(self):
        turns = [turn(0.0, 1.0, "A"), turn(1.0, 2.0, "B"), turn(2.0, 3.0, "C")]
        _, first = speaker_shares(turns, 2.0, 3.0)
        self.assertEqual(first, 2)


class TestRankShares(unittest.TestCase):
    def test_normalises_by_covered_time_not_window_length(self):
        # A cue padded with silence is still wholly one speaker.
        ranked = rank_shares({"A": 0.5})
        self.assertEqual(ranked, [("A", 1.0)])

    def test_orders_by_share_descending(self):
        ranked = rank_shares({"A": 1.0, "B": 3.0})
        self.assertEqual([spk for spk, _ in ranked], ["B", "A"])

    def test_empty_shares_rank_to_nothing(self):
        self.assertEqual(rank_shares({}), [])


class TestSegmentLabeling(unittest.TestCase):
    def test_clear_majority_is_labeled(self):
        segments = [seg(0.0, 2.0)]
        stats = assign_speakers(segments, [turn(0.0, 1.9, "A"), turn(1.9, 2.0, "B")])
        self.assertEqual(segments[0]["speaker"], "A")
        self.assertEqual(segments[0]["speaker_confidence"], 0.95)
        self.assertEqual(stats.labeled, 1)

    def test_ambiguous_segment_is_left_unlabeled_but_scored(self):
        # 55/45 split: below the 0.7 default, so no label -- and therefore no merge veto.
        segments = [seg(0.0, 2.0)]
        stats = assign_speakers(segments, [turn(0.0, 1.1, "A"), turn(1.1, 2.0, "B")])
        self.assertNotIn("speaker", segments[0])
        self.assertEqual(segments[0]["speaker_confidence"], 0.55)
        self.assertEqual(stats.labeled, 0)
        self.assertEqual(stats.unlabeled, 1)

    def test_segment_with_no_diarized_speech_gets_no_keys_at_all(self):
        segments = [seg(5.0, 6.0)]
        assign_speakers(segments, [turn(0.0, 1.0, "A")])
        self.assertNotIn("speaker", segments[0])
        self.assertNotIn("speaker_confidence", segments[0])

    def test_empty_turns_leave_every_segment_untouched(self):
        segments = [seg(0.0, 1.0), seg(1.0, 2.0)]
        stats = assign_speakers(segments, [])
        self.assertTrue(all("speaker" not in s for s in segments))
        self.assertEqual(stats.speakers, ())
        self.assertEqual(stats.labeled, 0)

    def test_threshold_is_inclusive(self):
        segments = [seg(0.0, 2.0)]
        assign_speakers(
            segments,
            [turn(0.0, 1.4, "A"), turn(1.4, 2.0, "B")],
            SpeakerAssignmentConfig(min_dominant_share=0.7),
        )
        self.assertEqual(segments[0]["speaker"], "A")

    def test_stale_labels_are_cleared_on_reassignment(self):
        # A --load_debug_dir replay re-runs assignment at possibly different thresholds; a
        # segment that no longer qualifies must not keep its old label.
        segments = [seg(0.0, 2.0)]
        assign_speakers(segments, [turn(0.0, 2.0, "A")])
        self.assertEqual(segments[0]["speaker"], "A")
        assign_speakers(
            segments,
            [turn(0.0, 1.1, "A"), turn(1.1, 2.0, "B")],
            SpeakerAssignmentConfig(min_dominant_share=0.9),
        )
        self.assertNotIn("speaker", segments[0])

    def test_consecutive_segments_each_get_their_own_speaker(self):
        # The sweep threads one cursor through all segments; this catches it advancing
        # too far and starving a later segment of its turn.
        segments = [seg(0.0, 1.0), seg(1.0, 2.0), seg(2.0, 3.0)]
        stats = assign_speakers(
            segments, [turn(0.0, 1.0, "A"), turn(1.0, 2.0, "B"), turn(2.0, 3.0, "A")]
        )
        self.assertEqual([s["speaker"] for s in segments], ["A", "B", "A"])
        self.assertEqual(stats.speakers, ("A", "B"))

    def test_unsorted_turns_are_handled(self):
        segments = [seg(0.0, 1.0), seg(1.0, 2.0)]
        assign_speakers(segments, [turn(1.0, 2.0, "B"), turn(0.0, 1.0, "A")])
        self.assertEqual([s["speaker"] for s in segments], ["A", "B"])


class TestConflictFlag(unittest.TestCase):
    def test_flag_is_off_by_default(self):
        segments = [seg(0.0, 2.0)]
        stats = assign_speakers(segments, [turn(0.0, 1.5, "A"), turn(1.5, 2.0, "B")])
        self.assertNotIn("speaker_conflict", segments[0])
        self.assertEqual(stats.conflicts, 0)

    def test_runner_up_above_share_is_flagged(self):
        segments = [seg(0.0, 2.0)]
        stats = assign_speakers(
            segments,
            [turn(0.0, 1.5, "A"), turn(1.5, 2.0, "B")],
            SpeakerAssignmentConfig(flag_conflicts=True),
        )
        self.assertTrue(segments[0]["speaker_conflict"])
        self.assertEqual(stats.conflicts, 1)

    def test_runner_up_below_share_is_not_flagged(self):
        segments = [seg(0.0, 2.0)]
        assign_speakers(
            segments,
            [turn(0.0, 1.9, "A"), turn(1.9, 2.0, "B")],
            SpeakerAssignmentConfig(flag_conflicts=True),
        )
        self.assertNotIn("speaker_conflict", segments[0])

    def test_a_flagged_segment_can_still_be_confidently_labeled(self):
        # 75/25 clears both thresholds: usable label, and worth a look.
        segments = [seg(0.0, 2.0)]
        assign_speakers(
            segments,
            [turn(0.0, 1.5, "A"), turn(1.5, 2.0, "B")],
            SpeakerAssignmentConfig(flag_conflicts=True),
        )
        self.assertEqual(segments[0]["speaker"], "A")
        self.assertTrue(segments[0]["speaker_conflict"])


class TestWordLabeling(unittest.TestCase):
    def test_words_get_their_own_speaker(self):
        segments = [seg(0.0, 2.0, words=[word(0.1, 0.9), word(1.1, 1.9)])]
        assign_speakers(segments, [turn(0.0, 1.0, "A"), turn(1.0, 2.0, "B")])
        self.assertEqual([w["speaker"] for w in segments[0]["words"]], ["A", "B"])

    def test_words_are_never_conflict_flagged(self):
        segments = [seg(0.0, 2.0, words=[word(0.0, 2.0)])]
        assign_speakers(
            segments,
            [turn(0.0, 1.5, "A"), turn(1.5, 2.0, "B")],
            SpeakerAssignmentConfig(flag_conflicts=True),
        )
        self.assertNotIn("speaker_conflict", segments[0]["words"][0])

    def test_words_without_timings_are_skipped(self):
        # Alignment leaves start/end off tokens it could not place.
        untimed = {"word": "，", "score": 0.0}
        segments = [seg(0.0, 2.0, words=[untimed, word(0.1, 1.9)])]
        assign_speakers(segments, [turn(0.0, 2.0, "A")])
        self.assertNotIn("speaker", untimed)
        self.assertEqual(segments[0]["words"][1]["speaker"], "A")


class TestFormatStats(unittest.TestCase):
    def test_conflicts_are_only_mentioned_when_present(self):
        segments = [seg(0.0, 1.0)]
        stats = assign_speakers(segments, [turn(0.0, 1.0, "A")])
        self.assertEqual(format_stats(stats), "1 speaker(s), 1/1 cues confidently attributed")

    def test_conflict_count_is_reported(self):
        segments = [seg(0.0, 2.0)]
        stats = assign_speakers(
            segments,
            [turn(0.0, 1.5, "A"), turn(1.5, 2.0, "B")],
            SpeakerAssignmentConfig(flag_conflicts=True),
        )
        self.assertIn("1 flagged as multi-speaker", format_stats(stats))


class TestLabelScoping(unittest.TestCase):
    """Namespacing that keeps per-segment speaker identities from being compared."""

    def test_scope_id_is_zero_padded_and_sorts_naturally(self):
        self.assertEqual(segment_scope_id(7), "S0007")
        self.assertLess(segment_scope_id(9), segment_scope_id(10))

    def test_round_trip(self):
        label = scoped_speaker(segment_scope_id(3), "SPEAKER_00")
        self.assertEqual(label, "S0003/SPEAKER_00")
        self.assertEqual(speaker_scope(label), "S0003")

    def test_unscoped_label_has_no_scope(self):
        # Whole-file labels must keep comparing directly.
        self.assertIsNone(speaker_scope("SPEAKER_00"))

    def test_none_has_no_scope(self):
        self.assertIsNone(speaker_scope(None))

    def test_separator_is_absent_from_pyannote_labels(self):
        # The split relies on this: pyannote labels use "_", never "/".
        self.assertNotIn("/", "SPEAKER_00")

    def test_scoped_turns_flow_through_assignment_unchanged(self):
        segments = [seg(0.0, 1.0), seg(1.0, 2.0)]
        stats = assign_speakers(
            segments,
            [turn(0.0, 1.0, "S0000/SPEAKER_00"), turn(1.0, 2.0, "S0001/SPEAKER_00")],
        )
        self.assertEqual(
            [s["speaker"] for s in segments], ["S0000/SPEAKER_00", "S0001/SPEAKER_00"]
        )
        self.assertEqual(stats.speakers, ("S0000/SPEAKER_00", "S0001/SPEAKER_00"))


if __name__ == "__main__":
    unittest.main()

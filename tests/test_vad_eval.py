"""Tests for the VAD metrics (utils/vad_eval.py): synthetic references, hand-checked answers."""

import math
import unittest

from cantocaptions_ai.utils.vad_eval import StreamReference, Tolerances, summarize

TOL = Tolerances(onset_collar=0.2, end_slack=0.5, end_collar_after=0.2, clip_tolerance=0.1)


def ref(speech, duration=20.0, dontcare=(), negatives=()):
    return StreamReference(duration=duration, speech=speech, dontcare=dontcare,
                           negatives=negatives, tol=TOL)


class TestCoverage(unittest.TestCase):
    def test_everything_kept(self):
        r = ref([(2.0, 4.0, "a"), (8.0, 9.0, "b")])
        s = summarize([], [r.score_coverage([(0.0, 20.0)])])
        self.assertAlmostEqual(s["speech_recall"], 1.0)
        self.assertEqual(s["cues_clipped"], 0.0)
        self.assertEqual(s["cues_missed"], 0.0)
        self.assertAlmostEqual(s["kept_frac"], 1.0)

    def test_stopping_at_the_voice_is_not_charged_for_the_subtitle_tail(self):
        # Cue runs 2.0-4.0; its last 0.5 s is presumed subtitle padding, not speech.
        r = ref([(2.0, 4.0, "a")])
        s = summarize([], [r.score_coverage([(2.0, 3.5)])])
        self.assertAlmostEqual(s["speech_recall"], 1.0)
        self.assertEqual(s["cues_clipped"], 0.0)

    def test_short_cue_keeps_at_least_half_its_span_as_core(self):
        # A 0.4 s cue loses at most half of itself (0.2 s) to the end slack, not all of it.
        r = ref([(2.0, 2.4, "a")])
        c = r.score_coverage([(0.0, 1.0)])
        self.assertAlmostEqual(c["core_s"], 0.2, places=2)
        self.assertEqual(c["cues_missed"], 1)

    def test_late_onset_clips_the_line(self):
        r = ref([(2.0, 4.0, "a")])
        c = r.score_coverage([(2.3, 5.0)])  # first 0.3 s of the line lost
        self.assertEqual(c["cues_clipped"], 1)
        self.assertEqual(c["cues_missed"], 0)
        self.assertAlmostEqual(c["core_kept_s"], 1.2, places=2)

    def test_loss_under_the_tolerance_is_not_clipping(self):
        r = ref([(2.0, 4.0, "a")])
        self.assertEqual(r.score_coverage([(2.05, 5.0)])["cues_clipped"], 0)

    def test_dropped_line(self):
        r = ref([(2.0, 4.0, "a"), (10.0, 12.0, "b")])
        s = summarize([], [r.score_coverage([(1.5, 4.5)])])
        self.assertAlmostEqual(s["cues_missed"], 0.5)
        self.assertAlmostEqual(s["speech_recall"], 0.5)

    def test_pieces_of_one_line_score_as_one_line(self):
        # One subtitle line cut across two clips arrives as two rows with one cue id.
        r = ref([(2.0, 3.0, "a"), (3.5, 5.0, "a")])
        self.assertEqual(r.score_coverage([(0.0, 20.0)])["cues"], 1)

    def test_split_counts_edges_inside_a_line_only(self):
        r = ref([(2.0, 4.0, "a"), (6.0, 8.0, "b")])
        c = r.score_coverage([(1.0, 3.0), (3.0, 5.0), (5.5, 8.5)])
        self.assertEqual(c["cues_split"], 1)

    def test_nonspeech_and_gap_accounting(self):
        # Speech 2-4 (its uncertain tail runs to 4.2), don't-care 10-11, negative clip 15-19.
        r = ref([(2.0, 4.0, "a")], dontcare=[(10.0, 11.0)], negatives=[(15.0, 19.0)])
        c = r.score_coverage([(9.0, 12.0), (15.0, 16.0)])
        self.assertAlmostEqual(c["nonspeech_s"], 20.0 - 2.2 - 1.0, places=2)
        self.assertAlmostEqual(c["nonspeech_kept_s"], 2.0 + 1.0, places=2)
        self.assertAlmostEqual(c["gap_s"], 4.0, places=2)
        self.assertAlmostEqual(c["gap_kept_s"], 1.0, places=2)

    def test_no_negatives_reports_nan_not_zero(self):
        r = ref([(2.0, 4.0, "a")])
        self.assertTrue(math.isnan(summarize([], [r.score_coverage([])])["gap_kept_frac"]))


class TestDetection(unittest.TestCase):
    def test_exact_match_is_perfect(self):
        r = ref([(2.0, 4.0, "a"), (8.0, 9.0, "b")])
        d = r.score_detection([(2.0, 4.0), (8.0, 9.0)])
        s = summarize([d], [])
        self.assertAlmostEqual(s["det_precision"], 1.0)
        self.assertAlmostEqual(s["det_recall"], 1.0)

    def test_boundary_jitter_inside_the_collars_is_free(self):
        r = ref([(2.0, 4.0, "a")])
        s = summarize([r.score_detection([(2.15, 3.6)])], [])
        self.assertAlmostEqual(s["det_f1"], 1.0)

    def test_false_alarm_and_miss(self):
        r = ref([(2.0, 4.0, "a")], duration=10.0)
        d = r.score_detection([(6.0, 7.0)])
        self.assertAlmostEqual(d["fp"], 1.0, places=2)
        # Scored speech is 2.2..3.5 (onset collar and end slack removed).
        self.assertAlmostEqual(d["fn"], 1.3, places=2)
        self.assertEqual(d["tp"], 0.0)

    def test_dontcare_is_unscored(self):
        r = ref([(2.0, 4.0, "a")], dontcare=[(6.0, 8.0)])
        d = r.score_detection([(6.0, 8.0)])
        self.assertEqual(d["fp"], 0.0)


if __name__ == "__main__":
    unittest.main()

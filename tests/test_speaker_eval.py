"""Tests for the boundary-level diarization metrics (utils/speaker_eval.py).

All pure functions -- no models, no I/O.
"""

import unittest

from cantocaptions_ai.utils.speaker_eval import (
    Boundary,
    boundary_decision,
    decision_rates,
    hyphen_turns,
    separability,
    span_label,
)


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class TestHyphenTurns(unittest.TestCase):
    def test_two_hyphen_lines_are_two_speakers(self):
        self.assertEqual(hyphen_turns("-兩蚊\n-哦，好嘅"), ["兩蚊", "哦，好嘅"])

    def test_fullwidth_and_spaced_hyphens(self):
        self.assertEqual(hyphen_turns("－ 早晨\n- 早晨呀"), ["早晨", "早晨呀"])

    def test_unhyphenated_line_continues_the_current_speaker(self):
        self.assertEqual(hyphen_turns("-你去邊度\n呀？\n-返屋企"), ["你去邊度呀？", "返屋企"])

    def test_ordinary_cues_are_not_multi_speaker(self):
        self.assertIsNone(hyphen_turns("你好"))
        self.assertIsNone(hyphen_turns("你好\n早晨"))
        # A single dash is punctuation, not the convention.
        self.assertIsNone(hyphen_turns("你好\n-早晨"))

    def test_three_speakers(self):
        self.assertEqual(hyphen_turns("-甲\n-乙\n-丙"), ["甲", "乙", "丙"])


class TestBoundaryDecision(unittest.TestCase):
    TURNS = [turn(0.0, 2.0, "A"), turn(2.0, 4.0, "B"), turn(4.0, 4.5, "A")]

    def test_different_dominant_speakers_is_diff(self):
        b = Boundary("change", (0.5, 1.8), (2.2, 3.8))
        self.assertEqual(boundary_decision(self.TURNS, b), "diff")

    def test_same_dominant_speaker_is_same(self):
        b = Boundary("same", (0.2, 1.0), (1.0, 1.9))
        self.assertEqual(boundary_decision(self.TURNS, b), "same")

    def test_side_below_dominance_is_unknown(self):
        # 1.5-2.5: half A, half B -- neither reaches 0.7.
        b = Boundary("change", (0.2, 1.0), (1.5, 2.5))
        self.assertEqual(boundary_decision(self.TURNS, b), "unknown")
        self.assertEqual(boundary_decision(self.TURNS, b, min_share=0.5), "same")

    def test_side_without_speech_is_unknown(self):
        b = Boundary("adjacent", (0.2, 1.0), (5.0, 6.0))
        self.assertEqual(boundary_decision(self.TURNS, b), "unknown")
        self.assertEqual(span_label(self.TURNS, (5.0, 6.0), 0.7), (None, None))

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            Boundary("maybe", (0, 1), (1, 2))


class TestDecisionRates(unittest.TestCase):
    def test_rates_per_kind(self):
        bs = [Boundary("change", (0, 1), (1, 2))] * 4 + [Boundary("same", (0, 1), (1, 2))] * 2
        ds = ["diff", "same", "same", "unknown", "same", "diff"]
        rates = decision_rates(bs, ds)
        self.assertEqual(rates["change"]["n"], 4)
        self.assertAlmostEqual(rates["change"]["diff"], 0.25)
        self.assertAlmostEqual(rates["change"]["unknown"], 0.25)
        self.assertAlmostEqual(rates["same"]["diff"], 0.5)
        self.assertNotIn("adjacent", rates)


class TestSeparability(unittest.TestCase):
    def test_perfect_separation(self):
        sep = separability([0.0, 0.1, 0.2], [0.5, 0.6, 0.7, 0.8])
        self.assertAlmostEqual(sep.auc, 1.0)
        self.assertAlmostEqual(sep.eer, 0.0)

    def test_no_separation(self):
        values = [0.1, 0.2, 0.3, 0.4]
        sep = separability(values, values)
        self.assertAlmostEqual(sep.auc, 0.5)

    def test_recall_at_false_split_rate(self):
        same = [i / 100 for i in range(100)]          # 0.00 .. 0.99
        change = [-1.0] * 3 + [0.5] * 7               # 3 clearly below every same score
        sep = separability(change, same, false_split_rates=(0.01,))
        self.assertAlmostEqual(sep.recall_at[0.01], 0.3)

    def test_empty_side_is_an_error(self):
        with self.assertRaises(ValueError):
            separability([], [0.1])


if __name__ == "__main__":
    unittest.main()

"""Tests for align_best_of: score N candidate texts against one shared emission,
keep the best. Runs on synthetic emissions with a five-entry vocabulary (same
style as test_realign.py's VOCAB/_emission), so the orchestration logic is
exercised without a model. The underlying CTC search (get_trellis/backtrack) is
already covered elsewhere (test_realign.py, test_align_primer.py); these tests
instead check align_best_of against get_score -- an already-established
primitive -- as the oracle for "which candidate should win", rather than
hand-predicting the DP's frame-by-frame behaviour.
"""

import unittest

import numpy as np
import torch

from cantocaptions_ai.pipeline.alignment import align_best_of, get_score

VOCAB = {"[pad]": 0, "A": 1, "B": 2, "C": 3, "D": 4}
BLANK = 0


def _emission(labels, confident=-0.05, other=-12.0):
    """Log-probs over VOCAB: a named token confident on its frame, blank/other
    everywhere else. `labels[i]` is the token confident at frame i, or None for
    a blank frame."""
    data = np.full((len(labels), len(VOCAB)), other, dtype=np.float32)
    for i, label in enumerate(labels):
        data[i, BLANK if label is None else VOCAB[label]] = confident
    return torch.from_numpy(data)


def _score(emission, text):
    return get_score(emission, [VOCAB[c] for c in text], BLANK)


class TestAlignBestOf(unittest.TestCase):
    def test_picks_the_higher_scoring_candidate(self):
        emission = _emission(["A", "A", None, "B", "B", None, "C", "C"])
        candidates = ["ABC", "DBC"]  # DBC starts on a token with no support anywhere
        scores = {c: _score(emission, c) for c in candidates}
        expected_winner = max(scores, key=scores.get)
        self.assertGreater(scores["ABC"], scores["DBC"])  # scenario is discriminating

        result = align_best_of(emission, VOCAB, BLANK, candidates)
        self.assertEqual(result.text, expected_winner)
        self.assertAlmostEqual(result.mean_score, scores[expected_winner])

    def test_candidate_with_no_dictionary_coverage_is_skipped_not_crashed(self):
        emission = _emission(["A", "A", None, "B", "B", None, "C", "C"])
        # "XYZ" has zero characters in VOCAB -- must be skipped, not raise.
        result = align_best_of(emission, VOCAB, BLANK, ["XYZ", "ABC"])
        self.assertEqual(result.text, "ABC")

    def test_all_candidates_fail_returns_none(self):
        emission = _emission(["A", "A", None, "B", "B", None, "C", "C"])
        # Every candidate has zero dictionary coverage.
        self.assertIsNone(align_best_of(emission, VOCAB, BLANK, ["XYZ", "QQQ"]))
        # A single-frame emission cannot possibly host a 3-token forced walk.
        tiny = _emission(["A"])
        self.assertIsNone(align_best_of(tiny, VOCAB, BLANK, ["ABC"]))

    def test_empty_candidate_list_returns_none(self):
        emission = _emission(["A"])
        self.assertIsNone(align_best_of(emission, VOCAB, BLANK, []))

    def test_winning_segments_match_merge_repeats_of_the_winner(self):
        emission = _emission(["A", "A", None, "B", "B", None, "C", "C"])
        result = align_best_of(emission, VOCAB, BLANK, ["ABC"])
        self.assertEqual([s.label for s in result.segments], list("ABC"))
        # Segments are contiguous and monotonic.
        for prev, nxt in zip(result.segments, result.segments[1:]):
            self.assertLessEqual(prev.end, nxt.start)


if __name__ == "__main__":
    unittest.main()

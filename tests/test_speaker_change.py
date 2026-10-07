"""Tests for speaker-change scoring (pipeline/speaker_change.py).

The feature arithmetic and the decision are pure; the networks are replaced by a fake that
returns fixed embeddings and activity, so no model or GPU is needed.
"""

import unittest

import numpy as np

from cantocaptions_ai.pipeline.speaker_change import (
    LOCAL_UNKNOWN,
    ChangeModel,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    SpeakerChangeScorer,
    boundary_pairs,
    knn_pull,
    local_change,
    mark_breaks,
    score_segments,
    trim_toward,
)
from cantocaptions_ai.utils.audio import SAMPLE_RATE


def seg(start, end, text="話"):
    return {"start": start, "end": end, "text": text, "words": [], "chars": None}


class TestBoundaryPairs(unittest.TestCase):
    def test_adjacent_pairs_with_long_enough_sides(self):
        segs = [seg(0.0, 1.0), seg(1.1, 2.0), seg(2.05, 2.2), seg(2.3, 3.0)]
        pairs = boundary_pairs(segs)
        # 1|2 scored; 2|3 and 3|4 have a 0.15 s side.
        self.assertEqual([p[0] for p in pairs], [1])
        self.assertEqual(pairs[0][1:], ((0.0, 1.0), (1.1, 2.0)))

    def test_wide_gap_is_not_scored(self):
        self.assertEqual(boundary_pairs([seg(0.0, 1.0), seg(2.5, 3.5)]), [])


class TestTrimToward(unittest.TestCase):
    def test_short_span_unchanged(self):
        self.assertEqual(trim_toward((0.0, 2.0), 2.0, 4.0), (0.0, 2.0))

    def test_left_side_keeps_its_end(self):
        self.assertEqual(trim_toward((0.0, 10.0), 10.0, 4.0), (6.0, 10.0))

    def test_right_side_keeps_its_start(self):
        self.assertEqual(trim_toward((10.0, 20.0), 10.0, 4.0), (10.0, 14.0))


class TestLocalChange(unittest.TestCase):
    # One 10 s window of 100 frames (0.1 s each) starting at t=0.
    STARTS = np.array([0.0])

    def activity(self, left_speaker, right_speaker):
        act = np.zeros((1, 100, 3), np.int8)
        act[0, 10:30, left_speaker] = 1
        act[0, 30:50, right_speaker] = 1
        return act

    def test_different_local_speakers_score_one(self):
        score = local_change(self.activity(0, 1), self.STARTS, 10.0, (1.0, 3.0), (3.0, 5.0))
        self.assertAlmostEqual(score, 1.0)

    def test_same_local_speaker_scores_zero(self):
        score = local_change(self.activity(0, 0), self.STARTS, 10.0, (1.0, 3.0), (3.0, 5.0))
        self.assertAlmostEqual(score, 0.0)

    def test_no_window_holding_both_sides_is_unknown(self):
        score = local_change(self.activity(0, 1), self.STARTS, 10.0, (1.0, 3.0), (9.0, 11.0))
        self.assertEqual(score, LOCAL_UNKNOWN)

    def test_silent_side_is_unknown(self):
        act = np.zeros((1, 100, 3), np.int8)
        act[0, 10:30, 0] = 1
        self.assertEqual(local_change(act, self.STARTS, 10.0, (1.0, 3.0), (3.0, 5.0)), LOCAL_UNKNOWN)


class TestKnnPull(unittest.TestCase):
    def test_pulls_towards_neighbours(self):
        side = np.array([1.0, 0.0])
        pool = np.array([[0.6, 0.8]] * 3 + [[-1.0, 0.0]])
        pulled = knn_pull(side, pool, k=3)
        self.assertAlmostEqual(np.linalg.norm(pulled), 1.0)
        self.assertGreater(pulled[1], 0.0)

    def test_empty_pool_is_identity(self):
        side = np.array([0.0, 1.0])
        np.testing.assert_array_equal(knn_pull(side, np.zeros((0, 2))), side)


class TestDecision(unittest.TestCase):
    def test_model_is_monotone_in_its_features(self):
        base = {"local": 0.5, "cos": 0.3, "knn_cos": 0.5}
        p = DEFAULT_MODEL.probability(base)
        self.assertGreater(DEFAULT_MODEL.probability({**base, "local": 0.9}), p)
        self.assertLess(DEFAULT_MODEL.probability({**base, "knn_cos": 0.9}), p)

    def test_mark_breaks_applies_the_threshold_and_clears_stale_marks(self):
        segs = [seg(0, 1), {**seg(1, 2), "speaker_change": 0.9},
                {**seg(2, 3), "speaker_change": 0.5, "speaker_break": True}]
        self.assertEqual(mark_breaks(segs, 0.8), 1)
        self.assertTrue(segs[1]["speaker_break"])
        self.assertNotIn("speaker_break", segs[2])
        self.assertNotIn("speaker_break", segs[0])


class FakeScorer(SpeakerChangeScorer):
    """Embeds by audio content: a constant +1 signal is voice A, -1 is voice B."""

    def __init__(self):
        class _Emb:
            dimension = 2
        super().__init__(segmentation=None, embedding=_Emb(), device="cpu")

    def embed(self, chunks, batch=32):
        return np.array([[1.0, 0.0] if np.mean(c) > 0 else [0.0, 1.0] for c in chunks])

    def activity(self, segment):
        # One window over the segment; local speaker follows the signal's sign.
        audio = segment["audio"]
        frames = 100
        act = np.zeros((1, frames, 3), np.int8)
        per = len(audio) / frames
        for f in range(frames):
            sample = audio[int(f * per)]
            act[0, f, 0 if sample > 0 else 1] = 1
        return act, np.array([segment["start"]]), (segment["end"] - segment["start"])


class TestScoreSegments(unittest.TestCase):
    def test_change_scores_high_and_continuation_low(self):
        # 6 s VAD segment: voice A for 0-4 s, voice B for 4-6 s.
        audio = np.concatenate([np.ones(4 * SAMPLE_RATE), -np.ones(2 * SAMPLE_RATE)]).astype(np.float32)
        vad = [{"start": 0.0, "end": 6.0, "audio": audio}]
        segs = [seg(0.0, 2.0), seg(2.05, 4.0), seg(4.05, 6.0)]
        records = score_segments(FakeScorer(), segs, vad)
        self.assertEqual(len(records), 2)
        self.assertLess(segs[1]["speaker_change"], DEFAULT_THRESHOLD)
        self.assertGreaterEqual(segs[2]["speaker_change"], DEFAULT_THRESHOLD)
        self.assertNotIn("speaker_change", segs[0])
        self.assertEqual(mark_breaks(segs, DEFAULT_THRESHOLD), 1)

    def test_rescoring_clears_old_scores(self):
        audio = np.ones(4 * SAMPLE_RATE, np.float32)
        vad = [{"start": 0.0, "end": 4.0, "audio": audio}]
        segs = [seg(0.0, 2.0), {**seg(2.05, 2.2), "speaker_change": 0.99}]
        score_segments(FakeScorer(), segs, vad)
        self.assertNotIn("speaker_change", segs[1])


if __name__ == "__main__":
    unittest.main()

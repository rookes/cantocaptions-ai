"""Tests for the VAD score binarizer (pipeline/vads/pyannote.py).

Drives Binarize with synthetic score curves — no model, no audio. Covers the three stages
and, in particular, that padding/min_duration_off and max_duration now compose: they used to
be mutually exclusive (setting any smoothing raised NotImplementedError whenever max_duration
was finite, which it always is, since it is the ASR chunk budget).
"""

import unittest

import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature

from cantocaptions_ai.pipeline.vads.pyannote import Binarize, Pyannote

FRAME = 0.02  # seconds per score frame


def scores_from(spans, total_duration, high=0.9, low=0.1):
    """Build a score curve that is `high` inside each (start, end) span and `low` elsewhere."""
    n = int(round(total_duration / FRAME))
    data = np.full((n, 1), low, dtype=np.float32)
    for start, end in spans:
        i0, i1 = int(round(start / FRAME)), int(round(end / FRAME))
        data[i0:i1, 0] = high
    return SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=FRAME, step=FRAME))


def regions(annotation):
    return [(round(s.start, 3), round(s.end, 3)) for s in annotation.get_timeline()]


class TestSmoothingComposesWithMaxDuration(unittest.TestCase):
    def test_padding_with_finite_max_duration_does_not_raise(self):
        # The regression: this combination used to raise NotImplementedError.
        binarize = Binarize(onset=0.5, offset=0.3, pad_onset=0.2, pad_offset=0.2,
                            min_duration_off=0.25, max_duration=28)
        out = binarize(scores_from([(1.0, 3.0)], 10.0))
        self.assertEqual(len(regions(out)), 1)

    def test_padding_widens_the_region(self):
        plain = Binarize(onset=0.5, offset=0.3, max_duration=28)(scores_from([(1.0, 3.0)], 10.0))
        padded = Binarize(onset=0.5, offset=0.3, pad_onset=0.2, pad_offset=0.2,
                          max_duration=28)(scores_from([(1.0, 3.0)], 10.0))
        (p_start, p_end), = regions(plain)
        (q_start, q_end), = regions(padded)
        self.assertAlmostEqual(q_start, p_start - 0.2, places=2)
        self.assertAlmostEqual(q_end, p_end + 0.2, places=2)


class TestGapBridging(unittest.TestCase):
    def test_short_dip_does_not_split_a_region(self):
        # 100ms of silence mid-word: one region with min_duration_off, two without.
        curve = scores_from([(1.0, 2.0), (2.1, 3.0)], 10.0)
        split = Binarize(onset=0.5, offset=0.3, max_duration=28)(curve)
        self.assertEqual(len(regions(split)), 2)

        joined = Binarize(onset=0.5, offset=0.3, min_duration_off=0.25, max_duration=28)(curve)
        self.assertEqual(len(regions(joined)), 1)

    def test_long_gap_still_splits(self):
        curve = scores_from([(1.0, 2.0), (5.0, 6.0)], 10.0)
        out = Binarize(onset=0.5, offset=0.3, min_duration_off=0.25, max_duration=28)(curve)
        self.assertEqual(len(regions(out)), 2)

    def test_min_duration_on_drops_blips(self):
        curve = scores_from([(1.0, 1.05), (3.0, 5.0)], 10.0)
        out = Binarize(onset=0.5, offset=0.3, min_duration_on=0.2, max_duration=28)(curve)
        self.assertEqual(len(regions(out)), 1)


class TestMaxDurationCap(unittest.TestCase):
    def test_long_run_is_split_below_max_duration(self):
        out = Binarize(onset=0.5, offset=0.3, max_duration=10)(scores_from([(1.0, 41.0)], 45.0))
        got = regions(out)
        self.assertGreater(len(got), 1)
        for start, end in got:
            self.assertLessEqual(end - start, 10 + 1e-6)

    def test_splits_are_contiguous_so_no_audio_is_dropped(self):
        got = regions(Binarize(onset=0.5, offset=0.3, max_duration=10)(scores_from([(1.0, 41.0)], 45.0)))
        for (_, prev_end), (next_start, _) in zip(got, got[1:]):
            self.assertAlmostEqual(prev_end, next_start, places=6)

    def test_cap_holds_after_padding_widens_regions(self):
        # Padding is applied before the cap, so it cannot push a region over the ASR budget.
        out = Binarize(onset=0.5, offset=0.3, pad_onset=0.5, pad_offset=0.5,
                       min_duration_off=0.25, max_duration=10)(scores_from([(1.0, 41.0)], 45.0))
        for start, end in regions(out):
            self.assertLessEqual(end - start, 10 + 1e-6)

    def test_split_prefers_the_lowest_scoring_frame(self):
        # A dip at 7.0-7.1 is the only low-score point in the second half of the window.
        n = int(round(20.0 / FRAME))
        data = np.full((n, 1), 0.9, dtype=np.float32)
        data[:int(0.5 / FRAME), 0] = 0.1
        data[int(7.0 / FRAME):int(7.1 / FRAME), 0] = 0.35
        curve = SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=FRAME, step=FRAME))
        got = regions(Binarize(onset=0.5, offset=0.3, max_duration=10)(curve))
        self.assertGreater(len(got), 1)
        self.assertAlmostEqual(got[0][1], 7.01, places=1)


class TestClamping(unittest.TestCase):
    def test_padding_never_produces_a_negative_start(self):
        # A negative start becomes a negative sample index when the caller slices the
        # waveform, which silently grabs audio from the end of the file.
        out = Binarize(onset=0.5, offset=0.3, pad_onset=2.0, pad_offset=2.0,
                       max_duration=28)(scores_from([(0.1, 3.0)], 10.0))
        for start, end in regions(out):
            self.assertGreaterEqual(start, 0.0)

    def test_padding_never_runs_past_the_audio(self):
        out = Binarize(onset=0.5, offset=0.3, pad_onset=2.0, pad_offset=2.0,
                       max_duration=28)(scores_from([(6.0, 9.95)], 10.0))
        for start, end in regions(out):
            self.assertLessEqual(end, 10.0 + 1e-6)


class TestCoverChunks(unittest.TestCase):
    """Split-only chunking for --realign: VAD picks the cuts, but keeps every sample."""

    def _cover(self, spans, duration, chunk_size):
        return Pyannote.cover_chunks(scores_from(spans, duration), chunk_size, duration)

    def test_chunks_tile_the_whole_file(self):
        chunks = self._cover([(2.0, 8.0), (40.0, 55.0)], 100.0, 30)
        self.assertAlmostEqual(chunks[0]["start"], 0.0, places=6)
        self.assertAlmostEqual(chunks[-1]["end"], 100.0, places=6)
        for a, b in zip(chunks, chunks[1:]):
            # Contiguous, not merely non-overlapping: a gap here is discarded audio.
            self.assertAlmostEqual(a["end"], b["start"], places=6)

    def test_every_chunk_is_within_the_budget(self):
        for duration, chunk_size in ((100.0, 30), (100.0, 10), (7.0, 30), (61.0, 20)):
            chunks = self._cover([(2.0, 8.0)], duration, chunk_size)
            self.assertTrue(chunks, f"{duration}s/{chunk_size}s produced nothing")
            for chunk in chunks:
                self.assertLessEqual(chunk["end"] - chunk["start"], chunk_size + 1e-6)
                self.assertGreater(chunk["end"], chunk["start"])

    def test_silence_only_audio_still_gets_covered(self):
        # merge_chunks returns nothing here; cover_chunks must still hand back the audio,
        # because a transcript line may exist for speech VAD scored below threshold.
        chunks = self._cover([], 70.0, 30)
        self.assertAlmostEqual(sum(c["end"] - c["start"] for c in chunks), 70.0, places=6)

    def test_cuts_prefer_the_quiet_frames(self):
        # Speech either side of a silent trough; the only cut should land in the trough.
        chunks = self._cover([(0.0, 24.0), (26.0, 50.0)], 50.0, 30)
        cuts = [c["start"] for c in chunks[1:]]
        self.assertEqual(len(cuts), 1)
        self.assertGreaterEqual(cuts[0], 24.0)
        self.assertLessEqual(cuts[0], 26.0)


class TestNoSpeech(unittest.TestCase):
    def test_all_silence_yields_no_regions(self):
        out = Binarize(onset=0.5, offset=0.3, pad_onset=0.2, min_duration_off=0.25,
                       max_duration=28)(scores_from([], 10.0))
        self.assertEqual(regions(out), [])


if __name__ == "__main__":
    unittest.main()

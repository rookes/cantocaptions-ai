"""Tests for the VAD score binarizer (pipeline/vads/pyannote.py).

Drives Binarize with synthetic score curves — no model, no audio. Covers the three stages
and, in particular, that padding/min_duration_off and max_duration now compose: they used to
be mutually exclusive (setting any smoothing raised NotImplementedError whenever max_duration
was finite, which it always is, since it is the ASR chunk budget).
"""

import unittest

import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature

from cantocaptions_ai.pipeline.vads.curve import CurveVad
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


class TestMinSplitDuration(unittest.TestCase):
    """The min-cut's search floor: None keeps the old max_duration/2 behavior; a caller-supplied
    value (Pyannote.merge_chunks passes pad_onset + pad_offset + min_duration_off) widens the
    search into the first half of the window too, so a real dip there is not ignored just
    because it is early. cover_chunks must be unaffected -- see its own tests above.

    Both dips stay well above ``offset`` (0.3) so hysteresis/smoothing never treat either as a
    real gap -- exactly like the existing test_split_prefers_the_lowest_scoring_frame above --
    isolating this to a pure _split_long search-window question.
    """

    def _curve_with_two_dips(self, total=16.0):
        # max_duration=10 so the old floor is 5.0. A deeper dip sits at 2.0-2.1 (first half,
        # unreachable by the old floor) and a shallower one at 7.0-7.1 (second half, what the
        # old code would have picked instead).
        n = int(round(total / FRAME))
        data = np.full((n, 1), 0.9, dtype=np.float32)
        data[int(2.0 / FRAME):int(2.1 / FRAME), 0] = 0.32
        data[int(7.0 / FRAME):int(7.1 / FRAME), 0] = 0.35
        return SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=FRAME, step=FRAME))

    def test_default_none_ignores_the_first_half_as_before(self):
        curve = self._curve_with_two_dips()
        got = regions(Binarize(onset=0.5, offset=0.3, max_duration=10)(curve))
        self.assertGreater(len(got), 1)
        self.assertAlmostEqual(got[0][1], 7.01, places=1)

    def test_small_floor_finds_the_deeper_earlier_dip(self):
        curve = self._curve_with_two_dips()
        got = regions(Binarize(onset=0.5, offset=0.3, max_duration=10,
                               min_split_duration=1.0)(curve))
        self.assertGreater(len(got), 1)
        self.assertAlmostEqual(got[0][1], 2.01, places=1)

    def test_floor_still_bounds_progress_near_start(self):
        # The only dip sits inside the excluded zone below the floor; the search must not
        # reach back for it, so the first split lands at (or past) the floor itself instead.
        n = int(round(20.0 / FRAME))
        data = np.full((n, 1), 0.9, dtype=np.float32)
        data[int(0.3 / FRAME):int(0.4 / FRAME), 0] = 0.4
        curve = SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=FRAME, step=FRAME))
        got = regions(Binarize(onset=0.5, offset=0.3, max_duration=10,
                               min_split_duration=1.0)(curve))
        self.assertGreater(len(got), 1)
        self.assertGreaterEqual(got[0][1], 1.0 - 1e-6)
        self.assertNotAlmostEqual(got[0][1], 0.35, places=1)

    def test_merge_chunks_wires_the_floor_from_configured_vad_params(self):
        # End-to-end through Pyannote.merge_chunks, the actual production call site. Total
        # duration is chosen just over chunk_size (10.5 > 10) so exactly one split is needed
        # and Vad.merge_chunks' own outer grouping -- which would otherwise re-absorb a small
        # early piece back into its neighbour -- cannot: the combined span already exceeds
        # chunk_size regardless of where the inner split landed, so the split survives to the
        # final output and is a direct probe of which one _split_long picked.
        n = int(round(10.5 / FRAME))
        data = np.full((n, 1), 0.9, dtype=np.float32)
        data[int(2.0 / FRAME):int(2.1 / FRAME), 0] = 0.32  # only reachable with the new floor
        curve = SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=FRAME, step=FRAME))

        merged = Pyannote.merge_chunks(curve, chunk_size=10, onset=0.5, offset=0.3,
                                       pad_onset=0.2, pad_offset=0.2, min_duration_off=0.25)
        starts = sorted(m["start"] for m in merged)
        self.assertTrue(any(abs(s - 2.01) < 0.1 for s in starts[1:]),
                        f"expected a split near 2.0s, got starts={starts}")

    def test_cover_chunks_keeps_the_old_half_window_floor(self):
        # cover_chunks builds its own Binarize(max_duration=chunk_size) with no
        # min_split_duration -- the earlier, deeper dip must NOT be picked; --realign's
        # "every piece is at least half the budget" guarantee must not move.
        chunks = Pyannote.cover_chunks(self._curve_with_two_dips(), 10, 16.0)
        cuts = sorted(c["start"] for c in chunks[1:])
        self.assertAlmostEqual(cuts[0], 7.01, places=1)


class TestNoSliverAfterACut(unittest.TestCase):
    """A region only just over max_duration must not be cut in its own trailing pad.

    The frames after the speech stops score lowest, so an unbounded search cut there and left
    a few-millisecond piece that became its own VAD chunk. Under 240 samples, alignment's
    feature extractor raised "negative dimensions are not allowed" and lost a whole batch of
    files. The curve here is that case: speech, then a tail falling away into silence, then a
    second region long enough that the sliver could not share a chunk with it.
    """

    STEP = 0.016875  # pyannote segmentation-3.0's frame step

    def _curve(self, speech_s, total_s=80.0):
        n = int(total_s / self.STEP)
        t = np.arange(n) * self.STEP
        data = np.full((n, 1), 0.02, dtype=np.float32)
        end = 5.0 + speech_s
        data[(t >= 5.0) & (t < end), 0] = 0.9
        tail = (t >= end) & (t < end + 0.4)
        data[tail, 0] = np.linspace(0.14, 0.01, tail.sum())
        data[(t >= end + 3.0) & (t < end + 29.5), 0] = 0.9
        return SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=self.STEP, step=self.STEP))

    # The pyannote settings before and after 287d846; both reach the bug.
    SETTINGS = {
        "old": dict(onset=0.45, offset=0.30, pad_onset=1.00, pad_offset=0.20, min_duration_off=0.25),
        "new": dict(onset=0.15, offset=0.15, pad_onset=0.25, pad_offset=0.20, min_duration_off=0.25),
    }

    def test_no_chunk_is_shorter_than_the_split_floor(self):
        chunk_size = 28
        for name, cfg in self.SETTINGS.items():
            floor = cfg["pad_onset"] + cfg["pad_offset"] + cfg["min_duration_off"]
            # Sweep the padded region's length across the few frames either side of chunk_size,
            # where the cut used to land in the tail.
            first = chunk_size - cfg["pad_onset"] - cfg["pad_offset"]
            for speech_s in np.arange(first - 0.1, first + 0.1, 0.001):
                chunks = Pyannote.merge_chunks(self._curve(speech_s), chunk_size, **cfg)
                for c in chunks:
                    with self.subTest(settings=name, speech_s=round(float(speech_s), 3)):
                        self.assertGreaterEqual(c["end"] - c["start"], floor - 1e-6)

    def test_the_cut_moves_before_the_tail_rather_than_leaving_a_region_whole(self):
        # The fix must still cut an over-long region; only where it cuts changes.
        cfg = self.SETTINGS["new"]
        out = Binarize(max_duration=28, min_split_duration=0.7, **cfg)(self._curve(27.56))
        for start, end in regions(out):
            self.assertLessEqual(end - start, 28 + 1e-6)
            self.assertGreaterEqual(end - start, 0.7 - 1e-6)

    def test_cover_chunks_leaves_no_short_last_piece(self):
        # Contiguous chunking used to allow a short final piece; it now meets the same floor.
        for duration in (28.2, 28.5, 56.1, 60.0):
            chunks = Pyannote.cover_chunks(scores_from([(2.0, 8.0)], duration), 28, duration)
            for c in chunks:
                with self.subTest(duration=duration):
                    self.assertGreaterEqual(c["end"] - c["start"], 14 - 1e-6)
                    self.assertLessEqual(c["end"] - c["start"], 28 + 1e-6)


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


def reference_hysteresis(onset, offset, timestamps, k_scores):
    """The frame-by-frame loop Binarize._hysteresis replaced, kept as its specification."""
    regions = []
    start = timestamps[0]
    is_active = k_scores[0] > onset
    t = start
    for t, y in zip(timestamps[1:], k_scores[1:]):
        if is_active:
            if y < offset:
                regions.append((start, t))
                is_active = False
        elif y > onset:
            start = t
            is_active = True
    if is_active:
        regions.append((start, t))
    return regions


class TestHysteresisMatchesFrameLoop(unittest.TestCase):
    def test_random_curves(self):
        rng = np.random.default_rng(0)
        for trial in range(300):
            n = int(rng.integers(1, 400))
            # Mix smooth and jagged curves so runs of every length occur.
            y = rng.random(n) if trial % 2 else np.clip(np.cumsum(rng.normal(0, 0.15, n)), 0, 1)
            y = y.astype(np.float32)
            t = [i * FRAME + FRAME / 2 for i in range(n)]
            onset = float(rng.uniform(0.2, 0.8))
            # Includes offset > onset, where the closing frame must not re-open a region.
            offset = float(rng.uniform(0.05, 0.9))
            got = Binarize(onset=onset, offset=offset)._hysteresis(t, y)
            self.assertEqual(got, reference_hysteresis(onset, offset, t, y), (trial, onset, offset))

    def test_scores_exactly_at_threshold_neither_open_nor_close(self):
        t = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = np.array([0.5, 0.6, 0.3, 0.2, 0.7], dtype=np.float32)
        got = Binarize(onset=0.5, offset=0.3)._hysteresis(t, y)
        self.assertEqual(got, reference_hysteresis(0.5, 0.3, t, y))
        self.assertEqual(got, [(1.0, 3.0), (4.0, 4.0)])


class TestNoSpeech(unittest.TestCase):
    def test_all_silence_yields_no_regions(self):
        out = Binarize(onset=0.5, offset=0.3, pad_onset=0.2, min_duration_off=0.25,
                       max_duration=28)(scores_from([], 10.0))
        self.assertEqual(regions(out), [])


if __name__ == "__main__":
    unittest.main()


class TestChunksStayInsideTheAudio(unittest.TestCase):
    """The score curve's frame grid can end past the audio (pyannote's last window reaches up
    to ~60 ms beyond the final sample), and Binarize pads speech out to the grid's end. A
    chunk whose end passed the audio got a shorter slice than its timestamps said, and
    alignment, spacing emission frames over end - start, stretched its timings late."""

    SR = 16000
    DURATION = 10.0

    class _GridPastTheEnd(CurveVad):
        def __init__(self):
            super().__init__(0.5)

        @staticmethod
        def preprocess_audio(audio):
            return audio

        def __call__(self, audio, hook=None, **kwargs):
            window, step = 0.062, 0.017
            # As pyannote does: frames until one ends past the audio.
            n = int(np.ceil((TestChunksStayInsideTheAudio.DURATION - window) / step)) + 2
            starts = np.arange(n) * step
            data = np.where(starts >= 6.0, 0.9, 0.05).astype(np.float32)[:, None]
            return SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=window, step=step))

    def _process(self, **kwargs):
        from cantocaptions_ai.pipeline.vad import VadProcessor
        audio = np.zeros(int(self.DURATION * self.SR), dtype=np.float32)
        processor = VadProcessor(self._GridPastTheEnd(), vad_onset=0.5, vad_offset=0.3,
                                 chunk_size=28, vad_pad_onset=0.25, vad_pad_offset=0.2,
                                 **kwargs)
        return processor.process(audio)

    def test_the_grid_does_end_past_the_audio(self):
        scores = self._GridPastTheEnd()(None)
        self.assertGreater(scores.sliding_window[len(scores.data) - 1].end, self.DURATION)

    def test_speech_to_the_end_stops_at_the_last_sample(self):
        segments = self._process()
        self.assertEqual(len(segments), 1)
        seg = segments[0]
        self.assertAlmostEqual(seg["end"], self.DURATION)
        self.assertAlmostEqual(len(seg["audio"]) / self.SR, seg["end"] - seg["start"],
                               delta=1 / self.SR)

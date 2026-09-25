"""Tests for the Silero VAD score-curve backend (pipeline/vads/silero.py).

Synthetic audio only: no fixtures, nothing from outside this repository."""

import math
import os
import unittest

import numpy as np

from cantocaptions_ai.pipeline.vads import CurveVad, Silero
from cantocaptions_ai.pipeline.vads.silero import FRAME_SAMPLES, MODEL_PATH

SR = 16000


def tone_bursts(seconds=6.0, bursts=((1.0, 2.5), (4.0, 5.0)), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.001, int(seconds * SR)).astype(np.float32)
    t = np.arange(len(x)) / SR
    for a, b in bursts:
        m = (t >= a) & (t < b)
        # A crude voiced sound: a 140 Hz harmonic stack, amplitude-modulated at 5 Hz.
        x[m] += (0.2 * (1 + np.sin(2 * np.pi * 5 * t[m]))
                 * sum(np.sin(2 * np.pi * 140 * k * t[m]) / k for k in range(1, 8)))
    return x


class TestSilero(unittest.TestCase):
    def test_model_is_vendored(self):
        self.assertTrue(os.path.isfile(MODEL_PATH))

    def test_curve_grid_and_range(self):
        audio = tone_bursts()
        vad = Silero(vad_onset=0.5)
        curve = vad({"waveform": vad.preprocess_audio(audio), "sample_rate": SR})
        self.assertEqual(curve.data.shape, (math.ceil(len(audio) / FRAME_SAMPLES), 1))
        self.assertAlmostEqual(curve.sliding_window.step, FRAME_SAMPLES / SR)
        self.assertTrue(np.all((curve.data >= 0) & (curve.data <= 1)))

    def test_thread_count_is_restored(self):
        import torch
        before = torch.get_num_threads()
        vad = Silero(vad_onset=0.5)
        vad({"waveform": vad.preprocess_audio(tone_bursts(1.0)), "sample_rate": SR})
        self.assertEqual(torch.get_num_threads(), before)

    def test_state_is_reset_between_files(self):
        # The model is recurrent; scoring the same audio twice must not depend on the call
        # before it.
        vad = Silero(vad_onset=0.5)
        wav = vad.preprocess_audio(tone_bursts())
        first = vad({"waveform": wav, "sample_rate": SR}).data
        vad({"waveform": vad.preprocess_audio(tone_bursts(seed=1)), "sample_rate": SR})
        again = vad({"waveform": wav, "sample_rate": SR}).data
        np.testing.assert_array_equal(first, again)

    def test_feeds_the_shared_chunker(self):
        vad = Silero(vad_onset=0.5)
        curve = vad({"waveform": vad.preprocess_audio(tone_bursts(30.0)), "sample_rate": SR})
        chunks = CurveVad.cover_chunks(curve, 10, 30.0)
        self.assertAlmostEqual(chunks[0]["start"], 0.0)
        self.assertAlmostEqual(chunks[-1]["end"], 30.0)


if __name__ == "__main__":
    unittest.main()

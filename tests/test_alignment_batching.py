"""Tests for _compute_vad_emissions_batched's indexing/slicing logic.

Uses a fake model and fake bert_processor (no real weights, no network) so this
stays fast and CI-friendly, mirroring how tests/test_reference_correction.py
mocks out heavy dependencies. Real-model numerical correctness is instead
validated by scripts/bench_alignment_batching.py, which runs the real
alignment model and asserts batched output matches the sequential path.
"""

import unittest

import numpy as np
import torch

from cantocaptions_ai.pipeline.alignment import _compute_vad_emissions_batched


def _make_segments(lengths):
    """VAD-segment-like dicts whose 'audio' length stands in for frame count."""
    segments = []
    t = 0.0
    for length in lengths:
        segments.append({
            "start": t,
            "end": t + length,
            "audio": np.zeros(length, dtype=np.float32),
        })
        t += length + 1
    return segments


def _fake_bert_processor(wavs, **kwargs):
    """Pads each wav to the batch max on a 1:1 sample-to-'frame' basis."""
    lens = [len(w) for w in wavs]
    max_len = max(lens) if lens else 0
    batch = len(wavs)
    input_features = torch.zeros(batch, max_len, 1)
    attention_mask = torch.zeros(batch, max_len, dtype=torch.long)
    for row, w in enumerate(wavs):
        length = len(w)
        input_features[row, :length, 0] = torch.from_numpy(np.asarray(w, dtype=np.float32))
        attention_mask[row, :length] = 1
    return {"input_features": input_features, "attention_mask": attention_mask}


class _FakeModel:
    """Identity passthrough: logits shape mirrors input_features shape."""

    def __init__(self, fail_first_call=False):
        self.calls = 0
        self._fail_first_call = fail_first_call
        self._has_failed = False

    def __call__(self, input_features, attention_mask=None):
        self.calls += 1
        if self._fail_first_call and not self._has_failed:
            self._has_failed = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return _FakeOutput(logits=input_features)

    def parameters(self):
        # _compute_vad_emissions_batched reads the model's dtype off its
        # parameters to cast input_features to match.
        yield torch.zeros(1, dtype=torch.float32)


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class TestComputeVadEmissionsBatched(unittest.TestCase):
    def test_order_and_trimming_preserved_across_batches(self):
        lengths = [3, 7, 2, 10, 5]
        segments = _make_segments(lengths)
        model = _FakeModel()

        results = _compute_vad_emissions_batched(segments, model, _fake_bert_processor, "cpu", batch_size=2)

        self.assertEqual(len(results), len(lengths))
        for i, expected_len in enumerate(lengths):
            emission, _ = results[i]
            self.assertEqual(
                emission.shape[0], expected_len,
                f"segment {i}: expected {expected_len} frames (unpadded), got {emission.shape[0]}",
            )

    def test_batch_size_larger_than_segment_count(self):
        lengths = [4, 9]
        segments = _make_segments(lengths)
        model = _FakeModel()

        results = _compute_vad_emissions_batched(segments, model, _fake_bert_processor, "cpu", batch_size=8)

        for i, expected_len in enumerate(lengths):
            self.assertEqual(results[i][0].shape[0], expected_len)

    def test_empty_input_returns_empty_list(self):
        model = _FakeModel()
        results = _compute_vad_emissions_batched([], model, _fake_bert_processor, "cpu", batch_size=4)
        self.assertEqual(results, [])

    def test_oom_on_first_batch_halves_and_recovers(self):
        lengths = [3, 7, 2, 10, 5]
        segments = _make_segments(lengths)
        model = _FakeModel(fail_first_call=True)

        results = _compute_vad_emissions_batched(segments, model, _fake_bert_processor, "cpu", batch_size=4)

        self.assertGreater(model.calls, 1, "expected at least one retry after the simulated OOM")
        for i, expected_len in enumerate(lengths):
            emission, _ = results[i]
            self.assertEqual(emission.shape[0], expected_len)


if __name__ == "__main__":
    unittest.main()

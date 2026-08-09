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
    """Stands in for Wav2Vec2BertForCTC.

    With ``adapter_stride=1`` it is an identity passthrough (a model with no adapter, where
    input-feature frames and emission frames are the same unit). With ``adapter_stride=2`` it
    halves the frame dimension the way alvanlii/wav2vec2-BERT-cantonese's adapter does, so
    ``attention_mask.sum()`` is *not* a valid emission length — the condition that caused
    batch padding to leak into the emission and compress timestamps.
    """

    def __init__(self, fail_first_call=False, adapter_stride=1):
        self.calls = 0
        self.batch_max_frames = []
        self.adapter_stride = adapter_stride
        self._fail_first_call = fail_first_call
        self._has_failed = False

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        if self.adapter_stride == 1:
            return input_lengths
        # Mirrors transformers' conv arithmetic for kernel=3, stride=2, padding=1.
        return torch.div(input_lengths + 2 * 1 - 3, 2, rounding_mode="floor") + 1

    def __call__(self, input_features, attention_mask=None):
        self.calls += 1
        self.batch_max_frames.append(input_features.shape[1])
        if self._fail_first_call and not self._has_failed:
            self._has_failed = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self.adapter_stride == 1:
            return _FakeOutput(logits=input_features)
        out_frames = int(self._get_feat_extract_output_lengths(
            torch.tensor(input_features.shape[1])
        ).item())
        return _FakeOutput(logits=input_features[:, :out_frames, :])

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

    def test_batches_processed_longest_first(self):
        # Guards against the ascending-sort bug: processing shortest-first meant
        # every batch that needed a new largest-yet shape forced a fresh, ever-larger
        # CUDA allocation (old smaller cached blocks can't be reused for it), so
        # reserved VRAM climbed monotonically over the stage instead of being bounded
        # by the single largest batch allocated up front.
        lengths = [3, 20, 7, 15, 2, 10, 5]
        segments = _make_segments(lengths)
        model = _FakeModel()

        _compute_vad_emissions_batched(segments, model, _fake_bert_processor, "cpu", batch_size=1)

        self.assertEqual(
            model.batch_max_frames, sorted(model.batch_max_frames, reverse=True),
            "batches should be processed in non-increasing max-frame order (longest first)",
        )
        self.assertEqual(model.batch_max_frames[0], max(lengths))

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


class TestAdapterFrameRateConversion(unittest.TestCase):
    """The align model's adapter halves the frame rate, so attention_mask.sum() (input
    feature frames) is the wrong unit to trim the emission with. Trimming with it is a
    silent no-op that leaves batch padding on every non-longest row, which shows up
    downstream as timestamps compressed toward each segment's start.
    """

    def test_emission_trimmed_to_adapter_output_length(self):
        lengths = [40, 20, 36, 8]
        segments = _make_segments(lengths)
        model = _FakeModel(adapter_stride=2)

        results = _compute_vad_emissions_batched(
            segments, model, _fake_bert_processor, "cpu", batch_size=2,
        )

        for i, in_len in enumerate(lengths):
            expected = (in_len + 1) // 2  # ceil(L/2) for kernel=3, stride=2, padding=1
            self.assertEqual(
                results[i][0].shape[0], expected,
                f"segment {i}: expected {expected} emission frames, got {results[i][0].shape[0]}",
            )

    def test_frame_rate_is_constant_across_segment_lengths(self):
        # The real symptom: a short segment sharing a batch with a long one inherited the
        # long one's frame count, inflating its frame_rate and compressing its timestamps.
        lengths = [40, 20, 36, 8]
        segments = _make_segments(lengths)
        model = _FakeModel(adapter_stride=2)

        results = _compute_vad_emissions_batched(
            segments, model, _fake_bert_processor, "cpu", batch_size=2,
        )

        rates = [r[1] for r in results]
        for rate in rates:
            self.assertAlmostEqual(
                rate, rates[0], places=1,
                msg=f"frame rates should not vary with segment length, got {rates}",
            )

    def test_no_adapter_model_is_unaffected(self):
        # _get_feat_extract_output_lengths is the identity without an adapter, so a future
        # align model without one keeps the 1:1 mapping.
        lengths = [10, 4]
        segments = _make_segments(lengths)
        model = _FakeModel(adapter_stride=1)

        results = _compute_vad_emissions_batched(
            segments, model, _fake_bert_processor, "cpu", batch_size=2,
        )

        for i, expected_len in enumerate(lengths):
            self.assertEqual(results[i][0].shape[0], expected_len)


if __name__ == "__main__":
    unittest.main()

"""Tests for align-model audio priming.

Two halves, matching the two ways priming can go wrong:

- ``TestTailPrimer`` — the primer produces ``prefix + audio`` with the original bytes intact.
- ``TestPrimedEmissionSlicing`` — ``_compute_vad_emissions_batched`` gives back exactly the
  segment's own frames. This is the half that matters: the primer's whole purpose is to
  absorb an artifact at frame 0, so a slice that is one frame generous hands that artifact
  straight back, silently.

Reuses the fake model from test_alignment_batching so this stays fast and needs no
weights or network.
"""

import unittest

import numpy as np
import torch

from cantocaptions_ai.pipeline.align_profiles import (
    ALIGN_PROFILES,
    DEFAULT_ALIGN_PROFILE,
    AlignProfile,
    TailPrimer,
    get_align_profile,
)
from cantocaptions_ai.pipeline.alignment import _compute_vad_emissions_batched
from tests.test_alignment_batching import _FakeModel, _make_segments

SR = 16000


class TestTailPrimer(unittest.TestCase):
    def test_prepends_tail_and_leaves_audio_untouched(self):
        audio = np.arange(SR * 3, dtype=np.float32)
        out = TailPrimer(seconds=1.0, reverse=False)(audio, SR)
        self.assertEqual(len(out), len(audio) + SR)
        np.testing.assert_array_equal(out[SR:], audio)
        np.testing.assert_array_equal(out[:SR], audio[-SR:])

    def test_reverse_flips_only_the_prefix(self):
        audio = np.arange(SR * 3, dtype=np.float32)
        out = TailPrimer(seconds=1.0, reverse=True)(audio, SR)
        np.testing.assert_array_equal(out[SR:], audio)
        np.testing.assert_array_equal(out[:SR], audio[-SR:][::-1])

    def test_prefix_capped_at_segment_length(self):
        """A segment shorter than the primer contributes only what it has."""
        audio = np.arange(SR // 2, dtype=np.float32)
        out = TailPrimer(seconds=1.0)(audio, SR)
        self.assertEqual(len(out), 2 * len(audio))

    def test_empty_audio_is_returned_unchanged(self):
        audio = np.zeros(0, dtype=np.float32)
        np.testing.assert_array_equal(TailPrimer()(audio, SR), audio)

    def test_output_is_a_real_contiguous_array(self):
        """Reversal must not leave a negative-strided view for the feature extractor."""
        out = TailPrimer(seconds=1.0, reverse=True)(np.arange(SR * 2, dtype=np.float32), SR)
        self.assertTrue(out.flags["C_CONTIGUOUS"])


class TestAlignProfileRegistry(unittest.TestCase):
    def test_current_align_model_primes(self):
        profile = get_align_profile("alvanlii/wav2vec2-BERT-cantonese")
        self.assertIsInstance(profile.primer, TailPrimer)

    def test_unknown_model_is_a_no_op(self):
        """A model added later must run unprimed rather than inherit someone else's fix."""
        for name in ("some/other-align-model", None, ""):
            with self.subTest(name=name):
                self.assertIs(get_align_profile(name), DEFAULT_ALIGN_PROFILE)
                self.assertIsNone(get_align_profile(name).primer)

    def test_every_registered_primer_is_callable(self):
        for name, profile in ALIGN_PROFILES.items():
            with self.subTest(model=name):
                self.assertTrue(profile.primer is None or callable(profile.primer))


def _two_channel_bert_processor(wavs, **kwargs):
    """Like test_alignment_batching's fake processor, but with a 2-wide feature dim.

    The shared one is 1-wide, which makes ``log_softmax`` over the last dim collapse to
    zeros — fine for asserting *shapes*, useless for asserting *which* frames came back.
    With two channels the sample value survives as ``emission[:, 0] - emission[:, 1]``.
    """
    lens = [len(w) for w in wavs]
    max_len = max(lens) if lens else 0
    input_features = torch.zeros(len(wavs), max_len, 2)
    attention_mask = torch.zeros(len(wavs), max_len, dtype=torch.long)
    for row, w in enumerate(wavs):
        input_features[row, :len(w), 0] = torch.from_numpy(np.asarray(w, dtype=np.float32))
        attention_mask[row, :len(w)] = 1
    return {"input_features": input_features, "attention_mask": attention_mask}


def _recover(emission):
    """Undo log_softmax to get back the sample values the processor encoded."""
    return (emission[:, 0] - emission[:, 1]).numpy()


class TestPrimedEmissionSlicing(unittest.TestCase):
    """The fake processor maps 1 sample -> 1 feature frame, so with adapter_stride=1 a
    segment of N samples is worth exactly N emission frames. That makes the expected slice
    exact and independent of any real conv arithmetic."""

    def _run(self, lengths, primer, adapter_stride=1, batch_size=2, processor=None):
        segments = _make_segments(lengths)
        # Distinct values per segment so a mis-slice shows up as wrong content, not just
        # wrong length: the primer prepends a *copy* of the segment's own tail, which a
        # length-only assertion would happily accept in the wrong place.
        for n, seg in enumerate(segments):
            seg["audio"] = np.arange(1, len(seg["audio"]) + 1, dtype=np.float32) + 100 * n
        model = _FakeModel(adapter_stride=adapter_stride)
        results = _compute_vad_emissions_batched(
            segments, model, processor or _two_channel_bert_processor, "cpu",
            batch_size=batch_size, primer=primer,
        )
        return segments, results

    def test_primed_emission_matches_unprimed_content(self):
        lengths = [3, 7, 2, 10, 5]
        primer = TailPrimer(seconds=2 / SR, reverse=False)  # 2 samples of prefix
        segments, results = self._run(lengths, primer)
        for seg, (emission, _) in zip(segments, results):
            np.testing.assert_allclose(
                _recover(emission), seg["audio"], rtol=1e-5,
                err_msg="primed emission must contain the segment's own frames, not the primer's",
            )

    def test_a_one_frame_generous_slice_would_be_caught(self):
        """Guards the guard: the assertion above must actually notice an off-by-one.

        The primer prepends a copy of the tail, so an emission that kept one primer frame
        is still plausible-looking audio of almost the right length — exactly the silent
        failure this test exists to prevent.
        """
        segments, results = self._run([6], TailPrimer(seconds=2 / SR, reverse=False))
        emission = results[0][0]
        off_by_one = torch.cat([emission[:1], emission])  # what a generous slice yields
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(_recover(off_by_one), segments[0]["audio"], rtol=1e-5)

    def test_primer_none_is_identical_to_before(self):
        lengths = [3, 7, 2, 10, 5]
        _, primed = self._run(lengths, TailPrimer(seconds=2 / SR, reverse=False))
        _, plain = self._run(lengths, None)
        for (a, _), (b, _) in zip(primed, plain):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5)

    def test_frame_rate_reported_for_the_segment_not_the_primed_audio(self):
        """frame_rate feeds _get_emission_for_segment's index math; counting primer frames
        into it would rescale every timestamp in the segment."""
        _, unprimed = self._run([10, 10], None)
        _, primed = self._run([10, 10], TailPrimer(seconds=4 / SR, reverse=False))
        for (_, a), (_, b) in zip(unprimed, primed):
            self.assertAlmostEqual(a, b)

    def test_works_with_an_adapter_that_halves_the_frame_rate(self):
        lengths = [8, 16, 4]
        model_stride = 2
        segments, results = self._run(
            lengths, TailPrimer(seconds=4 / SR, reverse=False), adapter_stride=model_stride,
        )
        model = _FakeModel(adapter_stride=model_stride)
        for seg, (emission, _) in zip(segments, results):
            expected = int(model._get_feat_extract_output_lengths(
                torch.tensor([len(seg["audio"])])
            )[0])
            self.assertEqual(emission.size(0), expected)

    def test_long_primer_on_short_segment(self):
        """The primer caps itself at the segment length; slicing must still be exact."""
        segments, results = self._run([4, 4], TailPrimer(seconds=10.0, reverse=False))
        for seg, (emission, _) in zip(segments, results):
            np.testing.assert_allclose(_recover(emission), seg["audio"], rtol=1e-5)

    def test_bad_primer_that_drops_audio_is_rejected(self):
        """A primer returning less than the original audio would silently mis-slice."""
        class _TruncatingPrimer:
            def __call__(self, audio, sample_rate):
                return audio[:-2]

        with self.assertRaises(RuntimeError) as ctx:
            self._run([8, 8], _TruncatingPrimer())
        self.assertIn("prefix followed by the original audio", str(ctx.exception))


class TestAlignProfileDefaults(unittest.TestCase):
    def test_profile_fields_default_to_no_ops(self):
        self.assertIsNone(AlignProfile().primer)


if __name__ == "__main__":
    unittest.main()

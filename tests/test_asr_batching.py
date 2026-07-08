"""Tests for the native ASR backend's VRAM handling and OOM-recovery wiring.

Two things are covered, both CPU-only with fakes (no GPU/checkpoint/network),
mirroring tests/test_alignment_batching.py's _FakeModel(fail_first_call=True)
pattern:

1. compute_cuda_memory_fraction's pure arithmetic (the formula behind
   cap_cuda_memory, which forces a catchable OOM before the driver pages GPU
   memory into host RAM).
2. That a CUDA-OOM raised inside the ASR generate() call is caught and the
   batch halved end-to-end via run_adaptive_batches — i.e. the memory cap's
   whole point (turn a near-OOM into graceful halving) actually works through
   the ASR path.

cap_cuda_memory itself early-returns off CUDA (vram_stats returns None), so it
isn't directly unit-testable without a GPU; its logic lives entirely in the
pure helper tested here, matching how sibling tests defer real-model checks to
scripts/bench_*.
"""

import unittest

import numpy as np
import torch

from cantocaptions_ai.utils.model_utils import compute_cuda_memory_fraction
from cantocaptions_ai.pipeline._asr_native import QwenPipelineNative


class TestComputeCudaMemoryFraction(unittest.TestCase):
    def test_matches_reported_scenario(self):
        # From the reported run: reserved≈5399, free≈5338, total=10737, headroom=512
        # -> cap 10225 / 10737 ≈ 0.952.
        frac = compute_cuda_memory_fraction(5399, 5338, 10737, 512)
        self.assertAlmostEqual(frac, 10225 / 10737, places=4)

    def test_clamped_to_one_when_cap_exceeds_total(self):
        # Almost nothing reserved by others -> cap would exceed 1.0, clamp to 1.0.
        self.assertEqual(compute_cuda_memory_fraction(0, 10000, 10000, 0), 1.0)

    def test_degenerate_total_returns_one(self):
        self.assertEqual(compute_cuda_memory_fraction(0, 0, 0, 512), 1.0)

    def test_floor_clamped_to_minimum(self):
        # Tiny cap (headroom nearly the whole card) floors at 0.05 rather than 0/negative.
        self.assertEqual(compute_cuda_memory_fraction(0, 100, 10000, 9999), 0.05)

    def test_less_free_lowers_fraction(self):
        # Another app taking VRAM (lower free) tightens the cap.
        more_free = compute_cuda_memory_fraction(1000, 5000, 10000, 512)
        less_free = compute_cuda_memory_fraction(1000, 2000, 10000, 512)
        self.assertLess(less_free, more_free)


class _FakeInputs(dict):
    """Mapping so ``**inputs`` unpacks into generate(), plus a no-op ``.to()``."""

    def to(self, *args, **kwargs):
        return self


class _FakeProcessor:
    """Identity-carrying fake: each segment's marker (encoded in its audio[0]) is
    threaded input_ids -> generate() output -> decode() so a test can assert each
    decoded text landed on the right segment after cross-file batching/reorder.
    """

    N_INPUT = 4

    def apply_transcription_request(self, audio, language):
        markers = [int(round(float(w[0]))) for w in audio]
        ids = torch.tensor([[m] * self.N_INPUT for m in markers], dtype=torch.long)
        return _FakeInputs(input_ids=ids)

    def decode(self, sequences, **kwargs):
        # _infer_batch passes sequences[:, n_input:]; column 0 carries the marker.
        return [str(int(sequences[k, 0])) for k in range(sequences.shape[0])]


class _FakeModel:
    """Fake ASR model that echoes each row's marker into the generated tail so it
    survives to decode(). With fail_first_call=True its generate() also raises a
    CUDA OOM on the first call (mirroring the message run_adaptive_batches filters
    on) then succeeds, so the ASR path must catch it and retry at a halved batch."""

    def __init__(self, fail_first_call=True):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.calls = 0
        self._fail_first_call = fail_first_call
        self._has_failed = False

    def generate(self, **kwargs):
        self.calls += 1
        if self._fail_first_call and not self._has_failed:
            self._has_failed = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        input_ids = kwargs["input_ids"]
        marker = input_ids[:, :1]  # (batch, 1) — each row's marker
        tail = marker.repeat(1, 2)  # appended cols; decode reads column 0
        return torch.cat([input_ids, tail], dim=1)  # (batch, n_input + 2)


def _make_segments(n, audio_len=128):
    segs = []
    t = 0.0
    for _ in range(n):
        segs.append({"start": t, "end": t + 1.0, "audio": np.zeros(audio_len, dtype=np.float32)})
        t += 1.01
    return segs


def _make_marked_segments(markers, audio_len=128):
    """Segments whose identity marker is encoded in audio[0] (see _FakeProcessor)."""
    segs = []
    t = 0.0
    for m in markers:
        audio = np.zeros(audio_len, dtype=np.float32)
        audio[0] = float(m)
        segs.append({"start": t, "end": t + 1.0, "audio": audio})
        t += 1.01
    return segs


class _RecordingModel(_FakeModel):
    """Fake model that records the marker of every segment it processes, in order,
    so a test can assert the BatchExecutor's longest-first processing order."""

    def __init__(self):
        super().__init__(fail_first_call=False)
        self.seen_markers = []

    def generate(self, **kwargs):
        self.seen_markers.extend(int(x) for x in kwargs["input_ids"][:, 0])
        return super().generate(**kwargs)


class TestAsrOomHalving(unittest.TestCase):
    def test_generate_oom_is_caught_and_batch_halved(self):
        model = _FakeModel(fail_first_call=True)
        pipe = QwenPipelineNative(
            model=model, processor=_FakeProcessor(), device="cpu",
            language="yue", batch_size=4,
        )
        segments = _make_segments(4)

        result = pipe.process(segments)

        self.assertGreater(model.calls, 1, "expected a retry after the simulated OOM")
        self.assertEqual(len(result["segments"]), 4)
        for seg in result["segments"]:
            self.assertIn("text", seg)


class TestAsrScatterAndOrder(unittest.TestCase):
    """Lock in that batching maps each decoded text back to its originating
    segment. These guard the correctness a future reorder/executor refactor could
    break; they assert no particular batch *order* (none is added this round)."""

    def test_run_scatters_texts_to_correct_items(self):
        # Files with differing segment counts; every segment's marker is unique
        # (file_idx*1000 + seg_idx) so a mis-scatter across the cross-file batches
        # would surface as a wrong text.
        pipe = QwenPipelineNative(
            model=_FakeModel(fail_first_call=False), processor=_FakeProcessor(),
            device="cpu", language="yue", batch_size=3,
        )
        counts = [4, 1, 5, 2]  # 12 segments over 4 files; batch_size=3 spans file boundaries
        items = []
        for fidx, count in enumerate(counts):
            markers = [fidx * 1000 + sdx for sdx in range(count)]
            items.append({"audio_path": f"f{fidx}.wav", "vad_segments": _make_marked_segments(markers)})

        result_items = pipe.run(items)

        self.assertEqual(len(result_items), len(items))
        for fidx, item in enumerate(result_items):
            segs = item["result"]["segments"]
            self.assertEqual(len(segs), counts[fidx])
            for sdx, seg in enumerate(segs):
                self.assertEqual(seg["text"], str(fidx * 1000 + sdx))

    def test_process_preserves_input_order(self):
        # Markers deliberately out of positional order: identity != index, so a
        # reorder that forgot to restore original order would be caught.
        pipe = QwenPipelineNative(
            model=_FakeModel(fail_first_call=False), processor=_FakeProcessor(),
            device="cpu", language="yue", batch_size=2,
        )
        markers = [50, 10, 30, 20, 40]
        result = pipe.process(_make_marked_segments(markers))

        texts = [seg["text"] for seg in result["segments"]]
        self.assertEqual(texts, [str(m) for m in markers])

    def test_process_processes_longest_segment_first(self):
        # BatchExecutor sorts jobs longest-audio-first; with batch_size=1 the model
        # sees exactly the processing order. Audio length grows with the marker, so
        # longest-first == descending marker — while OUTPUT stays in input order.
        model = _RecordingModel()
        pipe = QwenPipelineNative(
            model=model, processor=_FakeProcessor(),
            device="cpu", language="yue", batch_size=1,
        )
        markers = [2, 0, 3, 1]
        segments = []
        t = 0.0
        for m in markers:
            audio = np.zeros(100 + m * 50, dtype=np.float32)  # longer audio for larger marker
            audio[0] = float(m)
            segments.append({"start": t, "end": t + 1.0, "audio": audio})
            t += 1.01

        result = pipe.process(segments)

        self.assertEqual(model.seen_markers, [3, 2, 1, 0], "expected longest-first processing")
        self.assertEqual([seg["text"] for seg in result["segments"]], [str(m) for m in markers])


if __name__ == "__main__":
    unittest.main()

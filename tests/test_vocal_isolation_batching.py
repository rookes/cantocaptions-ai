"""Tests for MbRoformerProcessor.run()'s memory scheduling and batching logic.

Uses a fake model (identity passthrough, no real weights/network) and synthetic
VAD segments, mirroring how tests/test_alignment_batching.py mocks out heavy
dependencies for _compute_vad_emissions_batched. Real-model numerical
correctness is validated separately by scripts/bench_vocal_isolation_batching.py.

The concurrency tests guard against a bug where MbRoformerProcessor.run() built
overlap-add scratch buffers (mixture/result/counter) for every segment of every
file before running any inference, so peak host RAM scaled with total dataset
duration instead of being bounded per file/window.
"""

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from cantocaptions_ai.pipeline.vocal_isolation import MbRoformerProcessor


class _FakeMbModel:
    """Identity passthrough: (B, 2, C) in -> (B, 2, C) out, matching the shape
    infer_fn expects from the real single-stem vocals model."""

    def __init__(self):
        self.calls = 0

    def eval(self):
        pass

    def __call__(self, batch_t):
        self.calls += 1
        return batch_t


class _CountingMbRoformerProcessor(MbRoformerProcessor):
    """Tracks how many segment-state buffer sets (mixture/result/counter) are
    concurrently alive during run() — the quantity that distinguishes the
    unbounded-memory bug (peak == total segment count across the whole run)
    from a properly windowed fix (peak bounded independent of file count)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.alive = 0
        self.peak_alive = 0

    def _prepare_mixture(self, audio):
        result = super()._prepare_mixture(audio)
        self.alive += 1
        self.peak_alive = max(self.peak_alive, self.alive)
        return result

    def _finalize_segment(self, key, st, estimated, item_out):
        super()._finalize_segment(key, st, estimated, item_out)
        self.alive -= 1


def _make_items(n_files, segs_per_file, audio_len=256, sample_rate=16000):
    """Synthetic 'files', each with segs_per_file VAD segments of audio_len samples."""
    items = []
    for f in range(n_files):
        segs = []
        t = 0.0
        for _ in range(segs_per_file):
            segs.append({
                "start": t,
                "end": t + audio_len / sample_rate,
                "audio": np.zeros(audio_len, dtype=np.float32),
            })
            t += audio_len / sample_rate + 0.01
        items.append({"audio_path": f"file_{f}.wav", "vad_segments": segs})
    return items


def _no_resample_config(chunk_size=64, num_overlap=2):
    """model.sample_rate == SAMPLE_RATE (16000) so _prepare_mixture skips resampling
    entirely — keeps the memory/ordering tests fast and focused on run()'s scheduling."""
    return OmegaConf.create({
        "model": {"sample_rate": 16000},
        "inference": {"chunk_size": chunk_size, "num_overlap": num_overlap},
    })


class TestMbRoformerRunMemoryBound(unittest.TestCase):
    def test_peak_concurrent_segments_bounded_across_many_files(self):
        # 80 files x 4 segments = 320 total segments. An unbounded/unwindowed
        # run() would hold all 320 segments' state alive at once; a fix that
        # windows per file should bound peak_alive to one file's segment count
        # (4), independent of how many files are processed.
        proc = _CountingMbRoformerProcessor(
            model=_FakeMbModel(), config=_no_resample_config(),
            device=torch.device("cpu"), batch_size=3,
        )
        items = _make_items(n_files=80, segs_per_file=4)
        proc.run(items, debug_dir=None, load_debug_dir=None)
        self.assertLess(proc.peak_alive, 80 * 4)
        self.assertLessEqual(proc.peak_alive, 4)

    def test_defensive_cap_bounds_single_pathological_file(self):
        from cantocaptions_ai.pipeline.vocal_isolation import _MAX_SEGMENTS_PER_WINDOW
        proc = _CountingMbRoformerProcessor(
            model=_FakeMbModel(), config=_no_resample_config(),
            device=torch.device("cpu"), batch_size=3,
        )
        items = _make_items(n_files=1, segs_per_file=300)
        proc.run(items, debug_dir=None, load_debug_dir=None)
        self.assertLess(proc.peak_alive, 300)
        self.assertLessEqual(proc.peak_alive, _MAX_SEGMENTS_PER_WINDOW)


class TestMbRoformerRunCorrectness(unittest.TestCase):
    def test_order_and_duration_preserved_across_files_and_windows(self):
        proc = MbRoformerProcessor(
            model=_FakeMbModel(), config=_no_resample_config(),
            device=torch.device("cpu"), batch_size=3,
        )
        segs_per_file = 4
        items = _make_items(n_files=5, segs_per_file=segs_per_file)
        out = proc.run(items, debug_dir=None, load_debug_dir=None)

        self.assertEqual(len(out), 5)
        for item in out:
            self.assertEqual(len(item["vad_segments"]), segs_per_file)
            for seg in item["vad_segments"]:
                self.assertEqual(len(seg["audio"]), 256)

    def test_handles_sample_rate_mismatch_end_to_end(self):
        # Exercises the real torchaudio resample + reflect-border-pad +
        # resample-back path (not skipped, unlike the tests above) to guard
        # the windowing restructure against breaking _prepare_mixture/
        # _finalize_segment's numerics.
        config = OmegaConf.create({
            "model": {"sample_rate": 44100},
            "inference": {"chunk_size": 128, "num_overlap": 2},
        })
        proc = MbRoformerProcessor(
            model=_FakeMbModel(), config=config,
            device=torch.device("cpu"), batch_size=4,
        )
        items = _make_items(n_files=3, segs_per_file=3, audio_len=1600)
        out = proc.run(items, debug_dir=None, load_debug_dir=None)

        for item in out:
            for seg in item["vad_segments"]:
                self.assertEqual(len(seg["audio"]), 1600)


class TestWholeSegmentMode(unittest.TestCase):
    """`whole` mode: one forward pass per segment at its natural length.

    chunk_size belongs to the inference harness, not the model, so a segment can
    be separated in a single pass -- dropping the 2x overlap redundancy and the
    chunk seams. These guard the two properties the dataset build depends on:
    exact output length, and the long-segment fallback that bounds peak VRAM.
    """

    @staticmethod
    def _config(whole_max_s=35.0, border_s=1.0, sample_rate=16000):
        return OmegaConf.create({
            "model": {"sample_rate": sample_rate},
            "inference": {
                "chunk_size": 128, "num_overlap": 2,
                "mode": "whole", "border_s": border_s, "whole_max_s": whole_max_s,
            },
        })

    def test_one_forward_pass_per_segment(self):
        model = _FakeMbModel()
        proc = MbRoformerProcessor(
            model=model, config=self._config(),
            device=torch.device("cpu"), batch_size=4,
        )
        items = _make_items(n_files=3, segs_per_file=4, audio_len=1600)
        out = proc.run(items, debug_dir=None, load_debug_dir=None)

        # 12 segments, all under whole_max_s -> exactly 12 calls. The chunked path
        # would slide a 128-sample window over each and call many more times.
        self.assertEqual(model.calls, 12)
        for item in out:
            for seg in item["vad_segments"]:
                self.assertEqual(len(seg["audio"]), 1600)

    def test_output_length_is_exact_across_resample_round_trip(self):
        # 16k -> 44.1k -> 16k is not length-preserving (ratio 2.75625, each leg
        # rounds independently). Segments came back up to ~9 ms short before
        # _finalize_segment pinned them to the source length.
        proc = MbRoformerProcessor(
            model=_FakeMbModel(), config=self._config(sample_rate=44100),
            device=torch.device("cpu"), batch_size=4,
        )
        for audio_len in (1600, 2411, 4099):
            with self.subTest(audio_len=audio_len):
                items = _make_items(n_files=1, segs_per_file=2, audio_len=audio_len)
                out = proc.run(items, debug_dir=None, load_debug_dir=None)
                for seg in out[0]["vad_segments"]:
                    self.assertEqual(len(seg["audio"]), audio_len)

    def test_long_segment_falls_back_to_chunked(self):
        # whole_max_s below the segment length forces the chunked path, which
        # bounds peak VRAM for a segment longer than VAD is supposed to emit.
        model = _FakeMbModel()
        # 1600 samples @ 16 kHz = 0.1 s; cap at 0.01 s so every segment overflows.
        proc = MbRoformerProcessor(
            model=model, config=self._config(whole_max_s=0.01),
            device=torch.device("cpu"), batch_size=4,
        )
        items = _make_items(n_files=1, segs_per_file=2, audio_len=1600)
        out = proc.run(items, debug_dir=None, load_debug_dir=None)

        # Chunked: a 128-sample window over ~1600 samples is many calls, not 2.
        self.assertGreater(model.calls, 2)
        for seg in out[0]["vad_segments"]:
            self.assertEqual(len(seg["audio"]), 1600)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            MbRoformerProcessor(
                model=_FakeMbModel(), config=self._config(),
                device=torch.device("cpu"), segment_mode="sideways",
            )

    def test_segment_mode_argument_overrides_config(self):
        model = _FakeMbModel()
        proc = MbRoformerProcessor(
            model=model, config=self._config(),
            device=torch.device("cpu"), batch_size=4, segment_mode="chunked",
        )
        self.assertEqual(proc._mode, "chunked")
        items = _make_items(n_files=1, segs_per_file=1, audio_len=1600)
        proc.run(items, debug_dir=None, load_debug_dir=None)
        self.assertGreater(model.calls, 1)


if __name__ == "__main__":
    unittest.main()

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
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from cantocaptions_ai.pipeline import vocal_isolation as vi
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


class TestChunkGrid(unittest.TestCase):
    """The chunk grid skips chunks that would hold only border padding."""

    def _proc(self, model=None):
        return MbRoformerProcessor(
            model=model or _FakeMbModel(), config=_no_resample_config(chunk_size=64),
            device=torch.device("cpu"), batch_size=3,
        )

    def test_drops_exactly_the_padding_only_chunk(self):
        proc = self._proc()  # chunk 64, step 32, border 32
        for audio_len in (65, 100, 256, 257):
            with self.subTest(audio_len=audio_len):
                total = audio_len + 64
                grid = list(range(0, total, 32))
                offsets = proc._chunk_offsets(total, padded=True)
                self.assertEqual(offsets, grid[:-1])
                self.assertGreaterEqual(offsets[-1] + 64, total - 32)  # the audio's end is covered

    def test_unpadded_segments_keep_every_chunk(self):
        self.assertEqual(self._proc()._chunk_offsets(60, padded=False), [0, 32])

    def test_output_matches_running_every_grid_chunk(self):
        def run(skip):
            proc = self._proc(_ScaleModel())
            if not skip:
                proc._chunk_offsets = lambda total, padded: list(range(0, total, proc._step))
            return proc.run(TestCompiledChunks._noisy_items(2), debug_dir=None, load_debug_dir=None)
        for a, b in zip(run(False), run(True)):
            for sa, sb in zip(a["vad_segments"], b["vad_segments"]):
                np.testing.assert_array_equal(sa["audio"], sb["audio"])

    def test_progress_estimate_counts_the_chunks_run(self):
        proc = self._proc()
        items = TestCompiledChunks._noisy_items(1)
        estimate = sum(proc._estimate_num_offsets(s) for s in items[0]["vad_segments"])
        seen = []
        proc._separate_chunks = lambda b: (seen.append(b.shape[0]), b)[1]
        proc.run(items, debug_dir=None, load_debug_dir=None)
        self.assertEqual(sum(seen), estimate)


class _ScaleModel:
    """Not an identity, so a chunk's position in the overlap-add actually matters."""

    def eval(self):
        pass

    def __call__(self, batch_t):
        return batch_t * 0.5 + 0.25 * batch_t.flip(-1)


def _file_with_segments(spans_s, seconds=2.0, sr=16000, seed=1):
    """A decoded 'file' and VAD-style segments cut from it as views, with their source."""
    full = np.random.default_rng(seed).standard_normal(int(seconds * sr)).astype(np.float32)
    segs = []
    for start, end in spans_s:
        f1, f2 = int(start * sr), int(end * sr)
        segs.append({"start": start, "end": end, "audio": full[f1:f2], "source": (full, f1)})
    return full, {"audio_path": "f.wav", "vad_segments": segs}


class TestSpans(unittest.TestCase):
    """--vocal_isolation_span_gap: neighbouring segments isolated as one stretch of the file."""

    SEGS = [(0.10, 0.30), (0.35, 0.60), (0.62, 0.70), (1.20, 1.50), (1.55, 1.90)]

    def _proc(self, span_gap, model=None, mode="chunked"):
        return MbRoformerProcessor(
            model=model or _FakeMbModel(), config=_no_resample_config(chunk_size=64),
            device=torch.device("cpu"), batch_size=3, segment_mode=mode, span_gap=span_gap,
        )

    def _run(self, proc, item):
        return proc.run([item], debug_dir=None, load_debug_dir=None)[0]["vad_segments"]

    def test_grouping(self):
        _, item = _file_with_segments(self.SEGS)
        segs = item["vad_segments"]
        group = lambda gap, **kw: [[segs.index(s) for s in sp]
                                   for sp in self._proc(gap, **kw)._spans(segs)]
        self.assertEqual(group(0.1), [[0, 1, 2], [3, 4]])
        self.assertEqual(group(0.6), [[0, 1, 2, 3, 4]])
        self.assertEqual(group(0.0), [[0], [1], [2], [3], [4]])  # off
        self.assertEqual(group(0.6, mode="whole"), [[0], [1], [2], [3], [4]])
        with mock.patch.object(vi, "_SPAN_MAX_S", 1.0):
            self.assertEqual(group(0.6), [[0, 1, 2], [3, 4]])

    def test_each_segment_is_cut_back_at_its_own_samples(self):
        # An identity model hands the audio back, so a wrong offset shows as wrong audio.
        _, item = _file_with_segments(self.SEGS)
        out = self._run(self._proc(0.1), item)
        self.assertEqual(len(out), len(self.SEGS))
        for seg, iso in zip(item["vad_segments"], out):
            self.assertEqual((iso["start"], iso["end"]), (seg["start"], seg["end"]))
            np.testing.assert_allclose(iso["audio"], seg["audio"], atol=1e-6)
            self.assertNotIn("source", iso)

    def test_span_context_is_the_real_audio_between(self):
        # Unlike the identity, a model that mixes in the time-reversed chunk depends on what
        # surrounds each segment, so spans and single segments give different results.
        _, item = _file_with_segments(self.SEGS)
        alone = self._run(self._proc(0.0, _ScaleModel()), item)
        joined = self._run(self._proc(0.1, _ScaleModel()), item)
        self.assertFalse(np.allclose(alone[0]["audio"], joined[0]["audio"]))

    def test_a_replay_decodes_the_file_and_isolates_the_same_audio(self):
        full, item = _file_with_segments(self.SEGS)
        fresh = self._run(self._proc(0.1, _ScaleModel()), item)
        replay_item = {"audio_path": "f.wav", "vad_segments": [
            {k: v for k, v in seg.items() if k != "source"} for seg in item["vad_segments"]]}
        from cantocaptions_ai.pipeline.vad import VadProcessor
        with mock.patch.object(VadProcessor, "_extract", return_value=full) as extract:
            replay = self._run(self._proc(0.1, _ScaleModel()), replay_item)
        extract.assert_called_once()
        for a, b in zip(fresh, replay):
            np.testing.assert_array_equal(a["audio"], b["audio"])

    def test_audio_that_does_not_match_falls_back_to_single_segments(self):
        full, item = _file_with_segments(self.SEGS)
        for seg in item["vad_segments"]:
            seg["source"] = (full * 2, seg["source"][1])
        with self.assertLogs(vi.logger, "WARNING"):
            out = self._run(self._proc(0.1, _ScaleModel()), item)
        alone = self._run(self._proc(0.0, _ScaleModel()), _file_with_segments(self.SEGS)[1])
        for a, b in zip(alone, out):
            np.testing.assert_array_equal(a["audio"], b["audio"])

    def test_persisted_keys_follow_each_segment(self):
        _, item = _file_with_segments(self.SEGS)
        item["vad_segments"][1]["expanded"] = [[0.35, 0.4]]
        out = self._run(self._proc(0.1), item)
        self.assertEqual(out[1]["expanded"], [[0.35, 0.4]])
        self.assertNotIn("expanded", out[0])

    def test_progress_estimate_counts_the_chunks_run(self):
        # Proportioned like the real model (a 4 s step, gaps <= 1 s, segments many steps
        # long): only then does joining segments save chunks rather than add the gaps' own.
        _, item = _file_with_segments([(i * 1.05, i * 1.05 + 1.0) for i in range(5)], seconds=6.0)
        proc = MbRoformerProcessor(
            model=_FakeMbModel(), config=_no_resample_config(chunk_size=4096),
            device=torch.device("cpu"), batch_size=3, span_gap=0.1,
        )
        estimate = sum(
            proc._estimate_chunks(int(round((sp[-1]["end"] - sp[0]["start"]) * 16000)))
            if len(sp) > 1 else proc._estimate_num_offsets(sp[0])
            for sp in proc._spans(item["vad_segments"]))
        seen = []
        proc._separate_chunks = lambda b: (seen.append(b.shape[0]), b)[1]
        self._run(proc, item)
        self.assertEqual(sum(seen), estimate)
        self.assertLess(estimate, sum(proc._estimate_num_offsets(s) for s in item["vad_segments"]))


class _FakeCompiled:
    """Stands in for torch.compile's output: records the batch sizes it is fed, and can
    raise once to act out a failed compile or an OOM."""

    def __init__(self, model, fail=None):
        self.model, self.fail, self.batches = model, fail, []

    def __call__(self, batch_t):
        self.batches.append(batch_t.shape[0])
        if self.fail is not None:
            error, self.fail = self.fail, None
            raise error
        return self.model(batch_t)


class TestCompiledChunks(unittest.TestCase):
    """--vocal_isolation_compile: when the chunked path compiles, and that a compiled run
    sees one batch shape and gives the eager run's output.

    40 chunks per file here: 4 segments of 256 samples, each reflect-padded to 320 and
    cut at a 32-sample step. At batch_size 3 every file ends on a 1-chunk batch.
    """

    def setUp(self):
        vi._COMPILED_SHAPES.clear()
        vi._COMPILE_FAILED = False
        self.compiled = []

    tearDown = setUp

    def _compile(self, fail=None):
        def compile_model(model):
            self.compiled.append(_FakeCompiled(model, fail))
            return self.compiled[-1]
        return mock.patch.object(vi, "_compile_model", side_effect=compile_model)

    @staticmethod
    def _proc(compile, mode="chunked", batch_size=3):
        return MbRoformerProcessor(
            model=_FakeMbModel(), config=_no_resample_config(), device=torch.device("cpu"),
            batch_size=batch_size, segment_mode=mode, compile=compile,
        )

    @staticmethod
    def _noisy_items(n_files):
        rng = np.random.default_rng(0)
        items = _make_items(n_files=n_files, segs_per_file=4)
        for item in items:
            for seg in item["vad_segments"]:
                seg["audio"] = rng.standard_normal(256).astype(np.float32)
        return items

    def test_off_never_compiles(self):
        with self._compile():
            self._proc("off").run(_make_items(2, 4), debug_dir=None, load_debug_dir=None)
        self.assertEqual(self.compiled, [])

    def test_compiled_run_sees_one_batch_size_and_matches_eager(self):
        eager = self._proc("off").run(self._noisy_items(2), debug_dir=None, load_debug_dir=None)
        with self._compile():
            out = self._proc("on").run(self._noisy_items(2), debug_dir=None, load_debug_dir=None)
        self.assertEqual(set(self.compiled[0].batches), {3})  # each file's last batch padded
        for a, b in zip(eager, out):
            for sa, sb in zip(a["vad_segments"], b["vad_segments"]):
                np.testing.assert_array_equal(sa["audio"], sb["audio"])

    def test_auto_compiles_only_a_run_long_enough_to_repay_it(self):
        with self._compile(), mock.patch.object(vi, "_can_compile", return_value=True), \
                mock.patch.object(vi, "_COMPILE_MIN_CHUNKS", 100):
            self._proc("auto").run(_make_items(2, 4), debug_dir=None, load_debug_dir=None)
            self.assertEqual(self.compiled, [])  # 80 chunks
            self._proc("auto").run(_make_items(3, 4), debug_dir=None, load_debug_dir=None)
            self.assertEqual(len(self.compiled), 1)  # 120 chunks
            # Compiled already in this process: free, so even a short run uses it.
            self._proc("auto").run(_make_items(1, 1), debug_dir=None, load_debug_dir=None)
            self.assertEqual(len(self.compiled), 2)

    def test_auto_does_not_compile_without_cuda_and_triton(self):
        with self._compile(), mock.patch.object(vi, "_COMPILE_MIN_CHUNKS", 1):
            self._proc("auto").run(_make_items(3, 4), debug_dir=None, load_debug_dir=None)
        self.assertEqual(self.compiled, [])  # a CPU device

    def test_whole_mode_does_not_compile(self):
        with self._compile():
            self._proc("on", mode="whole").run(_make_items(1, 2), debug_dir=None,
                                              load_debug_dir=None)
        self.assertEqual(self.compiled, [])

    def test_a_failed_compile_falls_back_to_eager_for_the_process(self):
        with self._compile(fail=RuntimeError("triton: no working compiler")):
            with self.assertLogs(vi.logger, "WARNING"):
                out = self._proc("on").run(self._noisy_items(1), debug_dir=None,
                                           load_debug_dir=None)
            self.assertEqual(self.compiled[0].batches, [3])  # tried once, never again
            self.assertTrue(vi._COMPILE_FAILED)
            self._proc("on").run(_make_items(1, 1), debug_dir=None, load_debug_dir=None)
            self.assertEqual(len(self.compiled), 1)
        eager = self._proc("off").run(self._noisy_items(1), debug_dir=None, load_debug_dir=None)
        for sa, sb in zip(eager[0]["vad_segments"], out[0]["vad_segments"]):
            np.testing.assert_array_equal(sa["audio"], sb["audio"])

    def test_padding_follows_an_oom_retry_down(self):
        # Padding a retried half batch back up to full size would run out of memory again.
        with self._compile(fail=torch.cuda.OutOfMemoryError("CUDA out of memory")):
            self._proc("on", batch_size=4).run(_make_items(1, 4), debug_dir=None,
                                               load_debug_dir=None)
        batches = self.compiled[0].batches
        self.assertEqual(batches[0], 4)
        self.assertEqual(set(batches[1:]), {2})


if __name__ == "__main__":
    unittest.main()

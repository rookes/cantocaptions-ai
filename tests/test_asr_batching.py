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


class _LoopingModel:
    """Fake model whose rows with an odd marker never emit EOS: without a repetition
    penalty they run the whole token budget (a repetition loop); with one they stop.
    Rows with an even marker always finish with EOS straight away."""

    EOS = 99

    def __init__(self, loops_under_penalty=False):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.generation_config = type("GC", (), {"eos_token_id": [self.EOS, 98]})()
        self.loops_under_penalty = loops_under_penalty
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        max_new = kwargs["max_new_tokens"]
        penalised = kwargs.get("repetition_penalty") is not None
        rows = []
        for marker in input_ids[:, 0].tolist():
            loops = marker % 2 == 1 and (not penalised or self.loops_under_penalty)
            tail = [marker] * max_new if loops else [marker, self.EOS] + [98] * (max_new - 2)
            rows.append(tail)
        return torch.cat([input_ids, torch.tensor(rows, dtype=torch.long)], dim=1)


class TestAsrLoopRetry(unittest.TestCase):
    """A row that spends the whole budget without EOS is a repetition loop; it is
    re-generated once, alone, with a repetition penalty."""

    def _pipe(self, model, penalty=1.1):
        return QwenPipelineNative(
            model=model, processor=_FakeProcessor(), device="cpu",
            language="yue", batch_size=4, loop_retry_penalty=penalty,
        )

    def test_looping_row_is_retried_with_penalty(self):
        model = _LoopingModel()
        result = self._pipe(model).process(_make_marked_segments([2, 3, 4]))
        segs = result["segments"]
        self.assertEqual([s["text"] for s in segs], ["2", "3", "4"])
        self.assertEqual((segs[1].get("budget_hit"), segs[1].get("retried")), (True, True))
        for s in (segs[0], segs[2]):
            self.assertNotIn("budget_hit", s)
            self.assertNotIn("retried", s)
        self.assertEqual(len(model.calls), 2)
        retry = model.calls[1]
        self.assertEqual(retry["repetition_penalty"], 1.1)
        self.assertEqual(retry["input_ids"][:, 0].tolist(), [3], "only the looping row is retried")

    def test_retry_disabled_still_flags_the_loop(self):
        model = _LoopingModel()
        segs = self._pipe(model, penalty=0).process(_make_marked_segments([3]))["segments"]
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(segs[0]["budget_hit"])
        self.assertNotIn("retried", segs[0])

    def test_no_loop_no_retry(self):
        model = _LoopingModel()
        segs = self._pipe(model).process(_make_marked_segments([2, 4]))["segments"]
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(all("budget_hit" not in s for s in segs))

    def test_retry_that_still_loops_keeps_its_text(self):
        model = _LoopingModel(loops_under_penalty=True)
        segs = self._pipe(model).process(_make_marked_segments([5]))["segments"]
        self.assertEqual(segs[0]["text"], "5")
        self.assertTrue(segs[0]["retried"])


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


class TestLoadModelNativeWithSuppliedParts(unittest.TestCase):
    """A caller that builds its own model and processor (cantocaptions-dataset,
    for a merged LoRA adapter whose directory has no processor files) must get
    them back inside the pipeline untouched, with no hub round-trip."""

    def test_supplied_model_and_processor_skip_download_and_loading(self):
        from unittest import mock

        from cantocaptions_ai.pipeline import _asr_native

        model, processor = _FakeModel(fail_first_call=False), _FakeProcessor()
        with mock.patch.object(_asr_native, "ensure_hf_model_downloaded") as download,                 mock.patch("transformers.AutoProcessor.from_pretrained") as load_processor:
            pipe = _asr_native.load_model_native(
                "runs/some-adapter/merged", device="cpu",
                model=model, processor=processor, batch_size=2,
            )
        download.assert_not_called()
        load_processor.assert_not_called()
        self.assertIs(pipe.model, model)
        self.assertIs(pipe.processor, processor)



class TestNormalizeLanguage(unittest.TestCase):
    def test_code_maps_to_qwen_name(self):
        from cantocaptions_ai.pipeline.asr import _normalize_language
        self.assertEqual(_normalize_language("yue"), "Cantonese")

    def test_missing_language_is_an_error_not_cantonese(self):
        from cantocaptions_ai.pipeline.asr import _normalize_language
        with self.assertRaises(ValueError):
            _normalize_language(None)


if __name__ == "__main__":
    unittest.main()


class _LabellingProcessor(_FakeProcessor):
    """Decodes each marker to a scripted reply: ``language <label><asr_text><raw>`` undecorated,
    and the text after the processor's repetition pass for ``transcription_only``."""

    def __init__(self, replies):
        self.replies = replies  # marker -> (label, raw, folded text)
        self.languages = []

    def apply_transcription_request(self, audio, language):
        self.languages.append(language)
        return super().apply_transcription_request(audio, language)

    def decode(self, sequences, return_format="raw", **kwargs):
        out = []
        for k in range(sequences.shape[0]):
            label, raw, folded = self.replies[int(sequences[k, 0])]
            out.append(folded if return_format == "transcription_only"
                       else f"language {label}<asr_text>{raw}")
        return out


class TestAsrDiagnostics(unittest.TestCase):
    """The ASR stage records the language the model said it heard and the text before the
    processor folded a loop away -- diagnostics for ASR repair, never a change of text."""

    def _segments(self, replies, markers):
        processor = _LabellingProcessor(replies)
        pipe = QwenPipelineNative(model=_FakeModel(fail_first_call=False), processor=processor,
                                  device="cpu", language="yue", batch_size=4)
        return pipe.process(_make_marked_segments(markers))["segments"], pipe, processor

    def test_a_label_other_than_the_requested_language_is_recorded(self):
        segs, _, _ = self._segments({2: ("Cantonese", "你好", "你好"), 4: ("Italian", "Ciao", "Ciao"),
                                     6: ("None", "", "")}, [2, 4, 6])
        self.assertNotIn("asr_language", segs[0])
        self.assertEqual(segs[1]["asr_language"], "Italian")
        self.assertEqual(segs[2]["asr_language"], "none")
        self.assertEqual([s["text"] for s in segs], ["你好", "Ciao", ""])

    def test_a_folded_loop_keeps_its_raw_text(self):
        segs, _, _ = self._segments({2: ("Cantonese", "噔" * 30, "噔"), 4: ("Cantonese", "好", "好")},
                                    [2, 4])
        self.assertEqual((segs[0]["text"], segs[0]["raw_text"]), ("噔", "噔" * 30))
        self.assertNotIn("raw_text", segs[1])

    def test_transcribe_can_leave_the_language_to_the_model(self):
        _, pipe, processor = self._segments({2: ("Italian", "Ciao", "Ciao")}, [2])
        texts = pipe.transcribe([np.full(8, 2.0, dtype=np.float32)], None)
        self.assertEqual(processor.languages[-1], None)
        self.assertEqual((str(texts[0]), texts[0].language), ("Ciao", "Italian"))


class TestRetryBudgetHit(unittest.TestCase):
    def test_a_retry_that_still_loops_says_so(self):
        model = _LoopingModel(loops_under_penalty=True)
        pipe = QwenPipelineNative(model=model, processor=_FakeProcessor(), device="cpu",
                                  language="yue", batch_size=4, loop_retry_penalty=1.1)
        seg = pipe.process(_make_marked_segments([5]))["segments"][0]
        self.assertTrue(seg["retry_budget_hit"])

    def test_a_retry_that_recovers_does_not(self):
        pipe = QwenPipelineNative(model=_LoopingModel(), processor=_FakeProcessor(), device="cpu",
                                  language="yue", batch_size=4, loop_retry_penalty=1.1)
        seg = pipe.process(_make_marked_segments([5]))["segments"][0]
        self.assertTrue(seg["retried"])
        self.assertNotIn("retry_budget_hit", seg)


class _SteppingModel:
    """Decodes one token per step and honours ``stopping_criteria`` the way generate() does:
    a row the criteria end is padded (with an EOS id, as Qwen3-ASR pads) from then on.
    Rows with an odd marker repeat a three-token phrase forever; even ones say 5 then EOS."""

    EOS = 99

    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.generation_config = type("GC", (), {"eos_token_id": [self.EOS]})()
        self.steps = 0

    def generate(self, **kwargs):
        ids = kwargs["input_ids"]
        criteria = kwargs.get("stopping_criteria") or []
        done = torch.zeros(ids.shape[0], dtype=torch.bool)
        for step in range(kwargs["max_new_tokens"]):
            nxt = []
            for r, marker in enumerate(ids[:, 0].tolist()):
                if done[r]:
                    nxt.append(self.EOS)
                elif marker % 2:
                    nxt.append(7 + step % 3)
                else:
                    nxt.append(5 if step == 0 else self.EOS)
            ids = torch.cat([ids, torch.tensor(nxt).unsqueeze(1)], dim=1)
            self.steps += 1
            done |= torch.tensor([t == self.EOS for t in nxt])
            for c in criteria:
                done |= c(ids, None)
            if done.all():
                break
        return ids


class TestLoopStop(unittest.TestCase):
    def test_a_looping_row_is_stopped_and_flagged(self):
        model = _SteppingModel()
        pipe = QwenPipelineNative(model=model, processor=_FakeProcessor(), device="cpu",
                                  language="yue", batch_size=4, loop_retry_penalty=0)
        segs = pipe.process(_make_marked_segments([2, 3]))["segments"]
        self.assertLess(model.steps, 40, "the loop ran on toward its token budget")
        self.assertTrue(segs[1]["budget_hit"])
        self.assertNotIn("budget_hit", segs[0])

    def test_off_runs_the_whole_budget(self):
        model = _SteppingModel()
        pipe = QwenPipelineNative(model=model, processor=_FakeProcessor(), device="cpu",
                                  language="yue", batch_size=4, loop_retry_penalty=0,
                                  loop_stop=False)
        segs = pipe.process(_make_marked_segments([3]))["segments"]
        self.assertEqual(model.steps, pipe.max_new_tokens)
        self.assertTrue(segs[0]["budget_hit"])

    def test_the_rule(self):
        from cantocaptions_ai.pipeline._asr_native import _LoopStop

        stop = _LoopStop(n_input=1, eos_ids={99})

        def rows(*tails):
            return torch.tensor([[0] + list(t) for t in tails])

        phrase = [7, 8, 9, 10]
        self.assertTrue(stop(rows(phrase * 6))[0])            # 24 tokens, 6 copies
        self.assertFalse(stop(rows([1] * 3 + phrase * 4 + [2] * 7))[0])  # 4 copies, then on
        self.assertFalse(stop(rows([5] + [99] * 30))[0])      # finished: padding repeats
        self.assertFalse(stop(rows(list(range(30))))[0])

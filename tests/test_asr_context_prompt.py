"""Tests for ASR context biasing in the native backend (pipeline/_asr_native.py).

Covers:
- the no-context path still goes through apply_transcription_request unchanged
- the context path builds the official SDK's message shape (system = context alone,
  language forced by an assistant prefill), NOT transformers' language-in-system shape
- contexts are threaded from segment dicts through run() / process() in batch order

Mostly processor-mocked so it runs without model weights; one test renders through the
real chat template and skips when the checkpoint is not in the local HF cache.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cantocaptions_ai.pipeline._asr_native import QwenPipelineNative


def _pipeline(processor):
    """A QwenPipelineNative with everything but the processor stubbed out."""
    pipe = QwenPipelineNative.__new__(QwenPipelineNative)
    pipe.processor = processor
    pipe.model = MagicMock()
    pipe.max_new_tokens = 200
    pipe.policy = MagicMock(enabled=False)
    pipe.preset_language = "yue"
    import torch
    pipe.device = torch.device("cpu")
    return pipe


def _mock_processor():
    processor = MagicMock()
    processor.apply_chat_template.side_effect = lambda convs, **kw: [
        f"<|im_start|>system\n{c[0]['content']}<|im_end|>\n"
        "<|im_start|>user\n<audio><|im_end|>\n<|im_start|>assistant\n"
        for c in convs
    ]
    return processor


class TestBuildContextPrompts(unittest.TestCase):
    def test_system_carries_context_alone(self):
        processor = _mock_processor()
        pipe = _pipeline(processor)

        pipe._build_context_prompts(["ctx one", "ctx two"], "Cantonese")

        conversations = processor.apply_chat_template.call_args[0][0]
        self.assertEqual(len(conversations), 2)
        for conv, expected in zip(conversations, ["ctx one", "ctx two"]):
            self.assertEqual(conv[0]["role"], "system")
            # The language name must NOT be concatenated in here: the chat template
            # joins system text items with no separator.
            self.assertEqual(conv[0]["content"], expected)
            self.assertEqual(conv[1]["role"], "user")
            self.assertIn("audio", conv[1]["content"][0])

    def test_language_is_an_assistant_prefill(self):
        pipe = _pipeline(_mock_processor())
        prompts = pipe._build_context_prompts(["ctx"], "Cantonese")
        self.assertTrue(prompts[0].endswith("language Cantonese<asr_text>"))

    def test_generation_prompt_requested(self):
        processor = _mock_processor()
        _pipeline(processor)._build_context_prompts(["ctx"], "Cantonese")
        self.assertTrue(processor.apply_chat_template.call_args[1]["add_generation_prompt"])
        self.assertFalse(processor.apply_chat_template.call_args[1]["tokenize"])

    def test_none_context_becomes_empty_system(self):
        processor = _mock_processor()
        _pipeline(processor)._build_context_prompts([None], "Cantonese")
        self.assertEqual(processor.apply_chat_template.call_args[0][0][0][0]["content"], "")


class TestInferBatchRouting(unittest.TestCase):
    """The no-context path must stay byte-for-byte what it was before this feature."""

    def setUp(self):
        self.processor = _mock_processor()
        self.processor.decode.return_value = ["text"]
        self.pipe = _pipeline(self.processor)
        self.wavs = [np.zeros(1600, dtype=np.float32)]

    def _run(self, contexts):
        inputs = MagicMock()
        inputs.__getitem__.return_value.shape = [1, 4]
        self.processor.apply_transcription_request.return_value = inputs
        self.processor.return_value = inputs
        with patch("torch.inference_mode"):
            self.pipe.model.generate.return_value = MagicMock(shape=[1, 8])
            self.pipe._infer_batch(self.wavs, "Cantonese", contexts)

    def test_no_contexts_uses_apply_transcription_request(self):
        self._run(None)
        self.processor.apply_transcription_request.assert_called_once()
        self.processor.apply_chat_template.assert_not_called()

    def test_all_empty_contexts_uses_apply_transcription_request(self):
        self._run(["", None])
        self.processor.apply_transcription_request.assert_called_once()
        self.processor.apply_chat_template.assert_not_called()

    def test_any_context_uses_the_sdk_path(self):
        self._run(["ctx"])
        self.processor.apply_transcription_request.assert_not_called()
        self.processor.apply_chat_template.assert_called_once()
        # padding must be on: batched generation relies on the processor's
        # padding_side="left" default.
        self.assertTrue(self.processor.call_args[1]["padding"])


class TestContextThreading(unittest.TestCase):
    """Contexts must follow their own segment through BatchExecutor reordering."""

    def test_process_passes_matching_contexts(self):
        pipe = _pipeline(_mock_processor())
        pipe._batch_size = 8
        pipe.normalization = None
        segments = [
            {"start": 0.0, "end": 1.0, "audio": np.zeros(16000, dtype=np.float32), "context": "A"},
            {"start": 1.0, "end": 2.0, "audio": np.zeros(8000, dtype=np.float32), "context": "B"},
        ]
        seen = {}

        def fake_infer(wavs, language, contexts=None):
            for wav, ctx in zip(wavs, contexts):
                seen[len(wav)] = ctx
            return ["x"] * len(wavs)

        pipe._infer_batch = fake_infer
        with patch("cantocaptions_ai.pipeline._asr_native.normalize_segment_text",
                   side_effect=lambda seg, norm: seg):
            pipe.process(segments)

        self.assertEqual(seen, {16000: "A", 8000: "B"})

    def test_missing_context_key_is_none(self):
        pipe = _pipeline(_mock_processor())
        pipe._batch_size = 8
        pipe.normalization = None
        segments = [{"start": 0.0, "end": 1.0, "audio": np.zeros(1600, dtype=np.float32)}]
        captured = []

        pipe._infer_batch = lambda wavs, lang, contexts=None: (captured.append(contexts), ["x"])[1]
        with patch("cantocaptions_ai.pipeline._asr_native.normalize_segment_text",
                   side_effect=lambda seg, norm: seg):
            pipe.process(segments)

        self.assertEqual(captured, [[None]])


class TestGenerationBudget(unittest.TestCase):
    """Segments over the chunk budget need a proportionally larger token budget."""

    def _capture(self, seconds):
        processor = _mock_processor()
        processor.decode.return_value = ["t"]
        pipe = _pipeline(processor)
        inputs = MagicMock()
        inputs.__getitem__.return_value.shape = [1, 4]
        processor.apply_transcription_request.return_value = inputs
        pipe.model.generate.return_value = MagicMock(shape=[1, 8])
        with patch("torch.inference_mode"):
            pipe._infer_batch([np.zeros(int(seconds * 16000), dtype=np.float32)], "Cantonese")
        return pipe.model.generate.call_args[1]["max_new_tokens"]

    def test_short_segment_keeps_the_default(self):
        self.assertEqual(self._capture(10.0), 200)

    def test_at_the_chunk_budget_keeps_the_default(self):
        self.assertEqual(self._capture(25.0), 200)

    def test_over_long_segment_scales_up(self):
        # 78s was the longest produced by reference expansion on the Doraemon fixture;
        # at ~4.3 chars/s it needs ~336 characters, far past the 200-token default.
        self.assertEqual(self._capture(78.0), 78 * 8)

    def test_scales_off_the_longest_in_the_batch(self):
        processor = _mock_processor()
        processor.decode.return_value = ["t", "t"]
        pipe = _pipeline(processor)
        inputs = MagicMock(); inputs.__getitem__.return_value.shape = [2, 4]
        processor.apply_transcription_request.return_value = inputs
        pipe.model.generate.return_value = MagicMock(shape=[2, 8])
        wavs = [np.zeros(16000 * 5, dtype=np.float32), np.zeros(16000 * 60, dtype=np.float32)]
        with patch("torch.inference_mode"):
            pipe._infer_batch(wavs, "Cantonese")
        self.assertEqual(pipe.model.generate.call_args[1]["max_new_tokens"], 60 * 8)


class TestRealChatTemplate(unittest.TestCase):
    """Render through the actual Qwen3-ASR chat template, if it is cached locally."""

    def test_rendered_prompt_matches_sdk_shape(self):
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-ASR-1.7B-hf", local_files_only=True
            )
        except Exception as e:  # not cached / transformers too old
            self.skipTest(f"Qwen3-ASR processor unavailable locally: {e}")

        pipe = _pipeline(processor)
        prompt = pipe._build_context_prompts(["Bluey plays keepy uppy"], "Cantonese")[0]

        self.assertIn("<|im_start|>system\nBluey plays keepy uppy<|im_end|>", prompt)
        # The language must not have leaked into the system turn.
        self.assertNotIn("Cantonese<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("language Cantonese<asr_text>"))


if __name__ == "__main__":
    unittest.main()

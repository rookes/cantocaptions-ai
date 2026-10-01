"""The ASR backend registry, and the batching every backend shares.

Which backend runs is a property of the model: a registered model names its backend, any
other is identified from its checkpoint's ``config.json``. Tiny configs are written to
temp dirs here, so nothing is downloaded.
"""
import importlib
import importlib.util
import inspect

import numpy as np
import pytest

from cantocaptions_ai.pipeline import asr
from cantocaptions_ai.pipeline.asr import (
    ASR_BACKENDS,
    AsrStage,
    BatchedAsrStage,
    QwenPipeline,
    backend_for,
)
from cantocaptions_ai.text_profiles import TextNormalization


def _save_config(tmp_path, config):
    path = tmp_path / config.model_type
    config.save_pretrained(path)
    return str(path)


# --- registry and detection ----------------------------------------------------------

def test_registered_models_name_their_backend():
    assert backend_for("cantocaptions-cantonese-ASR").name == "qwen3-asr"
    assert backend_for(None, "yue").name == "qwen3-asr"  # the language's default model
    assert backend_for("Qwen3-ASR").supports_context


def test_an_unsupported_model_type_is_named_in_the_error(tmp_path):
    from transformers import BertConfig

    with pytest.raises(ValueError, match="'bert'"):
        backend_for(_save_config(tmp_path, BertConfig()))


def test_only_qwen_takes_a_context_prompt():
    assert {b.name for b in ASR_BACKENDS.values() if b.supports_context} == {"qwen3-asr"}


def test_every_backend_loader_resolves():
    for backend in ASR_BACKENDS.values():
        module, _, func = backend.loader.partition(":")
        assert callable(getattr(importlib.import_module(module), func)), backend.name


def test_qwen_without_native_support_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(asr, "_has_native_qwen3asr", lambda: False)
    with pytest.raises(ImportError, match="transformers_qwen"):
        asr.load_model("cantocaptions-cantonese-ASR", device="cpu")


def test_the_legacy_backend_is_gone():
    assert importlib.util.find_spec("cantocaptions_ai.pipeline._asr_legacy") is None


def test_qwenpipeline_is_kept_as_the_stage_base():
    assert QwenPipeline is AsrStage


def test_load_model_native_keeps_the_signature_the_dataset_repo_calls():
    from cantocaptions_ai.pipeline._asr_native import load_model_native

    params = inspect.signature(load_model_native).parameters
    for name in ("model_name", "device", "device_index", "model", "processor", "batch_size",
                 "vram_checks", "normalization"):
        assert name in params, name


# --- the shared batched stage --------------------------------------------------------

class _EchoAsr(BatchedAsrStage):
    """Transcribes each segment as its length in samples, recording each batch."""

    backend_label = "echo"

    def __init__(self, **kwargs):
        super().__init__(device="cpu", language="en", **kwargs)
        self.batches = []

    def _infer_batch(self, wavs, language, contexts=None):
        self.batches.append([len(w) for w in wavs])
        return [f"{language}:{len(w)}" for w in wavs]


def _segments(*lengths):
    return [{"start": float(i), "end": float(i) + 0.5, "audio": np.zeros(n, np.float32)}
            for i, n in enumerate(lengths)]


def test_batches_pool_segments_across_files_longest_first():
    stage = _EchoAsr(batch_size=3)
    items = [{"audio_path": "a.wav", "vad_segments": _segments(10, 30)},
             {"audio_path": "b.wav", "vad_segments": _segments(20, 40, 5)}]
    out = stage.run(items)
    assert stage.batches == [[40, 30, 20], [10, 5]]
    assert [s["text"] for s in out[0]["result"]["segments"]] == ["en:10", "en:30"]
    assert [s["text"] for s in out[1]["result"]["segments"]] == ["en:20", "en:40", "en:5"]
    assert out[1]["result"]["language"] == "en"


def test_the_language_packs_normalization_is_applied():
    stage = _EchoAsr(batch_size=4, normalization=TextNormalization(chars_hk=True))
    result = stage.process(_segments(10))
    assert result["segments"][0]["text"] == "en:10"  # nothing HK-specific to rewrite here
    assert stage.normalization.chars_hk

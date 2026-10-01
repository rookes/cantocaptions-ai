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
    with pytest.raises(ImportError, match="transformers>=5.13"):
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


# --- whisper -------------------------------------------------------------------------

def test_whisper_models_are_registered_and_detected(tmp_path):
    from transformers import WhisperConfig

    assert backend_for("whisper-large-v3").name == "whisper"
    assert backend_for(_save_config(tmp_path, WhisperConfig())).name == "whisper"
    assert not backend_for("whisper-large-v3-turbo").supports_context


def test_whisper_takes_qwens_cantonese_conventions():
    from cantocaptions_ai.languages.yue import YUE

    assert YUE.conventions["whisper-large-v3"] is YUE.conventions["Qwen3-ASR"]
    assert YUE.resolve("whisper-large-v3").cleaning.manifest == "pipeline_qwen.toml"


_TS = 100  # the fakes' first timestamp token id
_WORDS = {1: "準備好了嗎?", 2: "好厲害呀", 3: "我是小白"}


class _FakeWhisperProcessor:
    class tokenizer:
        @staticmethod
        def convert_tokens_to_ids(token):
            assert token == "<|0.00|>"
            return _TS

    def __call__(self, wavs, sampling_rate, return_tensors, return_attention_mask):
        import torch

        assert sampling_rate == 16000 and return_tensors == "pt"
        return {"input_features": torch.zeros(len(wavs), 4, 6),
                "attention_mask": torch.ones(len(wavs), 6, dtype=torch.long)}

    def batch_decode(self, phrases, skip_special_tokens):
        assert skip_special_tokens
        return [" ".join(_WORDS[t] for t in phrase if t in _WORDS) for phrase in phrases]


class _FakeWhisperModel:
    """Every row: <|0.00|> phrase 1 <|1.00|><|1.20|> phrases 2 and 3 <|3.00|>, eos padding."""

    def __init__(self):
        import torch

        self.device, self.dtype, self.calls = torch.device("cpu"), torch.float32, []

    def generate(self, input_features, attention_mask, language, task, return_timestamps):
        import torch

        self.calls.append((language, task, return_timestamps, input_features.shape[0]))
        row = [_TS, 1, _TS + 50, _TS + 60, 2, _TS + 70, _TS + 75, 3, _TS + 150, 0]
        return torch.tensor([row] * input_features.shape[0])


def test_whisper_joins_its_timestamped_phrases_as_clauses():
    from cantocaptions_ai.pipeline._asr_whisper import WhisperAsr

    model = _FakeWhisperModel()
    stage = WhisperAsr(model, _FakeWhisperProcessor(), device="cpu", language="yue", batch_size=2)
    result = stage.process(_segments(100, 200, 300))
    assert [s["text"] for s in result["segments"]] == ["準備好了嗎？好厲害呀，我是小白"] * 3
    assert result["language"] == "yue"
    assert model.calls == [("yue", "transcribe", True, 2), ("yue", "transcribe", True, 1)]


def test_join_clauses_marks_unpunctuated_boundaries_in_the_languages_own_marks():
    from cantocaptions_ai.text_profiles import (
        CJK_PUNCTUATION,
        CJK_SCRIPT,
        LATIN_PUNCTUATION,
        SPACED_SCRIPT,
    )

    assert asr.join_clauses(["來吧, 天竺", "", "好呀!", "Hello there, Desmond"],
                            CJK_PUNCTUATION, CJK_SCRIPT) == "來吧，天竺，好呀！Hello there, Desmond"
    assert asr.join_clauses(["hello there.", "how are you", " fine "],
                            LATIN_PUNCTUATION, SPACED_SCRIPT) == "hello there. how are you, fine"


def test_whisper_loader_takes_the_common_load_arguments():
    from cantocaptions_ai.pipeline._asr_whisper import load_model_whisper

    params = inspect.signature(load_model_whisper).parameters
    for name in ("model_name", "device", "device_index", "compute_type", "language",
                 "download_root", "local_files_only", "batch_size", "vram_checks",
                 "vram_headroom_mb", "normalization"):
        assert name in params, name


# --- ctc -----------------------------------------------------------------------------

_CTC_VOCAB = ["<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c"]


def _tiny_ctc_checkpoint(tmp_path):
    """A random 1-layer wav2vec2 CTC model with its processor, saved like a hub checkpoint."""
    import json

    import torch
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    config = Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(16,) * 7, vocab_size=len(_CTC_VOCAB), pad_token_id=0,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(0)
    model = Wav2Vec2ForCTC(config).eval()
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps({tok: i for i, tok in enumerate(_CTC_VOCAB)}))
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_file), pad_token="<pad>", unk_token="<unk>",
                                     word_delimiter_token="|")
    extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                         do_normalize=True, return_attention_mask=True)
    checkpoint = tmp_path / "ctc"
    Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer).save_pretrained(checkpoint)
    model.save_pretrained(checkpoint)
    return str(checkpoint)


def _noise(*seconds):
    rng = np.random.default_rng(0)
    return [{"start": float(i), "end": float(i) + s,
             "audio": rng.standard_normal(int(s * 16000)).astype(np.float32)}
            for i, s in enumerate(seconds)]


def test_a_wav2vec2_checkpoint_is_detected_and_loads_as_ctc(tmp_path):
    from cantocaptions_ai.pipeline._asr_ctc import CtcAsr

    checkpoint = _tiny_ctc_checkpoint(tmp_path)
    assert backend_for(checkpoint).name == "ctc"
    stage = asr.load_model(checkpoint, device="cpu", language="en", local_files_only=True,
                           batch_size=2, vram_checks=False)
    assert isinstance(stage, CtcAsr)
    result = stage.process(_noise(1.0, 0.5, 1.5))
    assert len(result["segments"]) == 3
    assert all(isinstance(s["text"], str) for s in result["segments"])
    assert result["language"] == "en"


def test_ctc_decodes_greedily_and_collapses_repeats_per_segment(tmp_path):
    """Scripted logits: every segment spells "a a <pad> a b | c", then blanks to its end."""
    import torch
    from transformers import AutoProcessor

    from cantocaptions_ai.pipeline._asr_ctc import CtcAsr

    checkpoint = _tiny_ctc_checkpoint(tmp_path)
    processor = AutoProcessor.from_pretrained(checkpoint)
    script = [_CTC_VOCAB.index(t) for t in ("a", "a", "<pad>", "a", "b", "|", "c")]

    from transformers import Wav2Vec2ForCTC

    class _Scripted(Wav2Vec2ForCTC):
        def forward(self, input_values, attention_mask=None, **_):
            batch, samples = input_values.shape
            frames = int(self._get_feat_extract_output_lengths(torch.tensor(samples)))
            logits = torch.zeros(batch, frames, len(_CTC_VOCAB))
            logits[:, :, 0] = 1.0
            for t, token in enumerate(script):
                logits[:, t, token] = 5.0
            return type("Out", (), {"logits": logits})()

    model = _Scripted.from_pretrained(checkpoint).eval()
    stage = CtcAsr(model, processor, device="cpu", language="en", batch_size=4,
                   vram_checks=False)
    result = stage.process(_noise(1.0, 2.0))
    assert [s["text"] for s in result["segments"]] == ["aab c", "aab c"]


def test_ctc_gives_a_segment_too_short_for_one_frame_no_text(tmp_path):
    from transformers import AutoModelForCTC, AutoProcessor

    from cantocaptions_ai.pipeline._asr_ctc import CtcAsr

    checkpoint = _tiny_ctc_checkpoint(tmp_path)
    stage = CtcAsr(AutoModelForCTC.from_pretrained(checkpoint).eval(),
                   AutoProcessor.from_pretrained(checkpoint), device="cpu", language="en",
                   vram_checks=False)
    assert stage._infer_batch([np.zeros(100, np.float32)], "en") == [""]


@pytest.mark.parametrize("model", ["whisper-large-v3", "ctc"])
def test_asr_context_is_refused_for_a_backend_without_it(tmp_path, model):
    from cantocaptions_ai.pipeline.config import PipelineConfig
    from cantocaptions_ai.errors import ConfigError
    from cantocaptions_ai.pipeline.transcribe import validate_config

    if model == "ctc":
        model = _tiny_ctc_checkpoint(tmp_path)
    cfg = PipelineConfig(model=model, reference_subtitle="ref.srt", asr_context=True,
                         model_cache_only=True)
    with pytest.raises(ConfigError, match="asr_context is not supported"):
        validate_config(cfg)


def test_ctc_pauses_cut_only_inside_long_internal_silences():
    from cantocaptions_ai.pipeline._asr_ctc import _split_at_pauses

    _, a, b = 0, 5, 6
    path = [_, _, _, a, _, _, b, _, _, _, _, a, _, _, _]
    # Leading/trailing silence is never a cut; the 2-frame gap is too short; the 4-frame one cuts.
    assert _split_at_pauses(path, {0}, 3) == [path[:9], path[9:]]
    assert _split_at_pauses(path, {0}, 5) == [path]


def test_ctc_writes_pauses_as_the_languages_mergeable_mark(tmp_path):
    from transformers import AutoProcessor

    from cantocaptions_ai.pipeline._asr_ctc import CtcAsr
    from cantocaptions_ai.text_profiles import LATIN_PUNCTUATION, SPACED_SCRIPT

    processor = AutoProcessor.from_pretrained(_tiny_ctc_checkpoint(tmp_path))
    a, b, c, bar = (_CTC_VOCAB.index(t) for t in ("a", "b", "c", "|"))
    path = [a, b, bar, c] + [0] * 10 + [b, a]   # "ab c", a 0.4 s pause at 25 fps, "ba"

    cjk = CtcAsr(None, processor, device="cpu", vram_checks=False)
    assert cjk._decode(path, frame_rate=25.0) == "ab c，ba"
    latin = CtcAsr(None, processor, device="cpu", vram_checks=False,
                   punctuation=LATIN_PUNCTUATION, script=SPACED_SCRIPT)
    assert latin._decode(path, frame_rate=25.0) == "ab c, ba"
    assert latin._decode(path, frame_rate=100.0) == "ab cba"   # a 0.1 s gap: no cut

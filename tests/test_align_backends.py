"""The alignment backend registry: which family a model is, and each family's own path.

Nothing is downloaded. The Hugging Face family uses a tiny random wav2vec2 checkpoint
written to a temp dir; the torchaudio family uses a tiny random torchaudio wav2vec2,
served by a fake pipeline bundle so the whole load path runs as it does for
``WAV2VEC2_ASR_BASE_960H``.
"""
import inspect
import json
import logging

import numpy as np
import pytest
import torch
import torchaudio

from cantocaptions_ai.pipeline import align_backends, alignment
from cantocaptions_ai.pipeline.align_backends import (
    ALIGN_BACKENDS,
    CTC_MODEL_TYPES,
    align_backend_for,
    get_align_backend,
)
from cantocaptions_ai.pipeline.align_profiles import TailPrimer
from cantocaptions_ai.utils.audio import SAMPLE_RATE

LABELS = ("-", "|", "E", "T", "A", "O", "N", "I")


def _tiny_torchaudio_wav2vec2():
    """Same conv stride (320 samples, 50 fps) as the real bundles; everything else tiny."""
    torch.manual_seed(0)
    return torchaudio.models.wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=[(16, 10, 5)] + [(16, 3, 2)] * 4 + [(16, 2, 2)] * 2,
        extractor_conv_bias=False, encoder_embed_dim=16, encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=16, encoder_pos_conv_groups=2, encoder_num_layers=1,
        encoder_num_heads=2, encoder_attention_dropout=0.0, encoder_ff_interm_features=32,
        encoder_ff_interm_dropout=0.0, encoder_dropout=0.0, encoder_layer_norm_first=False,
        encoder_layer_drop=0.0, aux_num_out=len(LABELS),
    ).eval()


class _FakeBundle:
    def get_model(self, dl_kwargs=None):
        return _tiny_torchaudio_wav2vec2()

    def get_labels(self):
        return LABELS


@pytest.fixture
def fake_bundle(monkeypatch):
    name = "TINY_TEST_BUNDLE"
    monkeypatch.setattr(torchaudio.pipelines, name, _FakeBundle(), raising=False)
    monkeypatch.setattr(torchaudio.pipelines, "__all__", [*torchaudio.pipelines.__all__, name])
    return name


def _hf_checkpoint(tmp_path):
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    vocab = ["<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c"]
    config = Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(16,) * 7, vocab_size=len(vocab), pad_token_id=0,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(0)
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps({tok: i for i, tok in enumerate(vocab)}))
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_file), pad_token="<pad>", unk_token="<unk>",
                                     word_delimiter_token="|")
    extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE,
                                         padding_value=0.0, do_normalize=True,
                                         return_attention_mask=False)
    checkpoint = tmp_path / "hf-ctc"
    Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer).save_pretrained(checkpoint)
    Wav2Vec2ForCTC(config).save_pretrained(checkpoint)
    return str(checkpoint)


def _segments(*seconds):
    rng = np.random.default_rng(0)
    out, t = [], 0.0
    for s in seconds:
        out.append({"start": t, "end": t + s,
                    "audio": rng.standard_normal(int(s * SAMPLE_RATE)).astype(np.float32)})
        t += s + 0.5
    return out


# --- registry and detection ----------------------------------------------------------

def test_backend_names_are_the_metadata_types_callers_read():
    assert set(ALIGN_BACKENDS) == {"huggingface", "torchaudio"}
    assert ALIGN_BACKENDS["huggingface"].supports_primer
    assert not ALIGN_BACKENDS["torchaudio"].supports_primer


def test_a_torchaudio_bundle_name_is_the_torchaudio_backend():
    assert align_backend_for("WAV2VEC2_ASR_BASE_960H").name == "torchaudio"


def test_a_ctc_checkpoint_is_the_huggingface_backend(tmp_path):
    assert align_backend_for(_hf_checkpoint(tmp_path), local_files_only=True).name == "huggingface"


def test_a_non_ctc_checkpoint_is_refused_by_its_model_type(tmp_path):
    from transformers import BertConfig

    path = tmp_path / "bert"
    BertConfig().save_pretrained(path)
    with pytest.raises(ValueError, match="'bert' model; alignment needs a CTC model"):
        align_backend_for(str(path))


def test_an_unknown_model_is_not_found(tmp_path):
    with pytest.raises(ValueError, match="could not be found"):
        align_backend_for(str(tmp_path / "missing"), local_files_only=True)


def test_an_unreachable_hub_is_re_raised_for_the_offline_fallback(monkeypatch):
    from huggingface_hub.errors import OfflineModeIsEnabled
    from transformers import AutoConfig

    def offline(*args, **kwargs):
        raise OfflineModeIsEnabled("hub unreachable")

    monkeypatch.setattr(AutoConfig, "from_pretrained", offline)
    with pytest.raises(OfflineModeIsEnabled):
        align_backend_for("someone/some-ctc-model")


def test_an_unknown_type_in_hand_built_metadata_is_a_clear_error():
    with pytest.raises(ValueError, match="'onnx' not supported"):
        get_align_backend("onnx")


def test_the_ctc_asr_backend_detects_the_same_families():
    from cantocaptions_ai.pipeline.asr import _MODEL_TYPE_BACKENDS

    assert {t for t, b in _MODEL_TYPE_BACKENDS.items() if b == "ctc"} == set(CTC_MODEL_TYPES)


# --- torchaudio ----------------------------------------------------------------------

def test_load_align_model_runs_a_torchaudio_bundle_end_to_end(fake_bundle):
    model, metadata = alignment.load_align_model(
        "en", "cpu", model_name=fake_bundle, vram_checks=False, char_substitution="off",
    )
    assert metadata["type"] == "torchaudio"
    assert metadata["processor"] is None
    assert metadata["frame_rate"] == 50.0
    assert metadata["dictionary"] == {c.lower(): i for i, c in enumerate(LABELS)}

    results = alignment.compute_vad_emissions(
        _segments(1.0, 0.5), model, metadata["type"], metadata["processor"], "cpu",
        vram_checks=False,
    )
    assert [tuple(e.shape) for e, _ in results] == [(49, len(LABELS)), (24, len(LABELS))]
    for emission, rate in results:
        assert torch.allclose(emission.exp().sum(-1), torch.ones(emission.shape[0]), atol=1e-4)
        assert rate == pytest.approx(emission.shape[0] / (len(emission) / 50), rel=0.05)


def test_torchaudio_warns_that_it_does_not_apply_a_primer(caplog):
    model = _tiny_torchaudio_wav2vec2()
    segments = _segments(1.0)
    plain = alignment.compute_vad_emissions(segments, model, "torchaudio", None, "cpu",
                                            vram_checks=False)
    with caplog.at_level(logging.WARNING):
        primed = alignment.compute_vad_emissions(segments, model, "torchaudio", None, "cpu",
                                                 vram_checks=False, primer=TailPrimer())
    assert "does not apply it" in caplog.text
    assert torch.equal(plain[0][0], primed[0][0])


def test_run_model_inference_dispatches_by_type(tmp_path):
    waveform = torch.from_numpy(_segments(1.0)[0]["audio"]).unsqueeze(0)
    torch_out = alignment._run_model_inference(
        _tiny_torchaudio_wav2vec2(), "torchaudio", waveform, None, "cpu")
    assert tuple(torch_out.shape) == (1, 49, len(LABELS))

    from transformers import AutoModelForCTC, AutoProcessor

    checkpoint = _hf_checkpoint(tmp_path)
    hf_out = alignment._run_model_inference(
        AutoModelForCTC.from_pretrained(checkpoint).eval(), "huggingface", waveform,
        AutoProcessor.from_pretrained(checkpoint), "cpu")
    assert tuple(hf_out.shape) == (1, 49, 8)


# --- the surface other repos use ------------------------------------------------------

def test_the_names_cantocaptions_dataset_imports_still_resolve():
    for name in ("load_align_model", "_run_model_inference", "_get_blank_id", "get_trellis",
                 "backtrack", "merge_repeats", "align_best_of", "compute_vad_emissions",
                 "_compute_vad_emissions_batched", "MIN_ALIGN_SAMPLES"):
        assert hasattr(alignment, name), name
    params = list(inspect.signature(alignment._run_model_inference).parameters)
    assert params[:5] == ["model", "model_type", "audio", "processor", "device"]


def test_hf_metadata_keeps_the_keys_callers_read(tmp_path):
    _, metadata = alignment.load_align_model(
        "en", "cpu", model_name=_hf_checkpoint(tmp_path), model_cache_only=True,
        vram_checks=False, char_substitution="off",
    )
    assert {"language", "dictionary", "type", "processor", "frame_rate", "vocab_repair",
            "profile"} <= set(metadata)
    assert metadata["type"] == "huggingface"
    assert align_backends.get_align_backend(metadata["type"]) is ALIGN_BACKENDS["huggingface"]


# --- the default model comes from the language pack -----------------------------------

def test_a_registered_packs_default_align_model_reaches_library_callers(fake_bundle, monkeypatch):
    from cantocaptions_ai import languages
    from cantocaptions_ai.languages import LanguagePack
    from cantocaptions_ai.text_profiles import LATIN_PUNCTUATION, SPACED_SCRIPT

    pack = LanguagePack("xx", SPACED_SCRIPT, LATIN_PUNCTUATION, default_align_model=fake_bundle)
    monkeypatch.setitem(languages._PACKS, "xx", pack)
    _, metadata = alignment.load_align_model("xx", "cpu", vram_checks=False,
                                             char_substitution="off")
    assert metadata["type"] == "torchaudio"
    assert metadata["language"] == "xx"


def test_a_language_with_no_default_align_model_still_says_so():
    with pytest.raises(ValueError, match="No default align-model for language: sw"):
        alignment.load_align_model("sw", "cpu", vram_checks=False)

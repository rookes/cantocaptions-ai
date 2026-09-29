"""Alignment must work with any Hugging Face CTC model, not only wav2vec2-BERT.

The align stage used to feed every Hugging Face model through the Cantonese wav2vec2-BERT
feature extractor (fbank ``input_features``), so a plain wav2vec2 model -- which wants raw
``input_values`` -- crashed in its first conv layer. Each model now brings its own processor
in ``align_metadata["processor"]``.

Everything here is built from a tiny random config and saved to a temp dir: no weights are
downloaded, so the timings are meaningless and only shapes and lengths are asserted.
"""
import json

import numpy as np
import pytest
import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from cantocaptions_ai.pipeline.alignment import (
    _input_key,
    _passes_attention_mask,
    compute_vad_emissions,
    load_align_model,
)
from cantocaptions_ai.utils.audio import SAMPLE_RATE

VOCAB = ["<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c"]
# The default wav2vec2 conv encoder downsamples by 320 samples: 50 emission frames a second.
WAV2VEC2_FPS = SAMPLE_RATE / 320


def _tiny_wav2vec2(return_attention_mask: bool):
    config = Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(16,) * 7, vocab_size=len(VOCAB), pad_token_id=0,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(0)
    model = Wav2Vec2ForCTC(config).eval()
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0, do_normalize=True,
        return_attention_mask=return_attention_mask,
    )
    return model, extractor


def _segments(durations):
    rng = np.random.default_rng(0)
    out, t = [], 0.0
    for d in durations:
        audio = rng.standard_normal(int(d * SAMPLE_RATE)).astype(np.float32)
        out.append({"start": t, "end": t + d, "audio": audio})
        t += d + 0.5
    return out


@pytest.mark.parametrize("return_attention_mask", [False, True])
def test_plain_wav2vec2_emissions_are_trimmed_to_each_segment(return_attention_mask):
    """Batched segments of different lengths come back unpadded, at the model's frame rate."""
    model, extractor = _tiny_wav2vec2(return_attention_mask)
    segments = _segments([1.0, 2.5, 0.6])
    results = compute_vad_emissions(
        segments, model, "huggingface", extractor, "cpu", batch_size=3, vram_checks=False,
    )
    for seg, (emission, frame_rate) in zip(segments, results):
        n_samples = len(seg["audio"])
        expected = int(model._get_feat_extract_output_lengths(torch.tensor(n_samples)))
        assert emission.shape == (expected, len(VOCAB))
        # Within a frame of nominal: the conv encoder loses up to one frame at the edges,
        # which is 3% of a 0.6 s segment.
        assert frame_rate == pytest.approx(WAV2VEC2_FPS, rel=0.05)


def test_model_is_not_given_a_mask_its_extractor_does_not_expect():
    """Group-norm wav2vec2 checkpoints must see zero padding, not an attention mask."""
    model, extractor = _tiny_wav2vec2(return_attention_mask=False)
    seen = []
    original = model.forward

    def spy(input_values, attention_mask=None, **kwargs):
        seen.append(attention_mask)
        return original(input_values, attention_mask=attention_mask, **kwargs)

    model.forward = spy
    compute_vad_emissions(_segments([1.0, 2.0]), model, "huggingface", extractor, "cpu",
                          batch_size=2, vram_checks=False)
    assert seen and all(mask is None for mask in seen)


def test_input_key_and_mask_policy_follow_the_processor():
    _, plain = _tiny_wav2vec2(return_attention_mask=False)
    assert _input_key(plain) == "input_values"
    assert _passes_attention_mask(plain) is False
    # A bare callable (the fakes in test_alignment_batching) keeps the wav2vec2-BERT defaults.
    assert _input_key(lambda *a, **k: None) == "input_features"
    assert _passes_attention_mask(lambda *a, **k: None) is True


def test_huggingface_model_without_a_processor_is_an_error():
    model, _ = _tiny_wav2vec2(return_attention_mask=False)
    with pytest.raises(ValueError, match="processor"):
        compute_vad_emissions(_segments([1.0]), model, "huggingface", None, "cpu",
                              vram_checks=False)


def test_load_align_model_returns_the_models_own_processor(tmp_path):
    """A local wav2vec2 checkpoint loads through the Auto classes, processor included."""
    model, extractor = _tiny_wav2vec2(return_attention_mask=False)
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps({tok: i for i, tok in enumerate(VOCAB)}))
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_file), pad_token="<pad>", unk_token="<unk>",
                                     word_delimiter_token="|")
    checkpoint = tmp_path / "ckpt"
    Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer).save_pretrained(checkpoint)
    model.save_pretrained(checkpoint)

    loaded, metadata = load_align_model(
        "en", "cpu", model_name=str(checkpoint), model_cache_only=True, vram_checks=False,
        char_substitution="off",
    )
    assert isinstance(loaded, Wav2Vec2ForCTC)
    assert metadata["type"] == "huggingface"
    assert _input_key(metadata["processor"]) == "input_values"
    assert metadata["dictionary"]["a"] == VOCAB.index("a")

    results = compute_vad_emissions(
        _segments([1.0]), loaded, metadata["type"], metadata["processor"], "cpu",
        vram_checks=False,
    )
    assert results[0][0].shape[-1] == len(VOCAB)

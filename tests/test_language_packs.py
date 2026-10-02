"""The language registry: what a pack provides, and what an unregistered language gets."""
import pytest

from cantocaptions_ai import languages
from cantocaptions_ai.languages import (
    LANGUAGE_PACKS,
    LanguagePack,
    generic_pack,
    get_language_pack,
)
from cantocaptions_ai.languages.yue import YUE
from cantocaptions_ai.pipeline.model_profiles import MODEL_PROFILES
from cantocaptions_ai.text_profiles import CJK_SCRIPT, LATIN_PUNCTUATION, SPACED_SCRIPT


def test_cantonese_is_registered_and_fully_supported():
    assert get_language_pack("yue") is YUE
    assert LANGUAGE_PACKS["yue"] is YUE
    assert YUE.fully_supported
    assert get_language_pack(None) is YUE  # the project's default language


def test_yue_resolves_its_default_model():
    profile = YUE.resolve(None)
    assert (profile.model, profile.hf_id) == (
        "cantocaptions-cantonese-ASR", "rookes/cantocaptions-cantonese-asr")
    assert YUE.default_align_model == "alvanlii/wav2vec2-BERT-cantonese"


def test_every_model_yue_has_conventions_for_is_a_known_model():
    assert set(YUE.conventions) <= set(MODEL_PROFILES) | {"Qwen3-ASR-lora"}


def test_an_unregistered_language_gets_a_raw_generic_pack():
    en = get_language_pack("en")
    assert "en" not in LANGUAGE_PACKS
    assert (en.script, en.punctuation) == (SPACED_SCRIPT, LATIN_PUNCTUATION)
    assert en.default_align_model == "facebook/wav2vec2-base-960h"
    assert en.default_model is None and en.cleaning is None and not en.fully_supported
    assert generic_pack("ja").script == CJK_SCRIPT
    assert generic_pack("sw").default_align_model is None


def test_track_choice_follows_the_pack():
    streams = [{"tags": {"language": "eng"}}, {"tags": {"language": "yue"}},
               {"tags": {"language": "jpn"}}]
    assert YUE.choose_track(streams) == 1
    assert get_language_pack("ja").choose_track(streams) == 2
    custom = LanguagePack("en", SPACED_SCRIPT, LATIN_PUNCTUATION, track_selector=lambda s: 2)
    assert custom.choose_track(streams) == 2


def test_registering_a_pack_makes_it_the_languages_pack(monkeypatch):
    pack = LanguagePack("en", SPACED_SCRIPT, LATIN_PUNCTUATION, default_model="some/english-asr")
    monkeypatch.setitem(languages._PACKS, "en", pack)
    assert get_language_pack("en") is pack
    assert get_language_pack("en").resolve(None).model == "some/english-asr"


def test_yue_cleaning_spec_builds_a_working_cleaner():
    from cantocaptions_ai.cleaning import SubtitleCleaner

    spec = YUE.cleaning
    cleaner = SubtitleCleaner(rules_dir=spec.rules_dir, manifest=spec.manifest,
                              builtin_steps=spec.builtin_steps(),
                              noise_tokens=spec.noise_tokens, max_line_count=2)
    assert cleaner.clean("十一月二十三號") == "11月23號"
    assert cleaner.is_noise("嗯")


@pytest.mark.parametrize("code", ["yue", "en"])
def test_resolving_never_needs_torch(code):
    """Pack resolution runs in preflight checks: it must not import the model stack."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; from cantocaptions_ai.languages import get_language_pack;"
         f"get_language_pack('{code}').resolve('Qwen3-ASR'); print('torch' in sys.modules)"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.stdout.strip() == "False", out.stderr


def test_a_pack_defers_its_pronunciation_data():
    """char_readings is a factory: reading the registry must not load pycantonese or OpenCC."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; from cantocaptions_ai.languages import get_language_pack;"
         "p = get_language_pack('yue'); p.resolve(None); assert p.char_readings is not None;"
         "print(sorted(m for m in ('pycantonese', 'opencc') if m in sys.modules))"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.stdout.strip() == "[]", out.stderr

"""Tests for the per-model downstream configuration (pipeline/model_profiles.py) and
the profile-driven text normalization / punctuation helpers in cantonese/text.py.

Covers:
- get_model_profile: registered vs unknown (all-default no-op) profiles
- normalize_segment_text: no-op default vs OpenCC / HK-variant application
- PunctuationConfig.sentence_spans splitting
- the migrated Qwen spot-check table (incl. the 咁->噉 weight)
"""

import unittest

from cantocaptions_ai.cantonese.text import (
    DEFAULT_NORMALIZATION,
    PunctuationConfig,
    SegmentationConfig,
    SpotCheck,
    TextNormalization,
    normalize_segment_text,
    standardize_chars_hk,
)
from cantocaptions_ai.pipeline.model_profiles import MODEL_PROFILES, get_model_profile


class TestGetModelProfile(unittest.TestCase):
    def test_unknown_model_is_all_default_noop(self):
        profile = get_model_profile("some/random-model-path")
        self.assertEqual(profile.hf_id, "some/random-model-path")
        # No OpenCC, no HK-variant rewriting.
        self.assertIsNone(profile.normalization.opencc_config)
        self.assertFalse(profile.normalization.chars_hk)
        # No alignment spot checks.
        self.assertEqual(dict(profile.spotchecks), {})
        # Standard punctuation.
        self.assertEqual(profile.punctuation, PunctuationConfig())
        # No discourse markers get a widened rescue window.
        self.assertEqual(profile.segmentation, SegmentationConfig())

    def test_vanilla_qwen_preserves_behavior(self):
        profile = get_model_profile("Qwen3-ASR")
        self.assertEqual(profile.hf_id, "Qwen/Qwen3-ASR-1.7B-hf")
        self.assertEqual(profile.normalization.opencc_config, "s2t_c.json")
        self.assertTrue(profile.normalization.chars_hk)
        # The 咁 spot-check keeps the historical +0.8 bias toward 噉.
        self.assertIn("咁", profile.spotchecks)
        gam = profile.spotchecks["咁"]
        self.assertEqual(gam.candidates, ("咁", "噉"))
        self.assertAlmostEqual(gam.weights.get("噉"), 0.8)
        # Qwen punctuates leading discourse markers off as their own clause.
        self.assertIn("嗱", profile.segmentation.leading_markers)

    def test_finetuned_lora_is_clean_slate(self):
        # Whether registered (env set) or resolved as a passthrough (env unset), the
        # LoRA model gets all-default no-op downstream behavior.
        profile = get_model_profile("Qwen3-ASR-lora")
        self.assertIsNone(profile.normalization.opencc_config)
        self.assertFalse(profile.normalization.chars_hk)
        self.assertEqual(dict(profile.spotchecks), {})
        self.assertEqual(profile.segmentation.leading_markers, ())

    def test_registry_keys_are_the_cli_choices_source(self):
        # __main__ derives --model choices from these keys.
        self.assertIn("Qwen3-ASR", MODEL_PROFILES)

    def test_lora_profile_registered_only_when_env_set(self):
        import os
        from unittest import mock
        from cantocaptions_ai.pipeline.model_profiles import _build_profiles, _LORA_MODEL_DIR_ENV

        # Unset: no machine-specific path ships; the choice is simply unavailable.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_LORA_MODEL_DIR_ENV, None)
            self.assertNotIn("Qwen3-ASR-lora", _build_profiles())

        # Set: the profile appears, pointing at the given directory.
        with mock.patch.dict(os.environ, {_LORA_MODEL_DIR_ENV: "/models/lora-merged"}):
            profiles = _build_profiles()
            self.assertIn("Qwen3-ASR-lora", profiles)
            self.assertEqual(profiles["Qwen3-ASR-lora"].hf_id, "/models/lora-merged")


class TestNormalizeSegmentText(unittest.TestCase):
    def _seg(self, text):
        return {"text": text, "start": 0.0, "end": 1.0}

    def test_default_is_noop(self):
        seg = self._seg("愛你 简体")
        out = normalize_segment_text(seg, DEFAULT_NORMALIZATION)
        self.assertEqual(out["text"], "愛你 简体")

    def test_opencc_converts_simplified(self):
        # With OpenCC on, a simplified string should be rewritten (traditional/HK forms).
        seg = self._seg("简体")
        out = normalize_segment_text(seg, TextNormalization(opencc_config="s2t_c.json"))
        self.assertNotEqual(out["text"], "简体")

    def test_chars_hk_matches_standardize(self):
        text = "你哋"
        seg = self._seg(text)
        out = normalize_segment_text(seg, TextNormalization(chars_hk=True))
        self.assertEqual(out["text"], standardize_chars_hk(text))

    def test_does_not_mutate_input(self):
        seg = self._seg("简体")
        normalize_segment_text(seg, TextNormalization(opencc_config="s2t_c.json"))
        self.assertEqual(seg["text"], "简体")


class TestPunctuationConfig(unittest.TestCase):
    def test_sentence_spans_default(self):
        pc = PunctuationConfig()
        text = "你好，世界。再見"
        spans = pc.sentence_spans(text)
        self.assertEqual([text[a:b] for a, b in spans], ["你好", "世界", "再見"])

    def test_sentence_spans_custom_split_chars(self):
        pc = PunctuationConfig(split_chars=("|",))
        self.assertEqual(pc.sentence_spans("a|b"), [(0, 1), (2, 3)])

    def test_no_split_chars_yields_single_span(self):
        pc = PunctuationConfig()
        self.assertEqual(pc.sentence_spans("abc"), [(0, 3)])


class TestSpotCheck(unittest.TestCase):
    def test_default_weights_empty(self):
        sc = SpotCheck(("喇", "啦"))
        self.assertEqual(dict(sc.weights), {})
        self.assertEqual(sc.candidates, ("喇", "啦"))


if __name__ == "__main__":
    unittest.main()

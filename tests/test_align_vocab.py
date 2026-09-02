"""Tests for align_vocab: giving the align model a token for characters it lacks.

The property that matters throughout is that this is a *token* substitution and not a text
edit -- the transcript's own character has to survive into the subtitle, the way punctuation
does while being tokenised as blank.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cantocaptions_ai.pipeline.align_vocab import (
    KIND_HOMOPHONE,
    KIND_NEAR,
    KIND_OVERRIDE,
    KIND_VARIANT,
    LEVEL_HOMOPHONE,
    LEVEL_NEAR,
    LEVEL_OFF,
    LEVEL_VARIANT,
    NOTE_NO_TOKEN,
    Substitution,
    VocabRepair,
    _repairable,
    filter_spotchecks,
    load_substitution_overrides,
    reading_of,
    substitution_notes,
)
from cantocaptions_ai.utils.schema import add_note, merge_segments

PAD = "[PAD]"


def dictionary(*chars) -> dict:
    """A stand-in align vocabulary; ids are arbitrary but distinct."""
    out = {PAD: 0}
    for i, char in enumerate(chars, 1):
        out[char] = i
    return out


class TestRepairable(unittest.TestCase):
    """Only letters are repaired; see the module docstring on digits and punctuation."""

    def test_a_chinese_character_is_repairable(self):
        for char in "駒嘿摷𥄫":
            self.assertTrue(_repairable(char), char)

    def test_a_digit_is_not(self):
        # 8 has a reading (baat3) and would substitute happily, but a digit in a transcript
        # may be spoken in Cantonese, in English, or digit by digit.
        for char in "0389３":
            self.assertFalse(_repairable(char), char)

    def test_punctuation_and_ascii_are_not(self):
        for char in "，。？.,!z、~":
            self.assertFalse(_repairable(char), char)


class TestVariantFold(unittest.TestCase):
    """A Simplified form is not an acoustic problem -- it is the wrong character set."""

    def test_a_simplified_character_folds_to_the_traditional_one(self):
        repair = VocabRepair(dictionary("輝"), LEVEL_VARIANT)   # 輝
        got = repair.resolve("辉")                              # 辉
        self.assertIsNotNone(got)
        self.assertEqual((got.replacement, got.kind), ("輝", KIND_VARIANT))

    def test_the_fold_is_skipped_when_the_traditional_form_is_absent_too(self):
        repair = VocabRepair(dictionary("區"), LEVEL_VARIANT)   # 區 only
        self.assertIsNone(repair.resolve("辉"))

    def test_variant_level_stops_before_homophones(self):
        repair = VocabRepair(dictionary("區"), LEVEL_VARIANT)
        self.assertIsNone(repair.resolve("駒"), "駒 should need the homophone tier")


class TestHomophones(unittest.TestCase):
    """駒 (keoi1) is absent, 區 (keoi1) is present, and they are the same syllable."""

    def test_a_homophone_in_the_vocabulary_is_used(self):
        repair = VocabRepair(dictionary("區"), LEVEL_HOMOPHONE)
        got = repair.resolve("駒")
        self.assertEqual((got.replacement, got.kind, got.reading),
                         ("區", KIND_HOMOPHONE, "keoi1"))

    def test_a_character_reads_through_its_variant_when_it_has_no_reading_itself(self):
        # 撺 is unknown to the reading data; 攛 (cyun1) is not, and 村 is cyun1.
        self.assertIsNone(reading_of("撺"))
        repair = VocabRepair(dictionary("村"), LEVEL_HOMOPHONE)
        got = repair.resolve("撺")
        self.assertIsNotNone(got, "the variant's reading should carry the lookup")
        self.assertEqual(got.replacement, "村")

    def test_a_different_tone_is_not_a_homophone(self):
        # 悍 is hon5; 漢 is hon3. Same syllable, different tone: needs the 'near' tier.
        repair = VocabRepair(dictionary("漢"), LEVEL_HOMOPHONE)
        self.assertIsNone(repair.resolve("悍"))

    def test_the_near_tier_accepts_it(self):
        repair = VocabRepair(dictionary("漢"), LEVEL_NEAR)
        got = repair.resolve("悍")
        self.assertEqual((got.replacement, got.kind), ("漢", KIND_NEAR))

    def test_a_character_never_substitutes_for_itself(self):
        repair = VocabRepair(dictionary("區"), LEVEL_NEAR)
        got = repair.resolve("區")
        self.assertIsNone(got, "it is already in the vocabulary")

    def test_the_choice_is_deterministic(self):
        vocab = dictionary("區", "驅", "俱", "拘")   # all keoi1
        first = VocabRepair(dict(vocab), LEVEL_HOMOPHONE).resolve("駒")
        second = VocabRepair(dict(vocab), LEVEL_HOMOPHONE).resolve("駒")
        self.assertEqual(first.replacement, second.replacement)


class TestLevels(unittest.TestCase):
    def test_off_resolves_nothing(self):
        repair = VocabRepair(dictionary("區", "輝"), LEVEL_OFF)
        self.assertIsNone(repair.resolve("駒"))
        self.assertIsNone(repair.resolve("辉"))
        self.assertEqual(repair.augment(["駒辉"]).substitutions, {})

    def test_an_unknown_level_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            VocabRepair(dictionary("區"), "aggressive")


class TestOverrides(unittest.TestCase):
    """A hand-picked substitution beats every automatic tier, both ways."""

    def test_an_override_wins_over_the_automatic_choice(self):
        repair = VocabRepair(dictionary("區", "驅"), LEVEL_HOMOPHONE,
                             {"駒": "驅"})
        got = repair.resolve("駒")
        self.assertEqual((got.replacement, got.kind), ("驅", KIND_OVERRIDE))

    def test_an_empty_override_leaves_the_character_alone(self):
        repair = VocabRepair(dictionary("區"), LEVEL_NEAR, {"駒": ""})
        self.assertIsNone(repair.resolve("駒"),
                          "an empty value must switch off one character, not fall through")

    def test_an_override_to_a_character_the_model_lacks_substitutes_nothing(self):
        repair = VocabRepair(dictionary("區"), LEVEL_HOMOPHONE, {"駒": "驅"})
        self.assertIsNone(repair.resolve("駒"))

    def test_overrides_load_from_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "subs.toml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('[substitutions]\n"駒" = "區"\n"摾" = ""\n')
            self.assertEqual(load_substitution_overrides(path),
                             {"駒": "區", "摾": ""})

    def test_a_multi_character_substitution_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "subs.toml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('[substitutions]\n"駒" = "區驅"\n')
            with self.assertRaises(ValueError):
                load_substitution_overrides(path)


class TestAugment(unittest.TestCase):
    """augment edits the dictionary in place; that is the whole delivery mechanism."""

    def test_the_dictionary_gains_the_replacement_token_id(self):
        vocab = dictionary("區")
        VocabRepair(vocab, LEVEL_HOMOPHONE).augment(["駒仔"])
        self.assertEqual(vocab["駒"], vocab["區"])

    def test_a_second_call_changes_nothing(self):
        vocab = dictionary("區")
        repair = VocabRepair(vocab, LEVEL_HOMOPHONE)
        first = repair.augment(["駒"])
        again = repair.augment(["駒"])
        self.assertEqual(len(first.substitutions), 1)
        self.assertEqual(again.substitutions, {},
                         "the coarse search and alignment must not resolve it twice")
        self.assertEqual(vocab["駒"], vocab["區"])

    def test_an_unresolvable_character_is_reported_with_its_count(self):
        vocab = dictionary("區")
        report = VocabRepair(vocab, LEVEL_OFF).augment([])
        self.assertEqual(report.unresolved, {})
        report = VocabRepair(vocab, LEVEL_HOMOPHONE).augment(["仔仔"])
        self.assertEqual(report.unresolved.get("仔"), 2)
        self.assertNotIn("仔", vocab)

    def test_digits_and_punctuation_are_left_out_of_the_report_entirely(self):
        report = VocabRepair(dictionary("區"), LEVEL_NEAR).augment(["8，."])
        self.assertEqual(report.occurrences, {})

    def test_occurrences_count_every_use_not_every_character(self):
        report = VocabRepair(dictionary("區"), LEVEL_HOMOPHONE).augment(
            ["駒駒", "駒"])
        self.assertEqual(report.occurrences["駒"], 3)
        self.assertEqual(len(report.substitutions), 1)


class TestSubstitutionIsNotATextEdit(unittest.TestCase):
    """The transcript's own character must reach the subtitle; only the token differs."""

    def test_preprocess_keeps_the_original_character(self):
        from cantocaptions_ai.pipeline.alignment import _preprocess_segment

        vocab = dictionary("區", "仔")
        VocabRepair(vocab, LEVEL_HOMOPHONE).augment(["駒仔"])
        data = _preprocess_segment("駒仔", "zh", vocab)
        self.assertEqual(data["clean_char"], ["駒", "仔"])

    def test_line_tokens_uses_the_replacements_token(self):
        from cantocaptions_ai.pipeline.realign import line_tokens

        vocab = dictionary("區", "仔")
        VocabRepair(vocab, LEVEL_HOMOPHONE).augment(["駒仔"])
        self.assertEqual(line_tokens("駒仔", "zh", vocab, vocab[PAD]),
                         [vocab["區"], vocab["仔"]])

    def test_without_the_repair_the_character_is_dropped_outright(self):
        from cantocaptions_ai.pipeline.alignment import _preprocess_segment

        vocab = dictionary("區", "仔")
        data = _preprocess_segment("駒仔", "zh", vocab)
        self.assertEqual(data["clean_char"], ["仔"],
                         "this is the failure the feature exists to fix")


class TestNotes(unittest.TestCase):
    def test_a_substituted_character_produces_a_note(self):
        subs = {"駒": Substitution("駒", "區", KIND_HOMOPHONE, "keoi1")}
        self.assertEqual(substitution_notes("家駒，", subs),
                         ["homophone:駒→區"])

    def test_an_unresolved_character_produces_one_too(self):
        self.assertEqual(substitution_notes("摾摾", {}, {"摾"}),
                         [f"{NOTE_NO_TOKEN}:摾"])

    def test_a_repeated_character_is_noted_once(self):
        subs = {"駒": Substitution("駒", "區", KIND_HOMOPHONE)}
        self.assertEqual(len(substitution_notes("駒駒駒", subs)), 1)

    def test_an_untouched_line_has_no_notes(self):
        subs = {"駒": Substitution("駒", "區", KIND_HOMOPHONE)}
        self.assertEqual(substitution_notes("你好", subs, {"摾"}), [])

    def test_add_note_ignores_a_duplicate(self):
        segment = {"start": 0.0, "end": 1.0, "text": "a"}
        add_note(segment, "variant:辉→輝")
        add_note(segment, "variant:辉→輝")
        self.assertEqual(segment["notes"], ["variant:辉→輝"])

    def test_merging_two_cues_keeps_both_sides_notes(self):
        left = {"start": 0.0, "end": 1.0, "text": "a", "words": [], "chars": None,
                "notes": ["variant:辉→輝"]}
        right = {"start": 1.0, "end": 2.0, "text": "b", "words": [], "chars": None,
                 "notes": ["homophone:駒→區", "variant:辉→輝"]}
        merged = merge_segments(left, right)
        self.assertEqual(merged["notes"],
                         ["variant:辉→輝", "homophone:駒→區"])

    def test_merging_unannotated_cues_adds_no_key(self):
        left = {"start": 0.0, "end": 1.0, "text": "a", "words": [], "chars": None}
        right = {"start": 1.0, "end": 2.0, "text": "b", "words": [], "chars": None}
        self.assertNotIn("notes", merge_segments(left, right))


class TestNotesSrt(unittest.TestCase):
    def test_only_annotated_cues_are_written(self):
        from cantocaptions_ai.utils.debug import write_segment_notes

        segments = [
            {"start": 0.0, "end": 1.0, "text": "你好"},
            {"start": 1.0, "end": 2.0, "text": "家駒",
             "notes": ["homophone:駒→區"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(write_segment_notes("film.wav", segments, tmp), 1)
            path = os.path.join(tmp, "film", "notes", "notes.srt")
            with open(path, encoding="utf-8") as fh:
                body = fh.read()
        self.assertIn("[homophone:駒→區] 家駒", body)
        self.assertNotIn("你好", body)

    def test_nothing_is_written_when_nothing_is_annotated(self):
        from cantocaptions_ai.utils.debug import write_segment_notes

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                write_segment_notes("film.wav", [{"start": 0.0, "end": 1.0, "text": "a"}], tmp),
                0)
            self.assertFalse(os.path.exists(os.path.join(tmp, "film", "notes")))

    def test_an_inverted_cue_still_yields_a_valid_file(self):
        from cantocaptions_ai.utils.debug import write_labelled_srt

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.srt")
            write_labelled_srt(path, [(2.0, 1.0, ["x"], "text")])
            with open(path, encoding="utf-8") as fh:
                stamps = fh.read().splitlines()[1]
        start, end = stamps.split(" --> ")
        self.assertLessEqual(start, end)


class TestSpotcheckGuard(unittest.TestCase):
    """A substituted character cannot be asked which particle the audio supports.

    Its token is some homophone's, so its score says nothing about the candidate -- and two
    candidates substituting to the same token would score identically, leaving a candidate
    weight to decide the rewrite on its own.
    """

    def setUp(self):
        from cantocaptions_ai.cantonese.text import SpotCheck
        self.checks = {
            "咁": SpotCheck(("咁", "噉"), weights={"噉": 0.8}),
            "喇": SpotCheck(("喇", "啦", "囉")),
        }

    def test_nothing_substituted_returns_the_table_untouched(self):
        self.assertIs(filter_spotchecks(self.checks, set()), self.checks)

    def test_a_substituted_candidate_is_removed(self):
        got = filter_spotchecks(self.checks, {"囉"})
        self.assertEqual(got["喇"].candidates, ("喇", "啦"))
        self.assertEqual(got["咁"].candidates, ("咁", "噉"), "untouched checks survive")

    def test_a_check_left_with_one_candidate_is_dropped(self):
        got = filter_spotchecks(self.checks, {"噉"})
        self.assertNotIn("咁", got, "one candidate left means the weight decides alone")
        self.assertIn("喇", got)

    def test_the_shipped_profile_is_never_reduced(self):
        # Checked separately against the real tokenizer: every character in Qwen3-ASR's
        # spot-check table (啊吖呀咐喇啦囉咋啫咁噉) is natively in
        # alvanlii/wav2vec2-BERT-cantonese's vocabulary, so nothing there is ever
        # substituted and the guard is inert in practice. This asserts the half a unit test
        # can: given a vocabulary holding them, augment leaves them alone and the table
        # comes back untouched.
        from cantocaptions_ai.pipeline.model_profiles import get_model_profile

        checks = get_model_profile("Qwen3-ASR").spotchecks
        chars = set(checks) | {c for check in checks.values() for c in check.candidates}
        repair = VocabRepair(dictionary(*chars), LEVEL_NEAR)
        repair.augment(["".join(sorted(chars))])
        self.assertEqual(repair.substitutions, {})
        self.assertIs(filter_spotchecks(checks, repair.substitutions), checks)


if __name__ == "__main__":
    unittest.main()

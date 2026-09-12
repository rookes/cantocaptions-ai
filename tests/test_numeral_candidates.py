"""Tests for numeral_candidates: Arabic-digit -> Chinese-numeral reading
candidates, the reverse direction of numbers.py. Pure logic, no model needed.
"""

import unittest

from cantocaptions_ai.cantonese.numeral_candidates import has_digits, numeral_readings


class TestHasDigits(unittest.TestCase):
    def test_plain_digit_detected(self):
        self.assertTrue(has_digits("1891年"))
        self.assertTrue(has_digits("25個"))

    def test_no_digits(self):
        self.assertFalse(has_digits("嫲嫲曾經講過"))

    def test_jyutping_gloss_is_not_a_digit(self):
        # Corpus convention: a parenthetical pronunciation gloss, not a quantity.
        self.assertFalse(has_digits("踩(jaai2)"))
        self.assertFalse(has_digits("m4(噉)"))
        self.assertFalse(has_digits("teng1香"))
        self.assertFalse(has_digits("嗰陣(go2 zan6)"))

    def test_real_digit_alongside_jyutping_gloss_still_detected(self):
        self.assertTrue(has_digits("踩(jaai2)咗18歲"))


class TestNumeralReadings(unittest.TestCase):
    def test_no_digits_returns_text_unchanged_as_only_variant(self):
        self.assertEqual(numeral_readings("嫲嫲曾經講過"), ["嫲嫲曾經講過"])

    def test_combined_and_digitwise_readings(self):
        variants = numeral_readings("1891年")
        self.assertEqual(variants, ["一千八百九十一年", "一八九一年"])

    def test_small_number_combined_and_digitwise_coincide_dedupes(self):
        # Single-digit numbers read the same either way -- only one variant.
        self.assertEqual(numeral_readings("2個"), ["二個"])

    def test_loeng_substitution_before_hundred_thousand_ten_thousand(self):
        self.assertEqual(numeral_readings("200萬")[0], "兩百萬")
        self.assertEqual(numeral_readings("2000蚊")[0], "兩千蚊")
        self.assertEqual(numeral_readings("20000人")[0], "兩萬人")
        # Never on 二十 (twenty) or a bare 二.
        self.assertEqual(numeral_readings("20歲")[0], "二十歲")
        self.assertEqual(numeral_readings("2蚊")[0], "二蚊")

    def test_leading_zero_is_always_digitwise(self):
        # A leading zero is never a positional quantity -- only one variant.
        self.assertEqual(numeral_readings("007"), ["零零七"])
        self.assertEqual(numeral_readings("09年"), ["零九年"])
        self.assertEqual(numeral_readings("012"), ["零一二"])

    def test_decimal_fraction_is_digit_by_digit(self):
        self.assertEqual(numeral_readings("1.5公斤"), ["一點五公斤"])
        self.assertEqual(numeral_readings("0.01公分"), ["零點零一公分"])
        self.assertEqual(numeral_readings("14.2%"), ["十四點二%", "一四點二%"])

    def test_thousands_separator_joined(self):
        # Guarded to exactly 3 trailing digits -- the fullwidth comma is this
        # corpus's ordinary sentence separator everywhere else.
        variants = numeral_readings("204，000戶")
        self.assertIn("二十萬四千戶", variants)

    def test_thousands_separator_not_joined_when_not_exactly_three_digits(self):
        # Only a genuine 3-digit thousands group joins; "20" is 2 digits, so the
        # comma is an ordinary sentence separator and the two numbers stay
        # separate (not absorbed into one "10,020"-shaped reading).
        variants = numeral_readings("有100，20蚊")
        self.assertIn("，", variants[0])
        self.assertIn("一百", variants[0])
        self.assertIn("二十", variants[0])

    def test_fullwidth_digits_normalized(self):
        self.assertEqual(numeral_readings("４"), numeral_readings("4"))

    def test_jyutping_gloss_never_misread_as_a_digit(self):
        self.assertEqual(numeral_readings("踩(jaai2)"), ["踩(jaai2)"])
        self.assertEqual(numeral_readings("m4(噉)"), ["m4(噉)"])

    def test_jyutping_gloss_alongside_a_real_number_only_converts_the_number(self):
        variants = numeral_readings("踩(jaai2)咗18歲")
        for v in variants:
            self.assertIn("(jaai2)", v)
        self.assertEqual(variants, ["踩(jaai2)咗十八歲", "踩(jaai2)咗一八歲"])

    def test_max_variants_caps_the_result(self):
        self.assertEqual(len(numeral_readings("1891年", max_variants=1)), 1)
        self.assertLessEqual(len(numeral_readings("1891年", max_variants=2)), 2)

    def test_multiple_numbers_rendered_in_the_same_style_per_variant(self):
        # Deliberate simplification: every number in a line renders in the SAME
        # style within one variant, not mixed per-number.
        variants = numeral_readings("1個2蚊，3個5蚊")
        self.assertIn("一個二蚊，三個五蚊", variants)
        self.assertIn("一個二蚊，三個五蚊", [v for v in variants])

    def test_known_positional_spellings_are_reachable(self):
        # Regression net pinning a few hand-checked positional spellings,
        # including the 兩-substitution and multi-group cases.
        cases = {
            "104": "一百零四",
            "845": "八百四十五",
            "1974": "一九七四",
            "12674": "一萬兩千六百七十四",
        }
        for digits, gold in cases.items():
            variants = numeral_readings(digits)
            self.assertIn(gold, variants, f"{digits} -> {variants}, expected {gold!r}")


if __name__ == "__main__":
    unittest.main()

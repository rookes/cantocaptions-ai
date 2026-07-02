"""Tests for the Cantonese subtitle text cleaner.

Covers:
- rules.py TOML loader (ordering, error reporting, built-in rule files)
- questions.py segment splitting and question detection
- numbers.py Chinese numeral conversion edge cases
- SubtitleCleaner end-to-end cleaning (assertions ported from the legacy
  canto_subtitle_cleaner test suite)
- _merge_and_write integration: segment dropping + pre-cleaning debug SRT
"""

import os
import tempfile
import unittest
from pathlib import Path

from cantocaptions_ai.cantonese.cleaner import SubtitleCleaner
from cantocaptions_ai.cantonese.numbers import convert_chinese_numbers
from cantocaptions_ai.cantonese.questions import is_question, split_segments
from cantocaptions_ai.cantonese.rules import BUILTIN_RULES_DIR, load_ruleset
from cantocaptions_ai.cantonese.text import is_removable


def _cleaner() -> SubtitleCleaner:
    # Legacy expectations assume a 21-char two-line layout.
    return SubtitleCleaner(line_max_length=21, max_line_count=2)


# ---------------------------------------------------------------------------
# rules.py — TOML rule engine
# ---------------------------------------------------------------------------

class TestRuleLoader(unittest.TestCase):

    def test_all_builtin_rule_files_load(self):
        toml_files = sorted(BUILTIN_RULES_DIR.glob("*.toml"))
        self.assertGreater(len(toml_files), 9)  # 9 rule files + pipeline.toml
        for path in toml_files:
            if path.name == "pipeline.toml":
                continue
            rules = load_ruleset(path)
            self.assertGreater(len(rules), 0, f"{path.name} loaded no rules")

    def test_order_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.toml"
            path.write_text(
                "[[rules]]\npattern = 'a'\nreplace = 'b'\n"
                "[[rules]]\npattern = 'b'\nreplace = 'c'\n",
                encoding="utf-8",
            )
            rules = load_ruleset(path)
            self.assertEqual([r.pattern.pattern for r in rules], ["a", "b"])
            # Sequential application: a -> b, then that b also -> c
            from cantocaptions_ai.cantonese.rules import apply_ruleset
            self.assertEqual(apply_ruleset("ab", rules), "cc")

    def test_bad_regex_names_file_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text(
                "[[rules]]\npattern = 'ok'\nreplace = ''\n"
                "[[rules]]\npattern = '(unclosed'\nreplace = ''\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_ruleset(path)
            self.assertIn("bad.toml", str(ctx.exception))
            self.assertIn("#2", str(ctx.exception))

    def test_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text("[[rules]]\npattern = 'a'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ruleset(path)


class TestCleanerConstruction(unittest.TestCase):

    def test_missing_manifest_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                SubtitleCleaner(rules_dir=tmp)

    def test_unknown_builtin_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pipeline.toml").write_text(
                "[[steps]]\ntype = 'builtin'\nname = 'nope'\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                SubtitleCleaner(rules_dir=tmp)
            self.assertIn("nope", str(ctx.exception))

    def test_override_rules_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pipeline.toml").write_text(
                "[[steps]]\ntype = 'rules'\nfile = 'only.toml'\n", encoding="utf-8"
            )
            (Path(tmp) / "only.toml").write_text(
                "[[rules]]\npattern = '幾好嘛'\nreplace = '你好嘛'\n", encoding="utf-8"
            )
            cleaner = SubtitleCleaner(rules_dir=tmp)
            self.assertEqual(cleaner.clean("幾好嘛"), "你好嘛")
            # Built-in steps are not applied with an override manifest
            self.assertEqual(cleaner.clean("吓？？"), "吓？？")


# ---------------------------------------------------------------------------
# questions.py
# ---------------------------------------------------------------------------

class TestParseFunctions(unittest.TestCase):

    def test_segments(self):
        self.assertEqual(split_segments(""), [])
        self.assertEqual(split_segments("！"), ["！"])
        self.assertEqual(split_segments("噉你係咪好開心？我都係咁諗！"), ["噉你係咪好開心？", "我都係咁諗！"])
        self.assertEqual(split_segments("為為為為為為為為揾揾揾揾揾揾揾"), ["為為為為為為為為揾揾揾揾揾揾揾"])

    def test_is_question(self):
        self.assertFalse(is_question("噉你係咪好開心？我都係咁諗！"))
        self.assertTrue(is_question("噉你係咪好開心？"))
        self.assertTrue(is_question("唔使，但我可唔可以跟埋嚟呀？"))
        self.assertTrue(is_question("我會唔會再見到佢嘎？"))
        self.assertFalse(is_question("乜你唔覺得我傻啊？"))

    def test_short_乜_segment_does_not_crash(self):
        # Legacy code raised IndexError on a 1-char question segment starting 乜
        cleaner = _cleaner()
        cleaner.clean("乜？")


# ---------------------------------------------------------------------------
# numbers.py
# ---------------------------------------------------------------------------

class TestChineseNumbers(unittest.TestCase):

    def test_round_numbers_not_converted(self):
        self.assertEqual(convert_chinese_numbers("一百"), "一百")
        self.assertEqual(convert_chinese_numbers("一千"), "一千")
        self.assertEqual(convert_chinese_numbers("一萬"), "一萬")

    def test_uncertain_words_not_converted(self):
        self.assertEqual(convert_chinese_numbers("幾十個"), "幾十個")
        self.assertEqual(convert_chinese_numbers("十幾個"), "十幾個")

    def test_digit_strings_not_converted(self):
        self.assertEqual(convert_chinese_numbers("一零一"), "一零一")
        self.assertEqual(convert_chinese_numbers("一二三"), "一二三")

    def test_zero_placeholder(self):
        self.assertEqual(convert_chinese_numbers("五千零一十"), "5010")
        self.assertEqual(convert_chinese_numbers("一百零八歲"), "108歲")

    def test_small_numbers_not_converted(self):
        self.assertEqual(convert_chinese_numbers("三個"), "三個")
        self.assertEqual(convert_chinese_numbers("十"), "十")

    def test_dates(self):
        self.assertEqual(convert_chinese_numbers("十一月二十三號"), "11月23號")
        self.assertEqual(convert_chinese_numbers("一月二十三號"), "1月23號")

    def test_years_digit_by_digit(self):
        self.assertEqual(convert_chinese_numbers("一二三四年"), "1234年")

    def test_compound(self):
        self.assertEqual(convert_chinese_numbers("四萬五千零一十蚊"), "45010蚊")


# ---------------------------------------------------------------------------
# SubtitleCleaner end-to-end (ported from legacy test_canto_subtitle_cleaner.py)
# ---------------------------------------------------------------------------

class TestCleanSubtitle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cleaner = _cleaner()

    def clean(self, text: str) -> str:
        return self.cleaner.clean(text)

    def test_linebreak(self):
        self.assertEqual(self.clean("雖然話大家係親戚,不過,我哋其實只係遠房親戚,而佢哋就負責輪流照顧我。"),
                         "雖然話大家係親戚，不過\n我哋其實只係遠房親戚，而佢哋就負責輪流照顧我")

    def test_linebreak_skipped_for_single_line_output(self):
        single = SubtitleCleaner(line_max_length=21, max_line_count=1)
        self.assertNotIn("\n", single.clean("雖然話大家係親戚,不過,我哋其實只係遠房親戚,而佢哋就負責輪流照顧我。"))

    def test_repeating_phrases(self):
        self.assertEqual(self.clean("快啲啦快啲啦"), "快啲啦…")
        self.assertEqual(self.clean("快啲啦，快啲啦"), "快啲啦…")

        self.assertEqual(self.clean("唔好，唔好食"), "唔好，唔好食")
        self.assertEqual(self.clean("唔好，唔好，食"), "唔好…食")

        self.assertEqual(self.clean("快啲，快啲走"), "快啲，快啲走")
        self.assertEqual(self.clean("快啲，快啲，快啲走"), "快啲…快啲走")

        self.assertEqual(self.clean("大佬大佬大佬"), "大佬…")
        self.assertEqual(self.clean("大佬大佬"), "大佬…")

        self.assertEqual(self.clean("喂喂喂"), "喂…")
        self.assertEqual(self.clean("喂喂"), "喂喂")
        self.assertEqual(self.clean("喂喂喂？"), "喂…？")
        self.assertEqual(self.clean("喂!喂！喂？"), "喂…？")

        self.assertEqual(self.clean("停啊，停啊，"), "停啊…")
        self.assertEqual(self.clean("靜㗎，靜㗎，"), "靜㗎…")

        self.assertEqual(self.clean("佢…佢，佢好鍾意"), "佢…佢好鍾意")

    def test_repeated_characters_into_word(self):
        self.assertEqual(self.clean("你你你佢"), "你…佢")
        self.assertEqual(self.clean("你，你，你佢"), "你…你佢")
        self.assertEqual(self.clean("同你你你佢"), "同你…佢")
        self.assertEqual(self.clean("同你，你，你佢"), "同你…你佢")

    def test_乜_questions(self):
        # Third case differs from the (stale, failing) legacy test file: 乜-initial
        # question segments get 啊？→呀？, matching actual legacy behavior.
        self.assertEqual(self.clean("乜你唔覺得我傻啊？"), "乜你唔覺得我傻呀？")
        self.assertEqual(self.clean("乜嘢係代表我傻啊？"), "乜嘢係代表我傻啊？")
        self.assertEqual(self.clean("乜都得，你係咪覺得我傻啊？"), "乜都得，你係咪覺得我傻呀？")

    def test_english_spacing(self):
        # 咁 stays 咁 (the 咁→噉 rules are commented out in legacy); the stale legacy
        # test expected 噉.
        self.assertEqual(self.clean("同 埋有陣時學,大家覺得好似係\nall or nothing,"),
                         "同埋有陣時學，大家覺得好似係\nall or nothing")
        self.assertEqual(self.clean("咁你就要走去Adobe\nIllustrator。"), "咁你就要走去Adobe\nIllustrator")

    def test_number_retention(self):
        self.assertEqual(self.clean("呢個字體可以幫你完全,唔可以完全嘅,99.7％嘅時候,揀中呢一個"),
                         "呢個字體可以幫你完全\n唔可以完全嘅，99.7%嘅時候，揀中呢一個")

    def test_chinese_numbers(self):
        self.assertEqual(self.clean("一二三四五六七八九十"), "一二三四五六七八九十")
        self.assertEqual(self.clean("十一月二十三號"), "11月23號")
        self.assertEqual(self.clean("一月二十三號"), "1月23號")
        self.assertEqual(self.clean("你二十三歲？一二三四五六七八九十"), "你23歲？一二三四五六七八九十")
        self.assertEqual(self.clean("一二三四年果陣佢計咗數，「一二三」"), "1234年嗰陣佢計咗數，「一二三」")
        self.assertEqual(self.clean("加埋一齊係四萬五千零一十蚊"), "加埋一齊係45010蚊")
        self.assertEqual(self.clean("會活到一百零八歲"), "會活到108歲")

    def test_㗎嘛(self):
        # Actual legacy behavior: 㗎嘛 -> 𠺢嘛 (the stale legacy test expected 㗎咩).
        self.assertEqual(self.clean("你覺得我難睇㗎嘛？"), "你覺得我難睇𠺢嘛？")

    def test_interjections(self):
        self.assertEqual(self.clean("吓？？"), "")
        self.assertEqual(self.clean("吼"), "")
        self.assertEqual(self.clean("吓吼"), "吓吼")

    def test_removed_text_is_removable(self):
        self.assertTrue(is_removable(self.clean("吓？？")))
        self.assertTrue(is_removable(""))
        self.assertFalse(is_removable("好呀"))


# ---------------------------------------------------------------------------
# _merge_and_write integration
# ---------------------------------------------------------------------------

class TestMergeAndWrite(unittest.TestCase):

    def test_cleaning_drops_noise_and_writes_precleaning_srt(self):
        from cantocaptions_ai.pipeline.transcribe import _merge_and_write

        written = {}

        def fake_writer(result, audio_path, options):
            written["segments"] = result["segments"]

        def _seg(start, end, text):
            return {"start": start, "end": end, "text": text, "words": []}

        items = [{
            "audio_path": os.path.join("some", "dir", "episode.wav"),
            "result": {
                "language": "yue",
                "segments": [
                    _seg(0.0, 1.0, "吓？？"),           # noise -> dropped
                    _seg(5.0, 6.0, "十一月二十三號"),    # cleaned
                ],
            },
        }]

        with tempfile.TemporaryDirectory() as tmp:
            _merge_and_write(
                items, fake_writer, "yue",
                align_merge_distance=0.12, align_padding=0.04, writer_args={},
                cleaner=_cleaner(), debug_dir=tmp,
            )

            srt_path = Path(tmp) / "episode" / "pre_cleaning" / "episode.srt"
            self.assertTrue(srt_path.is_file())
            srt_content = srt_path.read_text(encoding="utf-8")
            self.assertIn("十一月二十三號", srt_content)  # pre-cleaning text

            json_path = Path(tmp) / "episode" / "pre_cleaning" / "result.json"
            self.assertTrue(json_path.is_file())

        self.assertEqual(len(written["segments"]), 1)
        self.assertEqual(written["segments"][0]["text"], "11月23號")

    def test_no_cleaner_no_debug_is_passthrough(self):
        from cantocaptions_ai.pipeline.transcribe import _merge_and_write

        written = {}

        def fake_writer(result, audio_path, options):
            written["segments"] = result["segments"]

        items = [{
            "audio_path": "episode.wav",
            "result": {
                "language": "yue",
                "segments": [{"start": 0.0, "end": 1.0, "text": "吓？？", "words": []}],
            },
        }]
        _merge_and_write(items, fake_writer, "yue", 0.12, 0.04, {})
        self.assertEqual(written["segments"][0]["text"], "吓？？")


if __name__ == "__main__":
    unittest.main()

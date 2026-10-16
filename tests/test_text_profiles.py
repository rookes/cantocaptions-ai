"""The language-dependent text conventions (text_profiles.py) and the seams that use them.

Cantonese behaviour is pinned throughout the suite; these cover the other branch -- a
space-separated script -- at each point the pipeline used to assume CJK.
"""
import pytest

from cantocaptions_ai.cleaning import LAYOUTS, SubtitleCleaner, linebreak_step
from cantocaptions_ai.languages.yue.builtins import CantoneseCleaner
from cantocaptions_ai.pipeline.realign import realign_punctuation
from cantocaptions_ai.text_profiles import (
    CJK_PUNCTUATION,
    CJK_SCRIPT,
    LATIN_PUNCTUATION,
    LATIN_WORD,
    SPACED_SCRIPT,
    is_mergeable,
    script_for_language,
    word_wrap,
)
from cantocaptions_ai.utils.schema import merge_segments


def test_scripts_follow_the_language():
    assert script_for_language("yue") is CJK_SCRIPT
    assert script_for_language("ja") is CJK_SCRIPT
    assert script_for_language("en") is SPACED_SCRIPT


def test_join():
    assert CJK_SCRIPT.join("你好", "世界") == "你好世界"
    assert SPACED_SCRIPT.join("hello", "world") == "hello world"
    assert SPACED_SCRIPT.join("hello ", "world") == "hello world"
    assert SPACED_SCRIPT.join("", "world") == "world"


def test_merged_segments_join_the_scripts_way():
    a = {"start": 0.0, "end": 1.0, "text": "hi.", "words": []}
    b = {"start": 1.0, "end": 2.0, "text": "how are you", "words": []}
    assert merge_segments(a, b)["text"] == "hi.how are you"          # CJK default
    assert merge_segments(a, b, join=SPACED_SCRIPT.join)["text"] == "hi. how are you"


def test_mergeability_counts_the_separator():
    # 5 + 1 + 5 = 11 characters once joined with a space.
    assert is_mergeable("hello", "world", LATIN_PUNCTUATION, max_chars=11, script=SPACED_SCRIPT)
    assert not is_mergeable("hello", "world", LATIN_PUNCTUATION, max_chars=10, script=SPACED_SCRIPT)
    assert not is_mergeable("done.", "next", LATIN_PUNCTUATION, max_chars=40, script=SPACED_SCRIPT)
    assert is_mergeable("well,", "next", LATIN_PUNCTUATION, max_chars=40, script=SPACED_SCRIPT)


@pytest.mark.parametrize("text,width,expected", [
    ("short line", 20, "short line"),
    ("how are you doing today?", 16, "how are you\ndoing today?"),
    ("unbreakablewordthatislong", 10, "unbreakablewordthatislong"),
    ("already\nbroken line here", 5, "already\nbroken line here"),
])
def test_word_wrap(text, width, expected):
    assert word_wrap(text, width) == expected


def test_line_layout_is_chosen_by_the_script():
    assert set(LAYOUTS) == {"cjk", "word"}
    wrap = linebreak_step(16, 2, layout=SPACED_SCRIPT.layout)
    assert wrap("how are you doing today?") == "how are you\ndoing today?"


def test_cleaner_takes_its_own_builtins_and_noise(tmp_path):
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "pipeline.toml").write_text(
        '[[steps]]\ntype = "builtin"\nname = "shout"\n', encoding="utf-8")
    cleaner = SubtitleCleaner(rules_dir=rules, builtin_steps={"shout": str.upper},
                              noise_tokens=("UM",))
    assert cleaner.clean("hello") == "HELLO"
    assert cleaner.is_noise(cleaner.clean("um")) and not cleaner.is_noise("HELLO")
    # The Cantonese preset keeps the Cantonese defaults.
    assert CantoneseCleaner().is_noise("嗯")


def test_a_space_is_a_realign_pause_only_without_word_spacing():
    assert " " in realign_punctuation(CJK_PUNCTUATION, CJK_SCRIPT).split_chars
    assert " " not in realign_punctuation(LATIN_PUNCTUATION, SPACED_SCRIPT).split_chars
    assert "\n" in realign_punctuation(LATIN_PUNCTUATION, SPACED_SCRIPT).split_chars


def test_sentence_splitting_falls_back_to_punctuation_offline(monkeypatch):
    from cantocaptions_ai.pipeline import alignment

    monkeypatch.setattr(alignment, "_punkt_splitter", lambda lang: None)
    text = "Hello there. How are you?"
    assert alignment._get_sentence_spans(text, "en", LATIN_PUNCTUATION) == [(0, 11), (12, 24), (25, 25)]
    # CJK never consults NLTK at all.
    assert alignment._get_sentence_spans("你好。再見", "yue", CJK_PUNCTUATION) == [(0, 2), (3, 5)]


# --- Code-switched English in a CJK line ----------------------------------------------

def test_cjk_width_counts_half_width_characters_as_half():
    assert CJK_SCRIPT.width("我對今次考試好有信心") == 10
    assert CJK_SCRIPT.width("雖然話，不過…「係」") == len("雖然話，不過…「係」")
    assert CJK_SCRIPT.width("open-minded") == 5.5
    assert CJK_SCRIPT.width("regret後悔") == 5
    assert CJK_SCRIPT.width("45,010蚊") == 4
    assert CJK_SCRIPT.width("café") == 2


def test_spaced_width_counts_a_cjk_character_as_two():
    assert SPACED_SCRIPT.width("hello") == 5
    assert SPACED_SCRIPT.width("hi 你好") == 7


def test_latin_words_include_contractions_hyphens_and_numbers():
    found = [m.group() for m in LATIN_WORD.finditer("you've 好open-minded 3.5次 25個")]
    assert found == ["you've", "open-minded", "3.5", "25"]


def test_english_merges_by_width_not_characters():
    # 22 characters, but 11 Chinese characters wide.
    assert is_mergeable("Considerate，", "thoughtful", CJK_PUNCTUATION, max_chars=18)
    assert not is_mergeable("Considerate，", "thoughtful", CJK_PUNCTUATION, max_chars=10)


@pytest.mark.parametrize("text", [
    "25 common Cantonese vocabulary for intermediates",
    "I get very nervous every time I have a job interview",
    "I'm a little disappointed with his or her performance",
    "My mom is very open-minded and willing to listen to my opinions",
    "I'm truly grateful for all the help you've given me",
    "我真係好鍾意呢個vocabulary，因為佢好有用又好易記",
    "Supercalifragilisticexpialidociousness is a long word indeed",
])
def test_cjk_linebreak_never_splits_an_english_word(text):
    broken = LAYOUTS["cjk"](text, 18)
    assert broken.count("\n") == 1
    for line in broken.split("\n"):
        assert line == line.strip()
    words = set(LATIN_WORD.findall(text))
    for line in broken.split("\n"):
        for word in LATIN_WORD.findall(line):
            assert word in words


def test_cjk_linebreak_balances_english_by_width():
    first, second = LAYOUTS["cjk"]("I get very nervous every time I have a job interview", 18).split("\n")
    assert first == "I get very nervous every"
    assert second == "time I have a job interview"


def test_cjk_linebreak_leaves_a_line_that_fits_by_width():
    # 30 characters, but 15 wide.
    assert LAYOUTS["cjk"]("You misunderstood what I meant", 18) == "You misunderstood what I meant"
    assert LAYOUTS["cjk"]("我對今次考試好有信心呢個exam真係好難", 18) == "我對今次考試好有信心呢個exam真係好難"



def test_cjk_linebreak_prefers_english_punctuation():
    assert LAYOUTS["cjk"]("After waiting for a whole hour, I started getting annoyed", 18) \
        == "After waiting for a whole hour,\nI started getting annoyed"

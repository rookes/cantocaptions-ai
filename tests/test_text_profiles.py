"""The language-dependent text conventions (text_profiles.py) and the seams that use them.

Cantonese behaviour is pinned throughout the suite; these cover the other branch -- a
space-separated script -- at each point the pipeline used to assume CJK.
"""
import pytest

from cantocaptions_ai.cantonese.cleaner import LAYOUTS, SubtitleCleaner, linebreak_step
from cantocaptions_ai.pipeline.realign import realign_punctuation
from cantocaptions_ai.text_profiles import (
    CJK_PUNCTUATION,
    CJK_SCRIPT,
    LATIN_PUNCTUATION,
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
    cleaner = SubtitleCleaner(rules_dir=str(rules), builtin_steps={"shout": str.upper},
                              noise_tokens=("UM",))
    assert cleaner.clean("hello") == "HELLO"
    assert cleaner.is_noise(cleaner.clean("um")) and not cleaner.is_noise("HELLO")
    # The Cantonese defaults are still what a plain SubtitleCleaner gets.
    assert SubtitleCleaner().is_noise("嗯")


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

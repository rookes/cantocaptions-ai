"""Import paths other repos rely on keep working after the language-pack move.

cantocaptions-dataset and mj-asr-worker import from this package directly. The Cantonese
code moved from ``cantonese/`` to ``languages/yue/`` (and the cleaning engine to
``cleaning/``); these are the paths they use, pinned so a later move cannot break them
silently.
"""
import subprocess
import sys


def test_cantonese_text_paths_resolve_to_the_yue_pack():
    from cantocaptions_ai.cantonese.text import TextNormalization, simplified_to_traditional
    from cantocaptions_ai.languages.yue import text

    assert simplified_to_traditional is text.simplified_to_traditional
    assert TextNormalization().opencc_config is None


def test_cantonese_cleaner_keeps_its_cantonese_defaults():
    from cantocaptions_ai.cantonese.cleaner import SubtitleCleaner

    cleaner = SubtitleCleaner(max_line_count=2)
    assert cleaner.clean("十一月二十三號") == "11月23號"
    assert cleaner.is_noise("嗯")


def test_cantonese_rules_builtin_ruleset():
    from cantocaptions_ai.cantonese.rules import BUILTIN_RULES_DIR, apply_ruleset, get_builtin_ruleset

    assert (BUILTIN_RULES_DIR / "chars_hk.toml").is_file()
    assert isinstance(apply_ruleset("你好", get_builtin_ruleset("chars_hk")), str)


def test_every_old_submodule_imports():
    import importlib

    for name in ("acronyms", "cleaner", "linebreak", "numbers", "numeral_candidates",
                 "questions", "rules", "text"):
        importlib.import_module(f"cantocaptions_ai.cantonese.{name}")


def test_select_cantonese_track_is_still_in_utils_audio():
    from cantocaptions_ai.utils.audio import select_cantonese_track

    assert select_cantonese_track([{"tags": {"language": "eng"}}, {"tags": {"language": "yue"}}]) == 1


def test_language_registry_and_model_profiles_import_without_torch():
    """The worker reads these in a cheap preflight check that must not load torch."""
    for module in ("cantocaptions_ai.languages", "cantocaptions_ai.pipeline.model_profiles"):
        out = subprocess.run(
            [sys.executable, "-c", f"import sys, {module}; print('torch' in sys.modules)"],
            capture_output=True, text=True, timeout=120,
        )
        assert out.stdout.strip() == "False", (module, out.stdout, out.stderr)

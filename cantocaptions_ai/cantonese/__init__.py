"""Compatibility shim: the Cantonese code moved to ``cantocaptions_ai.languages.yue``.

Every old import path keeps working -- ``cantocaptions_ai.cantonese.text`` *is*
``cantocaptions_ai.languages.yue.text``, and so on -- because each submodule is registered
under its old name when this package is imported. Two modules were split in the move and
are rebuilt here from their new homes:

* ``cantonese.cleaner``: the generic engine is ``cantocaptions_ai.cleaning``; its
  ``SubtitleCleaner`` here is ``languages.yue.builtins.CantoneseCleaner``, which keeps the
  old Cantonese defaults.
* ``cantonese.rules``: the engine is ``cantocaptions_ai.cleaning.rules``; the packaged rule
  files are ``languages.yue.paths.RULES_DIR``.

New code should import from the new locations.
"""
import sys
import types

from cantocaptions_ai.languages.yue import (
    acronyms,
    linebreak,
    numbers,
    numeral_candidates,
    questions,
    text,
)

for _module in (acronyms, linebreak, numbers, numeral_candidates, questions, text):
    sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[-1]}"] = _module


def _build_rules() -> types.ModuleType:
    from cantocaptions_ai.cleaning import rules as engine
    from cantocaptions_ai.languages.yue.paths import RULES_DIR

    module = types.ModuleType(f"{__name__}.rules", engine.__doc__)
    for name in ("Rule", "RuleSet", "apply_ruleset", "load_ruleset", "load_ruleset_cached"):
        setattr(module, name, getattr(engine, name))
    module.BUILTIN_RULES_DIR = RULES_DIR
    module.get_builtin_ruleset = lambda name: engine.load_ruleset_cached(RULES_DIR / f"{name}.toml")
    return module


def _build_cleaner() -> types.ModuleType:
    from cantocaptions_ai.cleaning.layout import LAYOUTS, linebreak_step
    from cantocaptions_ai.languages.yue.builtins import CANTONESE_BUILTIN_STEPS, CantoneseCleaner

    module = types.ModuleType(f"{__name__}.cleaner", "Moved: see cantocaptions_ai.cleaning.")
    module.SubtitleCleaner = CantoneseCleaner
    module.CANTONESE_BUILTIN_STEPS = CANTONESE_BUILTIN_STEPS
    module.LAYOUTS = LAYOUTS
    module.linebreak_step = linebreak_step
    return module


rules = sys.modules[f"{__name__}.rules"] = _build_rules()
cleaner = sys.modules[f"{__name__}.cleaner"] = _build_cleaner()

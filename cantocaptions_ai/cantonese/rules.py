"""TOML-backed regex rule engine for Cantonese subtitle cleaning.

Rule files live in ``cantonese/rules/`` (or a user-supplied override directory) and
contain an ordered array of ``[[rules]]`` tables with ``pattern``/``replace`` strings
and an optional ``comment``. Rules are applied sequentially, one ``re.sub`` pass each.
"""

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

BUILTIN_RULES_DIR = Path(__file__).parent / "rules"


@dataclass(frozen=True)
class Rule:
    pattern: "re.Pattern[str]"
    replace: str
    comment: Optional[str] = None


RuleSet = List[Rule]


def load_ruleset(path: Path) -> RuleSet:
    """Load and compile an ordered rule list from a TOML file.

    Raises ValueError naming the file and rule index on a missing key or bad regex.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    rules: RuleSet = []
    for i, entry in enumerate(data.get("rules", [])):
        try:
            pattern = re.compile(entry["pattern"])
            replace = entry["replace"]
        except KeyError as e:
            raise ValueError(f"{path}: rule #{i + 1} is missing required key {e}") from e
        except re.error as e:
            raise ValueError(f"{path}: rule #{i + 1} has an invalid regex: {e}") from e
        rules.append(Rule(pattern, replace, entry.get("comment")))

    return rules


def apply_ruleset(text: str, rules: RuleSet) -> str:
    for rule in rules:
        text = rule.pattern.sub(rule.replace, text)
    return text


@functools.lru_cache(maxsize=None)
def get_builtin_ruleset(name: str) -> RuleSet:
    """Load a packaged ruleset by name (e.g. ``"chars_hk"``), cached per process."""
    return load_ruleset(BUILTIN_RULES_DIR / f"{name}.toml")

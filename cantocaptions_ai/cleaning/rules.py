"""TOML-backed regex rule engine for subtitle cleaning.

A rule file holds an ordered array of ``[[rules]]`` tables with ``pattern``/``replace``
strings, an optional ``comment``, and an optional ``when``: a regex the line must contain
(``re.search``) for the rule to apply at all, e.g. only to a line with Chinese in it. Rules
are applied sequentially, one ``re.sub`` pass each. Language packs ship their rule files
(e.g. ``languages/yue/rules/``); a user may point ``--clean_rules_dir`` at their own.
"""
import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Rule:
    pattern: "re.Pattern[str]"
    replace: str
    comment: Optional[str] = None
    when: Optional["re.Pattern[str]"] = None


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
            when = re.compile(entry["when"]) if entry.get("when") is not None else None
        except KeyError as e:
            raise ValueError(f"{path}: rule #{i + 1} is missing required key {e}") from e
        except re.error as e:
            raise ValueError(f"{path}: rule #{i + 1} has an invalid regex: {e}") from e
        rules.append(Rule(pattern, replace, entry.get("comment"), when))
    return rules


def apply_ruleset(text: str, rules: RuleSet) -> str:
    for rule in rules:
        if rule.when is not None and not rule.when.search(text):
            continue
        text = rule.pattern.sub(rule.replace, text)
    return text


@functools.lru_cache(maxsize=None)
def _load_ruleset_cached(resolved: str) -> RuleSet:
    return load_ruleset(Path(resolved))


def load_ruleset_cached(path: Union[str, Path]) -> RuleSet:
    """:func:`load_ruleset`, cached per process by resolved path.

    Packaged rule files are read once however many cleaners are built; the cache is keyed
    on the resolved path, so two spellings of one file share an entry.
    """
    return _load_ruleset_cached(str(Path(path).resolve()))

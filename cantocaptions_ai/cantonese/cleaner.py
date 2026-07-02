"""Manifest-driven Cantonese subtitle text cleaner.

``SubtitleCleaner`` folds each subtitle line through the step sequence declared in
``rules/pipeline.toml``: TOML regex rule files interleaved with coded builtin steps
(question-aware particle fixes, Chinese numeral conversion, line breaking, trimming).
Point ``rules_dir`` at a directory with its own ``pipeline.toml`` to swap rule sets.

Cleaning may return an empty string (noise-only lines); callers should drop those
subtitles (see ``text.is_removable``).
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from cantocaptions_ai.cantonese.linebreak import linebreak, trim
from cantocaptions_ai.cantonese.numbers import convert_chinese_numbers
from cantocaptions_ai.cantonese.questions import clean_question_particles
from cantocaptions_ai.cantonese.rules import (
    BUILTIN_RULES_DIR,
    apply_ruleset,
    get_builtin_ruleset,
    load_ruleset,
)


class SubtitleCleaner:
    """Applies the configured cleaning steps to a single subtitle line at a time."""

    BUILTIN_STEPS = {
        "question_particles": clean_question_particles,
        "chinese_numbers": convert_chinese_numbers,
        "trim": trim,
        # "linebreak" is constructed per instance (depends on line settings)
    }

    def __init__(
        self,
        rules_dir: Optional[str] = None,
        line_max_length: int = 18,
        max_line_count: Optional[int] = 1,
    ) -> None:
        self.rules_dir = Path(rules_dir) if rules_dir is not None else BUILTIN_RULES_DIR
        self.line_max_length = line_max_length
        self.max_line_count = max_line_count
        # Fails fast on a missing/invalid manifest, rule file, or regex so problems
        # surface at pipeline start rather than after hours of ASR.
        self._steps = self._load_steps()

    def _load_steps(self) -> List[Tuple[str, Callable[[str], str]]]:
        manifest_path = self.rules_dir / "pipeline.toml"
        if not manifest_path.is_file():
            raise ValueError(f"Cleaning manifest not found: {manifest_path}")

        with open(manifest_path, "rb") as f:
            manifest = tomllib.load(f)

        steps: List[Tuple[str, Callable[[str], str]]] = []
        for i, entry in enumerate(manifest.get("steps", [])):
            step_type = entry.get("type")
            if step_type == "rules":
                file = entry.get("file")
                if not file:
                    raise ValueError(f"{manifest_path}: step #{i + 1} is missing 'file'")
                if self.rules_dir == BUILTIN_RULES_DIR:
                    rules = get_builtin_ruleset(Path(file).stem)
                else:
                    rule_path = self.rules_dir / file
                    if not rule_path.is_file():
                        raise ValueError(f"{manifest_path}: step #{i + 1} rule file not found: {rule_path}")
                    rules = load_ruleset(rule_path)
                steps.append((file, lambda text, _rules=rules: apply_ruleset(text, _rules)))
            elif step_type == "builtin":
                name = entry.get("name")
                if name == "linebreak":
                    if self.max_line_count is not None and self.max_line_count < 2:
                        continue  # a single-line output can't take a break
                    steps.append((name, lambda text: linebreak(text, self.line_max_length)))
                elif name in self.BUILTIN_STEPS:
                    steps.append((name, self.BUILTIN_STEPS[name]))
                else:
                    raise ValueError(f"{manifest_path}: step #{i + 1} has unknown builtin '{name}'")
            else:
                raise ValueError(f"{manifest_path}: step #{i + 1} has unknown type '{step_type}'")

        return steps

    def clean(self, text: str) -> str:
        """Clean a single subtitle line. May return an empty string (drop the subtitle)."""
        for _name, step in self._steps:
            text = step(text)
        return text

"""Manifest-driven subtitle text cleaner.

``SubtitleCleaner`` folds each subtitle line through the step sequence declared in a
manifest: TOML regex rule files (``cleaning/rules.py``) interleaved with coded builtin
steps. Nothing here knows a language. A language pack supplies the rules directory, the
builtin steps its manifests may name, the lines it treats as noise and its line breaker
(see ``languages/base.py`` ``CleaningSpec``); the Cantonese ones live in
``languages/yue``.

Which manifest is a per-model decision, since it depends on how much the model's raw
output already follows the target convention -- the language pack's conventions for the
model supply it. Point ``rules_dir`` at a directory with its own manifest to swap rule
sets entirely.

Cleaning may return an empty string (noise-only lines); callers should drop those
subtitles (see :meth:`SubtitleCleaner.is_noise`).

A manifest may also declare ``[[pre_align]]`` steps, in the same format. Those run on
the raw ASR text *before* alignment (``SubtitleCleaner.pre_align``), so punctuation
they insert becomes a clause boundary alignment can split a cue on, rather than
landing mid-cue or on a cue's end after the cues are already cut.
"""
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from cantocaptions_ai.cleaning.layout import linebreak_step
from cantocaptions_ai.cleaning.rules import apply_ruleset, load_ruleset_cached
from cantocaptions_ai.text_profiles import DEFAULT_CLEANING
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


class SubtitleCleaner:
    """Applies the configured cleaning steps to a single subtitle line at a time."""

    def __init__(
        self,
        rules_dir: Union[str, Path],
        line_max_length: int = 18,
        max_line_count: Optional[int] = 1,
        manifest: str = DEFAULT_CLEANING.manifest,
        builtin_steps: Optional[Mapping[str, Callable[[str], str]]] = None,
        noise_tokens: Sequence[str] = (),
        layout: str = "cjk",
    ) -> None:
        """``rules_dir`` holds the manifest and its rule files. ``builtin_steps`` are the
        coded steps a manifest may name (``linebreak`` is always available, built from the
        line settings and ``layout``); ``noise_tokens`` are the lines :meth:`is_noise`
        treats as droppable."""
        self.rules_dir = Path(rules_dir)
        self.manifest = manifest
        self.line_max_length = line_max_length
        self.max_line_count = max_line_count
        self.builtin_steps = dict(builtin_steps or {})
        self.noise_tokens = tuple(noise_tokens)
        self.layout = layout
        # Fails fast on a missing/invalid manifest, rule file, or regex so problems
        # surface at pipeline start rather than after hours of ASR.
        manifest_path, manifest_data = self._load_manifest()
        self._steps = self._load_steps(manifest_path, manifest_data, "steps")
        self._pre_align_steps = self._load_steps(manifest_path, manifest_data, "pre_align")

    def _load_manifest(self) -> Tuple[Path, dict]:
        manifest_path = self.rules_dir / self.manifest
        # An override directory supplies whatever rule set the caller wrote and need not
        # know which manifest the chosen model asks for, so fall back to the documented
        # `pipeline.toml` contract rather than failing on a model-specific name.
        if not manifest_path.is_file() and self.manifest != DEFAULT_CLEANING.manifest:
            fallback = self.rules_dir / DEFAULT_CLEANING.manifest
            if fallback.is_file():
                logger.info(
                    "Cleaning manifest %r not found in %s; using %s",
                    self.manifest, self.rules_dir, DEFAULT_CLEANING.manifest,
                )
                manifest_path = fallback
        if not manifest_path.is_file():
            raise ValueError(f"Cleaning manifest not found: {manifest_path}")

        with open(manifest_path, "rb") as f:
            return manifest_path, tomllib.load(f)

    def _load_steps(
        self, manifest_path: Path, manifest: dict, key: str,
    ) -> List[Tuple[str, Callable[[str], str]]]:
        steps: List[Tuple[str, Callable[[str], str]]] = []
        for i, entry in enumerate(manifest.get(key, [])):
            step_type = entry.get("type")
            if step_type == "rules":
                file = entry.get("file")
                if not file:
                    raise ValueError(f"{manifest_path}: {key} #{i + 1} is missing 'file'")
                rule_path = self.rules_dir / file
                if not rule_path.is_file():
                    raise ValueError(f"{manifest_path}: {key} #{i + 1} rule file not found: {rule_path}")
                rules = load_ruleset_cached(rule_path)
                steps.append((file, lambda text, _rules=rules: apply_ruleset(text, _rules)))
            elif step_type == "builtin":
                name = entry.get("name")
                if name == "linebreak":
                    step = linebreak_step(self.line_max_length, self.max_line_count, self.layout)
                    if step is not None:
                        steps.append((name, step))
                elif name in self.builtin_steps:
                    steps.append((name, self.builtin_steps[name]))
                else:
                    raise ValueError(f"{manifest_path}: {key} #{i + 1} has unknown builtin '{name}'")
            else:
                raise ValueError(f"{manifest_path}: {key} #{i + 1} has unknown type '{step_type}'")

        return steps

    def clean(self, text: str) -> str:
        """Clean a single subtitle line. May return an empty string (drop the subtitle)."""
        for _name, step in self._steps:
            text = step(text)
        return text

    def is_noise(self, text: str) -> bool:
        """True if a cleaned line holds nothing worth showing (empty, or a noise token)."""
        return len(text) == 0 or text in self.noise_tokens

    def pre_align(self, text: str) -> str:
        """Apply the manifest's ``pre_align`` steps to raw ASR text (a no-op if it has none)."""
        for _name, step in self._pre_align_steps:
            text = step(text)
        return text

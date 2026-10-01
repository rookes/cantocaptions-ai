"""Layered config-file resolution for the CLI.

Precedence (lowest -> highest):
    PipelineConfig.defaults()  ->  one cfg file (default.cfg or --cfg NAME)
    ->  user.cfg (personal overrides; optional)
    ->  stage-preset flags (--vocal_isolation/--asr/--align)  ->  explicit CLI flags

The shipped cfg files (default.cfg and the named presets, e.g. cpu.cfg) are package
data, in cantocaptions_ai/presets/, so they work the same from a checkout or a pip
install. A file of the same name in your config directory takes their place, and
user.cfg is read from there. The config directory is $CANTOCAPTIONS_CONFIG_DIR if set,
else ./config if the working directory has one, else the checkout's config/, else the
platform's user config directory (~/.config/cantocaptions-ai on Linux). Nothing is
ever created in it.

Config files are INI (stdlib configparser), a ``[pipeline]`` block and/or per-section
blocks (``[vad]``, ``[alignment]``, ...; see config.CONFIG_SECTIONS), read
as raw strings and coerced using each argparse action's own ``type=``/
``choices=`` metadata (see load_cfg_file) -- no second type table to keep in
sync with __main__.py's flag definitions.

Both full-line and trailing ``#`` comments are stripped, because the shipped
presets/default.cfg annotates its values inline and configparser does NOT do
this by default -- an unstripped ``attn_implementation = sdpa # ...`` reaches
the choices= check as the whole run-on string and aborts the run. The cost is
that a value cannot itself contain a literal ``#``; nothing the pipeline takes
(paths, model ids, numbers, enums) plausibly does.

Only PipelineConfig field names are legal cfg-file keys; CLI-only args
(log_level, log_file, input_dir, recursive, cfg, and the 3 preset dests
themselves) are rejected as "unknown key" if present in a cfg file. Keys in
REMOVED_KEYS (fields since deleted) are skipped with a warning instead.
"""
import argparse
import configparser
import os
import sys
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

from cantocaptions_ai.pipeline.config import CONFIG_SECTIONS, PipelineConfig
from cantocaptions_ai.utils.output import str2bool

CONFIG_DIR_NAME = "config"
DEFAULT_CFG_FILENAME = "default.cfg"
USER_CFG_FILENAME = "user.cfg"

# config/ in a source checkout: cantocaptions_ai/pipeline/cli_config.py -> repo root.
_REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / CONFIG_DIR_NAME
# The shipped default.cfg and named presets, installed with the package.
PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"
CONFIG_DIR_ENV = "CANTOCAPTIONS_CONFIG_DIR"
_SECTION = "pipeline"

_PIPELINE_FIELD_NAMES = {f.name for f in fields(PipelineConfig)}

# Keys that were once PipelineConfig fields and have since been removed because nothing
# read them (Whisper-era ASR options the Qwen backends never supported, and formatting
# switches no writer implemented). A cfg file written before the removal still loads: the
# key is skipped with a warning rather than failing as an unknown key.
REMOVED_KEYS = frozenset({
    "fp16", "segment_resolution", "highlight_words", "initial_prompt", "hotwords",
    "suppress_tokens", "suppress_numerals", "condition_on_previous_text",
})

# One entry per stage-preset flag: dest -> tier name -> the field(s) it sets.
_STAGE_PRESETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "vocal_isolation": {
        "fast": {"vocal_isolation_compute_type": "float16"},
        "quality": {"vocal_isolation_compute_type": "float32"},
    },
    "asr": {
        "fast": {"asr_compute_type": "int8"},
        "quality": {"asr_compute_type": "float32"},
    },
    "align": {
        "fast": {"align_compute_type": "float16"},
        "quality": {"align_compute_type": "float32"},
    },
}


def _user_config_dir() -> Path:
    """The platform's per-user config directory for this tool."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "cantocaptions-ai"


def default_config_dir() -> Optional[Path]:
    """Your config directory (for user.cfg, and your own copies of the presets), or None.

    $CANTOCAPTIONS_CONFIG_DIR wins outright. Otherwise ./config when the working directory
    has one (the `uv run cantocaptions` from-the-repo-root workflow), then the checkout's
    own config/ so running from another directory still reads your settings, then the
    platform's user config directory, so a pip install has somewhere to keep user.cfg.
    """
    env = os.environ.get(CONFIG_DIR_ENV)
    if env:
        return Path(env)
    for candidate in (Path.cwd() / CONFIG_DIR_NAME, _REPO_CONFIG_DIR, _user_config_dir()):
        if candidate.is_dir():
            return candidate
    return None


def _is_bool_flag(action: argparse.Action) -> bool:
    """True for store_true/store_false actions (nargs=0, const is a bool)."""
    return action.nargs == 0 and isinstance(getattr(action, "const", None), bool)


def shipped_presets() -> list:
    """Names of the presets that ship with the package (``--cfg NAME``)."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.cfg") if p.stem != "default")


def resolve_cfg_path(
    cfg_name: Optional[str],
    parser: argparse.ArgumentParser,
    config_dir: Optional[Path] = None,
) -> Path:
    """The cfg file to read: ``default.cfg`` when *cfg_name* is None, else ``NAME.cfg``.

    Your config directory's copy wins; otherwise the one shipped with the package. A name
    found in neither is an error (parser.error), listing the shipped presets.
    """
    config_dir = config_dir if config_dir is not None else default_config_dir()
    if cfg_name is None:
        name = "default"
    else:
        name = cfg_name[:-4] if cfg_name.endswith(".cfg") else cfg_name
        if not name or "/" in name or "\\" in name or ".." in name:
            parser.error(f"--cfg: invalid config name '{cfg_name}'")
    candidates = ([config_dir / f"{name}.cfg"] if config_dir is not None else [])
    candidates.append(PRESETS_DIR / f"{name}.cfg")
    for path in candidates:
        if path.is_file():
            return path
    parser.error(
        f"--cfg '{cfg_name}': no such config file (looked for "
        f"{' and '.join(str(c) for c in candidates)}); shipped presets: "
        f"{', '.join(shipped_presets()) or 'none'}"
    )


def load_cfg_file(path: Path, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """Read the file's settings, coercing each value via the matching argparse action's
    type=/choices=.

    Settings may sit in one flat ``[pipeline]`` block, in per-section blocks (``[vad]``,
    ``[alignment]``, ... -- see config.CONFIG_SECTIONS), or both. A setting in a section
    block must belong to that section, and none may be given twice.

    Fails fast (via parser.error) on a missing section, unknown key, or a
    value that fails type=/choices= validation -- mirroring
    cantocaptions_ai/cleaning/engine.py's SubtitleCleaner._load_steps,
    which fails fast on a bad manifest/rule file so problems surface at
    pipeline start rather than mid-run.
    """
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    if not cp.read(path, encoding="utf-8"):
        parser.error(f"could not read config file: {path}")
    unknown_sections = [s for s in cp.sections() if s != _SECTION and s not in CONFIG_SECTIONS]
    if unknown_sections:
        parser.error(
            f"{path}: unknown section(s) {', '.join(f'[{s}]' for s in unknown_sections)} "
            f"(use [{_SECTION}] or one of: {', '.join(CONFIG_SECTIONS)})"
        )
    if not cp.sections():
        parser.error(f"{path}: no settings: expected a [{_SECTION}] block or section blocks")

    dest_to_action = {
        a.dest: a for a in parser._actions if a.dest in _PIPELINE_FIELD_NAMES
    }
    resolved: Dict[str, Any] = {}
    entries = [
        (section, key, raw)
        for section in cp.sections()
        for key, raw in cp.items(section, raw=True)
    ]
    for section, key, raw in entries:
        if key in _PIPELINE_FIELD_NAMES and section != _SECTION:
            home = PipelineConfig.section_of(key)
            if home != section:
                parser.error(f"{path}: '{key}' belongs in [{home}], not [{section}]")
        if key in resolved:
            parser.error(f"{path}: '{key}' is set more than once")
        if key in REMOVED_KEYS:
            warnings.warn(
                f"{path}: '{key}' has been removed and is ignored (it never had any "
                f"effect); delete it from the file to silence this warning"
            )
            continue
        action = dest_to_action.get(key)
        if action is None:
            parser.error(f"{path}: unknown config key '{key}'")
        try:
            # The literal string "None" is this project's existing sentinel
            # for an unset Optional field (see utils/output.py's optional_int/
            # optional_float) -- applied universally here, not just for those
            # two types, since PipelineConfig.defaults() writes str(None) for
            # every Optional field regardless of its action's type= callable
            # (e.g. --audio_start's type=float can't parse "None" itself).
            if raw == "None":
                value = None
            elif action.type is not None:
                value = action.type(raw)
            elif _is_bool_flag(action):
                value = str2bool(raw)
            else:
                value = raw
        except ValueError as e:
            parser.error(f"{path}: bad value for '{key}': {e}")
        if value is not None and action.choices is not None and value not in action.choices:
            parser.error(
                f"{path}: invalid value for '{key}': {raw!r} "
                f"(choose from {sorted(map(str, action.choices))})"
            )
        resolved[key] = value
    return resolved


def resolve_pipeline_args(
    parser: argparse.ArgumentParser,
    explicit: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """The 5-layer merge: dataclass defaults -> cfg file -> user.cfg -> stage
    presets -> explicit CLI flags.

    user.cfg sits above whichever cfg was selected, so personal settings (batch
    sizes for your card, a gated model you have access to) hold under --cfg too.

    `explicit` is vars(parser.parse_args()); thanks to default=argparse.SUPPRESS
    on every add_argument() in __main__.py, it contains ONLY keys the user
    actually typed (plus the always-present positional `audio`).

    This ordering is also the tie-break for "preset flag vs. its own granular
    flag both given": preset_layer only gets a key when the *preset* flag was
    passed, and explicit only gets a key when that *specific* flag was passed
    -- so a plain dict-merge makes the granular flag win with no special-case
    code, regardless of argument order on the command line.
    """
    config_dir = config_dir if config_dir is not None else default_config_dir()
    cfg_path = resolve_cfg_path(explicit.get("cfg"), parser, config_dir)
    cfg_layer = load_cfg_file(cfg_path, parser)
    user_path = config_dir / USER_CFG_FILENAME if config_dir is not None else None
    user_layer = (
        load_cfg_file(user_path, parser) if user_path is not None and user_path.is_file() else {}
    )

    preset_layer: Dict[str, Any] = {}
    for preset_dest, tiers in _STAGE_PRESETS.items():
        tier = explicit.get(preset_dest)
        if tier is not None:
            preset_layer.update(tiers[tier])

    return {**PipelineConfig.defaults(), **cfg_layer, **user_layer, **preset_layer, **explicit}


class ConfigAwareHelpFormatter(argparse.HelpFormatter):
    """Like ArgumentDefaultsHelpFormatter, but resolves the displayed default
    from a supplied dict (PipelineConfig.defaults()) instead of action.default,
    which is intentionally argparse.SUPPRESS on every action (see __main__.py).
    """

    def __init__(self, prog, defaults: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(prog, **kwargs)
        self._defaults = defaults or {}

    def _get_help_string(self, action: argparse.Action) -> str:
        help_str = action.help or ""
        if action.dest in self._defaults and "(default:" not in help_str:
            help_str = f"{help_str} (default: {self._defaults[action.dest]})"
        return help_str


def describe_config(parser: Optional[argparse.ArgumentParser] = None) -> list:
    """Every setting, by section, as plain data: what a settings UI needs to draw a form.

    Each section is ``{"name", "title", "settings": [...]}`` in ``--help`` order, and each
    setting ``{"name", "flag", "default", "type", "choices", "help"}``. Help text and
    choices come from the CLI's own flags, so this cannot drift from ``--help``.
    """
    import typing

    from cantocaptions_ai.pipeline.config import SECTION_TITLES

    if parser is None:
        from cantocaptions_ai.__main__ import build_parser
        parser = build_parser()
    actions = {a.dest: a for a in parser._actions if a.dest in _PIPELINE_FIELD_NAMES}
    types = {
        name: hint.__name__ if isinstance(hint, type) else str(hint).replace("typing.", "")
        for name, hint in typing.get_type_hints(PipelineConfig).items()
    }
    defaults = PipelineConfig.defaults()
    out = []
    for section, keys in CONFIG_SECTIONS.items():
        settings = []
        for key in keys:
            action = actions[key]
            settings.append({
                "name": key,
                "flag": max(action.option_strings, key=len),
                "default": defaults[key],
                "type": types[key],
                "choices": list(action.choices) if action.choices is not None else None,
                "help": action.help,
            })
        out.append({"name": section, "title": SECTION_TITLES[section], "settings": settings})
    return out

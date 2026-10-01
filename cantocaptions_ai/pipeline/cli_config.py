"""Layered config-file resolution for the CLI.

Precedence (lowest -> highest):
    PipelineConfig.defaults()  ->  one cfg file (default.cfg or --cfg NAME)
    ->  config/user.cfg (personal overrides, gitignored; optional)
    ->  stage-preset flags (--vocal_isolation/--asr/--align)  ->  explicit CLI flags

The config directory is ./config if the working directory has one, else the
config/ beside the package in a source checkout. With neither (an installed
package run from elsewhere) no file is read and the built-in defaults apply.

Config files are INI (stdlib configparser), one ``[pipeline]`` section, read
as raw strings and coerced using each argparse action's own ``type=``/
``choices=`` metadata (see load_cfg_file) -- no second type table to keep in
sync with __main__.py's flag definitions.

Both full-line and trailing ``#`` comments are stripped, because the shipped
config/default.cfg annotates its values inline and configparser does NOT do
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
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.utils.output import str2bool

CONFIG_DIR_NAME = "config"
DEFAULT_CFG_FILENAME = "default.cfg"
USER_CFG_FILENAME = "user.cfg"

# config/ in a source checkout: cantocaptions_ai/pipeline/cli_config.py -> repo root.
_REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / CONFIG_DIR_NAME
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


def default_config_dir() -> Optional[Path]:
    """The config directory to read, or None if there is none.

    ./config wins when the working directory has one (the `uv run cantocaptions`
    from-the-repo-root workflow). Otherwise the checkout's own config/ is used, so
    running from another directory still reads your settings instead of silently
    creating a fresh config/default.cfg wherever you happen to be.
    """
    cwd_dir = Path.cwd() / CONFIG_DIR_NAME
    if cwd_dir.is_dir():
        return cwd_dir
    if _REPO_CONFIG_DIR.is_dir():
        return _REPO_CONFIG_DIR
    return None


def _is_bool_flag(action: argparse.Action) -> bool:
    """True for store_true/store_false actions (nargs=0, const is a bool)."""
    return action.nargs == 0 and isinstance(getattr(action, "const", None), bool)


def ensure_default_cfg_exists(config_dir: Path) -> Path:
    """Create config/default.cfg from PipelineConfig.defaults() if missing.

    Never overwrites an existing file -- once created, it's the user's
    personal baseline to hand-edit.
    """
    path = config_dir / DEFAULT_CFG_FILENAME
    if path.exists():
        return path
    config_dir.mkdir(parents=True, exist_ok=True)
    cp = configparser.ConfigParser()
    cp[_SECTION] = {k: str(v) for k, v in PipelineConfig.defaults().items()}
    with open(path, "w", encoding="utf-8") as f:
        cp.write(f)
    return path


def resolve_cfg_path(
    cfg_name: Optional[str],
    parser: argparse.ArgumentParser,
    config_dir: Optional[Path] = None,
) -> Optional[Path]:
    """cfg_name is None -> auto-create/reuse config/default.cfg. Otherwise
    resolve config/{cfg_name}.cfg, erroring out (parser.error) if it doesn't
    exist -- same style as __main__.py's existing --input_dir validation.

    Returns None (built-in defaults only) when no config directory was found.
    """
    config_dir = config_dir if config_dir is not None else default_config_dir()
    if config_dir is None:
        if cfg_name is not None:
            parser.error(
                f"--cfg '{cfg_name}': no config directory found "
                f"(looked for ./{CONFIG_DIR_NAME} and {_REPO_CONFIG_DIR})"
            )
        return None
    if cfg_name is None:
        return ensure_default_cfg_exists(config_dir)

    name = cfg_name[:-4] if cfg_name.endswith(".cfg") else cfg_name
    if not name or "/" in name or "\\" in name or ".." in name:
        parser.error(f"--cfg: invalid config name '{cfg_name}'")
    path = config_dir / f"{name}.cfg"
    if not path.is_file():
        parser.error(
            f"--cfg '{cfg_name}': no such config file '{path}' "
            f"(looked in '{config_dir}'; run with no --cfg to auto-create default.cfg)"
        )
    return path


def load_cfg_file(path: Path, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """Read the [pipeline] section, coercing each value via the matching
    argparse action's type=/choices=.

    Fails fast (via parser.error) on a missing section, unknown key, or a
    value that fails type=/choices= validation -- mirroring
    cantocaptions_ai/cleaning/engine.py's SubtitleCleaner._load_steps,
    which fails fast on a bad manifest/rule file so problems surface at
    pipeline start rather than mid-run.
    """
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    if not cp.read(path, encoding="utf-8"):
        parser.error(f"could not read config file: {path}")
    if not cp.has_section(_SECTION):
        parser.error(f"{path}: missing required [{_SECTION}] section")

    dest_to_action = {
        a.dest: a for a in parser._actions if a.dest in _PIPELINE_FIELD_NAMES
    }
    resolved: Dict[str, Any] = {}
    for key, raw in cp[_SECTION].items():
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
    cfg_layer = load_cfg_file(cfg_path, parser) if cfg_path is not None else {}
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

"""The language registry: one :class:`LanguagePack` per supported language.

``get_language_pack(code)`` returns the registered pack for a language, or a
:func:`generic_pack` for any other so the pipeline can still run it raw. Packs register
themselves with :func:`register_language_pack`; the built-in ones are registered on import.

A pack may also ship as its own distribution: any installed package declaring an entry point
in the ``cantocaptions_ai.languages`` group (a ``LanguagePack``, or a zero-argument callable
returning one) is registered the first time the registry is read. See ``languages/base.py``
and ``docs/adding-a-language.md``.

Importing this package must stay cheap (no torch): the model registry and the worker's
preflight checks read it.
"""
import logging
from typing import Dict, Iterator, Mapping, Optional

from cantocaptions_ai.languages.base import (
    CleaningSpec,
    CorrectionPrompts,
    LanguagePack,
    ModelConventions,
    RunProfile,
    generic_pack,
)

# The language a caller that names none gets -- the one this project is built around.
DEFAULT_LANGUAGE = "yue"

# The entry-point group installed language packs register under.
ENTRY_POINT_GROUP = "cantocaptions_ai.languages"

logger = logging.getLogger(__name__)

_PACKS: Dict[str, LanguagePack] = {}
_plugins_loaded = False


def _load_plugin_packs() -> None:
    """Register every installed plugin pack, once. A broken one is skipped with a warning
    naming the distribution it came from, so one bad package cannot stop the pipeline."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    from importlib.metadata import entry_points

    try:
        found = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - broken site-packages metadata
        logger.warning("Could not look up language pack plugins: %s", exc)
        return
    for entry in found:
        source = getattr(getattr(entry, "dist", None), "name", None) or entry.value
        try:
            obj = entry.load()
            pack = obj if isinstance(obj, LanguagePack) else obj()
            if not isinstance(pack, LanguagePack):
                raise TypeError(f"{entry.value} gave a {type(pack).__name__}, not a LanguagePack")
        except Exception as exc:
            logger.warning("Language pack plugin %r from %s is skipped: %s", entry.name, source,
                           exc)
            continue
        if pack.code in _PACKS:
            logger.info("Language pack %r from %s replaces the one registered before it",
                        pack.code, source)
        _PACKS[pack.code] = pack


class _RegisteredPacks(Mapping):
    """Read-only view of the registered packs, by code; reading it loads the plugins."""

    def __getitem__(self, code: str) -> LanguagePack:
        _load_plugin_packs()
        return _PACKS[code]

    def __iter__(self) -> Iterator[str]:
        _load_plugin_packs()
        return iter(dict(_PACKS))

    def __len__(self) -> int:
        _load_plugin_packs()
        return len(_PACKS)

    def __repr__(self) -> str:
        return f"LANGUAGE_PACKS({sorted(self)})"


# Read-only view of the registered packs, by language code.
LANGUAGE_PACKS: Mapping[str, LanguagePack] = _RegisteredPacks()


def register_language_pack(pack: LanguagePack) -> None:
    """Register (or replace) the pack for ``pack.code``. Installed plugin packs are loaded
    first, so a pack registered here wins over one of theirs for the same language."""
    _load_plugin_packs()
    _PACKS[pack.code] = pack


def get_language_pack(code: Optional[str] = None) -> LanguagePack:
    """The pack for *code* (default: yue), or a generic one if none is registered."""
    code = code or DEFAULT_LANGUAGE
    _load_plugin_packs()
    return _PACKS.get(code) or generic_pack(code)


def _register_builtin_packs() -> None:
    # Straight into the table: register_language_pack would load the plugins first, and
    # they must come after the built-ins so that one can replace a built-in pack.
    from cantocaptions_ai.languages.yue import YUE
    _PACKS[YUE.code] = YUE


_register_builtin_packs()

__all__ = [
    "CleaningSpec", "CorrectionPrompts", "DEFAULT_LANGUAGE", "ENTRY_POINT_GROUP",
    "LANGUAGE_PACKS", "LanguagePack",
    "ModelConventions", "RunProfile", "generic_pack", "get_language_pack",
    "register_language_pack",
]

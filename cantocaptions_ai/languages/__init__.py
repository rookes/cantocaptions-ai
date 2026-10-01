"""The language registry: one :class:`LanguagePack` per supported language.

``get_language_pack(code)`` returns the registered pack for a language, or a
:func:`generic_pack` for any other so the pipeline can still run it raw. Packs register
themselves with :func:`register_language_pack`; the built-in ones are registered on import.
See ``languages/base.py`` and ``docs/adding-a-language.md``.

Importing this package must stay cheap (no torch): the model registry and the worker's
preflight checks read it.
"""
from types import MappingProxyType
from typing import Dict, Mapping, Optional

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

_PACKS: Dict[str, LanguagePack] = {}

# Read-only view of the registered packs, by language code.
LANGUAGE_PACKS: Mapping[str, LanguagePack] = MappingProxyType(_PACKS)


def register_language_pack(pack: LanguagePack) -> None:
    """Register (or replace) the pack for ``pack.code``."""
    _PACKS[pack.code] = pack


def get_language_pack(code: Optional[str] = None) -> LanguagePack:
    """The pack for *code* (default: yue), or a generic one if none is registered."""
    code = code or DEFAULT_LANGUAGE
    return _PACKS.get(code) or generic_pack(code)


def _register_builtin_packs() -> None:
    from cantocaptions_ai.languages.yue import YUE
    register_language_pack(YUE)


_register_builtin_packs()

__all__ = [
    "CleaningSpec", "CorrectionPrompts", "DEFAULT_LANGUAGE", "LANGUAGE_PACKS", "LanguagePack",
    "ModelConventions", "RunProfile", "generic_pack", "get_language_pack",
    "register_language_pack",
]

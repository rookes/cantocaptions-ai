"""Typed, catchable pipeline errors.

Kept in a dependency-free module so any layer (audio I/O, config validation, the
service entry point, a web worker) can raise/catch these without import cycles.
The CLI adapter translates them into ``parser.error(...)`` (exit 2); a server
maps them onto HTTP status codes instead of crashing the worker.
"""


class CantoCaptionsError(Exception):
    """Base class for all cantocaptions-ai errors callers may want to handle."""


class ConfigError(CantoCaptionsError, ValueError):
    """The requested configuration is invalid (bad/contradictory options)."""


class InputError(CantoCaptionsError, ValueError):
    """The input media file is missing, unreadable, or otherwise unusable."""

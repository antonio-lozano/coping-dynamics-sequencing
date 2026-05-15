"""coping-dynamics-sequencing source modules.

Keep package imports lightweight so tool-specific entrypoints do not pull
optional visualization dependencies unless they are actually used.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "config",
    "io_utils",
    "metrics",
    "plotting",
    "transition_utils",
    "ml",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

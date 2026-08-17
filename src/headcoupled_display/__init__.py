"""Head-coupled 3D display controller.

The re-exports below are resolved lazily (PEP 562). The FaceMesh producer runs in a
separate Python 3.10 / CUDA environment and imports only ``headcoupled_display.protocol``,
which is deliberately dependency-free; eagerly importing ``models`` here would drag in
3.11+ syntax and break that process at import time.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .models import HardwareProfile, TrackingState, UserProfile

__all__ = ["HardwareProfile", "TrackingState", "UserProfile"]
__version__ = "0.1.0"

_LAZY_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "__version__"])

"""dtype narrowing helpers for values crossing the OpenCV boundary.

cv2's type stubs describe nearly every return value as `MatLike`
(`ndarray[Any, dtype[integer | floating]]`), which drops the concrete dtype the
rest of tagcal relies on. These helpers re-attach it once, at the boundary, so
internal signatures can stay precise instead of degrading to `Any`.

`np.asarray` is a no-op view when the dtype already matches, which it always
does for the OpenCV calls used here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_uint8(value: Any) -> NDArray[np.uint8]:
    """Narrow a decoded frame or mask; capture and image APIs deliver 8-bit data."""
    return np.asarray(value, dtype=np.uint8)


def as_int32(value: Any) -> NDArray[np.int32]:
    """Narrow marker id arrays returned by the ArUco detector."""
    return np.asarray(value, dtype=np.int32)


def as_float32(value: Any) -> NDArray[np.float32]:
    """Narrow corner/contour arrays, which OpenCV emits as single precision."""
    return np.asarray(value, dtype=np.float32)


def as_float64(value: Any) -> NDArray[np.float64]:
    """Narrow calibration outputs, which OpenCV emits as double precision."""
    return np.asarray(value, dtype=np.float64)


def as_float32_list(values: Iterable[Any]) -> list[NDArray[np.float32]]:
    """Narrow a sequence of corner arrays."""
    return [as_float32(value) for value in values]


def as_float64_list(values: Iterable[Any]) -> list[NDArray[np.float64]]:
    """Narrow a sequence of per-view rotation/translation vectors."""
    return [as_float64(value) for value in values]

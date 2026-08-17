"""Helpers for generating small synthetic video fixtures used by producer tests.

Kept separate from ``tests/unit/test_producer.py`` so any future FrameSource-adjacent
test module (workdoc steps 28-29) can reuse the same fixture-generation helper without
duplicating OpenCV ``VideoWriter`` boilerplate. Writing (not just reading) a real .avi is
deliberate: it exercises the exact ``cv2.VideoCapture`` code path FrameSource uses,
instead of mocking around it, while staying fast and hardware-free (no real camera).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_synthetic_avi(
    path: Path | str,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
) -> None:
    """Write ``frame_count`` solid-color frames to an MJPG-encoded .avi at ``path``.

    Each frame ``i`` is a distinct solid color (``min(i * 100, 255)``), so a test can
    tell frames apart -- and thus verify read order and EOF -- by inspecting a single
    pixel, without needing real face content. The 100-unit spacing (rather than
    something finer) is deliberate: MJPG's per-frame JPEG compression is lossy even on a
    flat color (observed +/-2 quantization drift), so callers comparing a decoded pixel
    back against ``index * 100`` should use a generous tolerance (e.g. +/-20), not exact
    equality. MJPG is used because it round-trips reliably through the OpenCV build
    available in this environment.
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {path!r} for writing (MJPG, {width}x{height})")
    try:
        for index in range(frame_count):
            value = min(index * 100, 255)
            frame = np.full((height, width, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

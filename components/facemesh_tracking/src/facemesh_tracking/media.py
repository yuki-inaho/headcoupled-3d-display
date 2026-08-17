"""Frame sources and sinks.

Isolates OpenCV I/O from the inference code so the CLI (and any caller) can treat a
still image, a video file and a camera identically.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


@dataclass(frozen=True)
class SourceInfo:
    """Static properties of a frame source; ``frame_count`` is 0 when unknown (camera)."""

    width: int
    height: int
    fps: float
    frame_count: int
    is_image: bool


class FrameSource:
    """Iterable of BGR frames plus the metadata a writer needs."""

    def __init__(self, frames: Iterator[np.ndarray], info: SourceInfo) -> None:
        self._frames = frames
        self.info = info

    def __iter__(self) -> Iterator[np.ndarray]:
        return self._frames


def _capture_frames(capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


@contextmanager
def open_source(
    spec: str, *, width: int | None = None, height: int | None = None
) -> Iterator[FrameSource]:
    """Open ``spec``: a camera index (``"0"``), a video path, or a still-image path."""
    path = Path(spec)
    if not spec.isdecimal() and path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        info = SourceInfo(image.shape[1], image.shape[0], 0.0, 1, is_image=True)
        yield FrameSource(iter([image]), info)
        return

    capture = cv2.VideoCapture(int(spec) if spec.isdecimal() else str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {spec}")
    try:
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        info = SourceInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0),
            is_image=False,
        )
        yield FrameSource(_capture_frames(capture), info)
    finally:
        capture.release()


@contextmanager
def open_writer(path: Path, info: SourceInfo, fourcc: str = "mp4v") -> Iterator[cv2.VideoWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        info.fps or 30.0,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        yield writer
    finally:
        writer.release()

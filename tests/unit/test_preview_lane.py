"""The server must forward a producer-compressed preview, not decode and re-encode it.

Decoding a JPEG on the display machine only to draw on it and encode it again costs a
decode plus an encode every frame and produces nothing the producer could not have drawn
before compressing. The counters here make "the server re-encoded nothing" an assertion
rather than an inference from timings.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from headcoupled_display.models import HardwareProfile, UserProfile
from headcoupled_display.tracking import (
    FaceMeshInputFrame,
    FaceMeshPoseProvider,
    jpeg_dimensions,
)

ROOT = Path(__file__).resolve().parents[2]

PREVIEW_WIDTH_PX = 640
PREVIEW_HEIGHT_PX = 360


def encode_preview(width: int, height: int) -> bytes:
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    assert ok
    return encoded.tobytes()


class SingleFrameSource:
    def __init__(self, frame: FaceMeshInputFrame) -> None:
        self.frame = frame
        self.closed = False

    def next_frame(self) -> FaceMeshInputFrame:
        return self.frame

    def close(self) -> None:
        self.closed = True


def make_provider(preview_jpeg: bytes | None) -> FaceMeshPoseProvider:
    hardware = HardwareProfile.load(ROOT / "config" / "hardware_profile.demo.json")
    frame = FaceMeshInputFrame(
        frame_bgr=np.zeros(
            (hardware.camera.image_height_px, hardware.camera.image_width_px, 3), np.uint8
        ),
        faces=(),
        label="test",
        preview_jpeg=preview_jpeg,
    )
    return FaceMeshPoseProvider(
        hardware,
        UserProfile(),
        source="ipc",
        frame_source=SingleFrameSource(frame),
    )


def test_jpeg_dimensions_reads_the_sof_marker() -> None:
    assert jpeg_dimensions(encode_preview(640, 360)) == (640, 360)
    assert jpeg_dimensions(encode_preview(1280, 720)) == (1280, 720)


def test_a_producer_preview_is_forwarded_byte_for_byte() -> None:
    preview = encode_preview(PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)
    provider = make_provider(preview)

    _state, forwarded = provider.sample()

    assert forwarded is preview
    assert provider.preview_encode_count == 0
    assert provider.preview_forward_count == 1
    assert jpeg_dimensions(forwarded) == (PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)


def test_forwarding_stays_at_zero_encodes_across_many_frames() -> None:
    provider = make_provider(encode_preview(PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX))
    for _ in range(20):
        provider.sample()
    assert provider.preview_encode_count == 0
    assert provider.preview_forward_count == 20


def test_sources_without_a_producer_preview_still_encode_locally() -> None:
    """Recorded replay and synthetic input have no producer to compress for them."""

    provider = make_provider(None)
    _state, encoded = provider.sample()

    assert encoded[:2] == b"\xff\xd8"
    assert provider.preview_encode_count == 1
    assert provider.preview_forward_count == 0


def test_a_full_resolution_preview_is_detectable_without_decoding() -> None:
    """The contract is 640x360; a producer that forgot to resize must be catchable."""

    oversized = encode_preview(1280, 720)
    assert jpeg_dimensions(oversized) != (PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)
    with pytest.raises(ValueError):
        jpeg_dimensions(b"not a jpeg at all")

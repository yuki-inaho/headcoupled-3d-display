from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from tagcal.board import AprilGridBoard
from tagcal.capture import fourcc_text, timestamped_video_path
from tagcal.cvtypes import as_uint8
from tagcal.detection import AprilTagDetector
from tagcal.models import CaptureSpec, PatternSpec
from tagcal.screen import ScreenLayout, plan_layout, render_board

PX_PER_MM = 1920 / 527.0


def test_fourcc_round_trips() -> None:
    assert fourcc_text(cv2.VideoWriter.fourcc(*"MJPG")) == "MJPG"
    assert fourcc_text(0) == "unknown"


def test_recordings_do_not_collide(tmp_path: Path) -> None:
    """A session must never overwrite the previous recording."""
    path = timestamped_video_path(tmp_path)

    assert path.parent == tmp_path
    assert path.suffix == ".mp4"
    assert path.name != "capture.mp4"
    assert path.stem.startswith("capture_")


def test_capture_spec_defaults_to_a_compressed_stream() -> None:
    """Uncompressed YUYV is bandwidth limited to a few fps above VGA."""
    assert CaptureSpec().input_fourcc == "MJPG"


def test_capture_spec_validates_the_fourcc() -> None:
    with pytest.raises(ValueError, match="four-character"):
        CaptureSpec(input_fourcc="MJP")
    assert CaptureSpec(input_fourcc=None).input_fourcc is None


def _rendered_board() -> tuple[NDArray[np.uint8], ScreenLayout]:
    layout = plan_layout(
        PatternSpec(columns=3, rows=2, tag_size_mm=45.0, gap_mm=12.0),
        px_per_mm=PX_PER_MM,
        snap=True,
    )
    return render_board(layout, show_info=False), layout


def test_sharpness_drops_when_the_image_is_blurred() -> None:
    image, layout = _rendered_board()
    detector = AprilTagDetector(AprilGridBoard(layout.spec))

    sharp = detector.detect(image)
    blurred = detector.detect(as_uint8(cv2.GaussianBlur(image, (0, 0), 2.0)))

    assert sharp is not None and blurred is not None
    assert blurred.metrics.sharpness < sharp.metrics.sharpness


def test_sharpness_is_insensitive_to_overall_contrast() -> None:
    """Normalising by region variance is what makes neighbouring frames comparable."""
    image, layout = _rendered_board()
    detector = AprilTagDetector(AprilGridBoard(layout.spec))

    full = detector.detect(image)
    # Same sharpness, half the contrast.
    faded = detector.detect(as_uint8(cv2.convertScaleAbs(image, alpha=0.5, beta=64)))

    assert full is not None and faded is not None
    ratio = faded.metrics.sharpness / full.metrics.sharpness
    assert 0.5 < ratio < 2.0

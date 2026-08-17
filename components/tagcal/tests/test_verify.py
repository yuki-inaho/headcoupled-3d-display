from pathlib import Path

import cv2
import numpy as np
import pytest

from tagcal.models import PatternManifest, PatternSpec
from tagcal.screen import plan_layout, write_manifest
from tagcal.verify import load_intrinsics, verify_board

PX_PER_MM = 1920 / 527.0
CAMERA_MATRIX = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])


def _render_view(
    manifest: PatternManifest,
    board_png: Path,
    px_per_mm: float,
    margin_px: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Project the rendered board sheet as a camera would see it at a known pose."""
    sheet = cv2.imread(str(board_png), cv2.IMREAD_GRAYSCALE)
    assert sheet is not None
    height_px, width_px = sheet.shape

    # Sheet pixel -> board frame in mm; the grid origin sits one margin in.
    def to_board_mm(u: float, v: float) -> list[float]:
        return [(u - margin_px) / px_per_mm, (v - margin_px) / px_per_mm, 0.0]

    object_corners = np.array(
        [
            to_board_mm(0, 0),
            to_board_mm(width_px - 1, 0),
            to_board_mm(width_px - 1, height_px - 1),
            to_board_mm(0, height_px - 1),
        ]
    )
    projected, _ = cv2.projectPoints(object_corners, rvec, tvec, CAMERA_MATRIX, np.zeros(5))
    source = np.array(
        [[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1], [0, height_px - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, projected.reshape(-1, 2).astype(np.float32))

    scene = np.full((720, 1280), 110, np.uint8)
    warped = cv2.warpPerspective(sheet, homography, (1280, 720), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full(sheet.shape, 255, np.uint8), homography, (1280, 720), flags=cv2.INTER_NEAREST
    )
    return np.where(mask > 0, warped, scene).astype(np.uint8)


def test_verify_recovers_the_true_distance(tmp_path: Path) -> None:
    layout = plan_layout(
        PatternSpec(columns=3, rows=2, tag_size_mm=45.0, gap_mm=12.0),
        px_per_mm=PX_PER_MM,
        snap=True,
    )
    manifest = write_manifest(layout, tmp_path)

    # solvePnP reports the distance to the board origin, so that is the ground truth.
    rvec = np.array([0.18, -0.25, 0.05]).reshape(3, 1)
    tvec = np.array([-90.0, -70.0, 620.0]).reshape(3, 1)
    frame = _render_view(
        manifest,
        tmp_path / "apriltag_board.png",
        layout.px_per_mm,
        layout.margin_px,
        rvec,
        tvec,
    )

    result = verify_board(
        manifest,
        [frame],
        camera_matrix=CAMERA_MATRIX,
        distortion=np.zeros((1, 5)),
    )

    assert result.frames_used == 1
    assert result.detected_tags_mean == layout.spec.marker_count
    assert result.distance_mm_mean is not None
    assert result.distance_mm_mean == pytest.approx(float(np.linalg.norm(tvec)), rel=0.01)
    assert result.tag_size_mm == pytest.approx(layout.actual_tag_size_mm)


def test_verify_without_intrinsics_reports_pixels_only(tmp_path: Path) -> None:
    layout = plan_layout(
        PatternSpec(columns=2, rows=2, tag_size_mm=40.0, gap_mm=10.0),
        px_per_mm=PX_PER_MM,
        snap=True,
    )
    manifest = write_manifest(layout, tmp_path)
    frame = _render_view(
        manifest,
        tmp_path / "apriltag_board.png",
        layout.px_per_mm,
        layout.margin_px,
        np.zeros((3, 1)),
        np.array([-60.0, -45.0, 700.0]).reshape(3, 1),
    )

    result = verify_board(manifest, [frame])

    assert result.distance_mm_mean is None
    assert result.tag_side_px_mean > 0.0


def test_load_intrinsics_reads_calibration_json(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"camera_matrix": [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],'
        ' "distortion_coefficients": [-0.1, 0.01, 0.0, 0.0, 0.0]}',
        encoding="utf-8",
    )

    matrix, distortion = load_intrinsics(path)

    assert matrix[0][0] == pytest.approx(900.0)
    assert distortion.shape == (1, 5)

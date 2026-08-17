from pathlib import Path

import cv2
import numpy as np
import pytest

from tagcal.board import AprilGridBoard
from tagcal.calibration import CameraCalibrator
from tagcal.models import (
    CalibrationSpec,
    FrameMetrics,
    KeyframeRecord,
    PatternSpec,
    SelectionReport,
)


def _synthetic_report() -> tuple[SelectionReport, np.ndarray]:
    rng = np.random.default_rng(7)
    width, height = 1280, 720
    expected_matrix = np.array(
        [[920.0, 0.0, 638.0], [0.0, 900.0, 356.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected_distortion = np.array(
        [-0.08, 0.025, 0.0008, -0.0005, -0.006],
        dtype=np.float64,
    )

    board = AprilGridBoard(
        PatternSpec(columns=6, rows=4, tag_size_mm=30.0, gap_mm=7.0)
    )
    object_points = np.concatenate(board.cv_board.getObjPoints(), axis=0).astype(np.float32)
    records: list[KeyframeRecord] = []

    for index in range(22):
        rotation = np.array(
            [
                rng.uniform(-0.38, 0.38),
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.28, 0.28),
            ],
            dtype=np.float64,
        ).reshape(3, 1)
        translation = np.array(
            [
                rng.uniform(-85.0, 85.0),
                rng.uniform(-55.0, 55.0),
                rng.uniform(520.0, 850.0),
            ],
            dtype=np.float64,
        ).reshape(3, 1)
        image_points, _ = cv2.projectPoints(
            object_points,
            rotation,
            translation,
            expected_matrix,
            expected_distortion,
        )
        image_points = image_points.reshape(-1, 2).astype(np.float64)
        image_points += rng.normal(0.0, 0.12, image_points.shape)
        records.append(
            KeyframeRecord(
                image_path=f"keyframes/synthetic_{index:03d}.jpg",
                frame_index=index,
                timestamp_seconds=index / 6.0,
                marker_ids=board.ids.astype(int).tolist(),
                object_points=object_points.astype(float).tolist(),
                image_points=image_points.astype(float).tolist(),
                metrics=FrameMetrics(
                    detected_tags=board.marker_count,
                    detected_points=len(object_points),
                    sharpness=500.0,
                    board_coverage=0.15,
                    center_x=0.5,
                    center_y=0.5,
                    quality_score=0.95,
                    descriptor=[0.5, 0.5, 0.4, 0.0, 0.0, 0.0, 1.0],
                ),
            )
        )

    report = SelectionReport(
        video_path="synthetic.mp4",
        image_width=width,
        image_height=height,
        source_fps=30.0,
        total_frames=660,
        sampled_frames=132,
        valid_candidates=22,
        selected=records,
        rejected_summary={},
    )
    return report, expected_matrix


@pytest.mark.parametrize(
    ("rational_model", "coefficient_count", "distortion_model"),
    [
        (False, 5, "plumb_bob"),
        (True, 8, "rational_polynomial"),
    ],
)
def test_synthetic_intrinsic_calibration(
    tmp_path: Path,
    rational_model: bool,
    coefficient_count: int,
    distortion_model: str,
) -> None:
    report, expected_matrix = _synthetic_report()
    artifacts = CameraCalibrator(
        CalibrationSpec(
            min_views=12,
            max_view_error_px=1.0,
            rational_model=rational_model,
        )
    ).calibrate(report, tmp_path)
    result = artifacts.result
    actual_matrix = np.asarray(result.camera_matrix)

    assert result.rms_reprojection_error_px < 0.35
    assert abs(actual_matrix[0, 0] - expected_matrix[0, 0]) / expected_matrix[0, 0] < 0.03
    assert abs(actual_matrix[1, 1] - expected_matrix[1, 1]) / expected_matrix[1, 1] < 0.03
    assert abs(actual_matrix[0, 2] - expected_matrix[0, 2]) < 15.0
    assert abs(actual_matrix[1, 2] - expected_matrix[1, 2]) < 15.0
    assert len(result.distortion_coefficients) == coefficient_count
    assert result.distortion_model == distortion_model
    assert artifacts.json_path.exists()
    assert artifacts.opencv_yaml_path.exists()
    assert artifacts.ros_yaml_path.exists()
    assert artifacts.report_path.exists()

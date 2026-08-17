"""Synthetic camera tests for the real FaceMesh PnP geometry path."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from headcoupled_display.face_model import (
    HEAD_TO_OPENCV,
    LEFT_IRIS_CENTRE,
    RIGHT_IRIS_CENTRE,
    canonical_face_model,
    load_personal_face_model,
)
from headcoupled_display.models import CameraIntrinsics, CameraMount, HardwareProfile, UserProfile
from headcoupled_display.tracking import HeadPoseEstimator


@pytest.fixture
def hardware() -> HardwareProfile:
    return HardwareProfile(
        profile_id="test-hardware",
        provenance="measured",
        display={"pixel_width": 1920, "pixel_height": 1080, "width_m": 0.53, "height_m": 0.298},
        camera=CameraIntrinsics(
            image_width_px=1280,
            image_height_px=720,
            camera_matrix=((900.0, 0.0, 640.0), (0.0, 902.0, 360.0), (0.0, 0.0, 1.0)),
        ),
        camera_mount=CameraMount(height_above_center_m=0.0),
        camera_to_display_matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def synthetic_landmarks(estimator: HeadPoseEstimator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rvec = np.array([[0.08], [-0.15], [0.04]], dtype=np.float64)
    tvec = np.array([[0.015], [-0.02], [0.72]], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        estimator.face_model.pnp_points_opencv_m[estimator.LANDMARK_INDICES],
        rvec,
        tvec,
        estimator.camera_matrix,
        estimator.distortion,
    )
    landmarks = np.zeros((478, 2), dtype=np.float64)
    landmarks[estimator.LANDMARK_INDICES] = projected.reshape(-1, 2)
    return landmarks, rvec, tvec


def test_sq_pnp_recovers_eyes_and_forward_in_front_of_camera(hardware: HardwareProfile) -> None:
    user = UserProfile(
        left_eye_center_head_m=(-0.03, 0.02, 0.01),
        right_eye_center_head_m=(0.03, 0.02, 0.01),
    )
    estimator = HeadPoseEstimator(hardware, user)
    landmarks, rvec, tvec = synthetic_landmarks(estimator)

    left, right, cyclopean, forward = estimator.estimate(landmarks)

    rotation, _ = cv2.Rodrigues(rvec)
    expected_left = rotation @ (estimator.left_eye_head_m * HEAD_TO_OPENCV) + tvec.reshape(3)
    expected_right = rotation @ (estimator.right_eye_head_m * HEAD_TO_OPENCV) + tvec.reshape(3)
    expected_forward = rotation @ (np.array([0.0, 0.0, 1.0]) * HEAD_TO_OPENCV)
    np.testing.assert_allclose(left, expected_left, atol=1e-6)
    np.testing.assert_allclose(right, expected_right, atol=1e-6)
    np.testing.assert_allclose(cyclopean, 0.5 * (expected_left + expected_right), atol=1e-6)
    np.testing.assert_allclose(forward, expected_forward / np.linalg.norm(expected_forward), atol=1e-6)


def test_solution_behind_camera_is_rejected(hardware: HardwareProfile, monkeypatch: pytest.MonkeyPatch) -> None:
    estimator = HeadPoseEstimator(hardware, UserProfile())
    landmarks, _, _ = synthetic_landmarks(estimator)
    monkeypatch.setattr(
        "headcoupled_display.tracking.cv2.solvePnP",
        lambda *args, **kwargs: (True, np.zeros((3, 1)), np.array([[0.0], [0.0], [-1.0]])),
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        estimator.estimate(landmarks)


def write_personal_pcd(path: Path) -> np.ndarray:
    points_head_m = np.zeros((478, 3), dtype=np.float64)
    points_head_m[:468] = canonical_face_model().points_head_m
    points_head_m[LEFT_IRIS_CENTRE] = (-0.031, 0.026, 0.030)
    points_head_m[RIGHT_IRIS_CENTRE] = (0.033, 0.027, 0.031)
    points_opencv_mm = points_head_m * HEAD_TO_OPENCV * 1000.0
    path.write_text(
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "WIDTH 478\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 478\nDATA ascii\n"
        + "".join(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in points_opencv_mm),
        encoding="ascii",
    )
    return points_head_m


def test_personal_mesh_drives_iris_eye_positions(hardware: HardwareProfile, tmp_path: Path) -> None:
    pcd = tmp_path / "shape.pcd"
    points_head_m = write_personal_pcd(pcd)
    estimator = HeadPoseEstimator(hardware, UserProfile(face_model_path=str(pcd)))
    landmarks, rvec, tvec = synthetic_landmarks(estimator)

    left, right, cyclopean, _ = estimator.estimate(landmarks)

    rotation, _ = cv2.Rodrigues(rvec)
    expected = [
        rotation @ (points_head_m[index] * HEAD_TO_OPENCV) + tvec.reshape(3)
        for index in (LEFT_IRIS_CENTRE, RIGHT_IRIS_CENTRE)
    ]
    np.testing.assert_allclose(left, expected[0], atol=1e-6)
    np.testing.assert_allclose(right, expected[1], atol=1e-6)
    np.testing.assert_allclose(cyclopean, 0.5 * (expected[0] + expected[1]), atol=1e-6)


def test_face_model_rejects_unflipped_head_coordinates(tmp_path: Path) -> None:
    pcd = tmp_path / "wrong-frame.pcd"
    points_head_m = write_personal_pcd(pcd)
    wrong_frame_mm = points_head_m * 1000.0
    text = pcd.read_text(encoding="ascii")
    pcd.write_text(text.split("DATA ascii\n", maxsplit=1)[0] + "DATA ascii\n" + "".join(
        f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in wrong_frame_mm
    ), encoding="ascii")

    with pytest.raises(ValueError, match="canonical metric frame"):
        load_personal_face_model(pcd)

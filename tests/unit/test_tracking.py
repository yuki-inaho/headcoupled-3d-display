"""Synthetic camera tests for the real FaceMesh PnP geometry path."""

from __future__ import annotations

import json
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
from headcoupled_display.filtering import Se3EKF, Se3PoseFilter
from headcoupled_display.models import CameraIntrinsics, CameraMount, HardwareProfile, UserProfile
from headcoupled_display.tracking import (
    FaceMeshFrameSource,
    FaceMeshInputFrame,
    FaceMeshObservation,
    FaceMeshPoseProvider,
    FaceMeshReplayProvider,
    HeadPoseEstimate,
    HeadPoseEstimator,
)


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
    np.testing.assert_allclose(
        forward, expected_forward / np.linalg.norm(expected_forward), atol=1e-6
    )


def test_solution_behind_camera_is_rejected(
    hardware: HardwareProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    pcd.write_text(
        text.split("DATA ascii\n", maxsplit=1)[0]
        + "DATA ascii\n"
        + "".join(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in wrong_frame_mm),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="canonical metric frame"):
        load_personal_face_model(pcd)


def test_face_model_points_are_immutable_after_runtime_type_validation() -> None:
    model = canonical_face_model()

    assert not model.points_head_m.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        model.points_head_m[0, 0] = 1.0


def test_estimate_pose_returns_reprojection_quality(hardware: HardwareProfile) -> None:
    estimator = HeadPoseEstimator(hardware, UserProfile())
    landmarks, _, _ = synthetic_landmarks(estimator)
    estimate = estimator.estimate_pose(landmarks, timestamp_unix_ns=1_000_000)
    assert isinstance(estimate, HeadPoseEstimate)
    # A noise-free synthetic projection reproduces the 12 points exactly.
    assert estimate.reprojection_rms_px < 1e-6
    assert estimate.inlier_count == len(estimator.LANDMARK_INDICES)
    assert estimate.timestamp_unix_ns == 1_000_000
    # T_S_H must be a proper rigid transform.
    rotation = estimate.T_S_H[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)


def test_eyes_from_pose_preserves_ipd(hardware: HardwareProfile) -> None:
    estimator = HeadPoseEstimator(hardware, UserProfile())
    landmarks, rvec, tvec = synthetic_landmarks(estimator)
    estimator.estimate_pose(landmarks)  # warm the PnP path; result not needed here
    rotation, _ = cv2.Rodrigues(rvec)
    # An arbitrary rotated/translated pose must keep the inter-ocular distance fixed.
    big = np.eye(4)
    big[:3, :3] = rotation
    big[:3, 3] = tvec.reshape(3) + np.array([0.1, -0.05, 0.2])
    left, right, _, _ = estimator.eyes_from_pose(big)
    expected_ipd = float(np.linalg.norm(estimator.left_eye_head_m - estimator.right_eye_head_m))
    assert np.isclose(np.linalg.norm(left - right), expected_ipd, atol=1e-12)


class _ScriptedSource:
    """An in-memory FaceMeshFrameSource that serves a queue of landmark sets."""

    def __init__(self, landmark_sets: list[np.ndarray]) -> None:
        self._sets = landmark_sets
        self._index = 0
        self.closed = False

    def next_frame(self) -> FaceMeshInputFrame:
        landmarks = self._sets[self._index % len(self._sets)]
        self._index += 1
        return FaceMeshInputFrame(
            frame_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
            faces=(FaceMeshObservation(score=0.99, landmarks_xy=landmarks),),
            label="test",
            frame_index=self._index - 1,
        )

    def close(self) -> None:
        self.closed = True


def test_provider_emits_filtered_pose_fields(hardware: HardwareProfile) -> None:
    estimator = HeadPoseEstimator(hardware, UserProfile())
    landmarks, _, _ = synthetic_landmarks(estimator)
    provider = FaceMeshPoseProvider(
        hardware,
        UserProfile(),
        source="replay",
        frame_source=_ScriptedSource([landmarks]),
    )
    state, _ = provider.sample()
    provider.close()
    assert state.tracking_valid is True
    assert state.cyclopean_eye_display_m[2] > 0.0
    assert state.head_position_display_m is not None
    assert state.head_orientation_display_xyzw is not None
    assert state.reprojection_rms_px is not None
    assert state.reprojection_rms_px < 1e-6


def test_provider_default_filter_is_ekf(hardware: HardwareProfile) -> None:
    provider = FaceMeshPoseProvider(
        hardware,
        UserProfile(),
        source="replay",
        frame_source=_ScriptedSource([np.zeros((478, 2), dtype=np.float64)]),
    )
    assert isinstance(provider._filter, Se3EKF)
    provider.close()


def test_provider_pose_filter_ema_switches_filter(hardware: HardwareProfile) -> None:
    ema_hardware = hardware.model_copy(
        update={"quality_metrics": {**hardware.quality_metrics, "pose_filter": "ema"}}
    )
    provider = FaceMeshPoseProvider(
        ema_hardware,
        UserProfile(),
        source="replay",
        frame_source=_ScriptedSource([np.zeros((478, 2), dtype=np.float64)]),
    )
    assert isinstance(provider._filter, Se3PoseFilter)
    provider.close()


def test_provider_pose_filter_ekf_switches_filter(hardware: HardwareProfile) -> None:
    ekf_hardware = hardware.model_copy(
        update={"quality_metrics": {**hardware.quality_metrics, "pose_filter": "ekf"}}
    )
    provider = FaceMeshPoseProvider(
        ekf_hardware,
        UserProfile(),
        source="replay",
        frame_source=_ScriptedSource([np.zeros((478, 2), dtype=np.float64)]),
    )
    assert isinstance(provider._filter, Se3EKF)
    provider.close()


def test_provider_pose_filter_unknown_raises(hardware: HardwareProfile) -> None:
    bad_hardware = hardware.model_copy(
        update={"quality_metrics": {**hardware.quality_metrics, "pose_filter": "kalman"}}
    )
    with pytest.raises(ValueError, match="pose_filter"):
        FaceMeshPoseProvider(
            bad_hardware,
            UserProfile(),
            source="replay",
            frame_source=_ScriptedSource([np.zeros((478, 2), dtype=np.float64)]),
        )


def test_provider_outlier_rejection_keeps_last_good(hardware: HardwareProfile) -> None:
    estimator = HeadPoseEstimator(hardware, UserProfile())
    landmarks, _, _ = synthetic_landmarks(estimator)
    bad_landmarks = landmarks.copy()
    bad_landmarks[estimator.LANDMARK_INDICES] += 200.0
    provider = FaceMeshPoseProvider(
        hardware,
        UserProfile(),
        source="replay",
        frame_source=_ScriptedSource([landmarks, bad_landmarks]),
    )
    good_state, _ = provider.sample()
    rejected_state, _ = provider.sample()
    provider.close()
    assert rejected_state.tracking_valid is False
    assert rejected_state.diagnostics.get("rejection") in {"velocity_gate", "reprojection_rms"}
    assert np.allclose(
        rejected_state.cyclopean_eye_display_m,
        good_state.cyclopean_eye_display_m,
        atol=1e-9,
    )


def test_recorded_landmarks_and_video_replay_through_metric_pose(
    hardware: HardwareProfile, tmp_path: Path
) -> None:
    estimator = HeadPoseEstimator(hardware, UserProfile())
    landmarks, _, _ = synthetic_landmarks(estimator)
    landmark_path = tmp_path / "recording.landmarks.json"
    landmark_path.write_text(
        json.dumps(
            [
                {
                    "frame": frame_index,
                    "faces": [{"score": 0.99, "landmarks": landmarks.tolist()}],
                }
                for frame_index in range(2)
            ]
        ),
        encoding="utf-8",
    )
    video_path = tmp_path / "recording.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1280, 720))
    assert writer.isOpened()
    for value in (40, 80):
        writer.write(np.full((720, 1280, 3), value, dtype=np.uint8))
    writer.release()

    provider = FaceMeshReplayProvider(
        hardware, UserProfile(), landmarks_path=landmark_path, video_path=video_path
    )
    try:
        first_state, first_frame = provider.sample()
        second_state, second_frame = provider.sample()
        looped_state, _ = provider.sample()
    finally:
        provider.close()

    assert first_state.source == "replay"
    assert first_state.diagnostics["input_frame"] == 0
    assert second_state.diagnostics["input_frame"] == 1
    assert looped_state.diagnostics["input_frame"] == 0
    assert first_state.cyclopean_eye_display_m[2] > 0.0
    assert first_frame.startswith(b"\xff\xd8") and second_frame.startswith(b"\xff\xd8")


def test_metric_pose_provider_accepts_an_independent_frame_source(
    hardware: HardwareProfile,
) -> None:
    landmarks, _, _ = synthetic_landmarks(HeadPoseEstimator(hardware, UserProfile()))

    class InMemoryFrameSource:
        def __init__(self) -> None:
            self.closed = False

        def next_frame(self) -> FaceMeshInputFrame:
            return FaceMeshInputFrame(
                frame_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
                faces=(FaceMeshObservation(score=0.99, landmarks_xy=landmarks),),
                label="test input",
                frame_index=17,
            )

        def close(self) -> None:
            self.closed = True

    input_source = InMemoryFrameSource()
    assert isinstance(input_source, FaceMeshFrameSource)
    provider = FaceMeshPoseProvider(
        hardware, UserProfile(), source="replay", frame_source=input_source
    )
    state, preview = provider.sample()
    provider.close()

    assert state.diagnostics["input_frame"] == 17
    assert preview.startswith(b"\xff\xd8")
    assert input_source.closed

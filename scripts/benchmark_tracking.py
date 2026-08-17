"""Repeatable CPU microbenchmark for the cached head-pose hot path."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from headcoupled_display.models import CameraIntrinsics, CameraMount, HardwareProfile, UserProfile
from headcoupled_display.tracking import HeadPoseEstimator


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        profile_id="benchmark",
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


def _project_landmarks(estimator: HeadPoseEstimator) -> np.ndarray:
    image_points, _ = cv2.projectPoints(
        estimator.face_model.pnp_points_opencv_m[estimator.LANDMARK_INDICES],
        np.array([[0.08], [-0.15], [0.04]], dtype=np.float64),
        np.array([[0.015], [-0.02], [0.72]], dtype=np.float64),
        estimator.camera_matrix,
        estimator.distortion,
    )
    landmarks = np.zeros((468, 2), dtype=np.float64)
    landmarks[estimator.LANDMARK_INDICES] = image_points.reshape(-1, 2)
    return landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5_000)
    arguments = parser.parse_args()
    if arguments.iterations < 1:
        raise ValueError("--iterations must be positive")

    estimator = HeadPoseEstimator(_hardware(), UserProfile())
    landmarks = _project_landmarks(estimator)
    for _ in range(100):
        estimator.estimate(landmarks)

    started = time.perf_counter_ns()
    for _ in range(arguments.iterations):
        estimator.estimate(landmarks)
    elapsed_ns = time.perf_counter_ns() - started
    per_estimate_us = elapsed_ns / arguments.iterations / 1_000.0
    print(f"SQPNP cached pose: {per_estimate_us:.2f} us/estimate ({arguments.iterations} iterations)")


if __name__ == "__main__":
    main()

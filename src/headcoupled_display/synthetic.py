"""Deterministic artificial data for calibration, runtime demos, and tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import sin

import cv2
import numpy as np

from .calibration import compare_to_ground_truth, fit_display_transform
from .geometry import (
    build_camera_to_display_matrix,
    display_target_point,
    invert_rigid_transform,
    normalize,
    transform_direction,
    transform_point,
)
from .models import (
    CalibrationDataset,
    CalibrationResult,
    CalibrationSample,
    CameraMount,
    HardwareProfile,
    TrackingState,
)

TARGET_GRID: tuple[tuple[float, float], ...] = tuple(
    (u, v) for v in (0.2, 0.5, 0.8) for u in (0.2, 0.5, 0.8)
)


def generate_synthetic_calibration_dataset(
    profile: HardwareProfile,
    *,
    seed: int = 20260817,
    angular_noise_deg: float = 0.12,
) -> CalibrationDataset:
    rng = np.random.default_rng(seed)
    camera_to_display = build_camera_to_display_matrix(profile.camera_mount)
    display_to_camera = invert_rigid_transform(camera_to_display)
    base_eye_positions = (
        np.array([-0.085, -0.030, 0.62], dtype=np.float64),
        np.array([0.075, 0.020, 0.69], dtype=np.float64),
        np.array([0.000, 0.055, 0.75], dtype=np.float64),
        np.array([0.040, -0.050, 0.66], dtype=np.float64),
    )
    noise_scale = np.deg2rad(angular_noise_deg)
    samples: list[CalibrationSample] = []
    for target_index, target_uv in enumerate(TARGET_GRID):
        target_display = display_target_point(profile.display, target_uv)
        for position_index, base_eye in enumerate(base_eye_positions):
            eye_display = base_eye + rng.normal(0.0, [0.003, 0.003, 0.005], 3)
            direction_display = normalize(target_display - eye_display)
            eye_camera = transform_point(display_to_camera, eye_display)
            direction_camera = transform_direction(display_to_camera, direction_display)
            noisy_direction = normalize(direction_camera + rng.normal(0.0, noise_scale, 3))
            samples.append(
                CalibrationSample(
                    target_uv=target_uv,
                    cyclopean_eye_camera_m=tuple(float(value) for value in eye_camera),
                    head_forward_camera=tuple(float(value) for value in noisy_direction),
                    confidence=float(rng.uniform(0.90, 1.0)),
                    sample_id=f"t{target_index:02d}-p{position_index:02d}",
                )
            )
    return CalibrationDataset(
        display=profile.display,
        samples=tuple(samples),
        metadata={
            "source": "deterministic_synthetic_head_rays",
            "seed": seed,
            "angular_noise_deg": angular_noise_deg,
            "ground_truth_profile_id": profile.profile_id,
        },
    )


def perturbed_mount_prior(profile: HardwareProfile) -> HardwareProfile:
    mount = profile.camera_mount
    perturbed = CameraMount(
        horizontal_offset_m=mount.horizontal_offset_m + 0.008,
        height_above_center_m=mount.height_above_center_m + 0.018,
        forward_offset_m=mount.forward_offset_m - 0.012,
        pitch_down_deg=mount.pitch_down_deg - 2.5,
        yaw_right_deg=mount.yaw_right_deg + 1.2,
        roll_clockwise_deg=mount.roll_clockwise_deg + 0.8,
    )
    return profile.model_copy(
        update={
            "profile_id": f"{profile.profile_id}-perturbed-prior",
            "camera_mount": perturbed,
            "camera_to_display_matrix": None,
            "notes": (*profile.notes, "Synthetic calibration starts from a deliberately perturbed prior."),
        }
    )


def run_synthetic_calibration(
    ground_truth_profile: HardwareProfile,
    *,
    seed: int = 20260817,
) -> tuple[CalibrationDataset, CalibrationResult]:
    dataset = generate_synthetic_calibration_dataset(ground_truth_profile, seed=seed)
    prior = perturbed_mount_prior(ground_truth_profile)
    result = fit_display_transform(dataset, prior)
    return dataset, compare_to_ground_truth(result, ground_truth_profile)


@dataclass(slots=True)
class SyntheticTrackingProvider:
    """A camera-free provider that emits head motion and a generated JPEG preview."""

    profile: HardwareProfile
    width: int = 640
    height: int = 360
    _sequence: int = 0
    _started_at: float = 0.0

    def __post_init__(self) -> None:
        self._started_at = time.perf_counter()

    def sample(self) -> tuple[TrackingState, bytes]:
        now = time.time()
        t = time.perf_counter() - self._started_at
        eye = np.array(
            [
                0.095 * sin(t * 0.55),
                0.038 * sin(t * 0.73 + 0.4),
                0.67 + 0.035 * sin(t * 0.31),
            ],
            dtype=np.float64,
        )
        left_eye = eye + np.array([-0.032, 0.0, 0.0], dtype=np.float64)
        right_eye = eye + np.array([0.032, 0.0, 0.0], dtype=np.float64)
        forward = normalize(np.array([-eye[0] * 0.25, -eye[1] * 0.25, -1.0]))
        self._sequence += 1
        state = TrackingState(
            sequence=self._sequence,
            timestamp_unix_s=now,
            source="synthetic",
            confidence=0.99,
            cyclopean_eye_display_m=tuple(float(value) for value in eye),
            left_eye_display_m=tuple(float(value) for value in left_eye),
            right_eye_display_m=tuple(float(value) for value in right_eye),
            head_forward_display=tuple(float(value) for value in forward),
            tracking_fps=30.0,
            inference_ms=1.2,
            stable=abs(np.cos(t * 0.55)) < 0.75,
            diagnostics={
                "mode": "artificial",
                "profile_provenance": self.profile.provenance,
                "note": "No physical camera or neural model is used in synthetic mode.",
            },
        )
        return state, self._render_preview(state)

    def close(self) -> None:
        return None

    def _render_preview(self, state: TrackingState) -> bytes:
        image = np.full((self.height, self.width, 3), (20, 25, 34), dtype=np.uint8)
        eye = np.asarray(state.cyclopean_eye_display_m)
        center = (
            int(self.width / 2 - eye[0] * 900),
            int(self.height / 2 - eye[1] * 900),
        )
        cv2.ellipse(image, center, (78, 102), 0, 0, 360, (82, 102, 132), 2, cv2.LINE_AA)
        cv2.circle(image, (center[0] - 31, center[1] - 22), 8, (214, 225, 242), 2)
        cv2.circle(image, (center[0] + 31, center[1] - 22), 8, (214, 225, 242), 2)
        cv2.circle(image, center, 4, (87, 214, 181), -1, cv2.LINE_AA)
        cv2.line(image, (self.width // 2, 0), (self.width // 2, self.height), (46, 58, 76), 1)
        cv2.line(image, (0, self.height // 2), (self.width, self.height // 2), (46, 58, 76), 1)
        cv2.putText(
            image,
            "SYNTHETIC CAMERA / NO BIOMETRIC INPUT",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (87, 214, 181),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"eye S=({eye[0]:+.3f}, {eye[1]:+.3f}, {eye[2]:.3f}) m",
            (18, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (214, 225, 242),
            1,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("failed to encode synthetic preview")
        return encoded.tobytes()

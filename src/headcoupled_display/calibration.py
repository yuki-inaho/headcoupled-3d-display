"""Head-ray/display-target calibration using robust nonlinear least squares."""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import (
    display_target_point,
    invert_rigid_transform,
    matrix_to_tuple,
    mount_summary_from_camera_to_display,
    normalize,
)
from .models import (
    CalibrationDataset,
    CalibrationMetrics,
    CalibrationResult,
    CameraMount,
    HardwareProfile,
)
from .profiles import resolved_camera_to_display


@dataclass(frozen=True, slots=True)
class CalibrationOptions:
    ray_residual_sigma_m: float = 0.004
    translation_prior_sigma_m: float = 0.050
    rotation_prior_sigma_deg: float = 8.0
    robust_loss: str = "soft_l1"
    max_nfev: int = 3000

    def __post_init__(self) -> None:
        if self.ray_residual_sigma_m <= 0:
            raise ValueError("ray_residual_sigma_m must be positive")
        if self.translation_prior_sigma_m <= 0:
            raise ValueError("translation_prior_sigma_m must be positive")
        if self.rotation_prior_sigma_deg <= 0:
            raise ValueError("rotation_prior_sigma_deg must be positive")


def _transform_from_parameters(parameters: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform[:3, 3] = parameters[3:6]
    return transform


def _parameters_from_transform(transform: np.ndarray) -> np.ndarray:
    parameters = np.zeros(6, dtype=np.float64)
    parameters[:3] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    parameters[3:6] = transform[:3, 3]
    return parameters


def _point_to_ray_residual(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    unit_direction = normalize(direction)
    projector = np.eye(3, dtype=np.float64) - np.outer(unit_direction, unit_direction)
    return projector @ (point - origin)


def fit_display_transform(
    dataset: CalibrationDataset,
    prior_profile: HardwareProfile,
    *,
    options: CalibrationOptions | None = None,
) -> CalibrationResult:
    """Estimate display-to-camera pose from known screen targets and measured head rays.

    The profile's mount is used as a weak prior. The returned profile contains an explicit
    camera-to-display matrix, which is authoritative over the human-readable prior fields.
    """

    settings = options or CalibrationOptions()
    prior_camera_to_display = resolved_camera_to_display(prior_profile)
    prior_display_to_camera = invert_rigid_transform(prior_camera_to_display)
    initial = _parameters_from_transform(prior_display_to_camera)
    prior_rotation = prior_display_to_camera[:3, :3]
    prior_translation = prior_display_to_camera[:3, 3]
    rotation_sigma_rad = np.deg2rad(settings.rotation_prior_sigma_deg)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        display_to_camera = _transform_from_parameters(parameters)
        rotation = display_to_camera[:3, :3]
        translation = display_to_camera[:3, 3]
        values: list[float] = []
        for sample in dataset.samples:
            target_display = display_target_point(dataset.display, sample.target_uv)
            target_camera = rotation @ target_display + translation
            origin = np.asarray(sample.cyclopean_eye_camera_m, dtype=np.float64)
            direction = np.asarray(sample.head_forward_camera, dtype=np.float64)
            residual = _point_to_ray_residual(target_camera, origin, direction)
            weight = float(np.sqrt(sample.confidence)) / settings.ray_residual_sigma_m
            values.extend((residual * weight).tolist())

        relative_rotation = Rotation.from_matrix(rotation @ prior_rotation.T).as_rotvec()
        values.extend((relative_rotation / rotation_sigma_rad).tolist())
        values.extend(
            ((translation - prior_translation) / settings.translation_prior_sigma_m).tolist()
        )
        return np.asarray(values, dtype=np.float64)

    optimization = least_squares(
        residuals,
        initial,
        loss=settings.robust_loss,
        f_scale=1.0,
        max_nfev=settings.max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    estimated_display_to_camera = _transform_from_parameters(optimization.x)
    estimated_camera_to_display = invert_rigid_transform(estimated_display_to_camera)
    summary = mount_summary_from_camera_to_display(estimated_camera_to_display)

    distances_m: list[float] = []
    angular_errors_deg: list[float] = []
    for sample in dataset.samples:
        target_display = display_target_point(dataset.display, sample.target_uv)
        target_camera = (
            estimated_display_to_camera[:3, :3] @ target_display
            + estimated_display_to_camera[:3, 3]
        )
        origin = np.asarray(sample.cyclopean_eye_camera_m, dtype=np.float64)
        direction = normalize(np.asarray(sample.head_forward_camera, dtype=np.float64))
        target_direction = normalize(target_camera - origin)
        distances_m.append(
            float(np.linalg.norm(_point_to_ray_residual(target_camera, origin, direction)))
        )
        angular_errors_deg.append(
            degrees(float(np.arccos(np.clip(np.dot(direction, target_direction), -1.0, 1.0))))
        )

    distance_array = np.asarray(distances_m, dtype=np.float64) * 1000.0
    angle_array = np.asarray(angular_errors_deg, dtype=np.float64)
    metrics = CalibrationMetrics(
        sample_count=len(dataset.samples),
        unique_target_count=len({sample.target_uv for sample in dataset.samples}),
        mean_point_to_ray_error_mm=float(np.mean(distance_array)),
        median_point_to_ray_error_mm=float(np.median(distance_array)),
        max_point_to_ray_error_mm=float(np.max(distance_array)),
        mean_angular_error_deg=float(np.mean(angle_array)),
        max_angular_error_deg=float(np.max(angle_array)),
        optimizer_cost=float(optimization.cost),
        optimizer_evaluations=int(optimization.nfev),
        optimizer_success=bool(optimization.success),
        optimizer_message=str(optimization.message),
    )

    measured_mount = CameraMount(
        horizontal_offset_m=summary.horizontal_offset_cm / 100.0,
        height_above_center_m=summary.height_above_center_cm / 100.0,
        forward_offset_m=summary.forward_offset_cm / 100.0,
        pitch_down_deg=summary.pitch_down_deg,
        yaw_right_deg=summary.yaw_right_deg,
        roll_clockwise_deg=summary.roll_clockwise_deg,
    )
    estimated_profile = prior_profile.model_copy(
        update={
            "profile_id": f"{prior_profile.profile_id}-estimated",
            "provenance": "estimated_from_head_targets",
            "camera_mount": measured_mount,
            "camera_to_display_matrix": matrix_to_tuple(estimated_camera_to_display),
            "quality_metrics": {
                **prior_profile.quality_metrics,
                "mean_point_to_ray_error_mm": metrics.mean_point_to_ray_error_mm,
                "max_point_to_ray_error_mm": metrics.max_point_to_ray_error_mm,
                "mean_angular_error_deg": metrics.mean_angular_error_deg,
                "optimizer_success": metrics.optimizer_success,
                "sample_count": metrics.sample_count,
            },
        }
    )
    return CalibrationResult(
        estimated_camera_to_display_matrix=matrix_to_tuple(estimated_camera_to_display),
        mount_summary=summary,
        metrics=metrics,
        estimated_profile=estimated_profile,
    )


def compare_to_ground_truth(
    result: CalibrationResult,
    ground_truth_profile: HardwareProfile,
) -> CalibrationResult:
    ground_truth = resolved_camera_to_display(ground_truth_profile)
    estimated = np.asarray(result.estimated_camera_to_display_matrix, dtype=np.float64)
    translation_error = estimated[:3, 3] - ground_truth[:3, 3]
    relative_rotation = Rotation.from_matrix(estimated[:3, :3] @ ground_truth[:3, :3].T)
    comparison = {
        "translation_error_mm": float(np.linalg.norm(translation_error) * 1000.0),
        "height_error_mm": float(abs(translation_error[1]) * 1000.0),
        "horizontal_error_mm": float(abs(translation_error[0]) * 1000.0),
        "forward_error_mm": float(abs(translation_error[2]) * 1000.0),
        "rotation_error_deg": float(np.rad2deg(relative_rotation.magnitude())),
        "pitch_error_deg": float(
            abs(
                result.mount_summary.pitch_down_deg
                - mount_summary_from_camera_to_display(ground_truth).pitch_down_deg
            )
        ),
    }
    return result.model_copy(update={"comparison_to_ground_truth": comparison})

"""Metric face-model loading and coordinate-boundary validation.

The reconstructed ``shape.pcd`` written by ``facemesh_tracking reconstruct`` stores
478 MediaPipe landmarks in **millimetres** and OpenCV object coordinates: +X right,
+Y down, +Z away from the face.  The display controller keeps its head frame as +X
right, +Y up, +Z forward, and crosses into OpenCV only at the PnP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

CANONICAL_LANDMARK_COUNT = 468
LANDMARK_COUNT_WITH_IRISES = 478
LEFT_IRIS_CENTRE = 468
RIGHT_IRIS_CENTRE = 473

# head frame (+Y up, +Z face-forward) -> OpenCV object frame (+Y down, +Z away).
HEAD_TO_OPENCV = np.array([1.0, -1.0, -1.0], dtype=np.float64)


class FaceModelValidationError(ValueError):
    """Raised when a reconstructed mesh is not a compatible metric face model."""


@dataclass(frozen=True)
class FaceModel:
    """A 468/478-point metric face in the controller's head coordinate frame."""

    points_head_m: np.ndarray
    source: str
    is_personal: bool

    @property
    def pnp_points_opencv_m(self) -> np.ndarray:
        return np.ascontiguousarray(self.points_head_m * HEAD_TO_OPENCV)

    @property
    def left_iris_head_m(self) -> np.ndarray:
        return self.points_head_m[LEFT_IRIS_CENTRE].copy()

    @property
    def right_iris_head_m(self) -> np.ndarray:
        return self.points_head_m[RIGHT_IRIS_CENTRE].copy()


def canonical_face_model() -> FaceModel:
    """Load the bundled average face (canonical source is centimetres, head frame)."""

    resource = resources.files("headcoupled_display.resources").joinpath("canonical_face.npz")
    with resource.open("rb") as handle, np.load(handle) as data:
        points_cm = np.asarray(data["vertices"], dtype=np.float64)
    if points_cm.shape != (CANONICAL_LANDMARK_COUNT, 3):  # pragma: no cover - packaged asset
        raise RuntimeError(f"unexpected canonical face shape: {points_cm.shape}")
    return FaceModel(points_head_m=points_cm / 100.0, source="canonical", is_personal=False)


def _read_ascii_pcd(path: Path) -> np.ndarray:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise FaceModelValidationError(f"cannot read face model {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise FaceModelValidationError(
            f"face model {path} must be an ASCII PCD written by facemesh_tracking"
        ) from exc

    header: dict[str, str] = {}
    data_line = None
    for index, line in enumerate(lines):
        fields = line.split(maxsplit=1)
        if not fields or fields[0].startswith("#"):
            continue
        key = fields[0].upper()
        value = fields[1] if len(fields) == 2 else ""
        header[key] = value
        if key == "DATA":
            data_line = index
            break
    if data_line is None or header.get("DATA", "").lower() != "ascii":
        raise FaceModelValidationError("face model must be an ASCII PCD (DATA ascii)")
    if header.get("FIELDS", "").split() != ["x", "y", "z"]:
        raise FaceModelValidationError("face model must contain exactly PCD fields x y z")
    try:
        expected_count = int(header["POINTS"])
        points_mm = np.asarray(
            [[float(component) for component in line.split()] for line in lines[data_line + 1 :] if line],
            dtype=np.float64,
        )
    except (KeyError, ValueError) as exc:
        raise FaceModelValidationError("face model has an invalid PCD POINTS/data section") from exc
    if points_mm.shape != (expected_count, 3):
        raise FaceModelValidationError(
            f"face model declares {expected_count} points but contains {points_mm.shape}"
        )
    if not np.isfinite(points_mm).all():
        raise FaceModelValidationError("face model contains non-finite coordinates")
    return points_mm


def _kabsch_report(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Return proper-rotation determinant, angle (degrees), and RMS residual (metres)."""

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    residual = (source - source_center) @ rotation.T + target_center - target
    angle_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    )
    return float(np.linalg.det(rotation)), angle_deg, float(np.sqrt(np.mean(residual**2)))


def load_personal_face_model(path: Path) -> FaceModel:
    """Load and validate a 478-point ``shape.pcd`` produced by reconstruction.

    The Kabsch check deliberately validates coordinate convention without applying its
    fitted transform: the PCD's own metric coordinates are the ones that PnP must use.
    It rejects swapped axes, reflections, and wrong units before a live camera session.
    """

    points_opencv_m = _read_ascii_pcd(path) / 1000.0
    if points_opencv_m.shape != (LANDMARK_COUNT_WITH_IRISES, 3):
        raise FaceModelValidationError(
            "a personal face model must be the 478-point shape.pcd so iris centres are available"
        )
    points_head_m = points_opencv_m * HEAD_TO_OPENCV
    canonical = canonical_face_model().points_head_m
    determinant, angle_deg, rms_m = _kabsch_report(
        canonical, points_head_m[:CANONICAL_LANDMARK_COUNT]
    )
    if determinant <= 0.0 or angle_deg > 8.0 or rms_m > 0.015:
        raise FaceModelValidationError(
            "face model is not in the reconstructed canonical metric frame "
            f"(det={determinant:.3f}, rotation={angle_deg:.2f} deg, rms={rms_m * 1000:.2f} mm)"
        )
    return FaceModel(points_head_m=points_head_m, source=str(path), is_personal=True)

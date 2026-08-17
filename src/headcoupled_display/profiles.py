"""Hardware profile loading, normalization, and tagcal interoperability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import yaml

from .geometry import (
    build_camera_to_display_matrix,
    matrix_to_tuple,
    mount_summary_from_camera_to_display,
    tuple_to_matrix,
)
from .models import CameraIntrinsics, HardwareProfile, MountSummary, UserProfile


def resolved_camera_to_display(profile: HardwareProfile) -> np.ndarray:
    if profile.camera_to_display_matrix is not None:
        return tuple_to_matrix(profile.camera_to_display_matrix)
    return build_camera_to_display_matrix(profile.camera_mount)


def profile_with_resolved_matrix(profile: HardwareProfile) -> HardwareProfile:
    if profile.camera_to_display_matrix is not None:
        return profile
    return profile.model_copy(
        update={"camera_to_display_matrix": matrix_to_tuple(resolved_camera_to_display(profile))}
    )


def load_user_profile(path: Path) -> UserProfile:
    """Load a user profile and make its optional personal-mesh path absolute."""

    profile = UserProfile.load(path)
    if profile.face_model_path is None:
        return profile
    face_model_path = Path(profile.face_model_path).expanduser()
    if not face_model_path.is_absolute():
        face_model_path = path.parent / face_model_path
    return profile.model_copy(update={"face_model_path": str(face_model_path.resolve())})


def summarize_profile(profile: HardwareProfile) -> MountSummary:
    return mount_summary_from_camera_to_display(resolved_camera_to_display(profile))


def load_tagcal_calibration(path: Path) -> CameraIntrinsics:
    """Load either tagcal ``calibration.json`` or its OpenCV YAML output."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return _camera_intrinsics_from_tagcal_json(data)
    if suffix in {".yaml", ".yml"}:
        opencv = _try_camera_intrinsics_from_opencv_yaml(path)
        if opencv is not None:
            return opencv
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return _camera_intrinsics_from_yaml(data)
    raise ValueError(f"unsupported calibration file extension: {path.suffix}")


def _try_camera_intrinsics_from_opencv_yaml(path: Path) -> CameraIntrinsics | None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        return None
    try:
        matrix = storage.getNode("camera_matrix").mat()
        if matrix is None:
            return None
        distortion = storage.getNode("distortion_coefficients").mat()
        width = int(storage.getNode("image_width").real())
        height = int(storage.getNode("image_height").real())
        distortion_model = storage.getNode("distortion_model").string() or "plumb_bob"
        rms_node = storage.getNode("rms_reprojection_error_px")
        rms = None if rms_node.empty() else float(rms_node.real())
        return CameraIntrinsics(
            image_width_px=width,
            image_height_px=height,
            camera_matrix=cast(
                tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
                tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(matrix, dtype=np.float64).reshape(3, 3)
                ),
            ),
            distortion_coefficients=tuple(
                float(value)
                for value in np.asarray(
                    [] if distortion is None else distortion,
                    dtype=np.float64,
                ).reshape(-1)
            ),
            distortion_model=distortion_model,
            rms_reprojection_error_px=rms,
        )
    finally:
        storage.release()


def _camera_intrinsics_from_tagcal_json(data: dict[str, Any]) -> CameraIntrinsics:
    root = data.get("calibration", data)
    width = root.get("image_width") or root.get("image_width_px")
    height = root.get("image_height") or root.get("image_height_px")
    matrix = root.get("camera_matrix")
    distortion = root.get("distortion_coefficients", root.get("distortion", []))
    if width is None or height is None or matrix is None:
        raise ValueError("tagcal JSON is missing image size or camera_matrix")
    return CameraIntrinsics(
        image_width_px=int(width),
        image_height_px=int(height),
        camera_matrix=cast(
            tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
            tuple(tuple(float(value) for value in row) for row in matrix),
        ),
        distortion_coefficients=tuple(float(value) for value in np.asarray(distortion).reshape(-1)),
        distortion_model=str(root.get("distortion_model", "plumb_bob")),
        rms_reprojection_error_px=(
            None
            if root.get("rms_reprojection_error_px") is None
            else float(root["rms_reprojection_error_px"])
        ),
    )


def _camera_intrinsics_from_yaml(data: dict[str, Any]) -> CameraIntrinsics:
    matrix_node = data.get("camera_matrix")
    distortion_node = data.get("distortion_coefficients", [])
    matrix_data = matrix_node.get("data") if isinstance(matrix_node, dict) else matrix_node
    distortion_data = (
        distortion_node.get("data") if isinstance(distortion_node, dict) else distortion_node
    )
    matrix = np.asarray(matrix_data, dtype=np.float64).reshape(3, 3)
    return CameraIntrinsics(
        image_width_px=int(data["image_width"]),
        image_height_px=int(data["image_height"]),
        camera_matrix=cast(
            tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
            tuple(tuple(float(value) for value in row) for row in matrix),
        ),
        distortion_coefficients=tuple(
            float(value) for value in np.asarray(distortion_data, dtype=np.float64).reshape(-1)
        ),
        distortion_model=str(data.get("distortion_model", "plumb_bob")),
        rms_reprojection_error_px=(
            None
            if data.get("rms_reprojection_error_px") is None
            else float(data["rms_reprojection_error_px"])
        ),
    )

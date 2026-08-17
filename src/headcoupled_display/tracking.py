"""Tracking provider contracts and the optional FaceMesh live-camera adapter."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .face_model import HEAD_TO_OPENCV, FaceModel, canonical_face_model, load_personal_face_model
from .geometry import normalize, transform_direction, transform_point
from .models import HardwareProfile, TrackingState, UserProfile
from .profiles import resolved_camera_to_display


@runtime_checkable
class TrackingProvider(Protocol):
    def sample(self) -> tuple[TrackingState, bytes]: ...

    def close(self) -> None: ...


class FaceMeshUnavailableError(RuntimeError):
    pass


class HeadPoseEstimator:
    """Solve a metric dense-landmark PnP pose and derive pupil positions.

    ``SOLVEPNP_SQPNP`` has no fragile initial estimate; cheirality is then enforced so a
    numerically successful mirrored solution cannot put the face behind the camera.
    """

    LANDMARK_INDICES = np.array(
        [1, 6, 33, 133, 362, 263, 61, 291, 199, 168, 94, 4], dtype=np.int32
    )

    def __init__(self, hardware: HardwareProfile, user: UserProfile) -> None:
        self.hardware = hardware
        self.user = user
        self.camera_matrix = np.asarray(hardware.camera.camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(
            hardware.camera.distortion_coefficients, dtype=np.float64
        ).reshape(-1, 1)
        self.camera_to_display = resolved_camera_to_display(hardware)
        self.face_model: FaceModel = (
            load_personal_face_model(Path(user.face_model_path))
            if user.face_model_path is not None
            else canonical_face_model()
        )
        if self.face_model.is_personal:
            self.left_eye_head_m = self.face_model.left_iris_head_m
            self.right_eye_head_m = self.face_model.right_iris_head_m
        else:
            self.left_eye_head_m = np.asarray(user.left_eye_center_head_m, dtype=np.float64)
            self.right_eye_head_m = np.asarray(user.right_eye_center_head_m, dtype=np.float64)
        self.cyclopean_eye_head_m = 0.5 * (self.left_eye_head_m + self.right_eye_head_m)

    def estimate(
        self,
        landmarks_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        landmarks = np.asarray(landmarks_xy, dtype=np.float64)
        if landmarks.ndim != 2 or landmarks.shape[1] < 2 or len(landmarks) <= int(self.LANDMARK_INDICES.max()):
            raise ValueError("FaceMesh landmarks must contain the 468 base landmarks")
        image_points = np.ascontiguousarray(landmarks[self.LANDMARK_INDICES, :2])
        if not np.isfinite(image_points).all():
            raise ValueError("FaceMesh landmarks contain non-finite image coordinates")
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.face_model.pnp_points_opencv_m[self.LANDMARK_INDICES],
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success or translation_vector.reshape(3)[2] <= 0.0:
            raise RuntimeError("head pose PnP did not converge")
        rotation_camera_head, _ = cv2.Rodrigues(rotation_vector)
        translation_camera_head = translation_vector.reshape(3)

        def head_to_camera(point_head: np.ndarray) -> np.ndarray:
            return (
                rotation_camera_head @ (np.asarray(point_head) * HEAD_TO_OPENCV)
                + translation_camera_head
            )

        left_camera = head_to_camera(self.left_eye_head_m)
        right_camera = head_to_camera(self.right_eye_head_m)
        cyclopean_camera = head_to_camera(self.cyclopean_eye_head_m)
        forward_camera = normalize(
            rotation_camera_head
            @ (np.asarray(self.user.neutral_forward_axis_head) * HEAD_TO_OPENCV)
        )

        left_display = transform_point(self.camera_to_display, left_camera)
        right_display = transform_point(self.camera_to_display, right_camera)
        cyclopean_display = transform_point(self.camera_to_display, cyclopean_camera)
        forward_display = transform_direction(self.camera_to_display, forward_camera)
        return left_display, right_display, cyclopean_display, forward_display


class FaceMeshTrackingProvider:
    """Adapter for the uploaded ``facemesh_tracking`` package.

    The uploaded project has a separate Python/CUDA constraint set. It can be installed in
    the same environment where compatible, or its ``src`` directory can be supplied via
    ``FACEMESH_TRACKING_SOURCE``. Synthetic mode remains the reproducible default.
    """

    def __init__(
        self,
        hardware: HardwareProfile,
        user: UserProfile,
        *,
        camera_index: int = 0,
        backend: str = "cpu",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        source_path = os.getenv("FACEMESH_TRACKING_SOURCE")
        if source_path:
            sys.path.insert(0, str(Path(source_path).expanduser().resolve()))
        try:
            from facemesh_tracking.pipeline import FaceMeshPipeline
            from facemesh_tracking.runtime import Backend
        except Exception as exc:  # pragma: no cover - optional GPU integration
            raise FaceMeshUnavailableError(
                "facemesh_tracking is unavailable. Use synthetic mode or install the "
                "uploaded face tracking project and set FACEMESH_TRACKING_SOURCE to its src/."
            ) from exc

        self._pipeline = FaceMeshPipeline.create(backend=Backend(backend))
        self._capture = cv2.VideoCapture(camera_index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            raise RuntimeError(f"unable to open camera index {camera_index}")
        self._estimator = HeadPoseEstimator(hardware, user)
        self._sequence = 0
        self._last_timestamp = time.perf_counter()
        self._fps_ema = 0.0
        self._last_eye = np.array([0.0, 0.0, 0.65], dtype=np.float64)

    def sample(self) -> tuple[TrackingState, bytes]:  # pragma: no cover - hardware dependent
        started = time.perf_counter()
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("camera frame capture failed")
        result = self._pipeline.process(frame)
        confidence = 0.0
        stable = False
        left = self._last_eye + np.array([-0.032, 0.0, 0.0])
        right = self._last_eye + np.array([0.032, 0.0, 0.0])
        forward = np.array([0.0, 0.0, -1.0])
        diagnostics: dict[str, object] = {"face_count": len(result.faces)}

        if result.faces:
            face = max(result.faces, key=lambda value: value.score)
            confidence = float(face.score)
            left, right, eye, forward = self._estimator.estimate(face.xy)
            movement = float(np.linalg.norm(eye - self._last_eye))
            stable = movement < 0.004 and confidence >= 0.75
            self._last_eye = eye
            diagnostics["landmark_count"] = int(face.points.shape[0])
            diagnostics["eye_movement_m"] = movement
            for point in face.xy[::8]:
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 1, (90, 220, 170), -1)
        else:
            eye = self._last_eye

        now_perf = time.perf_counter()
        interval = max(now_perf - self._last_timestamp, 1e-6)
        instant_fps = 1.0 / interval
        self._fps_ema = instant_fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * instant_fps
        self._last_timestamp = now_perf
        inference_ms = (now_perf - started) * 1000.0
        self._sequence += 1
        state = TrackingState(
            sequence=self._sequence,
            timestamp_unix_s=time.time(),
            source="facemesh",
            confidence=confidence,
            cyclopean_eye_display_m=tuple(float(value) for value in eye),
            left_eye_display_m=tuple(float(value) for value in left),
            right_eye_display_m=tuple(float(value) for value in right),
            head_forward_display=tuple(float(value) for value in forward),
            tracking_fps=float(self._fps_ema),
            inference_ms=float(inference_ms),
            stable=stable,
            diagnostics=diagnostics,
        )
        cv2.putText(
            frame,
            f"FaceMesh {self._fps_ema:.1f} fps / {confidence:.2f}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (90, 220, 170),
            2,
            cv2.LINE_AA,
        )
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not encoded_ok:
            raise RuntimeError("failed to encode camera preview")
        return state, encoded.tobytes()

    def close(self) -> None:  # pragma: no cover - hardware dependent
        self._capture.release()

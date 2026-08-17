"""Tracking provider contracts and the optional FaceMesh live-camera adapter."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .face_model import HEAD_TO_OPENCV, FaceModel, canonical_face_model, load_personal_face_model
from .geometry import normalize
from .models import HardwareProfile, TrackingState, UserProfile
from .profiles import resolved_camera_to_display


@runtime_checkable
class TrackingProvider(Protocol):
    def sample(self) -> tuple[TrackingState, bytes]: ...

    def close(self) -> None: ...


class FaceMeshUnavailableError(RuntimeError):
    pass


class _FaceObservation(Protocol):
    score: float
    xy: np.ndarray
    points: np.ndarray


class _FaceMeshResult(Protocol):
    faces: Sequence[_FaceObservation]


@dataclass(frozen=True)
class _PoseMeasurement:
    left: np.ndarray
    right: np.ndarray
    cyclopean: np.ndarray
    forward: np.ndarray
    confidence: float
    face_count: int
    landmark_count: int
    movement_m: float


class HeadPoseEstimator:
    """Solve a metric dense-landmark PnP pose and derive pupil positions.

    ``SOLVEPNP_SQPNP`` has no fragile initial estimate; cheirality is then enforced so a
    numerically successful mirrored solution cannot put the face behind the camera.
    """

    LANDMARK_INDICES = np.array(
        [1, 6, 33, 133, 362, 263, 61, 291, 199, 168, 94, 4], dtype=np.int32
    )

    def __init__(self, hardware: HardwareProfile, user: UserProfile) -> None:
        self.camera_matrix = np.asarray(hardware.camera.camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(
            hardware.camera.distortion_coefficients, dtype=np.float64
        ).reshape(-1, 1)
        camera_to_display = resolved_camera_to_display(hardware)
        self._display_rotation = np.ascontiguousarray(camera_to_display[:3, :3])
        self._display_translation = np.ascontiguousarray(camera_to_display[:3, 3])
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
        self._pnp_object_points = np.ascontiguousarray(
            self.face_model.pnp_points_opencv_m[self.LANDMARK_INDICES]
        )
        self._left_eye_opencv_m = self.left_eye_head_m * HEAD_TO_OPENCV
        self._right_eye_opencv_m = self.right_eye_head_m * HEAD_TO_OPENCV
        self._cyclopean_eye_opencv_m = self.cyclopean_eye_head_m * HEAD_TO_OPENCV
        self._forward_axis_opencv = np.asarray(user.neutral_forward_axis_head) * HEAD_TO_OPENCV

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
            self._pnp_object_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success or translation_vector.reshape(3)[2] <= 0.0:
            raise RuntimeError("head pose PnP did not converge")
        rotation_camera_head, _ = cv2.Rodrigues(rotation_vector)
        translation_camera_head = translation_vector.reshape(3)

        left_camera = rotation_camera_head @ self._left_eye_opencv_m + translation_camera_head
        right_camera = rotation_camera_head @ self._right_eye_opencv_m + translation_camera_head
        cyclopean_camera = rotation_camera_head @ self._cyclopean_eye_opencv_m + translation_camera_head
        forward_camera = rotation_camera_head @ self._forward_axis_opencv

        left_display = self._display_rotation @ left_camera + self._display_translation
        right_display = self._display_rotation @ right_camera + self._display_translation
        cyclopean_display = self._display_rotation @ cyclopean_camera + self._display_translation
        forward_display = normalize(self._display_rotation @ forward_camera)
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
        camera_device: str = "/dev/video0",
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
        source = int(camera_device) if camera_device.isdecimal() else camera_device
        self._capture = cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            raise RuntimeError(f"unable to open camera device {camera_device!r}")
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
        measurement = self._measure_result(result, frame)
        state = self._build_state(measurement, started)
        return state, self._encode_preview(frame, measurement)

    def _measure_result(
        self, result: _FaceMeshResult, frame: np.ndarray
    ) -> _PoseMeasurement:
        if not result.faces:
            return self._fallback_measurement(len(result.faces))
        face = max(result.faces, key=lambda value: value.score)
        left, right, eye, forward = self._estimator.estimate(face.xy)
        movement = float(np.linalg.norm(eye - self._last_eye))
        self._last_eye = eye
        self._draw_landmarks(frame, face.xy)
        return _PoseMeasurement(
            left=left,
            right=right,
            cyclopean=eye,
            forward=forward,
            confidence=float(face.score),
            face_count=len(result.faces),
            landmark_count=int(face.points.shape[0]),
            movement_m=movement,
        )

    def _fallback_measurement(self, face_count: int) -> _PoseMeasurement:
        eye = self._last_eye
        return _PoseMeasurement(
            left=eye + np.array([-0.032, 0.0, 0.0]),
            right=eye + np.array([0.032, 0.0, 0.0]),
            cyclopean=eye,
            forward=np.array([0.0, 0.0, -1.0]),
            confidence=0.0,
            face_count=face_count,
            landmark_count=0,
            movement_m=0.0,
        )

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, landmarks_xy: np.ndarray) -> None:
        for point in landmarks_xy[::8]:
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 1, (90, 220, 170), -1)

    def _build_state(self, measurement: _PoseMeasurement, started: float) -> TrackingState:
        now = time.perf_counter()
        self._fps_ema = self._smoothed_fps(now)
        self._sequence += 1
        stable = measurement.movement_m < 0.004 and measurement.confidence >= 0.75
        diagnostics: dict[str, object] = {"face_count": measurement.face_count}
        if measurement.landmark_count:
            diagnostics.update(
                landmark_count=measurement.landmark_count,
                eye_movement_m=measurement.movement_m,
            )
        return TrackingState(
            sequence=self._sequence,
            timestamp_unix_s=time.time(),
            source="facemesh",
            confidence=measurement.confidence,
            cyclopean_eye_display_m=tuple(float(value) for value in measurement.cyclopean),
            left_eye_display_m=tuple(float(value) for value in measurement.left),
            right_eye_display_m=tuple(float(value) for value in measurement.right),
            head_forward_display=tuple(float(value) for value in measurement.forward),
            tracking_fps=self._fps_ema,
            inference_ms=(now - started) * 1000.0,
            stable=stable,
            diagnostics=diagnostics,
        )

    def _smoothed_fps(self, now: float) -> float:
        interval = max(now - self._last_timestamp, 1e-6)
        self._last_timestamp = now
        instant_fps = 1.0 / interval
        self._fps_ema = instant_fps if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * instant_fps
        return self._fps_ema

    def _encode_preview(self, frame: np.ndarray, measurement: _PoseMeasurement) -> bytes:
        cv2.putText(
            frame,
            f"FaceMesh {self._fps_ema:.1f} fps / {measurement.confidence:.2f}",
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
        return encoded.tobytes()

    def close(self) -> None:  # pragma: no cover - hardware dependent
        self._capture.release()

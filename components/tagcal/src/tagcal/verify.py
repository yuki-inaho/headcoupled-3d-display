"""End-to-end scale check: point the calibrated camera at the board and measure it.

Calibration can converge to a low reprojection error while every distance it
produces is wrong by a constant factor, because a wrong tag size is absorbed
entirely into translation. The only way to catch that is to compare an estimated
distance against a tape measure, which is what this module supports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from tagcal.board import AprilGridBoard
from tagcal.cvtypes import as_float64, as_uint8
from tagcal.detection import AprilTagDetector
from tagcal.models import CaptureSpec, PatternManifest


@dataclass(frozen=True, slots=True)
class VerificationResult:
    frames_used: int
    frames_attempted: int
    detected_tags_mean: float
    tag_side_px_mean: float
    tag_side_px_std: float
    distance_mm_mean: float | None
    distance_mm_std: float | None
    tag_size_mm: float

    def describe(self) -> list[str]:
        lines = [
            f"使用フレーム   : {self.frames_used} / {self.frames_attempted}",
            f"検出タグ数     : 平均 {self.detected_tags_mean:.1f}",
            f"タグ辺長(画像) : {self.tag_side_px_mean:.2f} px (std {self.tag_side_px_std:.2f})",
            f"タグ実寸       : {self.tag_size_mm:.4f} mm",
        ]
        if self.distance_mm_mean is not None and self.distance_mm_std is not None:
            lines.append(
                f"推定距離       : {self.distance_mm_mean:.1f} mm "
                f"(std {self.distance_mm_std:.1f})"
            )
        return lines


def load_intrinsics(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read K and D from a tagcal calibration.json or an OpenCV FileStorage YAML."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            as_float64(payload["camera_matrix"]),
            as_float64(payload["distortion_coefficients"]).reshape(1, -1),
        )

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Cannot open {path}")
    try:
        matrix = storage.getNode("camera_matrix").mat()
        distortion = storage.getNode("distortion_coefficients").mat()
    finally:
        storage.release()
    if matrix is None:
        raise ValueError(f"{path} has no camera_matrix node")
    return (
        as_float64(matrix),
        as_float64(distortion) if distortion is not None else np.zeros((1, 5)),
    )


def verify_board(
    manifest: PatternManifest,
    frames: list[NDArray[np.uint8]],
    *,
    camera_matrix: NDArray[np.float64] | None = None,
    distortion: NDArray[np.float64] | None = None,
) -> VerificationResult:
    """Measure the board in each frame; estimate distance when intrinsics are given."""
    if not frames:
        raise ValueError("No frames to verify")

    board = AprilGridBoard(manifest.pattern)
    detector = AprilTagDetector(board)
    sides: list[float] = []
    distances: list[float] = []
    tag_counts: list[int] = []

    for frame in frames:
        observation = detector.detect(frame)
        if observation is None:
            continue
        tag_counts.append(observation.metrics.detected_tags)
        for corners in observation.marker_corners:
            quad = corners.reshape(4, 2)
            sides.append(
                float(
                    np.mean(
                        [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
                    )
                )
            )
        if camera_matrix is None:
            continue
        coefficients = distortion if distortion is not None else np.zeros((1, 5))
        ok, _, translation = cv2.solvePnP(
            observation.object_points,
            observation.image_points,
            camera_matrix,
            coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            # Object points are in millimetres, so the translation already is too.
            distances.append(float(np.linalg.norm(translation)))

    if not sides:
        raise RuntimeError(
            "ボードを検出できませんでした。露出、輝度、quiet zone、ピントを確認してください。"
        )

    return VerificationResult(
        frames_used=len(tag_counts),
        frames_attempted=len(frames),
        detected_tags_mean=float(np.mean(tag_counts)),
        tag_side_px_mean=float(np.mean(sides)),
        tag_side_px_std=float(np.std(sides)),
        distance_mm_mean=float(np.mean(distances)) if distances else None,
        distance_mm_std=float(np.std(distances)) if distances else None,
        tag_size_mm=manifest.pattern.effective_tag_size_mm,
    )


def grab_frames(spec: CaptureSpec, count: int) -> list[NDArray[np.uint8]]:
    """Grab consecutive frames from a camera without recording them."""
    from tagcal.capture import open_camera

    capture = open_camera(spec)
    try:
        frames: list[NDArray[np.uint8]] = []
        for _ in range(count):
            ok, captured = capture.read()
            if ok and captured is not None:
                frames.append(as_uint8(captured))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("カメラからフレームを取得できませんでした。")
    return frames

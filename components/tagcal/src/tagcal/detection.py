from __future__ import annotations

from dataclasses import dataclass
from math import atan2, log1p, pi, sqrt

import cv2
import numpy as np
from numpy.typing import NDArray

from tagcal.board import AprilGridBoard
from tagcal.cvtypes import as_float32, as_float32_list, as_int32, as_uint8
from tagcal.models import FrameMetrics


@dataclass(frozen=True, slots=True)
class DetectionObservation:
    marker_corners: list[NDArray[np.float32]]
    marker_ids: NDArray[np.int32]
    object_points: NDArray[np.float32]
    image_points: NDArray[np.float32]
    metrics: FrameMetrics


class AprilTagDetector:
    """Detect the configured AprilTag board and derive quality/pose descriptors."""

    def __init__(self, board: AprilGridBoard) -> None:
        self._board = board
        parameters = cv2.aruco.DetectorParameters()
        parameters.markerBorderBits = board.spec.border_bits
        # AprilTag's own edge-fitting refinement, rather than the generic corner
        # subpixel search: it fits the tag's black-border edges and intersects them,
        # which is what defines the corner of a tag in the first place.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 0.005
        # Both of these default to working on a decimated image. Downscaling finds
        # tags faster but locates their corners on a coarser grid, and calibration
        # is bounded by corner accuracy, not by detection speed.
        parameters.aprilTagQuadDecimate = 1.0
        parameters.useAruco3Detection = False
        # Mild presmoothing before edge fitting; measured to give the tightest
        # intrinsic standard deviations on real footage.
        parameters.aprilTagQuadSigma = 0.8
        parameters.minMarkerPerimeterRate = 0.015
        parameters.maxMarkerPerimeterRate = 4.0
        self._detector = cv2.aruco.ArucoDetector(board.dictionary, parameters)

    def detect(self, frame: NDArray[np.uint8]) -> DetectionObservation | None:
        if frame.ndim not in {2, 3}:
            raise ValueError(f"Expected grayscale or BGR frame, got shape {frame.shape}")
        gray = frame if frame.ndim == 2 else as_uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corners, ids, rejected = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        corners, ids, _, _ = self._detector.refineDetectedMarkers(
            gray,
            self._board.cv_board,
            corners,
            ids,
            rejected,
        )
        filtered_corners, filtered_ids = self._board.filter_detections(
            as_float32_list(corners),
            as_int32(ids) if ids is not None else None,
        )
        if filtered_ids is None or not filtered_corners:
            return None

        matched = self._board.match_image_points(filtered_corners, filtered_ids)
        if matched.image_points.shape[0] < 4:
            return None

        height, width = gray.shape[:2]
        hull = as_float32(cv2.convexHull(matched.image_points.reshape(-1, 1, 2)))
        hull_area = float(cv2.contourArea(hull))
        coverage = max(0.0, min(1.0, hull_area / float(width * height)))
        center = np.mean(matched.image_points, axis=0)
        center_x = float(center[0] / width)
        center_y = float(center[1] / height)
        sharpness = self._masked_sharpness(gray, hull)
        descriptor = self._pose_descriptor(
            matched.object_points,
            matched.image_points,
            width,
            height,
            coverage,
        )

        detected_tags = int(filtered_ids.size)
        tag_score = min(1.0, detected_tags / max(1, self._board.marker_count))
        # Sharpness deliberately does not enter the quality score: its scale is
        # scene-dependent, so any fixed mapping either saturates or dominates.
        # Blur is handled by comparing neighbouring frames during selection.
        coverage_score = min(1.0, coverage / 0.20)
        quality_score = 0.68 * tag_score + 0.32 * coverage_score

        metrics = FrameMetrics(
            detected_tags=detected_tags,
            detected_points=int(matched.image_points.shape[0]),
            sharpness=float(sharpness),
            board_coverage=float(coverage),
            center_x=center_x,
            center_y=center_y,
            quality_score=float(quality_score),
            descriptor=[float(value) for value in descriptor],
        )
        return DetectionObservation(
            marker_corners=filtered_corners,
            marker_ids=filtered_ids,
            object_points=matched.object_points,
            image_points=matched.image_points,
            metrics=metrics,
        )

    @staticmethod
    def draw_overlay(
        frame: NDArray[np.uint8],
        observation: DetectionObservation | None,
        status: str | None = None,
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        if observation is not None:
            cv2.aruco.drawDetectedMarkers(
                output,
                observation.marker_corners,
                observation.marker_ids,
            )
            metrics = observation.metrics
            text = (
                f"tags={metrics.detected_tags} "
                f"coverage={metrics.board_coverage:.3f} "
                f"sharpness={metrics.sharpness:.0f} "
                f"quality={metrics.quality_score:.2f}"
            )
            cv2.putText(
                output,
                text,
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                output,
                "AprilTag board not detected",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if status:
            cv2.putText(
                output,
                status,
                (16, output.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return output

    @staticmethod
    def _masked_sharpness(
        gray: NDArray[np.uint8],
        hull: NDArray[np.float32],
    ) -> float:
        """Normalised Tenengrad over the board region.

        Tenengrad (mean squared Sobel gradient) is preferred over the variance of
        the Laplacian because a second derivative amplifies sensor noise, which a
        calibration recording has plenty of at short exposures; comparative studies
        of focus measures rank gradient- and statistics-based operators as the more
        noise-robust choice (Pertuz et al., Pattern Recognition 46(5), 2013).

        Dividing by the region's intensity variance removes most of the dependence
        on how bright and contrasty the board happens to be, so values from
        different distances and exposures are closer to comparable. They are still
        not comparable across scenes, which is why selection compares each frame
        against its temporal neighbours rather than against a fixed threshold.
        """
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        inside = mask > 0
        if int(np.count_nonzero(inside)) < 16:
            return 0.0

        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gradient_x[inside] ** 2 + gradient_y[inside] ** 2))
        contrast = float(np.var(gray[inside].astype(np.float64)))
        return tenengrad / max(contrast, 1.0)

    def _pose_descriptor(
        self,
        object_points: NDArray[np.float32],
        image_points: NDArray[np.float32],
        width: int,
        height: int,
        coverage: float,
    ) -> NDArray[np.float64]:
        center = np.mean(image_points, axis=0)
        center_x = float(center[0] / width)
        center_y = float(center[1] / height)
        focal_guess = float(max(width, height) * 1.2)
        camera_matrix = np.array(
            [
                [focal_guess, 0.0, width / 2.0],
                [0.0, focal_guess, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((5, 1), dtype=np.float64)

        normal_x = 0.0
        normal_y = 0.0
        roll = 0.0
        distance = 0.0
        solved = False
        for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
            try:
                solved, rotation_vector, translation_vector = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=method,
                )
            except cv2.error:
                solved = False
            if solved:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                normal = rotation_matrix[:, 2]
                normal_x = float(normal[0])
                normal_y = float(normal[1])
                roll = float(atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) / pi)
                distance = float(np.linalg.norm(translation_vector))
                break

        normalized_distance = (
            log1p(distance / max(1e-6, self._board.diagonal_mm)) if solved else 0.0
        )
        return np.asarray(
            [
                center_x,
                center_y,
                sqrt(max(0.0, coverage)),
                normal_x,
                normal_y,
                roll,
                normalized_distance,
            ],
            dtype=np.float64,
        )

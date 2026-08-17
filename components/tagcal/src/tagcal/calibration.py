from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from tagcal.cvtypes import as_float64, as_float64_list
from tagcal.models import (
    CalibrationResult,
    CalibrationSpec,
    CalibrationView,
    KeyframeRecord,
    SelectionReport,
)

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CalibrationArtifacts:
    result: CalibrationResult
    json_path: Path
    opencv_yaml_path: Path
    ros_yaml_path: Path
    preview_path: Path | None
    report_path: Path


@dataclass(slots=True)
class _CalibrationRun:
    rms: float
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]
    rotation_vectors: list[NDArray[np.float64]]
    translation_vectors: list[NDArray[np.float64]]
    intrinsic_std: NDArray[np.float64]
    per_view_errors: NDArray[np.float64]


class CameraCalibrator:
    """Calibrate intrinsics and iteratively reject high-reprojection-error views."""

    def __init__(self, spec: CalibrationSpec) -> None:
        self._spec = spec

    def calibrate(
        self,
        selection: SelectionReport,
        output_dir: Path,
        *,
        keyframe_root: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> CalibrationArtifacts:
        notify = progress or (lambda _: None)
        if len(selection.selected) < self._spec.min_views:
            raise RuntimeError(
                f"Calibration requires at least {self._spec.min_views} views; "
                f"received {len(selection.selected)}."
            )

        object_points = [
            np.asarray(record.object_points, dtype=np.float32).reshape(-1, 3)
            for record in selection.selected
        ]
        image_points = [
            np.asarray(record.image_points, dtype=np.float32).reshape(-1, 2)
            for record in selection.selected
        ]
        for index, (objects, images) in enumerate(zip(object_points, image_points, strict=True)):
            if objects.shape[0] != images.shape[0] or objects.shape[0] < 4:
                raise ValueError(f"Invalid point correspondence in keyframe {index}")

        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if self._spec.rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if self._spec.zero_tangent_distortion:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        if self._spec.fix_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

        active = list(range(len(selection.selected)))
        excluded: list[dict[str, object]] = []
        final_run: _CalibrationRun | None = None
        image_size = (selection.image_width, selection.image_height)

        for iteration in range(self._spec.max_outlier_iterations + 1):
            active_objects = [object_points[index] for index in active]
            active_images = [image_points[index] for index in active]
            final_run = self._run_calibration(active_objects, active_images, image_size, flags)
            errors = final_run.per_view_errors
            notify(
                f"Calibration iteration {iteration + 1}: {len(active)} views, "
                f"RMS={final_run.rms:.4f}px, max view={float(np.max(errors)):.4f}px."
            )

            if iteration >= self._spec.max_outlier_iterations:
                break
            if len(active) <= self._spec.min_views:
                break

            threshold = self._outlier_threshold(errors)
            worst_local_index = int(np.argmax(errors))
            worst_error = float(errors[worst_local_index])
            if worst_error <= threshold:
                break

            removed_global_index = active.pop(worst_local_index)
            record = selection.selected[removed_global_index]
            excluded.append(
                {
                    "image_path": record.image_path,
                    "frame_index": record.frame_index,
                    "timestamp_seconds": record.timestamp_seconds,
                    "reprojection_error_px": worst_error,
                    "reason": f"error above robust threshold {threshold:.4f}px",
                }
            )
            notify(
                f"Excluded frame {record.frame_index} with {worst_error:.4f}px "
                f"reprojection error."
            )

        if final_run is None:
            raise RuntimeError("Calibration did not produce a result")

        views = self._build_views(selection.selected, active, final_run)
        errors = np.asarray([view.reprojection_error_px for view in views], dtype=np.float64)
        distortion_model = "rational_polynomial" if self._spec.rational_model else "plumb_bob"
        result = CalibrationResult(
            image_width=selection.image_width,
            image_height=selection.image_height,
            camera_matrix=final_run.camera_matrix.astype(float).tolist(),
            distortion_coefficients=(
                self._model_distortion(final_run.distortion).astype(float).tolist()
            ),
            distortion_model=distortion_model,
            rms_reprojection_error_px=float(final_run.rms),
            mean_view_error_px=float(np.mean(errors)),
            median_view_error_px=float(np.median(errors)),
            max_view_error_px=float(np.max(errors)),
            intrinsic_standard_deviations=final_run.intrinsic_std.reshape(-1).astype(float).tolist(),
            used_views=views,
            excluded_views=excluded,
            flags=int(flags),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "calibration.json"
        result.save_json(json_path)
        opencv_yaml_path = output_dir / "calibration_opencv.yaml"
        self._write_opencv_yaml(opencv_yaml_path, result)
        ros_yaml_path = output_dir / "camera_info.yaml"
        self._write_ros_yaml(ros_yaml_path, result)
        preview_path = self._write_preview(
            keyframe_root or output_dir,
            output_dir,
            selection,
            views,
            result,
        )
        report_path = output_dir / "report.html"
        self._write_html_report(report_path, result, preview_path)

        notify(
            f"Calibration complete: RMS={result.rms_reprojection_error_px:.4f}px, "
            f"median view error={result.median_view_error_px:.4f}px."
        )
        return CalibrationArtifacts(
            result=result,
            json_path=json_path,
            opencv_yaml_path=opencv_yaml_path,
            ros_yaml_path=ros_yaml_path,
            preview_path=preview_path,
            report_path=report_path,
        )

    def _run_calibration(
        self,
        object_points: list[NDArray[np.float32]],
        image_points: list[NDArray[np.float32]],
        image_size: tuple[int, int],
        flags: int,
    ) -> _CalibrationRun:
        initial_matrix = cv2.initCameraMatrix2D(
            object_points,
            image_points,
            image_size,
            aspectRatio=0.0,
        )
        if not np.all(np.isfinite(initial_matrix)):
            initial_matrix = np.array(
                [
                    [max(image_size), 0.0, image_size[0] / 2.0],
                    [0.0, max(image_size), image_size[1] / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        distortion_length = 8 if self._spec.rational_model else 5
        initial_distortion = np.zeros((distortion_length, 1), dtype=np.float64)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-10,
        )
        (
            rms,
            camera_matrix,
            distortion,
            rotation_vectors,
            translation_vectors,
            intrinsic_std,
            _,
            _,
        ) = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            image_size,
            initial_matrix,
            initial_distortion,
            flags=flags,
            criteria=criteria,
        )

        rotations = as_float64_list(rotation_vectors)
        translations = as_float64_list(translation_vectors)
        matrix = as_float64(camera_matrix)
        coefficients = as_float64(distortion)
        manual_errors = self._reprojection_errors(
            object_points,
            image_points,
            rotations,
            translations,
            matrix,
            coefficients,
        )
        return _CalibrationRun(
            rms=float(rms),
            camera_matrix=matrix,
            distortion=coefficients,
            rotation_vectors=rotations,
            translation_vectors=translations,
            intrinsic_std=as_float64(intrinsic_std),
            per_view_errors=manual_errors,
        )

    @staticmethod
    def _reprojection_errors(
        object_points: list[NDArray[np.float32]],
        image_points: list[NDArray[np.float32]],
        rotation_vectors: tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]],
        translation_vectors: tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]],
        camera_matrix: NDArray[np.float64],
        distortion: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        errors: list[float] = []
        for objects, images, rotation, translation in zip(
            object_points,
            image_points,
            rotation_vectors,
            translation_vectors,
            strict=True,
        ):
            projected, _ = cv2.projectPoints(
                objects,
                rotation,
                translation,
                camera_matrix,
                distortion,
            )
            residual = projected.reshape(-1, 2) - images.reshape(-1, 2)
            rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            errors.append(rmse)
        return np.asarray(errors, dtype=np.float64)

    def _outlier_threshold(self, errors: NDArray[np.float64]) -> float:
        values = [float(value) for value in errors]
        center = median(values)
        absolute_deviations = [abs(value - center) for value in values]
        mad = median(absolute_deviations)
        robust_sigma = 1.4826 * mad
        robust_threshold = center + self._spec.outlier_mad_scale * robust_sigma
        return max(
            self._spec.min_outlier_threshold_px,
            min(self._spec.max_view_error_px, robust_threshold),
        )

    @staticmethod
    def _build_views(
        records: list[KeyframeRecord],
        active: list[int],
        run: _CalibrationRun,
    ) -> list[CalibrationView]:
        views: list[CalibrationView] = []
        for local_index, global_index in enumerate(active):
            record = records[global_index]
            views.append(
                CalibrationView(
                    image_path=record.image_path,
                    frame_index=record.frame_index,
                    timestamp_seconds=record.timestamp_seconds,
                    reprojection_error_px=float(run.per_view_errors[local_index]),
                    rotation_vector=run.rotation_vectors[local_index]
                    .reshape(-1)
                    .astype(float)
                    .tolist(),
                    translation_vector=run.translation_vectors[local_index]
                    .reshape(-1)
                    .astype(float)
                    .tolist(),
                )
            )
        return views

    def _model_distortion(
        self,
        distortion: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        coefficient_count = 8 if self._spec.rational_model else 5
        flattened = np.asarray(distortion, dtype=np.float64).reshape(-1)
        if flattened.size < coefficient_count:
            raise RuntimeError(
                f"OpenCV returned {flattened.size} distortion coefficients; "
                f"expected at least {coefficient_count}."
            )
        return flattened[:coefficient_count]

    @staticmethod
    def _write_opencv_yaml(path: Path, result: CalibrationResult) -> None:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        if not storage.isOpened():
            raise RuntimeError(f"Unable to write OpenCV YAML: {path}")
        try:
            storage.write("image_width", result.image_width)
            storage.write("image_height", result.image_height)
            storage.write("camera_matrix", np.asarray(result.camera_matrix, dtype=np.float64))
            storage.write(
                "distortion_coefficients",
                np.asarray(result.distortion_coefficients, dtype=np.float64).reshape(1, -1),
            )
            storage.write("distortion_model", result.distortion_model)
            storage.write("rms_reprojection_error_px", result.rms_reprojection_error_px)
        finally:
            storage.release()

    @staticmethod
    def _write_ros_yaml(path: Path, result: CalibrationResult) -> None:
        camera_matrix = np.asarray(result.camera_matrix, dtype=float)
        projection = np.zeros((3, 4), dtype=float)
        projection[:, :3] = camera_matrix
        data = {
            "image_width": result.image_width,
            "image_height": result.image_height,
            "camera_name": "tagcal_camera",
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": camera_matrix.reshape(-1).tolist(),
            },
            "distortion_model": result.distortion_model,
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(result.distortion_coefficients),
                "data": result.distortion_coefficients,
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": np.eye(3, dtype=float).reshape(-1).tolist(),
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": projection.reshape(-1).tolist(),
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _write_preview(
        keyframe_root: Path,
        output_dir: Path,
        selection: SelectionReport,
        views: list[CalibrationView],
        result: CalibrationResult,
    ) -> Path | None:
        if not views:
            return None
        best_view = min(views, key=lambda view: view.reprojection_error_px)
        image_path = keyframe_root / best_view.image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None

        camera_matrix = np.asarray(result.camera_matrix, dtype=np.float64)
        distortion = np.asarray(result.distortion_coefficients, dtype=np.float64)
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (selection.image_width, selection.image_height),
            0.0,
            (selection.image_width, selection.image_height),
        )
        corrected = cv2.undistort(image, camera_matrix, distortion, None, new_matrix)
        cv2.putText(
            image,
            "original",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            corrected,
            "undistorted",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        comparison = np.hstack([image, corrected])
        preview_path = output_dir / "undistorted_preview.jpg"
        cv2.imwrite(str(preview_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return preview_path

    @staticmethod
    def _write_html_report(
        path: Path,
        result: CalibrationResult,
        preview_path: Path | None,
    ) -> None:
        matrix_rows = "".join(
            f"<tr>{''.join(f'<td>{value:.8f}</td>' for value in row)}</tr>"
            for row in result.camera_matrix
        )
        used_rows = "".join(
            "<tr>"
            f"<td>{html.escape(view.image_path)}</td>"
            f"<td>{view.frame_index}</td>"
            f"<td>{view.timestamp_seconds:.3f}</td>"
            f"<td>{view.reprojection_error_px:.5f}</td>"
            "</tr>"
            for view in sorted(result.used_views, key=lambda item: item.reprojection_error_px)
        )
        excluded_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(view.get('image_path', '')))}</td>"
            f"<td>{view.get('frame_index', '')}</td>"
            f"<td>{float(view.get('reprojection_error_px', 0.0)):.5f}</td>"
            f"<td>{html.escape(str(view.get('reason', '')))}</td>"
            "</tr>"
            for view in result.excluded_views
        ) or '<tr><td colspan="4">None</td></tr>'
        preview_html = (
            f'<h2>Undistortion preview</h2><img src="{html.escape(preview_path.name)}" '
            'style="max-width:100%;height:auto">'
            if preview_path is not None
            else ""
        )
        document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AprilTag Camera Calibration Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #bbb; padding: .45rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
code {{ background: #eee; padding: .1rem .25rem; }}
</style>
</head>
<body>
<h1>AprilTag Camera Intrinsic Calibration</h1>
<p>Image size: <code>{result.image_width} × {result.image_height}</code></p>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>RMS reprojection error</td><td>{result.rms_reprojection_error_px:.6f} px</td></tr>
<tr><td>Mean view error</td><td>{result.mean_view_error_px:.6f} px</td></tr>
<tr><td>Median view error</td><td>{result.median_view_error_px:.6f} px</td></tr>
<tr><td>Maximum view error</td><td>{result.max_view_error_px:.6f} px</td></tr>
<tr><td>Views used</td><td>{len(result.used_views)}</td></tr>
<tr><td>Views excluded</td><td>{len(result.excluded_views)}</td></tr>
<tr><td>Distortion model</td><td>{html.escape(result.distortion_model)}</td></tr>
</table>
<h2>Camera matrix</h2>
<table>{matrix_rows}</table>
<h2>Distortion coefficients</h2>
<p><code>{html.escape(str(result.distortion_coefficients))}</code></p>
{preview_html}
<h2>Used views</h2>
<table><tr><th>Image</th><th>Frame</th><th>Time (s)</th><th>Error (px)</th></tr>{used_rows}</table>
<h2>Excluded views</h2>
<table><tr><th>Image</th><th>Frame</th><th>Error (px)</th><th>Reason</th></tr>{excluded_rows}</table>
</body>
</html>
"""
        path.write_text(document, encoding="utf-8")

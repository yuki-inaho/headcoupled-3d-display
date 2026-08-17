from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tagcal.board import AprilGridBoard
from tagcal.calibration import CalibrationArtifacts, CameraCalibrator
from tagcal.capture import RecordingResult, record_video
from tagcal.detection import AprilTagDetector
from tagcal.models import (
    CalibrationSpec,
    CaptureSpec,
    PatternManifest,
    SelectionSpec,
    utc_now_iso,
)
from tagcal.selection import VideoKeyframeSelector

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PipelineArtifacts:
    calibration: CalibrationArtifacts
    selection_report_path: Path
    recording: RecordingResult | None = None


class CalibrationPipeline:
    """Application service coordinating selection, calibration, and reproducible output."""

    def __init__(
        self,
        *,
        selection_spec: SelectionSpec | None = None,
        calibration_spec: CalibrationSpec | None = None,
    ) -> None:
        self._selection_spec = selection_spec or SelectionSpec()
        self._calibration_spec = calibration_spec or CalibrationSpec()

    def process_video(
        self,
        video_path: Path,
        manifest_path: Path,
        output_dir: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> PipelineArtifacts:
        notify = progress or (lambda _: None)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = PatternManifest.load(manifest_path)
        board = AprilGridBoard(manifest.pattern)
        self._write_session_metadata(output_dir, video_path, manifest_path)
        self._copy_pattern_inputs(manifest, manifest_path, output_dir)

        selector = VideoKeyframeSelector(board, self._selection_spec)
        selection = selector.select(video_path, output_dir, progress=notify)
        calibrator = CameraCalibrator(self._calibration_spec)
        calibration = calibrator.calibrate(selection, output_dir, progress=notify)
        return PipelineArtifacts(
            calibration=calibration,
            selection_report_path=output_dir / "selection_report.json",
        )

    def record_and_process(
        self,
        manifest_path: Path,
        output_dir: Path,
        capture_spec: CaptureSpec,
        *,
        progress: ProgressCallback | None = None,
    ) -> PipelineArtifacts:
        notify = progress or (lambda _: None)
        manifest = PatternManifest.load(manifest_path)
        board = AprilGridBoard(manifest.pattern)
        detector = AprilTagDetector(board)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_suffix = ".avi" if capture_spec.codec.upper() == "MJPG" else ".mp4"
        video_path = output_dir / f"capture{video_suffix}"
        recording = record_video(
            video_path,
            capture_spec,
            detector=detector,
            progress=notify,
        )
        processed = self.process_video(
            recording.video_path,
            manifest_path,
            output_dir,
            progress=notify,
        )
        return PipelineArtifacts(
            calibration=processed.calibration,
            selection_report_path=processed.selection_report_path,
            recording=recording,
        )

    def _write_session_metadata(
        self,
        output_dir: Path,
        video_path: Path,
        manifest_path: Path,
    ) -> None:
        data = {
            "created_at": utc_now_iso(),
            "video_path": str(video_path.resolve()),
            "source_manifest_path": str(manifest_path.resolve()),
            "selection": asdict(self._selection_spec),
            "calibration": asdict(self._calibration_spec),
        }
        with (output_dir / "session.json").open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    @staticmethod
    def _copy_pattern_inputs(
        manifest: PatternManifest,
        manifest_path: Path,
        output_dir: Path,
    ) -> None:
        png_source = manifest.resolve_png(manifest_path).resolve()
        if not png_source.exists():
            raise FileNotFoundError(png_source)
        png_destination = output_dir / png_source.name
        if png_source != png_destination.resolve():
            shutil.copy2(png_source, png_destination)

        pdf_name: str | None = None
        pdf_source = manifest.resolve_pdf(manifest_path)
        if pdf_source is not None:
            pdf_source = pdf_source.resolve()
            if pdf_source.exists():
                pdf_destination = output_dir / pdf_source.name
                if pdf_source != pdf_destination.resolve():
                    shutil.copy2(pdf_source, pdf_destination)
                pdf_name = pdf_destination.name

        session_manifest = replace(
            manifest,
            png_path=png_destination.name,
            pdf_path=pdf_name,
        )
        session_manifest.save(output_dir / "pattern.json")

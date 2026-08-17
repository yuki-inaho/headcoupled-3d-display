"""Background worker that runs the calibration pipeline off the Qt event loop."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from tagcal.models import CalibrationSpec, SelectionSpec
from tagcal.pipeline import CalibrationPipeline


class PipelineWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        manifest_path: Path,
        output_dir: Path,
        selection_spec: SelectionSpec,
        calibration_spec: CalibrationSpec,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._manifest_path = manifest_path
        self._output_dir = output_dir
        self._selection_spec = selection_spec
        self._calibration_spec = calibration_spec

    @Slot()
    def run(self) -> None:
        try:
            pipeline = CalibrationPipeline(
                selection_spec=self._selection_spec,
                calibration_spec=self._calibration_spec,
            )
            artifacts = pipeline.process_video(
                self._video_path,
                self._manifest_path,
                self._output_dir,
                progress=self.progress.emit,
            )
            result = artifacts.calibration.result
            self.finished.emit(
                f"完了: RMS={result.rms_reprojection_error_px:.5f}px, "
                f"使用ビュー={len(result.used_views)}, 出力={self._output_dir}"
            )
        except Exception:
            self.failed.emit(traceback.format_exc())

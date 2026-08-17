"""Minimal operating panel for a calibration session.

The full GUI exposes every parameter; this one exposes only the three actions a
session actually needs -- put the board on screen, watch what the camera sees,
record -- so the operator can keep both hands on the camera. Stopping a recording
runs keyframe selection and calibration automatically.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tagcal.board import AprilGridBoard
from tagcal.capture import (
    actual_fps,
    create_video_writer,
    describe_capture,
    fourcc_text,
    mode_mismatch,
    open_camera,
    preferred_backend_name,
    probe_cameras,
    timestamped_video_path,
)
from tagcal.cvtypes import as_uint8
from tagcal.detection import AprilTagDetector, DetectionObservation
from tagcal.models import (
    CalibrationSpec,
    CaptureSpec,
    PatternManifest,
    PatternSpec,
    SelectionSpec,
    utc_now_iso,
)
from tagcal.qtworker import PipelineWorker
from tagcal.screen import (
    Monitor,
    configure_physical_pixels,
    plan_layout,
    query_monitors,
    write_manifest,
)
from tagcal.screenview import BoardWindow

_FRAME_INTERVAL_MS = 30
_REQUESTED_FPS = 30.0


class PreviewWindow(QLabel):
    """Separate window for the live camera image, closable without touching capture."""

    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("tagcal camera")
        self.setStyleSheet("background:#161616; color:#ddd; padding:8px;")
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        self.resize(960, 540)
        self.setText("カメラ映像を待機しています…")
        self._frame: QImage | None = None

    def show_frame(self, image: QImage) -> None:
        """Fit the frame to the window.

        A QLabel neither scales nor scrolls an oversized pixmap: it shows the
        middle of it and drops the rest, so a 1920x1080 frame in a smaller window
        looks like a cropped close-up. Scaling here keeps the whole field of view
        visible, which is what the operator needs to judge coverage.
        """
        self._frame = image
        self.setWindowTitle(f"tagcal camera — {image.width()}×{image.height()}")
        self.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._frame is not None:
            self.show_frame(self._frame)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)


class PanelWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("tagcal panel")
        self.resize(720, 520)

        self._monitors: list[Monitor] = []
        self._monitor_error: str | None = None
        self._capture: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._detector: AprilTagDetector | None = None
        self._observation: DetectionObservation | None = None
        self._manifest_path: Path | None = None
        self._video_path: Path | None = None
        self._recorded_frames = 0
        self._recording = False
        self._capture_fps = 30.0
        self._board_window: BoardWindow | None = None
        self._preview_window: PreviewWindow | None = None
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._read_frame)

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.addWidget(self._build_settings())
        root.addWidget(self._build_actions())

        self.status = QLabel("準備完了")
        self.status.setStyleSheet("padding:4px;")
        root.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        root.addWidget(self.log, stretch=1)
        self.setCentralWidget(central)

        self._load_monitors()
        self._load_cameras()

    # ------------------------------------------------------------------ layout

    def _build_settings(self) -> QWidget:
        box = QWidget()
        grid = QGridLayout(box)

        self.monitor = QComboBox()
        grid.addWidget(QLabel("モニタ"), 0, 0)
        grid.addWidget(self.monitor, 0, 1)

        self.tag_size = QDoubleSpinBox()
        self.tag_size.setRange(5.0, 2000.0)
        self.tag_size.setValue(40.0)
        self.tag_size.setSuffix(" mm")
        grid.addWidget(QLabel("タグ"), 0, 2)
        grid.addWidget(self.tag_size, 0, 3)

        self.columns = QSpinBox()
        self.columns.setRange(1, 20)
        self.columns.setValue(4)
        grid.addWidget(QLabel("列"), 0, 4)
        grid.addWidget(self.columns, 0, 5)

        self.rows = QSpinBox()
        self.rows.setRange(1, 20)
        self.rows.setValue(3)
        grid.addWidget(QLabel("行"), 0, 6)
        grid.addWidget(self.rows, 0, 7)

        self.calibration_factor = QDoubleSpinBox()
        self.calibration_factor.setRange(0.5, 2.0)
        self.calibration_factor.setDecimals(4)
        self.calibration_factor.setSingleStep(0.001)
        self.calibration_factor.setValue(1.0)
        self.calibration_factor.setToolTip(
            "定規で測った長さ / 表示された長さ。EDIDの物理サイズがずれている場合に補正します。"
        )
        grid.addWidget(QLabel("px/mm校正"), 1, 0)
        grid.addWidget(self.calibration_factor, 1, 1)

        self.camera = QComboBox()
        grid.addWidget(QLabel("カメラ"), 1, 2)
        grid.addWidget(self.camera, 1, 3)

        self.frame_width = QSpinBox()
        self.frame_width.setRange(160, 7680)
        self.frame_width.setValue(1280)
        grid.addWidget(QLabel("幅"), 1, 4)
        grid.addWidget(self.frame_width, 1, 5)

        self.frame_height = QSpinBox()
        self.frame_height.setRange(120, 4320)
        self.frame_height.setValue(720)
        grid.addWidget(QLabel("高さ"), 1, 6)
        grid.addWidget(self.frame_height, 1, 7)

        self.output_dir = QLineEdit("artifacts/session")
        browse = QPushButton("参照…")
        browse.clicked.connect(self._browse_output)
        grid.addWidget(QLabel("出力先"), 2, 0)
        grid.addWidget(self.output_dir, 2, 1, 1, 5)
        grid.addWidget(browse, 2, 6, 1, 2)

        self.manifest = QLineEdit()
        self.manifest.setPlaceholderText("パターン表示で自動設定されます（印刷ボードの場合は選択）")
        pick = QPushButton("選択…")
        pick.clicked.connect(self._browse_manifest)
        grid.addWidget(QLabel("manifest"), 3, 0)
        grid.addWidget(self.manifest, 3, 1, 1, 5)
        grid.addWidget(pick, 3, 6, 1, 2)
        return box

    def _build_actions(self) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)

        self.pattern_button = QPushButton("パターン表示 (V)")
        self.pattern_button.setCheckable(True)
        self.pattern_button.setShortcut(QKeySequence("V"))
        self.pattern_button.clicked.connect(self._toggle_pattern)

        self.preview_button = QPushButton("カメラ映像 (C)")
        self.preview_button.setCheckable(True)
        self.preview_button.setShortcut(QKeySequence("C"))
        self.preview_button.clicked.connect(self._toggle_preview)

        self.record_button = QPushButton("録画開始 (R)")
        self.record_button.setShortcut(QKeySequence("R"))
        self.record_button.clicked.connect(self._toggle_recording)

        self.settings_button = QPushButton("設定を出力 (S)")
        self.settings_button.setShortcut(QKeySequence("S"))
        self.settings_button.clicked.connect(self._dump_settings)

        for button in (
            self.pattern_button,
            self.preview_button,
            self.record_button,
            self.settings_button,
        ):
            button.setMinimumHeight(46)
            row.addWidget(button)
        return box

    # ---------------------------------------------------------------- settings

    def _actual_camera(self) -> dict[str, Any] | None:
        """What the device granted, which only exists while it is open."""
        if self._capture is None:
            return None
        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": actual_fps(self._capture, _REQUESTED_FPS),
            "fourcc": fourcc_text(self._capture.get(cv2.CAP_PROP_FOURCC)),
            "backend": self._capture.getBackendName(),
        }

    def _settings_payload(self) -> dict[str, Any]:
        """Everything needed to reproduce this session's geometry and capture mode."""
        spec = self._pattern_spec()
        monitor = self._selected_monitor()
        payload: dict[str, Any] = {
            "created_at": utc_now_iso(),
            "camera": {
                "index": int(self.camera.currentData() or 0),
                "requested": {
                    "width": self.frame_width.value(),
                    "height": self.frame_height.value(),
                    "fps": _REQUESTED_FPS,
                    "input_fourcc": CaptureSpec().input_fourcc,
                },
                "actual": self._actual_camera(),
                "recording": self._recording,
            },
            "pattern": asdict(spec),
            "paths": {
                "output_dir": str(Path(self.output_dir.text()).expanduser()),
                "manifest": self.manifest.text().strip() or None,
            },
        }
        if monitor is None:
            return payload

        payload["monitor"] = {
            "name": monitor.name,
            "resolution_px": [monitor.width_px, monitor.height_px],
            "physical_mm": [monitor.width_mm, monitor.height_mm],
            "px_per_mm": monitor.px_per_mm,
            "ppi": monitor.ppi,
            "aspect_mismatch_percent": monitor.aspect_mismatch_percent,
            "calibration_factor": self.calibration_factor.value(),
        }
        try:
            layout = plan_layout(
                spec,
                monitor=monitor,
                calibration_factor=self.calibration_factor.value(),
                snap=True,
            )
        except (RuntimeError, ValueError) as exc:
            payload["screen_layout_error"] = str(exc)
            return payload

        payload["screen_layout"] = {
            "px_per_mm": layout.px_per_mm,
            "cells_per_tag": layout.cells_per_tag,
            "cell_px": layout.cell_px,
            "tag_px": layout.tag_px,
            "gap_px": layout.gap_px,
            "margin_px": layout.margin_px,
            "actual_tag_size_mm": layout.actual_tag_size_mm,
            "actual_gap_mm": layout.spec.gap_mm,
            "tag_size_error_mm": layout.tag_size_error_mm,
            "board_px": [layout.board_width_px, layout.board_height_px],
            "displayed": self._board_window is not None,
        }
        return payload

    @Slot()
    def _dump_settings(self) -> None:
        payload = self._settings_payload()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(text, flush=True)
        self._log(text)
        try:
            output_dir = Path(self.output_dir.text()).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "panel_settings.json"
            path.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            self._fail(f"設定を保存できませんでした: {exc}")
            return
        self.status.setText(f"設定を出力しました: {path}")

    # ----------------------------------------------------------------- helpers

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _fail(self, message: str) -> None:
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "tagcal", message)

    def _load_monitors(self) -> None:
        try:
            self._monitors = query_monitors()
        except RuntimeError as exc:
            self._monitor_error = str(exc)
            self.monitor.addItem("(検出できません)")
            self.monitor.setEnabled(False)
            self.pattern_button.setEnabled(False)
            self._log(f"モニタを検出できませんでした: {exc}")
            return
        for index, monitor in enumerate(self._monitors):
            self.monitor.addItem(monitor.describe(), monitor)
            if monitor.primary:
                self.monitor.setCurrentIndex(index)

    def _load_cameras(self) -> None:
        # A UVC camera exposes several device nodes (capture, metadata, an h264
        # stream). Only the one the native backend opens is the real capture node;
        # the others ignore the requested format and quietly hand back something else.
        native = preferred_backend_name()
        preferred_index: int | None = None
        for device in probe_cameras(6):
            supported = device.backend == native
            label = (
                f"{device.index}: {device.width}×{device.height} "
                f"{device.fourcc} {device.backend}"
            )
            if not supported:
                label += "  ※非推奨"
            self.camera.addItem(label, device.index)
            if supported and preferred_index is None:
                preferred_index = self.camera.count() - 1
        if self.camera.count() == 0:
            self.camera.addItem("0", 0)
            self._log("カメラを自動検出できませんでした。index 0 を既定にします。")
        elif preferred_index is not None:
            self.camera.setCurrentIndex(preferred_index)
        else:
            self._log(f"警告: {native}で開けるカメラがありません。解像度指定が効かない場合があります。")

    def _selected_monitor(self) -> Monitor | None:
        data = self.monitor.currentData()
        return data if isinstance(data, Monitor) else None

    def _pattern_spec(self) -> PatternSpec:
        # Gap and margin follow the tag size so the quiet zone stays adequate at any scale.
        spacing_mm = max(4.0, self.tag_size.value() * 0.25)
        return PatternSpec(
            columns=self.columns.value(),
            rows=self.rows.value(),
            tag_size_mm=self.tag_size.value(),
            gap_mm=spacing_mm,
            margin_mm=spacing_mm,
        )

    @Slot()
    def _browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "出力先", self.output_dir.text())
        if chosen:
            self.output_dir.setText(chosen)

    @Slot()
    def _browse_manifest(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "pattern.json", "", "JSON (*.json)")
        if chosen:
            self.manifest.setText(chosen)
            self._manifest_path = Path(chosen)
            self._detector = None

    # ------------------------------------------------------------ board window

    @Slot()
    def _toggle_pattern(self) -> None:
        if self.pattern_button.isChecked():
            self._show_pattern()
        else:
            self._hide_pattern()

    def _show_pattern(self) -> None:
        monitor = self._selected_monitor()
        if monitor is None:
            self.pattern_button.setChecked(False)
            self._fail(self._monitor_error or "モニタを選択できません。")
            return
        try:
            layout = plan_layout(
                self._pattern_spec(),
                monitor=monitor,
                calibration_factor=self.calibration_factor.value(),
                snap=True,
            )
            screen_dir = Path(self.output_dir.text()).expanduser() / "screen"
            write_manifest(layout, screen_dir)
            self._manifest_path = screen_dir / "pattern.json"
            self.manifest.setText(str(self._manifest_path))
            self._detector = None

            self._board_window = BoardWindow(layout, on_top=True, warn=self._log)
            self._board_window.closed.connect(self._on_board_closed)
        except (RuntimeError, ValueError) as exc:
            self.pattern_button.setChecked(False)
            self._fail(str(exc))
            return

        for line in layout.describe():
            self._log(line)
        self._log(f"manifest: {self._manifest_path}")
        self.status.setText(f"パターン表示中 — 実効タグ {layout.actual_tag_size_mm:.3f} mm")

    def _hide_pattern(self) -> None:
        if self._board_window is not None:
            window, self._board_window = self._board_window, None
            window.close()
        self.status.setText("パターン非表示")

    @Slot()
    def _on_board_closed(self) -> None:
        self._board_window = None
        self.pattern_button.setChecked(False)

    # ---------------------------------------------------------------- capture

    @Slot()
    def _toggle_preview(self) -> None:
        if self.preview_button.isChecked():
            if not self._ensure_capture():
                self.preview_button.setChecked(False)
                return
            self._preview_window = PreviewWindow()
            self._preview_window.closed.connect(self._on_preview_closed)
            self._preview_window.show()
        else:
            self._close_preview()
            self._release_capture_if_idle()

    def _close_preview(self) -> None:
        if self._preview_window is not None:
            window, self._preview_window = self._preview_window, None
            window.close()

    @Slot()
    def _on_preview_closed(self) -> None:
        self._preview_window = None
        self.preview_button.setChecked(False)
        self._release_capture_if_idle()

    def _ensure_capture(self) -> bool:
        if self._capture is not None:
            return True
        spec = CaptureSpec(
            camera_index=int(self.camera.currentData() or 0),
            width=self.frame_width.value(),
            height=self.frame_height.value(),
            fps=_REQUESTED_FPS,
            preview=False,
        )
        try:
            self._capture = open_camera(spec)
        except RuntimeError as exc:
            self._fail(str(exc))
            return False
        mismatch = mode_mismatch(self._capture, spec)
        if mismatch is not None:
            self._log(f"警告: {mismatch}")
            self.status.setText("警告: カメラ設定が要求どおりではありません")
        # Poll at the rate the device actually delivers; asking faster starves the
        # V4L2 buffers and the preview shows torn frames.
        self._capture_fps = actual_fps(self._capture, 30.0)
        self._timer.setInterval(max(10, round(1000.0 / self._capture_fps)))
        self._timer.start()
        self._log(f"カメラを開きました: {describe_capture(self._capture)}")
        return True

    def _release_capture_if_idle(self) -> None:
        if self._recording or self._preview_window is not None:
            return
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._log("カメラを解放しました。")
        self._observation = None

    def _ensure_detector(self) -> AprilTagDetector | None:
        if self._detector is not None:
            return self._detector
        text = self.manifest.text().strip()
        if not text:
            return None
        try:
            manifest: PatternManifest = PatternManifest.load(Path(text).expanduser())
        except (OSError, ValueError) as exc:
            self._log(f"manifestを読めません: {exc}")
            return None
        self._detector = AprilTagDetector(AprilGridBoard(manifest.pattern))
        return self._detector

    @Slot()
    def _read_frame(self) -> None:
        if self._capture is None:
            return
        ok, captured = self._capture.read()
        if not ok or captured is None:
            self._log("フレームを取得できませんでした。")
            return
        frame = as_uint8(captured)

        if self._recording and self._writer is not None:
            self._writer.write(frame)
            self._recorded_frames += 1
            self.status.setText(f"録画中 — {self._recorded_frames} フレーム")

        if self._preview_window is None:
            return
        detector = self._ensure_detector()
        if detector is not None and self._recorded_frames % 3 == 0:
            self._observation = detector.detect(frame)
        display = (
            detector.draw_overlay(frame, self._observation)
            if detector is not None
            else frame
        )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._preview_window.show_frame(image)

    # -------------------------------------------------------------- recording

    @Slot()
    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._thread is not None:
            self._fail("解析の実行中です。完了までお待ちください。")
            return
        if not self._ensure_capture() or self._capture is None:
            return
        output_dir = Path(self.output_dir.text()).expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._video_path = timestamped_video_path(output_dir)
            self._writer = create_video_writer(
                self._video_path,
                int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                self._capture_fps,
            )
        except (OSError, RuntimeError) as exc:
            self._fail(str(exc))
            return

        self._recorded_frames = 0
        self._recording = True
        self.record_button.setText("録画終了 (R)")
        self._log(f"録画開始: {self._video_path}")
        self.status.setText("録画中")

    def _stop_recording(self) -> None:
        self._recording = False
        self.record_button.setText("録画開始 (R)")
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._log(f"録画終了: {self._recorded_frames} フレーム")
        self._release_capture_if_idle()

        if self._video_path is None or self._recorded_frames == 0:
            self.status.setText("録画されたフレームがありません")
            return
        manifest_text = self.manifest.text().strip()
        if not manifest_text:
            self.status.setText("manifestが無いため解析していません")
            self._log("manifestを指定するか、パターン表示を実行してから録画してください。")
            return
        self._start_analysis(Path(manifest_text).expanduser())

    def _start_analysis(self, manifest_path: Path) -> None:
        if self._video_path is None:
            return
        output_dir = Path(self.output_dir.text()).expanduser()
        worker = PipelineWorker(
            self._video_path,
            manifest_path,
            output_dir,
            SelectionSpec(),
            CalibrationSpec(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._log)
        worker.finished.connect(self._on_analysis_finished)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        self.record_button.setEnabled(False)
        self.status.setText("解析中…")
        thread.start()

    @Slot(str)
    def _on_analysis_finished(self, message: str) -> None:
        self._log(message)
        self.status.setText(message)

    @Slot(str)
    def _on_analysis_failed(self, message: str) -> None:
        self._log(message)
        self.status.setText("解析に失敗しました")

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.record_button.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._recording = False
        self._timer.stop()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._hide_pattern()
        self._close_preview()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)


def run_panel() -> int:
    configure_physical_pixels()  # read by Qt when the application object is built
    app = QApplication.instance() or QApplication(sys.argv)
    window = PanelWindow()
    window.show()
    return int(app.exec())  # type: ignore[union-attr]

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast, get_args

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
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
    open_camera,
    probe_cameras,
    timestamped_video_path,
)
from tagcal.cvtypes import as_uint8
from tagcal.detection import AprilTagDetector, DetectionObservation
from tagcal.models import (
    AprilTagFamily,
    CalibrationSpec,
    CaptureSpec,
    PatternManifest,
    PatternSpec,
    SelectionSpec,
)
from tagcal.pattern import PatternRenderer, confirm_display_scale
from tagcal.qtworker import PipelineWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("tagcal - AprilTag Camera Calibration")
        self.resize(1040, 900)

        self._capture: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._detector: AprilTagDetector | None = None
        self._latest_observation: DetectionObservation | None = None
        self._recorded_frames = 0
        self._recording_video_path: Path | None = None
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._read_frame)

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.addWidget(self._build_pattern_group())
        root.addWidget(self._build_capture_group())

        self.preview = QLabel("録画開始後にカメラ映像を表示します。")
        self.preview.setMinimumSize(800, 420)
        self.preview.setStyleSheet("background:#161616; color:#ddd; padding:8px;")
        self.preview.setScaledContents(False)
        root.addWidget(self.preview, stretch=1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        root.addWidget(self.log, stretch=0)
        self.setCentralWidget(central)
        self._refresh_cameras()

    def _build_pattern_group(self) -> QGroupBox:
        group = QGroupBox("1. AprilTagパターン")
        layout = QGridLayout(group)

        self.pattern_output = QLineEdit(str(Path.cwd() / "artifacts" / "pattern"))
        browse_output = QPushButton("保存先")
        browse_output.clicked.connect(lambda: self._choose_directory(self.pattern_output))
        layout.addWidget(QLabel("生成先"), 0, 0)
        layout.addWidget(self.pattern_output, 0, 1, 1, 4)
        layout.addWidget(browse_output, 0, 5)

        self.columns = QSpinBox()
        self.columns.setRange(1, 20)
        self.columns.setValue(6)
        self.rows = QSpinBox()
        self.rows.setRange(1, 20)
        self.rows.setValue(4)
        self.tag_size = QDoubleSpinBox()
        self.tag_size.setRange(1.0, 500.0)
        self.tag_size.setDecimals(2)
        self.tag_size.setValue(35.0)
        self.tag_size.setSuffix(" mm")
        self.gap_size = QDoubleSpinBox()
        self.gap_size.setRange(0.0, 200.0)
        self.gap_size.setDecimals(2)
        self.gap_size.setValue(8.0)
        self.gap_size.setSuffix(" mm")
        self.family = QComboBox()
        self.family.addItems(list(get_args(AprilTagFamily)))
        self.family.setCurrentText(PatternSpec().family)

        layout.addWidget(QLabel("列"), 1, 0)
        layout.addWidget(self.columns, 1, 1)
        layout.addWidget(QLabel("行"), 1, 2)
        layout.addWidget(self.rows, 1, 3)
        layout.addWidget(QLabel("Family"), 1, 4)
        layout.addWidget(self.family, 1, 5)
        layout.addWidget(QLabel("タグ辺長"), 2, 0)
        layout.addWidget(self.tag_size, 2, 1)
        layout.addWidget(QLabel("間隔"), 2, 2)
        layout.addWidget(self.gap_size, 2, 3)

        generate = QPushButton("PNG/PDF生成")
        generate.clicked.connect(self._generate_pattern)
        layout.addWidget(generate, 2, 4, 1, 2)

        self.manifest_path = QLineEdit()
        browse_manifest = QPushButton("選択")
        browse_manifest.clicked.connect(self._choose_manifest)
        layout.addWidget(QLabel("Manifest"), 3, 0)
        layout.addWidget(self.manifest_path, 3, 1, 1, 4)
        layout.addWidget(browse_manifest, 3, 5)

        self.measured_reference = QDoubleSpinBox()
        self.measured_reference.setRange(1.0, 1000.0)
        self.measured_reference.setDecimals(3)
        self.measured_reference.setValue(100.0)
        self.measured_reference.setSuffix(" mm")
        confirm = QPushButton("表示スケール反映")
        confirm.clicked.connect(self._confirm_scale)
        open_pattern = QPushButton("パターンを開く")
        open_pattern.clicked.connect(self._open_pattern)
        layout.addWidget(QLabel("100 mm参照バーの実測値"), 4, 0, 1, 2)
        layout.addWidget(self.measured_reference, 4, 2)
        layout.addWidget(confirm, 4, 3, 1, 2)
        layout.addWidget(open_pattern, 4, 5)
        return group

    def _build_capture_group(self) -> QGroupBox:
        group = QGroupBox("2. 録画・自動選定・キャリブレーション")
        outer = QVBoxLayout(group)
        form = QFormLayout()

        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        refresh = QPushButton("再検出")
        refresh.clicked.connect(self._refresh_cameras)
        camera_row.addWidget(self.camera_combo, stretch=1)
        camera_row.addWidget(refresh)
        form.addRow("カメラ", camera_row)

        resolution_row = QHBoxLayout()
        self.capture_width = QSpinBox()
        self.capture_width.setRange(160, 7680)
        self.capture_width.setValue(1920)
        self.capture_height = QSpinBox()
        self.capture_height.setRange(120, 4320)
        self.capture_height.setValue(1080)
        self.capture_fps = QDoubleSpinBox()
        self.capture_fps.setRange(1.0, 240.0)
        self.capture_fps.setValue(30.0)
        resolution_row.addWidget(QLabel("幅"))
        resolution_row.addWidget(self.capture_width)
        resolution_row.addWidget(QLabel("高さ"))
        resolution_row.addWidget(self.capture_height)
        resolution_row.addWidget(QLabel("FPS"))
        resolution_row.addWidget(self.capture_fps)
        form.addRow("撮影モード", resolution_row)

        output_row = QHBoxLayout()
        self.session_output = QLineEdit(str(Path.cwd() / "artifacts" / "session"))
        browse_session = QPushButton("保存先")
        browse_session.clicked.connect(lambda: self._choose_directory(self.session_output))
        output_row.addWidget(self.session_output, stretch=1)
        output_row.addWidget(browse_session)
        form.addRow("セッション出力", output_row)

        analysis_row = QHBoxLayout()
        self.target_frames = QSpinBox()
        self.target_frames.setRange(10, 100)
        self.target_frames.setValue(24)
        self.min_tags = QSpinBox()
        self.min_tags.setRange(1, 100)
        self.min_tags.setValue(4)
        self.rational_model = QCheckBox("Rational distortion model")
        analysis_row.addWidget(QLabel("キーフレーム"))
        analysis_row.addWidget(self.target_frames)
        analysis_row.addWidget(QLabel("最小タグ数"))
        analysis_row.addWidget(self.min_tags)
        analysis_row.addWidget(self.rational_model)
        form.addRow("解析", analysis_row)
        outer.addLayout(form)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("レコード開始")
        self.start_button.clicked.connect(self._start_recording)
        self.stop_button = QPushButton("停止してキャリブレーション")
        self.stop_button.clicked.connect(self._stop_and_process)
        self.stop_button.setEnabled(False)
        self.process_button = QPushButton("既存動画を解析")
        self.process_button.clicked.connect(self._choose_video)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.process_button)
        outer.addLayout(buttons)
        return group

    @Slot()
    def _generate_pattern(self) -> None:
        try:
            output_dir = Path(self.pattern_output.text()).expanduser()
            spec = PatternSpec(
                columns=self.columns.value(),
                rows=self.rows.value(),
                tag_size_mm=self.tag_size.value(),
                gap_mm=self.gap_size.value(),
                family=cast(AprilTagFamily, self.family.currentText()),
            )
            artifacts = PatternRenderer().generate(output_dir, spec, output_format="both")
            self.manifest_path.setText(str(artifacts.manifest_path))
            self._append_log(f"パターンを生成しました: {artifacts.manifest_path}")
        except Exception as exc:
            self._show_error(str(exc))

    @Slot()
    def _confirm_scale(self) -> None:
        try:
            manifest = self._manifest()
            updated = confirm_display_scale(
                manifest,
                self.measured_reference.value(),
            )
            self._append_log(
                f"表示スケール={updated.pattern.display_scale:.8f}, "
                f"実効タグ辺長={updated.pattern.effective_tag_size_mm:.4f} mm"
            )
        except Exception as exc:
            self._show_error(str(exc))

    @Slot()
    def _open_pattern(self) -> None:
        try:
            manifest_path = self._manifest()
            manifest = PatternManifest.load(manifest_path)
            image_path = manifest.resolve_png(manifest_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(image_path)))
        except Exception as exc:
            self._show_error(str(exc))

    @Slot()
    def _refresh_cameras(self) -> None:
        self.camera_combo.clear()
        try:
            devices = probe_cameras(10)
            for device in devices:
                self.camera_combo.addItem(
                    f"[{device.index}] {device.width}×{device.height} "
                    f"{device.fps:.1f}fps {device.backend}",
                    device.index,
                )
            if not devices:
                self.camera_combo.addItem("カメラ未検出", -1)
        except Exception as exc:
            self.camera_combo.addItem(f"検出エラー: {exc}", -1)

    @Slot()
    def _start_recording(self) -> None:
        try:
            if self._thread is not None:
                raise RuntimeError("解析中です。完了後に録画してください。")
            manifest_path = self._manifest()
            manifest = PatternManifest.load(manifest_path)
            camera_index = int(self.camera_combo.currentData())
            if camera_index < 0:
                raise RuntimeError("利用可能なカメラを選択してください。")

            capture_spec = CaptureSpec(
                camera_index=camera_index,
                width=self.capture_width.value(),
                height=self.capture_height.value(),
                fps=self.capture_fps.value(),
                preview=False,
            )
            self._capture = open_camera(capture_spec)
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = actual_fps(self._capture, capture_spec.fps)
            self._timer.setInterval(max(10, round(1000.0 / fps)))

            output_dir = Path(self.session_output.text()).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            self._recording_video_path = timestamped_video_path(output_dir)
            self._writer = create_video_writer(
                self._recording_video_path,
                width,
                height,
                fps,
                "mp4v",
            )
            self._detector = AprilTagDetector(AprilGridBoard(manifest.pattern))
            self._recorded_frames = 0
            self._latest_observation = None
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.process_button.setEnabled(False)
            self._timer.start()
            self._append_log(
                f"録画開始: camera={camera_index}, {describe_capture(self._capture)}"
            )
        except Exception as exc:
            self._release_capture()
            self._show_error(str(exc))

    @Slot()
    def _read_frame(self) -> None:
        if self._capture is None or self._writer is None:
            return
        ok, captured = self._capture.read()
        if not ok or captured is None:
            self._show_error("カメラフレームの取得に失敗しました。")
            self._release_capture()
            return

        frame = as_uint8(captured)
        self._writer.write(frame)
        self._recorded_frames += 1
        if self._detector is not None and self._recorded_frames % 3 == 1:
            self._latest_observation = self._detector.detect(frame)
        display = (
            self._detector.draw_overlay(
                frame,
                self._latest_observation,
                f"REC frames={self._recorded_frames}",
            )
            if self._detector is not None
            else frame
        )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(),
            aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
            mode=Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    @Slot()
    def _stop_and_process(self) -> None:
        video_path = self._recording_video_path
        frames = self._recorded_frames
        self._release_capture()
        if video_path is None or frames == 0:
            self._show_error("録画フレームがありません。")
            return
        self._append_log(f"録画停止: {frames} frames, {video_path}")
        self._start_processing(video_path)

    @Slot()
    def _choose_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "解析する動画を選択",
            str(Path.cwd()),
            "Video (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if filename:
            self._start_processing(Path(filename))

    def _start_processing(self, video_path: Path) -> None:
        try:
            if self._thread is not None:
                raise RuntimeError("既に解析中です。")
            manifest_path = self._manifest()
            output_dir = Path(self.session_output.text()).expanduser()
            target = self.target_frames.value()
            min_views = min(10, target)
            selection = SelectionSpec(
                target_frames=target,
                min_keyframes=min_views,
                min_detected_tags=self.min_tags.value(),
            )
            calibration = CalibrationSpec(
                min_views=min_views,
                rational_model=self.rational_model.isChecked(),
            )
            self._thread = QThread(self)
            self._worker = PipelineWorker(
                video_path,
                manifest_path,
                output_dir,
                selection,
                calibration,
            )
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._append_log)
            self._worker.finished.connect(self._processing_finished)
            self._worker.failed.connect(self._processing_failed)
            self._worker.finished.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._processing_thread_closed)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.process_button.setEnabled(False)
            self._append_log(f"解析開始: {video_path}")
            self._thread.start()
        except Exception as exc:
            self._show_error(str(exc))

    @Slot(str)
    def _processing_finished(self, message: str) -> None:
        self._append_log(message)
        QMessageBox.information(self, "tagcal", message)

    @Slot(str)
    def _processing_failed(self, trace: str) -> None:
        self._append_log(trace)
        self._show_error(trace.splitlines()[-1] if trace.splitlines() else trace)

    @Slot()
    def _processing_thread_closed(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.start_button.setEnabled(True)
        self.process_button.setEnabled(True)

    def _release_capture(self) -> None:
        self._timer.stop()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._detector = None
        self._latest_observation = None
        self.start_button.setEnabled(self._thread is None)
        self.stop_button.setEnabled(False)
        self.process_button.setEnabled(self._thread is None)

    def _manifest(self) -> Path:
        value = self.manifest_path.text().strip()
        if not value:
            raise ValueError("Pattern manifestを選択してください。")
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _choose_manifest(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Pattern manifestを選択",
            str(Path.cwd()),
            "Pattern manifest (pattern.json);;JSON (*.json)",
        )
        if filename:
            self.manifest_path.setText(filename)

    def _choose_directory(self, target: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "ディレクトリを選択",
            target.text() or str(Path.cwd()),
        )
        if directory:
            target.setText(directory)

    @Slot(str)
    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _show_error(self, message: str) -> None:
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "tagcal error", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(
                self,
                "tagcal",
                "解析処理中は終了できません。処理完了後にウィンドウを閉じてください。",
            )
            event.ignore()
            return
        self._release_capture()
        event.accept()


def run_gui() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1
AprilTagFamily = Literal["tag16h5", "tag25h9", "tag36h10", "tag36h11"]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero: {value}")


def _non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative: {value}")


@dataclass(frozen=True, slots=True)
class PatternSpec:
    columns: int = 6
    rows: int = 4
    tag_size_mm: float = 35.0
    gap_mm: float = 8.0
    margin_mm: float = 10.0
    family: AprilTagFamily = "tag36h11"
    first_id: int = 0
    border_bits: int = 1
    reference_bar_mm: float = 100.0
    display_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.columns < 1 or self.rows < 1:
            raise ValueError("columns and rows must be at least 1")
        _positive("tag_size_mm", self.tag_size_mm)
        _non_negative("gap_mm", self.gap_mm)
        _non_negative("margin_mm", self.margin_mm)
        _positive("reference_bar_mm", self.reference_bar_mm)
        _positive("display_scale", self.display_scale)
        if self.first_id < 0:
            raise ValueError("first_id must be non-negative")
        if self.border_bits < 1:
            raise ValueError("border_bits must be at least 1")

    @property
    def marker_count(self) -> int:
        return self.columns * self.rows

    @property
    def nominal_board_width_mm(self) -> float:
        return self.columns * self.tag_size_mm + (self.columns - 1) * self.gap_mm

    @property
    def nominal_board_height_mm(self) -> float:
        return self.rows * self.tag_size_mm + (self.rows - 1) * self.gap_mm

    @property
    def effective_tag_size_mm(self) -> float:
        return self.tag_size_mm * self.display_scale

    @property
    def effective_gap_mm(self) -> float:
        return self.gap_mm * self.display_scale

    @property
    def effective_board_width_mm(self) -> float:
        return self.nominal_board_width_mm * self.display_scale

    @property
    def effective_board_height_mm(self) -> float:
        return self.nominal_board_height_mm * self.display_scale

    def with_measured_reference(self, measured_reference_mm: float) -> PatternSpec:
        _positive("measured_reference_mm", measured_reference_mm)
        return replace(self, display_scale=measured_reference_mm / self.reference_bar_mm)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PatternSpec:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PatternManifest:
    pattern: PatternSpec
    png_path: str
    pdf_path: str | None
    dpi: int
    page: str
    image_width_px: int
    image_height_px: int
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")
        if self.image_width_px < 1 or self.image_height_px < 1:
            raise ValueError("image dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PatternManifest:
        copied = dict(value)
        copied["pattern"] = PatternSpec.from_dict(copied["pattern"])
        return cls(**copied)

    @classmethod
    def load(cls, path: Path) -> PatternManifest:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        manifest = cls.from_dict(data)
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema: {manifest.schema_version}; expected {SCHEMA_VERSION}"
            )
        return manifest

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def resolve_png(self, manifest_path: Path) -> Path:
        png = Path(self.png_path)
        return png if png.is_absolute() else manifest_path.parent / png

    def resolve_pdf(self, manifest_path: Path) -> Path | None:
        if self.pdf_path is None:
            return None
        pdf = Path(self.pdf_path)
        return pdf if pdf.is_absolute() else manifest_path.parent / pdf

    def with_pattern(self, pattern: PatternSpec) -> PatternManifest:
        return replace(self, pattern=pattern)


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    camera_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    duration_seconds: float | None = None
    codec: str = "mp4v"
    preview: bool = True
    input_fourcc: str | None = "MJPG"
    """Pixel format requested from the camera.

    Left unset, V4L2 hands out uncompressed YUYV, which USB bandwidth caps at a
    few frames per second above VGA -- the capture then lags and tears. MJPG is
    lossy but is the only way most webcams reach full resolution at full rate.
    Use "YUYV" when uncompressed pixels matter more than frame rate, or None to
    accept whatever the driver defaults to.
    """

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        if self.width < 1 or self.height < 1:
            raise ValueError("capture width and height must be positive")
        _positive("fps", self.fps)
        if self.duration_seconds is not None:
            _positive("duration_seconds", self.duration_seconds)
        if len(self.codec) != 4:
            raise ValueError("codec must be a four-character code")
        if self.input_fourcc is not None and len(self.input_fourcc) != 4:
            raise ValueError("input_fourcc must be a four-character code")


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    sample_fps: float = 12.0
    """How densely the video is examined. Must be fine enough that several frames
    fall inside `blur_window_seconds`, otherwise there is nothing to compare."""
    target_frames: int = 24
    min_keyframes: int = 10
    max_candidates: int = 500
    min_detected_tags: int = 4
    min_board_coverage: float = 0.025
    min_sharpness: float = 0.0
    """Absolute sharpness floor. Off by default: the measure's scale is scene
    dependent, so blur is judged against neighbouring frames instead."""
    blur_window_seconds: float = 0.20
    """Half-width of the window a frame must be the sharpest in to be kept.

    Chosen by sweeping a real recording and reading the intrinsic standard
    deviations rather than the RMS: a wider window keeps fewer, sharper frames and
    lowers the RMS, but it also throws away distinct poses, and the uncertainty on
    every parameter grows. At 0.6s only 16-18 views survive and sigma(fx) doubles.
    """
    blur_relative_floor: float = 0.55
    """Extra cutoff against the recording's median sharpness, so a stretch where
    every frame is blurred does not contribute its least-blurred frame.

    Kept deliberately loose. Raising it to 1.2 halves the RMS-relevant view count
    and sigma(fx) quadruples -- the surviving frames are all from the moments the
    operator held still, which is exactly the pose degeneracy calibration must
    avoid. Blur rejection must not become pose selection.
    """
    min_candidate_distance: float = 0.035
    jpeg_quality: int = 94

    def __post_init__(self) -> None:
        _positive("sample_fps", self.sample_fps)
        if self.target_frames < 3:
            raise ValueError("target_frames must be at least 3")
        if self.min_keyframes < 3 or self.min_keyframes > self.target_frames:
            raise ValueError("min_keyframes must be between 3 and target_frames")
        if self.max_candidates < self.target_frames:
            raise ValueError("max_candidates must be at least target_frames")
        if self.min_detected_tags < 1:
            raise ValueError("min_detected_tags must be at least 1")
        if not 0.0 <= self.min_board_coverage <= 1.0:
            raise ValueError("min_board_coverage must be between 0 and 1")
        _non_negative("min_sharpness", self.min_sharpness)
        _positive("blur_window_seconds", self.blur_window_seconds)
        if not 0.0 <= self.blur_relative_floor <= 1.0:
            raise ValueError("blur_relative_floor must be between 0 and 1")
        _non_negative("min_candidate_distance", self.min_candidate_distance)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    min_views: int = 10
    max_outlier_iterations: int = 8
    outlier_mad_scale: float = 3.0
    min_outlier_threshold_px: float = 0.65
    max_view_error_px: float = 1.50
    rational_model: bool = False
    zero_tangent_distortion: bool = False
    fix_principal_point: bool = False

    def __post_init__(self) -> None:
        if self.min_views < 3:
            raise ValueError("min_views must be at least 3")
        if self.max_outlier_iterations < 0:
            raise ValueError("max_outlier_iterations must be non-negative")
        _positive("outlier_mad_scale", self.outlier_mad_scale)
        _positive("min_outlier_threshold_px", self.min_outlier_threshold_px)
        _positive("max_view_error_px", self.max_view_error_px)
        if self.min_outlier_threshold_px > self.max_view_error_px:
            raise ValueError("min_outlier_threshold_px cannot exceed max_view_error_px")


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    detected_tags: int
    detected_points: int
    sharpness: float
    board_coverage: float
    center_x: float
    center_y: float
    quality_score: float
    descriptor: list[float]


@dataclass(frozen=True, slots=True)
class KeyframeRecord:
    image_path: str
    frame_index: int
    timestamp_seconds: float
    marker_ids: list[int]
    object_points: list[list[float]]
    image_points: list[list[float]]
    metrics: FrameMetrics

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KeyframeRecord:
        copied = dict(value)
        copied["metrics"] = FrameMetrics(**copied["metrics"])
        return cls(**copied)


@dataclass(frozen=True, slots=True)
class SelectionReport:
    video_path: str
    image_width: int
    image_height: int
    source_fps: float
    total_frames: int
    sampled_frames: int
    valid_candidates: int
    selected: list[KeyframeRecord]
    rejected_summary: dict[str, int]
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> SelectionReport:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        copied = dict(value)
        copied["selected"] = [KeyframeRecord.from_dict(item) for item in copied["selected"]]
        report = cls(**copied)
        if report.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported selection schema: {report.schema_version}; expected {SCHEMA_VERSION}"
            )
        return report


@dataclass(frozen=True, slots=True)
class CalibrationView:
    image_path: str
    frame_index: int
    timestamp_seconds: float
    reprojection_error_px: float
    rotation_vector: list[float]
    translation_vector: list[float]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    image_width: int
    image_height: int
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    distortion_model: str
    rms_reprojection_error_px: float
    mean_view_error_px: float
    median_view_error_px: float
    max_view_error_px: float
    intrinsic_standard_deviations: list[float]
    used_views: list[CalibrationView]
    excluded_views: list[dict[str, Any]]
    flags: int
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

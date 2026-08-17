from pathlib import Path

import cv2
from PIL import Image

from tagcal.board import AprilGridBoard
from tagcal.cvtypes import as_uint8
from tagcal.detection import AprilTagDetector
from tagcal.models import PatternManifest, PatternSpec
from tagcal.pattern import PatternRenderer, confirm_display_scale
from tagcal.pipeline import CalibrationPipeline


def test_pattern_generation_and_detection(tmp_path: Path) -> None:
    spec = PatternSpec(
        columns=4,
        rows=3,
        tag_size_mm=24.0,
        gap_mm=6.0,
        margin_mm=8.0,
        family="tag36h11",
        reference_bar_mm=100.0,
        border_bits=2,
    )
    artifacts = PatternRenderer().generate(
        tmp_path,
        spec,
        dpi=180,
        page="board",
        output_format="both",
    )

    assert artifacts.manifest_path.exists()
    assert artifacts.png_path.exists()
    assert artifacts.pdf_path is not None and artifacts.pdf_path.exists()

    manifest = PatternManifest.load(artifacts.manifest_path)
    with Image.open(artifacts.png_path) as image:
        assert image.size == (manifest.image_width_px, manifest.image_height_px)

    frame = cv2.imread(str(artifacts.png_path), cv2.IMREAD_COLOR)
    assert frame is not None
    observation = AprilTagDetector(AprilGridBoard(spec)).detect(as_uint8(frame))
    assert observation is not None
    assert observation.metrics.detected_tags == spec.marker_count
    assert observation.metrics.board_coverage > 0.1


def test_display_scale_confirmation(tmp_path: Path) -> None:
    artifacts = PatternRenderer().generate(
        tmp_path,
        PatternSpec(tag_size_mm=40.0, gap_mm=10.0, reference_bar_mm=100.0),
        dpi=150,
        output_format="png",
    )
    updated = confirm_display_scale(artifacts.manifest_path, 95.0)

    assert updated.pattern.display_scale == 0.95
    assert updated.pattern.effective_tag_size_mm == 38.0
    assert updated.pattern.effective_gap_mm == 9.5


def test_scale_manifest_rebases_assets_when_written_elsewhere(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    artifacts = PatternRenderer().generate(
        source_dir,
        PatternSpec(columns=2, rows=2),
        dpi=150,
        output_format="both",
    )
    output_manifest = target_dir / "scaled.json"
    updated = confirm_display_scale(
        artifacts.manifest_path,
        102.0,
        output_path=output_manifest,
    )

    assert output_manifest.exists()
    assert artifacts.pdf_path is not None
    assert updated.resolve_png(output_manifest).resolve() == artifacts.png_path.resolve()
    resolved_pdf = updated.resolve_pdf(output_manifest)
    assert resolved_pdf is not None
    assert resolved_pdf.resolve() == artifacts.pdf_path.resolve()


def test_session_manifest_is_self_contained(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    scaled_dir = tmp_path / "scaled"
    session_dir = tmp_path / "session"
    artifacts = PatternRenderer().generate(
        source_dir,
        PatternSpec(columns=3, rows=2),
        dpi=150,
        output_format="both",
    )
    scaled_manifest_path = scaled_dir / "pattern.json"
    scaled_manifest = confirm_display_scale(
        artifacts.manifest_path,
        98.5,
        output_path=scaled_manifest_path,
    )

    session_dir.mkdir()
    CalibrationPipeline._copy_pattern_inputs(
        scaled_manifest,
        scaled_manifest_path,
        session_dir,
    )

    session_manifest_path = session_dir / "pattern.json"
    session_manifest = PatternManifest.load(session_manifest_path)
    assert artifacts.pdf_path is not None
    assert session_manifest.resolve_png(session_manifest_path).resolve() == (
        session_dir / artifacts.png_path.name
    ).resolve()
    resolved_pdf = session_manifest.resolve_pdf(session_manifest_path)
    assert resolved_pdf is not None
    assert resolved_pdf.resolve() == (session_dir / artifacts.pdf_path.name).resolve()

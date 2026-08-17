import numpy as np
import pytest

from tagcal.board import AprilGridBoard
from tagcal.detection import AprilTagDetector
from tagcal.models import PatternManifest, PatternSpec
from tagcal.screen import plan_layout, render_board, write_manifest

# Pixel pitch of a 1920x1080 / 527mm panel; deliberately not a round number.
PX_PER_MM = 1920 / 527.0


def test_snapped_layout_has_uniform_cells() -> None:
    layout = plan_layout(
        PatternSpec(columns=3, rows=2, tag_size_mm=40.0, gap_mm=10.0),
        px_per_mm=PX_PER_MM,
        snap=True,
    )

    assert layout.tag_px % layout.cells_per_tag == 0
    assert len(set(layout.cell_widths_px)) == 1
    assert layout.cell_spread_mm == 0.0
    # The manifest must describe the pixels that were rendered, not what was asked for.
    assert layout.spec.effective_tag_size_mm == pytest.approx(layout.actual_tag_size_mm)
    assert layout.spec.display_scale == 1.0
    assert abs(layout.tag_size_error_mm) < layout.actual_tag_size_mm / layout.cells_per_tag


def test_unsnapped_layout_reports_its_cell_spread() -> None:
    layout = plan_layout(
        PatternSpec(columns=2, rows=2, tag_size_mm=40.0),
        px_per_mm=PX_PER_MM,
        snap=False,
    )

    assert layout.cell_spread_mm > 0.0
    assert abs(layout.tag_size_error_mm) < 1.0 / PX_PER_MM


def test_calibration_factor_scales_the_pitch() -> None:
    spec = PatternSpec(columns=2, rows=2, tag_size_mm=40.0)
    nominal = plan_layout(spec, px_per_mm=PX_PER_MM, snap=False)
    corrected = plan_layout(spec, px_per_mm=PX_PER_MM, calibration_factor=0.95, snap=False)

    assert corrected.px_per_mm == pytest.approx(PX_PER_MM * 0.95)
    assert corrected.tag_px < nominal.tag_px


def test_rendered_board_detects_at_the_declared_size() -> None:
    spec = PatternSpec(columns=3, rows=2, tag_size_mm=45.0, gap_mm=12.0)
    layout = plan_layout(spec, px_per_mm=PX_PER_MM, snap=True)
    image = render_board(layout, show_info=False, show_ruler=False)

    board = AprilGridBoard(layout.spec)
    observation = AprilTagDetector(board).detect(image)
    assert observation is not None
    assert observation.metrics.detected_tags == spec.marker_count

    corners = observation.marker_corners[0].reshape(4, 2)
    side_px = float(
        np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])
    )
    measured_mm = side_px / layout.px_per_mm
    # aruco returns the centre of the outermost black pixel, so a synthetic render
    # measures ~1px short; on a real camera the blurred edge removes that bias.
    assert measured_mm == pytest.approx(layout.actual_tag_size_mm, abs=1.5 / PX_PER_MM)


def test_written_manifest_round_trips(tmp_path) -> None:
    layout = plan_layout(
        PatternSpec(columns=2, rows=2, tag_size_mm=30.0, gap_mm=8.0),
        px_per_mm=PX_PER_MM,
        snap=True,
    )
    manifest = write_manifest(layout, tmp_path)
    loaded = PatternManifest.load(tmp_path / "pattern.json")

    assert loaded.resolve_png(tmp_path / "pattern.json").exists()
    assert loaded.pattern.effective_tag_size_mm == pytest.approx(layout.actual_tag_size_mm)
    assert loaded.image_width_px == manifest.image_width_px

"""Physically accurate display of the AprilTag board on a monitor.

A monitor is a far flatter target than a printed sheet glued to a board, which
matters because calibration assumes the target is planar. It is only usable as a
metric target when three separate things hold, and each one silently corrupts the
scale when it does not:

1. The pixel pitch is known. Taken from the EDID via `xrandr`, which is often
   rounded and occasionally absent, so it must be verifiable against a ruler
   (`draw_ruler`) and correctable (`calibration_factor`).
2. The tag size definition matches the detector's. A tag size is the outer edge
   of the black border. AprilTag images are also distributed with an extra white
   cell on each side; measuring that as the tag inflates every derived distance
   by the ratio of the two (1.25x for tag36h11).
3. One image pixel maps to one physical pixel. Qt's HiDPI scaling, a scaled
   pixmap, or a resizable window each break this, so scaling is disabled and the
   image is placed without resampling.

Cell quantisation (`snap`) is the remaining trade-off: pixels per mm is generally
irrational, so a round millimetre size and integer cell widths cannot both hold.
Uneven cells distort the tag coordinate system, while a non-round actual size
costs nothing as long as the manifest records it -- which is what this module
writes out. So cells are snapped by default and the resulting true size is what
gets reported downstream.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from tagcal.board import AprilGridBoard
from tagcal.models import PatternManifest, PatternSpec

MM_PER_INCH = 25.4
_RULER_HEIGHT_PX = 110
_INFO_LINE_HEIGHT_PX = 22

_XRANDR_PATTERN = re.compile(
    r"^(?P<name>\S+) connected (?P<primary>primary )?"
    r"(?P<width>\d+)x(?P<height>\d+)\+(?P<x>\d+)\+(?P<y>\d+)"
    r".*?(?P<width_mm>\d+)mm x (?P<height_mm>\d+)mm",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class Monitor:
    """A connected output and the physical size it reports through EDID."""

    name: str
    width_px: int
    height_px: int
    x: int
    y: int
    width_mm: float
    height_mm: float
    primary: bool

    @property
    def px_per_mm(self) -> float:
        return self.width_px / self.width_mm

    @property
    def px_per_mm_vertical(self) -> float:
        return self.height_px / self.height_mm

    @property
    def ppi(self) -> float:
        return self.px_per_mm * MM_PER_INCH

    @property
    def aspect_mismatch_percent(self) -> float:
        """Horizontal vs vertical pixel pitch disagreement; a rounding-level check on EDID."""
        return abs(self.px_per_mm / self.px_per_mm_vertical - 1.0) * 100.0

    def describe(self) -> str:
        return (
            f"{self.name} {self.width_px}x{self.height_px} @({self.x},{self.y}) "
            f"{self.width_mm:.0f}x{self.height_mm:.0f}mm "
            f"{self.px_per_mm:.4f}px/mm ({self.ppi:.1f}PPI) "
            f"H/V {self.aspect_mismatch_percent:.2f}%"
            + (" [primary]" if self.primary else "")
        )


@dataclass(frozen=True, slots=True)
class ScreenLayout:
    """A board quantised to whole screen pixels, plus the true size that produces."""

    spec: PatternSpec
    """Effective spec: sizes rewritten to what the screen actually shows."""

    monitor: Monitor | None
    px_per_mm: float
    cell_px: int
    tag_px: int
    gap_px: int
    margin_px: int
    requested_tag_size_mm: float
    cells_per_tag: int
    snapped: bool

    @property
    def actual_tag_size_mm(self) -> float:
        return self.tag_px / self.px_per_mm

    @property
    def tag_size_error_mm(self) -> float:
        return self.actual_tag_size_mm - self.requested_tag_size_mm

    @property
    def board_width_px(self) -> int:
        return self.spec.columns * self.tag_px + (self.spec.columns - 1) * self.gap_px

    @property
    def board_height_px(self) -> int:
        return self.spec.rows * self.tag_px + (self.spec.rows - 1) * self.gap_px

    @property
    def canvas_width_px(self) -> int:
        return self.board_width_px + 2 * self.margin_px

    @property
    def cell_widths_px(self) -> list[int]:
        """Width each cell really occupies after nearest-neighbour scaling."""
        index = np.floor((np.arange(self.tag_px) + 0.5) * self.cells_per_tag / self.tag_px)
        return np.bincount(index.astype(int), minlength=self.cells_per_tag).tolist()

    @property
    def cell_spread_mm(self) -> float:
        """Spread between the widest and narrowest cell; zero when snapped."""
        widths = self.cell_widths_px
        return (max(widths) - min(widths)) / self.px_per_mm

    def describe(self) -> list[str]:
        cell = f"{self.cell_px} px" if self.snapped else f"{self.tag_px / self.cells_per_tag:.3f} px"
        lines = [
            f"px/mm         : {self.px_per_mm:.4f} ({self.px_per_mm * MM_PER_INCH:.1f} PPI)",
            f"grid          : {self.spec.columns} x {self.spec.rows} {self.spec.family}"
            f" IDs {self.spec.first_id}..{self.spec.first_id + self.spec.marker_count - 1}",
            f"cell          : {cell} ({self.cells_per_tag} cells incl. black border)",
            f"tag           : {self.tag_px} px",
            f"ACTUAL tag    : {self.actual_tag_size_mm:.4f} mm "
            f"(requested {self.requested_tag_size_mm:g} mm, "
            f"error {self.tag_size_error_mm:+.4f} mm)",
            f"ACTUAL gap    : {self.spec.gap_mm:.4f} mm ({self.gap_px} px)",
            f"board         : {self.board_width_px} x {self.board_height_px} px",
        ]
        if not self.snapped:
            lines.append(
                f"cell spread   : {self.cell_spread_mm:.4f} mm "
                "-- uneven cells; --snap removes this"
            )
        return lines


def query_monitors() -> list[Monitor]:
    """Enumerate connected outputs through xrandr (X11)."""
    try:
        output = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"xrandr failed; on-screen display needs X11: {exc}") from exc

    monitors = [
        Monitor(
            name=match["name"],
            width_px=int(match["width"]),
            height_px=int(match["height"]),
            x=int(match["x"]),
            y=int(match["y"]),
            width_mm=float(match["width_mm"]),
            height_mm=float(match["height_mm"]),
            primary=bool(match["primary"]),
        )
        for match in (found.groupdict() for found in _XRANDR_PATTERN.finditer(output))
    ]
    if not monitors:
        raise RuntimeError("No connected monitor reported a physical size (check xrandr --query)")
    return monitors


def pick_monitor(monitors: list[Monitor], name: str | None) -> Monitor:
    if name is None:
        return next((monitor for monitor in monitors if monitor.primary), monitors[0])
    for monitor in monitors:
        if monitor.name == name:
            return monitor
    available = ", ".join(monitor.name for monitor in monitors)
    raise ValueError(f"Monitor {name!r} not found. Available: {available}")


def plan_layout(
    spec: PatternSpec,
    *,
    monitor: Monitor | None = None,
    px_per_mm: float | None = None,
    calibration_factor: float = 1.0,
    snap: bool = True,
) -> ScreenLayout:
    """Quantise a board to whole pixels and report the size it will really have.

    `calibration_factor` scales the pixel pitch: measure the on-screen ruler and
    multiply by measured/nominal to correct an EDID that is off.
    """
    if px_per_mm is None:
        if monitor is None:
            raise ValueError("Either a monitor or an explicit px_per_mm is required")
        px_per_mm = monitor.px_per_mm
    if px_per_mm <= 0.0:
        raise ValueError("px_per_mm must be positive")
    if calibration_factor <= 0.0:
        raise ValueError("calibration_factor must be positive")

    pitch = px_per_mm * calibration_factor
    board = AprilGridBoard(spec)
    cells = board.cells_per_tag
    wanted_px = spec.effective_tag_size_mm * pitch

    if snap:
        cell_px = max(1, round(wanted_px / cells))
        tag_px = cell_px * cells
    else:
        tag_px = max(cells, round(wanted_px))
        cell_px = max(1, round(tag_px / cells))

    gap_px = max(0, round(spec.effective_gap_mm * pitch))
    margin_px = max(1, round(spec.margin_mm * pitch))

    # The manifest must describe the screen, not the request: sizes are rewritten
    # to what the quantised pixels represent, so display_scale is no longer needed.
    effective = replace(
        spec,
        tag_size_mm=tag_px / pitch,
        gap_mm=gap_px / pitch,
        margin_mm=margin_px / pitch,
        display_scale=1.0,
    )
    return ScreenLayout(
        spec=effective,
        monitor=monitor,
        px_per_mm=pitch,
        cell_px=cell_px,
        tag_px=tag_px,
        gap_px=gap_px,
        margin_px=margin_px,
        requested_tag_size_mm=spec.effective_tag_size_mm,
        cells_per_tag=cells,
        snapped=snap,
    )


def suggest_tag_sizes(
    layout: ScreenLayout,
    *,
    span: int = 4,
) -> list[tuple[int, int, float]]:
    """Cell sizes near the requested one, as (cell_px, tag_px, actual_mm)."""
    center = max(1, round(layout.requested_tag_size_mm * layout.px_per_mm / layout.cells_per_tag))
    return [
        (cell, cell * layout.cells_per_tag, cell * layout.cells_per_tag / layout.px_per_mm)
        for cell in range(max(1, center - span), center + span + 1)
    ]


def draw_ruler(
    canvas: NDArray[np.uint8],
    px_per_mm: float,
    origin: tuple[int, int],
    length_mm: int = 100,
) -> None:
    """Draw a millimetre scale so the pixel pitch can be checked against a real ruler."""
    x0, y0 = origin
    end_x = x0 + round(length_mm * px_per_mm)
    cv2.line(canvas, (x0, y0), (end_x, y0), 0, 1, cv2.LINE_8)
    for millimetre in range(length_mm + 1):
        x = x0 + round(millimetre * px_per_mm)
        if millimetre % 10 == 0:
            tick = 18
            cv2.putText(
                canvas,
                str(millimetre),
                (x - 6, y0 + tick + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                0,
                1,
                cv2.LINE_AA,
            )
        elif millimetre % 5 == 0:
            tick = 11
        else:
            tick = 6
        cv2.line(canvas, (x, y0), (x, y0 + tick), 0, 1, cv2.LINE_8)
    cv2.putText(
        canvas,
        f"{length_mm} mm",
        (end_x + 10, y0 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        0,
        1,
        cv2.LINE_AA,
    )


def render_board(
    layout: ScreenLayout,
    *,
    show_info: bool = True,
    show_ruler: bool = False,
    size: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    """Compose the board at 1:1 on white, with the margin kept clear of annotations."""
    board = AprilGridBoard(layout.spec)
    grid = board.render_grid_pixels(tag_px=layout.tag_px, gap_px=layout.gap_px)

    info = _info_lines(layout) if show_info else []
    footer = (len(info) * _INFO_LINE_HEIGHT_PX + 14 if info else 0) + (
        _RULER_HEIGHT_PX if show_ruler else 0
    )
    width, height = size or (
        layout.canvas_width_px,
        layout.board_height_px + 2 * layout.margin_px + footer,
    )
    canvas: NDArray[np.uint8] = np.full((height, width), 255, dtype=np.uint8)

    grid_height, grid_width = grid.shape
    left = max(0, (width - grid_width) // 2)
    top = max(0, (height - footer - grid_height) // 2)
    canvas[top : top + grid_height, left : left + grid_width] = grid[
        : max(0, height - footer - top), : max(0, width - left)
    ]

    # Annotations live below the board so they never intrude on the quiet zone.
    y = height - footer + 8
    for line in info:
        y += _INFO_LINE_HEIGHT_PX
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv2.LINE_AA)
    if show_ruler:
        draw_ruler(canvas, layout.px_per_mm, (30, y + 38))
    return canvas


def _info_lines(layout: ScreenLayout) -> list[str]:
    monitor = layout.monitor.name if layout.monitor else "custom"
    return [
        f"{layout.spec.family} {layout.spec.columns}x{layout.spec.rows}  "
        f"tag={layout.actual_tag_size_mm:.3f}mm  gap={layout.spec.gap_mm:.3f}mm  "
        f"cell={layout.cell_px}px  {layout.px_per_mm:.4f}px/mm  [{monitor}]",
    ]


def write_manifest(
    layout: ScreenLayout,
    output_dir: Path,
    *,
    stem: str = "apriltag_board",
) -> PatternManifest:
    """Save the displayed board and a manifest whose sizes match the screen exactly.

    The manifest is what `tagcal process` consumes, so writing the quantised sizes
    here is what keeps the detected scale honest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image = render_board(layout, show_info=False, show_ruler=False)
    png_path = output_dir / f"{stem}.png"
    cv2.imwrite(str(png_path), image)

    manifest = PatternManifest(
        pattern=layout.spec,
        png_path=png_path.name,
        pdf_path=None,
        dpi=round(layout.px_per_mm * MM_PER_INCH),
        page="board",
        image_width_px=int(image.shape[1]),
        image_height_px=int(image.shape[0]),
    )
    manifest.save(output_dir / "pattern.json")
    return manifest


def configure_physical_pixels() -> None:
    """Disable Qt's logical-pixel scaling. Qt reads these when the application is built.

    Without this a 'pixel' becomes a device-independent unit and the board is drawn
    at whatever the desktop scale factor says, which silently rescales the target.
    """
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")

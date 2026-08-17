from __future__ import annotations

import os
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from tagcal.board import AprilGridBoard
from tagcal.models import PatternManifest, PatternSpec

PageKind = Literal["board", "a4", "letter"]
OutputFormat = Literal["png", "pdf", "both"]
MM_PER_INCH = 25.4
POINTS_PER_MM = 72.0 / MM_PER_INCH


@dataclass(frozen=True, slots=True)
class PatternArtifacts:
    manifest_path: Path
    png_path: Path
    pdf_path: Path | None


class PatternRenderer:
    """Render a metric AprilTag grid to PNG/PDF and write a machine-readable manifest."""

    def generate(
        self,
        output_dir: Path,
        spec: PatternSpec,
        *,
        dpi: int = 300,
        page: PageKind = "board",
        output_format: OutputFormat = "both",
        stem: str = "apriltag_board",
    ) -> PatternArtifacts:
        if dpi < 72:
            raise ValueError("dpi must be at least 72")
        if output_format not in {"png", "pdf", "both"}:
            raise ValueError(f"Unsupported output format: {output_format}")
        if page not in {"board", "a4", "letter"}:
            raise ValueError(f"Unsupported page: {page}")

        output_dir.mkdir(parents=True, exist_ok=True)
        board = AprilGridBoard(spec)
        image = self._render_image(board, dpi)

        png_path = output_dir / f"{stem}.png"
        image.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=True)

        pdf_path: Path | None = None
        if output_format in {"pdf", "both"}:
            pdf_path = output_dir / f"{stem}.pdf"
            self._write_pdf(image, pdf_path, dpi=dpi, page=page)

        # PNG is always emitted because the GUI and manifest use it as the canonical preview.
        manifest = PatternManifest(
            pattern=spec,
            png_path=png_path.name,
            pdf_path=pdf_path.name if pdf_path is not None else None,
            dpi=dpi,
            page=page,
            image_width_px=image.width,
            image_height_px=image.height,
        )
        manifest_path = output_dir / "pattern.json"
        manifest.save(manifest_path)
        return PatternArtifacts(
            manifest_path=manifest_path,
            png_path=png_path,
            pdf_path=pdf_path,
        )

    def _render_image(self, board: AprilGridBoard, dpi: int) -> Image.Image:
        spec = board.spec
        pixels_per_mm = dpi / MM_PER_INCH

        tag_px = max(32, round(spec.tag_size_mm * pixels_per_mm))
        gap_px = max(0, round(spec.gap_mm * pixels_per_mm))
        margin_px = max(1, round(spec.margin_mm * pixels_per_mm))
        board_width_px = spec.columns * tag_px + (spec.columns - 1) * gap_px
        board_height_px = spec.rows * tag_px + (spec.rows - 1) * gap_px

        reference_width_px = round(spec.reference_bar_mm * pixels_per_mm)
        reference_top_gap_px = round(8.0 * pixels_per_mm)
        reference_tick_px = max(3, round(5.0 * pixels_per_mm))
        reference_line_px = max(2, round(1.2 * pixels_per_mm))
        text_area_px = max(28, round(10.0 * pixels_per_mm))

        content_width_px = max(board_width_px, reference_width_px)
        image_width_px = content_width_px + 2 * margin_px
        image_height_px = (
            margin_px
            + board_height_px
            + reference_top_gap_px
            + reference_tick_px
            + text_area_px
            + margin_px
        )
        image = Image.new("L", (image_width_px, image_height_px), color=255)

        board_left = margin_px + (content_width_px - board_width_px) // 2
        board_top = margin_px
        grid = board.render_grid_pixels(tag_px=tag_px, gap_px=gap_px)
        image.paste(Image.fromarray(grid, mode="L"), (board_left, board_top))

        draw = ImageDraw.Draw(image)
        reference_left = margin_px + (content_width_px - reference_width_px) // 2
        reference_y = board_top + board_height_px + reference_top_gap_px
        reference_right = reference_left + reference_width_px
        draw.rectangle(
            [
                reference_left,
                reference_y - reference_line_px // 2,
                reference_right,
                reference_y + reference_line_px // 2,
            ],
            fill=0,
        )
        draw.rectangle(
            [
                reference_left - reference_line_px // 2,
                reference_y - reference_tick_px // 2,
                reference_left + reference_line_px // 2,
                reference_y + reference_tick_px // 2,
            ],
            fill=0,
        )
        draw.rectangle(
            [
                reference_right - reference_line_px // 2,
                reference_y - reference_tick_px // 2,
                reference_right + reference_line_px // 2,
                reference_y + reference_tick_px // 2,
            ],
            fill=0,
        )

        label = (
            f"REFERENCE {spec.reference_bar_mm:g} mm  |  "
            f"TAG {spec.tag_size_mm:g} mm  |  GAP {spec.gap_mm:g} mm  |  "
            f"{spec.family.upper()} IDs {spec.first_id}-{spec.first_id + spec.marker_count - 1}"
        )
        font = ImageFont.load_default(size=max(12, round(2.8 * pixels_per_mm)))
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = max(margin_px, (image_width_px - text_width) // 2)
        text_y = reference_y + reference_tick_px // 2 + max(3, round(1.5 * pixels_per_mm))
        draw.text((text_x, text_y), label, fill=0, font=font)
        return image

    def _write_pdf(self, image: Image.Image, path: Path, *, dpi: int, page: PageKind) -> None:
        image_width_mm = image.width * MM_PER_INCH / dpi
        image_height_mm = image.height * MM_PER_INCH / dpi
        image_width_points = image_width_mm * POINTS_PER_MM
        image_height_points = image_height_mm * POINTS_PER_MM

        if page == "board":
            page_size = (image_width_points, image_height_points)
            x = 0.0
            y = 0.0
        else:
            page_size = A4 if page == "a4" else LETTER
            printable_margin = 5.0 * POINTS_PER_MM
            if (
                image_width_points > page_size[0] - 2 * printable_margin
                or image_height_points > page_size[1] - 2 * printable_margin
            ):
                raise ValueError(
                    "The pattern does not fit on the selected page at 100% scale. "
                    "Use --page board, reduce dimensions, or use a larger display."
                )
            x = (page_size[0] - image_width_points) / 2.0
            y = (page_size[1] - image_height_points) / 2.0

        buffer = BytesIO()
        image.save(buffer, format="PNG", dpi=(dpi, dpi))
        buffer.seek(0)
        pdf = canvas.Canvas(str(path), pagesize=page_size, pageCompression=1)
        pdf.drawImage(
            ImageReader(buffer),
            x,
            y,
            width=image_width_points,
            height=image_height_points,
            preserveAspectRatio=True,
            mask="auto",
        )
        pdf.showPage()
        pdf.save()


def confirm_display_scale(
    manifest_path: Path,
    measured_reference_mm: float,
    output_path: Path | None = None,
) -> PatternManifest:
    manifest = PatternManifest.load(manifest_path)
    updated = manifest.with_pattern(
        manifest.pattern.with_measured_reference(measured_reference_mm)
    )
    destination = output_path or manifest_path
    if destination.parent.resolve() != manifest_path.parent.resolve():
        png_source = manifest.resolve_png(manifest_path).resolve()
        pdf_value: str | None = None
        pdf_source = manifest.resolve_pdf(manifest_path)
        if pdf_source is not None:
            pdf_value = os.path.relpath(pdf_source.resolve(), destination.parent.resolve())
        updated = replace(
            updated,
            png_path=os.path.relpath(png_source, destination.parent.resolve()),
            pdf_path=pdf_value,
        )
    updated.save(destination)
    return updated

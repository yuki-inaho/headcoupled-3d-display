from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import cv2
import typer
from rich.console import Console
from rich.table import Table

from tagcal import __version__
from tagcal.board import AprilGridBoard
from tagcal.calibration import CameraCalibrator
from tagcal.capture import probe_cameras, record_video
from tagcal.cvtypes import as_uint8
from tagcal.detection import AprilTagDetector
from tagcal.models import (
    CalibrationResult,
    CalibrationSpec,
    CaptureSpec,
    PatternManifest,
    PatternSpec,
    SelectionReport,
    SelectionSpec,
)
from tagcal.pattern import PatternRenderer, confirm_display_scale
from tagcal.pipeline import CalibrationPipeline
from tagcal.screen import (
    ScreenLayout,
    pick_monitor,
    plan_layout,
    query_monitors,
    suggest_tag_sizes,
    write_manifest,
)
from tagcal.verify import grab_frames, load_intrinsics, verify_board

console = Console()
app = typer.Typer(
    name="tagcal",
    help="AprilTagグリッドを用いたカメラ内部キャリブレーションツールです。",
    no_args_is_help=True,
    invoke_without_command=True,
)
pattern_app = typer.Typer(help="AprilTagパターンを生成・管理します。", no_args_is_help=True)
app.add_typer(pattern_app, name="pattern")
screen_app = typer.Typer(help="ディスプレイへ実寸でパターンを表示します。", no_args_is_help=True)
app.add_typer(screen_app, name="screen")


class Family(StrEnum):
    tag16h5 = "tag16h5"
    tag25h9 = "tag25h9"
    tag36h10 = "tag36h10"
    tag36h11 = "tag36h11"


class Page(StrEnum):
    board = "board"
    a4 = "a4"
    letter = "letter"


class Format(StrEnum):
    png = "png"
    pdf = "pdf"
    both = "both"


def _progress(message: str) -> None:
    console.print(message)


def _input_fourcc(value: str) -> str | None:
    """Accept "none" on the command line to mean "leave the driver default alone"."""
    return None if value.strip().lower() == "none" else value


def _pipeline(
    *,
    sample_fps: float,
    target_frames: int,
    min_tags: int,
    min_coverage: float,
    min_sharpness: float,
    min_views: int,
    rational_model: bool,
    max_view_error: float,
) -> CalibrationPipeline:
    selection = SelectionSpec(
        sample_fps=sample_fps,
        target_frames=target_frames,
        min_keyframes=min(min_views, target_frames),
        min_detected_tags=min_tags,
        min_board_coverage=min_coverage,
        min_sharpness=min_sharpness,
    )
    calibration = CalibrationSpec(
        min_views=min_views,
        rational_model=rational_model,
        max_view_error_px=max_view_error,
    )
    return CalibrationPipeline(selection_spec=selection, calibration_spec=calibration)


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option("--version", help="バージョンを表示して終了します。"),
    ] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@pattern_app.command("generate")
def generate_pattern(
    output_dir: Annotated[Path, typer.Argument(help="生成物を保存するディレクトリ")],
    columns: Annotated[int, typer.Option("--columns", "-x", min=1)] = 6,
    rows: Annotated[int, typer.Option("--rows", "-y", min=1)] = 4,
    tag_size_mm: Annotated[float, typer.Option("--tag-size-mm", min=0.1)] = 35.0,
    gap_mm: Annotated[float, typer.Option("--gap-mm", min=0.0)] = 8.0,
    margin_mm: Annotated[float, typer.Option("--margin-mm", min=0.0)] = 10.0,
    family: Annotated[Family, typer.Option("--family")] = Family.tag36h11,
    first_id: Annotated[int, typer.Option("--first-id", min=0)] = 0,
    border_bits: Annotated[int, typer.Option("--border-bits", min=1, max=4)] = 1,
    reference_bar_mm: Annotated[
        float, typer.Option("--reference-bar-mm", min=1.0)
    ] = 100.0,
    dpi: Annotated[int, typer.Option("--dpi", min=72)] = 300,
    page: Annotated[Page, typer.Option("--page")] = Page.board,
    output_format: Annotated[Format, typer.Option("--format")] = Format.both,
) -> None:
    spec = PatternSpec(
        columns=columns,
        rows=rows,
        tag_size_mm=tag_size_mm,
        gap_mm=gap_mm,
        margin_mm=margin_mm,
        family=family.value,
        first_id=first_id,
        border_bits=border_bits,
        reference_bar_mm=reference_bar_mm,
    )
    artifacts = PatternRenderer().generate(
        output_dir,
        spec,
        dpi=dpi,
        page=page.value,
        output_format=output_format.value,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"PNG:      {artifacts.png_path}")
    if artifacts.pdf_path:
        console.print(f"PDF:      {artifacts.pdf_path}")
    console.print(
        "表示時は参照バーを実測し、`tagcal pattern confirm-scale`で実寸を反映してください。"
    )


@pattern_app.command("confirm-scale")
def confirm_scale(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    measured_reference_mm: Annotated[
        float,
        typer.Option("--measured-reference-mm", "-m", min=0.1),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="省略時は元のmanifestを更新します。"),
    ] = None,
) -> None:
    updated = confirm_display_scale(manifest, measured_reference_mm, output)
    destination = output or manifest
    console.print(f"Updated manifest: [bold]{destination}[/bold]")
    console.print(f"Display scale:     {updated.pattern.display_scale:.8f}")
    console.print(f"Effective tag:     {updated.pattern.effective_tag_size_mm:.4f} mm")
    console.print(f"Effective gap:     {updated.pattern.effective_gap_mm:.4f} mm")


@pattern_app.command("info")
def pattern_info(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    loaded = PatternManifest.load(manifest)
    spec = loaded.pattern
    table = Table(title="AprilTag pattern")
    table.add_column("項目")
    table.add_column("値", justify="right")
    rows = [
        ("Family", spec.family),
        ("Grid", f"{spec.columns} × {spec.rows}"),
        ("IDs", f"{spec.first_id}..{spec.first_id + spec.marker_count - 1}"),
        ("Border bits", str(spec.border_bits)),
        ("Nominal tag", f"{spec.tag_size_mm:.4f} mm"),
        ("Effective tag", f"{spec.effective_tag_size_mm:.4f} mm"),
        ("Effective gap", f"{spec.effective_gap_mm:.4f} mm"),
        ("Display scale", f"{spec.display_scale:.8f}"),
        ("Board size", f"{spec.effective_board_width_mm:.2f} × {spec.effective_board_height_mm:.2f} mm"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)


@screen_app.command("list")
def screen_list() -> None:
    monitors = query_monitors()
    table = Table(title="Connected monitors")
    table.add_column("Name")
    table.add_column("Resolution", justify="right")
    table.add_column("Physical", justify="right")
    table.add_column("px/mm", justify="right")
    table.add_column("PPI", justify="right")
    table.add_column("H/V差", justify="right")
    for monitor in monitors:
        table.add_row(
            monitor.name + (" *" if monitor.primary else ""),
            f"{monitor.width_px}×{monitor.height_px}",
            f"{monitor.width_mm:.0f}×{monitor.height_mm:.0f} mm",
            f"{monitor.px_per_mm:.4f}",
            f"{monitor.ppi:.1f}",
            f"{monitor.aspect_mismatch_percent:.2f}%",
        )
    console.print(table)
    console.print(
        "EDIDの物理サイズは丸められていることがあります。"
        "`tagcal screen show --ruler`で定規と照合し、ずれていれば`--cal`で補正してください。"
    )


def _screen_layout(
    columns: int,
    rows: int,
    tag_size_mm: float,
    gap_mm: float,
    margin_mm: float,
    family: Family,
    first_id: int,
    monitor_name: str | None,
    px_per_mm: float | None,
    cal: float,
    snap: bool,
) -> ScreenLayout:
    spec = PatternSpec(
        columns=columns,
        rows=rows,
        tag_size_mm=tag_size_mm,
        gap_mm=gap_mm,
        margin_mm=margin_mm,
        family=family.value,
        first_id=first_id,
    )
    monitor = None if px_per_mm is not None else pick_monitor(query_monitors(), monitor_name)
    return plan_layout(
        spec,
        monitor=monitor,
        px_per_mm=px_per_mm,
        calibration_factor=cal,
        snap=snap,
    )


@screen_app.command("show")
def screen_show(
    output_dir: Annotated[
        Path,
        typer.Argument(help="表示した実寸に一致するpattern.jsonの保存先"),
    ] = Path("artifacts/screen"),
    monitor: Annotated[str | None, typer.Option("--monitor", help="出力名。省略時はprimary")] = None,
    columns: Annotated[int, typer.Option("--columns", "-x", min=1)] = 4,
    rows: Annotated[int, typer.Option("--rows", "-y", min=1)] = 3,
    tag_size_mm: Annotated[float, typer.Option("--tag-size-mm", min=1.0)] = 40.0,
    gap_mm: Annotated[float, typer.Option("--gap-mm", min=0.0)] = 10.0,
    margin_mm: Annotated[float, typer.Option("--margin-mm", min=0.0)] = 10.0,
    family: Annotated[Family, typer.Option("--family")] = Family.tag36h11,
    first_id: Annotated[int, typer.Option("--first-id", min=0)] = 0,
    px_per_mm: Annotated[
        float | None,
        typer.Option("--px-per-mm", help="EDIDを使わず実測値を直接指定します。"),
    ] = None,
    cal: Annotated[
        float,
        typer.Option("--cal", min=0.01, help="px/mmへ掛ける校正係数（実測値/表示値）。"),
    ] = 1.0,
    snap: Annotated[
        bool,
        typer.Option("--snap/--no-snap", help="1セルを整数pxに丸めます（推奨）。"),
    ] = True,
    ruler: Annotated[bool, typer.Option("--ruler", help="mm目盛りを併記します。")] = False,
    fullscreen: Annotated[bool, typer.Option("--fullscreen")] = False,
    frameless: Annotated[bool, typer.Option("--frameless")] = False,
    on_top: Annotated[bool, typer.Option("--on-top")] = False,
    at: Annotated[
        str | None,
        typer.Option("--at", metavar="X,Y", help="対象モニタ左上からのウィンドウ位置"),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show/--no-show", help="--no-showは計算とpattern.json出力のみ"),
    ] = True,
) -> None:
    layout = _screen_layout(
        columns,
        rows,
        tag_size_mm,
        gap_mm,
        margin_mm,
        family,
        first_id,
        monitor,
        px_per_mm,
        cal,
        snap,
    )
    if layout.monitor is not None:
        console.print(f"monitor       : {layout.monitor.describe()}")
    for line in layout.describe():
        console.print(line)

    manifest = write_manifest(layout, output_dir)
    console.print(f"Manifest      : [bold]{output_dir / 'pattern.json'}[/bold]")
    console.print(
        f"検出側にはこのmanifestを渡してください（実効タグ {manifest.pattern.effective_tag_size_mm:.4f} mm）。"
    )
    if not show:
        return

    position: tuple[int, int] | None = None
    if at is not None:
        try:
            x_text, y_text = at.split(",")
            position = (int(x_text), int(y_text))
        except ValueError as exc:
            raise ValueError("--at は 'X,Y' 形式で指定してください（例 --at 100,100）") from exc

    try:
        from tagcal.screenview import show_board_window
    except ImportError as exc:
        raise RuntimeError(
            "GUI依存関係がありません。`uv sync --extra gui`を実行してください。"
        ) from exc
    console.print("[q]/[Esc] 終了   [i] 情報表示   [r] 定規")
    show_board_window(
        layout,
        show_ruler=ruler,
        fullscreen=fullscreen,
        frameless=frameless,
        on_top=on_top,
        position=position,
    )


@screen_app.command("suggest")
def screen_suggest(
    monitor: Annotated[str | None, typer.Option("--monitor")] = None,
    tag_size_mm: Annotated[float, typer.Option("--tag-size-mm", min=1.0)] = 40.0,
    family: Annotated[Family, typer.Option("--family")] = Family.tag36h11,
    px_per_mm: Annotated[float | None, typer.Option("--px-per-mm")] = None,
    cal: Annotated[float, typer.Option("--cal", min=0.01)] = 1.0,
) -> None:
    layout = _screen_layout(
        1, 1, tag_size_mm, 0.0, 10.0, family, 0, monitor, px_per_mm, cal, True
    )
    table = Table(title=f"セルが整数pxになるタグサイズ ({layout.px_per_mm:.4f} px/mm)")
    table.add_column("cell [px]", justify="right")
    table.add_column("tag [px]", justify="right")
    table.add_column("実サイズ [mm]", justify="right")
    table.add_column("要求との差 [mm]", justify="right")
    for cell_px, tag_px, actual_mm in suggest_tag_sizes(layout):
        table.add_row(
            str(cell_px),
            str(tag_px),
            f"{actual_mm:.4f}",
            f"{actual_mm - tag_size_mm:+.4f}",
        )
    console.print(table)
    console.print(
        "実サイズが切りの良い値にならないことは問題ありません。"
        "manifestに実サイズが記録されるため精度上の不利はありません。"
    )


@app.command("devices")
def devices(
    max_index: Annotated[int, typer.Option("--max-index", min=1, max=64)] = 10,
    input_fourcc: Annotated[
        str,
        typer.Option("--input-fourcc", help="要求する画素フォーマット。noneでドライバ既定"),
    ] = "MJPG",
) -> None:
    found = probe_cameras(max_index, None if input_fourcc.lower() == "none" else input_fourcc)
    if not found:
        console.print("利用可能なカメラを検出できませんでした。")
        raise typer.Exit(code=1)
    table = Table(title="Detected cameras")
    table.add_column("Index", justify="right")
    table.add_column("Resolution", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("Format")
    table.add_column("Backend")
    for device in found:
        table.add_row(
            str(device.index),
            f"{device.width}×{device.height}",
            f"{device.fps:.2f}" if device.fps else "unknown",
            device.fourcc,
            device.backend,
        )
    console.print(table)
    console.print(
        "FPSは要求フォーマットに依存します。YUYVは無圧縮ですが帯域上、"
        "高解像度では数fpsまで落ちます。"
    )


@app.command("record")
def record(
    output_video: Annotated[Path, typer.Argument(help="出力動画（.mp4または.avi）")],
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", exists=True, dir_okay=False, help="ライブ検出表示用"),
    ] = None,
    camera: Annotated[int, typer.Option("--camera", min=0)] = 0,
    width: Annotated[int, typer.Option("--width", min=1)] = 1920,
    height: Annotated[int, typer.Option("--height", min=1)] = 1080,
    fps: Annotated[float, typer.Option("--fps", min=1.0)] = 30.0,
    duration: Annotated[
        float | None,
        typer.Option("--duration", min=0.1, help="省略時はQ/Escまで継続します。"),
    ] = None,
    codec: Annotated[str, typer.Option("--codec", help="出力動画のFourCC。例: mp4v, MJPG")] = "mp4v",
    input_fourcc: Annotated[
        str,
        typer.Option("--input-fourcc", help="カメラへ要求する画素フォーマット。noneで既定"),
    ] = "MJPG",
    preview: Annotated[bool, typer.Option("--preview/--no-preview")] = True,
) -> None:
    detector = None
    if manifest is not None:
        loaded = PatternManifest.load(manifest)
        detector = AprilTagDetector(AprilGridBoard(loaded.pattern))
    result = record_video(
        output_video,
        CaptureSpec(
            camera_index=camera,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration,
            codec=codec,
            input_fourcc=_input_fourcc(input_fourcc),
            preview=preview,
        ),
        detector=detector,
        progress=_progress,
    )
    console.print(
        f"Saved {result.frames_written} frames, {result.duration_seconds:.2f}s: "
        f"[bold]{result.video_path}[/bold]"
    )


@app.command("process")
def process(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    sample_fps: Annotated[float, typer.Option("--sample-fps", min=0.1)] = 12.0,
    target_frames: Annotated[int, typer.Option("--target-frames", min=3)] = 24,
    min_tags: Annotated[int, typer.Option("--min-tags", min=1)] = 4,
    min_coverage: Annotated[float, typer.Option("--min-coverage", min=0.0, max=1.0)] = 0.025,
    min_sharpness: Annotated[float, typer.Option("--min-sharpness", min=0.0, help="絶対閾値。既定0＝近傍比較のみで判定")] = 0.0,
    min_views: Annotated[int, typer.Option("--min-views", min=3)] = 10,
    rational_model: Annotated[bool, typer.Option("--rational-model/--standard-model")] = False,
    max_view_error: Annotated[float, typer.Option("--max-view-error", min=0.1)] = 1.5,
) -> None:
    if min_views > target_frames:
        raise ValueError("--min-views cannot exceed --target-frames")
    pipeline = _pipeline(
        sample_fps=sample_fps,
        target_frames=target_frames,
        min_tags=min_tags,
        min_coverage=min_coverage,
        min_sharpness=min_sharpness,
        min_views=min_views,
        rational_model=rational_model,
        max_view_error=max_view_error,
    )
    artifacts = pipeline.process_video(video, manifest, output_dir, progress=_progress)
    _print_calibration_summary(artifacts.calibration.result, output_dir)


@app.command("run")
def run_pipeline(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    camera: Annotated[int, typer.Option("--camera", min=0)] = 0,
    width: Annotated[int, typer.Option("--width", min=1)] = 1920,
    height: Annotated[int, typer.Option("--height", min=1)] = 1080,
    fps: Annotated[float, typer.Option("--fps", min=1.0)] = 30.0,
    duration: Annotated[
        float | None,
        typer.Option("--duration", min=0.1, help="省略時はQ/Escまで録画します。"),
    ] = None,
    codec: Annotated[str, typer.Option("--codec")] = "mp4v",
    input_fourcc: Annotated[
        str,
        typer.Option("--input-fourcc", help="カメラへ要求する画素フォーマット。noneで既定"),
    ] = "MJPG",
    sample_fps: Annotated[float, typer.Option("--sample-fps", min=0.1)] = 12.0,
    target_frames: Annotated[int, typer.Option("--target-frames", min=3)] = 24,
    min_tags: Annotated[int, typer.Option("--min-tags", min=1)] = 4,
    min_coverage: Annotated[float, typer.Option("--min-coverage", min=0.0, max=1.0)] = 0.025,
    min_sharpness: Annotated[float, typer.Option("--min-sharpness", min=0.0, help="絶対閾値。既定0＝近傍比較のみで判定")] = 0.0,
    min_views: Annotated[int, typer.Option("--min-views", min=3)] = 10,
    rational_model: Annotated[bool, typer.Option("--rational-model/--standard-model")] = False,
    max_view_error: Annotated[float, typer.Option("--max-view-error", min=0.1)] = 1.5,
) -> None:
    if min_views > target_frames:
        raise ValueError("--min-views cannot exceed --target-frames")
    pipeline = _pipeline(
        sample_fps=sample_fps,
        target_frames=target_frames,
        min_tags=min_tags,
        min_coverage=min_coverage,
        min_sharpness=min_sharpness,
        min_views=min_views,
        rational_model=rational_model,
        max_view_error=max_view_error,
    )
    artifacts = pipeline.record_and_process(
        manifest,
        output_dir,
        CaptureSpec(
            camera_index=camera,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration,
            codec=codec,
            input_fourcc=_input_fourcc(input_fourcc),
            preview=True,
        ),
        progress=_progress,
    )
    _print_calibration_summary(artifacts.calibration.result, output_dir)


@app.command("calibrate")
def calibrate_selection(
    selection_report: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    min_views: Annotated[int, typer.Option("--min-views", min=3)] = 10,
    rational_model: Annotated[bool, typer.Option("--rational-model/--standard-model")] = False,
    max_view_error: Annotated[float, typer.Option("--max-view-error", min=0.1)] = 1.5,
) -> None:
    report = SelectionReport.load(selection_report)
    artifacts = CameraCalibrator(
        CalibrationSpec(
            min_views=min_views,
            rational_model=rational_model,
            max_view_error_px=max_view_error,
        )
    ).calibrate(
        report,
        output_dir,
        keyframe_root=selection_report.parent,
        progress=_progress,
    )
    _print_calibration_summary(artifacts.result, output_dir)


@app.command("verify")
def verify(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    calibration: Annotated[
        Path | None,
        typer.Option("--calibration", exists=True, dir_okay=False, help="calibration.json"),
    ] = None,
    camera: Annotated[int, typer.Option("--camera", min=0)] = 0,
    image: Annotated[
        Path | None,
        typer.Option("--image", exists=True, dir_okay=False, help="カメラの代わりに静止画を使う"),
    ] = None,
    frames: Annotated[int, typer.Option("--frames", min=1)] = 30,
    width: Annotated[int, typer.Option("--width", min=1)] = 1280,
    height: Annotated[int, typer.Option("--height", min=1)] = 720,
) -> None:
    """表示・印刷したボードを実測し、スケールが合っているか確認します。"""
    loaded = PatternManifest.load(manifest)
    camera_matrix = None
    distortion = None
    if calibration is not None:
        camera_matrix, distortion = load_intrinsics(calibration)
        console.print(
            f"fx={camera_matrix[0][0]:.2f} fy={camera_matrix[1][1]:.2f} "
            f"cx={camera_matrix[0][2]:.2f} cy={camera_matrix[1][2]:.2f}"
        )

    if image is not None:
        loaded_image = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded_image is None:
            raise FileNotFoundError(f"画像を読めません: {image}")
        captured = [as_uint8(loaded_image)]
    else:
        captured = grab_frames(
            CaptureSpec(camera_index=camera, width=width, height=height, preview=False),
            frames,
        )

    result = verify_board(
        loaded,
        captured,
        camera_matrix=camera_matrix,
        distortion=distortion,
    )
    for line in result.describe():
        console.print(line)
    if result.distance_mm_mean is None:
        console.print("--calibrationを渡すと距離を推定します。")
    else:
        console.print(
            "巻尺の実測値と比例した差がある場合は、タグ実寸かディスプレイのpx/mmを疑ってください。"
            "距離はレンズ主点から液晶面（ガラス表面ではない）までです。"
        )


@app.command("gui")
def gui() -> None:
    try:
        from tagcal.gui import run_gui
    except ImportError as exc:
        raise RuntimeError(
            "GUI依存関係がありません。`uv sync --extra gui`を実行してください。"
        ) from exc
    raise typer.Exit(code=run_gui())


@app.command("panel")
def panel() -> None:
    """パターン表示・カメラ映像・録画だけを備えた簡易操作パネルを開きます。"""
    try:
        from tagcal.panel import run_panel
    except ImportError as exc:
        raise RuntimeError(
            "GUI依存関係がありません。`uv sync --extra gui`を実行してください。"
        ) from exc
    raise typer.Exit(code=run_panel())


def _print_calibration_summary(calibration: CalibrationResult, output_dir: Path) -> None:
    table = Table(title="Calibration result")
    table.add_column("項目")
    table.add_column("値", justify="right")
    table.add_row("RMS reprojection error", f"{calibration.rms_reprojection_error_px:.6f} px")
    table.add_row("Median view error", f"{calibration.median_view_error_px:.6f} px")
    table.add_row("Maximum view error", f"{calibration.max_view_error_px:.6f} px")
    table.add_row("Views used", str(len(calibration.used_views)))
    table.add_row("Views excluded", str(len(calibration.excluded_views)))
    # Standard deviations, not the RMS, say whether a parameter is actually pinned
    # down: fewer views always fit better while estimating worse.
    deviations = calibration.intrinsic_standard_deviations
    entries = (
        ("fx", calibration.camera_matrix[0][0]),
        ("fy", calibration.camera_matrix[1][1]),
        ("cx", calibration.camera_matrix[0][2]),
        ("cy", calibration.camera_matrix[1][2]),
    )
    for position, (name, value) in enumerate(entries):
        sigma = deviations[position] if position < len(deviations) else float("nan")
        table.add_row(name, f"{value:.4f} ± {sigma:.4f}")
    console.print(table)
    console.print(f"Outputs: [bold]{output_dir}[/bold]")
    console.print(
        "±は各パラメータの標準偏差です。RMSが小さくても±が大きい場合は、"
        "ビュー数か姿勢の多様性が不足しています。"
    )


def main() -> None:
    try:
        app()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main()

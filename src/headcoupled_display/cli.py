"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
import uvicorn

from .api import create_app, default_hardware_profile_path
from .models import HardwareProfile, TrackingSource
from .profiles import load_tagcal_calibration, profile_with_resolved_matrix, summarize_profile
from .synthetic import run_synthetic_calibration

cli = typer.Typer(no_args_is_help=True, add_completion=False)


@cli.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    profile: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    user_profile: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    source: Annotated[TrackingSource, typer.Option()] = "synthetic",
    camera_device: Annotated[
        str, typer.Option(help="V4L2 device path (use /dev/video0 on this machine)")
    ] = "/dev/video0",
    backend: Annotated[Literal["cpu", "cuda", "tensorrt"], typer.Option()] = "cpu",
    replay_landmarks: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="FaceMesh --save-json output")
    ] = None,
    replay_video: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Video used for that JSON")
    ] = None,
    face_model: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Personal 478-point shape.pcd")
    ] = None,
    intrinsics: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="tagcal calibration.json/YAML")
    ] = None,
) -> None:
    """Serve the dashboard, with live FaceMesh or a recorded FaceMesh replay."""

    if source == "replay" and any(
        value is None for value in (replay_landmarks, replay_video, face_model, intrinsics)
    ):
        raise typer.BadParameter(
            "replay requires --replay-landmarks, --replay-video, --face-model, and --intrinsics"
        )

    application = create_app(
        profile_path=profile or default_hardware_profile_path(),
        user_profile_path=user_profile,
        source=source,
        camera_device=camera_device,
        backend=backend,
        replay_landmarks_path=replay_landmarks,
        replay_video_path=replay_video,
        face_model_path=face_model,
        intrinsics_path=intrinsics,
    )
    uvicorn.run(application, host=host, port=port, log_level="info")


@cli.command("profile-summary")
def profile_summary(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate a profile and print the mount geometry derived from its transform."""

    profile = profile_with_resolved_matrix(HardwareProfile.load(path))
    typer.echo(
        json.dumps(
            {
                "profile_id": profile.profile_id,
                "provenance": profile.provenance,
                "mount_summary": summarize_profile(profile).model_dump(mode="json"),
                "camera_to_display_matrix": profile.camera_to_display_matrix,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@cli.command("import-tagcal")
def import_tagcal(
    calibration: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    profile: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "config/hardware_profile.measured.json"
    ),
) -> None:
    """Merge tagcal camera intrinsics into an existing hardware profile."""

    base = HardwareProfile.load(profile or default_hardware_profile_path())
    intrinsics = load_tagcal_calibration(calibration)
    merged = base.model_copy(
        update={
            "profile_id": f"{base.profile_id}-tagcal",
            "camera": intrinsics,
            "quality_metrics": {
                **base.quality_metrics,
                "tagcal_source": str(calibration),
                "camera_intrinsics_imported": True,
            },
            "notes": (
                *base.notes,
                "Camera intrinsics were imported from tagcal; camera-display extrinsics remain independent.",
            ),
        }
    )
    merged.save(output)
    typer.echo(f"wrote: {output}")


@cli.command("synthetic-calibrate")
def synthetic_calibrate(
    profile: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/synthetic_calibration_result.json"
    ),
    dataset_output: Annotated[Path | None, typer.Option(dir_okay=False)] = Path(
        "artifacts/synthetic_calibration_dataset.json"
    ),
    seed: Annotated[int, typer.Option()] = 20260817,
) -> None:
    """Generate deterministic head rays and recover the display-camera transform."""

    ground_truth = HardwareProfile.load(profile or default_hardware_profile_path())
    dataset, result = run_synthetic_calibration(ground_truth, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if dataset_output is not None:
        dataset.save(dataset_output)
    typer.echo(f"result:  {output}")
    if dataset_output is not None:
        typer.echo(f"dataset: {dataset_output}")
    typer.echo(json.dumps(result.comparison_to_ground_truth, indent=2))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

"""FastAPI control plane, WebSocket streams, and static dashboard."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .calibration import fit_display_transform
from .face_model import load_personal_face_model
from .models import CalibrationDataset, HardwareProfile, UserProfile
from .profiles import load_user_profile, profile_with_resolved_matrix, summarize_profile
from .runtime import RuntimeCoordinator
from .synthetic import SyntheticTrackingProvider, run_synthetic_calibration
from .tracking import FaceMeshTrackingProvider, TrackingProvider

PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "static"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of these paths exists: {paths}")


def default_hardware_profile_path() -> Path:
    configured = os.getenv("HEADCOUPLED_PROFILE")
    candidates = [] if not configured else [Path(configured).expanduser()]
    candidates.extend(
        [
            Path.cwd() / "config" / "hardware_profile.demo.json",
            RESOURCE_ROOT / "hardware_profile.demo.json",
        ]
    )
    return _first_existing(candidates)


def default_user_profile_path() -> Path:
    configured = os.getenv("HEADCOUPLED_USER_PROFILE")
    candidates = [] if not configured else [Path(configured).expanduser()]
    candidates.extend(
        [
            Path.cwd() / "config" / "user_profile.demo.json",
            RESOURCE_ROOT / "user_profile.demo.json",
        ]
    )
    return _first_existing(candidates)


def _provider_factory(
    source: Literal["synthetic", "facemesh"],
    hardware: HardwareProfile,
    user: UserProfile,
    *,
    camera_index: int,
    backend: str,
) -> Callable[[], TrackingProvider]:
    if source == "synthetic":
        return lambda: SyntheticTrackingProvider(hardware)
    return lambda: FaceMeshTrackingProvider(
        hardware,
        user,
        camera_index=camera_index,
        backend=backend,
        width=hardware.camera.image_width_px,
        height=hardware.camera.image_height_px,
    )


def create_app(
    *,
    profile_path: Path | None = None,
    user_profile_path: Path | None = None,
    source: Literal["synthetic", "facemesh"] | None = None,
    camera_index: int = 0,
    backend: str = "cpu",
) -> FastAPI:
    hardware = profile_with_resolved_matrix(
        HardwareProfile.load(profile_path or default_hardware_profile_path())
    )
    user = load_user_profile(user_profile_path or default_user_profile_path())
    # Validate a configured mesh even in synthetic mode, so a bad profile is caught
    # before a real camera is attached.
    if user.face_model_path is not None:
        load_personal_face_model(Path(user.face_model_path))
    selected_source: Literal["synthetic", "facemesh"] = source or os.getenv(
        "HEADCOUPLED_SOURCE", "synthetic"
    )  # type: ignore[assignment]
    if selected_source not in {"synthetic", "facemesh"}:
        raise ValueError(f"unsupported tracking source: {selected_source}")

    runtime = RuntimeCoordinator(
        hardware,
        _provider_factory(
            selected_source,
            hardware,
            user,
            camera_index=camera_index,
            backend=backend,
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.hardware_profile = hardware
        app.state.user_profile = user
        app.state.runtime = runtime
        app.state.source = selected_source
        app.state.last_calibration = None
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    application = FastAPI(
        title="Head-Coupled 3D Display",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @application.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "runtime": runtime.status(selected_source).model_dump(),
        }

    @application.get("/api/profile")
    async def profile() -> dict[str, object]:
        summary = summarize_profile(hardware)
        return {
            "hardware_profile": hardware.model_dump(mode="json"),
            "user_profile": user.model_dump(mode="json"),
            "mount_summary": summary.model_dump(mode="json"),
            "coordinate_convention": {
                "display": "origin=center, +X=right, +Y=up, +Z=toward viewer",
                "camera": "OpenCV: +X=image right, +Y=image down, +Z=optical forward",
                "transform": "p_display = T_display_camera * p_camera",
            },
            "warning": (
                "同梱プロファイルは実測値ではなく、動作確認用の人工値です。"
                if hardware.provenance == "synthetic_demo_not_measured"
                else None
            ),
        }

    @application.get("/api/runtime")
    async def runtime_status() -> dict[str, object]:
        return runtime.status(selected_source).model_dump(mode="json")

    @application.get("/api/frame.jpg", responses={503: {"description": "No frame yet"}})
    async def latest_frame() -> Response:
        frame = runtime.latest_frame
        if frame is None:
            raise HTTPException(status_code=503, detail="camera frame is not ready")
        return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @application.post("/api/calibration/synthetic")
    async def synthetic_calibration(seed: int = 20260817) -> dict[str, object]:
        dataset, result = await asyncio.to_thread(run_synthetic_calibration, hardware, seed=seed)
        application.state.last_calibration = result
        return {
            "status": "success" if result.metrics.optimizer_success else "failed",
            "dataset": {
                "sample_count": len(dataset.samples),
                "unique_target_count": len({sample.target_uv for sample in dataset.samples}),
                "metadata": dataset.metadata,
            },
            "result": result.model_dump(mode="json"),
        }

    @application.post("/api/calibration/fit")
    async def calibration_fit(dataset: CalibrationDataset) -> dict[str, object]:
        result = await asyncio.to_thread(fit_display_transform, dataset, hardware)
        application.state.last_calibration = result
        return {"status": "success", "result": result.model_dump(mode="json")}

    @application.get("/api/calibration/status")
    async def calibration_status() -> dict[str, object]:
        result = application.state.last_calibration
        return {
            "available": result is not None,
            "result": None if result is None else result.model_dump(mode="json"),
        }

    @application.websocket("/ws/pose")
    async def pose_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        sequence = -1
        try:
            while True:
                try:
                    state = await runtime.wait_for_state(sequence)
                except TimeoutError:
                    await websocket.send_json({"type": "heartbeat", "sequence": sequence})
                    continue
                sequence = state.sequence
                await websocket.send_json({"type": "tracking", "payload": state.model_dump(mode="json")})
        except WebSocketDisconnect:
            return

    @application.websocket("/ws/camera")
    async def camera_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        sequence = -1
        try:
            while True:
                try:
                    sequence, frame = await runtime.wait_for_frame(sequence)
                except TimeoutError:
                    continue
                await websocket.send_bytes(frame)
        except WebSocketDisconnect:
            return

    return application


app = create_app()

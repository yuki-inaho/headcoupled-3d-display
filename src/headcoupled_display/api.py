"""FastAPI control plane, WebSocket streams, and static dashboard."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .calibration import fit_display_transform
from .face_model import load_personal_face_model
from .models import (
    CalibrationDataset,
    HardwareProfile,
    SceneProfile,
    TrackingSource,
    UserProfile,
)
from .profiles import (
    load_tagcal_calibration,
    load_user_profile,
    profile_with_resolved_matrix,
    summarize_profile,
)
from .runtime import RuntimeCoordinator
from .synthetic import SyntheticTrackingProvider, run_synthetic_calibration
from .tracking import (
    FaceMeshIpcProvider,
    FaceMeshReplayProvider,
    FaceMeshTrackingProvider,
    IpcFaceMeshInput,
    TrackingProvider,
)

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


def default_scene_profile_path() -> Path:
    configured = os.getenv("HEADCOUPLED_SCENE")
    candidates = [] if not configured else [Path(configured).expanduser()]
    candidates.extend(
        [
            Path.cwd() / "config" / "scene_profile.default.json",
            RESOURCE_ROOT / "scene_profile.default.json",
        ]
    )
    return _first_existing(candidates)


def _provider_factory(
    source: TrackingSource,
    hardware: HardwareProfile,
    user: UserProfile,
    *,
    camera_device: str,
    backend: str,
    replay_landmarks_path: Path | None,
    replay_video_path: Path | None,
    ipc_input: IpcFaceMeshInput | None,
) -> Callable[[], TrackingProvider]:
    if source == "synthetic":
        return lambda: SyntheticTrackingProvider(hardware)
    if source == "replay":
        if replay_landmarks_path is None or replay_video_path is None:
            raise ValueError("replay source requires both landmarks JSON and video paths")
        return lambda: FaceMeshReplayProvider(
            hardware,
            user,
            landmarks_path=replay_landmarks_path,
            video_path=replay_video_path,
        )
    if source == "ipc":
        if ipc_input is None:  # pragma: no cover - guarded while the context is constructed
            raise ValueError("IPC source requires an IPC input port")
        return lambda: FaceMeshIpcProvider(hardware, user, frame_source=ipc_input)
    return lambda: FaceMeshTrackingProvider(
        hardware,
        user,
        camera_device=camera_device,
        backend=backend,
        width=hardware.camera.image_width_px,
        height=hardware.camera.image_height_px,
    )


@dataclass(frozen=True)
class _ApplicationContext:
    hardware: HardwareProfile
    user: UserProfile
    scene: SceneProfile
    runtime: RuntimeCoordinator
    source: TrackingSource
    ipc_input: IpcFaceMeshInput | None


def _select_source(source: TrackingSource | None) -> TrackingSource:
    selected = source or os.getenv("HEADCOUPLED_SOURCE", "synthetic")
    if selected not in {"synthetic", "facemesh", "replay", "ipc"}:
        raise ValueError(f"unsupported tracking source: {selected}")
    return cast(TrackingSource, selected)


def _with_runtime_intrinsics(
    hardware: HardwareProfile, calibration_path: Path | None
) -> HardwareProfile:
    if calibration_path is None:
        return hardware
    intrinsics = load_tagcal_calibration(calibration_path)
    return hardware.model_copy(
        update={
            "profile_id": f"{hardware.profile_id}-tagcal",
            "camera": intrinsics,
            "quality_metrics": {
                **hardware.quality_metrics,
                "tagcal_source": str(calibration_path.resolve()),
                "camera_intrinsics_imported": True,
            },
            "notes": (
                *hardware.notes,
                "Camera intrinsics were imported from tagcal; camera-display extrinsics remain independent.",
            ),
        }
    )


def _profile_warning(hardware: HardwareProfile) -> str | None:
    if hardware.quality_metrics.get("camera_intrinsics_imported") is True:
        return (
            "カメラ内部パラメータはtagcalの実測値です。カメラ・ディスプレイ外部姿勢は"
            "このプロファイルの値であり、別途実測または頭部レイ較正が必要です。"
        )
    if hardware.provenance == "synthetic_demo_not_measured":
        return "同梱プロファイルは実測値ではなく、動作確認用の人工値です。"
    return None


def _build_context(
    *,
    profile_path: Path | None = None,
    user_profile_path: Path | None = None,
    scene_path: Path | None = None,
    source: TrackingSource | None = None,
    camera_device: str = "/dev/video0",
    backend: str = "cpu",
    replay_landmarks_path: Path | None = None,
    replay_video_path: Path | None = None,
    face_model_path: Path | None = None,
    intrinsics_path: Path | None = None,
) -> _ApplicationContext:
    hardware = _with_runtime_intrinsics(
        profile_with_resolved_matrix(
            HardwareProfile.load(profile_path or default_hardware_profile_path())
        ),
        intrinsics_path,
    )
    scene = SceneProfile.load(scene_path or default_scene_profile_path())
    user = load_user_profile(user_profile_path or default_user_profile_path())
    if face_model_path is not None:
        user = user.model_copy(
            update={"face_model_path": str(face_model_path.expanduser().resolve())}
        )
    # Validate a configured mesh even in synthetic mode, so a bad profile is caught
    # before a real camera is attached.
    if user.face_model_path is not None:
        load_personal_face_model(Path(user.face_model_path))
    selected_source = _select_source(source)
    ipc_input = IpcFaceMeshInput(hardware) if selected_source == "ipc" else None
    runtime = RuntimeCoordinator(
        hardware,
        _provider_factory(
            selected_source,
            hardware,
            user,
            camera_device=camera_device,
            backend=backend,
            replay_landmarks_path=replay_landmarks_path,
            replay_video_path=replay_video_path,
            ipc_input=ipc_input,
        ),
    )
    return _ApplicationContext(
        hardware=hardware,
        user=user,
        scene=scene,
        runtime=runtime,
        source=selected_source,
        ipc_input=ipc_input,
    )


def _lifespan(context: _ApplicationContext) -> Callable[[FastAPI], AsyncIterator[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.hardware_profile = context.hardware
        app.state.user_profile = context.user
        app.state.runtime = context.runtime
        app.state.source = context.source
        app.state.last_calibration = None
        await context.runtime.start()
        try:
            yield
        finally:
            await context.runtime.stop()

    return lifespan


def _register_http_routes(application: FastAPI, context: _ApplicationContext) -> None:
    hardware, user, scene, runtime, selected_source = (
        context.hardware,
        context.user,
        context.scene,
        context.runtime,
        context.source,
    )

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @application.get("/api/health")
    async def health() -> dict[str, object]:
        # server_unix_ns lets a client measure the offset between its own Unix clock and
        # this server's, NTP-style, by bracketing the request with two local readings.
        # Success condition 10 compares a producer timestamp against a browser timestamp;
        # without this, the two clocks are assumed identical instead of being checked.
        return {
            "status": "ok",
            "server_unix_ns": time.time_ns(),
            "runtime": runtime.status(selected_source).model_dump(),
        }

    @application.get("/api/profile")
    async def profile() -> dict[str, object]:
        summary = summarize_profile(hardware)
        return {
            "hardware_profile": hardware.model_dump(mode="json"),
            "user_profile": user.model_dump(mode="json"),
            "scene_profile": scene.model_dump(mode="json"),
            "mount_summary": summary.model_dump(mode="json"),
            "coordinate_convention": {
                "display": "origin=center, +X=right, +Y=up, +Z=toward viewer",
                "camera": "OpenCV: +X=image right, +Y=image down, +Z=optical forward",
                "transform": "p_display = T_display_camera * p_camera",
            },
            "warning": _profile_warning(hardware),
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


def _register_ipc_input_routes(application: FastAPI, ipc_input: IpcFaceMeshInput | None) -> None:
    """Register the two independent IPC lanes (workdoc steps 36-38): control (a fixed-
    size binary pose packet, ``application/octet-stream``, see ``protocol.py``) and
    preview (an already-compressed JPEG, ``image/jpeg``). These are deliberately
    separate endpoints and payload formats rather than one JSON envelope, so a broken
    or throttled preview can never block or delay a control update -- see
    ``IpcFaceMeshInput``'s own docstring.
    """

    def _require_ipc_source() -> IpcFaceMeshInput:
        if ipc_input is None:
            raise HTTPException(status_code=409, detail="server was not started with --source ipc")
        return ipc_input

    @application.post("/api/input/facemesh/control")
    async def publish_facemesh_control(request: Request) -> dict[str, int]:
        """Accept one raw, fixed-length control packet (see ``protocol.PACKET_SIZE``)."""

        port = _require_ipc_source()
        body = await request.body()
        try:
            sequence = port.publish_control(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"accepted_sequence": sequence}

    @application.post("/api/input/facemesh/preview", status_code=204)
    async def publish_facemesh_preview(request: Request) -> Response:
        """Accept one raw, already-compressed JPEG preview; forwarded, never decoded."""

        port = _require_ipc_source()
        body = await request.body()
        try:
            port.publish_preview(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(status_code=204)


def _register_clock_route(application: FastAPI) -> None:
    @application.websocket("/ws/clock")
    async def clock_websocket(websocket: WebSocket) -> None:
        """Echo this server's Unix clock for each message, so a client can measure offset.

        Deliberately a WebSocket rather than an HTTP endpoint. Success condition 10
        subtracts a producer Unix timestamp from a browser one, and the bound on that
        difference is half the round trip used to compare the two clocks. An HTTP fetch
        from the page carries request parsing and response handling that put that bound
        several milliseconds above the 2 ms the condition allows; an echo on an
        already-open socket does not.
        """

        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
                await websocket.send_json({"server_unix_ns": time.time_ns()})
        except WebSocketDisconnect:
            return


def _register_websocket_routes(application: FastAPI, runtime: RuntimeCoordinator) -> None:

    @application.websocket("/ws/pose")
    async def pose_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        # Subscribers follow the coordinator's generation, not TrackingState.sequence:
        # a "tracking stalled" notice has to reach clients even though the provider
        # produced no new frame to number it with.
        generation = 0
        try:
            while True:
                try:
                    generation, state = await runtime.wait_for_state(generation)
                except TimeoutError:
                    await websocket.send_json({"type": "heartbeat", "sequence": generation})
                    continue
                await websocket.send_json(
                    {"type": "tracking", "payload": state.model_dump(mode="json")}
                )
        except WebSocketDisconnect:
            return

    @application.websocket("/ws/camera")
    async def camera_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        generation = 0
        try:
            while True:
                try:
                    generation, frame = await runtime.wait_for_frame(generation)
                except TimeoutError:
                    continue
                await websocket.send_bytes(frame)
        except WebSocketDisconnect:
            return


def create_app(
    *,
    profile_path: Path | None = None,
    user_profile_path: Path | None = None,
    scene_path: Path | None = None,
    source: TrackingSource | None = None,
    camera_device: str = "/dev/video0",
    backend: str = "cpu",
    replay_landmarks_path: Path | None = None,
    replay_video_path: Path | None = None,
    face_model_path: Path | None = None,
    intrinsics_path: Path | None = None,
) -> FastAPI:
    context = _build_context(
        profile_path=profile_path,
        user_profile_path=user_profile_path,
        scene_path=scene_path,
        source=source,
        camera_device=camera_device,
        backend=backend,
        replay_landmarks_path=replay_landmarks_path,
        replay_video_path=replay_video_path,
        face_model_path=face_model_path,
        intrinsics_path=intrinsics_path,
    )
    application = FastAPI(
        title="Head-Coupled 3D Display",
        version="0.1.0",
        lifespan=_lifespan(context),
    )
    application.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    _register_http_routes(application, context)
    _register_ipc_input_routes(application, context.ipc_input)
    _register_clock_route(application)
    _register_websocket_routes(application, context.runtime)

    return application


app = create_app()

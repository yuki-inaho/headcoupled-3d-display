"""Publish live FaceMesh observations to a local head-coupled display server.

Run this script with the ``facemesh_tracking`` Python environment.  It deliberately uses
only Python's standard-library HTTP client for transport, so the producer does not import
the Python 3.13 display package or inherit its dependency constraints.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import cv2


@dataclass
class IpcPublisher:
    """Persistent HTTP publisher with one reconnect attempt per complete frame."""

    endpoint: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "http" or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http:// URL")
        self._host = parsed.netloc
        self._path = parsed.path or "/"
        if parsed.query:
            self._path = f"{self._path}?{parsed.query}"
        self._connection: http.client.HTTPConnection | None = None

    def publish(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for attempt in range(2):
            try:
                connection = self._connection or http.client.HTTPConnection(self._host, timeout=3.0)
                self._connection = connection
                connection.request(
                    "POST",
                    self._path,
                    body=body,
                    headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
                )
                response = connection.getresponse()
                response_body = response.read().decode("utf-8", errors="replace")
                if 200 <= response.status < 300:
                    return
                raise RuntimeError(f"IPC server returned HTTP {response.status}: {response_body}")
            except OSError:
                self.close()
                if attempt:
                    raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Split out so tests can inspect it without invoking main()."""
    parser = argparse.ArgumentParser(description="Publish live FaceMesh frames to headcoupled IPC")
    parser.add_argument("--camera", default="/dev/video0", help="V4L2 path or numeric camera index")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/input/facemesh")
    parser.add_argument(
        "--backend",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "Execution backend. TensorRT is a non-goal here (R-PERF-2): "
            "'libnvinfer.so.10' is absent, so it would silently fall back to CUDA "
            "instead of failing, which is exactly the kind of fallback this producer "
            "must not hide. Use --backend cuda (the default) for production runs; "
            "--backend cpu is only for debugging without a GPU and skips the CUDA "
            "provider check below."
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--jpeg-quality", type=int, default=82, choices=range(1, 101))
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = until Ctrl-C)"
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _actual_providers(component: Any, label: str) -> list[str]:
    """Read the *actual* ONNX Runtime providers active for one pipeline stage.

    ``component.providers`` (set by facemesh_tracking's UniFace wrappers) is only the
    provider list ONNX Runtime was *asked* to try. onnxruntime silently keeps walking that
    list (e.g. down to CPUExecutionProvider) when an earlier entry fails to load, so the
    requested list is not evidence that CUDA is actually running.

    facemesh_tracking has no public accessor for the live session, so this reaches through
    UniFace's own ``_model.session`` attribute layout instead (UniFace's ``session`` is
    itself public; only the wrapper's ``_model`` is internal). If that internal layout
    ever changes, this raises rather than silently trusting the requested provider list as
    a stand-in for reality.
    """
    session = getattr(getattr(component, "_model", None), "session", None)
    get_providers = getattr(session, "get_providers", None)
    if get_providers is None:
        raise RuntimeError(
            f"{label}: cannot reach an ONNX Runtime session via '_model.session' on "
            f"{type(component).__name__!r}; refusing to trust the requested provider list "
            "as evidence of the actual execution provider"
        )
    return list(get_providers())


def assert_cuda_providers(pipeline: Any) -> dict[str, list[str]]:
    """Fail startup unless both the detector and estimator sessions actually run on CUDA.

    ``--backend cuda`` only requests CUDAExecutionProvider; onnxruntime falls back to
    CPUExecutionProvider without raising when CUDA cannot load (see
    ``facemesh_tracking.runtime.providers_for``). Treating that fallback as success would
    silently violate R-PERF-2 / TR-4, so this inspects the real session providers instead
    of the requested list and names every stage that did not lead with CUDA.

    Returns the actual provider list per stage so the caller can log it once.
    """
    actual: dict[str, list[str]] = {}
    failed: list[str] = []
    for label, component in (("detector", pipeline.detector), ("estimator", pipeline.estimator)):
        providers = _actual_providers(component, label)
        actual[label] = providers
        if not providers or providers[0] != "CUDAExecutionProvider":
            failed.append(f"{label} (actual={providers!r})")
    if failed:
        raise RuntimeError(
            "CUDA execution provider is not active for: "
            + ", ".join(failed)
            + " - CPU fallback is not treated as success (R-PERF-2)"
        )
    return actual


def main() -> None:
    args = parse_args()
    from facemesh_tracking.pipeline import FaceMeshPipeline
    from facemesh_tracking.runtime import Backend

    pipeline = FaceMeshPipeline.create(backend=Backend(args.backend))
    if args.backend == "cuda":
        # Print the actual (not merely requested) providers exactly once, so a CPU
        # fallback is both fatal and visible in the log leading up to the failure.
        print(f"providers (actual): {assert_cuda_providers(pipeline)}")
    else:
        print(f"providers (requested, backend={args.backend}): {pipeline.detector.providers}")
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera {args.camera!r}")
    publisher = IpcPublisher(args.endpoint)
    print(f"camera: {args.camera} -> {args.endpoint}")
    frame_index = 0
    started = time.perf_counter()
    try:
        while not args.max_frames or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            result = pipeline.process(frame)
            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not encoded_ok:
                raise RuntimeError("failed to encode camera frame")
            publisher.publish(
                {
                    "frame_index": frame_index,
                    "faces": [
                        {"score": float(face.score), "landmarks": face.points.tolist()}
                        for face in result.faces
                    ],
                    "frame_jpeg_base64": base64.b64encode(encoded).decode("ascii"),
                }
            )
            frame_index += 1
            if frame_index % 30 == 0:
                fps = frame_index / max(time.perf_counter() - started, 1e-6)
                print(f"published={frame_index}  faces={len(result.faces)}  {fps:.1f} FPS")
    finally:
        capture.release()
        publisher.close()


if __name__ == "__main__":
    main()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish live FaceMesh frames to headcoupled IPC")
    parser.add_argument("--camera", default="/dev/video0", help="V4L2 path or numeric camera index")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/input/facemesh")
    parser.add_argument("--backend", choices=("cuda", "cpu", "tensorrt"), default="cuda")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--jpeg-quality", type=int, default=82, choices=range(1, 101))
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = until Ctrl-C)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from facemesh_tracking.pipeline import FaceMeshPipeline
    from facemesh_tracking.runtime import Backend

    pipeline = FaceMeshPipeline.create(backend=Backend(args.backend))
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera {args.camera!r}")
    publisher = IpcPublisher(args.endpoint)
    print(f"providers: {pipeline.detector.providers}")
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

"""Command line entry point: `facemesh {run,bench}`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .media import open_source, open_writer
from .pipeline import FaceMeshPipeline, FaceMeshResult
from .runtime import Backend
from .uniface_models import DEFAULT_MODELS_DIR
from .visualize import DrawingMode, render

WINDOW_NAME = "FaceMesh"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Where the ONNX weights are cached (fetched on first use)",
    )
    parser.add_argument(
        "--backend",
        type=Backend,
        choices=list(Backend),
        default=Backend.CUDA,
        help="Execution provider (default: cuda)",
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--detection-threshold", type=float, default=0.5, help="Minimum detector score"
    )
    parser.add_argument(
        "--landmark-threshold", type=float, default=0.5, help="Minimum face-presence probability"
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.0,
        help="Extra box expansion before the mesh stage; the mesh already expands by 25%%",
    )


def _build_pipeline(args: argparse.Namespace) -> FaceMeshPipeline:
    return FaceMeshPipeline.create(
        backend=args.backend,
        models_dir=args.models_dir,
        detection_threshold=args.detection_threshold,
        landmark_threshold=args.landmark_threshold,
        margin_ratio=args.margin_ratio,
        device_id=args.device_id,
    )


def _landmarks_to_json(frame_index: int, result: FaceMeshResult) -> dict:
    return {
        "frame": frame_index,
        "faces": [
            {
                "score": face.score,
                "bbox": [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2],
                "landmarks": face.points.tolist(),
            }
            for face in result.faces
        ],
    }


def cmd_run(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline(args)
    print(f"providers: {pipeline.detector.providers}")

    mode = DrawingMode(args.mode)
    records: list[dict] = []
    elapsed_ms: list[float] = []

    with open_source(args.source, width=args.width, height=args.height) as source:
        writer_ctx = (
            open_writer(args.output, source.info)
            if args.output and not source.info.is_image
            else None
        )
        writer = writer_ctx.__enter__() if writer_ctx else None
        try:
            for index, frame in enumerate(source):
                if args.max_frames and index >= args.max_frames:
                    break
                started = time.perf_counter()
                result = pipeline.process(frame)
                elapsed_ms.append((time.perf_counter() - started) * 1e3)

                canvas = render(
                    frame,
                    result.faces,
                    result.boxes,
                    mode=mode,
                    show_background=not args.no_background,
                    show_boxes=not args.no_boxes,
                )
                if args.save_json:
                    records.append(_landmarks_to_json(index, result))
                if writer is not None:
                    writer.write(canvas)
                if args.output and source.info.is_image:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.output), canvas)
                if args.show:
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(0 if source.info.is_image else 1) == 27:  # ESC
                        break
        finally:
            if writer_ctx:
                writer_ctx.__exit__(None, None, None)
            if args.show:
                cv2.destroyAllWindows()

    if elapsed_ms:
        times = np.asarray(elapsed_ms)
        print(
            f"frames={times.size}  mean={times.mean():.1f} ms  "
            f"median={np.median(times):.1f} ms  ({1e3 / np.median(times):.1f} FPS)"
        )
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(records))
        print(f"landmarks -> {args.save_json}")
    if args.output:
        print(f"rendered  -> {args.output}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline(args)
    if args.source:
        frame = cv2.imread(args.source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.source}")
    else:
        # Synthetic noise contains no face, so only the detector stage is exercised.
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        pipeline.process(frame)

    height, width = frame.shape[:2]
    total, detection, landmarks = [], [], []
    faces: list = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        boxes = [
            b.expanded(pipeline.margin_ratio, width, height)
            for b in pipeline.detector.detect(frame)
        ]
        detected_at = time.perf_counter()
        faces = pipeline.estimator.estimate(frame, boxes)
        finished = time.perf_counter()
        total.append((finished - started) * 1e3)
        detection.append((detected_at - started) * 1e3)
        landmarks.append((finished - detected_at) * 1e3)

    times = np.asarray(total)
    print(f"backend  : {args.backend.value}")
    print(f"providers: {pipeline.detector.providers}")
    print(f"frame    : {width}x{height}, faces={len(faces)}")
    print(
        f"total    : mean={times.mean():.2f} ms  median={np.median(times):.2f} ms  "
        f"min={times.min():.2f} ms  max={times.max():.2f} ms  ({1e3 / times.mean():.1f} FPS)"
    )
    print(f"detection: mean={np.mean(detection):.2f} ms")
    print(f"facemesh : mean={np.mean(landmarks):.2f} ms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facemesh",
        description="FaceMesh 478-landmark estimation (YOLOv8-Face + MediaPipe) on onnxruntime-gpu",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run on an image, a video file or a camera")
    run.add_argument("--source", required=True, help="Image path, video path, or camera index")
    run.add_argument("--output", type=Path, default=None, help="Rendered image/video output path")
    run.add_argument("--save-json", type=Path, default=None, help="Dump landmarks as JSON")
    run.add_argument(
        "--mode", choices=[m.value for m in DrawingMode], default=DrawingMode.FULL.value
    )
    run.add_argument("--show", action="store_true", help="Display the result in a window")
    run.add_argument(
        "--no-background", action="store_true", help="Draw on black instead of the frame"
    )
    run.add_argument("--no-boxes", action="store_true", help="Hide detection boxes")
    run.add_argument("--width", type=int, default=None, help="Requested capture width")
    run.add_argument("--height", type=int, default=None, help="Requested capture height")
    run.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    _add_common_arguments(run)
    run.set_defaults(func=cmd_run)

    bench = subparsers.add_parser("bench", help="Measure per-stage latency on a single frame")
    bench.add_argument("--source", default=None, help="Image path (default: synthetic noise)")
    bench.add_argument("--iterations", type=int, default=50)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--width", type=int, default=640, help="Synthetic frame width")
    bench.add_argument("--height", type=int, default=480, help="Synthetic frame height")
    _add_common_arguments(bench)
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""FaceMesh dense landmark estimation on onnxruntime-gpu.

Typical use::

    from facemesh_tracking import Backend, FaceMeshPipeline, render

    pipeline = FaceMeshPipeline.create(backend=Backend.CUDA)
    result = pipeline.process(frame_bgr)
    canvas = render(frame_bgr, result.faces, result.boxes)
"""

from __future__ import annotations

from .geometry import NUM_LANDMARKS, NUM_LANDMARKS_WITH_IRISES, BBox, FaceLandmarks
from .media import open_source, open_writer
from .pipeline import FaceMeshPipeline, FaceMeshResult
from .protocols import FaceDetector, LandmarkEstimator
from .runtime import Backend, preload_cuda_libraries, providers_for
from .uniface_models import (
    DEFAULT_MODELS_DIR,
    MESH_V1_468,
    MESH_V2_478,
    UnifaceFaceDetector,
    UnifaceFaceMesh,
)
from .visualize import DrawingMode, draw_boxes, draw_landmarks, render

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_MODELS_DIR",
    "MESH_V1_468",
    "MESH_V2_478",
    "NUM_LANDMARKS",
    "NUM_LANDMARKS_WITH_IRISES",
    "BBox",
    "Backend",
    "DrawingMode",
    "FaceDetector",
    "FaceLandmarks",
    "FaceMeshPipeline",
    "FaceMeshResult",
    "LandmarkEstimator",
    "UnifaceFaceDetector",
    "UnifaceFaceMesh",
    "draw_boxes",
    "draw_landmarks",
    "open_source",
    "open_writer",
    "preload_cuda_libraries",
    "providers_for",
    "render",
]

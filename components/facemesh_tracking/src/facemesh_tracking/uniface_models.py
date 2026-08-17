"""Detection and dense-mesh stages, backed by UniFace.

[UniFace](https://github.com/yakhyo/uniface) ships the ONNX weights and inference for both
halves of this pipeline. These wrappers adapt them to :class:`FaceDetector` /
:class:`LandmarkEstimator` so the pipeline, the renderer and the CLI never see UniFace
types, and so a caller can replace either stage with their own.

Why this pairing
----------------
* ``YOLOv8Face`` returns a *face* box plus the 5-point alignment template, so the mesh can
  rotate the crop upright before inference. Skipping that rotation is what makes dense
  landmarks drift on tilted faces.
* ``FaceMesh`` V2_478 comes from Google's ``face_landmarker.task`` bundle: 478 points
  (468 + irises) in float32, i.e. sub-pixel positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .geometry import BBox, FaceLandmarks
from .runtime import Backend, preload_cuda_libraries, providers_for

#: Where UniFace caches its auto-downloaded weights for this project.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

#: 478-point mesh with irises. Preferred; see the module docstring.
MESH_V2_478 = "V2_478"
#: 468-point mesh — the older MediaPipe export, no iris points.
MESH_V1_468 = "V1_468"


def configure_cache(models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Point UniFace's auto-download cache at ``models_dir`` instead of ``~/.uniface``.

    Also preloads the CUDA libraries, since UniFace imports onnxruntime on first use.
    """
    preload_cuda_libraries()
    from uniface.model_store import set_cache_dir  # noqa: PLC0415 - after CUDA preload

    models_dir.mkdir(parents=True, exist_ok=True)
    set_cache_dir(str(models_dir))
    return models_dir


class UnifaceFaceDetector:
    """``YOLOv8Face`` — face boxes with the 5-point alignment template."""

    def __init__(
        self,
        backend: Backend = Backend.CUDA,
        *,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        device_id: int = 0,
        models_dir: Path = DEFAULT_MODELS_DIR,
    ) -> None:
        configure_cache(models_dir)
        from uniface.detection import YOLOv8Face  # noqa: PLC0415 - after CUDA preload

        self.providers = providers_for(backend, device_id)
        self._model = YOLOv8Face(
            confidence_threshold=score_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            providers=self.providers,
        )

    def detect(self, image_bgr: np.ndarray) -> list[BBox]:
        height, width = image_bgr.shape[:2]
        boxes: list[BBox] = []
        for face in self._model.detect(image_bgr):
            x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])
            box = BBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                score=float(face.confidence),
                keypoints=(
                    None if face.landmarks is None else np.asarray(face.landmarks, np.float32)
                ),
            ).clipped(width, height)
            if box.is_valid:
                boxes.append(box)
        return boxes


class UnifaceFaceMesh:
    """``FaceMesh`` — dense 3D mesh, aligned via the detector's eye keypoints."""

    def __init__(
        self,
        backend: Backend = Backend.CUDA,
        *,
        model_name: str = MESH_V2_478,
        score_threshold: float = 0.5,
        margin: float = 0.25,
        device_id: int = 0,
        models_dir: Path = DEFAULT_MODELS_DIR,
    ) -> None:
        configure_cache(models_dir)
        from uniface import FaceMesh  # noqa: PLC0415 - after CUDA preload
        from uniface.constants import FaceMeshWeights  # noqa: PLC0415

        self.providers = providers_for(backend, device_id)
        self.model_name = model_name
        self.score_threshold = score_threshold
        #: UniFace expands the box by this ratio itself while aligning the crop, which is
        #: why the pipeline's own margin defaults to 0.0 — expanding twice hurts accuracy.
        self.margin = margin
        self._model = FaceMesh(model_name=FaceMeshWeights[model_name], providers=self.providers)

    def estimate(self, image_bgr: np.ndarray, boxes: Sequence[BBox]) -> list[FaceLandmarks]:
        """Estimate landmarks for every box in a single batched inference call."""
        usable = [box for box in boxes if box.is_valid]
        if not usable:
            return []

        bboxes = np.asarray(
            [[box.x1, box.y1, box.x2, box.y2, box.score] for box in usable], dtype=np.float32
        )
        keypoints = (
            np.stack([box.keypoints for box in usable])
            if all(box.keypoints is not None for box in usable)
            else None
        )
        results = self._model.predict(
            image_bgr, bboxes=bboxes, keypoints=keypoints, margin=self.margin
        )
        return [
            FaceLandmarks(
                points=np.asarray(r.landmarks, np.float32), score=float(r.score), bbox=box
            )
            for r, box in zip(results, usable, strict=True)
            if float(r.score) >= self.score_threshold
        ]

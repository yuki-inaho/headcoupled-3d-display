from __future__ import annotations

import pytest

from facemesh_tracking.uniface_models import DEFAULT_MODELS_DIR

#: UniFace fetches these itself on first use; tests must not trigger a download.
UNIFACE_MODELS = ("yolov8n_face.onnx", "face_landmarker.onnx", "face_mesh.onnx")


def requires_uniface(*names: str):
    """Skip a test when the UniFace weights it needs have not been cached yet."""
    missing = [n for n in (names or UNIFACE_MODELS) if not (DEFAULT_MODELS_DIR / n).is_file()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"UniFace weights missing ({', '.join(missing)}); run `just image <img>` once",
    )

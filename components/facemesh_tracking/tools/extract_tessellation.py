"""Regenerate ``src/facemesh_tracking/assets/tessellation.json``.

The MediaPipe FaceMesh edge lists are taken from the upstream demo
(PINTO0309/facemesh_onnx_tensorrt, Apache-2.0). Run only when the asset needs refreshing::

    python tools/extract_tessellation.py
"""

from __future__ import annotations

import ast
import json
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/PINTO0309/facemesh_onnx_tensorrt/main/demo_video.py"
VARIABLES = {"FACEMESH_TESSELATION_FULL": "full", "FACEMESH_TESSELATION_PARTIAL": "partial"}
TARGET = Path(__file__).resolve().parents[1] / "src/facemesh_tracking/assets/tessellation.json"


def main() -> None:
    with urllib.request.urlopen(SOURCE_URL) as response:  # noqa: S310
        source = response.read().decode("utf-8")

    edges: dict[str, list[list[int]]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = VARIABLES.get(node.targets[0].id)
            if key:
                edges[key] = [list(map(int, edge)) for edge in ast.literal_eval(node.value)]

    missing = set(VARIABLES.values()) - set(edges)
    if missing:
        raise RuntimeError(f"Could not extract {missing} from {SOURCE_URL}")

    TARGET.write_text(json.dumps(edges, separators=(",", ":")) + "\n")
    for key, value in edges.items():
        print(f"{key}: {len(value)} edges")


if __name__ == "__main__":
    main()

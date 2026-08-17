# Third-party and bundled component notices

## AprilTag camera calibrator

`components/tagcal/` is the user-supplied `apriltag-camera-calibrator-main` snapshot.
Its bundled `LICENSE` declares the MIT License. The original license file is retained.

## FaceMesh tracking component

`components/facemesh_tracking/` is the user-supplied
`facemesh_tracking_reconstruction-main` snapshot. The supplied snapshot does not contain
a top-level license file and its `pyproject.toml` does not declare a license. Confirm
redistribution rights before publishing this component outside the context in which it
was supplied.

The component downloads UniFace/ONNX model assets on first use. Those packages and model
weights remain subject to their own licenses and notices.

## Python and browser dependencies

FastAPI, Uvicorn, NumPy, SciPy, OpenCV, Pydantic, PyYAML, Typer, pytest, HTTPX, and
Playwright are not vendored in this archive. They are installed by `uv` and remain under
their respective licenses.

## bunny.pcd

`src/headcoupled_display/static/assets/bunny.pcd` is an original synthetic point cloud
created by `scripts/generate_bunny.py`. It is not the Stanford Bunny and does not contain
third-party scan data.

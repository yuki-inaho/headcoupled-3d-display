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

## bunny.pcd -- Stanford Bunny (third-party scan data)

> **Corrected 2026-08-17.** This section previously stated that `bunny.pcd` was an
> original synthetic cloud and "is not the Stanford Bunny". That is no longer true, and
> leaving it would have been a licence-attribution error, not just a stale note.

`src/headcoupled_display/static/assets/bunny.pcd` is the **Stanford Bunny**
(`bun_zipper` reconstruction, 35,947 vertices) from the Stanford 3D Scanning Repository,
<http://graphics.stanford.edu/data/3Dscanrep/>. Source: Stanford University Computer
Graphics Laboratory.

It is converted from a local ASCII PLY by `scripts/import_stanford_bunny.py`, which
retains every source vertex, keeps the source orientation, and writes a height-based
colour ramp. The geometry is the repository's; the
orientation and colour are this project's presentation choices, applied at asset-build
time. SHA-256 of the converted file: `ec3f5de7243fc500eca380b41a539c0c6f1052e728d2dfc21cccf1990dbb83f5`.

**Usage terms:** the repository permits research use and free redistribution with
attribution to the Stanford Computer Graphics Laboratory; commercial use or inclusion in
a product for sale requires Stanford's permission. The terms as read on 2026-08-17 are
quoted in full, together with what could and could not be verified, in
`docs/input_manifest.md`.

`scripts/generate_bunny.py` still exists and still produces a synthetic bunny-shaped
cloud, but **it is no longer the source of the displayed asset**.

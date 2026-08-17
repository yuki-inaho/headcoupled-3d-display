#!/usr/bin/env python3
"""Convert the real Stanford Bunny into the ASCII PCD the WebGL renderer loads.

This replaces the demo asset that used to sit at
``src/headcoupled_display/static/assets/bunny.pcd``: that file was a synthetic
"bunny-like" point cloud produced by ``generate_bunny.py`` from sampled
ellipsoids, not a scan of anything. This script instead reads the actual
Stanford Bunny mesh vertices and writes them out as the asset.

Source and provenance
----------------------
Stanford 3D Scanning Repository (http://graphics.stanford.edu/data/3Dscanrep/),
"Stanford Bunny" entry. Per that page: Source = Stanford University Computer
Graphics Laboratory, Scanner = Cyberware 3030 MS, 10 range scans, "zipper"
reconstruction, 35947 vertices / 69451 triangles, "contains 5 holes in the
bottom" (the physical model's base, where it stood on the scanner turntable).
The default ``--input`` below is the local copy Open3D bundles as its
"BunnyMesh" example dataset -- confirmed to be this exact reconstruction
because its header reports the same vertex/triangle counts as the repository
listing (35947 / 69451) and ``comment zipper output``. That file is read-only
input here and is never modified or copied into this repository.

Usage terms for this data (quoted from the repository page, verified by
fetching it directly on 2026-08-17; full quote also recorded in
``docs/input_manifest.md``): research use and free mirroring/redistribution
are permitted with attribution to the Stanford Computer Graphics Laboratory;
commercial use, or use in a product for sale, requires Stanford's permission.

Design decisions
-----------------
* Y-axis orientation: kept as-is. The raw PLY is already Y-up in the same
  sense the scene expects (ears at +Y, base/feet at -Y): its measured
  bounding box is X 0.1557 m, Y 0.1543 m, Z 0.1207 m, matching the commonly
  cited ~156x154x121 mm envelope for this exact reconstruction.
* Heading: rotated 180 deg about Y (``rotate_y_180``). The raw PLY faces
  -Z, but the display frame puts the observer at +Z (a head-coupled display
  is meant to be looked *into*, so the subject should look back). Verified
  numerically, not assumed: on the unrotated cloud the head (top 10% by Y)
  has a mean Z of -0.0215 m against a whole-cloud mean Z of +0.0089 m, i.e.
  the head leans away from the viewer; ``head_faces_viewer`` re-checks this
  before and after the turn and ``main`` aborts if the direction is ever
  wrong, so a future change to the input can't silently ship a
  backwards-facing bunny.
* Colour: a rainbow ramp over height (``rainbow_colors``), not the uniform
  grey-blue the retired synthetic asset used. The source PLY carries no RGB
  -- only per-vertex ``confidence`` (registration quality from zippering,
  not appearance) and ``intensity`` (constant 0.5 for every vertex in this
  file, so also not real colour) -- so this is a display choice made at
  asset-build time, not scanner data reinterpreted as colour. Height (Y)
  was chosen over depth (Z) or a per-triangle scheme because the bunny
  stands upright, so a vertical ramp reads as a rainbow across the whole
  animal and does not change as the observer moves around it.
* No decimation. All 35947 vertices are kept (up from the synthetic asset's
  13810 points, ratio ~2.6x). Measured output: ~1.4-1.5 MB ASCII (vs. the
  synthetic asset's 595 KB) -- see the printed summary below for the exact
  figure from the file actually written. That is a small, one-time local
  asset served over loopback HTTP in tests, and WebGL2 drawing ~36k points as
  ``gl.POINTS`` is trivial even under headless SwiftShader; there is no
  rendering or E2E budget in this repository that scales with point count
  (checked: no FPS/latency assertion in tests/e2e/test_browser.py is a
  function of point count). Keeping every vertex is also the more faithful
  choice: this is the official reconstruction, not a resampling of it.

The FIELDS/TYPE layout matches the retired synthetic asset exactly
(``x y z r g b``, ``F F F U U U``) so the existing ``pcd.js`` parser and the
tested header-consistency checks need no changes.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

DEFAULT_INPUT = Path("/home/inaho-omen/open3d_data/extract/BunnyMesh/BunnyMesh.ply")
DEFAULT_OUTPUT = Path("src/headcoupled_display/static/assets/bunny.pcd")

# Reused verbatim from the retired scripts/generate_bunny.py "fur" colour -- see the
# module docstring above for why a uniform colour was chosen over deriving one from
# the source PLY's confidence/intensity fields.
#: Hue sweep used to colour the cloud by height. Stops short of a full turn so the
#: bottom and the top do not land on the same hue and read as one flat colour.
HUE_START_DEG = 250.0
HUE_SWEEP_DEG = -290.0
#: Kept high and fixed: the ramp is meant to read as depth/height, and letting
#: saturation or value vary as well would make two different heights indistinguishable
#: wherever the third channel compensated.
COLOR_SATURATION = 0.72
COLOR_VALUE = 0.98


def rotate_y_180(points: np.ndarray) -> np.ndarray:
    """Turn the bunny to face the observer.

    The display frame has +Z toward the viewer, and bun_zipper faces the other way.
    Measured on the unrotated cloud: the head (the top 10% by Y, where the ears are)
    has a mean Z of -0.0215 m while the whole cloud averages +0.0089 m, so the head
    leans away from the viewer. A half turn about Y -- the axis the bunny stands on --
    is the correction; it leaves the upright pose alone and only changes the heading.
    """

    turned = points.copy()
    turned[:, 0] = -turned[:, 0]
    turned[:, 2] = -turned[:, 2]
    return turned


def head_faces_viewer(points: np.ndarray) -> bool:
    """True when the head leans toward +Z, i.e. toward the observer.

    Used as a post-condition rather than a belief: the rotation above is only correct
    if this flips, and a silently wrong-facing asset looks plausible on screen.
    """

    head = points[points[:, 1] > np.percentile(points[:, 1], 90)]
    return float(head[:, 2].mean()) > float(points[:, 2].mean())


def rainbow_colors(points: np.ndarray) -> np.ndarray:
    """Colour every point by its height, as ``(N, 3)`` uint8 RGB.

    Height rather than depth: the bunny stands upright, so a vertical ramp reads as a
    rainbow over the whole animal, and it stays stable as the observer moves. Colouring
    by Z would change what a given point looks like depending on nothing but the model's
    own thickness, which is harder to read.

    The source PLY carries no RGB -- only ``confidence`` and a constant ``intensity`` --
    so nothing here is scan data being reinterpreted as colour. This is a display choice,
    and it is applied at asset-build time so the renderer keeps no colour rules of its own.
    """

    y = points[:, 1]
    span = float(y.max() - y.min())
    t = (y - y.min()) / span if span > 0 else np.zeros_like(y)
    hue = (HUE_START_DEG + HUE_SWEEP_DEG * t) % 360.0
    # HSV -> RGB with fixed S and V, written out rather than pulling in a dependency.
    h = hue / 60.0
    c = COLOR_VALUE * COLOR_SATURATION
    x = c * (1.0 - np.abs((h % 2.0) - 1.0))
    m = COLOR_VALUE - c
    zeros = np.zeros_like(h)
    sector = np.floor(h).astype(int) % 6
    r = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [c, x, zeros, zeros, x],
        default=c,
    )
    g = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [x, c, c, x, zeros],
        default=zeros,
    )
    b = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [zeros, zeros, x, c, c],
        default=x,
    )
    rgb = np.stack([r, g, b], axis=1) + m
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_vertex_element(lines: Iterator[str], path: Path) -> tuple[int, int, int, int]:
    """Consume PLY header lines up to and including ``end_header``.

    Returns ``(vertex_count, x_index, y_index, z_index)`` for the vertex element's
    property list, e.g. ``x y z confidence intensity`` -> ``(N, 0, 1, 2)``. Split out
    of `read_ascii_ply_vertices` purely to keep that function's branching simple.
    """
    vertex_count: int | None = None
    vertex_properties: list[str] = []
    in_vertex_element = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("comment"):
            continue
        if stripped.startswith("element"):
            _, name, count = stripped.split()
            in_vertex_element = name == "vertex"
            if in_vertex_element:
                vertex_count = int(count)
            continue
        if stripped.startswith("property") and in_vertex_element:
            vertex_properties.append(stripped.split()[-1])
            continue
        if stripped == "end_header":
            break

    if vertex_count is None:
        raise ValueError(f"no 'element vertex' found in {path}")
    for axis in ("x", "y", "z"):
        if axis not in vertex_properties:
            raise ValueError(f"vertex element is missing '{axis}': {vertex_properties}")
    return (
        vertex_count,
        vertex_properties.index("x"),
        vertex_properties.index("y"),
        vertex_properties.index("z"),
    )


def read_ascii_ply_vertices(path: Path) -> np.ndarray:
    """Read only the x, y, z vertex coordinates from an ASCII PLY file.

    This is a narrow reader for the Stanford ``bun_zipper`` layout
    (``property float x/y/z/confidence/intensity`` on the vertex element,
    followed by a face list) -- not a general-purpose PLY parser. It fails
    loudly on anything it cannot account for (binary format, missing x/y/z)
    rather than silently reading nonsense.
    """
    with path.open("r", encoding="ascii") as handle:
        lines = iter(handle)
        if next(lines).strip() != "ply":
            raise ValueError(f"not a PLY file: {path}")
        header_format = next(lines).strip()
        if header_format != "format ascii 1.0":
            raise ValueError(f"expected 'format ascii 1.0', got {header_format!r} in {path}")

        vertex_count, x_index, y_index, z_index = _parse_vertex_element(lines, path)

        points = np.empty((vertex_count, 3), dtype=np.float64)
        for row in range(vertex_count):
            values = next(lines).split()
            points[row] = (
                float(values[x_index]),
                float(values[y_index]),
                float(values[z_index]),
            )
    return points


def write_ascii_pcd(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    source_path: Path,
    source_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = points.shape[0]
    header = f"""# .PCD v0.7 - Point Cloud Data file format
# Stanford Bunny (bun_zipper reconstruction), Stanford 3D Scanning Repository.
# http://graphics.stanford.edu/data/3Dscanrep/ -- Source: Stanford University
# Computer Graphics Laboratory. Cyberware 3030 MS scanner, 10 range scans,
# "zipper" reconstruction, 35947 vertices / 69451 triangles per the repository's
# own listing for this model (verified to match this file's vertex count).
# This is a real 3D scan, NOT the synthetic asset scripts/generate_bunny.py
# produces. Converted by scripts/import_stanford_bunny.py; all {count} source
# vertices retained (no decimation -- see that script's docstring for why).
# Usage terms (Stanford 3D Scanning Repository, quoted in full in
# docs/input_manifest.md): research use and free mirroring/redistribution
# permitted with attribution; commercial use requires Stanford's permission.
# Source file: {source_path}
# Source SHA-256: {source_sha256}
# Orientation: rotated 180 deg about Y so the head faces the observer (+Z).
# Colour: rainbow ramp over height; the source PLY has no RGB, see script docstring.
VERSION 0.7
FIELDS x y z r g b
SIZE 4 4 4 1 1 1
TYPE F F F U U U
COUNT 1 1 1 1 1 1
WIDTH {count}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {count}
DATA ascii
"""
    with path.open("w", encoding="ascii") as handle:
        handle.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors, strict=True):
            handle.write(f"{x:.7f} {y:.7f} {z:.7f} {r} {g} {b}\n")


def summarize_written_pcd(path: Path) -> None:
    """Re-parse the file exactly as written (same rounding pcd.js will see) and
    print the bounds a test author would independently compute with NumPy."""
    positions = []
    with path.open("r", encoding="ascii") as handle:
        in_data = False
        for line in handle:
            if not in_data:
                if line.strip().upper() == "DATA ASCII":
                    in_data = True
                continue
            values = line.split()
            positions.append((float(values[0]), float(values[1]), float(values[2])))
    points = np.array(positions, dtype=np.float64)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) / 2
    centroid = points.mean(axis=0)
    longest_edge = float((mx - mn).max())
    size_bytes = path.stat().st_size

    print(f"wrote {len(points)} points to {path} ({size_bytes / 1024:.1f} KiB)")
    print(f"  min    = ({mn[0]:.7f}, {mn[1]:.7f}, {mn[2]:.7f})")
    print(f"  max    = ({mx[0]:.7f}, {mx[1]:.7f}, {mx[2]:.7f})")
    print(f"  center = ({center[0]:.7f}, {center[1]:.7f}, {center[2]:.7f})  # AABB midpoint")
    print(
        f"  centroid = ({centroid[0]:.8f}, {centroid[1]:.8f}, {centroid[2]:.8f})  # mean, NOT the AABB center"
    )
    print(f"  longest_edge_m = {longest_edge:.7f}")
    print(f"  derived_uniform_scale (0.24 m target) = {0.24 / longest_edge:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(
            f"input PLY not found: {args.input}\n"
            "This script converts a local copy of the Stanford Bunny "
            "('bun_zipper', 35947 vertices) -- e.g. Open3D's bundled "
            "'BunnyMesh' example dataset, or bun_zipper.ply from "
            "http://graphics.stanford.edu/data/3Dscanrep/ -- and does not "
            "download or bundle it itself."
        )

    source_sha256 = sha256_of(args.input)
    points = read_ascii_ply_vertices(args.input)
    if points.shape[0] != 35947:
        print(
            f"warning: expected 35947 vertices (bun_zipper), got {points.shape[0]} "
            f"from {args.input}; proceeding anyway"
        )
    if head_faces_viewer(points):
        raise SystemExit(
            "the source cloud already faces the viewer; the half turn below would turn "
            "it away. Re-check the input before changing this."
        )
    points = rotate_y_180(points)
    if not head_faces_viewer(points):
        raise SystemExit("the half turn about Y did not make the head face +Z")
    colors = rainbow_colors(points)
    write_ascii_pcd(
        args.output,
        points,
        colors,
        source_path=args.input,
        source_sha256=source_sha256,
    )
    summarize_written_pcd(args.output)


if __name__ == "__main__":
    main()

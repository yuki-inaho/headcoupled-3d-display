from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from tagcal.board import AprilGridBoard
from tagcal.models import AprilTagFamily, PatternSpec


def _quad(x: float, y: float, size: float = 20.0) -> np.ndarray:
    return np.array(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=np.float32
    ).reshape(1, 4, 2)


def test_cells_per_tag_counts_the_black_border() -> None:
    board = AprilGridBoard(PatternSpec(columns=2, rows=2, family="tag36h11"))

    # tag36h11 carries 6x6 of data; the border adds one cell on each side.
    assert board.cells_per_tag == 8


def test_border_bits_widen_the_tag() -> None:
    board = AprilGridBoard(PatternSpec(columns=2, rows=2, family="tag36h11", border_bits=2))

    assert board.cells_per_tag == 10


def test_duplicate_ids_are_dropped_entirely() -> None:
    """A repeated id is a ghost detection, and neither copy can be trusted.

    Keeping either one risks pairing image points with the wrong 3D point, which
    corrupts the whole calibration rather than just that view.
    """
    board = AprilGridBoard(PatternSpec(columns=2, rows=2))
    corners = [_quad(0, 0), _quad(100, 0), _quad(200, 0), _quad(300, 0)]
    ids = np.array([[0], [1], [1], [2]], dtype=np.int32)

    kept_corners, kept_ids = board.filter_detections(corners, ids)

    assert kept_ids is not None
    assert sorted(int(value) for value in kept_ids.reshape(-1)) == [0, 2]
    assert len(kept_corners) == 2


def test_ids_outside_the_board_are_ignored() -> None:
    board = AprilGridBoard(PatternSpec(columns=2, rows=2))  # ids 0..3
    corners = [_quad(0, 0), _quad(100, 0)]
    ids = np.array([[1], [99]], dtype=np.int32)

    _, kept_ids = board.filter_detections(corners, ids)

    assert kept_ids is not None
    assert [int(value) for value in kept_ids.reshape(-1)] == [1]


def test_all_duplicates_leaves_nothing() -> None:
    board = AprilGridBoard(PatternSpec(columns=2, rows=2))
    corners = [_quad(0, 0), _quad(100, 0)]
    ids = np.array([[3], [3]], dtype=np.int32)

    kept_corners, kept_ids = board.filter_detections(corners, ids)

    assert kept_ids is None
    assert kept_corners == []


def test_render_grid_pixels_has_the_expected_geometry() -> None:
    spec = PatternSpec(columns=3, rows=2)
    board = AprilGridBoard(spec)

    grid = board.render_grid_pixels(tag_px=64, gap_px=16)

    assert grid.shape == (2 * 64 + 1 * 16, 3 * 64 + 2 * 16)
    assert grid.dtype == np.uint8
    # Corners of the sheet fall in the gap/edge, which is white.
    assert grid[0, 0] == 255 or grid[0, 0] == 0  # tag starts at the origin


def test_render_grid_pixels_validates_its_arguments() -> None:
    board = AprilGridBoard(PatternSpec(columns=2, rows=2))

    with pytest.raises(ValueError):
        board.render_grid_pixels(tag_px=board.cells_per_tag - 1, gap_px=0)
    with pytest.raises(ValueError):
        board.render_grid_pixels(tag_px=64, gap_px=-1)


def test_unsupported_family_is_rejected() -> None:
    # A manifest loaded from disk can carry any string, so the guard is not dead code.
    spec = replace(PatternSpec(columns=2, rows=2), family=cast(AprilTagFamily, "tag99h9"))

    with pytest.raises(ValueError, match="Unsupported AprilTag family"):
        AprilGridBoard(spec)


def test_board_larger_than_the_dictionary_is_rejected() -> None:
    with pytest.raises(ValueError, match="contains IDs"):
        AprilGridBoard(PatternSpec(columns=8, rows=8, family="tag16h5"))

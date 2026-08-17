import numpy as np
import pytest

from tagcal.models import SelectionSpec
from tagcal.selection import local_sharpness_maxima


def test_keeps_the_peak_of_each_burst() -> None:
    # Two intended poses, blurred motion in between.
    indices = [0, 2, 4, 6, 8, 20, 22, 24, 26, 28]
    sharpness = [1.0, 3.0, 9.0, 3.0, 1.0, 1.0, 2.0, 8.0, 2.0, 1.0]

    kept = local_sharpness_maxima(indices, sharpness, window_frames=5, relative_floor=0.0)

    assert [indices[position] for position in kept] == [4, 24]


def test_wider_window_keeps_fewer_frames() -> None:
    rng = np.random.default_rng(1)
    indices = list(range(0, 200, 2))
    sharpness = rng.permutation(len(indices)).astype(float).tolist()  # all distinct

    narrow = local_sharpness_maxima(indices, sharpness, window_frames=2, relative_floor=0.0)
    wide = local_sharpness_maxima(indices, sharpness, window_frames=40, relative_floor=0.0)
    whole = local_sharpness_maxima(indices, sharpness, window_frames=10_000, relative_floor=0.0)

    assert len(narrow) > len(wide) > len(whole)
    assert len(whole) == 1  # a window spanning everything leaves the global maximum


def test_relative_floor_drops_a_uniformly_blurred_stretch() -> None:
    # The second burst peaks well below the recording's median.
    indices = [0, 2, 4, 20, 22, 24]
    sharpness = [5.0, 10.0, 5.0, 0.2, 0.4, 0.2]

    without_floor = local_sharpness_maxima(indices, sharpness, window_frames=5, relative_floor=0.0)
    with_floor = local_sharpness_maxima(indices, sharpness, window_frames=5, relative_floor=0.5)

    assert [indices[position] for position in without_floor] == [2, 22]
    assert [indices[position] for position in with_floor] == [2]


def test_absolute_scale_does_not_matter() -> None:
    """The measure's units are scene dependent, so only relative order may count."""
    indices = [0, 2, 4, 6]
    sharpness = [1.0, 4.0, 2.0, 1.0]

    baseline = local_sharpness_maxima(indices, sharpness, window_frames=3, relative_floor=0.5)
    scaled = local_sharpness_maxima(
        indices, [value * 1000.0 for value in sharpness], window_frames=3, relative_floor=0.5
    )

    assert baseline == scaled


def test_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError):
        local_sharpness_maxima([0, 1], [1.0], window_frames=2, relative_floor=0.0)
    with pytest.raises(ValueError):
        local_sharpness_maxima([0, 1], [1.0, 2.0], window_frames=0, relative_floor=0.0)


def test_empty_input_is_not_an_error() -> None:
    assert local_sharpness_maxima([], [], window_frames=3, relative_floor=0.5) == []


def test_spec_rejects_an_out_of_range_floor() -> None:
    with pytest.raises(ValueError):
        SelectionSpec(blur_relative_floor=1.5)
    with pytest.raises(ValueError):
        SelectionSpec(blur_window_seconds=0.0)


def test_sampling_is_dense_enough_for_the_blur_window() -> None:
    """Several frames must fall inside the window, or there is nothing to compare."""
    spec = SelectionSpec()
    frames_per_window = spec.blur_window_seconds * spec.sample_fps

    assert frames_per_window >= 2.0


def test_maxima_are_returned_in_order() -> None:
    rng = np.random.default_rng(3)
    indices = list(range(0, 200, 2))
    sharpness = rng.uniform(1.0, 10.0, len(indices)).tolist()

    kept = local_sharpness_maxima(indices, sharpness, window_frames=6, relative_floor=0.0)

    assert kept == sorted(kept)
    assert all(0 <= position < len(indices) for position in kept)

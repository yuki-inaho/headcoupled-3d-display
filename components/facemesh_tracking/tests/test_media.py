from __future__ import annotations

import cv2
import numpy as np
import pytest

from facemesh_tracking.media import open_source, open_writer


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.full((30, 40, 3), 128, dtype=np.uint8))
    return path


def test_image_source_yields_exactly_one_frame(image_file):
    with open_source(str(image_file)) as source:
        frames = list(source)
    assert len(frames) == 1
    assert frames[0].shape == (30, 40, 3)
    assert source.info.is_image
    assert (source.info.width, source.info.height) == (40, 30)


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError), open_source(str(tmp_path / "nope.png")):
        pass


def test_unopenable_video_raises(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="Could not open"), open_source(str(path)):
        pass


def test_video_round_trip(tmp_path, image_file):
    with open_source(str(image_file)) as source:
        info = source.info
    video_path = tmp_path / "out.mp4"

    with open_writer(video_path, info) as writer:
        for value in (10, 20, 30):
            writer.write(np.full((info.height, info.width, 3), value, dtype=np.uint8))

    with open_source(str(video_path)) as source:
        frames = list(source)
    assert len(frames) == 3
    assert not source.info.is_image
    assert (source.info.width, source.info.height) == (info.width, info.height)

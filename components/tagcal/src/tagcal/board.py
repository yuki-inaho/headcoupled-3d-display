from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from tagcal.cvtypes import as_uint8
from tagcal.models import PatternSpec

FAMILY_TO_DICTIONARY_ID: dict[str, int] = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass(frozen=True, slots=True)
class MatchedPoints:
    object_points: NDArray[np.float32]
    image_points: NDArray[np.float32]


class AprilGridBoard:
    """OpenCV-backed AprilTag grid board with explicit physical dimensions."""

    def __init__(self, spec: PatternSpec) -> None:
        if spec.family not in FAMILY_TO_DICTIONARY_ID:
            supported = ", ".join(sorted(FAMILY_TO_DICTIONARY_ID))
            raise ValueError(f"Unsupported AprilTag family: {spec.family}. Supported: {supported}")

        self._spec = spec
        self._dictionary = cv2.aruco.getPredefinedDictionary(
            FAMILY_TO_DICTIONARY_ID[spec.family]
        )
        dictionary_size = int(self._dictionary.bytesList.shape[0])
        last_id = spec.first_id + spec.marker_count - 1
        if last_id >= dictionary_size:
            raise ValueError(
                f"Board requires marker ID {last_id}, but {spec.family} contains IDs "
                f"0..{dictionary_size - 1}"
            )

        self._ids = np.arange(
            spec.first_id,
            spec.first_id + spec.marker_count,
            dtype=np.int32,
        )
        self._board = cv2.aruco.GridBoard(
            (spec.columns, spec.rows),
            float(spec.effective_tag_size_mm),
            float(spec.effective_gap_mm),
            self._dictionary,
            self._ids,
        )
        self._id_set = frozenset(int(value) for value in self._ids)

    @property
    def spec(self) -> PatternSpec:
        return self._spec

    @property
    def dictionary(self) -> cv2.aruco.Dictionary:
        return self._dictionary

    @property
    def cv_board(self) -> cv2.aruco.GridBoard:
        return self._board

    @property
    def ids(self) -> NDArray[np.int32]:
        return self._ids.copy()

    @property
    def id_set(self) -> frozenset[int]:
        return self._id_set

    @property
    def marker_count(self) -> int:
        return self._spec.marker_count

    @property
    def cells_per_tag(self) -> int:
        """Cells along one tag edge, black border included.

        This is what a tag size refers to: the outer edge of the black border,
        never the white quiet zone around it. Reading the quiet zone as part of
        the tag inflates every distance by the ratio of the two.
        """
        return int(self._dictionary.markerSize) + 2 * self._spec.border_bits

    @property
    def diagonal_mm(self) -> float:
        return float(
            np.hypot(
                self._spec.effective_board_width_mm,
                self._spec.effective_board_height_mm,
            )
        )

    def render_grid_pixels(self, *, tag_px: int, gap_px: int) -> NDArray[np.uint8]:
        """Render the tag grid on white at an exact pixel pitch, without outer margin.

        `tag_px` should be a multiple of `cells_per_tag`; otherwise OpenCV resamples
        the marker and individual cells end up 1px wider or narrower than each other,
        which is a distortion of the tag coordinate system itself.
        """
        if tag_px < self.cells_per_tag:
            raise ValueError(f"tag_px must be at least {self.cells_per_tag} px")
        if gap_px < 0:
            raise ValueError("gap_px must not be negative")

        spec = self._spec
        width = spec.columns * tag_px + (spec.columns - 1) * gap_px
        height = spec.rows * tag_px + (spec.rows - 1) * gap_px
        canvas = np.full((height, width), 255, dtype=np.uint8)
        marker_ids = self._ids.tolist()
        for row in range(spec.rows):
            for column in range(spec.columns):
                index = row * spec.columns + column
                marker = as_uint8(
                    cv2.aruco.generateImageMarker(
                        self._dictionary,
                        int(marker_ids[index]),
                        tag_px,
                        borderBits=spec.border_bits,
                    )
                )
                top = row * (tag_px + gap_px)
                left = column * (tag_px + gap_px)
                canvas[top : top + tag_px, left : left + tag_px] = marker
        return canvas

    def filter_detections(
        self,
        corners: tuple[NDArray[np.float32], ...] | list[NDArray[np.float32]],
        ids: NDArray[np.int32] | None,
    ) -> tuple[list[NDArray[np.float32]], NDArray[np.int32] | None]:
        if ids is None or len(ids) == 0:
            return [], None

        flattened = ids.reshape(-1)
        selected_corners: list[NDArray[np.float32]] = []
        selected_ids: list[int] = []
        for marker_corners, marker_id in zip(corners, flattened, strict=True):
            value = int(marker_id)
            if value in self._id_set:
                selected_corners.append(np.asarray(marker_corners, dtype=np.float32))
                selected_ids.append(value)

        # The board carries every id exactly once, so a repeat is necessarily a
        # false detection -- a reflection, or the board showing up a second time on
        # a monitor in frame. Nothing distinguishes the genuine copy from the ghost,
        # and keeping either one risks pairing image points with the wrong 3D point,
        # which corrupts the whole calibration rather than just that view.
        duplicated = {value for value, count in Counter(selected_ids).items() if count > 1}
        if duplicated:
            kept = [index for index, value in enumerate(selected_ids) if value not in duplicated]
            selected_corners = [selected_corners[index] for index in kept]
            selected_ids = [selected_ids[index] for index in kept]

        if not selected_ids:
            return [], None
        return selected_corners, np.asarray(selected_ids, dtype=np.int32).reshape(-1, 1)

    def match_image_points(
        self,
        corners: list[NDArray[np.float32]],
        ids: NDArray[np.int32],
    ) -> MatchedPoints:
        object_points, image_points = self._board.matchImagePoints(corners, ids)
        object_array = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        image_array = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        if object_array.shape[0] != image_array.shape[0]:
            raise RuntimeError("OpenCV returned mismatched object/image point counts")
        return MatchedPoints(object_points=object_array, image_points=image_array)

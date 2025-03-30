from dataclasses import dataclass, field
from math import log2
import os

import cv2
import numpy as np

from censorengine.backend.models.detected_part import Part
from censorengine.backend.constants.typing import Mask


@dataclass
class VideoProcessor:
    file_path: str
    new_file_name: str

    width: int = field(init=False)
    height: int = field(init=False)
    fps: int = field(init=False)
    total_frames: int = field(init=False)

    video_capture: cv2.VideoCapture = field(init=False)
    video_writer: cv2.VideoWriter = field(init=False)

    def __post_init__(self):
        self.video_capture = cv2.VideoCapture(self.file_path)  # type: ignore
        if not self.video_capture.isOpened():
            raise ValueError(f"Could not open video: {self.file_path}")

        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*self._get_codec_from_extension(self.file_path))

        self.video_writer = cv2.VideoWriter(
            self.new_file_name,  # type: ignore
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _get_codec_from_extension(self, filename: str) -> str:
        """
        This function is used to find the correct codec used by OpenCV.

        :param str filename: Name of the file, it will determine the extension
        """
        ext = os.path.splitext(filename)[-1].lower()
        CODEC_MAPPING = {
            ".mp4": "mp4v",  # MPEG-4
            ".avi": "XVID",  # AVI format
            ".mov": "avc1",  # QuickTime
            ".mkv": "X264",  # Matroska
            ".webm": "VP80",  # WebM format
        }
        return CODEC_MAPPING.get(ext, "mp4v")  # Default to 'mp4v' if unknown


@dataclass
class VideoFrame:
    """
    This class handles the processing of the parts between frames, such to
    improve the quality of the output.

    Currently, # TODO
    # FIXME Make persistence work with parts rather than frame (see name change)

    :return _type_: _description_
    """

    frame_difference_threshold: float  # Minimum Required difference to change
    frame_hold_amount: int
    size_change_tolerance: float = field(default=1.00)  # Protects against Outliers

    frame_lag_counter: int = field(default=0, init=False)
    debug_counter: int = field(default=0, init=False)

    current_frame: dict[str, Part] = field(default_factory=dict, init=False)
    last_frame: dict[str, Part] = field(default_factory=dict, init=False)
    held_frame: dict[str, Part] = field(default_factory=dict, init=False)

    first_frame: bool = field(default=True, init=False)

    def _convert_to_edges(self, mask: Mask, temp_disable: bool = False):
        if temp_disable:
            return mask
        edges = cv2.Canny(mask, 100, 200)
        kernel = np.ones((15, 15), np.uint8)
        return cv2.dilate(edges, kernel, iterations=2)

    def _compare_parts_areas(
        self,
        current_part: Part,
        held_part: Part,
        for_part_persistence: bool = False,
    ) -> bool:
        """
        TODO

        :param Part current_part: Current part, tested to see if the area is bad
        :param Part held_part: Replaces the current part if it's bad
        :param bool for_part_persistence: Toggle for part persistence due to the check not being neighbouring frames
        :return bool: True means use the current part, False means use the held part
        """
        # Compute individual mask areas
        current_area = np.count_nonzero(current_part.mask)
        held_area = np.count_nonzero(held_part.mask)

        # Compute pixel difference Equations
        diff_percentage = (current_area - held_area) / max(current_area, held_area)

        # Comparisons
        size_tolerance = self.size_change_tolerance
        if for_part_persistence and (self.frame_hold_amount > 1):
            # NOTE: This is done because when this function is used for the
            #       frame persistence function, the code doesn't account for
            #       the fact that it isn't looking at neighbouring frames, but
            #       rather N frames apart, so a difference in area might be
            #       reasonable.
            #
            frames = self.frame_hold_amount + 1
            rate_factor = 0.1  # TODO: Make Config accessible
            size_tolerance *= 1 + rate_factor * log2(frames)  # Adds 1 for safety margin

        non_trivial_diff = abs(diff_percentage) > size_tolerance

        # Output
        if non_trivial_diff:
            # NOTE: Due to the inverse for part persistence
            if for_part_persistence:
                return not diff_percentage < 0

            return diff_percentage < 0

        return True

    def _compare_frames(self, old_part: Part, new_part: Part) -> bool:
        temp_disable = True  # Change to False if you want edge-based comparison
        old_mask = self._convert_to_edges(old_part.mask, temp_disable)
        new_mask = self._convert_to_edges(new_part.mask, temp_disable)

        # Check movement and size change constraints
        return (
            self._compare_parts_areas(old_part.mask, new_part.mask)
            and bool(np.any(old_mask))
            and bool(np.any(new_mask))
        )

    def load_parts(self, parts: list[Part]) -> None:
        self.current_frame = {f"{part.get_name()}": part for part in parts}
        if self.first_frame:
            # Correct Frame
            self.first_frame = False

            # Save Last Frame
            self.last_frame = self.current_frame.copy()

            # Save Held Frame
            self.held_frame = self.current_frame.copy()

    def apply_part_persistence(self) -> None:
        """
        Holds the frame for the duration specified by frame_hold_amount.
        If the frame has been held for enough time, it will be updated.
        """
        # Counter Logic
        if self.frame_lag_counter >= self.frame_hold_amount:
            dict_temp_held = {}
            for key, value in self.current_frame.items():
                if not self.held_frame.get(key):
                    dict_temp_held[key] = value
                    continue

                # Check if Parts are actually Correct over Previous
                output = self._compare_parts_areas(
                    value,
                    self.held_frame[key],
                    for_part_persistence=True,  # TODO: Allow it to be turned off
                )

                dict_temp_held[key] = value if output else self.held_frame[key]

            self.held_frame.clear()
            self.held_frame = dict_temp_held

            # Counter Logic
            self.frame_lag_counter = 0  # Reset lag Counter
        else:
            # Used to Add Missing Parts Normally
            for key, value in self.held_frame.items():
                if not self.current_frame.get(key):
                    self.current_frame[key] = value

            # Counter Logic
            self.frame_lag_counter += 1  # Update Lag Counter

    def apply_part_size_correction(self) -> None:
        for key, value in self.held_frame.items():
            # Determine Part
            output = self._compare_parts_areas(self.current_frame[key], value)
            output_part = self.current_frame[key] if not output else value

            # Update Frames
            self.current_frame[key] = output_part

    def apply_frame_stability(self) -> None:
        """
        Applies frame stability by ensuring that parts don't update for small differences.
        A significant difference (as determined by frame_difference_threshold) is required to update the part.
        """
        for part_name in self.current_frame.keys():
            # Ignore new parts that don't exist in the last frame
            if part_name not in self.last_frame:
                continue

            old_part = self.last_frame[part_name]
            this_part = self.current_frame[part_name]

            # Check if the difference between frames is large enough to update
            self.current_frame[part_name] = (
                this_part if self._compare_frames(old_part, this_part) else old_part
            )

    def save_frame(self):
        self.last_frame = self.current_frame.copy()

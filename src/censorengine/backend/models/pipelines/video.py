from dataclasses import dataclass, field
import os

import cv2

from censorengine.backend.models.structures.detected_part import Part


@dataclass(slots=True)
class VideoProcessor:
    file_path: str
    new_file_name: str

    _width: int = field(init=False)
    _height: int = field(init=False)
    _fps: int = field(init=False)
    _total_frames: int = field(init=False)

    video_capture: cv2.VideoCapture = field(init=False)
    video_writer: cv2.VideoWriter = field(init=False)

    def __post_init__(self):
        self.video_capture = cv2.VideoCapture(self.file_path)  # type: ignore
        if not self.video_capture.isOpened():
            raise ValueError(f"Could not open video: {self.file_path}")

        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*self._get_codec_from_extension(self.file_path))

        self.video_writer = cv2.VideoWriter(
            self.new_file_name,  # type: ignore
            fourcc,
            self._fps,
            (self._width, self._height),
        )
        self._total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

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

    def get_fps(self):
        return self._fps


@dataclass
class FramePart:
    part: Part

    part_name: str = field(init=False)
    is_merged: bool = field(init=False)

    def __post_init__(self):
        self.part_name = self.part.get_name()
        self.is_merged = self.part.is_merged


@dataclass
class FrameProcessor:
    """
    This class handles the processing of the parts between frames, such to
    improve the quality of the output.

    Currently, # TODO
    # FIXME Make persistence work with parts rather than frame (see name change)

    :return _type_: _description_
    """

    frame_difference_threshold: float  # Minimum Required difference to change
    part_frame_hold_seconds: float

    frame_lag_counter: int = field(default=0, init=False)
    debug_counter: int = field(default=0, init=False)

    current_frame: dict[str, FramePart] = field(default_factory=dict, init=False)
    last_frame: dict[str, FramePart] = field(default_factory=dict, init=False)
    held_frame: dict[str, FramePart] = field(default_factory=dict, init=False)

    first_frame: bool = field(default=True, init=False)

    # def _convert_to_edges(self, mask: Mask, temp_disable: bool = False):
    #     if temp_disable:
    #         return mask
    #     edges = cv2.Canny(mask, 100, 200)
    #     kernel = np.ones((15, 15), np.uint8)
    #     return cv2.dilate(edges, kernel, iterations=2)

    # def _compare_frames(self, old_part: Part, new_part: Part) -> bool:
    #     temp_disable = True  # Change to False if you want edge-based comparison
    #     old_mask = self._convert_to_edges(old_part.mask, temp_disable)
    #     new_mask = self._convert_to_edges(new_part.mask, temp_disable)

    #     # Check movement and size change constraints
    #     return (
    #         self._compare_parts_areas(old_part.mask, new_part.mask)
    #         and bool(np.any(old_mask))
    #         and bool(np.any(new_mask))
    #     )

    def load_parts(self, parts: list[Part]) -> None:
        frame_parts = [FramePart(part) for part in parts]
        self.current_frame = {
            frame_part.part_name: frame_part for frame_part in frame_parts
        }
        if self.first_frame:
            # Correct Frame
            self.first_frame = False

            # Save Last Frame
            self.last_frame = self.current_frame.copy()

            # Save Held Frame
            self.held_frame = self.current_frame.copy()

    def set_held_frame(self):
        dict_temp_held = {}
        for key, value in self.held_frame.items():
            if not self.held_frame.get(key):
                dict_temp_held[key] = value
                continue

            # If it's an Okay Size, Current is used, else use the last held one
            dict_temp_held[key] = (
                current_value
                if (current_value := self.current_frame.get(key))
                else value
            )

        self.held_frame.clear()
        self.held_frame = dict_temp_held

    def update_missing_parts_to_held_frame(self):
        for key, value in self.held_frame.items():
            if not self.current_frame.get(key):
                self.current_frame[key] = value

    def apply_part_persistence(self) -> None:
        """
        Holds the frame for the duration specified by frame_hold_amount.
        If the frame has been held for enough time, it will be updated.
        """
        # Counter Logic
        if self.frame_lag_counter >= self.part_frame_hold_seconds:
            self.set_held_frame()
            self.frame_lag_counter = 0
        else:
            self.update_missing_parts_to_held_frame()
            self.frame_lag_counter += 1

    # def apply_frame_stability(self) -> None:
    #     """
    #     Applies frame stability by ensuring that parts don't update for small differences.
    #     A significant difference (as determined by frame_difference_threshold) is required to update the part.
    #     """
    #     for part_name in self.current_frame.keys():
    #         # Ignore new parts that don't exist in the last frame
    #         if part_name not in self.last_frame:
    #             continue

    #         old_part = self.last_frame[part_name]
    #         this_part = self.current_frame[part_name]

    #         # Check if the difference between frames is large enough to update
    #         self.current_frame[part_name] = (
    #             this_part if self._compare_frames(old_part, this_part) else old_part
    #         )

    def save_frame(self):
        self.last_frame = self.current_frame.copy()

    def retrieve_parts(self):
        return [frame_part.part for frame_part in self.current_frame.values()]

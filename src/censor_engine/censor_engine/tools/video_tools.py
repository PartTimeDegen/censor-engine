from dataclasses import dataclass, field

import cv2

from censor_engine.censor_engine.tools.debugger import DebugLevels
from censor_engine.censor_engine.video import (
    FrameProcessor,
    VideoProcessor,
)
from censor_engine.detected_part import Part
from censor_engine.typing import Image


class InfoGenerator:
    default_offset: int = 2
    position: list[int] = [default_offset, default_offset]  # pixels
    longest_column_pxl: int = -1  # pixels

    _font: int = cv2.FONT_HERSHEY_TRIPLEX
    _font_scale: int | float = 0.5
    _thickness: int = 1

    def reset_position(self) -> None:
        InfoGenerator.position = [self.default_offset, self.default_offset]

    def generate_info(
        self,
        frame: Image,
        colour: tuple[int, int, int],
        info: list[str],
        group_title: str,
    ) -> Image:
        spacer = 0
        info = [group_title] + info
        for row in info:
            # Determine Text
            (text_width, text_height), baseline = cv2.getTextSize(
                row,
                self._font,
                self._font_scale,
                self._thickness,
            )

            # Change Position
            if text_width > self.longest_column_pxl:
                self.longest_column_pxl = text_width

            if (
                new_height := (self.position[1] + baseline + text_height)
            ) < frame.shape[0]:  # type: ignore
                InfoGenerator.position[1] = new_height
                spacer = baseline + text_height
            else:
                InfoGenerator.position[1] = self.default_offset
                InfoGenerator.position[0] = (
                    self.longest_column_pxl + self.default_offset
                )
                self.longest_column_pxl = -1

            # Put Text
            pos: tuple[int, int] = tuple(self.position)  # type: ignore

            # # Outline
            cv2.putText(
                frame,
                row,
                pos,
                self._font,
                self._font_scale,
                (0, 0, 0),
                self._thickness * 2,
                lineType=cv2.LINE_AA,
            )

            # # Text
            cv2.putText(
                frame,
                row,
                pos,
                self._font,
                self._font_scale,
                colour,
                self._thickness,
                lineType=cv2.LINE_AA,
            )

        InfoGenerator.position[1] = self.position[1] + spacer
        return frame


@dataclass(slots=True)
class VideoInfo:
    initial_frame: Image
    frame_count: int
    raw_parts: list[Part]
    video_processor: VideoProcessor
    frame_processor: FrameProcessor
    level: DebugLevels

    show_parts: bool = False  # UNUSED

    # Internals
    _frame_info: list[str] = field(init=False, default_factory=list)
    _part_info: list[str] = field(init=False, default_factory=list)

    def _get_counter(self, value: int, max_value: int) -> str:
        value_str = str(value)
        max_value_str = str(max_value)
        difference = len(max_value_str) - len(value_str)

        fixed_value = "0" * difference + value_str

        return f"{fixed_value}/{max_value_str}"

    def _get_frame_info(
        self, output_image, frame_processor: FrameProcessor
    ) -> Image:
        info = [
            f"Frame: {self._get_counter(self.frame_count, int(self.video_processor.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))}",
            f"FPS: {self.video_processor.get_fps()}",
            # f"Held Counter: {frame_processor.frame_lag_counter} / {frame_processor.part_frame_hold_frames}",
        ]
        return InfoGenerator().generate_info(
            output_image,
            (255, 255, 255),
            info,
            group_title="Frame Info",
        )

    def _get_part_dictionary(
        self, output_image, frame_processor: FrameProcessor
    ) -> Image:
        return InfoGenerator().generate_info(
            output_image,
            (0, 0, 255),
            [
                f"({self._get_counter(part.lifespan_frames, frame_processor.part_frame_hold_frames)}) {index} | {part} (approx%={part.part.part_area.approximate_percent_region:%})"
                for index, part in frame_processor.part_dictionary.items()
            ],
            group_title="Part Dictionary",
        )

    def _get_shown_part_info(self, output_image, frame_processor) -> Image:
        return InfoGenerator().generate_info(
            output_image,
            (255, 0, 0),
            [
                f"{value} ({key})"
                for key, value in frame_processor.current_frame.items()
            ],
            group_title="Shown Parts",
        )

    def _get_initial_part_info(
        self, output_image, frame_processor: FrameProcessor
    ) -> Image:
        return InfoGenerator().generate_info(
            output_image,
            (255, 0, 255),
            [
                f"{value} ({key})"
                for key, value in frame_processor.loaded_frame.items()
            ],
            group_title="Initially Found Parts",
        )

    def get_debug_info(
        self, output_image: Image, frame_processor: FrameProcessor
    ):
        # Frame Information
        output_image = self._get_frame_info(output_image, frame_processor)

        # Part's in held buffer
        output_image = self._get_part_dictionary(output_image, frame_processor)

        # Shown Parts on Video
        output_image = self._get_shown_part_info(output_image, frame_processor)

        # Initially Found Frame Parts
        output_image = self._get_initial_part_info(
            output_image, frame_processor
        )

        InfoGenerator().reset_position()
        return output_image

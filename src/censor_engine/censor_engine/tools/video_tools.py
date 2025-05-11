from dataclasses import dataclass, field
import cv2

from censor_engine.typing import CVImage
from censor_engine.censor_engine.video import (
    FrameProcessor,
    VideoProcessor,
)
from censor_engine.detected_part.base import Part
from censor_engine.censor_engine.tools.debugger import DebugLevels


class InfoGenerator:
    default_offset: int = 2
    position: list[int] = [default_offset, default_offset]  # pixels
    longest_column_pxl: int = -1  # pixels

    _font: int = cv2.FONT_HERSHEY_TRIPLEX
    _font_scale: int = 1
    _thickness: int = 2

    def reset_position(self) -> None:
        InfoGenerator.position = [self.default_offset, self.default_offset]

    def generate_info(
        self,
        frame: CVImage,
        colour: tuple[int, int, int],
        info: list[str],
    ) -> CVImage:
        spacer = 0
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
            cv2.putText(
                frame,
                row,
                pos,
                self._font,
                self._font_scale,
                colour,
                self._thickness,
            )

        InfoGenerator.position[1] = self.position[1] + spacer
        return frame


@dataclass(slots=True)
class VideoInfo:
    initial_frame: CVImage
    frame_count: int
    raw_parts: list[Part]
    video_processor: VideoProcessor
    frame_processor: FrameProcessor
    level: DebugLevels

    show_parts: bool = False  # UNUSED

    # Internals
    _frame_info: list[str] = field(init=False, default_factory=list)
    _part_info: list[str] = field(init=False, default_factory=list)
    _frame_info: list[str] = field(init=False, default_factory=list)

    def _get_frame_info(self, output_image) -> CVImage:
        info = [
            f"Frame: {self.frame_count} / {int(self.video_processor.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))}",
            f"FPS: {self.video_processor.get_fps()}",
        ]
        return InfoGenerator().generate_info(output_image, (0, 0, 0), info)

    def _get_part_info(self, output_image) -> CVImage:
        info = []

        for part in self.raw_parts:
            info.append(part.get_name_and_merged())

        return InfoGenerator().generate_info(output_image, (255, 0, 0), info)

    def get_debug_info(self, output_image: CVImage):
        output_image = self._get_frame_info(output_image)
        output_image = self._get_part_info(output_image)

        InfoGenerator().reset_position()
        return output_image

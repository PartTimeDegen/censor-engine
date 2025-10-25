from dataclasses import dataclass
from pathlib import Path

import cv2

from .input_image import (
    HORIZONTAL_ROWS,
    ImageGenerator,
)


@dataclass
class MotionTrack:
    duration: int  # seconds
    velocity: tuple[int, int]  # y, x

    current_frame_count: int = 0

    def calculate_location(
        self, current_pos: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            current_pos[0] + self.velocity[0] * self.current_frame_count,
            current_pos[1] + self.velocity[1] * self.current_frame_count,
        )

    def add_frame_count(self) -> None:
        self.current_frame_count += 1


@dataclass
class MotionInformation:
    part_name: str
    starting_position: tuple[str, int, str]  # PartName, Vertical, Horizontal
    motion_data: MotionTrack

    def is_finished(self, fps: int) -> bool:
        data = self.motion_data
        return data.current_frame_count > (data.duration * fps)

    def get_current_position(self, fps: int):
        coords = (
            self.starting_position[1],
            HORIZONTAL_ROWS[self.starting_position[2]],
        )
        if not self.is_finished(fps):
            self.motion_data.add_frame_count()

        return self.motion_data.calculate_location(coords)


class VideoGenerator(ImageGenerator):
    file_name: str = "input_video.mp4"
    frame_data: list[list] | None = None

    def __init__(self, input_path: Path, file_name: str):
        super().__init__(input_path)
        self.file_name = file_name
        self.frame_data = []

    def make_test_video(
        self,
        motion_information: list[MotionInformation],
        duration: int = 3,
        fps: int = 30,
        codec: str = "mp4v",
    ):
        total_frames = duration * fps

        # Make VideoWriter Object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(self.input_path / self.file_name),
            fourcc,
            fps,
            self.size,
        )

        # Iterate Frames
        for frame in range(total_frames):
            # Create Image Parts
            input_data = []
            for part_data in motion_information:
                # if part_data.is_finished(fps):
                #     continue

                vert, hori = part_data.get_current_position(fps)
                input_data += [(part_data.part_name, vert, hori)]

            self._create_parts(input_data)
            if self.frame_data is not None:
                self.frame_data.append(self.parts)

            frame_image = self.make_test_image()

            # Write Frames
            out.write(frame_image)

        out.release()

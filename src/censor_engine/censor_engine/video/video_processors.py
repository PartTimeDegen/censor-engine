from dataclasses import dataclass, field
import os

import cv2


@dataclass(slots=True)
class VideoProcessor:
    file_path: str
    new_file_name: str
    flags: dict[str, bool]

    _width: int = field(init=False)
    _height: int = field(init=False)
    _fps: int = field(init=False)
    total_frames: int = field(init=False)

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

    def get_fps(self):
        return self._fps

    def write_frame(self, file_output):
        self.video_writer.write(file_output)

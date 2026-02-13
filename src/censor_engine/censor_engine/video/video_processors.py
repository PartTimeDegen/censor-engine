import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from censor_engine.typing import Image

TEMP_AUDIO_NAME = "temp_audio.aac"
TEMP_VIDEO_NAME = "temp_video"


def handle_exit(signum, frame) -> None:  # noqa: ANN001, ARG001
    """
    This handles the exiting to ensure the video is closed properly.

    :param _type_ signum: _description_
    :param _type_ frame: _description_
    """
    print(  # noqa: T201
        "\n"
        "*** Stop requested — finishing current frame and muxing partial "
        "video. ***"
    )
    VideoProcessor.force_stop = True  # type: ignore


signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_exit)  # kill


@dataclass(slots=True)
class VideoProcessor:
    # Inputs
    file_path: str
    new_file_name: str
    flags: dict[str, bool]

    # Meta
    force_stop: bool = False
    _video_has_audio: bool = False

    # File Stuff
    _file_path: Path = field(init=False)
    _folder: Path = field(init=False)
    _ext: str = field(init=False)

    _temp_audio_path: Path = field(init=False)
    _temp_video_path: Path = field(init=False)
    _final_video_path: Path = field(init=False)

    # Video Information Stuff
    _width: int = field(init=False)
    _height: int = field(init=False)

    _fps: int = field(init=False)
    total_frames: int = field(init=False)

    # OpenCV Wranglers
    video_capture: cv2.VideoCapture = field(init=False)
    video_writer: cv2.VideoWriter = field(init=False)

    def __post_init__(self):
        # File Handling
        self._file_path = Path(self.file_path)
        self._final_video_path = Path(self.new_file_name)

        self._folder = self._final_video_path.parent
        self._ext = self._final_video_path.suffix

        self._temp_audio_path = self._folder / TEMP_AUDIO_NAME
        self._temp_video_path = self._folder / f"{TEMP_VIDEO_NAME}{self._ext}"

        # Get Video
        self.video_capture = cv2.VideoCapture(self.file_path)  # type: ignore

        # Check if Can Open Video File
        if not self.video_capture.isOpened():
            msg = f"Could not open video: {self.file_path}"
            raise ValueError(msg)

        # Get Video File Settings
        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(  # type: ignore # Not sure why this isn't working
            *self.__get_codec_from_extension(self.file_path),
        )

        # Create Writer
        self.video_writer = cv2.VideoWriter(
            str(self._temp_video_path),  # type: ignore
            fourcc,
            self._fps,
            (self._width, self._height),
        )

        # Extract Audio
        self.__extract_audio()

        # Save Total Frames for Reference
        self.total_frames = int(
            self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT),
        )

    def __get_codec_from_extension(self, filename: str) -> str:
        """
        This function is used to find the correct codec used by OpenCV.

        :param str filename: Name of the file, it will determine the extension
        """
        ext = Path(filename).suffix.lower()
        codec_mapping = {
            ".mp4": "mp4v",  # MPEG-4
            ".avi": "XVID",  # AVI format
            ".mov": "avc1",  # QuickTime
            ".mkv": "X264",  # Matroska
            ".webm": "VP80",  # WebM format
        }
        return codec_mapping.get(ext, "mp4v")  # Default to 'mp4v' if unknown

    def __extract_audio(self) -> None:
        ffmpeg = os.environ.get("FFMPEG_BINARY", "ffmpeg")

        def has_audio(path: Path) -> bool:
            p = subprocess.run(  # noqa: S603 # TODO: Check, should be okay
                [ffmpeg, "-i", str(path)],
                check=False,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                text=True,
            )
            return "Audio:" in p.stderr

        self._video_has_audio = has_audio(self._file_path)
        if not self._video_has_audio:
            return

        is_webm = self._file_path.suffix.lower() == ".webm"

        # Choose audio format
        if is_webm:
            self._temp_audio_path = self._folder / f"{TEMP_AUDIO_NAME}.opus"
            audio_cmd = ["-c:a", "copy"]  # Opus/Vorbis stays intact
        else:
            self._temp_audio_path = self._folder / f"{TEMP_AUDIO_NAME}.aac"
            audio_cmd = ["-c:a", "aac", "-b:a", "128k"]

        subprocess.run(  # noqa: S603 # TODO: Check, should be okay
            [
                ffmpeg,
                "-y",
                "-i",
                str(self._file_path),
                "-vn",
                *audio_cmd,
                str(self._temp_audio_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def __mux_audio_and_video(self) -> None:
        ffmpeg = os.environ.get("FFMPEG_BINARY", "ffmpeg")
        is_webm = self._final_video_path.suffix.lower() == ".webm"

        if self._final_video_path.exists():
            self._final_video_path.unlink()

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(self._temp_video_path),
        ]

        if self._video_has_audio:
            cmd += ["-i", str(self._temp_audio_path)]

        # Video
        if is_webm:
            cmd += ["-c:v", "libvpx", "-tag:v", "VP80"]
        else:
            cmd += ["-c:v", "copy"]

        # Audio
        if self._video_has_audio:
            if is_webm:
                cmd += ["-c:a", "libopus"]
            else:
                cmd += ["-c:a", "copy"]

            cmd += ["-shortest"]

        cmd.append(str(self._final_video_path))

        subprocess.run(  # noqa: S603 # TODO: Check, should be okay
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def get_fps(self):
        return self._fps

    def write_frame(self, file_output: Image) -> None:
        """
        This writes the current frame to the temp video file.

        :param Image file_output: Image of frame
        """
        self.video_writer.write(file_output)

    def close_video(self) -> None:
        self.video_capture.release()
        self.video_writer.release()
        self.__mux_audio_and_video()

        if self._temp_video_path.exists():
            self._temp_video_path.unlink()
        if self._video_has_audio and self._temp_audio_path.exists():
            self._temp_audio_path.unlink()

import argparse
from dataclasses import dataclass, field
from glob import glob
import os

import cv2

import progressbar

from censorengine.backend.constants.files import (
    APPROVED_FORMATS_IMAGE,
    APPROVED_FORMATS_VIDEO,
)
from censorengine.backend.models.censor_manager import CensorManager
from censorengine.backend.models.config import Config

from censorengine.libs.detector_library.catalogue import (
    enabled_detectors,
    enabled_determiners,
)
from censorengine.libs.shape_library.catalogue import shape_catalogue
from censorengine.libs.style_library.catalogue import style_catalogue


@dataclass
class CensorEngine:
    # Booleans
    test_mode: bool = field(init=False)
    config: Config = field(init=False)
    censor_mode: str = "image"
    used_boot_config: bool = False

    # Information
    main_files_path: str = field(init=False)
    full_files_path: str = field(init=False)
    files: list[str] = field(default_factory=list)
    indexed_files: list[str] = field(default_factory=list)

    # Files
    is_file: bool = False
    force_png: bool = False
    # Statistics
    max_file_index: int = 1
    max_index_chars: int = 1

    durations: list[str] = field(default_factory=list)

    def __init__(
        self,
        main_file_path: str,
        censor_mode: str = "video",  # image, video, default=auto
        config: str = "00_default.yml",
        test_mode: bool = False,
    ):
        # Mount Init Settings
        self.test_mode = test_mode
        self.main_files_path = main_file_path
        self.censor_mode = censor_mode

        # Get Pre-init Arguments
        if not test_mode:
            self._get_pre_init_arguments()

        # Load Config
        if not self.used_boot_config:
            self.load_config(config)

        # Get Post-init Arguments
        if not test_mode:
            self._get_post_init_arguments()

    def _make_progress_bar_widgets(self, total_amount: int):
        return [
            progressbar.Counter(),
            "/",
            f"{total_amount}" " | ",
            progressbar.Percentage(),
            " [",
            progressbar.Timer(),
            "] ",
            "(",
            progressbar.ETA(),
            ") ",
            progressbar.GranularBar(),
        ]

    def _find_files(self):
        self.files = []
        self.indexed_files = []

        self.full_files_path = os.path.join(
            self.main_files_path,
            self.config.uncensored_folder,
        )
        approved_formats = APPROVED_FORMATS_IMAGE + APPROVED_FORMATS_VIDEO
        if any(self.full_files_path.endswith(ext) for ext in approved_formats):
            if not os.path.exists(self.full_files_path):
                raise FileNotFoundError
            self.is_file = True
            self.files = [self.full_files_path]
            self.indexed_files = [(1, self.full_files_path)]
            return

        files = [
            filename
            for filename in glob(
                os.path.join(self.full_files_path, "**/**"),
                recursive=True,
            )
            if not (filename.endswith("/") or filename.endswith("\\"))
            and filename[filename.rfind(".") :] in approved_formats
        ]

        self.files = files
        self.indexed_files = [
            (file_index, file_name)
            for file_index, file_name in enumerate(files, start=1)
        ]
        if len(files) == 0:
            raise FileNotFoundError(f"Empty Folder: {self.full_files_path}")

        self.max_file_index = self.indexed_files[-1][0]
        self.max_index_chars = len(str(self.max_file_index))

    def _get_index_text(self, index):
        index_percent = index / self.max_file_index
        leading_spaces = " " * (self.max_index_chars - len(str(index)))
        leading_spaces_pc = " " * (3 - len(str(int(100 * index_percent))))

        return f"{leading_spaces}{index}/{self.max_file_index} ({leading_spaces_pc}{index_percent:0.1%})"

    def _image_pipeline(self):
        for index, file_path in self.indexed_files:
            index_text = self._get_index_text(index)

            print(f'{index_text}Censoring: "{file_path.split(os.sep)[-1]}"')
            file_image = cv2.imread(file_path)
            censor_manager = CensorManager(
                file_image=file_image,
                config=self.config,
            )
            censor_manager.start()
            file_output = censor_manager.return_output()

            self.force_png = censor_manager.force_png

            new_file_name = self._save_file(file_path)

            # File Save
            cv2.imwrite(new_file_name, file_output)
            print(f"- Written to:\t{new_file_name}")

    def _video_pipeline(self):
        def get_codec_from_extension(filename: str):
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

        for index, file_path in self.indexed_files:
            # Get Index
            index_text = self._get_index_text(index)
            print(f'{index_text}Censoring: "{file_path.split(os.sep)[-1]}"')

            # Get Video Capture
            video_capture = cv2.VideoCapture(file_path)
            if not video_capture.isOpened():
                raise ValueError("Could not open video")

            # Get video properties
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))

            # Get Video Writer Objects
            fourcc = cv2.VideoWriter_fourcc(*get_codec_from_extension(file_path))
            new_file_name = self._save_file(file_path)
            out = cv2.VideoWriter(new_file_name, fourcc, fps, (width, height))

            # Progress Bar Stuff
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Iterate through Frames
            for frame in progressbar.progressbar(
                range(total_frames),
                widgets=self._make_progress_bar_widgets(total_frames),
            ):
                # Check Frames
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Run Censor Manager
                censor_manager = CensorManager(
                    file_image=frame,
                    config=self.config,
                )
                censor_manager.start()

                # Save Output
                file_output = censor_manager.return_output()
                out.write(file_output)

            # Spacer
            print()

    def _get_pre_init_arguments(self):
        # Parser
        parser = argparse.ArgumentParser(
            prog="CensorEngine",
            description="Censors Images",
        )

        # Add Args
        parser.add_argument("-loc", action="store")

        parser.add_argument("-config", action="store")

        parser.add_argument("-dev", action="store")  # Debug
        parser.add_argument("-output", action="store")  # TODO: debug, true false

        self.args = parser.parse_args()

        # Handle Args
        if self.args.config:
            self.load_config(self.args.config)
            self.used_boot_config = True

        if self.args.dev:
            # TODO: This needs a dev handler to handle non-boolean values
            self.config.debug_mode(True)

    def _get_post_init_arguments(self):
        if self.args.loc:
            self.config.uncensored_folder = self.args.loc

    def _save_file(self, file_name):
        # Get Name
        full_path, ext = os.path.splitext(file_name)

        # Add Word Fixes
        list_path = full_path.split(os.sep)

        fixed_file_list = [
            word
            for word in [
                self.config.file_prefix,
                list_path[-1],
                self.config.file_suffix,
            ]
            if word != ""
        ]
        new_file_name = "_".join(fixed_file_list)

        # Get New Location
        loc_base = str(self.main_files_path)
        folders_loc_base = len(loc_base.split(os.sep))

        base_folder = list_path[:folders_loc_base]
        new_folder = list_path[folders_loc_base:]

        censored_folder = self.config.censored_folder.split(os.sep)
        new_folder = censored_folder + new_folder[len(censored_folder) :]

        new_folder.pop()

        new_folder = base_folder + new_folder
        new_folder = os.sep.join(new_folder)

        os.makedirs(new_folder, exist_ok=True)
        file_path_new = os.sep.join([new_folder, new_file_name])

        if self.force_png:
            ext = ".png"

        # File Proper
        return f"{file_path_new}{ext}"

    def load_config(self, config_path):
        self.config = Config(self.main_files_path, config_path)

    def start(self):
        # Find Files
        self._find_files()

        # What to Censor
        if self.censor_mode == "image":
            self._image_pipeline()
        elif self.censor_mode == "video":
            self._video_pipeline()
        else:
            self._image_pipeline()
            self._video_pipeline()

    # Reporting
    def get_detectors(self):
        return [detector.model_name for detector in enabled_detectors]

    def get_determiners(self):
        return [detector.model_name for detector in enabled_determiners]

    def get_shapes(self):
        return list(shape_catalogue.keys())

    def get_censor_styles(self):
        return list(style_catalogue.keys())

    # Sample
    def get_parts(self):
        files = []
        for index, file_path in self.indexed_files:
            index_text = self._get_index_text(index)

            censor_manager = CensorManager(
                file_image=file_path,
                config=self.config,
                index_text=index_text,
            )

            files.append(
                {"file_path": file_path, "parts_found": censor_manager.detected_parts}
            )
        return files

    def get_determinations(self):
        files = []
        for index, file_path in self.indexed_files:
            index_text = self._get_index_text(index)

            censor_manager = CensorManager(
                file_image=file_path,
                config=self.config,
                index_text=index_text,
            )

            files.append(
                {
                    "file_path": file_path,
                    "determined_features": censor_manager.extracted_information,
                }
            )
        return files

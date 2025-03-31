import argparse
from dataclasses import dataclass, field
from glob import glob
import os
from typing import Any

import cv2

import progressbar

import censorengine
from censorengine.backend.constants.files import (
    APPROVED_FORMATS_IMAGE,
    APPROVED_FORMATS_VIDEO,
)
from censorengine.backend.models.image import ImageProcessor
from censorengine.backend.models.config import Config

from censorengine.backend.models.debugger import DebugLevels
from censorengine.libs.detector_library.catalogue import (
    enabled_detectors,
    enabled_determiners,
)
from censorengine.libs.shape_library.catalogue import shape_catalogue
from censorengine.libs.style_library.catalogue import style_catalogue
from censorengine.backend.models.video import VideoFrame, VideoProcessor


@dataclass
class CensorEngine:
    # Information
    main_files_path: str
    test_mode: bool = False
    censor_mode: str = "auto"
    config_data: str | dict[str, Any] = "00_default.yml"

    # Internal State Variables
    full_files_path: str = field(init=False)
    config: Config = field(init=False)

    used_boot_config: bool = field(default=False, init=False)
    is_file: bool = field(default=False, init=False)
    force_png: bool = field(default=False, init=False)

    max_file_index: int = field(default=1, init=False)
    max_index_chars: int = field(default=1, init=False)

    durations: list[str] = field(default_factory=list, init=False)
    files: list[str] = field(default_factory=list, init=False)
    indexed_files: list[tuple[int, str, str]] = field(default_factory=list, init=False)

    # Constants
    FRAME_DIFFERENCE_THRESHOLD: float = 0.02  # Percentage # TODO: Add to config
    FRAME_HOLD = 3

    def __post_init__(self):
        # Get Pre-init Arguments
        if not self.test_mode:
            self._parse_pre_arguments()

        # Load Config
        if not self.used_boot_config:
            self.load_config(self.config_data)

        # Get Post-init Arguments
        if not self.test_mode:
            self._get_post_arguments()

    def _make_progress_bar_widgets(
        self, index_text: str, file_name: str, total_amount: int
    ) -> list:
        return [
            f"{index_text} ",
            f'Censoring "{file_name}" > ',
            progressbar.Counter(),
            "/",
            f"{total_amount}" " |",
            progressbar.Percentage(),
            " [",
            progressbar.Timer(),
            "] ",
            "(",
            progressbar.ETA(),
            ") ",
            progressbar.GranularBar(),
        ]

    def _find_files(self):  # FIXME Single File Mode
        # Clear Lists
        self.files.clear()
        self.indexed_files.clear()

        # Get Full Path
        self.full_files_path = os.path.join(
            self.main_files_path,
            self.config.file_settings.uncensored_folder,
        )

        # Check and Filter for Approved File Formats Only
        approved_formats = APPROVED_FORMATS_IMAGE + APPROVED_FORMATS_VIDEO
        if any(self.full_files_path.endswith(ext) for ext in approved_formats):
            if not os.path.exists(self.full_files_path):
                raise FileNotFoundError(f"File not found: {self.full_files_path}")

            self.is_file = True
            self.files = [self.full_files_path]
            self.indexed_files = [(1, self.full_files_path, "placeholder")]
            return

        # Scan Folders
        self.files = [
            file_name
            for file_name in glob(
                os.path.join(self.full_files_path, "**", "*"), recursive=True
            )
            if os.path.isfile(file_name)
            and file_name[file_name.rfind(".") :] in approved_formats
        ]

        self.indexed_files = [
            (
                index,
                file_name,
                "video"
                if file_name.endswith(tuple(APPROVED_FORMATS_VIDEO))
                else "image",
            )
            for index, file_name in enumerate(self.files, start=1)
        ]

        # Throw Error on Empty Folder
        if not self.files:
            raise FileNotFoundError(f"Empty folder: {self.full_files_path}")

        # Handle Max Index
        self.max_file_index = len(self.indexed_files)
        self.max_index_chars = len(str(self.max_file_index))

    def _get_index_text(self, index: int):
        index_percent = index / self.max_file_index
        leading_spaces = " " * (self.max_index_chars - len(str(index)))
        leading_spaces_pc = " " * (3 - len(str(int(100 * index_percent))))

        return f"{leading_spaces}{index}/{self.max_file_index} ({leading_spaces_pc}{index_percent:0.1%})"

    def _image_pipeline(self):
        if not [f for f in self.indexed_files if f[-1] == "image"]:
            return
        for index, file_path, file_type in self.indexed_files:
            # Check it's an Image
            if file_type != "image":
                continue

            # Print that it's Censoring
            index_text = self._get_index_text(index)

            # Read the File
            file_image = cv2.imread(file_path)

            # Run the Censor Manager
            censor_manager = ImageProcessor(
                file_image=file_image,
                config=self.config,
            )
            censor_manager.start()

            # Output File Handling
            file_output = censor_manager.return_output()
            self.force_png = censor_manager.force_png
            new_file_name = self._save_file(file_path)

            # File Save
            cv2.imwrite(new_file_name, file_output)
            print(f'{index_text} Censored: "{file_path.split(os.sep)[-1]}"')

        print("Finished Censoring Images!")

    def _video_pipeline(self):
        for index, file_path, file_type in self.indexed_files:
            # Check it's an Image
            if file_type != "video":
                continue

            # Get Video Capture
            video_processor = VideoProcessor(
                file_path,
                self._save_file(file_path),
            )

            # Iterate through Frames
            frame_processor = VideoFrame(
                frame_difference_threshold=self.config.video_settings.frame_difference_threshold,
                frame_hold_amount=self.config.video_settings.part_frame_hold,
                size_change_tolerance=self.config.video_settings.size_change_tolerance,
            )
            progress_bar = progressbar.progressbar(
                range(video_processor.total_frames),
                widgets=self._make_progress_bar_widgets(
                    index_text=self._get_index_text(index),
                    file_name=file_path.split(os.sep)[-1],
                    total_amount=video_processor.total_frames,
                ),
            )
            for _ in progress_bar:
                # Check Frames
                ret, frame = video_processor.video_capture.read()
                if not ret:
                    break

                # Run Censor Manager
                censor_manager = ImageProcessor(file_image=frame, config=self.config)
                censor_manager.generate_parts_and_shapes()

                # # Apply Stability Stuff
                """
                NOTE:   This section is used to make videos more stable,
                        currently the processing effects performed are:

                            -   Holding frames for a certain number of frames
                                to avoid issues where a part doesn't get
                                detected, thus causing a flickering effect.

                            -   Maintaining the last frame instead of the 
                                current if the difference is negligible, this
                                avoids issues where the the detected areas are
                                slightly different thus causes the censors to
                                "spasm".

                """
                frame_processor.load_parts(censor_manager.parts)
                # frame_processor.apply_part_persistence()
                # frame_processor.apply_part_size_correction()
                # frame_processor.apply_frame_stability()
                # frame_processor.save_frame()

                # Update the Parts
                censor_manager.parts = list(frame_processor.current_frame.values())

                # Apply Censors
                censor_manager.compile_masks()
                censor_manager.apply_censors()

                # Save Output
                file_output = censor_manager.return_output()

                video_processor.video_writer.write(file_output)

            # Spacer
            print()

    def _parse_pre_arguments(self):
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
            self.config.dev_settings.debug_level = DebugLevels.BASIC

    def _get_post_arguments(self):
        if self.args.loc:
            self.config.file_settings.uncensored_folder = self.args.loc

    def _save_file(self, file_name):
        # Get Name
        full_path, ext = os.path.splitext(file_name)

        # Add Word Fixes
        list_path = full_path.split(os.sep)

        fixed_file_list = [
            word
            for word in [
                self.config.file_settings.file_prefix,
                list_path[-1],
                self.config.file_settings.file_suffix,
            ]
            if word != ""
        ]
        new_file_name = "_".join(fixed_file_list)

        # Get New Location
        loc_base = str(self.main_files_path)
        folders_loc_base = len(loc_base.split(os.sep))

        base_folder = list_path[:folders_loc_base]
        new_folder = list_path[folders_loc_base:]

        censored_folder = self.config.file_settings.censored_folder.split(os.sep)
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

    def load_config(self, config_data: str | dict[str, Any]):
        if isinstance(config_data, str):
            self.config = Config.from_yaml(self.main_files_path, config_data)
        elif isinstance(config_data, dict):
            self.config = Config.from_dictionary(config_data)
        else:
            raise TypeError("invalid type used")

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

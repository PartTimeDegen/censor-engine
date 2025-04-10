import argparse
from dataclasses import dataclass, field
from glob import glob
import os
import statistics
from typing import Any

import cv2

from cycler import V
import progressbar

from censorengine.backend.constants.files import (
    APPROVED_FORMATS_IMAGE,
    APPROVED_FORMATS_VIDEO,
)
from censorengine.backend.models.pipelines.image import ImageProcessor
from censorengine.backend.models.config import Config

from censorengine.backend.models.tools.debugger import DebugLevels
from censorengine.backend.models.tools.dev_tools import DevTools
from censorengine.libs.detector_library.catalogue import (
    enabled_detectors,
    enabled_determiners,
)
from censorengine.libs.shape_library.catalogue import shape_catalogue
from censorengine.libs.style_library.catalogue import style_catalogue
from censorengine.backend.models.pipelines.video import VideoFrame, VideoProcessor


@dataclass(slots=True, repr=False, eq=False, order=False, match_args=False)
class CensorEngine:
    # Information
    main_files_path: str
    test_mode: bool = False
    censor_mode: str = "auto"
    config_data: str | dict[str, Any] = "00_default.yml"

    # Debug
    _show_stats: bool = False
    _debug_level: DebugLevels = DebugLevels.NONE
    _time_durations: list[float] = field(init=False, default_factory=list)

    # Dev Tools
    _dev_tools: DevTools | None = field(init=False, default=None)

    # Flags
    _flags: dict[str, bool] = field(init=False, default_factory=dict)

    # Internal State Variables
    _arg_loc: str = field(init=False)
    _full_files_path: str = field(init=False)
    _config: Config = field(init=False)

    _used_boot_config: bool = field(default=False, init=False)
    _is_file: bool = field(default=False, init=False)
    _force_png: bool = field(default=False, init=False)

    _max_file_index: int = field(default=1, init=False)
    _max_index_chars: int = field(default=1, init=False)

    _durations: list[str] = field(default_factory=list, init=False)
    _files: list[str] = field(default_factory=list, init=False)
    _indexed_files: list[tuple[int, str, str]] = field(default_factory=list, init=False)

    # Constants
    FRAME_DIFFERENCE_THRESHOLD: float = 0.02  # Percentage # TODO: Add to config
    FRAME_HOLD = 3

    def __post_init__(self):
        # Get Pre-init Arguments
        if not self.test_mode:
            self._parse_pre_arguments()

        # Load Config
        if not self._used_boot_config:
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
        self._files.clear()
        self._indexed_files.clear()

        # Get Full Path
        self._full_files_path = os.path.join(
            self.main_files_path,
            self._config.file_settings.uncensored_folder,
        )

        # Check and Filter for Approved File Formats Only
        approved_formats = APPROVED_FORMATS_IMAGE + APPROVED_FORMATS_VIDEO
        if any(self._full_files_path.endswith(ext) for ext in approved_formats):
            if not os.path.exists(self._full_files_path):
                raise FileNotFoundError(f"File not found: {self._full_files_path}")

            self._is_file = True
            self._files = [self._full_files_path]

            self._indexed_files = [
                (
                    1,
                    self._full_files_path,
                    "video"
                    if self._full_files_path.endswith(tuple(APPROVED_FORMATS_VIDEO))
                    else "image",
                )
            ]
            return

        # Scan Folders
        self._files = [
            file_name
            for file_name in glob(
                os.path.join(self._full_files_path, "**", "*"), recursive=True
            )
            if os.path.isfile(file_name)
            and file_name[file_name.rfind(".") :] in approved_formats
        ]

        self._indexed_files = [
            (
                index,
                file_name,
                "video"
                if file_name.endswith(tuple(APPROVED_FORMATS_VIDEO))
                else "image",
            )
            for index, file_name in enumerate(self._files, start=1)
        ]

        # Throw Error on Empty Folder
        if not self._files:
            raise FileNotFoundError(f"Empty folder: {self._full_files_path}")

        # Handle Max Index
        self._max_file_index = len(self._indexed_files)
        self._max_index_chars = len(str(self._max_file_index))

    def _get_index_text(self, index: int):
        index_percent = index / self._max_file_index
        leading_spaces = " " * (self._max_index_chars - len(str(index)))
        leading_spaces_pc = " " * (3 - len(str(int(100 * index_percent))))

        return f"{leading_spaces}{index}/{self._max_file_index} ({leading_spaces_pc}{index_percent:0.1%})"

    def _image_pipeline(self):
        if not [f for f in self._indexed_files if f[-1] == "image"]:
            return
        for index, file_path, file_type in self._indexed_files:
            # Check it's an Image
            if file_type != "image":
                continue

            # Print that it's Censoring
            index_text = self._get_index_text(index)

            # Dev Tools
            if self._flags["dev_tools"]:
                self._dev_tools = DevTools(
                    output_folder=file_path,
                    main_files_path=self.main_files_path,
                    using_full_output_path=self._flags["show_full_output_path"],
                )

            # Read the File
            file_image = cv2.imread(file_path)

            # Run the Censor Manager
            censor_manager = ImageProcessor(
                file_image=file_image,
                config=self._config,
                debug_level=self._debug_level,
                dev_tools=self._dev_tools,
            )
            censor_manager.start()

            # Dev Tools
            if self._dev_tools:
                self._dev_tools.dev_decompile_masks(
                    censor_manager.parts,
                    subfolder="zz_complete",
                )

            # Output File Handling
            file_output = censor_manager.return_output()
            self._force_png = censor_manager.force_png
            new_file_name = self._save_file(file_path)

            # File Save
            cv2.imwrite(new_file_name, file_output)

            # Print Out
            prefix = ""
            if not self._flags["show_full_output_path"]:
                prefix = "./"
                new_file_name = new_file_name.replace(self.main_files_path, "", 1)[1:]
            print(f'{index_text} Censored: "{prefix}{new_file_name}"')

            # Save Duration
            self._time_durations.append(censor_manager.get_duration())
            if self._flags["pad_individual_items"]:
                print()

        print("Finished Censoring Images!")

    def _video_pipeline(self):
        for index, file_path, file_type in self._indexed_files:
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
                frame_difference_threshold=self._config.video_settings.frame_difference_threshold,
                frame_hold_amount=self._config.video_settings.part_frame_hold,
                size_change_tolerance=self._config.video_settings.size_change_tolerance,
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
                censor_manager = ImageProcessor(file_image=frame, config=self._config)
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
                """
                -   Keep parts (hold them, if -1, always hold)
                -   check sizes for parts, flag any bad ones
                -   replace them with the held part
                -   if the held part is bad, update it to a better one (biggest?)
                -   
                """
                frame_processor.apply_part_persistence()
                # frame_processor.apply_part_size_correction()
                # frame_processor.apply_frame_stability()
                # frame_processor.save_frame()

                # Update the Parts
                censor_manager.parts = frame_processor.retrieve_parts()

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
        arg_mapper = {
            "uncensored_location": "loc",
            "config_location": "config",
            "debug_level": "debug",
        }

        flag_mapper = {
            "show_stat_metrics": "sm",
            "pad_individual_items": "pi",
            "dev_tools": "dt",
            "show_full_output_path": "fo",
        }

        # Add Args
        for value in arg_mapper.values():
            parser.add_argument(f"--{value}", action="store")

        # Add Flags
        for long_flag_name, short_flag_name in flag_mapper.items():
            parser.add_argument(
                f"-{short_flag_name}",
                f"--{long_flag_name.replace('_', '-')}",
                dest=long_flag_name,
                action="store_true",
                help=f"Enable {long_flag_name.replace('_', ' ')}",
            )

        # Collect Args
        args = parser.parse_args()

        # Handle Args
        if loc := args.loc:
            self._arg_loc = loc
        if config := args.config:
            self.load_config(config)
            self._used_boot_config = True

        if debug_word := args.debug:
            # TODO: This needs a dev handler to handle non-boolean values
            try:
                self._debug_level = DebugLevels[debug_word.upper()]
                print(f"**Using Debug Mode: {self._debug_level.name}**")
            except ValueError:
                raise ValueError(f"Invalid DebugLevels value: {str(debug_word)}")

        # Handle Handle Flags
        self._flags = {key: getattr(args, key) for key in flag_mapper}
        for key, value in self._flags.items():
            if value:
                print(f"**{key.replace('_', ' ').title()} Activated!**")

    def _get_post_arguments(self):
        if loc := self._arg_loc:
            if loc.startswith("./"):
                loc = self._config.file_settings.uncensored_folder + loc[1:]
            self._config.file_settings.uncensored_folder = loc

    def _save_file(self, file_name):
        # Get Name
        full_path, ext = os.path.splitext(file_name)

        # Add Word Fixes
        list_path = full_path.split(os.sep)

        fixed_file_list = [
            word
            for word in [
                self._config.file_settings.file_prefix,
                list_path[-1],
                self._config.file_settings.file_suffix,
            ]
            if word != ""
        ]
        new_file_name = "_".join(fixed_file_list)

        # Get New Location
        loc_base = str(self.main_files_path)
        folders_loc_base = len(loc_base.split(os.sep))

        base_folder = list_path[:folders_loc_base]
        new_folder = list_path[folders_loc_base:]

        censored_folder = self._config.file_settings.censored_folder.split(os.sep)
        new_folder = censored_folder + new_folder[len(censored_folder) :]

        new_folder.pop()

        new_folder = base_folder + new_folder
        new_folder = os.sep.join(new_folder)

        os.makedirs(new_folder, exist_ok=True)
        file_path_new = os.sep.join([new_folder, new_file_name])

        if self._force_png:
            ext = ".png"

        # File Proper
        return f"{file_path_new}{ext}"

    def load_config(self, config_data: str | dict[str, Any]):
        if isinstance(config_data, str):
            self._config = Config.from_yaml(self.main_files_path, config_data)
        elif isinstance(config_data, dict):
            self._config = Config.from_dictionary(config_data)
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
        if self._show_stats and len(self._time_durations) != 0:
            self.display_bulk_stats()

    # Reporting
    def get_detectors(self):
        return [detector.model_name for detector in enabled_detectors]

    def get_determiners(self):
        return [detector.model_name for detector in enabled_determiners]

    def get_shapes(self):
        return list(shape_catalogue.keys())

    def get_censor_styles(self):
        return list(style_catalogue.keys())

    def display_bulk_stats(self):
        times = self._time_durations
        mean = statistics.mean(times)

        dict_stats = {
            "Mean": mean,
            "Median": statistics.median(times),
            "Min": min(times),
            "Max": max(times),
            "Range": max(times) - min(times),
        }
        if len(times) > 1:
            stdev = statistics.stdev(times)
            coefficient_of_variation = stdev / mean
            dict_stats["Stdev"] = stdev
            dict_stats["CoV"] = coefficient_of_variation

        max_key_length = max(len(key) for key in dict_stats) + 4
        print()
        print("Run Statistics:")
        for key, value in dict_stats.items():
            if key != "CoV":
                print(f"- {key:<{max_key_length}}: {value*1000:>6.3f} ms")
            else:
                print(f"- {key:<{max_key_length}}: {value:>2.3%}")

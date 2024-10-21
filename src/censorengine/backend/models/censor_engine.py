import argparse
from dataclasses import dataclass, field
from glob import glob
import os

import cv2
import yaml

from censorengine.backend.constants.files import APPROVED_FORMATS_IMAGE
from censorengine.backend.constants.typing import Config
from censorengine.backend.models.censor_manager import CensorManager
from censorengine.backend.constants.files import CONFIGS_FOLDER


@dataclass
class CensorEngine:
    # Booleans
    test_mode: bool = field(init=False)
    config: Config = field(init=False)

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

    # Debug Stuff
    show_duration: bool = field(init=False)
    debug_mode: int = field(init=False)
    debug_log_time: bool = field(init=False)

    def __init__(
        self,
        main_file_path: str,
        censor_mode: str = "image",  # image, video, gif, default=auto
        config: str = "00_default.yml",
        show_duration: bool = False,
        test_mode: bool = False,
        debug_mode: int = 0,
        debug_log_time: bool = False,
    ):
        self.test_mode = test_mode
        self.main_files_path = main_file_path

        self.show_duration = show_duration
        self.debug_mode = debug_mode
        self.debug_log_time = debug_log_time

        # Load Config
        self._load_config(config)

        # Get Arguments
        self._get_arguments()

        # Find Files
        self._find_files()

        # What to Censor
        if censor_mode == "image":
            self._image_pipeline()
        elif censor_mode == "video":
            self._video_pipeline()
        else:
            self._image_pipeline()
            self._video_pipeline()

    def _find_files(self):
        self.files = []
        self.indexed_files = []

        self.full_files_path = os.path.join(
            self.main_files_path,
            self.config["uncensored_folder"],
        )

        if any(
            self.full_files_path.endswith(ext)
            for ext in APPROVED_FORMATS_IMAGE
        ):
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
            and filename[filename.rfind(".") :] in APPROVED_FORMATS_IMAGE
        ]

        self.files = files
        self.indexed_files = [
            (file_index, file_name)
            for file_index, file_name in enumerate(files, start=1)
        ]
        self.max_file_index = files[-1][0]
        self.max_index_chars = len(str(self.max_file_index))

    def _image_pipeline(self):
        for index, file_path in self.indexed_files:
            file_output = CensorManager(
                file_path,
                config=self.config,
                debug_level=0,
                show_duration=True,
                debug_log_time=True,
            ).return_output()
            self._save_file(file_output)

    def _video_pipeline(self):
        pass

    def _load_config(self, config_path):

        full_config_path = os.path.join(
            CONFIGS_FOLDER,
            config_path,
        )

        with open(full_config_path) as stream:
            try:
                # Check if Can Access Data
                config_data = yaml.safe_load(stream)

                # Reset Variable
                self.config = {}

                # Update Variable
                self.config.update(**config_data)

            except yaml.YAMLError as exc:
                raise ValueError(exc)

        return full_config_path

    def _get_arguments(self):
        # Parser
        parser = argparse.ArgumentParser(
            prog="CensorEngine",
            description="Censors Images",
        )

        # Add Args
        parser.add_argument("-loc", action="store")

        parser.add_argument("-config", action="store")

        parser.add_argument("-dev", action="store")  # Debug, Profile
        parser.add_argument(
            "-output", action="store"
        )  # TODO: Quiet, Slim, Verbose, Very Verbose (vv)

        args = parser.parse_args()

        # Handle Args
        if args.config:
            self._load_config(args.config)

        if args.dev:
            self.config["debug_mode"] = (
                True  # TODO: This needs a dev handler to handle non-boolean values
            )

        if args.loc:
            self.config.update({"uncensored_folder": args.loc})

    def _save_file(self, file_output):
        # Get Name
        full_path, ext = os.path.splitext(self.full_files_path)

        # Add Word Fixes
        list_path = full_path.split(os.sep)

        fixed_file_list = [
            word
            for word in [
                self.config["file_prefix"],
                list_path[-1],
                self.config["file_suffix"],
            ]
            if word != ""
        ]
        new_file_name = "_".join(fixed_file_list)

        # Get New Location
        loc_base = str(self.main_files_path)
        folders_loc_base = len(loc_base.split(os.sep))

        new_folder = list_path[folders_loc_base:]
        base_folder = list_path[:folders_loc_base]
        new_folder[0] = self.config["censored_folder"]
        new_folder.pop()

        new_folder = base_folder + new_folder
        new_folder = os.sep.join(new_folder)

        os.makedirs(new_folder, exist_ok=True)
        file_path_new = os.sep.join([new_folder, new_file_name])

        if self.force_png:
            ext = ".png"

        # File Proper
        cv2.imwrite(f"{file_path_new}{ext}", file_output)

        print(f"- Written to:\t{file_path_new}{ext}")

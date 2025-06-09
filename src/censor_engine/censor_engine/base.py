import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from censor_engine.paths import PathManager
from censor_engine.typing import CVImage

from .mixin_pipeline_image import MixinImagePipeline
from .mixin_pipeline_video import MixinVideoPipeline
from .mixin_reporting import MixinReporting
from .mixin_utils import MixinUtils

from censor_engine.models.config import Config

from .tools.debugger import DebugLevels
from .tools.dev_tools import DevTools


@dataclass(slots=True, repr=False, eq=False, order=False)
class CensorEngine(
    MixinImagePipeline,
    MixinVideoPipeline,
    MixinReporting,
    MixinUtils,
):
    # Information
    base_folder: Path | str
    test_mode: bool = False
    censor_mode: str = "auto"
    config_data: str | dict[str, Any] = "00_default.yml"

    # Debug & Dev Tools
    _debug_level: DebugLevels = DebugLevels.NONE
    _time_durations: list[float] = field(init=False, default_factory=list)
    _dev_tools: DevTools | None = field(init=False, default=None)

    # Flags
    _flags: dict[str, bool] = field(init=False, default_factory=dict)

    # Internal State Variables
    _arg_loc: Path | None = field(init=False, default=None)
    _full_files_path: str = field(init=False)
    _config: Config = field(init=False)
    _used_boot_config: bool = field(default=False, init=False)
    _durations: list[str] = field(default_factory=list, init=False)
    _path_manager: PathManager = field(init=False)

    def __post_init__(self):
        # Conversions
        self.base_folder = Path(self.base_folder)

        # Get Pre-init Arguments
        if not self.test_mode:
            self._parse_pre_arguments()

        # Load Config
        if not self._used_boot_config:
            self.load_config(self.config_data)

        # Get Post-init Arguments
        if not self.test_mode:
            self._get_post_arguments()

        # Finalise PathManager
        self._path_manager = PathManager(
            self.base_folder,
            self._config,
            self._flags,
            self._arg_loc,
        )

    def _parse_pre_arguments(self):
        # Parser
        parser = argparse.ArgumentParser(
            prog="CensorEngine",
            description="Censors Images",
        )
        arg_mapper = {
            "uncensored_location": "uncensored-location",
            "config_location": "config-location",
            "debug_level": "debug-level",
        }

        flag_mapper = {
            "show_stat_metrics": "sm",
            "pad_individual_items": "pi",
            "dev_tools": "dt",
            "show_full_output_path": "fo",
            "using_test_data": "td",
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
        # # File Location
        if loc := args.uncensored_location:
            self._arg_loc = Path(loc)

        # # Config Location
        if config := args.config_location:
            self.load_config(config)
            self._used_boot_config = True

        if debug_word := args.debug_level:
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
        if self._arg_loc and (self._arg_loc.parts and self._arg_loc.parts[0] == "."):
            self._flags["_using_shortcut"] = True

    def load_config(self, config_data: str | dict[str, Any]):
        if isinstance(config_data, str):
            self._config = Config.from_yaml(self.base_folder, config_data)
        elif isinstance(config_data, dict):
            self._config = Config.from_dictionary(config_data)
        else:
            raise TypeError("invalid type used")

    def display_times(self):
        # Reporting
        if self._flags["show_stat_metrics"] and len(self._time_durations) != 0:
            self.display_bulk_stats(self._time_durations)

    def start(self) -> list[CVImage] | None:
        # Find Files
        args: dict[str, Any] = {
            "main_files_path": self.base_folder,
            "indexed_files": self._find_files(self._path_manager),
            "config": self._config,
            "debug_level": self._debug_level,
            "in_place_durations": self._time_durations,
            "function_get_index": self._get_index_text,
            "flags": self._flags,
            "path_manager": self._path_manager,
        }
        video_args = args.copy()
        video_args["function_display_times"] = self.display_times

        # What to Censor
        if self.censor_mode == "image":
            self._image_pipeline(**args)
        elif self.censor_mode == "video":
            self._video_pipeline(**video_args)
        else:
            self._image_pipeline(**args)
            self._video_pipeline(**video_args)
        self.display_times()

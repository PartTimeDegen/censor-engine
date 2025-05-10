import argparse
from dataclasses import dataclass, field
from typing import Any

from censorengine.backend.models.censor_engine.components.pipeline_image import (
    ComponentImagePipeline,
)
from censorengine.backend.models.censor_engine.components.pipeline_video import (
    ComponentVideoPipeline,
)
from censorengine.backend.models.censor_engine.components.reporting import (
    ComponentReporting,
)
from censorengine.backend.models.censor_engine.components.uitls import ComponentUtils
from censorengine.backend.models.config import Config

from censorengine.backend.models.tools.debugger import DebugLevels
from censorengine.backend.models.tools.dev_tools import DevTools


@dataclass(slots=True, repr=False, eq=False, order=False, match_args=False)
class CensorEngine(
    ComponentReporting,
    ComponentUtils,
    ComponentImagePipeline,
    ComponentVideoPipeline,
):
    # Information
    main_files_path: str
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
    _arg_loc: str = field(init=False)
    _full_files_path: str = field(init=False)
    _config: Config = field(init=False)
    _used_boot_config: bool = field(default=False, init=False)
    _durations: list[str] = field(default_factory=list, init=False)

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

    def load_config(self, config_data: str | dict[str, Any]):
        if isinstance(config_data, str):
            self._config = Config.from_yaml(self.main_files_path, config_data)
        elif isinstance(config_data, dict):
            self._config = Config.from_dictionary(config_data)
        else:
            raise TypeError("invalid type used")

    def display_times(self):
        # Reporting
        if self._flags["show_stat_metrics"] and len(self._time_durations) != 0:
            self.display_bulk_stats(self._time_durations)

    def start(self):
        # Find Files
        args: dict[str, Any] = {
            "main_files_path": self.main_files_path,
            "indexed_files": self._find_files(self.main_files_path, self._config),
            "config": self._config,
            "debug_level": self._debug_level,
            "in_place_durations": self._time_durations,
            "function_get_index": self._get_index_text,
            "function_save_file": self._save_file,
            "flags": self._flags,
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

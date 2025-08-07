from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import __main__
from censor_engine.models.config import Config
from censor_engine.models.lib_models.detectors import DetectedPartSchema
from censor_engine.paths import PathManager
from censor_engine.typing import Image

from .mixin_arguments import MixinArguments
from .mixin_pipeline_image import MixinImagePipeline
from .mixin_pipeline_video import MixinVideoPipeline
from .mixin_reporting import MixinReporting
from .mixin_utils import MixinUtils
from .tools.debugger import DebugLevels
from .tools.dev_tools import DevTools


@dataclass(slots=True, repr=False, eq=False, order=False)
class CensorEngine(
    MixinImagePipeline,
    MixinVideoPipeline,
    MixinReporting,
    MixinArguments,
    MixinUtils,
):
    # Information
    uncensored_folder: str | Path | None = None
    censored_folder: str | Path | None = None
    base_folder: Path | str = Path(__main__.__file__).resolve().parent
    censor_mode: str = "auto"
    config_data: str | dict[str, Any] = "00_default.yml"

    # Test Stuff
    _test_mode: bool = False
    _test_detection_output: list[DetectedPartSchema] | None = None

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
    _durations: list[str] = field(default_factory=list, init=False)
    _path_manager: PathManager = field(init=False)

    def __post_init__(self):
        # Conversions
        self.base_folder = Path(self.base_folder)

        # Handle Config and Arguments
        arguments: dict[str, Any] = {
            "arg_loc": self._arg_loc,
            "debug_level": self._debug_level,
            "config": self.config_data,
            "flags": self._flags,
        }
        settings = self._parse_arguments(
            self.base_folder,
            self.config_data,
            arguments,
        )  # Same Data as `arguments`

        self._arg_loc = settings["arg_loc"]
        self._debug_level = settings["debug_level"]
        self._config = settings["config"]
        self._flags = settings["flags"]

        # Alter Config with Function Argument Settings
        if uncen_folder := self.uncensored_folder:
            self._config.file_settings.uncensored_folder = Path(uncen_folder)
        if cen_folder := self.censored_folder:
            self._config.file_settings.censored_folder = Path(cen_folder)

        # Test Stuff
        if self._test_mode:
            # Paths Fix
            self._config.file_settings.uncensored_folder = Path()
            self._config.file_settings.censored_folder = Path()

        # Finalise PathManager
        self._path_manager = PathManager(
            self.base_folder,
            self._config,
            self._flags,
            self._arg_loc,
            self._test_mode,
        )

    def display_times(self):
        # Reporting
        if self._flags["show_stat_metrics"] and len(self._time_durations) != 0:
            self.display_bulk_stats(self._time_durations)

    def start(self) -> list[Image]:
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
            "inline_mode": self._test_mode,  # TODO: This should be a feature
            "_test_detection_output": self._test_detection_output,  # Need to Improve
        }
        video_args = args.copy()
        video_args["function_display_times"] = self.display_times

        # What to Censor
        memory_files: list[
            Image
        ] = []  # TODO This should be a dict or class maybe
        if self.censor_mode == "image":
            memory_files.extend(self._image_pipeline(**args))
        elif self.censor_mode == "video":
            memory_files.extend(self._video_pipeline(**video_args))
        else:
            memory_files.extend(self._image_pipeline(**args))
            memory_files.extend(self._video_pipeline(**video_args))
        self.display_times()

        return memory_files

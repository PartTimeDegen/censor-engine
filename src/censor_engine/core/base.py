from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from censor_engine._typing import Image
from censor_engine.core.paths import PathManager
from censor_engine.core_engine.cli.parser import build_parser
from censor_engine.core_engine.cli.processor import process_arguments
from censor_engine.core_engine.structs import EngineSettings
from censor_engine.core_engine.utils import load_config
from censor_engine.models.enums import CensorMode
from censor_engine.models.lib_models.detectors.ai_models import AIModel
from censor_engine.models.lib_models.detectors.schemas import (
    DetectedPart,
)

from ..models.models_core.engine.mixin_model_management import (
    MixinModelManagement,
)
from ..models.models_core.path_manager.mixin_utils import MixinUtils
from ..models.models_core.pipelines.image.mixin_pipeline_image import (
    MixinImagePipeline,
)
from ..models.models_core.pipelines.video.mixin_pipeline_video import (
    MixinVideoPipeline,
)
from ..models.models_core.tools.debugger.dev_tools import DevTools


@dataclass(slots=True, repr=False, eq=False, order=False)
class CensorEngine(
    MixinImagePipeline,
    MixinVideoPipeline,
    MixinModelManagement,
    MixinUtils,
):
    """
    This is the main class of CensorEngine. This handles all the censoring
    tools and functionality.

    :param MixinImagePipeline: This is the pipeline for Images
    :param MixinVideoPipeline: This is the pipeline for Videos
    :param MixinUtils: This is a utils Mixin
    """

    # Information
    uncensored_folder: str | Path = Path("uncensored")
    censored_folder: str | Path = Path("censored")
    base_folder: Path | str = field(default_factory=Path.cwd)
    censor_mode: CensorMode = CensorMode.AUTO
    config_data: str | dict[str, Any] = "basic/default.yml"

    # Test Stuff
    _test_mode: bool = False
    _test_detection_output: (
        list[DetectedPart] | list[list[DetectedPart]] | None
    ) = None

    # Settings
    _settings: EngineSettings = field(init=False)

    # Debug & Dev Tools
    _time_durations: list[float] = field(init=False, default_factory=list)
    _dev_tools: DevTools | None = field(init=False, default=None)

    # Internal State Variables
    _base_folder: Path = field(init=False)
    _full_files_path: str = field(init=False)
    _durations: list[str] = field(default_factory=list, init=False)
    _path_manager: PathManager = field(init=False)
    _live_detectors: list[AIModel] = field(init=False)

    def __post_init__(self):
        # Type Fixing
        self._base_folder = Path(self.base_folder)

        # Load Settings
        self._settings = EngineSettings(
            config=load_config(self._base_folder, self.config_data)
        )
        self.__handle_init_args()

        # Handle Folder Logic
        self.__handle_init_folders()

        # Test Stuff
        bool_using_test_data = (
            self._settings.flags.using_test_data
            or self._settings.flags.example_preview
        )
        if not bool_using_test_data:
            self.__handle_test_init_stuff()

        # Finalise PathManager
        self._path_manager = PathManager(
            self.base_folder,
            self._config,
            self._flags,
            self._arg_loc,
            self._test_mode,
        )

        # Enable Detectors
        self._live_detectors = self._activate_used_models(
            self._settings.config
        )

    # Post-Init Helpers
    def __handle_init_args(self) -> None:
        # Make, Parse, and Process Arguments
        parser = build_parser()
        args, _ = parser.parse_known_args()
        settings = process_arguments(args=args, base_folder=self._base_folder)

        # Load Settings
        self._settings.load_parsed_args(settings)

    def __handle_init_folders(self) -> None:
        # Helper Function
        file_settings = self._settings.config.file_settings

        # Core Folder Overrides
        if uncen_folder := self.uncensored_folder:
            file_settings.uncensored_folder = Path(uncen_folder)
        if cen_folder := self.censored_folder:
            file_settings.censored_folder = Path(cen_folder)

        # CLI Override
        if override_path := self._settings.uncensored_location_override:
            file_settings.uncensored_folder = override_path

    def __handle_test_init_stuff(self) -> None:
        self._test_mode = True
        self.censor_mode = CensorMode.PREVIEW

        # Paths Fix
        file_settings = self._settings.config.file_settings
        file_settings.uncensored_folder = Path()
        file_settings.censored_folder = Path()

    def start(self) -> list[Image]:
        """
        This is the main entrypoint for censorengine.

        TODO: Change the output to a dict or dataclass
        TODO: Make Inline mode a feature
        TODO: Import the test detection output thing

        :return list[Image]: List of censored images.
        """
        # Find Files
        # TODO: Change to Context
        args: dict[str, Any] = {
            "main_files_path": self.base_folder,
            "indexed_files": self._find_files(self._path_manager),
            "config": self._config,
            "debug_level": self._debug_level,
            "function_get_index": self._get_index_text,
            "flags": self._flags,
            "path_manager": self._path_manager,
            "inline_mode": self._test_mode,
            "_test_detection_output": self._test_detection_output,
            "live_detectors": self._live_detectors,
        }

        # Video Args
        video_args = args.copy()

        # What to Censor
        memory_files: list[Image] = []
        if self.censor_mode in {"image", "preview"}:
            memory_files.extend(self._image_pipeline(**args))
        elif self.censor_mode == "video":
            memory_files.extend(self.run_video_pipeline(**video_args))
        else:
            memory_files.extend(self._image_pipeline(**args))
            memory_files.extend(self.run_video_pipeline(**video_args))
        self.display_times()

        return memory_files

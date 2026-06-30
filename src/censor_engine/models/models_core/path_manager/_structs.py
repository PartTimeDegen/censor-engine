from dataclasses import dataclass, field
from pathlib import Path

from censor_engine.models.models_library.configs.settings._file_handling import (  # noqa: E501
    FileHandingSettings,
)


@dataclass(slots=True)
class PathManagerFlags:
    using_test_data: bool = False
    using_shortcut: bool = False
    display_full_output: bool = False


@dataclass(slots=True)
class ToolPaths:
    ffmpeg_file_path: Path | None = None


@dataclass(slots=True)
class ConfigShortcuts:
    file_config: FileHandingSettings
    uncensored_base_dir: Path = field(init=False)
    censored_base_dir: Path = field(init=False)

    def __post_init__(self):
        self.uncensored_base_dir = self.file_config.folders.uncensored
        self.censored_base_dir = self.file_config.folders.censored

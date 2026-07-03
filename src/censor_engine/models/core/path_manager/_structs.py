from dataclasses import dataclass, field
from pathlib import Path

from censor_engine.models.core.path_manager.schemas import FileType
from censor_engine.models.libraries.configs.settings._file_handling import (
    FileHandingSettings,
)


@dataclass(slots=True)
class PathManagerFlags:
    using_test_data: bool = False
    using_shortcut: bool = False
    display_full_output: bool = False
    example_preview: bool = False


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


@dataclass(frozen=True)
class ApprovedFormats:
    image_formats: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp"}
    )
    video_formats: frozenset[str] = frozenset({".mp4", ".webm", ".mov"})

    @property
    def all_formats(self) -> frozenset[str]:
        return self.image_formats | self.video_formats

    def get_format_type(self, file_path: Path) -> FileType:
        return (
            FileType.VIDEO
            if file_path.suffix.lower() in self.video_formats
            else FileType.IMAGE
        )

    def check_is_approved(self, file_path: Path) -> bool:
        if file_path.is_dir():
            return False
        return file_path.suffix.lower() in self.all_formats

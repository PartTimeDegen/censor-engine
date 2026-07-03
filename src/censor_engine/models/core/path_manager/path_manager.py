from dataclasses import dataclass, field
from pathlib import Path

from censor_engine.models.core.cli.structs import ParsedArguments
from censor_engine.models.libraries.configs.config import Config

from ._endpoint_handler import EndpointHandler
from ._file_explorer import FileExplorer
from ._structs import (
    ConfigShortcuts,
    PathManagerFlags,
    ToolPaths,
)
from .schemas import IndexedFile


@dataclass(slots=True)
class PathManager:
    """
    This dataclass is used for the path management and handling, such that
    the paths are contained in a single area.

    The PathManager handles stuff like how the uncensored folder is managed and
    generated, especially for custom features like the commandline arg or test
    mode.

    """

    # Input Variables
    base_dir: Path
    args: ParsedArguments

    # Public Paths
    config: Config = field(init=False)
    paths: EndpointHandler = field(init=False)

    # Easy Access Classes
    _flags: PathManagerFlags = field(init=False)
    _shortcuts: ConfigShortcuts = field(init=False)

    # Pathing Classes
    _tools: ToolPaths = field(default_factory=ToolPaths)  # TODO: Need to Do

    # Internal Tools
    _file_explorer: FileExplorer = field(init=False)

    def __post_init__(self):
        # Config Shortcut
        config = self.args.config
        if config is None:
            msg = "Missing Config"
            raise TypeError(msg)
        self.config = config

        self._shortcuts = ConfigShortcuts(self.config.file_handling)

        # Apply Flags
        flags = self.args.flags
        self._flags = PathManagerFlags(
            using_test_data=flags.using_test_data,
            using_shortcut=flags.using_shortcut,
            display_full_output=flags.show_full_output_path,
            example_preview=flags.example_preview,
        )

        # Apply Paths
        self.paths = EndpointHandler(
            base_dir=self.base_dir,
            uncensored_base_dir=self._shortcuts.uncensored_base_dir,
            censored_base_dir=self._shortcuts.censored_base_dir,
            using_test_data=self._flags.using_test_data,
            using_shortcut=self._flags.using_shortcut,
        )

        # Apply Internal Tools
        self._file_explorer = FileExplorer(
            root_path=self.base_dir,
            preview_mode=self._flags.example_preview,
        )

    # Loader
    def load_media_path_into_manager(self, relative_media_path: Path) -> Path:
        # Load Media Path into Path Handler
        self.paths.raw_uncensored_path = relative_media_path
        censored_folder = self.paths.absolute_censored_media_path

        # # Make Folder From Path
        censored_folder.mkdir(parents=True, exist_ok=True)
        return censored_folder

    # Properties
    @property
    def folder_for_output_display(self) -> str:
        if self._flags.display_full_output:
            return str(self.paths.relative_censored_media_path)
        return self.paths.relative_censored_media_path.name

    # Methods
    def get_files(self) -> list[IndexedFile]:
        return self._file_explorer.find_files()

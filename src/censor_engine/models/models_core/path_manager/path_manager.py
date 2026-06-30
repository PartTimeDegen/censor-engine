from dataclasses import dataclass, field
from pathlib import Path

from censor_engine.core.cli.structs import ParsedArguments

from ._endpoint_handler import EndpointHandler
from ._structs import (
    ConfigShortcuts,
    PathManagerFlags,
    ToolPaths,
)


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

    # Easy Access Classes
    _flags: PathManagerFlags = field(init=False)
    _shortcuts: ConfigShortcuts = field(init=False)

    # Pathing Classes
    _paths: EndpointHandler = field(init=False)
    _tools: ToolPaths = field(default_factory=ToolPaths)  # TODO: Need to Do

    def __post_init__(self):
        # Config Shortcut
        config = self.args.config
        if config is None:
            msg = "Missing Config"
            raise TypeError(msg)

        self._shortcuts = ConfigShortcuts(config.file_handling)

        # Apply Flags
        flags = self.args.flags
        self._flags = PathManagerFlags(
            using_test_data=flags.using_test_data,
            using_shortcut=flags.using_shortcut,
            display_full_output=flags.show_full_output_path,
        )

        # Apply Paths
        self._paths = EndpointHandler(
            base_dir=self.base_dir,
            uncensored_base_dir=self._shortcuts.uncensored_base_dir,
            censored_base_dir=self._shortcuts.censored_base_dir,
            using_test_data=self._flags.using_test_data,
            using_shortcut=self._flags.using_shortcut,
        )

    # Loader
    def load_media_path_into_manager(self, relative_media_path: Path) -> Path:
        # Load Media Path into Path Handler
        self._paths.raw_uncensored_path = relative_media_path
        censored_folder = self._paths.absolute_censored_media_path

        # # Make Folder From Path
        censored_folder.mkdir(parents=True, exist_ok=True)
        return censored_folder

    # Properties
    @property
    def folder_for_output_display(self):
        if self._flags.display_full_output:
            return self._paths.relative_censored_media_path
        return self._paths.relative_censored_media_path.name

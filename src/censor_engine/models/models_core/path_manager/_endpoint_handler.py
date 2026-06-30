from dataclasses import dataclass
from pathlib import Path

from ._paths import PATH_TEST_DATA


@dataclass(slots=True)
class EndpointHandler:
    base_dir: Path
    uncensored_base_dir: Path
    censored_base_dir: Path

    using_test_data: bool
    using_shortcut: bool

    raw_uncensored_path: Path | None = None

    def _handle_shortcut(self, path: Path) -> Path:
        """
        This method is used to handle the "using_shortcut" flag.

        The shortcut is a flag where instead of having "uncensored/..." or what
        ever the base uncensored folder is, the user can use "./" instead.

        Note that there's a weird behaviour with Path where
        Path("./") == Path("").

        :param Path path: Used path
        :raises ValueError: Using the shortcut with the unshortcutted path
        :raises ValueError: Not using one or the other (folder or flag)
        :return Path: The fixed path
        """
        using_uncen_folder = path.is_relative_to(self.uncensored_base_dir)
        if self.using_shortcut and using_uncen_folder:
            msg = "Used the shortcut flag with the uncensored folder!"
            raise ValueError(msg)

        if not self.using_shortcut and not using_uncen_folder:
            msg = "Missing either the uncensored folder or the shortcut flag!"
            raise ValueError(msg)

        if self.using_shortcut:
            return self.uncensored_base_dir / path

        return path

    def _handle_test_data(self, path: Path) -> Path:
        """
        This method is used to handle the test data flag.

        This gives the folders the prefix "./test_data"

        :param Path path: Path used
        :return Path: Path with prefix
        """
        if not self.using_test_data:
            return path

        return PATH_TEST_DATA / path

    def _handle_flag_cases(self, path: Path) -> Path:
        """
        This method handles the collective flags changes.

        :param Path path: Path used
        :return Path: Path with changes
        """
        path = self._handle_shortcut(path)
        return self._handle_test_data(path)

    @property
    def relative_uncensored_media_path(self) -> Path:
        if self.raw_uncensored_path is None:
            msg = "File Needs to be Loaded"
            raise ValueError(msg)

        return self._handle_flag_cases(self.raw_uncensored_path)

    @property
    def absolute_uncensored_media_path(self) -> Path:
        if self.raw_uncensored_path is None:
            msg = "File Needs to be Loaded"
            raise ValueError(msg)

        return self.base_dir / self.relative_uncensored_media_path

    @property
    def media_path(self) -> Path:
        if self.raw_uncensored_path is None:
            msg = "File Needs to be Loaded"
            raise ValueError(msg)

        return self._handle_shortcut(self.raw_uncensored_path).relative_to(
            self.uncensored_base_dir
        )

    @property
    def relative_censored_media_path(self) -> Path:
        if self.raw_uncensored_path is None:
            msg = "File Needs to be Loaded"
            raise ValueError(msg)
        return self._handle_test_data(self.censored_base_dir / self.media_path)

    @property
    def absolute_censored_media_path(self) -> Path:
        if self.raw_uncensored_path is None:
            msg = "File Needs to be Loaded"
            raise ValueError(msg)
        return self.base_dir / self.relative_censored_media_path

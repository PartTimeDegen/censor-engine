import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from censor_engine.models.config import Config

PATH_TEST_DATA = Path(".test_data")
PATH_SHORTCUT_UNCENSORED = Path()


@dataclass(slots=True)
class PathManager:
    """
    This dataclass is used for the path management and handling, such that
    the paths are contained in a single area.

    The PathManager handles stuff like how the uncensored folder is managed and
    generated, especially for custom features like the commandline arg or test
    mode.

    """

    base_directory: Path
    config: Config
    flags: dict[str, bool]
    args_loc: Path | None
    test_mode: bool

    # Internal
    _is_using_test_data: bool = field(init=False, default=False)
    _is_using_shortcut: bool = field(init=False, default=False)
    _is_using_full_output: bool = field(init=False, default=False)

    _uncensored_folder: Path = field(init=False)
    _censored_folder: Path = field(init=False)

    _cache_uncensored_folder: Path | None = field(init=False, default=None)
    _cache_censored_folder: Path | None = field(init=False, default=None)

    # Tools/Binaries
    ffmpeg_file_path: Path = field(init=False)

    def __post_init__(self):
        self._is_using_test_data = self.flags.get("using_test_data", False)
        self._is_using_shortcut = self.flags.get("_using_shortcut", False)
        self._is_using_full_output = self.flags.get(
            "show_full_output_path",
            False,
        )

        if self._is_using_shortcut and self.args_loc:
            self.args_loc = (
                self.config.file_settings.uncensored_folder
                / self.args_loc.relative_to(PATH_SHORTCUT_UNCENSORED)
            )

        uncensored_folder = (
            self.args_loc
            if self.args_loc
            else self.config.file_settings.uncensored_folder
        )
        censored_folder = (
            self.args_loc
            if self.args_loc
            else self.config.file_settings.censored_folder
        )

        if self._is_using_test_data:
            self.base_directory = self.base_directory / PATH_TEST_DATA
            uncensored_folder = (
                self.base_directory / "uncensored" / uncensored_folder
            )
            censored_folder = (
                self.base_directory / "censored" / censored_folder
            )

        self._uncensored_folder = uncensored_folder
        self._censored_folder = censored_folder

        if self.test_mode:
            self._censored_folder = self.base_directory

        # FFMPeg for Video
        self.__get_correct_ffmpeg_binary()
        self.__mount_ffmpeg()

    def __get_correct_ffmpeg_binary(self) -> None:
        base_path = Path("tools/ffmpeg")
        if sys.platform.startswith("win"):
            self.ffmpeg_file_path = base_path / "ffmpeg.exe"
        elif sys.platform.startswith("linux") or sys.platform.startswith(
            "darwin"
        ):
            self.ffmpeg_file_path = base_path / "ffmpeg"
        else:
            msg = f"Unsupported OS: {sys.platform}"
            raise RuntimeError(msg)

        if not self.ffmpeg_file_path.exists():
            msg = f"FFmpeg binary not found at {self.ffmpeg_file_path}"
            raise FileNotFoundError(msg)

        # Get Full Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.ffmpeg_file_path = (repo_root / self.ffmpeg_file_path).resolve()
        self.ffmpeg_file_path.chmod(0o755)

    def __mount_ffmpeg(self):
        if not os.environ.get("FFMPEG_BINARY"):
            os.environ["FFMPEG_BINARY"] = str(self.ffmpeg_file_path)
            print(f"FFmpeg set to: {self.ffmpeg_file_path}")  # noqa: T201

    def get_uncensored_folder(self) -> Path:
        if folder := self._cache_uncensored_folder:
            return folder

        self._cache_uncensored_folder = (
            self.base_directory / self._uncensored_folder
        )
        return self._cache_uncensored_folder

    def get_censored_folder(self) -> Path:
        if (folder := self._cache_censored_folder) is not None:
            return folder

        base_dir = self.base_directory
        base_dir = base_dir / self._censored_folder

        self._cache_censored_folder = (
            base_dir
            / base_dir.resolve().relative_to(self._censored_folder.resolve())
        )
        return self._cache_censored_folder

    def get_output_censored_folder(self) -> Path | None:
        if self._is_using_test_data:
            return PATH_SHORTCUT_UNCENSORED / self._censored_folder

        if self._is_using_shortcut:
            return (
                PATH_SHORTCUT_UNCENSORED
                / self._censored_folder.relative_to(
                    self.config.file_settings.censored_folder,
                )
            )

        return None

    def get_flag_is_using_full_path(self) -> bool:
        return self._is_using_full_output

    def get_save_file_path(
        self,
        file_name: str,
        *,
        force_png: bool = False,
    ) -> str:
        file_path = Path(file_name)

        # Compute relative path inside uncensored folder and map to censored
        # folder
        try:
            relative = file_path.relative_to(self.get_uncensored_folder())
            new_path = self.get_censored_folder() / relative
        except ValueError:
            # If file_path is not under uncensored folder, keep as is but in
            # censored folder base
            new_path = self.get_censored_folder() / file_path.name

        # Split filename and extension
        stem = new_path.stem
        ext = new_path.suffix

        # Build list with prefix, original stem, and suffix (skip empty parts)
        parts = [
            self.config.file_settings.file_prefix,
            stem,
            self.config.file_settings.file_suffix,  # FIXME: Dumb AI
        ]
        fixed_parts = [part for part in parts if part]

        # Join parts with underscore to make new filename
        new_file_name = "_".join(fixed_parts)

        # Construct the new full path, preserving any subfolders relative to
        # censored folder
        if new_path.parent != self.get_censored_folder():
            # Keep subfolders structure
            new_folder = new_path.parent
        else:
            new_folder = self.get_censored_folder()

        # Ensure directory exists
        new_folder.mkdir(parents=True, exist_ok=True)

        # Change extension if force_png
        if force_png:
            ext = ".png"

        # Final full path
        final_path = new_folder / f"{new_file_name}{ext}"

        return str(final_path)

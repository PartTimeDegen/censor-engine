import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from censor_engine.models.config import Config
from censor_engine.models.config.file import FileConfig

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
    # # Flags (hence f)
    _f_test_data: bool = field(init=False, default=False)
    _f_shortcut: bool = field(init=False, default=False)
    _f_full_output: bool = field(init=False, default=False)

    # # Config Shortcut
    _conf_file: FileConfig = field(init=False)

    # # Paths
    _p_uncen: Path = field(init=False)
    _p_cen: Path = field(init=False)
    _p_cache: Path = field(init=False)

    # Tools/Binaries
    ffmpeg_file_path: Path = field(init=False)

    def __post_init__(self):
        # Quality of Life Variable
        self._conf_file = self.config.file_settings

        self._f_test_data = self.flags.get("using_test_data", False)
        self._f_shortcut = self.flags.get("_using_shortcut", False)
        self._f_full_output = self.flags.get(
            "show_full_output_path",
            False,
        )

        # Handle when Shortcut is Used
        if self._f_shortcut and self.args_loc:
            self.args_loc = (
                self._conf_file.uncensored_folder
                / self.args_loc.relative_to(PATH_SHORTCUT_UNCENSORED)
            )

        # Handle if CLI Args Change Loc
        uncen = self._conf_file.uncensored_folder
        cen = self._conf_file.censored_folder
        cache = Path(".cache")
        if cli_loc := self.args_loc:
            uncen = cli_loc
            cen = cli_loc

        # Check if using Test Data
        if self._f_test_data:
            self.base_directory = self.base_directory / PATH_TEST_DATA
            uncen = self.base_directory / "uncensored" / uncen
            cen = self.base_directory / "censored" / cen
            cache = self.base_directory / ".cache" / cache

        # Save Paths
        self._p_uncen = uncen
        self._p_cen = cen
        self._p_cache = cen

        # Check if doing Testing
        if self.test_mode:
            self._p_cen = self.base_directory

        # FFMPeg for Video
        self.__get_correct_ffmpeg_binary()
        self.__mount_ffmpeg()

        # Create Caching Folder
        self.__create_cache()

    # Tools
    # # FFmpeg
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

    def __download_ffmpeg(self): ...

    def __mount_ffmpeg(self):
        if not os.environ.get("FFMPEG_BINARY"):
            os.environ["FFMPEG_BINARY"] = str(self.ffmpeg_file_path)
            print(f"FFmpeg set to: {self.ffmpeg_file_path}")  # noqa: T201

    # Caching
    # # Meta Data
    def save_cache_meta(self): ...
    def load_cache_meta(self): ...

    # # Checking Media Hash
    def get_media_hash(self): ...
    def compare_media_hash(self): ...

    # # Saving Frame Data
    def save_cache_frame(self, frame: int): ...
    def load_cache_frame(self, frame: int): ...

    # # Saving Cache
    def __create_cache(self):
        self.get_cache_folder().mkdir(parents=True, exist_ok=True)

    # Public
    def get_cache_folder(self):
        return self.base_directory / ".cache"

    def get_file_cache_folder(self, file_name: str):
        return self.get_cache_folder() / Path(file_name).relative_to(
            self.base_directory
        )

    def get_uncensored_folder(self) -> Path:
        return self.base_directory / self._p_uncen

    def get_censored_folder(self) -> Path:
        return self.base_directory / self._p_cen

    def get_output_censored_folder(self) -> Path | None:
        if self._f_test_data:
            return PATH_SHORTCUT_UNCENSORED / self._p_cen

        if self._f_shortcut:
            return PATH_SHORTCUT_UNCENSORED / self._p_cen.relative_to(
                self._conf_file.censored_folder,
            )

        return None

    def get_flag_is_using_full_path(self) -> bool:
        return self._f_full_output

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
            self._conf_file.file_prefix,
            stem,
            self._conf_file.file_suffix,  # FIXME: Dumb AI
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

import platform
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ToolDownloader:
    base_folder: Path
    tool_name: str
    links: dict = field(
        default_factory=lambda: {
            "Windows": None,
            "Linux": None,
            "Darwin": None,
        }
    )
    file_name: dict[str, str | None] = field(
        default_factory=lambda: {
            "Windows": None,
            "Linux": None,
            "Darwin": None,
        }
    )
    archive_extensions: dict[str, str] = field(
        default_factory=lambda: {
            "Windows": ".zip",
            "Linux": ".tar.xz",
            "Darwin": ".zip",
        }
    )

    _system: str = ""
    _tool_folder: Path = Path()

    _archive_dir: Path = Path()
    _file_dir: Path = Path()

    def __post_init__(self):
        # System Stuff
        self._system = platform.system()

        is_in_link_keys = self._system in self.links
        has_link = self.links.get(self._system)
        if not is_in_link_keys or not has_link:
            msg = f"Missing OS: {self._system}"
            raise KeyError(msg)

        # Folder Stuff
        self._tool_folder = self.base_folder / self.tool_name

        # File Stuff
        if self._system not in self.file_name:
            msg = f"Missing File Name for: {self._system}"
            raise KeyError(msg)

        file_name = self.file_name[self._system]
        if file_name is None:
            msg = f"Missing File Name for: {self._system}"
            raise KeyError(msg)

        self._file_dir = self._tool_folder / self._system / file_name

    def __extract(self, archive: Path):
        ffmpeg_dir = self._tool_folder / self._system
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as z:
                for member in z.namelist():
                    fname = Path(member).name
                    if fname.lower() in [
                        "ffmpeg",
                        "ffmpeg.exe",
                        "ffprobe",
                        "ffprobe.exe",
                    ]:
                        z.extract(member, ffmpeg_dir)
                        # Move to top-level
                        shutil.move(
                            str(ffmpeg_dir / member), ffmpeg_dir / fname
                        )

        else:
            with tarfile.open(archive, "r:*") as t:
                for member in t.getmembers():
                    fname = Path(member.name).name  # type: ignore
                    if fname in ["ffmpeg", "ffprobe"]:
                        t.extract(member, ffmpeg_dir)
                        shutil.move(
                            str(ffmpeg_dir / member.name),  # type: ignore
                            ffmpeg_dir / fname,
                        )

                t.extractall(self._tool_folder / self._system)  # noqa: S202
        archive.unlink()

    def __download_file(self) -> Path | None:
        if self._file_dir.exists():
            return None
        download_link = self.links.get(self._system)

        if download_link is None:
            msg = f"Cannot download: Missing link for {self._system}"
            raise ValueError(msg)

        tool_name = self._file_dir.name + self.archive_extensions[self._system]
        archive_dir = self._tool_folder / tool_name

        url = self.links[self._system]
        print(f"Couldn't find {self.tool_name}, downloading from {url}")  # noqa: T201
        print(f"To: {archive_dir}")  # noqa: T201
        urllib.request.urlretrieve(url, archive_dir)  # noqa: S310

        return archive_dir

    def get_tool(self) -> Path:
        archive_folder = self.__download_file()
        if archive_folder is not None:
            self.__extract(archive_folder)
        return self._file_dir

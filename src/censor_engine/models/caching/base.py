import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .caching_schemas import AIOutputData, Meta
from .video import VideoCache


@dataclass(slots=True)
class Cache:
    cache_path: Path
    base_dir: Path
    file_name: str
    is_video: bool

    _full_cache_path: Path = field(init=False)
    _current_hash: str = field(init=False)
    _video_cache: VideoCache = field(init=False)
    _image_cache: Path = field(init=False)

    def __post_init__(self):
        file_path = Path(self.file_name)
        self._current_hash = self.__get_hash(file_path)
        self._full_cache_path = self.cache_path / file_path.relative_to(
            self.base_dir
        )

        self.start()

        if self.is_video:
            self._video_cache = VideoCache(self._full_cache_path)
        else:
            self._image_cache = self._full_cache_path / "ai_output.json"

    def __get_hash(self, media_path: Path):
        sha = hashlib.sha256()

        buffer_size = 65536  # 64 KB chunks (fast + memory efficient)

        with media_path.open(mode="rb") as f:
            while chunk := f.read(buffer_size):
                sha.update(chunk)

        return sha.hexdigest()

    def __check_cache_data_exists(self):
        if not self._full_cache_path.exists():
            return False

        # Check Media is the Same as Cached (Avoids Same Name Issues)
        meta_file = self._full_cache_path / "meta.json"
        if meta_file.exists():
            with meta_file.open() as f:
                meta_data = f.read()

            # Check Hash
            meta_object = Meta.model_validate_json(meta_data)
            found_media_hash = meta_object.hash_data

            if found_media_hash == self._current_hash:
                return True

        return False

    def __create_cache_folder(self):
        # Reset Folder if Exists
        if self._full_cache_path.exists():
            shutil.rmtree(str(self._full_cache_path))
        self._full_cache_path.mkdir(parents=True)

        # Create Meta Data
        meta_file = self._full_cache_path / "meta.json"
        meta_entry = Meta(hash_data=self._current_hash)
        with meta_file.open("w") as f:
            f.write(meta_entry.model_dump_json())

    def start(self):
        if not self.__check_cache_data_exists():
            self.__create_cache_folder()

    def save_frame(self, frame: int | None, output: AIOutputData) -> None:
        if self.is_video:
            if frame is None:
                msg = "Missing Frame Number!"
                raise TypeError(msg)
            self._video_cache.set_frame_data(frame, output)
        else:
            with self._image_cache.open("w") as f:
                f.write(output.model_dump_json())

    def get_frame(self, frame: int | None) -> AIOutputData:
        if self.is_video:
            if frame is None:
                msg = "Missing Frame Number!"
                raise TypeError(msg)
            return self._video_cache.get_frame_data(frame)
        with self._image_cache.open() as f:
            return AIOutputData.model_validate_json(f.read())

    def check_for_frame(self, frame: int | None) -> bool:
        if self.is_video:
            if frame is None:
                msg = "Missing Frame Number!"
                raise TypeError(msg)
            return self._video_cache.frame_exists(frame)
        return self._image_cache.exists()

    def close(self):
        if self.is_video:
            self._video_cache.close()

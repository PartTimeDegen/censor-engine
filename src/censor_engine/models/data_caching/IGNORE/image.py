import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import META_FILE, IMAGE_OUTPUT
from ..schemas import AIOutputData, CacheData, Meta


@dataclass(slots=True)
class ImageCache:
    cache_path: Path
    base_dir: Path
    file_name: str

    _full_cache_path: Path = field(init=False)
    _current_hash: str = field(init=False)
    _image_cache: Path = field(init=False)

    def __post_init__(self):
        file_path = Path(self.file_name)
        self._current_hash = self.__get_hash(file_path)
        self._full_cache_path = self.cache_path / file_path.relative_to(
            self.base_dir
        )

        self.start()
        self._image_cache = self._full_cache_path / IMAGE_OUTPUT



    def __create_cache_folder(self):
        # Reset Folder if Exists
        if self._full_cache_path.exists():
            shutil.rmtree(str(self._full_cache_path))
        self._full_cache_path.mkdir(parents=True)

        # Create Meta Data
        meta_file = self._full_cache_path / META_FILE
        meta_entry = Meta(hash_data=self._current_hash)
        with meta_file.open("w") as f:
            f.write(meta_entry.model_dump_json())

    def start(self):
        if not self.__check_cache_data_exists():
            self.__create_cache_folder()

    def save_frame(self, frame: int | None, output: CacheData) -> None:
        with self._image_cache.open("w") as f:
            f.write(output.model_dump_json())

    def get_frame(self, frame: int | None, model_name: str) -> AIOutputData:
        with self._image_cache.open() as f:
            return AIOutputData.model_validate_json(f.read())

    def check_for_frame(self, frame: int | None, model_name: str) -> bool:
        if self._image_cache.exists():
            with self._image_cache.open() as f:
                dict_data = CacheData.model_validate_json(f.read())

            return any(
                model.model_name == model_name
                for model in dict_data.cache_data
            )
        return False

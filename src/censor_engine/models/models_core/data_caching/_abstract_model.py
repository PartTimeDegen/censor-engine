from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ._common_schemas import AIOutputData, CacheData
from ._utils import (
    check_cache_hash_matches_file,
    create_cache_folder,
    create_meta_file,
    delete_cache_folder_contents,
    get_cache_path,
    get_hash,
)


@dataclass(slots=True)
class Query:
    """
    This dataclass is used so the method of getting data is uniform.

    """

    model_name: str
    frame: int = 1


@dataclass(slots=True)
class Cache(ABC):
    base_dir: Path
    media_path: Path

    # Internal
    _full_media_path: Path = field(init=False)
    _cache_path: Path = field(init=False)

    def __post_init__(self):
        self._full_media_path = self.base_dir / self.media_path
        self._cache_path = get_cache_path(self.base_dir, self.media_path)

    # Crud Stuff
    @abstractmethod
    def save(self, data_to_cache: CacheData) -> None:
        """
        This method is used to save the data from the cache.

        :param Query query: Query Parameters
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, query: Query | None = None) -> AIOutputData:
        """
        This method is used to get the data from the cache.

        :param Query | None query: Query Parameters
        :return AIOutputData: The Cached output
        """
        raise NotImplementedError

    @abstractmethod
    def peak(self, query: Query) -> bool:
        """
        This method is used to tell if the data exists in the cache.

        :param Query query: Query Parameters
        :return bool: Does the query exist
        """
        raise NotImplementedError

    # Start and Close
    def _start_cache(self) -> Path:
        """
        This method is used to initiate the caching mechanism.

        The method is used to confirm:
        1) That the cache folder exists.
        2) That the metadata file exists and matches the current media file.

        :return Path: The cache path
        """
        # If Folder Missing
        is_cache_exists = self._cache_path.exists()
        is_cache_populated = len(list(self._cache_path.glob("*"))) != 0
        if not is_cache_exists or not is_cache_populated:
            hash_data = get_hash(self._full_media_path)
            create_cache_folder(self._cache_path)
            create_meta_file(self._cache_path, hash_data)

            return self._cache_path

        # If Cache is not Empty
        is_meta_file_matching = check_cache_hash_matches_file(
            self._cache_path,
            self.media_path,
        )

        # Check if Folder and Meta File Exists/Matches
        if is_meta_file_matching:
            return self._cache_path

        # Create Folder and Meta File
        delete_cache_folder_contents(self._cache_path)
        hash_data = get_hash(self._full_media_path)
        create_cache_folder(self._cache_path)
        create_meta_file(self._cache_path, hash_data)
        return self._cache_path

    @abstractmethod
    def close(self) -> None:
        """
        This method is used to close the cache if needed. This is primarily for
        the video cache which uses SQL and needs the engine properly closed.
        """
        raise NotImplementedError

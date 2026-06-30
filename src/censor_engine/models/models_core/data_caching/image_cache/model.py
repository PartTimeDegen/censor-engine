from dataclasses import dataclass
from pathlib import Path

from censor_engine.models.models_core.data_caching._abstract_model import (
    Cache,
    Query,
)
from censor_engine.models.models_core.data_caching._common_schemas import (
    AIOutputData,
    CacheData,
)

from ._paths import IMAGE_OUTPUT


@dataclass(slots=True)
class ImageCache(Cache):
    def start(self) -> Path:
        return self._start_cache()

    def save(self, data_to_cache: CacheData) -> None:
        """
        This method is used to save the data from the cache.

        :param Query query: Query Parameters
        """
        image_cache_path = self._cache_path / IMAGE_OUTPUT
        with image_cache_path.open("w") as f:
            f.write(data_to_cache.model_dump_json())

    def get(self, query: Query | None = None) -> AIOutputData:
        """
        This method is used to get the data from the cache.

        :param Query query: Query Parameters
        :return AIOutputData: The Cached output
        """
        image_cache_path = self._cache_path / IMAGE_OUTPUT

        with image_cache_path.open() as f:
            cache_data_info = CacheData.model_validate_json(f.read())

        return cache_data_info.cache_data[0]

    def peak(self, query: Query) -> bool:
        """
        This method is used to tell if the data exists in the cache.

        :param Query query: Query Parameters
        :return bool: Does the query exist
        """
        image_cache_path = self._cache_path / IMAGE_OUTPUT
        return image_cache_path.exists()

    def close(self) -> None:
        """
        This function isn't used for the image cache however the abstract class
        needs this implemented.
        """

from pathlib import Path

import pytest

from censor_engine.models.models_core.data_caching._abstract_model import Cache
from censor_engine.models.models_core.data_caching._common_schemas import (
    AIOutputData,
)
from censor_engine.models.models_core.data_caching._utils import get_cache_path


class DummyCache(Cache):
    def save(self, data_to_cache):
        pass

    def get(self, query=None):
        return AIOutputData(model_name="", output_data=[])

    def peak(self, query):
        return False

    def close(self):
        pass


@pytest.fixture
def media_file(tmp_path):
    media = tmp_path / "file.txt"
    media.write_bytes(b"hello world")
    return Path("file.txt")


@pytest.fixture
def cache(tmp_path, media_file):
    return DummyCache(
        base_dir=tmp_path,
        media_path=media_file,
    )


class TestCacheInitialization:
    def test_cache_path_is_set(self, cache):
        expected = get_cache_path(
            cache.base_dir,
            cache.media_path,
        )

        assert cache._cache_path == expected


class TestCacheStart:
    def test_returns_cache_path(self, cache):
        result = cache._start_cache()

        assert result == cache._cache_path

    def test_creates_cache_directory(self, cache):
        cache._start_cache()

        assert cache._cache_path.exists()
        assert cache._cache_path.is_dir()

    def test_creates_meta_file(self, cache):
        cache._start_cache()

        meta_file = cache._cache_path / "metadata.json"

        assert meta_file.exists()

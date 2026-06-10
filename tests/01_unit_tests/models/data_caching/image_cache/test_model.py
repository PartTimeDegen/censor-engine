from pathlib import Path

import pytest

from censor_engine.models.data_caching._common_schemas import (
    AIOutputData,
    CacheData,
)
from censor_engine.models.data_caching.image_cache.model import ImageCache
from censor_engine.models.data_caching.image_cache._paths import IMAGE_OUTPUT
from censor_engine.models.library_models.detectors.schemas import (
    DetectedPart,
)

@pytest.fixture
def media_file(tmp_path):
    media = tmp_path / "file.txt"
    media.write_bytes(b"hello world")
    return Path("file.txt")

DUMMY_DATA = AIOutputData(model_name="dummy", output_data=[
    DetectedPart(origin="dummy", part_id=0, label="dummy_0", score=0.67),
    DetectedPart(origin="dummy", part_id=1, label="dummy_1", score=0.67),
    DetectedPart(origin="dummy", part_id=2, label="dummy_2", score=0.67),
    DetectedPart(origin="dummy", part_id=3, label="dummy_3", score=0.67),
])

@pytest.fixture
def image_cache(tmp_path, media_file):
    cache = ImageCache(
        base_dir=tmp_path,
        media_path=media_file,
    )
    cache._start_cache()
    return cache


class TestImageCacheSave:
    def test_creates_output_file(self, image_cache):
        cache_data = CacheData(
            cache_data=[
                DUMMY_DATA
            ]
        )

        image_cache.save(cache_data)

        output_file = image_cache._cache_path / IMAGE_OUTPUT

        assert output_file.exists()
        assert output_file.is_file()

    def test_writes_cache_data_to_file(self, image_cache):
        expected = CacheData(
            cache_data=[
                DUMMY_DATA
            ]
        )

        image_cache.save(expected)

        output_file = image_cache._cache_path / IMAGE_OUTPUT

        with output_file.open() as f:
            actual = CacheData.model_validate_json(f.read())

        assert actual == expected


class TestImageCacheGet:
    def test_returns_cached_data(self, image_cache):
        expected = DUMMY_DATA

        image_cache.save(
            CacheData(
                cache_data=[expected],
            )
        )

        result = image_cache.get()

        assert result == expected


class TestImageCacheIntegration:
    def test_save_then_get(self, image_cache):
        expected = DUMMY_DATA
        image_cache.save(
            CacheData(
                cache_data=[expected],
            )
        )

        actual = image_cache.get()

        assert actual == expected
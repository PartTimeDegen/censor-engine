from pathlib import Path

import pytest

from censor_engine.models.core.data_caching._abstract_model import Query
from censor_engine.models.core.data_caching._common_schemas import (
    AIOutputData,
    CacheData,
)
from censor_engine.models.core.data_caching.video_cache.model import (
    VideoCache,
)

# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def cache(tmp_path: Path):
    base_dir = tmp_path
    media_file = Path("dummy_video.mp4")

    media_path = base_dir / media_file
    media_path.write_bytes(b"fake video content")

    c = VideoCache(base_dir=base_dir, media_path=media_file)
    c.start()

    yield c

    c.close()


def make_data(frame=1, model="model_a"):
    return CacheData(
        cache_data=[
            AIOutputData(
                model_name=model,
                frame=frame,
                output_data=[],
            )
        ]
    )


# -------------------------
# Tests
# -------------------------


class TestVideoCache:
    def test_db_is_created(self, cache):
        db_file = cache._cache_path / "video_data.db"
        assert db_file.exists()

    def test_save_and_get(self, cache):
        data = make_data(frame=1, model="model_a")

        cache.save(data)

        result = cache.get(Query(model_name="model_a", frame=1))

        assert result.model_name == "model_a"
        assert result.frame == 1
        assert result.output_data == []

    def test_peak_returns_true_when_exists(self, cache):
        data = make_data(frame=5, model="model_x")

        cache.save(data)

        assert cache.peak(Query(model_name="model_x", frame=5)) is True

    def test_peak_returns_false_when_missing(self, cache):
        assert cache.peak(Query(model_name="missing", frame=999)) is False

    def test_upsert_overwrites_existing_row(self, cache):
        data1 = make_data(frame=1, model="model_a")
        data2 = make_data(frame=1, model="model_a")

        cache.save(data1)
        cache.save(data2)

        result = cache.get(Query(model_name="model_a", frame=1))

        assert result.model_name == "model_a"
        assert result.frame == 1

    def test_multiple_models_same_frame(self, cache):
        a = make_data(frame=1, model="A")
        b = make_data(frame=1, model="B")

        cache.save(a)
        cache.save(b)

        res_a = cache.get(Query(model_name="A", frame=1))
        res_b = cache.get(Query(model_name="B", frame=1))

        assert res_a.model_name == "A"
        assert res_b.model_name == "B"

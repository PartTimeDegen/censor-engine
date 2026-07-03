import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from censor_engine.models.core.data_caching._abstract_model import (
    Cache,
    Query,
)
from censor_engine.models.core.data_caching._common_schemas import (
    AIOutputData,
    CacheData,
)

from ._paths import DATABASE_NAME
from ._queries import (
    CREATE_FRAMES_TABLE,
    FRAME_EXISTS,
    GET_FRAME,
    UPSERT_FRAME,
)


@dataclass(slots=True)
class VideoCache(Cache):
    _connection: sqlite3.Connection = field(init=False)

    def start(self) -> Path:
        cache_path = self._start_cache()

        self._connection = sqlite3.connect(
            str(cache_path / DATABASE_NAME), isolation_level=None
        )

        self._connection.execute(CREATE_FRAMES_TABLE)

        return cache_path

    def save(self, data_to_cache: CacheData) -> None:
        rows = [
            (
                item.frame,
                item.model_name,
                item.model_dump_json(),
            )
            for item in data_to_cache.cache_data
        ]

        self._connection.executemany(
            UPSERT_FRAME,
            rows,
        )

    def get(self, query: Query | None = None) -> AIOutputData:
        if query is None:
            msg = "Query required"
            raise ValueError(msg)

        row = self._connection.execute(
            GET_FRAME,
            (query.frame, query.model_name),
        ).fetchone()

        if row is None:
            msg = f"Missing frame={query.frame} model={query.model_name}"
            raise ValueError(msg)

        return AIOutputData.model_validate_json(row[0])

    def peak(self, query: Query) -> bool:
        return (
            self._connection.execute(
                FRAME_EXISTS,
                (query.frame, query.model_name),
            ).fetchone()
            is not None
        )

    def close(self) -> None:
        self._connection.close()

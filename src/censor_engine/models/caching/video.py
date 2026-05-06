import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from censor_engine.models.caching.caching_schemas import AIOutputData


@dataclass(slots=True)
class VideoCache:
    cache_path: Path

    _cache_path: Path = field(init=False)
    _connection: sqlite3.Connection = field(init=False)

    def __post_init__(self):
        self._cache_path = self.cache_path / "video_data.db"
        self.__ensure_db()

    def __ensure_db(self):
        first_time = not self._cache_path.exists()

        self._connection = sqlite3.connect(
            str(self._cache_path),
            isolation_level=None,  # autocommit
        )

        if first_time:
            self._connection.execute("""
                CREATE TABLE frames (
                    frame INTEGER,
                    model TEXT,
                    data TEXT,
                    PRIMARY KEY (frame, model)
                );
            """)

    # ---------------------
    # FRAME SET
    # ---------------------
    def set_frame_data(self, frame_number: int, model: AIOutputData):
        data_json = model.model_dump_json()

        self._connection.execute(
            """
            INSERT INTO frames(frame, model, data)
            VALUES (?, ?, ?)
            ON CONFLICT(frame, model)
            DO UPDATE SET data=excluded.data
            """,
            (frame_number, model.model_name, data_json),
        )

    # ---------------------
    # FRAME GET
    # ---------------------
    def get_frame_data(
        self, frame_number: int, model_name: str
    ) -> AIOutputData:
        row = self._connection.execute(
            """
            SELECT data FROM frames
            WHERE frame=? AND model=?
            """,
            (frame_number, model_name),
        ).fetchone()

        if row:
            return AIOutputData.model_validate_json(row[0])

        msg = f"Missing Frame Data for frame={frame_number}model={model_name}"
        raise ValueError(msg)

    # ---------------------
    # UTIL
    # ---------------------
    def frame_exists(self, frame_number: int, model_name: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM frames WHERE frame=? AND model=?",
                (frame_number, model_name),
            ).fetchone()
            is not None
        )

    def close(self):
        if self._connection:
            self._connection.close()

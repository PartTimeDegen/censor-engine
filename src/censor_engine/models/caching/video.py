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
        """Create database schema if missing."""
        first_time = not self._cache_path.exists()
        self._connection = sqlite3.connect(
            str(self._cache_path),
            isolation_level=None,
        )  # autocommit
        if first_time:
            self._connection.execute("""
                CREATE TABLE frames (
                    frame INTEGER PRIMARY KEY,
                    data TEXT
                );
            """)

    # ---------------------
    # FRAME GET/SET
    # ---------------------
    def set_frame_data(self, frame_number: int, model: BaseModel):
        """Store Pydantic model output for a frame."""
        data_json = model.model_dump_json()
        self._connection.execute(
            "INSERT INTO frames(frame, data) VALUES (?, ?) "
            "ON CONFLICT(frame) DO UPDATE SET data=?",
            (frame_number, data_json, data_json),
        )

    def get_frame_data(self, frame_number: int) -> AIOutputData:
        """Retrieve Pydantic output for a given frame."""
        row = self._connection.execute(
            "SELECT data FROM frames WHERE frame=?", (frame_number,)
        ).fetchone()
        if row:
            return AIOutputData.model_validate_json(row[0])
        msg = "Missing Frame Data, this shouldn't happen!"
        raise ValueError(msg)

    def frame_exists(self, frame_number: int) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM frames WHERE frame=?", (frame_number,)
            ).fetchone()
            is not None
        )

    def close(self):
        self._connection.close()

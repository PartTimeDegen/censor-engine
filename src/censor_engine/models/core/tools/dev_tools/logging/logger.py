from enum import IntEnum
from pathlib import Path


class LogLevel(IntEnum):
    NOTSET = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class _LogPaths:
    def __init__(self, root: Path = Path("_logs")) -> None:
        self.root = root
        self.masks = root / "masks"
        self.log_files = root / "log_files"

    def create(self) -> None:
        self.masks.mkdir(parents=True, exist_ok=True)
        self.log_files.mkdir(parents=True, exist_ok=True)


class Logger:
    set_log_level: LogLevel = LogLevel.NOTSET

    def _write_to_log(self): ...
    def _write_to_print(self): ...

    def DEBUG(self) -> None:
        pass


LOGGER = Logger

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FileType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    PREVIEW = "preview"


@dataclass(slots=True, frozen=True)
class IndexedFile:
    index: int
    path: Path
    file_type: FileType

    def get_index(self, max_file_index: int) -> str:
        width = len(str(max_file_index))
        fraction = self.index / max_file_index

        return f"{self.index:>{width}}/{max_file_index} ({fraction:>6.1%})"

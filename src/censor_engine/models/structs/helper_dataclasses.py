from dataclasses import dataclass


@dataclass(slots=True)
class IndexedFile:
    index: int
    path: str
    file_type: str  # image, video, preview

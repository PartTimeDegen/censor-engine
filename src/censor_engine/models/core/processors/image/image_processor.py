from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(slots=True)
class ImageProcessor:
    _file_uuid: UUID = field(init=False)

    def __post_init__(self):
        self._file_uuid = uuid4()

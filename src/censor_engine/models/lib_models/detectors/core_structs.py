from abc import ABC, abstractmethod
from enum import Enum, auto

from pydantic import BaseModel

from censor_engine.typing import Image


class ModelType(Enum):
    bbox = auto()
    segment = auto()
    information = auto()
    depth = auto()


class DetectedPartSchema(BaseModel):
    part_id: int = 0

    def set_part_id(self, number: int) -> None:
        self.part_id = number


class Detector(ABC):
    """
    This is the model used for detectors, it's pretty simple, just maintains a
    valid method to use/overwrite, and some meta information.

    The meta information isn't useful for code (custom models may vary),
    however for a documentation POV, it's useful to know the internal name
    (maybe logging) and what it's producing (Trust me, it's better than having
    to find NudeNet's repo to find the classifiers).

    :raises NotImplementedError: This is just to throw an error to ensure the
    model devs know they didn't properly implement the method under the
    correct name
    """

    model_name: str
    model_classifiers: tuple[str, ...]
    model_type: ModelType

    @abstractmethod
    def detect_image(
        self,
        file_images_or_path: str,
    ) -> list:
        raise NotImplementedError

    @abstractmethod
    def detect_batch(
        self,
        file_images_or_paths: list[str] | list[Image],
        batch_size: int,
    ) -> dict[int, list]:
        raise NotImplementedError

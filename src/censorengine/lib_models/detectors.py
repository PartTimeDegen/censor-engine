from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class DetectedPartSchema:
    label: str
    score: float
    relative_box: tuple[int, int, int, int]


class Detector:
    model_name: str
    model_classifiers: tuple[str, ...]

    @abstractmethod
    def detect_image(self, file_path: str) -> list[DetectedPartSchema]:
        raise NotImplementedError

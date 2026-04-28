from abc import abstractmethod

from censor_engine.models.enums import MaskType
from censor_engine.typing import Image

from .core_structs import DetectedPartSchema, Detector, ModelType


class SegmentPartSchema(DetectedPartSchema):
    label: str
    relative_box: MaskType


class SegmentDetector(Detector):
    model_type = ModelType.bbox

    @abstractmethod
    def detect_image(
        self,
        file_images_or_path: str,
    ) -> list[SegmentPartSchema]:
        raise NotImplementedError

    @abstractmethod
    def detect_batch(
        self,
        file_images_or_paths: list[str] | list[Image],
        batch_size: int,
    ) -> dict[int, list[SegmentPartSchema]]:
        raise NotImplementedError

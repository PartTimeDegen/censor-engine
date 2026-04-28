from abc import abstractmethod

from pydantic import field_validator

from censor_engine.typing import Image

from .core_structs import DetectedPartSchema, Detector, ModelType


class BBoxPartSchema(DetectedPartSchema):
    label: str
    score: float
    relative_box: tuple[int, int, int, int]  # X, Y, Width, Height

    @field_validator("relative_box", mode="before")
    def ensure_tuple(cls, v):  # noqa: ANN001, N805
        return tuple(v)


class BBoxDetector(Detector):
    model_type = ModelType.bbox

    @abstractmethod
    def detect_image(
        self,
        file_images_or_path: str,
    ) -> list[BBoxPartSchema]:
        raise NotImplementedError

    @abstractmethod
    def detect_batch(
        self,
        file_images_or_paths: list[str] | list[Image],
        batch_size: int,
    ) -> dict[int, list[BBoxPartSchema]]:
        raise NotImplementedError

from abc import abstractmethod

from censor_engine.models.lib_models.detectors.core_structs import (
    DetectedPartSchema,
    Detector,
)
from censor_engine.typing import Image


class SegmentDetector(Detector):
    @abstractmethod
    def detect_image(
        self,
        image: Image,
    ) -> list[DetectedPartSchema]:
        raise NotImplementedError

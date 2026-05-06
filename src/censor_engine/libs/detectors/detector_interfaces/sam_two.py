import logging

from censor_engine.libs.detectors.ai_models.sam_two import SAMTwo
from censor_engine.libs.registries import AIModelRegistry, DetectorRegistry
from censor_engine.models.lib_models.detectors.core_structs import (
    DetectedPartSchema,
)
from censor_engine.models.lib_models.detectors.interfaces.segment import (
    SegmentDetector,
)
from censor_engine.typing import Image

logging.getLogger("ultralytics").setLevel(logging.ERROR)


@DetectorRegistry.register()
class SamTwoDetector(SegmentDetector):
    model_name: str = "SAM2"
    model_classifiers: tuple[str, ...] = ("_body_segmentation",)
    model_object: SAMTwo = AIModelRegistry.get_all()["SAMTwo"]()  # type: ignore

    def detect_image(self, image: Image) -> list[DetectedPartSchema]:
        return [
            DetectedPartSchema(
                masks=[found_part.masks],  # type: ignore
                score=found_part.score,
            )
            for found_part in self.model_object.predict(image)
        ]

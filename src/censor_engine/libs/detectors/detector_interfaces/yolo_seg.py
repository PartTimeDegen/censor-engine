import logging

from censor_engine.libs.detectors.ai_models.yolo_seg import YoloSeg
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
class YoloSegDetector(SegmentDetector):
    model_name: str = "YoloSeg"
    model_classifiers: tuple[str, ...] = ("_body_segmentation", "_body_roi")
    model_object: YoloSeg = AIModelRegistry.get_all()["YoloSeg"]()  # type: ignore

    def detect_image(self, image: Image) -> list[DetectedPartSchema]:
        return [
            DetectedPartSchema(
                masks=[found_part.masks],  # type: ignore
                label=found_part.label,
                score=found_part.score,
                bbox=found_part.bbox,
            )
            for found_part in self.model_object.predict(image)
        ]

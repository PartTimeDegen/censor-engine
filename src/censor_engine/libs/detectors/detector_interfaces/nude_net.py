import logging

from censor_engine.libs.detectors.ai_models.nude_net import NudeNetModel
from censor_engine.libs.registries import AIModelRegistry, DetectorRegistry
from censor_engine.models.lib_models.detectors.core_structs import (
    DetectedPartSchema,
)
from censor_engine.models.lib_models.detectors.interfaces.bbox import (
    BBoxDetector,
)
from censor_engine.typing import Image

logging.getLogger("ultralytics").setLevel(logging.ERROR)


@DetectorRegistry.register()
class NudeNetDetector(BBoxDetector):
    model_name: str = "NudeNet"
    model_classifiers: tuple[str, ...] = (
        "FACE_FEMALE",
        "ARMPITS_EXPOSED",
        "ARMPITS_COVERED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "BELLY_COVERED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "ANUS_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "FEET_EXPOSED",
        "FEET_COVERED",
        "FACE_MALE",
        "MALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
    )
    model_object: NudeNetModel = AIModelRegistry.get_all()["NudeNetModel"]()  # type: ignore

    def detect_image(self, image: Image) -> list[DetectedPartSchema]:
        return [
            DetectedPartSchema(
                label=found_part.label,
                score=found_part.score,
                bbox=found_part.bbox,
            )
            for found_part in self.model_object.predict(image)
        ]

import logging

from censor_engine.models.lib_models.detectors import (
    BBoxDetector,
    BBoxPartSchema,
)
from censor_engine.typing import Image

from .model import NudeNetModel

logging.getLogger("ultralytics").setLevel(logging.ERROR)


class NudeNetDetector(BBoxDetector):
    """
    This Detector is the code of CensorEngine, it is the NudeNet model.

    This handles the core labels of the engine.

    """

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
    model_object = NudeNetModel()

    def detect_image(
        self,
        file_images_or_path: str,
    ) -> list[BBoxPartSchema]:
        return [
            BBoxPartSchema(
                label=found_part["class"],
                score=found_part["score"],
                relative_box=found_part["box"],
            )
            for _, found_part in enumerate(
                self.model_object.detect(file_images_or_path),
            )
        ]

    def detect_batch(
        self,
        file_images_or_paths: list[str] | list[Image],
        batch_size: int,
    ) -> dict[int, list[BBoxPartSchema]]:
        raise NotImplementedError("This needs work")  # noqa: EM101

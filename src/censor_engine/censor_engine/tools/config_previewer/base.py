from typing import TypedDict

from censor_engine.models.lib_models.detectors import DetectedPartSchema
from censor_engine.typing import Image

from .example_image import ImageGenerator


class ConfigInfo(TypedDict):
    preview: Image
    detection_data: list[DetectedPartSchema]


def get_config_preview(list_of_parts_enabled: list[str] | str) -> ConfigInfo:
    image_generator = ImageGenerator()
    return ConfigInfo(
        preview=image_generator.make_test_image(),
        detection_data=image_generator.return_detected_parts(
            list_of_parts_enabled
        ),
    )

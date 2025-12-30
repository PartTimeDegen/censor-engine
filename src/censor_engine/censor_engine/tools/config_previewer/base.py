from typing import TypedDict

from censor_engine.typing import Image


class ConfigInfo(TypedDict):
    preview: Image
    detection_data: dict  # TODO: Find


def get_config_preview():
    pass

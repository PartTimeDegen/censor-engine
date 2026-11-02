from pathlib import Path

import cv2
from natsort import natsorted

from censor_engine.typing import Image
from tests.helpers.controls import USE_NSFW_IMAGES, USE_SHORT_LIST

# NOTE: This is used to test the models since it needs images with actual stuff
#       to make sure it works okay (could mock outputs, one day, not today)


def get_images(
    subfolder: str | None = None,
    enable: bool = USE_NSFW_IMAGES,
    use_short_list: bool | int = USE_SHORT_LIST,
):
    if not enable:
        return ()

    base_folder = Path("tests/00_nsfw_images")
    if subfolder is not None:
        base_folder = base_folder / subfolder
    folder = tuple(
        [
            str(path)
            for path in natsorted(base_folder.glob("*"))
            if not str(path).endswith(".gitkeep")
        ]
    )
    if isinstance(use_short_list, bool) and use_short_list:
        return folder[:20]
    if isinstance(use_short_list, bool):
        return folder
    return folder[:use_short_list]


def read(image_path: str) -> Image:
    input_image = cv2.imread(image_path)
    return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


def write(file_name: str, image: Image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, bgr_image)

import os
import pytest

import cv2
import numpy as np
from censor_engine.backend.constants.typing import CVImage
from censor_engine.lib_models.detectors import DetectedPartSchema


class ImageGenerator:
    size: tuple[int, int] = (500, 500)
    seed: int = 69
    parts: tuple[DetectedPartSchema, ...] = (
        DetectedPartSchema("BREAST", 0.2, (140, 130, 100, 100)),
        DetectedPartSchema("BREAST", 0.2, (260, 130, 100, 100)),
        DetectedPartSchema("VAG", 0.2, (200, 300, 100, 100)),
    )

    def __init__(self):
        pass

    def _generate_background(self):
        height, width = self.size
        rng = np.random.default_rng(self.seed)
        noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

        return noise

    def _draw_circle(self, image: CVImage, part: DetectedPartSchema):
        x, y, w, h = part.relative_box
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(min(w, h) / 2)

        cv2.circle(image, center, radius, (266, 256, 256), -1)
        return image

    def make_test_image(self):
        image = self._generate_background()
        for part in self.parts:
            image = self._draw_circle(image, part)

        return image


@pytest.fixture(scope="function")
def root_path():
    file_path = os.path.dirname(__file__)  # noqa: F821
    list_path = file_path.split(os.sep)
    list_path = list_path[: list_path.index("tests")]
    yield os.sep.join([*list_path, "src"])


@pytest.fixture(autouse=True)
def setup_temp_dir_and_image(temp_path):
    temp_dir = temp_path.mktemp()
    temp_file = os.path.join(temp_dir, "test.jpg")

    ig = ImageGenerator().make_test_image()

    yield {
        "directory": temp_dir,
        "file_path": temp_file,
        "image": ig,
    }  # TODO: Make CensorEngine Either Take an Image or File, Then can use the file object directly
    # TODO: Make the functions more testable

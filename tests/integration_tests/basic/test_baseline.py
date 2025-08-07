from pathlib import Path

import cv2

from tests.input_image import ImageGenerator
from tests.utils import (
    load_config_base_yaml,
    run_image_test,
)


def test_test_working(dummy_input_image_data):
    config = load_config_base_yaml()

    run_image_test(dummy_input_image_data, config)


def test_image_generator():
    ig = ImageGenerator(Path())
    cv2.imwrite("tests/test_data/input_data/example.jpg", ig.make_test_image())

    assert True

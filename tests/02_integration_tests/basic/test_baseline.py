
import cv2

from censor_engine.censor_engine.tools.config_previewer.example_image import (
    ImageGenerator,
)
from tests.utils import (
    load_config_base_yaml,
    run_image_test,
)


def test_test_working(dummy_input_image_data) -> None:
    config = load_config_base_yaml()

    run_image_test(dummy_input_image_data, config=config)


def test_image_generator() -> None:
    ig = ImageGenerator()
    cv2.imwrite("example.jpg", ig.make_test_image())

    assert True


import cv2
import numpy as np
import pytest
from censor_engine.typing import CVImage
from dev_tools import ImageGenerator
from pathlib import Path


# Utils
def assert_image(
    output_image: CVImage,
    expected_image: CVImage,
    dump_path: str = "",  # Make Path
):
    if np.array_equal(output_image, expected_image):
        assert True
    else:
        # TODO Handle Folders
        # TODO Check if the file exists, if it doesn't generate a file ".checkme.jpg" then fail the test
        cv2.imwrite(str(dump_path / Path("output_image.jpg")), output_image)
        cv2.imwrite(str(dump_path / Path("expected_image.jpg")), expected_image)
        assert False


@pytest.fixture
def dummy_input_image(tmp_path) -> tuple:
    image_generator = ImageGenerator()
    dummy_img = ImageGenerator().make_test_image()
    input_path = tmp_path / "input.jpg"
    cv2.imwrite(str(input_path), dummy_img)
    return (input_path, image_generator)


@pytest.fixture
def expected_output_image_path(output_path: str):
    # Points to a pre-generated, committed reference image
    return "tests/expected_outputs/expected_blurred.png"

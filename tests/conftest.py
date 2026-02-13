from collections.abc import Generator

import cv2
import pytest

from censor_engine.censor_engine.tools.config_previewer.example_image import (
    ImageGenerator,
)
from tests.utils import ImageFixtureData


@pytest.fixture
def dummy_input_image_data(tmp_path) -> Generator[ImageFixtureData]:  # noqa: ANN001, D103
    input_path = tmp_path / "input.jpg"

    image_generator = ImageGenerator()
    dummy_img = image_generator.make_test_image()

    cv2.imwrite(str(input_path), dummy_img)
    return ImageFixtureData(
        path=input_path,
        generator=image_generator,
        parts=image_generator.parts,
    )  # type: ignore

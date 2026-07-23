from pathlib import Path

import pytest

from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)
IMAGES = [
    "base_image",
    "base_mask",
]

params = [(image, image) for image in IMAGES]


@pytest.fixture
def image(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "image_name, image", params, indirect=["image"], ids=IMAGES
)
def test_effect_fixture_images(image_name, image):
    handle_test_data(
        test_name=image_name,
        image=image,
        file_path=file_path,
    )

from pathlib import Path

import pytest

from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)
MASKS = [
    "mask_empty",
    "mask_full",
    "mask_single_part",
    "mask_three_triangle_bottom",
    "mask_three_triangle_top",
    "mask_two_parts_diagonal",
    "mask_two_parts_inline",
    "mask_two_parts_quad_square",
    "mask_two_parts_vertical",
]

params = [(mask, mask) for mask in MASKS]


@pytest.fixture
def mask(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    ("mask_name", "mask"), params, indirect=["mask"], ids=MASKS
)
def test_mask_fixture_images(mask_name, mask):
    handle_test_data(
        test_name=mask_name,
        image=mask,
        file_path=file_path,
    )

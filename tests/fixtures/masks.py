import cv2
import numpy as np
import pytest

from censor_engine._typing import MaskImage
from tests.fixtures._dimensions import (
    CANVAS_SIZE,
    HALF_PART_SIZE,
    MIDPOINT,
    THIRD,
    TWO_THIRD,
)


# BASIC
@pytest.fixture
def mask_empty() -> MaskImage:
    return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)


# JOINT
# NOTE: Ellipse is used as "base_shape" however it's basically the same as
#       circle for equal axis shapes
@pytest.fixture
def mask_single_part(mask_empty) -> MaskImage:
    mask = mask_empty.copy()
    cv2.circle(mask, (MIDPOINT, MIDPOINT), HALF_PART_SIZE, 255, thickness=-1)

    return mask


@pytest.fixture
def mask_two_parts_inline(mask_empty) -> MaskImage:
    mask = mask_empty.copy()
    cv2.circle(mask, (THIRD, MIDPOINT), HALF_PART_SIZE, 255, thickness=-1)
    cv2.circle(mask, (TWO_THIRD, MIDPOINT), HALF_PART_SIZE, 255, thickness=-1)
    return mask


@pytest.fixture
def mask_two_parts_diagonal(mask_empty) -> MaskImage:
    mask = mask_empty.copy()
    cv2.circle(mask, (THIRD, THIRD), HALF_PART_SIZE, 255, thickness=-1)
    cv2.circle(mask, (TWO_THIRD, TWO_THIRD), HALF_PART_SIZE, 255, thickness=-1)
    return mask


@pytest.fixture
def mask_two_parts_vertical(mask_empty) -> MaskImage:
    mask = mask_empty.copy()
    cv2.circle(mask, (MIDPOINT, THIRD), HALF_PART_SIZE, 255, thickness=-1)
    cv2.circle(mask, (MIDPOINT, TWO_THIRD), HALF_PART_SIZE, 255, thickness=-1)
    return mask


@pytest.fixture
def mask_two_parts_quad_square(mask_empty) -> MaskImage:
    mask = mask_empty.copy()
    cv2.circle(mask, (THIRD, TWO_THIRD), HALF_PART_SIZE, 255, thickness=-1)
    cv2.circle(mask, (THIRD, THIRD), HALF_PART_SIZE, 255, thickness=-1)
    cv2.circle(mask, (TWO_THIRD, TWO_THIRD), HALF_PART_SIZE, 255, thickness=-1)
    cv2.circle(mask, (THIRD, TWO_THIRD), HALF_PART_SIZE, 255, thickness=-1)
    return mask


@pytest.fixture
def mask_full() -> MaskImage:
    return np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255


# Double Angle Check
@pytest.fixture
def mask_three_triangle_top(mask_empty) -> MaskImage:
    mask = mask_empty.copy()
    cv2.circle(
        mask, (THIRD, THIRD - MIDPOINT // 4), HALF_PART_SIZE, 255, thickness=-1
    )
    cv2.circle(
        mask,
        (TWO_THIRD, THIRD - MIDPOINT // 2),
        HALF_PART_SIZE,
        255,
        thickness=-1,
    )
    return mask


@pytest.fixture
def mask_three_triangle_bottom(mask_empty) -> MaskImage:
    mask = mask_empty.copy()
    cv2.circle(
        mask,
        (TWO_THIRD, THIRD + MIDPOINT // 2),
        HALF_PART_SIZE,
        255,
        thickness=-1,
    )
    cv2.circle(
        mask,
        (THIRD, THIRD + MIDPOINT // 4),
        HALF_PART_SIZE,
        255,
        thickness=-1,
    )
    return mask

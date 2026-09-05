import cv2
import numpy as np
import pytest

from censor_engine._typing import Image, MaskImage
from tests.fixtures._dimensions import (
    BIG_HALF_PART_SIZE,
    CANVAS_SIZE,
    MIDPOINT,
)

from ._constants import WHITE_3C


@pytest.fixture
def noisy_image():
    np.random.seed(42)
    return np.random.randint(
        0, 256, (CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8
    )


@pytest.fixture
def base_image(noisy_image) -> Image:
    bg = noisy_image.copy()

    cv2.circle(
        bg,
        (MIDPOINT, MIDPOINT),
        BIG_HALF_PART_SIZE,
        WHITE_3C,
        thickness=-1,
    )

    return bg


@pytest.fixture
def base_mask() -> MaskImage:
    bg = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    cv2.circle(
        bg,
        (MIDPOINT, MIDPOINT),
        int(BIG_HALF_PART_SIZE * 2),
        WHITE_3C,
        thickness=-1,
    )

    return bg


@pytest.fixture
def base_image_offset(noisy_image) -> Image:
    bg = noisy_image.copy()

    cv2.circle(
        bg,
        (MIDPOINT // 2, MIDPOINT // 2),
        BIG_HALF_PART_SIZE,
        WHITE_3C,
        thickness=-1,
    )

    return bg


@pytest.fixture
def base_mask_offset() -> MaskImage:
    bg = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    cv2.circle(
        bg,
        (MIDPOINT // 2, MIDPOINT // 2),
        int(BIG_HALF_PART_SIZE * 2),
        WHITE_3C,
        thickness=-1,
    )

    return bg


@pytest.fixture
def base_mask_blanket() -> MaskImage:
    bg = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    top = MIDPOINT - int(BIG_HALF_PART_SIZE * 2)
    cv2.rectangle(
        bg,
        (0, top),
        (CANVAS_SIZE, CANVAS_SIZE),
        WHITE_3C,
        thickness=-1,
    )

    return bg

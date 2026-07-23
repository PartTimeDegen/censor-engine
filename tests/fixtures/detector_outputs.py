import pytest

from censor_engine._typing import MaskImage
from censor_engine.models.libraries.detectors.schemas import (
    AbsoluteBBox,
    DetectorOutput,
)
from tests.fixtures._dimensions import HALF_PART_SIZE, MIDPOINT


# BASIC
@pytest.fixture
def bbox() -> AbsoluteBBox:
    topleft = int(MIDPOINT - HALF_PART_SIZE)
    bottomright = int(MIDPOINT + HALF_PART_SIZE)
    return AbsoluteBBox.from_xyxy((topleft, topleft, bottomright, bottomright))


@pytest.fixture
def bbox_long_hor() -> AbsoluteBBox:
    return AbsoluteBBox.from_xyxy(
        (
            int(MIDPOINT * 0.5),
            int(MIDPOINT - HALF_PART_SIZE),
            int(MIDPOINT * 1.5),
            int(MIDPOINT + HALF_PART_SIZE),
        )
    )


@pytest.fixture
def bbox_long_vert() -> AbsoluteBBox:
    return AbsoluteBBox.from_xyxy(
        (
            int(MIDPOINT - HALF_PART_SIZE),
            int(MIDPOINT * 0.5),
            int(MIDPOINT + HALF_PART_SIZE),
            int(MIDPOINT * 1.5),
        )
    )


@pytest.fixture
def detector_output(
    bbox: AbsoluteBBox,
    mask_empty: MaskImage,
    mask_full: MaskImage,
    single_part,
) -> DetectorOutput:
    return DetectorOutput(
        bbox=bbox,
        # assumes MaskImage is bytes-like in tests
        masks=[mask_empty, mask_full],
        origin="test_origin",
        part_id=1,
        label=single_part,
        score=0.95,
    )

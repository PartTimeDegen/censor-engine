import pytest

from censor_engine._typing import Image, MaskImage
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


@pytest.fixture
def effect_context(
    detection_properties: PartProperties,
    base_image: Image,
    base_mask: MaskImage,
):
    return EffectContext(base_image, base_mask, detection_properties)


@pytest.fixture
def effect_context_offset(
    detection_properties: PartProperties,
    base_image_offset: Image,
    base_mask_offset: MaskImage,
):
    return EffectContext(
        base_image_offset, base_mask_offset, detection_properties
    )


@pytest.fixture
def effect_context_blanket(
    detection_properties: PartProperties,
    base_image: Image,
    base_mask_blanket: MaskImage,
):
    return EffectContext(base_image, base_mask_blanket, detection_properties)


@pytest.fixture
def effect_context_fully(
    detection_properties: PartProperties,
    base_image: Image,
    mask_full: MaskImage,
):
    return EffectContext(base_image, mask_full, detection_properties)

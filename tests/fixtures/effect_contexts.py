import pytest

from censor_engine._typing import Image, MaskImage
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.effects.schemas import EffectContext


@pytest.fixture
def effect_context(
    detection_properties: PartProperties,
    base_image: Image,
    base_mask: MaskImage,
):
    return EffectContext(base_image, base_mask, detection_properties)

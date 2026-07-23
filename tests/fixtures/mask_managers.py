import pytest

from censor_engine.models.core.detection_part._mask_manager import MaskManager
from censor_engine.models.libraries.masks.schemas import MaskContext


@pytest.fixture
def mask_manager(mask_context: MaskContext) -> MaskManager:
    return MaskManager("Bar", None, mask_context)

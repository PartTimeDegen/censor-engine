from pathlib import Path

import pytest

from censor_engine.models.libraries.detectors.schemas import AbsoluteBBox
from censor_engine.models.libraries.masks.enums import MaskType
from censor_engine.models.libraries.masks.schemas import MaskContext
from tests.helpers.image_test_handlers import general_image_library_test
from tests.helpers.masks import get_masks

file_path = Path(__file__)


@pytest.mark.parametrize("shape_cls", get_masks(MaskType.BASIC))
class TestShapes:
    parent_folder = Path("shapes")

    def test_standard(self, shape_cls, mask_context_no_part: MaskContext):
        general_image_library_test(
            shape_cls,
            mask_context_no_part,
            file_path=file_path,
            method_used="generate_mask",
            prefix=self.parent_folder / "standard",
        )

    def test_wide(
        self,
        shape_cls,
        mask_context_no_part: MaskContext,
        bbox_long_hor: AbsoluteBBox,
    ):
        mask_context_no_part.part_properties.bbox = bbox_long_hor
        general_image_library_test(
            shape_cls,
            mask_context_no_part,
            file_path=file_path,
            method_used="generate_mask",
            prefix=self.parent_folder / "wide",
        )

    def test_tall(
        self,
        shape_cls,
        mask_context_no_part: MaskContext,
        bbox_long_vert: AbsoluteBBox,
    ):
        mask_context_no_part.part_properties.bbox = bbox_long_vert
        general_image_library_test(
            shape_cls,
            mask_context_no_part,
            file_path=file_path,
            method_used="generate_mask",
            prefix=self.parent_folder / "tall",
        )

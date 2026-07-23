from pathlib import Path

import pytest

from censor_engine.models.libraries.detectors.schemas import AbsoluteBBox
from censor_engine.models.libraries.masks.enums import MaskType
from censor_engine.models.libraries.masks.schemas import MaskContext
from tests.helpers.image_test_handlers import general_image_library_test
from tests.helpers.masks import get_masks

file_path = Path(__file__)


@pytest.mark.parametrize("shape_cls", get_masks(MaskType.BASIC))
class TestMechanisms:
    class TestThickness:
        parent_folder = Path("thickness")

        def test_thickness(self, shape_cls, mask_context_no_part: MaskContext):
            """
            TODO: THIS IS EXPERIMENTAL
            """
            mask_context_no_part.settings.thickness = 0.5
            general_image_library_test(
                shape_cls,
                mask_context_no_part,
                file_path=file_path,
                method_used="_internal_generate_mask",
                prefix=self.parent_folder / "base",
            )

        def test_when_long(
            self,
            shape_cls,
            mask_context_no_part: MaskContext,
            bbox_long_vert: AbsoluteBBox,
        ):
            """
            TODO: THIS IS EXPERIMENTAL
            """
            mask_context_no_part.settings.thickness = 0.5
            mask_context_no_part.part_properties.bbox = bbox_long_vert
            general_image_library_test(
                shape_cls,
                mask_context_no_part,
                file_path=file_path,
                method_used="_internal_generate_mask",
                prefix=self.parent_folder / "long",
            )

        def test_when_wide(
            self,
            shape_cls,
            mask_context_no_part: MaskContext,
            bbox_long_hor: AbsoluteBBox,
        ):
            """
            TODO: THIS IS EXPERIMENTAL
            """
            mask_context_no_part.settings.thickness = 0.5
            mask_context_no_part.part_properties.bbox = bbox_long_hor
            general_image_library_test(
                shape_cls,
                mask_context_no_part,
                file_path=file_path,
                method_used="_internal_generate_mask",
                prefix=self.parent_folder / "wide",
            )

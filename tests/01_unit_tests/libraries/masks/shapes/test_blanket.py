from pathlib import Path

import pytest

from censor_engine.models.libraries.masks.enums import MaskType
from censor_engine.models.libraries.masks.schemas import MaskContext
from tests.helpers.image_test_handlers import general_image_library_test
from tests.helpers.masks import get_masks

file_path = Path(__file__)


@pytest.mark.parametrize("shape_cls", get_masks(MaskType.BLANKET))
class TestShapes:
    parent_folder = Path("shape")

    def test_standard(
        self, shape_cls, mask_context_inline_two_parts: MaskContext
    ):
        general_image_library_test(
            shape_cls,
            mask_context_inline_two_parts,
            file_path=file_path,
            method_used="_internal_generate_mask",
            prefix=self.parent_folder / "standard",
        )

    def test_single_part(
        self, shape_cls, mask_context_single_part: MaskContext
    ):
        general_image_library_test(
            shape_cls,
            mask_context_single_part,
            file_path=file_path,
            method_used="_internal_generate_mask",
            prefix=self.parent_folder / "single_part",
        )

    def test_diagonal(self, shape_cls, mask_context_diagonal: MaskContext):
        general_image_library_test(
            shape_cls,
            mask_context_diagonal,
            file_path=file_path,
            method_used="_internal_generate_mask",
            prefix=self.parent_folder / "diagonal",
        )

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import cv2
import pytest

from censor_engine.libraries.masks.bar import Bar
from censor_engine.models.libraries.masks.bar_structs import (
    BarInfo,
    BoundingInfo,
)
from censor_engine.models.libraries.masks.enums import MaskType
from censor_engine.models.libraries.masks.schemas import MaskContext
from tests.helpers.image_test_handlers import general_image_library_test
from tests.helpers.masks import get_masks
from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)


@pytest.fixture
def bounding_info(mask_context_inline_two_parts):
    mask_obj = Mask()
    mask_obj.joint_mask = "JointEllipse"
    new_mask = mask_obj.generate_joint_mask(mask_context_inline_two_parts)
    cnt, _ = cv2.findContours(
        new_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    input_contours = max(cnt, key=cv2.contourArea)

    bi = BarInfo(False, False, False, False)
    return BoundingInfo.create_bounding_info(
        contours=input_contours, bar_info=bi
    )


@pytest.mark.parametrize("shape_cls", get_masks(MaskType.BAR))
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

    def test_vertical_stack(
        self, shape_cls, mask_context_vertical: MaskContext
    ):
        general_image_library_test(
            shape_cls,
            mask_context_vertical,
            file_path=file_path,
            method_used="_internal_generate_mask",
            prefix=self.parent_folder / "vertical_stack",
        )

    def test_quad_square(
        self, shape_cls, mask_context_quad_square: MaskContext
    ):
        general_image_library_test(
            shape_cls,
            mask_context_quad_square,
            file_path=file_path,
            method_used="_internal_generate_mask",
            prefix=self.parent_folder / "quad_square",
        )


class TestModelInternals:
    parent_folder = Path("bar_edge_cases")

    def test_save_angle(
        self,
        mask_context_three_triangle_top: MaskContext,
        mask_context_three_triangle_bottom: MaskContext,
    ):
        # Angle Persistance
        old_uuid = uuid4()
        mask_context_three_triangle_top.file_uuid = old_uuid
        mask_context_three_triangle_bottom.file_uuid = old_uuid

        top_mask = Bar()._internal_generate_mask(
            deepcopy(mask_context_three_triangle_top)
        )
        bottom_mask = Bar()._internal_generate_mask(
            deepcopy(mask_context_three_triangle_bottom)
        )
        mask_both = cv2.add(top_mask, bottom_mask)

        handle_test_data(
            test_name=str(self.parent_folder / "angle_persistance"),
            image=mask_both,
            file_path=file_path,
        )

        # Angle Reset
        new_uuid = uuid4()
        mask_context_three_triangle_bottom.file_uuid = new_uuid
        general_image_library_test(
            Bar,  # type: ignore
            mask_context_three_triangle_bottom,
            file_path=file_path,
            method_used="_internal_generate_mask",
            prefix=str(self.parent_folder / "angle_reset"),
            include_shape_name=False,
        )

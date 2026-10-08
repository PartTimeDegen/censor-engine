import numpy as np
import pytest

from censor_engine._typing import MaskImage
from censor_engine.models.core.detection_part._mask_manager import MaskManager
from censor_engine.models.libraries.masks.enums import MaskType
from censor_engine.models.libraries.masks.schemas import MaskContext
from tests.fixtures._dimensions import CANVAS_SIZE


@pytest.fixture
def mask_manager(mask_context_no_part: MaskContext) -> MaskManager:
    return MaskManager(
        "Box",
        None,
        mask_context_no_part,
    )


class TestMaskManager:
    def test_working(self, mask_context_quad_square: MaskContext):
        MaskManager(
            "Box",
            None,
            mask_context_quad_square,
        )

    class TestPostInitVariables:
        def test_working(self, mask_context_no_part: MaskContext):
            mm = MaskManager(
                "Box",
                None,
                mask_context_no_part,
            )

            # Created
            assert mm.image_shape == (CANVAS_SIZE, CANVAS_SIZE)
            assert len(mm.layers_of_mask) == 1

            # Generated
            assert type(mm._obj_mask) == type(
                MaskManager.get_mask_class(mm.mask_name)
            )
            assert type(mm._obj_mask_protected) == type(mm._obj_mask)

        def test_with_protected(self, mask_context_no_part: MaskContext):
            mm = MaskManager(
                "Box",
                "Circle",
                mask_context_no_part,
            )

            # Created
            assert mm.image_shape == (CANVAS_SIZE, CANVAS_SIZE)
            assert len(mm.layers_of_mask) == 1

            # Generated
            assert type(mm._obj_mask) == type(
                MaskManager.get_mask_class(mm.mask_name)
            )
            assert type(mm._obj_mask_protected) == type(
                MaskManager.get_mask_class(mm.protection_mask_name)  # type: ignore
            )
            assert type(mm._obj_mask_protected) != type(mm._obj_mask)

    class TestMethods:
        def test_get_mask_class(self, mask_manager: MaskManager):
            mask_class = mask_manager.get_mask_class("Box")

            assert type(mask_class).__name__ == "Box"
            assert mask_class.mask_type == MaskType.BASIC

        def test_add_to_current_mask(
            self,
            mask_manager: MaskManager,
            mask_two_parts_inline: MaskImage,
        ):
            base_mask = mask_manager.current_mask.copy()
            mask_manager.add_to_current_mask(mask_two_parts_inline)

            assert not np.array_equal(base_mask, mask_manager.current_mask)
            assert not np.array_equal(
                mask_two_parts_inline, mask_manager.current_mask
            )

        def test_subtract_from_current_mask(self, mask_manager: MaskManager):
            mask_manager.subtract_from_current_mask(mask_manager.current_mask)

            assert np.count_nonzero(mask_manager.current_mask) == 0

        def test_compile_base_masks(
            self, mask_manager: MaskManager, mask_two_parts_inline: MaskImage
        ):
            mask_manager.layers_of_mask += [mask_two_parts_inline]
            base_mask = mask_manager.current_mask.copy()

            assert len(mask_manager.layers_of_mask) == 2
            mask_manager.compile_base_masks()
            assert not np.array_equal(base_mask, mask_manager.current_mask)
            assert not np.array_equal(
                mask_two_parts_inline, mask_manager.current_mask
            )

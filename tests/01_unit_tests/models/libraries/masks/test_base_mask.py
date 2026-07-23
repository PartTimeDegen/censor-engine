from unittest.mock import Mock, patch

import numpy as np
import pytest

from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.libraries.masks._base_mask import Mask
from censor_engine.models.libraries.masks.enums import MaskType
from censor_engine.models.libraries.masks.schemas import MaskContext


@pytest.fixture
def mask() -> Mask:
    mask = Mask()
    mask.base_mask = "Box"
    mask.joint_mask = "JointBox"
    mask.single_mask = "Box"
    mask.mask_type = MaskType.BASIC
    return mask


class TestBaseMask:
    class TestCoreMethods:
        def test_pre_process(
            self,
            mask_context_inline_two_parts: MaskContext,
            mask: Mask,
        ):
            # Non-Bar Case
            mc, _ = mask._pre_process(mask_context_inline_two_parts)
            assert mc == mask_context_inline_two_parts

            # Bar Case
            joint_mask = MaskRegistry.get("JointBox")().generate_mask(
                mask_context_inline_two_parts
            )

            mask.mask_type = MaskType.BAR
            pp_mc, _ = mask._pre_process(mask_context_inline_two_parts)
            assert np.array_equal(
                pp_mc.mask,
                joint_mask,
            )

        def test_generate(
            self,
            mask_context_inline_two_parts: MaskContext,
            mask: Mask,
        ):
            with pytest.raises(NotImplementedError):
                mask.generate_mask(mask_context_inline_two_parts)

        def test_internal_generate(
            self,
            mask_context_inline_two_parts: MaskContext,
            mask: Mask,
        ):
            # Base
            with patch.object(
                type(mask),
                "generate_mask",
                new_callable=Mock,
                return_value=mask_context_inline_two_parts.mask,
            ):
                output = mask._internal_generate_mask(
                    mask_context_inline_two_parts
                )
                assert np.array_equal(
                    output, mask_context_inline_two_parts.mask
                )

                # Hollow
                mask_context_inline_two_parts.settings.thickness = 0.5
                with patch.object(
                    type(mask._mechanisms),
                    "hollow_mechanism",
                    new_callable=Mock,
                    return_value=mask_context_inline_two_parts.mask,
                ):
                    output = mask._internal_generate_mask(
                        mask_context_inline_two_parts
                    )
                assert np.array_equal(
                    output, mask_context_inline_two_parts.mask
                )

    class TestAPIHelpers:
        def test_generate_single_mask(
            self,
            mask_context_inline_two_parts: MaskContext,
            mask: Mask,
        ):
            single_mask = MaskRegistry.get("Box")().generate_mask(
                mask_context_inline_two_parts
            )
            assert np.array_equal(
                mask.generate_single_mask(mask_context_inline_two_parts),
                single_mask,
            )

        def test_generate_joint_mask(
            self,
            mask_context_inline_two_parts: MaskContext,
            mask: Mask,
        ):
            joint_mask = MaskRegistry.get("JointBox")().generate_mask(
                mask_context_inline_two_parts
            )
            assert np.array_equal(
                mask.generate_joint_mask(mask_context_inline_two_parts),
                joint_mask,
            )

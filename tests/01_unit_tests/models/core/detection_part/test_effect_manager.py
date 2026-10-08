from pathlib import Path

import cv2

from censor_engine._typing import Image, MaskImage
from censor_engine.models.core.detection_part._effect_manager import (
    EffectManager,
)
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.censors import Censor
from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)


class TestEffectManager:
    def test_working(self):
        EffectManager([], [])

    class TestApplyCensor:
        def test_working(self, effect_context: EffectContext):
            em = EffectManager([], [])
            ec_output = em._apply_censor_effect(effect_context, Censor("Blur"))

            handle_test_data("apply_censor", ec_output.image, file_path)

    class TestApplyListCensors:
        def test_working(
            self,
            base_image: Image,
            mask_two_parts_quad_square: MaskImage,
            detection_properties: PartProperties,
        ):
            em = EffectManager([Censor("Blur"), Censor("Outline")], [])
            image = em.apply_effects_from_list_of_censors(
                base_image,
                mask_two_parts_quad_square,  # NOTE: This has 3 dots, that's the mask
                detection_properties,
            )

            handle_test_data("apply_list_of_censors_working", image, file_path)

        def test_empty(
            self,
            base_image: Image,
            mask_two_parts_quad_square: MaskImage,
            detection_properties: PartProperties,
        ):
            em = EffectManager([], [])
            image = em.apply_effects_from_list_of_censors(
                base_image,
                mask_two_parts_quad_square,
                detection_properties,
            )

            handle_test_data("apply_list_of_censors_empty", image, file_path)

    class TestApplyListReverseCensors:
        def test_working(
            self,
            base_image: Image,
            mask_two_parts_quad_square: MaskImage,
            mask_three_triangle_top: MaskImage,
            detection_properties: PartProperties,
        ):
            em = EffectManager([], [Censor("Blur"), Censor("Outline")])

            multiple_parts_mask = cv2.add(
                mask_two_parts_quad_square,
                mask_three_triangle_top,
            )
            inverse_mask = cv2.bitwise_not(multiple_parts_mask)
            image = em.apply_effects_from_list_of_reverse_censors(
                base_image,
                inverse_mask,  # NOTE: This has 3 dots, that's the mask
                detection_properties,
            )

            handle_test_data(
                "apply_list_of_reverse_censors_working", image, file_path
            )

        def test_empty(
            self,
            base_image: Image,
            mask_two_parts_quad_square: MaskImage,
            detection_properties: PartProperties,
        ):
            em = EffectManager([], [])
            image = em.apply_effects_from_list_of_censors(
                base_image,
                mask_two_parts_quad_square,
                detection_properties,
            )

            handle_test_data(
                "apply_list_of_reverse_censors_empty", image, file_path
            )

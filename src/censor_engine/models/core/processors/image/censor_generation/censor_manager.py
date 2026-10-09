from functools import reduce

import cv2

from censor_engine._typing import Image
from censor_engine.models.core.detection_part.detection_part import Part


class CensorManager:
    def _apply_censors_from_list_of_parts(
        self, input_image: Image, part: Part
    ):
        return part.effect_manager.apply_effects_from_list_of_censors(
            input_image,
            part.mask_manager.current_mask,
            part.properties,
        )

    def _handle_reverse_censor(
        self,
        input_image: Image,
        list_of_parts: list[Part],
    ) -> Image:
        if (
            not list_of_parts
            or not list_of_parts[0].config.image.reverse_censor
        ):
            return input_image

        # Create Inverse Mesh
        masks = [part.mask_manager.current_mask for part in list_of_parts]
        combined_masks = reduce(cv2.add, masks)  # type: ignore
        inverse_combined_mask = cv2.bitwise_not(combined_masks)  #  type: ignore

        # Use Effect Manager from First Part
        effect_manager = list_of_parts[0].effect_manager
        return effect_manager.apply_effects_from_list_of_reverse_censors(
            input_image,
            inverse_combined_mask,  # type: ignore
            list_of_parts[0].properties,
        )

    def _handle_normal_censor(
        self,
        input_image: Image,
        list_of_parts: list[Part],
    ) -> Image:
        return reduce(
            self._apply_censors_from_list_of_parts,
            list_of_parts,
            input_image,
        )

    def run_censor_generation_pipeline(
        self,
        input_image: Image,
        list_of_parts: list[Part],
    ) -> Image:
        reverse_image = self._handle_reverse_censor(input_image, list_of_parts)
        return self._handle_normal_censor(reverse_image, list_of_parts)

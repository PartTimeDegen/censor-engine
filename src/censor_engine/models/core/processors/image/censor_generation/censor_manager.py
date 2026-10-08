from functools import reduce

import cv2

from censor_engine._typing import Image
from censor_engine.models.core.detection_part.detection_part import Part


class CensorManager:
    def _handle_reverse_censor(
        self,
        input_image: Image,
        list_of_parts: list[Part],
    ) -> Image:
        if (
            not list_of_parts
            # or not list_of_parts[0].config.image.reverse_censor
        ):
            return input_image

        masks = [part.mask_manager.current_mask for part in list_of_parts]

        combined_masks = reduce(cv2.add, masks)  # type: ignore

        return combined_masks

    def _handle_normal_censor(
        self,
        input_image: Image,
        list_of_parts: list[Part],
    ) -> Image:
        return input_image

    def run_censor_generation_pipeline(
        self,
        input_image: Image,
        list_of_parts: list[Part],
    ) -> Image:
        reverse_image = self._handle_reverse_censor(input_image, list_of_parts)
        return self._handle_normal_censor(reverse_image, list_of_parts)

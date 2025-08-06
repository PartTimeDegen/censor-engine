from abc import ABC, abstractmethod
from typing import Literal
import cv2

from censor_engine.detected_part import Part
from censor_engine.models.enums import StyleType
from .mixin_contour_masking import MixinContourMasking
from .mixin_image_blending import MixinImageBlending
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask, ProcessedImage


class Style(ABC, MixinContourMasking, MixinImageBlending):
    # Information
    style_type: StyleType = StyleType.INVALID
    default_linetype = cv2.LINE_AA

    # Supporting Information
    force_png: bool = False
    default_linetype: int = cv2.LINE_AA
    using_reverse_censor: bool = False

    def _merge_processed_to_input_image(
        self,
        image: Image,
        mask: Mask,
        processed_image: ProcessedImage,
        fade_width: int = 0,
        gradient_mode: Literal["linear", "gaussian"] = "linear",
        mask_thickness: int = -1,
    ) -> ProcessedImage:
        if fade_width > 0:
            return self.blend_with_fade(
                image,
                processed_image,
                mask,
                fade_width,
                gradient_mode=gradient_mode,
                mask_thickness=mask_thickness,
            )

        return self.apply_hard_mask(image, processed_image, mask)

    def internal_run_style(
        self,
        image: Image,
        contours: list[Contour],
        mask: Mask,
        part: Part | None,  # None is for Reverse Censor
        mask_thickness: int = -1,
        fade_width: int = 0,
        gradient_mode: Literal["linear", "gaussian"] = "linear",
        **kwargs,
    ) -> ProcessedImage:
        processed_image = self.apply_style(
            image,
            mask,
            contours,
            part,
            **kwargs,
        )

        working_image = self._merge_processed_to_input_image(
            image,
            mask,
            processed_image,
            fade_width,
            gradient_mode,
            mask_thickness,
        )

        return working_image

    @abstractmethod
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part | None,
        thickness: int = -1,
        *args,
        **kwargs,
    ) -> ProcessedImage:
        raise NotImplementedError

    def change_linetype(self, enable_aa: bool) -> None:
        if enable_aa:
            self.default_linetype = cv2.LINE_AA
        else:
            self.default_linetype = cv2.LINE_4

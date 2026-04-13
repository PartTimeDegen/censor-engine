import cv2

from censor_engine.detected_part import Part
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import ColourEffect
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, TypeMask


# Colour Filters
@EffectRegistry.register()
class Greyscale(ColourEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        alpha: float = 1,
    ) -> Image:
        # Get TypeMask
        mask_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        mask_image = cv2.cvtColor(
            mask_image,
            cv2.COLOR_GRAY2BGR,
        )
        grey = mask_image
        return cv2.addWeighted(grey, alpha, image, 1 - alpha, 0)

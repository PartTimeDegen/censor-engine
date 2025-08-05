import cv2

from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import ColourStyle
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image


# Colour Filters
@StyleRegistry.register()
class Greyscale(ColourStyle):
    def apply_style(
        self,
        image: Image,
        contour: Contour,
        alpha=1,
    ) -> Image:
        # Get Mask
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

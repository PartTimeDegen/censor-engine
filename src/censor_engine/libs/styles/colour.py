import cv2

from censor_engine.models.styles import ColourStyle
from censor_engine.typing import CVImage


# Colour Filters
class Greyscale(ColourStyle):
    style_name: str = "greyscale"

    def apply_style(
        self,
        image: CVImage,
        contour,
        alpha=1,
    ) -> CVImage:
        # Get Mask
        mask_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        mask_image = cv2.cvtColor(
            mask_image,
            cv2.COLOR_GRAY2BGR,
        )
        grey = self.draw_effect_on_mask([contour], mask_image, image)
        return cv2.addWeighted(grey, alpha, image, 1 - alpha, 0)


effects = {
    "greyscale": Greyscale,
}

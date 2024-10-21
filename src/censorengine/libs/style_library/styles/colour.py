from dataclasses import dataclass
import cv2
import numpy as np

from censorengine.lib_models import styles as ip
from censorengine.lib_models.styles import ColourStyle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import CVImage


@dataclass
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
        grey = self.draw_effect([contour], mask_image, image)
        return cv2.addWeighted(grey, alpha, image, 1 - alpha, 0)


def cutout(image, contours: tuple):
    # Add Alpha Channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    mask_zeros = np.ones(image.shape, dtype=np.uint8) * 255
    mask_filter = cv2.drawContours(
        mask_zeros,
        contours[0],
        -1,
        (0, 0, 0, 0),
        -1,
        hierarchy=contours[1],
    )

    return np.where(
        mask_filter == (0, 0, 0, 0),
        mask_filter,
        image,
    )


effects = {
    "greyscale": Greyscale,
}

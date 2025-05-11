import cv2
import numpy as np

from censor_engine.models.styles import TransparentStyle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censor_engine.typing import CVImage


class Cutout(TransparentStyle):
    style_name: str = "cutout"
    force_png: bool = True

    def apply_style(
        self,
        image: "CVImage",
        contour,
    ) -> "CVImage":
        # Add Alpha Channel
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        mask_zeros = np.ones(image.shape, dtype=np.uint8) * 255
        mask_filter = cv2.drawContours(
            mask_zeros,  # type: ignore
            contour[0],
            -1,
            (0, 0, 0, 0),  # type: ignore
            -1,
            hierarchy=contour[1],
        )

        return np.where(
            mask_filter == (0, 0, 0, 0),
            mask_filter,
            image,
        )


effects = {
    "cutout": Cutout,
}

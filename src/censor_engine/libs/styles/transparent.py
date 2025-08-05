import cv2
import numpy as np

from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import TransparentStyle
from typing import TYPE_CHECKING

from censor_engine.models.structs.contours import Contour

if TYPE_CHECKING:
    from censor_engine.typing import Image


@StyleRegistry.register()
class Cutout(TransparentStyle):
    force_png: bool = True

    def apply_style(
        self,
        image: "Image",
        contour: Contour,
    ) -> "Image":
        # Add Alpha Channel
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        mask_zeros = np.ones(image.shape, dtype=np.uint8) * 255
        mask_filter = cv2.drawContours(
            mask_zeros,  # type: ignore
            contour.points,
            -1,
            (0, 0, 0, 0),  # type: ignore
            -1,
            hierarchy=contour.hierarchy,  # type:ignore
        )

        return np.where(
            mask_filter == (0, 0, 0, 0),
            mask_filter,
            image,
        )


@StyleRegistry.register()
class NoCensor(TransparentStyle):
    def apply_style(self, image, contour):
        return image

import cv2
import numpy as np

from censor_engine.detected_part import Part
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import TransparentStyle
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask


@StyleRegistry.register()
class Cutout(TransparentStyle):
    force_png: bool = True

    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        alpha: float = 0.5,
    ) -> Image:
        # Normalize alpha float and clamp
        alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))
        alpha_value = int(alpha * 255)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        elif image.shape[2] == 4:
            pass
        else:
            raise ValueError(
                f"Unsupported number of channels: {image.shape[2]}"
            )

        # Build new alpha channel: 0 where mask is white, else alpha_value
        black_pixels = np.all(mask == 000, axis=2)
        new_alpha = np.full(image.shape[:2], alpha_value, dtype=np.uint8)
        new_alpha[black_pixels] = 0

        # Replace alpha channel in the image
        image[:, :, 3] = new_alpha

        return image


@StyleRegistry.register()
class NoCensor(TransparentStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
    ):
        return image

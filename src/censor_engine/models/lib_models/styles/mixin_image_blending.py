from typing import Literal
import cv2
import numpy as np

from censor_engine.models.structs.meta_structs import Mixin
from censor_engine.typing import Mask, Image


class MixinImageBlending(Mixin):
    def blend_with_fade(
        self,
        image: Image,
        image_with_effect: Image,
        mask: Mask,
        fade_width: int,
        gradient_mode: Literal["linear", "gaussian"] = "linear",
        mask_thickness: int = -1,
    ) -> Image:
        if gradient_mode == "gaussian":
            dist_transform = cv2.GaussianBlur(
                mask.astype(np.float32),  # type: ignore
                (0, 0),
                fade_width / 2,
            )
            dist_transform = cv2.normalize(
                dist_transform,
                None,  # type: ignore
                0,
                1.0,
                cv2.NORM_MINMAX,
            )
            dist_transform = 1 - dist_transform
        else:
            dist_transform = cv2.distanceTransform(
                cv2.bitwise_not(mask), cv2.DIST_L2, 5
            )
            dist_transform = np.clip(dist_transform / fade_width, 0, 1)

        alpha = (1 - dist_transform).astype(np.float32)[..., None]  # type: ignore
        overlay_f = image_with_effect.astype(np.float32)
        image_f = image.astype(np.float32)
        blended = (overlay_f * alpha + image_f * (1 - alpha)).astype(np.uint8)
        return blended

    def apply_hard_mask(
        self,
        image: Image,
        image_with_effect: Image,
        mask: Mask,
    ) -> Image:
        return np.where(mask == 255, image_with_effect, image)

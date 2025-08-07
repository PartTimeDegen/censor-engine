from typing import Literal

import cv2
import numpy as np

from censor_engine.models.structs.meta_structs import Mixin
from censor_engine.typing import Image, Mask


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
        # Ensure both images have the same number of channels
        if image.shape[2] == 3 and image_with_effect.shape[2] == 4:
            # Upgrade base image to 4 channels by adding opaque alpha
            alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
            image = np.dstack((image, alpha))

        elif image.shape[2] == 4 and image_with_effect.shape[2] == 3:
            # Upgrade effect image to 4 channels by adding opaque alpha
            alpha = np.full(image_with_effect.shape[:2], 255, dtype=np.uint8)
            image_with_effect = np.dstack((image_with_effect, alpha))

        # Create a single-channel boolean mask where all 3 channels are 255 (white)
        single_channel_mask = (
            np.all(mask == 255, axis=2) if mask.ndim == 3 else mask == 255
        )

        # Broadcast to shape (H, W, channels)
        mask_expanded = single_channel_mask[..., None]  # shape (H, W, 1)

        return np.where(mask_expanded, image_with_effect, image)

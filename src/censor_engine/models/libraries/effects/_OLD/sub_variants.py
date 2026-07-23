import cv2
import numpy as np

from censor_engine.models.enums import EffectType
from censor_engine.models.structs.colours import Colour
from censor_engine.typing import Image, Mask

from .base import Effect


class TransparentEffect(Effect):
    effect_type: EffectType = EffectType.TRANSPARENCY
    force_png: bool = True  # Needed for alpha channel to work


class BlurEffect(Effect):
    effect_type: EffectType = EffectType.BLUR

    def normalise_factor(
        self,
        image: Image,
        factor: float,
    ) -> int | float:
        # factor = 1, size = 1
        # factor = 100, size = minimum_size/blur_cap
        blur_cap = 1
        blur_rate = 0.25
        factor_cap = 100

        minimum_size = min(
            image.shape[0],
            image.shape[1],
        )

        normalised_size = minimum_size / blur_cap
        normalised_factor = factor / factor_cap

        new_factor = int(normalised_size * normalised_factor * blur_rate)

        minimum_factor = 2
        if new_factor > minimum_factor:
            return int(factor)

        return new_factor

    def apply_factor(
        self,
        image: Image,
        factor: float,
    ) -> tuple[int, int]:
        # Fixing Strength
        factor = factor * 4 + 1

        factor = self.normalise_factor(image, factor)

        if factor < 1:
            factor = 1
        elif factor % 2 == 0:
            factor += 1

        image_ratio = (max(image.shape) - min(image.shape)) / min(image.shape)

        factor_ratio = factor / image_ratio
        return (
            int(factor_ratio * min(image.shape)),  # Min Factor
            int(factor_ratio * max(image.shape)),  # Max Factor
        )


class PixelateEffect(BlurEffect):
    effect_type: EffectType = EffectType.PIXELATION


class NoiseEffect(BlurEffect):
    effect_type: EffectType = EffectType.NOISE


class OverlayEffect(Effect):
    effect_type: EffectType = EffectType.OVERLAY

    def _apply_mask_as_overlay(
        self,
        image: Image,
        mask: Mask,
        colour: Colour,
        alpha: float,
    ) -> Image:
        overlay = image.copy()

        # Create a single-channel boolean mask from any RGB mask channel
        mask_bool = mask[:, :, 0] > 0

        if not np.any(mask_bool):
            return overlay  # Nothing to do if mask is empty

        # Create an array of mask (H, W, 3) with the target color
        color_array = np.full_like(image, colour.value, dtype=image.dtype)

        # Alpha blending only on masked region
        if alpha < 1.0:
            # Blend only in masked region
            overlay[mask_bool] = (
                (1 - alpha) * image[mask_bool] + alpha * color_array[mask_bool]
            ).astype(image.dtype)
        else:
            # Hard color replace in masked region
            overlay[mask_bool] = color_array[mask_bool]

        return overlay


class ColourEffect(Effect):
    effect_type: EffectType = EffectType.COLOUR


class StyliseEffect(Effect):
    effect_type: EffectType = EffectType.STYLISATION


class TextEffect(Effect):
    effect_type: EffectType = EffectType.TEXT


class DevEffect(TextEffect):
    effect_type: EffectType = EffectType.DEV


class EdgeDetectionEffect(Effect):
    effect_type: EffectType = EffectType.EDGE_DETECTION

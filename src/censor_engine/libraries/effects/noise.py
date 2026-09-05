import math
from functools import lru_cache

import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.libraries.effects._default_args import (
    OFFSET,
    SEED,
    STRENGTH,
)
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import NoiseEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


@EffectRegistry.register()
class ChromaticAberration(NoiseEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        offset: float = OFFSET,  # Percent
        angle: int = -45,
    ) -> Image:
        # Create a copy for the noise effect
        offset *= 1 / 1000  # Scaler
        offset *= min(effect_context.image_shape)
        offset = -int(offset)

        # Correct angle components for x and y directions
        comp_x = math.cos(math.radians(angle))  # Horizontal shift
        comp_y = math.sin(math.radians(angle))  # Vertical shift

        # Split into B, G, R channels
        channels = cv2.split(effect_context.image)
        channels = list(channels)

        # Loop through each color channel (B=0, G=1, R=2)
        for i, channel in enumerate(channels):
            # Calculate the shift amount per channel
            dx = int(offset * (i + 1) * comp_x)
            dy = int(offset * (i + 1) * comp_y)

            # Create the affine transformation matrix for shifting
            matrix_moment = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore

            # Apply the shift using warpAffine (faster than np.roll)

            channels[i] = cv2.warpAffine(  # type: ignore
                channel,
                matrix_moment,  # type: ignore
                (channel.shape[1], channel.shape[0]),
                borderMode=cv2.BORDER_REFLECT,
            )  # type: ignore

        # Merge the shifted channels back
        return cv2.merge(tuple(channels))  # type: ignore


@EffectRegistry.register()
class CentricChromaticAberration(NoiseEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        offset: float = OFFSET,
        use_mask_centre: bool = True,
    ) -> Image:
        image = effect_context.image
        h, w = effect_context.image_shape

        # Convert offset to pixels
        offset *= min(h, w) / 1000.0

        if use_mask_centre:
            cx, cy = effect_context.mask_info.minimum_area.centre
        else:
            cx = w / 2.0
            cy = h / 2.0

        # Coordinate grid
        x, y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

        # Direction from center
        dx = x - cx
        dy = y - cy

        dist = np.sqrt(dx * dx + dy * dy)
        dist[dist == 0] = 1.0

        # Normalize
        nx = dx / dist
        ny = dy / dist

        channels = cv2.split(image)
        shifted = []

        # Amount each channel moves.
        # Blue inward, Green stationary, Red outward
        scales = (-offset, 0.0, offset)

        for channel, scale in zip(channels, scales, strict=False):
            map_x = x - nx * scale
            map_y = y - ny * scale

            shifted.append(
                cv2.remap(
                    channel,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
            )

        return cv2.merge(shifted)  # type: ignore


@EffectRegistry.register()
class Noise(NoiseEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        intensity: float = 1,
        grain_size: int = 1,
        seed: int = SEED,
    ) -> Image:
        # Set Seed
        np.random.seed(seed)

        # Get Size
        h, w = effect_context.image_shape

        noise = np.random.normal(
            0,
            255 * intensity,
            (h // grain_size, w // grain_size, 3),
        ).astype(np.uint8)  # type: ignore

        return cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)  # type: ignore


@EffectRegistry.register()
class DeNoise(NoiseEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        strength: int = STRENGTH,
    ) -> Image:
        return cv2.fastNlMeansDenoisingColored(
            effect_context.image,
            h=strength,
        )  # type: ignore


@EffectRegistry.register()
class Glitch(NoiseEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        strength: int = STRENGTH,
    ) -> Image:
        # TODO: This is just ChromoAbb
        b, g, r = cv2.split(effect_context.image)

        rows, cols = b.shape

        M = np.float32([[1, 0, strength], [0, 1, 0]])  # type: ignore

        r_shifted = cv2.warpAffine(r, M, (cols, rows))  # type: ignore
        b_shifted = cv2.warpAffine(b, M, (cols, rows))  # type: ignore

        return cv2.merge([b_shifted, g, r_shifted])  # type: ignore


@EffectRegistry.register()
class Dither(NoiseEffect):
    @staticmethod
    def _generate_bayer(size: int):
        if size == 1:
            return np.array([[0]])

        bayer = np.array([[0]])

        while bayer.shape[0] < size:
            bayer = np.block(
                [[4 * bayer, 4 * bayer + 2], [4 * bayer + 3, 4 * bayer + 1]]
            )

        return bayer

    @lru_cache(maxsize=8)
    @staticmethod
    def _bayer_threshold(size: int):
        bayer = Dither._generate_bayer(size)
        return (bayer + 0.5) / (size * size)

    def _ordered_dither_colour(
        self, image: Image, bayer_power: int, levels: int
    ):
        size = 2**bayer_power
        bayer = Dither._bayer_threshold(size)
        h, w, _ = image.shape
        threshold = np.tile(bayer, (h // size + 1, w // size + 1))

        threshold = threshold[:h, :w]
        img = image.astype(np.float32) / 255.0
        threshold -= 0.5
        img += threshold[..., None] / levels
        img = np.round(img * (levels - 1)) / (levels - 1)

        return np.clip(
            img * 255,
            0,
            255,
        ).astype(np.uint8)

    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        bayer_power: int = 3,
        levels: int = 4,
    ) -> Image:

        image = effect_context.image

        return self._ordered_dither_colour(
            image,
            bayer_power,
            levels,
        )

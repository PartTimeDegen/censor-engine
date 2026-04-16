import math

import cv2
import numpy as np

from censor_engine.api.effects import EffectContext
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import NoiseEffect
from censor_engine.typing import ProcessedImage


@EffectRegistry.register()
class ChromaticAberration(NoiseEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        offset: float = 10,  # Percent
        angle: int = -45,
    ) -> ProcessedImage:
        # Create a copy for the noise effect
        offset *= 1 / 1000  # Scaler
        offset *= min(effect_context.shape[:2])
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
        channels = tuple(channels)
        return cv2.merge(channels)  # type: ignore

        # Apply the effect to the masked area


@EffectRegistry.register()
class CentricChromaticAberration(NoiseEffect):
    # FIXME: This doesn't give centric aberration

    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        offset: float = 10,
        blur: int = 0,
    ) -> ProcessedImage:
        offset *= 1 / 1000  # Scaler
        offset *= min(effect_context.mask.shape[:2])
        offset = -int(offset)

        # Step 2: Simulate outward shift (could be improved with pixelwise
        #         later)
        dx_vector = 1.0
        dy_vector = 1.0
        norm = math.hypot(dx_vector, dy_vector)
        comp_x = dx_vector / norm
        comp_y = dy_vector / norm

        # Step 3: Shift each channel
        channels = list(cv2.split(effect_context.image))
        for i, channel in enumerate(channels):
            dx = int(offset * (i + 1) * comp_x)
            dy = int(offset * (i + 1) * comp_y)

            m = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore
            channels[i] = cv2.warpAffine(
                channel,
                m,  # type: ignore
                (channel.shape[1], channel.shape[0]),
                borderMode=cv2.BORDER_REFLECT,
            )

        effect_context.image = cv2.merge(tuple(channels))  # type: ignore

        # Step 4: Optional blur
        if blur > 0:
            ksize = max(1, int(blur) // 2 * 2 + 1)  # Ensure it's odd
            effect_context.image = cv2.GaussianBlur(
                effect_context.image, (ksize, ksize), 0
            )

        # Step 5: TypeMask the result onto the original image
        return effect_context.image


@EffectRegistry.register()
class Noise(NoiseEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        intensity: float = 1,
        grain_size: int = 1,
        seed: int = 69,
        coloured: bool = True,
    ) -> ProcessedImage:
        np.random.seed(seed)

        h, w, c = effect_context.shape

        noise = np.random.normal(
            0,
            255 * intensity,
            (h // grain_size, w // grain_size, c),
        ).astype(np.uint8)  # type: ignore
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)  # type: ignore

        if not coloured:
            noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)  # type: ignore
            noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)  # type: ignore
        return noise


@EffectRegistry.register()
class DeNoise(NoiseEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        strength: int = 10,
    ) -> ProcessedImage:
        return cv2.fastNlMeansDenoisingColored(
            effect_context.image,
            h=strength,
        )

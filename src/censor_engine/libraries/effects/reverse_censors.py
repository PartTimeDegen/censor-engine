from typing import Literal

import cv2
import numpy as np

from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import ReverseCensorEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.colours import Colour, TypeColour
from censor_engine.typing import Image


@EffectRegistry.register()
class Glow(ReverseCensorEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        strength: float = 1.0,
        sigma: int = 25,
        colour: TypeColour = "BLACK",
    ) -> Image:  # type: ignore
        # TODO: Tests
        # Get the inverse mask
        inverse_mask = cv2.bitwise_not(effect_context.mask)

        # Convert types
        image_f = effect_context.image.astype(np.float32)
        mask_f = inverse_mask.astype(np.float32) / 255.0

        # Make glow
        glow_mask = cv2.GaussianBlur(mask_f, (0, 0), sigma)  # type: ignore

        # Remove original mask
        outer_glow = np.clip(glow_mask - mask_f, 0.0, 1.0)

        # Add colour
        glow_colour = np.asarray(Colour(colour).value, dtype=np.float32)

        # Alpha controls the glow's opacity
        alpha = np.clip(outer_glow * strength, 0.0, 1.0)[:, :, None]

        # Blend glow colour with the original image
        result = image_f * (1.0 - alpha) + glow_colour * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

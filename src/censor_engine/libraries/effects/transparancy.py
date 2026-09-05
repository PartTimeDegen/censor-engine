import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import TransparentEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


@EffectRegistry.register()
class Cutout(TransparentEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
    ) -> Image:
        image = effect_context.image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        # Build new alpha channel: 0 where mask is white, else alpha_value
        new_alpha = np.zeros(effect_context.image_shape, dtype=np.uint8)

        # Replace alpha channel in the image
        image[:, :, 3] = new_alpha

        return image


@EffectRegistry.register()
class NoCensor(TransparentEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
    ) -> Image:
        return effect_context.image  # type: ignore

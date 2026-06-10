import cv2
import numpy as np

from censor_engine.api.effects import EffectContext
from censor_engine.constant import DIM_COLOUR, DIM_GREY, DIM_RGBA
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import TransparentEffect
from censor_engine.typing import Image, ProcessedImage


@EffectRegistry.register()
class Cutout(TransparentEffect):
    force_png: bool = True

    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
    ) -> Image:
        image = effect_context.image
        if image.ndim == DIM_GREY:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif image.shape[2] == DIM_COLOUR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        elif image.shape[2] == DIM_RGBA:
            pass
        else:
            msg = f"Unsupported number of channels: {image.shape[2]}"
            raise ValueError(
                msg,
            )

        # Build new alpha channel: 0 where mask is white, else alpha_value
        new_alpha = np.zeros(image.shape[:2], dtype=np.uint8)

        # Replace alpha channel in the image
        image[:, :, 3] = new_alpha

        return image


@EffectRegistry.register()
class NoCensor(TransparentEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
    ) -> ProcessedImage:
        return effect_context.image

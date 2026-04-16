import cv2

from censor_engine.api.effects import EffectContext
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import ColourEffect
from censor_engine.typing import ProcessedImage


# Colour Filters
@EffectRegistry.register()
class Greyscale(ColourEffect):
    def apply_effect(self, effect_context: EffectContext) -> ProcessedImage:  # type: ignore
        # Get TypeMask
        mask_image = cv2.cvtColor(
            effect_context.image,
            cv2.COLOR_BGR2GRAY,
        )
        return cv2.cvtColor(
            mask_image,
            cv2.COLOR_GRAY2BGR,
        )

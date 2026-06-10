import cv2

from censor_engine.api.effects import EffectContext
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import StyliseEffect
from censor_engine.typing import ProcessedImage


@EffectRegistry.register()
class Painting(StyliseEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
    ) -> ProcessedImage:
        return cv2.stylization(effect_context.image, sigma_s, sigma_r)  # type: ignore


@EffectRegistry.register()
class Pencil(StyliseEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
        shade_factor: float = 0.2,
        coloured: bool = False,
    ) -> ProcessedImage:
        grey, colour = cv2.pencilSketch(
            effect_context.image,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            shade_factor=shade_factor,
        )  # type: ignore

        return colour if coloured else cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

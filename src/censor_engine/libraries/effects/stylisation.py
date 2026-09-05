import cv2

from censor_engine._typing import Image
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import StylisationEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


@EffectRegistry.register()
class Painting(StylisationEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
    ) -> Image:
        return cv2.stylization(effect_context.image, sigma_s, sigma_r)  # type: ignore


@EffectRegistry.register()
class Pencil(StylisationEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
        shade_factor: float = 0.2,
    ) -> Image:
        grey, colour = cv2.pencilSketch(
            effect_context.image,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            shade_factor=shade_factor,
        )  # type: ignore

        return (
            cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)  # type: ignore
            if effect_context.general_settings.pre_processing.greyscale
            else colour  # type: ignore
        )


@EffectRegistry.register()
class EdgeInk(StylisationEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        threshold_lower: float = 50.0,
        threshold_upper: float = 150.0,
    ) -> Image:
        gray = cv2.cvtColor(effect_context.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold_lower, threshold_upper)
        result = effect_context.image.copy()
        result[edges > 0] = 0
        return result  # type: ignore

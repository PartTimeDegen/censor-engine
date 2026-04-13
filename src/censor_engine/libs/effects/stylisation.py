import cv2

from censor_engine.detected_part import Part
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import StyliseEffect
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, TypeMask


@EffectRegistry.register()
class Painting(StyliseEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
    ) -> Image:
        return cv2.stylization(image, sigma_s, sigma_r)  # type: ignore


@EffectRegistry.register()
class Pencil(StyliseEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        coloured: bool = False,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
        shade_factor: float = 0.2,
    ) -> Image:
        grey, colour = cv2.pencilSketch(
            image,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            shade_factor=shade_factor,
        )  # type: ignore

        return colour if coloured else cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

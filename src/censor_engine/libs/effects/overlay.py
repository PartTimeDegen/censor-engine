import cv2
import numpy as np

from censor_engine.api.effects import EffectContext
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import OverlayEffect
from censor_engine.models.structs.colours import Colour
from censor_engine.typing import ProcessedImage


@EffectRegistry.register()
class MissingEffect(OverlayEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
    ) -> ProcessedImage:
        return effect_context.image


@EffectRegistry.register()
class Overlay(OverlayEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        colour: tuple[int, int, int] | str = "WHITE",
    ) -> ProcessedImage:
        return np.full_like(
            effect_context.image,
            Colour(colour).value,
            dtype=effect_context.image.dtype,
        )


@EffectRegistry.register()
class Outline(OverlayEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        colour: tuple[int, int, int] | str = "WHITE",
        thickness: int = 2,
        softness: int = 0,
    ) -> ProcessedImage:
        colour_obj = Colour(colour)

        # Extract points from your Contour objects
        contours_points = [
            contour.points for contour in effect_context.contours
        ]

        # Draw contours on a copy of the image
        cv2.drawContours(
            effect_context.image,
            contours_points,
            -1,
            colour_obj.value,
            thickness,
            lineType=self.default_linetype,
        )

        if softness > 0:
            ksize = max(3, softness * 2 + 1)
            effect_context.image = cv2.GaussianBlur(
                effect_context.image, (ksize, ksize), 0
            )

        return effect_context.image


@EffectRegistry.register()
class OutlinedOverlay(OverlayEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        colour_box: tuple[int, int, int] | str = "WHITE",
        colour_outline: tuple[int, int, int] | str = "BLACK",
        thickness: int = 2,
        softness: int = 0,
    ) -> ProcessedImage:
        overlay = np.full_like(
            effect_context.image,
            Colour(colour_box).value,
            dtype=effect_context.image.dtype,
        )

        contours_points = [
            contour.points for contour in effect_context.contours
        ]
        cv2.drawContours(
            overlay,
            contours_points,
            -1,
            Colour(colour_outline).value,
            thickness,
            lineType=self.default_linetype,
        )

        if softness > 0:
            ksize = max(3, softness * 2 + 1)
            overlay = cv2.GaussianBlur(overlay, (ksize, ksize), 0)

        return overlay

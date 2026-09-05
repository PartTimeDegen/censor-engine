import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import OverlayEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.colours import Colour


@EffectRegistry.register()
class MissingEffect(OverlayEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
    ) -> Image:
        return effect_context.image


@EffectRegistry.register()
class Overlay(OverlayEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        colour: tuple[int, int, int] | str = "WHITE",
    ) -> Image:
        return np.full_like(
            effect_context.image,
            Colour(colour).value,
            dtype=effect_context.image.dtype,
        )


@EffectRegistry.register()
class Outline(OverlayEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        colour: tuple[int, int, int] | str = "WHITE",
        thickness: int = 2,
        linetype: int = cv2.LINE_AA,
        include_image_borders: bool = False,
    ) -> Image:
        colour_obj = Colour(colour)

        # Extract points from your Contour objects
        contours_points = [
            contour.points for contour in effect_context.contours
        ]
        if not include_image_borders:
            buffer = 2 * thickness
            h, w = effect_context.image_shape

            pts = contours_points[0][:, 0]
            pts[pts[:, 0] == 0, 0] = -buffer
            pts[pts[:, 0] == w - 1, 0] = w - 1 + buffer

            pts[pts[:, 1] == 0, 1] = -buffer
            pts[pts[:, 1] == h - 1, 1] = h - 1 + buffer

        # Draw contours on a copy of the image
        cv2.drawContours(
            effect_context.image,
            contours_points,
            -1,
            colour_obj.value,
            thickness,
            lineType=linetype,
        )

        return effect_context.image


@EffectRegistry.register()
class OutlinedOverlay(OverlayEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        colour_box: tuple[int, int, int] | str = "WHITE",
        colour_outline: tuple[int, int, int] | str = "BLACK",
        thickness: int = 2,
        linetype: int = cv2.LINE_AA,
        include_image_borders: bool = False,
    ) -> Image:
        effect_context.image = Overlay().generate_effect(
            effect_context,
            colour=colour_box,  # type: ignore
        )

        return Outline().generate_effect(
            effect_context,
            colour=colour_outline,  # type: ignore
            thickness=thickness,  # type: ignore
            include_image_borders=include_image_borders,  # type: ignore
            linetype=linetype,
        )

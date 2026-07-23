import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import ColourEffect
from censor_engine.models.libraries.effects.schemas import EffectContext
from censor_engine.structs.colours import Colour


@EffectRegistry.register()
class Greyscale(ColourEffect):
    def generate_effect(  # type: ignore
        self, effect_context: EffectContext
    ) -> Image:
        mask_image = cv2.cvtColor(
            effect_context.image,
            cv2.COLOR_BGR2GRAY,
        )
        return cv2.cvtColor(  # type: ignore
            mask_image,
            cv2.COLOR_GRAY2BGR,
        )


@EffectRegistry.register()
class DuoTone(ColourEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        colour_one: tuple[int, int, int] | str = "BLUE",
        colour_two: tuple[int, int, int] | str = "PURPLE",
    ) -> Image:
        grey = cv2.cvtColor(effect_context.image, cv2.COLOR_BGR2GRAY)
        grey_norm = grey / 255.0

        colour_one_tuple = Colour(colour_one).value
        colour_two_tuple = Colour(colour_two).value

        result = np.zeros_like(effect_context.image, dtype=np.float32)

        for i in range(3):
            result[:, :, i] = (
                grey_norm * colour_two_tuple[i]
                + (1 - grey_norm) * colour_one_tuple[i]
            )

        return result.astype(np.uint8)


@EffectRegistry.register()
class HeatMap(ColourEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        colour_map: int = cv2.COLORMAP_JET,
    ) -> Image:
        grey = cv2.cvtColor(effect_context.image, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(grey, colour_map)  # type: ignore


@EffectRegistry.register()
class Contrast(ColourEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        contrast_alpha: float = 1.5,
        contrast_beta: float = 0,
    ) -> Image:
        return cv2.convertScaleAbs(  # type: ignore
            effect_context.image,
            alpha=contrast_alpha,
            beta=contrast_beta,
        )


@EffectRegistry.register()
class ColourMask(ColourEffect):
    def _string_to_tuple(self, string: str) -> tuple[int, int, int]:
        string = string.strip("( )")
        return tuple(int(x) for x in string.split(","))  # type: ignore

    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        hsv_lower_limit: str = "(60, 60, 60)",
        hsv_upper_limit: str = "(180, 180, 180)",
        use_greyscale: bool = True,
    ) -> Image:
        # https://pythonprogramming.net/color-filter-python-opencv-tutorial/

        # Fix for Python Tuples being Read as Strings
        hsv_lower_limit = self._string_to_tuple(hsv_lower_limit)  # type: ignore
        hsv_upper_limit = self._string_to_tuple(hsv_upper_limit)  # type: ignore

        # Convert Image to HSV for Getting a Better Range
        hsv = cv2.cvtColor(effect_context.image, cv2.COLOR_BGR2HSV)

        # Get Mask of Values
        lb = np.array(hsv_lower_limit)
        ub = np.array(hsv_upper_limit)
        if lb[0] > ub[0]:
            lower_left = np.array([lb[0], lb[1], lb[2]])
            upper_left = np.array([179, ub[1], ub[2]])
            mask_left = cv2.inRange(hsv, lower_left, upper_left)  # type: ignore

            lower_right = np.array([0, lb[1], lb[2]])
            upper_right = np.array([ub[0], ub[1], ub[2]])
            mask_right = cv2.inRange(hsv, lower_right, upper_right)  # type: ignore

            mask = cv2.bitwise_or(mask_left, mask_right)
        else:
            mask = cv2.inRange(hsv, lb, ub)  # type: ignore

        # Get the Area that's in the Mask
        foreground = cv2.bitwise_and(
            effect_context.image,
            effect_context.image,
            mask=mask,
        )

        # Use a Black Background if not using Greyscale
        if not use_greyscale:
            return foreground  # type: ignore

        grey_image = Greyscale().generate_effect(effect_context)
        inv_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(grey_image, grey_image, mask=inv_mask)
        return cv2.add(foreground, background)  # type: ignore


@EffectRegistry.register()
class Posterise(ColourEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        levels: int = 4,
    ) -> Image:
        factor = max(1, 256 // levels)
        return (effect_context.image // factor) * factor


@EffectRegistry.register()
class Negative(ColourEffect):
    def generate_effect(  # type: ignore
        self, effect_context: EffectContext
    ) -> Image:
        return cv2.bitwise_not(effect_context.image)  # type: ignore


@EffectRegistry.register()
class Sepia(ColourEffect):
    def generate_effect(  # type: ignore
        self, effect_context: EffectContext
    ) -> Image:
        kernel = np.array(
            [
                [0.272, 0.534, 0.131],  # B
                [0.349, 0.686, 0.168],  # G
                [0.393, 0.769, 0.189],  # R
            ]
        )

        result = cv2.transform(effect_context.image, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

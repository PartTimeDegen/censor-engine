import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import BlurEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)

from ._default_args import FACTOR


@EffectRegistry.register()
class Blur(BlurEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        factor: float = FACTOR,
    ) -> Image:
        new_factor = int(
            self.helpers.normalise_factor(effect_context.image_shape, factor)
        )
        return cv2.blur(effect_context.image, (new_factor, new_factor))  # type: ignore


@EffectRegistry.register()
class GaussianBlur(BlurEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        factor: float = FACTOR,
    ) -> Image:
        new_factor = int(
            self.helpers.normalise_factor(effect_context.image_shape, factor)
        )
        return cv2.GaussianBlur(
            effect_context.image,  # type: ignore
            (new_factor, new_factor),
            0,
        )


@EffectRegistry.register()
class MedianBlur(BlurEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        factor: float = FACTOR,
    ) -> Image:
        new_factor = int(
            self.helpers.normalise_factor(effect_context.image_shape, factor)
        )
        return cv2.medianBlur(effect_context.image, new_factor)  # type: ignore


@EffectRegistry.register()
class BilateralBlur(BlurEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        distance: float = FACTOR,
        sigma_colour: int = 150,
        sigma_space: int = 150,
    ) -> Image:
        new_distance = int(
            self.helpers.normalise_factor(effect_context.image_shape, distance)
        )
        return cv2.bilateralFilter(
            effect_context.image,
            new_distance,
            sigma_colour,
            sigma_space,
        )  # type: ignore


@EffectRegistry.register()
class MotionBlur(BlurEffect):
    # TODO: Need to Check
    current_angle: int = -45

    def _rotate(self, rotation: int) -> None:
        if rotation > 0:
            type(self).current_angle += 1
        else:
            type(self).current_angle -= 1

        horizontal_threshold = 180
        is_angle = (
            type(self).current_angle >= horizontal_threshold
            or type(self).current_angle <= -horizontal_threshold
        )
        if is_angle:
            type(self).current_angle = 0

    def _apply_factor_to_kernel(self, factor: int, angle: int) -> Image:
        kernel = np.zeros((factor, factor))
        kernel[int((factor - 1) / 2), :] = np.ones(factor)
        kernel = kernel / factor

        # Step 2: Rotate the kernel to the specified angle
        center = (factor // 2, factor // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(kernel, rot_matrix, (factor, factor))  # type: ignore

    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        offset: int = FACTOR,
        angle: int = current_angle,
        video_rotate: int = 0,  # Neg, neutral, positive
    ) -> Image:
        if video_rotate != 0:
            self._rotate(video_rotate)
            angle = self.current_angle

        new_factor = int(
            self.helpers.normalise_factor(effect_context.image_shape, offset)
        )
        rotated_kernel = self._apply_factor_to_kernel(new_factor, angle)

        # Step 3: Apply the kernel to the image
        return cv2.filter2D(effect_context.image, -1, rotated_kernel)  # type: ignore

import cv2
import numpy as np

from censor_engine.detected_part import Part
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import BlurEffect
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, TypeMask


@EffectRegistry.register()
class Blur(BlurEffect):
    """This is a Average Blur."""

    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        factor: float = 5,
    ) -> Image:
        factors = self.apply_factor(image, factor)

        try:
            blurred_image = cv2.blur(image, (factors[1], factors[1]))
        except Exception:  # noqa: BLE001 # Forgot Error
            blurred_image = cv2.blur(image, (factors[0], factors[0]))

        return blurred_image


@EffectRegistry.register()
class GaussianBlur(BlurEffect):
    """This is a Gaussian Blur."""

    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        factor: float = 5,
    ) -> Image:
        factors = self.apply_factor(image, factor)

        try:
            blurred_image = cv2.GaussianBlur(
                image,
                (factors[1], factors[1]),
                0,
            )
        except Exception:  # noqa: BLE001 # Forgot Error
            blurred_image = cv2.GaussianBlur(
                image,
                (factors[0], factors[0]),
                0,
            )

        return blurred_image


@EffectRegistry.register()
class MedianBlur(BlurEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        factor: float = 5,
    ) -> Image:
        factors = self.apply_factor(image, factor)

        try:
            blurred_image = cv2.medianBlur(image, factors[1])
        except Exception:  # noqa: BLE001 # Forgot Error
            blurred_image = cv2.medianBlur(image, factors[0])

        return blurred_image


@EffectRegistry.register()
class BilateralBlur(BlurEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        distance: float = 20,
        sigma_colour: int = 150,
        sigma_space: int = 150,
    ) -> Image:
        factors = self.apply_factor(image, distance)

        try:
            blurred_image = cv2.bilateralFilter(
                image,
                factors[1],
                sigma_colour,
                sigma_space,
            )
        except Exception:  # noqa: BLE001 # Forgot Error
            blurred_image = cv2.bilateralFilter(
                image,
                factors[0],
                sigma_colour,
                sigma_space,
            )

        return blurred_image


@EffectRegistry.register()
class MotionBlur(BlurEffect):
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

    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        offset: int = 10,
        angle: int = current_angle,
        video_rotate: int = 0,  # Neg, neutral, positive
    ) -> Image:
        blur_image = image.copy()

        if video_rotate != 0:
            self._rotate(video_rotate)
            angle = self.current_angle

        factors = self.apply_factor(image, offset)

        def apply_factor_to_kernel(factor: int) -> Image:
            kernel = np.zeros((factor, factor))
            kernel[int((factor - 1) / 2), :] = np.ones(factor)
            kernel = kernel / factor

            # Step 2: Rotate the kernel to the specified angle
            center = (factor // 2, factor // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            return cv2.warpAffine(kernel, rot_matrix, (factor, factor))

        try:
            rotated_kernel = apply_factor_to_kernel(factors[1])
        except Exception:  # noqa: BLE001 # TODO: Forgot what it was lol
            rotated_kernel = apply_factor_to_kernel(factors[0])

        # Step 3: Apply the kernel to the image
        return cv2.filter2D(blur_image, -1, rotated_kernel)

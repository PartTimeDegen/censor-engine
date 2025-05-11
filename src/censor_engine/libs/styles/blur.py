import cv2
import numpy as np
from censor_engine.models.styles import BlurStyle
from censor_engine.typing import CVImage


class Blur(BlurStyle):
    """
    This is a Average Blur

    """

    style_name: str = "blur"

    def apply_style(self, image: CVImage, contour, factor: int | float = 5) -> CVImage:
        factors = self.apply_factor(image, factor)

        try:
            blurred_image = cv2.blur(image, (factors[1], factors[1]))
        except Exception:
            blurred_image = cv2.blur(image, (factors[0], factors[0]))

        return self.draw_effect_on_mask([contour], blurred_image, image)


class GaussianBlur(BlurStyle):
    """
    This is a Gaussian Blur

    """

    style_name: str = "gaussian_blur"

    def apply_style(self, image: CVImage, contour, factor: int | float = 5) -> CVImage:
        factors = self.apply_factor(image, factor)

        try:
            blurred_image = cv2.GaussianBlur(image, (factors[1], factors[1]), 0)
        except Exception:
            blurred_image = cv2.GaussianBlur(image, (factors[0], factors[0]), 0)

        return self.draw_effect_on_mask([contour], blurred_image, image)


class MedianBlur(BlurStyle):
    style_name: str = "median_blur"

    def apply_style(self, image: CVImage, contour, factor: int | float = 5) -> CVImage:
        factors = self.apply_factor(image, factor)

        try:
            blurred_image = cv2.medianBlur(image, factors[1])
        except Exception:
            blurred_image = cv2.medianBlur(image, factors[0])

        return self.draw_effect_on_mask([contour], blurred_image, image)


class BilateralBlur(BlurStyle):
    style_name: str = "bilateral_blur"

    def apply_style(
        self,
        image: CVImage,
        contour,
        distance: int | float = 5,
        sigma_colour: int = 150,
        sigma_space: int = 150,
    ) -> CVImage:
        factors = self.apply_factor(image, distance)

        try:
            blurred_image = cv2.bilateralFilter(
                image, factors[1], sigma_colour, sigma_space
            )
        except Exception:
            blurred_image = cv2.bilateralFilter(
                image, factors[0], sigma_colour, sigma_space
            )

        return self.draw_effect_on_mask([contour], blurred_image, image)


class MotionBlur(BlurStyle):
    style_name: str = "motion_blur"

    current_angle: int = -45

    def _rotate(self, rotation: int):
        if rotation > 0:
            type(self).current_angle += 1
        else:
            type(self).current_angle -= 1

        if type(self).current_angle >= 180:
            type(self).current_angle = 0
        elif type(self).current_angle <= -180:
            type(self).current_angle = 0

    def apply_style(
        self,
        image: CVImage,
        contour,
        offset: int = 10,
        angle: int = current_angle,
        video_rotate: int = 0,  # Neg, neutral, positive
    ) -> CVImage:
        blur_image = image.copy()

        if video_rotate != 0:
            self._rotate(video_rotate)
            angle = self.current_angle

        factors = self.apply_factor(image, offset)

        def apply_factor_to_kernel(factor: int) -> CVImage:
            kernel = np.zeros((factor, factor))
            kernel[int((factor - 1) / 2), :] = np.ones(factor)
            kernel = kernel / factor

            # Step 2: Rotate the kernel to the specified angle
            center = (factor // 2, factor // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            return cv2.warpAffine(kernel, rot_matrix, (factor, factor))

        try:
            rotated_kernel = apply_factor_to_kernel(factors[1])
        except Exception:
            rotated_kernel = apply_factor_to_kernel(factors[0])

        # Step 3: Apply the kernel to the image
        noise_image = cv2.filter2D(blur_image, -1, rotated_kernel)

        return self.draw_effect_on_mask([contour], noise_image, image)


effects = {
    "blur": Blur,
    "gaussian_blur": GaussianBlur,
    "median_blur": MedianBlur,
    "bilateral_blur": BilateralBlur,
    "motion_blur": MotionBlur,
}

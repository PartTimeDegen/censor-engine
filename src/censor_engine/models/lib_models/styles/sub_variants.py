import cv2

from censor_engine.models.enums import StyleType
from censor_engine.typing import Image
from .base import Style


class TransparentStyle(Style):
    style_type: StyleType = StyleType.TRANSPARENCY
    force_png: bool = True  # Needed for alpha channel to work


class BlurStyle(Style):
    style_type: StyleType = StyleType.BLUR

    def normalise_factor(
        self,
        image: Image,
        factor: int | float,
    ) -> int | float:
        # factor = 1, size = 1
        # factor = 100, size = minimum_size/blur_cap
        blur_cap = 1
        blur_rate = 0.25
        factor_cap = 100

        minimum_size = min(
            image.shape[0],
            image.shape[1],
        )

        normalised_size = minimum_size / blur_cap
        normalised_factor = factor / factor_cap

        new_factor = int(normalised_size * normalised_factor * blur_rate)

        if new_factor > 2:
            return int(factor)

        return new_factor

    def apply_factor(self, image: Image, factor: int | float) -> tuple[int, int]:
        # Fixing Strength
        factor = factor * 4 + 1

        factor = self.normalise_factor(image, factor)

        if factor < 1:
            factor = 1
        elif factor % 2 == 0:
            factor += 1

        image_ratio = (max(image.shape) - min(image.shape)) / min(image.shape)

        factor_ratio = factor / image_ratio
        return (
            int(factor_ratio * min(image.shape)),  # Min Factor
            int(factor_ratio * max(image.shape)),  # Max Factor
        )


class PixelateStyle(BlurStyle):
    style_type: StyleType = StyleType.PIXELATION


class NoiseStyle(BlurStyle):
    style_type: StyleType = StyleType.NOISE


class BoxStyle(Style):
    style_type: StyleType = StyleType.BOX


class ColourStyle(Style):
    style_type: StyleType = StyleType.COLOUR


class StyliseStyle(Style):
    style_type: StyleType = StyleType.STYLISATION


class TextStyle(Style):
    style_type: StyleType = StyleType.TEXT


class DevStyle(Style):
    style_type: StyleType = StyleType.DEV


class EdgeDetectionStyle(Style):
    style_type: StyleType = StyleType.EDGE_DETECTION
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def prepare_mask(self, mask_image):
        mask_image = cv2.cvtColor(
            mask_image,
            cv2.COLOR_BGR2GRAY,
        )

        mask_image = cv2.GaussianBlur(
            mask_image, (3, 3), 0
        )  # Minor blur for better results

        return mask_image

    def clean_image(self, mask_image):
        # Dilute/Erode to Connect Noise
        mask_image = cv2.dilate(mask_image, self.kernel, iterations=2)
        mask_image = cv2.erode(mask_image, self.kernel, iterations=2)
        # mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, self.kernel)
        # mask_image = cv2.erode(mask_image, self.kernel, iterations=1)

        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        return mask_image

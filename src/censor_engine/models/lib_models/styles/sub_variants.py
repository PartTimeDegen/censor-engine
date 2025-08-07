import cv2
import numpy as np

from censor_engine.models.enums import StyleType
from censor_engine.models.structs.colours import Colour
from censor_engine.typing import Image, Mask

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

    def apply_factor(
        self, image: Image, factor: int | float
    ) -> tuple[int, int]:
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


class OverlayStyle(Style):
    style_type: StyleType = StyleType.OVERLAY

    def _apply_mask_as_overlay(
        self,
        image: Image,
        mask: Mask,
        colour: Colour,
        alpha: float,
    ) -> Image:
        overlay = image.copy()

        # Create a single-channel boolean mask from any RGB mask channel
        mask_bool = mask[:, :, 0] > 0

        if not np.any(mask_bool):
            return overlay  # Nothing to do if mask is empty

        # Create an array of shape (H, W, 3) with the target color
        color_array = np.full_like(image, colour.value, dtype=image.dtype)

        # Alpha blending only on masked region
        if alpha < 1.0:
            # Blend only in masked region
            overlay[mask_bool] = (
                (1 - alpha) * image[mask_bool] + alpha * color_array[mask_bool]
            ).astype(image.dtype)
        else:
            # Hard color replace in masked region
            overlay[mask_bool] = color_array[mask_bool]

        return overlay


class ColourStyle(Style):
    style_type: StyleType = StyleType.COLOUR


class StyliseStyle(Style):
    style_type: StyleType = StyleType.STYLISATION


class TextStyle(Style):
    style_type: StyleType = StyleType.TEXT

    def put_text(
        self,
        image: Image,
        text: str | list[str],
        coord_origin: tuple[int, int],
        color: Colour,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: int = 1,
        thickness: int = 2,
        line_type: int = cv2.LINE_AA,
        line_spacing: float = 1.2,
    ):
        if isinstance(text, list):
            text = "\n".join(text)

        coord_x, coord_y = coord_origin
        for index, line in enumerate(text.split("\n")):
            y_line = int(
                coord_y
                + index
                * (
                    cv2.getTextSize(line, font, font_scale, thickness)[0][1]
                    * line_spacing
                )
            )
            cv2.putText(
                image,
                line,
                (coord_x, y_line),
                font,
                font_scale,
                color.value,
                thickness,
                line_type,
            )
        return image


class DevStyle(TextStyle):
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
        mask_image = mask_image.astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        return mask_image

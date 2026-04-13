import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image as PImage
from PIL import ImageDraw, ImageFont

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
        factor: float,
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

        minimum_factor = 2
        if new_factor > minimum_factor:
            return int(factor)

        return new_factor

    def apply_factor(
        self,
        image: Image,
        factor: float,
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

    # Structural Helpers
    def _get_font(self, font: str) -> str:
        fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
        font_path = None
        for font_name in fonts:
            if font in font_name:
                font_path = font_name
                break
        if font_path is None:
            msg = f"Trying to use unavailable font! {fonts}"
            raise ValueError(msg)

        return font

    def get_rotation(self, mask): ...

    # Basic Vector Helpers
    def _add_cords(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> tuple[int, int]:
        return (a[0] + b[0], a[1] + b[1])

    def _subtract_cords(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> tuple[int, int]:
        return (a[0] - b[0], a[1] - b[1])

    # Aids to Text Randomly being Bottom Left rather than Top Left
    def _normalise_text_coord(
        self,
        coords: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[int, int]:
        return (coords[0], coords[1] - size[1])

    def _get_centre(self, size: tuple[int, int]) -> tuple[int, int]:
        return (int(size[0] * 0.5), int(size[1] * 0.5))

    def _convert_rel_box_to_coords_and_size(
        self, rel_box: tuple[int, int, int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        coords = rel_box[:2]
        size = rel_box[2:]
        return coords, size

    def _get_middle_coords(
        self,
        rel_box: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        coords, size = self._convert_rel_box_to_coords_and_size(rel_box)
        half_size = self._get_centre(size)
        return (
            coords[0] + half_size[0],
            coords[1] + half_size[1],
        )

    def _convert_middle_to_bottom_left_coords(
        self,
        coords: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[int, int]:
        middle = self._get_centre(size)
        return (
            coords[0] - middle[0],
            coords[1] + middle[1],
        )

    # Pillow Module for Custom Fonts
    def _put_custom_font(
        self,
        image: Image,
        word: str,
        coords: tuple[int, int],
        font: str,
        font_percent: float,
        colour: tuple[int, int, int],
        mask_size: tuple[int, int],
        outline_width: int,
        outline_colour: tuple[int, int, int],
    ):
        # Convert to Pillow
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = PImage.fromarray(img)

        # Create Draw Object
        draw = ImageDraw.Draw(img_pil)

        # Test Word on Base Size
        base_font_loaded = ImageFont.truetype(font + ".ttf", 20)
        base_bbox = draw.textbbox(
            (0, 0),
            word,
            font=base_font_loaded,
            font_size=20,
        )
        base_sizes = (
            base_bbox[2] - base_bbox[0],
            base_bbox[3] - base_bbox[1],
        )

        # Calculate New Font Size
        font_size = int(
            font_percent
            * 20
            * min(
                (
                    mask_size[0] / base_sizes[0],
                    mask_size[1] / base_sizes[1],
                )
            )
        )
        font_loaded = ImageFont.truetype(font + ".ttf", font_size)
        bbox = draw.textbbox(
            (0, 0),
            word,
            font=font_loaded,
            font_size=font_size,
        )

        # Align to Middle
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        top_left_coords = (
            int(coords[0] - text_w * 0.5 - bbox[0]),
            int(coords[1] - text_h * 0.5 - bbox[1]),
        )

        # Draw Text
        draw.text(
            top_left_coords,
            word,
            font=font_loaded,
            fill=colour,
            stroke_width=outline_width,
            stroke_fill=outline_colour,
        )
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class DevStyle(TextStyle):
    style_type: StyleType = StyleType.DEV


class EdgeDetectionStyle(Style):
    style_type: StyleType = StyleType.EDGE_DETECTION
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def prepare_mask(self, mask_image: Image) -> Image:
        mask_image = cv2.cvtColor(
            mask_image,
            cv2.COLOR_BGR2GRAY,
        )

        return cv2.GaussianBlur(
            mask_image,
            (3, 3),
            0,
        )  # Minor blur for better results

    def clean_image(self, mask_image: Image) -> Image:
        # Dilute/Erode to Connect Noise
        mask_image = cv2.dilate(mask_image, self.kernel, iterations=2)
        mask_image = cv2.erode(mask_image, self.kernel, iterations=2)
        mask_image = mask_image.astype(np.uint8)
        return cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    def get_auto_tolerances(
        self,
        mask_image: Image,
        multiplier: float,
    ) -> tuple[int, int]:
        mean, stddev = cv2.meanStdDev(mask_image)
        mean_white = mean[0][0]
        stddev_white = stddev[0][0]
        return (
            mean_white - stddev_white * multiplier,
            mean_white + stddev_white * multiplier,
        )

    def process_lines(
        self,
        mask_image: Image,
        tols: tuple[int, int] | None,
        multiplier: float,
    ) -> Image:
        if not tols:
            tols = self.get_auto_tolerances(mask_image, multiplier)

        mask_image[mask_image < tols[0]] = 0
        mask_image[mask_image > tols[1]] = 255
        return cv2.multiply(mask_image, multiplier)  # type: ignore

import math

import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import PixelateEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.contours import Contour


@EffectRegistry.register()
class Pixelate(PixelateEffect):
    def _get_distortion_factor(
        self,
        image: Image,
        contours: list[Contour],
        factor: int,
    ):
        bounding_rect = contours[0].as_bounding_box()  # Only Biggest Matters
        _, _, box_width, box_height = bounding_rect

        fixed_size = max(box_width, box_height)

        distortion_ratio = (
            box_width / fixed_size,
            box_height / fixed_size,
        )
        distortion_ratio = (
            distortion_ratio[0] / min(distortion_ratio),
            distortion_ratio[1] / min(distortion_ratio),
        )

        size_image_ratio = (
            image.shape[0] / box_width,
            image.shape[1] / box_height,
        )
        size_image_ratio = (
            size_image_ratio[0] / min(size_image_ratio),
            size_image_ratio[1] / min(size_image_ratio),
        )

        factor_ratio = (
            int(factor / size_image_ratio[0] / distortion_ratio[0]),
            int(factor / size_image_ratio[1] / distortion_ratio[1]),
        )
        factor_ratio = (
            factor - min(factor_ratio) + factor_ratio[0],
            factor - min(factor_ratio) + factor_ratio[1],
        )
        return (
            int(factor_ratio[0] * factor / 2),
            int(factor_ratio[1] * factor / 2),
        )

    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        factor: int = 12,
    ) -> Image:
        factors = self._get_distortion_factor(
            effect_context.image,
            effect_context.contours,
            factor,
        )

        # Code Proper
        down_image = cv2.resize(
            effect_context.image,  # type: ignore
            factors,  # type: ignore
            interpolation=cv2.INTER_LINEAR,
        )
        shape = effect_context.image_shape
        return cv2.resize(
            down_image,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )  # type: ignore


@EffectRegistry.register()
class HexagonPixelate(PixelateEffect):
    def _hexagon_corners(self, center_x: float, center_y: float, size: float):
        """Compute hexagon vertices around a center using NumPy arrays."""
        w_half = math.sqrt(3) * size / 2
        h_half = size
        return np.array(
            [
                [center_x - w_half, center_y - h_half / 2],
                [center_x, center_y - h_half],
                [center_x + w_half, center_y - h_half / 2],
                [center_x + w_half, center_y + h_half / 2],
                [center_x, center_y + h_half],
                [center_x - w_half, center_y + h_half / 2],
            ],
            dtype=np.int32,
        )

    def _hexagonify(self, image: Image, hexagon_size: float):
        """Apply hexagonal pixelation using NumPy and OpenCV."""
        img_h, img_w = image.shape[:2]

        # Hexagon width & height
        w, h = math.sqrt(3) * hexagon_size, 2 * hexagon_size
        w_half, h_half, h_three_quarter = w / 2, h / 2, h * 3 / 4

        # Number of hexagons
        num_hor = math.ceil(img_w / w) + 1
        num_ver = math.ceil(img_h / h_three_quarter) + 1

        # Output image (copy of original)
        output = np.zeros_like(image)

        for row in range(num_ver):
            for col in range(num_hor):
                center_x = col * w + (row % 2) * w_half
                center_y = row * h_three_quarter

                # Bounding box for color sampling
                x_min, x_max = int(center_x - w_half), int(center_x + w_half)
                y_min, y_max = int(center_y - h_half), int(center_y + h_half)

                # Ensure indices are within bounds
                x_min, x_max = max(0, x_min), min(img_w, x_max)
                y_min, y_max = max(0, y_min), min(img_h, y_max)

                # Compute average color
                slice_region = image[y_min:y_max, x_min:x_max]
                color = (
                    np.mean(slice_region, axis=(0, 1), dtype=np.float32)
                    if slice_region.size
                    else [0, 0, 0]
                )

                # Fill the hexagon with the computed color
                hex_corners = self._hexagon_corners(
                    center_x,
                    center_y,
                    hexagon_size,
                )
                cv2.fillPoly(
                    output,  # type: ignore
                    [hex_corners],
                    color=tuple(map(int, color)),  # type: ignore
                )

        return output

    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        factor: float = 12,
    ) -> Image:
        """Apply hexagonal pixelation to the image within the given contour."""
        return self._hexagonify(effect_context.image, factor)

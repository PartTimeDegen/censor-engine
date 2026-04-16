import math

import cv2
import numpy as np

from censor_engine.api.effects import EffectContext
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import PixelateEffect
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, ProcessedImage


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
        factors = (
            int(factor_ratio[0] * factor / 2),
            int(factor_ratio[1] * factor / 2),
        )

        return (
            self.normalise_factor(image, factors[0]),
            self.normalise_factor(image, factors[1]),
        )

    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        factor: int = 12,
    ) -> ProcessedImage:
        factors = self._get_distortion_factor(
            effect_context.image,
            effect_context.contours,
            factor,
        )

        # Code Proper
        down_image = cv2.resize(
            effect_context.image,
            factors,  # type: ignore
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(
            down_image,
            (effect_context.shape[1], effect_context.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )


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

    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        factor: float = 12,
    ) -> ProcessedImage:
        """Apply hexagonal pixelation to the image within the given contour."""
        factor = self.normalise_factor(effect_context.image, factor)
        return self._hexagonify(effect_context.image, factor)


@EffectRegistry.register()
class Crystallise(PixelateEffect):
    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        point_density: int = 50,
    ) -> ProcessedImage:
        h, w, _ = effect_context.shape

        points = np.vstack(
            [
                # TODO: Errors are valid, however this looks like tedium to fix
                np.random.randint(0, w, point_density),
                np.random.randint(0, h, point_density),
            ],
        ).T.astype(np.float32)

        subdiv = cv2.Subdiv2D((0, 0, w, h))  # type: ignore
        for p in points:
            subdiv.insert((p[0], p[1]))

        triangle_list = subdiv.getTriangleList().astype(np.int32)  # type: ignore

        result = np.zeros_like(effect_context.image)

        for t in triangle_list:
            pts = t.reshape(3, 2)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 1)  # type: ignore
            mean_color = cv2.mean(effect_context.image, mask=mask)[:3]
            cv2.fillConvexPoly(result, pts, mean_color)  # type: ignore

        return result


@EffectRegistry.register()
class HexagonPixelateSoft(HexagonPixelate):
    def _blend_color(
        self,
        center_x: float,
        center_y: float,
        image: Image,
        hexagon_size: float,
        softness: float,
    ):
        """
        Blend the color of a hexagon with its surrounding hexagons for a
        smoother transition.
        """
        w = math.sqrt(3) * hexagon_size
        h = 2 * hexagon_size

        w_half = w / 2 * (1.0 + softness * 0.5)
        h_half = h / 2 * (1.0 + softness * 0.5)

        # Sampling bounds
        x_min, x_max = int(center_x - w_half), int(center_x + w_half)
        y_min, y_max = int(center_y - h_half), int(center_y + h_half)
        x_min, x_max = max(0, x_min), min(image.shape[1], x_max)
        y_min, y_max = max(0, y_min), min(image.shape[0], y_max)

        region = image[y_min:y_max, x_min:x_max]
        if region.size == 0:
            return [0, 0, 0]

        return np.mean(region, axis=(0, 1), dtype=np.float32)

    def _hexagonify(
        self, image: Image, hexagon_size: float, softness: float = 1.0
    ):
        """Apply soft hexagonal pixelation with controlled softness."""
        img_h, img_w = image.shape[:2]

        w, h = math.sqrt(3) * hexagon_size, 2 * hexagon_size
        w_half, _, h_three_quarter = w / 2, h / 2, h * 3 / 4

        num_hor = math.ceil(img_w / w) + 1
        num_ver = math.ceil(img_h / h_three_quarter) + 1

        output = np.zeros_like(image)

        for row in range(num_ver):
            for col in range(num_hor):
                center_x = col * w + (row % 2) * w_half
                center_y = row * h_three_quarter

                # Get the soft blended color
                color = self._blend_color(
                    center_x,
                    center_y,
                    image,
                    hexagon_size,
                    softness,
                )

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

        # Optional: Gaussian blur the final output based on softness
        if softness > 0:
            ksize = int(3 + softness * 2) | 1  # must be odd
            output = cv2.GaussianBlur(output, (ksize, ksize), sigmaX=softness)

        return output

    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        factor: float = 12,
        softness: float = 2.0,
    ) -> ProcessedImage:
        """
        Apply soft hexagonal pixelation to the image within the given contour.

        :param image: input image
        :param contours: list[Contour] mask
        :param factor: hexagon size
        :param softness: softness amount (default=1.0)
        """
        factor = self.normalise_factor(effect_context.image, factor)
        return self._hexagonify(effect_context.image, factor, softness)

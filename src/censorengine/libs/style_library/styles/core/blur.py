import math

import cv2
import numpy as np
from PIL import Image, ImageDraw
from censorengine.lib_models.styles import BlurStyle
from censorengine.backend.constants.typing import CVImage


class Blur(BlurStyle):
    style_name: str = "blur"

    def apply_style(
        self,
        image: CVImage,
        contour,
        factor: int | float = 12,
    ) -> CVImage:
        # Fixing Strength
        factor = factor * 4 + 1

        factor = self.normalise_factor(image, factor)

        if factor < 1:
            factor = 1
        elif factor % 2 == 0:
            factor += 1

        image_ratio = (max(image.shape) - min(image.shape)) / min(image.shape)
        max_factor = int(factor * max(image.shape) / image_ratio)

        try:
            blurred_image = cv2.GaussianBlur(
                image,
                (max_factor, max_factor),
                0,
            )
        except Exception:
            min_Factor = int(factor * min(image.shape) / image_ratio)
            blurred_image = cv2.GaussianBlur(
                image,
                (min_Factor, min_Factor),
                0,
            )

        return self.draw_effect_on_mask([contour], blurred_image, image)


class Pixelate(BlurStyle):
    style_name: str = "pixelate"

    def _get_distortion_factor(
        self,
        image,
        contour,
        factor,
    ):
        contour_array = contour[0][0]
        bounding_rect = cv2.boundingRect(contour_array)
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

    def apply_style(
        self,
        image: CVImage,
        contour,
        factor: int | float = 12,
    ) -> CVImage:
        factors = self._get_distortion_factor(
            image,
            contour,
            factor,
        )

        # Code Proper
        down_image = cv2.resize(
            image,
            factors,
            interpolation=cv2.INTER_LINEAR,
        )
        pixel_image = cv2.resize(
            down_image,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return self.draw_effect_on_mask([contour], pixel_image, image)


class HexagonPixelate(BlurStyle):
    style_name: str = "hexagon_pixelate"

    def _hexagon_corners(self, center, size):
        x = center[0]
        y = center[1]

        w = math.sqrt(3) * size
        h = 2 * size

        return [
            (x - w / 2, y - h / 4),
            (x, y - h / 2),
            (x + w / 2, y - h / 4),
            (x + w / 2, y + h / 4),
            (x, y + h / 2),
            (x - w / 2, y + h / 4),
        ]

    def _rectangle_corners(self, center, width, height):
        x = center[0]
        y = center[1]

        return [
            (x - width / 2, y - height / 2),
            (x + width / 2, y - height / 2),
            (x + width / 2, y + height / 2),
            (x - width / 2, y + height / 2),
        ]

    def _hexagonify(self, image, hexagon_size):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)

        image_from_numpy = np.asarray(im)
        draw = ImageDraw.Draw(im)

        w = math.sqrt(3) * hexagon_size
        h = 2 * hexagon_size

        # numer of hexagons horizontally and vertically
        num_hor = int(im.size[0] / w) + 2
        num_ver = int(im.size[1] / h * 4 / 3) + 2

        for i in range(0, num_hor * num_ver):
            column = i % num_hor
            row = i // num_hor
            even = (
                row % 2
            )  # the even rows of hexagons has w/2 offset on the x-axis compared to odd rows.

            p = self._hexagon_corners(
                (column * w + even * w / 2, row * h * 3 / 4), hexagon_size
            )

            # compute the average color of the hexagon, use a rectangle approximation.
            raw = self._rectangle_corners(
                (column * w + even * w / 2, row * h * 3 / 4), w, h
            )
            r = []
            for points in raw:
                np0 = int(np.clip(points[0], 0, im.size[0]))
                np1 = int(np.clip(points[1], 0, im.size[1]))
                r.append((np0, np1))

            color = np.average(
                image_from_numpy[r[0][1] : r[3][1], r[0][0] : r[1][0]],
                axis=(0, 1),
            )
            color = tuple(color.astype(np.uint8))

            draw.polygon(p, fill=color)
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def apply_style(
        self,
        image: CVImage,
        contour,
        factor: int | float = 12,
    ) -> CVImage:
        # """
        # Credit : https://github.com/McJazzy/hexagonpy
        # """

        factor = self.normalise_factor(image, factor)
        pixel_image = self._hexagonify(image, factor)

        return self.draw_effect_on_mask(contour, pixel_image, image)


effects = {
    "blur": Blur,
    "pixelate": Pixelate,
    "hexagon_pixelate": HexagonPixelate,
}

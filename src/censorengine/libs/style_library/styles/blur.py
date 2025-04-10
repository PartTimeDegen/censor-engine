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
            factors,  # type: ignore
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

    def _hexagon_corners(self, center_x, center_y, size):
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

    def _hexagonify(self, image, hexagon_size):
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
                hex_corners = self._hexagon_corners(center_x, center_y, hexagon_size)
                cv2.fillPoly(output, [hex_corners], color=tuple(map(int, color)))

        return output

    def apply_style(self, image: CVImage, contour, factor: int | float = 12) -> CVImage:
        """Apply hexagonal pixelation to the image within the given contour."""
        factor = self.normalise_factor(image, factor)
        pixel_image = self._hexagonify(image, factor)
        return self.draw_effect_on_mask(contour, pixel_image, image)


class MotionBlur(BlurStyle):
    style_name: str = "motion_blur"

    def apply_style(
        self,
        image: CVImage,
        contour,
        offset: int = 20,
        angle: int = -45,
    ) -> CVImage:
        blur_image = image.copy()

        kernel = np.zeros((offset, offset))
        kernel[int((offset - 1) / 2), :] = np.ones(offset)
        kernel = kernel / offset

        # Step 2: Rotate the kernel to the specified angle
        center = (offset // 2, offset // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_kernel = cv2.warpAffine(kernel, rot_matrix, (offset, offset))

        # Step 3: Apply the kernel to the image
        noise_image = cv2.filter2D(blur_image, -1, rotated_kernel)

        return self.draw_effect_on_mask([contour], noise_image, image)


class Crystallise(BlurStyle):
    style_name: str = "crystallise"

    def apply_style(
        self,
        image: CVImage,
        contour,
        point_density: int = 50,
    ) -> CVImage:
        blur_image = image.copy()
        h, w, c = image.shape

        points = np.vstack(
            [
                np.random.randint(0, w, point_density),
                np.random.randint(0, h, point_density),
            ]
        ).T.astype(np.float32)

        subdiv = cv2.Subdiv2D((0, 0, w, h))  # type: ignore
        for p in points:
            subdiv.insert((p[0], p[1]))

        triangle_list = subdiv.getTriangleList().astype(np.int32)  # type: ignore

        result = np.zeros_like(blur_image)

        for t in triangle_list:
            pts = t.reshape(3, 2)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 1)  # type: ignore
            mean_color = cv2.mean(image, mask=mask)[:3]
            cv2.fillConvexPoly(result, pts, mean_color)  # type: ignore

        return self.draw_effect_on_mask([contour], result, image)


effects = {
    "blur": Blur,
    "pixelate": Pixelate,
    "hexagon_pixelate": HexagonPixelate,
    "motion_blur": MotionBlur,
    "crystallise": Crystallise,
}

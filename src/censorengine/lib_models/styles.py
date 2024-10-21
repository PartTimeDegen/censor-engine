# TODO: Remove me once you made a model for styles
from dataclasses import dataclass
import cv2
import numpy as np

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import Mask, CVImage


@dataclass
class Style:
    style_name: str = "invalid_style"
    style_type: str = "invalid_style"

    force_png: bool = False

    def apply_style(
        image: CVImage,
        contour,
        *args: Any,
        **kwargs: str | int | float,
    ) -> CVImage:
        raise NotImplementedError

    def draw_effect_on_mask(
        contours,
        mask_image: Mask,
        image: CVImage,
    ) -> CVImage:

        mask = cv2.drawContours(
            np.zeros(image.shape, dtype=np.uint8),
            contours[0],
            -1,
            (255, 255, 255),
            -1,
            hierarchy=contours[1],
            lineType=cv2.LINE_AA,
        )
        return np.where(
            mask == np.array([255, 255, 255]),
            mask_image,
            image,
        )


class TransparentStyle(Style):
    style_type: str = "transparant"
    force_png: bool = True  # Needed for alpha channel to work


class BlurStyle(Style):
    style_type: str = "blur"

    def normalise_factor(
        image: CVImage,
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

        if new_factor == 0:
            return int(factor)

        return new_factor


class BoxStyle(Style):
    style_type: str = "box"


class ColourStyle(Style):
    style_type: str = "colour"


class TextStyle(Style):
    style_type: str = "text"


class DevStyle(Style):
    style_type: str = "dev"

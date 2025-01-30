from abc import ABC, abstractmethod
import cv2
import numpy as np

from censorengine.backend.constants.typing import Mask, CVImage, ProcessedImage


class Style(ABC):
    # Information
    style_name: str = "invalid_style"
    style_type: str = "invalid_style"

    # Supporting Information
    force_png: bool = False
    default_linetype: int = cv2.LINE_AA
    using_reverse_censor: bool = False

    # This is the ra
    @abstractmethod
    def apply_style(
        self,
        image: CVImage,
        contour,
        *args,
        **kwargs,
    ) -> ProcessedImage:
        raise NotImplementedError

    def draw_effect_on_mask(
        self,
        contours,
        mask_image: Mask,
        image: CVImage,
    ) -> CVImage:
        if len(contours) == 1:
            contours = contours[0]

        mask = cv2.drawContours(
            np.zeros(image.shape, dtype=np.uint8),
            contours[0],
            -1,
            (255, 255, 255),
            -1,
            hierarchy=contours[1],
            lineType=self.default_linetype,
        )
        return np.where(
            mask == np.array([255, 255, 255]),
            mask_image,
            image,
        )

    def change_linetype(self, enable_aa: bool) -> None:
        if not enable_aa:
            self.default_linetype = cv2.LINE_4
        else:
            self.default_linetype = cv2.LINE_AA


class TransparentStyle(Style):
    style_type: str = "transparant"
    force_png: bool = True  # Needed for alpha channel to work


class BlurStyle(Style):
    style_type: str = "blur"

    def normalise_factor(
        self,
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

        if new_factor > 2:
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


class EdgeDetectionStyle(Style):
    style_type = "edge_detection"
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

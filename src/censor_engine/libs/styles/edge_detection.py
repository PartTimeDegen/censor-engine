import cv2
import numpy as np

from censor_engine.detected_part import Part
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import EdgeDetectionStyle
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask

# Edge Detection Effects
# https://blog.roboflow.com/edge-detection/
# https://www.geeksforgeeks.org/comprehensive-guide-to-edge-detection-algorithms/


@StyleRegistry.register()
class EdgeDetectionCanny(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        threshold: int = 100,
        alpha=1,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Canny(mask_image, threshold, threshold)

        # Clean Image
        mask_image = self.clean_image(mask_image)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionSobel(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        kernel_size: int = 5,
        alpha=1,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Sobel(mask_image, cv2.CV_64F, 1, 1, ksize=kernel_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionScharr(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        alpha=1,
        kernel_size: int = 5,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        sobelx = cv2.Sobel(mask_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(mask_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        mask_image = cv2.magnitude(sobelx, sobely)
        mask_image = np.uint8(np.clip(mask_image, 0, 255))  # if needed

        # Clean Image
        mask_image = self.clean_image(mask_image)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionLapacian(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        kernel_size: int = 5,
        alpha=1,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Laplacian(mask_image, cv2.CV_64F, ksize=kernel_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)

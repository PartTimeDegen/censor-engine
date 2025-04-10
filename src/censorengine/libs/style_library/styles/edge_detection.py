import cv2

from censorengine.lib_models.styles import EdgeDetectionStyle
from censorengine.backend.constants.typing import CVImage

# Edge Detection Effects
# https://blog.roboflow.com/edge-detection/
# https://www.geeksforgeeks.org/comprehensive-guide-to-edge-detection-algorithms/


class EdgeDetectionCanny(EdgeDetectionStyle):
    style_name: str = "edge_detection_canny"

    def apply_style(
        self,
        image: CVImage,
        contour,
        threshold: int = 100,
        alpha=1,
    ) -> CVImage:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Canny(mask_image, threshold, threshold)

        # Clean Image
        mask_image = self.clean_image(mask_image)

        effect = self.draw_effect_on_mask([contour], mask_image, image)
        return cv2.addWeighted(effect, alpha, image, 1 - alpha, 0)


class EdgeDetectionSobel(EdgeDetectionStyle):
    style_name: str = "edge_detection_sobel"

    def apply_style(
        self,
        image: CVImage,
        contour,
        kernal_size: int = 5,
        alpha=1,
    ) -> CVImage:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Sobel(mask_image, cv2.CV_64F, 1, 1, ksize=kernal_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)

        effect = self.draw_effect_on_mask([contour], mask_image, image)
        return cv2.addWeighted(effect, alpha, image, 1 - alpha, 0)


class EdgeDetectionScharr(EdgeDetectionStyle):
    style_name: str = "edge_detection_scharr"

    def apply_style(
        self,
        image: CVImage,
        contour,
        alpha=1,
    ) -> CVImage:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Sobel(mask_image, cv2.CV_64F, 1, 1, ksize=-1)

        # Clean Image
        mask_image = self.clean_image(mask_image)

        effect = self.draw_effect_on_mask([contour], mask_image, image)
        return cv2.addWeighted(effect, alpha, image, 1 - alpha, 0)


class EdgeDetectionLapacian(EdgeDetectionStyle):
    style_name: str = "edge_detection_laplacian"

    def apply_style(
        self,
        image: CVImage,
        contour,
        kernal_size: int = 5,
        alpha=1,
    ) -> CVImage:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Laplacian(mask_image, cv2.CV_64F, ksize=kernal_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)

        effect = self.draw_effect_on_mask([contour], mask_image, image)
        return cv2.addWeighted(effect, alpha, image, 1 - alpha, 0)


effects = {
    "edge_detection_canny": EdgeDetectionCanny,
    "edge_detection_sobel": EdgeDetectionCanny,
    "edge_detection_scharr": EdgeDetectionCanny,
    "edge_detection_laplacian": EdgeDetectionCanny,
}

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
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        threshold: int = 100,
        alpha: float = 1,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Canny(mask_image, threshold, threshold)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionSobel(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        kernel_size: int = 5,
        alpha: float = 1,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Sobel(mask_image, cv2.CV_64F, 1, 1, ksize=kernel_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionScharr(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        alpha: float = 1,
        kernel_size: int = 5,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        sobelx = cv2.Sobel(mask_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(mask_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        mask_image = cv2.magnitude(sobelx, sobely)
        mask_image = np.uint8(np.clip(mask_image, 0, 255))  # type: ignore

        # Clean Image
        mask_image = self.clean_image(mask_image)  # type: ignore
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionLapacian(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        kernel_size: int = 5,
        alpha: float = 1,
    ) -> Image:
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = cv2.Laplacian(mask_image, cv2.CV_64F, ksize=kernel_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionDoubleGaussian(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        alpha: float = 1.0,
        ksize: int = 0,
    ) -> Image:
        # Prepare image for better results
        gray = self.prepare_mask(image)

        # Apply two Gaussian blurs
        blur1 = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma1)
        blur2 = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma2)

        # Difference of Gaussians
        dog = cv2.subtract(blur1, blur2)

        # Normalize to full range
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore

        # Clean image (reuse your existing logic)
        dog = self.clean_image(dog)

        # Blend with original image
        return cv2.addWeighted(dog, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionRoberts(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        alpha: float = 1.0,
    ) -> Image:
        def roberts(img: Image):
            kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            x = cv2.filter2D(img, cv2.CV_64F, kernelx)
            y = cv2.filter2D(img, cv2.CV_64F, kernely)
            return cv2.convertScaleAbs(np.sqrt(x**2 + y**2))

        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = roberts(mask_image)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class EdgeDetectionPrewitt(EdgeDetectionStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        alpha: float = 1.0,
    ) -> Image:
        def prewitt(img: Image):
            kernelx = np.array(
                [[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32
            )
            kernely = np.array(
                [[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32
            )
            x = cv2.filter2D(img, cv2.CV_64F, kernelx)
            y = cv2.filter2D(img, cv2.CV_64F, kernely)
            return cv2.convertScaleAbs(np.sqrt(x**2 + y**2))

        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get Mask
        mask_image = prewitt(mask_image)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)

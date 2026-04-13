import cv2
import numpy as np

from censor_engine.detected_part import Part
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import EdgeDetectionEffect
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, TypeMask

# Edge Detection Effects
# https://blog.roboflow.com/edge-detection/
# https://www.geeksforgeeks.org/comprehensive-guide-to-edge-detection-algorithms/


@EffectRegistry.register()
class EdgeDetectionCanny(EdgeDetectionEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        threshold: int = 100,
        alpha: float = 1,
    ) -> Image:
        """
        This effect uses the Canny Edge Detection.

        References:
            -   https://en.wikipedia.org/wiki/Canny_edge_detector

        :param Image image: Original Image, used to apply the effects.
        :param TypeMask mask: TypeMask of the applied zone.
        :param list[Contour] contours: List of Contours (Sometimes used)
        :param Part part: Part of the mask (Sometimes used)
        :param tuple[int, int] | None tolerances: Tolerances of the lines for
        post-processing, defaults to None
        :param float multiplier: Multiplier for post-processing to make the
        lines stronger, defaults to 1.0
        :param int threshold: Threshold for the edges, used in post-processing
        to remove noise, defaults to 100
        :param float alpha: Alpha of the effect, defaults to 1

        :return Image: Image with effect applied.

        """
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get TypeMask
        mask_image = cv2.Canny(mask_image, threshold, threshold)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@EffectRegistry.register()
class EdgeDetectionSobel(EdgeDetectionEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        alpha: float = 1,
        kernel_size: int = 5,
    ) -> Image:
        """
        This effect uses the Canny Edge Detection.

        References:
            -   https://en.wikipedia.org/wiki/Sobel_operator

        :param Image image: Original Image, used to apply the effects.
        :param TypeMask mask: TypeMask of the applied zone.
        :param list[Contour] contours: List of Contours (Sometimes used)
        :param Part part: Part of the mask (Sometimes used)
        :param tuple[int, int] | None tolerances: Tolerances of the lines for
        post-processing, defaults to None
        :param float multiplier: Multiplier for post-processing to make the
        lines stronger, defaults to 1.0
        :param int threshold: Threshold for the edges, used in post-processing
        to remove noise, defaults to 100
        :param int kernel_size: The kernel size, defaults to 5

        :return Image: Image with effect applied.

        """
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get TypeMask
        mask_image = cv2.Sobel(mask_image, cv2.CV_64F, 1, 1, ksize=kernel_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@EffectRegistry.register()
class EdgeDetectionScharr(EdgeDetectionEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        alpha: float = 1,
        kernel_size: int = 5,
    ) -> Image:
        """
        This effect uses the Canny Edge Detection.

        References:
            -   https://en.wikipedia.org/wiki/Sobel_operator

        :param Image image: Original Image, used to apply the effects.
        :param TypeMask mask: TypeMask of the applied zone.
        :param list[Contour] contours: List of Contours (Sometimes used)
        :param Part part: Part of the mask (Sometimes used)
        :param tuple[int, int] | None tolerances: Tolerances of the lines for
        post-processing, defaults to None
        :param float multiplier: Multiplier for post-processing to make the
        lines stronger, defaults to 1.0
        :param int threshold: Threshold for the edges, used in post-processing
        to remove noise, defaults to 100

        :return Image: Image with effect applied.

        """
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get TypeMask
        sobelx = cv2.Sobel(mask_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(mask_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        mask_image = cv2.magnitude(sobelx, sobely)
        mask_image = np.uint8(np.clip(mask_image, 0, 255))  # type: ignore

        # Clean Image
        mask_image = self.clean_image(mask_image)  # type: ignore
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@EffectRegistry.register()
class EdgeDetectionLapacian(EdgeDetectionEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        kernel_size: int = 5,
        alpha: float = 1,
    ) -> Image:
        """
        This effect uses the Canny Edge Detection.

        References:
            -   https://en.wikipedia.org/wiki/Sobel_operator

        :param Image image: Original Image, used to apply the effects.
        :param TypeMask mask: TypeMask of the applied zone.
        :param list[Contour] contours: List of Contours (Sometimes used)
        :param Part part: Part of the mask (Sometimes used)
        :param tuple[int, int] | None tolerances: Tolerances of the lines for
        post-processing, defaults to None
        :param float multiplier: Multiplier for post-processing to make the
        lines stronger, defaults to 1.0
        :param int threshold: Threshold for the edges, used in post-processing
        to remove noise, defaults to 100

        :return Image: Image with effect applied.

        """
        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get TypeMask
        mask_image = cv2.Laplacian(mask_image, cv2.CV_64F, ksize=kernel_size)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@EffectRegistry.register()
class EdgeDetectionDoubleGaussian(EdgeDetectionEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
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
        """
        This effect uses the Difference of Gaussians Detection.


        References:
            -   https://en.wikipedia.org/wiki/Difference_of_Gaussians


        :param tuple[int, int] | None tolerances: Tolerances of the lines for
        post-processing, defaults to None
        :param float multiplier: Multiplier for post-processing to make the
        lines stronger, defaults to 1.0
        :param int threshold: Threshold for the edges, used in post-processing
        to remove noise, defaults to 100

        :return Image: Image with effect applied.

        """
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


@EffectRegistry.register()
class EdgeDetectionRoberts(EdgeDetectionEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        alpha: float = 1.0,
    ) -> Image:
        """
        This effect uses the Roberts Cross Detection.


        References:
            -   https://en.wikipedia.org/wiki/Roberts_cross


        :param tuple[int, int] | None tolerances: Tolerances of the lines for
        post-processing, defaults to None
        :param float multiplier: Multiplier for post-processing to make the
        lines stronger, defaults to 1.0
        :param int threshold: Threshold for the edges, used in post-processing
        to remove noise, defaults to 100

        :return Image: Image with effect applied.

        """

        def roberts(img: Image):
            kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            x = cv2.filter2D(img, cv2.CV_64F, kernelx)
            y = cv2.filter2D(img, cv2.CV_64F, kernely)
            return cv2.convertScaleAbs(np.sqrt(x**2 + y**2))

        # Prepare Image to get better Results
        mask_image = self.prepare_mask(image)

        # Get TypeMask
        mask_image = roberts(mask_image)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)


@EffectRegistry.register()
class EdgeDetectionPrewitt(EdgeDetectionEffect):
    def apply_effect(
        self,
        image: Image,
        mask: TypeMask,
        contours: list[Contour],
        part: Part,
        *,
        tolerances: tuple[int, int] | None = None,
        multiplier: float = 1.0,
        alpha: float = 1.0,
    ) -> Image:
        """
        This effect uses the Prewitt Detection.


        References:
            -   https://en.wikipedia.org/wiki/Prewitt_operator


        :param tuple[int, int] | None tolerances: Tolerances of the lines for
        post-processing, defaults to None
        :param float multiplier: Multiplier for post-processing to make the
        lines stronger, defaults to 1.0
        :param int threshold: Threshold for the edges, used in post-processing
        to remove noise, defaults to 100

        :return Image: Image with effect applied.

        """

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

        # Get TypeMask
        mask_image = prewitt(mask_image)

        # Clean Image
        mask_image = self.clean_image(mask_image)
        mask_image = self.process_lines(mask_image, tolerances, multiplier)

        return cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)

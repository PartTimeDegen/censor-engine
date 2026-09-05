import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.libraries.effects._default_args import (
    BLACK_AND_WHITE,
    INVERSE,
    KERNAL_SIZE,
)
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import EdgeDetectionEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


@EffectRegistry.register()
class EdgeDetectionCanny(EdgeDetectionEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        threshold: int = 100,
        inverse: bool = INVERSE,
        black_and_white: bool = BLACK_AND_WHITE,
    ) -> Image:
        mono_image = cv2.Canny(effect_context.image, threshold, threshold)
        return np.repeat(mono_image[..., None], 3, axis=2)  # type: ignore


@EffectRegistry.register()
class EdgeDetectionSobel(EdgeDetectionEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        kernel_size: int = KERNAL_SIZE,
        inverse: bool = INVERSE,
        black_and_white: bool = BLACK_AND_WHITE,
    ) -> Image:
        return cv2.Sobel(  # type: ignore
            effect_context.image, cv2.CV_64F, 1, 1, ksize=kernel_size
        )


@EffectRegistry.register()
class EdgeDetectionScharr(EdgeDetectionEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        kernel_size: int = KERNAL_SIZE,
        inverse: bool = INVERSE,
        black_and_white: bool = BLACK_AND_WHITE,
    ) -> Image:
        sobelx = cv2.Sobel(
            effect_context.image, cv2.CV_64F, 1, 0, ksize=kernel_size
        )
        sobely = cv2.Sobel(
            effect_context.image, cv2.CV_64F, 0, 1, ksize=kernel_size
        )
        mask_image = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(mask_image, 0, 255))  # type: ignore


@EffectRegistry.register()
class EdgeDetectionLapacian(EdgeDetectionEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        kernel_size: int = KERNAL_SIZE,
        inverse: bool = INVERSE,
        black_and_white: bool = BLACK_AND_WHITE,
    ) -> Image:
        return cv2.Laplacian(  # type: ignore
            effect_context.image,
            cv2.CV_64F,
            ksize=kernel_size,
        )


@EffectRegistry.register()
class EdgeDetectionDoubleGaussian(EdgeDetectionEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        ksize: int = 0,
        inverse: bool = INVERSE,
        black_and_white: bool = BLACK_AND_WHITE,
    ) -> Image:
        blur1 = cv2.GaussianBlur(
            effect_context.image,
            (ksize, ksize),
            sigmaX=sigma1,
        )
        blur2 = cv2.GaussianBlur(
            effect_context.image,
            (ksize, ksize),
            sigmaX=sigma2,
        )

        # Difference of Gaussians
        dog = cv2.subtract(blur1, blur2)

        # Normalize to full range
        return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore


@EffectRegistry.register()
class EdgeDetectionRoberts(EdgeDetectionEffect):
    def generate_effect(self, effect_context: EffectContext) -> Image:  # type: ignore
        def roberts(img: Image):
            kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            x = cv2.filter2D(img, cv2.CV_64F, kernelx)
            y = cv2.filter2D(img, cv2.CV_64F, kernely)
            return cv2.convertScaleAbs(np.sqrt(x**2 + y**2))

        return roberts(effect_context.image)


@EffectRegistry.register()
class EdgeDetectionPrewitt(EdgeDetectionEffect):
    def generate_effect(self, effect_context: EffectContext) -> Image:  # type: ignore
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

        return prewitt(effect_context.image)

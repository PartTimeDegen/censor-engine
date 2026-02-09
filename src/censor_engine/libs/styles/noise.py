import math

import cv2
import numpy as np

from censor_engine.detected_part import Part
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import NoiseStyle
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask


@StyleRegistry.register()
class ChromaticAberration(NoiseStyle):
    style_name: str = "chromatic_aberration"

    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        offset: int = 20,
        angle: int = -45,
    ) -> Image:
        # Create a copy for the noise effect
        noise_image = image.copy()
        offset = -offset

        # Correct angle components for x and y directions
        comp_x = math.cos(math.radians(angle))  # Horizontal shift
        comp_y = math.sin(math.radians(angle))  # Vertical shift

        # Split into B, G, R channels
        channels = cv2.split(noise_image)
        channels = list(channels)

        # Loop through each color channel (B=0, G=1, R=2)
        for i, channel in enumerate(channels):
            # Calculate the shift amount per channel
            dx = int(offset * (i + 1) * comp_x)
            dy = int(offset * (i + 1) * comp_y)

            # Create the affine transformation matrix for shifting
            matrix_moment = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore

            # Apply the shift using warpAffine (faster than np.roll)

            channels[i] = cv2.warpAffine(  # type: ignore
                channel,
                matrix_moment,  # type: ignore
                (channel.shape[1], channel.shape[0]),
                borderMode=cv2.BORDER_REFLECT,
            )  # type: ignore

        # Merge the shifted channels back
        channels = tuple(channels)
        return cv2.merge(channels)  # type: ignore

        # Apply the effect to the masked area


@StyleRegistry.register()
class CentricChromaticAberration(NoiseStyle):
    # FIXME This doesn't give centric aberration

    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        offset: int = 20,
        blur: int = 0,
    ) -> Image:
        noise_image = image.copy()
        offset = -offset
        contour = contours[0]  # Get Biggest

        # Step 1: Get center of contour or fallback to image center
        if contour is not None and len(contour.points) > 0:
            M = cv2.moments(contour.points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = image.shape[1] // 2, image.shape[0] // 2
        else:
            cx, cy = image.shape[1] // 2, image.shape[0] // 2  # noqa: F841

        # Step 2: Simulate outward shift (could be improved with pixelwise later)
        dx_vector = 1.0
        dy_vector = 1.0
        norm = math.hypot(dx_vector, dy_vector)
        comp_x = dx_vector / norm
        comp_y = dy_vector / norm

        # Step 3: Shift each channel
        channels = list(cv2.split(noise_image))
        for i, channel in enumerate(channels):
            dx = int(offset * (i + 1) * comp_x)
            dy = int(offset * (i + 1) * comp_y)

            M = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore
            channels[i] = cv2.warpAffine(
                channel,
                M,  # type: ignore
                (channel.shape[1], channel.shape[0]),
                borderMode=cv2.BORDER_REFLECT,
            )

        noise_image = cv2.merge(tuple(channels))  # type: ignore

        # Step 4: Optional blur
        if blur > 0:
            ksize = max(1, int(blur) // 2 * 2 + 1)  # Ensure it's odd
            noise_image = cv2.GaussianBlur(noise_image, (ksize, ksize), 0)

        # Step 5: Mask the result onto the original image
        return noise_image


@StyleRegistry.register()
class Noise(NoiseStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        alpha: float = 1,
        coloured: bool = True,
        intensity: float = 1,
        grain_size: int = 1,
        seed: int = 69,
    ) -> Image:
        np.random.seed(seed)
        image.copy()

        h, w, c = image.shape

        noise = np.random.normal(
            0,
            255 * intensity,
            (h // grain_size, w // grain_size, c),
        ).astype(np.uint8)  # type: ignore
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)  # type: ignore

        if not coloured:
            noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)  # type: ignore
            noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)  # type: ignore
        return cv2.addWeighted(noise, alpha, image, 1 - alpha, 0)


@StyleRegistry.register()
class DeNoise(NoiseStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        strength: int = 10,
    ) -> Image:
        return cv2.fastNlMeansDenoisingColored(image, h=strength)

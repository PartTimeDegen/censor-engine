import math
import cv2
import numpy as np
from censor_engine.models.styles import BlurStyle
from censor_engine.typing import CVImage


class ChromaticAberration(BlurStyle):
    style_name: str = "chromatic_aberration"

    def apply_style(
        self,
        image: CVImage,
        contour,
        offset: int = 20,
        angle: int = -45,
    ) -> CVImage:
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
            M = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore

            # Apply the shift using warpAffine (faster than np.roll)

            channels[i] = cv2.warpAffine(  # type: ignore
                channel,
                M,  # type: ignore
                (channel.shape[1], channel.shape[0]),
                borderMode=cv2.BORDER_REFLECT,
            )  # type: ignore

        # Merge the shifted channels back
        channels = tuple(channels)
        noise_image = cv2.merge(channels)  # type: ignore

        # Apply the effect to the masked area
        return self.draw_effect_on_mask([contour], noise_image, image)


class Noise(BlurStyle):
    style_name: str = "noise"

    def apply_style(
        self,
        image: CVImage,
        contour,
        alpha: int | float = 1,
        coloured: bool = True,
        intensity: float = 1,
        grain_size: int = 1,
    ) -> CVImage:
        noise_image = image.copy()

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
        noise_image = cv2.addWeighted(noise, alpha, image, 1 - alpha, 0)

        return self.draw_effect_on_mask([contour], noise_image, image)


class DeNoise(BlurStyle):
    style_name: str = "denoise"

    def apply_style(self, image: CVImage, contour, strength: int = 10) -> CVImage:
        noise_image = cv2.fastNlMeansDenoisingColored(image, h=strength)
        return self.draw_effect_on_mask([contour], noise_image, image)


effects = {
    "chromatic_aberration": ChromaticAberration,
    "noise": Noise,
    "denoise": DeNoise,
}

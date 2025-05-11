import cv2
from censor_engine.models.styles import StyliseStyle
from censor_engine.typing import CVImage


class Painting(StyliseStyle):
    style_name: str = "painting"

    def apply_style(
        self,
        image: CVImage,
        contour,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
    ) -> CVImage:
        new_image = cv2.stylization(image, sigma_s, sigma_r)  # type: ignore
        # Apply the effect to the masked area
        return self.draw_effect_on_mask([contour], new_image, image)


class Pencil(StyliseStyle):
    style_name: str = "pencil"

    def apply_style(
        self,
        image: CVImage,
        contour,
        coloured=False,
        sigma_s: int = 60,
        sigma_r: float = 0.45,
        shade_factor: float = 0.2,
    ) -> CVImage:
        grey, colour = cv2.pencilSketch(
            image, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor
        )  # type: ignore

        new_image = colour if coloured else cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

        # Apply the effect to the masked area
        return self.draw_effect_on_mask([contour], new_image, image)


effects = {
    "painting": Painting,
    "pencil": Pencil,
}

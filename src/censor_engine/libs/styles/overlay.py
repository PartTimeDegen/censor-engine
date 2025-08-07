import cv2

from censor_engine.detected_part import Part
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import OverlayStyle
from censor_engine.models.structs.colours import Colour
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask


@StyleRegistry.register()
class MissingStyle(OverlayStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
    ) -> Image:
        return image


@StyleRegistry.register()
class Overlay(OverlayStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        colour: tuple[int, int, int] | str = "WHITE",
        alpha: float = 1.0,
    ) -> Image:
        return self._apply_mask_as_overlay(image, mask, Colour(colour), alpha)


@StyleRegistry.register()
class Outline(OverlayStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        colour: tuple[int, int, int] | str = "WHITE",
        thickness: int = 2,
        softness: int = 0,
    ) -> Image:
        colour_obj = Colour(colour)

        # Extract points from your Contour objects
        contours_points = [contour.points for contour in contours]

        # Draw contours on a copy of the image
        cv2.drawContours(
            image,
            contours_points,
            -1,
            colour_obj.value,
            thickness,
            lineType=self.default_linetype,
        )

        if softness > 0:
            ksize = max(3, softness * 2 + 1)
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return image


@StyleRegistry.register()
class OutlinedOverlay(OverlayStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        colour_box: tuple[int, int, int] | str = "WHITE",
        colour_outline: tuple[int, int, int] | str = "BLACK",
        thickness: int = 2,
        alpha: float = 1.0,
        softness: int = 0,
    ) -> Image:
        overlay = self._apply_mask_as_overlay(
            image, mask, Colour(colour_box), alpha
        )

        contours_points = [contour.points for contour in contours]
        cv2.drawContours(
            overlay,
            contours_points,
            -1,
            Colour(colour_outline).value,
            thickness,
            lineType=self.default_linetype,
        )

        if softness > 0:
            ksize = max(3, softness * 2 + 1)
            overlay = cv2.GaussianBlur(overlay, (ksize, ksize), 0)

        return overlay

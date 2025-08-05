import cv2
import numpy as np

from censor_engine.models.structs.colours import Colour
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import BoxStyle

from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image


@StyleRegistry.register()
class MissingStyle(BoxStyle):
    def apply_style(self, image: Image, contour) -> Image:
        return image


@StyleRegistry.register()
class Overlay(BoxStyle):
    @staticmethod  # FIXME: Implement to Outlined Box
    def _apply_overlay(
        self,  # type: ignore
        image: Image,
        contour: Contour,
        colour: str | tuple[int, int, int] = "WHITE",
        alpha: float = 0.5,
    ) -> Image:
        colour_obj = Colour(colour)

        mask = np.zeros(image.shape, dtype=np.uint8)

        # Reverse Contour if Reversed
        if colour == (0, 0, 0):
            mask = cv2.bitwise_not(mask)  # type: ignore

        mask = cv2.drawContours(
            mask,
            contour.points,
            -1,
            colour_obj.value,
            -1,
            hierarchy=contour.hierarchy,  # type: ignore
            lineType=self.default_linetype,
        )  # type: ignore
        mask = np.where(mask == np.array(colour_obj.value), mask, image)

        return cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)

    def apply_style(
        self,
        image: Image,
        contour: Contour,
        colour="WHITE",
        alpha: float = 0.5,
    ) -> Image:
        return self._apply_overlay(self, image, contour, colour, alpha)


@StyleRegistry.register()
class Outline(BoxStyle):
    def _reverse_censor(self, image, contour):
        # Size Test for Reverse Censor
        dimensions_of_contour = cv2.boundingRect(contour[0][0])
        conditions = (
            dimensions_of_contour[0] == 0
            and dimensions_of_contour[1] == 0
            and dimensions_of_contour[2] >= (image.shape[1] - 1)
            and dimensions_of_contour[3] >= (image.shape[0] - 1)
        )
        if conditions:
            temp_mask = np.ones(image.shape, dtype=np.uint8) * 255
            temp_image = cv2.drawContours(
                temp_mask,  # type: ignore
                contour[0],
                -1,
                (0, 0, 0),
                -1,
                lineType=cv2.LINE_8,
            )
            if len(temp_image) > 2:
                temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)

            if temp_image.dtype != "unit8":
                temp_image = temp_image.astype(np.uint8)

            return cv2.findContours(
                image=temp_image,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )[0]

    def apply_style(
        self,
        image: Image,
        contour: Contour,
        colour: tuple[int, int, int] | str = "WHITE",
        thickness: int = 2,
        softness: int = 0,
    ) -> Image:
        colour_obj = Colour(colour)

        contour_shape = contour.points
        if self.using_reverse_censor:
            contour_shape = self._reverse_censor(image, contour)[0]  # type: ignore

        # Draw contours

        image = cv2.drawContours(
            image,
            contour_shape,  # type: ignore
            -1,
            colour_obj.value,
            thickness,
            lineType=self.default_linetype,
        )

        if softness == 0:
            return image

        # Ensure softness is at least 1
        softness = max(1, softness)

        # Calculate blur kernel size (must be odd and >1)
        ksize = max(3, softness * 2 + 1)

        # Apply Gaussian blur
        return cv2.GaussianBlur(image, (ksize, ksize), 0)


@StyleRegistry.register()
class Box(BoxStyle):
    def apply_style(
        self,
        image: Image,
        contour: Contour,
        colour: tuple[int, int, int] | str = "WHITE",
    ) -> Image:
        return contour.draw_contour(
            image,
            thickness=-1,
            colour=Colour(colour),
            linetype=self.default_linetype,
        )


@StyleRegistry.register()
class OutlinedBox(BoxStyle):
    def apply_style(
        self,
        image: Image,
        contour: Contour,
        colour_box: tuple[int, int, int] | str = "WHITE",
        colour_outline: tuple[int, int, int] | str = "BLACK",
        thickness: int = 2,
        alpha: float = 1.0,
    ) -> Image:
        image = Overlay._apply_overlay(  # FIXME: Isolate this
            self,
            image,
            contour,
            colour_box,
            alpha=alpha,
        )

        colour_obj = Colour(colour_outline)
        return cv2.drawContours(
            image,
            contour.points,
            -1,
            colour_obj.value,
            thickness,
            lineType=self.default_linetype,
        )

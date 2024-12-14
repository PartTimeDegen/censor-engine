import cv2
import numpy as np

from censorengine.backend.constants.colours import get_colour, rgb_to_bgr
from censorengine.lib_models.styles import BoxStyle

from censorengine.backend.constants.typing import CVImage


class MissingStyle(BoxStyle):
    style_name = "null"

    def apply_style(self, image: CVImage, contour) -> CVImage:
        return image


class Overlay(BoxStyle):
    style_name = "overlay"

    @staticmethod  # FIXME: Implement to Outlined Box
    def _apply_overlay(
        self,
        image,
        contour,
        colour="WHITE",
        alpha=1.0,
    ) -> CVImage:
        # FIXME: It colours the whole Image
        if isinstance(colour, str):
            colour = get_colour(colour)

        mask = np.zeros(image.shape, dtype=np.uint8)

        # Reverse Contour if Reversed
        if colour == (0, 0, 0):
            mask = cv2.bitwise_not(mask)  # type: ignore

        mask = cv2.drawContours(
            mask,
            contour[0],
            -1,
            rgb_to_bgr(colour),
            -1,
            hierarchy=contour[1],
            lineType=self.default_linetype,
        )  # type: ignore
        mask = np.where(mask == np.array(rgb_to_bgr(colour)), mask, image)

        return cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)

    def apply_style(
        self,
        image: CVImage,
        contour,
        colour="WHITE",
        alpha=1.0,
    ) -> CVImage:
        return self._apply_overlay(self, image, contour, colour, alpha)


class Outline(BoxStyle):
    style_name = "outline"

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
                temp_mask,
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
        image: CVImage,
        contour,
        colour: tuple[int, int, int] | str = "WHITE",
        thickness: int = 2,
    ) -> CVImage:
        colour = get_colour(colour)
        contour_shape = contour[0]

        if self.using_reverse_censor:
            contour_shape = self._reverse_censor(image, contour)

        return cv2.drawContours(
            image,
            contour_shape,
            -1,
            rgb_to_bgr(colour),
            thickness,
            lineType=self.default_linetype,
        )


class Box(BoxStyle):
    style_name = "box"

    def apply_style(
        self,
        image: CVImage,
        contour,
        colour: tuple[int, int, int] | str = "WHITE",
    ) -> CVImage:
        colour = get_colour(colour)
        contour = contour[0]
        return cv2.drawContours(
            image,
            contour,
            -1,
            rgb_to_bgr(colour),
            cv2.FILLED,
            lineType=self.default_linetype,
        )


class OutlinedBox(BoxStyle):
    style_name = "outlined_box"

    def apply_style(
        self,
        image: CVImage,
        contour,
        colour_box: tuple[int, int, int] | str = "WHITE",
        colour_outline: tuple[int, int, int] | str = "WHITE",
        thickness: int = 2,
        alpha: float = 1.0,
    ) -> CVImage:
        colour_box = get_colour(colour_box)
        colour_outline = get_colour(colour_outline)

        image = Overlay._apply_overlay(  # FIXME: Isolate this
            self,
            image,
            contour,
            colour_box,
            alpha=alpha,
        )

        colour = get_colour(colour_outline)
        contour = contour[0]
        return cv2.drawContours(
            image,
            contour,
            -1,
            rgb_to_bgr(colour),
            thickness,
            lineType=self.default_linetype,
        )


effects = {
    "overlay": Overlay,
    "outline": Outline,
    "box": Box,
    "outlined_box": OutlinedBox,
    "null": MissingStyle,
}

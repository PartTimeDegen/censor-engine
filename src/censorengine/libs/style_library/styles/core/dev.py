import cv2

from censorengine.backend.constants.colours import get_colour, rgb_to_bgr
from censorengine.lib_models.styles import DevStyle

from censorengine.backend.constants.typing import CVImage

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from censorengine.backend.models.structures.detected_part import Part


class Debug(DevStyle):
    style_name = "dev_debug"

    def _write_caption_text(
        self,
        image,
        text="Nope!",
        coordinates=(0, 0),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        text_colour=(0, 0, 0),
        background_colour=(255, 255, 255),
        fontscale=0.5,
        text_thickness=1,
        outline_thickness=0,
        margin=4,
    ):
        # Get New Lines
        list_text = text.split("\n")

        # Coords
        vertical_spacing = 1.5
        coords_text = (
            coordinates[0] + margin + outline_thickness,
            coordinates[1]
            + outline_thickness
            + cv2.getTextSize(text, font, fontscale, text_thickness)[0][1],
        )
        coords_box = (
            coords_text[0]
            + cv2.getTextSize(max(list_text, key=len), font, fontscale, text_thickness)[
                0
            ][0]
            + margin * 2,
            coords_text[1]
            + int(
                cv2.getTextSize(text, font, fontscale, text_thickness)[0][1]
                * len(list_text)
                * vertical_spacing
            ),
        )

        # Contrast
        is_bright = False
        for channel in background_colour:
            if channel > 124:
                is_bright = True
        if is_bright:
            text_colour = (
                255 - text_colour[0],
                255 - text_colour[1],
                255 - text_colour[2],
            )

        # Actions
        image_box = cv2.rectangle(
            image,
            (coordinates[0], coordinates[1]),
            coords_box,
            background_colour,
            thickness=-1,
        )
        for index, line in enumerate(list_text):
            image_box = cv2.putText(
                image_box,
                text=line,
                org=(
                    coords_text[0],
                    int(
                        coords_text[1]
                        + cv2.getTextSize(text, font, fontscale, text_thickness)[0][1]
                        * index
                        * vertical_spacing
                        + margin
                    ),
                ),
                fontFace=font,
                fontScale=fontscale,
                color=text_colour,
                thickness=text_thickness,
            )
        return image_box

    def apply_style(
        self,
        image: CVImage,
        contour,
        part: "Part",
        colour: tuple[int, int, int] | str = "WHITE",
    ) -> CVImage:
        colour = get_colour(colour)
        # Avoiding it for testing
        # if not part:
        #     return

        # Inputs
        text = f"{part.part_name}\nSCORE={float(part.score):0.2%}"
        coords = (
            part.relative_box[0],
            part.relative_box[1] + part.relative_box[3],
        )

        # Actions
        colour = rgb_to_bgr(colour)
        image_box = cv2.drawContours(
            image,
            contour[0],
            -1,
            color=colour,
            thickness=2,
            lineType=self.default_linetype,
        )  # type: ignore

        image_box = self._write_caption_text(
            image_box,
            text,
            coords,
            text_colour=(255, 255, 255),
            text_thickness=1,
            background_colour=colour,
            outline_thickness=2,
        )

        return image_box


effects = {
    "dev_debug": Debug,
}

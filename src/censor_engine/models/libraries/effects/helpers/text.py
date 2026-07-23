import cv2

from censor_engine._typing import Image
from censor_engine.structs.colours import Colour


class TextHelpers:
    def put_text(
        self,
        image: Image,
        text: str | list[str],
        coord_origin: tuple[int, int],
        color: Colour,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: int = 1,
        thickness: int = 2,
        line_type: int = cv2.LINE_AA,
        line_spacing: float = 1.2,
    ) -> Image:
        if isinstance(text, list):
            text = "\n".join(text)

        coord_x, coord_y = coord_origin
        for index, line in enumerate(text.split("\n")):
            y_line = int(
                coord_y
                + index
                * (
                    cv2.getTextSize(line, font, font_scale, thickness)[0][1]
                    * line_spacing
                ),
            )
            cv2.putText(
                image,
                line,
                (coord_x, y_line),
                font,
                font_scale,
                color.value,
                thickness,
                line_type,
            )
        return image

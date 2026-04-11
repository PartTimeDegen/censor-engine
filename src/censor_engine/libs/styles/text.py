from censor_engine.detected_part import Part
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.lib_models.styles import TextStyle
from censor_engine.models.structs.colours import Colour
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask


@StyleRegistry.register()
class CentreText(TextStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        text: str | list[str] = "Nope!",
        font: str = "arial",
        font_percent: float = 1.0,
        colour: tuple[int, int, int] | str = "PINK",
        outline_width: int = 3,
        outline_colour: tuple[int, int, int] | str | None = "WHITE",
    ) -> Image:
        colour_main = tuple(Colour(colour).value[::-1])
        if outline_colour is None:
            colour_outline = colour_main
        else:
            colour_outline = tuple(Colour(outline_colour).value[::-1])

        # Type Narrowing
        if isinstance(text, str):
            text = [text]

        # Get Mask Coords
        rel_box = contours[0].as_bounding_box()
        mask_coords = self._get_middle_coords(rel_box)

        # Get Font Size
        _, mask_size = self._convert_rel_box_to_coords_and_size(rel_box)

        for word in text:
            image = self._put_custom_font(
                image=image,
                word=word,
                coords=mask_coords,
                font=self.get_font(font),
                font_percent=font_percent,
                colour=colour_main,  # type: ignore
                mask_size=mask_size,
                outline_width=outline_width,
                outline_colour=colour_outline,  # type: ignore
            )

        return image


@StyleRegistry.register()
class CentreText(TextStyle):
    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        *,
        text: str | list[str] = "Nope!",
        font: str = "arial",
        font_percent: float = 1.0,
        colour: tuple[int, int, int] | str = "PINK",
        outline_width: int = 3,
        outline_colour: tuple[int, int, int] | str | None = "WHITE",
    ) -> Image:
        colour_main = tuple(Colour(colour).value[::-1])
        if outline_colour is None:
            colour_outline = colour_main
        else:
            colour_outline = tuple(Colour(outline_colour).value[::-1])

        # Type Narrowing
        if isinstance(text, str):
            text = [text]

        # Get Mask Coords
        rel_box = contours[0].as_bounding_box()
        mask_coords = self._get_middle_coords(rel_box)

        # Get Font Size
        _, mask_size = self._convert_rel_box_to_coords_and_size(rel_box)

        for word in text:
            image = self._put_custom_font(
                image=image,
                word=word,
                coords=mask_coords,
                font=self.get_font(font),
                font_percent=font_percent,
                colour=colour_main,  # type: ignore
                mask_size=mask_size,
                outline_width=outline_width,
                outline_colour=colour_outline,  # type: ignore
            )

        return image

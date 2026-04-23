import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image as PImage
from PIL import ImageDraw, ImageFont

from censor_engine.api.effects import EffectContext
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.lib_models.effects import TextEffect
from censor_engine.models.structs.colours import Colour
from censor_engine.typing import Image, ProcessedImage


@EffectRegistry.register()
class DevText(TextEffect):
    # Basic Vector Helpers
    def _add_cords(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> tuple[int, int]:
        return (a[0] + b[0], a[1] + b[1])

    def _subtract_cords(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> tuple[int, int]:
        return (a[0] - b[0], a[1] - b[1])

    # Aids to Text Randomly being Bottom Left rather than Top Left
    def _normalise_text_coord(
        self,
        coords: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[int, int]:
        return (coords[0], coords[1] - size[1])

    def _get_centre(self, size: tuple[int, int]) -> tuple[int, int]:
        return (int(size[0] * 0.5), int(size[1] * 0.5))

    def _convert_rel_box_to_coords_and_size(
        self, rel_box: tuple[int, int, int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        coords = rel_box[:2]
        size = rel_box[2:]
        return coords, size

    def _get_middle_coords(
        self,
        rel_box: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        coords, size = self._convert_rel_box_to_coords_and_size(rel_box)
        half_size = self._get_centre(size)
        return (
            coords[0] + half_size[0],
            coords[1] + half_size[1],
        )

    def _convert_middle_to_bottom_left_coords(
        self,
        coords: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[int, int]:
        middle = self._get_centre(size)
        return (
            coords[0] - middle[0],
            coords[1] + middle[1],
        )

    # Pillow Module for Custom Fonts
    def _put_custom_font(
        self,
        image: Image,
        word: str,
        coords: tuple[int, int],
        font: str,
        font_percent: float,
        colour: tuple[int, int, int],
        mask_size: tuple[int, int],
        outline_width: int,
        outline_colour: tuple[int, int, int],
    ):
        # Convert to Pillow
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = PImage.fromarray(img)

        # Create Draw Object
        draw = ImageDraw.Draw(img_pil)

        # Test Word on Base Size
        base_font_loaded = ImageFont.truetype(font + ".ttf", 20)
        base_bbox = draw.textbbox(
            (0, 0),
            word,
            font=base_font_loaded,
            font_size=20,
        )
        base_sizes = (
            base_bbox[2] - base_bbox[0],
            base_bbox[3] - base_bbox[1],
        )

        # Calculate New Font Size
        font_size = int(
            font_percent
            * 20
            * min(
                (
                    mask_size[0] / base_sizes[0],
                    mask_size[1] / base_sizes[1],
                )
            )
        )
        font_loaded = ImageFont.truetype(font + ".ttf", font_size)
        bbox = draw.textbbox(
            (0, 0),
            word,
            font=font_loaded,
            font_size=font_size,
        )

        # Align to Middle
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        top_left_coords = (
            int(coords[0] - text_w * 0.5 - bbox[0]),
            int(coords[1] - text_h * 0.5 - bbox[1]),
        )

        # Draw Text
        draw.text(
            top_left_coords,
            word,
            font=font_loaded,
            fill=colour,
            stroke_width=outline_width,
            stroke_fill=outline_colour,
        )
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def apply_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        text: str | list[str] = "Nope!",
        font: str = "arial",
        font_percent: float = 1.0,
        colour: tuple[int, int, int] | str = "PINK",
        outline_width: int = 3,
        outline_colour: tuple[int, int, int] | str | None = "WHITE",
    ) -> ProcessedImage:
        colour_main = tuple(Colour(colour).value[::-1])
        if outline_colour is None:
            colour_outline = colour_main
        else:
            colour_outline = tuple(Colour(outline_colour).value[::-1])

        # Type Narrowing
        if isinstance(text, str):
            text = [text]

        # Text Settings
        fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
        font_path = None
        for font_name in fonts:
            if font in font_name:
                font_path = font_name
                break
        if font_path is None:
            msg = f"Trying to use unavailable font! {fonts}"
            raise ValueError(msg)

        # Get TypeMask Coords
        rel_box = effect_context.contours[0].as_bounding_box()
        mask_coords = self._get_middle_coords(rel_box)

        # Get Font Size
        _, mask_size = self._convert_rel_box_to_coords_and_size(rel_box)

        for word in text:
            effect_context.image = self._put_custom_font(
                image=effect_context.image,
                word=word,
                coords=mask_coords,
                font=font,
                font_percent=font_percent,
                colour=colour_main,  # type: ignore
                mask_size=mask_size,
                outline_width=outline_width,
                outline_colour=colour_outline,  # type: ignore
            )

        return effect_context.image

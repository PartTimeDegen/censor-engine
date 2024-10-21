import cv2
import numpy as np

from censorengine.backend.constants.colours import get_colour, rgb_to_bgr


def _write_text(
    image,
    text="Nope!",
    coordinates=(0, 0),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    colour=(0, 0, 0),
    fontscale=1,
    text_thickness=1,
    outline_thickness=0,
    margin=2,
):
    if colour != (0, 0, 0):
        colour = get_colour(colour)

    coords = (
        coordinates[0] + margin * 2 + outline_thickness,
        coordinates[1]
        + cv2.getTextSize(text, font, fontscale, text_thickness)[1] * 2
        + margin * 2,
    )
    return cv2.putText(
        image,
        text=text,
        org=coords,
        fontFace=font,
        fontScale=fontscale,
        color=colour,
        thickness=text_thickness,
    )


def _write_caption_text(
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
        + cv2.getTextSize(
            max(list_text, key=len), font, fontscale, text_thickness
        )[0][0]
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
                    + cv2.getTextSize(text, font, fontscale, text_thickness)[
                        0
                    ][1]
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


def overlay(image, contours, rgb_colour, alpha=1.0):
    # FIXME: Colours the whole Image
    if isinstance(rgb_colour, str):
        rgb_colour = get_colour(rgb_colour)

    mask = np.zeros(image.shape, dtype=np.uint8)

    # Reverse Contour if Reversed
    if rgb_colour == (0, 0, 0):
        mask = cv2.bitwise_not(mask)

    mask = cv2.drawContours(
        mask,
        contours[0],
        -1,
        rgb_to_bgr(rgb_colour),
        -1,
        hierarchy=contours[1],
        lineType=cv2.LINE_AA,
    )
    mask = np.where(mask == np.array(rgb_to_bgr(rgb_colour)), mask, image)

    return cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)


def outline(
    image,
    contour: tuple,
    rgb_colour: tuple[int, int, int] | str = "WHITE",
    pxl_thickness: int = 2,
):
    rgb_colour = get_colour(rgb_colour)

    contour = contour[0]
    return cv2.drawContours(
        image,
        contour,
        -1,
        rgb_to_bgr(rgb_colour),
        pxl_thickness,
        lineType=cv2.LINE_AA,
    )


def box(
    image,
    contour: tuple,
    rgb_colour: tuple[int, int, int] | str = "WHITE",
):

    return outline(
        image,
        contour,
        rgb_colour,
        pxl_thickness=cv2.FILLED,
    )


def outlined_box(
    image,
    box: list[tuple[int, int]],
    rgb_colour_box: tuple[int, int, int] = "WHITE",
    rgb_colour_outline: tuple[int, int, int] = "WHITE",
    pxl_thickness: int = 2,
    alpha: float = 1.0,
):
    rgb_colour_box = get_colour(rgb_colour_box)
    rgb_colour_outline = get_colour(rgb_colour_outline)

    image = overlay(
        image,
        box,
        rgb_colour_box,
        alpha=alpha,
    )
    return outline(
        image=image,
        contour=box,
        rgb_colour=rgb_colour_outline,
        pxl_thickness=pxl_thickness,
    )


def dev_debug(
    image,
    contour: tuple,
    part,
    rgb_colour: tuple[int, int, int] = "WHITE",
    pxl_thickness: int = 2,
):
    rgb_colour = get_colour(rgb_colour)

    # Inputs
    text = f"{part['class']}\nSCORE={float(part['score']):0.2%}"
    coords = (int(part["box"][0]), int(part["box"][1] + part["box"][3]))

    # Actions
    colour = rgb_to_bgr(rgb_colour)
    image_box = outline(image, contour, rgb_colour, pxl_thickness)

    image_box = _write_caption_text(
        image_box,
        text,
        coords,
        text_colour=(255, 255, 255),
        text_thickness=1,
        background_colour=colour,
        outline_thickness=pxl_thickness,
    )

    return image_box


effects = {
    "overlay": overlay,
    "outline": outline,
    "box": box,
    "outlined_box": outlined_box,
    "dev_debug": dev_debug,
}

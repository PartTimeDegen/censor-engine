from statistics import fmean

import cv2

from censor_engine.detected_part import Part
from censor_engine.libs.detectors.box_based_detectors.nude_net import (
    NudeNetDetector,
)
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.enums import MergeMethod
from censor_engine.models.lib_models.styles import DevStyle
from censor_engine.models.structs.colours import Colour, _colours
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask

# ruff: noqa

colour_dict = dict(
    zip(
        NudeNetDetector.model_classifiers,
        list(_colours.keys())[3:],  # offset due to a weird glitch with black
        strict=False,
    ),
)


def _get_contours_from_mask(mask: Mask) -> list[Contour]:
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [
        Contour(
            points=cnt,
            hierarchy=hierarchy[0][i] if hierarchy is not None else None,
        )
        for i, cnt in enumerate(contours)
    ]


def draw_text_below_box(
    img,
    text,
    box_x,
    box_y,
    box_w,
    max_box_h,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    min_scale=0.1,
    max_scale=5.0,
    min_width_ratio=0.5,  # 10% minimum width ratio of box_w
    color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=1,
    line_spacing=1.2,
    padding=4,
):
    lines = text.split("\n")

    def measure_block(scl):
        sizes = [
            cv2.getTextSize(line, font, scl, thickness)[0] for line in lines
        ]
        max_line_width = max(w for w, _ in sizes)
        total_height = sum(h for _, h in sizes) + int(
            (len(lines) - 1) * sizes[0][1] * (line_spacing - 1),
        )
        return max_line_width, total_height

    base_width, base_height = measure_block(1.0)
    if base_width == 0 or base_height == 0:
        return img, (box_w, max_box_h), 0

    # Scale to fit width
    scale_width = (box_w - 2 * padding) / base_width
    # Scale to fit max height
    scale_height = (max_box_h - 2 * padding) / base_height
    scale = min(scale_width, scale_height)

    # Enforce minimum width ratio
    min_scale_width = (box_w * min_width_ratio) / base_width
    scale = max(scale, min_scale_width)

    # Clamp scale
    scale = max(min_scale, min(scale, max_scale))

    # Measure at chosen scale
    text_w, text_h = measure_block(scale)

    # Actual height box = text height + padding, capped by max_box_h
    used_box_h = min(int(text_h + 2 * padding), max_box_h)

    # Draw background box BELOW the original bounding box at (box_x, box_y)
    cv2.rectangle(
        img,
        (box_x, box_y),
        (box_x + box_w, box_y + used_box_h),
        bg_color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    y_offset = box_y + padding
    for line in lines:
        (line_w, line_h), baseline = cv2.getTextSize(
            line,
            font,
            scale,
            thickness,
        )
        x_offset = box_x + padding
        y_offset += line_h
        cv2.putText(
            img,
            line,
            (x_offset, y_offset),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y_offset += int(line_h * (line_spacing - 1))

    return img, (box_w, used_box_h), scale


@StyleRegistry.register()
class Debug(DevStyle):
    is_done: bool = False

    def apply_style(
        self,
        image: Image,
        mask: Mask,
        contours: list[Contour],
        part: Part,
        part_list: list[Part],
    ) -> Image:
        if part.config.rendering_settings.merge_method != MergeMethod.NONE:
            msg = "Requires No Merging"
            raise ValueError(msg)

        if self.is_done:
            return image

        # First loop — draw contours
        for part_obj in part_list:
            colour_obj = Colour(colour_dict[part_obj.get_name()])
            linetype = cv2.LINE_4
            contours_points = [
                contour.points
                for contour in _get_contours_from_mask(part_obj.mask)
            ]
            cv2.drawContours(
                image,
                contours_points,
                -1,
                colour_obj.value,
                thickness=2,
                lineType=linetype,
            )

        # Second loop — draw scaled multi-line text
        for part_obj in part_list:
            colour_obj = Colour(colour_dict[part_obj.get_name()])
            text = f"{part_obj.part_name}\nSCORE={float(part_obj.score):0.1%}"

            box_x = part_obj.relative_box[0]
            box_y = (
                part_obj.relative_box[1] + part_obj.relative_box[3]
            )  # bottom of the original bounding box
            box_w = part_obj.relative_box[2]

            # Decide max height for the text box (for example, 10%-15% of image height or fixed pixels)
            max_text_box_h = int(image.shape[0] * 0.1)

            text_color = (
                Colour("WHITE")
                if int(fmean(colour_obj.value)) <= 80
                else Colour("BLACK")
            )

            # We'll compute the text box height dynamically inside the helper but cap it at max_text_box_h

            # Modified helper to accept max height and return used height
            image, (final_w, final_h), used_scale = draw_text_below_box(
                image,
                text,
                box_x,
                box_y,
                box_w,
                max_text_box_h,
                color=text_color.value,
                bg_color=colour_obj.value,
                thickness=1,
                padding=4,
            )

        self.is_done = True
        return image

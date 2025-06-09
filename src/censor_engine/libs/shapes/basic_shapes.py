import cv2

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censor_engine.typing import Mask
    from censor_engine.detected_part import Part

from censor_engine.models.shapes import Shape


class Box(Shape):
    shape_name: str = "box"
    base_shape: str = "box"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        box = part.part_area.region.get_corners()
        mask = cv2.rectangle(empty_mask, box[0], box[1], (255, 255, 255), -1)  # type: ignore
        return mask


class Circle(Shape):
    shape_name: str = "circle"
    base_shape: str = "circle"
    single_shape: str = "circle"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        mask = cv2.circle(
            empty_mask,
            part.part_area.region.centre.convert_to_tuple(),
            min(part.part_area.region.radius),
            (255, 255, 255),
            -1,
        )
        return mask


class Ellipse(Shape):
    shape_name: str = "ellipse"
    base_shape: str = "ellipse"
    single_shape: str = "ellipse"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        mask = cv2.ellipse(
            empty_mask,  # type: ignore
            part.part_area.region.centre.convert_to_tuple(),
            part.part_area.region.radius,
            0,
            0,
            360,
            color=(255, 255, 255),
            thickness=-1,
        )
        return mask


class RoundedBox(Shape):
    shape_name: str = "rounded_box"
    base_shape: str = "rounded_box"
    single_shape: str = "rounded_box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        box = part.part_area.region.get_corners()
        mask = cv2.rectangle(empty_mask, box[0], box[1], (255, 255, 255), -1)  # type: ignore

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        iterations = 2

        mask_changed = cv2.erode(
            mask,
            kernel,
            iterations=iterations >> 1,
        )
        mask_changed = cv2.dilate(
            mask_changed,
            kernel,
            iterations=iterations,
        )

        return mask_changed


shapes = {
    "box": Box,
    "ellipse": Ellipse,
    "circle": Circle,
    "rounded_box": RoundedBox,
}

import cv2

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import Mask
    from censorengine.backend.models.structures.detected_part import Part


from censorengine.lib_models.shapes import Shape


class Box(Shape):
    shape_name: str = "box"
    base_shape: str = "box"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        box = part.box
        mask = cv2.rectangle(empty_mask, box[0], box[1], (255, 255, 255), -1)  # type: ignore
        return mask


class Circle(Shape):
    shape_name: str = "circle"
    base_shape: str = "circle"
    single_shape: str = "circle"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        box = part.box

        centre = (
            int((box[0][0] + box[1][0]) / 2),
            int((box[0][1] + box[1][1]) / 2),
        )

        radius = min(
            abs(int((box[0][0] - box[1][0]) / 2)),
            abs(int((box[0][1] - box[1][1]) / 2)),
        )
        mask = cv2.circle(empty_mask, centre, radius, (255, 255, 255), -1)
        return mask


class Ellipse(Shape):
    shape_name: str = "ellipse"
    base_shape: str = "ellipse"
    single_shape: str = "ellipse"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        box = part.box
        centre = (
            int((box[0][0] + box[1][0]) / 2),
            int((box[0][1] + box[1][1]) / 2),
        )

        radius = (
            abs(int((box[0][0] - box[1][0]) / 2)),
            abs(int((box[0][1] - box[1][1]) / 2)),
        )
        mask = cv2.ellipse(
            empty_mask,
            centre,
            radius,
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
        box = part.box
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

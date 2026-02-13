from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from censor_engine.detected_part import Part
    from censor_engine.typing import Mask

from censor_engine.libs.registries import ShapeRegistry
from censor_engine.models.lib_models.shapes import Shape


@ShapeRegistry.register()
class Box(Shape):
    base_shape: str = "box"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        box = part.part_area.region.get_corners()
        return cv2.rectangle(empty_mask, box[0], box[1], (255, 255, 255), -1)  # type: ignore


@ShapeRegistry.register()
class Circle(Shape):
    base_shape: str = "circle"
    single_shape: str = "circle"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        return cv2.circle(
            empty_mask,
            part.part_area.region.centre.convert_to_tuple(),
            min(part.part_area.region.radius),
            (255, 255, 255),
            -1,
        )


@ShapeRegistry.register()
class Ellipse(Shape):
    base_shape: str = "ellipse"
    single_shape: str = "ellipse"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        return cv2.ellipse(
            empty_mask,  # type: ignore
            part.part_area.region.centre.convert_to_tuple(),
            part.part_area.region.radius,
            0,
            0,
            360,
            color=(255, 255, 255),
            thickness=-1,
        )


@ShapeRegistry.register()
class RoundedBox(Shape):
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
        return cv2.dilate(
            mask_changed,
            kernel,
            iterations=iterations,
        )

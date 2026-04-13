from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from censor_engine.detected_part import Part
    from censor_engine.typing import TypeMask

from censor_engine.libs.registries import MaskRegistry
from censor_engine.models.lib_models.masks import Mask


@MaskRegistry.register()
class Box(Mask):
    base_mask: str = "Box"
    single_mask: str = "Box"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
        box = part.part_area.region.get_corners()
        return cv2.rectangle(empty_mask, box[0], box[1], (255, 255, 255), -1)  # type: ignore


@MaskRegistry.register()
class Circle(Mask):
    base_mask: str = "Circle"
    single_mask: str = "Circle"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
        return cv2.circle(
            empty_mask,
            part.part_area.region.centre.convert_to_tuple(),
            min(part.part_area.region.radius),
            (255, 255, 255),
            -1,
        )


@MaskRegistry.register()
class Ellipse(Mask):
    base_mask: str = "Ellipse"
    single_mask: str = "Ellipse"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
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


@MaskRegistry.register()
class RoundedBox(Mask):
    base_mask: str = "RoundedBox"
    single_mask: str = "RoundedBox"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
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

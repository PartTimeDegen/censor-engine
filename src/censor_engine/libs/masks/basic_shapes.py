from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from censor_engine.typing import TypeMask

from censor_engine.api.masks import MaskContext
from censor_engine.libs.registries import MaskRegistry
from censor_engine.models.lib_models.masks import Mask


@MaskRegistry.register()
class Box(Mask):
    base_mask: str = "Box"
    single_mask: str = "Box"

    def generate(self, mask_context: MaskContext) -> "TypeMask":
        box = mask_context.part.part_area.region.get_corners()
        return cv2.rectangle(
            mask_context.empty_mask, box[0], box[1], (255, 255, 255), -1
        )  # type: ignore


@MaskRegistry.register()
class Circle(Mask):
    base_mask: str = "Circle"
    single_mask: str = "Circle"

    def generate(self, mask_context: MaskContext) -> "TypeMask":
        return cv2.circle(
            mask_context.empty_mask,
            mask_context.part.part_area.region.centre.convert_to_tuple(),
            min(mask_context.part.part_area.region.radius),
            (255, 255, 255),
            -1,
        )


@MaskRegistry.register()
class Ellipse(Mask):
    base_mask: str = "Ellipse"
    single_mask: str = "Ellipse"

    def generate(self, mask_context: MaskContext) -> "TypeMask":
        return cv2.ellipse(
            mask_context.empty_mask,  # type: ignore
            mask_context.part.part_area.region.centre.convert_to_tuple(),
            mask_context.part.part_area.region.radius,
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

    def generate(self, mask_context: MaskContext) -> "TypeMask":
        box = mask_context.part.part_area.region.get_corners()
        mask = cv2.rectangle(
            mask_context.empty_mask, box[0], box[1], (255, 255, 255), -1
        )  # type: ignore

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

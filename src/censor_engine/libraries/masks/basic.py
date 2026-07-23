import cv2

from censor_engine._typing import MaskImage
from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.libraries.masks.masks import BasicMask
from censor_engine.models.libraries.masks.schemas import MaskContext


@MaskRegistry.register()
class Box(BasicMask):
    base_mask = "Box"
    single_mask = "Box"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        bbox = mask_context.part_properties.bbox
        if bbox is None:
            return mask_context.empty_mask

        points = bbox.points

        return cv2.rectangle(
            mask_context.empty_mask,
            points[0],
            points[1],
            **mask_context.cv2_standard_settings,
        )  # type: ignore


@MaskRegistry.register()
class Circle(BasicMask):
    base_mask = "Circle"
    single_mask = "Circle"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        bbox = mask_context.part_properties.bbox
        if bbox is None:
            return mask_context.empty_mask

        return cv2.circle(
            mask_context.empty_mask,
            center=bbox.centre,  # type: ignore
            radius=max(bbox.radius),  # type: ignore
            **mask_context.cv2_standard_settings,
        )  # type: ignore


@MaskRegistry.register()
class Ellipse(BasicMask):
    base_mask = "Ellipse"
    single_mask = "Ellipse"

    def generate_mask(  # type: ignore
        self,
        mask_context: MaskContext,
        *,
        scale: float = 1.0,
    ) -> MaskImage:
        bbox = mask_context.part_properties.bbox
        if bbox is None:
            return mask_context.empty_mask

        min_size = min(bbox.axes)
        axes = [int(scale * x) if x == min_size else x for x in bbox.axes]

        return cv2.ellipse(
            img=mask_context.empty_mask,  # type: ignore
            center=bbox.centre,
            axes=axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            **mask_context.cv2_standard_settings,
        )  # type: ignore


@MaskRegistry.register()
class ThickEllipse(BasicMask):
    base_mask = "Ellipse"
    single_mask = "Ellipse"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return Ellipse().generate_mask(mask_context=mask_context, scale=1.5)


@MaskRegistry.register()
class ThinEllipse(BasicMask):
    base_mask = "Ellipse"
    single_mask = "Ellipse"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return Ellipse().generate_mask(mask_context=mask_context, scale=0.5)


@MaskRegistry.register()
class RoundedBox(BasicMask):
    base_mask = "RoundedBox"
    single_mask = "RoundedBox"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        mask = Box().generate_mask(mask_context=mask_context)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        iterations = 2

        mask_changed = cv2.erode(
            mask,
            kernel,
            iterations=iterations >> 1,
        )
        return cv2.dilate(mask_changed, kernel, iterations=iterations)  # type: ignore

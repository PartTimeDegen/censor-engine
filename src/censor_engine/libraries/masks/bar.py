import cv2
import numpy as np

from censor_engine._typing import MaskImage
from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.libraries.masks.bar_structs import (
    BarInfo,
    BoundingInfo,
)
from censor_engine.models.libraries.masks.masks import BarMask
from censor_engine.models.libraries.masks.schemas import MaskContext


@MaskRegistry.register()
class Bar(BarMask):
    base_mask = "Ellipse"
    joint_mask = "JointBox"
    single_mask = "Bar"

    def _draw_bar(
        self,
        mask_context: MaskContext,
        bounding_info: BoundingInfo,
    ):
        # Get Bar Length
        # NOTE: Creating a bar twice as long as the diagonal length of the
        #       image avoids the issue where you can tell the bar shows on the
        #       ends.
        #
        bar_length = int(np.hypot(*mask_context.image_shape)) * 2
        bar_thickness = bounding_info.bar_thickness

        # Handle Better Thickness for Forced Orientations
        thickness = bounding_info.handle_forced_directions()
        bar_thickness = thickness if thickness is not None else bar_thickness

        rect_dimensions = (
            bounding_info.centre,
            (bar_length, bar_thickness),
            bounding_info.angle,
        )
        box = cv2.boxPoints(rect_dimensions).astype(np.int32)

        mask = mask_context.empty_mask.copy()
        cv2.fillPoly(mask, [box], 255)  # type: ignore

        return mask

    def generate_mask(  # type: ignore # TODO: Not wrong, not easy
        self,
        mask_context: MaskContext,
        *,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> MaskImage:
        if not mask_context.is_merged and not force_vertical:
            force_horizontal = True
        # Find Contours via Joint Ellipse
        contours = self.get_contours(mask_context, mode=cv2.RETR_EXTERNAL)

        if len(contours) == 0:
            return mask_context.empty_mask

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 4:  # noqa: PLR2004
            return mask_context.empty_mask

        # Get Bar Info
        bar_info = BarInfo(
            force_horizontal=force_horizontal,
            force_vertical=force_vertical,
            is_tight_bar=tight_bar,
            is_long_direction=long_direction,
            file_uuid=mask_context.file_uuid,
        )

        # Get Bounding Info
        bounding_info = BoundingInfo.create_bounding_info(
            contours=cnt,  # type: ignore
            bar_info=bar_info,
        )

        bounding_info.fix_angles()
        bounding_info.snap_angle()

        # Force Specific Orientation if Set
        if force_horizontal:
            bounding_info.angle = 0
        elif force_vertical:
            bounding_info.angle = 90

        # Cache Bar Angle to Maintain Consistent Bar Angle across Parts
        if BarInfo.bar_angle is None:
            BarInfo.bar_angle = bounding_info.angle
        else:
            bounding_info.check_if_dimensions_flip(BarInfo.bar_angle)
            bounding_info.angle = BarInfo.bar_angle

        return self._draw_bar(mask_context, bounding_info)


@MaskRegistry.register()
class HorizontalBar(Bar):
    def generate_mask(  # type: ignore
        self,
        mask_context: MaskContext,
        *,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> MaskImage:
        return super().generate_mask(
            mask_context,
            force_horizontal=True,
            tight_bar=tight_bar,
        )


@MaskRegistry.register()
class VerticalBar(Bar):
    def generate_mask(  # type: ignore
        self,
        mask_context: MaskContext,
        *,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> MaskImage:
        return super().generate_mask(
            mask_context,
            force_vertical=True,
            tight_bar=tight_bar,
        )


@MaskRegistry.register()
class LongBar(Bar):
    def generate_mask(  # type: ignore
        self,
        mask_context: MaskContext,
        *,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> "MaskImage":
        return super().generate_mask(mask_context, long_direction=True)


@MaskRegistry.register()
class EllipseBasedBar(Bar):
    joint_mask = "JointEllipse"

    def generate_mask(  # type: ignore
        self,
        mask_context: MaskContext,
        *,
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> MaskImage:
        return super().generate_mask(mask_context, tight_bar=True)

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import cv2
import numpy as np

from censor_engine.models.lib_models.shapes import BarShape

if TYPE_CHECKING:
    from censor_engine.detected_part import Part
    from censor_engine.typing import Mask

from censor_engine.libs.registries import ShapeRegistry


class _BarInfo:
    # Values
    bar_angle: float | None = None

    # Settings
    force_already_determined: bool = False
    force_horizontal: bool = False
    force_vertical: bool = False

    # Meta
    file_uuid: UUID = uuid4()


@ShapeRegistry.register()
class Bar(BarShape, _BarInfo):
    base_shape: str = "ellipse"
    joint_shape: str = "joint_box"
    single_shape: str = "bar"

    # Controls
    deg_angle_snap: int = 1

    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",  # TODO: Update Shapes with same format as Styles
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> "Mask":
        if not part.is_merged:
            force_horizontal = True
        # Find Contours via Joint Ellipse
        contours, _ = cv2.findContours(
            part.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            msg = f"No contours found in mask: {part.part_name}"
            raise ValueError(msg)

        # Fit Ellipse to get Contour Back
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            msg = "Not enough points to fit an ellipse."
            raise ValueError(msg)

        if tight_bar:
            ellipse = cv2.fitEllipse(cnt)
            centre, axes, angle = ellipse  # axes = (major, minor)
        else:
            rect = cv2.minAreaRect(cnt)
            centre, (w, h), angle = rect
            if w < h:
                axes = (h, w)
                if abs(angle) > self.deg_angle_snap:
                    angle += 90
            else:
                axes = (w, h)

        # Handle cv2 having a weird Axes System
        if long_direction or tight_bar:
            angle += 90

        # Normalize Angle between 0 and 180
        corrected_angle = angle % 180

        # Snap Angle if too close to Horizontal or Vertical
        if (
            abs(corrected_angle - 0) <= self.deg_angle_snap
            or abs(corrected_angle - 180) <= self.deg_angle_snap
        ):
            corrected_angle = 0
        elif abs(corrected_angle - 90) <= self.deg_angle_snap:
            corrected_angle = 90

        # Force Specific Orientation if Set
        if force_horizontal:
            corrected_angle = 0
        elif force_vertical:
            corrected_angle = 90

        # Create A Blank Mask And Draw The Rotated Rectangle Bar
        height, width = part.mask.shape

        bar_length = (
            int(np.hypot(width, height)) * 2
        )  # long enough to span image plus any offset
        bar_thickness = int(max(axes)) if long_direction else int(min(axes))

        # Handle Better Thickness for Forced Orientations
        if force_horizontal or force_vertical:
            _, _, w, h = cv2.boundingRect(cnt)
            if force_horizontal:
                bar_thickness = int(h)  # height of bounding box
            elif force_vertical:
                bar_thickness = int(w)  # width of bounding box

        rect = (centre, (bar_length, bar_thickness), corrected_angle)
        box = cv2.boxPoints(rect).astype(np.int32)

        mask = empty_mask
        cv2.fillPoly(mask, [box], 255)  # type: ignore

        # Save for Other Parts
        if self.bar_angle:
            angle = self.bar_angle
        else:
            self.bar_angle = angle

        return mask


@ShapeRegistry.register()
class HorizontalBar(Bar):
    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> "Mask":
        return super().generate(part, empty_mask, force_horizontal=True)


@ShapeRegistry.register()
class VerticalBar(Bar):
    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> "Mask":
        return super().generate(part, empty_mask, force_vertical=True)


@ShapeRegistry.register()
class LongBar(Bar):
    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> "Mask":
        return super().generate(part, empty_mask, long_direction=True)


@ShapeRegistry.register()
class EllipseBasedBar(Bar):
    joint_shape: str = "joint_ellipse"

    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
        force_horizontal: bool = False,
        force_vertical: bool = False,
        long_direction: bool = False,
        tight_bar: bool = False,
    ) -> "Mask":
        return super().generate(part, empty_mask, tight_bar=True)

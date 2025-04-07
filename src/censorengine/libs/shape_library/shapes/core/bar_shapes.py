from uuid import UUID, uuid4

import cv2
import numpy as np

from censorengine.lib_models.shapes import BarShape
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import Mask
    from censorengine.backend.models.structures.detected_part import Part


class _BarInfo:
    # Values
    bar_gradient: Optional[float] = None

    # Settings
    force_already_determined: bool = False
    force_horizontal: bool = False
    force_vertical: bool = False

    # Meta
    file_uuid: UUID = uuid4()


class Bar(BarShape, _BarInfo):
    shape_name: str = "bar"
    base_shape: str = "ellipse"
    single_shape: str = "bar"

    # Controls
    landscape_qualifier: float = 1.2
    cutoff_horizontal_gradient: float = 0.2

    def _get_bar_coord_info(self, sorted_points):
        """Extract relevant bar coordinates and tilt direction."""
        top_point = sorted_points[0]
        point_one, point_two = sorted_points[1:3]

        # Use squared distances to avoid sqrt for efficiency
        radius_one_sq = (top_point[0] - point_one[0]) ** 2 + (
            top_point[1] - point_one[1]
        ) ** 2
        radius_two_sq = (top_point[0] - point_two[0]) ** 2 + (
            top_point[1] - point_two[1]
        ) ** 2

        side_point, length_point = (
            (point_one, point_two)
            if radius_one_sq < radius_two_sq
            else (point_two, point_one)
        )

        # Tilt direction based on y-difference
        tilt = 1 if (top_point[1] - length_point[1]) < 0 else -1

        dict_corners = {
            "top_left": sorted_points[1] if tilt > 0 else sorted_points[0],
            "top_right": sorted_points[0] if tilt > 0 else sorted_points[1],
            "bottom_left": sorted_points[3] if tilt > 0 else sorted_points[2],
            "bottom_right": sorted_points[2] if tilt > 0 else sorted_points[3],
            "top_point": top_point,
            "length_point": length_point,
            "side_point": side_point,
            "bottom_point": sorted_points[3],
            "tilt": tilt,
        }

        dx = dict_corners["bottom_left"][0] - dict_corners["top_left"][0]
        dy = dict_corners["top_right"][1] - dict_corners["top_left"][1]
        dict_corners["part_height_width_ratio"] = dx / dy if dy else 0

        return dict_corners

    @staticmethod
    def _line(gradient, coords, x_coord):
        x1, y1 = coords[1], coords[0]
        if gradient == float("inf"):  # Handle vertical line (infinite gradient)
            return int(y1), int(x_coord)
        else:
            return int(gradient * (x_coord - x1) + y1), int(x_coord)

    def _reset_bar(self, new_file_path):
        Bar.bar_gradient = None
        Bar.force_already_determined = False
        Bar.force_horizontal = False
        Bar.force_vertical = False
        Bar.file_path = new_file_path

    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
        force_horizontal=False,
        force_vertical=False,
    ) -> "Mask":
        """Generates the bar mask based on detected contours."""
        if part.file_uuid != self.file_uuid:
            self._reset_bar(part.file_uuid)

        contours = cv2.findContours(
            part.mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        points = (
            cv2.boxPoints(cv2.minAreaRect(np.vstack(contours).squeeze())).tolist()
            if contours
            else []
        )

        if not points:
            return part.mask

        sorted_points = np.array(points, dtype=np.int32)
        dict_corners = self._get_bar_coord_info(sorted_points)

        # Compute gradient with error handling (avoid repeated code)
        frac_bottom = dict_corners["top_right"][0] - dict_corners["top_left"][0]
        if frac_bottom == 0:
            gradient = float("inf")  # Infinite gradient for vertical lines
        else:
            gradient = -(dict_corners["top_right"][1] - dict_corners["top_left"][1]) / (
                frac_bottom
            )

        height, width = empty_mask.shape[:2]
        is_landscape = width > (self.landscape_qualifier * height)
        is_gradient_low = abs(gradient) < self.cutoff_horizontal_gradient

        if is_gradient_low:
            dict_corners["tilt"] = 0

        # Simplified logic for gradient assignment and bar shape orientation
        gradient = Bar.bar_gradient if Bar.bar_gradient is not None else gradient
        if not Bar.bar_gradient:
            Bar.bar_gradient = gradient
        Bar.force_already_determined = True

        should_be_vertical = is_landscape and is_gradient_low
        should_be_horizontal = is_gradient_low

        if force_horizontal:
            Bar.force_horizontal = True
            Bar.force_vertical = (
                False  # Ensure it doesn't force vertical at the same time
            )
        elif force_vertical:
            Bar.force_vertical = True
            Bar.force_horizontal = (
                False  # Ensure it doesn't force horizontal at the same time
            )
        elif not Bar.force_already_determined:
            Bar.force_horizontal = should_be_horizontal
            Bar.force_vertical = should_be_vertical

        Bar.force_already_determined = True  # Lock the determination

        # Define bar shape based on orientation
        if Bar.force_horizontal:
            y_top, y_bottom = (
                dict_corners["top_point"][1],
                dict_corners["bottom_point"][1],
            )
            if dict_corners["tilt"] > 0:
                y_top, y_bottom = (
                    dict_corners["top_left"][1],
                    dict_corners["bottom_right"][1],
                )
            elif dict_corners["tilt"] < 0:
                y_top, y_bottom = (
                    dict_corners["top_right"][1],
                    dict_corners["bottom_left"][1],
                )

            bar_points = [(width, y_top), (0, y_top), (0, y_bottom), (width, y_bottom)]

        elif Bar.force_vertical:
            x_left, x_right = (
                dict_corners["top_point"][0],
                dict_corners["bottom_point"][0],
            )
            bar_points = [
                (x_right, 0),
                (x_left, 0),
                (x_left, height),
                (x_right, height),
            ]

        else:
            # Optimized multiple cases for gradient-based calculation
            top_right_bl = (
                dict_corners["length_point"]
                if dict_corners["tilt"] > 0
                else dict_corners["top_right"]
            )
            top_left_br = (
                dict_corners["bottom_left"]
                if dict_corners["tilt"] > 0
                else dict_corners["top_left"]
            )

            bar_points = [
                self._line(Bar.bar_gradient, top_right_bl, 0),
                self._line(Bar.bar_gradient, top_left_br, 0),
                self._line(Bar.bar_gradient, top_left_br, height),
                self._line(Bar.bar_gradient, top_right_bl, height),
            ]

        # Convert to NumPy and draw polygon efficiently
        mask = cv2.fillPoly(
            empty_mask,  # type: ignore
            [np.array(bar_points, dtype=np.int32)],
            (255, 255, 255),
            lineType=cv2.LINE_AA,
        )
        return mask


class HorizontalBar(Bar):
    shape_name: str = "horizontal_bar"
    base_shape: str = "ellipse"
    single_shape: str = "horizontal_bar"

    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
        force_horizontal: bool = False,
        force_vertical: bool = False,
    ) -> "Mask":
        return super().generate(part, empty_mask, force_horizontal=True)


class VerticalBar(Bar):
    shape_name: str = "vertical_bar"
    base_shape: str = "ellipse"
    single_shape: str = "vertical_bar"

    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
        force_horizontal: bool = False,
        force_vertical: bool = False,
    ) -> "Mask":
        return super().generate(part, empty_mask, force_vertical=True)


shapes = {
    "bar": Bar,
    "horizontal_bar": HorizontalBar,
    "vertical_bar": VerticalBar,
}

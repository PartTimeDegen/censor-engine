import math

import cv2
import numpy as np

from censorengine.lib_models.shapes import BarShape
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import Part, Mask


class _BarInfo:
    # Values
    bar_gradient: Optional[float] = None

    # Settings
    force_already_determined: bool = False
    force_horizontal: bool = False
    force_vertical: bool = False

    # Meta
    file_path: str = ""


class Bar(BarShape, _BarInfo):
    shape_name: str = "bar"
    base_shape: str = "ellipse"
    single_shape: str = "bar"

    # Controls
    landscape_qualifier: float = 1.2
    cutoff_horizontal_gradient: float = 0.2

    # Bar Info
    def _get_bar_coord_info(self, sorted_points):
        """
        We know top_right is higher than top_left (top_left.y > top_right.y),
        therefore tilt can be found to be positive if top_right.x > top_left.x as
        well. However if rotation is so much that the bottom_right is higher than
        top_left, then the calculation is invalid. Therefore tests need to be done
        to determine which is higher.

        We do not know which point is which, so by taking the middle two points and
        comparing them to the top to find the one with the one with the shortest
        distance, we can find a "side" of the rectangle, which means the other one
        must be the other top.

        """
        # Get Known Points
        top_point = sorted_points[0]

        # Calculate Radii
        point_one = sorted_points[1]
        point_two = sorted_points[2]

        radius_one = math.sqrt(
            (top_point[0] - point_one[0]) ** 2
            + (top_point[1] - point_one[1]) ** 2
        )
        radius_two = math.sqrt(
            (top_point[0] - point_two[0]) ** 2
            + (top_point[1] - point_two[1]) ** 2
        )

        # Find Side
        if radius_one < radius_two:
            side_point = point_one
            length_point = point_two
        else:
            side_point = point_two
            length_point = point_one

        # Determine Which Side the Length Point is on via Delta X
        # NOTE: Positive means it's on the left of the top point, therefore the
        #       tilt is positive, and vice versa for if it's one the right
        #
        delta_x = top_point[1] - length_point[1]
        eq_tilt = delta_x < 0
        tilt = 1 if eq_tilt else -1

        # Write Points
        dict_corners = {}
        if tilt > 0:
            dict_corners["top_left"] = sorted_points[1]
            dict_corners["top_right"] = sorted_points[0]

            dict_corners["bottom_left"] = sorted_points[3]
            dict_corners["bottom_right"] = sorted_points[2]

        else:
            dict_corners["top_left"] = sorted_points[0]
            dict_corners["top_right"] = sorted_points[1]

            dict_corners["bottom_left"] = sorted_points[2]
            dict_corners["bottom_right"] = sorted_points[3]

        # Useful Stuff
        dict_corners["top_point"] = top_point
        dict_corners["length_point"] = length_point
        dict_corners["side_point"] = side_point
        dict_corners["bottom_point"] = sorted_points[3]
        dict_corners["tilt"] = tilt
        dict_corners["part_height_width_ratio"] = (
            dict_corners["bottom_left"][0] - dict_corners["top_left"][0]
        ) / (dict_corners["top_right"][1] - dict_corners["top_left"][1])

        return dict_corners

    def _line(self, gradient, coords, x_coord_to_find_y):
        """
        The format of this is [Y, X] (technically [-Y, X] compared to the standard
        axis format).

        The calculation for Y is the point-slope formula:

            y - y_1 = m * (x - x_1)

        Which is when y is the subject:

            y = m * (x - x_1) + y_1

        """
        x_one = coords[1]
        y_one = coords[0]
        x_base = x_coord_to_find_y
        point_y = gradient * (x_base - x_one) + y_one
        coords = [int(point_y), int(x_base)]

        return tuple(coords)

    def _reset_bar(self, new_file_path):
        # Values
        Bar.bar_gradient = None

        # Settings
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
        # Check if Bar Info should be Reset
        if part.file_path != self.file_path:
            self._reset_bar(part.file_path)

        cont_boxes = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(cont_boxes[0]) >= 1:
            points = cv2.boxPoints(
                cv2.minAreaRect(np.vstack(cont_boxes[0]).squeeze())
            ).tolist()
        else:
            try:
                points = cont_boxes[0][0].tolist()
                points = tuple([tuple(point[0]) for point in points])
            except Exception:
                points = cont_boxes[0]

        if len(points) == 0:
            return part.mask

        points = [[int(point[0]), int(point[1])] for point in points]

        # Determine Points, Tilt, and Gradient
        # # Points
        sorted_points = sorted(points, key=lambda coord: (coord[0], coord[1]))

        # # Tilt
        dict_corners = self._get_bar_coord_info(sorted_points)

        # # Calculate Gradient
        try:
            gradient = -(
                dict_corners["top_right"][1] - dict_corners["top_left"][1]
            ) / (dict_corners["top_right"][0] - dict_corners["top_left"][0])
        except ZeroDivisionError:
            gradient = 0.0

        # # Check for if it Makes More Sense to Inverse Gradient
        HEIGHT = empty_mask.shape[0]
        WIDTH = empty_mask.shape[1]
        is_image_landscape = WIDTH > (self.landscape_qualifier * HEIGHT)

        # # Calculate Cutoffs
        is_gradient_in_horizontal_cutoff = (
            abs(gradient) < self.cutoff_horizontal_gradient
        )

        # # Determine if Tilt is even (used for single parts)
        if is_gradient_in_horizontal_cutoff:
            dict_corners["tilt"] = 0

        # # Save Gradient if not Use Saved Gradient
        if not Bar.bar_gradient:
            Bar.bar_gradient = gradient
        else:
            gradient = Bar.bar_gradient
            Bar.force_already_determined = True

        # # Declaring Booleans
        should_be_vertical = False
        should_be_horizontal = is_gradient_in_horizontal_cutoff

        # # Check for Force Settings
        if not Bar.force_already_determined:
            if is_image_landscape and is_gradient_in_horizontal_cutoff:
                should_be_vertical = True
                should_be_horizontal = False

            if force_horizontal or should_be_horizontal:
                Bar.force_horizontal = True

            if force_vertical or should_be_vertical:
                Bar.force_vertical = True

        if Bar.force_horizontal and Bar.force_vertical:
            if force_horizontal:
                Bar.force_vertical = False
            elif force_vertical:
                Bar.force_horizontal = False
            elif should_be_vertical:
                Bar.force_horizontal = False
            else:
                Bar.force_vertical = False

        # # Horizontal
        if Bar.force_horizontal:
            # List Coords for Readability
            y_top = dict_corners["top_point"][1]
            y_bottom = dict_corners["bottom_point"][1]
            x_left = 0
            x_right = empty_mask.shape[1]

            # Overwrite for Tilt
            if dict_corners["tilt"] > 0:
                y_top = dict_corners["top_left"][1]
                y_bottom = dict_corners["bottom_right"][1]
            elif dict_corners["tilt"] < 0:
                y_top = dict_corners["top_right"][1]
                y_bottom = dict_corners["bottom_left"][1]

            bar_points = [
                (x_right, y_top),
                (x_left, y_top),
                (x_left, y_bottom),
                (x_right, y_bottom),
            ]

        # # Vertical
        elif Bar.force_vertical:
            # List Coords for Readability
            x_left = dict_corners["top_point"][0]
            x_right = dict_corners["bottom_point"][0]
            y_top = 0
            y_bottom = empty_mask.shape[0]

            bar_points = [
                (x_right, y_top),
                (x_left, y_top),
                (x_left, y_bottom),
                (x_right, y_bottom),
            ]

        # # Angled
        else:
            if Bar.bar_gradient is None:
                raise ValueError

            # Multiple Part Positive Tilt
            if dict_corners["tilt"] > 0:
                top_right_bottom_left_point = dict_corners["length_point"]
                top_left_bottom_right_point = dict_corners["bottom_left"]

            # Single Part Negative Tilt
            elif dict_corners["tilt"] == 0 and Bar.bar_gradient > 0:
                top_right_bottom_left_point = dict_corners["top_right"]
                top_left_bottom_right_point = dict_corners["bottom_left"]

            # Single Part Positive Tilt
            elif dict_corners["tilt"] == 0 and Bar.bar_gradient < 0:
                top_right_bottom_left_point = dict_corners["bottom_right"]
                top_left_bottom_right_point = dict_corners["top_left"]

            # Multiple Part Negative Tilt
            else:
                top_right_bottom_left_point = dict_corners["top_right"]
                top_left_bottom_right_point = dict_corners["bottom_left"]

            bar_points = [
                self._line(
                    gradient=Bar.bar_gradient,
                    coords=top_right_bottom_left_point,
                    x_coord_to_find_y=0,
                ),  # Top Right Corner
                self._line(
                    gradient=Bar.bar_gradient,
                    coords=top_left_bottom_right_point,
                    x_coord_to_find_y=0,
                ),  # Top Left Corner
                self._line(
                    gradient=Bar.bar_gradient,
                    coords=top_left_bottom_right_point,
                    x_coord_to_find_y=empty_mask.shape[0],
                ),  # Bottom Right Corner
                self._line(
                    gradient=Bar.bar_gradient,
                    coords=top_right_bottom_left_point,
                    x_coord_to_find_y=empty_mask.shape[0],
                ),  # Bottom Left Corner
            ]

        np_bar_points = np.array(bar_points, dtype=np.int32)

        mask = cv2.fillPoly(
            img=empty_mask,
            pts=[np_bar_points],
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )  # type: ignore

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

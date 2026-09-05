from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from censor_engine._typing import Image
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.colours import Colour


class UnitShapeTemplates:
    TRIANGLE_A = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    TRIANGLE_B = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
    TRIANGLE_C = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    TRIANGLE_D = np.array([[0, 0], [1, 1], [0, 1]], dtype=np.int32)


class PolygonHelpers:
    shape_templates: UnitShapeTemplates

    def __init__(self):
        self.shape_templates = UnitShapeTemplates()

    def _fill_polygon(
        self,
        polygon: Image,
        output: Image,
        image: Image,
        outline_colour: Colour,
        outline_width: int,
    ):
        # Get ROI Dims
        x, y, width, height = cv2.boundingRect(polygon)
        if width <= 1 or height <= 1:
            return

        # Clamps
        x_start = max(x, 0)
        y_start = max(y, 0)
        x_end = min(x + width, image.shape[1])
        y_end = min(y + height, image.shape[0])

        if x_start >= x_end or y_start >= y_end:
            return

        width = x_end - x_start
        height = y_end - y_start

        # Get a ROI of the area of the Polygon
        roi = image[y_start:y_end, x_start:x_end]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Find the Colour of the Polygon Area
        shifted = polygon.copy()
        shifted[:, 0] -= x_start
        shifted[:, 1] -= y_start
        shifted[:, 0] = np.clip(shifted[:, 0], 0, width - 1)
        shifted[:, 1] = np.clip(shifted[:, 1], 0, height - 1)

        cv2.fillPoly(mask, [shifted], 1)  # type: ignore

        # Save the Colour of the Polygon
        colour = cv2.mean(roi, mask=mask)[:3]

        # Fill the Colour
        cv2.fillPoly(output, [polygon], colour)  # type: ignore

        # Add Lines if Included
        if outline_width:
            cv2.polylines(
                output,  # type: ignore
                [polygon],
                color=outline_colour.value,
                thickness=outline_width,
                isClosed=True,
            )

    def generate_polygons(
        self,
        effect_context: EffectContext,
        outline_width: int,
        outline_colour: tuple[int, int, int] | str,
        polygon_function: Callable[[int, int], list[np.ndarray]],
    ):
        # Get Helper Variables
        image = effect_context.image
        height, width = effect_context.image_shape

        # Create Polygons
        polygons = polygon_function(width, height)

        # Fill Polygons with Colours
        output_image = np.zeros_like(image)
        outline_colour_obj = Colour(outline_colour)

        fill = self._fill_polygon
        args = [
            output_image,
            image,
            outline_colour_obj,
            outline_width,
        ]
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda polygon: fill(polygon, *args), polygons))  # type:ignore

        return output_image

from dataclasses import dataclass

import cv2
import numpy as np

from censor_engine.constant import DIM_COLOUR
from censor_engine.models.structs.colours import Colour
from censor_engine.typing import Image, Mask


@dataclass
class Contour:
    points: np.ndarray
    hierarchy: np.ndarray | None = None

    def as_min_area_box(self) -> np.ndarray:
        rect = cv2.minAreaRect(self.points.astype(np.float32))
        box = cv2.boxPoints(rect)
        return box.astype(np.intp)

    def as_bounding_box(self) -> tuple[int, int, int, int]:
        return cv2.boundingRect(self.points)  # type: ignore # Type is correct

    def as_mask(
        self,
        image_shape: tuple[int, int],
        mask_thickness: int = -1,
    ) -> Mask:
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(
            mask,
            [self.points],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=mask_thickness,
        )
        return mask  # type: ignore

    def draw_contour(
        self,
        image: Image | Mask,
        thickness: int,
        linetype: int,
        colour: Colour | None = None,
    ) -> Mask:
        if not colour:
            colour = Colour()

        return cv2.drawContours(
            image,
            [self.points],
            contourIdx=-1,
            color=colour.value,
            thickness=thickness,
            hierarchy=self.hierarchy,  # type: ignore
            lineType=linetype,
        )


class ContourNormalizer:
    def __init__(
        self,
        mode: int = cv2.RETR_TREE,
        method: int = cv2.CHAIN_APPROX_SIMPLE,
    ) -> None:
        self.mode = mode
        self.method = method

    def from_mask(self, binary_mask: np.ndarray) -> list[Contour]:
        # OpenCV findContours returns (contours, hierarchy) in OpenCV 4.x+
        contours, hierarchy = cv2.findContours(
            binary_mask,
            self.mode,
            self.method,
        )

        if hierarchy is not None:
            hierarchy = hierarchy[0]  # shape (N, 4)

        normalized_contours = []
        for i, contour in enumerate(contours):
            flat = (
                contour.squeeze(axis=1)
                if contour.ndim == DIM_COLOUR
                else contour
            )  # shape (N, 2)
            norm = Contour(
                points=flat,
                hierarchy=hierarchy[i] if hierarchy is not None else None,
            )
            normalized_contours.append(norm)

        return normalized_contours

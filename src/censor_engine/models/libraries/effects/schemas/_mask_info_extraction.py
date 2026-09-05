from dataclasses import InitVar, dataclass, field

import cv2
import numpy as np

from censor_engine._typing import MaskImage

BoxPoints = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


@dataclass(slots=True)
class _MinimumAreaRectangle:
    flat_contours: InitVar[np.typing.NDArray]

    centre: tuple[float, float] = field(init=False)
    dimensions: tuple[float, float] = field(init=False)
    angle: float = field(init=False)

    # Calculated
    height: float = field(init=False)
    width: float = field(init=False)
    size: float = field(init=False)

    box_points: BoxPoints = field(init=False)

    def __post_init__(self, flat_contours: np.typing.NDArray):
        rect = cv2.minAreaRect(flat_contours)  # type: ignore
        self.centre, self.dimensions, self.angle = rect  # type: ignore

        self.height = self.dimensions[0]
        self.width = self.dimensions[1]

        self.box_points = np.intp(cv2.boxPoints(rect))  # type: ignore
        self.size = self.width * self.height


@dataclass(slots=True)
class _BoundingRectangle:
    flat_contours: InitVar[np.typing.NDArray]

    x: float = field(init=False)
    y: float = field(init=False)
    height: float = field(init=False)
    width: float = field(init=False)

    # Calculated
    size: float = field(init=False)

    box_points: BoxPoints = field(init=False)

    def __post_init__(self, flat_contours: np.typing.NDArray):
        rect = cv2.boundingRect(flat_contours)  # type: ignore
        self.x, self.y, self.width, self.height = rect  # type: ignore

        x, y, w, h = rect

        self.box_points = np.array(
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ],
            dtype=np.intp,
        )  # type: ignore
        self.size = w * h


@dataclass(slots=True)
class MaskInfo:
    mask: InitVar[MaskImage]

    # Calculated
    minimum_area: _MinimumAreaRectangle = field(init=False)
    bounding_box: _BoundingRectangle = field(init=False)

    def __post_init__(self, mask: MaskImage):
        cont, _ = cv2.findContours(
            image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        flat_contours = np.vstack(cont).squeeze()
        self.minimum_area = _MinimumAreaRectangle(flat_contours)
        self.bounding_box = _BoundingRectangle(flat_contours)

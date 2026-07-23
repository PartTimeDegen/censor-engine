from dataclasses import dataclass
from typing import ClassVar, Self, cast
from uuid import UUID, uuid4

import cv2
import numpy as np

from censor_engine.models.libraries.masks.constants import (
    MINIMUM_POINTS_FOR_FIT_ELLIPSE,
)


@dataclass
class BarInfo:
    # Settings
    force_horizontal: bool
    force_vertical: bool

    is_tight_bar: bool
    is_long_direction: bool

    file_uuid: UUID

    # Values to be Edited by Class
    force_already_determined: bool = False
    bar_angle: float | None = None
    flip_thickness_for_angle: bool = False

    # Controls
    deg_angle_snap: int = 1

    # Meta
    _stored_uuid: ClassVar[UUID] = uuid4()

    def __post_init__(self):
        if self.file_uuid != self._stored_uuid:
            type(self)._stored_uuid = self.file_uuid  # noqa: SLF001
            type(self).bar_angle = None
            self.flip_thickness_for_angle = False


@dataclass(slots=True)
class BoundingInfo:
    # Coord Stuff
    centre: tuple[float, float]
    dimensions: tuple[float, float]
    angle: float

    # Useful Info
    contours: np.typing.NDArray[np.int32]
    bar_info: BarInfo
    is_tight: bool

    # Meta

    _used_class_method: bool = False

    def __post_init__(self):
        if not self._used_class_method:
            msg = "Instance can't be made without the class method!"
            raise ValueError(msg)

        self.centre = (round(self.centre[0]), round(self.centre[1]))
        self.dimensions = (
            round(self.dimensions[0]),
            round(self.dimensions[1]),
        )
        self.angle = round(self.angle, 3)

    @classmethod
    def create_bounding_info(
        cls,
        contours: np.typing.NDArray[np.int32],
        bar_info: BarInfo,
    ) -> Self:

        if (
            bar_info.is_tight_bar
            and len(contours) >= MINIMUM_POINTS_FOR_FIT_ELLIPSE
        ):
            fitted_shape = cv2.fitEllipse(contours)
            is_tight = True
        else:
            fitted_shape = cv2.minAreaRect(contours)
            is_tight = False

        return cls(
            centre=cast("tuple[float, float]", fitted_shape[0]),
            dimensions=cast("tuple[float, float]", fitted_shape[1]),
            angle=fitted_shape[2],
            contours=contours,
            bar_info=bar_info,
            is_tight=is_tight,
            _used_class_method=True,
        )

    @property
    def is_wider_than_taller_rectangle(self) -> bool:
        w, h = self.dimensions
        return w < h

    @property
    def bar_thickness(self) -> int:
        return int(
            max(self.dimensions)
            if self.bar_info.is_long_direction
            or self.bar_info.flip_thickness_for_angle
            else min(self.dimensions)
        )

    @property
    def is_forced_direction(self) -> bool:
        return self.bar_info.force_horizontal or self.bar_info.force_vertical

    # Fix Angle Stuff
    def _fix_rectangle_axes(self) -> None:
        if self.bar_info.is_tight_bar:
            return

        if self.is_wider_than_taller_rectangle:
            if abs(self.angle) > self.bar_info.deg_angle_snap:
                self.angle += 90
        else:
            list_dim = list(self.dimensions)
            self.dimensions = tuple(list_dim[::-1])  # type: ignore

    def _fix_long_direction_axes(self) -> None:
        if self.bar_info.is_long_direction or self.is_tight:
            self.angle += 90

    def _normalise_angle(self) -> None:
        self.angle = float(self.angle % 180)

    # Public Methods
    def fix_angles(self) -> None:
        self._fix_rectangle_axes()
        self._fix_long_direction_axes()
        self._normalise_angle()

    def snap_angle(self) -> None:
        snap_angle = self.bar_info.deg_angle_snap
        snap_horizontal = (
            abs(self.angle - 0) <= snap_angle
            or abs(self.angle - 180) <= snap_angle
        )
        snap_vertical = abs(self.angle - 90) <= self.bar_info.deg_angle_snap

        # Snapping
        if snap_horizontal:
            self.angle = 0.0
        if snap_vertical:
            self.angle = 90.0

    def handle_forced_directions(self) -> int | None:
        _, _, width, height = cv2.boundingRect(self.contours)
        if self.bar_info.force_horizontal:
            return int(height)
        if self.bar_info.force_vertical:
            return int(width)

        return None

    def check_if_dimensions_flip(self, saved_angle: float):
        difference = abs((self.angle - saved_angle + 180) % 360 - 180)
        angle_flip = 90
        if difference >= angle_flip:
            self.bar_info.flip_thickness_for_angle = True

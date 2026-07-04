from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from censor_engine._typing import MaskImage
from censor_engine.models.libraries.configs.settings.schemas import Margins


class AbsoluteBBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_xyxy(
        cls,
        bbox: tuple[int, int, int, int],
    ) -> AbsoluteBBox:
        """
        Create a bounding box from absolute coordinates.

        :param bbox: (x1, y1, x2, y2)
        """
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

    @classmethod
    def from_xywh(
        cls,
        bbox: tuple[int, int, int, int],
    ) -> AbsoluteBBox:
        """
        Create a bounding box from absolute coordinates.

        :param bbox: (x1, y1,w, h)
        """
        return cls(
            x1=bbox[0],
            y1=bbox[1],
            x2=bbox[0] + bbox[2],
            y2=bbox[1] + bbox[3],
        )

    @classmethod
    def from_cxcywh(
        cls,
        bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> AbsoluteBBox:
        """
        Create a bounding box from normalized YOLO coordinates.

        :param bbox: (center_x, center_y, width, height)
        :param image_width: Image width in pixels.
        :param image_height: Image height in pixels.
        """
        cx, cy, w, h = bbox

        cx *= image_width
        cy *= image_height
        w *= image_width
        h *= image_height

        return cls(
            x1=round(cx - (w / 2)),
            y1=round(cy - (h / 2)),
            x2=round(cx + (w / 2)),
            y2=round(cy + (h / 2)),
        )

    # Information
    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """
        Return the center point.

        :returns: (center_x, center_y)
        """
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    # Outputs
    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """
        Return the bounding box as absolute coordinates.

        :returns: (x1, y1, x2, y2)
        """
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        """
        Return the bounding box in XYWH format.

        :returns: (x, y, width, height)
        """
        return (self.x1, self.y1, self.width, self.height)

    # Methods
    def to_cxcywh(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        """
        Convert to normalized YOLO coordinates.

        :param image_width: Image width in pixels.
        :param image_height: Image height in pixels.
        :returns: (center_x, center_y, width, height)
        """
        cx, cy = self.center

        return (
            cx / image_width,
            cy / image_height,
            self.width / image_width,
            self.height / image_height,
        )

    def rescale_bbox(self, margins: Margins) -> AbsoluteBBox:
        """
        Expand a bounding box according to relative margin values.

        Margins may be provided as a single scalar applied to both
        dimensions or as a dictionary containing separate `width`
        and `height` factors.

        Args:
            input_bbox (AbsoluteBBox): Bounding box to expand.

            margins (float | dict[str, float]): Relative expansion
                factors.

        Returns:
            AbsoluteBBox: Expanded bounding box.

        """
        # Get the Margin Data Depending on Type
        print(margins, type(margins))
        w_margin = margins.width
        h_margin = margins.height

        # Get The Differences in Width and Height
        x, y, width, height = self.xywh
        dw = int(width * w_margin)
        dh = int(height * h_margin)

        return AbsoluteBBox.from_xywh(
            (x - dw // 2, y - dh // 2, width + dw, height + dh)
        )


class DetectorOutput(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            np.ndarray: lambda v: v.tolist(),
        },
    }

    origin: str

    # Internal
    part_id: int = 0

    # Meta
    score: float  # TODO Make this so if it's missing, set to 100%
    label: str | None = None

    # Data Used for Information
    bbox: AbsoluteBBox | None = None  # XYXY
    masks: list[MaskImage] | None = None

    def set_part_id(self, number: int) -> None:
        self.part_id = number

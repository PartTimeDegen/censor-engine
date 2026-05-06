from dataclasses import dataclass, field
from typing import Literal

import cv2

from censor_engine.constant import DIM_COLOUR, DIM_GREY
from censor_engine.detected_part import Part
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask


@dataclass(slots=True)
class EffectContext:
    # Tools
    image: Image
    mask: Mask
    contours: list[Contour]
    part: Part | None
    part_list: list[Part] | None = None
    shape: tuple[int, int, int] = field(init=False)
    original_image: Image = field(init=False)

    # Params
    alpha: float = 1.0
    fade_width: int = 0
    fade_gradient_mode: Literal["linear", "gaussian"] = "linear"
    mask_thickness: int = -1

    def __post_init__(self):
        # Handle Shape
        shape = self.image.shape
        if len(shape) == DIM_GREY:
            shape = (shape[0], shape[1], 1)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        if len(shape) != DIM_COLOUR:
            msg = f"This shouldn't happen: {len(shape)=}"
            raise ValueError(msg)

        self.shape = shape
        self.original_image = self.image.copy()

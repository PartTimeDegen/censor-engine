from dataclasses import dataclass, field

import cv2

from censor_engine._typing import Image, MaskImage
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.structs.contours import Contour

from .constants import OBLIQUE


@dataclass(slots=True)
class GeneralParameters:
    alpha: float = OBLIQUE


@dataclass(slots=True)
class EffectContext:
    # Tools
    image: Image
    mask: MaskImage

    part_properties: PartProperties

    mask_bool: Image = field(init=False)
    original_image: Image = field(init=False)
    contours: list[Contour] = field(init=False)

    # Settings
    settings: GeneralParameters = field(default_factory=GeneralParameters)

    def __post_init__(self):
        self.mask_bool = self.mask > 0  # type: ignore
        self.original_image = self.image.copy()

        self.contours = self._get_contours_from_mask()

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.mask.shape[:2]

    def _get_contours_from_mask(self) -> list[Contour]:

        contours, hierarchy = cv2.findContours(
            self.mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return [
            Contour(
                points=cnt,
                hierarchy=hierarchy[0][i] if hierarchy is not None else None,
            )
            for i, cnt in enumerate(contours)
        ]

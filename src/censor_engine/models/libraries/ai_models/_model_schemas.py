from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from censor_engine._typing import BBox, Image, MaskImage


class ModelOutput(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    label: str | None = None
    score: float | None = None
    bbox: tuple[int, int, int, int] | None = None  # Pydantic didn't like BBox
    masks: list[MaskImage] | None = None

    def get_attributes(self) -> dict:
        return self.model_dump(exclude_none=True)


@dataclass(slots=True)
class ROIOutput:
    local_bbox: BBox
    original_image: Image

    # Created
    crop: Image = field(init=False)
    size: tuple[int, int] = field(init=False)

    def __post_init__(self):
        # Safety
        self.local_bbox = self.local_bbox.astype(int)  # type: ignore

        # Get Crop
        x1, y1, x2, y2 = self.local_bbox
        self.crop = self.original_image[y1:y2, x1:x2]

        # Get Size
        self.size = self.original_image.shape[:2]

    def convert_crop_mask_to_full_mask(
        self, mask_crop: MaskImage
    ) -> MaskImage:
        x1, y1, x2, y2 = self.local_bbox
        full_mask = np.zeros(self.size, dtype=bool)
        full_mask[y1:y2, x1:x2] = mask_crop
        return full_mask

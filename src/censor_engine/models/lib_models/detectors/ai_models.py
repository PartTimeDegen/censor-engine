from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

from censor_engine.typing import BBox, Image, Mask


class ModelOutput(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    label: str | None = None
    score: float | None = None
    bbox: np.ndarray | None = None  # Pydantic didn't like BBox
    masks: list[Mask] | None = None

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
        self.local_bbox = self.local_bbox.astype(int)

        # Get Crop
        x1, y1, x2, y2 = self.local_bbox
        self.crop = self.original_image[y1:y2, x1:x2]

        # Get Size
        self.size = self.original_image.shape[:2]

    def convert_crop_mask_to_full_mask(self, mask_crop: Mask) -> Mask:
        x1, y1, x2, y2 = self.local_bbox
        full_mask = np.zeros(self.size, dtype=bool)
        full_mask[y1:y2, x1:x2] = mask_crop
        return full_mask


class AIModel(ABC):
    model_path: str

    # Internal
    _model: Any = None
    _device: int | str = "cpu"

    # Cache Handling
    _image_count: int = 0
    _cache_limit: int = 1000

    def convert_image_to_roi(
        self, box: BBox, image: Image
    ) -> ROIOutput | None:
        return ROIOutput(
            original_image=image,
            local_bbox=box,
        )

    @abstractmethod
    def initiate_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image: Image,
        rois: list[ROIOutput] | None = None,
    ) -> dict | list:
        raise NotImplementedError

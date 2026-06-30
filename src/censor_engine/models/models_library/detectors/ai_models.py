from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

from censor_engine.models.lib_models.detectors.schemas import (
    DetectedPart,
)
from censor_engine.typing import BBox, Image, Mask


class ModelOutput(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    label: str | None = None
    score: float | None = None
    bbox: tuple[int, int, int, int] | None = None  # Pydantic didn't like BBox
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
        self.local_bbox = self.local_bbox.astype(int)  # type: ignore

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
    # Model Stuff
    model_path: str
    model_name: str
    model_classifiers: tuple[str, ...]

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
    def predict(self, image: Image) -> list[DetectedPart]:
        raise NotImplementedError

    def predict_with_rois(
        self,
        rois: list[ROIOutput],
    ) -> list[DetectedPart]:
        """
        This function is used primarily with the YOLOSeg "focused_roi"
        function, since sometimes, especially if the total image is too "big"
        (i.e., negative space if that's the right word, basically not nudity
        stuff), using this function let's the image focus on the person in
        particular which promises better results.
        # TODO: Fix this.

        :param list[ROIOutput] rois: List of ROIs from YoloSeg

        :return list[DetectedPartSchema]: List of all of the parts.
        """
        # Get Outputs per ROI
        # NOTE: Sometimes ROI has multiple, for example if you had a photo with
        #       two people, that's two ROI with their own parts.
        outputs = [
            (self.predict(roi.crop), roi) for roi in rois
        ]  # output, roi

        # Flatten List
        outputs_flat = [
            (item, output[1]) for output in outputs for item in output[0]
        ]

        # Fix the Coords from the cropped to the original size
        for part, roi in outputs_flat:
            if part.bbox is None:
                continue

            roi_x1, roi_y1, _, _ = roi.local_bbox
            x1, y1, x2, y2 = part.bbox
            new_bbox = (
                roi_x1 + x1,
                roi_y1 + y1,
                roi_x1 + x2,
                roi_y1 + y2,
            )
            part.bbox = new_bbox

        # Return just the Parts
        return [output[0] for output in outputs_flat]

from abc import ABC, abstractmethod
from typing import Any

from censor_engine._typing import BBox, Image

from ._model_schemas import ROIOutput
from .output_schemas import DetectorOutput


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

    def convert_image_to_roi(self, box: BBox, image: Image) -> ROIOutput:
        return ROIOutput(original_image=image, local_bbox=box)

    @abstractmethod
    def initiate_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image) -> list[DetectorOutput]:
        raise NotImplementedError

    def predict_with_rois(self, rois: list[ROIOutput]) -> list[DetectorOutput]:
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

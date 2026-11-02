import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from censor_engine.typing import BBox, Image, MaskImage

from ._caching import CachingHandler
from ._enums import ModelType
from ._helpers import Plotter
from ._startup_helpers import GPUHandler
from .model_schemas import ROIOutput
from .output_schemas import DetectorOutput


class AIModel(ABC):
    # Model Stuff
    model_file: str
    model_caching_type: ModelType
    _model_classifiers: tuple[str, ...]

    # Internal
    _model: Any = None

    _device: int | str
    _model_full_path: str

    _model_path_base_folder: Path = Path("tools/models")

    # Cache Handling
    _cache_limit: int = 1000

    # Tools
    caching: CachingHandler
    gpu_handler: GPUHandler

    plotter: Plotter = Plotter()

    def __init__(self, model_classifiers: tuple[str, ...] | None = None):
        if model_classifiers is not None:
            self._model_classifiers = model_classifiers
        # Logging and Pre-load Stuff
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        # File Loading
        self._model_full_path = str(
            self._model_path_base_folder / self.model_file
        )

        # Caching
        self.caching = CachingHandler(
            self.model_caching_type,
            cache_limit=self._cache_limit,
        )

        # GPU Loading
        self.gpu_handler = GPUHandler(self.model_caching_type)
        self._device = self.gpu_handler.get_GPU()

    # Properties
    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    @property
    def model_classifiers(self) -> tuple[str, ...]:
        return tuple(
            self._handle_label(label) for label in self._model_classifiers
        )

    # Helper Methods
    def _handle_mask(self, mask: MaskImage, input_image: Image) -> MaskImage:
        # Smooth Mask's Edges & Close Holes
        kernel = np.ones((7, 7), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)  # type: ignore
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )  # type: ignore

    def _handle_label(self, name: str) -> str:
        return f"{self.model_name}_{name}".upper()

    def _download_model_from_hf(self, repo_id: str, file_name: str) -> None:
        self._model_full_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            local_dir=self._model_path_base_folder,
        )

    def convert_image_to_roi(self, box: BBox, image: Image) -> ROIOutput:
        return ROIOutput(original_image=image, local_bbox=box)

    def print_model_and_device(self) -> None:
        model = self.model_name
        path = self._model_full_path
        device = (
            torch.cuda.get_device_name(self._device)
            if isinstance(self._device, int)
            else "CPU"
        )
        print(f"{model}: {path} ({device})")

    # Abstract Methods
    @abstractmethod
    def initiate_model(self) -> None:
        # TODO: Why does this exist over __init__ again?
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
            x1, y1, x2, y2 = part.bbox.xyxy
            new_bbox = (
                roi_x1 + x1,
                roi_y1 + y1,
                roi_x1 + x2,
                roi_y1 + y2,
            )
            part.bbox = part.bbox.from_xyxy(new_bbox)

        # Return just the Parts
        return [output[0] for output in outputs_flat]

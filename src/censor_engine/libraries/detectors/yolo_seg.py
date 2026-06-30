import logging

import torch
from ultralytics import YOLO

from censor_engine._typing import Image
from censor_engine.libs.registries import AIModelRegistry
from censor_engine.models.lib_models.detectors.ai_models import (
    AIModel,
    DetectedPart,
)

logging.getLogger("ultralytics").setLevel(logging.ERROR)


@AIModelRegistry.register()
class YoloSeg(AIModel):
    # Public
    model_classifiers: tuple[str, ...] = ("person",)
    model_name: str = "YoloSeg"

    # Internal
    _model_path = "yolov8n-seg.pt"

    def initiate_model(self):
        # GPU Check
        self.device = 0 if torch.cuda.is_available() else "cpu"
        device_used = "GPU" if self.device != "cpu" else "CPU"

        # Load Model
        self._model = YOLO(self.model_path)
        print(f"YOLO-seg model: {self.model_path} ({device_used})")  # noqa: T201

        # Cache Fixer
        self._image_count = 0

    def _handle_cache(self):
        if self._image_count % self._cache_limit == 0:
            torch.cuda.empty_cache()
        else:
            self._image_count += 1

    def predict(self, image: Image) -> list[DetectedPart]:
        if self._model is None:
            msg = f"Model [{self.model_path}] is not initialised"
            raise TypeError(msg)

        # Get Results
        results = self._model(image)[0]

        # Quick Return
        res_boxes = results.boxes
        if res_boxes is None:
            return []

        # Process the Results
        boxes = res_boxes.xyxy.cpu().numpy()
        classes = res_boxes.cls.cpu().numpy()
        scores = res_boxes.conf.cpu().numpy()
        masks = results.masks.data.cpu().numpy()
        names = results.names

        # Build Output
        return [
            DetectedPart(
                origin=self.model_name,
                label=names[class_name],
                score=float(score),
                bbox=b,
                masks=mask,
            )
            for b, score, class_name, mask in zip(
                boxes,
                scores,
                classes,
                masks,
                strict=False,
            )
        ]

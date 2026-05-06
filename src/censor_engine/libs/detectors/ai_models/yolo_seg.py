import logging

import torch
from ultralytics import YOLO

from censor_engine.libs.registries import AIModelRegistry
from censor_engine.models.lib_models.detectors.ai_models import (
    AIModel,
    ModelOutput,
)
from censor_engine.typing import BBox, Image, Mask

logging.getLogger("ultralytics").setLevel(logging.ERROR)


class YoloSegOutput(ModelOutput):
    label: str | None = None
    score: float | None = None
    bbox: BBox | None = None
    masks: list[Mask] | None = None


@AIModelRegistry.register()
class YoloSeg(AIModel):
    _model_path = "yolov8n-seg.pt"

    def initiate_model(self):
        # GPU Check
        self.device = 0 if torch.cuda.is_available() else "cpu"
        device_used = "GPU" if self.device != "cpu" else "CPU"

        # Load Model
        self._model = YOLO(self._model_path)
        print(f"YOLO-seg model: {self._model_path} ({device_used})")  # noqa: T201

        # Cache Fixer
        self._image_count = 0

    def _handle_cache(self):
        if self._image_count % self._cache_limit == 0:
            torch.cuda.empty_cache()
        else:
            self._image_count += 1

    def predict(self, image: Image) -> list[YoloSegOutput]:
        if self._model is None:
            msg = f"Model [{self._model_path}] is not initialised"
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
            YoloSegOutput(
                label=names[class_name], score=float(score), bbox=b, masks=mask
            )
            for b, score, class_name, mask in zip(
                boxes, scores, classes, masks, strict=False
            )
        ]

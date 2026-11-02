import cv2
import numpy as np
from ultralytics import YOLO

from censor_engine.libraries.registries import AIModelRegistry
from censor_engine.models.libraries.ai_models._enums import OutputType
from censor_engine.models.libraries.ai_models.ai_models import (
    AIModel,
    ModelType,
)
from censor_engine.models.libraries.ai_models.output_schemas import (
    AbsoluteBBox,
    DetectorOutput,
)
from censor_engine.typing import Image


@AIModelRegistry.register()
class HumanParse(AIModel):
    _model_classifiers: tuple[str, ...] = (
        "hair",
        "face",
        # "neck",
        "arm",
        # "hand",
        "back",
        "leg",
        "foot",
        "outfit",
        "person",
        # "phone",
    )
    model_caching_type: ModelType = ModelType.TORCH
    model_file = "yolo-human-parse-epoch-125.pt"

    def initiate_model(self) -> None:
        # Load Model
        self._download_model_from_hf(
            repo_id="MnLgt/yolo-human-parse",
            file_name=self.model_file,
        )
        self._model = YOLO(self._model_full_path)
        self.print_model_and_device()

    def predict(self, image: Image) -> list[DetectorOutput]:
        if self._model is None:
            msg = f"Model [{self.model_file}] is not initialised"
            raise TypeError(msg)

        # Get Results
        results = self._model.predict(
            source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            device=self._device,
            retina_masks=True,
            conf=0.001,
            iou=0.7,
        )[0]

        # Quick Return
        res_boxes = results.boxes
        if res_boxes is None:
            return []

        # Process the Results
        boxes = np.round(res_boxes.xyxy.cpu().numpy()).astype(int)
        classes = res_boxes.cls.cpu().numpy()
        scores = res_boxes.conf.cpu().numpy()
        names = results.names

        # # Masks Require Rescaling
        masks = list(results.masks.data.cpu().numpy() * 255)

        # Build Output
        return [
            DetectorOutput(
                origin=self.model_name,
                label=self._handle_label(names[class_name]),
                score=float(score),
                bbox=AbsoluteBBox.from_xyxy(bbox),
                mask=self._handle_mask(mask, image),
                output_type=OutputType.PART,
            )
            for bbox, score, class_name, mask in zip(
                boxes,
                scores,
                classes,
                masks,
                strict=False,
            )
            if names[class_name] in self._model_classifiers
        ]

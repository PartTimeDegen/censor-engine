import numpy as np
from ultralytics import YOLO

from censor_engine.libraries.registries import AIModelRegistry
from censor_engine.models.libraries.ai_models.ai_models import (
    AIModel,
    ModelType,
)
from censor_engine.models.libraries.ai_models.output_schemas import (
    AbsoluteBBox,
    DetectorOutput,
)
from censor_engine.typing import Image


# NOTE: Cool test, need to make new model for classification
#       would be good for the raceplay people
# @AIModelRegistry.register()
class FairFace(AIModel):
    # Public
    _model_classifiers: tuple[str, ...] = (
        "Black",
        "East Asian",
        "Indian",
        "Latino_Hispanic",
        "Middle Eastern",
        "Southeast Asian",
        "White",
    )
    model_caching_type: ModelType = ModelType.TORCH
    model_file: str = "Race-CLS-FairFace_yolo11x.pt"

    def initiate_model(self):
        # Load Model
        self._download_model_from_hf(
            repo_id="Anzhc/Race-Classification-FairFace-YOLOv8",
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
            source=image,
            device=self._device,
            retina_masks=True,
        )[0]
        class_id = results.probs.top1

        # Get the class name
        return results.names[class_id]

        # Quick Return
        res_boxes = results.boxes
        if res_boxes is None:
            return []

        # Process the Results
        boxes = np.round(res_boxes.xyxy.cpu().numpy()).astype(int)
        classes = res_boxes.cls.cpu().numpy()
        scores = res_boxes.conf.cpu().numpy()
        names = results.names

        # Build Output
        return [
            DetectorOutput(
                origin=self.model_name,
                label=self._handle_label(names[class_name]),
                score=float(score),
                bbox=AbsoluteBBox.from_xyxy(bbox),
            )
            for bbox, score, class_name in zip(
                boxes,
                scores,
                classes,
                strict=False,
            )
            if names[class_name] in self._model_classifiers
        ]

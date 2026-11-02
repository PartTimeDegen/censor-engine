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
class Nevermore(AIModel):
    # Public
    _model_classifiers: tuple[str, ...] = (
        "person",
        "face",
        "female_breast_exposed",
        "female_breast_covered",
        "male_breast_exposed",
        "male_breast_covered",
        "nipple_exposed",
        "nipple_covered",
        "penis_exposed",
        "penis_covered",
        "vagina_exposed",
        "vagina_covered",
        "buttocks_exposed",
        "buttocks_covered",
        "foot_exposed",
        "foot_covered",
    )
    model_caching_type: ModelType = ModelType.TORCH
    model_file: str = "det_v11_big/det_v11_big_640.onnx"

    def initiate_model(self) -> None:
        # Load Model
        self._download_model_from_hf(
            repo_id="kinkyrvn/nevermore",
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
                output_type=OutputType.PART,
            )
            for bbox, score, class_name in zip(
                boxes,
                scores,
                classes,
                strict=False,
            )
            if names[class_name] in self._model_classifiers
        ]

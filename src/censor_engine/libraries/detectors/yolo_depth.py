from ultralytics import YOLO

from censor_engine.libraries.registries import AIModelRegistry
from censor_engine.models.libraries.ai_models._enums import OutputType
from censor_engine.models.libraries.ai_models.ai_models import (
    AIModel,
    ModelType,
)
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from censor_engine.typing import Image


@AIModelRegistry.register()
class YoloDepth(AIModel):
    # Public
    _model_classifiers: tuple[str, ...] = ()
    model_caching_type: ModelType = ModelType.TORCH
    model_file: str = "yolo26x-depth.pt"

    def initiate_model(self):
        # Load Model
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

        # Process the Results
        depth = results[0].depth.data.cpu().numpy()

        # Build Output
        return [
            DetectorOutput(
                origin=self.model_name,
                label=self._handle_label("depth"),
                depth_map=depth,
                output_type=OutputType.DEPTH,
            )
        ]

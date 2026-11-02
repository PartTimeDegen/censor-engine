from censor_engine.models.core.processors.image.model_manager import (
    ModelManager,
)
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from censor_engine.typing import Image

from ._advanced_masks import InferredDetectionParts


class DetectionManager:
    def run_detection_pipeline(
        self,
        input_image: Image,
        model_manager: ModelManager,
    ) -> list[DetectorOutput]:
        # TODO: Add Caching Here

        # Predict the Image with the Models
        predictions: list[DetectorOutput] = [
            output
            for model in model_manager.running_models
            for output in model.predict(input_image)
        ]

        # Infer addition stuff
        inferred = InferredDetectionParts()

        # # Skin
        if model_manager.extra_settings.skin_segmentation:
            predictions += inferred.generate_skin_part(predictions)

        # Cleanup Disabled Features
        return [
            pred
            for pred in predictions
            if pred.origin not in model_manager.subtraction_models
        ]

import numpy as np
from fashn_human_parser import FashnHumanParser

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

# NOTE: do not use this, it's shit, keep for reference


# @AIModelRegistry.register()
class Fashn(AIModel):
    _model_classifiers: tuple[str, ...] = (
        # "background",
        "face",
        # "hair",
        "arms",
        "hands",
        "legs",
        "feet",
        "torso",
        # "top",
        # "dress",
        # "skirt",
        # "pants",
        # "belt",
        # "scarf",
        # "bag",
        # "hat",
        # "glasses",
        # "jewelry",
    )
    model_caching_type: ModelType = ModelType.TORCH
    model_file = "model.safetensors"

    def initiate_model(self) -> None:
        self._model = FashnHumanParser()
        self.print_model_and_device()

    def predict(self, image: Image) -> list[DetectorOutput]:
        internal_ids = (
            "background",
            "face",
            "hair",
            "top",
            "dress",
            "skirt",
            "pants",
            "belt",
            "bag",
            "hat",
            "scarf",
            "glasses",
            "arms",
            "hands",
            "legs",
            "feet",
            "torso",
            "jewelry",
        )

        if self._model is None:
            msg = f"Model [{self.model_file}] is not initialised"
            raise TypeError(msg)

        results = self._model.predict(image)

        # Build Output
        return [
            DetectorOutput(
                origin=self.model_name,
                label=self._handle_label(label),
                mask=(results == index).astype(np.uint8) * 255,
                output_type=OutputType.PART,
            )
            for index, label in enumerate(internal_ids)
            if label in self._model_classifiers
        ]

from pathlib import Path
from urllib.request import urlretrieve

import mediapipe as mp  # type: ignore
import numpy as np

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

_model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_multiclass_256x256/"
    "float32/latest/selfie_multiclass_256x256.tflite"
)


def _download_model(model_path: Path) -> None:
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = model_path.with_suffix(".tmp")

    try:
        urlretrieve(_model_url, temp_path)
        temp_path.replace(model_path)
    finally:
        temp_path.unlink(missing_ok=True)


_classes = {
    0: "background",
    1: "hair",
    2: "body_skin",
    3: "face_skin",
    4: "clothes",
    5: "accessories",
}


# NOTE: This should work, however my computer is so shit it doesn't have AVX
# @AIModelRegistry.register()
class MediaPipeSeg(AIModel):
    _model_classifiers: tuple[str, ...] = tuple(_classes.values())
    model_caching_type: ModelType = ModelType.MEDIAPIPE
    model_file: str = "selfie_multiclass_256x256.tflite"

    def initiate_model(self) -> None:
        model_path = Path(f"tools/models/{self.model_file}")
        _download_model(model_path)

        base_options = mp.tasks.BaseOptions
        image_segmenter_options = mp.tasks.vision.ImageSegmenterOptions

        # NOTE: This is only for MediaPipe
        model_options = image_segmenter_options(
            base_options=base_options(model_asset_path=str(model_path)),
            output_category_mask=True,
            output_confidence_masks=True,
        )

        # Load Model
        self._model = mp.tasks.vision.ImageSegmenter.create_from_options(
            model_options
        )

        self.print_model_and_device()

    def predict(self, image: Image) -> list[DetectorOutput]:
        if self._model is None:
            msg = f"Model [{self.model_file}] is not initialised"
            raise TypeError(msg)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image,
        )

        result = self._model.segment(mp_image)

        category_mask = result.category_mask.numpy_view()

        confidence_masks = [
            mask.numpy_view() for mask in result.confidence_masks
        ]

        outputs = []
        for label_id, label in _classes.items():
            mask = category_mask == label_id

            if not np.any(mask):
                continue

            score = float(np.mean(confidence_masks[label_id][mask]))

            outputs.append(
                DetectorOutput(
                    origin=self.model_name,
                    label=self._handle_label(label),
                    score=score,
                    mask=mask.astype(np.uint8),
                    output_type=OutputType.PART,
                )
            )

        return outputs

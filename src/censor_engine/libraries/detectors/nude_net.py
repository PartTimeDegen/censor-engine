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
class NudeNet(AIModel):
    """
    This detector is the core of CensorEngine, it uses the NudeNet model.

    The model collects the main components of engine (see below in
    model_classifiers).

    The reason this model is local is interesting, essentially there is a
    NudeNet python package that auto-downloads the model however it's an onnx
    file which has several problems:
        1)  Hard to run in general.
        2)  Doesn't make GPU acceleration easy to do since you need to run your
            own CUDA files (PyTorch does this itself).

    The other issue is that the package doesn't actually allow for CUDA
    rendering, it has the argument in the class/method however it's dangling,
    because of that I had to download the file manually if I wanted to access
    it for CUDA. During this I found that he released a PyTorch version which
    is magnitudes easier to use especially for GPU.

    Because GitHub didn't allow me to automate downloading it from releases,
    I just kept it local in this repo, especially since the size is small.

    """

    _model_classifiers: tuple[str, ...] = (
        "FACE_FEMALE",
        "ARMPITS_EXPOSED",
        "ARMPITS_COVERED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "BELLY_COVERED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "ANUS_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "FEET_EXPOSED",
        "FEET_COVERED",
        "FACE_MALE",
        "MALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
    )
    model_caching_type: ModelType = ModelType.TORCH
    model_file = "640m"

    def use_smaller_model(self):
        self.model_file = "320n"

    def use_bigger_model(self):
        self.model_file = "640m"

    def initiate_model(self) -> None:
        # Load Model
        self._download_model_from_hf(
            repo_id="vladmandic/nudenet",
            file_name=f"nudenet-v34-{self.model_file}.pt",
        )
        self._model = YOLO(self._model_full_path)
        self.print_model_and_device()

    def predict(self, image: Image) -> list[DetectorOutput]:
        if self._model is None:
            msg = f"Model [{self.model_file}] is not initialised"
            raise TypeError(msg)

        results = self._model.predict(source=image, device=self._device)[0]
        boxes = results.boxes

        # Save Resources if Empty
        if boxes is None or len(boxes) == 0:
            return []

        # Move Data to CPU
        xyxy = np.rint(boxes.xyxy.cpu().numpy()).astype(np.int32)
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        names = results.names

        # Build Output
        return [
            DetectorOutput(
                origin=self.model_name,
                label=self._handle_label(names[label]),
                score=float(score),
                bbox=AbsoluteBBox.from_xyxy(bbox),
                output_type=OutputType.PART,
            )
            for bbox, score, label in zip(xyxy, conf, cls, strict=False)
        ]

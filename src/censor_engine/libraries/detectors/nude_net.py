import logging

import numpy as np
import torch
from ultralytics import YOLO

from censor_engine.libs.registries import AIModelRegistry
from censor_engine.models.lib_models.detectors.ai_models import (
    AIModel,
    ROIOutput,
)
from censor_engine.models.lib_models.detectors.schemas import (
    DetectedPart,
)
from censor_engine.typing import Image

logging.getLogger("ultralytics").setLevel(logging.ERROR)


@AIModelRegistry.register()
class NudeNetModel(AIModel):
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

    model_name: str = "NudeNet"
    model_classifiers: tuple[str, ...] = (
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

    def initiate_model(self) -> None:
        # Determine Model (Left in for future)
        use_bigger_model = False
        used_model = "640m.pt" if use_bigger_model else "320n.pt"

        # GPU Check
        self._device = 0 if torch.cuda.is_available() else "cpu"
        device_used = "GPU" if self._device != "cpu" else "CPU"

        # Load Model
        self._model = YOLO(f"tools/models/{used_model}")
        print(f"NudeNet model: {used_model} ({device_used})")  # noqa: T201

        # Cache Fixer
        self._image_count = 0

    def _handle_cache(self) -> None:
        if self._image_count % self._cache_limit == 0:
            torch.cuda.empty_cache()
        else:
            self._image_count += 1

    def predict(
        self,
        image: Image,
    ) -> list[DetectedPart]:
        """
        This is the prediction function for NudeNet, this is the core of
        CensorEngine.

        :param Image image: Image to check

        :raises TypeError: Missing Model

        :return list[DetectedPartSchema]: Formatted list of outputs
        """
        if self._model is None:
            msg = "Model [NudeNet] is not initialised"
            raise TypeError(msg)

        with torch.no_grad():
            results = self._model(
                image,
                device=self._device,
                verbose=False,
            )[0]
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
            DetectedPart(
                origin=self.model_name,
                label=names[c],
                score=float(s),
                bbox=b,
            )
            for b, s, c in zip(xyxy, conf, cls, strict=False)
        ]

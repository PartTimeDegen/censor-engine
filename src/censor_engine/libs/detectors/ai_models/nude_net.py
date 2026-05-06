import logging

import numpy as np
import torch
from ultralytics import YOLO

from censor_engine.libs.registries import AIModelRegistry
from censor_engine.models.lib_models.detectors.ai_models import (
    AIModel,
    ModelOutput,
    ROIOutput,
)
from censor_engine.typing import BBox, Image

logging.getLogger("ultralytics").setLevel(logging.ERROR)


class NudeNetOutput(ModelOutput):
    label: str | None = None
    score: float | None = None
    bbox: BBox | None = None


@AIModelRegistry.register()
class NudeNetModel(AIModel):
    """
    This detector is the core of CensorEngine, it uses the NudeNet model.

    TODO: Explain what it gets
    TODO: Explain why it's local


    """

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
        rois: list[ROIOutput] | None = None,
    ) -> list[NudeNetOutput]:
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
            NudeNetOutput(label=names[c], score=float(s), bbox=b)
            for b, s, c in zip(xyxy, conf, cls, strict=False)
        ]

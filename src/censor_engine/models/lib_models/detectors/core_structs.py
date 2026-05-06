from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from censor_engine.models.lib_models.detectors.ai_models import AIModel
from censor_engine.typing import Image


class DetectedPartSchema(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            np.ndarray: lambda v: v.tolist(),
        },
    }

    # Internal
    part_id: int = 0

    # Meta
    label: str | None = None
    score: float | None = None

    # Data Used for Information
    bbox: np.ndarray | None = None  # XYXY
    masks: list[np.ndarray] | None = None

    def set_part_id(self, number: int) -> None:
        self.part_id = number


class Detector(ABC):
    """
    This is the model used for detectors, it's pretty simple, just maintains a
    valid method to use/overwrite, and some meta information.

    The meta information isn't useful for code (custom models may vary),
    however for a documentation POV, it's useful to know the internal name
    (maybe logging) and what it's producing (Trust me, it's better than having
    to find NudeNet's repo to find the classifiers).

    :raises NotImplementedError: This is just to throw an error to ensure the
    model devs know they didn't properly implement the method under the
    correct name
    """

    model_name: str
    model_classifiers: tuple[str, ...]
    model_requirements: list | None  # Detectors

    model_object: AIModel
    model_loaded: bool = False

    @abstractmethod
    def detect_image(
        self,
        image: Image,
    ) -> list:
        raise NotImplementedError

    # @abstractmethod
    # def detect_batch(
    #     self,
    #     images: list[Image],
    #     batch_size: int,
    # ) -> dict[int, list]:
    #     raise NotImplementedError

    def turn_on_model(self):
        if not self.model_loaded:
            self.model_object.initiate_model()
            self.model_loaded = True

import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from censor_engine.models.lib_models.detectors import (
    DetectedPartSchema,
    Detector,
)
from censor_engine.typing import Image

logging.getLogger("ultralytics").setLevel(logging.ERROR)


class NudeNetModel:
    model: YOLO
    device: int | str
    image_count: int

    cache_limit: int = 1000

    def __init__(self, *, use_bigger_model: bool = False):
        # Model Check
        model_exists = Path("tools/models/640m.pt").exists()
        used_model = (
            "640m.pt" if use_bigger_model and model_exists else "320n.pt"
        )
        self.model = YOLO(f"tools/models/{used_model}")
        print(f"NudeNet model: {used_model}")  # noqa: T201

        # GPU Check
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"NudeNet using GPU?: {self.device != 'cpu'}")  # noqa: T201

        # Cache Fixer
        self.image_count = 0

    def detect(self, image_path: str):
        if self.image_count % self.cache_limit == 0:
            torch.cuda.empty_cache()
        else:
            self.image_count += 1

        with torch.no_grad():
            results = self.model(
                image_path, device=self.device, verbose=False
            )[0]
        boxes = results.boxes

        # Save Resources if Empty
        if boxes is None or len(boxes) == 0:
            return []

        # Move Data to CPU
        xyxy = boxes.xyxy.cpu()
        conf = boxes.conf.cpu()
        cls = boxes.cls.cpu()

        # Convert to Numpy
        xyxy = xyxy.numpy()
        conf = conf.numpy()
        cls = cls.numpy().astype(int)

        names = results.names

        # Build Output
        return [
            {
                "class": names[c],
                "score": float(s),
                "box": [
                    int(x1),
                    int(y1),
                    int(x2 - x1),  # width
                    int(y2 - y1),  # height
                ],
            }
            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls, strict=False)
        ]


class NudeNetDetector(Detector):
    """
    This Detector is the code of CensorEngine, it is the NudeNet model.

    This handles the core labels of the engine.

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
    model_object = NudeNetModel()

    def detect_image(
        self,
        file_images_or_path: str,
    ) -> list[DetectedPartSchema]:
        return [
            DetectedPartSchema(
                label=found_part["class"],
                score=found_part["score"],
                relative_box=found_part["box"],
            )
            for _, found_part in enumerate(
                self.model_object.detect(file_images_or_path),
            )
        ]

    def detect_batch(
        self,
        file_images_or_paths: list[str] | list[Image],
        batch_size: int,
    ) -> dict[int, list[DetectedPartSchema]]:
        raise NotImplementedError("This needs work")  # noqa: EM101
        output = self.model_object.detect_batch(
            file_images_or_paths,
            batch_size,
        )

        dict_output = {}
        for index, image in enumerate(output):
            dict_output[index] = [
                DetectedPartSchema(
                    part_id=index,
                    label=found_part["class"],
                    score=found_part["score"],
                    relative_box=found_part["box"],
                )
                for index, found_part in enumerate(image)
            ]

        return dict_output

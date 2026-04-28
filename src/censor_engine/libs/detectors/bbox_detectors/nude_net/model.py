import logging
from pathlib import Path

import torch
from ultralytics import YOLO

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

import logging
from typing import Any

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from censor_engine._typing import BBox, Image, MaskImage
from censor_engine.libraries.registries import AIModelRegistry
from censor_engine.models.lib_models.detectors.ai_models import (
    AIModel,
)
from censor_engine.models.lib_models.detectors.schemas import (
    DetectedPart,
)

logging.getLogger("ultralytics").setLevel(logging.ERROR)


@AIModelRegistry.register()
class SAMTwo(AIModel):
    _model_path = "facebook/sam2-hiera-base-plus"
    _THRESHOLD = 0.45

    model_classifiers: tuple[str, ...] = ("BODY",)
    model_name: str = "SAMTwo"

    def initiate_model(self):
        # GPU Check
        self.device = 0 if torch.cuda.is_available() else "cpu"
        device_used = "GPU" if self.device != "cpu" else "CPU"

        # Load Model
        self.sam_processor = Sam2Processor.from_pretrained(self._model_path)
        self.sam_model = Sam2Model.from_pretrained(self._model_path)
        print(f"SAM2 model: {self._model_path} ({device_used})")  # noqa: T201

        # GPU Stuff
        self.sam_model = self.sam_model.to(self.device)  # type: ignore
        self.sam_model.eval()
        if self.device == "cuda":
            self.sam_model = self.sam_model.to(torch.float16)  # type: ignore

        # Cache Fixer
        self._image_count = 0

    # ------------------ #
    # Prediction Helpers #
    # ------------------ #
    def __get_centroid(self, image: Image) -> tuple[float, float]:
        h, w = image.shape[:2]
        return (w / 2, h / 2)

    def __setup_inputs(
        self,
        image: Image,
        point: tuple[float, float],
    ):
        sam_inputs = self.sam_processor(
            images=image,
            input_points=[[[list(point)]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        )

        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in sam_inputs.items()
        }

    def __get_safe_float(
        self,
        x: MaskImage | float | list[float],
    ) -> np.ndarray | float:
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)  # type: ignore
        if isinstance(x, np.ndarray):
            return float(x.item()) if x.size == 1 else float(x.flat[0])
        return float(x)  # type: ignore

    def __get_normalised_mask(self, output: Any, inputs: dict) -> MaskImage:  # noqa: ANN401
        # Convert Outputs to Masks
        masks = self.sam_processor.post_process_masks(
            output.pred_masks,
            inputs["original_sizes"],
        )[0]

        # Convert Values to Numpy and Move to CPU
        masks = masks.detach().cpu().numpy()

        if masks.ndim == 4:  # noqa: PLR2004 # TODO: Sort this out
            masks = masks[0]

        return np.asarray(masks)

    def __handle_mask_logic(
        self,
        masks: MaskImage,
        scores: list[float],
        image: Image,
    ) -> tuple[MaskImage, float]:
        # Handle Empty Masks
        if masks.shape[0] == 0:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0

        # Handle Single Masks
        if masks.shape[0] == 1:
            score = (
                self.__get_safe_float(scores)
                if np.ndim(scores) == 0
                else self.__get_safe_float(scores[0])
            )
            return masks[0].astype(bool), score  # type: ignore

        # Multiple Masks
        if len(scores) != masks.shape[0]:
            score = (
                self.__get_safe_float(scores[0]) if len(scores) > 0 else 0.5
            )
            return masks[0].astype(bool), score  # type: ignore

        # Get Best Mask
        best = int(np.argmax(scores))
        best = min(best, masks.shape[0] - 1)

        return (masks[best].astype(bool), self.__get_safe_float(scores[best]))  # type: ignore

    def __get_mask(
        self,
        image: Image,
        point: tuple[float, float],
    ) -> tuple[MaskImage, float]:

        inputs = self.__setup_inputs(image=image, point=point)

        with torch.no_grad():
            model_outputs = self.sam_model(**inputs)

        masks = self.__get_normalised_mask(model_outputs, inputs)

        scores = model_outputs.iou_scores[0].detach().cpu().numpy()

        return self.__handle_mask_logic(masks, scores, image)

    def __compute_mask_confidence(
        self,
        full_mask: MaskImage,
        bbox: BBox,
        score: float,
    ) -> float:
        if full_mask.sum() == 0:
            return 0.0

        x1, y1, x2, y2 = bbox

        # IoU-like overlap
        ys, xs = np.where(full_mask)

        mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()

        ix1 = max(x1, mx1)
        iy1 = max(y1, my1)
        ix2 = min(x2, mx2)
        iy2 = min(y2, my2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2 - x1) * (y2 - y1) + 1e-6

        box_overlap = inter / union

        # density (foreground vs background sanity)
        density = full_mask.mean()

        return 0.5 * score + 0.3 * box_overlap + 0.2 * density

    def predict(
        self,
        image: Image,
    ) -> list[DetectedPart]:

        if self.sam_model is None:
            msg = f"Model [{self._model_path}] is not initialised"
            raise TypeError(msg)

        # Compute centroid
        point = self.__get_centroid(image)

        # Generate mask
        mask, score = self.__get_mask(image, point)

        if mask.sum() == 0:
            return []

        return [
            DetectedPart(
                origin=self.model_name,
                masks=[mask.astype(bool)],
                score=score,
            )
        ]

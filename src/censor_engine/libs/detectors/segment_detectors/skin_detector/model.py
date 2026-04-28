from dataclasses import dataclass

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor
from ultralytics import YOLO

from censor_engine.models.enums import MaskType
from censor_engine.typing import BBox, Image


@dataclass(slots=True)
class ROIOutput:
    crop: Image
    local_bbox: BBox


class BodyDetectorModel:
    """
    This detector model is used to find the body of a person for censoring.

    This model is quite advanced and uses two AI models to generate the output.

    The AI models used are YOLO (for person bbox) and SAM2 (for body Masking).
    The reason YOLO is utilised first is to provide the ROI of the person
    because SAM2 only works via "points" selected (SAM3 uses words, waiting for
    that to release however), so in order to get a good mask, it's better to
    reduce the area.

    Methodology:
        1)  Initialise models and push to GPU if possible.
        2)  Detect the bbox for the person.
        3)
        4)  Generate the SAM2 mask.
        5)  Clean the mask.

    Notes:
        a)  I'm aware sometimes the mask doesn't select the right thing,
            creating the exact opposite of what it's masking, I'm working on
            it.

    Dev Notes:
        -   SAM2 separates body and hair normally, it's annoying.
        -   SAM2 also doesn't label what it's masked.
        -   Given that a bounding box is normally the majority the thing it's
            looking for (and people tend to not have holes in them), the mask
            can be improved with some assumptions:
                1)  The centroid of the bbox will contain the person.
                2)  The majority of the area will also be of the person.

    Thoughts:
        -   This model is very useful for other ideas:
            -   Using a depth model and nude net, parts can be split into
                layers which can be made to make a layered censor.
            -   If I separate out the YOLO model at some point and make it as
                a dependency to inject, I can implement it into new models,
                for example "shoes" or "clothes".

    Todo:
        -   Upgrade to YOLO26n-seg
        -   Isolate YOLO model
        -   Check if the YOLO model can be good enough
        -   Add Negative points

    """

    yolo_model: YOLO

    sam_processor: Sam2Processor
    sam_model: Sam2Model

    device: torch.device

    # ========================== #
    # Stage 1: Initialise Models #
    # ========================== #
    def __init__(self):
        # Controls
        yolo_id = "yolov8n-seg.pt"
        sam_id = "facebook/sam2-hiera-base-plus"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # YOLO Stuff
        self.yolo_model = YOLO(yolo_id)

        # Sam Stuff
        self.sam_processor = Sam2Processor.from_pretrained(sam_id)
        self.sam_model = Sam2Model.from_pretrained(sam_id).to(self.device)  # type: ignore

        self.sam_model.eval()
        if self.device == "cuda":
            self.sam_model = self.sam_model.to(torch.float16)  # type: ignore

    # ============================= #
    # Stage 2: Get YOLO Person Bbox #
    # ============================= #
    def _get_yolo_person_box(
        self,
        image: Image,
    ) -> list[BBox]:
        """
        This is used to get the YOLO output for the person. The reason this is
        done is to reduce the ROI for SAM2 to just the person, to properly
        select the mask points.

        :param Image image: Input Image
        :return list[tuple[int, int, int, int]]: Output of YOLO, the bbox
        """
        # Get Results
        results = self.yolo_model(image)[0]

        # Quick Return
        if results.boxes is None:
            return []

        # Process the Results
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        # Filter Results Down to "Person"
        return [
            box
            for box, c in zip(boxes, classes, strict=False)
            if self.yolo_model.names[int(c)] == "person"
        ]

    # ======================= #
    # Stage 3: Process Bboxes #
    # ======================= #
    def _convert_image_to_roi(
        self, box: BBox, image: Image
    ) -> ROIOutput | None:

        x1, y1, x2, y2 = box.astype(int)
        h, w = image.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return ROIOutput(
            crop=image[y1:y2, x1:x2],
            local_bbox=np.array([0, 0, x2 - x1, y2 - y1], dtype=np.float32),
        )

    # ====================== #
    # Stage 4: Get SAM2 Mask #
    # ====================== #
    def _get_sam_mask(self, roi: ROIOutput):
        # Helpers
        box = roi.local_bbox
        x1, y1, x2, y2 = box

        # Notable Points
        centre_point = [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)]

        # Inputs
        sam_inputs = self.sam_processor(
            images=roi.crop,
            input_boxes=[[box.tolist()]],
            input_points=[[[centre_point]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        )

        # Move Inputs to GPU
        sam_inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in sam_inputs.items()
        }

        # Run Inference
        with torch.no_grad():
            model_outputs = self.sam_model(**sam_inputs)

        # Convert Outputs to Masks
        masks = self.sam_processor.post_process_masks(
            model_outputs.pred_masks,
            sam_inputs["original_sizes"],
        )[0]

        # Convert Values to Numpy and Move to CPU
        masks = masks.detach().cpu().numpy()
        scores = model_outputs.iou_scores[0].detach().cpu().numpy()

    # ======== #
    # Pipeline #
    # ======== #
    def _multiprocess_box_iteration(
        self,
        box: BBox,
        image: Image,
    ) -> BBox | None:
        roi = self._convert_image_to_roi(box, image)
        if roi is None:
            return None
        return roi.local_bbox

    def predict(self, image: Image) -> MaskType:  # type: ignore
        bboxes = self._get_yolo_person_box(image=image)

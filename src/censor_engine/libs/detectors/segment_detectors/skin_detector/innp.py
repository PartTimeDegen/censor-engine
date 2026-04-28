from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor
from ultralytics import YOLO


@dataclass
class YoloSAM2Segmenter:
    name: str = "yolo_sam2_confidence_clean"

    yolo_path: str = "yolov8n-seg.pt"
    sam_model_id: str = "facebook/sam2-hiera-base-plus"
    device: str = "cuda"

    yolo: YOLO = field(init=False)
    sam_model: Sam2Model = field(init=False)
    sam_processor: Sam2Processor = field(init=False)

    _image = None
    _boxes = None
    _masks = None
    _confidences = None

    # -------------------------
    # INIT
    # -------------------------
    def __post_init__(self):
        self.yolo = YOLO(self.yolo_path)

        self.sam_processor = Sam2Processor.from_pretrained(self.sam_model_id)
        self.sam_model = Sam2Model.from_pretrained(self.sam_model_id).to(
            self.device
        )
        self.sam_model.eval()

        if self.device == "cuda":
            self.sam_model = self.sam_model.to(torch.float16)

    # -------------------------
    # YOLO PERSON BOXES
    # -------------------------
    def get_boxes(self, image):
        res = self.yolo(image)[0]

        if res.boxes is None:
            return []

        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()

        return [
            box
            for box, c in zip(boxes, classes)
            if self.yolo.names[int(c)] == "person"
        ]

    # -------------------------
    # MASK CLEANING PIPELINE
    # -------------------------
    def clean_mask(self, mask):
        mask = mask.astype(np.uint8)

        # morphology cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # largest component only
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest).astype(np.uint8)

        # smoothing
        mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        mask = mask > 0.5

        return mask

    # -------------------------
    # CONFIDENCE SCORE
    # -------------------------
    def compute_confidence(self, mask, box, sam_score):
        if mask.sum() == 0:
            return 0.0

        x1, y1, x2, y2 = box

        # IoU-like overlap
        ys, xs = np.where(mask)

        mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()

        ix1 = max(x1, mx1)
        iy1 = max(y1, my1)
        ix2 = min(x2, mx2)
        iy2 = min(y2, my2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2 - x1) * (y2 - y1) + 1e-6

        box_overlap = inter / union

        # density (foreground vs background sanity)
        density = mask.mean()

        return 0.5 * sam_score + 0.3 * box_overlap + 0.2 * density

    # -------------------------
    # SAM MASK (POINT BIAS FIXED)
    # -------------------------
    def sam_mask(self, image_rgb, box):
        x1, y1, x2, y2 = box

        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        inputs = self.sam_processor(
            images=image_rgb,
            input_boxes=[[box.tolist()]],
            input_points=[[[[cx, cy]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
        )[0]

        masks = masks.detach().cpu().numpy()
        scores = outputs.iou_scores[0].detach().cpu().numpy()

        # ----------------------------
        # SAFE NORMALIZATION
        # ----------------------------
        if masks.ndim == 4:
            masks = masks[0]

        masks = np.asarray(masks)

        # handle empty output
        if masks.shape[0] == 0:
            h, w = image_rgb.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0

        # ----------------------------
        # FIX SCORE TYPE (IMPORTANT)
        # ----------------------------
        def safe_float(x):
            if isinstance(x, np.ndarray):
                return float(x.item()) if x.size == 1 else float(x.flat[0])
            return float(x)

        # ----------------------------
        # SINGLE MASK CASE
        # ----------------------------
        if masks.shape[0] == 1:
            score = (
                safe_float(scores)
                if np.ndim(scores) == 0
                else safe_float(scores[0])
            )
            return masks[0].astype(bool), score

        # ----------------------------
        # MULTI MASK CASE (SAFE)
        # ----------------------------
        if len(scores) != masks.shape[0]:
            score = safe_float(scores[0]) if len(scores) > 0 else 0.5
            return masks[0].astype(bool), score

        best = int(np.argmax(scores))
        best = min(best, masks.shape[0] - 1)

        return masks[best].astype(bool), safe_float(scores[best])

    # -------------------------
    # PIPELINE RUN
    # -------------------------
    def _run(self):
        image = self._image
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = self.get_boxes(image)

        masks = []
        confidences = []

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_rgb[y1:y2, x1:x2]

            local_box = np.array([0, 0, x2 - x1, y2 - y1], dtype=np.float32)

            mask_crop, sam_score = self.sam_mask(crop, local_box)

            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[y1:y2, x1:x2] = mask_crop

            full_mask = self.clean_mask(full_mask)

            conf = self.compute_confidence(full_mask, box, sam_score)

            # filter bad masks
            if conf < 0.45:
                continue

            masks.append(full_mask)
            confidences.append(conf)

        self._boxes = boxes
        self._masks = masks
        self._confidences = confidences

    # -------------------------
    # MAIN OUTPUT
    # -------------------------
    def predict(self, image, alpha=0.5):
        self._image = image
        self._run()

        overlay = np.zeros_like(image)

        for m in self._masks:
            overlay[m] = (0, 255, 0)

        out = image.copy()
        idx = overlay.any(axis=2)

        out[idx] = ((1 - alpha) * out[idx] + alpha * overlay[idx]).astype(
            np.uint8
        )

        return out

    # -------------------------
    # BINARY MASK
    # -------------------------
    def get_binary_mask(self):
        if self._masks is None:
            raise RuntimeError("Run predict() first")

        h, w = self._image.shape[:2]
        out = np.zeros((h, w), dtype=bool)

        for m in self._masks:
            out |= m

        return out

    # -------------------------
    # BREAKDOWN VISUAL
    # -------------------------
    def get_mask_breakdown(self, alpha=0.5):
        if self._masks is None:
            raise RuntimeError("Run predict() first")

        base = self._image.copy()
        overlay = np.zeros_like(base)

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ]

        for i, m in enumerate(self._masks):
            overlay[m] = colors[i % len(colors)]

        idx = overlay.any(axis=2)
        base[idx] = ((1 - alpha) * base[idx] + alpha * overlay[idx]).astype(
            np.uint8
        )

        return base

    # -------------------------
    # YOLO DEBUG OVERLAY
    # -------------------------
    def draw_yolo_overlay(self, image):
        if self._image is None:
            raise RuntimeError("Run predict() first")

        img = image
        overlay = img.copy()

        for box in self._boxes:
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return overlay

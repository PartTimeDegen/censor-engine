import hashlib
from functools import reduce

import cv2
import numpy as np

from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from censor_engine.typing import Image


class Plotter:
    def _label_to_bgr(self, label: str) -> tuple[int, int, int]:
        digest = hashlib.md5(str(label).encode()).digest()
        hue = int.from_bytes(digest[:4], "big") % 180
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        return tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]))  # type: ignore

    def _get_colour_set(
        self, model_classifiers: tuple[str, ...]
    ) -> dict[str, tuple[int, int, int]]:

        # Base HSV Colours Based on Hash
        base_hsv = {key: self._label_to_bgr(key) for key in model_classifiers}

        # Apply the Colours with Brightness Factor
        return {
            key: tuple(
                map(
                    int,
                    cv2.cvtColor(
                        np.array(
                            [
                                [
                                    (
                                        colour,
                                        saturation,
                                        128
                                        if key.endswith("_COVERED")
                                        else 255,
                                    )
                                ]
                            ],
                            dtype=np.uint8,
                        ),
                        cv2.COLOR_HSV2BGR,
                    )[0][0],
                )
            )  # type: ignore
            for key, (
                colour,
                saturation,
                _,
            ) in base_hsv.items()
        }

    # ======== #
    # BBoxes
    # ======== #
    def _draw_bbox(
        self,
        input_image: Image,
        colour_map: dict[str, tuple[int, int, int]] | None,
        detection: DetectorOutput,
    ):
        if detection.bbox is None:
            return input_image
        # Constants
        colour = (
            colour_map[detection.label]
            if detection.label is not None and colour_map is not None
            else (0, 0, 0)
        )
        threshold_for_light = 60
        is_light_bg = np.average(colour) > threshold_for_light
        text_colour = (0, 0, 0) if is_light_bg else (255, 255, 255)
        thickness = 2
        font_scale = 0.6
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Extract bbox coordinates
        x1, y1, x2, y2 = detection.bbox.xyxy

        # Draw bounding box
        cv2.rectangle(
            input_image,
            (x1, y1),
            (x2, y2),
            colour,
            thickness,
        )

        # Add label and confidence score
        text = f"{detection.label}: {detection.score:.2%}"

        # Handle Text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        cv2.rectangle(
            input_image,
            (x1, y1),
            (
                x1 + 4 + text_width,
                y1 - text_height - thickness * 2 - baseline - 4,
            ),
            colour,
            -1,
        )

        cv2.putText(
            input_image,
            text,
            (x1 + 2, max(y1 - 10, 20)),
            font,
            font_scale,
            text_colour,
            thickness - 1,
        )
        return input_image

    def plot_bbox(
        self,
        input_image: Image,
        output_data: list[DetectorOutput],
        model_classifiers: tuple[str, ...] | None = None,
    ) -> Image:
        labels = model_classifiers or tuple(
            out.label for out in output_data if out.label
        )
        colour_map = self._get_colour_set(labels) if labels else (0, 0, 0)

        return reduce(
            lambda image, detection: self._draw_bbox(
                image, colour_map, detection
            ),
            output_data,
            input_image,
        )

    # ======== #
    # Masks
    # ======== #
    def _draw_mask(
        self,
        input_image: Image,
        colour_map: dict[str, tuple[int, int, int]] | None,
        detection: DetectorOutput,
        alpha: float,
    ):
        if detection.mask is None:
            return input_image
        # Constants
        colour = (
            colour_map[detection.label]
            if detection.label is not None and colour_map is not None
            else (0, 0, 0)
        )
        threshold_for_light = 60
        is_light_bg = np.average(colour) > threshold_for_light
        text_colour = (0, 0, 0) if is_light_bg else (255, 255, 255)
        thickness = 2
        font_scale = 0.6
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Add label and confidence score
        text = f"{detection.label}: {detection.score:.2%}"

        # Handle Text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw Mask
        image_copy = input_image.copy()
        image_copy[detection.mask > 0] = colour
        input_image = cv2.addWeighted(
            input_image, 1 - alpha, image_copy, alpha, 0
        )  # type: ignore

        if detection.bbox is not None:
            # Extract bbox coordinates
            x1, y1, x2, y2 = detection.bbox.xyxy

            cv2.rectangle(
                input_image,
                (x1, y1),
                (
                    x1 + 4 + text_width,
                    y1 - text_height - thickness * 2 - baseline - 4,
                ),
                colour,
                -1,
            )

            cv2.putText(
                input_image,
                text,
                (x1 + 2, max(y1 - 10, 20)),
                font,
                font_scale,
                text_colour,
                thickness - 1,
            )
        return input_image

    def plot_mask(
        self,
        input_image: Image,
        output_data: list[DetectorOutput],
        model_classifiers: tuple[str, ...] | None = None,
        alpha: float = 0.5,
    ) -> Image:
        colour_map = (
            self._get_colour_set(model_classifiers)
            if model_classifiers
            else None
        )

        return reduce(
            lambda image, detection: self._draw_mask(
                image, colour_map, detection, alpha
            ),
            output_data,
            input_image,
        )

    def plot_depth_map(
        self,
        input_image: Image,
        output_data: list[DetectorOutput],
        alpha: float = 0.5,
    ) -> Image:
        if not output_data or output_data[0].depth_map is None:
            return input_image
        depth_image = cv2.normalize(
            output_data[0].depth_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)  # type: ignore
        depth_image = 255 - depth_image

        depth_colored = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)  # type: ignore
        return cv2.addWeighted(input_image, 1 - alpha, depth_colored, alpha, 0)  # type: ignore

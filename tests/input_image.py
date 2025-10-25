from pathlib import Path

import cv2
import numpy as np
from src.censor_engine.typing import Image

from censor_engine.models.lib_models.detectors import DetectedPartSchema

HORIZONTAL_ROWS = {
    "far-left": 100,
    "left": 150,
    "inner-left": 225,
    "centre": 250,
    "inner-right": 275,
    "right": 350,
    "far-right": 400,
}

VERTICAL_ROWS_AND_DATA = [
    # Single Item
    ("FACE_FEMALE", 30, "centre"),
    ("BELLY_EXPOSED", 260, "left"),
    ("BELLY_COVERED", 260, "right"),
    ("FEMALE_GENITALIA_EXPOSED", 440, "left"),
    ("FEMALE_GENITALIA_COVERED", 440, "right"),
    # Regular
    ("ARMPITS_EXPOSED", 60, "left"),
    ("ARMPITS_EXPOSED", 60, "right"),
    ("ARMPITS_COVERED", 90, "left"),
    ("ARMPITS_COVERED", 90, "right"),
    # Jointed Test
    ("FEMALE_BREAST_EXPOSED", 140, "left"),
    ("FEMALE_BREAST_EXPOSED", 160, "right"),
    ("FEMALE_BREAST_COVERED", 180, "inner-left"),
    ("FEMALE_BREAST_COVERED", 200, "inner-right"),
    # Inside Test
    ("BUTTOCKS_EXPOSED", 340, "left"),
    ("BUTTOCKS_COVERED", 340, "right"),
    ("ANUS_EXPOSED", 340, "left"),
    ("ANUS_COVERED", 340, "right"),
    # Pairs
    ("FEET_EXPOSED", 400, "far-left"),
    ("FEET_EXPOSED", 400, "left"),
    ("FEET_COVERED", 400, "right"),
    ("FEET_COVERED", 400, "far-right"),
    ("FACE_MALE", 260, "centre"),
    ("MALE_GENITALIA_EXPOSED", 400, "inner-right"),
    ("MALE_BREAST_EXPOSED", 400, "inner-left"),
]
COLOURS = {
    "FACE_FEMALE": (255, 255, 255),  # White
    "ARMPITS_EXPOSED": (255, 0, 0),  # Red
    "ARMPITS_COVERED": (128, 0, 0),
    "FEMALE_BREAST_EXPOSED": (0, 255, 0),  # Green
    "FEMALE_BREAST_COVERED": (0, 128, 0),
    "BELLY_EXPOSED": (0, 0, 255),  # Blue
    "BELLY_COVERED": (0, 0, 128),
    "BUTTOCKS_EXPOSED": (255, 255, 0),  # Yellow
    "BUTTOCKS_COVERED": (128, 128, 0),
    "ANUS_EXPOSED": (255, 0, 255),  # Purple
    "ANUS_COVERED": (128, 0, 128),
    "FEMALE_GENITALIA_EXPOSED": (0, 255, 255),  # Cyan
    "FEMALE_GENITALIA_COVERED": (0, 128, 128),
    "FEET_EXPOSED": (255, 128, 128),  # Skin Coloured
    "FEET_COVERED": (128, 64, 64),  # Wart-looking?
    "FACE_MALE": (128, 255, 128),  # Lime
    "MALE_GENITALIA_EXPOSED": (64, 64, 64),  # Dark Grey
    "MALE_BREAST_EXPOSED": (0, 0, 0),  # Black
}

CIRCLE_SIZE = 30

CUSTOM_CIRCLES = {
    "BELLY_EXPOSED": 60,
    "BELLY_COVERED": 60,
    "BUTTOCKS_EXPOSED": 80,
    "BUTTOCKS_COVERED": 80,
    "ANUS_EXPOSED": 20,
    "ANUS_COVERED": 20,
    "FEET_EXPOSED": 20,
    "FEET_COVERED": 20,
}


# Generators
class ImageGenerator:
    input_path: Path
    size: tuple[int, int] = (500, 500)
    seed: int = 69
    parts: list[DetectedPartSchema]

    def __init__(
        self,
        input_path: Path,
    ):
        self.input_path = input_path
        self._create_parts(self._create_parts())

    def _create_parts(
        self,
        input_data: list[tuple[str, int, str]] | None = None,
    ):
        if isinstance(input_data, list) and not len(input_data):
            return
        self.parts = []

        if input_data is None:
            input_data = VERTICAL_ROWS_AND_DATA

        for index, (key, v_row, h_row) in enumerate(input_data):
            circle_size = CUSTOM_CIRCLES.get(key, CIRCLE_SIZE)
            circle_radius = int(circle_size / 2)
            hori = HORIZONTAL_ROWS[h_row] if isinstance(h_row, str) else h_row
            relbox = (
                hori - circle_radius,
                v_row - circle_radius,
                circle_size,
                circle_size,
            )

            # Add Part
            self.parts.append(
                DetectedPartSchema(
                    label=key,
                    score=0.2,
                    relative_box=relbox,
                    part_id=index,
                ),
            )

    def _generate_background(self):
        height, width = self.size
        rng = np.random.default_rng(self.seed)
        noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        grey = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
        return cv2.merge([grey, grey, grey])  # type: ignore

    def _draw_circle(self, image: Image, part: DetectedPartSchema):
        x, y, w, h = part.relative_box
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(min(w, h) / 2)
        colour = COLOURS[part.label][::-1]

        cv2.circle(image, center, radius, colour, -1)  # type: ignore
        return image

    def make_test_image(self):
        image = self._generate_background()
        for part in self.parts:
            image = self._draw_circle(image, part)

        return image

    def return_detected_parts(
        self,
        list_parts_enabled: list[str] | str | None = None,
    ) -> list[DetectedPartSchema]:
        if not list_parts_enabled:
            return self.parts
        if isinstance(list_parts_enabled, str):
            list_parts_enabled = [list_parts_enabled]
        return [
            part for part in self.parts if part.label in list_parts_enabled
        ]


if __name__ == "__main__":
    cv2.imwrite("test_test_meta.jpg", ImageGenerator(Path()).make_test_image())

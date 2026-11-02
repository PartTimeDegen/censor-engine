from functools import reduce
from pathlib import Path

import cv2
import pytest
import torch

from censor_engine.libraries.detectors.human_parse import HumanParse
from censor_engine.libraries.detectors.yolo_seg import YoloSeg
from censor_engine.models.core.processors.image.detection._advanced_masks import (
    InferredDetectionParts,
)
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from tests.helpers.nsfw_images import get_images, read, write

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


# TODO: Inject masks


class TestInferredDetectionParts:
    class TestGenerateSkinPart:
        def test_working_mask(self):
            # Load Models
            human_parse = HumanParse(
                model_classifiers=("outfit", "hair", "phone")
            )
            human_parse.initiate_model()
            yolo = YoloSeg(model_classifiers=("person",))
            yolo.initiate_model()

            debug_folder = [
                image
                for image in get_images()
                # if str(image).endswith("27241943_127_939c.jpg")
            ]
            for image_path in debug_folder:
                # Input
                input_image = read(image_path)

                # Get Yolo Output
                yolo_output = yolo.predict(input_image)
                person_do = [
                    out
                    for out in yolo_output
                    if "YOLOSEG_PERSON" in out.label  # type: ignore
                ]

                # Get Human Parse Output
                hp_output = human_parse.predict(input_image)
                outfit_do = [
                    out
                    for out in hp_output
                    if "HUMANPARSE_OUTFIT" in out.label  # type: ignore
                    if out.score >= 0.40
                ]

                # Check Critical Masks are There
                if not person_do or not outfit_do:
                    continue

                # Flattens Outfit Masks
                flatten_outfit = DetectorOutput(
                    origin="HumanParse",
                    label="HUMANPARSE_OUTFIT",
                    score=1,
                    mask=reduce(
                        cv2.add, [outfit.mask for outfit in outfit_do]
                    ),
                )

                # Flatten Non-skin Detections, and Subtract from Person Mask
                mask_subtractions = reduce(
                    cv2.add,
                    [
                        out.mask
                        for out in hp_output
                        if out.label in ("HUMANPARSE_PHONE", "HUMANPARSE_HAIR")
                        and out.mask is not None
                    ],
                )

                output = [
                    InferredDetectionParts().generate_skin_part(
                        person,
                        flatten_outfit,
                        mask_subtractions=mask_subtractions,
                    )
                    for person in person_do
                ]

                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/advanced/skin_only/"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                output_image = human_parse.plotter.plot_mask(
                    input_image, output, ("INFERRED_SKIN",), alpha=1
                )
                write(str(fixed_image_path), output_image)

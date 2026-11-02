from pathlib import Path

import cv2
import pytest
import torch

from censor_engine.libraries.detectors.human_parse import HumanParse
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from tests.helpers.nsfw_images import get_images, read, write

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


@pytest.fixture(scope="class")
def human_parse() -> HumanParse:
    model = HumanParse()
    model.initiate_model()
    return model


class TestHumanParse:
    class TestInitiateModel:
        def test_working(self):
            model = HumanParse()
            model.initiate_model()

            assert model.model_name == "HumanParse"
            assert model._device == 0 if torch.cuda.is_available() else "cpu"

    class TestPredict:
        def test_working_bbox(self, human_parse: HumanParse):
            for image_path in get_images():
                input_image = read(image_path)
                output = human_parse.predict(input_image)
                output_image = human_parse.plotter.plot_bbox(
                    input_image,
                    output,
                    human_parse.model_classifiers,
                )
                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/human_parse/bbox"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

        def test_working_mask(self, human_parse: HumanParse):
            for image_path in get_images():
                input_image = read(image_path)
                output = human_parse.predict(input_image)
                output_image = human_parse.plotter.plot_mask(
                    input_image,
                    output,
                    human_parse.model_classifiers,
                )

                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/human_parse/mask"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

        def test_hair_and_outfit(self):
            human_parse = HumanParse()
            human_parse._model_classifiers = ("outfit", "hair")
            human_parse.initiate_model()
            for image_path in get_images():
                input_image = read(image_path)
                output = human_parse.predict(input_image)
                output_image = human_parse.plotter.plot_mask(
                    input_image,
                    output,
                    human_parse.model_classifiers,
                )

                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/human_parse/hair_and_outfit"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

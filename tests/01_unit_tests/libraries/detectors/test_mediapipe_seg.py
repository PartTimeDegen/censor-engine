from pathlib import Path

import cv2
import pytest
import torch

from censor_engine.libraries.detectors.mediapipe_seg import MediaPipeSeg
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from tests.helpers.controls import USE_ALL_MODELS
from tests.helpers.nsfw_images import get_images, read, write

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


@pytest.fixture(scope="class")
def model() -> MediaPipeSeg:
    if not USE_ALL_MODELS:
        return None
    model = MediaPipeSeg()
    model.initiate_model()
    return model


class TestMediaPipeSeg:
    class TestInitiateModel:
        def test_working(self):
            if not USE_ALL_MODELS:
                return
            model = MediaPipeSeg()
            model.initiate_model()

            assert model.model_name == "MediaPipeSeg"
            assert model._device == 0 if torch.cuda.is_available() else "cpu"

    class TestPredict:
        def test_working_bbox(self, model: MediaPipeSeg):
            if not USE_ALL_MODELS:
                return
            for image_path in get_images():
                input_image = read(image_path)
                output = model.predict(input_image)
                output_image = model.plotter.plot_bbox(
                    input_image,
                    output,
                    model.model_classifiers,
                )
                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/media_pipe_seg/bbox"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

        def test_working_mask(self, model: MediaPipeSeg):
            if not USE_ALL_MODELS:
                return
            for image_path in get_images():
                input_image = read(image_path)
                output = model.predict(input_image)
                output_image = model.plotter.plot_mask(
                    input_image,
                    output,
                    model.model_classifiers,
                )

                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/media_pipe_seg/mask"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

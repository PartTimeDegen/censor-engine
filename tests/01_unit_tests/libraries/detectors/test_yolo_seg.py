from pathlib import Path

import cv2
import pytest
import torch

from censor_engine.libraries.detectors.yolo_seg import YoloSeg
from tests.helpers.nsfw_images import get_images, read, write
from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


@pytest.fixture(scope="class")
def yolo_seg() -> YoloSeg:
    yolo = YoloSeg()
    yolo.initiate_model()
    return yolo


class TestYoloSeg:
    class TestInitiateModel:
        def test_working(self):
            yolo = YoloSeg()
            yolo.initiate_model()

            assert yolo.model_name == "YoloSeg"
            assert yolo._device == 0 if torch.cuda.is_available() else "cpu"

    class TestPredict:
        def test_working_bbox(self, yolo_seg: YoloSeg):
            for image_path in get_images():
                input_image = read(image_path)
                output = yolo_seg.predict(input_image)
                output_image = yolo_seg.plotter.plot_bbox(
                    input_image,
                    output,
                    yolo_seg.model_classifiers,
                )

                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/yolo_seg/bbox"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

        def test_working_mask(self, yolo_seg: YoloSeg):
            for image_path in get_images():
                input_image = read(image_path)
                output = yolo_seg.predict(input_image)
                output_image = yolo_seg.plotter.plot_mask(
                    input_image,
                    output,
                    yolo_seg.model_classifiers,
                )

                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/yolo_seg/mask"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

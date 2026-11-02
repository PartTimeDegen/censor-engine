from pathlib import Path

import cv2
import pytest
import torch

from censor_engine.libraries.detectors.fairface import FairFace
from tests.helpers.nsfw_images import get_images
from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


@pytest.fixture(scope="class")
def fairface() -> FairFace:
    fairface = FairFace()
    fairface.initiate_model()
    return fairface


class TestFairFace:
    class TestInitiateModel:
        def test_working(self):
            yolo = FairFace()
            yolo.initiate_model()

            assert yolo.model_name == "FairFace"
            assert yolo._device == 0 if torch.cuda.is_available() else "cpu"

    class TestPredict:
        def test_working_bbox(self, fairface: FairFace):
            return
            for image_path in get_images():
                input_image = cv2.imread(image_path)
                output = fairface.predict(input_image)
                print(image_path)
                print(output)
                print()
                # output_image = fairface.plotter.plot_bbox(
                #     input_image,
                #     output,
                #     fairface.model_classifiers,
                # )

                # fixed_image_path = easy_file_path / Path(
                #     "00_test_data/nsfw_output/fairface"
                # )
                # fixed_image_path.mkdir(parents=True, exist_ok=True)
                # fixed_image_path = (
                #     fixed_image_path / Path(image_path).parts[-1]
                # )
                # cv2.imwrite(str(fixed_image_path), output_image)

            raise

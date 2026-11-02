from pathlib import Path

import cv2
import pytest
import torch

from censor_engine.libraries.detectors.nude_net import NudeNet
from tests.helpers.nsfw_images import get_images, read, write
from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


@pytest.fixture(scope="class")
def nude_net() -> NudeNet:
    nn = NudeNet()
    nn.initiate_model()
    return nn


class TestNudeNet:
    class TestInitiateModel:
        def test_working(self):
            nn = NudeNet()
            nn.initiate_model()

            assert nn.model_name == "NudeNet"
            assert nn._device == 0 if torch.cuda.is_available() else "cpu"

    class TestPredict:
        def test_working(self, nude_net: NudeNet):
            for image_path in get_images():
                input_image = read(image_path)

                output = nude_net.predict(input_image)
                output_image = nude_net.plotter.plot_bbox(
                    input_image,
                    output,
                    nude_net.model_classifiers,
                )

                fixed_image_path = easy_file_path / Path(
                    "00_test_data/nsfw_output/models/nude_net"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

        def test_difference_in_model(self):
            skip = True
            for version in ["320n", "640m"]:
                if skip:
                    continue
                nude_net = NudeNet()
                nude_net.model_file = version
                nude_net.initiate_model()

                for index, image_path in enumerate(get_images()):
                    fixed_image_path = easy_file_path / Path(
                        "00_test_data/nsfw_output/models/nude_net_comparison"
                    )
                    file_name = f"{index}-{version}.jpg"
                    fixed_image_path_full = fixed_image_path / file_name

                    input_image = read(image_path)

                    output = nude_net.predict(input_image)
                    output_image = nude_net.plotter.plot_bbox(
                        input_image,
                        output,
                        nude_net.model_classifiers,
                    )

                    fixed_image_path.mkdir(parents=True, exist_ok=True)

                    write(str(fixed_image_path_full), output_image)

                    if version == "640m":
                        small = read(
                            str(fixed_image_path / f"{index}-320n.jpg")
                        )
                        big = read(str(fixed_image_path / f"{index}-640m.jpg"))
                        print(str(fixed_image_path / f"{index}-640m.jpg"))

                        write(
                            str(fixed_image_path / f"{index}-diff.jpg"),
                            cv2.absdiff(small, big),
                        )

from pathlib import Path

import cv2
import pytest

from censor_engine.models.core.processors.image.image_processor import (
    ImageProcessor,
)
from censor_engine.models.core.processors.image.model_manager import (
    ModelManager,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.typing import Image
from tests.fixtures.configs import CONFIG_DICTS, PROCESSOR_DICT
from tests.helpers.nsfw_images import get_images, read, write

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


class TestImageProcessor:
    class TestStartImageProcess:
        @pytest.mark.parametrize("config", CONFIG_DICTS.keys())
        def test_working(self, base_image: Image, config: dict):
            output = ImageProcessor().start_image_process(
                base_image,
                ModelManager(Config.model_validate(CONFIG_DICTS[config])),
            )

        @pytest.mark.parametrize("config", PROCESSOR_DICT.keys())
        def test_live(self, config: dict):
            ip = ImageProcessor()

            model_manager = ModelManager(
                Config.model_validate(PROCESSOR_DICT[config])
            )

            # folder_length = ["tests/00_nsfw_images/10475887_014_30bb.jpg"]
            for image_path in get_images():
                input_image = read(image_path)

                output = ip.start_image_process(input_image, model_manager)

                fixed_image_path = easy_file_path / Path(
                    f"00_test_data/nsfw_output/image_processor/{config}"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output)
                # print(str(fixed_image_path))
                # raise

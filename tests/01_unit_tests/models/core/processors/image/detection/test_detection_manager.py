from pathlib import Path

import cv2
import pytest

from censor_engine.models.core.processors.image.detection.detection_manager import (
    DetectionManager,
)
from censor_engine.models.core.processors.image.model_manager import (
    ModelManager,
)
from censor_engine.models.libraries.ai_models._helpers import Plotter
from censor_engine.models.libraries.configs.config import Config
from tests.fixtures.configs import CONFIG_DICTS
from tests.helpers.nsfw_images import get_images, read, write

file_path = Path(__file__)
file_path_parts = file_path.parts
test_folder_index = file_path_parts.index("tests")
easy_file_path = Path(*file_path_parts[: test_folder_index + 1])


class TestDetectionManager:
    class TestRunDetectionPipeline:
        @pytest.mark.parametrize("config_data", CONFIG_DICTS.keys())
        def test_working(self, config_data: str):
            plotter = Plotter()
            dm = DetectionManager()
            model_manager = ModelManager(
                Config.model_validate(CONFIG_DICTS[config_data])
            )
            for image_path in get_images():
                input_image = read(image_path)

                output = dm.run_detection_pipeline(input_image, model_manager)

                labels = tuple(out.label for out in output if out.label)
                if "segment" in config_data:
                    output_image = plotter.plot_mask(
                        input_image,
                        output,
                    )
                elif "depth" in config_data:
                    output_image = plotter.plot_depth_map(input_image, output)
                else:
                    output_image = input_image

                if "segment" not in config_data:
                    output_image = plotter.plot_bbox(
                        output_image, output, model_classifiers=labels
                    )
                fixed_image_path = easy_file_path / Path(
                    f"00_test_data/nsfw_output/detection_pipeline/{config_data}"
                )
                fixed_image_path.mkdir(parents=True, exist_ok=True)
                fixed_image_path = (
                    fixed_image_path / Path(image_path).parts[-1]
                )
                write(str(fixed_image_path), output_image)

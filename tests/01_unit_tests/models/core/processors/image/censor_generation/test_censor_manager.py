from pathlib import Path

from censor_engine._typing import Image
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.core.processors.image.censor_generation.censor_manager import (
    CensorManager,
)
from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)


class TestCensorManager:
    class TestReverseCensor: ...

    class TestNormalCensor: ...

    class TestRunCensorGenerationPipeline:
        def test_baseline(self, list_of_parts: list[Part], base_image: Image):
            output = CensorManager().run_censor_generation_pipeline(
                base_image, list_of_parts
            )

            handle_test_data(
                "censor_manager_pipeline",
                image=output,
                file_path=file_path,
            )

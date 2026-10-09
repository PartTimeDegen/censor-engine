from pathlib import Path

import cv2

from censor_engine._typing import Image
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.core.processors.image.censor_generation.censor_manager import (
    CensorManager,
)
from tests.helpers.test_data_handler import handle_test_data

file_path = Path(__file__)


class TestCensorManager:
    class TestApplyCensorsFromListOfParts:
        def test_working(
            self,
            list_of_parts_for_censor_manager: list[Part],
            base_image: Image,
        ):

            output = CensorManager()._apply_censors_from_list_of_parts(
                base_image, list_of_parts_for_censor_manager[0]
            )
            cv2.imwrite("output.jpg", output)

            handle_test_data(
                Path("censor_manager/apply_censor_from_list_of_parts"),
                image=output,
                file_path=file_path,
            )

    class TestReverseCensor:
        def test_working(
            self,
            list_of_parts_for_censor_manager: list[Part],
            base_image: Image,
        ):
            output = CensorManager()._handle_reverse_censor(
                base_image, list_of_parts_for_censor_manager
            )

            handle_test_data(
                Path("censor_manager/reverse_censor/working"),
                image=output,
                file_path=file_path,
            )

        def test_no_censor(
            self,
            list_of_parts_for_censor_manager: list[Part],
            base_image: Image,
        ):
            list_of_parts_for_censor_manager[
                0
            ].config.image.reverse_censor = []
            output = CensorManager()._handle_reverse_censor(
                base_image, list_of_parts_for_censor_manager
            )

            handle_test_data(
                Path("censor_manager/reverse_censor/no_censors"),
                image=output,
                file_path=file_path,
            )

    class TestNormalCensor:
        def test_working(
            self,
            list_of_parts_for_censor_manager: list[Part],
            base_image: Image,
        ):
            output = CensorManager()._handle_normal_censor(
                base_image, list_of_parts_for_censor_manager
            )

            handle_test_data(
                Path("censor_manager/normal_censor"),
                image=output,
                file_path=file_path,
            )

    class TestRunCensorGenerationPipeline:
        def test_working(
            self,
            list_of_parts_for_censor_manager: list[Part],
            base_image: Image,
        ):
            output = CensorManager().run_censor_generation_pipeline(
                base_image, list_of_parts_for_censor_manager
            )

            handle_test_data(
                Path("censor_manager/pipeline"),
                image=output,
                file_path=file_path,
            )

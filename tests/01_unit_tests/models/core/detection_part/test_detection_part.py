from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from censor_engine._typing import Image
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.detectors.schemas import DetectorOutput


@pytest.fixture
def part(
    detector_output: DetectorOutput,
    config_with_parts: Config,
    base_image: Image,
) -> Part:
    return Part(
        detector_output=detector_output,
        config=config_with_parts,
        file_uuid=uuid4(),
        image_shape=base_image.shape[0:2],
    )


class TestPart:
    def test_initiate(
        self,
        detector_output: DetectorOutput,
        config_with_parts: Config,
        base_image: Image,
    ):
        Part(
            detector_output=detector_output,
            config=config_with_parts,
            file_uuid=uuid4(),
            image_shape=base_image.shape[0:2],
        )

    # class TestProperties:
    #     def test_properties(self, part: Part):
    #         assert part.properties.settings ==
    #         assert part.properties.bbox ==

    #         assert part.properties.detector_origin ==
    #         assert part.properties.part_id ==
    #         assert part.properties.label ==
    #         assert part.properties.score ==
    #         assert part.properties.original_bbox ==
    #         assert part.properties.masks ==

    #         assert part.properties.is_merged ==
    #         assert part.properties.group_merge_id ==
    #         assert part.properties.group_merge ==

    #         assert part.properties.group_persist_id ==
    #         assert part.properties.group_persist ==

    #     def test_detector_data(self, part: Part): ...
    #     def test_config_data(self, part: Part): ...
    #     def test_group_data(self, part: Part): ...

    # class TestMaskManager:
    #     def test_properties(self, part: Part):
    #         assert part.masks.mask_name ==
    #         assert part.masks.protection_mask_name ==
    #         assert part.masks.mask_context ==

    #         assert part.masks.image_shape ==
    #         assert part.masks.current_mask ==
    #         assert part.masks.layers_of_mask ==

    #         assert part.masks._obj_mask ==
    #         assert part.masks._obj_mask_protected ==

    #     def test_create_empty_mask(self, part: Part): ...
    #     def test_add_to_current_mask(self, part: Part): ...
    #     def test_subtract_from_current_mask(self, part: Part): ...
    #     def test_compile_base_masks(self, part: Part): ...

from uuid import uuid4

import pytest

from censor_engine._typing import MaskImage
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.libraries.ai_models.output_schemas import (
    AbsoluteBBox,
    DetectorOutput,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.structs.censors import Censor
from tests.fixtures._dimensions import CANVAS_SIZE


@pytest.fixture
def part_base(
    config_with_parts: Config, detector_output: DetectorOutput
) -> Part:
    return Part(
        detector_output=detector_output,
        config=config_with_parts,
        file_uuid=uuid4(),
        image_shape=(CANVAS_SIZE, CANVAS_SIZE),
    )


@pytest.fixture
def list_of_parts(
    config_with_parts: Config,
    detector_output: DetectorOutput,
    mask_three_triangle_top: MaskImage,
    mask_three_triangle_bottom: MaskImage,
) -> list[Part]:
    file_uuid = uuid4()

    masks = [mask_three_triangle_top, mask_three_triangle_bottom]

    list_of_parts = []

    for mask in masks:
        part = Part(
            detector_output=detector_output,
            config=config_with_parts,
            file_uuid=file_uuid,
            image_shape=(CANVAS_SIZE, CANVAS_SIZE),
        )
        part.mask_manager.current_mask = mask
        part.properties.settings.censors = [Censor("Blur"), Censor("Outline")]
        part.config.image.reverse_censor = [Censor("Pixelate")]

        list_of_parts.append(part)

    return list_of_parts


@pytest.fixture
def list_of_parts_for_censor_manager(
    groups: list[list[str]],
    bbox: AbsoluteBBox,
    mask_three_triangle_top: MaskImage,
    mask_three_triangle_bottom: MaskImage,
) -> list[Part]:
    file_uuid = uuid4()

    detections = [
        DetectorOutput(
            bbox=bbox,
            # assumes MaskImage is bytes-like in tests
            masks=[],
            origin="test_origin",
            part_id=1,
            label=name,
            score=0.95,
        )
        for name in groups[0]
    ]

    config = Config.from_dict(
        {
            "detection": {
                "enabled_parts": groups[0],
                "default_settings": {"censors": ["Blur", "Outline"]},
            },
            "image": {"reverse_censor": ["Pixelate"]},
        }
    )

    masks = [mask_three_triangle_top, mask_three_triangle_bottom]

    list_of_parts = []

    for mask, detection in zip(masks, detections):
        part = Part(
            detector_output=detection,
            config=config,
            file_uuid=file_uuid,
            image_shape=(CANVAS_SIZE, CANVAS_SIZE),
        )
        part.mask_manager.current_mask = mask

        list_of_parts.append(part)

    return list_of_parts

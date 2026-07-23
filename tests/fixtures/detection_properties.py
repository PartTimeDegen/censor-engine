import pytest

from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.detectors.schemas import DetectorOutput

from .configs import config_part_minimum, config_with_parts


@pytest.fixture
def detection_properties(
    detector_output: DetectorOutput, config_part_minimum: Config
) -> PartProperties:
    return PartProperties(
        detector_output=detector_output, config=config_part_minimum
    )


@pytest.fixture
def detection_properties_extras(
    detector_output: DetectorOutput, config_with_parts: Config
) -> PartProperties:
    return PartProperties(
        detector_output=detector_output, config=config_with_parts
    )

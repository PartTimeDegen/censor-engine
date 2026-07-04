import numpy as np
import pytest

from censor_engine._typing import MaskImage
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.detectors.schemas import (
    AbsoluteBBox,
    DetectorOutput,
)

GROUPS = [
    ["boobs", "bum"],
    ["car", "vehicle"],
]


@pytest.fixture
def config_filled() -> Config:
    return Config.from_dict(
        {
            "groups": {"merging": GROUPS, "persistance": GROUPS},
            "detection": {
                "enabled_parts": [GROUPS[0][0]],
                "parts": {GROUPS[0][0]: {"margins": 0.2}},
            },
        }
    )


@pytest.fixture
def bbox() -> AbsoluteBBox:
    return AbsoluteBBox.from_xyxy((25, 25, 75, 75))


@pytest.fixture
def mask_empty() -> MaskImage:
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def mask_full() -> MaskImage:
    return np.ones((100, 100), dtype=np.uint8) * 255


@pytest.fixture
def detector_output(
    bbox: AbsoluteBBox, mask_empty: MaskImage, mask_full: MaskImage
) -> DetectorOutput:
    return DetectorOutput(
        bbox=bbox,
        # assumes MaskImage is bytes-like in tests
        masks=[mask_empty, mask_full],
        origin="test_origin",
        part_id=1,
        label=GROUPS[0][0],
        score=0.95,
    )


@pytest.fixture
def groups_merge() -> list[list[str]]:
    return GROUPS


@pytest.fixture
def groups_persistence() -> list[list[str]]:
    return GROUPS


@pytest.fixture
def detection_properties(detector_output: DetectorOutput) -> PartProperties:
    return PartProperties(
        detector_output=detector_output,
        config=Config.from_dict(
            {
                "detection": {
                    "enabled_parts": [GROUPS[0][0]],
                    "parts": {GROUPS[0][0]: {}},
                }
            }
        ),
    )


@pytest.fixture
def detection_properties_extras(
    detector_output: DetectorOutput, config_filled: Config
) -> PartProperties:
    return PartProperties(
        detector_output=detector_output, config=config_filled
    )


class TestPartProperties:
    def test_initiate(
        self, detector_output: DetectorOutput, config_filled: Config
    ):

        PartProperties(detector_output=detector_output, config=config_filled)

    class TestGeneratedFields:
        class TestCorrectedBox:
            def test_baseline(
                self,
                detection_properties: PartProperties,
                bbox: AbsoluteBBox,
            ):
                dp = detection_properties
                assert dp.bbox == bbox
                assert dp.bbox.center == bbox.center
                assert dp.bbox.area == bbox.area

            def test_margin(
                self,
                detection_properties_extras: PartProperties,
                bbox: AbsoluteBBox,
            ):
                dp = detection_properties_extras

                expected_change = round(bbox.height * 0.2) // 2
                lower = bbox.x1 - expected_change
                upper = bbox.x2 + expected_change
                expected_bbox = AbsoluteBBox(
                    x1=lower, y1=lower, x2=upper, y2=upper
                )

                assert dp.bbox == expected_bbox
                assert dp.bbox.center == expected_bbox.center
                assert dp.bbox.area == expected_bbox.area

        class TestIsMerged:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.is_merged == False

            def test_no_merge(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.is_merged == True

        class TestGroupMergeID:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_merge_id == None

            def test_no_merge_group(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.group_merge_id == 1

        class TestGroupMerge:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_merge == []

            def test_no_merge_group(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.group_merge == GROUPS[0]

        class TestGroupPersistID:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_persist_id == None

            def test_no_persist_group(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.group_persist_id == 1

        class TestGroupPersist:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_persist == []

            def test_no_persist_group(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.group_persist == GROUPS[0]
